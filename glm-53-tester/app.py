"""FastAPI 后端入口
提供：
  - 静态资源（Web UI: 配置页 + 实时执行面板 + 报告详情）
  - /api/config    GET/PUT 能力声明与端点配置
  - /api/run       POST 启动一次完整自测（SSE流式推送进度）
  - /api/reports   GET 历史报告列表 / GET {id} 查看详情
  - /api/download/{kind}/{file}  下载 HTML/JSON/Excel
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 确保包导入
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import TesterConfig, get_config, save_config, OUTPUT_DIR, BASE_DIR as CFG_BASE_DIR  # noqa: E402
from tests.base import TestResult, TestStatus, run_test_suite  # noqa: E402
from tests.suite import ALL_FUNCTIONAL_CASES  # noqa: E402
from benchmark import BenchmarkResult, run_benchmark  # noqa: E402
from report import export_all  # noqa: E402


app = FastAPI(title="GLM-5.3 模型验收测试工具", version="1.0.0")
WEB_DIR = BASE_DIR / "web"


# -------------------- 诊断日志 --------------------
import logging as _logging
import time as _t
from fastapi import Request as _FA_Request
from fastapi.responses import Response as _FA_Response

_diag_logger = _logging.getLogger("k3_diag")
_diag_logger.setLevel(_logging.INFO)
if not _diag_logger.handlers:
    _h = _logging.StreamHandler(sys.stderr)
    _h.setFormatter(_logging.Formatter("[K3-DIAG %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    _diag_logger.addHandler(_h)


# -------------------- 启动预检 --------------------
@app.on_event("startup")
async def _preconfigure_resources():
    """启动自检：确认用例可加载、依赖齐全。"""
    _diag_logger.info("[STARTUP] GLM-5.3 验收测试工具启动，用例数=%d", len(ALL_FUNCTIONAL_CASES))


@app.middleware("http")
async def _diagnostic_middleware(request: _FA_Request, call_next):
    req_id = uuid.uuid4().hex[:8]
    t0 = _t.perf_counter()
    method, path = request.method, request.url.path
    _diag_logger.info(">> %s %s  (req=%s client=%s)",
                      method, path, req_id, request.client.host if request.client else "?")
    try:
        response: _FA_Response = await call_next(request)
    except Exception:
        dt_ms = int((_t.perf_counter() - t0) * 1000)
        _diag_logger.error("!! %s %s  EXCEPTION after %dms  req=%s\n%s",
                           method, path, dt_ms, req_id, traceback.format_exc())
        # 注意：raise 必须在 except 块内。初版把它写在了 try/except 之外的
        # 函数层级，此时异常已脱离处理上下文，裸 raise 会抛
        # RuntimeError: No active exception to re-raise，把真实异常掩盖掉。
        raise
    dt_ms = int((_t.perf_counter() - t0) * 1000)
    _diag_logger.info("<< %s %s  status=%s  dt=%dms  req=%s",
                      method, path, response.status_code, dt_ms, req_id)
    response.headers["X-K3-Request-Id"] = req_id
    response.headers["X-K3-Duration-Ms"] = str(dt_ms)
    return response


def _diag_event(sid: str, ev_type: str, detail: str = ""):
    _diag_logger.info("[EVENT sid=%s] %s %s", sid[-6:], ev_type, detail)


# -------------------- 全局运行状态 --------------------

@dataclass
class RunSession:
    id: str
    started_at: float
    finished: bool = False
    cancelled: bool = False
    progress_events: List[Dict[str, Any]] = field(default_factory=list)
    report_files: Optional[Dict[str, Any]] = None
    task: Optional[Any] = None


# 内存里保留的历史会话上限。初版只增不删，长期运行内存持续增长
# （每轮自测的 progress_events 可达数千条）。
MAX_KEPT_SESSIONS = 20


class _Store:
    def __init__(self):
        self.sessions: Dict[str, RunSession] = {}
        self._lock = threading.RLock()

    def new_session(self) -> RunSession:
        s = RunSession(id=uuid.uuid4().hex[:12], started_at=time.time())
        with self._lock:
            self.sessions[s.id] = s
            self._evict_locked()
        return s

    def _evict_locked(self):
        finished = sorted(
            (x for x in self.sessions.values() if x.finished),
            key=lambda x: x.started_at,
        )
        while len(self.sessions) > MAX_KEPT_SESSIONS and finished:
            drop = finished.pop(0)
            self.sessions.pop(drop.id, None)

    def attach_task(self, sid: str, task: Any):
        with self._lock:
            s = self.sessions.get(sid)
            if s:
                s.task = task

    def running_session(self) -> Optional[str]:
        with self._lock:
            for s in self.sessions.values():
                if not s.finished and not s.cancelled:
                    return s.id
        return None

    def push(self, sid: str, event: Dict[str, Any]):
        with self._lock:
            s = self.sessions.get(sid)
            if s:
                event.setdefault("ts", time.time())
                s.progress_events.append(event)
                # 诊断日志：打印关键事件到stderr（phase/report_done/error/functional.item失败类）
                try:
                    t = event.get("type", "?")
                    if t in ("phase", "phase_done", "phase_skip", "report_gen", "report_done", "error"):
                        detail = event.get("msg") or event.get("final_verdict") or ""
                        _diag_event(sid, t, str(detail)[:200])
                    elif t == "functional.item":
                        data = event.get("data") or {}
                        st = data.get("status", "?")
                        if st in ("fail", "error", "waive"):
                            _diag_event(sid, f"case-{st}",
                                        f"{data.get('id')} {data.get('name')}  fail={data.get('failures',[])[:1]}")
                    elif t == "benchmark.item":
                        if event.get("done", 0) % 50 == 0:
                            _diag_event(sid, "bench",
                                        f"turns {event.get('done')}/{event.get('total')}")
                except Exception:
                    pass

    def finish(self, sid: str, report_files: Optional[Dict[str, Any]]):
        with self._lock:
            s = self.sessions.get(sid)
            if s:
                s.finished = True
                s.report_files = report_files


STORE = _Store()


# -------------------- API Models --------------------

class ConfigUpdate(BaseModel):
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    capability: Optional[Dict[str, Any]] = None
    benchmark: Optional[Dict[str, Any]] = None
    run_functional: Optional[bool] = None
    run_benchmark: Optional[bool] = None
    long_context_test_tokens: Optional[int] = None


class RunRequest(BaseModel):
    # 允许在运行时临时覆盖配置（不持久化）
    config_override: Optional[Dict[str, Any]] = None


# -------------------- 静态页面 --------------------

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h2>Web UI 文件缺失，请检查 web/index.html</h2>")


if WEB_DIR.exists():
    # 挂载 /css /js
    for sub in ("css", "js"):
        p = WEB_DIR / sub
        if p.exists():
            app.mount(f"/{sub}", StaticFiles(directory=str(p)), name=sub)


# -------------------- Config API --------------------

@app.get("/api/config")
def get_config_api():
    cfg = get_config()
    data = cfg.redacted()
    return data


@app.put("/api/config")
def update_config_api(payload: ConfigUpdate):
    d: Dict[str, Any] = {}
    for k in ("api_base", "api_key", "model_id", "timeout_seconds", "max_retries",
              "capability", "benchmark", "run_functional", "run_benchmark",
              "long_context_test_tokens"):
        v = getattr(payload, k, None)
        if v is not None:
            d[k] = v
    saved = save_config(d)
    return {"ok": True, "config": saved.redacted()}


# -------------------- Run API (SSE Progress) --------------------

def _apply_override(cfg: TesterConfig, override: Optional[Dict[str, Any]]) -> TesterConfig:
    if not override:
        return cfg
    data = cfg.model_dump()
    # 简单合并一层
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    return TesterConfig(**data)


def _result_to_jsonable(r: TestResult) -> Dict[str, Any]:
    # 精简版：用于前端实时展示
    return {
        "id": r.meta.id,
        "category": r.meta.category,
        "name": r.meta.name,
        "required": r.meta.required,
        "doc_ref": r.meta.doc_ref,
        "status": r.status,
        "http_status": r.http_status,
        "duration_ms": r.duration_ms,
        "pass_rate": round(r.pass_rate * 100, 1),
        "assertions": f"{r.passed_assertions}/{r.total_assertions}" if r.total_assertions else "—",
        "error_kind": r.error_kind,
        "retried": r.retried,
        "failures": r.failures[:5],
        "notes": r.notes[:5],
    }


async def _run_worker(sid: str, cfg: TesterConfig):
    """主执行流：连通性探测 → 功能 → 性能 → 报告导出"""
    func_results: List[TestResult] = []
    bench: Optional[BenchmarkResult] = None

    def _cancelled() -> bool:
        s = STORE.sessions.get(sid)
        return bool(s and s.cancelled)

    # 压测跳过原因（用户取消 / 配置关闭），用于报告如实留痕
    bench_skip_reason: Optional[str] = None

    try:
        # 不通就直接中止，不浪费时间跑全量用例全部失败
        STORE.push(sid, {"type": "phase", "phase": "connectivity",
                         "msg": "正在探测 API 连通性..."})
        import httpx
        from tests.base import TestResult as _TR, TestCaseMeta as _TCM, TestStatus as _TS
        probe = _TR(meta=_TCM(id="G0", category="连通性", name="API连通性探测",
                              required=True, check_points=["HTTP 200 且响应格式正确"]))
        probe_payload = {
            "model": cfg.model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
            "max_tokens": 32,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0, pool=10.0)) as probe_client:
                headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
                resp = await probe_client.post(cfg.endpoint, headers=headers, json=probe_payload)
                probe.http_status = resp.status_code
                try:
                    probe.response_body = resp.json()
                except Exception:
                    probe.response_body = resp.text[:200]
        except Exception as e:
            probe.http_status = 0
            probe.response_body = {"error": str(e)}

        if probe.http_status == 200 and isinstance(probe.response_body, dict) and "choices" in probe.response_body:
            STORE.push(sid, {"type": "phase_done", "phase": "connectivity",
                             "msg": "✅ 连通性探测通过（HTTP 200）"})
        else:
            err_detail = ""
            if isinstance(probe.response_body, dict):
                err = probe.response_body.get("error")
                if isinstance(err, dict):
                    err_detail = err.get("message", "") or err.get("type", "")
                elif isinstance(err, str):
                    err_detail = err
            msg = (f"❌ 连通性探测失败：HTTP {probe.http_status}"
                   + (f"，{err_detail}" if err_detail else ""))
            STORE.push(sid, {"type": "phase_done", "phase": "connectivity", "msg": msg})
            STORE.push(sid, {"type": "error",
                             "msg": f"{msg}。后续功能/性能测试已中止，请检查 API Key、API Base、模型ID 是否正确。"})
            # 生成一个只含连通性失败的报告
            probe.status = _TS.FAIL
            probe.add_failure(msg)
            func_results = [probe]
            STORE.push(sid, {"type": "report_gen", "msg": "正在生成报告（仅连通性探测结果）..."})
            files = export_all(cfg, func_results, None, None)
            paths_only = {k: str(v) for k, v in files.items() if k not in ("_report_object", "_errors")}
            STORE.push(sid, {"type": "report_done", "paths": paths_only,
                             "final_verdict": "连通性失败，测试中止",
                             "reasons": [msg]})
            STORE.finish(sid, paths_only)
            return

        # ---------- 功能测试 ----------
        if cfg.run_functional:
            STORE.push(sid, {"type": "phase", "phase": "functional",
                             "msg": f"开始执行功能测试 {len(ALL_FUNCTIONAL_CASES)} 项用例"})

            async def _progress(done: int, total: int, r: TestResult):
                STORE.push(sid, {
                    "type": "functional.item",
                    "done": done, "total": total,
                    "data": _result_to_jsonable(r),
                })

            func_results = await run_test_suite(
                ALL_FUNCTIONAL_CASES, cfg, progress_cb=_progress, should_cancel=_cancelled,
            )
            STORE.push(sid, {
                "type": "phase_done", "phase": "functional",
                "msg": f"功能测试完成：{sum(1 for r in func_results if r.status==TestStatus.PASS)}"
                       f"/{len(func_results)} 通过",
            })
        else:
            STORE.push(sid, {"type": "phase_skip", "phase": "functional", "msg": "跳过功能测试（配置关闭）"})

        # ---------- 性能压测 ----------
        # 用户取消后不再进入压测阶段；但要在提示与报告中写明原因，
        # 否则用户会以为"勾选了压测却没跑"是配置丢失（实际是取消导致）。
        if cfg.run_benchmark and not _cancelled():
            STORE.push(sid, {"type": "phase", "phase": "benchmark",
                             "msg": f"开始性能压测：会话{cfg.benchmark.total_sessions}，"
                                    f"爬坡{cfg.benchmark.ramp_duration_seconds}s，"
                                    f"稳态{cfg.benchmark.steady_duration_seconds}s"})

            def _bench_cb(kind, done, total, extra):
                STORE.push(sid, {"type": "benchmark.item", "done": done,
                                 "total": total, "extra": extra})

            bench = await run_benchmark(cfg, progress_cb=_bench_cb, should_cancel=_cancelled)
            # turn 级原始指标立即落盘：初版只在内存里持有，报告导出一崩溃
            # （例如上一轮的 NameError），压测明细就全丢了，只能从日志捡汇总数字。
            try:
                from dataclasses import asdict as _ad
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                _bench_dump = OUTPUT_DIR / f"benchmark-turns-{sid[-8:]}.json"
                _bench_dump.write_text(json.dumps({
                    "sid": sid,
                    "dumped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": {k: v for k, v in bench.to_dict().items() if k != "turns"},
                    "turns": [_ad(t) for t in bench.turns],
                }, ensure_ascii=False, default=str), encoding="utf-8")
                _diag_logger.info("[EVENT sid=%s] 压测明细已落盘 %s（%d turns）",
                                  sid[-6:], _bench_dump.name, len(bench.turns))
            except Exception as dump_err:
                _diag_logger.error("压测明细落盘失败（不影响主流程）: %s", dump_err)
            cache_txt = (f"{round((bench.cache_hit_rate_overall or 0)*100,2)}%"
                         if bench.cache_measurable else "无法测量")
            STORE.push(sid, {
                "type": "phase_done", "phase": "benchmark",
                "msg": f"性能压测完成：稳态吞吐 {bench.steady_throughput_req_per_s} req/s，"
                       f"TTFT_P50 {bench.ttft_ms.get('p50',0)}ms，缓存命中 {cache_txt}",
            })
        else:
            if cfg.run_benchmark:
                # 用户勾了压测但中途取消：压测不会再执行，必须明示原因
                bench_skip_reason = "运行被取消，压测未执行"
                STORE.push(sid, {"type": "phase_skip", "phase": "benchmark",
                                 "msg": "已取消，跳过性能压测（勾选仍在，重新完整运行即可执行）"})
            else:
                STORE.push(sid, {"type": "phase_skip", "phase": "benchmark", "msg": "跳过性能压测（配置关闭）"})

        # ---------- 报告导出 ----------
        STORE.push(sid, {"type": "report_gen", "msg": "正在生成报告（HTML/JSON/Excel/PDF）..."})
        files = export_all(cfg, func_results, bench, None, bench_skip_reason=bench_skip_reason)
        paths_only = {k: str(v) for k, v in files.items()
                      if k not in ("_report_object", "_errors")}
        errors = files.get("_errors") or {}
        if errors:
            STORE.push(sid, {"type": "report_partial",
                             "msg": f"部分格式导出失败: {errors}（其余格式已生成）",
                             "errors": errors})
        STORE.push(sid, {
            "type": "report_done",
            "paths": paths_only,
            "errors": errors,
            "final_verdict": files["_report_object"].acceptance_summary["最终结论"],
        })
        STORE.finish(sid, paths_only)
    except Exception as e:
        tb = traceback.format_exc()
        STORE.push(sid, {"type": "error", "msg": f"执行异常: {type(e).__name__}: {e}", "trace": tb})
        STORE.finish(sid, None)


@app.post("/api/run")
async def run_self_test(req: RunRequest):
    cfg = _apply_override(get_config(), req.config_override)
    # 基础校验
    if not cfg.api_key or not cfg.model_id:
        raise HTTPException(status_code=400, detail="请先配置API Key和Model ID")
    if not cfg.api_base:
        raise HTTPException(status_code=400, detail="请先配置 API Base")

    # 并发互斥：同一时刻只允许一轮自测。初版没有这道闸，连点两次会同时
    # 对同一个供应商发压，两轮的压测指标互相污染。
    running = STORE.running_session()
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"已有一轮测试正在执行（sid={running}）。"
                   f"请等待其结束，或调用 POST /api/cancel/{running} 取消。",
        )

    session = STORE.new_session()

    task = asyncio.create_task(_run_worker(session.id, cfg))
    STORE.attach_task(session.id, task)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'session_start', 'sid': session.id}, ensure_ascii=False)}\n\n"
        last_sent = 0
        while True:
            s = STORE.sessions.get(session.id)
            if s is None:
                break
            events = s.progress_events
            for ev in events[last_sent:]:
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            last_sent = len(events)
            if s.finished:
                for ev in s.progress_events[last_sent:]:
                    yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
                yield f"data: {json.dumps({'type': 'session_end', 'sid': session.id}, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/cancel/{sid}")
def cancel_run(sid: str):
    """请求取消一轮自测。

    RunSession.cancelled 字段初版就定义了但从来没有接口去设置它。
    这里做协作式取消：各阶段在安全点检查该标志后退出，已完成的结果保留。
    """
    s = STORE.sessions.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    if s.finished:
        return {"ok": False, "msg": "该会话已结束"}
    s.cancelled = True
    STORE.push(sid, {"type": "phase", "phase": "cancel",
                     "msg": "已收到取消请求，将在当前用例/请求结束后停止"})
    return {"ok": True, "sid": sid}


@app.get("/api/session/{sid}")
def get_session(sid: str):
    s = STORE.sessions.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return {
        "id": s.id,
        "started_at": s.started_at,
        "finished": s.finished,
        "cancelled": s.cancelled,
        "events_count": len(s.progress_events),
        "report_files": s.report_files,
    }


# -------------------- Reports --------------------

def _safe_report_path(filename: str) -> Path:
    # 防穿越
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "非法文件名")
    p = OUTPUT_DIR / filename
    if not p.exists():
        raise HTTPException(404, "文件不存在")
    if p.resolve().parent != OUTPUT_DIR.resolve():
        raise HTTPException(400, "非法路径")
    return p


@app.get("/api/reports")
def list_reports():
    files = list(OUTPUT_DIR.glob("glm-53-report-*"))
    groups: Dict[str, Dict[str, str]] = {}
    for f in sorted(files, reverse=True):
        stem = f.stem
        # kimi-k3-report-20260903_120000.ext
        parts = stem.split(".", 1)
        key = parts[0]
        groups.setdefault(key, {})[f.suffix.lstrip(".")] = f.name
    out = []
    for key, m in groups.items():
        out.append({"key": key, "files": m, "has_html": "html" in m})
    return out


@app.get("/api/download/{kind}/{filename}")
def download_report(kind: str, filename: str):
    p = _safe_report_path(filename)
    if kind not in ("html", "json", "xlsx", "pdf"):
        raise HTTPException(400, "kind 必须是 html/json/xlsx/pdf")
    media = {
        "html": "text/html; charset=utf-8",
        "json": "application/json",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf": "application/pdf",
    }[kind]
    return FileResponse(str(p), media_type=media, filename=p.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", host="127.0.0.1", port=8788,
        reload=False, log_level="info",
    )
