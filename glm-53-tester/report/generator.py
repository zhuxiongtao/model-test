"""GLM-5.3 模型验收测试报告生成器

定位：独立的模型验收测试（非供应商自测评），对照 GLM-5.3 官方公开的
参数范围与能力声明逐项验证，只陈述事实、不下整改指令。

输出：
  1. HTML报告（可视化，适合直接审阅 / 打印PDF）
  2. JSON报告（机器可读，方便接口回传）
  3. Excel报告（便于邮件提交/归档）
  4. PDF报告（便于归档）
  5. artifacts/ 目录：每个用例的完整请求体/响应体/SSE分片

结论呈现方式（按用户要求）：
  - 首屏给通俗结论：一句话陈述通过/未通过数量，紧接
    「有问题的（N项）」「没问题的（N项）」「性能实测」三块事实清单
  - 未执行的测试项（如 KVV、未开启的压测）整节不出现
  - 后文再展开详细数据（用例明细、压测分布、基线对比）
"""
from __future__ import annotations

import html
import json
import statistics
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

from config import MIN_REQUIRED_CAPABILITIES, OUTPUT_DIR, TesterConfig
from tests.base import TestResult, TestStatus
from benchmark import BenchmarkResult

# GLM-5.3 暂不接入 KVV 精度测试，保留类型占位
KVVResult = Any

# -------------------- 官方能力对账（GLM-5.3） --------------------
# 对照 https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3 官方文档逐条映射到本工具用例。
COVERAGE_NOTE = (
    "本工具用例对照 GLM-5.3 官方文档《模型能力说明》逐条映射：协议完整性按 OpenAI "
    "ChatCompletions 规范校验；思考行为/采样参数/上下文/工具调用/结构化输出按官方"
    "公开的参数范围与能力声明设计；另含基础能力（代码/数学/多轮）抽查与性能压测"
    "（TTFT、吞吐、缓存命中率）。"
)

DOC_COVERAGE_MAP: List[Dict[str, Any]] = [
    {
        "doc_category": "协议完整性",
        "doc_subitems": "OpenAI ChatCompletions 兼容：流式分片、finish_reason、usage 回传（非流式/流式）",
        "doc_count": 4,
        "doc_required": "全部必过",
        "case_ids": ["G1", "G2", "G3", "G4"],
        "note": "",
    },
    {
        "doc_category": "思考行为",
        "doc_subitems": "始终思考（默认开启）；不支持 disabled；reasoning_effort 支持 low/high/max 三档",
        "doc_count": 3,
        "doc_required": "全部必过",
        "case_ids": ["G5", "G6", "G7"],
        "note": "G6 期望关闭思考被拒绝（HTTP 400）",
    },
    {
        "doc_category": "采样参数",
        "doc_subitems": "temperature (0.0,1.0]；top_p [0.01,1.0]；do_sample=false 确定性输出",
        "doc_count": 3,
        "doc_required": "全部必过",
        "case_ids": ["G8", "G9", "G10"],
        "note": "",
    },
    {
        "doc_category": "上下文能力",
        "doc_subitems": "长上下文召回；最大输出 128K tokens",
        "doc_count": 2,
        "doc_required": "全部必过",
        "case_ids": ["G11", "G12"],
        "note": "",
    },
    {
        "doc_category": "工具调用",
        "doc_subitems": "Function Calling；流式工具调用；tool_choice 多分支（auto/none/required）",
        "doc_count": 3,
        "doc_required": "全部必过",
        "case_ids": ["G13", "G14", "G15"],
        "note": "",
    },
    {
        "doc_category": "结构化输出",
        "doc_subitems": "response_format：JSONObject / JSONSchema 严格模式",
        "doc_count": 2,
        "doc_required": "全部必过",
        "case_ids": ["G16", "G17"],
        "note": "",
    },
    {
        "doc_category": "基础能力",
        "doc_subitems": "代码生成可运行；数学推理；多轮上下文保持",
        "doc_count": 3,
        "doc_required": "抽查项",
        "case_ids": ["G18", "G19", "G20"],
        "note": "",
    },
]


STATUS_LABEL = {
    TestStatus.PASS: ("通过", "#10b981"),
    TestStatus.FAIL: ("失败", "#ef4444"),
    TestStatus.WAIVE: ("豁免", "#f59e0b"),
    TestStatus.ERROR: ("异常", "#7c3aed"),
    TestStatus.SKIP: ("未测", "#94a3b8"),
}

# 能力声明标签：必须与 config.CapabilityDeclare 的字段一一对应。
# 初版从 Kimi-K3 版复制，残留了多模态/ToolChoice 等键——GLM 的能力模型里
# 根本没有这些字段，报告里全部渲染成"不支持"，用户实际从未声明过。
CAP_LABELS = {
    "support_thinking": "思考/推理链输出",
    "thinking_default_enabled": "默认思考行为=开启",
    "support_reasoning_effort": "reasoning_effort 推理强度（low/high/max）",
    "support_streaming": "流式输出",
    "support_streaming_tool": "流式工具调用（tool_stream）",
    "support_function_calling": "Function Calling",
    "support_tool_choice": "tool_choice（auto/none/required）",
    "support_structured_output": "结构化输出（JSON / JSONSchema）",
    "support_json_schema_strict": "JSONSchema 严格模式",
    "support_do_sample": "do_sample 确定性输出",
    "support_context_cache": "上下文缓存",
    "support_long_context_1m": "长上下文（1M）",
    "support_max_output_128k": "最大输出 128K",
}


@dataclass
class FullReport:
    timestamp: str
    run_id: str
    environment: Dict[str, Any]
    capability_declare: Dict[str, Any]
    capability_warnings: List[Dict[str, str]]
    doc_coverage: List[Dict[str, Any]]
    coverage_note: str
    functional_results: List[Dict[str, Any]]
    benchmark: Optional[Dict[str, Any]]
    kvv: Optional[Dict[str, Any]]
    acceptance_summary: Dict[str, Any]
    artifacts_dir: str = ""


# -------------------- artifacts 落盘 --------------------

def _write_artifacts(run_id: str, func_results: List[TestResult],
                     out_dir: Path) -> Tuple[Path, Dict[str, str]]:
    """把每个用例的完整请求/响应/SSE分片写到独立文件。

    文档 2.1 要求每个用例都回传完整请求体与响应体（或流式分片）。报告正文里
    塞不下（单个多模态请求的 base64 就有几十KB），因此正文放摘要、全文落盘。
    """
    art_dir = out_dir / "artifacts" / run_id
    art_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, str] = {}
    for r in func_results:
        payload = {
            "case_id": r.meta.id,
            "case_name": r.meta.name,
            "doc_ref": r.meta.doc_ref,
            "status": r.status,
            "http_status": r.http_status,
            "duration_ms": r.duration_ms,
            "error_kind": r.error_kind,
            "failures": r.failures,
            "notes": r.notes,
            "usage": r.usage,
            "records": r.records,
            "request_body": r.request_body,
            "response_body": r.response_body,
            "stream_chunks": [
                {
                    "index": c.index, "elapsed_ms": c.elapsed_ms, "raw": c.raw,
                    "delta_content": c.delta_content,
                    "reasoning_delta": c.reasoning_delta,
                }
                for c in r.stream_chunks
            ],
            "sub_requests": [
                {
                    "label": s.label,
                    "http_status": s.http_status,
                    "duration_ms": s.duration_ms,
                    "attempts": s.attempts,
                    "error_kind": s.error_kind,
                    "error_detail": s.error_detail,
                    "usage": s.usage,
                    "request_body": s.request_body,
                    "response_body": s.response_body,
                    "stream_chunks": s.stream_chunks,
                }
                for s in r.sub_requests
            ],
        }
        fname = f"{r.meta.id}.json"
        (art_dir / fname).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        index[r.meta.id] = fname
    return art_dir, index


def _truncate(v: Any, limit: int = 4000) -> str:
    if v is None:
        return ""
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + f"\n…(已截断，全文见 artifacts，共 {len(s)} 字符)"


# -------------------- 报告组装 --------------------

def build_full_report(
    cfg: TesterConfig,
    func_results: List[TestResult],
    bench: Optional[BenchmarkResult],
    kvv: Optional[KVVResult],
    out_dir: Optional[Path] = None,
    bench_skip_reason: Optional[str] = None,
) -> FullReport:
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or OUTPUT_DIR
    env = {
        "model_id": cfg.model_id,
        "api_endpoint": cfg.endpoint,
        "api_base": cfg.api_base,
        "test_date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "tester_version": "1.1.0",
        "max_retries": cfg.max_retries,
    }
    capability = cfg.capability.model_dump()

    art_dir, art_index = _write_artifacts(run_id, func_results, out_dir)

    # ---------- 能力声明 vs 文档最低要求 ----------
    capability_warnings: List[Dict[str, str]] = []
    for key, reason in MIN_REQUIRED_CAPABILITIES.items():
        if not capability.get(key, False):
            capability_warnings.append({
                "capability": CAP_LABELS.get(key, key),
                "key": key,
                "reason": reason,
                "impact": "对应用例被豁免，但该项属于文档规定的最低能力要求，"
                          "审核方需据此判断是否接受。",
            })

    # ---------- 功能用例展开 ----------
    functional_rows: List[Dict[str, Any]] = []
    for r in func_results:
        base_row = {
            "category": r.meta.category,
            "required": r.meta.required,
            "doc_ref": r.meta.doc_ref,
            "check_points": r.meta.check_points,
            "status": r.status,
            "pass_rate": r.pass_rate,
            "total_assertions": r.total_assertions,
            "passed_assertions": r.passed_assertions,
            "failures": r.failures,
            "notes": r.notes,
            "error_kind": r.error_kind,
            "error_trace": r.error_trace,
            "retried": r.retried,
            "artifact": art_index.get(r.meta.id, ""),
        }
        # reasoning_effort 展开为三档子行（对齐文档"3项"统计口径）
        if r.meta.id == "F22" and r.records:
            for rec in r.records:
                sub_req = next(
                    (s for s in r.sub_requests if s.label == f"effort={rec['level']}"), None
                )
                functional_rows.append({
                    **base_row,
                    "id": f"F22-{rec['level']}",
                    "name": f"reasoning_effort: {rec['level']} 档",
                    "http_status": rec.get("http_status"),
                    # 初版这里恒为 0：sub 级 TestResult 的 duration_ms 从未被赋值
                    "duration_ms": rec.get("duration_ms", 0),
                    "usage": {
                        "prompt_tokens": rec.get("prompt_tokens"),
                        "completion_tokens": rec.get("completion_tokens"),
                        "reasoning_tokens": rec.get("reasoning_tokens"),
                    },
                    "request_body": sub_req.request_body if sub_req else r.request_body,
                    "response_body": sub_req.response_body if sub_req else r.response_body,
                    "stream_chunks_count": 0,
                })
            continue
        functional_rows.append({
            **base_row,
            "id": r.meta.id,
            "name": r.meta.name,
            "http_status": r.http_status,
            "duration_ms": r.duration_ms,
            "usage": r.usage,
            "request_body": r.request_body,
            "response_body": r.response_body,
            "stream_chunks_count": len(r.stream_chunks),
        })

    # ---------- 验收标准汇总 ----------
    total_required = sum(1 for r in func_results if r.meta.required)
    passed_required = sum(
        1 for r in func_results if r.meta.required and r.status == TestStatus.PASS
    )
    failed_required = [
        r.meta.id for r in func_results
        if r.meta.required and r.status not in (TestStatus.PASS, TestStatus.WAIVE)
    ]

    cap_map: Dict[str, Dict[str, Any]] = {}
    for r in func_results:
        key = r.meta.capability_key
        if not key or not capability.get(key, False):
            continue
        m = cap_map.setdefault(key, {"total": 0, "passed": 0, "failed_ids": []})
        m["total"] += 1
        if r.status == TestStatus.PASS:
            m["passed"] += 1
        elif r.status != TestStatus.WAIVE:
            m["failed_ids"].append(r.meta.id)
    capability_pass = all(v["passed"] + 0 == v["total"] for v in cap_map.values()) if cap_map else True

    # ---------- 性能 ----------
    # GLM-5.3 无官方性能基线，压测只采事实数据（baseline_comparison 为有效性自检，
    # pass 恒为 None），不做达标判定，故不再计算 perf_ok/perf_unmeasurable。
    bench_summary = None
    if bench is not None:
        bench_summary = bench.to_dict()
        turns_by_round: Dict[int, Dict[str, Any]] = {}
        for t in bench.turns:
            b = turns_by_round.setdefault(t.round_idx, {
                "count": 0, "ok": 0,
                "ttft_ms": [], "latency_ms": [], "tpot_ms": [],
                "input_tokens": [], "output_tokens": [], "cached_tokens": [],
            })
            b["count"] += 1
            if t.http_status == 200 and t.error is None:
                b["ok"] += 1
                b["ttft_ms"].append(t.ttft_ms)
                b["latency_ms"].append(t.latency_ms)
                b["tpot_ms"].append(round(t.tpot_ms, 2))
                b["input_tokens"].append(t.input_tokens)
                b["output_tokens"].append(t.output_tokens)
                if t.cached_tokens is not None:
                    b["cached_tokens"].append(t.cached_tokens)

        def _avg(xs):
            try:
                return statistics.fmean(xs)
            except AttributeError:
                return sum(xs) / len(xs)

        for _r, b in turns_by_round.items():
            for k in ("ttft_ms", "latency_ms", "tpot_ms",
                      "input_tokens", "output_tokens", "cached_tokens"):
                b[k] = ({"avg": round(_avg(b[k]), 2), "p50": round(statistics.median(b[k]), 2)}
                        if b[k] else {"avg": 0, "p50": 0})
        bench_summary["turns_summary_by_round"] = turns_by_round
        # 实测工作负载口径：报告标题/总览引用真实数据而非声明值，
        # 避免出现"标题写32k、实测才1k"的失真（初版就发生过）。
        good_turns = [t for t in bench.turns if t.http_status == 200 and t.error is None]
        if good_turns:
            bench_summary["avg_input_tokens"] = round(
                sum(t.input_tokens for t in good_turns) / len(good_turns))
            bench_summary["avg_output_tokens"] = round(
                sum(t.output_tokens for t in good_turns) / len(good_turns))
        bench_summary.pop("turns", None)

    # KVV 精度测试未纳入 GLM-5.3 验收范围（kvv 参数保留仅为接口兼容，恒为 None）
    kvv_summary = None

    acceptance = {
        "必过用例": {
            "total": total_required,
            "passed": passed_required,
            "failed_ids": failed_required,
            "pass_rate_100": passed_required == total_required and total_required > 0,
            "verdict": (
                f"{passed_required}/{total_required} 全部通过"
                if passed_required == total_required and total_required > 0
                else f"{passed_required}/{total_required}，未通过："
                     f"{', '.join(failed_required) if failed_required else ''}"
            ),
        },
        "声明支持能力用例": {
            "detail": {
                k: {
                    "total": v["total"], "passed": v["passed"],
                    "failed_ids": v["failed_ids"],
                    "verdict": "通过" if v["passed"] == v["total"] else "有失败",
                }
                for k, v in cap_map.items()
            },
            "all_pass": capability_pass,
        },
        "文档最低能力要求": {
            "warnings": capability_warnings,
            "all_pass": len(capability_warnings) == 0,
        },
    }

    # ---------- 通俗结论：只陈述事实，一眼看出哪些有问题、哪些没问题 ----------
    def _short_failure(r: TestResult) -> str:
        """从失败原因里提取一句通俗短话。"""
        if not r.failures:
            return "执行异常" if r.status == TestStatus.ERROR else "未通过"
        f = r.failures[0]
        # 去掉修饰性后缀（官方文档引用、原始响应截取），只留主干
        for sep in ("。官方文档", "。响应", "官方文档：", "，实际HTTP", "。官方", "\n"):
            idx = f.find(sep)
            if idx > 8:
                f = f[:idx]
                break
        f = f.lstrip("✗🌐✅⚠️ ").strip()
        return f[:60]

    n_pass = sum(1 for r in func_results if r.status == TestStatus.PASS)
    n_fail = sum(1 for r in func_results if r.status == TestStatus.FAIL)
    n_error = sum(1 for r in func_results if r.status == TestStatus.ERROR)
    n_waive = sum(1 for r in func_results if r.status == TestStatus.WAIVE)
    n_total = len(func_results)

    acceptance["统计"] = {
        "total": n_total, "pass": n_pass, "fail": n_fail,
        "waive": n_waive, "error": n_error,
    }

    verdict_parts = [f"共 {n_total} 项测试：{n_pass} 项通过"]
    if n_fail:
        verdict_parts.append(f"{n_fail} 项未通过")
    if n_error:
        verdict_parts.append(f"{n_error} 项执行异常")
    if n_waive:
        verdict_parts.append(f"{n_waive} 项豁免（声明不支持，未测）")
    acceptance["最终结论"] = "，".join(verdict_parts) + "。"

    acceptance["问题清单"] = [
        {"id": r.meta.id, "name": r.meta.name, "summary": _short_failure(r)}
        for r in sorted(func_results, key=lambda x: x.meta.id)
        if r.status in (TestStatus.FAIL, TestStatus.ERROR)
    ]
    acceptance["正常项"] = [
        {"id": r.meta.id, "name": r.meta.name}
        for r in sorted(func_results, key=lambda x: x.meta.id)
        if r.status == TestStatus.PASS
    ]

    # 性能实测摘要（有跑压测才给事实数据，没跑就不出现）
    if bench is not None:
        ttft = bench.ttft_ms or {}
        perf_bits = [f"稳态吞吐 {bench.steady_throughput_req_per_s} req/s"]
        if ttft.get("p50") is not None:
            perf_bits.append(f"首字延迟中位数 {ttft['p50']} ms")
        if bench.cache_measurable and bench.cache_hit_rate_overall is not None:
            perf_bits.append(f"缓存命中率 {round(bench.cache_hit_rate_overall * 100, 2)}%")
        acceptance["性能实测"] = "，".join(perf_bits) + "。"
    elif bench_skip_reason:
        # 用户勾了压测但运行被取消：报告如实留痕，避免误以为配置丢失
        acceptance["性能实测"] = f"未执行（{bench_skip_reason}）。"

    return FullReport(
        timestamp=now.isoformat(timespec="seconds"),
        run_id=run_id,
        environment=env,
        capability_declare=capability,
        capability_warnings=capability_warnings,
        doc_coverage=DOC_COVERAGE_MAP,
        coverage_note=COVERAGE_NOTE,
        functional_results=functional_rows,
        benchmark=bench_summary,
        kvv=kvv_summary,
        acceptance_summary=acceptance,
        artifacts_dir=str(art_dir),
    )


# -------------------- 导出 JSON --------------------

def export_json(report: FullReport, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"glm-53-report-{report.run_id}.json"
    data = {
        "timestamp": report.timestamp,
        "run_id": report.run_id,
        "environment": report.environment,
        "capability_declare": report.capability_declare,
        "capability_warnings": report.capability_warnings,
        "doc_coverage": report.doc_coverage,
        "coverage_note": report.coverage_note,
        "acceptance_summary": report.acceptance_summary,
        "functional_results": report.functional_results,
        "benchmark": report.benchmark,
        "artifacts_dir": report.artifacts_dir,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


# -------------------- 导出 Excel --------------------

def export_excel(report: FullReport, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"glm-53-report-{report.run_id}.xlsx"

    wb = Workbook()
    thin = Side(border_style="thin", color="d0d0d0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="1e3a8a")
    head_font = Font(bold=True, color="ffffff", size=12)
    pass_fill = PatternFill("solid", fgColor="d1fae5")
    fail_fill = PatternFill("solid", fgColor="fee2e2")
    waive_fill = PatternFill("solid", fgColor="fef3c7")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_wrap = Alignment(horizontal="left", vertical="top", wrap_text=True)

    def _head(ws, row, cols):
        for i, h in enumerate(cols, 1):
            c = ws.cell(row=row, column=i, value=h)
            c.fill = head_fill; c.font = head_font; c.border = border; c.alignment = center

    a = report.acceptance_summary

    # ---------- Sheet1: 测试结论 ----------
    ws = wb.active
    ws.title = "测试结论"
    ws["A1"] = "GLM-5.3 模型验收测试报告"
    ws["A1"].font = Font(bold=True, size=16, color="1e3a8a")
    ws.merge_cells("A1:D1")
    stats = a["统计"]
    ws["A3"] = "测试结论"
    ws["B3"] = a["最终结论"]
    ws["B3"].font = Font(bold=True, size=14,
                         color="FF10B981" if stats["fail"] == 0 and stats["error"] == 0
                         else "FFEF4444")
    r0 = 4
    if a.get("问题清单"):
        ws.cell(row=r0, column=1, value=f"有问题的（{len(a['问题清单'])} 项）")
        ws.cell(row=r0, column=1).font = Font(bold=True, color="dc2626")
        for p in a["问题清单"]:
            r0 += 1
            ws.cell(row=r0, column=2, value=f"{p['id']}　{p['summary']}").alignment = left_wrap
        r0 += 1
    if a.get("正常项"):
        ws.cell(row=r0, column=1, value=f"没问题的（{len(a['正常项'])} 项）")
        ws.cell(row=r0, column=1).font = Font(bold=True, color="059669")
        ws.cell(row=r0, column=2,
                value=" ".join(p["id"] for p in a["正常项"])).alignment = left_wrap
        r0 += 1
    if a.get("性能实测"):
        ws.cell(row=r0, column=1, value="性能实测")
        ws.cell(row=r0, column=1).font = Font(bold=True)
        ws.cell(row=r0, column=2, value=a["性能实测"]).alignment = left_wrap
        r0 += 1

    row = r0 + 1
    _head(ws, row, ["项目", "详情", "是否满足", "备注"])

    def add_row(r, name, detail, ok, note=""):
        ws.cell(row=r, column=1, value=name).border = border
        cell = ws.cell(row=r, column=2,
                       value=json.dumps(detail, ensure_ascii=False)[:900]
                       if isinstance(detail, (dict, list)) else str(detail))
        cell.alignment = left_wrap; cell.border = border
        label = "✅ 满足" if ok is True else ("❌ 不满足" if ok is False else "⚠️ 无法判定")
        c3 = ws.cell(row=r, column=3, value=label)
        c3.alignment = center; c3.border = border
        c3.fill = pass_fill if ok is True else (fail_fill if ok is False else waive_fill)
        c4 = ws.cell(row=r, column=4, value=note)
        c4.alignment = left_wrap; c4.border = border
        return r + 1

    row += 1
    row = add_row(row, "必过用例通过率100%", a["必过用例"]["verdict"], a["必过用例"]["pass_rate_100"])
    row = add_row(row, "声明支持的能力用例全部通过",
                  {k: v["verdict"] for k, v in a["声明支持能力用例"]["detail"].items()},
                  a["声明支持能力用例"]["all_pass"])
    row = add_row(row, "文档最低能力要求",
                  [w["capability"] for w in a["文档最低能力要求"]["warnings"]] or "全部满足",
                  a["文档最低能力要求"]["all_pass"],
                  "；".join(w["reason"] for w in a["文档最低能力要求"]["warnings"]))

    for col, w in zip("ABCD", (34, 90, 16, 50)):
        ws.column_dimensions[col].width = w

    # ---------- Sheet2: 文档口径对账 ----------
    ws0 = wb.create_sheet("文档口径对账")
    ws0.cell(row=1, column=1, value="与《GLM-5.3 官方能力》的逐条对账").font = \
        Font(bold=True, size=14, color="1e3a8a")
    ws0.cell(row=2, column=1, value=report.coverage_note).alignment = left_wrap
    ws0.merge_cells("A2:E2")
    _head(ws0, 4, ["文档类别", "文档子项", "文档用例数", "是否必过", "本工具用例"])
    r = 5
    for m in report.doc_coverage:
        ws0.cell(row=r, column=1, value=m["doc_category"]).border = border
        ws0.cell(row=r, column=2, value=m["doc_subitems"]).alignment = left_wrap
        ws0.cell(row=r, column=3, value=m["doc_count"]).alignment = center
        ws0.cell(row=r, column=4, value=m["doc_required"]).alignment = left_wrap
        ws0.cell(row=r, column=5,
                 value=", ".join(m["case_ids"]) + (f"　※{m['note']}" if m["note"] else "")
                 ).alignment = left_wrap
        r += 1
    for col, w in zip("ABCDE", (22, 46, 12, 26, 60)):
        ws0.column_dimensions[col].width = w

    # ---------- Sheet3: 环境信息 + 能力声明 ----------
    ws2 = wb.create_sheet("环境信息与能力声明")
    ws2.cell(row=1, column=1, value="环境信息").font = Font(bold=True, size=14, color="1e3a8a")
    r = 3
    for k, v in report.environment.items():
        ws2.cell(row=r, column=1, value=k).border = border
        ws2.cell(row=r, column=2, value=str(v)).border = border
        r += 1
    r += 2
    ws2.cell(row=r, column=1, value="供应商能力声明").font = Font(bold=True, size=14, color="1e3a8a")
    r += 1
    _head(ws2, r, ["能力项", "是否支持", "说明"])
    r += 1
    warn_keys = {w["key"] for w in report.capability_warnings}
    for k, label in CAP_LABELS.items():
        v = report.capability_declare.get(k)
        ws2.cell(row=r, column=1, value=label).border = border
        ws2.cell(row=r, column=2, value="是" if v else "否").alignment = center
        ws2.cell(row=r, column=2).border = border
        note = ""
        if k in warn_keys:
            note = "⚠️ 文档规定的最低能力要求，声明为不支持需审核方确认"
            ws2.cell(row=r, column=2).fill = waive_fill
        ws2.cell(row=r, column=3, value=note).alignment = left_wrap
        ws2.cell(row=r, column=3).border = border
        r += 1
    for col, w in zip("ABC", (50, 16, 70)):
        ws2.column_dimensions[col].width = w

    # ---------- Sheet4: 功能测试结果表 ----------
    ws3 = wb.create_sheet("功能测试结果")
    headers = ["用例ID", "文档章节", "类别", "用例名称", "必过", "通过率", "HTTP状态",
               "耗时(ms)", "子断言", "结论", "失败归类", "失败原因", "诊断说明", "留档文件"]
    _head(ws3, 1, headers)
    r = 2
    for row_data in report.functional_results:
        vals = [
            row_data["id"], row_data.get("doc_ref", ""), row_data["category"], row_data["name"],
            "是" if row_data["required"] else "否",
            round(row_data["pass_rate"] * 100, 1) if row_data["total_assertions"]
            else ("豁免" if row_data["status"] == TestStatus.WAIVE else "—"),
            row_data["http_status"],
            row_data["duration_ms"],
            f"{row_data['passed_assertions']}/{row_data['total_assertions']}"
            if row_data["total_assertions"] else "—",
            STATUS_LABEL.get(row_data["status"], ("未知", "#888"))[0],
            row_data.get("error_kind", ""),
            "\n".join(f"• {x}" for x in (row_data["failures"] or [])),
            "\n".join(f"• {x}" for x in (row_data["notes"] or [])),
            row_data.get("artifact", ""),
        ]
        for i, v in enumerate(vals, 1):
            c = ws3.cell(row=r, column=i, value=v)
            c.border = border
            c.alignment = left_wrap if i in (12, 13) else center
        label, color = STATUS_LABEL.get(row_data["status"], ("未知", "#888"))
        ws3.cell(row=r, column=10).fill = {
            "#10b981": pass_fill, "#ef4444": fail_fill, "#f59e0b": waive_fill,
        }.get(color, PatternFill("solid", fgColor="ffffff"))
        r += 1
    for i, w in enumerate([12, 10, 20, 30, 8, 10, 10, 10, 10, 10, 14, 70, 70, 16], 1):
        ws3.column_dimensions[ws3.cell(row=1, column=i).column_letter].width = w
    ws3.freeze_panes = "A2"

    # ---------- Sheet5: 性能测试结果 ----------
    if report.benchmark:
        b = report.benchmark
        ws4 = wb.create_sheet("性能测试结果")
        ws4.cell(row=1, column=1, value="总览指标").font = Font(bold=True, size=14, color="1e3a8a")
        cache_txt = (f"{round((b.get('cache_hit_rate_overall') or 0) * 100, 2)}%"
                     if b.get("cache_measurable") else "无法测量")
        overview = [
            ("总请求数", b.get("total_requests")),
            ("成功 / 失败", f"{b.get('successful_requests')} / {b.get('failed_requests')}"),
            ("单请求平均输入/输出(tok)",
             f"{b.get('avg_input_tokens', '-')} / {b.get('avg_output_tokens', '-')}"),
            ("计划轮次 / 实际完成", f"{b.get('planned_turns')} / {b.get('completed_turns')}"),
            ("压测总时长(s)", b.get("duration_seconds")),
            ("稳态窗口(s)", str(b.get("steady_window"))
             + ("" if b.get("steady_metrics_valid", True) else "  ⚠️ 窗口内无样本，已退化为全程口径")),
            ("计划发压速率 offered(req/s)", b.get("offered_load_req_per_s")),
            ("实际吞吐 achieved 全程(req/s)", b.get("throughput_req_per_s")),
            ("实际吞吐 achieved 稳态(req/s)",
             b.get("steady_throughput_req_per_s") if b.get("steady_metrics_valid", True)
             else "未覆盖稳态窗口"),
            ("是否压到饱和",
             "计划轮次提前跑完，未持续施压" if b.get("ran_out_of_work") and not b.get("steady_metrics_valid", True)
             else ("是" if b.get("saturated") else "否（吞吐受发压速率约束）")),
            ("输入吞吐(tok/s)", b.get("throughput_input_tok_per_s")),
            ("输出吞吐(tok/s)", b.get("throughput_output_tok_per_s")),
            ("整体缓存命中率", cache_txt),
            ("疑似非增量流式请求数", b.get("degenerate_stream_count")),
        ]
        r = 3
        for k, v in overview:
            ws4.cell(row=r, column=1, value=k).border = border
            ws4.cell(row=r, column=2, value=v).border = border
            r += 1
        if b.get("warnings"):
            r += 1
            ws4.cell(row=r, column=1, value="执行告警").font = Font(bold=True, size=12, color="b45309")
            for w in b["warnings"]:
                r += 1
                ws4.cell(row=r, column=1, value="• " + str(w)).alignment = left_wrap
                ws4.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)

        r += 2
        ws4.cell(row=r, column=1,
                 value="延迟分位数（稳态窗口口径，已剔除非增量流式请求）").font = \
            Font(bold=True, size=14, color="1e3a8a")
        r += 1
        _head(ws4, r, ["指标", "avg", "p50", "p75", "p90", "p95", "p99"])
        for name, key, unit in [("TTFT", "ttft_ms", "ms"), ("Latency", "latency_ms", "ms"),
                                ("TPOT", "tpot_ms", "ms"), ("ITL", "itl_ms", "ms")]:
            r += 1
            ws4.cell(row=r, column=1, value=f"{name}({unit})").border = border
            vals = b.get(key, {}) or {}
            for i, k in enumerate(["avg", "p50", "p75", "p90", "p95", "p99"], 2):
                c = ws4.cell(row=r, column=i, value=vals.get(k, 0))
                c.border = border; c.alignment = center

        r += 2
        ws4.cell(row=r, column=1, value="压测有效性自检").font = \
            Font(bold=True, size=14, color="1e3a8a")
        r += 1
        _head(ws4, r, ["检查项", "结果", "说明"])
        for k, v in (b.get("baseline_comparison") or {}).items():
            r += 1
            ws4.cell(row=r, column=1, value=k).border = border
            ws4.cell(row=r, column=2, value=str(v.get("actual"))).border = border
            ws4.cell(row=r, column=3, value=v.get("note", "")).alignment = left_wrap

        r += 2
        ws4.cell(row=r, column=1, value="分轮次缓存命中率").font = Font(bold=True, size=14, color="1e3a8a")
        r += 1
        if b.get("cache_measurable"):
            _head(ws4, r, ["Round", "命中率", "请求数"])
            tbr = b.get("turns_summary_by_round") or {}
            rates = b.get("cache_hit_rate_by_round") or {}
            for k in sorted(tbr.keys(), key=lambda x: int(x)):
                r += 1
                rate = rates.get(str(k), rates.get(int(k), 0))
                ws4.cell(row=r, column=1, value=f"Round {k}").border = border
                ws4.cell(row=r, column=2, value=f"{round(float(rate) * 100, 2)}%").border = border
                ws4.cell(row=r, column=3, value=tbr[k]["count"]).border = border
        else:
            ws4.cell(row=r, column=1, value=b.get("cache_note", "无法测量")).alignment = left_wrap
            ws4.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
        for col, w in zip("ABCDEFG", (40, 20, 20, 16, 60, 12, 12)):
            ws4.column_dimensions[col].width = w

    # KVV 精度测试未纳入 GLM-5.3 验收范围，Excel 不生成 KVV sheet

    wb.save(str(path))
    return path


# -------------------- 导出 HTML --------------------

def export_html(report: FullReport, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"glm-53-report-{report.run_id}.html"

    def _esc(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return html.escape(json.dumps(v, ensure_ascii=False, default=str))
        return html.escape(str(v))

    a = report.acceptance_summary
    final_verdict = a["最终结论"]
    _stats = a["统计"]
    verdict_color = ("#10b981" if _stats["fail"] == 0 and _stats["error"] == 0 else "#ef4444")

    total_func = len(report.functional_results)
    pass_func = sum(1 for r in report.functional_results if r["status"] == TestStatus.PASS)
    fail_func = sum(1 for r in report.functional_results if r["status"] == TestStatus.FAIL)
    waive_func = sum(1 for r in report.functional_results if r["status"] == TestStatus.WAIVE)
    error_func = sum(1 for r in report.functional_results if r["status"] == TestStatus.ERROR)

    warn_keys = {w["key"] for w in report.capability_warnings}
    capability_rows_html = "".join(
        f'<tr><td class="k">{_esc(label)}</td>'
        f'<td class="v">{"✅ 支持" if report.capability_declare.get(k) else "🟡 不支持（豁免对应用例）"}'
        f'{"<br><span style=color:#b45309>⚠️ 文档规定的最低能力要求</span>" if k in warn_keys else ""}'
        f'</td></tr>'
        for k, label in CAP_LABELS.items()
    )

    coverage_rows = "".join(
        f'<tr><td>{_esc(m["doc_category"])}</td><td>{_esc(m["doc_subitems"])}</td>'
        f'<td style="text-align:center">{m["doc_count"]}</td>'
        f'<td>{_esc(m["doc_required"])}</td>'
        f'<td class="mono">{_esc(", ".join(m["case_ids"]))}</td>'
        f'<td class="small">{_esc(m["note"])}</td></tr>'
        for m in report.doc_coverage
    )

    func_rows_html = ""
    for row in report.functional_results:
        status_text, status_color = STATUS_LABEL.get(row["status"], ("未知", "#888"))
        req_marker = ('<span style="color:#ef4444;font-weight:700">是</span>'
                      if row["required"] else "否")
        pct = round(row["pass_rate"] * 100, 1) if row["total_assertions"] else "—"
        failures_html = "<br>".join(
            f'<span style="color:#b91c1c">• {_esc(f)}</span>' for f in (row["failures"] or [])
        )
        notes_html = "<br>".join(
            f'<span style="color:#475569">• {_esc(n)}</span>' for n in (row["notes"] or [])
        )
        trace_html = ""
        if row["error_trace"]:
            trace_html = (f'<details><summary>异常堆栈</summary>'
                          f'<pre>{_esc(row["error_trace"][:2000])}</pre></details>')
        # 文档 2.1：每个用例都要回传请求体与响应体，不再只对失败项输出
        req_html = _esc(_truncate(row["request_body"], 2000))
        resp_html = _esc(_truncate(row["response_body"], 3000))
        detail_html = (
            f'<details><summary>请求体 / 响应体</summary>'
            f'<pre>{req_html}</pre><hr><pre>{resp_html}</pre>'
            f'<div class="small">完整留档（含SSE分片、子请求）: '
            f'artifacts/{_esc(report.run_id)}/{_esc(row.get("artifact", ""))}</div>'
            f'</details>'
        )
        retry_badge = ('<span class="badge-warn">重试后通过</span>'
                       if row.get("retried") and row["status"] == TestStatus.PASS else "")
        func_rows_html += f'''
          <tr>
            <td class="mono">{_esc(row["id"])}{retry_badge}</td>
            <td class="small">{_esc(row.get("doc_ref", ""))}</td>
            <td>{_esc(row["category"])}</td>
            <td>{_esc(row["name"])}</td>
            <td style="text-align:center">{req_marker}</td>
            <td style="text-align:center">{pct}</td>
            <td style="text-align:center">{row["http_status"] if row["http_status"] is not None else "—"}</td>
            <td style="text-align:center">{row["duration_ms"]}</td>
            <td style="text-align:center">{row["passed_assertions"]}/{row["total_assertions"] if row["total_assertions"] else "—"}</td>
            <td style="text-align:center;background:{status_color}22;color:{status_color};font-weight:700">{status_text}</td>
            <td class="small">{failures_html}{trace_html}</td>
            <td class="small">{notes_html}{detail_html}</td>
          </tr>
        '''

    # 性能
    perf_html = ""
    if report.benchmark:
        b = report.benchmark
        cache_val = ("无法测量" if not b.get("cache_measurable")
                     else f"{round((b.get('cache_hit_rate_overall') or 0) * 100, 2)}%")
        overview = [
            ("总请求数", b.get("total_requests", 0)),
            ("成功 / 失败", f"{b.get('successful_requests', 0)} / {b.get('failed_requests', 0)}"),
            ("计划 / 完成轮次", f"{b.get('planned_turns', 0)} / {b.get('completed_turns', 0)}"),
            ("单请求平均输入(tok)", b.get("avg_input_tokens", 0)),
            ("单请求平均输出(tok)", b.get("avg_output_tokens", 0)),
            ("压测时长(s)", b.get("duration_seconds", 0)),
            ("offered 发压速率(req/s)", b.get("offered_load_req_per_s", 0)),
            ("achieved 稳态吞吐(req/s)",
             b.get("steady_throughput_req_per_s", 0) if b.get("steady_metrics_valid", True)
             else "未覆盖稳态窗口"),
            ("是否压到饱和",
             "轮次提前跑完" if b.get("ran_out_of_work") and not b.get("steady_metrics_valid", True)
             else ("是" if b.get("saturated") else "否")),
            ("输出吞吐(tok/s)", b.get("throughput_output_tok_per_s", 0)),
            ("整体缓存命中率", cache_val),
            ("疑似非增量流式", b.get("degenerate_stream_count", 0)),
        ]
        perf_html += '<h3>3.1 总览</h3><div class="grid">'
        for k, v in overview:
            perf_html += (f'<div class="card-item"><div class="label">{_esc(k)}</div>'
                          f'<div class="val">{_esc(v)}</div></div>')
        perf_html += "</div>"

        if b.get("warnings"):
            perf_html += '<div class="warnbox"><b>执行告警</b><ul>'
            for w in b["warnings"]:
                perf_html += f"<li>{_esc(w)}</li>"
            perf_html += "</ul></div>"

        def dist_table(title, key, unit, is_float=False):
            vals = b.get(key, {}) or {}
            rows = ""
            for k in ("avg", "p50", "p75", "p90", "p95", "p99"):
                v = vals.get(k, 0)
                s = f"{v:.2f} {unit}" if is_float else f"{int(v)} {unit}"
                rows += f"<tr><td>{k.upper()}</td><td>{s}</td></tr>"
            return f'<div class="dist"><h4>{title}</h4><table>{rows}</table></div>'

        perf_html += ('<h3>3.2 分位数分布 '
                      '<span class="small">（稳态窗口口径，已剔除非增量流式请求）</span></h3>'
                      '<div class="dist-grid">')
        perf_html += dist_table("TTFT 首Token延迟", "ttft_ms", "ms", False)
        perf_html += dist_table("Latency 端到端延迟", "latency_ms", "ms", False)
        perf_html += dist_table("TPOT 每输出Token耗时", "tpot_ms", "ms", True)
        perf_html += dist_table("ITL Token间延迟", "itl_ms", "ms", True)
        perf_html += "</div>"

        perf_html += "<h3>3.3 压测有效性自检</h3><table>"
        perf_html += '<tr><th>检查项</th><th>结果</th><th>说明</th></tr>'
        for k, v in (b.get("baseline_comparison", {}) or {}).items():
            perf_html += (
                f'<tr><td>{_esc(k)}</td>'
                f'<td style="font-weight:700">{_esc(v.get("actual"))}</td>'
                f'<td class="small">{_esc(v.get("note", ""))}</td></tr>'
            )
        perf_html += "</table>"

        perf_html += "<h3>3.4 分轮次缓存命中率</h3>"
        if b.get("cache_measurable"):
            rounds_rows = ""
            for k in sorted((b.get("cache_hit_rate_by_round") or {}).keys(), key=lambda x: int(x)):
                rate = b["cache_hit_rate_by_round"][k]
                rounds_rows += f'<tr><td>Round {k}</td><td>{round(float(rate) * 100, 2)}%</td></tr>'
            perf_html += f'<table><tr><th>轮次</th><th>命中率</th></tr>{rounds_rows}</table>'
            perf_html += f'<div class="small">{_esc(b.get("cache_note", ""))}</div>'
        else:
            perf_html += f'<div class="warnbox">{_esc(b.get("cache_note", "无法测量"))}</div>'

    # KVV 精度测试未纳入 GLM-5.3 验收范围，报告不再渲染该章节

    reasons_html = ""
    if a.get("问题清单"):
        reasons_html += (f'<div class="problembox"><b>❌ 有问题的（{len(a["问题清单"])} 项）</b><ul class="plain">'
                         + "".join(
                             f'<li><b>{_esc(p["id"])}</b> {_esc(p["summary"])}</li>'
                             for p in a["问题清单"]) + "</ul></div>")
    if a.get("正常项"):
        reasons_html += (f'<div class="okbox"><b>✅ 没问题的（{len(a["正常项"])} 项）</b> '
                         + " · ".join(_esc(p["id"]) for p in a["正常项"])
                         + "</div>")
    if a.get("性能实测"):
        reasons_html += f'<div class="infobox">{_esc(a["性能实测"])}</div>'
    cap_warn_html = ""
    if report.capability_warnings:
        cap_warn_html = '<div class="warnbox"><b>⚠️ 文档最低能力要求未满足</b><ul>'
        for w in report.capability_warnings:
            cap_warn_html += (f'<li><b>{_esc(w["capability"])}</b> — {_esc(w["reason"])}<br>'
                              f'<span class="small">{_esc(w["impact"])}</span></li>')
        cap_warn_html += "</ul></div>"

    doc = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>GLM-5.3 模型验收测试报告</title>
<style>
  :root {{ --brand:#1e40af; --bg:#f6f8fc; --card:#fff; --border:#e5e7eb; --text:#0f172a; --muted:#64748b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px; font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); }}
  .wrap {{ max-width:1500px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 8px; color:var(--brand); }}
  .sub {{ color:var(--muted); margin-bottom:24px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:24px; margin-bottom:20px; }}
  h2 {{ font-size:18px; margin:0 0 16px; border-left:4px solid var(--brand); padding-left:10px; }}
  h3 {{ font-size:16px; margin:24px 0 12px; color:var(--brand); }}
  h4 {{ font-size:14px; margin:0 0 8px; }}
  .verdict {{ padding:18px 20px; border-radius:12px; font-size:20px; font-weight:700; color:#fff; background:{verdict_color}; }}
  .stats {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-top:14px; }}
  .stat {{ border-radius:10px; padding:14px; background:#f8fafc; border:1px solid var(--border); }}
  .stat .l {{ color:var(--muted); font-size:12px; }}
  .stat .v {{ font-size:22px; font-weight:700; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ border:1px solid var(--border); padding:8px 10px; vertical-align:top; }}
  th {{ background:#f1f5f9; color:var(--brand); font-weight:600; text-align:left; }}
  .mono {{ font-family:ui-monospace,Menlo,Consolas,monospace; color:#7c3aed; font-weight:600; }}
  .small {{ font-size:12px; color:#334155; }}
  .grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; }}
  .card-item {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:12px; }}
  .card-item .label {{ font-size:12px; color:var(--muted); }}
  .card-item .val {{ font-size:17px; font-weight:700; margin-top:4px; color:var(--brand); }}
  .dist-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }}
  .dist {{ background:#f8fafc; padding:12px; border-radius:10px; border:1px solid var(--border); }}
  .warnbox {{ background:#fffbeb; border:1px solid #fcd34d; border-radius:10px; padding:12px 16px; margin:12px 0; font-size:13px; }}
  .problembox {{ background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:12px 16px; margin:12px 0; font-size:13px; }}
  .problembox ul.plain {{ margin:6px 0 0; padding:0; list-style:none; line-height:2; }}
  .okbox {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:10px 16px; margin:12px 0; font-size:13px; color:#166534; }}
  .infobox {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:10px 16px; margin:12px 0; font-size:13px; color:#1e40af; }}
  .badge-warn {{ background:#fef3c7; color:#92400e; font-size:10px; padding:1px 5px; border-radius:4px; margin-left:4px; }}
  details {{ margin-top:6px; }}
  pre {{ margin:0; font-size:11px; line-height:1.5; white-space:pre-wrap; background:#f8fafc; padding:6px; border-radius:4px; }}
  .cap-table td:first-child {{ font-weight:600; width:320px; }}
  @media print {{ body {{ padding:0; background:#fff; }} .card {{ break-inside:avoid; }} }}
</style></head><body>
<div class="wrap">
  <h1>GLM-5.3 模型验收测试报告</h1>
  <div class="sub">生成时间：{_esc(report.timestamp)} · 模型：{_esc(report.environment.get("model_id"))} · 运行ID：{_esc(report.run_id)}</div>

  <div class="card">
    <h2>📌 测试结论</h2>
    <div class="verdict">{_esc(final_verdict)}</div>
    {reasons_html}
    <div class="stats">
      <div class="stat"><div class="l">功能用例总数</div><div class="v">{total_func}</div></div>
      <div class="stat"><div class="l">✅ 通过</div><div class="v" style="color:#10b981">{pass_func}</div></div>
      <div class="stat"><div class="l">❌ 失败</div><div class="v" style="color:#ef4444">{fail_func}</div></div>
      <div class="stat"><div class="l">🟡 豁免</div><div class="v" style="color:#f59e0b">{waive_func}</div></div>
      <div class="stat"><div class="l">💥 异常</div><div class="v" style="color:#7c3aed">{error_func}</div></div>
    </div>
    {cap_warn_html}
  </div>

  <div class="card">
    <h2>0. 测试范围与官方能力对账</h2>
    <div class="warnbox">{_esc(report.coverage_note)}</div>
    <table>
      <tr><th>文档类别</th><th>文档子项</th><th style="width:90px">文档用例数</th>
          <th style="width:180px">是否必过</th><th style="width:200px">本工具用例</th><th>备注</th></tr>
      {coverage_rows}
    </table>
  </div>

  <div class="card">
    <h2>1. 环境信息</h2>
    <table>
      <tr><th style="width:220px">项目</th><th>值</th></tr>
      <tr><td>模型ID (model-id)</td><td>{_esc(report.environment.get("model_id"))}</td></tr>
      <tr><td>API端点 (/v1/chat/completions)</td><td>{_esc(report.environment.get("api_endpoint"))}</td></tr>
      <tr><td>API Base</td><td>{_esc(report.environment.get("api_base"))}</td></tr>
      <tr><td>测试日期</td><td>{_esc(report.environment.get("test_date"))}</td></tr>
      <tr><td>测试器版本</td><td>{_esc(report.environment.get("tester_version"))}</td></tr>
      <tr><td>瞬时错误重试次数</td><td>{_esc(report.environment.get("max_retries"))}</td></tr>
      <tr><td>完整留档目录</td><td class="mono">{_esc(report.artifacts_dir)}</td></tr>
    </table>
    <h2 style="margin-top:28px">1.1 能力声明与豁免说明</h2>
    <table class="cap-table">{capability_rows_html}</table>
  </div>

  <div class="card">
    <h2>2. 功能测试结果表（{total_func} 行）</h2>
    <table>
      <tr>
        <th style="width:110px">用例ID</th><th style="width:60px">文档</th>
        <th style="width:130px">类别</th><th>用例名称</th>
        <th style="width:56px">必过</th><th style="width:70px">通过率</th>
        <th style="width:64px">HTTP</th><th style="width:80px">耗时(ms)</th>
        <th style="width:80px">子断言</th><th style="width:70px">结论</th>
        <th style="width:320px">失败原因</th>
        <th style="width:340px">诊断说明 / 请求响应留档</th>
      </tr>
      {func_rows_html}
    </table>
  </div>

  {f'<div class="card"><h2>3. 性能测试结果（多轮会话压测，实测平均输入 {b.get("avg_input_tokens", "-")} tok / 输出 {b.get("avg_output_tokens", "-")} tok）</h2>{perf_html}</div>' if perf_html else ''}

  <div class="card">
    <h2>附录：判定标准说明</h2>
    <ul>
      <li>「必过」用例共 {a["必过用例"]["total"]} 项，要求通过率100%、HTTP 200。</li>
      <li>声明支持的能力，其对应用例须全部通过。</li>
    </ul>
    <p class="small">说明：本报告对"无法测量"的指标（如供应商未回传 cached_tokens）
    单独标注，不计为达标也不计为劣化，需补充数据后复核。</p>
  </div>
</div></body></html>'''
    path.write_text(doc, encoding="utf-8")
    return path


# -------------------- 导出 PDF --------------------

def export_pdf(report: FullReport, out_dir: Path | None = None) -> Path:
    """使用 reportlab 生成 PDF（中文字体：内置 STSong-Light CID，无需外部字体文件）"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )

    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"glm-53-report-{report.run_id}.pdf"

    font_name = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    except Exception:
        font_name = "Helvetica"

    navy = colors.HexColor("#1e3a8a")
    green = colors.HexColor("#10b981")
    red = colors.HexColor("#ef4444")
    amber = colors.HexColor("#f59e0b")
    gray = colors.HexColor("#64748b")

    st_title = ParagraphStyle("title", fontName=font_name, fontSize=18, textColor=navy,
                              spaceAfter=4, leading=24)
    st_sub = ParagraphStyle("sub", fontName=font_name, fontSize=9, textColor=gray,
                            spaceAfter=10, leading=12)
    st_h2 = ParagraphStyle("h2", fontName=font_name, fontSize=13, textColor=navy,
                           spaceBefore=14, spaceAfter=6, leading=18)
    st_body = ParagraphStyle("body", fontName=font_name, fontSize=9, leading=13)

    story = []
    a = report.acceptance_summary
    _stats = a["统计"]
    ok = _stats["fail"] == 0 and _stats["error"] == 0

    story.append(Paragraph("GLM-5.3 模型验收测试报告", st_title))
    story.append(Paragraph(
        f"生成时间: {report.timestamp}　|　模型: {report.environment.get('model_id')}"
        f"　|　工具版本: {report.environment.get('tester_version', '1.1.0')}", st_sub))
    st_verdict = ParagraphStyle("verdict", fontName=font_name, fontSize=14,
                                textColor=green if ok else red,
                                leading=20, spaceAfter=8)
    story.append(Paragraph(f"测试结论：{a['最终结论']}", st_verdict))
    if a.get("问题清单"):
        story.append(Paragraph("❌ 有问题的（%d 项）" % len(a["问题清单"]), st_h2))
        for p in a["问题清单"]:
            story.append(Paragraph(f"{p['id']}　{p['summary']}", st_body))
    if a.get("正常项"):
        story.append(Paragraph(
            "✅ 没问题的（%d 项）：" % len(a["正常项"])
            + " ".join(p["id"] for p in a["正常项"]), st_body))
    if a.get("性能实测"):
        story.append(Paragraph(a["性能实测"], st_body))

    def kv_table(data, widths, header=True):
        st_header = ParagraphStyle("hdr", fontName=font_name, fontSize=8,
                                   textColor=colors.white, leading=11)
        st_cell = ParagraphStyle("cell", fontName=font_name, fontSize=8,
                                 textColor=colors.HexColor("#1e293b"), leading=11)
        if header:
            data = ([[Paragraph(str(c), st_header) for c in data[0]]]
                    + [[Paragraph(str(c), st_cell) for c in row] for row in data[1:]])
        else:
            data = [[Paragraph(str(c), st_cell) for c in row] for row in data]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), navy))
        t.setStyle(TableStyle(style))
        return t

    def _ok_txt(v):
        return "满足" if v is True else ("不满足" if v is False else "无法判定")

    story.append(Paragraph("一、验收明细汇总", st_h2))
    acc_rows = [["验收项", "结果", "是否满足"]]
    acc_rows.append(["必过用例通过率100%", a["必过用例"]["verdict"],
                     _ok_txt(a["必过用例"]["pass_rate_100"])])
    cap_txt = "；".join(f"{k}: {v['verdict']}({v['passed']}/{v['total']})"
                       for k, v in a["声明支持能力用例"]["detail"].items()) or "无声明能力用例"
    acc_rows.append(["声明支持的能力用例全部通过", cap_txt,
                     _ok_txt(a["声明支持能力用例"]["all_pass"])])
    acc_rows.append(["文档最低能力要求",
                     "；".join(w["capability"] for w in a["文档最低能力要求"]["warnings"]) or "全部满足",
                     _ok_txt(a["文档最低能力要求"]["all_pass"])])
    story.append(kv_table(acc_rows, [50*mm, 95*mm, 25*mm]))

    story.append(Paragraph("二、环境信息与能力声明", st_h2))
    env_rows = [["项目", "值"]] + [[k, str(v)] for k, v in report.environment.items()]
    story.append(kv_table(env_rows, [55*mm, 115*mm]))
    story.append(Spacer(1, 6))
    warn_keys = {w["key"] for w in report.capability_warnings}
    cap_rows = [["能力项", "声明", "备注"]]
    for k, label in CAP_LABELS.items():
        cap_rows.append([
            label,
            "支持" if report.capability_declare.get(k) else "不支持",
            "文档最低要求" if k in warn_keys else "",
        ])
    story.append(kv_table(cap_rows, [95*mm, 30*mm, 45*mm]))

    story.append(PageBreak())
    story.append(Paragraph("三、功能用例明细", st_h2))
    story.append(Paragraph(
        f"共 {len(report.functional_results)} 条记录。完整请求/响应/SSE分片留档于 "
        f"{report.artifacts_dir}", st_body))
    story.append(Spacer(1, 4))
    st_hdr = ParagraphStyle("hdr2", fontName=font_name, fontSize=8,
                            textColor=colors.white, leading=11)
    st_cell = ParagraphStyle("body2", fontName=font_name, fontSize=8,
                             textColor=colors.HexColor("#1e293b"), leading=11)
    st_red = ParagraphStyle("bodyr", fontName=font_name, fontSize=8,
                            textColor=colors.HexColor("#dc2626"), leading=11)
    func_data = [[Paragraph(h, st_hdr)
                  for h in ["用例", "名称", "状态", "HTTP", "耗时ms", "失败原因(截取)"]]]
    for r in report.functional_results:
        label, _c = STATUS_LABEL.get(r["status"], (r["status"], None))
        fail_txt = "<br/>".join(html.escape(f[:120]) for f in (r.get("failures") or [])[:3])
        st = st_red if r["status"] == TestStatus.FAIL else st_cell
        func_data.append([
            Paragraph(str(r["id"]), st), Paragraph(html.escape(str(r["name"])), st),
            Paragraph(label, st), Paragraph(str(r.get("http_status") or "-"), st),
            Paragraph(str(r.get("duration_ms") or "-"), st),
            Paragraph(fail_txt or "-", st),
        ])
    func_t = Table(func_data, colWidths=[15*mm, 38*mm, 13*mm, 13*mm, 16*mm, 65*mm], repeatRows=1)
    func_t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), navy),
    ]))
    story.append(func_t)

    b = report.benchmark
    if b:
        story.append(Paragraph("四、性能压测结果", st_h2))
        rows = [["指标", "数值"]]
        rows.append(["总请求数", str(b.get("total_requests", "-"))])
        rows.append(["成功/失败",
                     f"{b.get('successful_requests', '-')} / {b.get('failed_requests', '-')}"])
        rows.append(["单请求平均输入/输出(tok)",
                     f"{b.get('avg_input_tokens', '-')} / {b.get('avg_output_tokens', '-')}"])
        rows.append(["持续时长(s)", f"{b.get('duration_seconds', 0):.0f}"])
        rows.append(["offered 发压速率(req/s)", str(b.get("offered_load_req_per_s", "-"))])
        rows.append(["achieved 稳态吞吐(req/s)", str(b.get("steady_throughput_req_per_s", "-"))])
        rows.append(["是否压到饱和", "是" if b.get("saturated") else "否"])
        ttft = b.get("ttft_ms") or {}
        rows.append(["TTFT avg/p50/p95(ms)",
                     f"{ttft.get('avg', '-')} / {ttft.get('p50', '-')} / {ttft.get('p95', '-')}"])
        tpot = b.get("tpot_ms") or {}
        rows.append(["TPOT avg/p50/p95(ms)",
                     f"{tpot.get('avg', '-')} / {tpot.get('p50', '-')} / {tpot.get('p95', '-')}"])
        rows.append(["缓存命中率",
                     f"{(b.get('cache_hit_rate_overall') or 0) * 100:.2f}%"
                     if b.get("cache_measurable") else "无法测量"])
        rows.append(["疑似非增量流式请求", str(b.get("degenerate_stream_count", 0))])
        story.append(kv_table(rows, [70*mm, 100*mm]))
        comp = b.get("baseline_comparison") or {}
        if comp:
            story.append(Spacer(1, 6))
            story.append(Paragraph("压测有效性自检", st_h2))
            comp_rows = [["检查项", "结果", "说明"]]
            for k, v in comp.items():
                comp_rows.append([
                    k, str(v.get("actual")),
                    str(v.get("note", "")),
                ])
            story.append(kv_table(comp_rows, [45*mm, 45*mm, 80*mm]))

    # KVV 精度测试未纳入 GLM-5.3 验收范围，不渲染该章节

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm,
        title="GLM-5.3 模型验收测试报告",
    )
    doc.build(story)
    return path


def export_all(
    cfg: TesterConfig,
    func_results: List[TestResult],
    bench: Optional[BenchmarkResult],
    kvv: Optional[KVVResult],
    out_dir: Path | None = None,
    bench_skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """导出全部格式。

    每种格式独立 try/except：初版四种串行调用，PDF 一抛异常（例如未安装
    reportlab）就会让前面已生成的 JSON/Excel/HTML 全部作废，整轮测试结果丢失。
    """
    report = build_full_report(cfg, func_results, bench, kvv, out_dir,
                               bench_skip_reason=bench_skip_reason)
    out: Dict[str, Any] = {"_report_object": report, "_errors": {}}
    for kind, fn in (
        ("json", export_json), ("xlsx", export_excel),
        ("html", export_html), ("pdf", export_pdf),
    ):
        try:
            out[kind] = fn(report, out_dir)
        except Exception as e:
            out["_errors"][kind] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    return out
