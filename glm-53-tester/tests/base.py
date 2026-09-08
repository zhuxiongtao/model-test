"""测试基类与通用工具

严格执行文档 2.1 通用要求：
  - 每个用例通过率需达100%，HTTP状态码为200
  - 请求/响应结构符合 OpenAI ChatCompletions 规范
  - 回传：HTTP状态码、耗时(ms)、完整请求体、完整响应体（或流式分片）

本版相对初版的关键修正：
  1. failures 只放"判定失败的原因"，诊断性信息走 notes，避免通过的用例
     在报告里挂一堆红色 [INFO]。
  2. 多子请求用例（思考开关、reasoning_effort、多模态形状回退）每个子请求
     单独计时并完整留档到 sub_requests，文档 2.1 要求"每个用例回传耗时与
     完整请求/响应"，此前子请求的耗时恒为 0、错误诊断被整段丢弃。
  3. 5xx / 超时做有限重试。文档要求通过率 100%，但网关偶发 504 不等于功能
     缺失；重试信息记入 notes，让评审能区分"稳定性差"和"能力缺失"。
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from config import TesterConfig, get_config

# -------------------- 数据结构 --------------------

class TestStatus:
    PASS = "pass"               # 通过
    FAIL = "fail"               # 失败
    WAIVE = "waive"             # 豁免（供应商声明不支持 / 样本不可用）
    ERROR = "error"             # 执行异常（非断言失败）
    SKIP = "skip"               # 未执行


class ErrorKind:
    """失败归类。同样是"没通过"，原因不同对供应商的整改要求完全不同。"""
    NONE = ""
    AUTH = "auth"                   # 401/403
    RATE_LIMIT = "rate_limit"       # 429
    GATEWAY = "gateway"             # 502/503/504 网关超时或不可用
    SERVER = "server"               # 其他 5xx
    TOO_LARGE = "payload_too_large"  # 413
    BAD_REQUEST = "bad_request"     # 400，通常是参数/格式不被接受
    NETWORK = "network"             # 客户端侧网络异常
    PROTOCOL = "protocol"           # 200 但响应结构不符合规范
    CAPABILITY = "capability"       # 链路正常，但能力表现不达标


# 重试的状态码：这些是"再来一次可能就好了"的瞬时错误
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# 网关超时类：这类请求本身已经耗掉了网关的整个超时预算（几十秒到几分钟），
# 原样重试的代价极高而成功率很低，因此单独限制重试次数。
GATEWAY_TIMEOUT_STATUS = {502, 503, 504}
MAX_GATEWAY_RETRIES = 1


class GatewayHealth:
    """全局熔断：供应商持续网关超时时，停止无谓的重试。

    实测教训：一轮功能测试跑了 32 分钟，其中 22.5 分钟耗在 5 条用例对 504 的
    重复重试上（每次 504 本身就已经等满了网关超时）。供应商这段时间明显处于
    不健康状态，继续重试既拖长总时长，也不会得到不同的结果。
    """

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.consecutive_gateway_failures = 0
        self.tripped = False

    def reset(self):
        self.consecutive_gateway_failures = 0
        self.tripped = False

    def record(self, status: Optional[int]):
        if status in GATEWAY_TIMEOUT_STATUS or status is None:
            self.consecutive_gateway_failures += 1
            if self.consecutive_gateway_failures >= self.threshold:
                self.tripped = True
        elif status == 200:
            self.consecutive_gateway_failures = 0
            self.tripped = False

    def allow_retry(self) -> bool:
        return not self.tripped


GATEWAY_HEALTH = GatewayHealth()


@dataclass
class SSEChunk:
    """流式SSE分片记录"""
    index: int
    raw: str                           # data: 后的原始字符串（[DONE]或json）
    event: str = "data"
    delta_content: str = ""            # 提取出的文本增量
    reasoning_delta: str = ""          # 思考增量
    tool_calls_delta: List[Dict] = field(default_factory=list)
    parsed_json: Optional[Dict] = None # 成功解析则存放
    elapsed_ms: int = 0                # 相对请求开始的到达时刻，用于识别伪流式


@dataclass
class SubRequest:
    """一个用例内部的单次 HTTP 往返的完整留档（文档 2.1 回传要求）。"""
    label: str
    request_body: Optional[Dict] = None
    response_body: Optional[Any] = None
    stream_chunks: List[Dict[str, Any]] = field(default_factory=list)
    http_status: Optional[int] = None
    duration_ms: int = 0
    usage: Optional[Dict] = None
    attempts: int = 1
    error_kind: str = ErrorKind.NONE
    error_detail: str = ""


@dataclass
class TestCaseMeta:
    id: str                             # F1~F24
    category: str                       # 协议完整性/多模态/工具调用...
    name: str                           # 用例名称
    required: bool                      # 是否必过
    check_points: List[str]             # 检查点（文档原文）
    capability_key: Optional[str] = None  # 对应能力声明键；不支持时豁免
    doc_ref: str = ""                   # 对应自测指导的章节，便于逐条对账
    expect_non_200: bool = False        # 预期报错用例（如 G6）：非200是预期行为，
                                        # 豁免 run() 末尾的 HTTP 200 强制校验


@dataclass
class TestResult:
    meta: TestCaseMeta
    status: str = TestStatus.SKIP
    http_status: Optional[int] = None
    duration_ms: int = 0
    request_body: Optional[Dict] = None
    response_body: Optional[Any] = None     # 非流式用例的完整响应
    stream_chunks: List[SSEChunk] = field(default_factory=list)
    usage: Optional[Dict] = None            # 提取的usage（兼容流式/非流式）
    pass_rate: float = 0.0                  # 单case内部子断言通过率（多断言场景）
    total_assertions: int = 0
    passed_assertions: int = 0
    failures: List[str] = field(default_factory=list)    # 判定失败的原因
    notes: List[str] = field(default_factory=list)       # 诊断说明（不影响判定）
    sub_requests: List[SubRequest] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)  # 用例自定义结构化数据
    error_kind: str = ErrorKind.NONE
    error_trace: Optional[str] = None
    retried: bool = False                   # 是否发生过重试（稳定性信号）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add_failure(self, msg: str):
        self.failures.append(msg)

    def add_note(self, msg: str):
        """记录诊断信息。不参与 pass/fail 判定，报告里单独成列。"""
        self.notes.append(msg)

    def assert_true(self, condition: bool, msg: str) -> bool:
        self.total_assertions += 1
        if condition:
            self.passed_assertions += 1
            return True
        self.add_failure(f"✗ {msg}")
        return False

    def set_error_kind(self, kind: str):
        # 只记录第一个非空归类，避免被后续的连带失败覆盖
        if kind and not self.error_kind:
            self.error_kind = kind

    def absorb_notes(self, sub: TestResult, label: str = ""):
        """只并入诊断说明，不并入失败原因。

        用于多子项用例：子项的判定由调用方自己做，但子请求过程中产生的
        诊断（入参写法回退、重试等）仍需要保留到主结果里。
        """
        prefix = f"[{label}] " if label else ""
        for n in sub.notes:
            self.add_note(prefix + n)

    def absorb(self, sub: TestResult, label: str = ""):
        """把子请求的诊断信息并入主结果。

        此前各用例用一个临时 TestResult 承接 chat_request，chat_request 写入的
        401/403/429/5xx 详细诊断从未合并回主结果，报告里只剩一句笼统断言。
        """
        prefix = f"[{label}] " if label else ""
        for f in sub.failures:
            self.add_failure(prefix + f)
        for n in sub.notes:
            self.add_note(prefix + n)
        self.set_error_kind(sub.error_kind)
        if sub.retried:
            self.retried = True
        self.sub_requests.extend(sub.sub_requests)


@dataclass
class TestCase:
    meta: TestCaseMeta
    runner: Callable[["TestCase", TesterConfig, httpx.AsyncClient], Awaitable[TestResult]]

    async def run(self, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
        result = TestResult(meta=self.meta)
        # 判断豁免
        if self.meta.capability_key:
            cap = cfg.capability.model_dump()
            if not cap.get(self.meta.capability_key, False):
                result.status = TestStatus.WAIVE
                result.add_note(f"供应商未声明支持能力: {self.meta.capability_key}，按文档规则豁免本用例")
                return result
        start = time.perf_counter()
        try:
            result = await self.runner(self, cfg, client) or result
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_trace = traceback.format_exc()
            result.set_error_kind(ErrorKind.NETWORK)
            result.add_failure(f"执行异常: {type(e).__name__}: {e}")
        finally:
            result.duration_ms = int((time.perf_counter() - start) * 1000)
            # 计算通过率 & 最终状态（若状态仍为skip则根据断言判定）
            if result.total_assertions > 0:
                result.pass_rate = result.passed_assertions / result.total_assertions
            if result.status == TestStatus.SKIP:
                if result.total_assertions == 0:
                    result.status = TestStatus.PASS
                elif result.pass_rate == 1.0:
                    result.status = TestStatus.PASS
                else:
                    result.status = TestStatus.FAIL
            # HTTP 200强制校验（文档通用要求）
            # 多子请求用例的主 http_status 可能为空，此时以子请求为准。
            # 聚合类用例（如 F15A）自身不发 HTTP 请求，不适用本校验。
            made_request = bool(result.sub_requests) or result.http_status is not None
            effective_status = result.http_status
            if effective_status is None and result.sub_requests:
                # 优先取最后一次成功的状态；全失败时取最后一次的状态。
                ok = [s.http_status for s in result.sub_requests if s.http_status == 200]
                effective_status = ok[-1] if ok else result.sub_requests[-1].http_status
            # 多子请求用例此前主 http_status 一直是 None，报告里 HTTP 列显示为空，
            # 回传时看不出这些用例究竟打通没有。这里回填一个有代表性的值。
            if result.http_status is None and effective_status is not None:
                result.http_status = effective_status
            if (result.status == TestStatus.PASS and made_request
                    and effective_status != 200 and not self.meta.expect_non_200):
                result.status = TestStatus.FAIL
                result.add_failure(
                    f"HTTP状态码非200 (实际={effective_status}) - 违反文档 2.1 通用要求"
                )
            # 判定一致性：只要记录了失败原因就不能算通过。
            # 此前多子请求用例的结论取决于"最后一次子请求恰好是成功还是失败"——
            # 同样是「一个开关成功、另一个 504」，F17 判通过而 F18 判失败，
            # 仅因为失败的那次在前还是在后。
            if result.status == TestStatus.PASS and result.failures:
                result.status = TestStatus.FAIL
            if result.status == TestStatus.PASS and result.retried:
                result.add_note(
                    "⚠️ 本用例首次请求失败、重试后才通过。功能可用，但链路稳定性需供应商排查。"
                )
        return result


# -------------------- HTTP / SSE 工具 --------------------

def _classify_http(status: Optional[int]) -> str:
    if status is None:
        return ErrorKind.NETWORK
    if status == 200:
        return ErrorKind.NONE
    if status in (401, 403):
        return ErrorKind.AUTH
    if status == 413:
        return ErrorKind.TOO_LARGE
    if status == 429:
        return ErrorKind.RATE_LIMIT
    # 408 Request Timeout：多见于供应商拉取外部资源（图片/视频URL）超时。
    # 这是链路/取回问题，不是"请求参数不对"，归到 bad_request 会误导整改方向。
    if status == 408:
        return ErrorKind.GATEWAY
    if status in (502, 503, 504):
        return ErrorKind.GATEWAY
    if status >= 500:
        return ErrorKind.SERVER
    return ErrorKind.BAD_REQUEST


# 错误响应体里指向"供应商拉不到外部资源"的特征。
# 例：{"error":{"message":"timed out sending image request: https://... Timeout"}}
# 这类失败必须与"不具备多模态能力"分开——整改方向完全不同。
_REMOTE_FETCH_FAIL_SIGNS = (
    "timed out sending image request",
    "timed out sending video request",
    "failed to fetch", "failed to download",
    "error downloading", "download failed",
    "fetch image", "fetch video",
    "image request timeout", "timeout fetching",
    "无法下载", "无法获取", "拉取超时", "下载超时",
)


def error_body_indicates_fetch_failure(body: Any) -> str:
    """在错误响应体里查找"供应商拉不到远程资源"的证据，命中则返回原始片段。"""
    if not isinstance(body, dict):
        return ""
    try:
        blob = json.dumps(body, ensure_ascii=False)
    except Exception:
        blob = str(body)
    low = blob.lower()
    for sign in _REMOTE_FETCH_FAIL_SIGNS:
        if sign in low:
            idx = low.find(sign)
            return blob[max(0, idx - 40): idx + 160]
    return ""


async def chat_request(
    client: httpx.AsyncClient,
    cfg: TesterConfig,
    payload: Dict[str, Any],
    result: TestResult,
    collect_chunks: bool = True,
    label: str = "",
    retries: Optional[int] = None,
    expect_error: bool = False,
) -> Dict[str, Any] | List[SSEChunk]:
    """统一请求入口。

    非流式：返回完整响应 dict（同时写 result.response_body / http_status / usage）
    流式：返回 SSEChunk 列表（同时写 result.stream_chunks / http_status / 末包usage）

    无论成功失败，本次往返都会追加一条 SubRequest 到 result.sub_requests，
    包含完整请求体、响应体/分片、HTTP 状态、耗时、重试次数——这是文档 2.1
    "每个用例需回传"的落点。

    ⚠️ 各测试用例按需构造 payload，此处不自动清理参数（兼容性测试需要发送
    特定参数），但会统一处理重试、错误归类和留档。
    """
    max_retries = cfg.max_retries if retries is None else retries
    stream = bool(payload.get("stream", False))
    sub = SubRequest(label=label or ("stream" if stream else "request"),
                     request_body=payload)
    attempt = 0
    started_all = time.perf_counter()

    while True:
        attempt += 1
        t0 = time.perf_counter()
        try:
            if stream:
                outcome, status = await _do_stream(client, cfg, payload, sub, collect_chunks)
            else:
                outcome, status = await _do_plain(client, cfg, payload, sub)
            transient = status in RETRYABLE_STATUS
        except (httpx.TimeoutException, httpx.TransportError) as e:
            outcome, status = None, None
            sub.error_detail = f"{type(e).__name__}: {e}"
            transient = True
        sub.duration_ms = int((time.perf_counter() - t0) * 1000)
        sub.http_status = status
        sub.attempts = attempt
        GATEWAY_HEALTH.record(status)

        if not transient:
            break
        # 网关超时类单独限流：这类请求已经等满了网关超时预算，重试代价极高
        limit = (min(max_retries, MAX_GATEWAY_RETRIES)
                 if (status in GATEWAY_TIMEOUT_STATUS or status is None) else max_retries)
        if attempt > limit:
            break
        if not GATEWAY_HEALTH.allow_retry():
            result.add_note(
                f"{label or '请求'}: 供应商已连续 {GATEWAY_HEALTH.consecutive_gateway_failures} 次"
                f"网关超时/不可用，熔断生效，本次不再重试以免拖长整轮耗时。"
                f"这本身就是链路稳定性问题，请供应商排查。"
            )
            break
        result.retried = True
        result.add_note(
            f"{label or '请求'}: 第{attempt}次尝试失败"
            f"（HTTP={status or 'N/A'}{', ' + sub.error_detail if sub.error_detail else ''}），"
            f"退避后重试"
        )
        await asyncio.sleep(min(8.0, 1.5 * attempt))

    sub.error_kind = _classify_http(sub.http_status)
    result.sub_requests.append(sub)
    result.set_error_kind(sub.error_kind)

    # 把本次往返的结果同步到主 result（单请求用例直接可用）
    result.request_body = payload
    result.http_status = sub.http_status
    result.usage = sub.usage

    if sub.http_status is None:
        result.add_failure(
            f"🌐 请求未能完成（{sub.error_detail}）。已重试 {sub.attempts} 次仍失败，"
            f"请检查网络连通性、供应商端点可达性与超时配置。"
        )
        if stream:
            result.stream_chunks = []
            return []
        result.response_body = {"_transport_error": sub.error_detail}
        return result.response_body

    if stream:
        result.stream_chunks = _rehydrate_chunks(sub)
        return result.stream_chunks
    result.response_body = sub.response_body
    if expect_error and sub.http_status is not None and sub.http_status != 200:
        # 预期报错用例（如 G6 thinking 关闭应被拒绝）：非 200 是预期行为，
        # 不记为失败，仅留档说明，由用例自身的断言判定是否符合预期。
        result.add_note(
            f"[预期报错留档] HTTP {sub.http_status}: {str(sub.response_body)[:150]}"
        )
    else:
        _explain_http_error(result, sub)
    return sub.response_body if isinstance(sub.response_body, dict) else {}


def _rehydrate_chunks(sub: SubRequest) -> List[SSEChunk]:
    out: List[SSEChunk] = []
    for d in sub.stream_chunks:
        out.append(SSEChunk(**d))
    return out


async def _do_stream(
    client: httpx.AsyncClient, cfg: TesterConfig, payload: Dict[str, Any],
    sub: SubRequest, collect_chunks: bool,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    chunks: List[Dict[str, Any]] = []
    idx = 0
    last_usage: Optional[Dict] = None
    t0 = time.perf_counter()
    async with client.stream(
        "POST", cfg.endpoint, headers=headers, json=payload,
        timeout=httpx.Timeout(cfg.timeout_seconds, connect=30.0, pool=30.0),
    ) as resp:
        status = resp.status_code
        if status != 200:
            body_text = (await resp.aread()).decode("utf-8", "replace")
            sub.response_body = _try_json(body_text)
            sub.stream_chunks = []
            return [], status
        async for raw_line in resp.aiter_lines():
            line = raw_line.rstrip("\r")
            if not line:
                continue
            if line.startswith(":"):   # SSE 注释
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].lstrip()
            idx += 1
            chunk: Dict[str, Any] = {
                "index": idx, "raw": data, "event": "data",
                "delta_content": "", "reasoning_delta": "",
                "tool_calls_delta": [], "parsed_json": None,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            }
            if data == "[DONE]":
                if collect_chunks:
                    chunks.append(chunk)
                break
            try:
                obj = json.loads(data)
                chunk["parsed_json"] = obj
                if isinstance(obj, dict):
                    choices = obj.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta") or {}
                        chunk["delta_content"] = delta.get("content") or ""
                        if isinstance(delta.get("reasoning_content"), str):
                            chunk["reasoning_delta"] = delta["reasoning_content"]
                        for tc in (delta.get("tool_calls") or []):
                            if isinstance(tc, dict):
                                chunk["tool_calls_delta"].append(tc)
                    u = obj.get("usage")
                    if isinstance(u, dict) and any(
                        k in u for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    ):
                        last_usage = u
            except json.JSONDecodeError:
                pass
            if collect_chunks:
                chunks.append(chunk)
    sub.stream_chunks = chunks
    sub.usage = last_usage
    return chunks, status


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return {"_raw_text": text, "_parse_error": True}


async def _do_plain(
    client: httpx.AsyncClient, cfg: TesterConfig, payload: Dict[str, Any], sub: SubRequest,
) -> Tuple[Any, Optional[int]]:
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    resp = await client.post(
        cfg.endpoint, headers=headers, json=payload,
        timeout=httpx.Timeout(cfg.timeout_seconds, connect=30.0, pool=30.0),
    )
    body = _try_json(resp.text)
    sub.response_body = body
    if isinstance(body, dict) and isinstance(body.get("usage"), dict):
        sub.usage = body["usage"]
    return body, resp.status_code


def _explain_http_error(result: TestResult, sub: SubRequest) -> None:
    """把 HTTP 错误翻译成对供应商可执行的整改说明。"""
    status = sub.http_status
    if status == 200:
        return
    body = sub.response_body
    err_msg = ""
    if isinstance(body, dict):
        if isinstance(body.get("error"), dict):
            err_msg = body["error"].get("message", "") or body["error"].get("type", "")
        elif isinstance(body.get("error"), str):
            err_msg = body["error"]
        elif isinstance(body.get("message"), str):
            err_msg = body["message"]
        elif body.get("_parse_error"):
            raw = str(body.get("_raw_text", ""))
            err_msg = raw[:150]
    hint = {
        ErrorKind.AUTH: "请检查 API Key 是否正确、是否过期、是否有调用该模型的权限。",
        ErrorKind.RATE_LIMIT: "请降低并发或检查配额。压测阶段可适当下调到达率。",
        ErrorKind.GATEWAY: "供应商网关超时或后端不可用。请排查该类请求的处理链路与超时阈值。",
        ErrorKind.SERVER: "供应商后端异常。",
        ErrorKind.TOO_LARGE: "请求体超过网关体积限制。请调大网关上限，或确认多模态入参的体积约束。",
        ErrorKind.BAD_REQUEST: "供应商拒绝了该请求参数。若该参数本就不支持，属预期；否则需修复参数解析。",
    }.get(sub.error_kind, "")
    result.add_failure(f"HTTP {status}（{sub.error_kind}）：{err_msg[:200]} {hint}".strip())


# -------------------- 流式质量分析 --------------------

def analyze_stream_quality(chunks: List[SSEChunk]) -> Dict[str, Any]:
    """识别"伪流式"：响应被网关整段缓冲后一次性吐出。

    这种情况下 latency-ttft ≈ 0，TPOT/ITL 会算出 0.01ms 这种物理上不可能的值，
    如果直接跟基线比会得到"✅ 优秀"的假绿灯。必须单独识别出来。
    """
    content_chunks = [c for c in chunks if c.delta_content or c.reasoning_delta]
    n = len(content_chunks)
    if n == 0:
        return {"content_chunks": 0, "degenerate": True,
                "reason": "无任何内容增量分片"}
    first_ms = content_chunks[0].elapsed_ms
    last_ms = content_chunks[-1].elapsed_ms
    span_ms = max(0, last_ms - first_ms)
    total_text = sum(len(c.delta_content) + len(c.reasoning_delta) for c in content_chunks)
    degenerate = False
    reason = ""
    if n < 3:
        degenerate = True
        reason = f"内容分片仅 {n} 片，未体现增量输出"
    elif span_ms < 50 and total_text > 50:
        degenerate = True
        reason = (f"{total_text} 字符在 {span_ms}ms 内一次性到达，"
                  f"疑似响应被整段缓冲后吐出（非真正增量流式）")
    return {
        "content_chunks": n,
        "span_ms": span_ms,
        "total_chars": total_text,
        "degenerate": degenerate,
        "reason": reason,
    }


# -------------------- 协议一致性校验 --------------------

REQUIRED_RESP_FIELDS = {
    "id": (str,),
    "object": (str,),      # chat.completion 或 chat.completion.chunk
    "created": (int, float),
    "model": (str,),
    "choices": (list,),
}


def validate_openai_schema(result: TestResult, body: Dict[str, Any], is_chunk: bool = False) -> None:
    """验证响应是否符合 OpenAI ChatCompletions 结构。文档2.1协议一致性要求。"""
    if not isinstance(body, dict):
        result.assert_true(False, f"响应体不是dict，实际type={type(body).__name__}")
        return
    # 传输失败 / 非JSON响应 / HTTP错误：原因已在 chat_request 中记录，跳过字段检查
    if body.get("_parse_error") or body.get("_transport_error"):
        return
    if result.http_status is not None and result.http_status != 200:
        return
    # 检测 error 字段（有些供应商返回200但带error）
    if "error" in body and "choices" not in body:
        err = body["error"]
        err_msg = err.get("message", "") or err.get("type", "") if isinstance(err, dict) else str(err)
        result.set_error_kind(ErrorKind.PROTOCOL)
        result.assert_true(
            False,
            f"⚠️ 供应商返回错误响应（非标准ChatCompletions格式）："
            f"error={err_msg[:150]}。响应包含error字段但缺少choices。"
        )
        return
    for k, types in REQUIRED_RESP_FIELDS.items():
        if k not in body:
            result.set_error_kind(ErrorKind.PROTOCOL)
            result.assert_true(False, f"响应缺少顶层必填字段: {k}")
            continue
        if not isinstance(body[k], types):
            result.set_error_kind(ErrorKind.PROTOCOL)
            result.assert_true(
                False,
                f"字段 {k} 类型错误: 期望{[t.__name__ for t in types]}，实际{type(body[k]).__name__}"
            )
    choices = body.get("choices") or []
    if not isinstance(choices, list) or len(choices) == 0:
        if is_chunk:
            # 流式 chunk 允许 choices 为空（首包只有 role、或末包只有 usage）
            return
        result.set_error_kind(ErrorKind.PROTOCOL)
        result.assert_true(False, "choices为空或非list")
        return
    first = choices[0]
    if not isinstance(first, dict):
        result.assert_true(False, f"choices[0]非dict: {type(first).__name__}")
        return
    if "index" not in first:
        result.set_error_kind(ErrorKind.PROTOCOL)
        result.assert_true(False, "choices[0]缺少字段: index")
    if is_chunk:
        if "delta" not in first:
            result.set_error_kind(ErrorKind.PROTOCOL)
            result.assert_true(False, "流式chunk的choices[0]缺少delta字段")
        return

    if "message" not in first:
        result.set_error_kind(ErrorKind.PROTOCOL)
        result.assert_true(False, "非流式响应choices[0]缺少message字段")
        return
    msg = first["message"]
    if not isinstance(msg, dict):
        result.assert_true(False, f"message非dict: {type(msg).__name__}")
        return
    if "role" not in msg:
        result.set_error_kind(ErrorKind.PROTOCOL)
        result.assert_true(False, "message缺少role字段")
    # 协议一致性增强诊断：正式回答必须写在 message.content 字段
    content = msg.get("content")
    content_is_str_empty = isinstance(content, str) and len(content.strip()) == 0
    has_tool_calls = isinstance(msg.get("tool_calls"), list) and len(msg.get("tool_calls", [])) > 0
    if content_is_str_empty and not has_tool_calls:
        elsewhere: List[str] = []
        for cand in ("reasoning_content", "thinking", "thought", "provider_specific_fields"):
            v = msg.get(cand)
            if isinstance(v, str) and len(v.strip()) > 10:
                elsewhere.append(f"{cand}(len={len(v)})")
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, str) and len(sub_v.strip()) > 10:
                        elsewhere.append(f"{cand}.{sub_k}(len={len(sub_v)})")
        result.set_error_kind(ErrorKind.PROTOCOL)
        if elsewhere:
            result.assert_true(
                False,
                "📋 协议一致性偏移：message.content 为空但实际输出写在非标准字段 "
                f"{', '.join(elsewhere)}。OpenAI 规范要求正式回答文本必须放在 message.content，"
                "推理过程放 reasoning_content。请供应商修复响应字段映射。"
            )
        else:
            usage = body.get("usage") or {}
            ct = 0
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens_details"), dict):
                ct = usage["completion_tokens_details"].get("text_tokens", 0)
            result.assert_true(
                False,
                f"📋 message.content 为空字符串，且无其他字段承载输出。"
                f"usage.completion_tokens_details.text_tokens={ct}。"
            )


# -------------------- 通用提取工具 --------------------

def get_message(body: Any) -> Dict[str, Any]:
    try:
        msg = body["choices"][0]["message"]
        return msg if isinstance(msg, dict) else {}
    except Exception:
        return {}


def get_content_text(body: Any) -> str:
    msg = get_message(body)
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


THINKING_FIELD_CANDIDATES = ("reasoning_content", "reasoning", "thinking", "thought")


def get_reasoning_text(body: Any, chunks: Optional[List[SSEChunk]] = None) -> str:
    """提取思考内容，兼容流式/非流式与 <think> 内联写法。"""
    if chunks:
        return "".join(c.reasoning_delta for c in chunks if c.reasoning_delta)
    msg = get_message(body)
    for cand in THINKING_FIELD_CANDIDATES:
        v = msg.get(cand)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            for sub_v in v.values():
                if isinstance(sub_v, str) and sub_v.strip():
                    return sub_v
    c = msg.get("content")
    if isinstance(c, str) and "<think>" in c and "</think>" in c:
        s = c.index("<think>") + len("<think>")
        e = c.index("</think>")
        return c[s:e].strip()
    return ""


def strip_think_block(text: str) -> str:
    """去掉 content 里内联的 <think>…</think>，只留正式回答。"""
    if not isinstance(text, str) or "<think>" not in text:
        return text or ""
    out = text
    while "<think>" in out and "</think>" in out:
        s = out.index("<think>")
        e = out.index("</think>") + len("</think>")
        out = out[:s] + out[e:]
    return out.strip()


# -------------------- 套件执行 --------------------

async def run_test_suite(
    cases: List[TestCase],
    cfg: Optional[TesterConfig] = None,
    progress_cb: Optional[Callable[[int, int, TestResult], Awaitable[None]]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[TestResult]:
    cfg = cfg or get_config()
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=8)
    timeout = httpx.Timeout(cfg.timeout_seconds, connect=30.0, pool=30.0)
    results: List[TestResult] = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for i, case in enumerate(cases):
            if should_cancel and should_cancel():
                r = TestResult(meta=case.meta, status=TestStatus.SKIP)
                r.add_note("用户取消了本次执行")
                results.append(r)
                continue
            r = await case.run(cfg, client)
            results.append(r)
            if progress_cb:
                try:
                    await progress_cb(i + 1, len(cases), r)
                except Exception:
                    pass
            # 温和的请求间隔，避免触发网关限流
            await asyncio.sleep(0.3)
    return results
