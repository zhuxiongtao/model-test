"""性能压测引擎（多轮长上下文会话模拟 + 到达率爬坡/稳态调度）

等价于 MTBBenchmark 的轻量实现：
- 多轮长上下文对话模拟（平均32k输入/300输出）
- 会话到达率匀速爬坡，达到目标后进入稳态
- 指标：吞吐/TTFT/Latency/TPOT/ITL/缓存命中率（avg/p50/p75/p90/p95/p99）

本版相对初版的关键修正
----------------------
1. **缓存命中率改读真实字段**。初版用 prompt_tokens 的"理想值-实际值"估算，
   但 prefix cache 命中并不会减少 prompt_tokens，算出来是噪声（实测各轮
   0%~10% 随机跳动，与文档"Round0≈0、多轮后显著上升"的形态完全不符）。
   现改读 usage.prompt_tokens_details.cached_tokens 等真实回传字段；
   供应商完全不回传时，明确标记为"无法测量"而非给 0 并判 FAIL。

2. **区分 offered load 与 achieved throughput**。初版把
   total_requests/duration 当作吞吐去比基线，而这个值受发压速率上限
   (arrival_rate_end=1.0) 封顶，测的是工具的发压节奏而非供应商的服务能力。

3. **稳态窗口统计**。文档基线本就是稳态口径。初版把爬坡期和收尾期一起算进
   分位数，且爬坡结束后队列未排空会空转到 2× 硬 deadline，把 duration 拉长、
   吞吐稀释。现改为按稳态窗口单独统计，并修正调度退出条件。

4. **轮间间隔真实生效**。初版 turn_interval_* 五个配置项一次都没被读取，
   轮间间隔参数等于没实现。

5. **识别伪流式**。响应被网关整段缓冲时 latency-ttft≈0，TPOT 会算出 0.01ms
   这种物理上不可能的值并被判为"优于基线"。现单独识别并剔除。

6. **任务持引用**。初版 asyncio.create_task 返回值直接丢弃，事件循环只持
   弱引用，长任务存在被 GC 的风险。

7. **每会话唯一前缀**。初版所有会话共用同一段重复中文填充，跨会话前缀完全
   相同，会人为抬高前缀缓存命中率。
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from config import BenchmarkConfig, TesterConfig, get_config


def _fmean(values):
    """statistics.fmean 的兼容版兜底"""
    if not values:
        return 0.0
    try:
        return statistics.fmean(values)
    except AttributeError:
        return sum(values) / len(values)


# 汉字数 / token。实测校准（20260907_0728a00e 轮次）：初版按 0.7 估算，
# 但 GLM 分词器对重复文本压缩率很高，目标 2000 token 实际只生成了 ~820 token
# （实测 2.17 字/token）。按 2.2 生成才能逼近声明的 token 目标，
# 否则压测负载远小于报告口径，费用与 TTFT 数据失真。
PAD_CHARS_PER_TOKEN = 2.2

_PAD_BASE = (
    "人工智能自然语言处理技术广泛应用于各行各业，帮助用户提升工作效率，"
    "促进信息传播与知识获取，推动数字化转型与智能升级。"
)


def _build_chinese_padding(target_tokens: int, seed_text: str = "") -> str:
    """按目标token数量生成近似长度的中文文本。

    seed_text 用于给每个会话生成**唯一前缀**：初版所有会话共用同一段重复
    文本，跨会话前缀完全相同，供应商的前缀缓存会跨会话命中，把缓存命中率
    抬高到不真实的水平。真实场景（ShareGPT 类多轮对话）里各会话内容互不相同。
    """
    char_count = max(4, int(target_tokens * PAD_CHARS_PER_TOKEN))
    body = (_PAD_BASE * (char_count // len(_PAD_BASE) + 1))[:char_count]
    if seed_text:
        # 唯一前缀放在最前面，确保不同会话从第一个 token 就分叉
        return (seed_text + body)[:max(char_count, len(seed_text))]
    return body


# -------------------- 缓存命中率：真实字段提取 --------------------

def extract_cached_tokens(usage: Optional[Dict[str, Any]]) -> Optional[int]:
    """从 usage 中提取"命中缓存的 prompt token 数"。

    按优先级尝试各家的字段写法；**一个都没有时返回 None**（表示无法测量），
    而不是返回 0——"供应商没回传这个字段"和"缓存命中率是0"是两回事，
    前者是需要补实现的整改项，后者才是性能问题。
    """
    if not isinstance(usage, dict):
        return None
    # OpenAI 现行规范
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        for k in ("cached_tokens", "cache_read_input_tokens"):
            v = details.get(k)
            if isinstance(v, int) and v >= 0:
                return v
    # 各供应商的顶层写法
    for k in ("cached_tokens", "prompt_cache_hit_tokens",
              "cache_read_input_tokens", "prompt_cache_hit_token_count"):
        v = usage.get(k)
        if isinstance(v, int) and v >= 0:
            return v
    return None


@dataclass
class TurnMetrics:
    """单轮请求指标（流式）"""
    session_id: int
    round_idx: int               # 0-based
    http_status: int
    ttft_ms: int                 # 首token时间
    latency_ms: int              # 端到端总时长
    input_tokens: int = 0        # prompt_tokens
    output_tokens: int = 0       # completion_tokens
    reasoning_tokens: int = 0
    cached_tokens: Optional[int] = None   # None = 供应商未回传该字段
    itl_ms_list: List[int] = field(default_factory=list)  # token间延迟
    content_chunks: int = 0      # 含内容增量的分片数
    stream_span_ms: int = 0      # 首个到最后一个内容分片的跨度
    degenerate_stream: bool = False  # 疑似整段缓冲的伪流式
    started_at: float = 0.0      # 相对压测开始的秒数，用于稳态窗口切分
    error: Optional[str] = None

    @property
    def tpot_ms(self) -> float:
        """每输出token耗时 = (总延迟-TTFT) / 输出token数"""
        if self.output_tokens <= 0:
            return 0.0
        net_ms = max(0, self.latency_ms - self.ttft_ms)
        return net_ms / self.output_tokens


@dataclass
class BenchmarkResult:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    duration_seconds: float = 0.0
    # 原始指标
    turns: List[TurnMetrics] = field(default_factory=list)
    # 吞吐：offered = 计划发压速率，achieved = 实际完成速率
    offered_load_req_per_s: float = 0.0
    throughput_req_per_s: float = 0.0            # achieved，全程口径
    steady_throughput_req_per_s: float = 0.0     # achieved，稳态窗口口径
    throughput_input_tok_per_s: float = 0.0
    throughput_output_tok_per_s: float = 0.0
    saturated: bool = False                      # 是否压到饱和
    ran_out_of_work: bool = False                # 计划轮次提前跑完（≠饱和）
    steady_metrics_valid: bool = True            # 稳态窗口是否有样本
    # 分位数（稳态窗口口径）
    ttft_ms: Dict[str, int] = field(default_factory=dict)
    latency_ms: Dict[str, int] = field(default_factory=dict)
    tpot_ms: Dict[str, float] = field(default_factory=dict)
    itl_ms: Dict[str, float] = field(default_factory=dict)
    # 缓存
    cache_measurable: bool = False
    cache_hit_rate_overall: Optional[float] = None
    cache_hit_rate_by_round: Dict[int, float] = field(default_factory=dict)
    cache_note: str = ""
    # 流式质量
    degenerate_stream_count: int = 0
    stream_quality_note: str = ""
    # 执行完整度
    planned_turns: int = 0
    completed_turns: int = 0
    steady_window: Tuple[float, float] = (0.0, 0.0)
    warnings: List[str] = field(default_factory=list)
    # 参考基线对比
    baseline_comparison: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _percentiles(values: List[float]) -> Dict[str, float]:
    """返回 avg/p50/p75/p90/p95/p99"""
    if not values:
        return {k: 0 for k in ("avg", "p50", "p75", "p90", "p95", "p99")}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return s[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] + (s[c] - s[f]) * (k - f)

    return {
        "avg": _fmean(s),
        "p50": pct(0.50), "p75": pct(0.75),
        "p90": pct(0.90), "p95": pct(0.95), "p99": pct(0.99),
    }


def _to_int_dict(pct: Dict[str, float]) -> Dict[str, int]:
    return {k: int(round(v)) for k, v in pct.items()}


def _to_float_dict(pct: Dict[str, float], digits: int = 2) -> Dict[str, float]:
    return {k: round(v, digits) for k, v in pct.items()}


# -------------------- 单轮请求（流式） --------------------

async def _stream_chat_turn(
    client: httpx.AsyncClient,
    cfg: TesterConfig,
    messages: List[Dict[str, Any]],
    max_output_tokens: int,
) -> TurnMetrics:
    tm = TurnMetrics(session_id=0, round_idx=0, http_status=0, ttft_ms=0, latency_ms=0)
    # GLM-5.3 始终思考。压测若不控制 effort，思考 token 会吃掉输出预算：
    # 1) 正文为空/分片极少 → 被误判"伪流式"；
    # 2) TTFT 里大头在等思考，测不出输出性能。
    # 压测目的是测服务性能而非智力，统一 reasoning_effort=low，
    # 并给 completion 预算留思考余量（思考输出约占 1~2 倍）。
    payload: Dict[str, Any] = {
        "model": cfg.model_id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "max_completion_tokens": max(64, max_output_tokens * 3),
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    start = time.perf_counter()
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    usage: Optional[Dict[str, Any]] = None
    content_chunks = 0
    try:
        async with client.stream(
            "POST", cfg.endpoint, headers=headers, json=payload,
            timeout=httpx.Timeout(cfg.timeout_seconds, connect=30.0, pool=30.0),
        ) as resp:
            tm.http_status = resp.status_code
            if resp.status_code != 200:
                await resp.aread()
            else:
                async for raw_line in resp.aiter_lines():
                    line = raw_line.rstrip("\r")
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].lstrip()
                    now = time.perf_counter()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    try:
                        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                    except Exception:
                        delta = {}
                    has_delta = bool(delta.get("content") or delta.get("reasoning_content"))
                    if has_delta:
                        content_chunks += 1
                        if first_token_ts is None:
                            first_token_ts = now
                        elif last_token_ts is not None:
                            tm.itl_ms_list.append(int((now - last_token_ts) * 1000))
                        last_token_ts = now
                    u = obj.get("usage")
                    if isinstance(u, dict):
                        usage = u
        end = time.perf_counter()
        tm.latency_ms = int((end - start) * 1000)
        tm.content_chunks = content_chunks
        if first_token_ts is not None:
            tm.ttft_ms = int((first_token_ts - start) * 1000)
            if last_token_ts is not None:
                tm.stream_span_ms = int((last_token_ts - first_token_ts) * 1000)
        else:
            # 未识别到任何内容增量：TTFT 不可用，标记为 0 并在聚合时剔除
            tm.ttft_ms = 0
        if usage:
            tm.input_tokens = int(usage.get("prompt_tokens", 0) or 0)
            tm.output_tokens = int(usage.get("completion_tokens", 0) or 0)
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                tm.reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
            else:
                tm.reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
            tm.cached_tokens = extract_cached_tokens(usage)
        else:
            tm.output_tokens = max(1, content_chunks)

        # 伪流式识别：内容分片过少，或全部内容在极短时间内一次性到达。
        # 这类响应会让 TPOT/ITL 算出 0.0x ms 的不可能值并"优于基线"。
        if tm.http_status == 200:
            if content_chunks < 3:
                tm.degenerate_stream = True
            elif tm.stream_span_ms < 50 and tm.output_tokens > 50:
                tm.degenerate_stream = True
    except Exception as e:
        tm.error = f"{type(e).__name__}: {e}"
        tm.http_status = tm.http_status or 0
        end = time.perf_counter()
        tm.latency_ms = int((end - start) * 1000)
    return tm


# -------------------- 会话模拟 --------------------

def _sample_distribution(avg: float, p50: float, p75: float, p90: float, p95: float) -> float:
    """基于给定分位数的分段近似采样。"""
    r = random.random()
    if r < 0.50:
        return random.uniform(max(1, p50 * 0.5), p50)
    if r < 0.75:
        return random.uniform(p50, p75)
    if r < 0.90:
        return random.uniform(p75, p90)
    if r < 0.95:
        return random.uniform(p90, p95)
    return random.uniform(p95, p95 * 1.3)


def _sample_turn_interval(bc: BenchmarkConfig) -> float:
    """轮间间隔采样。初版完全没用到这组参数。"""
    return max(0.0, _sample_distribution(
        bc.turn_interval_avg, bc.turn_interval_p50, bc.turn_interval_p75,
        bc.turn_interval_p90, bc.turn_interval_p95,
    ))


def _build_session_rounds(bc: BenchmarkConfig) -> Tuple[int, int, List[int], List[int]]:
    """返回 (num_rounds, init_prompt_tokens, input_len_each_round, output_len_each_round)"""
    num_rounds = max(1, int(round(_sample_distribution(
        bc.num_rounds_avg, bc.num_rounds_p50, bc.num_rounds_p75,
        bc.num_rounds_p90, bc.num_rounds_p95,
    ))))
    init_len = int(_sample_distribution(
        bc.init_prompt_length_avg, bc.init_prompt_length_p50,
        bc.init_prompt_length_avg, bc.init_prompt_length_avg, bc.init_prompt_length_p95,
    ))
    input_lens: List[int] = []
    output_lens: List[int] = []
    for _ in range(num_rounds):
        input_lens.append(int(_sample_distribution(
            bc.input_length_avg, bc.input_length_p50,
            bc.input_length_avg, bc.input_length_avg, bc.input_length_p95,
        )))
        output_lens.append(int(_sample_distribution(
            bc.output_length_avg, bc.output_length_p50,
            bc.output_length_avg, bc.output_length_avg, bc.output_length_p95,
        )))
    return num_rounds, init_len, input_lens, output_lens


# -------------------- 调度器 --------------------

async def run_benchmark(
    cfg: Optional[TesterConfig] = None,
    progress_cb: Optional[Callable[[str, int, int, Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> BenchmarkResult:
    cfg = cfg or get_config()
    bc = cfg.benchmark
    result = BenchmarkResult()
    start_wall = time.perf_counter()

    limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
    timeout = httpx.Timeout(120.0, connect=30.0, pool=30.0)

    # 每个会话独立的对话蓝图
    sessions: List[Dict[str, Any]] = []
    for sid in range(bc.total_sessions):
        num_rounds, init_len, input_lens, output_lens = _build_session_rounds(bc)
        # 唯一前缀：确保各会话的 prefix 从第一个 token 就分叉，
        # 否则跨会话前缀缓存会把命中率抬到不真实的水平
        unique_prefix = f"[会话{sid}-{uuid.uuid4().hex[:12]}] "
        sessions.append({
            "sid": sid,
            "num_rounds": num_rounds,
            "input_lens": input_lens,
            "output_lens": output_lens,
            "round_idx": 0,
            "next_available_at": 0.0,     # 轮间间隔控制
            "messages": [{
                "role": "system",
                "content": "你是长上下文对话助手。以下是需要参考的背景材料：\n"
                           + _build_chinese_padding(init_len, unique_prefix),
            }],
        })

    total_duration = bc.ramp_duration_seconds + bc.steady_duration_seconds
    result.planned_turns = sum(s["num_rounds"] for s in sessions)

    def arrival_rate(t: float) -> float:
        if t < bc.ramp_duration_seconds:
            ratio = t / max(1e-6, bc.ramp_duration_seconds)
            return bc.arrival_rate_start + (bc.arrival_rate_end - bc.arrival_rate_start) * ratio
        return bc.arrival_rate_end

    sem = asyncio.Semaphore(64)
    all_turns: List[TurnMetrics] = []
    # 持引用：asyncio 只对运行中的 task 持弱引用，丢掉引用可能被 GC 掉
    inflight: set[asyncio.Task] = set()
    completed = 0
    dispatched_total = 0
    bench_t0 = time.perf_counter()

    async def _do_one_turn(s: Dict[str, Any], client: httpx.AsyncClient):
        nonlocal completed
        r = s["round_idx"]
        if r >= s["num_rounds"]:
            return
        s["round_idx"] = r + 1   # 先占位，避免同一会话被重复派发
        inp_pad = _build_chinese_padding(s["input_lens"][r])
        s["messages"].append({
            "role": "user",
            "content": inp_pad + "\n\n请基于以上内容继续对话，用中文回答。",
        })
        started_at = time.perf_counter() - bench_t0
        async with sem:
            tm = await _stream_chat_turn(client, cfg, s["messages"], s["output_lens"][r])
        tm.session_id = s["sid"]
        tm.round_idx = r
        tm.started_at = started_at
        all_turns.append(tm)
        # 追加模型回复，保证后续轮次的 prefix 稳定可缓存
        s["messages"].append({
            "role": "assistant",
            "content": _build_chinese_padding(max(32, s["output_lens"][r] // 2)),
        })
        # 轮间间隔（模拟真实用户的思考/输入停顿）
        s["next_available_at"] = time.perf_counter() + _sample_turn_interval(bc)
        completed += 1
        if progress_cb:
            try:
                progress_cb("benchmark", completed, result.planned_turns, {
                    "sid": s["sid"], "round": r,
                    "ttft": tm.ttft_ms, "latency": tm.latency_ms,
                    "http": tm.http_status, "error": tm.error,
                })
            except Exception:
                pass

    async def scheduler(client: httpx.AsyncClient):
        nonlocal dispatched_total
        bucket = 0.0
        last = time.perf_counter()
        # 硬上限只作为兜底；正常路径由"计划时长到点"或"轮次全部完成"退出
        wall_deadline = time.perf_counter() + max(300.0, total_duration * 1.5)
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now
            elapsed = now - bench_t0

            if should_cancel and should_cancel():
                result.warnings.append("用户取消了压测，指标基于已完成的请求计算")
                break
            if now > wall_deadline:
                result.warnings.append(
                    f"压测达到硬性墙钟上限（{int(max(300.0, total_duration * 1.5))}s）后退出"
                )
                break
            # 计划时长到点即停止发压。初版在此之后 rate=0 但队列非空，
            # 退出条件永不成立，只能空转到 2× deadline，把 duration 拉长、吞吐稀释。
            if elapsed >= total_duration:
                break
            if all(s["round_idx"] >= s["num_rounds"] for s in sessions):
                break

            bucket += arrival_rate(elapsed) * dt
            dispatched = 0
            while bucket >= 1.0:
                # 只挑"还有轮次 且 已过轮间间隔"的会话
                ready = [
                    s for s in sessions
                    if s["round_idx"] < s["num_rounds"] and s["next_available_at"] <= now
                ]
                if not ready:
                    break
                s = random.choice(ready)
                task = asyncio.create_task(_do_one_turn(s, client))
                inflight.add(task)
                task.add_done_callback(inflight.discard)
                bucket -= 1.0
                dispatched_total += 1
                dispatched += 1
                if dispatched >= 16:
                    break
            # bucket 不无限累积，避免长时间无 ready 会话后突然爆发
            bucket = min(bucket, 16.0)
            await asyncio.sleep(0.05)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        await scheduler(client)
        # 等待所有 in-flight 请求收尾（有上限，避免个别请求卡死拖垮整轮）
        if inflight:
            done, pending = await asyncio.wait(set(inflight), timeout=180.0)
            for t in pending:
                t.cancel()
            if pending:
                result.warnings.append(f"{len(pending)} 个请求超过收尾等待时间被取消")

    end_wall = time.perf_counter()
    result.duration_seconds = round(end_wall - start_wall, 2)
    result.turns = all_turns
    result.total_requests = len(all_turns)
    result.completed_turns = len(all_turns)
    good = [t for t in all_turns if t.http_status == 200 and t.error is None]
    result.successful_requests = len(good)
    result.failed_requests = result.total_requests - result.successful_requests

    _aggregate(result, good, bc, total_duration)
    _compare_baseline(result)
    return result


# -------------------- 聚合 --------------------

def _aggregate(result: BenchmarkResult, good: List[TurnMetrics],
               bc: BenchmarkConfig, total_duration: float) -> None:
    # ---- 稳态窗口：爬坡结束 → 计划时长结束 ----
    steady_start = float(bc.ramp_duration_seconds)
    steady_end = float(total_duration)
    result.steady_window = (steady_start, steady_end)
    steady = [t for t in good if steady_start <= t.started_at <= steady_end]
    steady_valid = bool(steady)
    if not steady_valid:
        # 压测被提前结束或全程都在爬坡期，退化为全程口径并说明
        steady = good
        result.warnings.append(
            "稳态窗口内没有样本：压测在爬坡阶段就结束了"
            f"（实际运行 {result.duration_seconds:.0f}s，爬坡需 {steady_start:.0f}s）。"
            "常见原因是计划轮次太少、所有会话提前跑完。"
            "分位数指标已退化为全程口径，与文档基线的稳态口径不完全可比；"
            "请调大 total_sessions 或轮次分布，使压测能覆盖完整的爬坡+稳态窗口。"
        )

    # ---- 吞吐 ----
    # offered load = 计划发压速率的时间平均，用于说明"压了多大"
    result.offered_load_req_per_s = round(
        (bc.arrival_rate_start + bc.arrival_rate_end) / 2
        * bc.ramp_duration_seconds / max(1e-6, total_duration)
        + bc.arrival_rate_end * bc.steady_duration_seconds / max(1e-6, total_duration),
        4,
    )
    result.throughput_req_per_s = round(
        result.total_requests / max(1e-6, result.duration_seconds), 4)
    # ⚠️ 分子分母必须同口径。此前无论稳态窗口是否有样本，都用
    # len(steady) / (steady_end - steady_start) 计算——稳态窗口为空时
    # steady 退化成了全部样本，于是变成「全程请求数 ÷ 稳态窗口时长」，
    # 算出 123/300=0.41 这种既不是全程(0.289)也不是稳态的数字，还拿去比基线。
    if steady_valid:
        steady_span = max(1e-6, steady_end - steady_start)
        result.steady_throughput_req_per_s = round(len(steady) / steady_span, 4)
    else:
        result.steady_throughput_req_per_s = result.throughput_req_per_s
    result.steady_metrics_valid = steady_valid
    total_input = sum(t.input_tokens for t in good)
    total_output = sum(t.output_tokens for t in good)
    result.throughput_input_tok_per_s = round(total_input / max(1e-6, result.duration_seconds), 2)
    result.throughput_output_tok_per_s = round(total_output / max(1e-6, result.duration_seconds), 2)

    # 是否压到饱和：实际完成速率显著低于计划发压速率，说明供应商已成为瓶颈。
    # 但"计划轮次提前跑完"导致的提前结束不是饱和，只是没活干了。
    ran_out_of_work = result.completed_turns >= result.planned_turns
    result.ran_out_of_work = ran_out_of_work
    if ran_out_of_work and not steady_valid:
        result.saturated = False
        result.warnings.append(
            f"所有计划轮次（{result.planned_turns}）在计划时长内提前跑完，"
            f"压测并未持续施压到稳态。此时吞吐数字反映的是「活干完了」，"
            f"既不能证明供应商达到了能力上限，也不适合直接与基线比较。"
        )
    else:
        result.saturated = (
            result.steady_throughput_req_per_s < bc.arrival_rate_end * 0.85
            if bc.arrival_rate_end > 0 else False
        )
        if not result.saturated:
            result.warnings.append(
                f"实际完成速率({result.steady_throughput_req_per_s} req/s)接近计划发压速率"
                f"({bc.arrival_rate_end} req/s)，说明**尚未压到供应商的能力上限**。"
                f"此时吞吐数字反映的是发压节奏而非服务能力上限，"
                f"如需测量真实容量请调高 arrival_rate_end 直到出现排队。"
            )

    # ---- 流式质量 ----
    degen = [t for t in good if t.degenerate_stream]
    result.degenerate_stream_count = len(degen)
    healthy = [t for t in steady if not t.degenerate_stream]
    if degen:
        result.stream_quality_note = (
            f"{len(degen)}/{len(good)} 个请求疑似非增量流式（响应被整段缓冲后一次性返回）。"
            f"这类请求的 TPOT/ITL 会趋近 0，不具备参考意义，已从分位数统计中剔除。"
            f"请供应商排查网关是否对 SSE 响应做了缓冲。"
        )
        result.warnings.append(result.stream_quality_note)
    if not healthy:
        healthy = steady
        if degen:
            result.warnings.append(
                "所有请求都被判定为非增量流式，TPOT/ITL 无有效样本，"
                "以下数值仅供参考，不能用于与基线比较。"
            )

    # ---- 分位数（稳态 + 剔除伪流式）----
    ttft_vals = [float(t.ttft_ms) for t in healthy if t.ttft_ms > 0]
    lat_vals = [float(t.latency_ms) for t in healthy if t.latency_ms > 0]
    tpot_vals = [t.tpot_ms for t in healthy if t.output_tokens > 0 and t.tpot_ms > 0]
    itl_vals = [float(v) for t in healthy for v in t.itl_ms_list]
    result.ttft_ms = _to_int_dict(_percentiles(ttft_vals))
    result.latency_ms = _to_int_dict(_percentiles(lat_vals))
    result.tpot_ms = _to_float_dict(_percentiles(tpot_vals), 2)
    result.itl_ms = _to_float_dict(_percentiles(itl_vals), 2)

    # ITL 双峰识别：大量 0ms 间隔 + 少量很大的间隔，说明供应商是"成批吐 token"
    # （一个 SSE 分片里塞多个 token，分片之间再等一段）。这时 ITL 的中位数会是 0，
    # 直接看分位数会误以为"token 间几乎无延迟"。
    if itl_vals:
        zero_ratio = sum(1 for v in itl_vals if v <= 0.5) / len(itl_vals)
        if zero_ratio >= 0.4 and result.itl_ms.get("p90", 0) > 50:
            result.warnings.append(
                f"ITL 呈双峰分布：{zero_ratio*100:.0f}% 的 token 间隔≈0ms，"
                f"但 P90 达 {result.itl_ms.get('p90')}ms。这说明供应商是成批下发 token"
                f"（单个 SSE 分片内含多个 token，分片之间再等待），"
                f"而非逐 token 流式。ITL 的中位数因此失去参考意义，"
                f"请以 TPOT（{result.tpot_ms.get('p50')}ms）为准评估输出速度。"
            )

    # ---- 缓存命中率：只用供应商真实回传的字段 ----
    measurable = [t for t in good if t.cached_tokens is not None]
    if not measurable:
        result.cache_measurable = False
        result.cache_hit_rate_overall = None
        result.cache_note = (
            "❗无法测量：服务方未在 usage 中回传缓存命中字段。"
            "已尝试的字段：usage.prompt_tokens_details.cached_tokens（OpenAI 规范）、"
            "usage.cached_tokens、usage.prompt_cache_hit_tokens、"
            "usage.cache_read_input_tokens。"
            "如需验收缓存命中率指标，请确认该字段已回传——这与「命中率为 0」是两回事。"
        )
        result.warnings.append(result.cache_note)
        return

    result.cache_measurable = True
    total_prompt = sum(t.input_tokens for t in measurable)
    total_cached = sum(int(t.cached_tokens or 0) for t in measurable)
    result.cache_hit_rate_overall = round(
        total_cached / total_prompt if total_prompt > 0 else 0.0, 4)
    by_round: Dict[int, List[Tuple[int, int]]] = {}
    for t in measurable:
        by_round.setdefault(t.round_idx, []).append((int(t.cached_tokens or 0), t.input_tokens))
    for r in sorted(by_round.keys()):
        c = sum(x[0] for x in by_round[r])
        p = sum(x[1] for x in by_round[r])
        result.cache_hit_rate_by_round[r] = round(c / p if p > 0 else 0.0, 4)
    covered = len(measurable)
    result.cache_note = (
        f"基于供应商回传的 cached_tokens 计算，覆盖 {covered}/{len(good)} 个成功请求。"
        f"文档提示：Round0 无历史，命中率接近 0 属正常；进入多轮后应显著上升。"
    )


def _compare_baseline(result: BenchmarkResult) -> None:
    """压测有效性自检（GLM-5.3 无官方公开性能基线，不做 pass/fail 判定）。

    初版这里沿用了 Kimi-K3 自测文档 3.3 的参考数字（TTFT≤8s 等）去判
    pass/fail——对 GLM-5.3 完全不适用：智谱官方未公开 TTFT/吞吐基线，
    拿别家文档的数字判这家"劣化"没有依据。

    改为只陈述事实：本轮压测本身是否有效（有没有压出压力、稳态窗口有没有
    覆盖、成功率高不高、有没有伪流式），实测数字交给评审自己解读。
    """

    def _fact(actual: Any, note: str = "") -> Dict[str, Any]:
        return {"actual": actual, "baseline": "—", "pass": None,
                "measurable": True, "note": note}

    if result.total_requests == 0:
        result.baseline_comparison = {}
        return

    success_rate = (result.successful_requests / result.total_requests * 100
                    if result.total_requests else 0.0)
    sat_note = ""
    if result.saturated:
        sat_note = "吞吐已低于发压速率的 85%，供应商成为瓶颈，实测数字反映其服务能力"
    elif result.ran_out_of_work:
        sat_note = "计划轮次提前跑完，未形成持续压力，吞吐数字反映发压节奏而非服务能力"
    else:
        sat_note = "完成速率接近发压速率，尚未压到能力上限，吞吐数字反映发压节奏"

    result.baseline_comparison = {
        "是否压到饱和": _fact("是" if result.saturated else "否", sat_note),
        "稳态窗口覆盖": _fact(
            "是" if result.steady_metrics_valid else "否（全程口径）",
            "" if result.steady_metrics_valid else
            "压测在爬坡期就结束了，分位数与稳态口径不可比"),
        "请求成功率": _fact(
            f"{result.successful_requests}/{result.total_requests}"
            f"（{success_rate:.1f}%）"),
        "伪流式请求数": _fact(
            result.degenerate_stream_count,
            "" if not result.degenerate_stream_count else
            "响应被整段缓冲后一次性返回，TPOT/ITL 无参考意义"),
        "TTFT口径说明": _fact(
            "含思考阶段（首个 reasoning_content 分片）",
            "GLM-5.3 始终思考，TTFT 统计从请求发出到首个思考/正文分片到达"),
    }
