"""G5-G7 深度思考 + reasoning_effort 三档

对齐官方文档：
  - GLM-5.3 始终启用思考，thinking.type 仅支持 enabled，传 disabled 会失败
  - reasoning_effort 支持 low/high/max（仅三档），默认 max
  - 传 thinking.type=disabled 请求应失败
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import httpx

from config import TesterConfig
from .base import (
    TestCase, TestCaseMeta, TestResult, chat_request, validate_openai_schema,
    get_reasoning_text, get_content_text,
)


def _extract_reasoning_tokens(usage: Dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    for k in ("reasoning_tokens", "thinking_tokens"):
        if isinstance(usage.get(k), int) and usage[k] >= 0:
            return usage[k]
    ctd = usage.get("completion_tokens_details")
    if isinstance(ctd, dict):
        for k in ("reasoning_tokens", "thinking_tokens"):
            if isinstance(ctd.get(k), int) and ctd[k] >= 0:
                return ctd[k]
    return 0


# ---------- G5: 思考强制开启 ----------
async def _run_g5(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    """不传 thinking 时默认即开启思考"""
    result = TestResult(meta=case.meta)
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
        "stream": False,
        "reasoning_effort": "low",
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    validate_openai_schema(result, body)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    reasoning = get_reasoning_text(body)
    result.assert_true(
        len(reasoning.strip()) > 0,
        "GLM-5.3 始终启用思考，不传 thinking 也应有 reasoning_content，实际为空。"
        "官方文档：GLM-5.3 会始终启用思考功能。"
    )
    content = get_content_text(body)
    result.assert_true(len(content.strip()) > 0, "正式回答 content 为空")
    return result


G5 = TestCase(
    meta=TestCaseMeta(
        id="G5", category="思考行为", name="思考强制开启(不传thinking)",
        required=True, capability_key="support_thinking",
        check_points=["不传 thinking 时默认开启思考，输出 reasoning_content"],
    ),
    runner=_run_g5,
)


# ---------- G6: 思考关闭应报错 ----------
async def _run_g6(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    """thinking.type=disabled 应返回错误（GLM-5.3 不支持关闭思考）"""
    result = TestResult(meta=case.meta)
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    body = await chat_request(client, cfg, payload, result, expect_error=True)
    assert isinstance(body, dict)
    status = result.http_status
    # 期望：400 错误或 200 但带 error
    is_reject = (
        status == 400
        or (isinstance(body, dict) and "error" in body and "choices" not in body)
    )
    result.assert_true(
        is_reject,
        f"thinking.type=disabled 应被拒绝，但实际 HTTP={status}。"
        f"官方文档：GLM-5.3 不再支持禁用思考功能，传 disabled 请求将失败。"
        f"响应: {str(body)[:200]}"
    )
    if is_reject:
        result.add_note("thinking.type=disabled 被正确拒绝，符合官方规范")
    return result


G6 = TestCase(
    meta=TestCaseMeta(
        id="G6", category="思考行为", name="思考关闭应报错",
        required=True, capability_key="support_thinking",
        check_points=["thinking.type=disabled 返回错误，符合官方规范"],
        expect_non_200=True,
    ),
    runner=_run_g6,
)


# ---------- G7: reasoning_effort 三档区分度 ----------
PROBLEM = (
    "用1、2、3、4、5各一次组成一个五位数，要求能被11整除。"
    "请一步步推理，给出尝试过的思路，再列出最终答案。"
)


async def _run_g7(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    if not cfg.capability.support_reasoning_effort:
        result.status = "waive"
        result.add_note("供应商未声明支持 reasoning_effort")
        return result

    records: List[Dict[str, Any]] = []
    levels = [("low", "low"), ("high", "high"), ("max", "max")]
    for label, value in levels:
        sub = TestResult(meta=case.meta)
        payload: Dict[str, Any] = {
            "model": cfg.model_id,
            "messages": [{"role": "user", "content": PROBLEM}],
            "stream": False,
            "thinking": {"type": "enabled"},
            "reasoning_effort": value,
            "seed": hash(label) & 0x7FFFFFFF,
        }
        body = await chat_request(client, cfg, payload, sub)
        assert isinstance(body, dict) or isinstance(body, list)
        if isinstance(body, dict):
            validate_openai_schema(result, body)
        result.assert_true(sub.http_status == 200, f"{label}档 HTTP非200: {sub.http_status}")
        rt = _extract_reasoning_tokens(sub.usage)
        p = sub.usage.get("prompt_tokens", 0) if isinstance(sub.usage, dict) else 0
        c = sub.usage.get("completion_tokens", 0) if isinstance(sub.usage, dict) else 0
        rec = {
            "level": label, "reasoning_tokens": rt,
            "prompt_tokens": p, "completion_tokens": c,
            "http_status": sub.http_status, "duration_ms": sub.duration_ms,
        }
        records.append(rec)
        if label == "max" and result.request_body is None:
            result.request_body = sub.request_body
            result.response_body = body
            result.http_status = sub.http_status
            result.usage = sub.usage

    result.records = records  # type: ignore[attr-defined]
    summary = " | ".join(
        f"{r['level']}: rt={r['reasoning_tokens']}, c={r['completion_tokens']}"
        for r in records
    )
    result.add_note(f"[INFO] 三档统计: {summary}")

    valid = [r for r in records if r["http_status"] == 200]
    if len(valid) < 2:
        result.assert_true(False, f"有效档位数不足({len(valid)}/{len(levels)})")
        return result

    low = next((r for r in valid if r["level"] == "low"), None)
    high = next((r for r in valid if r["level"] == "high"), None)
    mx = next((r for r in valid if r["level"] == "max"), None)

    if low and high:
        rt_low, rt_high = low["reasoning_tokens"], high["reasoning_tokens"]
        has_rt = any(rt > 0 for rt in (rt_low, rt_high, mx["reasoning_tokens"] if mx else 0))
        if not has_rt:
            metric_low, metric_high = low["completion_tokens"], high["completion_tokens"]
            metric_max = mx["completion_tokens"] if mx else metric_high
            field = "completion_tokens"
        else:
            metric_low, metric_high = rt_low, rt_high
            metric_max = mx["reasoning_tokens"] if mx else rt_high
            field = "reasoning_tokens"

        diff = metric_high - metric_low
        ratio = (metric_high / metric_low) if metric_low > 0 else float("inf")
        result.assert_true(
            (diff >= 20) or (ratio >= 1.2),
            f"{field} low({metric_low}) vs high({metric_high}) 区分度不足: "
            f"差={diff}(需≥20) 或 比率={ratio:.2f}(需≥1.2)。"
            f"reasoning_effort 参数可能未生效。"
        )
        # max 档记录（软判定，K3/GLM max 可能更高效而非更长）
        if mx and metric_high > 0 and metric_max > 0:
            ratio_mh = metric_max / metric_high
            if metric_max < metric_high * 0.5:
                result.add_failure(
                    f"⚠️ max档({field}={metric_max}) 显著低于 high档({metric_high})，"
                    f"比率={ratio_mh:.2f}(<0.5)。reasoning_effort=max 应为最高推理强度。"
                )
            else:
                result.add_note(f"[INFO] max vs high 比率={ratio_mh:.2f}")
    return result


G7 = TestCase(
    meta=TestCaseMeta(
        id="G7", category="思考行为", name="reasoning_effort三档区分度",
        required=True, capability_key="support_reasoning_effort",
        check_points=[
            "low/high/max 三档分别请求，usage 字段齐全",
            "low 与 high 有可观测区分度",
        ],
    ),
    runner=_run_g7,
)


THINKING_CASES: List[TestCase] = [G5, G6, G7]
