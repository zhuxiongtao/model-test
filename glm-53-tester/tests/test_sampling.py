"""G8-G10 采样参数 + do_sample 确定性

对齐官方文档：
  - temperature: (0.0, 1.0]，默认 1.0，最多2位小数
  - top_p: [0.01, 1.0]，默认 0.95
  - do_sample: 默认 true；false 时采用贪心策略，输出确定性
"""
from __future__ import annotations

from typing import Any, Dict, List

import httpx

from config import TesterConfig
from .base import TestCase, TestCaseMeta, TestResult, chat_request, get_content_text


# ---------- G8: temperature 范围 ----------
async def _run_g8(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    # 有效值 0.5 应通过
    payload_ok = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "说一个字"}],
        "stream": False, "temperature": 0.5,
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }
    body = await chat_request(client, cfg, payload_ok, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"temperature=0.5 应通过，实际HTTP={result.http_status}")

    # 超范围值 1.5 应被拒绝
    sub2 = TestResult(meta=case.meta)
    payload_bad = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "说一个字"}],
        "stream": False, "temperature": 1.5,
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }
    await chat_request(client, cfg, payload_bad, sub2)
    is_reject = sub2.http_status == 400 or (
        isinstance(sub2.response_body, dict) and "error" in sub2.response_body
    )
    result.assert_true(
        is_reject,
        f"temperature=1.5 超出 (0,1] 范围应被拒绝，实际HTTP={sub2.http_status}。"
        f"官方文档：temperature 取值范围 (0.0, 1.0]。"
    )
    if is_reject:
        result.add_note("temperature=1.5 被正确拒绝，符合官方范围约束")
    return result


G8 = TestCase(
    meta=TestCaseMeta(
        id="G8", category="采样参数", name="temperature范围约束",
        required=True,
        check_points=["temperature∈(0,1]，超范围返回错误"],
    ),
    runner=_run_g8,
)


# ---------- G9: top_p 范围 ----------
async def _run_g9(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    payload_ok = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "说一个字"}],
        "stream": False, "top_p": 0.8,
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }
    body = await chat_request(client, cfg, payload_ok, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"top_p=0.8 应通过，实际HTTP={result.http_status}")

    sub2 = TestResult(meta=case.meta)
    payload_bad = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "说一个字"}],
        "stream": False, "top_p": 0.0,
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }
    await chat_request(client, cfg, payload_bad, sub2)
    is_reject = sub2.http_status == 400 or (
        isinstance(sub2.response_body, dict) and "error" in sub2.response_body
    )
    result.assert_true(
        is_reject,
        f"top_p=0.0 超出 [0.01,1.0] 范围应被拒绝，实际HTTP={sub2.http_status}。"
        f"官方文档：top_p 取值范围 [0.01, 1.0]。"
    )
    if is_reject:
        result.add_note("top_p=0.0 被正确拒绝，符合官方范围约束")
    return result


G9 = TestCase(
    meta=TestCaseMeta(
        id="G9", category="采样参数", name="top_p范围约束",
        required=True,
        check_points=["top_p∈[0.01,1.0]，超范围返回错误"],
    ),
    runner=_run_g9,
)


# ---------- G10: do_sample=false 确定性输出 ----------
async def _run_g10(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    if not cfg.capability.support_do_sample:
        result.status = "waive"
        result.add_note("供应商未声明支持 do_sample")
        return result

    prompt = "用不超过10个字回答：中国的首都是哪里？只回答城市名。"
    outputs: List[str] = []
    for i in range(3):
        sub = TestResult(meta=case.meta)
        payload = {
            "model": cfg.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "do_sample": False,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
            "seed": 42,
        }
        body = await chat_request(client, cfg, payload, sub)
        assert isinstance(body, dict)
        result.assert_true(sub.http_status == 200, f"第{i+1}次 HTTP非200: {sub.http_status}")
        outputs.append(get_content_text(body).strip())

    result.records.append({"outputs": outputs})
    if len(set(outputs)) == 1:
        result.add_note(f"do_sample=false + seed=42 三次输出完全一致: '{outputs[0]}'")
    else:
        result.assert_true(
            False,
            f"do_sample=false + 相同 seed 输出不一致: {outputs}。"
            f"官方文档：do_sample=false 采用贪心策略，输出应确定性。"
        )
    return result


G10 = TestCase(
    meta=TestCaseMeta(
        id="G10", category="采样参数", name="do_sample=false确定性输出",
        required=False, capability_key="support_do_sample",
        check_points=["do_sample=false + 相同 seed 多次调用输出一致"],
    ),
    runner=_run_g10,
)


SAMPLING_CASES: List[TestCase] = [G8, G9, G10]
