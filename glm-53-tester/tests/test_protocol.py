"""G1-G4 协议完整性检测

对齐 GLM-5.3 官方文档（docs.bigmodel.cn/cn/guide/models/text/glm-5.3）：
  - OpenAI Chat Completion 协议兼容
  - 流式输出 delta.reasoning_content + delta.content
  - Usage 字段 prompt/completion/total_tokens
"""
from __future__ import annotations

from typing import Dict, List

import httpx

from config import TesterConfig
from .base import (
    TestCase, TestCaseMeta, TestResult, chat_request, validate_openai_schema,
    get_content_text, get_reasoning_text, analyze_stream_quality,
)


# ---------- G1: 流式输出完整性 ----------
async def _run_g1(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "用一句话介绍智谱 GLM-5.3 的核心能力。"}],
        "stream": True,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    chunks = await chat_request(client, cfg, payload, result)
    assert isinstance(chunks, list)
    for c in chunks:
        if c.parsed_json is not None:
            validate_openai_schema(result, c.parsed_json, is_chunk=True)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    real = [c for c in chunks if c.raw != "[DONE]"]
    result.assert_true(len(real) >= 2, f"有效分片不足: {len(real)}")
    if chunks:
        result.assert_true(chunks[-1].raw == "[DONE]", "未以[DONE]结束")
    # GLM-5.3 始终思考，流式应有 reasoning_content 增量
    reasoning = "".join(c.reasoning_delta for c in chunks if c.reasoning_delta)
    result.assert_true(
        len(reasoning.strip()) > 0,
        f"GLM-5.3 流式应输出 reasoning_content（思考过程），实际为空。"
        f"官方文档：GLM-5.3 始终启用思考功能。"
    )
    content = "".join(c.delta_content for c in chunks if c.delta_content)
    result.assert_true(
        len(content.strip()) >= 5,
        f"流式回答文本过少({len(content.strip())}字符): '{content[:80]}'"
    )
    # 流式质量（伪流式检测）
    sq = analyze_stream_quality(chunks)
    if sq["degenerate"]:
        result.add_note(f"[流式质量] {sq['reason']}")
    return result


G1 = TestCase(
    meta=TestCaseMeta(
        id="G1", category="协议完整性", name="流式输出完整性",
        required=True, capability_key="support_streaming",
        check_points=[
            "stream=true 返回 SSE 分片，以[DONE]结束",
            "delta.reasoning_content 承载思考过程（GLM-5.3 始终思考）",
            "delta.content 承载正式回答",
        ],
    ),
    runner=_run_g1,
)


# ---------- G2: Usage 字段(非流式) ----------
async def _run_g2(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "介绍 Python 语言，50字以内。"}],
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    validate_openai_schema(result, body)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    u = result.usage
    result.assert_true(isinstance(u, dict), "缺少 usage 字段")
    if isinstance(u, dict):
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            result.assert_true(
                isinstance(u.get(k), int) and u[k] >= 0,
                f"usage.{k} 缺失或非负整数: {u.get(k)}"
            )
        if all(isinstance(u.get(k), int) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            result.assert_true(
                u["total_tokens"] >= u["prompt_tokens"] + u["completion_tokens"],
                f"total({u['total_tokens']}) < prompt+completion({u['prompt_tokens']+u['completion_tokens']})"
            )
    return result


G2 = TestCase(
    meta=TestCaseMeta(
        id="G2", category="协议完整性", name="Usage字段(非流式)",
        required=True,
        check_points=["usage 含 prompt/completion/total_tokens", "total >= prompt+completion"],
    ),
    runner=_run_g2,
)


# ---------- G3: Usage 字段(流式) ----------
async def _run_g3(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "用一句话介绍机器学习。"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    chunks = await chat_request(client, cfg, payload, result)
    assert isinstance(chunks, list)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    u = result.usage
    if not isinstance(u, dict):
        for c in chunks:
            if isinstance(c.parsed_json, dict) and isinstance(c.parsed_json.get("usage"), dict):
                u = c.parsed_json["usage"]
                result.usage = u
                break
    result.assert_true(isinstance(u, dict), "流式响应未找到 usage 字段")
    if isinstance(u, dict):
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            result.assert_true(
                isinstance(u.get(k), int) and u[k] >= 0,
                f"usage.{k} 缺失或非负整数: {u.get(k)}"
            )
    has_done = any(c.raw == "[DONE]" for c in chunks)
    result.assert_true(has_done, "流中未出现[DONE]")
    return result


G3 = TestCase(
    meta=TestCaseMeta(
        id="G3", category="协议完整性", name="Usage字段(流式)",
        required=True, capability_key="support_streaming",
        check_points=["stream_options.include_usage=true 能拿到 usage", "三字段齐全"],
    ),
    runner=_run_g3,
)


# ---------- G4: 深度思考流式分离 ----------
async def _run_g4(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    """GLM-5.3 流式响应中 reasoning_content（思考）与 content（回答）应分离"""
    result = TestResult(meta=case.meta)
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "1+1等于几？请给出答案。"}],
        "stream": True,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    chunks = await chat_request(client, cfg, payload, result)
    assert isinstance(chunks, list)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    reasoning = "".join(c.reasoning_delta for c in chunks if c.reasoning_delta)
    content = "".join(c.delta_content for c in chunks if c.delta_content)
    result.assert_true(
        len(reasoning.strip()) > 0,
        "流式响应未输出 reasoning_content，GLM-5.3 应始终输出思考过程"
    )
    result.assert_true(
        len(content.strip()) > 0,
        "流式响应未输出 content（正式回答为空）"
    )
    # reasoning_content 和 content 不应是同一段文本
    if reasoning.strip() and content.strip():
        result.assert_true(
            reasoning.strip() != content.strip(),
            "reasoning_content 与 content 内容完全相同，思考与回答未分离"
        )
    result.records.append({
        "reasoning_length": len(reasoning),
        "content_length": len(content),
    })
    return result


G4 = TestCase(
    meta=TestCaseMeta(
        id="G4", category="协议完整性", name="思考与回答流式分离",
        required=True, capability_key="support_thinking",
        check_points=[
            "delta.reasoning_content 承载思考",
            "delta.content 承载正式回答",
            "两者内容不同",
        ],
    ),
    runner=_run_g4,
)


PROTOCOL_CASES: List[TestCase] = [G1, G2, G3, G4]
