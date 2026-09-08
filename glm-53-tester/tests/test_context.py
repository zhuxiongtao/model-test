"""G11-G12 长上下文 + 最大输出

对齐官方文档：
  - GLM-5.3 支持 1M 上下文窗口
  - 最大输出 Tokens 128K（默认 64K）
"""
from __future__ import annotations

import random
import string
from typing import Any, Dict, List

import httpx

from config import TesterConfig
from .base import TestCase, TestCaseMeta, TestResult, chat_request, get_content_text


def _make_filler_text(token_target: int) -> str:
    """生成约 token_target 个 token 的填充文本（中文约1.5字/token）"""
    chars = token_target * 2  # 粗估
    pool = string.ascii_letters + string.digits + " "
    return "".join(random.choice(pool) for _ in range(chars))


# ---------- G11: 长上下文(1M 窗口验证) ----------
async def _run_g11(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    """验证长上下文能力。默认测 32K token 量级（全量1M耗时过长）。
    策略：注入一个长文本，末尾植入一个关键答案，要求模型回忆。"""
    result = TestResult(meta=case.meta)
    if not cfg.capability.support_long_context_1m:
        result.status = "waive"
        result.add_note("供应商未声明支持 1M 长上下文")
        return result

    target = cfg.long_context_test_tokens  # 默认 32K
    secret = "GLM-5.3-LONG-CONTEXT-OK-7749"
    filler = _make_filler_text(target)
    prompt = (
        f"以下是一段参考资料，请仔细阅读：\n{filler}\n\n"
        f"参考资料结束。请回答：参考资料中是否出现了字符串 '{secret}'？"
        f"如果出现了，请原样输出该字符串；如果没有，回答'未出现'。"
    )
    # 把 secret 藏在中间
    mid = len(filler) // 2
    prompt_with_secret = (
        f"以下是一段参考资料：\n{filler[:mid]}\n{secret}\n{filler[mid:]}\n\n"
        f"请回答：上面参考资料中出现的特殊字符串是什么？请原样输出。"
    )
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": prompt_with_secret}],
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "max_tokens": 256,
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    content = get_content_text(body)
    found = secret in content
    result.assert_true(
        found,
        f"长上下文({target} tokens)中植入的字符串未被召回。"
        f"期望输出包含 '{secret}'，实际: '{content[:100]}'。"
        f"官方文档：GLM-5.3 支持 1M 上下文窗口。"
    )
    # 记录 prompt token 数，确认确实是长上下文
    if isinstance(result.usage, dict):
        pt = result.usage.get("prompt_tokens", 0)
        result.records.append({"prompt_tokens": pt, "secret_recalled": found})
        result.add_note(f"[INFO] 实际 prompt_tokens={pt}（目标约{target}）")
    return result


G11 = TestCase(
    meta=TestCaseMeta(
        id="G11", category="上下文能力", name="长上下文召回",
        required=False, capability_key="support_long_context_1m",
        check_points=[
            f"默认注入约{32000} token 上下文，末尾植入关键串并成功召回",
            "官方支持 1M 上下文窗口",
        ],
    ),
    runner=_run_g11,
)


# ---------- G12: 最大输出 128K ----------
async def _run_g12(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    """验证 max_tokens 可设到 128K（实际请求控制输出避免过长）。
    策略：设 max_tokens=131072 但让模型只输出少量内容，验证参数被接受。"""
    result = TestResult(meta=case.meta)
    if not cfg.capability.support_max_output_128k:
        result.status = "waive"
        result.add_note("供应商未声明支持 128K 最大输出")
        return result

    # max_tokens=131072 (128K) 应被接受
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "用一句话介绍你自己，不超过20字。"}],
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "max_tokens": 131072,
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    result.assert_true(
        result.http_status == 200,
        f"max_tokens=131072(128K) 应被接受，实际HTTP={result.http_status}。"
        f"官方文档：GLM-5.3 最大输出 Tokens 为 128K。"
    )
    if isinstance(result.usage, dict):
        ct = result.usage.get("completion_tokens", 0)
        result.records.append({"max_tokens_requested": 131072, "completion_tokens_actual": ct})
        result.add_note(f"[INFO] 请求 max_tokens=131072，实际 completion_tokens={ct}")
    return result


G12 = TestCase(
    meta=TestCaseMeta(
        id="G12", category="上下文能力", name="最大输出128K",
        required=False, capability_key="support_max_output_128k",
        check_points=["max_tokens=131072 被接受，官方支持 128K 最大输出"],
    ),
    runner=_run_g12,
)


CONTEXT_CASES: List[TestCase] = [G11, G12]
