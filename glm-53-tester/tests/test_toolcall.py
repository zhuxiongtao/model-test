"""G13-G15 工具调用 + 流式工具调用 tool_stream

对齐官方文档：
  - Function Calling 支持
  - tool_stream=true 流式工具调用参数拼接
  - tool_choice: auto/none/required/function
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from config import TesterConfig
from .base import TestCase, TestCaseMeta, TestResult, chat_request, get_content_text


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]


# ---------- G13: Function Calling 基础 ----------
async def _run_g13(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
        "stream": False,
        "tools": TOOLS,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    try:
        msg = body["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
    except Exception:
        tcs = []
    result.assert_true(
        len(tcs) > 0,
        "模型未发起工具调用。用户问天气，应调用 get_weather。"
        f"响应: {str(body)[:200]}"
    )
    if tcs:
        tc = tcs[0]
        fn = tc.get("function", {})
        result.assert_true(fn.get("name") == "get_weather", f"工具名错误: {fn.get('name')}")
        try:
            args = json.loads(fn.get("arguments", "{}"))
            result.assert_true("city" in args, f"工具参数缺少 city: {args}")
        except Exception:
            result.assert_true(False, f"工具参数非JSON: {fn.get('arguments')}")
    return result


G13 = TestCase(
    meta=TestCaseMeta(
        id="G13", category="工具调用", name="Function Calling基础",
        required=True, capability_key="support_function_calling",
        check_points=["模型正确发起工具调用，参数含 city"],
    ),
    runner=_run_g13,
)


# ---------- G14: 流式工具调用 tool_stream ----------
async def _run_g14(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    if not cfg.capability.support_streaming_tool:
        result.status = "waive"
        result.add_note("供应商未声明支持 tool_stream 流式工具调用")
        return result
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "上海天气怎么样？"}],
        "stream": True,
        "tool_stream": True,
        "tools": TOOLS,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    chunks = await chat_request(client, cfg, payload, result)
    assert isinstance(chunks, list)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    # 拼接 tool_calls arguments
    all_tc_deltas: List[Dict] = []
    for c in chunks:
        all_tc_deltas.extend(c.tool_calls_delta)
    result.assert_true(
        len(all_tc_deltas) > 0,
        "tool_stream=true 未收到任何 tool_calls 增量分片。"
        "官方文档：GLM-5.3 支持流式工具调用 tool_stream=true。"
    )
    # 拼接 arguments
    if all_tc_deltas:
        tc0 = all_tc_deltas[0]
        fn = tc0.get("function", {})
        name = fn.get("name", "")
        args_parts = [tc.get("function", {}).get("arguments", "") for tc in all_tc_deltas]
        args_str = "".join(args_parts)
        result.records.append({"tool_name": name, "arguments_streamed": args_str})
        result.assert_true(name == "get_weather", f"工具名错误: {name}")
        try:
            args = json.loads(args_str)
            result.assert_true("city" in args, f"流式拼接参数缺少 city: {args}")
            result.add_note(f"[INFO] 流式拼接工具参数成功: {args}")
        except Exception:
            result.assert_true(False, f"流式拼接 arguments 非合法JSON: '{args_str[:100]}'")
    return result


G14 = TestCase(
    meta=TestCaseMeta(
        id="G14", category="工具调用", name="流式工具调用(tool_stream)",
        required=False, capability_key="support_streaming_tool",
        check_points=["tool_stream=true 时 delta.tool_calls[*].function.arguments 流式拼接完整"],
    ),
    runner=_run_g14,
)


# ---------- G15: tool_choice 多分支 ----------
async def _run_g15(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    if not cfg.capability.support_tool_choice:
        result.status = "waive"
        result.add_note("供应商未声明支持 tool_choice")
        return result

    # required: 必须调用工具
    sub_req = TestResult(meta=case.meta)
    body_req = await chat_request(client, cfg, {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False, "tools": TOOLS, "tool_choice": "required",
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }, sub_req)
    msg_req = body_req.get("choices", [{}])[0].get("message", {}) if isinstance(body_req, dict) else {}
    has_tc = bool(msg_req.get("tool_calls"))
    result.assert_true(
        has_tc,
        "tool_choice=required 应强制调用工具，实际未调用。"
        f"响应: {str(body_req)[:200]}"
    )

    # none: 不调用工具，直接回答
    sub_none = TestResult(meta=case.meta)
    body_none = await chat_request(client, cfg, {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False, "tools": TOOLS, "tool_choice": "none",
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }, sub_none)
    msg_none = body_none.get("choices", [{}])[0].get("message", {}) if isinstance(body_none, dict) else {}
    has_tc_none = bool(msg_none.get("tool_calls"))
    content_none = get_content_text(body_none)
    result.assert_true(
        not has_tc_none and len(content_none.strip()) > 0,
        "tool_choice=none 应不调用工具并直接回答，实际"
        + ("调用了工具" if has_tc_none else "回答为空")
    )

    # auto: 自由选择
    sub_auto = TestResult(meta=case.meta)
    body_auto = await chat_request(client, cfg, {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": "北京天气怎么样？"}],
        "stream": False, "tools": TOOLS, "tool_choice": "auto",
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }, sub_auto)
    result.assert_true(sub_auto.http_status == 200, f"tool_choice=auto HTTP非200: {sub_auto.http_status}")
    return result


G15 = TestCase(
    meta=TestCaseMeta(
        id="G15", category="工具调用", name="tool_choice多分支",
        required=False, capability_key="support_tool_choice",
        check_points=["required强制调用", "none不调用", "auto自由选择"],
    ),
    runner=_run_g15,
)


TOOLCALL_CASES: List[TestCase] = [G13, G14, G15]
