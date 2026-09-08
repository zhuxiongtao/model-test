"""G16-G17 结构化输出 JSON / JSONSchema

对齐官方文档：支持 JSON 等结构化格式输出
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from config import TesterConfig
from .base import TestCase, TestCaseMeta, TestResult, chat_request, get_content_text


# ---------- G16: JSON Object 模式 ----------
async def _run_g16(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    if not cfg.capability.support_structured_output:
        result.status = "waive"
        result.add_note("供应商未声明支持结构化输出")
        return result
    payload = {
        "model": cfg.model_id,
        "messages": [
            {"role": "system", "content": "你是一个JSON生成器，只输出合法JSON。"},
            {"role": "user", "content": "输出一个JSON，包含 name(string)、age(number)、hobbies(array of string) 三个字段。"},
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    content = get_content_text(body).strip()
    try:
        obj = json.loads(content)
        result.assert_true(isinstance(obj, dict), f"输出非JSON对象: {type(obj).__name__}")
        if isinstance(obj, dict):
            result.assert_true("name" in obj, f"缺少 name 字段: {obj}")
            result.assert_true("age" in obj, f"缺少 age 字段: {obj}")
            result.assert_true("hobbies" in obj, f"缺少 hobbies 字段: {obj}")
    except json.JSONDecodeError as e:
        result.assert_true(False, f"输出非合法JSON: {e}。内容: {content[:100]}")
    return result


G16 = TestCase(
    meta=TestCaseMeta(
        id="G16", category="结构化输出", name="JSONObject模式",
        required=True, capability_key="support_structured_output",
        check_points=["response_format=json_object 输出合法JSON"],
    ),
    runner=_run_g16,
)


# ---------- G17: JSON Schema strict 模式 ----------
async def _run_g17(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    result = TestResult(meta=case.meta)
    if not cfg.capability.support_json_schema_strict:
        result.status = "waive"
        result.add_note("供应商未声明支持 JSONSchema strict 模式")
        return result
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature": {"type": "number"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city", "temperature", "unit"],
    }
    payload = {
        "model": cfg.model_id,
        "messages": [
            {"role": "system", "content": "你是天气数据生成器。"},
            {"role": "user", "content": "生成北京的天气数据。"},
        ],
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {"schema": schema, "name": "weather", "strict": True}},
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    content = get_content_text(body).strip()
    try:
        obj = json.loads(content)
        result.assert_true(isinstance(obj, dict), f"输出非JSON对象")
        if isinstance(obj, dict):
            result.assert_true("city" in obj and "temperature" in obj and "unit" in obj,
                               f"缺少必填字段: {list(obj.keys())}")
            result.assert_true(obj["unit"] in ("celsius", "fahrenheit"),
                               f"unit 不在枚举内: {obj.get('unit')}")
    except json.JSONDecodeError as e:
        result.assert_true(False, f"输出非合法JSON: {e}")
    return result


G17 = TestCase(
    meta=TestCaseMeta(
        id="G17", category="结构化输出", name="JSONSchema严格模式",
        required=False, capability_key="support_json_schema_strict",
        check_points=["response_format=json_schema+strict 输出符合 schema"],
    ),
    runner=_run_g17,
)


STRUCTURED_CASES: List[TestCase] = [G16, G17]
