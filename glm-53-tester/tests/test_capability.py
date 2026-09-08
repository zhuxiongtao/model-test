"""G18-G20 基础能力验证 - 防止"挂羊头卖狗肉"

验证模型是否真的具备官方宣称的编程/推理/多轮能力，
而非只看 token 数量或表面响应。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import httpx

from config import TesterConfig
from .base import TestCase, TestCaseMeta, TestResult, chat_request, get_content_text, get_reasoning_text


# ---------- G18: 代码生成能力 ----------
async def _run_g18(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    """验证模型能生成可编译运行的 Python 代码（官方宣称编程能力 SOTA）"""
    result = TestResult(meta=case.meta)
    payload = {
        "model": cfg.model_id,
        "messages": [
            {"role": "system", "content": "你是资深Python工程师，只输出可运行代码，不要解释。"},
            {"role": "user", "content": "写一个Python函数 fibonacci(n) 返回第n个斐波那契数(n>=0)。只输出函数代码，用```python包裹。"},
        ],
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    content = get_content_text(body)
    # 提取代码块
    m = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
    code = m.group(1).strip() if m else content.strip()
    result.records.append({"generated_code": code[:500]})
    # 尝试编译
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        result.assert_true(False, f"生成代码语法错误: {e}。代码: {code[:200]}")
        return result
    # 尝试运行并验证结果
    try:
        ns: Dict[str, Any] = {}
        exec(code, ns)
        fib = ns.get("fibonacci")
        if not callable(fib):
            result.assert_true(False, "未找到 fibonacci 函数")
            return result
        # 验证前几个值
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        for i, exp in enumerate(expected):
            got = fib(i)
            result.assert_true(got == exp, f"fibonacci({i})={got}，期望 {exp}")
        result.add_note("[INFO] 生成代码编译运行通过，fibonacci(0..9) 结果正确")
    except Exception as e:
        result.assert_true(False, f"代码运行失败: {type(e).__name__}: {e}")
    return result


G18 = TestCase(
    meta=TestCaseMeta(
        id="G18", category="基础能力", name="代码生成(可运行)",
        required=True,
        check_points=["生成 Python 函数可编译运行，fibonacci 结果正确"],
    ),
    runner=_run_g18,
)


# ---------- G19: 数学推理能力 ----------
async def _run_g19(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    """验证模型真的能做数学推理（而非记忆答案）。用经典逻辑题。"""
    result = TestResult(meta=case.meta)
    # 一个需要推理的数学题
    question = (
        "一个水池有进水管和出水管。单开进水管6小时注满，单开出水管8小时放完。"
        "现在两管同时开，几小时注满空池？请给出推理过程和最终答案（数字）。"
    )
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    content = get_content_text(body)
    reasoning = get_reasoning_text(body)
    result.records.append({"reasoning_length": len(reasoning), "answer_length": len(content)})
    # 正确答案：1/(1/6 - 1/8) = 1/(1/24) = 24 小时
    # 在回答中查找 24
    has_24 = "24" in content
    result.assert_true(
        has_24,
        f"数学题答案错误。水池问题正确答案是 24 小时，回答中未找到'24'。"
        f"回答: {content[:200]}"
    )
    # 推理过程应非空（GLM-5.3 始终思考）
    result.assert_true(
        len(reasoning.strip()) > 0 or len(content.strip()) > 50,
        "推理过程过短，疑似未真正推理。官方宣称 GLM-5.3 推理能力强。"
    )
    return result


G19 = TestCase(
    meta=TestCaseMeta(
        id="G19", category="基础能力", name="数学推理(水池问题)",
        required=True,
        check_points=["正确得出 24 小时", "有推理过程"],
    ),
    runner=_run_g19,
)


# ---------- G20: 多轮对话上下文保持 ----------
async def _run_g20(case: TestCase, cfg: TesterConfig, client: httpx.AsyncClient) -> TestResult:
    """验证多轮对话中模型能记住前文信息"""
    result = TestResult(meta=case.meta)
    messages = [
        {"role": "user", "content": "我叫小明，今年8岁。记住我的名字和年龄。"},
        {"role": "assistant", "content": "好的小明，我记住了你今年8岁。"},
        {"role": "user", "content": "我叫什么名字？今年几岁？"},
    ]
    payload = {
        "model": cfg.model_id,
        "messages": messages,
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    body = await chat_request(client, cfg, payload, result)
    assert isinstance(body, dict)
    result.assert_true(result.http_status == 200, f"HTTP非200: {result.http_status}")
    content = get_content_text(body)
    has_name = "小明" in content
    has_age = "8" in content
    result.assert_true(has_name, f"多轮对话未记住名字'小明'，回答: {content[:100]}")
    result.assert_true(has_age, f"多轮对话未记住年龄'8'，回答: {content[:100]}")
    return result


G20 = TestCase(
    meta=TestCaseMeta(
        id="G20", category="基础能力", name="多轮上下文保持",
        required=True,
        check_points=["多轮对话中记住前文名字和年龄"],
    ),
    runner=_run_g20,
)


CAPABILITY_CASES: List[TestCase] = [G18, G19, G20]
