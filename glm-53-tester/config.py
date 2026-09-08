"""全局配置管理 - GLM-5.3 供应商自测工具

对齐智谱 GLM-5.3 官方文档（docs.bigmodel.cn）：
  - 模型编码 glm-5.3，OpenAI 兼容接口 https://open.bigmodel.cn/api/paas/v4/
  - 最大上下文 1M，最大输出 128K
  - 强制深度思考（thinking={type:"enabled"}），关闭报错
  - reasoning_effort 支持 low/high/max（仅三档）
  - temperature (0,1] 默认1.0，top_p [0.01,1.0] 默认0.95
  - 支持流式工具调用 tool_stream=true
  - 支持 do_sample 确定性输出
  - 支持结构化输出 JSON/JSONSchema
"""
from __future__ import annotations

import os
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG_FILE = DATA_DIR / "config.json"


class CapabilityDeclare(BaseModel):
    """GLM-5.3 能力声明 - 未声明支持的能力将自动标记为豁免而非失败

    默认值按 GLM-5.3 官方文档设定：
      · thinking 强制开启（核心特性）
      · reasoning_effort 必过三档
      · function calling / 结构化输出 / 流式 为标准能力
      · 流式工具调用 tool_stream 为 GLM 特有能力
    """
    support_thinking: bool = True              # GLM-5.3 强制开启思考
    thinking_default_enabled: bool = True       # 默认即开启
    support_reasoning_effort: bool = True       # low/high/max 三档
    support_streaming: bool = True              # 流式输出
    support_streaming_tool: bool = True         # tool_stream 流式工具调用
    support_function_calling: bool = True       # Function Calling
    support_tool_choice: bool = True            # tool_choice (auto/none/required/function)
    support_structured_output: bool = True      # JSON / JSONSchema
    support_json_schema_strict: bool = True     # strict 模式
    support_do_sample: bool = True              # do_sample 确定性输出
    support_context_cache: bool = True          # 上下文缓存
    support_long_context_1m: bool = True        # 1M 上下文
    support_max_output_128k: bool = True        # 128K 最大输出


# 官方明确要求的「必过能力」。声明为 False 时报告单列「不满足官方能力声明」。
MIN_REQUIRED_CAPABILITIES: Dict[str, str] = {
    "support_thinking": "GLM-5.3 强制开启深度思考，不可关闭",
    "support_reasoning_effort": "reasoning_effort 支持 low/high/max 三档",
    "support_function_calling": "支持 Function Calling",
    "support_structured_output": "支持 JSON / JSONSchema 结构化输出",
}


class BenchmarkConfig(BaseModel):
    """性能压测参数 - 可按需配置

    上一轮实测教训（20260907_142518 轮次）：10 会话 × 平均5轮 = 65 个计划轮次，
    arrival_rate_end=0.5 req/s，结果 65 轮在 4 分钟内全部跑完就退出——
    测的是"活干完了"，吞吐数字反映发压节奏而非服务能力，压测等于没压。

    本轮调整原则（压测时长仍为 4 分钟，满足整轮 20 分钟预算）：
    - 到达率上限 0.8 req/s：4 分钟约 186 个请求，足够形成统计样本；
    - 会话数 40：GLM-5.3 思考模型单请求延迟长（实测 TTFT P50 约 15s），
      并发需求 ≈ 到达率 × 平均延迟 ≈ 0.8 × 40s ≈ 32 个在途，会话数必须大于它，
      否则吞吐会被"会话数不足"封顶并被误判为"供应商饱和"；
    - 平均 8 轮/会话 = 320 个计划轮次 >> 186：保证计划时长内不会提前跑完。

    负载口径（20260907 修正）：初版 init=2000/input=500 加上填充系数错误，
    实测单请求平均输入仅 1164 token，与报告声称的"平均32k输入"严重不符，
    费用极低、TTFT/缓存数据也不代表长上下文场景。现按 32k 级负载修正：
    初始 prompt ~24k，每轮追加输入 ~2k，多轮后上下文自然增长到 32k+；
    输出对齐 300 token 口径。注意：修正后单轮压测 token 消耗约为原来的
    25 倍（约 500 万输入 token），介意费用可在前端压测参数区调小。
    """
    total_sessions: int = 40
    arrival_rate_start: float = 0.1
    arrival_rate_end: float = 0.8
    ramp_duration_seconds: int = 120
    steady_duration_seconds: int = 120
    num_rounds_avg: float = 8.0
    num_rounds_p50: float = 6.0
    num_rounds_p75: float = 10.0
    num_rounds_p90: float = 12.0
    num_rounds_p95: float = 15.0
    turn_interval_avg: float = 5.0
    turn_interval_p50: float = 3.0
    turn_interval_p75: float = 7.0
    turn_interval_p90: float = 12.0
    turn_interval_p95: float = 20.0
    init_prompt_length_avg: int = 24000
    init_prompt_length_p50: int = 16000
    init_prompt_length_p95: int = 32000
    input_length_avg: int = 2000
    input_length_p50: int = 1200
    input_length_p95: int = 4000
    output_length_avg: int = 300
    output_length_p50: int = 200
    output_length_p95: int = 800


class TesterConfig(BaseModel):
    api_base: str = ""            # 默认 https://open.bigmodel.cn/api/paas/v4
    api_key: str = ""
    model_id: str = "glm-5.3"     # GLM-5.3 旗舰模型
    timeout_seconds: int = 90
    stream_idle_timeout: int = 60
    max_retries: int = 1
    capability: CapabilityDeclare = Field(default_factory=CapabilityDeclare)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    run_functional: bool = True
    run_benchmark: bool = False   # GLM-5.3 默认不跑压测，按需开启
    # 长上下文测试用的 token 长度。1M 全量测试耗时极长，默认测 16K 量级验证无损。
    long_context_test_tokens: int = 16000

    @property
    def endpoint(self) -> str:
        base = self.api_base.rstrip("/")
        return f"{base}/chat/completions"

    def redacted(self) -> Dict[str, Any]:
        data = self.model_dump()
        key = data.pop("api_key", "") or ""
        data["api_key_masked"] = (
            key[:4] + "***" + key[-4:] if len(key) > 8 else ("***" if key else "")
        )
        data["api_key_configured"] = bool(key)
        return data


_lock = threading.RLock()
_config: Optional[TesterConfig] = None


def _load_from_disk() -> TesterConfig:
    cfg = TesterConfig()
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            cfg = TesterConfig(**data)
        except Exception:
            cfg = TesterConfig()
    # 环境变量兜底
    if not cfg.api_key:
        for env_name in ("GLM_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY"):
            v = os.environ.get(env_name)
            if v and v.strip():
                cfg.api_key = v.strip()
                break
    if not cfg.api_base:
        v = os.environ.get("GLM_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        if v and v.strip():
            cfg.api_base = v.strip()
        else:
            cfg.api_base = "https://open.bigmodel.cn/api/paas/v4"
    if not cfg.model_id:
        v = os.environ.get("GLM_MODEL_ID") or os.environ.get("MODEL_NAME")
        if v and v.strip():
            cfg.model_id = v.strip()
    return cfg


def get_config() -> TesterConfig:
    global _config
    with _lock:
        if _config is None:
            _config = _load_from_disk()
        return _config.model_copy(deep=True)


def save_config(cfg: Dict[str, Any] | TesterConfig) -> TesterConfig:
    global _config
    with _lock:
        if isinstance(cfg, dict):
            merged = _load_from_disk().model_dump()
            for k, v in cfg.items():
                if k == "api_key" and not (isinstance(v, str) and v.strip()):
                    continue
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v
            _config = TesterConfig(**merged)
        else:
            _config = cfg
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            _config.model_dump_json(indent=2), encoding="utf-8"
        )
        try:
            CONFIG_FILE.chmod(0o600)
        except OSError:
            pass
        return _config.model_copy(deep=True)
