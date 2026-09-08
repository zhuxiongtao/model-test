"""多模态样本管理与"看懂了没有"的判定工具。

设计要点（针对旧实现的两类误判）：

1. 旧实现用"回答是否包含回避词"判定，回避词表里有 `视频数据` / `无法` 这种
   高频子串，正常回答（"从视频数据可以看到…"、"无法确定拍摄地点，但…"）
   会被误杀 → **假失败**。
2. 旧实现只要求"回答 ≥5 字且不含回避词"，供应商即使完全丢弃了图/视频、
   纯靠文字提示编一段话也能过 → **假通过**。

新实现改为 **ground truth 断言**：样本内容是我们自己生成的已知事实
（图=红底大白字7；视频=背景依次红→绿→蓝，且必须理解时序才能答对），
断言"回答是否命中这些事实"。同时提供 negative control（同样的问题不带
媒体再问一次），用于识别"模型其实没看，只是猜对了"的情况。
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"

IMAGE_SAMPLE = SAMPLES_DIR / "image_sample.jpg"
VIDEO_SAMPLE = SAMPLES_DIR / "video_sample.mp4"

# 图片样本的公网地址（MDN shared-assets，CC0）。与本地 image_sample.jpg 是同一张，
# 因此 F4（远程URL）与 F5（base64）用同一套 ground truth，无需额外配置。
IMAGE_PUBLIC_URL = "https://mdn.github.io/shared-assets/images/examples/elephant.jpg"

# base64 体积上限：超过就不发，直接判"样本过大"，避免把网关 413 误读成能力缺失
MAX_INLINE_B64_BYTES = 3 * 1024 * 1024


# -------------------- ground truth 定义 --------------------
# 每个元素是一组同义词，命中其中任意一个即算该条事实被答出。

# 图片样本 = 一张真实照片（大象剪影 / 夕阳 / 鸟群 / 树，内容已人工核实）。
# 用真实照片而不是合成色块，是因为它更接近实际接入后的使用场景，
# 同时又有多个互相独立、措辞容错的可验证事实。
IMAGE_FACTS: List[List[str]] = [
    ["大象", "象", "elephant"],
    ["鸟", "飞鸟", "鸟群", "bird"],
    ["日落", "夕阳", "黄昏", "落日", "傍晚", "sunset", "dusk"],
]
IMAGE_FACT_LABELS = ["主体=大象", "天空有鸟群", "光线=日落/黄昏"]
# 真实照片的描述措辞差异大，命中 2 条即视为确实看懂了画面
IMAGE_MIN_HITS = 2

VIDEO_FACTS: List[List[str]] = [
    ["红", "红色", "red"],
    ["绿", "绿色", "green"],
    ["蓝", "蓝色", "blue"],
]
VIDEO_FACT_LABELS = ["背景出现红色", "背景出现绿色", "背景出现蓝色"]
# 三种颜色按时间顺序出现，单帧无法答全；命中 ≥2 视为确实解析了多帧
VIDEO_MIN_HITS = 2

VIDEO_BONUS_FACTS: List[List[str]] = [
    ["1", "一"], ["2", "二"], ["3", "三"],
]

IMAGE_PROMPT = (
    "请描述这张照片：画面中最主要的动物是什么？天空中有什么？"
    "整体是什么光线条件（白天/日落/夜晚）？请简要作答。"
)
VIDEO_PROMPT = (
    "请观看这段视频并回答：视频的背景颜色从头到尾一共变化了几次？"
    "请按出现的先后顺序，依次列出每一段的背景颜色名称。"
    "只根据你实际看到的画面回答。"
)
# negative control：同样的问题，但不附带媒体。用于判断模型是不是在瞎猜。
IMAGE_CONTROL_PROMPT = IMAGE_PROMPT + "（注意：如果你没有收到任何图片，请直接回答「没有收到图片」。）"
VIDEO_CONTROL_PROMPT = VIDEO_PROMPT + "（注意：如果你没有收到任何视频，请直接回答「没有收到视频」。）"


# -------------------- 样本加载 --------------------

class SampleMissing(RuntimeError):
    """样本文件缺失或损坏。属于工具自身的问题，不应记为供应商能力缺失。"""


def _read_sample(path: Path, override: Optional[str] = None) -> Tuple[bytes, Path]:
    p = Path(override).expanduser() if override else path
    if not p.exists():
        raise SampleMissing(f"样本文件不存在: {p}")
    data = p.read_bytes()
    if len(data) < 512:
        raise SampleMissing(f"样本文件过小，疑似损坏: {p} ({len(data)} bytes)")
    return data, p


def load_image_sample(override: Optional[str] = None) -> Tuple[bytes, Path]:
    return _read_sample(IMAGE_SAMPLE, override)


def load_video_sample(override: Optional[str] = None) -> Tuple[bytes, Path]:
    return _read_sample(VIDEO_SAMPLE, override)


def to_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def self_check() -> List[str]:
    """启动自检：确认内置样本真的可解码。

    旧版本内置的 base64 PNG 缺少 IDAT/IEND 且 tIME 块 CRC 错误，任何解码器都
    打不开——而 F5 是必过项，等于工具自己制造了一条必失败用例。这里在启动时
    就把这类问题暴露出来。
    """
    problems: List[str] = []

    # 图像：必须能被真正解码
    try:
        data, path = load_image_sample()
        try:
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(data)) as im:
                im.load()
                if im.size[0] < 64 or im.size[1] < 64:
                    problems.append(f"图像样本尺寸过小({im.size})，部分视觉模型会拒绝处理")
        except ImportError:
            # 没装 Pillow 时退化为结构检查
            if not data.startswith(b"\xff\xd8\xff"):
                problems.append("图像样本不是合法 JPEG")
        except Exception as e:
            problems.append(f"图像样本无法解码: {type(e).__name__}: {e}")
    except SampleMissing as e:
        problems.append(str(e))

    # 视频：检查 MP4 容器基本结构
    try:
        data, path = load_video_sample()
        if data[4:8] != b"ftyp":
            problems.append("视频样本不是合法 MP4（缺少 ftyp box）")
        elif b"moov" not in data[:len(data)] or b"mdat" not in data:
            problems.append("视频样本缺少 moov/mdat box，可能不完整")
    except SampleMissing as e:
        problems.append(str(e))

    return problems


# -------------------- 判定工具 --------------------

def count_fact_hits(text: str, facts: Sequence[Sequence[str]]) -> Tuple[int, List[int]]:
    """统计回答命中了几条 ground truth，返回 (命中数, 命中的下标列表)。"""
    if not text:
        return 0, []
    low = text.lower()
    hit_idx: List[int] = []
    for i, synonyms in enumerate(facts):
        if any(s.lower() in low for s in synonyms):
            hit_idx.append(i)
    return len(hit_idx), hit_idx


def describe_hits(labels: Sequence[str], hit_idx: Sequence[int]) -> str:
    return "、".join(
        f"{lab}{'✓' if i in hit_idx else '✗'}" for i, lab in enumerate(labels)
    )


# 明确指向"拿不到媒体文件"的表述。这类失败是供应商侧的取回问题
# （网络不通 / 白名单 / 超时），不等于不具备多模态能力，必须分开归类。
_FETCH_FAILURE_PATTERNS = [
    r"无法(访问|下载|获取|打开|加载)",
    r"(下载|获取|拉取)(失败|超时)",
    r"链接(无效|失效|不可用)",
    r"(url|链接).{0,10}(无效|失败|不可达)",
    r"failed to (fetch|download|retrieve|load)",
    r"(cannot|could not|unable to) (access|fetch|download|open|retrieve)",
    r"invalid (url|image url|video url)",
    r"timeout.{0,20}(fetch|download)",
]

# 明确指向"我没收到媒体"的表述——用于 negative control 判定，
# 以及识别「供应商网关把多模态部分丢掉了」这种最该被抓出来的情况。
# 注意必须避免误伤正常回答里的否定（如"画面中没有看到人物，只有一只猫"），
# 因此限定主语是"对话/上下文/上传"层面的，而不是画面内容层面的。
_NO_MEDIA_PATTERNS = [
    r"没有(收到|看到|接收到|提供).{0,6}(图片|图像|视频)",
    r"未(收到|接收到|提供).{0,6}(图片|图像|视频)",
    r"(图片|图像|视频).{0,6}(未提供|没有提供|缺失|未上传)",
    r"(对话|上下文|消息|请求)(中|里)?.{0,6}(没有|未见|不存在).{0,6}(图片|图像|视频)",
    r"(no|not receive|didn't receive|haven't received).{0,20}(image|video|picture)",
    r"(image|video).{0,20}(was not|wasn't|not) (provided|received|attached)",
]


def looks_like_fetch_failure(text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low) for p in _FETCH_FAILURE_PATTERNS)


def looks_like_no_media(text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low) for p in _NO_MEDIA_PATTERNS)


# -------------------- 请求体形状变体 --------------------
# 不同供应商对视频入参的写法并不统一。只试一种写法就判"不支持视频"是误判，
# 因此按顺序尝试多种主流写法，并在报告里注明实际生效的是哪一种。

ImagePartBuilder = Callable[[str], Dict[str, Any]]
VideoPartBuilder = Callable[[str], Dict[str, Any]]

IMAGE_PART_SHAPES: List[Tuple[str, ImagePartBuilder]] = [
    ("image_url", lambda u: {"type": "image_url", "image_url": {"url": u}}),
    ("image", lambda u: {"type": "image", "image": u}),
]

VIDEO_PART_SHAPES: List[Tuple[str, VideoPartBuilder]] = [
    ("video_url", lambda u: {"type": "video_url", "video_url": {"url": u}}),
    ("video", lambda u: {"type": "video", "video": u}),
    ("input_video", lambda u: {"type": "input_video", "input_video": {"url": u}}),
]


def build_multimodal_messages(prompt: str, part: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}, part]}]
