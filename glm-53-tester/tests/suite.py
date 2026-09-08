"""G1~G20 全量用例聚合入口"""
from typing import List

from .base import TestCase
from .test_protocol import PROTOCOL_CASES          # G1 G2 G3 G4
from .test_thinking import THINKING_CASES          # G5 G6 G7
from .test_sampling import SAMPLING_CASES          # G8 G9 G10
from .test_context import CONTEXT_CASES            # G11 G12
from .test_toolcall import TOOLCALL_CASES          # G13 G14 G15
from .test_structured import STRUCTURED_CASES      # G16 G17
from .test_capability import CAPABILITY_CASES      # G18 G19 G20

ALL_FUNCTIONAL_CASES: List[TestCase] = [
    *PROTOCOL_CASES,
    *THINKING_CASES,
    *SAMPLING_CASES,
    *CONTEXT_CASES,
    *TOOLCALL_CASES,
    *STRUCTURED_CASES,
    *CAPABILITY_CASES,
]
