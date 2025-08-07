# POLY_Tool - 简化版工具包
# 只保留三个核心节点

from .POLY_StringConcatenation import POLY_StringConcatenation
from .POLY_DynamicPrompt import POLY_DynamicPrompt
from .POLY_GenderJudgment import POLY_GenderJudgment

NODE_CLASS_MAPPINGS = {
    "POLY_StringConcatenation": POLY_StringConcatenation,
    "POLY_DynamicPrompt": POLY_DynamicPrompt,
    "POLY_GenderJudgment": POLY_GenderJudgment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "POLY_StringConcatenation": "POLY 字符串拼接",
    "POLY_DynamicPrompt": "POLY 动态提示词",
    "POLY_GenderJudgment": "POLY 性别判断"
}
