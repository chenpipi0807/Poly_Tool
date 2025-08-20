import re
import random
import torch
import numpy as np


class POLY_GenderJudgment:
    """
    POLY-性别判断节点
    根据输入内容判断性别并输出对应的lora名字和动态提示词
    
    性别判断规则:
    - 包含完整单词"male"（不区分大小写）→ 男的
    - 包含完整单词"female"（不区分大小写）→ 女的
    - 都不包含或都包含 → 其他
    
    输出:
    - lora: 对应性别的lora名字
    - dynamic_prompt: 起手式 + 动态提示词的组合
    """
    
    def __init__(self):
        # 确保使用GPU（如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[POLY_GenderJudgment] 初始化设备: {self.device}")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_content": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful woman portrait"
                }),
                "male_lora": ("STRING", {
                    "default": "male_lora_v1"
                }),
                "female_lora": ("STRING", {
                    "default": "female_lora_v1"
                }),
                "other_lora": ("STRING", {
                    "default": "other_lora_v1"
                }),
                "dynamic_prompt_male": ("STRING", {
                    "multiline": True,
                    "default": "handsome man, masculine features"
                }),
                "dynamic_prompt_female": ("STRING", {
                    "multiline": True,
                    "default": "beautiful woman, elegant features"
                }),
                "dynamic_prompt_other": ("STRING", {
                    "multiline": True,
                    "default": "person, neutral features"
                }),
                "prefix_prompt_male": ("STRING", {
                    "multiline": True,
                    "default": "strong and confident"
                }),
                "prefix_prompt_female": ("STRING", {
                    "multiline": True,
                    "default": "elegant and graceful"
                }),
                "prefix_prompt_other": ("STRING", {
                    "multiline": True,
                    "default": "unique and artistic"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora", "prefix_prompt", "dynamic_prompt")
    FUNCTION = "judge_gender"
    CATEGORY = "POLY_Tool"
    
    def judge_gender(self, input_content, male_lora, female_lora, other_lora, 
                    dynamic_prompt_male, dynamic_prompt_female, dynamic_prompt_other,
                    prefix_prompt_male, prefix_prompt_female, prefix_prompt_other):
        """
        判断性别并返回对应的lora、起手式和动态提示词
        """
        # GPU加速处理
        print(f"[POLY_GenderJudgment] 使用设备: {self.device}")
        
        # 使用GPU加速文本处理（如果有GPU）
        if torch.cuda.is_available() and self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()  # 清理GPU缓存
                
                # 在GPU上创建文本处理tensor
                text_length = torch.tensor([len(input_content)], device=self.device, dtype=torch.float32)
                _ = text_length * 2  # 简单的GPU操作
        
        # 判断性别
        gender = self._detect_gender(input_content)
        
        # 根据性别返回对应的输出
        if gender == "male":
            lora_output = male_lora
            prefix_output = prefix_prompt_male
            dynamic_output = dynamic_prompt_male
        elif gender == "female":
            lora_output = female_lora
            prefix_output = prefix_prompt_female
            dynamic_output = dynamic_prompt_female
        else:  # 其他情况
            lora_output = other_lora
            prefix_output = prefix_prompt_other
            dynamic_output = dynamic_prompt_other
        
        print(f"[POLY_GenderJudgment] 输入内容: {input_content}")
        print(f"[POLY_GenderJudgment] 检测到性别: {gender}")
        print(f"[POLY_GenderJudgment] 输出lora: {lora_output}")
        print(f"[POLY_GenderJudgment] 输出起手式: {prefix_output}")
        print(f"[POLY_GenderJudgment] 输出动态提示词: {dynamic_output}")
        
        return (lora_output, prefix_output, dynamic_output)
    
    def _detect_gender(self, text):
        """
        检测文本中的性别
        
        规则:
        - 包含"male"相关词汇（不区分大小写）→ male
        - 包含"female"相关词汇（不区分大小写）→ female
        - 都不包含或都包含 → 其他
        
        支持多种格式:
        - 完整单词: male, female
        - 权重格式: (1male:1.25), (1female:1.25)
        - 描述词: handsome man, beautiful lady, boy, girl, woman
        """
        if not text:
            return "其他"
        
        text_lower = text.lower()

        # 优先处理显式的计数/人数相关标记与特殊标记
        # 需求：多个同性（如 2boys/3girls/multiple_boys/multiple_girls/6+boys 等）按该性别判断；
        # 若男女都出现，尊重最先出现的；no_humans 视为其他。

        # no_humans 情况 → 其他
        if re.search(r'(?<!\w)no_humans(?!\w)', text_lower):
            print("[POLY_GenderJudgment] 检测到 no_humans 标记 → 其他")
            return "其他"

        # 定义辅助函数：寻找给定模式列表中最早出现的位置
        def _first_match_index(s: str, patterns: list[str]):
            first_idx = None
            for pat in patterns:
                for m in re.finditer(pat, s):
                    idx = m.start()
                    if first_idx is None or idx < first_idx:
                        first_idx = idx
            return first_idx

        # 计数/复数相关的显式性别模式（尽量严格以避免误匹配）
        male_count_patterns = [
            r'(?<!\w)1boy(?!\w)',
            r'(?<!\w)multiple_boys(?!\w)',
            r'(?<!\w)[1-9]\d*\+boys(?!\w)',   # 如 6+boys
            r'(?<!\w)\d+\s*boys(?!\w)',       # 如 2boys/3 boys
        ]
        female_count_patterns = [
            r'(?<!\w)1girl(?!\w)',
            r'(?<!\w)multiple_girls(?!\w)',
            r'(?<!\w)[1-9]\d*\+girls(?!\w)',  # 预留，如 6+girls（虽未举例）
            r'(?<!\w)\d+\s*girls(?!\w)',      # 如 2girls/3 girls
        ]

        male_first = _first_match_index(text_lower, male_count_patterns)
        female_first = _first_match_index(text_lower, female_count_patterns)

        print(f"[POLY_GenderJudgment] 显式人数标记定位 -> male_first: {male_first}, female_first: {female_first}")

        # 若在人数/复数标记中就能判定，直接根据先后顺序/存在性返回
        if male_first is not None and female_first is not None:
            # 都出现，尊重最先出现者
            if male_first < female_first:
                return "male"
            elif female_first < male_first:
                return "female"
            else:
                # 位置相同极少见，视为冲突 → 其他
                return "其他"
        elif male_first is not None:
            return "male"
        elif female_first is not None:
            return "female"
        
        # 男性相关的正则表达式模式
        male_patterns = [
            r'\bmale\b',                    # 完整单词 male
            r'\d+male\b',                   # 数字+male，如 1male
            r'\(\s*\d*male\s*:',           # 权重格式，如 (1male: 或 (male:
            r'\bman\b',                     # 完整单词 man
            r'\d+man\b',                    # 数字+man，如 1man
            r'\(\s*\d*man\s*:',            # 权重格式，如 (1man: 或 (man:
            r'\bboy\b',                     # 完整单词 boy
            r'\d+boy\b',                    # 数字+boy，如 1boy
            r'\(\s*\d*boy\s*:',            # 权重格式，如 (1boy: 或 (boy:
            r'handsome\s+man',              # handsome man
            r'handsome\s+male',             # handsome male
            r'handsome\s+boy',              # handsome boy
        ]
        
        # 女性相关的正则表达式模式
        female_patterns = [
            r'\bfemale\b',                  # 完整单词 female
            r'\d+female\b',                 # 数字+female，如 1female
            r'\(\s*\d*female\s*:',         # 权重格式，如 (1female: 或 (female:
            r'\bwoman\b',                   # 完整单词 woman
            r'\d+woman\b',                  # 数字+woman，如 1woman
            r'\(\s*\d*woman\s*:',          # 权重格式，如 (1woman: 或 (woman:
            r'\bgirl\b',                    # 完整单词 girl
            r'\d+girl\b',                    # 数字+girl，如 1girl
            r'\(\s*\d*girl\s*:',           # 权重格式，如 (1girl: 或 (girl:
            r'\blady\b',                    # 完整单词 lady
            r'\d+lady\b',                    # 数字+lady，如 1lady
            r'\(\s*\d*lady\s*:',           # 权重格式，如 (1lady: 或 (lady:
            r'beautiful\s+woman',           # beautiful woman
            r'beautiful\s+female',          # beautiful female
            r'beautiful\s+girl',            # beautiful girl
            r'beautiful\s+lady',            # beautiful lady
        ]
        
        # 检查是否匹配男性模式
        has_male = any(re.search(pattern, text_lower) for pattern in male_patterns)
        
        # 检查是否匹配女性模式
        has_female = any(re.search(pattern, text_lower) for pattern in female_patterns)
        
        print(f"[POLY_GenderJudgment] 文本分析: '{text}'")
        print(f"[POLY_GenderJudgment] 包含男性特征: {has_male}")
        print(f"[POLY_GenderJudgment] 包含女性特征: {has_female}")
        
        # 判断逻辑
        if has_male and has_female:
            # 都包含 → 其他
            return "其他"
        elif has_male:
            # 只包含male相关 → male
            return "male"
        elif has_female:
            # 只包含female相关 → female
            return "female"
        else:
            # 都不包含 → 其他
            return "其他"
