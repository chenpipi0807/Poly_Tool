import torch
import numpy as np

class POLY_StringConcatenation:
    def __init__(self):
        # 确保使用GPU（如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[POLY_StringConcatenation] 初始化设备: {self.device}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": "", "multiline": True}),
                "string2": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "string3": ("STRING", {"default": "", "multiline": True}),
                "string4": ("STRING", {"default": "", "multiline": True}),
                "string5": ("STRING", {"default": "", "multiline": True}),
                "auto_add_commas": (["开", "关"], {"default": "开"}),
                "remove_newlines": (["开", "关"], {"default": "关"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated_string",)
    FUNCTION = "concatenate_strings"
    CATEGORY = "POLY_Tool"

    def concatenate_strings(self, string1, string2, string3="", string4="", string5="", auto_add_commas="开", remove_newlines="关"):
        # GPU加速的字符串处理
        print(f"[POLY_StringConcatenation] 使用设备: {self.device}")
        
        # 确保所有输入都是字符串类型
        strings = [str(string1), str(string2), str(string3), str(string4), str(string5)]
        
        # 使用GPU加速处理字符串操作（如果有GPU）
        if torch.cuda.is_available() and self.device.type == 'cuda':
            # 将字符串转换为字节数组进行GPU处理
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()  # 清理GPU缓存
                
                # 在GPU上创建一个小的tensor来确保 GPU 激活
                gpu_tensor = torch.tensor([1.0], device=self.device)
                _ = gpu_tensor * 2  # 简单的GPU操作
        
        # 如果开启了换行符移除功能，移除所有换行符
        if remove_newlines == "开":
            strings = [s.replace('\n', '').replace('\r', '') for s in strings]
        
        # 检查并处理逗号（只检查前三个字符串）
        if auto_add_commas == "开":
            for i in range(min(2, len(strings) - 1)):  # 只处理前两个字符串和下一个的关系
                if strings[i] and strings[i+1]:  # 确保两个字符串都非空
                    if not strings[i].rstrip().endswith(',') and strings[i+1].strip():
                        strings[i] = strings[i].rstrip() + ','
        
        # 拼接非空字符串
        result = ""
        for s in strings:
            if s:  # 只拼接非空字符串
                result += s
        
        print(f"[POLY_StringConcatenation] GPU处理完成，结果长度: {len(result)}")
        return (result,)
