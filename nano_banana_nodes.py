import os
import json
import base64
import requests
import torch
import numpy as np
import random
from PIL import Image
import io
from typing import List, Dict, Any, Tuple, Optional, Union

# 检查是否安装了所需的库
try:
    import requests
except ImportError:
    print("请安装requests库: pip install requests")

class NanoBananaTextToImage:
    """
    Nano-banana 文生图节点，使用OpenAI DALL-E格式的API
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "一只猫"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_base": ("STRING", {"default": "https://ai.comfly.chat", "multiline": False}),
                "model": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "headers": ("STRING", {"multiline": True, "default": "{}"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "info", "seed")
    FUNCTION = "generate"
    CATEGORY = "Gemini/Nano-banana"
    
    def generate(self, prompt, api_key, api_base, model, response_format, seed=-1, headers=None):
        try:
            # 处理种子
            if seed == -1:
                seed = random.randint(0, 0xffffffffffffffff)
            
            # 准备请求头
            request_headers = {
                "Content-Type": "application/json"
            }
            
            if api_key:
                request_headers["Authorization"] = f"Bearer {api_key}"
            
            # 添加自定义请求头
            if headers:
                try:
                    custom_headers = json.loads(headers)
                    request_headers.update(custom_headers)
                except json.JSONDecodeError:
                    print("警告: 自定义请求头格式无效，应为有效的JSON")
            
            # 准备请求体
            payload = {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "seed": seed
            }
            
            # 发送请求
            endpoint = f"{api_base}/v1/images/generations"
            response = requests.post(endpoint, headers=request_headers, json=payload)
            
            # 检查响应
            if response.status_code != 200:
                error_message = f"API请求失败: {response.status_code} - {response.text}"
                print(error_message)
                return (torch.zeros(1, 1, 1, 3), error_message, seed)
            
            response_data = response.json()
            
            # 处理响应
            if response_format == "url":
                image_url = response_data.get("data", [{}])[0].get("url", "")
                if not image_url:
                    return (torch.zeros(1, 1, 1, 3), "API返回的响应中没有图像URL", seed)
                
                # 下载图像
                img_response = requests.get(image_url)
                if img_response.status_code != 200:
                    return (torch.zeros(1, 1, 1, 3), f"无法下载图像: {img_response.status_code}", seed)
                
                # 转换为PIL图像
                img = Image.open(io.BytesIO(img_response.content))
            else:  # b64_json
                image_b64 = response_data.get("data", [{}])[0].get("b64_json", "")
                if not image_b64:
                    return (torch.zeros(1, 1, 1, 3), "API返回的响应中没有base64图像数据", seed)
                
                # 解码base64图像
                img_data = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_data))
            
            # 转换为RGB模式（如果不是）
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 转换为Tensor
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
            
            # 确保是BCHW格式 [batch, height, width, channels]
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # 交换通道顺序，确保是[batch, height, width, channels]
            if img_tensor.shape[1] == 3:
                img_tensor = img_tensor.permute(0, 2, 3, 1)
            
            return (img_tensor, json.dumps(response_data, indent=2), seed)
            
        except Exception as e:
            error_message = f"生成图像时发生错误: {str(e)}"
            print(error_message)
            return (torch.zeros(1, 1, 1, 3), error_message, seed)


class NanoBananaImageToImage:
    """
    Nano-banana 图生图节点，使用OpenAI DALL-E格式的API
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "一只猫"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_base": ("STRING", {"default": "https://ai.comfly.chat", "multiline": False}),
                "model": ("STRING", {"default": "nano-banana"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "headers": ("STRING", {"multiline": True, "default": "{}"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "info", "seed")
    FUNCTION = "generate"
    CATEGORY = "Gemini/Nano-banana"
    
    def generate(self, images, prompt, api_key, api_base, model, response_format, seed=-1, headers=None):
        try:
            # 处理种子
            if seed == -1:
                seed = random.randint(0, 0xffffffffffffffff)
                
            # 准备请求头
            request_headers = {}
            
            if api_key:
                request_headers["Authorization"] = f"Bearer {api_key}"
            
            # 添加自定义请求头
            if headers:
                try:
                    custom_headers = json.loads(headers)
                    request_headers.update(custom_headers)
                except json.JSONDecodeError:
                    print("警告: 自定义请求头格式无效，应为有效的JSON")
            
            # 获取第一张输入图像
            if len(images.shape) == 4:
                image = images[0]
            else:
                image = images
            
            # 转换为PIL图像
            image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # 将图像转换为字节流
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # 准备multipart/form-data请求
            files = {
                'image': ('image.png', img_byte_arr, 'image/png')
            }
            
            data = {
                'model': model,
                'prompt': prompt,
                'response_format': response_format,
                'seed': str(seed)
            }
            
            # 发送请求
            endpoint = f"{api_base}/v1/images/edits"
            response = requests.post(endpoint, headers=request_headers, data=data, files=files)
            
            # 检查响应
            if response.status_code != 200:
                error_message = f"API请求失败: {response.status_code} - {response.text}"
                print(error_message)
                return (torch.zeros(1, 1, 1, 3), error_message, seed)
            
            response_data = response.json()
            
            # 处理响应
            if response_format == "url":
                image_url = response_data.get("data", [{}])[0].get("url", "")
                if not image_url:
                    return (torch.zeros(1, 1, 1, 3), "API返回的响应中没有图像URL", seed)
                
                # 下载图像
                img_response = requests.get(image_url)
                if img_response.status_code != 200:
                    return (torch.zeros(1, 1, 1, 3), f"无法下载图像: {img_response.status_code}", seed)
                
                # 转换为PIL图像
                img = Image.open(io.BytesIO(img_response.content))
            else:  # b64_json
                image_b64 = response_data.get("data", [{}])[0].get("b64_json", "")
                if not image_b64:
                    return (torch.zeros(1, 1, 1, 3), "API返回的响应中没有base64图像数据", seed)
                
                # 解码base64图像
                img_data = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_data))
            
            # 转换为RGB模式（如果不是）
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 转换为Tensor
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
            
            # 确保是BCHW格式 [batch, height, width, channels]
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # 交换通道顺序，确保是[batch, height, width, channels]
            if img_tensor.shape[1] == 3:
                img_tensor = img_tensor.permute(0, 2, 3, 1)
            
            return (img_tensor, json.dumps(response_data, indent=2), seed)
            
        except Exception as e:
            error_message = f"生成图像时发生错误: {str(e)}"
            print(error_message)
            return (torch.zeros(1, 1, 1, 3), error_message, seed)


# 导出节点
NODE_CLASS_MAPPINGS = {
    "NanoBananaTextToImage": NanoBananaTextToImage,
    "NanoBananaImageToImage": NanoBananaImageToImage,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaTextToImage": "镜像站文生图",
    "NanoBananaImageToImage": "镜像站图生图",
} 