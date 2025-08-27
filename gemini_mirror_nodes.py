"""
Gemini 镜像站节点
支持自定义API地址，适配国内镜像站和代理服务
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import time
import random
from typing import Optional, Tuple, Dict, Any
import re

try:
    from .utils import (
        tensor_to_pil, pil_to_tensor, image_to_base64, base64_to_image,
        validate_api_key, format_error_message, resize_image_for_api
    )
    from .config import DEFAULT_CONFIG
except ImportError:
    # Fallback utility functions
    def tensor_to_pil(tensor):
        # 处理批次图像：如果是4维张量，取第一张图像
        if len(tensor.shape) == 4:
            # 形状: (batch, height, width, channels) 或 (batch, channels, height, width)
            tensor = tensor[0]  # 取第一张图像
        
        # 处理3维张量
        if len(tensor.shape) == 3:
            # 如果是 (channels, height, width) 格式，转换为 (height, width, channels)
            if tensor.shape[0] == 3 or tensor.shape[0] == 1:
                tensor = tensor.permute(1, 2, 0)
        
        # 确保数据类型正确并转换为PIL图像
        if tensor.dtype != torch.uint8:
            tensor = (tensor * 255).clamp(0, 255).byte()
        
        # 转换为numpy数组
        numpy_array = tensor.cpu().numpy()
        
        # 处理单通道图像
        if numpy_array.shape[-1] == 1:
            numpy_array = numpy_array.squeeze(-1)
            return Image.fromarray(numpy_array, mode='L')
        
        # 处理RGB图像
        return Image.fromarray(numpy_array)
    
    def pil_to_tensor(image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array).unsqueeze(0)
        return tensor
    
    def image_to_base64(image, format='JPEG'):
        buffer = io.BytesIO()
        if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        image.save(buffer, format=format, quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def validate_api_key(api_key):
        return api_key and len(api_key.strip()) > 10
    
    def format_error_message(error):
        return str(error)
    
    DEFAULT_CONFIG = {"timeout": 120, "max_retries": 3}


def validate_api_url(url):
    """验证API URL格式"""
    if not url or not url.strip():
        return False
    
    url = url.strip()
    # 基本URL格式检查
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def build_api_url(base_url, model):
    """构建完整的API URL"""
    base_url = base_url.strip().rstrip('/')
    
    # 如果用户提供的是完整URL，直接使用
    if '/models/' in base_url and ':generateContent' in base_url:
        return base_url
    
    # 如果是基础URL，构建完整路径
    if base_url.endswith('/v1beta') or base_url.endswith('/v1'):
        return f"{base_url}/models/{model}:generateContent"
    
    # 默认添加v1beta路径
    return f"{base_url}/v1beta/models/{model}:generateContent"


def smart_retry_delay(attempt, error_code=None):
    """智能重试延迟 - 根据错误类型调整等待时间"""
    base_delay = 2 ** attempt  # 指数退避
    
    if error_code == 429:  # 限流错误
        rate_limit_delay = 60 + random.uniform(10, 30)  # 60-90秒随机等待
        return max(base_delay, rate_limit_delay)
    elif error_code in [500, 502, 503, 504]:  # 服务器错误
        return base_delay + random.uniform(1, 5)  # 添加随机抖动
    else:
        return base_delay


class GeminiMirrorImageGeneration:
    """Gemini 镜像站图片生成节点 - 支持自定义API地址"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "https://generativelanguage.googleapis.com", 
                    "multiline": False,
                    "placeholder": "输入API地址，如: https://ai.comfly.chat"
                }),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "A beautiful mountain landscape at sunset", "multiline": True}),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation"], {"default": "gemini-2.5-flash-image-preview"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate_image"
    CATEGORY = "Gemini/Mirror"
    
    def generate_image(self, api_url: str, api_key: str, prompt: str, model: str, 
                      temperature: float, top_p: float, max_output_tokens: int) -> Tuple[torch.Tensor, str]:
        """使用镜像站API生成图片"""
        
        # 验证API URL
        if not validate_api_url(api_url):
            raise ValueError("API URL格式无效，请输入有效的URL地址")
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 构建完整的API URL
        full_url = build_api_url(api_url, model)
        print(f"🌐 使用API地址: {full_url}")
        
        # 构建请求数据
        request_data = {
            "contents": [{
                "parts": [{
                    "text": prompt.strip()
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip()
        }
        
        # 智能重试机制
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🎨 正在生成图片... (尝试 {attempt + 1}/{max_retries})")
                print(f"📝 提示词: {prompt[:100]}...")
                print(f"🔗 镜像站: {api_url}")
                
                # 发送请求
                response = requests.post(full_url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    # 解析响应
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取文本响应和图片
                    response_text = ""
                    generated_image = None
                    
                    if "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                # 提取文本
                                if "text" in part:
                                    response_text += part["text"]
                                
                                # 提取图片
                                if "inline_data" in part or "inlineData" in part:
                                    inline_data = part.get("inline_data") or part.get("inlineData")
                                    if inline_data and "data" in inline_data:
                                        try:
                                            # 解码图片数据
                                            image_data = inline_data["data"]
                                            image_bytes = base64.b64decode(image_data)
                                            generated_image = Image.open(io.BytesIO(image_bytes))
                                            print("✅ 成功提取生成的图片")
                                        except Exception as e:
                                            print(f"⚠️ 解码图片失败: {e}")
                    
                    # 如果没有生成图片，创建占位符
                    if generated_image is None:
                        print("⚠️ 未检测到生成的图片，创建占位符")
                        generated_image = Image.new('RGB', (512, 512), color='lightgray')
                        if not response_text:
                            response_text = "图片生成请求已发送，但未收到图片数据"
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(generated_image)
                    
                    print("✅ 图片生成完成")
                    return (image_tensor, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        print(f"❌ 错误文本: {response.text}")
                    
                    # 如果是最后一次尝试，抛出异常
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    
                    # 智能等待
                    delay = smart_retry_delay(attempt, response.status_code)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except requests.exceptions.RequestException as e:
                error_msg = format_error_message(e)
                print(f"❌ 请求失败: {error_msg}")
                if attempt == max_retries - 1:
                    raise ValueError(f"API请求失败: {error_msg}")
                else:
                    delay = smart_retry_delay(attempt)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except Exception as e:
                error_msg = format_error_message(e)
                print(f"❌ 处理失败: {error_msg}")
                raise ValueError(f"图片生成失败: {error_msg}")


class GeminiMirrorImageEdit:
    """Gemini 镜像站图片编辑节点 - 支持自定义API地址"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "https://generativelanguage.googleapis.com", 
                    "multiline": False,
                    "placeholder": "输入API地址，如: https://ai.comfly.chat"
                }),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Can you add a llama next to me?", "multiline": True}),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation"], {"default": "gemini-2.5-flash-image-preview"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "edit_image"
    CATEGORY = "Gemini/Mirror"
    
    def edit_image(self, api_url: str, api_key: str, image: torch.Tensor, prompt: str, model: str,
                   temperature: float, top_p: float, max_output_tokens: int) -> Tuple[torch.Tensor, str]:
        """使用镜像站API编辑图片"""
        
        # 验证API URL
        if not validate_api_url(api_url):
            raise ValueError("API URL格式无效，请输入有效的URL地址")
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 转换输入图片
        pil_image = tensor_to_pil(image)
        
        # 转换为base64
        image_base64 = image_to_base64(pil_image, format='JPEG')
        
        # 构建完整的API URL
        full_url = build_api_url(api_url, model)
        print(f"🌐 使用API地址: {full_url}")
        
        # 构建请求数据 - 包含文本和图片
        request_data = {
            "contents": [{
                "parts": [
                    {
                        "text": prompt.strip()
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip()
        }
        
        # 智能重试机制
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🖼️ 正在编辑图片... (尝试 {attempt + 1}/{max_retries})")
                print(f"📝 编辑指令: {prompt[:100]}...")
                print(f"🔗 镜像站: {api_url}")
                
                # 发送请求
                response = requests.post(full_url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    # 解析响应
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取文本响应和编辑后的图片
                    response_text = ""
                    edited_image = None
                    
                    if "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                # 提取文本
                                if "text" in part:
                                    response_text += part["text"]
                                
                                # 提取编辑后的图片
                                if "inline_data" in part or "inlineData" in part:
                                    inline_data = part.get("inline_data") or part.get("inlineData")
                                    if inline_data and "data" in inline_data:
                                        try:
                                            # 解码图片数据
                                            image_data = inline_data["data"]
                                            image_bytes = base64.b64decode(image_data)
                                            edited_image = Image.open(io.BytesIO(image_bytes))
                                            print("✅ 成功提取编辑后的图片")
                                        except Exception as e:
                                            print(f"⚠️ 解码图片失败: {e}")
                    
                    # 如果没有编辑后的图片，返回原图片
                    if edited_image is None:
                        print("⚠️ 未检测到编辑后的图片，返回原图片")
                        edited_image = pil_image
                        if not response_text:
                            response_text = "图片编辑请求已发送，但未收到编辑后的图片"
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(edited_image)
                    
                    print("✅ 图片编辑完成")
                    return (image_tensor, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        print(f"❌ 错误文本: {response.text}")
                    
                    # 如果是最后一次尝试，抛出异常
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    
                    # 智能等待
                    delay = smart_retry_delay(attempt, response.status_code)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except requests.exceptions.RequestException as e:
                error_msg = format_error_message(e)
                print(f"❌ 请求失败: {error_msg}")
                if attempt == max_retries - 1:
                    raise ValueError(f"API请求失败: {error_msg}")
                else:
                    delay = smart_retry_delay(attempt)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except Exception as e:
                error_msg = format_error_message(e)
                print(f"❌ 处理失败: {error_msg}")
                raise ValueError(f"图片编辑失败: {error_msg}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiMirrorImageGeneration": GeminiMirrorImageGeneration,
    "GeminiMirrorImageEdit": GeminiMirrorImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiMirrorImageGeneration": "Gemini 镜像站图片生成",
    "GeminiMirrorImageEdit": "Gemini 镜像站图片编辑",
}