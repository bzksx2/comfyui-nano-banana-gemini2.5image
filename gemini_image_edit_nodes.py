"""
Gemini 图像编辑节点
支持单图和多图输入，自动处理批次数据
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
from typing import Optional, Tuple, Dict, Any, List

try:
    from .tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info
    from .utils import (
        image_to_base64, base64_to_image,
        validate_api_key, format_error_message, resize_image_for_api
    )
    from .config import DEFAULT_CONFIG
except ImportError:
    from .tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info
    # Fallback utility functions - 如果无法导入，使用内置版本
    pass
    
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


def smart_retry_delay(attempt, error_code=None):
    """智能重试延迟"""
    base_delay = 2 ** attempt
    
    if error_code == 429:
        rate_limit_delay = 60 + random.uniform(10, 30)
        return max(base_delay, rate_limit_delay)
    elif error_code in [500, 502, 503, 504]:
        return base_delay + random.uniform(1, 5)
    else:
        return base_delay


class GeminiImageEdit:
    """Gemini 图像编辑节点 - 支持单图和多图输入"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_host": ("STRING", {"default": "https://generativelanguage.googleapis.com", "multiline": False}),
                "images": ("IMAGE",),  # 支持批次图像
                "prompt": ("STRING", {"default": "Describe these images and edit them", "multiline": True}),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation"], {"default": "gemini-2.0-flash-preview-image-generation"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "process_mode": (["first_image_only", "all_images_combined", "each_image_separately"], {"default": "each_image_separately"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "edit_images"
    CATEGORY = "Gemini"
    
    def edit_images(self, api_key: str, api_host: str, images: torch.Tensor, prompt: str, model: str,
                   temperature: float, top_p: float, max_output_tokens: int, process_mode: str) -> Tuple[torch.Tensor, str]:
        """批次处理图像编辑"""
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        print(f"📊 输入张量信息: {get_tensor_info(images)}")
        print(f"🔧 处理模式: {process_mode}")
        
        # 转换为PIL图像列表
        pil_images = batch_tensor_to_pil_list(images)
        print(f"🖼️ 转换得到 {len(pil_images)} 张图像")
        
        if process_mode == "first_image_only":
            # 只处理第一张图像
            return self._process_single_image(api_key, api_host, pil_images[0], prompt, model, temperature, top_p, max_output_tokens)
        
        elif process_mode == "all_images_combined":
            # 将所有图像合并发送
            return self._process_combined_images(api_key, api_host, pil_images, prompt, model, temperature, top_p, max_output_tokens)
        
        elif process_mode == "each_image_separately":
            # 分别处理每张图像，返回所有结果
            return self._process_images_separately(api_key, api_host, pil_images, prompt, model, temperature, top_p, max_output_tokens)    

    def _process_single_image(self, api_key: str, api_host: str, pil_image: Image.Image, prompt: str, model: str,
                             temperature: float, top_p: float, max_output_tokens: int) -> Tuple[torch.Tensor, str]:
        """处理单张图像"""
        
        # 转换为base64
        image_base64 = image_to_base64(pil_image, format='JPEG')
        
        # 构建API URL
        url = f"{api_host}"
        
        # 构建请求数据 - 更新为匹配官方示例的格式
        request_data = {
            "contents": [{
                "parts": [
                    {"text": prompt.strip()},
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
                "maxOutputTokens": max_output_tokens
            }
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip()
        }
        
        # 发送请求并处理响应
        return self._send_request_and_process(url, headers, request_data, pil_image, model)
    
    def _process_combined_images(self, api_key: str, api_host: str, pil_images: List[Image.Image], prompt: str, model: str,
                                temperature: float, top_p: float, max_output_tokens: int) -> Tuple[torch.Tensor, str]:
        """处理多张图像（合并发送）"""
        
        # 构建包含多张图像的请求
        parts = [{"text": prompt.strip()}]
        
        # 添加所有图像
        for i, pil_image in enumerate(pil_images):
            image_base64 = image_to_base64(pil_image, format='JPEG')
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            })
            print(f"📎 添加第 {i+1} 张图像到请求中")
        
        # 构建API URL
        url = f"{api_host}"
        
        # 构建请求数据 - 更新为匹配官方示例的格式
        request_data = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_output_tokens
            }
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip()
        }
        
        # 发送请求并处理响应
        return self._send_request_and_process(url, headers, request_data, pil_images[0], model)
    
    def _process_images_separately(self, api_key: str, api_host: str, pil_images: List[Image.Image], prompt: str, model: str,
                                  temperature: float, top_p: float, max_output_tokens: int) -> Tuple[torch.Tensor, str]:
        """分别处理每张图像"""
        
        all_edited_images = []
        all_responses = []
        
        for i, pil_image in enumerate(pil_images):
            print(f"🔄 处理第 {i+1}/{len(pil_images)} 张图像")
            
            # 为每张图像添加序号到提示词中
            numbered_prompt = f"Image {i+1}/{len(pil_images)}: {prompt}"
            
            # 处理单张图像
            edited_image, response = self._process_single_image(
                api_key, api_host, pil_image, numbered_prompt, model,
                temperature, top_p, max_output_tokens
            )
            
            all_edited_images.append(edited_image)
            all_responses.append(f"Image {i+1}/{len(pil_images)}:\n{response}\n")
        
        # 合并所有编辑后的图像和响应
        if len(all_edited_images) > 1:
            combined_images = torch.cat(all_edited_images, dim=0)
        else:
            combined_images = all_edited_images[0]
        
        combined_response = "\n---\n".join(all_responses)
        
        return combined_images, combined_response
    
    def _send_request_and_process(self, url: str, headers: dict, request_data: dict, 
                                 fallback_image: Image.Image, model: str) -> Tuple[torch.Tensor, str]:
        """发送请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🖼️ 正在处理图像... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                
                # 发送请求
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
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
                                            image_data = inline_data["data"]
                                            image_bytes = base64.b64decode(image_data)
                                            edited_image = Image.open(io.BytesIO(image_bytes))
                                            print("✅ 成功提取编辑后的图片")
                                        except Exception as e:
                                            print(f"⚠️ 解码图片失败: {e}")
                    
                    # 如果没有编辑后的图片，返回原图片
                    if edited_image is None:
                        print("⚠️ 未检测到编辑后的图片，返回原图片")
                        edited_image = fallback_image
                        if not response_text:
                            response_text = "图片处理请求已发送，但未收到编辑后的图片"
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(edited_image)
                    
                    print("✅ 图片处理完成")
                    return (image_tensor, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        print(f"❌ 错误文本: {response.text}")
                    
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    
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
                raise ValueError(f"图片处理失败: {error_msg}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiImageEdit": GeminiImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEdit": "Gemini 图像编辑",
}