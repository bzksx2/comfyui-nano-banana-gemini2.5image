"""\nOpenRouter 图像编辑节点\n支持单图和多图输入，自动处理批次数据\n"""

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
from openai import OpenAI
import logging
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info
    from .utils import (
        image_to_base64, base64_to_image,
        validate_api_key, format_error_message, resize_image_for_api
    )
    from .config import DEFAULT_CONFIG
except ImportError:
    logger.error("load tensor utils error")
    from tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info
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


class OpenRouterImageEdit:
    """OpenRouter 图像编辑节点 - 支持单图和多图输入"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": "https://openrouter.ai/api/v1", "multiline": False}),
                "site_url": ("STRING", {"default": "https://your-site.com", "multiline": False}),
                "site_name": ("STRING", {"default": "Your Site Name", "multiline": False}),
                "prompt": ("STRING", {"default": "Describe these images and edit them", "multiline": True}),
                "model": ([
                    "google/gemini-2.5-flash-image-preview:free",
                    "google/gemini-2.5-flash-image-preview"
                ], {"default": "google/gemini-2.5-flash-image-preview:free"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "process_mode": ([
                    "first_image_only",
                    "all_images_combined",
                    "each_image_separately"
                ], {"default": "each_image_separately"}),
            },
            "optional": {
                "images": ("IMAGE",),  # 支持批次图像
                "image_urls": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_images"
    CATEGORY = "OpenRouter"
    
    def edit_images(self, api_key: str, base_url: str, site_url: str, site_name: str,
                    prompt: str, model: str, temperature: float, top_p: float,
                    max_tokens: int, process_mode: str, images: Optional[torch.Tensor] = None,
                    image_urls: str = "") -> torch.Tensor:
        """批次处理图像编辑"""
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 验证输入：至少需要一个图像来源
        if images is None and not image_urls.strip():
            raise ValueError("必须提供至少一个图像来源：images 或 image_urls")
        
        # 如果提供了图像张量，转换为PIL图像列表
        pil_images = []
        if images is not None:
            print(f"📊 输入张量信息: {get_tensor_info(images)}")
            pil_images = batch_tensor_to_pil_list(images)
            print(f"🖼️ 转换得到 {len(pil_images)} 张图像")
        
        # 解析image_urls
        url_list = [url.strip() for url in image_urls.split('\n') if url.strip()] if image_urls else []
        if url_list:
            print(f"🔗 提供了 {len(url_list)} 个图像URL")
        
        # 如果只有URL，创建一个空的PIL图像作为占位符
        if not pil_images and url_list:
            pil_images = [Image.new('RGB', (1, 1), (0, 0, 0)) for _ in url_list]
        
        print(f"🔧 处理模式: {process_mode}")
        
        if process_mode == "first_image_only":
            # 只处理第一张图像
            return self._process_single_image(
                api_key, base_url, site_url, site_name, pil_images[0],
                prompt, model, temperature, top_p, max_tokens,
                url_list[0] if url_list else ""
            )[0]
        
        elif process_mode == "all_images_combined":
            # 将所有图像合并发送
            return self._process_combined_images(
                api_key, base_url, site_url, site_name, pil_images,
                prompt, model, temperature, top_p, max_tokens,
                image_urls
            )[0]
        
        elif process_mode == "each_image_separately":
            # 分别处理每张图像，返回所有结果
            return self._process_images_separately(
                api_key, base_url, site_url, site_name, pil_images,
                prompt, model, temperature, top_p, max_tokens,
                image_urls
            )[0]

    def _process_single_image(self, api_key: str, base_url: str, site_url: str, site_name: str,
                             pil_image: Image.Image, prompt: str, model: str,
                             temperature: float, top_p: float,
                             max_tokens: int, image_url: str = "") -> Tuple[torch.Tensor, str]:
        """处理单张图像"""
        
        # 根据是否提供URL决定使用base64还是直接URL
        if image_url and image_url.strip():
            image_data = {"url": image_url.strip()}
        else:
            # 转换为base64
            image_base64 = image_to_base64(pil_image, format='JPEG')
            image_data = {"url": f"data:image/jpeg;base64,{image_base64}"}
        
        # 初始化OpenAI客户端
        client = OpenAI(
            base_url=base_url,
            api_key=api_key.strip()
        )
        
        try:
            # 创建聊天完成请求
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": site_url,
                    "X-Title": site_name,
                },
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt.strip()
                            },
                            {
                                "type": "image_url",
                                "image_url": image_data
                            }
                        ]
                    }
                ],
                temperature=temperature,
                top_p=top_p
                # max_tokens=max_tokens
            )

            choice = completion["choices"][0]
            message = choice["message"]

            if "images" in message and message["images"]:
                image_url = message["images"][0]["image_url"]["url"]
                if image_url.startswith("data:image"):
                    base64_str = image_url.split(",", 1)[1]
                    image_bytes = base64.b64decode(base64_str)
                    edited_image = Image.open(io.BytesIO(image_bytes))
                    print("✅ 成功提取编辑后的图片")
                else:
                    raise ValueError("返回的不是 base64 图像数据")
            else:
                raise ValueError("响应中未找到图像数据")

            image_tensor = pil_to_tensor(edited_image)
            print("✅ 图片处理完成")
            return (image_tensor, '')
            
        except Exception as e:
            error_msg = format_error_message(e)
            print(f"❌ 处理失败: {error_msg}")
            raise ValueError(f"图片处理失败: {error_msg}")
    
    def _process_combined_images(self, api_key: str, base_url: str, site_url: str, site_name: str,
                                pil_images: List[Image.Image], prompt: str, model: str,
                                temperature: float, top_p: float,
                                max_tokens: int, image_urls: str = "") -> Tuple[torch.Tensor, str]:
        """处理多张图像（合并发送）"""
        
        # 初始化OpenAI客户端
        client = OpenAI(
            base_url=base_url,
            api_key=api_key.strip()
        )
        try:
            # 构建消息内容
            content = [
                {
                    "type": "text",
                    "text": prompt.strip()
                }
            ]
            
            # 解析image_urls（如果提供）
            url_list = [url.strip() for url in image_urls.split('\n') if url.strip()] if image_urls else []
            
            # 添加所有图像
            for i, pil_image in enumerate(pil_images):
                if i < len(url_list):
                    # 使用提供的URL
                    image_data = {"url": url_list[i]}
                else:
                    # 使用base64
                    image_base64 = image_to_base64(pil_image, format='JPEG')
                    image_data = {"url": f"data:image/jpeg;base64,{image_base64}"}
                
                content.append({
                    "type": "image_url",
                    "image_url": image_data
                })
                print(f"📎 添加第 {i+1} 张图像到请求中")
            
                # 创建聊天完成请求
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": site_url,
                        "X-Title": site_name,
                    },
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": content
                    }],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                
                choice = completion["choices"][0]
                message = choice["message"]

                if "images" in message and message["images"]:
                    image_url = message["images"][0]["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",", 1)[1]
                        image_bytes = base64.b64decode(base64_str)
                        edited_image = Image.open(io.BytesIO(image_bytes))
                        print("✅ 成功提取编辑后的图片")
                    else:
                        raise ValueError("返回的不是 base64 图像数据")
                else:
                    raise ValueError("响应中未找到图像数据")

                image_tensor = pil_to_tensor(edited_image)
                
                print("✅ 图片处理完成")
                return (image_tensor, "")
            
        except Exception as e:
            error_msg = format_error_message(e)
            print(f"❌ 处理失败: {error_msg}")
            raise ValueError(f"图片处理失败: {error_msg}")
    
    def _process_images_separately(self, api_key: str, base_url: str, site_url: str, site_name: str,
                                  pil_images: List[Image.Image], prompt: str, model: str,
                                  temperature: float, top_p: float,
                                  max_tokens: int, image_urls: str = "") -> Tuple[torch.Tensor, str]:
        """分别处理每张图像"""
        
        all_edited_images = []
        all_responses = []
        
        # 解析image_urls（如果提供）
        url_list = [url.strip() for url in image_urls.split('\n') if url.strip()] if image_urls else []
        
        for i, pil_image in enumerate(pil_images):
            print(f"🔄 处理第 {i+1}/{len(pil_images)} 张图像")
            
            # 为每张图像添加序号到提示词中
            numbered_prompt = f"Image {i+1}/{len(pil_images)}: {prompt}"
            
            # 处理单张图像
            image_url = url_list[i] if i < len(url_list) else ""
            edited_image, response = self._process_single_image(
                api_key, base_url, site_url, site_name, pil_image, numbered_prompt,
                model, temperature, top_p, max_tokens, image_url
            )
            
            all_edited_images.append(edited_image)
            all_responses.append(f"Image {i+1}/{len(pil_images)}:\n{response}\n")
        
        # 合并所有编辑后的图像和响应
        if len(all_edited_images) > 1:
            combined_images = torch.cat(all_edited_images, dim=0)
        else:
            combined_images = all_edited_images[0]
        
        combined_response = "\n---\n".join(all_responses)
        
        return combined_images


# 节点映射
NODE_CLASS_MAPPINGS = {
    "OpenRouterImageEdit": OpenRouterImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterImageEdit": "OpenRouter 图像编辑",
}