"""
张量处理工具函数
统一处理ComfyUI中的图像张量转换
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将ComfyUI张量转换为PIL图像
    支持批次和单张图像
    
    Args:
        tensor: 输入张量，支持以下格式：
               - (batch, height, width, channels)
               - (height, width, channels) 
               - (channels, height, width)
               - (batch, channels, height, width)
    
    Returns:
        PIL.Image: 转换后的PIL图像（如果是批次，返回第一张）
    """
    
    # 处理批次图像：如果是4维张量，取第一张图像
    if len(tensor.shape) == 4:
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
    if len(numpy_array.shape) == 3 and numpy_array.shape[-1] == 1:
        numpy_array = numpy_array.squeeze(-1)
        return Image.fromarray(numpy_array, mode='L')
    
    # 处理灰度图像
    if len(numpy_array.shape) == 2:
        return Image.fromarray(numpy_array, mode='L')
    
    # 处理RGB图像
    return Image.fromarray(numpy_array)


def batch_tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    """
    将批次张量转换为PIL图像列表
    
    Args:
        tensor: 批次张量 (batch, height, width, channels) 或 (batch, channels, height, width)
    
    Returns:
        List[PIL.Image]: PIL图像列表
    """
    images = []
    
    # 处理4维张量 (batch, height, width, channels) 或 (batch, channels, height, width)
    if len(tensor.shape) == 4:
        batch_size = tensor.shape[0]
        print(f"📊 处理批次张量，批次大小: {batch_size}")
        
        for i in range(batch_size):
            single_tensor = tensor[i]
            
            # 处理通道维度
            if len(single_tensor.shape) == 3:
                if single_tensor.shape[0] == 3 or single_tensor.shape[0] == 1:
                    single_tensor = single_tensor.permute(1, 2, 0)
            
            # 转换数据类型
            if single_tensor.dtype != torch.uint8:
                single_tensor = (single_tensor * 255).clamp(0, 255).byte()
            
            numpy_array = single_tensor.cpu().numpy()
            
            # 处理单通道图像
            if len(numpy_array.shape) == 3 and numpy_array.shape[-1] == 1:
                numpy_array = numpy_array.squeeze(-1)
                images.append(Image.fromarray(numpy_array, mode='L'))
            elif len(numpy_array.shape) == 2:
                images.append(Image.fromarray(numpy_array, mode='L'))
            else:
                images.append(Image.fromarray(numpy_array))
    
    # 处理3维张量（单张图像）
    elif len(tensor.shape) == 3:
        images.append(tensor_to_pil(tensor))
    
    return images


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    将PIL图像转换为ComfyUI张量格式
    
    Args:
        image: PIL图像
    
    Returns:
        torch.Tensor: ComfyUI格式的张量 (1, height, width, channels)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_array).unsqueeze(0)  # 添加批次维度
    return tensor


def get_tensor_info(tensor: torch.Tensor) -> str:
    """
    获取张量的详细信息，用于调试
    
    Args:
        tensor: 输入张量
    
    Returns:
        str: 张量信息字符串
    """
    info = f"形状: {tensor.shape}, 数据类型: {tensor.dtype}"
    
    if len(tensor.shape) == 4:
        batch, height, width, channels = tensor.shape
        info += f", 批次: {batch}, 尺寸: {height}x{width}, 通道: {channels}"
    elif len(tensor.shape) == 3:
        if tensor.shape[0] <= 4:  # 可能是通道在前
            channels, height, width = tensor.shape
            info += f", 通道: {channels}, 尺寸: {height}x{width} (CHW格式)"
        else:  # 可能是通道在后
            height, width, channels = tensor.shape
            info += f", 尺寸: {height}x{width}, 通道: {channels} (HWC格式)"
    
    return info


def normalize_tensor_format(tensor: torch.Tensor) -> torch.Tensor:
    """
    标准化张量格式为ComfyUI期望的格式
    
    Args:
        tensor: 输入张量
    
    Returns:
        torch.Tensor: 标准化后的张量 (batch, height, width, channels)
    """
    
    # 如果是3维张量，添加批次维度
    if len(tensor.shape) == 3:
        # 检查是否是CHW格式
        if tensor.shape[0] <= 4:  # 通道数通常不超过4
            tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
        tensor = tensor.unsqueeze(0)  # 添加批次维度
    
    # 如果是4维张量但通道在第二个位置 (batch, channels, height, width)
    elif len(tensor.shape) == 4 and tensor.shape[1] <= 4:
        tensor = tensor.permute(0, 2, 3, 1)  # BCHW -> BHWC
    
    return tensor