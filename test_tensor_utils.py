"""
测试张量处理工具函数
"""

import torch
import numpy as np
from PIL import Image
from tensor_utils import tensor_to_pil, batch_tensor_to_pil_list, get_tensor_info, normalize_tensor_format


def test_tensor_conversion():
    """测试各种张量格式的转换"""
    
    print("🧪 开始测试张量转换功能...")
    
    # 测试1: 批次张量 (1, 1, 683, 3) - 这是你遇到的错误格式
    print("\n📊 测试1: 批次张量 (1, 1, 683, 3)")
    test_tensor1 = torch.rand(1, 1, 683, 3)
    print(f"原始张量: {get_tensor_info(test_tensor1)}")
    
    try:
        pil_images1 = batch_tensor_to_pil_list(test_tensor1)
        print(f"✅ 成功转换为 {len(pil_images1)} 张PIL图像")
        print(f"第一张图像尺寸: {pil_images1[0].size}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
    
    # 测试2: 标准ComfyUI格式 (batch, height, width, channels)
    print("\n📊 测试2: 标准格式 (2, 512, 512, 3)")
    test_tensor2 = torch.rand(2, 512, 512, 3)
    print(f"原始张量: {get_tensor_info(test_tensor2)}")
    
    try:
        pil_images2 = batch_tensor_to_pil_list(test_tensor2)
        print(f"✅ 成功转换为 {len(pil_images2)} 张PIL图像")
        for i, img in enumerate(pil_images2):
            print(f"第{i+1}张图像尺寸: {img.size}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
    
    # 测试3: 单张图像 (height, width, channels)
    print("\n📊 测试3: 单张图像 (256, 256, 3)")
    test_tensor3 = torch.rand(256, 256, 3)
    print(f"原始张量: {get_tensor_info(test_tensor3)}")
    
    try:
        pil_image3 = tensor_to_pil(test_tensor3)
        print(f"✅ 成功转换为PIL图像，尺寸: {pil_image3.size}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
    
    # 测试4: CHW格式 (channels, height, width)
    print("\n📊 测试4: CHW格式 (3, 128, 128)")
    test_tensor4 = torch.rand(3, 128, 128)
    print(f"原始张量: {get_tensor_info(test_tensor4)}")
    
    try:
        pil_image4 = tensor_to_pil(test_tensor4)
        print(f"✅ 成功转换为PIL图像，尺寸: {pil_image4.size}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
    
    # 测试5: 标准化格式
    print("\n📊 测试5: 格式标准化")
    test_tensors = [
        torch.rand(1, 1, 683, 3),  # 你的问题格式
        torch.rand(2, 3, 512, 512),  # BCHW格式
        torch.rand(256, 256, 3),  # HWC格式
    ]
    
    for i, tensor in enumerate(test_tensors):
        print(f"原始张量{i+1}: {get_tensor_info(tensor)}")
        try:
            normalized = normalize_tensor_format(tensor)
            print(f"标准化后: {get_tensor_info(normalized)}")
        except Exception as e:
            print(f"❌ 标准化失败: {e}")
    
    print("\n🎉 测试完成!")


if __name__ == "__main__":
    test_tensor_conversion()