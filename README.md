# Gemini ComfyUI Plugin

一个强大的 ComfyUI 插件，集成 Google Gemini API 的图像生成和编辑功能。支持纯文本生成图像和基于现有图像的智能编辑。

## ✨ 功能特性

- 🎨 **图像生成**: 使用 Gemini 模型从文本提示生成高质量图像
- 🖼️ **图像编辑**: 基于输入图像和文本指令进行智能编辑
- 🔄 **智能重试**: 内置限流处理和错误恢复机制
- 🚀 **多模型支持**: 支持最新的 Gemini 2.5 和 2.0 模型
- 🛡️ **稳定性**: 增强的错误处理和超时管理

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 获取 API Key

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建新的 API Key
3. 在节点中输入你的 API Key

## 📦 节点说明

### Gemini 图片生成 (GeminiImageGeneration)

从文本提示生成全新图像。

**输入参数:**
- `api_key`: Google Gemini API 密钥
- `prompt`: 图像生成提示词
- `model`: 选择模型 (gemini-2.5-flash-image-preview 或 gemini-2.0-flash-preview-image-generation)
- `temperature`: 创造性控制 (0.0-2.0)
- `top_p`: 采样控制 (0.0-1.0)
- `max_output_tokens`: 最大输出令牌数

**输出:**
- `image`: 生成的图像 (IMAGE 类型)
- `response_text`: API 响应文本

### Gemini 图片编辑 (GeminiImageEdit)

基于输入图像进行智能编辑。

**输入参数:**
- `api_key`: Google Gemini API 密钥
- `image`: 输入图像 (IMAGE 类型)
- `prompt`: 编辑指令
- `model`: 选择模型
- `temperature`: 创造性控制
- `top_p`: 采样控制
- `max_output_tokens`: 最大输出令牌数

**输出:**
- `edited_image`: 编辑后的图像 (IMAGE 类型)
- `response_text`: API 响应文本

## 🔧 技术特性

### 智能重试机制

- **指数退避**: 自动调整重试间隔
- **限流处理**: 针对 429 错误的特殊处理
- **错误分类**: 根据不同错误类型采用不同策略

### 图像处理

- **格式转换**: 自动处理 RGB/RGBA 转换
- **Base64 编码**: 高效的图像数据传输
- **质量优化**: 95% JPEG 质量保证

### 错误处理

- **详细日志**: 完整的操作过程记录
- **友好提示**: 清晰的错误信息和解决建议
- **优雅降级**: 失败时的备用方案

## 📋 依赖项

```
torch>=1.9.0          # 深度学习框架
numpy>=1.21.0         # 数值计算
Pillow>=8.0.0         # 图像处理
requests>=2.25.0      # HTTP 请求
google-genai>=0.3.0   # Google AI SDK
```

## 🛠️ 使用示例

### 基础图像生成

```python
# 在 ComfyUI 中添加 "Gemini 图片生成" 节点
# 设置参数:
api_key = "your_gemini_api_key"
prompt = "A serene mountain landscape at sunset with a lake reflection"
model = "gemini-2.5-flash-image-preview"
```

### 图像编辑

```python
# 在 ComfyUI 中添加 "Gemini 图片编辑" 节点
# 连接输入图像并设置参数:
prompt = "Add a rainbow in the sky"
```

## ⚠️ 注意事项

1. **API 配额**: 注意 Google Gemini API 的使用限制
2. **网络连接**: 确保稳定的网络连接
3. **图像大小**: 大图像可能需要更长处理时间
4. **模型选择**: 不同模型有不同的特性和限制

## 🔍 故障排除

### 常见问题

**API Key 错误**
- 确保 API Key 格式正确且有效
- 检查 API Key 权限设置

**429 限流错误**
- 等待更长时间再试
- 检查 API 配额设置
- 考虑升级 API 计划

**网络超时**
- 检查网络连接
- 增加超时时间设置

## 📄 许可证

本项目遵循开源许可证。使用前请确保遵守 Google Gemini API 的使用条款。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

**注意**: 使用本插件需要有效的 Google Gemini API 密钥。请确保遵守相关的使用条款和限制。