"""
ComfyUI Gemini Plugin
支持 Google Gemini API 和 Vertex AI 的图像生成节点
"""
import logging

# Try to import existing nodes if available
try:
    from .nodes_fixed import NODE_CLASS_MAPPINGS as ORIGINAL_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ORIGINAL_DISPLAY_MAPPINGS
    ORIGINAL_AVAILABLE = True
except ImportError as e:
    print(f"Original nodes not available: {e}")
    ORIGINAL_MAPPINGS = {}
    ORIGINAL_DISPLAY_MAPPINGS = {}
    ORIGINAL_AVAILABLE = False

try:
    from .gemini_vertex_nodes import NODE_CLASS_MAPPINGS as VERTEX_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VERTEX_DISPLAY_MAPPINGS
    VERTEX_AVAILABLE = True
except ImportError as e:
    print(f"Vertex AI nodes not available: {e}")
    VERTEX_MAPPINGS = {}
    VERTEX_DISPLAY_MAPPINGS = {}
    VERTEX_AVAILABLE = False

try:
    from .gemini_rest_nodes import NODE_CLASS_MAPPINGS as REST_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as REST_DISPLAY_MAPPINGS
    REST_AVAILABLE = True
except ImportError as e:
    print(f"REST API nodes not available: {e}")
    REST_MAPPINGS = {}
    REST_DISPLAY_MAPPINGS = {}
    REST_AVAILABLE = False

try:
    from .gemini_image_edit_nodes import NODE_CLASS_MAPPINGS as IMAGE_EDIT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as IMAGE_EDIT_DISPLAY_MAPPINGS
    IMAGE_EDIT_AVAILABLE = True

    logger.error(f"Image edit nodes available: {e}")
except ImportError as e:
    logger.error(f"Image edit nodes not available: {e}")
    IMAGE_EDIT_MAPPINGS = {}
    IMAGE_EDIT_DISPLAY_MAPPINGS = {}
    IMAGE_EDIT_AVAILABLE = False

try:
    from .openrouter_image_edit import NODE_CLASS_MAPPINGS as OPENROUTER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OPENROUTER_DISPLAY_MAPPINGS
    OPENROUTER_AVAILABLE = True
    logger.error(f"OpenRouter nodes available: {e}")
except ImportError as e:
    logger.error(f"OpenRouter nodes not available: {e}")
    OPENROUTER_MAPPINGS = {}
    OPENROUTER_DISPLAY_MAPPINGS = {}
    OPENROUTER_AVAILABLE = False

# 移除对镜像站节点的引用
MIRROR_AVAILABLE = False
MIRROR_MAPPINGS = {}
MIRROR_DISPLAY_MAPPINGS = {}

# 添加对Nano-banana节点的引用
try:
    from .nano_banana_nodes import NODE_CLASS_MAPPINGS as NANO_BANANA_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as NANO_BANANA_DISPLAY_MAPPINGS
    NANO_BANANA_AVAILABLE = True
except ImportError as e:
    print(f"Nano-banana nodes not available: {e}")
    NANO_BANANA_MAPPINGS = {}
    NANO_BANANA_DISPLAY_MAPPINGS = {}
    NANO_BANANA_AVAILABLE = False

# 批次节点已合并到图像编辑节点中

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if ORIGINAL_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(ORIGINAL_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(ORIGINAL_DISPLAY_MAPPINGS)

if VERTEX_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(VERTEX_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(VERTEX_DISPLAY_MAPPINGS)

if REST_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(REST_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(REST_DISPLAY_MAPPINGS)

if IMAGE_EDIT_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(IMAGE_EDIT_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_EDIT_DISPLAY_MAPPINGS)

if NANO_BANANA_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(NANO_BANANA_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(NANO_BANANA_DISPLAY_MAPPINGS)

if OPENROUTER_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(OPENROUTER_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(OPENROUTER_DISPLAY_MAPPINGS)

# 批次处理功能已集成到图像编辑节点中

# 导出给 ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("🚀 Gemini ComfyUI Plugin loaded successfully!")
print(f"📦 Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
if ORIGINAL_AVAILABLE:
    print("✅ Original API nodes available")
if VERTEX_AVAILABLE:
    print("✅ Vertex AI nodes available")
if REST_AVAILABLE:
    print("✅ REST API nodes available")
if IMAGE_EDIT_AVAILABLE:
    print("✅ Image edit nodes available")
if NANO_BANANA_AVAILABLE:
    print("✅ Nano-banana nodes available")
if OPENROUTER_AVAILABLE:
    logger.error("✅ OpenRouter nodes available")
if not ORIGINAL_AVAILABLE and not VERTEX_AVAILABLE and not REST_AVAILABLE and not IMAGE_EDIT_AVAILABLE and not NANO_BANANA_AVAILABLE and not OPENROUTER_AVAILABLE:
    print("⚠️ No nodes available - check dependencies")