"""
ComfyUI Gemini Plugin
支持 Google Gemini API 和 Vertex AI 的图像生成节点
"""

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
except ImportError as e:
    print(f"Image edit nodes not available: {e}")
    IMAGE_EDIT_MAPPINGS = {}
    IMAGE_EDIT_DISPLAY_MAPPINGS = {}
    IMAGE_EDIT_AVAILABLE = False

try:
    from .openrouter_image_edit import NODE_CLASS_MAPPINGS as OPENROUTER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OPENROUTER_DISPLAY_MAPPINGS
    OPENROUTER_AVAILABLE = True
    print("✅ OpenRouter nodes available")
except ImportError as e:
    print(f"OpenRouter nodes not available: {e}")
    OPENROUTER_MAPPINGS = {}
    OPENROUTER_DISPLAY_MAPPINGS = {}
    OPENROUTER_AVAILABLE = False

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if OPENROUTER_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(OPENROUTER_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(OPENROUTER_DISPLAY_MAPPINGS)

# 导出给 ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("🚀 OpenRouter ComfyUI Plugin loaded successfully!")
print(f"📦 Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
if OPENROUTER_AVAILABLE:
    print("✅ OpenRouter nodes available")
else:
    print("⚠️ No nodes available - check dependencies")