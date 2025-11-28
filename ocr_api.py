"""
OCR 后端管理器（统一接口）。
- 负责根据配置/环境选择具体后端（目前支持 'easyocr'）
- 提供 `readtext(img)` 接口，返回与 easyocr 兼容的结果列表

优先顺序：
1. 从 `config_loader.get_ai_config()['ocr']['backend']` 读取
2. 环境变量 `OCR_BACKEND`
3. 默认 'easyocr'

如果所选后端不可用，会尝试回退到可用的后端。
"""

import os
import logging
from typing import List, Any

from config_loader import config_loader

# 延迟导入具体适配器
_backend_modules = {
    'easyocr': 'ocr_easyocr',
    # 'paddleocr': 'ocr_paddleocr'  # 未来可添加
}

_current_backend = None
_impl = None


def _choose_backend():
    # 1) 配置优先
    try:
        cfg = config_loader.get_ai_config()
        backend = cfg.get('ocr', {}).get('backend')
    except Exception:
        backend = None

    # 2) 环境变量
    if not backend:
        backend = os.environ.get('OCR_BACKEND')

    # 3) 默认
    if not backend:
        backend = 'easyocr'

    backend = backend.lower()
    return backend


def _load_impl(backend_name=None):
    global _current_backend, _impl
    if _impl is not None and _current_backend == backend_name:
        return _impl
    backend = backend_name or _choose_backend()
    mod_name = _backend_modules.get(backend)
    if not mod_name:
        logging.warning(f"未知 OCR 后端 {backend}，尝试回退到 easyocr")
        backend = 'easyocr'
        mod_name = _backend_modules.get(backend)
    try:
        mod = __import__(mod_name)
        # 检查模块是否有 readtext
        if hasattr(mod, 'readtext'):
            _impl = mod
            _current_backend = backend
            logging.info(f"已加载 OCR 后端: {backend}")
            return _impl
    except Exception as e:
        logging.warning(f"加载 OCR 后端模块 {mod_name} 失败: {e}")
    # 回退：尝试其他可用后端
    for name, mname in _backend_modules.items():
        try:
            mod = __import__(mname)
            if hasattr(mod, 'readtext'):
                _impl = mod
                _current_backend = name
                logging.info(f"回退并加载 OCR 后端: {name}")
                return _impl
        except Exception:
            continue
    logging.error("没有可用的 OCR 后端，OCR 调用将返回空结果")
    _impl = None
    return None


def readtext(img) -> List[Any]:
    """统一的 OCR 调用，返回后端的 readtext 结果或空列表。
    参数 img: numpy 图像（与 easyocr.readtext 接受的相同类型）。
    """
    impl = _load_impl(None)
    if impl is None:
        return []
    try:
        return impl.readtext(img)
    except Exception as e:
        logging.error(f"OCR 后端执行 readtext 失败: {e}")
        return []


def available_backends():
    """列出已实现的后端名称。"""
    return list(_backend_modules.keys())
