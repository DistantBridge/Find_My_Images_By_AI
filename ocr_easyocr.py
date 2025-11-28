"""
封装 easyocr 的适配器模块。
提供：
- available(): 检查 easyocr 是否可用
- readtext(img, langs=None): 接收 numpy 图像（BGR 或 RGB），返回与 easyocr.Reader.readtext 相同格式的列表

模块内部按需导入 easyocr，避免在主程序导入时因未安装 easyocr 导致失败。
"""

import logging
from typing import List, Tuple, Any

_reader = None


def available() -> bool:
    try:
        import easyocr  # type: ignore
        return True
    except Exception:
        return False


def _get_reader(langs=None):
    global _reader
    if _reader is not None:
        return _reader
    try:
        import easyocr  # type: ignore
        langs = langs or ['ch_sim', 'en']
        _reader = easyocr.Reader(langs)
        return _reader
    except Exception as e:
        logging.warning(f"easyocr 初始化失败: {e}")
        _reader = None
        return None


def readtext(img, langs=None) -> List[Tuple[Any, str, float]]:
    """读取图片并返回 OCR 结果，行为尽量兼容 easyocr.Reader.readtext 的返回格式。
    如果 easyocr 不可用，则返回空列表。
    """
    reader = _get_reader(langs=langs)
    if reader is None:
        logging.warning("easyocr 未安装或初始化失败，readtext 返回空列表")
        return []
    try:
        return reader.readtext(img)
    except Exception as e:
        logging.error(f"easyocr.readtext 运行错误: {e}")
        return []
