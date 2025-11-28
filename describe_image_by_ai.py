# -*- coding: utf-8 -*-
"""
Helper utilities for describing images via configured AI backends.
Includes timeout/ retry defaults to avoid hanging requests.
"""

import os
import json
import base64
import logging
import sys
from pathlib import Path

import piexif
from PIL import Image

from config_loader import config_loader

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("image_processing.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

if sys.platform.startswith("win"):
    os.environ["PYTHONUTF8"] = "1"
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def create_openai_client():
    from openai import OpenAI

    client_config = config_loader.get_openai_client_config()
    return OpenAI(**client_config)


def get_ai_models():
    return config_loader.get_ai_models()


def get_ai_params():
    return config_loader.get_ai_params()


def describe_image(image_path, prompt, ocr_text=None):
    """
    Describe an image with the configured model.
    - LM Studio: do not send base64 image; send text with filename + OCR.
    - OpenAI-compatible backends: send base64 inline.
    Adds timeout/max_retries defaults to avoid hang.
    """
    try:
        client = create_openai_client()
        models = get_ai_models()
        params = get_ai_params()
        try:
            req_params = params.get("image_description", {}).copy() if isinstance(params.get("image_description"), dict) else {}
        except Exception:
            req_params = {}
        # 默认参数：防挂起 & 兼容 LM Studio 示例
        req_params.setdefault("timeout", 60)       # 请求超时秒
        req_params.setdefault("max_retries", 1)    # 重试次数
        req_params.setdefault("temperature", 0.7)
        req_params.setdefault("max_tokens", -1)
        req_params.setdefault("stream", False)
        # 清理掉 None，避免传入不支持的参数键
        req_params = {k: v for k, v in req_params.items() if v is not None}

        try:
            base_url = config_loader.get_ai_config().get("server", {}).get("base_url", "")
        except Exception:
            base_url = ""
        is_lmstudio = False
        if base_url:
            lower = base_url.lower()
            if "lmstudio" in lower or "localhost" in lower or ":7272" in lower or "192.168." in lower:
                is_lmstudio = True
        if is_lmstudio:
            # LM Studio 不兼容 timeout/max_retries 等参数，过滤掉仅保留安全字段
            allowed = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "stream", "n", "stop", "max_tokens"}
            cleaned = {}
            for k, v in req_params.items():
                if k not in allowed:
                    continue
                if k == "max_tokens" and (v is None or v == -1):
                    # LM Studio 对 -1 可能报错，直接跳过
                    continue
                cleaned[k] = v
            req_params = cleaned

        if is_lmstudio:
            # 优先尝试 vision 格式（image_url + text），失败则回退文本 only
            try:
                with open(image_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode("utf-8")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": f"文件名: {os.path.basename(image_path)}"},
                            {"type": "text", "text": f"OCR:\n{ocr_text or ''}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                            },
                        ],
                    }
                ]
                resp = client.chat.completions.create(
                    model=models.get("image_description") or "qwen3-vl-32b-thinking",
                    messages=messages,
                    **req_params,
                )
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return getattr(resp.choices[0].message, "content", str(resp))
            except Exception as e:
                logging.warning(f"LM Studio vision 发送失败，回退文本: {e}")
                text_prompt = prompt + "\n\n注意：请根据文件名与 OCR 文本推测图片内容。\n"
                text_prompt += f"- 文件名: {os.path.basename(image_path)}\n"
                if ocr_text:
                    text_prompt += f"- OCR 文本:\n{ocr_text}\n"
                messages = [{"role": "user", "content": text_prompt}]
                resp = client.chat.completions.create(
                    model=models.get("image_description") or "qwen3-vl-32b-thinking",
                    messages=messages,
                    **req_params,
                )
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return getattr(resp.choices[0].message, "content", str(resp))

        # OpenAI style: include base64
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
        user_content = f"{prompt}\n\n[Image] data:image/jpeg;base64,{encoded_image}"
        messages = [{"role": "user", "content": user_content}]
        resp = client.chat.completions.create(
            model=models.get("image_description") or "qwen3-vl-32b-thinking",
            messages=messages,
            **req_params,
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            return getattr(resp.choices[0].message, "content", str(resp))
    except Exception as e:
        logging.error(f"AI描述图片时出错: {e}")
        return ""


def translate_text(text, from_lang="English", to_lang="Chinese"):
    """Translate text via configured model, with timeout/ retry defaults."""
    try:
        client = create_openai_client()
        models = get_ai_models()
        params = get_ai_params()
        try:
            trans_params = params.get("translation", {}).copy() if isinstance(params.get("translation"), dict) else {}
        except Exception:
            trans_params = {}
        trans_params.setdefault("timeout", 60)
        trans_params.setdefault("max_retries", 1)

        prompt = f"请将以下{from_lang}文本翻译成{to_lang}，保持原文的意思和风格:\n\n{text}"
        messages = [
            {"role": "system", "content": "你是一个专业的翻译助手，擅长精准翻译并保持原文风格。"},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat.completions.create(
            model=models.get("translation"),
            messages=messages,
            **trans_params,
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            return getattr(resp.choices[0].message, "content", str(resp))
    except Exception as e:
        logging.error(f"翻译时出错: {e}")
        return ""


def convert_to_degrees(value):
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])
    return d + (m / 60.0) + (s / 3600.0)


def get_exif_data(image_path):
    """提取图片 EXIF 关键信息（容错处理）。"""
    try:
        exif_data = {}
        with Image.open(image_path) as img:
            exif_data["image_width"] = img.width
            exif_data["image_height"] = img.height
            if "exif" in img.info:
                exif_dict = piexif.load(img.info["exif"])
                if piexif.ImageIFD.Make in exif_dict["0th"]:
                    exif_data["make"] = exif_dict["0th"][piexif.ImageIFD.Make].decode("utf-8", errors="ignore")
                if piexif.ImageIFD.Model in exif_dict["0th"]:
                    exif_data["model"] = exif_dict["0th"][piexif.ImageIFD.Model].decode("utf-8", errors="ignore")
                if piexif.ImageIFD.DateTime in exif_dict["0th"]:
                    exif_data["datetime"] = exif_dict["0th"][piexif.ImageIFD.DateTime].decode("utf-8", errors="ignore")
                if piexif.ImageIFD.Software in exif_dict["0th"]:
                    exif_data["software"] = exif_dict["0th"][piexif.ImageIFD.Software].decode("utf-8", errors="ignore")
                if piexif.ImageIFD.Copyright in exif_dict["0th"]:
                    exif_data["copyright"] = exif_dict["0th"][piexif.ImageIFD.Copyright].decode("utf-8", errors="ignore")
                if piexif.ExifIFD.ExposureTime in exif_dict["Exif"]:
                    exif_data["exposure_time"] = str(exif_dict["Exif"][piexif.ExifIFD.ExposureTime])
                if piexif.ExifIFD.FNumber in exif_dict["Exif"]:
                    exif_data["f_number"] = float(exif_dict["Exif"][piexif.ExifIFD.FNumber][0]) / float(exif_dict["Exif"][piexif.ExifIFD.FNumber][1])
                if piexif.ExifIFD.ISOSpeedRatings in exif_dict["Exif"]:
                    exif_data["iso"] = exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings]
                if piexif.ExifIFD.FocalLength in exif_dict["Exif"]:
                    exif_data["focal_length"] = float(exif_dict["Exif"][piexif.ExifIFD.FocalLength][0]) / float(exif_dict["Exif"][piexif.ExifIFD.FocalLength][1])
                if piexif.GPSIFD.GPSLatitude in exif_dict["GPS"]:
                    lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
                    lat_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef]
                    exif_data["gps_latitude"] = convert_to_degrees(lat) * (-1 if lat_ref == b"S" else 1)
                if piexif.GPSIFD.GPSLongitude in exif_dict["GPS"]:
                    lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
                    lon_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef]
                    exif_data["gps_longitude"] = convert_to_degrees(lon) * (-1 if lon_ref == b"W" else 1)
                if piexif.GPSIFD.GPSAltitude in exif_dict["GPS"]:
                    alt = exif_dict["GPS"][piexif.GPSIFD.GPSAltitude]
                    exif_data["gps_altitude"] = float(alt[0]) / float(alt[1])
        return exif_data
    except Exception as e:
        logging.error(f"提取EXIF信息时出错: {e}")
        return {}


if __name__ == "__main__":
    print("describe_image_by_ai helper module; run GUI入口 via image_search_gui.py")
