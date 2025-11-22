import os
import json
import base64
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
import easyocr
import sqlite3
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import piexif
from config_loader import config_loader

def create_openai_client():
    """创建OpenAI客户端"""
    from openai import OpenAI
    client_config = config_loader.get_openai_client_config()
    return OpenAI(**client_config)

def get_ai_models():
    """获取AI模型配置"""
    return config_loader.get_ai_models()

def get_ai_params():
    """获取AI参数配置"""
    return config_loader.get_ai_params()

if sys.platform.startswith('win'):
    # 在Windows系统启用长路径支持
    # 设置为 '1' 为合法值，避免无效值导致子进程报错
    os.environ["PYTHONUTF8"] = "1"
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

# 初始化OCR读取器
reader = easyocr.Reader(['ch_sim', 'en'])

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 使用配置文件中的数据库连接函数

def create_tables(connection):
    """创建必要的数据表"""
    try:
        cursor = connection.cursor()
        
        # 创建图片信息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT,
                folder_short_name TEXT,
                ocr_text TEXT,
                ai_description_en TEXT,
                ai_description_zh TEXT,
                original_image_path TEXT,
                processed_time TEXT,
                is_processed BOOLEAN DEFAULT 0,
                exif_make TEXT,
                exif_model TEXT,
                exif_datetime TEXT,
                exif_exposure_time TEXT,
                exif_f_number REAL,
                exif_iso INTEGER,
                exif_focal_length REAL,
                exif_lens_model TEXT,
                exif_gps_latitude REAL,
                exif_gps_longitude REAL,
                exif_gps_altitude REAL,
                exif_image_width INTEGER,
                exif_image_height INTEGER,
                                is_processed BOOLEAN DEFAULT 0,
                                is_featured BOOLEAN DEFAULT 0,
                exif_copyright TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        connection.commit()
        logging.info("数据表创建成功")
        
    except Exception as e:
        logging.error(f"创建数据表时出错: {e}")
        raise

def describe_image(image_path, prompt, ocr_text=None):
    """使用AI描述图片内容。

    如果后端是 LM Studio（本地 base_url），不要将图片以 base64 放入 messages；
    而是只发送文本提示，加上文件名与 OCR 文本，避免把 base64 当作长文本硬塞进模型。
    """
    try:
        # 使用配置文件创建客户端
        client = create_openai_client()
        models = get_ai_models()
        params = get_ai_params()

        # 检测后端是否为 LM Studio（常见本地地址或端口）
        try:
            base_url = config_loader.get_ai_config().get('server', {}).get('base_url', '')
        except Exception:
            base_url = ''

        is_lmstudio = False
        if base_url:
            lower = base_url.lower()
            if 'lmstudio' in lower or 'localhost' in lower or ':7272' in lower or '192.168.' in lower:
                is_lmstudio = True

        if is_lmstudio:
            # 首先尝试使用 LM Studio 的 Python SDK（如果已安装）来发送图片（SDK 会正确处理多模态）
            try:
                sdk_resp = try_lmstudio_sdk_describe(image_path, prompt, models.get('image_description'), params.get('image_description', {}))
                if sdk_resp is not None:
                    return sdk_resp
            except Exception as e:
                logging.warning(f"LM Studio SDK 调用失败或不可用，回退到文本-only 分支: {e}")

            # LM Studio REST 目前常把 data:image/... 当文本处理 -> 不发送内联图片
            text_prompt = prompt + "\n\n注意：模型无法直接接收内联图片（后端不支持该字段的视觉处理），"
            text_prompt += "因此请根据下面的文件信息和 OCR 文本推测并详细描述图片内容：\n"
            text_prompt += f"- 文件名: {os.path.basename(image_path)}\n"
            if ocr_text:
                text_prompt += f"- OCR 文本:\n{ocr_text}\n"

            messages = [{"role": "user", "content": text_prompt}]
            response = client.chat.completions.create(
                model=models['image_description'],
                messages=messages,
                **params.get('image_description', {})
            )
            try:
                return response.choices[0].message.content
            except Exception:
                return getattr(response.choices[0].message, 'content', str(response))

        # 否则走传统分支（对兼容 OpenAI 的后端可以发送图片）
        # 读取并编码图片（但应尽量压缩：上层可替换为 compress_image_to_base64）
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 构建消息：将提示与 base64 图片内联到同一文本消息中（仅用于兼容 OpenAI 风格后端）
        user_content = f"{prompt}\n\n[Image] data:image/jpeg;base64,{encoded_image}"

        messages = [{"role": "user", "content": user_content}]

        response = client.chat.completions.create(
            model=models['image_description'],
            messages=messages,
            **params.get('image_description', {})
        )

        try:
            return response.choices[0].message.content
        except Exception:
            return getattr(response.choices[0].message, 'content', str(response))
    except Exception as e:
        logging.error(f"AI描述图片时出错: {e}")
        return ""

def translate_text(text, from_lang="English", to_lang="Chinese"):
    """使用AI翻译文本"""
    try:
        # 使用配置文件创建客户端
        client = create_openai_client()
        models = get_ai_models()
        params = get_ai_params()
        
        prompt = f"请将以下{from_lang}文本翻译成{to_lang}，保持原文的意思和风格:\n\n{text}"
        
        messages = [
            {
                "role": "system", 
                "content": "你是一个专业的翻译助手，擅长将文本从一种语言翻译到另一种语言，同时保持原文的含义、风格和细节。"
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        response = client.chat.completions.create(
            model=models['translation'],
            messages=messages,
            **params['translation']
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"翻译时出错: {e}")
        return ""

def convert_to_degrees(value):
    """将GPS坐标转换为度数"""
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])
    return d + (m / 60.0) + (s / 3600.0)


def try_lmstudio_sdk_describe(image_path, prompt, model_name, model_params):
    """
    尝试使用 LM Studio 的 Python SDK 进行多模态描述（如果 SDK 已安装并且提供了多模态接口）。

    该函数为可选集成：
    - 若检测到可用的 SDK 且调用成功，返回字符串描述。
    - 若 SDK 不存在或调用失败，返回 None（上层会回退到文本-only 分支）。

    注意：不同版本的 LM Studio SDK 名称与 API 可能不同，这里尽量尝试常见的包名与构造，但仍可能
    需要你在本机安装官方 SDK 并根据其文档调整本函数。
    """
    # 尝试导入常见 LM Studio SDK 包名
    sdk_names = [
        'lmstudio',
        'lm_studio',
        'lmclient',
        'lm_client',
        'lm_sdk'
    ]
    for name in sdk_names:
        try:
            sdk = __import__(name)
            logging.info(f"找到 LM Studio SDK 包: {name}")
            break
        except Exception:
            sdk = None
    if sdk is None:
        # SDK 未安装
        logging.info("LM Studio SDK 未安装，无法使用 SDK 发送图片描述。可安装官方 SDK 后启用该路径。")
        return None

    # 下面的调用是一个安全尝试：根据不同 SDK 的类名与方法名尝试构造客户端并发送图片。
    try:
        # 获取 base_url 和 api_key
        cfg = config_loader.get_ai_config().get('server', {})
        base_url = cfg.get('base_url')
        api_key = cfg.get('api_key')

        # 常见 SDK 构造器尝试
        Client = None
        for cls_name in ('Client', 'LMClient', 'LMStudioClient'):
            Client = getattr(sdk, cls_name, None)
            if Client:
                break

        if Client is None:
            # 如果模块提供的是直接工厂函数，也尝试常见名
            for fn_name in ('create_client', 'Client'):
                if hasattr(sdk, fn_name):
                    Client = getattr(sdk, fn_name)
                    break

        if Client is None:
            logging.info('无法在 SDK 模块中定位 Client 类或创建函数，请根据 LM Studio 官方 SDK 文档修改本函数。')
            return None

        # 尝试实例化客户端（如果构造函数参数与预期不同，可能抛异常）
        try:
            client = Client(base_url=base_url, api_key=api_key)
        except Exception:
            try:
                client = Client(api_key=api_key)
            except Exception:
                client = None

        if client is None:
            logging.info('无法构造 LM Studio SDK 客户端实例，放弃 SDK 路径。')
            return None

        # 尝试几种常见的 SDK 接口来发送图片和 prompt
        # 1) chat.completions.create-like 接口（接收 files/attachments）
        try:
            files_arg = {'image': open(image_path, 'rb')}
            # 尝试 chat.completions.create
            if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                resp = client.chat.completions.create(model=model_name, messages=[{'role':'user','content':prompt}], files=files_arg, **(model_params or {}))
                # 尝试解析常见字段
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return str(resp)

            # 2) 尝试通用的 `predict` / `generate` 方法
            for method in ('predict', 'generate', 'chat'):
                fn = getattr(client, method, None)
                if fn:
                    try:
                        resp = fn(model=model_name, prompt=prompt, image=open(image_path, 'rb'), **(model_params or {}))
                        # 返回字符串化内容（具体字段视 SDK 而定）
                        if isinstance(resp, str):
                            return resp
                        if hasattr(resp, 'text'):
                            return resp.text
                        return str(resp)
                    except Exception:
                        continue

        except Exception as e:
            logging.warning(f"尝试通过 SDK 调用多模态接口失败: {e}")
            return None

    except Exception as e:
        logging.warning(f"LM Studio SDK 集成尝试发生异常: {e}")
        return None

    return None

def get_exif_data(image_path):
    """提取图片的EXIF信息"""
    try:
        exif_data = {}
        
        # 使用PIL打开图片
        with Image.open(image_path) as img:
            # 获取基本图片信息
            exif_data['image_width'] = img.width
            exif_data['image_height'] = img.height
            
            # 获取EXIF信息
            if 'exif' in img.info:
                exif_dict = piexif.load(img.info['exif'])
                
                # 提取相机信息
                if piexif.ImageIFD.Make in exif_dict['0th']:
                    exif_data['make'] = exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8', errors='ignore')
                if piexif.ImageIFD.Model in exif_dict['0th']:
                    exif_data['model'] = exif_dict['0th'][piexif.ImageIFD.Model].decode('utf-8', errors='ignore')
                if piexif.ImageIFD.DateTime in exif_dict['0th']:
                    exif_data['datetime'] = exif_dict['0th'][piexif.ImageIFD.DateTime].decode('utf-8', errors='ignore')
                if piexif.ImageIFD.Software in exif_dict['0th']:
                    exif_data['software'] = exif_dict['0th'][piexif.ImageIFD.Software].decode('utf-8', errors='ignore')
                if piexif.ImageIFD.Copyright in exif_dict['0th']:
                    exif_data['copyright'] = exif_dict['0th'][piexif.ImageIFD.Copyright].decode('utf-8', errors='ignore')
                
                # 提取拍摄参数
                if piexif.ExifIFD.ExposureTime in exif_dict['Exif']:
                    exif_data['exposure_time'] = str(exif_dict['Exif'][piexif.ExifIFD.ExposureTime])
                if piexif.ExifIFD.FNumber in exif_dict['Exif']:
                    exif_data['f_number'] = float(exif_dict['Exif'][piexif.ExifIFD.FNumber][0]) / float(exif_dict['Exif'][piexif.ExifIFD.FNumber][1])
                if piexif.ExifIFD.ISOSpeedRatings in exif_dict['Exif']:
                    exif_data['iso'] = exif_dict['Exif'][piexif.ExifIFD.ISOSpeedRatings]
                if piexif.ExifIFD.FocalLength in exif_dict['Exif']:
                    exif_data['focal_length'] = float(exif_dict['Exif'][piexif.ExifIFD.FocalLength][0]) / float(exif_dict['Exif'][piexif.ExifIFD.FocalLength][1])
                
                # 提取GPS信息
                if piexif.GPSIFD.GPSLatitude in exif_dict['GPS']:
                    lat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
                    lat_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef]
                    exif_data['gps_latitude'] = convert_to_degrees(lat) * (-1 if lat_ref == b'S' else 1)
                
                if piexif.GPSIFD.GPSLongitude in exif_dict['GPS']:
                    lon = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
                    lon_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef]
                    exif_data['gps_longitude'] = convert_to_degrees(lon) * (-1 if lon_ref == b'W' else 1)
                
                if piexif.GPSIFD.GPSAltitude in exif_dict['GPS']:
                    alt = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
                    exif_data['gps_altitude'] = float(alt[0]) / float(alt[1])
        
        return exif_data
    except Exception as e:
        logging.error(f"提取EXIF信息时出错: {e}")
        return {}

def process_image(image_path, folder_short_name, connection):
    """处理单个图片并存储到数据库"""
    try:
        # 转换为绝对路径并验证
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            logging.error(f"图片文件不存在: {image_path}")
            return False

        # 获取图片文件名（不含扩展名）
        image_name = image_path.stem
        
        # 检查是否已处理过此图片
        cursor = connection.cursor()
        cursor.execute("SELECT id, is_processed FROM image_records WHERE original_image_path = ?", (str(image_path),))
        existing_record = cursor.fetchone()
        
        if existing_record:
            record_id, is_processed = existing_record
            if is_processed:
                print(f"\033[93m[跳过] {image_path} (已处理)\033[0m")  # 使用黄色显示跳过的文件
                logging.info(f"图片 {image_path} 已处理过，跳过处理")
                return True
            else:
                print(f"\033[93m[重新处理] {image_path} (未完成处理)\033[0m")  # 使用黄色显示重新处理的文件
                logging.info(f"图片 {image_path} 未完成处理，重新处理")
        
        # 使用二进制模式读取图片
        img_bytes = image_path.read_bytes()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            logging.error(f"无法解码图片文件: {image_path}")
            return False
        
        # 进行OCR处理
        ocr_result = reader.readtext(img)
        ocr_text = "\n".join([text[1] for text in ocr_result])
        
        # 直接使用配置中的中文提示生成中文描述（避免先英文再翻译的残留逻辑）
        prompts = config_loader.get_ai_prompts()
        prompt_text = prompts.get('image_description') if isinstance(prompts, dict) else None
        if not prompt_text:
            # 备用提示（中文）
            prompt_text = "请用中文描述这张图片的内容，尽量详细。"

        ai_description_zh = describe_image(image_path, prompt_text, ocr_text=ocr_text)
        ai_description_en = ""
        
        # 获取EXIF信息
        exif_data = get_exif_data(image_path)
        
        # 如果记录已存在，更新记录
        if existing_record:
            cursor.execute("""
                UPDATE image_records 
                SET image_name = ?,
                    folder_short_name = ?,
                    ocr_text = ?,
                    ai_description_en = ?,
                    ai_description_zh = ?,
                    processed_time = ?,
                    is_processed = ?,
                    exif_make = ?,
                    exif_model = ?,
                    exif_datetime = ?,
                    exif_exposure_time = ?,
                    exif_f_number = ?,
                    exif_iso = ?,
                    exif_focal_length = ?,
                    exif_lens_model = ?,
                    exif_gps_latitude = ?,
                    exif_gps_longitude = ?,
                    exif_gps_altitude = ?,
                    exif_image_width = ?,
                    exif_image_height = ?,
                    exif_software = ?,
                    exif_copyright = ?
                WHERE id = ?
            """, (
                image_name,
                folder_short_name,
                ocr_text,
                ai_description_en,
                ai_description_zh,
                datetime.now(),
                True,  # 标记为已处理
                exif_data.get('make'),
                exif_data.get('model'),
                exif_data.get('datetime'),
                exif_data.get('exposure_time'),
                exif_data.get('f_number'),
                exif_data.get('iso'),
                exif_data.get('focal_length'),
                exif_data.get('lens_model'),
                exif_data.get('gps_latitude'),
                exif_data.get('gps_longitude'),
                exif_data.get('gps_altitude'),
                exif_data.get('image_width'),
                exif_data.get('image_height'),
                exif_data.get('software'),
                exif_data.get('copyright'),
                existing_record[0]  # 使用现有记录的ID
            ))
        else:
            # 插入新记录
            cursor.execute("""
                INSERT INTO image_records 
                (image_name, folder_short_name, ocr_text, ai_description_en, 
                 ai_description_zh, original_image_path, processed_time, is_processed,
                 exif_make, exif_model, exif_datetime, exif_exposure_time, 
                 exif_f_number, exif_iso, exif_focal_length, exif_lens_model,
                 exif_gps_latitude, exif_gps_longitude, exif_gps_altitude,
                 exif_image_width, exif_image_height, exif_software, exif_copyright)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_name,
                folder_short_name,
                ocr_text,
                ai_description_en,
                ai_description_zh,
                str(image_path),
                datetime.now(),
                True,  # 标记为已处理
                exif_data.get('make'),
                exif_data.get('model'),
                exif_data.get('datetime'),
                exif_data.get('exposure_time'),
                exif_data.get('f_number'),
                exif_data.get('iso'),
                exif_data.get('focal_length'),
                exif_data.get('lens_model'),
                exif_data.get('gps_latitude'),
                exif_data.get('gps_longitude'),
                exif_data.get('gps_altitude'),
                exif_data.get('image_width'),
                exif_data.get('image_height'),
                exif_data.get('software'),
                exif_data.get('copyright')
            ))
        
        connection.commit()
        
        # 输出成功处理的信息到终端
        print(f"\033[92m[成功] {image_path}\033[0m")  # 使用绿色显示成功信息
        
        logging.info(f"成功处理并存储图片: {image_path}")
        return True
        
    except Exception as e:
        print(f"\033[91m[失败] {image_path} - {str(e)}\033[0m")  # 使用红色显示失败信息
        logging.error(f"处理图片 {image_path} 时出错: {e}")
        return False

def process_folders(config_path):
    """处理配置文件中的所有文件夹"""
    try:
        # 创建数据库连接
        connection = config_loader.ensure_database_exists()
        connection = sqlite3.connect(connection)
        if not connection:
            return
        
        # 创建数据表
        create_tables(connection)
        
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 处理每个文件夹
        for folder_path_str, short_name in config.items():
            # 转换为Path对象并解析路径
            folder_path = Path(folder_path_str).resolve()
            
            logging.info(f"开始处理文件夹: {folder_path}")
            print(f"\n\033[94m{'='*50}\033[0m")
            print(f"\033[94m处理文件夹: {folder_path}\033[0m")
            print(f"\033[94m{'='*50}\033[0m")
            
            # 确保文件夹存在
            if not folder_path.exists():
                print(f"\033[91m[错误] 文件夹不存在: {folder_path}\033[0m")
                logging.error(f"文件夹不存在: {folder_path}")
                continue
            if not folder_path.is_dir():
                print(f"\033[91m[错误] 路径不是目录: {folder_path}\033[0m")
                logging.error(f"路径不是目录: {folder_path}")
                continue
            
            # 递归处理文件夹及其所有子文件夹中的图片
            for img_path in folder_path.rglob('*'):
                if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    process_image(img_path, short_name, connection)
        
        print(f"\n\033[92m{'='*50}\033[0m")
        print(f"\033[92m所有文件夹处理完成！\033[0m")
        print(f"\033[92m{'='*50}\033[0m")
        logging.info("所有文件夹处理完成")
        
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
    finally:
        if connection:
            try:
                connection.close()
            except Exception:
                pass

if __name__ == "__main__":
    # 设置配置文件路径
    config_path = "folders_config.json"
    
    # 开始处理
    process_folders(config_path) 