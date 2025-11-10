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
    os.environ["PYTHONUTF8"] = "on"
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

def describe_image(image_path, prompt):
    """使用AI描述图片内容"""
    try:
        # 使用配置文件创建客户端
        client = create_openai_client()
        models = get_ai_models()
        params = get_ai_params()
        
        # 读取并编码图片
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建多模态消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    }
                ]
            }
        ]
        
        # 调用API
        response = client.chat.completions.create(
            model=models['image_description'],
            messages=messages,
            **params['image_description']
        )
        return response.choices[0].message.content
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
        
        # 进行AI描述（英文）
        ai_description_en = describe_image(image_path, 
            "Please describe in detail the content of this picture, including the scene, objects, colors, people and possible time frame, etc. information.")
        
        # 使用AI翻译描述
        ai_description_zh = translate_text(ai_description_en, from_lang="English", to_lang="Chinese")
        
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
        if connection and connection.is_connected():
            connection.close()

if __name__ == "__main__":
    # 设置配置文件路径
    config_path = "folders_config.json"
    
    # 开始处理
    process_folders(config_path) 