import os
import sqlite3
from datetime import datetime
from config_loader import config_loader

def batch_insert_unprocessed_images(folder_path, folder_short_name):
    """批量插入未处理图片，自动跳过已存在记录"""
    try:
        # 使用配置文件连接数据库
        db_path = config_loader.ensure_database_exists()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 🔽 新增步骤：获取已存在的路径集合 [2,7](@ref)
        cursor.execute("SELECT original_image_path FROM image_records")
        existing_paths = {os.path.normpath(row[0]) for row in cursor.fetchall()}

        # 遍历文件夹并过滤已存在记录
        image_count = 0
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # 🔽 规范化路径格式（解决Windows路径分隔符问题）
                    file_path = os.path.normpath(os.path.join(root, filename))
                    
                    # 🔽 新增条件：跳过已存在的路径 [6,8](@ref)
                    if file_path in existing_paths:
                        print(f"跳过已存在文件: {file_path}")
                        continue
                    
                    # 构造插入语句（参数化查询）
                    sql = """INSERT INTO image_records (
                            image_name, 
                            folder_short_name,
                            original_image_path,
                            is_processed,
                            processed_time,
                            is_featured
                        ) VALUES (?, ?, ?, ?, ?, ?)"""
                    
                    # 执行插入
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute(sql, (
                        os.path.splitext(filename)[0],
                        folder_short_name,
                        file_path,
                        1,
                        current_time,
                        0
                    ))
                    image_count += 1
                    existing_paths.add(file_path)  # 更新本地缓存

        conn.commit()
        print(f"新增 {image_count} 条记录")
        
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def process_all_folders():
    """处理配置文件中的所有文件夹"""
    try:
        import json
        from pathlib import Path
        
        # 读取配置文件
        with open('folders_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        total_new = 0
        
        # 处理每个文件夹
        for folder_path, short_name in config.items():
            folder_path = Path(folder_path).resolve()
            if folder_path.exists() and folder_path.is_dir():
                print(f"处理文件夹: {folder_path}")
                batch_insert_unprocessed_images(str(folder_path), short_name)
                total_new += 1
        
        print(f"所有文件夹处理完成，共处理 {total_new} 个文件夹")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    process_all_folders()