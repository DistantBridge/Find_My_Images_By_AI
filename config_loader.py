import json
import os
import sqlite3
from pathlib import Path
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication

class ConfigLoader:
    """配置加载器，用于加载AI和数据库配置"""
    
    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.ai_config = None
        self.sqlite_config = None
        self.load_configs()
    
    def load_configs(self):
        """加载所有配置文件"""
        try:
            # 加载AI配置
            ai_config_path = self.config_dir / "ai_config.json"
            if ai_config_path.exists():
                with open(ai_config_path, 'r', encoding='utf-8') as f:
                    self.ai_config = json.load(f)
            else:
                raise FileNotFoundError(f"AI配置文件不存在: {ai_config_path}")
            
            # 加载SQLite配置
            sqlite_config_path = self.config_dir / "sqlite_config.json"
            if sqlite_config_path.exists():
                with open(sqlite_config_path, 'r', encoding='utf-8') as f:
                    self.sqlite_config = json.load(f)
            else:
                raise FileNotFoundError(f"SQLite配置文件不存在: {sqlite_config_path}")
                
        except Exception as e:
            print(f"加载配置文件时出错: {e}")
            raise
    
    def get_ai_config(self):
        """获取AI配置"""
        if self.ai_config is None:
            raise RuntimeError("AI配置未加载")
        return self.ai_config
    
    def get_sqlite_config(self):
        """获取SQLite配置"""
        if self.sqlite_config is None:
            raise RuntimeError("SQLite配置未加载")
        return self.sqlite_config
    
    def get_openai_client_config(self):
        """获取OpenAI客户端配置"""
        ai_config = self.get_ai_config()
        return {
            'base_url': ai_config['server']['base_url'],
            'api_key': ai_config['server']['api_key']
        }
    
    def get_database_path(self):
        """获取数据库路径，如果未设置则提示用户选择"""
        sqlite_config = self.get_sqlite_config()
        db_path = sqlite_config.get('database_path', '')
        
        if not db_path:
            # 如果路径未设置，弹出文件选择对话框
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            db_path, _ = QFileDialog.getSaveFileName(
                None,
                "选择数据库文件位置",
                str(self.config_dir / "image_database.db"),
                "SQLite数据库文件 (*.db);;所有文件 (*)"
            )
            
            if db_path:
                # 更新配置文件
                sqlite_config['database_path'] = db_path
                self.save_sqlite_config()
            else:
                raise RuntimeError("未选择数据库文件路径")
        
        return db_path
    
    def save_sqlite_config(self):
        """保存SQLite配置到文件"""
        sqlite_config_path = self.config_dir / "sqlite_config.json"
        with open(sqlite_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.sqlite_config, f, indent=4, ensure_ascii=False)
    
    def ensure_database_exists(self):
        """确保数据库文件存在，如果不存在则创建"""
        db_path = self.get_database_path()
        
        if not os.path.exists(db_path):
            # 创建数据库文件
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 创建图片记录表
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
                        is_featured BOOLEAN DEFAULT 0,
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
                    exif_software TEXT,
                    exif_copyright TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            # 创建标签表和映射表（多对多）
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_name TEXT UNIQUE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_tag_map (
                    image_id INTEGER,
                    tag_id INTEGER,
                    PRIMARY KEY (image_id, tag_id),
                    FOREIGN KEY (image_id) REFERENCES image_records(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES image_tags(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
            conn.close()
            print(f"数据库文件已创建: {db_path}")
        
        return db_path
    
    def get_ai_models(self):
        """获取AI模型配置"""
        ai_config = self.get_ai_config()
        return ai_config['models']
    
    def get_ai_params(self):
        """获取AI参数配置"""
        ai_config = self.get_ai_config()
        return ai_config['parameters']
    
    def get_ai_prompts(self):
        """获取AI提示词配置"""
        ai_config = self.get_ai_config()
        return ai_config['prompts']

# 全局配置加载器实例
config_loader = ConfigLoader()
