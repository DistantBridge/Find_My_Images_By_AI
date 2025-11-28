
# -*- coding: utf-8 -*-
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import sqlite3
import os
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QLabel,
    QDialog, QTextEdit, QScrollArea, QMessageBox, QComboBox, QCheckBox,
    QTabWidget, QProgressBar, QProgressDialog, QFileDialog, QListWidget, QListWidgetItem, QInputDialog,
    QAbstractItemView, QFormLayout
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap, QImage, QFont

import easyocr
from config_loader import config_loader
from describe_image_by_ai import get_exif_data
import hashlib
from ocr_api import available_backends

# ---------------------------
# Database helpers
# ---------------------------

def row_get(row, key, default=None):
    try:
        val = row[key]
        return default if val is None else val
    except Exception:
        return default


def get_database_connection():
    db_path = config_loader.ensure_database_exists()
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def compute_file_hashes(path: str):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                md5.update(chunk)
                sha256.update(chunk)
        md5_hex = md5.hexdigest() if md5 else None
        sha256_hex = sha256.hexdigest() if sha256 else None
        if md5_hex:
            md5_hex = md5_hex.strip().lower()
        if sha256_hex:
            sha256_hex = sha256_hex.strip().lower()
        return md5_hex, sha256_hex
    except Exception:
        return None, None

# ---------------------------
# Table column configuration
# ---------------------------

COLUMN_CONFIG_FILE = "column_config.json"
DEFAULT_COLUMNS = [
    {"key": "id", "label": "ID", "visible": True},
    {"key": "image_name", "label": "图片名称", "visible": True},
    {"key": "folder_short_name", "label": "文件夹", "visible": True},
    {"key": "is_processed", "label": "处理状态", "visible": True},
    {"key": "is_featured", "label": "精选", "visible": True},
    {"key": "tags", "label": "标签", "visible": True},
    {"key": "ocr_text", "label": "OCR文本", "visible": True, "max_length": 20},
    {"key": "ai_description_en", "label": "AI描述（英文）", "visible": True, "max_length": 20},
    {"key": "ai_description_zh", "label": "AI描述（中文）", "visible": True, "max_length": 20},
    {"key": "ai_description_zh_v2", "label": "AI描述（中文v2）", "visible": True, "max_length": 20},
    {"key": "remark", "label": "备注", "visible": True, "max_length": 20},
    {"key": "ai_description_zh_thinking", "label": "AI描述（中文thinking）", "visible": True, "max_length": 20}
]

# ---------------------------
# OpenAI helpers
# ---------------------------

def create_openai_client():
    from openai import OpenAI
    client_config = config_loader.get_openai_client_config()
    return OpenAI(**client_config)

def get_ai_models():
    return config_loader.get_ai_models()

def get_ai_params():
    return config_loader.get_ai_params()

def get_ai_prompts():
    return config_loader.get_ai_prompts()
# ---------------------------
# Subprocess thread (for skip script)
# ---------------------------

class _BaseSubprocessThread(QThread):
    progress_updated = pyqtSignal(str)
    process_finished = pyqtSignal(bool, str)

    def __init__(self, script_path: str):
        super().__init__()
        self.script_path = script_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            cmd = [sys.executable, self.script_path]
            cwd = Path(__file__).parent
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            self.progress_updated.emit(f"启动脚本：{self.script_path}")
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', bufsize=1,
                cwd=str(cwd), env=env
            )
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ''):
                if self._cancel:
                    process.terminate()
                    self.process_finished.emit(False, "处理已取消")
                    return
                cleaned = line.rstrip('\r\n')
                if cleaned:
                    self.progress_updated.emit(cleaned)
            return_code = process.wait()
            if return_code == 0:
                self.process_finished.emit(True, "子进程已完成")
            else:
                self.process_finished.emit(False, f"子进程失败，返回码：{return_code}")
        except Exception as e:
            self.process_finished.emit(False, f"启动/运行失败: {str(e)}")

class SkipProcessThread(_BaseSubprocessThread):
    pass
# ---------------------------
# 扫描新图片线程
# ---------------------------

class ScanImagesThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, int)

    def __init__(self, folders_config: dict):
        super().__init__()
        self.folders_config = folders_config
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            total_new = 0
            items = list(self.folders_config.items())
            total_folders = len(items)
            for fi, (folder_path, short_name) in enumerate(items, start=1):
                if self._cancel:
                    self.finished.emit(False, "扫描已取消", total_new)
                    return
                folder = Path(folder_path).resolve()
                if not folder.exists() or not folder.is_dir():
                    self.progress.emit(int(fi * 100 / max(total_folders, 1)), f"跳过无效文件夹：{short_name}")
                    continue
                image_files = list(folder.rglob('*'))
                total_files = len(image_files) or 1
                for i, p in enumerate(image_files, start=1):
                    if self._cancel:
                        self.finished.emit(False, "扫描已取消", total_new)
                        return
                    if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif', '.bmp'):
                        md5_hex, sha256_hex = compute_file_hashes(str(p))
                        if md5_hex:
                            md5_hex = md5_hex.strip().lower()
                        if sha256_hex:
                            sha256_hex = sha256_hex.strip().lower()
                        found = None
                        if md5_hex and sha256_hex:
                            cursor.execute(
                                "SELECT id, original_image_path, alternate_paths FROM image_records WHERE md5 = ? AND sha256 = ?",
                                (md5_hex, sha256_hex)
                            )
                            found = cursor.fetchone()
                        if found:
                            rec_id = found['id'] if isinstance(found, sqlite3.Row) else found[0]
                            existing_orig = found['original_image_path'] if isinstance(found, sqlite3.Row) else found[1]
                            alt_json = found['alternate_paths'] if isinstance(found, sqlite3.Row) else found[2]
                            try:
                                alt_list = json.loads(alt_json) if alt_json else []
                            except Exception:
                                alt_list = []
                            new_path = str(p)
                            if new_path != existing_orig and new_path not in alt_list:
                                alt_list.append(new_path)
                                cursor.execute("UPDATE image_records SET alternate_paths = ? WHERE id = ?",
                                               (json.dumps(alt_list, ensure_ascii=False), rec_id))
                        else:
                            cursor.execute(
                                """
                                INSERT INTO image_records
                                (image_name, folder_short_name, original_image_path,
                                 md5, sha256, alternate_paths,
                                 processed_time, is_processed, is_featured)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    p.stem, short_name, str(p),
                                    md5_hex, sha256_hex,
                                    json.dumps([], ensure_ascii=False),
                                    None, 0, 0,
                                )
                            )
                            total_new += 1
                    percent = int((i / total_files) * 100)
                    self.progress.emit(percent, f"[{short_name}] 扫描进度 {i}/{total_files}")
            conn.commit()
            cursor.close(); conn.close()
            self.finished.emit(True, "扫描完成", total_new)
        except Exception as e:
            self.finished.emit(False, f"扫描出错：{e}", 0)
# ---------------------------
# 批处理未处理图片线程
# ---------------------------

class ProcessImagesThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, int)

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        processed_count = 0
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, original_image_path FROM image_records WHERE is_processed = ?", (0,))
            rows = cursor.fetchall()
            total = len(rows)
            if total == 0:
                cursor.close(); conn.close()
                self.finished.emit(True, "没有未处理的图片", 0)
                return
            reader = easyocr.Reader(['ch_sim', 'en'])
            prompts = get_ai_prompts()
            for i, row in enumerate(rows, start=1):
                if self._cancel:
                    self.finished.emit(False, "处理已取消", processed_count)
                    return
                record_id = row["id"]
                image_path = Path(row["original_image_path"])
                if not image_path.exists():
                    self.progress.emit(int(i * 100 / total), f"文件不存在，跳过: {image_path.name}")
                    continue
                self.progress.emit(int(i * 100 / total), f"处理 {i}/{total}: {image_path.name}")
                try:
                    img_array = np.fromfile(str(image_path), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError("无法解码图片")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ocr_result = reader.readtext(img)
                    ocr_text = "\n".join([t[1] for t in ocr_result]) if ocr_result else ""
                    ai_cfg = config_loader.get_ai_config()
                    desc_mode = ai_cfg.get('selected_scheme') or ai_cfg.get('description_mode', '')
                    if isinstance(desc_mode, (int, str)) and str(desc_mode).isdigit():
                        scheme = ai_cfg.get('schemes', {}).get(str(desc_mode), {})
                        mode_key = scheme.get('key') if scheme else str(desc_mode)
                    else:
                        mode_key = desc_mode
                    if isinstance(prompts, dict):
                        if mode_key in ('old_en_then_translate', 'en_only'):
                            prompt_text = prompts.get('image_description_en') or prompts.get('image_description')
                        elif mode_key in ('qwen30b_vl', 'qwen32b_vl_thinking'):
                            prompt_text = prompts.get('image_description_zh') or prompts.get('image_description')
                        else:
                            prompt_text = prompts.get('image_description')
                    else:
                        prompt_text = None
                    ai_en = ai_zh = ai_zh_v2 = ai_zh_thinking = ""
                    if mode_key == 'qwen30b_vl':
                        from describe_image_by_ai import describe_image
                        ai_zh_v2 = describe_image(str(image_path), prompt_text, ocr_text=ocr_text)
                    elif mode_key == 'qwen32b_vl_thinking':
                        from describe_image_by_ai import describe_image
                        ai_zh_thinking = describe_image(str(image_path), prompt_text, ocr_text=ocr_text)
                    elif mode_key == 'old_en_then_translate':
                        from describe_image_by_ai import describe_image, translate_text
                        ai_en = describe_image(str(image_path), prompt_text, ocr_text=ocr_text)
                        ai_zh = translate_text(ai_en, from_lang='English', to_lang='Chinese')
                    elif mode_key == 'en_only':
                        from describe_image_by_ai import describe_image
                        ai_en = describe_image(str(image_path), prompt_text, ocr_text=ocr_text)
                    else:
                        from describe_image_by_ai import describe_image
                        ai_en = describe_image(str(image_path), prompt_text, ocr_text=ocr_text)
                    exif = get_exif_data(str(image_path)) or {}
                    cursor.execute(
                        """
                        UPDATE image_records
                        SET ocr_text = ?,
                            ai_description_en = ?,
                            ai_description_zh = ?,
                            ai_description_zh_v2 = ?,
                            ai_description_zh_thinking = ?,
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
                        """,
                        (
                            ocr_text, ai_en, ai_zh, ai_zh_v2, ai_zh_thinking,
                            datetime.now(), 1,
                            exif.get('make'), exif.get('model'), exif.get('datetime'),
                            exif.get('exposure_time'), exif.get('f_number'), exif.get('iso'),
                            exif.get('focal_length'), exif.get('lens_model'),
                            exif.get('gps_latitude'), exif.get('gps_longitude'),
                            exif.get('gps_altitude'), exif.get('image_width'), exif.get('image_height'),
                            exif.get('software'), exif.get('copyright'),
                            record_id,
                        ),
                    )
                    processed_count += 1
                except Exception as e:
                    logging.warning(f"处理图片 {image_path} 时出错: {e}")
                    continue
            conn.commit(); cursor.close(); conn.close()
            self.finished.emit(True, "处理完成", processed_count)
        except Exception as e:
            self.finished.emit(False, f"处理出错：{e}", processed_count)
# ---------------------------
# 哈希线程
# ---------------------------

class ComputeHashesThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, int)

    def __init__(self, mode='missing'):
        super().__init__()
        self.mode = mode
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            if self.mode == 'missing':
                cursor.execute("SELECT id, original_image_path FROM image_records WHERE md5 IS NULL OR md5 = '' OR sha256 IS NULL OR sha256 = ''")
            else:
                cursor.execute("SELECT id, original_image_path FROM image_records")
            rows = cursor.fetchall()
            total = len(rows)
            if total == 0:
                cursor.close(); conn.close()
                self.finished.emit(True, "没有需要处理的记录", 0)
                return
            updated = 0
            for i, r in enumerate(rows, start=1):
                if self._cancel:
                    self.finished.emit(False, "已取消", updated)
                    return
                rec_id = r['id'] if isinstance(r, sqlite3.Row) else r[0]
                path = r['original_image_path'] if isinstance(r, sqlite3.Row) else r[1]
                self.progress.emit(int(i * 100 / max(total, 1)), f"计算哈希 {i}/{total}: {os.path.basename(path)}")
                try:
                    if not path or not os.path.exists(path):
                        continue
                    md5_hex, sha256_hex = compute_file_hashes(path)
                    if md5_hex:
                        md5_hex = md5_hex.strip().lower()
                    if sha256_hex:
                        sha256_hex = sha256_hex.strip().lower()
                    if md5_hex and sha256_hex:
                        cursor.execute("UPDATE image_records SET md5 = ?, sha256 = ? WHERE id = ?", (md5_hex, sha256_hex, rec_id))
                        updated += 1
                except Exception:
                    continue
            conn.commit(); cursor.close(); conn.close()
            self.finished.emit(True, f"哈希计算完成，更新 {updated} 条记录", updated)
        except Exception as e:
            self.finished.emit(False, f"计算哈希失败: {e}", 0)
# ---------------------------
# 详情对话框
# ---------------------------

class ImageDetailDialog(QDialog):
    def __init__(self, record_data, parent=None):
        super().__init__(parent)
        self.record_data = record_data or {}
        if 'exif' not in self.record_data or self.record_data['exif'] is None:
            self.record_data['exif'] = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"图片详情 - {self.record_data.get('image_name','')}")
        self.setMinimumSize(1300, 700)
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        self.image_label = QLabel("预览加载中...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_image()
        left_layout.addWidget(self.image_label)

        right_layout = QVBoxLayout()
        link_layout = QHBoxLayout()
        folder_btn = QPushButton("打开文件夹"); folder_btn.clicked.connect(self.open_folder)
        image_btn = QPushButton("打开图片"); image_btn.clicked.connect(self.open_image)
        alt_btn = QPushButton("备用路径"); alt_btn.clicked.connect(self.open_alternate_paths)
        link_layout.addWidget(folder_btn); link_layout.addWidget(image_btn); link_layout.addWidget(alt_btn)

        self.processed_checkbox = QCheckBox("已处理"); self.processed_checkbox.setChecked(bool(self.record_data.get('is_processed', False)))
        self.featured_checkbox = QCheckBox("精选"); self.featured_checkbox.setChecked(bool(self.record_data.get('is_featured', False)))

        tab_widget = QTabWidget()

        main_path = self.record_data.get('original_image_path') or ""
        all_paths = self.collect_all_paths(main_path, self.record_data.get('alternate_paths'))

        def make_readonly_line(val: str):
            line = QLineEdit(val or "")
            line.setReadOnly(True)
            line.setCursorPosition(0)
            return line

        file_tab = QWidget(); file_layout = QVBoxLayout(file_tab)
        file_form = QFormLayout()
        file_form.setLabelAlignment(Qt.AlignmentFlag.AlignTop)
        file_name = self.record_data.get('image_name') or (Path(main_path).name if main_path else "")
        file_form.addRow("文件名:", make_readonly_line(file_name))
        file_form.addRow("文件大小:", QLabel(self.get_file_size_text(main_path)))
        file_form.addRow("MD5:", make_readonly_line(self.record_data.get('md5') or ""))
        file_form.addRow("SHA256:", make_readonly_line(self.record_data.get('sha256') or ""))
        file_form.addRow("主路径:", make_readonly_line(main_path))
        file_layout.addLayout(file_form)
        file_layout.addWidget(QLabel("所有路径:"))
        self.all_paths_text = QTextEdit("\n".join(all_paths))
        self.all_paths_text.setReadOnly(True)
        self.all_paths_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        if not all_paths:
            self.all_paths_text.setPlaceholderText("暂无可用路径")
        file_layout.addWidget(self.all_paths_text)
        file_layout.addStretch()

        ocr_tab = QWidget(); ocr_layout = QVBoxLayout(ocr_tab)
        ocr_layout.addWidget(QLabel("OCR文本:"))
        self.ocr_edit = QTextEdit(); self.ocr_edit.setText(self.record_data.get('ocr_text') or "")
        ocr_layout.addWidget(self.ocr_edit)

        ai_en_tab = QWidget(); ai_en_layout = QVBoxLayout(ai_en_tab)
        ai_en_layout.addWidget(QLabel("AI描述（英文）:"))
        self.ai_en_edit = QTextEdit(); self.ai_en_edit.setText(self.record_data.get('ai_description_en') or "")
        ai_en_layout.addWidget(self.ai_en_edit)

        ai_zh_tab = QWidget(); ai_zh_layout = QVBoxLayout(ai_zh_tab)
        ai_zh_layout.addWidget(QLabel("AI描述（中文）:"))
        self.ai_zh_edit = QTextEdit(); self.ai_zh_edit.setText(self.record_data.get('ai_description_zh') or "")
        ai_zh_layout.addWidget(self.ai_zh_edit)

        ai_zh_v2_tab = QWidget(); ai_zh_v2_layout = QVBoxLayout(ai_zh_v2_tab)
        ai_zh_v2_layout.addWidget(QLabel("AI描述（中文 v2）:"))
        self.ai_zh_v2_edit = QTextEdit(); self.ai_zh_v2_edit.setText(self.record_data.get('ai_description_zh_v2') or "")
        ai_zh_v2_layout.addWidget(self.ai_zh_v2_edit)

        ai_zh_thinking_tab = QWidget(); ai_zh_th_layout = QVBoxLayout(ai_zh_thinking_tab)
        ai_zh_th_layout.addWidget(QLabel("AI描述（中文 thinking）:"))
        self.ai_zh_thinking_edit = QTextEdit(); self.ai_zh_thinking_edit.setText(self.record_data.get('ai_description_zh_thinking') or "")
        ai_zh_th_layout.addWidget(self.ai_zh_thinking_edit)

        exif_tab = QWidget(); exif_layout = QVBoxLayout(exif_tab)
        exif_layout.addWidget(QLabel("EXIF 信息："))
        self.exif_text = QTextEdit()
        self.exif_text.setReadOnly(True)
        self.exif_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.exif_text.setText(self.format_exif_text(self.record_data.get('exif', {})))
        exif_layout.addWidget(self.exif_text)

        props_tab = QWidget(); props_layout = QVBoxLayout(props_tab)
        props_layout.addWidget(QLabel("标签（逗号分隔）:"))
        self.tags_edit = QLineEdit(",".join(self.record_data.get('tags', [])))
        props_layout.addWidget(self.featured_checkbox)
        props_layout.addWidget(self.tags_edit)
        props_layout.addWidget(QLabel("备注:"))
        self.remark_edit = QTextEdit(); self.remark_edit.setText(self.record_data.get('remark', "") or "")
        props_layout.addWidget(self.remark_edit)

        tab_widget.addTab(file_tab, "文件信息")
        tab_widget.addTab(ocr_tab, "OCR")
        tab_widget.addTab(ai_en_tab, "llava直出")
        tab_widget.addTab(ai_zh_tab, "llava翻译")
        tab_widget.addTab(ai_zh_v2_tab, "中文qwen30b-moe ")
        tab_widget.addTab(ai_zh_thinking_tab, "中文qwen3-vl-32b-thinking")
        tab_widget.addTab(exif_tab, "EXIF")
        tab_widget.addTab(props_tab, "标签/备注")

        save_btn = QPushButton("保存修改"); save_btn.clicked.connect(self.save_changes)
        right_layout.addLayout(link_layout)
        right_layout.addWidget(self.processed_checkbox)
        right_layout.addWidget(tab_widget)
        right_layout.addWidget(save_btn)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

    def format_exif_text(self, exif_data: dict):
        if not isinstance(exif_data, dict):
            return "暂无 EXIF 信息"
        fields = [
            ("make", "设备品牌"),
            ("model", "设备型号"),
            ("lens_model", "镜头型号"),
            ("datetime", "拍摄时间"),
            ("exposure_time", "曝光时间"),
            ("f_number", "光圈"),
            ("iso", "ISO"),
            ("focal_length", "焦距"),
            ("image_width", "宽度"),
            ("image_height", "高度"),
            ("gps_latitude", "GPS 纬度"),
            ("gps_longitude", "GPS 经度"),
            ("gps_altitude", "GPS 海拔"),
            ("software", "软件"),
            ("copyright", "版权"),
        ]
        lines = []
        for key, label in fields:
            val = exif_data.get(key)
            if val is None or val == "":
                continue
            display_val = val
            if key in ("image_width", "image_height"):
                display_val = f"{val} px"
            elif key == "f_number":
                display_val = f"f/{val}"
            elif key == "focal_length":
                display_val = f"{val} mm"
            lines.append(f"{label}: {display_val}")
        return "\n".join(lines) if lines else "暂无 EXIF 信息"

    def get_file_size_text(self, path: str):
        try:
            if path and os.path.exists(path):
                size = os.path.getsize(path)
                units = ["B", "KB", "MB", "GB", "TB"]
                value = float(size)
                for unit in units:
                    if value < 1024 or unit == units[-1]:
                        return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
                    value /= 1024
            return "未知"
        except Exception as e:
            return f"获取失败: {e}"

    def collect_all_paths(self, main_path: str, alternate_paths):
        paths = []
        def add_path(p):
            if not p:
                return
            try:
                p = str(p).strip()
            except Exception:
                return
            if p and p not in paths:
                paths.append(p)
        add_path(main_path)
        if isinstance(alternate_paths, list):
            for p in alternate_paths:
                add_path(p)
        elif alternate_paths:
            add_path(alternate_paths)
        return paths

    def load_image(self):
        try:
            image_path = self.record_data.get('original_image_path')
            if not image_path or not os.path.exists(image_path):
                self.image_label.setText("无效路径")
                return
            img_array = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                self.image_label.setText("无法解码图片")
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            max_size = 400
            if w > h:
                new_w = max_size; new_h = int(h * (max_size / w))
            else:
                new_h = max_size; new_w = int(w * (max_size / h))
            img = cv2.resize(img, (new_w, new_h))
            bytes_per_line = ch * new_w
            q_img = QImage(img.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            self.image_label.setText(f"加载失败: {e}")

    def open_folder(self):
        try:
            folder_path = os.path.dirname(self.record_data['original_image_path'])
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开文件夹失败: {e}")

    def open_image(self):
        try:
            image_path = self.record_data['original_image_path']
            QDesktopServices.openUrl(QUrl.fromLocalFile(image_path))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开图片失败: {e}")

    def open_alternate_paths(self):
        raw_alt = self.record_data.get('alternate_paths') or []
        if not isinstance(raw_alt, list):
            raw_alt = [raw_alt] if raw_alt else []

        paths = []
        for raw in raw_alt:
            try:
                p = str(raw).strip()
            except Exception:
                continue
            if not p:
                continue
            paths.append(p)

        if not paths:
            QMessageBox.information(self, "备用路径", "该图片没有备用路径")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("备用路径")
        dlg.setMinimumSize(720, 320)
        layout = QVBoxLayout(dlg)

        tip_label = QLabel("请选择要打开的备用路径，可切换打开图片或所在文件夹。双击列表可直接打开图片。")
        tip_label.setWordWrap(True)
        layout.addWidget(tip_label)

        list_widget = QListWidget()
        for p in paths:
            item = QListWidgetItem(p)
            if not Path(p).exists():
                item.setForeground(Qt.GlobalColor.red)
                item.setToolTip("路径不存在")
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("打开方式："))
        open_mode = QComboBox()
        open_mode.addItems(["打开图片", "打开所在文件夹"])
        action_row.addWidget(open_mode)
        action_row.addStretch()
        open_btn = QPushButton("打开")
        close_btn = QPushButton("关闭")
        action_row.addWidget(open_btn)
        action_row.addWidget(close_btn)
        layout.addLayout(action_row)

        def do_open(target_item: QListWidgetItem | None):
            item = target_item or list_widget.currentItem()
            if not item:
                QMessageBox.warning(dlg, "提示", "请先选择一条备用路径")
                return
            path = item.text().strip()
            if not path:
                QMessageBox.warning(dlg, "提示", "路径为空，无法打开")
                return
            target = path if open_mode.currentIndex() == 0 else str(Path(path).parent)
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(target))
            except Exception as e:
                QMessageBox.critical(dlg, "错误", f"打开路径失败：{e}")

        open_btn.clicked.connect(lambda: do_open(None))
        close_btn.clicked.connect(dlg.accept)
        list_widget.itemDoubleClicked.connect(lambda item: do_open(item))

        dlg.exec()

    def save_changes(self):
        try:
            conn = get_database_connection(); cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE image_records
                SET ocr_text = ?, ai_description_en = ?, ai_description_zh = ?, ai_description_zh_v2 = ?, ai_description_zh_thinking = ?,
                    remark = ?, is_processed = ?, is_featured = ?
                WHERE id = ?
                """,
                (
                    self.ocr_edit.toPlainText(),
                    self.ai_en_edit.toPlainText(),
                    self.ai_zh_edit.toPlainText(),
                    self.ai_zh_v2_edit.toPlainText(),
                    self.ai_zh_thinking_edit.toPlainText(),
                    self.remark_edit.toPlainText(),
                    int(self.processed_checkbox.isChecked()),
                    int(self.featured_checkbox.isChecked()),
                    self.record_data['id'],
                ),
            )
            tags_text = self.tags_edit.text().strip()
            tags = [t.strip() for t in tags_text.split(',') if t.strip()]
            cursor.execute("DELETE FROM image_tag_map WHERE image_id = ?", (self.record_data['id'],))
            for tag in tags:
                cursor.execute("INSERT OR IGNORE INTO image_tags (tag_name) VALUES (?)", (tag,))
                cursor.execute("SELECT id FROM image_tags WHERE tag_name = ?", (tag,))
                tag_row = cursor.fetchone()
                if tag_row:
                    tag_id = tag_row[0] if not isinstance(tag_row, sqlite3.Row) else tag_row['id']
                    cursor.execute("INSERT OR IGNORE INTO image_tag_map (image_id, tag_id) VALUES (?, ?)", (self.record_data['id'], tag_id))
            conn.commit(); cursor.close(); conn.close()
            QMessageBox.information(self, "成功", "修改已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")
# ---------------------------
# 设置对话框
# ---------------------------

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(700, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("OCR 后端:"))
        self.ocr_combo = QComboBox()
        try:
            backends = available_backends()
        except Exception:
            backends = ['easyocr']
        self.ocr_combo.addItems(backends)
        try:
            cfg = config_loader.get_ai_config()
            cur = cfg.get('ocr', {}).get('backend')
            if cur and cur in backends:
                self.ocr_combo.setCurrentIndex(backends.index(cur))
        except Exception:
            pass
        layout.addWidget(self.ocr_combo)

        layout.addWidget(QLabel("AI 描述方案:"))
        self.desc_mode_combo = QComboBox()
        self.scheme_map = {}
        try:
            cfg = config_loader.get_ai_config()
            schemes = cfg.get('schemes', {})
            for sid, s in schemes.items():
                name = s.get('name') or f"方案 {sid}"
                self.scheme_map[name] = sid
            for name in sorted(self.scheme_map.keys()):
                self.desc_mode_combo.addItem(name)
            cur = cfg.get('selected_scheme') or cfg.get('description_mode')
            if cur:
                for name, sid in self.scheme_map.items():
                    if str(sid) == str(cur):
                        self.desc_mode_combo.setCurrentText(name)
                        break
        except Exception:
            pass
        layout.addWidget(self.desc_mode_combo)

        layout.addWidget(QLabel("需要扫描的文件夹（folders_config.json）:"))
        self.folders_list = QListWidget(); layout.addWidget(self.folders_list)

        row = QHBoxLayout()
        add_btn = QPushButton("添加文件夹"); add_btn.clicked.connect(self.add_folder)
        edit_btn = QPushButton("编辑选中"); edit_btn.clicked.connect(self.edit_selected_folder)
        remove_btn = QPushButton("移除选中"); remove_btn.clicked.connect(self.remove_selected_folder)
        row.addWidget(add_btn); row.addWidget(edit_btn); row.addWidget(remove_btn)
        layout.addLayout(row)

        layout.addWidget(QLabel("AI 配置（ai_config.json，JSON 编辑区）"))
        self.ai_edit = QTextEdit(); layout.addWidget(self.ai_edit)

        save_row = QHBoxLayout()
        save_btn = QPushButton("保存"); save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("取消"); cancel_btn.clicked.connect(self.reject)
        save_row.addStretch(); save_row.addWidget(save_btn); save_row.addWidget(cancel_btn)
        layout.addLayout(save_row)

        self.load_existing()

    def load_existing(self):
        try:
            with open('folders_config.json', 'r', encoding='utf-8') as f:
                folders = json.load(f)
            self.folders_list.clear()
            if isinstance(folders, dict):
                for path, short in folders.items():
                    item = QListWidgetItem(f"{short} || {path}")
                    item.setData(Qt.ItemDataRole.UserRole, (path, short))
                    self.folders_list.addItem(item)
        except Exception:
            pass
        try:
            with open('ai_config.json', 'r', encoding='utf-8') as f:
                full_cfg = json.load(f)
            sanitized = {}
            for key in ('description_mode', 'selected_scheme', 'schemes', 'server', 'models', 'parameters', 'prompts', 'ocr'):
                if key in full_cfg:
                    sanitized[key] = full_cfg[key]
            self.ai_edit.setPlainText(json.dumps(sanitized, indent=4, ensure_ascii=False))
        except Exception:
            self.ai_edit.setPlainText('{}')

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            short, ok = QInputDialog.getText(self, "文件夹简称", "请输入文件夹简称（用于列表显示）")
            if ok and short:
                item = QListWidgetItem(f"{short} || {folder}")
                item.setData(Qt.ItemDataRole.UserRole, (folder, short))
                self.folders_list.addItem(item)

    def remove_selected_folder(self):
        row = self.folders_list.currentRow()
        if row >= 0:
            self.folders_list.takeItem(row)

    def edit_selected_folder(self):
        row = self.folders_list.currentRow()
        if row < 0:
            return
        item = self.folders_list.item(row)
        path, short = item.data(Qt.ItemDataRole.UserRole)
        new_short, ok = QInputDialog.getText(self, "编辑简称", "请输入新的简称:", text=short)
        if ok and new_short:
            item.setText(f"{new_short} || {path}")
            item.setData(Qt.ItemDataRole.UserRole, (path, new_short))

    def save_settings(self):
        try:
            try:
                cfg = config_loader.get_ai_config()
            except Exception:
                cfg = {}
            cfg.setdefault('ocr', {})['backend'] = self.ocr_combo.currentText()
            cur_name = self.desc_mode_combo.currentText()
            if cur_name in self.scheme_map:
                cfg['selected_scheme'] = self.scheme_map[cur_name]
            with open('ai_config.json', 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存 AI 配置失败: {e}")
            return
        try:
            folders = {}
            for i in range(self.folders_list.count()):
                item = self.folders_list.item(i)
                path, short = item.data(Qt.ItemDataRole.UserRole)
                folders[path] = short
            with open('folders_config.json', 'w', encoding='utf-8') as f:
                json.dump(folders, f, indent=4, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存 folders_config.json 失败: {e}")
            return
        QMessageBox.information(self, "提示", "设置已保存，部分改动需重启生效")
        self.accept()
# ---------------------------
# 主窗口
# ---------------------------


# ---------------------------
# 列配置对话框
# ---------------------------

class ColumnConfigDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.columns = columns or []
        self.setWindowTitle("表格列显示设置")
        self.setMinimumSize(480, 520)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        tip = QLabel("使用“上移/下移”调整顺序，勾选控制显示。保存后立即写入 column_config.json 并应用到表格。")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        for col in self.columns:
            item = QListWidgetItem(col.get("label", col.get("key", "")))
            item.setFlags(
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(Qt.CheckState.Checked if col.get("visible", True) else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, col.get("key"))
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        row = QHBoxLayout()
        select_all_btn = QPushButton("全选"); select_all_btn.clicked.connect(self.select_all)
        select_none_btn = QPushButton("全不选"); select_none_btn.clicked.connect(self.select_none)
        row.addWidget(select_all_btn); row.addWidget(select_none_btn); row.addStretch()
        layout.addLayout(row)

        move_row = QHBoxLayout()
        up_btn = QPushButton("上移"); up_btn.clicked.connect(lambda: self.move_selected(-1))
        down_btn = QPushButton("下移"); down_btn.clicked.connect(lambda: self.move_selected(1))
        move_row.addWidget(up_btn); move_row.addWidget(down_btn); move_row.addStretch()
        layout.addLayout(move_row)

        action_row = QHBoxLayout()
        save_btn = QPushButton("保存"); save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消"); cancel_btn.clicked.connect(self.reject)
        action_row.addStretch(); action_row.addWidget(save_btn); action_row.addWidget(cancel_btn)
        layout.addLayout(action_row)

    def select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Checked)

    def select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)

    def move_selected(self, delta: int):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        target = row + delta
        target = max(0, min(target, self.list_widget.count() - 1))
        if target == row:
            return
        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(target, item)
        self.list_widget.setCurrentRow(target)

    def get_config(self):
        new_config = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            new_config.append({
                "key": item.data(Qt.ItemDataRole.UserRole),
                "label": item.text(),
                "visible": item.checkState() == Qt.CheckState.Checked
            })
        return new_config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.progress_dialog = None
        self.scan_thread = None
        self.proc_thread = None
        self.hash_thread = None
        self.auto_process_after_scan = False
        self.default_column_map = {c["key"]: c for c in DEFAULT_COLUMNS}
        self.column_config = self.load_column_config()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("图片记录管理系统")
        self.setMinimumSize(1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        search_layout = QHBoxLayout()
        self.search_type = QComboBox()
        self.search_type.addItems(["全部", "图片名称", "OCR文本", "AI描述", "处理状态", "精选", "标签", "备注"])
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("请输入关键词后回车")
        self.search_input.returnPressed.connect(self.search_records)
        self.exact_match = QCheckBox("精确匹配")

        settings_btn = QPushButton("设置"); settings_btn.clicked.connect(self.open_settings)
        scan_btn = QPushButton("仅扫描新图片"); scan_btn.clicked.connect(self.scan_new_images)
        recompute_btn = QPushButton("重新获取MD5/SHA256"); recompute_btn.clicked.connect(self.open_recompute_hashes_dialog)
        process_btn = QPushButton("处理未处理图片"); process_btn.clicked.connect(self.process_unprocessed_images)
        ai_process_btn = QPushButton("扫描图片并处理"); ai_process_btn.clicked.connect(self.run_ai_image_processing)
        skip_process_btn = QPushButton("跳过处理"); skip_process_btn.clicked.connect(self.run_skip_processing)
        search_btn = QPushButton("搜索"); search_btn.clicked.connect(self.search_records)
        column_btn = QPushButton("列显示/顺序"); column_btn.clicked.connect(self.open_column_config_dialog)

        search_layout.addWidget(self.search_type)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.exact_match)
        search_layout.addWidget(search_btn)
        search_layout.addWidget(scan_btn)
        search_layout.addWidget(recompute_btn)
        search_layout.addWidget(process_btn)
        search_layout.addWidget(ai_process_btn)
        search_layout.addWidget(skip_process_btn)
        search_layout.addWidget(column_btn)
        search_layout.addStretch()
        search_layout.addWidget(settings_btn)

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(self.show_detail)
        self.apply_table_headers(0)

        layout.addLayout(search_layout)
        layout.addWidget(self.table)
        self.load_all_records()
    def truncate_text(self, text, max_length=20):
        if text is None:
            return ''
        text = str(text)
        if not max_length or max_length <= 0:
            return text
        return (text[:max_length] + "...") if len(text) > max_length else text

    def load_column_config(self):
        defaults = [c.copy() for c in DEFAULT_COLUMNS]
        if not os.path.exists(COLUMN_CONFIG_FILE):
            self.save_column_config(defaults)
            return defaults
        try:
            with open(COLUMN_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.normalize_column_config(data)
        except Exception as e:
            logging.warning("加载列配置失败，使用默认列: %s", e)
            self.save_column_config(defaults)
            return defaults

    def save_column_config(self, config):
        try:
            with open(COLUMN_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.warning("写入列配置失败: %s", e)

    def normalize_column_config(self, config):
        default_map = {c["key"]: c for c in DEFAULT_COLUMNS}
        normalized = []
        used_keys = set()
        if isinstance(config, list):
            for item in config:
                if not isinstance(item, dict):
                    continue
                key = item.get("key")
                if key not in default_map:
                    continue
                merged = default_map[key].copy()
                merged["label"] = item.get("label") or merged.get("label")
                merged["visible"] = bool(item.get("visible", merged.get("visible", True)))
                if "max_length" in merged:
                    merged["max_length"] = merged.get("max_length")
                normalized.append(merged)
                used_keys.add(key)
        for col in DEFAULT_COLUMNS:
            if col["key"] not in used_keys:
                normalized.append(col.copy())
        if not normalized:
            normalized = [c.copy() for c in DEFAULT_COLUMNS]
        return normalized

    def get_visible_columns(self):
        cols = [c for c in self.column_config if c.get("visible", True)]
        return cols if cols else [c.copy() for c in DEFAULT_COLUMNS]

    def apply_table_headers(self, record_count: int):
        visible_columns = self.get_visible_columns()
        self.table.setColumnCount(len(visible_columns))
        headers = []
        for idx, col in enumerate(visible_columns):
            label = col.get("label") or col.get("key", "")
            if idx == 0:
                label = f"{label} (找到{record_count}条结果)"
            headers.append(label)
        self.table.setHorizontalHeaderLabels(headers)

    def get_column_definition(self, key: str):
        for col in DEFAULT_COLUMNS:
            if col["key"] == key:
                return col
        return {"key": key, "label": key}

    def get_column_max_length(self, key: str):
        for col in self.column_config:
            if col.get("key") == key and col.get("max_length"):
                return col.get("max_length")
        default_def = self.get_column_definition(key)
        return default_def.get("max_length")

    def build_cell_value(self, key: str, row_data) -> tuple[str, str]:
        try:
            if key == "id":
                value = str(row_data["id"])
            elif key == "image_name":
                value = row_get(row_data, "image_name", "")
            elif key == "folder_short_name":
                value = row_get(row_data, "folder_short_name", "")
            elif key == "is_processed":
                value = "已处理" if int(row_get(row_data, "is_processed", 0) or 0) else "未处理"
            elif key == "is_featured":
                value = "是" if int(row_get(row_data, "is_featured", 0) or 0) else "否"
            elif key == "tags":
                value = row_get(row_data, "tags", "")
            elif key == "ocr_text":
                value = row_get(row_data, "ocr_text", "")
            elif key == "ai_description_en":
                value = row_get(row_data, "ai_description_en", "")
            elif key == "ai_description_zh":
                value = row_get(row_data, "ai_description_zh", "")
            elif key == "ai_description_zh_v2":
                value = row_get(row_data, "ai_description_zh_v2", "")
            elif key == "ai_description_zh_thinking":
                value = row_get(row_data, "ai_description_zh_thinking", "")
            elif key == "remark":
                value = row_get(row_data, "remark", "")
            else:
                value = row_get(row_data, key, "")
        except Exception:
            value = ""
        tooltip = "" if value is None else str(value)
        max_len = self.get_column_max_length(key)
        display = self.truncate_text(value or "", max_len) if max_len else ("" if value is None else str(value))
        if tooltip and display != tooltip:
            return display, tooltip
        return display, tooltip

    def create_table_item(self, text: str, tooltip: str, record_id):
        item = QTableWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, record_id)
        if tooltip:
            item.setToolTip(tooltip)
        return item

    def get_record_id_from_row(self, row: int):
        try:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue
                rid = item.data(Qt.ItemDataRole.UserRole)
                if rid is not None:
                    return str(rid)
        except Exception:
            return None
        return None

    def load_all_records(self):
        self.search_records("")

    def open_column_config_dialog(self):
        dlg = ColumnConfigDialog([c.copy() for c in self.column_config], self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_config = self.normalize_column_config(dlg.get_config())
            self.column_config = new_config
            self.save_column_config(new_config)
            self.search_records(self.search_input.text().strip())
    def search_records(self, search_text=None):
        if search_text is None:
            search_text = self.search_input.text().strip()
        try:
            conn = get_database_connection(); cursor = conn.cursor()
            stype = self.search_type.currentText(); exact = self.exact_match.isChecked()
            def wrap(txt): return txt if exact else f"%{txt}%"
            where, params = [], []
            base_select = """
                SELECT r.*,
                       COALESCE(GROUP_CONCAT(t.tag_name, ', '), '') AS tags
                FROM image_records r
                LEFT JOIN image_tag_map m ON r.id = m.image_id
                LEFT JOIN image_tags t ON t.id = m.tag_id
            """
            if stype == "全部" and search_text:
                op = "=" if exact else "LIKE"
                where.append(f"(r.image_name {op} ? OR r.ocr_text {op} ? OR r.ai_description_en {op} ? OR r.ai_description_zh {op} ? OR r.remark {op} ?)")
                pat = wrap(search_text); params += [pat]*5
            elif stype == "图片名称" and search_text:
                op = "=" if exact else "LIKE"; where.append(f"r.image_name {op} ?"); params.append(wrap(search_text))
            elif stype == "OCR文本" and search_text:
                op = "=" if exact else "LIKE"; where.append(f"r.ocr_text {op} ?"); params.append(wrap(search_text))
            elif stype == "AI描述" and search_text:
                op = "=" if exact else "LIKE"
                where.append(f"(r.ai_description_en {op} ? OR r.ai_description_zh {op} ? OR r.ai_description_zh_v2 {op} ? OR r.ai_description_zh_thinking {op} ?)")
                pat = wrap(search_text); params += [pat]*4
            elif stype == "处理状态" and search_text:
                val = 1 if search_text.lower() in ('true','1','是','已处理','yes') else 0
                where.append("r.is_processed = ?"); params.append(val)
            elif stype == "精选":
                if search_text:
                    val = 1 if search_text.lower() in ('true','1','是','精选','yes') else 0
                else:
                    val = 1
                where.append("r.is_featured = ?"); params.append(val)
            elif stype == "标签" and search_text:
                op = "=" if exact else "LIKE"; where.append(f"t.tag_name {op} ?"); params.append(wrap(search_text))
            elif stype == "备注" and search_text:
                op = "=" if exact else "LIKE"; where.append(f"r.remark {op} ?"); params.append(wrap(search_text))
            sql = base_select + (" WHERE " + " AND ".join(where) if where else "") + " GROUP BY r.id ORDER BY r.id DESC"
            cursor.execute(sql, tuple(params)); records = cursor.fetchall()
            self.table.setSortingEnabled(False)
            self.table.setRowCount(0)
            visible_columns = self.get_visible_columns()
            self.apply_table_headers(len(records))
            for row_idx, r in enumerate(records):
                self.table.insertRow(row_idx)
                record_id = r["id"] if isinstance(r, sqlite3.Row) else r[0]
                for col_idx, col in enumerate(visible_columns):
                    text_val, tooltip = self.build_cell_value(col.get("key"), r)
                    item = self.create_table_item(text_val, tooltip, record_id)
                    self.table.setItem(row_idx, col_idx, item)
            self.table.setSortingEnabled(True)
            cursor.close(); conn.close()
            if len(records) == 0:
                QMessageBox.information(self, "搜索结果", "未找到匹配的记录")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"搜索失败: {e}")
    def show_detail(self, index):
        try:
            row = index.row()
            record_id = self.get_record_id_from_row(row)
            if record_id is None:
                QMessageBox.warning(self, "错误", "无法获取记录ID")
                return
            conn = get_database_connection(); cursor = conn.cursor()
            cursor.execute(
                """
                SELECT r.*,
                       COALESCE(GROUP_CONCAT(t.tag_name, ', '), '') AS tags_concat
                FROM image_records r
                LEFT JOIN image_tag_map m ON r.id = m.image_id
                LEFT JOIN image_tags t ON t.id = m.tag_id
                WHERE r.id = ?
                GROUP BY r.id
                """,
                (record_id,)
            )
            r = cursor.fetchone()
            if r:
                tags = []
                if r["tags_concat"]:
                    tags = [t.strip() for t in str(r["tags_concat"]).split(',') if t.strip()]
                record_data = {
                    'id': r["id"],
                    'image_name': row_get(r, "image_name", ""),
                    'folder_short_name': row_get(r, "folder_short_name", ""),
                    'ocr_text': row_get(r, "ocr_text", ""),
                    'ai_description_en': row_get(r, "ai_description_en", ""),
                    'ai_description_zh': row_get(r, "ai_description_zh", ""),
                    'ai_description_zh_v2': row_get(r, "ai_description_zh_v2", ""),
                    'ai_description_zh_thinking': row_get(r, "ai_description_zh_thinking", ""),
                    'original_image_path': row_get(r, "original_image_path", ""),
                    'md5': row_get(r, "md5", ""),
                    'sha256': row_get(r, "sha256", ""),
                    'remark': row_get(r, "remark", ""),
                    'is_processed': bool(row_get(r, "is_processed", 0)),
                    'is_featured': bool(row_get(r, "is_featured", 0)),
                    'tags': tags,
                    'alternate_paths': [],
                    'exif': {
                        'make': row_get(r, "exif_make", ""),
                        'model': row_get(r, "exif_model", ""),
                        'datetime': row_get(r, "exif_datetime", ""),
                        'exposure_time': row_get(r, "exif_exposure_time", ""),
                        'f_number': row_get(r, "exif_f_number", ""),
                        'iso': row_get(r, "exif_iso", ""),
                        'focal_length': row_get(r, "exif_focal_length", ""),
                        'lens_model': row_get(r, "exif_lens_model", ""),
                        'gps_latitude': row_get(r, "exif_gps_latitude", ""),
                        'gps_longitude': row_get(r, "exif_gps_longitude", ""),
                        'gps_altitude': row_get(r, "exif_gps_altitude", ""),
                        'image_width': row_get(r, "exif_image_width", ""),
                        'image_height': row_get(r, "exif_image_height", ""),
                        'software': row_get(r, "exif_software", ""),
                        'copyright': row_get(r, "exif_copyright", ""),
                    }
                }
                try:
                    ap = row_get(r, 'alternate_paths', '')
                    record_data['alternate_paths'] = json.loads(ap) if ap else []
                except Exception:
                    record_data['alternate_paths'] = []
                cursor.close(); conn.close()
                dlg = ImageDetailDialog(record_data, self); dlg.exec()
            else:
                cursor.close(); conn.close()
                QMessageBox.warning(self, "错误", "未找到记录")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示详情失败: {e}")

    def scan_new_images(self):
        try:
            cfg_path = 'folders_config.json'
            if not os.path.exists(cfg_path):
                QMessageBox.warning(self, "警告", "未找到配置文件 folders_config.json")
                return
            with open(cfg_path, 'r', encoding='utf-8') as f:
                folders = json.load(f)
            if not isinstance(folders, dict) or not folders:
                QMessageBox.warning(self, "警告", "folders_config.json 格式不正确或为空")
                return
            self.progress_dialog = QProgressDialog("正在扫描...", "取消", 0, 100, self)
            self.progress_dialog.setWindowTitle("扫描新图片")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)
            self.scan_thread = ScanImagesThread(folders)
            self.scan_thread.progress.connect(self.on_worker_progress)
            self.scan_thread.finished.connect(self.on_scan_finished)
            self.progress_dialog.canceled.connect(self.scan_thread.cancel)
            self.scan_thread.start()
            self.progress_dialog.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"扫描启动失败: {str(e)}")

    def open_recompute_hashes_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("重新获取 MD5/SHA256")
        dlg.setMinimumSize(400, 150)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("请选择重新计算范围："))
        btn_missing = QPushButton("仅为缺失项计算")
        btn_all = QPushButton("为所有项计算")
        btn_cancel = QPushButton("取消")
        btn_missing.clicked.connect(lambda: (dlg.accept(), self.start_compute_hashes('missing')))
        btn_all.clicked.connect(lambda: (dlg.accept(), self.start_compute_hashes('all')))
        btn_cancel.clicked.connect(dlg.reject)
        row = QHBoxLayout()
        row.addWidget(btn_missing); row.addWidget(btn_all); row.addWidget(btn_cancel)
        layout.addLayout(row)
        dlg.exec()

    def start_compute_hashes(self, mode: str):
        try:
            self.progress_dialog = QProgressDialog("正在计算哈希...", "取消", 0, 100, self)
            self.progress_dialog.setWindowTitle("重新获取 MD5/SHA256")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)
            self.hash_thread = ComputeHashesThread(mode=mode)
            self.hash_thread.progress.connect(self.on_worker_progress)
            self.hash_thread.finished.connect(self.on_recompute_finished)
            self.progress_dialog.canceled.connect(self.hash_thread.cancel)
            self.hash_thread.start()
            self.progress_dialog.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动哈希计算失败: {e}")

    def on_recompute_finished(self, success: bool, message: str, count: int):
        if self.progress_dialog:
            self.progress_dialog.close(); self.progress_dialog = None
        if success:
            QMessageBox.information(self, "完成", message)
            self.load_all_records()
        else:
            QMessageBox.critical(self, "失败", message)

    def on_scan_finished(self, success: bool, message: str, new_count: int):
        if self.progress_dialog:
            self.progress_dialog.close(); self.progress_dialog = None
        if not success:
            if self.auto_process_after_scan:
                self.auto_process_after_scan = False
            QMessageBox.critical(self, "扫描失败", message)
            return
        self.load_all_records()
        if self.auto_process_after_scan:
            self.auto_process_after_scan = False
            self.process_unprocessed_images()
        else:
            QMessageBox.information(self, "扫描完成", f"{message}\n新增 {new_count} 个图片")

    def process_unprocessed_images(self):
        try:
            self.progress_dialog = QProgressDialog("正在处理图片...", "取消", 0, 100, self)
            self.progress_dialog.setWindowTitle("处理图片")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)
            self.proc_thread = ProcessImagesThread()
            self.proc_thread.progress.connect(self.on_worker_progress)
            self.proc_thread.finished.connect(self.on_process_finished)
            self.progress_dialog.canceled.connect(self.proc_thread.cancel)
            self.proc_thread.start()
            self.progress_dialog.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动处理失败: {str(e)}")

    def on_process_finished(self, success: bool, message: str, count: int):
        if self.progress_dialog:
            self.progress_dialog.close(); self.progress_dialog = None
        if success:
            QMessageBox.information(self, "处理完成", f"{message}\n成功处理 {count} 张图片")
            self.load_all_records()
        else:
            QMessageBox.critical(self, "处理失败", message)

    def run_ai_image_processing(self):
        try:
            if not os.path.exists("folders_config.json"):
                QMessageBox.warning(self, "警告", "未找到配置文件 folders_config.json")
                return
            reply = QMessageBox.question(
                self,
                "确认扫描并处理",
                "将先扫描配置文件夹，然后处理所有未处理图片。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.auto_process_after_scan = True
            self.scan_new_images()
        except Exception as e:
            self.auto_process_after_scan = False
            QMessageBox.critical(self, "错误", f"启动扫描并处理失败: {str(e)}")

    def run_skip_processing(self):
        try:
            if not os.path.exists("folders_config.json"):
                QMessageBox.warning(self, "警告", "未找到配置文件 folders_config.json")
                return
            script_path = "skip_describe_image.py"
            if not os.path.exists(script_path):
                QMessageBox.warning(self, "警告", "未找到跳过处理脚本 skip_describe_image.py")
                return
            reply = QMessageBox.question(
                self,
                "确认跳过处理",
                "确定要开始跳过处理吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.progress_dialog = QProgressDialog("正在跳过处理...", "取消", 0, 0, self)
            self.progress_dialog.setWindowTitle("跳过处理")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)
            self.skip_process_thread = SkipProcessThread(script_path)
            self.skip_process_thread.progress_updated.connect(self.update_progress_message)
            self.skip_process_thread.process_finished.connect(self.on_subprocess_finished)
            self.progress_dialog.canceled.connect(self.skip_process_thread.cancel)
            self.skip_process_thread.start()
            self.progress_dialog.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动跳过处理失败: {str(e)}")

    def open_settings(self):
        try:
            dlg = SettingsDialog(self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                try:
                    config_loader.load_configs()
                except Exception:
                    pass
                self.load_all_records()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开设置失败: {e}")

    def on_worker_progress(self, percent: int, status: str):
        if self.progress_dialog:
            if self.progress_dialog.maximum() == 0:
                self.progress_dialog.setRange(0, 100)
            self.progress_dialog.setValue(max(0, min(100, percent)))
            self.progress_dialog.setLabelText(status)

    def update_progress_message(self, message: str):
        if self.progress_dialog:
            self.progress_dialog.setLabelText(message)

    def on_subprocess_finished(self, success: bool, message: str):
        if self.progress_dialog:
            self.progress_dialog.close(); self.progress_dialog = None
        if success:
            QMessageBox.information(self, "处理完成", message)
            self.load_all_records()
        else:
            QMessageBox.critical(self, "处理失败", message)

    def get_image_path(self, row):
        try:
            conn = get_database_connection(); cursor = conn.cursor()
            record_id = self.get_record_id_from_row(row)
            if record_id is None:
                QMessageBox.warning(self, "错误", "无法获取记录ID")
                return None
            cursor.execute("SELECT original_image_path FROM image_records WHERE id = ?", (record_id,))
            r = cursor.fetchone(); cursor.close(); conn.close()
            return r["original_image_path"] if r else None
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取图片路径失败: {str(e)}")
            return None
# ---------------------------
# 入口
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 指定常见中文字体，避免界面显示成问号
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
