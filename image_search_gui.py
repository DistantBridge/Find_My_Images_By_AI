import sys
import json
import base64
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
    QTabWidget, QProgressBar, QProgressDialog
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap, QImage

import easyocr
from config_loader import config_loader
from AI_Image_Process_To_Mysql import get_exif_data


# ---------------------------
# Database helpers
# ---------------------------


# --- 放在 import 之后、其它代码之前 ---
def row_get(row, key, default=None):
    """安全从 sqlite3.Row 取值；不存在或为 None 时给默认值"""
    try:
        val = row[key]
        return default if val is None else val
    except Exception:
        return default







def get_database_connection():
    """获取数据库连接（开启 WAL、设置超时、Row 映射）"""
    db_path = config_loader.ensure_database_exists()
    # WAL 模式 + 超时，缓解并发写锁冲突
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------
# OpenAI helpers
# ---------------------------
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

def get_ai_prompts():
    """获取AI提示词配置"""
    return config_loader.get_ai_prompts()


# ---------------------------
# 子进程线程（合并 stderr，支持取消）
# ---------------------------
class _BaseSubprocessThread(QThread):
    progress_updated = pyqtSignal(str)       # 日志文本
    process_finished = pyqtSignal(bool, str) # 成功/失败, 信息

    def __init__(self, script_path: str):
        super().__init__()
        self.script_path = script_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.progress_updated.emit(f"启动脚本：{self.script_path}")
            process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并，避免阻塞
                text=True,
                encoding='utf-8',
                bufsize=1
            )

            # 实时读取输出
            assert process.stdout is not None
            for line in process.stdout:
                if self._cancel:
                    self.progress_updated.emit("收到取消请求，终止子进程...")
                    process.terminate()
                    process.wait(timeout=5)
                    self.process_finished.emit(False, "处理已取消")
                    return
                self.progress_updated.emit(line.rstrip())

            return_code = process.wait()

            if return_code == 0:
                self.process_finished.emit(True, "子进程已完成")
            else:
                self.process_finished.emit(False, f"子进程失败，返回码：{return_code}")

        except Exception as e:
            self.process_finished.emit(False, f"启动/运行失败: {str(e)}")


class AIProcessThread(_BaseSubprocessThread):
    """运行 AI_Image_Process_To_Mysql.py 的线程"""
    pass


class SkipProcessThread(_BaseSubprocessThread):
    """运行 Skip_Process.py 的线程"""
    pass


# ---------------------------
# 扫描新图片线程
# ---------------------------
class ScanImagesThread(QThread):
    progress = pyqtSignal(int, str)         # 百分比, 状态文本
    finished = pyqtSignal(bool, str, int)   # 成功/失败, 信息, 新增数量

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
                    self.progress.emit(int(fi * 100 / max(total_folders, 1)),
                                       f"跳过无效文件夹：{short_name}")
                    continue

                image_files = list(folder.rglob('*'))
                total_files = len(image_files) or 1

                for i, p in enumerate(image_files, start=1):
                    if self._cancel:
                        self.finished.emit(False, "扫描已取消", total_new)
                        return
                    if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif', '.bmp'):
                        cursor.execute(
                            "SELECT id FROM image_records WHERE original_image_path = ?",
                            (str(p),)
                        )
                        if cursor.fetchone() is None:
                            # 插入新记录：processed_time 置空，is_processed=0
                            cursor.execute("""
                                INSERT INTO image_records
                                (image_name, folder_short_name, original_image_path,
                                 processed_time, is_processed, is_featured)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                p.stem,
                                short_name,
                                str(p),
                                None,   # processed_time 保持空
                                0,      # 未处理
                                0
                            ))
                            total_new += 1

                    percent = int((i / total_files) * 100)
                    self.progress.emit(percent, f"[{short_name}] 扫描进度 {i}/{total_files}")

            conn.commit()
            cursor.close()
            conn.close()
            self.finished.emit(True, "扫描完成", total_new)

        except Exception as e:
            self.finished.emit(False, f"扫描出错：{e}", 0)


# ---------------------------
# 批处理未处理图片线程
# ---------------------------
class ProcessImagesThread(QThread):
    progress = pyqtSignal(int, str)       # 百分比, 状态
    finished = pyqtSignal(bool, str, int) # 成功/失败, 信息, 已处理数

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            # 列出未处理
            cursor.execute("SELECT id, original_image_path FROM image_records WHERE is_processed = ?", (0,))
            rows = cursor.fetchall()
            total = len(rows)
            if total == 0:
                cursor.close()
                conn.close()
                self.finished.emit(True, "没有未处理的图片", 0)
                return

            # 重用 OCR/AI 资源
            reader = easyocr.Reader(['ch_sim', 'en'])
            client = create_openai_client()
            models = get_ai_models()
            params = get_ai_params()
            prompts = get_ai_prompts()

            processed_count = 0

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
                    # 读图（兼容中文路径）
                    img_array = np.fromfile(str(image_path), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError("OpenCV 无法解码该图像")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # OCR
                    ocr_result = reader.readtext(img)
                    ocr_text = "\n".join([t[1] for t in ocr_result]) if ocr_result else ""

                    # AI 英文描述
                    with open(image_path, "rb") as f:
                        encoded_image = base64.b64encode(f.read()).decode('utf-8')

                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompts["image_description"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]
                    }]

                    resp = client.chat.completions.create(
                        model=models['image_description'],
                        messages=messages,
                        **params['image_description']
                    )
                    ai_en = resp.choices[0].message.content

                    # 中文翻译
                    prompt = f"请将以下English文本翻译成Chinese，保持原文的意思和风格:\n\n{ai_en}"
                    messages = [
                        {"role": "system", "content": prompts["translation_system"]},
                        {"role": "user", "content": prompt}
                    ]
                    resp2 = client.chat.completions.create(
                        model=models['translation'],
                        messages=messages,
                        **params['translation']
                    )
                    ai_zh = resp2.choices[0].message.content

                    # EXIF
                    exif = get_exif_data(str(image_path)) or {}

                    # 更新数据库
                    cursor.execute("""
                        UPDATE image_records
                        SET ocr_text = ?,
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
                        ocr_text,
                        ai_en,
                        ai_zh,
                        datetime.now(),
                        1,
                        exif.get('make'),
                        exif.get('model'),
                        exif.get('datetime'),
                        exif.get('exposure_time'),
                        exif.get('f_number'),
                        exif.get('iso'),
                        exif.get('focal_length'),
                        exif.get('lens_model'),
                        exif.get('gps_latitude'),
                        exif.get('gps_longitude'),
                        exif.get('gps_altitude'),
                        exif.get('image_width'),
                        exif.get('image_height'),
                        exif.get('software'),
                        exif.get('copyright'),
                        record_id
                    ))
                    conn.commit()
                    processed_count += 1

                except Exception as e:
                    logging.error(f"处理图片 {image_path} 出错：{e}")
                    # 不中断整体流程，继续下一张

            cursor.close()
            conn.close()
            self.finished.emit(True, "处理完成", processed_count)

        except Exception as e:
            self.finished.emit(False, f"处理流程异常：{e}", 0)


# ---------------------------
# 简单进度对话框（备用：未使用 QProgressDialog 时可用）
# ---------------------------
class ProgressDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(400, 150)
        layout = QVBoxLayout(self)
        self.status_label = QLabel("准备中...")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self.setLayout(layout)

    def update_progress(self, value, status=None):
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)


# ---------------------------
# 图片详情对话框（移除多余编码转换）
# ---------------------------
class ImageDetailDialog(QDialog):
    def __init__(self, record_data, parent=None):
        super().__init__(parent)
        self.record_data = record_data
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"图片详情 - {self.record_data['image_name']}")
        self.setMinimumSize(1200, 800)
        main_layout = QHBoxLayout()

        # 左侧：图片预览
        left_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_image()
        left_layout.addWidget(self.image_label)

        # 右侧：内容与操作
        right_layout = QVBoxLayout()

        # 文件操作按钮
        link_layout = QHBoxLayout()
        folder_btn = QPushButton("打开文件夹")
        folder_btn.clicked.connect(self.open_folder)
        image_btn = QPushButton("打开图片")
        image_btn.clicked.connect(self.open_image)
        link_layout.addWidget(folder_btn)
        link_layout.addWidget(image_btn)

        # 处理状态与精选
        self.processed_checkbox = QCheckBox("已处理")
        self.processed_checkbox.setChecked(bool(self.record_data.get('is_processed', False)))

        tab_widget = QTabWidget()
        main_tab = QWidget()
        main_layout_tab = QVBoxLayout(main_tab)

        # OCR
        main_layout_tab.addWidget(QLabel("OCR文本:"))
        self.ocr_edit = QTextEdit()
        self.ocr_edit.setText(self.record_data.get('ocr_text') or "")
        self.ocr_edit.setPlaceholderText("OCR文本")
        main_layout_tab.addWidget(self.ocr_edit)

        # AI 英文
        main_layout_tab.addWidget(QLabel("AI描述（英文）:"))
        self.ai_en_edit = QTextEdit()
        self.ai_en_edit.setText(self.record_data.get('ai_description_en') or "")
        self.ai_en_edit.setPlaceholderText("AI描述（英文）")
        main_layout_tab.addWidget(self.ai_en_edit)

        # AI 中文
        main_layout_tab.addWidget(QLabel("AI描述（中文）:"))
        self.ai_zh_edit = QTextEdit()
        self.ai_zh_edit.setText(self.record_data.get('ai_description_zh') or "")
        self.ai_zh_edit.setPlaceholderText("AI描述（中文）")
        main_layout_tab.addWidget(self.ai_zh_edit)

        self.featured_checkbox = QCheckBox("精选")
        self.featured_checkbox.setChecked(bool(self.record_data.get('is_featured', False)))
        main_layout_tab.addWidget(self.featured_checkbox)

        # 标签（逗号分隔）
        main_layout_tab.addWidget(QLabel("标签（逗号分隔）:"))
        self.tags_edit = QLineEdit()
        self.tags_edit.setText(",".join(self.record_data.get('tags', [])))
        main_layout_tab.addWidget(self.tags_edit)

        # EXIF Tab
        exif_tab = QWidget()
        exif_layout_tab = QVBoxLayout(exif_tab)

        exif_fields = [
            ("相机信息", [
                ("制造商", "exif_make"),
                ("型号", "exif_model"),
                ("软件", "exif_software"),
                ("版权", "exif_copyright")
            ]),
            ("拍摄参数", [
                ("拍摄时间", "exif_datetime"),
                ("曝光时间", "exif_exposure_time"),
                ("光圈值", "exif_f_number"),
                ("ISO", "exif_iso"),
                ("焦距", "exif_focal_length"),
                ("镜头型号", "exif_lens_model")
            ]),
            ("GPS信息", [
                ("纬度", "exif_gps_latitude"),
                ("经度", "exif_gps_longitude"),
                ("海拔", "exif_gps_altitude")
            ]),
            ("图片信息", [
                ("宽度", "exif_image_width"),
                ("高度", "exif_image_height")
            ])
        ]

        exif_scroll = QScrollArea()
        exif_scroll.setWidgetResizable(True)
        exif_content = QWidget()
        exif_content_layout = QVBoxLayout(exif_content)

        for group_name, fields in exif_fields:
            group_label = QLabel(f"<b>{group_name}</b>")
            exif_content_layout.addWidget(group_label)
            for field_name, field_key in fields:
                field_layout = QHBoxLayout()
                label = QLabel(f"{field_name}:")
                value = QLabel(str(self.record_data.get(field_key, "") or ""))
                field_layout.addWidget(label)
                field_layout.addWidget(value)
                field_layout.addStretch()
                exif_content_layout.addLayout(field_layout)
            exif_content_layout.addSpacing(10)

        exif_scroll.setWidget(exif_content)
        exif_layout_tab.addWidget(exif_scroll)

        tab_widget.addTab(main_tab, "主要信息")
        tab_widget.addTab(exif_tab, "EXIF信息")

        save_btn = QPushButton("保存修改")
        save_btn.clicked.connect(self.save_changes)

        right_layout.addLayout(link_layout)
        right_layout.addWidget(self.processed_checkbox)
        right_layout.addWidget(tab_widget)
        right_layout.addWidget(save_btn)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.setLayout(main_layout)

    def load_image(self):
        try:
            image_path = self.record_data.get('original_image_path')
            if not image_path:
                self.image_label.setText("无图片路径")
                return
            img_array = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("无法加载图片")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            max_size = 400
            if w > h:
                new_w = max_size
                new_h = int(h * (max_size / w))
            else:
                new_h = max_size
                new_w = int(w * (max_size / h))
            img = cv2.resize(img, (new_w, new_h))
            bytes_per_line = ch * new_w
            q_img = QImage(img.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.image_label.setText(f"加载图片失败: {str(e)}")

    def open_folder(self):
        try:
            folder_path = os.path.dirname(self.record_data['original_image_path'])
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开文件夹失败: {str(e)}")

    def open_image(self):
        try:
            image_path = self.record_data['original_image_path']
            QDesktopServices.openUrl(QUrl.fromLocalFile(image_path))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开图片失败: {str(e)}")

    def save_changes(self):
        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE image_records
                SET ocr_text = ?,
                    ai_description_en = ?,
                    ai_description_zh = ?,
                    is_processed = ?,
                    is_featured = ?
                WHERE id = ?
            """, (
                self.ocr_edit.toPlainText(),
                self.ai_en_edit.toPlainText(),
                self.ai_zh_edit.toPlainText(),
                int(self.processed_checkbox.isChecked()),
                int(self.featured_checkbox.isChecked()),
                self.record_data['id']
            ))

            # 处理标签
            tags_text = self.tags_edit.text().strip()
            tags = [t.strip() for t in tags_text.split(',') if t.strip()]

            cursor.execute("DELETE FROM image_tag_map WHERE image_id = ?", (self.record_data['id'],))
            for tag in tags:
                cursor.execute("INSERT OR IGNORE INTO image_tags (tag_name) VALUES (?)", (tag,))
                cursor.execute("SELECT id FROM image_tags WHERE tag_name = ?", (tag,))
                tag_row = cursor.fetchone()
                if tag_row:
                    tag_id = tag_row["id"] if isinstance(tag_row, sqlite3.Row) else tag_row[0]
                    cursor.execute("INSERT OR IGNORE INTO image_tag_map (image_id, tag_id) VALUES (?, ?)",
                                   (self.record_data['id'], tag_id))

            conn.commit()
            cursor.close()
            conn.close()
            QMessageBox.information(self, "成功", "修改已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.process_thread = None         # 子进程（AI/Skip）
        self.skip_process_thread = None
        self.scan_thread = None            # 扫描线程
        self.proc_thread = None            # 批处理线程
        self.progress_dialog = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("图片记录搜索系统")
        self.setMinimumSize(1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 搜索区
        search_layout = QHBoxLayout()
        self.search_type = QComboBox()
        self.search_type.addItems(["全部", "图片名称", "OCR文本", "AI描述", "处理状态", "精选", "标签"])

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词搜索")
        self.search_input.returnPressed.connect(self.search_records)

        self.exact_match = QCheckBox("精确匹配")

        scan_btn = QPushButton("仅扫描新图片")
        scan_btn.clicked.connect(self.scan_new_images)

        process_btn = QPushButton("处理未处理的图片")
        process_btn.clicked.connect(self.process_unprocessed_images)

        ai_process_btn = QPushButton("扫描图片并处理")
        ai_process_btn.clicked.connect(self.run_ai_image_processing)

        skip_process_btn = QPushButton("跳过处理")
        skip_process_btn.clicked.connect(self.run_skip_processing)

        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.search_records)

        search_layout.addWidget(self.search_type)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.exact_match)
        search_layout.addWidget(search_btn)
        search_layout.addWidget(scan_btn)
        search_layout.addWidget(process_btn)
        search_layout.addWidget(ai_process_btn)
        search_layout.addWidget(skip_process_btn)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "ID", "图片名称", "文件夹", "处理状态", "精选", "标签",
            "OCR文本", "AI描述（英文）", "AI描述（中文）"
        ])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.doubleClicked.connect(self.show_detail)

        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 60)
        self.table.setColumnWidth(5, 150)
        self.table.setColumnWidth(6, 200)
        self.table.setColumnWidth(7, 200)
        self.table.setColumnWidth(8, 200)

        layout.addLayout(search_layout)
        layout.addWidget(self.table)

        self.load_all_records()

    # ---------- 工具 ----------
    def truncate_text(self, text, max_length=20):
        if text is None:
            return ''
        return (text[:max_length] + "...") if len(text) > max_length else text

    def load_all_records(self):
        self.search_records("")

    # ---------- 搜索 ----------
    def search_records(self, search_text=None):
        if search_text is None:
            search_text = self.search_input.text().strip()

        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            stype = self.search_type.currentText()
            exact = self.exact_match.isChecked()

            def wrap_pattern(txt: str):
                return txt if exact else f"%{txt}%"

            # 基础 SELECT，联表一次拉出标签，避免 N+1
            base_select = """
                SELECT r.*,
                       COALESCE(GROUP_CONCAT(t.tag_name, ', '), '') AS tags
                FROM image_records r
                LEFT JOIN image_tag_map m ON r.id = m.image_id
                LEFT JOIN image_tags t ON t.id = m.tag_id
            """
            where = []
            params = []

            if stype == "全部" and search_text:
                op = "=" if exact else "LIKE"
                where.append(f"(r.image_name {op} ? OR r.ocr_text {op} ? OR r.ai_description_en {op} ? OR r.ai_description_zh {op} ?)")
                pat = wrap_pattern(search_text)
                params.extend([pat, pat, pat, pat])
            elif stype == "图片名称" and search_text:
                op = "=" if exact else "LIKE"
                where.append(f"r.image_name {op} ?")
                params.append(wrap_pattern(search_text))
            elif stype == "OCR文本" and search_text:
                op = "=" if exact else "LIKE"
                where.append(f"r.ocr_text {op} ?")
                params.append(wrap_pattern(search_text))
            elif stype == "AI描述" and search_text:
                op = "=" if exact else "LIKE"
                where.append(f"(r.ai_description_en {op} ? OR r.ai_description_zh {op} ?)")
                pat = wrap_pattern(search_text)
                params.extend([pat, pat])
            elif stype == "处理状态" and search_text:
                val = 1 if search_text.lower() in ('true', '1', '是', '已处理', 'yes') else 0
                where.append("r.is_processed = ?")
                params.append(val)
            elif stype == "精选":
                if search_text:
                    val = 1 if search_text.lower() in ('true', '1', '是', '精选', 'yes') else 0
                else:
                    val = 1
                where.append("r.is_featured = ?")
                params.append(val)
            elif stype == "标签" and search_text:
                op = "=" if exact else "LIKE"
                where.append(f"t.tag_name {op} ?")
                params.append(wrap_pattern(search_text))

            sql = base_select
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " GROUP BY r.id ORDER BY r.id DESC"

            cursor.execute(sql, tuple(params))
            records = cursor.fetchall()

            # 刷表
            self.table.setRowCount(0)
            self.table.setHorizontalHeaderLabels([
                f"ID (找到{len(records)}条结果)",
                "图片名称", "文件夹", "处理状态", "精选", "标签",
                "OCR文本", "AI描述（英文）", "AI描述（中文）"
            ])

            for row_idx, r in enumerate(records):
                self.table.insertRow(row_idx)

                # 直接键名 + row_get 取默认值
                self.table.setItem(row_idx, 0, QTableWidgetItem(str(r["id"])))
                self.table.setItem(row_idx, 1, QTableWidgetItem(row_get(r, "image_name", "")))
                self.table.setItem(row_idx, 2, QTableWidgetItem(row_get(r, "folder_short_name", "")))

                is_processed = int(row_get(r, "is_processed", 0) or 0)
                self.table.setItem(row_idx, 3, QTableWidgetItem("已处理" if is_processed else "未处理"))

                is_featured = int(row_get(r, "is_featured", 0) or 0)
                self.table.setItem(row_idx, 4, QTableWidgetItem("是" if is_featured else "否"))

                self.table.setItem(row_idx, 5, QTableWidgetItem(row_get(r, "tags", "")))

                ocr = row_get(r, "ocr_text", "")
                en  = row_get(r, "ai_description_en", "")
                zh  = row_get(r, "ai_description_zh", "")

                ocr_item = QTableWidgetItem(self.truncate_text(ocr, 20));  ocr_item.setToolTip(ocr)
                en_item  = QTableWidgetItem(self.truncate_text(en, 20));   en_item.setToolTip(en)
                zh_item  = QTableWidgetItem(self.truncate_text(zh, 20));   zh_item.setToolTip(zh)

                self.table.setItem(row_idx, 6, ocr_item)
                self.table.setItem(row_idx, 7, en_item)
                self.table.setItem(row_idx, 8, zh_item)

            cursor.close()
            conn.close()

            if len(records) == 0:
                QMessageBox.information(self, "搜索结果", "未找到匹配的记录")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"搜索失败: {str(e)}")

    # ---------- 详情 ----------
    def show_detail(self, index):
        try:
            row = index.row()
            record_id = self.table.item(row, 0).text()

            conn = get_database_connection()
            cursor = conn.cursor()
            # 连标签
            cursor.execute("""
                SELECT r.*,
                       COALESCE(GROUP_CONCAT(t.tag_name, ', '), '') AS tags_concat
                FROM image_records r
                LEFT JOIN image_tag_map m ON r.id = m.image_id
                LEFT JOIN image_tags t ON t.id = m.tag_id
                WHERE r.id = ?
                GROUP BY r.id
            """, (record_id,))
            r = cursor.fetchone()

            if r:
                # 解析标签
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
                    'original_image_path': row_get(r, "original_image_path", ""),
                    'is_processed': bool(row_get(r, "is_processed", 0)),
                    'is_featured': bool(row_get(r, "is_featured", 0)),
                    'exif_make': row_get(r, "exif_make"),
                    'exif_model': row_get(r, "exif_model"),
                    'exif_datetime': row_get(r, "exif_datetime"),
                    'exif_exposure_time': row_get(r, "exif_exposure_time"),
                    'exif_f_number': row_get(r, "exif_f_number"),
                    'exif_iso': row_get(r, "exif_iso"),
                    'exif_focal_length': row_get(r, "exif_focal_length"),
                    'exif_lens_model': row_get(r, "exif_lens_model"),
                    'exif_gps_latitude': row_get(r, "exif_gps_latitude"),
                    'exif_gps_longitude': row_get(r, "exif_gps_longitude"),
                    'exif_gps_altitude': row_get(r, "exif_gps_altitude"),
                    'exif_image_width': row_get(r, "exif_image_width"),
                    'exif_image_height': row_get(r, "exif_image_height"),
                    'exif_software': row_get(r, "exif_software"),
                    'exif_copyright': row_get(r, "exif_copyright"),
                    'tags': tags
                }
                cursor.close()
                conn.close()

                dialog = ImageDetailDialog(record_data, self)
                dialog.exec()
            else:
                cursor.close()
                conn.close()
                QMessageBox.warning(self, "错误", "未找到记录")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示详情失败: {str(e)}")

    # ---------- 扫描 ----------
    def scan_new_images(self):
        try:
            # 读配置
            cfg_path = 'folders_config.json'
            if not os.path.exists(cfg_path):
                QMessageBox.warning(self, "警告", "未找到配置文件 folders_config.json")
                return
            with open(cfg_path, 'r', encoding='utf-8') as f:
                folders = json.load(f)
            if not isinstance(folders, dict) or not folders:
                QMessageBox.warning(self, "警告", "folders_config.json 格式不正确")
                return

            # 进度对话框
            self.progress_dialog = QProgressDialog("正在扫描...", "取消", 0, 100, self)
            self.progress_dialog.setWindowTitle("扫描新图片")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)

            # 线程
            self.scan_thread = ScanImagesThread(folders)
            self.scan_thread.progress.connect(self.on_worker_progress)
            self.scan_thread.finished.connect(self.on_scan_finished)
            self.progress_dialog.canceled.connect(self.scan_thread.cancel)

            self.scan_thread.start()
            self.progress_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"扫描启动失败: {str(e)}")

    def on_scan_finished(self, success: bool, message: str, new_count: int):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        if success:
            QMessageBox.information(self, "扫描完成", f"{message}\n新增 {new_count} 个图片")
            self.load_all_records()
        else:
            QMessageBox.critical(self, "扫描失败", message)

    # ---------- 批处理 ----------
    def process_unprocessed_images(self):
        try:
            # 进度对话框
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
            self.progress_dialog.close()
            self.progress_dialog = None
        if success:
            QMessageBox.information(self, "处理完成", f"{message}\n成功处理 {count} 张图片")
            self.load_all_records()
        else:
            QMessageBox.critical(self, "处理失败", message)

    # ---------- 运行外部脚本（AI处理 / 跳过处理） ----------
    def run_ai_image_processing(self):
        try:
            if not os.path.exists("folders_config.json"):
                QMessageBox.warning(self, "警告", "未找到配置文件 folders_config.json")
                return
            script_path = "AI_Image_Process_To_Mysql.py"
            if not os.path.exists(script_path):
                QMessageBox.warning(self, "警告", "未找到AI处理脚本 AI_Image_Process_To_Mysql.py")
                return

            reply = QMessageBox.question(
                self, "确认处理", "确定要开始AI图片处理吗？这可能需要较长时间。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            self.progress_dialog = QProgressDialog("正在处理图片...", "取消", 0, 0, self)
            self.progress_dialog.setWindowTitle("AI图片处理")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)

            self.process_thread = AIProcessThread(script_path)
            self.process_thread.progress_updated.connect(self.update_progress_message)
            self.process_thread.process_finished.connect(self.on_subprocess_finished)
            self.progress_dialog.canceled.connect(self.process_thread.cancel)

            self.process_thread.start()
            self.progress_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动处理失败: {str(e)}")

    def run_skip_processing(self):
        try:
            if not os.path.exists("folders_config.json"):
                QMessageBox.warning(self, "警告", "未找到配置文件 folders_config.json")
                return
            script_path = "Skip_Process.py"
            if not os.path.exists(script_path):
                QMessageBox.warning(self, "警告", "未找到跳过处理脚本 Skip_Process.py")
                return

            reply = QMessageBox.question(
                self, "确认跳过处理", "确定要开始跳过处理吗？这将批量插入未处理的图片记录到数据库。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
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

    # ---------- 通用进度/完成 ----------
    def on_worker_progress(self, percent: int, status: str):
        if self.progress_dialog:
            # 0..100 进度
            if self.progress_dialog.maximum() == 0:
                self.progress_dialog.setRange(0, 100)
            self.progress_dialog.setValue(max(0, min(100, percent)))
            self.progress_dialog.setLabelText(status)

    def update_progress_message(self, message: str):
        if self.progress_dialog:
            self.progress_dialog.setLabelText(message)

    def on_subprocess_finished(self, success: bool, message: str):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        if success:
            QMessageBox.information(self, "处理完成", message)
            self.load_all_records()
        else:
            QMessageBox.critical(self, "处理失败", message)

    # ---------- 取路径（可保留） ----------
    def get_image_path(self, row):
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            record_id = self.table.item(row, 0).text()
            cursor.execute("SELECT original_image_path FROM image_records WHERE id = ?", (record_id,))
            r = cursor.fetchone()
            cursor.close()
            conn.close()
            return r["original_image_path"] if r else None
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取图片路径失败: {str(e)}")
            return None


# ---------------------------
# 入口
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


