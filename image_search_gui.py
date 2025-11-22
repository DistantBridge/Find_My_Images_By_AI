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
from describe_image_by_ai import get_exif_data
import hashlib


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


def compute_file_hashes(path: str):
    """计算文件的 md5 与 sha256，返回 (md5_hex, sha256_hex)"""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                md5.update(chunk)
                sha256.update(chunk)
        return md5.hexdigest(), sha256.hexdigest()
    except Exception:
        return None, None


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
            # 记录将要运行的命令和当前工作目录、python 可执行路径
            cmd = [sys.executable, self.script_path]
            cwd = Path(__file__).parent
            env = os.environ.copy()
            # 强制覆盖 PYTHONUTF8，避免父进程中被错误值（例如 'on'）污染导致子进程启动失败
            env['PYTHONUTF8'] = '1'

            self.progress_updated.emit(f"启动脚本：{self.script_path}")
            logging.info(f"启动子进程: cmd={cmd}, cwd={cwd}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout，方便实时读取
                text=True,
                encoding='utf-8',
                bufsize=1,
                cwd=str(cwd),
                env=env
            )

            # 实时读取输出，兼容不同 Python/Windows 行结束
            assert process.stdout is not None
            try:
                for line in iter(process.stdout.readline, ''):
                    if line is None:
                        break
                    if self._cancel:
                        self.progress_updated.emit("收到取消请求，终止子进程...")
                        try:
                            process.terminate()
                            process.wait(timeout=5)
                        except Exception:
                            process.kill()
                        self.process_finished.emit(False, "处理已取消")
                        return
                    cleaned = line.rstrip('\r\n')
                    if cleaned:
                        self.progress_updated.emit(cleaned)
                        logging.info(f"子进程输出: {cleaned}")
            except Exception as e:
                logging.error(f"读取子进程输出时出错: {e}")

            # 等待子进程结束并获取返回码
            try:
                return_code = process.wait()
            except Exception as e:
                logging.error(f"等待子进程结束时出错: {e}")
                return_code = -1

            if return_code == 0:
                self.process_finished.emit(True, "子进程已完成")
            else:
                logging.error(f"子进程退出码: {return_code}")
                self.process_finished.emit(False, f"子进程失败，返回码：{return_code}")

        except Exception as e:
            self.process_finished.emit(False, f"启动/运行失败: {str(e)}")


class AIProcessThread(_BaseSubprocessThread):
    """运行 describe_image_by_ai.py 的线程"""
    pass


class SkipProcessThread(_BaseSubprocessThread):
    """运行 skip_describe_image.py 的线程"""
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
                            # 计算文件哈希，优先用哈希判重
                            md5_hex, sha256_hex = compute_file_hashes(str(p))
                            found = None
                            if md5_hex and sha256_hex:
                                cursor.execute(
                                    "SELECT id, original_image_path, alternate_paths FROM image_records WHERE md5 = ? AND sha256 = ?",
                                    (md5_hex, sha256_hex)
                                )
                                found = cursor.fetchone()

                            if found:
                                # 若已存在相同哈希的记录，添加为备用路径（若尚未包含）
                                rec_id = found['id'] if isinstance(found, sqlite3.Row) else found[0]
                                existing_orig = found['original_image_path'] if isinstance(found, sqlite3.Row) else found[1]
                                alt_json = found['alternate_paths'] if isinstance(found, sqlite3.Row) else found[2]
                                try:
                                    alt_list = json.loads(alt_json) if alt_json else []
                                except Exception:
                                    alt_list = []

                                new_path = str(p)
                                # 如果当前路径既不是主路径也不在备用路径中，则追加
                                if new_path != existing_orig and new_path not in alt_list:
                                    alt_list.append(new_path)
                                    cursor.execute("UPDATE image_records SET alternate_paths = ? WHERE id = ?",
                                                   (json.dumps(alt_list, ensure_ascii=False), rec_id))
                                    # don't count as new record
                            else:
                                # 新文件：插入记录并保存哈希与备用路径（空列表）
                                cursor.execute("""
                                    INSERT INTO image_records
                                    (image_name, folder_short_name, original_image_path,
                                     md5, sha256, alternate_paths,
                                     processed_time, is_processed, is_featured)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    p.stem,
                                    short_name,
                                    str(p),
                                    md5_hex,
                                    sha256_hex,
                                    json.dumps([], ensure_ascii=False),
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
                    # 使用 describe_image，传入 OCR 文本；describe_image 会根据后端决定是否传图片
                    try:
                        ai_zh = __import__('describe_image_by_ai').describe_image(str(image_path), prompts['image_description'], ocr_text=ocr_text)
                    except Exception:
                        # 作为后备，直接调用模块函数（更常规的导入方式）
                        from describe_image_by_ai import describe_image
                        ai_zh = describe_image(str(image_path), prompts['image_description'], ocr_text=ocr_text)
                    ai_en = ""

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
        alt_btn = QPushButton("打开备用路径")
        alt_btn.clicked.connect(self.open_alternate_paths)
        link_layout.addWidget(folder_btn)
        link_layout.addWidget(image_btn)
        link_layout.addWidget(alt_btn)

        # 处理状态与精选
        self.processed_checkbox = QCheckBox("已处理")
        self.processed_checkbox.setChecked(bool(self.record_data.get('is_processed', False)))

        # 将多个文本区域拆分为独立的选项卡，减少单个选项卡拥挤
        self.featured_checkbox = QCheckBox("精选")
        self.featured_checkbox.setChecked(bool(self.record_data.get('is_featured', False)))
        tab_widget = QTabWidget()

        # OCR Tab
        ocr_tab = QWidget()
        ocr_layout = QVBoxLayout(ocr_tab)
        ocr_layout.addWidget(QLabel("OCR文本:"))
        self.ocr_edit = QTextEdit()
        self.ocr_edit.setText(self.record_data.get('ocr_text') or "")
        self.ocr_edit.setPlaceholderText("OCR文本")
        ocr_layout.addWidget(self.ocr_edit)

        # AI 英文 Tab
        ai_en_tab = QWidget()
        ai_en_layout = QVBoxLayout(ai_en_tab)
        ai_en_layout.addWidget(QLabel("AI描述（英文）:"))
        self.ai_en_edit = QTextEdit()
        self.ai_en_edit.setText(self.record_data.get('ai_description_en') or "")
        self.ai_en_edit.setPlaceholderText("AI描述（英文）")
        ai_en_layout.addWidget(self.ai_en_edit)

        # AI 中文 Tab
        ai_zh_tab = QWidget()
        ai_zh_layout = QVBoxLayout(ai_zh_tab)
        ai_zh_layout.addWidget(QLabel("AI描述（中文）:"))
        self.ai_zh_edit = QTextEdit()
        self.ai_zh_edit.setText(self.record_data.get('ai_description_zh') or "")
        self.ai_zh_edit.setPlaceholderText("AI描述（中文）")
        ai_zh_layout.addWidget(self.ai_zh_edit)

        # 属性 Tab（合并：标签 / 精选 / 备注）
        properties_tab = QWidget()
        properties_layout = QVBoxLayout(properties_tab)
        # 标签（逗号分隔）
        properties_layout.addWidget(QLabel("标签（逗号分隔）:"))
        self.tags_edit = QLineEdit()
        self.tags_edit.setText(",".join(self.record_data.get('tags', [])))
        properties_layout.addWidget(self.tags_edit)
        # 精选复选框
        properties_layout.addWidget(self.featured_checkbox)
        # 备注
        properties_layout.addWidget(QLabel("备注:"))
        self.remark_edit = QTextEdit()
        self.remark_edit.setPlaceholderText("在此输入备注（会保存到数据库）")
        self.remark_edit.setText(self.record_data.get('remark', "") or "")
        properties_layout.addWidget(self.remark_edit)

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

        tab_widget.addTab(ai_zh_tab, "AI中文")
        tab_widget.addTab(ai_en_tab, "AI英文")
        tab_widget.addTab(ocr_tab, "OCR文本")
        tab_widget.addTab(properties_tab, "备注/标签")
        tab_widget.addTab(exif_tab, "EXIF信息")

        # 文件信息 Tab：显示路径、文件夹简称、备用路径、md5、sha256 等
        file_info_tab = QWidget()
        file_info_layout = QVBoxLayout(file_info_tab)

        file_info_layout.addWidget(QLabel("原始图片路径:"))
        self.path_label = QLabel(str(self.record_data.get('original_image_path', '') or ''))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_info_layout.addWidget(self.path_label)

        file_info_layout.addWidget(QLabel("文件夹简称:"))
        self.folder_label = QLabel(str(self.record_data.get('folder_short_name', '') or ''))
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_info_layout.addWidget(self.folder_label)

        file_info_layout.addWidget(QLabel("备用路径:"))
        alt_paths = self.record_data.get('alternate_paths') or []

        # 使用滚动区域和只读路径行替代可编辑文本，防止误改
        alt_scroll = QScrollArea()
        alt_scroll.setWidgetResizable(True)
        alt_widget = QWidget()
        alt_layout = QVBoxLayout(alt_widget)

        try:
            if isinstance(alt_paths, (list, tuple)) and alt_paths:
                for p in alt_paths:
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    path_label = QLabel(str(p))
                    path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                    row_layout.addWidget(path_label)

                    open_btn = QPushButton("打开")
                    def make_open(path):
                        def _open():
                            try:
                                if os.path.exists(path):
                                    QDesktopServices.openUrl(QUrl.fromLocalFile(path))
                                else:
                                    QMessageBox.warning(self, "路径不存在", f"路径不存在: {path}")
                            except Exception as e:
                                QMessageBox.critical(self, "错误", f"打开路径失败: {e}")
                        return _open
                    open_btn.clicked.connect(make_open(p))
                    row_layout.addWidget(open_btn)

                    copy_btn = QPushButton("复制")
                    def make_copy(path):
                        def _copy():
                            try:
                                QApplication.clipboard().setText(path)
                                QMessageBox.information(self, "复制", "路径已复制到剪贴板")
                            except Exception as e:
                                QMessageBox.critical(self, "错误", f"复制失败: {e}")
                        return _copy
                    copy_btn.clicked.connect(make_copy(p))
                    row_layout.addWidget(copy_btn)

                    alt_layout.addWidget(row_widget)
            else:
                alt_layout.addWidget(QLabel("(无备用路径)"))
        except Exception:
            alt_layout.addWidget(QLabel("(备用路径加载失败)"))

        alt_scroll.setWidget(alt_widget)
        file_info_layout.addWidget(alt_scroll)

        file_info_layout.addWidget(QLabel("MD5:"))
        self.md5_label = QLabel(str(self.record_data.get('md5', '') or ''))
        self.md5_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_info_layout.addWidget(self.md5_label)

        file_info_layout.addWidget(QLabel("SHA256:"))
        self.sha256_label = QLabel(str(self.record_data.get('sha256', '') or ''))
        self.sha256_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_info_layout.addWidget(self.sha256_label)

        tab_widget.addTab(file_info_tab, "文件信息")

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

    def open_alternate_paths(self):
        try:
            alt_paths = self.record_data.get('alternate_paths') or []
            if not alt_paths:
                QMessageBox.information(self, "备用路径", "该图片没有备用路径")
                return

            dlg = QDialog(self)
            dlg.setWindowTitle("打开备用路径")
            dlg.setMinimumSize(800, 300)
            layout = QVBoxLayout(dlg)

            # 列出所有备用路径，点击打开对应文件
            for p in alt_paths:
                btn = QPushButton(p)
                btn.setToolTip(p)
                def _open(path=p):
                    try:
                        if os.path.exists(path):
                            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
                        else:
                            QMessageBox.warning(self, "路径不存在", f"路径不存在: {path}")
                    except Exception as e:
                        QMessageBox.critical(self, "错误", f"打开路径失败: {e}")
                btn.clicked.connect(_open)
                layout.addWidget(btn)

            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dlg.accept)
            layout.addWidget(close_btn)

            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开备用路径失败: {str(e)}")

    def save_changes(self):
        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE image_records
                SET ocr_text = ?,
                    ai_description_en = ?,
                    ai_description_zh = ?,
                    remark = ?,
                    is_processed = ?,
                    is_featured = ?
                WHERE id = ?
            """, (
                self.ocr_edit.toPlainText(),
                self.ai_en_edit.toPlainText(),
                self.ai_zh_edit.toPlainText(),
                self.remark_edit.toPlainText(),
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
        self.auto_process_after_scan = False  # ★ 扫描结束后是否自动处理
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
        self.search_type.addItems(["全部", "图片名称", "OCR文本", "AI描述", "处理状态", "精选", "标签", "备注"])

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
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "ID", "图片名称", "文件夹", "处理状态", "精选", "标签",
            "OCR文本", "AI描述（英文）", "AI描述（中文）", "备注"
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
        self.table.setColumnWidth(9, 200)

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
                where.append(f"(r.image_name {op} ? OR r.ocr_text {op} ? OR r.ai_description_en {op} ? OR r.ai_description_zh {op} ? OR r.remark {op} ?)")
                pat = wrap_pattern(search_text)
                params.extend([pat, pat, pat, pat, pat])
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
            elif stype == "备注" and search_text:
                op = "=" if exact else "LIKE"
                where.append(f"r.remark {op} ?")
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
                "OCR文本", "AI描述（英文）", "AI描述（中文）", "备注"
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

                # 备注列
                remark = row_get(r, "remark", "")
                remark_item = QTableWidgetItem(self.truncate_text(remark, 20)); remark_item.setToolTip(remark)
                self.table.setItem(row_idx, 9, remark_item)

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
                    'md5': row_get(r, "md5", ""),
                    'sha256': row_get(r, "sha256", ""),
                    'remark': row_get(r, "remark", ""),
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
                    'tags': tags,
                    'alternate_paths': []
                }
                # 解析 alternate_paths（JSON）
                try:
                    ap = row_get(r, 'alternate_paths', '')
                    if ap:
                        record_data['alternate_paths'] = json.loads(ap)
                    else:
                        record_data['alternate_paths'] = []
                except Exception:
                    record_data['alternate_paths'] = []
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
        # 关闭扫描进度框
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # 扫描失败：无论是否处于“自动处理模式”，都只报错然后结束
        if not success:
            if self.auto_process_after_scan:
                self.auto_process_after_scan = False
            QMessageBox.critical(self, "扫描失败", message)
            return

        # 扫描成功，先刷新列表
        self.load_all_records()

        if self.auto_process_after_scan:
            # 来自“扫描图片并处理”按钮：不弹“扫描完成”，直接进入处理阶段
            self.auto_process_after_scan = False
            self.process_unprocessed_images()
        else:
            # 来自“仅扫描新图片”按钮：保持原来的提示行为
            QMessageBox.information(self, "扫描完成", f"{message}\n新增 {new_count} 个图片")

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
        """
        扫描图片并处理：不再调用外部脚本，
        而是顺序执行：
          1）scan_new_images（扫描新图片）
          2）process_unprocessed_images（处理未处理图片）
        """
        try:
            if not os.path.exists("folders_config.json"):
                QMessageBox.warning(self, "警告", "未找到配置文件 folders_config.json")
                return

            reply = QMessageBox.question(
                self,
                "确认扫描并处理",
                "将先扫描所有配置文件中的图片文件夹，"
                "然后处理数据库中所有未处理的图片。\n\n这可能需要较长时间，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            # 打开“链式模式”：扫描完成后自动调用处理逻辑
            self.auto_process_after_scan = True

            # 直接复用现有的扫描逻辑（会弹出扫描进度框）
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


