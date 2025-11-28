# AI Image Process Tool (Type B) - 中文说明

桌面端图像批处理与管理工具：批量扫描文件夹、去重入库（SQLite）、OCR 识别、AI 描述，并在 PyQt6 界面里检索与编辑。

- 多文件夹扫描，按 MD5/SHA256 去重，记录原始路径与备用路径。
- 批量 OCR（EasyOCR）+ AI 描述（OpenAI 兼容/LM Studio），并提取 EXIF 元数据。
- 表格检索与过滤：按名称/OCR/AI 描述/标签/状态等查询，双击行可查看详情、预览图片、编辑标签和备注。
- 自定义表格列显示/顺序；补算缺失或重算全部哈希。
- 设置对话框可直接编辑 `ai_config.json` 与 `folders_config.json`；提供“跳过处理”脚本，将记录标记为已处理。

## 安装
### 1) 创建 conda 环境
```powershell
conda env create -f environment.yml
conda activate ai-assist-image-description
```
无 NVIDIA GPU 时，可在 `environment.yml` 里去掉 `pytorch-cuda`，改为安装 `cpuonly` 版本的 PyTorch。

### 2) 配置文件
- `ai_config.json`：设置 OpenAI 兼容接口的 `server.base_url` 和 `server.api_key`（可用 LM Studio）。通过 `selected_scheme` 选择描述方案。
- `folders_config.json`：映射“文件夹绝对路径”到“简称”，示例：
  ```json
  {
    "D:/path/to/Photos": "Photos",
    "D:/path/to/Scans": "Scans"
  }
  ```
- `sqlite_config.json`：设置 `database_path`；如不存在会自动创建并迁移表结构。
- `column_config.json`：在界面调整列后自动生成/更新。

## 运行
```powershell
python image_search_gui.py
```
推荐流程：打开“设置”确认文件夹与 AI 方案 → “仅扫描新图片”导入 → “处理未处理图片”（或“扫描图片并处理”）执行 OCR+AI → 双击行查看详情，编辑标签/备注，打开图片/文件夹。

### 界面按钮
- “仅扫描新图片”：按配置扫描并写入数据库。
- “重新获取MD5/SHA256”：补算缺失或重算全部哈希。
- “处理未处理图片”：执行 OCR + AI 描述 + EXIF。
- “扫描图片并处理”：先扫描再自动处理。
- “跳过处理”：运行 `skip_describe_image.py`，仅写入记录不跑 AI。
- “列显示/顺序”：选择要显示的列并调整顺序。
- “搜索”：选择搜索范围，可切换精确匹配。

## 其他说明
- 日志输出在 `image_processing.log`。
- OCR 后端默认 EasyOCR，可在设置对话框或 `ai_config.json` 的 `ocr.backend` 修改。
- 如果图片解码失败，请确认已安装 `opencv-python-headless`（已在 `environment.yml` 中）。
