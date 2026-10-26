# Emotion Cam · 实时微笑 / 伤心识别

用电脑摄像头做人脸情绪识别，重点判断 **微笑（happy）** 与 **伤心（sad）**。

## 两个版本

| 文件 | 方案 | 依赖 |
|------|------|------|
| `main.py` | **高级**：DeepFace 深度学习 7 类情绪 + 时间平滑 + 置信度条 | TensorFlow / DeepFace（首次下模型） |
| `main_lite.py` | **轻量**：OpenCV Haar 笑脸检测 + 嘴部启发式 | 仅 opencv-python、numpy |

## 安装与运行

```bash
git clone https://github.com/Zhy-max442/emotion-cam.git
cd emotion-cam
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

若高级版安装太慢，可先跑轻量版：

```bash
pip install opencv-python numpy
python main_lite.py
```

## 操作

| 按键 | 作用 |
|------|------|
| `q` / `Esc` | 退出 |
| `s` | 截图到 `captures/` |
| `f` | 仅高级版：左右镜像开关 |

## 配置

在 `main.py` 顶部可改：

- `CAMERA_INDEX`：摄像头编号（默认 `0`）
- `ANALYZE_EVERY_N`：隔几帧推理一次（越大越流畅、越不跟手）

## 注意

- Windows 请允许应用访问摄像头。
- 正脸、光线均匀时效果更好。
- 高级版首次运行需联网下载模型。
