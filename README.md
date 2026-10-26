# Emotion Cam · 网页版微笑 / 伤心识别

**直接用浏览器**，打开摄像头识别微笑或伤心。无需安装软件。

## 在线使用

GitHub Pages：  
https://zhy-max442.github.io/emotion-cam/

（若打不开，用仓库里的 `web/index.html` 本地打开，或见下方本地预览。）

## 本地预览

```bash
cd emotion-cam
# 任意静态服务器均可，例如：
python -m http.server 8080
```

浏览器打开 http://localhost:8080/web/

## 说明

- 技术：`getUserMedia` + [face-api.js](https://github.com/justadudewhohacks/face-api.js)（浏览器端深度学习）
- 点「开启摄像头」并允许权限；正脸、光线好效果更稳
- 仓库里仍保留旧的 Python 桌面脚本（`main.py` / `main_lite.py`），可选

## 文件

- `web/index.html` / `web/app.js` / `web/style.css` — 网页版（推荐）
- `main.py` — 桌面 DeepFace 版
- `main_lite.py` — 桌面 OpenCV 轻量版
