# 🍓 草莓单目三维姿态与尺寸估计系统 - 启动指南

欢迎使用本系统！本指南将带你从零开始配置环境并运行整个项目。

## 📋 运行前准备

在开始之前，请确保你的电脑已安装以下基础环境：
1.  **Python 3.8+** (推荐使用 [Anaconda](https://www.anaconda.com/) 管理环境)
2.  **Node.js 16.x+** (用于运行前端界面，前往 [Node.js 官网](https://nodejs.org/) 下载)
3.  **摄像头设备** (用于实时视频流推理，若无设备可使用静态图片分析功能)

---

## 🚀 第一步：后端算法环境配置 (Backend)

后端负责运行深度神经网络模型、处理几何运算及提供 API 接口。

1.  **进入项目根目录**：
    打开终端或命令行（CMD/PowerShell），进入 `bishe` 文件夹。

2.  **创建并激活虚拟环境** (可选但推荐)：
    ```bash
    # 使用 conda 创建环境
    conda create -n strawberry python=3.9
    conda activate strawberry
    ```

3.  **安装必要依赖库**：
    执行以下命令安装深度学习与图像处理相关的库：
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # 若无显卡请直接 pip install torch
    pip install fastapi uvicorn opencv-python numpy pandas matplotlib pillow albumentations
    ```

4.  **检查模型权重文件**：
    确保根目录下存在 `darknet_strawberry_checkpoint.pth` 文件。如果没有该文件，系统将无法正确识别草莓。

5.  **启动后端服务器**：
    ```bash
    python app.py
    ```
    *   **启动成功标志**：终端显示 `INFO: Uvicorn running on http://127.0.0.1:8000`。
    *   **注意**：请保持此窗口开启，不要关闭。

---

## 🎨 第二步：前端交互界面配置 (Frontend)

前端提供可视化的控制面板，让你直观地查看实时检测结果。

1.  **进入前端目录**：
    重新打开一个终端窗口（不要关掉后端的），进入 `frontend` 文件夹：
    ```bash
    cd frontend
    ```

2.  **安装前端依赖**：
    ```bash
    npm install
    ```
    *若下载较慢，可使用阿里镜像：`npm install --registry=https://registry.npmmirror.com`*

3.  **启动开发服务器**：
    ```bash
    npm run serve
    ```

4.  **访问界面**：
    当终端显示 `App running at: - Local: http://localhost:8080/` 时，在浏览器输入该地址。

---

## 🎮 第三步：系统使用说明

### 1. 实时视觉反馈 (Real-time Mode)
*   进入页面后，浏览器会请求摄像头权限，请点击**“允许”**。
*   系统将自动开启摄像头并进行推理。当镜头内出现草莓时，界面会显示红色的**边界框**，右侧面板将实时更新该草莓的 **3D 坐标 (X, Y, Z)** 和 **物理尺寸 (长, 宽, 高)**。

### 2. 静态图像分析 (Static Analysis)
*   点击视频面板右上角的 **“📷 图像分析”** 按钮。
*   上传一张包含草莓的单目图片。
*   系统将弹出一个分析结果窗口，展示带有 3D 边界框的合成图及该图中所有目标的详细参数列表。

### 3. 系统日志
*   右下角的“系统日志”会记录当前系统的运行状态（如 API 调用成功、摄像头状态等），方便排查问题。

---

## ❓ 常见问题 (FAQ)

*   **Q: 后端报错 "CUDA out of memory"?**
    *   A: 显存不足。请在 `app.py` 中将 `DEVICE = torch.device("cuda" ...)` 改为 `DEVICE = torch.device("cpu")` 使用 CPU 推理（速度会慢一些）。
*   **Q: 前端页面空白，显示无法连接 API?**
    *   A: 请确认后端 `app.py` 是否正常运行，且终端没有报错。检查 API 地址是否为 `http://127.0.0.1:8000`。
*   **Q: 摄像头无法打开？**
    *   A: 确保没有其他程序（如腾讯会议、微信视频）正在占用摄像头，并检查浏览器的隐私设置是否允许该网页访问摄像头。

---
*祝你的毕业设计顺利完成！如有疑问请查看 `README.md` 或咨询开发者。*
