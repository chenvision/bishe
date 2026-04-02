# 🍓 草莓单目三维姿态与尺寸估计系统 (Strawberry 6D Pose & Size Estimation System)

这是一个基于深度学习与几何约束的单目视觉草莓三维姿态与尺寸估计系统。系统集成了高性能检测算法、全局感知架构以及直观的 Web 交互控制台，旨在为农业机器人精准作业提供核心视觉支持。

## ✨ 核心特性
- **单目 6D 姿态估计**：仅需单个标准 RGB 摄像头即可回归目标的 3D 物理位置 $(x, y, z)$ 及旋转角度。
- **物理尺寸预测**：实时精准估计草莓的长、宽、高（$l, w, h$）物理尺寸。
- **增强型架构**：采用 Darknet-19 骨干网络结合 **Transformer 全局感知模块**，显著提升了在复杂背景及遮挡环境下的推理鲁棒性。
- **工业级 Web 控制台**：基于 Vue 3 开发，支持实时视频流检测、静态图像深度分析、系统日志监控及遥测数据可视化。
- **几何约束优化**：利用 EPnP 算法结合网络回归的 2D 投影关键点，实现从像素空间到物理空间的高精度转换。

## 🛠️ 技术栈
- **后端算法**：Python, PyTorch, FastAPI, OpenCV, Albumentations
- **前端界面**：Vue 3, Vite, Element Plus, Axios
- **核心算法**：Improved Darknet, Transformer Encoder, EPnP, CoordConv, 2D Sine-Cosine PE

## 📂 项目结构概览
- `app.py`: 后端 FastAPI 服务入口及模型推理逻辑。
- `models/`: 包含深度学习模型定义及详细的[架构说明文档](./models/模型架构与算法原理.md)。
- `datasets/`: 数据集处理、标注解析逻辑及[维度推导文档](./datasets/数据处理与维度推导.md)。
- `frontend/`: 基于 Vue 3 的交互界面源码。
- `README_STARTUP.md`: 针对初学者的[系统启动与环境配置指南](./README_STARTUP.md)。

## 🚀 快速开始
请务必先查阅 [README_STARTUP.md](./README_STARTUP.md) 获取详细的安装步骤与运行指南。

## 📖 学术与设计文档
为了方便查阅论文相关素材，建议参考以下文档：
- [系统总体方案设计](./系统总体方案设计.md)
- [模型架构与算法原理](./models/模型架构与算法原理.md)
- [数据处理与维度推导](./datasets/数据处理与维度推导.md)

---
*本项目为毕业设计作品。*
