# UAVANS-无人机卫星自动巡检系统

## 项目简介
UAVANS 是一个基于大语言模型（LLM）与视觉语言模型（VLM）的无人机卫星自动巡检系统。系统支持自然语言输入，自动识别卫星图像中的建筑物，输出经纬度坐标，并自动生成无人机飞行巡检计划。

---

## 目录
- [项目简介](#项目简介)
- [环境与依赖](#环境与依赖)
- [安装与配置](#安装与配置)
- [文件说明](#文件说明)
- [快速开始](#快速开始)
- [系统流程](#系统流程)
- [性能数据](#性能数据)
- [待办事项](#待办事项)

---

## 环境与依赖
- Python 版本：**3.12**
- 推荐使用 Conda 环境管理

## 安装与配置
```bash
# 克隆项目
git clone https://github.com/luojin666/UAVANS.git
cd UAVANS

# 创建并激活环境
conda create -n UAVANS python=3.12
conda activate UAVANS

# 安装依赖
pip install -r requirements.txt
```

---

## 文件说明
| 文件名/目录                | 作用说明                                   |
|---------------------------|--------------------------------------------|
| run.py                    | 主程序，包含VLM和LLM的全套流程             |
| getmap.py                 | 卫星地图获取（开发中）                      |
| config.py                 | Prompt与配置信息                           |
| requirements.txt          | Python依赖包列表                           |
| created_missions/         | 自动生成的无人机飞行计划                   |
| data/                     | 原始图片、卫星图、解析中间数据等           |
| identified_new_data/      | 检测结果图片                               |
| models/                   | 训练好的模型文件                           |
| parser_for_coordinates.py | 坐标解析工具                               |
| draw_circles.py           | 路径与点可视化                             |
| recalculate_to_latlon.py  | 像素坐标转经纬度                           |
| optimized_prompt.py       | 优化后的Prompt相关代码                     |
| predict.py                | 单独的推理脚本                             |

---

## 快速开始
1. **环境准备**：确保 Python 3.12 已安装，并完成依赖安装。
2. **运行主程序**：
   ```bash
   python run.py
   ```
3. **输出**：
   - 自动生成无人机飞行计划（created_missions/mission1.txt）
   - 检测与路径可视化图片（identified_new_data/）

---

## 系统流程
1. 用户输入自然语言需求
2. 系统自动下载卫星图像
3. VLM模型分割建筑物，输出像素坐标
4. 坐标转换为经纬度
5. LLM推理生成无人机飞行计划
6. 自动生成可视化结果与飞行脚本

---

## 性能数据（30张卫星图）
- **GPT-4o**：
  - 分割时间：7分11秒
  - 推理时间：7分59秒
  - 总耗时：15分10秒
- **DeepSeek**：
  - 分割时间：7分38秒
  - 推理时间：13分51秒
  - 总耗时：21分29秒

---

## 项目亮点
- 支持多种大语言模型（如OpenAI GPT-4、DeepSeek等）
- 自动化无人机巡检任务生成
- 支持卫星图像自动下载与处理
- 像素坐标与经纬度自动转换
- 路径规划与可视化

---

## 待办事项
- [x] 支持 DeepSeek 作为 LLM
- [x] 图片像素坐标转换经纬度
- [x] 经纬度自动下载卫星图
- [ ] 微调 VLM 模型
- [x] 训练分割国内建筑物的视觉模型
- [ ] 支持建筑物类别分割
- [x] 优化智能自动导航能力
- [ ] 发布正式版本

---

如有问题或建议，欢迎提交 Issue 或 PR！