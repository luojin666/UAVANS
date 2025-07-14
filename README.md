# UAVANS-无人机卫星自动巡检系统

## 简介

beta0.00000001

## 环境与依赖

* **Python 版本**： Python 3.12

### 安装依赖

在项目根目录下执行：

```bash
pip install -r requirements.txt
```

## 文件说明

| 文件名                 | 作用                  |
|---------------------|---------------------|
| `generate_plans.py` | 主程序，包含VLM和LLM的全套流程。 |
| `getmap.py`         | 卫星地图获取，有bug，评估中。    |
| `config.py`         | Prompt存于此处。         |

## 快速开始

1. 环境准备：确保 Python 3.12 安装完成，并已安装依赖。
2. 运行系统：

   ```bash
   python generate_plans.py
   ```

## 待办事项
- [x] 使用deepseek作为LLM模型
- [x] 图片像素坐标转换经纬度
- [ ] 经纬度自动下载卫星图
- [ ] 微调VLM模型
- [ ] 训练一个计算机视觉模型分割建筑物
- [ ] 微调模型使其能分割建筑物的类别

## 其他问题

请咨询@luojin666

## 许可证

本项目采用 MIT 许可证，详见 `LICENSE` 文件。
