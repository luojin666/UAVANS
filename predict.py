from ultralytics import YOLO
import os

def main():
    model_path = 'models/best.pt'
    source = 'data/images/1.jpg'
    save_dir = 'predict_results'

    model = YOLO(model_path)

    # 1) 预测并保存可视化结果
    results = model.predict(
        source=source,
        conf=0.4,         # 只保留置信度 >= 0.4 的检测
        save=True,        # 自动保存可视化结果
        save_txt=False,
        save_conf=False,
        project=save_dir,
        name='exp',
        imgsz=768,
        device=0
    )

    # 2) 对每一张输入（这里只有一张 map.png），打印中心坐标
    for img_idx, result in enumerate(results):
        # xywh 格式：每行 [cx, cy, w, h]
        xywh = result.boxes.xywh.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs   = result.boxes.conf.cpu().numpy()

        print(f"\n=== 图像 {img_idx} 的检测结果 ===")
        for i, (cx, cy, w, h) in enumerate(xywh):
            label = result.names[classes[i]]
            conf  = confs[i]
            print(f"目标 {i}: 类别={label:>8s}, 置信度={conf:.2f}, 中心坐标=({cx:.1f}, {cy:.1f})")

    print(f"\n预测完成，结果保存在 “{os.path.join(save_dir, 'exp')}” 文件夹下。")

if __name__ == '__main__':
    main()
