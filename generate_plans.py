import json
from ultralytics import YOLO
from PIL import Image
import torch
import os
from typing import Tuple, List, Dict, Any
import logging
from time import time
import numpy as np
import cv2
import time as pytime
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from parser_for_coordinates import parse_points
from draw_circles import draw_dots_and_lines_on_image
from recalculate_to_latlon import recalculate_coordinates, percentage_to_lat_lon, read_coordinates_from_csv
from config import *

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 初始化大语言模型
def initialize_llm(model_type: str = "openai") -> Any:
    if model_type == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("环境变量中未找到OpenAI API key")
        return ChatOpenAI(
            api_key=SecretStr(api_key),
            model='gpt-4',
            temperature=0.0
        )

    elif model_type == "deepseek":
        # DeepSeek
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("环境变量中未找到DeepSeek API key")

        # 使用OpenAI兼容接口调用DeepSeek
        return ChatOpenAI(
            api_key=SecretStr(api_key),
            model='deepseek-chat',
            base_url="https://api.deepseek.com/v1",
            temperature=0.0
        )

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def find_objects(json_input: str, example_objects: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    list_of_the_resulted_coordinates_percentage = []
    list_of_the_resulted_coordinates_lat_lon = []
    all_result_coordinates = []  # 新增：用于收集所有图片的result_coordinates

    try:
        # 加载YOLO模型
        model_path = 'models/best.pt'
        model = YOLO(model_path)

        find_objects_json_input = json_input.replace("`", "").replace("json", "")
        find_objects_json_input_2 = json.loads(find_objects_json_input)

        logger.info(f'Processing 1 sample')

        for i in range(1, 2):  # 只处理第1张图片
            logger.info(f'Processing image {i}')

            try:
                image_path = f'data/images/{i}.jpg'

                # 使用YOLO进行预测
                results = model.predict(
                    source=image_path,
                    conf=0.4,  # 只保留置信度 >= 0.4 的检测
                    save=False,  # 不自动保存可视化结果
                    save_txt=False,
                    save_conf=False,
                    imgsz=768,
                    device=0
                )

                # 处理检测结果，生成VLM格式的XML字符串
                xml_points_str = '<points '
                for img_idx, result in enumerate(results):
                    # xywh 格式：每行 [cx, cy, w, h]
                    xywh = result.boxes.xywh.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()

                    # 获取图像尺寸
                    img_width = result.orig_shape[1]  # 原始图像宽度
                    img_height = result.orig_shape[0]  # 原始图像高度

                    for j, (cx, cy, w, h) in enumerate(xywh):
                        label = result.names[classes[j]]
                        conf = confs[j]

                        # 将像素坐标转换为百分比坐标
                        x_percent = float((cx / img_width) * 100)  # 转换为百分比并转为Python float
                        y_percent = float((cy / img_height) * 100)  # 转换为百分比并转为Python float

                        # 添加到XML字符串
                        xml_points_str += f'x{j + 1}="{x_percent:.1f}" y{j + 1}="{y_percent:.1f}" '

                    # 添加alt属性和结束标签
                    if len(xywh) > 0:
                        xml_points_str += f'alt="{label}s">{label}s</points>'
                    else:
                        xml_points_str += 'alt="building">building</points>'

                # 使用parse_points函数解析XML字符串
                parsed_points = parse_points(xml_points_str)

                logger.debug(f'Parsed points for image {i}: {parsed_points}')

                csv_file_path = 'data/parsed_coordinates.csv'
                coordinates_dict = read_coordinates_from_csv(csv_file_path)

                result_coordinates = recalculate_coordinates(parsed_points, i, coordinates_dict)

                output_path = f'identified_new_data/identified{i}.jpg'
                draw_dots_and_lines_on_image(image_path, parsed_points, output_path=output_path)

                list_of_the_resulted_coordinates_percentage.append(parsed_points)
                list_of_the_resulted_coordinates_lat_lon.append(result_coordinates)
                all_result_coordinates.append(result_coordinates)  # 新增：收集每张图片的result_coordinates

            except Exception as e:
                logger.error(f'Error processing image {i}: {str(e)}')
                # 即使出错也要保证all_result_coordinates有元素（可选：可append空列表或None）
                all_result_coordinates.append([])
                continue

    except Exception as e:
        logger.error(f'Error in find_objects: {str(e)}')
        raise

    # 返回所有图片的result_coordinates（如有需要可自定义格式）
    return json.dumps(
        all_result_coordinates), list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon


def nearest_neighbor_path(coords: List[Tuple[float, float]]) -> List[int]:
    n = len(coords)
    if n == 0:
        return []
    visited = [False] * n
    path = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = path[-1]
        min_dist = float('inf')
        next_idx = -1
        for j in range(n):
            if not visited[j]:
                dist = np.linalg.norm(np.array(coords[last]) - np.array(coords[j]))
                if dist < min_dist:
                    min_dist = dist
                    next_idx = j
        path.append(next_idx)
        visited[next_idx] = True
    return path


import cv2
import numpy as np

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """在 img 的 pos 位置以 alpha_mask 作为透明度，将 img_overlay 叠加到 img 上。"""
    x, y = pos
    # 叠加区域
    h, w = img_overlay.shape[:2]
    if x + w > img.shape[1] or y + h > img.shape[0] or x < 0 or y < 0:
        return  # 超出边界则跳过
    roi = img[y:y+h, x:x+w]
    # 混合
    alpha = alpha_mask / 255.0
    inv_alpha = 1.0 - alpha
    for c in range(0, 3):
        roi[:, :, c] = (alpha * img_overlay[:, :, c] + inv_alpha * roi[:, :, c])
    img[y:y+h, x:x+w] = roi

def animate_flight_path(image_path, ordered_points, window_name="Flight Path Animation", delay=30):
    """
    动态演示路径规划过程，窗口1280x728，图片等比缩放居中显示，飞机图标动画。
    """
    # 目标窗口尺寸
    target_w, target_h = 1280, 728
    # 读取底图
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    frame_bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    # 将百分比坐标转换为像素坐标
    points_px = [
        (int(x / 100.0 * w * scale) + offset_x,
         int(y / 100.0 * h * scale) + offset_y)
        for x, y in ordered_points
    ]

    # 1. 加载无人机图标（带透明通道），并统一缩放到合适大小
    icon = cv2.imread('data/dji.png', cv2.IMREAD_UNCHANGED)  # RGBA
    if icon is None or icon.shape[2] != 4:
        raise ValueError("请检查 data/dji.png 是否存在且带有透明通道")
    # 缩放到大约 40x40 像素左右（可根据需求调整 zoom_factor）
    zoom_factor = 40.0 / max(icon.shape[0], icon.shape[1])
    icon = cv2.resize(icon, (0, 0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_AREA)

    # 分离图标的 color & alpha
    icon_color = icon[:, :, :3]
    icon_alpha = icon[:, :, 3]

    unlocked_points = [points_px[0]]
    for seg in range(len(points_px) - 1):
        p0, p1 = points_px[seg], points_px[seg + 1]
        # 飞机起飞前，先“解锁”并绘制下一个目标点
        unlocked_points.append(p1)

        # 飞机飞向下一个点
        for t in np.linspace(0, 1, 20):
            x = int(p0[0] + (p1[0] - p0[0]) * t)
            y = int(p0[1] + (p1[1] - p0[1]) * t)
            # 计算航向角（度）
            # 1. 先计算 dx, dy
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            # 2. 把图像坐标系 y 轴（向下为正）翻转成数学坐标系 y 轴（向上为正），再算角度
            # 3. 图标默认朝上，额外 +90°
            angle = np.degrees(np.arctan2(-dy, dx)) + 90

            # 1) 准备底图
            frame = frame_bg.copy()
            frame[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = img_resized

            # 2) 绘制路径
            for j in range(seg):
                cv2.line(frame, points_px[j], points_px[j + 1], (0, 255, 0), 2)
            cv2.line(frame, p0, (x, y), (0, 255, 0), 2)

            # 3) 绘制已解锁的点
            for pt in unlocked_points:
                cv2.circle(frame, pt, 7, (0, 0, 255), -1)

            # 4) 旋转图标并叠加
            # 生成旋转矩阵
            rows, cols = icon_color.shape[:2]
            M = cv2.getRotationMatrix2D((cols//2, rows//2), angle, 1.0)
            rotated_icon = cv2.warpAffine(icon_color, M, (cols, rows), flags=cv2.INTER_LINEAR)
            rotated_alpha = cv2.warpAffine(icon_alpha, M, (cols, rows), flags=cv2.INTER_LINEAR)
            # 计算图标左上角放置坐标
            top_left = (x - cols//2, y - rows//2)
            overlay_image_alpha(frame, rotated_icon, top_left, rotated_alpha)

            # 显示
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay) & 0xFF == 27:  # ESC 退出
                cv2.destroyAllWindows()
                return

    # 最终画面：完整路径 + 终点上的无人机
    frame = frame_bg.copy()
    frame[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = img_resized
    for j in range(len(points_px) - 1):
        cv2.line(frame, points_px[j], points_px[j + 1], (0, 255, 0), 2)
    for pt in points_px:
        cv2.circle(frame, pt, 7, (0, 0, 255), -1)
    # 在终点放置不旋转的图标
    cols, rows = icon_color.shape[1], icon_color.shape[0]
    top_left = (points_px[-1][0] - cols//2, points_px[-1][1] - rows//2)
    overlay_image_alpha(frame, icon_color, top_left, icon_alpha)

    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def generate_drone_mission_python():
    try:
        t1_find_objects = time()
        # 只处理一张图片
        objects_json, coords_percentage, coords_latlon = find_objects('{"object_types": ["building"]}', example_objects)
        t2_find_objects = time()
        del_t_find_objects = (t2_find_objects - t1_find_objects) / 60
        logger.info(f'Found {len(coords_latlon)} coordinate sets')
        os.makedirs("created_missions", exist_ok=True)
        if coords_latlon and len(coords_latlon) > 0:
            coords_dict = coords_latlon[0]  # 经纬度字典
            percentage_dict = coords_percentage[0]  # 百分比坐标字典
            # 提取所有点的经纬度和百分比坐标，保持顺序一致
            latlon_points = [v["coordinates"] for v in coords_dict.values()]
            percentage_points = [v["coordinates"] for v in percentage_dict.values()]
            order = nearest_neighbor_path(latlon_points)
            # 生成mission.txt
            mission_lines = [
                "arm throttle",
                "takeoff 100"
            ]
            for idx in order:
                lat, lon = latlon_points[idx]
                mission_lines.append(f"mode guided {lat} {lon} 100")
                mission_lines.append("mode circle")
            mission_lines.append("mode rtl")
            mission_lines.append("disarm")
            mission_text = "\n".join(mission_lines)
            with open("created_missions/mission1.txt", "w", encoding="utf-8") as f:
                f.write(mission_text)
            logger.info('Generated mission plan (python optimized)')
            # 绘制路径图（用百分比坐标顺序）
            ordered_points_dict = {}
            for i, idx in enumerate(order):
                key = f"point{i + 1}"
                ordered_points_dict[key] = {"type": "building", "coordinates": percentage_points[idx]}
            image_path = "data/images/1.jpg"
            output_path = "identified_new_data/flight_path.jpg"
            draw_dots_and_lines_on_image(image_path, ordered_points_dict, output_path=output_path)
            # 动态演示路径
            animate_flight_path(image_path, [percentage_points[idx] for idx in order])
        else:
            logger.warning('No coordinates found, skipping mission generation')
            return "", del_t_find_objects, 0.0
        return mission_text, del_t_find_objects, 0.0
    except Exception as e:
        logger.error(f'Error in generate_drone_mission_python: {str(e)}')
        raise


def run():
    try:
        logger.info('Starting UAV mission generation (python optimized)')
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        logger.info(f'Processing 1 sample')
        flight_plan, vlm_time, mission_time = generate_drone_mission_python()
        logger.info('Mission generation complete')
        logger.info(f'VLM processing time: {vlm_time:.2f} mins')
    except Exception as e:
        logger.error(f'Error in main execution: {str(e)}')
        raise


if __name__ == "__main__":
    run()