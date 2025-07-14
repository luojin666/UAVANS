from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import os
from typing import Tuple, List, Dict, Any
import logging
from time import time
from pydantic import SecretStr

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
        processor = AutoProcessor.from_pretrained(
            'cyan2k/molmo-7B-O-bnb-4bit',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        model = AutoModelForCausalLM.from_pretrained(
            'cyan2k/molmo-7B-O-bnb-4bit',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        
        find_objects_json_input = json_input.replace("`", "").replace("json","")
        find_objects_json_input_2 = json.loads(find_objects_json_input)
        
        search_string = ""
        for obj_type in find_objects_json_input_2["object_types"]:
            search_string += obj_type
            
        logger.info(f'Processing {NUMBER_OF_SAMPLES} samples')
        
        for i in range(1, NUMBER_OF_SAMPLES+1):
            logger.info(f'Processing image {i}')
            
            try:
                image_path = f'data/images/{i}.jpg'
                inputs = processor.process(
                    images=[Image.open(image_path).convert("RGB")],
                    text=f'This is the satellite image of a city. Please, point all the next objects: {search_string}'
                )
                
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
                
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
                
                generated_tokens = output[0,inputs['input_ids'].size(1):]
                generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                parsed_points = parse_points(generated_text)
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
    return json.dumps(all_result_coordinates), list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon

def generate_drone_mission(command: str, model_type: str = "openai") -> Tuple[str, float, float]:
    """
    Generate a complete drone mission plan.
    
    Args:
        command: Natural language command describing the mission
        model_type: Type of LLM to use ("openai", "qianfan", "zhipu", "dashscope")
        
    Returns:
        Tuple containing:
        - Flight plan text
        - Time taken to find objects
        - Time taken to generate mission
    """
    try:
        # 使用新的LLM初始化函数
        llm = initialize_llm(model_type)
        
        # Step 1: Extract object types
        step_1_prompt = PromptTemplate(input_variables=["command"], template=step_1_template)
        step_1_chain = step_1_prompt | llm
        
        # Step 3: Generate flight plan
        step_3_prompt = PromptTemplate(input_variables=["command", "objects"], template=step_3_template)
        step_3_chain = step_3_prompt | llm
        
        object_types_response = step_1_chain.invoke({"command": command})
        object_types_json = object_types_response.content
        
        # Step 2: Find objects on the map
        t1_find_objects = time()
        objects_json, coords_percentage, coords_latlon = find_objects(str(object_types_json), example_objects)
        t2_find_objects = time()
        del_t_find_objects = (t2_find_objects - t1_find_objects)/60
        
        logger.info(f'Found {len(coords_latlon)} coordinate sets')
        
        # Step 3: Generate flight plans
        t1_generate_drone_mission = time()
        os.makedirs("created_missions", exist_ok=True)
        
        flight_plan_response = None  # 新增：初始化变量，防止未定义
        
        for i, coords in enumerate(coords_latlon, 1):
            flight_plan_response = step_3_chain.invoke({
                "command": command,
                "objects": coords
            })
            mission_file = f"created_missions/mission{i}.txt"
            with open(mission_file, "w") as file:
                file.write(str(flight_plan_response.content))
            logger.info(f'Generated mission plan {i}')
        
        t2_generate_drone_mission = time()
        del_t_generate_drone_mission = (t2_generate_drone_mission - t1_generate_drone_mission)/60
        
        # 新增：如果没有生成任何flight_plan_response，返回空字符串或提示
        if flight_plan_response is None:
            return "", del_t_find_objects, del_t_generate_drone_mission
        
        return str(flight_plan_response.content), del_t_find_objects, del_t_generate_drone_mission
        
    except Exception as e:
        logger.error(f'Error in generate_drone_mission: {str(e)}')
        raise

def run(model_type: str = "deepseek"):
    """
    Main entry point for the UAV mission generation system.
    
    Args:
        model_type: Type of LLM to use ("openai", "qianfan", "zhipu", "dashscope")
    """
    try:
        logger.info('Starting UAV mission generation')
        logger.info(f'Using LLM: {model_type}')
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        logger.info(f'Processing {NUMBER_OF_SAMPLES} samples')
        
        flight_plan, vlm_time, mission_time = generate_drone_mission(command, model_type)
        total_time = vlm_time + mission_time
        
        logger.info('Mission generation complete')
        logger.info(f'VLM processing time: {vlm_time:.2f} mins')
        logger.info(f'Mission generation time: {mission_time:.2f} mins')
        logger.info(f'Total computational time: {total_time:.2f} mins')
        
    except Exception as e:
        logger.error(f'Error in main execution: {str(e)}')
        raise

if __name__ == "__main__":
    run()