from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os
import logging
from typing import Optional
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLMRunner:
    """用于加载和推理 VLM 模型的类。"""

    def __init__(self, model_name: str = 'cyan2k/molmo-7B-O-bnb-4bit'):
        """
        初始化 VLM 执行器。

        参数:
            model_name: 要从 HuggingFace 加载的模型名称
        """
        self.model_name = model_name
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """
        加载 VLM 模型和处理器。

        异常:
            RuntimeError: 如果 CUDA 不可用
            Exception: 其他模型加载错误
        """
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA 不可用，模型将使用 CPU 加速。")

            logger.info(f"正在加载模型: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )

            logger.info("模型加载成功")

        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            raise

    def process_image(self, image_path: str, query: str) -> Optional[str]:
        """
        使用 VLM 模型处理单张图像。

        参数:
            image_path: 图像文件的路径
            query: 给模型的文本查询

        返回:
            生成的文本，处理失败时返回 None

        异常:
            FileNotFoundError: 如果图像文件不存在
            Exception: 其他处理错误
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像未找到: {image_path}")

            if self.processor is None or self.model is None:
                self.load_model()

            # 加载并确保图像是 RGB 格式
            image = Image.open(image_path).convert('RGB')

            # 处理图像和文本
            inputs = self.processor.process(
                images=[image],
                text=query
            )

            # 将输入移动到设备上
            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

            # 生成输出
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

            # 解码输出
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            return generated_text

        except Exception as e:
            logger.error(f"处理图像 {image_path} 时出错: {e}")
            return None


def run() -> None:
    """
    主函数，运行 VLM 处理单张图像。
    """
    try:
        # 获取图像文件路径
        image_path = 'data/images/1.jpg'  # 只处理单张图片
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件未找到: {image_path}")

        logger.info(f"正在处理单张图像: {image_path}")

        # 初始化 VLM 执行器
        vlm_runner = VLMRunner()

        # 处理单张图像
        query = "This is the satellite image of a city. Please, point all the buildings."

        generated_text = vlm_runner.process_image(image_path, query)

        if generated_text:
            with open('identified_points.txt', 'w') as f:
                f.write(f"1, {generated_text}\n")
            logger.debug(f"图像 1 生成的文本: {generated_text}")
        else:
            logger.warning(f"处理图像 1 失败")

        logger.info("处理完成")

    except Exception as e:
        logger.error(f"主执行过程中出错: {e}")
        raise


if __name__ == "__main__":
    run()
