import requests
from PIL import Image
import io
from typing import Optional
from dataclasses import dataclass

@dataclass
class GenerationRequest:
    prompt: str
    stage: int = 0
    cur_step: int = 0
    lora_tag: int = 1
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

class StableDiffusionClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8002"):
        self.base_url = base_url.rstrip('/')

    def generate_image(self, request: GenerationRequest) -> Image.Image:
        """
        生成图片并返回PIL Image对象
        
        参数:
            request: GenerationRequest对象，包含生成图片所需的参数
            
        返回:
            PIL Image对象
        """
        url = f"{self.base_url}/generate"
        
        # 准备请求数据
        data = {
            "prompt": request.prompt,
            "stage": request.stage,
            "cur_step": request.cur_step,
            "lora_tag": request.lora_tag,
            "negative_prompt": request.negative_prompt,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale
        }
        
        # 发送请求
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        # 将响应内容转换为PIL Image
        return Image.open(io.BytesIO(response.content))

    def get_server_status(self) -> dict:
        """
        获取服务器状态
        
        返回:
            包含服务器状态信息的字典
        """
        url = f"{self.base_url}/status"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# 使用示例
if __name__ == "__main__":
    # 创建客户端实例
    client = StableDiffusionClient("http://127.0.0.1:8002")
    
    # 创建生成请求
    request = GenerationRequest(
        prompt="一只可爱的猫咪",
        negative_prompt="模糊的，低质量的",
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    try:
        # 生成图片
        image = client.generate_image(request)
        # 保存图片
        image.save("output.png")
        print("图片已保存为 output.png")
        
        # 获取服务器状态
        status = client.get_server_status()
        print("服务器状态:", status)
        
    except requests.RequestException as e:
        print(f"请求出错: {e}")