from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import Optional, Dict, Any
import uvicorn

from collections import deque
import logging
import uuid
import io
from PIL import Image
import time
from optimum.onnxruntime import ORTStableDiffusionPipeline

# TODO: use real ort model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    prompt: str
    stage: int = 0
    cur_step: int = 0
    lora_tag: int = 1
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 50 # 后面可以得到不同的步长，以及不同的lora_tag
    guidance_scale: Optional[float] = 7.5

class MockModel:
    def text_worker(self, request):
        time.sleep(0.1)

    def unet_worker(self, request):
        time.sleep(0.3)
        
    def vae_worker(self, request):
        time.sleep(0.2)
        return [Image.new('RGB', (256, 256)) for _ in request]

class SdEngine:
    def __init__(self, path: str, device: str):
        self.model = ORTStableDiffusionPipeline.from_pretrained(path).to(device)
        self.requests = [deque() for _ in range(3)]
        self.max_batch_size = [8, 8, 8]
        self._running = False
        self._task = None

        self.pending_requests: Dict[str, asyncio.Event] = {}
        self.request_results: Dict[str, Any] = {}
        self.request_timestamps: Dict[str, float] = {}
        
        self.cleanup_task = None

        self.last_worker = 0

    async def start_cleanup(self):
        """Initialize the cleanup task - called during FastAPI startup"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_old_requests())
            logger.info("Started cleanup task")

    async def _cleanup_old_requests(self, max_age: int = 3600):
        """Cleanup requests older than max_age seconds"""
        while True:
            try:
                current_time = time.time()
                expired_requests = [
                    req_id for req_id, timestamp in self.request_timestamps.items()
                    if current_time - timestamp > max_age
                ]
                
                for req_id in expired_requests:
                    self.pending_requests.pop(req_id, None)
                    self.request_results.pop(req_id, None)
                    self.request_timestamps.pop(req_id, None)
                    
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # Sleep on error

    async def add_request(self, request: dict, worker: int) -> str:
        if worker == 0:
            request_id = str(uuid.uuid4())
            request['id'] = request_id
            self.pending_requests[request_id] = asyncio.Event()
            self.request_timestamps[request_id] = time.time()
        request_id = request.get('id',None)
        self.requests[worker].append(request)
        logger.info(f"Added request {request_id} to worker {worker}. Queue length: {len(self.requests[worker])}")
        
        return request_id

    async def schedule(self) -> tuple[list, int]:
        request = []
        worker = self.last_worker

        # for i in [2,0,1]:
        #     if len(self.requests[i]) > 0:
        #         worker = i
        #         request_queue = self.requests[i]
        #         do_unet = i == 1
        #         for idx in range(min(self.max_batch_size[i], len(request_queue))):
        #             if not do_unet:
        #                 request.append(request_queue.popleft())
        #             else:
        #                 request.append(request_queue[idx])
        #         break
        for _ in range(3):
            worker = (worker + 1) % 3
            if len(self.requests[worker]) > 0:
                do_unet = worker == 1
                request_queue = self.requests[worker]
                for idx in range(min(self.max_batch_size[worker], len(request_queue))):
                    if not do_unet:
                        request.append(request_queue.popleft())
                    else:
                        request.append(request_queue[idx])
                self.last_worker = worker
                break
        if not request:
            return request, -1
        return request, worker

    async def step(self, request: list, worker: int):
        if not request:
            return 
        try:
            match worker:
                case 0:
                    await asyncio.to_thread(self.model.text_worker, request)
                    logger.info(f"Processed batch of {len(request)} requests in text worker")
                case 1:
                    await asyncio.to_thread(self.model.unet_worker, request)
                    logger.info(f"Processed batch of {len(request)} requests in unet worker 1")
                case 2:
                    result = await asyncio.to_thread(self.model.vae_worker, request)
                    logger.info(f"Processed batch of {len(request)} requests in vae worker")
                    for idx, r in enumerate(request):
                        if 'id' in r:
                            self.request_results[r['id']] = result[idx]
                            event = self.pending_requests.get(r['id'])
                            if event:
                                event.set()
                case _:
                    raise ValueError("worker index out of range")
                    
            for r in request:
                if worker == 1:
                    r["cur_step"] += 1
                    if r["cur_step"] == r["inference_steps"]:
                        r["stage"] += 1 
                        self.requests[1].remove(r)
                        await self.add_request(r, 2)
                else:
                    next_stage = r["stage"] + 1
                    if next_stage < 3:
                        r["stage"] = next_stage
                        await self.add_request(r, next_stage)
                    
        except Exception as e:
            logger.error(f"Error in step execution: {str(e)}")
            for r in request:
                if 'id' in r:
                    event = self.pending_requests.get(r['id'])
                    if event:
                        event.set()
            raise

    async def run_forever(self):
        """Continuous processing loop that runs indefinitely"""
        self._running = True
        logger.info("Starting continuous processing loop")
        
        while self._running:
            try:
                request, worker = await self.schedule()
                if worker == -1 or not request:
                    await asyncio.sleep(0.01)
                    continue
                
                await self.step(request, worker)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(1)
                
        logger.info("Continuous processing loop stopped")

    async def start(self):
        """Start the continuous processing loop"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_forever())
            await self.start_cleanup()  # Start the cleanup task
            logger.info("Started continuous processing and cleanup tasks")

    async def stop(self):
        """Stop the continuous processing loop"""
        if self._running:
            self._running = False
            if self._task:
                await self._task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            logger.info("Stopped continuous processing loop")

    async def wait_for_result(self, request_id: str, timeout: float = 60.0) -> Image.Image:
        """Wait for the result of a specific request"""
        event = self.pending_requests.get(request_id)
        if not event:
            raise HTTPException(status_code=404, detail="Request not found")
            
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout")
            
        result = self.request_results.get(request_id)
        if result is None:
            raise HTTPException(status_code=500, detail="Processing failed")
            
        return result


engine = SdEngine("/home/xwh/project/optimum/sd_test/sf", "cuda:0")

from contextlib import asynccontextmanager

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    await engine.start()
    yield
    await engine.stop()

app = FastAPI(lifespan=app_lifespan)


@app.post("/generate")
async def generate_image(request: GenerationRequest):
    request_dict = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "stage": request.stage,
        "cur_step": request.cur_step,
        "lora_tag": request.lora_tag
    }
    
    request_id = await engine.add_request(request_dict, 0)
    
    image = await engine.wait_for_result(request_id)
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.get("/status")
async def get_status():
    return {
        "queue_lengths": [len(q) for q in engine.requests],
        "running": engine._running,
        "pending_requests": len(engine.pending_requests)
    }

if __name__ == "__main__":
    uvicorn.run(app, port=8002,)