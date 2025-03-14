import aiohttp
import asyncio
import time
import argparse
import json
from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import gamma
from dataclasses import dataclass
import logging
from pathlib import Path
import random

random.seed(0)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RequestStats:
    arrival_time: float  # When the request was generated
    start_time: float    # When the request started processing
    end_time: float      # When the request finished
    latency: float       # Total processing time
    status: int
    success: bool
    error_msg: str = ""

class GammaProcessSimulator:
    def __init__(self, arrival_rate: float, cv: float):
        """
        Initialize Gamma process simulator
        Args:
            shape: shape parameter (k) of the gamma distribution
            scale: scale parameter (θ) of the gamma distribution
        """
        self.shape = 1 / cv**2
        self.scale = cv**2 / arrival_rate

    def generate_interarrival_times(self, n: int) -> np.ndarray:
        """Generate n interarrival times following gamma distribution"""
        return gamma.rvs(self.shape, scale=self.scale, size=n)

    def generate_arrival_times(self, n: int, start_time: float = 0) -> np.ndarray:
        """Generate absolute arrival times for n requests"""
        interarrival_times = self.generate_interarrival_times(n)
        return np.cumsum(interarrival_times) + start_time

class SDBenchmark:
    def __init__(self, 
                 server_url: str, 
                 save_dir: str = "benchmark_results",
                 arrival_shape: float = 2.0,
                 arrival_scale: float = 0.5):
        
        self.server_url = server_url
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize gamma process simulator
        self.arrival_process = GammaProcessSimulator(arrival_shape, arrival_scale)
        
        # Statistics storage
        self.request_stats: List[RequestStats] = []
        self.start_time: float = 0
        self.end_time: float = 0

    async def make_single_request(self, 
                                session: aiohttp.ClientSession, 
                                prompt: str,
                                arrival_time: float,
                                index: int,
                                lora_tag: int,
                                step: int
                                ) -> RequestStats:
        start_time = time.time()
        
        try:
            payload = {
                "prompt": prompt,
                "negative_prompt": "",
                "num_inference_steps": step,
                "guidance_scale": 7.5,
                "lora_tag": lora_tag
            }
            
            async with session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                end_time = time.time()
                
                if response.status == 200:
                    # Save the image
                    image_data = await response.read()
                    image_path = self.save_dir / f"image_{index}.png"
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    
                    return RequestStats(
                        arrival_time=arrival_time,
                        start_time=start_time,
                        end_time=end_time,
                        latency=end_time - start_time,
                        status=response.status,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return RequestStats(
                        arrival_time=arrival_time,
                        start_time=start_time,
                        end_time=end_time,
                        latency=end_time - start_time,
                        status=response.status,
                        success=False,
                        error_msg=error_text
                    )
                    
        except Exception as e:
            end_time = time.time()
            return RequestStats(
                arrival_time=arrival_time,
                start_time=start_time,
                end_time=end_time,
                latency=end_time - start_time,
                status=500,
                success=False,
                error_msg=str(e)
            )

    async def run_gamma_process_benchmark(
        self,
        num_requests: int,
        prompt: str
    ):
        self.start_time = time.time()
        
        # Generate arrival times using gamma process
        arrival_times = self.arrival_process.generate_arrival_times(num_requests, self.start_time)
        lora_tag = [ random.randint(0,4) for _ in range(num_requests)]
        step_list = list(range(20,51, 5))
        steps = [random.choice(step_list) for _ in range(num_requests)]
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            async def scheduled_request(arrival_time: float, index: int):
                # Wait until the scheduled arrival time
                wait_time = arrival_time - time.time()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                return await self.make_single_request(session, prompt, arrival_time, index, lora_tag[index],steps[index])
            
            # Create tasks for all requests
            for idx, arrival_time in enumerate(arrival_times):
                tasks.append(asyncio.create_task(scheduled_request(arrival_time, idx)))
                
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks)
            self.request_stats.extend(results)
            
        self.end_time = time.time()

    def calculate_statistics(self) -> Dict:
        if not self.request_stats:
            return {}
            
        latencies = [stat.latency for stat in self.request_stats]
        interarrival_times = np.diff([stat.arrival_time for stat in self.request_stats])
        
        stats = {
            "total_requests": len(self.request_stats),
            "successful_requests": sum(stat.success for stat in self.request_stats),
            "failed_requests": sum(not stat.success for stat in self.request_stats),
            "total_time": self.end_time - self.start_time,
            
            # Latency statistics
            "latency_stats": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p90": np.percentile(latencies, 90),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": min(latencies),
                "max": max(latencies)
            },
            
            # Interarrival time statistics
            "interarrival_stats": {
                "mean": np.mean(interarrival_times),
                "median": np.median(interarrival_times),
                "p90": np.percentile(interarrival_times, 90),
                "p95": np.percentile(interarrival_times, 95)
            },
            
            "throughput": len(self.request_stats) / (self.end_time - self.start_time),
            "arrival_rate": len(self.request_stats) / (max([s.arrival_time for s in self.request_stats]) - min([s.arrival_time for s in self.request_stats]))
        }
        
        # Calculate empirical gamma parameters for actual interarrival times
        stats["empirical_interarrival_gamma"] = {
            "shape": np.mean(interarrival_times)**2 / np.var(interarrival_times),
            "scale": np.var(interarrival_times) / np.mean(interarrival_times)
        }
        
        # Error distribution
        error_counts = {}
        for stat in self.request_stats:
            if not stat.success:
                error_counts[stat.error_msg] = error_counts.get(stat.error_msg, 0) + 1
        stats["error_distribution"] = error_counts
        
        return stats

    def save_statistics(self, stats: Dict, filename: str = "benchmark_stats.json"):
        filepath = self.save_dir / filename
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {filepath}")

async def main():
    parser = argparse.ArgumentParser(description="Benchmark Stable Diffusion Server with Gamma Process")
    parser.add_argument("--server-url", default="http://localhost:8002", help="Server URL")
    parser.add_argument("--num-requests", type=int, default=400, help="Total number of requests")
    parser.add_argument("--prompt", default="a beautiful landscape with mountains", help="Prompt for image generation")
    parser.add_argument("--save-dir", default="benchmark_results", help="Directory to save results")
    parser.add_argument("--arrival-shape", type=float, default=2.0, help="Shape parameter for arrival process")
    parser.add_argument("--arrival-scale", type=float, default=1.0, help="Scale parameter for arrival process")
    
    args = parser.parse_args()
    
    logger.info(f"Starting gamma process benchmark with {args.num_requests} total requests")
    logger.info(f"Arrival process: shape={args.arrival_shape}, scale={args.arrival_scale}")
    
    benchmark = SDBenchmark(
        args.server_url,
        args.save_dir,
        args.arrival_shape,
        args.arrival_scale
    )
    
    await benchmark.run_gamma_process_benchmark(
        num_requests=args.num_requests,
        prompt=args.prompt
    )
    
    stats = benchmark.calculate_statistics()
    benchmark.save_statistics(stats)
    
    logger.info("\nBenchmark Results:")
    logger.info(f"Total Requests: {stats['total_requests']}")
    logger.info(f"Success Rate: {stats['successful_requests']/stats['total_requests']*100:.2f}%")
    logger.info(f"Average Latency: {stats['latency_stats']['mean']:.2f}s")
    logger.info(f"P95 Latency: {stats['latency_stats']['p95']:.2f}s")
    logger.info(f"Throughput: {stats['throughput']:.2f} requests/second")
    logger.info(f"Arrival Rate: {stats['arrival_rate']:.2f} requests/second")
    
    logger.info("\nEmpirical Parameters:")
    logger.info(f"Interarrival Time Gamma: shape={stats['empirical_interarrival_gamma']['shape']:.2f}, "
                f"scale={stats['empirical_interarrival_gamma']['scale']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())