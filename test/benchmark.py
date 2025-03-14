import aiohttp
import asyncio
import time
import argparse
import json
from typing import List, Dict
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RequestStats:
    latency: float
    status: int
    success: bool
    error_msg: str = ""

class SDBenchmark:
    def __init__(self, server_url: str, save_dir: str = "benchmark_results"):
        self.server_url = server_url
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Statistics storage
        self.request_stats: List[RequestStats] = []
        self.start_time: float = 0
        self.end_time: float = 0

    async def make_single_request(self, session: aiohttp.ClientSession, prompt: str) -> RequestStats:
        start_time = time.time()
        
        try:
            payload = {
                "prompt": prompt,
                "negative_prompt": "",
                "num_inference_steps": 50,
                "guidance_scale": 7.5
            }
            
            async with session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            ) as response:
                if response.status == 200:
                    # Save the image
                    image_data = await response.read()
                    image_path = self.save_dir / f"image_{len(self.request_stats)}.png"
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    
                    return RequestStats(
                        latency=time.time() - start_time,
                        status=response.status,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return RequestStats(
                        latency=time.time() - start_time,
                        status=response.status,
                        success=False,
                        error_msg=error_text
                    )
                    
        except Exception as e:
            return RequestStats(
                latency=time.time() - start_time,
                status=500,
                success=False,
                error_msg=str(e)
            )

    async def run_concurrent_requests(
        self,
        num_requests: int,
        concurrency: int,
        prompt: str
    ):
        self.start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_request():
                async with semaphore:
                    return await self.make_single_request(session, prompt)
            
            for _ in range(num_requests):
                tasks.append(asyncio.create_task(bounded_request()))
                
            results = await asyncio.gather(*tasks)
            self.request_stats.extend(results)
            
        self.end_time = time.time()

    def calculate_statistics(self) -> Dict:
        if not self.request_stats:
            return {}
            
        latencies = [stat.latency for stat in self.request_stats]
        successes = [stat.success for stat in self.request_stats]
        
        stats = {
            "total_requests": len(self.request_stats),
            "successful_requests": sum(successes),
            "failed_requests": len(self.request_stats) - sum(successes),
            "success_rate": sum(successes) / len(self.request_stats) * 100,
            "total_time": self.end_time - self.start_time,
            "average_latency": np.mean(latencies),
            "median_latency": np.median(latencies),
            "p90_latency": np.percentile(latencies, 90),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "throughput": len(self.request_stats) / (self.end_time - self.start_time)
        }
        
        # Collect error statistics
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
    parser = argparse.ArgumentParser(description="Benchmark Stable Diffusion Server")
    parser.add_argument("--server-url", default="http://localhost:8004", help="Server URL")
    parser.add_argument("--num-requests", type=int, default=10, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--prompt", default="a beautiful landscape with mountains", help="Prompt for image generation")
    parser.add_argument("--save-dir", default="benchmark_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    logger.info(f"Starting benchmark with {args.num_requests} total requests, {args.concurrency} concurrent requests")
    
    benchmark = SDBenchmark(args.server_url, args.save_dir)
    
    await benchmark.run_concurrent_requests(
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        prompt=args.prompt
    )
    
    stats = benchmark.calculate_statistics()
    benchmark.save_statistics(stats)
    
    logger.info("\nBenchmark Results:")
    logger.info(f"Total Requests: {stats['total_requests']}")
    logger.info(f"Success Rate: {stats['success_rate']:.2f}%")
    logger.info(f"Average Latency: {stats['average_latency']:.2f}s")
    logger.info(f"P95 Latency: {stats['p95_latency']:.2f}s")
    logger.info(f"Throughput: {stats['throughput']:.2f} requests/second")
    
    if stats['failed_requests'] > 0:
        logger.info("\nError Distribution:")
        for error, count in stats['error_distribution'].items():
            logger.info(f"{error}: {count} occurrences")

if __name__ == "__main__":
    asyncio.run(main())