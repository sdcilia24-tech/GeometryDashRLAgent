import time
import random
import psutil
import os
import numpy as np
from GeometryGym import GeometryGym
import cv2
import numpy as np
import time
import pydirectinput as pdi
from FrameProcessor import FrameProcessor
def get_memory_usage():
    """Returns RAM used by current Python process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)                                            

def run_benchmark(num_episodes=25):
    print("=" * 60, flush=True)
    print("STARTING GEOMETRYGYM BENCHMARK TEST", flush=True)
    print("=" * 60, flush=True)
    
    env = GeometryGym()
    total_steps = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---", flush=True)
        
        ep_start_time = time.time()
        ep_steps = 0
        state = env.reset()
        
        if state is None or state.shape != (4, 100, 100):
            print(f"State shape error on reset! Got: {type(state)}", flush=True)
            return

        dead = False
        while not dead:
            action = random.randint(0, 1)
            
            next_state, reward, dead, _ = env.step(action)
            
            ep_steps += 1
            total_steps += 1

            if next_state is None or next_state.shape != (4, 100, 100):
                print(f"State shape changed mid-episode: {type(next_state)}", flush=True)
                break

        ep_duration = time.time() - ep_start_time
        ep_fps = ep_steps / ep_duration if ep_duration > 0 else 0
        ram_mb = get_memory_usage()

        print(f"Episode {episode} Finished | Steps: {ep_steps} | Time: {ep_duration:.2f}s", flush=True)
        print(f"Episode Speed: {ep_fps:.1f} FPS | RAM: {ram_mb:.2f} MB", flush=True)

    total_duration = time.time() - start_time
    overall_fps = total_steps / total_duration if total_duration > 0 else 0
    
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total Episodes Run:  {num_episodes}", flush=True)
    print(f"Total Steps Taken:   {total_steps}", flush=True)
    print(f"Average Overall FPS: {overall_fps:.2f}", flush=True)
    print(f"Final RAM Usage:     {get_memory_usage():.2f} MB", flush=True)
    print("=" * 60, flush=True)
             
if __name__ == "__main__":
    print("starting in 3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    run_benchmark(num_episodes=5)                                                  