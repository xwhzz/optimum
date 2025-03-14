from optimum.onnxruntime import ORTStableDiffusionPipeline
import numpy as np
import time

model_1 = "../sd_test/org_sd_v15_1_onnx"
model_2 = "../sd_test/org_sd_v15_2_onnx"
model_3 = "../sd_test/org_sd_v15_3_onnx"
model_fuse = "../sd_test/sd_fuse_1"
p_1 = ORTStableDiffusionPipeline.from_pretrained(model_1).to('cuda:4')
p_2 = ORTStableDiffusionPipeline.from_pretrained(model_2).to('cuda:5')
# p_3 = ORTStableDiffusionPipeline.from_pretrained(model_3).to('cuda:6')
# p_fuse = ORTStableDiffusionPipeline.from_pretrained(model_fuse).to('cuda:2')
prompt = ["sailing ship in storm by Leonardo da Vinci"] * 3

def run_model(pipeline, model_index: int, times: int = 5, default: int = 3):
    exec_time = []
    for i in range(times):
        generator = np.random.RandomState(0)
        st = time.perf_counter()
        fuse_info = None
        if model_index == 4:
            fuse_info = [1, 1, 1]
        image = pipeline(prompt[:default],num_inference_steps=[50, 50, 50][:default], generator=generator, fuse_info=fuse_info).images
        exec_time.append(time.perf_counter() - st)
        # if i == 0:
        #     for j in range(default):
        #         image[j].save(f"images_1/image_{model_index}_{j}.png")
    return exec_time

print(run_model(p_1, 1,default=1)) #2.47
print(run_model(p_2, 2,default=1)) #2.71
# print(run_model(p_3, 3,default=3)) # -> 2.71 6.97
# print(run_model(p_fuse, 4)) # [40.353453871794045, 16.696387556381524, 16.679170640185475, 16.699736466631293, 16.682206073775887]
# 8.29 -> 7.48 11.8
# 50 step : 11.83

# 3.84 + 4.23 * 2 = 12.3