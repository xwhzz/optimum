import onnxruntime as ort
import numpy as np
import argparse
import time
import torch

np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='text_encoder')
parser.add_argument('--choice', choices=['c', 'p'], default='c')
args = parser.parse_args()
WARMUP = 3
f = open(f"log_{args.type}.txt", "w+")

# if args.choice == 'c':
#     provider = "CPUExecutionProvider"
# else:

stream = torch.cuda.Stream()
provider = [("CUDAExecutionProvider", {"device_id": 0,
                                        "user_compute_stream": str(stream.cuda_stream)}),]


def run_fuse(index: list[int], data):
    so = ort.SessionOptions()

    # so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    path = f"./{args.type}/model.onnx"

    sess = ort.InferenceSession(path, so, providers=provider)
    param = { 
        'input_ids': data,
        "info": np.array(index).astype(np.int64)
    }
    if args.choice == 'p':
        for _ in range(WARMUP):
            txout = sess.run(None, param)
        
        t = []
        for _ in range(5):
            st = time.perf_counter()
            txout = sess.run(None, param)
            t.append(time.perf_counter() - st)
        print(t, sum(t)/len(t), file=f)
    else:
        txout = sess.run(None, param)
    return txout[0]

def run_model(path: str, start: int, end: int, data):
    so = ort.SessionOptions()

    sess = ort.InferenceSession(path,so,providers=provider)

    param = {
        'input_ids': data[start: end]
    }

    if args.choice == 'p':
        for _ in range(WARMUP):
            txout = sess.run(None, param)
        
        t = []
        for _ in range(5):
            st = time.perf_counter()
            txout = sess.run(None, param)
            t.append(time.perf_counter() - st)
        print(t, sum(t)/len(t), file=f)
    else:
        txout = sess.run(None, param)
    return txout[0]

def test_res(model_path: list[str], index: list[int], data):
    res = []
    cur_index = 0
    for idx, p in enumerate(model_path):
        res.append(run_model(p, cur_index, cur_index + index[idx], data))
        res.append(run_model(p, 0, sum(index), data))
        cur_index += index[idx]

    # res_cat = np.concatenate(res, axis=0)

    print("======fuse_model=========")
    fuse_res = run_fuse(index, data)
    if args.choice == 'c':
        return
        assert np.allclose(res_cat, fuse_res, atol=1e-3), (res_cat, fuse_res)

if __name__ == '__main__':
    model_path = [f'../org_sd_v15_{i}_onnx/{args.type}/model.onnx' for i in range(1, 3)]
    data_ls = []

    for i in range(0,20):
        data = []
        ran = np.random.randint(10,40)
        data.append(49406)
        for _ in range(ran):
            data.append(np.random.randint(10, 1000))
        for _ in range(ran+1,77):
            data.append(49407)

        assert len(data) == 77, len(data)
        data_ls.append(data)
    
    data = np.array(data_ls).astype(np.int32)
    test_res(model_path, [1,1],data[:2])
    # test_res(model_path, [4,4],data[:8])
    f.close()

