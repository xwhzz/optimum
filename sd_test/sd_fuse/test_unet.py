import onnxruntime as ort
import numpy as np
import argparse
import time

np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='unet')
parser.add_argument('--choice', choices=['c', 'p'], default='p')
args = parser.parse_args()
WARMUP = 3
if args.choice == 'p':
    f = open(f"log_{args.type}.txt", "w+")

provider = "CUDAExecutionProvider"

def run_fuse(index: list[int], data):
    so = ort.SessionOptions()
    path = f"./{args.type}/model.onnx"

    sess = ort.InferenceSession(path, so, providers=[provider])
    param = {
        'sample': data[0],
        'timestep': data[1],
        'encoder_hidden_states': data[2],
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
        for _ in range(WARMUP):
            txout = sess.run(None, param)
        txout = sess.run(None, param)
    return txout[0]

def run_model(path: str, start: int, end: int, data):
    so = ort.SessionOptions()

    sess = ort.InferenceSession(path,so,providers=[provider])

    param = {
        'sample': data[0][start: end],
        'timestep': data[1][start: end],
        'encoder_hidden_states': data[2][start: end],
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
        for _ in range(WARMUP):
            txout = sess.run(None, param)
        txout = sess.run(None, param)
    return txout[0]

def test_res(model_path: list[str], index: list[int], data):
    res = []
    cur_index = 0
    for idx, p in enumerate(model_path):
        if index[idx] != 0:
            res.append(run_model(p, cur_index, cur_index + index[idx], data))
        if args.choice == 'p':
        # if True:
            res.append(run_model(p, 0, sum(index), data))
        cur_index += index[idx]

    res_cat = np.concatenate(res, axis=0)

    print("======fuse_model=========")
    fuse_res = run_fuse(index, data)
    if args.choice == 'c':
        assert np.allclose(res_cat, fuse_res, atol=1e-3), (res_cat, fuse_res)

if __name__ == '__main__':
    model_path = [f'../org_sd_v15_{i}_onnx/{args.type}/model.onnx' for i in range(1, 7)]
    data = [np.random.random((6,4,64,64)).astype(np.float16),
            np.array([70,60,50,40,30,50]).astype(np.int64),
            np.random.random((6,77,768)).astype(np.float16)]
    # test_res(model_path, [1,1,1,1,1,1],data)
    # test_res(model_path, [2,3],data)
    # test_res(model_path, [2,2,1],data)
    test_res(model_path, [3,3,0,0,0,0],data)
    f.close()

