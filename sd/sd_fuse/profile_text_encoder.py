import onnxruntime as ort
import numpy as np



provider = ("CUDAExecutionProvider", {'enable_cuda_graph': False})

def profile_model(index: list[int], data):
    so = ort.SessionOptions()
    path = f"./text_encoder/model.onnx"

    sess = ort.InferenceSession(path, so, providers=[provider])
    param = { 
        'input_ids': data,
        "info": np.array(index).astype(np.int64)
    }
    txout = sess.run(None, param)
    return txout[0]

if __name__ == '__main__':
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
    profile_model([5,5], data[:10])
    # test_res(model_path, [5,5],data[:10])
    # test_res(model_path, [4,4],data[:8])
    # f.close()

