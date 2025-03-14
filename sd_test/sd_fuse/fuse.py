import onnx
import argparse
import netron

from opt.converter import ONNXConverter
from opt.passes import fuse_base, add_op, combine

config = {}

def get_graph(path: str, index: int):
    model = onnx.load(path)
    if index == 0:
        config["input_name"] = [inp.name for inp in model.graph.input]
        config["output_name"] = [out.name for out in model.graph.output]
    model = onnx.compose.add_prefix(model, f'{index}_')
    return model.graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='text_encoder')

    args = parser.parse_args()

    model_type = args.type
    model_num = 6
    g_1 = get_graph(f'../sd_v15_1_onnx/{model_type}/model.onnx', 0)
    print(f"Config: {config}")
    converter = ONNXConverter([inp.type for inp in g_1.input], [out.type for out in g_1.output], model_num, config["input_name"], config["output_name"])
    
    g_1 = converter.to_graph(g_1)
    print('Convert g1 to graph!')
    for i in range(1,model_num):
        g = get_graph(f'../sd_v15_{i + 1}_onnx/{model_type}/model.onnx', i)
        g = converter.to_graph(g)
        print(f'Convert g{i+1} to graph!')
        g_1 = fuse_base(g_1, g)
        print(f"{i}th fuse complete!")
    add_op(g_1)
    print("Add Op Complete!")
    # if model_type != "unet":
    if True:
        combine(g_1)
        print("Remove Op Complete!")
    fuse_graph = converter.from_graph(g_1)
    large = False
    if model_type == "unet":
        large = True
    converter.export_file(fuse_graph, f'./{model_type}/model.onnx', large)
    netron.start(f'./{model_type}/model.onnx')
