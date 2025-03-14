import onnx

graph = onnx.load('model.onnx').graph

for node in graph.node:
    if '_route' in node.name or '_merge' in node.name:
        if node.attribute[0].i == -1:
            print(f'{node.name} error!')
