for i in $(seq 1 2); do
    # onnxsim ../org_sd_v15_$i\_onnx/text_encoder/model.onnx ../org_sd_v15_$i\_onnx/text_encoder/model.onnx
    python /home/xwh/project/model_fuse/tools/symbolic_shape_infer.py --input ../org_sd_v15_$i\_onnx/text_encoder/model.onnx --output ../sd_v15_$i\_onnx/text_encoder/model.onnx
    echo "text_encoder done"
    python /home/xwh/project/model_fuse/tools/symbolic_shape_infer.py --input ../org_sd_v15_$i\_onnx/vae_decoder/model.onnx --output ../sd_v15_$i\_onnx/vae_decoder/model.onnx
    echo "vae_decoder done"
    python /home/xwh/project/model_fuse/tools/symbolic_shape_infer.py --input ../org_sd_v15_$i\_onnx/unet/model.onnx --output ../sd_v15_$i\_onnx/unet/model.onnx --save_as_external_data --all_tensors_to_one_file --external_data_location model.onnx_data
done