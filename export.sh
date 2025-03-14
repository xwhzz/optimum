#!/bin/bash

# lora_list=("" "ki.safetensors" "wu.safetensors" "fa.safetensors" "ji.safetensors" "ko.safetensors" "mi.safetensors")
lora_list=("ca.safetensors")
# iterate all lora list
for i in "${!lora_list[@]}"; do
    lora=${lora_list[$i]}
    i=8
    model_output="sd_test/sd_v15_${i}_onnx"
    
    # generate lora param
    if [[ -z "$lora" ]]; then
        lora_param=""
    else
        lora_param="--lora lora/$lora"
    fi
    
    # execute export command
    optimum-cli export onnx \
        --model /data/xwh/stable-diffusion-v1-5 sd_test/org_sd_v15_$i\_onnx \
        --task stable-diffusion \
        --sequence_length 77 $lora_param \
        --device cuda --fp16 \
        --optimize O2

    # copy model 
    cp -r sd_test/org_sd_v15_$i\_onnx $model_output

    echo "shape inference in text_encoder"
    python /home/xwh/project/model_fuse/tools/symbolic_shape_infer.py --input sd_test/org_sd_v15_$i\_onnx/text_encoder/model.onnx --output $model_output/text_encoder/model.onnx
    
    
    echo "shape inference in unet"
    python /home/xwh/project/model_fuse/tools/symbolic_shape_infer.py --input sd_test/org_sd_v15_$i\_onnx/unet/model.onnx --output $model_output/unet/model.onnx --save_as_external_data --all_tensors_to_one_file --external_data_location model.onnx_data

done

# cp -r sd_test/org_sd_v15_0_onnx sd_test/sd_fuse
# echo "Generate sd_fuse model"