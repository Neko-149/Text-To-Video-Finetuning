set -x 

#pathvar="$( cd "$( dirname $0 )" && pwd )"
#cd $pathvar/../..
#export PYTHONPATH=PYTHONPATH:$pathvar/../..
#export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
#pip install accelerate==0.19
#pip install transformers==4.26.1
#pip install diffusers==0.21.0
#git+https://github.com/cloneofsimo/lora.git
#git+https://github.com/microsoft/LoRA
#pip install compel
#pip install loralib
#pip install imageio[ffmpeg]

#accelerate launch \
#   --multi_gpu --same_network --num_machines 1   --num_processes=8 --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT  --machine_rank $RANK --gpu_ids '0,1,2,3,4,5,6,7' \
#   /nas/yxq/project/Text-To-Video-Finetuning-main/train.py \
#   --config="/nas/yxq/project/Text-To-Video-Finetuning-main/configs/my_config.yaml"

accelerate launch \
    --multi_gpu --num_machines 1 --num_processes=8  --gpu_ids '0,1,2,3,4,5,6,7' \
    train.py \
    --config="configs/my_config.yaml"

#accelerate launch \
#    --num_machines 1 --num_processes=2  --gpu_ids '6,7' \
#    projects/t2v_pretain/train_t2v_unpaired_hdvg.py \
#    --config="configs/train_t2v_hdvg.yaml"
