#!/bin/bash

DATASET_list=("cifar10")  # "cifar10" "ffhq" "imagenet64" "lsun_bedroom" "ms_coco"

GPU_INDEX=0
get_pas_list=("yes")  # "yes" or "no"  Explanation: Whether to use the PAS method.
cal_fid="yes"  # "yes" "no"  Explanation: Whether to perform the FID computation

order_list=(3)  # 4 3 2  Explanation: Order for ipndm (recommended as 3) or deis (recommended as 4)
tea_NFE_list=(100)  # Explanation: Approximate NFE to obtain the teacher trajectory
tea_seeds_list=(0-4999)   # 0-9999  0-4999  Explanation: Number of teacher/ground truth trajectories
METHOD_list=(euler)     # euler ipndm deis  Explanation: Baseline solver used
METHOD_TEA_list=(heun)   # heun euler dpm  Explanation: Teacher solver used
PCA_num_list=(4)  # 1 2 3 4  Explanation: Number of PCA components used
use_xT_list=("yes")  # "yes" "no"  Explanation: Whether to use xT to span the trajectory space.
first_time_tau_list=(0.0001)  # 0.0001 0.01  Explanation: Tolerance for the first correction point
others_time_tau_list=(0.0001)  #  Explanation: Tolerance for the other correction points
schedule_type_list=("polynomial")  # "polynomial"  Explanation: Type of schedule used
loss_type_list=("L1")  # "L2" "L1" "LPIPS" "Pseudo_Huber"  Explanation: Type of loss used
learning_rate_list=(0.01)  # 10 1 0.1 0.01 0.001 0.0001  Explanation: Learning rate used
NFE_list=(5 6 8 10)  # 5 6 8 10  Explanation: Number of function evaluations (NFE) used

epochs=1000  # Explanation: Epochs for coordinate search via gradient descent
early_stop=10  # Explanation: Early stopping

for DATASET in "${DATASET_list[@]}"; do
for order in "${order_list[@]}"; do
for METHOD in "${METHOD_list[@]}"; do
for METHOD_TEA in "${METHOD_TEA_list[@]}"; do
for PCA_num in "${PCA_num_list[@]}"; do
for tea_NFE in "${tea_NFE_list[@]}"; do
for use_xT in "${use_xT_list[@]}"; do
for tea_seeds in "${tea_seeds_list[@]}"; do
for first_time_tau in "${first_time_tau_list[@]}"; do
for others_time_tau in "${others_time_tau_list[@]}"; do
for get_pas in "${get_pas_list[@]}"; do
for schedule_type in "${schedule_type_list[@]}"; do
for loss_type in "${loss_type_list[@]}"; do
for learning_rate in "${learning_rate_list[@]}"; do
for NFE in "${NFE_list[@]}"; do


if [ "$METHOD" = "euler" ] || [ "$METHOD" = "ipndm" ]; then
    steps=$((NFE + 1))
fi

if [ "$DATASET" = "ms_coco" ]; then
    GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
    seeds=0-9999
else
    GUIDANCE_FLAGS=""
    seeds=0-49999
fi

if [ "$DATASET" = "lsun_bedroom" ]; then
    BATCH_SIZE=16
elif [ "$DATASET" = "ms_coco" ]; then
    BATCH_SIZE=6
else
    BATCH_SIZE=64
fi


para_name="$DATASET-$METHOD-$METHOD_TEA-$NFE-$use_xT-$get_pas-$first_time_tau-$others_time_tau-$tea_NFE-$tea_seeds-$PCA_num-$schedule_type-$learning_rate-$loss_type-$order"
out_dir=out/$DATASET/$para_name
tea_traj_dir="out/traj/$DATASET-$NFE-$METHOD_TEA-$tea_NFE-$tea_seeds/"

echo -e "\nDATASET-METHOD-METHOD_TEA-NFE-use_xT-get_pas-first_time_tau-others_time_tau-tea_NFE-tea_seeds_len-PCA_num-schedule_type-learning_rate-loss_type-order"
echo -e "para_name: $para_name\n"

    if [ "$get_pas" = "yes" ]; then
        SOLVER_FLAGS="--tea_solver=$METHOD_TEA --num_steps=$steps"
        SCHEDULE_FLAGS="--schedule_type=$schedule_type --schedule_rho=7"
        CUDA_VISIBLE_DEVICES=$GPU_INDEX torchrun --standalone --nproc_per_node=1\
        get_tea_traj.py --dataset_name=$DATASET --batch=$BATCH_SIZE --seeds=$tea_seeds --tea_NFE=$tea_NFE\
        --plug_solver=$METHOD --tea_traj_dir=$tea_traj_dir\
        $SOLVER_FLAGS $SCHEDULE_FLAGS $GUIDANCE_FLAGS
    
        SOLVER_FLAGS="--num_steps=$steps"
        ADDITIONAL_FLAGS="--max_order=$order" 
        CUDA_VISIBLE_DEVICES=$GPU_INDEX torchrun --standalone --nproc_per_node=1\
        search_by_pca.py --dataset_name=$DATASET --use_xT=$use_xT  --loss_type=$loss_type --plug_solver=$METHOD\
        --epochs=$epochs --early_stop=$early_stop --learning_rate=$learning_rate --PCA_num=$PCA_num\
        --first_time_tau=$first_time_tau  --others_time_tau=$others_time_tau  --para_name=$para_name --tea_traj_dir=$tea_traj_dir\
        $SOLVER_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS
    fi

    if [ "$cal_fid" = "yes" ]; then
        SOLVER_FLAGS="--solver=$METHOD --num_steps=$steps"
        SCHEDULE_FLAGS="--schedule_type=$schedule_type --schedule_rho=7"
        ADDITIONAL_FLAGS="--max_order=$order" 
        CUDA_VISIBLE_DEVICES=$GPU_INDEX torchrun --standalone --nproc_per_node=1\
        sample_by_pca.py --dataset_name=$DATASET --outdir=$out_dir  --seeds=$seeds  --batch=$BATCH_SIZE\
        --PCA_num=$PCA_num --get_pas=$get_pas --use_xT=$use_xT --para_name=$para_name\
        $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS

        if [ "$DATASET" = "cifar10" ]; then
            ref_path="./stats/cifar10-32x32.npz"
        elif [ "$DATASET" = "ffhq" ]; then
            ref_path="./stats/ffhq-64x64.npz"
        elif [ "$DATASET" = "imagenet64" ]; then
            ref_path="./stats/imagenet-64x64.npz"
        elif [ "$DATASET" = "lsun_bedroom" ]; then
            ref_path="./stats/lsun_bedroom-256x256.npz"
        elif [ "$DATASET" = "ms_coco" ]; then
            ref_path="./stats/ms_coco-512x512.npz"
        fi

        CUDA_VISIBLE_DEVICES=$GPU_INDEX torchrun --standalone --nproc_per_node=1\
        fid.py calc --images=$out_dir --ref=$ref_path
    fi

done
done
done
done
done
done
done
done
done
done
done
done
done
done
done