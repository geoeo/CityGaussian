get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

TEST_PATH="city_gaussian_2_view/test"

COARSE_CONFIG="city_gaussian_2_view_coarse"
CONFIG="city_gaussian_2_view_r4"

out_name="val"  # i.e. TEST_PATH.split('/')[-1]
max_block_id=8  # i.e. x_dim * y_dim * z_dim - 1
port=4041

# train coarse global gaussian model
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
python train_large.py --config config/$COARSE_CONFIG.yaml

# train CityGaussian
# obtain data partitioning
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
python data_partition.py --config config/$CONFIG.yaml

# optimize each block, please adjust block number according to config
for num in $(seq 0 $max_block_id); do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Starting training block '$num'"
            WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $num --port $port &
            # Increment the port number for the next run
            ((port++))
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 120
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 120
        fi
    done
done
wait

# merge the blocks
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
python merge.py --config config/$CONFIG.yaml

# rendering and evaluation, add --load_vq in rendering if you want to load compressed model
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
python metrics_large.py -m output/$CONFIG -t $out_name