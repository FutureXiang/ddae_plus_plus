#!/bin/bash

# --- Default Settings ---
DEFAULT_GPUS=1 # when CUDA_VISIBLE_DEVICES is absent

# --- Function to find a free port ---
find_free_port() {
    python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null
}

# --- Argument Parsing ---
MODE=$1
NEEDS_CONFIG=true # Default assumption: mode needs a config file at $2

# --- Validate Mode ---
case "$MODE" in
    train)
        TARGET_SCRIPT="train.py"
        MODE_DEFAULT_ARGS="--use_bf16"
        ;;
    inference)
        TARGET_SCRIPT="inference.py"
        MODE_DEFAULT_ARGS=""
        ;;
    linear)
        TARGET_SCRIPT="linear.py"
        MODE_DEFAULT_ARGS="--use_bf16"
        ;;
    seg)
        TARGET_SCRIPT="seg_linear.py"
        MODE_DEFAULT_ARGS="--use_bf16"
        ;;
    extract)
        TARGET_SCRIPT="extract_features.py"
        MODE_DEFAULT_ARGS="--config tokenizer/configs/vavae_f16d32.yaml"
        NEEDS_CONFIG=false # Override: extract mode does not need config at $2
        ;;
    extract_seg)
        TARGET_SCRIPT="seg_extract_features.py"
        MODE_DEFAULT_ARGS="--config tokenizer/configs/vavae_f16d32.yaml"
        NEEDS_CONFIG=false # Override: extract mode does not need config at $2
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Must be 'train', 'inference', 'linear', or 'extract'."
        exit 1
        ;;
esac

# --- Process Remaining Arguments and Validate Config Path Conditionally ---
if [ "$NEEDS_CONFIG" = true ]; then
    # Modes requiring config path at $2
    if [ -z "$2" ]; then
        echo "Error: Config file path is required after mode '$MODE'."
        exit 1
    fi
    CONFIG_PATH=$2
    shift 2 # Remove MODE and CONFIG_PATH
    ADDITIONAL_ARGS="$@" # Store remaining arguments

    # --- Validate Config Path ---
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Config file not found at '$CONFIG_PATH'."
        exit 1
    fi

    CONFIG_ARG_TO_PASS="--config ${CONFIG_PATH}"
else
    # Mode is 'extract' (or others not needing config at $2)
    shift 1 # Only remove MODE
    ADDITIONAL_ARGS="$@" # Store all remaining arguments
    CONFIG_PATH="" # Explicitly clear CONFIG_PATH for this case

    CONFIG_ARG_TO_PASS=""
fi

# --- Determine Number of GPUs ---
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Warning: CUDA_VISIBLE_DEVICES is not set. Defaulting to $DEFAULT_GPUS GPU."
    NUM_GPUS=$DEFAULT_GPUS
else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    if [ "$NUM_GPUS" -eq 0 ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
      NUM_GPUS=1
    elif [ "$NUM_GPUS" -lt 1 ]; then
       echo "Warning: Could not determine GPU count from CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'. Defaulting to $DEFAULT_GPUS."
       NUM_GPUS=$DEFAULT_GPUS
    fi

fi

# --- Find Free Port ---
MASTER_PORT=$(find_free_port)
if [ -z "$MASTER_PORT" ]; then
    echo "Error: Failed to find an available port."
    exit 1
fi

# --- Execute torchrun ---
export OMP_NUM_THREADS=1

echo "======================================================"
echo "Starting run with mode:  ${MODE}"
echo "  GPUs (nproc_per_node): ${NUM_GPUS} (from CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<not set>}')"
echo "  Config File:           ${CONFIG_PATH}"
echo "  Master Port:           ${MASTER_PORT}"
echo "  Target Script:         ${TARGET_SCRIPT}"
echo "  Mode Default Args:     ${MODE_DEFAULT_ARGS:-<none>}"
echo "  Additional Args:       ${ADDITIONAL_ARGS:-<none>}"
echo "======================================================"


torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    ${TARGET_SCRIPT} \
    ${MODE_DEFAULT_ARGS} \
    ${CONFIG_ARG_TO_PASS} \
    ${ADDITIONAL_ARGS}
