  #!/bin/bash                                                                                                                                                                                               
  # ==============================================================================                                                                                                                          
  # Bandit REINFORCE++ Experiments Runner
  # ==============================================================================
  set -e
                                                                                                                                                                                                            
  # Load environment
  CONDA_BASE="/mnt/project_modelware/zhaojian/miniconda3"                                                                                                                                                   
  CONDA_ENV="/mnt/project_modelware/zhaojian/envs/roll"
  export PATH="$CONDA_ENV/bin:$CONDA_BASE/bin:$PATH"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export DS_SKIP_CUDA_CHECK=1
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  export VLLM_USE_V1=0                                                                                                                                                                   
  source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate roll 2>/dev/null || true                                                                                                                                                                   
  echo " Python: $(which python) ($(python --version 2>&1))"                                                                                                                                                
  
  # =============================================================================                                                                                                                           
  # CONFIG NAME   
  # =============================================================================
  CONFIG_NAME="${1:-exp1_bandit_ts_dapo}"
                                                                                                                                                                                                            
  # Paths
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"                                                                                                                                                
  ROLL_DIR="/mnt/project_modelware/zhaojian/weiyu/bandit_reinforce/bandit_reinforce-/ROLL"
  BASE_OUTPUT="/mnt/project_modelware/zhaojian/weiyu/bandit_reinforce/output"                                                                                                                               
  
  cd "$SCRIPT_DIR"                                                                                                                                                                                          
                  
  if [ ! -f "${CONFIG_NAME}.yaml" ]; then
      echo "Error: Config file '${CONFIG_NAME}.yaml' not found!"
      ls -1 *.yaml 2>/dev/null | sed 's/.yaml$//'                                                                                                                                                           
      exit 1
  fi                                                                                                                                                                                                        
                  
  export PYTHONPATH="$ROLL_DIR:$PYTHONPATH"                                                                                                                                                                 
  
  # Auto-detect GPUs                                                                                                                                                                                        
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$NUM_GPUS" -gt 0 ]; then
      export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
  else                                                                                                                                                                                                      
      export CUDA_VISIBLE_DEVICES=0,1
  fi                                                                                                                                                                                                        
                  
  # Container workarounds                                                                                                                                                                                   
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export NCCL_P2P_DISABLE=1                                                                                                                                                                                 
  export NCCL_IB_DISABLE=1
  export NCCL_DEBUG=WARN
                                                                                                                                                                                                            
  # ============================================================
  # Unified output directory with timestamp                                                                                                                                                                 
  # ============================================================
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  export TRAINING_TIMESTAMP="$TIMESTAMP"                                                                                                                                                                    
  OUTPUT_DIR="$BASE_OUTPUT/${CONFIG_NAME}_${TIMESTAMP}"
                                                                                                                                                                                                            
  mkdir -p "$OUTPUT_DIR/logs"
  mkdir -p "$OUTPUT_DIR/models"
  mkdir -p "$OUTPUT_DIR/wandb"                                                                                                                                                                              
  
  # Point ROLL framework logs to same directory                                                                                                                                                             
  export ROLL_LOG_DIR="$OUTPUT_DIR/logs"
                                                                                                                                                                                                            
  export WANDB_MODE=offline
  export WANDB_DIR="$OUTPUT_DIR/wandb"                                                                                                                                                                      
                  
  # Redirect all stdout/stderr to unified log                                                                                                                                                               
  LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"
  exec > >(tee -a "$LOG_FILE")                                                                                                                                                                              
  exec 2>&1                                                                                                                                                                                                 
  
  echo "========================================================"                                                                                                                                           
  echo " Bandit REINFORCE++ Experiment"
  echo "========================================================"
  echo " Config:     ${CONFIG_NAME}"
  echo " Timestamp:  $TIMESTAMP"
  echo " Log file:   $LOG_FILE"
  echo " Output dir: $OUTPUT_DIR"                                                                                                                                                                           
  echo " ROLL dir:   $ROLL_DIR"
  echo "========================================================"                                                                                                                                           
                  
  if [ -n "$TMUX" ]; then
      echo " tmux:       $(tmux display-message -p '#S / #W')"
  fi                                                                                                                                                                                                        
  echo ""
                                                                                                                                                                                                            
  echo "========================================================"
  echo " GPU Status"
  echo "========================================================"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"
  echo ""                                                                                                                                                                                                   
  
  echo "========================================================"                                                                                                                                           
  echo " Pre-flight Checks"
  echo "========================================================"
  MODEL_PATH="/mnt/project_modelware/zhaojian/models/pretrain/Qwen3-8B-Base"                                                                                                                                
  DATA_PATH="$ROLL_DIR/data/dapo_math_17k.jsonl"
  ENCODER_PATH="/mnt/project_modelware/zhaojian/models/pretrain/Qwen3-Embedding-0.6B"                                                                                                                       
                                                                                                                                                                                                            
  [ -d "$MODEL_PATH" ]   && echo " [OK] Model:   $MODEL_PATH" || echo " [FAIL] Model not found: $MODEL_PATH"                                                                                                
  [ -f "$DATA_PATH" ]    && echo " [OK] Data:    $DATA_PATH ($(wc -l < "$DATA_PATH") rows)" || echo " [FAIL] Data not found: $DATA_PATH"                                                                    
  [ -d "$ENCODER_PATH" ] && echo " [OK] Encoder: $ENCODER_PATH" || echo " [WARN] Encoder not found: $ENCODER_PATH"                                                                                          
  echo ""         
                                                                                                                                                                                                            
  echo "========================================================"
  echo " Cleaning up Ray..."
  echo "========================================================"                                                                                                                                           
  ray stop --force 2>/dev/null || true
  pkill -9 -u $USER ray 2>/dev/null || true                                                                                                                                                                 
  rm -rf /tmp/ray/* 2>/dev/null || true
  sleep 2
  echo " Ray cleanup completed."                                                                                                                                                                            
  echo ""
                                                                                                                                                                                                            
  echo "========================================================"
  echo " Training started at $(date)"
  echo "========================================================"
  echo ""

  cd "$ROLL_DIR"

  REL_CONFIG_PATH=$(python -c "import os; print(os.path.relpath('$SCRIPT_DIR', '$ROLL_DIR/examples'))")                                                                                                     
  echo " Config path (relative): $REL_CONFIG_PATH"
  echo ""                                                                                                                                                                                                   
                  
  # ============================================================
  # Run with Hydra overrides to unify all output paths
  # ============================================================
  python examples/start_agentic_pipeline.py --config_path "$REL_CONFIG_PATH" --config_name "$CONFIG_NAME"                                                                                                                                               
  
  echo ""                                                                                                                                                                                                   
  echo "========================================================"
  echo " Training completed at $(date)"
  echo "========================================================"
  echo " Config:     ${CONFIG_NAME}"
  echo " Output dir: $OUTPUT_DIR"
  echo " Log file:   $LOG_FILE"
  echo " All logs:   $OUTPUT_DIR/logs/"                                                                                                                                                                     
  echo "========================================================"