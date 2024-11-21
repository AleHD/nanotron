#= PRELUDE: Command line utilities and handling the model size =#
SCRIPT_VERSION=v2

# Print usage function.
usage () {
	echo "Usage: llama.sh <size> [options...]"
	echo "<size>: 1/8"
	echo "Options:"
	echo " --help: Displays this message"

	echo " --rcp: Train on rcp (otherwise todi)"

	echo " --opt <adamW/ademamix/SFadamW>: Optimizer"
	echo " --lr <lr>: Learning rate"
	echo " --beta3 <float>: ademamix beta3"
	echo " --alpha <float>: ademamix alpha"
	echo " --ademamix-warmup <int/lr>: ademamix alpha and beta3 warmup"

	echo " --beta1 <float>: AdamW/ademamix beta1"
	echo " --beta2 <float>: AdamW/ademamix beta2"
	echo " --eps <float>: AdamW/ademamix eps"
	echo " --clip <float>: Gradient clip"

	echo " --extra-name <name>: Add a suffix to the name"
	echo " --wandbid <name>: Specify wandbid"
}

if [[ $# -eq 0 ]]; then
	echo "Invalid argument count: $#"
	usage
	exit 1
fi

SEQ_LEN=8192
SIZE=$1
shift

RCP=false
OPT=adamW
CHANGED_BETA1=false
BETA1=0.9

CHANGED_BETA2=false
BETA2=""
BETA2_adam=0.95
BETA2_ademamix=0.999
BETA2_sf=0.99

CHANGED_ALPHA=false
ALPHA=8

CHANGED_BETA3=false
BETA3=0.9999

CHANGED_EPS=false
EPS=0.00000001

CHANGED_CLIP=false
CLIP=1.0

CHANGED_LR=false

DECAY=""

ADEMAMIX_WARMUP=""

EXTRA_NAME=""
WANDB_ID=""

SUFFIX=""
while [[ $# -gt 0 ]]; do
	case $1 in
		--help)
			usage; exit 0;;

		--rcp)
			RCP=true; shift;;
		--opt)
			OPT=$2; shift 2;;
		--lr)
			LR=$2; CHANGED_LR=true; shift 2;;
		--decay)
			DECAY=$2; shift 2;;
		--ademamix-warmup)
			ADEMAMIX_WARMUP=$2; shift 2;;
		--alpha)
			ALPHA=$2; CHANGED_ALPHA=true; shift 2;;
		--beta1)
			BETA1=$2; CHANGED_BETA1=true; shift 2;;
		--beta2)
			BETA2=$2; CHANGED_BETA2=true; shift 2;;
		--beta3)
			BETA3=$2; CHANGED_BETA3=true; shift 2;;
		--eps)
			EPS=$2; CHANGED_EPS=true; shift 2;;
		--clip)
			CLIP=$2; CHANGED_CLIP=true; shift 2;;

		--extra-name)
			EXTRA_NAME="-$2"; shift 2;;
		--wandbid)
			WANDB_ID=$2; shift 2;;
		*)
			echo "Unexpected argument $1"
			usage
			exit 1
	esac
done

if [[ $RCP = true ]]; then
	GPUS=8
else
	GPUS=4
fi


if [[ $SIZE -eq 1 ]]; then 
	# batch_size = ~0.98 M
	# total_tokens = ~98.3B tokens
	# rcp time per iter (1 node): 3s
	# rcp ETA: 3d11h
	BASE_CONFIG=llama3_large_baseline
	DP=$GPUS
	MBS=3
	GBS=120
	SAVE_FREQ=1000  # every ~50m
	ITERS=100000
	LR=0.0006
elif [[ $SIZE -eq 8 ]]; then
	# batch_size = ~1.23M
	# total_tokens = ~24.58B
	# todi time per iter (1 node): 48.1s
	# todi ETA: 11d3h
	BASE_CONFIG=llama3_8b
	DP=$((GPUS/4))
	MBS=3
	GBS=150
	ITERS=20000
	SAVE_FREQ=500
    	LR=0.0003
else
	echo "Invalid llama size: $1"
	usage
	exit 1
fi
LR_WARMUP=$(( $ITERS/10 ))

# Get acc from GBS.
if (( $GBS % ($DP*$MBS) != 0 )); then
	echo "GBS $GBS not divisible by DP*MBS $DP*$MBS"
	exit 1
fi
ACC=$(( $GBS/$DP/$MBS ))



#= MIDDLE: Set up arguments depending on the commandline =#
SUFFIX=""
OPT_ARGS=()
PREFIX="nanotron.optimizer.optimizer_factory"
REMOVE_ADAM_ARGS=(" ~$PREFIX.adam_beta1" "~$PREFIX.adam_beta2" "~$PREFIX.adam_eps" "~$PREFIX.torch_adam_is_fused")
if [ $OPT = ademamix ]; then
	SUFFIX=$SUFFIX-ademamix
	BETA2=$([ "$BETA2" = "" ] && echo $BETA2_ademamix || echo $BETA2)
	if [ "$ADEMAMIX_WARMUP" = "" ]; then
		ADEMAMIX_WARMUP=$ITERS
	elif [ $ADEMAMIX_WARMUP = lr ]; then
		ADEMAMIX_WARMUP=$LR_WARMUP
		SUFFIX=$SUFFIX-ademawarmLR
	else
		SUFFIX=$SUFFIX-ademawarm$ADEMAMIX_WARMUP
	fi
	OPT_ARGS+=(
		"$PREFIX.name=ademamix"
		"+$PREFIX.adema_beta1=$BETA1"
		"+$PREFIX.adema_beta2=$BETA2"
		"+$PREFIX.adema_beta3=$BETA3"
		"+$PREFIX.adema_eps=$EPS"
		"+$PREFIX.adema_alpha=$ALPHA"
    		"+$PREFIX.adema_t_alpha_beta3=$ADEMAMIX_WARMUP"
	)
	OPT_ARGS+=${REMOVE_ADAM_ARGS[@]}
elif [ $OPT = adamW ]; then
	BETA2=$([ "$BETA2" = "" ] && echo $BETA2_adam || echo $BETA2)
	OPT_ARGS+=(
		"$PREFIX.name=adamW"
		"$PREFIX.adam_beta1=$BETA1"
		"$PREFIX.adam_beta2=$BETA2"
		"$PREFIX.adam_eps=$EPS"
	)
elif [ $OPT = SFadamW ]; then
	SUFFIX=$SUFFIX-sf
	BETA2=$([ "$BETA2" = "" ] && echo $BETA2_sf|| echo $BETA2)
	OPT_ARGS+=(
		"$PREFIX.name=SFadamW"
		"+$PREFIX.sf_beta1=$BETA1"
		"+$PREFIX.sf_beta2=$BETA2"
		"+$PREFIX.sf_eps=$EPS"
		"+$PREFIX.sf_r=0.0"
		"+$PREFIX.sf_weight_lr_power=2.0"
	)
	OPT_ARGS+=${REMOVE_ADAM_ARGS[@]}
else
	echo "Unknown optimizer $OPT"
	exit 1
fi
OPT_ARGS+=("nanotron.optimizer.clip_grad=$CLIP")

# Modify suffix depending on the overrides.
if [ "$DECAY" != "" ]; then
	SUFFIX=$SUFFIX-decay_$DECAY
	OPT_ARGS+=("nanotron.optimizer.weight_decay=$DECAY")
fi
if [ $CHANGED_LR = true ]; then
	SUFFIX=$SUFFIX-lr_$LR
fi
if [ $CHANGED_CLIP = true ]; then
	SUFFIX=$SUFFIX-clip_$CLIP
fi
if [ $CHANGED_BETA1 = true ]; then
	SUFFIX=$SUFFIX-beta1_$BETA1
fi
if [ $CHANGED_BETA2 = true ]; then
	SUFFIX=$SUFFIX-beta2_$BETA2
fi
if [ $CHANGED_BETA3 = true ] && [ $OPT = ademamix ]; then
	SUFFIX=$SUFFIX-beta3_$BETA3
fi
if [ $CHANGED_ALPHA = true ] && [ $OPT = ademamix ]; then
	SUFFIX=$SUFFIX-alpha_$ALPHA
fi
if [ $CHANGED_EPS = true ] && [ $OPT = ademamix ]; then
	SUFFIX=$SUFFIX-eps_$EPS
fi

SCHEDULER_ARGS=()
if [ $OPT = SFadamW ]; then
	SCHEDULER_ARGS+=(
		"nanotron.optimizer.learning_rate_scheduler.lr_decay_starting_step=null"
		"nanotron.optimizer.learning_rate_scheduler.lr_decay_steps=null"
		"nanotron.optimizer.learning_rate_scheduler.lr_decay_style=linear"
		"nanotron.optimizer.learning_rate_scheduler.min_decay_lr=$LR"
	)
else
	SCHEDULER_ARGS+=(
		"nanotron.optimizer.learning_rate_scheduler.lr_decay_starting_step=$(( 8*$ITERS/10 ))"
		"nanotron.optimizer.learning_rate_scheduler.lr_decay_steps=$(( 2*$ITERS/10 ))"
		"nanotron.optimizer.learning_rate_scheduler.lr_decay_style=linear"
	)
fi
SCHEDULER_ARGS+=(
	"nanotron.optimizer.learning_rate_scheduler.lr_warmup_steps=$LR_WARMUP"
	"nanotron.optimizer.learning_rate_scheduler.learning_rate=$LR"
)

SUFFIX=$SUFFIX$EXTRA_NAME
NAME=llama${SIZE}b$SUFFIX
WANDB_PROJECT=optimizer_experiments_$SCRIPT_VERSION

if [[ "$WANDB_ID" = "" ]]; then
	WANDB_ID=$NAME
fi

WANDB_ARGS=(
	"+run.env.WANDB_RUN_ID=$WANDB_ID"
	"+run.env.WANDB_RESUME=allow"
)
TOKEN_ARGS=()
SLURM_ARGS=()
if [[ $RCP = true ]]; then
	DATA_PATH=/mloscratch/homes/alhernan/data/tokenized/nanotron/llama_3_1/finewebedu-100B
	NANOTRON_PATH=/mloscratch/homes/alhernan/project/workspace/nanotron
	PRETRAIN_PATH=/mloscratch/homes/alhernan/project/workspace/pretrain
	LOGS_ROOT=/mloscratch/homes/alhernan/checkpoints/nanotron
else
	DATA_PATH=/store/swissai/a06/datasets_tokenized/nanotron/Meta-Llama-3-8B/fineweb-edu-sample-100BT
	NANOTRON_PATH=/store/swissai/a06/users/ahernnde/workspace/mpagliar-nanotron
	PRETRAIN_PATH=/users/ahernnde/project/repos/pretrain
	LOGS_ROOT=/store/swissai/a06/users/ahernnde/checkpoints/nanotron
	# no need to set api keys in rcp.
	TOKEN_ARGS+=(
		"run.env.WANDB_API_KEY=$(cat /store/swissai/a06/users/ahernnde/.keys/wandb.txt)"
		"run.env.HF_TOKEN=$(cat /store/swissai/a06/users/ahernnde/.keys/hf.txt)"
	)
	SLURM_ARGS+=(
		"run.slurm.reservation=null"
		"run.slurm.time=0:30:00"
		"run.slurm.partition=debug"
	)
fi

if [ $DP -ge 1 ]; then
	ZERO_STAGE=1
else
	ZERO_STAGE=0
fi

#= WRAPPING UP: Set up the _ARGS variables that are going to be used in the end =#
# Misc.
FINAL_ARGS=(
	"nanotron=$BASE_CONFIG"
	"nanotron.checkpoints.resume_checkpoint_path=checkpoints"
	"nanotron.parallelism.dp=$DP"
	"nanotron.tokens.micro_batch_size=$MBS"
	"nanotron.tokens.batch_accumulation_per_replica=$ACC"
	"nanotron.tokens.train_steps=$ITERS"
	"nanotron.tokens.sequence_length=$SEQ_LEN"
	"nanotron.model.model_config.max_position_embeddings=$SEQ_LEN"
	"+nanotron.model.model_config.use_gated_mlp=true"
	"nanotron.checkpoints.checkpoint_interval=$SAVE_FREQ"
	"nanotron.general.project=$WANDB_PROJECT"
	"nanotron.general.run=$NAME"
	"nanotron.optimizer.zero_stage=$ZERO_STAGE"
	"nanotron.data_stages.0.data.dataset.dataset_folder=$DATA_PATH"
	"nanotron.parallelism.tp_linear_async_communication=true"
	"nanotron.parallelism.tp_mode=REDUCE_SCATTER"
	"+nanotron.parallelism.tp_recompute_allgather=false"
	"run.paths.nanotron_logs=$LOGS_ROOT"
	"run.paths.nanotron_src=$NANOTRON_PATH"
	"run.paths.run_ident=''"
)

ARGS="${OPT_ARGS[@]} ${WANDB_ARGS[@]} ${FINAL_ARGS[@]} ${SCHEDULER_ARGS[@]} ${TOKEN_ARGS[@]} ${SLURM_ARGS[@]}"


#= RUNNING: Run the launcher =#
if [[ $RCP = true ]]; then
	RCP_ARGS=(
		"run=rcp"
		"+run.env.PYTHONPATH=/mloscratch/homes/alhernan/project/workspace/nanotron/src"
		"nanotron.tokenizer.tokenizer_name_or_path=meta-llama/Meta-Llama-3.1-8B"
	)
	CMD="python $PRETRAIN_PATH/launcher.py $ARGS ${RCP_ARGS[@]}"
	echo Running command: $CMD
	echo ---
	NANOTRON_LAUNCHER_DONT_SUBMIT=1 $CMD
else
	CMD="poetry run python $PRETRAIN_PATH/launcher.py $ARGS"
	echo Running command: $CMD
	echo ---
	$CMD
fi
