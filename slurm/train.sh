# A script that will run Slurm using $JOB_NAME and $TRAIN_ARGS

export CMD="python train.py --config=configs/base_config.py --config.saving.name=$JOB_NAME $TRAIN_ARGS"
echo $CMD
mkdir results/slurm_logs -p
sbatch \
--job-name=$JOB_NAME \
--output=results/slurm_logs/$JOB_NAME.log \
--error=results/slurm_logs/$JOB_NAME.log \
$SLURM_ARGS \
slurm/callee.sh
