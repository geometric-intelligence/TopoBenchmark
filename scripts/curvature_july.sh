#!/bin/bash

# Define log files
LOG_FILE="scripts/script_output.log"
ERROR_LOG_FILE="scripts/script_error.log"
FAILED_LOG_FILE="scripts/failed_runs.log"

# Clear previous log files
> $LOG_FILE
> $ERROR_LOG_FILE
> $FAILED_LOG_FILE

# Function to run a command and check for failure
run_command() {
    local cmd="$1"
    
    # Run the command and capture the output and error
    { eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; } 2>> "$ERROR_LOG_FILE"
    
    # Check if the command failed
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Command failed: $cmd" >> "$FAILED_LOG_FILE"
        echo "Check $ERROR_LOG_FILE for details." >> "$FAILED_LOG_FILE"
    fi
}

# List of commands to execute
commands=(
# GCN 
# 'python -m topobenchmarkx model=graph/gcn transforms=no_transform dataset=graph/cocitation_citeseer optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
# 'python -m topobenchmarkx model=graph/gcn transforms=no_transform dataset=graph/cocitation_cora optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
# 'python -m topobenchmarkx model=graph/gcn transforms=no_transform dataset=graph/webkb_wisconsin optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
# 'python -m topobenchmarkx model=graph/gcn transforms=no_transform dataset=graph/cocitation_pubmed optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
# 'python -m topobenchmarkx model=graph/gcn transforms=no_transform dataset=graph/minesweeper optimizer.parameters.lr=0.01 model.feature_encoder.out_channels=64 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
# 'python -m topobenchmarkx model=graph/gcn transforms=no_transform dataset=graph/roman_empire optimizer.parameters.lr=0.01 model.feature_encoder.out_channels=64 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
# 'python -m topobenchmarkx model=graph/gcn transforms=no_transform dataset=graph/US-county-demos optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.loader.parameters.task_variable=Election dataset.loader.parameters.year=2012 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'

# GCN REWIRE

'python -m topobenchmarkx model=graph/gcn transforms=curvature_graph_rewire transforms.R.loops=1,3 transforms.R.lower_bound_eq="incorrect,correct" transforms.R.compute_every_it=true,false dataset=graph/cocitation_citeseer optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
'python -m topobenchmarkx model=graph/gcn transforms=curvature_graph_rewire transforms.R.loops=1,3 transforms.R.lower_bound_eq="incorrect,correct" transforms.R.compute_every_it=true,false dataset=graph/cocitation_cora optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
#'python -m topobenchmarkx model=graph/gcn transforms=curvature_graph_rewire transforms.R.loops=1,3 transforms.R.lower_bound_eq="incorrect,correct" transforms.R.compute_every_it=true,false dataset=graph/webkb_wisconsin optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
#'python -m topobenchmarkx model=graph/gcn transforms=curvature_graph_rewire transforms.R.loops=1,3 transforms.R.lower_bound_eq="incorrect,correct" transforms.R.compute_every_it=true,false dataset=graph/minesweeper optimizer.parameters.lr=0.01 model.feature_encoder.out_channels=64 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
#'python -m topobenchmarkx model=graph/gcn transforms=curvature_graph_rewire transforms.R.loops=1,3 transforms.R.lower_bound_eq="incorrect,correct" transforms.R.compute_every_it=true,false dataset=graph/roman_empire optimizer.parameters.lr=0.01 model.feature_encoder.out_channels=64 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
#'python -m topobenchmarkx model=graph/gcn transforms=curvature_graph_rewire transforms.R.loops=1,3 transforms.R.lower_bound_eq="incorrect,correct" transforms.R.compute_every_it=true,false dataset=graph/US-county-demos optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.loader.parameters.task_variable=Election dataset.loader.parameters.year=2012 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'
'python -m topobenchmarkx model=graph/gcn transforms=curvature_graph_rewire transforms.R.loops=1,3 transforms.R.lower_bound_eq="incorrect,correct" transforms.R.compute_every_it=true,false dataset=graph/cocitation_pubmed optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=128 model.backbone.num_layers=6,8,10 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=curvature_july_10th_noresconnection --multirun'

)

# Iterate over the commands and run them
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    run_command "$cmd"
done


