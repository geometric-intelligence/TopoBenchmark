python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/cccn \
    model.feature_encoder.out_channels=8,16,32 \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=1,2,4 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/cccn \
    model.feature_encoder.out_channels=64,128 \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=1,2,4 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[7\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=4 \
#     model.kan_params.spline_order=2 \
#     model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=2 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[1\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=4 \
#     model.kan_params.spline_order=3 \
#     model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=2 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[2\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=4 \
#     model.kan_params.spline_order=2 \
#     model.feature_encoder.encoder_name=AllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=2 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[3\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=4 \
#     model.kan_params.spline_order=3 \
#     model.feature_encoder.encoder_name=AllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=2 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[4\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_cccn \
    model.feature_encoder.out_channels=32 \
    model.kan_params.grid_size=4 \
    model.kan_params.spline_order=2 \
    model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
    model.backbone._target_=topobenchmarkx.nn.backbones.cell.CCCN \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=2 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[5\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_cccn \
    model.feature_encoder.out_channels=32 \
    model.kan_params.grid_size=4 \
    model.kan_params.spline_order=3 \
    model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
    model.backbone._target_=topobenchmarkx.nn.backbones.cell.CCCN \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=2 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=3 \
#     model.kan_params.spline_order=2 \
#     model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=2 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[1\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=3 \
#     model.kan_params.spline_order=3 \
#     model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=2 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[2\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=3 \
#     model.kan_params.spline_order=2 \
#     model.feature_encoder.encoder_name=AllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=2 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[3\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=3 \
#     model.kan_params.spline_order=3 \
#     model.feature_encoder.encoder_name=AllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=2 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[4\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_cccn \
    model.feature_encoder.out_channels=32 \
    model.kan_params.grid_size=3 \
    model.kan_params.spline_order=2 \
    model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
    model.backbone._target_=topobenchmarkx.nn.backbones.cell.CCCN \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=2 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[5\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_cccn \
    model.feature_encoder.out_channels=32 \
    model.kan_params.grid_size=3 \
    model.kan_params.spline_order=3 \
    model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
    model.backbone._target_=topobenchmarkx.nn.backbones.cell.CCCN \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=2 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=4 \
#     model.kan_params.spline_order=2 \
#     model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=4 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[1\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=4 \
#     model.kan_params.spline_order=3 \
#     model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=4 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[2\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=4 \
#     model.kan_params.spline_order=2 \
#     model.feature_encoder.encoder_name=AllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=4 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[3\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=4 \
#     model.kan_params.spline_order=3 \
#     model.feature_encoder.encoder_name=AllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=4 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[4\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_cccn \
    model.feature_encoder.out_channels=32 \
    model.kan_params.grid_size=4 \
    model.kan_params.spline_order=2 \
    model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
    model.backbone._target_=topobenchmarkx.nn.backbones.cell.CCCN \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=4 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[5\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_cccn \
    model.feature_encoder.out_channels=32 \
    model.kan_params.grid_size=4 \
    model.kan_params.spline_order=3 \
    model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
    model.backbone._target_=topobenchmarkx.nn.backbones.cell.CCCN \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=4 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=3 \
#     model.kan_params.spline_order=2 \
#     model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=4 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[1\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=3 \
#     model.kan_params.spline_order=3 \
#     model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=4 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[2\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=3 \
#     model.kan_params.spline_order=2 \
#     model.feature_encoder.encoder_name=AllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=4 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[3\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=cell/kan_cccn \
#     model.feature_encoder.out_channels=32 \
#     model.kan_params.grid_size=3 \
#     model.kan_params.spline_order=3 \
#     model.feature_encoder.encoder_name=AllCellFeatureEncoder \
#     optimizer.parameters.lr=0.001 \
#     model.backbone.n_layers=4 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown,NoReadOut \
#     logger.wandb.project=KAN \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[4\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_cccn \
    model.feature_encoder.out_channels=32 \
    model.kan_params.grid_size=3 \
    model.kan_params.spline_order=2 \
    model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
    model.backbone._target_=topobenchmarkx.nn.backbones.cell.CCCN \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=4 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[5\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_cccn \
    model.feature_encoder.out_channels=32 \
    model.kan_params.grid_size=3 \
    model.kan_params.spline_order=3 \
    model.feature_encoder.encoder_name=KANAllCellFeatureEncoder \
    model.backbone._target_=topobenchmarkx.nn.backbones.cell.CCCN \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=4 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &