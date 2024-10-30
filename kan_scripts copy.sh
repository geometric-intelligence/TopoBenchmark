python -m topobenchmarkx \
    dataset=graph/MUTAG \
    model=cell/cccn \
    model.feature_encoder.out_channels=8,16,32 \
    optimizer.parameters.lr=0.001 \
    model.backbone.n_layers=1,2,4 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown,NoReadOut \
    logger.wandb.project=KAN_best \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" 
