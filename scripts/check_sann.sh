python -m topobenchmarkx \
    model=simplicial/sann \
    dataset=graph/NCI1 \
    optimizer.parameters.lr=0.01,0.001 \
    optimizer.parameters.weight_decay=0.0,0.0001 \
    model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.0 \
    dataset.dataloader_params.batch_size=64,128 \
    dataset.split_params.data_seed=0,1,2,3\
    dataset.split_params.split_type=k-fold \
    dataset.split_params.k=10 \
    optimizer.scheduler=null \
    trainer.max_epochs=200 \
    trainer.min_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=default \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=25\
    logger.wandb.project=SANN_CHECK_SWEEP\
    transforms.sann_encoding.max_hop=1,2,3\
    transforms.sann_encoding.complex_dim=3\
    model.feature_encoder.out_channels=32,64,128\
    --multirun