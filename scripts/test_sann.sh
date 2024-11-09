python -m topobenchmarkx \
    model=simplicial/sann \
    dataset=graph/PROTEINS \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0001 \
    model.backbone.n_layers=2 \
    model.feature_encoder.proj_dropout=0.0 \
    dataset.dataloader_params.batch_size=64 \
    dataset.split_params.data_seed=0 \
    dataset.split_params.split_type=k-fold \
    dataset.split_params.k=10 \
    optimizer.scheduler=null \
    trainer.max_epochs=50 \
    trainer.min_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=cpu \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=100\
    logger.wandb.project=TopoBenchmarkX_main\
    --multirun