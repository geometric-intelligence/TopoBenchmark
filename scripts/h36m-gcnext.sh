python -m topobenchmarkx \
    dataset=graph/H36MDataset \
    model=graph/gcnext \
    dataset.dataloader_params.batch_size=256 \
    trainer.max_epochs=10 \
    optimizer.parameters.lr=0.0006 \
    trainer.check_val_every_n_epoch=4 \
    model.backbone.config.motion_mlp.use_skeletal_hyperedges=False\
    test=True

# python -m topobenchmarkx \
#     dataset=graph/H36MDataset \
#     model=graph/gcnext \
#     dataset.dataloader_params.batch_size=256 \
#     trainer.max_epochs=10 \
#     optimizer.parameters.lr=0.0006 \
#     trainer.check_val_every_n_epoch=4 \
#     model.backbone.config.motion_mlp.use_skeletal_hyperedges=False\
#     test=True