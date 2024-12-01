python -m topobenchmarkx \
    dataset=graph/H36MDataset \
    model=graph/gcnext \
    dataset.dataloader_params.batch_size=256 \
    trainer.max_epochs=25 \
    optimizer.parameters.lr=0.0006 \
    trainer.check_val_every_n_epoch=1 \
    test=True