python -m topobenchmarkx \
    dataset=graph/H36MDataset \
    model=graph/gcnext \
    dataset.dataloader_params.batch_size=256 \
    trainer.max_epochs=50 \
    optimizer.parameters.lr=0.0006 \