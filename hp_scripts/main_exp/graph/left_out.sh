### GCN
datasets=( roman_empire amazon_ratings tolokers questions minesweeper )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=graph/gcn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.num_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=7,9 \
    dataset.parameters.batch_size=128 \
    logger.wandb.project=TopoBenchmarkX_Graph \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    --multirun
done
