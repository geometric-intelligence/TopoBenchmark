datasets=( tolokers )

seeds=(0 3 5 7 9)

for seed in ${seeds[*]}
do 
  for dataset in ${datasets[*]}
  do
  python train.py \
    dataset=$dataset \
    model=cell/ccxn \
    model.optimizer.lr=0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=$seed \
    dataset.parameters.batch_size=1 \
    dataset.transforms.graph2cell_lifting.max_cell_length=10 \
    model.readout.readout_name="NoReadOut,PropagateSignalDown" \
    logger.wandb.project=TopoBenchmarkX_Cellular \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer=cpu \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[MainExperiment]" \
    --multirun &
done
