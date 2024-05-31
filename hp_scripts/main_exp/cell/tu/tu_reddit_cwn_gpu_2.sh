# Train rest of the TU graph datasets
datasets=( 'REDDIT-BINARY' )

for dataset in ${datasets[*]}
  do
    python ../../../topobenchmarkx/train.py \
      dataset=$dataset \
      model=cell/cwn \
      model.optimizer.lr=0.001 \
      model.feature_encoder.out_channels=32,64 \
      model.backbone.n_layers=1,2,3 \
      model.feature_encoder.proj_dropout=0.5 \
      dataset.parameters.data_seed=0,3,5,7,9 \
      dataset.parameters.batch_size=16 \
      dataset.transforms.graph2cell_lifting.max_cell_length=10 \
      model.readout.readout_name="NoReadOut,PropagateSignalDown" \
      logger.wandb.project=TopoBenchmarkX_Cellular \
      trainer.max_epochs=500 \
      trainer.min_epochs=50 \
      trainer.devices=\[2\] \
      trainer.check_val_every_n_epoch=5 \
      callbacks.early_stopping.patience=10 \
      --multirun &
  done
