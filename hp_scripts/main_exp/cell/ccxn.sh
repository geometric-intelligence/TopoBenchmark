# Train rest of the TU graph datasets
datasets=( 'IMDB-BINARY' )
batchsizes=(128)

for batch_size in ${batchsizes[*]}
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
      dataset.parameters.data_seed=0,3,5,7,9 \
      dataset.parameters.batch_size=$batch_size \
      dataset.transforms.graph2cell_lifting.max_cell_length=10 \
      model.readout.readout_name="NoReadOut,PropagateSignalDown" \
      logger.wandb.project=TopoBenchmarkX_Cellular \
      trainer.max_epochs=500 \
      trainer.min_epochs=50 \
      trainer.devices=\[0\] \
      trainer.check_val_every_n_epoch=5 \
      callbacks.early_stopping.patience=10 \
      --multirun &
  done
done


# Train rest of the TU graph datasets
datasets=( 'IMDB-BINARY' )
batchsizes=(256)

for batch_size in ${batchsizes[*]}
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
      dataset.parameters.data_seed=0,3,5,7,9 \
      dataset.parameters.batch_size=$batch_size \
      dataset.transforms.graph2cell_lifting.max_cell_length=10 \
      model.readout.readout_name="NoReadOut,PropagateSignalDown" \
      logger.wandb.project=TopoBenchmarkX_Cellular \
      trainer.max_epochs=500 \
      trainer.min_epochs=50 \
      trainer.devices=\[0\] \
      trainer.check_val_every_n_epoch=5 \
      callbacks.early_stopping.patience=10 \
      --multirun 
  done
done