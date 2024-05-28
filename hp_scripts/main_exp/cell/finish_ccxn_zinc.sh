seeds=( 23 150 )
for seed in ${seeds[*]}
do
python train.py \
  dataset=ZINC \
  seed=$seed \
  model=cell/ccxn \
  model.optimizer.lr=0.01,0.001 \
  model.optimizer.weight_decay=0 \
  model.feature_encoder.out_channels=32,64,128 \
  model.backbone.n_layers=2,4 \
  model.feature_encoder.proj_dropout=0.25,0.5 \
  dataset.parameters.batch_size=128,256 \
  dataset.transforms.one_hot_node_degree_features.degrees_fields=x \
  dataset.parameters.data_seed=0 \
  dataset.transforms.graph2cell_lifting.max_cell_length=10 \
  model.readout.readout_name="NoReadOut,PropagateSignalDown" \
  logger.wandb.project=TopoBenchmarkX_Cellular \
  trainer.max_epochs=500 \
  trainer.min_epochs=50 \
  callbacks.early_stopping.min_delta=0.005 \
  trainer.check_val_every_n_epoch=5 \
  callbacks.early_stopping.patience=10 \
  tags="[MainExperiment]" \
  --multirun &
done