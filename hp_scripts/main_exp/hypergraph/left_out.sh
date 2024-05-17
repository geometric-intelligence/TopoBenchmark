# ----Heterophilic datasets----

datasets=( questions )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=hypergraph/unignn2 \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.parameters.batch_size=128,256 \
    logger.wandb.project=TopoBenchmarkX_Hypergraph \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[MainExperiment]" \
    --multirun
done

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=hypergraph/edgnn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.All_num_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.parameters.batch_size=128,256 \
    logger.wandb.project=TopoBenchmarkX_Hypergraph \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[MainExperiment]" \
    --multirun
done

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=hypergraph/allsettransformer \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.parameters.batch_size=128,256 \
    logger.wandb.project=TopoBenchmarkX_Hypergraph \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[MainExperiment]" \
    --multirun
done