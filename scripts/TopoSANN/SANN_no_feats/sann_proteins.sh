dataset='PROTEINS'
project_name="TBX_SANN_nofeat_$dataset"
CUDA=3

seeds=(0 1 2 4)
batch_size=128
for seed in ${seeds[*]}
do
python topobenchmarkx/run.py\
    dataset=graph/$dataset\
    model=simplicial/sann\
    model.backbone.n_layers=2,4\
    model.feature_encoder.proj_dropout=0.25\
    dataset.split_params.data_seed=$seed\
    dataset.dataloader_params.batch_size=$batch_size\
    model.feature_encoder.out_channels=64,128\
    transforms.sann_encoding.max_hop=1,2,3\
    transforms.sann_encoding.complex_dim=3\
    optimizer.parameters.weight_decay=0,0.0001\
    optimizer.parameters.lr=0.01,0.001\
    trainer.max_epochs=500\
    trainer.min_epochs=50\
    trainer.devices=\[$CUDA\]\
    optimizer.scheduler=null\
    trainer.check_val_every_n_epoch=5\
    callbacks.early_stopping.patience=10\
    logger.wandb.project=$project_name\
    transforms.sann_encoding.use_initial_features=False \
    --multirun &
done


CUDA=4
seeds=(0 1 2 4)
batch_size=256
for seed in ${seeds[*]}
do
python topobenchmarkx/run.py\
    dataset=graph/$dataset\
    model=simplicial/sann\
    model.backbone.n_layers=2,4\
    model.feature_encoder.proj_dropout=0.25\
    dataset.split_params.data_seed=$seed\
    dataset.dataloader_params.batch_size=$batch_size\
    model.feature_encoder.out_channels=64,128\
    transforms.sann_encoding.max_hop=1,2,3\
    transforms.sann_encoding.complex_dim=3\
    optimizer.parameters.weight_decay=0,0.0001\
    optimizer.parameters.lr=0.01,0.001\
    trainer.max_epochs=500\
    trainer.min_epochs=50\
    trainer.devices=\[$CUDA\]\
    optimizer.scheduler=null\
    trainer.check_val_every_n_epoch=5\
    callbacks.early_stopping.patience=10\
    logger.wandb.project=$project_name\
    --multirun &
done

# transforms.sann_encoding.neighborhoods='[incidence_1,incidence_0]','[incidence_0]','[incidence_1,0_incidence_1,incidence_0]'\