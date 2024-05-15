# Create a logger file in the same repo to keep track of the experiments executed 

# SCN model - Fixed split
python train.py \
    dataset=ZINC \
    model=simplicial/scn \
    model.backbone.n_layers=1,2,4 \
    model.feature_encoder.out_channels=16,64 \
    model.optimizer.lr=0.01,0.001 \
    dataset.parameters.batch_size=128 \
    dataset.parameters.data_seed=0,3 \
    dataset.transforms.graph2simplicial_lifting.complex_dim=3 \
    dataset.transforms.graph2simplicial_lifting.signed=False \
    trainer=default \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    callbacks.early_stopping.min_delta=0.005 \
    logger.wandb.project=topobenchmark_22Apr2024 \
    --multirun

# Batch size = 1
python train.py \
    dataset=cocitation_cora \
    model=simplicial/scn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64 \
    model.backbone.n_layers=1,2 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.transforms.graph2simplicial_lifting.complex_dim=3 \
    dataset.transforms.graph2simplicial_lifting.signed=False \
    trainer=default \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=topobenchmark_22Apr2024 \
    --multirun

python train.py \
    dataset=cocitation_citeseer \
    model=simplicial/scn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64 \
    model.backbone.n_layers=1,2 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.transforms.graph2simplicial_lifting.complex_dim=3 \
    dataset.transforms.graph2simplicial_lifting.signed=False \
    trainer=default \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=topobenchmark_22Apr2024 \
    --multirun

python train.py \
    dataset=cocitation_pubmed \
    model=simplicial/scn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64 \
    model.backbone.n_layers=1,2 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.transforms.graph2simplicial_lifting.complex_dim=3 \
    dataset.transforms.graph2simplicial_lifting.signed=False \
    trainer=default \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=topobenchmark_22Apr2024 \
    --multirun

# Vary batch size
python train.py \
    dataset=PROTEINS_TU \
    model=simplicial/scn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=16,64 \
    model.backbone.n_layers=1,2 \
    dataset.parameters.batch_size=32 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.transforms.graph2simplicial_lifting.complex_dim=3 \
    dataset.transforms.graph2simplicial_lifting.signed=False \
    trainer=default \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=topobenchmark_22Apr2024 \
    --multirun

python train.py \
    dataset=NCI1 \
    model=simplicial/scn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=16,64 \
    model.backbone.n_layers=1,2 \
    dataset.parameters.batch_size=32 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.transforms.graph2simplicial_lifting.complex_dim=3 \
    dataset.transforms.graph2simplicial_lifting.signed=False \
    trainer=default \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=topobenchmark_22Apr2024 \
    --multirun

python train.py \
    dataset=MUTAG \
    model=simplicial/scn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=16,64 \
    model.backbone.n_layers=1,2 \
    dataset.parameters.batch_size=32 \
    dataset.parameters.data_seed=0,3,5 \
    dataset.transforms.graph2simplicial_lifting.complex_dim=3 \
    dataset.transforms.graph2simplicial_lifting.signed=False \
    trainer=default \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    logger.wandb.project=topobenchmark_22Apr2024 \
    --multirun
