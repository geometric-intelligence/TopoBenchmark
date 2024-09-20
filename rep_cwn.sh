# manyHasse reproducing CWN
python -m topobenchmarkx \
    dataset=graph/cocitation_pubmed,graph/amazon_ratings,graph/NCI1 \
    model=cell/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=IdentityGIN \
    model.backbone.GNN.num_layers=1 \
    model.backbone.routes=\[\[\[1,1\],adjacency\],\[\[0,1\],cbdry\],\[\[2,1\],bdry\]\] \
    model.backbone.layers=4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    transforms.graph2cell_lifting.max_cell_length=18 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_reproduceCWN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]"  \
     --multirun &

     python -m topobenchmarkx \
    dataset=graph/NCI109,graph/cocitation_cora,graph/PROTEINS,graph/MUTAG,graph/cocitation_citeseer \
    model=cell/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=IdentityGIN \
    model.backbone.GNN.num_layers=1 \
    model.backbone.routes=\[\[\[1,1\],adjacency\],\[\[0,1\],cbdry\],\[\[2,1\],bdry\]\] \
    model.backbone.layers=4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    transforms.graph2cell_lifting.max_cell_length=18 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_reproduceCWN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]"  \
    --multirun &

# oneHasse reproducing CWN
python -m topobenchmarkx \
    dataset=graph/cocitation_pubmed,graph/amazon_ratings,graph/NCI1 \
    model=cell/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=IdentityGIN \
    model.backbone.GNN.num_layers=1 \
    model.backbone.routes=\[\[\[1,1\],adjacency\],\[\[0,1\],cbdry\],\[\[2,1\],bdry\]\] \
    model.backbone.layers=4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    transforms.graph2cell_lifting.max_cell_length=18 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_reproduceCWN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]"  \
     --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI109,graph/cocitation_cora,graph/PROTEINS,graph/MUTAG,graph/cocitation_citeseer \
    model=cell/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=IdentityGIN \
    model.backbone.GNN.num_layers=1 \
    model.backbone.routes=\[\[\[1,1\],adjacency\],\[\[0,1\],cbdry\],\[\[2,1\],bdry\]\] \
    model.backbone.layers=4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    transforms.graph2cell_lifting.max_cell_length=18 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_reproduceCWN \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]"  \
    --multirun 