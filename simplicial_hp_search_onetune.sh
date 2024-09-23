python -m topobenchmarkx \
    dataset=graph/NCI109 \
    model=simplicial/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_oneCell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/cocitation_cora \
#     model=simplicial/onetune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_oneCell \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[1\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/PROTEINS \
    model=simplicial/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_oneCell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/MUTAG \
    model=simplicial/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_oneCell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/cocitation_citeseer \
#     model=simplicial/onetune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_oneCell \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[4\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/amazon_ratings \
#     model=simplicial/onetune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_oneCell \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[5\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=simplicial/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_oneCell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/cocitation_pubmed \
#     model=simplicial/onetune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\]\],\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\]\],\[\[\[0,0\],up_laplacian\],\[\[1,0\],incidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[2,1\],incidence\],\[\[2,2\],down_laplacian\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_oneCell \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[7\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI109 \
    model=simplicial/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[2,1\],incidence\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_oneCell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/cocitation_cora \
#     model=simplicial/onetune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[2,1\],incidence\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_oneCell \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[1\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/PROTEINS \
    model=simplicial/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[2,1\],incidence\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_oneCell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/MUTAG \
    model=simplicial/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[2,1\],incidence\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_oneCell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

# python -m topobenchmarkx \
#     dataset=graph/cocitation_citeseer \
#     model=simplicial/onetune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[2,1\],incidence\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_oneCell \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[4\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/amazon_ratings \
#     model=simplicial/onetune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[2,1\],incidence\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_oneCell \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[5\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=simplicial/onetune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[2,1\],incidence\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_oneCell \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun 

# python -m topobenchmarkx \
#     dataset=graph/cocitation_pubmed \
#     model=simplicial/onetune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up_laplacian\],\[\[0,1\],coincidence\],\[\[1,1\],down_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,2\],coincidence\],\[\[2,2\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[1,1\],up_laplacian\],\[\[1,1\],down_laplacian\]\],\[\[\[0,0\],up_laplacian\],\[\[2,1\],incidence\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7,9 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_oneCell \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[7\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &
