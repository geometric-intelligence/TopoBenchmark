python -m topobenchmarkx \
    dataset=graph/NCI109 \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_cora \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/PROTEINS \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/MUTAG \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_citeseer \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[4\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/amazon_ratings \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[5\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_pubmed \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[7\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI109 \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],boundary\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_cora \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],boundary\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/PROTEINS \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],boundary\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/MUTAG \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],boundary\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_citeseer \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],boundary\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[4\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/amazon_ratings \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],boundary\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[5\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],boundary\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[6\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_pubmed \
    model=simplicial/tune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],boundary\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[7\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &


# python -m topobenchmarkx \
#     dataset=graph/NCI109 \
#     model=simplicial/tune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_Simplicial \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[0\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/cocitation_cora \
#     model=simplicial/tune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_Simplicial \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[1\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/PROTEINS \
#     model=simplicial/tune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_Simplicial \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[2\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/MUTAG \
#     model=simplicial/tune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_Simplicial \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[3\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/cocitation_citeseer \
#     model=simplicial/tune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_Simplicial \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[4\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/amazon_ratings \
#     model=simplicial/tune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_Simplicial \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[5\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/NCI1 \
#     model=simplicial/tune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_Simplicial \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[6\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# python -m topobenchmarkx \
#     dataset=graph/cocitation_pubmed \
#     model=simplicial/tune \
#     model.feature_encoder.out_channels=32 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
#     model.backbone.layers=2,4,8 \
#     model.feature_encoder.proj_dropout=0.3 \
#     dataset.split_params.data_seed=1,3,5,7 \
#     model.readout.readout_name=PropagateSignalDown \
#     logger.wandb.project=TopoTune_Simplicial \
#     trainer.max_epochs=1000 \
#     trainer.min_epochs=50 \
#     trainer.devices=\[7\] \
#     trainer.check_val_every_n_epoch=1 \
#     callbacks.early_stopping.patience=50 \
#     tags="[FirstExperiments]" \
#     --multirun &

# # python -m topobenchmarkx \
# #     dataset=graph/NCI109,graph/cocitation_cora,graph/PROTEINS,graph/MUTAG,graph/cocitation_citeseer \
# #     model=cell/tune \
# #     model.feature_encoder.out_channels=32 \
# #     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
# #     model.backbone.GNN.num_layers=1,2 \
# #     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\] \
# #     model.backbone.layers=2,4,8 \
# #     model.feature_encoder.proj_dropout=0.3 \
# #     dataset.split_params.data_seed=1,3,5,7 \
# #     transforms.graph2cell_lifting.max_cell_length=18 \
# #     model.readout.readout_name=PropagateSignalDown \
# #     logger.wandb.project=TopoTune \
# #     trainer.max_epochs=1000 \
# #     trainer.min_epochs=50 \
# #     trainer.devices=\[5\] \
# #     trainer.check_val_every_n_epoch=1 \
# #     callbacks.early_stopping.patience=50 \
# #     tags="[FirstExperiments]" \
# #     --multirun &

# # python -m topobenchmarkx \
# #     dataset=graph/NCI109,graph/cocitation_cora,graph/PROTEINS,graph/MUTAG,graph/cocitation_citeseer \
# #     model=cell/tune \
# #     model.feature_encoder.out_channels=32 \
# #     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
# #     model.backbone.GNN.num_layers=1,2 \
# #     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,2\],down\]\] \
# #     model.backbone.layers=2,4,8 \
# #     model.feature_encoder.proj_dropout=0.3 \
# #     dataset.split_params.data_seed=1,3,5,7 \
# #     transforms.graph2cell_lifting.max_cell_length=18 \
# #     model.readout.readout_name=PropagateSignalDown \
# #     logger.wandb.project=TopoTune \
# #     trainer.max_epochs=1000 \
# #     trainer.min_epochs=50 \
# #     trainer.devices=\[6\] \
# #     trainer.check_val_every_n_epoch=1 \
# #     callbacks.early_stopping.patience=50 \
# #     tags="[FirstExperiments]" \
# #     --multirun &

# # python -m topobenchmarkx \
# #     dataset=graph/NCI109,graph/roman_empire,graph/REDDIT-BINARY,graph/minesweeper \
# #     model=cell/tune \
# #     model.feature_encoder.out_channels=32 \
# #     model.tune_gnn=GIN \
# #     model.backbone.GNN.num_layers=1,2 \
# #     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],down\]\] \
# #     model.backbone.layers=2,4,8 \
# #     model.feature_encoder.proj_dropout=0.3 \
# #     dataset.split_params.data_seed=1,3,5,7 \
# #     transforms.graph2cell_lifting.max_cell_length=18 \
# #     model.readout.readout_name=PropagateSignalDown \
# #     logger.wandb.project=TopoTune \
# #     trainer.max_epochs=1000 \
# #     trainer.min_epochs=50 \
# #     trainer.devices=\[1\] \
# #     trainer.check_val_every_n_epoch=1 \
# #     callbacks.early_stopping.patience=50 \
# #     tags="[FirstExperiments]" \
# #     --multirun &

# # python -m topobenchmarkx \
# #     dataset=graph/NCI109,graph/roman_empire,graph/REDDIT-BINARY,graph/minesweeper \
# #     model=cell/tune \
# #     model.feature_encoder.out_channels=32 \
# #     model.tune_gnn=GAT \
# #     +model.backbone.GNN.heads=1,2 \
# #     +model.backbone.GNN.v2=True,False \
# #     model.backbone.GNN.num_layers=1,2 \
# #     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],down\]\] \
# #     model.backbone.layers=2,4,8 \
# #     model.feature_encoder.proj_dropout=0.3 \
# #     dataset.split_params.data_seed=1,3,5,7 \
# #     transforms.graph2cell_lifting.max_cell_length=18 \
# #     model.readout.readout_name=PropagateSignalDown \
# #     logger.wandb.project=TopoTune \
# #     trainer.max_epochs=1000 \
# #     trainer.min_epochs=50 \
# #     trainer.devices=\[2\] \
# #     trainer.check_val_every_n_epoch=1 \
# #     callbacks.early_stopping.patience=50 \
# #     tags="[FirstExperiments]" \
# #     --multirun &

# # python -m topobenchmarkx \
# #     dataset=graph/NCI109,graph/roman_empire,graph/REDDIT-BINARY,graph/minesweeper \
# #     model=cell/tune \
# #     model.feature_encoder.out_channels=32 \
# #     model.tune_gnn=GraphSAGE \
# #     model.backbone.GNN.num_layers=1,2 \
# #     model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],down\]\] \
# #     model.backbone.layers=2,4,8 \
# #     model.feature_encoder.proj_dropout=0.3 \
# #     dataset.split_params.data_seed=1,3,5,7 \
# #     transforms.graph2cell_lifting.max_cell_length=18 \
# #     model.readout.readout_name=PropagateSignalDown \
# #     logger.wandb.project=TopoTune \
# #     trainer.max_epochs=1000 \
# #     trainer.min_epochs=50 \
# #     trainer.devices=\[3\] \
# #     trainer.check_val_every_n_epoch=1 \
# #     callbacks.early_stopping.patience=50 \
# #     tags="[FirstExperiments]" \
# #     --multirun &

# # # seeds=(1 3 5 7)

# # # for seed in ${seeds[*]}
# # #     do 
# # #         python -m topobenchmarkx \
# # #             dataset=graph/cocitation_cora \
# # #             model=cell/tune \
# # #             model.feature_encoder.out_channels=32 \
# # #             model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
# # #             model.backbone.GNN.num_layers=1,2 \
# # #             model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],down\]\] \
# # #             model.backbone.layers=2,4,8 \
# # #             model.feature_encoder.proj_dropout=0.3 \
# # #             dataset.split_params.data_seed=$seed \
# # #             dataset.dataloader_params.batch_size=1 \
# # #             transforms.graph2cell_lifting.max_cell_length=18 \
# # #             model.readout.readout_name=PropagateSignalDown \
# # #             logger.wandb.project=TopoTune \
# # #             trainer.max_epochs=1000 \
# # #             trainer.min_epochs=50 \
# # #             trainer.devices=\[0\] \
# # #             trainer.check_val_every_n_epoch=1 \
# # #             callbacks.early_stopping.patience=50 \
# # #             tags="[FirstExperiments]" \
# # #             --multirun
# # #     done
