python -m topobenchmarkx \
    dataset=graph/NCI109 \
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    dataset=graph/ZINC \
    model=simplicial/topotune, simplicial/topotune_onehasse \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[1,2\],cob\],\[\[2,1\],boundary\],\[\[2,2\],down\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],up\],\[\[2,1\],boundary\]\],\[\[\[0,0\],up\],\[\[0,1\],cob\],\[\[1,1\],up\],\[\[1,2\],cob\]\],\[\[\[0,0\],up\],\[\[1,0\],boundary\],\[\[1,1\],down\],\[\[1,1\],up\],\[\[2,1\],boundary\],\[\[2,2\],down\]\] \
    model.backbone.layers=2,4,8 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1,3,5,7,9 \
    model.readout.readout_name=PropagateSignalDown \
    transforms.one_hot_node_degree_features.degrees_fields=x \
    logger.wandb.project=TopoTune_Simplicial \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[FirstExperiments]" \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_cora \
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    model=simplicial/topotune, simplicial/topotune_onehasse \
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
    --multirun