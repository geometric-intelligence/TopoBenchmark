dataset='MUTAG'
project_name="TBX_SANN_$dataset"
CUDA=5
python -m topobenchmarkx\
    dataset=graph/$dataset\
    model=simplicial/sann\
    model.backbone.n_layers=1,2,3,4\
    model.feature_encoder.proj_dropout=0.25\
    model.feature_encoder.out_channels=64,128 \
    dataset.split_params.data_seed=0,1,2,3,4\
    dataset.dataloader_params.batch_size=128,256\
    +dataset.parameters.max_node_degree=10 \
    dataset.parameters.num_features=7 \
    trainer.max_epochs=500\
    trainer.min_epochs=50\
    trainer.devices=\[$CUDA\]\
    trainer.check_val_every_n_epoch=5\
    optimizer.scheduler=null\
    optimizer.parameters.lr=0.01,0.001\
    optimizer.parameters.weight_decay=0.0001\
    callbacks.early_stopping.patience=10\
    +transforms/data_manipulations@transforms.node_deg=node_degrees \
    +transforms/data_manipulations@transforms.node_feat=one_hot_node_degree_features \
    transforms.sann_encoding.max_hop=1,2,3\
    transforms.sann_encoding.complex_dim=3\
    logger.wandb.project=$project_name\
    --multirun 