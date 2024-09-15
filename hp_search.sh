seeds=(1 3 5 7)

for seed in ${seeds[*]}
    do 
        python -m topobenchmarkx \
            dataset=graph/cocitation_cora \
            model=cell/tune \
            model.feature_encoder.out_channels=32 \
            model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
            model.backbone.GNN.num_layers=1,2 \
            model.backbone.routes=\[\[\[0,0\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[1,1\],up\]\],\[\[\[0,0\],up\],\[\[1,1\],up\],\[\[1,1\],down\]\],\[\[\[0,0\],up\],\[\[2,1\],down\]\] \
            model.backbone.layers=2,4,8 \
            model.feature_encoder.proj_dropout=0.3 \
            dataset.split_params.data_seed=$seed \
            dataset.dataloader_params.batch_size=1 \
            transforms.graph2cell_lifting.max_cell_length=18 \
            model.readout.readout_name=PropagateSignalDown \
            logger.wandb.project=TopoTune \
            trainer.max_epochs=1000 \
            trainer.min_epochs=50 \
            trainer.devices=\[0\] \
            trainer.check_val_every_n_epoch=1 \
            callbacks.early_stopping.patience=50 \
            tags="[FirstExperiments]" \
            --multirun
    done
