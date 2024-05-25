
# ----Graph regression dataset----
# 
# SCCN
seeds=(42 3 5 23 150)
lrs=(0.001)
batch_sizes=(128)

for seed in ${seeds[*]}
do
    for lr in ${lrs[*]}
    do
        for batch_size in ${batch_sizes[*]}
        do
            python ../../../topobenchmarkx/train.py \
            dataset=ZINC \
            seed=$seed \
            model=simplicial/scn \
            model.optimizer.lr=$lr \
            model.optimizer.weight_decay=0 \
            model.feature_encoder.out_channels=32,64,128 \
            model.backbone.n_layers=2,4 \
            model.feature_encoder.proj_dropout=0.25,0.5 \
            dataset.parameters.batch_size=$batch_size \
            model.readout.readout_name="NoReadOut,PropagateSignalDown" \
            dataset.transforms.graph2simplicial_lifting.signed=True \
            dataset.transforms.one_hot_node_degree_features.degrees_fields=x \
            dataset.parameters.data_seed=0 \
            logger.wandb.project=TopoBenchmarkX_Simplicial \
            trainer.max_epochs=500 \
            trainer.min_epochs=50 \
            callbacks.early_stopping.min_delta=0.005 \
            trainer.check_val_every_n_epoch=5 \
            trainer.devices=\[2\] \
            callbacks.early_stopping.patience=10 \
            tags="[MainExperiment]" \
            --multirun &
        done
    done
done
