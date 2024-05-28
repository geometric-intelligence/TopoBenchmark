
# ----Heterophilic datasets----
# datasets=( roman_empire minesweeper )

datasets=( roman_empire )
lrs=(0.01 0.001)
drops=(0.25 0.5)
seeds=(0 3 5 7 9)

for seed in ${seeds[*]}
do 
    for lr in ${lrs[*]}
    do
        for dataset in ${datasets[*]}
        do
            for drop in ${drops[*]}
            do 
            python ../../../topobenchmarkx/train.py \
                dataset=$dataset \
                model=simplicial/scn \
                model.optimizer.lr=$lr \
                model.feature_encoder.out_channels=32,64,128 \
                model.backbone.n_layers=1,2,3,4 \
                model.feature_encoder.proj_dropout=$drop \
                dataset.parameters.data_seed=$seed \
                model.readout.readout_name="NoReadOut,PropagateSignalDown" \
                dataset.transforms.graph2simplicial_lifting.signed=True \
                dataset.parameters.batch_size=1 \
                logger.wandb.project=TopoBenchmarkX_Simplicial \
                trainer=cpu \
                trainer.max_epochs=1000 \
                trainer.min_epochs=50 \
                trainer.check_val_every_n_epoch=1 \
                callbacks.early_stopping.patience=50 \
                tags="[MainExperiment]" \
                --multirun &
            done
        done
    done
done



# Final totoal processes: 60