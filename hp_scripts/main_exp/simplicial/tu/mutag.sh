# ----TU graph datasets----
# MUTAG 
# TOTAL processes: 5
seeds=(0 3 5 7 9)

for seed in ${seeds[*]}
do
    python ../../../topobenchmarkx/train.py \
    dataset=MUTAG \
    model=simplicial/sccn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=$seed \
    dataset.parameters.batch_size=32,64 \
    model.readout.readout_name="NoReadOut,PropagateSignalDown" \
    dataset.transforms.graph2simplicial_lifting.signed=True \
    trainer=cpu \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    logger.wandb.project=TopoBenchmarkX_Simplicial \
    callbacks.early_stopping.patience=25 \
    tags="[MainExperiment]" \
    --multirun &
done