datasets=('ZINC')
neighborhoods=( '[incidence_1,incidence_0]' '[incidence_0]' '[incidence_1,0_incidence_1,incidence_0]')

for dataset in ${datasets[*]}
do
    for neighborhood in ${neighborhoods[*]}
    do
        python topobenchmarkx/run.py\
            dataset=graph/$dataset\
            model=simplicial/sann\
            model.backbone.n_layers=2,3,4\
            model.feature_encoder.proj_dropout=0,0.25\
            dataset.split_params.data_seed=0\
            dataset.dataloader_params.batch_size=32,64,128\
            trainer.max_epochs=500\
            trainer.min_epochs=50\
            trainer.devices=\[0\]\
            trainer.check_val_every_n_epoch=5\
            optimizer.scheduler=null\
            optimizer.parameters.lr=0.001\
            optimizer.parameters.weight_decay=0.0001\
            callbacks.early_stopping.patience=10\
            transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
            transforms.sann_encoding.pretrain_model=ZINC,GEOM,MOLPCBA,PCQM4MV2\
            logger.wandb.project=TBX_GPSE_ZINC\
            transforms.sann_encoding.neighborhoods=${neighborhood}\
            --multirun &
    done
done

datasets=('MUTAG')

for dataset in ${datasets[*]}
do
for neighborhood in ${neighborhoods[*]}
do
    python topobenchmarkx/run.py\
        dataset=graph/$dataset\
        model=simplicial/sann\
        model.backbone.n_layers=2,3,4\
        model.feature_encoder.proj_dropout=0,0.25\
        dataset.split_params.data_seed=0\
        dataset.dataloader_params.batch_size=32,64,128\
        trainer.max_epochs=500\
        trainer.min_epochs=50\
        trainer.devices=\[0\]\
        trainer.check_val_every_n_epoch=5\
        optimizer.scheduler=null\
        optimizer.parameters.lr=0.001\
        optimizer.parameters.weight_decay=0.0001\
        callbacks.early_stopping.patience=10\
        transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
        transforms.sann_encoding.pretrain_model=ZINC,GEOM,MOLPCBA,PCQM4MV2\
        logger.wandb.project=TBX_GPSE_MUTAG\
        transforms.sann_encoding.neighborhoods=${neighborhood}\
        --multirun &
    done
done


datasets=('NCI1')

for batch_size in ${batchsizes[*]}
do 
  for dataset in ${datasets[*]}
  do
    for neighborhood in ${neighborhoods[*]}
    do
        python topobenchmarkx/run.py\
            dataset=graph/$dataset\
            model=simplicial/sann\
            model.backbone.n_layers=2,3,4\
            model.feature_encoder.proj_dropout=0,0.25\
            dataset.split_params.data_seed=0\
            dataset.dataloader_params.batch_size=$batch_size\
            trainer.max_epochs=500\
            trainer.min_epochs=50\
            trainer.devices=\[0\]\
            trainer.check_val_every_n_epoch=5\
            optimizer.scheduler=null\
            optimizer.parameters.lr=0.001\
            optimizer.parameters.weight_decay=0.0001\
            callbacks.early_stopping.patience=10\
            transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
            transforms.sann_encoding.pretrain_model=ZINC,GEOM,MOLPCBA,PCQM4MV2\
            logger.wandb.project=TBX_GPSE_NCI1\
            transforms.sann_encoding.neighborhoods=${neighborhood}\
            --multirun &
        done
    done
done

datasets=('NCI109')

for batch_size in ${batchsizes[*]}
do 
  for dataset in ${datasets[*]}
  do
    for neighborhood in ${neighborhoods[*]}
    do
        python topobenchmarkx/run.py\
            dataset=graph/$dataset\
            model=simplicial/sann\
            model.backbone.n_layers=2,3,4\
            model.feature_encoder.proj_dropout=0,0.25\
            dataset.split_params.data_seed=0\
            dataset.dataloader_params.batch_size=$batch_size\
            trainer.max_epochs=500\
            trainer.min_epochs=50\
            trainer.devices=\[0\]\
            trainer.check_val_every_n_epoch=5\
            optimizer.scheduler=null\
            optimizer.parameters.lr=0.001\
            optimizer.parameters.weight_decay=0.0001\
            callbacks.early_stopping.patience=10\
            transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
            transforms.sann_encoding.pretrain_model=ZINC,GEOM,MOLPCBA,PCQM4MV2\
            logger.wandb.project=TBX_GPSE_NCI109\
            transforms.sann_encoding.neighborhoods=${neighborhood}\
            --multirun &
        done
    done
done