python train.py model=cell/cwn dataset=us_country_demos logger.wandb.project=topoxNormalizationExp tags="[Norm]" dataset.parameters.data_seed=0,3,5 --multirun 

python train.py model=cell/cwn model.backbone.n_layers=2 dataset=us_country_demos logger.wandb.project=topoxNormalizationExp tags="[Norm]" dataset.parameters.data_seed=0,3,5 --multirun

python train.py model=cell/cwn model.backbone.n_layers=3 dataset=us_country_demos logger.wandb.project=topoxNormalizationExp tags="[Norm]" dataset.parameters.data_seed=0,3,5 --multirun


python train.py model=cell/cwn dataset=cocitation_cora logger.wandb.project=topoxNormalizationExp tags="[Norm]" dataset.parameters.data_seed=0,3,5 --multirun

python train.py model=cell/cwn model.backbone.n_layers=2 dataset=cocitation_cora logger.wandb.project=topoxNormalizationExp tags="[Norm]" dataset.parameters.data_seed=0,3,5 --multirun

python train.py model=cell/cwn model.backbone.n_layers=3 dataset=cocitation_cora logger.wandb.project=topoxNormalizationExp tags="[Norm]" dataset.parameters.data_seed=0,3,5 --multirun
