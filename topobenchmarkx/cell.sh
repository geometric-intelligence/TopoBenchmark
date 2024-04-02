# CWN
# Batch size =1
python train.py dataset=cocitation_cora model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=cocitation_citeseer model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=cocitation_pubmed model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
# Vary batch size
python train.py dataset=PROTEINS_TU model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=NCI1 model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=IMDB-BINARY model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=IMDB-MULTI model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=MUTAG model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
# python train.py dataset=REDDIT-BINARY model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
# Fixed split
python train.py dataset=ZINC model=cell/cwn model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=18 --multirun
 
# CWN DCM
# Batch size =1
python train.py dataset=ZINC model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=18 --multirun
python train.py dataset=cocitation_cora model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=cocitation_citeseer model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=cocitation_pubmed model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
# Vary batch size
python train.py dataset=PROTEINS_TU model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=NCI1 model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=IMDB-BINARY model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=IMDB-MULTI model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
python train.py dataset=MUTAG model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.feature_encoder.proj_dropout=0.25,0.5 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
# python train.py dataset=REDDIT-BINARY model=cell/cwn_dcm model.optimizer.lr=0.01,0.001 model.feature_encoder.out_channels=16,32,64,128 model.backbone.n_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0,3,5 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 trainer=default logger.wandb.project=topobenchmark_0503 dataset.transforms.graph2cell_lifting.max_cell_length=6 --multirun
# Fixed split