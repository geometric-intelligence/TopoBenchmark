# SCCN (FINISH NCIs)
bash ../../../hp_scripts/main_exp/simplicial/tu/sccn_gpu_0.sh &
bash ../../../hp_scripts/main_exp/simplicial/tu/sccn_gpu_1.sh &

# SCN (Make NCIs)
bash ../../../hp_scripts/main_exp/simplicial/tu/scn_gpu_2.sh &
bash ../../../hp_scripts/main_exp/simplicial/tu/scn_gpu_3.sh 

# ------FIRTS WANDB CLEANING!------
bash ../../../hp_scripts/main_exp/simplicial/tu/sccn_imdb_gpu_0.sh &
bash ../../../hp_scripts/main_exp/simplicial/tu/sccn_imdb_gpu_1.sh &

bash ../../../hp_scripts/main_exp/simplicial/tu/sccnn_imdb_gpu_2.sh &
bash ../../../hp_scripts/main_exp/simplicial/tu/sccnn_imdb_gpu_3.sh 

bash ../../../hp_scripts/main_exp/simplicial/tu/scn_imdb_gpu_0.sh &
bash ../../../hp_scripts/main_exp/simplicial/tu/scn_imdb_gpu_1.sh 