
# Description: Main experiment script for GCN model.
# ----Node regression datasets: US County Demographics----
models=( 'simplicial/scn' 'cell/cwn' 'hypergraph/unignn2' )
for model in ${models[*]}
do


python dataset_statistics.py \
    dataset=us_country_demos \
    model=$model \
    
done
# # ----Cocitation datasets----
# datasets=( 'cocitation_cora' 'cocitation_citeseer' 'cocitation_pubmed' )

# for dataset in ${datasets[*]}
# do
#   python dataset_statistics.py \
#     dataset=$dataset \
#     model=$model
    
# done

# # ----Graph regression dataset----
# # Train on ZINC dataset
# python dataset_statistics.py \
#   dataset=ZINC \
#   model=$model \
#   dataset.transforms.one_hot_node_degree_features.degrees_fields=x
  

# # ----Heterophilic datasets----

# datasets=( roman_empire amazon_ratings tolokers questions minesweeper )

# for dataset in ${datasets[*]}
# do
#   python dataset_statistics.py \
#     dataset=$dataset \
#     model=$model
# done

# # ----TU graph datasets----
# # MUTAG have very few samples, so we use a smaller batch size
# # Train on MUTAG dataset
# python dataset_statistics.py \
#   dataset=MUTAG \
#   model=$model

# # Train rest of the TU graph datasets
# datasets=( 'PROTEINS_TU' 'NCI1' 'NCI109' 'REDDIT-BINARY' 'IMDB-BINARY' 'IMDB-MULTI') # 

# for dataset in ${datasets[*]}
# do
#   python dataset_statistics.py \
#     dataset=$dataset \
#     model=$model
# done

# done