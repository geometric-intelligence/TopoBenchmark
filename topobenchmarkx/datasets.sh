datasets=( 'ZINC' ) #'NCI1' 'NCI109' 'IMDB-BINARY' 'IMDB-MULTI' )
for dataset in ${datasets[*]}
do
    python train.py \
        dataset=$dataset \
        dataset.transforms.graph2simplicial_lifting.signed=True \
        dataset.transforms.one_hot_node_degree_features.degrees_fields=x  &
done