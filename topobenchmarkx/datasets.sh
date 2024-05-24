python train.py \
    dataset=ZINC \
    dataset.transforms.graph2simplicial_lifting.signed=True \
    dataset.transforms.one_hot_node_degree_features.degrees_fields=x &

python train.py \
    dataset=MUTAG \
    dataset.transforms.graph2simplicial_lifting.signed=True &

datasets=( 'PROTEINS_TU' 'NCI1' 'NCI109' 'roman_empire' 'minesweeper')
for dataset in ${datasets[*]}
do
    python train.py \
        dataset=$dataset \
        dataset.transforms.graph2simplicial_lifting.signed=True &
done

            