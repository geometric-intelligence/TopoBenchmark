datasets=(
  "graph/cocitation_cora"
  "graph/cocitation_citeseer"
  "graph/US-county-demos"
  "graph/cocitation_pubmed"
  "graph/MUTAG"
  "graph/NCI1"
  "graph/PROTEINS"
  "graph/NCI109"
  "graph/ZINC"
  "graph/minesweeper"
  "graph/roman_empire"
  # "graph/IMDB-BINARY"
  # "graph/IMDB-MULTI"
  # "graph/REDDIT-BINARY"

)

for dataset in "${datasets[@]}"; do
  python topobenchmarkx/curvature_plot_rewire.py dataset=$dataset
done

