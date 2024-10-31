python -m topobenchmarkx \
    dataset=graph/NCI1 \
    model=cell/kan_topotune \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.devices=\[0\] \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/MUTAG \
    model=cell/kan_topotune \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.devices=\[1\] \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_cora \
    model=cell/kan_topotune \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.devices=\[2\] \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/ZINC \
    model=cell/kan_topotune \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.devices=\[3\] \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/PROTEINS \
    model=cell/kan_topotune \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.devices=\[4\] \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_citeseer \
    model=cell/kan_topotune \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.devices=\[5\] \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/cocitation_pubmed \
    model=cell/kan_topotune \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.devices=\[6\] \
    --multirun &

python -m topobenchmarkx \
    dataset=graph/NCI109 \
    model=cell/kan_topotune \
    trainer.max_epochs=2 \
    trainer.min_epochs=1 \
    trainer.devices=\[7\] \
    --multirun &