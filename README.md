# UniKT

## Installation
Use the following command to install pyKT: Create conda envirment.
```
conda create -n pykt python=3.6
source activate pykt
```

```
cd UniKT
pip install -e .
```

## Download Datasets & Preprocess

### Download
You can download datasets from {https://pykt-toolkit.readthedocs.io/en/latest/datasets.html}

### Preprocess
```
cd examples
python data_preprocess.py --dataset_name=algebra2005
python data_preprocess.py --dataset_name=bridge2algebra2006
python data_preprocess.py --dataset_name=nips_task34
python data_preprocess.py --dataset_name=peiyou
python data_preprocess.py --dataset_name=assist2009
python data_preprocess.py --dataset_name=ednet5w
```

## Train & Evaluate
### Train
```
python -m torch.distributed.launch --nproc_per_node=1 wandb_gpt4kt_train.py --seq_len=200
```

## Evaluate
python wandb_predict.py --save_dir="/path/of/the/trained/model"