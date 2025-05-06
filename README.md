This is a pytorch implementation of **VLDB 25** paper [*Scalable Pre-Training of Compact Urban Spatio-Temporal Predictive Models on Large-Scale Multi-Domain Data*]

## Requirements

CompactST has been tested using Python 3.10 and CUDA Version: 12.1

To have consistent libraries and their versions, you can install the needed dependencies for this project by running the following command:

```shell
pip install -r requirements.txt
```

## Data
The datasets used in our paper will be publicly released soon.

## Run the Model

1. Pre-training

```shell
python pretrain.py
```

2. Fine-tuning

```shell
python few_shot.py --dataset {dataset} --unseen {dataset}
```

- `dataset`: the dataset name.
