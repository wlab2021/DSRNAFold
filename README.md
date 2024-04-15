# DSRNAFold

## Requirements

### The recommended requirements for DSRNAFold are specified as follows:

#### python: 3.7
#### numpy: 1.18.0
#### pytorch：1.13.0

### The dependencies can be installed by:

#### conda create -n DSRNAFold python=3.7

#### pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

## Data

[Click to download the data file](文件的原始链接)
### The dataset is sourced from the RNAStralign ArchiveII bpRNA.

## Usage

### Step1: Obtaining the encoding model

#### You need to navigate to the ts2vec file path and execute the following command.
```python
python get_encoding_model.py
```

###  Step2: Retrieve the .npz data files corresponding to the training and test sets.

#### You need to navigate to the RNAFold/code file path and execute the following command.

```python
python get_encoding_data.py
```

### Train the pre-trained model：

#### Execute `python pretrain.py` in the DSRNAfold/code folder.

### train the trained model:

#### Execute `python train.py` in the DSRNAfold/code folder.

### Use the trained model for prediction.

### Execute `python predict.py` in the DSRNAfold/code folder.

## Note

### All parameters are configured in JSON files located in the `util` directory.
