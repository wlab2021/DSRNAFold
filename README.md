# DSRNAFold

## Requirements

### The recommended requirements for DSRNAFold are specified as follows:

```python
python: 3.7
numpy: 1.18.0
pytorch：1.13.0
```

### The dependencies can be installed by:
```python
conda create -n DSRNAFold python=3.7
```
```python
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Data

### The dataset is sourced from the RNAStralign、ArchiveII and bpRNA, download and extract it to the ts2vec/datasets directory.
[original bpseq format file, the code is nsdt](https://pan.baidu.com/s/1wxnsEe9j12EAacZhUWAFvA?pwd=nsdt)


## Usage

### Step1: Obtaining the encoding model.

#### You need to navigate to the ts2vec file path and execute the following command：
```python
python get_encoding_model.py
```

###  Step2: Retrieve the .npz data files corresponding to the training and test sets.

#### You need to navigate to the RNAFold/code file path and execute the following command：

```python
python get_encoding_data.py
```

### Step3: To obtain the pre-trained model.

####  You need to navigate to the RNAFold/code file path and execute the following command：

```python
python pretrain.py

```

### Step4: To obtain the trained model.
####  You need to navigate to the RNAFold/code file path and execute the following command：
```python
python train.py
```

### Using the trained model for prediction.
####  You need to navigate to the RNAFold/code file path and execute the following command：
```python
python predict.py
```

## Note

### All parameters are configured in JSON files located in the `util` directory.
