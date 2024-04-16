# DSRNAFold

## Web server

A web server is working at http://123.60.79.219:5000/.

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
conda activate DSRNAFold
```
```python
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Data

### The dataset is sourced from the RNAStralign、ArchiveII and bpRNA. Download and extract it to the ts2vec/datasets directory.
[source data and trained model](https://drive.google.com/drive/folders/1Jk9e-gTk1xlpYomsDCJ9OyCJD0aFXJQF?usp=sharing) 

## Model
### Place the trained model into the corresponding folder under RNAFold/model.

## Usage

### If you want to retrain the model and make predictions, follow these steps sequentially.

### Step1: Obtaining the encoding model.(We have placed the trained encoding files into the "ts2vec/model_file" directory.)

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
## The data, code and model will be made publicly available after the acceptance of the paper.
