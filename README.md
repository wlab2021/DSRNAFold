# DSRNAFold

## Overview
DSRNAFold is a deep learning-based tool for predicting RNA secondary structures, integrating RNA sequence and structural context information. This README provides instructions on how to set up and use DSRNAFold.

The data, code and models will be made publicly available upon the acceptance of the paper. Currently, access is restricted to reviewers and can be obtained `using the password` provided in the `Code availability` section of the submitted manuscript.

## Web server

Visit the [DSRNAFold web server](http://123.60.79.219:5000) to access the online platform and utilize the model interactively.

## Requirements

### OS requirements
  The package development version is tested on Linux operating systems. The developmental version of the package has been tested on the following systems:
  
Linux: (Ubuntu 20.04.3 LTS)

### Hardware environment
Physical Memory: 515GB.
GPU: RTX 3090.

The recommended requirements for DSRNAFold are specified as follows:

```python
python: 3.7
numpy: 1.18.0
pytorch：1.13.0
```

The dependencies can be installed by:(code environment installation takes approximately `10 minutes`)
```python
conda create -n DSRNAFold python=3.7
```
```python
conda activate DSRNAFold
```
```python
pip install -r requirements.txt
```

## Data

The dataset is sourced from the RNAStralign, ArchiveII, bpRNA, Rfam14 and EternaBench. RNAStralign, ArchiveII, and bpRNA datasets are used for performance testing. EternaBench and Rfam14 datasets for chemical mapping and riboswitch validation experiments. RNAStralign dataset also used for RNA classification experiments. Download and extract it to the ts2vec/datasets folder.


## Model
Place the trained model into the corresponding folder under RNAfold/model.

### Due to limited space on GitHub, the raw data and model can be obtained from this link: [datasets and models](https://drive.google.com/drive/folders/1Jk9e-gTk1xlpYomsDCJ9OyCJD0aFXJQF?usp=sharing) .

## Usage

### If you want to retrain the model and make predictions, follow these steps sequentially.

### Prep
  All parameters are configured in `json` files located in the `util` directory. 
  You need to put the json file of the corresponding dataset into the `DSRNAFold/util` folder and update `args` parameter in the corresponding `.py` file.

### Step1: Obtaining the encoding model.(We have placed the trained encoding files into the "ts2vec/model_file" directory.)

You need to navigate to the ts2vec file path and execute the following command：
```python
python get_encoding_model.py
```

###  Step2: Retrieve the .npz data files corresponding to the training and test sets.

You need to navigate to the RNAFold/code file path and execute the following command：
Please note that due to the large size of the generated npz file, it is recommended that the host has at least 512GB of memory.

```python
python get_encoding_data.py
```

### Step3: To obtain the pre-trained model.

You need to navigate to the RNAFold/code file path and execute the following command：

```python
python pretrain.py

```

### Step4: To obtain the trained model.
You need to navigate to the RNAFold/code file path and execute the following command：
```python
python train.py
```

### Using the trained model for prediction.
You need to navigate to the RNAFold/code file path and execute the following command：
```python
python predict.py
```

## For Demo
  ### RNAStralign_ArchiveII-128(please note that this only contains partial data.):
    For Step1: Total run time: 13.91 minutes
    For Step2: Total run time: 2.71 minutes
    For Step3: Total run time: 4.16 minutes
    For Step4: Total run time: 1.01 minutes
    For predict: Total run time: 0.72 minutes
  ### RNAStralign_ArchiveII-512(please note that this only contains partial data.):

## Note
  For other verification experiments, please see readme.md in the corresponding folder.
