# LD-PLAM 
A Pseudo Label-wise Attention Network for Automatic ICD Coding

# Usage
## How to use it?
Firstly, you need to import the package.
```python
from utils import *
from DL_ClassifierModel import *
```
### 1. How to preprocess the raw data
Instance the MIMIC-III object and do preprocessing. 
```python
mimic = MIMIC_new(path='path to mimic3')
```
>**path** is the path of the raw MIMIC-III. 


After doing this, you can get a the output file named ***data.csv***. 
### 2. How to prepare the data class for the models. 
Instance the data utils class and get the pretrained word embedding. 
```python
dataClass = DataClass(dataPath='data.csv', mimicPath='path to mimic3', noteMaxLen=2500, minCount=2, topICD=-1)
dataClass.vectorize(noteFeaSize=128)
```
>**dataPath** is the path of "data.csv";.
>**mimicPath** is the path of the raw MIMIC-III.
>**noteMaxLen** is the maximum length of EMRs.
>**minCount** is the minimum frequency of words. 
>**topICD** is how many top-frequency ICD codes will be used. For MIMIC-III full, set it to -1. For MIMIC-III 50, set it to 50. 
>**noteFeaSize** is the embedding size of words in EMRs. 

In order to be consistent with previous experimental settings, we need to redivide the data set.
```python
dataClass = redivide_dataset(dataClass, camlPath='mimic3/caml')
```
>**dataClass** is the dataClass you obtained before. 
>**camlPath** is the path of caml's files, which is included in the folder mimic3 of this repository.

### 3. How to compute ICD vectors. 
Before train the model, we need to obtain the ICD vectors computed by ICDs' description first. 
```python
labDescVec = get_ICD_vectors(dataClass=dataClass, mimicPath="path to mimic3")
if dataClass.classNum=50:
    labDescVec = labDescVec[dataClass.icdIndex,:]
```
>**dataClass** is the dataClass you obtained before. 
>**mimicPath** is the path of the raw MIMIC-III.

Thus you can get the ICD vectors with shape (icdNum, 1024). 

## 4. How to train the models. 
Instance the model object and do training. 
```python
model = LD_PLAM(dataClass.classNum, dataClass.vector['noteEmbedding'], labDescVec, 
                rnnHiddenSize=128, attnList=[384], 
                embDropout=0.2, hdnDropout=0.2, device=torch.device('cuda:0'))
model.train(dataClass, trainSize=64, batchSize=64, epoch=128, 
            lr=0.0003, stopRounds=-1, threshold=0.5, earlyStop=30, 
            savePath='model/LD_PLAM', metrics="MiF", report=["MiF", "MiAUC", "P@5", "P@8"])
```
>**rnnHiddenSize** is the number of hidden units in BiLSTM. 
>**attnList** is the number of pseudo labels. 

Also, if you want to train the KAICD, CAML, DR-CAML, MVC-LDA or MVC-RLDA models, you just need to instance another model class (KAICD, CAML, MVCLDA, see lines 196~293 in DL_ClassifierModel.py for more details). 

## 5. How to do prediction
```python
model = LD_PLAM(dataClass.classNum, dataClass.vector['noteEmbedding'], labDescVec, 
                rnnHiddenSize=128, attnList=[384], 
                embDropout=0.2, hdnDropout=0.2, device=torch.device('cuda:0'))
model.load(path="xxx.pkl", map_location="cpu", dataClass=dataClass)
model.to_eval_mode()
Ypre,Y = model.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cpu')))
```
>**path** is your model saved path, which is a ".pkl" file. 

The output *Ypre* is your predicted values, *Y* is the true values. 
