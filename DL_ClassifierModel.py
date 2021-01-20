import torch,time,os,pickle
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import *
from nnLayer import *
from metrics import *

class BaseModel:
    def __init__(self):
        pass
    def calculate_y_logit(self):
        pass
    def train(self, dataClass, trainSize, batchSize, epoch, 
              lr=0.001, weightDecay=0.0, stopRounds=10, threshold=0.2, earlyStop=10, 
              savePath='model/KAICD', saveRounds=1, isHigherBetter=True, metrics="MiF", report=["ACC", "MiF"]):
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X, Y = next(trainStream)
                loss = self._train_step(X, Y, optimizer)
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print("After iters %d: [train] loss= %.3f;"%(e*itersPerEpoch+i+1,loss), end='')
                    if dataClass.validSampleNum>0:
                        X, Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(' [valid] loss= %.3f;'%loss, end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print('========== Epoch:%5d =========='%(e+1))
                print('[Total Train]', end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                metrictor.set_data(Y_pre, Y, threshold)
                metrictor(report)
                print('[Total Valid]', end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y, threshold)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print('Bingo!!! Get a better Model with val %s: %.3f!!!'%(metrics,mtc))
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print('The val %s has not improved for more than %d steps in epoch %d, stop training.'%(metrics,earlyStop,e+1))
                        break
        self.load("%s.pkl"%savePath, dataClass=dataClass)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y, threshold)
        print('[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y, threshold)
        print('[Total Valid]',end='')
        res = metrictor(report)
        #metrictor.each_class_indictor_show(dataClass.id2lab)
        print('================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['nword2id'],stateDict['tword2id'] = dataClass.nword2id,dataClass.tword2id
            stateDict['id2nword'],stateDict['id2tword'] = dataClass.id2nword,dataClass.id2tword
            stateDict['icd2id'],stateDict['id2icd'] = dataClass.id2icd,dataClass.icd2id
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList = parameters['trainIdList']
            dataClass.validIdList = parameters['validIdList']
            dataClass.testIdList = parameters['testIdList']

            dataClass.nword2id,dataClass.tword2id = parameters['nword2id'],parameters['tword2id']
            dataClass.id2nword,dataClass.id2tword = parameters['id2nword'],parameters['id2tword']
            dataClass.id2icd,dataClass.icd2id = parameters['icd2id'],parameters['id2icd']     
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)['y_logit']
        return torch.sigmoid(Y_pre)
    def calculate_y(self, X, threshold=0.2):
        Y_pre = self.calculate_y_prob(X)
        isONE = Y_pre>threshold
        Y_pre[isONE],Y_pre[~isONE] = 1,0
        return Y_pre
    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X)
        Y_logit = out['y_logit']
        
        addLoss = 0.0
        if 'loss' in out: addLoss += out['loss']
        return self.crition(Y_logit, Y) + addLoss
    def calculate_indicator_by_iterator(self, dataStream, classNum, report, threshold):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        Metrictor.set_data(Y_prob_pre, Y, threshold)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream, threshold=0.2):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        isONE = Y_preArr>threshold
        Y_preArr[isONE],Y_preArr[~isONE] = 1,0
        return Y_preArr, YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate
    
class LD_PLAM(BaseModel):
    def __init__(self, classNum, embedding, labDescVec,
                 rnnHiddenSize=64, attnList=[],
                 embDropout=0.2, hdnDropout=0.2, fcDropout=0.5, numLayers=1, device=torch.device('cuda:0'), useCircleLoss=False):
        self.embedding = TextEmbedding(embedding, dropout=embDropout).to(device)
        self.biLSTM = TextLSTM(embedding.shape[1], rnnHiddenSize, bidirectional=True, dropout=hdnDropout, num_layers=numLayers).to(device)
        self.LNandDP = LayerNormAndDropout(rnnHiddenSize*2, hdnDropout).to(device)
        self.icdAttn = DeepICDDescAttention(rnnHiddenSize*2, classNum, labDescVec.shape[1], hdnDropout=hdnDropout, attnList=attnList, labDescVec=labDescVec).to(device)
        self.fcLinear = MLP(labDescVec.shape[1], 1, [], dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.embedding,self.biLSTM,self.LNandDP,self.icdAttn,self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss() if not useCircleLoss else MultiLabelCircleLoss()
        self.labDescVec = torch.tensor(labDescVec, dtype=torch.float32).to(device)
        self.device = device
        self.hdnDropout = hdnDropout
        self.fcDropout = fcDropout
    def calculate_y_logit(self, input):
        x = input['noteArr']
        if torch.cuda.device_count() > 1:
            x = nn.parallel.data_parallel(self.embedding,x) # => batchSize × seqLen × feaSize
            x = nn.parallel.data_parallel(self.biLSTM,x) # => batchSize × seqLen × hiddenSize*2
            x = nn.parallel.data_parallel(self.LNandDP,x) # => batchSize × seqLen × hiddenSize*2
            x = nn.parallel.data_parallel(self.icdAttn,x) # => batchSize × classNum × inSize
            x = nn.parallel.data_parallel(self.fcLinear,x).squeeze(dim=2) # => batchSize × classNum
        else:
            x = self.embedding(x) # => batchSize × seqLen × feaSize
            x = self.biLSTM(x) # => batchSize × seqLen × hiddenSize*2
            x = self.LNandDP(x) # => batchSize × seqLen × hiddenSize*2
            x = self.icdAttn(x) # => batchSize × classNum × inSize
            x = self.fcLinear(x).squeeze(dim=2) # => batchSize × classNum
        return {'y_logit':x}

class KAICD(BaseModel):
    def __init__(self, classNum, noteEmbedding, titleEmbedding, tokenizedTitle, titleLen,
                 noteFeaSize=256, titleFeaSize=256, hiddenSize=64, filterSize=64, contextSizeList=[1,3,5], hiddenSizeList=[],
                 embDropout=0.2, fcDropout=0.5, device=torch.device("cuda:0")):
        self.titles,self.titleLens = torch.tensor(tokenizedTitle,dtype=torch.long,device=device),torch.tensor(titleLen,dtype=torch.long,device=device)
        self.noteEmbedding = TextEmbedding(noteEmbedding, freeze=False, dropout=embDropout, name='noteEmbedding').to(device)
        self.titleEmbedding = TextEmbedding(titleEmbedding, freeze=False, dropout=embDropout, name='titleEmbedding').to(device)
        self.textCNN = TextCNN(noteFeaSize, filterSize, contextSizeList).to(device)
        self.textBiRNN = TextGRU(titleFeaSize, hiddenSize, bidirectional=True).to(device)
        self.simpleAttn = SimpleAttention(hiddenSize*2).to(device)
        self.knowledgeAttn = KnowledgeAttention(len(contextSizeList)*filterSize, hiddenSize*2).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterSize+hiddenSize*2, classNum, hiddenSizeList, dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.noteEmbedding,self.titleEmbedding,self.textCNN,self.textBiRNN,self.simpleAttn,self.knowledgeAttn,self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device
    def calculate_y_logit(self, input):
        x = input['noteArr']
        # x: batchSize × seqLen; inputLabel: labelNum × seqLen × feaSize
        noteConved = self.textCNN(self.noteEmbedding(x)) # => batchSize × scaleNum*filterSize
        labelEncoded = self.textBiRNN(self.titleEmbedding(self.titles)) # => labelNum × seqLen × hiddenSize*2
        labelEncoded = self.simpleAttn(labelEncoded) # => labelNum × hiddenSize*2

        knowledgeAttned = self.knowledgeAttn(noteConved, labelEncoded) # => batchSize × hiddenSize*2
        return {'y_logit':self.fcLinear(torch.cat([knowledgeAttned, noteConved], dim=1))} # => batchSize × classNum

class CAML(BaseModel):
    def __init__(self, classNum, noteEmbedding, titleEmbedding=None, contextSizeList=[1,3,5], filterSize=64,
                 embDropout=0.2, fcDropout=0.5, 
                 DR=False, tokenizedTitle=None, titleLen=None, lamb=0.01,
                 device=torch.device("cuda:0")):
        self.DR = DR
        self.noteEmbedding = TextEmbedding(noteEmbedding, freeze=False, dropout=embDropout, name='noteEmbedding').to(device)
        self.textCNN = TextCNN(noteEmbedding.shape[1], filterSize, contextSizeList, reduction='none').to(device)
        self.ICDAttn = ICDAttention(filterSize*len(contextSizeList), classNum).to(device)
        self.fcLinear = MLP(filterSize*len(contextSizeList), classNum).to(device)
        self.moduleList = nn.ModuleList([self.noteEmbedding,self.textCNN,self.ICDAttn,self.fcLinear])
        if self.DR:
            self.titleEmbedding = TextEmbedding(titleEmbedding, freeze=False, dropout=embDropout, name='titleEmbedding').to(device)
            self.titles = torch.tensor(tokenizedTitle,dtype=torch.long,device=device)
            self.labConv = TextCNN(titleEmbedding.shape[1], filterSize, contextSizeList, name='labConv').to(device)
            self.moduleList.append(self.titleEmbedding)
            self.moduleList.append(self.labConv)
        self.lamb = lamb
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device

    def calculate_y_logit(self, input):
        x = input['noteArr']
        x = self.noteEmbedding(x) # => batchSize × seqLen × embSize
        x = torch.cat(self.textCNN(x), dim=2) # => batchSize × seqLen × feaSize
        x = self.ICDAttn(x) # => batchSize × classNum × feaSize
        y = self.fcLinear.out.weight.mul(x).sum(dim=2) + self.fcLinear.out.bias
        if self.DR:
            z = self.labConv(self.titleEmbedding(self.titles)) # => classNum × feaSize
            # self.fcLinear.out.weight: 1 × feaSize
            dr = ((z - self.fcLinear.out.weight)**2).sum(dim=1) # => batchSize × classNum
            dr = (dr*input['lab']).sum(axis=1) / (input['lab'].sum(axis=1)+1) # => batchSize
            return {'y_logit':y, 'loss':self.lamb*dr.mean()}
        # x: batchSize × seqLen;
        return {"y_logit":y} # => batchSize × classNum

class MVCLDA(BaseModel):
    def __init__(self, classNum, noteEmbedding, titleEmbedding, contextSizeList=[1,3,5], filterSize=64,
                 embDropout=0.2, fcDropout=0.5, 
                 DR=False, tokenizedTitle=None, titleLen=None, lamb=0.01,
                 device=torch.device("cuda:0")):
        self.DR = DR
        self.noteEmbedding = TextEmbedding(noteEmbedding, freeze=False, dropout=embDropout).to(device)
        self.textCNN = TextCNN(noteEmbedding.shape[1], filterSize, contextSizeList, reduction='cpool').to(device)
        self.ICDAttn = ICDAttention(filterSize, classNum).to(device)
        self.fcLinear = MLP(filterSize, classNum).to(device)
        self.kLinear = MLP(1, classNum, actFunc=nn.Sigmoid, outAct=True, name='kLinear').to(device)
        self.moduleList = nn.ModuleList([self.noteEmbedding,self.textCNN,self.ICDAttn,self.fcLinear,self.kLinear])
        if self.DR:
            self.titleEmbedding = TextEmbedding(titleEmbedding, freeze=False, dropout=embDropout, name='titleEmbedding').to(device)
            self.titles = torch.tensor(tokenizedTitle,dtype=torch.long,device=device)
            self.labConv = TextCNN(titleEmbedding.shape[1], filterSize, contextSizeList, reduction='cpool', name='labConv').to(device)
            self.moduleList.append(self.titleEmbedding)
            self.moduleList.append(self.labConv)
        self.lamb = lamb
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device

    def calculate_y_logit(self, input):
        x = input['noteArr']
        x = self.noteEmbedding(x) # => batchSize × seqLen × embSize
        x = self.textCNN(x) # => batchSize × seqLen × feaSize
        x = self.ICDAttn(x) # => batchSize × classNum × feaSize
        l = input['noteLen'].unsqueeze(dim=1).to(self.device) # => batchSize × 1
        y = self.fcLinear.out.weight.mul(x).sum(dim=2) + self.fcLinear.out.bias + self.kLinear(l)
        if self.DR:
            z,_ = torch.max(self.labConv(self.titleEmbedding(self.titles)), dim=1) # => classNum × feaSize
            # self.fcLinear.out.weight: classNum × feaSize
            dr = ((z - self.fcLinear.out.weight)**2).sum(dim=1) # => batchSize × classNum
            dr = (dr*input['lab']).sum(axis=1) / (input['lab'].sum(axis=1)+1) # => batchSize
            return {'y_logit':y, 'loss':self.lamb*dr.mean()}
        # x: batchSize × seqLen;
        return {"y_logit":y} # => batchSize × classNum