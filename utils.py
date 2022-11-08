import pickle,random,torch,logging,time,gensim,os,re,gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from tqdm import tqdm
from collections import Counter
from transformers import RobertaModel,RobertaTokenizer,BertTokenizer, BertForMaskedLM, BertConfig, BertLMHeadModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM, XLNetLMHeadModel, BertModel


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MIMIC:
    def __init__(self, ADMISSIONSPATH="ADMISSIONS.csv", NOTEEVENTSPATH="NOTEEVENTS.csv", 
                       DIAGNOSESICDPATH="DIAGNOSES_ICD.csv", DICDDIAGNOSESPATH="D_ICD_DIAGNOSES.csv"):
        self.ADMISSIONSPATH = ADMISSIONSPATH
        self.NOTEEVENTSPATH = NOTEEVENTSPATH
        self.DIAGNOSESICDPATH = DIAGNOSESICDPATH
        self.DICDDIAGNOSESPATH = DICDDIAGNOSESPATH
    def get_basic_data(self, outPath='data.csv', noteOutPath='note_corpus.txt', titleOutPath='title_corpus.txt'):
        term_pattern = re.compile('[A-Za-z0-9]+|[,;.!?()]|<br>|<:>', re.I)
        contentParser = re.compile(r'[ \r]|\[[^\]]+\]|admission date:|discharge date:|date of birth:|sex:|service:|dictated by:.*$|completed by:.*$|signed electronically by:.*$', re.S)
        formatParser = re.compile('\n+', re.S)
        # get base training data
        admissions = pd.read_csv(self.ADMISSIONSPATH)
        noteevents = pd.read_csv(self.NOTEEVENTSPATH)
        diagnosesIcd = pd.read_csv(self.DIAGNOSESICDPATH)
        dIcdDiagnoses = pd.read_csv(self.DICDDIAGNOSESPATH)
        noteevents = noteevents[(noteevents['CATEGORY']=='Discharge summary') & (noteevents['DESCRIPTION']=='Report')]
        validIcd = list(dIcdDiagnoses['ICD9_CODE'].values)
        isValid = [icd in validIcd for icd in diagnosesIcd['ICD9_CODE']]
        diagnosesIcd = diagnosesIcd[isValid]
        out = pd.merge(noteevents[['HADM_ID','TEXT']],diagnosesIcd[['HADM_ID','ICD9_CODE']],how='left').dropna()
        out = out.groupby(by=['HADM_ID','TEXT']).agg(lambda x: ';'.join(x.values)).reset_index()
        out = out.astype({'HADM_ID':np.int32, 'TEXT':'str', 'ICD9_CODE':'str'})
        out['TEXT'] = out['TEXT'].map(lambda x: formatParser.sub( ' <br> ', contentParser.sub(' ', x.lower())) )
        out.to_csv(outPath)
        # get base corpus
        print('Generating note corpus...')
        with open(noteOutPath, 'w') as f:
            for note in tqdm(noteevents['TEXT']):
                note = note.replace('\\',' ').lower()
                note = contentParser.sub(' ', note)
                note = formatParser.sub(' <br> ', note)
                f.write(' '.join(term_pattern.findall(note))+'\n')
        print('Generating title corpus...')
        with open(titleOutPath, 'w') as f:
            for i in tqdm(range(dIcdDiagnoses.shape[0])):
                title = dIcdDiagnoses.iloc[i]
                title = title['SHORT_TITLE'].lower() + ' <:> ' + title['LONG_TITLE'].lower()
                f.write(' '.join(term_pattern.findall(title))+'\n')

class MIMIC_new:
    def __init__(self, path='raw_data/MIMIC3'):
        self.ADMISSIONSPATH = os.path.join(path, "ADMISSIONS.csv")
        self.NOTEEVENTSPATH = os.path.join(path, "NOTEEVENTS.csv")
        self.DIAGNOSESICDPATH = os.path.join(path, "DIAGNOSES_ICD.csv")
        self.DICDDIAGNOSESPATH = os.path.join(path, "D_ICD_DIAGNOSES.csv")
        self.PROCEDURESICDPATH = os.path.join(path, "PROCEDURES_ICD.csv")
        self.DICDPROCEDURESPATH = os.path.join(path, "D_ICD_PROCEDURES.csv")
    def get_basic_data(self, outPath='data.csv', noteOutPath='note_corpus.txt', titleOutPath='title_corpus.txt'):
        term_pattern = re.compile('[A-Za-z0-9]+|[,;.!?()]|<br>|<:>', re.I)
        contentParser = re.compile(r'[ \r]|\[[^\]]+\]|admission date:|discharge date:|date of birth:|sex:|service:|dictated by:.*$|completed by:.*$|signed electronically by:.*$', re.S)
        formatParser = re.compile('\n+', re.S)
        # get base training data
        admissions = pd.read_csv(self.ADMISSIONSPATH)
        noteevents = pd.read_csv(self.NOTEEVENTSPATH)
        
        diagnosesIcd = pd.read_csv(self.DIAGNOSESICDPATH).dropna(subset=['HADM_ID','ICD9_CODE'])
        diagnosesIcd.ICD9_CODE = diagnosesIcd.ICD9_CODE.astype('str').map(lambda x:'dia_' + x)
        dIcdDiagnoses = pd.read_csv(self.DICDDIAGNOSESPATH)

        proceduresIcd = pd.read_csv(self.PROCEDURESICDPATH).dropna(subset=['HADM_ID','ICD9_CODE'])
        proceduresIcd.ICD9_CODE = proceduresIcd.ICD9_CODE.astype('str').map(lambda x:'pro_' + x)
        dIcdProcedures = pd.read_csv(self.DICDPROCEDURESPATH)

        icd = pd.concat([diagnosesIcd,proceduresIcd], axis=0)
        dIcd = pd.concat([dIcdDiagnoses,dIcdProcedures], axis=0)

        noteevents = noteevents[(noteevents['CATEGORY']=='Discharge summary')]
        #validIcd = list(dIcdDiagnoses['ICD9_CODE'].values)
        #isValid = [icd in validIcd for icd in diagnosesIcd['ICD9_CODE']]
        #diagnosesIcd = diagnosesIcd[isValid]

        out = pd.merge(noteevents[['HADM_ID','TEXT']].groupby('HADM_ID').agg(lambda x: ' <br> '.join(x.values)).reset_index(),icd[['HADM_ID','ICD9_CODE']],how='left').dropna()
        out = out.groupby(by=['HADM_ID','TEXT']).agg(lambda x: ';'.join(x.values)).reset_index()
        out = out.astype({'HADM_ID':np.int32, 'TEXT':'str', 'ICD9_CODE':'str'})
        out['TEXT'] = out['TEXT'].map(lambda x: formatParser.sub( ' <br> ', contentParser.sub(' ', x.lower())) )
        out.to_csv(outPath)

class DataClass:
    def __init__(self, dataPath, mimicPath='mimic/', stopWordPath="stopwords.txt", validSize=0.2, testSize=0.0, minCount=10, noteMaxLen=768, seed=9527, topICD=-1):
        term_pattern = re.compile('[A-Za-z0-9]+|[,;.!?()]|<br>|<:>', re.I)
        self.minCount = minCount
        validSize *= 1.0/(1.0-testSize)
        # Open files and load data
        print('Loading the data...')
        data = pd.read_csv(dataPath, usecols=['HADM_ID', 'TEXT','ICD9_CODE'])
        # Get word-splited notes and icd codes
        print('Getting the word-splited notes and icd codes...')
        self.hadmId = data['HADM_ID'].values.tolist()
        NOTE,ICD = [term_pattern.findall(i) for i in tqdm(data['TEXT'])],list(data['ICD9_CODE'].values)
        self.rawNOTE = [i+["<EOS>"] for i in NOTE]
        del data
        gc.collect()
        # Calculate the word count
        print('Calculating the word count...')
        wordCount = Counter()
        for s in tqdm(NOTE):
            wordCount += Counter(s)
        # Drop low-frequency words and stopwords
        with open(stopWordPath, 'r') as f:
            stopWords = [i[:-1].lower() for i in f.readlines()]
        for i,s in tqdm(enumerate(NOTE)):
            NOTE[i] = [w if ((wordCount[w]>=minCount) and (w not in stopWords) and (len(w)>2)) else "<UNK>" for w in s]
        # Drop invalid data
        keepIndexs = np.array([len(i) for i in NOTE])>0
        print('Find %d invalid data, drop them!'%(sum(~keepIndexs)))
        NOTE,ICD = list(np.array(NOTE)[keepIndexs]),list(np.array(ICD)[keepIndexs])
        # Drop low TF-IDF words
        print('Dropping the unimportant words...')
        NOTE = self._drop_unimportant_words(NOTE, noteMaxLen)
        self.notes = [i+['<EOS>'] for i in NOTE]
        # Get the mapping variables for note-word and id
        print('Getting the mapping variables for note-word and id...')
        self.nword2id,self.id2nword = {"<EOS>":0, "<UNK>":1}, ["<EOS>", "<UNK>"]
        cnt = 2
        for note in tqdm(self.notes):
            for w in note:
                if w not in self.nword2id:
                    self.nword2id[w] = cnt
                    self.id2nword.append(w)
                    cnt += 1
        self.nwordNum = cnt
        # Get mapping variables for icd and id
        print('Getting the mapping variables for icd and id...')
        self.icd2id,self.id2icd = {},[]
        cnt,tmp = 0,[]
        for icds in ICD:
            icds = icds.split(';')
            for icd in icds:
                if icd not in self.icd2id:
                    self.icd2id[icd] = cnt
                    self.id2icd.append(icd)
                    cnt += 1
            tmp.append([self.icd2id[icd] for icd in icds])
        self.icdNum = cnt
        self.Lab = np.zeros((len(ICD),cnt), dtype='int32')
        for i,icds in enumerate(tmp):
            self.Lab[i,icds] = 1
        if topICD>0:
            icdCtr = self.Lab.sum(axis=0)
            usedIndex = np.argsort(icdCtr)[-topICD:]
            self.Lab = self.Lab[:,usedIndex]
            self.icdNum = topICD
            self.icdIndex = usedIndex
        # Get the mapping variables for title-word and id
        print('Getting the mapping variables for title-word and id...')
        self.tword2id,self.id2tword = {"<EOS>":0},["<EOS>"]
        cnt = 1
        
        dIcdDiagnoses = pd.read_csv(os.path.join(mimicPath,'D_ICD_DIAGNOSES.csv'))
        dIcdDiagnoses['ICD9_CODE'] = 'dia_' + dIcdDiagnoses['ICD9_CODE'].astype('str')
        dIcdDiagnoses = dIcdDiagnoses.set_index('ICD9_CODE')
        dicdProcedures = pd.read_csv(os.path.join(mimicPath,'D_ICD_PROCEDURES.csv'))
        dicdProcedures['ICD9_CODE'] = 'pro_' + dicdProcedures['ICD9_CODE'].astype('str')
        dicdProcedures = dicdProcedures.set_index('ICD9_CODE')
        icdTitles = pd.concat([dIcdDiagnoses,dicdProcedures])
        self.titles = []
        for icd in self.id2icd:
            try: 
                desc = (icdTitles.loc[icd]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd]['LONG_TITLE']).lower().split()
            except:
                desc = " <:> ".split()
            self.titles.append(desc+["<EOS>"])
            for w in desc:
                if w not in self.tword2id:
                    self.tword2id[w] = cnt
                    self.id2tword.append(w)
                    cnt += 1
        self.titleLen = [len(i) for i in self.titles]
        titleMaxLen = max(self.titleLen)
        self.twordNum = cnt
        # Tokenize the notes and titles
        print('Tokenizing the notes and the titles...')
        self.tokenizedNote = np.array([[self.nword2id[w] for w in n] for n in tqdm(self.notes)], dtype='int32')
        self.tokenizedTitle = np.array([[self.tword2id[w] for w in t] + [0]*(titleMaxLen-len(t)) for t in self.titles], dtype='int32')
        # Get some variables might be used
        self.totalSampleNum = len(self.tokenizedNote)
        restIdList,testIdList = train_test_split(range(self.totalSampleNum), test_size=testSize, random_state=seed) if testSize>0.0 else (list(range(self.totalSampleNum)),[])
        trainIdList,validIdList = train_test_split(restIdList, test_size=validSize, random_state=seed) if validSize>0.0 else (restIdList,[])
        
        self.trainIdList,self.validIdList,self.testIdList = trainIdList,validIdList,testIdList
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(self.trainIdList),len(self.validIdList),len(self.testIdList)

        self.classNum,self.vector = self.icdNum,{}

    def change_seed(self, seed=20201247, validSize=0.2, testSize=0.0):
        restIdList,testIdList = train_test_split(range(self.totalSampleNum), test_size=testSize, random_state=seed) if testSize>0.0 else (list(range(self.totalSampleNum)),[])
        trainIdList,validIdList = train_test_split(restIdList, test_size=validSize, random_state=seed) if validSize>0.0 else (restIdList,[])
        
        self.trainIdList,self.validIdList,self.testIdList = trainIdList,validIdList,testIdList
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(self.trainIdList),len(self.validIdList),len(self.testIdList)

    def vectorize(self, method="skipgram", noteFeaSize=320, titleFeaSize=192, window=5, sg=1, iters=10, batchWords=1000000,
                  noteCorpusPath=None, workers=8, loadCache=True, suf=""):
        path = 'wordEmbedding/note_%s_d%d%s.pkl'%(method,noteFeaSize,suf)
        if os.path.exists(path) and loadCache:
            with open(path, 'rb') as f:
                self.vector['noteEmbedding'] = pickle.load(f)
            print('Loaded cache from cache/%s'%path)
        else:
            corpus = self.rawNOTE if noteCorpusPath is None else LineSentence(noteCorpusPath)
            if method=='skipgram':
                model = Word2Vec(corpus, min_count=self.minCount, window=window, size=noteFeaSize, workers=workers, sg=1, iter=iters, batch_words=batchWords)
                word2vec = np.zeros((self.nwordNum, noteFeaSize), dtype=np.float32)
                for i in range(self.nwordNum):
                    if self.id2nword[i] in model.wv:
                        word2vec[i] = model.wv[self.id2nword[i]]
                    else:
                        print('word %s not in word2vec.'%self.id2nword[i])
                        word2vec[i] =  np.random.random(noteFeaSize)
                self.vector['noteEmbedding'] = word2vec
            elif method=='cbow':
                model = Word2Vec(corpus, min_count=self.minCount, window=window, size=noteFeaSize, workers=workers, sg=0, iter=iters, batch_words=batchWords)
                word2vec = np.zeros((self.nwordNum, noteFeaSize), dtype=np.float32)
                for i in range(self.nwordNum):
                    word2vec[i] = model.wv[self.id2nword[i]] if self.id2nword[i] in model.wv else np.random.random(noteFeaSize)
                self.vector['noteEmbedding'] = word2vec
            elif method=='MLM':
                tokenizer = BertTokenizer.from_pretrained('MLM')
                model = BertForMaskedLM.from_pretrained('MLM')
                self.vector['noteEmbedding'] = {'tokenizer':tokenizer, 'encoder':model}
                
            with open(path, 'wb') as f:
                pickle.dump(self.vector['noteEmbedding'], f, protocol=4)

        path = 'wordEmbedding/title_%s_d%d%s.pkl'%(method,titleFeaSize,suf)
        if os.path.exists(path) and loadCache:
            with open(path, 'rb') as f:
                self.vector['titleEmbedding'] = pickle.load(f)
            print('Loaded cache from cache/%s'%path)
        else:
            if method=='skipgram':
                model = Word2Vec(self.titles, min_count=0, window=window, size=titleFeaSize, workers=workers, sg=1, iter=10)
                word2vec = np.zeros((self.twordNum, titleFeaSize), dtype=np.float32)
                for i in range(self.twordNum):
                    word2vec[i] = model.wv[self.id2tword[i]]
                self.vector['titleEmbedding'] = word2vec
            elif method=='cbow':
                model = Word2Vec(self.titles, min_count=0, window=window, size=titleFeaSize, workers=workers, sg=0, iter=10)
                word2vec = np.zeros((self.twordNum, titleFeaSize), dtype=np.float32)
                for i in range(self.twordNum):
                    word2vec[i] = model.wv[self.id2tword[i]]
                self.vector['titleEmbedding'] = word2vec
            elif method=='MLM':
                pass
            with open(path, 'wb') as f:
                pickle.dump(self.vector['titleEmbedding'], f, protocol=4)

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu')):
        if type == 'train':
            idList = list(self.trainIdList)
        elif type == 'valid':
            idList = list(self.validIdList)
        elif type == 'test':
            idList = list(self.testIdList)
        noteLen = np.array((self.tokenizedNote==self.nword2id["<EOS>"]).argmax(axis=1)+1, dtype=np.int32)
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                yield {
                        "noteArr": torch.tensor(self.tokenizedNote[samples], dtype=torch.long, device=device), \
                        "lab": torch.tensor(self.Lab[samples].sum(axis=1), dtype=torch.int, device=device), \
                        "noteLen": torch.tensor(noteLen[samples], dtype=torch.float32), \
                      }, torch.tensor(self.Lab[samples], dtype=torch.float, device=device)
    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu')):
        if type == 'train':
            idList = self.trainIdList
        elif type == 'valid':
            idList = self.validIdList
        elif type == 'test':
            idList = self.testIdList
        noteLen = np.array((self.tokenizedNote==self.nword2id["<EOS>"]).argmax(axis=1)+1, dtype=np.int32)
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            yield {
                    "noteArr": torch.tensor(self.tokenizedNote[samples], dtype=torch.long, device=device), \
                    "lab": torch.tensor(self.Lab[samples], dtype=torch.int, device=device), \
                    "noteLen": torch.tensor(noteLen[samples], dtype=torch.float32) \
                  }, torch.tensor(self.Lab[samples], dtype=torch.float, device=device)
    
    def get_GCN_L(self):
        A = np.eye(self.classNum, dtype=np.float32)
        for lab in tqdm(self.Lab[self.trainIdList]):
            for i in range(len(lab)):
                if lab[i]==0: continue
                A[i,lab==1] = 1
        D = np.zeros((self.classNum,self.classNum), dtype=np.float32)
        D[range(len(D)),range(len(D))] = A.sum(axis=0)-1
        D[D>0] = D[D>0]**(-0.5)
        return D.dot(A).dot(D)

    def _drop_unimportant_words(self, sents, seqMaxLen):
        if seqMaxLen<0:
            return sents
        # keep top tf-idf words
        wordIdf = {}
        for s in sents:
            s = set(s)
            for w in s:
                if w in wordIdf:
                    wordIdf[w] += 1
                else:
                    wordIdf[w] = 1
        dNum = len(sents)
        for w in wordIdf.keys():
            wordIdf[w] = np.log(dNum/(1+wordIdf[w]))
        for i,s in enumerate(tqdm(sents)):
            if len(s)>seqMaxLen:
                wordTf = Counter(s)
                tfidf = [wordTf[w]*wordIdf[w] for w in s]
                threshold = np.sort(tfidf)[-seqMaxLen]
                sents[i] = [w for i,w in enumerate(s) if tfidf[i]>threshold]
            if len(sents[i])<seqMaxLen:
                sents[i] = sents[i] + ['<EOS>' for i in range(seqMaxLen-len(sents[i]))]
        return sents


def redivide_dataset(dataClass, camlPath='mimic3/caml'):
    trainHID = pd.read_csv(os.path.join(camlPath,'train_full_hadm_ids.csv'), header=None)[0].values
    validHID = pd.read_csv(os.path.join(camlPath,'dev_full_hadm_ids.csv'), header=None)[0].values
    testHID  = pd.read_csv(os.path.join(camlPath,'test_full_hadm_ids.csv'), header=None)[0].values

    trainIdList,validIdList,testIdList = [],[],[]
    for i,hid in enumerate(dataClass.hadmId):
        if hid in trainHID:
            trainIdList.append(i)
        elif hid in validHID:
            validIdList.append(i)
        elif hid in testHID:
            testIdList.append(i)

    dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList = trainIdList,validIdList,testIdList
    dataClass.trainSampleNum,dataClass.validSampleNum,dataClass.testSampleNum = len(trainIdList),len(validIdList),len(testIdList)
    dataClass.totalSampleNum = len(trainIdList)+len(validIdList)+len(testIdList)
    return dataClass

def toCODE(x):
    x = x.split('_')
    return ' '.join(x[0:1] + list(x[1]))
def unique(x):
    tmp = x.iloc[0]
    for i in range(1,x.shape[0]):
        tmp.SHORT_TITLE += f"; {x.iloc[i].SHORT_TITLE}"
        tmp.LONG_TITLE += f"; {x.iloc[i].LONG_TITLE}"
    return tmp

def get_ICD_vectors(dataClass, mimicPath):
    dia_icd = pd.read_csv(os.path.join(mimicPath,'D_ICD_DIAGNOSES.csv'))
    dia_icd['ICD9_CODE'] = dia_icd['ICD9_CODE'].map(lambda x:'dia_'+str(x))
    dia_icd['CODE'] = dia_icd['ICD9_CODE'].map(toCODE)
    dia_icd = dia_icd.groupby('ICD9_CODE').apply(unique).set_index('ICD9_CODE')
    pro_icd = pd.read_csv(os.path.join(mimicPath,'D_ICD_PROCEDURES.csv'))
    pro_icd['ICD9_CODE'] = pro_icd['ICD9_CODE'].map(lambda x:'pro_'+str(x))
    pro_icd['CODE'] = pro_icd['ICD9_CODE'].map(toCODE)
    pro_icd = pro_icd.groupby('ICD9_CODE').apply(unique).set_index('ICD9_CODE')

    icd = pd.concat([dia_icd,pro_icd])

    for new in set(dataClass.id2icd) - set(icd.index):
        icd.loc[new] = [-1, "", "", toCODE(new)]
    icd = icd.loc[dataClass.id2icd]

    tokenizer = RobertaTokenizer.from_pretrained('trainMLM/model/roberta_large/')
    model = RobertaModel.from_pretrained(
                'trainMLM/model/roberta_large/',
            ).eval().to('cuda:0')

    id2descVec = np.zeros((len(icd),1024), dtype='float32')
    for i,item in enumerate(tqdm(icd.itertuples())):
        tmp = item.CODE + ' : ' + item.SHORT_TITLE + ' ; ' + item.LONG_TITLE
        inputs = tokenizer(tmp, 
                           padding=True,
                           truncation=True,
                           max_length=512, 
                           return_tensors="pt")
        tmp = model(**inputs.to('cuda:0'))[0].detach().data.cpu().numpy().mean(axis=1)
        id2descVec[i] = tmp
        
    return id2descVec

