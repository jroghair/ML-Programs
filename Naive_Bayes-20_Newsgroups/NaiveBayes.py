# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:03:15 2019

@author: jroghair

This is an implementation of the naive byaes algorithm for text classification. 
This particulary implementation is designed for classifying 20,0000 newspaper documents into 20 news categories using words in vocabulary.txt. The categories are:
    
1) alt.atheism
2) comp.graphics 
3) comp.os.ms-windows.misc 
4) comp.sys.ibm.pc.hardware 
5) comp.sys.mac.hardware
6) comp.windows.x 
7) misc.forsale
8) rec.autos 
9) rec.motorcycles 
10) rec.sport.baseball 
11) rec.sport.hockey
12) sci.crypt
13) sci.electronics 
14) sci.med 
15) sci.space 
16) soc.religion.christian 
17) talk.politics.guns 
18) talk.politics.mideast
19) talk.politics.misc 
20) talk.religion.misc
"""

import pandas as pd
import os
import math
import time

#Global Variables
prior_dict = dict()
wordCnt_dict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0}
posterior_dict = {1:{}, 2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{},10:{},11:{},12:{},13:{},14:{},15:{},16:{},17:{},18:{},19:{},20:{}}
mlEst_dict = {1:{}, 2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{},10:{},11:{},12:{},13:{},14:{},15:{},16:{},17:{},18:{},19:{},20:{}}
bayesEst_dict = {1:{}, 2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{},10:{},11:{},12:{},13:{},14:{},15:{},16:{},17:{},18:{},19:{},20:{}}
vocabMapping_dict = dict()
   

def main():
    start_time = time.time()
    print('Begin - Naive Bayes Algorithm')
    os.chdir(os.getcwd())
    train_label = pd.read_csv("train_label.csv", header=None, names=["Category"])
    train_data = pd.read_csv("train_data.csv", header=None, names=["docIdx", "wordIdx", "count"])
    test_label = pd.read_csv("test_label.csv", header=None, names=["Category"])
    test_data =  pd.read_csv("test_data.csv", header=None, names=["docIdx", "wordIdx", "count"])
    train_label.index+=1
    test_label.index+=1
    train_set = pd.merge(train_data, train_label, left_on="docIdx", right_index=True)
    test_set = pd.merge(test_data, test_label, left_on="docIdx", right_index=True)
    print("Successfully imported data")  
        
    #Learn the model        
    modelLearning(train_set)  
    
    #Evaluate training set using maximum likelihood estimater
    print("Evaluating training set using Maximum Likelihood Estimator.")
    df_train_MLE = evaluationMLE(train_set)
    performance(df_train_MLE)
    confusionMatrix(df_train_MLE)    
    
    #Evaluate training set using bayes estimator
    print("Evaluating training set using Bayes Likelihood Estimator.")
    df_train_Bayes = evaluationBayes(train_set)
    performance(df_train_Bayes)
    confusionMatrix(df_train_Bayes)    
    
    #Evaluate Test set using maximum likelihood estimater
    print("Evaluating test set using Maximum Likelihood Estimator.")
    df_test_MLE = evaluationMLE(test_set)
    performance(df_test_MLE)
    confusionMatrix(df_test_MLE)    
    
    #Evaluate training set using bayes estimator
    print("Evaluating test set using Bayes Estimator.")
    df_test_Bayes = evaluationBayes(test_set)
    performance(df_test_Bayes)
    confusionMatrix(df_test_Bayes)    
    print("End - Naive Bayes Algorithm")
    print("--- Ran in %s seconds ---" % (time.time() - start_time))
    
#Calculate class(newspaper) prior probabilities P(Wj)
def modelLearning(df):
    with open("vocabulary.txt", "r") as f:
        vocab_line = f.readlines()
        wordIdx = 1
        for word in vocab_line:
            key = wordIdx 
            val = word.strip()
            if key not in vocabMapping_dict.keys():
                vocabMapping_dict[key] = val
                for category in posterior_dict.keys():
                    posterior_dict[category][wordIdx] = 0 #only consider words in vocabulary
                    wordCnt_dict[category] = 0
            wordIdx+=1
    doc_cnt = 0
    #calculate counts prior probabilities P(Wj) and posterior probabilities P(Wk|wj)    
    #get count of number of documents in a specific category
    df_prior = df[['docIdx', 'Category']]
    df_prior = df_prior.drop_duplicates()
    for row in df_prior.itertuples(index=True, name='Pandas'):
        category = int(getattr(row, "Category"))
        if category not in prior_dict:
            prior_dict[category] = 0
        prior_dict[category] +=1
        doc_cnt +=1
    
    for row in df.itertuples(index=True, name='Pandas'):
        category = int(getattr(row, "Category"))
        wordCnt = int(getattr(row, "count"))
        wordId = int(getattr(row, "wordIdx"))
        specCategory = posterior_dict[category]
        if wordId not in specCategory.keys():
            continue  ##not considering those NOT in vocab
        posterior_dict[category][wordId] += wordCnt 
        wordCnt_dict[category] += wordCnt #total for each newsgroup 
            
    #Print prior probabilities P(Wj)
    print("Prior probabilities for each class: P(Wj)")
    for key in prior_dict.keys():
        val = prior_dict[key]
        prior_dict[key] = val/doc_cnt
        print("P(Omega = %d) = %.5f" % (key, prior_dict[key]))
    
    #Make maximum likelihood and bayesian estimator calculations 
    for category in posterior_dict.keys():
        classJ = posterior_dict[category]
        for word in classJ.keys():
            num =  posterior_dict[category][word]
            denom = wordCnt_dict[category]
            mlEst = num/denom
            mlEst_dict[category][word] = mlEst
            bayesEst = ((num+1)/(denom + len(vocabMapping_dict.keys())))
            bayesEst_dict[category][word] = bayesEst
            
    print("Model Trained, Learning Phase Complete")
    
    


def evaluationMLE(df):
    classifiedDocs = pd.DataFrame(columns=["docIdx", "Predicted"]); 
    document_dict = dict() #key = documentID, value = {newsgroup: posterior probability}
    
    #Calculate posterior probabilities for each word Xi for class Wj: P(Xi, Wj)
    for row in df.itertuples(index=True, name='Pandas'):
        doc = int(getattr(row, "docIdx"))
        wordCnt = int(getattr(row, "count"))
        word = int(getattr(row, "wordIdx"))
        if doc not in document_dict:
            document_dict[doc] = {}
        for category in range(1, 21):
            if category not in document_dict[doc].keys():
                document_dict[doc][category] = 0
            posterior= mlEst_dict[category][word]
            posterior = posterior*wordCnt #i positions in the document
            if posterior!=0:
                val = math.log(posterior)
                document_dict[doc][category] +=  val
            else: #This needs to be re-evaluated/changed - likely to zero 
                document_dict[doc][category] += float("-inf") 
                
    #add priors to each class and find the argmax 
    for doc in document_dict.keys():
        curMaxCategory = 1
        curMaxMLE = float("-inf") #0
        for category in document_dict[doc]:
            position_val = math.log(prior_dict[category])
            document_dict[doc][category] += position_val
            if curMaxMLE < document_dict[doc][category]: 
                curMaxMLE = document_dict[doc][category]
                curMaxCategory = category
        classifiedDocs.loc[len(classifiedDocs)] = [doc, curMaxCategory]
        
    df['docIdx']=df['docIdx'].astype(int)
    classifiedDocs['docIdx']=classifiedDocs['docIdx'].astype(int)
    df_with_pred = pd.merge(df, classifiedDocs , how='left', on=['docIdx'])
    df_with_pred = df_with_pred[['docIdx', 'Category', 'Predicted']]
    df_with_pred = df_with_pred.drop_duplicates()
    
    return df_with_pred
    
    
def evaluationBayes(df):
    classifiedDocs = pd.DataFrame(columns=["docIdx", "Predicted"]); 
    document_dict = dict() #key = documentID, value = {newsgroup: posterior probability}
    
    #Calculate posterior probabilities for each word Xi for class Wj: P(Xi, Wj)
    for row in df.itertuples(index=True, name='Pandas'):
        doc = int(getattr(row, "docIdx"))
        wordCnt = int(getattr(row, "count"))
        word = int(getattr(row, "wordIdx"))
        if doc not in document_dict:
            document_dict[doc] = {}
        for category in range(1, 21):
            docCategory = document_dict[doc]
            if category not in docCategory.keys():
                document_dict[doc][category] = 0
            posterior= bayesEst_dict[category][word]
            posterior *=wordCnt
            ln_post = math.log(posterior)
            document_dict[doc][category] +=  ln_post
        
    #add priors to each class and find the argmax 
    for doc in document_dict.keys():
        curMaxCategory = None
        curMaxBayes = float("-inf")
        for category in document_dict[doc]:
            position_val = math.log(prior_dict[category])
            document_dict[doc][category] += position_val
            if curMaxBayes < document_dict[doc][category]: 
                curMaxBayes = document_dict[doc][category]
                curMaxCategory = category
        classifiedDocs.loc[len(classifiedDocs)] = [doc, curMaxCategory]
    
    df['docIdx']=df['docIdx'].astype(int)
    classifiedDocs['docIdx']=classifiedDocs['docIdx'].astype(int)
    df_with_pred = pd.merge(df, classifiedDocs , how='left', on=['docIdx'])
    df_with_pred = df_with_pred[['docIdx', 'Category', 'Predicted']]
    df_with_pred = df_with_pred.drop_duplicates()
    
    return df_with_pred

def performance(df):
    newsgroups = dict()
    n, k = 0, 0
    
    for row in df.itertuples(index=True, name='Pandas'):
        category = int(getattr(row, "Category"))
        pred = int(getattr(row, "Predicted"))
        if category not in newsgroups.keys():
            newsgroups[category] = (0,0)
        newMatch = newsgroups[category][0]
        if(category==pred):
            k+=1
            newMatch = newsgroups[category][0] + 1
        newTotal = newsgroups[category][1] + 1
        newsgroups[category] = (newMatch, newTotal)
        n+=1
        
    print("Overall Accuracy = %.5f"% (k/n))
    print("Class Accuracy")
    
    for category in newsgroups.keys():
        num = newsgroups[category][0]
        denom = newsgroups[category][1]
        print("Group %d: %.5f" % (category, (num/denom)))


def confusionMatrix(df):
        conf_dict = dict()
        for row in df.itertuples(index=True, name='Pandas'):
            category = int(getattr(row, "Category"))
            predicted = int(getattr(row, "Predicted"))
            if category not in conf_dict.keys():
                conf_dict[category] = {}
            if predicted not in conf_dict[category]:
                conf_dict[category][predicted] = 0
            conf_dict[category][predicted] +=1 

        conf_matrix = []
        for category in range(1,21):
            clist = []
            for predicted in range(1,21):
                if predicted not in conf_dict[category]:
                    clist.append('  0')
                else:
                    val = conf_dict[category][predicted]
                    if len(str(val)) == 2:
                        clist.append((' ' + str(val)))
                    elif len(str(val)) == 1:
                        clist.append(('  ' + str(val)))
                    else: 
                        clist.append(str(val))
            conf_matrix.append(clist)
        
        print('[%s]' % ', '.join(map(str, conf_matrix[0])))
        for i in range(1,19):
            print('[%s]' % ', '.join(map(str, conf_matrix[i])))
        print('[%s]' % ', '.join(map(str, conf_matrix[19])))

main() 