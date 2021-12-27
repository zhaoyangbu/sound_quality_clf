import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse.construct import random
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC, SMOTEN, ADASYN, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle

df = pd.read_json('embeddings.json')
X_train, X_test, y_train, y_test = train_test_split(df['embedding'], df['label'], test_size=0.2, random_state=42)


X_train_res = list(X_train)
y_train_res = list(y_train)
X_test = list(X_test)
y_test = list(y_test)


sm = SMOTE(random_state=42)
blsm = BorderlineSMOTE(random_state=42)
smnc = SMOTENC(random_state=42, categorical_features=[19,18])
smen = SMOTEN(random_state=42)
ada = ADASYN(random_state=42)
kmsm = KMeansSMOTE(random_state=42, cluster_balance_threshold=0.001)
svmsm = SVMSMOTE(random_state=42)
sme = SMOTEENN(random_state=42)


#X_train_res, y_train_res = sme.fit_resample(X_train, y_train)
#X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# X_train_res, y_train_res = blsm.fit_resample(X_train, y_train)
# X_train_res, y_train_res = smnc.fit_resample(X_train, y_train)
# X_train_res, y_train_res = smen.fit_resample(X_train, y_train)
# X_train_res, y_train_res = ada.fit_resample(X_train, y_train)
# X_train_res, y_train_res = kmsm.fit_resample(X_train, y_train)
# X_train_res, y_train_res = svmsm.fit_resample(X_train, y_train)
# X_train_res, y_train_res = sme.fit_resample(X_train, y_train)
#X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


clf = LogisticRegression()
print('training...')
clf.fit(X_train_res, y_train_res)
print('done')

train_result = clf.score(X_train_res, y_train_res)
test_result =  clf.score(X_test, y_test)
print("X test 0 is ",len(X_test[0]))

y_train_pred = clf.predict(X_train_res)
y_test_pred = clf.predict(X_test)
f1_train = f1_score(y_train_res, y_train_pred, average='macro')
f1_test = f1_score(y_test, y_test_pred, average='macro')


def count_tp_fp(y_test, y_test_pred):
    num_tp = 0
    num_fp = 0
    num_tn = 0
    for i in range(len(y_test)):
        if y_test_pred[i] == 1:
            if y_test[i] == 1:
                num_tp += 1
            else:
                num_fp +=1
        else:
            if y_test[i] == 1:
                num_tn += 1
            else:
                continue

    return  num_tp, num_fp, num_tn
    
            

def get_roc(pos_prob,y_true):
    pos = y_true[y_true==1]
    neg = y_true[y_true==0]
    threshold = np.sort(pos_prob)[::-1]        # 按概率大小逆序排列
    y = y_true[pos_prob.argsort()[::-1]]
    tpr_all = [0] ; fpr_all = [0]
    tpr = 0 ; fpr = 0
    x_step = 1/float(len(neg))
    y_step = 1/float(len(pos))
    y_sum = 0                                  # 用于计算AUC
    for i in range(len(threshold)):
        if y[i] == 1:
            tpr += y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr += x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum += tpr
    return tpr_all,fpr_all,y_sum*x_step  



 

num_tp, num_fp, num_tn = count_tp_fp(y_test, y_test_pred)
tp_rate = num_tp/(num_tp+num_fp)
base_effi = Counter(y_test)[1]/(Counter(y_test)[1]+Counter(y_test)[0])


pos_prob_lr = clf.predict_proba(X_test)[:,1]
y_ts = np.array(y_test)


tpr_lr,fpr_lr,auc_lr = get_roc(pos_prob_lr,y_ts) 



print('training accuracy is ', train_result)
print('testing accuracy is ', test_result)
print('f1_train is', f1_train)
print('f1_test is ', f1_test)
print('testing dataset %s' % Counter(y_test))
print('test_pred dataset %s' % Counter(y_test_pred))
print('num_tp is {}, num_fp is {}, num_tn is {}, efficiency improved from {} to {}'.format(num_tp, num_fp, num_tn, base_effi, tp_rate))


plt.plot(fpr_lr,tpr_lr,label="Logistic Regression (AUC: {:.3f})".format(auc_lr),linewidth=2)
plt.xlabel("False Positive Rate",fontsize=16)
plt.ylabel("True Positive Rate",fontsize=16)
plt.title("ROC Curve",fontsize=16)
plt.legend(loc="lower right",fontsize=16)
#plt.show()


# filename = 'lr_model.sav'
# pickle.dump(clf, open(filename, 'wb'))
# print('model saved')