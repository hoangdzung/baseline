
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz')
    parser.add_argument('--splits')
    parser.add_argument('--labels')
    parser.add_argument('--emb')
    parser.add_argument('--clf', default='lr')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args 

def kmean_eval(X,y):
    metrics1, metrics2, metrics3 = [], [], []
    for i in range(3):
        kmeans = MiniBatchKMeans(n_clusters=len(set(y)), random_state=i)
        pred = kmeans.fit_predict(X)
    metrics1.append(metrics.fowlkes_mallows_score(y, pred))
    metrics2.append(metrics.homogeneity_score(y, pred))
    metrics3.append(metrics.completeness_score(y, pred))
    print(np.mean(metrics1))
    print(np.mean(metrics2))
    print(np.mean(metrics3))

def eval(final_emb, labels, splits=None, random_state=42, clf=['mlp','sgd','lr','svm']):
    scaler = StandardScaler()
    if splits is not None:
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []
        for node, emb in final_emb.items():
            if splits[node] == 1:
                X_train.append(emb)
                y_train.append(str(labels[node]))
            elif splits[node] == 2:
                X_val.append(emb)
                y_val.append(str(labels[node]))
            elif splits[node] == 3:
                X_test.append(emb)
                y_test.append(str(labels[node]))
                
        X_train=np.stack(X_train)
        y_train=np.array(y_train)
        X_val=np.stack(X_val)
        y_val=np.array(y_val)
        X_test=np.stack(X_test)
        y_test=np.array(y_test)

        scaler.fit(np.vstack([X_train, X_val, X_test]))
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        if 'mlp' in clf:
            print("MLPClassifier")
            lr = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(64,),max_iter=5000)
            lr.fit(X_train, y_train)
            print(lr.score(X_train,y_train))
            print(lr.score(X_val,y_val))
            print(lr.score(X_test,y_test))
        if 'lr' in clf:
            print("LogisticRegression")
            lr=LogisticRegression(multi_class='multinomial',max_iter=5000)
            lr.fit(X_train, y_train)
            print(lr.score(X_train,y_train))
            print(lr.score(X_val,y_val))
            print(lr.score(X_test,y_test))
        if 'sgd' in clf:
            print("SGDClassifier")
            lr=SGDClassifier(max_iter=5000, tol=1e-3)
            lr.fit(X_train, y_train)
            print(lr.score(X_train,y_train))
            print(lr.score(X_val,y_val))
            print(lr.score(X_test,y_test))
        if 'svm' in clf:
            print("SVC")
            lr=SVC(gamma='auto',max_iter=5000)
            lr.fit(X_train, y_train)
            print(lr.score(X_train,y_train))
            print(lr.score(X_val,y_val))
            print(lr.score(X_test,y_test))
        if 'kmean' in clf:
            X = np.vstack([X_train, X_val, X_test])
            y = np.concatenate([y_train, y_val, y_test])
            kmean_eval(X,y)
    else:
        X = []
        y = []
        for node, emb in final_emb.items():
            if labels[node] != -1:
                X.append(emb)
                y.append(str(labels[node]))
        X = np.array(X)
        y = np.array(y)
        
        X = scaler.fit_transform(X)

        data = []
        for i in range(3):
            data.append(train_test_split(X, y, test_size=0.33, random_state=i))
        if 'mlp' in clf:
            print("MLPClassifier")
            lr = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(64,),max_iter=5000)
            train_acc, test_acc = [], []
            for X_train, X_test, y_train, y_test in data:
                lr.fit(X_train, y_train)
                train_acc.append(lr.score(X_train,y_train))
                test_acc.append(lr.score(X_test,y_test))
            print(np.mean(train_acc))
            print(np.mean(test_acc))
        if 'lr' in clf:
            print("LogisticRegression")
            lr=LogisticRegression(multi_class='multinomial',max_iter=5000)
            train_acc, test_acc = [], []
            for X_train, X_test, y_train, y_test in data:
                lr.fit(X_train, y_train)
                train_acc.append(lr.score(X_train,y_train))
                test_acc.append(lr.score(X_test,y_test))
            print(np.mean(train_acc))
            print(np.mean(test_acc))
        if 'sgd' in clf:
            print("SGDClassifier")
            lr=SGDClassifier(max_iter=5000, tol=1e-3)
            train_acc, test_acc = [], []
            for X_train, X_test, y_train, y_test in data:
                lr.fit(X_train, y_train)
                train_acc.append(lr.score(X_train,y_train))
                test_acc.append(lr.score(X_test,y_test))
            print(np.mean(train_acc))
            print(np.mean(test_acc))
        if 'svm' in clf:
            print("SVC")
            lr=SVC(gamma='auto',max_iter=5000)
            train_acc, test_acc = [], []
            for X_train, X_test, y_train, y_test in data:
                lr.fit(X_train, y_train)
                train_acc.append(lr.score(X_train,y_train))
                test_acc.append(lr.score(X_test,y_test))
            print(np.mean(train_acc))
            print(np.mean(test_acc))
        if 'kmean' in clf:
            kmean_eval(X,y)

def main(args):
    if args.npz is not None:
        data = np.load(args.npz)
        emb_dict = {}
        embs = data['emb']
        for i in range(embs.shape[0]):
            emb_dict[i] = embs[i] 
        
        if 'train_ids' in data:
            splits = np.zeros((len(emb_dict),))
            splits[data['train_ids']] = 1
            splits[data['val_ids']] = 2
            splits[data['test_ids']] = 3
        else:
            splits = None
        labels = data['labels']      
    else:
        embs = np.loadtxt(args.emb)
        emb_dict = {}
        for i in range(embs.shape[0]):
            emb_dict[int(embs[i][0])] = embs[i][1:]

        labels = np.loadtxt(args.labels).astype(int)
        if args.splits is not None:
            splits = np.loadtxt(args.splits).astype(int)[:,1]
        else:
            splits = None 
    eval(emb_dict, labels, splits, random_state=args.seed, clf=args.clf.strip().split(","))

main(parse_args())