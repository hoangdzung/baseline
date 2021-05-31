
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

def eval(final_emb, labels, splits, random_state=42, clf=['mlp','sgd','lr','svm']):
    scaler = StandardScaler()
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    for node, emb in final_emb.items():
        if splits[node] == 1:
            X_train.append(emb)
            y_train.append(labels[node])
        elif splits[node] == 2:
            X_val.append(emb)
            y_val.append(labels[node])
        elif splits[node] == 3:
            X_test.append(emb)
            y_test.append(labels[node])
            
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
 
def main(args):
  
    embs = np.loadtxt(args.emb)
    emb_dict = {}
    for i in range(embs.shape[0]):
        emb_dict[int(embs[i][0])] = embs[i][1:]

    labels = np.loadtxt(args.labels).astype(int)
    label_dict = {}
    for i in range(labels.shape[0]):
        label_dict[labels[i][0]] = labels[i][1]

    splits = np.loadtxt(args.splits).astype(int)
    split_dict = {}
    for i in range(splits.shape[0]):
        split_dict[splits[i][0]] = splits[i][1]
    eval(emb_dict, label_dict, split_dict, random_state=args.seed, clf=args.clf.strip().split(","))

main(parse_args())