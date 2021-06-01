
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
import os 
from tqdm import tqdm

def project(x, y, common_nodes = None ):
    anchors_model = x
    project_model = y

    common_nodes_ = set(anchors_model).intersection(project_model)
    n_common_nodes = int(len(common_nodes_))
    common_nodes_ = list(common_nodes_)[:n_common_nodes]
    if common_nodes is None:
        common_nodes = common_nodes_
    else:
        common_nodes = set(common_nodes).intersection(common_nodes_)

    if len(common_nodes) == 0:
        new_model = y.copy()
        new_model.update(x)
        return new_model

    anchors_emb = np.stack([anchors_model[i] for i in common_nodes])
#     anchors_emb = anchors_emb/np.sqrt(np.sum(anchors_emb**2,axis=1,keepdims=True))

    tobechanged_emb = np.stack([project_model[i] for i in common_nodes])
#     tobechanged_emb = tobechanged_emb/np.sqrt(np.sum(tobechanged_emb**2,axis=1,keepdims=True))

    trans_matrix, c, _,_ = np.linalg.lstsq(tobechanged_emb, anchors_emb, rcond=-1)
    tobechanged_emb = np.stack([project_model[i] for i in project_model])
#     tobechanged_emb = tobechanged_emb/np.sqrt(np.sum(tobechanged_emb**2,axis=1,keepdims=True))

    new_embeddings = np.matmul(tobechanged_emb, trans_matrix)

    new_model = dict(zip(project_model.keys(), new_embeddings))
    new_model.update(anchors_model)

    return new_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits')
    parser.add_argument('--labels')
    parser.add_argument('--emb_dir')
    parser.add_argument('--core_rate', type=float, default=1.0)
    parser.add_argument('--join')
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
    
    emb_dicts = []
    for f in os.listdir(args.emb_dir):
        # embs = np.loadtxt(os.path.join(args.emb_dir, f))
        n_err = 0
        emb_dict = {}
        # for i in range(embs.shape[0]):
        #     emb_dict[int(embs[i][0])] = embs[i][1:]
        for line in tqdm(open(os.path.join(args.emb_dir, f))):
            node, emb = line.strip().split(" ",1)
            try:
                emb_dict[int(node)] = np.array(list(map(float, emb.split())))
            except:
                n_err +=1
        print(n_err)
        emb_dicts.append(emb_dict)

    labels = np.loadtxt(args.labels).astype(int)
    label_dict = {}
    for i in range(labels.shape[0]):
        label_dict[labels[i][0]] = labels[i][1]

    splits = np.loadtxt(args.splits).astype(int)
    split_dict = {}
    for i in range(splits.shape[0]):
        split_dict[splits[i][0]] = splits[i][1]

    if 'project' in args.join:
        corenodes=list(set(emb_dicts[0]).intersection(emb_dicts[1]))
        n_core = int(len(corenodes)*args.core_rate)
        final_emb_merge = emb_dicts[0]
        for i in range(1,part_ids[1]-part_ids[0]+1):
            final_emb_merge = project(final_emb_merge,emb_dicts[i],corenodes[:n_core])
        eval(final_emb_merge, label_dict, split_dict, random_state=args.seed, clf=args.clf.strip().split(","))

    if 'rand' in args.join:
        final_emb_merge = emb_dicts[0]
        for i in range(1,part_ids[1]-part_ids[0]+1):
            final_emb_merge.update(emb_dicts[i])
        eval(final_emb_merge, label_dict, split_dict, random_state=args.seed, clf=args.clf.strip().split(","))

main(parse_args())