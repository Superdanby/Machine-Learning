import pandas as pd
import pathlib
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from datetime import datetime


DATA_PATH1TRAIN = pathlib.Path('full1_upload/training_data').resolve()
DATA_PATH1TEST = pathlib.Path('full1_upload/test_feature').resolve()
DATA_PATH2TRAIN = pathlib.Path('upload_full2/training_data').resolve()
DATA_PATH2TEST = pathlib.Path('upload_full2/test_feature').resolve()

# OW = Observation Window
# PW = Prediction Window


def load_paths():
    """ Load paths """

    train1 = sorted(list(DATA_PATH1TRAIN.glob('*.csv')))
    test1 = sorted(list(DATA_PATH1TEST.glob('*.csv')))
    train2 = sorted(list(DATA_PATH2TRAIN.glob('*.csv')))
    test2 = sorted(list(DATA_PATH2TEST.glob('*.csv')))

    return train1, test1, train2, test2


def load_data(data_path=None):
    """ Load data """

    data = []
    print('Loading data...')
    for i, p in enumerate(tqdm(data_path)):
        # print(str(p))
        df = pd.read_csv(str(p), low_memory=False)
        # print(df.head(10))
        data.append(df)
    return data


def count_only(train=None, test=None):
    """ Extract count from DataFrame """

    df_cnt = []
    df_lbl = []
    print('\nExtracting "count" from training data...')
    for x in tqdm(train):
        lbl = x['Label']
        x = x[x.columns[x.loc[0] == 'count']].fillna(0)
        # x = pd.concat([x, lbl], axis=1, join_axes=[x.index])
        x = x.iloc[2:]
        lbl = lbl.iloc[2:]
        lbl = lbl.astype(bool)
        # print(x.head(5))
        # print(lbl.head(5))
        df_cnt.append(x)
        df_lbl.append(lbl)

    test_cnt = []
    test_meta = []
    print('\nExtracting "count" from test data...')
    for x in tqdm(test):
        meta = x.iloc[:, 1:3]
        x = x[x.columns[x.loc[0] == 'count']].fillna(0)
        x = x.iloc[2:]
        meta = meta.iloc[2:]
        meta.columns = ['machine', 'date']
        for i in meta.index:
            time_str = meta.at[i, 'date']
            dt = datetime.strptime(time_str, '%Y-%m-%d')
            time_str = dt.strftime("-%d/%m/%Y")
            meta.at[i, 'machine'] = meta.at[i, 'machine'] + time_str

        meta = meta.iloc[:, :1]
        # print(meta.head(5))
        # print(x.head(5))
        test_cnt.append(x)
        test_meta.append(meta)

    return df_cnt, df_lbl, test_cnt, test_meta


def train_predict(train_cnt, train_lbl, test_cnt, test_meta, opt_name):
    """ Train, validate, and predict """

    acc_list = []
    print('\nTraining...')
    for X, Y, Z, M, N in tqdm(zip(train_cnt, train_lbl, test_cnt, test_meta, opt_name)):
        # print(X.head(5))
        # print(Y.head(5))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y)
        # clf = svm.SVR()
        # clf.fit(X_train, Y_train)
        # clf_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
        # clf_rbf.fit(X_train, Y_train)
        param_grid = [
            {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['auto', 0.1, 1, 10], 'kernel': ['sigmoid', 'rbf'],},
            # {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['auto'], 'kernel': ['linear', 'poly'],}
        ]
        clf_gs = GridSearchCV(svm.SVR(), param_grid, verbose=2, n_jobs=-1, return_train_score=True)
        clf_gs.fit(X_train, Y_train)
        val_pred = clf_gs.predict(X_test)

        print('\nValidating...')
        correct_cnt = 0
        for pred_lbl, true_lbl in tqdm(zip(val_pred, Y_test)):
            if (pred_lbl > 0.5 and true_lbl) or (pred_lbl < 0.5 and not true_lbl):
                correct_cnt = correct_cnt + 1
        val_accuracy = correct_cnt / len(Y_test)
        print(f'val_acc: {val_accuracy}')
        acc_list.append(val_accuracy)

        print('\nPredicting...')
        # pred = clf.predict(Z)
        # pred = clf_rbf.predict(Z)
        pred = clf_gs.predict(Z)
        M.columns = ['id']
        # output = pd.concat([M, pred], axis=1, join_axes=[M.index])
        M['Label'] = pred
        # output.columns = ['id', 'Label']
        # output.to_csv(N + 'PW1.csv', sep=',', index=False)
        M.to_csv(N + 'pred.csv', sep=',', index=False)
    return acc_list


if __name__ == '__main__':

    path_train1, path_test1, path_train2, path_test2 = load_paths()
    train1 = load_data(path_train1)
    test1 = load_data(path_test1)
    train1_cnt, train1_lbl, test1_cnt, test1_meta = count_only(train1, test1)
    opt_name1 = [fpath.name[:-4] + '1' for fpath in path_test1]
    acc1 = train_predict(train1_cnt, train1_lbl, test1_cnt, test1_meta, opt_name1)

    train2 = load_data(path_train2)
    test2 = load_data(path_test2)
    train2_cnt, train2_lbl, test2_cnt, test2_meta = count_only(train2, test2)
    opt_name2 = [fpath.name[:-4] + '2' for fpath in path_test2]
    acc2 = train_predict(train2_cnt, train2_lbl, test2_cnt, test2_meta, opt_name2)

    print('Accuracy:')
    for acc, opt_name in tqdm(zip(acc1, opt_name1)):
        print(f'{opt_name}: {acc}')
    for acc, opt_name in tqdm(zip(acc2, opt_name2)):
        print(f'{opt_name}: {acc}')
