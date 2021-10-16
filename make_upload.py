from lightgbm import LGBMRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

def make_upload(imputer, model=LGBMRegressor(random_state=42), scaler=None, status=False, test_size=0.15, seed=42):
    tp_mean = 0.0
    for i in range(1,8,1):
        if imputer is not None:
            np_train, np_answer = np.array(globals()[f'train{i}']), np.array(globals()[f'answer{i}'])
            X_train, X_valid, y_train, y_valid = train_test_split(np_train, np_answer, test_size=test_size, random_state=seed)
            if scaler is True:
                X_train = scaler.fit_transform(X_train)
                X_valid = scaler.fit_transform(X_valid)

            ft_train = imputer.fit_transform(X_train)
            if "fancyimpute" in imputer.__module__:
                ft_valid = imputer.fit_transform(X_valid)
            else: ft_valid = imputer.transform(X_valid)
        else:
            np_train, np_answer = np.array(eval(f'r.temp_train{i}')), np.array(globals()[f'answer{i}'])
            ft_train, ft_valid, y_train, y_valid = train_test_split(np_train, np_answer, test_size=test_size, random_state=seed)

        model.fit(ft_train, y_train)
        output = model.predict(ft_valid)
        mae = mean_absolute_error(y_valid, output)
        tp_mean += mae
        print(mae, ft_valid.shape, end=' ')

        if status is True:
            if imputer is not None:
                X_test = np.array(globals()[f'test{i}'])
                if scaler is True: scaler.fit_transform(X_test)
                if "fancyimpute" in imputer.__module__:
                    ft_test = imputer.fit_transform(X_test)
                else: ft_test = imputer.transform(X_test)
            else:
                ft_test = np.array(eval(f'r.temp_test{i}'))
            print(ft_test.shape, end='')
            predict = model.predict(ft_test)
            upload_array = np.concatenate((ft_test, predict[:, np.newaxis]), axis=1)
            upload = pd.DataFrame(upload_array)
            ''' Need to modified the imputer parameter ASAP '''
            if imputer is not None:
                upload.to_csv(f'upload/{str(imputer)[:3]}_{str(model)[:4]}_upload{i}.csv', header=None, index=None, sep=',', mode='w')
            else:
                upload.to_csv(f'upload/Forest_{str(model)[:4]}_upload{i}.csv', header=None, index=None, sep=',', mode='w')
        print()
    print("TP_mean in valid set = {}".format(tp_mean))