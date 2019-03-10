import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_many_null_colums(df, null_ratio=0.8):
    """ 欠損値が多いカラム名の抽出

    パラメーター
    -------------
    df: pandas.DataFrame
        操作対象のデータ shape=(n_samples, n_features)
    null_ratio: float
        nullが多いと判断するデータ数の割合

    アウトプット
    -------------
    del_columns: list
        欠損値が多いカラム名のリスト

    """

    null_num = df.isnull().sum()
    null_ratio = null_num / len(df)
    null_info_df = pd.DataFrame(np.c_[null_num.index, null_num, null_ratio], columns=['column_name', '#_of_null_records', 'null_ratio'])
    print(null_info_df.sort_values(by='null_ratio', ascending=False).query('null_ratio > 0'))
    del_columns = list(null_info_df.loc[null_info_df['null_ratio'] > null_ratio, 'column_name'])
    
    return del_columns


def complete_missing_values(df, fill_num='median', fill_str='mode'):
    """ 欠損値の補完

    パラメーター
    -------------
    df: pandas.DataFrame
        操作対象のデータ shape=(n_samples, n_features)
    null_ratio: float
        nullが多いと判断するデータ数の割合
    fill_num: int, float, 'mean', 'median', 'mode'
        数値型のカラムの欠損値の補完方法　数値（int, float）, 平均値（'mean'）・中央値（'median'）・最頻値（'mode'）
    fill_str: string, 'mode'
        文字列型のカラムの欠損値の補完方法　文字列（string）, 最頻値（'mode'）
    
    アウトプット
    -------------
    df: pandas.DataFrame
        欠損値を補完したデータ

    """

    for column in df.columns:
        if df[column].dtype == 'int64' or df[column].dtype == 'float64':
            if fill_num == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif fill_num == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif fill_num == 'mode':
                df[column] = df[column].fillna(df[column].mode())
            elif type(fill_num) in [int, float]:
                df[column] = df[column].fillna(fill_num)
            else:
                print("comlete_missing_values of df['"+column+"'] fails. set fill_num as int value, float value, 'mean', 'median', 'mode'.")
        elif df[column].dtype == 'object':
            if fill_str == 'mode':
                df[column] = df[column].fillna(df[column].mode())
            elif type(fill_str) is str:
                df[column] = df[column].fillna(fill_str)
            else:
                print("comlete_missing_values of df['"+column+"'] fails. set fill_str as str value, 'mode'.")
    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.Categorical(df[column])
    
    return df

def convert_data_type(df):
    """ 型変換
            object型のデータをカテゴリ型に変換

    パラメーター
    -------------
    df: pandas.DataFrame
        操作対象のデータ shape=(n_samples, n_features)
    
    アウトプット
    -------------
    df: pandas.DataFrame
        型変換したデータ

    """

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.Categorical(df[column])
    
    return df


def add_dummy_valuables(df):
    """ カテゴリ型データのダミー変数を導入

    パラメーター
    -------------
    df: pandas.DataFrame
        操作対象のデータ shape=(n_samples, n_features)
    
    アウトプット
    -------------
    df: pandas.DataFrame
        カテゴリ型データのダミー変数を導入したデータ

    """

    category_df = df.select_dtypes(include=['category'])
    del_columns = category_df.columns
    dummy_df = pd.get_dummies(category_df, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(del_columns, axis=1)
    
    return df


def standardize_data(X):
    """ 標準化
    パラメーター
    -------------
    X: pandas.DataFrame
        操作対象のデータ shape=(n_samples, n_features)
    
    アウトプット
    -------------
    X_std: pandas.DataFrame
        標準化したデータ
    sc: sklearn.preprocessing.StandardScaler
        標準化に利用したStandardScalerインスタンス

    """

    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    return X_std, sc

