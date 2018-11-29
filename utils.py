import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class visualization:
    def view_variable_correlation_heatmap(X, Y, size=10):
        """
        plot figures that show correlation between the objective variable and the explanatory variables.

        Parameters
        ----------
        X: pandas.DataFrame, shape=[n_sample, n_features]
            explanatory variable vectors, where n_sample is the number of samples and n_features is the number of features.
        Y: pandas.DataFrame, shape=[n_sample, 1]
            objective variable vectors.
        size: int
            maximum number of features for which correlation with objective variables is displayed at once.
        ----------
        
        """
        max_num = len(X)
        for i in range(max_num // size + 1):
            column_list = []
            min_idx = i*size
            max_idx = np.min([(i+1)*size, max_num])
            for j in range(min_idx, max_idx):
                column_list.append(X.columns[j])
            df = pd.concat([Y, X[column_list]], axis=1])
            plt.figure(figsize=(18, 18))
            sns.heatmap(df.corr(), square=True, cmap=plt.cm.viridis, linecolor='white', annot=True)
            plt.show()

    def view_variable_scatter_plot(df):
        plt.figure(figsize=(18, 18))
        sns.pairplot(df, hue='obj_rsv_flg', diag_kind='kde')

class preprocessing:
    def dm_preprocessing(dm):
  
        # 利用しないカラムを削除する
        dm = dm.drop(['randn', 'spouse_flg_is_pred', 'child_flg_is_pred'], axis=1) # 乱数の列を削除
        for column in dm.columns:
            if column.endswith('_01w'):
            dm = dm.drop(column, axis=1)  # 予測対象週の1週間前のデータは利用しない（運用では利用できない）

        # 型変換する
        for column in dm.columns:
            if column.startswith('sum_') or column.startswith('avg_') or 'cnt' in column or 'flg' in column or column == 'period_from_last_rsv' or column == 'c_cap_age_current' or column == 'customer_history':
            dm[column] = dm[column].astype(np.float64)
            elif column == 'c_cap_pred_cd' or column == 'c_cap_wid_cd' or column == 'c_cap_sex' or column == 'first_rsv_month' or column == 'available':
            dm[column] = dm[column].astype(str)

        # 欠損値を補完する
        for column in dm.columns:
            if dm[column].dtype == 'float64':
            dm[column] = dm[column].fillna(0) # 数値型は0
            elif dm[column].dtype == 'object':
            dm[column] = dm[column].fillna('NULL') # 文字列型はNULL

        # カテゴリ変数をダミー変数にする
        dm_tmp = pd.DataFrame(index=dm.index, columns=[])
        drop_columns = []

        for column in dm.iloc[:, 6:].columns: # 説明変数が対象
            if dm[column].dtype == 'object':
            dm_tmp[column] = dm[column]
            drop_columns.append(column)

        dm_dummy = pd.get_dummies(dm_tmp, drop_first=True)  # カテゴリー変数をダミー変数にする
        dm = pd.concat([dm, dm_dummy], axis=1)              # ダミー変数をもとのDMに追加

        for column in drop_columns:
            dm = dm.drop(column, axis=1)                      # ダミー変数にしたカラムを削除
            
        dm.reset_index()
        
        return dm