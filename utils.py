import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class visualization:
    def view_variable_correlation_heatmap(X, Y, size=10, file_name=None):
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
        fine_name: string
            file path to save figures.
        ----------
        
        """
        if file_name != None:
            pdf = PdfPages(file_name)
        
        max_num = len(X)
        for i in range(max_num // size + 1):
            column_list = []
            min_idx = i*size
            max_idx = np.min([(i+1)*size, max_num])
            for column in exp_df.columns[min_idx:max_idx]:
                column_list.append(column)
            df = pd.concat([Y, X[column_list]], axis=1)

            plt.figure(figsize=(18, 18))
            sns.heatmap(df.corr(), square=True, cmap=plt.cm.viridis, linecolor='white', annot=True)
            if file_name != None:
                pdf.savefig()
            else:
                plt.show()

        if file_name != None:
            pdf.close()


    def view_variable_scatter_plot(X, Y, size=10, , objective='regression', file_name=None):
        """
        create scatter plots between the objective variable and the explanatory variables.

        Parameters
        ----------
        X: pandas.DataFrame, shape=[n_sample, n_features]
            explanatory variable vectors, where n_sample is the number of samples and n_features is the number of features.
        Y: pandas.DataFrame, shape=[n_sample, 1]
            objective variable vectors.
        size: int
            maximum number of features for which scatter plots with objective variables is displayed at once.
        objective: string
            type of objective variables. (This method create KDE(Kernel Density Estimate) plot for each categories.)
            'regression' for numerical variables, 'multiclass' for categorical variables.
        fine_name: string
            file path to save figures.
        ----------
        
        """
        if file_name != None:
            pdf = PdfPages(file_name)
        
        max_num = len(X)
        for i in range(max_num // size + 1):
            column_list = []
            min_idx = i*size
            max_idx = np.min([(i+1)*size, max_num])
            for column in exp_df.columns[min_idx:max_idx]:
                column_list.append(column)
            df = pd.concat([Y, X[column_list]], axis=1)

            plt.figure(figsize=(18, 18))
            if objective = 'regression':
                sns.pairplot(df)
            elif objective = 'multiclass':
                obj_column = Y.columns
                sns.pairplot(df, hue=obj_column, diag_kind='kde')
            else:
                print('Warning: argument \'objective\' must be \'regression\' or \'multiclass\'')
            if file_name != None:
                pdf.savefig()
            else:
                plt.show()

        if file_name != None:
            pdf.close()


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


class validation:
    def plot_score(model, X_train, Y_train, X_test, Y_test):
        Y_train_pred = model.predict(X_train)
        print('training accuracy: ', model.score(X_train, Y_train))
        print('training precision: ', precision_score(y_true=Y_train, y_pred=Y_train_pred))
        print('training recall: ', recall_score(y_true=Y_train, y_pred=Y_train_pred))
        print('training f1 score: ', f1_score(y_true=Y_train, y_pred=Y_train_pred))

        Y_test_pred = model.predict(X_test)
        print('\ntest accuracy: ', model.score(X_test, Y_test))
        print('test precision: ', precision_score(y_true=Y_test, y_pred=Y_test_pred))
        print('test recall: ', recall_score(y_true=Y_test, y_pred=Y_test_pred))
        print('test f1 score: ', f1_score(y_true=Y_test, y_pred=Y_test_pred))


    def plot_roc_curve(model, X_test, Y_test):
        fpr, tpr, th = roc_curve(Y_test, model.predict(X_test))
        auc_score = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc_score, color='orangered')
        plt.plot([0,1], [0,1], label='random', linestyle='--', color='limegreen')
        plt.plot([0,0,1], [0,1,1], label='ideal', linestyle='--', color='dodgerblue')
        
        plt.title('ROC curve')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.legend()
        plt.tight_layout()
  

    def plot_CAP_curve(model, X_test, Y_test):
        Y_pred = model.predict_proba(X_test)[:, 1]
        df = pd.DataFrame([Y_test, Y_pred]).T
        df.columns = ['Y_test', 'Y_pred']

        df = df.sort_values(by='Y_pred', ascending=False)

        rsv_cnt = df['Y_test'].sum()
        num = len(df)

        rsv_rate = rsv_cnt / num

        ideal = pd.DataFrame({'x':[0, rsv_rate, 1], 'y':[0, 1, 1]})
        x = np.arange(num) / (num - 1)
        y = df['Y_test'].cumsum() / rsv_cnt

        plt.plot(x, y, label='CAP curve', color='orangered')
        plt.plot(ideal['x'],ideal['y'], label='ideal', linestyle='--', color='dodgerblue')
        plt.plot(x, x, label='random', linestyle='--', color='limegreen')

        plt.title('CAP curve')
        plt.xlabel('rank of score')
        plt.ylabel('reservation probability')
        plt.legend()
        plt.tight_layout()

        plt.show()


from sklearn.utils import shuffle

    def plot_learning_curve(model, X_train, Y_train):
        X_shuf, Y_shuf = shuffle(X_train, Y_train)

        lr = LogisticRegression(penalty='l1', C=0.003)
        train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=X_shuf, y=Y_shuf, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', markersize=5, label='validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.grid()
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()


    def create_calibration_plot(model, X, Y, fine_name=None):
        """ create a calibration plot.
        
        Parameters
        ----------
        model:sklearn.estimator
            learned estimator.
        X: pandas.DataFrame, shape=[n_sample, n_features]
            explanatory variable vectors, where n_sample is the number of samples and n_features is the number of features.
        Y: pandas.DataFrame, shape=[n_sample, 1]
            objective variable vectors (binary).
        file_name: string
            file path to save a figure.
        ----------
        
        """
        
        fig, ax1 = plt.subplots()
        
        df = pd.DataFrame()
        df['Y_true'] = Y
        pred_prob = model.predict_proba(X)
        df['Y_pred'] = pred_prob[:, 1]
        df['bin'] = pd.cut(df['Y_pred'], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # plot histogram
        hist = pd.DataFrame({'flg = 0': df[df['Y_true']==0].groupby('bin').count()['Y_true'],
                            'flg = 1': df[df['Y_true']==1].groupby('bin').count()['Y_true']})
        hist.plot(ax=ax1, kind='bar', stacked=True, color=['skyblue', 'orangered'])
        
        plt.xlabel('predicted probability')
        plt.ylabel('count')
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.grid(False)
        
        # plot line graph
        ax2 = ax1.twinx()
        
        rsv_prob = pd.DataFrame({'prob': df[df['Y_true']==1].groupby('bin').count()['Y_true'] / df.groupby('bin').count()['Y_true']})
        rsv_prob.plot(ax=ax2, kind='line', marker = 'o', color='limegreen', secondary_y=True, mark_right=False, legend=False)
        
        plt.ylabel('actual probability')
        plt.legend(bbox_to_anchor=(1.1, 0.9), loc='upper left', borderaxespad=0)
        plt.grid(False)
        
        if file_name != None:
            plt.savefig(file_name)
        else:
            plt.show()