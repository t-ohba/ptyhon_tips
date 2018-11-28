import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def view_variable_correlation_heatmap(df):
    plt.figure(figsize=(18, 18))
    sns.heatmap(df.corr(), square=True, cmap=plt.cm.viridis, linecolor='white', annot=True)

def view_variable_scatter_plot(df):
    plt.figure(figsize=(18, 18))
    sns.pairplot(df, hue='obj_rsv_flg', diag_kind='kde')