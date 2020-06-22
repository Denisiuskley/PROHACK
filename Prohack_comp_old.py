try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from scipy.stats import ttest_ind
from itertools import combinations
from sklearn.ensemble import IsolationForest
from catboost import CatBoostRegressor, Pool
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy.interpolate import interp1d, splrep, splev 
from scipy.optimize import curve_fit
import time
import numba
from sklearn.svm import SVR
from tqdm import tqdm
from sklearn.base import clone

def percentil5 (x):
    x = x.dropna()
    if x.shape[0] > 0:
        return np.round(np.percentile(x, q=5), 3)
    else: return 0
def percentil95 (x): 
    x = x.dropna()
    if x.shape[0] > 0:
        return np.round(np.percentile(x, q=95), 3)
    else: return 0

def group_stat(df, group, for_stat):
    gr = df.groupby(group).agg(
        Par_min = (for_stat, 'min'),
        Par_quantil1 = (for_stat, percentil5),
        Par_median = (for_stat, 'median'),
        Par_mean = (for_stat, 'mean'),
        Par_quantil3 = (for_stat, percentil95),
        Par_max = (for_stat, 'max'),
        Par_sum = (for_stat, 'sum'),
        Par_count = (for_stat, 'count')).reset_index()
    return gr

def catBoost_rec(df, params, rec):
    model = CatBoostRegressor(iterations = 10000, learning_rate = 0.1, depth = 4,
                              custom_metric = 'R2', eval_metric = 'R2', verbose = 200)
    dfm = df[params+[rec]]
    dfm = dfm.dropna()
    X = dfm.drop([rec], axis = 1)
    y = dfm[rec]    
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.15)
    eval_pool = Pool(X_eval, y_eval, cat_features = [1])
    model.fit(X_train, y_train, cat_features = [1], eval_set=eval_pool, early_stopping_rounds=200)
    y_pred = model.predict(df[params])
    return y_pred

def param_int(df, col):
    dfn = df[['galactic year', col]]
    dfn = dfn.dropna()   
    f = interp1d(dfn['galactic year'], dfn[col], kind='nearest', 
                 bounds_error = False, fill_value = (dfn.iloc[0,1], dfn.iloc[-1,1]))
    return f(df['galactic year'])

#@numba.njit   
def func(df, a, b):     
    return a * df + b

def func_pow(x, a, b):     
    return a / (1 + np.exp(-b * x))

def korrect_y(df, par):
    gr = group_stat(df, ['galactic year'], par)
    y = gr['Par_mean'].median()
    for i in range(gr.shape[0]):
        if gr.loc[i, 'Par_mean'] == 0:
            koef = 1
        else:
            koef = y / gr.loc[i, 'Par_mean']
        df.loc[df['galactic year'] == gr.iloc[i, 0], par] =\
        df.loc[df['galactic year'] == gr.iloc[i, 0], par] * koef
    return df
def filter_iqr(df, par):
    p25 = df[par].quantile(0.25)
    p75 = df[par].quantile(0.75)
    iqr = p75 - p25
    df.loc[(df[par] > p75 + 1.5 * iqr) | (df[par] < p25 - 1.5 * iqr), par] = np.nan
    return df
def compute_meta_feature(model, X_train, X_test, y_train, cv, log = 0):
    X_meta_train = np.zeros((len(y_train), 1), dtype=np.float32)
    splits = cv.split(X_train)
    for train_fold_index, predict_fold_index in splits:
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]
        
        folded_clf = clone(model)
        folded_clf.fit(X_fold_train, y_fold_train, cat_features = [1])
        
        X_meta_train[predict_fold_index] = folded_clf.predict(X_fold_predict).reshape(-1, 1)
    
    meta_clf = clone(model)
    meta_clf.fit(X_train, y_train, cat_features = [1])
    
    X_meta_test = meta_clf.predict(X_test).reshape(-1, 1)
    if log == 1:
        X_meta_train = np.exp(X_meta_train)
        X_meta_test = np.exp(X_meta_test)
    return X_meta_train, X_meta_test

def generate_meta_features(models, X_train, X_test, y_train, cv):
   
    features = [
        compute_meta_feature(model, X_train, X_test, y_train, cv)
        for model in tqdm(models)
    ]
    stacked_features_train = np.hstack([
        features_train for features_train, features_test in features
    ])

    stacked_features_test = np.hstack([
        features_test for features_train, features_test in features
    ])
    
    return stacked_features_train, stacked_features_test

def compute_metric(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def movecol(df, cols_to_move=[], ref_col='', place='After'):
    
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])


df = pd.read_csv('train.csv')
dft = pd.read_csv('test.csv')
test_years = dft['galactic year'].value_counts()
dft['nulls'] = dft.isnull().sum(axis = 1)
gr_test = dft.groupby('galactic year')['nulls'].mean()
dft = dft.drop(['nulls'], axis = 1)

df['nulls'] = df.isnull().sum(axis = 1)
gr_train = df.groupby('galactic year')['nulls'].mean()
df = df.drop(['nulls'], axis = 1)

gr2 = group_stat(df, ['galaxy'], 'y')

'''Поищем похожие галактики по целевой переменной'''
df_galaxy = df['galaxy'].value_counts()
galaxys = list(df_galaxy.index)
df_years = df['galactic year'].value_counts()
years = list(df_years.index)
years.sort()
combins = list(combinations(galaxys, 2))
df_stats_gal = pd.DataFrame(columns = ['galaxy 1', 'galaxy 2', 'RMSE', 'R2', 'p_value'])
i2 = 0
for comb in combins:
    break_year = pd.merge(df.loc[df['galaxy'] == comb[0], ['galactic year', 'y']],
                          df.loc[df['galaxy'] == comb[1], ['galactic year', 'y']], on=['galactic year'], how='left')
    break_year = break_year.dropna()
    if break_year.shape[0] < 5:
        continue
    temp = pd.DataFrame({'galaxy 1': comb[0], 'galaxy 2': comb[1],
                         'RMSE': metrics.mean_squared_error(break_year.iloc[:, 1], break_year.iloc[:, 2], squared=False),
                         'R2': np.corrcoef(break_year.iloc[:, 1], break_year.iloc[:, 2])[0,1],
                         'p_value': ttest_ind(break_year.iloc[:, 1], break_year.iloc[:, 2]).pvalue}, index = [i2])
    df_stats_gal = df_stats_gal.append(temp, sort=False)
    i2 += 1

'''Выбираем наиболее похожие и создаем словарь галактика: похожие галактики'''
df_stats_gal_f = df_stats_gal[(df_stats_gal['R2'] > 0.95) & (df_stats_gal['p_value'] > 0.5)]
galaxy_comb = dict()
for i in range(df_stats_gal_f.shape[0]):
    if df_stats_gal_f.iloc[i, 0] in galaxy_comb:
        galaxy_comb[df_stats_gal_f.iloc[i, 0]].append([df_stats_gal_f.iloc[i, 1], df_stats_gal_f.iloc[i, 2]])
    else:
        galaxy_comb[df_stats_gal_f.iloc[i, 0]] = [[df_stats_gal_f.iloc[i, 1], df_stats_gal_f.iloc[i, 2]]]

for i in range(df_stats_gal_f.shape[0]):
    if df_stats_gal_f.iloc[i, 1] in galaxy_comb:
        galaxy_comb[df_stats_gal_f.iloc[i, 1]].append([df_stats_gal_f.iloc[i, 0], df_stats_gal_f.iloc[i, 2]])
    else:
        galaxy_comb[df_stats_gal_f.iloc[i, 1]] = [[df_stats_gal_f.iloc[i, 0], df_stats_gal_f.iloc[i, 2]]]    

'''Отбираем только те, где более 4 похожих'''
key_to_del = []
for key in galaxy_comb:
    if len(galaxy_comb[key]) < 3:
        key_to_del.append(key)
for key in key_to_del:
    galaxy_comb.pop(key, None)
    

dft['y'] = 0
dft = dft.append(df, sort=False).reset_index(drop=True)
correlation = df.corr()

cols = list(dft.columns)
'''Создадим и заполним предсказания целевой переменной'''
dft['y_pred'] = np.nan
dft.loc[dft['y'] == 0, 'y'] = np.nan
par = 'y_pred'
for galaxy in galaxy_comb:    
    data = dft.loc[dft['galaxy'] == galaxy, ['galactic year', 'y']]
    ind = data.index
    sum_error = []
    i2 = 0
    for galaxy2 in galaxy_comb[galaxy]:
        data = pd.merge(data, dft.loc[dft['galaxy'] == galaxy2[0], ['galactic year', 'y']], 
                        on=['galactic year'], how='left', suffixes = (str(i2 * 2), str(i2 * 2 + 1)))
        data.iloc[:, -1] = data.iloc[:, -1] / galaxy2[1]
        sum_error.append(1 / galaxy2[1])
        i2 += 1
    data['mean'] = data.apply(lambda x: x[2:].sum() / np.sum(x[2:]/x[2:]*sum_error) , axis = 1)
    data.index = ind
    dft.loc[ind, par] = data.loc[ind, 'mean']
print(dft[par].isnull().sum())
fig, ax = plt.subplots(figsize=(10, 10))
sns.regplot(par, 'y', data = dft, scatter_kws={"s": 80}, ax = ax)
correlation2 = dft.corr()

'''Корректируем и отрезаем значения перед заполнением'''
dft.loc[(dft['Population, ages 15–64 (millions)'] > 1000) | (dft['Population, ages 15–64 (millions)'] < 0),
        'Population, ages 15–64 (millions)'] = np.nan
dft.loc[dft['Population, urban (%)'] < 30, 'Population, urban (%)'] = np.nan
dft.loc[dft['Gross income per capita'] < 5000, 'Gross income per capita'] = np.nan
dft.loc[(dft['galaxy'] == 'UGC 8651 (DDO 181)') & (dft['Population using at least basic sanitation services (%)'] < 50),\
        'Population using at least basic sanitation services (%)'] = np.nan
dft.loc[(dft['galaxy'] == 'Andromeda Galaxy (M31)') & (dft['Gross capital formation (% of GGP)'] > 35),\
        'Gross capital formation (% of GGP)'] = np.nan
dft.loc[(dft['galaxy'] == 'UGC 4483') & (dft['Gross capital formation (% of GGP)'] > 35),\
        'Gross capital formation (% of GGP)'] = np.nan
dft.loc[(dft['galaxy'] == 'UGGA 292') & (dft['Gross capital formation (% of GGP)'] > 35),\
        'Gross capital formation (% of GGP)'] = np.nan
dft.loc[(dft['Population, ages 65 and older (millions)'] > 150) | (dft['Population, ages 65 and older (millions)'] < 0),
        'Population, ages 65 and older (millions)'] = np.nan
dft.loc[(dft['Population, under age 5 (millions)'] > 150) | (dft['Population, under age 5 (millions)'] < 0),
        'Population, under age 5 (millions)'] = np.nan
dft.loc[(dft['Total unemployment rate (female to male ratio)'] > 12) | (dft['Total unemployment rate (female to male ratio)'] < 0),
        'Total unemployment rate (female to male ratio)'] = np.nan
dft.loc[(dft['Youth unemployment rate (female to male ratio)'] > 10) | (dft['Youth unemployment rate (female to male ratio)'] < 0),
        'Youth unemployment rate (female to male ratio)'] = np.nan
dft.loc[(dft['Outer Galaxies direct investment, net inflows (% of GGP)'] > 50) | (dft['Outer Galaxies direct investment, net inflows (% of GGP)'] < 0),
        'Outer Galaxies direct investment, net inflows (% of GGP)'] = np.nan
dft.loc[(dft['Expected years of education, male (galactic years)'] > 25) | (dft['Expected years of education, male (galactic years)'] < 0),
        'Expected years of education, male (galactic years)'] = np.nan
dft.loc[(dft['Expected years of education, female (galactic years)'] > 25) | (dft['Expected years of education, female (galactic years)'] < 0),
        'Expected years of education, female (galactic years)'] = np.nan
dft.loc[(dft['Domestic credit provided by financial sector (% of GGP)'] > 350) | (dft['Domestic credit provided by financial sector (% of GGP)'] < 0),
        'Domestic credit provided by financial sector (% of GGP)'] = np.nan
dft.loc[(dft['Intergalactic inbound tourists (thousands)'] > 200000) | (dft['Intergalactic inbound tourists (thousands)'] < 0),
        'Intergalactic inbound tourists (thousands)'] = np.nan
dft.loc[(dft['Private galaxy capital flows (% of GGP)'] > 100) | (dft['Private galaxy capital flows (% of GGP)'] < 0),
        'Private galaxy capital flows (% of GGP)'] = np.nan

'''Заполняем средневзвешенными по RMSE параметрами'''
print(dft.isnull().sum().sum())
for galaxy in galaxy_comb:
    for par in cols[2:79]:
        data = dft.loc[dft['galaxy'] == galaxy, ['galactic year', par]]
        ind = data.index
        sum_error = []
        i2 = 0
        for galaxy2 in galaxy_comb[galaxy]:
            data = pd.merge(data, dft.loc[dft['galaxy'] == galaxy2[0], ['galactic year', par]], 
                            on=['galactic year'], how='left', suffixes = (str(i2 * 2), str(i2 * 2 + 1)))
            data.iloc[:, -1] = data.iloc[:, -1] / galaxy2[1]
            sum_error.append(1 / galaxy2[1])
            i2 += 1
        data['mean'] = data.apply(lambda x: x[2:].sum() / np.sum(x[2:]/x[2:]*sum_error) , axis = 1)
        data.index = ind
        isn = data.iloc[:, 1]
        isn = isn[isn.isnull()].index
        dft.loc[isn, par] = data.loc[isn, 'mean']
        
print(dft.isnull().sum().sum())


cols = list(dft.columns)

dft_galaxy = group_stat(dft[~dft['y'].isnull()], ['galaxy'], 'y').sort_values(by = 'Par_mean')
galaxys = dft_galaxy['galaxy'].tolist()

#dft.loc[dft['Population using at least basic drinking-water services (%)'] > 90, 'Population using at least basic drinking-water services (%)'] =\
#np.abs(100 - dft.loc[dft['Population using at least basic drinking-water services (%)'] > 90, 'Population using at least basic drinking-water services (%)'])

gr = group_stat(dft, ['galaxy'], 'Gross income per capita')
gr = gr[gr['Par_mean'] >= 100000]
dft.loc[dft['galaxy'].isin(gr['galaxy']), 'Gross income per capita'] =\
dft.loc[dft['galaxy'].isin(gr['galaxy']), 'Gross income per capita'] / 5

dft['existence expectancy at birth'] = dft['existence expectancy at birth']**8
dft['Expected years of education (galactic years)'] = dft['Expected years of education (galactic years)']**4
dft['Mean years of education (galactic years)'] = dft['Mean years of education (galactic years)']**3
dft['Intergalactic Development Index (IDI)'] = dft['Intergalactic Development Index (IDI)']**5
dft['Education Index'] = dft['Education Index']**4
dft['Intergalactic Development Index (IDI), Rank'] = dft['Intergalactic Development Index (IDI), Rank']**0.2
dft['Population using at least basic sanitation services (%)'] = dft['Population using at least basic sanitation services (%)']**4
dft['Life expectancy at birth, male (galactic years)'] = dft['Life expectancy at birth, male (galactic years)']**3
dft['Life expectancy at birth, female (galactic years)'] = dft['Life expectancy at birth, female (galactic years)']**3
dft['Mortality rate, under-five (per 1,000 live births)'] = 1 / dft['Mortality rate, under-five (per 1,000 live births)']
dft['Mortality rate, infant (per 1,000 live births)'] = 1 / dft['Mortality rate, infant (per 1,000 live births)']


dft['Maternal mortality ratio (deaths per 100,000 live births)'] = 1 / dft['Maternal mortality ratio (deaths per 100,000 live births)']
dft['Respiratory disease incidence (per 100,000 people)'] = 1 / dft['Respiratory disease incidence (per 100,000 people)']
dft['Gender Inequality Index (GII)'] = 1 / dft['Gender Inequality Index (GII)']


#par = cols[78]
#gr = group_stat(dft[dft['y'] != 0], ['galaxy'], par)
#gr2 = group_stat(dft[dft['y'] != 0], ['galaxy'], 'y')
#gro = pd.concat([gr['Par_mean'], np.log(gr2['Par_mean'])], axis = 1)
#gro.columns = [par, 'y']
#fig, ax = plt.subplots(figsize=(10, 10))
#sns.regplot(par, 'y', data = gro, scatter_kws={"s": 80}, ax = ax)
#raise Exception()
#
#scalerx = MinMaxScaler(feature_range = (1, 2))
#scalery = MinMaxScaler(feature_range = (1, 2))
#j2 = 0
#for i in range(0, 80, 10):    
#    if j2 % 4 == 0:
#        fig, ax = plt.subplots(figsize=(40, 20), nrows = 2, ncols = 4)
#        j2 = 0
#    data = dft[dft['galaxy'].isin(galaxys[i * 2 : (i + 1) * 2])]
#    data = data.dropna(subset = [par])
#    data[par] = data[par]
#    sns.lineplot(x="galactic year", y=par, hue="galaxy", data=data, ax = ax[0, j2])        
#    for galaxy in galaxys[i * 2 : (i + 1) * 2]:
#        x = data.loc[data['galaxy'] == galaxy, 'galactic year']
#        if x.shape[0] < 3:
#            continue
#        y = data.loc[data['galaxy'] == galaxy, par]
#        x_sc = scalerx.fit_transform(x.values.reshape(-1, 1))
#        y_sc = scalery.fit_transform(y.values.reshape(-1, 1))
#        x_sc_r = x_sc.ravel()
#        y_sc_r = y_sc.ravel()
#        popt, pcov = curve_fit(func_pow, x_sc_r, y_sc_r, maxfev = 100000)
#        y_pred = scalery.inverse_transform(func_pow(x_sc, *popt)).ravel()
#        sns.lineplot(data.loc[data['galaxy'] == galaxy, 'galactic year'], y=y_pred, data=data, ax = ax[0, j2])
#    ax[0, j2].set_ylabel(par)
#    sns.regplot(x="galactic year", y='y',
#                 data=dft[(dft['galaxy'].isin(galaxys[i * 2 : i * 2 + 1]))], ci = None, ax = ax[1, j2])
#    sns.regplot(x="galactic year", y='y',
#                 data=dft[(dft['galaxy'].isin(galaxys[(i + 1) * 2 : (i + 1) * 2 + 1]))], ci = None, ax = ax[1, j2])
#    j2 += 1
#
#raise Exception()    
'''Приведение всех параметров к константам (по среднему) по годам'''
for col in cols[2:-1]:
    dft = korrect_y(dft, col)

scalerx = MinMaxScaler(feature_range = (1,2))
scalery = MinMaxScaler(feature_range = (1,2))

start = time.monotonic()

dft['Tourists per citizen'] = dft['Intergalactic inbound tourists (thousands)'] / dft['Population, total (millions)'] / 1000
cols_drop = ['Remittances, inflows (% of GGP)', 'Population using at least basic drinking-water services (%)',
             'Population, total (millions)', 'Population, ages 65 and older (millions)', 'Intergalactic inbound tourists (thousands)']
dft = dft.drop(cols_drop, axis = 1)
dft = movecol(dft, cols_to_move=['y'], ref_col='Tourists per citizen', place='After')

cols = list(dft.columns)
for par in cols[2:-1]:
    new_col = pd.DataFrame(columns = ['galactic year', 'galaxy', 'par'])
    for galaxy in galaxys:
        data = dft[dft['galaxy'] == galaxy]
        data = data.dropna(subset = [par])
        x = dft.loc[dft['galaxy'] == galaxy, 'galactic year']
        y = dft.loc[dft['galaxy'] == galaxy, par]
        if data.shape[0] < 3:
            new_col = new_col.append(pd.DataFrame({'galactic year' : x, 'galaxy' : galaxy, 'par' :y}), ignore_index=True)
            continue
        
        x_sc = scalerx.fit_transform(data['galactic year'].values.reshape(-1, 1))
        xa_sc = scalerx.transform(x.values.reshape(-1, 1))
        y_sc = scalery.fit_transform(data[par].values.reshape(-1, 1))
        x_sc_r = x_sc.ravel()
        y_sc_r = y_sc.ravel()
        popt, pcov = curve_fit(func_pow, x_sc_r, y_sc_r, maxfev = 100000)
        y = scalery.inverse_transform(func_pow(xa_sc, *popt)).ravel()
        
        new_col = new_col.append(pd.DataFrame({'galactic year' : x, 'galaxy' : galaxy, 'par' : y}), ignore_index=True)
    new_col['galactic year'] = new_col['galactic year'].astype(int)
    dft = pd.merge(dft, new_col, on=['galaxy', 'galactic year'], how='left')
    dft[par] = dft['par']
    dft = dft.drop(['par'], axis = 1)
end = time.monotonic()
print(end - start)    
#dft.loc[(dft['Population, total (millions)'] > 1000) | (dft['Population, total (millions)'] < 0), 'Population, total (millions)'] = np.nan
raise Exception
'''Заполнение функцией'''
for galaxy in galaxys:
    dfg = dft.loc[dft['galaxy'] == galaxy]
    if dfg.shape[0] > 5:
        for col in cols[2:-1]:
            if dfg[col].isnull().sum() < dfg.shape[0] - 3 & dfg[col].isnull().sum() > 0:
                dft.loc[dft['galaxy'] == galaxy, col] = param_int(dfg, col)
##raise Exception()
#                
##good_years = ['995006', '1000000', '1005006', '1007012', '1008016', '1009020', '1010025', '1011030', '1012036', '1013042', '1014049', '1015056', '1016064']
##dft = dft[dft['galactic year'].isin(good_years)]
#
for col in cols[2:-1]:
    dft.loc[dft[col].isnull(), col] = dft.groupby('galaxy')[col].transform('mean')

dft = movecol(dft, cols_to_move=['y_pred'], ref_col='Tourists per citizen', place='After')
cols = list(dft.columns)
for col in cols[2:-2]:
    dft[col] = dft[col].fillna(0)

#raise Exception()
    
#dft['y_start'] = 0
#dft['y_din'] = 0
#for galaxy in galaxys:
#    dfg = dft.loc[dft['galaxy'] == galaxy].sort_values(by = 'galactic year')
#    y_din = dfg[dfg['galactic year'] < 1000000]['y'].tolist()     
#    dft.loc[dft['galaxy'] == galaxy, 'y_start'] = y_din[0]    
#    if y_din[-1] > y_din[0]:
#        dft.loc[dft['galaxy'] == galaxy, 'y_din'] = 1
#    elif y_din[-1] < y_din[0]:
#        dft.loc[dft['galaxy'] == galaxy, 'y_din'] = -1
        
cols_mf = [i for i in cols if 'male' in i or 'female' in i]

'''Параметры отношения показателей по полам'''

dft['Expected years of education ratio']  = dft['Expected years of education, male (galactic years)'] /\
                                            dft['Expected years of education, female (galactic years)']

dft['Population with secondary education ratio']  = dft['Population with at least some secondary education, male (% ages 25 and older)'] /\
                                                    dft['Population with at least some secondary education, female (% ages 25 and older)']

dft['Intergalactic Development Index ratio']  = dft['Intergalactic Development Index (IDI), male'] /\
                                                dft['Intergalactic Development Index (IDI), female']


dft['y'] = np.log(dft['y'])

dft = dft.drop('y_pred', axis = 1)

df = dft.loc[~dft['y'].isnull(), :]    
    
X = df.drop('y', axis = 1)
y = df['y']

kf = KFold(n_splits=5, shuffle = True, random_state = 0)
t = kf.split(X, y)
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns
i2 = 0
results = []
y_pred_m = np.zeros((y.shape[0]))
for train, test in kf.split(X, y):
    model = CatBoostRegressor(iterations = 20000,learning_rate = 0.01, depth = 4,
                                              l2_leaf_reg = 0.1, bagging_temperature = 1, 
                                              custom_metric = 'RMSE', eval_metric = 'RMSE', verbose = 1000)
    model.fit(X.iloc[train,:], y.iloc[train], cat_features = [1])
    y_pred = model.predict(X.iloc[test,:])
    y_pred = np.exp(y_pred)
    y_pred_m[test] = y_pred
    feature_importances[i2] = model.get_feature_importance(data=None,
       prettified=False, thread_count=-1, verbose=False)
    print('RMSE score:', np.round(metrics.mean_squared_error(np.exp(y.iloc[test]), y_pred)**0.5, 4))
    results.append(metrics.mean_squared_error(np.exp(y.iloc[test]), y_pred)**0.5)
    i2 += 1

fi_plot = pd.DataFrame()
for i in range(i2):
    temp = feature_importances[['feature',i]]
    temp.columns = ['feature', 'importance']    
    fi_plot = fi_plot.append(temp, sort=False)
feature_importances['mean'] = feature_importances.iloc[:,1:].mean(axis = 1)
fi_plot = pd.merge(fi_plot, feature_importances[['feature', 'mean']], 
                            on=['feature'], how='left')
plt.figure(figsize=(16, 16))
sns.barplot(data=fi_plot.sort_values(by='mean', ascending=False), x='importance', y='feature', capsize=.2)

fig, ax = plt.subplots(figsize=(10, 10))
sns.regplot(y_pred_m, np.exp(y), scatter_kws={"s": 80}, ax = ax)

print(np.round(np.mean(results), 5), np.round(metrics.r2_score(np.exp(y), y_pred_m), 4))


df = dft.loc[~dft['y'].isnull(), :]
t_df = dft.loc[dft['y'].isnull(), :]

y_train = df['y']  
X_train = df.drop(['y'], axis = 1)

X_test = t_df.drop(['y'], axis = 1)

outputCB = pd.DataFrame({'Index': X_test.index})

models = [CatBoostRegressor(iterations = 10000,learning_rate = 0.01, depth = 4,
                          l2_leaf_reg = 0.1, custom_metric = 'RMSE', eval_metric = 'RMSE', verbose = 1000),
          CatBoostRegressor(iterations = 20000,learning_rate = 0.01, depth = 4,
                          l2_leaf_reg = 0.1, custom_metric = 'RMSE', eval_metric = 'RMSE', verbose = 1000),
          CatBoostRegressor(iterations = 15000,learning_rate = 0.01, depth = 4,
                          l2_leaf_reg = 1, custom_metric = 'RMSE', eval_metric = 'RMSE', verbose = 1000)]
cv = KFold(n_splits=5, shuffle=True, random_state=42)
stacked_features_train, stacked_features_test = generate_meta_features(models, X_train.values, X_test.values, y_train.values, cv)
clf = LinearRegression(normalize = True)
y_pred = compute_metric(clf, stacked_features_train, y_train, stacked_features_test)
       
outputCB['pred'] = np.exp(y_pred)

Potential_for_increase_in_the_Index = -np.log(outputCB['pred'].values + 0.01) + 3
Potential_for_increase_in_the_Index2 = Potential_for_increase_in_the_Index**2 / 1000

dfe = pd.read_csv('test.csv')
eei = dfe[['galactic year', 'galaxy', 'existence expectancy index']]

eeiv = eei['existence expectancy index'].values
eei_ind = np.where(eeiv < 0.7)[0]


def opt_f(x):
     return -np.sum(x * Potential_for_increase_in_the_Index2)
 
from scipy.optimize import minimize

cons = ({'type': 'ineq', 'fun': lambda x:  np.array([49999.99 - np.sum(x),
                                                   np.sum(x[eei_ind]) - np.sum(x) * 0.1 + 0.1])})
x0 = 50 * np.ones((eei.shape[0]))
bnds = tuple((0,100) for x in x0)

res = minimize(opt_f, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 250, 'ftol': 1e-9, 'disp': True})
xs = res.x
print(np.sum(xs), np.sum(xs[eei_ind]))
outputCB['opt_pred'] = xs
outputCB[['Index','pred','opt_pred']].to_csv('old_250_iter.csv', index=False)








