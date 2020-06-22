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
    f = interp1d(dfn['galactic year'], dfn[col], kind='linear', 
                 bounds_error = False, assume_sorted = True, fill_value = (dfn.iloc[0,1], dfn.iloc[-1,1]))
    y2 = f(df['galactic year'])
#    spl = splrep(dfn['galactic year'], dfn[col], k = 2, s = 10)
#    y2 = splev(df['galactic year'], spl)
    return y2

#@numba.njit   
def func(x, a, b):     
    return a * x + b

def func_pow(x, a, b):     
    return 1 / (a + np.exp(-b * x))

def exp2(x, a, b, c):     
    return a + b * x**c


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

def filter_iqr_dist(df, par):
    df_log = df[par].dropna()
    p125, p25, p50, p75, p875 = df_log.quantile([0.125, 0.25, 0.5, 0.75, 0.875])
    iqr = p75 - p25

    QS = ((p75 - p50) - (p50 - p25)) / (p75 - p25)
    OS = ((p875 - p50) - (p50 - p125)) / (p875 - p125)
    mc = (3 * QS + 2 * OS) / 5
    if mc >= 0:
        l = p25 - 2 * np.exp(-3.5 * mc) * iqr
        u = p75 + 2 * np.exp(4 * mc) * iqr
    else:
        l = p25 - 2 * np.exp(-4 * mc) * iqr
        u = p75 + 2 * np.exp(3.5 * mc) * iqr
    isn1 = df[df[par] <= l].index
    df.loc[isn1, par] = np.nan
    isn2 = df[df[par] >= u].index
    df.loc[isn2, par] = np.nan
    if len(list(isn1)) > 0 or len(list(isn2)) > 0:
        filter_iqr(df, par)
    return df

def filter_iqr(df, par):
    p25 = df[par].quantile(0.25)
    p75 = df[par].quantile(0.75)
    iqr = p75 - p25
    df.loc[(df[par] > p75 + 1.5 * iqr) | (df[par] < p25 - 1.5 * iqr), par] = np.nan
    return df

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

def valid_ind(df):
    df_train = df[~df['galactic year'].isin([1007012,1008016,1009020,1010025,1011030,1012036,1013042,1014049,1015056])]
    df_test = df[df['galactic year'].isin([1007012,1008016,1009020,1010025,1011030,1012036,1013042,1014049,1015056])]
    
    df_test, df_dop = train_test_split(df_test, test_size=0.3, random_state = 0)
    
    df_train = df_train.append(df_dop[df_dop['galactic year'] < 1015056])
    df_test = df_test.append(df_dop[df_dop['galactic year'] == 1015056])
    return (df_train.index), (df_test.index)

def fill_by_RMSE(dft, df_stats_gal, par = 'y_cor', R2 = 0.97, p_val = 0.5, RMSE = 0.01, n = 3):
    '''Выбираем наиболее похожие и создаем словарь галактика: похожие галактики'''
    df_stats_gal_f = df_stats_gal[(df_stats_gal['R2'] > R2) & (df_stats_gal['p_value'] > p_val) & (df_stats_gal['RMSE'] < RMSE)]
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
    
    '''Отбираем только те, где более 3 похожих'''
    key_to_del = []
    for key in galaxy_comb:
        if len(galaxy_comb[key]) < n:
            key_to_del.append(key)
    for key in key_to_del:
        galaxy_comb.pop(key, None)
    
    '''Заполняем средневзвешенными по RMSE параметрами на основе словаря по переменной fact'''
    print(dft[par].isnull().sum())
    for galaxy in galaxy_comb:
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
        isn = isn[(isn.isnull()) & (data.iloc[:, 0] >=1011030)].index
        dft.loc[isn, par] = data.loc[isn, 'mean']
    print(dft[par].isnull().sum())
    return dft

df = pd.read_csv('train.csv')
dft = pd.read_csv('test.csv')

df = df[df['galaxy'] != 'NGC 5253']

test_years = dft['galactic year'].value_counts()
dft['nulls'] = dft.isnull().sum(axis = 1)
gr_test = dft.groupby('galactic year')['nulls'].mean()
dft = dft.drop(['nulls'], axis = 1)

df['nulls'] = df.isnull().sum(axis = 1)
gr_train = df.groupby('galactic year')['nulls'].mean()
df = df.drop(['nulls'], axis = 1)

gr2 = group_stat(df, ['galaxy'], 'y')

gr = group_stat(df, ['galactic year'], 'y')

df_galaxy = df['galaxy'].value_counts()
galaxys = list(df_galaxy.index)
cols = list(df.columns)
df_years = df['galactic year'].value_counts()
years = list(df_years.index)
years.sort()
    
dft['y'] = 0
dft = dft.append(df, sort=False).reset_index(drop=True)

dft.loc[dft['y'] == 0, 'y'] = np.nan
gr = dft.groupby('galaxy').mean()

dft['y'] = np.log(dft['y'])


dft['Population using at least basic drinking-water services (%)'] = dft['Population using at least basic drinking-water services (%)'] - 15
dft.loc[(dft['Population using at least basic drinking-water services (%)'] > 100), 'Population using at least basic drinking-water services (%)'] = 100


dft.loc[dft['Gross income per capita'] <= 0, 'Gross income per capita'] = np.nan
dft['Gross income per capita'] = np.log(dft['Gross income per capita'])

dft['Population using at least basic sanitation services (%)'] = \
dft['Population using at least basic sanitation services (%)'].max() - \
dft['Population using at least basic sanitation services (%)'] 


list_log = ['Population using at least basic sanitation services (%)',
            'Mortality rate, under-five (per 1,000 live births)',
            'Mortality rate, infant (per 1,000 live births)',
            'Old age dependency ratio (old age (65 and older) per 100 creatures (ages 15-64))',
            'Adolescent birth rate (births per 1,000 female creatures ages 15-19)',
            'Unemployment, total (% of labour force)',
            'Unemployment, youth (% ages 15–24)',
            'Mortality rate, female grown up (per 1,000 people)',
            'Mortality rate, male grown up (per 1,000 people)',
            'Infants lacking immunization, red hot disease (% of one-galactic year-olds)',
            'Infants lacking immunization, Combination Vaccine (% of one-galactic year-olds)',
            'Gross galactic product (GGP) per capita',
            'Natural resource depletion',
            'Maternal mortality ratio (deaths per 100,000 live births)',
            'Estimated gross galactic income per capita, male',
            'Estimated gross galactic income per capita, female',
            'Domestic credit provided by financial sector (% of GGP)',
            'Remittances, inflows (% of GGP)']

for par in list_log:
    dft.loc[dft[par] < 0, par] = np.nan
dft[list_log] = np.log(dft[list_log] + 0.1)


dft.loc[(dft['galactic year'] < 1000000), 'Intergalactic Development Index (IDI), Rank'] = \
dft.loc[(dft['galactic year'] < 1000000), 'Intergalactic Development Index (IDI), Rank'] + 15


'''Приведение всех параметров к константам (по среднему) по годам'''
for col in cols[2:-1]:
    dft = korrect_y(dft, col)
#j2 = 0
#i2 = 0
#for par in cols[2:]:
#    data = dft.copy()
#    data = dft.dropna(subset = [par])
#    p1 = data.loc[(data['galactic year'] < 1011030) & data['y'] > 0, par]
#    p2 = data.loc[(data['galactic year'] >= 1011030) & data['y'] > 0, par]
#    #p3 = data.loc[data['y'] == 0, par]
#    if i2 % 8 == 0:
#        fig, ax = plt.subplots(figsize=(40, 20), nrows = 2, ncols = 4)
#        j1 = 0
#        j2 = 0
#    if j2 % 4 == 0 and i2 % 8 != 0:
#        j1 = 1
#        j2 = 0
#    sns.distplot(p1, hist = True, fit=norm, kde = False, label = 'before 1011030', ax=ax[j1, j2])
#    sns.distplot(p2, hist = True, fit=norm, kde = False, label = 'after 1011030', ax=ax[j1, j2])
##    if par != 'y':
##        sns.distplot(p3, hist = True, fit=norm, kde = False, label = 'y = 0', ax=ax[j1, j2])
#    i2 += 1
#    j2 += 1
#raise Exception

dft['Population, total calc'] = dft['Population, ages 15–64 (millions)'] + dft['Population, ages 65 and older (millions)'] + \
dft['Population, ages 15–64 (millions)'] * dft['Young age (0-14) dependency ratio (per 100 creatures ages 15-64)'] / 100

dft['total dependency ratio calc'] = dft['Population, ages 15–64 (millions)'] / \
(dft['Population, total calc'] - dft['Population, ages 15–64 (millions)'])

dft['Gross galactic product (GGP) per capita calc'] = dft['Gross galactic product (GGP), total'] / \
dft['Population, total calc']

dft['HDI'] = (dft['existence expectancy index'] * dft['Income Index'] * dft['Education Index'])**(1/3)

dft['Life expectancy index'] = (dft['existence expectancy at birth'] - 20) / (85 - 20)
data = dft[['Life expectancy index', 'existence expectancy index', 'existence expectancy at birth']]

'''Параметры отношения показателей по полам'''

dft['Expected years of education ratio']  = dft['Expected years of education, male (galactic years)'] /\
                                            dft['Expected years of education, female (galactic years)']
dft['Population with secondary education ratio']  = dft['Population with at least some secondary education, male (% ages 25 and older)'] /\
                                                    dft['Population with at least some secondary education, female (% ages 25 and older)']
dft['Intergalactic Development Index ratio']  = dft['Intergalactic Development Index (IDI), male'] /\
                                                dft['Intergalactic Development Index (IDI), female']
dft['Estimated gross galactic income per capita ratio']  = dft['Estimated gross galactic income per capita, male'] /\
                                                           dft['Estimated gross galactic income per capita, female']
dft['Intergalactic Development Index ratio, Rank']  = dft['Intergalactic Development Index (IDI), male, Rank'] /\
                                                      dft['Intergalactic Development Index (IDI), female, Rank']
dft['Labour force participation rate (% ages 15 and older) ratio']  = dft['Labour force participation rate (% ages 15 and older), male'] /\
                                                                      dft['Labour force participation rate (% ages 15 and older), female']
dft['Life expectancy at birth ratio']  = dft['Life expectancy at birth, male (galactic years)'] /\
                                         dft['Life expectancy at birth, female (galactic years)']
dft['Mortality rate ratio']  = dft['Mortality rate, male grown up (per 1,000 people)'] /\
                               dft['Mortality rate, female grown up (per 1,000 people)']
dft['Expected to mean ratio']  = dft['Expected years of education (galactic years)'] / dft['Mean years of education (galactic years)']

dft['Tourists per citizen'] = dft['Intergalactic inbound tourists (thousands)'] / dft['Population, total calc'] / 1000
dft['Estimated gross galactic income per capita mean'] = (dft['Estimated gross galactic income per capita, female'] + dft['Estimated gross galactic income per capita, male']) / 2
dft['Expected years of education (galactic years) mean'] = (dft['Expected years of education, female (galactic years)'] + dft['Expected years of education, male (galactic years)']) / 2
dft['Intergalactic Development Index (IDI) mean'] = (dft['Intergalactic Development Index (IDI), female'] + dft['Intergalactic Development Index (IDI), male']) / 2
dft['Intergalactic Development Index (IDI) mean, Rank'] = (dft['Intergalactic Development Index (IDI), female, Rank'] + dft['Intergalactic Development Index (IDI), male, Rank']) / 2
dft['Labour force participation rate (% ages 15 and older) mean'] = (dft['Labour force participation rate (% ages 15 and older), female'] + dft['Labour force participation rate (% ages 15 and older), male']) / 2
dft['Mean years of education (galactic years) mean'] = (dft['Mean years of education, female (galactic years)'] + dft['Mean years of education, male (galactic years)']) / 2
dft['Mortality rate mean'] = (dft['Mortality rate, male grown up (per 1,000 people)'] + dft['Mortality rate, female grown up (per 1,000 people)']) / 2
dft['Population with at least some secondary education mean'] = (dft['Population with at least some secondary education, female (% ages 25 and older)'] + dft['Population with at least some secondary education, male (% ages 25 and older)']) / 2
dft['Private galaxy capital flows (% of GGP) per capita'] = dft['Private galaxy capital flows (% of GGP)'] / dft['Population, total calc']


cols_drop = ['Remittances, inflows (% of GGP)', 'Population, total (millions)', 'Population, total calc', 
             'Population, ages 65 and older (millions)', 'Intergalactic inbound tourists (thousands)',
             'Population, ages 15–64 (millions)', 'Population, under age 5 (millions)', 'existence expectancy at birth',
             'Estimated gross galactic income per capita, female', 'Estimated gross galactic income per capita, male',
             'Expected years of education, female (galactic years)', 'Expected years of education, male (galactic years)',
             'Intergalactic Development Index (IDI), female', 'Intergalactic Development Index (IDI), male',
             'Intergalactic Development Index (IDI), female, Rank', 'Intergalactic Development Index (IDI), male, Rank',
             'Mean years of education, female (galactic years)', 'Mean years of education, male (galactic years)',
             'Mortality rate, male grown up (per 1,000 people)', 'Mortality rate, female grown up (per 1,000 people)',
             'Population with at least some secondary education, female (% ages 25 and older)', 
             'Population with at least some secondary education, male (% ages 25 and older)',
             'Labour force participation rate (% ages 15 and older), female', 'Labour force participation rate (% ages 15 and older), male',
             'Outer Galaxies direct investment, net inflows (% of GGP)', 'Jungle area (% of total land area)',
             'Gross galactic product (GGP), total', 'Gross capital formation (% of GGP)', 'Total unemployment rate (female to male ratio)',
             'Gross fixed capital formation (% of GGP)','Gross enrolment ratio, primary (% of primary under-age population)',
             'Renewable energy consumption (% of total final energy consumption)', 'Private galaxy capital flows (% of GGP)']
dft = dft.drop(cols_drop, axis = 1)

dft = movecol(dft, cols_to_move=['y'], ref_col='Private galaxy capital flows (% of GGP) per capita', place='After')
cols = list(dft.columns)
dft_galaxy = group_stat(dft, ['galaxy'], 'y').sort_values(by = 'Par_mean')
galaxys = dft_galaxy['galaxy'].tolist()

scalerx = MinMaxScaler(feature_range = (1, 2))
scalery = MinMaxScaler(feature_range = (1, 2))

dft['galactic year sc'] = scalerx.fit_transform(dft['galactic year'].values.reshape(-1, 1))
dft['y_start'] = 0
dft['y_din'] = 0

for galaxy in galaxys:
    dfg = dft.loc[dft['galaxy'] == galaxy] 
    dfg = dfg.dropna(subset = ['y'])
    gy = dfg[dfg['galactic year'] < 1011030]['galactic year sc'].values
    y_din = dfg[dfg['galactic year'] < 1011030]['y'].values
    popt, pcov = curve_fit(func, gy, y_din, maxfev = 100000)    
    dft.loc[dft['galaxy'] == galaxy, 'y_din'] = popt[0]    
    dft.loc[dft['galaxy'] == galaxy, 'y_start'] = popt[1]
dft = dft.drop('galactic year sc', axis = 1)

i2 = 0
model = CatBoostRegressor(iterations = 10000,learning_rate = 0.05, depth = 4,
                                              l2_leaf_reg = 0.1, custom_metric = 'RMSE', eval_metric = 'RMSE', verbose = False)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(4)
dft = dft.sort_values(by = 'galactic year')
start = time.monotonic()
n = 4
for par in cols[2:-1]:
    print(par, 'before', np.corrcoef(dft.loc[(~dft['y'].isnull()) & (~dft[par].isnull()), par], dft.loc[(~dft['y'].isnull()) & (~dft[par].isnull()), 'y'])[0, 1])
#    fig, ax = plt.subplots(figsize=(20, 20))
#    sns.distplot(dft.loc[~dft[par].isnull(), par], hist = True, fit=norm, kde = False, ax=ax)
    list_x = ['galactic year', 'Par1', 'y_start', 'y_din']
    dft['Par1'] = np.random.randint(0, 10000, dft.shape[0])
    data = dft
    data = data.dropna(subset = [par])
    x = poly.fit_transform(dft[list_x])
    y = dft[par].copy()

    xa_sc = scalerx.fit_transform(x)
    x_sc = scalerx.transform(poly.transform(data[list_x]))
    y_sc = scalery.fit_transform(data[par].values.reshape(-1, 1))
    y_sc_r = y_sc.ravel()
    
    y_pred = np.zeros(xa_sc.shape[0])
    cv = KFold(n_splits=n, shuffle=True)
    splits = cv.split(x_sc)
    for train_fold_index, predict_fold_index in splits:
        eval_pool = Pool(x_sc[predict_fold_index], y_sc_r[predict_fold_index])
        model.fit(x_sc[train_fold_index], y_sc_r[train_fold_index], eval_set=eval_pool, early_stopping_rounds=50)
        y_pred += scalery.inverse_transform(model.predict(xa_sc).reshape(-1, 1)).ravel()
   
    
    y_pred = y_pred / n
    
    
    dft[par] = y_pred
    dft['par'] = y
    
    for galaxy in galaxys:
        spl = splrep(dft[dft['galaxy'] == galaxy]['galactic year'],dft[dft['galaxy'] == galaxy][par], k = 3, s = 150)
        y_pred = splev(dft[dft['galaxy'] == galaxy]['galactic year'], spl)
        dft.loc[dft['galaxy'] == galaxy, par] = y_pred 
   
#    sns.distplot(dft[par], hist = True, fit=norm, kde = False, ax=ax)
#    for galaxy in galaxys:
#        if i2 % 8 == 0:
#            fig, ax = plt.subplots(figsize=(40, 20), nrows = 2, ncols = 4)
#            j1 = 0
#            j2 = 0
#        if j2 % 4 == 0 and i2 % 8 != 0:
#            j1 = 1
#            j2 = 0
#        spl = splrep(dft[dft['galaxy'] == galaxy]['galactic year'],dft[dft['galaxy'] == galaxy][par], k = 3, s = 150)
#        y_pred = splev(dft[dft['galaxy'] == galaxy]['galactic year'], spl)
#         
#        
#        sns.regplot(dft[dft['galaxy'] == galaxy]['galactic year'], dft[dft['galaxy'] == galaxy]['par'], scatter_kws={"s": 80}, ci = None, ax = ax[j1, j2])
#        sns.regplot(dft[dft['galaxy'] == galaxy]['galactic year'], dft[dft['galaxy'] == galaxy][par], scatter_kws={"s": 80}, ci = None, ax = ax[j1, j2])
#        sns.lineplot(dft[dft['galaxy'] == galaxy]['galactic year'], y_pred, ax = ax[j1, j2])
#        ax[j1, j2].set_ylabel(par)
#        ax[j1, j2].set_title(galaxy)
#        
#        dft.loc[dft['galaxy'] == galaxy, par] = y_pred
#        i2 += 1
#        j2 += 1
    print(par, 'after', np.corrcoef(dft.loc[(~dft['y'].isnull()) & (~dft[par].isnull()), par], dft.loc[(~dft['y'].isnull()) & (~dft[par].isnull()), 'y'])[0, 1])
    #raise Exception
end = time.monotonic()
print(end - start) 
dft = dft.sort_index() 

dft = dft.drop(['par', 'Par1', 'y_start', 'y_din'], axis = 1)

dft.to_csv('dft_new.csv', index=False)

dft = pd.read_csv('dft_new.csv')
 
'''Добавление бинарного признака переломного года'''
dft['break_year'] = dft['galactic year'].apply(lambda x: 0 if x < 1011030 else 1)
dft = movecol(dft, cols_to_move=['y'], ref_col='break_year', place='After')
cor_all = dft.corr()

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
    model = CatBoostRegressor(iterations = 15000,learning_rate = 0.01, depth = 4,
                                              l2_leaf_reg = 0.1, bagging_temperature = 1, 
                                              custom_metric = 'RMSE', eval_metric = 'RMSE', verbose = False)
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
outputCB[['Index','pred','opt_pred']].to_csv('new_250_iter.csv', index=False)



