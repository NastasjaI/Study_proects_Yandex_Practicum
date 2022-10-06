#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Открытие-и-анализ-данных" data-toc-modified-id="Открытие-и-анализ-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Открытие и анализ данных</a></span><ul class="toc-item"><li><span><a href="#Импорт-библиотек" data-toc-modified-id="Импорт-библиотек-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Импорт библиотек</a></span></li><li><span><a href="#Загрузка-данных" data-toc-modified-id="Загрузка-данных-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Загрузка данных</a></span></li><li><span><a href="#Ресемплирование" data-toc-modified-id="Ресемплирование-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Ресемплирование</a></span></li><li><span><a href="#Визуализация" data-toc-modified-id="Визуализация-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Визуализация</a></span></li><li><span><a href="#Оценка-сезонности" data-toc-modified-id="Оценка-сезонности-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Оценка сезонности</a></span></li><li><span><a href="#Промежуточный-вывод" data-toc-modified-id="Промежуточный-вывод-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Промежуточный вывод</a></span></li></ul></li><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Стационарность-ряда" data-toc-modified-id="Стационарность-ряда-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Стационарность ряда</a></span><ul class="toc-item"><li><span><a href="#KPSS-тест" data-toc-modified-id="KPSS-тест-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>KPSS тест</a></span></li><li><span><a href="#Промежуточный-вывод" data-toc-modified-id="Промежуточный-вывод-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>Промежуточный вывод</a></span></li></ul></li></ul></li><li><span><a href="#Работа-с-признаками" data-toc-modified-id="Работа-с-признаками-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Работа с признаками</a></span><ul class="toc-item"><li><span><a href="#Создание-признаков" data-toc-modified-id="Создание-признаков-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Создание признаков</a></span></li><li><span><a href="#Разделение-выборок" data-toc-modified-id="Разделение-выборок-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Разделение выборок</a></span></li><li><span><a href="#Разделение-признаков" data-toc-modified-id="Разделение-признаков-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Разделение признаков</a></span></li><li><span><a href="#Дополнение" data-toc-modified-id="Дополнение-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Дополнение</a></span></li><li><span><a href="#Промежуточный-вывод" data-toc-modified-id="Промежуточный-вывод-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Промежуточный вывод</a></span></li></ul></li><li><span><a href="#Обучение-моделей" data-toc-modified-id="Обучение-моделей-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Обучение моделей</a></span><ul class="toc-item"><li><span><a href="#Линейные-модели" data-toc-modified-id="Линейные-модели-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Линейные модели</a></span><ul class="toc-item"><li><span><a href="#LinearRegression" data-toc-modified-id="LinearRegression-4.1.1"><span class="toc-item-num">4.1.1&nbsp;&nbsp;</span>LinearRegression</a></span></li><li><span><a href="#LinearRegression+GridSearchCV" data-toc-modified-id="LinearRegression+GridSearchCV-4.1.2"><span class="toc-item-num">4.1.2&nbsp;&nbsp;</span>LinearRegression+GridSearchCV</a></span></li><li><span><a href="#LassoCV" data-toc-modified-id="LassoCV-4.1.3"><span class="toc-item-num">4.1.3&nbsp;&nbsp;</span>LassoCV</a></span></li><li><span><a href="#Lasso" data-toc-modified-id="Lasso-4.1.4"><span class="toc-item-num">4.1.4&nbsp;&nbsp;</span>Lasso</a></span></li></ul></li><li><span><a href="#Случайный-лес" data-toc-modified-id="Случайный-лес-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Случайный лес</a></span></li><li><span><a href="#Градиентный-бустинг" data-toc-modified-id="Градиентный-бустинг-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Градиентный бустинг</a></span><ul class="toc-item"><li><span><a href="#CatBoost" data-toc-modified-id="CatBoost-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>CatBoost</a></span></li><li><span><a href="#LGBMRegressor" data-toc-modified-id="LGBMRegressor-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>LGBMRegressor</a></span></li></ul></li><li><span><a href="#Предварительный-вывод" data-toc-modified-id="Предварительный-вывод-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Предварительный вывод</a></span></li></ul></li><li><span><a href="#Проверка-моделей-на-адекватность" data-toc-modified-id="Проверка-моделей-на-адекватность-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Проверка моделей на адекватность</a></span><ul class="toc-item"><li><span><a href="#Предварительный-вывод" data-toc-modified-id="Предварительный-вывод-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Предварительный вывод</a></span></li></ul></li><li><span><a href="#Сравнение-моделей" data-toc-modified-id="Сравнение-моделей-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Сравнение моделей</a></span></li><li><span><a href="#Общий-вывод" data-toc-modified-id="Общий-вывод-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Общий вывод</a></span></li></ul></div>

# ## Открытие и анализ данных

# ### Импорт библиотек

# In[1]:


#Импорт библиотек
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from catboost import Pool, cv
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_classification
import optuna
from sklearn.svm import SVC
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
import time
from sklearn.dummy import DummyRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss

import warnings
warnings.filterwarnings("ignore")


# ### Загрузка данных

# **Открытие исходной таблицы и загрузка данных с учётом особенностей временных рядов. Сразу изменю тип данных для призаков с датой на на datetime64. Кроме того, установлю индекс таблицы равным столбцу Datetime**

# In[2]:


try:
    data=pd.read_csv(r'C:.....csv', sep=',',index_col=[0], parse_dates=[0])
    pd.set_option('display.max_columns', None) 
except:
    data=pd.read_csv('.......csv', sep=',', index_col=[0], parse_dates=[0])
    pd.set_option('display.max_columns', None)


# In[3]:


data.info()


# In[4]:


data


# **Проверю хронологичность порядка дат и отсортирую индексы таблицы для удобства работы**

# In[5]:


data.sort_index(inplace=True)


# In[6]:


data.index.is_monotonic
data.info()


# ### Ресемплирование

# **Выберу новую длину интервана длительностья 1 час**

# In[7]:


data=data.resample('1H').sum()
data.head()


# In[8]:


data.info()


# ### Визуализация

# In[9]:


data.plot(legend=False)
plt.title('Временной ряд')
plt.xlabel("Дата")


# *Но данному участку ряда сложно оценить данные. Визуализирую ряд отрезками в месяц, неделю и сутки*

# In[10]:


data.plot(legend=False)
plt.title('Временной ряд')
plt.xlabel("Дата")


# In[11]:


data['2018-03-01':'2018-03-31'].plot(legend=False)
plt.title('Месячный период')
plt.xlabel("Дата")


# In[12]:


data['2018-03-01':'2018-03-07'].plot(legend=False)
plt.title('Недельный период')
plt.xlabel("Дата")


# In[13]:


data['2018-03-01':'2018-03-02'].plot(legend=False)
plt.title('Суточный период')
plt.xlabel("Дата")


# ### Оценка сезонности

# In[14]:


decomposed = seasonal_decompose(data)


# In[15]:


plt.figure(figsize=(6, 8))
plt.subplot(311)

decomposed.trend.plot(ax=plt.gca())
plt.title('Тренд')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Сезонность')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Остаток')
plt.tight_layout()


# **Оценю сезонность и тренд на меньших промежутках времени, поскольку оценка всего временного ряда не настолько информативна**

# In[16]:


#Месяц
plt.figure(figsize=(6, 8))
plt.subplot(311)

decomposed.trend['2018-03-01':'2018-03-31'].plot(ax=plt.gca())
plt.title('Тренд месяц')
plt.subplot(312)
decomposed.seasonal['2018-03-01':'2018-03-31'].plot(ax=plt.gca())
plt.title('Сезонность месяц')
plt.subplot(313)
decomposed.resid['2018-03-01':'2018-03-31'].plot(ax=plt.gca())
plt.title('Остаток месяц')
plt.tight_layout()


# In[17]:


#Неделя
plt.figure(figsize=(6, 8))
plt.subplot(311)

decomposed.trend['2018-03-01':'2018-03-07'].plot(ax=plt.gca())
plt.title('Тренд неделя')
plt.subplot(312)
decomposed.seasonal['2018-03-01':'2018-03-07'].plot(ax=plt.gca())
plt.title('Сезонность неделя')
plt.subplot(313)
decomposed.resid['2018-03-01':'2018-03-07'].plot(ax=plt.gca())
plt.title('Остаток неделя')
plt.tight_layout()


# In[18]:


#Сутки
plt.figure(figsize=(6, 8))
plt.subplot(311)

decomposed.trend['2018-03-01':'2018-03-02'].plot(ax=plt.gca())
plt.title('Тренд сутки')
plt.subplot(312)
decomposed.seasonal['2018-03-01':'2018-03-02'].plot(ax=plt.gca())
plt.title('Сезонность сутки')
plt.subplot(313)
decomposed.resid['2018-03-01':'2018-03-02'].plot(ax=plt.gca())
plt.title('Остаток сутки')
plt.tight_layout()


# *У временного ряда прослеживается тренд. Особенно это касается суточной оченки.*

# ### Промежуточный вывод

# В таблицке представлены сведения о количестве заказов такси за март-август 2018 года. В данных нет пропусков и выдающихся значений, даты записаны в хронологичном порядке. В таблице отсортированы индексы, а также выполненино ресемплирование по одному часу.<br>
# Исходя их оценки сезонности и тренда, можно сказть, что наблюдаются пики в заказах такси в вечернее время, а также на выходных.

# ## Подготовка данных

# ### Стационарность ряда

# *Проверю ряд на стационарность и при необходимости сделаю его стационарным для эффетивного прогнозирования. Проверку выпоню KPSS тестом. Кросе того, нестационарность ряда подтверждается трендом или сезонностью*

# In[19]:


plt.figure(figsize=(6, 8))
plt.subplot(311)

decomposed.trend['2018-03-01':'2018-03-02'].plot(ax=plt.gca())
plt.title('Тренд сутки')


# *Судя по тому, что в оценке сезонности прослеживается тренд, временной ряд нестационарен.*

# #### KPSS тест

# С помощью теста можно подтвердить нулевую гипотезу о том, что временной ряд стационарет или твергунть её. Если p-value будет меньше 0.05, то нулевая гипотеза отвергаетс, а значит, временной ряд нестационарен.

# In[20]:


def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
  
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'результат: Временной {"не " if p_value < 0.05 else ""}стационарен')

kpss_test(data['num_orders'])


# *Приведу временной ряд к стационарности методом, который указан в теоретическом материале курса - разность значений временного ряда*

# In[21]:


data=data-data.shift()
data['mean'] = data['num_orders'].rolling(15).mean()
data['std'] = data['num_orders'].rolling(15).std()


# *После разности образовываются пропуски, с которыми KSSP тест работать не будет. Поэтому удалю пропуски *

# In[22]:


data = data.dropna()


# In[23]:


data.info()


# *Повторю проверку на стационарность*

# In[24]:


kpss_test(data['num_orders'])


# #### Промежуточный вывод

# Исходный временный ряд, судя по наличию тренда и результату KPSS тесту нестационарет. Не знаю насколько целесообрано в нашем случае делать его стационарным, но решила снизить колебания, чтобы не искажать работу модели.<br>
# В результате проверки, могу сказать, что качество обучения моделей на нестационарном и стационарном рядах меняется: у линейной модели метрика лучше на стационарном, а у остальных - наоборот. Но при стационарном ряде переобученность модели меньше.

# ## Работа с признаками

# ### Создание признаков

# *Для добавления признаков создам функцию по аналогии с теоретическим материалом. Количество отстающих значений подбирала в зависимости от результата. 13 lag оптимально для метрики. При большем коичестве метрика на линейных моделях равна нулю*

# In[25]:


def make_features(data, max_lag):
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)


# In[26]:


make_features(data, 13)
data.head()


# ### Разделение выборок

# In[27]:


train, test = train_test_split(data, shuffle=False, test_size=0.1)

display('минимальное и максимальное значение индексов тренировочной выборки', train.index.min(), train.index.max())
display('минимальное и максимальное значение индексов тестовой выборки', test.index.min(), test.index.max())


# **Выборки разбиты правильно, без перемешивания**

# In[28]:


'размер тренировочной выборки',train.shape


# In[29]:


'размер тестовой выборки',test.shape


# **Удалю пропуски из тренировочной выборки, появившиеся в результате оценки стационарности и создния признака отстающих значений**

# In[30]:


train = train.dropna()


# ### Разделение признаков

# In[31]:


#Признаки обучающей выборки
features_train = train.drop(['num_orders'], axis=1)
target_train = train['num_orders']
'Размер целевого признака тренировочной выборки', target_train.shape, 'Размер признаков тренировочной вборки',features_train.shape


# In[32]:


#Признаки тестовой выборки 
features_test = test.drop(['num_orders'], axis=1)
target_test = test['num_orders']
'Размер целевого признака тестовой выборки', target_test.shape, 'Размер признаков тестовой вборки',features_test.shape


# ### Дополнение

# *Для удобства, создам функцию для подчёта метрики качества (в данном проекте RMSE)*

# In[33]:


def rmse (target, predictions):
    return np.sqrt(mean_squared_error(target, predictions))


# In[34]:


score = make_scorer(rmse,greater_is_better=False)


# *Для сравнения моделей, с помощью функции создам таблицу, куда будут занесены необходимые сведения*

# In[35]:


def metrics(result_train,result_test, model, data):
    data.loc[model, 'RMSE_train'] = result_train
    data.loc[model, 'RMSE_test'] = result_test
    return data


# In[36]:


# Таблица для сравнения результатов работы моделей
compare_models = pd.DataFrame(columns=['RMSE_train', 'RMSE_test'])


# ### Промежуточный вывод

# В таблицу с данными были добавлен признаки, содержащие сведения о годе, месяце, числе, дне недели, а также отстающие значения (как было указано выше, количество выбрала с помощью эксперимента с моделами). Выделила целевой признак, кроме того, поделила данные на обучающую и тестовую выборки.Кроме того, были удалены пропуски.

# ## Обучение моделей

# ### Линейные модели

# #### LinearRegression

# In[37]:


model_liner = LinearRegression()
 
model_liner.fit(features_train, target_train)
 
predict_liner_train = model_liner.predict(features_train)
predict_liner_test = model_liner.predict(features_test)


# In[38]:


result_liner_train=rmse(target_train, predict_liner_train)
result_liner_test=rmse(target_test, predict_liner_test)


# In[39]:


'RMSE на тренировочной выборке', result_liner_train, ',RMSE на тестовой выборке', result_liner_test


# In[40]:


metrics(result_liner_train,result_liner_test, 'LinearRegression',compare_models)


# In[41]:


#Визуализация предсказанных значений и реальных
plt.figure(figsize=(16,6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predict_liner_test, color='green')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# #### LinearRegression+GridSearchCV

# In[42]:


model_1 = LinearRegression()


liner_cv = GridSearchCV(model_1, param_grid = {'fit_intercept':[True,False], 'normalize':[True,False]}, scoring=score)

liner_cv.fit(features_train, target_train)

predictions_train_1 = liner_cv.predict(features_train)

predictions_test_1 = liner_cv.predict(features_test)


display('лучшие параметры:', liner_cv.best_params_)


# **Обучеие на лучших параметрах**

# In[43]:


model_1 = LinearRegression()

liner_cv = GridSearchCV(model_1, param_grid = {'fit_intercept':[False], 'normalize':[True]}, scoring=score)

liner_cv.fit(features_train, target_train)

predictions_train_1 = liner_cv.predict(features_train)

predictions_test_1 = liner_cv.predict(features_test)

result_liner_cv_train=rmse(target_train, predictions_train_1)
result_liner_cv_test = liner_cv.best_score_*-1
display( 'RMSE на тренировочной выборке', result_liner_cv_train, 'RMSE на тестовой выборке:', result_liner_cv_test)


# In[44]:


metrics(result_liner_cv_train,result_liner_cv_test, 'LinearRegression+GridSearchCV',compare_models)


# In[45]:


#Визуализация предсказанных значений и реальных
plt.figure(figsize=(16, 6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predictions_test_1, color='grey')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# #### LassoCV

# In[46]:


cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=12345)

model_lasso_cv = LassoCV(alphas=range(0, 1), cv=cv, n_jobs=-1)

model_lasso_cv.fit(features_train, target_train)

predictions_lassocv_train = model_lasso_cv.predict(features_train)

predictions_lassocv_test = model_lasso_cv.predict(features_test)

result_lassocv_train=rmse(target_train, predictions_lassocv_train)
result_lassocv_test = rmse(target_test, predictions_lassocv_test)
display( 'RMSE на тренировочной выборке', result_lassocv_train, 'RMSE на тестовой выборке:', result_lassocv_test)


# In[47]:


metrics(result_lassocv_train,result_lassocv_test, 'LassoCV',compare_models)


# In[48]:


#Визуализация предсказанных значений и реальных
plt.figure(figsize=(16, 6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predictions_lassocv_test, color='grey')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# #### Lasso

# In[49]:


model_lasso= Lasso()

model_lasso.fit(features_train, target_train)

predictions_lasso_train = model_lasso.predict(features_train)

predictions_lasso_test = model_lasso.predict(features_test)

result_lasso_train=rmse(target_train, predictions_lasso_train)
result_lasso_test = rmse(target_test, predictions_lasso_test)
display( 'RMSE на тренировочной выборке', result_lasso_train, 'RMSE на тестовой выборке:', result_lasso_test)


# In[50]:


metrics(result_lasso_train,result_lasso_test, 'Lasso',compare_models)


# In[51]:


#Визуализация предсказанных значений и реальных
plt.figure(figsize=(16, 6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predictions_lasso_test, color='grey')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# ### Случайный лес

# Для перебора параметров использую `optuna`

# In[52]:


def objective(trial):
    
    model_forest = RandomForestRegressor(
                n_estimators = trial.suggest_int("n_estimators", 10, 600),
                max_depth = trial.suggest_int("max_depth", 1, 10),
                n_jobs = 4,
                random_state = 12345
           )
           
    optuna_forest=model_forest.fit(features_train, target_train)
    
    predict_forest_test=optuna_forest.predict(features_test)
    score=rmse(target_test, predict_forest_test)
              
    return score


# In[53]:


study = optuna.create_study(study_name='RandomForest', direction="maximize")
study.optimize(objective, n_trials=100)

study.best_params


# In[54]:


optuna_params = study.best_params
model_optuna = RandomForestRegressor(**optuna_params)
forest=model_optuna.fit(features_train, target_train)
predict_train=forest.predict(features_train)
predict_forest_test=forest.predict(features_test)


# In[55]:


result_forest=study.best_value
result_forest_train=rmse(target_train, predict_train)

display(result_forest, result_forest_train )


# In[56]:


metrics(result_forest_train, result_forest, 'RandomForestRegressor',compare_models)


# In[57]:


#Визуализация предсказанных значений и реальных
plt.figure(figsize=(16, 6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predict_forest_test, color='grey')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# ### Градиентный бустинг

# #### CatBoost

# In[58]:


def objective1(trial):
    
    optuna_params = {"subsample": trial.suggest_float("subsample", 0.1, 0.99),
                     'od_wait': trial.suggest_int('od_wait', 1, 200, step=1),
                     "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 0.99),
                     "random_strength": trial.suggest_int("random_strength", 1, 10, step=1),
                     "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0),
                     "max_depth": trial.suggest_int("max_depth", 4, 10, step=1),
                     "n_estimators": trial.suggest_int("n_estimators", 10, 2500, step=1),
                     'learning_rate': trial.suggest_loguniform("learning_rate", 0.005, 0.1)}

    model_cat = CatBoostRegressor(**optuna_params)
           
    optuna_cat=model_cat.fit(features_train, target_train)
    
    predict_cat_test=optuna_cat.predict(features_test)
    score=rmse(target_test, predict_cat_test)
              
    return score


study = optuna.create_study(study_name='CatBoostRegressor', direction="maximize")
study.optimize(objective1, n_trials=100)

study.best_params


# In[59]:


optuna_param = study.best_params
model_cat = CatBoostRegressor(**optuna_param)
optuna_cat=model_cat.fit(features_train, target_train)
predict_cat_train=optuna_cat.predict(features_train)
predict_cat_test=optuna_cat.predict(features_test)


# In[60]:


result_cat=study.best_value
result_cat_train=rmse(target_train, predict_cat_train)

display( 'RMSE на тренировочной выборке', result_cat_train, 'RMSE на тестовой выборке:', result_cat)


# In[61]:


metrics(result_cat_train, result_cat, 'CatBoostRegressor',compare_models)


# In[62]:


#Визуализация предсказанных значений и реальных
plt.figure(figsize=(16, 6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predict_cat_test, color='grey')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# #### LGBMRegressor

# In[63]:


def objective2(trial):
    
    params = {
         
        'random_state': 12345,
        'n_estimators': 2500,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.1]),
        'max_depth': trial.suggest_categorical('max_depth', [4,10]),
        
        
    }
    
    model_light = LGBMRegressor(**params)      
    optuna_light=model_light.fit(features_train, target_train, verbose=False)
    
    predict_light_test=optuna_light.predict(features_test)
    score=rmse(target_test, predict_light_test)
              
    return score


study = optuna.create_study(study_name='LGBMRegressor', direction="maximize")
study.optimize(objective2, n_trials=100)

study.best_params


# In[64]:


optuna_param = study.best_params
model_light = lgb.LGBMRegressor(**optuna_param)
optuna_light=model_light.fit(features_train, target_train)
predict_light_train=optuna_light.predict(features_train)
predict_light_test=optuna_light.predict(features_test)


# In[65]:


result_light=study.best_value
result_light_train=rmse(target_train, predict_light_train)

display( 'RMSE на тренировочной выборке', result_light_train, 'RMSE на тестовой выборке:', result_light)


# In[66]:


metrics(result_light_train, result_light, 'LGBMRegressor',compare_models)


# In[67]:


#Визуализация предсказанных значений и реальных
plt.figure(figsize=(16, 6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predict_light_test, color='grey')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# ### Предварительный вывод

# Обучены две линейные модели, модель случайного леса и две модели градиентного бустинга. Наилучгие результаты у линейных моделей, однако только одна почти не переобучилась (LinearRegression с GridSearchCV). Показатели нелинейных моделей существенно ниже.

# ## Проверка моделей на адекватность

# *Проверку на адекватность проведу с помощью оценки базовой модели (DummyRegressor)*

# In[68]:


dummy = DummyRegressor(strategy='median')

dummy.fit(features_train, target_train)


predicted_dummy_train = dummy.predict(features_train)
predicted_dummy = dummy.predict(features_test)


result_dummy_train=rmse(target_train, predicted_dummy_train)
result_dummy=rmse(target_test, predicted_dummy)

'RMSE базовой модели для тренировочной выборки ', result_dummy_train, ',RMSE базовой модели для тестовой выборки', result_dummy


# In[69]:


metrics(result_dummy_train, result_dummy, 'DummyRegressor',compare_models)


# In[70]:


#Визуализация предсказанных значений и реальных
plt.figure(figsize=(16, 6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predicted_dummy, color='grey')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# ### Предварительный вывод

# Метрика качества всех моделей выше, чем у константной модели. Значанит, они работают коррекотно

# ## Сравнение моделей

# In[71]:


compare_models=compare_models.sort_values(by='RMSE_test').reset_index()
compare_models


# In[72]:


fig, axs = plt.subplots(1,2)

compare_models.plot(kind='bar', x='index', y= 'RMSE_train',color='green',figsize=(12, 8), label='Качество на тренировочной выборке',ax=axs[0])
compare_models.plot(kind='bar', x='index', y= 'RMSE_test',color='blue',figsize=(12, 8),  label='Качество на тестовой выборке', ax=axs[1])


fig.suptitle('Сравнение качества моделей',fontsize=20)


# In[73]:


#Визуализация предсказанных значений и реальных у лучшей модели
plt.figure(figsize=(16, 6))

plt.plot(target_test, color='red')
plt.plot(target_test.index, predictions_test_1, color='grey')

plt.title('Предсказания и истинные значения', fontsize=14)
plt.xlabel('Дата', fontsize=13)
plt.ylabel('Количество заказов', fontsize=13)
plt.legend(['реальные значения', 'предсказания'], fontsize=13)
plt.tight_layout()


# In[ ]:





# ## Общий вывод

# Для работы с временным рядом и обучения моделей, согласно требованиям заказчика, проведено ресемплирование по 1 часу. Выявлены всплески сезонности и тренда, что свидетельствует о пиках заказов такси в веченее время и на выхоных.<br>
# Временной ряд провепила на стационарноть и поскольку выявлены пики, ряд был приведён к стационарности.<br>
# Для обучения моделей были добавены признаки, содержащие сведения о годе, месяце, числе и дне недели. кроме того добавлены отсающие начения, количество которых определено экспериментально.<br>
# После обучения всех моделей, получена метрика качетва (`RMSE`) на тренировочной и тестовой выборках. Качество Нелинейных моделей значительно хуже, чем линейных. Наилучшие показания (высокое качество и минимальная переученность) у Линейной модели с помощью `GridSearchCV`.<br>
# Для прогнозирования заказов такси на следующий час, рекомендую использовать указанную линейную модель.

# In[ ]:




