import numpy as np  # модуль для математичних та числових операцій
import pandas as pd # програмна бібліотека для роботи з даними
import matplotlib.pyplot as plt # інтерфейс для побудови графіків
from scipy.constants import hp
from statsmodels.graphics.tsaplots import plot_acf  # автокореляційний графік

from sklearn import metrics  # завантаження пакету метрик
from sklearn.metrics import mean_squared_error  # завантаження метрик MSE
from sklearn.model_selection import train_test_split  # функція для поділу вибірки
from sklearn.model_selection import cross_val_score  # перехресна перевірка
from sklearn.model_selection import cross_val_predict # розрахунок оцінки точності
from sklearn.pipeline import make_pipeline # побудова конвеєрів
from sklearn.preprocessing import RobustScaler  # попередня обробка даних
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet # моделі регресій
from sklearn.kernel_ridge import KernelRidge # модель регресії
from sklearn.ensemble import BaggingRegressor  # модель бегінг
from sklearn.ensemble import GradientBoostingRegressor  # модель градієнтного бустингу
from sklearn.ensemble import StackingRegressor  # модель стекінгу

from sktime.forecasting.base import ForecastingHorizon # горизонт прогнозування
from sktime.forecasting.naive import NaiveForecaster # "наївний" прогноз
from sktime.forecasting.exp_smoothing import ExponentialSmoothing # експонентне згладжування

import statsmodels.api as sm  # ARIMA/SARIMA
import pmdarima as pm # ARIMA/SARIMA
import xgboost as xgb # модель XGboost
from xgboost.sklearn import XGBRegressor

import lightgbm as lgb  # модель lightGBM
import imageio # робота із зображеннями

import time  # пакет для розрахунку часу роботи коду
import os # модуль для роботи з операційною системою
import warnings # попередження про помилки

n_jobs = -1  # Цей параметр контролює паралельну обробку. -1 означає використання всіх процесорів.
random_state = 42  # Цей параметр контролює випадковість даних.
                   # Використання певного значення int для отримання однакових результатів
                   # під час кожного запуску цього коду.

# Завантажуємо дані та налаштовуємо частоту:
df = pd.read_excel('Beer.xlsx', parse_dates=['Дата'], index_col=[0])
df.sort_index(inplace=True)
df = df.resample('M').mean()
df = df.replace(np.nan, 0)

# Графік:
plt.figure(figsize = (15,5))
plt.plot('.',col='green')
plt.plot('.',col='darkorange')
plt.plot('.',col='red')
plt.plot('.',col='darkblue')
plt.plot('.',col='black')
plt.ylabel('Продажі',fontsize=10)
plt.xlabel('Дата',fontsize=10)
plt.grid(True)
plt.legend()


# Функція для поділу вибірку на навчальну та тестову:
def split_data(data, split_date):
    return data[data.index <= split_date].copy(), data[data.index > split_date].copy()
train, test = split_data(df, '2017-12-31')


# Створюємо більше інформації за даними:
def create_features(df):
    df['Дата'] = df.index
    df['Місяць'] = df['Дата'].dt.month
    df['Рік'] = df['Дата'].dt.year
    X = df[['Місяць', 'Рік']]
    return X

# Розділяємо вибірку:
X_train, y_train = create_features(train), train['Мін.вода']  # У дужках вказуємо назву стовпця для якого будуємо прогноз
X_test, y_test = create_features(test), test['Мін.вода']
print(f'Навчальний набір: X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'Тестовий набір: X_test: {X_test.shape}, y_test: {y_test.shape}')

# Функції для обчислення помилок:
# RMSE
def rmse(model):
    model.fit (X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)
# False повертає значення RMSE
# MAPE
def MAPE(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100
# MAE(MAD)
def error(y_true, y_pred):
    return y_true-y_pred
def mae(y_true, y_pred):
    return np.median(np.abs(error(y_true,y_pred)))

# Створимо порожні змінні для запису результатів роботи методів:
all_scores = []  # загальний для всіх методів
models_scores = []
simple_scores = []
ensemble_scores =[]

# Попередні розрахунки для моделей:

# LinearRegression
# відповідає лінійній моделі з коефіцієнтами w = (w1, ..., wp)
# для мінімізації залишкової суми квадратів між цілями, що спостерігаються в наборі даних і цілями,
# передбачуваними лінійною апроксимацією.
start_time = time.time()
linear_regression = make_pipeline(LinearRegression())
score = rmse(linear_regression)
models_scores.append(['LinearRegression', score, time.time() - start_time])
print(f'LinearRegression Score= {score}')
print("--- %s seconds ---" % (time.time() - start_time))

# Lasso Regression
# Це лінійна модель, яка оцінює розріджені коефіцієнти.
# Це корисно в деяких контекстах через свою тенденцію віддавати перевагу рішенням з меншою кількістю
# ненульових коефіцієнтів, ефективно зменшуючи кількість функцій, від яких залежить дане рішення.
# Ця модель може бути дуже чутливою до викидів. Тому нам потрібно зробити їх більш надійними.
# Для цього ми використовуємо метод Robustscaler() sklearn на джерелі інформації.
start_time = time.time()
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=random_state))
score = rmse(lasso)
models_scores.append(['Lasso', score, time.time() - start_time])
print(f'Lasso Score= {score}')
print("--- %s seconds ---" % (time.time() - start_time))

# ElasticNet Regression
# Корисна, коли є кілька функцій, які корелюють один з одним.
# Лассо, ймовірно, вибере одне з них навмання, а Elastic-Net - і те, і інше, що дозволяє їй успадкувати частину
# стабільності Ridge при чергуванні.
# Ця модель може бути дуже чутливою до викидів, тому нам потрібно зробити їх більш надійними.
# Для цього ми використовуємо метод Robustscaler() sklearn на джерелі інформації.
start_time = time.time()
elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=random_state))
score = rmse(elastic_net)
models_scores.append(['ElasticNet', score, time.time() - start_time])
print(f'ElasticNet Score= {score}')
print("--- %s seconds ---" % (time.time() - start_time))

# KernelRidge Regression
# Ріджева регресія вирішує деякі проблеми звичайних найменших квадратів, накладаючи штраф на розмір коефіцієнтів.
start_time = time.time()
kernel_ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmse(kernel_ridge)
models_scores.append(['KernelRidge', score, time.time() - start_time])
print(f'KernelRidge Score= {score}')
print("--- %s seconds ---" % (time.time() - start_time))

# Рейтинг кожної моделі:
pd.DataFrame(models_scores,columns=['model','RMSE','time']).sort_values(by=[' ;RMSE'], ascending=True)

# Прості методи прогнозування:
# Горизонт прогнозування
fh = ForecastingHorizon(y_test.index, is_relative=False)
# Наївний прогноз з урахуванням сезонності
start_time = time.time()
SeasonalNaive = NaiveForecaster(strategy="last", sp=12).fit(y_train)
SeasonalNaive_pred = SeasonalNaive.predict(fh)
all_scores.append(['Наївний прогноз', mae(y_test,SeasonalNaive_pred), MAPE(y_test,SeasonalNaive_pred), time.time() - start_time])
simple_scores.append(['Наївний прогноз', mae(y_test,SeasonalNaive_pred), MAPE(y_test,SeasonalNaive_pred), time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))


#Модель Хольта - Вінтерса
start_time = time.time()
HW = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12).fit()
HW_pred = HW.predict(start = y_test.index[0],end=y_test.index[-1])
all_scores.append(['Модель Хольта-Вінтерса', mae(y_test,HW_pred), MAPE(y_test,HW_pred), time.time() - start_time])
simple_scores.append(['Модель Хольта-Вінтерса', mae(y_test,HW_pred), MAPE(y_test,HW_pred), time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))

# Ковзне середнє (Simple Average Method)
start_time = time.time()
S_mean = NaiveForecaster(strategy="mean", sp=12)
S_mean.fit(y_train)
S_mean_pred = S_mean.predict(fh)
all_scores.append(['Ковзне середнє', mae(y_test, S_mean_pred), MAPE(y_test, S_mean_pred), time.time() - start_time])
simple_scores.append(['Ковзне середнє', mae(y_test, S_mean_pred), MAPE(y_test, S_mean_pred), time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))

# Вибір параметрів SARIMA
AutoSARIMA = pm.auto_arima(y_train, seasonal=True, m=12)
AutoSARIMA.summary()

# SARIMA
start_time = time.time()
SARIMA = sm.tsa.arima.ARIMA(y_train, order=(0, 1, 0), seasonal_order=(0, 1, 1, 12))
SARIMA = SARIMA.fit(low_memory=True, cov_type=' none')
SARIMA_pred = SARIMA.forecast(y_test.index[-1], dynamic=True)
all_scores.append(['SARIMA ', mae(y_test, SARIMA_pred), MAPE(y_test, SARIMA_pred), time.time() - start_time])
simple_scores.append(['SARIMA ', mae(y_test, SARIMA_pred), MAPE(y_test, SARIMA_pred), time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))

# Створюємо таблицю з помилками:
pd.DataFrame(simple_scores, columns=['Модель', 'MAD', 'MAPE', 'Час роботи']).sort_values(by=['MAPE'], ascending=True)

# Ансамблі моделей
# Bagging
start_time = time.time()
bag_reg = BaggingRegressor(base_estimator=XGBRegressor(), n_estimators=100,
                           max_samples=1.0, max_features=1.0,
                           bootstrap=True, bootstrap_features=False, oob_score=True, n_jobs=n_jobs,
                           random_state=random_state)
bag_reg.fit(X_train, y_train)
X_test_bag_pred = bag_reg.predict(X_test)
ensemble_scores.append(['Bagging', mae(y_test, X_test_bag_pred), MAPE(y_test, X_test_bag_pred), time.time() - start_time])
all_scores.append(['Bagging', mae(y_test, X_test_bag_pred), MAPE(y_test, X_test_bag_pred), time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))

# Boosting
# GradientBoostingRegressor, XGBRegressor, LGBMRregressor.
#GradientBoostingRegressor
start_time = time.time()
GBreg = GradientBoostingRegressor (n_estimators = 100,
                                   learning_rate = 0.1,
                                   max_depth = 3,
                                   min_samples_leaf = 2,
                                   min_samples_split = 12,
                                   random_state = random_state)
GBreg.fit(X_train, y_train)
X_test_GB_pred = GBreg.predict(X_test)
ensemble_scores.append(['GradientBoosting', mae(y_test,X_test_GB_pred), MAPE(y_test,X_test_GB_pred),time.time() - start_time])
all_scores.append(['GradientBoosting', mae(y_test,X_test_GB_pred), MAPE(y_test,X_test_GB_pred),time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))


#XGBRegressor
params_last = {'base_score': 0.5,
               'booster': 'gbtree', 'colsample_bylevel': 1,
               'colsample_bynode': 1,
               'colsample_bytree': 0.4,
               'gamma': 0,

               'max_depth': 2,
               'min_child_weight': 5,
               'reg_alpha': 0,
               'reg_lambda': 1,
               'seed': 38,
               'subsample': 0.7,
               'verbosity': 1, 'learning_rate': 0.01
               }
start_time = time.time()
XGB_reg = xgb.XGBRegressor(**params_last, n_estimators=2000).fit(X_train, y_train)
X_test_XGB_pred = XGB_reg.predict(X_test)
ensemble_scores.append(['XGBoost', mae(y_test, X_test_XGB_pred), MAPE(y_test, X_test_XGB_pred), time.time() - start_time])
all_scores.append(['XGBoost', mae(y_test, X_test_XGB_pred), MAPE(y_test, X_test_XGB_pred), time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))

#LGBMRegressor
#LightGBM parameters
lgb_reg_params = {
    'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
    'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample': hp.uniform('subsample', 0.8, 1),
    'n_estimators': 100
}
# Fit parameters
lgb_fit_params = {
    'eval_metric': 'l2',
    'early_stopping_rounds': 10,
    'verbose': False
}
# Loss function
lgb_para = {
    'reg_params': lgb_reg_params,
    'fit_params': lgb_fit_params,
    'loss_func': lambda y, pred: np.sqrt(mean_squared_error(y, pred))
}
start_time = time.time()
lgbm_reg = lgb.LGBMRegressor(boosting_type = 'dart',objective='regression', num_leaves=5,
                             learning_rate=0.05,
                             n_estimators=96,
                             ax_bin=55,
                             agging_fraction=1,
                             bagging_freq = 5,
                             feature_fraction = 0.2319,
                             feature_fraction_seed=9,
                             bagging_seed=9,
                             min_data_in_leaf =12,
                             min_sum_hessian_in_leaf = 11,
                             random_state = random_state)
lgbm_reg.fit(X_train, y_train)
X_test_lgbm_pred = lgbm_reg.predict(X_test)
ensemble_scores.append(['Light GBM7', mae(y_test,X_test_lgbm_pred), MAPE(y_test,X_test_lgbm_pred),time.time() - start_time])
all_scores.append(['Light GBM', mae(y_test,X_test_lgbm_pred), MAPE(y_test,X_test_lgbm_pred),time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))


# Stacking
start_time = time.time()
estimators = [ (linear_regression, linear_regression), ('kernel_ridge', kernel_ridge), ('xgb_regressor', XGB_reg) ]
st_reg = StackingRegressor (estimators = estimators,
                            final_estimator = kernel_ridge,
                            cv = 5,
                            n_jobs = n_jobs,
                            passthrough = False)
st_reg.fit(X_train, y_train)
X_test_st_pred = st_reg.predict(X_test)
ensemble_scores.append(['Stacking ', mae(y_test,X_test_st_pred), MAPE(y_test,X_test_st_pred),time.time() - start_time])
all_scores.append(['Stacking ', mae(y_test,X_test_st_pred), MAPE(y_test,X_test_st_pred),time.time() - start_time])
print("--- %s seconds ---" % (time.time() - start_time))

# Створюємо таблицю з помилками ансамблів
pd.DataFrame(ensemble_scores, columns=['Модель', 'MAE', 'MAPE', 'Час роботи']).sort_values(by=['MAPE'], ascending=True)

# Створюємо загальну таблицю з помилками
pd.DataFrame(all_scores, columns=['Модель', 'MAD', 'MAPE', 'Час роботи']).sort_values(by=['MAPE'], ascending=True)

#ГРАФІКИ

# Сезонний наївний прогноз
plt.plot(y_test,'o', col='sandybrown')
plt.plot(SeasonalNaive_pred,'X', col='forestgreen')
plt.ylabel('Sales', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.grid(True)
plt.legend(['Факт','Наївний прогноз'], fontsize=20)


# Модель Хольта - Вінтерса
plt.legend(['Факт', 'Модель Хольта-Вінтерса'], fontsize=20)

# Метод ковзної середньої
plt.figure(figsize=(25, 7))
plt.plot(y_test, 'o', col='sandybrown')
plt.plot(S_mean_pred, 'X', col='rebeccapurple')
plt.ylabel('Sales', fontsize=14)
plt.xlabel(' ;Date', fontsize=14)
plt.grid(True)
plt.legend(['Факт', 'Ковзна середня'], fontsize=20)

#SARIMA
#Ансамблі
#Bagging
def plot_performance(base_data, date_from, date_to, title=None):
    plt.figure(figsize=(25, 7))
    if title is None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.ylabel('Sales', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    plot_performance(df, df.index[0].date(), df.index[-1].date(), 'Original and Predicted Data (Bagging)')
    plot_performance(y_test, y_test.index[0].date(), y_test.index[-1].date(), 'Test and Predicted Data (Bagging)')

#Boosting
#GradientBoosting
def plot_performance_GB(base_data, date_from, date_to, title=None):
    plt.figure(figsize=(25, 7))
    if title is None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.ylabel('Sales', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    plot_performance_GB(df, df.index[0].date(), df.index[-1].date(), 'Original and Predicted Data (GradientBoosting)')
    plot_performance_GB(y_test, y_test.index[0].date(), y_test.index[-1].date(), 'Test and Predicted Data (GradientBoosting)')

#XGBoost
def plot_performance_XGB(base_data, date_from, date_to, title=None):
    plt.figure(figsize=(25, 7))
    if title is None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.ylabel('Sales', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    plot_performance_XGB(df, df.index[0].date(), df.index[-1].date(), 'Original and Predicted Data (XGBoost)')
    plot_performance_XGB(y_test, y_test.index[0].date(), y_test.index[-1].date(), 'Test and Predicted Data (XGBoost)')

#LightGBM
def plot_performance_LGBM(base_data, date_from, date_to, title=None):
    plt.figure(figsize=(25, 7))
    if title is None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.ylabel('Sales', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    plot_performance_LGBM(df, df.index[0].date(), df.index[-1].date(), 'Original and Predicted Data (LightGBM)')
    plot_performance_LGBM(y_test, y_test.index[0].date(), y_test.index[-1].date(), 'Test and Predicted Data (LightGBM)')

#Stacking
def plot_performance_st(base_data, date_from, date_to, title=None):
    plt.figure(figsize=(25, 7))
    if title is None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.ylabel('Sales', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    plot_performance_st(df, df.index[0].date(), df.index[-1].date(), 'Original and Predicted Data (Stacking)')
    plot_performance_st(y_test, y_test.index[0].date(), y_test.index[-1].date(), 'Test and Predicted Data (Stacking)')

