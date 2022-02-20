from binance.client import Client
import keyboard
# import plotly.graph_objs as go
import talib as ta
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier  # árbol de decisión para clasificación
import os
import time
from datetime import timedelta
import winsound
import smtplib
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import OneSidedSelection
import itertools as it
from itertools import combinations
from sklearn import tree

# Connect with your API key
key = os.environ.get('bi_ap')
secret = os.environ.get('bi_ke')
client = Client(api_key=key, api_secret=secret)

cPar = 'BTCUSDT'  #'ETHBTC'

nInterval = 30  # min
nMaxRetencion = 240  # maximo tiempo de retencion en minutos
nTar = 0.003  # take profit 0.005
nSL = -nTar/(1/0.3)  # stop loss
nPerMax = int(nMaxRetencion / nInterval)  # período para take profit

cIntervalUni = 'm'
nStart = 60  #días a analizar
cStartUni = 'd'
nPorInvertir = 0.05  # % del monto de la cuenta a invertir en cada operación
nCom = 0.001  # 0.1% # comisión
period = 20  # periodo media movil larga
nTrainPer=0.7

#def prueba(nInt,nMaxRet,nTa):
#nInterval = nInt  # min
#nMaxRetencion = nMaxRet  # maximo tiempo de retencion en minutos
#nTar = nTa  # take profit 0.005
#nSL = -nTar/2  # stop loss
#nPer = int(nMaxRetencion / nInterval)  # período para take profit

def recupera_datos(nDesde,nInterval,cIntervalUni):
    # Get data
    data = client.get_historical_klines(cPar, interval=str(nInterval)+cIntervalUni, start_str=str(nDesde)+cStartUni)

    # Data cleaning
    df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav',
                      'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])

    #df = df.drop(['ignore'], axis=1)
    df= df.drop(["close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"],axis=1)# .head()
    # df.to_csv("C:/Users/axelt/Documents/Axel/datos.csv")

    # da formato a la fecha
    i = 0
    Date = []
    Hora = []
    while i < len(data):
        Date.append(datetime.fromtimestamp(data[i][0]/1000).strftime('%Y-%m-%d %H:%M:%S'))
        Hora.append(datetime.fromtimestamp(data[i][0]/1000).strftime('%Y-%m-%d %H'))
        i += 1
    df['Date'] = Date
    #df.set_index('Date', inplace=True)

    df['Hora'] = Hora

    # Data visualisation
    df['MA5'] = df['close'].ewm(span=10, adjust=False).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA100'] = df['close'].rolling(100).mean()
    df['MA200'] = df['close'].rolling(200).mean()

    # small time Moving average. calculate 20 moving average using Pandas over close price
    # data['sma'] = symbol_df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()  # Get standard deviation
    df['upper'] = df['MA20'] + (2 * df['std'])  # Calculate Upper Bollinger band
    df['lower'] = df['MA20'] - (2 * df['std'])  # Calculate Lower Bollinger band
    # convierte en formato float
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)

    df['Dif_MA5'] = df['close'].astype(float)-df['MA5'].astype(float)
#    df.loc[df.Dif_MA5 >= 0, 'Dif_MA5'] = 1
#    df.loc[df.Dif_MA5 < 0, 'Dif_MA5'] = -1

    df['Dif_MA20'] = df['close'].astype(float)-df['MA20'].astype(float)
#    df.loc[df.Dif_MA20 >= 0, 'Dif_MA20'] = 1
#    df.loc[df.Dif_MA20 < 0, 'Dif_MA20'] = -1

    df['Dif_MA100'] = df['close'].astype(float)-df['MA100'].astype(float)
#    df.loc[df.Dif_MA100 >= 0, 'Dif_MA100'] = 1
#    df.loc[df.Dif_MA100 < 0, 'Dif_MA100'] = -1

    df['Dif_MA200'] = df['close'].astype(float)-df['MA200'].astype(float)
#    df.loc[df.Dif_MA200 >= 0, 'Dif_MA200'] = 1
#    df.loc[df.Dif_MA200 < 0, 'Dif_MA200'] = -1

    df['Dif_upper'] = df['close'].astype(float)-df['upper'].astype(float)

    df['Dif_lower'] = df['close'].astype(float)-df['lower'].astype(float)
#    df.loc[df.Dif_lower >= 0, 'Dif_lower'] = 1
#    df.loc[df.Dif_lower < 0, 'Dif_lower'] = -1

    #df['Simple MA'] = ta.SMA(ohlcv['Close'], 10)
    #df['EMA'] = ta.EMA(ohlcv['Close'], timeperiod=20)
    #df['WMA'] = ta.WMA(ohlcv['Close'], timeperiod=50)

    # Bollinger Bands
    #df['upper_band'],df['middle_band'], df['lower_band'] = ta.BBANDS(ohlcv['Close'], timeperiod=20)

    # Momentum Indicator Functions
    df['RSI'] = ta.RSI(df['close'], 14)

    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df['close'],
                                                                    fastperiod=12,
                                                                    slowperiod=26,
                                                                    signalperiod=9)

    df['slowk'], df['slowd'] = ta.STOCH(df['high'], df['low'],
                                              df['close'], fastk_period=14,
                                              slowk_period=3, slowk_matype=0,
                                              slowd_period=3, slowd_matype=0)

    # Volume Indicator Functions
    df['OBV'] = ta.OBV(df['close'], df['volume'])

    # Volatility Indicator Functions
    df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # Pattern Recognition Functions
    df['BELTHOLD'] = ta.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close'])

    df['ADX'] = ta.ADX(df['high'],df['low'], df['close'], timeperiod=20)
    df.loc[df.ADX >= 25, 'ADX'] = 1
    df.loc[df.ADX <= 25, 'ADX'] = 0

    df['APO'] = ta.APO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    df['AROONOSC'] = ta.AROONOSC(df['high'], df['low'], timeperiod=14)
    df['BOP'] = ta.BOP(df['open'], df['high'], df['low'], df['close'])
    df['CCI'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['CMO'] = ta.CMO(df['close'], timeperiod=14)
    df['DX'] = ta.DX(df['high'], df['low'], df['close'], timeperiod=14)
    df['PPO'] = ta.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    df['ULTOSC'] = ta.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)


#### señales ########
    df['S_lower'] = 0.0
    df.loc[df.Dif_lower <= 0, 'S_lower'] = 1

    df['S_macd'] = 0.0
    df['S_macd'] = np.where(df['macd'] > df['macdsignal'], 1.0, 0.0)
#    df['P_macd'] = df['S_macd'].diff()

    df['S_RSI'] = 0.0
    df.loc[df.RSI <= 30, 'S_RSI'] = 1

##### conversión tipos #######
    df['RSI'] = df['RSI'].astype(float)
    df['macd'] = df['macd'].astype(float)
    df['macdsignal'] = df['macdsignal'].astype(float)
    df['macdhist'] = df['macdhist'].astype(float)
    df['slowk'] = df['slowk'].astype(float)
    df['slowd'] = df['slowd'].astype(float)
    df['slowd'] = df['slowd'].astype(float)
    df['OBV'] = df['OBV'].astype(float)
    df['ATR'] = df['ATR'].astype(float)
    df['BELTHOLD'] = df['BELTHOLD'].astype(float)
    df['ADX'] = df['ADX'].astype(float)
    df['APO'] = df['APO'].astype(float)
    df['AROONOSC'] = df['AROONOSC'].astype(float)
    df['BOP'] = df['BOP'].astype(float)
    df['CCI'] = df['CCI'].astype(float)
    df['CMO'] = df['CMO'].astype(float)
    df['DX'] = df['DX'].astype(float)
    df['PPO'] = df['PPO'].astype(float)
    df['ULTOSC'] = df['ULTOSC'].astype(float)

    # 0-fecha 1-apertura 2-Max 3-Min 4-cierre 5-volumen
    # 6- Media 5 7- Media 20 8- desv std 9-upper 10-lower 11-dif_ma5 18-dif Ma20
    # 12-dif upper 13- dif_lower 14-RSI 15-MACD 16-MACD sig 17-MACD hist 18-slowk 19-slowd 20-OBD 21-ATR 22-BELTHold

    #data = df.to_numpy()
    return df#data

#dfshort = recupera_datos(nStart,nInterval,cIntervalUni)
#dflong = recupera_datos(nStart,1,'h')
#dftotal = pd.merge(dfshort,dflong,suffixes=('_Short', '_Long'),on='Hora')
dftotal = recupera_datos(nStart,nInterval,cIntervalUni)
#dftotal = dftotal.drop(["Hora", "date_Long", "open_Long", "high_Long", "low_Long", "close_Long", "volume_Long", "MA5_Long", "MA20_Long", "std_Long", "upper_Long", "lower_Long"], axis=1)  # .head()
dftotal = dftotal.dropna()
#print(dftotal.columns)

dftotal.to_csv("C:/Users/axelt/Documents/Axel/datos.csv")

start_time = datetime.now() #datetime.strftime((data[len(data)-1][0]/1000),'%Y-%m-%d %H:%M:%S') #dfshort.iloc[-1, dfshort.columns.get_loc("date")] #dfshort(inplace=True)

# estima ganancia en los próximos n períodos
#dftotal=dftotal.assign(minpor='0')
#dftotal=dftotal.assign(maxpor='0')
#dftotal=dftotal.assign(maxminpor='0')
#dftotal=dftotal.assign(relmaxmin='0')
dftotal=dftotal.assign(target='0')
rRowIni=dftotal.index.min()
nRow = dftotal.index[-1] #  len(dftotal.index)
for y in range(rRowIni,nRow-nPerMax):  # dftotal.index:
#    if y == 12938:
#        print("")
    nPer=1
    while nPer < nPerMax:
        if (dftotal["high"][y-rRowIni+1:y-rRowIni+nPer+1].max()-dftotal["close"][y])/dftotal["close"][y]>nTar and (dftotal["low"][y-rRowIni+1:y-rRowIni+nPer+1].min()-dftotal["close"][y])/dftotal["close"][y]>nSL:
            dftotal.loc[y, 'target'] = '1'
            nPer = nPerMax
        else:
            nPer = nPer+1

#    if dftotal["close"][y]>0: dftotal.loc[y, 'maxpor'] = round((dftotal["high"][y+1:y+nPer+1].max()-dftotal["close"][y])/dftotal["close"][y],3)
#    if dftotal["close"][y]>0: dftotal.loc[y, 'minpor'] = round((dftotal["low"][y+1:y+nPer+1].min()-dftotal["close"][y])/dftotal["close"][y],3)
#    if dftotal["close"][y]>0: dftotal.loc[y, 'relmaxmin'] = round(((dftotal["high"][y+1:y+nPer+1].max()-dftotal["close"][y])/dftotal["close"][y])/((dftotal["low"][y+1:y+nPer+1].min()-dftotal["close"][y])/dftotal["close"][y]),3)
#    if dftotal["minpor"][y]>0: dftotal.loc[y, 'maxminpor'] = str(dftotal.loc[y, 'minpor']) + str(dftotal.loc[y, 'maxpor'])

#    if (dftotal["high_Short"][y+1:y+nPer+1].max()-dftotal["close_Short"][y])/dftotal["close_Short"][y]>nTar and (dftotal["low_Short"][y+1:y+nPer+1].min()-dftotal["close_Short"][y])/dftotal["close_Short"][y]>nSL: dftotal.loc[y, 'target'] = '1'
dftotal['target'] = dftotal['target'].astype(float)

#dftotal.to_csv('C:/Users/axelt/Documents/Axel/datos.csv')

nTrain=int(nTrainPer*len(dftotal.index))

#X_train_tmp = dftotal.loc[:nTrain,['OBV','Dif_MA200','Dif_MA100','Dif_MA20','Dif_MA5']]#'macd','slowk','slowd','RSI','ADX']]
#y_train_tmp = dftotal.loc[:nTrain,'target']
#X_test = dftotal.loc[nTrain+1:,['OBV','Dif_MA200','Dif_MA100','Dif_MA20','Dif_MA5']]#,'macd','slowk','slowd','RSI','ADX']]
#y_test = dftotal.loc[nTrain+1:,'target']
#dftotal=dftotal[(dftotal.MA20 >= dftotal.MA200)]

X_train_tmp = dftotal.iloc[:nTrain,15:-1]
y_train_tmp = dftotal.iloc[:nTrain,-1]
X_test = dftotal.iloc[nTrain+1:,15:-1]
y_test = dftotal.iloc[nTrain+1:,-1]

def grafica(x,y,df,dftgt):
    colors = {0: "Crimson", 1: "RoyalBlue"}
    colores = dftgt.target.map(colors)
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y], color=colores)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.show()

    # 11-dif_ma5 18-dif Ma20
    # 12-dif upper 13- dif_lower 14-RSI 15-MACD 16-MACD sig 17-MACD hist 18-slowk 19-slowd 20-OBD 21-ATR 22-BELTHold


#cc = combinations(dftotal.iloc[:-1,15:-1].columns,2)
#for combination in cc:
#    grafica(combination[0], combination[1], dftotal, dftotal)
# difma_100 / 200 y OBV - rsi y slowk / slowd y OBV - macd y OBV - slowd y OBV

#plt.plot(y_train_tmp[0],y_train_tmp[1],marker="o")
#plt.gcf().set_size_inches(9, 7)
#plt.show()

def entrena(md, mss, msl, rs, X_train_tmp, y_train_tmp):
#    print(y_train_tmp.shape[0],y_train_tmp.sum())

    rus = RandomUnderSampler(random_state=0)
    X_train, y_train = rus.fit_resample(X_train_tmp, y_train_tmp)

#    print(y_train.shape[0],y_train_tmp.sum())

#    nm1 = NearMiss(version=int(rs))
#    X_train, y_train = nm1.fit_resample(X_train_tmp, y_train_tmp)

#    print(y_train.shape[0],y_train_tmp.sum())

#estim = AdaBoostClassifier(
    #    DecisionTreeClassifier(max_depth=int(md)),
    #    n_estimators=int(mss),
    #    learning_rate=rs #,
    #    algorithm="SAMME",
    #)
    #estim.fit(X_train, y_train)

    # Vamos a crear y entrenar un árbol de decisión para clasificar los datos de Iris
    estim = DecisionTreeClassifier(max_depth=int(md), class_weight={0:0.5,1:0.5}, min_samples_split=int(mss), min_samples_leaf=int(msl), random_state=42)  # vamos a usar un árbol de profundidad 2
    estim.fit(X_train, y_train)

    y_score = estim.score(X_test, y_test)

    importancia_predictores = pd.DataFrame(
                                {'predictor': X_train.columns,
                                 'importancia': estim.feature_importances_}
                                )
    #print("Importancia de los predictores en el modelo")
    #print("-------------------------------------------")
    #importancia_predictores.sort_values('importancia', ascending=False)
    #print(importancia_predictores)

    #print("Períodos:",len(y_test.index),"Máximo:",y_test.sum(),'Accuracy:', y_score)
    ny_test=y_test.to_numpy()
    #print("Predecidos:",np.add.reduce(tree.predict(X_test)),"Acertados:",np.add.reduce(np.multiply(ny_test, tree.predict(X_test))),"Porcentaje test",np.add.reduce(np.multiply(ny_test, tree.predict(X_test)))/np.add.reduce(tree.predict(X_test)))

    #print("---Óptimo---")
    #print(np.add.reduce(df1[nTrain:nRow-nPer]))
    #print("Predecidosxhora:",np.add.reduce(tree.predict(X_test))/(nStart*(1-nTrainPer))/24)

    #print("---Prob---")
    #print(tree.predict_proba(X_test)[:,:1])
    return np.add.reduce(np.multiply(ny_test, tree.predict(X_test)))/np.add.reduce(tree.predict(X_test))

def llenaMatriz(desde,hasta,paso):
    n = int((hasta-desde)/paso)
    mat = np.empty(n)
    i=1
    while i <= n:
        mat[i-1] = i*paso
        i=i+1
    return mat

matrizmd = np.array([2, 5, 10])  # llenaMatriz(1,10,2)
matrizmss = np.array([2, 5])  # llenaMatriz(0,16,2)
matrizmsl = np.array([2, 5])  # llenaMatriz(0,16,2)
matrizmrs = np.array([1])  # llenaMatriz(1,2,1)

combinations= it.product(matrizmd,matrizmss,matrizmsl,matrizmrs)

#rowAlt = aux.shape[0]

#for combination in combinations:
#    print(combination,"{0:.3f}".format(entrena(combination[0],combination[1],combination[2],combination[3], X_train_tmp, y_train_tmp)))


#matrizper = np.array([3, 5, 15, 30])
#matrizret = np.array([30, 60, 120, 1440])
#matrizgan = np.array([0.003, 0.005, 0.007, 0.01])
#combinations = it.product(matrizper, matrizret, matrizgan)
#for combination in combinations:
#    print(combination)
#    prueba(combination[0], combination[1], combination[2])

rus = RandomUnderSampler(random_state=10)
X_train, y_train = rus.fit_resample(X_train_tmp, y_train_tmp)

#from sklearn.metrics import roc_auc_score
## Allow a decision tree to grow to its full depth
#clf = DecisionTreeClassifier(random_state=42)
#clf.fit(X_train, y_train)
#
## compute ccp_alpha values
#path = clf.cost_complexity_pruning_path(X_train, y_train)
#ccp_alphas, impurities = path.ccp_alphas, path.impurities
#
## train DT classifier for each ccp_alpha value
#clfs = []
#for ccp_alpha in ccp_alphas:
#    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
#    clf.fit(X_train, y_train)
#    clfs.append(clf)
#
## Plot train and test score for each of the above trained model
#clfs = clfs[:-1]
#ccp_alphas = ccp_alphas[:-1]
#
#train_scores = [roc_auc_score(y_train, clf.predict(X_train)) for clf in clfs]
#test_scores = [roc_auc_score(y_test, clf.predict(X_test)) for clf in clfs]
#
#fig, ax = plt.subplots()
#ax.set_xlabel("alpha")
#ax.set_ylabel("accuracy")
#ax.set_title("AUC-ROC score vs alpha")
#ax.plot(ccp_alphas, train_scores, marker='o', label="train")
#ax.plot(ccp_alphas, test_scores, marker='o', label="test")
#ax.legend()
#plt.show()

estim = DecisionTreeClassifier(max_depth=5,  min_samples_split=200, min_samples_leaf=50, random_state=42)  #class_weight={0:0.6,1:0.4} #, ccp_alpha=0.0016)

estim.fit(X_train, y_train)

from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
dot_data = export_graphviz(estim,
                           feature_names=X_train.columns.values)
graph = graph_from_dot_data(dot_data)
graph.write_png('C:/Users/axelt/Documents/Axel/tree.png')

y_score = estim.score(X_test, y_test)

importancia_predictores = pd.DataFrame(
    {'predictor': X_train.columns,
     'importancia': estim.feature_importances_}
)
print("Importancia de los predictores en el modelo")
print("-------------------------------------------")
importancia_predictores.sort_values('importancia', ascending=False)
print(importancia_predictores)

print("Períodos:",len(y_test.index),"Máximo:",y_test.sum(),'Accuracy:', y_score)
ny_test = y_test.to_numpy()
print("Predecidos:",np.add.reduce(estim.predict(X_test)),"Acertados:",np.add.reduce(np.multiply(ny_test, estim.predict(X_test))),"Porcentaje test",np.add.reduce(np.multiply(ny_test, estim.predict(X_test)))/np.add.reduce(estim.predict(X_test)))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, estim.predict(X_test), normalize='pred'))
print(confusion_matrix(y_test, estim.predict(X_test)))

# print("---Óptimo---")
# print(np.add.reduce(df1[nTrain:nRow-nPer]))
# print("Predecidosxhora:",np.add.reduce(tree.predict(X_test))/(nStart*(1-nTrainPer))/24)

# print("---Prob---")
# print(tree.predict_proba(X_test)[:,:1])

print("---Ganancia asegurada---")

#print((np.add.reduce(np.multiply(df1[nTrain:nRow-nPer], tree.predict(data[nTrain:nRow-nPer,16:23])))*(nTar-nCom)/(nStart*(1-nTrainPer))*30*nPorInvertir)+((np.add.reduce(tree.predict(data[nTrain:nRow-nPer, 16:23]))-np.add.reduce(np.multiply(df1[nTrain:nRow-nPer], tree.predict(data[nTrain:nRow-nPer,16:23]))))*(nSL-nCom)/(nStart*(1-nTrainPer))*30*nPorInvertir))

def envia_correo(fec, monto):
    from_addr = 'axel.tonso@gmail.com'
    to = 'axel.tonso@gmail.com'
    message = fec + '-' + str(monto) + '-' + str(monto*(1+nTar)) + '-' + str(monto*(1+nSL))

    # Reemplaza estos valores con tus credenciales de Google Mail
    username = ''
    password = ''

    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username, password)
    server.sendmail(from_addr, to, message)

    server.quit()

# envia_correo(datetime.fromtimestamp(data[data.shape[0]-1][0]/1000).strftime('%Y-%m-%d %H:%M:%S'),data[data.shape[0]-1,4:4])

nCont = 0
npTest = np.empty([0,5])

while True:
    if keyboard.is_pressed('p'):
        print('se presionó [p]arar!')
        break
    else:
        # print(data.shape)
        # print(datetime.fromtimestamp(data[data.shape[0] - 1][0] / 1000).strftime('%Y-%m-%d %H:%M:%S'))
        if estim.predict(dftotal.iloc[-1:,15:-1]) == 1:
            print('Eureka!')
            print(dftotal.loc[-1:,'date'])
            print(estim.predict_proba(dftotal.iloc[-1:,15:-1]))
            #winsound.Beep(2500, 500)
            nCont = nCont + 1
            npTest = np.append(npTest, [dftotal.loc[-1:,'date'],dftotal.loc[-1:,'close'], dftotal.loc[-1:,'close'] * (1 + nTar),dftotal.loc[-1:,'close'] * (1 + nSL)], axis=0)
            #npTest= np.concatenate((npTest,[[datetime.fromtimestamp(data[data.shape[0]-1][0]/1000).strftime('%Y-%m-%d %H:%M:%S')],[data[data.shape[0]-1,4:5]],[data[data.shape[0]-1,4:5]]*(1+nTar),[data[data.shape[0]-1,4:5]*(1+nSL)]]),axis=0)
            # envia_correo(datetime.fromtimestamp(data[data.shape[0]-1][0]/1000).strftime('%Y-%m-%d %H:%M:%S'),data[data.shape[0]-1,4:4])
            if nCont == 20: break
    end_time = start_time + timedelta(minutes=nInterval)
    start_time = end_time
    #print(end_time-datetime.now())
    time.sleep((end_time-datetime.now()).total_seconds())  # espera en segundos
#    time.sleep(60*nInterval)  # espera en segundos
    #data = recupera_datos(nStart,nInterval,cIntervalUni).to_numpy()  # recupera solo el período más reciente
    #dfshort = recupera_datos(nStart,nInterval,cIntervalUni)
    #dflong = recupera_datos(nStart,1,'h')
    #dftotal = pd.merge(dfshort,dflong,suffixes=('_Short', '_Long'),on='Hora')
    #dftotal = dftotal.drop(["Hora", "date_Long", "open_Long", "high_Long", "low_Long", "close_Long", "volume_Long", "MA5_Long", "MA20_Long", "std_Long", "upper_Long", "lower_Long"], axis=1)  # .head()
    #print(dftotal.columns)

    dftotal=recupera_datos(5,nInterval,cIntervalUni)
    dftotal = dftotal.assign(target='0')

data=dftotal.to_numpy()
print(npTest)
nRow=npTest.shape[0]
nGanPer=0
for x in range (0,nRow):
    y = np.where(data[:,0:1] == npTest[x,0])[0]
    while (data[y,3]-npTest[x,2])/npTest[x,1]<nTar and (data[y,3]-npTest[x,2])/npTest[x,2]>-1*nSL and y<data.shape[0]: y = y + 1
    #print(npTest[x,4],data[y,4],(data[y,4]-npTest[x,0])/npTest[x,0])
    if y<data.shape[0]:
        if (data[y,3]-npTest[x,2])/npTest[x,2]>nTar:nGanPer=nGanPer+nTar #(data[y,3]-npTest[x,2])/npTest[x,2]
        else: nGanPer=nGanPer+nSL #(data[y,3]-npTest[x,2])/npTest[x,2]
print(nGanPer,(npTest[0,0]-npTest[nRow,0]))

# nDispo = client.get_asset_balance(asset='BUSD')
# print('Ini: ', data[data.shape[0]-1,4:4])
# print('TP: ', data[data.shape[0]-1,4:4]*(1+nTar))
# print('SL: ', data[data.shape[0]-1,4:4]*(1+nSL))
# def colocaOrden():
    # buy_order_limit = client.create_order(
    #    symbol='cPar',
    #    side='BUY',
    #    type='LIMIT',
    #    timeInForce='GTC',
    #    quantity = nDispo * nPorInvertir,
    #    price=data[data.shape[0]-1,4:4])

    #    order = client.create_oco_order(
    #        symbol='cPar',
    #        side='SELL',
    #        quantity = nDispo * nPorInvertir,
    #        price=data[data.shape[0]-1,4:4]*(1+nTar),
    #        stopPrice=data[data.shape[0]-1,4:4]*(1+nSL),
    #        stopLimitPrice=data[data.shape[0]-1,4:4]*(1+nSL),
    #        stopLimitTimeInForce='GTC')

# for name, importance in zip(df2.feature_names, tree.feature_importances_):
#    print(name + ': ' + str(importance))

# declare figurec
# fig = go.Figure()

# Candlestick
# fig.add_trace(go.Candlestick(x=data.index,
#                open=data['open'],
#                high=data['high'],
#                low=data['low'],
#                close=data['close'], name = 'market data'))

# Trading indicator
# fig.add_trace(go.Scatter(x=data.index, y= data['MA20'],line=dict(color='blue', width=1.5), name = 'Long Term MA'))
# fig.add_trace(go.Scatter(x=data.index, y= data['MA5'],line=dict(color='orange', width=1.5), name = 'Short Term MA'))

# Show
# fig.show()
