# -*- coding: utf-8 -*-

'''Importa Librerias'''
if True:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats

    from sklearn import preprocessing
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn import tree
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

'''Carga ds'''
df0 = pd.read_csv('ds/hotel_bookings.csv')

'''Primera Inspeccion de los ds'''
# print (len(df0))
# print(df0.info())
# print (df0.describe())
# print (df0.head())
# print (df0.columns)
''' Nombre de Variables:
1'hotel', 
2'is_canceled', 
3'lead_time', 
4'arrival_date_year',
5'arrival_date_month', 
6'arrival_date_week_number',
7'arrival_date_day_of_month', 
8'stays_in_weekend_nights',
9'stays_in_week_nights', 
10'adults', 
11'children', 
12'babies', 
13'meal',
14'country', 
15'market_segment', 
16'distribution_channel',
17'is_repeated_guest', 
18'previous_cancellations',
19'previous_bookings_not_canceled', 
20'reserved_room_type',
21'assigned_room_type', 
22'booking_changes', 
23'deposit_type', 
24'agent',
25'company', 
26'days_in_waiting_list', 
27'customer_type', 
28'adr',
29'required_car_parking_spaces', 
30'total_of_special_requests',
31'reservation_status', 
32'reservation_status_date'
'''

'''A)describir los atributos realizando una breve explicación de qué representan y 
del tipo de variable (categórica, numérica u ordinal). En caso de que haya variables no numéricas, 
reportar los posibles valores que toman y cuán frecuentemente lo hacen.
B) Reportar si hay valores faltantes. ¿Cuántos son y en qué atributos se encuentran? 
En caso de haberlos, ¿es necesario y posible asignarles un valor? '''

for col in df0.columns:
    continue
    print(col)
    valcounts = df0[col].value_counts(normalize=True)
    print(valcounts)
    print('Cantidad de ds: ' + str(len(valcounts)))
    # Chequeo si tengo nans
    print('NaN: ' + str(df0[col].isna().sum()))
    print('\n')
    # username = input("Siguiente:")
    try:
        Df_Var = df0[col].sort_values(ascending=False).reset_index()
        plt.title(col)
        plt.plot(Df_Var.index, Df_Var[col], 'o')
        plt.show()
    except:
        continue

'''Completa Faltantes'''
if 1 == 2:
    colcNaN = ['country', 'agent', 'company']
    # Agent: Identificación del agente (si se reserva a través de un agente)
    # company: ID de la empresa (si una cuenta estaba asociada a ella)
    # Country: Identificación ISO del país del titular de la reserva principal.
    for col in colcNaN:
        print(col)
        print('% de faltantes: ', round(df0[col].isna().sum() / len(df0[col]) * 100, 2))
        if col == 'country':
            df0[col] = df0[col].fillna('SinDato')
        if col == 'agent':
            df0[col] = df0[col].fillna('Particular')
        if col == 'company':
            df0[col] = df0[col].fillna('Particular')
            # print(df0[col].isna().sum())
    df0.to_csv('ds/df1.csv', index=False)

    if 1 == 2:  # Otras tecnicas para completar faltantes
        # Puedo reemplazar los nans por el valor promedio de ese atributo
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        year_vector = imp.fit_transform(year_vector)
        print(np.sum(np.isnan(year_vector)))

        # Imputar valores nulos
        X_train[0, 0] = np.nan

        print('Antes de imputar nan: {}'.format(X_train[0, 0]))
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        transformed_values = imputer.fit_transform(X_train)
        print('Después de imputar nan: {}'.format(transformed_values[0, 0]))

'''Transforma ds'''


def transform_jenks_breaks(data, column_name, new_column_name, do_plot):
    print(column_name)
    breaks = jenkspy.jenks_breaks(data[column_name], nb_class=5)
    print(breaks)
    data[new_column_name] = pd.cut(data[column_name], bins=breaks,
                                labels=breaks[1:], include_lowest=True)  # ,retbins=True)
    print(data[new_column_name].value_counts())

    if do_plot:
        plot_column(data, column_name, new_column_name)


def plot_column(data, column_name, new_column_name):
    # Df para Plotear
    df_var = data[column_name].to_frame()
    df_var[new_column_name] = data[new_column_name]
    df_var.sort_values([column_name, ], axis=0, inplace=True)
    df_var = df_var.reset_index()

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Reservas')
    ax1.set_ylabel(column_name, color=color)
    ax1.plot(df_var.index, df_var[column_name], 'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Transformacion', color=color)  # we already handled the x-label with ax1
    ax2.plot(df_var.index, df_var[new_column_name], 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(column_name)
    plt.show()


def transform_reserved_room_type(data):
    column_name = 'reserved_room_type'
    # continue
    print(column_name)
    data['room_change'] = np.nan
    for index, row in data.iterrows():
        if row[column_name] != row['assigned_room_type']:
            if row[column_name] < row['assigned_room_type']:
                data.loc[index, 'room_change'] = 'mejor'
            else:
                data.loc[index, 'room_change'] = 'peor'
        else:
            data.loc[index, 'room_change'] = 'sincambio'


def transform(data, column_name, x, y):
    print(column_name)
    unique_values = data[col].value_counts(normalize=True) * 100
    unique_values_id = unique_values[unique_values > x].index
    df0.loc[~data[col].isin(unique_values_id), col] = y
    unique_values = data[col].value_counts(normalize=True) * 100
    print(unique_values)


def transformations(data):
    # lead_time
    transform_jenks_breaks(data, 'lead_time', 'LT_Transf', True)

    # reserved_room_type
    transform_reserved_room_type(data)

    # adr
    # Cambia el valor de una variable
    # 5400 a 540 porque el siguiente es 510. Perece error de tipeo
    data.loc[data['adr'] > 1000, 'adr'] = 540
    transform_jenks_breaks(data, 'adr', 'adr_Transf', True)

    # 'stays_in_weekend_nights' [0, 1, 2, 3, 4, 'Otro']/ Otro: >4 / Son menos del 0.3%
    transform(data, 'stays_in_weekend_nights', 1.0, 'Otro')

    # 'stays_in_week_nights' [0, 1, 2, 3, 4, 5, 6, 'Otro']/ Otro: >6 / Son menos del 3%
    transform(data, 'stays_in_week_nights', 1.0, 'Otro')

    # 'adults' [1, 2, 3, 'Otro']/ Otro: 0 y >3 / Son menos del 1%
    transform(data, 'adults', 1.0, 'Otro')

    # 'children' [0, 1, 2, 'Otro']/ Otro: 0 y >2 / Son menos del 0.1%
    transform(data, 'children', 1.0, 'Otro')

    # 'babies' [0, 1]/ Con o sin bebe/ ConBebe menos del 1%
    transform(data, 'babies', 1.0, 'Otro')

    # 'previous_cancellations' [0, 1]/ SinAntesCancelo o No / Si cancelo  5.4%
    transform(data, 'previous_cancellations', 6.0, 1)

    # 'previous_bookings_not_canceled' [0, 1]/ Con O Sin Reserva previa / Con Res 3.0%
    transform(data, 'previous_bookings_not_canceled', 1.5, 1)

    # 'booking_changes' [0, 1, Otro]/ Otro: >2 / 1.3%
    transform(data, 'booking_changes', 1.0, 'Otro')

    # 'agent' ['9.0', 'Particular', '240.0', '1.0', '14.0','OtroAgente']/ Otro: <3% / OtroAgente son el 38.8%
    transform(data, 'agent', 3.0, 'OtroAgente')

    # 'company' ['Particular', 'Compania']/ Compania: 5.7%
    transform(data, 'company', 1.0, 'Compania')

    # 'days_in_waiting_list' [0, 1]/ 1 son los que tuvieron al menos un dia de espera: 3.1%
    transform(data, 'days_in_waiting_list', 1.0, 1)

    # 'required_car_parking_spaces' [0, 1]/ 0 No 1 Si 6.2%
    transform(data, 'required_car_parking_spaces', 1.0, 1)

    # 'total_of_special_requests' [0, 1, 2, 'Mas']/ Mas: 2.4%
    transform(data, 'total_of_special_requests', 3.0, 'Mas')


if 1 == 1:
    # conda install -c conda-forge jenkspy
    import jenkspy

    df0 = pd.read_csv('ds/df1.csv')

    ################################
    # hay demasiadas posibles categorías?
    # Nos quedaremos con las más frecuentes y las demás las asignamos a Other.
    # Que otra tecnica/criterio podemos usar?

    transformations(df0)
    df0.to_csv('ds/df2.csv', index=False)

'''
c) ¿Qué variables se correlacionan más con la cancelación de la reserva? 
Para las cuatro más correlacionadas, mostrar un scatter plot en el que 
el eje x corresponda a la variable correlacionada, y el eje y a la cancelación.'''
df0 = pd.read_csv('ds/df2.csv')


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


variable = []
crV = []
CorrR = []

if 1 == 2:  # Calcula correlaciones
    for col in df0.columns:
        if col in ('is_canceled',): continue
        # print (col)
        crVi = cramers_v(df0[col], df0['is_canceled'])
        CorrRi = correlation_ratio(df0[col], df0['is_canceled'])

        variable.append(col)
        crV.append(crVi)
        CorrR.append(CorrRi)

    # dictionary of lists
    dict = {'Variable': variable, 'cram_v': crV, 'corr_ratio': CorrR}
    Df_Corr = pd.DataFrame(dict)
    Df_Corr.to_csv('ds/Df_Corr2.csv')

if 1 == 2:  # Graficos scatter plot / heatmap
    Df_Corr = pd.read_csv('ds/Df_Corr2.csv')
    Df_MasCorr = Df_Corr.sort_values(by='cram_v', ascending=False).head(18).reset_index()
    print(Df_MasCorr)

    # Graficos scatter plot / heatmap
    for index, row in Df_MasCorr.iterrows():
        varAplot = row['Variable']
        DistVal = df0[varAplot].unique()
        print(varAplot + ': ', len(DistVal))
        if len(df0[varAplot].unique()) > 10: continue
        x = df0[varAplot].values
        y = df0['is_canceled'].values
        # sns.set(style="ticks", color_codes=True)
        # sns.catplot(x=varAplot, y='is_canceled', data=df0)

        # plt.scatter(x, y, marker='o')
        voter_tab = pd.crosstab(x, y, margins=False)
        voter_tab.columns = ["0", "1"]
        voter_tab.index = DistVal
        dict = {'x': x, 'is_canceled': y}
        Df_heatmap = pd.DataFrame(dict)
        sns.heatmap(voter_tab, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.title(varAplot)
        plt.show()
    quit()

'''d) El dueño de un hotel les solicita que predigan con cierta antelación si un cliente 
cancelará su reserva. ¿Qué atributos utilizará como variables predictoras? ¿Por qué?
'''
# Analizar correlacion y cantidad de valores de cada variable
# Al final

# Objetivo
'is_canceled',

# En ppio a usar
'hotel', 'lead_time', 'arrival_date_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
'adults', 'children', 'babies', 'meal', 'market_segment', 'distribution_channel', 'is_repeated_guest',
'previous_cancellations', 'previous_bookings_not_canceled', 'reserved_room_type', 'assigned_room_type', 'cambio_room_type',
'booking_changes', 'deposit_type', 'agent', 'company', 'days_in_waiting_list', 'customer_type',
'adr', 'required_car_parking_spaces', 'total_of_special_requests'

# Se Descartan
'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
'country', 'reservation_status', 'reservation_status_date'

'''e) ¿Se encuentra balanceado el conjunto de ds que utilizará para desarrollar el algoritmo 
diseñado para contestar el punto d)? En base a lo respondido, ¿qué métricas de performance reportaría y por qué? 
'''
# Veamos qué tan balanceados están los ds:
# print (df_data['target'].value_counts(normalize=True))

'''f) Suponiendo que al dueño del hotel le importa detectar todas las cancelaciones. 
¿Qué medida de performance utilizaría? Si utiliza Fβ-Score, ¿qué valor de β eligiría?
'''
# Al final

'''g) Implementar el algoritmo introducido en el punto d) utilizando árboles de decisión. En primer lugar, se deberá 
separar un 20% de los ds para usarlos como conjunto de evaluación (test set). El conjunto restante (80%) es el 
de desarrollo y es con el que se deberá continuar haciendo el trabajo. Realizar los siguientes puntos:
1) Armar conjuntos de entrenamiento y validación con proporción 80-20 del conjunto de desarrollo de forma aleatoria. 
Usar 50 semillas distintas y realizar un gráfico de caja y bigotes que muestre cómo varía la métrica elegida en c) 
en esas 50 particiones distintas.
2) Usar validación cruzada de 50 iteraciones (50-fold cross validation). Realizar un gráfico de caja y bigotes que 
muestre cómo varía la métrica elegida en esas 50 particiones distintas.
'''

df0 = pd.read_csv('ds/df2.csv')

# Variables a utilizar
listavar = ['LT_Transf', 'room_change', 'adr_Transf', 'required_car_parking_spaces', 'total_of_special_requests',
            'deposit_type', 'market_segment', 'previous_cancellations']

'''Arregla ds de Entrada al modelo'''
# # Codificar valores categóricos
# # nombres de clases a números. LabelEncoder
# labels = ['Piano','Guitarra','Guitarra','Bateria','Bateria','Piano','Bajo']
# le = preprocessing.LabelEncoder()
# le.fit_transform(labels)


# One hot encoding
ohe = OneHotEncoder()
encoded = ohe.fit_transform(df0[listavar])
# print(ohe.categories_)


# year_vector = df0[''].values[:,np.newaxis]

'''Arma atributos y target'''
# Armo los atributos y target
# x_data = np.concatenate([year_vector,encoded.toarray()],axis=1)
x_data = encoded.toarray()
y_data = df0['is_canceled'].values

features = listavar
target = 'is_canceled'

# df_data[features].values
# df_data[target].values

'''Divide los ds de entrada'''
if 1 == 1:
    # En entrenamiento, validacion y prueba. Hagamos 80, 10, 10.
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    # x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size=0.5,stratify=y_test)

    # Podemos ver ahora como quedo
    print('Entrenamiento: {}'.format(len(x_train)))
    # print('Validacion: {}'.format(len(x_val)))
    print('Prueba: {}'.format(len(x_test)))

    # Tambien podemos ver como quedaron balanceadas las clases en cada split:
    # Ahora los splits tienen mas o menos el mismo balance
    for split_name, split in zip(['Entrenamiento', 'Prueba'], [y_train, y_test]):  # ,'Validacion',y_val
        print('{}: {:.3f}'.format(split_name, pd.Series(split).value_counts(normalize=True)[0]))

''' Entrena el Arbol'''
if 1 == 1:
    arbol_sklearn = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, ccp_alpha=0.01)
    arbol_sklearn.fit(x_train, y_train)

    y_pred_val = arbol_sklearn.predict(x_val)
    print(accuracy_score(y_val, y_pred_val))

    quit()

'''
1) Armar conjuntos de entrenamiento y validación con proporción 80-20 del conjunto de desarrollo de forma aleatoria. 
Usar 50 semillas distintas y realizar un gráfico de caja y bigotes que muestre cómo varía la métrica elegida en c) 
en esas 50 particiones distintas.

Hacer 50 divisiones distintas train-val. Medir accuracy, precision, recall y F1. 
Hacer un gráfico de caja y bigotes de estas métricas.

Comparar las gráficas de dos tipos de árboles:
* Árbol sencillo: profundidad < 10, min_samples_leaf>10, ccp_alpha=0.2
* Árbol complejo: profundidad > 20, min_samples_leaf<5, ccp_alpha=0

¿Cómo varían las distribuciones? ¿Por qué? ¿Para este problema, qué es preferible, un mayor precision o recall?
'''
# Graficar el árbol sencillo con sus decisiones
n_seeds = 50
accs = []
precisions = []
recalls = []
f1s = []

for seed in range(n_seeds):
    x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed,
                                                        stratify=y_train)
    arbol_sklearn = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, ccp_alpha=0.01)
    arbol_sklearn.fit(x_train_, y_train_)

    y_pred_val = arbol_sklearn.predict(x_val)
    accs.append(accuracy_score(y_val, y_pred_val))
    precisions.append(precision_score(y_val, y_pred_val))
    recalls.append(recall_score(y_val, y_pred_val))
    f1s.append(f1_score(y_val, y_pred_val))

all_metrics = accs + precisions + recalls + f1s
metric_labels = ['Accuracy'] * len(accs) + ['Precision'] * len(precisions) + ['Recall'] * len(recalls) + [
    'F1 Score'] * len(f1s)

sns.set_context('talk')
plt.figure(figsize=(15, 8))
sns.boxplot(metric_labels, all_metrics)

'''
h) Graficar el árbol de decisión con mejor performance encontrado en el punto g2). 
Analizar el árbol de decisión armado (atributos elegidos y decisiones evaluadas).
'''

# Plotea el Arbol
# featnames = ['year'] + list(ohe.get_feature_names())
featnames = list(ohe.get_feature_names())

plt.figure(figsize=(20, 10))
plot_tree(arbol_sklearn, feature_names=featnames, filled=True);
plt.show()

# quit()


'''
i) Usando validación cruzada de 10 iteraciones (10-fold cross validation), 
probar distintos valores de α del algoritmo de poda mínima de complejidad de costos 
(algoritmo de poda de sklearn). Hacer gráficos de la performance en validación y entrenamiento 
en función del α. Explicar cómo varía la profundidad de los árboles al realizar la poda con 
distintos valores de α.
'''
### Búsqueda de hiperparámetros y validación cruzada
'''
Ahora queremos entrenar árboles de decisión. Teníamos varios **hiperparámetros** para decidir, 
como la profundidad, función de costo, número de instancias mínimo en una hoja, etc... 
Con sklearn, es posible explorar combinaciones de estos hiperparámetros asi elegimos 
la que mejores resultados den.

Ahora bien, tenemos que elegir una métrica en base a la cual tomar la decisión, 
y tenemos que tener ds en los que evaluar. En lugar de usar splits de 
entrenamiento-validación, vamos a usar k-fold cross-validation.
'''

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

params_grid = {'criterion': ['gini', 'entropy'],
               'max_depth': list(range(1, 30)),
               'ccp_alpha': np.linspace(0, 0.5, 100)}

kfoldcv = StratifiedKFold(n_splits=10)
base_tree = DecisionTreeClassifier()
scorer_fn = make_scorer(f1_score)
randomcv = RandomizedSearchCV(estimator=base_tree, param_distributions=params_grid, scoring=scorer_fn, cv=kfoldcv,
                              n_iter=100)
randomcv.fit(x_train, y_train);

# Podemos ver todas las combinaciones de parámetros y qué puntajes obtuvieron,
# junto a tiempos de entrenamiento y predicción.
pd.DataFrame(randomcv.cv_results_).head()

# Podemos ver cuál fue la mejor combinación de hiperparámetros:
print(randomcv.best_params_)

# Podemos pedir el árbol correspondiente ya entrenado y ver cuán importante es cada atributo
best_tree = randomcv.best_estimator_
feat_imps = best_tree.feature_importances_
print(feat_imps)
# Se puede ver que una gran parte de los atributos fueron ignorados por el árbol y por eso valen cero.
# Es decir, el árbol no tomo ninguna decisión basándose en ellos. Ahora veamos a qué variables corresponden
# los índices con importancia mayor a 0

for feat_imp, feat in sorted(zip(feat_imps, features)):
    if feat_imp > 0:
        print('{}: {}'.format(feat, feat_imp))

quit()
'''
j) Evaluar en el conjunto de evaluación, el árbol correspondiente al α que maximice la performance 
en el conjunto de validación. Comparar con el caso sin poda (α=0)
'''

### Metricas

from sklearn.metrics import confusion_matrix, classification_report

best_model = randomcv.best_estimator_
y_pred = best_model.predict(x_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap='Blues', annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# A veces se utiliza la proporción de instancias en la hoja de la predicción como
# una probabilidad arrojada por el modelo. Si graficamos esas probabilidades para las
# instancias de ambas clases nos da algo asi:

y_scores = best_model.predict_proba(x_test)
out_probs = y_scores[:, 1]

# plt.figure(figsize=(15,7))
# sns.kdeplot(out_probs[y_test==0],shade=True,c='r')
# sns.kdeplot(out_probs[y_test==1],shade=True,c='b')
# plt.show()

# El modelo nunca tiene incertidumbre (las probabilidades siempre estan cerca de 0 o 1, nunca de 0.5).
# Esto es porque justamente todas las hojas quedan bastante puras. Sin embargo, hay errores (curva roja
# donde debia estar azul y curva azul donde debia ser roja). Además se pueden observar picos, esto es
# porque las probabilidades son discretas en vez de continuas. Hay tantas probabilidades posibles como
# hojas tenga el árbol.

from sklearn.calibration import calibration_curve

fraction_of_positives, mean_predicted_value = calibration_curve(y_test, out_probs, n_bins=10)
plt.figure(figsize=(10, 10))
plt.plot(mean_predicted_value, fraction_of_positives)
plt.plot(mean_predicted_value, mean_predicted_value, 'k--')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

# ROC Curve
# http://arogozhnikov.github.io/2015/10/05/roc-curve.html

from sklearn.metrics import roc_curve

fpr, tpr, th = roc_curve(y_test, out_probs)

plt.title('ROC decision tree')
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, fpr, '--k')
plt.show()

'''
k) Para el árbol sin poda, obtener la importancia de los descriptores usando la técnica de eliminación 
recursiva. Reentrenar el árbol usando sólo los 3 descriptores más importantes. Comparar la performance 
en el conjunto de prueba.
'''
