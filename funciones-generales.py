import pandas as pd

from numpy import nan
from numpy.random import choice

from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer

from requests import get
from IPython.display import display
from re import compile
from scipy.stats import zscore

#  _____   _   _   _   _    ____   ___    ___    _   _   _____   ____  
# |  ___| | | | | | \ | |  / ___| |_ _|  / _ \  | \ | | | ____| / ___| 
# | |_    | | | | |  \| | | |      | |  | | | | |  \| | |  _|   \___ \ 
# |  _|   | |_| | | |\  | | |___   | |  | |_| | | |\  | | |___   ___) |
# |_|      \___/  |_| \_|  \____| |___|  \___/  |_| \_| |_____| |____/ 
#   ___  _   ___   __  _    ___  _    __      __    ___     __     __    _____    __     __  
#  / _/ | | | __| |  \| |  / _/ | |  /  \    | _\  | __|   | _\   /  \  |_   _|  /__\  /' _/ 
# | \__ | | | _|  | | ' | | \__ | | | /\ |   | v | | _|    | v | | /\ |   | |   | \/ | `._`. 
#  \__/ |_| |___| |_|\__|  \__/ |_| |_||_|   |__/  |___|   |__/  |_||_|   |_|    \__/  |___/
#

def estadisticas_datos(dataframe: pd.core.frame.DataFrame):
    # Genera multiples tablas para visualizar el dataframe.
    print('### HEAD ###')
    display(dataframe.head(10))

    print('\n### TAIL ###')
    display(dataframe.tail(10))

    print('\n### FORMA ###')
    display(dataframe.shape)

    print('\n### TIPOS ###')
    display(dataframe.dtypes)

    print('\n### INFORMACIÓN ###')
    display(dataframe.info())

    print('\n### DESCRIPCIÓN ###')
    display(dataframe.describe(include = "all"))

    print('\n### COLUMNAS CON VALORES FALTANTES ###')
    lista_nulos = obtener_var_con_datos_nulos(dataframe)

    if len(lista_nulos) > 0:
        for columna in lista_nulos:
            num_nulos = dataframe[columna].isnull().sum()

            if num_nulos > 1:
                print(columna + ': ' + str(num_nulos) + ' valores faltantes.')
            else:
                print(columna + ': ' + str(num_nulos) + ' valor faltante.')
    else:
        print('No hay valores nulos en el dataframe.')


def obtener_var_numericas(dataframe: pd.core.frame.DataFrame) -> list:
    # retorna lista con las columnas numericas.
    return dataframe.copy().select_dtypes(include='number').columns.tolist()


def obtener_var_booleanas(dataframe: pd.core.frame.DataFrame) -> list:
    # retorna lista con las columnas booleanas.
    return dataframe.copy().select_dtypes(include='bool').columns.tolist()


def obtener_var_cualitativas(dataframe: pd.core.frame.DataFrame) -> list:
    # retorna lista con las columnas categoricas o con texto.
    return dataframe.copy().select_dtypes(include=['object','category']).columns.tolist()


def obtener_var_con_datos_nulos(dataframe: pd.core.frame.DataFrame) -> list:
    # retorna lista con las columnas con datos nulos.
    return dataframe.copy().columns[dataframe.isnull().sum() > 0].tolist()


def obtener_var_categoricas_con_dos_cat(dataframe: pd.core.frame.DataFrame) -> list:
    # retorna lista con las columnas categoricas con solo dos categorias, útil para convertirlos en bool.
    lista_final = []
    df_temp = dataframe.copy().select_dtypes(include=['object','category'])

    for columna in df_temp.columns:

        if df_temp[columna].dtype == 'object':
            df_temp[columna] = df_temp[columna].astype('category')

        if len(df_temp[columna].cat.categories) == 2:
            lista_final.append(columna)

    return lista_final


def obtener_var_con_correlacion(matriz: pd.core.frame.DataFrame, variable: str, tope: float = 0.5) -> list:
    # Retorna lista con las columnas con correlación con la variable especificada.
    lista_correlacion = []

    for fila in matriz[variable].items():
        if abs(fila[1]) > tope and fila[0] != variable:
            lista_correlacion.append(fila[0])

    return lista_correlacion


def convertir_col_categoria_a_bool(columna: pd.core.series.Series, categoria_verdadera: str) -> pd.core.series.Series:
    # Regresa una serie de pandas booleana.
    # Deben existir solo dos categorias.
    # Ejemplo:
    # dataframe['columna_booleana'] = convertir_col_categoria_a_bool(dataframe['columna_categorica'], 'yes')

    # Crea una serie donde los valores iguales a la categoria especificada sean True y el resto False
    columna_booleana = columna == categoria_verdadera

    # Añade los valores nulos de regreso
    columna_booleana = columna_booleana.mask(columna.isnull())
    #columna_booleana.loc[columna.isnull()] = np.nan # Causa errores pero funciona

    return columna_booleana


def convertir_col_categoria_a_num(columna: pd.core.series.Series, tipo: str = 'float') -> pd.core.series.Series:
    # Regresa una serie de pandas numerica, removiendo las unidades de medida y todo texto.
    # Todos los valores deben tener algún número o se volveran nulos. Los valores con separadores de miles tienen su propio tipo.
    # Tipos válidos: 'int', 'int sep', 'float sep coma', 'float sep punto', 'float'
    # Ejemplo:
    # dataframe['columna_numerica'] = convertir_col_categoria_a_num(dataframe['columna'])
    columna_nueva = columna.copy()
    regex_coma = compile(r'[\,]')

    if tipo == 'int': # Ejemplo: $1200300
        regex_numero = compile(r'-?[0-9]+')
        columna_nueva.loc[columna_nueva.notnull()] = columna_nueva.loc[columna_nueva.notnull()].apply(lambda x: int(regex_numero.search(x).group(0)) if regex_numero.search(x) != None else nan)
        columna_nueva = columna_nueva.astype('Int64') # Nótese la I mayúscula porque en minúscula causa error

    elif tipo == 'int sep': # Ejemplo: $1.200.300
        regex_numero = compile(r'[^0-9\-]')

        # Borra todo caracter excepto "-" y números
        columna_nueva.loc[columna_nueva.notnull()] = columna_nueva.loc[columna_nueva.notnull()].apply(lambda x: int(regex_numero.sub('', x)))
        columna_nueva = columna_nueva.astype('Int64') # Nótese la I mayúscula porque en minúscula causa error

    elif tipo == 'float sep coma': # Ejemplo: $1,200,300.0
        regex_numero = compile(r'[^0-9\-\.]')

        columna_nueva.loc[columna_nueva.notnull()] = columna_nueva.loc[columna_nueva.notnull()].apply(lambda x: float(regex_numero.sub('', x)))
        columna_nueva = columna_nueva.astype('Float64')

    elif tipo == 'float sep punto': # Ejemplo: $1.200.300,0
        regex_numero = compile(r'[^0-9\-\,]')

        columna_nueva.loc[columna_nueva.notnull()] = columna_nueva.loc[columna_nueva.notnull()].apply(lambda x: float(regex_coma.sub('.', regex_numero.sub('', x) )))
        columna_nueva = columna_nueva.astype('Float64')

    else: # Float y otros. Ejemplo: $1200300.0
        regex_numero = compile(r'-?[0-9]+(?:[\,\.][0-9]+)?')

        # busca el número en el texto, si no encuentra nada almacena NaN, si existe entonces intercambia las comas por puntos para hacerlo compatible con Python.
        columna_nueva.loc[columna_nueva.notnull()] = columna_nueva.loc[columna_nueva.notnull()].apply(lambda x: float(regex_coma.sub('.', regex_numero.search(x).group(0))) if regex_numero.search(x) != None else nan)
        columna_nueva = columna_nueva.astype('Float64')

    return columna_nueva


def convertir_nombres_columnas(nombres_columnas: list) -> list:
    # Formatea correctamente el nombre de las columnas del dataframe.
    # Retorna una lista con los nuevos nombres.
    # Intercambia espacios por guion bajo, elimina los espacios al inicio y al final, y se pone en minusculas.
    # Ejemplo:
    # dataframe.columns = convertir_nombres_columnas(dataframe.columns.tolist())
    regex_nombre = compile(r'\s+')
    return list(map(lambda x: regex_nombre.sub('_', x.strip().lower()), nombres_columnas.copy()))


def crear_dataframe_one_hot_encoding(dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    # Regresa dataframe de pandas luego de convertir las variables categoricas en booleanas con dummy variables
    # Debe haber al menos una variable categorica

    dataframe_categorico = dataframe.copy().select_dtypes(include=['object','category'])
    dataframe_no_categorico = dataframe.copy().select_dtypes(exclude=['object','category'])

    dummy_dataframe = pd.concat([dataframe_no_categorico, pd.get_dummies(data=dataframe_categorico)], axis='columns')

    lista_viejas_columnas = dataframe_categorico.columns.tolist()
    lista_nuevas_columnas = list( set(dummy_dataframe.columns) - set(dataframe_no_categorico.columns) )

    # por cada columna antigua,
    for col in lista_viejas_columnas:
        # Busca el patrón 'columna_valor'
        regex_columna = compile(col+r'\S+')
        # añade a la lista las columnas nuevas relacionadas con la columna antigua.
        lista_temp = list(filter(regex_columna.match, lista_nuevas_columnas))

        for col_nueva in lista_temp:
            dummy_dataframe[col_nueva] = dummy_dataframe[col_nueva].astype('bool').mask(dataframe[col].isnull())
            #dummy_dataframe.loc[dataframe[col].isnull(), col_nueva] = np.nan # Causa errores pero funciona

    return dummy_dataframe


def crear_dataframe_imputacion_random(dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    # Realiza una imputación random en un nuevo dataframe y lo retorna.
    df_temp = dataframe.copy()

    # Determinar que columnas tienen datos faltantes
    lista_columnas_faltantes = obtener_var_con_datos_nulos(df_temp)

    for columna in lista_columnas_faltantes:
        # retorna un entero con la cantidad de valores nulos
        cant_faltantes = df_temp[columna].isnull().sum()     

        # retorna una serie con los valores no nulos
        valores_observados = df_temp.loc[df_temp[columna].notnull(), columna]

        # retorna valores random entre los ya observados, los asigna a los valores nulos de la columna actual
        df_temp.loc[df_temp[columna].isnull(), columna] = choice(valores_observados, cant_faltantes, replace = True)

    return df_temp


def crear_columna_regresion_lineal(dataframe: pd.core.frame.DataFrame, columnas_con_correlacion: list, columna_a_predecir: str) -> pd.core.series.Series:
    # Realiza regresión lineal para predecir los valores de una columna.
    # En caso de variables cuantitativas el dataframe debe tener dummies previamente creados.
    # La columna a predecir debe ser cuantitativa; quizás en el futuro se pueda utilizar otro modelo para cualitativas pero no es posible por el momento. (quizás usar binomial)

    # todas las columnas a utilizar (variable objetivo + variables correlacionadas)
    todas_las_columnas_a_utilizar = set(columnas_con_correlacion)
    todas_las_columnas_a_utilizar.add(columna_a_predecir)
    # retorna una lista con las variables correlacionadas
    parametros = list(set(columnas_con_correlacion) - {columna_a_predecir})

    df_reg = dataframe[list(todas_las_columnas_a_utilizar)].astype('Float64').copy()
    df_random1 = crear_dataframe_imputacion_random(df_reg)
    df_random2 = crear_dataframe_imputacion_random(df_reg)

    # en X se especifica el dataframe con solo las columnas completas.
    # en Y se especifica la columna con datos aleatorios que vamos a usar para entrenar.
    modelo = LinearRegression().fit(X = df_random1[parametros], y = df_random1[columna_a_predecir])

    # Retorna la predicción del modelo utilizando como parametro un dataframe con solo las columnas sin datos faltantes.
    # Es asignado a su respectivo valor nulo en el nuevo dataframe.
    df_reg.loc[df_reg[columna_a_predecir].isnull(), columna_a_predecir] = modelo.predict(df_random2[parametros])[df_reg[columna_a_predecir].isnull()]

    return df_reg[columna_a_predecir]


def crear_dataframe_imputacion_multiple(dataframe: pd.core.frame.DataFrame, features: list) -> pd.core.frame.DataFrame:
    # realiza la imputación multiple y regresa el nuevo dataframe.
    df_impu_multiple = dataframe.copy()
    imputer = IterativeImputer(max_iter=10, random_state=0)
    df_impu_multiple[features] = imputer.fit_transform(df_impu_multiple[features])

    return df_impu_multiple


def simplificar_matriz_correlacion(matriz: pd.core.frame.DataFrame, tope: float = 0.5) -> pd.core.frame.DataFrame:
    # Convierte las correlaciones por debajo del tope (por defecto 0.5) a 0.
    nueva_matriz = matriz.copy()

    for columna in nueva_matriz.columns:
        nueva_matriz.loc[nueva_matriz[columna].abs() < tope, columna] = 0

    return nueva_matriz


def normalizar_datos(columna: pd.core.series.Series) -> pd.core.series.Series:
    # Regresa una serie de pandas con los datos normalizados.
    # La serie deben ser númerica.
    # Ejemplo: 
    # dataframe['columna_normalizada'] = normalizar_datos(dataframe['columna'])
    return (columna - columna.min()) / (columna.max() - columna.min())


def obtener_outliers(dataframe: pd.core.frame.DataFrame, columna: str) -> pd.core.frame.DataFrame:
    # Regresa un dataframe con los outliers de la columna especificada (datos atipicos).
    return dataframe.copy()[ abs(zscore(dataframe[columna])) >= 3 ]


def eliminar_outliers(dataframe: pd.core.frame.DataFrame, columna: str) -> pd.core.frame.DataFrame:
    # Regresa un dataframe con los outliers de la columna especificada eliminados (datos atipicos).
    # Recuerda revisar si hay outliers con mostrar_outliers()
    return dataframe.copy()[ abs(zscore(dataframe[columna])) < 3 ]


def obtener_duplicados(dataframe: pd.core.frame.DataFrame, columnas: str|list) -> pd.core.frame.DataFrame:
    # Regresa un dataframe con solo los rows duplicados.
    # Solo las columnas especificadas van a contar para determinar sí dos rows son duplicados.
    return dataframe.copy()[dataframe.duplicated(subset=columnas, keep=False)].sort_values(by=columnas, axis='index')


def eliminar_duplicados(dataframe: pd.core.frame.DataFrame, columnas: str|list) -> pd.core.frame.DataFrame:
    # Regresa un dataframe con los valores duplicados eliminados.
    # Recuerda revisar que se está eliminando con mostrar_duplicados().
    return dataframe.copy().drop_duplicates(subset=columnas, keep='first', inplace=False)


def descargar_datos(url: str, funcion_exito: function) -> pd.core.frame.DataFrame:
    # Descarga datos de una URL, y en caso de ser exitoso, ejecuta una función con esos datos.
    # Retorna el resultado de la función (debe ser un dataframe).
    response = get(url)
    resultado = pd.DataFrame()

    # Verificar que la solicitud fue exitosa (código 200)
    if response.status_code == 200:
        print("Solicitud exitosa.")
        resultado = funcion_exito(response)
    else:
        print(f'Error al acceder a la página. Código de estado: {response.status_code}')

    return resultado


def guardar_datos(dataframe: pd.core.frame.DataFrame, nombre: str = 'dataframe.csv'):
    # guarda el dataframe como un archivo CSV.
    # el nombre puede ir con o sin extensión, es añadida automaticamente.
    regex_format = compile(r'\S+[\.]csv')
    nombre_mod = nombre.strip().lower()

    if regex_format.search(nombre_mod) == None:
        nombre_mod = nombre_mod + '.csv'

    dataframe.to_csv(nombre_mod, index=False, header=True, encoding='utf-8')


def guardar_grafico(grafico, nombre: str = 'grafico.png'):
    # guarda el plot de Seaborn como un archivo PNG.
    # el nombre puede ir con o sin extensión, es añadida automaticamente.
    regex_format = compile(r'\S+[\.]png')
    nombre_mod = nombre.strip().lower()

    if regex_format.search(nombre_mod) == None:
        nombre_mod = nombre_mod + '.png'

    grafico.figure.savefig(nombre_mod)


if __name__ == '__main__':
    print('No intente ejecutar este módulo directamente por favor.')
