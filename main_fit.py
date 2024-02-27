import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import pickle

# --------------------------------------------------------------------------- FUNÇÕES -----------------------------------------------------------------------------------------------------------------------------------


def avalia_modelo(modelo, x, y, dummy=False, multiclass=False):

    if dummy:
        resultados = cross_validate(modelo, x, y,
                                    scoring=["accuracy"],
                                    cv=10, n_jobs=-1)
        print(f'Modelo: {modelo}')

        media_acuracia = sum(resultados["test_accuracy"])/10
        print(f'Acurácia --> {media_acuracia}')

    if multiclass:
        resultados = cross_validate(modelo, x, y,
                                    scoring=["accuracy", "f1_weighted",
                                             "precision_weighted", "recall_weighted"],
                                    cv=10, n_jobs=-1)
        print(f'Modelo: {modelo}')

        media_acuracia = sum(resultados["test_accuracy"])/10
        media_precisao = sum(resultados["test_precision_weighted"])/10
        media_recall = sum(resultados["test_recall_weighted"])/10
        media_f1 = sum(resultados["test_f1_weighted"])/10
        print(f'Acurácia --> {media_acuracia}')
        print(f'Precisão --> {media_precisao }')
        print(f'Recall --> {media_recall}')
        print(f'F1 --> {media_f1}')
        y_pred = cross_val_predict(modelo, x, y, cv=10)
        report = classification_report(y, y_pred)
        print(report)

    else:
        resultados = cross_validate(modelo, x, y,
                                    scoring=["accuracy", "f1",
                                             "precision", "recall"],
                                    cv=10, n_jobs=-1)
        print(f'Modelo: {modelo}')

        media_acuracia = sum(resultados["test_accuracy"])/10
        media_precisao = sum(resultados["test_precision"])/10
        media_recall = sum(resultados["test_recall"])/10
        media_f1 = sum(resultados["test_f1"])/10
        print(f'Acurácia 2--> {media_acuracia}')
        print(f'Precisão 2--> {media_precisao}')
        print(f'Recall 2--> {media_recall}')
        print(f'F1 2--> {media_f1}')
        y_pred = cross_val_predict(modelo, x, y, cv=10)
        report = classification_report(y, y_pred)
        print(report)

# ---------------------------------------------------------------------------IMPORTAÇÃO DADOS-------------------------------------------------------------------------------------------------------------------------------------------


# IMPORTAÇÃO DO CSV
tabela = pd.read_csv(
    "C:/Users/phmnsilva/Documents/Meneses/Faculdade/2024.1/TCC/codigo/InfracaoTransito_processado.csv", encoding="UTF-8", sep=";")

# tabela = pd.read_csv(
#     "C:/Users/phmnsilva/Documents/Meneses/Faculdade/2024.1/TCC/peleteiro/Dataset Tratado (Colunas Extras).csv", encoding="UTF-8", sep=",")
# ATRIBUIÇÃO A UM DATAFRAME
df = pd.DataFrame(tabela)

# ----------------------------------------DEFINIÇÃO DE VARIAVEIS X E Y ------------------------------------

# definindo quem é Y e Quem nao vai ser X
y = df["grav_tipo"]
x = df.drop(columns=["grav_tipo"])

SEED = 42  # <-------------------- PESQUISAR O Q É !!!!!!!!!!!!!!!!!!!!!!!!!!!!

# scaler = StandardScaler()
# copia = df.copy()
# copia = copia.drop(columns=["grav_tipo"])
# dataset_scaled = scaler.fit_transform(copia)
# dataset_scaled = pd.DataFrame(dataset_scaled, columns=copia.columns)

# ----------------- COM MIN MAX Scaler

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3, random_state=SEED, stratify=y)


# scaler = MinMaxScaler()
# x_treino_scaled = pd.DataFrame(scaler.fit_transform(x_treino), columns = x_treino.columns)
# x_teste_scaled = pd.DataFrame(scaler.fit_transform(x_teste), columns = x_teste.columns)


# sampler = RandomUnderSampler(random_state=SEED)
# x_under, y_under = sampler.fit_resample(dataset_scaled, y)

# ----------------------------------------------------------------------------------TREINO E TESTE-----------------------------------------------------------------------------------------------------------------------------------

# DEFINIÇÃO DE INSTANCIAS DOS MODELOS ML APLICADOS
# knn = KNeighborsClassifier()
# dt = DecisionTreeClassifier(random_state=SEED)
rf = RandomForestClassifier(random_state=SEED)
# mlp = MLPClassifier(random_state=SEED, max_iter=1000)

rf.fit(x_treino, y_treino)


# SAIDA DOS VALORES DE RETORNO DA FUNÇÃO DE METRICAS
# print(avalia_modelo(knn, x_under, y_under, multiclass=True))
# print(avalia_modelo(dt, x_under, y_under, multiclass=True))
# print(avalia_modelo(rf, x_under, y_under, multiclass=True))
# print(avalia_modelo(mlp, x_under, y_under, multiclass=True))

# print(avalia_modelo(knn, x_teste, y_teste, multiclass=True))
# print(avalia_modelo(dt, x_teste, y_teste, multiclass=True))
# print(avalia_modelo(rf, x_teste, y_teste, multiclass=True))
# print(avalia_modelo(mlp, x_teste, y_teste, multiclass=True))


with open('modelo_treinado.pkl', 'wb') as file:
    pickle.dump(rf, file)

print("Arquivo de treino criado!")
