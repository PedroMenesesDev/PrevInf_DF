import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle

# --------------------------------------------------------------------------- FUNÇÕES -----------------------------------------------------------------------------------------------------------------------------------


def agrupar_mes(mes):
    if mes in ('Janeiro', 'Fevereiro', 'Março', 'Abril'):
        return 'A'
    elif mes in ('Maio', 'Junho', 'Julho', 'Agosto'):
        return 'B'
    elif mes in ('Setembro', 'Outubro', 'Novembro', 'Dezembro'):
        return 'C'
    else:
        return 'X'


def agrupar_veiculo(veiculo):
    if veiculo in ('Automóvel'):
        return 'A'
    else:
        return 'B'


def reorg_local_rodovia(local):
    if local[:2].upper() == 'DF':
        return local[:6].replace(" ", "").replace("-", "").upper()
    else:
        return 'X'


def agrup_horario(hora):
    trata_hora = hora.replace(":", "")
    int_hora = int(trata_hora)

    if int_hora >= 0 and int_hora < 600:
        return 'MADRUGADA'
    elif int_hora >= 600 and int_hora < 1200:
        return 'MANHA'
    elif int_hora >= 1200 and int_hora < 1800:
        return 'TARDE'
    else:
        return 'NOITE'


def agrup_hora(hora):
    trata_hora = hora.split(":")[0]
    int_hora = int(trata_hora)

    return int_hora


def tratamento_km(km):

    if km == 'Em Branco':
        km_int = 0
    else:
        trata_km = km.replace("KM ", "").replace(
            " ", "").replace(",", ".").upper()
        trata_km = re.sub(r'[^0-9.]', '', trata_km)
        if trata_km:
            km_int = int(round(float(trata_km)))
        else:
            km_int = 0

    return km_int


def trata_referencia(referencia):

    upper_referencia = referencia.upper()

    if upper_referencia == 'EM BRANCO':
        referencia_retorno = 'XXX'
    else:
        referencia_retorno = upper_referencia

    return referencia_retorno

# ---------------------------------------------------------------------------IMPORTAÇÃO DADOS-------------------------------------------------------------------------------------------------------------------------------------------


# IMPORTAÇÃO DO CSV
tabela = pd.read_csv(
    "C:/Users/phmnsilva/Documents/Meneses/Faculdade/2024.1/TCC/codigo/InfracaoTransito.csv", encoding="UTF-8", sep=";")

# ATRIBUIÇÃO A UM DATAFRAME
df = pd.DataFrame(tabela)

# Caminho para salvar o arquivo CSV
caminho_csv_final = "C:/Users/phmnsilva/Documents/Meneses/Faculdade/2024.1/TCC/codigo/InfracaoTransito_processado.csv"

# df = df.drop_duplicates()
# ---------------------------------------------------------------------------TRATAMENTO DOS DADOS-----------------------------------------------------------------------------------------------------------------------------------------

df = df.drop(columns=["tipo_infracao", "descricao", "cometimento", "ano_cometimento",  # muitos valores em branco e desbalanceado
             "auinf_local_complemento", "auinf_local_latitude", "auinf_local_longitude"])  # Essas duas colunas podem ficar sem compor mesmo, ela contem muitos valores em branco e desbalanceado

# Apaga as naturezas de infração q estiver com "Em Branco" pq só me interessa os LEVE, MEDIA, GRAVE e GRAVISSIMA
df = df.drop(df[df['grav_tipo'] == 'Em Branco'].index)
df = df.drop(df[df['tipo_veiculo'] == "Em Branco"].index)
df = df.dropna()  # deleta os registros q estao vazios

# Pega somente a rodovia, tipo somente BR324, BA101 e nesse caso os DF000
df['auinf_local_rodovia'] = df['auinf_local_rodovia'].apply(
    reorg_local_rodovia)
# AGRUPAMENTO SOMENTE PELA HORA, SE 16:21 --> 16, 09:05 --> 09
df['hora_cometimento'] = df['hora_cometimento'].apply(agrup_hora)
# df['tipo_veiculo'] = df['tipo_veiculo'].apply(agrupar_veiculo)
# PEGA SOMENTE O NUMERO DO KM NA RODOVIA, RETORNANDO SOMENTE UM INTEIRO
df['auinf_local_km'] = df['auinf_local_km'].apply(tratamento_km)
# ALGUNS VALORES SAO SEMELHANTES POREM EM MAIUSCULO E MINUSCULO, ASSIM COLOCANDO TUDO EM MAIUSCULO
df['auinf_local_referencia'] = df['auinf_local_referencia'].apply(
    trata_referencia)

# --------------------------------------------------------------------------CODIFICAÇÃO DAS VARIAVEIS CATEGORICAS--------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------LABEL ENCODER-------------------------------------


variaveis_categoricas = ['tipo_infrator', 'tipo_veiculo', 'mes_cometimento',
                         'auinf_local_rodovia', 'auinf_local_km', 'auinf_local_referencia']

# Criando um dicionário de LabelEncoders
label_encoders = {}

for coluna_categorica in variaveis_categoricas:
    # Treinamento do LabelEncoder para cada variável categórica
    encoder = LabelEncoder()
    df[coluna_categorica] = encoder.fit_transform(df[coluna_categorica])

    # Salvando o LabelEncoder correspondente a cada variável categórica
    label_encoders[coluna_categorica] = encoder
    joblib.dump(encoder, f'label_encoder_{coluna_categorica}.pkl')

# Salvando o DataFrame modificado em um novo arquivo CSV
df.to_csv(caminho_csv_final, index=False, encoding="UTF-8", sep=";")

# Salvando o dicionário de LabelEncoders para uso futuro
joblib.dump(label_encoders, 'label_encoders_dict.pkl')

print(f"O ETL foi salvo no arquivo: {caminho_csv_final}")


# -----------------------------------------------------------------------------------------
# # Parte de codificação de variaveis, transformando letras em variaveis int
# encoder = LabelEncoder()

# df["mes_cometimento"] = encoder.fit_transform(df["mes_cometimento"])
# df["tipo_infrator"] = encoder.fit_transform(df["tipo_infrator"])
# df["tipo_veiculo"] = encoder.fit_transform(df["tipo_veiculo"])
# df["auinf_local_rodovia"] = encoder.fit_transform(df["auinf_local_rodovia"])
# df["auinf_local_km"] = encoder.fit_transform(df["auinf_local_km"])
# df["auinf_local_referencia"] = encoder.fit_transform(df["auinf_local_referencia"])

# joblib.dump(encoder, 'label_encoder.pkl')
# # with open('label_encoder.pkl', 'wb') as file:
# #     pickle.dump(encoder, file)


# # df = df.drop_duplicates()  # deleta dados duplicados

# Salvando o DataFrame modificado em um novo arquivo CSV
# df.to_csv(caminho_csv_final, index=False, encoding="UTF-8", sep=";")

# print(f"O ETL foi salvo no arquivo: {caminho_csv_final}")
