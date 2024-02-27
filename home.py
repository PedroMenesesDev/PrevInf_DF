import streamlit as st
import re
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# --------------------------------------------------------------------------- FUNÇÕES -----------------------------------------------------------------------------------------------------------------------------------


def reorg_local_rodovia(local):
    if local[:2].upper() == 'dados':
        return local[:6].replace(" ", "").replace("-", "").upper()
    else:
        return 'X'


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


def fazer_previsao(modelo, dados):

    dados['auinf_local_rodovia'] = dados['auinf_local_rodovia'].apply(
        reorg_local_rodovia)
    dados['hora_cometimento'] = dados['hora_cometimento'].apply(agrup_hora)
    dados['auinf_local_km'] = dados['auinf_local_km'].apply(tratamento_km)
    dados['auinf_local_referencia'] = dados['auinf_local_referencia'].apply(
        trata_referencia)

    variaveis_categoricas = ['mes_cometimento', 'tipo_infrator', 'tipo_veiculo',
                             'auinf_local_rodovia', 'auinf_local_km', 'auinf_local_referencia']
    label_encoders = {}

    for coluna_categorica in variaveis_categoricas:
        encoder = joblib.load(f'label_encoder_{coluna_categorica}.pkl')
        dados[coluna_categorica] = encoder.transform(dados[coluna_categorica])
        label_encoders[coluna_categorica] = encoder

    dados_preparados = dados.copy()
    previsao = modelo.predict(dados_preparados)
    return previsao


def nivel_alerta(natureza_infracao):
    if natureza_infracao == 'Leve':
        alerta = f"""
                    Alerta para as seguintes infrações na localidade informada: 

                    1 - Transitar com o veículo na faixa ou pista da direita, regulamentada como de circulação exclusiva para determinado tipo de veículo, 
                    exceto para acesso a imóveis lindeiros ou conversões à direita;
                    2 - Dirigir sem atenção ou sem os cuidados indispensáveis à segurança;
                    3 - Estacionar nos acostamentos, salvo motivo de força maior;
                    Medida administrativa - Remoção do veículo

                    Penalidade - Multa R$ 88,38 e 3 pontos na CNH"""
    elif natureza_infracao == 'Média':
        alerta = f"""
                    Alerta para as seguintes infrações na localidade informada:   

                    1 - Transitar em velocidade superior à máxima permitida quando a velocidade for superior à máxima em até 20% (vinte por cento);                    
                    2 - Conduzir o veículo com defeito no sistema de iluminação, de sinalização ou com lâmpadas queimadas;                                       
                    3 - Quando o veículo estiver em movimento deixar de manter acesa a luz baixa durante a noite;

                    Penalidade - Multa R$ 130,16 e 4 pontos na CNH"""

    elif natureza_infracao == 'Grave':
        alerta = f"""
                    Alerta para as seguintes infrações na localidade informada: 
                    
                    1 - Transitar em velocidade superior à máxima permitida quando a velocidade for superior à máxima em mais de 20% (vinte por cento) até 50% (cinqüenta por cento);
                    2 - Transitar com o veículo na faixa ou pista da esquerda regulamentada como de circulação exclusiva para determinado tipo de veículo;
                    3 - Deixar o condutor ou passageiro de usar o cinto de segurança, conforme previsto no art. 65;                    
                    Medida administrativa - retenção do veículo até colocação do cinto pelo infrator.
                    
                    Penalidade - Multa R$ 195,23 e 5 pontos na CNH"""
    else:
        alerta = f"""
                    Alerta para as seguintes infrações na localidade informada: 
                    
                    1 - Dirigir o veículo com apenas uma das mãos, no caso de o condutor estar segurando ou manuseando telefone celular;
                    2 - Avançar o sinal vermelho do semáforo ou o de parada obrigatória;
                    3 - Transitar com o veículo em calçadas, passeios, passarelas, ciclovias, ciclofaixas, ilhas, refúgios, ajardinamentos, canteiros centrais e divisores de pista de rolamento, acostamentos, marcas de canalização, gramados e jardins públicos:\n
                    
                    Penalidade - Multa R$ 293,47 (três vezes) e 7 pontos na CNH"""
    return alerta


def main():
    st.set_page_config(page_title="PrevInf-DF")
    st.title("Modelo de Previsão de Risco de Infração do DF")
    st.write("---")

    valor5 = st.text_input("Digite a Rodovia:")

    col1, col2 = st.columns(2)

    # Campos de entrada para os dados
    with col1:
        valor6 = st.text_input("Digite o KM:")

        mes_cometimento_opcoes = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio",
                                  "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]

        valor3 = st.selectbox("Escolha o Mês:", mes_cometimento_opcoes)
        tipo_infrator_opcoes = ["Condutor", "Proprietário", "Emb/Transp",
                                "Servidor Público", "Pessoa Física", "Pessoa Jurídica"]

        valor1 = st.selectbox("Escolha o tipo_infrator:", tipo_infrator_opcoes)

    with col2:
        valor7 = st.text_input("Digite a Referencia:")

        hora_cometimento = ("00:00", "00:15", "00:30", "00:45", "01:00", "01:15", "01:30", "01:45", "02:00", "02:15", "02:30", "02:45", "03:00", "03:15", "03:30", "03:45", "04:00", "04:15", "04:30",
                            "04:45", "05:00", "05:15", "05:30", "05:45", "06:00", "06:15", "06:30", "06:45", "07:00", "07:15", "07:30", "07:45", "08:00", "08:15", "08:30", "08:45", "09:00", "09:15",
                            "09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30",
                            "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00", "15:15", "15:30", "15:45", "16:00", "16:15",
                            "16:30", "16:45", "17:00", "17:15", "17:30", "17:45", "18:00", "18:15", "18:30", "18:45", "19:00", "19:15", "19:30", "19:45", "20:00", "20:15", "20:30", "20:45", "21:00",
                            "21:15", "21:30", "21:45", "22:00", "22:15", "22:30", "22:45", "23:00", "23:15", "23:30", "23:45")
        valor4 = st.selectbox("Digite o Horario:", hora_cometimento)

        tipo_veiculo_opcoes = ["Automóvel", "Caminhonete", "Camioneta", "Motocicleta", "Utilitário", "Caminhão", "Caminnhão Trator", "Semi-Reboque",
                               "Microônibus", "Ônibus", "Motoneta", "Reboque", "Ciclomotor", "Triciclo", "Motorcasa", "Trator de Rodas", "Chassi Plataforma",
                               "Side-Bar", "Trator Misto", "Bicicleta", "Charrete", "Carro-de-mão", "Bonde", "Trator de Esteira", "Quadriciclo"]
        valor2 = st.selectbox(
            "Escolha o tipo do veiculo:", tipo_veiculo_opcoes)

    # Criando um DataFrame com os dados inseridos
    dados = pd.DataFrame({
        "tipo_infrator": [valor1],
        "tipo_veiculo": [valor2],
        "mes_cometimento": [valor3],
        "hora_cometimento": [valor4],
        "auinf_local_rodovia": [valor5],
        "auinf_local_km": [valor6],
        "auinf_local_referencia": [valor7]
    })

    # Carregar o modelo treinado
    with open('modelo_treinado.pkl', 'rb') as file:
        modelo = pickle.load(file)

    st.write(f"\n")
    # Fazer a previsão com o modelo treinado
    if st.button("Fazer Previsão"):
        # return previsao
        resultado_previsao = fazer_previsao(modelo, dados)
        st.write(nivel_alerta([resultado_previsao][0]))


if __name__ == "__main__":
    main()
