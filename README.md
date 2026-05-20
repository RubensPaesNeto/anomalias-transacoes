 Este projeto foi desenvolvido com o objetivo de estudar técnicas de Machine Learning aplicadas à detecção de fraudes em cartões de crédito. Foram utilizados diferentes modelos de classificação, como Regressão Logística, Random Forest e XGBoost, para analisar transações bancárias e identificar possíveis operações fraudulentas.

O dataset utilizado neste projeto é disponibilizado pelo TensorFlow e pode ser baixado pelo link abaixo:
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Após realizar o download, coloque o arquivo creditcard.csv na mesma pasta dos arquivos Python do projeto.

Para instalar as dependências necessárias, execute:
pip install -r requirements.txt

Depois disso, basta executar um dos modelos:
python main_logistic.py
python random_forest.py
python xgboost_model.py

projeto utiliza bibliotecas como Pandas, Scikit-learn e XGBoost para processamento dos dados, treinamento dos modelos e avaliação dos resultados.