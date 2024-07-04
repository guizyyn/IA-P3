from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Carregamento e busca da base de dados

statlog_german_credit_data = fetch_ucirepo(id=144)

#Dados com DataFrames pandas

x = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

#Metadados

print(statlog_german_credit_data.metadado)

#informações das varáveis

print(statlog_german_credit_data.variables)

#ajusta a coluna de destino

y = (y - 1).values.ravel()

#Identificação das colunas e numericas e categoricas

num_cols = x.select_dtypes(include=['int64', 'float64']).columns
cat_cols = x.select_dtypes(include=['object']).columns

# Pre Processamento

numeric_transformer = StandardScaler()
categorical_transfomer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transfomer, cat_cols)
    ]
)

x_preprocessed = preprocessor.fit_transform(x)

#configuração do modelo

model = RandomForestClassifier(random_state=42)

#configuração da validação cruzada

cv = StratifiedKFold(n_splits=5)

#avaliação do modelo com a validação cruzada

scores = cross_val_score(model, x_preprocessed, y, cv=cv, scoring='accuracy')

print(f'Acurácias nas dobras: {scores}')
print(f'Acurácia média: {scores.mean()}')

#definição da grade dos hiperparâmetros

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#config da busca em grade

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

#execução da busca em grade

grid_search.fit(x_preprocessed, y)

print(f'Melhores hiperparâmetros: {grid_search.best_params_}')
print(f'Melhor acurácia: {grid_search.best_score_}')

#treinamento do modelo final com os melhores hiperparâmetros

best_model = grid_search.best_estimator_
best_model.fit(x_preprocessed, y)

#Previsões e avaliações

y_pred = cross_val_predict(best_model, x_preprocessed, y, cv=cv)

print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))

#Conclusão: temos o resultado das avaliações dos cinco subconjuntos(Acurácias nas dobras) e a media(Acurácia média) delas, a variabilidade
#dos resultados da avaliação dos subconjuntos nos diz que ele tem uma consistência na sua performace, e
#a sua media podemos tomar ela como uma media razóavel