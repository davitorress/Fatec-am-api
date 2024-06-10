import matplotlib
matplotlib.use('agg')

import io
import os
import math
import random
import base64
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

class GeneticAlgorithm:
    def __init__(self, dataset: dict[str, int]):
        self.children = []
        self.population = []
        self.range = [dataset['rangeMin'], dataset['rangeMax']]
        self.generations = dataset['generations']
        self.childrenSize = dataset['childrenSize']
        self.mutationRate = dataset['mutationRate']
        self.populationSize = dataset['populationSize']


    def avaliar_individuo(self, x1: int, x2: int):
        x1_value = x1 * math.sin(math.sqrt(abs(x1)))
        x2_value = x2 * math.sin(math.sqrt(abs(x2)))
        return 837.9658 - (x1_value + x2_value)
    

    def criar_populacao(self):
        for i in range(self.populationSize):
            x1 = random.uniform(self.range[0], self.range[1])
            x2 = random.uniform(self.range[0], self.range[1])
            fitness = self.avaliar_individuo(x1, x2)
            individual = [x1, x2, fitness]

            self.population.append(individual)


    def selecionar_pai(self):
        popAux = self.population.copy()
        
        pos_cand1 = random.randint(0, len(popAux) - 1)
        pai1 = popAux[pos_cand1]
        popAux.remove(pai1)
        
        pos_cand2 = random.randint(0, len(popAux) - 1)
        pai2 = popAux[pos_cand2]
        
        if (pai1[len(pai1) - 1] < pai2[len(pai2) - 1]):
            return self.population.index(pai1)
        else:
            return self.population.index(pai2)
    

    def realizar_mutacao(self, filho: list):
        x1 = random.randint(0, 100)
        x2 = random.randint(0, 100)

        if (x1 <= self.mutationRate):
            filho[0] = random.uniform(self.range[0], self.range[1])

        if (x2 <= self.mutationRate):
            filho[1] = random.uniform(self.range[0], self.range[1])

        return filho
    

    def reproduzir(self):
        for i in range(self.childrenSize):
            pos_pai1 = self.selecionar_pai()
            pos_pai2 = self.selecionar_pai()

            x1f1 = self.population[pos_pai1][0]
            x2f1 = self.population[pos_pai2][1]
            fitnessf1 = self.avaliar_individuo(x1f1, x2f1)
            
            filho1 = [x1f1, x2f1, fitnessf1]
            filho1 = self.realizar_mutacao(filho1)

            x1f2 = self.population[pos_pai2][0]
            x2f2 = self.population[pos_pai1][1]
            fitnessf2 = self.avaliar_individuo(x1f2, x2f2)

            filho2 = [x1f2, x2f2, fitnessf2]
            filho2 = self.realizar_mutacao(filho2)

            self.children.append(filho1)
            self.children.append(filho2)


    def selecionar_descarte(self):
        popAux = self.population.copy()
        
        pos_cand1 = random.randint(0, len(popAux) - 1)
        pai1 = popAux[pos_cand1]
        popAux.remove(pai1)
        
        pos_cand2 = random.randint(0, len(popAux) - 1)
        pai2 = popAux[pos_cand2]
        
        if (pai1[len(pai1) - 1] >= pai2[len(pai2) - 1]):
            return self.population.index(pai1)
        else:
            return self.population.index(pai2)
    

    def realizar_descarte(self):
        self.population = sorted(self.population, key=lambda x:x[len(x) - 1], reverse=True)

        for i in range(self.childrenSize):
            del self.population[self.selecionar_descarte()]

    
    def verificar_melhor_individuo(self):
        return self.population[len(self.population) - 1]


    def iniciar_execucao(self):
        self.criar_populacao()

        for i in range(self.generations):
            self.children = []
            self.reproduzir()
            self.population.extend(self.children)
            self.realizar_descarte()
            
        return self.verificar_melhor_individuo()


def genetic_algorithm(dataset: dict[str, int]):
    genetic = GeneticAlgorithm(dataset)
    result = genetic.iniciar_execucao()
    
    return {
        'name': "Algoritmo Genético",
        'individual': result,
        'algorithm': '837.9658 - ((x1 * sen(√x1)) + (x2 * sen(√x2)))',
    }


def read_dataset(dataset):
    with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmpfile:
        tmpfile.write(dataset)

    try:
        data = pd.read_csv(tmpfile.name)
    finally:
        os.remove(tmpfile.name)
        
    return data


def knn_algorithm(dataset):
    data = read_dataset(dataset)
    data.dropna(inplace=True)

    class_name = data.columns[-1]
    x = np.array(data.drop(class_name, axis=1))
    y = np.array(data[class_name])

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)

    neighbors = 7
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return {
        'name': 'KNN',
        'accuracy': f"{accuracy * 100:.2f}%",
    }


def decision_tree_algorithm(dataset):
    data = read_dataset(dataset)
    data.dropna(inplace=True)

    class_name = data.columns[-1]
    x = data.drop(class_name, axis=1)
    y = data[class_name]
    x = x.apply(pd.to_numeric)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)

    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    predictions = tree.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    fig, ax = plt.subplots(figsize=(20, 18))
    plot_tree(tree, feature_names=x.columns, class_names=[str(cls) for cls in y.unique()], filled=True, rounded=True, proportion=True, fontsize=16, ax=ax)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        'name': 'Árvore de Decisão',
        'accuracy': f"{accuracy * 100:.2f}%",
        'image': f"data:image/png;base64,{img_base64}",
    }

