import math
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class GeneticAlgorithm:
    def __init__(self, dataset: dict):
        self.children = []
        self.population = []
        self.range: list[int] = [dataset['rangeMin'], dataset['rangeMax']]
        self.generations: int = dataset['generations']
        self.childrenSize: int = dataset['childrenSize']
        self.mutationRate: int = dataset['mutationRate']
        self.populationSize: int = dataset['populationSize']


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


def genetic_algorithm(dataset: dict):
    print(dataset)
    genetic = GeneticAlgorithm(dataset)
    result = genetic.iniciar_execucao()
    
    return {
        'individual': result,
        'algorithm': '837.9658 - ((x1 * math.sin(math.sqrt(abs(x1)))) + (x2 * math.sin(math.sqrt(abs(x2)))))'
    }


def knn_algorithm(dataset):
    print(dataset)
    data = pd.read_csv(dataset)
    data.dropna(inplace=True)

    print(data)

    x = np.array(data.iloc[:, :-1])
    target = np.array(data['target'])
    class_column = np.array(data['class'])

    print(target, class_column)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.3, train_size=0.7)

    neighbors = 7
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)

    accuracy = accuracy_score(y_test, predictions) * 100
    return accuracy
