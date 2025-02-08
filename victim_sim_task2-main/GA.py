import numpy as np
import random
from vs.environment import VS
from map import Map
from bfs import *

class GA:
    """
    Representação do Algoritmo Genético
    """
    def __init__(self, map, victims, tlim, method = None, priority = False):
        self.map = map
        print(f"Vítimas: {victims}")
        self.victims = self.init_victims(victims)
        self.tlim = tlim
        self.method = method
        self.priority = priority
        self.best = None
        self.best_cost = None

    def init_victims(self, victims):
        """
        Inicializa as vítimas
        """
        #print("Vítimas: ",victims)
        new_sequence = []
        for k, v in victims.items():
            victim = (v[0][0], v[0][1], v[1][5], k)
            new_sequence.append(victim)
        return new_sequence
    
    def create_population(self, size=200):
        """
        Cria a população inicial, diversas rotas aleatórias

        Recebe a lista de vítimas e retorna uma sequência aleatória
        """
        temp = []
        for i in range(size):
            route = self.victims.copy()
            random.shuffle(route)
            temp.append(route)
        # self.population = temp
        return temp

    def distance(self, route):
        """
        Calcula a distância total de uma rota
        """
        total = 0
        bfs = BFS(self.map)
        route_temp = route.copy()
        route_temp.insert(0, (0, 0))
        #temp.append((0, 0))
        for i in range(len(route_temp)-1):
            pos1 = route_temp[i][:2]
            pos2 = route_temp[i+1][:2]
            plan, cost = bfs.search(pos1, pos2)
            total += cost
            #print(f"From {route[i]} to {route[i+1]} cost: {cost}")
        return total
    
    def distance_manhattan(self, route):
        """
        Calcula a distância total de uma rota utilizando o método de manhattan
        """
        total = 0
        temp = route.copy()

        temp.insert(0, (0, 0))
        #temp.append((0, 0))
        
        for i in range(len(temp)-1):
            total += abs(temp[i][0] - temp[i+1][0]) + abs(temp[i][1] - temp[i+1][1])
        return total

    def fitness_prob(self, population):
        """
        Calcula a probabilidade fitness de cada rota
        """
        fitness = []
        total = 0
        for route in population:
            if self.method == "manhattan":
                dist = self.distance_manhattan(route)
            else:
                dist = self.distance(route)
        
            # Penaliza rotas que não priorizam vítimas de maior severidade
            total_sev = 0
            if self.priority:
                for i in range(len(route)):
                    percentage = (len(route) - i) / (len(route))
                    severity = (percentage) * (route[i][2]/10)
                    total_sev += severity
            
            dist -= total_sev
            fitness_value = 1 / dist if dist != 0 else float('inf')  # Inverso da distância
            total += fitness_value
            fitness.append(fitness_value)
        fitness = [f / total for f in fitness]  # Normalizar aptidões
        return fitness

    def selection(self, population):
        """
        Seleciona os melhores indivíduos baseado na sua probabilidade
        """
        fitness = self.fitness_prob(population)
        temp = []
        for i in range(len(population)):
            temp.append(random.choices(population, weights=fitness)[0])
        population = temp
        return population
    
    def crossover(self, parent1, parent2):
        """
        Implementa uma estratégia de cruzamento simples entre dois pais.
        Entrada:
        1- parent1
        2- parent2 
        Saída:
        1- offspring1
        2- offspring2
        """
        n_victims = len(self.victims) - 1
        cut = round(random.uniform(1, n_victims))
        
        offspring1 = parent1[0:cut]
        offspring1 += [city for city in parent2 if city not in offspring1]
        
        offspring2 = parent2[0:cut]
        offspring2 += [city for city in parent1 if city not in offspring2]
        
        return offspring1, offspring2
    
    def mutation(self, offspring):
        """
        Implementa a estratégia de mutação em um único descendente
        """
        n_cities_cut = len(offspring) - 1
        index_1 = round(random.uniform(0, n_cities_cut))
        index_2 = round(random.uniform(0, n_cities_cut))

        # Troca os elementos nas posições index_1 e index_2
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        
        return offspring

    def get_best(self, population):
        """
        Retorna a melhor rota encontrada

        Esse método está calculando o custo de resgate errado, conversar com o professor para resolver
        """
        rtime = self.tlim
        best = None

        # Remove rotas duplicadas
        #print(f"Population size: {len(population)}")
        unique_routes = []
        for route in population:
            if route not in unique_routes:
                unique_routes.append(route)
        population = unique_routes
        #print(f"Unique Routes: {len(population)}")

        # Encontra a melhor rota
        for route in population:
            if best is None or self.distance(route) < self.distance(best):
                best = route
        
        # Remove vítimas que não podem ser resgatadas (tempo excedido)
        start = (0, 0, 0)
        final_route = []
        for i in range(len(best)):
            bfs = BFS(self.map)

            plan, cost = bfs.search(start[:2], best[i][:2])
            plan_return, return_cost = bfs.search(best[i][:2], (0, 0))
            cost += 1 * self.map.get_difficulty(best[i][:2]) * 1.2                                 # Adiciona o custo de resgatar a vítima (multiplicado pela dificuldade pq sim)

            if rtime - (cost + return_cost) < 0:
                print(f"Time exceeded: {rtime - (cost + return_cost)}")
                final_route = final_route[:-1]
                break
            
            final_route.append(best[i])
            rtime -= cost
            start = best[i]
        
        self.best = final_route
        print(f"Best route: {final_route}")
        return final_route

    def run(self, population_size, n_generations, crossover_per, mutation_per):
        """
        Executa o algoritmo genético
        """
        population = self.create_population(population_size)
        fitness = self.fitness_prob(population)

        parents = []
        for i in range(0, int(crossover_per*population_size)):
            parents += (self.selection(population))

        descendants = []
        for i in range(0, len(parents), 2):
            offspring1, offspring2 = self.crossover(parents[i], parents[i+1])

            mutate_threshold = random.random()
            if mutate_threshold < 1-mutation_per:
                offspring1 = self.mutation(offspring1)
            
            mutate_threshold = random.random()
            if mutate_threshold < 1-mutation_per:
                offspring2 = self.mutation(offspring2)
            
            descendants.append(offspring1)
            descendants.append(offspring2)

        offspring = parents + descendants

        fitness = self.fitness_prob(descendants)

        # Ordena a população de acordo com a aptidão (maior fitness primeiro)
        sorted_index = np.argsort(fitness)[::-1]

        best_index = sorted_index[0:population_size]

        best_offspring = []
        for i in best_index:
            best_offspring.append(offspring[i])

        for i in range(0, n_generations):
            fitness = self.fitness_prob(best_offspring)

            parents = []
            for j in range(0, int(crossover_per*population_size)):
                parents.extend(self.selection(best_offspring))
            
            descendants = []

            for j in range(0, len(parents)-1, 2):
                offspring1, offspring2 = self.crossover(parents[j], parents[j+1])

                mutate_threshold = random.random()
                if mutate_threshold < 1-mutation_per:
                    offspring1 = self.mutation(offspring1)
                
                mutate_threshold = random.random()
                if mutate_threshold < 1-mutation_per:
                    offspring2 = self.mutation(offspring2)

                descendants.append(offspring1)
                descendants.append(offspring2)

            offspring = parents + descendants

            fitness = self.fitness_prob(offspring)
            sorted_index = np.argsort(fitness)[::-1]
            best_index = sorted_index[0:int(population_size*1)]       # Antes era 0.8

            best_offspring = []
            for i in best_index:
                best_offspring.append(offspring[i])
        
        self.get_best(best_offspring)
        self.best_cost = self.distance(self.best)

def main():
    map = Map()
    map.data = {
        (0, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END]),
        (1, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END]),
        (2, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.WALL, VS.WALL, VS.CLEAR, VS.CLEAR, VS.END]),
        (3, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.END, VS.END, VS.WALL, VS.WALL, VS.CLEAR, VS.END]),   
        (0, 1): (1, 1, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END, VS.END]),
        (1, 1): (1, 2, [VS.CLEAR, VS.CLEAR, VS.WALL, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR]),
        (0, 2): (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END, VS.END, VS.END, VS.END]),
        (1, 2): (1, VS.NO_VICTIM, [VS.CLEAR, VS.WALL, VS.CLEAR, VS.END, VS.END, VS.END, VS.CLEAR, VS.CLEAR]),
        (2, 2): (1, VS.NO_VICTIM, [VS.WALL, VS.WALL, VS.CLEAR, VS.END, VS.END, VS.END, VS.CLEAR, VS.CLEAR]),
        (3, 2): (1, 3, [VS.WALL, VS.END, VS.END, VS.END, VS.END, VS.END, VS.CLEAR, VS.WALL]),
    }
    map.draw()
    victims = [(0, 0), (0, 1), (1, 1), (3, 2)]
    ga = GA(map, victims)
    ga.run(8, 200, 0.8, 0.1)

#if __name__ == "__main__":
#    main()