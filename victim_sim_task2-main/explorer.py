# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
    
    def __init__(self, env, config_file, resc, direction):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals

        # inicia um grafo, um dicionário, para representar o ambiente
        # no grafo, cada posição é um nó e as arestas são as posições adjacentes
        # nós sem arestas representam obstáculos ou limites do grid
        self.returnbase = False
        self.returncost = 0
        self.graph = {(0,0): []}        # Grafo do ambiente, posições conhecidas
        self.visited = set()            # Conjunto de posições visitadas
        self.stack = [(0,0)]            # Armazena o caminho de retorno
        self.path = [(0,0)]          # Armazena o caminho de retorno para o algoritmo A*
        self.direction = direction

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_next_position(self):
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
    
        # Loop until a CLEAR position is found
        while True:
            # Get a random direction
            direction = random.randint(0, 7)
            # Check if the corresponding position in walls_and_lim is CLEAR
            if obstacles[direction] == VS.CLEAR:
                return Explorer.AC_INCR[direction]
        
    def explore(self):
        # Obter um incremento aleatório para x e y
        dx, dy = self.get_next_position_online()

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()


        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        self.check_and_add_positions()

        # forth and back: go, read the vital signals and come back to the position
        if not self.returnbase:
            cost = self.path_cost()
            self.come_back_astar()
            if cost < self.get_rtime():
                self.explore()
                return True
            else:
                self.returnbase = True

        # hora de voltar para a base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # hora de acordar o resgatador
            # passa as paredes e as vítimas (aqui, elas estão vazias)
            print(f"{self.NAME}: rtime {self.get_rtime()}, invocando o resgatador")
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            #self.resc.go_save_victims(self.map, self.victims)
            #print(self.map.data)
            #input("Press Enter to continue...")
            return False

        if self.path:
            dx, dy = self.path.pop(0)
            dx -= self.x
            dy -= self.y
            self.walk(dx, dy)
            self.x += dx
            self.y += dy

        return True

    def check_and_add_positions(self):
        """
        Verifica se a posição atual é um nó já visitado, se não, marca como visitado.
        Se já foi visitado, então os seus vizinhos já foram adicionados ao grafo, sai da função.
        Se ainda não foi visitado, verifica os vizinhos adjacentes e os adiciona ao grafo do mapa.
        Se forem elegíveis (não estão no grafo, não foram visitados, não são obstáculos e não são limites do grid),
        são adicionados ao grafo.
        """
        # Define a posição atual
        pos = (self.x, self.y)

        if pos in self.visited:
            return

        # Marca a posição atual como visitada
        self.visited.add(pos)

        obstacles = self.check_walls_and_lim()

        # Define os offsets para as posições adjacentes
        offsets = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

        # Verifica se a chave existe no grafo
        if pos not in self.graph:
            # Se não, cria uma nova
            self.graph[pos] = []

        # Verifica cada vizinho
        for offset in offsets:
            x = self.x + offset[0]
            y = self.y + offset[1]
            neighbor_pos = (x, y)

            # Se a posição estiver limpa e não visitada, adicione ao grafo, com as arestas elegíveis (não obstáculos).
            if obstacles[offsets.index(offset)] == 0: 
                # Adiciona o vizinho à lista de adjacência.
                self.graph[pos].append(neighbor_pos)

            # Se uma posição é um obstáculo ou limite de grid, adicionar ao grafo SEM ARESTAS.
            elif obstacles[offsets.index(offset)] in [1, 2] and neighbor_pos not in self.visited:
                continue

    def get_next_position_online(self):
        """ 
        O agente explorador escolhe o próximo movimento com base na exploração DFS online.
        """
        pos = (self.x, self.y)
        neighbors = self.graph[pos]

        # Remove vizinhos visitados
        unvisited = [neighbor for neighbor in neighbors if neighbor not in self.visited]

        # Escolhe o próximo vizinho não visitado em ordem de DFS
        # Garante uma direção diferente para cada explorador
        if unvisited:
            if self.direction == 0:
                unvisited.reverse()
                
            elif self.direction == 2:
                right_neighbor = (self.x + 1, self.y)
                if right_neighbor in unvisited:
                    while unvisited[0] != right_neighbor:
                        unvisited = unvisited[1:] + unvisited[:1]

            elif self.direction == 4:
                bottom_neighbor = (self.x, self.y + 1)
                if bottom_neighbor in unvisited:
                    while unvisited[0] != bottom_neighbor:
                        unvisited = unvisited[1:] + unvisited[:1]

            elif self.direction == 6:
                left_neighbor = (self.x - 1, self.y)
                unvisited.reverse()
                if left_neighbor in unvisited:
                    while unvisited[0] != left_neighbor:
                        unvisited = unvisited[1:] + unvisited[:1]

            next_pos = unvisited[0] 
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y

            # Adiciona o movimento à pilha
            self.stack.append((dx, dy))
            return (dx, dy)

        # Se não houver vizinhos não visitados, volte para o nó anterior, backtracking
        else:
            if(self.stack):
                dx, dy = self.stack.pop()
                dx = dx * -1
                dy = dy * -1
                return (dx, dy)
            return (0,0)
        
    def come_back_astar(self):
        base = (0, 0)
        pos = (self.x, self.y)

        if pos == base:
            return []

        open_set = {pos}
        closed_set = set()
        cost_so_far = {pos: 0}
        heuristic = {pos: self.get_heuristic(pos, base)}
        came_from = {}

        while open_set:
            current = min(open_set, key=lambda node: cost_so_far[node] + heuristic[node])

            if current == base:
                self.path = self.reconstruct_path(came_from, current)
                return self.path

            open_set.remove(current)
            closed_set.add(current)

            neighbors = self.graph.get(current, [])

            for neighbor in neighbors:
                if neighbor in closed_set or neighbor not in self.visited:
                    continue

                tentative_cost = cost_so_far[current] + 1

                if neighbor not in open_set or tentative_cost < cost_so_far[neighbor]:
                    came_from[neighbor] = current
                    cost_so_far[neighbor] = tentative_cost
                    heuristic[neighbor] = self.get_heuristic(neighbor, base)
                    open_set.add(neighbor)

        self.path = None
        print("No path found")
        return None

    def get_heuristic(self, node1, node2):
        """
        Calcula a distância entre dois nós utilizando a heurística de Manhattan.
        """
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstrói o caminho a partir do dicionário came_from e do nó atual.
        Retorna uma lista com os nós do caminho.
        """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path
    
    def path_cost(self):
        """
        Calcula o custo do caminho
        """
        if self.path == None:
            # Se não há um caminho encontrado, retorna custo zero
            print("Nenhum caminho encontrado")
            cost = 0
            return cost
        
        cost = 0
        ini_pos = (self.x, self.y)
        
        for i in range(len(self.path)):
            dx = self.path[i][0] - ini_pos[0]
            dy = self.path[i][1] - ini_pos[1]
            
            difficulty, __, __ = self.map.get((self.path[i][0], self.path[i][1]))
            difficulty = math.ceil(difficulty) * 1.4
            
            if abs(dx) == 1 and abs(dy) == 1:
                cost += self.COST_DIAG * difficulty  # Movimento diagonal
            else:
                cost += self.COST_LINE * difficulty  # Movimento linear

            current_pos = (self.path[i][0], self.path[i][1])
            
            for id, victim in enumerate(self.victims.values()):
                if current_pos == victim[0]:
                    cost += self.COST_READ * difficulty  # Leitura de sinais vitais
            
            ini_pos = self.path[i]
        
        return cost