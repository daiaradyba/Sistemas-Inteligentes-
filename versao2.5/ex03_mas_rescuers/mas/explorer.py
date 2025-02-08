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
    MAX_DIFFICULTY = 1

    def __init__(self, env, config_file, resc, initial_direction):
        super().__init__(env, config_file)
        self.walk_stack = Stack()  # Stack to store movements
        self.visited = set()      # Set to track visited cells
        self.walk_time = 0        # Time consumed walking
        self.set_state(VS.ACTIVE)
        self.resc = resc
        self.x = 0                # Current x position
        self.y = 0                # Current y position
        self.map = Map()          # Map to represent the environment
        self.victims = {}         # Dictionary of found victims
        self.plan = []

        # Mark the base as visited and add it to the map
        self.visited.add((self.x, self.y))
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

        self.initial_direction = initial_direction  # Direção inicial
        self.moved_initially = False  # Flag para controlar o movimento inicial

        # Define o ID do agente com base na direção inicial
        if initial_direction == (1, 1):
            self.agent_id = 1
        elif initial_direction == (1, -1):
            self.agent_id = 2
        elif initial_direction == (-1, 1):
            self.agent_id = 3
        elif initial_direction == (-1, -1):
            self.agent_id = 4
        else:
            raise ValueError(f"Direção inicial inválida: {initial_direction}")

        self.exploration_mode = True  # Flag para alternar entre a exploração inicial e o DFS

    def get_neighbors(self):
        """ Returns a list of neighboring positions that are CLEAR and not visited. """
        neighbors = []
        obstacles = self.check_walls_and_lim()
        for direction, (dx, dy) in Explorer.AC_INCR.items():
            if obstacles[direction] == VS.CLEAR:
                neighbor = (self.x + dx, self.y + dy)
                if neighbor not in self.visited:
                    neighbors.append((dx, dy))
        return neighbors

    def explore(self):
        if self.exploration_mode:
            # Exploração inicial na direção pré-definida
            dx, dy = self.initial_direction
            rtime_bef = self.get_rtime()  # Tempo antes do movimento
            result = self.walk(dx, dy)
            rtime_aft = self.get_rtime()  # Tempo após o movimento

            if result == VS.EXECUTED:
                self.x += dx
                self.y += dy
                self.visited.add((self.x, self.y))
                self.walk_stack.push((dx, dy))
                # Atualizar walk_time
                self.walk_time += (rtime_bef - rtime_aft)

            else:
                # Se não puder continuar na direção inicial, troca para o DFS
                self.exploration_mode = False
            return

        # DFS específico após atingir o limite inicial
        neighbors = self.get_neighbors()

        if neighbors:
            # Escolhe a próxima posição a explorar com base no agente
            if self.agent_id in [1, 4]:  # DFS coluna
                neighbors.sort(key=lambda d: d[0])  # Prioriza movimentos verticais
            else:  # DFS linha
                neighbors.sort(key=lambda d: d[1])  # Prioriza movimentos horizontais

            dx, dy = neighbors[0]
            result = self.walk(dx, dy)
            if result == VS.EXECUTED:
                self.walk_stack.push((dx, dy))
                self.x += dx
                self.y += dy
                self.visited.add((self.x, self.y))
                seq = self.check_for_victim()
                if seq != VS.NO_VICTIM:
                    vs = self.read_vital_signals()
                    self.victims[vs[0]] = ((self.x, self.y), vs)

        else:
            # Backtracking quando não há vizinhos desvisitados
            if not self.walk_stack.is_empty():
                dx, dy = self.walk_stack.pop()
                self.walk(-dx, -dy)
                self.x -= dx
                self.y -= dy

    def come_back(self):
        """ Moves the agent back to the base by retracing its steps using the stack. """
        if not self.walk_stack.is_empty():
            dx, dy = self.walk_stack.pop()  # Retira o último movimento
            dx = -dx  # Inverte a direção
            dy = -dy  # Inverte a direção

            # Realiza o movimento
            result = self.walk(dx, dy)
            if result == VS.EXECUTED:
                # Atualiza a posição atual
                self.x += dx
                self.y += dy

                # Marca a célula como visitada
                self.visited.add((self.x, self.y))
            else:
                print(f"Failed to walk back to ({self.x + dx}, {self.y + dy}).")
        else:
            print(f"{self.NAME}: No moves left to return.")

    def deliberate(self) -> bool:
        """ The agent chooses the next action. """

        # Distância e tempo estimado para retorno
        distance_to_base = abs(self.x) + abs(self.y)
        estimated_return_time = distance_to_base * Explorer.MAX_DIFFICULTY * self.COST_DIAG

        # Logs de depuração
        print(
            f"{self.NAME}: Walk time: {self.walk_time}, Remaining time: {self.get_rtime()}, Estimated return: {estimated_return_time}")

        # Iniciar retorno se o tempo restante for insuficiente
        if self.get_rtime() - estimated_return_time <= 2500:
            if not self.walk_stack.is_empty():
                self.come_back()

                return True
            if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
                print(f"{self.NAME}: At base or no moves left. Synchronizing data.")
                self.resc.sync_explorers(self.map, self.victims)
                return False

        # Continuar explorando
        self.explore()
        return True
