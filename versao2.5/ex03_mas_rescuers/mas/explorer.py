import math
import logging
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

# Configuração básica do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
FLUXOGRAMA E RACIONAL DO CÓDIGO DO EXPLORER.PY

1. INICIALIZAÇÃO:
   - O agente inicia sua execução na base (posição (0,0)).
   - São criadas e configuradas as estruturas essenciais:
       • Mapa: Utilizado para registrar informações sobre obstáculos, dificuldades de travessia e localização de vítimas.
       • Grafo de Exploração: Representa as posições conhecidas (vértices) e as conexões entre elas (arestas). 
         Isso evita a reexploração desnecessária e facilita o planejamento de rotas.
       • Pilha para Backtracking: Armazena os movimentos realizados, permitindo que o agente retorne (backtrack) 
         caso não encontre novos vizinhos para explorar.
       • Lista de movimentos/rota (self.path): Armazena o caminho calculado pelo algoritmo A* para o retorno à base.
       • Conjunto de posições visitadas (visited) e histórico de movimentos (stack): Auxiliam na organização da exploração.

2. EXPLORAÇÃO (UTILIZANDO DFS):
   - Método “deliberate” é chamado a cada ciclo de decisão.
   - O agente atualiza seu grafo com os vizinhos da posição atual:
       • A estratégia de busca em profundidade (DFS) é empregada para escolher o próximo movimento.
       • DFS permite que o agente avance profundamente em regiões desconhecidas, registrando os nós (posições)
         visitados e suas conexões.
       • Se houver vizinhos não visitados, o agente segue essa direção; caso contrário, recorre ao backtracking
         (usando a pilha) para voltar e explorar outras áreas.
   - Após mover-se, o agente atualiza sua posição e o mapa, e verifica se há vítimas na nova posição, registrando-as se encontradas.

3. PLANEJAMENTO DE RETORNO (UTILIZANDO A*):
   - Em algum ponto, o agente determina que precisa retornar à base – isso ocorre quando o custo estimado do 
     caminho de retorno (calculado via A*) se aproxima ou excede o tempo restante.
   - O método “voltar_a_star” é invocado para calcular uma rota ótima de retorno à base (posição (0,0)):
       • O algoritmo A* é empregado porque utiliza uma função heurística (neste caso, a distância Manhattan) 
         para encontrar o caminho de menor custo.
       • A* permite considerar os custos de movimento (linear ou diagonal) e também incorpora custos adicionais 
         (por exemplo, tempo de leitura de sinais vitais) para determinar a rota.
   - Se o A* não conseguir encontrar um caminho viável, o agente recorre ao retorno manual (backtracking) 
     utilizando a pilha, garantindo assim uma estratégia de fallback robusta.

4. RETORNO E SINCRONIZAÇÃO:
   - Quando o agente atinge a base (ou não há mais movimentos possíveis), ele sincroniza o mapa e as informações 
     das vítimas com o agente Rescuer.
   - Essa sincronização é fundamental para a fase de resgate, permitindo que o agente de resgate utilize os 
     dados coletados durante a exploração.

RACIOCÍNIO POR TRÁS DA ESCOLHA DOS ALGORITMOS:
   - DFS para Exploração:
       • Permite a exploração aprofundada e sistemática do ambiente desconhecido.
       • Facilita a construção do grafo de exploração, prevenindo a reexploração de áreas já visitadas.
   - A* para Retorno:
       • Utiliza uma heurística (distância Manhattan) para encontrar o caminho de menor custo.
       • Garante eficiência no retorno à base, otimizando o tempo e evitando caminhos desnecessariamente longos.
   - Backtracking Manual:
       • Serve como um mecanismo de segurança (fallback) quando A* não encontra um caminho viável.
       • Utiliza a pilha de movimentos previamente armazenados para permitir o retorno ao ponto de partida.

Este fluxo de decisão garante que o Explorer maximize a área explorada enquanto mantém a segurança, 
retornando à base antes que o tempo se esgote, sincronizando as informações cruciais com o agente de resgate.
"""


"""
NOTAS SOBRE AS ESTRUTURAS DE DADOS E TIPOS UTILIZADOS

1. Dictionary (Dic):
   - Um dicionário (dict) é uma coleção de pares chave-valor.
   - Utilizamos dicionários para armazenar dados onde a associação de uma chave a um valor é importante, 
     como no caso do "self.grafo" (que mapeia cada posição conhecida a seus vizinhos) e "self.victims" 
     (que armazena informações sobre vítimas encontradas).
   - Os dicionários permitem buscas rápidas por chave, melhorando eficiência.

2. List:
   - Uma lista é uma coleção ordenada e mutável de elementos.
   - Usamos listas para manter sequências onde a ordem dos itens importa, como "self.path" (a rota calculada pelo A*) 
     ou "self.move_history" (histórico de movimentos planejados para backtracking).


3. Tuple[int, int]:
   - Uma tupla é uma coleção imutável de elementos. A anotação Tuple[int, int] indica que esperamos uma tupla com exatamente 
     dois inteiros, geralmente usados para representar coordenadas (x, y).
   - Tuplas garantem que os valores (como coordenadas) não sejam alterados acidentalmente.

4. deque (do módulo collections):
   - Um deque (double-ended queue) é uma estrutura de dados que permite inserções e remoções eficientes em ambas as extremidades.
   - É uma alternativa mais eficiente do que usar listas quando se tratam operações frequentes de inserção e remoção no final 
     ou no início da coleção, como é comum em operações de pilha ou fila.
   - Optamos por usar "deque" para implementar o controle dos movimentos (backtracking), pois ele oferece desempenho O(1)
     para operações de append/pop.

5. Por que NÃO usamos uma classe Stack personalizada?
   - Embora seja possível implementar uma classe Stack do zero, Python já fornece estruturas nativas (como listas e deque) que 
     implementam essa funcionalidade de forma otimizada.
"""

class Explorer(AbstAgent):
    def __init__(self, env: Any, config_file: str, direcao: int, resc: Any) -> None:
        """
        Inicializa o agente Explorer.
        
        :param env: Referência ao ambiente.
        :param config_file: Caminho para o arquivo de configuração.
        :param direcao: Parâmetro para definir a ordem na exploração.
        :param resc: Referência ao agente de resgate para sincronização.
        """
        super().__init__(env, config_file)
        self.set_state(VS.ACTIVE)
        self.resc = resc

        # Posição atual (inicia na base 0,0)
        self.x: int = 0
        self.y: int = 0

        # Mapa e vítimas
        self.map: Map = Map()
        self.victims: Dict[Any, Tuple[Tuple[int, int], List[Any]]] = {}

        # Estruturas para controle de movimentos
        self.walk_stack: deque[Tuple[int, int]] = deque()  # Movimentos executados (para backtracking)
        self.move_history: List[Tuple[int, int]] = []         # Movimentos planejados (para DFS)

        self.retornarbase: bool = False  # Indica se é hora de retornar

        # Estrutura de grafo e controle de posições
        self.grafo: Dict[Tuple[int, int], List[Tuple[int, int]]] = {(0, 0): []}
        self.visited: set[Tuple[int, int]] = set()
        self.path: List[Tuple[int, int]] = []  # Rota calculada pelo A*

        self.direcao: int = direcao

        # Registra a base no mapa
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def explore(self) -> None:
        """
        Executa a exploração: define o próximo movimento, atualiza a posição e o mapa,
        e verifica a presença de vítimas.
        """
        dx, dy = self.get_prox_posicao_online()
        rtime_before = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_after = self.get_rtime()

        if result == VS.BUMPED:
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())

        if result == VS.EXECUTED:
            self.walk_stack.append((dx, dy))
            self.x += dx
            self.y += dy

            victim_id = self.check_for_victim()
            if victim_id != VS.NO_VICTIM:
                signals = self.read_vital_signals()
                self.victims[signals[0]] = ((self.x, self.y), signals)

            # Calcula a dificuldade com base no tempo gasto
            difficulty = (rtime_before - rtime_after)
            difficulty /= self.COST_LINE if dx == 0 or dy == 0 else self.COST_DIAG
            self.map.add((self.x, self.y), difficulty, victim_id, self.check_walls_and_lim())

    def voltar(self) -> None:
        """
        Realiza o retorno manual (backtracking) utilizando o último movimento armazenado.
        """
        if not self.walk_stack:
            logger.warning("Stack de movimentos vazio; não é possível voltar.")
            return

        dx, dy = self.walk_stack.pop()
        dx, dy = -dx, -dy
        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            logger.info(f"{self.NAME}: Colidiu ao voltar em ({self.x+dx}, {self.y+dy}) - rtime: {self.get_rtime()}")
        elif result == VS.EXECUTED:
            self.x += dx
            self.y += dy

    def deliberate(self) -> bool:
        """
        Método de decisão principal do Explorer.
        
        Em cada ciclo:
          - Atualiza o grafo com os vizinhos da posição atual.
          - Verifica se há tempo para continuar explorando ou se deve retornar.
          - Caso haja rota de retorno via A*, segue-a; caso contrário, usa backtracking.
          - Se na base, sincroniza com o agente de resgate.
        
        :return: True se ainda há ações a realizar; False caso contrário.
        """
        self.verifica_e_add_posicao()

        if not self.retornarbase:
            planned_path = self.voltar_a_star()
            if planned_path is None:
                logger.info(f"{self.NAME}: A* não encontrou rota, utilizando retorno manual.")
                self.voltar()
                return True

            estimated_cost = self.custo_caminho()
            logger.info(f"{self.NAME}: rtime {self.get_rtime()}, custo estimado para retorno {estimated_cost}")
            if estimated_cost < self.get_rtime():
                self.explore()
                return True
            else:
                self.retornarbase = True

        if not self.walk_stack or (self.x == 0 and self.y == 0):
            logger.info(f"{self.NAME}: rtime {self.get_rtime()}, sincronizando com o Rescuer.")
            self.resc.sync_explorers(self.map, self.victims)
            return False

        if self.path:
            next_node = self.path.pop(0)
            dx = next_node[0] - self.x
            dy = next_node[1] - self.y
            result = self.walk(dx, dy)
            if result == VS.EXECUTED:
                self.x += dx
                self.y += dy
        else:
            self.voltar()

        return True

    def verifica_e_add_posicao(self) -> None:
        """
        Verifica se a posição atual já foi visitada e, se não, adiciona-a ao grafo,
        incluindo os vizinhos elegíveis.
        """
        current_pos = (self.x, self.y)
        if current_pos in self.visited:
            return

        self.visited.add(current_pos)
        obstacles = self.check_walls_and_lim()
        self.grafo.setdefault(current_pos, [])

        offsets = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        for off in offsets:
            neighbor = (self.x + off[0], self.y + off[1])
            if obstacles[offsets.index(off)] == 0:
                self.grafo[current_pos].append(neighbor)

    def get_prox_posicao_online(self) -> Tuple[int, int]:
        """
        Retorna o próximo movimento baseado na busca DFS, considerando os vizinhos não visitados.
        """
        current = (self.x, self.y)
        neighbors = self.grafo.get(current, [])
        unvisited = [n for n in neighbors if n not in self.visited]

        if unvisited:
            # Reordena a lista de acordo com a direção desejada
            if self.direcao in (0, 2, 4, 6):
                try:
                    # Em vez de reordenar repetidamente, encontramos o índice desejado:
                    target = {
                        0: None,  # reverso
                        2: (self.x + 1, self.y),
                        4: (self.x, self.y + 1),
                        6: (self.x - 1, self.y)
                    }.get(self.direcao)
                    if target and target in unvisited:
                        index = unvisited.index(target)
                        unvisited = unvisited[index:] + unvisited[:index]
                    elif self.direcao == 0:
                        unvisited.reverse()
                except ValueError:
                    pass

            next_pos = unvisited[0]
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            self.move_history.append((dx, dy))
            return (dx, dy)
        else:
            if self.move_history:
                dx, dy = self.move_history.pop()
                return (-dx, -dy)
            return (0, 0)

    def voltar_a_star(self) -> Optional[List[Tuple[int, int]]]:
        """
        Calcula a rota de retorno até a base (0,0) utilizando o algoritmo A*.
        
        :return: Lista de nós representando a rota ou None se não houver caminho.
        """
        base = (0, 0)
        start = (self.x, self.y)
        if start == base:
            return []

        open_set = {start}
        closed_set = set()
        cost_so_far = {start: 0}
        heuristic = {start: self.calcula_heuristica(start, base)}
        came_from = {}

        while open_set:
            current = min(open_set, key=lambda node: cost_so_far[node] + heuristic[node])
            if current == base:
                self.path = self.reconstroi_caminho(came_from, current)
                return self.path

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.grafo.get(current, []):
                if neighbor in closed_set or neighbor not in self.visited:
                    continue
                tentative_cost = cost_so_far[current] + 1
                if neighbor not in open_set or tentative_cost < cost_so_far.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    cost_so_far[neighbor] = tentative_cost
                    heuristic[neighbor] = self.calcula_heuristica(neighbor, base)
                    open_set.add(neighbor)

        self.path = None
        logger.info("Nenhum caminho encontrado")
        return None

    def calcula_heuristica(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> int:
        """
        Calcula a heurística (distância Manhattan) entre dois nós.
        """
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    def reconstroi_caminho(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstrói a rota a partir do dicionário 'came_from'.
        """
        path: List[Tuple[int, int]] = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path

    def custo_caminho(self) -> float:
        """
        Estima o custo da rota gerada pelo A*, considerando os custos de movimento e leitura de sinais.
        """
        if self.path is None:
            logger.info("Nenhum caminho encontrado")
            return 0

        total_cost = 0.0
        current_pos = (self.x, self.y)
        for node in self.path:
            dx = node[0] - current_pos[0]
            dy = node[1] - current_pos[1]
            difficulty, _, _ = self.map.get(node)
            difficulty = math.ceil(difficulty) * 1.4

            if abs(dx) == 1 and abs(dy) == 1:
                total_cost += self.COST_DIAG * difficulty
            else:
                total_cost += self.COST_LINE * difficulty

            for victim in self.victims.values():
                if node == victim[0]:
                    total_cost += self.COST_READ * difficulty
            current_pos = node
        return total_cost
