"""
RESCUER AGENT
@Author: Tacla (UTFPR) / Adaptado para retorno dinâmico

Esta versão do agente Rescuer implementa:
- Agrupamento (clustering) de vítimas por quadrantes da região explorada.
- Definição de uma sequência de resgate para os clusters.
- Atribuição de um cluster para cada resgatador.
- Cálculo de trajetos entre pares de vítimas utilizando busca em largura (BFS).

Além disso, integra uma estratégia dinâmica de retorno para evitar que o agente fique sem tempo,
replanejando o caminho para a base se o custo estimado do plano exceder o tempo restante.
"""

import os
import random
import csv
import sys
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod


class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1, clusters=[]):
        """ 
        @param env: referência à instância do ambiente
        @param config_file: caminho absoluto para o arquivo de configuração do agente
        @param nb_of_explorers: número de exploradores a serem aguardados
        @param clusters: lista de clusters de vítimas sob responsabilidade deste agente
        """
        super().__init__(env, config_file)

        # Inicializações específicas do agente resgatador
        self.nb_of_explorers = nb_of_explorers       # número de exploradores para aguardar
        self.received_maps = 0                        # contador dos mapas recebidos
        self.map = Map()                              # o mapa recebido dos exploradores
        self.victims = {}                             # dicionário de vítimas encontradas: [vic_id]: ((x,y), [<vs>])
        self.plan = []                                # plano de ações (lista de pares dx, dy)
        self.plan_x = 0                               # posição x durante a fase de planejamento
        self.plan_y = 0                               # posição y durante a fase de planejamento
        self.plan_visited = set()                       # posições já planejadas
        self.plan_rtime = self.TLIM                   # tempo restante previsto durante o planejamento
        self.x = 0                                    # posição x atual do agente
        self.y = 0                                    # posição y atual do agente
        self.clusters = clusters                      # clusters de vítimas atribuídos ao agente
        self.sequences = clusters                     # sequência de resgate para cada cluster

        # Inicializa o agente no estado IDLE; o estado muda para ACTIVE quando o mapa é recebido
        self.set_state(VS.IDLE)

    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # coordenadas x,y
                vs = values[1]        # lista de sinais vitais
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # coordenadas x,y
                vs = values[1]        # lista de sinais vitais
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self):
        """
        Realiza um agrupamento simples das vítimas por quadrantes da área explorada.
        @returns: lista de clusters (cada cluster é um dicionário com [vic_id]: ((x,y), [<vs>]))
        """
        # Determina os limites inferior e superior para x e y
        lower_xlim = sys.maxsize    
        lower_ylim = sys.maxsize
        upper_xlim = -sys.maxsize - 1
        upper_ylim = -sys.maxsize - 1

        for key, values in self.victims.items():
            x, y = values[0]
            lower_xlim = min(lower_xlim, x)
            upper_xlim = max(upper_xlim, x)
            lower_ylim = min(lower_ylim, y)
            upper_ylim = max(upper_ylim, y)
        
        # Calcula os pontos médios
        mid_x = lower_xlim + (upper_xlim - lower_xlim) / 2
        mid_y = lower_ylim + (upper_ylim - lower_ylim) / 2
        print(f"{self.NAME} ({lower_xlim}, {lower_ylim}) - ({upper_xlim}, {upper_ylim})")
        print(f"{self.NAME} cluster mid_x, mid_y = {mid_x}, {mid_y}")
    
        # Divide as vítimas em quadrantes
        upper_left = {}
        upper_right = {}
        lower_left = {}
        lower_right = {}
        
        for key, values in self.victims.items():
            x, y = values[0]
            if x <= mid_x:
                if y <= mid_y:
                    upper_left[key] = values
                else:
                    lower_left[key] = values
            else:
                if y <= mid_y:
                    upper_right[key] = values
                else:
                    lower_right[key] = values
    
        return [upper_left, upper_right, lower_left, lower_right]

    def predict_severity_and_class(self):
        """
        @TODO: Substituir por um classificador e um regressor para determinar a gravidade e a classe da vítima.
        Nesta implementação, valores aleatórios são atribuídos.
        """
        for vic_id, values in self.victims.items():
            severity_value = random.uniform(0.1, 99.9)          # a ser substituído por um regressor 
            severity_class = random.randint(1, 4)               # a ser substituído por um classificador
            values[1].extend([severity_value, severity_class])  # adiciona ao final da lista de sinais vitais

    def sequencing(self):
        """
        Ordena as vítimas (em cada cluster) pelo valor das coordenadas (ex.: x seguido de y).
        @TODO: Substituir por algoritmo de otimização (por exemplo, Algoritmo Genético) para obter a melhor sequência.
        """
        new_sequences = []
        for seq in self.sequences:   # cada sequência é um dicionário
            seq = dict(sorted(seq.items(), key=lambda item: item[1]))
            new_sequences.append(seq)
        self.sequences = new_sequences

    def planner(self):
        """
        Calcula, de forma offline, o caminho entre vítimas utilizando busca em largura (BFS).
        A partir da base (0,0), gera um plano de movimento (lista de pares dx, dy) para visitar as vítimas e retornar à base.
        """
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)

        # Se não houver sequência atribuída, não há nada a fazer.
        if not self.sequences:
            return

        # Considera apenas a primeira sequência (caso mais simples)
        sequence = self.sequences[0]
        start = (0, 0)  # inicia sempre na base
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan_segment, time_taken = bfs.search(start, goal, self.plan_rtime)
            self.plan += plan_segment
            self.plan_rtime -= time_taken
            start = goal

        # Planeja o retorno à base
        goal = (0, 0)
        plan_segment, time_taken = bfs.search(start, goal, self.plan_rtime)
        self.plan += plan_segment
        self.plan_rtime -= time_taken

    def sync_explorers(self, explorer_map, victims):
        """
        Método chamado pelo agente mestre para unificar o mapa e as informações das vítimas
        enviadas pelos exploradores.
        """
        self.received_maps += 1
        print(f"{self.NAME} Map recebido de um explorador")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} todos os mapas dos exploradores foram recebidos")
            #@TODO: Invocar método para desenhar o mapa, se necessário
            #print(f"{self.NAME} vítimas encontradas:\n{self.victims}")

            # Prediz a gravidade e a classe das vítimas
            self.predict_severity_and_class()

            # Agrupa as vítimas em 4 clusters
            clusters_of_vic = self.cluster_victims()

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i + 1)
  
            # Instancia os demais resgatadores (caso haja)
            rescuers = [None] * 4
            rescuers[0] = self  # o agente mestre é o índice 0

            # Atribui ao mestre o primeiro cluster
            self.clusters = [clusters_of_vic[0]]

            # Instancia os demais resgatadores e atribui um cluster para cada
            for i in range(1, 4):
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]])
                rescuers[i].map = self.map

            # Define as sequências de resgate (neste exemplo, cada agente possui um cluster)
            self.sequences = self.clusters

            # Para cada resgatador, ordena a sequência e calcula o plano de resgate
            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i + 1)
                    else:
                        self.save_sequence_csv(sequence, (i + 1) + j * 10)
                rescuer.planner()
                rescuer.set_state(VS.ACTIVE)

    # ========================
    # Métodos de Retorno Dinâmico
    # ========================

    def tempo_restante(self):
        """
        Retorna o tempo restante para o agente (assume-se que AbstAgent fornece get_rtime()).
        """
        return self.get_rtime()

    def custo_estimado(self, plan):
        """
        Estima o custo para executar o plano atual.
        Para cada passo, se o movimento for diagonal, usa o custo COST_DIAG;
        caso contrário, usa COST_LINE.
        """
        cost = 0
        for (dx, dy) in plan:
            if abs(dx) == 1 and abs(dy) == 1:
                cost += self.COST_DIAG
            else:
                cost += self.COST_LINE
        return cost

    def recalcular_caminho_para_base(self):
        """
        Em situação de emergência (tempo insuficiente ou falha no movimento),
        tenta recalcular um caminho seguro até a base (posição (0,0)) utilizando BFS.
        @returns: um novo plano (lista de movimentos) ou None se não for possível encontrar um caminho.
        """
        bfs_instance = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        start = (self.x, self.y)
        goal = (0, 0)
        plan, time_taken = bfs_instance.search(start, goal, self.get_rtime())
        if plan and len(plan) > 0:
            print(f"{self.NAME}: novo plano calculado para retornar à base.")
            return plan
        else:
            print(f"{self.NAME}: falha ao recalcular caminho para a base!")
            return None

    # ========================
    # Método de Deliberação (Loop Principal)
    # ========================

    def deliberate(self) -> bool:
        """
        Método chamado a cada ciclo de raciocínio quando o agente está ACTIVE.
        Retorna:
          True  -> se ainda há ações a executar.
          False -> se não há mais ações.
        """
        # Se não houver mais ações no plano, encerra
        if not self.plan:
            print(f"{self.NAME} finalizou o plano.")
            return False

        # Verifica se o tempo restante é suficiente para completar o plano atual
        remaining = self.tempo_restante()
        estimated_cost = self.custo_estimado(self.plan)
        if remaining < estimated_cost:
            print(f"{self.NAME}: tempo insuficiente ({remaining}) para completar o plano (custo estimado: {estimated_cost}). Replanejando retorno...")
            novo_plano = self.recalcular_caminho_para_base()
            if novo_plano is not None:
                self.plan = novo_plano
            else:
                print(f"{self.NAME}: não foi possível replanejar um caminho seguro!")
                return False

        # Executa o próximo passo do plano
        dx, dy = self.plan.pop(0)
        resultado = self.walk(dx, dy)
        if resultado == VS.EXECUTED:
            self.x += dx
            self.y += dy
            # Verifica se há vítima na posição atual e, se houver, realiza o resgate
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
        else:
            print(f"{self.NAME}: erro ao executar movimento (dx: {dx}, dy: {dy}). Tentando replanejar retorno...")
            novo_plano = self.recalcular_caminho_para_base()
            if novo_plano is not None:
                self.plan = novo_plano
            else:
                print(f"{self.NAME}: não foi possível replanejar após erro no movimento!")
                return False

        return True
