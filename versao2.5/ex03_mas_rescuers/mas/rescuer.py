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
import numpy as np
import matplotlib.pyplot as plt

from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler

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

    def cluster_victims_quadrants(self):
        """
        Agrupa as vítimas separando-as nos quatro quadrantes do mapa, forçando K-Means a usar inicialização fixa.
        """
        if not self.victims:
            return []

        # Extrai as coordenadas (x, y) das vítimas
        victims_list = list(self.victims.items())
        features = np.array([v[0] for _, v in victims_list])  # Coordenadas (x, y)

        # Define centróides iniciais para os quadrantes
        x_min, x_max = features[:, 0].min(), features[:, 0].max()
        y_min, y_max = features[:, 1].min(), features[:, 1].max()

        initial_centroids = np.array([
            [x_max - (x_max - x_min) / 4, y_max - (y_max - y_min) / 4],  # Quadrante 1 (superior direito)
            [x_min + (x_max - x_min) / 4, y_max - (y_max - y_min) / 4],  # Quadrante 2 (superior esquerdo)
            [x_min + (x_max - x_min) / 4, y_min + (y_max - y_min) / 4],  # Quadrante 3 (inferior esquerdo)
            [x_max - (x_max - x_min) / 4, y_min + (y_max - y_min) / 4]   # Quadrante 4 (inferior direito)
        ])

        # Aplica K-Means com centróides fixos
        kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=1, random_state=42)
        cluster_ids = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_

        # Cria dicionários para armazenar os clusters
        clusters = [{} for _ in range(4)]
        for idx, (vid, data) in enumerate(victims_list):
            clusters[cluster_ids[idx]][vid] = data

        # Plotar os clusters e centróides
        plt.figure(figsize=(10, 7))
        cores = ['r', 'g', 'b', 'y']
        for i in range(4):
            pontos = features[cluster_ids == i]
            plt.scatter(pontos[:, 0], pontos[:, 1], color=cores[i], label=f'Cluster {i + 1}')
        plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x', s=100, label='Centroides')
        plt.title('Agrupamento de Vítimas em Quadrantes')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.legend()
        plt.savefig('agrupamento_quadrantes.png')
        plt.close()

        return clusters
    
    def cluster_victims(self, balanced: bool = True) -> List[Dict[Any, Tuple[Tuple[int, int], List[Any]]]]:
        """
        Realiza o agrupamento das vítimas utilizando o algoritmo KMeans, considerando as coordenadas (x, y)
        e a gravidade (severity) presente nos sinais vitais (índice 6). Se o parâmetro 'balanced' for True,
        procura distribuir as vítimas de forma uniforme entre os clusters.

        :param balanced: Booleano que indica se os clusters devem ter tamanhos balanceados.
        :return: Uma lista de 4 dicionários, onde cada dicionário representa um cluster no formato:
                {victim_id: ((x, y), [vital_signals])}.
        """
        # Se não houver vítimas, retorna uma lista vazia
        if not self.victims:
            return []

        # Converte o dicionário de vítimas para listas de IDs e características
        victims_list = list(self.victims.items())
        feature_list = []
        victim_keys = []

        # Para cada vítima, utiliza (x, y) e o valor de severity (índice 6) para formar o vetor de features
        for vid, (position, signals) in victims_list:
            x, y = position
            severity = signals[6]
            feature_list.append([x, y, severity])
            victim_keys.append(vid)

        features = np.array(feature_list)

        # Normaliza os dados
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)



        # Define o número de clusters e aplica o KMeans
        total_clusters = 4
        kmeans = KMeans(n_clusters=total_clusters, init='k-means++', random_state=42)
        cluster_ids = kmeans.fit_predict(features_scaled)
        centroids = kmeans.cluster_centers_

        # Converte os centróides de volta para o espaço original
        original_centroids = scaler.inverse_transform(centroids)

        # Inicializa uma lista de dicionários para armazenar os clusters
        clusters = [{} for _ in range(total_clusters)]

        if balanced:
            total_victims = len(victim_keys)
            # Determina a quantidade ideal de vítimas por cluster
            ideal_counts = [total_victims // total_clusters] * total_clusters
            for i in range(total_victims % total_clusters):
                ideal_counts[i] += 1

            overflow = []  # Armazena vítimas que excedem o tamanho ideal do cluster

            # Primeira alocação: insere cada vítima no cluster sugerido pelo KMeans, se houver espaço
            for idx, vid in enumerate(victim_keys):
                assigned_cluster = cluster_ids[idx]
                if len(clusters[assigned_cluster]) < ideal_counts[assigned_cluster]:
                    clusters[assigned_cluster][vid] = self.victims[vid]
                else:
                    overflow.append((vid, self.victims[vid], features[idx]))

            # Redistribuição: para cada vítima em excesso, procura o cluster mais próximo que ainda não atingiu o limite
            for vid, victim_data, feat in overflow:
                # Calcula a distância da vítima a cada centróide
                dists = [np.linalg.norm(feat - centroids[i]) for i in range(total_clusters)]
                sorted_clusters = np.argsort(dists)
                placed = False
                for cluster_idx in sorted_clusters:
                    if len(clusters[cluster_idx]) < ideal_counts[cluster_idx]:
                        clusters[cluster_idx][vid] = victim_data
                        placed = True
                        break
                if not placed:
                    # Se todos estiverem completos, adiciona àquele com menos vítimas
                    min_cluster = np.argmin([len(c) for c in clusters])
                    clusters[min_cluster][vid] = victim_data
        else:
            # Alocação simples baseada nos rótulos do KMeans
            for idx, vid in enumerate(victim_keys):
                clusters[cluster_ids[idx]][vid] = self.victims[vid]

        # Cria o gráfico dos clusters utilizando apenas as coordenadas (x, y)
        plt.figure(figsize=(10, 7))
        cores = ['r', 'g', 'b', 'y']
        for i in range(total_clusters):
            pontos = features[cluster_ids == i, :2]
            plt.scatter(pontos[:, 0], pontos[:, 1], color=cores[i], label=f'Cluster {i + 1}')
        # Plota os centróides com marcador 'x'
        #plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x', s=100, label='Centroides')
        plt.scatter(original_centroids[:, 0], original_centroids[:, 1], color='k', marker='x', s=100, label='Centroides')
        plt.title('Agrupamento de Vítimas')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.legend()
        plt.savefig('agrupamento_vitimas.png')
        plt.close()

        return clusters
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

            print(f"Total de vítimas únicas: {len(self.victims)}")

            # Agrupa as vítimas em 4 clusters, levando em conta severidade
            clusters_of_vic = self.cluster_victims()

            # Agrupa as vítimas em 4 clusters, em 4 quadrantes fixos.
            #clusters_of_vic = self.cluster_victims_quadrants()

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