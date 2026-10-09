"""Representacoes vetoriais de uma jogada para as baselines (RF e MLP).

Todas partem dos mesmos 13 atributos por jogador e dos mesmos 10 atributos
globais que a GCN recebe (ver GNNTrainer.convert_nx_to_pytorch_geometric).
O que muda e como os 22 jogadores viram um vetor de tamanho fixo:

    mean              media dos atributos dos jogadores (representacao do texto)
    stats             media, desvio-padrao, minimo e maximo de cada atributo
    stats_team        "stats" calculado separadamente para ataque e defesa
    concat_raw        concatenacao na ordem em que os nos aparecem no grafo
                      (ordem arbitraria; controle negativo)
    concat_xy         concatenacao ordenada usando SO os atributos que a GCN ja recebe:
                      os 22 jogadores em ordem de profundidade no sentido do ataque
                      (x, espelhado pela playDirection) e, em empate, por y.
                      Nao usa nenhuma informacao extra
    concat_team_y     concatenacao ordenada: ataque e depois defesa; dentro de
                      cada time, da lateral esquerda para a direita (y no
                      sentido do ataque)
    concat_team_role  concatenacao ordenada: ataque e depois defesa; dentro de
                      cada time, por funcao (QB, RB, WR, TE, OL / DL, LB, CB, S)
                      e depois por y

ATENCAO (comparacao justa): stats_team, concat_team_y e concat_team_role usam o
atributo auxiliar isOffense (club == possessionTeam) para agrupar/ordenar os
jogadores. Ele nao entra como coluna do vetor, mas a posicao no vetor passa a
indicar o lado do jogador, informacao que a GCN nao recebe explicitamente.
mean, stats, concat_raw e concat_xy usam apenas os atributos da GCN.

Nas representacoes concatenadas por time cada time ocupa 11 posicoes; se faltar jogador
a posicao e preenchida com zeros (nao ocorreu nos dados, mas o codigo trata).
"""
import numpy as np

NODE_FEATURES = ['x', 'y', 's', 'a', 'dis', 'o', 'dir', 'height', 'weight',
                 'position', 'club', 'playDirection', 'totalDis']
GLOBAL_FEATURES = ['quarter', 'down', 'yardsToGo', 'absoluteYardlineNumber', 'playClockAtSnap',
                   'possessionTeamPointDiff', 'possessionTeam', 'gameClock', 'offenseFormation',
                   'receiverAlignment']
PLAYERS_PER_TEAM = 11
FIELD_WIDTH = 160 / 3  # 53,3 jardas

REPRESENTATIONS = ['mean', 'stats', 'concat_raw', 'concat_xy', 'stats_team', 'concat_team_y', 'concat_team_role']
# Representacoes que usam apenas os atributos de entrada da GCN (sem isOffense)
STRICT_REPRESENTATIONS = ['mean', 'stats', 'concat_raw', 'concat_xy']

# Ordem das funcoes no ataque e na defesa (posicoes do players.csv)
OFFENSE_ROLE = {'QB': 0, 'RB': 1, 'FB': 1, 'HB': 1, 'WR': 2, 'TE': 3, 'T': 4, 'G': 4, 'C': 4, 'OT': 4, 'OG': 4}
DEFENSE_ROLE = {'DE': 0, 'DT': 0, 'NT': 0, 'DL': 0, 'OLB': 1, 'ILB': 1, 'MLB': 1, 'LB': 1,
                'CB': 2, 'DB': 3, 'FS': 3, 'SS': 3, 'S': 3}
UNKNOWN_ROLE = 9


def extract_play(G):
    """Extrai de um grafo os arrays usados pelas representacoes.

    Os nos sao colocados em ordem de nflId para que o resultado nao dependa da
    ordem de insercao no grafo (que muda conforme a topologia de arestas).
    """
    nodes = sorted(G.nodes())
    feats = np.array([[float(G.nodes[n].get(f, 0)) for f in NODE_FEATURES] for n in nodes])
    raw_order_feats = np.array([[float(G.nodes[n].get(f, 0)) for f in NODE_FEATURES] for n in G.nodes()])
    return {
        'key': (G.graph.get('gameId'), G.graph.get('playId')),
        'node_feats': feats,
        'raw_order_feats': raw_order_feats,
        'is_offense': np.array([int(G.nodes[n].get('isOffense', 0)) for n in nodes]),
        'position_name': [str(G.nodes[n].get('positionName', '')) for n in nodes],
        'global_feats': np.array([float(G.graph.get(f, 0)) for f in GLOBAL_FEATURES]),
        'label': 1 if G.graph['playResult'] == 1 else 0,
    }


def _stats(feats):
    if len(feats) == 0:
        return np.zeros(4 * len(NODE_FEATURES))
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0), feats.min(axis=0), feats.max(axis=0)])


def _x_attack(play):
    """x no sentido do ataque: espelha o campo quando a jogada vai para a esquerda"""
    x = play['node_feats'][:, NODE_FEATURES.index('x')]
    direction = play['node_feats'][:, NODE_FEATURES.index('playDirection')]
    return np.where(direction == 0, 120 - x, x)


def _y_attack(play):
    """y no sentido do ataque: espelha o campo quando a jogada vai para a esquerda"""
    y = play['node_feats'][:, NODE_FEATURES.index('y')]
    direction = play['node_feats'][:, NODE_FEATURES.index('playDirection')]
    return np.where(direction == 0, FIELD_WIDTH - y, y)


def _team_block(play, offense, by_role):
    mask = play['is_offense'] == (1 if offense else 0)
    idx = np.where(mask)[0]
    y = _y_attack(play)[idx]
    if by_role:
        table = OFFENSE_ROLE if offense else DEFENSE_ROLE
        role = np.array([table.get(play['position_name'][i], UNKNOWN_ROLE) for i in idx])
        order = np.lexsort((y, role))  # primeiro funcao, depois y
    else:
        order = np.argsort(y, kind='stable')
    block = play['node_feats'][idx[order]][:PLAYERS_PER_TEAM]
    padded = np.zeros((PLAYERS_PER_TEAM, len(NODE_FEATURES)))
    padded[:len(block)] = block
    return padded.flatten()


def play_to_vector(play, representation):
    feats = play['node_feats']
    if representation == 'mean':
        node_part = feats.mean(axis=0)
    elif representation == 'stats':
        node_part = _stats(feats)
    elif representation == 'stats_team':
        node_part = np.concatenate([_stats(feats[play['is_offense'] == 1]),
                                    _stats(feats[play['is_offense'] == 0])])
    elif representation == 'concat_raw':
        padded = np.zeros((2 * PLAYERS_PER_TEAM, len(NODE_FEATURES)))
        block = play['raw_order_feats'][:2 * PLAYERS_PER_TEAM]
        padded[:len(block)] = block
        node_part = padded.flatten()
    elif representation == 'concat_xy':
        order = np.lexsort((_y_attack(play), _x_attack(play)))  # primeiro x, depois y
        padded = np.zeros((2 * PLAYERS_PER_TEAM, len(NODE_FEATURES)))
        block = feats[order][:2 * PLAYERS_PER_TEAM]
        padded[:len(block)] = block
        node_part = padded.flatten()
    elif representation == 'concat_team_y':
        node_part = np.concatenate([_team_block(play, True, False), _team_block(play, False, False)])
    elif representation == 'concat_team_role':
        node_part = np.concatenate([_team_block(play, True, True), _team_block(play, False, True)])
    else:
        raise ValueError(f"Unknown representation: {representation}")
    return np.concatenate([node_part, play['global_feats']])


def build_matrix(plays, representation):
    X = np.array([play_to_vector(p, representation) for p in plays])
    y = np.array([p['label'] for p in plays])
    return X, y
