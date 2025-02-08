# Map Class
# @Author: Cesar A. Tacla, UTFPR
#
## A map representing the explored region of the 2D grid
## The map is a dictionaire whose keys are pairs (x, y).
## The map contains only visited positions.
##
## An entry in the dictionary is: [(x,y)] : ( difficulty, vic_id, [actions' results] )
## - (x,y): the key; the position of the grid (or the cell)
## - difficulty: the degree of difficulty to access the cell
## - vic_id: the victim identification number (if there is one) or VS.NO_VICTIM if there is no victim
## - actions' results: the known actions' results from the cell represented as vector of 8 integers, in the following
##   order: [up, up-right, right, down-right, down, down-left, left, up-left]. Each position may
##   have the following values:
##   VS.UNK  the agent ignores if it is possible to go towards the direction (useful if you want
##           to not use the check_walls_and_lim method of the AbstAgent and save only tried actions)
##   VS.WALL the agent cannot execute the action (there is a wall),
##   VS.END  the agent cannot execute the action (end of grid)
##   VS.CLEAR the agent can execute the action
##
## Example of a map entry WITH VICTIM:
## (10, 8): (3.0, 10, [VS.WALL, VS.WALL, VS.CLEAR, VS.CLEAR, VS.WALL, VS.END, VS.END, VS.END])
## the position x=10, y=8 has a difficulty of 3.0 and the victim number 10 is there.
##   +--+--+--+--+
##   !!!|XX|XX|    
##   +--+--+--+--+      AG is the currently visited position (10,8) where the victim 10 is located
##   !!!|AG|  |         XX is a wall (
##   +--+--+--+--+      !! is the end of the grid
##   !!!|XX|  |       
##   +--+--+--+--+
##
## Example of a map entry WITHOUT VICTIM:
## (11, 8): (0, VS.NO_VICTIM,[VS.WALL, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.WALL, VS.CLEAR, VS.WALL])   
##
from vs.constants import VS
COST_LINE = 1.0
COST_DIAG = 1.5

class Map:
    def __init__(self):
        self.data = {}

    
    def in_map(self, coord):
        if coord in self.data:
            return True

        return False        
                
    def get(self, coord):
        """ get all the values associated to a coord key: a triple (diff, vic_id, [actions' results])
            @param coord: a pair (x, y), the key of the dictionary"""
        return self.data.get(coord)

    def get_difficulty(self, coord):
        """ get only the difficulty value associated to a coord key: a triple (diff, vic_id, [actions' results])
            @param coord: a pair (x, y), the key of the dictionary"""
        return self.data.get(coord)[0]

    def get_vic_id(self, coord):
        """ get only the victim id number associated to a coord key: a triple (diff, vic_id, [actions' results])
            @param coord: a pair (x, y), the key of the dictionary"""
        return self.data.get(coord)[1]

    def get_actions_results(self, coord):
        """ get only the actions' results associated to a coord key: a triple (diff, vic_id, [actions' results])
            @param coord: a pair (x, y), the key of the dictionary"""
        return self.data.get(coord)[2]

        
    def add(self, coord, difficulty, vic_id, actions_res):
        """ @param coord: a pair (x, y)
            @param difficulty: the degree of difficulty to acess the cell at coord
            @param vic_id: the id number of the victim returned by the Environment
            @param actions_res: the results of the possible actions from the position (x, y) """
        self.data[coord] = (difficulty, vic_id, actions_res)

    def update(self, another_map):
        """ Itupdates the current map with the entries of another map.
            If the keys are identical, the entry of the another map replaces the entry of the current map.
            @param another_map: other instance of Map """
        self.data.update(another_map.data)

    def draw(self):
        if not self.data:
            print("Map is empty.")
            return

        min_x = min(key[0] for key in self.data.keys())
        max_x = max(key[0] for key in self.data.keys())
        min_y = min(key[1] for key in self.data.keys())
        max_y = max(key[1] for key in self.data.keys())

        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                item = self.get((x, y))
                if item:
                    if item[1] == VS.NO_VICTIM:
                        row += f"[{item[0]:7.2f}  no] "
                    else:
                        row += f"[{item[0]:7.2f} {item[1]:3d}] "
                else:
                    row += f"[     ?     ] "
            print(row)

# A* Algorithm Teste
class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost from current node to goal
        self.f = g + h  # Total cost

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(map, start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start, None, 0, heuristic(start, goal))
    open_list.append(start_node)

    while open_list:
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)

        if current_node.position == goal:
            return reconstruct_path(current_node)

        closed_set.add(current_node.position)

        for direction, (dx, dy) in enumerate([(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]):
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)
            if neighbor_pos in map:
                difficulty, vic_id, actions = map[neighbor_pos]
                if actions[direction] in [VS.WALL, VS.END]:
                    continue

                if dx == 0 or dy == 0:
                    move_cost = COST_LINE * difficulty
                else:
                    move_cost = COST_DIAG * difficulty

                neighbor_node = Node(neighbor_pos, current_node, current_node.g + move_cost, heuristic(neighbor_pos, goal))
                if neighbor_pos in closed_set:
                    continue

                if all(neighbor_node.f < node.f for node in open_list if node.position == neighbor_pos):
                    open_list.append(neighbor_node)

    return None, 0

def reconstruct_path(node):
    path = []
    total_cost = 0
    while node:
        path.append(node.position)
        if node.parent is not None:
            total_cost += node.g - node.parent.g
        node = node.parent
    return path[::-1], total_cost

def main():
    # Exemplo de uso
    map = {(0, 0): (1.0, -1, [2, 2, 0, 1, 0, 2, 2, 2]), (0, 1): (1.0, -1, [0, 0, 1, 0, 0, 2, 2, 2]), (0, 2): (1.0, -1, [0, 1, 0, 0, 0, 2, 2, 2]), (0, 3): (1.0, -1, [0, 0, 0, 0, 0, 2, 2, 2]), (0, 4): (1.0, -1, [0, 0, 0, 0, 0, 2, 2, 2]), (0, 5): (1.0, -1, [0, 0, 0, 0, 0, 2, 2, 2]), (0, 6): (1.0, -1, [0, 0, 0, 0, 0, 2, 2, 2]), (0, 7): (1.0, -1, [0, 0, 0, 0, 0, 2, 2, 2]), (0, 8): (1.0, -1, [0, 0, 0, 0, 0, 2, 2, 2]), (0, 9): (1.0, -1, [0, 0, 0, 0, 0, 2, 2, 2]), (0, 10): (1.0, -1, [0, 0, 0, 0, 0, 2, 2, 2]), (0, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 2, 2]), (1, 11): (0.1499999999996362, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (2, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (1, 10): (0.3000000000001819, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (2, 10): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (1, 9): (0.75, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (2, 9): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (1, 8): (1.5, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (2, 8): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (1, 7): (0.25, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (2, 7): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (1, 6): (0.5, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (2, 6): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (1, 5): (0.75, 1, [0, 0, 0, 0, 0, 0, 0, 0]), (2, 5): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (1, 4): (1.0, -1, [0, 1, 0, 0, 0, 0, 0, 0]), (2, 4): (1.0, -1, [1, 1, 0, 0, 0, 0, 0, 0]), (1, 3): (1.25, -1, [0, 1, 1, 0, 0, 0, 0, 0]), (1, 2): (1.5, -1, [1, 0, 1, 1, 0, 0, 0, 0]), (2, 1): (1.0, -1, [0, 0, 0, 0, 1, 0, 1, 0]), (1, 0): (1.0, -1, [2, 2, 0, 0, 1, 0, 0, 2]), (2, 0): (1.0, -1, [2, 2, 0, 0, 0, 1, 0, 2]), (3, 1): (1.0, -1, [0, 0, 0, 0, 0, 1, 0, 0]), (3, 2): (1.0, -1, [0, 0, 0, 0, 1, 1, 1, 0]), (4, 3): (1.0, -1, [0, 0, 0, 0, 0, 0, 1, 0]), (3, 4): (1.0, -1, [1, 0, 0, 0, 0, 0, 0, 1]), (3, 5): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (3, 6): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (3, 7): (1.0, 2, [0, 0, 0, 0, 0, 0, 0, 0]), (3, 8): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (3, 9): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (3, 10): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (3, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (4, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (5, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (4, 10): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (5, 10): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (4, 9): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (5, 9): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (4, 8): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (5, 8): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (4, 7): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (5, 7): (1.0, -1, [0, 1, 0, 0, 0, 0, 0, 0]), (4, 6): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (5, 6): (1.0, -1, [0, 0, 1, 0, 0, 0, 0, 0]), (4, 5): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (5, 5): (1.0, -1, [0, 0, 0, 1, 0, 0, 0, 0]), (4, 4): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 1]), (5, 4): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (6, 5): (1.0, -1, [0, 0, 0, 0, 1, 0, 0, 0]), (7, 6): (1.0, -1, [0, 0, 0, 0, 1, 0, 1, 0]), (6, 7): (1.0, -1, [1, 0, 1, 1, 0, 0, 0, 0]), (6, 8): (1.0, 3, [0, 1, 1, 0, 0, 0, 0, 0]), (6, 9): (1.0, -1, [0, 1, 0, 0, 0, 0, 0, 0]), (6, 10): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (6, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (7, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (8, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (7, 10): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (8, 10): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (7, 9): (1.0, -1, [1, 1, 0, 0, 0, 0, 0, 0]), (8, 9): (1.0, -1, [1, 0, 0, 0, 0, 0, 0, 1]), (9, 10): (1.0, 8, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 11): (1.0, -1, [0, 0, 0, 2, 2, 2, 0, 0]), (10, 11): (1.0, -1, [0, 0, 1, 2, 2, 2, 0, 0]), (11, 10): (1.0, -1, [0, 2, 2, 2, 1, 0, 0, 0]), (10, 9): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 8): (1.0, 6, [0, 0, 0, 0, 0, 0, 1, 0]), (8, 7): (1.0, 4, [0, 0, 0, 0, 1, 1, 1, 0]), (9, 7): (1.0, 5, [0, 0, 0, 0, 0, 1, 0, 0]), (8, 6): (1.0, -1, [0, 0, 0, 0, 0, 1, 0, 0]), (7, 5): (1.0, -1, [0, 0, 0, 0, 0, 1, 0, 0]), (6, 4): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (5, 3): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (4, 2): (1.0, 0, [0, 0, 0, 0, 0, 1, 0, 0]), (5, 2): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (4, 1): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (3, 0): (1.0, -1, [2, 2, 0, 0, 0, 0, 0, 2]), (4, 0): (1.0, -1, [2, 2, 0, 0, 0, 0, 0, 2]), (5, 1): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (6, 2): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (6, 3): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (7, 4): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (8, 5): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 6): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (10, 7): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (10, 8): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 9): (1.0, 7, [0, 0, 0, 0, 0, 0, 0, 1]), (10, 10): (1.0, -1, [0, 0, 0, 1, 0, 0, 0, 0]), (11, 9): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (11, 8): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (11, 7): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (10, 6): (1.0, 9, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 5): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (8, 4): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (7, 3): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (8, 3): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (7, 2): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (6, 1): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (5, 0): (1.0, -1, [2, 2, 0, 0, 0, 0, 0, 2]), (6, 0): (1.0, -1, [2, 2, 0, 0, 0, 0, 0, 2]), (7, 1): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (8, 2): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 3): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 4): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (10, 5): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (11, 6): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (11, 5): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (10, 4): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (11, 4): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (10, 3): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 2): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (8, 1): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (7, 0): (1.0, -1, [2, 2, 0, 0, 0, 0, 0, 2]), (8, 0): (1.0, -1, [2, 2, 0, 0, 0, 0, 0, 2]), (9, 1): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (10, 2): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (11, 3): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (11, 2): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (10, 1): (1.0, -1, [0, 0, 0, 0, 0, 0, 0, 0]), (9, 0): (1.0, -1, [2, 2, 0, 0, 0, 0, 0, 2]), (10, 0): (1.0, -1, [2, 2, 0, 0, 0, 0, 0, 2]), (11, 1): (1.0, -1, [0, 2, 2, 2, 0, 0, 0, 0]), (11, 0): (1.0, -1, [2, 2, 2, 2, 0, 0, 0, 2])}

    start = (1, 2)
    goal = (1, 3)
    path, cost = a_star(map, start, goal)
    print(path, cost)

if __name__ == "__main__":
    main()