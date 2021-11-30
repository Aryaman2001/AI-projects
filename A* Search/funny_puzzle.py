import numpy as np
import heapq

# computes manhattan distance heuristic for a list
def heuristic(state):
    state = np.reshape(np.array(state), (3,3))
    distance = 0
    orig_coordinates = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]]
    for i in range(1, 9):
        cod1 = orig_coordinates[i - 1]
        cod2 = []

        # finding the tile in the given list
        for j in range(0, 3):
            for k in range(0, 3):
                if state[j][k] == i:
                    cod2 = [j ,k]
        
        distance += abs(cod1[0]- cod2[0]) + abs(cod1[1] - cod2[1])
    return distance

# finds successors to a given state
def find_succ(state):
    # converting 1-D list into np 2-D matrix
    state = np.array(state)
    state = np.reshape(state, (3,3))
    # find the blank tile
    blank_cod = []
    states = []
    for j in range(0, 3):
            for k in range(0, 3):
                if state[j][k] == 0:
                    blank_cod = [j ,k]
    
    # checking all four possibilites
    # 1. moving it up
    if blank_cod[0] != 0:
        new_state = np.copy(state)
        # storing the digit before moving blank tile
        temp = new_state[blank_cod[0] - 1][blank_cod[1]]
        new_state[blank_cod[0] - 1][blank_cod[1]] = 0
        new_state[blank_cod[0]][blank_cod[1]] = temp
        states.append(new_state.flatten().tolist())
    
    # 2. moving it down
    if blank_cod[0] != 2:
        new_state = np.copy(state)
        # storing the digit before moving blank tile
        temp = new_state[blank_cod[0] + 1][blank_cod[1]]
        new_state[blank_cod[0] + 1][blank_cod[1]] = 0
        new_state[blank_cod[0]][blank_cod[1]] = temp
        states.append(new_state.flatten().tolist())

    # 3. moving it left
    if blank_cod[1] != 0:
        new_state = np.copy(state)
        # storing the digit before moving blank tile
        temp = new_state[blank_cod[0]][blank_cod[1] - 1]
        new_state[blank_cod[0]][blank_cod[1] - 1] = 0
        new_state[blank_cod[0]][blank_cod[1]] = temp
        states.append(new_state.flatten().tolist())
    
    # 4. moving it right
    if blank_cod[1] != 2:
        new_state = np.copy(state)
        # storing the digit before moving blank tile
        temp = new_state[blank_cod[0]][blank_cod[1] + 1]
        new_state[blank_cod[0]][blank_cod[1] +1] = 0
        new_state[blank_cod[0]][blank_cod[1]] = temp
        states.append(new_state.flatten().tolist())

    states = sorted(states)
    return states                   

# traces all parent nodes and prints them in order
def tracepath(node, CLOSED):
    print_list = []
    print_list.append(node)

    while True:
        index = node[2][2]

        if index == -1:
            break
        
        node = CLOSED[index]
        print_list.insert(0, node)

    moves = 0
    for node in print_list:
        print('{} h={} moves: {}'.format(node[1], node[2][1], moves))
        moves += 1        
    return

def print_succ(state):
    states = find_succ(state)
    for state in states:
        h = heuristic(state)
        print('{} h={}'.format(state, h))

def solve(state): 
    open = []
    closed = []
    h = heuristic(state)
    g = 0
    heapq.heappush(open, (g + h, state, (g, h, -1)))
    if len(open) == 0:
        return
    else:
        while True:
            heapq.heapify(open)
            pop = heapq.heappop(open)
            closed.append(pop)
            if pop[2][1] == 0:
                print_sol(pop, closed)
                return
            successors = succ(pop[1])
            g_successors = pop[2][0] + 1
            for successor in successors:
                p = False
                h_successors = heuristic(successor)
                for j in range(len(closed)):
                    if closed[j][1] == successor:
                        p = True
                        if closed[j][2][0] > g_successors:
                            p = False            
                        else:
                             continue
                        break
                for j in range(len(open)):
                    if open[j][1] == successor:
                        p = True
                        if open[j][2][0] > g_successors:
                            open.pop(j)
                            heapq.heapify(open)
                            p = False
                        else:
                            continue
                        break
                if not p:
                    heapq.heappush(open, (g_successors + h_successors, successor, (g_successors, h_successors, pop[2][2] + 1)))

#print_succ([1,2,3,4,5,0,6,7,8])
solve([4,3,8,5,1,6,7,2,0])
#solve([1,2,3,4,5,6,7,0,8])
