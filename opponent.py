import random
from definitions import State
from collections import defaultdict 

def random_opponent(state):
    moves = []
    x, y = state.opponent_x, state.opponent_y
    if x > 0:
        moves.append(1)  # left
    if x < 3:
        moves.append(2)  # right
    if y > 0:
        moves.append(3)  # up
    if y < 3:
        moves.append(4)  # down
    return random.choice(moves)


def greedy_opponent(state):
    moves = []
    ox, oy = state.opponent_x, state.opponent_y
    if state.possesion == 1:
        px, py = state.player_1_x, state.player_1_y
    else:
        px, py = state.player_2_x, state.player_2_y

    dx = px - ox
    dy = py - oy

    # Prioritize horizontal or vertical movement towards the player
    if abs(dx) > abs(dy):
        if dx < 0 and ox > 0:
            moves.append(1)  # left
        elif dx > 0 and ox < 3:
            moves.append(2)  # right
    if abs(dy) >= abs(dx) or not moves:
        if dy < 0 and oy > 0:
            moves.append(3)  # up
        elif dy > 0 and oy < 3:
            moves.append(4)  # down

    if not moves:
        # If no greedy move possible, stay in bounds
        if ox > 0:
            moves.append(1)
        if ox < 3:
            moves.append(2)
        if oy > 0:
            moves.append(3)
        if oy < 3:
            moves.append(4)

    return random.choice(moves)

def defensive_opponent(state):
    ox, oy = state.opponent_x, state.opponent_y

    # Define defensive zone
    defensive_x = [2, 3]
    defensive_y = [1, 2]

    # If in defensive zone, move randomly within the zone
    if ox in defensive_x and oy in defensive_y:
        moves = []
        if ox > 2:
            moves.append(1)  # left
        if ox < 3:
            moves.append(2)  # right
        if oy > 1:
            moves.append(3)  # up
        if oy < 2:
            moves.append(4)  # down
        return random.choice(moves)

    # Otherwise, move towards the defensive zone
    moves = []
    if ox < 2:
        moves.append(2)  # right
    elif ox > 3:
        moves.append(1)  # left
    elif ox == 2 and ox < 3:
        moves.append(2)  # right

    if oy < 1:
        moves.append(4)  # down
    elif oy > 2:
        moves.append(3)  # up
    elif oy == 1 and oy < 2:
        moves.append(4)  # down

    # If both x and y are in range, but not both at once, allow both moves
    if not moves:
        if ox > 0:
            moves.append(1)
        if ox < 3:
            moves.append(2)
        if oy > 0:
            moves.append(3)
        if oy < 3:
            moves.append(4)

    return random.choice(moves)