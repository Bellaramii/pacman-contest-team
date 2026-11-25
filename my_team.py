# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import heapq
import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    #The contest creates our two agents using their indices
    return [eval(first)(first_index), eval(second)(second_index)]


class ReflexCaptureAgent(CaptureAgent):
    """Some common functions used by both agents."""

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.border = []

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        walls = game_state.get_walls()
        middle = walls.width // 2

        #Only use border positions where we can cross to the other side
        if self.red:
            home_x, other_x = middle - 1, middle
        else:
            home_x, other_x = middle, middle - 1

        for y in range(1, walls.height - 1):
            if not walls[home_x][y] and not walls[other_x][y]:
                self.border.append((home_x, y))

        if not self.border:
            self.border = [nearest_point(self.start)]

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def get_visible_ghosts(self, game_state):
        ghosts = []
        for index in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(index)
            pos = enemy.get_position()
            #Scared ghosts are not dangerous while their timer lasts
            if pos is not None and not enemy.is_pacman and enemy.scared_timer <= 5:
                ghosts.append(nearest_point(pos))
        return ghosts

    def astar_action(self, game_state, goals, dangerous_positions=None):
        """A* finds a short route. Ghosts make nearby squares more costly."""
        if not goals:
            return None

        start = game_state.get_agent_position(self.index)
        if start is None:
            return None
        start = nearest_point(start)
        targets = set(nearest_point(pos) for pos in goals)
        legal = game_state.get_legal_actions(self.index)

        if start in targets:
            return Directions.STOP if Directions.STOP in legal else None

        walls = game_state.get_walls()
        middle = walls.width // 2
        ghosts = dangerous_positions or []

        #We can use Manhattan distance as the A* heuristic
        def heuristic(pos):
            x, y = pos
            return min(abs(x - gx) + abs(y - gy) for gx, gy in targets)

        moves = [
            (Directions.NORTH, 0, 1),
            (Directions.SOUTH, 0, -1),
            (Directions.EAST, 1, 0),
            (Directions.WEST, -1, 0)
        ]

        queue = [(heuristic(start), 0, 0, start, None)]
        best_cost = {start: 0}
        number = 0

        while queue:
            _, cost, _, pos, first_action = heapq.heappop(queue)
            if cost > best_cost.get(pos, float('inf')):
                continue

            if pos in targets:
                return first_action

            for action, dx, dy in moves:
                x, y = pos[0] + dx, pos[1] + dy
                if x < 0 or x >= walls.width or y < 0 or y >= walls.height:
                    continue
                if walls[x][y]:
                    continue
                if pos == start and action not in legal:
                    continue

                #Extra cost for going close to a dangerous enemy ghost
                risk = 0
                enemy_side = x >= middle if self.red else x < middle
                if enemy_side:
                    for gx, gy in ghosts:
                        dist = abs(x - gx) + abs(y - gy)
                        if dist <= 1:
                            risk += 60
                        elif dist == 2:
                            risk += 18
                        elif dist == 3:
                            risk += 5

                next_pos = (x, y)
                new_cost = cost + 1 + risk
                if new_cost < best_cost.get(next_pos, float('inf')):
                    best_cost[next_pos] = new_cost
                    number += 1
                    first = action if pos == start else first_action
                    heapq.heappush(queue, (new_cost + heuristic(next_pos),
                                           new_cost, number, next_pos, first))

        return None

    def choose_closest_action(self, game_state, target, scared_enemies=None):
        """Used by the defender when chasing or patrolling."""
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return None

        my_pos = game_state.get_agent_position(self.index)
        if my_pos == target and Directions.STOP in actions and not scared_enemies:
            return Directions.STOP

        best_action = actions[0]
        best_value = -float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            my_state = successor.get_agent_state(self.index)
            pos = my_state.get_position()
            if pos is None:
                continue

            value = -self.get_maze_distance(pos, target)
            if my_state.is_pacman:
                value -= 100
            if action == Directions.STOP:
                value -= 2

            #When scared, it is better to keep distance from invaders
            if scared_enemies:
                distance = min(self.get_maze_distance(pos, p) for p in scared_enemies)
                if distance <= 1:
                    value -= 100
                elif distance == 2:
                    value -= 15

            if value > best_value:
                best_value = value
                best_action = action

        return best_action


class OffensiveReflexAgent(ReflexCaptureAgent):
    """Collect food, avoid active ghosts and bring food back safely."""

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return None

        state = game_state.get_agent_state(self.index)
        pos = state.get_position()
        if pos is not None:
            pos = nearest_point(pos)
        food = self.get_food(game_state).as_list()
        ghosts = self.get_visible_ghosts(game_state)
        carrying = state.num_carrying

        if pos is None:
            return random.choice(actions)

        #Estimate how far we are from our own border
        distance_home = min(self.get_maze_distance(pos, home) for home in self.border)
        closest_ghost = min((self.get_maze_distance(pos, ghost) for ghost in ghosts),
                            default=999)

        data = getattr(game_state, 'data', None)
        time_left = getattr(data, 'timeleft', 1200)
        own_score = self.get_score(game_state)

        #Return earlier if we are in danger or running out of time
        go_home = carrying >= 3
        if carrying > 0 and closest_ghost <= 4:
            go_home = True
        if carrying > 0 and time_left <= 4 * (distance_home + 7):
            go_home = True
        if carrying > 0 and own_score > 0 and time_left < 120:
            go_home = True
        if carrying > 0 and len(food) <= 2:
            go_home = True

        if state.is_pacman and go_home:
            action = self.astar_action(game_state, self.border, ghosts)
            if action in actions:
                return action

        #Capsules can help us cross an area guarded by ghosts
        capsules = []
        if hasattr(self, 'get_capsules'):
            capsules = self.get_capsules(game_state)

        if ghosts and closest_ghost <= 5 and capsules and not go_home:
            action = self.astar_action(game_state, capsules, ghosts)
            if action in actions:
                return action

        #Choose a food target with A*: a safe path can beat a short risky path
        if food:
            action = self.astar_action(game_state, food, ghosts)
            if action in actions:
                return action

        #If there is no more food, return to our side
        action = self.astar_action(game_state, self.border, ghosts)
        if action in actions:
            return action

        return random.choice([a for a in actions if a != Directions.STOP] or actions)


class DefensiveReflexAgent(ReflexCaptureAgent):
    """Chase invaders and investigate food that disappears on our side."""

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.old_food = set()
        self.last_food_eaten = None
        self.memory_turns = 0
        self.patrol_index = 0
        self.patrol_turns = 0

        #Remember the food we must defend
        if hasattr(self, 'get_food_you_are_defending'):
            self.old_food = set(self.get_food_you_are_defending(game_state).as_list())

        #Use several border entries for patrolling instead of only one
        self.patrol_points = []
        for index in [len(self.border) // 4,
                      len(self.border) // 2,
                      (3 * len(self.border)) // 4]:
            point = self.border[min(index, len(self.border) - 1)]
            if point not in self.patrol_points:
                self.patrol_points.append(point)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return None

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        if my_pos is not None:
            my_pos = nearest_point(my_pos)
        if my_pos is None:
            return random.choice(actions)

        invaders = []
        for index in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(index)
            if enemy.is_pacman and enemy.get_position() is not None:
                invaders.append(nearest_point(enemy.get_position()))

        #If defended food disappears, someone was probably there
        if hasattr(self, 'get_food_you_are_defending'):
            current_food = set(self.get_food_you_are_defending(game_state).as_list())
            missing_food = self.old_food - current_food
            self.old_food = current_food
            if missing_food:
                self.last_food_eaten = min(
                    missing_food, key=lambda p: self.get_maze_distance(my_pos, p)
                )
                self.memory_turns = 10

        if invaders:
            #Priority 1: chase the nearest visible invader
            target = min(invaders, key=lambda p: self.get_maze_distance(my_pos, p))
            self.memory_turns = 0
            scared = invaders if my_state.scared_timer > 0 else None
            return self.choose_closest_action(game_state, target, scared)

        if self.memory_turns > 0 and self.last_food_eaten is not None:
            #Priority 2: investigate where food was eaten
            self.memory_turns -= 1
            return self.choose_closest_action(game_state, self.last_food_eaten)

        #Priority 3: patrol different routes into our territory
        self.patrol_turns += 1
        target = self.patrol_points[self.patrol_index]
        if self.patrol_turns >= 15 or self.get_maze_distance(my_pos, target) <= 1:
            self.patrol_turns = 0
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
            target = self.patrol_points[self.patrol_index]

        return self.choose_closest_action(game_state, target)
