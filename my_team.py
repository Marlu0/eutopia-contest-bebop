import random
import contest.util as util
import time
import threading
from itertools import cycle

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='CoordinatedDefensiveAgent', second='CoordinatedDefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """

        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class CoordinatedDefensiveAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free.
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.endgame_offense = False
        self.endgame_mission_complete = False
        self.messages = ["Más vale pájaro en mano que ciento volando", "El que tiene vergüenza ni come ni almuerza", "Sarna con gusto no pica", "Dos dias de paranoIA, suspendido en IA", "Más perdido que un pulpo en un garaje", "Café y cigarro, no sé ni a qué me agarro", "Esta IA no computa, menudo hijo de la gran", "Qué felicidad, estoy en la gloria, esta me la llevo a segunda convocatoria"]
        self.message_cycle = cycle(self.messages)
        self.print_thread = threading.Thread(target=self._timed_print, daemon=True)
        self.print_thread.start()

    def _timed_print(self):
        while True:
            print(next(self.message_cycle))
            time.sleep(10)

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)

        # Mission is complete once the agent is back on the home side
        if self.endgame_offense and my_state.num_carrying > 0 and not my_state.is_pacman:
            self.endgame_mission_complete = True
            self.endgame_offense = False

        # Trigger endgame offense if conditions are met
        if game_state.data.timeleft <= 150 and not self.endgame_mission_complete and not my_state.is_pacman:
            self.endgame_offense = True
        elif self.endgame_mission_complete:
            self.endgame_offense = False

        return super().choose_action(game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Feature to drive the agent towards the target
        features['move_to_target'] = 0
        if self.a_star_search(game_state, self.get_target(game_state)) == action:
            features['move_to_target'] = 1
            
        # Penalizes stopping and reversing
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
            
        if self.endgame_offense:
            food_list = self.get_food(successor).as_list()
            if len(food_list) > 0:
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance
        else:
            # Defensive features
            features['on_defense'] = 1
            if my_state.is_pacman:
                features['on_defense'] = 0

            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            if len(invaders) > 0:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists)
            
        return features

    def get_weights(self, game_state, action):
        if self.endgame_offense and not self.endgame_mission_complete:
            return {'move_to_target': 1000, 'stop': -100, 'reverse': -2, 'distance_to_food': -1}
        
        return {'move_to_target': 1000, 'on_defense': 100, 'invader_distance': -100, 'stop': -100, 'reverse': -2}

    def get_target(self, game_state):
        my_state = game_state.get_agent_state(self.index)

        if self.endgame_offense and not self.endgame_mission_complete:
            # If carrying food, return home
            if my_state.num_carrying > 0:
                return self.start
            # Otherwise, find food
            else:
                food_list = self.get_food(game_state).as_list()
                if food_list:
                    my_pos = my_state.get_position()
                    return min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))

        # Defensive phase
        target = None
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if invaders:
            my_pos = my_state.get_position()
            teammates = self.get_team(game_state)
            teammates.remove(self.index)
            teammate_index = teammates[0]
            teammate_pos = game_state.get_agent_position(teammate_index)

            if len(invaders) == 1:
                invader_pos = invaders[0].get_position()
                dist_me = self.get_maze_distance(my_pos, invader_pos)
                dist_teammate = self.get_maze_distance(teammate_pos, invader_pos)

                if dist_me < dist_teammate:
                    target = invader_pos
                elif dist_me == dist_teammate and self.index < teammate_index:
                    target = invader_pos
            
            elif len(invaders) >= 2:
                # Sort invaders by position to ensure deterministic order for both agents
                invaders.sort(key=lambda a: a.get_position())
                invader1_pos = invaders[0].get_position()
                invader2_pos = invaders[1].get_position()

                dist_me_i1 = self.get_maze_distance(my_pos, invader1_pos)
                dist_me_i2 = self.get_maze_distance(my_pos, invader2_pos)
                dist_t_i1 = self.get_maze_distance(teammate_pos, invader1_pos)
                dist_t_i2 = self.get_maze_distance(teammate_pos, invader2_pos)

                cost1 = dist_me_i1 + dist_t_i2
                cost2 = dist_me_i2 + dist_t_i1

                if cost1 <= cost2:
                    target = invader1_pos
                else:
                    target = invader2_pos
        
        if target is not None:
            return target

        # Patrol coordination
        mid_x = game_state.data.layout.width // 2
        patrol_x = mid_x - 1
        if not self.red:  # Agent is on the blue team
            patrol_x = mid_x

        patrol_points = []
        for y in range(game_state.data.layout.height):
            if not game_state.has_wall(patrol_x, y):
                patrol_points.append((patrol_x, y))

        mid_y = game_state.data.layout.height // 2
        if self.index < 2:  # Lower index agent
            patrol_points = [p for p in patrol_points if p[1] <= mid_y]
        else:  # Higher index agent
            patrol_points = [p for p in patrol_points if p[1] > mid_y]

        my_pos = game_state.get_agent_position(self.index)
        if patrol_points:
            return min(patrol_points, key=lambda p: self.get_maze_distance(my_pos, p))
        return self.start

    def get_risk_cost(self, game_state, position):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        unscared_ghosts = [
            p for p in enemies if not p.is_pacman and p.scared_timer == 0 and p.get_position() is not None
        ]

        if not unscared_ghosts:
            return 0

        risk = 0
        for ghost in unscared_ghosts:
            dist = self.get_maze_distance(position, ghost.get_position())
            if dist > 0:
                risk_factor = 5 if game_state.data.timeleft <= 150 else 1.0
                risk += risk_factor / dist
        return risk

    def a_star_search(self, game_state, target_pos):
        start_pos = game_state.get_agent_position(self.index)
        open_set = util.PriorityQueue()
        open_set.push((start_pos, []), 0)  # (position, path)
        closed_set = set()

        while not open_set.is_empty():
            current_pos, path = open_set.pop()

            if current_pos == target_pos:
                return path[0] if path else Directions.STOP

            if current_pos not in closed_set:
                closed_set.add(current_pos)

                for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                    x, y = current_pos
                    dx, dy = Actions.direction_to_vector(action)
                    next_pos = (int(x + dx), int(y + dy))

                    if not game_state.has_wall(next_pos[0], next_pos[1]):
                        new_path = path + [action]
                        g_cost = len(new_path) + self.get_risk_cost(game_state, next_pos)
                        h_cost = self.get_maze_distance(next_pos, target_pos)
                        f_cost = g_cost + h_cost
                        open_set.push((next_pos, new_path), f_cost)

        return Directions.STOP
