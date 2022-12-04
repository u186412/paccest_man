# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import contest.util as util
import math
from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    return [eval('SwitchAgent')(first_index), eval('SwitchAgent')(second_index)]


class SwitchAgent(CaptureAgent):
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
        implementation of dynamic agent roles
        """
        actions = game_state.get_legal_actions(self.index)

        # useful info for agent communication
        Indexes = CaptureAgent.get_team(self, game_state)
        Indexes.remove(self.index)  # the index of the other agent on your team
        allyIndex = Indexes[0]
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        time = game_state.data.timeleft  # all operations using time assume the total time during contest matches will be 1200 agent moves, as stated in the lab description
        if game_state.is_on_red_team(self.index):
            frontier = [(math.floor((width - 1) / 2), y) for y in range(height)]
        else:
            frontier = [(math.ceil((width - 1) / 2), y) for y in range(height)]  # frontier on blue team
        validFrontier = []  # frontier positions that an agent can traverse
        walls = game_state.get_walls()
        for pos in frontier:
            if walls[pos[0]][pos[1]] == False:
                validFrontier.append(pos)

        # identify if agent should take the ATTACKING role or DEFENDING role
        # ATTACKER: closer in maze distance to frontier / in enemy territory / Fixed to an agent at game start to allow initial flank
        # DEFENDING: agent is further back / other agent is in enemy territory / Fixed to an agent at game start
        isAttacker = False
        if time > 1000:  # 50 moves
            if self.index < allyIndex:
                isAttacker = True
        else:
            if game_state.get_agent_state(self.index).is_pacman:
                isAttacker = True
            elif not game_state.get_agent_state(allyIndex).is_pacman:
                # alternative: compare X coordinates (less precise more efficient)
                if min([self.get_maze_distance(game_state.get_agent_position(self.index), frontierPos) for frontierPos
                        in validFrontier]) <= min(
                        [self.get_maze_distance(game_state.get_agent_position(allyIndex), frontierPos) for frontierPos
                         in validFrontier]):
                    isAttacker = True

        if isAttacker:
            return self.actionATTACK(game_state, actions, validFrontier)
        else:
            return self.actionDEFENSE(game_state, actions, validFrontier)

    def actionATTACK(self, game_state, actions, validFrontier):
        """
        choose action for attacking agent
        """
        values = [self.evaluateATTACK(game_state, a, validFrontier) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def actionDEFENSE(self, game_state, actions, validFrontier):
        """
        choose action for defending agent
        """
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluateDEFENSE(game_state, a, validFrontier) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluateATTACK(self, game_state, action, validFrontier):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_featuresATTACK(game_state, action, validFrontier)
        weights = self.get_weightsATTACK(game_state, action)
        return features * weights

    def evaluateDEFENSE(self, game_state, action, validFrontier):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_featuresDEFENSE(game_state, action, validFrontier)
        weights = self.get_weightsDEFENSE(game_state, action)
        return features * weights

    def get_featuresATTACK(self, game_state, action, validFrontier):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        capsule_list = self.get_capsules(successor)
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        prevState = game_state.get_agent_state(self.index)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_fooddistance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_fooddistance

        # punishment for stopping
        if action == Directions.STOP: features['stop'] = 1

        frontierDist = min([self.get_maze_distance(my_pos, frontierPos) for frontierPos in validFrontier])
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        seenGhosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        ghostDists = [self.get_maze_distance(my_pos, a.get_position()) for a in seenGhosts]
        seenPacs = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        pacDists = [self.get_maze_distance(my_pos, a.get_position()) for a in seenPacs]
        # if encountering an invader while still a ghost, acts as a backup defensive agent
        if not prevState.is_pacman and len(pacDists) > 0 and min(pacDists) <= 5:
            features['num_invaders'] = len(seenPacs)
            if len(seenPacs) > 0:
                features['invader_distance'] = min(pacDists)
                # If invader with power capsule is seen, do not come in contact but follow closely to kill as soon as it ends
                if my_state.scared_timer > 0 and min(pacDists) <= 1:
                    features['threat'] = 1

        # do not cross frontier if seen by an enemy ghost
        if not prevState.is_pacman and my_state.is_pacman and len(seenGhosts) > 0 and min(ghostDists) <= 5:
            for ghost in seenGhosts:
                if ghost.scared_timer <= 2:
                    features['inviable_attack'] = 1

        # pacman will avoid contact with enemy ghosts unless they're scared
        if len(seenGhosts) > 0 and min(
                ghostDists) <= 1:  # will not chase because i assume other contestants' defense agents are decent and wont get themselves killed
            for ghost in seenGhosts:
                if ghost.scared_timer <= 2:
                    features['threat'] = 1

        # if pacman is seen by enemy ghost and capsule is somewhat close, move towards it
        features['successor_capsule'] = -len(
            capsule_list)  # pacman does not want to eat the capsule unless severely punished
        if len(seenGhosts) > 0:
            minscaredTimer = min([ghost.scared_timer for ghost in seenGhosts])
        prevcapsule_list = self.get_capsules(game_state)
        if len(capsule_list) >= 0 and len(prevcapsule_list) > 0:
            minCapsuleDistance = min([self.get_maze_distance(my_pos, capsule) for capsule in prevcapsule_list])
            if minCapsuleDistance <= 20 and len(seenGhosts) > 0 and min(ghostDists) <= 5 and minscaredTimer <= 4:
                features['capsule_distance'] = 20 - minCapsuleDistance
                features['distance_to_food'] = 0  # pacman's natural instincts lead him to commit the sin of gluttony.
                features['successor_score'] = 0

        # agent flank through the lower lane at game start
        if game_state.data.timeleft > 1000 and not my_state.is_pacman:
            if game_state.is_on_red_team(self.index):
                features['initial_pos'] = self.get_maze_distance(my_pos, validFrontier[len(validFrontier) - 1])
            else:
                features['initial_pos'] = self.get_maze_distance(my_pos, validFrontier[0])

        """"
        PACMAN RETURN CONDITIONS
        """
        # agent needs to check non-succesor values to avoid infinite loops (will avoid return conditions because of the negative reward)
        # it may be borderline abusive to give pacman almost exclusively negative rewards
        prevpos = prevState.get_position()
        prevFood = self.get_food(game_state).as_list()
        if len(prevFood) > 0: prevmin_fooddistance = min([self.get_maze_distance(prevpos, food) for food in prevFood])
        prevfrontierDist = min([self.get_maze_distance(prevpos, frontierPos) for frontierPos in validFrontier])
        if prevState.is_pacman:
            # secure 1 point at match start
            if prevState.num_carrying == 1 and self.get_score(game_state) == 0 and prevState.num_returned == 0:
                features['frontier_RUSH'] = frontierDist
                features['distance_to_food'] = 0
                features['successor_score'] = 0
            # secure points if close to frontier while escaping
            elif min([self.get_maze_distance(prevpos, frontierPos) for frontierPos in
                      validFrontier]) <= 5 and prevState.num_carrying > 0:
                features['frontier_RUSH'] = frontierDist
                features['distance_to_food'] = 0
                features['successor_score'] = 0
            # return if only 2 points left
            elif len(prevFood) <= 2:
                features['frontier_RUSH'] = frontierDist
                features['distance_to_food'] = 0
                features['successor_score'] = 0
            # return if distance to next pellet is greater than distance to frontier and pacman is carrying pellets
            elif len(prevFood) > 0 and prevmin_fooddistance + 2 > prevfrontierDist and prevState.num_carrying >= 1:
                features['frontier_RUSH'] = frontierDist
                features['distance_to_food'] = 0
                features['successor_score'] = 0
            # return if pacman has enough pellets. 6 gives good results
            elif prevState.num_carrying >= 6:
                features['frontier_RUSH'] = frontierDist
                features['distance_to_food'] = 0
                features['successor_score'] = 0
            # return if carrying pellets and time about to run out (4 moves of tolerance)
            elif game_state.data.timeleft / 4 <= prevfrontierDist + 4 and prevState.num_carrying >= 1:
                features['frontier_RUSH'] = frontierDist
                features['distance_to_food'] = 0
                features['successor_score'] = 0
        return features

    def get_weightsATTACK(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 100, 'successor_capsule': 10000, 'distance_to_food': -1, 'inviable_attack': -30,
                'invader_distance': -20, 'threat': -5000, 'frontier_RUSH': -30, 'stop': -50, 'capsule_distance': 5,
                'initial_pos': -50}

    def get_featuresDEFENSE(self, game_state, action, validFrontier):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # keep defending agent on own team's side
        if not my_state.is_pacman: features['is_ghost'] = 1

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            # If invader with power capsule is seen, do not come in contact but follow closely to kill as soon as it ends
            if my_state.scared_timer > 0 and min(dists) <= 1:
                features['invader_threat'] = 1

        # direction penalizations
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Encourages minimizing the noisy distance reading
        # this is barely functioning because we dont have acces to a position reading, just distances.
        # i dont think it works at all
        opponentIndexes = self.get_opponents(successor)
        noiseReadings = game_state.get_agent_distances()
        enemyNoise = [noiseReadings[i] for i in opponentIndexes]
        features['noisy_distance'] = min(
            enemyNoise)  # not average because i assume most people will use one attacking agent at a time

        # Encourages staying close to the frontier
        features['frontier_distance'] = min(
            [self.get_maze_distance(my_pos, frontierPos) for frontierPos in validFrontier])

        # assume initial position at opposite flank of attack agent
        if game_state.data.timeleft > 1075 and not my_state.is_pacman:
            if game_state.is_on_red_team(self.index):
                features['initial_pos'] = self.get_maze_distance(my_pos, validFrontier[0])
            else:
                features['initial_pos'] = self.get_maze_distance(my_pos, validFrontier[len(validFrontier) - 1])

        return features

    def get_weightsDEFENSE(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'num_invaders': -1000, 'is_ghost': 100, 'invader_distance': -20, 'noisy_distance': -2,
                'frontier_distance': -1, 'stop': -10, 'reverse': -2, 'invader_threat': -1000, 'initial_pos': -50}