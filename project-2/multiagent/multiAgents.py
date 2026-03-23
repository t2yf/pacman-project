# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return 99999
        
        if successorGameState.isLose():
            return -99999
        
        score = successorGameState.getScore()

        # penaliza ficar parado
        if action == Directions.STOP:
            score -= 20

        newFood = newFood.asList()
        #distancia da comida mais perto
        dist = float('inf')
        if len(newFood) > 0:
            for food in newFood:
                manDist = manhattanDistance(newPos, food)
                if manDist < dist:
                    dist = manDist
            score += 10 / (dist + 1)

        #distancia dos fantasmas
        dist = 0
        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            dist = manhattanDistance(ghostPos, newPos)
            # fantasma vulneravel, avançar
            if newScaredTimes[i] > 0:
                score += 30 / (dist + 1)
            else:
                #fantasma perto
                if dist < 2:
                    score -= 200 #sair de perto logo
                score -= 20 / (dist + 1) #sair de perto
        
        return score        
            

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(gameState: GameState, depth, agentIndex):
            # se chegou ao fim, ou na profundidade máxima retornar a utilidade do estado
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # se não tem mais ações
            if len(gameState.getLegalActions(agentIndex)) == 0:
                    return self.evaluationFunction(gameState)
            
            #pacman -> max
            if agentIndex == 0:
                v = -float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    v = max(v, minimax(successor, depth, 1))
                return v
            
            # fantasminhas -> min
            else:
                v = float('inf')
                nextAgent = agentIndex +1
                #se acabar os fantasmas passa para o pacman, profundidade aumenta porque todo mundo já jogou
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth +=1
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    v = min(v, minimax(successor, depth, nextAgent))
                return v
        
        bestValue = -float('inf')
        bestAction = None

        #acoes do pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action) # pacman ja jogou
            v = minimax(successor, 0, 1)

            if v > bestValue:
                bestValue = v
                bestAction = action

        return bestAction
    
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBetaPruning(gameState: GameState, depth, alpha, beta, agentIndex):
             # se chegou ao fim retornar a utilidade do estado
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # se não tem mais ações
            if len(gameState.getLegalActions(agentIndex)) == 0:
                    return self.evaluationFunction(gameState)
            if agentIndex == 0:
                v = -float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    v = max(v, alphaBetaPruning(successor, depth, alpha, beta, 1))
                    if v > beta:
                        return v
                    else:
                        alpha = max(alpha, v)
                return v
            else:
                v = float('inf')
                nextAgent = agentIndex +1

                #se acabar os fantasmas passa para o pacman, profundidade aumenta porque todo mundo já jogou
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth +=1

                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    v = min(v, alphaBetaPruning(successor, depth, alpha, beta, nextAgent))
                    if v < alpha:
                        return v 
                    else:
                        beta = min(beta, v)
                return v 
        
        alpha = -float('inf')
        beta = float('inf')

        bestValue = -float('inf')
        bestAction = None

        #acoes do pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action) # pacman ja jogou
            v = alphaBetaPruning(successor, 0, alpha, beta, 1)

            if v > bestValue:
                bestValue = v
                bestAction = action

            alpha = max(alpha, bestValue)

        return bestAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(gameState: GameState, depth, agentIndex):
            # se chegou ao fim retornar a utilidade do estado
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # se não tem mais ações
            if len(gameState.getLegalActions(agentIndex)) == 0:
                    return self.evaluationFunction(gameState)
            
            if agentIndex == 0:
                v = -float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    v = max(v, expectimax(successor, depth, 1))
                return v
            else:
                v = 0
                nextAgent = agentIndex +1
                #se acabar os fantasmas passa para o pacman, profundidade aumenta porque todo mundo já jogou
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth +=1

                actions = gameState.getLegalActions(agentIndex)
                probability = 1.0/ len(actions) # cada ação tem mesma prob de acontecer

                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    v += probability * expectimax(successor, depth, nextAgent)
                return v

        bestValue = -float('inf')
        bestAction = None

        #acoes do pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action) # pacman ja jogou
            v = expectimax(successor, 0, 1)

            if v > bestValue:
                bestValue = v
                bestAction = action

        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    Se o pacman já ganhou ou perdeu, não tem o que calcular. Se ele ficar parado, penalizar com o restante das comidas. Fazer ele ir até as comidas mais próximas, mas ao mesmo tempo fugir do fantasma, principalmente se ele está muito perto (alta penalidade), caso contrário apenas se afastar (sem muita penalidade), ou caçá-lo (se comer bolinha)
    """
    # considerar comida e fantasminhas

    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foodList = currentGameState.getFood().asList()

    if currentGameState.isWin():
            return 99999
        
    if currentGameState.isLose():
            return -99999
    
    score = currentGameState.getScore()

    # penaliza ficar parado, quanto mais comida sobrar pior
    score -= 4 * len(foodList)

    # distancia da comida mais perta
    dist = float('inf')
    if len(foodList) > 0:
        for food in foodList:
            manDist = manhattanDistance(pacmanPos, food)
            if manDist < dist:
                dist = manDist 
        score += 30 / (dist + 1)
    
    # distancia dos fantasmas
    dist = 0
    for i, ghost in enumerate(ghostStates):
        ghostPos = ghost.getPosition()
        dist = manhattanDistance(ghostPos, pacmanPos)
        #fantasma vulneravel > avançar
        if scaredTimes[i] > 0:
            score += 40 / (dist + 1)
        else:
            #fantasma perto
            if dist < 2:
                score -= 200 # sair logo
            score -= 10/ (dist + 1)
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
