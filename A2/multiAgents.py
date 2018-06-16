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
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        # print(legalMoves[chosenIndex])
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        foodEaten = currentGameState.getNumFood() - successorGameState.getNumFood()

        for ghost in newGhostStates:
            ghostPos = ghost.configuration.getPosition()
            if not ghost.scaredTimer and withinGhostReach(newPos, ghostPos):
                # print(ghost.scaredTimer, newPos,ghost.configuration.getPosition())
                return -float('inf')

        backtrackPenalty = 0

        # Check going back and forth
        # pacDir = currentGameState.getPacmanState().getDirection()
        # print(pacDir)

        if newPos == currentGameState.getPacmanPosition():
            backtrackPenalty = 2

        if newPos in currentGameState.getCapsules():
            return float('inf')

        if foodEaten:
            # print("yum")
            return float('inf')

        # print(newPos)
        # print(successorGameState.getNumFood())

        return successorGameState.getScore() - distanceToClosestFood(newPos, newFood) - backtrackPenalty


def withinGhostReach(pacmanPos, ghostPos):
    if (pacmanPos == ghostPos) or (abs(pacmanPos[0] - ghostPos[0]) + abs(pacmanPos[1] - ghostPos[1])) == 1:
        return True
    return False


def distanceToClosestFood(pacmanPos, foodGrid):
    distanceToClosestFood = float('inf')

    for x in range(foodGrid.width):
        for y in range(len(foodGrid[x])):
            if (foodGrid[x][y]):

                MHD = manhattanDistance(pacmanPos, (x, y))
                if MHD < distanceToClosestFood:
                    distanceToClosestFood = MHD

    return distanceToClosestFood


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        if not self.depth:
            return Directions.STOP

        print(gameState.getNumAgents() * self.depth)

        maxPac(gameState, self.evaluationFunction, self.depth)

        # # Collect legal moves and successor states
        # legalMoves = gameState.getLegalActions()

        # # Choose one of the best actions
        # scores = [self.evaluationFunction(
        #     gameState, action) for action in legalMoves]
        # bestScore = max(scores)
        # bestIndices = [index for index in range(
        #     len(scores)) if scores[index] == bestScore]
        # # Pick randomly among the best
        # chosenIndex = random.choice(bestIndices)

        # "Add more of your code here if you want to"

        # # print(legalMoves[chosenIndex])
        # return legalMoves[chosenIndex]


def maxPac(gameState, evaluationFunction, depth):

    pacActions = gameState.getLegalPacmanActions()

    if not pacActions:
        return evaluationFunction(gameState)

    maxScore = -float('inf')

    for pacMove in pacActions:

        successorGameState = gameState.generatePacmanSuccessor(pacMove)

        numGhosts = successorGameState.getNumAgents() - 1
        minGhost(successorGameState, 1, numGhosts, evaluationFunction, depth)


def minGhost(gameState, currentGhost, numGhosts, evaluationFunction, depth):

    if not gameState.getNumFood() or gameState.isWin() or gameState.isLose():
        return evaluationFunction(gameState)

    if currentGhost > numGhosts:
        if depth > 0:
            maxPac(gameState, evaluationFunction, depth - 1)
        else:
            return evaluationFunction(gameState)

    ghostActions = gameState.getLegalActions(currentGhost)

    minScore = float('inf')

    if not ghostActions:
        return evaluationFunction(gameState)

    for ghostMove in ghostActions:
        ghostGameState = gameState.generateSuccessor(currentGhost, ghostMove)
        ghostScore = minGhost(ghostGameState, currentGhost + 1,
                              numGhosts, evaluationFunction, depth)

        if ghostScore < minGhost:
            minGhost = ghostScore

    return minGhost


def minMaxRecursion(gameState, action, evaluationFunction, turnsLeft):

    print(turnsLeft)

    if not gameState.getNumFood() or gameState.isWin() or gameState.isLose() or turnsLeft < 1:
        return [action, evaluationFunction(gameState)]

    pacActions = gameState.getLegalPacmanActions()

    if not pacActions:
        print(pacActions)
        return [action, evaluationFunction(gameState)]

    bestAction = None
    maxScore = -float('inf')

    for pacMove in pacActions:

        successorGameState = gameState.generateSuccessor(0, pacMove)

        numGhosts = successorGameState.getNumAgents() - 1

        maxGhost = -float('inf')

        for ghost in range(1, numGhosts+1):
            pass

    return [bestAction, maxScore]

    # ghostActions = successorGameState.getLegalActions(ghost)

    # if not ghostActions:
    #     continue

    # minScore = float('inf')

    # for ghostMove in ghostActions:
    #     ghostGameState = successorGameState.generateSuccessor(
    #         ghost, ghostMove)

    #     ghostScore = minMaxRecursion(
    #         ghostGameState, pacMove, evaluationFunction, turnsLeft - 1)[1]

    #     if ghostScore < minScore:
    #         minScore = ghostScore

    # if minScore > maxGhost:
    #     maxGhost = minScore

    #         for ghostMove in ghostActions:
    #             ghostGameState = successorGameState.generateSuccessor(
    #                 ghost, ghostMove)

    #             terminalScore = float('inf')

    #             if turnsLeft < 1:
    #                 terminalScore = evaluationFunction(ghostGameState)

    #             else:
    #                 terminalScore = minMaxRecursion(
    #                     ghostGameState, evaluationFunction, turnsLeft - 1)[1]

    #             if terminalScore < minScore:
    #                 minScore = terminalScore

    #         if pacMove == 'Stop':
    #             minScore -= 1
    #         if minScore > bestScore:
    #             bestAction = pacMove
    #             bestScore = minScore

    # # if bestScore == -9999:
    # #     print(gameState.getPacmanPosition())
    # # print [bestAction, bestScore]
    # # print(gameState.getLegalActions(0))
    # return [bestAction, bestScore]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        pass


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    return 1


# Abbreviation
better = betterEvaluationFunction
