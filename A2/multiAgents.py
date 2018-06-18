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
        return self.maxPac(gameState, self.depth)[1]

    def maxPac(self, gameState, depth):

        if depth < 1 or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), Directions.STOP]

        pacActions = gameState.getLegalActions(0)

        if not pacActions:
            return [self.evaluationFunction(gameState), Directions.STOP]

        maxScore = -float('inf')
        bestAction = None

        for pacMove in pacActions:

            successorGameState = gameState.generateSuccessor(0, pacMove)

            numGhosts = successorGameState.getNumAgents() - 1
            ghostScore = self.minGhost(
                successorGameState, 1, numGhosts, depth)[0]

            if ghostScore > maxScore:
                maxScore = ghostScore
                bestAction = pacMove

        return [maxScore, bestAction]

    def minGhost(self, gameState, currentGhost, numGhosts, depth):

        if gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), Directions.STOP]

        if currentGhost > numGhosts:
            if depth > 0:
                return self.maxPac(gameState, depth - 1)
            else:
                return [self.evaluationFunction(gameState), Directions.STOP]

        ghostActions = gameState.getLegalActions(currentGhost)

        minScore = float('inf')

        if not ghostActions:
            return [self.evaluationFunction(gameState), Directions.STOP]

        bestAction = None

        for ghostMove in ghostActions:
            ghostGameState = gameState.generateSuccessor(
                currentGhost, ghostMove)
            ghostScore = self.minGhost(
                ghostGameState, currentGhost + 1, numGhosts,  depth)[0]

            if ghostScore < minScore:
                minScore = ghostScore
                bestAction = ghostMove

        return [minScore, bestAction]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
	#Set alpha and beta
        alpha = -float('inf')
	beta = float('inf')
    	return self.chooseMove(gameState, alpha, beta, self.depth, 0)[1]
    def chooseMove(self, gameState, alpha, beta, depth, gameAgent):
	#If we have reached depth or the game is over	
	if depth < 1 or gameState.isWin() or gameState.isLose():
		return [self.evaluationFunction(gameState), Directions.STOP]
	if gameAgent == 0:
		#It is pacman and a max node
		return self.pacmanMove(gameState, alpha, beta, depth)
	if gameAgent > 0:
		#It is a ghost and a min node
		return self.ghostMove(gameState, alpha, beta, depth, gameAgent)
    def pacmanMove(self, gameState, alpha, beta, depth):
	moves = gameState.getLegalActions(0)
	value = alpha
	if not moves:
		return [self.evaluationFunction(gameState)]	
	bestMove = None
	for move in moves:
	    if alpha >= beta:
		#Don't bother searching other children		
		break;
	    nextState = gameState.generateSuccessor(0, move)
	    score = self.chooseMove(nextState, alpha, beta, depth, 1)
	    if score[0] > value:
		value = score[0]
		alpha = score[0]
		bestMove = move
	return [alpha, bestMove]
    def ghostMove(self, gameState, alpha, beta, depth, gameAgent):
	value = beta
	if gameAgent >= gameState.getNumAgents():
		#We've been through all the ghosts
		return self.chooseMove(gameState, alpha, beta, depth - 1, 0)
	#It is still a ghost 
	moves = gameState.getLegalActions(gameAgent)
	if not moves:
		return [self.evaluationFunction(gameState), Directions.STOP]
	scores = 0	
	bestMove = None
	for move in moves:
		if beta <= alpha:
			#Don't bother searching other children
			break;
		nextState = gameState.generateSuccessor(gameAgent, move)
		score = self.chooseMove(nextState, alpha, beta, depth, gameAgent + 1)
		if score[0] < value:
			value = score[0]
			beta = score[0]
			bestMove = move
	return [beta, bestMove]


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
        return self.chooseMove(gameState, self.depth, 0)[1]

    def chooseMove(self, gameState, depth, gameAgent):
        # print(depth)
        if depth < 1 or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), Directions.STOP]
        if gameAgent == 0:
            # It is pacman and a max node
            return self.pacmanMove(gameState, depth)
        if gameAgent > 0:
            # It is a ghost and a exp node
            return self.ghostMove(gameState, depth, gameAgent)

    def pacmanMove(self, gameState, depth):
        moves = gameState.getLegalActions(0)
        if not moves:
            return [self.evaluationFunction(gameState)]
        bestScore = -float('inf')
        bestMove = None
        for move in moves:
            nextState = gameState.generateSuccessor(0, move)

            score = (self.chooseMove(nextState, depth, 1))

            if move == Directions.STOP:
                # print (1)
                score[0] -= 2

            # print(score)
            if float(score[0]) > bestScore:
                bestScore = score[0]
                bestMove = move
        return [bestScore, bestMove]

    def ghostMove(self, gameState, depth, gameAgent):
        if gameAgent >= gameState.getNumAgents():
                # We've been through all the ghosts
            return self.chooseMove(gameState, depth - 1, 0)
        # It is still a ghost
        moves = gameState.getLegalActions(gameAgent)
        if not moves:
            return [self.evaluationFunction(gameState)]
        scores = 0
        bestMove = None
        prob = 1.0/float(len(moves))
        for move in moves:
            nextState = gameState.generateSuccessor(gameAgent, move)
            score = self.chooseMove(nextState, depth, gameAgent + 1)
            # print(score)
            scores += (float(score[0]) * prob)
        return [scores]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

    DESCRIPTION: Try to not go into positions which can lead to dying to a ghost.
    """

    if currentGameState.isLose():
        return scoreEvaluationFunction(currentGameState) - 500

    numFood = currentGameState.getNumFood()

    if not numFood or currentGameState.isWin():
        return scoreEvaluationFunction(currentGameState)

    pacPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    numGhosts = currentGameState.getNumAgents() - 1

    numCapsules = len(currentGameState.getCapsules())

    foodGrid = currentGameState.getFood()

    for ghost in ghostStates:
        ghostPos = ghost.configuration.getPosition()
        if not ghost.scaredTimer and withinGhostReach(pacPos, ghostPos):
            # print(ghost.scaredTimer, newPos,ghost.configuration.getPosition())
            return scoreEvaluationFunction(currentGameState) - distanceToClosestFood(pacPos, foodGrid) - numFood*10 - 500 - numCapsules*10

    return scoreEvaluationFunction(currentGameState) - distanceToClosestFood(pacPos, foodGrid) - numFood*10 - numGhosts*10 - numCapsules*10


# Abbreviation
better = betterEvaluationFunction
