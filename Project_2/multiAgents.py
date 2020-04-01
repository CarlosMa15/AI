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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # A List of our food items
        myListOfFood = newFood.asList()

        # This will hold the distance to the closes food item
        # The value has an impossible default value of -1
        myDistanceToClosesFoodItem = -1

        # Iterating through all food items
        for foodItem in myListOfFood:

            # Calculating the distance to the food item
            myDistanceToFood = util.manhattanDistance(newPos, foodItem)

            # If This is the first item set that as your new distance
            if myDistanceToClosesFoodItem == -1:
                
                # Assigning the distance to our variable
                myDistanceToClosesFoodItem = myDistanceToFood

            # If the food item is closer to the current one, assign value of it
            if myDistanceToClosesFoodItem >= myDistanceToFood:
                
                # Assigning the distance to our variable
                myDistanceToClosesFoodItem = myDistanceToFood

        # This is the total distance to all ghost
        # Default is 1 because of divition by zero exception guard
        myTotalDistanceToGhost = 1

        # Iterating through all the ghost to calculate distance
        for ghostItem in successorGameState.getGhostPositions():

            # Calculating the distance to ghost
            myGhostDistance = util.manhattanDistance(newPos, ghostItem)

            # Adding up the total distance
            myTotalDistanceToGhost += myGhostDistance

        # This races the alarm if there is a ghost that is too close
        myGhostDangerExtraAlarm = 0

        #  Iterating through all the ghost and check if they are too close
        for ghostItem in successorGameState.getGhostPositions():
            
            # Calculating the distance to ghost
            myGhostDistance = util.manhattanDistance(newPos, ghostItem)

            # If the ghost is too close raise the alarms
            if myGhostDistance == 1:

                # Raising the alarm
                myGhostDangerExtraAlarm += 1

        # return successorGameState.getScore() + myTotalDistanceToGhost
        # return successorGameState.getScore() - myDistanceToClosesFoodItem
        # return successorGameState.getScore() + myDistanceToClosesFoodItem - myTotalDistanceToGhost
        # return successorGameState.getScore() - myDistanceToClosesFoodItem + myTotalDistanceToGhost
        # return successorGameState.getScore() - myDistanceToClosesFoodItem + myTotalDistanceToGhost + myGhostDangerExtraAlarm
        # return successorGameState.getScore() + myDistanceToClosesFoodItem - myTotalDistanceToGhost - myGhostDangerExtraAlarm
        return successorGameState.getScore() + (1 / float(myDistanceToClosesFoodItem)) - (1 / float(myTotalDistanceToGhost)) - myGhostDangerExtraAlarm

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Slides 05-AdversarialSearch.pdf on slide 20

        # Max Value helper method for pacman
        def myMaxValue(myGameState, myDepth):

            # if you won, lost, or reached the depth limit you are done
            # Base case of recursion method call
            if myGameState.isWin() or myGameState.isLose() or myDepth == 0:

                # return because you are done
                return self.evaluationFunction(myGameState)

            # initialize result = -∞
            result = float('-Inf')

            # for each successor of state:
            for myNextAction in myGameState.getLegalActions(0):

                # result = max(result, value(successor))
                result = max(result, myMinValue(myGameState.generateSuccessor(0, myNextAction), myDepth, 1))

            # returning result
            return result


        # Min Value helper method mostly for ghost
        def myMinValue(myGameState, myDepth, myAgentType):

            # If We won or lost we are done
            if myGameState.isWin() or myGameState.isLose():

                # return because you are done
                return self.evaluationFunction(myGameState)

            # initialize result = +∞
            result = float('Inf')

            # for each successor of state:
            for action in myGameState.getLegalActions(myAgentType):

                # Deals with the pacman agents
                if myAgentType == myGameState.getNumAgents() - 1:

                    # Checking the pacman
                    result = min(result, myMaxValue(myGameState.generateSuccessor(myAgentType, action), myDepth - 1))

                # Deals with the ghost agents
                else:

                    # Checking the ghost
                    result = min(result, myMinValue(myGameState.generateSuccessor(myAgentType, action), myDepth, myAgentType + 1))

            # returning result
            return result

        # My Minimax Agorithm
        # The Root is a special max algorithm
        def myValue(myGameState):

            # initialize result = -∞
            result = float('-Inf')

            # for each successor of state:
            for myNextAction in myGameState.getLegalActions(0):

                # Minimax algorithm starts with max then move to min dealing with ghosts
                # myGameState.generateSuccessor(0, myNextAction) gets the successor based off next action
                # starts with depth at max then decromented as we move along
                # agentIndex=0 means Pacman, ghosts are >= 1
                myReturnValue = myMinValue(myGameState.generateSuccessor(0, myNextAction), self.depth, 1)

                # The root of minimax is a max, so we are calculateing the max
                # if the value is greater than the current, replace it with it
                if myReturnValue > result:

                    # Replacing the max value
                    result = myReturnValue

                    # Setting the next action to take
                    move = myNextAction

            # Return the next moves
            return move

        # return the next value from the minimax algorithm
        return myValue(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # This is the max value helper method
        def myMaxValue(myGameState, myDepth, myAlpha, myBeta):

            # If you won,lost, or reach depth limit you are done
            if myGameState.isWin() or myGameState.isLose() or myDepth == self.depth:
                
                # Return result because you done
                return (self.evaluationFunction(myGameState), "")

            # The returning result, initialize result = -∞
            result = (float('-Inf'), "")

            # for each successor of state:
            for myNextAction in myGameState.getLegalActions(0):

                # The next state after the max is the min, for ghost
                myMoveResult = myMinValue(myGameState.generateSuccessor(0, myNextAction), myDepth, 1, myAlpha, myBeta)

                # This is the max value, if the new result is greater update it
                if myMoveResult[0] > result[0]:

                    # update the result to the new max
                    result = (myMoveResult[0], myNextAction)

                # If beta <= alpha pruning time
                if result[0] > myBeta:

                    # Doing the pruning and returning result
                    return result
                    
                # Updating the alpha based off the result that might have changed
                myAlpha = max(result[0], myAlpha)

            # returning the max results
            return result

        # This is the min value helper method
        def myMinValue(mGameState, myDepth, myAgentNumberType, myAlpha, myBeta):

             # If you won or lost you are done
            if mGameState.isWin() or mGameState.isLose():

                # Return result because you done
                return (self.evaluationFunction(mGameState), "")

            # The returning result, initialize result = +∞
            result = (float('Inf'), "")

            # for each successor of state:
            for myNextAction in mGameState.getLegalActions(myAgentNumberType):

                # agentIndex=0 means Pacman, ghosts are >= 1
                # Dealing with pacman
                if myAgentNumberType == mGameState.getNumAgents() - 1:

                    # Dealing with pacman result
                    myMoveResult = myMaxValue(mGameState.generateSuccessor(myAgentNumberType, myNextAction), myDepth + 1, myAlpha, myBeta)

                # Dealing with ghost
                else:

                    # Dealing with ghost result
                    myMoveResult = myMinValue(mGameState.generateSuccessor(myAgentNumberType, myNextAction), myDepth, myAgentNumberType + 1, myAlpha, myBeta)

                # This is the min value, if the new result is lesser update it
                if myMoveResult[0] < result[0]:

                    # update the result to the new min
                    result = (myMoveResult[0], myNextAction)

                # If beta <= alpha pruning time
                if result[0] < myAlpha:

                    # Doing the pruning and returning result
                    return result

                # Updating the beta based off the result that might have changed   
                myBeta = min(result[0], myBeta)

            # returning the min results
            return result

        # Calling the minimax algorithm with alpha beta prunning, the root is a max value
        result = myMaxValue(gameState, 0, float('-Inf'), float('Inf'))

        # Return the result
        return result[1]

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

        # Slides 06-AdversarialSearchII.pdf slide 25

        # def max-value(state):
        def myMaxValue(myGameState, myDepth):

            #  If you won,lost, or reach depth limit you are done
            if myGameState.isWin() or myGameState.isLose() or myDepth == 0:

                # Return result because you done
                return self.evaluationFunction(myGameState)

            # initialize result = -∞
            result = float('-Inf')

            # for each successor of state:
            for myNextAction in myGameState.getLegalActions(0):

                # v = max(v, value(successor))
                result = max(result, myExpectedValue(myGameState.generateSuccessor(0, myNextAction), myDepth, 1))

            # Returning the max result
            return result

        # def exp-value(state):
        def myExpectedValue(myGameState, myDepth, myAgentNumberType):

            #  If you won or lost you are done
            if myGameState.isWin() or myGameState.isLose():

                # Return result because you done
                return self.evaluationFunction(myGameState)

            # initialize result = 0
            result = 0

            # for each successor of state:
            for myNextAction in myGameState.getLegalActions(myAgentNumberType):

                # agentIndex=0 means Pacman, ghosts are >= 1
                # Dealing with pacman
                if myAgentNumberType == myGameState.getNumAgents() - 1:

                    # p = probability(successor)
                    # result += p * value(successor)
                    # Dealing with pacman result
                    result += myMaxValue(myGameState.generateSuccessor(myAgentNumberType, myNextAction), myDepth - 1)/len(myGameState.getLegalActions(myAgentNumberType))

                # Dealing with ghost 
                else:

                    # p = probability(successor)
                    # result += p * value(successor)
                    # Dealing with ghost result
                    result += myExpectedValue(myGameState.generateSuccessor(myAgentNumberType, myNextAction), myDepth, myAgentNumberType + 1)/len(myGameState.getLegalActions(myAgentNumberType))

            # returning the myExpectedValue results
            return result

        # def value(state):
        def myValue(gameState):

            # initialize myFinalMaxResult = -∞
            myFinalMaxResult = float('-Inf')

            # for each successor of state:
            for myAgentNumberType in gameState.getLegalActions(0):

                # The root is a max Value so the second is the Expected Value
                myResult = myExpectedValue(gameState.generateSuccessor(0, myAgentNumberType), self.depth, 1)

                # The root is a max value, if the value is greater then the current then replace it
                if myResult > myFinalMaxResult:

                    # Replacing the current value with the greater value
                    myFinalMaxResult = myResult

                    # Replacing the move with the next best move
                    myNextMove = myAgentNumberType

            # retruning the next best move
            return myNextMove

        # Calling and returing the value from Expectimax algorithm
        return myValue(gameState)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Loadding Information the same way as in Question 1
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # This holds the distance to the closes ghost
    myMinGhostDistance = float('Inf')

    # Iterating through the ghosts
    for myCurrentGhost in newGhostStates:

        # calculating the distance to the current ghost
        myCurrentDistance = util.manhattanDistance(newPos, myCurrentGhost.getPosition())

        # This checks if the current ghost is closer than the closes
        if myCurrentDistance < myMinGhostDistance:

            # Replacing the distance with a closer one
            myMinGhostDistance = myCurrentDistance

    # The min distance between a food item
    myMinFoodDistance = float('Inf')

    # Iteration through the food items
    for myCurrentFood in newFood.asList():

        # Calculates distance to current foood
        myCurrentFoodDistance = util.manhattanDistance(newPos, myCurrentFood)

        # Checks if the food is closer than current closes
        if myCurrentFoodDistance < myMinFoodDistance:

            # Replacing the closes food item
            myMinFoodDistance = myCurrentFoodDistance

    # return currentGameState.getScore() - myMinGhostDistance + myMinFoodDistance
    # return currentGameState.getScore() + myMinGhostDistance - myMinFoodDistance
    # return currentGameState.getScore() + myMinGhostDistance + myMinFoodDistance
    # return currentGameState.getScore() + myMinGhostDistance / myMinFoodDistance
    return currentGameState.getScore() + myMinGhostDistance / myMinFoodDistance + sum(newScaredTimes)

# Abbreviation
better = betterEvaluationFunction
