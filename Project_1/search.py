# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # This is the stack of the depth first search algorithm
    myStack = util.Stack()

    # This is a set that keeps track of the visited vertex, taking in graph terms
    myVisitedVertex = set()

    # The start vertex is added to the stack
    # We are using the stack to keep track of the path from that vertex to the start vertex
    myStack.push((problem.getStartState(), []))

    # The depth first search algorithm
    while True:

        # The vertex item we are expanding
        myExpandedVertex = myStack.pop()

        # The vertex we are expanding
        myCurrentVertex = myExpandedVertex[0]

        # The path to the vertex we are expanding
        myPathToStart = myExpandedVertex[1]

        # If the vertex is the goal, then we are done
        # The path is in myPathToStart
        if problem.isGoalState(myCurrentVertex):  # Exit on encountering goal node
            break

        # If the vertex is not the goal, we keep looking with depth first search
        else:

            # We only care about unvisited vertex, the rest will be visted later or already in the stack
            if myCurrentVertex not in myVisitedVertex:

                # If not visited then adding it
                myVisitedVertex.add(myCurrentVertex)

                # For a given state, this should return a list of triples, (successor,
                # action, stepCost), where 'successor' is a successor to the current
                # state, 'action' is the action required to get there, and 'stepCost' is
                # the incremental cost of expanding to that successor.
                myChildVertexes = problem.getSuccessors(myCurrentVertex)

                # Add Processing the child vertexes
                for myChildVertexItem in myChildVertexes:

                    # Getting the Child Vertex Information
                    myChildVertex = myChildVertexItem[0]

                    # Getting the Child Vertex Path
                    myChildPath = myChildVertexItem[1]

                    # Creating the Full Path
                    myFullPathToStart = myPathToStart + [myChildPath]

                    # Adding the child vertex to the stack
                    myStack.push((myChildVertex, myFullPathToStart))

    # returning the complete path from the start to goal
    return myPathToStart

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # This is the queue of the depth first search algorithm
    myQueue = util.Queue()

    # This is a set that keeps track of the visited vertex, taking in graph terms
    myVisitedVertex = set()

    # The start vertex is added to the stack
    # We are using the queue to keep track of the path from that vertex to the start vertex
    myQueue.push((problem.getStartState(), []))

    # The depth first search algorithm
    while True:

        # The vertex item we are expanding
        myExpandedVertex = myQueue.pop()

        # The vertex we are expanding
        myCurrentVertex = myExpandedVertex[0]

        # The path to the vertex we are expanding
        myPathToStart = myExpandedVertex[1]

        # If the vertex is the goal, then we are done
        # The path is in myPathToStart
        if problem.isGoalState(myCurrentVertex):  # Exit on encountering goal node
            break

        # If the vertex is not the goal, we keep looking with breadth first search
        else:

            # We only care about unvisited vertex, the rest will be visted later or already in the queue
            if myCurrentVertex not in myVisitedVertex:

                # If not visited then adding it
                myVisitedVertex.add(myCurrentVertex)

                # For a given state, this should return a list of triples, (successor,
                # action, stepCost), where 'successor' is a successor to the current
                # state, 'action' is the action required to get there, and 'stepCost' is
                # the incremental cost of expanding to that successor.
                myChildVertexes = problem.getSuccessors(myCurrentVertex)

                # Add Processing the child vertexes
                for myChildVertexItem in myChildVertexes:

                    # Getting the Child Vertex Information
                    myChildVertex = myChildVertexItem[0]

                    # Getting the Child Vertex Path
                    myChildPath = myChildVertexItem[1]

                    # Creating the Full Path
                    myFullPathToStart = myPathToStart + [myChildPath]

                    # Adding the child vertex to the queue
                    myQueue.push((myChildVertex, myFullPathToStart))

    # returning the complete path from the start to goal
    return myPathToStart

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # This is the queue of the depth first search algorithm
    myQueue = util.PriorityQueue()

    # This is a set that keeps track of the visited vertex, taking in graph terms
    myVisitedVertex = set()

    # The start vertex is added to the stack
    # We are using the queue to keep track of the path from that vertex to the start vertex
    # We are also keeping track of the cost to the vertex in the queue
    myQueue.push((problem.getStartState(), [], 0), 0)

    # The depth first search algorithm
    while True:

        # The vertex item we are expanding
        myExpandedVertex = myQueue.pop()

        # The vertex we are expanding
        myCurrentVertex = myExpandedVertex[0]

        # The path to the vertex we are expanding
        myPathToStart = myExpandedVertex[1]

        # The cost to the vertex
        myVertexCost = myExpandedVertex[2]

        # If the vertex is the goal, then we are done
        # The path is in myPathToStart
        if problem.isGoalState(myCurrentVertex):  # Exit on encountering goal node
            break

        # If the vertex is not the goal, we keep looking with uniform Cost Search
        else:

            # We only care about unvisited vertex, the rest will be visted later or already in the queue
            if myCurrentVertex not in myVisitedVertex:

                # If not visited then adding it
                myVisitedVertex.add(myCurrentVertex)

                # For a given state, this should return a list of triples, (successor,
                # action, stepCost), where 'successor' is a successor to the current
                # state, 'action' is the action required to get there, and 'stepCost' is
                # the incremental cost of expanding to that successor.
                myChildVertexes = problem.getSuccessors(myCurrentVertex)

                # Add Processing the child vertexes
                for myChildVertexItem in myChildVertexes:

                    # Getting the Child Vertex Information
                    myChildVertex = myChildVertexItem[0]

                    # Getting the Child Vertex Path
                    myChildPath = myChildVertexItem[1]

                    # Getting the Child Vertex Cost
                    myChildCost = myChildVertexItem[2]

                    # Creating the Full Cost to the Child Vertex
                    myFullCost = myVertexCost + myChildCost

                    # Creating the Full Path to the Child Vertex
                    myFullPathToStart = myPathToStart + [myChildPath]

                    # Adding the child vertex to the queue
                    myQueue.push((myChildVertex, myFullPathToStart, myFullCost), myFullCost)

    # returning the complete path from the start to goal
    return myPathToStart

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # This is the queue of the depth first search algorithm
    myQueue = util.PriorityQueue()

    # This is a set that keeps track of the visited vertex, taking in graph terms
    myVisitedVertex = set()

    # The start vertex is added to the stack
    # We are using the queue to keep track of the path from that vertex to the start vertex
    # We are also keeping track of the cost to the vertex in the queue
    myQueue.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem))

    # The depth first search algorithm
    while True:

        # The vertex item we are expanding
        myExpandedVertex = myQueue.pop()

        # The vertex we are expanding
        myCurrentVertex = myExpandedVertex[0]

        # The path to the vertex we are expanding
        myPathToStart = myExpandedVertex[1]

        # The cost to the vertex
        myVertexCost = myExpandedVertex[2]

        # If the vertex is the goal, then we are done
        # The path is in myPathToStart
        if problem.isGoalState(myCurrentVertex):  # Exit on encountering goal node
            break

        # If the vertex is not the goal, we keep looking with A Star Search
        else:

            # We only care about unvisited vertex, the rest will be visted later or already in the queue
            if myCurrentVertex not in myVisitedVertex:

                # If not visited then adding it
                myVisitedVertex.add(myCurrentVertex)

                # For a given state, this should return a list of triples, (successor,
                # action, stepCost), where 'successor' is a successor to the current
                # state, 'action' is the action required to get there, and 'stepCost' is
                # the incremental cost of expanding to that successor.
                myChildVertexes = problem.getSuccessors(myCurrentVertex)

                # Add Processing the child vertexes
                for myChildVertexItem in myChildVertexes:

                    # Getting the Child Vertex Information
                    myChildVertex = myChildVertexItem[0]

                    # Getting the Child Vertex Path
                    myChildPath = myChildVertexItem[1]

                    # Getting the Child Vertex Cost
                    myChildCost = myChildVertexItem[2]

                    # Creating the Full Cost to the Child Vertex
                    myFullCost = myVertexCost + myChildCost

                    # Creating the Full Path to the Child Vertex
                    myFullPathToStart = myPathToStart + [myChildPath]

                    # Adding the child vertex to the queue
                    myQueue.push((myChildVertex, myFullPathToStart, myFullCost), myFullCost + heuristic(myChildVertex, problem))

    # returning the complete path from the start to goal
    return myPathToStart


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
