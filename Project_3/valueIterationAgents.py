# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Iterating through it
        for index in range(self.iterations):

            # Hard coping the values
            myUpdatedValues = self.values.copy()

            # iterating through the states
            for myCurrentState in self.mdp.getStates():

                # Checks the current state
                if self.mdp.isTerminal(myCurrentState):
                    continue

                # getting my possible actions
                myPossibleActions = self.mdp.getPossibleActions(myCurrentState)

                # getting the best action of out all possible
                myBestAction = max([self.getQValue(myCurrentState,action) for action in myPossibleActions])

                # updating the current value
                myUpdatedValues[myCurrentState] = myBestAction

            # updating the old values
            self.values = myUpdatedValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # https://inst.eecs.berkeley.edu/~cs188/fa18/project3.html#Q1
        # Question 1 (4 points): Value Iteration
        # Math equation

        result = 0

        # The equation summation
        for myS, myT in self.mdp.getTransitionStatesAndProbs(state, action):

            # equation calculation and summing the results
            result += myT * ( self.mdp.getReward(state, action, myS) + self.discount*self.getValue(myS) )

        # returning the results
        return result

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # A Counter is a dict with default 0
        myPolicyCollection = util.Counter()

        # iterating through the actions
        for myCurrentAction in self.mdp.getPossibleActions(state):

            # Calculating the new policy
            myPolicyCollection[myCurrentAction] = self.getQValue(state, myCurrentAction)

        # returns the indices of the maximum values
        return myPolicyCollection.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # You did not teach this. I went off google explaination.

        # getting the total action
        myTotalState = self.mdp.getStates()

        # Iteration
        for index in range(self.iterations):

            # getting the state
            myState = myTotalState[index % len(myTotalState)]

            # Checking the state
            if self.mdp.isTerminal(myState):
                continue

            # getting the actions
            myActions = self.mdp.getPossibleActions(myState)

            # calculating the best action
            myBestAction = max([self.getQValue(myState,myAction) for myAction in myActions])

            # updating the one state
            self.values[myState] = myBestAction

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        """
        Compute predecessors of all states.
        Initialize an empty priority queue.
        For each non-terminal state s, do: (note: to make the autograder 
        work for this question, you must iterate over states in the order returned by self.mdp.getStates())
        Find the absolute value of the difference between the current value of s 
        in self.values and the highest Q-value across all possible actions from s (this 
        represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
        Push s into the priority queue with priority -diff (note that this is negative). 
        We use a negative because the priority queue is a min heap, but we want to 
        prioritize updating states that have a higher error.
        For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        If the priority queue is empty, then terminate.
        Pop a state s off the priority queue.
        Update s's value (if it is not a terminal state) in self.values.
        For each predecessor p of s, do:
        Find the absolute value of the difference between the current value of p 
        in self.values and the highest Q-value across all possible actions from p 
        (this represents what the value should be); call this number diff. Do NOT update self.values[p] in this step.
        If diff > theta, push p into the priority queue with priority -diff (note 
        that this is negative), as long as it does not already exist in the priority 
        queue with equal or lower priority. As before, we use a negative because the 
        priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        """

        # https://inst.eecs.berkeley.edu/~cs188/fa18/project3.html#Q5
        # Question 5 (3 points): Prioritized Sweeping Value Iteration
        # This was not taught I used the given website and google to come up with the answer

        # myPriorityQueue
        myPriorityQueue = util.PriorityQueue()

        # getting total states
        myTotalState = self.mdp.getStates()

        # initialzing predictions
        myPredictions = {}

        # iterating through total states
        for myCurrentState in myTotalState:

            # checks the current state
            if self.mdp.isTerminal(myCurrentState):
                continue

            # iterating through possible actions
            for myCurrentAction in self.mdp.getPossibleActions(myCurrentState):


                # iterating through transition state
                for myTransitionState, myProb in self.mdp.getTransitionStatesAndProbs(myCurrentState, myCurrentAction):

                    # checking is transition state is in predictions
                    if myTransitionState in myPredictions:

                        # adding current state tp prediction
                        myPredictions[myTransitionState].add(myCurrentState)

                    # updating my current state
                    else:
                        myPredictions[myTransitionState] = {myCurrentState}

        # Iterating through states
        for myCurrentState in self.mdp.getStates():

            # checking current state
            if self.mdp.isTerminal(myCurrentState):
                continue

            # calculating the difference
            myDiffernce = abs(self.values[myCurrentState] - max([ self.computeQValueFromValues(myCurrentState, myCurrentAction) for myCurrentAction in self.mdp.getPossibleActions(myCurrentState) ]) )

            # updating the priority Queue
            myPriorityQueue.update(myCurrentState, -myDiffernce)

        # iterating 
        for index in range(self.iterations):

            # checking the priority Queue
            if myPriorityQueue.isEmpty():
                break

            # get current state
            myCurrentState = myPriorityQueue.pop()

            # checks current state
            if not self.mdp.isTerminal(myCurrentState):

                # update the corresponding value
                self.values[myCurrentState] = max([self.computeQValueFromValues(myCurrentState, myCurrentAction) for myCurrentAction in self.mdp.getPossibleActions(myCurrentState)])

            # iterating through predictions
            for myPrediction in myPredictions[myCurrentState]:

                # checking prediction
                if self.mdp.isTerminal(myPrediction):
                    continue

                # calculating difference
                myDiffernce = abs(self.values[myPrediction] - max([self.computeQValueFromValues(myPrediction, myCurrentAction) for myCurrentAction in self.mdp.getPossibleActions(myPrediction)]))

                # comparing to theta
                if myDiffernce > self.theta:
                        # updating priority Queue
                        myPriorityQueue.update(myPrediction, -myDiffernce)