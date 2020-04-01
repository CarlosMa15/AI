# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        # initializing Q-values here
        self.values  = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # return 0.0 if we have never seen a state or the Q node value otherwise
        return self.values[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # setting my result to min
        result = float('-inf')

        # iteration through action
        for myAction in self.getLegalActions(state):

            # calculates the max result
            result  = max(self.getQValue(state, myAction), result)

        # If nothing happens return at least zero
        if result == float('-inf'):

            # return zero
            return 0.0

        # if stuff happens return that result
        else : return result

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # get my legal actions
        myLegalActions = self.getLegalActions(state)

        """
        Note that if there are no legal actions, which is the 
        case at the terminal state, you should return None.
        """
        if len(myLegalActions) is 0:
            return None

        # computing the value from Q values
        myValue = self.computeValueFromQValues(state)

        # empty collection to store the actions we want
        myActions = []

        # Iterating through action
        for myAction in myLegalActions:

            # check against the value
            if myValue is self.getQValue(state, myAction):

                # add the actions we want
                myActions.append(myAction)

        # randomly choosing and returning an action
        return random.choice(myActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        # flip coin
        if util.flipCoin(self.epsilon):

            # randomly choosing action
            action = random.choice(legalActions)

        else:

            # otherwise compute action from Q value
            action = self.computeActionFromQValues(state)

        # returning the action we want
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # inverse 
        inverse = (1-self.alpha)

        # value
        value = self.values[(state,action)]

        # alpha
        alpha = self.alpha

        # next value
        nextValue = self.computeValueFromQValues(nextState)

        # Q value update here
        self.values[(state, action)] = inverse * value + alpha * (reward + self.discount * nextValue)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        # Get weight
        weight = self.getWeights()

        # get feature vector 
        featureVector = self.featExtractor.getFeatures(state,action)

        # returns weight * feature vector
        return weight * featureVector

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        # calculating the difference
        myDifference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state,action)

        # extracting my features
        myfeatExtractorFeatures = self.featExtractor.getFeatures(state,action)

        # iterating through features
        for myFeatures in myfeatExtractorFeatures.keys():

            # updating your weights based on transition
            self.weights[myFeatures] += self.alpha * myDifference * myfeatExtractorFeatures[myFeatures]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # NO THANK YOU
            pass
