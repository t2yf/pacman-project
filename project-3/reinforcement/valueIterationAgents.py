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
        allStates = self.mdp.getStates()
        iterations = self.iterations
        for i in range(iterations):
            newValues = util.Counter()
            for state in allStates:
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    allPossibleActions = self.mdp.getPossibleActions(state)
                    QValue = []

                    for action in allPossibleActions:
                        QValue.append(self.getQValue(state, action))
                    newValues[state] = max(QValue)
                
            self.values = newValues

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
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0 

        for nextState, transitionFn in transition:
            immediateReward = self.mdp.getReward(state, action, nextState)
            pastValues = self.getValue(nextState)

            QValue += transitionFn*(immediateReward + self.discount*pastValues)

        return QValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # se já está no estado final, se houver
        if self.mdp.isTerminal(state):
            return None
        else:
            allPossibleActions = self.mdp.getPossibleActions(state)
            QValue = util.Counter() #inicializa com 0, counter eh um dict
            for action in allPossibleActions:
                QValue[action] = self.getQValue(state, action)

            return QValue.argMax() #retorna a melhor ação

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        all_states = self.mdp.getStates()
   

        # Predecessores
        predecessors = {}
        for state in all_states:
            predecessors[state] = set()

        for state in all_states:
            for action in self.mdp.getPossibleActions(state):
                for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

        # Fila de prioridade
        priorityQueue = util.PriorityQueue()

        for state in all_states:
            if not self.mdp.isTerminal(state):
                diff = abs(self.getValue(state) - self.computeQValues(state))
                priorityQueue.push(state, -diff)

        # Atualizar
        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            else:
                state = priorityQueue.pop()
                if not self.mdp.isTerminal(state):
                    self.values[state] = self.computeQValues(state)
                for p in predecessors[state]:
                    if self.mdp.isTerminal(p):
                        diff = abs(self.getValues(p))
                    else: 
                        diff = abs(self.getValue(p) - self.computeQValues(p))
                    
                    if diff > self.theta:
                        priorityQueue.update(p, -diff)

    def computeQValues(self, state):
        all_actions = self.mdp.getPossibleActions(state)
        qValues = []
        for action in all_actions :
            qValues.append(self.getQValue(state, action))
        
        return max(qValues)
 
