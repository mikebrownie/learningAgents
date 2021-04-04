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
        for iteration in range(self.iterations):
            next_values = util.Counter()
            for state in self.mdp.getStates():
                best_action = self.computeActionFromValues(state)
                if best_action:
                    next_values[state] = self.getQValue(state, best_action)

            self.values = next_values

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
        val = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            val += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.getValue(next_state))
        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        # if len(self.mdp.getPossibleActions(state)) == 0 or self.mdp.getPossibleActions(state)[0] == "exit":
        #     return None  # No actions, terminal state

        max_act = None
        max_value = -111111
        for action in self.mdp.getPossibleActions(state):
            cur_value = self.computeQValueFromValues(state, action)
            if max_act is None or max_value <= cur_value:
                max_act = action
                max_value = cur_value
        return max_act

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
        states = iter(self.mdp.getStates())
        for iteration in range(self.iterations):
            state = next(states, -1)  # set state to -1 when list is exhausted
            if state == -1:
                states = iter(self.mdp.getStates())
                state = next(states, -1)
            best_action = self.computeActionFromValues(state)
            if best_action:
                self.values[state] = self.getQValue(state, best_action)

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
        #  Initializing predecessors
        preds = {}
        # for state in self.mdp.getStates():
        #     preds[state] = set()
        #     for action in self.mdp.getPossibleActions(state):
        #         for (next_state, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
        #             if prob > 0:
        #                 preds[state].add(next_state)
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    for(next_state, prob) in self.mdp.getTransitionStatesAndProbs(state,action):
                        if next_state in preds :
                            preds[next_state].add(state)
                        else:
                            preds[next_state] = {state}
        pq = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                best_action = self.computeActionFromValues(state)
                if best_action:
                    diff = abs(self.values[state] - self.getQValue(state, best_action))
                    pq.update(state, -diff)

        for iteration in range(0, self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                best_action = self.computeActionFromValues(state)
                if best_action:
                    self.values[state] = self.getQValue(state, best_action)
            for pred in preds[state]:
                best_action = self.computeActionFromValues(pred)
                if best_action:
                    diff = abs(self.values[pred] - self.getQValue(pred, best_action))
                    if diff > self.theta:
                        pq.update(pred, -diff)
