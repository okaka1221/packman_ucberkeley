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
    """

    stack = util.Stack()
    closed = set()

    #each node contains
    #(potsition, [path])
    start_node = (problem.getStartState(), [])
    stack.push(start_node)

    while not stack.isEmpty():
        current_node = stack.pop()
        if current_node[0] not in closed:
            closed.add(current_node[0])

            if problem.isGoalState(current_node[0]):
                return current_node[1]

            successors = problem.getSuccessors(current_node[0])
            for successor in successors:
                path = current_node[1] + [successor[1]]
                stack.push((successor[0], path))

    return False

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    queue = util.Queue()
    closed = set()

    #each node contains
    #(potsition, [path])
    start_node = (problem.getStartState(), [])
    queue.push(start_node,)

    while not queue.isEmpty():
        current_node = queue.pop()
        if current_node[0] not in closed:
            closed.add(current_node[0])
            if problem.isGoalState(current_node[0]):
                return current_node[1]
                
            successors = problem.getSuccessors(current_node[0])
            for successor in successors:
                path = current_node[1] + [successor[1]]
                queue.push((successor[0], path))
                
    return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    p_queue = util.PriorityQueue()
    closed = set()

    #each node contains
    #(potsition, [path], cost)
    start_node = (problem.getStartState(), [], 0)
    p_queue.push((start_node), 0)

    while not p_queue.isEmpty():
        current_node = p_queue.pop()
        if current_node[0] not in closed:
            closed.add(current_node[0])

            if problem.isGoalState(current_node[0]):
                return current_node[1]

            successors = problem.getSuccessors(current_node[0])
            for successor in successors:
                path = current_node[1] + [successor[1]]
                cost = current_node[2] + successor[2]
                p_queue.push((successor[0], path, cost), cost)
            
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    p_queue = util.PriorityQueue()
    closed = set()

    #each node contains
    #(potsition, [path], cost)
    start_node = (problem.getStartState(), [], 0)
    p_queue.push((start_node), heuristic(start_node[0], problem))

    while not p_queue.isEmpty():
        current_node = p_queue.pop()
        
        if current_node[0] not in closed:
            closed.add(current_node[0])

            if problem.isGoalState(current_node[0]):
                return current_node[1]

            successors = problem.getSuccessors(current_node[0])
            for successor in successors:
                path = current_node[1] + [successor[1]]
                cost = current_node[2] + successor[2]
                p_queue.push((successor[0], path, cost), cost + heuristic(successor[0], problem))

    return False

def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    depth_limit = 1
    stack = util.Stack()
    
    # For duplicate detection
    duplicates = {}

    while True:
        closed = set()

        #each node contains
        #(potsition, [path], depth)
        start_node = (problem.getStartState(), [], 1, None)
        stack.push(start_node)
        
        while not stack.isEmpty():
            current_node = stack.pop()
            
            """
            IMPORTANT!!!
            Followin lines commented out are for duplicate detection.
            However, it fails test with duplicate detection and I decided
            to remove them.
            """
            # key = str(current_node)

            # if key in duplicates:
            #     current_node = duplicates[key]["node"]
            #     successors = duplicates[key]["successors"]
            #     closed.add(current_node[0])

            #     for successor in successors:
            #         path = current_node[1] + [successor[1]]
            #         depth = current_node[2] + 1
            #         stack.push((successor[0], path, depth))

            if current_node[0] not in closed and current_node[2] <= depth_limit:
                closed.add(current_node[0])
                
                successors = problem.getSuccessors(current_node[0])

                # node_info = {}
                # node_info["node"] = current_node
                # node_info["successors"] = successors

                # duplicates[key] = node_info

                for successor in successors:
                    path = current_node[1] + [successor[1]]
                    depth = current_node[2] + 1

                    if problem.isGoalState(successor[0]):
                        return path
                    
                    stack.push((successor[0], path, depth))

            if stack.isEmpty():
                break

        depth_limit += 1
        
    return False

def waStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has has the weighted (x 2) lowest combined cost and heuristic first."""
    weight = 2
    p_queue = util.PriorityQueue()
    closed = set()

    #each node contains
    #(potsition, [path], cost)
    start_node = (problem.getStartState(), [], 0)
    p_queue.push((start_node), heuristic(start_node[0], problem))

    while not p_queue.isEmpty():
        current_node = p_queue.pop()
        
        if current_node[0] not in closed:
            closed.add(current_node[0])

            if problem.isGoalState(current_node[0]):
                return current_node[1]

            successors = problem.getSuccessors(current_node[0])
            for successor in successors:
                path = current_node[1] + [successor[1]]
                cost = current_node[2] + successor[2]
                
                p_queue.push((successor[0], path, cost), cost + weight * heuristic(successor[0], problem))

    return False

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
wastar = waStarSearch
