Packman project from UC Berkeley
================================

Overview
--------
This is a project for my [AI Planning for Autonomy](https://handbook.unimelb.edu.au/2019/subjects/comp90054) course. The project is derived from [UC Berkely - The Pac-Man Projects](https://inst.eecs.berkeley.edu/~cs188/fa18/project1.html).

Although project task is different from UC Berkeley project, I also solved all questions from UC Berkeley Project 1.

In this project, I implement the Iterative Deepening Search algorithm and Weighted A* algorithm. And also I define and solve a more complicated problem. Like in Q7 of the Berkerley project, I create agent that eat all of dots in a maze after eating capsules. Agent must not eat any foods before eating all capsules that exist in the maze.

Usage
-----

Except for the codes described in UC Berkeley home page, you can run following codes.

### Iterative Deepening Search algorithm
```
python pacman.py -l mediumMaze -p SearchAgent -a fn=ids
```

### Weighted A* algorithm 
```
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=wastar,heuristic=manhattanHeuristic
```

### Eat all capsules, and then eat all foods
```
python pacman.py -l capsuleSearch -p CapsuleSearchAgent -a fn=wastar,prob=CapsuleSearchProblem,heuristic=foodHeuristic
```
