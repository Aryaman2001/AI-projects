# A* Search - 8 tile puzzle

The 8-tile puzzle was invented and popularized by Noyes Palmer Chapman in the 1870s. It is played on a 3x3 grid with 8 tiles labeled 1 through 8 and an empty grid. The goal is to rearrange the tiles so that they are in order. You solve the puzzle by moving the tiles around. For each step, you can only move one of the neighbor tiles (left, right, top, bottom) into an empty grid. And all tiles must stay in the 3x3 grid. 

Given these rules for the puzzle, my code generates a state space and solves this puzzle using the A* search algorithm. Note that not all 8-tile puzzles are solvable. For this project, the input puzzle is always solvable.
