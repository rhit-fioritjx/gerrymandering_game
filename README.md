# Gerrymandering Game

This project is a small board/videogame which I am working on which teaches players about [gerrymandering](https://en.wikipedia.org/wiki/Gerrymandering) by having them actually do it. Currently it is played in the terminal through a python script, but I hope to later improve the graphics and possibly the performance.

## How to play
Run the python program. By default it runs a 7x7 board with 7 districts and 2 parties but you can play around with that in the code.

To add some nodes to a district list out the nodes followed by the district seperated by spaces e.g. "0 1 7 e 15 0" would put nodes 0, 1, 7, e, and 15 into district 0.

Because the ANSI codes can be a bit unreliable as a display method you can peek at the original state of the board using "flash" "f" "show" or "peek" which will show the original state until you press enter again.

To reset the board to its original state you can use "reset" or "restart".

If you cannot solve the puzzle you can use "forfeit" "quit" or "give up" to view a solution. 

### Win Condition
Each district must be:
 - Of Equal Size
 - Contiguous (all nodes must be connected)
 - Have 1 clear plurality winner
The set of districts must cover the entire graph.
The minority party must have won the most districts of any party.
