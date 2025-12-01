import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """

        # If the number of cells is equal to the number of mines, 
        # we can safely infer that all of the cells are therefore mines
        if len(self.cells) == self.count:
            return self.cells
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """

        # If the count of mines is 0 then we can safely infer that
        # there are no mines
        if self.count == 0:
            return self.cells
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """

        # When passed a known mine, it will be removed from the sentence
        # and count is decremented
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """

        # When passed a safe cell, it will be removed from the sentence
        if cell in self.cells and self.count != 0:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """

        # Adds safe cell to list, and updates sentences in knowledge base
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """

        # Adds mine cell to list, and updates sentences in knowledge base
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        
        # If a move has been made and it is not game over, 
        # we can deduce it was a safe move, and add it to
        # our list of safe moves and made moves
        self.moves_made.add(cell)
        self.mark_safe(cell)

        # Populating array of neighbors as long as cell
        # is a valid area and not out of bounds
        neighbors = {(x, y) for x in range(cell[0]-1, 
                                           cell[0]+2) 
                                           if 0 <= x < self.height 
                     for y in range(cell[1]-1, 
                                    cell[1]+2) 
                                    if 0 <= y < self.width}
        
        # Removing current cell (already made move) from neighbors
        neighbors.remove(cell)

        # Loops over known safes and removes them from neighbors
        for safe in self.safes:
            if safe in neighbors:
                neighbors.remove(safe)

        # Loops over known mines and removes them from neighbors,
        # decrementing count for each removal
        for mine in self.mines:
            if mine in neighbors:
                neighbors.remove(mine)
                count -= 1

        # Finally, if a non-empty neighbors array has been constructed
        # instantiate a sentence and ass to knowledge base
        if neighbors:
            self.knowledge.append(Sentence(neighbors, count))

        # Looping over all sentences in our knowledge base
        # starting with the newest ones
        for sentence in reversed(self.knowledge.copy()):
            
            # If all cells in sentence can be deduced as safe
            # they are marked, and the sentence is updated
            for safe in sentence.known_safes().copy():
                self.mark_safe(safe)

            # If all cells in a sentence can be inferred to
            # contain mines, they are marked and the sentence
            # is updated
            for mine in sentence.known_mines().copy():
                self.mark_mine(mine)
            
            # If we are left with an empty set of cells, we remove the sentence
            if not sentence.cells or not sentence.count:
                self.knowledge.remove(sentence)

        # Looping over pairs of sentences, checking if one is a subset
        # of the other, and if so constructing a new sentence based
        # on the differences between them
        for sentence_A in self.knowledge:
            for sentence_B in self.knowledge:
                if sentence_A == sentence_B:
                    continue
                if sentence_A.cells.issubset(sentence_B.cells):
                    sentence_B.cells -= sentence_A.cells
                    sentence_B.count -= sentence_A.count
                if sentence_B.cells.issubset(sentence_A.cells):
                    sentence_A.cells -= sentence_B.cells 
                    sentence_A.count -= sentence_B.count


    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """

        # Returns a safe move that has not already been made
        safe_moves = self.safes.difference(self.moves_made)
        if safe_moves:
            return safe_moves.pop()

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        
        # When no safe moves are known, returns a random move
        # that is not a mine
        while True:
            move = (random.randint(0, 7), random.randint(0, 7))
            if move not in self.moves_made and move not in self.mines:
                return move
            