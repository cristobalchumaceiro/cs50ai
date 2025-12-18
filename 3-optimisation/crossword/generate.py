import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable, values in self.domains.items():
            for value in values.copy():
                if len(value) != variable.length:
                    self.domains[variable].remove(value)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]

        if not overlap:
            return
        
        # Loop over values for a variable, and check for a match with 
        # Any value in the overlapping variable. If there is a match,
        # We can keep the value in the domain, if not, we toss it
        for x_value in self.domains[x].copy():
            match = False
            for y_value in self.domains[y].copy():
                if (x_value[overlap[0]] == y_value[overlap[1]]) & (x_value != y_value):
                    match = True
                    break
            if match == False:
                self.domains[x].remove(x_value)
                revised = True
        
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        # Populates the queue with all arcs in csp
        if arcs is None:
            queue = []
            for var in self.domains:
                neighbors = self.crossword.neighbors(var)
                for neighbor in neighbors:
                    queue.append((var, neighbor))
        else:
            queue = arcs

        # Looping over all arcs in queue and calling revise, enforcing
        # Arc consistency for each arc
        while queue:
            (x, y) = queue.pop(0)
            if self.revise(x, y):
                
                # If a revision was made, and the domain is empty, the
                # problem is no longer solvable
                if len(self.domains[x]) == 0:
                    print("domains: ", self.domains)
                    return False
                
                # Otherwise, we made a change successfully, and now need
                # To add more arcs to the queue to revise with new domains
                else:
                    for z in self.crossword.neighbors(x) - {y}:
                        queue.append((z, x))
        
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if self.domains.keys() == assignment.keys():
            return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        assigned = []

        for variable, value in assignment.items():
            
            # Checking for conflicted lengths in variable constraint
            if (len(value) != variable.length):
                return False
            
            # Checking for duplicate words in assignment
            if (value in assigned):
                return False
            
            # Checking for conflicted characters in overlap
            neighbors = self.crossword.neighbors(variable)
            for neighbor in neighbors:
                if neighbor in assignment.keys():
                    overlap = self.crossword.overlaps[variable, neighbor]
                    x_char = value[overlap[0]]
                    y_char = assignment[neighbor][overlap[1]]
                    if x_char != y_char:
                        return False 
            
            # If assignment passes checks, can be added to assigned
            else:
                assigned.append(value)
        
        return True
    
    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # Eliminating variables that have already been assigned from consideration
        neighbors = self.crossword.neighbors(var)
        neighbors = neighbors - set(assignment)

        n_domains = {neighbor:self.domains[neighbor] for neighbor in neighbors}

        # Keeping count of the eliminations that each value assignment would accrue
        elims = {val:0 for val in self.domains[var]}

        # Looping over values and incrementing their eliminations based on how many
        # Values in their home variable's neighbor's domain they would eliminate if
        # they were successfully assigned.
        for val in elims:
            for n_var, n_domain in n_domains.items():
                overlap = self.crossword.overlaps[var, n_var]
                for n_val in n_domain:

                    # Duplicates count as eliminations
                    if val == n_val:
                        elims[val] += 1

                    # If a value is not a dupe, then unmatched overlaps are counted
                    elif val[overlap[0]] != n_val[overlap[1]]:
                        elims[val] += 1

        # Sorting elims dictionary in ascending order of eliminations
        ordered = dict(sorted(elims.items(), key=lambda item: item[1]))
        ordered = list(ordered.keys())

        return ordered
    
    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Assembling dictionary to keep track of variables and their domain sizes
        # Sorting by domain size ascending
        mrv = {var:len(val) for var, val in self.domains.items() if var not in assignment}
        mrv = dict(sorted(mrv.items(), key=lambda item: item[1]))

        # Variable with the smallest domain (at this point, may be a tie)
        min = list(mrv.values())[0]

        tied_vars = [var for var in mrv if mrv[var] == min]

        # If there is more than 1 variable with a domain size equal to the minimum, 
        # Then we compare the quantity of neighbors (degrees)
        if len(tied_vars) > 1:
            degree = (0,0)
            for var in tied_vars:
                # The variable with he highest degree is saved, and returned
                if len(self.crossword.neighbors(var)) > degree[1]:
                    degree = (var, len(self.crossword.neighbors(var)))
            return degree[0]
        
        return list(mrv.keys())[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # If all variables have been assigned, we can return the assignment
        if self.assignment_complete(assignment):
            return assignment
        
        # Otherwise, we select an unassigned variable, and a value to test
        var = self.select_unassigned_variable(assignment)
        for val in self.order_domain_values(var, assignment):

            # If the assignment of value to variable is consistent we 
            # Add it to assignments and call backtrack recursively
            if self.consistent(assignment={var:val}):
                assignment[var] = val
                result = self.backtrack(assignment)
                if result:
                    return result
            
            # If the assignment is not consistent, then we can delete it
            # and move onto next value in domain
            del assignment[var]
        return None

def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
