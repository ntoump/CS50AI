import sys
import operator
import copy

from crossword import *

from time import perf_counter


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
                        w, h = draw.textsize(letters[i][j], font=font)
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
        t0 = perf_counter()
        self.enforce_node_consistency()
        self.ac3()
        x = self.backtrack(dict())
        t1 = perf_counter()
        print("time:", t1-t0)
        return x

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        for var in self.crossword.variables:
            words = list(self.domains.get(var))
            for w in words:
                if var.length != len(w):
                    self.domains[var].remove(w)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        # Initialize useful variables
        revised = False
        rmv = []
        constraint = self.crossword.overlaps.get((x, y))

        # No possible routes for crossword combinations, unless the have distinct directions
        if x.direction == y.direction or constraint is None:
            return revised

        # Therefore, instead use these two lines
        constraint_x = constraint[0]
        constraint_y = constraint[1]

        # Loop through possible domain values of x
        for value in self.domains[x]:
            found = False

            # Loop through possible domain values of y
            for val in self.domains[y]:

                # Check if found
                if value[constraint_x] == val[constraint_y]:
                    found = True

            # If found move on to next one
            if found:
                continue
            else:
                rmv.append(value)
                revised = True

        # Remove all values in rmv list from domains of x
        for not_match in rmv:
            self.domains[x].remove(not_match)

        # Returned boolean revised
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        # Initialize useful variables
        initial_queue = (list(self.crossword.variables) if arcs is None else list(arcs))
        queue, seen = [], []

        # Loop through initial que
        for not_x in initial_queue:
            for not_y in initial_queue:

                # If there are overlaps and they are not in a way or another in list seen, append to queue and seen
                if self.crossword.overlaps.get((not_x, not_y)) and (not_x, not_y) not in seen and (not_y, not_x)\
                        not in seen:
                    queue.append((not_x, not_y))
                    seen.append((not_x, not_y))

        # While there are elements in queue
        while queue:

            # Get x and y
            x, y = queue.pop()

            # If x is arc consistent with y
            if self.revise(x, y):

                # Check if there are no possible domain values
                if len(self.domains[x]) == 0:
                    return False

                # Loop through neighbors of x
                for neighbor in self.crossword.neighbors(x):

                    # Append (neighbor, x) to queue to make all neighbors of x arc consistent with it
                    if neighbor != y:
                        queue.append((neighbor, x))

        # Return if success
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """

        # Return False if there is at least one key in assignment without value, else True
        return (False if not assignment.get(key) else True for key in assignment)

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # Initialize useful variables
        queue, seen, ex_v = [], [], []
        initial_queue = list(self.crossword.variables)

        # This ensures no duplicates
        if assignment:
            for key in assignment:
                if assignment.get(key) not in ex_v:
                    v = assignment.get(key)
                    ex_v.append(v)
                else:
                    return False

        # This ensures no conflicting chars
        for not_x in initial_queue:
            for not_y in initial_queue:
                if self.crossword.overlaps.get((not_x, not_y)) and (not_x, not_y) not in seen and (not_y, not_x) \
                        not in seen:
                    seen.append((not_x, not_y))
                    queue.append((not_x, not_y))

        # Repeat
        while queue:

            # Get x, y
            x, y = queue.pop()

            # Check for overlaps
            if x != y and (x, y) and self.crossword.overlaps.get((x, y)):
                constraint = self.crossword.overlaps.get((x, y))
                constraint_x = constraint[0]
                constraint_y = constraint[1]

                # Check if assignment complete, return False if not
                value, val = assignment.get(x), assignment.get(y)
                if value and val:
                    if value[constraint_x] != val[constraint_y]:
                        return False

        # Return True if complete
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        # Initialize useful variables
        domains = list(self.domains.get(var))
        neighbors = self.crossword.neighbors(var)
        seen = {}
        returning_list = []

        # Loop through neighbors
        for neighbor in neighbors:

            # If there are overlaps between var and neighbor, get them in a list
            if self.crossword.overlaps.get((var, neighbor)) and var != neighbor:
                overlap = list(self.crossword.overlaps.get((var, neighbor)))

                # Loop through domains
                for domain in domains:
                    counter = 0

                    # Count how many neighbor domains are excluded with current domain
                    for neighbor_domain in self.domains.get(neighbor):
                        if neighbor_domain in assignment:
                            continue
                        if domain[overlap[0]] != neighbor_domain[overlap[1]]:
                            counter += 1

                    # Insert finding in dict as value and domain as key
                    seen[domain] = counter

        # Sort dict according to ascending value (counter) - exclude fewer neighbor domains, go higher in list
        temp_list = sorted((value, key) for key, value in seen.items())

        # Loop through entries in list
        for entry in temp_list:

            # Append only second element of each entry
            returning_list.append(entry[1])

        # Return list
        return returning_list

        # # Old version:
        # return list(self.domains.get(var))

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        archive = {}

        # Loop through variables
        for var in list(self.crossword.variables):

            # Filter vars already in assignment
            if var not in assignment:

                # Call order_domain_values and get its length in receiver
                receiver = len(self.order_domain_values(var, assignment))

                # Insert {var: receiver} in archive
                archive[var] = receiver

        # Sort archive according to value (length of domain) in ascending order
        sorted_archive = dict(sorted(archive.items(), key=operator.itemgetter(1)))

        # Make a copy
        copier = copy.copy(sorted_archive)

        # Loop through first
        for variable in sorted_archive:

            # If more than one entries
            if len(sorted_archive) > 1:

                # Loop through second
                for vv in copier:

                    # Ensure different keys
                    if variable != vv:

                        # If variable domain length is equal to vv domain length
                        if sorted_archive.get(variable) == copier.get(vv):
                            first_neighbors = len(self.crossword.neighbors(variable))
                            second_neighbors = len(self.crossword.neighbors(vv))

                            # Return the one with the most neighbors, as per Specification
                            if first_neighbors >= second_neighbors:
                                return variable
                            else:
                                return vv
                        else:
                            break

            # If only one entry left, return it
            return variable

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

    # Essentially, this function enforces a smarter instance of brute force

        if len(assignment.values()) == len(list(self.crossword.variables)):
            return assignment

        # Call select_unassigned_variable and save it in var
        var = self.select_unassigned_variable(assignment)

        # Loop through domains of var
        for domain in self.domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = domain

            # Check for full consistency
            if self.consistent(new_assignment):

                # Recursively call same function and save in result
                result = self.backtrack(new_assignment)

                # Return result if not None
                if result:
                    return result

        # No possible assignment
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
