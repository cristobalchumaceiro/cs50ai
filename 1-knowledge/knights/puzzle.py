from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # Setting rules for the game using XOR Logic for A
    And(
        Or(AKnight, AKnave),
        Not(And(AKnight, AKnave))
    ),
    # If A is telling the truth, it must be a Knight and a Knave
    Implication(AKnight, And(AKnight, AKnave)),
    # If A is telling a lie, it must not be a Knight and a Knave
    Implication(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # Setting rules for the game using XOR Logic for A and B
    And(
        Or(AKnight, AKnave), 
        Not(And(AKnight, AKnave))
    ),
    And(
        Or(BKnight, BKnave), 
        Not(And(BKnight, BKnave))
    ),
    # If A is telling the truth, they must both be knaves
    Implication(AKnight, And(AKnave, BKnave)),

    # If A is telling a lie, they must not both be knaves
    Implication(AKnave, Not(And(AKnave, BKnave)))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # Setting rules for the game using XOR Logic for A and B
    And(
        Or(AKnight, AKnave), 
        Not(And(AKnight, AKnave))
    ),
    And(
        Or(BKnight, BKnave), 
        Not(And(BKnight, BKnave))
    ),

    # If A is telling the truth, they must both be of the same kind
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    
    # If A is telling a lie, they must not be of the same kind
    Implication(AKnave, And(Not(And(AKnight, BKnight)), Not(And(AKnave, BKnave)))),
    
    # If B is telling the truth, they must both be of different kinds
    Implication(BKnight, Or(And(AKnave, BKnight), And(AKnave, BKnight))),

    # If B is telling a lie, they must both be of the same kind
    Implication(BKnave, Or(And(AKnight, BKnight), And(AKnave, BKnave)))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # Setting rules for the game using XOR Logic for A, B, and C
    And(
        Or(AKnight, AKnave), 
        Not(And(AKnight, AKnave))
    ),
    And(
        Or(BKnight, BKnave), 
        Not(And(BKnight, BKnave))
    ),
    And(
        Or(CKnight, CKnave),
        Not(And(CKnight, CKnave)) 
    ),
    
    # If A is telling the truth about being a knight, then it must be a knight
    Implication(AKnight, AKnight),
       
    # If A is telling the truth about being a knave, then it must be a knight
    Implication(AKnave, AKnight),
        
    # If B is telling the truth, then A and C must both be knaves
    Implication(BKnight, And(AKnave, CKnave)),
        
    # If B is telling a lie, then A and C must both be knights
    Implication(BKnave, And(AKnight, CKnight)),
        
    # If C is telling the truth, A must be a knight
    Implication(CKnight, AKnight),
        
    # If C is telling a lie, A must be a knave
    Implication(CKnave, AKnave)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
