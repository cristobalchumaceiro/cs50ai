import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    # Builds dictionary for where we will keep individual's
    # probabilities, as well as their membership in each set
    knowledge = {k:{} for k in people}

    joint_prob = 1

    # Initial loop to populate knowledge dictionary
    for person in people:
        if person in one_gene:
            knowledge[person]["gene"] = 1
        elif person in two_genes:
            knowledge[person]["gene"] = 2
        else:
            knowledge[person]["gene"] = 0
        if person in have_trait:
            knowledge[person]["trait"] = True
        else:
            knowledge[person]["trait"] = False
        knowledge[person]["prob"] = 1

    for person in knowledge:

        # Updating the probability of having no gene
        if knowledge[person]["gene"] == 0:
            
            # If a person has no parents, we can assign them the
            # unconditional probability of having 0 genes
            if people[person]["mother"] == None:
                knowledge[person]["prob"] *= PROBS["gene"][0]
            
            else:
                inherit_none = 1
                
                # Loading their parent's gene values
                m_gene = knowledge[people[person]["mother"]]["gene"]
                d_gene = knowledge[people[person]["father"]]["gene"]

                if m_gene == 0:
                    inherit_none *= (1 - PROBS["mutation"])
                elif m_gene == 1:
                    inherit_none *= 0.5
                else:
                    inherit_none *= (PROBS["mutation"])

                if d_gene == 0:
                    inherit_none *= (1 - PROBS["mutation"])
                elif d_gene == 1:
                    inherit_none *= 0.5
                else:
                    inherit_none *= (PROBS["mutation"])
                
                # For a child to inherit 0 genes, it must avoid
                # inheriting a gene from either parent, therefore
                # there is only one possibility to take into account
                knowledge[person]["prob"] *= inherit_none

        # Updating probability of having one gene
        if knowledge[person]["gene"] == 1:
            
            # If a person has no parents, we can assign them the
            # unconditional probability of having 1 gene
            if people[person]["mother"] == None:
                knowledge[person]["prob"] *= PROBS["gene"][1]
            else:

                # We have to keep track of two possibilities in this
                # scenario. The child inheriting 1 gene from the Mom
                # and not the Dad, and the inverse. So we keep them separate
                inherit_from = {
                    "Mom" : 1,
                    "Dad" : 1
                }

                # Loading their parent's gene values
                m_gene = knowledge[people[person]["mother"]]["gene"]
                d_gene = knowledge[people[person]["father"]]["gene"]

                if m_gene == 0:
                    inherit_from["Mom"] *= PROBS["mutation"]
                    inherit_from["Dad"] *= (1 - PROBS["mutation"])
                elif m_gene == 1:
                    inherit_from["Mom"] *= 0.5
                    inherit_from["Dad"] *= 0.5
                else:
                    inherit_from["Mom"] *= (1 - PROBS["mutation"])
                    inherit_from["Dad"] *= PROBS["mutation"]
                
                if d_gene == 0:
                    inherit_from["Dad"] *= PROBS["mutation"]
                    inherit_from["Mom"] *= (1 - PROBS["mutation"])
                elif d_gene == 1:
                    inherit_from["Dad"] *= 0.5
                    inherit_from["Mom"] *= 0.5
                else:
                    inherit_from["Dad"] *= (1 - PROBS["mutation"])
                    inherit_from["Mom"] *= PROBS["mutation"]
                
                # Both possibilities represent two halves of a single 
                # scenario of inheriting 1 gene, therefore we can add 
                # them together and commute this to the joint probability
                knowledge[person]["prob"] *= (inherit_from["Mom"] + inherit_from["Dad"])
        
        # Updating probability of having two genes
        if knowledge[person]["gene"] == 2:
            
            # If a person has no parents, we can assign them the
            # unconditional probability of having 2 genes
            if people[person]["mother"] == None:
                knowledge[person]["prob"] *= PROBS["gene"][2]
            else:
                inherit_from_both = 1
                
                # Loading their parent's gene values
                m_gene = knowledge[people[person]["mother"]]["gene"]
                d_gene = knowledge[people[person]["father"]]["gene"]

                if m_gene == 0:
                    inherit_from_both *= PROBS["mutation"]
                elif m_gene == 1:
                    inherit_from_both *= 0.5
                else:
                    inherit_from_both *= (1 - PROBS["mutation"])
                
                if d_gene == 0:
                    inherit_from_both *= PROBS["mutation"]
                elif d_gene == 1:
                    inherit_from_both *= 0.5
                else:
                    inherit_from_both *= (1 - PROBS["mutation"])

                # For a child to inherit 2 genes, it must inherit
                # both genes from both parents, therefore there is
                # only one possibility to take into account
                knowledge[person]["prob"] *= inherit_from_both
    
        # Updating probability of having or not having the trait
        knowledge[person]["prob"] *= PROBS["trait"][knowledge[person]["gene"]][knowledge[person]["trait"]]

        # Commuting person's probability calculated to joint_prob
        joint_prob *= knowledge[person]["prob"]

    return joint_prob

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """

    for person in probabilities:
        # Checks person for membership in sets and adds variable 'p'
        # To the correct variable in the probabilities dictionary
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for person in probabilities:
        # Iterate over the two main distribution types: 'gene' and 'trait'
        for distribution in probabilities[person]:
            total = sum(probabilities[person][distribution].values())
            
            # If avoiding ZeroDivisonError
            if total:
                for prob in probabilities[person][distribution]:
                    probabilities[person][distribution][prob] /= total

if __name__ == "__main__":
    main()
