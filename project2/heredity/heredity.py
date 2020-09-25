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

    parents = []
    children = []
    compare = {}

    # Categorize people to children (if they have a mother, they also have a father) and to parents
    for person in people:
        if people.get(person)['mother'] is not None:
            children.append(person)
        else:
            parents.append(person)

    # Go through parents first
    for parent in parents:

        # Get the parent's number of gene copies and trait presence
        gene = (1 if parent in one_gene else 2 if parent in two_genes else 0)
        trait = (True if parent in have_trait else False)

        # Get the corresponding probabilities in float
        prob = float(PROBS["gene"][gene])
        prob1 = float(PROBS["trait"][gene][trait])

        # Multiply the first with the second probs
        result = prob * prob1

        # Update (create) the entry {parent: result}, in which dict it will be used for later operations
        compare.update({parent: round(result, 4)})

    # Next, go through the children
    for child in children:

        # Get the mother and father names
        mother = people.get(child)['mother']
        father = people.get(child)['father']

        # Add them to the "family" list
        family = [mother, father]

        fam_total = []

        # Set the list(prob) based on the number of gene copies of the parents
        prob = ([(1 / 2) if parent in one_gene else (1 - PROBS['mutation']) if parent in two_genes
                else (PROBS['mutation']) for parent in family])

        # Append the resulting probability list
        fam_total.append(prob)
        fam_total = fam_total[0]

        # Set the corresponding probabilities
        prob_mother = float(fam_total[0])
        prob_father = float(fam_total[1])

        # Get the independent part of the equation, like with the parents
        gene = (1 if child in one_gene else 2 if child in two_genes else 0)
        trait = (True if child in have_trait else False)
        prob1 = float(PROBS["trait"][gene][trait])

        # And the dependent to the parents part
        child_prob = ((1 - prob_mother) * prob_father + (1 - prob_father) * prob_mother if gene == 1
                      else prob_mother * prob_father if gene == 2 else (1 - prob_mother) * (1 - prob_father))

        # Get the result and update the dict
        result = round(child_prob * prob1, 5)
        compare.update({child: result})

    # Multiply all the probabilities to get the total and return it
    total = 1
    for key, value in compare.items():
        total *= value
    return total


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Loop through every person
    for person in probabilities:

        # First check for the number of gene copies and update correspondingly with p
        if person in one_gene:
            probabilities[person]['gene'][1] += p
        elif person in two_genes:
            probabilities[person]['gene'][2] += p
        else:
            probabilities[person]['gene'][0] += p

        # Then check for the presence of the trait and update correspondingly with p
        if person in have_trait:
            probabilities[person]['trait'][True] += p
        else:
            probabilities[person]['trait'][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Loop through every person
    for person in probabilities:

        # Get the sums of the gene and trait values
        gene_sum = sum(probabilities[person]['gene'].values())
        trait_sum = sum(probabilities[person]['trait'].values())

        # Update the probabilities dict by normalising the distributions
        probabilities[person]['gene'] = {key: (values / gene_sum) for key, values
                                         in probabilities[person]['gene'].items()}
        probabilities[person]['trait'] = {key: (values / trait_sum) for key, values
                                          in probabilities[person]['trait'].items()}


if __name__ == "__main__":
    main()
