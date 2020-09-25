import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Initialize variables
    next_page = dict()
    getter = corpus.get(page)

    # If page has a value
    if len(getter) != 0:
        prob = damping_factor / len(getter)

    x = (1 - damping_factor) / len(corpus)
    y = 1 / len(corpus)

    # Traverse corpus and update values of i (for each i) with the probabilities format appropriate for each
    for i in corpus:
        if not getter:
            next_page.update({i: y})
        else:
            next_page.update({i: x})
    for link in getter:
        next_page.update({link: prob + x})

    return next_page


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initiate list for samples
    samples = []

    # Copy the corpus items
    ncpy = list(corpus.copy().items())

    # Choose randomly the first page and take the key part of the key-value pair
    page = random.choice(ncpy)
    page = page[0]

    # Decrease n by one and append the first page in the samples list
    n -= 1
    samples.append(page)

    # Repeat for all remaining sample slots
    for sample in range(n):

        # Choose randomly from the weighted transition_model output
        page = random.choices(ncpy, transition_model(corpus, page, damping_factor).values())
        page = page[0][0]
        samples.append(page)

    # Count how many times each sample is found in samples[]
    false_pagerank = {i: samples.count(i) for i in samples}

    pagerank = {}

    # Update the pairs with the probability of choosing each value
    for i in false_pagerank:
        val = false_pagerank.get(i)
        true_val = val / (n + 1)
        pagerank.update({i: true_val})

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Store pairs in a dict
    rankings = dict()

    # Acceptable deviation / 10
    limit = 0.0001

    d = damping_factor
    n = len(corpus)

    # Set initial dict values equal to 1/n
    for pair in corpus:
        rankings[pair] = 1 / n

    # Check for lack of values
    for key, value in corpus.items():

        if len(value) == 0:
            # If no value is found, every page in the corpus (including this one) is added as a value

            for i in corpus:
                corpus[key].add(i)

    # Iterate until it breaks
    while True:

        # Initialize a counter
        count = 0

        # Loop over corpus' elements - 1st
        for pair in corpus:
            new = (1 - d) / n
            summing = 0

            # Loop over corpus' elements - 2nd
            for page in corpus:

                # Check if the pair is linked to by the corpus[page]
                if pair in corpus[page]:

                    # If it is, add the probability of choosing pair, given that page is chosen
                    num_links = len(corpus[page])
                    summing += rankings[page] / num_links

            # Update existing variables
            summing *= d
            new += summing

            # Check if the deviation is yet acceptable or not
            if abs(rankings[pair] - new) < limit:
                count += 1

            # Update the dictionary with the latest rank (new) for the pair
            rankings[pair] = new

        # Check if each and every key has a deviation of less than the limit
        if count == n:
            break  # If it does, break and return

    return rankings


if __name__ == "__main__":
    main()
