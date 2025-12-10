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
    
    # Populating results dict with the probability of landing on each page
    # calculated as P(1 - damping_factor / Number of Links)
    res = {k:(1-damping_factor)/len(corpus) for k in corpus}
    
    # Looping over the values in corpus, if there is any page with no links
    # we can assume it may have links to any page (including itself) and 
    # updating it's probability
    for value in corpus[page]:
        links = len(corpus[page])
        if not links:
            links = len(corpus)
        res[value] += damping_factor/links
    return res
    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    samples = {k:0 for k in corpus}

    i = 0
    res = None
 
    while i < n:

        # If it is the first iteration, it will choose a page at random
        # from all pages available
        if not res:
            page = random.choice(list(corpus.keys()))

        # Otherwise, it will chose a page from a list of links available
        # on the current page, and increments the value of the page chosen
        else:
            page = random.choices(list(res.keys()), weights = list(res.values()))[0]
            samples[page] += 1

        # Call transition model on current page to return probability
        # distribution for the available links
        res = transition_model(corpus, page, damping_factor)
        i += 1
    
    # Sums the total visits for each page and generates a final 
    # probability value based on the number of samples
    total = sum(list(samples.values()))
    for k, v in samples.items():
        samples[k] = v / total
    
    return samples

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Checks if every page in corpus contains links to other pages
    # if not, gives it links to all pages in corpus
    for k, v in corpus.items():
        if not len(v):
            corpus[k] = set(corpus.keys())

    # Assigning initial probability of 1 / Number of Pages
    res = {k:(1/len(corpus)) for k in corpus}

    diffs = {k:0 for k in corpus}

    while True:
        for page in res.keys():
            
            # Generates list of pages that link to current page
            linked_by = [key for key, val in corpus.items() if page in val]

            # Generates probability that we were on a page in linked_by 
            # and chose the link to the current page for every page in linked_by
            link_p = [res[link]/len(corpus[link]) for link in linked_by]

            # Calculates new Page Rank for current page according to formula
            new_rank = ((1-damping_factor)/len(corpus)) + damping_factor * sum(link_p)
            
            # Calculates difference between new and old rank to test exit condition
            diffs[page] = abs(new_rank - res[page])
            
            res[page] = new_rank
        
        if max(list(diffs.values())) < 0.001:
            break

    return res


if __name__ == "__main__":
    main()
