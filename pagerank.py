import os
import random
import numpy
from numpy import random
import re
import sys
import math


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
    pageslist = corpus.keys()
    transitionmodel = {}
    linkspro = {}
    if len(corpus[page]) != 0:
        links = corpus[page]
        for i in pageslist:
            transitionmodel[i] = round((1 - damping_factor) * 1 / len(pageslist),10)
        for i in links:
            linkspro[i] = (damping_factor) * 1 / len(corpus[page])
            linkspro[i] = linkspro[i] + transitionmodel[i]
            transitionmodel[i] = linkspro[i]
        return transitionmodel
    if len(corpus[page]) == 0:
        for i in pageslist:
            transitionmodel[i] = 1 / len(pageslist)
        return transitionmodel


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagelist = corpus.keys()
    result = []
    final = {}
    randompage = random.choice(list(pagelist))
    transtition_model = transition_model(corpus, randompage, damping_factor)
    for i in range(n):
        transitionpages = list(transtition_model.keys())
        transitionprobability = list(transtition_model.values())
        resultpage = random.choice(transitionpages, p=transitionprobability)
        result.append(resultpage)
        transtition_model = transition_model(corpus, resultpage, damping_factor)
    for i in pagelist:
        count = result.count(i)
        pageprobability = count/n
        final[i] = pageprobability
    return final




def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    d = damping_factor
    transitionmodel = {}
    pagerank = {}
    pageslist = corpus.keys()
    testingrank = {}
    links = {}
    for i in pageslist:
        linksset = set()
        if len(corpus[i]) == 0:
            for k in pageslist:
                linksset.add(k)
            links[i] = linksset
        for j in pageslist:
            if i in corpus[j]:
                linksset.add(j)
        links[i] = linksset
    sn = ((1 - d) / len(pageslist))
    for i in pageslist:
        transitionmodel[i] = 1 / len(pageslist)
        pagerank[i] = transitionmodel[i]
    while True:
        sse = []
        for i in pageslist:
            testingrank[i] = pagerank[i]
            spf = 0
            for j in links[i]:
                if len(corpus[j]) == 0:
                    spf = spf + (d * (pagerank[j] / 1))
                else:
                    spf = spf + (d * (pagerank[j] / len(corpus[j])))
            pagerank[i] = sn + spf
            sse.append(math.fabs(pagerank[i] - testingrank[i]))
        if max(sse) <= 0.001:
            break
    return pagerank


if __name__ == "__main__":
    main()
