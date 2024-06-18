import math
import random

from deap import tools


def sel_random(individuals, buckets, k):
    chosen_inds = []
    chosen_buckets = []
    for i in range(k):
        r = random.randrange(len(individuals))
        chosen_inds.append(individuals[r])
        chosen_buckets.append(buckets[r])

    return chosen_inds, chosen_buckets


def sel_least_complex(individuals, complexity_func):
    if len(individuals) == 1:
        return individuals[0]
    else:
        lowest_complexity = math.inf
        for ind in individuals:
            complexity = complexity_func(ind)
            if complexity < lowest_complexity:
                lowest_complexity = complexity
                least_complex = ind
        return least_complex


def selElitistAndTournament(individuals, k, tournsize, elitism):
    return tools.selBest(individuals, elitism) + tools.selTournament(individuals, k - elitism, tournsize)


def selElitistAndselEpsilonLexicase(individuals, k, tournsize, elitism):
    return tools.selBest(individuals, elitism) + tools.selEpsilonLexicase(individuals, k - elitism, tournsize)