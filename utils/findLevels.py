import numpy as np


def find_levels(sigIn, expected_levels=4, max_iteration=10000):
    print(f"Trying to find {expected_levels} signal levels from data.")
    n=0
    clusters = np.array([])
    while 1:
        if sigIn[n] not in clusters:
            clusters = np.append(clusters, sigIn[n])
        if len(clusters)==expected_levels:
            break
        if n>max_iteration:
            print(f'{len(clusters)} levels found. Scanning too long to find required levels, quit...')
            break
        n = n + 1
    clusters.sort()
    print(f'{len(clusters)} signal level clusters found: {clusters}')
    return clusters, n