from numpy import random as rd
from numpy import full, exp

INFINITE = 255**2 + 1

def patch_reordering(patches, B: int, epsilon: float, omega):
    '''
    Return two reordering arrays based on the
    parameter omega(distance function), the
    second array the inverse ordering of the 
    first one.
    '''

    # setup auxiliar data
    Np1, Np2, _ = patches.shape
    Np = Np1 * Np2
    half1 = half2 = (B >> 1)

    visited, unvisited, ordering = full((Np1, Np2), False), set(range(Np)), []

    # choise a random start patch
    last = (rd.choice(Np1), rd.choice(Np2)) 

    # mark as visited and add to the ordering
    visited[last] = True
    pos = last[1] * Np1 + last[0]
    unvisited.remove(pos)
    ordering.append(pos)

    # iter for each patch
    for _ in range(1, Np):
        # calculate the BxB neighborhood box
        i1, i2 = max(0, last[0] - half1), min(Np1, last[0] + half2)
        j1, j2 = max(0, last[1] - half1), min(Np2, last[1] + half2)

        # setup values of the 2 minimun distances
        min1_d = min2_d = INFINITE
        pos1 = pos2 = None

        # array for no matching patches in the neighborhood
        diff = []

        # iter for each patch in the neighborhood
        for i in range(i1, i2):
            for j in range(j1, j2):
                if not visited[i][j]: # if not visited calculate distance to this patch
                    d = omega(patches[last], patches[i][j])
                    if d < 0: # if no match, ignore it
                        diff.append((i, j))
                    elif d < min1_d: # if less than minimun, update values
                        pos2, min2_d = pos1, min1_d
                        pos1, min1_d = (i, j), d
                    elif d < min2_d: # if less than 2nd minimun, update values
                        pos2, min2_d = (i, j), d

        if pos1 is None: # no matching patch found
            if diff: # select 1 from no-matching patches
                last = diff[(len(diff) >> 1)]
            else: # select 1 from outside of neighborhood
                pos = unvisited.pop()
                ordering.append(pos)
                last = pos % Np1, pos // Np1
                visited[last] = True
                continue
        elif pos2 is None: # only 1 matching patch, select it
            last = pos1
        else: # select randomly one of the 2 minimun distance patches
            p1, p2 = exp(-min1_d / epsilon), exp(-min2_d / epsilon)
            psum = p1 + p2
            p1, p2 = p1 / psum, p2 / psum
            last = pos1 if rd.choice([True, False], p=[p1, p2]) else pos2

        # mark as visited last patch and add to the ordering
        visited[last] = True
        pos = last[1] * Np1 + last[0]
        unvisited.remove(pos)
        ordering.append(pos)

    # generating the inverse ordering
    iordering = [-1] * Np
    for i in ordering: 
        iordering[ordering[i]] = i

    return ordering, iordering