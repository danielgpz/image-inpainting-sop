from numpy import random as rd
from numpy import full, exp


INFINITE = 255**2 + 1

def patch_reordering(shape: tuple, patches, B: int, epsilon: float, omega):
    '''
    Return two reordering arrays based on the
    parameter omega(distance function), the
    second array the inverse ordering of the 
    first one.
    '''

    # setup auxiliar data
    Np1, Np2 = shape
    Np = Np1 * Np2
    half1 = half2 = (B >> 1)

    visited, unvisited, ordering = full(Np, False), set(range(Np)), []

    # choise a random start patch
    last = rd.choice(Np) 

    # mark as visited and add to the ordering
    visited[last] = True
    unvisited.remove(last)
    ordering.append(last)

    # iter for each patch
    for _ in range(1, Np):
        # calculate the BxB neighborhood box
        i, j = last % Np1, last // Np1
        il, ir = max(0, i - half1), min(Np1, i + half2)
        jl, jr = max(0, j - half1), min(Np2, j + half2)

        # setup values of the 2 minimun distances
        min1_d = min2_d = INFINITE
        pos1 = pos2 = -1

        # array for no matching patches in the neighborhood
        diff = []

        # iter for each patch in the neighborhood
        for ii in range(il, ir):
            for pos in range(ii + jl * Np1, ii + jr * Np1, Np1):
                if not visited[pos]: # if not visited calculate distance to this patch
                    d = omega(patches[last], patches[pos])
                    if d < 0: # if no match, ignore it
                        diff.append(pos)
                    elif d < min1_d: # if less than minimun, update values
                        pos2, min2_d = pos1, min1_d
                        pos1, min1_d = pos, d
                    elif d < min2_d: # if less than 2nd minimun, update values
                        pos2, min2_d = pos, d

        if pos1 == -1: # no matching patch found
            if diff: # select 1 from no-matching patches
                last = diff[(len(diff) >> 1)]
            else: # select 1 from outside of neighborhood
                last = unvisited.pop()
                visited[last] = True
                ordering.append(last)
                continue
        elif pos2 == -1: # only 1 matching patch, select it
            last = pos1
        else: # select randomly one of the 2 minimun distance patches
            p1, p2 = exp(-min1_d / epsilon), exp(-min2_d / epsilon)
            psum = p1 + p2
            p1, p2 = p1 / psum, p2 / psum
            last = rd.choice([pos1, pos2], p=[p1, p2])

        # mark as visited last patch and add to the ordering
        visited[last] = True
        unvisited.remove(last)
        ordering.append(last)

    # generating the inverse ordering
    iordering = [-1] * Np
    for i in ordering: 
        iordering[ordering[i]] = i

    return ordering, iordering