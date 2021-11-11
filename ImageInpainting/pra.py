from numpy import random as rd
from numpy import full, exp


INFINITE = 255**2 + 1

def patch_reordering(shape: tuple, patches, empty_patches, B: int, epsilon: float, omega):
    '''
    Return two reordering arrays based on the
    parameter omega(distance function), the
    second array the inverse ordering of the 
    first one.
    '''

    # setup auxiliar data
    Np1, Np2 = shape
    Np = Np1 * Np2
    halfB = (B >> 1)
    unvisited, ordering = full(Np, True), []

    # choise a random start patch
    last = rd.choice(Np)
    # mark as visited and add to the ordering
    unvisited[last] = False
    ordering.append(last)

    # iter for each patch
    for _ in range(1, Np):
        # calculate the BxB neighborhood box
        i, j = last % Np1, last // Np1
        il, ir, jl, jr  = i, i + 1, j, j + 1
        # setup values of the 2 minimun distances
        min1_d = min2_d = INFINITE
        pos1 = pos2 = -1
        # array for no matching patches in the neighborhood
        empty_last, no_matches = empty_patches[last], []
        # iter for each patch in the neighborhood
        for l in range(max(il, jl, Np1 - ir, Np2 - jr)):
            if (l >= halfB or empty_last) and (no_matches or pos1 > -1):
                break
            ranges = []
            if il > 0:
                il -= 1
                ranges.append(range(jl*Np1 + il, jr*Np1 + il, Np1))
            if ir < Np1:
                ir += 1
                ranges.append(range(jl*Np1 + ir - 1, jr*Np1 + ir - 1, Np1))
            if jl > 0:
                jl -= 1
                ranges.append(range(jl*Np1 + il, jl*Np1 + ir))
            if jr < Np2:
                jr += 1
                ranges.append(range((jr - 1)*Np1 + il, (jr - 1)*Np1 + ir))
                    
            for rg in ranges:
                for pos in rg:
                    if unvisited[pos]:
                        if empty_patches[pos] and not empty_last: # if empty ignore
                            no_matches.append(pos)
                        elif empty_last: # if last is empty, take any unvisited
                            pos1 = pos
                        else: # both last and current arent empty
                            d = omega(patches[last], patches[pos])
                            if d < 0: # if no match, ignore it
                                no_matches.append(pos)
                            elif d < min1_d: # if less than minimun, update values
                                pos2, min2_d = pos1, min1_d
                                pos1, min1_d = pos, d
                            elif d < min2_d: # if less than 2nd minimun, update values
                                pos2, min2_d = pos, d

        if pos1 == -1: # no matching patch found
            if no_matches: # select 1 from no-matching patches
                last = no_matches[0]
            else: # select 1 from outside of neighborhood       
                rest = [i for i, u in enumerate(unvisited) if u]
                print(f'{len(rest)}/{Np}')
                ordering.extend(rest)
                break
        elif pos2 == -1: # only 1 matching patch, select it
            last = pos1
        else: # select randomly one of the 2 minimun distance patches
            p1, p2 = exp(-min1_d / epsilon), exp(-min2_d / epsilon)
            psum = p1 + p2
            p1, p2 = p1 / psum, p2 / psum
            last = rd.choice([pos1, pos2], p=[p1, p2])

        # mark as visited last patch and add to the ordering
        unvisited[last] = False
        ordering.append(last)

    # generating the inverse ordering
    iordering = [-1] * Np
    for i in ordering: 
        iordering[ordering[i]] = i

    return ordering, iordering