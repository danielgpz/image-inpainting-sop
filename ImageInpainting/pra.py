from numpy import e as E
from numpy import random as rd


def get_neighborhood(shapes: tuple, patches: list, pos, B:int):
    '''
    Return the BxB square neighborhood arround
    the patche at position 'pos' in patches on
    a matrix of given 'shapes'
    '''

    a, b = pos // shapes[0], pos % shapes[0]

    half1 = B // 2
    half2 = B - half1

    a1, a2 = max(0, a - half1), min(shapes[1] - 1, a + half2)
    b1, b2 = max(0, b - half1), min(shapes[0] - 1, b + half2)

    return [i * shapes[0] + j for i in range(a1, a2) for j in range(b1, b2)]


def patch_reordering(shapes: tuple, patches, w, e: float, B: int):
    '''
    Return two reordering functions based
    on the parameter w(distance function),
    the second one is the inverse of the
    first one
    '''

    N1, N2 = shapes
    N = len(patches)

    assert N1 * N2 == N, f'The multiplication of shapes: {shapes} must result the total of patches: {N}'

    unvisited, visited, ordering = set(range(N)), [False] * N, []

    last = rd.choice(N)

    unvisited.discard(last)
    visited[last] = True
    ordering.append(last)

    for _ in range(1, N):
        neighborhood = get_neighborhood(shapes, patches, last, B)
        diff = [neighbor for neighbor in neighborhood if not visited[neighbor]]

        if not diff:
            last = unvisited.pop()
        elif len(diff) == 1:
            last = diff.pop()
        else:
            j1, j2 = diff.pop(), diff.pop()
            d1, d2 = w(patches[last], patches[j1]), w(patches[last], patches[j2])

            if d1 > d2:
                j1, j2 = j2, j1
                d1, d2 = d2, d1

            for j in diff:
                d = w(patches[last], patches[j])

                if d < d1:
                    j1, j2 = j, j1
                    d1, d2 = d, d1
                elif d < d2:
                    j2 = j
                    d2 = d

            p1, p2 = E ** (- d1 / e), E ** (- d2 / e)
            psum = p1 + p2
            p1, p2 = p1 / psum, p2 / psum

            last = rd.choice([j1, j2], p=[p1, p2])

        unvisited.discard(last)
        visited[last] = True
        ordering.append(last)

    iordering = [-1] * N
    for i in ordering: iordering[ordering[i]] = i

    per = lambda M: [M[i] for i in ordering]
    iper = lambda M: [M[i] for i in iordering]

    return per, iper