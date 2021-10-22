from numpy import e as E
import numpy.random as rd
from time import perf_counter

def constructive_method(n, w):
    path = list(range(n))
    rd.shuffle(path)

    solution = [(-1, 0)] * n
    distance = 0

    for u, v in zip(path[:-1], path[1:]):
        d = w(u, v)
        solution[u] = (v, d)
        distance += d

    solution.append((path[0], path[-1]))

    return solution, distance


def reverse_path(solution, u, v):
    x, (y, d) = u, solution[u]
    
    while x != v:
        aux = y
        x, (y, d), solution[aux] = y, solution[y], (x, d)
        b, e = solution[-1]
        
    if u == b:
        solution[-1] = (v, solution[-1][1])
    
    if v == e:
        solution[-1] = (solution[-1][0], u)
        solution[u] = (-1, 0) 
        

def destroy(solution, distance):
    u = rd.choice(len(solution) - 1)
    v, d = solution[u]

    distance -= d

    return solution, distance, (u, v)


def repair(w, solution, distance, destroyed):
    u, v = destroyed

    if v != -1:
        b, e = solution[-1]

        d1, d2, d3 = w(e, b), w(u, e), w(b, v)

        d = min(d1, d2, d3)

        if d == d1:
            solution[e] = (b, d1)
            solution[u] = (-1, 0)
            solution[-1] = (v, u)
        elif d == d2:
            reverse_path(solution, v, e)
            solution[u] = (e, d2)
        else:
            reverse_path(solution, b, u)
            solution[b] = (v, d3)

        distance += d
    return solution, distance


def accept(tdistance, distance, temperature, d_ratio):
    if tdistance <= distance:
        return True, temperature

    prob = E ** ((distance - tdistance) / temperature)
    
    if rd.choice([True, False], p=[prob, 1 - prob]):
        return True, temperature * d_ratio

    return False, temperature


def lns_method(solution, disatnce, w, duration=1.0, temperature=2.0**8, d_ratio=0.5):
    x, x_count = solution.copy(), disatnce
    xb, xb_count = x, x_count

    start = perf_counter()
    while perf_counter() - start <= duration:
        xt, xt_count = repair(w, *destroy(x.copy(), x_count))

        acc, temperature = accept(xt_count, x_count, temperature, d_ratio)
        if acc:
            x, x_count = xt, xt_count

        if xt_count < xb_count:
            xb, xb_count = xt, xt_count

    return xb, xb_count


def alns_method(solution, distance, w, duration=1.0, W=(64.0,8.0,1.0,0.125), decay=0.875, temperature=2.0**8, d_ratio=0.5):
    x, x_count = solution.copy(), distance
    xb, xb_count = x, x_count

    # destroys
    len_d = sum_d = 10
    probs_d = [1.0 for _ in range(len_d)]
    destroys = [destroy for _ in range(len_d)]

    # repair
    len_r = sum_r = 10
    probs_r = [1.0 for _ in range(len_r)]
    repairs = [repair for _ in range(len_r)]

    start = perf_counter()
    while perf_counter() - start <= duration:
        arg_d = rd.choice(len_d, p=[prob / sum_d for prob in probs_d])
        arg_r = rd.choice(len_r, p=[prob / sum_r for prob in probs_r])

        xt, xt_count = repairs[arg_r](w, *destroys[arg_d](x.copy(), x_count))

        score = W[3]

        acc, temperature = accept(xt_count, x_count, temperature, d_ratio)
        if acc:
            x, x_count = xt, xt_count
            score = W[2]

        if xt_count < x_count:
            score = W[1]

        if xt_count < xb_count:
            xb, xb_count = xt, xt_count
            score = W[0]

        sum_d, sum_r = sum_d - probs_d[arg_d], sum_r - probs_r[arg_r]
        probs_d[arg_d] = decay * probs_d[arg_d] + (1.0 - decay) * score 
        probs_r[arg_r] = decay * probs_r[arg_r] + (1.0 - decay) * score 
        sum_d, sum_r = sum_d + probs_d[arg_d], sum_r + probs_r[arg_r]

    return xb, xb_count

def patch_reordering(patches, w, duration=1.0):
    _w = lambda i, j: w(patches[i], patches[j])

    N = len(patches)
    constructive, distance = constructive_method(N, _w)

    solution, distance = lns_method(constructive, distance, _w, duration)

    b, e = solution[-1]
    ordering = [b]

    while b != e:
        b, _ = solution[b]
        ordering.append(b)

    iordering = [-1] * N
    for i in ordering: iordering[ordering[i]] = i

    per = lambda M: [M[i] for i in ordering]
    iper = lambda M: [M[i] for i in iordering]

    return per, iper