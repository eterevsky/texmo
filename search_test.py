import heapq

import search

done = set()
todo = [(0, 'dense.1.tanh', 'dense.1.relu')]

for i in range(100000):
    dist, spec, parent = heapq.heappop(todo)
    done.add(spec)

    found_parent = False
    for neighbor in search.spec_neighbors(spec):
        assert neighbor != spec
        assert neighbor != ""

        if neighbor == parent:
            found_parent = True
        if neighbor not in done:
            heapq.heappush(todo, (dist + 1, neighbor, spec))

    # if not found_parent:
    #     print('parent:', parent, 'spec:', spec)

    # assert found_parent or parent.endswith('.relu')

