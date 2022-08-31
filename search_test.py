import heapq

import search

done = set()
todo = [(0, "suffix.4-dense.128.relu-norm", None)]

for i in range(1000000):
    dist, spec, parent = heapq.heappop(todo)
    done.add(spec)

    for neighbor in search.spec_neighbors(spec):
        assert neighbor != spec
        assert neighbor != ""
        assert "norm-norm" not in neighbor

        if neighbor not in done:
            heapq.heappush(todo, (dist + 1, neighbor, spec))

    # if not found_parent:
    #     print('parent:', parent, 'spec:', spec)

    # assert found_parent or parent.endswith('.relu')
