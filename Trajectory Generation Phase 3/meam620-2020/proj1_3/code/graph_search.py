from heapq import heappush, heappop  # Recommended.
import numpy as np
from flightsim.world import World
from proj1_3.code.occupancy_map import OccupancyMap  # Recommended.


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """

    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    # print('start',start)
    # print('end',goal)
    # Initialization----------------------------------------------------------------------------------------------------
    # PQ = [(0, start_index)]  # Priority queue of open/ alive nodes
    g_v_cost = np.full(occ_map.map.shape, np.inf)  # Initially set all distances as inf
    p_v_parent = np.zeros((occ_map.map.shape[0], occ_map.map.shape[1], occ_map.map.shape[2], 3))  # parent node
    neighbour = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                neighbour.append([i, j, k])
    neighbour.remove([0, 0, 0])
    # ------------------------------------------------------------------------------------------------------------------
    # astar = True
    if not astar:  # Dijkstra's Algorithm

        g_v_cost[start_index] = 0  # distance from start to start is 0
        PQ = [(g_v_cost[start_index], start_index)]  # Priority queue initializes
        # , (g_v_cost[goal_index], goal_index)
        # heapq.heapify(PQ)
        # heapq.heapify(PQ)
        # print(PQ)
        #
        # print(g_v_cost[start_index])

        # heappop(PQ)
        #
        # print('new',PQ)

        # if (g_v_cost[goal_index],goal_index) in PQ:
        #     print('Yes')

        # print(np.min(g_v_cost))
        cnt = 0
        while len(PQ) > 0:
            # while (g_v_cost[goal_index], goal_index) in PQ and np.min(g_v_cost) < np.inf:
            min_element = heappop(PQ)  # eg: (0.0,(2,2,2))
            u = min_element[1]  # note: here u is tuple. cannot do addition like array

            for i in range(26):
                v = np.asarray(u) + neighbour[i]  # one neighbour
                if not occ_map.is_valid_index(v) or occ_map.is_occupied_index(v):
                    pass
                elif g_v_cost[(v[0], v[1], v[2])] != np.inf:
                    pass
                else:
                    d = g_v_cost[u] + np.linalg.norm(v - u)
                    if d < g_v_cost[
                        (v[0], v[1], v[2])]:  # need tuple to access g_v_cost # A shorter path to v has been found
                        g_v_cost[(v[0], v[1], v[2])] = d
                        heappush(PQ, (d, (v[0], v[1], v[2])))
                        p_v_parent[v[0], v[1], v[2], :] = u
                        # print(p_v_parent[v[0], v[1], v[2],:])
            cnt += 1
        # print('Dijk Node', cnt)

    else:  # A*
        F = np.full(occ_map.map.shape, np.inf)  # F(v) = g(v) + h(v)
        g_v_cost[start_index] = 0  # distance from start to start is 0
        F[start_index] = g_v_cost[start_index] + np.linalg.norm(np.asarray(goal_index) - np.asarray(start_index))

        PQ = [(F[start_index], start_index), (F[goal_index], goal_index)]  # Priority queue initializes

        # while len(PQ) > 0:
        count = 0
        while (F[goal_index], goal_index) in PQ and np.min(F) < np.inf:
            min_element = heappop(PQ)  # eg: (0.0,(2,2,2))
            u = min_element[1]  # note: here u is tuple. cannot do addition like array

            for i in range(26):
                v = np.asarray(u) + neighbour[i]  # one neighbour
                if not occ_map.is_valid_index(v) or occ_map.is_occupied_index(v):
                    pass
                elif g_v_cost[(v[0], v[1], v[2])] != np.inf:  # I dont have to find 26 neighbours  all the time
                    pass
                else:
                    d = g_v_cost[u] + np.linalg.norm(v - u)
                    if d < g_v_cost[(v[0], v[1], v[2])]:  # need tuple to access g_v_cost # A shorter path to v has been found
                        g_v_cost[(v[0], v[1], v[2])] = d
                        heappush(PQ, (d, (v[0], v[1], v[2])))
                        p_v_parent[v[0], v[1], v[2], :] = u

                        F[(v[0], v[1], v[2])] = g_v_cost[(v[0], v[1], v[2])] + np.linalg.norm(np.asarray(goal_index) - np.asarray(v))

            count += 1
        # print('A Star Node:', count)

    # Find Path---------------------------------------------------------------------------------------------------------

    Path = []
    temp = goal_index
    while (temp[0], temp[1], temp[2]) != (start_index[0], start_index[1], start_index[2]):
        Path.append(occ_map.index_to_metric_center(temp))
        temp = p_v_parent[int(temp[0]), int(temp[1]), int(temp[2])]
    Path.append(occ_map.index_to_metric_center(start_index))
    Path.append(start)
    Path.reverse()
    Path.append(goal)

    return np.asarray(Path)

    # ------------------------------------------------------------------------------------------------------------------
