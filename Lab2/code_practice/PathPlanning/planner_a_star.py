from operator import ne
import cv2
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.h = {} # Distance from start to node
        self.g = {} # Distance from node to goal
        self.goal_node = None

    def planning(self, start=(100,200), goal=(375,520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize 
        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)

        while(1):
            min_dist = 99999
            min_id = None

            for idx, node in enumerate(self.queue):
                s_to_t = self.g[node]
                t_to_e = self.h[node]

                if s_to_t + t_to_e < min_dist:
                    min_dist = s_to_t + t_to_e
                    min_id = idx
                
            # get the nearsest node
            nearsest_node = self.queue.pop(min_id)

            # confront obstacle
            if self.map[nearsest_node[1], nearsest_node[0]] < 0.5:
                continue

            if utils.distance(nearsest_node, goal) < inter:
                self.goal_node = nearsest_node
                break

            # eight direction
            nodes = [   (nearsest_node[0] + inter, nearsest_node[1]), (nearsest_node[0], nearsest_node[1] + inter), 
                        (nearsest_node[0] - inter, nearsest_node[1]), (nearsest_node[0], nearsest_node[1] - inter), 
                        (nearsest_node[0] + inter, nearsest_node[1] + inter), (nearsest_node[0] - inter, nearsest_node[1] + inter), 
                        (nearsest_node[0] - inter, nearsest_node[1] - inter), (nearsest_node[0] + inter, nearsest_node[1] - inter)]

            for node in nodes:
                if node not in self.parent:
                    self.queue.append(node)
                    self.parent[node] = nearsest_node
                    self.g[node] = self.g[nearsest_node] + inter
                    self.h[node] = utils.distance(node, goal)
                elif self.g[node] > self.g[node] + inter:
                    self.parent[node] = nearsest_node
                    self.g[node] = self.g[nearsest_node] + inter

            if img is not None:
                cv2.circle(img, (start[0], start[1]), 5, (0, 0, 1), 3)
                cv2.circle(img, (goal[0], goal[1]), 5, (0, 1, 0), 3)
                cv2.circle(img, nearsest_node, 2, (0, 0, 1), 1)
                img_ = cv2.flip(img, 0)
                cv2.imshow("A* Test", img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break


        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while(True):
            path.insert(0,p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
