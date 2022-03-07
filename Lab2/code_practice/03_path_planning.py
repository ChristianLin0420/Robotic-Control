import numpy as np
import cv2
import argparse

def pos_int(p):
    return (int(p[0]), int(p[1]))

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--planner", type=str, default="a_star", help="a_star/rrt/rrt_star")
    parser.add_argument("--smooth", action="store_true", help="true/false")
    args = parser.parse_args()

    # Read Map
    img = cv2.flip(cv2.imread("Maps/map2.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    m = 1-cv2.dilate(1-m, np.ones((20,20)))
    img = img.astype(float)/255.

    # Choose Planner
    start=(100,200)
    goal=(380,520)
    if args.planner == "a_star":
        from PathPlanning.planner_a_star import PlannerAStar as Planner
        planner = Planner(m, 20)
        path = planner.planning(start, goal, img=img)
    elif args.planner == "rrt":
        from PathPlanning.planner_rrt import PlannerRRT as Planner
        planner = Planner(m, 40)
        path = planner.planning(start, goal, img=img)
    elif args.planner == "rrt_star":
        from PathPlanning.planner_rrt_star import PlannerRRTStar as Planner
        planner = Planner(m, 30)
        path = planner.planning(start, goal, img=img)
    else:
        print("Unknown planner !!")
        exit(0)
    
    print(path)

    cv2.circle(img,(start[0],start[1]),5,(0,0,1),3)
    cv2.circle(img,(goal[0],goal[1]),5,(0,1,0),3)
    # Extract Path
    if not args.smooth:
        for i in range(len(path)-1):
            cv2.line(img, pos_int(path[i]), pos_int(path[i+1]), (1,0,0), 2)
    else:
        from PathPlanning.cubic_spline import *
        path = np.array(cubic_spline_2d(path, interval=1))
        for i in range(len(path)-1):
            cv2.line(img, pos_int(path[i]), pos_int(path[i+1]), (1,0,0), 2)
    
    img_ = cv2.flip(img,0)
    cv2.imshow("Path Planning",img_)
    k = cv2.waitKey(0)
