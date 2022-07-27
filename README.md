# skeleton-based-Voronoi-path-planning
skeleton based Voronoi path planning，you can redevelop over this  
===================
Our work is based on Zhang-Suen thinning algorithm,we want to realize the algorithm projected by paper"Voronoi Path Planning Based on Improved Skeleton Extraction".

This is an implementation of an Improved Skeleton Based Voronoi Path Planning algorithm using OpenCV,The algorithm is explained in paper "Voronoi Path Planning Based on Improved Skeleton Extraction".In our code ,  we do not use smooth algorithm here,you can use your smooth algorithm based on the global path deque d3.

In order to extract the skeleton,we use Zhang-Suen thinning algorithm using OpenCV. The algorithm is explained in "A fast parallel algorithm for thinning digital patterns" by T.Y. Zhang and C.Y. Suen. See the tutorial in [opencv-code.com].

To use the path planniing function,compile the find_point.cpp,and then run.
In the cpp file,deque d3 store the global path as you can see when you run this cpp file. 


Contact
-------


Credit
------

