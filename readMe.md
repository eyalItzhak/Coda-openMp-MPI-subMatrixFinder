## Cuda-openMp-MPI 

A simple project written in C demonstrating the use of Cuda-openMp-MPI tools.

In this project there are matrices (in the input file in the demo are very large) and images that need to be located if they appear in one of the matrices and if so where.

This is a seemingly simple problem that can be solved with a simple algorithm, but it will take a long time for this algorithm to find the answers.


In this solution we try to get more processing power in order to shorten the times by several methods:
1. Division of the problem into sub-problems when each process checks whether one of the images is on the matrix.
2. Each such process addresses via CUDA the video card to calculate whether a single image is in any thread.
3. **If** necessary, the code supports sending data (tested in Linux) to another computer on the same network to obtain its processing power for further shortening of times.


Dynamic memory and its deletion, reading and writing to a file, and structures are also used.
