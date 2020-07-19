# --------- run.sh // TESTING SECTION ---------------
# This particular section of run.sh adds additional arguments to our mpirun for various testing features
    # mpirun -np [a] ./lab3 [1] [2] [3] [4] [5] [6]
    # ________________ MANDATORY (for submission)
    # a: (int)    a - # of processors to use (1, 2, 4, 8, 16, 32, 64)
    #
    # 1: (int)    n - number of lines
    # 2: (int)    m - number of columns
    # 3: (int)    tp - number of timesteps/iterations
    # 4: (double) td - discretized time
    # 5: (float)  h - size of each tile subdivison (square h x h)

make all
echo ""
echo "<<<"
mpirun -np 16 ./lab3 10 10 20 0.01 1
echo ">>>"

echo ""
echo "<<<"
mpirun -np 16 ./lab3 5 9 300 0.01 1
echo ">>>"

make clean








# --------- run.sh // ORIGINAL VERSION ---------------
# make all

# echo ""
# echo "<<<"
# mpirun -np 2 ./lab3 9 9 360 0.00025 0.1
# echo ">>>"

# echo ""
# echo "<<<"
# mpirun -np 2 ./lab3 5 9 300 0.01 1
# echo ">>>"

# make clean
