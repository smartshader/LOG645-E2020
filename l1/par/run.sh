make all

echo '>>> ==================================== [LAB 1] ===================================='
echo ''
echo '>>> __________________________________ [EVALUATION : Problem #1 Par] __________________________________'
echo ''
echo '>>> 1 0 1'
mpirun -np 16 ./lab1 1 0 1
echo '<<<'

echo '>>> 1 5 3'
mpirun -np 16 ./lab1 1 5 3
echo '<<<'
echo '>>> __________________________________ [EVALUATION : Problem #2 Par] __________________________________'
echo ''
echo '>>> 2 0 1'
mpirun -np 16 ./lab1 2 0 1
echo '<<<'

echo '>>> 2 5 3'
mpirun -np 16 ./lab1 2 5 3
echo '<<<'

make clean
