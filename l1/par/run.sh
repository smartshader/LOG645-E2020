make all

echo ''
echo '>>> 1 0 1'
mpirun -np 16 ./lab1 1 0 1
echo '<<<'

echo '>>> 1 5 3'
# mpirun -np 16 ./lab1 1 5 3
echo '<<<'

echo '>>> 2 0 1'
mpirun -np 16 ./lab1 2 0 1
echo '<<<'

echo '>>> 2 5 3'
mpirun -np 16 ./lab1 2 5 3
echo '<<<'

make clean
