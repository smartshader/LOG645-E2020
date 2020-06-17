make all

echo '>>> ________ [EVALUATION : Problem #1] ________'
echo ''
echo '>>> 16 1 10 2'
./lab2 16 1 10 2
echo '<<<'

echo '>>> ________ [EVALUATION : Problem #1 sur 600 iterations] ________'
echo ''
echo '>>> 53 1 3 600'
./lab2 53 1 3 600
echo '<<<'

echo '>>> ________ [EVALUATION : Problem #2] ________'
echo ''
echo '>>> 39 2 10 3'
./lab2 39 2 10 3
echo '<<<'

echo '>>> ________ [EVALUATION : Problem #2 sur 600 iterations] ________'
echo ''
echo '>>> 23 2 7 600'
./lab2 23 2 7 600
echo '<<<'

make clean
