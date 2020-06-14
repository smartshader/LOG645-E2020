make all

echo ''
echo '>>> 16 1 10 2'
./lab2 16 1 10 2
echo '<<<'

echo ''
echo '>>> 16 1 10 4'
./lab2 16 1 12 4
echo '<<<'

echo ''
echo '>>> 16 1 10 3'
./lab2 16 1 12 3
echo '<<<'

echo '>>> ________ [EVALUATION : Problem #1] ________'
echo ''
echo '>>> 53 1 3 600 (Acceleration doit etre plus que 45)'
./lab2 53 1 3 600
echo '<<<'
# echo '>>> ________ [EVALUATION : Problem #2] ________'
# echo ''
# echo '>>> 23 2 7 600 (Acceleration doit etre plus que 15)'
# ./lab2 23 2 7 600
# echo '<<<'

make clean
