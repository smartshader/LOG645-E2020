make all
echo '>>> ==================================== [LAB 1] ===================================='
echo ''
echo '>>> __________________________________ [EVALUATION : Problem #1] __________________________________'
echo ''
echo '>>> 1 0 1'
./lab1 1 0 1
echo '<<<'

echo '>>> 1 5 3'
./lab1 1 5 3
echo '<<<'
echo '>>> __________________________________ [EVALUATION : Problem #2] __________________________________'
echo ''
echo '>>> 2 0 1'
./lab1 2 0 1
echo '<<<'

echo '>>> 2 5 3'
./lab1 2 5 3
echo '<<<'

# echo '>>> ==================================== [LAB 1] ===================================='
# echo ''
# echo '>>> __________________________________ [MEASUREMENTS : Problem #1 SEQ] __________________________________'
# echo "<<< Problem: 1, InitVal: 5"
# for i in {1..100..2}
# do
#     echo '>>> 1 5 '$i
#     ./lab1 1 5 $i
#     wait $PID
# done

# echo ''
# echo '>>> __________________________________ [MEASUREMENTS : Problem #2 SEQ] __________________________________'
# echo "<<< Problem: 2, InitVal: 5"
# for i in {1..100..2}
# do
#     echo '>>> 2 5 '$i
#     ./lab1 2 5 $i
#     wait $PID
# done

make clean
