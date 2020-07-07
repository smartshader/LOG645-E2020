make clean
make all
## Measurements ##

_detailed="0"

echo ""
echo "<<< Measurements 1.A) Variation sur colonnes (m), CPU : 64"
for i in {50..500..50}
do
  mpirun -np 64 ./lab3 50 $i 100 0.00025 1 $_detailed
  wait $PID
done
echo ">>>"

echo ""
echo "<<< Measurements 1.B) Variation sur lignes (n), CPU : 64"
for i in {50..500..50}
do
  mpirun -np 64 ./lab3 $i 50 100 0.00025 1 $_detailed
  wait $PID
done
echo ">>>"

echo ""
echo "<<< Measurements 1.C) Variation sur # pas de temps, CPU : 64"
for i in {50..500..50}
do
  mpirun -np 64 ./lab3 50 50 $i 0.00025 1 $_detailed
  wait $PID
done
echo ">>>"

echo ""
echo "<<< Measurements 2) n = 300, m = 200,  np = 100. Variation accel selon le nombre de processus P."
mpirun -np 1 ./lab3 300 200 100 0.00025 1 $_detailed
mpirun -np 2 ./lab3 300 200 100 0.00025 1 $_detailed
mpirun -np 4 ./lab3 300 200 100 0.00025 1 $_detailed
mpirun -np 8 ./lab3 300 200 100 0.00025 1 $_detailed
mpirun -np 16 ./lab3 300 200 100 0.00025 1 $_detailed
mpirun -np 32 ./lab3 300 200 100 0.00025 1 $_detailed
mpirun -np 64 ./lab3 300 200 100 0.00025 1 $_detailed
echo ">>>"


echo ""
echo "<<< Measurements 3) CPU = 16. Ratio plaque l = 1.5 (n), L = 1 (m). np = 100. Variation h : 0,15 - 0,0025. 9 data points. Variation Acceleration selon nombres de subdivisions"
mpirun -np 16 ./lab3	10	7 100	0.00025	0.1500 $_detailed
mpirun -np 16 ./lab3	15 10	100	0.00025	0.1000 $_detailed
mpirun -np 16 ./lab3	24 16	100	0.00025	0.0625 $_detailed
mpirun -np 16 ./lab3	30 20	100	0.00025	0.0500 $_detailed
mpirun -np 16 ./lab3	60 40	100	0.00025	0.0250 $_detailed
mpirun -np 16 ./lab3	75 50	100	0.00025	0.0200 $_detailed
mpirun -np 16 ./lab3	120 80 100	0.00025	0.0125 $_detailed
mpirun -np 16 ./lab3	150	100	100	0.00025	0.0100 $_detailed
mpirun -np 16 ./lab3	300	200	100	0.00025	0.0050 $_detailed
mpirun -np 16 ./lab3	600	400	100	0.00025	0.0025 $_detailed


make clean
