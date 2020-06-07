# 2020_E_LOG645_LABO1
Pour lancer le programme
/par run.sh
/seq run.sh

Ou:
cd /par
make all
mpirun -np NB_PROCESSEUR ./lab1 Problem Valeur_départ nb_Itération
make clean

cd /seq
make all
./lab1 Problem Valeur_départ nb_Itération
make clean