# Exécuter laplace_seq avec différentes valeurs de OMP_NUM_THREADS
for threads in 1 2 4 8 16 20; do
    echo "Running laplace_seq with OMP_NUM_THREADS=$threads"
    export OMP_NUM_THREADS=$threads
    ./laplace_seq
done
