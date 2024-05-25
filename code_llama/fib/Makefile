FLAGS=-O2

CC=mpicc

RM=rm -f

EXEC=fib_mpi

all: $(EXEC)

$(EXEC):
	$(CC) $(FLAGS) fib_mpi.c -o $(EXEC)

run:
	mpirun -np 4 ./$(EXEC) < input

clean:
	$(RM) $(EXEC)