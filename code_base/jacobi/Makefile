CC=gcc
FLAGS=-O3 -std=c99
EXEC=laplace_seq

all: $(EXEC)

$(EXEC):
	$(CC) $(FLAGS) $(EXEC).c   -c -o $(EXEC).o
	$(CC) $(FLAGS) $(EXEC).o -o $(EXEC)

clean:
	rm -f laplace_seq *.o
