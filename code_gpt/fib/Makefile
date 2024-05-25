FLAGS = -O2

CC = mpicc

RM = rm -f

EXEC = fib

all: $(EXEC)

$(EXEC): fib.c
	$(CC) $(FLAGS) $< -o $@

clean:
	$(RM) $(EXEC)
