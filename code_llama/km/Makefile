FLAGS = -O3 -lmpi 

CC = mpicc

RM = rm -f

EXEC = km

all: $(EXEC)

$(EXEC): km.c
	$(CC) $(FLAGS) $< -o $@ -lm

clean:
	$(RM) $(EXEC)
