FLAGS = -O3 -lssl -lcrypto

CC = gcc
NVCC = nvcc

RM = rm -f

EXEC = password_bf

all: $(EXEC)

$(EXEC): password_bf.o
	$(NVCC) password_bf.o -o $(EXEC) $(FLAGS)

password_bf.o: password_bf.c
	$(NVCC) -c password_bf.c -o password_bf.o $(FLAGS)

run:
	./$(EXEC)

clean:
	$(RM) password_bf.o $(EXEC)
