FLAGS=-O3 -lssl -lcrypto

NVCC=nvcc

RM=rm -f

EXEC=password_bf

all: $(EXEC)

$(EXEC):
	$(NVCC) $(FLAGS) password_bf.cu -o $(EXEC)

run:
	./$(EXEC)

clean:
	$(RM) $(EXEC)