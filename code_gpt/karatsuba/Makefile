FLAGS = -O3 -fopenmp

CXX = mpicc

RM = rm -f

EXEC = karatsuba

all: $(EXEC)

$(EXEC):
	$(CXX) $(FLAGS) $(EXEC).c -o $(EXEC)

clean:
	$(RM) $(EXEC).o $(EXEC)
