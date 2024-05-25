EXEFILE = mdf
CPUCC = mpicc
CPPFLAGS = -std=c11 -m64 -O3
DEFS =
INCLUDES =
LIBDIR =
LIBS = -lm
LINK = $(LIBDIR) $(LIBS)

CPU_COMPILE = $(CPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)

all: mdf

$(EXEFILE): mdf.o
	$(CPU_COMPILE) mdf.o $(LINK) -o $(EXEFILE)

mdf.o: mdf.c
	$(CPU_COMPILE) -c mdf.c

clean:
	rm -f *.o $(EXEFILE)
