ALL: libmpi

CC	= gcc

libmpi: mpi.o
	ar r libmpi.a mpi.o
	ranlib libmpi.a
	$(RM) *.o

clean:
	$(RM) *.o
