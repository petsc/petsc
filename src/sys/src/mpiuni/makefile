ALL: libmpi

CC	= gcc

libmpi: mpi.o mpif.o
	ar r libmpi.a mpi.o mpif.o
	ranlib libmpi.a
	$(RM) *.o

clean:
	$(RM) *.o
