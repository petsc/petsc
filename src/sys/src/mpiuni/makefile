ALL: libmpi

libmpi: mpi.o
	ar r libmpi.a mpi.o
	ranlib libmpi.a
	$(RM) *.o

clean:
	$(RM) *.
