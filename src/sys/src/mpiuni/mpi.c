#include <stdio.h>
#define MPI_SUCCESS 0
#if defined(PARCH_sun4) && !defined(__cplusplus)
extern int fprintf(FILE*,const char*,...);
#endif

long MPID_DUMMY = 0;
void * _v_ = 0;
double MPI_Wtime()
{
 fprintf(stderr,"MPI_Wtime: use Petsctime instead\n");
 return 0.0;
}
void mpi_init__(int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void  MPI_INIT (int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void  mpi_init_ (int *ierr)
{
  *ierr = MPI_SUCCESS;
}

