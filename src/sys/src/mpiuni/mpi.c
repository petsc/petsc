long MPID_DUMMY = 0;
#define MPI_SUCCESS 0
double MPI_Wtime()
{
 printf("MPI_Wtime: use Petsctime instead\n");
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

