
#include "petsc.h"
#define MPI_SUCCESS 0
void * MPID_TMP  = 0;
int _v_[2] = {0, 0};
void *MPID_DUMMY =(void *)_v_;

int MPI_Attr_get( int comm, int key_val, void* attr_val, int* flg)
{
  *flg = 1;
  void *x;
  x = MPID_DUMMY;
  (*(void**)attr_val)  = x;
  return 0;
}

double MPI_Wtime()
{
  fprintf(stderr,"MPI_Wtime: use Petsctime instead\n");
  return 0.0;
}

/*     Fortran versions of several routines */

#if defined(__cplusplus)
extern "C" {
#endif

/******mpi_init*******/
void  mpi_init(int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void  mpi_init_(int *ierr)
{
  *ierr = MPI_SUCCESS;
}

void  MPI_INIT(int *ierr)
{
  *ierr = MPI_SUCCESS;
}

/******mpi_comm_size*******/
void mpi_comm_size(int *comm, int *size, int *ierr) 
{
  *size = 1;
  *ierr = 0;
}

void mpi_comm_size_(int *comm, int *size, int *ierr) 
{
  *size = 1;
  *ierr = 0;
}

void MPI_COMM_SIZE(int *comm, int *size, int *ierr) 
{
  *size = 1;
  *ierr = 0;
}

/******mpi_comm_rank*******/
void mpi_comm_rank(int *comm,int *rank,int *ierr)
{
  *rank=0;
  *ierr=MPI_SUCCESS;
}

void mpi_comm_rank_(int *comm,int *rank,int *ierr)
{
  *rank=0;
  *ierr=MPI_SUCCESS;
}

void MPI_COMM_RANK(int *comm,int *rank,int *ierr)
{
  *rank=0;
  *ierr=MPI_SUCCESS;
}

/******mpi_wtick*******/
double mpi_wtick() 
{
  fprintf(stderr,"MPI_Wtime: use PetscTime instead\n");
  return 0.0;
}

double mpi_wtick_() 
{
  fprintf(stderr,"MPI_Wtime: use PetscTime instead\n");
  return 0.0;
}

double MPI_WTICK() 
{
  fprintf(stderr,"MPI_Wtime: use PetscTime instead\n");
  return 0.0;
}

/*******mpi_wtime******/
double mpi_wtime()
{
  fprintf(stderr,"MPI_Wtime: use PetscTime instead\n");
  return 0.0;
}

double mpi_wtime_()
{
  fprintf(stderr,"MPI_Wtime: use PetscTime instead\n");
  return 0.0;
}

double MPI_WTIME()
{
  fprintf(stderr,"MPI_Wtime: use PetscTime instead\n");
  return 0.0;
}
#if defined(__cplusplus)
}
#endif

