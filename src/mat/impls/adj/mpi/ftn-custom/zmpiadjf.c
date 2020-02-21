#include <../src/mat/impls/adj/mpi/mpiadj.h>
#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatempiadj_                 MATCREATEMPIADJ
#define matmpiadjsetpreallocation_       MATMPIADJSETPREALLOCATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatempiadj_                 matcreatempiadj
#define matmpiadjsetpreallocation_       matmpiadjsetpreallocation
#endif

PETSC_EXTERN void matcreatempiadj_(MPI_Comm *comm,PetscInt *m,PetscInt *n,PetscInt *i,PetscInt *j,PetscInt *values,Mat *A,PetscErrorCode *ierr)
{
  Mat_MPIAdj *adj;

  CHKFORTRANNULLINTEGER(values);
  *ierr        = MatCreateMPIAdj(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*m,*n,i,j,values,A);
  adj          = (Mat_MPIAdj*)(*A)->data;
  adj->freeaij = PETSC_FALSE;
}

PETSC_EXTERN void matmpiadjsetpreallocation_(Mat *mat,PetscInt *i,PetscInt *j,PetscInt *values, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(values);
  *ierr = MatMPIAdjSetPreallocation(*mat,i,j,values);
}

