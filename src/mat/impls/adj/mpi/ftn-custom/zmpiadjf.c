#include <../src/mat/impls/adj/mpi/mpiadj.h>
#include <petsc/private/ftnimpl.h>
#include <petscmat.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define matcreatempiadj_ MATCREATEMPIADJ
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define matcreatempiadj_ matcreatempiadj
#endif

PETSC_EXTERN void matcreatempiadj_(MPI_Comm *comm, PetscInt *m, PetscInt *n, PetscInt *i, PetscInt *j, PetscInt *values, Mat *A, PetscErrorCode *ierr)
{
  Mat_MPIAdj *adj;

  CHKFORTRANNULLINTEGER(values);
  *ierr        = MatCreateMPIAdj(MPI_Comm_f2c(*(MPI_Fint *)&*comm), *m, *n, i, j, values, A);
  adj          = (Mat_MPIAdj *)(*A)->data;
  adj->freeaij = PETSC_FALSE;
}
