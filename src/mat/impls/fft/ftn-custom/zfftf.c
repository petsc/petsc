#include <petsc/private/fortranimpl.h>
#include <petsc/private/matimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matcreatefft_ MATCREATEFFT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matcreatefft_ matcreatefft
#endif

PETSC_EXTERN void matcreatefft_(MPI_Comm *comm,PetscInt *ndim,PetscInt *dim,char* type_name,Mat *A,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *mattype;

  FIXCHAR(type_name,len,mattype);
  *ierr = MatCreateFFT(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*ndim,dim,mattype,A);if (*ierr) return;
  FREECHAR(type_name,mattype);
}
