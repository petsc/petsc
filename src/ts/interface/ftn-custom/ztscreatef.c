#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tscreate_                            TSCREATE
#define tsdestroy_                           TSDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tscreate_                            tscreate
#define tsdestroy_                           tsdestroy
#endif

PETSC_EXTERN void tscreate_(MPI_Comm *comm,TS *outts,PetscErrorCode *ierr)
{
  *ierr = TSCreate(MPI_Comm_f2c(*(MPI_Fint*)&*comm),outts);
}

PETSC_EXTERN void tsdestroy_(TS *x,int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = TSDestroy(x); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}
