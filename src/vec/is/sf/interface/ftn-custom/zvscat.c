#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define vecscattercreatetoall_  VECSCATTERCREATETOALL
  #define vecscattercreatetozero_ VECSCATTERCREATETOZERO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define vecscattercreatetoall_  vecscattercreatetoall
  #define vecscattercreatetozero_ vecscattercreatetozero
#endif

PETSC_EXTERN void vecscattercreatetoall_(Vec *vin, VecScatter *ctx, Vec *vout, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToAll(*vin, ctx, vout);
}

PETSC_EXTERN void vecscattercreatetozero_(Vec *vin, VecScatter *ctx, Vec *vout, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToZero(*vin, ctx, vout);
}
