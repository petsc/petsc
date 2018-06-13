
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fillupvector_            FILLUPVECTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fillupvector_            fillupvector
#endif

PETSC_EXTERN void fillupvector(Vec *v,PetscErrorCode *ierr)
{
  *ierr = VecSet(*v,1.0);
  return;
}

