#include <petsc/private/fortranimpl.h>
#include <petscpc.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcdestroy_ PCDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcdestroy_ pcdestroy
#endif

PETSC_EXTERN void pcdestroy_(PC *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = PCDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}
