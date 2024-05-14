#include <petsc/private/fortranimpl.h>
#include <petscpc.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcview_            PCVIEW
  #define pcgetoperators_    PCGETOPERATORS
  #define pcviewfromoptions_ PCVIEWFROMOPTIONS
  #define pcdestroy_         PCDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcview_            pcview
  #define pcgetoperators_    pcgetoperators
  #define pcviewfromoptions_ pcviewfromoptions
  #define pcdestroy_         pcdestroy
#endif

PETSC_EXTERN void pcview_(PC *pc, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PCView(*pc, v);
}

PETSC_EXTERN void pcviewfromoptions_(PC *ao, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PCViewFromOptions(*ao, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}

PETSC_EXTERN void pcdestroy_(PC *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = PCDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}
