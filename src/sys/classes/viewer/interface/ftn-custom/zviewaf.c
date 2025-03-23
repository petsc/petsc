#include <petsc/private/ftnimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerandformatcreate_  PETSCVIEWERANDFORMATCREATE
  #define petscviewerandformatdestroy_ PETSCVIEWERANDFORMATDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerandformatcreate_  petscviewerandformatcreate
  #define petscviewerandformatdestroy_ petscviewerandformatdestroy
#endif

PETSC_EXTERN void petscviewerandformatcreate_(PetscViewer *vin, PetscViewerFormat *format, PetscViewerAndFormat **vf, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerAndFormatCreate(v, *format, vf);
}

PETSC_EXTERN void petscviewerandformatdestroy_(PetscViewerAndFormat **vf, PetscErrorCode *ierr)
{
  *ierr = PetscViewerAndFormatDestroy(vf);
}
