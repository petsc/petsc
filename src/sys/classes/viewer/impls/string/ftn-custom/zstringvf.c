#include <petsc/private/ftnimpl.h>
#include <petscviewer.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define petscviewerstringopen_          PETSCVIEWERSTRINGOPEN
  #define petscviewerstringsetstring_     PETSCVIEWERSTRINGSETSTRING
  #define petscviewerstringgetstringread_ PETSCVIEWERSTRINGGETSTRINGREAD
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerstringopen_          petscviewerstringopen
  #define petscviewerstringsetstring_     petscviewerstringsetstring
  #define petscviewerstringgetstringread_ petscviewerstringgetstringread
#endif

PETSC_EXTERN void petscviewerstringopen_(MPI_Comm *comm, char *name, PetscViewer *str, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1)
{
  PETSC_FORTRAN_OBJECT_CREATE(str);
  *ierr = PetscViewerStringOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm), name, len1, str);
}

PETSC_EXTERN void petscviewerstringsetstring_(PetscViewer *str, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1)
{
  PetscViewer v_a = PetscPatchDefaultViewers(str);
  *ierr           = PetscViewerStringSetString(v_a, name, len1);
}

PETSC_EXTERN void petscviewerstringgetstringread_(PetscViewer *viewer, char string[], PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T l_string)
{
  size_t      len;
  char       *c_string = PETSC_NULLPTR;
  PetscViewer v_viewer = PetscPatchDefaultViewers(viewer);
  *ierr                = PetscViewerStringGetStringRead(v_viewer, (const char **)&c_string, &len);
  if (*ierr) return;
  *ierr = PetscStrncpy((char *)string, c_string, l_string);
  if (*ierr) return;
}
