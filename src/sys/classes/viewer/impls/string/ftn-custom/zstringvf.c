#include <petsc/private/ftnimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerstringopen_          PETSCVIEWERSTRINGOPEN
  #define petscviewerstringsetstring_     PETSCVIEWERSTRINGSETSTRING
  #define petscviewerstringgetstringread_ PETSCVIEWERSTRINGGETSTRINGREAD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
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

PETSC_EXTERN void petscviewerstringgetstringread_(PetscViewer *a, char b[], PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T l_b)
{
  size_t      len;
  char       *c_b = PETSC_NULLPTR;
  PetscViewer v_a = PetscPatchDefaultViewers(a);
  *ierr           = PetscViewerStringGetStringRead(v_a, (const char **)&c_b, &len);
  if (*ierr) return;
  *ierr = PetscStrncpy((char *)b, c_b, l_b);
  if (*ierr) return;
}
