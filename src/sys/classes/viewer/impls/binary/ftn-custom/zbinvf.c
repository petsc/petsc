#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerfilesetmode_         PETSCVIEWERFILESETMODE
  #define petscviewerbinaryopen_          PETSCVIEWERBINARYOPEN
  #define petscviewerbinarygetdescriptor_ PETSCVIEWERBINARYGETDESCRIPTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerfilesetmode_         petscviewerfilesetmode
  #define petscviewerbinaryopen_          petscviewerbinaryopen
  #define petscviewerbinarygetdescriptor_ petscviewerbinarygetdescriptor
#endif

PETSC_EXTERN void petscviewerfilesetmode_(PetscViewer *viewer, PetscFileMode *type, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscViewerFileSetMode(v, *type);
}

PETSC_EXTERN void petscviewerbinaryopen_(MPI_Comm *comm, char *name, PetscFileMode *type, PetscViewer *binv, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;
  FIXCHAR(name, len, c1);
  *ierr = PetscViewerBinaryOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm), c1, *type, binv);
  if (*ierr) return;
  FREECHAR(name, c1);
}

PETSC_EXTERN void petscviewerbinarygetdescriptor_(PetscViewer *viewer, int *fd, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscViewerBinaryGetDescriptor(v, fd);
}
