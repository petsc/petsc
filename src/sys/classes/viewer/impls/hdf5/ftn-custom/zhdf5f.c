#include <petsc/private/ftnimpl.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerhdf5opengroup_ PETSCVIEWERHDF5OPENGROUP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerhdf5opengroup_ petscviewerhdf5opengroup
#endif

PETSC_EXTERN void petscviewerhdf5opengroup_(PetscViewer *viewer, char path[], hid_t *fileId, hid_t *groupId, int *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(path, len, c1);
  *ierr = PetscViewerHDF5OpenGroup(*viewer, c1, fileId, groupId);
  FREECHAR(path, c1);
}
