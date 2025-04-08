#include <petsc/private/ftnimpl.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerhdf5opengroup_            PETSCVIEWERHDF5OPENGROUP
  #define petscviewerhdf5writeattributeint_    PETSCVIEWERHDF5WRITEATTRIBUTEINT
  #define petscviewerhdf5writeattributescalar_ PETSCVIEWERHDF5WRITEATTRIBUTESCALAR
  #define petscviewerhdf5writeattributereal_   PETSCVIEWERHDF5WRITEATTRIBUTEREAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerhdf5opengroup_            petscviewerhdf5opengroup
  #define petscviewerhdf5writeattributeint_    petscviewerhdf5writeattributeint
  #define petscviewerhdf5writeattributescalar_ petscviewerhdf5writeattributescalar
  #define petscviewerhdf5writeattributereal_   petscviewerhdf5writeattributereal
#endif

PETSC_EXTERN void petscviewerhdf5opengroup_(PetscViewer *viewer, char path[], hid_t *fileId, hid_t *groupId, int *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(path, len, c1);
  *ierr = PetscViewerHDF5OpenGroup(*viewer, c1, fileId, groupId);
  FREECHAR(path, c1);
}

PETSC_EXTERN void petscviewerhdf5writeattributeint_(PetscViewer *viewer, const char parent[], const char name[], PetscInt *value, int *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char *c1;
  char *c2;

  FIXCHAR(parent, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscViewerHDF5WriteAttribute(*viewer, c1, c2, PETSC_INT, value);
  FREECHAR(parent, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscviewerhdf5writeattributescalar_(PetscViewer *viewer, const char parent[], const char name[], PetscScalar *value, int *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char *c1;
  char *c2;

  FIXCHAR(parent, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscViewerHDF5WriteAttribute(*viewer, c1, c2, PETSC_SCALAR, value);
  FREECHAR(parent, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscviewerhdf5writeattributereal_(PetscViewer *viewer, const char parent[], const char name[], PetscReal *value, int *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char *c1;
  char *c2;

  FIXCHAR(parent, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscViewerHDF5WriteAttribute(*viewer, c1, c2, PETSC_REAL, value);
  FREECHAR(parent, c1);
  FREECHAR(name, c2);
}
