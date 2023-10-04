#include <petsc/private/fortranimpl.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerhdf5open_           PETSCVIEWERHDF5OPEN
  #define petscviewerhdf5pushgroup_      PETSCVIEWERHDF5PUSHGROUP
  #define petscviewerhdf5getgroup_       PETSCVIEWERHDF5GETGROUP
  #define petscviewerhdf5hasdataset_     PETSCVIEWERHDF5HASDATASET
  #define petscviewerhdf5hasattribute_   PETSCVIEWERHDF5HASATTRIBUTE
  #define petscviewerhdf5writeattribute_ PETSCVIEWERHDF5WRITEATTRIBUTE
  #define petscviewerhdf5readattribute_  PETSCVIEWERHDF5READATTRIBUTE
  #define petscviewerhdf5hasgroup_       PETSCVIEWERHDF5HASGROUP
  #define petscviewerhdf5opengroup_      PETSCVIEWERHDF5OPENGROUP
  #define petscviewerhdf5writegroup_     PETSCVIEWERHDF5WRITEGROUP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerhdf5open_           petscviewerhdf5open
  #define petscviewerhdf5pushgroup_      petscviewerhdf5pushgroup
  #define petscviewerhdf5getgroup_       petscviewerhdf5getgroup
  #define petscviewerhdf5hasdataset_     petscviewerhdf5hasdataset
  #define petscviewerhdf5hasattribute_   petscviewerhdf5hasattribute
  #define petscviewerhdf5writeattribute_ petscviewerhdf5writeattribute
  #define petscviewerhdf5readattribute_  petscviewerhdf5readattribute
  #define petscviewerhdf5hasgroup_       petscviewerhdf5hasgroup
  #define petscviewerhdf5opengroup_      petscviewerhdf5opengroup
  #define petscviewerhdf5writegroup_     petscviewerhdf5writegroup
#endif

PETSC_EXTERN void petscviewerhdf5open_(MPI_Comm *comm, char *name, PetscFileMode *type, PetscViewer *binv, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(name, len, c1);
  *ierr = PetscViewerHDF5Open(MPI_Comm_f2c(*(MPI_Fint *)&*comm), c1, *type, binv);
  if (*ierr) return;
  FREECHAR(name, c1);
}

PETSC_EXTERN void petscviewerhdf5pushgroup_(PetscViewer *viewer, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(name, len, c1);
  *ierr = PetscViewerHDF5PushGroup(*viewer, c1);
  if (*ierr) return;
  FREECHAR(name, c1);
}

PETSC_EXTERN void petscviewerhdf5getgroup_(PetscViewer *viewer, char *path, char *abspath, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len0, PETSC_FORTRAN_CHARLEN_T len1)
{
  char *c0;
  char *c1;

  FIXCHAR(path, len0, c0);
  *ierr = PetscViewerHDF5GetGroup(*viewer, c0, &c1);
  if (*ierr) return;
  *ierr = PetscStrncpy(abspath, c1, len1);
  if (*ierr) return;
  *ierr = PetscFree(c1);
  if (*ierr) return;
  FREECHAR(path, c0);
  FIXRETURNCHAR(PETSC_TRUE, abspath, len1);
}

PETSC_EXTERN void petscviewerhdf5hasattribute_(PetscViewer *viewer, char *parent, char *name, PetscBool *has, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T plen, PETSC_FORTRAN_CHARLEN_T nlen)
{
  char *c1, *c2;

  FIXCHAR(parent, plen, c1);
  FIXCHAR(name, nlen, c2);
  *ierr = PetscViewerHDF5HasAttribute(*viewer, c1, c2, has);
  if (*ierr) return;
  FREECHAR(parent, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscviewerhdf5writeattribute_(PetscViewer *viewer, char *parent, char *name, PetscDataType *datatype, const void *value, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T plen, PETSC_FORTRAN_CHARLEN_T nlen)
{
  char *c1, *c2;

  FIXCHAR(parent, plen, c1);
  FIXCHAR(name, nlen, c2);
  *ierr = PetscViewerHDF5WriteAttribute(*viewer, c1, c2, *datatype, value);
  if (*ierr) return;
  FREECHAR(parent, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscviewerhdf5readattribute_(PetscViewer *viewer, char *parent, char *name, PetscDataType *datatype, void *defaultValue, void *value, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T plen, PETSC_FORTRAN_CHARLEN_T nlen)
{
  char *c1, *c2;

  FIXCHAR(parent, plen, c1);
  FIXCHAR(name, nlen, c2);
  *ierr = PetscViewerHDF5ReadAttribute(*viewer, c1, c2, *datatype, defaultValue, value);
  if (*ierr) return;
  FREECHAR(parent, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscviewerhdf5hasdataset_(PetscViewer *viewer, char *path, PetscBool *has, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(path, len, c1);
  *ierr = PetscViewerHDF5HasDataset(*viewer, c1, has);
  if (*ierr) return;
  FREECHAR(path, c1);
}

PETSC_EXTERN void petscviewerhdf5hasgroup_(PetscViewer *viewer, char *path, PetscBool *has, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(path, len, c1);
  *ierr = PetscViewerHDF5HasGroup(*viewer, c1, has);
  if (*ierr) return;
  FREECHAR(path, c1);
}

PETSC_EXTERN void petscviewerhdf5opengroup_(PetscViewer *viewer, char path[], hid_t *fileId, hid_t *groupId, int *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(path, len, c1);
  *ierr = PetscViewerHDF5OpenGroup(*viewer, c1, fileId, groupId);
  FREECHAR(path, c1);
}

PETSC_EXTERN void petscviewerhdf5writegroup_(PetscViewer *viewer, char path[], int *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(path, len, c1);
  *ierr = PetscViewerHDF5WriteGroup(*viewer, c1);
  FREECHAR(path, c1);
}
