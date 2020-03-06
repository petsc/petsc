#include <petsc/private/fortranimpl.h>
#include <petsclayouthdf5.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_HDF5)

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerhdf5readsizes_       PETSCVIEWERHDF5READSIZES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerhdf5readsizes_       petscviewerhdf5readsizes
#endif

PETSC_EXTERN void petscviewerhdf5readsizes_(PetscViewer *viewer, char* name,
    PetscInt *bs, PetscInt *N, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
   char *c1;

   FIXCHAR(name, len, c1);
   *ierr = PetscViewerHDF5ReadSizes(*viewer, c1, bs, N);
   FREECHAR(name, c1);
}

#endif /* defined(PETSC_HAVE_HDF5) */
