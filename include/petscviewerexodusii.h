
#if !defined(PETSCVIEWEREXODUSII_H)
#define PETSCVIEWEREXODUSII_H

#include <petscviewer.h>

#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>

PETSC_EXTERN PetscErrorCode PetscViewerExodusIIGetId(PetscViewer,int*);

PETSC_EXTERN PetscErrorCode PetscViewerExodusIIOpen(MPI_Comm,const char[],PetscFileMode,PetscViewer*);
#endif  /* defined(PETSC_HAVE_HDF5) */
#endif
