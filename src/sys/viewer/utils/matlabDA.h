#include <petsc.h>
#include <petscda.h>
#include <petscbag.h>

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscWriteOutputInitialize(MPI_Comm, const char [], PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscWriteOutputFinalize(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscWriteOutputBag(PetscViewer, const char [], PetscBag);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscWriteOutputVec(PetscViewer, const char [], Vec);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscWriteOutputVecDA(PetscViewer, const char [], Vec, DA);
