#ifndef PETSC4PY_COMPAT_HDF5_H
#define PETSC4PY_COMPAT_HDF5_H

#include <petscviewerhdf5.h>
#if !defined(PETSC_HAVE_HDF5)

#define PetscViewerHDF5Error do {               \
    PetscFunctionBegin; \
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires HDF5",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PetscViewerHDF5PushGroup(PETSC_UNUSED PetscViewer vw,PETSC_UNUSED const char g[]){PetscViewerHDF5Error;}
PetscErrorCode PetscViewerHDF5PopGroup(PETSC_UNUSED PetscViewer vw){PetscViewerHDF5Error;}
PetscErrorCode PetscViewerHDF5GetGroup(PETSC_UNUSED PetscViewer vw,PETSC_UNUSED const char *g[]){PetscViewerHDF5Error;}
PetscErrorCode PetscViewerHDF5PushTimestepping(PETSC_UNUSED PetscViewer vw){PetscViewerHDF5Error;}
PetscErrorCode PetscViewerHDF5PopTimestepping(PETSC_UNUSED PetscViewer vw){PetscViewerHDF5Error;}
PetscErrorCode PetscViewerHDF5SetTimestep(PETSC_UNUSED PetscViewer vw, PETSC_UNUSED PetscInt n){PetscViewerHDF5Error;}
PetscErrorCode PetscViewerHDF5GetTimestep(PETSC_UNUSED PetscViewer vw, PETSC_UNUSED PetscInt*n){PetscViewerHDF5Error;}
PetscErrorCode PetscViewerHDF5IncrementTimestep(PETSC_UNUSED PetscViewer vw){PetscViewerHDF5Error;}

#undef PetscViewerHDF5Error

#endif

#endif/*PETSC4PY_COMPAT_HDF5_H*/
