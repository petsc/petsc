#ifndef PETSC4PY_COMPAT_HDF5_H
#define PETSC4PY_COMPAT_HDF5_H

#include <petscviewerhdf5.h>
#if !defined(PETSC_HAVE_HDF5)
#define PetscViewerHDF5Error do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() requires HDF5"); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)
#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5PushGroup"
PetscErrorCode PetscViewerHDF5PushGroup(PETSC_UNUSED PetscViewer vw,PETSC_UNUSED const char g[]){PetscViewerHDF5Error;}
#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5PopGroup"
PetscErrorCode PetscViewerHDF5PopGroup(PETSC_UNUSED PetscViewer vw){PetscViewerHDF5Error;}
#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5GetGroup"
PetscErrorCode PetscViewerHDF5GetGroup(PETSC_UNUSED PetscViewer vw,PETSC_UNUSED const char *g[]){PetscViewerHDF5Error;}
#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5SetTimestep"
PetscErrorCode PetscViewerHDF5SetTimestep(PETSC_UNUSED PetscViewer vw, PETSC_UNUSED PetscInt n){PetscViewerHDF5Error;}
#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5GetTimestep"
PetscErrorCode PetscViewerHDF5GetTimestep(PETSC_UNUSED PetscViewer vw, PETSC_UNUSED PetscInt*n){PetscViewerHDF5Error;}
#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5IncrementTimestep"
PetscErrorCode PetscViewerHDF5IncrementTimestep(PETSC_UNUSED PetscViewer vw){PetscViewerHDF5Error;}
#undef PetscViewerHDF5Error
#endif

#endif/*PETSC4PY_COMPAT_HDF5_H*/
