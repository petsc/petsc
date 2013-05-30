#ifndef PETSC4PY_COMPAT_H
#define PETSC4PY_COMPAT_H

#include <petsc.h>
#include "compat/mpi.h"

#ifndef PETSC_VERSION_EQ
#define PETSC_VERSION_EQ(MAJOR,MINOR,SUBMINOR) \
  (PETSC_VERSION_(MAJOR,MINOR,SUBMINOR))
#endif
#ifndef PETSC_VERSION_LT
#define PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR) \
  (PETSC_VERSION_MAJOR < (MAJOR) ||            \
   (PETSC_VERSION_MAJOR == (MAJOR) &&          \
    (PETSC_VERSION_MINOR < (MINOR) ||          \
     (PETSC_VERSION_MINOR == (MINOR) &&        \
      (PETSC_VERSION_SUBMINOR < (SUBMINOR))))))
#define PETSC_VERSION_LE(MAJOR,MINOR,SUBMINOR) \
  (PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR) ||   \
   PETSC_VERSION_EQ(MAJOR,MINOR,SUBMINOR))
#endif
#ifndef PETSC_VERSION_GT
#define PETSC_VERSION_GT(MAJOR,MINOR,SUBMINOR) \
  (!PETSC_VERSION_LE(MAJOR,MINOR,SUBMINOR))
#define PETSC_VERSION_GE(MAJOR,MINOR,SUBMINOR) \
  (!PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR))
#endif

#if PETSC_VERSION_(3,3,0)
#include "compat/petsc-33.h"
#endif

#if PETSC_VERSION_(3,2,0)
#include "compat/petsc-32.h"
#endif

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

#endif/*PETSC4PY_COMPAT_H*/
