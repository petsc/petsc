#ifndef __VIEWERHDF5IMPL_H
#define __VIEWERHDF5IMPL_H

#if defined(H5_VERSION)
#  error "viewerhdf5impl.h must be included *before* any other HDF5 headers"
#else
#  define H5_USE_18_API
#endif
#include <petscviewerhdf5.h>
#include <petsc/private/viewerimpl.h>

#if defined(PETSC_HAVE_HDF5)

/*
  HDF5 function specifications usually read:
  Returns a non-negative value if successful; otherwise returns a negative value.
  (see e.g. https://support.hdfgroup.org/HDF5/doc/RM/RM_H5O.html#Object-Close)
*/
#define PetscStackCallHDF5(func,args) do {                        \
    herr_t _status;                                               \
    PetscStackPush(#func);_status = func args;PetscStackPop; if (_status < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in HDF5 call %s() Status %d",#func,(int)_status); \
  } while (0)

#define PetscStackCallHDF5Return(ret,func,args) do {              \
    PetscStackPush(#func);ret = func args;PetscStackPop; if (ret < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in HDF5 call %s() Status %d",#func,(int)ret); \
  } while (0)

typedef struct PetscViewerHDF5GroupList {
  const char       *name;
  struct PetscViewerHDF5GroupList *next;
} PetscViewerHDF5GroupList;

typedef struct {
  char          *filename;
  PetscFileMode btype;
  hid_t         file_id;
  hid_t         dxpl_id;   /* H5P_DATASET_XFER property list controlling raw data transfer (read/write). Properties are modified using H5Pset_dxpl_* functions. */
  PetscBool     timestepping;    /* Flag to indicate whether objects are stored by tiem index */
  PetscInt      timestep;        /* The time index to look for an object at */
  PetscBool     defTimestepping; /* Default if timestepping attribute is not found. Support for legacy files with no timestepping attribute */
  PetscViewerHDF5GroupList *groups;
  PetscBool     basedimension2;  /* save vectors and DMDA vectors with a dimension of at least 2 even if the bs/dof is 1 */
  PetscBool     spoutput;        /* write data in single precision even if PETSc is compiled with double precision PetscReal */
  PetscBool     horizontal;      /* store column vectors as blocks (needed for MATDENSE I/O) */
} PetscViewer_HDF5;

PETSC_EXTERN PetscErrorCode PetscViewerHDF5CheckTimestepping_Internal(PetscViewer, const char[]); /* currently used in src/dm/impls/da/gr2.c so needs to be extern */
#endif
#endif
