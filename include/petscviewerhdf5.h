
#if !defined(__PETSCVIEWERHDF5_H)
#define __PETSCVIEWERHDF5_H

#include <petscviewer.h>

#if defined(PETSC_HAVE_HDF5)

#include <hdf5.h>
PETSC_EXTERN PetscErrorCode PetscViewerHDF5GetFileId(PetscViewer,hid_t*);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5OpenGroup(PetscViewer, hid_t *, hid_t *);

/* On 32 bit systems HDF5 is limited by size of integer, because hsize_t is defined as size_t */
#define PETSC_HDF5_INT_MAX  2147483647
#define PETSC_HDF5_INT_MIN -2147483647

#undef __FUNCT__
#define __FUNCT__ "PetscHDF5IntCast"
PETSC_STATIC_INLINE PetscErrorCode PetscHDF5IntCast(PetscInt a,hsize_t *b)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES) && (PETSC_SIZEOF_SIZE_T == 4)
  if ((a) > PETSC_HDF5_INT_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Array too long for HDF5");
#endif
  *b =  (hsize_t)(a);
  PetscFunctionReturn(0);
}

#endif  /* defined(PETSC_HAVE_HDF5) */

PETSC_EXTERN PetscErrorCode PetscViewerHDF5WriteSDS(PetscViewer,float *,int,int *,int);

PETSC_EXTERN PetscErrorCode PetscViewerHDF5Open(MPI_Comm,const char[],PetscFileMode,PetscViewer*);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5PushGroup(PetscViewer,const char *);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5PopGroup(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5GetGroup(PetscViewer, const char **);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5IncrementTimestep(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5SetTimestep(PetscViewer,PetscInt);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5GetTimestep(PetscViewer,PetscInt*);

/* Reset __FUNCT__ in case the user does not define it themselves */
#undef __FUNCT__
#define __FUNCT__ "User provided function"

#endif
