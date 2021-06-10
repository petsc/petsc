#if !defined(__MPICUSPARSEMATIMPL)
#define __MPICUSPARSEMATIMPL

#include <cusparse_v2.h>
#include <petsc/private/cudavecimpl.h>

typedef struct {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatCUSPARSEStorageFormat   diagGPUMatFormat;
  MatCUSPARSEStorageFormat   offdiagGPUMatFormat;
  cudaStream_t               stream;
  cusparseHandle_t           handle;
  PetscSplitCSRDataStructure deviceMat;
  PetscInt                   coo_nd,coo_no;
  THRUSTINTARRAY             *coo_p;
  THRUSTARRAY                *coo_pw;
} Mat_MPIAIJCUSPARSE;

PETSC_INTERN PetscErrorCode MatCUSPARSESetStream(Mat, const cudaStream_t stream);
PETSC_INTERN PetscErrorCode MatCUSPARSESetHandle(Mat, const cusparseHandle_t handle);
PETSC_INTERN PetscErrorCode MatCUSPARSEClearHandle(Mat);

#endif
