#if !defined(__MPICUSPARSEMATIMPL)
#define __MPICUSPARSEMATIMPL

#include <cusparse_v2.h>
#include <petsc/private/cudavecimpl.h>

struct Mat_MPIAIJCUSPARSE {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatCUSPARSEStorageFormat   diagGPUMatFormat;
  MatCUSPARSEStorageFormat   offdiagGPUMatFormat;
  cudaStream_t               stream;
  cusparseHandle_t           handle;
  PetscSplitCSRDataStructure deviceMat;
  PetscInt                   coo_nd,coo_no; /* number of nonzero entries in coo for the diag/offdiag part */
  THRUSTINTARRAY             *coo_p; /* the permutation array that partitions the coo array into diag/offdiag parts */
  THRUSTARRAY                *coo_pw; /* the work array that stores the partitioned coo scalar values */

  /* Extended COO stuff */
  PetscCount  *Aimap1_d,*Ajmap1_d,*Aperm1_d; /* Local entries to diag */
  PetscCount  *Bimap1_d,*Bjmap1_d,*Bperm1_d; /* Local entries to offdiag */
  PetscCount  *Aimap2_d,*Ajmap2_d,*Aperm2_d; /* Remote entries to diag */
  PetscCount  *Bimap2_d,*Bjmap2_d,*Bperm2_d; /* Remote entries to offdiag */
  PetscCount  *Cperm1_d; /* Permutation to fill send buffer. 'C' for communication */
  PetscScalar *sendbuf_d,*recvbuf_d; /* Buffers for remote values in MatSetValuesCOO() */
  PetscBool   use_extended_coo;

  Mat_MPIAIJCUSPARSE() {
    diagGPUMatFormat    = MAT_CUSPARSE_CSR;
    offdiagGPUMatFormat = MAT_CUSPARSE_CSR;
    coo_p               = NULL;
    coo_pw              = NULL;
    stream              = 0;
    deviceMat           = NULL;
    use_extended_coo    = PETSC_FALSE;
  }
};

PETSC_INTERN PetscErrorCode MatCUSPARSESetStream(Mat, const cudaStream_t stream);
PETSC_INTERN PetscErrorCode MatCUSPARSESetHandle(Mat, const cusparseHandle_t handle);
PETSC_INTERN PetscErrorCode MatCUSPARSEClearHandle(Mat);

#endif
