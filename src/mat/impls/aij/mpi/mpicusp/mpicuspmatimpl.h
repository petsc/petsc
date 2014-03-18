#if !defined(__MPICUSPMATIMPL)
#define __MPICUSPMATIMPL

typedef struct {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatCUSPStorageFormat diagGPUMatFormat;
  MatCUSPStorageFormat offdiagGPUMatFormat;
  cudaStream_t         stream;
} Mat_MPIAIJCUSP;

PETSC_INTERN PetscErrorCode MatCUSPSetStream(Mat, const cudaStream_t stream);
#endif
