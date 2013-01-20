#if !defined(__MPICUSPMATIMPL)
#define __MPICUSPMATIMPL

typedef struct {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
#if defined(PETSC_HAVE_TXPETSCGPU)
  MatCUSPStorageFormat diagGPUMatFormat;
  MatCUSPStorageFormat offdiagGPUMatFormat;
#endif
} Mat_MPIAIJCUSP;

#endif
