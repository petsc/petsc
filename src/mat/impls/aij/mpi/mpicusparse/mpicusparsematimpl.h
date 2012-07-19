#ifndef __MPICUSPARSEMATIMPL 
#define __MPICUSPARSEMATIMPL

typedef struct {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatCUSPARSEStorageFormat diagGPUMatFormat;
  MatCUSPARSEStorageFormat offdiagGPUMatFormat;
} Mat_MPIAIJCUSPARSE;

#endif
