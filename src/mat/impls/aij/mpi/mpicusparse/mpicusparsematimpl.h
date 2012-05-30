#ifndef __MPICUSPARSEMATIMPL 
#define __MPICUSPARSEMATIMPL

typedef struct {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatOption diagGPUMatFormat;
  MatOption offdiagGPUMatFormat;
} Mat_MPIAIJCUSPARSE;

#endif
