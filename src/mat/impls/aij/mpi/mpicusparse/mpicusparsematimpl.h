#pragma once

#include <petsc/private/veccupmimpl.h>

struct Mat_MPIAIJCUSPARSE {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatCUSPARSEStorageFormat diagGPUMatFormat    = MAT_CUSPARSE_CSR;
  MatCUSPARSEStorageFormat offdiagGPUMatFormat = MAT_CUSPARSE_CSR;
};
