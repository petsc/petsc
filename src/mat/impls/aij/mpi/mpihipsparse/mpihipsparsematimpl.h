/* Portions of this code are under:
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#pragma once

#include <petscpkg_version.h>
#include <hipsparse/hipsparse.h>
#include <petsc/private/veccupmimpl.h>

struct Mat_MPIAIJHIPSPARSE {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatHIPSPARSEStorageFormat diagGPUMatFormat    = MAT_HIPSPARSE_CSR;
  MatHIPSPARSEStorageFormat offdiagGPUMatFormat = MAT_HIPSPARSE_CSR;
};
