/* Portions of this code are under:
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#ifndef __MPIHIPSPARSEMATIMPL
#define __MPIHIPSPARSEMATIMPL

#if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  #include <hipsparse/hipsparse.h>
#else
  #include <hipsparse.h>
#endif
#include <petsc/private/hipvecimpl.h>

struct Mat_MPIAIJHIPSPARSE {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatHIPSPARSEStorageFormat  diagGPUMatFormat;
  MatHIPSPARSEStorageFormat  offdiagGPUMatFormat;
  PetscSplitCSRDataStructure deviceMat;
  PetscInt                   coo_nd, coo_no; /* number of nonzero entries in coo for the diag/offdiag part */
  THRUSTINTARRAY            *coo_p;          /* the permutation array that partitions the coo array into diag/offdiag parts */
  THRUSTARRAY               *coo_pw;         /* the work array that stores the partitioned coo scalar values */

  /* Extended COO stuff */
  PetscCount  *Ajmap1_d, *Aperm1_d;            /* Local entries to diag */
  PetscCount  *Bjmap1_d, *Bperm1_d;            /* Local entries to offdiag */
  PetscCount  *Aimap2_d, *Ajmap2_d, *Aperm2_d; /* Remote entries to diag */
  PetscCount  *Bimap2_d, *Bjmap2_d, *Bperm2_d; /* Remote entries to offdiag */
  PetscCount  *Cperm1_d;                       /* Permutation to fill send buffer. 'C' for communication */
  PetscScalar *sendbuf_d, *recvbuf_d;          /* Buffers for remote values in MatSetValuesCOO() */
  PetscBool    use_extended_coo;

  Mat_MPIAIJHIPSPARSE()
  {
    diagGPUMatFormat    = MAT_HIPSPARSE_CSR;
    offdiagGPUMatFormat = MAT_HIPSPARSE_CSR;
    coo_p               = NULL;
    coo_pw              = NULL;
    deviceMat           = NULL;
    use_extended_coo    = PETSC_FALSE;
  }
};

#endif
