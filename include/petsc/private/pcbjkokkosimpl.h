#pragma once

#include <petscvec_kokkos.hpp>
#include <petsc/private/pcimpl.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/kspimpl.h>

#include "Kokkos_Core.hpp"

#if defined(PETSC_HAVE_CUDA)
  #if PETSC_PKG_CUDA_VERSION_GE(10, 0, 0)
    #include <nvtx3/nvToolsExt.h>
  #else
    #include <nvToolsExt.h>
  #endif
#endif

#define PCBJKOKKOS_SHARED_LEVEL 1 // 0 is shared, 1 is global
#define PCBJKOKKOS_VEC_SIZE     16
#define PCBJKOKKOS_TEAM_SIZE    16

#define PCBJKOKKOS_VERBOSE_LEVEL 1

typedef enum {
  BATCH_KSP_BICG_IDX,
  BATCH_KSP_TFQMR_IDX,
  BATCH_KSP_GMRESKK_IDX,
  BATCH_KSP_PREONLY_IDX,
  NUM_BATCH_TYPES
} KSPIndex;

typedef Kokkos::DefaultExecutionSpace exec_space;
using layout           = Kokkos::LayoutRight;
using IntView          = Kokkos::View<PetscInt **, layout, exec_space>;
using AMatrixValueView = const Kokkos::View<PetscScalar **, layout, exec_space>;
using XYType           = const Kokkos::View<PetscScalar **, layout, exec_space>;

typedef struct {
  Vec                                               vec_diag;
  PetscInt                                          nBlocks; /* total number of blocks */
  PetscInt                                          n;       // cache host version of d_bid_eqOffset_k[nBlocks]
  KSP                                               ksp;     // Used just for options. Should have one for each block
  Kokkos::View<PetscInt *, Kokkos::LayoutRight>    *d_bid_eqOffset_k;
  Kokkos::View<PetscScalar *, Kokkos::LayoutRight> *d_idiag_k;
  Kokkos::View<PetscInt *>                         *d_isrow_k;
  Kokkos::View<PetscInt *>                         *d_isicol_k;
  KSPIndex                                          ksp_type_idx;
  PetscInt                                          nwork;
  PetscInt                                          const_block_size; // used to decide to use shared memory for work vectors
  PetscInt                                         *dm_Nf;            // Number of fields in each DM
  PetscInt                                          num_dms;
  // diagnostics
  PetscBool reason;
  PetscBool monitor;
  PetscInt  batch_target;
  PetscInt  rank_target;
  PetscInt  nsolves_team;
  PetscInt  max_nits;
  // caches
  IntView          *rowOffsets;
  IntView          *colIndices;
  XYType           *batch_b;
  XYType           *batch_x;
  AMatrixValueView *batch_values;
} PC_PCBJKOKKOS;

typedef Kokkos::TeamPolicy<>::member_type team_member;
#if defined(PETSC_HAVE_KOKKOS_KERNELS_BATCH)
PETSC_INTERN PetscErrorCode PCApply_BJKOKKOSKERNELS(PC, const PetscScalar *, PetscScalar *, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt, MatInfo, const PetscInt, PCFailedReason *);
#endif
