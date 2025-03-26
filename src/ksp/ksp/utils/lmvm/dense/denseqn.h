#pragma once

#include <../src/ksp/ksp/utils/lmvm/lmvm.h>
#include <../src/ksp/ksp/utils/lmvm/rescale/symbrdnrescale.h>

/*
  dense representation for the limited-memory BFGS/DFP method.
*/

typedef struct {
  PetscInt          num_updates;
  PetscInt          num_mult_updates;
  Mat               HY, BS; // Stored in recycled order
  Vec               StFprev;
  Mat               StY_triu;        // triu(StY) is the R matrix
  Mat               StY_triu_strict; // strict_triu(YtS) is the R matrix
  Mat               YtS_triu_strict; // strict_triu(YtS) is the L^T matrix
  Mat               YtS_triu;        // triu(YtS) is the L matrix
  Mat               YtHY;
  Mat               StBS;
  Mat               J;
  Mat               temp_mat;
  Vec              *PQ; /* P for BFGS, Q for DFP */
  Vec               diag_vec;
  Vec               diag_vec_recycle_order;
  Vec               inv_diag_vec;
  Vec               column_work, column_work2, rwork1, rwork2, rwork3;
  Vec               rwork2_local, rwork3_local;
  Vec               local_work_vec, local_work_vec_copy;
  Vec               cyclic_work_vec;
  MatType           dense_type;
  MatLMVMDenseType  strategy;
  SymBroydenRescale rescale; /* context for diagonal or scalar rescaling */

  PetscReal       *ytq, *stp, *yts;
  PetscScalar     *workscalar;
  PetscInt         S_count, St_count, Y_count, Yt_count;
  PetscInt         watchdog, max_seq_rejects;        /* tracker to reset after a certain # of consecutive rejects */
  PetscBool        allocated, use_recursive, needPQ; /* P for BFGS, Q for DFP */
  Vec              Fprev_ref;
  PetscObjectState Fprev_state;
} Mat_DQN;

PETSC_INTERN PetscErrorCode MatView_LMVMDDFP(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatView_LMVMDBFGS(Mat, PetscViewer);

PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlace_CUPM(PetscBool, PetscInt, const PetscScalar[], PetscInt, PetscScalar[], PetscInt);
PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlaceCyclic_CUPM(PetscBool, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscInt, PetscScalar[], PetscInt);

PETSC_INTERN PetscErrorCode VecCyclicShift(Mat, Vec, PetscInt, Vec);
PETSC_INTERN PetscErrorCode VecRecycleOrderToHistoryOrder(Mat, Vec, PetscInt, Vec);
PETSC_INTERN PetscErrorCode VecHistoryOrderToRecycleOrder(Mat, Vec, PetscInt, Vec);
PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlace(Mat, Mat, Vec, PetscBool, PetscInt, MatLMVMDenseType);
PETSC_INTERN PetscErrorCode MatMove_LR3(Mat, Mat, PetscInt);
