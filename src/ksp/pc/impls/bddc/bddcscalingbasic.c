#include <petsc/private/pcbddcimpl.h>
#include <petsc/private/pcbddcprivateimpl.h>
#include <petscblaslapack.h>
#include <../src/mat/impls/dense/seq/dense.h>

/* prototypes for deluxe functions */
static PetscErrorCode PCBDDCScalingCreate_Deluxe(PC);
static PetscErrorCode PCBDDCScalingDestroy_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Private(PC);
static PetscErrorCode PCBDDCScalingReset_Deluxe_Solvers(PCBDDCDeluxeScaling);

static PetscErrorCode PCBDDCMatTransposeMatSolve_SeqDense(Mat A, Mat B, Mat X)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *)A->data;
  const PetscScalar *b;
  PetscScalar       *x;
  PetscInt           n;
  PetscBLASInt       nrhs, info, m;
  PetscBool          flg;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->rmap->n, &m));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix B must be MATDENSE matrix");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix X must be MATDENSE matrix");

  PetscCall(MatGetSize(B, NULL, &n));
  PetscCall(PetscBLASIntCast(n, &nrhs));
  PetscCall(MatDenseGetArrayRead(B, &b));
  PetscCall(MatDenseGetArray(X, &x));
  PetscCall(PetscArraycpy(x, b, m * nrhs));
  PetscCall(MatDenseRestoreArrayRead(B, &b));

  if (A->factortype == MAT_FACTOR_LU) {
    PetscCallBLAS("LAPACKgetrs", LAPACKgetrs_("T", &m, &nrhs, mat->v, &mat->lda, mat->pivots, x, &m, &info));
    PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "GETRS - Bad solve");
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only LU factor supported");

  PetscCall(MatDenseRestoreArray(X, &x));
  PetscCall(PetscLogFlops(nrhs * (2.0 * m * m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingExtension_Basic(PC pc, Vec local_interface_vector, Vec global_vector)
{
  PC_IS   *pcis   = (PC_IS *)pc->data;
  PC_BDDC *pcbddc = (PC_BDDC *)pc->data;

  PetscFunctionBegin;
  /* Apply partition of unity */
  PetscCall(VecPointwiseMult(pcbddc->work_scaling, pcis->D, local_interface_vector));
  PetscCall(VecSet(global_vector, 0.0));
  PetscCall(VecScatterBegin(pcis->global_to_B, pcbddc->work_scaling, global_vector, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(pcis->global_to_B, pcbddc->work_scaling, global_vector, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingExtension_Deluxe(PC pc, Vec x, Vec y)
{
  PC_IS              *pcis       = (PC_IS *)pc->data;
  PC_BDDC            *pcbddc     = (PC_BDDC *)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;

  PetscFunctionBegin;
  PetscCall(VecSet(pcbddc->work_scaling, 0.0));
  PetscCall(VecSet(y, 0.0));
  if (deluxe_ctx->n_simple) { /* scale deluxe vertices using diagonal scaling */
    PetscInt           i;
    const PetscScalar *array_x, *array_D;
    PetscScalar       *array;
    PetscCall(VecGetArrayRead(x, &array_x));
    PetscCall(VecGetArrayRead(pcis->D, &array_D));
    PetscCall(VecGetArray(pcbddc->work_scaling, &array));
    for (i = 0; i < deluxe_ctx->n_simple; i++) array[deluxe_ctx->idx_simple_B[i]] = array_x[deluxe_ctx->idx_simple_B[i]] * array_D[deluxe_ctx->idx_simple_B[i]];
    PetscCall(VecRestoreArray(pcbddc->work_scaling, &array));
    PetscCall(VecRestoreArrayRead(pcis->D, &array_D));
    PetscCall(VecRestoreArrayRead(x, &array_x));
  }
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication or a matvec and a solve */
  if (deluxe_ctx->seq_mat) {
    PetscInt i;
    for (i = 0; i < deluxe_ctx->seq_n; i++) {
      if (deluxe_ctx->change) {
        PetscCall(VecScatterBegin(deluxe_ctx->seq_scctx[i], x, deluxe_ctx->seq_work2[i], INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(deluxe_ctx->seq_scctx[i], x, deluxe_ctx->seq_work2[i], INSERT_VALUES, SCATTER_FORWARD));
        if (deluxe_ctx->change_with_qr) {
          Mat change;

          PetscCall(KSPGetOperators(deluxe_ctx->change[i], &change, NULL));
          PetscCall(MatMultTranspose(change, deluxe_ctx->seq_work2[i], deluxe_ctx->seq_work1[i]));
        } else {
          PetscCall(KSPSolve(deluxe_ctx->change[i], deluxe_ctx->seq_work2[i], deluxe_ctx->seq_work1[i]));
        }
      } else {
        PetscCall(VecScatterBegin(deluxe_ctx->seq_scctx[i], x, deluxe_ctx->seq_work1[i], INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(deluxe_ctx->seq_scctx[i], x, deluxe_ctx->seq_work1[i], INSERT_VALUES, SCATTER_FORWARD));
      }
      PetscCall(MatMultTranspose(deluxe_ctx->seq_mat[i], deluxe_ctx->seq_work1[i], deluxe_ctx->seq_work2[i]));
      if (deluxe_ctx->seq_mat_inv_sum[i]) {
        PetscScalar *x;

        PetscCall(VecGetArray(deluxe_ctx->seq_work2[i], &x));
        PetscCall(VecPlaceArray(deluxe_ctx->seq_work1[i], x));
        PetscCall(VecRestoreArray(deluxe_ctx->seq_work2[i], &x));
        PetscCall(MatSolveTranspose(deluxe_ctx->seq_mat_inv_sum[i], deluxe_ctx->seq_work1[i], deluxe_ctx->seq_work2[i]));
        PetscCall(VecResetArray(deluxe_ctx->seq_work1[i]));
      }
      if (deluxe_ctx->change) {
        Mat change;

        PetscCall(KSPGetOperators(deluxe_ctx->change[i], &change, NULL));
        PetscCall(MatMult(change, deluxe_ctx->seq_work2[i], deluxe_ctx->seq_work1[i]));
        PetscCall(VecScatterBegin(deluxe_ctx->seq_scctx[i], deluxe_ctx->seq_work1[i], pcbddc->work_scaling, INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(deluxe_ctx->seq_scctx[i], deluxe_ctx->seq_work1[i], pcbddc->work_scaling, INSERT_VALUES, SCATTER_REVERSE));
      } else {
        PetscCall(VecScatterBegin(deluxe_ctx->seq_scctx[i], deluxe_ctx->seq_work2[i], pcbddc->work_scaling, INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(deluxe_ctx->seq_scctx[i], deluxe_ctx->seq_work2[i], pcbddc->work_scaling, INSERT_VALUES, SCATTER_REVERSE));
      }
    }
  }
  /* put local boundary part in global vector */
  PetscCall(VecScatterBegin(pcis->global_to_B, pcbddc->work_scaling, y, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(pcis->global_to_B, pcbddc->work_scaling, y, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScalingExtension(PC pc, Vec local_interface_vector, Vec global_vector)
{
  PC_BDDC *pcbddc = (PC_BDDC *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(local_interface_vector, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(global_vector, VEC_CLASSID, 3);
  PetscCheck(local_interface_vector != pcbddc->work_scaling, PETSC_COMM_SELF, PETSC_ERR_SUP, "Local vector cannot be pcbddc->work_scaling!");
  PetscUseMethod(pc, "PCBDDCScalingExtension_C", (PC, Vec, Vec), (pc, local_interface_vector, global_vector));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingRestriction_Basic(PC pc, Vec global_vector, Vec local_interface_vector)
{
  PC_IS *pcis = (PC_IS *)pc->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(pcis->global_to_B, global_vector, local_interface_vector, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_B, global_vector, local_interface_vector, INSERT_VALUES, SCATTER_FORWARD));
  /* Apply partition of unity */
  PetscCall(VecPointwiseMult(local_interface_vector, pcis->D, local_interface_vector));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingRestriction_Deluxe(PC pc, Vec x, Vec y)
{
  PC_IS              *pcis       = (PC_IS *)pc->data;
  PC_BDDC            *pcbddc     = (PC_BDDC *)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;

  PetscFunctionBegin;
  /* get local boundary part of global vector */
  PetscCall(VecScatterBegin(pcis->global_to_B, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_B, x, y, INSERT_VALUES, SCATTER_FORWARD));
  if (deluxe_ctx->n_simple) { /* scale deluxe vertices using diagonal scaling */
    PetscInt           i;
    PetscScalar       *array_y;
    const PetscScalar *array_D;
    PetscCall(VecGetArray(y, &array_y));
    PetscCall(VecGetArrayRead(pcis->D, &array_D));
    for (i = 0; i < deluxe_ctx->n_simple; i++) array_y[deluxe_ctx->idx_simple_B[i]] *= array_D[deluxe_ctx->idx_simple_B[i]];
    PetscCall(VecRestoreArrayRead(pcis->D, &array_D));
    PetscCall(VecRestoreArray(y, &array_y));
  }
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication or a matvec and a solve */
  if (deluxe_ctx->seq_mat) {
    PetscInt i;
    for (i = 0; i < deluxe_ctx->seq_n; i++) {
      if (deluxe_ctx->change) {
        Mat change;

        PetscCall(VecScatterBegin(deluxe_ctx->seq_scctx[i], y, deluxe_ctx->seq_work2[i], INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(deluxe_ctx->seq_scctx[i], y, deluxe_ctx->seq_work2[i], INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(KSPGetOperators(deluxe_ctx->change[i], &change, NULL));
        PetscCall(MatMultTranspose(change, deluxe_ctx->seq_work2[i], deluxe_ctx->seq_work1[i]));
      } else {
        PetscCall(VecScatterBegin(deluxe_ctx->seq_scctx[i], y, deluxe_ctx->seq_work1[i], INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(deluxe_ctx->seq_scctx[i], y, deluxe_ctx->seq_work1[i], INSERT_VALUES, SCATTER_FORWARD));
      }
      if (deluxe_ctx->seq_mat_inv_sum[i]) {
        PetscScalar *x;

        PetscCall(VecGetArray(deluxe_ctx->seq_work1[i], &x));
        PetscCall(VecPlaceArray(deluxe_ctx->seq_work2[i], x));
        PetscCall(VecRestoreArray(deluxe_ctx->seq_work1[i], &x));
        PetscCall(MatSolve(deluxe_ctx->seq_mat_inv_sum[i], deluxe_ctx->seq_work1[i], deluxe_ctx->seq_work2[i]));
        PetscCall(VecResetArray(deluxe_ctx->seq_work2[i]));
      }
      PetscCall(MatMult(deluxe_ctx->seq_mat[i], deluxe_ctx->seq_work1[i], deluxe_ctx->seq_work2[i]));
      if (deluxe_ctx->change) {
        if (deluxe_ctx->change_with_qr) {
          Mat change;

          PetscCall(KSPGetOperators(deluxe_ctx->change[i], &change, NULL));
          PetscCall(MatMult(change, deluxe_ctx->seq_work2[i], deluxe_ctx->seq_work1[i]));
        } else {
          PetscCall(KSPSolveTranspose(deluxe_ctx->change[i], deluxe_ctx->seq_work2[i], deluxe_ctx->seq_work1[i]));
        }
        PetscCall(VecScatterBegin(deluxe_ctx->seq_scctx[i], deluxe_ctx->seq_work1[i], y, INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(deluxe_ctx->seq_scctx[i], deluxe_ctx->seq_work1[i], y, INSERT_VALUES, SCATTER_REVERSE));
      } else {
        PetscCall(VecScatterBegin(deluxe_ctx->seq_scctx[i], deluxe_ctx->seq_work2[i], y, INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(deluxe_ctx->seq_scctx[i], deluxe_ctx->seq_work2[i], y, INSERT_VALUES, SCATTER_REVERSE));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScalingRestriction(PC pc, Vec global_vector, Vec local_interface_vector)
{
  PC_BDDC *pcbddc = (PC_BDDC *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(global_vector, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(local_interface_vector, VEC_CLASSID, 3);
  PetscCheck(local_interface_vector != pcbddc->work_scaling, PETSC_COMM_SELF, PETSC_ERR_SUP, "Local vector cannot be pcbddc->work_scaling!");
  PetscUseMethod(pc, "PCBDDCScalingRestriction_C", (PC, Vec, Vec), (pc, global_vector, local_interface_vector));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScalingSetUp(PC pc)
{
  PC_IS   *pcis   = (PC_IS *)pc->data;
  PC_BDDC *pcbddc = (PC_BDDC *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCall(PetscLogEventBegin(PC_BDDC_Scaling[pcbddc->current_level], pc, 0, 0, 0));
  /* create work vector for the operator */
  PetscCall(VecDestroy(&pcbddc->work_scaling));
  PetscCall(VecDuplicate(pcis->vec1_B, &pcbddc->work_scaling));
  /* always rebuild pcis->D */
  if (pcis->use_stiffness_scaling) {
    PetscScalar *a;
    PetscInt     i, n;

    PetscCall(MatGetDiagonal(pcbddc->local_mat, pcis->vec1_N));
    PetscCall(VecScatterBegin(pcis->N_to_B, pcis->vec1_N, pcis->D, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcis->N_to_B, pcis->vec1_N, pcis->D, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecAbs(pcis->D));
    PetscCall(VecGetLocalSize(pcis->D, &n));
    PetscCall(VecGetArray(pcis->D, &a));
    for (i = 0; i < n; i++)
      if (PetscAbsScalar(a[i]) < PETSC_SMALL) a[i] = 1.0;
    PetscCall(VecRestoreArray(pcis->D, &a));
  }
  PetscCall(VecSet(pcis->vec1_global, 0.0));
  PetscCall(VecScatterBegin(pcis->global_to_B, pcis->D, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(pcis->global_to_B, pcis->D, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(pcis->global_to_B, pcis->vec1_global, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_B, pcis->vec1_global, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecPointwiseDivide(pcis->D, pcis->D, pcis->vec1_B));
  /* now setup */
  if (pcbddc->use_deluxe_scaling) {
    if (!pcbddc->deluxe_ctx) PetscCall(PCBDDCScalingCreate_Deluxe(pc));
    PetscCall(PCBDDCScalingSetUp_Deluxe(pc));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBDDCScalingRestriction_C", PCBDDCScalingRestriction_Deluxe));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBDDCScalingExtension_C", PCBDDCScalingExtension_Deluxe));
  } else {
    PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBDDCScalingRestriction_C", PCBDDCScalingRestriction_Basic));
    PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBDDCScalingExtension_C", PCBDDCScalingExtension_Basic));
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_Scaling[pcbddc->current_level], pc, 0, 0, 0));

  /* test */
  if (pcbddc->dbg_flag) {
    Mat         B0_B  = NULL;
    Vec         B0_Bv = NULL, B0_Bv2 = NULL;
    Vec         vec2_global;
    PetscViewer viewer = pcbddc->dbg_viewer;
    PetscReal   error;

    /* extension -> from local to parallel */
    PetscCall(VecSet(pcis->vec1_global, 0.0));
    PetscCall(VecSetRandom(pcis->vec1_B, NULL));
    PetscCall(VecScatterBegin(pcis->global_to_B, pcis->vec1_B, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_B, pcis->vec1_B, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecDuplicate(pcis->vec1_global, &vec2_global));
    PetscCall(VecCopy(pcis->vec1_global, vec2_global));
    PetscCall(VecScatterBegin(pcis->global_to_B, pcis->vec1_global, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcis->global_to_B, pcis->vec1_global, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
    if (pcbddc->benign_n) {
      IS is_dummy;

      PetscCall(ISCreateStride(PETSC_COMM_SELF, pcbddc->benign_n, 0, 1, &is_dummy));
      PetscCall(MatCreateSubMatrix(pcbddc->benign_B0, is_dummy, pcis->is_B_local, MAT_INITIAL_MATRIX, &B0_B));
      PetscCall(ISDestroy(&is_dummy));
      PetscCall(MatCreateVecs(B0_B, NULL, &B0_Bv));
      PetscCall(VecDuplicate(B0_Bv, &B0_Bv2));
      PetscCall(MatMult(B0_B, pcis->vec1_B, B0_Bv));
    }
    PetscCall(PCBDDCScalingExtension(pc, pcis->vec1_B, pcis->vec1_global));
    if (pcbddc->benign_saddle_point) {
      PetscReal errorl = 0.;
      PetscCall(VecScatterBegin(pcis->global_to_B, pcis->vec1_global, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcis->global_to_B, pcis->vec1_global, pcis->vec1_B, INSERT_VALUES, SCATTER_FORWARD));
      if (pcbddc->benign_n) {
        PetscCall(MatMult(B0_B, pcis->vec1_B, B0_Bv2));
        PetscCall(VecAXPY(B0_Bv, -1.0, B0_Bv2));
        PetscCall(VecNorm(B0_Bv, NORM_INFINITY, &errorl));
      }
      PetscCallMPI(MPI_Allreduce(&errorl, &error, 1, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)pc)));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Error benign extension %1.14e\n", (double)error));
    }
    PetscCall(VecAXPY(pcis->vec1_global, -1.0, vec2_global));
    PetscCall(VecNorm(pcis->vec1_global, NORM_INFINITY, &error));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Error scaling extension %1.14e\n", (double)error));
    PetscCall(VecDestroy(&vec2_global));

    /* restriction -> from parallel to local */
    PetscCall(VecSet(pcis->vec1_global, 0.0));
    PetscCall(VecSetRandom(pcis->vec1_B, NULL));
    PetscCall(VecScatterBegin(pcis->global_to_B, pcis->vec1_B, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_B, pcis->vec1_B, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(PCBDDCScalingRestriction(pc, pcis->vec1_global, pcis->vec1_B));
    PetscCall(VecScale(pcis->vec1_B, -1.0));
    PetscCall(VecScatterBegin(pcis->global_to_B, pcis->vec1_B, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_B, pcis->vec1_B, pcis->vec1_global, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecNorm(pcis->vec1_global, NORM_INFINITY, &error));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Error scaling restriction %1.14e\n", (double)error));
    PetscCall(MatDestroy(&B0_B));
    PetscCall(VecDestroy(&B0_Bv));
    PetscCall(VecDestroy(&B0_Bv2));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScalingDestroy(PC pc)
{
  PC_BDDC *pcbddc = (PC_BDDC *)pc->data;

  PetscFunctionBegin;
  if (pcbddc->deluxe_ctx) PetscCall(PCBDDCScalingDestroy_Deluxe(pc));
  PetscCall(VecDestroy(&pcbddc->work_scaling));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBDDCScalingRestriction_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBDDCScalingExtension_C", NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingCreate_Deluxe(PC pc)
{
  PC_BDDC            *pcbddc = (PC_BDDC *)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&deluxe_ctx));
  pcbddc->deluxe_ctx = deluxe_ctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingDestroy_Deluxe(PC pc)
{
  PC_BDDC *pcbddc = (PC_BDDC *)pc->data;

  PetscFunctionBegin;
  PetscCall(PCBDDCScalingReset_Deluxe_Solvers(pcbddc->deluxe_ctx));
  PetscCall(PetscFree(pcbddc->deluxe_ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingReset_Deluxe_Solvers(PCBDDCDeluxeScaling deluxe_ctx)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(PetscFree(deluxe_ctx->idx_simple_B));
  deluxe_ctx->n_simple = 0;
  for (i = 0; i < deluxe_ctx->seq_n; i++) {
    PetscCall(VecScatterDestroy(&deluxe_ctx->seq_scctx[i]));
    PetscCall(VecDestroy(&deluxe_ctx->seq_work1[i]));
    PetscCall(VecDestroy(&deluxe_ctx->seq_work2[i]));
    PetscCall(MatDestroy(&deluxe_ctx->seq_mat[i]));
    PetscCall(MatDestroy(&deluxe_ctx->seq_mat_inv_sum[i]));
  }
  PetscCall(PetscFree5(deluxe_ctx->seq_scctx, deluxe_ctx->seq_work1, deluxe_ctx->seq_work2, deluxe_ctx->seq_mat, deluxe_ctx->seq_mat_inv_sum));
  PetscCall(PetscFree(deluxe_ctx->workspace));
  deluxe_ctx->seq_n = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingSetUp_Deluxe(PC pc)
{
  PC_IS              *pcis       = (PC_IS *)pc->data;
  PC_BDDC            *pcbddc     = (PC_BDDC *)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
  PCBDDCSubSchurs     sub_schurs = pcbddc->sub_schurs;

  PetscFunctionBegin;
  /* reset data structures if the topology has changed */
  if (pcbddc->recompute_topography) PetscCall(PCBDDCScalingReset_Deluxe_Solvers(deluxe_ctx));

  /* Compute data structures to solve sequential problems */
  PetscCall(PCBDDCScalingSetUp_Deluxe_Private(pc));

  /* diagonal scaling on interface dofs not contained in cc */
  if (sub_schurs->is_vertices || sub_schurs->is_dir) {
    PetscInt n_com, n_dir;
    n_com = 0;
    if (sub_schurs->is_vertices) PetscCall(ISGetLocalSize(sub_schurs->is_vertices, &n_com));
    n_dir = 0;
    if (sub_schurs->is_dir) PetscCall(ISGetLocalSize(sub_schurs->is_dir, &n_dir));
    if (!deluxe_ctx->n_simple) {
      deluxe_ctx->n_simple = n_dir + n_com;
      PetscCall(PetscMalloc1(deluxe_ctx->n_simple, &deluxe_ctx->idx_simple_B));
      if (sub_schurs->is_vertices) {
        PetscInt        nmap;
        const PetscInt *idxs;

        PetscCall(ISGetIndices(sub_schurs->is_vertices, &idxs));
        PetscCall(ISGlobalToLocalMappingApply(pcis->BtoNmap, IS_GTOLM_DROP, n_com, idxs, &nmap, deluxe_ctx->idx_simple_B));
        PetscCheck(nmap == n_com, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error when mapping simply scaled dofs (is_vertices)! %" PetscInt_FMT " != %" PetscInt_FMT, nmap, n_com);
        PetscCall(ISRestoreIndices(sub_schurs->is_vertices, &idxs));
      }
      if (sub_schurs->is_dir) {
        PetscInt        nmap;
        const PetscInt *idxs;

        PetscCall(ISGetIndices(sub_schurs->is_dir, &idxs));
        PetscCall(ISGlobalToLocalMappingApply(pcis->BtoNmap, IS_GTOLM_DROP, n_dir, idxs, &nmap, deluxe_ctx->idx_simple_B + n_com));
        PetscCheck(nmap == n_dir, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error when mapping simply scaled dofs (sub_schurs->is_dir)! %" PetscInt_FMT " != %" PetscInt_FMT, nmap, n_dir);
        PetscCall(ISRestoreIndices(sub_schurs->is_dir, &idxs));
      }
      PetscCall(PetscSortInt(deluxe_ctx->n_simple, deluxe_ctx->idx_simple_B));
    } else {
      PetscCheck(deluxe_ctx->n_simple == n_dir + n_com, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of simply scaled dofs %" PetscInt_FMT " is different from the previous one computed %" PetscInt_FMT, n_dir + n_com, deluxe_ctx->n_simple);
    }
  } else {
    deluxe_ctx->n_simple     = 0;
    deluxe_ctx->idx_simple_B = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Private(PC pc)
{
  PC_BDDC            *pcbddc     = (PC_BDDC *)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
  PCBDDCSubSchurs     sub_schurs = pcbddc->sub_schurs;
  PetscScalar        *matdata, *matdata2;
  PetscInt            i, max_subset_size, cum, cum2;
  const PetscInt     *idxs;
  PetscBool           newsetup = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheck(sub_schurs, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "Missing PCBDDCSubSchurs");
  if (!sub_schurs->n_subs) PetscFunctionReturn(0);

  /* Allocate arrays for subproblems */
  if (!deluxe_ctx->seq_n) {
    deluxe_ctx->seq_n = sub_schurs->n_subs;
    PetscCall(PetscCalloc5(deluxe_ctx->seq_n, &deluxe_ctx->seq_scctx, deluxe_ctx->seq_n, &deluxe_ctx->seq_work1, deluxe_ctx->seq_n, &deluxe_ctx->seq_work2, deluxe_ctx->seq_n, &deluxe_ctx->seq_mat, deluxe_ctx->seq_n, &deluxe_ctx->seq_mat_inv_sum));
    newsetup = PETSC_TRUE;
  } else PetscCheck(deluxe_ctx->seq_n == sub_schurs->n_subs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of deluxe subproblems %" PetscInt_FMT " is different from the sub_schurs %" PetscInt_FMT, deluxe_ctx->seq_n, sub_schurs->n_subs);

  /* the change of basis is just a reference to sub_schurs->change (if any) */
  deluxe_ctx->change         = sub_schurs->change;
  deluxe_ctx->change_with_qr = sub_schurs->change_with_qr;

  /* Create objects for deluxe */
  max_subset_size = 0;
  for (i = 0; i < sub_schurs->n_subs; i++) {
    PetscInt subset_size;
    PetscCall(ISGetLocalSize(sub_schurs->is_subs[i], &subset_size));
    max_subset_size = PetscMax(subset_size, max_subset_size);
  }
  if (newsetup) PetscCall(PetscMalloc1(2 * max_subset_size, &deluxe_ctx->workspace));
  cum = cum2 = 0;
  PetscCall(ISGetIndices(sub_schurs->is_Ej_all, &idxs));
  PetscCall(MatSeqAIJGetArray(sub_schurs->S_Ej_all, &matdata));
  PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_all, &matdata2));
  for (i = 0; i < deluxe_ctx->seq_n; i++) {
    PetscInt subset_size;

    PetscCall(ISGetLocalSize(sub_schurs->is_subs[i], &subset_size));
    if (newsetup) {
      IS sub;
      /* work vectors */
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, subset_size, deluxe_ctx->workspace, &deluxe_ctx->seq_work1[i]));
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, subset_size, deluxe_ctx->workspace + subset_size, &deluxe_ctx->seq_work2[i]));

      /* scatters */
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, subset_size, idxs + cum, PETSC_COPY_VALUES, &sub));
      PetscCall(VecScatterCreate(pcbddc->work_scaling, sub, deluxe_ctx->seq_work1[i], NULL, &deluxe_ctx->seq_scctx[i]));
      PetscCall(ISDestroy(&sub));
    }

    /* S_E_j */
    PetscCall(MatDestroy(&deluxe_ctx->seq_mat[i]));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, subset_size, subset_size, matdata + cum2, &deluxe_ctx->seq_mat[i]));

    /* \sum_k S^k_E_j */
    PetscCall(MatDestroy(&deluxe_ctx->seq_mat_inv_sum[i]));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, subset_size, subset_size, matdata2 + cum2, &deluxe_ctx->seq_mat_inv_sum[i]));
    PetscCall(MatSetOption(deluxe_ctx->seq_mat_inv_sum[i], MAT_SPD, sub_schurs->is_posdef));
    PetscCall(MatSetOption(deluxe_ctx->seq_mat_inv_sum[i], MAT_HERMITIAN, sub_schurs->is_hermitian));
    if (sub_schurs->is_hermitian) {
      PetscCall(MatCholeskyFactor(deluxe_ctx->seq_mat_inv_sum[i], NULL, NULL));
    } else {
      PetscCall(MatLUFactor(deluxe_ctx->seq_mat_inv_sum[i], NULL, NULL, NULL));
    }
    if (pcbddc->deluxe_singlemat) {
      Mat X, Y;
      if (!sub_schurs->is_hermitian) {
        PetscCall(MatTranspose(deluxe_ctx->seq_mat[i], MAT_INITIAL_MATRIX, &X));
      } else {
        PetscCall(PetscObjectReference((PetscObject)deluxe_ctx->seq_mat[i]));
        X = deluxe_ctx->seq_mat[i];
      }
      PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y));
      if (!sub_schurs->is_hermitian) {
        PetscCall(PCBDDCMatTransposeMatSolve_SeqDense(deluxe_ctx->seq_mat_inv_sum[i], X, Y));
      } else {
        PetscCall(MatMatSolve(deluxe_ctx->seq_mat_inv_sum[i], X, Y));
      }

      PetscCall(MatDestroy(&deluxe_ctx->seq_mat_inv_sum[i]));
      PetscCall(MatDestroy(&deluxe_ctx->seq_mat[i]));
      PetscCall(MatDestroy(&X));
      if (deluxe_ctx->change) {
        Mat C, CY;
        PetscCheck(deluxe_ctx->change_with_qr, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only QR based change of basis");
        PetscCall(KSPGetOperators(deluxe_ctx->change[i], &C, NULL));
        PetscCall(MatMatMult(C, Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CY));
        PetscCall(MatMatTransposeMult(CY, C, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
        PetscCall(MatDestroy(&CY));
        PetscCall(MatProductClear(Y)); /* clear internal matproduct structure of Y since CY is destroyed */
      }
      PetscCall(MatTranspose(Y, MAT_INPLACE_MATRIX, &Y));
      deluxe_ctx->seq_mat[i] = Y;
    }
    cum += subset_size;
    cum2 += subset_size * subset_size;
  }
  PetscCall(ISRestoreIndices(sub_schurs->is_Ej_all, &idxs));
  PetscCall(MatSeqAIJRestoreArray(sub_schurs->S_Ej_all, &matdata));
  PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_all, &matdata2));
  if (pcbddc->deluxe_singlemat) {
    deluxe_ctx->change         = NULL;
    deluxe_ctx->change_with_qr = PETSC_FALSE;
  }

  if (deluxe_ctx->change && !deluxe_ctx->change_with_qr) {
    for (i = 0; i < deluxe_ctx->seq_n; i++) {
      if (newsetup) {
        PC pc;

        PetscCall(KSPGetPC(deluxe_ctx->change[i], &pc));
        PetscCall(PCSetType(pc, PCLU));
        PetscCall(KSPSetFromOptions(deluxe_ctx->change[i]));
      }
      PetscCall(KSPSetUp(deluxe_ctx->change[i]));
    }
  }
  PetscFunctionReturn(0);
}
