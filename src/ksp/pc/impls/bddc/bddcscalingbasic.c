#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <petscblaslapack.h>
#include <../src/mat/impls/dense/seq/dense.h>

/* prototypes for deluxe functions */
static PetscErrorCode PCBDDCScalingCreate_Deluxe(PC);
static PetscErrorCode PCBDDCScalingDestroy_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Private(PC);
static PetscErrorCode PCBDDCScalingReset_Deluxe_Solvers(PCBDDCDeluxeScaling);

static PetscErrorCode PCBDDCMatTransposeMatSolve_SeqDense(Mat A,Mat B,Mat X)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *b;
  PetscScalar       *x;
  PetscInt          n;
  PetscBLASInt      nrhs,info,m;
  PetscBool         flg;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(A->rmap->n,&m));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)B,&flg,MATSEQDENSE,MATMPIDENSE,NULL));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");

  CHKERRQ(MatGetSize(B,NULL,&n));
  CHKERRQ(PetscBLASIntCast(n,&nrhs));
  CHKERRQ(MatDenseGetArrayRead(B,&b));
  CHKERRQ(MatDenseGetArray(X,&x));
  CHKERRQ(PetscArraycpy(x,b,m*nrhs));
  CHKERRQ(MatDenseRestoreArrayRead(B,&b));

  if (A->factortype == MAT_FACTOR_LU) {
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("T",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
    PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only LU factor supported");

  CHKERRQ(MatDenseRestoreArray(X,&x));
  CHKERRQ(PetscLogFlops(nrhs*(2.0*m*m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingExtension_Basic(PC pc, Vec local_interface_vector, Vec global_vector)
{
  PC_IS*         pcis = (PC_IS*)pc->data;
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  /* Apply partition of unity */
  CHKERRQ(VecPointwiseMult(pcbddc->work_scaling,pcis->D,local_interface_vector));
  CHKERRQ(VecSet(global_vector,0.0));
  CHKERRQ(VecScatterBegin(pcis->global_to_B,pcbddc->work_scaling,global_vector,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,pcbddc->work_scaling,global_vector,ADD_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingExtension_Deluxe(PC pc, Vec x, Vec y)
{
  PC_IS*              pcis=(PC_IS*)pc->data;
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;

  PetscFunctionBegin;
  CHKERRQ(VecSet(pcbddc->work_scaling,0.0));
  CHKERRQ(VecSet(y,0.0));
  if (deluxe_ctx->n_simple) { /* scale deluxe vertices using diagonal scaling */
    PetscInt          i;
    const PetscScalar *array_x,*array_D;
    PetscScalar       *array;
    CHKERRQ(VecGetArrayRead(x,&array_x));
    CHKERRQ(VecGetArrayRead(pcis->D,&array_D));
    CHKERRQ(VecGetArray(pcbddc->work_scaling,&array));
    for (i=0;i<deluxe_ctx->n_simple;i++) {
      array[deluxe_ctx->idx_simple_B[i]] = array_x[deluxe_ctx->idx_simple_B[i]]*array_D[deluxe_ctx->idx_simple_B[i]];
    }
    CHKERRQ(VecRestoreArray(pcbddc->work_scaling,&array));
    CHKERRQ(VecRestoreArrayRead(pcis->D,&array_D));
    CHKERRQ(VecRestoreArrayRead(x,&array_x));
  }
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication or a matvec and a solve */
  if (deluxe_ctx->seq_mat) {
    PetscInt i;
    for (i=0;i<deluxe_ctx->seq_n;i++) {
      if (deluxe_ctx->change) {
        CHKERRQ(VecScatterBegin(deluxe_ctx->seq_scctx[i],x,deluxe_ctx->seq_work2[i],INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(deluxe_ctx->seq_scctx[i],x,deluxe_ctx->seq_work2[i],INSERT_VALUES,SCATTER_FORWARD));
        if (deluxe_ctx->change_with_qr) {
          Mat change;

          CHKERRQ(KSPGetOperators(deluxe_ctx->change[i],&change,NULL));
          CHKERRQ(MatMultTranspose(change,deluxe_ctx->seq_work2[i],deluxe_ctx->seq_work1[i]));
        } else {
          CHKERRQ(KSPSolve(deluxe_ctx->change[i],deluxe_ctx->seq_work2[i],deluxe_ctx->seq_work1[i]));
        }
      } else {
        CHKERRQ(VecScatterBegin(deluxe_ctx->seq_scctx[i],x,deluxe_ctx->seq_work1[i],INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(deluxe_ctx->seq_scctx[i],x,deluxe_ctx->seq_work1[i],INSERT_VALUES,SCATTER_FORWARD));
      }
      CHKERRQ(MatMultTranspose(deluxe_ctx->seq_mat[i],deluxe_ctx->seq_work1[i],deluxe_ctx->seq_work2[i]));
      if (deluxe_ctx->seq_mat_inv_sum[i]) {
        PetscScalar *x;

        CHKERRQ(VecGetArray(deluxe_ctx->seq_work2[i],&x));
        CHKERRQ(VecPlaceArray(deluxe_ctx->seq_work1[i],x));
        CHKERRQ(VecRestoreArray(deluxe_ctx->seq_work2[i],&x));
        CHKERRQ(MatSolveTranspose(deluxe_ctx->seq_mat_inv_sum[i],deluxe_ctx->seq_work1[i],deluxe_ctx->seq_work2[i]));
        CHKERRQ(VecResetArray(deluxe_ctx->seq_work1[i]));
      }
      if (deluxe_ctx->change) {
        Mat change;

        CHKERRQ(KSPGetOperators(deluxe_ctx->change[i],&change,NULL));
        CHKERRQ(MatMult(change,deluxe_ctx->seq_work2[i],deluxe_ctx->seq_work1[i]));
        CHKERRQ(VecScatterBegin(deluxe_ctx->seq_scctx[i],deluxe_ctx->seq_work1[i],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE));
        CHKERRQ(VecScatterEnd(deluxe_ctx->seq_scctx[i],deluxe_ctx->seq_work1[i],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE));
      } else {
        CHKERRQ(VecScatterBegin(deluxe_ctx->seq_scctx[i],deluxe_ctx->seq_work2[i],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE));
        CHKERRQ(VecScatterEnd(deluxe_ctx->seq_scctx[i],deluxe_ctx->seq_work2[i],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE));
      }
    }
  }
  /* put local boundary part in global vector */
  CHKERRQ(VecScatterBegin(pcis->global_to_B,pcbddc->work_scaling,y,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,pcbddc->work_scaling,y,ADD_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScalingExtension(PC pc, Vec local_interface_vector, Vec global_vector)
{
  PC_BDDC        *pcbddc=(PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(local_interface_vector,VEC_CLASSID,2);
  PetscValidHeaderSpecific(global_vector,VEC_CLASSID,3);
  PetscCheckFalse(local_interface_vector == pcbddc->work_scaling,PETSC_COMM_SELF,PETSC_ERR_SUP,"Local vector cannot be pcbddc->work_scaling!");
  CHKERRQ(PetscUseMethod(pc,"PCBDDCScalingExtension_C",(PC,Vec,Vec),(pc,local_interface_vector,global_vector)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingRestriction_Basic(PC pc, Vec global_vector, Vec local_interface_vector)
{
  PC_IS          *pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(pcis->global_to_B,global_vector,local_interface_vector,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,global_vector,local_interface_vector,INSERT_VALUES,SCATTER_FORWARD));
  /* Apply partition of unity */
  CHKERRQ(VecPointwiseMult(local_interface_vector,pcis->D,local_interface_vector));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingRestriction_Deluxe(PC pc, Vec x, Vec y)
{
  PC_IS*              pcis=(PC_IS*)pc->data;
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;

  PetscFunctionBegin;
  /* get local boundary part of global vector */
  CHKERRQ(VecScatterBegin(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD));
  if (deluxe_ctx->n_simple) { /* scale deluxe vertices using diagonal scaling */
    PetscInt          i;
    PetscScalar       *array_y;
    const PetscScalar *array_D;
    CHKERRQ(VecGetArray(y,&array_y));
    CHKERRQ(VecGetArrayRead(pcis->D,&array_D));
    for (i=0;i<deluxe_ctx->n_simple;i++) {
      array_y[deluxe_ctx->idx_simple_B[i]] *= array_D[deluxe_ctx->idx_simple_B[i]];
    }
    CHKERRQ(VecRestoreArrayRead(pcis->D,&array_D));
    CHKERRQ(VecRestoreArray(y,&array_y));
  }
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication or a matvec and a solve */
  if (deluxe_ctx->seq_mat) {
    PetscInt i;
    for (i=0;i<deluxe_ctx->seq_n;i++) {
      if (deluxe_ctx->change) {
        Mat change;

        CHKERRQ(VecScatterBegin(deluxe_ctx->seq_scctx[i],y,deluxe_ctx->seq_work2[i],INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(deluxe_ctx->seq_scctx[i],y,deluxe_ctx->seq_work2[i],INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(KSPGetOperators(deluxe_ctx->change[i],&change,NULL));
        CHKERRQ(MatMultTranspose(change,deluxe_ctx->seq_work2[i],deluxe_ctx->seq_work1[i]));
      } else {
        CHKERRQ(VecScatterBegin(deluxe_ctx->seq_scctx[i],y,deluxe_ctx->seq_work1[i],INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(deluxe_ctx->seq_scctx[i],y,deluxe_ctx->seq_work1[i],INSERT_VALUES,SCATTER_FORWARD));
      }
      if (deluxe_ctx->seq_mat_inv_sum[i]) {
        PetscScalar *x;

        CHKERRQ(VecGetArray(deluxe_ctx->seq_work1[i],&x));
        CHKERRQ(VecPlaceArray(deluxe_ctx->seq_work2[i],x));
        CHKERRQ(VecRestoreArray(deluxe_ctx->seq_work1[i],&x));
        CHKERRQ(MatSolve(deluxe_ctx->seq_mat_inv_sum[i],deluxe_ctx->seq_work1[i],deluxe_ctx->seq_work2[i]));
        CHKERRQ(VecResetArray(deluxe_ctx->seq_work2[i]));
      }
      CHKERRQ(MatMult(deluxe_ctx->seq_mat[i],deluxe_ctx->seq_work1[i],deluxe_ctx->seq_work2[i]));
      if (deluxe_ctx->change) {
        if (deluxe_ctx->change_with_qr) {
          Mat change;

          CHKERRQ(KSPGetOperators(deluxe_ctx->change[i],&change,NULL));
          CHKERRQ(MatMult(change,deluxe_ctx->seq_work2[i],deluxe_ctx->seq_work1[i]));
        } else {
          CHKERRQ(KSPSolveTranspose(deluxe_ctx->change[i],deluxe_ctx->seq_work2[i],deluxe_ctx->seq_work1[i]));
        }
        CHKERRQ(VecScatterBegin(deluxe_ctx->seq_scctx[i],deluxe_ctx->seq_work1[i],y,INSERT_VALUES,SCATTER_REVERSE));
        CHKERRQ(VecScatterEnd(deluxe_ctx->seq_scctx[i],deluxe_ctx->seq_work1[i],y,INSERT_VALUES,SCATTER_REVERSE));
      } else {
        CHKERRQ(VecScatterBegin(deluxe_ctx->seq_scctx[i],deluxe_ctx->seq_work2[i],y,INSERT_VALUES,SCATTER_REVERSE));
        CHKERRQ(VecScatterEnd(deluxe_ctx->seq_scctx[i],deluxe_ctx->seq_work2[i],y,INSERT_VALUES,SCATTER_REVERSE));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScalingRestriction(PC pc, Vec global_vector, Vec local_interface_vector)
{
  PC_BDDC        *pcbddc=(PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(global_vector,VEC_CLASSID,2);
  PetscValidHeaderSpecific(local_interface_vector,VEC_CLASSID,3);
  PetscCheckFalse(local_interface_vector == pcbddc->work_scaling,PETSC_COMM_SELF,PETSC_ERR_SUP,"Local vector cannot be pcbddc->work_scaling!");
  CHKERRQ(PetscUseMethod(pc,"PCBDDCScalingRestriction_C",(PC,Vec,Vec),(pc,global_vector,local_interface_vector)));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScalingSetUp(PC pc)
{
  PC_IS*         pcis=(PC_IS*)pc->data;
  PC_BDDC*       pcbddc=(PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscLogEventBegin(PC_BDDC_Scaling[pcbddc->current_level],pc,0,0,0));
  /* create work vector for the operator */
  CHKERRQ(VecDestroy(&pcbddc->work_scaling));
  CHKERRQ(VecDuplicate(pcis->vec1_B,&pcbddc->work_scaling));
  /* always rebuild pcis->D */
  if (pcis->use_stiffness_scaling) {
    PetscScalar *a;
    PetscInt    i,n;

    CHKERRQ(MatGetDiagonal(pcbddc->local_mat,pcis->vec1_N));
    CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecAbs(pcis->D));
    CHKERRQ(VecGetLocalSize(pcis->D,&n));
    CHKERRQ(VecGetArray(pcis->D,&a));
    for (i=0;i<n;i++) if (PetscAbsScalar(a[i])<PETSC_SMALL) a[i] = 1.0;
    CHKERRQ(VecRestoreArray(pcis->D,&a));
  }
  CHKERRQ(VecSet(pcis->vec1_global,0.0));
  CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->D,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->D,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecPointwiseDivide(pcis->D,pcis->D,pcis->vec1_B));
  /* now setup */
  if (pcbddc->use_deluxe_scaling) {
    if (!pcbddc->deluxe_ctx) {
      CHKERRQ(PCBDDCScalingCreate_Deluxe(pc));
    }
    CHKERRQ(PCBDDCScalingSetUp_Deluxe(pc));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",PCBDDCScalingRestriction_Deluxe));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",PCBDDCScalingExtension_Deluxe));
  } else {
    CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",PCBDDCScalingRestriction_Basic));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",PCBDDCScalingExtension_Basic));
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_Scaling[pcbddc->current_level],pc,0,0,0));

  /* test */
  if (pcbddc->dbg_flag) {
    Mat         B0_B = NULL;
    Vec         B0_Bv = NULL, B0_Bv2 = NULL;
    Vec         vec2_global;
    PetscViewer viewer = pcbddc->dbg_viewer;
    PetscReal   error;

    /* extension -> from local to parallel */
    CHKERRQ(VecSet(pcis->vec1_global,0.0));
    CHKERRQ(VecSetRandom(pcis->vec1_B,NULL));
    CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecDuplicate(pcis->vec1_global,&vec2_global));
    CHKERRQ(VecCopy(pcis->vec1_global,vec2_global));
    CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
    if (pcbddc->benign_n) {
      IS is_dummy;

      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&is_dummy));
      CHKERRQ(MatCreateSubMatrix(pcbddc->benign_B0,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B));
      CHKERRQ(ISDestroy(&is_dummy));
      CHKERRQ(MatCreateVecs(B0_B,NULL,&B0_Bv));
      CHKERRQ(VecDuplicate(B0_Bv,&B0_Bv2));
      CHKERRQ(MatMult(B0_B,pcis->vec1_B,B0_Bv));
    }
    CHKERRQ(PCBDDCScalingExtension(pc,pcis->vec1_B,pcis->vec1_global));
    if (pcbddc->benign_saddle_point) {
      PetscReal errorl = 0.;
      CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      if (pcbddc->benign_n) {
        CHKERRQ(MatMult(B0_B,pcis->vec1_B,B0_Bv2));
        CHKERRQ(VecAXPY(B0_Bv,-1.0,B0_Bv2));
        CHKERRQ(VecNorm(B0_Bv,NORM_INFINITY,&errorl));
      }
      CHKERRMPI(MPI_Allreduce(&errorl,&error,1,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)pc)));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"Error benign extension %1.14e\n",error));
    }
    CHKERRQ(VecAXPY(pcis->vec1_global,-1.0,vec2_global));
    CHKERRQ(VecNorm(pcis->vec1_global,NORM_INFINITY,&error));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Error scaling extension %1.14e\n",error));
    CHKERRQ(VecDestroy(&vec2_global));

    /* restriction -> from parallel to local */
    CHKERRQ(VecSet(pcis->vec1_global,0.0));
    CHKERRQ(VecSetRandom(pcis->vec1_B,NULL));
    CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(PCBDDCScalingRestriction(pc,pcis->vec1_global,pcis->vec1_B));
    CHKERRQ(VecScale(pcis->vec1_B,-1.0));
    CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecNorm(pcis->vec1_global,NORM_INFINITY,&error));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Error scaling restriction %1.14e\n",error));
    CHKERRQ(MatDestroy(&B0_B));
    CHKERRQ(VecDestroy(&B0_Bv));
    CHKERRQ(VecDestroy(&B0_Bv2));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScalingDestroy(PC pc)
{
  PC_BDDC*       pcbddc=(PC_BDDC*)pc->data;

  PetscFunctionBegin;
  if (pcbddc->deluxe_ctx) {
    CHKERRQ(PCBDDCScalingDestroy_Deluxe(pc));
  }
  CHKERRQ(VecDestroy(&pcbddc->work_scaling));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingCreate_Deluxe(PC pc)
{
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&deluxe_ctx));
  pcbddc->deluxe_ctx = deluxe_ctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingDestroy_Deluxe(PC pc)
{
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PCBDDCScalingReset_Deluxe_Solvers(pcbddc->deluxe_ctx));
  CHKERRQ(PetscFree(pcbddc->deluxe_ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingReset_Deluxe_Solvers(PCBDDCDeluxeScaling deluxe_ctx)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(deluxe_ctx->idx_simple_B));
  deluxe_ctx->n_simple = 0;
  for (i=0;i<deluxe_ctx->seq_n;i++) {
    CHKERRQ(VecScatterDestroy(&deluxe_ctx->seq_scctx[i]));
    CHKERRQ(VecDestroy(&deluxe_ctx->seq_work1[i]));
    CHKERRQ(VecDestroy(&deluxe_ctx->seq_work2[i]));
    CHKERRQ(MatDestroy(&deluxe_ctx->seq_mat[i]));
    CHKERRQ(MatDestroy(&deluxe_ctx->seq_mat_inv_sum[i]));
  }
  CHKERRQ(PetscFree5(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,deluxe_ctx->seq_work2,deluxe_ctx->seq_mat,deluxe_ctx->seq_mat_inv_sum));
  CHKERRQ(PetscFree(deluxe_ctx->workspace));
  deluxe_ctx->seq_n = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingSetUp_Deluxe(PC pc)
{
  PC_IS               *pcis=(PC_IS*)pc->data;
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx=pcbddc->deluxe_ctx;
  PCBDDCSubSchurs     sub_schurs=pcbddc->sub_schurs;

  PetscFunctionBegin;
  /* reset data structures if the topology has changed */
  if (pcbddc->recompute_topography) {
    CHKERRQ(PCBDDCScalingReset_Deluxe_Solvers(deluxe_ctx));
  }

  /* Compute data structures to solve sequential problems */
  CHKERRQ(PCBDDCScalingSetUp_Deluxe_Private(pc));

  /* diagonal scaling on interface dofs not contained in cc */
  if (sub_schurs->is_vertices || sub_schurs->is_dir) {
    PetscInt n_com,n_dir;
    n_com = 0;
    if (sub_schurs->is_vertices) {
      CHKERRQ(ISGetLocalSize(sub_schurs->is_vertices,&n_com));
    }
    n_dir = 0;
    if (sub_schurs->is_dir) {
      CHKERRQ(ISGetLocalSize(sub_schurs->is_dir,&n_dir));
    }
    if (!deluxe_ctx->n_simple) {
      deluxe_ctx->n_simple = n_dir + n_com;
      CHKERRQ(PetscMalloc1(deluxe_ctx->n_simple,&deluxe_ctx->idx_simple_B));
      if (sub_schurs->is_vertices) {
        PetscInt       nmap;
        const PetscInt *idxs;

        CHKERRQ(ISGetIndices(sub_schurs->is_vertices,&idxs));
        CHKERRQ(ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,n_com,idxs,&nmap,deluxe_ctx->idx_simple_B));
        PetscCheckFalse(nmap != n_com,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error when mapping simply scaled dofs (is_vertices)! %D != %D",nmap,n_com);
        CHKERRQ(ISRestoreIndices(sub_schurs->is_vertices,&idxs));
      }
      if (sub_schurs->is_dir) {
        PetscInt       nmap;
        const PetscInt *idxs;

        CHKERRQ(ISGetIndices(sub_schurs->is_dir,&idxs));
        CHKERRQ(ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,n_dir,idxs,&nmap,deluxe_ctx->idx_simple_B+n_com));
        PetscCheckFalse(nmap != n_dir,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error when mapping simply scaled dofs (sub_schurs->is_dir)! %D != %D",nmap,n_dir);
        CHKERRQ(ISRestoreIndices(sub_schurs->is_dir,&idxs));
      }
      CHKERRQ(PetscSortInt(deluxe_ctx->n_simple,deluxe_ctx->idx_simple_B));
    } else {
      PetscCheckFalse(deluxe_ctx->n_simple != n_dir + n_com,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of simply scaled dofs %D is different from the previous one computed %D",n_dir + n_com,deluxe_ctx->n_simple);
    }
  } else {
    deluxe_ctx->n_simple = 0;
    deluxe_ctx->idx_simple_B = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Private(PC pc)
{
  PC_BDDC                *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling    deluxe_ctx=pcbddc->deluxe_ctx;
  PCBDDCSubSchurs        sub_schurs = pcbddc->sub_schurs;
  PetscScalar            *matdata,*matdata2;
  PetscInt               i,max_subset_size,cum,cum2;
  const PetscInt         *idxs;
  PetscBool              newsetup = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheck(sub_schurs,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Missing PCBDDCSubSchurs");
  if (!sub_schurs->n_subs) PetscFunctionReturn(0);

  /* Allocate arrays for subproblems */
  if (!deluxe_ctx->seq_n) {
    deluxe_ctx->seq_n = sub_schurs->n_subs;
    CHKERRQ(PetscCalloc5(deluxe_ctx->seq_n,&deluxe_ctx->seq_scctx,deluxe_ctx->seq_n,&deluxe_ctx->seq_work1,deluxe_ctx->seq_n,&deluxe_ctx->seq_work2,deluxe_ctx->seq_n,&deluxe_ctx->seq_mat,deluxe_ctx->seq_n,&deluxe_ctx->seq_mat_inv_sum));
    newsetup = PETSC_TRUE;
  } else PetscCheckFalse(deluxe_ctx->seq_n != sub_schurs->n_subs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of deluxe subproblems %D is different from the sub_schurs %D",deluxe_ctx->seq_n,sub_schurs->n_subs);

  /* the change of basis is just a reference to sub_schurs->change (if any) */
  deluxe_ctx->change         = sub_schurs->change;
  deluxe_ctx->change_with_qr = sub_schurs->change_with_qr;

  /* Create objects for deluxe */
  max_subset_size = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscInt subset_size;
    CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    max_subset_size = PetscMax(subset_size,max_subset_size);
  }
  if (newsetup) {
    CHKERRQ(PetscMalloc1(2*max_subset_size,&deluxe_ctx->workspace));
  }
  cum = cum2 = 0;
  CHKERRQ(ISGetIndices(sub_schurs->is_Ej_all,&idxs));
  CHKERRQ(MatSeqAIJGetArray(sub_schurs->S_Ej_all,&matdata));
  CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_all,&matdata2));
  for (i=0;i<deluxe_ctx->seq_n;i++) {
    PetscInt     subset_size;

    CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    if (newsetup) {
      IS  sub;
      /* work vectors */
      CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,subset_size,deluxe_ctx->workspace,&deluxe_ctx->seq_work1[i]));
      CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,subset_size,deluxe_ctx->workspace+subset_size,&deluxe_ctx->seq_work2[i]));

      /* scatters */
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,subset_size,idxs+cum,PETSC_COPY_VALUES,&sub));
      CHKERRQ(VecScatterCreate(pcbddc->work_scaling,sub,deluxe_ctx->seq_work1[i],NULL,&deluxe_ctx->seq_scctx[i]));
      CHKERRQ(ISDestroy(&sub));
    }

    /* S_E_j */
    CHKERRQ(MatDestroy(&deluxe_ctx->seq_mat[i]));
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,matdata+cum2,&deluxe_ctx->seq_mat[i]));

    /* \sum_k S^k_E_j */
    CHKERRQ(MatDestroy(&deluxe_ctx->seq_mat_inv_sum[i]));
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,matdata2+cum2,&deluxe_ctx->seq_mat_inv_sum[i]));
    CHKERRQ(MatSetOption(deluxe_ctx->seq_mat_inv_sum[i],MAT_SPD,sub_schurs->is_posdef));
    CHKERRQ(MatSetOption(deluxe_ctx->seq_mat_inv_sum[i],MAT_HERMITIAN,sub_schurs->is_hermitian));
    if (sub_schurs->is_hermitian) {
      CHKERRQ(MatCholeskyFactor(deluxe_ctx->seq_mat_inv_sum[i],NULL,NULL));
    } else {
      CHKERRQ(MatLUFactor(deluxe_ctx->seq_mat_inv_sum[i],NULL,NULL,NULL));
    }
    if (pcbddc->deluxe_singlemat) {
      Mat X,Y;
      if (!sub_schurs->is_hermitian) {
        CHKERRQ(MatTranspose(deluxe_ctx->seq_mat[i],MAT_INITIAL_MATRIX,&X));
      } else {
        CHKERRQ(PetscObjectReference((PetscObject)deluxe_ctx->seq_mat[i]));
        X    = deluxe_ctx->seq_mat[i];
      }
      CHKERRQ(MatDuplicate(X,MAT_DO_NOT_COPY_VALUES,&Y));
      if (!sub_schurs->is_hermitian) {
        CHKERRQ(PCBDDCMatTransposeMatSolve_SeqDense(deluxe_ctx->seq_mat_inv_sum[i],X,Y));
      } else {
        CHKERRQ(MatMatSolve(deluxe_ctx->seq_mat_inv_sum[i],X,Y));
      }

      CHKERRQ(MatDestroy(&deluxe_ctx->seq_mat_inv_sum[i]));
      CHKERRQ(MatDestroy(&deluxe_ctx->seq_mat[i]));
      CHKERRQ(MatDestroy(&X));
      if (deluxe_ctx->change) {
        Mat C,CY;
        PetscCheck(deluxe_ctx->change_with_qr,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only QR based change of basis");
        CHKERRQ(KSPGetOperators(deluxe_ctx->change[i],&C,NULL));
        CHKERRQ(MatMatMult(C,Y,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CY));
        CHKERRQ(MatMatTransposeMult(CY,C,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y));
        CHKERRQ(MatDestroy(&CY));
        CHKERRQ(MatProductClear(Y)); /* clear internal matproduct structure of Y since CY is destroyed */
      }
      CHKERRQ(MatTranspose(Y,MAT_INPLACE_MATRIX,&Y));
      deluxe_ctx->seq_mat[i] = Y;
    }
    cum += subset_size;
    cum2 += subset_size*subset_size;
  }
  CHKERRQ(ISRestoreIndices(sub_schurs->is_Ej_all,&idxs));
  CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->S_Ej_all,&matdata));
  CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_all,&matdata2));
  if (pcbddc->deluxe_singlemat) {
    deluxe_ctx->change         = NULL;
    deluxe_ctx->change_with_qr = PETSC_FALSE;
  }

  if (deluxe_ctx->change && !deluxe_ctx->change_with_qr) {
    for (i=0;i<deluxe_ctx->seq_n;i++) {
      if (newsetup) {
        PC pc;

        CHKERRQ(KSPGetPC(deluxe_ctx->change[i],&pc));
        CHKERRQ(PCSetType(pc,PCLU));
        CHKERRQ(KSPSetFromOptions(deluxe_ctx->change[i]));
      }
      CHKERRQ(KSPSetUp(deluxe_ctx->change[i]));
    }
  }
  PetscFunctionReturn(0);
}
