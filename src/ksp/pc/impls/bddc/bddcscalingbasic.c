#include "bddc.h"
#include "bddcprivate.h"

/* prototypes for deluxe public functions */
#if 0
extern PetscErrorCode PCBDDCScalingCreateDeluxe(PC);
extern PetscErrorCode PCBDDCScalingDestroyDeluxe(PC);
extern PetscErrorCode PCBDDCScalingSetUpDeluxe(PC);
#endif

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingExtension_Basic"
static PetscErrorCode PCBDDCScalingExtension_Basic(PC pc, Vec local_interface_vector, Vec global_vector)
{
  PC_IS* pcis = (PC_IS*)pc->data;
  PC_BDDC* pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Apply partition of unity */
  ierr = VecPointwiseMult(pcbddc->work_scaling,pcis->D,local_interface_vector);CHKERRQ(ierr);
  ierr = VecSet(global_vector,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcbddc->work_scaling,global_vector,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,pcbddc->work_scaling,global_vector,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingExtension_Deluxe"
static PetscErrorCode PCBDDCScalingExtension_Deluxe(PC pc, Vec x, Vec y)
{
  PC_IS*              pcis=(PC_IS*)pc->data;
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
  PetscScalar         *array_x,*array_D,*array;
  PetscScalar         zero=0.0;
  PetscInt            i;
  PetscMPIInt         color_rank;
  PetscErrorCode      ierr;

  /* TODO CHECK STUFF RELATED WITH FAKE WORK */
  PetscFunctionBegin;
  ierr = VecSet(pcbddc->work_scaling,zero);CHKERRQ(ierr); /* needed by the fake work below */
  /* scale deluxe vertices using diagonal scaling */
  ierr = VecGetArray(x,&array_x);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->D,&array_D);CHKERRQ(ierr);
  ierr = VecGetArray(pcbddc->work_scaling,&array);CHKERRQ(ierr);
  for (i=0;i<deluxe_ctx->n_simple;i++) {
    array[deluxe_ctx->idx_simple_B[i]] = array_x[deluxe_ctx->idx_simple_B[i]]*array_D[deluxe_ctx->idx_simple_B[i]];
  }
  ierr = VecRestoreArray(pcbddc->work_scaling,&array);CHKERRQ(ierr);
  ierr = VecRestoreArray(pcis->D,&array_D);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&array_x);CHKERRQ(ierr);
  /* sequential part : all problems and Schur applications collapsed into seq_mat at setup phase */
  if (deluxe_ctx->seq_mat) {
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,x,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,x,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatMult(deluxe_ctx->seq_mat,deluxe_ctx->seq_work1,deluxe_ctx->seq_work2);CHKERRQ(ierr);
    ierr = KSPSolve(deluxe_ctx->seq_ksp,deluxe_ctx->seq_work2,deluxe_ctx->seq_work1);CHKERRQ(ierr);
    /* fake work due to final ADD VALUES and vertices scaling needed? TODO: check it */
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  /* parallel part */
  for (i=0;i<deluxe_ctx->par_colors;i++) {
    if (deluxe_ctx->par_ksp[i]) {
      ierr = MPI_Comm_rank(deluxe_ctx->par_subcomm[i]->comm,&color_rank);CHKERRQ(ierr);
      ierr = VecSet(deluxe_ctx->work1_B,zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],x,deluxe_ctx->work1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],x,deluxe_ctx->work1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      /* apply local schur on subset S_j^-1 */
      ierr = PCBDDCApplySchur(pc,deluxe_ctx->work1_B,deluxe_ctx->work2_B,(Vec)0,deluxe_ctx->work1_D,deluxe_ctx->work2_D);CHKERRQ(ierr);
      /* parallel transpose solve (\sum_j S_j)^-1 */
      ierr = VecSet(deluxe_ctx->par_vec[i],zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->work2_B,deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->work2_B,deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(deluxe_ctx->par_ksp[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
      if (!color_rank) {
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      } else { /* fake work due to final ADD VALUES and vertices scaling */
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->work1_B,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->work1_B,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      }
    }
  }
  /* put local boundary part in global vector */
  ierr = VecSet(y,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcbddc->work_scaling,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,pcbddc->work_scaling,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingExtension"
PetscErrorCode PCBDDCScalingExtension(PC pc, Vec local_interface_vector, Vec global_vector)
{
  PC_BDDC *pcbddc=(PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(local_interface_vector,VEC_CLASSID,2);
  PetscValidHeaderSpecific(global_vector,VEC_CLASSID,3);
  if (local_interface_vector == pcbddc->work_scaling) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Local vector cannot be pcbddc->work_scaling!\n");
  }
  ierr = PetscTryMethod(pc,"PCBDDCScalingExtension_C",(PC,Vec,Vec),(pc,local_interface_vector,global_vector));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingRestriction_Basic"
static PetscErrorCode PCBDDCScalingRestriction_Basic(PC pc, Vec global_vector, Vec local_interface_vector)
{
  PetscErrorCode ierr;
  PC_IS* pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  ierr = VecScatterBegin(pcis->global_to_B,global_vector,local_interface_vector,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,global_vector,local_interface_vector,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Apply partition of unity */
  ierr = VecPointwiseMult(local_interface_vector,pcis->D,local_interface_vector);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingRestriction_Deluxe"
static PetscErrorCode PCBDDCScalingRestriction_Deluxe(PC pc, Vec x, Vec y)
{
  PC_IS*              pcis=(PC_IS*)pc->data;
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
  PetscScalar         *array_y,*array_D,zero=0.0;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* get local boundary part of global vector */
  ierr = VecScatterBegin(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* scale vertices using diagonal scaling -> every scaling perform the same */
  ierr = VecGetArray(y,&array_y);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->D,&array_D);CHKERRQ(ierr);
  for (i=0;i<deluxe_ctx->n_simple;i++) {
    array_y[deluxe_ctx->idx_simple_B[i]] *= array_D[deluxe_ctx->idx_simple_B[i]];
  }
  ierr = VecRestoreArray(pcis->D,&array_D);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&array_y);CHKERRQ(ierr);
  /* sequential part : all problems and Schur applications collapsed into seq_mat at setup phase */
  if (deluxe_ctx->seq_mat) {
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,y,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,y,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(deluxe_ctx->seq_ksp,deluxe_ctx->seq_work1,deluxe_ctx->seq_work2);CHKERRQ(ierr);
    ierr = MatMultTranspose(deluxe_ctx->seq_mat,deluxe_ctx->seq_work2,deluxe_ctx->seq_work1);CHKERRQ(ierr);
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  /* parallel part */
  for (i=0;i<deluxe_ctx->par_colors;i++) {
    if (deluxe_ctx->par_ksp[i]) {
      /* parallel solve (\sum_j S_j)^-1 */
      ierr = VecSet(deluxe_ctx->par_vec[i],zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],y,deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],y,deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(deluxe_ctx->par_ksp[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
      /* apply local schur S_j^-1 */
      ierr = VecSet(deluxe_ctx->work1_B,zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->work1_B,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->work1_B,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = PCBDDCApplySchurTranspose(pc,deluxe_ctx->work1_B,deluxe_ctx->work2_B,(Vec)0,deluxe_ctx->work1_D,deluxe_ctx->work2_D);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],deluxe_ctx->work2_B,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],deluxe_ctx->work2_B,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingRestriction"
PetscErrorCode PCBDDCScalingRestriction(PC pc, Vec global_vector, Vec local_interface_vector)
{
  PC_BDDC        *pcbddc=(PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(global_vector,VEC_CLASSID,2);
  PetscValidHeaderSpecific(local_interface_vector,VEC_CLASSID,3);
  if (local_interface_vector == pcbddc->work_scaling) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Local vector should cannot be pcbddc->work_scaling!\n");
  }
  ierr = PetscTryMethod(pc,"PCBDDCScalingRestriction_C",(PC,Vec,Vec),(pc,global_vector,local_interface_vector));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingSetUp"
PetscErrorCode PCBDDCScalingSetUp(PC pc)
{
  PC_IS* pcis=(PC_IS*)pc->data;
  PC_BDDC* pcbddc=(PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  /* create work vector for the operator */
  ierr = VecDuplicate(pcis->vec1_B,&pcbddc->work_scaling);CHKERRQ(ierr);
  /* rebuild pcis->D if stiffness scaling has been requested */
  if (pcis->use_stiffness_scaling) {
    ierr = MatGetDiagonal(pcbddc->local_mat,pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  ierr = VecCopy(pcis->D,pcis->vec1_B);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(pcis->D,pcis->D,pcis->vec1_B);CHKERRQ(ierr);
  /* now setup */
#if 0
  if (pcbddc->use_deluxe_scaling) {
    ierr = PCBDDCScalingCreateDeluxe(pc);CHKERRQ(ierr);
    ierr = PCBDDCScalingSetUpDeluxe(pc);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",PCBDDCScalingRestriction_Deluxe);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",PCBDDCScalingExtension_Deluxe);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",PCBDDCScalingRestriction_Basic);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",PCBDDCScalingExtension_Basic);CHKERRQ(ierr);
#if 0
  }
#endif
  /* test */
  if (pcbddc->dbg_flag) {
    PetscViewer viewer=pcbddc->dbg_viewer;
    PetscReal   error,gerror;
    MPI_Comm    test_comm;

    /* extension -> from local to parallel */
    ierr = PetscObjectGetComm((PetscObject)pc,&test_comm);CHKERRQ(ierr);
    ierr = VecSetRandom(pcis->vec1_global,NULL);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PCBDDCScalingExtension(pc,pcis->vec1_B,pcis->vec1_global);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcbddc->work_scaling,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcbddc->work_scaling,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecAXPY(pcis->vec1_B,-1.0,pcbddc->work_scaling);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_B,NORM_INFINITY,&error);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&error,&gerror,1,MPIU_REAL,MPI_MAX,test_comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Error scaling extension %1.14e\n",error);CHKERRQ(ierr);
    if (PetscAbsReal(gerror)>1.e-8) {
      ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecView(pcis->vec1_global,viewer);CHKERRQ(ierr);
    }
    /* restriction -> from parallel to local */
    ierr = VecSetRandom(pcis->vec1_global,NULL);CHKERRQ(ierr);
    ierr = PCBDDCScalingRestriction(pc,pcis->vec1_global,pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcbddc->work_scaling,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcbddc->work_scaling,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecAXPY(pcis->vec1_B,-1.0,pcbddc->work_scaling);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_B,NORM_INFINITY,&error);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&error,&gerror,1,MPIU_REAL,MPI_MAX,test_comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Error scaling restriction %1.14e\n",gerror);CHKERRQ(ierr);
    if (PetscAbsReal(gerror)>1.e-8) {
      ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecView(pcis->vec1_global,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingDestroy"
PetscErrorCode PCBDDCScalingDestroy(PC pc)
{
  PC_BDDC* pcbddc=(PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if 0
  if (pcbddc->use_deluxe_scaling) {
    ierr = PCBDDCScalingDestroyDeluxe(pc);CHKERRQ(ierr);
  }
#endif
  ierr = VecDestroy(&pcbddc->work_scaling);CHKERRQ(ierr);
  /* remove functions */
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

