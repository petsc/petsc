#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>

/* prototypes for deluxe functions */
static PetscErrorCode PCBDDCScalingCreate_Deluxe(PC);
static PetscErrorCode PCBDDCScalingDestroy_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Par(PC,PetscInt,PetscInt,PetscInt[],PetscInt[]);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Seq(PC);
static PetscErrorCode PCBDDCScalingReset_Deluxe_Solvers(PCBDDCDeluxeScaling);

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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingExtension_Deluxe"
static PetscErrorCode PCBDDCScalingExtension_Deluxe(PC pc, Vec x, Vec y)
{
  PC_IS*              pcis=(PC_IS*)pc->data;
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
  PCBDDCSubSchurs     sub_schurs = pcbddc->sub_schurs;
  PetscInt            i;
  PetscErrorCode      ierr;

  /* TODO CHECK STUFF RELATED WITH FAKE WORK */
  PetscFunctionBegin;
  ierr = VecSet(pcbddc->work_scaling,0.0);CHKERRQ(ierr); /* needed by the fake work below */
  if (deluxe_ctx->n_simple) {
    /* scale deluxe vertices using diagonal scaling */
    const PetscScalar *array_x,*array_D;
    PetscScalar       *array;
    ierr = VecGetArrayRead(x,&array_x);CHKERRQ(ierr);
    ierr = VecGetArrayRead(pcis->D,&array_D);CHKERRQ(ierr);
    ierr = VecGetArray(pcbddc->work_scaling,&array);CHKERRQ(ierr);
    for (i=0;i<deluxe_ctx->n_simple;i++) {
      array[deluxe_ctx->idx_simple_B[i]] = array_x[deluxe_ctx->idx_simple_B[i]]*array_D[deluxe_ctx->idx_simple_B[i]];
    }
    ierr = VecRestoreArray(pcbddc->work_scaling,&array);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(pcis->D,&array_D);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x,&array_x);CHKERRQ(ierr);
  }
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication and ksp solution */
  if (deluxe_ctx->seq_ksp) {
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,x,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,x,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatMultTranspose(sub_schurs->S_Ej_all,deluxe_ctx->seq_work1,deluxe_ctx->seq_work2);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(deluxe_ctx->seq_ksp,deluxe_ctx->seq_work2,deluxe_ctx->seq_work1);CHKERRQ(ierr);
    /* fake work due to final ADD VALUES and vertices scaling needed? TODO: check it */
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  /* parallel part */
  for (i=0;i<deluxe_ctx->par_colors;i++) {
    if (deluxe_ctx->par_ksp[i]) {
      PetscMPIInt color_rank;
      PetscInt    subidx = deluxe_ctx->par_col2sub[i];
      /* restrict on subset */
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],x,deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],x,deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      /* S_Ej */
      ierr = MatMult(sub_schurs->S_Ej[subidx],deluxe_ctx->par_work1[i],deluxe_ctx->par_work2[i]);CHKERRQ(ierr);
      /* (\sum_j S_Ej)^-1 */
      ierr = VecSet(deluxe_ctx->par_vec[i],0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work2[i],deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work2[i],deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(deluxe_ctx->par_ksp[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)deluxe_ctx->par_ksp[i]),&color_rank);CHKERRQ(ierr);
      /* get back solution on subset */
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      if (!color_rank) { /* only the master process in coloured comm copies the computed values */
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],deluxe_ctx->par_work1[i],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],deluxe_ctx->par_work1[i],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      }
    }
  }
  /* put local boundary part in global vector */
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcbddc->work_scaling,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,pcbddc->work_scaling,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingRestriction_Deluxe"
static PetscErrorCode PCBDDCScalingRestriction_Deluxe(PC pc, Vec x, Vec y)
{
  PC_IS*              pcis=(PC_IS*)pc->data;
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx = pcbddc->deluxe_ctx;
  PCBDDCSubSchurs     sub_schurs = pcbddc->sub_schurs;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* get local boundary part of global vector */
  ierr = VecScatterBegin(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (deluxe_ctx->n_simple) {
    /* scale deluxe vertices using diagonal scaling */
    PetscScalar       *array_y;
    const PetscScalar *array_D;
    ierr = VecGetArray(y,&array_y);CHKERRQ(ierr);
    ierr = VecGetArrayRead(pcis->D,&array_D);CHKERRQ(ierr);
    for (i=0;i<deluxe_ctx->n_simple;i++) {
      array_y[deluxe_ctx->idx_simple_B[i]] *= array_D[deluxe_ctx->idx_simple_B[i]];
    }
    ierr = VecRestoreArrayRead(pcis->D,&array_D);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&array_y);CHKERRQ(ierr);
  }
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication and ksp solution */
  if (deluxe_ctx->seq_ksp) {
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,y,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,y,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = KSPSolve(deluxe_ctx->seq_ksp,deluxe_ctx->seq_work1,deluxe_ctx->seq_work2);CHKERRQ(ierr);
    ierr = MatMult(sub_schurs->S_Ej_all,deluxe_ctx->seq_work2,deluxe_ctx->seq_work1);CHKERRQ(ierr);
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work1,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  /* parallel part */
  for (i=0;i<deluxe_ctx->par_colors;i++) {
    if (deluxe_ctx->par_ksp[i]) {
      PetscInt subidx = deluxe_ctx->par_col2sub[i];
      /* restrict on subset */
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],y,deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],y,deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      /* (\sum_j S_Ej)^-T */
      ierr = VecSet(deluxe_ctx->par_vec[i],0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work1[i],deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work1[i],deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(deluxe_ctx->par_ksp[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      /* S_Ej^T */
      ierr = MatMultTranspose(sub_schurs->S_Ej[subidx],deluxe_ctx->par_work1[i],deluxe_ctx->par_work2[i]);CHKERRQ(ierr);
      /* extend to boundary */
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],deluxe_ctx->par_work2[i],y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],deluxe_ctx->par_work2[i],y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

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
  ierr = VecDestroy(&pcbddc->work_scaling);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_B,&pcbddc->work_scaling);CHKERRQ(ierr);
  /* always rebuild pcis->D */
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
  if (pcbddc->use_deluxe_scaling) {
    if (!pcbddc->deluxe_ctx) {
      ierr = PCBDDCScalingCreate_Deluxe(pc);CHKERRQ(ierr);
    }
    ierr = PCBDDCScalingSetUp_Deluxe(pc);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",PCBDDCScalingRestriction_Deluxe);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",PCBDDCScalingExtension_Deluxe);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",PCBDDCScalingRestriction_Basic);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",PCBDDCScalingExtension_Basic);CHKERRQ(ierr);
  }

  /* test */
  if (pcbddc->dbg_flag) {
    Vec         vec2_global;
    PetscViewer viewer=pcbddc->dbg_viewer;
    PetscReal   error;

    /* extension -> from local to parallel */
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecSetRandom(pcis->vec1_B,NULL);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec1_global,&vec2_global);CHKERRQ(ierr);
    ierr = VecCopy(pcis->vec1_global,vec2_global);CHKERRQ(ierr);

    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PCBDDCScalingExtension(pc,pcis->vec1_B,pcis->vec1_global);CHKERRQ(ierr);
    ierr = VecAXPY(pcis->vec1_global,-1.0,vec2_global);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_global,NORM_INFINITY,&error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Error scaling extension %1.14e\n",error);CHKERRQ(ierr);
    if (error>1.e-8 && pcbddc->dbg_flag>1) {
      ierr = VecView(pcis->vec1_global,viewer);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&vec2_global);CHKERRQ(ierr);

    /* restriction -> from parallel to local */
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecSetRandom(pcis->vec1_B,NULL);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = PCBDDCScalingRestriction(pc,pcis->vec1_global,pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecScale(pcis->vec1_B,-1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_global,NORM_INFINITY,&error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Error scaling restriction %1.14e\n",error);CHKERRQ(ierr);
    if (error>1.e-8 && pcbddc->dbg_flag>1) {
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
  if (pcbddc->deluxe_ctx) {
    ierr = PCBDDCScalingDestroy_Deluxe(pc);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&pcbddc->work_scaling);CHKERRQ(ierr);
  /* remove functions */
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingRestriction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCScalingExtension_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingCreate_Deluxe"
static PetscErrorCode PCBDDCScalingCreate_Deluxe(PC pc)
{
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&deluxe_ctx);CHKERRQ(ierr);
  pcbddc->deluxe_ctx = deluxe_ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingDestroy_Deluxe"
static PetscErrorCode PCBDDCScalingDestroy_Deluxe(PC pc)
{
  PC_BDDC*            pcbddc=(PC_BDDC*)pc->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PCBDDCScalingReset_Deluxe_Solvers(pcbddc->deluxe_ctx);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->deluxe_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingReset_Deluxe_Solvers"
static PetscErrorCode PCBDDCScalingReset_Deluxe_Solvers(PCBDDCDeluxeScaling deluxe_ctx)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscFree(deluxe_ctx->idx_simple_B);CHKERRQ(ierr);
  deluxe_ctx->n_simple = 0;
  if (deluxe_ctx->seq_ksp) {
    ierr = VecScatterDestroy(&deluxe_ctx->seq_scctx);CHKERRQ(ierr);
    ierr = VecDestroy(&deluxe_ctx->seq_work1);CHKERRQ(ierr);
    ierr = VecDestroy(&deluxe_ctx->seq_work2);CHKERRQ(ierr);
    ierr = KSPDestroy(&deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  }
  if (deluxe_ctx->par_colors) {
    PetscInt i;
    for (i=0;i<deluxe_ctx->par_colors;i++) {
      ierr = VecScatterDestroy(&deluxe_ctx->par_scctx_s[i]);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&deluxe_ctx->par_scctx_p[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&deluxe_ctx->par_work1[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&deluxe_ctx->par_work2[i]);CHKERRQ(ierr);
      ierr = KSPDestroy(&deluxe_ctx->par_ksp[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree7(deluxe_ctx->par_ksp,
                      deluxe_ctx->par_scctx_s,
                      deluxe_ctx->par_scctx_p,
                      deluxe_ctx->par_vec,
                      deluxe_ctx->par_work1,
                      deluxe_ctx->par_work2,
                      deluxe_ctx->par_col2sub);CHKERRQ(ierr);
  }
  deluxe_ctx->par_colors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingSetUp_Deluxe"
static PetscErrorCode PCBDDCScalingSetUp_Deluxe(PC pc)
{
  PC_IS               *pcis=(PC_IS*)pc->data;
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx=pcbddc->deluxe_ctx;
  PCBDDCSubSchurs     sub_schurs=pcbddc->sub_schurs;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* (TODO: reuse) throw away the solvers */
  ierr = PCBDDCScalingReset_Deluxe_Solvers(deluxe_ctx);CHKERRQ(ierr);

  /* Compute data structures to solve parallel problems */
  ierr = PCBDDCScalingSetUp_Deluxe_Par(pc,sub_schurs->n_subs_par,sub_schurs->n_subs_par_g,
                                       sub_schurs->auxglobal_parallel,
                                       sub_schurs->index_parallel);CHKERRQ(ierr);

  /* Compute data structures to solve sequential problems */
  ierr = PCBDDCScalingSetUp_Deluxe_Seq(pc);CHKERRQ(ierr);

  /* diagonal scaling on interface dofs not contained in cc */
  if (sub_schurs->is_Ej_com) {
    PetscInt       nmap;
    const PetscInt *idxs;
    ierr = ISGetLocalSize(sub_schurs->is_Ej_com,&deluxe_ctx->n_simple);CHKERRQ(ierr);
    ierr = ISGetIndices(sub_schurs->is_Ej_com,&idxs);CHKERRQ(ierr);
    ierr = PetscMalloc1(deluxe_ctx->n_simple,&deluxe_ctx->idx_simple_B);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,deluxe_ctx->n_simple,idxs,&nmap,deluxe_ctx->idx_simple_B);CHKERRQ(ierr);
    if (nmap != deluxe_ctx->n_simple) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error when mapping simply scaled dofs! %d != %d",nmap,deluxe_ctx->n_simple);
    }
    ierr = ISRestoreIndices(sub_schurs->is_Ej_com,&idxs);CHKERRQ(ierr);
  } else {
    deluxe_ctx->n_simple = 0;
    deluxe_ctx->idx_simple_B = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingSetUp_Deluxe_Par"
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Par(PC pc, PetscInt n_local_parallel_problems,PetscInt n_parallel_problems,PetscInt global_parallel[],PetscInt index_parallel[])
{
  PC_BDDC                *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling    deluxe_ctx=pcbddc->deluxe_ctx;
  /* coloring */
  Mat                    parallel_problems;
  MatColoring            coloring_obj;
  ISColoring             coloring_parallel_problems;
  IS                     *par_is_colors,*is_colors;
  /* working stuff */
  PetscInt               i,j;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (!n_parallel_problems) {
    PetscFunctionReturn(0);
  }
  /* Color parallel subproblems */
  ierr = MatCreate(PetscObjectComm((PetscObject)pc),&parallel_problems);CHKERRQ(ierr);
  ierr = MatSetSizes(parallel_problems,PETSC_DECIDE,PETSC_DECIDE,n_parallel_problems,n_parallel_problems);CHKERRQ(ierr);
  ierr = MatSetType(parallel_problems,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(parallel_problems);CHKERRQ(ierr);
  ierr = MatSetOption(parallel_problems,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOption(parallel_problems,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0;i<n_local_parallel_problems;i++) {
    PetscInt row = global_parallel[i];
    for (j=0;j<n_local_parallel_problems;j++) {
      PetscInt col = global_parallel[j];
      if (row != col) {
        ierr = MatSetValue(parallel_problems,row,col,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(parallel_problems,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(parallel_problems,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (pcbddc->dbg_flag > 1) {
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Adj matrix for deluxe parallel problems\n");CHKERRQ(ierr);
    ierr = MatView(parallel_problems,pcbddc->dbg_viewer);CHKERRQ(ierr);
  }
  ierr = MatColoringCreate(parallel_problems,&coloring_obj);CHKERRQ(ierr);
  ierr = MatColoringSetDistance(coloring_obj,1);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring_obj,MATCOLORINGJP);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring_obj,&coloring_parallel_problems);CHKERRQ(ierr);
  ierr = ISColoringGetIS(coloring_parallel_problems,&deluxe_ctx->par_colors,&par_is_colors);CHKERRQ(ierr);
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Number of colors %d for parallel part of deluxe\n",deluxe_ctx->par_colors);CHKERRQ(ierr);
  }

  /* all procs should know the color distribution */
  ierr = PetscMalloc1(deluxe_ctx->par_colors,&is_colors);CHKERRQ(ierr);
  for (i=0;i<deluxe_ctx->par_colors;i++) {
    if (pcbddc->dbg_flag) {
      ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Global problem indexes for color %d\n",i);CHKERRQ(ierr);
      ierr = ISView(par_is_colors[i],pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }
    ierr = ISAllGather(par_is_colors[i],&is_colors[i]);CHKERRQ(ierr);
  }

  /* free unneeded objects */
  ierr = ISColoringRestoreIS(coloring_parallel_problems,&par_is_colors);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&coloring_parallel_problems);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring_obj);CHKERRQ(ierr);
  ierr = MatDestroy(&parallel_problems);CHKERRQ(ierr);

  /* allocate deluxe arrays for parallel problems */
  ierr = PetscCalloc7(deluxe_ctx->par_colors,&deluxe_ctx->par_ksp,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_scctx_s,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_scctx_p,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_vec,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_work1,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_work2,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_col2sub);CHKERRQ(ierr);

  /* cycle on colors */
  for (i=0;i<deluxe_ctx->par_colors;i++) {
    PetscSubcomm    par_subcomm;
    const PetscInt* idxs_subproblems;
    PetscInt        color_size;
    PetscMPIInt     rank,active_color;

    /* get local index of i-th parallel colored problem */
    ierr = ISGetLocalSize(is_colors[i],&color_size);CHKERRQ(ierr);
    ierr = ISGetIndices(is_colors[i],&idxs_subproblems);CHKERRQ(ierr);
    /* split comm for computing parallel problems for this color */
    /* Processes not partecipating at this stage will have color = color_size */
    /* because PetscCommDuplicate does not handle MPI_COMM_NULL */
    active_color = color_size;
    deluxe_ctx->par_col2sub[i] = -1;
    for (j=0;j<n_local_parallel_problems;j++) {
      PetscInt local_idx;
      ierr = PetscFindInt(global_parallel[j],color_size,idxs_subproblems,&local_idx);CHKERRQ(ierr);
      if (local_idx > -1) {
        ierr = PetscMPIIntCast(local_idx,&active_color);CHKERRQ(ierr);
        deluxe_ctx->par_col2sub[i] = index_parallel[j];
        break;
      }
    }
    ierr = ISRestoreIndices(is_colors[i],&idxs_subproblems);CHKERRQ(ierr);
    ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)pc),&par_subcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(par_subcomm,color_size+1);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
    ierr = PetscSubcommSetTypeGeneral(par_subcomm,active_color,rank);CHKERRQ(ierr);
    /* print debug info */
    if (pcbddc->dbg_flag) {
      PetscMPIInt crank,csize;
      ierr = MPI_Comm_rank(par_subcomm->comm,&crank);CHKERRQ(ierr);
      ierr = MPI_Comm_size(par_subcomm->comm,&csize);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Color %d: size %d, details follows.\n",i,color_size);CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(pcbddc->dbg_viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"  Subdomain %d: color in subcomm %d (rank %d out of %d) (lidx %d)\n",PetscGlobalRank,par_subcomm->color,crank,csize,deluxe_ctx->par_col2sub[i]);CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }

    if (deluxe_ctx->par_col2sub[i] >= 0) {
      PC                     pctemp;
      PC_IS                  *pcis=(PC_IS*)pc->data;
      Mat                    color_mat,color_mat_is,temp_mat;
      ISLocalToGlobalMapping WtoNmap,l2gmap_subset;
      IS                     is_local_numbering,isB_local,isW_local,isW;
      PCBDDCSubSchurs        sub_schurs = pcbddc->sub_schurs;
      PetscInt               subidx,n_local_dofs,n_global_dofs;
      PetscInt               *global_numbering,*local_numbering;
      char                   ksp_prefix[256];
      size_t                 len;

      /* Local index for schur complement on subset */
      subidx = deluxe_ctx->par_col2sub[i];

      /* Parallel numbering for dofs in colored subset */
      ierr = ISSum(sub_schurs->is_I_layer,sub_schurs->is_subs[subidx],&is_local_numbering);CHKERRQ(ierr);
      ierr = ISGetLocalSize(is_local_numbering,&n_local_dofs);CHKERRQ(ierr);
      ierr = ISGetIndices(is_local_numbering,(const PetscInt **)&local_numbering);CHKERRQ(ierr);
      ierr = PCBDDCSubsetNumbering(par_subcomm->comm,pcbddc->mat_graph->l2gmap,n_local_dofs,local_numbering,PETSC_NULL,&n_global_dofs,&global_numbering);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is_local_numbering,(const PetscInt **)&local_numbering);CHKERRQ(ierr);

      /* L2Gmap from relevant dofs to local dofs */
      ierr = ISLocalToGlobalMappingCreateIS(is_local_numbering,&WtoNmap);CHKERRQ(ierr);

      /* L2Gmap from local to global dofs */
      ierr = ISLocalToGlobalMappingCreate(par_subcomm->comm,1,n_local_dofs,global_numbering,PETSC_COPY_VALUES,&l2gmap_subset);CHKERRQ(ierr);

      /* compute parallel matrix (extended dirichlet problem on subset) */
      ierr = MatCreateIS(par_subcomm->comm,1,PETSC_DECIDE,PETSC_DECIDE,n_global_dofs,n_global_dofs,l2gmap_subset,&color_mat_is);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,is_local_numbering,is_local_numbering,MAT_INITIAL_MATRIX,&temp_mat);CHKERRQ(ierr);
      ierr = MatISSetLocalMat(color_mat_is,temp_mat);CHKERRQ(ierr);
      ierr = MatDestroy(&temp_mat);CHKERRQ(ierr);
      ierr = MatISGetMPIXAIJ(color_mat_is,MAT_INITIAL_MATRIX,&color_mat);CHKERRQ(ierr);
      ierr = MatDestroy(&color_mat_is);CHKERRQ(ierr);

      /* work vector for (parallel) extended dirichlet problem */
      ierr = MatCreateVecs(color_mat,&deluxe_ctx->par_vec[i],NULL);CHKERRQ(ierr);

      /* compute scatters */
      /* deluxe_ctx->par_scctx_p[i] extension from local subset to extended dirichlet problem
         deluxe_ctx->par_scctx_s[i] restriction from local boundary to subset -> simple copy of selected values */
      ierr = ISGlobalToLocalMappingApplyIS(pcis->BtoNmap,IS_GTOLM_DROP,sub_schurs->is_subs[subidx],&isB_local);CHKERRQ(ierr);
      ierr = MatCreateVecs(sub_schurs->S_Ej[subidx],&deluxe_ctx->par_work1[i],&deluxe_ctx->par_work2[i]);CHKERRQ(ierr);
      ierr = VecScatterCreate(pcbddc->work_scaling,isB_local,deluxe_ctx->par_work1[i],NULL,&deluxe_ctx->par_scctx_s[i]);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(WtoNmap,IS_GTOLM_DROP,sub_schurs->is_subs[subidx],&isW_local);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApplyIS(l2gmap_subset,isW_local,&isW);CHKERRQ(ierr);
      ierr = VecScatterCreate(deluxe_ctx->par_work1[i],NULL,deluxe_ctx->par_vec[i],isW,&deluxe_ctx->par_scctx_p[i]);CHKERRQ(ierr);

      /* free objects no longer neeeded */
      ierr = ISDestroy(&isW);CHKERRQ(ierr);
      ierr = ISDestroy(&isW_local);CHKERRQ(ierr);
      ierr = ISDestroy(&isB_local);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&WtoNmap);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&l2gmap_subset);CHKERRQ(ierr);
      ierr = ISDestroy(&is_local_numbering);CHKERRQ(ierr);
      ierr = PetscFree(global_numbering);CHKERRQ(ierr);

      /* KSP for extended dirichlet problem */
      ierr = KSPCreate(par_subcomm->comm,&deluxe_ctx->par_ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetOperators(deluxe_ctx->par_ksp[i],color_mat,color_mat);CHKERRQ(ierr);
      ierr = KSPSetTolerances(deluxe_ctx->par_ksp[i],1.e-12,1.e-12,1.e10,10000);CHKERRQ(ierr);
      ierr = KSPSetType(deluxe_ctx->par_ksp[i],KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(deluxe_ctx->par_ksp[i],&pctemp);CHKERRQ(ierr);
      ierr = PCSetType(pctemp,PCREDUNDANT);CHKERRQ(ierr);
      ierr = PetscStrlen(((PetscObject)(pcbddc->ksp_D))->prefix,&len);CHKERRQ(ierr);
      len -= 10; /* remove "dirichlet_" */
      ierr = PetscStrncpy(ksp_prefix,((PetscObject)(pcbddc->ksp_D))->prefix,len+1);CHKERRQ(ierr); /* PetscStrncpy puts a terminating char at the end */
      ierr = PetscStrcat(ksp_prefix,"deluxe_par_");CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(deluxe_ctx->par_ksp[i],ksp_prefix);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(deluxe_ctx->par_ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetUp(deluxe_ctx->par_ksp[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&color_mat);CHKERRQ(ierr);
    }
    ierr = PetscSubcommDestroy(&par_subcomm);CHKERRQ(ierr);
  }
  for (i=0;i<deluxe_ctx->par_colors;i++) {
    ierr = ISDestroy(&is_colors[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(is_colors);CHKERRQ(ierr);

  if (pcbddc->dbg_flag) {
    Vec test_vec;
    PetscReal error;
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
    /* test partition of unity of coloured schur complements  */
    for (i=0;i<deluxe_ctx->par_colors;i++) {
      PetscInt  subidx = deluxe_ctx->par_col2sub[i];
      PetscBool error_found = PETSC_FALSE;
      ierr = PetscViewerASCIISynchronizedAllow(pcbddc->dbg_viewer,PETSC_TRUE);CHKERRQ(ierr);

      if (deluxe_ctx->par_ksp[i]) {
        /* create random test vec being zero on internal nodes of the extende dirichlet problem */
        ierr = VecDuplicate(deluxe_ctx->par_vec[i],&test_vec);CHKERRQ(ierr);
        ierr = VecSetRandom(deluxe_ctx->par_work1[i],PETSC_NULL);CHKERRQ(ierr);
        ierr = VecSet(test_vec,0.0);CHKERRQ(ierr);
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work1[i],test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work1[i],test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        /* w_j */
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],test_vec,deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],test_vec,deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        /* S_j*w_j */
        ierr = MatMult(sub_schurs->S_Ej[subidx],deluxe_ctx->par_work1[i],deluxe_ctx->par_work2[i]);CHKERRQ(ierr);
        /* \sum_j S_j*w_j */
        ierr = VecSet(deluxe_ctx->par_vec[i],0.0);CHKERRQ(ierr);
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work2[i],deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work2[i],deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        /* (\sum_j S_j)^(-1)(\sum_j S_j*w_j) */
        ierr = KSPSolve(deluxe_ctx->par_ksp[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_work1[i],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecSet(deluxe_ctx->par_vec[i],0.0);CHKERRQ(ierr);
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work1[i],deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_work1[i],deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        /* test partition of unity */
        ierr = VecAXPY(test_vec,-1.0,deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
        ierr = VecNorm(test_vec,NORM_INFINITY,&error);CHKERRQ(ierr);
        if (PetscAbsReal(error) > 1.e-2) {
          /* ierr = VecView(test_vec,0);CHKERRQ(ierr); */
          error_found = PETSC_TRUE;
        }
        ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
      }
      if (error_found) {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Error testing local schur for color %d and subdomain %d\n",i,PetscGlobalRank);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingSetUp_Deluxe_Seq"
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Seq(PC pc)
{
  PC_BDDC                *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling    deluxe_ctx=pcbddc->deluxe_ctx;
  PCBDDCSubSchurs        sub_schurs = pcbddc->sub_schurs;
  PC                     pc_temp;
  MatSolverPackage       solver=NULL;
  char                   ksp_prefix[256];
  size_t                 len;
  PetscInt               local_size;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (!sub_schurs->n_subs_seq_g) {
    PetscFunctionReturn(0);
  }

  /* Create work vectors for sequential part of deluxe */
  ierr = MatCreateVecs(sub_schurs->sum_S_Ej_all,&deluxe_ctx->seq_work1,&deluxe_ctx->seq_work2);CHKERRQ(ierr);

  /* Compute deluxe sequential scatter */
  ierr = VecScatterCreate(pcbddc->work_scaling,sub_schurs->is_Ej_all,deluxe_ctx->seq_work1,NULL,&deluxe_ctx->seq_scctx);CHKERRQ(ierr);

  /* Create KSP object for sequential part of deluxe scaling */
  ierr = KSPCreate(PETSC_COMM_SELF,&deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(deluxe_ctx->seq_ksp,sub_schurs->sum_S_Ej_all,sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
  ierr = KSPSetType(deluxe_ctx->seq_ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(deluxe_ctx->seq_ksp,&pc_temp);CHKERRQ(ierr);
  ierr = PCSetType(pc_temp,PCLU);CHKERRQ(ierr);
  ierr = KSPGetPC(pcbddc->ksp_D,&pc_temp);CHKERRQ(ierr);
  ierr = PCFactorGetMatSolverPackage(pc_temp,(const MatSolverPackage*)&solver);CHKERRQ(ierr);
  ierr = MatGetSize(sub_schurs->sum_S_Ej_all,&local_size,NULL);CHKERRQ(ierr);
  if (solver && local_size) { /* if local_size is null, some external packages will report errors */
    PC     new_pc;
    PCType type;
    ierr = PCGetType(pc_temp,&type);CHKERRQ(ierr);
    ierr = KSPGetPC(deluxe_ctx->seq_ksp,&new_pc);CHKERRQ(ierr);
    ierr = PCSetType(new_pc,type);CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverPackage(new_pc,solver);CHKERRQ(ierr);
  }
  ierr = PetscStrlen(((PetscObject)(pcbddc->ksp_D))->prefix,&len);CHKERRQ(ierr);
  len -= 10; /* remove "dirichlet_" */
  ierr = PetscStrncpy(ksp_prefix,((PetscObject)(pcbddc->ksp_D))->prefix,len+1);CHKERRQ(ierr);
  ierr = PetscStrcat(ksp_prefix,"deluxe_seq_");CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(deluxe_ctx->seq_ksp,ksp_prefix);CHKERRQ(ierr);
  if (local_size) {
    ierr = KSPSetFromOptions(deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  }
  ierr = KSPSetUp(deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
