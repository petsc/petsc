#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>

/* prototypes for deluxe functions */
static PetscErrorCode PCBDDCScalingCreate_Deluxe(PC);
static PetscErrorCode PCBDDCScalingDestroy_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Par(PC,PetscInt,PetscInt,PetscInt[],PetscInt[]);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Seq(PC,PetscInt,PetscInt,PetscInt[],PetscInt[]);
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
  PCBDDCSubSchurs     sub_schurs = deluxe_ctx->sub_schurs;
  PetscInt            i;
  PetscErrorCode      ierr;

  /* TODO CHECK STUFF RELATED WITH FAKE WORK */
  PetscFunctionBegin;
  ierr = VecSet(pcbddc->work_scaling,0.0);CHKERRQ(ierr); /* needed by the fake work below */
  if (deluxe_ctx->n_simple) {
    /* scale deluxe vertices using diagonal scaling */
    PetscScalar *array_x,*array_D,*array;
    ierr = VecGetArray(x,&array_x);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->D,&array_D);CHKERRQ(ierr);
    ierr = VecGetArray(pcbddc->work_scaling,&array);CHKERRQ(ierr);
    for (i=0;i<deluxe_ctx->n_simple;i++) {
      array[deluxe_ctx->idx_simple_B[i]] = array_x[deluxe_ctx->idx_simple_B[i]]*array_D[deluxe_ctx->idx_simple_B[i]];
    }
    ierr = VecRestoreArray(pcbddc->work_scaling,&array);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcis->D,&array_D);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&array_x);CHKERRQ(ierr);
  }
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication and ksp solution */
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
      PetscMPIInt color_rank;
      PetscInt    subidx = deluxe_ctx->par_col2sub[i];
      /* restrict on subset */
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],x,sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],x,sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      /* S_Ej */
      ierr = MatMult(sub_schurs->S_Ej[subidx],sub_schurs->work1[subidx],sub_schurs->work2[subidx]);CHKERRQ(ierr);
      /* (\sum_j S_Ej)^-1 */
      ierr = VecSet(deluxe_ctx->par_vec[i],0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],sub_schurs->work2[subidx],deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],sub_schurs->work2[subidx],deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(deluxe_ctx->par_ksp[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)deluxe_ctx->par_ksp[i]),&color_rank);CHKERRQ(ierr);
      /* get back solution on subset */
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      if (!color_rank) { /* only the master process in coloured comm copies the computed values */
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],sub_schurs->work1[subidx],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],sub_schurs->work1[subidx],pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
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
  PCBDDCSubSchurs     sub_schurs = deluxe_ctx->sub_schurs;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* get local boundary part of global vector */
  ierr = VecScatterBegin(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (deluxe_ctx->n_simple) {
    /* scale deluxe vertices using diagonal scaling */
    PetscScalar *array_y,*array_D;
    ierr = VecGetArray(y,&array_y);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->D,&array_D);CHKERRQ(ierr);
    for (i=0;i<deluxe_ctx->n_simple;i++) {
      array_y[deluxe_ctx->idx_simple_B[i]] *= array_D[deluxe_ctx->idx_simple_B[i]];
    }
    ierr = VecRestoreArray(pcis->D,&array_D);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&array_y);CHKERRQ(ierr);
  }
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication and ksp solution */
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
      PetscInt subidx = deluxe_ctx->par_col2sub[i];
      /* restrict on subset */
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],y,sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],y,sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      /* (\sum_j S_Ej)^-T */
      ierr = VecSet(deluxe_ctx->par_vec[i],0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],sub_schurs->work1[subidx],deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],sub_schurs->work1[subidx],deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(deluxe_ctx->par_ksp[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      /* S_Ej^T */
      ierr = MatMultTranspose(sub_schurs->S_Ej[subidx],sub_schurs->work1[subidx],sub_schurs->work2[subidx]);CHKERRQ(ierr);
      /* extend to boundary */
      ierr = VecScatterBegin(deluxe_ctx->par_scctx_s[i],sub_schurs->work2[subidx],y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(deluxe_ctx->par_scctx_s[i],sub_schurs->work2[subidx],y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
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
  ierr = PCBDDCSubSchursCreate(&deluxe_ctx->sub_schurs);CHKERRQ(ierr);
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
  ierr = PCBDDCSubSchursDestroy(&(pcbddc->deluxe_ctx->sub_schurs));CHKERRQ(ierr);
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
  if (deluxe_ctx->seq_mat) {
    ierr = VecScatterDestroy(&deluxe_ctx->seq_scctx);CHKERRQ(ierr);
    ierr = VecDestroy(&deluxe_ctx->seq_work1);CHKERRQ(ierr);
    ierr = VecDestroy(&deluxe_ctx->seq_work2);CHKERRQ(ierr);
    ierr = MatDestroy(&deluxe_ctx->seq_mat);CHKERRQ(ierr);
    ierr = KSPDestroy(&deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  }
  if (deluxe_ctx->par_colors) {
    PetscInt i;
    for (i=0;i<deluxe_ctx->par_colors;i++) {
      ierr = VecScatterDestroy(&deluxe_ctx->par_scctx_s[i]);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&deluxe_ctx->par_scctx_p[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
      ierr = KSPDestroy(&deluxe_ctx->par_ksp[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree5(deluxe_ctx->par_ksp,
                      deluxe_ctx->par_scctx_s,
                      deluxe_ctx->par_scctx_p,
                      deluxe_ctx->par_vec,
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
  PCBDDCSubSchurs     sub_schurs=deluxe_ctx->sub_schurs;
  PCBDDCGraph         graph;
  IS                  *faces,*edges,*all_cc;
  PetscBT             bitmask;
  PetscInt            *index_sequential,*index_parallel;
  PetscInt            *auxlocal_sequential,*auxlocal_parallel;
  PetscInt            *auxglobal_sequential,*auxglobal_parallel;
  PetscInt            *auxmapping,*idxs;
  PetscInt            i,max_subset_size;
  PetscInt            n_sequential_problems,n_local_sequential_problems,n_parallel_problems,n_local_parallel_problems;
  PetscInt            n_faces,n_edges,n_all_cc;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* throw away the solvers */
  ierr = PCBDDCScalingReset_Deluxe_Solvers(deluxe_ctx);CHKERRQ(ierr);

  /* attach interface graph for determining subsets */
  if (pcbddc->deluxe_rebuild) { /* in case rebuild has been requested, it uses a graph generated only by the neighbouring information */
    PetscInt *idx_V_N;
    IS       verticesIS;
    ierr = PCBDDCGetPrimalVerticesLocalIdx(pc,&i,&idx_V_N);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,i,idx_V_N,PETSC_OWN_POINTER,&verticesIS);CHKERRQ(ierr);
    ierr = PCBDDCGraphCreate(&graph);CHKERRQ(ierr);
    ierr = PCBDDCGraphInit(graph,pcbddc->mat_graph->l2gmap);CHKERRQ(ierr);
    ierr = PCBDDCGraphSetUp(graph,0,NULL,pcbddc->DirichletBoundariesLocal,0,NULL,verticesIS);CHKERRQ(ierr);
    ierr = PCBDDCGraphComputeConnectedComponents(graph);CHKERRQ(ierr);
    ierr = ISDestroy(&verticesIS);CHKERRQ(ierr);
/*
    if (pcbddc->dbg_flag) {
      ierr = PCBDDCGraphASCIIView(graph,pcbddc->dbg_flag,pcbddc->dbg_viewer);CHKERRQ(ierr);
    }
*/
  } else {
    graph = pcbddc->mat_graph;
  }

  /* get index sets for faces and edges */
  ierr = PCBDDCGraphGetCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,NULL);CHKERRQ(ierr);
  n_all_cc = n_faces+n_edges;
  ierr = PetscMalloc1(n_all_cc,&all_cc);CHKERRQ(ierr);
  for (i=0;i<n_faces;i++) {
    all_cc[i] = faces[i];
  }
  for (i=0;i<n_edges;i++) {
    all_cc[n_faces+i] = edges[i];
  }

  /* map interface's subsets */
  max_subset_size = 0;
  for (i=0;i<n_all_cc;i++) {
    PetscInt subset_size;
    ierr = ISGetLocalSize(all_cc[i],&subset_size);CHKERRQ(ierr);
    max_subset_size = PetscMax(max_subset_size,subset_size);
  }
  ierr = PetscMalloc5(max_subset_size,&auxmapping,
                      graph->ncc,&auxlocal_sequential,
                      graph->ncc,&auxlocal_parallel,
                      graph->ncc,&index_sequential,
                      graph->ncc,&index_parallel);CHKERRQ(ierr);

  /* if threshold is negative, uses all sequential problems */
  if (pcbddc->deluxe_threshold < 0) pcbddc->deluxe_threshold = max_subset_size;

  /* workspace */
  ierr = PetscBTCreate(pcis->n,&bitmask);CHKERRQ(ierr);
  ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&idxs);CHKERRQ(ierr);
  for (i=0;i<pcis->n-pcis->n_B;i++) {
    ierr = PetscBTSet(bitmask,idxs[i]);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&idxs);CHKERRQ(ierr);

  /* determine which problem has to be solved in parallel or sequentially */
  n_local_sequential_problems = 0;
  n_local_parallel_problems = 0;
  for (i=0;i<n_all_cc;i++) {
    PetscInt subset_size,j,min_loc = 0;

    ierr = ISGetLocalSize(all_cc[i],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(all_cc[i],(const PetscInt**)&idxs);CHKERRQ(ierr);
    for (j=0;j<subset_size;j++) {
      ierr = PetscBTSet(bitmask,idxs[j]);CHKERRQ(ierr);
    }
    ierr = ISLocalToGlobalMappingApply(graph->l2gmap,subset_size,idxs,auxmapping);CHKERRQ(ierr);
    for (j=1;j<subset_size;j++) {
      if (auxmapping[j]<auxmapping[min_loc]) {
        min_loc = j;
      }
    }
    if (subset_size > pcbddc->deluxe_threshold) {
      index_parallel[n_local_parallel_problems] = i;
      auxlocal_parallel[n_local_parallel_problems] = idxs[min_loc];
      n_local_parallel_problems++;
    } else {
      index_sequential[n_local_sequential_problems] = i;
      auxlocal_sequential[n_local_sequential_problems] = idxs[min_loc];
      n_local_sequential_problems++;
    }
    ierr = ISRestoreIndices(all_cc[i],(const PetscInt**)&idxs);CHKERRQ(ierr);
  }

  /* diagonal scaling on interface dofs not contained in cc */
  deluxe_ctx->n_simple = 0;
  for (i=0;i<pcis->n;i++) {
    if (!PetscBTLookup(bitmask,i)) {
      deluxe_ctx->n_simple++;
    }
  }
  ierr = PetscMalloc1(deluxe_ctx->n_simple,&deluxe_ctx->idx_simple_B);CHKERRQ(ierr);
  deluxe_ctx->n_simple = 0;
  for (i=0;i<pcis->n;i++) {
    if (!PetscBTLookup(bitmask,i)) {
      deluxe_ctx->idx_simple_B[deluxe_ctx->n_simple++] = i;
    }
  }
  ierr = ISGlobalToLocalMappingApply(pcbddc->BtoNmap,IS_GTOLM_DROP,deluxe_ctx->n_simple,deluxe_ctx->idx_simple_B,&i,deluxe_ctx->idx_simple_B);CHKERRQ(ierr);
  if (i != deluxe_ctx->n_simple) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error when mapping simple scaling dofs! %d != %d",i,deluxe_ctx->n_simple);
  }
  ierr = PetscBTDestroy(&bitmask);CHKERRQ(ierr);

  /* SetUp local schur complements on subsets TODO better reuse procedure */
  if (!sub_schurs->n_subs) {
    Mat       S_j;
    PetscBool free_used_adj;
    PetscInt  *used_xadj,*used_adjncy;

    /* decide the adjacency to be used for determining internal problems for local schur on subsets */
    free_used_adj = PETSC_FALSE;
    if (pcbddc->deluxe_layers == -1) {
      used_xadj = NULL;
      used_adjncy = NULL;
    } else {
      if ((pcbddc->deluxe_use_useradj && pcbddc->mat_graph->xadj) || !pcbddc->deluxe_compute_rowadj) {
        used_xadj = pcbddc->mat_graph->xadj;
        used_adjncy = pcbddc->mat_graph->adjncy;
      } else {
        Mat            mat_adj;
        PetscBool      flg_row=PETSC_TRUE;
        const PetscInt *xadj,*adjncy;
        PetscInt       nvtxs;

        ierr = MatConvert(pcbddc->local_mat,MATMPIADJ,MAT_INITIAL_MATRIX,&mat_adj);CHKERRQ(ierr);
        ierr = MatGetRowIJ(mat_adj,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
        if (!flg_row) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatGetRowIJ called in %s\n",__FUNCT__);
        }
        ierr = PetscMalloc2(nvtxs+1,&used_xadj,xadj[nvtxs],&used_adjncy);CHKERRQ(ierr);
        ierr = PetscMemcpy(used_xadj,xadj,(nvtxs+1)*sizeof(*xadj));CHKERRQ(ierr);
        ierr = PetscMemcpy(used_adjncy,adjncy,(xadj[nvtxs])*sizeof(*adjncy));CHKERRQ(ierr);
        ierr = MatRestoreRowIJ(mat_adj,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
        if (!flg_row) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatRestoreRowIJ called in %s\n",__FUNCT__);
        }
        ierr = MatDestroy(&mat_adj);CHKERRQ(ierr);
        free_used_adj = PETSC_TRUE;
      }
    }

    /* Create Schur complement matrix */
    ierr = MatCreateSchurComplement(pcis->A_II,pcis->A_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&S_j);CHKERRQ(ierr);
    ierr = MatSchurComplementSetKSP(S_j,pcbddc->ksp_D);CHKERRQ(ierr);

    /* setup Schur complements on subsets */
    ierr = PCBDDCSubSchursSetUp(sub_schurs,S_j,pcis->is_I_local,pcis->is_B_local,n_all_cc,all_cc,used_xadj,used_adjncy,pcbddc->deluxe_layers);CHKERRQ(ierr);
    ierr = MatDestroy(&S_j);CHKERRQ(ierr);
    /* free adjacency */
    if (free_used_adj) {
      ierr = PetscFree2(used_xadj,used_adjncy);CHKERRQ(ierr);
    }
  }
  for (i=0;i<n_all_cc;i++) {
    ierr = ISDestroy(&all_cc[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(all_cc);CHKERRQ(ierr);

  /* Number parallel problems */
  auxglobal_parallel = 0;
  ierr = PCBDDCSubsetNumbering(PetscObjectComm((PetscObject)pc),graph->l2gmap,n_local_parallel_problems,auxlocal_parallel,PETSC_NULL,&n_parallel_problems,&auxglobal_parallel);CHKERRQ(ierr);
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Deluxe global number of parallel subproblems: %d\n",n_parallel_problems);CHKERRQ(ierr);
  }

  /* Compute data structures to solve parallel problems */
  ierr = PCBDDCScalingSetUp_Deluxe_Par(pc,n_local_parallel_problems,n_parallel_problems,auxglobal_parallel,index_parallel);CHKERRQ(ierr);
  ierr = PetscFree(auxglobal_parallel);CHKERRQ(ierr);


  /* Number sequential problems */
  auxglobal_sequential = 0;
  ierr = PCBDDCSubsetNumbering(PetscObjectComm((PetscObject)pc),graph->l2gmap,n_local_sequential_problems,auxlocal_sequential,PETSC_NULL,&n_sequential_problems,&auxglobal_sequential);CHKERRQ(ierr);
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Deluxe global number of sequential subproblems: %d\n",n_sequential_problems);CHKERRQ(ierr);
  }

  /* Compute data structures to solve sequential problems */
  ierr = PCBDDCScalingSetUp_Deluxe_Seq(pc,n_local_sequential_problems,n_sequential_problems,auxglobal_sequential,index_sequential);CHKERRQ(ierr);
  ierr = PetscFree(auxglobal_sequential);CHKERRQ(ierr);

  /* free workspace */
  ierr = PetscFree5(auxmapping,auxlocal_sequential,auxlocal_parallel,index_sequential,index_parallel);CHKERRQ(ierr);

  /* free graph struct */
  if (pcbddc->deluxe_rebuild) {
    ierr = PCBDDCGraphDestroy(&graph);CHKERRQ(ierr);
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
  ierr = PetscMalloc5(deluxe_ctx->par_colors,&deluxe_ctx->par_ksp,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_scctx_s,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_scctx_p,
                      deluxe_ctx->par_colors,&deluxe_ctx->par_vec,
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
      ierr = MPI_Comm_rank(PetscSubcommChild(par_subcomm),&crank);CHKERRQ(ierr);
      ierr = MPI_Comm_size(PetscSubcommChild(par_subcomm),&csize);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Color %d: size %d, details follows.\n",i,color_size);CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(pcbddc->dbg_viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"  Subdomain %d: color in subcomm %d (rank %d out of %d) (lidx %d)\n",PetscGlobalRank,par_subcomm->color,crank,csize,deluxe_ctx->par_col2sub[i]);CHKERRQ(ierr);
      ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }

    if (deluxe_ctx->par_col2sub[i] >= 0) {
      PC                     dpc;
      Mat                    color_mat,color_mat_is,temp_mat;
      ISLocalToGlobalMapping WtoNmap,l2gmap_subset;
      IS                     is_local_numbering,isB_local,isW_local,isW;
      PCBDDCSubSchurs        sub_schurs = deluxe_ctx->sub_schurs;
      PetscInt               subidx,n_local_dofs,n_global_dofs;
      PetscInt               *global_numbering,*local_numbering;
      char                   ksp_prefix[256];
      size_t                 len;

      /* Local index for schur complement on subset */
      subidx = deluxe_ctx->par_col2sub[i];

      /* Parallel numbering for dofs in colored subset */
      ierr = ISSum(sub_schurs->is_AEj_I[subidx],sub_schurs->is_AEj_B[subidx],&is_local_numbering);CHKERRQ(ierr);
      ierr = ISGetLocalSize(is_local_numbering,&n_local_dofs);CHKERRQ(ierr);
      ierr = ISGetIndices(is_local_numbering,(const PetscInt **)&local_numbering);CHKERRQ(ierr);
      ierr = PCBDDCSubsetNumbering(PetscSubcommChild(par_subcomm),pcbddc->mat_graph->l2gmap,n_local_dofs,local_numbering,PETSC_NULL,&n_global_dofs,&global_numbering);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is_local_numbering,(const PetscInt **)&local_numbering);CHKERRQ(ierr);

      /* L2Gmap from relevant dofs to local dofs */
      ierr = ISLocalToGlobalMappingCreateIS(is_local_numbering,&WtoNmap);CHKERRQ(ierr);

      /* L2Gmap from local to global dofs */
      ierr = ISLocalToGlobalMappingCreate(PetscSubcommChild(par_subcomm),1,n_local_dofs,global_numbering,PETSC_COPY_VALUES,&l2gmap_subset);CHKERRQ(ierr);

      /* compute parallel matrix (extended dirichlet problem on subset) */
      ierr = MatCreateIS(PetscSubcommChild(par_subcomm),1,PETSC_DECIDE,PETSC_DECIDE,n_global_dofs,n_global_dofs,l2gmap_subset,&color_mat_is);CHKERRQ(ierr);
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
      ierr = ISGlobalToLocalMappingApplyIS(pcbddc->BtoNmap,IS_GTOLM_DROP,sub_schurs->is_AEj_B[subidx],&isB_local);CHKERRQ(ierr);
      ierr = VecScatterCreate(pcbddc->work_scaling,isB_local,sub_schurs->work1[subidx],NULL,&deluxe_ctx->par_scctx_s[i]);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(WtoNmap,IS_GTOLM_DROP,sub_schurs->is_AEj_B[subidx],&isW_local);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApplyIS(l2gmap_subset,isW_local,&isW);CHKERRQ(ierr);
      ierr = VecScatterCreate(sub_schurs->work1[subidx],NULL,deluxe_ctx->par_vec[i],isW,&deluxe_ctx->par_scctx_p[i]);CHKERRQ(ierr);

      /* free objects no longer neeeded */
      ierr = ISDestroy(&isW);CHKERRQ(ierr);
      ierr = ISDestroy(&isW_local);CHKERRQ(ierr);
      ierr = ISDestroy(&isB_local);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&WtoNmap);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&l2gmap_subset);CHKERRQ(ierr);
      ierr = ISDestroy(&is_local_numbering);CHKERRQ(ierr);
      ierr = PetscFree(global_numbering);CHKERRQ(ierr);

      /* KSP for extended dirichlet problem */
      ierr = KSPCreate(PetscSubcommChild(par_subcomm),&deluxe_ctx->par_ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(deluxe_ctx->par_ksp[i],pc->erroriffailure);CHKERRQ(ierr);
      ierr = KSPSetOperators(deluxe_ctx->par_ksp[i],color_mat,color_mat);CHKERRQ(ierr);
      ierr = KSPSetTolerances(deluxe_ctx->par_ksp[i],1.e-12,1.e-12,1.e10,10000);CHKERRQ(ierr);
      ierr = KSPSetType(deluxe_ctx->par_ksp[i],KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(deluxe_ctx->par_ksp[i],&dpc);CHKERRQ(ierr);
      ierr = PCSetType(dpc,PCREDUNDANT);CHKERRQ(ierr);
      ierr = PetscStrlen(((PetscObject)(pcbddc->ksp_D))->prefix,&len);CHKERRQ(ierr);
      len -= 10; /* remove "dirichlet_" */
      ierr = PetscStrncpy(ksp_prefix,((PetscObject)(pcbddc->ksp_D))->prefix,len+1);CHKERRQ(ierr); /* PetscStrncpy puts a terminating char at the end */
      ierr = PetscStrcat(ksp_prefix,"deluxe_par_");CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(deluxe_ctx->par_ksp[i],ksp_prefix);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(deluxe_ctx->par_ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetUp(deluxe_ctx->par_ksp[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&color_mat);CHKERRQ(ierr);
    } else { /* not partecipating in color */
      deluxe_ctx->par_ksp[i] = 0;
      deluxe_ctx->par_vec[i] = 0;
      deluxe_ctx->par_scctx_p[i] = 0;
      deluxe_ctx->par_scctx_s[i] = 0;
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
    PCBDDCSubSchurs sub_schurs = deluxe_ctx->sub_schurs;
    /* test partition of unity of coloured schur complements  */
    for (i=0;i<deluxe_ctx->par_colors;i++) {
      PetscInt  subidx = deluxe_ctx->par_col2sub[i];
      PetscBool error_found = PETSC_FALSE;
      ierr = PetscViewerASCIISynchronizedAllow(pcbddc->dbg_viewer,PETSC_TRUE);CHKERRQ(ierr);

      if (deluxe_ctx->par_ksp[i]) {
        /* create random test vec being zero on internal nodes of the extende dirichlet problem */
        ierr = VecDuplicate(deluxe_ctx->par_vec[i],&test_vec);CHKERRQ(ierr);
        ierr = VecSetRandom(sub_schurs->work1[subidx],PETSC_NULL);CHKERRQ(ierr);
        ierr = VecSet(test_vec,0.0);CHKERRQ(ierr);
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],sub_schurs->work1[subidx],test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],sub_schurs->work1[subidx],test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        /* w_j */
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],test_vec,sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],test_vec,sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        /* S_j*w_j */
        ierr = MatMult(sub_schurs->S_Ej[subidx],sub_schurs->work1[subidx],sub_schurs->work2[subidx]);CHKERRQ(ierr);
        /* \sum_j S_j*w_j */
        ierr = VecSet(deluxe_ctx->par_vec[i],0.0);CHKERRQ(ierr);
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],sub_schurs->work2[subidx],deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],sub_schurs->work2[subidx],deluxe_ctx->par_vec[i],ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        /* (\sum_j S_j)^(-1)(\sum_j S_j*w_j) */
        ierr = KSPSolve(deluxe_ctx->par_ksp[i],deluxe_ctx->par_vec[i],deluxe_ctx->par_vec[i]);CHKERRQ(ierr);
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],deluxe_ctx->par_vec[i],sub_schurs->work1[subidx],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
        ierr = VecSet(deluxe_ctx->par_vec[i],0.0);CHKERRQ(ierr);
        ierr = VecScatterBegin(deluxe_ctx->par_scctx_p[i],sub_schurs->work1[subidx],deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(deluxe_ctx->par_scctx_p[i],sub_schurs->work1[subidx],deluxe_ctx->par_vec[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Seq(PC pc,PetscInt n_local_sequential_problems,PetscInt n_sequential_problems,PetscInt global_sequential[],PetscInt local_sequential[])
{
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling deluxe_ctx=pcbddc->deluxe_ctx;
  PCBDDCSubSchurs     sub_schurs = deluxe_ctx->sub_schurs;
  Mat                 global_schur_subsets,*submat_global_schur_subsets,work_mat;
  IS                  is_to,is_from;
  PetscScalar         *array,*fill_vals;
  PetscInt            *all_local_idx_G,*all_local_idx_B,*all_local_idx_N,*all_permutation_G,*dummy_idx;
  PetscInt            i,j,k,local_problem_index;
  PetscInt            subset_size,max_subset_size,max_subset_size_red;
  PetscInt            local_size,global_size;
  PC                  pc_temp;
  MatSolverPackage    solver=NULL;
  char                ksp_prefix[256];
  size_t              len;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (!n_sequential_problems) {
    PetscFunctionReturn(0);
  }
  /* Get info on subset sizes and sum of all subsets sizes */
  max_subset_size = 0;
  local_size = 0;
  for (i=0;i<n_local_sequential_problems;i++) {
    local_problem_index = local_sequential[i];
    ierr = ISGetLocalSize(sub_schurs->is_AEj_B[local_problem_index],&subset_size);CHKERRQ(ierr);
    max_subset_size = PetscMax(subset_size,max_subset_size);
    local_size += subset_size;
  }

  /* Work arrays for local indices */
  ierr = PetscMalloc1(local_size,&all_local_idx_B);CHKERRQ(ierr);
  ierr = PetscMalloc1(local_size,&all_local_idx_N);CHKERRQ(ierr);

  /* Get local indices in local whole numbering and local boundary numbering */
  local_size = 0;
  for (i=0;i<n_local_sequential_problems;i++) {
    PetscInt *idxs;
    /* get info on local problem */
    local_problem_index = local_sequential[i];
    ierr = ISGetLocalSize(sub_schurs->is_AEj_B[local_problem_index],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(sub_schurs->is_AEj_B[local_problem_index],(const PetscInt**)&idxs);CHKERRQ(ierr);
    /* subset indices in local numbering */
    ierr = PetscMemcpy(all_local_idx_N+local_size,idxs,subset_size*sizeof(PetscInt));CHKERRQ(ierr);
    /* subset indices in local boundary numbering */
    ierr = ISGlobalToLocalMappingApply(pcbddc->BtoNmap,IS_GTOLM_DROP,subset_size,idxs,&j,&all_local_idx_B[local_size]);CHKERRQ(ierr);
    ierr = ISRestoreIndices(sub_schurs->is_AEj_B[local_problem_index],(const PetscInt**)&idxs);CHKERRQ(ierr);
    if (j != subset_size) {
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in BDDC deluxe serial %d (BtoNmap)! %d != %d\n",local_problem_index,subset_size,j);
    }
    local_size += subset_size;
  }

  /* Number dofs on all subsets (parallel) and sort numbering */
  ierr = PCBDDCSubsetNumbering(PetscObjectComm((PetscObject)pc),pcbddc->mat_graph->l2gmap,local_size,all_local_idx_N,PETSC_NULL,&global_size,&all_local_idx_G);CHKERRQ(ierr);
  ierr = PetscMalloc1(local_size,&all_permutation_G);CHKERRQ(ierr);
  for (i=0;i<local_size;i++) {
    all_permutation_G[i]=i;
  }
  ierr = PetscSortIntWithPermutation(local_size,all_local_idx_G,all_permutation_G);CHKERRQ(ierr);

  /* Local matrix of all local Schur on subsets */
  ierr = MatCreate(PETSC_COMM_SELF,&deluxe_ctx->seq_mat);CHKERRQ(ierr);
  ierr = MatSetSizes(deluxe_ctx->seq_mat,PETSC_DECIDE,PETSC_DECIDE,local_size,local_size);CHKERRQ(ierr);
  ierr = MatSetType(deluxe_ctx->seq_mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(deluxe_ctx->seq_mat,max_subset_size,PETSC_NULL);CHKERRQ(ierr);

  /* Global matrix of all assembled Schur on subsets */
  ierr = MatCreate(PetscObjectComm((PetscObject)pc),&global_schur_subsets);CHKERRQ(ierr);
  ierr = MatSetSizes(global_schur_subsets,PETSC_DECIDE,PETSC_DECIDE,global_size,global_size);CHKERRQ(ierr);
  ierr = MatSetType(global_schur_subsets,MATAIJ);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&max_subset_size,&max_subset_size_red,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(global_schur_subsets,max_subset_size_red,PETSC_NULL,max_subset_size_red,PETSC_NULL);CHKERRQ(ierr);

  /* Work arrays */
  ierr = PetscMalloc2(max_subset_size,&dummy_idx,max_subset_size*max_subset_size,&fill_vals);CHKERRQ(ierr);

  /* Loop on local problems to compute Schur complements explicitly */
  local_size = 0;
  for (i=0;i<n_local_sequential_problems;i++) {
    /* get info on local problem */
    local_problem_index = local_sequential[i];
    ierr = ISGetLocalSize(sub_schurs->is_AEj_B[local_problem_index],&subset_size);CHKERRQ(ierr);
    /* local Schur */
    for (j=0;j<subset_size;j++) {
      ierr = VecSet(sub_schurs->work1[local_problem_index],0.0);CHKERRQ(ierr);
      ierr = VecSetValue(sub_schurs->work1[local_problem_index],j,1.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatMult(sub_schurs->S_Ej[local_problem_index],sub_schurs->work1[local_problem_index],sub_schurs->work2[local_problem_index]);CHKERRQ(ierr);
      /* store vals */
      ierr = VecGetArray(sub_schurs->work2[local_problem_index],&array);CHKERRQ(ierr);
      for (k=0;k<subset_size;k++) {
        fill_vals[k*subset_size+j] = array[k];
      }
      ierr = VecRestoreArray(sub_schurs->work2[local_problem_index],&array);CHKERRQ(ierr);
    }
    for (j=0;j<subset_size;j++) {
      dummy_idx[j]=local_size+j;
    }
    ierr = MatSetValues(deluxe_ctx->seq_mat,subset_size,dummy_idx,subset_size,dummy_idx,fill_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(global_schur_subsets,subset_size,&all_local_idx_G[local_size],subset_size,&all_local_idx_G[local_size],fill_vals,ADD_VALUES);CHKERRQ(ierr);
    local_size += subset_size;
  }
  ierr = MatAssemblyBegin(deluxe_ctx->seq_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(deluxe_ctx->seq_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(global_schur_subsets,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(global_schur_subsets,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree2(dummy_idx,fill_vals);CHKERRQ(ierr);

  /* Create work vectors for sequential part of deluxe */
  ierr = MatCreateVecs(deluxe_ctx->seq_mat,&deluxe_ctx->seq_work1,&deluxe_ctx->seq_work2);CHKERRQ(ierr);

  /* Compute deluxe sequential scatter */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_B,PETSC_OWN_POINTER,&is_from);CHKERRQ(ierr);
  ierr = VecScatterCreate(pcbddc->work_scaling,is_from,deluxe_ctx->seq_work1,NULL,&deluxe_ctx->seq_scctx);CHKERRQ(ierr);
  ierr = ISDestroy(&is_from);CHKERRQ(ierr);

  /* Get local part of (\sum_j S_Ej) */
  for (i=0;i<local_size;i++) {
    all_local_idx_N[i] = all_local_idx_G[all_permutation_G[i]];
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),local_size,all_local_idx_N,PETSC_OWN_POINTER,&is_to);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(global_schur_subsets,1,&is_to,&is_to,MAT_INITIAL_MATRIX,&submat_global_schur_subsets);CHKERRQ(ierr);
  ierr = MatDestroy(&global_schur_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&is_to);CHKERRQ(ierr);
  for (i=0;i<local_size;i++) {
    all_local_idx_G[all_permutation_G[i]] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_G,PETSC_OWN_POINTER,&is_from);CHKERRQ(ierr);
  ierr = ISSetPermutation(is_from);CHKERRQ(ierr);
  ierr = MatPermute(submat_global_schur_subsets[0],is_from,is_from,&work_mat);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&submat_global_schur_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&is_from);CHKERRQ(ierr);
  ierr = PetscFree(all_permutation_G);CHKERRQ(ierr);

  /* Create KSP object for sequential part of deluxe scaling */
  ierr = KSPCreate(PETSC_COMM_SELF,&deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(deluxe_ctx->seq_ksp,pc->erroriffailure);CHKERRQ(ierr);
  ierr = KSPSetOperators(deluxe_ctx->seq_ksp,work_mat,work_mat);CHKERRQ(ierr);
  ierr = KSPSetType(deluxe_ctx->seq_ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(deluxe_ctx->seq_ksp,&pc_temp);CHKERRQ(ierr);
  ierr = PCSetType(pc_temp,PCLU);CHKERRQ(ierr);
  ierr = KSPGetPC(pcbddc->ksp_D,&pc_temp);CHKERRQ(ierr);
  ierr = PCFactorGetMatSolverPackage(pc_temp,(const MatSolverPackage*)&solver);CHKERRQ(ierr);
  if (solver) {
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
  ierr = KSPSetFromOptions(deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&work_mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
