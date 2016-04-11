#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>

/* prototypes for deluxe functions */
static PetscErrorCode PCBDDCScalingCreate_Deluxe(PC);
static PetscErrorCode PCBDDCScalingDestroy_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe(PC);
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Private(PC);
static PetscErrorCode PCBDDCScalingReset_Deluxe_Solvers(PCBDDCDeluxeScaling);

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingExtension_Basic"
static PetscErrorCode PCBDDCScalingExtension_Basic(PC pc, Vec local_interface_vector, Vec global_vector)
{
  PC_IS*         pcis = (PC_IS*)pc->data;
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = VecSet(pcbddc->work_scaling,0.0);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  if (deluxe_ctx->n_simple) { /* scale deluxe vertices using diagonal scaling */
    PetscInt          i;
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
  /* sequential part : all problems and Schur applications collapsed into a single matrix vector multiplication or a matvec and a solve */
  if (deluxe_ctx->seq_mat) {
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,x,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,x,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatMultTranspose(deluxe_ctx->seq_mat,deluxe_ctx->seq_work1,deluxe_ctx->seq_work2);CHKERRQ(ierr);
    if (deluxe_ctx->seq_ksp) {
      ierr = KSPSolveTranspose(deluxe_ctx->seq_ksp,deluxe_ctx->seq_work2,deluxe_ctx->seq_work2);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work2,pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work2,pcbddc->work_scaling,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  /* put local boundary part in global vector */
  ierr = VecScatterBegin(pcis->global_to_B,pcbddc->work_scaling,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,pcbddc->work_scaling,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingExtension"
PetscErrorCode PCBDDCScalingExtension(PC pc, Vec local_interface_vector, Vec global_vector)
{
  PC_BDDC        *pcbddc=(PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(local_interface_vector,VEC_CLASSID,2);
  PetscValidHeaderSpecific(global_vector,VEC_CLASSID,3);
  if (local_interface_vector == pcbddc->work_scaling) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Local vector cannot be pcbddc->work_scaling!\n");
  ierr = PetscUseMethod(pc,"PCBDDCScalingExtension_C",(PC,Vec,Vec),(pc,local_interface_vector,global_vector));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingRestriction_Basic"
static PetscErrorCode PCBDDCScalingRestriction_Basic(PC pc, Vec global_vector, Vec local_interface_vector)
{
  PetscErrorCode ierr;
  PC_IS          *pcis = (PC_IS*)pc->data;

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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* get local boundary part of global vector */
  ierr = VecScatterBegin(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (deluxe_ctx->n_simple) { /* scale deluxe vertices using diagonal scaling */
    PetscInt          i;
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
  if (deluxe_ctx->seq_mat) {
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,y,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,y,deluxe_ctx->seq_work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (deluxe_ctx->seq_ksp) {
      ierr = KSPSolve(deluxe_ctx->seq_ksp,deluxe_ctx->seq_work1,deluxe_ctx->seq_work1);CHKERRQ(ierr);
    }
    ierr = MatMult(deluxe_ctx->seq_mat,deluxe_ctx->seq_work1,deluxe_ctx->seq_work2);CHKERRQ(ierr);
    ierr = VecScatterBegin(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work2,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(deluxe_ctx->seq_scctx,deluxe_ctx->seq_work2,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
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
  if (local_interface_vector == pcbddc->work_scaling) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Local vector cannot be pcbddc->work_scaling!\n");
  ierr = PetscUseMethod(pc,"PCBDDCScalingRestriction_C",(PC,Vec,Vec),(pc,global_vector,local_interface_vector));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingSetUp"
PetscErrorCode PCBDDCScalingSetUp(PC pc)
{
  PC_IS*         pcis=(PC_IS*)pc->data;
  PC_BDDC*       pcbddc=(PC_BDDC*)pc->data;
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
  ierr = VecScatterDestroy(&deluxe_ctx->seq_scctx);CHKERRQ(ierr);
  ierr = VecDestroy(&deluxe_ctx->seq_work1);CHKERRQ(ierr);
  ierr = VecDestroy(&deluxe_ctx->seq_work2);CHKERRQ(ierr);
  ierr = MatDestroy(&deluxe_ctx->seq_mat);CHKERRQ(ierr);
  ierr = KSPDestroy(&deluxe_ctx->seq_ksp);CHKERRQ(ierr);
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

  /* Compute data structures to solve sequential problems */
  ierr = PCBDDCScalingSetUp_Deluxe_Private(pc);CHKERRQ(ierr);

  /* diagonal scaling on interface dofs not contained in cc */
  if (sub_schurs->is_vertices || sub_schurs->is_dir) {
    PetscInt n_com,n_dir;
    n_com = 0;
    if (sub_schurs->is_vertices) {
      ierr = ISGetLocalSize(sub_schurs->is_vertices,&n_com);CHKERRQ(ierr);
    }
    n_dir = 0;
    if (sub_schurs->is_dir) {
      ierr = ISGetLocalSize(sub_schurs->is_dir,&n_dir);CHKERRQ(ierr);
    }
    deluxe_ctx->n_simple = n_dir + n_com;
    ierr = PetscMalloc1(deluxe_ctx->n_simple,&deluxe_ctx->idx_simple_B);CHKERRQ(ierr);
    if (sub_schurs->is_vertices) {
      PetscInt       nmap;
      const PetscInt *idxs;

      ierr = ISGetIndices(sub_schurs->is_vertices,&idxs);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,n_com,idxs,&nmap,deluxe_ctx->idx_simple_B);CHKERRQ(ierr);
      if (nmap != n_com) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error when mapping simply scaled dofs (is_vertices)! %D != %D",nmap,n_com);
      ierr = ISRestoreIndices(sub_schurs->is_vertices,&idxs);CHKERRQ(ierr);
    }
    if (sub_schurs->is_dir) {
      PetscInt       nmap;
      const PetscInt *idxs;

      ierr = ISGetIndices(sub_schurs->is_dir,&idxs);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,n_dir,idxs,&nmap,deluxe_ctx->idx_simple_B+n_com);CHKERRQ(ierr);
      if (nmap != n_dir) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error when mapping simply scaled dofs (sub_schurs->is_dir)! %D != %D",nmap,n_dir);
      ierr = ISRestoreIndices(sub_schurs->is_dir,&idxs);CHKERRQ(ierr);
    }
    ierr = PetscSortInt(deluxe_ctx->n_simple,deluxe_ctx->idx_simple_B);CHKERRQ(ierr);
  } else {
    deluxe_ctx->n_simple = 0;
    deluxe_ctx->idx_simple_B = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCScalingSetUp_Deluxe_Private"
static PetscErrorCode PCBDDCScalingSetUp_Deluxe_Private(PC pc)
{
  PC_BDDC                *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCDeluxeScaling    deluxe_ctx=pcbddc->deluxe_ctx;
  PCBDDCSubSchurs        sub_schurs = pcbddc->sub_schurs;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (!sub_schurs->n_subs) {
    PetscFunctionReturn(0);
  }

  /* Create work vectors for sequential part of deluxe */
  ierr = MatCreateVecs(sub_schurs->S_Ej_all,&deluxe_ctx->seq_work1,&deluxe_ctx->seq_work2);CHKERRQ(ierr);

  /* Compute deluxe sequential scatter */
  if (sub_schurs->reuse_mumps && !sub_schurs->is_dir) {
    PCBDDCReuseMumps reuse_mumps = sub_schurs->reuse_mumps;
    ierr = PetscObjectReference((PetscObject)reuse_mumps->correction_scatter_B);CHKERRQ(ierr);
    deluxe_ctx->seq_scctx = reuse_mumps->correction_scatter_B;
  } else {
    ierr = VecScatterCreate(pcbddc->work_scaling,sub_schurs->is_Ej_all,deluxe_ctx->seq_work1,NULL,&deluxe_ctx->seq_scctx);CHKERRQ(ierr);
  }

  /* Create Mat object for deluxe scaling */
  ierr = PetscObjectReference((PetscObject)sub_schurs->S_Ej_all);CHKERRQ(ierr);
  deluxe_ctx->seq_mat = sub_schurs->S_Ej_all;
  if (sub_schurs->sum_S_Ej_all) { /* if this matrix is present, then we need to create the KSP object to invert it */
    PC               pc_temp;
    MatSolverPackage solver=NULL;
    char             ksp_prefix[256];
    size_t           len;

    ierr = KSPCreate(PETSC_COMM_SELF,&deluxe_ctx->seq_ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(deluxe_ctx->seq_ksp,sub_schurs->sum_S_Ej_all,sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
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
    ierr = PetscStrcat(ksp_prefix,"deluxe_");CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(deluxe_ctx->seq_ksp,ksp_prefix);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(deluxe_ctx->seq_ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(deluxe_ctx->seq_ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
