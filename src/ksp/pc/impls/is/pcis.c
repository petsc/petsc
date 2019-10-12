
#include <../src/ksp/pc/impls/is/pcis.h> /*I "petscpc.h" I*/

static PetscErrorCode PCISSetUseStiffnessScaling_IS(PC pc, PetscBool use)
{
  PC_IS *pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  pcis->use_stiffness_scaling = use;
  PetscFunctionReturn(0);
}

/*@
 PCISSetUseStiffnessScaling - Tells PCIS to construct partition of unity using
                              local matrices' diagonal.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  use - whether or not pcis use matrix diagonal to build partition of unity.

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCISSetUseStiffnessScaling(PC pc, PetscBool use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,use,2);
  ierr = PetscTryMethod(pc,"PCISSetUseStiffnessScaling_C",(PC,PetscBool),(pc,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCISSetSubdomainDiagonalScaling_IS(PC pc, Vec scaling_factors)
{
  PetscErrorCode ierr;
  PC_IS          *pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  ierr    = PetscObjectReference((PetscObject)scaling_factors);CHKERRQ(ierr);
  ierr    = VecDestroy(&pcis->D);CHKERRQ(ierr);
  pcis->D = scaling_factors;
  if (pc->setupcalled) {
    PetscInt sn;

    ierr = VecGetSize(pcis->D,&sn);CHKERRQ(ierr);
    if (sn == pcis->n) {
      ierr = VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecDestroy(&pcis->D);CHKERRQ(ierr);
      ierr = VecDuplicate(pcis->vec1_B,&pcis->D);CHKERRQ(ierr);
      ierr = VecCopy(pcis->vec1_B,pcis->D);CHKERRQ(ierr);
    } else if (sn != pcis->n_B) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Invalid size for scaling vector. Expected %D (or full %D), found %D",pcis->n_B,pcis->n,sn); 
  }
  PetscFunctionReturn(0);
}

/*@
 PCISSetSubdomainDiagonalScaling - Set diagonal scaling for PCIS.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  scaling_factors - scaling factors for the subdomain

   Level: intermediate

   Notes:
   Intended to use with jumping coefficients cases.

.seealso: PCBDDC
@*/
PetscErrorCode PCISSetSubdomainDiagonalScaling(PC pc, Vec scaling_factors)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(scaling_factors,VEC_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCISSetSubdomainDiagonalScaling_C",(PC,Vec),(pc,scaling_factors));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCISSetSubdomainScalingFactor_IS(PC pc, PetscScalar scal)
{
  PC_IS *pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  pcis->scaling_factor = scal;
  if (pcis->D) {
    PetscErrorCode ierr;

    ierr = VecSet(pcis->D,pcis->scaling_factor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
 PCISSetSubdomainScalingFactor - Set scaling factor for PCIS.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  scal - scaling factor for the subdomain

   Level: intermediate

   Notes:
   Intended to use with jumping coefficients cases.

.seealso: PCBDDC
@*/
PetscErrorCode PCISSetSubdomainScalingFactor(PC pc, PetscScalar scal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCISSetSubdomainScalingFactor_C",(PC,PetscScalar),(pc,scal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCISSetUp -
*/
PetscErrorCode  PCISSetUp(PC pc, PetscBool computematrices, PetscBool computesolvers)
{
  PC_IS          *pcis  = (PC_IS*)(pc->data);
  Mat_IS         *matis;
  MatReuse       reuse;
  PetscErrorCode ierr;
  PetscBool      flg,issbaij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATIS,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Requires preconditioning matrix of type MATIS");
  matis = (Mat_IS*)pc->pmat->data;
  if (pc->useAmat) {
    ierr = PetscObjectTypeCompare((PetscObject)pc->mat,MATIS,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Requires linear system matrix of type MATIS");
  }

  /* first time creation, get info on substructuring */
  if (!pc->setupcalled) {
    PetscInt    n_I;
    PetscInt    *idx_I_local,*idx_B_local,*idx_I_global,*idx_B_global;
    PetscBT     bt;
    PetscInt    i,j;

    /* get info on mapping */
    ierr = PetscObjectReference((PetscObject)pc->pmat->rmap->mapping);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&pcis->mapping);CHKERRQ(ierr);
    pcis->mapping = pc->pmat->rmap->mapping;
    ierr = ISLocalToGlobalMappingGetSize(pcis->mapping,&pcis->n);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetInfo(pcis->mapping,&(pcis->n_neigh),&(pcis->neigh),&(pcis->n_shared),&(pcis->shared));CHKERRQ(ierr);

    /* Identifying interior and interface nodes, in local numbering */
    ierr = PetscBTCreate(pcis->n,&bt);CHKERRQ(ierr);
    for (i=0;i<pcis->n_neigh;i++)
      for (j=0;j<pcis->n_shared[i];j++) {
        ierr = PetscBTSet(bt,pcis->shared[i][j]);CHKERRQ(ierr);
      }

    /* Creating local and global index sets for interior and inteface nodes. */
    ierr = PetscMalloc1(pcis->n,&idx_I_local);CHKERRQ(ierr);
    ierr = PetscMalloc1(pcis->n,&idx_B_local);CHKERRQ(ierr);
    for (i=0, pcis->n_B=0, n_I=0; i<pcis->n; i++) {
      if (!PetscBTLookup(bt,i)) {
        idx_I_local[n_I] = i;
        n_I++;
      } else {
        idx_B_local[pcis->n_B] = i;
        pcis->n_B++;
      }
    }

    /* Getting the global numbering */
    idx_B_global = idx_I_local + n_I; /* Just avoiding allocating extra memory, since we have vacant space */
    idx_I_global = idx_B_local + pcis->n_B;
    ierr         = ISLocalToGlobalMappingApply(pcis->mapping,pcis->n_B,idx_B_local,idx_B_global);CHKERRQ(ierr);
    ierr         = ISLocalToGlobalMappingApply(pcis->mapping,n_I,idx_I_local,idx_I_global);CHKERRQ(ierr);

    /* Creating the index sets */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,pcis->n_B,idx_B_local,PETSC_COPY_VALUES, &pcis->is_B_local);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcis->n_B,idx_B_global,PETSC_COPY_VALUES,&pcis->is_B_global);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_local,PETSC_COPY_VALUES, &pcis->is_I_local);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),n_I,idx_I_global,PETSC_COPY_VALUES,&pcis->is_I_global);CHKERRQ(ierr);

    /* Freeing memory */
    ierr = PetscFree(idx_B_local);CHKERRQ(ierr);
    ierr = PetscFree(idx_I_local);CHKERRQ(ierr);
    ierr = PetscBTDestroy(&bt);CHKERRQ(ierr);

    /* Creating work vectors and arrays */
    ierr = VecDuplicate(matis->x,&pcis->vec1_N);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec1_N,&pcis->vec2_N);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF,&pcis->vec1_D);CHKERRQ(ierr);
    ierr = VecSetSizes(pcis->vec1_D,pcis->n-pcis->n_B,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(pcis->vec1_D,((PetscObject)pcis->vec1_N)->type_name);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec1_D,&pcis->vec2_D);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec1_D,&pcis->vec3_D);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec1_D,&pcis->vec4_D);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF,&pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecSetSizes(pcis->vec1_B,pcis->n_B,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetType(pcis->vec1_B,((PetscObject)pcis->vec1_N)->type_name);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec1_B,&pcis->vec2_B);CHKERRQ(ierr);
    ierr = VecDuplicate(pcis->vec1_B,&pcis->vec3_B);CHKERRQ(ierr);
    ierr = MatCreateVecs(pc->pmat,&pcis->vec1_global,0);CHKERRQ(ierr);
    ierr = PetscMalloc1(pcis->n,&pcis->work_N);CHKERRQ(ierr);
    /* scaling vector */
    if (!pcis->D) { /* it can happen that the user passed in a scaling vector via PCISSetSubdomainDiagonalScaling */
      ierr = VecDuplicate(pcis->vec1_B,&pcis->D);CHKERRQ(ierr);
      ierr = VecSet(pcis->D,pcis->scaling_factor);CHKERRQ(ierr);
    }

    /* Creating the scatter contexts */
    ierr = VecScatterCreate(pcis->vec1_N,pcis->is_I_local,pcis->vec1_D,(IS)0,&pcis->N_to_D);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcis->vec1_global,pcis->is_I_global,pcis->vec1_D,(IS)0,&pcis->global_to_D);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcis->vec1_N,pcis->is_B_local,pcis->vec1_B,(IS)0,&pcis->N_to_B);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcis->vec1_global,pcis->is_B_global,pcis->vec1_B,(IS)0,&pcis->global_to_B);CHKERRQ(ierr);

    /* map from boundary to local */
    ierr = ISLocalToGlobalMappingCreateIS(pcis->is_B_local,&pcis->BtoNmap);CHKERRQ(ierr);
  }

  {
    PetscInt sn;

    ierr = VecGetSize(pcis->D,&sn);CHKERRQ(ierr);
    if (sn == pcis->n) {
      ierr = VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecDestroy(&pcis->D);CHKERRQ(ierr);
      ierr = VecDuplicate(pcis->vec1_B,&pcis->D);CHKERRQ(ierr);
      ierr = VecCopy(pcis->vec1_B,pcis->D);CHKERRQ(ierr);
    } else if (sn != pcis->n_B) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Invalid size for scaling vector. Expected %D (or full %D), found %D",pcis->n_B,pcis->n,sn); 
  }

  /*
    Extracting the blocks A_II, A_BI, A_IB and A_BB from A. If the numbering
    is such that interior nodes come first than the interface ones, we have

        [ A_II | A_IB ]
    A = [------+------]
        [ A_BI | A_BB ]
  */
  if (computematrices) {
    PetscBool amat = (PetscBool)(pc->mat != pc->pmat && pc->useAmat);
    PetscInt  bs,ibs;

    reuse = MAT_INITIAL_MATRIX;
    if (pcis->reusesubmatrices && pc->setupcalled) {
      if (pc->flag == SAME_NONZERO_PATTERN) {
        reuse = MAT_REUSE_MATRIX;
      } else {
        reuse = MAT_INITIAL_MATRIX;
      }
    }
    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = MatDestroy(&pcis->A_II);CHKERRQ(ierr);
      ierr = MatDestroy(&pcis->pA_II);CHKERRQ(ierr);
      ierr = MatDestroy(&pcis->A_IB);CHKERRQ(ierr);
      ierr = MatDestroy(&pcis->A_BI);CHKERRQ(ierr);
      ierr = MatDestroy(&pcis->A_BB);CHKERRQ(ierr);
    }

    ierr = ISLocalToGlobalMappingGetBlockSize(pcis->mapping,&ibs);CHKERRQ(ierr);
    ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(matis->A,pcis->is_I_local,pcis->is_I_local,reuse,&pcis->pA_II);CHKERRQ(ierr);
    if (amat) {
      Mat_IS *amatis = (Mat_IS*)pc->mat->data;
      ierr = MatCreateSubMatrix(amatis->A,pcis->is_I_local,pcis->is_I_local,reuse,&pcis->A_II);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)pcis->pA_II);CHKERRQ(ierr);
      ierr = MatDestroy(&pcis->A_II);CHKERRQ(ierr);
      pcis->A_II = pcis->pA_II;
    }
    ierr = MatSetBlockSize(pcis->A_II,bs == ibs ? bs : 1);CHKERRQ(ierr);
    ierr = MatSetBlockSize(pcis->pA_II,bs == ibs ? bs : 1);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(matis->A,pcis->is_B_local,pcis->is_B_local,reuse,&pcis->A_BB);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
    if (!issbaij) {
      ierr = MatCreateSubMatrix(matis->A,pcis->is_I_local,pcis->is_B_local,reuse,&pcis->A_IB);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(matis->A,pcis->is_B_local,pcis->is_I_local,reuse,&pcis->A_BI);CHKERRQ(ierr);
    } else {
      Mat newmat;

      ierr = MatConvert(matis->A,MATSEQBAIJ,MAT_INITIAL_MATRIX,&newmat);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(newmat,pcis->is_I_local,pcis->is_B_local,reuse,&pcis->A_IB);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(newmat,pcis->is_B_local,pcis->is_I_local,reuse,&pcis->A_BI);CHKERRQ(ierr);
      ierr = MatDestroy(&newmat);CHKERRQ(ierr);
    }
    ierr = MatSetBlockSize(pcis->A_BB,bs == ibs ? bs : 1);CHKERRQ(ierr);
  }

  /* Creating scaling vector D */
  ierr = PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_is_use_stiffness_scaling",&pcis->use_stiffness_scaling,NULL);CHKERRQ(ierr);
  if (pcis->use_stiffness_scaling) {
    PetscScalar *a;
    PetscInt    i,n;

    if (pcis->A_BB) {
      ierr = MatGetDiagonal(pcis->A_BB,pcis->D);CHKERRQ(ierr);
    } else {
      ierr = MatGetDiagonal(matis->A,pcis->vec1_N);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
    ierr = VecAbs(pcis->D);CHKERRQ(ierr);
    ierr = VecGetLocalSize(pcis->D,&n);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->D,&a);CHKERRQ(ierr);
    for (i=0;i<n;i++) if (PetscAbsScalar(a[i])<PETSC_SMALL) a[i] = 1.0;
    ierr = VecRestoreArray(pcis->D,&a);CHKERRQ(ierr);
  }
  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcis->D,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,pcis->D,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(pcis->D,pcis->D,pcis->vec1_B);CHKERRQ(ierr);
  /* See historical note 01, at the bottom of this file. */

  /* Creating the KSP contexts for the local Dirichlet and Neumann problems */
  if (computesolvers) {
    PC pc_ctx;

    pcis->pure_neumann = matis->pure_neumann;
    /* Dirichlet */
    ierr = KSPCreate(PETSC_COMM_SELF,&pcis->ksp_D);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(pcis->ksp_D,pc->erroriffailure);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcis->ksp_D,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcis->ksp_D,pcis->A_II,pcis->A_II);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcis->ksp_D,"is_localD_");CHKERRQ(ierr);
    ierr = KSPGetPC(pcis->ksp_D,&pc_ctx);CHKERRQ(ierr);
    ierr = PCSetType(pc_ctx,PCLU);CHKERRQ(ierr);
    ierr = KSPSetType(pcis->ksp_D,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(pcis->ksp_D);CHKERRQ(ierr);
    /* the vectors in the following line are dummy arguments, just telling the KSP the vector size. Values are not used */
    ierr = KSPSetUp(pcis->ksp_D);CHKERRQ(ierr);
    /* Neumann */
    ierr = KSPCreate(PETSC_COMM_SELF,&pcis->ksp_N);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(pcis->ksp_N,pc->erroriffailure);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcis->ksp_N,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcis->ksp_N,matis->A,matis->A);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcis->ksp_N,"is_localN_");CHKERRQ(ierr);
    ierr = KSPGetPC(pcis->ksp_N,&pc_ctx);CHKERRQ(ierr);
    ierr = PCSetType(pc_ctx,PCLU);CHKERRQ(ierr);
    ierr = KSPSetType(pcis->ksp_N,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(pcis->ksp_N);CHKERRQ(ierr);
    {
      PetscBool damp_fixed                    = PETSC_FALSE,
                remove_nullspace_fixed        = PETSC_FALSE,
                set_damping_factor_floating   = PETSC_FALSE,
                not_damp_floating             = PETSC_FALSE,
                not_remove_nullspace_floating = PETSC_FALSE;
      PetscReal fixed_factor,
                floating_factor;

      ierr = PetscOptionsGetReal(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_damp_fixed",&fixed_factor,&damp_fixed);CHKERRQ(ierr);
      if (!damp_fixed) fixed_factor = 0.0;
      ierr = PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_damp_fixed",&damp_fixed,NULL);CHKERRQ(ierr);

      ierr = PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_remove_nullspace_fixed",&remove_nullspace_fixed,NULL);CHKERRQ(ierr);

      ierr = PetscOptionsGetReal(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_set_damping_factor_floating",
                              &floating_factor,&set_damping_factor_floating);CHKERRQ(ierr);
      if (!set_damping_factor_floating) floating_factor = 0.0;
      ierr = PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_set_damping_factor_floating",&set_damping_factor_floating,NULL);CHKERRQ(ierr);
      if (!set_damping_factor_floating) floating_factor = 1.e-12;

      ierr = PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_not_damp_floating",&not_damp_floating,NULL);CHKERRQ(ierr);

      ierr = PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_not_remove_nullspace_floating",&not_remove_nullspace_floating,NULL);CHKERRQ(ierr);

      if (pcis->pure_neumann) {  /* floating subdomain */
        if (!(not_damp_floating)) {
          ierr = PCFactorSetShiftType(pc_ctx,MAT_SHIFT_NONZERO);CHKERRQ(ierr);
          ierr = PCFactorSetShiftAmount(pc_ctx,floating_factor);CHKERRQ(ierr);
        }
        if (!(not_remove_nullspace_floating)) {
          MatNullSpace nullsp;
          ierr = MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&nullsp);CHKERRQ(ierr);
          ierr = MatSetNullSpace(matis->A,nullsp);CHKERRQ(ierr);
          ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
        }
      } else {  /* fixed subdomain */
        if (damp_fixed) {
          ierr = PCFactorSetShiftType(pc_ctx,MAT_SHIFT_NONZERO);CHKERRQ(ierr);
          ierr = PCFactorSetShiftAmount(pc_ctx,floating_factor);CHKERRQ(ierr);
        }
        if (remove_nullspace_fixed) {
          MatNullSpace nullsp;
          ierr = MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&nullsp);CHKERRQ(ierr);
          ierr = MatSetNullSpace(matis->A,nullsp);CHKERRQ(ierr);
          ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
        }
      }
    }
    /* the vectors in the following line are dummy arguments, just telling the KSP the vector size. Values are not used */
    ierr = KSPSetUp(pcis->ksp_N);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISDestroy -
*/
PetscErrorCode  PCISDestroy(PC pc)
{
  PC_IS          *pcis = (PC_IS*)(pc->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&pcis->is_B_local);CHKERRQ(ierr);
  ierr = ISDestroy(&pcis->is_I_local);CHKERRQ(ierr);
  ierr = ISDestroy(&pcis->is_B_global);CHKERRQ(ierr);
  ierr = ISDestroy(&pcis->is_I_global);CHKERRQ(ierr);
  ierr = MatDestroy(&pcis->A_II);CHKERRQ(ierr);
  ierr = MatDestroy(&pcis->pA_II);CHKERRQ(ierr);
  ierr = MatDestroy(&pcis->A_IB);CHKERRQ(ierr);
  ierr = MatDestroy(&pcis->A_BI);CHKERRQ(ierr);
  ierr = MatDestroy(&pcis->A_BB);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->D);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcis->ksp_N);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcis->ksp_D);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec1_N);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec2_N);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec1_D);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec2_D);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec3_D);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec4_D);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec1_B);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec2_B);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec3_B);CHKERRQ(ierr);
  ierr = VecDestroy(&pcis->vec1_global);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcis->global_to_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcis->N_to_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcis->N_to_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcis->global_to_B);CHKERRQ(ierr);
  ierr = PetscFree(pcis->work_N);CHKERRQ(ierr);
  if (pcis->n_neigh > -1) {
    ierr = ISLocalToGlobalMappingRestoreInfo(pcis->mapping,&(pcis->n_neigh),&(pcis->neigh),&(pcis->n_shared),&(pcis->shared));CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&pcis->mapping);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&pcis->BtoNmap);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCISSetUseStiffnessScaling_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainScalingFactor_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainDiagonalScaling_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISCreate -
*/
PetscErrorCode  PCISCreate(PC pc)
{
  PC_IS          *pcis = (PC_IS*)(pc->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pcis->n_neigh          = -1;
  pcis->scaling_factor   = 1.0;
  pcis->reusesubmatrices = PETSC_TRUE;
  /* composing functions */
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCISSetUseStiffnessScaling_C",PCISSetUseStiffnessScaling_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainScalingFactor_C",PCISSetSubdomainScalingFactor_IS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainDiagonalScaling_C",PCISSetSubdomainDiagonalScaling_IS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISApplySchur -

   Input parameters:
.  pc - preconditioner context
.  v - vector to which the Schur complement is to be applied (it is NOT modified inside this function, UNLESS vec2_B is null)

   Output parameters:
.  vec1_B - result of Schur complement applied to chunk
.  vec2_B - garbage (used as work space), or null (and v is used as workspace)
.  vec1_D - garbage (used as work space)
.  vec2_D - garbage (used as work space)

*/
PetscErrorCode  PCISApplySchur(PC pc, Vec v, Vec vec1_B, Vec vec2_B, Vec vec1_D, Vec vec2_D)
{
  PetscErrorCode ierr;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  if (!vec2_B) vec2_B = v;

  ierr = MatMult(pcis->A_BB,v,vec1_B);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_IB,v,vec1_D);CHKERRQ(ierr);
  ierr = KSPSolve(pcis->ksp_D,vec1_D,vec2_D);CHKERRQ(ierr);
  ierr = KSPCheckSolve(pcis->ksp_D,pc,vec2_D);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_BI,vec2_D,vec2_B);CHKERRQ(ierr);
  ierr = VecAXPY(vec1_B,-1.0,vec2_B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISScatterArrayNToVecB - Scatters interface node values from a big array (of all local nodes, interior or interface,
   including ghosts) into an interface vector, when in SCATTER_FORWARD mode, or vice-versa, when in SCATTER_REVERSE
   mode.

   Input parameters:
.  pc - preconditioner context
.  array_N - [when in SCATTER_FORWARD mode] Array to be scattered into the vector
.  v_B - [when in SCATTER_REVERSE mode] Vector to be scattered into the array

   Output parameter:
.  array_N - [when in SCATTER_REVERSE mode] Array to receive the scattered vector
.  v_B - [when in SCATTER_FORWARD mode] Vector to receive the scattered array

   Notes:
   The entries in the array that do not correspond to interface nodes remain unaltered.
*/
PetscErrorCode  PCISScatterArrayNToVecB(PetscScalar *array_N, Vec v_B, InsertMode imode, ScatterMode smode, PC pc)
{
  PetscInt       i;
  const PetscInt *idex;
  PetscErrorCode ierr;
  PetscScalar    *array_B;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  ierr = VecGetArray(v_B,&array_B);CHKERRQ(ierr);
  ierr = ISGetIndices(pcis->is_B_local,&idex);CHKERRQ(ierr);

  if (smode == SCATTER_FORWARD) {
    if (imode == INSERT_VALUES) {
      for (i=0; i<pcis->n_B; i++) array_B[i] = array_N[idex[i]];
    } else {  /* ADD_VALUES */
      for (i=0; i<pcis->n_B; i++) array_B[i] += array_N[idex[i]];
    }
  } else {  /* SCATTER_REVERSE */
    if (imode == INSERT_VALUES) {
      for (i=0; i<pcis->n_B; i++) array_N[idex[i]] = array_B[i];
    } else {  /* ADD_VALUES */
      for (i=0; i<pcis->n_B; i++) array_N[idex[i]] += array_B[i];
    }
  }
  ierr = ISRestoreIndices(pcis->is_B_local,&idex);CHKERRQ(ierr);
  ierr = VecRestoreArray(v_B,&array_B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISApplyInvSchur - Solves the Neumann problem related to applying the inverse of the Schur complement.
   More precisely, solves the problem:
                                        [ A_II  A_IB ] [ . ]   [ 0 ]
                                        [            ] [   ] = [   ]
                                        [ A_BI  A_BB ] [ x ]   [ b ]

   Input parameters:
.  pc - preconditioner context
.  b - vector of local interface nodes (including ghosts)

   Output parameters:
.  x - vector of local interface nodes (including ghosts); returns the application of the inverse of the Schur
       complement to b
.  vec1_N - vector of local nodes (interior and interface, including ghosts); returns garbage (used as work space)
.  vec2_N - vector of local nodes (interior and interface, including ghosts); returns garbage (used as work space)

*/
PetscErrorCode  PCISApplyInvSchur(PC pc, Vec b, Vec x, Vec vec1_N, Vec vec2_N)
{
  PetscErrorCode ierr;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  /*
    Neumann solvers.
    Applying the inverse of the local Schur complement, i.e, solving a Neumann
    Problem with zero at the interior nodes of the RHS and extracting the interface
    part of the solution. inverse Schur complement is applied to b and the result
    is stored in x.
  */
  /* Setting the RHS vec1_N */
  ierr = VecSet(vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->N_to_B,b,vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->N_to_B,b,vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* Checking for consistency of the RHS */
  {
    PetscBool flg = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-pc_is_check_consistency",&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      PetscScalar average;
      PetscViewer viewer;
      ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pc),&viewer);CHKERRQ(ierr);

      ierr    = VecSum(vec1_N,&average);CHKERRQ(ierr);
      average = average / ((PetscReal)pcis->n);
      ierr    = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      if (pcis->pure_neumann) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d is floating. Average = % 1.14e\n",PetscGlobalRank,PetscAbsScalar(average));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d is fixed.    Average = % 1.14e\n",PetscGlobalRank,PetscAbsScalar(average));CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    }
  }
  /* Solving the system for vec2_N */
  ierr = KSPSolve(pcis->ksp_N,vec1_N,vec2_N);CHKERRQ(ierr);
  ierr = KSPCheckSolve(pcis->ksp_N,pc,vec2_N);CHKERRQ(ierr);
  /* Extracting the local interface vector out of the solution */
  ierr = VecScatterBegin(pcis->N_to_B,vec2_N,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->N_to_B,vec2_N,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
