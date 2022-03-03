
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,use,2);
  CHKERRQ(PetscTryMethod(pc,"PCISSetUseStiffnessScaling_C",(PC,PetscBool),(pc,use)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCISSetSubdomainDiagonalScaling_IS(PC pc, Vec scaling_factors)
{
  PC_IS          *pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)scaling_factors));
  CHKERRQ(VecDestroy(&pcis->D));
  pcis->D = scaling_factors;
  if (pc->setupcalled) {
    PetscInt sn;

    CHKERRQ(VecGetSize(pcis->D,&sn));
    if (sn == pcis->n) {
      CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecDestroy(&pcis->D));
      CHKERRQ(VecDuplicate(pcis->vec1_B,&pcis->D));
      CHKERRQ(VecCopy(pcis->vec1_B,pcis->D));
    } else PetscCheckFalse(sn != pcis->n_B,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Invalid size for scaling vector. Expected %D (or full %D), found %D",pcis->n_B,pcis->n,sn);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(scaling_factors,VEC_CLASSID,2);
  CHKERRQ(PetscTryMethod(pc,"PCISSetSubdomainDiagonalScaling_C",(PC,Vec),(pc,scaling_factors)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCISSetSubdomainScalingFactor_IS(PC pc, PetscScalar scal)
{
  PC_IS *pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  pcis->scaling_factor = scal;
  if (pcis->D) {

    CHKERRQ(VecSet(pcis->D,pcis->scaling_factor));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscTryMethod(pc,"PCISSetSubdomainScalingFactor_C",(PC,PetscScalar),(pc,scal)));
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc->pmat,MATIS,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Requires preconditioning matrix of type MATIS");
  matis = (Mat_IS*)pc->pmat->data;
  if (pc->useAmat) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc->mat,MATIS,&flg));
    PetscCheck(flg,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Requires linear system matrix of type MATIS");
  }

  /* first time creation, get info on substructuring */
  if (!pc->setupcalled) {
    PetscInt    n_I;
    PetscInt    *idx_I_local,*idx_B_local,*idx_I_global,*idx_B_global;
    PetscBT     bt;
    PetscInt    i,j;

    /* get info on mapping */
    CHKERRQ(PetscObjectReference((PetscObject)matis->rmapping));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&pcis->mapping));
    pcis->mapping = matis->rmapping;
    CHKERRQ(ISLocalToGlobalMappingGetSize(pcis->mapping,&pcis->n));
    CHKERRQ(ISLocalToGlobalMappingGetInfo(pcis->mapping,&(pcis->n_neigh),&(pcis->neigh),&(pcis->n_shared),&(pcis->shared)));

    /* Identifying interior and interface nodes, in local numbering */
    CHKERRQ(PetscBTCreate(pcis->n,&bt));
    for (i=0;i<pcis->n_neigh;i++)
      for (j=0;j<pcis->n_shared[i];j++) {
        CHKERRQ(PetscBTSet(bt,pcis->shared[i][j]));
      }

    /* Creating local and global index sets for interior and inteface nodes. */
    CHKERRQ(PetscMalloc1(pcis->n,&idx_I_local));
    CHKERRQ(PetscMalloc1(pcis->n,&idx_B_local));
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
    CHKERRQ(ISLocalToGlobalMappingApply(pcis->mapping,pcis->n_B,idx_B_local,idx_B_global));
    CHKERRQ(ISLocalToGlobalMappingApply(pcis->mapping,n_I,idx_I_local,idx_I_global));

    /* Creating the index sets */
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,pcis->n_B,idx_B_local,PETSC_COPY_VALUES, &pcis->is_B_local));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcis->n_B,idx_B_global,PETSC_COPY_VALUES,&pcis->is_B_global));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_local,PETSC_COPY_VALUES, &pcis->is_I_local));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),n_I,idx_I_global,PETSC_COPY_VALUES,&pcis->is_I_global));

    /* Freeing memory */
    CHKERRQ(PetscFree(idx_B_local));
    CHKERRQ(PetscFree(idx_I_local));
    CHKERRQ(PetscBTDestroy(&bt));

    /* Creating work vectors and arrays */
    CHKERRQ(VecDuplicate(matis->x,&pcis->vec1_N));
    CHKERRQ(VecDuplicate(pcis->vec1_N,&pcis->vec2_N));
    CHKERRQ(VecCreate(PETSC_COMM_SELF,&pcis->vec1_D));
    CHKERRQ(VecSetSizes(pcis->vec1_D,pcis->n-pcis->n_B,PETSC_DECIDE));
    CHKERRQ(VecSetType(pcis->vec1_D,((PetscObject)pcis->vec1_N)->type_name));
    CHKERRQ(VecDuplicate(pcis->vec1_D,&pcis->vec2_D));
    CHKERRQ(VecDuplicate(pcis->vec1_D,&pcis->vec3_D));
    CHKERRQ(VecDuplicate(pcis->vec1_D,&pcis->vec4_D));
    CHKERRQ(VecCreate(PETSC_COMM_SELF,&pcis->vec1_B));
    CHKERRQ(VecSetSizes(pcis->vec1_B,pcis->n_B,PETSC_DECIDE));
    CHKERRQ(VecSetType(pcis->vec1_B,((PetscObject)pcis->vec1_N)->type_name));
    CHKERRQ(VecDuplicate(pcis->vec1_B,&pcis->vec2_B));
    CHKERRQ(VecDuplicate(pcis->vec1_B,&pcis->vec3_B));
    CHKERRQ(MatCreateVecs(pc->pmat,&pcis->vec1_global,NULL));
    CHKERRQ(PetscMalloc1(pcis->n,&pcis->work_N));
    /* scaling vector */
    if (!pcis->D) { /* it can happen that the user passed in a scaling vector via PCISSetSubdomainDiagonalScaling */
      CHKERRQ(VecDuplicate(pcis->vec1_B,&pcis->D));
      CHKERRQ(VecSet(pcis->D,pcis->scaling_factor));
    }

    /* Creating the scatter contexts */
    CHKERRQ(VecScatterCreate(pcis->vec1_N,pcis->is_I_local,pcis->vec1_D,(IS)0,&pcis->N_to_D));
    CHKERRQ(VecScatterCreate(pcis->vec1_global,pcis->is_I_global,pcis->vec1_D,(IS)0,&pcis->global_to_D));
    CHKERRQ(VecScatterCreate(pcis->vec1_N,pcis->is_B_local,pcis->vec1_B,(IS)0,&pcis->N_to_B));
    CHKERRQ(VecScatterCreate(pcis->vec1_global,pcis->is_B_global,pcis->vec1_B,(IS)0,&pcis->global_to_B));

    /* map from boundary to local */
    CHKERRQ(ISLocalToGlobalMappingCreateIS(pcis->is_B_local,&pcis->BtoNmap));
  }

  {
    PetscInt sn;

    CHKERRQ(VecGetSize(pcis->D,&sn));
    if (sn == pcis->n) {
      CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecDestroy(&pcis->D));
      CHKERRQ(VecDuplicate(pcis->vec1_B,&pcis->D));
      CHKERRQ(VecCopy(pcis->vec1_B,pcis->D));
    } else PetscCheckFalse(sn != pcis->n_B,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Invalid size for scaling vector. Expected %D (or full %D), found %D",pcis->n_B,pcis->n,sn);
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
      CHKERRQ(MatDestroy(&pcis->A_II));
      CHKERRQ(MatDestroy(&pcis->pA_II));
      CHKERRQ(MatDestroy(&pcis->A_IB));
      CHKERRQ(MatDestroy(&pcis->A_BI));
      CHKERRQ(MatDestroy(&pcis->A_BB));
    }

    CHKERRQ(ISLocalToGlobalMappingGetBlockSize(pcis->mapping,&ibs));
    CHKERRQ(MatGetBlockSize(matis->A,&bs));
    CHKERRQ(MatCreateSubMatrix(matis->A,pcis->is_I_local,pcis->is_I_local,reuse,&pcis->pA_II));
    if (amat) {
      Mat_IS *amatis = (Mat_IS*)pc->mat->data;
      CHKERRQ(MatCreateSubMatrix(amatis->A,pcis->is_I_local,pcis->is_I_local,reuse,&pcis->A_II));
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)pcis->pA_II));
      CHKERRQ(MatDestroy(&pcis->A_II));
      pcis->A_II = pcis->pA_II;
    }
    CHKERRQ(MatSetBlockSize(pcis->A_II,bs == ibs ? bs : 1));
    CHKERRQ(MatSetBlockSize(pcis->pA_II,bs == ibs ? bs : 1));
    CHKERRQ(MatCreateSubMatrix(matis->A,pcis->is_B_local,pcis->is_B_local,reuse,&pcis->A_BB));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&issbaij));
    if (!issbaij) {
      CHKERRQ(MatCreateSubMatrix(matis->A,pcis->is_I_local,pcis->is_B_local,reuse,&pcis->A_IB));
      CHKERRQ(MatCreateSubMatrix(matis->A,pcis->is_B_local,pcis->is_I_local,reuse,&pcis->A_BI));
    } else {
      Mat newmat;

      CHKERRQ(MatConvert(matis->A,MATSEQBAIJ,MAT_INITIAL_MATRIX,&newmat));
      CHKERRQ(MatCreateSubMatrix(newmat,pcis->is_I_local,pcis->is_B_local,reuse,&pcis->A_IB));
      CHKERRQ(MatCreateSubMatrix(newmat,pcis->is_B_local,pcis->is_I_local,reuse,&pcis->A_BI));
      CHKERRQ(MatDestroy(&newmat));
    }
    CHKERRQ(MatSetBlockSize(pcis->A_BB,bs == ibs ? bs : 1));
  }

  /* Creating scaling vector D */
  CHKERRQ(PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_is_use_stiffness_scaling",&pcis->use_stiffness_scaling,NULL));
  if (pcis->use_stiffness_scaling) {
    PetscScalar *a;
    PetscInt    i,n;

    if (pcis->A_BB) {
      CHKERRQ(MatGetDiagonal(pcis->A_BB,pcis->D));
    } else {
      CHKERRQ(MatGetDiagonal(matis->A,pcis->vec1_N));
      CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD));
    }
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
  /* See historical note 01, at the bottom of this file. */

  /* Creating the KSP contexts for the local Dirichlet and Neumann problems */
  if (computesolvers) {
    PC pc_ctx;

    pcis->pure_neumann = matis->pure_neumann;
    /* Dirichlet */
    CHKERRQ(KSPCreate(PETSC_COMM_SELF,&pcis->ksp_D));
    CHKERRQ(KSPSetErrorIfNotConverged(pcis->ksp_D,pc->erroriffailure));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pcis->ksp_D,(PetscObject)pc,1));
    CHKERRQ(KSPSetOperators(pcis->ksp_D,pcis->A_II,pcis->A_II));
    CHKERRQ(KSPSetOptionsPrefix(pcis->ksp_D,"is_localD_"));
    CHKERRQ(KSPGetPC(pcis->ksp_D,&pc_ctx));
    CHKERRQ(PCSetType(pc_ctx,PCLU));
    CHKERRQ(KSPSetType(pcis->ksp_D,KSPPREONLY));
    CHKERRQ(KSPSetFromOptions(pcis->ksp_D));
    /* the vectors in the following line are dummy arguments, just telling the KSP the vector size. Values are not used */
    CHKERRQ(KSPSetUp(pcis->ksp_D));
    /* Neumann */
    CHKERRQ(KSPCreate(PETSC_COMM_SELF,&pcis->ksp_N));
    CHKERRQ(KSPSetErrorIfNotConverged(pcis->ksp_N,pc->erroriffailure));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pcis->ksp_N,(PetscObject)pc,1));
    CHKERRQ(KSPSetOperators(pcis->ksp_N,matis->A,matis->A));
    CHKERRQ(KSPSetOptionsPrefix(pcis->ksp_N,"is_localN_"));
    CHKERRQ(KSPGetPC(pcis->ksp_N,&pc_ctx));
    CHKERRQ(PCSetType(pc_ctx,PCLU));
    CHKERRQ(KSPSetType(pcis->ksp_N,KSPPREONLY));
    CHKERRQ(KSPSetFromOptions(pcis->ksp_N));
    {
      PetscBool damp_fixed                    = PETSC_FALSE,
                remove_nullspace_fixed        = PETSC_FALSE,
                set_damping_factor_floating   = PETSC_FALSE,
                not_damp_floating             = PETSC_FALSE,
                not_remove_nullspace_floating = PETSC_FALSE;
      PetscReal fixed_factor,
                floating_factor;

      CHKERRQ(PetscOptionsGetReal(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_damp_fixed",&fixed_factor,&damp_fixed));
      if (!damp_fixed) fixed_factor = 0.0;
      CHKERRQ(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_damp_fixed",&damp_fixed,NULL));

      CHKERRQ(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_remove_nullspace_fixed",&remove_nullspace_fixed,NULL));

      ierr = PetscOptionsGetReal(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_set_damping_factor_floating",
                              &floating_factor,&set_damping_factor_floating);CHKERRQ(ierr);
      if (!set_damping_factor_floating) floating_factor = 0.0;
      CHKERRQ(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_set_damping_factor_floating",&set_damping_factor_floating,NULL));
      if (!set_damping_factor_floating) floating_factor = 1.e-12;

      CHKERRQ(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_not_damp_floating",&not_damp_floating,NULL));

      CHKERRQ(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_not_remove_nullspace_floating",&not_remove_nullspace_floating,NULL));

      if (pcis->pure_neumann) {  /* floating subdomain */
        if (!(not_damp_floating)) {
          CHKERRQ(PCFactorSetShiftType(pc_ctx,MAT_SHIFT_NONZERO));
          CHKERRQ(PCFactorSetShiftAmount(pc_ctx,floating_factor));
        }
        if (!(not_remove_nullspace_floating)) {
          MatNullSpace nullsp;
          CHKERRQ(MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&nullsp));
          CHKERRQ(MatSetNullSpace(matis->A,nullsp));
          CHKERRQ(MatNullSpaceDestroy(&nullsp));
        }
      } else {  /* fixed subdomain */
        if (damp_fixed) {
          CHKERRQ(PCFactorSetShiftType(pc_ctx,MAT_SHIFT_NONZERO));
          CHKERRQ(PCFactorSetShiftAmount(pc_ctx,floating_factor));
        }
        if (remove_nullspace_fixed) {
          MatNullSpace nullsp;
          CHKERRQ(MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&nullsp));
          CHKERRQ(MatSetNullSpace(matis->A,nullsp));
          CHKERRQ(MatNullSpaceDestroy(&nullsp));
        }
      }
    }
    /* the vectors in the following line are dummy arguments, just telling the KSP the vector size. Values are not used */
    CHKERRQ(KSPSetUp(pcis->ksp_N));
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

  PetscFunctionBegin;
  CHKERRQ(ISDestroy(&pcis->is_B_local));
  CHKERRQ(ISDestroy(&pcis->is_I_local));
  CHKERRQ(ISDestroy(&pcis->is_B_global));
  CHKERRQ(ISDestroy(&pcis->is_I_global));
  CHKERRQ(MatDestroy(&pcis->A_II));
  CHKERRQ(MatDestroy(&pcis->pA_II));
  CHKERRQ(MatDestroy(&pcis->A_IB));
  CHKERRQ(MatDestroy(&pcis->A_BI));
  CHKERRQ(MatDestroy(&pcis->A_BB));
  CHKERRQ(VecDestroy(&pcis->D));
  CHKERRQ(KSPDestroy(&pcis->ksp_N));
  CHKERRQ(KSPDestroy(&pcis->ksp_D));
  CHKERRQ(VecDestroy(&pcis->vec1_N));
  CHKERRQ(VecDestroy(&pcis->vec2_N));
  CHKERRQ(VecDestroy(&pcis->vec1_D));
  CHKERRQ(VecDestroy(&pcis->vec2_D));
  CHKERRQ(VecDestroy(&pcis->vec3_D));
  CHKERRQ(VecDestroy(&pcis->vec4_D));
  CHKERRQ(VecDestroy(&pcis->vec1_B));
  CHKERRQ(VecDestroy(&pcis->vec2_B));
  CHKERRQ(VecDestroy(&pcis->vec3_B));
  CHKERRQ(VecDestroy(&pcis->vec1_global));
  CHKERRQ(VecScatterDestroy(&pcis->global_to_D));
  CHKERRQ(VecScatterDestroy(&pcis->N_to_B));
  CHKERRQ(VecScatterDestroy(&pcis->N_to_D));
  CHKERRQ(VecScatterDestroy(&pcis->global_to_B));
  CHKERRQ(PetscFree(pcis->work_N));
  if (pcis->n_neigh > -1) {
    CHKERRQ(ISLocalToGlobalMappingRestoreInfo(pcis->mapping,&(pcis->n_neigh),&(pcis->neigh),&(pcis->n_shared),&(pcis->shared)));
  }
  CHKERRQ(ISLocalToGlobalMappingDestroy(&pcis->mapping));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&pcis->BtoNmap));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCISSetUseStiffnessScaling_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainScalingFactor_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainDiagonalScaling_C",NULL));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISCreate -
*/
PetscErrorCode  PCISCreate(PC pc)
{
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  pcis->n_neigh          = -1;
  pcis->scaling_factor   = 1.0;
  pcis->reusesubmatrices = PETSC_TRUE;
  /* composing functions */
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCISSetUseStiffnessScaling_C",PCISSetUseStiffnessScaling_IS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainScalingFactor_C",PCISSetSubdomainScalingFactor_IS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainDiagonalScaling_C",PCISSetSubdomainDiagonalScaling_IS));
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
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  if (!vec2_B) vec2_B = v;

  CHKERRQ(MatMult(pcis->A_BB,v,vec1_B));
  CHKERRQ(MatMult(pcis->A_IB,v,vec1_D));
  CHKERRQ(KSPSolve(pcis->ksp_D,vec1_D,vec2_D));
  CHKERRQ(KSPCheckSolve(pcis->ksp_D,pc,vec2_D));
  CHKERRQ(MatMult(pcis->A_BI,vec2_D,vec2_B));
  CHKERRQ(VecAXPY(vec1_B,-1.0,vec2_B));
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
  PetscScalar    *array_B;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(v_B,&array_B));
  CHKERRQ(ISGetIndices(pcis->is_B_local,&idex));

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
  CHKERRQ(ISRestoreIndices(pcis->is_B_local,&idex));
  CHKERRQ(VecRestoreArray(v_B,&array_B));
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
  CHKERRQ(VecSet(vec1_N,0.0));
  CHKERRQ(VecScatterBegin(pcis->N_to_B,b,vec1_N,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd  (pcis->N_to_B,b,vec1_N,INSERT_VALUES,SCATTER_REVERSE));
  /* Checking for consistency of the RHS */
  {
    PetscBool flg = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-pc_is_check_consistency",&flg,NULL));
    if (flg) {
      PetscScalar average;
      PetscViewer viewer;
      CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pc),&viewer));

      CHKERRQ(VecSum(vec1_N,&average));
      average = average / ((PetscReal)pcis->n);
      CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
      if (pcis->pure_neumann) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d is floating. Average = % 1.14e\n",PetscGlobalRank,PetscAbsScalar(average)));
      } else {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d is fixed.    Average = % 1.14e\n",PetscGlobalRank,PetscAbsScalar(average)));
      }
      CHKERRQ(PetscViewerFlush(viewer));
      CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
    }
  }
  /* Solving the system for vec2_N */
  CHKERRQ(KSPSolve(pcis->ksp_N,vec1_N,vec2_N));
  CHKERRQ(KSPCheckSolve(pcis->ksp_N,pc,vec2_N));
  /* Extracting the local interface vector out of the solution */
  CHKERRQ(VecScatterBegin(pcis->N_to_B,vec2_N,x,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd  (pcis->N_to_B,vec2_N,x,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}
