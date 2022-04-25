
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

.seealso: `PCBDDC`
@*/
PetscErrorCode PCISSetUseStiffnessScaling(PC pc, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,use,2);
  PetscTryMethod(pc,"PCISSetUseStiffnessScaling_C",(PC,PetscBool),(pc,use));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCISSetSubdomainDiagonalScaling_IS(PC pc, Vec scaling_factors)
{
  PC_IS          *pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)scaling_factors));
  PetscCall(VecDestroy(&pcis->D));
  pcis->D = scaling_factors;
  if (pc->setupcalled) {
    PetscInt sn;

    PetscCall(VecGetSize(pcis->D,&sn));
    if (sn == pcis->n) {
      PetscCall(VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecDestroy(&pcis->D));
      PetscCall(VecDuplicate(pcis->vec1_B,&pcis->D));
      PetscCall(VecCopy(pcis->vec1_B,pcis->D));
    } else PetscCheck(sn == pcis->n_B,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Invalid size for scaling vector. Expected %" PetscInt_FMT " (or full %" PetscInt_FMT "), found %" PetscInt_FMT,pcis->n_B,pcis->n,sn);
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

.seealso: `PCBDDC`
@*/
PetscErrorCode PCISSetSubdomainDiagonalScaling(PC pc, Vec scaling_factors)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(scaling_factors,VEC_CLASSID,2);
  PetscTryMethod(pc,"PCISSetSubdomainDiagonalScaling_C",(PC,Vec),(pc,scaling_factors));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCISSetSubdomainScalingFactor_IS(PC pc, PetscScalar scal)
{
  PC_IS *pcis = (PC_IS*)pc->data;

  PetscFunctionBegin;
  pcis->scaling_factor = scal;
  if (pcis->D) {

    PetscCall(VecSet(pcis->D,pcis->scaling_factor));
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

.seealso: `PCBDDC`
@*/
PetscErrorCode PCISSetSubdomainScalingFactor(PC pc, PetscScalar scal)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCISSetSubdomainScalingFactor_C",(PC,PetscScalar),(pc,scal));
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
  PetscBool      flg,issbaij;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat,MATIS,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Requires preconditioning matrix of type MATIS");
  matis = (Mat_IS*)pc->pmat->data;
  if (pc->useAmat) {
    PetscCall(PetscObjectTypeCompare((PetscObject)pc->mat,MATIS,&flg));
    PetscCheck(flg,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Requires linear system matrix of type MATIS");
  }

  /* first time creation, get info on substructuring */
  if (!pc->setupcalled) {
    PetscInt    n_I;
    PetscInt    *idx_I_local,*idx_B_local,*idx_I_global,*idx_B_global;
    PetscBT     bt;
    PetscInt    i,j;

    /* get info on mapping */
    PetscCall(PetscObjectReference((PetscObject)matis->rmapping));
    PetscCall(ISLocalToGlobalMappingDestroy(&pcis->mapping));
    pcis->mapping = matis->rmapping;
    PetscCall(ISLocalToGlobalMappingGetSize(pcis->mapping,&pcis->n));
    PetscCall(ISLocalToGlobalMappingGetInfo(pcis->mapping,&(pcis->n_neigh),&(pcis->neigh),&(pcis->n_shared),&(pcis->shared)));

    /* Identifying interior and interface nodes, in local numbering */
    PetscCall(PetscBTCreate(pcis->n,&bt));
    for (i=0;i<pcis->n_neigh;i++)
      for (j=0;j<pcis->n_shared[i];j++) {
        PetscCall(PetscBTSet(bt,pcis->shared[i][j]));
      }

    /* Creating local and global index sets for interior and inteface nodes. */
    PetscCall(PetscMalloc1(pcis->n,&idx_I_local));
    PetscCall(PetscMalloc1(pcis->n,&idx_B_local));
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
    PetscCall(ISLocalToGlobalMappingApply(pcis->mapping,pcis->n_B,idx_B_local,idx_B_global));
    PetscCall(ISLocalToGlobalMappingApply(pcis->mapping,n_I,idx_I_local,idx_I_global));

    /* Creating the index sets */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,pcis->n_B,idx_B_local,PETSC_COPY_VALUES, &pcis->is_B_local));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcis->n_B,idx_B_global,PETSC_COPY_VALUES,&pcis->is_B_global));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_local,PETSC_COPY_VALUES, &pcis->is_I_local));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),n_I,idx_I_global,PETSC_COPY_VALUES,&pcis->is_I_global));

    /* Freeing memory */
    PetscCall(PetscFree(idx_B_local));
    PetscCall(PetscFree(idx_I_local));
    PetscCall(PetscBTDestroy(&bt));

    /* Creating work vectors and arrays */
    PetscCall(VecDuplicate(matis->x,&pcis->vec1_N));
    PetscCall(VecDuplicate(pcis->vec1_N,&pcis->vec2_N));
    PetscCall(VecCreate(PETSC_COMM_SELF,&pcis->vec1_D));
    PetscCall(VecSetSizes(pcis->vec1_D,pcis->n-pcis->n_B,PETSC_DECIDE));
    PetscCall(VecSetType(pcis->vec1_D,((PetscObject)pcis->vec1_N)->type_name));
    PetscCall(VecDuplicate(pcis->vec1_D,&pcis->vec2_D));
    PetscCall(VecDuplicate(pcis->vec1_D,&pcis->vec3_D));
    PetscCall(VecDuplicate(pcis->vec1_D,&pcis->vec4_D));
    PetscCall(VecCreate(PETSC_COMM_SELF,&pcis->vec1_B));
    PetscCall(VecSetSizes(pcis->vec1_B,pcis->n_B,PETSC_DECIDE));
    PetscCall(VecSetType(pcis->vec1_B,((PetscObject)pcis->vec1_N)->type_name));
    PetscCall(VecDuplicate(pcis->vec1_B,&pcis->vec2_B));
    PetscCall(VecDuplicate(pcis->vec1_B,&pcis->vec3_B));
    PetscCall(MatCreateVecs(pc->pmat,&pcis->vec1_global,NULL));
    PetscCall(PetscMalloc1(pcis->n,&pcis->work_N));
    /* scaling vector */
    if (!pcis->D) { /* it can happen that the user passed in a scaling vector via PCISSetSubdomainDiagonalScaling */
      PetscCall(VecDuplicate(pcis->vec1_B,&pcis->D));
      PetscCall(VecSet(pcis->D,pcis->scaling_factor));
    }

    /* Creating the scatter contexts */
    PetscCall(VecScatterCreate(pcis->vec1_N,pcis->is_I_local,pcis->vec1_D,(IS)0,&pcis->N_to_D));
    PetscCall(VecScatterCreate(pcis->vec1_global,pcis->is_I_global,pcis->vec1_D,(IS)0,&pcis->global_to_D));
    PetscCall(VecScatterCreate(pcis->vec1_N,pcis->is_B_local,pcis->vec1_B,(IS)0,&pcis->N_to_B));
    PetscCall(VecScatterCreate(pcis->vec1_global,pcis->is_B_global,pcis->vec1_B,(IS)0,&pcis->global_to_B));

    /* map from boundary to local */
    PetscCall(ISLocalToGlobalMappingCreateIS(pcis->is_B_local,&pcis->BtoNmap));
  }

  {
    PetscInt sn;

    PetscCall(VecGetSize(pcis->D,&sn));
    if (sn == pcis->n) {
      PetscCall(VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecDestroy(&pcis->D));
      PetscCall(VecDuplicate(pcis->vec1_B,&pcis->D));
      PetscCall(VecCopy(pcis->vec1_B,pcis->D));
    } else PetscCheck(sn == pcis->n_B,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Invalid size for scaling vector. Expected %" PetscInt_FMT " (or full %" PetscInt_FMT "), found %" PetscInt_FMT,pcis->n_B,pcis->n,sn);
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
      PetscCall(MatDestroy(&pcis->A_II));
      PetscCall(MatDestroy(&pcis->pA_II));
      PetscCall(MatDestroy(&pcis->A_IB));
      PetscCall(MatDestroy(&pcis->A_BI));
      PetscCall(MatDestroy(&pcis->A_BB));
    }

    PetscCall(ISLocalToGlobalMappingGetBlockSize(pcis->mapping,&ibs));
    PetscCall(MatGetBlockSize(matis->A,&bs));
    PetscCall(MatCreateSubMatrix(matis->A,pcis->is_I_local,pcis->is_I_local,reuse,&pcis->pA_II));
    if (amat) {
      Mat_IS *amatis = (Mat_IS*)pc->mat->data;
      PetscCall(MatCreateSubMatrix(amatis->A,pcis->is_I_local,pcis->is_I_local,reuse,&pcis->A_II));
    } else {
      PetscCall(PetscObjectReference((PetscObject)pcis->pA_II));
      PetscCall(MatDestroy(&pcis->A_II));
      pcis->A_II = pcis->pA_II;
    }
    PetscCall(MatSetBlockSize(pcis->A_II,bs == ibs ? bs : 1));
    PetscCall(MatSetBlockSize(pcis->pA_II,bs == ibs ? bs : 1));
    PetscCall(MatCreateSubMatrix(matis->A,pcis->is_B_local,pcis->is_B_local,reuse,&pcis->A_BB));
    PetscCall(PetscObjectTypeCompare((PetscObject)matis->A,MATSEQSBAIJ,&issbaij));
    if (!issbaij) {
      PetscCall(MatCreateSubMatrix(matis->A,pcis->is_I_local,pcis->is_B_local,reuse,&pcis->A_IB));
      PetscCall(MatCreateSubMatrix(matis->A,pcis->is_B_local,pcis->is_I_local,reuse,&pcis->A_BI));
    } else {
      Mat newmat;

      PetscCall(MatConvert(matis->A,MATSEQBAIJ,MAT_INITIAL_MATRIX,&newmat));
      PetscCall(MatCreateSubMatrix(newmat,pcis->is_I_local,pcis->is_B_local,reuse,&pcis->A_IB));
      PetscCall(MatCreateSubMatrix(newmat,pcis->is_B_local,pcis->is_I_local,reuse,&pcis->A_BI));
      PetscCall(MatDestroy(&newmat));
    }
    PetscCall(MatSetBlockSize(pcis->A_BB,bs == ibs ? bs : 1));
  }

  /* Creating scaling vector D */
  PetscCall(PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_is_use_stiffness_scaling",&pcis->use_stiffness_scaling,NULL));
  if (pcis->use_stiffness_scaling) {
    PetscScalar *a;
    PetscInt    i,n;

    if (pcis->A_BB) {
      PetscCall(MatGetDiagonal(pcis->A_BB,pcis->D));
    } else {
      PetscCall(MatGetDiagonal(matis->A,pcis->vec1_N));
      PetscCall(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD));
    }
    PetscCall(VecAbs(pcis->D));
    PetscCall(VecGetLocalSize(pcis->D,&n));
    PetscCall(VecGetArray(pcis->D,&a));
    for (i=0;i<n;i++) if (PetscAbsScalar(a[i])<PETSC_SMALL) a[i] = 1.0;
    PetscCall(VecRestoreArray(pcis->D,&a));
  }
  PetscCall(VecSet(pcis->vec1_global,0.0));
  PetscCall(VecScatterBegin(pcis->global_to_B,pcis->D,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(pcis->global_to_B,pcis->D,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterBegin(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_B,pcis->vec1_global,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecPointwiseDivide(pcis->D,pcis->D,pcis->vec1_B));
  /* See historical note 01, at the bottom of this file. */

  /* Creating the KSP contexts for the local Dirichlet and Neumann problems */
  if (computesolvers) {
    PC pc_ctx;

    pcis->pure_neumann = matis->pure_neumann;
    /* Dirichlet */
    PetscCall(KSPCreate(PETSC_COMM_SELF,&pcis->ksp_D));
    PetscCall(KSPSetErrorIfNotConverged(pcis->ksp_D,pc->erroriffailure));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)pcis->ksp_D,(PetscObject)pc,1));
    PetscCall(KSPSetOperators(pcis->ksp_D,pcis->A_II,pcis->A_II));
    PetscCall(KSPSetOptionsPrefix(pcis->ksp_D,"is_localD_"));
    PetscCall(KSPGetPC(pcis->ksp_D,&pc_ctx));
    PetscCall(PCSetType(pc_ctx,PCLU));
    PetscCall(KSPSetType(pcis->ksp_D,KSPPREONLY));
    PetscCall(KSPSetFromOptions(pcis->ksp_D));
    /* the vectors in the following line are dummy arguments, just telling the KSP the vector size. Values are not used */
    PetscCall(KSPSetUp(pcis->ksp_D));
    /* Neumann */
    PetscCall(KSPCreate(PETSC_COMM_SELF,&pcis->ksp_N));
    PetscCall(KSPSetErrorIfNotConverged(pcis->ksp_N,pc->erroriffailure));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)pcis->ksp_N,(PetscObject)pc,1));
    PetscCall(KSPSetOperators(pcis->ksp_N,matis->A,matis->A));
    PetscCall(KSPSetOptionsPrefix(pcis->ksp_N,"is_localN_"));
    PetscCall(KSPGetPC(pcis->ksp_N,&pc_ctx));
    PetscCall(PCSetType(pc_ctx,PCLU));
    PetscCall(KSPSetType(pcis->ksp_N,KSPPREONLY));
    PetscCall(KSPSetFromOptions(pcis->ksp_N));
    {
      PetscBool damp_fixed                    = PETSC_FALSE,
                remove_nullspace_fixed        = PETSC_FALSE,
                set_damping_factor_floating   = PETSC_FALSE,
                not_damp_floating             = PETSC_FALSE,
                not_remove_nullspace_floating = PETSC_FALSE;
      PetscReal fixed_factor,
                floating_factor;

      PetscCall(PetscOptionsGetReal(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_damp_fixed",&fixed_factor,&damp_fixed));
      if (!damp_fixed) fixed_factor = 0.0;
      PetscCall(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_damp_fixed",&damp_fixed,NULL));

      PetscCall(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_remove_nullspace_fixed",&remove_nullspace_fixed,NULL));

      PetscCall(PetscOptionsGetReal(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_set_damping_factor_floating",&floating_factor,&set_damping_factor_floating));
      if (!set_damping_factor_floating) floating_factor = 0.0;
      PetscCall(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_set_damping_factor_floating",&set_damping_factor_floating,NULL));
      if (!set_damping_factor_floating) floating_factor = 1.e-12;

      PetscCall(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_not_damp_floating",&not_damp_floating,NULL));

      PetscCall(PetscOptionsGetBool(((PetscObject)pc_ctx)->options,((PetscObject)pc_ctx)->prefix,"-pc_is_not_remove_nullspace_floating",&not_remove_nullspace_floating,NULL));

      if (pcis->pure_neumann) {  /* floating subdomain */
        if (!(not_damp_floating)) {
          PetscCall(PCFactorSetShiftType(pc_ctx,MAT_SHIFT_NONZERO));
          PetscCall(PCFactorSetShiftAmount(pc_ctx,floating_factor));
        }
        if (!(not_remove_nullspace_floating)) {
          MatNullSpace nullsp;
          PetscCall(MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&nullsp));
          PetscCall(MatSetNullSpace(matis->A,nullsp));
          PetscCall(MatNullSpaceDestroy(&nullsp));
        }
      } else {  /* fixed subdomain */
        if (damp_fixed) {
          PetscCall(PCFactorSetShiftType(pc_ctx,MAT_SHIFT_NONZERO));
          PetscCall(PCFactorSetShiftAmount(pc_ctx,floating_factor));
        }
        if (remove_nullspace_fixed) {
          MatNullSpace nullsp;
          PetscCall(MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&nullsp));
          PetscCall(MatSetNullSpace(matis->A,nullsp));
          PetscCall(MatNullSpaceDestroy(&nullsp));
        }
      }
    }
    /* the vectors in the following line are dummy arguments, just telling the KSP the vector size. Values are not used */
    PetscCall(KSPSetUp(pcis->ksp_N));
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
  PetscCall(ISDestroy(&pcis->is_B_local));
  PetscCall(ISDestroy(&pcis->is_I_local));
  PetscCall(ISDestroy(&pcis->is_B_global));
  PetscCall(ISDestroy(&pcis->is_I_global));
  PetscCall(MatDestroy(&pcis->A_II));
  PetscCall(MatDestroy(&pcis->pA_II));
  PetscCall(MatDestroy(&pcis->A_IB));
  PetscCall(MatDestroy(&pcis->A_BI));
  PetscCall(MatDestroy(&pcis->A_BB));
  PetscCall(VecDestroy(&pcis->D));
  PetscCall(KSPDestroy(&pcis->ksp_N));
  PetscCall(KSPDestroy(&pcis->ksp_D));
  PetscCall(VecDestroy(&pcis->vec1_N));
  PetscCall(VecDestroy(&pcis->vec2_N));
  PetscCall(VecDestroy(&pcis->vec1_D));
  PetscCall(VecDestroy(&pcis->vec2_D));
  PetscCall(VecDestroy(&pcis->vec3_D));
  PetscCall(VecDestroy(&pcis->vec4_D));
  PetscCall(VecDestroy(&pcis->vec1_B));
  PetscCall(VecDestroy(&pcis->vec2_B));
  PetscCall(VecDestroy(&pcis->vec3_B));
  PetscCall(VecDestroy(&pcis->vec1_global));
  PetscCall(VecScatterDestroy(&pcis->global_to_D));
  PetscCall(VecScatterDestroy(&pcis->N_to_B));
  PetscCall(VecScatterDestroy(&pcis->N_to_D));
  PetscCall(VecScatterDestroy(&pcis->global_to_B));
  PetscCall(PetscFree(pcis->work_N));
  if (pcis->n_neigh > -1) {
    PetscCall(ISLocalToGlobalMappingRestoreInfo(pcis->mapping,&(pcis->n_neigh),&(pcis->neigh),&(pcis->n_shared),&(pcis->shared)));
  }
  PetscCall(ISLocalToGlobalMappingDestroy(&pcis->mapping));
  PetscCall(ISLocalToGlobalMappingDestroy(&pcis->BtoNmap));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCISSetUseStiffnessScaling_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainScalingFactor_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainDiagonalScaling_C",NULL));
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
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCISSetUseStiffnessScaling_C",PCISSetUseStiffnessScaling_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainScalingFactor_C",PCISSetSubdomainScalingFactor_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCISSetSubdomainDiagonalScaling_C",PCISSetSubdomainDiagonalScaling_IS));
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

  PetscCall(MatMult(pcis->A_BB,v,vec1_B));
  PetscCall(MatMult(pcis->A_IB,v,vec1_D));
  PetscCall(KSPSolve(pcis->ksp_D,vec1_D,vec2_D));
  PetscCall(KSPCheckSolve(pcis->ksp_D,pc,vec2_D));
  PetscCall(MatMult(pcis->A_BI,vec2_D,vec2_B));
  PetscCall(VecAXPY(vec1_B,-1.0,vec2_B));
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
  PetscCall(VecGetArray(v_B,&array_B));
  PetscCall(ISGetIndices(pcis->is_B_local,&idex));

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
  PetscCall(ISRestoreIndices(pcis->is_B_local,&idex));
  PetscCall(VecRestoreArray(v_B,&array_B));
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
  PetscCall(VecSet(vec1_N,0.0));
  PetscCall(VecScatterBegin(pcis->N_to_B,b,vec1_N,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd  (pcis->N_to_B,b,vec1_N,INSERT_VALUES,SCATTER_REVERSE));
  /* Checking for consistency of the RHS */
  {
    PetscBool flg = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-pc_is_check_consistency",&flg,NULL));
    if (flg) {
      PetscScalar average;
      PetscViewer viewer;
      PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pc),&viewer));

      PetscCall(VecSum(vec1_N,&average));
      average = average / ((PetscReal)pcis->n);
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      if (pcis->pure_neumann) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d is floating. Average = % 1.14e\n",PetscGlobalRank,(double)PetscAbsScalar(average)));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d is fixed.    Average = % 1.14e\n",PetscGlobalRank,(double)PetscAbsScalar(average)));
      }
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    }
  }
  /* Solving the system for vec2_N */
  PetscCall(KSPSolve(pcis->ksp_N,vec1_N,vec2_N));
  PetscCall(KSPCheckSolve(pcis->ksp_N,pc,vec2_N));
  /* Extracting the local interface vector out of the solution */
  PetscCall(VecScatterBegin(pcis->N_to_B,vec2_N,x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd  (pcis->N_to_B,vec2_N,x,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}
