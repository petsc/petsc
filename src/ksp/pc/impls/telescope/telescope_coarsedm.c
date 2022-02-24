
#include <petsc/private/matimpl.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/dmimpl.h>
#include <petscksp.h>           /*I "petscksp.h" I*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>

#include "../src/ksp/pc/impls/telescope/telescope.h"

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
"@inproceedings{MaySananRuppKnepleySmith2016,\n"
"  title     = {Extreme-Scale Multigrid Components within PETSc},\n"
"  author    = {Dave A. May and Patrick Sanan and Karl Rupp and Matthew G. Knepley and Barry F. Smith},\n"
"  booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},\n"
"  series    = {PASC '16},\n"
"  isbn      = {978-1-4503-4126-4},\n"
"  location  = {Lausanne, Switzerland},\n"
"  pages     = {5:1--5:12},\n"
"  articleno = {5},\n"
"  numpages  = {12},\n"
"  url       = {https://doi.acm.org/10.1145/2929908.2929913},\n"
"  doi       = {10.1145/2929908.2929913},\n"
"  acmid     = {2929913},\n"
"  publisher = {ACM},\n"
"  address   = {New York, NY, USA},\n"
"  keywords  = {GPU, HPC, agglomeration, coarse-level solver, multigrid, parallel computing, preconditioning},\n"
"  year      = {2016}\n"
"}\n";

typedef struct {
  DM              dm_fine,dm_coarse; /* these DM's should be topologically identical but use different communicators */
  Mat             permutation;
  Vec             xp;
  PetscErrorCode  (*fp_dm_field_scatter)(DM,Vec,ScatterMode,DM,Vec);
  PetscErrorCode  (*fp_dm_state_scatter)(DM,ScatterMode,DM);
  void            *dmksp_context_determined;
  void            *dmksp_context_user;
} PC_Telescope_CoarseDMCtx;

PetscErrorCode PCTelescopeSetUp_scatters_CoarseDM(PC pc,PC_Telescope sred,PC_Telescope_CoarseDMCtx *ctx)
{
  Vec            xred,yred,xtmp,x,xp;
  VecScatter     scatter;
  IS             isin;
  Mat            B;
  PetscInt       m,bs,st,ed;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  CHKERRQ(PCGetOperators(pc,NULL,&B));
  CHKERRQ(MatCreateVecs(B,&x,NULL));
  CHKERRQ(MatGetBlockSize(B,&bs));
  CHKERRQ(VecDuplicate(x,&xp));
  m = 0;
  xred = NULL;
  yred = NULL;
  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(DMCreateGlobalVector(ctx->dm_coarse,&xred));
    CHKERRQ(VecDuplicate(xred,&yred));
    CHKERRQ(VecGetOwnershipRange(xred,&st,&ed));
    CHKERRQ(ISCreateStride(comm,ed-st,st,1,&isin));
    CHKERRQ(VecGetLocalSize(xred,&m));
  } else {
    CHKERRQ(VecGetOwnershipRange(x,&st,&ed));
    CHKERRQ(ISCreateStride(comm,0,st,1,&isin));
  }
  CHKERRQ(ISSetBlockSize(isin,bs));
  CHKERRQ(VecCreate(comm,&xtmp));
  CHKERRQ(VecSetSizes(xtmp,m,PETSC_DECIDE));
  CHKERRQ(VecSetBlockSize(xtmp,bs));
  CHKERRQ(VecSetType(xtmp,((PetscObject)x)->type_name));
  CHKERRQ(VecScatterCreate(x,isin,xtmp,NULL,&scatter));
  sred->xred    = xred;
  sred->yred    = yred;
  sred->isin    = isin;
  sred->scatter = scatter;
  sred->xtmp    = xtmp;
  ctx->xp       = xp;
  CHKERRQ(VecDestroy(&x));
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSetUp_CoarseDM(PC pc,PC_Telescope sred)
{
  PC_Telescope_CoarseDMCtx *ctx;
  DM                       dm,dm_coarse = NULL;
  MPI_Comm                 comm;
  PetscBool                has_perm,has_kspcomputeoperators,using_kspcomputeoperators;

  PetscFunctionBegin;
  CHKERRQ(PetscInfo(pc,"PCTelescope: setup (CoarseDM)\n"));
  CHKERRQ(PetscNew(&ctx));
  sred->dm_ctx = (void*)ctx;

  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  CHKERRQ(PCGetDM(pc,&dm));
  CHKERRQ(DMGetCoarseDM(dm,&dm_coarse));
  ctx->dm_fine   = dm;
  ctx->dm_coarse = dm_coarse;

  /* attach coarse dm to ksp on sub communicator */
  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(KSPSetDM(sred->ksp,ctx->dm_coarse));
    if (sred->ignore_kspcomputeoperators) {
      CHKERRQ(KSPSetDMActive(sred->ksp,PETSC_FALSE));
    }
  }

  /* check if there is a method to provide a permutation */
  has_perm = PETSC_FALSE;
  has_kspcomputeoperators = PETSC_FALSE;
  using_kspcomputeoperators = PETSC_FALSE;

  /* if no permutation is provided, we must rely on KSPSetComputeOperators */
  {
    PetscErrorCode (*dmfine_kspfunc)(KSP,Mat,Mat,void*) = NULL;
    void           *dmfine_kspctx = NULL,*dmcoarse_kspctx = NULL;
    void           *dmfine_appctx = NULL,*dmcoarse_appctx = NULL;
    void           *dmfine_shellctx = NULL,*dmcoarse_shellctx = NULL;

    CHKERRQ(DMKSPGetComputeOperators(dm,&dmfine_kspfunc,&dmfine_kspctx));
    if (dmfine_kspfunc) { has_kspcomputeoperators = PETSC_TRUE; }

    CHKERRQ(DMGetApplicationContext(ctx->dm_fine,&dmfine_appctx));
    CHKERRQ(DMShellGetContext(ctx->dm_fine,&dmfine_shellctx));

    /* need to define dmcoarse_kspctx */
    if (dmfine_kspfunc && !sred->ignore_kspcomputeoperators) {

      CHKERRQ(PetscInfo(pc,"PCTelescope: KSPSetComputeOperators fetched from parent DM\n"));
      if (PCTelescope_isActiveRank(sred)) {
        CHKERRQ(DMGetApplicationContext(ctx->dm_coarse,&dmcoarse_appctx));
        CHKERRQ(DMShellGetContext(ctx->dm_coarse,&dmcoarse_shellctx));
      }

      /* Assume that if the fine operator didn't require any context, neither will the coarse */
      if (!dmfine_kspctx) {
        dmcoarse_kspctx = NULL;
        CHKERRQ(PetscInfo(pc,"PCTelescope: KSPSetComputeOperators using NULL context\n"));
      } else {

        CHKERRQ(PetscInfo(pc,"PCTelescope: KSPSetComputeOperators detected non-NULL context from parent DM \n"));
        if (PCTelescope_isActiveRank(sred)) {

          if (dmfine_kspctx == dmfine_appctx) {
            dmcoarse_kspctx = dmcoarse_appctx;
            CHKERRQ(PetscInfo(pc,"PCTelescope: KSPSetComputeOperators using context from DM->ApplicationContext\n"));
            PetscCheckFalse(!dmcoarse_kspctx,PETSC_COMM_SELF,PETSC_ERR_USER,"Non NULL dmfine->kspctx == dmfine->appctx. NULL dmcoarse->appctx found. Likely this is an error");
          } else if (dmfine_kspctx == dmfine_shellctx) {
            dmcoarse_kspctx = dmcoarse_shellctx;
            CHKERRQ(PetscInfo(pc,"PCTelescope: KSPSetComputeOperators using context from DMShell->Context\n"));
            PetscCheckFalse(!dmcoarse_kspctx,PETSC_COMM_SELF,PETSC_ERR_USER,"Non NULL dmfine->kspctx == dmfine.shell->ctx. NULL dmcoarse.shell->ctx found. Likely this is an error");
          }
          ctx->dmksp_context_determined = dmcoarse_kspctx;

          /* look for user provided method to fetch the context */
          {
            PetscErrorCode (*fp_get_coarsedm_context)(DM,void**) = NULL;
            void *dmcoarse_context_user = NULL;
            char dmcoarse_method[PETSC_MAX_PATH_LEN];

            CHKERRQ(PetscSNPrintf(dmcoarse_method,sizeof(dmcoarse_method),"PCTelescopeGetCoarseDMKSPContext"));
            CHKERRQ(PetscObjectQueryFunction((PetscObject)ctx->dm_coarse,dmcoarse_method,&fp_get_coarsedm_context));
            if (fp_get_coarsedm_context) {
              CHKERRQ(PetscInfo(pc,"PCTelescope: Found composed method PCTelescopeGetCoarseDMKSPContext from coarse DM\n"));
              CHKERRQ(fp_get_coarsedm_context(ctx->dm_coarse,&dmcoarse_context_user));
              ctx->dmksp_context_user = dmcoarse_context_user;
              dmcoarse_kspctx = dmcoarse_context_user;
            } else {
              CHKERRQ(PetscInfo(pc,"PCTelescope: Failed to find composed method PCTelescopeGetCoarseDMKSPContext from coarse DM\n"));
            }
          }

          if (!dmcoarse_kspctx) {
            CHKERRQ(PetscInfo(pc,"PCTelescope: KSPSetComputeOperators failed to determine the context to use on sub-communicator\n"));
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot determine which context with use for KSPSetComputeOperators() on sub-communicator");
          }
        }
      }
    }

    if (dmfine_kspfunc && !sred->ignore_kspcomputeoperators) {
      using_kspcomputeoperators = PETSC_TRUE;

      if (PCTelescope_isActiveRank(sred)) {
        /* sub ksp inherits dmksp_func and context provided by user */
        CHKERRQ(KSPSetComputeOperators(sred->ksp,dmfine_kspfunc,dmcoarse_kspctx));
        /*CHKERRQ(PetscObjectCopyFortranFunctionPointers((PetscObject)dm,(PetscObject)ctx->dmrepart));*/
        CHKERRQ(KSPSetDMActive(sred->ksp,PETSC_TRUE));
      }
    }
  }

  PetscCheckFalse(!has_perm && has_kspcomputeoperators && !using_kspcomputeoperators,comm,PETSC_ERR_SUP,"No method to permute an operator was found on the parent DM. A method for KSPSetComputeOperators() was provided but it was requested to be ignored. Telescope setup cannot proceed");
  PetscCheckFalse(!has_perm && !has_kspcomputeoperators,comm,PETSC_ERR_SUP,"No method to permute an operator was found on the parent DM. No method for KSPSetComputeOperators() was provided. Telescope setup cannot proceed");

  {
    char dmfine_method[PETSC_MAX_PATH_LEN];

    CHKERRQ(PetscSNPrintf(dmfine_method,sizeof(dmfine_method),"PCTelescopeFieldScatter"));
    CHKERRQ(PetscObjectQueryFunction((PetscObject)ctx->dm_fine,dmfine_method,&ctx->fp_dm_field_scatter));

    CHKERRQ(PetscSNPrintf(dmfine_method,sizeof(dmfine_method),"PCTelescopeStateScatter"));
    CHKERRQ(PetscObjectQueryFunction((PetscObject)ctx->dm_fine,dmfine_method,&ctx->fp_dm_state_scatter));
  }

  if (ctx->fp_dm_state_scatter) {
    CHKERRQ(PetscInfo(pc,"PCTelescope: Found composed method PCTelescopeStateScatter from parent DM\n"));
  } else {
    CHKERRQ(PetscInfo(pc,"PCTelescope: Failed to find composed method PCTelescopeStateScatter from parent DM\n"));
  }

  if (ctx->fp_dm_field_scatter) {
    CHKERRQ(PetscInfo(pc,"PCTelescope: Found composed method PCTelescopeFieldScatter from parent DM\n"));
  } else {
    CHKERRQ(PetscInfo(pc,"PCTelescope: Failed to find composed method PCTelescopeFieldScatter from parent DM\n"));
    SETERRQ(comm,PETSC_ERR_SUP,"No method to scatter fields between the parent DM and coarse DM was found. Must call PetscObjectComposeFunction() with the parent DM. Telescope setup cannot proceed");
  }

  /*CHKERRQ(PCTelescopeSetUp_permutation_CoarseDM(pc,sred,ctx));*/
  CHKERRQ(PCTelescopeSetUp_scatters_CoarseDM(pc,sred,ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PCApply_Telescope_CoarseDM(PC pc,Vec x,Vec y)
{
  PC_Telescope             sred = (PC_Telescope)pc->data;
  Vec                      xred,yred;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  xred = sred->xred;
  yred = sred->yred;

  CHKERRQ(PetscCitationsRegister(citation,&cited));

  if (ctx->fp_dm_state_scatter) {
    CHKERRQ(ctx->fp_dm_state_scatter(ctx->dm_fine,SCATTER_FORWARD,ctx->dm_coarse));
  }

  CHKERRQ(ctx->fp_dm_field_scatter(ctx->dm_fine,x,SCATTER_FORWARD,ctx->dm_coarse,xred));

  /* solve */
  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(KSPSolve(sred->ksp,xred,yred));
  }

  CHKERRQ(ctx->fp_dm_field_scatter(ctx->dm_fine,y,SCATTER_REVERSE,ctx->dm_coarse,yred));
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSubNullSpaceCreate_CoarseDM(PC pc,PC_Telescope sred,MatNullSpace nullspace,MatNullSpace *sub_nullspace)
{
  PetscBool                has_const;
  PetscInt                 k,n = 0;
  const Vec                *vecs;
  Vec                      *sub_vecs = NULL;
  MPI_Comm                 subcomm;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  subcomm = sred->subcomm;
  CHKERRQ(MatNullSpaceGetVecs(nullspace,&has_const,&n,&vecs));

  if (PCTelescope_isActiveRank(sred)) {
    /* create new vectors */
    if (n) {
      CHKERRQ(VecDuplicateVecs(sred->xred,n,&sub_vecs));
    }
  }

  /* copy entries */
  for (k=0; k<n; k++) {
    CHKERRQ(ctx->fp_dm_field_scatter(ctx->dm_fine,vecs[k],SCATTER_FORWARD,ctx->dm_coarse,sub_vecs[k]));
  }

  if (PCTelescope_isActiveRank(sred)) {
    /* create new (near) nullspace for redundant object */
    CHKERRQ(MatNullSpaceCreate(subcomm,has_const,n,sub_vecs,sub_nullspace));
    CHKERRQ(VecDestroyVecs(n,&sub_vecs));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatNullSpaceCreate_CoarseDM(PC pc,PC_Telescope sred,Mat sub_mat)
{
  Mat                      B;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  CHKERRQ(PCGetOperators(pc,NULL,&B));
  {
    MatNullSpace nullspace,sub_nullspace;
    CHKERRQ(MatGetNullSpace(B,&nullspace));
    if (nullspace) {
      CHKERRQ(PetscInfo(pc,"PCTelescope: generating nullspace (CoarseDM)\n"));
      CHKERRQ(PCTelescopeSubNullSpaceCreate_CoarseDM(pc,sred,nullspace,&sub_nullspace));

      /* attach any user nullspace removal methods and contexts */
      if (PCTelescope_isActiveRank(sred)) {
        void *context = NULL;
        if (nullspace->remove && !nullspace->rmctx) {
          CHKERRQ(MatNullSpaceSetFunction(sub_nullspace,nullspace->remove,context));
        } else if (nullspace->remove && nullspace->rmctx) {
          char           dmcoarse_method[PETSC_MAX_PATH_LEN];
          PetscErrorCode (*fp_get_coarsedm_context)(DM,void**) = NULL;

          CHKERRQ(PetscSNPrintf(dmcoarse_method,sizeof(dmcoarse_method),"PCTelescopeGetCoarseDMNullSpaceUserContext"));
          CHKERRQ(PetscObjectQueryFunction((PetscObject)ctx->dm_coarse,dmcoarse_method,&fp_get_coarsedm_context));
          PetscCheckFalse(!context,PETSC_COMM_SELF,PETSC_ERR_SUP,"Propagation of user null-space removal method with non-NULL context requires the coarse DM be composed with a function named \"%s\"",dmcoarse_method);
          CHKERRQ(MatNullSpaceSetFunction(sub_nullspace,nullspace->remove,context));
        }
      }

      if (PCTelescope_isActiveRank(sred)) {
        CHKERRQ(MatSetNullSpace(sub_mat,sub_nullspace));
        CHKERRQ(MatNullSpaceDestroy(&sub_nullspace));
      }
    }
  }
  {
    MatNullSpace nearnullspace,sub_nearnullspace;
    CHKERRQ(MatGetNearNullSpace(B,&nearnullspace));
    if (nearnullspace) {
      CHKERRQ(PetscInfo(pc,"PCTelescope: generating near nullspace (CoarseDM)\n"));
      CHKERRQ(PCTelescopeSubNullSpaceCreate_CoarseDM(pc,sred,nearnullspace,&sub_nearnullspace));

      /* attach any user nullspace removal methods and contexts */
      if (PCTelescope_isActiveRank(sred)) {
        void *context = NULL;
        if (nearnullspace->remove && !nearnullspace->rmctx) {
          CHKERRQ(MatNullSpaceSetFunction(sub_nearnullspace,nearnullspace->remove,context));
        } else if (nearnullspace->remove && nearnullspace->rmctx) {
          char           dmcoarse_method[PETSC_MAX_PATH_LEN];
          PetscErrorCode (*fp_get_coarsedm_context)(DM,void**) = NULL;

          CHKERRQ(PetscSNPrintf(dmcoarse_method,sizeof(dmcoarse_method),"PCTelescopeGetCoarseDMNearNullSpaceUserContext"));
          CHKERRQ(PetscObjectQueryFunction((PetscObject)ctx->dm_coarse,dmcoarse_method,&fp_get_coarsedm_context));
          PetscCheckFalse(!context,PETSC_COMM_SELF,PETSC_ERR_SUP,"Propagation of user near null-space removal method with non-NULL context requires the coarse DM be composed with a function named \"%s\"",dmcoarse_method);
          CHKERRQ(MatNullSpaceSetFunction(sub_nearnullspace,nearnullspace->remove,context));
        }
      }

      if (PCTelescope_isActiveRank(sred)) {
        CHKERRQ(MatSetNearNullSpace(sub_mat,sub_nearnullspace));
        CHKERRQ(MatNullSpaceDestroy(&sub_nearnullspace));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCReset_Telescope_CoarseDM(PC pc)
{
  PC_Telescope             sred = (PC_Telescope)pc->data;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  ctx->dm_fine = NULL; /* since I did not increment the ref counter we set these to NULL */
  ctx->dm_coarse = NULL; /* since I did not increment the ref counter we set these to NULL */
  ctx->permutation = NULL; /* this will be fetched from the dm so no need to call destroy */
  CHKERRQ(VecDestroy(&ctx->xp));
  ctx->fp_dm_field_scatter = NULL;
  ctx->fp_dm_state_scatter = NULL;
  ctx->dmksp_context_determined = NULL;
  ctx->dmksp_context_user = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PCApplyRichardson_Telescope_CoarseDM(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt its,PetscBool zeroguess,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_Telescope             sred = (PC_Telescope)pc->data;
  Vec                      yred = NULL;
  PetscBool                default_init_guess_value = PETSC_FALSE;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  yred = sred->yred;

  PetscCheckFalse(its > 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"PCApplyRichardson_Telescope_CoarseDM only supports max_it = 1");
  *reason = (PCRichardsonConvergedReason)0;

  if (!zeroguess) {
    CHKERRQ(PetscInfo(pc,"PCTelescopeCoarseDM: Scattering y for non-zero-initial guess\n"));

    CHKERRQ(ctx->fp_dm_field_scatter(ctx->dm_fine,y,SCATTER_FORWARD,ctx->dm_coarse,yred));
  }

  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(KSPGetInitialGuessNonzero(sred->ksp,&default_init_guess_value));
    if (!zeroguess) CHKERRQ(KSPSetInitialGuessNonzero(sred->ksp,PETSC_TRUE));
  }

  CHKERRQ(PCApply_Telescope_CoarseDM(pc,x,y));

  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(KSPSetInitialGuessNonzero(sred->ksp,default_init_guess_value));
  }

  if (!*reason) *reason = PCRICHARDSON_CONVERGED_ITS;
  *outits = 1;
  PetscFunctionReturn(0);
}
