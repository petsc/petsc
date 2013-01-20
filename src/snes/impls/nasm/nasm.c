#include <petsc-private/snesimpl.h>             /*I   "petscsnes.h"   I*/
#include <petscdm.h>

typedef struct {
  PetscInt   n;                   /* local subdomains */
  SNES       *subsnes;            /* nonlinear solvers for each subdomain */

  Vec        *x;                  /* solution vectors */
  Vec        *xl;                 /* solution local vectors */
  Vec        *y;                  /* step vectors */
  Vec        *b;                  /* rhs vectors */

  VecScatter *oscatter;           /* scatter from global space to the subdomain global space */
  VecScatter *iscatter;           /* scatter from global space to the nonoverlapping subdomain space */
  VecScatter *gscatter;           /* scatter from global space to the subdomain local space */

  PCASMType type;                 /* ASM type */

  PetscBool  usesdm;               /* use the DM for setting up the subproblems */
} SNES_NASM;

#undef __FUNCT__
#define __FUNCT__ "SNESReset_NASM"
PetscErrorCode SNESReset_NASM(SNES snes)
{
  SNES_NASM      *nasm = (SNES_NASM *)snes->data;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscFunctionBegin;
  for (i=0;i<nasm->n;i++) {
    if (nasm->xl) { ierr = VecDestroy(&nasm->xl[i]);CHKERRQ(ierr); }
    if (nasm->x) { ierr = VecDestroy(&nasm->x[i]);CHKERRQ(ierr); }
    if (nasm->y) { ierr = VecDestroy(&nasm->y[i]);CHKERRQ(ierr); }
    if (nasm->b) { ierr = VecDestroy(&nasm->b[i]);CHKERRQ(ierr); }

    if (nasm->subsnes) { ierr = SNESDestroy(&nasm->subsnes[i]);CHKERRQ(ierr); }
    if (nasm->oscatter) { ierr = VecScatterDestroy(&nasm->oscatter[i]);CHKERRQ(ierr); }
    if (nasm->iscatter) { ierr = VecScatterDestroy(&nasm->iscatter[i]);CHKERRQ(ierr); }
    if (nasm->gscatter) { ierr = VecScatterDestroy(&nasm->gscatter[i]);CHKERRQ(ierr); }
  }

  if (nasm->x) {ierr = PetscFree(nasm->x);CHKERRQ(ierr);}
  if (nasm->xl) {ierr = PetscFree(nasm->xl);CHKERRQ(ierr);}
  if (nasm->y) {ierr = PetscFree(nasm->y);CHKERRQ(ierr);}
  if (nasm->b) {ierr = PetscFree(nasm->b);CHKERRQ(ierr);}

  if (nasm->subsnes) {ierr = PetscFree(nasm->subsnes);CHKERRQ(ierr);}
  if (nasm->oscatter) {ierr = PetscFree(nasm->oscatter);CHKERRQ(ierr);}
  if (nasm->iscatter) {ierr = PetscFree(nasm->iscatter);CHKERRQ(ierr);}
  if (nasm->gscatter) {ierr = PetscFree(nasm->gscatter);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NASM"
PetscErrorCode SNESDestroy_NASM(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESReset_NASM(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalSubDomainDirichletHook_Private"
PetscErrorCode DMGlobalToLocalSubDomainDirichletHook_Private(DM dm,Vec g,InsertMode mode,Vec l,void *ctx)
{
  PetscErrorCode ierr;
  Vec bcs = (Vec)ctx;
  PetscFunctionBegin;
  ierr = VecCopy(bcs,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NASM"
PetscErrorCode SNESSetUp_NASM(SNES snes)
{
  SNES_NASM      *nasm = (SNES_NASM *)snes->data;
  PetscErrorCode ierr;
  DM             dm,ddm;
  DM             *subdms;
  PetscInt       i;
  const char     *optionsprefix;
  Vec            F;

  PetscFunctionBegin;

  if (!nasm->subsnes) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    if (dm) {
      nasm->usesdm = PETSC_TRUE;
      ierr = DMCreateDomainDecomposition(dm,&nasm->n,PETSC_NULL,PETSC_NULL,PETSC_NULL,&subdms);CHKERRQ(ierr);
      if (!subdms) {
        ierr = DMCreateDomainDecompositionDM(dm,"default",&ddm);CHKERRQ(ierr);
        if (!ddm) {
          SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"DM has no default decomposition defined.  Set subsolves manually with SNESNASMSetSubdomains().");
        }
        ierr = SNESSetDM(snes,ddm);CHKERRQ(ierr);
        ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
        ierr = DMCreateDomainDecomposition(dm,&nasm->n,PETSC_NULL,PETSC_NULL,PETSC_NULL,&subdms);CHKERRQ(ierr);
      }
      if (!subdms) {
        SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"DM has no default decomposition defined.  Set subsolves manually with SNESNASMSetSubdomains().");
      }
      ierr = DMCreateDomainDecompositionScatters(dm,nasm->n,subdms,&nasm->iscatter,&nasm->oscatter,&nasm->gscatter);CHKERRQ(ierr);

      ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
      ierr = PetscMalloc(nasm->n*sizeof(SNES),&nasm->subsnes);CHKERRQ(ierr);

      for (i=0;i<nasm->n;i++) {
        ierr = SNESCreate(PETSC_COMM_SELF,&nasm->subsnes[i]);CHKERRQ(ierr);
        ierr = SNESAppendOptionsPrefix(nasm->subsnes[i],optionsprefix);CHKERRQ(ierr);
        ierr = SNESAppendOptionsPrefix(nasm->subsnes[i],"sub_");CHKERRQ(ierr);
        ierr = SNESSetDM(nasm->subsnes[i],subdms[i]);CHKERRQ(ierr);
        ierr = SNESSetFromOptions(nasm->subsnes[i]);CHKERRQ(ierr);
        ierr = DMDestroy(&subdms[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(subdms);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot construct local problems automatically without a DM!");
    }
  } else {
    SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set subproblems manually if there is no DM!");
  }
  /* allocate the global vectors */
  ierr = PetscMalloc(nasm->n*sizeof(Vec),&nasm->x);CHKERRQ(ierr);
  ierr = PetscMalloc(nasm->n*sizeof(Vec),&nasm->xl);CHKERRQ(ierr);
  ierr = PetscMalloc(nasm->n*sizeof(Vec),&nasm->y);CHKERRQ(ierr);
  ierr = PetscMalloc(nasm->n*sizeof(Vec),&nasm->b);CHKERRQ(ierr);

  for (i=0;i<nasm->n;i++) {
    DM subdm;
    ierr = SNESGetFunction(nasm->subsnes[i],&F,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecDuplicate(F,&nasm->x[i]);CHKERRQ(ierr);
    ierr = VecDuplicate(F,&nasm->y[i]);CHKERRQ(ierr);
    ierr = VecDuplicate(F,&nasm->b[i]);CHKERRQ(ierr);
    ierr = SNESGetDM(nasm->subsnes[i],&subdm);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(subdm,&nasm->xl[i]);CHKERRQ(ierr);
    ierr = DMGlobalToLocalHookAdd(subdm,DMGlobalToLocalSubDomainDirichletHook_Private,PETSC_NULL,nasm->xl[i]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NASM"
PetscErrorCode SNESSetFromOptions_NASM(SNES snes)
{
  PetscErrorCode ierr;
  DM             dm,ddm;
  char           ddm_name[1024];
  PCASMType      asmtype;
  const char *const SNESNASMTypes[]       = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCASMType","PC_ASM_",0};
  PetscBool      flg;
  SNES_NASM      *nasm = (SNES_NASM *)snes->data;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Nonlinear Additive Schwartz options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_nasm_type","Type of restriction/extension","",SNESNASMTypes,(PetscEnum)nasm->type,(PetscEnum*)&asmtype,&flg);CHKERRQ(ierr);
  if (flg) {nasm->type = asmtype;}
  ierr = PetscOptionsString("-snes_nasm_decomposition", "Name of the DM defining the composition", "SNESSetDM", ddm_name, ddm_name,1024,&flg);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  if (flg) {
    if (dm) {
      ierr = DMCreateDomainDecompositionDM(dm, ddm_name, &ddm);CHKERRQ(ierr);
      if (!ddm) {
        SETERRQ1(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONGSTATE, "Unknown DM decomposition name %s", ddm_name);
      }
      ierr = PetscInfo(snes,"Using domain decomposition DM defined using options database\n");CHKERRQ(ierr);
      ierr = SNESSetDM(snes,ddm);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONGSTATE, "No DM to decompose");
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_NASM"
PetscErrorCode SNESView_NASM(SNES snes, PetscViewer viewer)
{
  PetscFunctionBegin;
  

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNASMSetSubdomains"
PetscErrorCode SNESNASMSetSubdomains(SNES snes,PetscInt n,SNES subsnes[],VecScatter iscatter[],VecScatter oscatter[],VecScatter gscatter[])
{
  PetscErrorCode ierr;
  PetscErrorCode (*f)(SNES,PetscInt,SNES*,VecScatter*,VecScatter*,VecScatter*);
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetSubdomains_C",(void (**)(void))&f);CHKERRQ(ierr);
  ierr = (f)(snes,n,subsnes,iscatter,oscatter,gscatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESNASMSetSubdomains_NASM"
PetscErrorCode SNESNASMSetSubdomains_NASM(SNES snes,PetscInt n,SNES subsnes[],VecScatter iscatter[],VecScatter oscatter[],VecScatter gscatter[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  SNES_NASM      *nasm = (SNES_NASM *)snes->data;
  PetscFunctionBegin;
  if (snes->setupcalled)SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"SNESNASMSetSubdomains() should be called before calling SNESSetUp().");

  /* tear down the previously set things */
  ierr = SNESReset(snes);CHKERRQ(ierr);

  nasm->n        = n;
  if (oscatter) {
    for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)oscatter[i]);CHKERRQ(ierr);}
  }
  if (iscatter) {
    for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)iscatter[i]);CHKERRQ(ierr);}
  }
  if (gscatter) {
    for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)gscatter[i]);CHKERRQ(ierr);}
  }
  if (oscatter) {
    ierr = PetscMalloc(n*sizeof(IS),&nasm->oscatter);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      nasm->oscatter[i] = oscatter[i];
    }
  }
  if (iscatter) {
    ierr = PetscMalloc(n*sizeof(IS),&nasm->iscatter);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      nasm->iscatter[i] = iscatter[i];
    }
  }
  if (gscatter) {
    ierr = PetscMalloc(n*sizeof(IS),&nasm->gscatter);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      nasm->gscatter[i] = gscatter[i];
    }
  }

  if (subsnes) {
    ierr = PetscMalloc(n*sizeof(SNES),&nasm->subsnes);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      nasm->subsnes[i] = subsnes[i];
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SNESNASMSolveLocal_Private"
PetscErrorCode SNESNASMSolveLocal_Private(SNES snes,Vec B,Vec Y,Vec X)
{
  SNES_NASM      *nasm = (SNES_NASM *)snes->data;
  SNES           subsnes;
  PetscInt       i;
  PetscErrorCode ierr;
  Vec            Xlloc,Xl,Bl,Yl;
  VecScatter     iscat,oscat,gscat;
  DM             dm,subdm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = VecSet(Y,0);CHKERRQ(ierr);
  for(i=0;i<nasm->n;i++) {
    subsnes = nasm->subsnes[i];
    ierr = SNESGetDM(subsnes,&subdm);CHKERRQ(ierr);
    iscat = nasm->iscatter[i];
    oscat = nasm->oscatter[i];
    gscat = nasm->gscatter[i];
    ierr = DMSubDomainRestrict(dm,oscat,gscat,subdm);CHKERRQ(ierr);

    /* scatter the solution to the local solution */
    Xl = nasm->x[i];
    Xlloc = nasm->xl[i];
    Yl = nasm->y[i];

    ierr = VecScatterBegin(oscat,X,Xl,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(oscat,X,Xl,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(gscat,X,Xlloc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecCopy(Xl,Yl);CHKERRQ(ierr);

    if (B) {
      /* scatter the RHS to the local RHS */
      Bl = nasm->b[i];
      ierr = VecScatterBegin(oscat,B,Bl,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscat,B,Bl,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    } else {
      Bl = PETSC_NULL;
    }
    ierr = SNESSolve(subsnes,Bl,Yl);CHKERRQ(ierr);

    ierr = VecAXPY(Yl,-1.0,Xl);CHKERRQ(ierr);

    if (nasm->type == PC_ASM_BASIC) {
      ierr = VecScatterBegin(oscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    } else if (nasm->type == PC_ASM_RESTRICT) {
      ierr = VecScatterBegin(iscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(iscat,Yl,Y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Only basic and interpolate types are supported for SNESNASM");
    }
  }

  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);

  ierr = VecAXPY(X,1.0,Y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NASM"
PetscErrorCode SNESSolve_NASM(SNES snes)
{
  Vec            F;
  Vec            X;
  Vec            B;
  Vec            Y;
  PetscInt       i;
  PetscReal      fnorm;
  PetscErrorCode ierr;
  SNESNormType   normtype;

  PetscFunctionBegin;

  X = snes->vec_sol;
  Y = snes->vec_sol_update;
  F = snes->vec_func;
  B = snes->vec_rhs;

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITERATING;
  ierr = SNESGetNormType(snes, &normtype);CHKERRQ(ierr);
  if (normtype == SNES_NORM_FUNCTION || normtype == SNES_NORM_INITIAL_ONLY || normtype == SNES_NORM_INITIAL_FINAL_ONLY) {
    /* compute the initial function and preconditioned update delX */
    if (!snes->vec_func_init_set) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
    } else {
      snes->vec_func_init_set = PETSC_FALSE;
    }

    /* convergence test */
    if (!snes->norm_init_set) {
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    } else {
      fnorm = snes->norm_init;
      snes->norm_init_set = PETSC_FALSE;
    }
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = 0;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);

    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;

    /* test convergence */
    ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
  }

  for (i = 0; i < snes->max_its; i++) {
    ierr = SNESNASMSolveLocal_Private(snes,B,Y,X);CHKERRQ(ierr);
    if (normtype == SNES_NORM_FUNCTION || ((i == snes->max_its - 1) && (normtype == SNES_NORM_INITIAL_FINAL_ONLY || normtype == SNES_NORM_FINAL_ONLY))) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    }
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    if (normtype == SNES_NORM_FUNCTION) ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
  }
  if (normtype == SNES_NORM_FUNCTION) {
    if (i == snes->max_its) {
      ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
      if(!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
    }
  } else {
    if (!snes->reason) snes->reason = SNES_CONVERGED_ITS; /* NASM is meant to be used as a preconditioner */
  }
  PetscFunctionReturn(0);
}

/*MC
  SNESNASM - Nonlinear Additive Schwartz

   Level: advanced

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NASM"
PetscErrorCode SNESCreate_NASM(SNES snes)
{
  SNES_NASM        *nasm;
  PetscErrorCode ierr;

  PetscFunctionBegin;


  ierr = PetscNewLog(snes, SNES_NASM, &nasm);CHKERRQ(ierr);
  snes->data = (void*)nasm;

  nasm->n                 = PETSC_DECIDE;
  nasm->subsnes           = 0;
  nasm->x                 = 0;
  nasm->xl                = 0;
  nasm->y                 = 0;
  nasm->b                 = 0;
  nasm->oscatter          = 0;
  nasm->iscatter          = 0;
  nasm->gscatter          = 0;

  nasm->type              = PC_ASM_BASIC;

  snes->ops->destroy        = SNESDestroy_NASM;
  snes->ops->setup          = SNESSetUp_NASM;
  snes->ops->setfromoptions = SNESSetFromOptions_NASM;
  snes->ops->view           = SNESView_NASM;
  snes->ops->solve          = SNESSolve_NASM;
  snes->ops->reset          = SNESReset_NASM;

  snes->usesksp             = PETSC_FALSE;
  snes->usespc              = PETSC_FALSE;

  if (!snes->tolerancesset) {
    snes->max_its             = 10000;
    snes->max_funcs           = 10000;
  }

    ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESNASMSetSubdomains_C","SNESNASMSetSubdomains_NASM",
                    SNESNASMSetSubdomains_NASM);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
