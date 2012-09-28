#include <petsc-private/snesimpl.h>             /*I   "petscsnes.h"   I*/
#include <petscdmshell.h>

typedef struct {
  PetscInt   n;                   /* local subdomains */
  SNES       *subsnes;            /* nonlinear solvers for each subdomain */

  Vec        *r;                  /* function vectors */
  Vec        *x;                  /* solution vectors */
  Vec        *b;                  /* rhs vectors */

  IS         *ois;
  IS         *iis;
  PetscBool  usesdm;               /* use the outer DM's communication facilities rather than ISes */
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
    if (nasm->x) ierr = VecDestroy(&nasm->x[i]);CHKERRQ(ierr);
    if (nasm->r) ierr = VecDestroy(&nasm->r[i]);CHKERRQ(ierr);
    if (nasm->b) ierr = VecDestroy(&nasm->b[i]);CHKERRQ(ierr);

    if (nasm->subsnes) ierr = SNESDestroy(&nasm->subsnes[i]);CHKERRQ(ierr);
    if (nasm->ois) ierr = ISDestroy(&nasm->ois[i]);CHKERRQ(ierr);
    if (nasm->iis) ierr = ISDestroy(&nasm->iis[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NASM"
PetscErrorCode SNESDestroy_NASM(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESReset_NASM(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NASM"
PetscErrorCode SNESSetUp_NASM(SNES snes)
{
  SNES_NASM      *nasm = (SNES_NASM *)snes->data;
  PetscErrorCode ierr;
  DM             dm;
  DM             subdm;
  PetscErrorCode (*f)(SNES,Vec,Vec,void*);

  Mat            A;
  PetscErrorCode (*fj)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

  PetscInt       n;
  void           *ctx;
  const char     *optionsprefix;

  PetscFunctionBegin;

  if (!nasm->subsnes) {
    if (snes->dm) {
      ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
      nasm->n      = 1;
      nasm->usesdm = PETSC_TRUE;
      /* create a single solver */
      ierr = PetscMalloc(sizeof(SNES),&nasm->subsnes);CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(Vec),&nasm->r);CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(Vec),&nasm->x);CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(Vec),&nasm->b);CHKERRQ(ierr);
      ierr = SNESCreate(PETSC_COMM_SELF,&nasm->subsnes[0]);CHKERRQ(ierr);

      ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
      ierr = SNESAppendOptionsPrefix(nasm->subsnes[0],optionsprefix);CHKERRQ(ierr);
      ierr = SNESAppendOptionsPrefix(nasm->subsnes[0],"sub_");CHKERRQ(ierr);

      ierr = SNESGetDM(nasm->subsnes[0],&subdm);CHKERRQ(ierr);

      /* set up the local function */
      ierr = DMSNESGetBlockFunction(dm,&f,&ctx);CHKERRQ(ierr);
      ierr = DMCreateLocalVector(dm,&nasm->r[0]);CHKERRQ(ierr);
      ierr = DMShellSetGlobalVector(dm,nasm->r[0]);CHKERRQ(ierr);
      ierr = DMCreateLocalVector(dm,&nasm->b[0]);CHKERRQ(ierr);
      ierr = DMCreateLocalVector(dm,&nasm->x[0]);CHKERRQ(ierr);

      if (!f) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Need to provide a block solve!");CHKERRQ(ierr);

      ierr = SNESSetFunction(nasm->subsnes[0],nasm->r[0],f,ctx);CHKERRQ(ierr);

      /* set up the local jacobian -- TODO: do this correctly */
      ierr = DMSNESGetBlockJacobian(dm,&fj,&ctx);CHKERRQ(ierr);
      ierr = VecGetSize(nasm->r[0],&n);CHKERRQ(ierr);
      ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,PETSC_DEFAULT,PETSC_NULL,&A);CHKERRQ(ierr);
      ierr = DMShellSetMatrix(subdm,A);CHKERRQ(ierr);
      ierr = SNESSetJacobian(nasm->subsnes[0],A,A,fj,ctx);CHKERRQ(ierr);

      ierr = SNESSetFromOptions(nasm->subsnes[0]);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot construct local problems automatically without a DM!");
    }
  } else {
    SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set subproblems manually if there is no DM!");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NASM"
PetscErrorCode SNESSetFromOptions_NASM(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NASM options");CHKERRQ(ierr);
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
PetscErrorCode SNESNASMSetSubdomains(SNES snes,PetscInt n,SNES subsnes[],IS iis[],IS ois[]) {
  PetscErrorCode ierr;
  PetscErrorCode (*f)(SNES,PetscInt,SNES*,IS*,IS*);
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESNASMSetSubdomains_C",(void (**)(void))&f);CHKERRQ(ierr);
  ierr = (f)(snes,n,subsnes,iis,ois);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESNASMSetSubdomains_NASM"
PetscErrorCode SNESNASMSetSubdomains_NASM(SNES snes,PetscInt n,SNES subsnes[],IS iis[],IS ois[]) {
  PetscInt       i;
  PetscErrorCode ierr;
  SNES_NASM      *nasm = (SNES_NASM *)snes->data;
  PetscFunctionBegin;
  if (snes->setupcalled)SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"SNESNASMSetSubdomains() should be called before calling SNESSetUp().");

  if (!snes->setupcalled) {
    nasm->n       = n;
    nasm->ois     = 0;
    nasm->iis     = 0;
    if (ois) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)ois[i]);CHKERRQ(ierr);}
    }
    if (iis) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)iis[i]);CHKERRQ(ierr);}
    }
    if (ois) {
      ierr = PetscMalloc(n*sizeof(IS),&nasm->ois);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        nasm->ois[i] = ois[i];
      }
      if (!iis) {
        ierr = PetscMalloc(n*sizeof(IS),&nasm->iis);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)ois[i]);CHKERRQ(ierr);}
          nasm->iis[i] = ois[i];
        }
      }
    }
    if (iis) {
      ierr = PetscMalloc(n*sizeof(IS),&nasm->iis);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        nasm->iis[i] = iis[i];
      }
      if (!ois) {
        ierr = PetscMalloc(n*sizeof(IS),&nasm->ois);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          for (i=0; i<n; i++) {
            ierr = PetscObjectReference((PetscObject)iis[i]);CHKERRQ(ierr);
            nasm->ois[i] = iis[i];
          }
        }
      }
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
PetscErrorCode SNESNASMSolveLocal_Private(SNES snes,Vec B,Vec X) {
  SNES_NASM      *nasm = (SNES_NASM *)snes->data;
  PetscInt       i;
  PetscErrorCode ierr;
  Vec            Xl,Bl;
  DM             dm;
  PetscFunctionBegin;
    /* restrict to the local system */
  if (nasm->usesdm) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,nasm->x[0]);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,nasm->x[0]);CHKERRQ(ierr);
    if (B) {
      ierr = DMGlobalToLocalBegin(dm,B,INSERT_VALUES,nasm->b[0]);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dm,B,INSERT_VALUES,nasm->b[0]);CHKERRQ(ierr);
    }
  } else {
    /*
    for (i = 0;i < nasm->n;i++) {
      ierr = VecScatterBegin(nasm->gorestriction,X,nasm->gx,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(nasm->gorestriction,X,nasm->gx,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      if (B) {
        ierr = VecScatterBegin(nasm->gorestriction,B,nasm->gb,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(nasm->gorestriction,B,nasm->gb,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
     } else {
     }
     }
     */
  }
  for(i=0;i<nasm->n;i++) {
    Xl = nasm->x[i];
    if (B) {
      Bl = nasm->b[i];
    } else {
      Bl = PETSC_NULL;
    }
    ierr = SNESSolve(nasm->subsnes[i],Bl,Xl);CHKERRQ(ierr);
  }

  if (nasm->usesdm) {
    ierr = DMLocalToGlobalBegin(dm,Xl,INSERT_VALUES,X);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,Xl,INSERT_VALUES,X);CHKERRQ(ierr);
  } else {
    /*
    ierr = VecScatterBegin(nasm->girestriction,nasm->gx,X,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(nasm->girestriction,nasm->gx,X,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
     */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NASM"
PetscErrorCode SNESSolve_NASM(SNES snes)
{
  Vec            F;
  Vec            X;
  Vec            B;
  PetscInt       i;
  PetscReal      fnorm;
  PetscErrorCode ierr;
  SNESNormType   normtype;

  PetscFunctionBegin;

  X = snes->vec_sol;
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
    ierr = SNESNASMSolveLocal_Private(snes,B,X);CHKERRQ(ierr);
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
  nasm->b                 = 0;
  nasm->ois               = 0;
  nasm->iis               = 0;

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
