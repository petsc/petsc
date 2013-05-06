
/*
      Defines a SNES that can consist of a collection of SNESes
*/
#include <petsc-private/snesimpl.h> /*I "petscsnes.h" I*/

const char *const        SNESCompositeTypes[]   = {"ADDITIVE","MULTIPLICATIVE","SNESCompositeType","SNES_COMPOSITE",0};

typedef struct _SNES_CompositeLink *SNES_CompositeLink;
struct _SNES_CompositeLink {
  SNES               snes;
  PetscReal          dmp;
  SNES_CompositeLink next;
  SNES_CompositeLink previous;
};

typedef struct {
  SNES_CompositeLink head;
  SNESCompositeType  type;
  Vec                Xorig;
} SNES_Composite;

#undef __FUNCT__
#define __FUNCT__ "SNESCompositeApply_Multiplicative"
static PetscErrorCode SNESCompositeApply_Multiplicative(SNES snes,Vec X,Vec B,Vec F,PetscReal *fnorm)
{
  PetscErrorCode     ierr;
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head;
  Vec                FSub;
  PetscReal          fsubnorm;

  PetscFunctionBegin;
  if (!next) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");
  if (snes->normtype == SNES_NORM_FUNCTION) {
    ierr = SNESSetInitialFunction(next->snes,F);CHKERRQ(ierr);
    if (fnorm) {ierr = SNESSetInitialFunctionNorm(next->snes,*fnorm);CHKERRQ(ierr);}
  }
  ierr = SNESSolve(next->snes,B,X);CHKERRQ(ierr);

  while (next->next) {
    /* only copy the function over in the case where the functions correspond */
    if (next->snes->pcside == PC_RIGHT && next->snes->normtype != SNES_NORM_NONE) {
      ierr = SNESGetFunction(next->snes,&FSub,NULL,NULL);CHKERRQ(ierr);
      ierr = SNESGetFunctionNorm(next->snes,&fsubnorm);CHKERRQ(ierr);
      next = next->next;
      ierr = SNESSetInitialFunction(next->snes,FSub);CHKERRQ(ierr);
      ierr = SNESSetInitialFunctionNorm(next->snes,fsubnorm);CHKERRQ(ierr);
    } else {
      next = next->next;
    }
    ierr = SNESSolve(next->snes,B,X);CHKERRQ(ierr);
  }
  if (next->snes->pcside == PC_RIGHT) {
    ierr = SNESGetFunction(next->snes,&FSub,NULL,NULL);CHKERRQ(ierr);
    ierr = VecCopy(FSub,F);CHKERRQ(ierr);
    if (fnorm) {ierr = SNESGetFunctionNorm(next->snes,fnorm);CHKERRQ(ierr);}
  } else if (snes->normtype == SNES_NORM_FUNCTION) {
    SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    if (fnorm) {ierr = VecNorm(F,NORM_2,fnorm);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESCompositeApply_Additive"
static PetscErrorCode SNESCompositeApply_Additive(SNES snes,Vec X,Vec B,Vec F,PetscReal *fnorm)
{
  PetscErrorCode     ierr;
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head;
  Vec                Y,Xorig;

  PetscFunctionBegin;
  Y = snes->vec_sol_update;
  if (!jac->Xorig) {ierr = VecDuplicate(X,&jac->Xorig);CHKERRQ(ierr);}
  Xorig = jac->Xorig;
  ierr = VecCopy(X,Xorig);
  if (!next) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");
  if (snes->normtype == SNES_NORM_FUNCTION) {
    ierr = SNESSetInitialFunction(next->snes,F);CHKERRQ(ierr);
    if (fnorm) {ierr = SNESSetInitialFunctionNorm(next->snes,*fnorm);CHKERRQ(ierr);}
    while (next->next) {
      next = next->next;
      ierr = SNESSetInitialFunction(next->snes,F);CHKERRQ(ierr);
      if (fnorm) {ierr = SNESSetInitialFunctionNorm(next->snes,*fnorm);CHKERRQ(ierr);}
    }
  }
  next = jac->head;
  ierr = VecCopy(Xorig,Y);CHKERRQ(ierr);
  ierr = SNESSolve(next->snes,B,Y);CHKERRQ(ierr);
  ierr = VecAXPY(Y,-1.0,Xorig);CHKERRQ(ierr);
  ierr = VecAXPY(X,next->dmp,Y);CHKERRQ(ierr);
  while (next->next) {
    next = next->next;
    ierr = VecCopy(Xorig,Y);CHKERRQ(ierr);
    ierr = SNESSolve(next->snes,B,Y);CHKERRQ(ierr);
    ierr = VecAXPY(Y,-1.0,Xorig);CHKERRQ(ierr);
    ierr = VecAXPY(X,next->dmp,Y);CHKERRQ(ierr);
  }
  if (snes->normtype == SNES_NORM_FUNCTION) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    if (fnorm) {ierr = VecNorm(F,NORM_2,fnorm);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_Composite"
static PetscErrorCode SNESSetUp_Composite(SNES snes)
{
  PetscErrorCode   ierr;
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head;

  PetscFunctionBegin;
  while (next) {
    ierr = SNESSetFromOptions(next->snes);CHKERRQ(ierr);
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_Composite"
static PetscErrorCode SNESReset_Composite(SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  PetscErrorCode   ierr;
  SNES_CompositeLink next = jac->head;

  PetscFunctionBegin;
  while (next) {
    ierr = SNESReset(next->snes);CHKERRQ(ierr);
    next = next->next;
  }
  ierr = VecDestroy(&jac->Xorig);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_Composite"
static PetscErrorCode SNESDestroy_Composite(SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  PetscErrorCode   ierr;
  SNES_CompositeLink next = jac->head,next_tmp;

  PetscFunctionBegin;
  ierr = SNESReset_Composite(snes);CHKERRQ(ierr);
  while (next) {
    ierr     = SNESDestroy(&next->snes);CHKERRQ(ierr);
    next_tmp = next;
    next     = next->next;
    ierr     = PetscFree(next_tmp);CHKERRQ(ierr);
  }
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_Composite"
static PetscErrorCode SNESSetFromOptions_Composite(SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  PetscErrorCode     ierr;
  PetscInt           nmax = 8,i;
  SNES_CompositeLink next;
  char               *sneses[8];
  PetscReal          dmps[8];
  PetscBool          flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Composite preconditioner options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_composite_type","Type of composition","SNESCompositeSetType",SNESCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&jac->type,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESCompositeSetType(snes,jac->type);CHKERRQ(ierr);
  }
  ierr = PetscOptionsStringArray("-snes_composite_sneses","List of composite solvers","SNESCompositeAddSNES",sneses,&nmax,&flg);CHKERRQ(ierr);
  if (flg) {
    for (i=0; i<nmax; i++) {
      ierr = SNESCompositeAddSNES(snes,sneses[i]);CHKERRQ(ierr);
      ierr = PetscFree(sneses[i]);CHKERRQ(ierr);   /* deallocate string sneses[i], which is allocated in PetscOptionsStringArray() */
    }
  }
  ierr = PetscOptionsRealArray("-snes_composite_damping","Damping of the additive composite solvers","SNESCompositeSetDamping",dmps,&nmax,&flg);CHKERRQ(ierr);
  if (flg) {
    for (i=0; i<nmax; i++) {
      ierr = SNESCompositeSetDamping(snes,i,dmps[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  next = jac->head;
  while (next) {
    ierr = SNESSetFromOptions(next->snes);CHKERRQ(ierr);
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_Composite"
static PetscErrorCode SNESView_Composite(SNES snes,PetscViewer viewer)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  PetscErrorCode   ierr;
  SNES_CompositeLink next = jac->head;
  PetscBool        iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Composite SNES type - %s\n",SNESCompositeTypes[jac->type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"SNESes on composite preconditioner follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  }
  if (iascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  }
  while (next) {
    ierr = SNESView(next->snes,viewer);CHKERRQ(ierr);
    next = next->next;
  }
  if (iascii) {
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "SNESCompositeSetType_Composite"
static PetscErrorCode  SNESCompositeSetType_Composite(SNES snes,SNESCompositeType type)
{
  SNES_Composite *jac = (SNES_Composite*)snes->data;

  PetscFunctionBegin;
  jac->type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESCompositeAddSNES_Composite"
static PetscErrorCode  SNESCompositeAddSNES_Composite(SNES snes,SNESType type)
{
  SNES_Composite     *jac;
  SNES_CompositeLink next,ilink;
  PetscErrorCode   ierr;
  PetscInt         cnt = 0;
  const char       *prefix;
  char             newprefix[8];
  DM               dm;

  PetscFunctionBegin;
  ierr        = PetscNewLog(snes,struct _SNES_CompositeLink,&ilink);CHKERRQ(ierr);
  ilink->next = 0;
  ierr        = SNESCreate(PetscObjectComm((PetscObject)snes),&ilink->snes);CHKERRQ(ierr);
  ierr        = PetscLogObjectParent((PetscObject)snes,(PetscObject)ilink->snes);CHKERRQ(ierr);
  ierr        = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr        = SNESSetDM(ilink->snes,dm);CHKERRQ(ierr);

  jac  = (SNES_Composite*)snes->data;
  next = jac->head;
  if (!next) {
    jac->head       = ilink;
    ilink->previous = NULL;
  } else {
    cnt++;
    while (next->next) {
      next = next->next;
      cnt++;
    }
    next->next      = ilink;
    ilink->previous = next;
  }
  ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(ilink->snes,prefix);CHKERRQ(ierr);
  sprintf(newprefix,"sub_%d_",(int)cnt);
  ierr = SNESAppendOptionsPrefix(ilink->snes,newprefix);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ilink->snes,(PetscObject)snes,1);CHKERRQ(ierr);
  ierr = SNESSetType(ilink->snes,type);CHKERRQ(ierr);
  ilink->dmp = 1.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESCompositeGetSNES_Composite"
static PetscErrorCode  SNESCompositeGetSNES_Composite(SNES snes,PetscInt n,SNES *subsnes)
{
  SNES_Composite     *jac;
  SNES_CompositeLink next;
  PetscInt         i;

  PetscFunctionBegin;
  jac  = (SNES_Composite*)snes->data;
  next = jac->head;
  for (i=0; i<n; i++) {
    if (!next->next) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Not enough SNESes in composite preconditioner");
    next = next->next;
  }
  *subsnes = next->snes;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SNESCompositeSetType"
/*@
   SNESCompositeSetType - Sets the type of composite preconditioner.

   Logically Collective on SNES

   Input Parameter:
+  snes - the preconditioner context
-  type - SNES_COMPOSITE_ADDITIVE (default), SNES_COMPOSITE_MULTIPLICATIVE

   Options Database Key:
.  -snes_composite_type <type: one of multiplicative, additive, special> - Sets composite preconditioner type

   Level: Developer

.keywords: SNES, set, type, composite preconditioner, additive, multiplicative
@*/
PetscErrorCode  SNESCompositeSetType(SNES snes,SNESCompositeType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveEnum(snes,type,2);
  ierr = PetscTryMethod(snes,"SNESCompositeSetType_C",(SNES,SNESCompositeType),(snes,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESCompositeAddSNES"
/*@C
   SNESCompositeAddSNES - Adds another SNES to the composite SNES.

   Collective on SNES

   Input Parameters:
+  snes - the preconditioner context
-  type - the type of the new preconditioner

   Level: Developer

.keywords: SNES, composite preconditioner, add
@*/
PetscErrorCode  SNESCompositeAddSNES(SNES snes,SNESType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESCompositeAddSNES_C",(SNES,SNESType),(snes,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESCompositeGetSNES"
/*@
   SNESCompositeGetSNES - Gets one of the SNES objects in the composite SNES.

   Not Collective

   Input Parameter:
+  snes - the preconditioner context
-  n - the number of the snes requested

   Output Parameters:
.  subsnes - the SNES requested

   Level: Developer

.keywords: SNES, get, composite preconditioner, sub preconditioner

.seealso: SNESCompositeAddSNES()
@*/
PetscErrorCode  SNESCompositeGetSNES(SNES snes,PetscInt n,SNES *subsnes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(subsnes,3);
  ierr = PetscUseMethod(snes,"SNESCompositeGetSNES_C",(SNES,PetscInt,SNES*),(snes,n,subsnes));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESCompositeSetDamping_Composite"
static PetscErrorCode  SNESCompositeSetDamping_Composite(SNES snes,PetscInt n,PetscReal dmp)
{
  SNES_Composite     *jac;
  SNES_CompositeLink next;
  PetscInt         i;

  PetscFunctionBegin;
  jac  = (SNES_Composite*)snes->data;
  next = jac->head;
  for (i=0; i<n; i++) {
    if (!next->next) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Not enough SNESes in composite preconditioner");
    next = next->next;
  }
  next->dmp = dmp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESCompositeSetDamping"
/*@
   SNESCompositeSetDamping - Sets the damping of a subsolver when using additive composite SNES.

   Not Collective

   Input Parameter:
+  snes - the preconditioner context
.  n - the number of the snes requested
-  dmp - the damping

   Level: Developer

.keywords: SNES, get, composite preconditioner, sub preconditioner

.seealso: SNESCompositeAddSNES()
@*/
PetscErrorCode  SNESCompositeSetDamping(SNES snes,PetscInt n,PetscReal dmp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscUseMethod(snes,"SNESCompositeSetDamping_C",(SNES,PetscInt,PetscReal),(snes,n,dmp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_Composite"
PetscErrorCode SNESSolve_Composite(SNES snes)
{
  Vec            F;
  Vec            X;
  Vec            B;
  PetscInt       i;
  PetscReal      fnorm = 0.0;
  PetscErrorCode ierr;
  SNESNormType   normtype;
  SNES_Composite *comp = (SNES_Composite*)snes->data;

  PetscFunctionBegin;
  X = snes->vec_sol;
  F = snes->vec_func;
  B = snes->vec_rhs;

  ierr         = PetscObjectAMSTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter   = 0;
  snes->norm   = 0.;
  ierr         = PetscObjectAMSGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITERATING;
  ierr         = SNESGetNormType(snes, &normtype);CHKERRQ(ierr);
  if (normtype == SNES_NORM_FUNCTION || normtype == SNES_NORM_INITIAL_ONLY || normtype == SNES_NORM_INITIAL_FINAL_ONLY) {
    if (!snes->vec_func_init_set) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
    } else snes->vec_func_init_set = PETSC_FALSE;

    /* convergence test */
    if (!snes->norm_init_set) {
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) {
        snes->reason = SNES_DIVERGED_FNORM_NAN;
        PetscFunctionReturn(0);
      }
    } else {
      fnorm               = snes->norm_init;
      snes->norm_init_set = PETSC_FALSE;
    }
    ierr       = PetscObjectAMSTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = 0;
    snes->norm = fnorm;
    ierr       = PetscObjectAMSGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);

    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;

    /* test convergence */
    ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    ierr = PetscObjectAMSGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
  }

  for (i = 0; i < snes->max_its; i++) {
    if (comp->type == SNES_COMPOSITE_ADDITIVE) {
      ierr = SNESCompositeApply_Additive(snes,X,B,F,&fnorm);CHKERRQ(ierr);
    } else if (comp->type == SNES_COMPOSITE_MULTIPLICATIVE) {
      ierr = SNESCompositeApply_Multiplicative(snes,X,B,F,&fnorm);CHKERRQ(ierr);
    } else {
    }
    if ((i == snes->max_its - 1) && (normtype == SNES_NORM_INITIAL_FINAL_ONLY || normtype == SNES_NORM_FINAL_ONLY)) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        break;
      }
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) {
        snes->reason = SNES_DIVERGED_FNORM_NAN;
        break;
      }
    }
    /* Monitor convergence */
    ierr       = PetscObjectAMSTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr       = PetscObjectAMSGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    if (normtype == SNES_NORM_FUNCTION) {ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);}
    if (snes->reason) break;
    /* Call general purpose update function */
    if (snes->ops->update) {ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);}
  }
  if (normtype == SNES_NORM_FUNCTION) {
    if (i == snes->max_its) {
      ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
      if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
    }
  } else if (!snes->reason) snes->reason = SNES_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/

/*MC
     SNESCOMPOSITE - Build a preconditioner by composing together several nonlinear solvers

   Options Database Keys:
+  -snes_composite_type <type: one of multiplicative, additive, symmetric_multiplicative, special> - Sets composite preconditioner type
-  -snes_composite_sneses - <snes0,snes1,...> list of SNESes to compose

   Level: intermediate

   Concepts: composing solvers

.seealso:  SNESCreate(), SNESSetType(), SNESType (for list of available types), SNES,
           SNESSHELL, SNESCompositeSetType(), SNESCompositeSpecialSetAlpha(), SNESCompositeAddSNES(),
           SNESCompositeGetSNES()

M*/

#undef __FUNCT__
#define __FUNCT__ "SNESCreate_Composite"
PETSC_EXTERN PetscErrorCode SNESCreate_Composite(SNES snes)
{
  PetscErrorCode ierr;
  SNES_Composite   *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(snes,SNES_Composite,&jac);CHKERRQ(ierr);

  snes->ops->solve           = SNESSolve_Composite;
  snes->ops->setup           = SNESSetUp_Composite;
  snes->ops->reset           = SNESReset_Composite;
  snes->ops->destroy         = SNESDestroy_Composite;
  snes->ops->setfromoptions  = SNESSetFromOptions_Composite;
  snes->ops->view            = SNESView_Composite;

  snes->data = (void*)jac;
  jac->type  = SNES_COMPOSITE_ADDITIVE;
  jac->head  = 0;

  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetType_C",SNESCompositeSetType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeAddSNES_C",SNESCompositeAddSNES_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeGetSNES_C",SNESCompositeGetSNES_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetDamping_C",SNESCompositeSetDamping_Composite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

