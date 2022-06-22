
/*
      Defines a SNES that can consist of a collection of SNESes
*/
#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petscblaslapack.h>

const char *const        SNESCompositeTypes[]   = {"ADDITIVE","MULTIPLICATIVE","ADDITIVEOPTIMAL","SNESCompositeType","SNES_COMPOSITE",NULL};

typedef struct _SNES_CompositeLink *SNES_CompositeLink;
struct _SNES_CompositeLink {
  SNES               snes;
  PetscReal          dmp;
  Vec                X;
  SNES_CompositeLink next;
  SNES_CompositeLink previous;
};

typedef struct {
  SNES_CompositeLink head;
  PetscInt           nsnes;
  SNESCompositeType  type;
  Vec                Xorig;
  PetscInt           innerFailures; /* the number of inner failures we've seen */

  /* context for ADDITIVEOPTIMAL */
  Vec                *Xes,*Fes;      /* solution and residual vectors for the subsolvers */
  PetscReal          *fnorms;        /* norms of the solutions */
  PetscScalar        *h;             /* the matrix formed as q_ij = (rdot_i, rdot_j) */
  PetscScalar        *g;             /* the dotproducts of the previous function with the candidate functions */
  PetscBLASInt       n;              /* matrix dimension -- nsnes */
  PetscBLASInt       nrhs;           /* the number of right hand sides */
  PetscBLASInt       lda;            /* the padded matrix dimension */
  PetscBLASInt       ldb;            /* the padded vector dimension */
  PetscReal          *s;             /* the singular values */
  PetscScalar        *beta;          /* the RHS and combination */
  PetscReal          rcond;          /* the exit condition */
  PetscBLASInt       rank;           /* the effective rank */
  PetscScalar        *work;          /* the work vector */
  PetscReal          *rwork;         /* the real work vector used for complex */
  PetscBLASInt       lwork;          /* the size of the work vector */
  PetscBLASInt       info;           /* the output condition */

  PetscReal          rtol;           /* restart tolerance for accepting the combination */
  PetscReal          stol;           /* restart tolerance for the combination */
} SNES_Composite;

static PetscErrorCode SNESCompositeApply_Multiplicative(SNES snes,Vec X,Vec B,Vec F,PetscReal *fnorm)
{
  SNES_Composite      *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink  next = jac->head;
  Vec                 FSub;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  PetscCheck(next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");
  if (snes->normschedule == SNES_NORM_ALWAYS) {
    PetscCall(SNESSetInitialFunction(next->snes,F));
  }
  PetscCall(SNESSolve(next->snes,B,X));
  PetscCall(SNESGetConvergedReason(next->snes,&reason));
  if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
    jac->innerFailures++;
    if (jac->innerFailures >= snes->maxFailures) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }

  while (next->next) {
    /* only copy the function over in the case where the functions correspond */
    if (next->snes->npcside== PC_RIGHT && next->snes->normschedule != SNES_NORM_NONE) {
      PetscCall(SNESGetFunction(next->snes,&FSub,NULL,NULL));
      next = next->next;
      PetscCall(SNESSetInitialFunction(next->snes,FSub));
    } else {
      next = next->next;
    }
    PetscCall(SNESSolve(next->snes,B,X));
    PetscCall(SNESGetConvergedReason(next->snes,&reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      jac->innerFailures++;
      if (jac->innerFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    }
  }
  if (next->snes->npcside== PC_RIGHT) {
    PetscCall(SNESGetFunction(next->snes,&FSub,NULL,NULL));
    PetscCall(VecCopy(FSub,F));
    if (fnorm) {
      if (snes->xl && snes->xu) {
        PetscCall(SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm));
      } else {
        PetscCall(VecNorm(F, NORM_2, fnorm));
      }
      SNESCheckFunctionNorm(snes,*fnorm);
    }
  } else if (snes->normschedule == SNES_NORM_ALWAYS) {
    PetscCall(SNESComputeFunction(snes,X,F));
    if (fnorm) {
      if (snes->xl && snes->xu) {
        PetscCall(SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm));
      } else {
        PetscCall(VecNorm(F, NORM_2, fnorm));
      }
      SNESCheckFunctionNorm(snes,*fnorm);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESCompositeApply_Additive(SNES snes,Vec X,Vec B,Vec F,PetscReal *fnorm)
{
  SNES_Composite      *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink  next = jac->head;
  Vec                 Y,Xorig;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  Y = snes->vec_sol_update;
  if (!jac->Xorig) PetscCall(VecDuplicate(X,&jac->Xorig));
  Xorig = jac->Xorig;
  PetscCall(VecCopy(X,Xorig));
  PetscCheck(next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");
  if (snes->normschedule == SNES_NORM_ALWAYS) {
    PetscCall(SNESSetInitialFunction(next->snes,F));
    while (next->next) {
      next = next->next;
      PetscCall(SNESSetInitialFunction(next->snes,F));
    }
  }
  next = jac->head;
  PetscCall(VecCopy(Xorig,Y));
  PetscCall(SNESSolve(next->snes,B,Y));
  PetscCall(SNESGetConvergedReason(next->snes,&reason));
  if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
    jac->innerFailures++;
    if (jac->innerFailures >= snes->maxFailures) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }
  PetscCall(VecAXPY(Y,-1.0,Xorig));
  PetscCall(VecAXPY(X,next->dmp,Y));
  while (next->next) {
    next = next->next;
    PetscCall(VecCopy(Xorig,Y));
    PetscCall(SNESSolve(next->snes,B,Y));
    PetscCall(SNESGetConvergedReason(next->snes,&reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      jac->innerFailures++;
      if (jac->innerFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    }
    PetscCall(VecAXPY(Y,-1.0,Xorig));
    PetscCall(VecAXPY(X,next->dmp,Y));
  }
  if (snes->normschedule == SNES_NORM_ALWAYS) {
    PetscCall(SNESComputeFunction(snes,X,F));
    if (fnorm) {
      if (snes->xl && snes->xu) {
        PetscCall(SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm));
      } else {
        PetscCall(VecNorm(F, NORM_2, fnorm));
      }
      SNESCheckFunctionNorm(snes,*fnorm);
    }
  }
  PetscFunctionReturn(0);
}

/* approximately solve the overdetermined system:

 2*F(x_i)\cdot F(\x_j)\alpha_i = 0
 \alpha_i                      = 1

 Which minimizes the L2 norm of the linearization of:
 ||F(\sum_i \alpha_i*x_i)||^2

 With the constraint that \sum_i\alpha_i = 1
 Where x_i is the solution from the ith subsolver.
 */
static PetscErrorCode SNESCompositeApply_AdditiveOptimal(SNES snes,Vec X,Vec B,Vec F,PetscReal *fnorm)
{
  SNES_Composite      *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink  next = jac->head;
  Vec                 *Xes = jac->Xes,*Fes = jac->Fes;
  PetscInt            i,j;
  PetscScalar         tot,total,ftf;
  PetscReal           min_fnorm;
  PetscInt            min_i;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  PetscCheck(next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");

  if (snes->normschedule == SNES_NORM_ALWAYS) {
    next = jac->head;
    PetscCall(SNESSetInitialFunction(next->snes,F));
    while (next->next) {
      next = next->next;
      PetscCall(SNESSetInitialFunction(next->snes,F));
    }
  }

  next = jac->head;
  i = 0;
  PetscCall(VecCopy(X,Xes[i]));
  PetscCall(SNESSolve(next->snes,B,Xes[i]));
  PetscCall(SNESGetConvergedReason(next->snes,&reason));
  if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
    jac->innerFailures++;
    if (jac->innerFailures >= snes->maxFailures) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }
  while (next->next) {
    i++;
    next = next->next;
    PetscCall(VecCopy(X,Xes[i]));
    PetscCall(SNESSolve(next->snes,B,Xes[i]));
    PetscCall(SNESGetConvergedReason(next->snes,&reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      jac->innerFailures++;
      if (jac->innerFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    }
  }

  /* all the solutions are collected; combine optimally */
  for (i=0;i<jac->n;i++) {
    for (j=0;j<i+1;j++) {
      PetscCall(VecDotBegin(Fes[i],Fes[j],&jac->h[i + j*jac->n]));
    }
    PetscCall(VecDotBegin(Fes[i],F,&jac->g[i]));
  }

  for (i=0;i<jac->n;i++) {
    for (j=0;j<i+1;j++) {
      PetscCall(VecDotEnd(Fes[i],Fes[j],&jac->h[i + j*jac->n]));
      if (i == j) jac->fnorms[i] = PetscSqrtReal(PetscRealPart(jac->h[i + j*jac->n]));
    }
    PetscCall(VecDotEnd(Fes[i],F,&jac->g[i]));
  }

  ftf = (*fnorm)*(*fnorm);

  for (i=0; i<jac->n; i++) {
    for (j=i+1;j<jac->n;j++) {
      jac->h[i + j*jac->n] = jac->h[j + i*jac->n];
    }
  }

  for (i=0; i<jac->n; i++) {
    for (j=0; j<jac->n; j++) {
      jac->h[i + j*jac->n] = jac->h[i + j*jac->n] - jac->g[j] - jac->g[i] + ftf;
    }
    jac->beta[i] = ftf - jac->g[i];
  }

  jac->info  = 0;
  jac->rcond = -1.;
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&jac->n,&jac->n,&jac->nrhs,jac->h,&jac->lda,jac->beta,&jac->lda,jac->s,&jac->rcond,&jac->rank,jac->work,&jac->lwork,jac->rwork,&jac->info));
#else
  PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&jac->n,&jac->n,&jac->nrhs,jac->h,&jac->lda,jac->beta,&jac->lda,jac->s,&jac->rcond,&jac->rank,jac->work,&jac->lwork,&jac->info));
#endif
  PetscCall(PetscFPTrapPop());
  PetscCheck(jac->info >= 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"Bad argument to GELSS");
  PetscCheck(jac->info <= 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD failed to converge");
  tot = 0.;
  total = 0.;
  for (i=0; i<jac->n; i++) {
    PetscCheck(!snes->errorifnotconverged || !PetscIsInfOrNanScalar(jac->beta[i]),PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD generated inconsistent output");
    PetscCall(PetscInfo(snes,"%" PetscInt_FMT ": %g\n",i,(double)PetscRealPart(jac->beta[i])));
    tot += jac->beta[i];
    total += PetscAbsScalar(jac->beta[i]);
  }
  PetscCall(VecScale(X,(1. - tot)));
  PetscCall(VecMAXPY(X,jac->n,jac->beta,Xes));
  PetscCall(SNESComputeFunction(snes,X,F));

  if (snes->xl && snes->xu) {
    PetscCall(SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm));
  } else {
    PetscCall(VecNorm(F, NORM_2, fnorm));
  }

  /* take the minimum-normed candidate if it beats the combination by a factor of rtol or the combination has stagnated */
  min_fnorm = jac->fnorms[0];
  min_i     = 0;
  for (i=0; i<jac->n; i++) {
    if (jac->fnorms[i] < min_fnorm) {
      min_fnorm = jac->fnorms[i];
      min_i     = i;
    }
  }

  /* stagnation or divergence restart to the solution of the solver that failed the least */
  if (PetscRealPart(total) < jac->stol || min_fnorm*jac->rtol < *fnorm) {
    PetscCall(VecCopy(jac->Xes[min_i],X));
    PetscCall(VecCopy(jac->Fes[min_i],F));
    *fnorm = min_fnorm;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_Composite(SNES snes)
{
  DM                 dm;
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head;
  PetscInt           n=0,i;
  Vec                F;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&dm));

  if (snes->ops->computevariablebounds) {
    /* SNESVI only ever calls computevariablebounds once, so calling it once here is justified */
    if (!snes->xl) PetscCall(VecDuplicate(snes->vec_sol,&snes->xl));
    if (!snes->xu) PetscCall(VecDuplicate(snes->vec_sol,&snes->xu));
    PetscCall((*snes->ops->computevariablebounds)(snes,snes->xl,snes->xu));
  }

  while (next) {
    n++;
    PetscCall(SNESSetDM(next->snes,dm));
    PetscCall(SNESSetApplicationContext(next->snes, snes->user));
    if (snes->xl && snes->xu) {
      if (snes->ops->computevariablebounds) {
        PetscCall(SNESVISetComputeVariableBounds(next->snes, snes->ops->computevariablebounds));
      } else {
        PetscCall(SNESVISetVariableBounds(next->snes,snes->xl,snes->xu));
      }
    }

    next = next->next;
  }
  jac->nsnes = n;
  PetscCall(SNESGetFunction(snes,&F,NULL,NULL));
  if (jac->type == SNES_COMPOSITE_ADDITIVEOPTIMAL) {
    PetscCall(VecDuplicateVecs(F,jac->nsnes,&jac->Xes));
    PetscCall(PetscMalloc1(n,&jac->Fes));
    PetscCall(PetscMalloc1(n,&jac->fnorms));
    next = jac->head;
    i = 0;
    while (next) {
      PetscCall(SNESGetFunction(next->snes,&F,NULL,NULL));
      jac->Fes[i] = F;
      PetscCall(PetscObjectReference((PetscObject)F));
      next = next->next;
      i++;
    }
    /* allocate the subspace direct solve area */
    jac->nrhs  = 1;
    jac->lda   = jac->nsnes;
    jac->ldb   = jac->nsnes;
    jac->n     = jac->nsnes;

    PetscCall(PetscMalloc1(jac->n*jac->n,&jac->h));
    PetscCall(PetscMalloc1(jac->n,&jac->beta));
    PetscCall(PetscMalloc1(jac->n,&jac->s));
    PetscCall(PetscMalloc1(jac->n,&jac->g));
    jac->lwork = 12*jac->n;
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscMalloc1(jac->lwork,&jac->rwork));
#endif
    PetscCall(PetscMalloc1(jac->lwork,&jac->work));
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode SNESReset_Composite(SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head;

  PetscFunctionBegin;
  while (next) {
    PetscCall(SNESReset(next->snes));
    next = next->next;
  }
  PetscCall(VecDestroy(&jac->Xorig));
  if (jac->Xes) PetscCall(VecDestroyVecs(jac->nsnes,&jac->Xes));
  if (jac->Fes) PetscCall(VecDestroyVecs(jac->nsnes,&jac->Fes));
  PetscCall(PetscFree(jac->fnorms));
  PetscCall(PetscFree(jac->h));
  PetscCall(PetscFree(jac->s));
  PetscCall(PetscFree(jac->g));
  PetscCall(PetscFree(jac->beta));
  PetscCall(PetscFree(jac->work));
  PetscCall(PetscFree(jac->rwork));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_Composite(SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head,next_tmp;

  PetscFunctionBegin;
  PetscCall(SNESReset_Composite(snes));
  while (next) {
    PetscCall(SNESDestroy(&next->snes));
    next_tmp = next;
    next     = next->next;
    PetscCall(PetscFree(next_tmp));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeAddSNES_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeGetSNES_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetDamping_C",NULL));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_Composite(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  PetscInt           nmax = 8,i;
  SNES_CompositeLink next;
  char               *sneses[8];
  PetscReal          dmps[8];
  PetscBool          flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Composite preconditioner options");
  PetscCall(PetscOptionsEnum("-snes_composite_type","Type of composition","SNESCompositeSetType",SNESCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&jac->type,&flg));
  if (flg) {
    PetscCall(SNESCompositeSetType(snes,jac->type));
  }
  PetscCall(PetscOptionsStringArray("-snes_composite_sneses","List of composite solvers","SNESCompositeAddSNES",sneses,&nmax,&flg));
  if (flg) {
    for (i=0; i<nmax; i++) {
      PetscCall(SNESCompositeAddSNES(snes,sneses[i]));
      PetscCall(PetscFree(sneses[i]));   /* deallocate string sneses[i], which is allocated in PetscOptionsStringArray() */
    }
  }
  PetscCall(PetscOptionsRealArray("-snes_composite_damping","Damping of the additive composite solvers","SNESCompositeSetDamping",dmps,&nmax,&flg));
  if (flg) {
    for (i=0; i<nmax; i++) {
      PetscCall(SNESCompositeSetDamping(snes,i,dmps[i]));
    }
  }
  PetscCall(PetscOptionsReal("-snes_composite_stol","Step tolerance for restart on the additive composite solvers","",jac->stol,&jac->stol,NULL));
  PetscCall(PetscOptionsReal("-snes_composite_rtol","Residual tolerance for the additive composite solvers","",jac->rtol,&jac->rtol,NULL));
  PetscOptionsHeadEnd();

  next = jac->head;
  while (next) {
    PetscCall(SNESSetFromOptions(next->snes));
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_Composite(SNES snes,PetscViewer viewer)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head;
  PetscBool          iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  type - %s\n",SNESCompositeTypes[jac->type]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  SNESes on composite preconditioner follow\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n"));
  }
  if (iascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
  }
  while (next) {
    PetscCall(SNESView(next->snes,viewer));
    next = next->next;
  }
  if (iascii) {
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n"));
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

static PetscErrorCode  SNESCompositeSetType_Composite(SNES snes,SNESCompositeType type)
{
  SNES_Composite *jac = (SNES_Composite*)snes->data;

  PetscFunctionBegin;
  jac->type = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode  SNESCompositeAddSNES_Composite(SNES snes,SNESType type)
{
  SNES_Composite     *jac;
  SNES_CompositeLink next,ilink;
  PetscInt           cnt = 0;
  const char         *prefix;
  char               newprefix[20];
  DM                 dm;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(snes,&ilink));
  ilink->next = NULL;
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)snes),&ilink->snes));
  PetscCall(PetscLogObjectParent((PetscObject)snes,(PetscObject)ilink->snes));
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(SNESSetDM(ilink->snes,dm));
  PetscCall(SNESSetTolerances(ilink->snes,snes->abstol,snes->rtol,snes->stol,1,snes->max_funcs));
  PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)snes,(PetscObject)ilink->snes));
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
  PetscCall(SNESGetOptionsPrefix(snes,&prefix));
  PetscCall(SNESSetOptionsPrefix(ilink->snes,prefix));
  PetscCall(PetscSNPrintf(newprefix,sizeof(newprefix),"sub_%d_",(int)cnt));
  PetscCall(SNESAppendOptionsPrefix(ilink->snes,newprefix));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)ilink->snes,(PetscObject)snes,1));
  PetscCall(SNESSetType(ilink->snes,type));
  PetscCall(SNESSetNormSchedule(ilink->snes, SNES_NORM_FINAL_ONLY));

  ilink->dmp = 1.0;
  jac->nsnes++;
  PetscFunctionReturn(0);
}

static PetscErrorCode  SNESCompositeGetSNES_Composite(SNES snes,PetscInt n,SNES *subsnes)
{
  SNES_Composite     *jac;
  SNES_CompositeLink next;
  PetscInt           i;

  PetscFunctionBegin;
  jac  = (SNES_Composite*)snes->data;
  next = jac->head;
  for (i=0; i<n; i++) {
    PetscCheck(next->next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Not enough SNESes in composite preconditioner");
    next = next->next;
  }
  *subsnes = next->snes;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */
/*@C
   SNESCompositeSetType - Sets the type of composite preconditioner.

   Logically Collective on SNES

   Input Parameters:
+  snes - the preconditioner context
-  type - SNES_COMPOSITE_ADDITIVE (default), SNES_COMPOSITE_MULTIPLICATIVE

   Options Database Key:
.  -snes_composite_type <type: one of multiplicative, additive, special> - Sets composite preconditioner type

   Level: Developer

@*/
PetscErrorCode  SNESCompositeSetType(SNES snes,SNESCompositeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveEnum(snes,type,2);
  PetscTryMethod(snes,"SNESCompositeSetType_C",(SNES,SNESCompositeType),(snes,type));
  PetscFunctionReturn(0);
}

/*@C
   SNESCompositeAddSNES - Adds another SNES to the composite SNES.

   Collective on SNES

   Input Parameters:
+  snes - the preconditioner context
-  type - the type of the new preconditioner

   Level: Developer

@*/
PetscErrorCode  SNESCompositeAddSNES(SNES snes,SNESType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscTryMethod(snes,"SNESCompositeAddSNES_C",(SNES,SNESType),(snes,type));
  PetscFunctionReturn(0);
}

/*@
   SNESCompositeGetSNES - Gets one of the SNES objects in the composite SNES.

   Not Collective

   Input Parameters:
+  snes - the preconditioner context
-  n - the number of the snes requested

   Output Parameters:
.  subsnes - the SNES requested

   Level: Developer

.seealso: `SNESCompositeAddSNES()`
@*/
PetscErrorCode  SNESCompositeGetSNES(SNES snes,PetscInt n,SNES *subsnes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(subsnes,3);
  PetscUseMethod(snes,"SNESCompositeGetSNES_C",(SNES,PetscInt,SNES*),(snes,n,subsnes));
  PetscFunctionReturn(0);
}

/*@
   SNESCompositeGetNumber - Get the number of subsolvers in the composite SNES.

   Logically Collective on SNES

   Input Parameter:
   snes - the preconditioner context

   Output Parameter:
   n - the number of subsolvers

   Level: Developer

@*/
PetscErrorCode  SNESCompositeGetNumber(SNES snes,PetscInt *n)
{
  SNES_Composite     *jac;
  SNES_CompositeLink next;

  PetscFunctionBegin;
  jac  = (SNES_Composite*)snes->data;
  next = jac->head;

  *n = 0;
  while (next) {
    *n = *n + 1;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  SNESCompositeSetDamping_Composite(SNES snes,PetscInt n,PetscReal dmp)
{
  SNES_Composite     *jac;
  SNES_CompositeLink next;
  PetscInt           i;

  PetscFunctionBegin;
  jac  = (SNES_Composite*)snes->data;
  next = jac->head;
  for (i=0; i<n; i++) {
    PetscCheck(next->next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Not enough SNESes in composite preconditioner");
    next = next->next;
  }
  next->dmp = dmp;
  PetscFunctionReturn(0);
}

/*@
   SNESCompositeSetDamping - Sets the damping of a subsolver when using additive composite SNES.

   Not Collective

   Input Parameters:
+  snes - the preconditioner context
.  n - the number of the snes requested
-  dmp - the damping

   Level: Developer

.seealso: `SNESCompositeAddSNES()`
@*/
PetscErrorCode  SNESCompositeSetDamping(SNES snes,PetscInt n,PetscReal dmp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscUseMethod(snes,"SNESCompositeSetDamping_C",(SNES,PetscInt,PetscReal),(snes,n,dmp));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_Composite(SNES snes)
{
  Vec              F,X,B,Y;
  PetscInt         i;
  PetscReal        fnorm = 0.0, xnorm = 0.0, snorm = 0.0;
  SNESNormSchedule normtype;
  SNES_Composite   *comp = (SNES_Composite*)snes->data;

  PetscFunctionBegin;
  X = snes->vec_sol;
  F = snes->vec_func;
  B = snes->vec_rhs;
  Y = snes->vec_sol_update;

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter   = 0;
  snes->norm   = 0.;
  comp->innerFailures = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  snes->reason = SNES_CONVERGED_ITERATING;
  PetscCall(SNESGetNormSchedule(snes, &normtype));
  if (normtype == SNES_NORM_ALWAYS || normtype == SNES_NORM_INITIAL_ONLY || normtype == SNES_NORM_INITIAL_FINAL_ONLY) {
    if (!snes->vec_func_init_set) {
      PetscCall(SNESComputeFunction(snes,X,F));
    } else snes->vec_func_init_set = PETSC_FALSE;

    if (snes->xl && snes->xu) {
      PetscCall(SNESVIComputeInactiveSetFnorm(snes, F, X, &fnorm));
    } else {
      PetscCall(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- ||F||  */
    }
    SNESCheckFunctionNorm(snes,fnorm);
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = 0;
    snes->norm = fnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,0));
    PetscCall(SNESMonitor(snes,0,snes->norm));

    /* test convergence */
    PetscCall((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,0));
    PetscCall(SNESMonitor(snes,0,snes->norm));
  }

  for (i = 0; i < snes->max_its; i++) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      PetscCall((*snes->ops->update)(snes, snes->iter));
    }

    /* Copy the state before modification by application of the composite solver;
       we will subtract the new state after application */
    PetscCall(VecCopy(X, Y));

    if (comp->type == SNES_COMPOSITE_ADDITIVE) {
      PetscCall(SNESCompositeApply_Additive(snes,X,B,F,&fnorm));
    } else if (comp->type == SNES_COMPOSITE_MULTIPLICATIVE) {
      PetscCall(SNESCompositeApply_Multiplicative(snes,X,B,F,&fnorm));
    } else if (comp->type == SNES_COMPOSITE_ADDITIVEOPTIMAL) {
      PetscCall(SNESCompositeApply_AdditiveOptimal(snes,X,B,F,&fnorm));
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Unsupported SNESComposite type");
    if (snes->reason < 0) break;

    /* Compute the solution update for convergence testing */
    PetscCall(VecAYPX(Y, -1.0, X));

    if ((i == snes->max_its - 1) && (normtype == SNES_NORM_INITIAL_FINAL_ONLY || normtype == SNES_NORM_FINAL_ONLY)) {
      PetscCall(SNESComputeFunction(snes,X,F));

      if (snes->xl && snes->xu) {
        PetscCall(VecNormBegin(X, NORM_2, &xnorm));
        PetscCall(VecNormBegin(Y, NORM_2, &snorm));
        PetscCall(SNESVIComputeInactiveSetFnorm(snes, F, X, &fnorm));
        PetscCall(VecNormEnd(X, NORM_2, &xnorm));
        PetscCall(VecNormEnd(Y, NORM_2, &snorm));
      } else {
        PetscCall(VecNormBegin(F, NORM_2, &fnorm));
        PetscCall(VecNormBegin(X, NORM_2, &xnorm));
        PetscCall(VecNormBegin(Y, NORM_2, &snorm));

        PetscCall(VecNormEnd(F, NORM_2, &fnorm));
        PetscCall(VecNormEnd(X, NORM_2, &xnorm));
        PetscCall(VecNormEnd(Y, NORM_2, &snorm));
      }
      SNESCheckFunctionNorm(snes,fnorm);
    } else if (normtype == SNES_NORM_ALWAYS) {
      PetscCall(VecNormBegin(X, NORM_2, &xnorm));
      PetscCall(VecNormBegin(Y, NORM_2, &snorm));
      PetscCall(VecNormEnd(X, NORM_2, &xnorm));
      PetscCall(VecNormEnd(Y, NORM_2, &snorm));
    }
    /* Monitor convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = snorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,0));
    PetscCall(SNESMonitor(snes,snes->iter,snes->norm));
    /* Test for convergence */
    if (normtype == SNES_NORM_ALWAYS) PetscCall((*snes->ops->converged)(snes,snes->iter,xnorm,snorm,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) break;
  }
  if (normtype == SNES_NORM_ALWAYS) {
    if (i == snes->max_its) {
      PetscCall(PetscInfo(snes,"Maximum number of iterations has been reached: %" PetscInt_FMT "\n",snes->max_its));
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

.seealso: `SNESCreate()`, `SNESSetType()`, `SNESType`, `SNES`,
          `SNESSHELL`, `SNESCompositeSetType()`, `SNESCompositeSpecialSetAlpha()`, `SNESCompositeAddSNES()`,
          `SNESCompositeGetSNES()`

   References:
.  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

M*/

PETSC_EXTERN PetscErrorCode SNESCreate_Composite(SNES snes)
{
  SNES_Composite *jac;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(snes,&jac));

  snes->ops->solve           = SNESSolve_Composite;
  snes->ops->setup           = SNESSetUp_Composite;
  snes->ops->reset           = SNESReset_Composite;
  snes->ops->destroy         = SNESDestroy_Composite;
  snes->ops->setfromoptions  = SNESSetFromOptions_Composite;
  snes->ops->view            = SNESView_Composite;

  snes->usesksp        = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  snes->data = (void*)jac;
  jac->type  = SNES_COMPOSITE_ADDITIVEOPTIMAL;
  jac->Fes   = NULL;
  jac->Xes   = NULL;
  jac->fnorms = NULL;
  jac->nsnes = 0;
  jac->head  = NULL;
  jac->stol  = 0.1;
  jac->rtol  = 1.1;

  jac->h     = NULL;
  jac->s     = NULL;
  jac->beta  = NULL;
  jac->work  = NULL;
  jac->rwork = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetType_C",SNESCompositeSetType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeAddSNES_C",SNESCompositeAddSNES_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeGetSNES_C",SNESCompositeGetSNES_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetDamping_C",SNESCompositeSetDamping_Composite));
  PetscFunctionReturn(0);
}
