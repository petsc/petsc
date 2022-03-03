
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
    CHKERRQ(SNESSetInitialFunction(next->snes,F));
  }
  CHKERRQ(SNESSolve(next->snes,B,X));
  CHKERRQ(SNESGetConvergedReason(next->snes,&reason));
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
      CHKERRQ(SNESGetFunction(next->snes,&FSub,NULL,NULL));
      next = next->next;
      CHKERRQ(SNESSetInitialFunction(next->snes,FSub));
    } else {
      next = next->next;
    }
    CHKERRQ(SNESSolve(next->snes,B,X));
    CHKERRQ(SNESGetConvergedReason(next->snes,&reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      jac->innerFailures++;
      if (jac->innerFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    }
  }
  if (next->snes->npcside== PC_RIGHT) {
    CHKERRQ(SNESGetFunction(next->snes,&FSub,NULL,NULL));
    CHKERRQ(VecCopy(FSub,F));
    if (fnorm) {
      if (snes->xl && snes->xu) {
        CHKERRQ(SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm));
      } else {
        CHKERRQ(VecNorm(F, NORM_2, fnorm));
      }
      SNESCheckFunctionNorm(snes,*fnorm);
    }
  } else if (snes->normschedule == SNES_NORM_ALWAYS) {
    CHKERRQ(SNESComputeFunction(snes,X,F));
    if (fnorm) {
      if (snes->xl && snes->xu) {
        CHKERRQ(SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm));
      } else {
        CHKERRQ(VecNorm(F, NORM_2, fnorm));
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
  if (!jac->Xorig) CHKERRQ(VecDuplicate(X,&jac->Xorig));
  Xorig = jac->Xorig;
  CHKERRQ(VecCopy(X,Xorig));
  PetscCheck(next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");
  if (snes->normschedule == SNES_NORM_ALWAYS) {
    CHKERRQ(SNESSetInitialFunction(next->snes,F));
    while (next->next) {
      next = next->next;
      CHKERRQ(SNESSetInitialFunction(next->snes,F));
    }
  }
  next = jac->head;
  CHKERRQ(VecCopy(Xorig,Y));
  CHKERRQ(SNESSolve(next->snes,B,Y));
  CHKERRQ(SNESGetConvergedReason(next->snes,&reason));
  if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
    jac->innerFailures++;
    if (jac->innerFailures >= snes->maxFailures) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }
  CHKERRQ(VecAXPY(Y,-1.0,Xorig));
  CHKERRQ(VecAXPY(X,next->dmp,Y));
  while (next->next) {
    next = next->next;
    CHKERRQ(VecCopy(Xorig,Y));
    CHKERRQ(SNESSolve(next->snes,B,Y));
    CHKERRQ(SNESGetConvergedReason(next->snes,&reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      jac->innerFailures++;
      if (jac->innerFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    }
    CHKERRQ(VecAXPY(Y,-1.0,Xorig));
    CHKERRQ(VecAXPY(X,next->dmp,Y));
  }
  if (snes->normschedule == SNES_NORM_ALWAYS) {
    CHKERRQ(SNESComputeFunction(snes,X,F));
    if (fnorm) {
      if (snes->xl && snes->xu) {
        CHKERRQ(SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm));
      } else {
        CHKERRQ(VecNorm(F, NORM_2, fnorm));
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
    CHKERRQ(SNESSetInitialFunction(next->snes,F));
    while (next->next) {
      next = next->next;
      CHKERRQ(SNESSetInitialFunction(next->snes,F));
    }
  }

  next = jac->head;
  i = 0;
  CHKERRQ(VecCopy(X,Xes[i]));
  CHKERRQ(SNESSolve(next->snes,B,Xes[i]));
  CHKERRQ(SNESGetConvergedReason(next->snes,&reason));
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
    CHKERRQ(VecCopy(X,Xes[i]));
    CHKERRQ(SNESSolve(next->snes,B,Xes[i]));
    CHKERRQ(SNESGetConvergedReason(next->snes,&reason));
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
      CHKERRQ(VecDotBegin(Fes[i],Fes[j],&jac->h[i + j*jac->n]));
    }
    CHKERRQ(VecDotBegin(Fes[i],F,&jac->g[i]));
  }

  for (i=0;i<jac->n;i++) {
    for (j=0;j<i+1;j++) {
      CHKERRQ(VecDotEnd(Fes[i],Fes[j],&jac->h[i + j*jac->n]));
      if (i == j) jac->fnorms[i] = PetscSqrtReal(PetscRealPart(jac->h[i + j*jac->n]));
    }
    CHKERRQ(VecDotEnd(Fes[i],F,&jac->g[i]));
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
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&jac->n,&jac->n,&jac->nrhs,jac->h,&jac->lda,jac->beta,&jac->lda,jac->s,&jac->rcond,&jac->rank,jac->work,&jac->lwork,jac->rwork,&jac->info));
#else
  PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&jac->n,&jac->n,&jac->nrhs,jac->h,&jac->lda,jac->beta,&jac->lda,jac->s,&jac->rcond,&jac->rank,jac->work,&jac->lwork,&jac->info));
#endif
  CHKERRQ(PetscFPTrapPop());
  PetscCheckFalse(jac->info < 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"Bad argument to GELSS");
  PetscCheckFalse(jac->info > 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD failed to converge");
  tot = 0.;
  total = 0.;
  for (i=0; i<jac->n; i++) {
    PetscCheckFalse(snes->errorifnotconverged && PetscIsInfOrNanScalar(jac->beta[i]),PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD generated inconsistent output");
    CHKERRQ(PetscInfo(snes,"%D: %g\n",i,(double)PetscRealPart(jac->beta[i])));
    tot += jac->beta[i];
    total += PetscAbsScalar(jac->beta[i]);
  }
  CHKERRQ(VecScale(X,(1. - tot)));
  CHKERRQ(VecMAXPY(X,jac->n,jac->beta,Xes));
  CHKERRQ(SNESComputeFunction(snes,X,F));

  if (snes->xl && snes->xu) {
    CHKERRQ(SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm));
  } else {
    CHKERRQ(VecNorm(F, NORM_2, fnorm));
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
    CHKERRQ(VecCopy(jac->Xes[min_i],X));
    CHKERRQ(VecCopy(jac->Fes[min_i],F));
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
  CHKERRQ(SNESGetDM(snes,&dm));

  if (snes->ops->computevariablebounds) {
    /* SNESVI only ever calls computevariablebounds once, so calling it once here is justified */
    if (!snes->xl) CHKERRQ(VecDuplicate(snes->vec_sol,&snes->xl));
    if (!snes->xu) CHKERRQ(VecDuplicate(snes->vec_sol,&snes->xu));
    CHKERRQ((*snes->ops->computevariablebounds)(snes,snes->xl,snes->xu));
  }

  while (next) {
    n++;
    CHKERRQ(SNESSetDM(next->snes,dm));
    CHKERRQ(SNESSetApplicationContext(next->snes, snes->user));
    if (snes->xl && snes->xu) {
      if (snes->ops->computevariablebounds) {
        CHKERRQ(SNESVISetComputeVariableBounds(next->snes, snes->ops->computevariablebounds));
      } else {
        CHKERRQ(SNESVISetVariableBounds(next->snes,snes->xl,snes->xu));
      }
    }

    next = next->next;
  }
  jac->nsnes = n;
  CHKERRQ(SNESGetFunction(snes,&F,NULL,NULL));
  if (jac->type == SNES_COMPOSITE_ADDITIVEOPTIMAL) {
    CHKERRQ(VecDuplicateVecs(F,jac->nsnes,&jac->Xes));
    CHKERRQ(PetscMalloc1(n,&jac->Fes));
    CHKERRQ(PetscMalloc1(n,&jac->fnorms));
    next = jac->head;
    i = 0;
    while (next) {
      CHKERRQ(SNESGetFunction(next->snes,&F,NULL,NULL));
      jac->Fes[i] = F;
      CHKERRQ(PetscObjectReference((PetscObject)F));
      next = next->next;
      i++;
    }
    /* allocate the subspace direct solve area */
    jac->nrhs  = 1;
    jac->lda   = jac->nsnes;
    jac->ldb   = jac->nsnes;
    jac->n     = jac->nsnes;

    CHKERRQ(PetscMalloc1(jac->n*jac->n,&jac->h));
    CHKERRQ(PetscMalloc1(jac->n,&jac->beta));
    CHKERRQ(PetscMalloc1(jac->n,&jac->s));
    CHKERRQ(PetscMalloc1(jac->n,&jac->g));
    jac->lwork = 12*jac->n;
#if defined(PETSC_USE_COMPLEX)
    CHKERRQ(PetscMalloc1(jac->lwork,&jac->rwork));
#endif
    CHKERRQ(PetscMalloc1(jac->lwork,&jac->work));
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode SNESReset_Composite(SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head;

  PetscFunctionBegin;
  while (next) {
    CHKERRQ(SNESReset(next->snes));
    next = next->next;
  }
  CHKERRQ(VecDestroy(&jac->Xorig));
  if (jac->Xes) CHKERRQ(VecDestroyVecs(jac->nsnes,&jac->Xes));
  if (jac->Fes) CHKERRQ(VecDestroyVecs(jac->nsnes,&jac->Fes));
  CHKERRQ(PetscFree(jac->fnorms));
  CHKERRQ(PetscFree(jac->h));
  CHKERRQ(PetscFree(jac->s));
  CHKERRQ(PetscFree(jac->g));
  CHKERRQ(PetscFree(jac->beta));
  CHKERRQ(PetscFree(jac->work));
  CHKERRQ(PetscFree(jac->rwork));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_Composite(SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head,next_tmp;

  PetscFunctionBegin;
  CHKERRQ(SNESReset_Composite(snes));
  while (next) {
    CHKERRQ(SNESDestroy(&next->snes));
    next_tmp = next;
    next     = next->next;
    CHKERRQ(PetscFree(next_tmp));
  }
  CHKERRQ(PetscFree(snes->data));
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
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Composite preconditioner options"));
  CHKERRQ(PetscOptionsEnum("-snes_composite_type","Type of composition","SNESCompositeSetType",SNESCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&jac->type,&flg));
  if (flg) {
    CHKERRQ(SNESCompositeSetType(snes,jac->type));
  }
  CHKERRQ(PetscOptionsStringArray("-snes_composite_sneses","List of composite solvers","SNESCompositeAddSNES",sneses,&nmax,&flg));
  if (flg) {
    for (i=0; i<nmax; i++) {
      CHKERRQ(SNESCompositeAddSNES(snes,sneses[i]));
      CHKERRQ(PetscFree(sneses[i]));   /* deallocate string sneses[i], which is allocated in PetscOptionsStringArray() */
    }
  }
  CHKERRQ(PetscOptionsRealArray("-snes_composite_damping","Damping of the additive composite solvers","SNESCompositeSetDamping",dmps,&nmax,&flg));
  if (flg) {
    for (i=0; i<nmax; i++) {
      CHKERRQ(SNESCompositeSetDamping(snes,i,dmps[i]));
    }
  }
  CHKERRQ(PetscOptionsReal("-snes_composite_stol","Step tolerance for restart on the additive composite solvers","",jac->stol,&jac->stol,NULL));
  CHKERRQ(PetscOptionsReal("-snes_composite_rtol","Residual tolerance for the additive composite solvers","",jac->rtol,&jac->rtol,NULL));
  CHKERRQ(PetscOptionsTail());

  next = jac->head;
  while (next) {
    CHKERRQ(SNESSetFromOptions(next->snes));
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  type - %s\n",SNESCompositeTypes[jac->type]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  SNESes on composite preconditioner follow\n"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n"));
  }
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
  }
  while (next) {
    CHKERRQ(SNESView(next->snes,viewer));
    next = next->next;
  }
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n"));
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
  CHKERRQ(PetscNewLog(snes,&ilink));
  ilink->next = NULL;
  CHKERRQ(SNESCreate(PetscObjectComm((PetscObject)snes),&ilink->snes));
  CHKERRQ(PetscLogObjectParent((PetscObject)snes,(PetscObject)ilink->snes));
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(SNESSetDM(ilink->snes,dm));
  CHKERRQ(SNESSetTolerances(ilink->snes,snes->abstol,snes->rtol,snes->stol,1,snes->max_funcs));
  CHKERRQ(PetscObjectCopyFortranFunctionPointers((PetscObject)snes,(PetscObject)ilink->snes));
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
  CHKERRQ(SNESGetOptionsPrefix(snes,&prefix));
  CHKERRQ(SNESSetOptionsPrefix(ilink->snes,prefix));
  CHKERRQ(PetscSNPrintf(newprefix,sizeof(newprefix),"sub_%d_",(int)cnt));
  CHKERRQ(SNESAppendOptionsPrefix(ilink->snes,newprefix));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ilink->snes,(PetscObject)snes,1));
  CHKERRQ(SNESSetType(ilink->snes,type));
  CHKERRQ(SNESSetNormSchedule(ilink->snes, SNES_NORM_FINAL_ONLY));

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
  CHKERRQ(PetscTryMethod(snes,"SNESCompositeSetType_C",(SNES,SNESCompositeType),(snes,type)));
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
  CHKERRQ(PetscTryMethod(snes,"SNESCompositeAddSNES_C",(SNES,SNESType),(snes,type)));
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

.seealso: SNESCompositeAddSNES()
@*/
PetscErrorCode  SNESCompositeGetSNES(SNES snes,PetscInt n,SNES *subsnes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(subsnes,3);
  CHKERRQ(PetscUseMethod(snes,"SNESCompositeGetSNES_C",(SNES,PetscInt,SNES*),(snes,n,subsnes)));
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

.seealso: SNESCompositeAddSNES()
@*/
PetscErrorCode  SNESCompositeSetDamping(SNES snes,PetscInt n,PetscReal dmp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  CHKERRQ(PetscUseMethod(snes,"SNESCompositeSetDamping_C",(SNES,PetscInt,PetscReal),(snes,n,dmp)));
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

  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter   = 0;
  snes->norm   = 0.;
  comp->innerFailures = 0;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
  snes->reason = SNES_CONVERGED_ITERATING;
  CHKERRQ(SNESGetNormSchedule(snes, &normtype));
  if (normtype == SNES_NORM_ALWAYS || normtype == SNES_NORM_INITIAL_ONLY || normtype == SNES_NORM_INITIAL_FINAL_ONLY) {
    if (!snes->vec_func_init_set) {
      CHKERRQ(SNESComputeFunction(snes,X,F));
    } else snes->vec_func_init_set = PETSC_FALSE;

    if (snes->xl && snes->xu) {
      CHKERRQ(SNESVIComputeInactiveSetFnorm(snes, F, X, &fnorm));
    } else {
      CHKERRQ(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- ||F||  */
    }
    SNESCheckFunctionNorm(snes,fnorm);
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = 0;
    snes->norm = fnorm;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
    CHKERRQ(SNESLogConvergenceHistory(snes,snes->norm,0));
    CHKERRQ(SNESMonitor(snes,0,snes->norm));

    /* test convergence */
    CHKERRQ((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
    CHKERRQ(SNESLogConvergenceHistory(snes,snes->norm,0));
    CHKERRQ(SNESMonitor(snes,0,snes->norm));
  }

  for (i = 0; i < snes->max_its; i++) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      CHKERRQ((*snes->ops->update)(snes, snes->iter));
    }

    /* Copy the state before modification by application of the composite solver;
       we will subtract the new state after application */
    CHKERRQ(VecCopy(X, Y));

    if (comp->type == SNES_COMPOSITE_ADDITIVE) {
      CHKERRQ(SNESCompositeApply_Additive(snes,X,B,F,&fnorm));
    } else if (comp->type == SNES_COMPOSITE_MULTIPLICATIVE) {
      CHKERRQ(SNESCompositeApply_Multiplicative(snes,X,B,F,&fnorm));
    } else if (comp->type == SNES_COMPOSITE_ADDITIVEOPTIMAL) {
      CHKERRQ(SNESCompositeApply_AdditiveOptimal(snes,X,B,F,&fnorm));
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Unsupported SNESComposite type");
    if (snes->reason < 0) break;

    /* Compute the solution update for convergence testing */
    CHKERRQ(VecAYPX(Y, -1.0, X));

    if ((i == snes->max_its - 1) && (normtype == SNES_NORM_INITIAL_FINAL_ONLY || normtype == SNES_NORM_FINAL_ONLY)) {
      CHKERRQ(SNESComputeFunction(snes,X,F));

      if (snes->xl && snes->xu) {
        CHKERRQ(VecNormBegin(X, NORM_2, &xnorm));
        CHKERRQ(VecNormBegin(Y, NORM_2, &snorm));
        CHKERRQ(SNESVIComputeInactiveSetFnorm(snes, F, X, &fnorm));
        CHKERRQ(VecNormEnd(X, NORM_2, &xnorm));
        CHKERRQ(VecNormEnd(Y, NORM_2, &snorm));
      } else {
        CHKERRQ(VecNormBegin(F, NORM_2, &fnorm));
        CHKERRQ(VecNormBegin(X, NORM_2, &xnorm));
        CHKERRQ(VecNormBegin(Y, NORM_2, &snorm));

        CHKERRQ(VecNormEnd(F, NORM_2, &fnorm));
        CHKERRQ(VecNormEnd(X, NORM_2, &xnorm));
        CHKERRQ(VecNormEnd(Y, NORM_2, &snorm));
      }
      SNESCheckFunctionNorm(snes,fnorm);
    } else if (normtype == SNES_NORM_ALWAYS) {
      CHKERRQ(VecNormBegin(X, NORM_2, &xnorm));
      CHKERRQ(VecNormBegin(Y, NORM_2, &snorm));
      CHKERRQ(VecNormEnd(X, NORM_2, &xnorm));
      CHKERRQ(VecNormEnd(Y, NORM_2, &snorm));
    }
    /* Monitor convergence */
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = snorm;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
    CHKERRQ(SNESLogConvergenceHistory(snes,snes->norm,0));
    CHKERRQ(SNESMonitor(snes,snes->iter,snes->norm));
    /* Test for convergence */
    if (normtype == SNES_NORM_ALWAYS) CHKERRQ((*snes->ops->converged)(snes,snes->iter,xnorm,snorm,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) break;
  }
  if (normtype == SNES_NORM_ALWAYS) {
    if (i == snes->max_its) {
      CHKERRQ(PetscInfo(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its));
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

.seealso:  SNESCreate(), SNESSetType(), SNESType (for list of available types), SNES,
           SNESSHELL, SNESCompositeSetType(), SNESCompositeSpecialSetAlpha(), SNESCompositeAddSNES(),
           SNESCompositeGetSNES()

   References:
.  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

M*/

PETSC_EXTERN PetscErrorCode SNESCreate_Composite(SNES snes)
{
  SNES_Composite *jac;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(snes,&jac));

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

  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetType_C",SNESCompositeSetType_Composite));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeAddSNES_C",SNESCompositeAddSNES_Composite));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeGetSNES_C",SNESCompositeGetSNES_Composite));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetDamping_C",SNESCompositeSetDamping_Composite));
  PetscFunctionReturn(0);
}
