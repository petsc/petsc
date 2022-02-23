
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
  PetscErrorCode      ierr;
  SNES_Composite      *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink  next = jac->head;
  Vec                 FSub;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  PetscCheckFalse(!next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");
  if (snes->normschedule == SNES_NORM_ALWAYS) {
    ierr = SNESSetInitialFunction(next->snes,F);CHKERRQ(ierr);
  }
  ierr = SNESSolve(next->snes,B,X);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(next->snes,&reason);CHKERRQ(ierr);
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
      ierr = SNESGetFunction(next->snes,&FSub,NULL,NULL);CHKERRQ(ierr);
      next = next->next;
      ierr = SNESSetInitialFunction(next->snes,FSub);CHKERRQ(ierr);
    } else {
      next = next->next;
    }
    ierr = SNESSolve(next->snes,B,X);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(next->snes,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      jac->innerFailures++;
      if (jac->innerFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    }
  }
  if (next->snes->npcside== PC_RIGHT) {
    ierr = SNESGetFunction(next->snes,&FSub,NULL,NULL);CHKERRQ(ierr);
    ierr = VecCopy(FSub,F);CHKERRQ(ierr);
    if (fnorm) {
      if (snes->xl && snes->xu) {
        ierr = SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm);CHKERRQ(ierr);
      } else {
        ierr = VecNorm(F, NORM_2, fnorm);CHKERRQ(ierr);
      }
      SNESCheckFunctionNorm(snes,*fnorm);
    }
  } else if (snes->normschedule == SNES_NORM_ALWAYS) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    if (fnorm) {
      if (snes->xl && snes->xu) {
        ierr = SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm);CHKERRQ(ierr);
      } else {
        ierr = VecNorm(F, NORM_2, fnorm);CHKERRQ(ierr);
      }
      SNESCheckFunctionNorm(snes,*fnorm);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESCompositeApply_Additive(SNES snes,Vec X,Vec B,Vec F,PetscReal *fnorm)
{
  PetscErrorCode      ierr;
  SNES_Composite      *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink  next = jac->head;
  Vec                 Y,Xorig;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  Y = snes->vec_sol_update;
  if (!jac->Xorig) {ierr = VecDuplicate(X,&jac->Xorig);CHKERRQ(ierr);}
  Xorig = jac->Xorig;
  ierr = VecCopy(X,Xorig);CHKERRQ(ierr);
  PetscCheckFalse(!next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");
  if (snes->normschedule == SNES_NORM_ALWAYS) {
    ierr = SNESSetInitialFunction(next->snes,F);CHKERRQ(ierr);
    while (next->next) {
      next = next->next;
      ierr = SNESSetInitialFunction(next->snes,F);CHKERRQ(ierr);
    }
  }
  next = jac->head;
  ierr = VecCopy(Xorig,Y);CHKERRQ(ierr);
  ierr = SNESSolve(next->snes,B,Y);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(next->snes,&reason);CHKERRQ(ierr);
  if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
    jac->innerFailures++;
    if (jac->innerFailures >= snes->maxFailures) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }
  ierr = VecAXPY(Y,-1.0,Xorig);CHKERRQ(ierr);
  ierr = VecAXPY(X,next->dmp,Y);CHKERRQ(ierr);
  while (next->next) {
    next = next->next;
    ierr = VecCopy(Xorig,Y);CHKERRQ(ierr);
    ierr = SNESSolve(next->snes,B,Y);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(next->snes,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      jac->innerFailures++;
      if (jac->innerFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    }
    ierr = VecAXPY(Y,-1.0,Xorig);CHKERRQ(ierr);
    ierr = VecAXPY(X,next->dmp,Y);CHKERRQ(ierr);
  }
  if (snes->normschedule == SNES_NORM_ALWAYS) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    if (fnorm) {
      if (snes->xl && snes->xu) {
        ierr = SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm);CHKERRQ(ierr);
      } else {
        ierr = VecNorm(F, NORM_2, fnorm);CHKERRQ(ierr);
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
  PetscErrorCode      ierr;
  SNES_Composite      *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink  next = jac->head;
  Vec                 *Xes = jac->Xes,*Fes = jac->Fes;
  PetscInt            i,j;
  PetscScalar         tot,total,ftf;
  PetscReal           min_fnorm;
  PetscInt            min_i;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  PetscCheckFalse(!next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"No composite SNESes supplied via SNESCompositeAddSNES() or -snes_composite_sneses");

  if (snes->normschedule == SNES_NORM_ALWAYS) {
    next = jac->head;
    ierr = SNESSetInitialFunction(next->snes,F);CHKERRQ(ierr);
    while (next->next) {
      next = next->next;
      ierr = SNESSetInitialFunction(next->snes,F);CHKERRQ(ierr);
    }
  }

  next = jac->head;
  i = 0;
  ierr = VecCopy(X,Xes[i]);CHKERRQ(ierr);
  ierr = SNESSolve(next->snes,B,Xes[i]);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(next->snes,&reason);CHKERRQ(ierr);
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
    ierr = VecCopy(X,Xes[i]);CHKERRQ(ierr);
    ierr = SNESSolve(next->snes,B,Xes[i]);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(next->snes,&reason);CHKERRQ(ierr);
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
      ierr = VecDotBegin(Fes[i],Fes[j],&jac->h[i + j*jac->n]);CHKERRQ(ierr);
    }
    ierr = VecDotBegin(Fes[i],F,&jac->g[i]);CHKERRQ(ierr);
  }

  for (i=0;i<jac->n;i++) {
    for (j=0;j<i+1;j++) {
      ierr = VecDotEnd(Fes[i],Fes[j],&jac->h[i + j*jac->n]);CHKERRQ(ierr);
      if (i == j) jac->fnorms[i] = PetscSqrtReal(PetscRealPart(jac->h[i + j*jac->n]));
    }
    ierr = VecDotEnd(Fes[i],F,&jac->g[i]);CHKERRQ(ierr);
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
  ierr          = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&jac->n,&jac->n,&jac->nrhs,jac->h,&jac->lda,jac->beta,&jac->lda,jac->s,&jac->rcond,&jac->rank,jac->work,&jac->lwork,jac->rwork,&jac->info));
#else
  PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&jac->n,&jac->n,&jac->nrhs,jac->h,&jac->lda,jac->beta,&jac->lda,jac->s,&jac->rcond,&jac->rank,jac->work,&jac->lwork,&jac->info));
#endif
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  PetscCheckFalse(jac->info < 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"Bad argument to GELSS");
  PetscCheckFalse(jac->info > 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD failed to converge");
  tot = 0.;
  total = 0.;
  for (i=0; i<jac->n; i++) {
    PetscCheckFalse(snes->errorifnotconverged && PetscIsInfOrNanScalar(jac->beta[i]),PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD generated inconsistent output");
    ierr = PetscInfo(snes,"%D: %g\n",i,(double)PetscRealPart(jac->beta[i]));CHKERRQ(ierr);
    tot += jac->beta[i];
    total += PetscAbsScalar(jac->beta[i]);
  }
  ierr = VecScale(X,(1. - tot));CHKERRQ(ierr);
  ierr = VecMAXPY(X,jac->n,jac->beta,Xes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);

  if (snes->xl && snes->xu) {
    ierr = SNESVIComputeInactiveSetFnorm(snes, F, X, fnorm);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(F, NORM_2, fnorm);CHKERRQ(ierr);
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
    ierr = VecCopy(jac->Xes[min_i],X);CHKERRQ(ierr);
    ierr = VecCopy(jac->Fes[min_i],F);CHKERRQ(ierr);
    *fnorm = min_fnorm;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_Composite(SNES snes)
{
  PetscErrorCode     ierr;
  DM                 dm;
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  SNES_CompositeLink next = jac->head;
  PetscInt           n=0,i;
  Vec                F;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);

  if (snes->ops->computevariablebounds) {
    /* SNESVI only ever calls computevariablebounds once, so calling it once here is justified */
    if (!snes->xl) {ierr = VecDuplicate(snes->vec_sol,&snes->xl);CHKERRQ(ierr);}
    if (!snes->xu) {ierr = VecDuplicate(snes->vec_sol,&snes->xu);CHKERRQ(ierr);}
    ierr = (*snes->ops->computevariablebounds)(snes,snes->xl,snes->xu);CHKERRQ(ierr);
  }

  while (next) {
    n++;
    ierr = SNESSetDM(next->snes,dm);CHKERRQ(ierr);
    ierr = SNESSetApplicationContext(next->snes, snes->user);CHKERRQ(ierr);
    if (snes->xl && snes->xu) {
      if (snes->ops->computevariablebounds) {
        ierr = SNESVISetComputeVariableBounds(next->snes, snes->ops->computevariablebounds);CHKERRQ(ierr);
      } else {
        ierr = SNESVISetVariableBounds(next->snes,snes->xl,snes->xu);CHKERRQ(ierr);
      }
    }

    next = next->next;
  }
  jac->nsnes = n;
  ierr = SNESGetFunction(snes,&F,NULL,NULL);CHKERRQ(ierr);
  if (jac->type == SNES_COMPOSITE_ADDITIVEOPTIMAL) {
    ierr = VecDuplicateVecs(F,jac->nsnes,&jac->Xes);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&jac->Fes);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&jac->fnorms);CHKERRQ(ierr);
    next = jac->head;
    i = 0;
    while (next) {
      ierr = SNESGetFunction(next->snes,&F,NULL,NULL);CHKERRQ(ierr);
      jac->Fes[i] = F;
      ierr = PetscObjectReference((PetscObject)F);CHKERRQ(ierr);
      next = next->next;
      i++;
    }
    /* allocate the subspace direct solve area */
    jac->nrhs  = 1;
    jac->lda   = jac->nsnes;
    jac->ldb   = jac->nsnes;
    jac->n     = jac->nsnes;

    ierr = PetscMalloc1(jac->n*jac->n,&jac->h);CHKERRQ(ierr);
    ierr = PetscMalloc1(jac->n,&jac->beta);CHKERRQ(ierr);
    ierr = PetscMalloc1(jac->n,&jac->s);CHKERRQ(ierr);
    ierr = PetscMalloc1(jac->n,&jac->g);CHKERRQ(ierr);
    jac->lwork = 12*jac->n;
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc1(jac->lwork,&jac->rwork);CHKERRQ(ierr);
#endif
    ierr = PetscMalloc1(jac->lwork,&jac->work);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

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
  if (jac->Xes) {ierr = VecDestroyVecs(jac->nsnes,&jac->Xes);CHKERRQ(ierr);}
  if (jac->Fes) {ierr = VecDestroyVecs(jac->nsnes,&jac->Fes);CHKERRQ(ierr);}
  ierr = PetscFree(jac->fnorms);CHKERRQ(ierr);
  ierr = PetscFree(jac->h);CHKERRQ(ierr);
  ierr = PetscFree(jac->s);CHKERRQ(ierr);
  ierr = PetscFree(jac->g);CHKERRQ(ierr);
  ierr = PetscFree(jac->beta);CHKERRQ(ierr);
  ierr = PetscFree(jac->work);CHKERRQ(ierr);
  ierr = PetscFree(jac->rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_Composite(SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  PetscErrorCode     ierr;
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

static PetscErrorCode SNESSetFromOptions_Composite(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  PetscErrorCode     ierr;
  PetscInt           nmax = 8,i;
  SNES_CompositeLink next;
  char               *sneses[8];
  PetscReal          dmps[8];
  PetscBool          flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Composite preconditioner options");CHKERRQ(ierr);
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
  ierr = PetscOptionsReal("-snes_composite_stol","Step tolerance for restart on the additive composite solvers","",jac->stol,&jac->stol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_composite_rtol","Residual tolerance for the additive composite solvers","",jac->rtol,&jac->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  next = jac->head;
  while (next) {
    ierr = SNESSetFromOptions(next->snes);CHKERRQ(ierr);
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_Composite(SNES snes,PetscViewer viewer)
{
  SNES_Composite     *jac = (SNES_Composite*)snes->data;
  PetscErrorCode     ierr;
  SNES_CompositeLink next = jac->head;
  PetscBool          iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  type - %s\n",SNESCompositeTypes[jac->type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  SNESes on composite preconditioner follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n");CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n");CHKERRQ(ierr);
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
  PetscErrorCode     ierr;
  PetscInt           cnt = 0;
  const char         *prefix;
  char               newprefix[20];
  DM                 dm;

  PetscFunctionBegin;
  ierr        = PetscNewLog(snes,&ilink);CHKERRQ(ierr);
  ilink->next = NULL;
  ierr        = SNESCreate(PetscObjectComm((PetscObject)snes),&ilink->snes);CHKERRQ(ierr);
  ierr        = PetscLogObjectParent((PetscObject)snes,(PetscObject)ilink->snes);CHKERRQ(ierr);
  ierr        = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr        = SNESSetDM(ilink->snes,dm);CHKERRQ(ierr);
  ierr        = SNESSetTolerances(ilink->snes,snes->abstol,snes->rtol,snes->stol,1,snes->max_funcs);CHKERRQ(ierr);
  ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)snes,(PetscObject)ilink->snes);CHKERRQ(ierr);
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
  ierr = PetscSNPrintf(newprefix,sizeof(newprefix),"sub_%d_",(int)cnt);CHKERRQ(ierr);
  ierr = SNESAppendOptionsPrefix(ilink->snes,newprefix);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ilink->snes,(PetscObject)snes,1);CHKERRQ(ierr);
  ierr = SNESSetType(ilink->snes,type);CHKERRQ(ierr);
  ierr = SNESSetNormSchedule(ilink->snes, SNES_NORM_FINAL_ONLY);CHKERRQ(ierr);

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
    PetscCheckFalse(!next->next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Not enough SNESes in composite preconditioner");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveEnum(snes,type,2);
  ierr = PetscTryMethod(snes,"SNESCompositeSetType_C",(SNES,SNESCompositeType),(snes,type));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESCompositeAddSNES_C",(SNES,SNESType),(snes,type));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(subsnes,3);
  ierr = PetscUseMethod(snes,"SNESCompositeGetSNES_C",(SNES,PetscInt,SNES*),(snes,n,subsnes));CHKERRQ(ierr);
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
    PetscCheckFalse(!next->next,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Not enough SNESes in composite preconditioner");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscUseMethod(snes,"SNESCompositeSetDamping_C",(SNES,PetscInt,PetscReal),(snes,n,dmp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_Composite(SNES snes)
{
  Vec              F,X,B,Y;
  PetscInt         i;
  PetscReal        fnorm = 0.0, xnorm = 0.0, snorm = 0.0;
  PetscErrorCode   ierr;
  SNESNormSchedule normtype;
  SNES_Composite   *comp = (SNES_Composite*)snes->data;

  PetscFunctionBegin;
  X = snes->vec_sol;
  F = snes->vec_func;
  B = snes->vec_rhs;
  Y = snes->vec_sol_update;

  ierr         = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter   = 0;
  snes->norm   = 0.;
  comp->innerFailures = 0;
  ierr         = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITERATING;
  ierr         = SNESGetNormSchedule(snes, &normtype);CHKERRQ(ierr);
  if (normtype == SNES_NORM_ALWAYS || normtype == SNES_NORM_INITIAL_ONLY || normtype == SNES_NORM_INITIAL_FINAL_ONLY) {
    if (!snes->vec_func_init_set) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    } else snes->vec_func_init_set = PETSC_FALSE;

    if (snes->xl && snes->xu) {
      ierr = SNESVIComputeInactiveSetFnorm(snes, F, X, &fnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
    }
    SNESCheckFunctionNorm(snes,fnorm);
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = 0;
    snes->norm = fnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);

    /* test convergence */
    ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    ierr = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);
  }

  for (i = 0; i < snes->max_its; i++) {
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /* Copy the state before modification by application of the composite solver;
       we will subtract the new state after application */
    ierr = VecCopy(X, Y);CHKERRQ(ierr);

    if (comp->type == SNES_COMPOSITE_ADDITIVE) {
      ierr = SNESCompositeApply_Additive(snes,X,B,F,&fnorm);CHKERRQ(ierr);
    } else if (comp->type == SNES_COMPOSITE_MULTIPLICATIVE) {
      ierr = SNESCompositeApply_Multiplicative(snes,X,B,F,&fnorm);CHKERRQ(ierr);
    } else if (comp->type == SNES_COMPOSITE_ADDITIVEOPTIMAL) {
      ierr = SNESCompositeApply_AdditiveOptimal(snes,X,B,F,&fnorm);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Unsupported SNESComposite type");
    if (snes->reason < 0) break;

    /* Compute the solution update for convergence testing */
    ierr = VecAYPX(Y, -1.0, X);CHKERRQ(ierr);

    if ((i == snes->max_its - 1) && (normtype == SNES_NORM_INITIAL_FINAL_ONLY || normtype == SNES_NORM_FINAL_ONLY)) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);

      if (snes->xl && snes->xu) {
        ierr = VecNormBegin(X, NORM_2, &xnorm);CHKERRQ(ierr);
        ierr = VecNormBegin(Y, NORM_2, &snorm);CHKERRQ(ierr);
        ierr = SNESVIComputeInactiveSetFnorm(snes, F, X, &fnorm);CHKERRQ(ierr);
        ierr = VecNormEnd(X, NORM_2, &xnorm);CHKERRQ(ierr);
        ierr = VecNormEnd(Y, NORM_2, &snorm);CHKERRQ(ierr);
      } else {
        ierr = VecNormBegin(F, NORM_2, &fnorm);CHKERRQ(ierr);
        ierr = VecNormBegin(X, NORM_2, &xnorm);CHKERRQ(ierr);
        ierr = VecNormBegin(Y, NORM_2, &snorm);CHKERRQ(ierr);

        ierr = VecNormEnd(F, NORM_2, &fnorm);CHKERRQ(ierr);
        ierr = VecNormEnd(X, NORM_2, &xnorm);CHKERRQ(ierr);
        ierr = VecNormEnd(Y, NORM_2, &snorm);CHKERRQ(ierr);
      }
      SNESCheckFunctionNorm(snes,fnorm);
    } else if (normtype == SNES_NORM_ALWAYS) {
      ierr = VecNormBegin(X, NORM_2, &xnorm);CHKERRQ(ierr);
      ierr = VecNormBegin(Y, NORM_2, &snorm);CHKERRQ(ierr);
      ierr = VecNormEnd(X, NORM_2, &xnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(Y, NORM_2, &snorm);CHKERRQ(ierr);
    }
    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = snorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    if (normtype == SNES_NORM_ALWAYS) {ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,snorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);}
    if (snes->reason) break;
  }
  if (normtype == SNES_NORM_ALWAYS) {
    if (i == snes->max_its) {
      ierr = PetscInfo(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
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
.  1. - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

M*/

PETSC_EXTERN PetscErrorCode SNESCreate_Composite(SNES snes)
{
  PetscErrorCode ierr;
  SNES_Composite *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(snes,&jac);CHKERRQ(ierr);

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

  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetType_C",SNESCompositeSetType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeAddSNES_C",SNESCompositeAddSNES_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeGetSNES_C",SNESCompositeGetSNES_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESCompositeSetDamping_C",SNESCompositeSetDamping_Composite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
