#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

typedef struct {
  PetscInt    method;   /* 1 or 2 */
  PetscInt    curl;     /* Current number of basis vectors */
  PetscInt    maxl;     /* Maximum number of basis vectors */
  PetscBool   monitor;
  PetscScalar *alpha;   /* */
  Vec         *xtilde;  /* Saved x vectors */
  Vec         *btilde;  /* Saved b vectors, method 1 */
  Vec         Ax;       /* method 2 */
  Vec         guess;
} KSPGuessFischer;

static PetscErrorCode KSPGuessReset_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscLayout     Alay = NULL,vlay = NULL;
  PetscBool       cong;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  itg->curl = 0;
  /* destroy vectors if the size of the linear system has changed */
  if (guess->A) {
    ierr = MatGetLayouts(guess->A,&Alay,NULL);CHKERRQ(ierr);
  }
  if (itg->xtilde) {
    ierr = VecGetLayout(itg->xtilde[0],&vlay);CHKERRQ(ierr);
  }
  cong = PETSC_FALSE;
  if (vlay && Alay) {
    ierr = PetscLayoutCompare(Alay,vlay,&cong);CHKERRQ(ierr);
  }
  if (!cong) {
    ierr = VecDestroyVecs(itg->maxl,&itg->btilde);CHKERRQ(ierr);
    ierr = VecDestroyVecs(itg->maxl,&itg->xtilde);CHKERRQ(ierr);
    ierr = VecDestroy(&itg->guess);CHKERRQ(ierr);
    ierr = VecDestroy(&itg->Ax);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetUp_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!itg->alpha) {
    ierr = PetscMalloc1(itg->maxl,&itg->alpha);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)guess,itg->maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (!itg->xtilde) {
    ierr = KSPCreateVecs(guess->ksp,itg->maxl,&itg->xtilde,0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(guess,itg->maxl,itg->xtilde);CHKERRQ(ierr);
  }
  if (!itg->btilde && itg->method == 1) {
    ierr = KSPCreateVecs(guess->ksp,itg->maxl,&itg->btilde,0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(guess,itg->maxl,itg->btilde);CHKERRQ(ierr);
  }
  if (!itg->Ax && itg->method == 2) {
    ierr = VecDuplicate(itg->xtilde[0],&itg->Ax);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)guess,(PetscObject)itg->Ax);CHKERRQ(ierr);
  }
  if (!itg->guess) {
    ierr = VecDuplicate(itg->xtilde[0],&itg->guess);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)guess,(PetscObject)itg->guess);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessDestroy_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(itg->alpha);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->maxl,&itg->btilde);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->maxl,&itg->xtilde);CHKERRQ(ierr);
  ierr = VecDestroy(&itg->guess);CHKERRQ(ierr);
  ierr = VecDestroy(&itg->Ax);CHKERRQ(ierr);
  ierr = PetscFree(itg);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)guess,"KSPGuessFischerSetModel_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Note: do not change the b right hand side as is done in the publication */
static PetscErrorCode KSPGuessFormGuess_Fischer_1(KSPGuess guess,Vec b,Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecMDot(b,itg->curl,itg->btilde,itg->alpha);CHKERRQ(ierr);
  if (itg->monitor) {
    ierr = PetscPrintf(((PetscObject)guess)->comm,"KSPFischerGuess alphas = ");CHKERRQ(ierr);
    for (i=0; i<itg->curl; i++) {
      ierr = PetscPrintf(((PetscObject)guess)->comm,"%g ",(double)PetscAbsScalar(itg->alpha[i]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(((PetscObject)guess)->comm,"\n");CHKERRQ(ierr);
  }
  ierr = VecMAXPY(x,itg->curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);
  ierr = VecCopy(x,itg->guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_Fischer_1(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscReal       norm;
  PetscErrorCode  ierr;
  int             curl = itg->curl,i;

  PetscFunctionBegin;
  if (curl == itg->maxl) {
    ierr = KSP_MatMult(guess->ksp,guess->A,x,itg->btilde[0]);CHKERRQ(ierr);
    /* ierr = VecCopy(b,itg->btilde[0]);CHKERRQ(ierr); */
    ierr = VecNormalize(itg->btilde[0],&norm);CHKERRQ(ierr);
    ierr = VecCopy(x,itg->xtilde[0]);CHKERRQ(ierr);
    ierr = VecScale(itg->xtilde[0],1.0/norm);CHKERRQ(ierr);
    itg->curl = 1;
  } else {
    if (!curl) {
      ierr = VecCopy(x,itg->xtilde[curl]);CHKERRQ(ierr);
    } else {
      ierr = VecWAXPY(itg->xtilde[curl],-1.0,itg->guess,x);CHKERRQ(ierr);
    }
    ierr = KSP_MatMult(guess->ksp,guess->A,itg->xtilde[curl],itg->btilde[curl]);CHKERRQ(ierr);
    ierr = VecMDot(itg->btilde[curl],curl,itg->btilde,itg->alpha);CHKERRQ(ierr);
    for (i=0; i<curl; i++) itg->alpha[i] = -itg->alpha[i];
    ierr = VecMAXPY(itg->btilde[curl],curl,itg->alpha,itg->btilde);CHKERRQ(ierr);
    ierr = VecMAXPY(itg->xtilde[curl],curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);
    ierr = VecNormalize(itg->btilde[curl],&norm);CHKERRQ(ierr);
    if (norm) {
      ierr = VecScale(itg->xtilde[curl],1.0/norm);CHKERRQ(ierr);
      itg->curl++;
    } else {
      ierr = PetscInfo(guess->ksp,"Not increasing dimension of Fischer space because new direction is identical to previous\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  Given a basis generated already this computes a new guess x from the new right hand side b
  Figures out the components of b in each btilde direction and adds them to x
  Note: do not change the b right hand side as is done in the publication
*/
static PetscErrorCode KSPGuessFormGuess_Fischer_2(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecMDot(b,itg->curl,itg->xtilde,itg->alpha);CHKERRQ(ierr);
  if (itg->monitor) {
    ierr = PetscPrintf(((PetscObject)guess)->comm,"KSPFischerGuess alphas = ");CHKERRQ(ierr);
    for (i=0; i<itg->curl; i++) {
      ierr = PetscPrintf(((PetscObject)guess)->comm,"%g ",(double)PetscAbsScalar(itg->alpha[i]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(((PetscObject)guess)->comm,"\n");CHKERRQ(ierr);
  }
  ierr = VecMAXPY(x,itg->curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);
  ierr = VecCopy(x,itg->guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_Fischer_2(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscScalar     norm;
  PetscErrorCode  ierr;
  int             curl = itg->curl,i;

  PetscFunctionBegin;
  if (curl == itg->maxl) {
    ierr = KSP_MatMult(guess->ksp,guess->A,x,itg->Ax);CHKERRQ(ierr); /* norm = sqrt(x'Ax) */
    ierr = VecDot(x,itg->Ax,&norm);CHKERRQ(ierr);
    ierr = VecCopy(x,itg->xtilde[0]);CHKERRQ(ierr);
    ierr = VecScale(itg->xtilde[0],1.0/PetscSqrtScalar(norm));CHKERRQ(ierr);
    itg->curl = 1;
  } else {
    if (!curl) {
      ierr = VecCopy(x,itg->xtilde[curl]);CHKERRQ(ierr);
    } else {
      ierr = VecWAXPY(itg->xtilde[curl],-1.0,itg->guess,x);CHKERRQ(ierr);
    }
    ierr = KSP_MatMult(guess->ksp,guess->A,itg->xtilde[curl],itg->Ax);CHKERRQ(ierr);
    ierr = VecMDot(itg->Ax,curl,itg->xtilde,itg->alpha);CHKERRQ(ierr);
    for (i=0; i<curl; i++) itg->alpha[i] = -itg->alpha[i];
    ierr = VecMAXPY(itg->xtilde[curl],curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);

    ierr = KSP_MatMult(guess->ksp,guess->A,itg->xtilde[curl],itg->Ax);CHKERRQ(ierr); /* norm = sqrt(xtilde[curl]'Axtilde[curl]) */
    ierr = VecDot(itg->xtilde[curl],itg->Ax,&norm);CHKERRQ(ierr);
    if (PetscAbsScalar(norm) != 0.0) {
      ierr = VecScale(itg->xtilde[curl],1.0/PetscSqrtScalar(norm));CHKERRQ(ierr);
      itg->curl++;
    } else {
      ierr = PetscInfo(guess->ksp,"Not increasing dimension of Fischer space because new direction is identical to previous\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetFromOptions_Fischer(KSPGuess guess)
{
  KSPGuessFischer *ITG = (KSPGuessFischer *)guess->data;
  PetscInt        nmax = 2, model[2];
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  model[0] = ITG->method;
  model[1] = ITG->maxl;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)guess),((PetscObject)guess)->prefix,"Fischer guess options","KSPGuess");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-ksp_guess_fischer_model","Model type and dimension of basis","KSPGuessFischerSetModel",model,&nmax,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPGuessFischerSetModel(guess,model[0],model[1]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-ksp_guess_fischer_monitor","Monitor the guess",NULL,ITG->monitor,&ITG->monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessView_Fischer(KSPGuess guess,PetscViewer viewer)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscBool       isascii;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Model %D, size %D\n",itg->method,itg->maxl);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   KSPGuessFischerSetModel - Use the Paul Fischer algorithm

   Logically Collective on guess

   Input Parameters:
+  guess - the initial guess context
.  model - use model 1, model 2 or any other number to turn it off
-  size  - size of subspace used to generate initial guess

    Options Database:
.   -ksp_guess_fischer_model <model,size> - uses the Fischer initial guess generator for repeated linear solves

   Level: advanced

.seealso: KSPGuess, KSPGuessCreate(), KSPSetUseFischerGuess(), KSPSetGuess(), KSPGetGuess(), KSP
@*/
PetscErrorCode  KSPGuessFischerSetModel(KSPGuess guess,PetscInt model,PetscInt size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,1);
  PetscValidLogicalCollectiveInt(guess,model,2);
  PetscValidLogicalCollectiveInt(guess,model,3);
  ierr = PetscTryMethod(guess,"KSPGuessFischerSetModel_C",(KSPGuess,PetscInt,PetscInt),(guess,model,size));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessFischerSetModel_Fischer(KSPGuess guess,PetscInt model,PetscInt size)
{
  KSPGuessFischer *itg = (KSPGuessFischer*)guess->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (model == 1) {
    guess->ops->update    = KSPGuessUpdate_Fischer_1;
    guess->ops->formguess = KSPGuessFormGuess_Fischer_1;
  } else if (model == 2) {
    guess->ops->update    = KSPGuessUpdate_Fischer_2;
    guess->ops->formguess = KSPGuessFormGuess_Fischer_2;
  } else {
    guess->ops->update    = NULL;
    guess->ops->formguess = NULL;
    itg->method           = 0;
    PetscFunctionReturn(0);
  }
  if (size != itg->maxl) {
    ierr = PetscFree(itg->alpha);CHKERRQ(ierr);
    ierr = VecDestroyVecs(itg->maxl,&itg->btilde);CHKERRQ(ierr);
    ierr = VecDestroyVecs(itg->maxl,&itg->xtilde);CHKERRQ(ierr);
    ierr = VecDestroy(&itg->guess);CHKERRQ(ierr);
    ierr = VecDestroy(&itg->Ax);CHKERRQ(ierr);
  }
  itg->method = model;
  itg->maxl   = size;
  PetscFunctionReturn(0);
}

/*
    KSPGUESSFISCHER - Implements Paul Fischer's initial guess algorithm Method 1 and 2 for situations where
    a linear system is solved repeatedly

  References:
.   1. -   https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19940020363_1994020363.pdf

   Notes:
    the algorithm is different from the paper because we do not CHANGE the right hand side of the new
    problem and solve the problem with an initial guess of zero, rather we solve the original problem
    with a nonzero initial guess (this is done so that the linear solver convergence tests are based on
    the original RHS). We use the xtilde = x - xguess as the new direction so that it is not
    mostly orthogonal to the previous solutions.

    These are not intended to be used directly, they are called by KSP automatically with the command line options -ksp_guess_type fischer -ksp_guess_fischer_model <int,int> or programmatically as
.vb
    KSPGetGuess(ksp,&guess);
    KSPGuessSetType(guess,KSPGUESSFISCHER);
    KSPGuessFischerSetModel(guess,model,basis);

    Method 2 is only for positive definite matrices, since it uses the A norm.

    Developer note: the option -ksp_fischer_guess <int,int> is still available for backward compatibility

    Level: intermediate

@*/
PetscErrorCode KSPGuessCreate_Fischer(KSPGuess guess)
{
  KSPGuessFischer *fischer;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(guess,&fischer);CHKERRQ(ierr);
  fischer->method = 1;  /* defaults to method 1 */
  fischer->maxl   = 10;
  guess->data     = fischer;

  guess->ops->setfromoptions = KSPGuessSetFromOptions_Fischer;
  guess->ops->destroy        = KSPGuessDestroy_Fischer;
  guess->ops->setup          = KSPGuessSetUp_Fischer;
  guess->ops->view           = KSPGuessView_Fischer;
  guess->ops->reset          = KSPGuessReset_Fischer;
  guess->ops->update         = KSPGuessUpdate_Fischer_1;
  guess->ops->formguess      = KSPGuessFormGuess_Fischer_1;

  ierr = PetscObjectComposeFunction((PetscObject)guess,"KSPGuessFischerSetModel_C",KSPGuessFischerSetModel_Fischer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
