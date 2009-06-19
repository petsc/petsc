#define PETSCKSP_DLL

#include "private/kspimpl.h"

/* ---------------------------------------Method 1------------------------------------------------------------*/
typedef struct {
    PetscInt    method,    /* 1 or 2 */
                curl,     /* Current number of basis vectors */
                maxl,     /* Maximum number of basis vectors */
                refcnt;
    PetscTruth  monitor;
    Mat         mat;
    KSP         ksp;
    PetscScalar *alpha;   /* */
    Vec         *xtilde,  /* Saved x vectors */
                *btilde;  /* Saved b vectors */
    Vec         guess;
} KSPFischerGuess_Method1;


#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessCreate_Method1" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessCreate_Method1(KSP ksp,int  maxl,KSPFischerGuess_Method1 **ITG)
{
  KSPFischerGuess_Method1 *itg;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscMalloc(sizeof(KSPFischerGuess_Method1),&itg);CHKERRQ(ierr);
  ierr = PetscMalloc(maxl * sizeof(PetscScalar),&itg->alpha);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSPFischerGuess_Method1) + maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->xtilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->xtilde);CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->btilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->btilde);CHKERRQ(ierr);
  ierr = VecDuplicate(itg->xtilde[0],&itg->guess);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ksp,itg->guess);CHKERRQ(ierr);
  *ITG = itg;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessDestroy_Method1" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessDestroy_Method1(KSPFischerGuess_Method1 *itg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(itg->alpha);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->btilde,itg->maxl);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->xtilde,itg->maxl);CHKERRQ(ierr);
  ierr = VecDestroy(itg->guess);CHKERRQ(ierr);
  ierr = PetscFree(itg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
        Given a basis generated already this computes a new guess x from the new right hand side b
     Figures out the components of b in each btilde direction and adds them to x
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessFormGuess_Method1"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessFormGuess_Method1(KSPFischerGuess_Method1 *itg,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(itg,2);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);  
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecMDot(b,itg->curl,itg->btilde,itg->alpha);CHKERRQ(ierr);
  if (itg->monitor) {
    ierr = PetscPrintf(((PetscObject)itg->ksp)->comm,"KSPFischerGuess alphas = ");CHKERRQ(ierr);
    for (i=0; i<itg->curl; i++ ){
      ierr = PetscPrintf(((PetscObject)itg->ksp)->comm,"%G ",PetscAbsScalar(itg->alpha[i]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(((PetscObject)itg->ksp)->comm,"\n");CHKERRQ(ierr);
  }
  ierr = VecMAXPY(x,itg->curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);   
  ierr = VecCopy(x,itg->guess);CHKERRQ(ierr);
  /* Note: do not change the b right hand side as is done in the publication */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessUpdate_Method1"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessUpdate_Method1(KSPFischerGuess_Method1 *itg,Vec x)
{
  PetscReal      norm;
  PetscErrorCode ierr;
  int            curl = itg->curl,i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(itg,3);
  if (curl == itg->maxl) {
    ierr = KSP_MatMult(itg->ksp,itg->mat,x,itg->btilde[0]);CHKERRQ(ierr);
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

    ierr = KSP_MatMult(itg->ksp,itg->mat,itg->xtilde[curl],itg->btilde[curl]);CHKERRQ(ierr);
    ierr = VecMDot(itg->btilde[curl],curl,itg->btilde,itg->alpha);CHKERRQ(ierr);
    for (i=0; i<curl; i++) itg->alpha[i] = -itg->alpha[i];
    ierr = VecMAXPY(itg->btilde[curl],curl,itg->alpha,itg->btilde);CHKERRQ(ierr);
    ierr = VecMAXPY(itg->xtilde[curl],curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);

    ierr = VecNormalize(itg->btilde[curl],&norm);CHKERRQ(ierr);
    if (norm) {
      ierr = VecScale(itg->xtilde[curl],1.0/norm);CHKERRQ(ierr);
      itg->curl++;
    } else {
      ierr = PetscInfo(itg->ksp,"Not increasing dimension of Fischer space because new direction is identical to previous");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------Method 2------------------------------------------------------------*/
typedef struct {
    PetscInt    method,    /* 1 or 2 */
                curl,     /* Current number of basis vectors */
                maxl,     /* Maximum number of basis vectors */
                refcnt;
    PetscTruth  monitor;
    Mat         mat;
    KSP         ksp;
    PetscScalar *alpha;   /* */
    Vec         *xtilde;  /* Saved x vectors */
  Vec         Ax,guess;
} KSPFischerGuess_Method2;

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessCreate_Method2" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessCreate_Method2(KSP ksp,int  maxl,KSPFischerGuess_Method2 **ITG)
{
  KSPFischerGuess_Method2 *itg;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscMalloc(sizeof(KSPFischerGuess_Method2),&itg);CHKERRQ(ierr);
  ierr = PetscMalloc(maxl * sizeof(PetscScalar),&itg->alpha);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSPFischerGuess_Method2) + maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->xtilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->xtilde);CHKERRQ(ierr);
  ierr = VecDuplicate(itg->xtilde[0],&itg->Ax);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ksp,itg->Ax);CHKERRQ(ierr);
  ierr = VecDuplicate(itg->xtilde[0],&itg->guess);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(ksp,itg->guess);CHKERRQ(ierr);
  *ITG = itg;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessDestroy_Method2" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessDestroy_Method2(KSPFischerGuess_Method2 *itg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(itg->alpha);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->xtilde,itg->maxl);CHKERRQ(ierr);
  ierr = VecDestroy(itg->Ax);CHKERRQ(ierr);
  ierr = VecDestroy(itg->guess);CHKERRQ(ierr);
  ierr = PetscFree(itg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
        Given a basis generated already this computes a new guess x from the new right hand side b
     Figures out the components of b in each btilde direction and adds them to x
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessFormGuess_Method2"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessFormGuess_Method2(KSPFischerGuess_Method2 *itg,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(itg,2);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);  
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecMDot(b,itg->curl,itg->xtilde,itg->alpha);CHKERRQ(ierr);
  if (itg->monitor) {
    ierr = PetscPrintf(((PetscObject)itg->ksp)->comm,"KSPFischerGuess alphas = ");CHKERRQ(ierr);
    for (i=0; i<itg->curl; i++ ){
      ierr = PetscPrintf(((PetscObject)itg->ksp)->comm,"%G ",PetscAbsScalar(itg->alpha[i]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(((PetscObject)itg->ksp)->comm,"\n");CHKERRQ(ierr);
  }
  ierr = VecMAXPY(x,itg->curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);   
  ierr = VecCopy(x,itg->guess);CHKERRQ(ierr);
  /* Note: do not change the b right hand side as is done in the publication */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessUpdate_Method2"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessUpdate_Method2(KSPFischerGuess_Method2 *itg,Vec x)
{
  PetscScalar    norm;
  PetscErrorCode ierr;
  int            curl = itg->curl,i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(itg,3);
  if (curl == itg->maxl) {
    ierr = KSP_MatMult(itg->ksp,itg->mat,x,itg->Ax);CHKERRQ(ierr); /* norm = sqrt(x'Ax) */
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
    ierr = KSP_MatMult(itg->ksp,itg->mat,itg->xtilde[curl],itg->Ax);CHKERRQ(ierr); 
    ierr = VecMDot(itg->Ax,curl,itg->xtilde,itg->alpha);CHKERRQ(ierr);
    for (i=0; i<curl; i++) itg->alpha[i] = -itg->alpha[i];
    ierr = VecMAXPY(itg->xtilde[curl],curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);

    ierr = KSP_MatMult(itg->ksp,itg->mat,itg->xtilde[curl],itg->Ax);CHKERRQ(ierr); /* norm = sqrt(xtilde[curl]'Axtilde[curl]) */
    ierr = VecDot(itg->xtilde[curl],itg->Ax,&norm);CHKERRQ(ierr);
    if (PetscAbsScalar(norm) != 0.0) {
      ierr = VecScale(itg->xtilde[curl],1.0/PetscSqrtScalar(norm));CHKERRQ(ierr);
      itg->curl++;
    } else {
      ierr = PetscInfo(itg->ksp,"Not increasing dimension of Fischer space because new direction is identical to previous");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessCreate" 
/*@C
    KSPFischerGuessCreate - Implements Paul Fischer's initial guess algorithm Method 1 and 2 for situations where
    a linear system is solved repeatedly 

  References:
      http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19940020363_1994020363.pdf

   Notes: the algorithm is different from the paper because we do not CHANGE the right hand side of the new 
    problem and solve the problem with an initial guess of zero, rather we solve the original new problem
    with a nonzero initial guess (this is done so that the linear solver convergence tests are based on
    the original RHS.) But we use the xtilde = x - xguess as the new direction so that it is not
    mostly orthogonal to the previous solutions.

    These are not intended to be used directly, they are called by KSP automatically when the 
    KSP option KSPSetFischerGuess(KSP,PetscInt,PetscInt) or -ksp_guess_fischer <int,int>

    Method 2 is only for positive definite matrices, since it uses the A norm.

    This is not currently programmed as a PETSc class because there are only two methods; if more methods
    are introduced it should be changed. For example the Knoll guess should be included

    Level: advanced

@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessCreate(KSP ksp,PetscInt method,PetscInt maxl,KSPFischerGuess *itg)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (method == 1) {
    ierr = KSPFischerGuessCreate_Method1(ksp,maxl,(KSPFischerGuess_Method1 **)itg);CHKERRQ(ierr);
  } else if (method == 2) {
    ierr = KSPFischerGuessCreate_Method2(ksp,maxl,(KSPFischerGuess_Method2 **)itg);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Method can only be 1 or 2");
  (*itg)->method  = method;
  (*itg)->curl    = 0;
  (*itg)->maxl    = maxl;
  (*itg)->ksp     = ksp;
  (*itg)->refcnt  = 1;
  ierr = KSPGetOperators(ksp,&(*itg)->mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessSetFromOptions"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessSetFromOptions(KSPFischerGuess ITG)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetTruth(((PetscObject)ITG->ksp)->prefix,"-ksp_fischer_guess_monitor",&ITG->monitor,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessDestroy" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessDestroy(KSPFischerGuess ITG)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (--ITG->refcnt) PetscFunctionReturn(0);

  if (ITG->method == 1) {
    ierr = KSPFischerGuessDestroy_Method1((KSPFischerGuess_Method1 *)ITG);CHKERRQ(ierr);
  } else if (ITG->method == 2) {
    ierr = KSPFischerGuessDestroy_Method2((KSPFischerGuess_Method2 *)ITG);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Method can only be 1 or 2");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessUpdate"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessUpdate(KSPFischerGuess itg,Vec x)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (itg->method == 1) {
    ierr = KSPFischerGuessUpdate_Method1((KSPFischerGuess_Method1 *)itg,x);CHKERRQ(ierr);
  } else if (itg->method == 2) {
    ierr = KSPFischerGuessUpdate_Method2((KSPFischerGuess_Method2 *)itg,x);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Method can only be 1 or 2");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessFormGuess"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessFormGuess(KSPFischerGuess itg,Vec b,Vec x)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (itg->method == 1) {
    ierr = KSPFischerGuessFormGuess_Method1((KSPFischerGuess_Method1 *)itg,b,x);CHKERRQ(ierr);
  } else if (itg->method == 2) {
    ierr = KSPFischerGuessFormGuess_Method2((KSPFischerGuess_Method2 *)itg,b,x);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Method can only be 1 or 2");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessReset"
/*
    KSPFischerGuessReset - This is called whenever KSPSetOperators() is called to tell the
      initial guess object that the matrix is changed and so the initial guess object
      must restart from scratch building the subspace where the guess is computed from.
*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessReset(KSPFischerGuess itg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  itg->curl = 0;
  ierr = KSPGetOperators(itg->ksp,&itg->mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

