#define PETSCKSP_DLL

#include "include/private/kspimpl.h"


typedef struct {
    PetscInt    model,    /* 1 or 2 */
                curl,     /* Current number of basis vectors */
                maxl,     /* Maximum number of basis vectors */
                refcnt;
    Mat         mat;
    KSP         ksp;
    PetscScalar *alpha;   /* */
    Vec         *xtilde,  /* Saved x vectors */
                *btilde;  /* Saved b vectors */
} KSPFischerGuess_Model1;


#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessCreate_Model1" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessCreate_Model1(KSP ksp,int  maxl,KSPFischerGuess_Model1 **ITG)
{
  KSPFischerGuess_Model1 *itg;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscMalloc(sizeof(KSPFischerGuess_Model1),&itg);CHKERRQ(ierr);
  ierr = PetscMalloc(maxl * sizeof(PetscScalar),&itg->alpha);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSPFischerGuess_Model1) + maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->xtilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->xtilde);CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->btilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->btilde);CHKERRQ(ierr);
  *ITG = itg;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessDestroy_Model1" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessDestroy_Model1(KSPFischerGuess_Model1 *itg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(itg->alpha);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->btilde,itg->maxl);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->xtilde,itg->maxl);CHKERRQ(ierr);
  ierr = PetscFree(itg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
        Given a basis generated already this computes a new guess x from the new right hand side b
     Figures out the components of b in each btilde direction and adds them to x
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessFormGuess_Model1"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessFormGuess_Model1(KSPFischerGuess_Model1 *itg,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(itg,2);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);  
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecMDot(b,itg->curl,itg->btilde,itg->alpha);CHKERRQ(ierr);
  ierr = VecMAXPY(x,itg->curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);   
  /* Note: do not change the b right hand side as is done in the publication */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessUpdate_Model1"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessUpdate_Model1(KSPFischerGuess_Model1 *itg,Vec x)
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
    ierr = VecCopy(x,itg->xtilde[curl]);CHKERRQ(ierr);

    ierr = KSP_MatMult(itg->ksp,itg->mat,itg->xtilde[curl],itg->btilde[curl]);CHKERRQ(ierr);
    ierr = VecMDot(itg->btilde[curl],curl,itg->btilde,itg->alpha);CHKERRQ(ierr);
    for (i=0; i<curl; i++) itg->alpha[i] = -itg->alpha[i];
    ierr = VecMAXPY(itg->btilde[curl],curl,itg->alpha,itg->btilde);CHKERRQ(ierr);
    ierr = VecMAXPY(itg->xtilde[curl],curl,itg->alpha,itg->xtilde);CHKERRQ(ierr);

    ierr = VecNormalize(itg->btilde[curl],&norm);CHKERRQ(ierr);
    ierr = VecScale(itg->xtilde[curl],1.0/norm);CHKERRQ(ierr);
    itg->curl++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessCreate" 
/*@C
    KSPFischerGuessCreate - Implements Paul Fischer's initial guess algorithm Model 1 and 2 for situations where
    a linear system is solved repeatedly 

  References:
      http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19940020363_1994020363.pdf

   Notes: the algorithm is different from the paper because we do not CHANGE the right hand side of the new 
    problem and solve the problem with an initial guess of zero, rather we solve the original new problem
    with a nonzero initial guess (this is done so that the linear solver convergence tests are based on
    the original RHS.)

    These are not intended to be used directly, they are called by KSP automatically when the 
    KSP option KSPSetFischerGuess(KSP,PetscInt,PetscInt) or -ksp_guess_fischer <int,int>

    This is not currently programmed as a PETSc class because there are only two models; if more models
    are introduced it should be changed.

@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessCreate(KSP ksp,PetscInt model,PetscInt maxl,KSPFischerGuess *itg)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (model == 1) {
    ierr = KSPFischerGuessCreate_Model1(ksp,maxl,(KSPFischerGuess_Model1 **)itg);CHKERRQ(ierr);
  } else if (model == 2) {
    ;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model can only be 1 or 2");
  (*itg)->model   = 1;
  (*itg)->curl    = 0;
  (*itg)->maxl    = maxl;
  (*itg)->ksp     = ksp;
  (*itg)->refcnt  = 1;
  ierr = KSPGetOperators(ksp,&(*itg)->mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessDestroy" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessDestroy(KSPFischerGuess ITG)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (--ITG->refcnt) PetscFunctionReturn(0);

  if (ITG->model == 1) {
    ierr = KSPFischerGuessDestroy_Model1((KSPFischerGuess_Model1 *)ITG);CHKERRQ(ierr);
  } else if (ITG->model == 2) {
    ;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model can only be 1 or 2");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessUpdate"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessUpdate(KSPFischerGuess itg,Vec x)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (itg->model == 1) {
    ierr = KSPFischerGuessUpdate_Model1((KSPFischerGuess_Model1 *)itg,x);CHKERRQ(ierr);
  } else if (itg->model == 2) {
    ;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model can only be 1 or 2");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPFischerGuessFormGuess"
PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessFormGuess(KSPFischerGuess itg,Vec b,Vec x)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (itg->model == 1) {
    ierr = KSPFischerGuessFormGuess_Model1((KSPFischerGuess_Model1 *)itg,b,x);CHKERRQ(ierr);
  } else if (itg->model == 2) {
    ;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model can only be 1 or 2");
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

