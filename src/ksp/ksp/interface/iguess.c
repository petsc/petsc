#define PETSCKSP_DLL

#include "include/private/kspimpl.h"

/* 
    KSPGuessFischerCreate - Implements Paul Fischer's initial guess algorithm Model 1 and 2 for situations where
    a linear system is solved repeatedly 

  References:
      http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19940020363_1994020363.pdf

   Notes: the algorithm is different from the paper because we do not CHANGE the right hand side of the new 
    problem and solve the problem with an initial guess of zero, rather we solve the original new problem
    with a nonzero initial guess (this is done so that the linear solver convergence tests are based on
    the original RHS.)

    These are not intended to be used directly, they are called by KSP automatically when the 
    KSP option KSPSetGuessFischer(KSP,PetscInt,PetscInt) or -ksp_guess_fischer <int,int>

    This is not currently programmed as a PETSc class because there are only two models; if more models
    are introduced it should be changed.

 */

typedef struct {
    PetscInt    model,    /* 1 or 2 */
                curl,     /* Current number of basis vectors */
                maxl;     /* Maximum number of basis vectors */
    Mat         mat;
    KSP         ksp;
    PetscScalar *alpha;   /* */
    Vec         *xtilde,  /* Saved x vectors */
                *btilde;  /* Saved b vectors */
} KSPGuessFischer_Model1;


#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerCreate_Model1" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerCreate_Model1(KSP ksp,int  maxl,KSPGuessFischer_Model1 **ITG)
{
  KSPGuessFischer_Model1 *itg;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscMalloc(sizeof(KSPGuessFischer_Model1),&itg);CHKERRQ(ierr);
  ierr = PetscMalloc(maxl * sizeof(PetscScalar),&itg->alpha);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSPGuessFischer_Model1) + maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->xtilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->xtilde);CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->btilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->btilde);CHKERRQ(ierr);
  *ITG = itg;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerDestroy_Model1" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerDestroy_Model1(KSPGuessFischer_Model1 *itg)
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
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerFormGuess_Model1"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerFormGuess_Model1(KSPGuessFischer_Model1 *itg,Vec b,Vec x)
{
  PetscErrorCode ierr;
  int            i;

  PetscFunctionBegin;
  PetscValidPointer(itg,2);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);  
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  for (i=1; i<=itg->curl; i++) {
    ierr = VecDot(itg->btilde[i-1],b,&(itg->alpha[i-1]));CHKERRQ(ierr);
    ierr = VecAXPY(x,itg->alpha[i-1],itg->xtilde[i-1]);CHKERRQ(ierr);
    /* Note: do not change the b right hand side as is done in the publication */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerUpdate_Model1"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerUpdate_Model1(KSPGuessFischer_Model1 *itg,Vec x)
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
    ierr = VecScale(itg->xtilde[0],norm);CHKERRQ(ierr);
    itg->curl = 1;
  } else {
    ierr = VecCopy(x,itg->xtilde[curl]);CHKERRQ(ierr);
    ierr = KSP_MatMult(itg->ksp,itg->mat,itg->xtilde[curl],itg->btilde[curl]);CHKERRQ(ierr);
    for (i=1; i<=curl; i++) {
      ierr = VecDot(itg->btilde[curl],itg->btilde[i-1],itg->alpha+i-1);CHKERRQ(ierr);
    }
    for (i=1; i<=curl; i++) {
      ierr = VecAXPY(itg->btilde[curl],-itg->alpha[i-1],itg->btilde[i-1]);CHKERRQ(ierr);
      ierr = VecAXPY(itg->xtilde[curl],-itg->alpha[i-1],itg->xtilde[i-1]);CHKERRQ(ierr);
    }
    ierr = VecNormalize(itg->btilde[curl],&norm);CHKERRQ(ierr);
    ierr = VecScale(itg->xtilde[curl],norm);CHKERRQ(ierr);
    itg->curl++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerCreate" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerCreate(KSP ksp,PetscInt model,PetscInt maxl,KSPGuessFischer *itg)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (model == 1) {
    ierr = KSPGuessFischerCreate_Model1(ksp,maxl,(KSPGuessFischer_Model1 **)itg);CHKERRQ(ierr);
  } else if (model == 2) {
    ;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model can only be 1 or 2");
  (*itg)->model = 1;
  (*itg)->curl = 0;
  (*itg)->maxl = maxl;
  (*itg)->ksp  = ksp;
  ierr = KSPGetOperators(ksp,&(*itg)->mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerDestroy" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerDestroy(KSPGuessFischer ITG)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (ITG->model == 1) {
    ierr = KSPGuessFischerDestroy_Model1((KSPGuessFischer_Model1 *)ITG);CHKERRQ(ierr);
  } else if (ITG->model == 2) {
    ;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model can only be 1 or 2");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerUpdate"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerUpdate(KSPGuessFischer itg,Vec x)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (itg->model == 1) {
    ierr = KSPGuessFischerUpdate_Model1((KSPGuessFischer_Model1 *)itg,x);CHKERRQ(ierr);
  } else if (itg->model == 2) {
    ;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model can only be 1 or 2");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerFormGuess"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerFormGuess(KSPGuessFischer itg,Vec b,Vec x)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  if (itg->model == 1) {
    ierr = KSPGuessFischerFormGuess_Model1((KSPGuessFischer_Model1 *)itg,b,x);CHKERRQ(ierr);
  } else if (itg->model == 2) {
    ;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model can only be 1 or 2");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFischerReset"
/*
    KSPGuessFischerReset - This is called whenever KSPSetOperators() is called to tell the
      initial guess object that the matrix is changed and so the initial guess object
      must restart from scratch building the subspace where the guess is computed from.
*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFischerReset(KSPGuessFischer itg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  itg->curl = 0;
  ierr = KSPGetOperators(itg->ksp,&itg->mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

