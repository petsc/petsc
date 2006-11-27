#define PETSCKSP_DLL

#include "include/private/kspimpl.h"  /*I "petscksp.h" I*/
/* 
  This code inplements Paul Fischer's initial guess code for situations where
  a linear system is solved repeatedly 
 */

typedef struct {
    int         curl,     /* Current number of basis vectors */
                maxl;     /* Maximum number of basis vectors */
    PetscScalar *alpha;   /* */
    Vec         *xtilde,  /* Saved x vectors */
                *btilde;  /* Saved b vectors */
} KSPIGUESS;

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessCreate" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessCreate(KSP ksp,int  maxl,void **ITG)
{
  KSPIGUESS *itg;
  PetscErrorCode ierr;

  *ITG = 0;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscMalloc(sizeof(KSPIGUESS),&itg);CHKERRQ(ierr);
  itg->curl = 0;
  itg->maxl = maxl;
  ierr = PetscMalloc(maxl * sizeof(PetscScalar),&itg->alpha);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSPIGUESS) + maxl*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->xtilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->xtilde);CHKERRQ(ierr);
  ierr = KSPGetVecs(ksp,maxl,&itg->btilde,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,maxl,itg->btilde);CHKERRQ(ierr);
  *ITG = (void*)itg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessDestroy" 
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessDestroy(KSP ksp,KSPIGUESS *itg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscFree(itg->alpha);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->btilde,itg->maxl);CHKERRQ(ierr);
  ierr = VecDestroyVecs(itg->xtilde,itg->maxl);CHKERRQ(ierr);
  ierr = PetscFree(itg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFormB"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFormB(KSP ksp,KSPIGUESS *itg,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(itg,2);
  PetscValidHeaderSpecific(b,VEC_COOKIE,3);  
  for (i=1; i<=itg->curl; i++) {
    ierr = VecDot(itg->btilde[i-1],b,&(itg->alpha[i-1]));CHKERRQ(ierr);
    ierr = VecAXPY(b,-itg->alpha[i-1],itg->btilde[i-1]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessFormX"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessFormX(KSP ksp,KSPIGUESS *itg,Vec x)
{
  PetscErrorCode ierr;
  int i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidPointer(itg,2);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);  
  ierr = VecCopy(x,itg->xtilde[itg->curl]);CHKERRQ(ierr);
  for (i=1; i<=itg->curl; i++) {
    ierr = VecAXPY(x,itg->alpha[i-1],itg->xtilde[i-1]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGuessUpdate"
PetscErrorCode PETSCKSP_DLLEXPORT KSPGuessUpdate(KSP ksp,Vec x,KSPIGUESS *itg)
{
  PetscReal    normax,norm;
  PetscScalar  tmp;
  MatStructure pflag;
  PetscErrorCode ierr;
  int          curl = itg->curl,i;
  Mat          Amat,Pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(itg,3);
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  if (curl == itg->maxl) {
    ierr = KSP_MatMult(ksp,Amat,x,itg->btilde[0]);CHKERRQ(ierr);
    ierr = VecNorm(itg->btilde[0],NORM_2,&normax);CHKERRQ(ierr);
    tmp = 1.0/normax; ierr = VecScale(itg->btilde[0],tmp);CHKERRQ(ierr);
    /* VCOPY(ksp->vc,x,itg->xtilde[0]); */
    ierr = VecScale(itg->xtilde[0],tmp);CHKERRQ(ierr);
  } else {
    ierr = KSP_MatMult(ksp,Amat,itg->xtilde[curl],itg->btilde[curl]);CHKERRQ(ierr);
    for (i=1; i<=curl; i++) {
      ierr = VecDot(itg->btilde[curl],itg->btilde[i-1],itg->alpha+i-1);CHKERRQ(ierr);
    }
    for (i=1; i<=curl; i++) {
      ierr = VecAXPY(itg->btilde[curl],-itg->alpha[i-1],itg->btilde[i-1]);CHKERRQ(ierr);
      ierr = VecAXPY(itg->xtilde[curl],itg->alpha[i-1],itg->xtilde[i-1]);CHKERRQ(ierr);
    }
    ierr = VecNormalize(itg->btilde[curl],&norm);CHKERRQ(ierr);
    ierr = VecNormalize(itg->xtilde[curl],&norm);CHKERRQ(ierr);
    itg->curl++;
  }
  PetscFunctionReturn(0);
}
