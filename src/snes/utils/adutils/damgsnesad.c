#define PETSCSNES_DLL
 
#include "petscda.h"      /*I      "petscda.h"    I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/
#include "petscdmmg.h"    /*I      "petscdmmg.h"  I*/
#include "../src/mat/blockinvert.h"
#include "../src/snes/impls/ls/ls.h"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFRelax_DAAD(NLF,MatSORType,PetscInt,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFRelax_DAAD4(NLF,MatSORType,PetscInt,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFRelax_DAAD9(NLF,MatSORType,PetscInt,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFRelax_DAADb(NLF,MatSORType,PetscInt,Vec);
EXTERN_C_END
EXTERN PetscErrorCode DMMGFormFunction(SNES,Vec,Vec,void *);
EXTERN PetscErrorCode SNESLSCheckLocalMin_Private(Mat,Vec,Vec,PetscReal,PetscTruth*);

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianWithAdic"
/*
    DMMGComputeJacobianWithAdic - Evaluates the Jacobian via Adic when the user has provided
    a local function evaluation routine.
*/
PetscErrorCode DMMGComputeJacobianWithAdic(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG           dmmg = (DMMG) ptr;
  PetscErrorCode ierr;
  Vec            localX;
  DA             da = (DA) dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAComputeJacobian1WithAdic(da,localX,*B,dmmg->user);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDAComputeJacobianWithAdic"
/*@
    SNESDAComputeJacobianWithAdic - This is a universal Jacobian evaluation routine
    that may be used with SNESSetJacobian() as long as the user context has a DA as
    its first record and DASetLocalAdicFunction() has been called.  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  J - Jacobian
.  B - Jacobian used in preconditioner (usally same as J)
.  flag - indicates if the matrix changed its structure
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalAdicFunction(), SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDAComputeJacobianWithAdic(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DA             da = *(DA*) ptr;
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAComputeJacobian1WithAdic(da,localX,*B,ptr);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#include "../src/ksp/pc/impls/mg/mgimpl.h"                    /*I "petscmg.h" I*/
/*
          This is pre-beta FAS code. It's design should not be taken seriously!

              R is the usual multigrid restriction (e.g. the tranpose of piecewise linear interpolation)
              Q is either a scaled injection or the usual R
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGSolveFAS"
PetscErrorCode DMMGSolveFAS(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscReal      norm;
  PC_MG          **mg;
  PC             pc;

  PetscFunctionBegin;
  ierr = VecSet(dmmg[level]->r,0.0);CHKERRQ(ierr);
  for (j=1; j<=level; j++) {
    if (!dmmg[j]->inject) {
      ierr = DMGetInjection(dmmg[j-1]->dm,dmmg[j]->dm,&dmmg[j]->inject);CHKERRQ(ierr);
    }
  }

  ierr = KSPGetPC(dmmg[level]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG**)pc->data);

  for(i = 0; i < 100; i++) {

    for(j = level; j > 0; j--) {

      /* Relax residual_fine --> F(x_fine) = 0 */
      for(k = 0; k < dmmg[j]->presmooth; k++) {
        ierr = NLFRelax_DAAD(dmmg[j]->nlf, SOR_SYMMETRIC_SWEEP, 1, dmmg[j]->x);CHKERRQ(ierr);
      }

      /* R*(residual_fine - F(x_fine)) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);

      if (j == level || dmmg[j]->monitorall) {
        /* norm( residual_fine - f(x_fine) ) */
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        if (j == level) {
          if (norm < dmmg[level]->abstol) goto theend; 
          if (i == 0) {
            dmmg[level]->rrtol = norm*dmmg[level]->rtol;
          } else {
            if (norm < dmmg[level]->rrtol) goto theend;
          }
        }
        if (dmmg[j]->monitorall) {
          for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
          ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %G\n",norm);CHKERRQ(ierr);
        }
      }

      ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F(Q*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

      /* residual_coarse = F(Q*x_fine) + R*(residual_fine - F(x_fine)) */
      ierr = VecAYPX(dmmg[j-1]->r,1.0,dmmg[j-1]->w);CHKERRQ(ierr);

      /* save Q*x_fine into b (needed when interpolating compute x back up */
      ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);
    }

    for (j=0; j<dmmg[0]->coarsesmooth; j++) {
      ierr = NLFRelax_DAAD(dmmg[0]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[0]->x);CHKERRQ(ierr);
    }
    if (dmmg[0]->monitorall){ 
      ierr = DMMGFormFunction(0,dmmg[0]->x,dmmg[0]->w,dmmg[0]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[0]->w,-1.0,dmmg[0]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[0]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level+1; k++) {ierr = PetscPrintf(dmmg[0]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %G\n",norm);CHKERRQ(ierr);
    }

    for (j=1; j<=level; j++) {
      /* x_fine = x_fine + R'*(x_coarse - Q*x_fine) */
      ierr = VecAXPY(dmmg[j-1]->x,-1.0,dmmg[j-1]->b);CHKERRQ(ierr);
      ierr = MatInterpolateAdd(mg[j]->interpolate,dmmg[j-1]->x,dmmg[j]->x,dmmg[j]->x);CHKERRQ(ierr);

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
        ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
        ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %G\n",norm);CHKERRQ(ierr);
      }

      /* Relax residual_fine - F(x_fine)  = 0 */
      for (k=0; k<dmmg[j]->postsmooth; k++) {
        ierr = NLFRelax_DAAD(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
      }

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
        ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
        ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %G\n",norm);CHKERRQ(ierr);
      }
    }

    if (dmmg[level]->monitor){
      ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(dmmg[level]->comm,"%D FAS function norm %G\n",i+1,norm);CHKERRQ(ierr);
    }
  }
  theend:
  PetscFunctionReturn(0);
}

/*
    This is the point-block version of FAS
*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASb"
PetscErrorCode DMMGSolveFASb(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscReal      norm;
  PC_MG          **mg;
  PC             pc;

  PetscFunctionBegin;
  ierr = VecSet(dmmg[level]->r,0.0);CHKERRQ(ierr);
  for (j=1; j<=level; j++) {
    if (!dmmg[j]->inject) {
      ierr = DMGetInjection(dmmg[j-1]->dm,dmmg[j]->dm,&dmmg[j]->inject);CHKERRQ(ierr);
    }
  }

  ierr = KSPGetPC(dmmg[level]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG**)pc->data);

  for (i=0; i<100; i++) {

    for (j=level; j>0; j--) {

      /* Relax residual_fine - F(x_fine) = 0 */
      for (k=0; k<dmmg[j]->presmooth; k++) {
	ierr = NLFRelax_DAADb(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
      }

      /* R*(residual_fine - F(x_fine)) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);

      if (j == level || dmmg[j]->monitorall) {
        /* norm( residual_fine - f(x_fine) ) */
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        if (j == level) {
	  if (norm < dmmg[level]->abstol) goto theend; 
          if (i == 0) {
            dmmg[level]->rrtol = norm*dmmg[level]->rtol;
          } else {
            if (norm < dmmg[level]->rrtol) goto theend;
	  }
        }
      }

      if (dmmg[j]->monitorall) {
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %G\n",norm);CHKERRQ(ierr);
      }
      ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F(Q*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

      /* residual_coarse = F(Q*x_fine) + R*(residual_fine - F(x_fine)) */
      ierr = VecAYPX(dmmg[j-1]->r,1.0,dmmg[j-1]->w);CHKERRQ(ierr);

      /* save Q*x_fine into b (needed when interpolating compute x back up */
      ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);
    }

    for (j=0; j<dmmg[0]->coarsesmooth; j++) {
      ierr = NLFRelax_DAADb(dmmg[0]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[0]->x);CHKERRQ(ierr);
    }
    if (dmmg[0]->monitorall){ 
      ierr = DMMGFormFunction(0,dmmg[0]->x,dmmg[0]->w,dmmg[0]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[0]->w,-1.0,dmmg[0]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[0]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level+1; k++) {ierr = PetscPrintf(dmmg[0]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %G\n",norm);CHKERRQ(ierr);
    }

    for (j=1; j<=level; j++) {
      /* x_fine = x_fine + R'*(x_coarse - Q*x_fine) */
      ierr = VecAXPY(dmmg[j-1]->x,-1.0,dmmg[j-1]->b);CHKERRQ(ierr);
      ierr = MatInterpolateAdd(mg[j]->interpolate,dmmg[j-1]->x,dmmg[j]->x,dmmg[j]->x);CHKERRQ(ierr);

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
	ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
	ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %G\n",norm);CHKERRQ(ierr);
      }

      /* Relax residual_fine - F(x_fine)  = 0 */
      for (k=0; k<dmmg[j]->postsmooth; k++) {
	ierr = NLFRelax_DAADb(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
      }

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
	ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
	ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %G\n",norm);CHKERRQ(ierr);
      }
    }

    if (dmmg[level]->monitor){
      ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(dmmg[level]->comm,"%D FAS function norm %G\n",i+1,norm);CHKERRQ(ierr);
    }
  }
  theend:
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#include "adic/ad_utils.h"
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscADView"
PetscErrorCode PetscADView(PetscInt N,PetscInt nc,double *ptr,PetscViewer viewer)
{
  PetscInt       i,j,nlen  = PetscADGetDerivTypeSize();
  char           *cptr = (char*)ptr;
  double         *values;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<N; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Element %D value %G derivatives: ",i,*(double*)cptr);CHKERRQ(ierr);
    values = PetscADGetGradArray(cptr);
    for (j=0; j<nc; j++) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"%G ",*values++);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
    cptr += nlen;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFAS4"
PetscErrorCode DMMGSolveFAS4(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscReal      norm;
  PetscScalar    zero = 0.0,mone = -1.0,one = 1.0;
  PC_MG          **mg;
  PC             pc;

  PetscFunctionBegin;
  ierr = VecSet(dmmg[level]->r,zero);CHKERRQ(ierr);
  for (j=1; j<=level; j++) {
    if (!dmmg[j]->inject) {
      ierr = DMGetInjection(dmmg[j-1]->dm,dmmg[j]->dm,&dmmg[j]->inject);CHKERRQ(ierr);
    }
  }

  ierr = KSPGetPC(dmmg[level]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG**)pc->data);
  for (i=0; i<100; i++) {

    for (j=level; j>0; j--) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"I am here");CHKERRQ(ierr);
      /* Relax residual_fine - F(x_fine) = 0 */
      for (k=0; k<dmmg[j]->presmooth; k++) {
        ierr = NLFRelax_DAAD4(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
      }

      /* R*(residual_fine - F(x_fine)) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAYPX(dmmg[j]->w,mone,dmmg[j]->r);CHKERRQ(ierr);

      if (j == level || dmmg[j]->monitorall) {
        /* norm( residual_fine - f(x_fine) ) */
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        if (j == level) {
          if (norm < dmmg[level]->abstol) goto theend; 
          if (i == 0) {
            dmmg[level]->rrtol = norm*dmmg[level]->rtol;
          } else {
            if (norm < dmmg[level]->rrtol) goto theend;
          }
        }
      }  ierr = PetscPrintf(PETSC_COMM_WORLD,"I am here");CHKERRQ(ierr);

      if (dmmg[j]->monitorall) {
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
      }
      ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F(R*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

      /* residual_coarse = F(R*x_fine) + R*(residual_fine - F(x_fine)) */
      ierr = VecAYPX(dmmg[j-1]->r,one,dmmg[j-1]->w);CHKERRQ(ierr);

      /* save R*x_fine into b (needed when interpolating compute x back up */
      ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);
    }

    for (j=0; j<dmmg[0]->presmooth; j++) {
      ierr = NLFRelax_DAAD4(dmmg[0]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[0]->x);CHKERRQ(ierr);
    }
    if (dmmg[0]->monitorall){ 
      ierr = DMMGFormFunction(0,dmmg[0]->x,dmmg[0]->w,dmmg[0]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[0]->w,mone,dmmg[0]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[0]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level+1; k++) {ierr = PetscPrintf(dmmg[0]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %g\n",norm);CHKERRQ(ierr);
    }

    for (j=1; j<=level; j++) {
      /* x_fine = x_fine + R'*(x_coarse - R*x_fine) */
      ierr = VecAXPY(dmmg[j-1]->x,mone,dmmg[j-1]->b);CHKERRQ(ierr);
      ierr = MatInterpolateAdd(mg[j]->interpolate,dmmg[j-1]->x,dmmg[j]->x,dmmg[j]->x);CHKERRQ(ierr);

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
        ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
        ierr = VecAXPY(dmmg[j]->w,mone,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
      }

      /* Relax residual_fine - F(x_fine)  = 0 */
      for (k=0; k<dmmg[j]->postsmooth; k++) {
        ierr = NLFRelax_DAAD4(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
      }

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
        ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
        ierr = VecAXPY(dmmg[j]->w,mone,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
      }
    }

    if (dmmg[level]->monitor){
      ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(dmmg[level]->comm,"%D FAS function norm %g\n",i,norm);CHKERRQ(ierr);
    }
  }
  theend:
  PetscFunctionReturn(0);
}

/*
   This function provide several FAS v_cycle iteration 

   iter: the number of FAS it run         

*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASn"
PetscErrorCode DMMGSolveFASn(DMMG *dmmg,PetscInt level,PetscInt iter)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscReal      norm;
  PC_MG          **mg;
  PC             pc;

  PetscFunctionBegin;
  ierr = VecSet(dmmg[level]->r,0.0);CHKERRQ(ierr);
  for (j=1; j<=level; j++) {
    if (!dmmg[j]->inject) {
      ierr = DMGetInjection(dmmg[j-1]->dm,dmmg[j]->dm,&dmmg[j]->inject);CHKERRQ(ierr);
    }
  }

  ierr = KSPGetPC(dmmg[level]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG**)pc->data);

  for (i=0; i<iter; i++) {

    for (j=level; j>0; j--) {

      /* Relax residual_fine - F(x_fine) = 0 */
      for (k=0; k<dmmg[j]->presmooth; k++) {
	ierr = NLFRelax_DAAD(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
      }

      /* R*(residual_fine - F(x_fine)) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);

      if (j == level || dmmg[j]->monitorall) {
        /* norm( residual_fine - f(x_fine) ) */
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        if (j == level) {
	  if (norm < dmmg[level]->abstol) goto theend; 
          if (i == 0) {
            dmmg[level]->rrtol = norm*dmmg[level]->rtol;
          } else {
            if (norm < dmmg[level]->rrtol) goto theend;
	  }
        }
      }

      if (dmmg[j]->monitorall) {
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
      }
      ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F(RI*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

      /* residual_coarse = F(RI*x_fine) + R*(residual_fine - F(x_fine)) */
      ierr = VecAYPX(dmmg[j-1]->r,1.0,dmmg[j-1]->w);CHKERRQ(ierr);

      /* save RI*x_fine into b (needed when interpolating compute x back up */
      ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);
    }

    for (j=0; j<dmmg[0]->coarsesmooth; j++) {
      ierr = NLFRelax_DAAD(dmmg[0]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[0]->x);CHKERRQ(ierr);
    }
    if (dmmg[0]->monitorall){ 
      ierr = DMMGFormFunction(0,dmmg[0]->x,dmmg[0]->w,dmmg[0]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[0]->w,-1.0,dmmg[0]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[0]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level+1; k++) {ierr = PetscPrintf(dmmg[0]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %g\n",norm);CHKERRQ(ierr);
    }

    for (j=1; j<=level; j++) {
      /* x_fine = x_fine + R'*(x_coarse - RI*x_fine) */
      ierr = VecAXPY(dmmg[j-1]->x,-1.0,dmmg[j-1]->b);CHKERRQ(ierr);
      ierr = MatInterpolateAdd(mg[j]->interpolate,dmmg[j-1]->x,dmmg[j]->x,dmmg[j]->x);CHKERRQ(ierr);

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
	ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
	ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
      }

      /* Relax residual_fine - F(x_fine)  = 0 */
      for (k=0; k<dmmg[j]->postsmooth; k++) {
	ierr = NLFRelax_DAAD(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
      }

      if (dmmg[j]->monitorall) {
        /* norm( F(x_fine) - residual_fine ) */
	ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
	ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
        ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
      }
    }

    if (dmmg[level]->monitor){
      ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(dmmg[level]->comm,"%D FAS function norm %g\n",i+1,norm);CHKERRQ(ierr);
    }
  }
  theend:
  PetscFunctionReturn(0);
}
/*
          This is a simple FAS setup function
*/


#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASSetUp"
PetscErrorCode DMMGSolveFASSetUp(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       j;//,nlevels=dmmg[0]->nlevels-1;
  //PC             pc;

  PetscFunctionBegin;
  ierr = VecSet(dmmg[level]->r,0.0);CHKERRQ(ierr);
  for (j=1; j<=level; j++) {
    if (!dmmg[j]->inject) {
      ierr = DMGetInjection(dmmg[j-1]->dm,dmmg[j]->dm,&dmmg[j]->inject);CHKERRQ(ierr);
    }
  }
  ierr = VecSet(dmmg[level]->r,0.0);CHKERRQ(ierr); 
  dmmg[level]->rrtol = 0.0001*dmmg[level]->rtol;//I want to get more precise solution with FAS
   PetscFunctionReturn(0);
}


/*
  This is function to implement multiplicative FAS 


Options:
 
-dmmg_fas_cycles 1 : FAS v-cycle
                 2 : FAS w-cycle


*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASMCycle"
PetscErrorCode DMMGSolveFASMCycle(DMMG *dmmg,PetscInt level,PetscTruth* converged)
{
  PetscErrorCode ierr;
  PetscInt       j,k,cycles=1,nlevels=level;//nlevels=dmmg[0]->nlevels-1; 
                  // I need to put nlevels=level in order to get grid sequence correctly
  PetscReal      norm;
  PC_MG          **mg;
  PC             pc;
  
  PetscFunctionBegin;
       

  ierr = PetscOptionsGetInt(PETSC_NULL,"-dmmg_fas_cycles",&cycles,PETSC_NULL);CHKERRQ(ierr);
 
  ierr = KSPGetPC(dmmg[level]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG**)pc->data); 

  j=level;

  if(j) {/* not the coarsest level */
    /* Relax residual_fine - F(x_fine) = 0 */
    for (k=0; k<dmmg[j]->presmooth; k++) {
      ierr = NLFRelax_DAAD(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
    }
     
     

    /* R*(residual_fine - F(x_fine)) */
    ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
    ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);

    if (j == nlevels || dmmg[j]->monitorall) {
      /* norm( residual_fine - f(x_fine) ) */
      ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
     
      if (j == nlevels) {
	if (norm < dmmg[level]->abstol) {
	  *converged = PETSC_TRUE;
           goto theend; 
	}

	  if (norm < dmmg[level]->rrtol){
          *converged = PETSC_TRUE;
          goto theend;
	  
	}
      }
    }

    if (dmmg[j]->monitorall) {
      for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
    }
    ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
    /* F(RI*x_fine) */
    ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

    /* residual_coarse = F(RI*x_fine) + R*(residual_fine - F(x_fine)) */
    ierr = VecAYPX(dmmg[j-1]->r,1.0,dmmg[j-1]->w);CHKERRQ(ierr);

    /* save RI*x_fine into b (needed when interpolating compute x back up */
    ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);
     
    
    while (cycles--) {
      ierr = DMMGSolveFASMCycle(dmmg,level-1,converged);
    }
  }
  else { /* for the coarsest level */
    for (k=0; k<dmmg[0]->coarsesmooth; k++) {
      ierr = NLFRelax_DAAD(dmmg[0]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[0]->x);CHKERRQ(ierr);
    }
    if (dmmg[0]->monitorall){ 
      ierr = DMMGFormFunction(0,dmmg[0]->x,dmmg[0]->w,dmmg[0]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[0]->w,-1.0,dmmg[0]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[0]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level+1; k++) {ierr = PetscPrintf(dmmg[0]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %g\n",norm);CHKERRQ(ierr);
    }
 if (j == nlevels || dmmg[j]->monitorall) {
      /* norm( residual_fine - f(x_fine) ) */
      ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
     
      if (j == nlevels) {
	if (norm < dmmg[level]->abstol) {
	  *converged = PETSC_TRUE;
           goto theend; 
	}
	
	  if (norm < dmmg[level]->rrtol){
          *converged = PETSC_TRUE;
          goto theend;
	  
	}
      }
    }

    
  }
  j=level;
  if(j) { /* not for the coarsest level */
    /* x_fine = x_fine + R'*(x_coarse - RI*x_fine) */
    ierr = VecAXPY(dmmg[j-1]->x,-1.0,dmmg[j-1]->b);CHKERRQ(ierr);
    ierr = MatInterpolateAdd(mg[j]->interpolate,dmmg[j-1]->x,dmmg[j]->x,dmmg[j]->x);CHKERRQ(ierr);

    if (dmmg[j]->monitorall) {
      /* norm( F(x_fine) - residual_fine ) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
    }

    /* Relax residual_fine - F(x_fine)  = 0 */
    for (k=0; k<dmmg[j]->postsmooth; k++) {
      ierr = NLFRelax_DAAD(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
    }

    if (dmmg[j]->monitorall) {
      /* norm( F(x_fine) - residual_fine ) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
    }
  
  
}
theend:
PetscFunctionReturn(0);
}

/*
  This is function to implement multiplicative FAS with block smoother


Options:
 
-dmmg_fas_cycles 1 : FAS v-cycle
                 2 : FAS w-cycle


*/


#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASMCycle9"
PetscErrorCode DMMGSolveFASMCycle9(DMMG *dmmg,PetscInt level,PetscTruth* converged)
{
  PetscErrorCode ierr;
  PetscInt       j,k,cycles=1,nlevels=level;//nlevels=dmmg[0]->nlevels-1; 
                  // I need to put nlevels=level in order to get grid sequence correctly
  PetscReal      norm;
  PC_MG          **mg;
  PC             pc;
  
  PetscFunctionBegin;
       
  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dmmg_fas_cycles",&cycles,PETSC_NULL);CHKERRQ(ierr);
 
  ierr = KSPGetPC(dmmg[level]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG**)pc->data); 
  //   for (j=level; j>0; j--) {
  j=level;
  //ierr = PetscPrintf(dmmg[level]->comm,"j=%d,nlevels=%d",j,nlevels);CHKERRQ(ierr);
  if(j) {/* not the coarsest level */
    /* Relax residual_fine - F(x_fine) = 0 */
    for (k=0; k<dmmg[j]->presmooth; k++) {
      ierr = NLFRelax_DAAD9(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
    }
     
     

    /* R*(residual_fine - F(x_fine)) */
    ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
    ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);

    if (j == nlevels || dmmg[j]->monitorall) {
      /* norm( residual_fine - f(x_fine) ) */
      ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
     
      if (j == nlevels) {
	if (norm < dmmg[level]->abstol) {
	  *converged = PETSC_TRUE;
           goto theend; 
	}
	/*	if (i == 0) {
	  dmmg[level]->rrtol = norm*dmmg[level]->rtol;
	  } else {*/
	  if (norm < dmmg[level]->rrtol){
          *converged = PETSC_TRUE;
          goto theend;
	  
	}
      }
    }

    if (dmmg[j]->monitorall) {
      for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
    }
    ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
    /* F(RI*x_fine) */
    ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

    /* residual_coarse = F(RI*x_fine) + R*(residual_fine - F(x_fine)) */
    ierr = VecAYPX(dmmg[j-1]->r,1.0,dmmg[j-1]->w);CHKERRQ(ierr);

    /* save RI*x_fine into b (needed when interpolating compute x back up */
    ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);
     
    
    while (cycles--) {
      ierr = DMMGSolveFASMCycle9(dmmg,level-1,converged);
    }
  }
  else { /* for the coarsest level */
    for (k=0; k<dmmg[0]->coarsesmooth; k++) {
      ierr = NLFRelax_DAAD9(dmmg[0]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[0]->x);CHKERRQ(ierr);
    }
    if (dmmg[0]->monitorall){ 
      ierr = DMMGFormFunction(0,dmmg[0]->x,dmmg[0]->w,dmmg[0]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[0]->w,-1.0,dmmg[0]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[0]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level+1; k++) {ierr = PetscPrintf(dmmg[0]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %g\n",norm);CHKERRQ(ierr);
    }
 if (j == nlevels || dmmg[j]->monitorall) {
      /* norm( residual_fine - f(x_fine) ) */
      ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
     
      if (j == nlevels) {
	if (norm < dmmg[level]->abstol) {
	  *converged = PETSC_TRUE;
           goto theend; 
	}
	/*	if (i == 0) {
	  dmmg[level]->rrtol = norm*dmmg[level]->rtol;
	  } else {*/
	  if (norm < dmmg[level]->rrtol){
          *converged = PETSC_TRUE;
          goto theend;
	  
	}
      }
    }

    
  }
  j=level;
  if(j) { /* not for the coarsest level */
    /* x_fine = x_fine + R'*(x_coarse - RI*x_fine) */
    ierr = VecAXPY(dmmg[j-1]->x,-1.0,dmmg[j-1]->b);CHKERRQ(ierr);
    ierr = MatInterpolateAdd(mg[j]->interpolate,dmmg[j-1]->x,dmmg[j]->x,dmmg[j]->x);CHKERRQ(ierr);

    if (dmmg[j]->monitorall) {
      /* norm( F(x_fine) - residual_fine ) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
    }

    /* Relax residual_fine - F(x_fine)  = 0 */
    for (k=0; k<dmmg[j]->postsmooth; k++) {
      ierr = NLFRelax_DAAD9(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
    }

    if (dmmg[j]->monitorall) {
      /* norm( F(x_fine) - residual_fine ) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAXPY(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);
      ierr = VecNorm(dmmg[j]->w,NORM_2,&norm);CHKERRQ(ierr);
      for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
      ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
    }
  
    /* if(j==nlevels){
     if (dmmg[level]->monitor){
    ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
    ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
      
    ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
    
    }
    }*/
}
theend:
PetscFunctionReturn(0);
}

/*
  This is function to implement full FAS with block smoother(9 points together)


Options:
 
-dmmg_fas_cycles 1 : FAS v-cycle
                 2 : FAS w-cycle


*/


#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASFCycle"
PetscErrorCode DMMGSolveFASFCycle(DMMG *dmmg,PetscInt l,PetscTruth* converged)
{
  PetscErrorCode ierr;
  PetscInt       j;//l = dmmg[0]->nlevels-1;
  PC_MG          **mg;
  PC             pc;
  PetscFunctionBegin;

  ierr = KSPGetPC(dmmg[l]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG**)pc->data); 
  // restriction all the way down to the coarse level 
  if(l>0) {
    for(j=l;j>0;j--) {
  
      /* R*(residual_fine - F(x_fine)) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);

      ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F(RI*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

      /* residual_coarse = F(RI*x_fine) + R*(residual_fine - F(x_fine)) */
      ierr = VecAYPX(dmmg[j-1]->r,1.0,dmmg[j-1]->w);CHKERRQ(ierr);

      /* save RI*x_fine into b (needed when interpolating compute x back up */
      ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);

    }

    // all the way up to the finest level
    for (j=0; j<l; j++) {
      ierr = DMMGSolveFASMCycle(dmmg,j,PETSC_NULL);CHKERRQ(ierr);
      /* x_fine = x_fine + R'*(x_coarse - RI*x_fine) */
      ierr = VecAXPY(dmmg[j]->x,-1.0,dmmg[j]->b);CHKERRQ(ierr);
      ierr = MatInterpolateAdd(mg[j+1]->interpolate,dmmg[j]->x,dmmg[j+1]->x,dmmg[j+1]->x);CHKERRQ(ierr);

    }
  }
  ierr = DMMGSolveFASMCycle(dmmg,l,converged);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*
  This is function to implement full FAS  with block smoother ( 9 points together)


Options:
 
-dmmg_fas_cycles 1 : FAS v-cycle
                 2 : FAS w-cycle


*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASFCycle"
PetscErrorCode DMMGSolveFASFCycle9(DMMG *dmmg,PetscInt l,PetscTruth* converged)
{
  PetscErrorCode ierr;
  PetscInt       j;//l = dmmg[0]->nlevels-1;
  PC_MG          **mg;
  PC             pc;
  PetscFunctionBegin;

  ierr = KSPGetPC(dmmg[l]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG**)pc->data); 
  // restriction all the way down to the coarse level 
  if(l>0) {
    for(j=l;j>0;j--) {
  
      /* R*(residual_fine - F(x_fine)) */
      ierr = DMMGFormFunction(0,dmmg[j]->x,dmmg[j]->w,dmmg[j]);CHKERRQ(ierr);
      ierr = VecAYPX(dmmg[j]->w,-1.0,dmmg[j]->r);CHKERRQ(ierr);

      ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F(RI*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->inject,dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

      /* residual_coarse = F(RI*x_fine) + R*(residual_fine - F(x_fine)) */
      ierr = VecAYPX(dmmg[j-1]->r,1.0,dmmg[j-1]->w);CHKERRQ(ierr);

      /* save RI*x_fine into b (needed when interpolating compute x back up */
      ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);

    }

    // all the way up to the finest level
    for (j=0; j<l; j++) {
      ierr = DMMGSolveFASMCycle9(dmmg,j,PETSC_NULL);CHKERRQ(ierr);
      /* x_fine = x_fine + R'*(x_coarse - RI*x_fine) */
      ierr = VecAXPY(dmmg[j]->x,-1.0,dmmg[j]->b);CHKERRQ(ierr);
      ierr = MatInterpolateAdd(mg[j+1]->interpolate,dmmg[j]->x,dmmg[j+1]->x,dmmg[j+1]->x);CHKERRQ(ierr);

    }
  }
  ierr = DMMGSolveFASMCycle9(dmmg,l,converged);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
          This is function is to solve nonlinear system with FAS

Options:

-dmmg_fas_9:    using block smoother 
 
-dmmg_fas_full: using full FAS


*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASCycle"
PetscErrorCode DMMGSolveFASCycle(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscTruth     converged = PETSC_FALSE, flg = PETSC_FALSE,flgb = PETSC_FALSE;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr =  DMMGSolveFASSetUp(dmmg,level);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmmg_fas_9",&flgb,PETSC_NULL);CHKERRQ(ierr);
 
  if(flgb){

    ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmmg_fas_full",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      for(i=0;i<1000;i++){
	ierr = PetscPrintf(dmmg[level]->comm,"%D ",i+1);CHKERRQ(ierr);
	ierr = DMMGSolveFASFCycle9(dmmg,level,&converged);CHKERRQ(ierr); 
	if (dmmg[level]->monitor){
	  ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
	  ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
	  ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
	}
	if (converged) PetscFunctionReturn(0);
      }
    }
    else{
      for(i=0;i<1000;i++){
	ierr = PetscPrintf(dmmg[level]->comm,"%D ",i+1);CHKERRQ(ierr);
	ierr = DMMGSolveFASMCycle9(dmmg,level,&converged);CHKERRQ(ierr);
	if (dmmg[level]->monitor){
	  ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
	  ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
	  ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
	}

	if (converged) PetscFunctionReturn(0);
      }
    }
  }
  else {
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmmg_fas_full",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      for(i=0;i<1000;i++){
	ierr = PetscPrintf(dmmg[level]->comm,"%D ",i+1);CHKERRQ(ierr);
	ierr = DMMGSolveFASFCycle(dmmg,level,&converged);CHKERRQ(ierr); 
	if (dmmg[level]->monitor){
	  ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
	  ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
	  ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
	}
	if (converged) PetscFunctionReturn(0);
      }
    }
    else{
      for(i=0;i<1000;i++){
	ierr = PetscPrintf(dmmg[level]->comm,"%D ",i+1);CHKERRQ(ierr);
	ierr = DMMGSolveFASMCycle(dmmg,level,&converged);CHKERRQ(ierr);
	if (dmmg[level]->monitor){
	  ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
	  ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
	  ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
	}

	if (converged) PetscFunctionReturn(0);
      }

    }
  }
  PetscFunctionReturn(0);
} 

/*
          This is function is to implement one  FAS iteration 

Options:

-dmmg_fas_9:    using block smoother 
 
-dmmg_fas_full: using full FAS

*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFASCyclen"
PetscErrorCode DMMGSolveFASCyclen(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscTruth     converged = PETSC_FALSE, flg = PETSC_FALSE,flgb = PETSC_FALSE;
  PetscReal      norm;
  // PC_MG          **mg;
  //PC             pc;

  PetscFunctionBegin;
  ierr =  DMMGSolveFASSetUp(dmmg,level);CHKERRQ(ierr);
     ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmmg_fas_9",&flgb,PETSC_NULL);CHKERRQ(ierr);
 
  if(flgb){

    ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmmg_fas_full",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
    
	ierr = DMMGSolveFASFCycle9(dmmg,level,&converged);CHKERRQ(ierr); 
	if (dmmg[level]->monitor){
	  ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
	  ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
	  ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
	}

    }
    else{
 
	ierr = DMMGSolveFASMCycle9(dmmg,level,&converged);CHKERRQ(ierr);
	if (dmmg[level]->monitor){
	  ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
	  ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
	  ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
	}


    }
  }
  else {
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmmg_fas_full",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
   
	ierr = DMMGSolveFASFCycle(dmmg,level,&converged);CHKERRQ(ierr); 
	if (dmmg[level]->monitor){
	  ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
	  ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
	  ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
	}

    }
    else{
      
	ierr = DMMGSolveFASMCycle(dmmg,level,&converged);CHKERRQ(ierr);
	if (dmmg[level]->monitor){
	  ierr = DMMGFormFunction(0,dmmg[level]->x,dmmg[level]->w,dmmg[level]);CHKERRQ(ierr);
	  ierr = VecNorm(dmmg[level]->w,NORM_2,&norm);CHKERRQ(ierr);
	  ierr = PetscPrintf(dmmg[level]->comm," FAS function norm %g\n",norm);CHKERRQ(ierr);
	}



    }
  }
  

  PetscFunctionReturn(0);
} 


/*

This is function to implement Nonlinear CG to accelerate FAS

In order to use this acceleration, the option is

-dmmg_fas_NCG



*/


#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFAS_NCG"
PetscErrorCode DMMGSolveFAS_NCG(DMMG *dmmg, PetscInt level)
{ 
  SNES           snes = dmmg[level]->snes;
  SNES_LS        *neP = (SNES_LS*)snes->data;
  PetscErrorCode ierr;
  PetscInt       maxits,i,lits;
  PetscTruth     lssucceed;
  // MatStructure   flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal      fnorm,gnorm,xnorm,ynorm,betaFR,betaPR,beta,betaHS,betaDY;
  Vec            Y,X,F,G,W,Gradold,Sk;
  //KSP            ksp;

  PetscFunctionBegin;

  ierr = VecDuplicate(dmmg[level]->x,&Sk);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dmmg[level]->x);CHKERRQ(ierr);
  if (snes->vec_sol) { ierr = VecDestroy(snes->vec_sol);CHKERRQ(ierr); }
  snes->vec_sol = dmmg[level]->x;
  if (!snes->setupcalled) { ierr = SNESSetUp(snes);CHKERRQ(ierr); }
  if (snes->conv_hist_reset) snes->conv_hist_len = 0;
  ierr = PetscLogEventBegin(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
  snes->nfuncs = 0; snes->linear_its = 0; snes->numFailures = 0;


  snes->reason  = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  
  X    = dmmg[level]->x;
  ierr = VecCopy(X,Y);CHKERRQ(ierr); 
  ierr = VecCopy(X,G);CHKERRQ(ierr); 

  // to get the residual for the F
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  
  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);	/* fnorm <- ||F||  */
  if (fnorm != fnorm) SETERRQ(PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  SNESMonitor(snes,0,fnorm);

  if (fnorm < snes->abstol) {snes->reason = SNES_CONVERGED_FNORM_ABS; PetscFunctionReturn(0);}

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  // dmmg[level]->rrtol= snes->ttol;

  // set this to store the old grad 
  Gradold=snes->vec_sol_update;
  
  // compute the search direction Y
  // I need to put Q(x)=x-FAS(x) here
  ierr = DMMGSolveFASCyclen(dmmg,level);CHKERRQ(ierr);
  //  F    = X - dmmg[level]->x; this is the gradient direction
  ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr); 
  // copy the gradient to the old 
  ierr = VecCopy(Y,Gradold);CHKERRQ(ierr);
  // copy X back  
  ierr = VecCopy(G,X);CHKERRQ(ierr); 
  ierr = VecWAXPY(Sk,-1.0,X,X);CHKERRQ(ierr);
  // so far I put X=X_c, F= F(x_c),  Gradold= Y=grad(x_c)

  //  for (i=0; i<maxits; i++) {

   for (i=0; i<10000; i++) {


    ierr = PetscPrintf(PETSC_COMM_WORLD,"iter=%d",i+1);CHKERRQ(ierr);
 
   
    // X=x_c, F=F(x_c),Y search direction; G=F(x_new), W=x_new, 
    ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); 
    ierr = (*neP->LineSearch)(snes,neP->lsP,X,F,G,Y,W,fnorm,xnorm,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"SNESSolve_LS: fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",fnorm,gnorm,ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"step=%g,oldnorm=%g,norm=%g ",ynorm,fnorm,gnorm);CHKERRQ(ierr);
    
    fnorm=gnorm; //copy the new function norm; this will effect the line_search 
     ierr = VecWAXPY(Sk,-1.0,X,W);CHKERRQ(ierr);
    //update the new solution
    ierr=VecCopy(W,X);CHKERRQ(ierr);
    ierr=VecCopy(G,F);CHKERRQ(ierr);

   
  // compute the new search direction G
  // I need to put Q(x)=x-FAS(x) here

  ierr = DMMGSolveFASCyclen(dmmg,level);CHKERRQ(ierr); 
  //G    = X - dmmg[level]->x; G is the new gradient, Y is old gradient
 
  ierr = VecWAXPY(G,-1.0,X,W);CHKERRQ(ierr);
  // copy W back to X 
  ierr = VecCopy(W,X);CHKERRQ(ierr);
  ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);  
  ierr = VecNorm(Gradold,NORM_2,&ynorm);CHKERRQ(ierr);  
  betaFR = gnorm*gnorm/(ynorm*ynorm); //FR_beta
  
  ierr = VecWAXPY(W,-1.0,Gradold,G);CHKERRQ(ierr);
  ierr = VecDot(W,G,&gnorm);CHKERRQ(ierr);
  //  ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);  
  ierr = VecNorm(Gradold,NORM_2,&ynorm);CHKERRQ(ierr);  
  betaPR = gnorm/(ynorm*ynorm); //PR_beta
  
  if ( betaPR<-betaFR) 
    {
      beta =- betaFR;
    }
  else {
    if (betaPR>betaFR)
      {beta=betaFR;}
    else{

      beta=betaPR;
    }
  } 
  //  beta=betaFR;
  //beta=betaPR;

  // try another beta
  
     ierr = VecWAXPY(W,-1.0,Gradold,G);CHKERRQ(ierr);
     ierr = VecDot(W,G,&betaHS);CHKERRQ(ierr);
     ierr = VecDot(W,Y,&gnorm);CHKERRQ(ierr);
     betaHS=-betaHS/gnorm;
     ierr = VecDot(G,G,&betaDY);CHKERRQ(ierr);
     betaDY=-betaDY/gnorm;
     if(betaHS<betaDY)
       beta=betaHS;
     else
       beta=betaDY;
     if(beta<0)
       beta=0;
  
     ierr = PetscPrintf(PETSC_COMM_WORLD,"betaHS=%g,betaDY=%g\n",betaHS,betaDY);CHKERRQ(ierr);
  

 
    // compute the c_2
    ierr = VecDot(G,Y,&gnorm);CHKERRQ(ierr);
    ierr = VecDot(Gradold,Y,&ynorm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"beta=%g,c_2=%g\n",beta,fabs(gnorm/ynorm));CHKERRQ(ierr);
     ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
    ierr = VecNorm(Y,NORM_2,&ynorm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"size=%g\n",fabs(gnorm/(beta*ynorm)));CHKERRQ(ierr);

    // update the direction: Y= G + beta * Y 
    ierr = VecAXPBY(Y,1.0,beta,G);CHKERRQ(ierr);
    ierr = VecCopy(G,Gradold);CHKERRQ(ierr);
    //ierr =VecCopy(G,Y);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,fnorm,lits);
    SNESMonitor(snes,i+1,fnorm);
    
     if (!lssucceed) { 
      PetscTruth ismin;
      beta=0;
      if (++snes->numFailures >= snes->maxFailures) {
      snes->reason = SNES_DIVERGED_LS_FAILURE;
        ierr = SNESLSCheckLocalMin_Private(snes->jacobian,F,W,fnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
      } 
    

    /* Test for convergence */
    if (snes->ops->converged) {
      ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm = || X || */
      ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,1.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
      if (snes->reason) {
        break;
      }
    }
  } 
  if (X != snes->vec_sol) {
    ierr = VecCopy(X,snes->vec_sol);CHKERRQ(ierr);
  }
  if (F != snes->vec_func) {
    ierr = VecCopy(F,snes->vec_func);CHKERRQ(ierr);
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"SNESSolve_LS: Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    snes->reason = SNES_DIVERGED_MAX_IT;
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"reason=%d\n",snes->reason);CHKERRQ(ierr);
  // VecView(X,PETSC_VIEWER_STDOUT_WORLD);
  PetscFunctionReturn(0);
}



/*

This is function to implement NonGMRES  to accelerate FAS

In order to use this acceleration, the option is

-dmmg_fas_NGMRES


Options:

-dmmg_fas_ngmres_m : the number of previous solutions to keep

*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFAS_NGMRES"
PetscErrorCode DMMGSolveFAS_NGMRES(DMMG *dmmg, PetscInt level)
{ 
  SNES           snes = dmmg[level]->snes;
   PetscErrorCode ierr;
  PetscInt       maxits=10000,i,k,l,j,subm=3,iter;
 ierr = PetscOptionsGetInt(PETSC_NULL,"-dmmg_fas_ngmres_m",&subm,PETSC_NULL);CHKERRQ(ierr);

  PetscTruth     restart=PETSC_FALSE, selectA=PETSC_FALSE;
  PetscReal      fnorm,gnorm,dnorm,dnormtemp,dminnorm,fminnorm,tol=1.e-12,gammaA=2,epsilonB=0.1,deltaB=0.9,gammaC;
  Vec            X,F,G,W,D,u[subm],res[subm];
   PetscScalar    H[subm][subm],q[subm][subm],beta[subm],xi[subm],alpha[subm],alphasum,det,Hinv[16];

 PetscFunctionBegin;

 gammaC=2; if (gammaA>gammaC) gammaC=gammaA;
 ierr = VecDuplicate(dmmg[level]->x,&X);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&F);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&W);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&G);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&D);CHKERRQ(ierr);

 for(i=0;i<subm;i++) {/* get the space for the solution */
   ierr = VecDuplicate(dmmg[level]->x,&u[i]);CHKERRQ(ierr);
   ierr = VecDuplicate(dmmg[level]->x,&res[i]);CHKERRQ(ierr);
 }

  X    = dmmg[level]->x;
  ierr = VecCopy(X,u[0]);CHKERRQ(ierr); 
  
  // to get the residual for the F
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  ierr = VecCopy(F,res[0]);CHKERRQ(ierr); 
  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);	/* fnorm <- ||F||  */
  fnorm=fnorm*fnorm;
  iter=1;
  restartline:
   
  q[0][0] = fnorm; fminnorm=fnorm;


   for (k=1; k<maxits; k++) {

  
     ierr = PetscPrintf(PETSC_COMM_WORLD,"\n k=%d,iter=%d fmin=%g ",k,iter++,sqrt(fminnorm));CHKERRQ(ierr);
 
    /* compute the X=u^M , F=r, fnorm =||F||*/

    ierr = DMMGSolveFASCyclen(dmmg,level);CHKERRQ(ierr); 
    ierr = SNESComputeFunction(snes,X,F); 
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
    if (fnorm < tol) { PetscFunctionReturn(0);}
    fnorm =fnorm*fnorm;
    if (fnorm<fminnorm) fminnorm=fnorm;   
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"fmin=%g",sqrt(fminnorm));CHKERRQ(ierr);
    /* compute u^A */
    //l=min(subm,k)
    l=subm;
    if (k<l) l=k;
    
    /* compute the matrix H and RHS */
 
          
    for(i=0;i<l;i++){
      ierr   = VecDot(F,res[i],&xi[i]);
      beta[i]=fnorm-xi[i];
    }
  
    for(i=0;i<l;i++){
      for(j=0;j<l;j++){
       H[i][j]=q[i][j]-xi[i]-xi[j]+fnorm;
      }
    }
    /* Here is special for subm=2 */
    if(l==1){
      //   H[0][0] = q[0][0]-xi[0]-xi[0]+fnorm;
      alpha[0]= beta[0]/H[0][0];
    }
    if(l==2){
     
      alpha[0]= ( beta[0]*H[1][1] -beta[1]*H[0][1] )/( H[0][0]*H[1][1]- H[0][1]*H[1][0]);
      alpha[1]= ( beta[1]*H[0][0] -beta[0]*H[1][0] )/( H[0][0]*H[1][1]- H[0][1]*H[1][0]);
      
    }
    if(l==3) {
    
  det = H[0][0]*(H[1][1]*H[2][2]-H[1][2]*H[2][1])-H[0][1]*(H[1][0]*H[2][2]-H[1][2]*H[2][0])+H[0][2]*(H[1][0]*H[2][1]-H[1][1]*H[2][0]);
      alpha[0]= (beta[0]*(H[1][1]*H[2][2]-H[1][2]*H[2][1])-H[0][1]*(beta[1]*H[2][2]-H[1][2]*beta[2])+H[0][2]*(beta[1]*H[2][1]-H[1][1]*beta[2]))/det;
      alpha[1]=(H[0][0]*(beta[1]*H[2][2]-H[1][2]*beta[2])-beta[0]*(H[1][0]*H[2][2]-H[1][2]*H[2][0])+H[0][2]*(H[1][0]*beta[2]-beta[1]*H[2][0]))/det;
      alpha[2]=(H[0][0]*(H[1][1]*beta[2]-beta[1]*H[2][1])-H[0][1]*(H[1][0]*beta[2]-beta[1]*H[2][0])+beta[0]*(H[1][0]*H[2][1]-H[1][1]*H[2][0]))/det;


    }
  
    if(l==4){
        Hinv[0]=H[0][0];Hinv[1]=H[0][1];Hinv[2]=H[0][2];Hinv[3]=H[0][3];
      Hinv[4]=H[1][0];Hinv[5]=H[1][1];Hinv[6]=H[1][2];Hinv[7]=H[1][3];
      Hinv[8]=H[2][0];Hinv[9]=H[2][1];Hinv[10]=H[2][2];Hinv[11]=H[2][3];
      Hinv[12]=H[3][0];Hinv[13]=H[3][1];Hinv[14]=H[3][2];Hinv[15]=H[3][3];
      Kernel_A_gets_inverse_A_4(Hinv,0.0);   
      for(i=0;i<l;i++)
        alpha[i]=Hinv[4*i]*beta[0]+Hinv[4*i+1]*beta[1]+Hinv[4*i+2]*beta[2]+Hinv[4*i+3]*beta[3];
          
    }
    alphasum=0;
    for (i=0;i<l;i++)
      alphasum=alphasum+alpha[i];
   
    /* W= u^A */
    ierr = VecCopy(X,W);CHKERRQ(ierr);
    ierr = VecAXPBY(W,0.0,1-alphasum,X);CHKERRQ(ierr);
         
    for(i=0;i<l;i++)
         ierr = VecAXPY(W,alpha[i],u[i]);CHKERRQ(ierr);
    
    /* W= F(G) */
    ierr = SNESComputeFunction(snes,W,G);  
    ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
    gnorm=gnorm*gnorm;


   
    /* select the uA or uM */
    // Criterion A 
    if(sqrt(gnorm)<gammaA*sqrt(fminnorm)){
      //ierr = PetscPrintf(PETSC_COMM_WORLD,"Crite A\n");CHKERRQ(ierr);
       selectA=PETSC_TRUE; 
    }
    // Criterion B
    
    ierr=VecCopy(W,D);CHKERRQ(ierr);   
    ierr=VecAXPY(D,-1,X);CHKERRQ(ierr);   
    ierr=VecNorm(D,NORM_2,&dnorm);CHKERRQ(ierr);     
    dminnorm=10000000;
    for(i=0;i<l;i++) {
      ierr=VecCopy(W,D);CHKERRQ(ierr);   
      ierr=VecAXPY(D,-1,u[i]);CHKERRQ(ierr);   
      ierr=VecNorm(D,NORM_2,&dnormtemp);CHKERRQ(ierr);        
      if(dnormtemp<dminnorm) dminnorm=dnormtemp;
    }
     if( epsilonB*dnorm<dminnorm || sqrt(gnorm)<deltaB*sqrt(fminnorm))
      selectA =PETSC_TRUE;
    else
      selectA=PETSC_FALSE;
    
    if(selectA){  /* uA selected */
      selectA=PETSC_FALSE; 
      //   ierr = PetscPrintf(PETSC_COMM_WORLD,"Select A\n");CHKERRQ(ierr);
      ierr = VecCopy(W,X);CHKERRQ(ierr); 
      ierr = VecCopy(G,F);CHKERRQ(ierr); 
      fnorm=gnorm;
    }
     if (fnorm<fminnorm) fminnorm=fnorm; 
     

    /* Test for convergence */
    if (sqrt(fnorm)<tol) {
      PetscFunctionReturn(0);
    }
 
    /* Test for restart */
    if(sqrt(gnorm)>gammaC*sqrt(fminnorm)) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"restart for C ");CHKERRQ(ierr);
      restart=PETSC_TRUE;
    }
    if(epsilonB*dnorm>dminnorm && sqrt(gnorm)>deltaB*sqrt(fminnorm)) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"restart for D ");CHKERRQ(ierr);
      restart=PETSC_TRUE;
      }          
    /* Prepation for the next iteration */
    
     //turn off restart
    //restart=PETSC_FALSE;
    if(restart){
      restart=PETSC_FALSE;
      goto restartline;
    }
    else {
      j=k%subm;  
      ierr = VecCopy(F,res[j]);CHKERRQ(ierr);
      ierr = VecCopy(X,u[j]);CHKERRQ(ierr);
      for(i=0;i<l;i++){
  	ierr= VecDot(F,res[i],&q[j][i]);CHKERRQ(ierr);
	q[i][j]=q[j][i];
      } 
      if(l<subm)
      q[j][j]=fnorm;
    }
   } 

  PetscFunctionReturn(0);
}

/*
This is a function to provide the function value:

x-FAS(x)

*/



#undef __FUNCT__  
#define __FUNCT__ "DMMGFASFunction"
PetscErrorCode DMMGFASFunction(SNES snes,Vec x,Vec f,void *ptr)
{ 
      DMMG*          dmmg=(DMMG*)ptr;
      PetscErrorCode ierr;
      Vec            temp;
      PetscInt       level=dmmg[0]->nlevels-1;
      PetscFunctionBegin;
 
      ierr = VecDuplicate(dmmg[level]->x,&temp);CHKERRQ(ierr);
      
     
      ierr = VecCopy(dmmg[level]->x,temp);CHKERRQ(ierr);
      ierr = VecCopy(x,dmmg[level]->x);CHKERRQ(ierr);
      ierr = VecCopy(x,f);CHKERRQ(ierr);
      
      // I need to put -F(x)=x-FAS(x) here
      ierr = DMMGSolveFASCyclen(dmmg,level);CHKERRQ(ierr); 
      ierr = VecAXPY(f,-1.0,dmmg[level]->x);CHKERRQ(ierr); 
      // y = alpha x + y. 
      ierr=VecCopy(temp,dmmg[level]->x);CHKERRQ(ierr);
      //copy W back to X
      
 PetscFunctionReturn(0);    
}



/* this function is to implement Quasi-Newton method with implicit Broyden updating methods(limit memory version)

In order to use this method, the option is -dmmg_fas_QNewton

Options:

-dmmg_fas_QNewton_m: the number of the vectors to keep for inverse of Jacobian

-dmmg_fas_initialJacobian: will use matrix-free GMRES to solve the initial Jacobian 

                        with  options -snes_mf -snes_max_it 1 -ksp_max_it n


In this function, does not have line search and nonlinear gmres acceleration

*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFAS_QNewton"
PetscErrorCode DMMGSolveFAS_QNewton(DMMG *dmmg, PetscInt level)
{ 

  

  SNES           snes = dmmg[level]->snes, snes0;
  PetscErrorCode ierr;
  PetscInt       maxits=10000,i,k,l,subm=3,subm01;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dmmg_fas_QNewton_m",&subm,PETSC_NULL);CHKERRQ(ierr);
  subm01=subm-1;
   PetscTruth   flg = PETSC_FALSE;  
   PetscReal      fnorm,gnorm,tol=1.e-12;
  Vec            X,F,G,W,D,Y,v[subm],w[subm],s0,s1,F0,F1;
 

 PetscFunctionBegin;

 ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmmg_fas_initialJacobian",&flg,PETSC_NULL);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&X);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&F);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&W);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&G);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&D);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&Y);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&s0);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&s1);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&F0);CHKERRQ(ierr);
 ierr = VecDuplicate(dmmg[level]->x,&F1);CHKERRQ(ierr);
 
 // creat a snes for solve the initial Jacobian
 ierr = SNESCreate(dmmg[level]->comm,&snes0);CHKERRQ(ierr);
 ierr = SNESSetFunction(snes0,F,DMMGFASFunction,dmmg);CHKERRQ(ierr);
 ierr = SNESSetFromOptions(snes0);CHKERRQ(ierr);

 for(i=0;i<subm;i++) {/* get the space for the solution */
   ierr = VecDuplicate(dmmg[level]->x,&v[i]);CHKERRQ(ierr);
   ierr = VecDuplicate(dmmg[level]->x,&w[i]);CHKERRQ(ierr);
 }

 //We first try B0==I
   X    = dmmg[level]->x;
 
   if(flg){
     ierr= VecAXPBY(Y,0.0,0.0,X);CHKERRQ(ierr);
     ierr= VecCopy(X,s0);CHKERRQ(ierr);
     ierr= SNESSolve(snes0,Y,s0);CHKERRQ(ierr);
     ierr= VecAXPY(s0,-1.0,X);CHKERRQ(ierr);
   }
   else{
     ierr=VecCopy(X,W);CHKERRQ(ierr);
     ierr=VecCopy(X,Y);CHKERRQ(ierr);
      
     // I need to put -F(x)=x-FAS(x) here

     ierr = DMMGSolveFASCyclen(dmmg,level);CHKERRQ(ierr); 
     ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr); 
     // y = alpha x + y. 
     ierr=VecCopy(W,X);CHKERRQ(ierr);
     //copy W back to X
    
     // Y stores the -F(x) 
     ierr= VecAXPBY(Y,0.0,-1.0,X);CHKERRQ(ierr);
     ierr= VecCopy(Y,s0);CHKERRQ(ierr);

   }
 
   ierr = VecAXPY(X,1.0,s0);CHKERRQ(ierr); 
 
for(k=0; k<maxits; k++){
     
 /* Test for convergence */
   ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
   ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);	/* fnorm <- ||F||  */
   ierr = PetscPrintf(PETSC_COMM_WORLD,"k=%d, fnorm=%g\n",
		       k,fnorm);CHKERRQ(ierr);
    if (fnorm<tol) {
      PetscFunctionReturn(0);
    }
 

    if(flg){
      //     ierr= SNESSolve(snes0,Y,s1);CHKERRQ(ierr);
      ierr= VecAXPBY(Y,0.0,0.0,X);CHKERRQ(ierr);
      ierr= VecCopy(dmmg[level]->x,s1);CHKERRQ(ierr);
      ierr= SNESSolve(snes0,Y,s1);CHKERRQ(ierr);
      ierr= VecAXPY(s1,-1.0,X);CHKERRQ(ierr);
 
    }
    else{
      ierr=VecCopy(X,W);CHKERRQ(ierr);
      ierr=VecCopy(X,Y);CHKERRQ(ierr);
      
      // I need to put -F(x)=x-FAS(x) here

      ierr = DMMGSolveFASCyclen(dmmg,level);CHKERRQ(ierr); 
      ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr); 
      // y = alpha x + y. 
      ierr=VecCopy(W,X);CHKERRQ(ierr);
      //copy W back to X
 
      //So far, I got X=x_k, Y=-F(x_k) 
      // I should solve the G=-B_0^{-1}F(x_k) first, but I choose B_0^{-1}=I,
      ierr=VecCopy(Y,F1);CHKERRQ(ierr);
      ierr= VecAXPBY(Y,0.0,-1.0,X);CHKERRQ(ierr);
      ierr=VecCopy(Y,s1);CHKERRQ(ierr);

    }
   
   l=subm;
   if (k<l) l=k;
   
    
   for (i=0;i<l;i++){
     // compute [I+v(i)w(i)^T]*s(k)
     ierr= VecDot(w[i],s1,&gnorm);CHKERRQ(ierr);
     ierr = VecAXPY(s1,gnorm,v[i]);CHKERRQ(ierr); 
     
   }
   if(l==subm) {
     for(i=0;i<subm01;i++){
       ierr= VecCopy(w[i+1],w[i]);CHKERRQ(ierr);
       ierr= VecCopy(v[i+1],v[i]);CHKERRQ(ierr);
     }
     l--;
   }
     ierr= VecCopy(s0,w[l]);CHKERRQ(ierr);
     ierr= VecCopy(s0,Y);  CHKERRQ(ierr);
     ierr= VecCopy(s1,v[l]);CHKERRQ(ierr);
     ierr= VecAXPY(Y,-1.0,s1);CHKERRQ(ierr); 
     ierr= VecDot(w[l],Y,&gnorm);CHKERRQ(ierr);
     ierr= VecAXPBY(v[l],0.0,1.0/gnorm,w[l]);CHKERRQ(ierr);

     ierr= VecDot(s1,w[l],&gnorm);CHKERRQ(ierr);
     ierr= VecAXPY(s1,gnorm,v[l]);CHKERRQ(ierr);
     ierr= VecCopy(s1,s0);CHKERRQ(ierr);
     ierr=VecAXPY(X,1.0,s1);CHKERRQ(ierr);
 }  
  PetscFunctionReturn(0);
}
