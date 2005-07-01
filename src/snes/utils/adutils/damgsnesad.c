#define PETSCSNES_DLL
 
#include "petscda.h"      /*I      "petscda.h"    I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/
#include "petscdmmg.h"    /*I      "petscdmmg.h"  I*/


EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFRelax_DAAD(NLF,MatSORType,PetscInt,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFRelax_DAADb(NLF,MatSORType,PetscInt,Vec);
EXTERN_C_END
EXTERN PetscErrorCode DMMGFormFunction(SNES,Vec,Vec,void *);

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
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR);CHKERRQ(ierr);
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
  ierr  = MatSetOption(*B,MAT_NEW_NONZERO_LOCATION_ERR);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#include "src/ksp/pc/impls/mg/mgimpl.h"                    /*I "petscmg.h" I*/
/*
          This is pre-beta FAS code. It's design should not be taken seriously!

              R is the usual multigrid restriction (e.g. the tranpose of peicewise linear interpolation)
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

  for (i=0; i<100; i++) {

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
      
      /* F(Q*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD,dmmg[j]->inject);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD,dmmg[j]->inject);CHKERRQ(ierr);
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
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %g\n",norm);CHKERRQ(ierr);
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
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
      }
      ierr = MatRestrict(mg[j]->restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F(Q*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD,dmmg[j]->inject);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD,dmmg[j]->inject);CHKERRQ(ierr);
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
      ierr = PetscPrintf(dmmg[0]->comm,"FAS coarse grid function norm %g\n",norm);CHKERRQ(ierr);
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
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_SELF,"Element %D value %g derivatives: ",i,*(double*)cptr);CHKERRQ(ierr);
    values = PetscADGetGradArray(cptr);
    for (j=0; j<nc; j++) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"%g ",*values++);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
    cptr += nlen;
  }

  PetscFunctionReturn(0);
}

