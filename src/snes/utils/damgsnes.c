#define PETSCSNES_DLL
 
#include "petscda.h"      /*I      "petscda.h"    I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/
#include "petscdmmg.h"    /*I      "petscdmmg.h"  I*/

/*
      period of -1 indicates update only on zeroth iteration of SNES
*/
#define ShouldUpdate(l,it) (((dmmg[l-1]->updatejacobianperiod == -1) && (it == 0)) || \
                            ((dmmg[l-1]->updatejacobianperiod >   0) && !(it % dmmg[l-1]->updatejacobianperiod)))
/*
   Evaluates the Jacobian on all of the grids. It is used by DMMG to provide the 
   ComputeJacobian() function that SNESSetJacobian() requires.
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian_Multigrid"
PetscErrorCode DMMGComputeJacobian_Multigrid(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG           *dmmg = (DMMG*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels,it;
  KSP            ksp,lksp;
  PC             pc;
  PetscTruth     ismg;
  Vec            W;
  MatStructure   flg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as user context which should contain DMMG");
  ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);

  /* compute Jacobian on finest grid */
  if (dmmg[nlevels-1]->updatejacobian && ShouldUpdate(nlevels,it)) {
    ierr = (*DMMGGetFine(dmmg)->computejacobian)(snes,X,J,B,flag,DMMGGetFine(dmmg));CHKERRQ(ierr);
  } else {
    ierr = PetscLogInfo((0,"DMMGComputeJacobian_Multigrid:Skipping Jacobian, SNES iteration %D frequence %D level %D\n",it,dmmg[nlevels-1]->updatejacobianperiod,nlevels-1));CHKERRQ(ierr);
    *flag = SAME_PRECONDITIONER;
  }
  ierr = MatSNESMFSetBase(DMMGGetFine(dmmg)->J,X);CHKERRQ(ierr);

  /* create coarser grid Jacobians for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = PCMGGetSmoother(pc,nlevels-1,&lksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(lksp,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->B,*flag);CHKERRQ(ierr);

    if (dmmg[0]->galerkin) {
      for (i=nlevels-2; i>-1; i--) {
        PetscTruth JeqB = (PetscTruth)( dmmg[i]->B == dmmg[i]->J);
        ierr = MatDestroy(dmmg[i]->B);CHKERRQ(ierr);
        ierr = MatPtAP(dmmg[i+1]->B,dmmg[i+1]->R,MAT_INITIAL_MATRIX,1.0,&dmmg[i]->B);CHKERRQ(ierr);
        if (JeqB) dmmg[i]->J = dmmg[i]->B;
	ierr = PCMGGetSmoother(pc,i,&lksp);CHKERRQ(ierr);
	ierr = KSPSetOperators(lksp,dmmg[i]->J,dmmg[i]->B,*flag);CHKERRQ(ierr);
      }   
    } else {
      for (i=nlevels-1; i>0; i--) {
	if (!dmmg[i-1]->w) {
	  ierr = VecDuplicate(dmmg[i-1]->x,&dmmg[i-1]->w);CHKERRQ(ierr);
	}
	W    = dmmg[i-1]->w;
	/* restrict X to coarser grid */
	ierr = MatRestrict(dmmg[i]->R,X,W);CHKERRQ(ierr);
	X    = W;      
	/* scale to "natural" scaling for that grid */
	ierr = VecPointwiseMult(X,X,dmmg[i]->Rscale);CHKERRQ(ierr);
	/* tell the base vector for matrix free multiplies */
	ierr = MatSNESMFSetBase(dmmg[i-1]->J,X);CHKERRQ(ierr);
	/* compute Jacobian on coarse grid */
	if (dmmg[i-1]->updatejacobian && ShouldUpdate(i,it)) {
	  ierr = (*dmmg[i-1]->computejacobian)(snes,X,&dmmg[i-1]->J,&dmmg[i-1]->B,&flg,dmmg[i-1]);CHKERRQ(ierr);
	  flg = SAME_NONZERO_PATTERN;
	} else {
	  ierr = PetscLogInfo((0,"DMMGComputeJacobian_Multigrid:Skipping Jacobian, SNES iteration %D frequence %D level %D\n",it,dmmg[i-1]->updatejacobianperiod,i-1));CHKERRQ(ierr);
	  flg = SAME_PRECONDITIONER;
	}
	ierr = PCMGGetSmoother(pc,i-1,&lksp);CHKERRQ(ierr);
	ierr = KSPSetOperators(lksp,dmmg[i-1]->J,dmmg[i-1]->B,flg);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "DMMGFormFunction"
/* 
   DMMGFormFunction - This is a universal global FormFunction used by the DMMG code
   when the user provides a local function.

   Input Parameters:
+  snes - the SNES context
.  X - input vector
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector

 */
PetscErrorCode DMMGFormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  PetscErrorCode ierr;
  Vec            localX;
  DA             da = (DA)dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAFormFunction1(da,localX,F,dmmg->user);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "SNESDAFormFunction"
/*@C 
   SNESDAFormFunction - This is a universal function evaluation routine that
   may be used with SNESSetFunction() as long as the user context has a DA
   as its first record and the user has called DASetLocalFunction().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  F - function vector
-  ptr - pointer to a structure that must have a DA as its first entry. For example this 
         could be a DMMG

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalJacobian(), DASetLocalAdicFunction(), DASetLocalAdicMFFunction(),
          SNESSetFunction(), SNESSetJacobian()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDAFormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  PetscErrorCode ierr;
  Vec            localX;
  DA             da = *(DA*)ptr;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAFormFunction1(da,localX,F,ptr);
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = DARestoreLocalVector(da,&localX);CHKERRQ(pierr);
  }
  CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

/* ---------------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianWithFD"
PetscErrorCode DMMGComputeJacobianWithFD(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  
  PetscFunctionBegin;
  ierr = SNESDefaultComputeJacobianColor(snes,x1,J,B,flag,dmmg->fdcoloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianWithMF"
PetscErrorCode DMMGComputeJacobianWithMF(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ADIC)
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
#endif

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian"
/*
    DMMGComputeJacobian - Evaluates the Jacobian when the user has provided
    a local function evaluation routine.
*/
PetscErrorCode DMMGComputeJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG           dmmg = (DMMG) ptr;
  PetscErrorCode ierr;
  Vec            localX;
  DA             da = (DA) dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAComputeJacobian1(da,localX,*B,dmmg->user);CHKERRQ(ierr);
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

#if defined(PETSC_HAVE_ADIC)
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
#endif

#undef __FUNCT__
#define __FUNCT__ "SNESDAComputeJacobianWithAdifor"
/*
    SNESDAComputeJacobianWithAdifor - This is a universal Jacobian evaluation routine
    that may be used with SNESSetJacobian() from Fortran as long as the user context has 
    a DA as its first record and DASetLocalAdiforFunction() has been called.  

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

*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDAComputeJacobianWithAdifor(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DA             da = *(DA*) ptr;
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAComputeJacobian1WithAdifor(da,localX,*B,ptr);CHKERRQ(ierr);
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
#define __FUNCT__ "SNESDAComputeJacobian"
/*
   SNESDAComputeJacobian - This is a universal Jacobian evaluation routine for a
   locally provided Jacobian.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  X - input vector
.  J - Jacobian
.  B - Jacobian used in preconditioner (usally same as J)
.  flag - indicates if the matrix changed its structure
-  ptr - optional user-defined context, as set by SNESSetFunction()

   Level: intermediate

.seealso: DASetLocalFunction(), DASetLocalJacobian(), SNESSetFunction(), SNESSetJacobian()

*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDAComputeJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DA             da = *(DA*) ptr;
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAComputeJacobian1(da,localX,*B,ptr);CHKERRQ(ierr);
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
#define __FUNCT__ "DMMGSolveSNES"
PetscErrorCode DMMGSolveSNES(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  dmmg[0]->nlevels = level+1;
  ierr             = SNESSolve(dmmg[level]->snes,PETSC_NULL,dmmg[level]->x);CHKERRQ(ierr);
  dmmg[0]->nlevels = nlevels;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFCreate_DAAD(NLF*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFRelax_DAAD(NLF,MatSORType,PetscInt,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFDAADSetDA_DAAD(NLF,DA);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFDAADSetCtx_DAAD(NLF,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFDAADSetResidual_DAAD(NLF,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT NLFDAADSetNewtonIterations_DAAD(NLF,PetscInt);
EXTERN_C_END

#if defined(PETSC_HAVE_ADIC)
#include "src/ksp/pc/impls/mg/mgimpl.h"                    /*I "petscmg.h" I*/
/*
          This is pre-beta FAS code. It's design should not be taken seriously!
*/
#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveFAS"
PetscErrorCode DMMGSolveFAS(DMMG *dmmg,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscReal      norm;
  PetscScalar    zero = 0.0,mone = -1.0,one = 1.0;
  PC_MG          *mg;
  PC             pc;

  PetscFunctionBegin;
  ierr = VecSet(dmmg[level]->r,zero);CHKERRQ(ierr);
  for (j=1; j<=level; j++) {
    if (!dmmg[j]->inject) {
      ierr = DMGetInjection(dmmg[j-1]->dm,dmmg[j]->dm,&dmmg[j]->inject);CHKERRQ(ierr);
    }
  }

  ierr = KSPGetPC(dmmg[level]->ksp,&pc);CHKERRQ(ierr);
  mg   = ((PC_MG*)pc->data);

  for (i=0; i<100; i++) {

    for (j=level; j>0; j--) {

      /* Relax residual_fine - F(x_fine) = 0 */
      for (k=0; k<dmmg[j]->presmooth; k++) {
	ierr = NLFRelax_DAAD(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
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
      }

      if (dmmg[j]->monitorall) {
        for (k=0; k<level-j+1; k++) {ierr = PetscPrintf(dmmg[j]->comm,"  ");CHKERRQ(ierr);}
        ierr = PetscPrintf(dmmg[j]->comm,"FAS function norm %g\n",norm);CHKERRQ(ierr);
      }
      ierr = MatRestrict(mg[j].restrct,dmmg[j]->w,dmmg[j-1]->r);CHKERRQ(ierr); 
      
      /* F(R*x_fine) */
      ierr = VecScatterBegin(dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD,dmmg[j]->inject);CHKERRQ(ierr);
      ierr = VecScatterEnd(dmmg[j]->x,dmmg[j-1]->x,INSERT_VALUES,SCATTER_FORWARD,dmmg[j]->inject);CHKERRQ(ierr);
      ierr = DMMGFormFunction(0,dmmg[j-1]->x,dmmg[j-1]->w,dmmg[j-1]);CHKERRQ(ierr);

      /* residual_coarse = F(R*x_fine) + R*(residual_fine - F(x_fine)) */
      ierr = VecAYPX(dmmg[j-1]->r,one,dmmg[j-1]->w);CHKERRQ(ierr);

      /* save R*x_fine into b (needed when interpolating compute x back up */
      ierr = VecCopy(dmmg[j-1]->x,dmmg[j-1]->b);CHKERRQ(ierr);
    }

    for (j=0; j<dmmg[0]->presmooth; j++) {
      ierr = NLFRelax_DAAD(dmmg[0]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[0]->x);CHKERRQ(ierr);
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
      ierr = MatInterpolateAdd(mg[j].interpolate,dmmg[j-1]->x,dmmg[j]->x,dmmg[j]->x);CHKERRQ(ierr);

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
	ierr = NLFRelax_DAAD(dmmg[j]->nlf,SOR_SYMMETRIC_SWEEP,1,dmmg[j]->x);CHKERRQ(ierr);
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
#endif

/* ===========================================================================================================*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNES"
/*@C
    DMMGSetSNES - Sets the nonlinear function that defines the nonlinear set of equations
    to be solved using the grid hierarchy.

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
-   jacobian - optional function to compute Jacobian

    Options Database Keys:
+    -dmmg_snes_monitor
.    -dmmg_jacobian_fd
.    -dmmg_jacobian_ad
.    -dmmg_jacobian_mf_fd_operator
.    -dmmg_jacobian_mf_fd
.    -dmmg_jacobian_mf_ad_operator
.    -dmmg_jacobian_mf_ad
-    -dmmg_jacobian_period <p> - Indicates how often in the SNES solve the Jacobian is recomputed (on all levels)
                                 as suggested by Florin Dobrian if p is -1 then Jacobian is computed only on first
                                 SNES iteration (i.e. -1 is equivalent to infinity) 

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNESLocal()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT DMMGSetSNES(DMMG *dmmg,PetscErrorCode (*function)(SNES,Vec,Vec,void*),PetscErrorCode (*jacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*))
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       i,nlevels = dmmg[0]->nlevels,period = 1;
  PetscTruth     snesmonitor,mffdoperator,mffd,fdjacobian;
#if defined(PETSC_HAVE_ADIC)
  PetscTruth     mfadoperator,mfad,adjacobian;
#endif
  PetscViewer    ascii;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!dmmg)     SETERRQ(PETSC_ERR_ARG_NULL,"Passing null as DMMG");
  if (!jacobian) jacobian = DMMGComputeJacobianWithFD;

  ierr = PetscOptionsBegin(dmmg[0]->comm,PETSC_NULL,"DMMG Options","SNES");CHKERRQ(ierr);
    ierr = PetscOptionsName("-dmmg_snes_monitor","Monitor nonlinear convergence","SNESSetMonitor",&snesmonitor);CHKERRQ(ierr);


    ierr = PetscOptionsName("-dmmg_jacobian_fd","Compute sparse Jacobian explicitly with finite differencing","DMMGSetSNES",&fdjacobian);CHKERRQ(ierr);
    if (fdjacobian) jacobian = DMMGComputeJacobianWithFD;
#if defined(PETSC_HAVE_ADIC)
    ierr = PetscOptionsName("-dmmg_jacobian_ad","Compute sparse Jacobian explicitly with ADIC (automatic differentiation)","DMMGSetSNES",&adjacobian);CHKERRQ(ierr);
    if (adjacobian) jacobian = DMMGComputeJacobianWithAdic;
#endif

    ierr = PetscOptionsTruthGroupBegin("-dmmg_jacobian_mf_fd_operator","Apply Jacobian via matrix free finite differencing","DMMGSetSNES",&mffdoperator);CHKERRQ(ierr);
    ierr = PetscOptionsTruthGroupEnd("-dmmg_jacobian_mf_fd","Apply Jacobian via matrix free finite differencing even in computing preconditioner","DMMGSetSNES",&mffd);CHKERRQ(ierr);
    if (mffd) mffdoperator = PETSC_TRUE;
#if defined(PETSC_HAVE_ADIC)
    ierr = PetscOptionsTruthGroupBegin("-dmmg_jacobian_mf_ad_operator","Apply Jacobian via matrix free ADIC (automatic differentiation)","DMMGSetSNES",&mfadoperator);CHKERRQ(ierr);
    ierr = PetscOptionsTruthGroupEnd("-dmmg_jacobian_mf_ad","Apply Jacobian via matrix free ADIC (automatic differentiation) even in computing preconditioner","DMMGSetSNES",&mfad);CHKERRQ(ierr);
    if (mfad) mfadoperator = PETSC_TRUE;
#endif
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* create solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = SNESCreate(dmmg[i]->comm,&dmmg[i]->snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(dmmg[i]->snes,&dmmg[i]->ksp);CHKERRQ(ierr);
    if (snesmonitor) {
      ierr = PetscObjectGetComm((PetscObject)dmmg[i]->snes,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
      ierr = PetscViewerASCIISetTab(ascii,nlevels-i);CHKERRQ(ierr);
      ierr = SNESSetMonitor(dmmg[i]->snes,SNESDefaultMonitor,ascii,(PetscErrorCode(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }

    if (mffdoperator) {
      ierr = MatCreateSNESMF(dmmg[i]->snes,dmmg[i]->x,&dmmg[i]->J);CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work1);CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work2);CHKERRQ(ierr);
      ierr = MatSNESMFSetFunction(dmmg[i]->J,dmmg[i]->work1,function,dmmg[i]);CHKERRQ(ierr);
      if (mffd) {
        dmmg[i]->B = dmmg[i]->J;
        jacobian   = DMMGComputeJacobianWithMF;
      }
#if defined(PETSC_HAVE_ADIC)
    } else if (mfadoperator) {
      ierr = MatRegisterDAAD();CHKERRQ(ierr);
      ierr = MatCreateDAAD((DA)dmmg[i]->dm,&dmmg[i]->J);CHKERRQ(ierr);
      ierr = MatDAADSetCtx(dmmg[i]->J,dmmg[i]->user);CHKERRQ(ierr);
      if (mfad) {
        dmmg[i]->B = dmmg[i]->J;
        jacobian   = DMMGComputeJacobianWithMF;
      }
#endif
    }
    
    if (!dmmg[i]->B) {
      ierr = MPI_Comm_size(dmmg[i]->comm,&size);CHKERRQ(ierr);
      ierr = DMGetMatrix(dmmg[i]->dm,MATAIJ,&dmmg[i]->B);CHKERRQ(ierr);
    } 
    if (!dmmg[i]->J) {
      dmmg[i]->J = dmmg[i]->B;
    }

    ierr = DMMGSetUpLevel(dmmg,dmmg[i]->ksp,i+1);CHKERRQ(ierr);
    
    /*
       if the number of levels is > 1 then we want the coarse solve in the grid sequencing to use LU
       when possible 
    */
    if (nlevels > 1 && i == 0) {
      PC         pc;
      KSP        cksp;
      PetscTruth flg1,flg2,flg3;

      ierr = KSPGetPC(dmmg[i]->ksp,&pc);CHKERRQ(ierr);
      ierr = PCMGGetCoarseSolve(pc,&cksp);CHKERRQ(ierr);
      ierr = KSPGetPC(cksp,&pc);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCILU,&flg1);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCSOR,&flg2);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PETSC_NULL,&flg3);CHKERRQ(ierr);
      if (flg1 || flg2 || flg3) {
        ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      }
    }

    dmmg[i]->solve           = DMMGSolveSNES;
    dmmg[i]->computejacobian = jacobian;
    dmmg[i]->computefunction = function;
  }

  if (jacobian == DMMGComputeJacobianWithFD) {
    ISColoring iscoloring;
    for (i=0; i<nlevels; i++) {
      ierr = DMGetColoring(dmmg[i]->dm,IS_COLORING_LOCAL,&iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(dmmg[i]->B,iscoloring,&dmmg[i]->fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(dmmg[i]->fdcoloring,(PetscErrorCode(*)(void))function,dmmg[i]);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(dmmg[i]->fdcoloring);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_ADIC)
  } else if (jacobian == DMMGComputeJacobianWithAdic) {
    for (i=0; i<nlevels; i++) {
      ISColoring iscoloring;
      ierr = DMGetColoring(dmmg[i]->dm,IS_COLORING_GHOSTED,&iscoloring);CHKERRQ(ierr);
      ierr = MatSetColoring(dmmg[i]->B,iscoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
    }
#endif
  }

  for (i=0; i<nlevels; i++) {
    ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_Multigrid,dmmg);CHKERRQ(ierr);
    ierr = SNESSetFunction(dmmg[i]->snes,dmmg[i]->b,function,dmmg[i]);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(dmmg[i]->snes);CHKERRQ(ierr);
  }

  /* Create interpolation scaling */
  for (i=1; i<nlevels; i++) {
    ierr = DMGetInterpolationScale(dmmg[i-1]->dm,dmmg[i]->dm,dmmg[i]->R,&dmmg[i]->Rscale);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetInt(PETSC_NULL,"-dmmg_jacobian_period",&period,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    dmmg[i]->updatejacobian       = PETSC_TRUE;
    dmmg[i]->updatejacobianperiod = period;
  }

#if defined(PETSC_HAVE_ADIC)
  { 
    PetscTruth flg;
    ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_fas",&flg);CHKERRQ(ierr);
    if (flg) {
      PetscInt newton_its;
      ierr = PetscOptionsHasName(0,"-dmmg_fas_view",&flg);CHKERRQ(ierr);
      for (i=0; i<nlevels; i++) {
	ierr = NLFCreate_DAAD(&dmmg[i]->nlf);CHKERRQ(ierr);
	ierr = NLFDAADSetDA_DAAD(dmmg[i]->nlf,(DA)dmmg[i]->dm);CHKERRQ(ierr);
	ierr = NLFDAADSetCtx_DAAD(dmmg[i]->nlf,dmmg[i]->user);CHKERRQ(ierr);
	ierr = NLFDAADSetResidual_DAAD(dmmg[i]->nlf,dmmg[i]->r);CHKERRQ(ierr);
        ierr = VecDuplicate(dmmg[i]->b,&dmmg[i]->w);CHKERRQ(ierr);

        dmmg[i]->monitor    = PETSC_FALSE;
        ierr = PetscOptionsHasName(0,"-dmmg_fas_monitor",&dmmg[i]->monitor);CHKERRQ(ierr);
        dmmg[i]->monitorall = PETSC_FALSE;
        ierr = PetscOptionsHasName(0,"-dmmg_fas_monitor_all",&dmmg[i]->monitorall);CHKERRQ(ierr);
        dmmg[i]->presmooth  = 2;
        ierr = PetscOptionsGetInt(0,"-dmmg_fas_presmooth",&dmmg[i]->presmooth,0);CHKERRQ(ierr);
        dmmg[i]->postsmooth = 2;
        ierr = PetscOptionsGetInt(0,"-dmmg_fas_postsmooth",&dmmg[i]->postsmooth,0);CHKERRQ(ierr);
        dmmg[i]->coarsesmooth = 2;
        ierr = PetscOptionsGetInt(0,"-dmmg_fas_coarsesmooth",&dmmg[i]->coarsesmooth,0);CHKERRQ(ierr);

        dmmg[i]->rtol = 1.e-8;
        ierr = PetscOptionsGetReal(0,"-dmmg_fas_rtol",&dmmg[i]->rtol,0);CHKERRQ(ierr);
        dmmg[i]->abstol = 1.e-50;
        ierr = PetscOptionsGetReal(0,"-dmmg_fas_atol",&dmmg[i]->abstol,0);CHKERRQ(ierr);

        newton_its = 2;
        ierr = PetscOptionsGetInt(0,"-dmmg_fas_newton_its",&newton_its,0);CHKERRQ(ierr);
        ierr = NLFDAADSetNewtonIterations_DAAD(dmmg[i]->nlf,newton_its);CHKERRQ(ierr);

        if (flg) {
          if (i == 0) {
            ierr = PetscPrintf(dmmg[i]->comm,"FAS Solver Parameters\n");CHKERRQ(ierr);
            ierr = PetscPrintf(dmmg[i]->comm,"  rtol %g atol %g\n",dmmg[i]->rtol,dmmg[i]->abstol);CHKERRQ(ierr);
	    ierr = PetscPrintf(dmmg[i]->comm,"             coarsesmooths %D\n",dmmg[i]->coarsesmooth);CHKERRQ(ierr);
            ierr = PetscPrintf(dmmg[i]->comm,"             Newton iterations %D\n",newton_its);CHKERRQ(ierr);
          } else {
	    ierr = PetscPrintf(dmmg[i]->comm,"  level %D   presmooths    %D\n",i,dmmg[i]->presmooth);CHKERRQ(ierr);
	    ierr = PetscPrintf(dmmg[i]->comm,"             postsmooths   %D\n",dmmg[i]->postsmooth);CHKERRQ(ierr);
            ierr = PetscPrintf(dmmg[i]->comm,"             Newton iterations %D\n",newton_its);CHKERRQ(ierr);
          }
        }
	dmmg[i]->solve = DMMGSolveFAS;
      }
    }
  }
#endif
   
  PetscFunctionReturn(0);
}

/*M
    DMMGSetSNESLocal - Sets the local user function that defines the nonlinear set of equations
    that will use the grid hierarchy and (optionally) its derivative.

    Collective on DMMG

   Synopsis:
   PetscErrorCode DMMGSetSNESLocal(DMMG *dmmg,DALocalFunction1 function, DALocalFunction1 jacobian,
                        DALocalFunction1 ad_function, DALocalFunction1 admf_function);

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
.   jacobian - function defines the local part of the Jacobian
.   ad_function - the name of the function with an ad_ prefix. This is ignored if ADIC is
                  not installed
-   admf_function - the name of the function with an ad_ prefix. This is ignored if ADIC is
                  not installed

    Options Database Keys:
+    -dmmg_snes_monitor
.    -dmmg_jacobian_fd
.    -dmmg_jacobian_ad
.    -dmmg_jacobian_mf_fd_operator
.    -dmmg_jacobian_mf_fd
.    -dmmg_jacobian_mf_ad_operator
.    -dmmg_jacobian_mf_ad
-    -dmmg_jacobian_period <p> - Indicates how often in the SNES solve the Jacobian is recomputed (on all levels)
                                 as suggested by Florin Dobrian if p is -1 then Jacobian is computed only on first
                                 SNES iteration (i.e. -1 is equivalent to infinity) 


    Level: intermediate

    Notes: 
    If ADIC or ADIFOR have been installed, this routine can use ADIC or ADIFOR to compute
    the derivative; however, that function cannot call other functions except those in
    standard C math libraries.

    If ADIC/ADIFOR have not been installed and the Jacobian is not provided, this routine
    uses finite differencing to approximate the Jacobian.

.seealso DMMGCreate(), DMMGDestroy, DMMGSetKSP(), DMMGSetSNES()

M*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocal_Private"
PetscErrorCode DMMGSetSNESLocal_Private(DMMG *dmmg,DALocalFunction1 function,DALocalFunction1 jacobian,DALocalFunction1 ad_function,DALocalFunction1 admf_function)
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;
  PetscErrorCode (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = 0;


  PetscFunctionBegin;
  if (jacobian)         computejacobian = DMMGComputeJacobian;
#if defined(PETSC_HAVE_ADIC)
  else if (ad_function) computejacobian = DMMGComputeJacobianWithAdic;
#endif

  ierr = DMMGSetSNES(dmmg,DMMGFormFunction,computejacobian);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    ierr = DASetLocalFunction((DA)dmmg[i]->dm,function);CHKERRQ(ierr);
    ierr = DASetLocalJacobian((DA)dmmg[i]->dm,jacobian);CHKERRQ(ierr);
    ierr = DASetLocalAdicFunction((DA)dmmg[i]->dm,ad_function);CHKERRQ(ierr);
    ierr = DASetLocalAdicMFFunction((DA)dmmg[i]->dm,admf_function);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGFunctioni"
static PetscErrorCode DMMGFunctioni(PetscInt i,Vec u,PetscScalar* r,void* ctx)
{
  DMMG           dmmg = (DMMG)ctx;
  Vec            U = dmmg->lwork1;
  PetscErrorCode ierr;
  VecScatter     gtol;

  PetscFunctionBegin;
  /* copy u into interior part of U */
  ierr = DAGetScatter((DA)dmmg->dm,0,&gtol,0);CHKERRQ(ierr);
  ierr = VecScatterBegin(u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(u,U,INSERT_VALUES,SCATTER_FORWARD_LOCAL,gtol);CHKERRQ(ierr);
  ierr = DAFormFunctioni1((DA)dmmg->dm,i,U,r,dmmg->user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGFunctioniBase"
static PetscErrorCode DMMGFunctioniBase(Vec u,void* ctx)
{
  DMMG           dmmg = (DMMG)ctx;
  Vec            U = dmmg->lwork1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,u,INSERT_VALUES,U);CHKERRQ(ierr);  
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,u,INSERT_VALUES,U);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocali_Private"
PetscErrorCode DMMGSetSNESLocali_Private(DMMG *dmmg,PetscErrorCode (*functioni)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*),PetscErrorCode (*adi)(DALocalInfo*,MatStencil*,void*,void*,void*),PetscErrorCode (*adimf)(DALocalInfo*,MatStencil*,void*,void*,void*))
{
  PetscErrorCode ierr;
  PetscInt       i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    ierr = DASetLocalFunctioni((DA)dmmg[i]->dm,functioni);CHKERRQ(ierr);
    ierr = DASetLocalAdicFunctioni((DA)dmmg[i]->dm,adi);CHKERRQ(ierr);
    ierr = DASetLocalAdicMFFunctioni((DA)dmmg[i]->dm,adimf);CHKERRQ(ierr);
    ierr = MatSNESMFSetFunctioni(dmmg[i]->J,DMMGFunctioni);CHKERRQ(ierr);
    ierr = MatSNESMFSetFunctioniBase(dmmg[i]->J,DMMGFunctioniBase);CHKERRQ(ierr);    
    ierr = DACreateLocalVector((DA)dmmg[i]->dm,&dmmg[i]->lwork1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#if defined(PETSC_HAVE_ADIC)
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

#endif



