/*$Id: damgsnes.c,v 1.39 2001/05/22 03:01:46 bsmith Exp bsmith $*/
 
#include "petscda.h"      /*I      "petscda.h"     I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/

/*
      These evaluate the Jacobian on all of the grids. It is used by DMMG to "replace"
   the user provided Jacobian function. In fact, it calls the user provided one at each level.
*/
/*
          Version for matrix-free Jacobian 
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian_MF"
int DMMGComputeJacobian_MF(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG       *dmmg = (DMMG*)ptr;
  int        ierr,i,nlevels = dmmg[0]->nlevels;
  SLES       sles,lsles;
  PC         pc;
  PetscTruth ismg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as user context which should contain DMMG");

  /* The finest level matrix is "shared" by the corresponding SNES object so we need
     only call MatAssemblyXXX() on it to indicate it is being used in a new solve */
  ierr = MatAssemblyBegin(dmmg[nlevels-1]->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(dmmg[nlevels-1]->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     The other levels MUST be told the vector from which we are doing the differencing
  */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = MGGetSmoother(pc,nlevels-1,&lsles);CHKERRQ(ierr);
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->B,*flag);CHKERRQ(ierr);

    for (i=nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(dmmg[i]->R,X,dmmg[i-1]->work2);CHKERRQ(ierr);
      X    = dmmg[i-1]->work2;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(dmmg[i]->Rscale,X,X);CHKERRQ(ierr);

      ierr = MatSNESMFSetBase(dmmg[i-1]->J,X);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(dmmg[i-1]->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(dmmg[i-1]->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      ierr = MGGetSmoother(pc,i-1,&lsles);CHKERRQ(ierr);
      ierr = SLESSetOperators(lsles,dmmg[i-1]->J,dmmg[i-1]->B,*flag);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
    Version for user provided Jacobian
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian_User"
int DMMGComputeJacobian_User(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG       *dmmg = (DMMG*)ptr;
  int        ierr,i,nlevels = dmmg[0]->nlevels;
  SLES       sles,lsles;
  PC         pc;
  PetscTruth ismg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as user context which should contain DMMG");

  ierr = (*DMMGGetFine(dmmg)->computejacobian)(snes,X,J,B,flag,DMMGGetFine(dmmg));CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = MGGetSmoother(pc,nlevels-1,&lsles);CHKERRQ(ierr);
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->B,*flag);CHKERRQ(ierr);

    for (i=nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(dmmg[i]->R,X,dmmg[i-1]->x);CHKERRQ(ierr);
      X    = dmmg[i-1]->x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(dmmg[i]->Rscale,X,X);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      ierr = (*dmmg[i-1]->computejacobian)(snes,X,&dmmg[i-1]->J,&dmmg[i-1]->B,flag,dmmg[i-1]);CHKERRQ(ierr);

      ierr = MGGetSmoother(pc,i-1,&lsles);CHKERRQ(ierr);
      ierr = SLESSetOperators(lsles,dmmg[i-1]->J,dmmg[i-1]->B,*flag);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
/*
    Version for Jacobian computed via PETSc finite differencing. This is the same 
  as DMMGComputeJacobian_User() except passes in the fdcoloring as the private context
*/
#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobian_FD"
int DMMGComputeJacobian_FD(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG       *dmmg = (DMMG*)ptr;
  int        ierr,i,nlevels = dmmg[0]->nlevels;
  SLES       sles,lsles;
  PC         pc;
  PetscTruth ismg;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as user context which should contain DMMG");

  ierr = (*DMMGGetFine(dmmg)->computejacobian)(snes,X,J,B,flag,DMMGGetFine(dmmg)->fdcoloring);CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = MGGetSmoother(pc,nlevels-1,&lsles);CHKERRQ(ierr);
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->B,*flag);CHKERRQ(ierr);

    for (i=nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(dmmg[i]->R,X,dmmg[i-1]->x);CHKERRQ(ierr);
      X    = dmmg[i-1]->x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(dmmg[i]->Rscale,X,X);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      ierr = (*dmmg[i-1]->computejacobian)(snes,X,&dmmg[i-1]->J,&dmmg[i-1]->B,flag,dmmg[i-1]->fdcoloring);CHKERRQ(ierr);

      ierr = MGGetSmoother(pc,i-1,&lsles);CHKERRQ(ierr);
      ierr = SLESSetOperators(lsles,dmmg[i-1]->J,dmmg[i-1]->B,*flag);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

extern int DMMGComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__  
#define __FUNCT__ "DMMGSolveSNES"
int DMMGSolveSNES(DMMG *dmmg,int level)
{
  int  ierr,nlevels = dmmg[0]->nlevels,its;

  PetscFunctionBegin;
  dmmg[0]->nlevels = level+1;
  ierr = SNESSolve(dmmg[level]->snes,dmmg[level]->x,&its);CHKERRQ(ierr);
  dmmg[0]->nlevels = nlevels;
  PetscFunctionReturn(0);
}

EXTERN int DMMGSetUpLevel(DMMG*,SLES,int);

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNES"
/*@C
    DMMGSetSNES - Sets the nonlinear function that defines the nonlinear set of equations
      to be solved will use the grid hierarchy

    Collective on DMMG

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
-   jacobian - optional function to compute Jacobian

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetSLES(), DMMGSetSNESLocal()

@*/
int DMMGSetSNES(DMMG *dmmg,int (*function)(SNES,Vec,Vec,void*),int (*jacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*))
{
  int         ierr,i,nlevels = dmmg[0]->nlevels;
  PetscTruth  usefd,snesmonitor;
  SLES        sles;
  PetscViewer ascii;
  MPI_Comm    comm;

  PetscFunctionBegin;
  if (!dmmg) SETERRQ(1,"Passing null as DMMG");

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_snes_monitor",&snesmonitor);CHKERRQ(ierr);
  /* create solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = SNESCreate(dmmg[i]->comm,SNES_NONLINEAR_EQUATIONS,&dmmg[i]->snes);CHKERRQ(ierr);
    if (snesmonitor) {
      ierr = PetscObjectGetComm((PetscObject)dmmg[i]->snes,&comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(comm,"stdout",&ascii);CHKERRQ(ierr);
      ierr = PetscViewerASCIISetTab(ascii,nlevels-i);CHKERRQ(ierr);
      ierr = SNESSetMonitor(dmmg[i]->snes,SNESDefaultMonitor,ascii,(int(*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
    }
    if (dmmg[0]->matrixfree) {
      ierr = MatCreateSNESMF(dmmg[i]->snes,dmmg[i]->x,&dmmg[i]->J);CHKERRQ(ierr);
      if (!dmmg[i]->B) dmmg[i]->B = dmmg[i]->J;
      if (i != nlevels-1) {
        ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work1);CHKERRQ(ierr);
        ierr = VecDuplicate(dmmg[i]->x,&dmmg[i]->work2);CHKERRQ(ierr);
        ierr = MatSNESMFSetFunction(dmmg[i]->J,dmmg[i]->work1,function,dmmg[i]);CHKERRQ(ierr);
      }
    }

    ierr = SNESGetSLES(dmmg[i]->snes,&sles);CHKERRQ(ierr);
    ierr = DMMGSetUpLevel(dmmg,sles,i+1);CHKERRQ(ierr);
    
    /*
       if the number of levels is > 1 then we want the coarse solve in the grid sequencing to use LU
       when possible 
    */
    if (nlevels > 1 && i == 0) {
      PC         pc;
      SLES       csles;
      PetscTruth flg1,flg2,flg3;

      ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
      ierr = MGGetCoarseSolve(pc,&csles);CHKERRQ(ierr);
      ierr = SLESGetPC(csles,&pc);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCILU,&flg1);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCSOR,&flg2);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PETSC_NULL,&flg3);CHKERRQ(ierr);
      if (flg1 || flg2 || flg3) {
        ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      }
    }

    ierr = SNESSetFromOptions(dmmg[i]->snes);CHKERRQ(ierr);
    dmmg[i]->solve = DMMGSolveSNES;
    dmmg[i]->computejacobian = jacobian;
    dmmg[i]->computefunction = function;
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_fd",&usefd);CHKERRQ(ierr);
  if ((!jacobian && !dmmg[0]->matrixfree) || usefd) {
    ISColoring iscoloring;
    for (i=0; i<nlevels; i++) {
      ierr = DMGetColoring(dmmg[i]->dm,IS_COLORING_LOCAL,MATMPIAIJ,&iscoloring,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(dmmg[i]->J,iscoloring,&dmmg[i]->fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(dmmg[i]->fdcoloring,(int(*)(void))function,dmmg[i]);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(dmmg[i]->fdcoloring);CHKERRQ(ierr);
      dmmg[i]->computejacobian = SNESDefaultComputeJacobianColor;
    }
#if defined(PETSC_HAVE_ADIC)
  } else if (jacobian == DMMGComputeJacobianWithAdic) {
    for (i=0; i<nlevels; i++) {
      ISColoring iscoloring;
      ierr = DMGetColoring(dmmg[i]->dm,IS_COLORING_GHOSTED,MATMPIAIJ,&iscoloring,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatSetColoring(dmmg[i]->B,iscoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
    }
#endif
  }

  for (i=0; i<nlevels; i++) {
    if (dmmg[i]->matrixfree) {
      ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_MF,dmmg);CHKERRQ(ierr);
    } else if (dmmg[i]->computejacobian == SNESDefaultComputeJacobianColor) {
      ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_FD,dmmg);CHKERRQ(ierr);
    } else {
      ierr = SNESSetJacobian(dmmg[i]->snes,dmmg[i]->J,dmmg[i]->B,DMMGComputeJacobian_User,dmmg);CHKERRQ(ierr);
    }
    ierr = SNESSetFunction(dmmg[i]->snes,dmmg[i]->b,function,dmmg[i]);CHKERRQ(ierr);
  }

  /* Create interpolation scaling */
  for (i=1; i<nlevels; i++) {
    ierr = DMGetInterpolationScale(dmmg[i-1]->dm,dmmg[i]->dm,dmmg[i]->R,&dmmg[i]->Rscale);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetInitialGuess"
/*@C
    DMMGSetInitialGuess - Sets the function that computes an initial guess, if not given
         uses 0.

    Collective on DMMG and SNES

    Input Parameter:
+   dmmg - the context
-   guess - the function

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetSLES()

@*/
int DMMGSetInitialGuess(DMMG *dmmg,int (*guess)(SNES,Vec,void*))
{
  int i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
  for (i=0; i<nlevels; i++) {
    dmmg[i]->initialguess = guess;
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
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector

 */
int DMMGFormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG             dmmg = (DMMG)ptr;
  int              ierr;
  Vec              localX;
  DA               da = (DA)dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAFormFunction1(da,localX,F,dmmg->user);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "SNESDAFormFunction"
/*@C 
   SNESDAFormFunction - This is a universal form function that may be used with 
     SNESSetFunction() so long as the user context has a DA as the first record
     and has called DASetLocalFunction()

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
int SNESDAFormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  int              ierr;
  Vec              localX;
  DA               da = *(DA*)ptr;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAFormFunction1(da,localX,F,ptr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

/* ---------------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "DMMGComputeJacobianWithAdic"
/*
    DMMGComputeJacobianWithAdic - Evaluates the Jacobian via Adic when the user has provide
        a local form function
*/
int DMMGComputeJacobianWithAdic(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DMMG             dmmg = (DMMG) ptr;
  int              ierr;
  Vec              localX;
  DA               da = (DA) dmmg->dm;

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
    SNESDAComputeJacobianWithAdic - This is a universal form function that may be used with 
     SNESSetJacobian() so long as the user context has a DA as the first record
     and has called DASetLocalAdicFunction()

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
int SNESDAComputeJacobianWithAdic(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DA   da = *(DA*) ptr;
  int  ierr;
  Vec  localX;

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

#undef __FUNCT__
#define __FUNCT__ "SNESDAComputeJacobianWithAdifor"
/*
    SNESDAComputeJacobianWithAdifor - This is a universal form function that may be used with 
     SNESSetJacobian() from Fortran

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
int SNESDAComputeJacobianWithAdifor(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DA   da = *(DA*) ptr;
  int  ierr;
  Vec  localX;

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
    SNESDAComputeJacobian - This is a universal form function for a locally provided Jacobian

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
int SNESDAComputeJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DA   da = *(DA*) ptr;
  int  ierr;
  Vec  localX;

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

/*M
    DMMGSetSNESLocal - Sets the local user function that defines the nonlinear set of equations
          that will use the grid hierarchy and (optionally) its derivative

    Collective on DMMG

   Synopsis:
   int DMMGSetSNESLocal(DMMG *dmmg,DALocalFunction1 function, DALocalFunction1 jacobian,
                        DALocalFunction1 ad_function, DALocalFunction1 admf_function);

    Input Parameter:
+   dmmg - the context
.   function - the function that defines the nonlinear system
.   jacobian - function defines the local part of the Jacobian (not currently supported)
.   ad_function - the name of the function with an ad_ prefix. This is ignored if ADIC is
                  not installed
-   admf_function - the name of the function with an ad_ prefix. This is ignored if ADIC is
                  not installed

    Level: intermediate

    Notes: If ADIC or ADIFOR are installed this can use ADIC/ADIFOR to compute the derivative, however the 
       the function cannot call other
       functions except those in standard C math libraries.

       If ADIC/ADIFOR are not installed and jacobian is not provided, this used finite differencing to approximate
       the Jacobian

.seealso DMMGCreate(), DMMGDestroy, DMMGSetSLES(), DMMGSetSNES()

M*/

#undef __FUNCT__  
#define __FUNCT__ "DMMGSetSNESLocal_Private"
int DMMGSetSNESLocal_Private(DMMG *dmmg,DALocalFunction1 function,DALocalFunction1 jacobian,DALocalFunction1 ad_function,
                             DALocalFunction1 admf_function)
{
  int ierr,i,nlevels = dmmg[0]->nlevels;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_ADIC)
  ierr = DMMGSetSNES(dmmg,DMMGFormFunction,DMMGComputeJacobianWithAdic);CHKERRQ(ierr);
#else 
  ierr = DMMGSetSNES(dmmg,DMMGFormFunction,0);CHKERRQ(ierr);
#endif
  for (i=0; i<nlevels; i++) {
    ierr = DASetLocalFunction((DA)dmmg[i]->dm,function);CHKERRQ(ierr);
    ierr = DASetLocalJacobian((DA)dmmg[i]->dm,jacobian);CHKERRQ(ierr);
    ierr = DASetLocalAdicFunction((DA)dmmg[i]->dm,ad_function);CHKERRQ(ierr);
    ierr = DASetLocalAdicMFFunction((DA)dmmg[i]->dm,admf_function);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}






