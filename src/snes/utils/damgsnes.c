/*$Id: damgsnes.c,v 1.11 2001/03/15 19:50:38 bsmith Exp balay $*/
 
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
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->J,*flag);CHKERRQ(ierr);

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
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->J,*flag);CHKERRQ(ierr);

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
    ierr = SLESSetOperators(lsles,DMMGGetFine(dmmg)->J,DMMGGetFine(dmmg)->J,*flag);CHKERRQ(ierr);

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
    DMMGSetSNES - Sets the nonlinear solver object that will use the grid hierarchy

    Collective on DMMG and SNES

    Input Parameter:
+   dmmg - the context
-   snes - the nonlinear solver object

    Level: advanced

.seealso DMMGCreate(), DMMGDestroy, DMMGSetSLES()

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
    ierr = SNESSetFromOptions(dmmg[i]->snes);CHKERRQ(ierr);
    dmmg[i]->solve = DMMGSolveSNES;
    dmmg[i]->computejacobian = jacobian;
    dmmg[i]->computefunction = function;
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-dmmg_fd",&usefd);CHKERRQ(ierr);
  if ((!jacobian && !dmmg[0]->matrixfree) || usefd) {
    ISColoring iscoloring;
    for (i=0; i<nlevels; i++) {
      ierr = DMGetColoring(dmmg[i]->dm,MATMPIAIJ,&iscoloring,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(dmmg[i]->J,iscoloring,&dmmg[i]->fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(dmmg[i]->fdcoloring,(int(*)(void))function,dmmg[i]);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(dmmg[i]->fdcoloring);CHKERRQ(ierr);
      dmmg[i]->computejacobian = SNESDefaultComputeJacobianColor;
    }
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

  for (i=0; i<nlevels-1; i++) {
    ierr = SNESSetOptionsPrefix(dmmg[i]->snes,"dmmg_levels_");CHKERRQ(ierr);
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











