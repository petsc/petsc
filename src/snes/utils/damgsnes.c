
/*$Id: damgsnes.c,v 1.2 2000/07/13 03:53:36 bsmith Exp bsmith $*/
 
#include "petscda.h"      /*I      "petscda.h"     I*/
#include "petscmg.h"      /*I      "petscmg.h"    I*/

/*
      This evaluates the Jacobian on all of the grids. It is used by DAMG to "replace"
   the user provided function. In fact, it calls the user provided one at each level.
*/
#undef __FUNC__
#define __FUNC__ "DAMGComputeJacobian"
int DAMGComputeJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  DAMG       *damg = (DAMG*)ptr;
  int        ierr,i,nlevels = damg[0]->nlevels,ntrue,Xsize;
  SLES       sles,lsles;
  PC         pc;
  PetscTruth ismg;
  void       *pptr;

  PetscFunctionBegin;
  if (!damg) SETERRQ(1,1,"Passing null as user context which should contain DAMG");

  /* Determine the finest level. This is a little tacky, uses length of vector passed in
     compared to saved values. */
  ierr = VecGetSize(X,&Xsize);CHKERRQ(ierr);
  for (i=0; i<nlevels; i++) {
    if (Xsize == damg[i]->Xsize) {
      ntrue = i;
      break;
    }
  }
  if (DAMGGetFine(damg)->computejacobian == SNESDefaultComputeJacobianColor) pptr = DAMGGetFine(damg)->fdcoloring;
  else pptr = DAMGGetFine(damg);

  ierr = (*DAMGGetFine(damg)->computejacobian)(snes,X,J,B,flag,pptr);CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = MGGetSmoother(pc,nlevels-1,&lsles);CHKERRQ(ierr);
    ierr = SLESSetOperators(lsles,DAMGGetFine(damg)->J,DAMGGetFine(damg)->J,*flag);CHKERRA(ierr);

    for (i=nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(damg[i]->R,X,damg[i-1]->x);CHKERRQ(ierr);
      X    = damg[i-1]->x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(damg[i]->Rscale,X,X);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      if (damg[i-1]->computejacobian == SNESDefaultComputeJacobianColor) pptr = damg[i-1]->fdcoloring;
      else pptr = damg[i-1];

      ierr = (*damg[i-1]->computejacobian)(snes,X,&damg[i-1]->J,&damg[i-1]->B,flag,pptr);CHKERRQ(ierr);

      ierr = MGGetSmoother(pc,i-1,&lsles);CHKERRQ(ierr);
      ierr = SLESSetOperators(lsles,damg[i-1]->J,damg[i-1]->B,*flag);CHKERRA(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSolveSNES"></a>*/"DAMGSolveSNES"
int DAMGSolveSNES(DAMG *damg,int level)
{
  int  ierr,i,nlevels = damg[0]->nlevels,its;
  SNES snes;

  PetscFunctionBegin;
  ierr = SNESSolve(damg[level]->snes,damg[level]->x,&its);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DAMGSetSNES"></a>*/"DAMGSetSNES"
/*@C
    DAMGSetSNES - Sets the nonlinear solver object that will use the grid hierarchy

    Collective on DAMG and SNES

    Input Parameter:
+   damg - the context
-   snes - the nonlinear solver object

    Level: advanced

.seealso DAMGCreate(), DAMGDestroy, DAMGSetCoarseDA(), DAMGSetSLES()

@*/
int DAMGSetSNES(DAMG *damg,int (*function)(SNES,Vec,Vec,void*),int (*jacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*))
{
  int        ierr,i,j,nlevels = damg[0]->nlevels,flag;
  MPI_Comm   comm;
  PetscTruth flg,usefd;
  SLES       sles;

  PetscFunctionBegin;
  if (!damg) SETERRQ(1,1,"Passing null as DAMG");

  /* create solvers for each level */
  for (i=0; i<nlevels; i++) {
    ierr = SNESCreate(damg[i]->comm,SNES_NONLINEAR_EQUATIONS,&damg[i]->snes);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(damg[i]->snes);CHKERRQ(ierr);
    ierr = SNESGetSLES(damg[i]->snes,&sles);CHKERRQ(ierr);
    ierr = DAMGSetUpLevel(damg,sles,i+1);CHKERRQ(ierr);
    damg[i]->solve = DAMGSolveSNES;
    damg[i]->computejacobian = jacobian;
    damg[i]->computefunction = function;
    ierr = DACreateLocalVector(damg[i]->da,&damg[i]->localX);CHKERRQ(ierr);
    ierr = DACreateLocalVector(damg[i]->da,&damg[i]->localF);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-damg_fd",&usefd);CHKERRQ(ierr);
  if (!jacobian || usefd) {
    ISColoring iscoloring;

    for (i=0; i<nlevels; i++) {
      ierr = DAGetColoring(damg[i]->da,&iscoloring,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(damg[i]->J,iscoloring,&damg[i]->fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(damg[i]->fdcoloring,(int(*)(void))function,damg[i]);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(damg[i]->fdcoloring);CHKERRQ(ierr);
      damg[i]->computejacobian = SNESDefaultComputeJacobianColor;
    }
  }

  for (i=0; i<nlevels; i++) {
    ierr = SNESSetJacobian(damg[i]->snes,damg[i]->J,damg[i]->B,DAMGComputeJacobian,damg);CHKERRQ(ierr);
    ierr = SNESSetFunction(damg[i]->snes,damg[i]->b,function,damg[i]);CHKERRQ(ierr);
  }

  /* Create interpolation scaling */
  for (i=1; i<nlevels; i++) {
    ierr = DAGetInterpolationScale(damg[i-1]->da,damg[i]->da,damg[i]->R,&damg[i]->Rscale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}














