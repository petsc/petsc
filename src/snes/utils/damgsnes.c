/*$Id: damgsnes.c,v 1.1 2000/07/10 20:35:41 bsmith Exp bsmith $*/
 
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
  int        ierr,i,nlevels = damg[0]->nlevels;
  SLES       sles;
  PC         pc;
  PetscTruth ismg;
  void       *pptr;

  PetscFunctionBegin;
  if (DAMGGetFine(damg)->computejacobian == SNESDefaultComputeJacobianColor) pptr = DAMGGetFine(damg);
  else pptr = DAMGGetFine(damg)->fdcoloring;

  ierr = (*DAMGGetFine(damg)->computejacobian)(snes,X,J,B,flag,pptr);CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = SLESSetOperators(DAMGGetFine(damg)->sles,DAMGGetFine(damg)->J,DAMGGetFine(damg)->J,*flag);CHKERRA(ierr);

    for (i=nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(damg[i]->R,X,damg[i-1]->x);CHKERRQ(ierr);
      X    = damg[i-1]->x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(damg[i]->Rscale,X,X);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      if (damg[i-1]->computejacobian == SNESDefaultComputeJacobianColor) pptr = damg[i-1];
      else pptr = damg[i-1]->fdcoloring;

      ierr = (*damg[i-1]->computejacobian)(snes,X,&damg[i-1]->J,&damg[i-1]->B,flag,pptr);CHKERRQ(ierr);

      ierr = SLESSetOperators(damg[i-1]->sles,damg[i-1]->J,damg[i-1]->B,*flag);CHKERRA(ierr);
    }
  }
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
int DAMGSetSNES(DAMG *damg,SNES snes)
{
  int        ierr,i,j,nlevels = damg[0]->nlevels,flag;
  MPI_Comm   comm;
  PetscTruth flg,usefd;
  SLES       sles;
  int        (*computefunction)(SNES,Vec,Vec,DAMG);
  int        (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,damg[0]->comm,&flag);CHKERRQ(ierr);
  if (flag != MPI_CONGRUENT && flag != MPI_IDENT) {
    SETERRQ(PETSC_ERR_ARG_NOTSAMECOMM,0,"Different communicators in the DAMG and the SNES");
  }

  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = DAMGSetSLES(damg,sles,PETSC_NULL);CHKERRQ(ierr);

  ierr = SNESGetJacobian(snes,0,0,0,&computejacobian);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-damg_use_fd",&usefd);CHKERRQ(ierr);
  if (!computejacobian || usefd) {
    ISColoring iscoloring;

    for (i=0; i<nlevels; i++) {
      ierr = DAGetColoring(damg[i]->da,&iscoloring,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(damg[i]->J,iscoloring,&damg[i]->fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetFunction(damg[i]->fdcoloring,(int(*)(void))computefunction,damg[i]);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(damg[i]->fdcoloring);CHKERRQ(ierr);
      damg[i]->computejacobian = SNESDefaultComputeJacobianColor;
    }
  }

  /* Create work vectors and matrix for each level */
  for (i=0; i<nlevels; i++) {
    ierr = DACreateLocalVector(damg[i]->da,&damg[i]->localX);CHKERRQ(ierr);
    ierr = DACreateLocalVector(damg[i]->da,&damg[i]->localF);CHKERRQ(ierr);
    if (!damg[i]->computejacobian) damg[i]->computejacobian = computejacobian;
    if (!damg[i]->computefunction) damg[i]->computefunction = computefunction;
  }

  /* Create interpolation/restriction between levels */
  for (i=1; i<nlevels; i++) {
    ierr = DAGetInterpolationScale(damg[i-1]->da,damg[i]->da,damg[i]->R,&damg[i]->Rscale);CHKERRQ(ierr);
  }

  /* overwrite the SNES Jacobian calculation with our own */
  ierr = SNESSetJacobian(snes,DAMGGetFine(damg)->J,DAMGGetFine(damg)->B,DAMGComputeJacobian,damg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}











