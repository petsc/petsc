#ifndef lint
static char vcid[] = "$Id: mpiaijpc.c,v 1.13 1996/02/24 19:49:30 balay Exp curfman $";
#endif
/*
   Defines a block Jacobi preconditioner for the MPIAIJ format.
   Handles special case of  single block per processor.
   This file knows about storage formats for MPIAIJ matrices.
   The general case is handled in aijpc.c
*/
#include "mpiaij.h"
#include "src/pc/pcimpl.h"
#include "src/pc/impls/bjacobi/bjacobi.h"
#include "sles.h"

typedef struct {
  Vec  x, y;
} PC_BJacobiMPIAIJ;

int PCDestroy_BJacobiMPIAIJ(PetscObject obj)
{
  PC               pc = (PC) obj;
  PC_BJacobi       *jac = (PC_BJacobi *) pc->data;
  PC_BJacobiMPIAIJ *bjac = (PC_BJacobiMPIAIJ *) jac->data;
  int              ierr;

  ierr = SLESDestroy(jac->sles[0]); CHKERRQ(ierr);
  PetscFree(jac->sles);
  ierr = VecDestroy(bjac->x); CHKERRQ(ierr);
  ierr = VecDestroy(bjac->y); CHKERRQ(ierr);
  if (jac->l_lens) PetscFree(jac->l_lens);
  if (jac->g_lens) PetscFree(jac->g_lens);
  if (jac->l_true) PetscFree(jac->l_true);
  if (jac->g_true) PetscFree(jac->g_true);
  PetscFree(bjac); PetscFree(jac); 
  return 0;
}

int PCApply_BJacobiMPIAIJ(PC pc,Vec x, Vec y)
{
  int              ierr,its;
  PC_BJacobi       *jac = (PC_BJacobi *) pc->data;
  PC_BJacobiMPIAIJ *bjac = (PC_BJacobiMPIAIJ *) jac->data;
  Scalar           *x_array,*x_true_array, *y_array,*y_true_array;

  /* 
      The VecPlaceArray() is to avoid having to copy the 
    y vector into the bjac->x vector. The reason for 
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArray(x,&x_array); CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array); CHKERRQ(ierr);
  ierr = VecGetArray(bjac->x,&x_true_array); CHKERRQ(ierr);
  ierr = VecGetArray(bjac->y,&y_true_array); CHKERRQ(ierr); 
  ierr = VecPlaceArray(bjac->x,x_array); CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_array); CHKERRQ(ierr);
  ierr = SLESSolve(jac->sles[0],bjac->x,bjac->y,&its); CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,x_true_array); CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_true_array); CHKERRQ(ierr);

  return 0;
}

int PCSetUp_BJacobiMPIAIJ(PC pc)
{
  PC_BJacobi       *jac = (PC_BJacobi *) pc->data;
  Mat              mat = pc->mat, pmat = pc->pmat;
  Mat_MPIAIJ       *pmatin = (Mat_MPIAIJ *) pmat->data;
  Mat_MPIAIJ       *matin = 0;
  int              ierr, m;
  SLES             sles;
  Vec              x,y;
  PC_BJacobiMPIAIJ *bjac;
  KSP              subksp;
  PC               subpc;
  MatType          type;

  if (jac->use_true_local) {
    MatGetType(pc->mat,&type,PETSC_NULL);
    if (type != MATMPIAIJ) SETERRQ(1,"PCSetUp_BJacobiMPIAIJ:Incompatible matrix type.");
    matin = (Mat_MPIAIJ *) mat->data;
  }

  /* set default direct solver with no Krylov method */
  if (!pc->setupcalled) {
    ierr = SLESCreate(MPI_COMM_SELF,&sles); CHKERRQ(ierr);
    PLogObjectParent(pc,sles);
    ierr = SLESGetKSP(sles,&subksp); CHKERRQ(ierr);
    ierr = KSPSetType(subksp,KSPPREONLY); CHKERRQ(ierr);
    ierr = SLESGetPC(sles,&subpc); CHKERRQ(ierr);
    ierr = PCSetType(subpc,PCLU); CHKERRQ(ierr);
    ierr = SLESSetOptionsPrefix(sles,"sub_"); CHKERRQ(ierr);
    ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);
/*
   This is not so good. The only reason we need to generate this vector
  is so KSP may generate seq vectors for the local solves
*/
    ierr = MatGetSize(pmatin->A,&m,&m); CHKERRQ(ierr);
    ierr = VecCreateSeq(MPI_COMM_SELF,m,&x); CHKERRQ(ierr);
    ierr = VecCreateSeq(MPI_COMM_SELF,m,&y); CHKERRQ(ierr);
    PLogObjectParent(pmat,x);
    PLogObjectParent(pmat,y);

    pc->destroy  = PCDestroy_BJacobiMPIAIJ;
    pc->apply    = PCApply_BJacobiMPIAIJ;

    bjac         = (PC_BJacobiMPIAIJ *) PetscMalloc(sizeof(PC_BJacobiMPIAIJ));CHKPTRQ(bjac);
    PLogObjectMemory(pc,sizeof(PC_BJacobiMPIAIJ));
    bjac->x      = x;
    bjac->y      = y;

    jac->sles    = (SLES*) PetscMalloc( sizeof(SLES) ); CHKPTRQ(jac->sles);
    jac->sles[0] = sles;
    jac->data    = (void *) bjac;
  }
  else {
    sles = jac->sles[0];
    bjac = (PC_BJacobiMPIAIJ *)jac->data;
  }
  if (jac->l_true[0] == USE_TRUE_MATRIX) {
    ierr = SLESSetOperators(sles,matin->A,matin->A,pc->flag);
  }
  else if (jac->use_true_local)
    ierr = SLESSetOperators(sles,matin->A,pmatin->A,pc->flag);
  else
    ierr = SLESSetOperators(sles,pmatin->A,pmatin->A,pc->flag);
  CHKERRQ(ierr);
  ierr = SLESSetUp(sles,bjac->x,bjac->y); CHKERRQ(ierr);  
  return 0;
}


