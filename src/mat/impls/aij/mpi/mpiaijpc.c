#ifndef lint
static char vcid[] = "$Id: mpiaijpc.c,v 1.7 1995/12/03 02:42:38 bsmith Exp bsmith $";
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
  Vec  x;
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
  PetscFree(bjac); PetscFree(jac); 
  return 0;
}

int PCApply_BJacobiMPIAIJ(PC pc,Vec x, Vec y)
{
  int              ierr,its;
  PC_BJacobi       *jac = (PC_BJacobi *) pc->data;
  PC_BJacobiMPIAIJ *bjac = (PC_BJacobiMPIAIJ *) jac->data;
  Scalar           *array,*true_array;

  /* 
      The VecPlaceArray() is to avoid having to copy the 
    y vector into the bjac->x vector. The reason for 
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArray(y,&array); CHKERRQ(ierr);
  ierr = VecGetArray(bjac->x,&true_array); CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,array); CHKERRQ(ierr);
  ierr = SLESSolve(jac->sles[0],x,bjac->x,&its); CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,true_array); CHKERRQ(ierr);

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
  Vec              x;
  PC_BJacobiMPIAIJ *bjac;
  KSP              subksp;
  PC               subpc;
  MatType          type;

  if (jac->use_true_local) {
    MatGetType(pc->mat,&type);
    if (type != MATMPIAIJ) SETERRQ(1,"PCSetUp_BJacobiMPIAIJ:Incompatible matrix type.");
    matin = (Mat_MPIAIJ *) mat->data;
  }

  /* set default direct solver with no Krylov method */
  if (!pc->setupcalled) {
    ierr = SLESCreate(MPI_COMM_SELF,&sles); CHKERRQ(ierr);
    PLogObjectParent(pc,sles);
    ierr = SLESGetKSP(sles,&subksp); CHKERRQ(ierr);
    ierr = KSPSetMethod(subksp,KSPPREONLY); CHKERRQ(ierr);
    ierr = SLESGetPC(sles,&subpc); CHKERRQ(ierr);
    ierr = PCSetMethod(subpc,PCLU); CHKERRQ(ierr);
    ierr = SLESSetOptionsPrefix(sles,"-sub_"); CHKERRQ(ierr);
    ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);
/*
   This is not so good. The only reason we need to generate this vector
  is so KSP may generate seq vectors for the local solves
*/
    ierr = MatGetSize(pmatin->A,&m,&m); CHKERRQ(ierr);
    ierr = VecCreateSeq(MPI_COMM_SELF,m,&x); CHKERRQ(ierr);
    PLogObjectParent(pmat,x);

    pc->destroy  = PCDestroy_BJacobiMPIAIJ;
    pc->apply    = PCApply_BJacobiMPIAIJ;

    bjac         = (PC_BJacobiMPIAIJ *) PetscMalloc(sizeof(PC_BJacobiMPIAIJ));CHKPTRQ(bjac);
    PLogObjectMemory(pc,sizeof(PC_BJacobiMPIAIJ));
    bjac->x      = x;

    jac->sles    = (SLES*) PetscMalloc( sizeof(SLES) ); CHKPTRQ(jac->sles);
    jac->sles[0] = sles;
    jac->data    = (void *) bjac;
  }
  else {
    sles = jac->sles[0];
  }
  if (jac->l_true[0] == USE_TRUE_MATRIX) {
    ierr = SLESSetOperators(sles,matin->A,matin->A,pc->flag);
  }
  else if (jac->use_true_local)
    ierr = SLESSetOperators(sles,matin->A,pmatin->A,pc->flag);
  else
    ierr = SLESSetOperators(sles,pmatin->A,pmatin->A,pc->flag);
  CHKERRQ(ierr);

  return 0;
}


