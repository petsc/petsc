#ifndef lint
static char vcid[] = "$Id: mpiaijpc.c,v 1.6 1995/11/01 23:18:18 bsmith Exp bsmith $";
#endif
/*
   Defines a block Jacobi preconditioner for the MPIAIJ format.
   At the moment only supports a single block per processor.
   This file knows about storage formats for MPIAIJ matrices.
   This code is nearly identical to that for the MPIROW format;
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
  int              i,ierr;

  for ( i=0; i<jac->n_local; i++ ) {
    ierr = SLESDestroy(jac->sles[i]); CHKERRQ(ierr);
  }
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

  ierr = SLESSolve(jac->sles[0],x,bjac->x,&its); CHKERRQ(ierr);
  ierr = VecCopy(bjac->x,y); CHKERRQ(ierr);
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


