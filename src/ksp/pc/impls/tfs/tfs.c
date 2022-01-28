/*
        Provides an interface to the Tufo-Fischer parallel direct solver
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/ksp/pc/impls/tfs/tfs.h>

typedef struct {
  xxt_ADT  xxt;
  xyt_ADT  xyt;
  Vec      b,xd,xo;
  PetscInt nd;
} PC_TFS;

PetscErrorCode PCDestroy_TFS(PC pc)
{
  PC_TFS         *tfs = (PC_TFS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free the XXT datastructures */
  if (tfs->xxt) {
    ierr = XXT_free(tfs->xxt);CHKERRQ(ierr);
  }
  if (tfs->xyt) {
    ierr = XYT_free(tfs->xyt);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&tfs->b);CHKERRQ(ierr);
  ierr = VecDestroy(&tfs->xd);CHKERRQ(ierr);
  ierr = VecDestroy(&tfs->xo);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_TFS_XXT(PC pc,Vec x,Vec y)
{
  PC_TFS            *tfs = (PC_TFS*)pc->data;
  PetscScalar       *yy;
  const PetscScalar *xx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  ierr = XXT_solve(tfs->xxt,yy,(PetscScalar*)xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_TFS_XYT(PC pc,Vec x,Vec y)
{
  PC_TFS            *tfs = (PC_TFS*)pc->data;
  PetscScalar       *yy;
  const PetscScalar *xx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  ierr = XYT_solve(tfs->xyt,yy,(PetscScalar*)xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTFSLocalMult_TFS(PC pc,PetscScalar *xin,PetscScalar *xout)
{
  PC_TFS         *tfs = (PC_TFS*)pc->data;
  Mat            A    = pc->pmat;
  Mat_MPIAIJ     *a   = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecPlaceArray(tfs->b,xout);CHKERRQ(ierr);
  ierr = VecPlaceArray(tfs->xd,xin);CHKERRQ(ierr);
  ierr = VecPlaceArray(tfs->xo,xin+tfs->nd);CHKERRQ(ierr);
  ierr = MatMult(a->A,tfs->xd,tfs->b);CHKERRQ(ierr);
  ierr = MatMultAdd(a->B,tfs->xo,tfs->b,tfs->b);CHKERRQ(ierr);
  ierr = VecResetArray(tfs->b);CHKERRQ(ierr);
  ierr = VecResetArray(tfs->xd);CHKERRQ(ierr);
  ierr = VecResetArray(tfs->xo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_TFS(PC pc)
{
  PC_TFS         *tfs = (PC_TFS*)pc->data;
  Mat            A    = pc->pmat;
  Mat_MPIAIJ     *a   = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       *localtoglobal,ncol,i;
  PetscBool      ismpiaij;

  /*
  PetscBool      issymmetric;
  Petsc Real tol = 0.0;
  */

  PetscFunctionBegin;
  PetscAssertFalse(A->cmap->N != A->rmap->N,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_SIZ,"matrix must be square");
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  PetscAssertFalse(!ismpiaij,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Currently only supports MPIAIJ matrices");

  /* generate the local to global mapping */
  ncol = a->A->cmap->n + a->B->cmap->n;
  ierr = PetscMalloc1(ncol,&localtoglobal);CHKERRQ(ierr);
  for (i=0; i<a->A->cmap->n; i++) localtoglobal[i] = A->cmap->rstart + i + 1;
  for (i=0; i<a->B->cmap->n; i++) localtoglobal[i+a->A->cmap->n] = a->garray[i] + 1;

  /* generate the vectors needed for the local solves */
  ierr    = VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->A->rmap->n,NULL,&tfs->b);CHKERRQ(ierr);
  ierr    = VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->A->cmap->n,NULL,&tfs->xd);CHKERRQ(ierr);
  ierr    = VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->B->cmap->n,NULL,&tfs->xo);CHKERRQ(ierr);
  tfs->nd = a->A->cmap->n;

  /*  ierr =  MatIsSymmetric(A,tol,&issymmetric); */
  /*  if (issymmetric) { */
  ierr = PetscBarrier((PetscObject)pc);CHKERRQ(ierr);
  if (A->symmetric) {
    tfs->xxt       = XXT_new();
    ierr           = XXT_factor(tfs->xxt,localtoglobal,A->rmap->n,ncol,(PetscErrorCode (*)(void*,PetscScalar*,PetscScalar*))PCTFSLocalMult_TFS,pc);CHKERRQ(ierr);
    pc->ops->apply = PCApply_TFS_XXT;
  } else {
    tfs->xyt       = XYT_new();
    ierr           = XYT_factor(tfs->xyt,localtoglobal,A->rmap->n,ncol,(PetscErrorCode (*)(void*,PetscScalar*,PetscScalar*))PCTFSLocalMult_TFS,pc);CHKERRQ(ierr);
    pc->ops->apply = PCApply_TFS_XYT;
  }

  ierr = PetscFree(localtoglobal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_TFS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
static PetscErrorCode PCView_TFS(PC pc,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
     PCTFS - A parallel direct solver intended for problems with very few unknowns (like the
         coarse grid in multigrid). Performs a Cholesky or LU factorization of a matrix defined by
         its local matrix vector product.

   Implemented by  Henry M. Tufo III and Paul Fischer originally for Nek5000 and called XXT or XYT

   Level: beginner

   Notes:
    Only implemented for the MPIAIJ matrices

    Only works on a solver object that lives on all of PETSC_COMM_WORLD!

    Only works for real numbers (is not built if PetscScalar is complex)

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/
PETSC_EXTERN PetscErrorCode PCCreate_TFS(PC pc)
{
  PetscErrorCode ierr;
  PC_TFS         *tfs;
  PetscMPIInt    cmp;

  PetscFunctionBegin;
  ierr = MPI_Comm_compare(PETSC_COMM_WORLD,PetscObjectComm((PetscObject)pc),&cmp);CHKERRMPI(ierr);
  PetscAssertFalse(cmp != MPI_IDENT && cmp != MPI_CONGRUENT,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"TFS only works with PETSC_COMM_WORLD objects");
  ierr = PetscNewLog(pc,&tfs);CHKERRQ(ierr);

  tfs->xxt = NULL;
  tfs->xyt = NULL;
  tfs->b   = NULL;
  tfs->xd  = NULL;
  tfs->xo  = NULL;
  tfs->nd  = 0;

  pc->ops->apply               = NULL;
  pc->ops->applytranspose      = NULL;
  pc->ops->setup               = PCSetUp_TFS;
  pc->ops->destroy             = PCDestroy_TFS;
  pc->ops->setfromoptions      = PCSetFromOptions_TFS;
  pc->ops->view                = PCView_TFS;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  pc->data                     = (void*)tfs;
  PetscFunctionReturn(0);
}

