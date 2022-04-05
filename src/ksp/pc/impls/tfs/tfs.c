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

  PetscFunctionBegin;
  /* free the XXT datastructures */
  if (tfs->xxt) {
    PetscCall(XXT_free(tfs->xxt));
  }
  if (tfs->xyt) {
    PetscCall(XYT_free(tfs->xyt));
  }
  PetscCall(VecDestroy(&tfs->b));
  PetscCall(VecDestroy(&tfs->xd));
  PetscCall(VecDestroy(&tfs->xo));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_TFS_XXT(PC pc,Vec x,Vec y)
{
  PC_TFS            *tfs = (PC_TFS*)pc->data;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  PetscCall(XXT_solve(tfs->xxt,yy,(PetscScalar*)xx));
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_TFS_XYT(PC pc,Vec x,Vec y)
{
  PC_TFS            *tfs = (PC_TFS*)pc->data;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  PetscCall(XYT_solve(tfs->xyt,yy,(PetscScalar*)xx));
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTFSLocalMult_TFS(PC pc,PetscScalar *xin,PetscScalar *xout)
{
  PC_TFS         *tfs = (PC_TFS*)pc->data;
  Mat            A    = pc->pmat;
  Mat_MPIAIJ     *a   = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(VecPlaceArray(tfs->b,xout));
  PetscCall(VecPlaceArray(tfs->xd,xin));
  PetscCall(VecPlaceArray(tfs->xo,xin+tfs->nd));
  PetscCall(MatMult(a->A,tfs->xd,tfs->b));
  PetscCall(MatMultAdd(a->B,tfs->xo,tfs->b,tfs->b));
  PetscCall(VecResetArray(tfs->b));
  PetscCall(VecResetArray(tfs->xd));
  PetscCall(VecResetArray(tfs->xo));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_TFS(PC pc)
{
  PC_TFS         *tfs = (PC_TFS*)pc->data;
  Mat            A    = pc->pmat;
  Mat_MPIAIJ     *a   = (Mat_MPIAIJ*)A->data;
  PetscInt       *localtoglobal,ncol,i;
  PetscBool      ismpiaij;

  /*
  PetscBool      issymmetric;
  Petsc Real tol = 0.0;
  */

  PetscFunctionBegin;
  PetscCheck(A->cmap->N == A->rmap->N,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_SIZ,"matrix must be square");
  PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat,MATMPIAIJ,&ismpiaij));
  PetscCheck(ismpiaij,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Currently only supports MPIAIJ matrices");

  /* generate the local to global mapping */
  ncol = a->A->cmap->n + a->B->cmap->n;
  PetscCall(PetscMalloc1(ncol,&localtoglobal));
  for (i=0; i<a->A->cmap->n; i++) localtoglobal[i] = A->cmap->rstart + i + 1;
  for (i=0; i<a->B->cmap->n; i++) localtoglobal[i+a->A->cmap->n] = a->garray[i] + 1;

  /* generate the vectors needed for the local solves */
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->A->rmap->n,NULL,&tfs->b));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->A->cmap->n,NULL,&tfs->xd));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->B->cmap->n,NULL,&tfs->xo));
  tfs->nd = a->A->cmap->n;

  /*  ierr =  MatIsSymmetric(A,tol,&issymmetric); */
  /*  if (issymmetric) { */
  PetscCall(PetscBarrier((PetscObject)pc));
  if (A->symmetric) {
    tfs->xxt       = XXT_new();
    PetscCall(XXT_factor(tfs->xxt,localtoglobal,A->rmap->n,ncol,(PetscErrorCode (*)(void*,PetscScalar*,PetscScalar*))PCTFSLocalMult_TFS,pc));
    pc->ops->apply = PCApply_TFS_XXT;
  } else {
    tfs->xyt       = XYT_new();
    PetscCall(XYT_factor(tfs->xyt,localtoglobal,A->rmap->n,ncol,(PetscErrorCode (*)(void*,PetscScalar*,PetscScalar*))PCTFSLocalMult_TFS,pc));
    pc->ops->apply = PCApply_TFS_XYT;
  }

  PetscCall(PetscFree(localtoglobal));
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
  PC_TFS         *tfs;
  PetscMPIInt    cmp;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_compare(PETSC_COMM_WORLD,PetscObjectComm((PetscObject)pc),&cmp));
  PetscCheckFalse(cmp != MPI_IDENT && cmp != MPI_CONGRUENT,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"TFS only works with PETSC_COMM_WORLD objects");
  PetscCall(PetscNewLog(pc,&tfs));

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
