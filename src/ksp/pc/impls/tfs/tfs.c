/* 
        Provides an interface to the Tufo-Fischer parallel direct solver
*/

#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/ksp/pc/impls/tfs/tfs.h>

typedef struct {
  xxt_ADT  xxt;
  xyt_ADT  xyt;                                                                                                                                                                       
  Vec      b,xd,xo;
  PetscInt nd;
} PC_TFS;

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_TFS"
PetscErrorCode PCDestroy_TFS(PC pc)
{
  PC_TFS *tfs = (PC_TFS*)pc->data;
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

#undef __FUNCT__
#define __FUNCT__ "PCApply_TFS_XXT"
static PetscErrorCode PCApply_TFS_XXT(PC pc,Vec x,Vec y)
{
  PC_TFS *tfs = (PC_TFS*)pc->data;
  PetscScalar    *xx,*yy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  ierr = XXT_solve(tfs->xxt,yy,xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_TFS_XYT"
static PetscErrorCode PCApply_TFS_XYT(PC pc,Vec x,Vec y)
{
  PC_TFS *tfs = (PC_TFS*)pc->data;
  PetscScalar    *xx,*yy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  ierr = XYT_solve(tfs->xyt,yy,xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCTFSLocalMult_TFS"
static PetscErrorCode PCTFSLocalMult_TFS(PC pc,PetscScalar *xin,PetscScalar *xout)
{
  PC_TFS        *tfs = (PC_TFS*)pc->data;
  Mat           A = pc->pmat;
  Mat_MPIAIJ    *a = (Mat_MPIAIJ*)A->data; 
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

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_TFS"
static PetscErrorCode PCSetUp_TFS(PC pc)
{
  PC_TFS        *tfs = (PC_TFS*)pc->data;
  Mat            A = pc->pmat;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt      *localtoglobal,ncol,i;
  PetscBool      ismpiaij;

  /*
  PetscBool      issymmetric;
  Petsc Real tol = 0.0;
  */

  PetscFunctionBegin;
  if (A->cmap->N != A->rmap->N) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  if (!ismpiaij) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Currently only supports MPIAIJ matrices");

  /* generate the local to global mapping */
  ncol = a->A->cmap->n + a->B->cmap->n;
  ierr = PetscMalloc((ncol)*sizeof(PetscInt),&localtoglobal);CHKERRQ(ierr);
  for (i=0; i<a->A->cmap->n; i++) {
    localtoglobal[i] = A->cmap->rstart + i + 1;
  }
  for (i=0; i<a->B->cmap->n; i++) {
    localtoglobal[i+a->A->cmap->n] = a->garray[i] + 1;
  }
  /* generate the vectors needed for the local solves */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->A->rmap->n,PETSC_NULL,&tfs->b);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->A->cmap->n,PETSC_NULL,&tfs->xd);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,a->B->cmap->n,PETSC_NULL,&tfs->xo);CHKERRQ(ierr);
  tfs->nd = a->A->cmap->n;


  /*  ierr =  MatIsSymmetric(A,tol,&issymmetric); */
  /*  if (issymmetric) { */
  ierr = PetscBarrier((PetscObject)pc);CHKERRQ(ierr);
  if (A->symmetric) {
    tfs->xxt       = XXT_new();
    ierr           = XXT_factor(tfs->xxt,localtoglobal,A->rmap->n,ncol,(void*)PCTFSLocalMult_TFS,pc);CHKERRQ(ierr);
    pc->ops->apply = PCApply_TFS_XXT;
  } else {
    tfs->xyt       = XYT_new();
    ierr           = XYT_factor(tfs->xyt,localtoglobal,A->rmap->n,ncol,(void*)PCTFSLocalMult_TFS,pc);CHKERRQ(ierr);
    pc->ops->apply = PCApply_TFS_XYT;
  }

  ierr = PetscFree(localtoglobal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_TFS"
static PetscErrorCode PCSetFromOptions_TFS(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCView_TFS"
static PetscErrorCode PCView_TFS(PC pc,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_TFS"
/*MC
     PCTFS - A parallel direct solver intended for problems with very few unknowns (like the 
         coarse grid in multigrid).

   Implemented by  Henry M. Tufo III and Paul Fischer

   Level: beginner

   Notes: Only implemented for the MPIAIJ matrices

          Only works on a solver object that lives on all of PETSC_COMM_WORLD!

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/
PetscErrorCode  PCCreate_TFS(PC pc)
{
  PetscErrorCode ierr;
  PC_TFS         *tfs;
  PetscMPIInt    cmp;

  PetscFunctionBegin;
  ierr = MPI_Comm_compare(PETSC_COMM_WORLD,((PetscObject)pc)->comm,&cmp);CHKERRQ(ierr);
  if (cmp != MPI_IDENT && cmp != MPI_CONGRUENT) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"TFS only works with PETSC_COMM_WORLD objects");
  ierr = PetscNewLog(pc,PC_TFS,&tfs);CHKERRQ(ierr);

  tfs->xxt = 0;
  tfs->xyt = 0;
  tfs->b   = 0;
  tfs->xd  = 0;
  tfs->xo  = 0;
  tfs->nd  = 0;

  pc->ops->apply               = 0;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_TFS;
  pc->ops->destroy             = PCDestroy_TFS;
  pc->ops->setfromoptions      = PCSetFromOptions_TFS;
  pc->ops->view                = PCView_TFS;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  pc->data                     = (void*)tfs;
  PetscFunctionReturn(0);
}
EXTERN_C_END

