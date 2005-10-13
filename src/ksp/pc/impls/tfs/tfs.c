#define PETSCKSP_DLL
/* 
        Provides an interface to the Tufo-Fischer parallel direct solver
*/

#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/ksp/pc/impls/tfs/tfs.h"

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
  if (tfs->b) {
  ierr = VecDestroy(tfs->b);CHKERRQ(ierr);
  }
  if (tfs->xd) {
  ierr = VecDestroy(tfs->xd);CHKERRQ(ierr);
  }
  if (tfs->xo) {
  ierr = VecDestroy(tfs->xo);CHKERRQ(ierr);
  }
  ierr = PetscFree(tfs);CHKERRQ(ierr);
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
#define __FUNCT__ "LocalMult_TFS"
static PetscErrorCode LocalMult_TFS(PC pc,PetscScalar *xin,PetscScalar *xout)
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
  PetscTruth     ismpiaij;

  /*
  PetscTruth     issymmetric;
  Petsc Real tol = 0.0;
  */

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  if (!ismpiaij) {
    SETERRQ(PETSC_ERR_SUP,"Currently only supports MPIAIJ matrices");
  }

  /* generate the local to global mapping */
  ncol = a->A->n + a->B->n;
  ierr = PetscMalloc((ncol)*sizeof(int),&localtoglobal);CHKERRQ(ierr);
  for (i=0; i<a->A->n; i++) {
    localtoglobal[i] = a->cstart + i + 1;
  }
  for (i=0; i<a->B->n; i++) {
    localtoglobal[i+a->A->n] = a->garray[i] + 1;
  }
  /* generate the vectors needed for the local solves */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->A->m,PETSC_NULL,&tfs->b);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->A->n,PETSC_NULL,&tfs->xd);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->B->n,PETSC_NULL,&tfs->xo);CHKERRQ(ierr);
  tfs->nd = a->A->n;


  /*  ierr =  MatIsSymmetric(A,tol,&issymmetric); */
  /*  if (issymmetric) { */
  ierr = PetscBarrier((PetscObject)pc);CHKERRQ(ierr);
  if (A->symmetric) {
    tfs->xxt       = XXT_new();
    ierr           = XXT_factor(tfs->xxt,localtoglobal,A->m,ncol,(void*)LocalMult_TFS,pc);CHKERRQ(ierr);
    pc->ops->apply = PCApply_TFS_XXT;
  } else {
    tfs->xyt       = XYT_new();
    ierr           = XYT_factor(tfs->xyt,localtoglobal,A->m,ncol,(void*)LocalMult_TFS,pc);CHKERRQ(ierr);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_TFS(PC pc)
{
  PetscErrorCode ierr;
  PC_TFS         *tfs;

  PetscFunctionBegin;
  ierr = PetscNew(PC_TFS,&tfs);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_TFS));CHKERRQ(ierr);

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

