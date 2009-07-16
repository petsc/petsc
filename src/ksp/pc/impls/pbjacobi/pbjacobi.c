#define PETSCKSP_DLL

/* 
   Include files needed for the PBJacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include "private/pcimpl.h"   /*I "petscpc.h" I*/

/* 
   Private context (data structure) for the PBJacobi preconditioner.  
*/
typedef struct {
  PetscScalar *diag;
  PetscInt    bs,mbs;
} PC_PBJacobi;

/*
   Currently only implemented for baij matrices and directly access baij
  data structures.
*/
#include "../src/mat/impls/baij/mpi/mpibaij.h"
#include "../src/mat/blockinvert.h"

#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_2"
static PetscErrorCode PCApply_PBJacobi_2(PC pc,Vec x,Vec y)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,m = jac->mbs;
  PetscScalar    *diag = jac->diag,x0,x1,*xx,*yy;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    x0 = xx[2*i]; x1 = xx[2*i+1];
    yy[2*i]   = diag[0]*x0 + diag[2]*x1;
    yy[2*i+1] = diag[1]*x0 + diag[3]*x1;
    diag     += 4;
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(6.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_3"
static PetscErrorCode PCApply_PBJacobi_3(PC pc,Vec x,Vec y)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,m = jac->mbs;
  PetscScalar    *diag = jac->diag,x0,x1,x2,*xx,*yy;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    x0 = xx[3*i]; x1 = xx[3*i+1]; x2 = xx[3*i+2];
    yy[3*i]   = diag[0]*x0 + diag[3]*x1 + diag[6]*x2;
    yy[3*i+1] = diag[1]*x0 + diag[4]*x1 + diag[7]*x2;
    yy[3*i+2] = diag[2]*x0 + diag[5]*x1 + diag[8]*x2;
    diag     += 9;
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(15.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_4"
static PetscErrorCode PCApply_PBJacobi_4(PC pc,Vec x,Vec y)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,m = jac->mbs;
  PetscScalar    *diag = jac->diag,x0,x1,x2,x3,*xx,*yy;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    x0 = xx[4*i]; x1 = xx[4*i+1]; x2 = xx[4*i+2]; x3 = xx[4*i+3];
    yy[4*i]   = diag[0]*x0 + diag[4]*x1 + diag[8]*x2  + diag[12]*x3;
    yy[4*i+1] = diag[1]*x0 + diag[5]*x1 + diag[9]*x2  + diag[13]*x3;
    yy[4*i+2] = diag[2]*x0 + diag[6]*x1 + diag[10]*x2 + diag[14]*x3;
    yy[4*i+3] = diag[3]*x0 + diag[7]*x1 + diag[11]*x2 + diag[15]*x3;
    diag     += 16;
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(28.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_5"
static PetscErrorCode PCApply_PBJacobi_5(PC pc,Vec x,Vec y)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,m = jac->mbs;
  PetscScalar    *diag = jac->diag,x0,x1,x2,x3,x4,*xx,*yy;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    x0 = xx[5*i]; x1 = xx[5*i+1]; x2 = xx[5*i+2]; x3 = xx[5*i+3]; x4 = xx[5*i+4];
    yy[5*i]   = diag[0]*x0 + diag[5]*x1 + diag[10]*x2  + diag[15]*x3 + diag[20]*x4;
    yy[5*i+1] = diag[1]*x0 + diag[6]*x1 + diag[11]*x2  + diag[16]*x3 + diag[21]*x4;
    yy[5*i+2] = diag[2]*x0 + diag[7]*x1 + diag[12]*x2 + diag[17]*x3 + diag[22]*x4;
    yy[5*i+3] = diag[3]*x0 + diag[8]*x1 + diag[13]*x2 + diag[18]*x3 + diag[23]*x4;
    yy[5*i+4] = diag[4]*x0 + diag[9]*x1 + diag[14]*x2 + diag[19]*x3 + diag[24]*x4;
    diag     += 25;
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(45.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_PBJacobi"
static PetscErrorCode PCSetUp_PBJacobi(PC pc)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscTruth     seqbaij,mpibaij,baij;
  Mat            A = pc->pmat;
  Mat_SeqBAIJ    *a;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSEQBAIJ,&seqbaij);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATMPIBAIJ,&mpibaij);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATBAIJ,&baij);CHKERRQ(ierr);
  if (!seqbaij && !mpibaij && !baij) {
    SETERRQ(PETSC_ERR_SUP,"Currently only supports BAIJ matrices");
  }
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  if (mpibaij || (baij && (size > 1))) A = ((Mat_MPIBAIJ*)A->data)->A;
  if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_ERR_SUP,"Supported only for square matrices and square storage");

  ierr        =  MatSeqBAIJInvertBlockDiagonal(A);CHKERRQ(ierr);
  a           = (Mat_SeqBAIJ*)A->data;
  jac->diag   = a->idiag;
  jac->bs     = A->rmap->bs;
  jac->mbs    = a->mbs;
  switch (jac->bs){
    case 2:
      pc->ops->apply = PCApply_PBJacobi_2;
      break;
    case 3:
      pc->ops->apply = PCApply_PBJacobi_3;
      break;
    case 4:
      pc->ops->apply = PCApply_PBJacobi_4;
      break;
    case 5:
      pc->ops->apply = PCApply_PBJacobi_5;
      break;
    default: 
      SETERRQ1(PETSC_ERR_SUP,"not supported for block size %D",jac->bs);
  }

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_PBJacobi"
static PetscErrorCode PCDestroy_PBJacobi(PC pc)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*MC
     PCPBJACOBI - Point block Jacobi

   Level: beginner

  Concepts: point block Jacobi

   Notes: Only implemented for the BAIJ matrix formats.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_PBJacobi"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_PBJacobi(PC pc)
{
  PC_PBJacobi    *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNewLog(pc,PC_PBJacobi,&jac);CHKERRQ(ierr);
  pc->data  = (void*)jac;

  /*
     Initialize the pointers to vectors to ZERO; these will be used to store
     diagonal entries of the matrix for fast preconditioner application.
  */
  jac->diag          = 0;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = 0; /*set depending on the block size */
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_PBJacobi;
  pc->ops->destroy             = PCDestroy_PBJacobi;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END


