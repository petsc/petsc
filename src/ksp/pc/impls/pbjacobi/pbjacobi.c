/*$Id: pbjacobi.c,v 1.4 2001/08/07 03:03:42 balay Exp $*/

/* 
   Include files needed for the PBJacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include "src/ksp/pc/pcimpl.h"   /*I "petscpc.h" I*/

/* 
   Private context (data structure) for the PBJacobi preconditioner.  
*/
typedef struct {
  PetscScalar *diag;
  int         bs,mbs;
} PC_PBJacobi;

/*
   Currently only implemented for baij matrices and directly access baij
  data structures.
*/
#include "src/mat/impls/baij/mpi/mpibaij.h"
#include "src/inline/ilu.h"

#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_2"
static int PCApply_PBJacobi_2(PC pc,Vec x,Vec y)
{
  PC_PBJacobi *jac = (PC_PBJacobi*)pc->data;
  int         ierr,i,m = jac->mbs;
  PetscScalar *diag = jac->diag,x0,x1,*xx,*yy;
  
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
  PetscLogFlops(6*m);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_3"
static int PCApply_PBJacobi_3(PC pc,Vec x,Vec y)
{
  PC_PBJacobi *jac = (PC_PBJacobi*)pc->data;
  int         ierr,i,m = jac->mbs;
  PetscScalar *diag = jac->diag,x0,x1,x2,*xx,*yy;
  
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
  PetscLogFlops(15*m);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_4"
static int PCApply_PBJacobi_4(PC pc,Vec x,Vec y)
{
  PC_PBJacobi *jac = (PC_PBJacobi*)pc->data;
  int         ierr,i,m = jac->mbs;
  PetscScalar *diag = jac->diag,x0,x1,x2,x3,*xx,*yy;
  
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
  PetscLogFlops(28*m);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_5"
static int PCApply_PBJacobi_5(PC pc,Vec x,Vec y)
{
  PC_PBJacobi *jac = (PC_PBJacobi*)pc->data;
  int         ierr,i,m = jac->mbs;
  PetscScalar *diag = jac->diag,x0,x1,x2,x3,x4,*xx,*yy;
  
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
  PetscLogFlops(45*m);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_PBJacobi"
static int PCSetUp_PBJacobi(PC pc)
{
  PC_PBJacobi *jac = (PC_PBJacobi*)pc->data;
  int         ierr,i,*diag_offset,bs2,size;
  PetscTruth  seqbaij,mpibaij,baij;
  Mat         A = pc->pmat;
  PetscScalar *diag,*odiag,*v;
  Mat_SeqBAIJ *a;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSEQBAIJ,&seqbaij);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATMPIBAIJ,&mpibaij);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATBAIJ,&baij);CHKERRQ(ierr);
  if (!seqbaij && !mpibaij && !baij) {
    SETERRQ(1,"Currently only supports BAIJ matrices");
  }
  ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  if (mpibaij || (baij && (size > 1))) A = ((Mat_MPIBAIJ*)A->data)->A;
  if (A->m != A->n) SETERRQ(1,"Supported only for square matrices and square storage");
  ierr        = MatMarkDiagonal_SeqBAIJ(A);CHKERRQ(ierr);
  a           = (Mat_SeqBAIJ*)A->data;
  bs2         = a->bs*a->bs;
  diag_offset = a->diag;
  v           = a->a;
  ierr        = PetscMalloc((bs2*a->mbs+1)*sizeof(PetscScalar),&diag);CHKERRQ(ierr);
  PetscLogObjectMemory(pc,bs2*a->mbs*sizeof(PetscScalar));
  jac->diag   = diag;
  jac->bs     = a->bs;
  jac->mbs    = a->mbs;

  /* factor and invert each block */
  switch (a->bs){
    case 2:
      for (i=0; i<a->mbs; i++) {
        odiag   = v + 4*diag_offset[i];
        diag[0] = odiag[0]; diag[1] = odiag[1]; diag[2] = odiag[2]; diag[3] = odiag[3];
	ierr    = Kernel_A_gets_inverse_A_2(diag);CHKERRQ(ierr);
	diag   += 4;
      }
      pc->ops->apply = PCApply_PBJacobi_2;
      break;
    case 3:
      for (i=0; i<a->mbs; i++) {
        odiag   = v + 9*diag_offset[i];
        diag[0] = odiag[0]; diag[1] = odiag[1]; diag[2] = odiag[2]; diag[3] = odiag[3];
        diag[4] = odiag[4]; diag[5] = odiag[5]; diag[6] = odiag[6]; diag[7] = odiag[7];
        diag[8] = odiag[8]; diag[9] = odiag[9]; 
	ierr    = Kernel_A_gets_inverse_A_3(diag);CHKERRQ(ierr);
	diag   += 9;
      }
      pc->ops->apply = PCApply_PBJacobi_3;
      break;
    case 4:
      for (i=0; i<a->mbs; i++) {
        odiag = v + 16*diag_offset[i];
        ierr  = PetscMemcpy(diag,odiag,16*sizeof(PetscScalar));CHKERRQ(ierr);
	ierr  = Kernel_A_gets_inverse_A_4(diag);CHKERRQ(ierr);
	diag += 16;
      }
      pc->ops->apply = PCApply_PBJacobi_4;
      break;
    case 5:
      for (i=0; i<a->mbs; i++) {
        odiag = v + 25*diag_offset[i];
        ierr  = PetscMemcpy(diag,odiag,25*sizeof(PetscScalar));CHKERRQ(ierr);
	ierr  = Kernel_A_gets_inverse_A_5(diag);CHKERRQ(ierr);
	diag += 25;
      }
      pc->ops->apply = PCApply_PBJacobi_5;
      break;
    default: 
      SETERRQ1(1,"not supported for block size %d",a->bs);
  }

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_PBJacobi"
static int PCDestroy_PBJacobi(PC pc)
{
  PC_PBJacobi *jac = (PC_PBJacobi*)pc->data;
  int         ierr;

  PetscFunctionBegin;
  if (jac->diag)     {ierr = PetscFree(jac->diag);CHKERRQ(ierr);}

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_PBJacobi"
int PCCreate_PBJacobi(PC pc)
{
  PC_PBJacobi *jac;
  int       ierr;

  PetscFunctionBegin;

  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNew(PC_PBJacobi,&jac);CHKERRQ(ierr);
  pc->data  = (void*)jac;

  /*
     Logs the memory usage; this is not needed but allows PETSc to 
     monitor how much memory is being used for various purposes.
  */
  PetscLogObjectMemory(pc,sizeof(PC_PBJacobi));

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


