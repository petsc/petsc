
/* 
   Include files needed for the PBJacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include <petsc-private/matimpl.h>
#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/

/* 
   Private context (data structure) for the PBJacobi preconditioner.  
*/
typedef struct {
  const MatScalar *diag;
  PetscInt    bs,mbs;
} PC_PBJacobi;


#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_1"
static PetscErrorCode PCApply_PBJacobi_1(PC pc,Vec x,Vec y)
{
  PC_PBJacobi       *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          i,m = jac->mbs;
  const MatScalar   *diag = jac->diag;
  const PetscScalar *xx;
  PetscScalar       *yy;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    yy[i] = diag[i]*xx[i];
  }
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_2"
static PetscErrorCode PCApply_PBJacobi_2(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,*xx,*yy;
  
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
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,*xx,*yy;
  
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
  PC_PBJacobi      *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode   ierr;
  PetscInt         i,m = jac->mbs;
  const MatScalar  *diag = jac->diag;
  PetscScalar      x0,x1,x2,x3,*xx,*yy;
  
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
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,x3,x4,*xx,*yy;
  
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
#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_6"
static PetscErrorCode PCApply_PBJacobi_6(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,x3,x4,x5,*xx,*yy;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    x0 = xx[6*i]; x1 = xx[6*i+1]; x2 = xx[6*i+2]; x3 = xx[6*i+3]; x4 = xx[6*i+4]; x5 = xx[6*i+5];
    yy[6*i]   = diag[0]*x0 + diag[6]*x1  + diag[12]*x2  + diag[18]*x3 + diag[24]*x4 + diag[30]*x5;
    yy[6*i+1] = diag[1]*x0 + diag[7]*x1  + diag[13]*x2  + diag[19]*x3 + diag[25]*x4 + diag[31]*x5;
    yy[6*i+2] = diag[2]*x0 + diag[8]*x1  + diag[14]*x2  + diag[20]*x3 + diag[26]*x4 + diag[32]*x5;
    yy[6*i+3] = diag[3]*x0 + diag[9]*x1  + diag[15]*x2  + diag[21]*x3 + diag[27]*x4 + diag[33]*x5;
    yy[6*i+4] = diag[4]*x0 + diag[10]*x1 + diag[16]*x2  + diag[22]*x3 + diag[28]*x4 + diag[34]*x5;
    yy[6*i+5] = diag[5]*x0 + diag[11]*x1 + diag[17]*x2  + diag[23]*x3 + diag[29]*x4 + diag[35]*x5;
    diag     += 36;
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(66.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCApply_PBJacobi_7"
static PetscErrorCode PCApply_PBJacobi_7(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode  ierr;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,x3,x4,x5,x6,*xx,*yy;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    x0 = xx[7*i]; x1 = xx[7*i+1]; x2 = xx[7*i+2]; x3 = xx[7*i+3]; x4 = xx[7*i+4]; x5 = xx[7*i+5]; x6 = xx[7*i+6];
    yy[7*i]   = diag[0]*x0 + diag[7]*x1  + diag[14]*x2  + diag[21]*x3 + diag[28]*x4 + diag[35]*x5 + diag[42]*x6;
    yy[7*i+1] = diag[1]*x0 + diag[8]*x1  + diag[15]*x2  + diag[22]*x3 + diag[29]*x4 + diag[36]*x5 + diag[43]*x6;
    yy[7*i+2] = diag[2]*x0 + diag[9]*x1  + diag[16]*x2  + diag[23]*x3 + diag[30]*x4 + diag[37]*x5 + diag[44]*x6;
    yy[7*i+3] = diag[3]*x0 + diag[10]*x1 + diag[17]*x2  + diag[24]*x3 + diag[31]*x4 + diag[38]*x5 + diag[45]*x6;
    yy[7*i+4] = diag[4]*x0 + diag[11]*x1 + diag[18]*x2  + diag[25]*x3 + diag[32]*x4 + diag[39]*x5 + diag[46]*x6;
    yy[7*i+5] = diag[5]*x0 + diag[12]*x1 + diag[19]*x2  + diag[26]*x3 + diag[33]*x4 + diag[40]*x5 + diag[47]*x6;
    yy[7*i+6] = diag[6]*x0 + diag[13]*x1 + diag[20]*x2  + diag[27]*x3 + diag[34]*x4 + diag[41]*x5 + diag[48]*x6;
    diag     += 49;
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  ierr = PetscLogFlops(80.0*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_PBJacobi"
static PetscErrorCode PCSetUp_PBJacobi(PC pc)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscErrorCode ierr;
  Mat            A = pc->pmat;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Supported only for square matrices and square storage");

  ierr        = MatInvertBlockDiagonal(A,&jac->diag);CHKERRQ(ierr);
  jac->bs     = A->rmap->bs;
  jac->mbs    = A->rmap->n/A->rmap->bs;
  switch (jac->bs){
    case 1:
      pc->ops->apply = PCApply_PBJacobi_1;
      break;
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
    case 6:
      pc->ops->apply = PCApply_PBJacobi_6;
      break;
    case 7:
      pc->ops->apply = PCApply_PBJacobi_7;
      break;
    default: 
      SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"not supported for block size %D",jac->bs);
  }

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_PBJacobi"
static PetscErrorCode PCDestroy_PBJacobi(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_PBJacobi"
static PetscErrorCode PCView_PBJacobi(PC pc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  point-block Jacobi: block size %D\n",jac->bs);CHKERRQ(ierr);
  } else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for point-block Jacobi",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PCPBJACOBI - Point block Jacobi

   Level: beginner

  Concepts: point block Jacobi


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_PBJacobi"
PetscErrorCode  PCCreate_PBJacobi(PC pc)
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
  pc->ops->view                = PCView_PBJacobi;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END


