
/*
   Include files needed for the PBJacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

/*
   Private context (data structure) for the PBJacobi preconditioner.
*/
typedef struct {
  const MatScalar *diag;
  PetscInt        bs,mbs;
} PC_PBJacobi;

static PetscErrorCode PCApply_PBJacobi_1(PC pc,Vec x,Vec y)
{
  PC_PBJacobi       *jac = (PC_PBJacobi*)pc->data;
  PetscInt          i,m = jac->mbs;
  const MatScalar   *diag = jac->diag;
  const PetscScalar *xx;
  PetscScalar       *yy;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  for (i=0; i<m; i++) yy[i] = diag[i]*xx[i];
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops(m));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PBJacobi_2(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,*yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  for (i=0; i<m; i++) {
    x0        = xx[2*i]; x1 = xx[2*i+1];
    yy[2*i]   = diag[0]*x0 + diag[2]*x1;
    yy[2*i+1] = diag[1]*x0 + diag[3]*x1;
    diag     += 4;
  }
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops(6.0*m));
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_3(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,*yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  for (i=0; i<m; i++) {
    x0 = xx[3*i]; x1 = xx[3*i+1]; x2 = xx[3*i+2];

    yy[3*i]   = diag[0]*x0 + diag[3]*x1 + diag[6]*x2;
    yy[3*i+1] = diag[1]*x0 + diag[4]*x1 + diag[7]*x2;
    yy[3*i+2] = diag[2]*x0 + diag[5]*x1 + diag[8]*x2;
    diag     += 9;
  }
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops(15.0*m));
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_4(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,x3,*yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  for (i=0; i<m; i++) {
    x0 = xx[4*i]; x1 = xx[4*i+1]; x2 = xx[4*i+2]; x3 = xx[4*i+3];

    yy[4*i]   = diag[0]*x0 + diag[4]*x1 + diag[8]*x2  + diag[12]*x3;
    yy[4*i+1] = diag[1]*x0 + diag[5]*x1 + diag[9]*x2  + diag[13]*x3;
    yy[4*i+2] = diag[2]*x0 + diag[6]*x1 + diag[10]*x2 + diag[14]*x3;
    yy[4*i+3] = diag[3]*x0 + diag[7]*x1 + diag[11]*x2 + diag[15]*x3;
    diag     += 16;
  }
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops(28.0*m));
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_5(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,x3,x4,*yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  for (i=0; i<m; i++) {
    x0 = xx[5*i]; x1 = xx[5*i+1]; x2 = xx[5*i+2]; x3 = xx[5*i+3]; x4 = xx[5*i+4];

    yy[5*i]   = diag[0]*x0 + diag[5]*x1 + diag[10]*x2  + diag[15]*x3 + diag[20]*x4;
    yy[5*i+1] = diag[1]*x0 + diag[6]*x1 + diag[11]*x2  + diag[16]*x3 + diag[21]*x4;
    yy[5*i+2] = diag[2]*x0 + diag[7]*x1 + diag[12]*x2 + diag[17]*x3 + diag[22]*x4;
    yy[5*i+3] = diag[3]*x0 + diag[8]*x1 + diag[13]*x2 + diag[18]*x3 + diag[23]*x4;
    yy[5*i+4] = diag[4]*x0 + diag[9]*x1 + diag[14]*x2 + diag[19]*x3 + diag[24]*x4;
    diag     += 25;
  }
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops(45.0*m));
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_6(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,x3,x4,x5,*yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
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
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops(66.0*m));
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_7(PC pc,Vec x,Vec y)
{
  PC_PBJacobi     *jac = (PC_PBJacobi*)pc->data;
  PetscInt        i,m = jac->mbs;
  const MatScalar *diag = jac->diag;
  PetscScalar     x0,x1,x2,x3,x4,x5,x6,*yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
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
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops(91.0*m)); /* 2*bs2 - bs */
  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_PBJacobi_N(PC pc,Vec x,Vec y)
{
  PC_PBJacobi       *jac = (PC_PBJacobi*)pc->data;
  PetscInt          i,ib,jb;
  const PetscInt    m = jac->mbs;
  const PetscInt    bs = jac->bs;
  const MatScalar   *diag = jac->diag;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  for (i=0; i<m; i++) {
    for (ib=0; ib<bs; ib++) {
      PetscScalar rowsum = 0;
      for (jb=0; jb<bs; jb++) {
        rowsum += diag[ib+jb*bs] * xx[bs*i+jb];
      }
      yy[bs*i+ib] = rowsum;
    }
    diag += bs*bs;
  }
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops((2.0*bs*bs-bs)*m)); /* 2*bs2 - bs */
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_PBJacobi_N(PC pc,Vec x,Vec y)
{
  PC_PBJacobi       *jac = (PC_PBJacobi*)pc->data;
  PetscInt          i,j,k,m = jac->mbs,bs=jac->bs;
  const MatScalar   *diag = jac->diag;
  const PetscScalar *xx;
  PetscScalar       *yy;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  for (i=0; i<m; i++) {
    for (j=0; j<bs; j++) yy[i*bs+j] = 0.;
    for (j=0; j<bs; j++) {
      for (k=0; k<bs; k++) {
        yy[i*bs+k] += diag[k*bs+j]*xx[i*bs+j];
      }
    }
    diag += bs*bs;
  }
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscCall(PetscLogFlops(m*bs*(2*bs-1)));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode PCSetUp_PBJacobi(PC pc)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  Mat            A = pc->pmat;
  MatFactorError err;
  PetscInt       nlocal;

  PetscFunctionBegin;
  PetscCall(MatInvertBlockDiagonal(A,&jac->diag));
  PetscCall(MatFactorGetError(A,&err));
  if (err) pc->failedreason = (PCFailedReason)err;

  PetscCall(MatGetBlockSize(A,&jac->bs));
  PetscCall(MatGetLocalSize(A,&nlocal,NULL));
  jac->mbs = nlocal/jac->bs;
  switch (jac->bs) {
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
    pc->ops->apply = PCApply_PBJacobi_N;
    break;
  }
  pc->ops->applytranspose = PCApplyTranspose_PBJacobi_N;
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode PCDestroy_PBJacobi(PC pc)
{
  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_PBJacobi(PC pc,PetscViewer viewer)
{
  PC_PBJacobi    *jac = (PC_PBJacobi*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  point-block size %" PetscInt_FMT "\n",jac->bs));
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PCPBJACOBI - Point block Jacobi preconditioner

   Notes:
     See PCJACOBI for point Jacobi preconditioning, PCVPBJACOBI for variable size point block Jacobi and PCBJACOBI for large blocks

     This works for AIJ and BAIJ matrices and uses the blocksize provided to the matrix

     Uses dense LU factorization with partial pivoting to invert the blocks; if a zero pivot
     is detected a PETSc error is generated.

   Developer Notes:
     This should support the PCSetErrorIfFailure() flag set to PETSC_TRUE to allow
     the factorization to continue even after a zero pivot is found resulting in a Nan and hence
     terminating KSP with a KSP_DIVERGED_NANORIF allowing
     a nonlinear solver/ODE integrator to recover without stopping the program as currently happens.

     Perhaps should provide an option that allows generation of a valid preconditioner
     even if a block is singular as the PCJACOBI does.

   Level: beginner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCJACOBI, PCVPBJACOBI, PCBJACOBI

M*/

PETSC_EXTERN PetscErrorCode PCCreate_PBJacobi(PC pc)
{
  PC_PBJacobi    *jac;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  PetscCall(PetscNewLog(pc,&jac));
  pc->data = (void*)jac;

  /*
     Initialize the pointers to vectors to ZERO; these will be used to store
     diagonal entries of the matrix for fast preconditioner application.
  */
  jac->diag = NULL;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = NULL; /*set depending on the block size */
  pc->ops->applytranspose      = NULL;
  pc->ops->setup               = PCSetUp_PBJacobi;
  pc->ops->destroy             = PCDestroy_PBJacobi;
  pc->ops->setfromoptions      = NULL;
  pc->ops->view                = PCView_PBJacobi;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  PetscFunctionReturn(0);
}
