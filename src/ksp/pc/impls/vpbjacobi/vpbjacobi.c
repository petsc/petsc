#include <../src/ksp/pc/impls/vpbjacobi/vpbjacobi.h>

static PetscErrorCode PCApply_VPBJacobi(PC pc, Vec x, Vec y)
{
  PC_VPBJacobi      *jac = (PC_VPBJacobi *)pc->data;
  PetscInt           i, ncnt = 0;
  const MatScalar   *diag = jac->diag;
  PetscInt           ib, jb, bs;
  const PetscScalar *xx;
  PetscScalar       *yy, x0, x1, x2, x3, x4, x5, x6;
  PetscInt           nblocks;
  const PetscInt    *bsizes;

  PetscFunctionBegin;
  PetscCall(MatGetVariableBlockSizes(pc->pmat, &nblocks, &bsizes));
  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(y, &yy));
  for (i = 0; i < nblocks; i++) {
    bs = bsizes[i];
    switch (bs) {
    case 1:
      yy[ncnt] = *diag * xx[ncnt];
      break;
    case 2:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      yy[ncnt]     = diag[0] * x0 + diag[2] * x1;
      yy[ncnt + 1] = diag[1] * x0 + diag[3] * x1;
      break;
    case 3:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      yy[ncnt]     = diag[0] * x0 + diag[3] * x1 + diag[6] * x2;
      yy[ncnt + 1] = diag[1] * x0 + diag[4] * x1 + diag[7] * x2;
      yy[ncnt + 2] = diag[2] * x0 + diag[5] * x1 + diag[8] * x2;
      break;
    case 4:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      x3           = xx[ncnt + 3];
      yy[ncnt]     = diag[0] * x0 + diag[4] * x1 + diag[8] * x2 + diag[12] * x3;
      yy[ncnt + 1] = diag[1] * x0 + diag[5] * x1 + diag[9] * x2 + diag[13] * x3;
      yy[ncnt + 2] = diag[2] * x0 + diag[6] * x1 + diag[10] * x2 + diag[14] * x3;
      yy[ncnt + 3] = diag[3] * x0 + diag[7] * x1 + diag[11] * x2 + diag[15] * x3;
      break;
    case 5:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      x3           = xx[ncnt + 3];
      x4           = xx[ncnt + 4];
      yy[ncnt]     = diag[0] * x0 + diag[5] * x1 + diag[10] * x2 + diag[15] * x3 + diag[20] * x4;
      yy[ncnt + 1] = diag[1] * x0 + diag[6] * x1 + diag[11] * x2 + diag[16] * x3 + diag[21] * x4;
      yy[ncnt + 2] = diag[2] * x0 + diag[7] * x1 + diag[12] * x2 + diag[17] * x3 + diag[22] * x4;
      yy[ncnt + 3] = diag[3] * x0 + diag[8] * x1 + diag[13] * x2 + diag[18] * x3 + diag[23] * x4;
      yy[ncnt + 4] = diag[4] * x0 + diag[9] * x1 + diag[14] * x2 + diag[19] * x3 + diag[24] * x4;
      break;
    case 6:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      x3           = xx[ncnt + 3];
      x4           = xx[ncnt + 4];
      x5           = xx[ncnt + 5];
      yy[ncnt]     = diag[0] * x0 + diag[6] * x1 + diag[12] * x2 + diag[18] * x3 + diag[24] * x4 + diag[30] * x5;
      yy[ncnt + 1] = diag[1] * x0 + diag[7] * x1 + diag[13] * x2 + diag[19] * x3 + diag[25] * x4 + diag[31] * x5;
      yy[ncnt + 2] = diag[2] * x0 + diag[8] * x1 + diag[14] * x2 + diag[20] * x3 + diag[26] * x4 + diag[32] * x5;
      yy[ncnt + 3] = diag[3] * x0 + diag[9] * x1 + diag[15] * x2 + diag[21] * x3 + diag[27] * x4 + diag[33] * x5;
      yy[ncnt + 4] = diag[4] * x0 + diag[10] * x1 + diag[16] * x2 + diag[22] * x3 + diag[28] * x4 + diag[34] * x5;
      yy[ncnt + 5] = diag[5] * x0 + diag[11] * x1 + diag[17] * x2 + diag[23] * x3 + diag[29] * x4 + diag[35] * x5;
      break;
    case 7:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      x3           = xx[ncnt + 3];
      x4           = xx[ncnt + 4];
      x5           = xx[ncnt + 5];
      x6           = xx[ncnt + 6];
      yy[ncnt]     = diag[0] * x0 + diag[7] * x1 + diag[14] * x2 + diag[21] * x3 + diag[28] * x4 + diag[35] * x5 + diag[42] * x6;
      yy[ncnt + 1] = diag[1] * x0 + diag[8] * x1 + diag[15] * x2 + diag[22] * x3 + diag[29] * x4 + diag[36] * x5 + diag[43] * x6;
      yy[ncnt + 2] = diag[2] * x0 + diag[9] * x1 + diag[16] * x2 + diag[23] * x3 + diag[30] * x4 + diag[37] * x5 + diag[44] * x6;
      yy[ncnt + 3] = diag[3] * x0 + diag[10] * x1 + diag[17] * x2 + diag[24] * x3 + diag[31] * x4 + diag[38] * x5 + diag[45] * x6;
      yy[ncnt + 4] = diag[4] * x0 + diag[11] * x1 + diag[18] * x2 + diag[25] * x3 + diag[32] * x4 + diag[39] * x5 + diag[46] * x6;
      yy[ncnt + 5] = diag[5] * x0 + diag[12] * x1 + diag[19] * x2 + diag[26] * x3 + diag[33] * x4 + diag[40] * x5 + diag[47] * x6;
      yy[ncnt + 6] = diag[6] * x0 + diag[13] * x1 + diag[20] * x2 + diag[27] * x3 + diag[34] * x4 + diag[41] * x5 + diag[48] * x6;
      break;
    default:
      for (ib = 0; ib < bs; ib++) {
        PetscScalar rowsum = 0;
        for (jb = 0; jb < bs; jb++) rowsum += diag[ib + jb * bs] * xx[ncnt + jb];
        yy[ncnt + ib] = rowsum;
      }
    }
    ncnt += bsizes[i];
    diag += bsizes[i] * bsizes[i];
  }
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(y, &yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_VPBJacobi(PC pc, Vec x, Vec y)
{
  PC_VPBJacobi      *jac = (PC_VPBJacobi *)pc->data;
  PetscInt           i, ncnt = 0;
  const MatScalar   *diag = jac->diag;
  PetscInt           ib, jb, bs;
  const PetscScalar *xx;
  PetscScalar       *yy, x0, x1, x2, x3, x4, x5, x6;
  PetscInt           nblocks;
  const PetscInt    *bsizes;

  PetscFunctionBegin;
  PetscCall(MatGetVariableBlockSizes(pc->pmat, &nblocks, &bsizes));
  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(y, &yy));
  for (i = 0; i < nblocks; i++) {
    bs = bsizes[i];
    switch (bs) {
    case 1:
      yy[ncnt] = *diag * xx[ncnt];
      break;
    case 2:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      yy[ncnt]     = diag[0] * x0 + diag[1] * x1;
      yy[ncnt + 1] = diag[2] * x0 + diag[3] * x1;
      break;
    case 3:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      yy[ncnt]     = diag[0] * x0 + diag[1] * x1 + diag[2] * x2;
      yy[ncnt + 1] = diag[3] * x0 + diag[4] * x1 + diag[5] * x2;
      yy[ncnt + 2] = diag[6] * x0 + diag[7] * x1 + diag[8] * x2;
      break;
    case 4:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      x3           = xx[ncnt + 3];
      yy[ncnt]     = diag[0] * x0 + diag[1] * x1 + diag[2] * x2 + diag[3] * x3;
      yy[ncnt + 1] = diag[4] * x0 + diag[5] * x1 + diag[6] * x2 + diag[7] * x3;
      yy[ncnt + 2] = diag[8] * x0 + diag[9] * x1 + diag[10] * x2 + diag[11] * x3;
      yy[ncnt + 3] = diag[12] * x0 + diag[13] * x1 + diag[14] * x2 + diag[15] * x3;
      break;
    case 5:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      x3           = xx[ncnt + 3];
      x4           = xx[ncnt + 4];
      yy[ncnt]     = diag[0] * x0 + diag[1] * x1 + diag[2] * x2 + diag[3] * x3 + diag[4] * x4;
      yy[ncnt + 1] = diag[5] * x0 + diag[6] * x1 + diag[7] * x2 + diag[8] * x3 + diag[9] * x4;
      yy[ncnt + 2] = diag[10] * x0 + diag[11] * x1 + diag[12] * x2 + diag[13] * x3 + diag[14] * x4;
      yy[ncnt + 3] = diag[15] * x0 + diag[16] * x1 + diag[17] * x2 + diag[18] * x3 + diag[19] * x4;
      yy[ncnt + 4] = diag[20] * x0 + diag[21] * x1 + diag[22] * x2 + diag[23] * x3 + diag[24] * x4;
      break;
    case 6:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      x3           = xx[ncnt + 3];
      x4           = xx[ncnt + 4];
      x5           = xx[ncnt + 5];
      yy[ncnt]     = diag[0] * x0 + diag[1] * x1 + diag[2] * x2 + diag[3] * x3 + diag[4] * x4 + diag[5] * x5;
      yy[ncnt + 1] = diag[6] * x0 + diag[7] * x1 + diag[8] * x2 + diag[9] * x3 + diag[10] * x4 + diag[11] * x5;
      yy[ncnt + 2] = diag[12] * x0 + diag[13] * x1 + diag[14] * x2 + diag[15] * x3 + diag[16] * x4 + diag[17] * x5;
      yy[ncnt + 3] = diag[18] * x0 + diag[19] * x1 + diag[20] * x2 + diag[21] * x3 + diag[22] * x4 + diag[23] * x5;
      yy[ncnt + 4] = diag[24] * x0 + diag[25] * x1 + diag[26] * x2 + diag[27] * x3 + diag[28] * x4 + diag[29] * x5;
      yy[ncnt + 5] = diag[30] * x0 + diag[31] * x1 + diag[32] * x2 + diag[33] * x3 + diag[34] * x4 + diag[35] * x5;
      break;
    case 7:
      x0           = xx[ncnt];
      x1           = xx[ncnt + 1];
      x2           = xx[ncnt + 2];
      x3           = xx[ncnt + 3];
      x4           = xx[ncnt + 4];
      x5           = xx[ncnt + 5];
      x6           = xx[ncnt + 6];
      yy[ncnt]     = diag[0] * x0 + diag[1] * x1 + diag[2] * x2 + diag[3] * x3 + diag[4] * x4 + diag[5] * x5 + diag[6] * x6;
      yy[ncnt + 1] = diag[7] * x0 + diag[8] * x1 + diag[9] * x2 + diag[10] * x3 + diag[11] * x4 + diag[12] * x5 + diag[13] * x6;
      yy[ncnt + 2] = diag[14] * x0 + diag[15] * x1 + diag[16] * x2 + diag[17] * x3 + diag[18] * x4 + diag[19] * x5 + diag[20] * x6;
      yy[ncnt + 3] = diag[21] * x0 + diag[22] * x1 + diag[23] * x2 + diag[24] * x3 + diag[25] * x4 + diag[26] * x5 + diag[27] * x6;
      yy[ncnt + 4] = diag[28] * x0 + diag[29] * x1 + diag[30] * x2 + diag[31] * x3 + diag[32] * x4 + diag[33] * x5 + diag[34] * x6;
      yy[ncnt + 5] = diag[35] * x0 + diag[36] * x1 + diag[37] * x2 + diag[38] * x3 + diag[39] * x4 + diag[40] * x5 + diag[41] * x6;
      yy[ncnt + 6] = diag[42] * x0 + diag[43] * x1 + diag[44] * x2 + diag[45] * x3 + diag[46] * x4 + diag[47] * x5 + diag[48] * x6;
      break;
    default:
      for (ib = 0; ib < bs; ib++) {
        PetscScalar rowsum = 0;
        for (jb = 0; jb < bs; jb++) rowsum += diag[ib * bs + jb] * xx[ncnt + jb];
        yy[ncnt + ib] = rowsum;
      }
    }
    ncnt += bsizes[i];
    diag += bsizes[i] * bsizes[i];
  }
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(y, &yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_Host(PC pc)
{
  PC_VPBJacobi   *jac = (PC_VPBJacobi *)pc->data;
  Mat             A   = pc->pmat;
  MatFactorError  err;
  PetscInt        i, nsize = 0, nlocal;
  PetscInt        nblocks;
  const PetscInt *bsizes;

  PetscFunctionBegin;
  PetscCall(MatGetVariableBlockSizes(pc->pmat, &nblocks, &bsizes));
  PetscCall(MatGetLocalSize(pc->pmat, &nlocal, NULL));
  PetscCheck(!nlocal || nblocks, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call MatSetVariableBlockSizes() before using PCVPBJACOBI");
  if (!jac->diag) {
    PetscInt max_bs = -1, min_bs = PETSC_MAX_INT;
    for (i = 0; i < nblocks; i++) {
      min_bs = PetscMin(min_bs, bsizes[i]);
      max_bs = PetscMax(max_bs, bsizes[i]);
      nsize += bsizes[i] * bsizes[i];
    }
    PetscCall(PetscMalloc1(nsize, &jac->diag));
    jac->nblocks = nblocks;
    jac->min_bs  = min_bs;
    jac->max_bs  = max_bs;
  }
  PetscCall(MatInvertVariableBlockDiagonal(A, nblocks, bsizes, jac->diag));
  PetscCall(MatFactorGetError(A, &err));
  if (err) pc->failedreason = (PCFailedReason)err;
  pc->ops->apply          = PCApply_VPBJacobi;
  pc->ops->applytranspose = PCApplyTranspose_VPBJacobi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_VPBJacobi(PC pc)
{
  PetscFunctionBegin;
  /* In PCCreate_VPBJacobi() pmat might have not been set, so we wait to the last minute to do the dispatch */
#if defined(PETSC_HAVE_CUDA)
  PetscBool isCuda;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)pc->pmat, &isCuda, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, ""));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscBool isKok;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)pc->pmat, &isKok, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, ""));
#endif

#if defined(PETSC_HAVE_CUDA)
  if (isCuda) PetscCall(PCSetUp_VPBJacobi_CUDA(pc));
  else
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    if (isKok)
    PetscCall(PCSetUp_VPBJacobi_Kokkos(pc));
  else
#endif
  {
    PetscCall(PCSetUp_VPBJacobi_Host(pc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_VPBJacobi(PC pc, PetscViewer viewer)
{
  PC_VPBJacobi *jac = (PC_VPBJacobi *)pc->data;
  PetscBool     iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  number of blocks: %" PetscInt_FMT "\n", jac->nblocks));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  block sizes: min=%" PetscInt_FMT " max=%" PetscInt_FMT "\n", jac->min_bs, jac->max_bs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PCDestroy_VPBJacobi(PC pc)
{
  PC_VPBJacobi *jac = (PC_VPBJacobi *)pc->data;

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  PetscCall(PetscFree(jac->diag));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCVPBJACOBI - Variable size point block Jacobi preconditioner

   Level: beginner

   Notes:
     See `PCJACOBI` for point Jacobi preconditioning, `PCPBJACOBI` for fixed point block size, and `PCBJACOBI` for large size blocks

     This works for `MATAIJ` matrices

     Uses dense LU factorization with partial pivoting to invert the blocks; if a zero pivot
     is detected a PETSc error is generated.

     One must call `MatSetVariableBlockSizes()` to use this preconditioner

   Developer Notes:
     This should support the `PCSetErrorIfFailure()` flag set to `PETSC_TRUE` to allow
     the factorization to continue even after a zero pivot is found resulting in a Nan and hence
     terminating `KSP` with a `KSP_DIVERGED_NANORINF` allowing
     a nonlinear solver/ODE integrator to recover without stopping the program as currently happens.

     Perhaps should provide an option that allows generation of a valid preconditioner
     even if a block is singular as the `PCJACOBI` does.

.seealso: `MatSetVariableBlockSizes()`, `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCJACOBI`, `PCPBJACOBI`, `PCBJACOBI`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_VPBJacobi(PC pc)
{
  PC_VPBJacobi *jac;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  PetscCall(PetscNew(&jac));
  pc->data = (void *)jac;

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
  pc->ops->apply               = PCApply_VPBJacobi;
  pc->ops->applytranspose      = NULL;
  pc->ops->setup               = PCSetUp_VPBJacobi;
  pc->ops->destroy             = PCDestroy_VPBJacobi;
  pc->ops->setfromoptions      = NULL;
  pc->ops->view                = PCView_VPBJacobi;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
