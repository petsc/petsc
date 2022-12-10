#include <petscviewer.h>
#include <petsc/private/matimpl.h>

PetscErrorCode MatView_Binary_BlockSizes(Mat mat, PetscViewer viewer)
{
  FILE       *info;
  PetscMPIInt rank;
  PetscInt    rbs, cbs;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSizes(mat, &rbs, &cbs));
  PetscCall(PetscViewerBinaryGetInfoPointer(viewer, &info));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  if (rank == 0 && info) {
    if (rbs != cbs) PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "-matload_block_size %" PetscInt_FMT ",%" PetscInt_FMT "\n", rbs, cbs));
    else PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "-matload_block_size %" PetscInt_FMT "\n", rbs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatLoad_Binary_BlockSizes(Mat mat, PetscViewer viewer)
{
  PetscInt  rbs, cbs, bs[2], n = 2;
  PetscBool set;

  PetscFunctionBegin;
  /* get current block sizes */
  PetscCall(MatGetBlockSizes(mat, &rbs, &cbs));
  bs[0] = rbs;
  bs[1] = cbs;
  /* get block sizes from the options database */
  PetscOptionsBegin(PetscObjectComm((PetscObject)viewer), NULL, "Options for loading matrix block size", "Mat");
  PetscCall(PetscOptionsIntArray("-matload_block_size", "Set the block size used to store the matrix", "MatLoad", bs, &n, &set));
  PetscOptionsEnd();
  if (!set) PetscFunctionReturn(PETSC_SUCCESS);
  if (n == 1) bs[1] = bs[0]; /* to support -matload_block_size <bs> */
  /* set matrix block sizes */
  if (bs[0] > 0) rbs = bs[0];
  if (bs[1] > 0) cbs = bs[1];
  PetscCall(MatSetBlockSizes(mat, rbs, cbs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
