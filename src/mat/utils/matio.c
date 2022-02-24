#include <petscviewer.h>
#include <petsc/private/matimpl.h>

PetscErrorCode MatView_Binary_BlockSizes(Mat mat,PetscViewer viewer)
{
  FILE           *info;
  PetscMPIInt    rank;
  PetscInt       rbs,cbs;

  PetscFunctionBegin;
  CHKERRQ(MatGetBlockSizes(mat,&rbs,&cbs));
  CHKERRQ(PetscViewerBinaryGetInfoPointer(viewer,&info));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank));
  if (rank == 0 && info) {
    if (rbs != cbs) CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,info,"-matload_block_size %" PetscInt_FMT ",%" PetscInt_FMT "\n",rbs,cbs));
    else            CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,info,"-matload_block_size %" PetscInt_FMT "\n",rbs));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_Binary_BlockSizes(Mat mat,PetscViewer viewer)
{
  PetscInt       rbs,cbs,bs[2],n = 2;
  PetscBool      set;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get current block sizes */
  CHKERRQ(MatGetBlockSizes(mat,&rbs,&cbs));
  bs[0] = rbs; bs[1] = cbs;
  /* get block sizes from the options database */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)viewer),NULL,"Options for loading matrix block size","Mat");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsIntArray("-matload_block_size","Set the block size used to store the matrix","MatLoad",bs,&n,&set));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!set) PetscFunctionReturn(0);
  if (n == 1) bs[1] = bs[0]; /* to support -matload_block_size <bs> */
  /* set matrix block sizes */
  if (bs[0] > 0) rbs = bs[0];
  if (bs[1] > 0) cbs = bs[1];
  CHKERRQ(MatSetBlockSizes(mat,rbs,cbs));
  PetscFunctionReturn(0);
}
