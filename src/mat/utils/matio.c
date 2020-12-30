#include <petscviewer.h>
#include <petsc/private/matimpl.h>

PetscErrorCode MatView_Binary_BlockSizes(Mat mat,PetscViewer viewer)
{
  FILE           *info;
  PetscMPIInt    rank;
  PetscInt       rbs,cbs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetBlockSizes(mat,&rbs,&cbs);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetInfoPointer(viewer,&info);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRMPI(ierr);
  if (!rank && info) {
    if (rbs != cbs) {ierr = PetscFPrintf(PETSC_COMM_SELF,info,"-matload_block_size %D,%D\n",rbs,cbs);CHKERRQ(ierr);}
    else            {ierr = PetscFPrintf(PETSC_COMM_SELF,info,"-matload_block_size %D\n",rbs);CHKERRQ(ierr);}
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
  ierr = MatGetBlockSizes(mat,&rbs,&cbs);CHKERRQ(ierr);
  bs[0] = rbs; bs[1] = cbs;
  /* get block sizes from the options database */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)viewer),NULL,"Options for loading matrix block size","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-matload_block_size","Set the block size used to store the matrix","MatLoad",bs,&n,&set);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!set) PetscFunctionReturn(0);
  if (n == 1) bs[1] = bs[0]; /* to support -matload_block_size <bs> */
  /* set matrix block sizes */
  if (bs[0] > 0) rbs = bs[0];
  if (bs[1] > 0) cbs = bs[1];
  ierr = MatSetBlockSizes(mat,rbs,cbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
