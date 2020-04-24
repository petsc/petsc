static char help[] = "Compare parallel partitioning strategies using matrix graphs\n\n";

#include <petscmat.h>



int main(int argc, char **args)
{
  MatPartitioning part;
  IS              partis;
  Mat             A        = NULL;
  PetscInt        max      = -1;
  PetscInt        min      = -1;
  PetscReal       balance  = 0.0;
  const PetscInt *ranges  = NULL;
  char            filein[PETSC_MAX_PATH_LEN];
  MPI_Comm        comm;
  PetscMPIInt     size;
  PetscInt        p;
  PetscBool       flg;
  PetscErrorCode  ierr;

  /*load matrix*/
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-fin",filein,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer view;
    ierr = PetscViewerBinaryOpen(comm,filein,FILE_MODE_READ,&view);CHKERRQ(ierr);
    ierr = MatCreate(comm,&A);CHKERRQ(ierr);
    ierr = MatLoad(A,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  }

  /*partition matrix*/
  ierr = MatPartitioningCreate(comm,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part, A);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  ierr = MatPartitioningApply(part, &partis);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(A, &ranges);CHKERRQ(ierr);
  ierr = MatGetSize(A, &min, NULL);CHKERRQ(ierr);
  for (p = 0; p < size; ++p) {
    const PetscInt partsize = ranges[p+1]-ranges[p];

    max = PetscMax(max, partsize);
    min = PetscMin(min, partsize);
  }
  balance = ((PetscReal) max)/min;
  ierr = PetscPrintf(comm, "ranges: ");CHKERRQ(ierr);
  for (p = 0; p <= size; ++p) {
    if (p > 0) {ierr = PetscPrintf(comm, ", ");CHKERRQ(ierr);}
    ierr = PetscPrintf(comm, "%D", ranges[p]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "max:%.0lf min:%.0lf balance:%.11lf\n", (double) max,(double) min,(double) balance);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)partis,NULL,"-partition_view");CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  ierr = ISDestroy(&partis);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;

}
