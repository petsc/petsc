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

  /*load matrix*/
  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fin",filein,sizeof(filein),&flg));
  if (flg) {
    PetscViewer view;
    CHKERRQ(PetscViewerBinaryOpen(comm,filein,FILE_MODE_READ,&view));
    CHKERRQ(MatCreate(comm,&A));
    CHKERRQ(MatLoad(A,view));
    CHKERRQ(PetscViewerDestroy(&view));
  }

  /*partition matrix*/
  CHKERRQ(MatPartitioningCreate(comm,&part));
  CHKERRQ(MatPartitioningSetAdjacency(part, A));
  CHKERRQ(MatPartitioningSetFromOptions(part));
  CHKERRQ(MatPartitioningApply(part, &partis));
  CHKERRQ(MatGetOwnershipRanges(A, &ranges));
  CHKERRQ(MatGetSize(A, &min, NULL));
  for (p = 0; p < size; ++p) {
    const PetscInt partsize = ranges[p+1]-ranges[p];

    max = PetscMax(max, partsize);
    min = PetscMin(min, partsize);
  }
  balance = ((PetscReal) max)/min;
  CHKERRQ(PetscPrintf(comm, "ranges: "));
  for (p = 0; p <= size; ++p) {
    if (p > 0) CHKERRQ(PetscPrintf(comm, ", "));
    CHKERRQ(PetscPrintf(comm, "%D", ranges[p]));
  }
  CHKERRQ(PetscPrintf(comm, "\n"));
  CHKERRQ(PetscPrintf(comm, "max:%.0lf min:%.0lf balance:%.11lf\n", (double) max,(double) min,(double) balance));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)partis,NULL,"-partition_view"));
  CHKERRQ(MatPartitioningDestroy(&part));
  CHKERRQ(ISDestroy(&partis));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;

}
