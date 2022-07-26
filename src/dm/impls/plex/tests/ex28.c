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
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-fin",filein,sizeof(filein),&flg));
  if (flg) {
    PetscViewer view;
    PetscCall(PetscViewerBinaryOpen(comm,filein,FILE_MODE_READ,&view));
    PetscCall(MatCreate(comm,&A));
    PetscCall(MatLoad(A,view));
    PetscCall(PetscViewerDestroy(&view));
  }

  /*partition matrix*/
  PetscCall(MatPartitioningCreate(comm,&part));
  PetscCall(MatPartitioningSetAdjacency(part, A));
  PetscCall(MatPartitioningSetFromOptions(part));
  PetscCall(MatPartitioningApply(part, &partis));
  PetscCall(MatGetOwnershipRanges(A, &ranges));
  PetscCall(MatGetSize(A, &min, NULL));
  for (p = 0; p < size; ++p) {
    const PetscInt partsize = ranges[p+1]-ranges[p];

    max = PetscMax(max, partsize);
    min = PetscMin(min, partsize);
  }
  balance = ((PetscReal) max)/min;
  PetscCall(PetscPrintf(comm, "ranges: "));
  for (p = 0; p <= size; ++p) {
    if (p > 0) PetscCall(PetscPrintf(comm, ", "));
    PetscCall(PetscPrintf(comm, "%" PetscInt_FMT, ranges[p]));
  }
  PetscCall(PetscPrintf(comm, "\n"));
  PetscCall(PetscPrintf(comm, "max:%.0lf min:%.0lf balance:%.11lf\n", (double) max,(double) min,(double) balance));
  PetscCall(PetscObjectViewFromOptions((PetscObject)partis,NULL,"-partition_view"));
  PetscCall(MatPartitioningDestroy(&part));
  PetscCall(ISDestroy(&partis));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;

}
