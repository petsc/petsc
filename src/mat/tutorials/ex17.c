static char help[] = "Example of using graph partitioning with a matrix in which some procs have empty ownership\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             A;
  MatPartitioning part;
  IS              is;
  PetscInt        i,m,N,rstart,rend,nemptyranks,*emptyranks,nbigranks,*bigranks;
  PetscMPIInt     rank,size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  nemptyranks = 10;
  nbigranks   = 10;
  PetscCall(PetscMalloc2(nemptyranks,&emptyranks,nbigranks,&bigranks));

  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Partitioning example options",NULL);
  PetscCall(PetscOptionsIntArray("-emptyranks","Ranks to be skipped by partition","",emptyranks,&nemptyranks,NULL));
  PetscCall(PetscOptionsIntArray("-bigranks","Ranks to be overloaded","",bigranks,&nbigranks,NULL));
  PetscOptionsEnd();

  m = 1;
  for (i=0; i<nemptyranks; i++) if (rank == emptyranks[i]) m = 0;
  for (i=0; i<nbigranks; i++) if (rank == bigranks[i]) m = 5;

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A,3,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,3,NULL,2,NULL));
  PetscCall(MatSeqBAIJSetPreallocation(A,1,3,NULL));
  PetscCall(MatMPIBAIJSetPreallocation(A,1,3,NULL,2,NULL));
  PetscCall(MatSeqSBAIJSetPreallocation(A,1,2,NULL));
  PetscCall(MatMPISBAIJSetPreallocation(A,1,2,NULL,1,NULL));

  PetscCall(MatGetSize(A,NULL,&N));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    const PetscInt    cols[] = {(i+N-1)%N,i,(i+1)%N};
    const PetscScalar vals[] = {1,1,1};
    PetscCall(MatSetValues(A,1,&i,3,cols,vals,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD,&part));
  PetscCall(MatPartitioningSetAdjacency(part,A));
  PetscCall(MatPartitioningSetFromOptions(part));
  PetscCall(MatPartitioningApply(part,&is));
  PetscCall(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&is));
  PetscCall(MatPartitioningDestroy(&part));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree2(emptyranks,bigranks));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 8
      args: -emptyranks 0,2,4 -bigranks 1,3,7 -mat_partitioning_type average
      # cannot test with external package partitioners since they produce different results on different systems

TEST*/
