static char help[] = "Example of using graph partitioning with a matrix in which some procs have empty ownership\n\n";

/*T
   Concepts: Mat^mat partitioning
   Concepts: Mat^image segmentation
   Processors: n
T*/

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             A;
  MatPartitioning part;
  IS              is;
  PetscInt        i,m,N,rstart,rend,nemptyranks,*emptyranks,nbigranks,*bigranks;
  PetscMPIInt     rank,size;
  PetscErrorCode  ierr;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  nemptyranks = 10;
  nbigranks   = 10;
  CHKERRQ(PetscMalloc2(nemptyranks,&emptyranks,nbigranks,&bigranks));

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Partitioning example options",NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsIntArray("-emptyranks","Ranks to be skipped by partition","",emptyranks,&nemptyranks,NULL));
  CHKERRQ(PetscOptionsIntArray("-bigranks","Ranks to be overloaded","",bigranks,&nbigranks,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  m = 1;
  for (i=0; i<nemptyranks; i++) if (rank == emptyranks[i]) m = 0;
  for (i=0; i<nbigranks; i++) if (rank == bigranks[i]) m = 5;

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqAIJSetPreallocation(A,3,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A,3,NULL,2,NULL));
  CHKERRQ(MatSeqBAIJSetPreallocation(A,1,3,NULL));
  CHKERRQ(MatMPIBAIJSetPreallocation(A,1,3,NULL,2,NULL));
  CHKERRQ(MatSeqSBAIJSetPreallocation(A,1,2,NULL));
  CHKERRQ(MatMPISBAIJSetPreallocation(A,1,2,NULL,1,NULL));

  CHKERRQ(MatGetSize(A,NULL,&N));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    const PetscInt    cols[] = {(i+N-1)%N,i,(i+1)%N};
    const PetscScalar vals[] = {1,1,1};
    CHKERRQ(MatSetValues(A,1,&i,3,cols,vals,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatPartitioningCreate(PETSC_COMM_WORLD,&part));
  CHKERRQ(MatPartitioningSetAdjacency(part,A));
  CHKERRQ(MatPartitioningSetFromOptions(part));
  CHKERRQ(MatPartitioningApply(part,&is));
  CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(MatPartitioningDestroy(&part));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFree2(emptyranks,bigranks));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 8
      args: -emptyranks 0,2,4 -bigranks 1,3,7 -mat_partitioning_type average
      # cannot test with external package partitioners since they produce different results on different systems

TEST*/
