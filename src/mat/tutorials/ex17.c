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

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  nemptyranks = 10;
  nbigranks   = 10;
  ierr        = PetscMalloc2(nemptyranks,&emptyranks,nbigranks,&bigranks);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Partitioning example options",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-emptyranks","Ranks to be skipped by partition","",emptyranks,&nemptyranks,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-bigranks","Ranks to be overloaded","",bigranks,&nbigranks,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  m = 1;
  for (i=0; i<nemptyranks; i++) if (rank == emptyranks[i]) m = 0;
  for (i=0; i<nbigranks; i++) if (rank == bigranks[i]) m = 5;

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,3,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A,1,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,1,3,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,1,2,NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,1,2,NULL,1,NULL);CHKERRQ(ierr);

  ierr = MatGetSize(A,NULL,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    const PetscInt    cols[] = {(i+N-1)%N,i,(i+1)%N};
    const PetscScalar vals[] = {1,1,1};
    ierr = MatSetValues(A,1,&i,3,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatPartitioningCreate(PETSC_COMM_WORLD,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
  ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree2(emptyranks,bigranks);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 8
      args: -emptyranks 0,2,4 -bigranks 1,3,7 -mat_partitioning_type average
      # cannot test with external package partitioners since they produce different results on different systems

TEST*/
