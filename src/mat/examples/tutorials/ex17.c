static char help[] = "Example of using graph partitioning with a matrix in which some procs have empty ownership\n\n";

/*T
   Concepts: Mat^mat partitioning
   Concepts: Mat^image segmentation
   Processors: n
T*/

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **args)
{
  Mat             A;
  MatPartitioning part;
  IS              is;
  PetscInt        i,m,N,rstart,rend,nemptyranks,*emptyranks,nbigranks,*bigranks;
  PetscMPIInt     rank,size;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  nemptyranks = 10;
  nbigranks = 10;
  ierr = PetscMalloc2(nemptyranks,PetscInt,&emptyranks,nbigranks,PetscInt,&bigranks);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Partitioning example options",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-emptyranks","Ranks to be skipped by partition","",emptyranks,&nemptyranks,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-bigranks","Ranks to be overloaded","",bigranks,&nbigranks,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  m = 1;
  for (i=0; i<nemptyranks; i++) if (rank == emptyranks[i]) m = 0;
  for (i=0; i<nbigranks; i++) if (rank == bigranks[i]) m = 5;

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,3,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,3,PETSC_NULL,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A,1,3,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,1,3,PETSC_NULL,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,1,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,1,2,PETSC_NULL,1,PETSC_NULL);CHKERRQ(ierr);

  ierr = MatGetSize(A,PETSC_NULL,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    const PetscInt cols[] = {(i+N-1)%N,i,(i+1)%N};
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
  return 0;
}
