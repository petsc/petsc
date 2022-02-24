/*
 * ex193.c
 *
 *  Created on: Jul 29, 2015
 *      Author: Fande Kong fdkong.jd@gmail.com
 */
/*
 * An example demonstrates how to use hierarchical partitioning approach
 */

#include <petscmat.h>

static char help[] = "Illustrates use of hierarchical partitioning.\n";

int main(int argc,char **args)
{
  Mat             A;                      /* matrix */
  PetscInt        m,n;                    /* mesh dimensions in x- and y- directions */
  PetscInt        i,j,Ii,J,Istart,Iend;
  PetscErrorCode  ierr;
  PetscMPIInt     size;
  PetscScalar     v;
  MatPartitioning part;
  IS              coarseparts,fineparts;
  IS              is,isn,isrows;
  MPI_Comm        comm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  ierr = PetscOptionsBegin(comm,NULL,"ex193","hierarchical partitioning");CHKERRQ(ierr);
  m = 15;
  CHKERRQ(PetscOptionsInt("-M","Number of mesh points in the x-direction","partitioning",m,&m,NULL));
  n = 15;
  CHKERRQ(PetscOptionsInt("-N","Number of mesh points in the y-direction","partitioning",n,&n,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /*
     Assemble the matrix for the five point stencil (finite difference), YET AGAIN
  */
  CHKERRQ(MatCreate(comm,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  /*
   Partition the graph of the matrix
  */
  CHKERRQ(MatPartitioningCreate(comm,&part));
  CHKERRQ(MatPartitioningSetAdjacency(part,A));
  CHKERRQ(MatPartitioningSetType(part,MATPARTITIONINGHIERARCH));
  CHKERRQ(MatPartitioningHierarchicalSetNcoarseparts(part,2));
  CHKERRQ(MatPartitioningHierarchicalSetNfineparts(part,4));
  CHKERRQ(MatPartitioningSetFromOptions(part));
  /* get new processor owner number of each vertex */
  CHKERRQ(MatPartitioningApply(part,&is));
  /* coarse parts */
  CHKERRQ(MatPartitioningHierarchicalGetCoarseparts(part,&coarseparts));
  CHKERRQ(ISView(coarseparts,PETSC_VIEWER_STDOUT_WORLD));
  /* fine parts */
  CHKERRQ(MatPartitioningHierarchicalGetFineparts(part,&fineparts));
  CHKERRQ(ISView(fineparts,PETSC_VIEWER_STDOUT_WORLD));
  /* partitioning */
  CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  /* get new global number of each old global number */
  CHKERRQ(ISPartitioningToNumbering(is,&isn));
  CHKERRQ(ISView(isn,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISBuildTwoSided(is,NULL,&isrows));
  CHKERRQ(ISView(isrows,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISDestroy(&coarseparts));
  CHKERRQ(ISDestroy(&fineparts));
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(ISDestroy(&isn));
  CHKERRQ(MatPartitioningDestroy(&part));
  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4
      args: -mat_partitioning_hierarchical_Nfineparts 2
      requires: parmetis
      TODO: cannot run because parmetis does reproduce across all machines, probably due to nonportable random number generator

TEST*/
