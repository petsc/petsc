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
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm,NULL,"ex193","hierarchical partitioning");CHKERRQ(ierr);
  m = 15;
  ierr = PetscOptionsInt("-M","Number of mesh points in the x-direction","partitioning",m,&m,NULL);CHKERRQ(ierr);
  n = 15;
  ierr = PetscOptionsInt("-N","Number of mesh points in the y-direction","partitioning",n,&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /*
     Assemble the matrix for the five point stencil (finite difference), YET AGAIN
  */
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /*
   Partition the graph of the matrix
  */
  ierr = MatPartitioningCreate(comm,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(part,MATPARTITIONINGHIERARCH);CHKERRQ(ierr);
  ierr = MatPartitioningHierarchicalSetNcoarseparts(part,2);CHKERRQ(ierr);
  ierr = MatPartitioningHierarchicalSetNfineparts(part,4);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  /* get new processor owner number of each vertex */
  ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
  /* coarse parts */
  ierr = MatPartitioningHierarchicalGetCoarseparts(part,&coarseparts);CHKERRQ(ierr);
  ierr = ISView(coarseparts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* fine parts */
  ierr = MatPartitioningHierarchicalGetFineparts(part,&fineparts);CHKERRQ(ierr);
  ierr = ISView(fineparts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* partitioning */
  ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* get new global number of each old global number */
  ierr = ISPartitioningToNumbering(is,&isn);CHKERRQ(ierr);
  ierr = ISView(isn,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISBuildTwoSided(is,NULL,&isrows);CHKERRQ(ierr);
  ierr = ISView(isrows,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&coarseparts);CHKERRQ(ierr);
  ierr = ISDestroy(&fineparts);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&isn);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
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
