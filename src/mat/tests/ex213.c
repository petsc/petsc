
static char help[] = "Tests MatMPIBAIJSetPreallocationCSR()\n\n";

/*T
   Concepts: partitioning
   Processors: 4
T*/

/*
  Include "petscmat.h" so that we can use matrices.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       *ia,*ja, bs = 2;
  PetscInt       N = 9, n;
  PetscInt       rstart, rend, row, col;
  PetscInt       i;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  Vec            v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Can only use at most 4 processors.");
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Get a partition range based on the vector size */
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &v);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(v, &rstart, &rend);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);

  ierr = PetscMalloc1(n+1,&ia);CHKERRQ(ierr);
  ierr = PetscMalloc1(3*n,&ja);CHKERRQ(ierr);

  /* Construct a tri-diagonal CSR indexing */
  i = 1;
  ia[0] = 0;
  for (row = rstart; row < rend; row++)
  {
    ia[i] = ia[i-1];

    /* diagonal */
    col = row;
    {
      ja[ia[i]] = col;
      ia[i]++;
    }

    /* lower diagonal */
    col = row-1;
    if (col >= 0)
    {
      ja[ia[i]] = col;
      ia[i]++;
    }

    /* upper diagonal */
    col = row+1;
    if (col < N)
    {
      ja[ia[i]] = col;
      ia[i]++;
    }
    i++;
  }

  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocationCSR(A, ia, ja, NULL);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, bs*n, bs*n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocationCSR(A, bs, ia, ja, NULL);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFree(ia);CHKERRQ(ierr);
  ierr = PetscFree(ja);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      nsize: 4

TEST*/

