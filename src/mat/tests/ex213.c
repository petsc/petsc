
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
  PetscMPIInt    rank,size;
  Vec            v;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size > 4,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Can only use at most 4 processors.");
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Get a partition range based on the vector size */
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &v));
  CHKERRQ(VecGetLocalSize(v, &n));
  CHKERRQ(VecGetOwnershipRange(v, &rstart, &rend));
  CHKERRQ(VecDestroy(&v));

  CHKERRQ(PetscMalloc1(n+1,&ia));
  CHKERRQ(PetscMalloc1(3*n,&ja));

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

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  CHKERRQ(MatSetType(A,MATMPIAIJ));
  CHKERRQ(MatMPIAIJSetPreallocationCSR(A, ia, ja, NULL));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(MatSetSizes(A, bs*n, bs*n, PETSC_DETERMINE, PETSC_DETERMINE));
  CHKERRQ(MatSetType(A,MATMPIBAIJ));
  CHKERRQ(MatMPIBAIJSetPreallocationCSR(A, bs, ia, ja, NULL));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(PetscFree(ia));
  CHKERRQ(PetscFree(ja));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: 4

TEST*/
