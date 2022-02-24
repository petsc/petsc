
static char help[] = "Tests MatMPIBAIJ format in sequential run \n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat            A,B;
  PetscInt       i,rstart,rend;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a MPIBAIJ matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,32,32));
  CHKERRQ(MatSetType(A,MATMPIBAIJ));
  CHKERRQ(MatSeqBAIJSetPreallocation(A,2,2,NULL));
  CHKERRQ(MatMPIBAIJSetPreallocation(A,2,2,NULL,2,NULL));

  v    = 1.0;
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Convert A to AIJ format */
  CHKERRQ(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     output_file: output/ex160.out

TEST*/
