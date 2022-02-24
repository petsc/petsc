static char help[] = "Tests MatMPI{AIJ,BAIJ,SBAIJ}SetPreallocationCSR\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       ia[3]={0,2,4};
  PetscInt       ja[4]={0,1,0,1};
  PetscScalar    c[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  PetscInt       ia2[5]={0,4,8,12,16};
  PetscInt       ja2[16]={0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
  PetscScalar    c2[16]={0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15};
  PetscMPIInt    size,rank;
  Mat            ssbaij;
  PetscBool      rect = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheckFalse(size < 2,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is an example with more then one processors");
  if (rank) {
    PetscInt i;
    for (i = 0; i < 3; i++) ia[i] = 0;
    for (i = 0; i < 5; i++) ia2[i] = 0;
  }
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-rect",&rect,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&ssbaij));
  CHKERRQ(MatSetBlockSize(ssbaij,2));
  if (rect) {
    CHKERRQ(MatSetType(ssbaij,MATMPIBAIJ));
    CHKERRQ(MatSetSizes(ssbaij,4,6,PETSC_DECIDE,PETSC_DECIDE));
  } else {
    CHKERRQ(MatSetType(ssbaij,MATMPISBAIJ));
    CHKERRQ(MatSetSizes(ssbaij,4,4,PETSC_DECIDE,PETSC_DECIDE));
  }
  CHKERRQ(MatSetFromOptions(ssbaij));
  CHKERRQ(MatMPIAIJSetPreallocationCSR(ssbaij,ia2,ja2,c2));
  CHKERRQ(MatMPIBAIJSetPreallocationCSR(ssbaij,2,ia,ja,c));
  CHKERRQ(MatMPISBAIJSetPreallocationCSR(ssbaij,2,ia,ja,c));
  CHKERRQ(MatViewFromOptions(ssbaij,NULL,"-view"));
  CHKERRQ(MatDestroy(&ssbaij));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    filter: grep -v type | sed -e "s/\.//g"
    suffix: aijbaij_csr
    nsize: 2
    args: -mat_type {{aij baij}} -view -rect {{0 1}}

  test:
    filter: sed -e "s/\.//g"
    suffix: sbaij_csr
    nsize: 2
    args: -mat_type sbaij -view -rect {{0 1}}

TEST*/
