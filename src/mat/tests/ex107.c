
static char help[] = "Test MatCreate() with MAT_STRUCTURE_ONLY .\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            mat;
  PetscInt       m = 7,n,i,j,rstart,rend;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscScalar    v;
  PetscBool      struct_only=PETSC_TRUE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-struct_only",&struct_only,NULL));
  n    = m;

  /* ------- Assemble matrix, test MatValid() --------- */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&mat));
  CHKERRQ(MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(mat));
  if (struct_only) {
    CHKERRQ(MatSetOption(mat,MAT_STRUCTURE_ONLY,PETSC_TRUE));
  }
  CHKERRQ(MatSetUp(mat));
  CHKERRQ(MatGetOwnershipRange(mat,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10.0*i+j;
      CHKERRQ(MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  CHKERRQ(MatDestroy(&mat));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      output_file: output/ex107.out

   test:
      suffix: 2
      args: -mat_type baij -mat_block_size 2 -m 10

TEST*/
