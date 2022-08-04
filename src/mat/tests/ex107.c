
static char help[] = "Test MatCreate() with MAT_STRUCTURE_ONLY .\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            mat;
  PetscInt       m = 7,n,i,j,rstart,rend;
  PetscMPIInt    size;
  PetscScalar    v;
  PetscBool      struct_only=PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-struct_only",&struct_only,NULL));
  n    = m;

  /* ------- Assemble matrix, test MatValid() --------- */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&mat));
  PetscCall(MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetFromOptions(mat));
  if (struct_only) PetscCall(MatSetOption(mat,MAT_STRUCTURE_ONLY,PETSC_TRUE));
  PetscCall(MatSetUp(mat));
  PetscCall(MatGetOwnershipRange(mat,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10.0*i+j;
      PetscCall(MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex107.out

   test:
      suffix: 2
      args: -mat_type baij -mat_block_size 2 -m 10

TEST*/
