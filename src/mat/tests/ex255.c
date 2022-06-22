static char help[] = "Tests MatConvert from AIJ to MATIS with a block size greater than 1.\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat            A,B;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      flg,equal;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

  /* Load an AIJ matrix */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,fd));

  /* Convert it to MATIS */
  PetscCall(MatConvert(A,MATIS,MAT_INITIAL_MATRIX,&B));

  /* Check they are equal */
  PetscCall(MatEqual(A,B,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A and B are not equal");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(PetscFinalize());
}

/*TEST
   test:
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -mat_type aij -matload_block_size {{1 2}} -f ${DATAFILESPATH}/matrices/smallbs2
     output_file: output/empty.out

TEST*/

