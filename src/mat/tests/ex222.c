static char help[] = "Tests MatComputeOperator() and MatComputeOperatorTranspose()\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,Ae,Aet;
  char           filename[PETSC_MAX_PATH_LEN];
  char           expltype[128],*etype = NULL;
  PetscInt       bs = 1;
  PetscBool      flg, check = PETSC_TRUE;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*) 0,help));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-expl_type",expltype,sizeof(expltype),&flg));
  if (flg) {
    CHKERRQ(PetscStrallocpy(expltype,&etype));
  }
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  if (!flg) {
    PetscInt M = 13,N = 6;

    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,&A));
    CHKERRQ(MatSetBlockSize(A,bs));
    CHKERRQ(MatSetRandom(A,NULL));
  } else {
    PetscViewer viewer;

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRQ(MatSetBlockSize(A,bs));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatLoad(A,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(PetscObjectSetName((PetscObject)A,"Matrix"));
  CHKERRQ(MatViewFromOptions(A,NULL,"-view_expl"));

  CHKERRQ(MatComputeOperator(A,etype,&Ae));
  CHKERRQ(PetscObjectSetName((PetscObject)Ae,"Explicit matrix"));
  CHKERRQ(MatViewFromOptions(Ae,NULL,"-view_expl"));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-check",&check,NULL));
  if (check) {
    Mat A2;
    PetscReal err,tol = PETSC_SMALL;

    CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tol",&tol,NULL));
    CHKERRQ(MatConvert(A,etype,MAT_INITIAL_MATRIX,&A2));
    CHKERRQ(MatAXPY(A2,-1.0,Ae,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatNorm(A2,NORM_FROBENIUS,&err));
    if (err > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error %g > %g (type %s)\n",(double)err,(double)tol,etype));
    }
    CHKERRQ(MatDestroy(&A2));
  }

  CHKERRQ(MatComputeOperatorTranspose(A,etype,&Aet));
  CHKERRQ(PetscObjectSetName((PetscObject)Aet,"Explicit matrix transpose"));
  CHKERRQ(MatViewFromOptions(Aet,NULL,"-view_expl"));

  CHKERRQ(PetscFree(etype));
  CHKERRQ(MatDestroy(&Ae));
  CHKERRQ(MatDestroy(&Aet));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     output_file: output/ex222_null.out

   testset:
     suffix: matexpl_rect
     output_file: output/ex222_null.out
     nsize: {{1 3}}
     args: -expl_type {{dense aij baij}}

   testset:
     suffix: matexpl_square
     output_file: output/ex222_null.out
     nsize: {{1 3}}
     args: -bs {{1 2 3}} -M 36 -N 36 -expl_type {{dense aij baij sbaij}}

TEST*/
