static char help[] = "Tests MatComputeOperator() and MatComputeOperatorTranspose()\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,Ae,Aet;
  char           filename[PETSC_MAX_PATH_LEN];
  char           expltype[128],*etype = NULL;
  PetscInt       bs = 1;
  PetscBool      flg, check = PETSC_TRUE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetString(NULL,NULL,"-expl_type",expltype,sizeof(expltype),&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrallocpy(expltype,&etype);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  if (!flg) {
    PetscInt M = 13,N = 6;

    ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,&A);CHKERRQ(ierr);
    ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
    ierr = MatSetRandom(A,NULL);CHKERRQ(ierr);
  } else {
    PetscViewer viewer;

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatLoad(A,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)A,"Matrix");CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-view_expl");CHKERRQ(ierr);

  ierr = MatComputeOperator(A,etype,&Ae);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Ae,"Explicit matrix");CHKERRQ(ierr);
  ierr = MatViewFromOptions(Ae,NULL,"-view_expl");CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-check",&check,NULL);CHKERRQ(ierr);
  if (check) {
    Mat A2;
    PetscReal err,tol = PETSC_SMALL;

    ierr = PetscOptionsGetReal(NULL,NULL,"-tol",&tol,NULL);CHKERRQ(ierr);
    ierr = MatConvert(A,etype,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
    ierr = MatAXPY(A2,-1.0,Ae,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(A2,NORM_FROBENIUS,&err);CHKERRQ(ierr);
    if (err > tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error %g > %g (type %s)\n",(double)err,(double)tol,etype);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A2);CHKERRQ(ierr);
  }

  ierr = MatComputeOperatorTranspose(A,etype,&Aet);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Aet,"Explicit matrix transpose");CHKERRQ(ierr);
  ierr = MatViewFromOptions(Aet,NULL,"-view_expl");CHKERRQ(ierr);

  ierr = PetscFree(etype);CHKERRQ(ierr);
  ierr = MatDestroy(&Ae);CHKERRQ(ierr);
  ierr = MatDestroy(&Aet);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
