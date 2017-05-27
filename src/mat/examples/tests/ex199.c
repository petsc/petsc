
static char help[] = "Tests the different MatColoring implementatons.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscErrorCode ierr;
  PetscViewer    viewer;
  char           file[128];
  PetscBool      flg;
  MatColoring    ctx;
  ISColoring     coloring;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must use -f filename to load sparse matrix");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);  
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatLoad(C,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatColoringCreate(C,&ctx);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(ctx);CHKERRQ(ierr);
  ierr = MatColoringApply(ctx,&coloring);CHKERRQ(ierr);
  ierr = MatColoringTestValid(ctx,coloring);CHKERRQ(ierr);

  /* Free data structures */
  ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


