static char help[] = "Generates random numbers for radioactive events for ex61.c.\n ./ex61gen [-random_seed <int>] \n";



#include <petscsys.h>
#include <petscvec.h>


int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscRandom    rand;
  PetscScalar    *values;
  PetscInt       n = 10000,seed;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv, (char*)0, help);if (ierr) return ierr;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,4*n,&x);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rand);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = VecGetArray(x,&values);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-random_seed",&seed,&flg);CHKERRQ(ierr);
  if (flg) {
    sprintf(filename,"ex61.random.%d",(int)seed);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcpy(filename,"ex61.random");CHKERRQ(ierr);
  }
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&n,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,values,4*n,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&values);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
