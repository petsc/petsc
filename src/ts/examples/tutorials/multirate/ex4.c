static char help[] = "test IS setting.\n";

#include <petscts.h>

int main(int argc,char **argv)
{
  TS    ts;
  Vec   U;
  IS    iss;
  IS    isf;
  PetscScalar *pU;
  Vec Ys;
  Vec Yf;
  PetscInt *indicess;
  PetscInt *indicesf;
  PetscInt n = 5;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if(ierr) return ierr;

  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,n,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(U);CHKERRQ(ierr);
  ierr = VecGetArray(U,&pU);CHKERRQ(ierr);
  pU[0] = 1.0;
  pU[1] = 1.1;
  pU[2] = 1.2;
  pU[3] = 1.3;
  pU[4] = 1.4;
  ierr = VecRestoreArray(U,&pU);CHKERRQ(ierr);

  ierr = PetscMalloc1(2,&indicess);CHKERRQ(ierr);
  indicess[0]=0;
  indicess[1]=1;
  ierr = PetscMalloc1(3,&indicesf);CHKERRQ(ierr);
  indicesf[0]=2;
  indicesf[1]=3;
  indicesf[2]=4;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,indicess,PETSC_COPY_VALUES,&iss);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,3,indicesf,PETSC_COPY_VALUES,&isf);CHKERRQ(ierr);

  
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetIS(ts,iss,isf);
  ierr = TSGetIS(ts,*iss,*isf);

  ierr = VecGetSubVector(U,iss,&Ys);CHKERRQ(ierr);
  ierr = VecGetSubVector(U,isf,&Yf);CHKERRQ(ierr);
  ierr = VecView(Ys,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(Yf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&Ys);CHKERRQ(ierr);
  ierr = VecDestroy(&Yf);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = ISDestroy(&iss);CHKERRQ(ierr);
  ierr = ISDestroy(&isf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
