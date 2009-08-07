static char help[] = "Solves -Laplacian u - exp(u) = 0,  0 < x < 1\n\n";
#include "petscda.h"
#include "petscsnes.h"
extern PetscErrorCode ComputeFunction(SNES,Vec,Vec,void*), ComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

int main(int argc,char **argv) {
  SNES snes; Vec x,f; Mat J; DA da;
  PetscInitialize(&argc,&argv,(char *)0,help);

  DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,8,1,1,PETSC_NULL,&da);
  DACreateGlobalVector(da,&x); VecDuplicate(x,&f);
  DAGetMatrix(da,MATAIJ,&J);

  SNESCreate(PETSC_COMM_WORLD,&snes);
  SNESSetFunction(snes,f,ComputeFunction,da);
  SNESSetJacobian(snes,J,J,ComputeJacobian,da);
  SNESSetFromOptions(snes);
  SNESSolve(snes,PETSC_NULL,x);

  MatDestroy(J); VecDestroy(x); VecDestroy(f); SNESDestroy(snes); DADestroy(da);
  PetscFinalize();
  return 0;}
PetscErrorCode ComputeFunction(SNES snes,Vec x,Vec f,void *ctx) {
  PetscInt i,Mx,xs,xm; PetscScalar *xx,*ff,hx; DA da = (DA) ctx; Vec xlocal;
  DAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  hx     = 1.0/(PetscReal)(Mx-1);
  DAGetLocalVector(da,&xlocal);DAGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal);  DAGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal);
  DAVecGetArray(da,xlocal,&xx); DAVecGetArray(da,f,&ff);
  DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);

  for (i=xs; i<xs+xm; i++) {
    if (i == 0 || i == Mx-1) ff[i] = xx[i]/hx; 
    else  ff[i] =  (2.0*xx[i] - xx[i-1] - xx[i+1])/hx - hx*PetscExpScalar(xx[i]); 
  }
  DAVecRestoreArray(da,xlocal,&xx); DARestoreLocalVector(da,&xlocal);DAVecRestoreArray(da,f,&ff);
  return 0;}
PetscErrorCode ComputeJacobian(SNES snes,Vec x,Mat *J,Mat *B,MatStructure *flag,void *ctx){
  DA da = (DA) ctx; PetscInt i,Mx,xm,xs; PetscScalar hx,*xx; Vec xlocal;
  DAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  hx = 1.0/(PetscReal)(Mx-1);
  DAGetLocalVector(da,&xlocal);DAGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal);  DAGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal);
  DAVecGetArray(da,xlocal,&xx);
  DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);

  for (i=xs; i<xs+xm; i++) {
    if (i == 0 || i == Mx-1) { MatSetValue(*J,i,i,1.0/hx,INSERT_VALUES);}
    else {
      MatSetValue(*J,i,i-1,-1.0/hx,INSERT_VALUES);
      MatSetValue(*J,i,i,2.0/hx - hx*PetscExpScalar(xx[i]),INSERT_VALUES);
      MatSetValue(*J,i,i+1,-1.0/hx,INSERT_VALUES);
    }
  }
  MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);  *flag = SAME_NONZERO_PATTERN;
  DAVecRestoreArray(da,xlocal,&xx);DARestoreLocalVector(da,&xlocal);
  return 0;}

