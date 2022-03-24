
#include <petscvec.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  Vec            x;
  PetscReal      norm;
  PetscLogDouble t1,t2;
  PetscInt       n = 10000;

  CHKERRQ(PetscInitialize(&argc,&argv,0,0));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(VecCreate(PETSC_COMM_SELF,&x));
  CHKERRQ(VecSetSizes(x,n,n));
  CHKERRQ(VecSetFromOptions(x));

  PetscPreLoadBegin(PETSC_TRUE,"VecNorm");
  CHKERRQ(PetscTime(&t1));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  PetscPreLoadEnd();
  CHKERRQ(PetscTime(&t2));
  fprintf(stdout,"%s : \n","VecNorm");
  fprintf(stdout," Time %g\n",t2-t1);
  CHKERRQ(PetscFinalize());
  return 0;
}
