
#include <petscvec.h>
#include <petsctime.h>

int main(int argc,char **argv)
{
  Vec            x;
  PetscReal      norm;
  PetscLogDouble t1,t2;
  PetscInt       n = 10000;

  PetscCall(PetscInitialize(&argc,&argv,0,0));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  PetscCall(VecCreate(PETSC_COMM_SELF,&x));
  PetscCall(VecSetSizes(x,n,n));
  PetscCall(VecSetFromOptions(x));

  PetscPreLoadBegin(PETSC_TRUE,"VecNorm");
  PetscCall(PetscTime(&t1));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscPreLoadEnd();
  PetscCall(PetscTime(&t2));
  fprintf(stdout,"%s : \n","VecNorm");
  fprintf(stdout," Time %g\n",t2-t1);
  PetscCall(PetscFinalize());
  return 0;
}
