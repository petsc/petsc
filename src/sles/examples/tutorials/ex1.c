
#include "sles.h"
#include "stdio.h"
#include "options.h"

int main(int argc,char **args)
{
  int    ierr,i,n = 10, col[3];
  Scalar none = -1.0, one = 1.0, value[3];
  Vec    x,b,u;
  Mat    A;
  SLES   sles;
  double norm;

  OptionsCreate(&argc,&args,0,0);
  OptionsGetInt(0,"-n",&n);

  if (ierr = VecCreateInitialVector(n,&x)) SETERR(ierr,0);
  if (ierr = VecCreate(x,&b)) SETERR(ierr,0);
  if (ierr = VecCreate(x,&u)) SETERR(ierr,0);
  if (ierr = VecSet(&one,u)) SETERR(ierr,0);

  if (ierr = MatCreateInitialMatrix(n,n,&A)) SETERR(ierr,0);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,InsertValues); CHKERR(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,value,InsertValues); CHKERR(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,InsertValues); CHKERR(ierr);
  ierr = MatBeginAssembly(A); CHKERR(ierr);
  ierr = MatEndAssembly(A); CHKERR(ierr);
  
  if (ierr = MatMult(A,u,b)) SETERR(ierr,0);

  if (ierr = SLESCreate(&sles)) SETERR(ierr,0);
  if (ierr = SLESSetMat(sles,A)) SETERR(ierr,0);
  if (ierr = SLESSetFromOptions(sles)) SETERR(ierr,0);
  if (ierr = SLESSolve(sles,b,x)) SETERR(ierr,0);

  /* check error */
  if (ierr = VecAXPY(&none,u,x)) SETERR(ierr,0);
  if (ierr = VecNorm(x,&norm)) SETERR(ierr,0);
  printf("Norm of error %g\n",norm);
 
  VecDestroy(x); VecDestroy(u); MatDestroy(A); SLESDestroy(sles);
  PetscFinalize();
  return 0;
}
    


