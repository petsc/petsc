
static char help[] = "Solves tridiagonal linear system with SLES";
#include "sles.h"
#include "stdio.h"
#include "options.h"

int main(int argc,char **args)
{
  int    ierr,i,n = 10, col[3], its;
  Scalar none = -1.0, one = 1.0, value[3];
  Vec    x,b,u;
  Mat    A;
  SLES   sles;
  double norm;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-n",&n);

  if ((ierr = VecCreateInitialVector(n,&x))) SETERRA(ierr,0);
  if ((ierr = VecCreate(x,&b))) SETERRA(ierr,0);
  if ((ierr = VecCreate(x,&u))) SETERRA(ierr,0);
  if ((ierr = VecSet(&one,u))) SETERRA(ierr,0);

  if ((ierr = MatCreateInitialMatrix(n,n,&A))) SETERRA(ierr,0);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,InsertValues); CHKERRA(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,value,InsertValues); CHKERRA(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,InsertValues); CHKERRA(ierr);
  ierr = MatBeginAssembly(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatEndAssembly(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  
  if ((ierr = MatMult(A,u,b))) SETERRA(ierr,0);

  if ((ierr = SLESCreate(&sles))) SETERRA(ierr,0);
  if ((ierr = SLESSetOperators(sles,A,A,0))) SETERRA(ierr,0);
  if ((ierr = SLESSetFromOptions(sles))) SETERRA(ierr,0);
  if ((ierr = SLESSolve(sles,b,x,&its))) SETERRA(ierr,0);

  /* check error */
  if ((ierr = VecAXPY(&none,u,x))) SETERRA(ierr,0);
  if ((ierr = VecNorm(x,&norm))) SETERRA(ierr,0);
  printf("Norm of error %g Iterations %d\n",norm,its);
 
  VecDestroy(x); VecDestroy(u); VecDestroy(b);
  MatDestroy(A); SLESDestroy(sles);
  PetscFinalize();
  return 0;
}
    


