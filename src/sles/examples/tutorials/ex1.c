
static char help[] = 
"This example solves tridiagonal linear system with SLES.\n";

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

  ierr = VecCreateInitialVector(MPI_COMM_WORLD,n,&x); CHKERR(ierr);
  ierr = VecCreate(x,&b); CHKERR(ierr);
  ierr = VecCreate(x,&u); CHKERR(ierr);
  ierr = VecSet(&one,u); CHKERR(ierr);

  ierr = MatCreateInitialMatrix(MPI_COMM_WORLD,n,n,&A); CHKERR(ierr);
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
  
  ierr = MatMult(A,u,b); CHKERR(ierr);

  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERR(ierr);
  ierr = SLESSetOperators(sles,A,A,0); CHKERR(ierr);
  ierr = SLESSetFromOptions(sles); CHKERR(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERR(ierr);

  /* check error */
  ierr = VecAXPY(&none,u,x); CHKERR(ierr);
  ierr  = VecNorm(x,&norm); CHKERR(ierr);
  MPE_printf(MPI_COMM_WORLD,"Norm of error %g Iterations %d\n",norm,its);
 
  VecDestroy(x); VecDestroy(u); VecDestroy(b);
  MatDestroy(A); SLESDestroy(sles);
  PetscFinalize();
  return 0;
}
    


