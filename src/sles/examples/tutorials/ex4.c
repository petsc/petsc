#ifndef lint
static char vcid[] = "$Id: ex11.c,v 1.20 1996/04/09 20:23:59 curfman Exp bsmith $";
#endif

static char help[] = "Ilustrates using a different preconditioner matrix and\n\
linear system matrix in the SLES solvers.  Note that different storage formats\n\
can be used for the different matrices.\n\n";

#include "mat.h"
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat        C, B;
  int        i, j, m = 15, n = 17, its, I, J, ierr, Istart, Iend, flg;
  Scalar     v,  one = 1.0, scale = 0.0;
  Vec        u, b, x, tmp;
  SLES       sles;
  PetscRandom rctx;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* Create the linear system matrix */
  ierr = MatCreateMPIBDiag(MPI_COMM_WORLD,PETSC_DECIDE,m*n,m*n,
         0,1,PETSC_NULL,PETSC_NULL,&C); CHKERRA(ierr);

  /* Create a different preconditioner matrix.  This is usually done
     to form a cheaper (or sparser) preconditioner matrix compared
     to the linear system matrix. */
  /* ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_NULL,PETSC_NULL,
         m*n,m*n,5,0,5,0,&B); CHKERRA(ierr); */
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&B); CHKERRA(ierr);

  /* Assemble the two matrices */
  ierr = MatGetOwnershipRange(C,&Istart,&Iend); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 ) {
      J=I-n; 
      ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(B,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    if ( i<m-1 ) {
      J=I+n; 
      ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(B,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    if ( j>0 ) {
      J=I-1; 
      ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    if ( j<n-1 ) {
      J=I+1; 
      ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    v = 5.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    v = 3.0; ierr = MatSetValues(B,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -0.5; i = I/n;
    if ( i>1 ) { 
      J=I-(n+1); ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    if ( i<m-2 ) {
      J=I+n+1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Create and set vectors */
  ierr = VecCreate(MPI_COMM_WORLD,m*n,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);

  /* Make solution be 1 to random noise */
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&tmp); CHKERRA(ierr);
  ierr = PetscRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT,&rctx); CHKERRA(ierr);
  ierr = VecSetRandom(rctx,tmp); CHKERRA(ierr);
  ierr = PetscRandomDestroy(rctx); CHKERRA(ierr);
  ierr = VecAXPY(&scale,tmp,u); CHKERRA(ierr);
  ierr = VecDestroy(tmp); CHKERRA(ierr);
  ierr = MatMult(C,u,b); CHKERRA(ierr);

  /* Create SLES context; set operators and options; solve linear system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,B,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(B); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
