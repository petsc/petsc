#ifndef lint
static char vcid[] = "$Id: ex14.c,v 1.6 1996/01/01 01:04:27 bsmith Exp bsmith $";
#endif

static char help[] = "Tests the preconditioner ASM\n\n";

#include "mat.h"
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat     C;
  int     i, j, m = 15, n = 17, its, I, J, ierr, Istart, Iend, N = 1, M = 2;
  int     overlap = 1, Nsub, flg;
  Scalar  v,  one = 1.0;
  Vec     u,b,x;
  SLES    sles;
  PC      pc;
  IS      *is;

  PetscInitialize(&argc,&args,0,0,help);
  OptionsGetInt(PETSC_NULL,"-m",&m,&flg);   /* mesh lines in x */
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);   /* mesh lines in y */
  OptionsGetInt(PETSC_NULL,"-M",&M,&flg);   /* subdomains in x */
  OptionsGetInt(PETSC_NULL,"-N",&N,&flg);   /* subdomains in y */
  OptionsGetInt(PETSC_NULL,"-overlap",&overlap,&flg);

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreateSeqAIJ(MPI_COMM_WORLD,m*n,m*n,5,PETSC_NULL,&C); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
    v = 4.0; MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);
  }
  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Create and set vectors */
  ierr = VecCreateSeq(MPI_COMM_WORLD,m*n,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = MatMult(C,u,b); CHKERRA(ierr);

  /* Create SLES context */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);

  /* set operators and options; solve linear system */
  ierr = PCSetType(pc,PCASM); CHKERRQ(ierr);
  ierr = PCASMCreateSubdomains2D(m,n,M,N,1,overlap,&Nsub,&is); CHKERRQ(ierr);
  ierr = PCASMSetSubdomains(pc,Nsub,is); CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,C,C,ALLMAT_DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
