#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.44 1999/05/04 20:34:27 balay Exp bsmith $";
#endif

static char help[] = "Tests PC and KSP on a tridiagonal matrix.  Note that most\n\
users should employ the SLES interface instead of using PC directly.\n\n";

#include "ksp.h"
#include "pc.h"
#include "petsc.h"
#include <stdio.h>

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     mat;          /* matrix */
  Vec     b, ustar, u;  /* vectors (RHS, exact solution, approx solution) */
  PC      pc;           /* PC context */
  KSP     ksp;          /* KSP context */
  int     ierr, n = 10, i, its, col[3];
  Scalar  value[3], mone = -1.0, one = 1.0, zero = 0.0;
  PCType  pcname;
  KSPType kspname;
  double  norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Create and initialize vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&ustar);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRA(ierr);
  ierr = VecSet(&one,ustar);CHKERRA(ierr);
  ierr = VecSet(&zero,u);CHKERRA(ierr);

  /* Create and assemble matrix */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&mat);CHKERRA(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(mat,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Compute right-hand-side vector */
  ierr = MatMult(mat,ustar,b);CHKERRA(ierr);

  /* Create PC context and set up data structures */
  ierr = PCCreate(PETSC_COMM_WORLD,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRA(ierr);
  ierr = PCSetFromOptions(pc);CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = PCSetVector(pc,u);  CHKERRA(ierr);
  ierr = PCSetUp(pc);CHKERRA(ierr);

  /* Create KSP context and set up data structures */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRA(ierr);
  ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRA(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRA(ierr);
  ierr = KSPSetSolution(ksp,u);CHKERRA(ierr);
  ierr = KSPSetRhs(ksp,b);CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = KSPSetPC(ksp,pc);CHKERRA(ierr);
  ierr = KSPSetUp(ksp);CHKERRA(ierr);

  /* Solve the problem */
  ierr = KSPGetType(ksp,&kspname);CHKERRA(ierr);
  ierr = PCGetType(pc,&pcname);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Running %s with %s preconditioning\n",kspname,pcname);CHKERRA(ierr);
  ierr = KSPSolve(ksp,&its);CHKERRA(ierr);
  ierr = VecAXPY(&mone,ustar,u);CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm);
  ierr = PetscPrintf(PETSC_COMM_SELF,"2 norm of error %A Number of iterations %d\n",norm,its);

  /* Free data structures */
  ierr = KSPDestroy(ksp);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(ustar);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(mat);CHKERRA(ierr);
  ierr = PCDestroy(pc);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
    


