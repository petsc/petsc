#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex7.c,v 1.3 1999/03/19 21:21:17 bsmith Exp bsmith $";
#endif

static char help[] = "Tests MatILUFactorSymbolic() on matrix with missing diagonal.\n\n"; 

#include "mat.h"
#include "pc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C,A; 
  int         i,j,ierr;
  Scalar      v;
  PC          pc;
  Vec         xtmp;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,3,3,&C); CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,3,&xtmp);CHKERRA(ierr);
  i = 0; j = 0; v = 4;
  ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  i = 0; j = 2; v = 1;
  ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  i = 1; j = 0; v = 1;
  ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  i = 1; j = 1; v = 4;
  ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  i = 2; j = 1; v = 1;
  ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = PCCreate(PETSC_COMM_WORLD,&pc); CHKERRA(ierr);
  ierr = PCSetFromOptions(pc); CHKERRA(ierr);
  ierr = PCSetOperators(pc,C,C,DIFFERENT_NONZERO_PATTERN); CHKERRA(ierr);
  ierr = PCSetVector(pc,xtmp); CHKERRA(ierr);
  ierr = PCSetUp(pc);CHKERRA(ierr);
  ierr = PCGetFactoredMatrix(pc,&A); CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = PCDestroy(pc);CHKERRA(ierr);
  ierr = VecDestroy(xtmp);CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);


  PetscFinalize();
  return 0;
}

 
