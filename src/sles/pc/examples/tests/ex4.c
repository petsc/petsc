#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.35 1999/03/19 21:21:17 bsmith Exp balay $";
#endif

static char help[] = "Demonstrates the use of fast Richardson for SOR and tests\n\
the MatRelax() routines.\n\n";

#include "pc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat       mat;
  Vec       b,u;
  PC        pc;
  int       ierr, n = 5, i, col[3];
  Scalar    value[3], zero = 0.0;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Create vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRA(ierr);

  /* Create and assemble matrix */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,PETSC_NULL,&mat);CHKERRA(ierr);
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

  /* Create PC context and set up data structures */
  ierr = PCCreate(PETSC_COMM_WORLD,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCSOR);CHKERRA(ierr);
  ierr = PCSetFromOptions(pc);CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = PCSetVector(pc,u);CHKERRA(ierr);
  ierr = PCSetUp(pc);CHKERRA(ierr);

  value[0] = 1.0;
  for ( i=0; i<n; i++ ) {
    ierr = VecSet(&zero,u);CHKERRA(ierr);
    ierr = VecSetValues(u,1,&i,value,INSERT_VALUES);CHKERRA(ierr);
    ierr = PCApply(pc,u,b);CHKERRA(ierr);
    ierr = VecView(b,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  }

  /* Free data structures */
  ierr = MatDestroy(mat);CHKERRA(ierr);
  ierr = PCDestroy(pc);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr); 
  PetscFinalize();
  return 0;
}
    


