#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex70.c,v 1.3 1999/03/19 21:19:59 bsmith Exp bsmith $";
#endif

static char help[] = "Tests Vec/MatSetValues() with negative row and column indices.\n\n"; 

#include "mat.h"
#include "pc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         i[2],j[2],ierr;
  Scalar      v[] = {1.0,2.0,3.0,4.0};
  Vec         x;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,3,3,&C); CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,3,&x);CHKERRA(ierr);
  
  i[0] = 1; i[1] = -1; j[0] = 1; j[1] = 2;
  ierr = MatSetValues(C,2,i,2,j,v,INSERT_VALUES); CHKERRA(ierr);
  ierr = MatSetValues(C,2,j,2,i,v,INSERT_VALUES); CHKERRA(ierr);
  ierr = VecSetValues(x,2,i,v,INSERT_VALUES); CHKERRA(ierr);
  
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

 
