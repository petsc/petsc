#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.3 1999/03/19 21:18:10 bsmith Exp balay $";
#endif

static char help[] = "Tests repeated VecSetType()\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           ierr, n = 5;
  Scalar        one = 1.0, two = 2.0;
  Vec           x,y;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* create vector */
  ierr = VecCreate(PETSC_COMM_SELF,n,PETSC_DECIDE,&x);CHKERRA(ierr);
  ierr = VecSetType(x,"mpi");CHKERRA(ierr);
  ierr = VecSetType(x,"seq");CHKERRA(ierr);
  ierr = VecDuplicate(x,&y);CHKERRA(ierr);
  ierr = VecSetType(x,"mpi");CHKERRA(ierr);

  ierr = VecSet(&one,x);CHKERRA(ierr);
  ierr = VecSet(&two,y);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
