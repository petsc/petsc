#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.2 1999/01/12 23:13:42 bsmith Exp $";
#endif

static char help[] = "Tests repeated VecSetType()\n\n";

#include "vec.h"
#include "sys.h"

int main(int argc,char **argv)
{
  int           ierr, n = 5;
  Scalar        one = 1.0, two = 2.0;
  Vec           x,y;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* create vector */
  ierr = VecCreate(PETSC_COMM_SELF,n,PETSC_DECIDE,&x); CHKERRA(ierr);
  ierr = VecSetType(x,"mpi");CHKERRA(ierr);
  ierr = VecSetType(x,"seq");CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);
  ierr = VecSetType(x,"mpi");CHKERRA(ierr);

  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecSet(&two,y); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
