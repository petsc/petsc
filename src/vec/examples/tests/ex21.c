/*$Id: ex21.c,v 1.2 1999/09/27 21:29:19 bsmith Exp bsmith $*/

static char help[] = "Tests VecMax() with index\
  -n <length> : vector length\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5, ierr, idx;
  Scalar        value;
  Vec           x;
  PetscRandom   rand;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  /* create vector */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRA(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT_REAL,&rand);CHKERRA(ierr);
  ierr = VecSetRandom(rand,x);CHKERRA(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRA(ierr);

  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecMax(x,&idx,&value);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Maximum value %g index %d\n",value,idx);CHKERRA(ierr);
  ierr = VecMin(x,&idx,&value);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Minimum value %g index %d\n",value,idx);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
