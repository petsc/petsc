/*$Id: PetscVecNorm.c,v 1.10 2000/01/11 21:03:44 bsmith Exp balay $*/

#include "petscvec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Vec        x;
  double     norm;
  PLogDouble t1,t2;
  int        ierr,n = 10000;

  PetscInitialize(&argc,&argv,0,0);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRA(ierr);

  /* To take care of paging effects */
  ierr = PetscGetTime(&t1);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);

  ierr = PetscGetTime(&t1);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscGetTime(&t2);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);

  fprintf(stderr,"%s : \n","PetscMemcpy");
  fprintf(stderr," Time %g\n",t2-t1);

  PetscFinalize();
  PetscFunctionReturn(0);
}
