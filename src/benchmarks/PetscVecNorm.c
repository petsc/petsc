#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscVecNorm.c,v 1.6 1999/03/19 21:24:35 bsmith Exp balay $";
#endif

#include "vec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv)
{
  Vec        x;
  double     norm;
  PLogDouble t1,t2;
  int        ierr,n = 10000,flg;

  PetscInitialize(&argc, &argv,0,0);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);

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
