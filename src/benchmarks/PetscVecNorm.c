#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscVecNorm.c,v 1.2 1997/07/10 03:46:48 bsmith Exp bsmith $";
#endif

#include "vec.h"

int main( int argc, char **argv)
{
  Vec        x;
  double     norm;
  PLogDouble t1,t2;
  int        ierr,n = 10000,flg;

  PetscInitialize(&argc, &argv,0,0);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  ierr = VecCreate(PETSC_COMM_SELF,n,&x); CHKERRA(ierr);

  /* To take care of paging effects */
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);

  t1 = PetscGetTime();
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  t2 = PetscGetTime();
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);

  fprintf(stderr,"%s : \n","PetscMemcpy");
  fprintf(stderr," Time %g\n",t2-t1);

  PetscFinalize();
  PetscFunctionReturn(0);
}
