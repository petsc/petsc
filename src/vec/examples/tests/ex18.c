#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex18.c,v 1.18 1999/05/04 20:30:57 balay Exp bsmith $";
#endif

/* np = 1 */

static char help[] = "Compares BLAS dots on different machines. Input\n\
arguments are\n\
  -n <length> : local vector length\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int          n = 15,ierr,i,flg;
  Scalar       v;
  Vec          x,y;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg); if (n < 5) n = 5;


  /* create two vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&y);CHKERRA(ierr);

  for ( i=0; i<n; i++ ) {
    v = ((double) i) + 1.0/(((double) i) + .35);
    ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
    v += 1.375547826473644376;
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(y);CHKERRA(ierr);
  ierr = VecAssemblyEnd(y);CHKERRA(ierr);

  ierr = VecDot(x,y,&v);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Vector inner product %16.12e\n",v);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
