
static char help[] = "This example displays a vector visually.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           i,n = 50, ierr;
  Scalar        v;
  Vec           x;
  DrawCtx       win;
  DrawLGCtx     lg;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  OptionsGetInt(0,"-n",&n);

  /* create vector */
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&x); CHKERRA(ierr);

  for ( i=0; i<n; i++ ) {
    v = (double) i;
    ierr = VecSetValues(x,1,&i,&v,INSERTVALUES); CHKERRA(ierr);
  }

  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  ierr = DrawOpenX(MPI_COMM_SELF,0,0,0,0,300,300,&win); CHKERRA(ierr);
  ierr = DrawLGCreate(win,1,&lg); CHKERRA(ierr);

  ierr = VecView(x,(Viewer) lg); CHKERRA(ierr);

  ierr = DrawLGDestroy(lg); CHKERRA(ierr);

  ierr = DrawDestroy(win); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
