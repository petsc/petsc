
static char help[] = "Displays a vector visually\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           i,n = 5, ierr;
  Scalar        v;
  Vec           x;
  DrawCtx       win;
  DrawLGCtx     lg;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-n",&n);

  /* create vector */
  ierr = VecCreateSequential(n,&x); CHKERRA(ierr);

  for ( i=0; i<n; i++ ) {
    v = (double) i;
    VecSetValues(x,1,&i,&v,InsertValues);
  }
  VecBeginAssembly(x);
  VecEndAssembly(x);

  ierr = DrawOpenX(MPI_COMM_SELF,0,0,0,0,300,300,&win); CHKERRA(ierr);
  ierr = DrawLGCreate(win,1,&lg); CHKERRA(ierr);

  VecView(x,(Viewer) lg);

  ierr = DrawLGDestroy(lg); CHKERRA(ierr);

  ierr = DrawDestroy(win); CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
