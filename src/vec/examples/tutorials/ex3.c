
static char help[] = "Displays a vector visually\n";

#include "petsc.h"
#include "comm.h"
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

  OptionsCreate(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-n",&n);

  /* create vector */
  ierr = VecCreateSequential(n,&x); CHKERR(ierr);

  for ( i=0; i<n; i++ ) {
    v = (double) i;
    VecSetValues(x,1,&i,&v,InsertValues);
  }
  VecBeginAssembly(x);
  VecEndAssembly(x);

  ierr = DrawOpenX(0,0,0,0,300,300,&win); CHKERR(ierr);
  ierr = DrawLGCreate(win,1,&lg); CHKERR(ierr);

  VecView(x,(Viewer) lg);
  sleep(5);

  ierr = DrawLGDestroy(lg); CHKERR(ierr);

  ierr = DrawDestroy(win); CHKERR(ierr);
  ierr = VecDestroy(x);CHKERR(ierr);

  PetscFinalize();
  return 0;
}
 
