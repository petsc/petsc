#ifndef lint
static char vcid[] = "$Id: ex15.c,v 1.26 1996/03/19 21:23:15 bsmith Exp bsmith $";
#endif

static char help[] = "Displays a vector visually.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  int        i,n = 50, ierr, flg;
  Scalar     v;
  Vec        x;
  Viewer     viewer;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  /* create vector */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&x); CHKERRA(ierr);

  for ( i=0; i<n; i++ ) {
    v = (double) i;
    ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
  }

  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  ierr = ViewerDrawOpenX(MPI_COMM_SELF,0,0,0,0,300,300,&viewer); CHKERRA(ierr);
  ierr = VecView(x,viewer); CHKERRA(ierr);
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
