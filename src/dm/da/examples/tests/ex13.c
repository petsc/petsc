/*$Id: ex13.c,v 1.8 1999/11/05 14:47:57 bsmith Exp balay $*/

static char help[] = "Tests loading DA vector from file\n\n";

#include "petscda.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int      ierr,M = PETSC_DECIDE,N = PETSC_DECIDE;
  DA       da;
  Vec      global;
  Viewer   bviewer;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRA(ierr);

  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",BINARY_RDONLY,&bviewer);CHKERRA(ierr);
  ierr = DALoad(bviewer,M,N,PETSC_DECIDE,&da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr); 
  ierr = VecLoadIntoVector(bviewer,global);CHKERRA(ierr); 
  ierr = ViewerDestroy(bviewer);CHKERRA(ierr);


  ierr = VecView(global,VIEWER_DRAW_WORLD);CHKERRA(ierr);


  /* Free memory */
  ierr = VecDestroy(global);CHKERRA(ierr); 
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
