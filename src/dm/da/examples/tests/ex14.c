/*$Id: ex14.c,v 1.9 2000/01/11 21:03:26 bsmith Exp balay $*/

static char help[] = "Tests saving DA vectors to files\n\n";

#include "petscda.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int      rank,M = 10,N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE,ierr;
  int      dof = 1;
  DA       da;
  Vec      local,global,natural;
  Scalar   value;
  Viewer   bviewer;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRA(ierr);

  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
                    M,N,m,n,dof,1,PETSC_NULL,PETSC_NULL,&da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);

  value = -3.0;
  ierr = VecSet(&value,global);CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  value = rank+1;
  ierr = VecScale(&value,local);CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,ADD_VALUES,global);CHKERRA(ierr);

  ierr = DACreateNaturalVector(da,&natural);CHKERRA(ierr);
  ierr = DAGlobalToNaturalBegin(da,global,INSERT_VALUES,natural);CHKERRA(ierr);
  ierr = DAGlobalToNaturalEnd(da,global,INSERT_VALUES,natural);CHKERRA(ierr);

  ierr = DASetFieldName(da,0,"First field");CHKERRA(ierr);
  ierr = VecView(global,VIEWER_DRAW_WORLD);CHKERRA(ierr); 

  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",BINARY_CREATE,&bviewer);CHKERRA(ierr);
  ierr = DAView(da,bviewer);CHKERRA(ierr);
  ierr = VecView(global,bviewer);CHKERRA(ierr);
  ierr = ViewerDestroy(bviewer);CHKERRA(ierr);

  /* Free memory */
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = VecDestroy(natural);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
