#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex13.c,v 1.1 1999/03/04 21:18:05 bsmith Exp bsmith $";
#endif

static char help[] = "Tests loading DA vector from file\n\n";

#include "da.h"
#include "sys.h"

int main(int argc,char **argv)
{
  int      ierr,flg,M,N,rank;
  int      dof = 1;
  DA       da;
  Vec      local, global, natural;
  Scalar   value;
  Viewer   bviewer;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);

  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",BINARY_RDONLY,&bviewer);CHKERRA(ierr);
  ierr = DALoad(bviewer,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&da);CHKERRA(ierr);
  ierr = ViewerDestroy(bviewer);CHKERRA(ierr);

  ierr = DACreateGlobalVector(da,&global); CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local); CHKERRA(ierr);

  value = -3.0;
  ierr = VecSet(&value,global); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  value = rank+1;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,ADD_VALUES,global); CHKERRA(ierr);

  ierr = DACreateNaturalVector(da,&natural);CHKERRA(ierr);
  ierr = DAGlobalToNaturalBegin(da,global,INSERT_VALUES,natural); CHKERRA(ierr);
  ierr = DAGlobalToNaturalEnd(da,global,INSERT_VALUES,natural); CHKERRA(ierr);

  ierr = VecView(global,VIEWER_DRAW_WORLD); CHKERRA(ierr);


  /* Free memory */
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = VecDestroy(natural); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
