#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.1 1999/02/21 03:35:08 bsmith Exp bsmith $";
#endif

static char help[] = "Tests VecView() contour plotting for 2d DAs.\n\n";

#include "da.h"
#include "sys.h"

int main(int argc,char **argv)
{
  int            rank, M = 10, N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE,ierr,flg;
  DA             da;
  Viewer         viewer;
  Vec            local, global;
  Scalar         value;
  DAPeriodicType ptype = DA_NONPERIODIC;
  DAStencilType  stype = DA_STENCIL_BOX;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer); CHKERRA(ierr);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-star_stencil",&flg);CHKERRA(ierr);
  if (flg) stype = DA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,ptype,stype,
                    M,N,m,n,1,1,PETSC_NULL,PETSC_NULL,&da); CHKERRA(ierr);
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

  ierr = DAView(da,viewer); CHKERRA(ierr);
  ierr = VecView(global,viewer); CHKERRA(ierr);

  /* Free memory */
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
