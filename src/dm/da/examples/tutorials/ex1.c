#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.33 1998/12/03 04:06:22 bsmith Exp $";
#endif

static char help[] = "Tests various DA routines.\n\n";

#include "da.h"
#include "sys.h"

int main(int argc,char **argv)
{
  int      rank, M = 10, N = 8, m = PETSC_DECIDE, n = PETSC_DECIDE, ierr,flg;
  DA       da;
  Viewer   viewer;
  Vec      local, global;
  Scalar   value;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer); CHKERRA(ierr);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
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

  ierr = VecView(global,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = DAView(da,viewer); CHKERRA(ierr);

  /* Free memory */
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
