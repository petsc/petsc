#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex14.c,v 1.2 1999/03/11 16:23:59 bsmith Exp bsmith $";
#endif

static char help[] = "Tests saving DA vectors to files\n\n";

#include "da.h"
#include "sys.h"

int main(int argc,char **argv)
{
  int      rank, M = 10, N = 8, m = PETSC_DECIDE, n = PETSC_DECIDE, ierr,flg;
  int      dof = 1;
  DA       da;
  Vec      local, global, natural;
  Scalar   value;
  Viewer   bviewer;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-dof",&dof,&flg); CHKERRA(ierr);

  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
                    M,N,m,n,dof,1,PETSC_NULL,PETSC_NULL,&da); CHKERRA(ierr);
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

  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",BINARY_CREATE,&bviewer);CHKERRA(ierr);
  ierr = DAView(da,bviewer);CHKERRA(ierr);
  ierr = VecView(global,bviewer); CHKERRA(ierr);
  ierr = ViewerDestroy(bviewer);CHKERRA(ierr);

  /* Free memory */
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = VecDestroy(natural); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
