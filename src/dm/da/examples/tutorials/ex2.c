#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.2 1999/04/16 16:11:37 bsmith Exp balay $";
#endif

static char help[] = "Tests DAGlobalToNaturalAllCreate() using contour plotting for 2d DAs.\n\n";

#include "da.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int            i,j,rank, M = 10, N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE,ierr,flg;
  DA             da;
  Viewer         viewer;
  Vec            localall, global;
  Scalar         value,*vlocal;
  DAPeriodicType ptype = DA_NONPERIODIC;
  DAStencilType  stype = DA_STENCIL_BOX;
  VecScatter     tolocalall,fromlocalall;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer);CHKERRA(ierr);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-star_stencil",&flg);CHKERRA(ierr);
  if (flg) stype = DA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,ptype,stype,
                    M,N,m,n,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,M*N,&localall);CHKERRA(ierr);

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  value = 5.0*rank;
  ierr = VecSet(&value,global);CHKERRA(ierr);

  ierr = VecView(global,viewer);CHKERRA(ierr);

  /*
     Create Scatter from global DA parallel vector to local vector that
   contains all entries
  */
  ierr = DAGlobalToNaturalAllCreate(da,&tolocalall);CHKERRA(ierr);
  ierr = DANaturalAllToGlobalCreate(da,&fromlocalall);CHKERRA(ierr);

  ierr = VecScatterBegin(global,localall,INSERT_VALUES,SCATTER_FORWARD,tolocalall);CHKERRA(ierr);
  ierr = VecScatterEnd(global,localall,INSERT_VALUES,SCATTER_FORWARD,tolocalall);CHKERRA(ierr);

  ierr = VecGetArray(localall,&vlocal);CHKERRA(ierr);
  for ( j=0; j<N; j++ ) {
    for ( i=0; i<M; i++ ) {
      *vlocal++ += i + j*M;
    }
  }
  ierr = VecRestoreArray(localall,&vlocal);CHKERRA(ierr);

  /* scatter back to global vector */
  ierr = VecScatterBegin(localall,global,INSERT_VALUES,SCATTER_FORWARD,fromlocalall);CHKERRA(ierr);
  ierr = VecScatterEnd(localall,global,INSERT_VALUES,SCATTER_FORWARD,fromlocalall);CHKERRA(ierr);

  ierr = VecView(global,viewer);CHKERRA(ierr);

  /* Free memory */
  ierr = VecScatterDestroy(tolocalall);CHKERRA(ierr);
  ierr = VecScatterDestroy(fromlocalall);CHKERRA(ierr);
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(localall);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
