#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.18 1996/03/10 17:29:57 bsmith Exp bsmith $";
#endif
      
/* Peter Mell created this file on 7/25/95 */

static char help[] = "Tests various 3-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  int            rank,M = 3, N = 5, P=3,s=1, w=2,flg; 
  int            m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE, ierr;
  DA             da;
  Viewer         viewer;
  Vec            local,global;
  Scalar         value;
  DAPeriodicType wrap = DA_XYPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_BOX;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,"",300,0,400,300,&viewer); CHKERRA(ierr);
  
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-P",&P,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-p",&p,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,&flg); CHKERRA(ierr);

  OptionsHasName(PETSC_NULL,"-star",&flg); if (flg) stencil_type =  DA_STENCIL_STAR;

  ierr = DACreate3d(MPI_COMM_WORLD,wrap,stencil_type,M,N,P,m,n,p,w,s,&da); 
  CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);
     
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  value = rank;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  if (M*N*P<40)
  {
    PetscPrintf(MPI_COMM_WORLD,"\nGlobal Vectors:\n");
    ierr = VecView(global,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 
    PetscPrintf(MPI_COMM_WORLD,"\n\n");
  }

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  if (M*N*P<40) {
    PetscPrintf(MPI_COMM_WORLD,"\nView Local Array - Processor [%d]\n",rank);
    ierr = VecView(local,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 
  }
  ierr = DAView(da,viewer); CHKERRA(ierr);

  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
  




















