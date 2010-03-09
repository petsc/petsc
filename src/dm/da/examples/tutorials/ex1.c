
static char help[] = "Tests VecView() contour plotting for 2d DAs.\n\n";

/* 
  MATLAB must be installed to configure PETSc to have MATLAB engine.
Unless you have specific important reasons for using the Matlab engine, we do not
recommend it. If you want to use Matlab for visualization and maybe a little post processing
then you can use the socket viewer and send the data to Matlab via that.

  VecView() on DA vectors first puts the Vec elements into global natural ordering before printing (or plotting)
them. In 2d 5 by 2 DA this means the numbering is

     5  6   7   8   9
     0  1   2   3   4

Now the default split across 2 processors with the DA  is (by rank)

    0  0   0  1   1
    0  0   0  1   1

So the global PETSc ordering is

    3  4  5   8  9
    0  1  2   6  7

Use the options
     -da_grid_x <nx> - number of grid points in x direction, if M < 0
     -da_grid_y <ny> - number of grid points in y direction, if N < 0
     -da_processors_x <MX> number of processors in x directio
     -da_processors_y <MY> number of processors in x direction
*/

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscInt       M = -10,N = -8;
  PetscErrorCode ierr;
  PetscTruth     flg = PETSC_FALSE;
  DA             da;
  PetscViewer    viewer;
  Vec            local,global;
  PetscScalar    value;
  DAPeriodicType ptype = DA_NONPERIODIC;
  DAStencilType  stype = DA_STENCIL_BOX;
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscViewer    mviewer;
#endif

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscViewerMatlabOpen(PETSC_COMM_WORLD,"tmp.mat",FILE_MODE_WRITE,&mviewer);CHKERRQ(ierr);
#endif

  ierr = PetscOptionsGetTruth(PETSC_NULL,"-star_stencil",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) stype = DA_STENCIL_STAR;
      
  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,ptype,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRQ(ierr);

  value = -3.0;
  ierr = VecSet(global,value);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  value = rank+1;
  ierr = VecScale(local,value);CHKERRQ(ierr);
  ierr = DALocalToGlobal(da,local,ADD_VALUES,global);CHKERRQ(ierr);
  
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-view_global", &flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) { /* view global vector in natural ordering */
    ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = DAView(da,viewer);CHKERRQ(ierr);
  ierr = VecView(global,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = DAView(da,mviewer);CHKERRQ(ierr);
  ierr = VecView(global,mviewer);CHKERRQ(ierr);
#endif

  /* Free memory */
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscViewerDestroy(mviewer);CHKERRQ(ierr);
#endif
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
