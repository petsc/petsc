
static char help[] = "This example passes a sparse matrix to Matlab.\n\n";
#include "sles.h"
#include <stdio.h>
#include "petsc.h"
#include "stencil.h"
#include "grid.h"

int main(int argc,char **args)
{
  int     ierr,m = 4,n = 5;
  Scalar  one = 1.0;
  Vec     x;
  Mat     A;
  Stencil stencil;
  Grid    grid;
  Viewer  viewer;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,"-m",&m);
  OptionsGetInt(0,"-n",&n);

  ierr = ViewerMatlabOpen("eagle",-1,&viewer); CHKERRA(ierr);

  if ((ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&A)))
                                                           SETERRA(ierr,0);
  ierr = GridCreateUniform2d(MPI_COMM_WORLD,m,0.0,1.0,n,0.0,1.0,&grid);
  ierr = StencilCreate(MPI_COMM_WORLD,STENCIL_Uxx,&stencil); CHKERRA(ierr);
  StencilAddStage(stencil,grid,0,0,0,A); CHKERRA(ierr);
  StencilDestroy(stencil);
  ierr = StencilCreate(MPI_COMM_WORLD,STENCIL_Uyy,&stencil); CHKERRA(ierr);
  StencilAddStage(stencil,grid,0,0,0,A); CHKERRA(ierr);
  StencilDestroy(stencil);
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = StencilCreate(MPI_COMM_WORLD,STENCIL_DIRICHLET,&stencil); 
  CHKERRA(ierr);  
  ierr = StencilAddStage(stencil,grid,0,0,0,A); CHKERRA(ierr);
  StencilDestroy(stencil);

  ierr = MatView(A,viewer); CHKERRA(ierr);

  ierr = VecCreateSequential(MPI_COMM_SELF,m,&x); CHKERRA(ierr);
  VecSet(&one,x);
  ierr = VecView(x,viewer); CHKERRA(ierr);
  
  sleep(30);
  ierr = PetscObjectDestroy((PetscObject) viewer); CHKERRA(ierr);

  VecDestroy(x); 
  MatDestroy(A);
  GridDestroy(grid);
  PetscFinalize();
  return 0;
}
    


