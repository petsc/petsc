
static char help[] = "Passes sparse matrix to matlab";
#include "sles.h"
#include "stdio.h"
#include "options.h"
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
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-m",&m);
  OptionsGetInt(0,0,"-n",&n);

  ierr = ViewerMatlabOpen("eagle",-1,&viewer); CHKERR(ierr);

  if ((ierr = MatCreateInitialMatrix(m*n,m*n,&A))) SETERR(ierr,0);
  ierr = GridCreateUniform2d(MPI_COMM_WORLD,m,0.0,1.0,n,0.0,1.0,&grid);
  ierr = StencilCreate(STENCIL_Uxx,&stencil); CHKERR(ierr);
  StencilAddStage(stencil,grid,0,0,0,A); CHKERR(ierr);
  StencilDestroy(stencil);
  ierr = StencilCreate(STENCIL_Uyy,&stencil); CHKERR(ierr);
  StencilAddStage(stencil,grid,0,0,0,A); CHKERR(ierr);
  StencilDestroy(stencil);
  ierr = MatBeginAssembly(A,FINAL_ASSEMBLY); CHKERR(ierr);
  ierr = MatEndAssembly(A,FINAL_ASSEMBLY); CHKERR(ierr);
  ierr = StencilCreate(STENCIL_DIRICHLET,&stencil); CHKERR(ierr);  
  ierr = StencilAddStage(stencil,grid,0,0,0,A); CHKERR(ierr);
  StencilDestroy(stencil);

  ierr = MatView(A,viewer); CHKERR(ierr);

  ierr = VecCreateSequential(m,&x); CHKERR(ierr);
  VecSet(&one,x);
  ierr = VecView(x,viewer); CHKERR(ierr);
  
  sleep(30);
  ierr = PetscDestroy((PetscObject) viewer); CHKERR(ierr);

  VecDestroy(x); 
  MatDestroy(A);
  GridDestroy(grid);
  PetscFinalize();
  return 0;
}
    


