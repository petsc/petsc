
static char help[] = "Solves 5pt stencil linear system with SLES";
#include "sles.h"
#include "stdio.h"
#include "options.h"
#include "stencil.h"
#include "grid.h"

int main(int argc,char **args)
{
  int     ierr,i,m = 4,n = 5, its;
  Scalar  none = -1.0, one = 1.0;
  Vec     x,b,u;
  Mat     A;
  SLES    sles;
  double  norm;
  Stencil stencil;
  Grid    grid;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-m",&m);
  OptionsGetInt(0,0,"-n",&n);

  if (ierr = VecCreateInitialVector(m*n,&x)) SETERR(ierr,0);
  if (ierr = VecCreate(x,&b)) SETERR(ierr,0);
  if (ierr = VecCreate(x,&u)) SETERR(ierr,0);
  if (ierr = VecSet(&one,u)) SETERR(ierr,0);

  if (ierr = MatCreateInitialMatrix(m*n,m*n,&A)) SETERR(ierr,0);
  ierr = GridCreateUniform2d(MPI_COMM_WORLD,m,0.0,1.0,n,0.0,1.0,&grid);
  ierr = StencilCreate(STENCIL_Uxx,&stencil); CHKERR(ierr);
  StencilAddStage(stencil,grid,0,A); CHKERR(ierr);
  StencilDestroy(stencil);
  ierr = StencilCreate(STENCIL_Uyy,&stencil); CHKERR(ierr);
  StencilAddStage(stencil,grid,0,A); CHKERR(ierr);
  StencilDestroy(stencil);
  ierr = MatBeginAssembly(A); CHKERR(ierr);
  ierr = MatEndAssembly(A); CHKERR(ierr);
  ierr = StencilCreate(STENCIL_DIRICHLET,&stencil); CHKERR(ierr);  
  ierr = StencilAddStage(stencil,grid,0,A); CHKERR(ierr);
  StencilDestroy(stencil);

  if (ierr = MatMult(A,u,b)) SETERR(ierr,0);

  if (ierr = SLESCreate(&sles)) SETERR(ierr,0);
  if (ierr = SLESSetMat(sles,A)) SETERR(ierr,0);
  if (ierr = SLESSetFromOptions(sles)) SETERR(ierr,0);
  if (ierr = SLESSolve(sles,b,x,&its)) SETERR(ierr,0);

  /* check error */
  if (ierr = VecAXPY(&none,u,x)) SETERR(ierr,0);
  if (ierr = VecNorm(x,&norm)) SETERR(ierr,0);
  printf("Norm of error %g Iterations %d\n",norm,its);
 
  VecDestroy(x); VecDestroy(u); MatDestroy(A); SLESDestroy(sles);
  PetscFinalize();
  return 0;
}
    


