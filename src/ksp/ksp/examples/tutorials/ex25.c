
/*
 Partial differential equation

   d  (1 + e*sine(2*pi*k*x)) d u = 1, 0 < x < 1,
   --                        ---
   dx                        dx
with boundary conditions

   u = 0 for x = 0, x = 1

   This uses multigrid to solve the linear system

*/

static char help[] = "Solves 1D variable coefficient Laplacian using multigrid.\n\n";

#include <petscdmda.h>
#include <petscksp.h>

static PetscErrorCode ComputeMatrix(KSP,Mat,Mat,MatStructure*,void*);
static PetscErrorCode ComputeRHS(KSP,Vec,void*);

typedef struct {
  PetscInt    k;
  PetscScalar e;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  KSP            ksp;
  DM             da;
  AppCtx         user;
  Mat            A;
  Vec            b,b2;
  Vec            x;
  PetscReal      nrm;

  PetscInitialize(&argc,&argv,(char *)0,help);

  user.k = 1;
  user.e = .99;
  ierr = PetscOptionsGetInt(0,"-k",&user.k,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(0,"-e",&user.e,0);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-3,1,1,0,&da);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,da);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,&user);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  ierr = KSPGetOperators(ksp,&A,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&b2);CHKERRQ(ierr);
  ierr = MatMult(A,x,b2);CHKERRQ(ierr);
  ierr = VecAXPY(b2,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(b2,NORM_MAX,&nrm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %G\n",nrm);CHKERRQ(ierr);

  ierr = VecDestroy(&b2);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
static PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       mx,idx[2];
  PetscScalar    h,v[2];
  DM             da;

  PetscFunctionBegin;
  ierr   = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr   = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h      = 1.0/((mx-1));
  ierr   = VecSet(b,h);CHKERRQ(ierr);
  idx[0] = 0; idx[1] = mx -1;
  v[0]   = v[1] = 0.0;
  ierr   = VecSetValues(b,2,idx,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr   = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr   = VecAssemblyEnd(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
static PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,MatStructure *str,void *ctx)
{
  AppCtx         *user = (AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,mx,xm,xs;
  PetscScalar    v[3],h,xlow,xhigh;
  MatStencil     row,col[3];
  DM             da;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  h    = 1.0/(mx-1);

  for (i=xs; i<xs+xm; i++){
    row.i = i;
    if (i==0 || i==mx-1){
      v[0] = 2.0;
      ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
    } else {
       xlow  = h*(PetscReal)i - .5*h;
       xhigh = xlow + h;
       v[0] = (-1.0 - user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xlow))/h;col[0].i = i-1;
       v[1] = (2.0 + user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xlow) + user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xhigh))/h;col[1].i = row.i;
       v[2] = (-1.0 - user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xhigh))/h;col[2].i = i+1;
      ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
