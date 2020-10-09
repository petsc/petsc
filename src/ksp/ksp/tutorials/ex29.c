/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Added at the request of Marc Garbey.

Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-x^2/\nu} e^{-y^2/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions

This uses multigrid to solve the linear system
*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscReal rho;
  PetscReal nu;
  BCType    bcType;
} UserContext;

int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  UserContext    user;
  const char     *bcTypes[2] = {"dirichlet","neumann"};
  PetscErrorCode ierr;
  PetscInt       bc;
  Vec            b,x;
  PetscBool      testsolver = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0,1,0,1,0,0);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"Pressure");CHKERRQ(ierr);

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMqq");CHKERRQ(ierr);
  user.rho    = 1.0;
  ierr        = PetscOptionsReal("-rho", "The conductivity", "ex29.c", user.rho, &user.rho, NULL);CHKERRQ(ierr);
  user.nu     = 0.1;
  ierr        = PetscOptionsReal("-nu", "The width of the Gaussian source", "ex29.c", user.nu, &user.nu, NULL);CHKERRQ(ierr);
  bc          = (PetscInt)DIRICHLET;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex29.c",bcTypes,2,bcTypes[0],&bc,NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr        = PetscOptionsBool("-testsolver", "Run solver multiple times, useful for performance studies of solver", "ex29.c", testsolver, &testsolver, NULL);CHKERRQ(ierr);
  ierr        = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,&user);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,da);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);

  if (testsolver) {
    ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
    ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);
    KSPSetDMActive(ksp,PETSC_FALSE);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    {
#if defined(PETSC_USE_LOG)
      PetscLogStage stage;
#endif
      PetscInt      i,n = 20;

      ierr = PetscLogStageRegister("Solve only",&stage);CHKERRQ(ierr);
      ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
      }
      ierr = PetscLogStagePop();CHKERRQ(ierr);
    }
  }

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;
  DM             da;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)*PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscErrorCode ComputeRho(PetscInt i, PetscInt j, PetscInt mx, PetscInt my, PetscReal centerRho, PetscReal *rho)
{
  PetscFunctionBeginUser;
  if ((i > mx/3.0) && (i < 2.0*mx/3.0) && (j > my/3.0) && (j < 2.0*my/3.0)) {
    *rho = centerRho;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscReal      centerRho;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hx,Hy,HydHx,HxdHy,rho;
  MatStencil     row, col[5];
  DM             da;
  PetscBool      check_matis = PETSC_FALSE;

  PetscFunctionBeginUser;
  ierr      = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  centerRho = user->rho;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx-1);
  Hy        = 1.0 / (PetscReal)(my-1);
  HxdHy     = Hx/Hy;
  HydHx     = Hy/Hx;
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      ierr  = ComputeRho(i, j, mx, my, centerRho, &rho);CHKERRQ(ierr);
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0*rho*(HxdHy + HydHx);
          ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else if (user->bcType == NEUMANN) {
          PetscInt numx = 0, numy = 0, num = 0;
          if (j!=0) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j-1;
            numy++; num++;
          }
          if (i!=0) {
            v[num] = -rho*HydHx;              col[num].i = i-1; col[num].j = j;
            numx++; num++;
          }
          if (i!=mx-1) {
            v[num] = -rho*HydHx;              col[num].i = i+1; col[num].j = j;
            numx++; num++;
          }
          if (j!=my-1) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j+1;
            numy++; num++;
          }
          v[num] = numx*rho*HydHx + numy*rho*HxdHy; col[num].i = i;   col[num].j = j;
          num++;
          ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(jac,NULL,"-view_mat");CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-check_matis",&check_matis,NULL);CHKERRQ(ierr);
  if (check_matis) {
    void      (*f)(void);
    Mat       J2;
    MatType   jtype;
    PetscReal nrm;

    ierr = MatGetType(jac,&jtype);CHKERRQ(ierr);
    ierr = MatConvert(jac,MATIS,MAT_INITIAL_MATRIX,&J2);CHKERRQ(ierr);
    ierr = MatViewFromOptions(J2,NULL,"-view_conv");CHKERRQ(ierr);
    ierr = MatConvert(J2,jtype,MAT_INPLACE_MATRIX,&J2);CHKERRQ(ierr);
    ierr = MatGetOperation(jac,MATOP_VIEW,&f);CHKERRQ(ierr);
    ierr = MatSetOperation(J2,MATOP_VIEW,f);CHKERRQ(ierr);
    ierr = MatSetDM(J2,da);CHKERRQ(ierr);
    ierr = MatViewFromOptions(J2,NULL,"-view_conv_assembled");CHKERRQ(ierr);
    ierr = MatAXPY(J2,-1.,jac,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(J2,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error MATIS %g\n",(double)nrm);CHKERRQ(ierr);
    ierr = MatViewFromOptions(J2,NULL,"-view_conv_err");CHKERRQ(ierr);
    ierr = MatDestroy(&J2);CHKERRQ(ierr);
  }
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*TEST

   test:
      args: -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -da_refine 8 -ksp_rtol 1.e-3

   test:
      suffix: 2
      args: -bc_type neumann -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -da_refine 8 -mg_coarse_pc_factor_shift_type nonzero
      requires: !single

   test:
      suffix: telescope
      nsize: 4
      args: -ksp_monitor_short -da_grid_x 257 -da_grid_y 257 -pc_type mg -pc_mg_galerkin pmat -pc_mg_levels 4 -ksp_type richardson -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_ignore_kspcomputeoperators -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type chebyshev -mg_coarse_telescope_mg_levels_pc_type jacobi -mg_coarse_pc_telescope_reduction_factor 4

   test:
      suffix: 3
      args: -ksp_view -da_refine 2 -pc_type mg -pc_mg_distinct_smoothup -mg_levels_up_pc_type jacobi

   test:
      suffix: 4
      args: -ksp_view -da_refine 2 -pc_type mg -pc_mg_distinct_smoothup -mg_levels_up_ksp_max_it 3 -mg_levels_ksp_max_it 4

   test:
      suffix: 5
      nsize: 2
      requires: hypre !complex
      args: -pc_type mg  -da_refine 2 -ksp_monitor  -matptap_via hypre -pc_mg_galerkin both

TEST*/
