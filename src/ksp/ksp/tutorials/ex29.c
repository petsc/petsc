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
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetUniformCoordinates(da,0,1,0,1,0,0));
  CHKERRQ(DMDASetFieldName(da,0,"Pressure"));

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMqq");CHKERRQ(ierr);
  user.rho    = 1.0;
  CHKERRQ(PetscOptionsReal("-rho", "The conductivity", "ex29.c", user.rho, &user.rho, NULL));
  user.nu     = 0.1;
  CHKERRQ(PetscOptionsReal("-nu", "The width of the Gaussian source", "ex29.c", user.nu, &user.nu, NULL));
  bc          = (PetscInt)DIRICHLET;
  CHKERRQ(PetscOptionsEList("-bc_type","Type of boundary condition","ex29.c",bcTypes,2,bcTypes[0],&bc,NULL));
  user.bcType = (BCType)bc;
  CHKERRQ(PetscOptionsBool("-testsolver", "Run solver multiple times, useful for performance studies of solver", "ex29.c", testsolver, &testsolver, NULL));
  ierr        = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(KSPSetComputeRHS(ksp,ComputeRHS,&user));
  CHKERRQ(KSPSetComputeOperators(ksp,ComputeMatrix,&user));
  CHKERRQ(KSPSetDM(ksp,da));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPSolve(ksp,NULL,NULL));

  if (testsolver) {
    CHKERRQ(KSPGetSolution(ksp,&x));
    CHKERRQ(KSPGetRhs(ksp,&b));
    KSPSetDMActive(ksp,PETSC_FALSE);
    CHKERRQ(KSPSolve(ksp,b,x));
    {
#if defined(PETSC_USE_LOG)
      PetscLogStage stage;
#endif
      PetscInt      i,n = 20;

      CHKERRQ(PetscLogStageRegister("Solve only",&stage));
      CHKERRQ(PetscLogStagePush(stage));
      for (i=0; i<n; i++) {
        CHKERRQ(KSPSolve(ksp,b,x));
      }
      CHKERRQ(PetscLogStagePop());
    }
  }

  CHKERRQ(DMDestroy(&da));
  CHKERRQ(KSPDestroy(&ksp));
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;
  DM             da;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&da));
  CHKERRQ(DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0));
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  CHKERRQ(DMDAVecGetArray(da, b, &array));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)*PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
    }
  }
  CHKERRQ(DMDAVecRestoreArray(da, b, &array));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
    CHKERRQ(MatNullSpaceRemove(nullspace,b));
    CHKERRQ(MatNullSpaceDestroy(&nullspace));
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
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hx,Hy,HydHx,HxdHy,rho;
  MatStencil     row, col[5];
  DM             da;
  PetscBool      check_matis = PETSC_FALSE;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&da));
  centerRho = user->rho;
  CHKERRQ(DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0));
  Hx        = 1.0 / (PetscReal)(mx-1);
  Hy        = 1.0 / (PetscReal)(my-1);
  HxdHy     = Hx/Hy;
  HydHx     = Hy/Hx;
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      CHKERRQ(ComputeRho(i, j, mx, my, centerRho, &rho));
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0*rho*(HxdHy + HydHx);
          CHKERRQ(MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES));
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
          CHKERRQ(MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES));
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        CHKERRQ(MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(jac,NULL,"-view_mat"));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-check_matis",&check_matis,NULL));
  if (check_matis) {
    void      (*f)(void);
    Mat       J2;
    MatType   jtype;
    PetscReal nrm;

    CHKERRQ(MatGetType(jac,&jtype));
    CHKERRQ(MatConvert(jac,MATIS,MAT_INITIAL_MATRIX,&J2));
    CHKERRQ(MatViewFromOptions(J2,NULL,"-view_conv"));
    CHKERRQ(MatConvert(J2,jtype,MAT_INPLACE_MATRIX,&J2));
    CHKERRQ(MatGetOperation(jac,MATOP_VIEW,&f));
    CHKERRQ(MatSetOperation(J2,MATOP_VIEW,f));
    CHKERRQ(MatSetDM(J2,da));
    CHKERRQ(MatViewFromOptions(J2,NULL,"-view_conv_assembled"));
    CHKERRQ(MatAXPY(J2,-1.,jac,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatNorm(J2,NORM_FROBENIUS,&nrm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error MATIS %g\n",(double)nrm));
    CHKERRQ(MatViewFromOptions(J2,NULL,"-view_conv_err"));
    CHKERRQ(MatDestroy(&J2));
  }
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
    CHKERRQ(MatSetNullSpace(J,nullspace));
    CHKERRQ(MatNullSpaceDestroy(&nullspace));
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
