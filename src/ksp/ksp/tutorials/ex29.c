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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da,0,1,0,1,0,0));
  PetscCall(DMDASetFieldName(da,0,"Pressure"));

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMqq");PetscCall(ierr);
  user.rho    = 1.0;
  PetscCall(PetscOptionsReal("-rho", "The conductivity", "ex29.c", user.rho, &user.rho, NULL));
  user.nu     = 0.1;
  PetscCall(PetscOptionsReal("-nu", "The width of the Gaussian source", "ex29.c", user.nu, &user.nu, NULL));
  bc          = (PetscInt)DIRICHLET;
  PetscCall(PetscOptionsEList("-bc_type","Type of boundary condition","ex29.c",bcTypes,2,bcTypes[0],&bc,NULL));
  user.bcType = (BCType)bc;
  PetscCall(PetscOptionsBool("-testsolver", "Run solver multiple times, useful for performance studies of solver", "ex29.c", testsolver, &testsolver, NULL));
  ierr        = PetscOptionsEnd();PetscCall(ierr);

  PetscCall(KSPSetComputeRHS(ksp,ComputeRHS,&user));
  PetscCall(KSPSetComputeOperators(ksp,ComputeMatrix,&user));
  PetscCall(KSPSetDM(ksp,da));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSolve(ksp,NULL,NULL));

  if (testsolver) {
    PetscCall(KSPGetSolution(ksp,&x));
    PetscCall(KSPGetRhs(ksp,&b));
    KSPSetDMActive(ksp,PETSC_FALSE);
    PetscCall(KSPSolve(ksp,b,x));
    {
#if defined(PETSC_USE_LOG)
      PetscLogStage stage;
#endif
      PetscInt      i,n = 20;

      PetscCall(PetscLogStageRegister("Solve only",&stage));
      PetscCall(PetscLogStagePush(stage));
      for (i=0; i<n; i++) {
        PetscCall(KSPSolve(ksp,b,x));
      }
      PetscCall(PetscLogStagePop());
    }
  }

  PetscCall(DMDestroy(&da));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp,&da));
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0));
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  PetscCall(DMDAVecGetArray(da, b, &array));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)*PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
    }
  }
  PetscCall(DMDAVecRestoreArray(da, b, &array));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
    PetscCall(MatNullSpaceRemove(nullspace,b));
    PetscCall(MatNullSpaceDestroy(&nullspace));
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
  PetscCall(KSPGetDM(ksp,&da));
  centerRho = user->rho;
  PetscCall(DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0));
  Hx        = 1.0 / (PetscReal)(mx-1);
  Hy        = 1.0 / (PetscReal)(my-1);
  HxdHy     = Hx/Hy;
  HydHx     = Hy/Hx;
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      PetscCall(ComputeRho(i, j, mx, my, centerRho, &rho));
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0*rho*(HxdHy + HydHx);
          PetscCall(MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES));
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
          PetscCall(MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES));
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        PetscCall(MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(jac,NULL,"-view_mat"));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-check_matis",&check_matis,NULL));
  if (check_matis) {
    void      (*f)(void);
    Mat       J2;
    MatType   jtype;
    PetscReal nrm;

    PetscCall(MatGetType(jac,&jtype));
    PetscCall(MatConvert(jac,MATIS,MAT_INITIAL_MATRIX,&J2));
    PetscCall(MatViewFromOptions(J2,NULL,"-view_conv"));
    PetscCall(MatConvert(J2,jtype,MAT_INPLACE_MATRIX,&J2));
    PetscCall(MatGetOperation(jac,MATOP_VIEW,&f));
    PetscCall(MatSetOperation(J2,MATOP_VIEW,f));
    PetscCall(MatSetDM(J2,da));
    PetscCall(MatViewFromOptions(J2,NULL,"-view_conv_assembled"));
    PetscCall(MatAXPY(J2,-1.,jac,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(J2,NORM_FROBENIUS,&nrm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error MATIS %g\n",(double)nrm));
    PetscCall(MatViewFromOptions(J2,NULL,"-view_conv_err"));
    PetscCall(MatDestroy(&J2));
  }
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace));
    PetscCall(MatSetNullSpace(J,nullspace));
    PetscCall(MatNullSpaceDestroy(&nullspace));
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

   test:
      suffix: 6
      args: -pc_type svd -pc_svd_monitor ::all

TEST*/
