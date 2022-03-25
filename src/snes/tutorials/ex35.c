static const char help[] = "-Laplacian u = b as a nonlinear problem.\n\n";

/*T
   Concepts: SNES^parallel Bratu example
   Concepts: DMDA^using distributed arrays;
   Concepts: IS coloirng types;
   Processors: n
T*/

/*

    The linear and nonlinear versions of these should give almost identical results on this problem

    Richardson
      Nonlinear:
        -snes_rtol 1.e-12 -snes_monitor -snes_type nrichardson -snes_linesearch_monitor

      Linear:
        -snes_rtol 1.e-12 -snes_monitor -ksp_rtol 1.e-12  -ksp_monitor -ksp_type richardson -pc_type none -ksp_richardson_self_scale -info

    GMRES
      Nonlinear:
       -snes_rtol 1.e-12 -snes_monitor  -snes_type ngmres

      Linear:
       -snes_rtol 1.e-12 -snes_monitor  -ksp_type gmres -ksp_monitor -ksp_rtol 1.e-12 -pc_type none

    CG
       Nonlinear:
            -snes_rtol 1.e-12 -snes_monitor  -snes_type ncg -snes_linesearch_monitor

       Linear:
             -snes_rtol 1.e-12 -snes_monitor  -ksp_type cg -ksp_monitor -ksp_rtol 1.e-12 -pc_type none

    Multigrid
       Linear:
          1 level:
            -snes_rtol 1.e-12 -snes_monitor  -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type none -mg_levels_ksp_monitor
            -mg_levels_ksp_richardson_self_scale -ksp_type richardson -ksp_monitor -ksp_rtol 1.e-12  -ksp_monitor_true_residual

          n levels:
            -da_refine n

       Nonlinear:
         1 level:
           -snes_rtol 1.e-12 -snes_monitor  -snes_type fas -fas_levels_snes_monitor

          n levels:
            -da_refine n  -fas_coarse_snes_type newtonls -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly

*/

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormMatrix(DM,Mat);
extern PetscErrorCode MyComputeFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode MyComputeJacobian(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode NonlinearGS(SNES,Vec);

int main(int argc,char **argv)
{
  SNES           snes;                                 /* nonlinear solver */
  SNES           psnes;                                /* nonlinear Gauss-Seidel approximate solver */
  Vec            x,b;                                  /* solution vector */
  PetscInt       its;                                  /* iterations for convergence */
  DM             da;
  PetscBool      use_ngs_as_npc = PETSC_FALSE;                /* use the nonlinear Gauss-Seidel approximate solver */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-use_ngs_as_npc",&use_ngs_as_npc,0));

  if (use_ngs_as_npc) {
    PetscCall(SNESGetNPC(snes,&psnes));
    PetscCall(SNESSetType(psnes,SNESSHELL));
    PetscCall(SNESShellSetSolve(psnes,NonlinearGS));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(SNESSetDM(snes,da));
  if (use_ngs_as_npc) {
    PetscCall(SNESShellSetContext(psnes,da));
  }
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(DMCreateGlobalVector(da,&b));
  PetscCall(VecSet(b,1.0));

  PetscCall(SNESSetFunction(snes,NULL,MyComputeFunction,NULL));
  PetscCall(SNESSetJacobian(snes,NULL,NULL,MyComputeJacobian,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSolve(snes,b,x));
  PetscCall(SNESGetIterationNumber(snes,&its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------- */
PetscErrorCode MyComputeFunction(SNES snes,Vec x,Vec F,void *ctx)
{
  Mat            J;
  DM             dm;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetApplicationContext(dm,&J));
  if (!J) {
    PetscCall(DMSetMatType(dm,MATAIJ));
    PetscCall(DMCreateMatrix(dm,&J));
    PetscCall(MatSetDM(J, NULL));
    PetscCall(FormMatrix(dm,J));
    PetscCall(DMSetApplicationContext(dm,J));
    PetscCall(DMSetApplicationContextDestroy(dm,(PetscErrorCode (*)(void**))MatDestroy));
  }
  PetscCall(MatMult(J,x,F));
  PetscFunctionReturn(0);
}

PetscErrorCode MyComputeJacobian(SNES snes,Vec x,Mat J,Mat Jp,void *ctx)
{
  DM             dm;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(FormMatrix(dm,Jp));
  PetscFunctionReturn(0);
}

PetscErrorCode FormMatrix(DM da,Mat jac)
{
  PetscInt       i,j,nrows = 0;
  MatStencil     col[5],row,*rows;
  PetscScalar    v[5],hx,hy,hxdhy,hydhx;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da,&info));
  hx    = 1.0/(PetscReal)(info.mx-1);
  hy    = 1.0/(PetscReal)(info.my-1);
  hxdhy = hx/hy;
  hydhx = hy/hx;

  PetscCall(PetscMalloc1(info.ym*info.xm,&rows));
  /*
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      row.j = j; row.i = i;
      /* boundary points */
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
        v[0]            = 2.0*(hydhx + hxdhy);
        PetscCall(MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES));
        rows[nrows].i   = i;
        rows[nrows++].j = j;
      } else {
        /* interior grid points */
        v[0] = -hxdhy;                                           col[0].j = j - 1; col[0].i = i;
        v[1] = -hydhx;                                           col[1].j = j;     col[1].i = i-1;
        v[2] = 2.0*(hydhx + hxdhy);                              col[2].j = row.j; col[2].i = row.i;
        v[3] = -hydhx;                                           col[3].j = j;     col[3].i = i+1;
        v[4] = -hxdhy;                                           col[4].j = j + 1; col[4].i = i;
        PetscCall(MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES));
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatZeroRowsColumnsStencil(jac,nrows,rows,2.0*(hydhx + hxdhy),NULL,NULL));
  PetscCall(PetscFree(rows));
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  PetscCall(MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
      Applies some sweeps on nonlinear Gauss-Seidel on each process

 */
PetscErrorCode NonlinearGS(SNES snes,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,its,l;
  PetscReal      hx,hy,hxdhy,hydhx;
  PetscScalar    **x,F,J,u,uxx,uyy;
  DM             da;
  Vec            localX;

  PetscFunctionBeginUser;
  PetscCall(SNESGetTolerances(snes,NULL,NULL,NULL,&its,NULL));
  PetscCall(SNESShellGetContext(snes,&da));

  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx    = 1.0/(PetscReal)(Mx-1);
  hy    = 1.0/(PetscReal)(My-1);
  hxdhy = hx/hy;
  hydhx = hy/hx;

  PetscCall(DMGetLocalVector(da,&localX));

  for (l=0; l<its; l++) {

    PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
    PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));
    /*
     Get a pointer to vector data.
     - For default PETSc vectors, VecGetArray() returns a pointer to
     the data array.  Otherwise, the routine is implementation dependent.
     - You MUST call VecRestoreArray() when you no longer need access to
     the array.
     */
    PetscCall(DMDAVecGetArray(da,localX,&x));

    /*
     Get local grid boundaries (for 2-dimensional DMDA):
     xs, ys   - starting grid indices (no ghost points)
     xm, ym   - widths of local grid (no ghost points)

     */
    PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
          /* boundary conditions are all zero Dirichlet */
          x[j][i] = 0.0;
        } else {
          u   = x[j][i];
          uxx = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx;
          uyy = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
          F   = uxx + uyy;
          J   = 2.0*(hydhx + hxdhy);
          u   = u - F/J;

          x[j][i] = u;
        }
      }
    }

    /*
     Restore vector
     */
    PetscCall(DMDAVecRestoreArray(da,localX,&x));
    PetscCall(DMLocalToGlobalBegin(da,localX,INSERT_VALUES,X));
    PetscCall(DMLocalToGlobalEnd(da,localX,INSERT_VALUES,X));
  }
  PetscCall(DMRestoreLocalVector(da,&localX));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -snes_monitor_short -snes_type nrichardson
      requires: !single

   test:
      suffix: 2
      args: -snes_monitor_short -ksp_monitor_short -ksp_type richardson -pc_type none -ksp_richardson_self_scale
      requires: !single

   test:
      suffix: 3
      args: -snes_monitor_short -snes_type ngmres

   test:
      suffix: 4
      args: -snes_monitor_short -ksp_type gmres -ksp_monitor_short -pc_type none

   test:
      suffix: 5
      args: -snes_monitor_short -snes_type ncg

   test:
      suffix: 6
      args: -snes_monitor_short -ksp_type cg -ksp_monitor_short -pc_type none

   test:
      suffix: 7
      args: -da_refine 2 -snes_monitor_short -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type none -mg_levels_ksp_monitor_short -mg_levels_ksp_richardson_self_scale -ksp_type richardson -ksp_monitor_short
      requires: !single

   test:
      suffix: 8
      args: -da_refine 2 -snes_monitor_short -snes_type fas -fas_levels_snes_monitor_short -fas_coarse_snes_type newtonls -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly -snes_type fas -snes_rtol 1.e-5

   test:
      suffix: 9
      args: -snes_monitor_short -ksp_type gmres -ksp_monitor_short -pc_type none -snes_type newtontrdc

   test:
      suffix: 10
      args: -snes_monitor_short -ksp_type gmres -ksp_monitor_short -pc_type none -snes_type newtontrdc -snes_trdc_use_cauchy false

TEST*/
