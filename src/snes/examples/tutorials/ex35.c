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
            -da_refine n  -fas_coarse_snes_type ls -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly

*/

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
*/
#include <petscdmda.h>
#include <petscsnes.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormMatrix(DM,Mat);
extern PetscErrorCode MyDMComputeFunction(DM,Vec,Vec);
extern PetscErrorCode MyDMComputeJacobian(DM,Vec,Mat,Mat,MatStructure*);
extern PetscErrorCode NonlinearGS(SNES,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES                   snes;                         /* nonlinear solver */
  SNES                   psnes;                        /* nonlinear Gauss-Seidel approximate solver */
  Vec                    x,b;                            /* solution vector */
  PetscInt               its;                          /* iterations for convergence */
  PetscErrorCode         ierr;
  DM                     da;
  PetscBool              use_ngs = PETSC_FALSE;         /* use the nonlinear Gauss-Seidel approximate solver */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_ngs",&use_ngs,0);CHKERRQ(ierr);

  if (use_ngs) {
    ierr = SNESGetPC(snes,&psnes);CHKERRQ(ierr);
    ierr = SNESSetType(psnes,SNESSHELL);CHKERRQ(ierr);
    ierr = SNESShellSetSolve(psnes,NonlinearGS);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  if (use_ngs) {
    ierr = SNESShellSetContext(psnes,da);CHKERRQ(ierr);
  }
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecSetRandom(b,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMSetFunction(da,MyDMComputeFunction);CHKERRQ(ierr);
  ierr = DMSetJacobian(da,MyDMComputeJacobian);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,b,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MyDMComputeFunction"
PetscErrorCode MyDMComputeFunction(DM dm,Vec x,Vec F)
{
  PetscErrorCode ierr;
  Mat            J;

  PetscFunctionBegin;
  ierr = DMGetApplicationContext(dm,&J);CHKERRQ(ierr);
  if (!J) {
    ierr = DMCreateMatrix(dm,MATAIJ,&J);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)J,"DM",(PetscObject)PETSC_NULL);CHKERRQ(ierr);
    ierr = FormMatrix(dm,J);CHKERRQ(ierr);
    ierr = DMSetApplicationContext(dm,J);CHKERRQ(ierr);
    ierr = DMSetApplicationContextDestroy(dm,(PetscErrorCode (*)(void**))MatDestroy);CHKERRQ(ierr);
  }
  ierr = MatMult(J,x,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyDMComputeJacobian"
PetscErrorCode MyDMComputeJacobian(DM dm,Vec x,Mat J,Mat Jp,MatStructure *str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FormMatrix(dm,Jp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormMatrix"
PetscErrorCode FormMatrix(DM da,Mat jac)
{
  PetscErrorCode ierr;
  PetscInt       i,j,nrows = 0;
  MatStencil     col[5],row,*rows;
  PetscScalar    v[5],hx,hy,hxdhy,hydhx;
  DMDALocalInfo  info;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx     = 1.0/(PetscReal)(info.mx-1);
  hy     = 1.0/(PetscReal)(info.my-1);
  hxdhy  = hx/hy;
  hydhx  = hy/hx;

  ierr = PetscMalloc(info.ym*info.xm*sizeof(MatStencil),&rows);CHKERRQ(ierr);
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
        v[0] = 2.0*(hydhx + hxdhy);
        ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        rows[nrows].i = i;
        rows[nrows++].j = j;
      } else {
      /* interior grid points */
        v[0] = -hxdhy;                                           col[0].j = j - 1; col[0].i = i;
        v[1] = -hydhx;                                           col[1].j = j;     col[1].i = i-1;
        v[2] = 2.0*(hydhx + hxdhy);                              col[2].j = row.j; col[2].i = row.i;
        v[3] = -hydhx;                                           col[3].j = j;     col[3].i = i+1;
        v[4] = -hxdhy;                                           col[4].j = j + 1; col[4].i = i;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatZeroRowsColumnsStencil(jac,nrows,rows,2.0*(hydhx + hxdhy),PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "NonlinearGS"
/*
      Applies some sweeps on nonlinear Gauss-Seidel on each process

 */
PetscErrorCode NonlinearGS(SNES snes,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,its,l;
  PetscErrorCode ierr;
  PetscReal      hx,hy,hxdhy,hydhx;
  PetscScalar    **x,F,J,u,uxx,uyy;
  DM             da;
  Vec            localX;

  PetscFunctionBegin;
  ierr = SNESGetTolerances(snes,PETSC_NULL,PETSC_NULL,PETSC_NULL,&its,PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESShellGetContext(snes,(void**)&da);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  hxdhy  = hx/hy;
  hydhx  = hy/hx;


  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);

  for (l=0; l<its; l++) {

    ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    /*
     Get a pointer to vector data.
     - For default PETSc vectors, VecGetArray() returns a pointer to
     the data array.  Otherwise, the routine is implementation dependent.
     - You MUST call VecRestoreArray() when you no longer need access to
     the array.
     */
    ierr = DMDAVecGetArray(da,localX,&x);CHKERRQ(ierr);

    /*
     Get local grid boundaries (for 2-dimensional DMDA):
     xs, ys   - starting grid indices (no ghost points)
     xm, ym   - widths of local grid (no ghost points)

     */
    ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
          /* boundary conditions are all zero Dirichlet */
          x[j][i] = 0.0;
        } else {
          u       = x[j][i];

          uxx     = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx;
          uyy     = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
          F        = uxx + uyy;
          J       = 2.0*(hydhx + hxdhy);
          u       = u - F/J;

          x[j][i] = u;
        }
      }
    }

    /*
     Restore vector
     */
    ierr = DMDAVecRestoreArray(da,localX,&x);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
