/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by the
    partial differential equation

            -Laplacian(u) - lambda * exp(u) = 0,  0 < x,y,z < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1, z = 0, z = 1

    A finite difference approximation with the usual 7-point stencil
    is used to discretize the boundary value problem to obtain a
    nonlinear system of equations. The problem is solved in a 3D
    rectangular domain, using distributed arrays (DAs) to partition
    the parallel grid.

  ------------------------------------------------------------------------- */

#include "Bratu3Dimpl.h"

PetscErrorCode FormInitGuess(DM da, Vec X, Params *p)
{
  PetscInt       i,j,k,Mx,My,Mz,xs,ys,zs,xm,ym,zm;
  PetscReal      lambda,temp1,hx,hy,hz,tempk,tempj;
  PetscScalar    ***x;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(da,PETSC_IGNORE,
                     &Mx,&My,&Mz,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE);

  lambda = p->lambda_;
  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  hz     = 1.0/(PetscReal)(Mz-1);
  temp1  = lambda/(lambda + 1.0);

  /*
    Get a pointer to vector data.

    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation
      dependent.

    - You MUST call VecRestoreArray() when you no longer need access
      to the array.
  */
  PetscCall(DMDAVecGetArray(da,X,&x));

  /*
    Get local grid boundaries (for 3-dimensional DMDA):

    - xs, ys, zs: starting grid indices (no ghost points)

    - xm, ym, zm: widths of local grid (no ghost points)
  */
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));

  /*
    Compute initial guess over the locally owned part of the grid
  */
  for (k=zs; k<zs+zm; k++) {
    tempk = (PetscReal)(PetscMin(k,Mz-k-1))*hz;
    for (j=ys; j<ys+ym; j++) {
      tempj = PetscMin((PetscReal)(PetscMin(j,My-j-1))*hy,tempk);
      for (i=xs; i<xs+xm; i++) {
        if (i == 0 || j == 0 || k == 0 || i == Mx-1 || j == My-1 || k == Mz-1) {
          /* boundary conditions are all zero Dirichlet */
          x[k][j][i] = 0.0;
        } else {
          x[k][j][i] = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,Mx-i-1))*hx,tempj));
        }
      }
    }
  }

  /*
    Restore vector
  */
  PetscCall(DMDAVecRestoreArray(da,X,&x));

  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction(DM da, Vec X, Vec F, Params *p)
{
  PetscInt       i,j,k,Mx,My,Mz,xs,ys,zs,xm,ym,zm;
  PetscReal      two = 2.0,lambda,hx,hy,hz,hxhzdhy,hyhzdhx,hxhydhz,sc;
  PetscScalar    u_north,u_south,u_east,u_west,u_up,u_down,u,u_xx,u_yy,u_zz,***x,***f;
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(da,PETSC_IGNORE,
                     &Mx,&My,&Mz,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE);

  lambda = p->lambda_;
  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  hz     = 1.0/(PetscReal)(Mz-1);
  sc     = hx*hy*hz*lambda;
  hxhzdhy = hx*hz/hy;
  hyhzdhx = hy*hz/hx;
  hxhydhz = hx*hy/hz;

  /*

   */
  PetscCall(DMGetLocalVector(da,&localX));

  /*
    Scatter ghost points to local vector,using the 2-step process
    DMGlobalToLocalBegin(),DMGlobalToLocalEnd().  By placing code
    between these two statements, computations can be done while
    messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));

  /*
    Get pointers to vector data.
  */
  PetscCall(DMDAVecGetArray(da,localX,&x));
  PetscCall(DMDAVecGetArray(da,F,&f));

  /*
    Get local grid boundaries.
  */
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));

  /*
    Compute function over the locally owned part of the grid.
  */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (i == 0 || j == 0 || k == 0 || i == Mx-1 || j == My-1 || k == Mz-1) {
          /* boundary points */
          f[k][j][i] = x[k][j][i] - 0.0;
        } else {
          /* interior grid points */
          u           = x[k][j][i];
          u_east      = x[k][j][i+1];
          u_west      = x[k][j][i-1];
          u_north     = x[k][j+1][i];
          u_south     = x[k][j-1][i];
          u_up        = x[k+1][j][i];
          u_down      = x[k-1][j][i];
          u_xx        = (-u_east + two*u - u_west)*hyhzdhx;
          u_yy        = (-u_north + two*u - u_south)*hxhzdhy;
          u_zz        = (-u_up + two*u - u_down)*hxhydhz;
          f[k][j][i]  = u_xx + u_yy + u_zz - sc*PetscExpScalar(u);
        }
      }
    }
  }

  /*
    Restore vectors.
  */
  PetscCall(DMDAVecRestoreArray(da,F,&f));
  PetscCall(DMDAVecRestoreArray(da,localX,&x));
  PetscCall(DMRestoreLocalVector(da,&localX));
  PetscCall(PetscLogFlops(11.0*ym*xm));

  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian(DM da, Vec X, Mat J, Params *p)
{
  PetscInt       i,j,k,Mx,My,Mz,xs,ys,zs,xm,ym,zm;
  PetscReal      lambda,hx,hy,hz,hxhzdhy,hyhzdhx,hxhydhz,sc;
  PetscScalar    v[7],***x;
  MatStencil     col[7],row;
  Vec            localX;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(da,PETSC_IGNORE,
                     &Mx,&My,&Mz,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE);

  lambda = p->lambda_;
  hx     = 1.0/(PetscReal)(Mx-1);
  hy     = 1.0/(PetscReal)(My-1);
  hz     = 1.0/(PetscReal)(Mz-1);
  sc     = hx*hy*hz*lambda;
  hxhzdhy = hx*hz/hy;
  hyhzdhx = hy*hz/hx;
  hxhydhz = hx*hy/hz;

  /*

   */
  PetscCall(DMGetLocalVector(da,&localX));

  /*
    Scatter ghost points to local vector, using the 2-step process
    DMGlobalToLocalBegin(), DMGlobalToLocalEnd().  By placing code
    between these two statements, computations can be done while
    messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));

  /*
    Get pointer to vector data.
  */
  PetscCall(DMDAVecGetArray(da,localX,&x));

  /*
    Get local grid boundaries.
  */
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));

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
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row.k = k; row.j = j; row.i = i;
        /* boundary points */
        if (i == 0 || j == 0 || k == 0|| i == Mx-1 || j == My-1 || k == Mz-1) {
          v[0] = 1.0;
          PetscCall(MatSetValuesStencil(J,1,&row,1,&row,v,INSERT_VALUES));
        } else {
        /* interior grid points */
          v[0] = -hxhydhz; col[0].k=k-1;col[0].j=j;  col[0].i = i;
          v[1] = -hxhzdhy; col[1].k=k;  col[1].j=j-1;col[1].i = i;
          v[2] = -hyhzdhx; col[2].k=k;  col[2].j=j;  col[2].i = i-1;
          v[3] = 2.0*(hyhzdhx+hxhzdhy+hxhydhz)-sc*PetscExpScalar(x[k][j][i]);col[3].k=row.k;col[3].j=row.j;col[3].i = row.i;
          v[4] = -hyhzdhx; col[4].k=k;  col[4].j=j;  col[4].i = i+1;
          v[5] = -hxhzdhy; col[5].k=k;  col[5].j=j+1;col[5].i = i;
          v[6] = -hxhydhz; col[6].k=k+1;col[6].j=j;  col[6].i = i;
          PetscCall(MatSetValuesStencil(J,1,&row,7,col,v,INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,localX,&x));
  PetscCall(DMRestoreLocalVector(da,&localX));

  /*
    Assemble matrix, using the 2-step process: MatAssemblyBegin(),
    MatAssemblyEnd().
  */
  PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));

  /*
    Tell the matrix we will never add a new nonzero location to the
    matrix. If we do, it will generate an error.
  */
  PetscCall(MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));

  PetscFunctionReturn(0);
}
