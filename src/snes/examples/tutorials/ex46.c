static char help[] = "Surface processes in geophysics.\n\n";

/*T
   Concepts: SNES^parallel Surface process example
   Concepts: DMDA^using distributed arrays;
   Concepts: IS coloirng types;
   Processors: n
T*/


#include <petscsnes.h>

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  PassiveReal D;  /* The diffusion coefficient */
  PassiveReal K;  /* The advection coefficient */
  PetscInt    m;  /* Exponent for A */
} AppCtx;

/* 
   User-defined routines
*/
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar**,Mat,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES                   snes;                 /* nonlinear solver */
  AppCtx                 user;                 /* user-defined work context */
  PetscInt               its;                  /* iterations for convergence */
  PetscErrorCode         ierr;
  DM                     da;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Surface Process Problem Options", "SNES");CHKERRQ(ierr);
    user.D = 1.0;
    ierr = PetscOptionsReal("-D", "The diffusion coefficient D", __FILE__, user.D, &user.D, PETSC_NULL);CHKERRQ(ierr);
    user.K = 1.0;
    ierr = PetscOptionsReal("-K", "The advection coefficient K", __FILE__, user.K, &user.K, PETSC_NULL);CHKERRQ(ierr);
    user.m = 1;
    ierr = PetscOptionsInt("-m", "The exponent for A", __FILE__, user.m, &user.m, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local function evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDASetLocalFunction(da, (DMDALocalFunction1) FormFunctionLocal);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,0,0);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "funcU"
PetscScalar funcU(DMDACoor2d *coords)
{
  return coords->x + coords->y;
}

#undef __FUNCT__
#define __FUNCT__ "funcA"
PetscScalar funcA(PetscScalar z, AppCtx *user)
{
  PetscScalar v = 1.0;
  PetscInt    i;

  for(i = 0; i < user->m; ++i) {
    v *= z;
  }
  return v;
}

#undef __FUNCT__
#define __FUNCT__ "funcADer"
PetscScalar funcADer(PetscScalar z, AppCtx *user)
{
  PetscScalar v = 1.0;
  PetscInt    i;

  for(i = 0; i < user->m-1; ++i) {
    v *= z;
  }
  return (PetscScalar)user->m*v;
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/* 
   FormFunctionLocal - Evaluates nonlinear function, F(x).
*/
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,AppCtx *user)
{
  DM             coordDA;
  Vec            coordinates;
  DMDACoor2d     **coords;
  PetscScalar    u, ux, uy, uxx, uyy;
  PetscReal      D, K, hx, hy, hxdhy, hydhx;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  D      = user->D;
  K      = user->K;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  hxdhy  = hx/hy; 
  hydhx  = hy/hx;
  /*
     Compute function over the locally owned part of the grid
  */
  ierr = DMDAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DMDAGetCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        f[j][i] = x[j][i];
      } else {
        u       = x[j][i];
        ux      = (x[j][i+1] - x[j][i])/hx;
        uy      = (x[j+1][i] - x[j][i])/hy;
        uxx     = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx;
        uyy     = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
        f[j][i] = D*(uxx + uyy) - (K*funcA(x[j][i], user)*sqrt(ux*ux + uy*uy) + funcU(&coords[j][i]))*hx*hy;
        if (PetscIsInfOrNanScalar(f[j][i])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FP, "Invalid residual: %g", PetscRealPart(f[j][i]));
      }
    }
  }
  ierr = DMDAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscLogFlops(11*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
   FormJacobianLocal - Evaluates Jacobian matrix.
*/
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,PetscScalar **x,Mat jac,AppCtx *user)
{
  MatStencil     col[5], row;
  PetscScalar    D, K, A, v[5], hx, hy, hxdhy, hydhx, ux, uy;
  PetscReal      normGradZ;
  PetscInt       i, j,k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  D      = user->D;
  K      = user->K;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  hxdhy  = hx/hy; 
  hydhx  = hy/hx;

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
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      row.j = j; row.i = i;
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        /* boundary points */
        v[0] = 1.0;
        ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        /* interior grid points */
        ux        = (x[j][i+1] - x[j][i])/hx;
        uy        = (x[j+1][i] - x[j][i])/hy;
        normGradZ = PetscRealPart(sqrt(ux*ux + uy*uy));
        //PetscPrintf(PETSC_COMM_SELF, "i: %d j: %d normGradZ: %g\n", i, j, normGradZ);
        if (normGradZ < 1.0e-8) {
          normGradZ = 1.0e-8;
        }
        A         = funcA(x[j][i], user);

        v[0] = -D*hxdhy;                                                                          col[0].j = j - 1; col[0].i = i;
        v[1] = -D*hydhx;                                                                          col[1].j = j;     col[1].i = i-1;
        v[2] = D*2.0*(hydhx + hxdhy) + K*(funcADer(x[j][i], user)*normGradZ - A/normGradZ)*hx*hy; col[2].j = row.j; col[2].i = row.i;
        v[3] = -D*hydhx + K*A*hx*hy/(2.0*normGradZ);                                              col[3].j = j;     col[3].i = i+1;
        v[4] = -D*hxdhy + K*A*hx*hy/(2.0*normGradZ);                                              col[4].j = j + 1; col[4].i = i;
        for(k = 0; k < 5; ++k) {
          if (PetscIsInfOrNanScalar(v[k])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FP, "Invalid residual: %g", PetscRealPart(v[k]));
        }
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
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
