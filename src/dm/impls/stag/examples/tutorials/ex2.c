static char help[] = "Solve a toy 2D problem on a staggered grid\n\n";
/*

  To demonstrate the basic functionality of DMStag, solves an isoviscous
  incompressible Stokes problem on a rectangular 2D domain, using a manufactured
  solution.

  u_xx + u_yy - p_x = f^x
  v_xx + v_yy - p_y = f^y
  u_x + v_y         = g

  g is 0 in the physical case.

  Boundary conditions give prescribed flow perpendicular to the boundaries,
  and zero derivative perpendicular to them (free slip).

  Use the -pinpressure option to fix a pressure node, instead of providing
  a constant-pressure nullspace. This allows use of direct solvers, e.g. to
  use UMFPACK,

     ./ex2 -pinpressure 1 -pc_type lu -pc_factor_mat_solver_type umfpack

  This example demonstrates the use of DMProduct to efficiently store coordinates
  on an orthogonal grid.

*/
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h> /* Includes petscdmproduct.h */

/* Shorter, more convenient names for DMStagStencilLocation entries */
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

static PetscErrorCode CreateSystem(DM,Mat*,Vec*,PetscBool);
static PetscErrorCode CreateReferenceSolution(DM,Vec*);
static PetscErrorCode AttachNullspace(DM,Mat);
static PetscErrorCode CheckSolution(Vec,Vec);

/* Manufactured solution. Chosen to be higher order than can be solved exactly,
and to have a zero derivative for flow parallel to the boundaries. That is,
d(ux)/dy = 0 at the top and bottom boundaries, and d(uy)/dx = 0 at the right
and left boundaries. */
static PetscScalar uxRef(PetscScalar x,PetscScalar y ){return 0.0*x + y*y - 2.0*y*y*y + y*y*y*y;}      /* no x-dependence  */
static PetscScalar uyRef(PetscScalar x,PetscScalar y) {return x*x - 2.0*x*x*x + x*x*x*x + 0.0*y;}      /* no y-dependence  */
static PetscScalar pRef (PetscScalar x,PetscScalar y) {return -1.0*(x-0.5) + -3.0/2.0*y*y + 0.5;}    /* zero integral    */
static PetscScalar fx   (PetscScalar x,PetscScalar y) {return 0.0*x + 2.0 -12.0*y + 12.0*y*y + 1.0;} /* no x-dependence  */
static PetscScalar fy   (PetscScalar x,PetscScalar y) {return 2.0 -12.0*x + 12.0*x*x + 3.0*y;}
static PetscScalar g    (PetscScalar x,PetscScalar y) {return 0.0*x*y;}                              /* identically zero */

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dmSol;
  Vec            sol,solRef,rhs;
  Mat            A;
  KSP            ksp;
  PC             pc;
  PetscBool      pinPressure;

  /* Initialize PETSc and process command line arguments */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  pinPressure = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-pinpressure",&pinPressure,NULL);CHKERRQ(ierr);

  /* Create 2D DMStag for the solution, and set up. */
  {
    const PetscInt dof0 = 0, dof1 = 1,dof2 = 1; /* 1 dof on each edge and element center */
    const PetscInt stencilWidth = 1;
    ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,7,9,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,DMSTAG_STENCIL_BOX,stencilWidth,NULL,NULL,&dmSol);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmSol);CHKERRQ(ierr);
    ierr = DMSetUp(dmSol);CHKERRQ(ierr);
  }

  /* Define uniform coordinates as a product of 1D arrays */
  ierr = DMStagSetUniformCoordinatesProduct(dmSol,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);

  /* Compute (manufactured) reference solution */
  ierr = CreateReferenceSolution(dmSol,&solRef);CHKERRQ(ierr);

  /* Assemble system */
  ierr = CreateSystem(dmSol,&A,&rhs,pinPressure);CHKERRQ(ierr);

  /* Attach a constant-pressure nullspace to the operator
  (logically, this should be in CreateSystem, but we separate it here to highlight
   the features of DMStag exposed, in particular DMStagMigrateVec() ) */
  if (!pinPressure) {
    ierr = AttachNullspace(dmSol,A);CHKERRQ(ierr);
  }

  /* Solve, using the default FieldSplit (Approximate Block Factorization) Preconditioner
     This is not intended to be an example of a good solver!  */
  ierr = DMCreateGlobalVector(dmSol,&sol);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPFGMRES);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
  ierr = PCFieldSplitSetDetectSaddlePoint(pc,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);

  /* Check Solution */
  ierr = CheckSolution(sol,solRef);CHKERRQ(ierr);

  /* Clean up and finalize PETSc */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  ierr = VecDestroy(&solRef);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dmSol);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*
Note: this system is not well-scaled! Generally one would adjust the equations
 to try to get matrix entries to be of comparable order, regardless of grid spacing
 or choice of coefficients.
*/
static PetscErrorCode CreateSystem(DM dmSol,Mat *pA,Vec *pRhs, PetscBool pinPressure)
{
  PetscErrorCode ierr;
  PetscInt       N[2];
  PetscInt       ex,ey,startx,starty,nx,ny;
  PetscInt       iprev,icenter,inext;
  Mat            A;
  Vec            rhs;
  PetscReal      hx,hy;
  PetscScalar    **cArrX,**cArrY;

  /* Here, we showcase two different methods for manipulating local vector entries.
     One can use DMStagStencil objects with DMStagVecSetValuesStencil(),
     making sure to call VecAssemble[Begin/End]() after all values are set.
     Alternately, one can use DMStagVecGetArray[Read]() and DMStagVecRestoreArray[Read]().
     The first approach is used to build the rhs, and the second is used to
     obtain coordinate values. Working with the array is almost certainly more efficient,
     but only allows setting local entries, requires understanding which "slot" to use,
     and doesn't correspond as precisely to the matrix assembly process using DMStagStencil objects */

  PetscFunctionBeginUser;
  ierr = DMCreateMatrix(dmSol,pA);CHKERRQ(ierr);
  A = *pA;
  ierr = DMCreateGlobalVector(dmSol,pRhs);CHKERRQ(ierr);
  rhs = *pRhs;
  ierr = DMStagGetCorners(dmSol,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmSol,&N[0],&N[1],NULL);CHKERRQ(ierr);
  hx = 1.0/N[0]; hy = 1.0/N[1];
  ierr = DMStagGetProductCoordinateArraysRead(dmSol,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,RIGHT,&inext);CHKERRQ(ierr);

  /* Loop over all local elements. Note that it may be more efficient in real
     applications to loop over each boundary separately */
  for (ey = starty; ey<starty+ny; ++ey) { /* With DMStag, always iterate x fastest, y second fastest, z slowest */
    for (ex = startx; ex<startx+nx; ++ex) {

      if (ex == N[0]-1) {
        /* Right Boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = RIGHT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = uxRef(cArrX[ex][inext],cArrY[ey][icenter]);
        ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (ey == N[1]-1) {
        /* Top boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = UP; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = uyRef(cArrX[ex][icenter],cArrY[ey][inext]);
        ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }

      if (ey == 0) {
        /* Bottom boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DOWN; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = uyRef(cArrX[ex][icenter],cArrY[ey][iprev]);
        ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y */
        DMStagStencil row,col[7];
        PetscScalar   valA[7],valRhs;
        PetscInt      nEntries;

        row.i    = ex  ; row.j    = ey  ; row.loc    = DOWN;    row.c     = 0;
        if (ex == 0) {
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) -2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
          /* Missing left element */
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
          col[4].i = ex  ; col[4].j = ey-1; col[4].loc = ELEMENT; col[4].c  = 0; valA[4] =  1.0 / hy;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] = -1.0 / hy;
        } else if (ex == N[0]-1) {
          /* Right boundary y velocity stencil */
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) -2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
          /* Missing right element */
          col[4].i = ex  ; col[4].j = ey-1; col[4].loc = ELEMENT; col[4].c  = 0; valA[4] =  1.0 / hy;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] = -1.0 / hy;
        } else {
          nEntries = 7;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) -2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
          col[4].i = ex+1; col[4].j = ey  ; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
          col[5].i = ex  ; col[5].j = ey-1; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hy;
          col[6].i = ex  ; col[6].j = ey  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hy;
        }
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = fy(cArrX[ex][icenter],cArrY[ey][iprev]);
        ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }

      if (ex == 0) {
        /* Left velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = LEFT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = uxRef(cArrX[ex][iprev],cArrY[ey][icenter]);
        ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        /* X-momentum equation : (u_xx + u_yy) - p_x = f^x */
        DMStagStencil row,col[7];
        PetscScalar   valA[7],valRhs;
        PetscInt      nEntries;
        row.i    = ex  ; row.j    = ey  ; row.loc    = LEFT;    row.c     = 0;

        if (ey == 0) {
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 /(hx*hx) -1.0 /(hy*hy);
          /* missing term from element below */
          col[1].i = ex  ; col[1].j = ey+1; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
          col[2].i = ex-1; col[2].j = ey  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
          col[4].i = ex-1; col[4].j = ey  ; col[4].loc = ELEMENT; col[4].c  = 0; valA[4] =  1.0 / hx;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] = -1.0 / hx;
        } else if (ey == N[1]-1) {
          /* Top boundary x velocity stencil */
          nEntries = 6;
          row.i    = ex  ; row.j    = ey  ; row.loc    = LEFT;    row.c     = 0;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) -1.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
          /* Missing element above term */
          col[2].i = ex-1; col[2].j = ey  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
          col[4].i = ex-1; col[4].j = ey  ; col[4].loc = ELEMENT; col[4].c  = 0; valA[4] =  1.0 / hx;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] = -1.0 / hx;
        } else {
          /* Note how this is identical to the stencil for U_y, with "DOWN" replaced by "LEFT" and the pressure derivative in the other direction */
          nEntries = 7;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
          col[4].i = ex+1; col[4].j = ey  ; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
          col[5].i = ex-1; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hx;
          col[6].i = ex  ; col[6].j = ey  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hx;

        }
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = fx(cArrX[ex][iprev],cArrY[ey][icenter]);
        ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }

      /* P equation : u_x + v_y = g
         Note that this includes an explicit zero on the diagonal. This is only needed for
         direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace) */
      if (pinPressure && ex == 0 && ey == 0) { /* Pin the first pressure node, if requested */
        DMStagStencil row;
        PetscScalar valA,valRhs;
        row.i = ex; row.j = ey; row.loc  = ELEMENT; row.c = 0;
        valA = 1.0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = pRef(cArrX[ex][icenter],cArrY[ey][icenter]);
        ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        DMStagStencil row,col[5];
        PetscScalar   valA[5],valRhs;

        row.i    = ex; row.j    = ey; row.loc    = ELEMENT; row.c    = 0;
        col[0].i = ex; col[0].j = ey; col[0].loc = LEFT;    col[0].c = 0; valA[0] = -1.0 / hx;
        col[1].i = ex; col[1].j = ey; col[1].loc = RIGHT;   col[1].c = 0; valA[1] =  1.0 / hx;
        col[2].i = ex; col[2].j = ey; col[2].loc = DOWN;    col[2].c = 0; valA[2] = -1.0 / hy;
        col[3].i = ex; col[3].j = ey; col[3].loc = UP;      col[3].c = 0; valA[3] =  1.0 / hy;
        col[4] = row;                                                     valA[4] = 0.0;
        ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,5,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = g(cArrX[ex][icenter],cArrY[ey][icenter]);
        ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMStagRestoreProductCoordinateArraysRead(dmSol,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create a pressure-only DMStag and use it to generate a nullspace vector
   - Create a compatible DMStag with one dof per element (and nothing else).
   - Create a constant vector and normalize it
   - Migrate it to a vector on the original dmSol (making use of the fact
   that this will fill in zeros for "extra" dof)
   - Set the nullspace for the operator
   - Destroy everything (the operator keeps the references it needs) */
static PetscErrorCode AttachNullspace(DM dmSol,Mat A)
{
  PetscErrorCode ierr;
  DM             dmPressure;
  Vec            constantPressure,basis;
  PetscReal      nrm;
  MatNullSpace   matNullSpace;

  PetscFunctionBeginUser;
  ierr = DMStagCreateCompatibleDMStag(dmSol,0,0,1,0,&dmPressure);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmPressure,&constantPressure);CHKERRQ(ierr);
  ierr = VecSet(constantPressure,1.0);CHKERRQ(ierr);
  ierr = VecNorm(constantPressure,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = VecScale(constantPressure,1.0/nrm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmSol,&basis);CHKERRQ(ierr);
  ierr = DMStagMigrateVec(dmPressure,constantPressure,dmSol,basis);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dmSol),PETSC_FALSE,1,&basis,&matNullSpace);CHKERRQ(ierr);
  ierr = VecDestroy(&basis);CHKERRQ(ierr);
  ierr = VecDestroy(&constantPressure);CHKERRQ(ierr);
  ierr = MatSetNullSpace(A,matNullSpace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&matNullSpace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create a reference solution.
   Here, we use the more direct method of iterating over arrays.  */
static PetscErrorCode CreateReferenceSolution(DM dmSol,Vec *pSolRef)
{
  PetscErrorCode ierr;
  PetscInt       startx,starty,nx,ny,nExtra[2],ex,ey;
  PetscInt       iuy,iux,ip,iprev,icenter;
  PetscScalar    ***arrSol,**cArrX,**cArrY;
  Vec            solRefLocal;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dmSol,pSolRef);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);

  /* Obtain indices to use in the raw arrays */
  ierr = DMStagGetLocationSlot(dmSol,DOWN,0,&iuy);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,LEFT,0,&iux);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,ELEMENT,0,&ip);CHKERRQ(ierr);

  /* Use high-level convenience functions to get raw arrays and indices for 1d coordinates */
  ierr = DMStagGetProductCoordinateArraysRead(dmSol,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,LEFT,&iprev);CHKERRQ(ierr);

  ierr = DMStagVecGetArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmSol,&startx,&starty,NULL,&nx,&ny,NULL,&nExtra[0],&nExtra[1],NULL);CHKERRQ(ierr);
  for (ey=starty; ey<starty + ny + nExtra[1]; ++ey) {
    for (ex=startx; ex<startx + nx + nExtra[0]; ++ex) {
      arrSol[ey][ex][iuy] = uyRef(cArrX[ex][icenter],cArrY[ey][iprev]);
      arrSol[ey][ex][iux] = uxRef(cArrX[ex][iprev],cArrY[ey][icenter]);
      if (ey < starty+ny && ex < startx+nx) { /* Don't fill on the dummy elements (though you could, and these values would just be ignored) */
        arrSol[ey][ex][ip]  = pRef(cArrX[ex][icenter],cArrY[ey][icenter]);
      }
    }
  }
  ierr = DMStagVecRestoreArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArraysRead(dmSol,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dmSol,solRefLocal,INSERT_VALUES,*pSolRef);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckSolution(Vec sol,Vec solRef)
{
  PetscErrorCode ierr;
  Vec            diff;
  PetscReal      normsolRef,errAbs,errRel;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(sol,&diff);CHKERRQ(ierr);
  ierr = VecCopy(sol,diff);CHKERRQ(ierr);
  ierr = VecAXPY(diff,-1.0,solRef);CHKERRQ(ierr);
  ierr = VecNorm(diff,NORM_2,&errAbs);CHKERRQ(ierr);
  ierr = VecNorm(solRef,NORM_2,&normsolRef);CHKERRQ(ierr);
  errRel = errAbs/normsolRef;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error (abs): %g\nError (rel): %g\n",(double)errAbs,(double)errRel);CHKERRQ(ierr);
  ierr = VecDestroy(&diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      nsize: 4
      args: -ksp_monitor_short -ksp_converged_reason

   test:
      suffix: direct_umfpack
      requires: suitesparse
      nsize: 1
      args: -pinpressure 1 -stag_grid_x 8 -stag_grid_y 6 -ksp_monitor_short -pc_type lu -pc_factor_mat_solver_type umfpack

TEST*/
