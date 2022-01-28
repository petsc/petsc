static char help[] = "Solve a toy 3D problem on a staggered grid\n\n";
/*

  To demonstrate the basic functionality of DMStag, solves an isoviscous
  incompressible Stokes problem on a rectangular 3D domain.

  u_{xx} + u_{yy} + u_{zz} - p_x = f^x
  v_{xx} + v_{yy} + u_{zz} - p_y = f^y
  w_{xx} + w_{yy} + w_{zz} - p_y = f^z
  u_x    + v_y    + w_z          = g

  g = 0 for the physical case.

  Boundary conditions give prescribed flow perpendicular to the boundaries,
  and zero derivative perpendicular to them (free slip). This involves
  using a modifed stencil at the boundaries. Another option would be to
  use DM_BOUNDARY_GHOSTED in DMStagCreate3d() and a matrix-free operator (MATSHELL)
  making use of the uniformly-available ghost layer.

  Use the -pinpressure option to fix a pressure node, instead of providing
  a constant-pressure nullspace. This allows use of direct solvers, e.g. to
  use UMFPACK,

     ./ex3 -pinpressure 1 -pc_type lu -pc_factor_mat_solver_type umfpack

*/
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>

/* Shorter, more convenient names for DMStagStencilLocation entries */
#define BACK_DOWN_LEFT   DMSTAG_BACK_DOWN_LEFT
#define BACK_DOWN        DMSTAG_BACK_DOWN
#define BACK_DOWN_RIGHT  DMSTAG_BACK_DOWN_RIGHT
#define BACK_LEFT        DMSTAG_BACK_LEFT
#define BACK             DMSTAG_BACK
#define BACK_RIGHT       DMSTAG_BACK_RIGHT
#define BACK_UP_LEFT     DMSTAG_BACK_UP_LEFT
#define BACK_UP          DMSTAG_BACK_UP
#define BACK_UP_RIGHT    DMSTAG_BACK_UP_RIGHT
#define DOWN_LEFT        DMSTAG_DOWN_LEFT
#define DOWN             DMSTAG_DOWN
#define DOWN_RIGHT       DMSTAG_DOWN_RIGHT
#define LEFT             DMSTAG_LEFT
#define ELEMENT          DMSTAG_ELEMENT
#define RIGHT            DMSTAG_RIGHT
#define UP_LEFT          DMSTAG_UP_LEFT
#define UP               DMSTAG_UP
#define UP_RIGHT         DMSTAG_UP_RIGHT
#define FRONT_DOWN_LEFT  DMSTAG_FRONT_DOWN_LEFT
#define FRONT_DOWN       DMSTAG_FRONT_DOWN
#define FRONT_DOWN_RIGHT DMSTAG_FRONT_DOWN_RIGHT
#define FRONT_LEFT       DMSTAG_FRONT_LEFT
#define FRONT            DMSTAG_FRONT
#define FRONT_RIGHT      DMSTAG_FRONT_RIGHT
#define FRONT_UP_LEFT    DMSTAG_FRONT_UP_LEFT
#define FRONT_UP         DMSTAG_FRONT_UP
#define FRONT_UP_RIGHT   DMSTAG_FRONT_UP_RIGHT

static PetscErrorCode CreateReferenceSolution(DM,Vec*);
static PetscErrorCode CreateSystem(DM,Mat*,Vec*,PetscBool);
static PetscErrorCode AttachNullspace(DM,Mat);
static PetscErrorCode CheckSolution(Vec,Vec);

/* Manufactured solution. Chosen to be higher order than can be solved exactly,
and to have a zero derivative for flow parallel to the boundaries. That is,
d(ux)/dy = 0 at the top and bottom boundaries, and d(uy)/dx = 0 at the right
and left boundaries.
These expressions could be made more interesting with more z terms,
and convergence could be confirmed.  */

static PetscScalar uxRef(PetscScalar x,PetscScalar y, PetscScalar z) {return 0.0*x + y*y - 2.0*y*y*y + y*y*y*y + 0.0*z;}
static PetscScalar uyRef(PetscScalar x,PetscScalar y, PetscScalar z) {return x*x - 2.0*x*x*x + x*x*x*x +0.0*y + 0.0*z;}
static PetscScalar uzRef(PetscScalar x,PetscScalar y, PetscScalar z) {return 0.0*x + 0.0*y + 0.0*z + 1.0;}
static PetscScalar pRef (PetscScalar x,PetscScalar y, PetscScalar z) {return -1.0*(x-0.5) + -3.0/2.0*y*y + 0.5 -2.0*(z-1.0);} /* zero integral */
static PetscScalar fx   (PetscScalar x,PetscScalar y, PetscScalar z) {return 0.0*x + 2.0 -12.0*y + 12.0*y*y + 0.0*z + 1.0;}
static PetscScalar fy   (PetscScalar x,PetscScalar y, PetscScalar z) {return 2.0 -12.0*x + 12.0*x*x + 3.0*y + 0.0*z;}
static PetscScalar fz   (PetscScalar x,PetscScalar y, PetscScalar z) {return 0.0*x + 0.0*y + 0.0*z + 2.0;}
static PetscScalar g    (PetscScalar x,PetscScalar y, PetscScalar z) {return 0.0*x + 0.0*y + 0.0*z + 0.0;}

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

  /* Create 3D DMStag for the solution, and set up. */
  {
    const PetscInt dof0 = 0, dof1 = 0,dof2 = 1, dof3 = 1; /* 1 dof on each face and element center */
    const PetscInt stencilWidth = 1;
    ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,5,6,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,dof3,DMSTAG_STENCIL_BOX,stencilWidth,NULL,NULL,NULL,&dmSol);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmSol);CHKERRQ(ierr);
    ierr = DMSetUp(dmSol);CHKERRQ(ierr);
    ierr = DMStagSetUniformCoordinatesExplicit(dmSol,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    /* Note: also see ex2.c, where another, more efficient option is demonstrated,
       using DMStagSetUniformCoordinatesProduct() */
  }

  /* Compute (manufactured) reference solution */
  ierr = CreateReferenceSolution(dmSol,&solRef);CHKERRQ(ierr);

  /* Assemble system */
  ierr = CreateSystem(dmSol,&A,&rhs,pinPressure);CHKERRQ(ierr);

  /* Attach a constant-pressure nullspace to the operator */
  if (!pinPressure) {
    ierr = AttachNullspace(dmSol,A);CHKERRQ(ierr);
  }

  /* Solve */
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

static PetscErrorCode CreateSystem(DM dmSol,Mat *pA,Vec *pRhs, PetscBool pinPressure)
{
  PetscErrorCode    ierr;
  Vec               rhs,coordLocal;
  Mat               A;
  PetscInt          startx,starty,startz,N[3],nx,ny,nz,ex,ey,ez,d;
  PetscInt          icp[3],icux[3],icuy[3],icuz[3],icux_right[3],icuy_up[3],icuz_front[3];
  PetscReal         hx,hy,hz;
  DM                dmCoord;
  PetscScalar       ****arrCoord;

  PetscFunctionBeginUser;
  ierr = DMCreateMatrix(dmSol,pA);CHKERRQ(ierr);
  A = *pA;
  ierr = DMCreateGlobalVector(dmSol,pRhs);CHKERRQ(ierr);
  rhs = *pRhs;

  ierr = DMStagGetCorners(dmSol,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmSol,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
  PetscAssertFalse(N[0] < 2 || N[1] < 2 || N[2] < 2,PetscObjectComm((PetscObject)dmSol),PETSC_ERR_ARG_SIZ,"This example requires at least two elements in each dimensions");
  hx = 1.0/N[0]; hy = 1.0/N[1]; hz = 1.0/N[2];
  ierr = DMGetCoordinateDM(dmSol,&dmCoord);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmSol,&coordLocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmCoord,coordLocal,&arrCoord);CHKERRQ(ierr);
  for (d=0; d<3; ++d) {
    ierr = DMStagGetLocationSlot(dmCoord,ELEMENT,d,&icp[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,LEFT,   d,&icux[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,DOWN,   d,&icuy[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,BACK,   d,&icuz[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,RIGHT,  d,&icux_right[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,UP,     d,&icuy_up[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,FRONT,  d,&icuz_front[d]);CHKERRQ(ierr);
  }

  /* Loop over all local elements. Note that it may be more efficient in real
     applications to loop over each boundary separately */
  for (ez = startz; ez<startz+nz; ++ez) { /* With DMStag, always iterate x fastest, y second fastest, z slowest */
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {

        if (ex == N[0]-1) {
          /* Right Boundary velocity Dirichlet */
          DMStagStencil row;
          PetscScalar   valRhs;
          const PetscScalar valA = 1.0;
          row.i = ex; row.j = ey; row.k = ez; row.loc = RIGHT; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = uxRef(arrCoord[ez][ey][ex][icux_right[0]], arrCoord[ez][ey][ex][icux_right[1]], arrCoord[ez][ey][ex][icux_right[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
        if (ey == N[1]-1) {
          /* Top boundary velocity Dirichlet */
          DMStagStencil row;
          PetscScalar   valRhs;
          const PetscScalar valA = 1.0;
          row.i = ex; row.j = ey; row.k = ez; row.loc = UP; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = uyRef(arrCoord[ez][ey][ex][icuy_up[0]],arrCoord[ez][ey][ex][icuy_up[1]],arrCoord[ez][ey][ex][icuy_up[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
        if (ez == N[2]-1) {
          /* Front boundary velocity Dirichlet */
          DMStagStencil row;
          PetscScalar   valRhs;
          const PetscScalar valA = 1.0;
          row.i = ex; row.j = ey; row.k = ez; row.loc = FRONT; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = uzRef(arrCoord[ez][ey][ex][icuz_front[0]],arrCoord[ez][ey][ex][icuz_front[1]],arrCoord[ez][ey][ex][icuz_front[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }

        /* Equation on left face of this element */
        if (ex == 0) {
          /* Left velocity Dirichlet */
          DMStagStencil row;
          PetscScalar   valRhs;
          const PetscScalar valA = 1.0;
          row.i = ex; row.j = ey; row.k = ez; row.loc = LEFT; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = uxRef(arrCoord[ez][ey][ex][icux[0]],arrCoord[ez][ey][ex][icux[1]],arrCoord[ez][ey][ex][icux[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          /* X-momentum interior equation : (u_xx + u_yy + u_zz) - p_x = f^x */
          DMStagStencil row,col[9];
          PetscScalar   valA[9],valRhs;
          PetscInt      nEntries;

          row.i = ex; row.j = ey; row.k = ez; row.loc = LEFT; row.c = 0;
          if (ey == 0) {
            if (ez == 0) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -1.0 / (hy*hy) -1.0 / (hz*hz);
              /* Missing down term */
              col[1].i = ex  ; col[1].j = ey+1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              /* Missing back term */
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez+1; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex-1; col[5].j = ey  ;  col[5].k = ez  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hx;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hx;
            } else if (ez == N[2]-1) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -1.0 / (hy*hy) -1.0 / (hz*hz);
              /* Missing down term */
              col[1].i = ex  ; col[1].j = ey+1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              /* Missing front term */
              col[5].i = ex-1; col[5].j = ey  ;  col[5].k = ez  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hx;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hx;
            } else {
              nEntries = 8;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -1.0 / (hy*hy) -2.0 / (hz*hz);
              /* Missing down term */
              col[1].i = ex  ; col[1].j = ey+1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = LEFT;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
              col[6].i = ex-1; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hx;
              col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hx;
            }
          } else if (ey == N[1]-1) {
            if (ez == 0) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              /* Missing up term */
              col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              /* Missing back entry */
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez+1; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex-1; col[5].j = ey  ;  col[5].k = ez  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hx;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hx;
            } else if (ez == N[2]-1) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              /* Missing up term */
              col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              /* Missing front term */
              col[5].i = ex-1; col[5].j = ey  ;  col[5].k = ez  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hx;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hx;
            } else {
              nEntries = 8;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -2.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              /* Missing up term */
              col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = LEFT;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
              col[6].i = ex-1; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hx;
              col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hx;
            }
          } else if (ez == 0) {
            nEntries = 8;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
            col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
            col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex+1; col[4].j = ey  ;  col[4].k = ez  ; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
            /* Missing back term */
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = LEFT;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            col[6].i = ex-1; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hx;
            col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hx;
          } else if (ez == N[2]-1) {
            nEntries = 8;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
            col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
            col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex+1; col[4].j = ey  ;  col[4].k = ez  ; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = LEFT;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            /* Missing front term */
            col[6].i = ex-1; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hx;
            col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hx;
          } else {
            nEntries = 9;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -2.0 / (hz*hz);
            col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
            col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex+1; col[4].j = ey  ;  col[4].k = ez  ; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = LEFT;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez+1; col[6].loc = LEFT;    col[6].c  = 0; valA[6] =  1.0 / (hz*hz);
            col[7].i = ex-1; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] =  1.0 / hx;
            col[8].i = ex  ; col[8].j = ey  ;  col[8].k = ez  ; col[8].loc = ELEMENT; col[8].c  = 0; valA[8] = -1.0 / hx;
          }
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = fx(arrCoord[ez][ey][ex][icux[0]], arrCoord[ez][ey][ex][icux[1]], arrCoord[ez][ey][ex][icux[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }

        /* Equation on bottom face of this element */
        if (ey == 0) {
          /* Bottom boundary velocity Dirichlet */
          DMStagStencil row;
          PetscScalar   valRhs;
          const PetscScalar valA = 1.0;
          row.i = ex; row.j = ey; row.k = ez; row.loc = DOWN; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = uyRef(arrCoord[ez][ey][ex][icuy[0]],arrCoord[ez][ey][ex][icuy[1]],arrCoord[ez][ey][ex][icuy[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          /* Y-momentum equation, (v_xx + v_yy + v_zz) - p_y = f^y */
          DMStagStencil row,col[9];
          PetscScalar   valA[9],valRhs;
          PetscInt      nEntries;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DOWN; row.c = 0;
          if (ex ==0) {
            if (ez == 0) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
              /* Left term missing */
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              /* Back term missing */
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez+1; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey-1;  col[5].k = ez  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hy;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hy;
            } else if (ez == N[2]-1) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
              /* Left term missing */
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              /* Front term missing */
              col[5].i = ex  ; col[5].j = ey-1;  col[5].k = ez  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hy;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hy;
            } else {
              nEntries = 8;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -2.0 / (hy*hy) -2.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
              /* Left term missing */
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = DOWN;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
              col[6].i = ex  ; col[6].j = ey-1;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hy;
              col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hy;
            }
          } else if (ex == N[0]-1) {
            if (ez == 0) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
              col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              /* Right term missing */
              /* Back term missing */
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez+1; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey-1;  col[5].k = ez  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hy;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hy;
            } else if (ez == N[2]-1) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
              col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              /* Right term missing */
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              /* Front term missing */
              col[5].i = ex  ; col[5].j = ey-1;  col[5].k = ez  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hy;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hy;
            } else {
              nEntries = 8;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -2.0 / (hy*hy) -2.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
              col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              /* Right term missing */
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = DOWN;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
              col[6].i = ex  ; col[6].j = ey-1;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hy;
              col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hy;
            }
          } else if (ez == 0) {
            nEntries = 8;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
            col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
            col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex+1; col[4].j = ey  ;  col[4].k = ez  ; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
            /* Back term missing */
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = DOWN;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            col[6].i = ex  ; col[6].j = ey-1;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hy;
            col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hy;
          } else if (ez == N[2]-1) {
            nEntries = 8;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -1.0 / (hz*hz);
            col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
            col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex+1; col[4].j = ey  ;  col[4].k = ez  ; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = DOWN;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            /* Front term missing */
            col[6].i = ex  ; col[6].j = ey-1;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hy;
            col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hy;
          } else {
            nEntries = 9;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -2.0 / (hz*hz);
            col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
            col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex+1; col[4].j = ey  ;  col[4].k = ez  ; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = DOWN;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez+1; col[6].loc = DOWN;    col[6].c  = 0; valA[6] =  1.0 / (hz*hz);
            col[7].i = ex  ; col[7].j = ey-1;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] =  1.0 / hy;
            col[8].i = ex  ; col[8].j = ey  ;  col[8].k = ez  ; col[8].loc = ELEMENT; col[8].c  = 0; valA[8] = -1.0 / hy;
          }
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = fy(arrCoord[ez][ey][ex][icuy[0]],arrCoord[ez][ey][ex][icuy[1]],arrCoord[ez][ey][ex][icuy[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }

        /* Equation on back face of this element */
        if (ez == 0) {
          /* Back boundary velocity Dirichlet */
          DMStagStencil row;
          PetscScalar   valRhs;
          const PetscScalar valA = 1.0;
          row.i = ex; row.j = ey; row.k = ez; row.loc = BACK; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = uzRef(arrCoord[ez][ey][ex][icuz[0]],arrCoord[ez][ey][ex][icuz[1]],arrCoord[ez][ey][ex][icuz[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          /* Z-momentum equation, (w_xx + w_yy + w_zz) - p_z = f^z */
          DMStagStencil row,col[9];
          PetscScalar   valA[9],valRhs;
          PetscInt      nEntries;

          row.i = ex; row.j = ey; row.k = ez; row.loc = BACK; row.c = 0;
          if (ex == 0) {
            if (ey == 0) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -1.0 / (hy*hy) -2.0 / (hz*hz);
              /* Down term missing */
              col[1].i = ex  ; col[1].j = ey+1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              /* Left term missing */
              col[2].i = ex+1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              col[3].i = ex  ; col[3].j = ey  ;  col[3].k = ez-1; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hz*hz);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez+1; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hz;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hz;
            } else if (ey == N[1]-1) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -1.0 / (hy*hy) -2.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              /* Up term missing */
              /* Left term missing */
              col[2].i = ex+1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              col[3].i = ex  ; col[3].j = ey  ;  col[3].k = ez-1; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hz*hz);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez+1; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hz;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hz;
            } else {
              nEntries = 8;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -2.0 / (hy*hy) -2.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
              /* Left term missing */
              col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = BACK;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez-1; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hz;
              col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hz;
            }
          } else if (ex == N[0]-1) {
            if (ey == 0) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -1.0 / (hy*hy) -2.0 / (hz*hz);
              /* Down term missing */
              col[1].i = ex  ; col[1].j = ey+1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              /* Right term missing */
              col[3].i = ex  ; col[3].j = ey  ;  col[3].k = ez-1; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hz*hz);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez+1; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hz;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hz;
            } else if (ey == N[1]-1) {
              nEntries = 7;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -1.0 / (hy*hy) -2.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              /* Up term missing */
              col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
              /* Right term missing */
              col[3].i = ex  ; col[3].j = ey  ;  col[3].k = ez-1; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hz*hz);
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez+1; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  1.0 / hz;
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -1.0 / hz;
            } else {
              nEntries = 8;
              col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -1.0 / (hx*hx) + -2.0 / (hy*hy) -2.0 / (hz*hz);
              col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
              col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
              col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
              /* Right term missing */
              col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
              col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = BACK;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
              col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez-1; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hz;
              col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hz;
            }
          } else if (ey == 0) {
            nEntries = 8;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -1.0 / (hy*hy) -2.0 / (hz*hz);
            /* Down term missing */
            col[1].i = ex  ; col[1].j = ey+1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
            col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = BACK;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez-1; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hz;
            col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hz;
          } else if (ey == N[1]-1) {
            nEntries = 8;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -1.0 / (hy*hy) -2.0 / (hz*hz);
            col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            /* Up term missing */
            col[2].i = ex-1; col[2].j = ey  ;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hx*hx);
            col[3].i = ex+1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex  ; col[4].j = ey  ;  col[4].k = ez-1; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hz*hz);
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez+1; col[5].loc = BACK;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez-1; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] =  1.0 / hz;
            col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez  ; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] = -1.0 / hz;
          } else {
            nEntries = 9;
            col[0].i = ex  ; col[0].j = ey  ;  col[0].k = ez  ; col[0].loc = BACK;    col[0].c  = 0; valA[0] = -2.0 / (hx*hx) + -2.0 / (hy*hy) -2.0 / (hz*hz);
            col[1].i = ex  ; col[1].j = ey-1;  col[1].k = ez  ; col[1].loc = BACK;    col[1].c  = 0; valA[1] =  1.0 / (hy*hy);
            col[2].i = ex  ; col[2].j = ey+1;  col[2].k = ez  ; col[2].loc = BACK;    col[2].c  = 0; valA[2] =  1.0 / (hy*hy);
            col[3].i = ex-1; col[3].j = ey  ;  col[3].k = ez  ; col[3].loc = BACK;    col[3].c  = 0; valA[3] =  1.0 / (hx*hx);
            col[4].i = ex+1; col[4].j = ey  ;  col[4].k = ez  ; col[4].loc = BACK;    col[4].c  = 0; valA[4] =  1.0 / (hx*hx);
            col[5].i = ex  ; col[5].j = ey  ;  col[5].k = ez-1; col[5].loc = BACK;    col[5].c  = 0; valA[5] =  1.0 / (hz*hz);
            col[6].i = ex  ; col[6].j = ey  ;  col[6].k = ez+1; col[6].loc = BACK;    col[6].c  = 0; valA[6] =  1.0 / (hz*hz);
            col[7].i = ex  ; col[7].j = ey  ;  col[7].k = ez-1; col[7].loc = ELEMENT; col[7].c  = 0; valA[7] =  1.0 / hz;
            col[8].i = ex  ; col[8].j = ey  ;  col[8].k = ez  ; col[8].loc = ELEMENT; col[8].c  = 0; valA[8] = -1.0 / hz;
          }
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = fz(arrCoord[ez][ey][ex][icuz[0]],arrCoord[ez][ey][ex][icuz[1]],arrCoord[ez][ey][ex][icuz[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }

        /* P equation : u_x + v_y + w_z = g
           Note that this includes an explicit zero on the diagonal. This is only needed for
           direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace) */
        if (pinPressure && ex == 0 && ey == 0 && ez == 0) { /* Pin the first pressure node, if requested */
          DMStagStencil row;
          PetscScalar valA,valRhs;
          row.i = ex; row.j = ey; row.k = ez; row.loc  = ELEMENT; row.c = 0;
          valA = 1.0;
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = pRef(arrCoord[ez][ey][ex][icp[0]],arrCoord[ez][ey][ex][icp[1]],arrCoord[ez][ey][ex][icp[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          DMStagStencil row,col[7];
          PetscScalar   valA[7],valRhs;

          row.i    = ex; row.j    = ey; row.k    = ez; row.loc    = ELEMENT; row.c    = 0;
          col[0].i = ex; col[0].j = ey; col[0].k = ez; col[0].loc = LEFT;    col[0].c = 0; valA[0] = -1.0 / hx;
          col[1].i = ex; col[1].j = ey; col[1].k = ez; col[1].loc = RIGHT;   col[1].c = 0; valA[1] =  1.0 / hx;
          col[2].i = ex; col[2].j = ey; col[2].k = ez; col[2].loc = DOWN;    col[2].c = 0; valA[2] = -1.0 / hy;
          col[3].i = ex; col[3].j = ey; col[3].k = ez; col[3].loc = UP;      col[3].c = 0; valA[3] =  1.0 / hy;
          col[4].i = ex; col[4].j = ey; col[4].k = ez; col[4].loc = BACK;    col[4].c = 0; valA[4] = -1.0 / hz;
          col[5].i = ex; col[5].j = ey; col[5].k = ez; col[5].loc = FRONT;   col[5].c = 0; valA[5] =  1.0 / hz;
          col[6]   = row;                                                                  valA[6] =  0.0;
          ierr = DMStagMatSetValuesStencil(dmSol,A,1,&row,7,col,valA,INSERT_VALUES);CHKERRQ(ierr);
          valRhs = g(arrCoord[ez][ey][ex][icp[0]],arrCoord[ez][ey][ex][icp[1]],arrCoord[ez][ey][ex][icp[2]]);
          ierr = DMStagVecSetValuesStencil(dmSol,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMStagVecRestoreArrayRead(dmCoord,coordLocal,&arrCoord);CHKERRQ(ierr);
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
  ierr = DMStagCreateCompatibleDMStag(dmSol,0,0,0,1,&dmPressure);CHKERRQ(ierr);
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

static PetscErrorCode CreateReferenceSolution(DM dmSol,Vec *pSolRef)
{
  PetscErrorCode    ierr;
  PetscInt          start[3],n[3],nExtra[3],ex,ey,ez,d;
  PetscInt          ip,iux,iuy,iuz,icp[3],icux[3],icuy[3],icuz[3];
  Vec               solRef,solRefLocal,coord,coordLocal;
  DM                dmCoord;
  PetscScalar       ****arrSol,****arrCoord;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dmSol,pSolRef);CHKERRQ(ierr);
  solRef = *pSolRef;
  ierr = DMStagGetCorners(dmSol,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&nExtra[0],&nExtra[1],&nExtra[2]);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dmSol,&dmCoord);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dmSol,&coord);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmCoord,&coordLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmCoord,coord,INSERT_VALUES,coordLocal);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,ELEMENT,0,&ip);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,LEFT,   0,&iux);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,DOWN,   0,&iuy);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,BACK,   0,&iuz);CHKERRQ(ierr);
  for (d=0; d<3; ++d) {
    ierr = DMStagGetLocationSlot(dmCoord,ELEMENT,d,&icp[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,LEFT,   d,&icux[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,DOWN,   d,&icuy[d]);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dmCoord,BACK,   d,&icuz[d]);CHKERRQ(ierr);
  }
  ierr = DMStagVecGetArrayRead(dmCoord,coordLocal,&arrCoord);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  for (ez=start[2]; ez<start[2] + n[2] + nExtra[2]; ++ez) {
    for (ey=start[1]; ey<start[1] + n[1] + nExtra[1]; ++ey) {
      for (ex=start[0]; ex<start[0] + n[0] + nExtra[0]; ++ex) {
        if (ex < start[0] + n[0] && ey < start[1] + n[1]) {
          arrSol[ez][ey][ex][iuz] = uzRef(
              arrCoord[ez][ey][ex][icuz[0]],
              arrCoord[ez][ey][ex][icuz[1]],
              arrCoord[ez][ey][ex][icuz[2]]);
        }
        if (ex < start[0] + n[0] && ey < start[2] + n[2]) {
          arrSol[ez][ey][ex][iuy] = uyRef(
              arrCoord[ez][ey][ex][icuy[0]],
              arrCoord[ez][ey][ex][icuy[1]],
              arrCoord[ez][ey][ex][icuy[2]]);
        }
        if (ex < start[1] + n[1] && ey < start[2] + n[2]) {
          arrSol[ez][ey][ex][iux] = uxRef(
              arrCoord[ez][ey][ex][icux[0]],
              arrCoord[ez][ey][ex][icux[1]],
              arrCoord[ez][ey][ex][icux[2]]);
        }
        if (ex < start[0] + n[0] && ey < start[1] + n[1] && ez < start[2] + n[2]) {
          arrSol[ez][ey][ex][ip]  = pRef(
              arrCoord[ez][ey][ex][icp[0]],
              arrCoord[ez][ey][ex][icp[1]],
              arrCoord[ez][ey][ex][icp[2]]);
        }
      }
    }
  }
  ierr = DMStagVecRestoreArrayRead(dmCoord,coordLocal,&arrCoord);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dmSol,solRefLocal,INSERT_VALUES,solRef);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmCoord,&coordLocal);CHKERRQ(ierr);
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
      requires: mumps
      nsize: 27
      args: -ksp_monitor_short -ksp_converged_reason -stag_ranks_x 3 -stag_ranks_y 3 -stag_ranks_z 3 -pc_fieldsplit_schur_fact_type diag -fieldsplit_0_ksp_type preonly -fieldsplit_1_pc_type none -fieldsplit_0_pc_type lu -fieldsplit_1_ksp_type gmres -fieldsplit_1_ksp_max_it 20

   test:
      suffix: 2
      requires: !single
      nsize: 4
      args: -ksp_monitor_short -ksp_converged_reason -pc_fieldsplit_schur_fact_type diag -fieldsplit_0_ksp_type preonly -fieldsplit_1_pc_type none -fieldsplit_0_pc_type gamg -fieldsplit_0_mg_levels_ksp_max_it 3 -fieldsplit_1_ksp_type gmres -fieldsplit_1_ksp_max_it 20 -fieldsplit_0_pc_gamg_esteig_ksp_type gmres

   test:
      suffix: direct_umfpack
      requires: suitesparse
      nsize: 1
      args: -pinpressure 1 -stag_grid_x 5 -stag_grid_y 3 -stag_grid_z 4 -ksp_monitor_short -pc_type lu -pc_factor_mat_solver_type umfpack

TEST*/
