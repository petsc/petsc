static char help[] = "Solves the incompressible, variable viscosity Stokes equation in 2D driven by buouyancy variations.\n\n";

#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>
#include <petscdmda.h>

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

/* An application context */
typedef struct {
  MPI_Comm    comm;
  DM          dmStokes,dmCoeff;
  Vec         coeff;
  PetscReal   xmax,ymax,xmin,ymin,hxCharacteristic,hyCharacteristic;
  PetscScalar eta1,eta2,rho1,rho2,gy,Kbound,Kcont,etaCharacteristic;
} CtxData;
typedef CtxData* Ctx;

/* Helper functions */
static PetscErrorCode PopulateCoefficientData(Ctx);
static PetscErrorCode CreateSystem(const Ctx,Mat*,Vec*);
static PetscErrorCode DumpSolution(Ctx,Vec);

/* Coefficient/forcing Functions */
static PetscScalar getRho(Ctx ctx,PetscScalar x) { return PetscRealPart(x) < (ctx->xmax-ctx->xmin)/2.0 ? ctx->rho1 : ctx->rho2; }
static PetscScalar getEta(Ctx ctx,PetscScalar x) { return PetscRealPart(x) < (ctx->xmax-ctx->xmin)/2.0 ? ctx->eta1 : ctx->eta2; }

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Ctx            ctx;
  Mat            A;
  Vec            x,b;
  KSP            ksp;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* Populate application context */
  PetscCall(PetscMalloc1(1,&ctx));
  ctx->comm = PETSC_COMM_WORLD;
  ctx->xmin = 0.0;
  ctx->xmax = 1e6;
  ctx->ymin = 0.0;
  ctx->ymax = 1.5e6;
  ctx->rho1 = 3200;
  ctx->rho2 = 3300;
  ctx->eta1 = 1e20;
  ctx->eta2 = 1e22;
  ctx->gy   = 10.0;

  /* Create two DMStag objects, corresponding to the same domain and parallel
     decomposition ("topology"). Each defines a different set of fields on
     the domain ("section"); the first the solution to the Stokes equations
     (x- and y-velocities and scalar pressure), and the second holds coefficients
     (viscosities on corners/elements and densities on corners) */
  ierr = DMStagCreate2d(
      ctx->comm,
      DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
      20,30,                                   /* Global element counts */
      PETSC_DECIDE,PETSC_DECIDE,               /* Determine parallel decomposition automatically */
      0,1,1,                                   /* dof: 0 per vertex, 1 per edge, 1 per face/element */
      DMSTAG_STENCIL_BOX,
      1,                                       /* elementwise stencil width */
      NULL,NULL,
      &ctx->dmStokes);PetscCall(ierr);
  PetscCall(DMSetFromOptions(ctx->dmStokes));
  PetscCall(DMSetUp(ctx->dmStokes));
  PetscCall(DMStagSetUniformCoordinatesExplicit(ctx->dmStokes,0.0,ctx->xmax,0.0,ctx->ymax,0.0,0.0));
  PetscCall(DMStagCreateCompatibleDMStag(ctx->dmStokes,2,0,1,0,&ctx->dmCoeff));
  PetscCall(DMSetUp(ctx->dmCoeff));
  PetscCall(DMStagSetUniformCoordinatesExplicit(ctx->dmCoeff,0.0,ctx->xmax,0.0,ctx->ymax,0.0,0.0));

  /* Note: see ex2.c for a more-efficient way to work with coordinates on an
     orthogonal grid, using DMStagSetUniformCoordinatesProduct() */

  /* Get scaling constants, knowing grid spacing */
  {
    PetscInt N[2];
    PetscReal hxAvgInv;
    PetscCall(DMStagGetGlobalSizes(ctx->dmStokes,&N[0],&N[1],NULL));
    ctx->hxCharacteristic = (ctx->xmax-ctx->xmin)/N[0];
    ctx->hyCharacteristic = (ctx->ymax-ctx->ymin)/N[1];
    ctx->etaCharacteristic = PetscMin(PetscRealPart(ctx->eta1),PetscRealPart(ctx->eta2));
    hxAvgInv = 2.0/(ctx->hxCharacteristic + ctx->hyCharacteristic);
    ctx->Kcont = ctx->etaCharacteristic*hxAvgInv;
    ctx->Kbound = ctx->etaCharacteristic*hxAvgInv*hxAvgInv;
  }

  /* Populate coefficient data */
  PetscCall(PopulateCoefficientData(ctx));

  /* Construct System */
  PetscCall(CreateSystem(ctx,&A,&b));

  /* Solve */
  PetscCall(VecDuplicate(b,&x));
  PetscCall(KSPCreate(ctx->comm,&ksp));
  PetscCall(KSPSetType(ksp,KSPFGMRES));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(PetscOptionsSetValue(NULL,"-ksp_converged_reason","")); /* To get info on direct solve success */
  PetscCall(KSPSetDM(ksp,ctx->dmStokes));
  PetscCall(KSPSetDMActive(ksp,PETSC_FALSE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));
  {
    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp,&reason));
    PetscCheck(reason >= 0,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Linear solve failed");
  }

  /* Dump solution by converting to DMDAs and dumping */
  PetscCall(DumpSolution(ctx,x));

  /* Destroy PETSc objects and finalize */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&ctx->coeff));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&ctx->dmStokes));
  PetscCall(DMDestroy(&ctx->dmCoeff));
  PetscCall(PetscFree(ctx));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode CreateSystem(const Ctx ctx,Mat *pA,Vec *pRhs)
{
  PetscInt       N[2];
  PetscInt       ex,ey,startx,starty,nx,ny;
  Mat            A;
  Vec            rhs;
  PetscReal      hx,hy;
  const PetscBool pinPressure = PETSC_TRUE;
  Vec            coeffLocal;

  PetscFunctionBeginUser;
  PetscCall(DMCreateMatrix(ctx->dmStokes,pA));
  A = *pA;
  PetscCall(DMCreateGlobalVector(ctx->dmStokes,pRhs));
  rhs = *pRhs;
  PetscCall(DMStagGetCorners(ctx->dmStokes,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(ctx->dmStokes,&N[0],&N[1],NULL));
  hx = ctx->hxCharacteristic;
  hy = ctx->hyCharacteristic;
  PetscCall(DMGetLocalVector(ctx->dmCoeff,&coeffLocal));
  PetscCall(DMGlobalToLocal(ctx->dmCoeff,ctx->coeff,INSERT_VALUES,coeffLocal));

  /* Loop over all local elements. Note that it may be more efficient in real
     applications to loop over each boundary separately */
  for (ey = starty; ey<starty+ny; ++ey) { /* With DMStag, always iterate x fastest, y second fastest, z slowest */
    for (ex = startx; ex<startx+nx; ++ex) {

      if (ey == N[1]-1) {
        /* Top boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = ctx->Kbound;
        row.i = ex; row.j = ey; row.loc = UP; row.c = 0;
        PetscCall(DMStagMatSetValuesStencil(ctx->dmStokes,A,1,&row,1,&row,&valA,INSERT_VALUES));
        valRhs = 0.0;
        PetscCall(DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,&valRhs,INSERT_VALUES));
      }

      if (ey == 0) {
        /* Bottom boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = ctx->Kbound;
        row.i = ex; row.j = ey; row.loc = DOWN; row.c = 0;
        PetscCall(DMStagMatSetValuesStencil(ctx->dmStokes,A,1,&row,1,&row,&valA,INSERT_VALUES));
        valRhs = 0.0;
        PetscCall(DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,&valRhs,INSERT_VALUES));
      } else {
        /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y : includes non-zero forcing */
        PetscInt      nEntries;
        DMStagStencil row,col[11];
        PetscScalar   valA[11];
        DMStagStencil rhoPoint[2];
        PetscScalar   rho[2],valRhs;
        DMStagStencil etaPoint[4];
        PetscScalar   eta[4],etaLeft,etaRight,etaUp,etaDown;

        /* get rho values  and compute rhs value*/
        rhoPoint[0].i = ex; rhoPoint[0].j = ey; rhoPoint[0].loc = DOWN_LEFT;  rhoPoint[0].c = 1;
        rhoPoint[1].i = ex; rhoPoint[1].j = ey; rhoPoint[1].loc = DOWN_RIGHT; rhoPoint[1].c = 1;
        PetscCall(DMStagVecGetValuesStencil(ctx->dmCoeff,coeffLocal,2,rhoPoint,rho));
        valRhs = -ctx->gy * 0.5 * (rho[0] + rho[1]);

        /* Get eta values */
        etaPoint[0].i = ex; etaPoint[0].j = ey;   etaPoint[0].loc = DOWN_LEFT;  etaPoint[0].c = 0; /* Left  */
        etaPoint[1].i = ex; etaPoint[1].j = ey;   etaPoint[1].loc = DOWN_RIGHT; etaPoint[1].c = 0; /* Right */
        etaPoint[2].i = ex; etaPoint[2].j = ey;   etaPoint[2].loc = ELEMENT;    etaPoint[2].c = 0; /* Up    */
        etaPoint[3].i = ex; etaPoint[3].j = ey-1; etaPoint[3].loc = ELEMENT;    etaPoint[3].c = 0; /* Down  */
        PetscCall(DMStagVecGetValuesStencil(ctx->dmCoeff,coeffLocal,4,etaPoint,eta));
        etaLeft = eta[0]; etaRight = eta[1]; etaUp = eta[2]; etaDown = eta[3];

        if (ex == 0) {
          /* Left boundary y velocity stencil */
          nEntries = 10;
          row.i    = ex  ; row.j    = ey  ; row.loc    = DOWN;     row.c     = 0;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DOWN;     col[0].c  = 0; valA[0]  = -2.0 * (etaDown + etaUp) / (hy*hy) - (etaRight) /(hx*hx);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DOWN;     col[1].c  = 0; valA[1]  =  2.0 * etaDown  / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DOWN;     col[2].c  = 0; valA[2]  =  2.0 * etaUp    / (hy*hy);
          /* No left entry */
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DOWN;     col[3].c  = 0; valA[3]  =        etaRight / (hx*hx);
          col[4].i = ex  ; col[4].j = ey-1; col[4].loc = LEFT;     col[4].c  = 0; valA[4]  =        etaLeft  / (hx*hy); /* down left x edge */
          col[5].i = ex  ; col[5].j = ey-1; col[5].loc = RIGHT;    col[5].c  = 0; valA[5]  = -      etaRight / (hx*hy); /* down right x edge */
          col[6].i = ex  ; col[6].j = ey  ; col[6].loc = LEFT;     col[6].c  = 0; valA[6]  = -      etaLeft  / (hx*hy); /* up left x edge */
          col[7].i = ex  ; col[7].j = ey  ; col[7].loc = RIGHT;    col[7].c  = 0; valA[7]  =        etaRight / (hx*hy); /* up right x edge */
          col[8].i = ex  ; col[8].j = ey-1; col[8].loc = ELEMENT;  col[8].c  = 0; valA[8]  =  ctx->Kcont / hy;
          col[9].i = ex  ; col[9].j = ey  ; col[9].loc = ELEMENT;  col[9].c  = 0; valA[9]  = -ctx->Kcont / hy;
        } else if (ex == N[0]-1) {
          /* Right boundary y velocity stencil */
          nEntries = 10;
          row.i    = ex  ; row.j    = ey  ; row.loc    = DOWN;     row.c     = 0;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DOWN;     col[0].c  = 0; valA[0]  = -2.0 * (etaDown + etaUp) / (hy*hy) - (etaLeft) /(hx*hx);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DOWN;     col[1].c  = 0; valA[1]  =  2.0 * etaDown  / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DOWN;     col[2].c  = 0; valA[2]  =  2.0 * etaUp    / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DOWN;     col[3].c  = 0; valA[3]  =        etaLeft  / (hx*hx);
          /* No right element */
          col[4].i = ex  ; col[4].j = ey-1; col[4].loc = LEFT;     col[4].c  = 0; valA[4]  =        etaLeft  / (hx*hy); /* down left x edge */
          col[5].i = ex  ; col[5].j = ey-1; col[5].loc = RIGHT;    col[5].c  = 0; valA[5]  = -      etaRight / (hx*hy); /* down right x edge */
          col[6].i = ex  ; col[6].j = ey  ; col[6].loc = LEFT;     col[6].c  = 0; valA[7]  = -      etaLeft  / (hx*hy); /* up left x edge */
          col[7].i = ex  ; col[7].j = ey  ; col[7].loc = RIGHT;    col[7].c  = 0; valA[7]  =        etaRight / (hx*hy); /* up right x edge */
          col[8].i = ex  ; col[8].j = ey-1; col[8].loc = ELEMENT;  col[8].c  = 0; valA[8]  =  ctx->Kcont / hy;
          col[9].i = ex  ; col[9].j = ey  ; col[9].loc = ELEMENT;  col[9].c  = 0; valA[9]  = -ctx->Kcont / hy;
        } else {
          /* U_y interior equation */
          nEntries = 11;
          row.i    = ex  ; row.j     = ey  ; row.loc     = DOWN;     row.c     = 0;
          col[0].i = ex  ; col[0].j  = ey  ; col[0].loc  = DOWN;     col[0].c  = 0; valA[0]  = -2.0 * (etaDown + etaUp) / (hy*hy) - (etaLeft + etaRight) /(hx*hx);
          col[1].i = ex  ; col[1].j  = ey-1; col[1].loc  = DOWN;     col[1].c  = 0; valA[1]  =  2.0 * etaDown  / (hy*hy);
          col[2].i = ex  ; col[2].j  = ey+1; col[2].loc  = DOWN;     col[2].c  = 0; valA[2]  =  2.0 * etaUp    / (hy*hy);
          col[3].i = ex-1; col[3].j  = ey  ; col[3].loc  = DOWN;     col[3].c  = 0; valA[3]  =        etaLeft  / (hx*hx);
          col[4].i = ex+1; col[4].j  = ey  ; col[4].loc  = DOWN;     col[4].c  = 0; valA[4]  =        etaRight / (hx*hx);
          col[5].i = ex  ; col[5].j  = ey-1; col[5].loc  = LEFT;     col[5].c  = 0; valA[5]  =        etaLeft  / (hx*hy); /* down left x edge */
          col[6].i = ex  ; col[6].j  = ey-1; col[6].loc  = RIGHT;    col[6].c  = 0; valA[6]  = -      etaRight / (hx*hy); /* down right x edge */
          col[7].i = ex  ; col[7].j  = ey  ; col[7].loc  = LEFT;     col[7].c  = 0; valA[7]  = -      etaLeft  / (hx*hy); /* up left x edge */
          col[8].i = ex  ; col[8].j  = ey  ; col[8].loc  = RIGHT;    col[8].c  = 0; valA[8]  =        etaRight / (hx*hy); /* up right x edge */
          col[9].i = ex  ; col[9].j  = ey-1; col[9].loc  = ELEMENT;  col[9].c  = 0; valA[9]  =  ctx->Kcont / hy;
          col[10].i = ex ; col[10].j = ey  ; col[10].loc = ELEMENT; col[10].c  = 0; valA[10] = -ctx->Kcont / hy;
        }

        /* Insert Y-momentum entries */
        PetscCall(DMStagMatSetValuesStencil(ctx->dmStokes,A,1,&row,nEntries,col,valA,INSERT_VALUES));
        PetscCall(DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,&valRhs,INSERT_VALUES));
      }

      if (ex == N[0]-1) {
        /* Right Boundary velocity Dirichlet */
        /* Redundant in the corner */
        DMStagStencil row;
        PetscScalar   valRhs;

        const PetscScalar valA = ctx->Kbound;
        row.i = ex; row.j = ey; row.loc = RIGHT; row.c = 0;
        PetscCall(DMStagMatSetValuesStencil(ctx->dmStokes,A,1,&row,1,&row,&valA,INSERT_VALUES));
        valRhs = 0.0;
        PetscCall(DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,&valRhs,INSERT_VALUES));
      }
      if (ex == 0) {
        /* Left velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = ctx->Kbound;
        row.i = ex; row.j = ey; row.loc = LEFT; row.c = 0;
        PetscCall(DMStagMatSetValuesStencil(ctx->dmStokes,A,1,&row,1,&row,&valA,INSERT_VALUES));
        valRhs = 0.0;
        PetscCall(DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,&valRhs,INSERT_VALUES));
      } else {
        /* X-momentum equation : (u_xx + u_yy) - p_x = f^x */
        PetscInt nEntries;
        DMStagStencil row,col[11];
        PetscScalar   valRhs,valA[11];
        DMStagStencil etaPoint[4];
        PetscScalar eta[4],etaLeft,etaRight,etaUp,etaDown;

        /* Get eta values */
        etaPoint[0].i = ex-1; etaPoint[0].j = ey; etaPoint[0].loc = ELEMENT;   etaPoint[0].c = 0; /* Left  */
        etaPoint[1].i = ex;   etaPoint[1].j = ey; etaPoint[1].loc = ELEMENT;   etaPoint[1].c = 0; /* Right */
        etaPoint[2].i = ex;   etaPoint[2].j = ey; etaPoint[2].loc = UP_LEFT;   etaPoint[2].c = 0; /* Up    */
        etaPoint[3].i = ex;   etaPoint[3].j = ey; etaPoint[3].loc = DOWN_LEFT; etaPoint[3].c = 0; /* Down  */
        PetscCall(DMStagVecGetValuesStencil(ctx->dmCoeff,coeffLocal,4,etaPoint,eta));
        etaLeft = eta[0]; etaRight = eta[1]; etaUp = eta[2]; etaDown = eta[3];

        if (ey == 0) {
          /* Bottom boundary x velocity stencil (with zero vel deriv) */
          nEntries = 10;
          row.i     = ex  ; row.j     = ey  ; row.loc     = LEFT;    row.c      = 0;
          col[0].i  = ex  ; col[0].j  = ey  ; col[0].loc  = LEFT;    col[0].c   = 0; valA[0]  = -2.0 * (etaLeft + etaRight) / (hx*hx) -(etaUp) / (hy*hy);
          /* Missing element below */
          col[1].i = ex  ; col[1].j  = ey+1; col[1].loc  = LEFT;    col[1].c   = 0; valA[1]  =        etaUp    / (hy*hy);
          col[2].i = ex-1; col[2].j  = ey  ; col[2].loc  = LEFT;    col[2].c   = 0; valA[2]  =  2.0 * etaLeft  / (hx*hx);
          col[3].i = ex+1; col[3].j  = ey  ; col[3].loc  = LEFT;    col[3].c   = 0; valA[3]  =  2.0 * etaRight / (hx*hx);
          col[4].i = ex-1; col[4].j  = ey  ; col[4].loc  = DOWN;    col[4].c   = 0; valA[4]  =        etaDown  / (hx*hy); /* down left */
          col[5].i = ex  ; col[5].j  = ey  ; col[5].loc  = DOWN;    col[5].c   = 0; valA[5]  = -      etaDown  / (hx*hy); /* down right */
          col[6].i = ex-1; col[6].j  = ey  ; col[6].loc  = UP;      col[6].c   = 0; valA[6]  = -      etaUp    / (hx*hy); /* up left */
          col[7].i = ex  ; col[7].j  = ey  ; col[7].loc  = UP;      col[7].c   = 0; valA[7]  =        etaUp    / (hx*hy); /* up right */
          col[8].i = ex-1; col[8].j  = ey  ; col[8].loc  = ELEMENT; col[8].c   = 0; valA[8]  =  ctx->Kcont / hx;
          col[9].i = ex  ; col[9].j  = ey  ; col[9].loc  = ELEMENT; col[9].c   = 0; valA[9]  = -ctx->Kcont / hx;
          valRhs = 0.0;
        } else if (ey == N[1]-1) {
          /* Top boundary x velocity stencil */
          nEntries = 10;
          row.i    = ex  ; row.j     = ey  ; row.loc     = LEFT;    row.c     = 0;
          col[0].i = ex  ; col[0].j  = ey  ; col[0].loc  = LEFT;    col[0].c  = 0; valA[0]  = -2.0 * (etaLeft + etaRight) / (hx*hx) -(etaDown) / (hy*hy);
          col[1].i = ex  ; col[1].j  = ey-1; col[1].loc  = LEFT;    col[1].c  = 0; valA[1]  =        etaDown  / (hy*hy);
          /* Missing element above */
          col[2].i = ex-1; col[2].j  = ey  ; col[2].loc  = LEFT;    col[2].c  = 0; valA[2]  =  2.0 * etaLeft  / (hx*hx);
          col[3].i = ex+1; col[3].j  = ey  ; col[3].loc  = LEFT;    col[3].c  = 0; valA[3]  =  2.0 * etaRight / (hx*hx);
          col[4].i = ex-1; col[4].j  = ey  ; col[4].loc  = DOWN;    col[4].c  = 0; valA[4]  =        etaDown  / (hx*hy); /* down left */
          col[5].i = ex  ; col[5].j  = ey  ; col[5].loc  = DOWN;    col[5].c  = 0; valA[5]  = -      etaDown  / (hx*hy); /* down right */
          col[6].i = ex-1; col[6].j  = ey  ; col[6].loc  = UP;      col[6].c  = 0; valA[6]  = -      etaUp    / (hx*hy); /* up left */
          col[7].i = ex  ; col[7].j  = ey  ; col[7].loc  = UP;      col[7].c  = 0; valA[7]  =        etaUp    / (hx*hy); /* up right */
          col[8].i = ex-1; col[8].j  = ey  ; col[8].loc  = ELEMENT; col[8].c  = 0; valA[8]  =  ctx->Kcont / hx;
          col[9].i = ex  ; col[9].j  = ey  ; col[9].loc = ELEMENT;  col[9].c  = 0; valA[9]  = -ctx->Kcont / hx;
          valRhs = 0.0;
        } else {
          /* U_x interior equation */
          nEntries = 11;
          row.i     = ex  ; row.j     = ey  ; row.loc     = LEFT;    row.c      = 0;
          col[0].i  = ex  ; col[0].j  = ey  ; col[0].loc  = LEFT;    col[0].c   = 0; valA[0]  = -2.0 * (etaLeft + etaRight) / (hx*hx) -(etaUp + etaDown) / (hy*hy);
          col[1].i  = ex  ; col[1].j  = ey-1; col[1].loc  = LEFT;    col[1].c   = 0; valA[1]  =        etaDown  / (hy*hy);
          col[2].i  = ex  ; col[2].j  = ey+1; col[2].loc  = LEFT;    col[2].c   = 0; valA[2]  =        etaUp    / (hy*hy);
          col[3].i  = ex-1; col[3].j  = ey  ; col[3].loc  = LEFT;    col[3].c   = 0; valA[3]  =  2.0 * etaLeft  / (hx*hx);
          col[4].i  = ex+1; col[4].j  = ey  ; col[4].loc  = LEFT;    col[4].c   = 0; valA[4]  =  2.0 * etaRight / (hx*hx);
          col[5].i  = ex-1; col[5].j  = ey  ; col[5].loc  = DOWN;    col[5].c   = 0; valA[5]  =        etaDown  / (hx*hy); /* down left */
          col[6].i  = ex  ; col[6].j  = ey  ; col[6].loc  = DOWN;    col[6].c   = 0; valA[6]  = -      etaDown  / (hx*hy); /* down right */
          col[7].i  = ex-1; col[7].j  = ey  ; col[7].loc  = UP;      col[7].c   = 0; valA[7]  = -      etaUp    / (hx*hy); /* up left */
          col[8].i  = ex  ; col[8].j  = ey  ; col[8].loc  = UP;      col[8].c   = 0; valA[8]  =        etaUp    / (hx*hy); /* up right */
          col[9].i  = ex-1; col[9].j  = ey  ; col[9].loc  = ELEMENT; col[9].c   = 0; valA[9]  =  ctx->Kcont / hx;
          col[10].i = ex  ; col[10].j = ey  ; col[10].loc = ELEMENT; col[10].c  = 0; valA[10] = -ctx->Kcont / hx;
          valRhs = 0.0;
        }
        PetscCall(DMStagMatSetValuesStencil(ctx->dmStokes,A,1,&row,nEntries,col,valA,INSERT_VALUES));
        PetscCall(DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,&valRhs,INSERT_VALUES));
      }

      /* P equation : u_x + v_y = 0
         Note that this includes an explicit zero on the diagonal. This is only needed for
         direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace) */
      if (pinPressure && ex == 0 && ey == 0) { /* Pin the first pressure node to zero, if requested */
        DMStagStencil row;
        PetscScalar valA,valRhs;
        row.i = ex; row.j = ey; row.loc = ELEMENT; row.c = 0;
        valA = ctx->Kbound;
        PetscCall(DMStagMatSetValuesStencil(ctx->dmStokes,A,1,&row,1,&row,&valA,INSERT_VALUES));
        valRhs = 0.0;
        PetscCall(DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,&valRhs,INSERT_VALUES));
      } else {
        DMStagStencil row,col[5];
        PetscScalar   valA[5],valRhs;

        row.i    = ex; row.j    = ey; row.loc    = ELEMENT; row.c    = 0;
        col[0].i = ex; col[0].j = ey; col[0].loc = LEFT;    col[0].c = 0; valA[0] = -ctx->Kcont / hx;
        col[1].i = ex; col[1].j = ey; col[1].loc = RIGHT;   col[1].c = 0; valA[1] =  ctx->Kcont / hx;
        col[2].i = ex; col[2].j = ey; col[2].loc = DOWN;    col[2].c = 0; valA[2] = -ctx->Kcont / hy;
        col[3].i = ex; col[3].j = ey; col[3].loc = UP;      col[3].c = 0; valA[3] =  ctx->Kcont / hy;
        col[4] = row;                                                     valA[4] = 0.0;
        PetscCall(DMStagMatSetValuesStencil(ctx->dmStokes,A,1,&row,5,col,valA,INSERT_VALUES));
        valRhs = 0.0;
        PetscCall(DMStagVecSetValuesStencil(ctx->dmStokes,rhs,1,&row,&valRhs,INSERT_VALUES));
      }
    }
  }
  PetscCall(DMRestoreLocalVector(ctx->dmCoeff,&coeffLocal));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(rhs));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyEnd(rhs));
  PetscFunctionReturn(0);
}

/* Here, we demonstrate getting coordinates from a vector by using DMStagStencil.
This would usually be done with direct array access, though. */
static PetscErrorCode PopulateCoefficientData(Ctx ctx)
{
  PetscInt       N[2],nExtra[2];
  PetscInt       ex,ey,startx,starty,nx,ny;
  Vec            coeffLocal,coordLocal;
  DM             dmCoord;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(ctx->dmCoeff,&ctx->coeff));
  PetscCall(DMGetLocalVector(ctx->dmCoeff,&coeffLocal));
  PetscCall(DMStagGetCorners(ctx->dmCoeff,&startx,&starty,NULL,&nx,&ny,NULL,&nExtra[0],&nExtra[1],NULL));
  PetscCall(DMStagGetGlobalSizes(ctx->dmCoeff,&N[0],&N[1],NULL));
  PetscCall(DMGetCoordinatesLocal(ctx->dmCoeff,&coordLocal));
  PetscCall(DMGetCoordinateDM(ctx->dmCoeff,&dmCoord));
  for (ey = starty; ey<starty+ny+nExtra[1]; ++ey) {
    for (ex = startx; ex<startx+nx+nExtra[0]; ++ex) {

      /* Eta (element) */
      if (ey < starty + ny && ex < startx + nx) {
        DMStagStencil point,pointCoordx;
        PetscScalar   val,x;
        point.i = ex; point.j = ey; point.loc = ELEMENT; point.c = 0;
        pointCoordx = point;
        PetscCall(DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x));
        val = getEta(ctx,x);
        PetscCall(DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES));
      }

      /* Rho */
      {
        DMStagStencil point,pointCoordx;
        PetscScalar   val,x;
        point.i = ex; point.j = ey; point.loc = DOWN_LEFT; point.c = 1; /* Note .c = 1 */
        pointCoordx = point; pointCoordx.c = 0;
        PetscCall(DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x));
        val = getRho(ctx,x);
        PetscCall(DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES));
      }

      /* Eta (corner) - populate extra corners on right/top of domain */
      {
        DMStagStencil point,pointCoordx;
        PetscScalar   val,x;
        point.i = ex; point.j = ey; point.loc = DOWN_LEFT; point.c = 0;
        pointCoordx = point;
        PetscCall(DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x));
        val = getEta(ctx,x);
        PetscCall(DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES));
      }
      if (ex == N[0]-1) {
        DMStagStencil point,pointCoordx;
        PetscScalar   val,x;
        point.i = ex; point.j = ey; point.loc = DOWN_RIGHT; point.c = 0;
        pointCoordx = point;
        PetscCall(DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x));
        val = getEta(ctx,x);
        PetscCall(DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES));
      }
      if (ey == N[1]-1) {
        DMStagStencil point,pointCoordx;
        PetscScalar   val,x;
        point.i = ex; point.j = ey; point.loc = UP_LEFT; point.c = 0;
        pointCoordx = point;
        PetscCall(DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x));
        val = getEta(ctx,x);
        PetscCall(DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES));
      }
      if (ex == N[0]-1 && ey == N[1]-1) {
        DMStagStencil point,pointCoordx;
        PetscScalar   val,x;
        point.i = ex; point.j = ey; point.loc = UP_RIGHT; point.c = 0;
        pointCoordx = point;
        PetscCall(DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x));
        val = getEta(ctx,x);
        PetscCall(DMStagVecSetValuesStencil(ctx->dmCoeff,ctx->coeff,1,&point,&val,INSERT_VALUES));
      }
    }
  }
  PetscCall(VecAssemblyBegin(ctx->coeff));
  PetscCall(VecAssemblyEnd(ctx->coeff));
  PetscCall(DMRestoreLocalVector(ctx->dmCoeff,&coeffLocal));
  PetscFunctionReturn(0);
}

static PetscErrorCode DumpSolution(Ctx ctx,Vec x)
{
  DM             dmVelAvg;
  Vec            velAvg;
  DM             daVelAvg,daP,daEtaElement,daEtaCorner,daRho;
  Vec            vecVelAvg,vecP,vecEtaElement,vecEtaCorner,vecRho;

  PetscFunctionBeginUser;

  /* For convenience, create a new DM and Vec which will hold averaged velocities
     Note that this could also be accomplished with direct array access, using
     DMStagVecGetArray() and related functions */
  PetscCall(DMStagCreateCompatibleDMStag(ctx->dmStokes,0,0,2,0,&dmVelAvg)); /* 2 dof per element */
  PetscCall(DMSetUp(dmVelAvg));
  PetscCall(DMStagSetUniformCoordinatesExplicit(dmVelAvg,0.0,ctx->xmax,0.0,ctx->ymax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(dmVelAvg,&velAvg));
  {
    PetscInt ex,ey,startx,starty,nx,ny;
    Vec      stokesLocal;
    PetscCall(DMGetLocalVector(ctx->dmStokes,&stokesLocal));
    PetscCall(DMGlobalToLocal(ctx->dmStokes,x,INSERT_VALUES,stokesLocal));
    PetscCall(DMStagGetCorners(dmVelAvg,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL));
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil from[4],to[2];
        PetscScalar   valFrom[4],valTo[2];
        from[0].i = ex; from[0].j = ey; from[0].loc = UP;    from[0].c = 0;
        from[1].i = ex; from[1].j = ey; from[1].loc = DOWN;  from[1].c = 0;
        from[2].i = ex; from[2].j = ey; from[2].loc = LEFT;  from[2].c = 0;
        from[3].i = ex; from[3].j = ey; from[3].loc = RIGHT; from[3].c = 0;
        PetscCall(DMStagVecGetValuesStencil(ctx->dmStokes,stokesLocal,4,from,valFrom));
        to[0].i = ex; to[0].j = ey; to[0].loc = ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = ex; to[1].j = ey; to[1].loc = ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        PetscCall(DMStagVecSetValuesStencil(dmVelAvg,velAvg,2,to,valTo,INSERT_VALUES));
      }
    }
    PetscCall(VecAssemblyBegin(velAvg));
    PetscCall(VecAssemblyEnd(velAvg));
    PetscCall(DMRestoreLocalVector(ctx->dmStokes,&stokesLocal));
  }

  /* Create individual DMDAs for sub-grids of our DMStag objects. This is
     somewhat inefficient, but allows use of the DMDA API without re-implementing
     all utilities for DMStag */

  PetscCall(DMStagVecSplitToDMDA(ctx->dmStokes,x,DMSTAG_ELEMENT,0,&daP,&vecP));
  PetscCall(PetscObjectSetName((PetscObject)vecP,"p (scaled)"));
  PetscCall(DMStagVecSplitToDMDA(ctx->dmCoeff,ctx->coeff,DMSTAG_DOWN_LEFT,0, &daEtaCorner, &vecEtaCorner));
  PetscCall(PetscObjectSetName((PetscObject)vecEtaCorner,"eta"));
  PetscCall(DMStagVecSplitToDMDA(ctx->dmCoeff,ctx->coeff,DMSTAG_ELEMENT,  0, &daEtaElement,&vecEtaElement));
  PetscCall(PetscObjectSetName((PetscObject)vecEtaElement,"eta"));
  PetscCall(DMStagVecSplitToDMDA(ctx->dmCoeff,ctx->coeff,DMSTAG_DOWN_LEFT,  1, &daRho,       &vecRho));
  PetscCall(PetscObjectSetName((PetscObject)vecRho,"density"));
  PetscCall(DMStagVecSplitToDMDA(dmVelAvg,    velAvg,    DMSTAG_ELEMENT,  -3,&daVelAvg,    &vecVelAvg)); /* note -3 : pad with zero */
  PetscCall(PetscObjectSetName((PetscObject)vecVelAvg,"Velocity (Averaged)"));

  /* Dump element-based fields to a .vtr file */
  {
    PetscViewer viewer;
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVelAvg),"ex4_element.vtr",FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(vecVelAvg,viewer));
    PetscCall(VecView(vecP,viewer));
    PetscCall(VecView(vecEtaElement,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Dump vertex-based fields to a second .vtr file */
  {
    PetscViewer viewer;
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)daEtaCorner),"ex4_vertex.vtr",FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(vecEtaCorner,viewer));
    PetscCall(VecView(vecRho,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Edge-based fields could similarly be dumped */

  /* Destroy DMDAs and Vecs */
  PetscCall(VecDestroy(&vecVelAvg));
  PetscCall(VecDestroy(&vecP));
  PetscCall(VecDestroy(&vecEtaCorner));
  PetscCall(VecDestroy(&vecEtaElement));
  PetscCall(VecDestroy(&vecRho));
  PetscCall(DMDestroy(&daVelAvg));
  PetscCall(DMDestroy(&daP));
  PetscCall(DMDestroy(&daEtaCorner));
  PetscCall(DMDestroy(&daEtaElement));
  PetscCall(DMDestroy(&daRho));
  PetscCall(VecDestroy(&velAvg));
  PetscCall(DMDestroy(&dmVelAvg));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: direct_umfpack
      requires: suitesparse !complex
      nsize: 1
      args: -stag_grid_x 12 -stag_grid_y 7 -pc_type lu -pc_factor_mat_solver_type umfpack

   test:
      suffix: direct_mumps
      requires: mumps !complex
      nsize: 9
      args: -stag_grid_x 13 -stag_grid_y 8 -pc_type lu -pc_factor_mat_solver_type mumps

TEST*/
