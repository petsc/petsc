static const char help[] = "Simple 2D or 3D finite difference seismic wave propagation, using\n"
"a staggered grid represented with DMStag objects.\n"
"-dim <2,3>  : dimension of problem\n"
"-nsteps <n> : number of timesteps\n"
"-dt <dt>    : length of timestep\n"
"\n";

/*
Reference:

@article{Virieux1986,
  title     = {{P-SV} Wave Propagation in Heterogeneous Media: {V}elocity-Stress Finite-Difference Method},
  author    = {Virieux, Jean},
  journal   = {Geophysics},
  volume    = {51},
  number    = {4},
  pages     = {889--901},
  publisher = {Society of Exploration Geophysicists},
  year      = {1986},
}

Notes:

* This example uses y in 2 dimensions, where the paper uses z.
* This example uses the dual grid of the one pictured in Fig. 1. of the paper,
so that velocities are on face boundaries, shear stresses are defined on vertices,
and normal stresses are defined on elements.
* There is a typo in the paragraph after (5) in the paper: Sigma, Xi, and Tau
represent tau_xx, tau_xz, and tau_zz, respectively (the last two entries are
transposed in the paper).
* This example treats the boundaries naively (by leaving ~zero velocity and stress there).

*/

#include <petscdmstag.h>
#include <petscts.h>

/* A struct defining the parameters of the problem */
typedef struct {
  DM          dm_velocity,dm_stress;
  DM          dm_buoyancy,dm_lame;
  Vec         buoyancy,lame; /* Global, but for efficiency one could store local vectors */
  PetscInt    dim; /* 2 or 3 */
  PetscScalar rho,lambda,mu; /* constant material properties */
  PetscReal   xmin,xmax,ymin,ymax,zmin,zmax;
  PetscReal   dt;
  PetscInt    timesteps;
  PetscBool   dump_output;
} Ctx;

static PetscErrorCode CreateLame(Ctx*);
static PetscErrorCode ForceStress(const Ctx*,Vec,PetscReal);
static PetscErrorCode DumpVelocity(const Ctx*,Vec,PetscInt);
static PetscErrorCode DumpStress(const Ctx*,Vec,PetscInt);
static PetscErrorCode UpdateVelocity(const Ctx*,Vec,Vec,Vec);
static PetscErrorCode UpdateStress(const Ctx*,Vec,Vec,Vec);

int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  Ctx            ctx;
  Vec            velocity,stress;
  PetscInt       timestep;

  /* Initialize PETSc */
  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;

  /* Populate application context */
  ctx.dim         = 2;
  ctx.rho         = 1.0;
  ctx.lambda      = 1.0;
  ctx.mu          = 1.0;
  ctx.xmin        = 0.0; ctx.xmax = 1.0;
  ctx.ymin        = 0.0; ctx.ymax = 1.0;
  ctx.zmin        = 0.0; ctx.zmax = 1.0;
  ctx.dt          = 0.001;
  ctx.timesteps   = 100;
  ctx.dump_output = PETSC_TRUE;

  /* Update context from command line options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&ctx.dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&ctx.dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nsteps",&ctx.timesteps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-dump_output",&ctx.dump_output,NULL);CHKERRQ(ierr);

  /* Create a DMStag, with uniform coordinates, for the velocities */
  {
    PetscInt dof0,dof1,dof2,dof3;
    const PetscInt stencilWidth = 1;

    switch (ctx.dim) {
      case 2:
        dof0 = 0; dof1 = 1; dof2 = 0; /* 1 dof per cell boundary */
        ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,100,100,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,DMSTAG_STENCIL_BOX,stencilWidth,NULL,NULL,&ctx.dm_velocity);CHKERRQ(ierr);
        break;
      case 3:
        dof0 = 0; dof1 = 0; dof2 = 1; dof3 = 0; /* 1 dof per cell boundary */
        ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,30,30,30,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,dof3,DMSTAG_STENCIL_BOX,stencilWidth,NULL,NULL,NULL,&ctx.dm_velocity);CHKERRQ(ierr);
        break;
      default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %D",ctx.dim);
    }
  }
  ierr = DMSetFromOptions(ctx.dm_velocity);CHKERRQ(ierr); /* Options control velocity DM */
  ierr = DMSetUp(ctx.dm_velocity);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(ctx.dm_velocity,ctx.xmin,ctx.xmax,ctx.ymin,ctx.ymax,ctx.zmin,ctx.zmax);CHKERRQ(ierr);

  /* Create a second, compatible DMStag for the stresses */
  switch (ctx.dim) {
    case 2:
      /* One shear stress component on element corners, two shear stress components on elements */
      ierr = DMStagCreateCompatibleDMStag(ctx.dm_velocity,1,0,2,0,&ctx.dm_stress);CHKERRQ(ierr);
      break;
    case 3:
      /* One shear stress component on element edges, three shear stress components on elements */
      ierr = DMStagCreateCompatibleDMStag(ctx.dm_velocity,0,1,0,3,&ctx.dm_stress);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %D",ctx.dim);
  }
  ierr = DMSetUp(ctx.dm_stress);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(ctx.dm_stress,ctx.xmin,ctx.xmax,ctx.ymin,ctx.ymax,ctx.zmin,ctx.zmax);CHKERRQ(ierr);

  /* Create two additional DMStag objects for the buoyancy and Lame parameters */
  switch (ctx.dim) {
    case 2:
      /* buoyancy on element boundaries (edges) */
      ierr = DMStagCreateCompatibleDMStag(ctx.dm_velocity,0,1,0,0,&ctx.dm_buoyancy);CHKERRQ(ierr);
      break;
    case 3:
      /* buoyancy on element boundaries (faces) */
      ierr = DMStagCreateCompatibleDMStag(ctx.dm_velocity,0,0,1,0,&ctx.dm_buoyancy);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %D",ctx.dim);
  }
  ierr = DMSetUp(ctx.dm_buoyancy);CHKERRQ(ierr);

  switch (ctx.dim) {
    case 2:
      /* mu and lambda + 2*mu on element centers, mu on corners */
      ierr = DMStagCreateCompatibleDMStag(ctx.dm_velocity,1,0,2,0,&ctx.dm_lame);CHKERRQ(ierr);
      break;
    case 3:
      /* mu and lambda + 2*mu on element centers, mu on edges */
      ierr = DMStagCreateCompatibleDMStag(ctx.dm_velocity,0,1,0,2,&ctx.dm_lame);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %D",ctx.dim);
  }
  ierr = DMSetUp(ctx.dm_lame);CHKERRQ(ierr);

  /* Print out some info */
  {
    PetscInt    N[3];
    PetscScalar dx,Vp;

    ierr = DMStagGetGlobalSizes(ctx.dm_velocity,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
    dx = (ctx.xmax - ctx.xmin)/N[0];
    Vp = PetscSqrtScalar((ctx.lambda + 2 * ctx.mu) / ctx.rho);
    if (ctx.dim == 2) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Using a %D x %D mesh\n",N[0],N[1]);CHKERRQ(ierr);
    } else if (ctx.dim == 3) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Using a %D x %D x %D mesh\n",N[0],N[1],N[2]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dx: %g\n",dx);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"dt: %g\n",ctx.dt);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"P-wave velocity: %g\n",PetscSqrtScalar((ctx.lambda + 2 * ctx.mu) / ctx.rho));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"V_p dt / dx: %g\n",Vp * ctx.dt / dx);CHKERRQ(ierr);
  }

  /* Populate the coefficient arrays */
  ierr = CreateLame(&ctx);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx.dm_buoyancy,&ctx.buoyancy);CHKERRQ(ierr);
  ierr = VecSet(ctx.buoyancy,1.0/ctx.rho);CHKERRQ(ierr);

  /* Create vectors to store the system state */
  ierr = DMCreateGlobalVector(ctx.dm_velocity,&velocity);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx.dm_stress,&stress);CHKERRQ(ierr);

  /* Initial State */
  ierr = VecSet(velocity,0.0);CHKERRQ(ierr);
  ierr = VecSet(stress,0.0);CHKERRQ(ierr);
  ierr = ForceStress(&ctx,stress,0.0);CHKERRQ(ierr);
  if (ctx.dump_output) {
    ierr = DumpVelocity(&ctx,velocity,0);CHKERRQ(ierr);
    ierr = DumpStress(&ctx,stress,0);CHKERRQ(ierr);
  }

  /* Time Loop */
  for (timestep = 1; timestep <= ctx.timesteps; ++timestep) {
    const PetscReal t = timestep * ctx.dt;

    ierr = UpdateVelocity(&ctx,velocity,stress,ctx.buoyancy);CHKERRQ(ierr);
    ierr = UpdateStress(&ctx,velocity,stress,ctx.lame);CHKERRQ(ierr);
    ierr = ForceStress(&ctx,stress,t);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Timestep %d, t = %g\n",timestep,(double)t);CHKERRQ(ierr);
    if (ctx.dump_output) {
      ierr = DumpVelocity(&ctx,velocity,timestep);CHKERRQ(ierr);
      ierr = DumpStress(&ctx,stress,timestep);CHKERRQ(ierr);
    }
  }

  /* Clean up and finalize PETSc */
  ierr = VecDestroy(&velocity);CHKERRQ(ierr);
  ierr = VecDestroy(&stress);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.lame);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.buoyancy);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.dm_velocity);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.dm_stress);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.dm_buoyancy);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.dm_lame);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode CreateLame(Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt       N[3],ex,ey,ez,startx,starty,startz,nx,ny,nz,extrax,extray,extraz;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(ctx->dm_lame,&ctx->lame);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(ctx->dm_lame,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
  ierr = DMStagGetCorners(ctx->dm_buoyancy,&startx,&starty,&startz,&nx,&ny,&nz,&extrax,&extray,&extraz);CHKERRQ(ierr);
  if (ctx->dim == 2) {
    /* Element values */
    for (ey=starty; ey<starty+ny; ++ey) {
      for (ex=startx; ex<startx+nx; ++ex) {
        DMStagStencil pos;

        pos.i = ex; pos.j = ey; pos.c = 0; pos.loc = DMSTAG_ELEMENT;
        ierr = DMStagVecSetValuesStencil(ctx->dm_lame,ctx->lame,1,&pos,&ctx->lambda,INSERT_VALUES);CHKERRQ(ierr);
        pos.i = ex; pos.j = ey; pos.c = 1; pos.loc = DMSTAG_ELEMENT;
        ierr = DMStagVecSetValuesStencil(ctx->dm_lame,ctx->lame,1,&pos,&ctx->mu,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    /* Vertex Values */
    for (ey=starty; ey<starty+ny+extray; ++ey) {
      for (ex=startx; ex<startx+nx+extrax; ++ex) {
        DMStagStencil pos;

        pos.i = ex; pos.j = ey; pos.c = 0; pos.loc = DMSTAG_DOWN_LEFT;
        ierr = DMStagVecSetValuesStencil(ctx->dm_lame,ctx->lame,1,&pos,&ctx->mu,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else if (ctx->dim == 3) {
    /* Element values */
    for (ez=startz; ez<startz+nz; ++ez) {
      for (ey=starty; ey<starty+ny; ++ey) {
        for (ex=startx; ex<startx+nx; ++ex) {
        DMStagStencil pos;

        pos.i = ex; pos.j = ey; pos.k = ez; pos.c = 0; pos.loc = DMSTAG_ELEMENT;
        ierr = DMStagVecSetValuesStencil(ctx->dm_lame,ctx->lame,1,&pos,&ctx->lambda,INSERT_VALUES);CHKERRQ(ierr);
        pos.i = ex; pos.j = ey; pos.k = ez; pos.c = 1; pos.loc = DMSTAG_ELEMENT;
        ierr = DMStagVecSetValuesStencil(ctx->dm_lame,ctx->lame,1,&pos,&ctx->mu,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    /* Edge Values */
    for (ez=startz; ez<startz+nz+extraz; ++ez) {
      for (ey=starty; ey<starty+ny+extray; ++ey) {
        for (ex=startx; ex<startx+nx+extrax; ++ex) {
          DMStagStencil   pos;

          if (ex < N[0]) {
            pos.i = ex; pos.j = ey; pos.k = ez; pos.c = 0; pos.loc = DMSTAG_BACK_DOWN;
            ierr = DMStagVecSetValuesStencil(ctx->dm_lame,ctx->lame,1,&pos,&ctx->mu,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (ey < N[1]) {
            pos.i = ex; pos.j = ey; pos.k = ez; pos.c = 0; pos.loc = DMSTAG_BACK_LEFT;
            ierr = DMStagVecSetValuesStencil(ctx->dm_lame,ctx->lame,1,&pos,&ctx->mu,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (ez < N[2]) {
            pos.i = ex; pos.j = ey; pos.k = ez; pos.c = 0; pos.loc = DMSTAG_DOWN_LEFT;
            ierr = DMStagVecSetValuesStencil(ctx->dm_lame,ctx->lame,1,&pos,&ctx->mu,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  } else SETERRQ1(PetscObjectComm((PetscObject)ctx->dm_velocity),PETSC_ERR_SUP,"Unsupported dim %d",ctx->dim);
  ierr = VecAssemblyBegin(ctx->lame);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ctx->lame);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ForceStress(const Ctx *ctx,Vec stress, PetscReal t)
{
  PetscErrorCode    ierr;
  PetscInt          start[3],n[3],N[3];
  DMStagStencil     pos;
  PetscBool         this_rank;
  const PetscScalar val = PetscExpReal(-100.0 * t);

  PetscFunctionBeginUser;
  ierr = DMStagGetGlobalSizes(ctx->dm_stress,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
  ierr = DMStagGetCorners(ctx->dm_stress,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],NULL,NULL,NULL);CHKERRQ(ierr);

  /* Normal stresses at a single point */
  this_rank = (PetscBool) (start[0] <= N[0]/2 && N[0]/2 <= start[0] + n[0]);
  this_rank = (PetscBool) (this_rank && start[1] <= N[1]/2 && N[1]/2 <= start[1] + n[1]);
  if(ctx->dim == 3 ) this_rank = (PetscBool) (this_rank && start[2] <= N[2]/2 && N[2]/2 <= start[2] + n[2]);
  if (this_rank) {
    /* Note integer division to pick element near the center */
    pos.i = N[0]/2; pos.j = N[1]/2; pos.k = N[2]/2; pos.c = 0; pos.loc = DMSTAG_ELEMENT;
    ierr = DMStagVecSetValuesStencil(ctx->dm_stress,stress,1,&pos,&val,INSERT_VALUES);CHKERRQ(ierr);
    pos.i = N[0]/2; pos.j = N[1]/2; pos.k = N[2]/2; pos.c = 1; pos.loc = DMSTAG_ELEMENT;
    ierr = DMStagVecSetValuesStencil(ctx->dm_stress,stress,1,&pos,&val,INSERT_VALUES);CHKERRQ(ierr);
    if (ctx->dim == 3) {
      pos.i = N[0]/2; pos.j = N[1]/2; pos.k = N[2]/2; pos.c = 2; pos.loc = DMSTAG_ELEMENT;
      ierr = DMStagVecSetValuesStencil(ctx->dm_stress,stress,1,&pos,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = VecAssemblyBegin(stress);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(stress);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode UpdateVelocity_2d(const Ctx *ctx,Vec velocity,Vec stress, Vec buoyancy)
{
  PetscErrorCode    ierr;
  Vec               velocity_local,stress_local,buoyancy_local;
  PetscInt          ex,ey,startx,starty,nx,ny;
  PetscInt          slot_coord_next,slot_coord_element,slot_coord_prev;
  PetscInt          slot_vx_left,slot_vy_down,slot_buoyancy_down,slot_buoyancy_left;
  PetscInt          slot_txx,slot_tyy,slot_txy_downleft,slot_txy_downright,slot_txy_upleft;
  const PetscScalar **arr_coord_x,**arr_coord_y;
  const PetscScalar ***arr_stress,***arr_buoyancy;
  PetscScalar       ***arr_velocity;

  PetscFunctionBeginUser;

  /* Prepare direct access to buoyancy data */
  ierr = DMStagGetLocationSlot(ctx->dm_buoyancy,DMSTAG_LEFT,0,&slot_buoyancy_left);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_buoyancy,DMSTAG_DOWN,0,&slot_buoyancy_down);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_buoyancy,&buoyancy_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_buoyancy,buoyancy,INSERT_VALUES,buoyancy_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(ctx->dm_buoyancy,buoyancy_local,&arr_buoyancy);CHKERRQ(ierr);

  /* Prepare read-only access to stress data */
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,   0,&slot_txx);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,   1,&slot_tyy);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_UP_LEFT,   0,&slot_txy_upleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_DOWN_LEFT, 0,&slot_txy_downleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_DOWN_RIGHT,0,&slot_txy_downright);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_stress,&stress_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_stress,stress,INSERT_VALUES,stress_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(ctx->dm_stress,stress_local,&arr_stress);CHKERRQ(ierr);

  /* Prepare read-write access to velocity data */
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_LEFT,0,&slot_vx_left);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_DOWN,0,&slot_vy_down);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_velocity,velocity,INSERT_VALUES,velocity_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(ctx->dm_velocity,velocity_local,&arr_velocity);CHKERRQ(ierr);

  /* Prepare read-only access to coordinate data */
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_LEFT,&slot_coord_prev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_RIGHT,&slot_coord_next);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_ELEMENT,&slot_coord_element);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArrays(ctx->dm_velocity,&arr_coord_x,&arr_coord_y,NULL);CHKERRQ(ierr);

  /* Iterate over interior of the domain, updating the velocities */
  ierr = DMStagGetCorners(ctx->dm_velocity,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  for (ey = starty; ey < starty + ny; ++ey) {
    for (ex = startx; ex < startx + nx; ++ex) {

      /* Update y-velocity */
      if (ey > 0) {
        const PetscScalar dx = arr_coord_x[ex][slot_coord_next]    - arr_coord_x[ex  ][slot_coord_prev];
        const PetscScalar dy = arr_coord_y[ey][slot_coord_element] - arr_coord_y[ey-1][slot_coord_element];
        const PetscScalar B = arr_buoyancy[ey][ex][slot_buoyancy_down];

        arr_velocity[ey][ex][slot_vy_down] += B * ctx->dt * (
              (arr_stress[ey][ex][slot_txy_downright] - arr_stress[ey  ][ex][slot_txy_downleft]) / dx
            + (arr_stress[ey][ex][slot_tyy]           - arr_stress[ey-1][ex][slot_tyy]         ) / dy );
      }

      /* Update x-velocity */
      if (ex > 0) {
        const PetscScalar dx = arr_coord_x[ex][slot_coord_element] - arr_coord_x[ex-1][slot_coord_element];
        const PetscScalar dy = arr_coord_y[ey][slot_coord_next]    - arr_coord_y[ey  ][slot_coord_prev];
        const PetscScalar B = arr_buoyancy[ey][ex][slot_buoyancy_left];

        arr_velocity[ey][ex][slot_vx_left] += B * ctx->dt * (
              (arr_stress[ey][ex][slot_txx]        - arr_stress[ey][ex-1][slot_txx]         ) / dx
            + (arr_stress[ey][ex][slot_txy_upleft] - arr_stress[ey][ex  ][slot_txy_downleft]) / dy );
      }
    }
  }

  /* Restore all access */
  ierr = DMStagVecRestoreArrayRead(ctx->dm_buoyancy,buoyancy_local,&arr_buoyancy);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_buoyancy,&buoyancy_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(ctx->dm_stress,stress_local,&arr_stress);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_stress,&stress_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(ctx->dm_velocity,velocity_local,&arr_velocity);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(ctx->dm_velocity,velocity_local,INSERT_VALUES,velocity);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArrays(ctx->dm_velocity,&arr_coord_x,&arr_coord_y,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode UpdateVelocity_3d(const Ctx *ctx,Vec velocity,Vec stress, Vec buoyancy)
{
  PetscErrorCode    ierr;
  Vec               velocity_local,stress_local,buoyancy_local;
  PetscInt          ex,ey,ez,startx,starty,startz,nx,ny,nz;
  PetscInt          slot_coord_next,slot_coord_element,slot_coord_prev;
  PetscInt          slot_vx_left,slot_vy_down,slot_vz_back,slot_buoyancy_down,slot_buoyancy_left,slot_buoyancy_back;
  PetscInt          slot_txx,slot_tyy,slot_tzz;
  PetscInt          slot_txy_downleft,slot_txy_downright,slot_txy_upleft;
  PetscInt          slot_txz_backleft,slot_txz_backright,slot_txz_frontleft;
  PetscInt          slot_tyz_backdown,slot_tyz_frontdown,slot_tyz_backup;
  const PetscScalar **arr_coord_x,**arr_coord_y,**arr_coord_z;
  const PetscScalar ****arr_stress,****arr_buoyancy;
  PetscScalar       ****arr_velocity;

  PetscFunctionBeginUser;

  /* Prepare direct access to buoyancy data */
  ierr = DMStagGetLocationSlot(ctx->dm_buoyancy,DMSTAG_LEFT,0,&slot_buoyancy_left);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_buoyancy,DMSTAG_DOWN,0,&slot_buoyancy_down);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_buoyancy,DMSTAG_BACK,0,&slot_buoyancy_back);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_buoyancy,&buoyancy_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_buoyancy,buoyancy,INSERT_VALUES,buoyancy_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(ctx->dm_buoyancy,buoyancy_local,&arr_buoyancy);CHKERRQ(ierr);

  /* Prepare read-only access to stress data */
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,   0,&slot_txx);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,   1,&slot_tyy);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,   2,&slot_tzz);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_UP_LEFT,   0,&slot_txy_upleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_DOWN_LEFT, 0,&slot_txy_downleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_DOWN_RIGHT,0,&slot_txy_downright);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_BACK_LEFT, 0,&slot_txz_backleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_BACK_RIGHT,0,&slot_txz_backright);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_FRONT_LEFT,0,&slot_txz_frontleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_BACK_DOWN, 0,&slot_tyz_backdown);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_BACK_UP,   0,&slot_tyz_backup);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_FRONT_DOWN,0,&slot_tyz_frontdown);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_stress,&stress_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_stress,stress,INSERT_VALUES,stress_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(ctx->dm_stress,stress_local,&arr_stress);CHKERRQ(ierr);

  /* Prepare read-write access to velocity data */
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_LEFT,0,&slot_vx_left);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_DOWN,0,&slot_vy_down);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_BACK,0,&slot_vz_back);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_velocity,velocity,INSERT_VALUES,velocity_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(ctx->dm_velocity,velocity_local,&arr_velocity);CHKERRQ(ierr);

  /* Prepare read-only access to coordinate data */
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_LEFT,&slot_coord_prev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_RIGHT,&slot_coord_next);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_ELEMENT,&slot_coord_element);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArrays(ctx->dm_velocity,&arr_coord_x,&arr_coord_y,&arr_coord_z);CHKERRQ(ierr);

  /* Iterate over interior of the domain, updating the velocities */
  ierr = DMStagGetCorners(ctx->dm_velocity,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL);CHKERRQ(ierr);
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {

        /* Update y-velocity */
        if (ey > 0) {
          const PetscScalar dx = arr_coord_x[ex][slot_coord_next]    - arr_coord_x[ex  ][slot_coord_prev];
          const PetscScalar dy = arr_coord_y[ey][slot_coord_element] - arr_coord_y[ey-1][slot_coord_element];
          const PetscScalar dz = arr_coord_z[ez][slot_coord_next]    - arr_coord_z[ez  ][slot_coord_prev];
          const PetscScalar B = arr_buoyancy[ez][ey][ex][slot_buoyancy_down];

          arr_velocity[ez][ey][ex][slot_vy_down] += B * ctx->dt * (
                (arr_stress[ez][ey][ex][slot_txy_downright] - arr_stress[ez][ey  ][ex][slot_txy_downleft]) / dx
              + (arr_stress[ez][ey][ex][slot_tyy]           - arr_stress[ez][ey-1][ex][slot_tyy]         ) / dy
              + (arr_stress[ez][ey][ex][slot_tyz_frontdown] - arr_stress[ez][ey  ][ex][slot_tyz_backdown]) / dz);
        }

        /* Update x-velocity */
        if (ex > 0) {
          const PetscScalar dx = arr_coord_x[ex][slot_coord_element] - arr_coord_x[ex-1][slot_coord_element];
          const PetscScalar dy = arr_coord_y[ey][slot_coord_next]    - arr_coord_y[ey  ][slot_coord_prev];
          const PetscScalar dz = arr_coord_z[ez][slot_coord_next]    - arr_coord_z[ez  ][slot_coord_prev];
          const PetscScalar B = arr_buoyancy[ez][ey][ex][slot_buoyancy_left];

          arr_velocity[ez][ey][ex][slot_vx_left] += B * ctx->dt * (
                (arr_stress[ez][ey][ex][slot_txx]           - arr_stress[ez][ey][ex-1][slot_txx]         ) / dx
              + (arr_stress[ez][ey][ex][slot_txy_upleft]    - arr_stress[ez][ey][ex  ][slot_txy_downleft]) / dy
              + (arr_stress[ez][ey][ex][slot_txz_frontleft] - arr_stress[ez][ey][ex  ][slot_txz_backleft]) / dz );
        }

        /* Update z-velocity */
        if (ez > 0) {
          const PetscScalar dx = arr_coord_x[ex][slot_coord_next]    - arr_coord_x[ex  ][slot_coord_prev];
          const PetscScalar dy = arr_coord_y[ey][slot_coord_next]    - arr_coord_y[ey  ][slot_coord_prev];
          const PetscScalar dz = arr_coord_z[ez][slot_coord_element] - arr_coord_z[ez-1][slot_coord_element];
          const PetscScalar B = arr_buoyancy[ez][ey][ex][slot_buoyancy_back];

          arr_velocity[ez][ey][ex][slot_vz_back] += B * ctx->dt * (
              (arr_stress[ez][ey][ex][slot_txz_backright] - arr_stress[ez  ][ey][ex][slot_txz_backleft] ) / dx
            + (arr_stress[ez][ey][ex][slot_tyz_backup]    - arr_stress[ez  ][ey][ex][slot_tyz_backdown] ) / dy
            + (arr_stress[ez][ey][ex][slot_tzz]           - arr_stress[ez-1][ey][ex][slot_tzz]          ) / dz);
        }
      }
    }
  }

  /* Restore all access */
  ierr = DMStagVecRestoreArrayRead(ctx->dm_buoyancy,buoyancy_local,&arr_buoyancy);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_buoyancy,&buoyancy_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(ctx->dm_stress,stress_local,&arr_stress);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_stress,&stress_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(ctx->dm_velocity,velocity_local,&arr_velocity);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(ctx->dm_velocity,velocity_local,INSERT_VALUES,velocity);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArrays(ctx->dm_velocity,&arr_coord_x,&arr_coord_y,&arr_coord_z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode UpdateVelocity(const Ctx *ctx,Vec velocity,Vec stress, Vec buoyancy)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (ctx->dim == 2) {
    ierr = UpdateVelocity_2d(ctx,velocity,stress,buoyancy);CHKERRQ(ierr);
  } else if (ctx->dim == 3) {
    ierr = UpdateVelocity_3d(ctx,velocity,stress,buoyancy);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)ctx->dm_velocity),PETSC_ERR_SUP,"Unsupported dim %d",ctx->dim);
  PetscFunctionReturn(0);
}

static PetscErrorCode UpdateStress_2d(const Ctx *ctx,Vec velocity,Vec stress, Vec lame)
{
  PetscErrorCode    ierr;
  Vec               velocity_local,stress_local,lame_local;
  PetscInt          ex,ey,startx,starty,nx,ny;
  PetscInt          slot_coord_next,slot_coord_element,slot_coord_prev;
  PetscInt          slot_vx_left,slot_vy_down,slot_vx_right,slot_vy_up;
  PetscInt          slot_mu_element,slot_lambda_element,slot_mu_downleft;
  PetscInt          slot_txx,slot_tyy,slot_txy_downleft;
  const PetscScalar **arr_coord_x,**arr_coord_y;
  const PetscScalar ***arr_velocity,***arr_lame;
  PetscScalar       ***arr_stress;

  PetscFunctionBeginUser;

  /* Prepare read-write access to stress data */
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,0,&slot_txx);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,1,&slot_tyy);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_DOWN_LEFT,0,&slot_txy_downleft);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_stress,&stress_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_stress,stress,INSERT_VALUES,stress_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(ctx->dm_stress,stress_local,&arr_stress);CHKERRQ(ierr);

  /* Prepare read-only access to velocity data */
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_LEFT,0,&slot_vx_left);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_RIGHT,0,&slot_vx_right);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_DOWN,0,&slot_vy_down);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_UP,0,&slot_vy_up);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_velocity,velocity,INSERT_VALUES,velocity_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(ctx->dm_velocity,velocity_local,&arr_velocity);CHKERRQ(ierr);

  /* Prepare read-only access to Lame' data */
  ierr = DMStagGetLocationSlot(ctx->dm_lame,DMSTAG_ELEMENT,0,&slot_lambda_element);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_lame,DMSTAG_ELEMENT,1,&slot_mu_element);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_lame,DMSTAG_DOWN_LEFT,0,&slot_mu_downleft);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_lame,&lame_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_lame,lame,INSERT_VALUES,lame_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(ctx->dm_lame,lame_local,&arr_lame);CHKERRQ(ierr);

  /* Prepare read-only access to coordinate data */
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_LEFT,&slot_coord_prev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_RIGHT,&slot_coord_next);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_ELEMENT,&slot_coord_element);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArrays(ctx->dm_velocity,&arr_coord_x,&arr_coord_y,NULL);CHKERRQ(ierr);

  /* Iterate over the interior of the domain, updating the stresses */
  ierr = DMStagGetCorners(ctx->dm_velocity,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  for (ey = starty; ey < starty + ny; ++ey) {
    for (ex = startx; ex < startx + nx; ++ex) {

      /* Update tau_xx and tau_yy*/
      {
        const PetscScalar dx   = arr_coord_x[ex][slot_coord_next] - arr_coord_x[ex][slot_coord_prev];
        const PetscScalar dy   = arr_coord_y[ey][slot_coord_next] - arr_coord_y[ey][slot_coord_prev];
        const PetscScalar L    = arr_lame[ey][ex][slot_lambda_element];
        const PetscScalar Lp2M = arr_lame[ey][ex][slot_lambda_element] + 2 * arr_lame[ey][ex][slot_mu_element];

        arr_stress[ey][ex][slot_txx] +=
            Lp2M * ctx->dt * (arr_velocity[ey][ex][slot_vx_right] - arr_velocity[ey][ex][slot_vx_left]) / dx
          + L    * ctx->dt * (arr_velocity[ey][ex][slot_vy_up]    - arr_velocity[ey][ex][slot_vy_down]) / dy;

        arr_stress[ey][ex][slot_tyy] +=
            Lp2M * ctx->dt * (arr_velocity[ey][ex][slot_vy_up]    - arr_velocity[ey][ex][slot_vy_down]) / dy
          + L    * ctx->dt * (arr_velocity[ey][ex][slot_vx_right] - arr_velocity[ey][ex][slot_vx_left]) / dx;
      }

      /* Update tau_xy */
      if (ex > 0 && ey > 0) {
        const PetscScalar dx = arr_coord_x[ex][slot_coord_element] - arr_coord_x[ex-1][slot_coord_element];
        const PetscScalar dy = arr_coord_y[ey][slot_coord_element] - arr_coord_y[ey-1][slot_coord_element];
        const PetscScalar M  = arr_lame[ey][ex][slot_mu_downleft];

        arr_stress[ey][ex][slot_txy_downleft] += M * ctx->dt * (
            (arr_velocity[ey][ex][slot_vx_left] - arr_velocity[ey-1][ex][slot_vx_left]) / dy
          + (arr_velocity[ey][ex][slot_vy_down] - arr_velocity[ey][ex-1][slot_vy_down]) / dx );
      }
    }
  }

  /* Restore all access */
  ierr = DMStagVecRestoreArray(ctx->dm_stress,stress_local,&arr_stress);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(ctx->dm_stress,stress_local,INSERT_VALUES,stress);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_stress,&stress_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(ctx->dm_velocity,velocity_local,&arr_velocity);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(ctx->dm_lame,lame_local,&arr_lame);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_lame,&lame_local);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArrays(ctx->dm_velocity,&arr_coord_x,&arr_coord_y,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode UpdateStress_3d(const Ctx *ctx,Vec velocity,Vec stress, Vec lame)
{
  PetscErrorCode    ierr;
  Vec               velocity_local,stress_local,lame_local;
  PetscInt          ex,ey,ez,startx,starty,startz,nx,ny,nz;
  PetscInt          slot_coord_next,slot_coord_element,slot_coord_prev;
  PetscInt          slot_vx_left,slot_vy_down,slot_vx_right,slot_vy_up,slot_vz_back,slot_vz_front;
  PetscInt          slot_mu_element,slot_lambda_element,slot_mu_downleft,slot_mu_backdown,slot_mu_backleft;
  PetscInt          slot_txx,slot_tyy,slot_tzz,slot_txy_downleft,slot_txz_backleft,slot_tyz_backdown;
  const PetscScalar **arr_coord_x,**arr_coord_y,**arr_coord_z;
  const PetscScalar ****arr_velocity,****arr_lame;
  PetscScalar       ****arr_stress;

  PetscFunctionBeginUser;

  /* Prepare read-write access to stress data */
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,0,&slot_txx);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,1,&slot_tyy);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_ELEMENT,2,&slot_tzz);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_DOWN_LEFT,0,&slot_txy_downleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_BACK_LEFT,0,&slot_txz_backleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_stress,DMSTAG_BACK_DOWN,0,&slot_tyz_backdown);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_stress,&stress_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_stress,stress,INSERT_VALUES,stress_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(ctx->dm_stress,stress_local,&arr_stress);CHKERRQ(ierr);

  /* Prepare read-only access to velocity data */
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_LEFT, 0,&slot_vx_left);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_RIGHT,0,&slot_vx_right);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_DOWN, 0,&slot_vy_down);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_UP,   0,&slot_vy_up);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_BACK, 0,&slot_vz_back);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_velocity,DMSTAG_FRONT,0,&slot_vz_front);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_velocity,velocity,INSERT_VALUES,velocity_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(ctx->dm_velocity,velocity_local,&arr_velocity);CHKERRQ(ierr);

  /* Prepare read-only access to Lame' data */
  ierr = DMStagGetLocationSlot(ctx->dm_lame,DMSTAG_ELEMENT,  0,&slot_lambda_element);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_lame,DMSTAG_ELEMENT,  1,&slot_mu_element);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_lame,DMSTAG_DOWN_LEFT,0,&slot_mu_downleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_lame,DMSTAG_BACK_LEFT,0,&slot_mu_backleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(ctx->dm_lame,DMSTAG_BACK_DOWN,0,&slot_mu_backdown);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_lame,&lame_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_lame,lame,INSERT_VALUES,lame_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(ctx->dm_lame,lame_local,&arr_lame);CHKERRQ(ierr);

  /* Prepare read-only access to coordinate data */
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_LEFT,&slot_coord_prev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_RIGHT,&slot_coord_next);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(ctx->dm_velocity,DMSTAG_ELEMENT,&slot_coord_element);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArrays(ctx->dm_velocity,&arr_coord_x,&arr_coord_y,&arr_coord_z);CHKERRQ(ierr);

  /* Iterate over the interior of the domain, updating the stresses */
  ierr = DMStagGetCorners(ctx->dm_velocity,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL);CHKERRQ(ierr);
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {

        /* Update tau_xx, tau_yy, and tau_zz*/
        {
          const PetscScalar dx   = arr_coord_x[ex][slot_coord_next] - arr_coord_x[ex][slot_coord_prev];
          const PetscScalar dy   = arr_coord_y[ey][slot_coord_next] - arr_coord_y[ey][slot_coord_prev];
          const PetscScalar dz   = arr_coord_z[ez][slot_coord_next] - arr_coord_z[ez][slot_coord_prev];
          const PetscScalar L    = arr_lame[ez][ey][ex][slot_lambda_element];
          const PetscScalar Lp2M = arr_lame[ez][ey][ex][slot_lambda_element] + 2 * arr_lame[ez][ey][ex][slot_mu_element];

          arr_stress[ez][ey][ex][slot_txx] +=
              Lp2M * ctx->dt * (arr_velocity[ez][ey][ex][slot_vx_right] - arr_velocity[ez][ey][ex][slot_vx_left]) / dx
            + L    * ctx->dt * (arr_velocity[ez][ey][ex][slot_vy_up]    - arr_velocity[ez][ey][ex][slot_vy_down]) / dy
            + L    * ctx->dt * (arr_velocity[ez][ey][ex][slot_vz_front] - arr_velocity[ez][ey][ex][slot_vz_back]) / dz;

          arr_stress[ez][ey][ex][slot_tyy] +=
              Lp2M * ctx->dt * (arr_velocity[ez][ey][ex][slot_vy_up]    - arr_velocity[ez][ey][ex][slot_vy_down]) / dy
            + L    * ctx->dt * (arr_velocity[ez][ey][ex][slot_vz_front] - arr_velocity[ez][ey][ex][slot_vz_back]) / dz
            + L    * ctx->dt * (arr_velocity[ez][ey][ex][slot_vx_right] - arr_velocity[ez][ey][ex][slot_vx_left]) / dx;

          arr_stress[ez][ey][ex][slot_tzz] +=
              Lp2M * ctx->dt * (arr_velocity[ez][ey][ex][slot_vz_front] - arr_velocity[ez][ey][ex][slot_vz_back]) / dz
            + L    * ctx->dt * (arr_velocity[ez][ey][ex][slot_vx_right] - arr_velocity[ez][ey][ex][slot_vx_left]) / dx
            + L    * ctx->dt * (arr_velocity[ez][ey][ex][slot_vy_up]    - arr_velocity[ez][ey][ex][slot_vy_down]) / dy;
        }

        /* Update tau_xy, tau_xz, tau_yz */
        {
          PetscScalar dx,dy,dz;

          if (ex > 0)  dx = arr_coord_x[ex][slot_coord_element] - arr_coord_x[ex-1][slot_coord_element];
          if (ey > 0)  dy = arr_coord_y[ey][slot_coord_element] - arr_coord_y[ey-1][slot_coord_element];
          if (ez > 0)  dz = arr_coord_z[ez][slot_coord_element] - arr_coord_z[ez-1][slot_coord_element];

          if (ex > 0 && ey > 0) {
            const PetscScalar M = arr_lame[ez][ey][ex][slot_mu_downleft];

            arr_stress[ez][ey][ex][slot_txy_downleft] += M * ctx->dt * (
                  (arr_velocity[ez][ey][ex][slot_vx_left] - arr_velocity[ez][ey-1][ex][slot_vx_left]) / dy
                + (arr_velocity[ez][ey][ex][slot_vy_down] - arr_velocity[ez][ey][ex-1][slot_vy_down]) / dx );
          }

          /* Update tau_xz */
          if (ex > 0 && ez > 0) {
            const PetscScalar M  = arr_lame[ez][ey][ex][slot_mu_backleft];

            arr_stress[ez][ey][ex][slot_txz_backleft] += M * ctx->dt * (
                (arr_velocity[ez][ey][ex][slot_vx_left] - arr_velocity[ez-1][ey][ex][slot_vx_left]) / dz
              + (arr_velocity[ez][ey][ex][slot_vz_back] - arr_velocity[ez][ey][ex-1][slot_vz_back]) / dx);
          }

          /* Update tau_yz */
          if (ey > 0 && ez > 0) {
            const PetscScalar M  = arr_lame[ez][ey][ex][slot_mu_backdown];

            arr_stress[ez][ey][ex][slot_tyz_backdown] += M * ctx->dt * (
                (arr_velocity[ez][ey][ex][slot_vy_down] - arr_velocity[ez-1][ey][ex][slot_vy_down]) / dz
              + (arr_velocity[ez][ey][ex][slot_vz_back] - arr_velocity[ez][ey-1][ex][slot_vz_back]) / dy);
          }
        }
      }
    }
  }

  /* Restore all access */
  ierr = DMStagVecRestoreArray(ctx->dm_stress,stress_local,&arr_stress);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(ctx->dm_stress,stress_local,INSERT_VALUES,stress);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_stress,&stress_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(ctx->dm_velocity,velocity_local,&arr_velocity);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(ctx->dm_lame,lame_local,&arr_lame);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_lame,&lame_local);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArrays(ctx->dm_velocity,&arr_coord_x,&arr_coord_y,&arr_coord_z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode UpdateStress(const Ctx *ctx,Vec velocity,Vec stress, Vec lame)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (ctx->dim == 2) {
    ierr = UpdateStress_2d(ctx,velocity,stress,lame);CHKERRQ(ierr);
  } else if (ctx->dim ==3) {
    ierr = UpdateStress_3d(ctx,velocity,stress,lame);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DumpStress(const Ctx *ctx,Vec stress,PetscInt timestep)
{
  PetscErrorCode ierr;
  DM             da_normal,da_shear = NULL;
  Vec            vec_normal,vec_shear = NULL;

  PetscFunctionBeginUser;

    ierr = DMStagVecSplitToDMDA(ctx->dm_stress,stress,DMSTAG_ELEMENT,-ctx->dim,&da_normal,&vec_normal);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec_normal,"normal stresses");CHKERRQ(ierr);

    /* Dump element-based fields to a .vtr file */
    {
      PetscViewer viewer;
      char        filename[PETSC_MAX_PATH_LEN];

      ierr = PetscSNPrintf(filename,sizeof(filename),"ex6_stress_normal_%.4D.vtr",timestep);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_normal),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec_normal,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Created %s\n",filename);CHKERRQ(ierr);
    }

  if (ctx->dim == 2) {
    /* (2D only) Dump vertex-based fields to a second .vtr file */
    ierr = DMStagVecSplitToDMDA(ctx->dm_stress,stress,DMSTAG_DOWN_LEFT,0,&da_shear,&vec_shear);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec_shear,"shear stresses");CHKERRQ(ierr);

    {
      PetscViewer viewer;
      char        filename[PETSC_MAX_PATH_LEN];

      ierr = PetscSNPrintf(filename,sizeof(filename),"ex6_stress_shear_%.4D.vtr",timestep);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_normal),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec_shear,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Created %s\n",filename);CHKERRQ(ierr);
    }
  }

  /* Destroy DMDAs and Vecs */
  ierr = DMDestroy(&da_normal);CHKERRQ(ierr);
  ierr = DMDestroy(&da_shear);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_normal);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_shear);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DumpVelocity(const Ctx *ctx,Vec velocity,PetscInt timestep)
{
  PetscErrorCode ierr;
  DM             dmVelAvg;
  Vec            velAvg;
  DM             daVelAvg;
  Vec            vecVelAvg;
  Vec            velocity_local;
  PetscInt       ex,ey,ez,startx,starty,startz,nx,ny,nz;

  PetscFunctionBeginUser;

  if (ctx->dim == 2) {
    ierr = DMStagCreateCompatibleDMStag(ctx->dm_velocity,0,0,2,0,&dmVelAvg);CHKERRQ(ierr); /* 2 dof per element */
  } else if (ctx->dim == 3) {
    ierr = DMStagCreateCompatibleDMStag(ctx->dm_velocity,0,0,0,3,&dmVelAvg);CHKERRQ(ierr); /* 3 dof per element */
  } else SETERRQ1(PetscObjectComm((PetscObject)ctx->dm_velocity),PETSC_ERR_SUP,"Unsupported dim %d",ctx->dim);
  ierr = DMSetUp(dmVelAvg);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmVelAvg,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax,ctx->zmin,ctx->zmax);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmVelAvg,&velAvg);CHKERRQ(ierr);
  ierr = DMGetLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(ctx->dm_velocity,velocity,INSERT_VALUES,velocity_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmVelAvg,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL);CHKERRQ(ierr);
  if (ctx->dim == 2) {
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil from[4],to[2];
        PetscScalar   valFrom[4],valTo[2];

        from[0].i = ex; from[0].j = ey; from[0].loc = DMSTAG_UP;    from[0].c = 0;
        from[1].i = ex; from[1].j = ey; from[1].loc = DMSTAG_DOWN;  from[1].c = 0;
        from[2].i = ex; from[2].j = ey; from[2].loc = DMSTAG_LEFT;  from[2].c = 0;
        from[3].i = ex; from[3].j = ey; from[3].loc = DMSTAG_RIGHT; from[3].c = 0;
        ierr = DMStagVecGetValuesStencil(ctx->dm_velocity,velocity_local,4,from,valFrom);CHKERRQ(ierr);
        to[0].i = ex; to[0].j = ey; to[0].loc = DMSTAG_ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = ex; to[1].j = ey; to[1].loc = DMSTAG_ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        ierr = DMStagVecSetValuesStencil(dmVelAvg,velAvg,2,to,valTo,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else if (ctx->dim == 3) {
    for (ez = startz; ez<startz+nz; ++ez) {
      for (ey = starty; ey<starty+ny; ++ey) {
        for (ex = startx; ex<startx+nx; ++ex) {
          DMStagStencil from[6],to[3];
          PetscScalar   valFrom[6],valTo[3];

          from[0].i = ex; from[0].j = ey; from[0].k = ez; from[0].loc = DMSTAG_UP;    from[0].c = 0;
          from[1].i = ex; from[1].j = ey; from[1].k = ez; from[1].loc = DMSTAG_DOWN;  from[1].c = 0;
          from[2].i = ex; from[2].j = ey; from[2].k = ez; from[2].loc = DMSTAG_LEFT;  from[2].c = 0;
          from[3].i = ex; from[3].j = ey; from[3].k = ez; from[3].loc = DMSTAG_RIGHT; from[3].c = 0;
          from[4].i = ex; from[4].j = ey; from[4].k = ez; from[4].loc = DMSTAG_FRONT; from[4].c = 0;
          from[5].i = ex; from[5].j = ey; from[5].k = ez; from[5].loc = DMSTAG_BACK;  from[5].c = 0;
          ierr = DMStagVecGetValuesStencil(ctx->dm_velocity,velocity_local,6,from,valFrom);CHKERRQ(ierr);
          to[0].i = ex; to[0].j = ey; to[0].k = ez; to[0].loc = DMSTAG_ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
          to[1].i = ex; to[1].j = ey; to[1].k = ez; to[1].loc = DMSTAG_ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
          to[2].i = ex; to[2].j = ey; to[2].k = ez; to[2].loc = DMSTAG_ELEMENT;    to[2].c = 2; valTo[2] = 0.5 * (valFrom[4] + valFrom[5]);
          ierr = DMStagVecSetValuesStencil(dmVelAvg,velAvg,3,to,valTo,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  } else SETERRQ1(PetscObjectComm((PetscObject)ctx->dm_velocity),PETSC_ERR_SUP,"Unsupported dim %d",ctx->dim);
  ierr = VecAssemblyBegin(velAvg);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(velAvg);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ctx->dm_velocity,&velocity_local);CHKERRQ(ierr);

  ierr = DMStagVecSplitToDMDA(dmVelAvg,velAvg,DMSTAG_ELEMENT,-3,&daVelAvg,&vecVelAvg);CHKERRQ(ierr); /* note -3 : pad with zero in 2D case */
  ierr = PetscObjectSetName((PetscObject)vecVelAvg,"Velocity (Averaged)");CHKERRQ(ierr);

  /* Dump element-based fields to a .vtr file */
  {
    PetscViewer viewer;
    char        filename[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(filename,sizeof(filename),"ex6_velavg_%.4D.vtr",timestep);CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVelAvg),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(vecVelAvg,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Created %s\n",filename);CHKERRQ(ierr);
  }

  /* Destroy DMDAs and Vecs */
  ierr = VecDestroy(&vecVelAvg);CHKERRQ(ierr);
  ierr = DMDestroy(&daVelAvg);CHKERRQ(ierr);
  ierr = VecDestroy(&velAvg);CHKERRQ(ierr);
  ierr = DMDestroy(&dmVelAvg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      requires: !complex
      nsize: 1
      args: -stag_grid_x 12 -stag_grid_y 7 -nsteps 4 -dump_output 0

   test:
      suffix: 2
      requires: !complex
      nsize: 9
      args: -stag_grid_x 12 -stag_grid_y 15 -nsteps 3 -dump_output 0

   test:
      suffix: 3
      requires: !complex
      nsize: 1
      args: -stag_grid_x 12 -stag_grid_y 7 -stag_grid_z 11 -nsteps 2 -dim 3 -dump_output 0

   test:
      suffix: 4
      requires: !complex
      nsize: 12
      args: -stag_grid_x 12 -stag_grid_y 15 -stag_grid_z 8 -nsteps 3 -dim 3 -dump_output 0

TEST*/
