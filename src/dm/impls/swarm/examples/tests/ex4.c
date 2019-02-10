static char help[] = "Example of simple hamiltonian system with particles and a basic symplectic integrator\n";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscksp.h>
#include <petsc/private/petscfeimpl.h>
#include <petscts.h>
typedef struct {
  PetscInt       dim;                              /* The topological mesh dimension */
  PetscInt       nts;                              /* print the energy at each nts time steps */
  PetscBool      simplex;                          /* Flag for simplices or tensor cells */
  PetscBool      monitor;                          /* Flag for use of the TS monitor */
  char           meshFilename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscInt       faces;                            /* Number of faces per edge if unit square/cube generated */
  PetscReal      domain_lo[3], domain_hi[3];       /* Lower left and upper right mesh corners */
  PetscReal omega;                                 /* Oscillation value omega */
  DMBoundaryType boundary[3];                      /* The domain boundary type, e.g. periodic */
  PetscInt       particlesPerCell;                 /* The number of partices per cell */
  PetscReal      particleRelDx;                    /* Relative particle position perturbation compared to average cell diameter h */
  PetscReal      meshRelDx;                        /* Relative vertex position perturbation compared to average cell diameter h */
  PetscInt       k;                                /* Mode number for test function */
  PetscReal      momentTol;                        /* Tolerance for checking moment conservation */
  PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
} AppCtx;

/* const char *const ex2FunctionTypes[] = {"linear","x2_x4","sin","ex2FunctionTypes","EX2_FUNCTION_",0}; */
static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *) a_ctx;
  PetscInt d;

  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d]/(ctx->domain_hi[d] - ctx->domain_lo[d]);
  return 0;
}

static PetscErrorCode x2_x4(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *) a_ctx;
  PetscInt d;

  u[0] = 1;
  for (d = 0; d < dim; ++d) u[0] *= PetscSqr(x[d])*PetscSqr(ctx->domain_hi[d]) - PetscPowRealInt(x[d], 4);
  return 0;
}

static PetscErrorCode sinx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx *) a_ctx;

  u[0] = sin(2*PETSC_PI*ctx->k*x[0]/(ctx->domain_hi[0] - ctx->domain_lo[0]));
  return 0;
}



static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       ii, bd;
  char           fstring[PETSC_MAX_PATH_LEN] = "linear";
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim              = 2;
  options->simplex          = PETSC_TRUE;
  options->monitor          = PETSC_FALSE;
  options->faces            = 1;
  options->domain_lo[0]     = 0.0;
  options->domain_lo[1]     = 0.0;
  options->domain_lo[2]     = 0.0;
  options->domain_hi[0]     = 1.0;
  options->domain_hi[1]     = 1.0;
  options->domain_hi[2]     = 1.0;
  options->boundary[0]      = DM_BOUNDARY_NONE; /* PERIODIC (plotting does not work in parallel, moments not conserved) */
  options->boundary[1]      = DM_BOUNDARY_NONE;
  options->boundary[2]      = DM_BOUNDARY_NONE;
  options->particlesPerCell = 1;
  options->k                = 1;
  options->particleRelDx    = 1.e-20;
  options->meshRelDx        = 1.e-20;
  options->momentTol        = 100.*PETSC_MACHINE_EPSILON;
  options->omega            = 64.;
  options->nts              = 100;
  
  ierr = PetscOptionsBegin(comm, "", "L2 Projection Options", "DMPLEX");CHKERRQ(ierr);
  
  ierr = PetscStrcpy(options->meshFilename, "");CHKERRQ(ierr);

  ierr = PetscOptionsInt("-next_output","time steps for next output point","<100>",options->nts,&options->nts,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsBool("-monitor", "To use the TS monitor or not", "ex4.c", options->monitor, &options->monitor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex2.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex2.c", options->meshFilename, options->meshFilename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-faces", "Number of faces per edge if unit square/cube generated", "ex2.c", options->faces, &options->faces, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "Mode number of test", "ex2.c", options->k, &options->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particlesPerCell", "Number of particles per cell", "ex2.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-omega","parameter","<64>",options->omega,&options->omega,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex2.c", options->particleRelDx, &options->particleRelDx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mesh_perturbation", "Relative perturbation of mesh points (0,1)", "ex2.c", options->meshRelDx, &options->meshRelDx, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_hi", "Domain size", "ex2.c", options->domain_hi, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_lo", "Domain size", "ex2.c", options->domain_lo, &ii, NULL);CHKERRQ(ierr);
  bd = options->boundary[0];
  ierr = PetscOptionsEList("-x_boundary", "The x-boundary", "ex2.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[0]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[0] = (DMBoundaryType) bd;
  bd = options->boundary[1];
  ierr = PetscOptionsEList("-y_boundary", "The y-boundary", "ex2.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[1]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[1] = (DMBoundaryType) bd;
  bd = options->boundary[2];
  ierr = PetscOptionsEList("-z_boundary", "The z-boundary", "ex2.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[2]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[2] = (DMBoundaryType) bd;
  ierr = PetscOptionsString("-function", "Name of test function", "ex2.c", fstring, fstring, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscStrcmp(fstring, "linear", &flag);CHKERRQ(ierr);
  if (flag) {
    options->func = linear;
  } else {
    ierr = PetscStrcmp(fstring, "sin", &flag);CHKERRQ(ierr);
    if (flag) {
      options->func = sinx;
    } else {
      ierr = PetscStrcmp(fstring, "x2_x4", &flag);CHKERRQ(ierr);
      options->func = x2_x4;
      if (!flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown function %s",fstring);
    }
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode PerturbVertices(DM dm, AppCtx *user)
{
  PetscRandom    rnd;
  PetscReal      interval = user->meshRelDx;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscReal      hh[3];
  PetscInt       d, cdim, N, p, bs;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (d = 0; d < user->dim; ++d) hh[d] = (user->domain_hi[d] - user->domain_lo[d])/user->faces;
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -interval, interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  if (bs != cdim) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Coordinate vector has wrong block size %D != %D", bs, cdim);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (p = 0; p < N; p += cdim) {
    PetscScalar *coord = &coords[p], value;

    for (d = 0; d < cdim; ++d) {
      ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr);
      coord[d] = PetscMax(user->domain_lo[d], PetscMin(user->domain_hi[d], coord[d] + value*hh[d]));
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->meshFilename, "", &flg);CHKERRQ(ierr);
  if (flg) {
    PetscInt faces[3];

    faces[0] = user->faces; faces[1] = user->faces; faces[2] = user->faces;
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, faces, user->domain_lo, user->domain_hi, user->boundary, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->meshFilename, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  }
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PerturbVertices(*dm, user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscRandom    rnd, rndp;
  PetscReal      interval = user->particleRelDx;
  PetscScalar    value, *vals;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ, *initialConditions;
  PetscInt      *cellid;
  PetscInt       Ncell, Np = user->particlesPerCell, p, c, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -1.0, 1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rndp);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rndp, -interval, interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rndp);CHKERRQ(ierr);

  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &Ncell);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, Ncell * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions);CHKERRQ(ierr);

  ierr = PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  for (c = 0; c < Ncell; ++c) {
    if (Np == 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
      cellid[c] = c;
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr); /* affine */
      for (d = 0; d < dim; ++d) xi0[d] = -1.0;
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        cellid[n] = c;
        for (d = 0; d < dim; ++d) {ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr); refcoords[d] = PetscRealPart(value); sum += refcoords[d];}
        if (user->simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
      }
    }
  }
  ierr = PetscFree5(centroid, xi0, v0, J, invJ);CHKERRQ(ierr);
  for (c = 0; c < Ncell; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;
      
      for (d = 0; d < dim; ++d) {ierr = PetscRandomGetValue(rndp, &value);CHKERRQ(ierr); coords[n*dim+d] += PetscRealPart(value);}
      user->func(dim, 0.0, &coords[n*dim], 1, &vals[c], user);
    }
  }
  for (p = 0; p < Np*Ncell; ++p) {
    initialConditions[p*2+0] = p+0.2; 
    initialConditions[p*2+1] = 0.0;
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rndp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create particle RHS Functions */
static PetscErrorCode RHSFunction1(TS ts,PetscReal t,Vec V,Vec Xres,void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt          Np, p;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(Xres,&xres);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Xres, &Np);CHKERRQ(ierr);

  for (p = 0; p < Np; ++p) {
     xres[p] = v[p];
  }

  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xres,&xres);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode RHSFunction2(TS ts,PetscReal t,Vec X,Vec Vres,void *ctx)
{
  AppCtx*           user = (AppCtx*)ctx;
  const PetscScalar *x;
  PetscInt          Np, p;
  PetscScalar       *vres;
  PetscErrorCode    ierr; 

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Vres,&vres);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Vres, &Np);CHKERRQ(ierr);

  for(p = 0; p < Np; ++p){
    vres[p] = -user->omega*user->omega*x[p];
  }

  ierr = VecRestoreArray(Vres,&vres);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts,PetscReal t,Vec U,Vec R,void *ctx)
{
  DM                dm;
  AppCtx*           user = (AppCtx*)ctx;
  const PetscScalar *u;
  PetscScalar       *r;
  PetscInt          Np, p;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  Np  /= 2;

  for(p = 0; p < Np; ++p){
    r[p*2+0] = u[p*2+1];
    r[p*2+1] = -user->omega*user->omega*u[p*2+0];
  }
  
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscReal         dt;
  PetscInt          p, Np;
  AppCtx*           user = (AppCtx*)ctx;

  PetscFunctionBeginUser;
  if (step%user->nts == 0) {
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
    ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
    Np /= 2;

    for(p = 0; p < Np; ++p){
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Energy: %g\n",(double) (u[p*2+1]*u[p*2+1]+user->omega*user->omega*u[p*2]*u[p*2])/2.);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Modified energy: %g\n", (double) (u[p*2+1]*u[p*2+1]+user->omega*user->omega*u[p*2]*u[p*2]-user->omega*user->omega*dt*u[p*2]*u[p*2+1])/2.);CHKERRQ(ierr);
    }

    ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS              ts;            /* nonlinear solver */
  IS              is1,is2;
  PetscReal       ftime   = 0.1;
  PetscInt        locsize, p, Np;
  Vec             f;              /* swarm vector */
  MPI_Comm        comm;
  DM              dm, sw;
  const PetscReal *fkin;
  AppCtx          user;

  PetscErrorCode  ierr;  

  
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  
  /* Create dm and particles */
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &f);CHKERRQ(ierr);
  
  ierr = VecGetLocalSize(f, &locsize);CHKERRQ(ierr);

  ierr = ISCreateStride(comm, locsize/2, 0, 2,&is1);CHKERRQ(ierr);
  ierr = ISCreateStride(comm, locsize/2, 1, 2,&is2);CHKERRQ(ierr);
  
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBASICSYMPLECTIC);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"position",is1);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"momentum",is2);CHKERRQ(ierr);
  
  ierr = TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user);CHKERRQ(ierr);
  
  ierr = TSSetRHSFunction(ts,NULL,RHSFunctionParticles,&user);CHKERRQ(ierr);

  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.00001);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,100);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  
  if (user.monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }
  
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,f);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  
  ierr = VecGetLocalSize(f, &Np);CHKERRQ(ierr);
  Np /= 2;

  ierr = VecGetArrayRead(f,&fkin);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {

    ierr = PetscPrintf(comm,"The particle solution for (xp, vp) at time %.6lf is [%g %g]\n",(double)ftime,fkin[p*2],fkin[p*2+1]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"The exact solution for (xp, vp) at time %.6lf is [%g %g]\n",(double)ftime,(double) (p+0.2)*PetscCosReal(user.omega*ftime),(double) -(p+0.2)*user.omega*PetscSinReal(user.omega*ftime));CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Exact Particle Energy: %g\n", (double) .5*64*64*(p+0.2)*(p+0.2));
  
  }
  ierr = VecRestoreArrayRead(f,&fkin);CHKERRQ(ierr);
  
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &f);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   build:
     requires: triangle !single !complex
   test:
     args: -dim 2 -faces 1 -particlesPerCell 1 -dm_view -sw_view -ts_basicsymplectic_type 1 -monitor
   test:
     suffix: 2
     args: -dim 2 -faces 1 -particlesPerCell 1 -dm_view -sw_view -ts_basicsymplectic_type 2 -monitor

TEST*/
