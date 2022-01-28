static char help[] = "Example application of the Bhatnagar-Gross-Krook (BGK) collision operator.\n\
This example is a 0D-1V setting for the kinetic equation\n\
https://en.wikipedia.org/wiki/Bhatnagar%E2%80%93Gross%E2%80%93Krook_operator\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>
#include <petscdraw.h>
#include <petscviewer.h>

typedef struct {
  PetscInt    particlesPerCell; /* The number of partices per cell */
  PetscReal   momentTol;        /* Tolerance for checking moment conservation */
  PetscBool   monitorhg;        /* Flag for using the TS histogram monitor */
  PetscBool   monitorsp;        /* Flag for using the TS scatter monitor */
  PetscBool   monitorks;        /* Monitor to perform KS test to the maxwellian */
  PetscBool   error;            /* Flag for printing the error */
  PetscInt    ostep;            /* print the energy at each ostep time steps */
  PetscDraw   draw;             /* The draw object for histogram monitoring */
  PetscDrawHG drawhg;           /* The histogram draw context for monitoring */
  PetscDrawSP drawsp;           /* The scatter plot draw context for the monitor */
  PetscDrawSP drawks;           /* Scatterplot draw context for KS test */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->monitorhg        = PETSC_FALSE;
  options->monitorsp        = PETSC_FALSE;
  options->monitorks        = PETSC_FALSE;
  options->particlesPerCell = 1;
  options->momentTol        = 100.0*PETSC_MACHINE_EPSILON;
  options->ostep            = 100;

  ierr = PetscOptionsBegin(comm, "", "Collision Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitorhg", "Flag to use the TS histogram monitor", "ex28.c", options->monitorhg, &options->monitorhg, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitorsp", "Flag to use the TS scatter plot monitor", "ex28.c", options->monitorsp, &options->monitorsp, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitorks", "Flag to plot KS test results", "ex28.c", options->monitorks, &options->monitorks, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex28.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-output_step", "Number of time steps between output", "ex28.c", options->ostep, &options->ostep, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Create the mesh for velocity space */
static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Since we are putting the same number of particles in each cell, this amounts to a uniform distribution of v */
static PetscErrorCode SetInitialCoordinates(DM sw)
{
  AppCtx        *user;
  PetscRandom    rnd;
  DM             dm;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ, *vals;
  PetscInt       dim, d, cStart, cEnd, c, Np, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) sw), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -1.0, 1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);

  ierr = DMGetApplicationContext(sw, &user);CHKERRQ(ierr);
  Np   = user->particlesPerCell;
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  ierr = DMSwarmGetCellDM(sw, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    if (Np == 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        coords[c*dim+d] = centroid[d];
        if ((coords[c*dim+d] >= -1) && (coords[c*dim+d] <= 1)) {
          vals[c] = 1.0;
        } else {
          vals[c] = 0.;
        }
      }
    } else {
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        for (d = 0; d < dim; ++d) {
          ierr = PetscRandomGetValueReal(rnd, &refcoords[d]);CHKERRQ(ierr);
          sum += refcoords[d];
        }
        if (simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
        vals[n] = 1.0;
        ierr = DMPlexReferenceToCoordinates(dm, c, 1, refcoords, &coords[n*dim]);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = PetscFree5(centroid, xi0, v0, J, invJ);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The intiial conditions are just the initial particle weights */
static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM             dm;
  AppCtx        *user;
  PetscReal     *vals;
  PetscScalar   *initialConditions;
  PetscInt       dim, d, cStart, cEnd, c, Np, p, n;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(u, &n);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dmSw, &user);CHKERRQ(ierr);
  Np   = user->particlesPerCell;
  ierr = DMSwarmGetCellDM(dmSw, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  PetscAssertFalse(n != (cEnd-cStart)*Np,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "TS solution local size %D != %D nm particles", n, (cEnd-cStart)*Np);
  ierr = DMSwarmGetField(dmSw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = VecGetArray(u, &initialConditions);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;
      for (d = 0; d < dim; d++) {
        initialConditions[n] = vals[n];
      }
    }
  }
  ierr = VecRestoreArray(u, &initialConditions);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmSw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np = user->particlesPerCell, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", dim, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;
      cellid[n] = c;
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  f_t = 1/\tau \left( f_eq - f \right)
  n_t = 1/\tau \left( \int f_eq - \int f \right) = 1/\tau (n - n) = 0
  v_t = 1/\tau \left( \int v f_eq - \int v f \right) = 1/\tau (v - v) = 0
  E_t = 1/\tau \left( \int v^2 f_eq - \int v^2 f \right) = 1/\tau (T - T) = 0

  Let's look at a single cell:

    \int_C f_t             = 1/\tau \left( \int_C f_eq - \int_C f \right)
    \sum_{x_i \in C} w^i_t = 1/\tau \left( F_eq(C) - \sum_{x_i \in C} w_i \right)
*/

/* This computes the 1D Maxwellian distribution for given mass n, velocity v, and temperature T */
static PetscReal ComputePDF(PetscReal m, PetscReal n, PetscReal T, PetscReal v[])
{
  return (n/PetscSqrtReal(2.0*PETSC_PI*T/m)) * PetscExpReal(-0.5*m*PetscSqr(v[0])/T);
}

/*
  erf z = \frac{2}{\sqrt\pi} \int^z_0 dt e^{-t^2} and erf \infty = 1

  We begin with our distribution

    \sqrt{\frac{m}{2 \pi T}} e^{-m v^2/2T}

  Let t = \sqrt{m/2T} v, z = \sqrt{m/2T} w, so that we now have

      \sqrt{\frac{m}{2 \pi T}} \int^w_0 dv e^{-m v^2/2T}
    = \sqrt{\frac{m}{2 \pi T}} \int^{\sqrt{m/2T} w}_0 \sqrt{2T/m} dt e^{-t^2}
    = 1/\sqrt{\pi} \int^{\sqrt{m/2T} w}_0 dt e^{-t^2}
    = 1/2 erf(\sqrt{m/2T} w)
*/
static PetscReal ComputeCDF(PetscReal m, PetscReal n, PetscReal T, PetscReal va, PetscReal vb)
{
  PetscReal alpha = PetscSqrtReal(0.5*m/T);
  PetscReal za    = alpha*va;
  PetscReal zb    = alpha*vb;
  PetscReal sum   = 0.0;

  sum += zb >= 0 ? erf(zb) : -erf(-zb);
  sum -= za >= 0 ? erf(za) : -erf(-za);
  return 0.5 * n * sum;
}

static PetscErrorCode CheckDistribution(DM dm, PetscReal m, PetscReal n, PetscReal T, PetscReal v[])
{
  PetscSection   coordSection;
  Vec            coordsLocal;
  PetscReal     *xq, *wq;
  PetscReal      vmin, vmax, neq, veq, Teq;
  PetscInt       Nq = 100, q, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetBoundingBox(dm, &vmin, &vmax);CHKERRQ(ierr);
  /* Check analytic over entire line */
  neq  = ComputeCDF(m, n, T, vmin, vmax);
  PetscAssertFalse(PetscAbsReal(neq - n) > PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Int f %g != %g mass (%g)", neq, n, neq-n);
  /* Check analytic over cells */
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  neq  = 0.0;
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *vcoords = NULL;

    ierr = DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, NULL, &vcoords);CHKERRQ(ierr);
    neq += ComputeCDF(m, n, T, vcoords[0], vcoords[1]);
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, NULL, &vcoords);CHKERRQ(ierr);
  }
  PetscAssertFalse(PetscAbsReal(neq - n) > PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell Int f %g != %g mass (%g)", neq, n, neq-n);
  /* Check quadrature over entire line */
  ierr = PetscMalloc2(Nq, &xq, Nq, &wq);CHKERRQ(ierr);
  ierr = PetscDTGaussQuadrature(100, vmin, vmax, xq, wq);CHKERRQ(ierr);
  neq  = 0.0;
  for (q = 0; q < Nq; ++q) {
    neq += ComputePDF(m, n, T, &xq[q])*wq[q];
  }
  PetscAssertFalse(PetscAbsReal(neq - n) > PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Int f %g != %g mass (%g)", neq, n, neq-n);
  /* Check omemnts with quadrature */
  veq  = 0.0;
  for (q = 0; q < Nq; ++q) {
    veq += xq[q]*ComputePDF(m, n, T, &xq[q])*wq[q];
  }
  veq /= neq;
  PetscAssertFalse(PetscAbsReal(veq - v[0]) > PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Int v f %g != %g velocity (%g)", veq, v[0], veq-v[0]);
  Teq  = 0.0;
  for (q = 0; q < Nq; ++q) {
    Teq += PetscSqr(xq[q])*ComputePDF(m, n, T, &xq[q])*wq[q];
  }
  Teq = Teq * m/neq - PetscSqr(veq);
  PetscAssertFalse(PetscAbsReal(Teq - T) > PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Int v^2 f %g != %g temperature (%g)", Teq, T, Teq-T);
  ierr = PetscFree2(xq, wq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  const PetscScalar *u;
  PetscSection       coordSection;
  Vec                coordsLocal;
  PetscScalar       *r, *coords;
  PetscReal          n = 0.0, v = 0.0, E = 0.0, T = 0.0, m = 1.0, cn = 0.0, cv = 0.0, cE = 0.0, pE = 0.0, eqE = 0.0;
  PetscInt           dim, d, Np, Ncp, p, cStart, cEnd, c;
  DM                 dmSw, plex;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetArray(R, &r);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dmSw);CHKERRQ(ierr);
  ierr = DMSwarmGetCellDM(dmSw, &plex);CHKERRQ(ierr);
  ierr = DMGetDimension(dmSw, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(plex, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(plex, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
  Np  /= dim;
  Ncp  = Np / (cEnd - cStart);
  /* Calculate moments of particle distribution, note that velocity is in the coordinate */
  ierr = DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    PetscReal m1 = 0.0, m2 = 0.0;

    for (d = 0; d < dim; ++d) {m1 += PetscRealPart(coords[p*dim+d]); m2 += PetscSqr(PetscRealPart(coords[p*dim+d]));}
    n += u[p];
    v += u[p]*m1;
    E += u[p]*m2;
  }
  v /= n;
  T  = E*m/n - v*v;
  ierr = PetscInfo(ts, "Time %.2f: mass %.4f velocity: %+.4f temperature: %.4f\n", t, n, v, T);CHKERRQ(ierr);
  ierr = CheckDistribution(plex, m, n, T, &v);CHKERRQ(ierr);
  /*
     Begin cellwise evaluation of the collision operator. Essentially, penalize the weights of the particles
     in that cell against the maxwellian for the number of particles expected to be in that cell
  */
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *vcoords = NULL;
    PetscReal    relaxation = 1.0, neq;
    PetscInt     sp      = c*Ncp, q;

    /* Calculate equilibrium occupation for this velocity cell */
    ierr = DMPlexVecGetClosure(plex, coordSection, coordsLocal, c, NULL, &vcoords);CHKERRQ(ierr);
    neq  = ComputeCDF(m, n, T, vcoords[0], vcoords[1]);
    ierr = DMPlexVecRestoreClosure(plex, coordSection, coordsLocal, c, NULL, &vcoords);CHKERRQ(ierr);
    for (q = 0; q < Ncp; ++q) r[sp+q] = (1.0/relaxation)*(neq - u[sp+q]);
  }
  /* Check update */
  for (p = 0; p < Np; ++p) {
    PetscReal m1 = 0.0, m2 = 0.0;
    PetscScalar *vcoords = NULL;

    for (d = 0; d < dim; ++d) {m1 += PetscRealPart(coords[p*dim+d]); m2 += PetscSqr(PetscRealPart(coords[p*dim+d]));}
    cn += r[p];
    cv += r[p]*m1;
    cE += r[p]*m2;
    pE  += u[p]*m2;
    ierr = DMPlexVecGetClosure(plex, coordSection, coordsLocal, p, NULL, &vcoords);CHKERRQ(ierr);
    eqE += ComputeCDF(m, n, T, vcoords[0], vcoords[1])*m2;
    ierr = DMPlexVecRestoreClosure(plex, coordSection, coordsLocal, p, NULL, &vcoords);CHKERRQ(ierr);
  }
  ierr = PetscInfo(ts, "Time %.2f: mass update %.8f velocity update: %+.8f energy update: %.8f (%.8f, %.8f)\n", t, cn, cv, cE, pE, eqE);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode HGMonitor(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx            *user  = (AppCtx *) ctx;
  const PetscScalar *u;
  DM                 sw, dm;
  PetscInt           dim, Np, p;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(0);
  if (((user->ostep > 0) && (!(step % user->ostep)))) {
    PetscDrawAxis axis;

    ierr = TSGetDM(ts, &sw);CHKERRQ(ierr);
    ierr = DMSwarmGetCellDM(sw, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscDrawHGReset(user->drawhg);CHKERRQ(ierr);
    ierr = PetscDrawHGGetAxis(user->drawhg,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Particles","V","f(V)");CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLimits(axis, -3, 3, 0, 100);CHKERRQ(ierr);
    ierr = PetscDrawHGSetLimits(user->drawhg,-3.0, 3.0, 0, 10);CHKERRQ(ierr);
    ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
    Np  /= dim;
    ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
    /* get points from solution vector */
    for (p = 0; p < Np; ++p) {ierr = PetscDrawHGAddValue(user->drawhg,u[p]);CHKERRQ(ierr);}
    ierr = PetscDrawHGDraw(user->drawhg);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SPMonitor(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx            *user  = (AppCtx *) ctx;
  const PetscScalar *u;
  PetscReal         *v, *coords;
  PetscInt           Np, p;
  DM                 dmSw;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;

  if (step < 0) PetscFunctionReturn(0);
  if (((user->ostep > 0) && (!(step % user->ostep)))) {
    PetscDrawAxis axis;

    ierr = TSGetDM(ts, &dmSw);CHKERRQ(ierr);
    ierr = PetscDrawSPReset(user->drawsp);CHKERRQ(ierr);
    ierr = PetscDrawSPGetAxis(user->drawsp,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Particles","V","w");CHKERRQ(ierr);
    ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
    ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
    /* get points from solution vector */
    ierr = PetscMalloc1(Np, &v);CHKERRQ(ierr);
    for (p = 0; p < Np; ++p) v[p] = PetscRealPart(u[p]);
    ierr = DMSwarmGetField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
    for (p = 0; p < Np-1; ++p) {ierr = PetscDrawSPAddPoint(user->drawsp, &coords[p], &v[p]);CHKERRQ(ierr);}
    ierr = PetscDrawSPDraw(user->drawsp, PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dmSw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
    ierr = PetscFree(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSConv(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx            *user  = (AppCtx *) ctx;
  const PetscScalar *u;
  PetscScalar       sup;
  PetscReal         *v, *coords, T=0., vel=0., step_cast, w_sum;
  PetscInt           dim, Np, p, cStart, cEnd;
  DM                 sw, plex;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(0);
  if (((user->ostep > 0) && (!(step % user->ostep)))) {
    PetscDrawAxis axis;
    ierr = PetscDrawSPGetAxis(user->drawks,&axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis,"Particles","t","D_n");CHKERRQ(ierr);
    ierr = PetscDrawSPSetLimits(user->drawks,0.,100,0.,3.5);CHKERRQ(ierr);
    ierr = TSGetDM(ts, &sw);CHKERRQ(ierr);
    ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
    ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
    /* get points from solution vector */
    ierr = PetscMalloc1(Np, &v);CHKERRQ(ierr);
    ierr = DMSwarmGetCellDM(sw, &plex);CHKERRQ(ierr);
    ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (p = 0; p < Np; ++p) v[p] = PetscRealPart(u[p]);
    ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
    w_sum = 0.;
    for (p = 0; p < Np; ++p) {
      w_sum += u[p];
      T += u[p]*coords[p]*coords[p];
      vel += u[p]*coords[p];
    }
    vel /= w_sum;
    T = T/w_sum - vel*vel;
    sup = 0.0;
    for (p = 0; p < Np; ++p) {
        PetscReal tmp = 0.;
        tmp = PetscAbs(u[p]-ComputePDF(1.0, w_sum, T, &coords[p*dim]));
        if (tmp > sup) sup = tmp;
    }
    step_cast = (PetscReal)step;
    ierr = PetscDrawSPAddPoint(user->drawks, &step_cast, &sup);CHKERRQ(ierr);
    ierr = PetscDrawSPDraw(user->drawks, PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
    ierr = PetscFree(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM             dm;
  AppCtx        *user;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = SetInitialCoordinates(dm);CHKERRQ(ierr);
  ierr = SetInitialConditions(dm, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  /*
     A single particle is generated for each velocity space cell using the dmswarmpicfield_coor and is used to evaluate collisions in that cell.
     0 weight ghost particles are initialized outside of a small velocity domain to ensure the tails of the amxwellian are resolved.
   */
int main(int argc,char **argv)
{
  TS             ts;     /* nonlinear solver */
  DM             dm, sw; /* Velocity space mesh and Particle Swarm */
  Vec            u, w;   /* swarm vector */
  MPI_Comm       comm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(sw, &user);CHKERRQ(ierr);
  ierr = TSCreate(comm, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, sw);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, 10.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 0.01);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, 100000);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (user.monitorhg) {
    ierr = PetscDrawCreate(comm, NULL, "monitor", 0,0,400,300, &user.draw);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(user.draw);CHKERRQ(ierr);
    ierr = PetscDrawHGCreate(user.draw, 20, &user.drawhg);CHKERRQ(ierr);
    ierr = PetscDrawHGSetColor(user.drawhg,3);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, HGMonitor, &user, NULL);CHKERRQ(ierr);
  }
  else if (user.monitorsp) {
    ierr = PetscDrawCreate(comm, NULL, "monitor", 0,0,400,300, &user.draw);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(user.draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(user.draw, 1, &user.drawsp);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, SPMonitor, &user, NULL);CHKERRQ(ierr);
  }
  else if (user.monitorks) {
    ierr = PetscDrawCreate(comm, NULL, "monitor", 0,0,400,300, &user.draw);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(user.draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(user.draw, 1, &user.drawks);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, KSConv, &user, NULL);CHKERRQ(ierr);
  }
  ierr = TSSetRHSFunction(ts, NULL, RHSFunctionParticles, &user);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetComputeInitialCondition(ts, InitializeSolve);CHKERRQ(ierr);
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &w);CHKERRQ(ierr);
  ierr = VecDuplicate(w, &u);CHKERRQ(ierr);
  ierr = VecCopy(w, u);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &w);CHKERRQ(ierr);
  ierr = TSComputeInitialCondition(ts, u);CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  if (user.monitorhg) {
    ierr = PetscDrawSave(user.draw);CHKERRQ(ierr);
    ierr = PetscDrawHGDestroy(&user.drawhg);CHKERRQ(ierr);
    ierr = PetscDrawDestroy(&user.draw);CHKERRQ(ierr);
  }
  if (user.monitorsp) {
    ierr = PetscDrawSave(user.draw);CHKERRQ(ierr);
    ierr = PetscDrawSPDestroy(&user.drawsp);CHKERRQ(ierr);
    ierr = PetscDrawDestroy(&user.draw);CHKERRQ(ierr);
  }
  if (user.monitorks) {
    ierr = PetscDrawSave(user.draw);CHKERRQ(ierr);
    ierr = PetscDrawSPDestroy(&user.drawks);CHKERRQ(ierr);
    ierr = PetscDrawDestroy(&user.draw);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   build:
     requires: double !complex
   test:
     suffix: 1
     args: -particles_per_cell 1 -output_step 10 -ts_type euler -dm_plex_dim 1 -dm_plex_box_faces 200 -dm_plex_box_lower -10 -dm_plex_box_upper 10 -dm_view -monitorsp
   test:
     suffix: 2
     args: -particles_per_cell 1 -output_step 50 -ts_type euler -dm_plex_dim 1 -dm_plex_box_faces 200 -dm_plex_box_lower -10 -dm_plex_box_upper 10 -dm_view -monitorks
TEST*/
