static char help[] = "Two stream instability from Birdsal and Langdon with DMSwarm and TS basic symplectic integrators\n";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscksp.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/tsimpl.h>
#include <petscts.h>
#include <petscmath.h>

typedef struct {
  PetscInt       dim;                              /* The topological mesh dimension */
  PetscBool      simplex;                          /* Flag for simplices or tensor cells */
  PetscBool      bdm;                              /* Flag for mixed form poisson */
  PetscBool      monitor;                          /* Flag for use of the TS monitor */
  PetscBool      uniform;                          /* Flag to uniformly space particles in x */
  char           meshFilename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscReal      sigma;                            /* Linear charge per box length */
  PetscReal      timeScale;                        /* Nondimensionalizing time scaling */
  PetscInt       particlesPerCell;                 /* The number of partices per cell */
  PetscReal      particleRelDx;                    /* Relative particle position perturbation compared to average cell diameter h */
  PetscInt       k;                                /* Mode number for test function */
  PetscReal      momentTol;                        /* Tolerance for checking moment conservation */
  SNES           snes;                             /* SNES object */
  PetscInt       steps;                            /* TS iterations */
  PetscReal      stepSize;                         /* Time stepper step size */
  PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim              = 2;
  options->simplex          = PETSC_TRUE;
  options->monitor          = PETSC_TRUE;
  options->particlesPerCell = 1;
  options->k                = 1;
  options->particleRelDx    = 1.e-20;
  options->momentTol        = 100.*PETSC_MACHINE_EPSILON;
  options->sigma            = 1.;
  options->timeScale        = 1.0e-6;
  options->uniform          = PETSC_FALSE;
  options->steps            = 1;
  options->stepSize         = 0.01;
  options->bdm              = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Two Stream options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscStrcpy(options->meshFilename, "");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-steps", "TS steps to take", "ex2.c", options->steps, &options->steps, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitor", "To use the TS monitor or not", "ex2.c", options->monitor, &options->monitor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex2.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-uniform", "Uniform particle spacing", "ex2.c", options->uniform, &options->uniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-bdm", "Use H1 instead of C0", "ex2.c", options->bdm, &options->bdm, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex2.c", options->meshFilename, options->meshFilename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "Mode number of test", "ex5.c", options->k, &options->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particlesPerCell", "Number of particles per cell", "ex2.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-sigma","parameter","<1>",options->sigma,&options->sigma,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-stepSize","parameter","<1e-2>",options->stepSize,&options->stepSize,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-timeScale","parameter","<1>",options->timeScale,&options->timeScale,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex2.c", options->particleRelDx, &options->particleRelDx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static void laplacian_f1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {f1[d] = u_x[d];}
}

static void laplacian(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {g3[d*dim+d] = 1.0;}
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscFE        fe;
  PetscDS        ds;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 0, NULL, laplacian_f1);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, laplacian);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Initialize particle coordinates uniformly and with opposing velocities
*/
static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscRandom    rnd, rndp;
  PetscReal      interval = user->particleRelDx;
  PetscScalar    value, *vals;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ, *initialConditions, normalized_vel;
  PetscInt      *cellid, cStart;
  PetscInt       Ncell, Np = user->particlesPerCell, p, c, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, 0.0, 1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rndp);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rndp, -interval, interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rndp);CHKERRQ(ierr);
  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", dim, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &Ncell);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, Ncell * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  for (c = cStart; c < Ncell; c++) {
    if (Np == 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
      cellid[c] = c;
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      for (d = 0; d < dim; ++d) xi0[d] = -1.0;
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      refcoords[3], spacing;

        cellid[n] = c;
        if (user->uniform) {
          spacing = 2./Np;
          ierr = PetscRandomGetValue(rnd, &value);
          for (d=0; d<dim; ++d) refcoords[d] = d == 0 ? -1. + spacing/2. + p*spacing + value/100. : 0.;
        }
        else{
          for (d = 0; d < dim; ++d) {ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr); refcoords[d] = d == 0 ? PetscRealPart(value) : 0. ;}
        }
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
        /* constant particle weights */
        for (d = 0; d < dim; ++d) vals[n] = user->sigma/Np;
      }
    }
  }
  ierr = PetscFree5(centroid, xi0, v0, J, invJ);CHKERRQ(ierr);
  normalized_vel = 1.;
  for (c = 0; c < Ncell; ++c) {
    for (p = 0; p < Np; ++p) {
      if (p%2 == 0) {
        for (d = 0; d < dim; ++d) initialConditions[(c*Np + p)*dim + d] = d == 0 ? normalized_vel : 0.;
      }
      else {
        for (d = 0; d < dim; ++d) initialConditions[(c*Np + p)*dim + d] = d == 0 ? -(normalized_vel) : 0.;
      }
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rndp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  ierr = DMLocalizeCoordinates(*sw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Solve for particle position updates */
static PetscErrorCode RHSFunction1(TS ts,PetscReal t,Vec V,Vec Posres,void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *posres;
  PetscInt          Np, p, dim, d;
  DM                dm;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(Posres, &Np);CHKERRQ(ierr);
  ierr = VecGetArray(Posres,&posres);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  Np  /= dim;
  for (p = 0; p < Np; ++p) {
     for (d = 0; d < dim; ++d) {
       posres[p*dim+d] = v[p*dim+d];
     }
  }
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(Posres,&posres);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

/*
  Solve for the gradient of the electric field and apply force to particles.
 */
static PetscErrorCode RHSFunction2(TS ts,PetscReal t,Vec X,Vec Vres,void *ctx)
{
 AppCtx             *user = (AppCtx *) ctx;
  DM                 dm, plex;
  PetscDS            prob;
  PetscFE            fe;
  Mat                M_p;
  Vec                phi, locPhi, rho, f;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscReal         *coords, phi_0;
  PetscInt           dim, d, cStart, cEnd, cell, cdim;
  PetscErrorCode     ierr;
  PetscReal          m_e = 9.11e-31, q_e = 1.60e-19, epsi_0 = 8.85e-12;

  PetscFunctionBeginUser;
  PetscObjectSetName((PetscObject) X, "rhsf2 position");
  VecViewFromOptions(X, NULL, "-rhsf2_x_view");
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Vres,&vres);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = SNESGetDM(user->snes, &plex);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(plex, &cdim);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(plex, &phi);CHKERRQ(ierr);
  ierr = DMGetLocalVector(plex, &locPhi);CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(dm, plex, &M_p);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M_p, NULL, "-mp_view");
  ierr = DMGetGlobalVector(plex, &rho);CHKERRQ(ierr);
  ierr = DMSwarmCreateGlobalVectorFromField(dm, "w_q", &f);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) f, "weights vector");
  ierr = VecViewFromOptions(f, NULL, "-weights_view");
  ierr = MatMultTranspose(M_p, f, rho);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(dm, "w_q", &f);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) rho, "rho");CHKERRQ(ierr);
  ierr = VecViewFromOptions(rho, NULL, "-poisson_rho_view");CHKERRQ(ierr);
  /* Take nullspace out of rhs */
  {
    PetscScalar sum;
    PetscInt    n;
    phi_0 = (user->sigma*user->sigma*user->sigma)*(user->timeScale*user->timeScale)/(m_e*q_e*epsi_0);

    ierr = VecGetSize(rho, &n);CHKERRQ(ierr);
    ierr = VecSum(rho, &sum);CHKERRQ(ierr);
    ierr = VecShift(rho, -sum/n);CHKERRQ(ierr);

    ierr = VecSum(rho, &sum);CHKERRQ(ierr);
    if (PetscAbsScalar(sum) > 1.0e-10) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Charge should have no DC component %g", sum);
    ierr = VecScale(rho, phi_0);CHKERRQ(ierr);
  }
  ierr = VecSet(phi, 0.0);CHKERRQ(ierr);
  ierr = SNESSolve(user->snes, rho, phi);CHKERRQ(ierr);
  ierr = VecViewFromOptions(phi, NULL, "-phi_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(plex, &rho);CHKERRQ(ierr);
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(plex, phi, INSERT_VALUES, locPhi);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(plex, phi, INSERT_VALUES, locPhi);CHKERRQ(ierr);
  ierr = DMSwarmSortGetAccess(dm);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscTabulation tab;
    PetscReal    v[3], J[9], invJ[9], detJ;
    PetscScalar *ph       = PETSC_NULL;
    PetscReal   *pcoord   = PETSC_NULL;
    PetscReal   *refcoord = PETSC_NULL;
    PetscInt    *points   = PETSC_NULL, Ncp, cp;
    PetscScalar  gradPhi[3];

    ierr = DMPlexComputeCellGeometryFEM(plex, cell, NULL, v, J, invJ, &detJ);CHKERRQ(ierr);
    ierr = DMSwarmSortGetPointsPerCell(dm, cell, &Ncp, &points);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, Ncp*cdim, MPIU_REAL, &pcoord);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, Ncp*cdim, MPIU_REAL, &refcoord);CHKERRQ(ierr);
    for (cp = 0; cp < Ncp; ++cp) {
      for (d = 0; d < cdim; ++d) {
        pcoord[cp*cdim+d] = coords[points[cp]*cdim+d];
      }
    }
    ierr = DMPlexCoordinatesToReference(plex, cell, Ncp, pcoord, refcoord);CHKERRQ(ierr);
    PetscFECreateTabulation(fe, 1, Ncp, refcoord, 1, &tab);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(plex, NULL, locPhi, cell, NULL, &ph);CHKERRQ(ierr);
    for (cp = 0; cp < Ncp; ++cp) {
      const PetscInt p = points[cp];
      gradPhi[0] = 0.0;
      gradPhi[1] = 0.0;
      gradPhi[2] = 0.0;
      const PetscReal *basisDer = tab->T[1];

      ierr = PetscFEFreeInterpolateGradient_Static(fe, basisDer, ph, cdim, invJ, NULL, cp, gradPhi);CHKERRQ(ierr);
      for (d = 0; d < cdim; ++d) {
        vres[p*cdim+d] = d == 0 ? gradPhi[d] : 0.;
      }
    }
    ierr = DMPlexVecRestoreClosure(plex, NULL, locPhi, cell, NULL, &ph);CHKERRQ(ierr);
    ierr = PetscTabulationDestroy(&tab);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, Ncp*cdim, MPIU_REAL, &pcoord);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, Ncp*cdim, MPIU_REAL, &refcoord);CHKERRQ(ierr);
    ierr = PetscFree(points);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmSortRestoreAccess(dm);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(plex, &locPhi);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(plex, &phi);CHKERRQ(ierr);
  ierr = VecRestoreArray(Vres,&vres);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecViewFromOptions(Vres, NULL, "-vel_res_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt          i, par;
  PetscInt          locSize, p, d, dim, Np, step, *idx1, *idx2;
  TS                ts;
  DM                dm, sw;
  AppCtx            user;
  MPI_Comm          comm;
  PetscErrorCode    ierr;
  Vec               coorVec, kinVec, probVec, solution, position, momentum;
  const PetscScalar *coorArr, *kinArr;
  PetscReal         ftime   = 10., *probArr, *probVecArr;
  IS                is1,is2;
  PetscReal         *coor, *kin, *pos, *mom;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  /* Create dm and particles */
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateFEM(dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = SNESCreate(comm, &user.snes);CHKERRQ(ierr);
  ierr = SNESSetDM(user.snes, dm);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(user.snes);CHKERRQ(ierr);
  {
    Mat          J;
    MatNullSpace nullSpace;

    ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullSpace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J, nullSpace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
    ierr = SNESSetJacobian(user.snes, J, J, NULL, NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
  }
  /* Place TSSolve in a loop to handle resetting the TS at every manual call of TSStep() */
  ierr = TSCreate(comm, &ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,user.stepSize);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,100000);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  for (step = 0; step < user.steps ; ++step){

    ierr = DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &kinVec);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &coorVec);CHKERRQ(ierr);
    ierr = VecViewFromOptions(kinVec, NULL, "-ic_vec_view");
    ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
    ierr = VecGetLocalSize(kinVec, &locSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(locSize, &idx1);CHKERRQ(ierr);
    ierr = PetscMalloc1(locSize, &idx2);CHKERRQ(ierr);
    ierr = PetscMalloc1(2*locSize, &probArr);CHKERRQ(ierr);
    Np = locSize/dim;
    ierr = VecGetArrayRead(kinVec, &kinArr);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coorVec, &coorArr);CHKERRQ(ierr);
    for (p=0; p<Np; ++p){
        for (d=0; d<dim;++d) {
            probArr[p*2*dim + d] = coorArr[p*dim+d];
            probArr[(p*2+1)*dim + d] = kinArr[p*dim+d];
        }
    }
    ierr = VecRestoreArrayRead(kinVec, &kinArr);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coorVec, &coorArr);CHKERRQ(ierr);
    /* Allocate for IS Strides that will contain x, y and vx, vy */
    for (p = 0; p < Np; ++p) {
      for (d = 0; d < dim; ++d) {
        idx1[p*dim+d] = (p*2+0)*dim + d;
        idx2[p*dim+d] = (p*2+1)*dim + d;
      }
    }

    ierr = ISCreateGeneral(comm, locSize, idx1, PETSC_OWN_POINTER, &is1);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, locSize, idx2, PETSC_OWN_POINTER, &is2);CHKERRQ(ierr);
    /* DM needs to be set before splits so it propogates to sub TSs */
    ierr = TSSetDM(ts, sw);CHKERRQ(ierr);
    ierr = TSSetType(ts,TSBASICSYMPLECTIC);CHKERRQ(ierr);
    ierr = TSRHSSplitSetIS(ts,"position",is1);CHKERRQ(ierr);
    ierr = TSRHSSplitSetIS(ts,"momentum",is2);CHKERRQ(ierr);
    ierr = TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user);CHKERRQ(ierr);
    ierr = TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user);CHKERRQ(ierr);
    ierr = TSSetTime(ts, step*user.stepSize);CHKERRQ(ierr);
    if (step == 0) {
      ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    }
    /* Compose vector from array for TS solve with all kinematic variables */
    ierr = VecCreate(comm,&probVec);CHKERRQ(ierr);
    ierr = VecSetBlockSize(probVec,1);CHKERRQ(ierr);
    ierr = VecSetSizes(probVec,PETSC_DECIDE,2*locSize);CHKERRQ(ierr);
    ierr = VecSetUp(probVec);CHKERRQ(ierr);
    ierr = VecGetArray(probVec,&probVecArr);
    for (i=0; i < 2*locSize; ++i) {
      probVecArr[i] = probArr[i];
    }
    ierr = VecRestoreArray(probVec,&probVecArr);CHKERRQ(ierr);
    ierr = TSSetSolution(ts, probVec);CHKERRQ(ierr);
    ierr = PetscFree(probArr);CHKERRQ(ierr);
    ierr = VecViewFromOptions(kinVec, NULL, "-ic_view");
    ierr = DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &kinVec);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &coorVec);CHKERRQ(ierr);
    ierr = TSMonitor(ts, step, ts->ptime, ts->vec_sol);CHKERRQ(ierr);
    if (!ts->steprollback) {
      ierr = TSPreStep(ts);CHKERRQ(ierr);
    }
    ierr = TSStep(ts);CHKERRQ(ierr);
    if (ts->steprollback) {
      ierr = TSPostEvaluate(ts);CHKERRQ(ierr);
    }
    if (!ts->steprollback) {

      TSPostStep(ts);
      ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coor);CHKERRQ(ierr);
      ierr = DMSwarmGetField(sw, "kinematics", NULL, NULL, (void **) &kin);CHKERRQ(ierr);
      ierr = TSGetSolution(ts, &solution);CHKERRQ(ierr);
      ierr = VecGetSubVector(solution,is1,&position);CHKERRQ(ierr);
      ierr = VecGetSubVector(solution,is2,&momentum);CHKERRQ(ierr);
      ierr = VecGetArray(position, &pos);CHKERRQ(ierr);
      ierr = VecGetArray(momentum, &mom);CHKERRQ(ierr);
      for (par = 0; par < Np; ++par){
        for (d=0; d<dim; ++d){
          if (pos[par*dim+d] < 0.) {
            coor[par*dim+d] = pos[par*dim+d] + 2.*PETSC_PI;
          }
          else if (pos[par*dim+d] > 2.*PETSC_PI) {
            coor[par*dim+d] = pos[par*dim+d] - 2.*PETSC_PI;
          }
          else{
            coor[par*dim+d] = pos[par*dim+d];
          }
          kin[par*dim+d] = mom[par*dim+d];
        }
      }
      ierr = VecRestoreArray(position, &pos);CHKERRQ(ierr);
      ierr = VecRestoreArray(momentum, &mom);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(solution,is1,&position);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(solution,is2,&momentum);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coor);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(sw, "kinematics", NULL, NULL, (void **) &kin);CHKERRQ(ierr);
    }
    ierr = DMSwarmMigrate(sw, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMLocalizeCoordinates(sw);CHKERRQ(ierr);
    ierr = TSReset(ts);CHKERRQ(ierr);
    ierr = VecDestroy(&probVec);CHKERRQ(ierr);
    ierr = ISDestroy(&is1);CHKERRQ(ierr);
    ierr = ISDestroy(&is2);CHKERRQ(ierr);
  }
  ierr = SNESDestroy(&user.snes);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: triangle !single !complex
   test:
     suffix: bsi1q3
     args: -particlesPerCell 200\
      -petscspace_degree 2\
      -petscfe_default_quadrature_order 3\
      -ts_basicsymplectic_type 1\
      -pc_type svd\
      -uniform\
      -sigma 1.0e-8\
      -timeScale 2.0e-14\
      -stepSize 1.0e-2\
      -ts_monitor_sp_swarm\
      -steps 10\
      -dm_view\
      -dm_plex_simplex 0 -dm_plex_dim 2\
      -dm_plex_box_lower 0,-1\
      -dm_plex_box_upper 6.283185307179586,1\
      -dm_plex_box_bd periodic,none\
      -dm_plex_box_faces 4,1
   test:
     suffix: bsi2q3
     args: -particlesPerCell 200\
      -petscspace_degree 2\
      -petscfe_default_quadrature_order 3\
      -ts_basicsymplectic_type 2\
      -pc_type svd\
      -uniform\
      -sigma 1.0e-8\
      -timeScale 2.0e-14\
      -stepSize 1.0e-2\
      -ts_monitor_sp_swarm\
      -steps 10\
      -dm_view\
      -dm_plex_simplex 0 -dm_plex_dim 2\
      -dm_plex_box_lower 0,-1\
      -dm_plex_box_upper 6.283185307179586,1\
      -dm_plex_box_bd periodic,none\
      -dm_plex_box_faces 4,1
TEST*/
