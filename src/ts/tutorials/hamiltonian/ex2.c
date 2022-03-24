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
  CHKERRQ(PetscStrcpy(options->meshFilename, ""));
  CHKERRQ(PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, NULL));
  CHKERRQ(PetscOptionsInt("-steps", "TS steps to take", "ex2.c", options->steps, &options->steps, NULL));
  CHKERRQ(PetscOptionsBool("-monitor", "To use the TS monitor or not", "ex2.c", options->monitor, &options->monitor, NULL));
  CHKERRQ(PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex2.c", options->simplex, &options->simplex, NULL));
  CHKERRQ(PetscOptionsBool("-uniform", "Uniform particle spacing", "ex2.c", options->uniform, &options->uniform, NULL));
  CHKERRQ(PetscOptionsBool("-bdm", "Use H1 instead of C0", "ex2.c", options->bdm, &options->bdm, NULL));
  CHKERRQ(PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex2.c", options->meshFilename, options->meshFilename, PETSC_MAX_PATH_LEN, NULL));
  CHKERRQ(PetscOptionsInt("-k", "Mode number of test", "ex5.c", options->k, &options->k, NULL));
  CHKERRQ(PetscOptionsInt("-particlesPerCell", "Number of particles per cell", "ex2.c", options->particlesPerCell, &options->particlesPerCell, NULL));
  CHKERRQ(PetscOptionsReal("-sigma","parameter","<1>",options->sigma,&options->sigma,PETSC_NULL));
  CHKERRQ(PetscOptionsReal("-stepSize","parameter","<1e-2>",options->stepSize,&options->stepSize,PETSC_NULL));
  CHKERRQ(PetscOptionsReal("-timeScale","parameter","<1>",options->timeScale,&options->timeScale,PETSC_NULL));
  CHKERRQ(PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex2.c", options->particleRelDx, &options->particleRelDx, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
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

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  CHKERRQ(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, "potential"));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSSetResidual(ds, 0, NULL, laplacian_f1));
  CHKERRQ(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, laplacian));
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

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  CHKERRQ(DMSetType(*sw, DMSWARM));
  CHKERRQ(DMSetDimension(*sw, dim));
  CHKERRQ(PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd));
  CHKERRQ(PetscRandomSetInterval(rnd, 0.0, 1.0));
  CHKERRQ(PetscRandomSetFromOptions(rnd));
  CHKERRQ(PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rndp));
  CHKERRQ(PetscRandomSetInterval(rndp, -interval, interval));
  CHKERRQ(PetscRandomSetFromOptions(rndp));
  CHKERRQ(DMSwarmSetType(*sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(*sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", dim, PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(*sw));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &Ncell));
  CHKERRQ(DMSwarmSetLocalSizes(*sw, Ncell * Np, 0));
  CHKERRQ(DMSetFromOptions(*sw));
  CHKERRQ(DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  CHKERRQ(DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **) &vals));
  CHKERRQ(DMSwarmGetField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions));
  CHKERRQ(PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ));
  for (c = cStart; c < Ncell; c++) {
    if (Np == 1) {
      CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
      cellid[c] = c;
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      for (d = 0; d < dim; ++d) xi0[d] = -1.0;
      CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ)); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      refcoords[3], spacing;

        cellid[n] = c;
        if (user->uniform) {
          spacing = 2./Np;
          CHKERRQ(PetscRandomGetValue(rnd, &value));
          for (d=0; d<dim; ++d) refcoords[d] = d == 0 ? -1. + spacing/2. + p*spacing + value/100. : 0.;
        }
        else{
          for (d = 0; d < dim; ++d) {CHKERRQ(PetscRandomGetValue(rnd, &value)); refcoords[d] = d == 0 ? PetscRealPart(value) : 0. ;}
        }
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
        /* constant particle weights */
        for (d = 0; d < dim; ++d) vals[n] = user->sigma/Np;
      }
    }
  }
  CHKERRQ(PetscFree5(centroid, xi0, v0, J, invJ));
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
  CHKERRQ(DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  CHKERRQ(DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **) &vals));
  CHKERRQ(DMSwarmRestoreField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions));
  CHKERRQ(PetscRandomDestroy(&rnd));
  CHKERRQ(PetscRandomDestroy(&rndp));
  CHKERRQ(PetscObjectSetName((PetscObject) *sw, "Particles"));
  CHKERRQ(DMViewFromOptions(*sw, NULL, "-sw_view"));
  CHKERRQ(DMLocalizeCoordinates(*sw));
  PetscFunctionReturn(0);
}

/* Solve for particle position updates */
static PetscErrorCode RHSFunction1(TS ts,PetscReal t,Vec V,Vec Posres,void *ctx)
{
  const PetscScalar *v;
  PetscScalar       *posres;
  PetscInt          Np, p, dim, d;
  DM                dm;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetLocalSize(Posres, &Np));
  CHKERRQ(VecGetArray(Posres,&posres));
  CHKERRQ(VecGetArrayRead(V,&v));
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  Np  /= dim;
  for (p = 0; p < Np; ++p) {
     for (d = 0; d < dim; ++d) {
       posres[p*dim+d] = v[p*dim+d];
     }
  }
  CHKERRQ(VecRestoreArrayRead(V,&v));
  CHKERRQ(VecRestoreArray(Posres,&posres));
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
  PetscReal          m_e = 9.11e-31, q_e = 1.60e-19, epsi_0 = 8.85e-12;

  PetscFunctionBeginUser;
  PetscObjectSetName((PetscObject) X, "rhsf2 position");
  VecViewFromOptions(X, NULL, "-rhsf2_x_view");
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(Vres,&vres));
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(SNESGetDM(user->snes, &plex));
  CHKERRQ(DMGetCoordinateDim(plex, &cdim));
  CHKERRQ(DMGetDS(plex, &prob));
  CHKERRQ(PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe));
  CHKERRQ(DMGetGlobalVector(plex, &phi));
  CHKERRQ(DMGetLocalVector(plex, &locPhi));
  CHKERRQ(DMCreateMassMatrix(dm, plex, &M_p));
  CHKERRQ(MatViewFromOptions(M_p, NULL, "-mp_view"));
  CHKERRQ(DMGetGlobalVector(plex, &rho));
  CHKERRQ(DMSwarmCreateGlobalVectorFromField(dm, "w_q", &f));
  CHKERRQ(PetscObjectSetName((PetscObject) f, "weights vector"));
  CHKERRQ(VecViewFromOptions(f, NULL, "-weights_view"));
  CHKERRQ(MatMultTranspose(M_p, f, rho));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(dm, "w_q", &f));
  CHKERRQ(PetscObjectSetName((PetscObject) rho, "rho"));
  CHKERRQ(VecViewFromOptions(rho, NULL, "-poisson_rho_view"));
  /* Take nullspace out of rhs */
  {
    PetscScalar sum;
    PetscInt    n;
    phi_0 = (user->sigma*user->sigma*user->sigma)*(user->timeScale*user->timeScale)/(m_e*q_e*epsi_0);

    CHKERRQ(VecGetSize(rho, &n));
    CHKERRQ(VecSum(rho, &sum));
    CHKERRQ(VecShift(rho, -sum/n));

    CHKERRQ(VecSum(rho, &sum));
    PetscCheck(PetscAbsScalar(sum) <= 1.0e-10,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Charge should have no DC component %g", sum);
    CHKERRQ(VecScale(rho, phi_0));
  }
  CHKERRQ(VecSet(phi, 0.0));
  CHKERRQ(SNESSolve(user->snes, rho, phi));
  CHKERRQ(VecViewFromOptions(phi, NULL, "-phi_view"));
  CHKERRQ(DMRestoreGlobalVector(plex, &rho));
  CHKERRQ(MatDestroy(&M_p));
  CHKERRQ(DMGlobalToLocalBegin(plex, phi, INSERT_VALUES, locPhi));
  CHKERRQ(DMGlobalToLocalEnd(plex, phi, INSERT_VALUES, locPhi));
  CHKERRQ(DMSwarmSortGetAccess(dm));
  CHKERRQ(DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd));
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscTabulation tab;
    PetscReal    v[3], J[9], invJ[9], detJ;
    PetscScalar *ph       = PETSC_NULL;
    PetscReal   *pcoord   = PETSC_NULL;
    PetscReal   *refcoord = PETSC_NULL;
    PetscInt    *points   = PETSC_NULL, Ncp, cp;
    PetscScalar  gradPhi[3];

    CHKERRQ(DMPlexComputeCellGeometryFEM(plex, cell, NULL, v, J, invJ, &detJ));
    CHKERRQ(DMSwarmSortGetPointsPerCell(dm, cell, &Ncp, &points));
    CHKERRQ(DMGetWorkArray(dm, Ncp*cdim, MPIU_REAL, &pcoord));
    CHKERRQ(DMGetWorkArray(dm, Ncp*cdim, MPIU_REAL, &refcoord));
    for (cp = 0; cp < Ncp; ++cp) {
      for (d = 0; d < cdim; ++d) {
        pcoord[cp*cdim+d] = coords[points[cp]*cdim+d];
      }
    }
    CHKERRQ(DMPlexCoordinatesToReference(plex, cell, Ncp, pcoord, refcoord));
    CHKERRQ(PetscFECreateTabulation(fe, 1, Ncp, refcoord, 1, &tab));
    CHKERRQ(DMPlexVecGetClosure(plex, NULL, locPhi, cell, NULL, &ph));
    for (cp = 0; cp < Ncp; ++cp) {
      const PetscInt p = points[cp];
      gradPhi[0] = 0.0;
      gradPhi[1] = 0.0;
      gradPhi[2] = 0.0;
      const PetscReal *basisDer = tab->T[1];

      CHKERRQ(PetscFEFreeInterpolateGradient_Static(fe, basisDer, ph, cdim, invJ, NULL, cp, gradPhi));
      for (d = 0; d < cdim; ++d) {
        vres[p*cdim+d] = d == 0 ? gradPhi[d] : 0.;
      }
    }
    CHKERRQ(DMPlexVecRestoreClosure(plex, NULL, locPhi, cell, NULL, &ph));
    CHKERRQ(PetscTabulationDestroy(&tab));
    CHKERRQ(DMRestoreWorkArray(dm, Ncp*cdim, MPIU_REAL, &pcoord));
    CHKERRQ(DMRestoreWorkArray(dm, Ncp*cdim, MPIU_REAL, &refcoord));
    CHKERRQ(PetscFree(points));
  }
  CHKERRQ(DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(DMSwarmSortRestoreAccess(dm));
  CHKERRQ(DMRestoreLocalVector(plex, &locPhi));
  CHKERRQ(DMRestoreGlobalVector(plex, &phi));
  CHKERRQ(VecRestoreArray(Vres,&vres));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecViewFromOptions(Vres, NULL, "-vel_res_view"));
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
  Vec               coorVec, kinVec, probVec, solution, position, momentum;
  const PetscScalar *coorArr, *kinArr;
  PetscReal         ftime   = 10., *probArr, *probVecArr;
  IS                is1,is2;
  PetscReal         *coor, *kin, *pos, *mom;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(ProcessOptions(comm, &user));
  /* Create dm and particles */
  CHKERRQ(CreateMesh(comm, &dm, &user));
  CHKERRQ(CreateFEM(dm, &user));
  CHKERRQ(CreateParticles(dm, &sw, &user));
  CHKERRQ(SNESCreate(comm, &user.snes));
  CHKERRQ(SNESSetDM(user.snes, dm));
  CHKERRQ(DMPlexSetSNESLocalFEM(dm,&user,&user,&user));
  CHKERRQ(SNESSetFromOptions(user.snes));
  {
    Mat          J;
    MatNullSpace nullSpace;

    CHKERRQ(DMCreateMatrix(dm, &J));
    CHKERRQ(MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullSpace));
    CHKERRQ(MatSetNullSpace(J, nullSpace));
    CHKERRQ(MatNullSpaceDestroy(&nullSpace));
    CHKERRQ(SNESSetJacobian(user.snes, J, J, NULL, NULL));
    CHKERRQ(MatDestroy(&J));
  }
  /* Place TSSolve in a loop to handle resetting the TS at every manual call of TSStep() */
  CHKERRQ(TSCreate(comm, &ts));
  CHKERRQ(TSSetMaxTime(ts,ftime));
  CHKERRQ(TSSetTimeStep(ts,user.stepSize));
  CHKERRQ(TSSetMaxSteps(ts,100000));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  for (step = 0; step < user.steps ; ++step){

    CHKERRQ(DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &kinVec));
    CHKERRQ(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &coorVec));
    CHKERRQ(VecViewFromOptions(kinVec, NULL, "-ic_vec_view"));
    CHKERRQ(DMGetDimension(sw, &dim));
    CHKERRQ(VecGetLocalSize(kinVec, &locSize));
    CHKERRQ(PetscMalloc1(locSize, &idx1));
    CHKERRQ(PetscMalloc1(locSize, &idx2));
    CHKERRQ(PetscMalloc1(2*locSize, &probArr));
    Np = locSize/dim;
    CHKERRQ(VecGetArrayRead(kinVec, &kinArr));
    CHKERRQ(VecGetArrayRead(coorVec, &coorArr));
    for (p=0; p<Np; ++p){
        for (d=0; d<dim;++d) {
            probArr[p*2*dim + d] = coorArr[p*dim+d];
            probArr[(p*2+1)*dim + d] = kinArr[p*dim+d];
        }
    }
    CHKERRQ(VecRestoreArrayRead(kinVec, &kinArr));
    CHKERRQ(VecRestoreArrayRead(coorVec, &coorArr));
    /* Allocate for IS Strides that will contain x, y and vx, vy */
    for (p = 0; p < Np; ++p) {
      for (d = 0; d < dim; ++d) {
        idx1[p*dim+d] = (p*2+0)*dim + d;
        idx2[p*dim+d] = (p*2+1)*dim + d;
      }
    }

    CHKERRQ(ISCreateGeneral(comm, locSize, idx1, PETSC_OWN_POINTER, &is1));
    CHKERRQ(ISCreateGeneral(comm, locSize, idx2, PETSC_OWN_POINTER, &is2));
    /* DM needs to be set before splits so it propogates to sub TSs */
    CHKERRQ(TSSetDM(ts, sw));
    CHKERRQ(TSSetType(ts,TSBASICSYMPLECTIC));
    CHKERRQ(TSRHSSplitSetIS(ts,"position",is1));
    CHKERRQ(TSRHSSplitSetIS(ts,"momentum",is2));
    CHKERRQ(TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user));
    CHKERRQ(TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user));
    CHKERRQ(TSSetTime(ts, step*user.stepSize));
    if (step == 0) {
      CHKERRQ(TSSetFromOptions(ts));
    }
    /* Compose vector from array for TS solve with all kinematic variables */
    CHKERRQ(VecCreate(comm,&probVec));
    CHKERRQ(VecSetBlockSize(probVec,1));
    CHKERRQ(VecSetSizes(probVec,PETSC_DECIDE,2*locSize));
    CHKERRQ(VecSetUp(probVec));
    CHKERRQ(VecGetArray(probVec,&probVecArr));
    for (i=0; i < 2*locSize; ++i) probVecArr[i] = probArr[i];
    CHKERRQ(VecRestoreArray(probVec,&probVecArr));
    CHKERRQ(TSSetSolution(ts, probVec));
    CHKERRQ(PetscFree(probArr));
    CHKERRQ(VecViewFromOptions(kinVec, NULL, "-ic_view"));
    CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &kinVec));
    CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &coorVec));
    CHKERRQ(TSMonitor(ts, step, ts->ptime, ts->vec_sol));
    if (!ts->steprollback) {
      CHKERRQ(TSPreStep(ts));
    }
    CHKERRQ(TSStep(ts));
    if (ts->steprollback) {
      CHKERRQ(TSPostEvaluate(ts));
    }
    if (!ts->steprollback) {

      TSPostStep(ts);
      CHKERRQ(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coor));
      CHKERRQ(DMSwarmGetField(sw, "kinematics", NULL, NULL, (void **) &kin));
      CHKERRQ(TSGetSolution(ts, &solution));
      CHKERRQ(VecGetSubVector(solution,is1,&position));
      CHKERRQ(VecGetSubVector(solution,is2,&momentum));
      CHKERRQ(VecGetArray(position, &pos));
      CHKERRQ(VecGetArray(momentum, &mom));
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
      CHKERRQ(VecRestoreArray(position, &pos));
      CHKERRQ(VecRestoreArray(momentum, &mom));
      CHKERRQ(VecRestoreSubVector(solution,is1,&position));
      CHKERRQ(VecRestoreSubVector(solution,is2,&momentum));
      CHKERRQ(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coor));
      CHKERRQ(DMSwarmRestoreField(sw, "kinematics", NULL, NULL, (void **) &kin));
    }
    CHKERRQ(DMSwarmMigrate(sw, PETSC_TRUE));
    CHKERRQ(DMLocalizeCoordinates(sw));
    CHKERRQ(TSReset(ts));
    CHKERRQ(VecDestroy(&probVec));
    CHKERRQ(ISDestroy(&is1));
    CHKERRQ(ISDestroy(&is2));
  }
  CHKERRQ(SNESDestroy(&user.snes));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&sw));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
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
