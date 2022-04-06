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

  PetscOptionsBegin(comm, "", "Two Stream options", "DMPLEX");
  PetscCall(PetscStrcpy(options->meshFilename, ""));
  PetscCall(PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, NULL));
  PetscCall(PetscOptionsInt("-steps", "TS steps to take", "ex2.c", options->steps, &options->steps, NULL));
  PetscCall(PetscOptionsBool("-monitor", "To use the TS monitor or not", "ex2.c", options->monitor, &options->monitor, NULL));
  PetscCall(PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex2.c", options->simplex, &options->simplex, NULL));
  PetscCall(PetscOptionsBool("-uniform", "Uniform particle spacing", "ex2.c", options->uniform, &options->uniform, NULL));
  PetscCall(PetscOptionsBool("-bdm", "Use H1 instead of C0", "ex2.c", options->bdm, &options->bdm, NULL));
  PetscCall(PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex2.c", options->meshFilename, options->meshFilename, PETSC_MAX_PATH_LEN, NULL));
  PetscCall(PetscOptionsInt("-k", "Mode number of test", "ex5.c", options->k, &options->k, NULL));
  PetscCall(PetscOptionsInt("-particlesPerCell", "Number of particles per cell", "ex2.c", options->particlesPerCell, &options->particlesPerCell, NULL));
  PetscCall(PetscOptionsReal("-sigma","parameter","<1>",options->sigma,&options->sigma,PETSC_NULL));
  PetscCall(PetscOptionsReal("-stepSize","parameter","<1e-2>",options->stepSize,&options->stepSize,PETSC_NULL));
  PetscCall(PetscOptionsReal("-timeScale","parameter","<1>",options->timeScale,&options->timeScale,PETSC_NULL));
  PetscCall(PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex2.c", options->particleRelDx, &options->particleRelDx, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, "potential"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, NULL, laplacian_f1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, laplacian));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, 0.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rndp));
  PetscCall(PetscRandomSetInterval(rndp, -interval, interval));
  PetscCall(PetscRandomSetFromOptions(rndp));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &Ncell));
  PetscCall(DMSwarmSetLocalSizes(*sw, Ncell * Np, 0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  PetscCall(DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **) &vals));
  PetscCall(DMSwarmGetField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions));
  PetscCall(PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ));
  for (c = cStart; c < Ncell; c++) {
    if (Np == 1) {
      PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
      cellid[c] = c;
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      for (d = 0; d < dim; ++d) xi0[d] = -1.0;
      PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ)); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      refcoords[3], spacing;

        cellid[n] = c;
        if (user->uniform) {
          spacing = 2./Np;
          PetscCall(PetscRandomGetValue(rnd, &value));
          for (d=0; d<dim; ++d) refcoords[d] = d == 0 ? -1. + spacing/2. + p*spacing + value/100. : 0.;
        }
        else{
          for (d = 0; d < dim; ++d) {PetscCall(PetscRandomGetValue(rnd, &value)); refcoords[d] = d == 0 ? PetscRealPart(value) : 0. ;}
        }
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
        /* constant particle weights */
        for (d = 0; d < dim; ++d) vals[n] = user->sigma/Np;
      }
    }
  }
  PetscCall(PetscFree5(centroid, xi0, v0, J, invJ));
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
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  PetscCall(DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **) &vals));
  PetscCall(DMSwarmRestoreField(*sw, "kinematics", NULL, NULL, (void **) &initialConditions));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(PetscRandomDestroy(&rndp));
  PetscCall(PetscObjectSetName((PetscObject) *sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscCall(DMLocalizeCoordinates(*sw));
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
  PetscCall(VecGetLocalSize(Posres, &Np));
  PetscCall(VecGetArray(Posres,&posres));
  PetscCall(VecGetArrayRead(V,&v));
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  Np  /= dim;
  for (p = 0; p < Np; ++p) {
     for (d = 0; d < dim; ++d) {
       posres[p*dim+d] = v[p*dim+d];
     }
  }
  PetscCall(VecRestoreArrayRead(V,&v));
  PetscCall(VecRestoreArray(Posres,&posres));
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
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(Vres,&vres));
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(SNESGetDM(user->snes, &plex));
  PetscCall(DMGetCoordinateDim(plex, &cdim));
  PetscCall(DMGetDS(plex, &prob));
  PetscCall(PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe));
  PetscCall(DMGetGlobalVector(plex, &phi));
  PetscCall(DMGetLocalVector(plex, &locPhi));
  PetscCall(DMCreateMassMatrix(dm, plex, &M_p));
  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(DMGetGlobalVector(plex, &rho));
  PetscCall(DMSwarmCreateGlobalVectorFromField(dm, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject) f, "weights vector"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));
  PetscCall(MatMultTranspose(M_p, f, rho));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(dm, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject) rho, "rho"));
  PetscCall(VecViewFromOptions(rho, NULL, "-poisson_rho_view"));
  /* Take nullspace out of rhs */
  {
    PetscScalar sum;
    PetscInt    n;
    phi_0 = (user->sigma*user->sigma*user->sigma)*(user->timeScale*user->timeScale)/(m_e*q_e*epsi_0);

    PetscCall(VecGetSize(rho, &n));
    PetscCall(VecSum(rho, &sum));
    PetscCall(VecShift(rho, -sum/n));

    PetscCall(VecSum(rho, &sum));
    PetscCheck(PetscAbsScalar(sum) <= 1.0e-10,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Charge should have no DC component %g", (double)PetscAbsScalar(sum));
    PetscCall(VecScale(rho, phi_0));
  }
  PetscCall(VecSet(phi, 0.0));
  PetscCall(SNESSolve(user->snes, rho, phi));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));
  PetscCall(DMRestoreGlobalVector(plex, &rho));
  PetscCall(MatDestroy(&M_p));
  PetscCall(DMGlobalToLocalBegin(plex, phi, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(plex, phi, INSERT_VALUES, locPhi));
  PetscCall(DMSwarmSortGetAccess(dm));
  PetscCall(DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd));
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscTabulation tab;
    PetscReal    v[3], J[9], invJ[9], detJ;
    PetscScalar *ph       = PETSC_NULL;
    PetscReal   *pcoord   = PETSC_NULL;
    PetscReal   *refcoord = PETSC_NULL;
    PetscInt    *points   = PETSC_NULL, Ncp, cp;
    PetscScalar  gradPhi[3];

    PetscCall(DMPlexComputeCellGeometryFEM(plex, cell, NULL, v, J, invJ, &detJ));
    PetscCall(DMSwarmSortGetPointsPerCell(dm, cell, &Ncp, &points));
    PetscCall(DMGetWorkArray(dm, Ncp*cdim, MPIU_REAL, &pcoord));
    PetscCall(DMGetWorkArray(dm, Ncp*cdim, MPIU_REAL, &refcoord));
    for (cp = 0; cp < Ncp; ++cp) {
      for (d = 0; d < cdim; ++d) {
        pcoord[cp*cdim+d] = coords[points[cp]*cdim+d];
      }
    }
    PetscCall(DMPlexCoordinatesToReference(plex, cell, Ncp, pcoord, refcoord));
    PetscCall(PetscFECreateTabulation(fe, 1, Ncp, refcoord, 1, &tab));
    PetscCall(DMPlexVecGetClosure(plex, NULL, locPhi, cell, NULL, &ph));
    for (cp = 0; cp < Ncp; ++cp) {
      const PetscInt p = points[cp];
      gradPhi[0] = 0.0;
      gradPhi[1] = 0.0;
      gradPhi[2] = 0.0;
      const PetscReal *basisDer = tab->T[1];

      PetscCall(PetscFEFreeInterpolateGradient_Static(fe, basisDer, ph, cdim, invJ, NULL, cp, gradPhi));
      for (d = 0; d < cdim; ++d) {
        vres[p*cdim+d] = d == 0 ? gradPhi[d] : 0.;
      }
    }
    PetscCall(DMPlexVecRestoreClosure(plex, NULL, locPhi, cell, NULL, &ph));
    PetscCall(PetscTabulationDestroy(&tab));
    PetscCall(DMRestoreWorkArray(dm, Ncp*cdim, MPIU_REAL, &pcoord));
    PetscCall(DMRestoreWorkArray(dm, Ncp*cdim, MPIU_REAL, &refcoord));
    PetscCall(PetscFree(points));
  }
  PetscCall(DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(DMSwarmSortRestoreAccess(dm));
  PetscCall(DMRestoreLocalVector(plex, &locPhi));
  PetscCall(DMRestoreGlobalVector(plex, &phi));
  PetscCall(VecRestoreArray(Vres,&vres));
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecViewFromOptions(Vres, NULL, "-vel_res_view"));
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

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  /* Create dm and particles */
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateFEM(dm, &user));
  PetscCall(CreateParticles(dm, &sw, &user));
  PetscCall(SNESCreate(comm, &user.snes));
  PetscCall(SNESSetDM(user.snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm,&user,&user,&user));
  PetscCall(SNESSetFromOptions(user.snes));
  {
    Mat          J;
    MatNullSpace nullSpace;

    PetscCall(DMCreateMatrix(dm, &J));
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullSpace));
    PetscCall(MatSetNullSpace(J, nullSpace));
    PetscCall(MatNullSpaceDestroy(&nullSpace));
    PetscCall(SNESSetJacobian(user.snes, J, J, NULL, NULL));
    PetscCall(MatDestroy(&J));
  }
  /* Place TSSolve in a loop to handle resetting the TS at every manual call of TSStep() */
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetMaxTime(ts,ftime));
  PetscCall(TSSetTimeStep(ts,user.stepSize));
  PetscCall(TSSetMaxSteps(ts,100000));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  for (step = 0; step < user.steps ; ++step){

    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "kinematics", &kinVec));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &coorVec));
    PetscCall(VecViewFromOptions(kinVec, NULL, "-ic_vec_view"));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(VecGetLocalSize(kinVec, &locSize));
    PetscCall(PetscMalloc1(locSize, &idx1));
    PetscCall(PetscMalloc1(locSize, &idx2));
    PetscCall(PetscMalloc1(2*locSize, &probArr));
    Np = locSize/dim;
    PetscCall(VecGetArrayRead(kinVec, &kinArr));
    PetscCall(VecGetArrayRead(coorVec, &coorArr));
    for (p=0; p<Np; ++p){
        for (d=0; d<dim;++d) {
            probArr[p*2*dim + d] = coorArr[p*dim+d];
            probArr[(p*2+1)*dim + d] = kinArr[p*dim+d];
        }
    }
    PetscCall(VecRestoreArrayRead(kinVec, &kinArr));
    PetscCall(VecRestoreArrayRead(coorVec, &coorArr));
    /* Allocate for IS Strides that will contain x, y and vx, vy */
    for (p = 0; p < Np; ++p) {
      for (d = 0; d < dim; ++d) {
        idx1[p*dim+d] = (p*2+0)*dim + d;
        idx2[p*dim+d] = (p*2+1)*dim + d;
      }
    }

    PetscCall(ISCreateGeneral(comm, locSize, idx1, PETSC_OWN_POINTER, &is1));
    PetscCall(ISCreateGeneral(comm, locSize, idx2, PETSC_OWN_POINTER, &is2));
    /* DM needs to be set before splits so it propogates to sub TSs */
    PetscCall(TSSetDM(ts, sw));
    PetscCall(TSSetType(ts,TSBASICSYMPLECTIC));
    PetscCall(TSRHSSplitSetIS(ts,"position",is1));
    PetscCall(TSRHSSplitSetIS(ts,"momentum",is2));
    PetscCall(TSRHSSplitSetRHSFunction(ts,"position",NULL,RHSFunction1,&user));
    PetscCall(TSRHSSplitSetRHSFunction(ts,"momentum",NULL,RHSFunction2,&user));
    PetscCall(TSSetTime(ts, step*user.stepSize));
    if (step == 0) {
      PetscCall(TSSetFromOptions(ts));
    }
    /* Compose vector from array for TS solve with all kinematic variables */
    PetscCall(VecCreate(comm,&probVec));
    PetscCall(VecSetBlockSize(probVec,1));
    PetscCall(VecSetSizes(probVec,PETSC_DECIDE,2*locSize));
    PetscCall(VecSetUp(probVec));
    PetscCall(VecGetArray(probVec,&probVecArr));
    for (i=0; i < 2*locSize; ++i) probVecArr[i] = probArr[i];
    PetscCall(VecRestoreArray(probVec,&probVecArr));
    PetscCall(TSSetSolution(ts, probVec));
    PetscCall(PetscFree(probArr));
    PetscCall(VecViewFromOptions(kinVec, NULL, "-ic_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "kinematics", &kinVec));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &coorVec));
    PetscCall(TSMonitor(ts, step, ts->ptime, ts->vec_sol));
    if (!ts->steprollback) {
      PetscCall(TSPreStep(ts));
    }
    PetscCall(TSStep(ts));
    if (ts->steprollback) {
      PetscCall(TSPostEvaluate(ts));
    }
    if (!ts->steprollback) {

      TSPostStep(ts);
      PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coor));
      PetscCall(DMSwarmGetField(sw, "kinematics", NULL, NULL, (void **) &kin));
      PetscCall(TSGetSolution(ts, &solution));
      PetscCall(VecGetSubVector(solution,is1,&position));
      PetscCall(VecGetSubVector(solution,is2,&momentum));
      PetscCall(VecGetArray(position, &pos));
      PetscCall(VecGetArray(momentum, &mom));
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
      PetscCall(VecRestoreArray(position, &pos));
      PetscCall(VecRestoreArray(momentum, &mom));
      PetscCall(VecRestoreSubVector(solution,is1,&position));
      PetscCall(VecRestoreSubVector(solution,is2,&momentum));
      PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coor));
      PetscCall(DMSwarmRestoreField(sw, "kinematics", NULL, NULL, (void **) &kin));
    }
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMLocalizeCoordinates(sw));
    PetscCall(TSReset(ts));
    PetscCall(VecDestroy(&probVec));
    PetscCall(ISDestroy(&is1));
    PetscCall(ISDestroy(&is2));
  }
  PetscCall(SNESDestroy(&user.snes));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
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
