static char help[] = "Nonlinear elasticity problem in 3d with simplicial finite elements.\n\
We solve a nonlinear elasticity problem, modelled as an incompressible Neo-Hookean solid, \n\
 with pressure loading in a rectangular domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*
Nonlinear elasticity problem, which we discretize using the finite
element method on an unstructured mesh. This uses both Dirichlet boundary conditions (fixed faces)
and nonlinear Neumann boundary conditions (pressure loading).
The Lagrangian density (modulo boundary conditions) for this problem is given by
\begin{equation}
  \frac{\mu}{2} (\mathrm{Tr}{C}-3) + J p + \frac{\kappa}{2} (J-1).
\end{equation}

Discretization:

We use PetscFE to generate a tabulation of the finite element basis functions
at quadrature points. We can currently generate an arbitrary order Lagrange
element.

Field Data:

  DMPLEX data is organized by point, and the closure operation just stacks up the
data from each sieve point in the closure. Thus, for a P_2-P_1 Stokes element, we
have

  cl{e} = {f e_0 e_1 e_2 v_0 v_1 v_2}
  x     = [u_{e_0} v_{e_0} u_{e_1} v_{e_1} u_{e_2} v_{e_2} u_{v_0} v_{v_0} p_{v_0} u_{v_1} v_{v_1} p_{v_1} u_{v_2} v_{v_2} p_{v_2}]

The problem here is that we would like to loop over each field separately for
integration. Therefore, the closure visitor in DMPlexVecGetClosure() reorders
the data so that each field is contiguous

  x'    = [u_{e_0} v_{e_0} u_{e_1} v_{e_1} u_{e_2} v_{e_2} u_{v_0} v_{v_0} u_{v_1} v_{v_1} u_{v_2} v_{v_2} p_{v_0} p_{v_1} p_{v_2}]

Likewise, DMPlexVecSetClosure() takes data partitioned by field, and correctly
puts it into the Sieve ordering.

*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef enum {RUN_FULL, RUN_TEST} RunType;

typedef struct {
  RunType   runType; /* Whether to run tests, or solve the full problem */
  PetscReal mu;      /* The shear modulus */
  PetscReal p_wall;  /* The wall pressure */
} AppCtx;

#if 0
static inline void Det2D(PetscReal *detJ, const PetscReal J[])
{
  *detJ = J[0]*J[3] - J[1]*J[2];
}
#endif

static inline void Det3D(PetscReal *detJ, const PetscScalar J[])
{
  *detJ = PetscRealPart(J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
                        J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
                        J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
}

#if 0
static inline void Cof2D(PetscReal C[], const PetscReal A[])
{
  C[0] =  A[3];
  C[1] = -A[2];
  C[2] = -A[1];
  C[3] =  A[0];
}
#endif

static inline void Cof3D(PetscReal C[], const PetscScalar A[])
{
  C[0*3+0] = PetscRealPart(A[1*3+1]*A[2*3+2] - A[1*3+2]*A[2*3+1]);
  C[0*3+1] = PetscRealPart(A[1*3+2]*A[2*3+0] - A[1*3+0]*A[2*3+2]);
  C[0*3+2] = PetscRealPart(A[1*3+0]*A[2*3+1] - A[1*3+1]*A[2*3+0]);
  C[1*3+0] = PetscRealPart(A[0*3+2]*A[2*3+1] - A[0*3+1]*A[2*3+2]);
  C[1*3+1] = PetscRealPart(A[0*3+0]*A[2*3+2] - A[0*3+2]*A[2*3+0]);
  C[1*3+2] = PetscRealPart(A[0*3+1]*A[2*3+0] - A[0*3+0]*A[2*3+1]);
  C[2*3+0] = PetscRealPart(A[0*3+1]*A[1*3+2] - A[0*3+2]*A[1*3+1]);
  C[2*3+1] = PetscRealPart(A[0*3+2]*A[1*3+0] - A[0*3+0]*A[1*3+2]);
  C[2*3+2] = PetscRealPart(A[0*3+0]*A[1*3+1] - A[0*3+1]*A[1*3+0]);
}

PetscErrorCode zero_scalar(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

PetscErrorCode zero_vector(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscInt Ncomp = dim;

  PetscInt       comp;
  for (comp = 0; comp < Ncomp; ++comp) u[comp] = 0.0;
  return 0;
}

PetscErrorCode coordinates(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscInt Ncomp = dim;

  PetscInt       comp;
  for (comp = 0; comp < Ncomp; ++comp) u[comp] = x[comp];
  return 0;
}

PetscErrorCode elasticityMaterial(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  u[0] = user->mu;
  return 0;
}

PetscErrorCode wallPressure(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  u[0] = user->p_wall;
  return 0;
}

void f1_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt  Ncomp = dim;
  const PetscReal mu = PetscRealPart(a[0]), kappa = 3.0;
  PetscReal       cofu_x[9/*Ncomp*dim*/], detu_x, p = PetscRealPart(u[Ncomp]);
  PetscInt        comp, d;

  Cof3D(cofu_x, u_x);
  Det3D(&detu_x, u_x);
  p += kappa * (detu_x - 1.0);

  /* f1 is the first Piola-Kirchhoff tensor */
  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = mu * u_x[comp*dim+d] + p * cofu_x[comp*dim+d];
    }
  }
}

void g3_uu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscInt  Ncomp = dim;
  const PetscReal mu = PetscRealPart(a[0]), kappa = 3.0;
  PetscReal       cofu_x[9/*Ncomp*dim*/], detu_x, pp, pm, p = PetscRealPart(u[Ncomp]);
  PetscInt        compI, compJ, d1, d2;

  Cof3D(cofu_x, u_x);
  Det3D(&detu_x, u_x);
  p += kappa * (detu_x - 1.0);
  pp = p/detu_x + kappa;
  pm = p/detu_x;

  /* g3 is the first elasticity tensor, i.e. A_i^I_j^J = S^{IJ} g_{ij} + C^{KILJ} F^k_K F^l_L g_{kj} g_{li} */
  for (compI = 0; compI < Ncomp; ++compI) {
    for (compJ = 0; compJ < Ncomp; ++compJ) {
      const PetscReal G = (compI == compJ) ? 1.0 : 0.0;

      for (d1 = 0; d1 < dim; ++d1) {
        for (d2 = 0; d2 < dim; ++d2) {
          const PetscReal g = (d1 == d2) ? 1.0 : 0.0;

          g3[((compI*Ncomp+compJ)*dim+d1)*dim+d2] = g * G * mu + pp * cofu_x[compI*dim+d1] * cofu_x[compJ*dim+d2] - pm * cofu_x[compI*dim+d2] * cofu_x[compJ*dim+d1];
        }
      }
    }
  }
}

void f0_bd_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt    Ncomp = dim;
  const PetscScalar p = a[aOff[1]];
  PetscReal         cofu_x[9/*Ncomp*dim*/];
  PetscInt          comp, d;

  Cof3D(cofu_x, u_x);
  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0, f0[comp] = 0.0; d < dim; ++d) f0[comp] += cofu_x[comp*dim+d] * n[d];
    f0[comp] *= p;
  }
}

void g1_bd_uu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
    PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  const PetscInt Ncomp = dim;
  PetscScalar    p = a[aOff[1]];
  PetscReal      cofu_x[9/*Ncomp*dim*/], m[3/*Ncomp*/], detu_x;
  PetscInt       comp, compI, compJ, d;

  Cof3D(cofu_x, u_x);
  Det3D(&detu_x, u_x);
  p /= detu_x;

  for (comp = 0; comp < Ncomp; ++comp) for (d = 0, m[comp] = 0.0; d < dim; ++d) m[comp] += cofu_x[comp*dim+d] * n[d];
  for (compI = 0; compI < Ncomp; ++compI) {
    for (compJ = 0; compJ < Ncomp; ++compJ) {
      for (d = 0; d < dim; ++d) {
        g1[(compI*Ncomp+compJ)*dim+d] = p * (m[compI] * cofu_x[compJ*dim+d] - cofu_x[compI*dim+d] * m[compJ]);
      }
    }
  }
}

void f0_p_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal detu_x;
  Det3D(&detu_x, u_x);
  f0[0] = detu_x - 1.0;
}

void g1_pu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal cofu_x[9/*Ncomp*dim*/];
  PetscInt  compI, d;

  Cof3D(cofu_x, u_x);
  for (compI = 0; compI < dim; ++compI)
    for (d = 0; d < dim; ++d) g1[compI*dim+d] = cofu_x[compI*dim+d];
}

void g2_up_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscReal cofu_x[9/*Ncomp*dim*/];
  PetscInt  compI, d;

  Cof3D(cofu_x, u_x);
  for (compI = 0; compI < dim; ++compI)
    for (d = 0; d < dim; ++d) g2[compI*dim+d] = cofu_x[compI*dim+d];
}

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *runTypes[2] = {"full", "test"};
  PetscInt       run;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->runType      = RUN_FULL;
  options->mu           = 1.0;
  options->p_wall       = 0.4;

  ierr = PetscOptionsBegin(comm, "", "Nonlinear elasticity problem options", "DMPLEX");CHKERRQ(ierr);
  run  = options->runType;
  CHKERRQ(PetscOptionsEList("-run_type", "The run type", "ex77.c", runTypes, 2, runTypes[options->runType], &run, NULL));
  options->runType = (RunType) run;
  CHKERRQ(PetscOptionsReal("-shear_modulus", "The shear modulus", "ex77.c", options->mu, &options->mu, NULL));
  CHKERRQ(PetscOptionsReal("-wall_pressure", "The wall pressure", "ex77.c", options->p_wall, &options->p_wall, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  /* TODO The P1 coordinate space gives wrong results when compared to the affine version. Track this down */
  if (0) {
    CHKERRQ(DMPlexCreateBoxMesh(comm, 3, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm));
  } else {
    CHKERRQ(DMCreate(comm, dm));
    CHKERRQ(DMSetType(*dm, DMPLEX));
  }
  CHKERRQ(DMSetFromOptions(*dm));
  /* Label the faces (bit of a hack here, until it is properly implemented for simplices) */
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-orig_dm_view"));
  {
    DM              cdm;
    DMLabel         label;
    IS              is;
    PetscInt        d, dim, b, f, Nf;
    const PetscInt *faces;
    PetscInt        csize;
    PetscScalar    *coords = NULL;
    PetscSection    cs;
    Vec             coordinates ;

    CHKERRQ(DMGetDimension(*dm, &dim));
    CHKERRQ(DMCreateLabel(*dm, "boundary"));
    CHKERRQ(DMGetLabel(*dm, "boundary", &label));
    CHKERRQ(DMPlexMarkBoundaryFaces(*dm, 1, label));
    CHKERRQ(DMGetStratumIS(*dm, "boundary", 1,  &is));
    if (is) {
      PetscReal faceCoord;
      PetscInt  v;

      CHKERRQ(ISGetLocalSize(is, &Nf));
      CHKERRQ(ISGetIndices(is, &faces));

      CHKERRQ(DMGetCoordinatesLocal(*dm, &coordinates));
      CHKERRQ(DMGetCoordinateDM(*dm, &cdm));
      CHKERRQ(DMGetLocalSection(cdm, &cs));

      /* Check for each boundary face if any component of its centroid is either 0.0 or 1.0 */
      for (f = 0; f < Nf; ++f) {
        CHKERRQ(DMPlexVecGetClosure(cdm, cs, coordinates, faces[f], &csize, &coords));
        /* Calculate mean coordinate vector */
        for (d = 0; d < dim; ++d) {
          const PetscInt Nv = csize/dim;
          faceCoord = 0.0;
          for (v = 0; v < Nv; ++v) faceCoord += PetscRealPart(coords[v*dim+d]);
          faceCoord /= Nv;
          for (b = 0; b < 2; ++b) {
            if (PetscAbs(faceCoord - b*1.0) < PETSC_SMALL) {
              CHKERRQ(DMSetLabelValue(*dm, "Faces", faces[f], d*2+b+1));
            }
          }
        }
        CHKERRQ(DMPlexVecRestoreClosure(cdm, cs, coordinates, faces[f], &csize, &coords));
      }
      CHKERRQ(ISRestoreIndices(is, &faces));
    }
    CHKERRQ(ISDestroy(&is));
  }
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

PetscErrorCode SetupProblem(DM dm, PetscInt dim, AppCtx *user)
{
  PetscDS        ds;
  PetscWeakForm  wf;
  DMLabel        label;
  PetscInt       bd;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSSetResidual(ds, 0, NULL, f1_u_3d));
  CHKERRQ(PetscDSSetResidual(ds, 1, f0_p_3d, NULL));
  CHKERRQ(PetscDSSetJacobian(ds, 0, 0, NULL, NULL,  NULL,  g3_uu_3d));
  CHKERRQ(PetscDSSetJacobian(ds, 0, 1, NULL, NULL,  g2_up_3d, NULL));
  CHKERRQ(PetscDSSetJacobian(ds, 1, 0, NULL, g1_pu_3d, NULL,  NULL));

  CHKERRQ(DMGetLabel(dm, "Faces", &label));
  CHKERRQ(DMAddBoundary(dm, DM_BC_NATURAL, "pressure", label, 0, NULL, 0, 0, NULL, NULL, NULL, user, &bd));
  CHKERRQ(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  CHKERRQ(PetscWeakFormSetIndexBdResidual(wf, label, 1, 0, 0, 0, f0_bd_u_3d, 0, NULL));
  CHKERRQ(PetscWeakFormSetIndexBdJacobian(wf, label, 1, 0, 0, 0, 0, NULL, 0, g1_bd_uu_3d, 0, NULL, 0, NULL));

  CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "fixed", label, 0, NULL, 0, 0, NULL, (void (*)(void)) coordinates, NULL, user, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*matFuncs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {elasticityMaterial, wallPressure};
  Vec            A;
  void          *ctxs[2];

  PetscFunctionBegin;
  ctxs[0] = user; ctxs[1] = user;
  CHKERRQ(DMCreateLocalVector(dmAux, &A));
  CHKERRQ(DMProjectFunctionLocal(dmAux, 0.0, matFuncs, ctxs, INSERT_ALL_VALUES, A));
  CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, A));
  CHKERRQ(VecDestroy(&A));
  PetscFunctionReturn(0);
}

PetscErrorCode SetupNearNullSpace(DM dm, AppCtx *user)
{
  /* Set up the near null space (a.k.a. rigid body modes) that will be used by the multigrid preconditioner */
  DM             subdm;
  MatNullSpace   nearNullSpace;
  PetscInt       fields = 0;
  PetscObject    deformation;

  PetscFunctionBeginUser;
  CHKERRQ(DMCreateSubDM(dm, 1, &fields, NULL, &subdm));
  CHKERRQ(DMPlexCreateRigidBody(subdm, 0, &nearNullSpace));
  CHKERRQ(DMGetField(dm, 0, NULL, &deformation));
  CHKERRQ(PetscObjectCompose(deformation, "nearnullspace", (PetscObject) nearNullSpace));
  CHKERRQ(DMDestroy(&subdm));
  CHKERRQ(MatNullSpaceDestroy(&nearNullSpace));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscInt NfAux, PetscFE feAux[], AppCtx *user)
{
  DM             dmAux, coordDM;
  PetscInt       f;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  CHKERRQ(DMGetCoordinateDM(dm, &coordDM));
  if (!feAux) PetscFunctionReturn(0);
  CHKERRQ(DMClone(dm, &dmAux));
  CHKERRQ(DMSetCoordinateDM(dmAux, coordDM));
  for (f = 0; f < NfAux; ++f) CHKERRQ(DMSetField(dmAux, f, NULL, (PetscObject) feAux[f]));
  CHKERRQ(DMCreateDS(dmAux));
  CHKERRQ(SetupMaterial(dm, dmAux, user));
  CHKERRQ(DMDestroy(&dmAux));
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  PetscFE         fe[2], feAux[2];
  PetscBool       simplex;
  PetscInt        dim;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  /* Create finite element */
  CHKERRQ(PetscFECreateDefault(comm, dim, dim, simplex, "def_", PETSC_DEFAULT, &fe[0]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[0], "deformation"));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[1]));

  CHKERRQ(PetscObjectSetName((PetscObject) fe[1], "pressure"));

  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, "elastMat_", PETSC_DEFAULT, &feAux[0]));
  CHKERRQ(PetscObjectSetName((PetscObject) feAux[0], "elasticityMaterial"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], feAux[0]));
  /* It is not yet possible to define a field on a submesh (e.g. a boundary), so we will use a normal finite element */
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, "wall_pres_", PETSC_DEFAULT, &feAux[1]));
  CHKERRQ(PetscObjectSetName((PetscObject) feAux[1], "wall_pressure"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], feAux[1]));

  /* Set discretization and boundary conditions for each mesh */
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe[0]));
  CHKERRQ(DMSetField(dm, 1, NULL, (PetscObject) fe[1]));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(SetupProblem(dm, dim, user));
  while (cdm) {
    CHKERRQ(SetupAuxDM(cdm, 2, feAux, user));
    CHKERRQ(DMCopyDisc(dm, cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEDestroy(&fe[0]));
  CHKERRQ(PetscFEDestroy(&fe[1]));
  CHKERRQ(PetscFEDestroy(&feAux[0]));
  CHKERRQ(PetscFEDestroy(&feAux[1]));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  DM             dm;                   /* problem definition */
  Vec            u,r;                  /* solution, residual vectors */
  Mat            A,J;                  /* Jacobian matrix */
  AppCtx         user;                 /* user-defined work context */
  PetscInt       its;                  /* iterations for convergence */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD, &snes));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(SNESSetDM(snes, dm));
  CHKERRQ(DMSetApplicationContext(dm, &user));

  CHKERRQ(SetupDiscretization(dm, &user));
  CHKERRQ(DMPlexCreateClosureIndex(dm, NULL));
  CHKERRQ(SetupNearNullSpace(dm, &user));

  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(VecDuplicate(u, &r));

  CHKERRQ(DMSetMatType(dm,MATAIJ));
  CHKERRQ(DMCreateMatrix(dm, &J));
  A = J;

  CHKERRQ(DMPlexSetSNESLocalFEM(dm,&user,&user,&user));
  CHKERRQ(SNESSetJacobian(snes, A, J, NULL, NULL));

  CHKERRQ(SNESSetFromOptions(snes));

  {
    PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx);
    initialGuess[0] = coordinates; initialGuess[1] = zero_scalar;
    CHKERRQ(DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u));
  }

  if (user.runType == RUN_FULL) {
    CHKERRQ(SNESSolve(snes, NULL, u));
    CHKERRQ(SNESGetIterationNumber(snes, &its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %D\n", its));
  } else {
    PetscReal res = 0.0;

    /* Check initial guess */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n"));
    CHKERRQ(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
    /* Check residual */
    CHKERRQ(SNESComputeFunction(snes, u, r));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n"));
    CHKERRQ(VecChop(r, 1.0e-10));
    CHKERRQ(VecView(r, PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecNorm(r, NORM_2, &res));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res));
    /* Check Jacobian */
    {
      Vec b;

      CHKERRQ(SNESComputeJacobian(snes, u, A, A));
      CHKERRQ(VecDuplicate(u, &b));
      CHKERRQ(VecSet(r, 0.0));
      CHKERRQ(SNESComputeFunction(snes, r, b));
      CHKERRQ(MatMult(A, u, r));
      CHKERRQ(VecAXPY(r, 1.0, b));
      CHKERRQ(VecDestroy(&b));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n"));
      CHKERRQ(VecChop(r, 1.0e-10));
      CHKERRQ(VecView(r, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(VecNorm(r, NORM_2, &res));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", (double)res));
    }
  }
  CHKERRQ(VecViewFromOptions(u, NULL, "-sol_vec_view"));

  if (A != J) CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: !complex

  testset:
    requires: ctetgen
    args: -run_type full -dm_plex_dim 3 \
          -def_petscspace_degree 2 -pres_petscspace_degree 1 -elastMat_petscspace_degree 0 -wall_pres_petscspace_degree 0 \
          -snes_rtol 1e-05 -snes_monitor_short -snes_converged_reason \
          -ksp_type fgmres -ksp_rtol 1e-10 -ksp_monitor_short -ksp_converged_reason \
          -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper \
            -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu \
            -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

    test:
      suffix: 0
      requires: !single
      args: -dm_refine 2 \
            -bc_fixed 1 -bc_pressure 2 -wall_pressure 0.4

    test:
      suffix: 1
      requires: superlu_dist
      nsize: 2
      args: -dm_refine 0 -petscpartitioner_type simple \
            -bc_fixed 1 -bc_pressure 2 -wall_pressure 0.4
      timeoutfactor: 2

    test:
      suffix: 4
      requires: superlu_dist
      nsize: 2
      args: -dm_refine 0 -petscpartitioner_type simple \
            -bc_fixed 1 -bc_pressure 2 -wall_pressure 0.4
      output_file: output/ex77_1.out

    test:
      suffix: 2
      requires: !single
      args: -dm_refine 2 \
            -bc_fixed 3,4,5,6 -bc_pressure 2 -wall_pressure 1.0

    test:
      suffix: 2_par
      requires: superlu_dist !single
      nsize: 4
      args: -dm_refine 2 -petscpartitioner_type simple \
            -bc_fixed 3,4,5,6 -bc_pressure 2 -wall_pressure 1.0
      output_file: output/ex77_2.out

TEST*/
