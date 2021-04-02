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
  PetscInt      debug;             /* The debugging level */
  RunType       runType;           /* Whether to run tests, or solve the full problem */
  PetscLogEvent createMeshEvent;
  PetscBool     showInitial, showSolution;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscBool     simplex;           /* Use simplices or tensor product cells */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  PetscBool     testPartition;     /* Use a fixed partitioning for testing */
  PetscReal     mu;                /* The shear modulus */
  PetscReal     p_wall;            /* The wall pressure */
} AppCtx;

#if 0
PETSC_STATIC_INLINE void Det2D(PetscReal *detJ, const PetscReal J[])
{
  *detJ = J[0]*J[3] - J[1]*J[2];
}
#endif

PETSC_STATIC_INLINE void Det3D(PetscReal *detJ, const PetscScalar J[])
{
  *detJ = PetscRealPart(J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
                        J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
                        J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
}

#if 0
PETSC_STATIC_INLINE void Cof2D(PetscReal C[], const PetscReal A[])
{
  C[0] =  A[3];
  C[1] = -A[2];
  C[2] = -A[1];
  C[3] =  A[0];
}
#endif

PETSC_STATIC_INLINE void Cof3D(PetscReal C[], const PetscScalar A[])
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
  options->debug           = 0;
  options->runType         = RUN_FULL;
  options->dim             = 3;
  options->interpolate     = PETSC_FALSE;
  options->simplex         = PETSC_TRUE;
  options->refinementLimit = 0.0;
  options->mu              = 1.0;
  options->p_wall          = 0.4;
  options->testPartition   = PETSC_FALSE;
  options->showInitial     = PETSC_FALSE;
  options->showSolution    = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Nonlinear elasticity problem options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex77.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex77.c", runTypes, 2, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);

  options->runType = (RunType) run;

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex77.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex77.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices or tensor product cells", "ex77.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex77.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex77.c", options->testPartition, &options->testPartition, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-shear_modulus", "The shear modulus", "ex77.c", options->mu, &options->mu, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-wall_pressure", "The wall pressure", "ex77.c", options->p_wall, &options->p_wall, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex77.c", options->showInitial, &options->showInitial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex77.c", options->showSolution, &options->showSolution, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMVecViewLocal(DM dm, Vec v, PetscViewer viewer)
{
  Vec            lv;
  PetscInt       p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRMPI(ierr);
  ierr = DMGetLocalVector(dm, &lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, lv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Local function\n");CHKERRQ(ierr);
  for (p = 0; p < size; ++p) {
    if (p == rank) {ierr = VecView(lv, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &lv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  const PetscInt cells[3]        = {3, 3, 3};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, user->simplex ? NULL : cells, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
  /* Label the faces (bit of a hack here, until it is properly implemented for simplices) */
  {
    DM              cdm;
    DMLabel         label;
    IS              is;
    PetscInt        d, dim = user->dim, b, f, Nf;
    const PetscInt *faces;
    PetscInt        csize;
    PetscScalar    *coords = NULL;
    PetscSection    cs;
    Vec             coordinates ;

    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    ierr = DMGetStratumIS(*dm, "boundary", 1,  &is);CHKERRQ(ierr);
    if (is) {
      PetscReal faceCoord;
      PetscInt  v;

      ierr = ISGetLocalSize(is, &Nf);CHKERRQ(ierr);
      ierr = ISGetIndices(is, &faces);CHKERRQ(ierr);

      ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
      ierr = DMGetCoordinateDM(*dm, &cdm);CHKERRQ(ierr);
      ierr = DMGetLocalSection(cdm, &cs);CHKERRQ(ierr);

      /* Check for each boundary face if any component of its centroid is either 0.0 or 1.0 */
      for (f = 0; f < Nf; ++f) {
        ierr = DMPlexVecGetClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
        /* Calculate mean coordinate vector */
        for (d = 0; d < dim; ++d) {
          const PetscInt Nv = csize/dim;
          faceCoord = 0.0;
          for (v = 0; v < Nv; ++v) faceCoord += PetscRealPart(coords[v*dim+d]);
          faceCoord /= Nv;
          for (b = 0; b < 2; ++b) {
            if (PetscAbs(faceCoord - b*1.0) < PETSC_SMALL) {
              ierr = DMSetLabelValue(*dm, "Faces", faces[f], d*2+b+1);CHKERRQ(ierr);
            }
          }
        }
        ierr = DMPlexVecRestoreClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(is, &faces);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  }
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);

    /* Refine mesh using a volume constraint */
    ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    if (user->simplex) {ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);}
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    /* Setup test partitioning */
    if (user->testPartition) {
      PetscInt         triSizes_n2[2]       = {4, 4};
      PetscInt         triPoints_n2[8]      = {3, 5, 6, 7, 0, 1, 2, 4};
      PetscInt         triSizes_n3[3]       = {2, 3, 3};
      PetscInt         triPoints_n3[8]      = {3, 5, 1, 6, 7, 0, 2, 4};
      PetscInt         triSizes_n5[5]       = {1, 2, 2, 1, 2};
      PetscInt         triPoints_n5[8]      = {3, 5, 6, 4, 7, 0, 1, 2};
      PetscInt         triSizes_ref_n2[2]   = {8, 8};
      PetscInt         triPoints_ref_n2[16] = {1, 5, 6, 7, 10, 11, 14, 15, 0, 2, 3, 4, 8, 9, 12, 13};
      PetscInt         triSizes_ref_n3[3]   = {5, 6, 5};
      PetscInt         triPoints_ref_n3[16] = {1, 7, 10, 14, 15, 2, 6, 8, 11, 12, 13, 0, 3, 4, 5, 9};
      PetscInt         triSizes_ref_n5[5]   = {3, 4, 3, 3, 3};
      PetscInt         triPoints_ref_n5[16] = {1, 7, 10, 2, 11, 13, 14, 5, 6, 15, 0, 8, 9, 3, 4, 12};
      PetscInt         tetSizes_n2[2]       = {3, 3};
      PetscInt         tetPoints_n2[6]      = {1, 2, 3, 0, 4, 5};
      const PetscInt  *sizes = NULL;
      const PetscInt  *points = NULL;
      PetscInt         cEnd;
      PetscMPIInt      rank, size;

      ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
      ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
      ierr = DMPlexGetHeightStratum(*dm, 0, NULL, &cEnd);CHKERRQ(ierr);
      if (!rank) {
        if (dim == 2 && user->simplex && size == 2 && cEnd == 8) {
           sizes = triSizes_n2; points = triPoints_n2;
        } else if (dim == 2 && user->simplex && size == 3 && cEnd == 8) {
          sizes = triSizes_n3; points = triPoints_n3;
        } else if (dim == 2 && user->simplex && size == 5 && cEnd == 8) {
          sizes = triSizes_n5; points = triPoints_n5;
        } else if (dim == 2 && user->simplex && size == 2 && cEnd == 16) {
           sizes = triSizes_ref_n2; points = triPoints_ref_n2;
        } else if (dim == 2 && user->simplex && size == 3 && cEnd == 16) {
          sizes = triSizes_ref_n3; points = triPoints_ref_n3;
        } else if (dim == 2 && user->simplex && size == 5 && cEnd == 16) {
          sizes = triSizes_ref_n5; points = triPoints_ref_n5;
        } else if (dim == 3 && user->simplex && size == 2 && cEnd == 6) {
          sizes = tetSizes_n2; points = tetPoints_n2;
        } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "No stored partition matching run parameters");
      }
      ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
      ierr = PetscPartitionerShellSetPartition(part, size, sizes, points);CHKERRQ(ierr);
    } else {
      ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    }
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupProblem(DM dm, PetscInt dim, AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, NULL, f1_u_3d);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_p_3d, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  g3_uu_3d);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_up_3d, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_pu_3d, NULL,  NULL);CHKERRQ(ierr);

  ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u_3d, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetBdJacobian(prob, 0, 0, NULL, g1_bd_uu_3d, NULL, NULL);CHKERRQ(ierr);

  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "fixed", "Faces", 0, 0, NULL, (void (*)(void)) coordinates, NULL, 0, NULL, user);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm, DM_BC_NATURAL, "pressure", "Faces", 0, 0, NULL, NULL, NULL, 0, NULL, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*matFuncs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {elasticityMaterial, wallPressure};
  Vec            A;
  void          *ctxs[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ctxs[0] = user; ctxs[1] = user;
  ierr = DMCreateLocalVector(dmAux, &A);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, matFuncs, ctxs, INSERT_ALL_VALUES, A);CHKERRQ(ierr);
  ierr = DMSetAuxiliaryVec(dm, NULL, 0, A);CHKERRQ(ierr);
  ierr = VecDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupNearNullSpace(DM dm, AppCtx *user)
{
  /* Set up the near null space (a.k.a. rigid body modes) that will be used by the multigrid preconditioner */
  DM             subdm;
  MatNullSpace   nearNullSpace;
  PetscInt       fields = 0;
  PetscObject    deformation;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateSubDM(dm, 1, &fields, NULL, &subdm);CHKERRQ(ierr);
  ierr = DMPlexCreateRigidBody(subdm, 0, &nearNullSpace);CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, NULL, &deformation);CHKERRQ(ierr);
  ierr = PetscObjectCompose(deformation, "nearnullspace", (PetscObject) nearNullSpace);CHKERRQ(ierr);
  ierr = DMDestroy(&subdm);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nearNullSpace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscInt NfAux, PetscFE feAux[], AppCtx *user)
{
  DM             dmAux, coordDM;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
  if (!feAux) PetscFunctionReturn(0);
  ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
  for (f = 0; f < NfAux; ++f) {ierr = DMSetField(dmAux, f, NULL, (PetscObject) feAux[f]);CHKERRQ(ierr);}
  ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
  ierr = SetupMaterial(dm, dmAux, user);CHKERRQ(ierr);
  ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm;
  const PetscInt  dim   = user->dim;
  const PetscBool simplex = user->simplex;
  PetscFE         fe[2], feAux[2];
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  /* Create finite element */
  ierr = PetscFECreateDefault(comm, dim, dim, simplex, "def_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "deformation");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[1]);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "elastMat_", PETSC_DEFAULT, &feAux[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feAux[0], "elasticityMaterial");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], feAux[0]);CHKERRQ(ierr);
  /* It is not yet possible to define a field on a submesh (e.g. a boundary), so we will use a normal finite element */
  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "wall_pres_", PETSC_DEFAULT, &feAux[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feAux[1], "wall_pressure");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], feAux[1]);CHKERRQ(ierr);

  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, dim, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = SetupAuxDM(cdm, 2, feAux, user);CHKERRQ(ierr);
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feAux[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feAux[1]);CHKERRQ(ierr);
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
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = SetupNearNullSpace(dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMSetMatType(dm,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  A = J;

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, NULL, NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  {
    PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx);
    initialGuess[0] = coordinates; initialGuess[1] = zero_scalar;
    ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
  }
  if (user.showInitial) {ierr = DMVecViewLocal(dm, u, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  if (user.runType == RUN_FULL) {
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %D\n", its);CHKERRQ(ierr);
    if (user.showSolution) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution\n");CHKERRQ(ierr);
      ierr = VecChop(u, 3.0e-9);CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  } else {
    PetscReal res = 0.0;

    /* Check initial guess */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* Check residual */
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Vec b;

      ierr = SNESComputeJacobian(snes, u, A, A);CHKERRQ(ierr);
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
      ierr = MatMult(A, u, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n");CHKERRQ(ierr);
      ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
      ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
    }
  }
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

  if (A != J) {ierr = MatDestroy(&A);CHKERRQ(ierr);}
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: !complex

  test:
    suffix: 0
    requires: ctetgen !single
    args: -run_type full -dim 3 -dm_refine 2 -interpolate 1 -bc_fixed 1 -bc_pressure 2 -wall_pressure 0.4 -def_petscspace_degree 2 -pres_petscspace_degree 1 -elastMat_petscspace_degree 0 -wall_pres_petscspace_degree 0 -snes_rtol 1e-05 -ksp_type fgmres -ksp_rtol 1e-10 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -show_solution 0

  test:
    suffix: 1
    requires: ctetgen superlu_dist
    nsize: 2
    args: -run_type full -dim 3 -dm_refine 0 -interpolate 1 -test_partition -bc_fixed 1 -bc_pressure 2 -wall_pressure 0.4 -def_petscspace_degree 2 -pres_petscspace_degree 1 -elastMat_petscspace_degree 0 -wall_pres_petscspace_degree 0 -snes_rtol 1e-05 -ksp_type fgmres -ksp_rtol 1e-10 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -show_solution 0
    timeoutfactor: 2

  test:
    suffix: 2
    requires: ctetgen !single
    args: -run_type full -dim 3 -dm_refine 2 -interpolate 1 -bc_fixed 3,4,5,6 -bc_pressure 2 -wall_pressure 1.0 -def_petscspace_degree 2 -pres_petscspace_degree 1 -elastMat_petscspace_degree 0 -wall_pres_petscspace_degree 0 -snes_rtol 1e-05 -ksp_type fgmres -ksp_rtol 1e-10 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -show_solution 0

  test:
    requires: ctetgen superlu_dist
    suffix: 4
    nsize: 2
    args: -run_type full -dim 3 -dm_refine 0 -interpolate 1 -test_partition -bc_fixed 1 -bc_pressure 2 -wall_pressure 0.4 -def_petscspace_degree 2 -pres_petscspace_degree 1 -elastMat_petscspace_degree 0 -wall_pres_petscspace_degree 0 -snes_rtol 1e-05 -ksp_type fgmres -ksp_rtol 1e-10 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -show_solution 0
    output_file: output/ex77_1.out

  test:
    requires: ctetgen !single
    suffix: 3
    args: -run_type full -dim 3 -dm_refine 2 -interpolate 1 -bc_fixed 3,4,5,6 -bc_pressure 2 -wall_pressure 1.0 -def_petscspace_degree 2 -pres_petscspace_degree 1 -elastMat_petscspace_degree 0 -wall_pres_petscspace_degree 0 -snes_rtol 1e-05 -ksp_type fgmres -ksp_rtol 1e-10 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -show_solution 0
    output_file: output/ex77_2.out

  #TODO: this example deadlocks for me when using ParMETIS
  test:
    requires: ctetgen superlu_dist !single
    suffix: 2_par
    nsize: 4
    args: -run_type full -dim 3 -dm_refine 2 -interpolate 1 -bc_fixed 3,4,5,6 -bc_pressure 2 -wall_pressure 1.0 -def_petscspace_degree 2 -pres_petscspace_degree 1 -elastMat_petscspace_degree 0 -wall_pres_petscspace_degree 0 -snes_rtol 1e-05 -ksp_type fgmres -ksp_rtol 1e-10 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -show_solution 0 -petscpartitioner_type simple
    output_file: output/ex77_2.out

TEST*/
