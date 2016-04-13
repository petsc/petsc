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
} AppCtx;

/* Kronecker-delta */
static const PetscInt delta2D[2*2] = {1,0,0,1};
static const PetscInt delta3D[3*3] = {1,0,0,0,1,0,0,0,1};

/* Levi-Civita symbol */
static const PetscInt epsilon2D[2*2] = {0,1,-1,0};
static const PetscInt epsilon3D[3*3*3] = {0,0,0,0,0,1,0,-1,0,0,0,-1,0,0,0,1,0,0,0,1,0,-1,0,0,0,0,0};

PETSC_STATIC_INLINE void Det2D(PetscReal *detJ, const PetscReal J[])
{
  *detJ = J[0]*J[3] - J[1]*J[2];
}

PETSC_STATIC_INLINE void Det3D(PetscReal *detJ, const PetscReal J[])
{
  *detJ = (J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
           J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
           J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
}

PETSC_STATIC_INLINE void Cof2D(PetscReal C[], const PetscReal A[])
{
  C[0] =  A[3];
  C[1] = -A[2];
  C[2] = -A[1];
  C[3] =  A[0];
}

PETSC_STATIC_INLINE void Cof3D(PetscReal C[], const PetscReal A[])
{
  C[0*3+0] = A[1*3+1]*A[2*3+2] - A[1*3+2]*A[2*3+1];
  C[0*3+1] = A[1*3+2]*A[2*3+0] - A[1*3+0]*A[2*3+2];
  C[0*3+2] = A[1*3+0]*A[2*3+1] - A[1*3+1]*A[2*3+0];
  C[1*3+0] = A[0*3+2]*A[2*3+1] - A[0*3+1]*A[2*3+2];
  C[1*3+1] = A[0*3+0]*A[2*3+2] - A[0*3+2]*A[2*3+0];
  C[1*3+2] = A[0*3+1]*A[2*3+0] - A[0*3+0]*A[2*3+1];
  C[2*3+0] = A[0*3+1]*A[1*3+2] - A[0*3+2]*A[1*3+1];
  C[2*3+1] = A[0*3+2]*A[1*3+0] - A[0*3+0]*A[1*3+2];
  C[2*3+2] = A[0*3+0]*A[1*3+1] - A[0*3+1]*A[1*3+0];
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

void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  const PetscInt Ncomp = dim;

  PetscInt comp;
  for (comp = 0; comp < Ncomp; ++comp) f0[comp] = 0.0;
}

void f1_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscInt  Ncomp = dim;
  const PetscReal mu = a[0], p = u[Ncomp], kappa = 3.0;
  PetscReal       cofu_x[Ncomp*dim], detu_x;

  Cof3D(cofu_x, u_x);
  Det3D(&detu_x, u_x);

  /* f1 is the first Piola-Kirchhoff tensor */
  PetscInt        comp, d;
  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = mu * u_x[comp*dim+d] + cofu_x[comp*dim+d] * (p + kappa * (detu_x - 1.0));
    }
  }
}

void g3_uu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  const PetscInt  Ncomp = dim;
  const PetscReal mu = a[0], p = u[Ncomp], kappa = 3.0;
  PetscReal       cofu_x[Ncomp*dim], detu_x;

  Cof3D(cofu_x, u_x);
  Det3D(&detu_x, u_x);

  /* g3 is the first elasticity tensor, i.e. A_i^I_j^J = S^{IJ} g_{ij} + C^{KILJ} F^k_K F^l_L g_{kj} g_{li} */
  PetscInt compI, compJ, compK, d1, d2, d3;
  for (compI = 0; compI < Ncomp; ++compI) {
    for (compJ = 0; compJ < Ncomp; ++compJ) {
      for (d1 = 0; d1 < dim; ++d1) {
        for (d2 = 0; d2 < dim; ++d2) {
          for (compK = 0; compK < Ncomp; ++compK) {
            for (d3 = 0; d3 < dim; ++d3) {
              g3[((compI*Ncomp+compJ)*dim+d1)*dim+d2] += (p + kappa * (detu_x - 1.0)) * epsilon3D[(compI*Ncomp+compJ)*Ncomp+compK] * epsilon3D[(d1*dim+d2)*dim+d3] * u_x[compK*dim+d3];
            }
          }
          g3[((compI*Ncomp+compJ)*dim+d1)*dim+d2] += kappa * cofu_x[compI*dim+d1] * cofu_x[compJ*dim+d2] + delta3D[compI*Ncomp+compJ] * delta3D[d1*dim+d2] * mu;
        }
      }
    }
  }
}

void f0_bd_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
    PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  const PetscInt  Ncomp = dim;
  const PetscReal p = 0.4; /* Should this be specified as an auxiliary field? */
  PetscReal       cofu_x[Ncomp*dim];

  Cof3D(cofu_x, u_x);

  PetscInt comp, d;
  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0, f0[comp] = 0.0; d < dim; ++d) f0[comp] += cofu_x[comp*dim+d] * n[d];
    f0[comp] *= p;
  }
}

void f1_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f1[])
{
  const PetscInt Ncomp = dim;

  PetscInt       comp, d;
  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 0.0;
    }
  }
}

void g1_bd_uu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
    PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscScalar g1[])
{
  const PetscInt  Ncomp = dim;
  const PetscReal p = 0.4; /* Should this be specified as an auxiliary field? */

  PetscInt compI, compJ, compK, d1, d2, d3;
  for (compI = 0; compI < Ncomp; ++compI) {
    for (compJ = 0; compJ < Ncomp; ++compJ) {
      for (d2 = 0; d2 < dim; ++d2) {
        for (compK = 0; compK < Ncomp; ++compK) {
          for (d3 = 0; d3 < dim; ++d3) {
            for (d1 = 0; d1 < dim; ++d1) {
              g1[(compI*Ncomp+compJ)*dim+d2] += epsilon3D[(compI*Ncomp+compJ)*Ncomp+compK] * epsilon3D[(d2*dim+d3)*dim+d1] * u_x[compK*dim+d3] * n[d1];
            }
          }
        }
        g1[(compI*Ncomp+compJ)*dim+d2] *= p;
      }
    }
  }
}

void f0_p_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  PetscReal detu_x;
  Det3D(&detu_x, u_x);
  f0[0] = detu_x - 1.0;
}

void f1_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = 0.0;
}

void f0_bd_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
    PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  f0[0] = 0.0;
}
void f1_bd_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = 0.0;
}

void g1_pu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g1[])
{
  Cof3D(g1, u_x);
}

void g2_up_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscReal u[], const PetscReal u_t[], const PetscReal u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscReal a[], const PetscReal a_t[], const PetscReal a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscReal g2[])
{
  Cof3D(g2, u_x);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
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

  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex77.c", options->showInitial, &options->showInitial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex77.c", options->showSolution, &options->showSolution, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMVecViewLocal"
PetscErrorCode DMVecViewLocal(DM dm, Vec v, PetscViewer viewer)
{
  Vec            lv;
  PetscInt       p;
  PetscMPIInt    rank, numProcs;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &numProcs);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, lv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, lv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Local function\n");CHKERRQ(ierr);
  for (p = 0; p < numProcs; ++p) {
    if (p == rank) {ierr = VecView(lv, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &lv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  const PetscInt cells[3]        = {3, 3, 3};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  if (user->simplex) {ierr = DMPlexCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);}
  else               {ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, dm);CHKERRQ(ierr);}
  /* Label the faces (bit of a hack here, until it is properly implemented for simplices) */
  {
    DM              cdm;
    DMLabel         label;
    IS              is;
    PetscInt        d, dim = user->dim, b, f, Nf;
    const PetscInt *faces;
    PetscInt        csize;
    PetscReal      *coords = NULL;
    PetscSection    cs;
    Vec             coordinates ;

    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, label);CHKERRQ(ierr);
    ierr = DMGetStratumIS(*dm, "boundary", 1,  &is);CHKERRQ(ierr);
    ierr = DMCreateLabel(*dm, "Faces");CHKERRQ(ierr);
    if (is) {
      ierr = ISGetLocalSize(is, &Nf);CHKERRQ(ierr);
      ierr = ISGetIndices(is, &faces);CHKERRQ(ierr);

      ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
      ierr = DMGetCoordinateDM(*dm, &cdm);CHKERRQ(ierr);
      ierr = DMGetDefaultSection(cdm, &cs);

      /* Check for each boundary face if any component of its centroid is either 0.0 or 1.0 */
      PetscReal faceCoord;
      PetscInt  v;
      for (f = 0; f < Nf; ++f) {
        ierr = DMPlexVecGetClosure(cdm, cs, coordinates, faces[f], &csize, &coords);
        /* Calculate mean coordinate vector */
        const PetscInt Nv = csize/dim;
        for (d = 0; d < dim; ++d) {
          faceCoord = 0.0;
          for (v = 0; v < Nv; ++v) faceCoord += coords[v*dim+d];
          faceCoord /= Nv;
          for (b = 0; b < 2; ++b) {
            if (PetscAbs(faceCoord - b*1.0) < PETSC_SMALL) {
              ierr = DMSetLabelValue(*dm, "Faces", faces[f], d*2+b+1);CHKERRQ(ierr);
            }
          }
        }
        ierr = DMPlexVecRestoreClosure(cdm, cs, coordinates, faces[f], &csize, &coords);
      }
      ierr = ISRestoreIndices(is, &faces);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "Faces", &label);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
  }
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

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
      const PetscInt  *sizes = NULL;
      const PetscInt  *points = NULL;
      PetscPartitioner part;
      PetscInt         cEnd;
      PetscMPIInt      rank, numProcs;

      ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
      ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(*dm, 0, NULL, &cEnd);CHKERRQ(ierr);
      if (!rank) {
        if (dim == 2 && user->simplex && numProcs == 2 && cEnd == 8) {
           sizes = triSizes_n2; points = triPoints_n2;
        } else if (dim == 2 && user->simplex && numProcs == 3 && cEnd == 8) {
          sizes = triSizes_n3; points = triPoints_n3;
        } else if (dim == 2 && user->simplex && numProcs == 5 && cEnd == 8) {
          sizes = triSizes_n5; points = triPoints_n5;
        } else if (dim == 2 && user->simplex && numProcs == 2 && cEnd == 16) {
           sizes = triSizes_ref_n2; points = triPoints_ref_n2;
        } else if (dim == 2 && user->simplex && numProcs == 3 && cEnd == 16) {
          sizes = triSizes_ref_n3; points = triPoints_ref_n3;
        } else if (dim == 2 && user->simplex && numProcs == 5 && cEnd == 16) {
          sizes = triSizes_ref_n5; points = triPoints_ref_n5;
        } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "No stored partition matching run parameters");
      }
      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
      ierr = PetscPartitionerShellSetPartition(part, numProcs, sizes, points);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "SetupProblem"
PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u_3d);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_p_3d, f1_p);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL,  NULL,  g3_uu_3d);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL,  g2_up_3d, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_pu_3d, NULL,  NULL);CHKERRQ(ierr);

  ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u_3d, f1_bd_u);CHKERRQ(ierr);
  ierr = PetscDSSetBdResidual(prob, 1, f0_bd_p, f1_bd_p);CHKERRQ(ierr);
  ierr = PetscDSSetBdJacobian(prob, 0, 0, NULL, g1_bd_uu_3d, NULL, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupMaterial"
PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*matFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {elasticityMaterial};
  Vec            nu;
  void *ctxs[] = {user};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dmAux, &nu);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, matFuncs, ctxs, INSERT_ALL_VALUES, nu);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) nu);CHKERRQ(ierr);
  ierr = VecDestroy(&nu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupDiscretization"
PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm, coordDM, dmAux;
  const PetscInt  dim   = user->dim;
  const PetscBool simplex = user->simplex;
  PetscFE         fe[2], feBd[2], feAux;
  PetscQuadrature q;
  PetscDS         prob, probAux;
  PetscInt        order;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, dim, simplex, "def_", -1, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "deformation");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetOrder(q, &order);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "pres_", order, &fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim-1, dim, simplex, "bd_def_", order, &feBd[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feBd[0], "deformation");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim-1, 1, simplex, "bd_pres_", order, &feBd[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feBd[1], "pressure");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "elastMat_", order, &feAux);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feAux, "elasticityMaterial");CHKERRQ(ierr);
 

  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);
    ierr = PetscDSSetBdDiscretization(prob, 0, (PetscObject) feBd[0]);CHKERRQ(ierr);
    ierr = PetscDSSetBdDiscretization(prob, 1, (PetscObject) feBd[1]);CHKERRQ(ierr);
    ierr = SetupProblem(cdm, user);CHKERRQ(ierr);

    ierr = DMClone(cdm, &dmAux);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(cdm, &coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(probAux, 0, (PetscObject) feAux);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) cdm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
    ierr = SetupMaterial(cdm, dmAux, user);CHKERRQ(ierr);
    ierr = DMDestroy(&dmAux);CHKERRQ(ierr);

    const PetscInt Ncomp = dim;
    const PetscInt components[] = {0,1,2};
    const PetscInt Nfid = 1;
    const PetscInt fid[] = {1}; /* The fixed faces */
    const PetscInt Npid = 1;
    const PetscInt pid[] = {2}; /* The faces with pressure loading */
    ierr = DMAddBoundary(cdm, PETSC_TRUE, "fixed", "Faces", 0, Ncomp, components, (void (*)()) coordinates, Nfid, fid, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(cdm, PETSC_FALSE, "pressure", "Faces", 0, Ncomp, components, NULL, Npid, pid, user);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }

  {
    /* Set up the near null space (a.k.a. rigid body modes) that will be used by the multigrid preconditioner */
    DM           subdm;
    MatNullSpace nearNullSpace;
    PetscInt     fields = 0;
    PetscObject  deformation;
    ierr = DMCreateSubDM(dm, 1, &fields, NULL, &subdm);CHKERRQ(ierr);
    ierr = DMPlexCreateRigidBody(subdm, &nearNullSpace);CHKERRQ(ierr);
    ierr = DMGetField(dm, 0, &deformation);CHKERRQ(ierr);
    ierr = PetscObjectCompose(deformation, "nearnullspace", (PetscObject) nearNullSpace);CHKERRQ(ierr);
    ierr = DMDestroy(&subdm);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nearNullSpace);CHKERRQ(ierr);
  }

  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feBd[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feBd[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  DM             dm;                   /* problem definition */
  Vec            u,r;                  /* solution, residual vectors */
  Mat            A,J;                  /* Jacobian matrix */
  MatNullSpace   nullSpace;            /* May be necessary for pressure */
  AppCtx         user;                 /* user-defined work context */
  PetscInt       its;                  /* iterations for convergence */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMSetMatType(dm,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  A = J;

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, NULL, NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {coordinates, zero_scalar};
  ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Vec          b;
      PetscBool    isNull;

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
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", res);CHKERRQ(ierr);
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
  return 0;
}
