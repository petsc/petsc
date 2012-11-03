static char help[] = "Stokes Problem in 2d and 3d with simplicial finite elements.\n\
We solve the Stokes problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMCOMPLEX) to discretize it.\n\n\n";

/*
The isoviscous Stokes problem, which we discretize using the finite
element method on an unstructured mesh. The weak form equations are

  < \nabla v, \nabla u + {\nabla u}^T > - < \nabla\cdot v, p > + < v, f > = 0
  < q, \nabla\cdot v >                                                    = 0

We start with homogeneous Dirichlet conditions. We will expand this as the set
of test problems is developed.

Discretization:

We use a Python script to generate a tabulation of the finite element basis
functions at quadrature points, which we put in a C header file. The generic
command would be:

    bin/pythonscripts/PetscGenerateFEMQuadrature.py dim order dim 1 laplacian dim order 1 1 gradient src/snes/examples/tutorials/ex62.h

We can currently generate an arbitrary order Lagrange element. The underlying
FIAT code is capable of handling more exotic elements, but these have not been
tested with this code.

Field Data:

  Sieve data is organized by point, and the closure operation just stacks up the
data from each sieve point in the closure. Thus, for a P_2-P_1 Stokes element, we
have

  cl{e} = {f e_0 e_1 e_2 v_0 v_1 v_2}
  x     = [u_{e_0} v_{e_0} u_{e_1} v_{e_1} u_{e_2} v_{e_2} u_{v_0} v_{v_0} p_{v_0} u_{v_1} v_{v_1} p_{v_1} u_{v_2} v_{v_2} p_{v_2}]

The problem here is that we would like to loop over each field separately for
integration. Therefore, the closure visitor in DMComplexVecGetClosure() reorders
the data so that each field is contiguous

  x'    = [u_{e_0} v_{e_0} u_{e_1} v_{e_1} u_{e_2} v_{e_2} u_{v_0} v_{v_0} u_{v_1} v_{v_1} u_{v_2} v_{v_2} p_{v_0} p_{v_1} p_{v_2}]

Likewise, DMComplexVecSetClosure() takes data partitioned by field, and correctly
puts it into the Sieve ordering.

Next Steps:

- Refine and show convergence of correct order automatically (use femTest.py)
- Fix InitialGuess for arbitrary disc (means making dual application work again)
- Redo slides from GUCASTutorial for this new example

For tensor product meshes, see SNES ex67, ex72
*/

#include <petscdmcomplex.h>
#include <petscsnes.h>

/*------------------------------------------------------------------------------
  This code can be generated using 'bin/pythonscripts/PetscGenerateFEMQuadrature.py dim order dim 1 laplacian dim order 1 1 gradient src/snes/examples/tutorials/ex62.h'
 -----------------------------------------------------------------------------*/
#include "ex62.h"

typedef enum {NEUMANN, DIRICHLET} BCType;
typedef enum {RUN_FULL, RUN_TEST} RunType;

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscFEM      fem;               /* REQUIRED to use DMComplexComputeResidualFEM() */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  RunType       runType;           /* Whether to run tests, or solve the full problem */
  PetscBool     jacobianMF;        /* Whether to calculate the Jacobian action on the fly */
  PetscLogEvent createMeshEvent;
  PetscBool     showInitial, showSolution;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  char          partitioner[2048]; /* The graph partitioner */
  /* GPU partitioning */
  PetscInt      numBatches;        /* The number of cell batches per kernel */
  PetscInt      numBlocks;         /* The number of concurrent blocks per kernel */
  /* Element quadrature */
  PetscQuadrature q[NUM_FIELDS];
  /* Problem definition */
  void (*f0Funcs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[]); /* f0_u(x,y,z), and f0_p(x,y,z) */
  void (*f1Funcs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[]); /* f1_u(x,y,z), and f1_p(x,y,z) */
  void (*g0Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g0[]); /* g0_uu(x,y,z), g0_up(x,y,z), g0_pu(x,y,z), and g0_pp(x,y,z) */
  void (*g1Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g1[]); /* g1_uu(x,y,z), g1_up(x,y,z), g1_pu(x,y,z), and g1_pp(x,y,z) */
  void (*g2Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g2[]); /* g2_uu(x,y,z), g2_up(x,y,z), g2_pu(x,y,z), and g2_pp(x,y,z) */
  void (*g3Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g3[]); /* g3_uu(x,y,z), g3_up(x,y,z), g3_pu(x,y,z), and g3_pp(x,y,z) */
  PetscScalar (*exactFuncs[NUM_BASIS_COMPONENTS_TOTAL])(const PetscReal x[]); /* The exact solution function u(x,y,z), v(x,y,z), and p(x,y,z) */
  BCType bcType;
} AppCtx;

PetscScalar zero(const PetscReal coords[]) {
  return 0.0;
}

/*
  In 2D we use exact solution:

    u = x^2 + y^2
    v = 2 x^2 - 2xy
    p = x + y - 1
    f_x = f_y = 3

  so that

    -\Delta u + \nabla p + f = <-4, -4> + <1, 1> + <3, 3> = 0
    \nabla \cdot u           = 2x - 2x                    = 0
*/
PetscScalar quadratic_u_2d(const PetscReal x[]) {
  return x[0]*x[0] + x[1]*x[1];
}

PetscScalar quadratic_v_2d(const PetscReal x[]) {
  return 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
}

PetscScalar linear_p_2d(const PetscReal x[]) {
  return x[0] + x[1] - 1.0;
}

void f0_u(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[]) {
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) {
    f0[comp] = 3.0;
  }
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
   u[Ncomp]          = {p} */
void f1_u(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[]) {
  const PetscInt dim   = SPATIAL_DIM_0;
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;
  PetscInt       comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      /* f1[comp*dim+d] = 0.5*(gradU[comp*dim+d] + gradU[d*dim+comp]); */
      f1[comp*dim+d] = gradU[comp*dim+d];
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z} */
void f0_p(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[]) {
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) {
    f0[0] += gradU[d*dim+d];
  }
}

void f1_p(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[]) {
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) {
    f1[d] = 0.0;
  }
}

/* < q, \nabla\cdot v >
   NcompI = 1, NcompJ = dim */
void g1_pu(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g1[]) {
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) {
    g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
  }
}

/* -< \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
void g2_up(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g2[]) {
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) {
    g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
  }
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g3[]) {
  const PetscInt dim   = SPATIAL_DIM_0;
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;
  PetscInt       compI, d;

  for (compI = 0; compI < Ncomp; ++compI) {
    for (d = 0; d < dim; ++d) {
      g3[((compI*Ncomp+compI)*dim+d)*dim+d] = 1.0;
    }
  }
}

/*
  In 3D we use exact solution:

    u = x^2 + y^2
    v = y^2 + z^2
    w = x^2 + y^2 - 2(x+y)z
    p = x + y + z - 3/2
    f_x = f_y = f_z = 3

  so that

    -\Delta u + \nabla p + f = <-4, -4, -4> + <1, 1, 1> + <3, 3, 3> = 0
    \nabla \cdot u           = 2x + 2y - 2(x + y)                   = 0
*/
PetscScalar quadratic_u_3d(const PetscReal x[]) {
  return x[0]*x[0] + x[1]*x[1];
}

PetscScalar quadratic_v_3d(const PetscReal x[]) {
  return x[1]*x[1] + x[2]*x[2];
}

PetscScalar quadratic_w_3d(const PetscReal x[]) {
  return x[0]*x[0] + x[1]*x[1] - 2.0*(x[0] + x[1])*x[2];
}

PetscScalar linear_p_3d(const PetscReal x[]) {
  return x[0] + x[1] + x[2] - 1.5;
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  const char    *bcTypes[2]  = {"neumann", "dirichlet"};
  const char    *runTypes[2] = {"full", "test"};
  PetscInt       bc, run;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->runType         = RUN_FULL;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;
  options->bcType          = DIRICHLET;
  options->numBatches      = 1;
  options->numBlocks       = 1;
  options->jacobianMF      = PETSC_FALSE;
  options->showInitial     = PETSC_FALSE;
  options->showSolution    = PETSC_TRUE;

  options->fem.quad    = (PetscQuadrature *) &options->q;
  options->fem.f0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f0Funcs;
  options->fem.f1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f1Funcs;
  options->fem.g0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g0Funcs;
  options->fem.g1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g1Funcs;
  options->fem.g2Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g2Funcs;
  options->fem.g3Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g3Funcs;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Stokes Problem Options", "DMCOMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex62.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  run = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex62.c", runTypes, 2, runTypes[options->runType], &run, PETSC_NULL);CHKERRQ(ierr);
  options->runType = (RunType) run;
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex62.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex62.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex62.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
  bc = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex62.c",bcTypes,2,bcTypes[options->bcType],&bc,PETSC_NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  ierr = PetscOptionsInt("-gpu_batches", "The number of cell batches per kernel", "ex62.c", options->numBatches, &options->numBatches, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-gpu_blocks", "The number of concurrent blocks per kernel", "ex62.c", options->numBlocks, &options->numBlocks, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-jacobian_mf", "Calculate the action of the Jacobian on the fly", "ex62.c", options->jacobianMF, &options->jacobianMF, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex62.c", options->showInitial, &options->showInitial, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex62.c", options->showSolution, &options->showSolution, PETSC_NULL);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject) dm)->comm, &numProcs);CHKERRQ(ierr);
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
  const char    *partitioner     = user->partitioner;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMComplexCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
  {
    DM refinedMesh     = PETSC_NULL;
    DM distributedMesh = PETSC_NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMComplexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    /* Distribute mesh over processes */
    ierr = DMComplexDistribute(*dm, partitioner, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupQuadrature"
PetscErrorCode SetupQuadrature(AppCtx *user) {
  PetscFunctionBegin;
  user->fem.quad[0].numQuadPoints = NUM_QUADRATURE_POINTS_0;
  user->fem.quad[0].quadPoints    = points_0;
  user->fem.quad[0].quadWeights   = weights_0;
  user->fem.quad[0].numBasisFuncs = NUM_BASIS_FUNCTIONS_0;
  user->fem.quad[0].numComponents = NUM_BASIS_COMPONENTS_0;
  user->fem.quad[0].basis         = Basis_0;
  user->fem.quad[0].basisDer      = BasisDerivatives_0;
  user->fem.quad[1].numQuadPoints = NUM_QUADRATURE_POINTS_1;
  user->fem.quad[1].quadPoints    = points_1;
  user->fem.quad[1].quadWeights   = weights_1;
  user->fem.quad[1].numBasisFuncs = NUM_BASIS_FUNCTIONS_1;
  user->fem.quad[1].numComponents = NUM_BASIS_COMPONENTS_1;
  user->fem.quad[1].basis         = Basis_1;
  user->fem.quad[1].basisDer      = BasisDerivatives_1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
/*
  There is a problem here with uninterpolated meshes. The index in numDof[] is not dimension in this case,
  but sieve depth.
*/
PetscErrorCode SetupSection(DM dm, AppCtx *user) {
  PetscSection   section;
  const PetscInt numFields           = NUM_FIELDS;
  PetscInt       dim                 = user->dim;
  PetscInt       numBC               = 0;
  PetscInt       numComp[NUM_FIELDS] = {NUM_BASIS_COMPONENTS_0, NUM_BASIS_COMPONENTS_1};
  PetscInt       bcFields[1]         = {0};
  IS             bcPoints[1]         = {PETSC_NULL};
  PetscInt       numDof[NUM_FIELDS*(SPATIAL_DIM_0+1)];
  PetscInt       f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dim != SPATIAL_DIM_0) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Spatial dimension %d should be %d", dim, SPATIAL_DIM_0);
  if (dim != SPATIAL_DIM_1) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Spatial dimension %d should be %d", dim, SPATIAL_DIM_1);
  for (d = 0; d <= dim; ++d) {
    numDof[0*(dim+1)+d] = numDof_0[d];
    numDof[1*(dim+1)+d] = numDof_1[d];
  }
  for (f = 0; f < numFields; ++f) {
    for (d = 1; d < dim; ++d) {
      if ((numDof[f*(dim+1)+d] > 0) && !user->interpolate) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  if (user->bcType == DIRICHLET) {
    numBC = 1;
    ierr  = DMComplexGetStratumIS(dm, "marker", 1, &bcPoints[0]);CHKERRQ(ierr);
  }
  ierr = DMComplexCreateSection(dm, dim, numFields, numComp, numDof, numBC, bcFields, bcPoints, &section);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 0, "velocity");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 1, "pressure");CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  if (user->bcType == DIRICHLET) {
    ierr = ISDestroy(&bcPoints[0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupExactSolution"
PetscErrorCode SetupExactSolution(DM dm, AppCtx *user) {
  PetscFEM      *fem = &user->fem;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  fem->f0Funcs[0] = f0_u;
  fem->f0Funcs[1] = f0_p;
  fem->f1Funcs[0] = f1_u;
  fem->f1Funcs[1] = f1_p;
  fem->g0Funcs[0] = PETSC_NULL;
  fem->g0Funcs[1] = PETSC_NULL;
  fem->g0Funcs[2] = PETSC_NULL;
  fem->g0Funcs[3] = PETSC_NULL;
  fem->g1Funcs[0] = PETSC_NULL;
  fem->g1Funcs[1] = PETSC_NULL;
  fem->g1Funcs[2] = g1_pu;      /* < q, \nabla\cdot v > */
  fem->g1Funcs[3] = PETSC_NULL;
  fem->g2Funcs[0] = PETSC_NULL;
  fem->g2Funcs[1] = g2_up;      /* < \nabla\cdot v, p > */
  fem->g2Funcs[2] = PETSC_NULL;
  fem->g2Funcs[3] = PETSC_NULL;
  fem->g3Funcs[0] = g3_uu;      /* < \nabla v, \nabla u + {\nabla u}^T > */
  fem->g3Funcs[1] = PETSC_NULL;
  fem->g3Funcs[2] = PETSC_NULL;
  fem->g3Funcs[3] = PETSC_NULL;
  switch(user->dim) {
  case 2:
    user->exactFuncs[0] = quadratic_u_2d;
    user->exactFuncs[1] = quadratic_v_2d;
    user->exactFuncs[2] = linear_p_2d;
    break;
  case 3:
    user->exactFuncs[0] = quadratic_u_3d;
    user->exactFuncs[1] = quadratic_v_3d;
    user->exactFuncs[2] = quadratic_w_3d;
    user->exactFuncs[3] = linear_p_3d;
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  ierr = DMComplexSetFEMIntegration(dm, FEMIntegrateResidualBatch, FEMIntegrateJacobianActionBatch, FEMIntegrateJacobianBatch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePressureNullSpace"
PetscErrorCode CreatePressureNullSpace(DM dm, AppCtx *user, MatNullSpace *nullSpace) {
  Vec            vec, localVec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localVec);CHKERRQ(ierr);
  ierr = VecSet(vec,  0.0);CHKERRQ(ierr);
  /* Put a constant in for all pressures
     Could change this to project the constant function onto the pressure space (when that is finished) */
  {
    PetscSection section;
    PetscInt     pStart, pEnd, p;
    PetscScalar *a;

    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = VecGetArray(localVec, &a);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt fDim, off, d;

      ierr = PetscSectionGetFieldDof(section, p, 1, &fDim);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(section, p, 1, &off);CHKERRQ(ierr);
      for (d = 0; d < fDim; ++d) {
        a[off+d] = 1.0;
      }
    }
    ierr = VecRestoreArray(localVec, &a);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobalBegin(dm, localVec, INSERT_VALUES, vec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localVec, INSERT_VALUES, vec);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localVec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, PETSC_NULL);CHKERRQ(ierr);
  if (user->debug) {
    ierr = PetscPrintf(((PetscObject) dm)->comm, "Pressure Null Space\n");CHKERRQ(ierr);
    ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceCreate(((PetscObject) dm)->comm, PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &vec);CHKERRQ(ierr);
  /* New style for field null spaces */
  {
    PetscObject  pressure;
    MatNullSpace nullSpacePres;

    ierr = DMGetField(dm, 1, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(pressure->comm, PETSC_TRUE, 0, PETSC_NULL, &nullSpacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullSpacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpacePres);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianAction"
/*
  FormJacobianAction - Form the global Jacobian action Y = JX from the global input X

  Input Parameters:
+ mat - The Jacobian shell matrix
- X  - Global input vector

  Output Parameter:
. Y  - Local output vector

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: FormJacobianActionLocal()
*/
PetscErrorCode FormJacobianAction(Mat J, Vec X,  Vec Y)
{
  JacActionCtx    *ctx;
  DM               dm;
  Vec              dummy, localX, localY;
  PetscInt         N, n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 3);
  ierr = MatShellGetContext(J, &ctx);CHKERRQ(ierr);
  dm = ctx->dm;

  /* determine whether X = localX */
  ierr = DMGetLocalVector(dm, &dummy);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localY);CHKERRQ(ierr);
  /* TODO: THIS dummy restore is necessary here so that the first available local vector has boundary conditions in it
   I think the right thing to do is have the user put BC into a local vector and give it to us
  */
  ierr = DMRestoreLocalVector(dm, &dummy);CHKERRQ(ierr);
  ierr = VecGetSize(X, &N);CHKERRQ(ierr);
  ierr = VecGetSize(localX, &n);CHKERRQ(ierr);

  if (n != N){ /* X != localX */
    ierr = VecSet(localX, 0.0);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DMComplexComputeJacobianActionFEM(dm, J, localX, localY, ctx->user);CHKERRQ(ierr);
  if (n != N){
    ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  }
  ierr = VecSet(Y, 0.0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localY, ADD_VALUES, Y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localY, ADD_VALUES, Y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localY);CHKERRQ(ierr);
  if (0) {
    Vec       r;
    PetscReal norm;

    ierr = VecDuplicate(X, &r);CHKERRQ(ierr);
    ierr = MatMult(ctx->J, X, r);CHKERRQ(ierr);
    ierr = VecAXPY(r, -1.0, Y);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &norm);CHKERRQ(ierr);
    if (norm > 1.0e-8) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Jacobian Action Input:\n");CHKERRQ(ierr);
      ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Jacobian Action Result:\n");CHKERRQ(ierr);
      ierr = VecView(Y, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Difference:\n");CHKERRQ(ierr);
      ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      SETERRQ1(((PetscObject) J)->comm, PETSC_ERR_ARG_WRONG, "The difference with assembled multiply is too large %g", norm);
    }
    ierr = VecDestroy(&r);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  Vec            u,r;                  /* solution, residual vectors */
  Mat            A,J;                  /* Jacobian matrix */
  MatNullSpace   nullSpace;            /* May be necessary for pressure */
  AppCtx         user;                 /* user-defined work context */
  JacActionCtx   userJ;                /* context for Jacobian MF action */
  PetscInt       its;                  /* iterations for convergence */
  PetscReal      error = 0.0;          /* L_2 error in the solution */
  const PetscInt numComponents = NUM_BASIS_COMPONENTS_TOTAL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, user.dm);CHKERRQ(ierr);

  ierr = SetupExactSolution(user.dm, &user);CHKERRQ(ierr);
  ierr = SetupQuadrature(&user);CHKERRQ(ierr);
  ierr = SetupSection(user.dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMCreateMatrix(user.dm, MATAIJ, &J);CHKERRQ(ierr);
  if (user.jacobianMF) {
    PetscInt M, m, N, n;

    ierr = MatGetSize(J, &M, &N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(J, &m, &n);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetSizes(A, m, n, M, N);CHKERRQ(ierr);
    ierr = MatSetType(A, MATSHELL);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A, MATOP_MULT, (void (*)(void)) FormJacobianAction);CHKERRQ(ierr);
    userJ.dm   = user.dm;
    userJ.J    = J;
    userJ.user = &user;
    ierr = DMCreateLocalVector(user.dm, &userJ.u);CHKERRQ(ierr);
    ierr = MatShellSetContext(A, &userJ);CHKERRQ(ierr);
  } else {
    A = J;
  }
  ierr = SNESSetJacobian(snes, A, J, SNESDMComputeJacobian, &user);CHKERRQ(ierr);
  ierr = CreatePressureNullSpace(user.dm, &user, &nullSpace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J, nullSpace);CHKERRQ(ierr);
  if (A != J) {
    ierr = MatSetNullSpace(A, nullSpace);CHKERRQ(ierr);
  }

  ierr = DMSetLocalFunction(user.dm, (DMLocalFunction1) DMComplexComputeResidualFEM);CHKERRQ(ierr);
  ierr = DMSetLocalJacobian(user.dm, (DMLocalJacobian1) DMComplexComputeJacobianFEM);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, SNESDMComputeFunction, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMComplexProjectFunction(user.dm, numComponents, user.exactFuncs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  if (user.showInitial) {ierr = DMVecViewLocal(user.dm, u, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
  if (user.runType == RUN_FULL) {
    PetscScalar (*initialGuess[numComponents])(const PetscReal x[]);
    PetscInt c;

    for (c = 0; c < numComponents; ++c) {initialGuess[c] = zero;}
    ierr = DMComplexProjectFunction(user.dm, numComponents, initialGuess, INSERT_VALUES, u);CHKERRQ(ierr);
    if (user.showInitial) {ierr = DMVecViewLocal(user.dm, u, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes, PETSC_NULL, u);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %D\n", its);CHKERRQ(ierr);
    ierr = DMComplexComputeL2Diff(user.dm, user.fem.quad, user.exactFuncs, u, &error);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %.3g\n", error);CHKERRQ(ierr);
    if (user.showSolution) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution\n");CHKERRQ(ierr);
      ierr = VecChop(u, 3.0e-9);CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  } else {
    PetscReal res = 0.0;

    /* Check discretization error */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMComplexComputeL2Diff(user.dm, user.fem.quad, user.exactFuncs, u, &error);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);
    /* Check residual */
    ierr = SNESDMComputeFunction(snes, u, r, &user);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Vec          b;
      MatStructure flag;
      PetscBool    isNull;

      ierr = SNESDMComputeJacobian(snes, u, &A, &A, &flag, &user);CHKERRQ(ierr);
      ierr = MatNullSpaceTest(nullSpace, J, &isNull);CHKERRQ(ierr);
      if (!isNull) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESDMComputeFunction(snes, r, b, &user);CHKERRQ(ierr);
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

  if (user.runType == RUN_FULL) {
    PetscContainer c;
    PetscSection   s;
    Vec            uLocal;
    PetscViewer    viewer;

    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "ex62_sol.vtk");CHKERRQ(ierr);

    ierr = DMGetLocalVector(user.dm, &uLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(user.dm, u, INSERT_VALUES, uLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user.dm, u, INSERT_VALUES, uLocal);CHKERRQ(ierr);

    ierr = DMGetDefaultSection(user.dm, &s);CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject) uLocal)->comm, &c);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(c, s);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) uLocal, "section", (PetscObject) c);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);

    ierr = PetscObjectReference((PetscObject) user.dm);CHKERRQ(ierr); /* Needed because viewer destroys the DM */
    ierr = PetscObjectReference((PetscObject) uLocal);CHKERRQ(ierr); /* Needed because viewer destroys the Vec */
    ierr = PetscViewerVTKAddField(viewer, (PetscObject) user.dm, DMComplexVTKWriteAll, PETSC_VTK_POINT_FIELD, (PetscObject) uLocal);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(user.dm, &uLocal);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  if (user.jacobianMF) {
    ierr = VecDestroy(&userJ.u);CHKERRQ(ierr);
  }
  if (A != J) {
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
