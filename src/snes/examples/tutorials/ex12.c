static char help[] = "Bratu nonlinear PDE in 2d.\n\
We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular\n\
domain, using a parallel unstructured mesh (DMMESH) to discretize it.\n\
The command line options include:\n\
  -lambda <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= lambda <= 6.81)\n\n";

/*T
   Concepts: SNES^parallel Bratu example
   Concepts: DMMESH^using unstructured grids;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation

            -Laplacian u - lambda*exp(u) = f(x,y),  0 < x,y < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1.

    A finite element approximation with the usual P_1 linear basis
    is used to discretize the boundary value problem to obtain a nonlinear
    system of equations.

      ./ex5 -draw_pause -1
      mpiexec -n 2 ./ex5 -log_summary

  ------------------------------------------------------------------------- */

/*
   Include "petscdmmesh.h" so that we can use unstructured meshes (DMMESHs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h    - base PETSc routines
     petscvec.h    - vectors               petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscdmmesh.h>
#include <petscsnes.h>

typedef enum {RUN_FULL, RUN_TEST, RUN_MESH} RunType;
typedef enum {NEUMANN, DIRICHLET} BCType;

/*------------------------------------------------------------------------------
  This code can be generated using config/PETSc/FEM.py

    import PETSc.FEM
    from FIAT.reference_element import default_simplex
    from FIAT.lagrange import Lagrange

    generator = PETSc.FEM.QuadratureGenerator()
    generator.setup()
    dim      = 2
    order    = 1
    elements = [Lagrange(default_simplex(dim), order))]
    generator.run(elements, filename)
 -----------------------------------------------------------------------------*/
#include <stdlib.h>

#define NUM_QUADRATURE_POINTS_0 1

/* Quadrature points
   - (x1,y1,x2,y2,...) */
static PetscReal points_0[1] = {0.0};

/* Quadrature weights
   - (v1,v2,...) */
static PetscReal weights_0[1] = {2.0};

#define NUM_BASIS_FUNCTIONS_0 2

/* Nodal basis function evaluations
    - basis function is fastest varying, then point */
static PetscReal Basis_0[2] = {
  0.5,
  0.5};

/* Nodal basis function derivative evaluations,
    - derivative direction fastest varying, then basis function, then point */
static PetscReal BasisDerivatives_0[2] = {
  -0.5,
  0.5};

#define NUM_DUAL_POINTS_0 2

/* Dual points
   - (x1,y1,x2,y2,...) */
static PetscReal dualPoints_0[2] = {
  -1.0,
  1.0};

#undef __FUNCT__

#define __FUNCT__ "IntegrateDualBasis_gen_0"

PetscReal IntegrateDualBasis_gen_0(const PetscReal *v0, const PetscReal *J, const int dualIndex, PetscReal (*func)(const PetscReal *coords))
{
  PetscReal refCoords[1];
  PetscReal coords[1];

  switch(dualIndex) {
    case 0:
      refCoords[0] = -1.0;
      break;
    case 1:
      refCoords[0] = 1.0;
      break;
    default:
      printf("dualIndex: %d\n", dualIndex);
      throw ALE::Exception("Bad dual index");
  }
  for(int d = 0; d < 1; d++)
  {
    coords[d] = v0[d];
    for(int e = 0; e < 1; e++)
    {
      coords[d] += J[d * 1 + e] * (refCoords[e] + 1.0);
    }
  }
  return (*func)(coords);
}

#undef __FUNCT__

#define __FUNCT__ "IntegrateBdDualBasis_gen_0"

PetscReal IntegrateBdDualBasis_gen_0(const PetscReal *v0, const PetscReal *J, const int dualIndex, PetscReal (*func)(const PetscReal *coords))
{
  PetscReal refCoords[1];
  PetscReal coords[2];

  switch(dualIndex) {
    case 0:
      refCoords[0] = -1.0;
      break;
    case 1:
      refCoords[0] = 1.0;
      break;
    default:
      printf("dualIndex: %d\n", dualIndex);
      throw ALE::Exception("Bad dual index");
  }
  for(int d = 0; d < 2; d++)
  {
    coords[d] = v0[d];
    for(int e = 0; e < 1; e++)
    {
      coords[d] += J[d * 2 + e] * (refCoords[e] + 1.0);
    }
  }
  return (*func)(coords);
}

#define NUM_QUADRATURE_POINTS_1 1

/* Quadrature points
   - (x1,y1,x2,y2,...) */
static PetscReal points_1[2] = {
  -0.333333333333,
  -0.333333333333};

/* Quadrature weights
   - (v1,v2,...) */
static PetscReal weights_1[1] = {2.0};

#define NUM_BASIS_FUNCTIONS_1 3

/* Nodal basis function evaluations
    - basis function is fastest varying, then point */
static PetscReal Basis_1[3] = {
  0.333333333333,
  0.333333333333,
  0.333333333333};

/* Nodal basis function derivative evaluations,
    - derivative direction fastest varying, then basis function, then point */
static PetscReal BasisDerivatives_1[6] = {
  -0.5,
  -0.5,
  0.5,
  0.0,
  0.0,
  0.5};

#define NUM_DUAL_POINTS_1 3

/* Dual points
   - (x1,y1,x2,y2,...) */
static PetscReal dualPoints_1[6] = {
  -1.0,
  -1.0,
  1.0,
  -1.0,
  -1.0,
  1.0};

#undef __FUNCT__

#define __FUNCT__ "IntegrateDualBasis_gen_1"

PetscReal IntegrateDualBasis_gen_1(const PetscReal *v0, const PetscReal *J, const int dualIndex, PetscReal (*func)(const PetscReal *coords))
{
  PetscReal refCoords[2];
  PetscReal coords[2];

  switch(dualIndex) {
    case 0:
      refCoords[0] = -1.0;
      refCoords[1] = -1.0;
      break;
    case 1:
      refCoords[0] = 1.0;
      refCoords[1] = -1.0;
      break;
    case 2:
      refCoords[0] = -1.0;
      refCoords[1] = 1.0;
      break;
    default:
      printf("dualIndex: %d\n", dualIndex);
      throw ALE::Exception("Bad dual index");
  }
  for(int d = 0; d < 2; d++)
  {
    coords[d] = v0[d];
    for(int e = 0; e < 2; e++)
    {
      coords[d] += J[d * 2 + e] * (refCoords[e] + 1.0);
    }
  }
  return (*func)(coords);
}

#undef __FUNCT__

#define __FUNCT__ "IntegrateBdDualBasis_gen_1"

PetscReal IntegrateBdDualBasis_gen_1(const PetscReal *v0, const PetscReal *J, const int dualIndex, PetscReal (*func)(const PetscReal *coords))
{
  PetscReal refCoords[2];
  PetscReal coords[3];

  switch(dualIndex) {
    case 0:
      refCoords[0] = -1.0;
      refCoords[1] = -1.0;
      break;
    case 1:
      refCoords[0] = 1.0;
      refCoords[1] = -1.0;
      break;
    case 2:
      refCoords[0] = -1.0;
      refCoords[1] = 1.0;
      break;
    default:
      printf("dualIndex: %d\n", dualIndex);
      throw ALE::Exception("Bad dual index");
  }
  for(int d = 0; d < 3; d++)
  {
    coords[d] = v0[d];
    for(int e = 0; e < 2; e++)
    {
      coords[d] += J[d * 3 + e] * (refCoords[e] + 1.0);
    }
  }
  return (*func)(coords);
}


#define NUM_QUADRATURE_POINTS_2 1

/* Quadrature points
   - (x1,y1,x2,y2,...) */
static PetscReal points_2[3] = {
  -0.5,
  -0.5,
  -0.5};

/* Quadrature weights
   - (v1,v2,...) */
static PetscReal weights_2[1] = {1.33333333333};

#define NUM_BASIS_FUNCTIONS_2 4

/* Nodal basis function evaluations
    - basis function is fastest varying, then point */
static PetscReal Basis_2[4] = {
  0.25,
  0.25,
  0.25,
  0.25};

/* Nodal basis function derivative evaluations,
    - derivative direction fastest varying, then basis function, then point */
static PetscReal BasisDerivatives_2[12] = {
  -0.5,
  -0.5,
  -0.5,
  0.5,
  0.0,
  0.0,
  0.0,
  0.5,
  0.0,
  0.0,
  0.0,
  0.5};

#define NUM_DUAL_POINTS_2 4

/* Dual points
   - (x1,y1,x2,y2,...) */
static PetscReal dualPoints_2[12] = {
  -1.0,
  -1.0,
  -1.0,
  1.0,
  -1.0,
  -1.0,
  -1.0,
  1.0,
  -1.0,
  -1.0,
  -1.0,
  1.0};

#undef __FUNCT__

#define __FUNCT__ "IntegrateDualBasis_gen_2"

PetscReal IntegrateDualBasis_gen_2(const PetscReal *v0, const PetscReal *J, const int dualIndex, PetscReal (*func)(const PetscReal *coords))
{
  PetscReal refCoords[3];
  PetscReal coords[3];

  switch(dualIndex) {
    case 0:
      refCoords[0] = -1.0;
      refCoords[1] = -1.0;
      refCoords[2] = -1.0;
      break;
    case 1:
      refCoords[0] = 1.0;
      refCoords[1] = -1.0;
      refCoords[2] = -1.0;
      break;
    case 2:
      refCoords[0] = -1.0;
      refCoords[1] = 1.0;
      refCoords[2] = -1.0;
      break;
    case 3:
      refCoords[0] = -1.0;
      refCoords[1] = -1.0;
      refCoords[2] = 1.0;
      break;
    default:
      printf("dualIndex: %d\n", dualIndex);
      throw ALE::Exception("Bad dual index");
  }
  for(int d = 0; d < 3; d++)
  {
    coords[d] = v0[d];
    for(int e = 0; e < 3; e++)
    {
      coords[d] += J[d * 3 + e] * (refCoords[e] + 1.0);
    }
  }
  return (*func)(coords);
}

#undef __FUNCT__

#define __FUNCT__ "IntegrateBdDualBasis_gen_2"

PetscReal IntegrateBdDualBasis_gen_2(const PetscReal *v0, const PetscReal *J, const int dualIndex, PetscReal (*func)(const PetscReal *coords))
{
  PetscReal refCoords[3];
  PetscReal coords[4];

  switch(dualIndex) {
    case 0:
      refCoords[0] = -1.0;
      refCoords[1] = -1.0;
      refCoords[2] = -1.0;
      break;
    case 1:
      refCoords[0] = 1.0;
      refCoords[1] = -1.0;
      refCoords[2] = -1.0;
      break;
    case 2:
      refCoords[0] = -1.0;
      refCoords[1] = 1.0;
      refCoords[2] = -1.0;
      break;
    case 3:
      refCoords[0] = -1.0;
      refCoords[1] = -1.0;
      refCoords[2] = 1.0;
      break;
    default:
      printf("dualIndex: %d\n", dualIndex);
      throw ALE::Exception("Bad dual index");
  }
  for(int d = 0; d < 4; d++)
  {
    coords[d] = v0[d];
    for(int e = 0; e < 3; e++)
    {
      coords[d] += J[d * 4 + e] * (refCoords[e] + 1.0);
    }
  }
  return (*func)(coords);
}

/*------------------------------------------------------------------------------
  end of generated code
 -----------------------------------------------------------------------------*/

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  DM            dm;                /* The unstructured mesh data structure */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  RunType       run;               /* The run type */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  char          partitioner[2048]; /* The graph partitioner */
  /* Element quadrature */
  PetscQuadrature q;
  /* Problem specific parameters */
  BCType        bcType;            /* The type of boundary conditions */
  PetscReal     lambda;            /* The Bratu problem parameter */
  PetscScalar (*rhsFunc)(const PetscReal []);   /* The rhs function f(x,y,z) */
  PetscScalar (*exactFunc)(const PetscReal []); /* The exact solution function u(x,y,z) */
  Vec           exactSol;          /* The discrete exact solution */
  Vec           error;             /* The discrete cell-wise error */
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode FormInitialGuess(Vec X, PetscScalar (*guessFunc)(const PetscReal []), InsertMode mode, AppCtx *user);
extern PetscErrorCode FormFunctionLocal(DM dm, Vec X, Vec F, AppCtx *user);
extern PetscErrorCode FormJacobianLocal(DM dm, Vec X, Mat J, AppCtx *user);

PetscReal lambda = 0.0;
PetscScalar guess(const PetscReal coords[]) {
  PetscScalar scale = lambda/(lambda+1.0);
  return scale*(0.5 - fabs(coords[0]-0.5))*(0.5 - fabs(coords[1]-0.5));
}

PetscScalar zero(const PetscReal coords[]) {
  return 0.0;
}

PetscScalar constant(const PetscReal x[]) {
  return -4.0;
};

PetscScalar nonlinear_2d(const PetscReal x[]) {
  return -4.0 - lambda*PetscExpScalar(x[0]*x[0] + x[1]*x[1]);
};

PetscScalar linear_2d(const PetscReal x[]) {
  return -6.0*(x[0] - 0.5) - 6.0*(x[1] - 0.5);
};

PetscScalar quadratic_2d(const PetscReal x[]) {
  return x[0]*x[0] + x[1]*x[1];
};

PetscScalar cubic_2d(const PetscReal x[]) {
  return x[0]*x[0]*x[0] - 1.5*x[0]*x[0] + x[1]*x[1]*x[1] - 1.5*x[1]*x[1] + 0.5;
};

PetscScalar nonlinear_3d(const PetscReal x[]) {
  return -4.0 - lambda*PetscExpScalar((2.0/3.0)*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]));
};

PetscScalar linear_3d(const PetscReal x[]) {
  return -6.0*(x[0] - 0.5) - 6.0*(x[1] - 0.5) - 6.0*(x[2] - 0.5);
};

PetscScalar quadratic_3d(const PetscReal x[]) {
  return (2.0/3.0)*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
};

PetscScalar cubic_3d(const PetscReal x[]) {
  return x[0]*x[0]*x[0] - 1.5*x[0]*x[0] + x[1]*x[1]*x[1] - 1.5*x[1]*x[1] + x[2]*x[2]*x[2] - 1.5*x[2]*x[2] + 0.75;
};

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  const char    *runTypes[3] = {"full", "test", "mesh"};
  const char    *bcTypes[2]  = {"neumann", "dirichlet"};
  PetscReal      bratu_lambda_max = 6.81, bratu_lambda_min = 0.0;
  PetscInt       run, bc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->run             = RUN_FULL;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;
  options->bcType          = DIRICHLET;
  options->lambda          = 6.0;
  options->rhsFunc         = zero;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMESH");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex12.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  run = options->run;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex12.c", runTypes, 3, runTypes[options->run], &run, PETSC_NULL);CHKERRQ(ierr);
  options->run = (RunType) run;
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex12.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex12.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex12.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
  bc = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex12.c",bcTypes,2,bcTypes[options->bcType],&bc,PETSC_NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  ierr = PetscOptionsReal("-lambda", "The parameter controlling nonlinearity", "ex12.c", options->lambda, &options->lambda, PETSC_NULL);CHKERRQ(ierr);
  if (options->lambda >= bratu_lambda_max || options->lambda < bratu_lambda_min) {
    SETERRQ3(PETSC_COMM_WORLD, 1, "Lambda, %g, is out of range, [%g, %g)", options->lambda, bratu_lambda_min, bratu_lambda_max);
  }
  ierr = PetscOptionsEnd();
  lambda = options->lambda;
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "SetupQuadrature"
PetscErrorCode SetupQuadrature(AppCtx *user) {
  PetscFunctionBegin;
  switch(user->dim) {
  case 1:
    user->q.numQuadPoints = NUM_QUADRATURE_POINTS_0;
    user->q.quadPoints    = points_0;
    user->q.quadWeights   = weights_0;
    user->q.numBasisFuncs = NUM_BASIS_FUNCTIONS_0;
    user->q.basis         = Basis_0;
    user->q.basisDer      = BasisDerivatives_0;
    break;
  case 2:
    user->q.numQuadPoints = NUM_QUADRATURE_POINTS_1;
    user->q.quadPoints    = points_1;
    user->q.quadWeights   = weights_1;
    user->q.numBasisFuncs = NUM_BASIS_FUNCTIONS_1;
    user->q.basis         = Basis_1;
    user->q.basisDer      = BasisDerivatives_1;
    break;
  case 3:
    user->q.numQuadPoints = NUM_QUADRATURE_POINTS_2;
    user->q.quadPoints    = points_2;
    user->q.quadWeights   = weights_2;
    user->q.numBasisFuncs = NUM_BASIS_FUNCTIONS_2;
    user->q.basis         = Basis_2;
    user->q.basisDer      = BasisDerivatives_2;
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
PetscErrorCode SetupSection(AppCtx *user) {
  PetscSection   section;
  /* These can be generated using config/PETSc/FEM.py */
  PetscInt       numDof_0[2] = {1, 0};
  PetscInt       numDof_1[3] = {1, 0, 0};
  PetscInt       numDof_2[4] = {1, 0, 0, 0};
  PetscInt      *numDof      = PETSC_NULL;
  PetscInt       numBC       = 0;
  PetscInt       bcField[1]  = {0};
  IS             bcPoints[1] = {PETSC_NULL};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(user->dim) {
  case 1:
    numDof = numDof_0;
    break;
  case 2:
    numDof = numDof_1;
    break;
  case 3:
    numDof = numDof_2;
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid spatial dimension %d", user->dim);
  }
  if (user->bcType == DIRICHLET) {
    numBC = 1;
    ierr  = DMMeshGetStratumIS(user->dm, "marker", 1, &bcPoints[0]);CHKERRQ(ierr);
  }
  ierr = DMMeshCreateSection(user->dm, user->dim, 1, PETSC_NULL, numDof, numBC, bcField, bcPoints, &section);CHKERRQ(ierr);
  ierr = DMMeshSetDefaultSection(user->dm, section);CHKERRQ(ierr);
  if (user->bcType == DIRICHLET) {
    ierr = ISDestroy(&bcPoints[0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupExactSolution"
PetscErrorCode SetupExactSolution(AppCtx *user) {
  PetscFunctionBegin;
  switch(user->dim) {
  case 2:
    if (user->bcType == DIRICHLET) {
      if (user->lambda > 0.0) {
        user->rhsFunc   = nonlinear_2d;
        user->exactFunc = quadratic_2d;
      } else {
        user->rhsFunc   = constant;
        user->exactFunc = quadratic_2d;
      }
    } else {
      user->rhsFunc   = linear_2d;
      user->exactFunc = cubic_2d;
    }
    break;
  case 3:
    if (user->bcType == DIRICHLET) {
      if (user->lambda > 0.0) {
        user->rhsFunc   = nonlinear_3d;
        user->exactFunc = quadratic_3d;
      } else {
        user->rhsFunc   = constant;
        user->exactFunc = quadratic_3d;
      }
    } else {
      user->rhsFunc   = linear_3d;
      user->exactFunc = cubic_3d;
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeError"
PetscErrorCode ComputeError(Vec X, PetscReal *error, AppCtx *user) {
  PetscScalar    (*exactFunc)(const PetscReal []) = user->exactFunc;
  const PetscInt   debug         = user->debug;
  const PetscInt   dim           = user->dim;
  const PetscInt   numQuadPoints = user->q.numQuadPoints;
  const PetscReal *quadPoints    = user->q.quadPoints;
  const PetscReal *quadWeights   = user->q.quadWeights;
  const PetscInt   numBasisFuncs = user->q.numBasisFuncs;
  const PetscReal *basis         = user->q.basis;
  Vec              localX;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscReal        localError;
  PetscInt         cStart, cEnd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(user->dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(user->dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;
    PetscReal          elemError = 0.0;

    ierr = DMMeshComputeCellGeometry(user->dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMMeshVecGetClosure(user->dm, localX, c, &x);CHKERRQ(ierr);
    if (debug) {ierr = DMPrintCellVector(c, "Solution", numBasisFuncs, x);CHKERRQ(ierr);}
    for(int q = 0; q < numQuadPoints; ++q) {
      for(int d = 0; d < dim; d++) {
        coords[d] = v0[d];
        for(int e = 0; e < dim; e++) {
          coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
        }
      }
      const PetscScalar funcVal     = (*exactFunc)(coords);
      PetscReal         interpolant = 0.0;
      for(int f = 0; f < numBasisFuncs; ++f) {
        interpolant += x[f]*basis[q*numBasisFuncs+f];
      }
      elemError += PetscSqr(interpolant - funcVal)*quadWeights[q]*detJ;
    }
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %d error %g\n", c, elemError);CHKERRQ(ierr);}
    localError += elemError;
  }
  ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &localX);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&localError, error, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
  *error = PetscSqrtReal(*error);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  Vec            u,r;                  /* solution, residual vectors */
  Mat            A,J;                  /* Jacobian matrix */
  AppCtx         user;                 /* user-defined work context */
  PetscInt       its;                  /* iterations for convergence */
  PetscReal      error;                /* L_2 error in the solution */
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create unstructured mesh (DMMESH) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMeshCreateBoxMesh(PETSC_COMM_WORLD, user.dim, user.interpolate, &user.dm);CHKERRQ(ierr);
  {
    DM refinedMesh     = PETSC_NULL;
    DM distributedMesh = PETSC_NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMMeshRefine(user.dm, user.refinementLimit, user.interpolate, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
      user.dm = refinedMesh;
    }
#if 0
    {
      ALE::Obj<PETSC_MESH_TYPE> oldMesh;
      ierr = DMMeshGetMesh(user.dm, oldMesh);CHKERRQ(ierr);
      oldMesh->setDebug(1);
    }
#endif
    /* Distribute mesh over processes */
    ierr = DMMeshDistribute(user.dm, user.partitioner, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
      user.dm = distributedMesh;
    }
    /* Mark boundary cells for higher order element calculations */
    if (user.bcType == DIRICHLET) {
      ierr = DMMeshMarkBoundaryCells(user.dm, "marker", 1, 2);CHKERRQ(ierr);
    }
  }
  ierr = DMSetFromOptions(user.dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, user.dm);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup dof layout.

     For a DMDA, this is automatic given the number of dof at each vertex.
     However, for a DMMesh, we need to specify this.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SetupExactSolution(&user);CHKERRQ(ierr);
  ierr = SetupQuadrature(&user);CHKERRQ(ierr);
  ierr = SetupSection(&user);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DM; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(user.dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine

     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner)
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* J can be type of MATAIJ, MATBAIJ or MATSBAIJ */
  ierr = DMCreateMatrix(user.dm, MATAIJ, &J);CHKERRQ(ierr);
  A    = J;
  ierr = SNESSetJacobian(snes, A, J, SNESDMMeshComputeJacobian, &user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local function evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMeshSetLocalFunction(user.dm, (DMMeshLocalFunction1) FormFunctionLocal);CHKERRQ(ierr);
  ierr = DMMeshSetLocalJacobian(user.dm, (DMMeshLocalJacobian1) FormJacobianLocal);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, SNESDMMeshComputeFunction, &user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup boundary conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialGuess(u, user.exactFunc, INSERT_ALL_VALUES, &user);CHKERRQ(ierr);
  if (user.run == RUN_FULL) {
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
     Note: The user should initialize the vector u, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = FormInitialGuess(u, guess, INSERT_VALUES, &user);CHKERRQ(ierr);
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = SNESSolve(snes, PETSC_NULL, u);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %D\n", its);CHKERRQ(ierr);
    ierr = ComputeError(u, &error, &user);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  } else {
    PetscReal res;

    /* Check discretization error */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ComputeError(u, &error, &user);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);
    /* Check residual */
    ierr = SNESDMMeshComputeFunction(snes, u, r, &user);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Output results
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (0) {
    PetscViewer viewer;

    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
    /*ierr = PetscViewerSetType(viewer, PETSCVIEWERDRAW);CHKERRQ(ierr);
      ierr = PetscViewerDrawSetInfo(viewer, PETSC_NULL, "Solution", PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr); */
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "ex12_sol.vtk");CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(user.dm, viewer);CHKERRQ(ierr);
    ierr = VecView(u, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/*
  FormInitialGuess - Forms initial approximation.

  Input Parameters:
+ user - user-defined application context
- guessFunc - The coordinate function to use for the guess

  Output Parameter:
. X - vector
*/
PetscErrorCode FormInitialGuess(Vec X, PetscScalar (*guessFunc)(const PetscReal []), InsertMode mode, AppCtx *user)
{
  Vec            localX, coordinates;
  PetscSection   section, cSection;
  PetscInt       vStart, vEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(user->dm, &localX);CHKERRQ(ierr);
  ierr = DMMeshGetDepthStratum(user->dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMMeshGetDefaultSection(user->dm, &section);CHKERRQ(ierr);
  ierr = DMMeshGetCoordinateSection(user->dm, &cSection);CHKERRQ(ierr);
  ierr = DMMeshGetCoordinateVec(user->dm, &coordinates);CHKERRQ(ierr);
  for(PetscInt v = vStart; v < vEnd; ++v) {
    PetscScalar  values[1];
    PetscScalar *coords;

    ierr = VecGetValuesSection(coordinates, cSection, v, &coords);CHKERRQ(ierr);
    values[0] = (*guessFunc)(coords);
    ierr = VecSetValuesSection(localX, section, v, values, mode);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cSection);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(user->dm, localX, INSERT_VALUES, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm, localX, INSERT_VALUES, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &localX);CHKERRQ(ierr);
#if 0
  /* This is necessary for higher order elements */
  ierr = MeshGetSectionReal(mesh, "exactSolution", &this->_options.exactSol.section);CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE::real_section_type>& s = this->_mesh->getRealSection("exactSolution");
  this->_mesh->setupField(s);
  const Obj<PETSC_MESH_TYPE::label_sequence>&     cells       = this->_mesh->heightStratum(0);
  const Obj<PETSC_MESH_TYPE::real_section_type>&  coordinates = this->_mesh->getRealSection("coordinates");
  const int                                       localDof    = this->_mesh->sizeWithBC(s, *cells->begin());
  PETSC_MESH_TYPE::real_section_type::value_type *values      = new PETSC_MESH_TYPE::real_section_type::value_type[localDof];
  PetscReal                                      *v0          = new PetscReal[dim()];
  PetscReal                                      *J           = new PetscReal[dim()*dim()];
  PetscReal                                       detJ;
  ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(this->_mesh->getSieve()->getMaxConeSize(), this->_mesh->depth())+1, true);

  for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    ALE::ISieveTraversal<PETSC_MESH_TYPE::sieve_type>::orientedClosure(*this->_mesh->getSieve(), *c_iter, pV);
    const PETSC_MESH_TYPE::point_type *oPoints = pV.getPoints();
    const int                          oSize   = pV.getSize();
    int                                v       = 0;

    this->_mesh->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
    for(int cl = 0; cl < oSize; ++cl) {
      const int pointDim = s->getFiberDimension(oPoints[cl]);

      if (pointDim) {
        for(int d = 0; d < pointDim; ++d, ++v) {
          values[v] = (*this->_options.integrate)(v0, J, v, initFunc);
        }
      }
    }
    this->_mesh->update(s, *c_iter, values);
    pV.clear();
  }
  delete [] values;
  delete [] v0;
  delete [] J;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x).
*/
PetscErrorCode FormFunctionLocal(DM dm, Vec X, Vec F, AppCtx *user)
{
  PetscScalar    (*rhsFunc)(const PetscReal []) = user->rhsFunc;
  const PetscInt   debug         = user->debug;
  const PetscInt   dim           = user->dim;
  const PetscInt   numQuadPoints = user->q.numQuadPoints;
  const PetscReal *quadPoints    = user->q.quadPoints;
  const PetscReal *quadWeights   = user->q.quadWeights;
  const PetscInt   numBasisFuncs = user->q.numBasisFuncs;
  const PetscReal *basis         = user->q.basis;
  const PetscReal *basisDer      = user->q.basisDer;
  const PetscReal  lambda        = user->lambda;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscScalar     *realSpaceDer, *fieldGrad, *elemVec;
  PetscInt         cStart, cEnd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscScalar,&realSpaceDer,dim,PetscScalar,&fieldGrad,numBasisFuncs,PetscScalar,&elemVec);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(user->dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;

    ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = DMMeshComputeCellGeometry(user->dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMMeshVecGetClosure(user->dm, X, c, &x);CHKERRQ(ierr);
    if (debug) {ierr = DMPrintCellVector(c, "Solution", numBasisFuncs, x);CHKERRQ(ierr);}

    for(int q = 0; q < numQuadPoints; ++q) {
      PetscScalar fieldVal = 0.0;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      for(int d = 0; d < dim; ++d) {
        fieldGrad[d] = 0.0;
        coords[d] = v0[d];
        for(int e = 0; e < dim; ++e) {
          coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
        }
        if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    coords[%d] %g\n", d, coords[d]);CHKERRQ(ierr);}
      }
      for(int f = 0; f < numBasisFuncs; ++f) {
        fieldVal += x[f]*basis[q*numBasisFuncs+f];

        for(int d = 0; d < dim; ++d) {
          realSpaceDer[d] = 0.0;
          for(int e = 0; e < dim; ++e) {
            realSpaceDer[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
          }
          fieldGrad[d] += realSpaceDer[d]*x[f];
        }
      }
      if (debug) {
        for(int d = 0; d < dim; ++d) {
          PetscPrintf(PETSC_COMM_SELF, "    fieldGrad[%d] %g\n", d, fieldGrad[d]);
        }
      }
      const PetscScalar funcVal = (*rhsFunc)(coords);
      for(int f = 0; f < numBasisFuncs; ++f) {
        /* Constant term: -f(x) */
        elemVec[f] -= basis[q*numBasisFuncs+f]*funcVal*quadWeights[q]*detJ;
        /* Linear term: -\Delta u */
        PetscScalar product = 0.0;
        for(int d = 0; d < dim; ++d) {
          realSpaceDer[d] = 0.0;
          for(int e = 0; e < dim; ++e) {
            realSpaceDer[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
          }
          product += realSpaceDer[d]*fieldGrad[d];
        }
        elemVec[f] += product*quadWeights[q]*detJ;
        /* Nonlinear term: -\lambda e^{u} */
        elemVec[f] -= basis[q*numBasisFuncs+f]*lambda*PetscExpScalar(fieldVal)*quadWeights[q]*detJ;
      }
    }
    if (debug) {ierr = DMPrintCellVector(c, "Residual", numBasisFuncs, elemVec);CHKERRQ(ierr);}
    ierr = DMMeshVecSetClosure(user->dm, F, c, elemVec, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops((cEnd-cStart)*numQuadPoints*numBasisFuncs*(dim*(dim*5+4)+14));CHKERRQ(ierr);
  ierr = PetscFree3(realSpaceDer,fieldGrad,elemVec);CHKERRQ(ierr);
  ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual:\n");CHKERRQ(ierr);
  for(int p = 0; p < user->numProcs; ++p) {
    if (p == user->rank) {ierr = VecView(F, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    ierr = PetscBarrier((PetscObject) user->dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
   FormJacobianLocal - Evaluates Jacobian matrix.
*/
PetscErrorCode FormJacobianLocal(DM dm, Vec X, Mat Jac, AppCtx *user)
{
  const PetscInt   debug         = user->debug;
  const PetscInt   dim           = user->dim;
  const PetscInt   numQuadPoints = user->q.numQuadPoints;
  const PetscReal *quadWeights   = user->q.quadWeights;
  const PetscInt   numBasisFuncs = user->q.numBasisFuncs;
  const PetscReal *basis         = user->q.basis;
  const PetscReal *basisDer      = user->q.basisDer;
  const PetscReal  lambda        = user->lambda;
  PetscReal       *v0, *J, *invJ, detJ;
  PetscScalar     *realSpaceTestDer, *realSpaceBasisDer, *elemMat;
  PetscInt         cStart, cEnd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(Jac);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscScalar,&realSpaceTestDer,dim,PetscScalar,&realSpaceBasisDer,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(user->dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;

    ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = DMMeshComputeCellGeometry(user->dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMMeshVecGetClosure(user->dm, X, c, &x);CHKERRQ(ierr);

    for(int q = 0; q < numQuadPoints; ++q) {
      PetscScalar fieldVal = 0.0;

      for(int f = 0; f < numBasisFuncs; ++f) {
        fieldVal += x[f]*basis[q*numBasisFuncs+f];
      }
      for(int f = 0; f < numBasisFuncs; ++f) {
        for(int d = 0; d < dim; ++d) {
          realSpaceTestDer[d] = 0.0;
          for(int e = 0; e < dim; ++e) {
            realSpaceTestDer[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
          }
        }
        for(int g = 0; g < numBasisFuncs; ++g) {
          for(int d = 0; d < dim; ++d) {
            realSpaceBasisDer[d] = 0.0;
            for(int e = 0; e < dim; ++e) {
              realSpaceBasisDer[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
            }
          }
          /* Linear term: -\Delta u */
          PetscScalar product = 0.0;
          for(int d = 0; d < dim; ++d) product += realSpaceTestDer[d]*realSpaceBasisDer[d];
          elemMat[f*numBasisFuncs+g] += product*quadWeights[q]*detJ;
          /* Nonlinear term: -\lambda e^{u} */
          elemMat[f*numBasisFuncs+g] -= basis[q*numBasisFuncs+f]*basis[q*numBasisFuncs+g]*lambda*PetscExpScalar(fieldVal)*quadWeights[q]*detJ;
        }
      }
    }
    if (debug) {ierr = DMPrintCellMatrix(c, "Jacobian", numBasisFuncs, numBasisFuncs, elemMat);CHKERRQ(ierr);}
    ierr = DMMeshMatSetClosure(user->dm, Jac, c, elemMat, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops((cEnd-cStart)*numQuadPoints*numBasisFuncs*(dim*(dim*5+4)+14));CHKERRQ(ierr);
  ierr = PetscFree3(realSpaceTestDer,realSpaceBasisDer,elemMat);CHKERRQ(ierr);
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);

  /* Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd(). */
  ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error. */
  ierr = MatSetOption(Jac, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  if (user->bcType == NEUMANN) {
    MatNullSpace nullsp;

    ierr = MatNullSpaceCreate(((PetscObject) Jac)->comm, PETSC_TRUE, 0, PETSC_NULL, &nullsp);CHKERRQ(ierr);
    ierr = MatSetNullSpace(Jac, nullsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
