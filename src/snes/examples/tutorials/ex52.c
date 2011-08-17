static char help[] = "Testbed for FEM operations on the GPU\n\n";

#include<petscdmmesh.h>
#include<petscsnes.h>

#ifndef PETSC_HAVE_SIEVE
#error This example requires Sieve. Reconfigure using --with-sieve
#endif

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  char          partitioner[2048]; /* The graph partitioner */
  PetscBool     computeFunction;   /* The flag for computing a residual */
  PetscBool     computeJacobian;   /* The flag for computing a Jacobian */
  /* Element quadrature */
  PetscQuadrature q;
} AppCtx;

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

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;
  options->computeFunction = PETSC_FALSE;
  options->computeJacobian = PETSC_FALSE;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMESH");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex52.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex52.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex52.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex52.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compute_function", "Compute the residual", "ex52.c", options->computeFunction, &options->computeFunction, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compute_jacobian", "Compute the Jacobian", "ex52.c", options->computeJacobian, &options->computeJacobian, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

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
  ierr = DMMeshCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
  {
    DM refinedMesh     = PETSC_NULL;
    DM distributedMesh = PETSC_NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMMeshRefine(*dm, refinementLimit, interpolate, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    /* Distribute mesh over processes */
    ierr = DMMeshDistribute(*dm, partitioner, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

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
PetscErrorCode SetupSection(DM dm, AppCtx *user) {
  PetscSection   section;
  /* These can be generated using config/PETSc/FEM.py */
  PetscInt       numDof_0[2] = {1, 0};
  PetscInt       numDof_1[3] = {1, 0, 0};
  PetscInt       numDof_2[4] = {1, 0, 0, 0};
  PetscInt       dim         = user->dim;
  PetscInt      *numDof;
  const char    *bcLabel = PETSC_NULL;
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
  //if (user->bcType == DIRICHLET) {
  //  bcLabel = "marker";
  //}
  ierr = DMMeshCreateSection(dm, dim, numDof, bcLabel, 1, &section);CHKERRQ(ierr);
  ierr = DMMeshSetSection(dm, "default", section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
 dm - The mesh
 X  - Local intput vector
 F  - Local output vector
 */
PetscErrorCode FormFunctionLocal(DM dm, Vec X, Vec F, AppCtx *user)
{
  //PetscScalar    (*rhsFunc)(const PetscReal []) = user->rhsFunc;
  const PetscInt   debug         = user->debug;
  const PetscInt   dim           = user->dim;
  const PetscInt   numQuadPoints = user->q.numQuadPoints;
  const PetscReal *quadPoints    = user->q.quadPoints;
  const PetscReal *quadWeights   = user->q.quadWeights;
  const PetscInt   numBasisFuncs = user->q.numBasisFuncs;
  const PetscReal *basis         = user->q.basis;
  const PetscReal *basisDer      = user->q.basisDer;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscScalar     *realSpaceDer, *fieldGrad, *elemVec;
  PetscInt         cStart, cEnd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscScalar,&realSpaceDer,dim,PetscScalar,&fieldGrad,numBasisFuncs,PetscScalar,&elemVec);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;

    ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = DMMeshComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);}
    ierr = DMMeshVecGetClosure(dm, X, c, &x);CHKERRQ(ierr);
    if (debug) {ierr = DMMeshPrintCellVector(c, "Solution", numBasisFuncs, x);CHKERRQ(ierr);}

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
      const PetscScalar funcVal = 0.0; //(*rhsFunc)(coords);
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
        elemVec[f] -= basis[q*numBasisFuncs+f]*0.0*PetscExpScalar(fieldVal)*quadWeights[q]*detJ;
      }
    }
    if (debug) {ierr = DMMeshPrintCellVector(c, "Residual", numBasisFuncs, elemVec);CHKERRQ(ierr);}
    ierr = DMMeshVecSetClosure(dm, F, c, elemVec, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops((cEnd-cStart)*numQuadPoints*numBasisFuncs*(dim*(dim*5+4)+14));CHKERRQ(ierr);
  ierr = PetscFree3(realSpaceDer,fieldGrad,elemVec);CHKERRQ(ierr);
  ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual:\n");CHKERRQ(ierr);
  for(int p = 0; p < user->numProcs; ++p) {
    if (p == user->rank) {ierr = VecView(F, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
  dm  - The mesh
  X   - The local input vector
  Jac - The output matrix
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
  PetscReal       *v0, *J, *invJ, detJ;
  PetscScalar     *realSpaceTestDer, *realSpaceBasisDer, *elemMat;
  PetscInt         cStart, cEnd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(Jac);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscScalar,&realSpaceTestDer,dim,PetscScalar,&realSpaceBasisDer,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;

    ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = DMMeshComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);}
    ierr = DMMeshVecGetClosure(dm, X, c, &x);CHKERRQ(ierr);

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
          elemMat[f*numBasisFuncs+g] -= basis[q*numBasisFuncs+f]*basis[q*numBasisFuncs+g]*0.0*PetscExpScalar(fieldVal)*quadWeights[q]*detJ;
        }
      }
    }
    if (debug) {ierr = DMMeshPrintCellMatrix(c, "Jacobian", numBasisFuncs, numBasisFuncs, elemMat);CHKERRQ(ierr);}
    ierr = DMMeshMatSetClosure(dm, Jac, c, elemMat, ADD_VALUES);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  SNES           snes;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
#ifndef PETSC_HAVE_CUDA
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example requires CUDA support.");
#endif
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SetupQuadrature(&user);CHKERRQ(ierr);
  ierr = SetupSection(dm, &user);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  if (user.computeFunction) {
    Vec            X, F;

    ierr = DMGetGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm, &F);CHKERRQ(ierr);
    ierr = DMMeshSetLocalFunction(dm, (PetscErrorCode (*)(DM, Vec, Vec, void*)) FormFunctionLocal);CHKERRQ(ierr);
    ierr = SNESMeshFormFunction(snes, X, F, &user);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &F);CHKERRQ(ierr);
  }
  if (user.computeJacobian) {
    Vec            X;
    Mat            J;
    MatStructure   flag;

    ierr = DMGetGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = DMGetMatrix(dm, MATAIJ, &J);CHKERRQ(ierr);
    ierr = DMMeshSetLocalJacobian(dm, (PetscErrorCode (*)(DM, Vec, Mat, void*)) FormJacobianLocal);CHKERRQ(ierr);
    ierr = SNESMeshFormJacobian(snes, X, &J, &J, &flag, &user);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &X);CHKERRQ(ierr);
  }
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
