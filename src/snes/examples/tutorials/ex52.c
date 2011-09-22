static char help[] = "Testbed for FEM operations on the GPU\n\n";

#if 0
MUST CHECK WITH:
 - Automate quadrature and basis tabulation generation in device code
 - Add example for linear elasticity (require vector field, so f_1 is a tensor)
 - Redo setup/cleanup in PyLith
 - Test with examples/3d/hex8/step01.cfg (Brad is making a flag)
 - Code up Jacobian (looks like original code)
 - Consider taking in coordinates rather than J^{-T}/|J|

 - Different quadrature (1pt and 3pt)
 - Different basis (linear and quadratic)
 - Different number of blocks (have to change it in the CUDA, so need a #define)
#endif

#include<petscdmmesh.h>
#include<petscsnes.h>

#ifndef PETSC_HAVE_SIEVE
#error This example requires Sieve. Reconfigure using --with-sieve
#endif

typedef enum {LAPLACIAN, ELASTICITY} OpType;

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
  PetscBool     batch;             /* The flag for batch assembly */
  PetscBool     gpu;               /* The flag for GPU integration */
  OpType        op;                /* The type of PDE operator (should use FFC/Ignition here) */
  /* GPU partitioning */
  PetscInt      numBatches;        /* The number of cell batches per kernel */
  /* Element quadrature */
  PetscQuadrature q;
} AppCtx;

/*------------------------------------------------------------------------------
  This code can be generated using 'bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 1 src/snes/examples/tutorials/ex52.h'
 -----------------------------------------------------------------------------*/
//#include "ex52.h"
#include "ex52_elas.h"

void quadratic_2d(const PetscReal x[], PetscScalar u[]) {
  u[0] = x[0]*x[0] + x[1]*x[1];
};

void quadratic_2d_elas(const PetscReal x[], PetscScalar u[]) {
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = x[0]*x[0] + x[1]*x[1];
};

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  const char    *opTypes[2]  = {"laplacian", "elasticity"};
  PetscInt       op;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;
  options->computeFunction = PETSC_FALSE;
  options->computeJacobian = PETSC_FALSE;
  options->batch           = PETSC_FALSE;
  options->gpu             = PETSC_FALSE;
  options->numBatches      = 1;
  options->op              = LAPLACIAN;

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
  ierr = PetscOptionsBool("-batch", "Use the batch assembly method", "ex52.c", options->batch, &options->batch, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-gpu", "Use the GPU for integration method", "ex52.c", options->gpu, &options->gpu, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-gpu_batches", "The number of cell batches per kernel", "ex52.c", options->numBatches, &options->numBatches, PETSC_NULL);CHKERRQ(ierr);
  op = options->op;
  ierr = PetscOptionsEList("-op_type","Type of PDE operator","ex52.c",opTypes,2,opTypes[options->op],&op,PETSC_NULL);CHKERRQ(ierr);
  options->op = (OpType) op;
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
  case 2:
    user->q.numQuadPoints = NUM_QUADRATURE_POINTS_0;
    user->q.quadPoints    = points_0;
    user->q.quadWeights   = weights_0;
    user->q.numBasisFuncs = NUM_BASIS_FUNCTIONS_0;
    user->q.numComponents = NUM_BASIS_COMPONENTS_0;
    user->q.basis         = Basis_0;
    user->q.basisDer      = BasisDerivatives_0;
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
  PetscInt       dim              = user->dim;
  PetscInt       numDof_Lap_0[2]  = {1, 0};
  PetscInt       numDof_Lap_1[3]  = {1, 0, 0};
  PetscInt       numDof_Lap_2[4]  = {1, 0, 0, 0};
  PetscInt       numDof_Elas_0[2] = {dim, 0};
  PetscInt       numDof_Elas_1[3] = {dim, 0, 0};
  PetscInt       numDof_Elas_2[4] = {dim, 0, 0, 0};
  const char    *bcLabel          = PETSC_NULL;
  PetscInt      *numDof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(user->op) {
  case LAPLACIAN:
    switch(user->dim) {
    case 1:
      numDof = numDof_Lap_0;break;
    case 2:
      numDof = numDof_Lap_1;break;
    case 3:
      numDof = numDof_Lap_2;break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid spatial dimension %d", user->dim);
    }
    break;
  case ELASTICITY:
    switch(user->dim) {
    case 1:
      numDof = numDof_Elas_0;break;
    case 2:
      numDof = numDof_Elas_1;break;
    case 3:
      numDof = numDof_Elas_2;break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid spatial dimension %d", user->dim);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid PDE operator %d", user->op);
  }
  //if (user->bcType == DIRICHLET) {
  //  bcLabel = "marker";
  //}
  ierr = DMMeshCreateSection(dm, dim, numDof, bcLabel, 1, &section);CHKERRQ(ierr);
  ierr = DMMeshSetSection(dm, "default", section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
PetscErrorCode FormInitialGuess(Vec X, void (*guessFunc)(const PetscReal [], PetscScalar []), InsertMode mode, AppCtx *user)
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
    PetscScalar  values[3];
    PetscScalar *coords;

    ierr = VecGetValuesSection(coordinates, cSection, v, &coords);CHKERRQ(ierr);
    (*guessFunc)(coords, values);
    ierr = VecSetValuesSection(localX, section, v, values, mode);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cSection);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(user->dm, localX, INSERT_VALUES, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm, localX, INSERT_VALUES, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocalLaplacian"
/*
 dm - The mesh
 X  - Local intput vector
 F  - Local output vector
 */
PetscErrorCode FormFunctionLocalLaplacian(DM dm, Vec X, Vec F, AppCtx *user)
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
    if (debug > 1) {ierr = DMMeshPrintCellVector(c, "Solution", numBasisFuncs, x);CHKERRQ(ierr);}

    for(int q = 0; q < numQuadPoints; ++q) {
      PetscScalar fieldVal = 0.0;

      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      for(int d = 0; d < dim; ++d) {
        fieldGrad[d] = 0.0;
        coords[d] = v0[d];
        for(int e = 0; e < dim; ++e) {
          coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
        }
        if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "    coords[%d] %g\n", d, coords[d]);CHKERRQ(ierr);}
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
      if (debug > 1) {
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
#define __FUNCT__ "FormFunctionLocalElasticity"
/*
 dm - The mesh
 X  - Local intput vector
 F  - Local output vector
 */
PetscErrorCode FormFunctionLocalElasticity(DM dm, Vec X, Vec F, AppCtx *user)
{
  //PetscScalar    (*rhsFunc)(const PetscReal []) = user->rhsFunc;
  const PetscInt   debug         = user->debug;
  const PetscInt   dim           = user->dim;
  const PetscInt   numQuadPoints = user->q.numQuadPoints;
  const PetscReal *quadPoints    = user->q.quadPoints;
  const PetscReal *quadWeights   = user->q.quadWeights;
  const PetscInt   numBasisFuncs = user->q.numBasisFuncs;
  const PetscInt   numBasisComps = user->q.numComponents;
  const PetscReal *basis         = user->q.basis;
  const PetscReal *basisDer      = user->q.basisDer;
  PetscReal       *coords, *v0, *J, *invJ, detJ;
  PetscScalar     *realSpaceDer, *fieldGrad, *elemVec;
  PetscInt         cStart, cEnd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscScalar,&realSpaceDer,dim*numBasisComps,PetscScalar,&fieldGrad,numBasisFuncs*numBasisComps,PetscScalar,&elemVec);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J,dim*dim,PetscReal,&invJ);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;

    ierr = PetscMemzero(elemVec, numBasisFuncs*numBasisComps * sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = DMMeshComputeCellGeometry(dm, c, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);}
    ierr = DMMeshVecGetClosure(dm, X, c, &x);CHKERRQ(ierr);
    if (debug > 1) {ierr = DMMeshPrintCellVector(c, "Solution", numBasisFuncs*numBasisComps, x);CHKERRQ(ierr);}

    for(int q = 0; q < numQuadPoints; ++q) {
      PetscScalar fieldVal[3] = {0.0,0.0,0.0};

      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      for(int d = 0; d < dim; ++d) {
        for(int comp = 0; comp < numBasisComps; ++comp) {
          fieldGrad[comp*dim+d] = 0.0;
        }
        coords[d] = v0[d];
        for(int e = 0; e < dim; ++e) {
          coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
        }
        if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "    coords[%d] %g\n", d, coords[d]);CHKERRQ(ierr);}
      }
      for(int f = 0; f < numBasisFuncs; ++f) {
        for(int comp = 0; comp < numBasisComps; ++comp) {
          const PetscInt cidx = f*numBasisComps+comp;

          fieldVal[comp] += x[cidx]*basis[q*numBasisFuncs*numBasisComps+cidx];

          for(int d = 0; d < dim; ++d) {
            realSpaceDer[d] = 0.0;
            for(int e = 0; e < dim; ++e) {
              realSpaceDer[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs*numBasisComps+cidx)*dim+e];
            }
            fieldGrad[comp*dim+d] += realSpaceDer[d]*x[cidx];
          }
        }
      }
      if (debug > 1) {
        for(int comp = 0; comp < numBasisComps; ++comp) {
          for(int d = 0; d < dim; ++d) {
            PetscPrintf(PETSC_COMM_SELF, "    field %d gradient[%d] %g\n", comp, d, fieldGrad[comp*dim+d]);
          }
        }
      }
      const PetscScalar funcVal[3] = {0.0, 0.0, 0.0}; //(*rhsFunc)(coords);
      for(int f = 0; f < numBasisFuncs; ++f) {
        for(int comp = 0; comp < numBasisComps; ++comp) {
          const PetscInt cidx = f*numBasisComps+comp;

          /* Constant term: -f(x) */
          elemVec[cidx] -= basis[q*numBasisFuncs*numBasisComps+cidx]*funcVal[comp]*quadWeights[q]*detJ;
          /* Linear term: -\Delta u */
          PetscScalar product = 0.0;
          for(int d = 0; d < dim; ++d) {
            realSpaceDer[d] = 0.0;
            for(int e = 0; e < dim; ++e) {
              realSpaceDer[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs*numBasisComps+cidx)*dim+e];
            }
            product += realSpaceDer[d]*0.5*(fieldGrad[comp*dim+d] + fieldGrad[d*dim+comp]);
          }
          elemVec[cidx] += product*quadWeights[q]*detJ;
          /* Nonlinear term: -\lambda e^{u} */
          elemVec[cidx] -= basis[q*numBasisFuncs*numBasisComps+cidx]*0.0*PetscExpScalar(fieldVal[comp])*quadWeights[q]*detJ;
        }
      }
    }
    if (debug) {ierr = DMMeshPrintCellVector(c, "Residual", numBasisFuncs*numBasisComps, elemVec);CHKERRQ(ierr);}
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

extern PetscErrorCode IntegrateElementBatchGPU(PetscInt Ne, PetscInt Nbatch, PetscInt Nbc, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[], PetscInt debug);

void f1_laplacian(PetscScalar u, const PetscScalar gradU[], PetscScalar f1[]) {
  f1[0] = gradU[0];
  f1[1] = gradU[1];
  return;
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateLaplacianBatchCPU"
PetscErrorCode IntegrateLaplacianBatchCPU(PetscInt Ne, PetscInt Nb, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscInt Nq, const PetscReal quadPoints[], const PetscReal quadWeights[], const PetscReal basisTabulation[], const PetscReal basisDerTabulation[], PetscScalar elemVec[], AppCtx *user) {
  const PetscInt debug = user->debug;
  const PetscInt dim = 2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(PetscInt e = 0; e < Ne; ++e) {
    const PetscReal  detJ = jacobianDeterminants[e];
    const PetscReal *invJ = &jacobianInverses[e*dim*dim];
    PetscScalar      f1[1*dim]; // Nq = 1

    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "  invJ: %g %g %g %g\n", invJ[0], invJ[1], invJ[2], invJ[3]);CHKERRQ(ierr);
    }
    for(PetscInt q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      PetscScalar u          = 0.0;
      PetscScalar gradU[dim] = {0.0, 0.0};

      for(PetscInt b = 0; b < Nb; ++b) {
        PetscScalar realSpaceDer[dim];

        u += coefficients[e*Nb+b]*basisTabulation[q*Nb+b];
        for(PetscInt d = 0; d < dim; ++d) {
          realSpaceDer[d] = 0.0;
          for(PetscInt f = 0; f < dim; ++f) {
            realSpaceDer[d] += invJ[f*dim+d]*basisDerTabulation[(q*Nb+b)*dim+f];
          }
          gradU[d] += coefficients[e*Nb+b]*realSpaceDer[d];
        }
      }
      f1_laplacian(u, gradU, &f1[q*dim]);
      f1[q*dim+0] *= detJ*quadWeights[q];
      f1[q*dim+1] *= detJ*quadWeights[q];
      if (debug > 1) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]: %g\n", 0, f1[q*dim+0]);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]: %g\n", 1, f1[q*dim+1]);CHKERRQ(ierr);
      }
    }
    for(PetscInt b = 0; b < Nb; ++b) {
      elemVec[e*Nb+b] = 0.0;
      for(PetscInt q = 0; q < Nq; ++q) {
        PetscScalar realSpaceDer[dim];

        for(PetscInt d = 0; d < dim; ++d) {
          realSpaceDer[d] = 0.0;
          for(PetscInt f = 0; f < dim; ++f) {
            realSpaceDer[d] += invJ[f*dim+d]*basisDerTabulation[(q*Nb+b)*dim+f];
          }
          elemVec[e*Nb+b] += realSpaceDer[d]*f1[q*dim+d];
        }
      }
    }
  }
  PetscFunctionReturn(0);
};

void f1_elasticity(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]) {
  const int dim   = 2;
  const int Ncomp = dim;

  for(int comp = 0; comp < Ncomp; ++comp) {
    for(int d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 0.5*(gradU[comp*dim+d] + gradU[d*dim+comp]);
    }
  }
  return;
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateElasticityBatchCPU"
PetscErrorCode IntegrateElasticityBatchCPU(PetscInt Ne, PetscInt Nb, PetscInt Ncomp, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscInt Nq, const PetscReal quadPoints[], const PetscReal quadWeights[], const PetscReal basisTabulation[], const PetscReal basisDerTabulation[], PetscScalar elemVec[], AppCtx *user) {
  const PetscInt debug = user->debug;
  const PetscInt dim = 2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(PetscInt e = 0; e < Ne; ++e) {
    const PetscReal  detJ = jacobianDeterminants[e];
    const PetscReal *invJ = &jacobianInverses[e*dim*dim];
    PetscScalar      f1[1*dim*dim]; // Nq = 1

    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "  invJ: %g %g %g %g\n", invJ[0], invJ[1], invJ[2], invJ[3]);CHKERRQ(ierr);
    }
    for(PetscInt q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      PetscScalar u[dim]         = {0.0, 0.0};
      PetscScalar gradU[dim*dim] = {0.0, 0.0, 0.0, 0.0};

      for(PetscInt b = 0; b < Nb; ++b) {
        for(int comp = 0; comp < Ncomp; ++comp) {
          const PetscInt cidx = b*Ncomp+comp;
          PetscScalar    realSpaceDer[dim];

          u[comp] += coefficients[e*Nb*Ncomp+cidx]*basisTabulation[q*Nb*Ncomp+cidx];
          for(PetscInt d = 0; d < dim; ++d) {
            realSpaceDer[d] = 0.0;
            for(PetscInt f = 0; f < dim; ++f) {
              realSpaceDer[d] += invJ[f*dim+d]*basisDerTabulation[(q*Nb*Ncomp+cidx)*dim+f];
            }
            gradU[comp*dim+d] += coefficients[e*Nb*Ncomp+cidx]*realSpaceDer[d];
          }
        }
      }
      f1_elasticity(u, gradU, &f1[q*Ncomp*dim]);
      for(int i = 0; i < Ncomp*dim; ++i) {
        f1[q*Ncomp*dim+i] *= detJ*quadWeights[q];
      }
      if (debug > 1) {
        for(int i = 0; i < Ncomp*dim; ++i) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]: %g\n", i, f1[q*Ncomp*dim+i]);CHKERRQ(ierr);
        }
      }
    }
    for(PetscInt b = 0; b < Nb; ++b) {
      for(int comp = 0; comp < Ncomp; ++comp) {
        const PetscInt cidx = b*Ncomp+comp;

        elemVec[e*Nb*Ncomp+cidx] = 0.0;
        for(PetscInt q = 0; q < Nq; ++q) {
          PetscScalar realSpaceDer[dim];

          for(PetscInt d = 0; d < dim; ++d) {
            realSpaceDer[d] = 0.0;
            for(PetscInt f = 0; f < dim; ++f) {
              realSpaceDer[d] += invJ[f*dim+d]*basisDerTabulation[(q*Nb*Ncomp+cidx)*dim+f];
            }
            elemVec[e*Nb*Ncomp+cidx] += realSpaceDer[d]*f1[(q*Ncomp+comp)*dim+d];
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocalBatch"
/*
 dm - The mesh
 X  - Local intput vector
 F  - Local output vector
 */
PetscErrorCode FormFunctionLocalBatch(DM dm, Vec X, Vec F, AppCtx *user)
{
  //PetscScalar    (*rhsFunc)(const PetscReal []) = user->rhsFunc;
  const PetscInt   debug         = user->debug;
  const PetscInt   dim           = user->dim;
  const PetscInt   numQuadPoints = user->q.numQuadPoints;
  const PetscReal *quadPoints    = user->q.quadPoints;
  const PetscReal *quadWeights   = user->q.quadWeights;
  const PetscInt   numBasisFuncs = user->q.numBasisFuncs;
  const PetscInt   numBasisComps = user->q.numComponents;
  const PetscReal *basis         = user->q.basis;
  const PetscReal *basisDer      = user->q.basisDer;
  PetscReal       *coords, *v0, *J, *invJ, *detJ;
  PetscScalar     *elemVec;
  PetscInt         cStart, cEnd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  //ierr = PetscMalloc3(dim,PetscScalar,&realSpaceDer,dim,PetscScalar,&fieldGrad,numBasisFuncs,PetscScalar,&elemVec);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  const PetscInt numCells = cEnd - cStart;
  PetscScalar   *u;

  ierr = PetscMalloc4(numCells*numBasisFuncs*numBasisComps,PetscScalar,&u,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*numBasisFuncs*numBasisComps,PetscScalar,&elemVec);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;

    ierr = DMMeshComputeCellGeometry(dm, c, v0, J, &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    ierr = DMMeshVecGetClosure(dm, X, c, &x);CHKERRQ(ierr);

    for(int f = 0; f < numBasisFuncs*numBasisComps; ++f) {
      u[c*numBasisFuncs*numBasisComps+f] = x[f];
    }
  }
  // Conforming batches
  PetscInt blockSize  = numBasisFuncs*numQuadPoints;
  PetscInt numBlocks  = 1;
  PetscInt batchSize  = numBlocks * blockSize;
  PetscInt numBatches = user->numBatches;
  PetscInt numChunks  = numCells / (numBatches*batchSize);
  if (user->gpu) {
    ierr = IntegrateElementBatchGPU(numChunks*numBatches*batchSize, numBatches, batchSize, u, invJ, detJ, elemVec, user->debug);CHKERRQ(ierr);
  } else {
    switch(user->op) {
    case LAPLACIAN:
      ierr = IntegrateLaplacianBatchCPU(numChunks*numBatches*batchSize, numBasisFuncs, u, invJ, detJ, numQuadPoints, quadPoints, quadWeights, basis, basisDer, elemVec, user);CHKERRQ(ierr);break;
    case ELASTICITY:
      ierr = IntegrateElasticityBatchCPU(numChunks*numBatches*batchSize, numBasisFuncs, numBasisComps, u, invJ, detJ, numQuadPoints, quadPoints, quadWeights, basis, basisDer, elemVec, user);CHKERRQ(ierr);break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid PDE operator %d", user->op);
    }
  }
  // Remainder
  PetscInt numRemainder = numCells % (numBatches * batchSize);
  PetscInt offset       = numCells - numRemainder;
  switch(user->op) {
  case LAPLACIAN:
    ierr = IntegrateLaplacianBatchCPU(numRemainder, numBasisFuncs, &u[offset*numBasisFuncs*numBasisComps], &invJ[offset*dim*dim], &detJ[offset],
                                      numQuadPoints, quadPoints, quadWeights, basis, basisDer, &elemVec[offset*numBasisFuncs*numBasisComps], user);CHKERRQ(ierr);break;
  case ELASTICITY:
    ierr = IntegrateElasticityBatchCPU(numRemainder, numBasisFuncs, numBasisComps, &u[offset*numBasisFuncs*numBasisComps], &invJ[offset*dim*dim], &detJ[offset],
                                       numQuadPoints, quadPoints, quadWeights, basis, basisDer, &elemVec[offset*numBasisFuncs*numBasisComps], user);CHKERRQ(ierr);break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid PDE operator %d", user->op);
  }
  for(PetscInt c = cStart; c < cEnd; ++c) {
    if (debug) {ierr = DMMeshPrintCellVector(c, "Residual", numBasisFuncs*numBasisComps, &elemVec[c*numBasisFuncs*numBasisComps]);CHKERRQ(ierr);}
    ierr = DMMeshVecSetClosure(dm, F, c, &elemVec[c*numBasisFuncs*numBasisComps], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree4(u,invJ,detJ,elemVec);CHKERRQ(ierr);

  ierr = PetscLogFlops(0);CHKERRQ(ierr);
  //ierr = PetscFree3(realSpaceDer,fieldGrad,elemVec);CHKERRQ(ierr);
  ierr = PetscFree3(coords,v0,J);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual:\n");CHKERRQ(ierr);
  for(int p = 0; p < user->numProcs; ++p) {
    if (p == user->rank) {ierr = VecView(F, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocalLaplacian"
/*
  dm  - The mesh
  X   - The local input vector
  Jac - The output matrix
*/
PetscErrorCode FormJacobianLocalLaplacian(DM dm, Vec X, Mat Jac, AppCtx *user)
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
#define __FUNCT__ "FormJacobianLocalElasticity"
/*
  dm  - The mesh
  X   - The local input vector
  Jac - The output matrix
*/
PetscErrorCode FormJacobianLocalElasticity(DM dm, Vec X, Mat Jac, AppCtx *user)
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
#define __FUNCT__ "FormJacobianLocalBatch"
/*
  dm  - The mesh
  X   - The local input vector
  Jac - The output matrix
*/
PetscErrorCode FormJacobianLocalBatch(DM dm, Vec X, Mat Jac, AppCtx *user)
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
    if (user.batch) {
      switch(user.op) {
      case LAPLACIAN:
        ierr = FormInitialGuess(X, quadratic_2d, INSERT_VALUES, &user);CHKERRQ(ierr);break;
      case ELASTICITY:
        ierr = FormInitialGuess(X, quadratic_2d_elas, INSERT_VALUES, &user);CHKERRQ(ierr);break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid PDE operator %d", user.op);
      }
      ierr = DMMeshSetLocalFunction(dm, (PetscErrorCode (*)(DM, Vec, Vec, void*)) FormFunctionLocalBatch);CHKERRQ(ierr);
    } else {
      switch(user.op) {
      case LAPLACIAN:
        ierr = FormInitialGuess(X, quadratic_2d, INSERT_VALUES, &user);CHKERRQ(ierr);
        ierr = DMMeshSetLocalFunction(dm, (PetscErrorCode (*)(DM, Vec, Vec, void*)) FormFunctionLocalLaplacian);CHKERRQ(ierr);break;
      case ELASTICITY:
        ierr = FormInitialGuess(X, quadratic_2d_elas, INSERT_VALUES, &user);CHKERRQ(ierr);
        ierr = DMMeshSetLocalFunction(dm, (PetscErrorCode (*)(DM, Vec, Vec, void*)) FormFunctionLocalElasticity);CHKERRQ(ierr);break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid PDE operator %d", user.op);
      }
    }
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
    if (user.batch) {
      ierr = DMMeshSetLocalJacobian(dm, (PetscErrorCode (*)(DM, Vec, Mat, void*)) FormJacobianLocalBatch);CHKERRQ(ierr);
    } else {
      switch(user.op) {
      case LAPLACIAN:
        ierr = DMMeshSetLocalJacobian(dm, (PetscErrorCode (*)(DM, Vec, Mat, void*)) FormJacobianLocalLaplacian);CHKERRQ(ierr);break;
      case ELASTICITY:
        ierr = DMMeshSetLocalJacobian(dm, (PetscErrorCode (*)(DM, Vec, Mat, void*)) FormJacobianLocalElasticity);CHKERRQ(ierr);break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid PDE operator %d", user.op);
      }
    }
    ierr = SNESMeshFormJacobian(snes, X, &J, &J, &flag, &user);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &X);CHKERRQ(ierr);
  }
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
