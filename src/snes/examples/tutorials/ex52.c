static const char help[] = "Testbed for FEM operations on the GPU.\n\n";

/*
#if 0
 - Automate testing of different:
   - elements, batches, blocks, operators
 - Process log_summary with Python (use existing tools)
 - Put results in PyTables

PYLITH:
 - Redo setup/cleanup in PyLith
 - Test with examples/3d/hex8/step01.cfg (Brad is making a flag)
 - Code up Jacobian (looks like original code)
 - Consider taking in coordinates rather than J^{-T}/|J|

MUST CHECK WITH:
 - 3D elements
 - Different quadrature (1pt and 3pt)
 - Different basis (linear and quadratic)
#endif
*/

#include<petscdmplex.h>
#include<petscsnes.h>

#ifdef PETSC_HAVE_GENERATOR
/*------------------------------------------------------------------------------
  This code can be generated using 'bin/pythonscripts/PetscGenerateFEMQuadrature.py dim 1 numComp 1 laplacian src/snes/examples/tutorials/ex52.h'
 -----------------------------------------------------------------------------*/
#include "ex52.h"
#endif

#define NUM_FIELDS 1
PetscInt spatialDim = 0;

typedef enum {LAPLACIAN = 0, ELASTICITY} OpType;

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscFEM      fem;               /* REQUIRED to use DMPlexComputeResidualFEM() */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  PetscBool     refinementUniform; /* Uniformly refine the mesh */
  PetscInt      refinementRounds;  /* The number of uniform refinements */
  char          partitioner[2048]; /* The graph partitioner */
  PetscBool     computeFunction;   /* The flag for computing a residual */
  PetscBool     computeJacobian;   /* The flag for computing a Jacobian */
  PetscBool     gpu;               /* The flag for GPU integration */
  OpType        op;                /* The type of PDE operator (should use FFC/Ignition here) */
  PetscBool     showResidual, showJacobian;
  PetscLogEvent createMeshEvent, residualEvent, residualBatchEvent, jacobianEvent, jacobianBatchEvent, integrateBatchCPUEvent, integrateBatchGPUEvent, integrateGPUOnlyEvent;
  /* GPU partitioning */
  PetscInt      numBatches;        /* The number of cell batches per kernel */
  PetscInt      numBlocks;         /* The number of concurrent blocks per kernel */
  /* Element definition */
  PetscFE       fe[NUM_FIELDS];
  void (*f0Funcs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[]);
  void (*f1Funcs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[]);
  void (*g0Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[]);
  void (*g1Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g1[]);
  void (*g2Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[]);
  void (*g3Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[]);
  void (**exactFuncs)(const PetscReal x[], PetscScalar *u);
} AppCtx;

void quadratic_2d(const PetscReal x[], PetscScalar u[])
{
  u[0] = x[0]*x[0] + x[1]*x[1];
};

void quadratic_2d_elas(const PetscReal x[], PetscScalar u[])
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = x[0]*x[0] + x[1]*x[1];
};

void f0_lap(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 4.0;
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_lap(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) {f1[d] = gradU[d];}
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_lap(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) {g3[d*spatialDim+d] = 1.0;}
}

void f0_elas(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])
{
  const PetscInt Ncomp = spatialDim;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) f0[comp] = 3.0;
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
   u[Ncomp]          = {p} */
void f1_elas(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  const PetscInt dim   = spatialDim;
  const PetscInt Ncomp = spatialDim;
  PetscInt       comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 0.5*(gradU[comp*dim+d] + gradU[d*dim+comp]);
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_elas(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  const PetscInt dim   = spatialDim;
  const PetscInt Ncomp = spatialDim;
  PetscInt       compI, d;

  for (compI = 0; compI < Ncomp; ++compI) {
    for (d = 0; d < dim; ++d) {
      g3[((compI*Ncomp+compI)*dim+d)*dim+d] = 1.0;
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char     *opTypes[2] = {"laplacian", "elasticity"};
  PetscInt       op;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug             = 0;
  options->dim               = 2;
  options->interpolate       = PETSC_FALSE;
  options->refinementLimit   = 0.0;
  options->refinementUniform = PETSC_FALSE;
  options->refinementRounds  = 1;
  options->computeFunction   = PETSC_FALSE;
  options->computeJacobian   = PETSC_FALSE;
  options->batch             = PETSC_FALSE;
  options->gpu               = PETSC_FALSE;
  options->numBatches        = 1;
  options->numBlocks         = 1;
  options->op                = LAPLACIAN;
  options->showResidual      = PETSC_TRUE;
  options->showJacobian      = PETSC_TRUE;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex52.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex52.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex52.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex52.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-refinement_uniform", "Uniformly refine the mesh", "ex52.c", options->refinementUniform, &options->refinementUniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-refinement_rounds", "The number of uniform refinements", "ex52.c", options->refinementRounds, &options->refinementRounds, NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "ex52.c", options->partitioner, options->partitioner, 2048, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compute_function", "Compute the residual", "ex52.c", options->computeFunction, &options->computeFunction, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compute_jacobian", "Compute the Jacobian", "ex52.c", options->computeJacobian, &options->computeJacobian, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-gpu", "Use the GPU for integration method", "ex52.c", options->gpu, &options->gpu, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-gpu_batches", "The number of cell batches per kernel", "ex52.c", options->numBatches, &options->numBatches, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-gpu_blocks", "The number of concurrent blocks per kernel", "ex52.c", options->numBlocks, &options->numBlocks, NULL);CHKERRQ(ierr);

  op = options->op;

  ierr = PetscOptionsEList("-op_type","Type of PDE operator","ex52.c",opTypes,2,opTypes[options->op],&op,NULL);CHKERRQ(ierr);

  options->op = (OpType) op;

  ierr = PetscOptionsBool("-show_residual", "Output the residual for verification", "ex52.c", options->showResidual, &options->showResidual, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_jacobian", "Output the Jacobian for verification", "ex52.c", options->showJacobian, &options->showJacobian, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",    DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Residual",      SNES_CLASSID, &options->residualEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ResidualBatch", SNES_CLASSID, &options->residualBatchEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Jacobian",      SNES_CLASSID, &options->jacobianEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("JacobianBatch", SNES_CLASSID, &options->jacobianBatchEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IntegBatchCPU", SNES_CLASSID, &options->integrateBatchCPUEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IntegBatchGPU", SNES_CLASSID, &options->integrateBatchGPUEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IntegGPUOnly",  SNES_CLASSID, &options->integrateGPUOnlyEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim               = user->dim;
  PetscBool      interpolate       = user->interpolate;
  PetscReal      refinementLimit   = user->refinementLimit;
  PetscBool      refinementUniform = user->refinementUniform;
  PetscInt       refinementRounds  = user->refinementRounds;
  const char     *partitioner      = user->partitioner;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, partitioner, 0, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
    /* Use regular refinement in parallel */
    if (refinementUniform) {
      PetscInt r;

      ierr = DMPlexSetRefinementUniform(*dm, refinementUniform);CHKERRQ(ierr);
      for (r = 0; r < refinementRounds; ++r) {
        ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
        if (refinedMesh) {
          ierr = DMDestroy(dm);CHKERRQ(ierr);
          *dm  = refinedMesh;
        }
      }
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);

  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupElement"
PetscErrorCode SetupElement(DM dm, AppCtx *user)
{
  const PetscInt  dim = user->dim;
  PetscFE         fem;
  PetscQuadrature q;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        order;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create space */
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P);CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(P, &order);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, PETSC_TRUE, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), &fem);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(fem);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(fem, 1);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature */
  ierr = PetscDTGaussJacobiQuadrature(dim, order, -1.0, 1.0, &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fem, q);CHKERRQ(ierr);
  user->fe[0] = fem;
  user->fem.fe = user->fe;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyElement"
PetscErrorCode DestroyElement(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFEDestroy(&user->fe[0]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
PetscErrorCode SetupSection(DM dm, AppCtx *user)
{
  PetscSection    section;
  PetscInt        dim   = user->dim;
  PetscInt        numBC = 0;
  PetscInt        numComp[1];
  const PetscInt *numDof;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscFEGetNumComponents(user->fe[0], &numComp[0]);CHKERRQ(ierr);
  ierr = PetscFEGetNumDof(user->fe[0], &numDof);CHKERRQ(ierr);
  ierr = DMPlexCreateSection(dm, dim, 1, numComp, numDof, numBC, NULL, NULL, &section);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
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
  PetscInt       vStart, vEnd, v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(user->dm, &localX);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(user->dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(user->dm, &section);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(user->dm, &cSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(user->dm, &coordinates);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar values[3];
    PetscScalar *coords;

    ierr = VecGetValuesSection(coordinates, cSection, v, &coords);CHKERRQ(ierr);
    (*guessFunc)(coords, values);
    ierr = VecSetValuesSection(localX, section, v, values, mode);CHKERRQ(ierr);
  }

  ierr = DMLocalToGlobalBegin(user->dm, localX, INSERT_VALUES, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm, localX, INSERT_VALUES, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode IntegrateElementBatchGPU(PetscInt Ne, PetscInt Nbatch, PetscInt Nbc, PetscInt Nbl, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[], PetscLogEvent event, PetscInt debug);

void f1_laplacian(PetscScalar u, const PetscScalar gradU[], PetscScalar f1[])
{
  const PetscInt dim = spatialDim;
  PetscInt       d;

  for (d = 0; d < dim; ++d) f1[d] = gradU[d];
  return;
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateLaplacianBatchCPU"
PetscErrorCode IntegrateLaplacianBatchCPU(PetscInt Ne, PetscInt Nb, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscInt Nq, const PetscReal quadPoints[], const PetscReal quadWeights[], const PetscReal basisTabulation[], const PetscReal basisDerTabulation[], PetscScalar elemVec[], AppCtx *user)
{
  const PetscInt debug = user->debug;
  const PetscInt dim   = spatialDim;
  PetscInt       e;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->integrateBatchCPUEvent,0,0,0,0);CHKERRQ(ierr);
  for (e = 0; e < Ne; ++e) {
    const PetscReal detJ  = jacobianDeterminants[e];
    const PetscReal *invJ = &jacobianInverses[e*dim*dim];
    PetscScalar     f1[spatialDim /*1*/]; /* Nq = 1 */
    PetscInt        q, b, d, f;

    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "  invJ: %g %g %g %g\n", invJ[0], invJ[1], invJ[2], invJ[3]);CHKERRQ(ierr);
    }
    for (q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      PetscScalar u = 0.0;
      PetscScalar gradU[spatialDim];

      for (d = 0; d < dim; ++d) gradU[d] = 0.0;
      for (b = 0; b < Nb; ++b) {
        PetscScalar realSpaceDer[spatialDim];

        u += coefficients[e*Nb+b]*basisTabulation[q*Nb+b];
        for (d = 0; d < dim; ++d) {
          realSpaceDer[d] = 0.0;
          for (f = 0; f < dim; ++f) {
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
    for (b = 0; b < Nb; ++b) {
      elemVec[e*Nb+b] = 0.0;
      for (q = 0; q < Nq; ++q) {
        PetscScalar realSpaceDer[spatialDim];

        for (d = 0; d < dim; ++d) {
          realSpaceDer[d] = 0.0;
          for (f = 0; f < dim; ++f) {
            realSpaceDer[d] += invJ[f*dim+d]*basisDerTabulation[(q*Nb+b)*dim+f];
          }
          elemVec[e*Nb+b] += realSpaceDer[d]*f1[q*dim+d];
        }
      }
    }
  }
  ierr = PetscLogFlops((((2+(2+2*dim)*dim)*Nb+2*dim)*Nq + (2+2*dim)*dim*Nq*Nb)*Ne);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->integrateBatchCPUEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

void f1_elasticity(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[])
{
  const PetscInt dim   = spatialDim;
  const PetscInt Ncomp = dim;
  PetscInt       comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 0.5*(gradU[comp*dim+d] + gradU[d*dim+comp]);
    }
  }
  return;
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateElasticityBatchCPU"
PetscErrorCode IntegrateElasticityBatchCPU(PetscInt Ne, PetscInt Nb, PetscInt Ncomp, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscInt Nq, const PetscReal quadPoints[], const PetscReal quadWeights[], const PetscReal basisTabulation[], const PetscReal basisDerTabulation[], PetscScalar elemVec[], AppCtx *user)
{
  const PetscInt debug = user->debug;
  const PetscInt dim   = spatialDim;
  PetscInt       e;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->integrateBatchCPUEvent,0,0,0,0);CHKERRQ(ierr);
  for (e = 0; e < Ne; ++e) {
    const PetscReal detJ  = jacobianDeterminants[e];
    const PetscReal *invJ = &jacobianInverses[e*dim*dim];
    PetscScalar     f1[spatialDim*spatialDim /*1*/]; /* Nq = 1 */
    PetscInt        q, b, comp, i;

    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "  invJ:");CHKERRQ(ierr);
      for (i = 0; i < dim*dim; ++i) {
        ierr = PetscPrintf(PETSC_COMM_SELF, " %g", invJ[i]);CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
    }
    for (q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      PetscScalar u[spatialDim];
      PetscScalar gradU[spatialDim*spatialDim];

      for (i = 0; i < dim; ++i) u[i] = 0.0;
      for (i = 0; i < dim*dim; ++i) gradU[i] = 0.0;
      for (b = 0; b < Nb; ++b) {
        for (comp = 0; comp < Ncomp; ++comp) {
          const PetscInt cidx = b*Ncomp+comp;
          PetscScalar    realSpaceDer[spatialDim];
          PetscInt       d, f;

          u[comp] += coefficients[e*Nb*Ncomp+cidx]*basisTabulation[q*Nb*Ncomp+cidx];
          for (d = 0; d < dim; ++d) {
            realSpaceDer[d] = 0.0;
            for (f = 0; f < dim; ++f) {
              realSpaceDer[d] += invJ[f*dim+d]*basisDerTabulation[(q*Nb*Ncomp+cidx)*dim+f];
            }
            gradU[comp*dim+d] += coefficients[e*Nb*Ncomp+cidx]*realSpaceDer[d];
          }
        }
      }
      f1_elasticity(u, gradU, &f1[q*Ncomp*dim]);
      for (i = 0; i < Ncomp*dim; ++i) {
        f1[q*Ncomp*dim+i] *= detJ*quadWeights[q];
      }
      if (debug > 1) {
        PetscInt c, d;
        for (c = 0; c < Ncomp; ++c) {
          for (d = 0; d < dim; ++d) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d]_%c: %g f1[%d]_%c: %g\n", c, 'x'+d, gradU[c*dim+d], c, 'x'+d, f1[(q*Ncomp + c)*dim+d]);CHKERRQ(ierr);
          }
        }
      }
    }
    for (b = 0; b < Nb; ++b) {
      for (comp = 0; comp < Ncomp; ++comp) {
        const PetscInt cidx = b*Ncomp+comp;
        PetscInt       q, d, f;

        elemVec[e*Nb*Ncomp+cidx] = 0.0;
        for (q = 0; q < Nq; ++q) {
          PetscScalar realSpaceDer[spatialDim];

          for (d = 0; d < dim; ++d) {
            realSpaceDer[d] = 0.0;
            for (f = 0; f < dim; ++f) {
              realSpaceDer[d] += invJ[f*dim+d]*basisDerTabulation[(q*Nb*Ncomp+cidx)*dim+f];
            }
            elemVec[e*Nb*Ncomp+cidx] += realSpaceDer[d]*f1[(q*Ncomp+comp)*dim+d];
          }
        }
      }
    }
  }
  ierr = PetscLogFlops((((2+(2+2*dim)*dim)*Ncomp*Nb+(2+2)*dim*Ncomp)*Nq + (2+2*dim)*dim*Nq*Ncomp*Nb)*Ne);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->integrateBatchCPUEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#if 0
PetscErrorCode FormFunctionLocalBatch(DM dm, Vec X, Vec F, AppCtx *user)
{
  if (user->gpu) {
    ierr = PetscLogEventBegin(user->integrateBatchGPUEvent,0,0,0,0);CHKERRQ(ierr);
    ierr = IntegrateElementBatchGPU(dim, numChunks*numBatches*batchSize, numBatches, batchSize, numBlocks, u, invJ, detJ, elemVec, user->integrateGPUOnlyEvent, user->debug, user->op);CHKERRQ(ierr);
    ierr = PetscLogFlops((((2+(2+2*dim)*dim)*numBasisComps*numBasisFuncs+(2+2)*dim*numBasisComps)*numQuadPoints + (2+2*dim)*dim*numQuadPoints*numBasisComps*numBasisFuncs)*numChunks*numBatches*batchSize);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(user->integrateBatchGPUEvent,0,0,0,0);CHKERRQ(ierr);
  } else {
    switch (user->op) {
    case LAPLACIAN:
      ierr = IntegrateLaplacianBatchCPU(numChunks*numBatches*batchSize, numBasisFuncs, u, invJ, detJ, numQuadPoints, quadPoints, quadWeights, basis, basisDer, elemVec, user);CHKERRQ(ierr);break;
    case ELASTICITY:
      ierr = IntegrateElasticityBatchCPU(numChunks*numBatches*batchSize, numBasisFuncs, numBasisComps, u, invJ, detJ, numQuadPoints, quadPoints, quadWeights, basis, basisDer, elemVec, user);CHKERRQ(ierr);break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid PDE operator %d", user->op);
    }
  }
}
#endif

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  SNES           snes;
  AppCtx         user;
  PetscInt       numComp;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_CUDA) && !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example requires CUDA or OpenCL support.");
#endif
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);

  ierr = SetupElement(user.dm, &user);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(user.fe[0], &numComp);CHKERRQ(ierr);
  ierr = PetscMalloc(numComp * sizeof(void (*)(const PetscReal[], PetscScalar *)), &user.exactFuncs);CHKERRQ(ierr);
  switch (user.op) {
  case LAPLACIAN:
    user.f0Funcs[0]    = f0_lap;
    user.f1Funcs[0]    = f1_lap;
    user.g0Funcs[0]    = NULL;
    user.g1Funcs[0]    = NULL;
    user.g2Funcs[0]    = NULL;
    user.g3Funcs[0]    = g3_lap;
    user.exactFuncs[0] = quadratic_2d;
    break;
  case ELASTICITY:
    user.f0Funcs[0]    = f0_elas;
    user.f1Funcs[0]    = f1_elas;
    user.g0Funcs[0]    = NULL;
    user.g1Funcs[0]    = NULL;
    user.g2Funcs[0]    = NULL;
    user.g3Funcs[0]    = g3_elas;
    user.exactFuncs[0] = quadratic_2d_elas;
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid PDE operator %d", user.op);
  }
  user.fem.f0Funcs = user.f0Funcs;
  user.fem.f1Funcs = user.f1Funcs;
  user.fem.g0Funcs = user.g0Funcs;
  user.fem.g1Funcs = user.g1Funcs;
  user.fem.g2Funcs = user.g2Funcs;
  user.fem.g3Funcs = user.g3Funcs;
  user.fem.bcFuncs = (void (**)(const PetscReal[], PetscScalar *)) user.exactFuncs;
  ierr = SetupSection(dm, &user);CHKERRQ(ierr);

  ierr = DMSNESSetFunctionLocal(user.dm,  (PetscErrorCode (*)(DM,Vec,Vec,void*))DMPlexComputeResidualFEM,&user);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(user.dm,  (PetscErrorCode (*)(DM,Vec,Mat,Mat,MatStructure*,void*))DMPlexComputeJacobianFEM,&user);CHKERRQ(ierr);
  if (user.computeFunction) {
    Vec X, F;

    ierr = DMGetGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm, &F);CHKERRQ(ierr);
    ierr = FormInitialGuess(X, user.exactFuncs[0], INSERT_VALUES, &user);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &F);CHKERRQ(ierr);
  }
  if (user.computeJacobian) {
    Vec          X;
    Mat          J;
    MatStructure flag;

    ierr = DMGetGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm, MATAIJ, &J);CHKERRQ(ierr);
    ierr = SNESComputeJacobian(snes, X, &J, &J, &flag);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &X);CHKERRQ(ierr);
  }
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);
  ierr = DestroyElement(&user);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
