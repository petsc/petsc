static char help[] = "Stokes Problem in 2d and 3d with hexahedral finite elements.\n\
We solve the Stokes problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMMESH) to discretize it.\n\
The command line options include:\n\
  -visc_model <name>, the viscosity model\n\n\n";

/*
 The variable-viscosity Stokes problem, which we discretize using the finite
element method on an unstructured mesh. The weak form equations are

  < \nabla v, \nabla u + {\nabla u}^T > - < \nabla\cdot v, p > + < v, f > = 0
  < q, \nabla\cdot v >                                                    = 0

We start with homogeneous Dirichlet conditions.

This is an extension of ex56. Please refer to that example for further documentation.
*/

#include <petscdmmesh.h>
#include <petscdmda.h>
#include <petscsnes.h>

#ifndef PETSC_HAVE_SIEVE
#error This example requires Sieve. Reconfigure using --with-sieve
#endif

/*------------------------------------------------------------------------------
  This code can be generated using 'bin/pythonscripts/PetscGenerateFEMQuadratureTensorProduct.py dim order dim 1 laplacian dim order 1 1 gradient src/snes/examples/tutorials/ex56.h'
 -----------------------------------------------------------------------------*/
#include "ex57.h"

const int numFields     = 2;
const int numComponents = NUM_BASIS_COMPONENTS_0+NUM_BASIS_COMPONENTS_1;

typedef enum {NEUMANN, DIRICHLET} BCType;
typedef enum {RUN_FULL, RUN_TEST} RunType;

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  RunType       runType;           /* Whether to run tests, or solve the full problem */
  PetscLogEvent createMeshEvent, residualEvent, jacobianEvent, integrateResCPUEvent, integrateJacCPUEvent;
  PetscBool     showInitial, showResidual, showJacobian;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscBool     redistribute;      /* Redistribute mesh using a partitioner */
  char          partitioner[2048]; /* The graph partitioner */
  /* Element quadrature */
  PetscQuadrature q[numFields];
  /* GPU partitioning */
  PetscInt      numBatches;        /* The number of cell batches per kernel */
  PetscInt      numBlocks;         /* The number of concurrent blocks per kernel */
  /* Problem definition */
  void        (*f0Funcs[numFields])(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]); /* The f_0 functions f0_u(x,y,z), and f0_p(x,y,z) */
  void        (*f1Funcs[numFields])(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]); /* The f_1 functions f1_u(x,y,z), and f1_p(x,y,z) */
  void        (*g0Funcs[numFields*numFields])(PetscScalar u[], const PetscScalar gradU[], PetscScalar g0[]); /* The g_0 functions g0_uu(x,y,z), g0_up(x,y,z), g0_pu(x,y,z), and g0_pp(x,y,z) */
  void        (*g1Funcs[numFields*numFields])(PetscScalar u[], const PetscScalar gradU[], PetscScalar g1[]); /* The g_1 functions g1_uu(x,y,z), g1_up(x,y,z), g1_pu(x,y,z), and g1_pp(x,y,z) */
  void        (*g2Funcs[numFields*numFields])(PetscScalar u[], const PetscScalar gradU[], PetscScalar g2[]); /* The g_2 functions g2_uu(x,y,z), g2_up(x,y,z), g2_pu(x,y,z), and g2_pp(x,y,z) */
  void        (*g3Funcs[numFields*numFields])(PetscScalar u[], const PetscScalar gradU[], PetscScalar g3[]); /* The g_3 functions g3_uu(x,y,z), g3_up(x,y,z), g3_pu(x,y,z), and g3_pp(x,y,z) */
  PetscScalar (*exactFuncs[numComponents])(const PetscReal x[]);                                   /* The exact solution function u(x,y,z), v(x,y,z), and p(x,y,z) */
  BCType        bcType;            /* The type of boundary conditions */
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
};

PetscScalar quadratic_v_2d(const PetscReal x[]) {
  return 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
};

PetscScalar linear_p_2d(const PetscReal x[]) {
  return x[0] + x[1] - 1.0;
};

void f0_u(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]) {
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;

  for(PetscInt comp = 0; comp < Ncomp; ++comp) {
    f0[comp] = 3.0;
  }
}

// gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
// u[Ncomp]          = {p}
void f1_u(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]) {
  const PetscInt dim   = SPATIAL_DIM_0;
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;

  for(PetscInt comp = 0; comp < Ncomp; ++comp) {
    for(PetscInt d = 0; d < dim; ++d) {
      //f1[comp*dim+d] = 0.5*(gradU[comp*dim+d] + gradU[d*dim+comp]);
      f1[comp*dim+d] = gradU[comp*dim+d];
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

// gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
void f0_p(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]) {
  const PetscInt dim = SPATIAL_DIM_0;

  f0[0] = 0.0;
  for(PetscInt d = 0; d < dim; ++d) {
    f0[0] += gradU[d*dim+d];
  }
}

void f1_p(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]) {
  const PetscInt dim = SPATIAL_DIM_0;

  for(PetscInt d = 0; d < dim; ++d) {
    f1[d] = 0.0;
  }
}

// < q, \nabla\cdot v >
// NcompI = 1, NcompJ = dim
void g1_pu(PetscScalar u[], const PetscScalar gradU[], PetscScalar g1[]) {
  const PetscInt dim = SPATIAL_DIM_0;

  for(PetscInt d = 0; d < dim; ++d) {
    g1[d*dim+d] = 1.0; // \frac{\partial\phi^{u_d}}{\partial x_d}
  }
}

// -< \nabla\cdot v, p >
// NcompI = dim, NcompJ = 1
void g2_up(PetscScalar u[], const PetscScalar gradU[], PetscScalar g2[]) {
  const PetscInt dim = SPATIAL_DIM_0;

  for(PetscInt d = 0; d < dim; ++d) {
    g2[d*dim+d] = -1.0; // \frac{\partial\psi^{u_d}}{\partial x_d}
  }
}

// < \nabla v, \nabla u + {\nabla u}^T >
// This just gives \nabla u, give the perdiagonal for the transpose
void g3_uu(PetscScalar u[], const PetscScalar gradU[], PetscScalar g3[]) {
  const PetscInt dim   = SPATIAL_DIM_0;
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;

  for(PetscInt compI = 0; compI < Ncomp; ++compI) {
    for(PetscInt d = 0; d < dim; ++d) {
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
};

PetscScalar quadratic_v_3d(const PetscReal x[]) {
  return x[1]*x[1] + x[2]*x[2];
};

PetscScalar quadratic_w_3d(const PetscReal x[]) {
  return x[0]*x[0] + x[1]*x[1] - 2.0*(x[0] + x[1])*x[2];
};

PetscScalar linear_p_3d(const PetscReal x[]) {
  return x[0] + x[1] + x[2] - 1.5;
};

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
  options->redistribute    = PETSC_FALSE;
  options->bcType          = DIRICHLET;
  options->numBatches      = 1;
  options->numBlocks       = 1;
  options->showResidual    = PETSC_FALSE;
  options->showResidual    = PETSC_FALSE;
  options->showJacobian    = PETSC_FALSE;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMESH");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex56.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  run = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex56.c", runTypes, 2, runTypes[options->runType], &run, PETSC_NULL);CHKERRQ(ierr);
  options->runType = (RunType) run;
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex56.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex56.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
  bc = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex56.c",bcTypes,2,bcTypes[options->bcType],&bc,PETSC_NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  ierr = PetscOptionsInt("-gpu_batches", "The number of cell batches per kernel", "ex56.c", options->numBatches, &options->numBatches, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-gpu_blocks", "The number of concurrent blocks per kernel", "ex56.c", options->numBlocks, &options->numBlocks, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex56.c", options->showInitial, &options->showInitial, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_residual", "Output the residual for verification", "ex56.c", options->showResidual, &options->showResidual, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_jacobian", "Output the Jacobian for verification", "ex56.c", options->showJacobian, &options->showJacobian, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",       DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Residual",         SNES_CLASSID, &options->residualEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IntegResBatchCPU", SNES_CLASSID, &options->integrateResCPUEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IntegJacBatchCPU", SNES_CLASSID, &options->integrateJacCPUEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Jacobian",         SNES_CLASSID, &options->jacobianEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             da;
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  const char    *partitioner     = user->partitioner;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDACreate(comm, &da);CHKERRQ(ierr);
  ierr = DMDASetDim(da, dim);CHKERRQ(ierr);
  ierr = DMDASetSizes(da, -3, -3, -3);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(da, 1);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da, DMDA_STENCIL_STAR);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da, 1);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(da, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMConvert(da, DMMESH, dm);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  if (user->redistribute) {
    DM distributedMesh = PETSC_NULL;

    /* Distribute mesh over processes */
    ierr = DMMeshDistribute(*dm, partitioner, &distributedMesh);CHKERRQ(ierr);
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
  PetscInt       dim = user->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  user->q[0].numQuadPoints = NUM_QUADRATURE_POINTS_0;
  user->q[0].quadPoints    = points_0;
  user->q[0].quadWeights   = weights_0;
  user->q[0].numBasisFuncs = NUM_BASIS_FUNCTIONS_0;
  user->q[0].numComponents = NUM_BASIS_COMPONENTS_0;
  user->q[0].basis         = Basis_0;
  user->q[0].basisDer      = BasisDerivatives_0;
  user->q[1].numQuadPoints = NUM_QUADRATURE_POINTS_1;
  user->q[1].quadPoints    = points_1;
  user->q[1].quadWeights   = weights_1;
  user->q[1].numBasisFuncs = NUM_BASIS_FUNCTIONS_1;
  user->q[1].numComponents = NUM_BASIS_COMPONENTS_1;
  user->q[1].basis         = Basis_1;
  user->q[1].basisDer      = BasisDerivatives_1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
PetscErrorCode SetupSection(DM dm, AppCtx *user) {
  PetscSection   section;
  PetscInt       dim                = user->dim;
  PetscInt       numBC              = 0;
  PetscInt       numComp[numFields] = {NUM_BASIS_COMPONENTS_0, NUM_BASIS_COMPONENTS_1};
  PetscInt       bcFields[1]        = {0};
  IS             bcPoints[1]        = {PETSC_NULL};
  PetscInt       numDof[numFields*(SPATIAL_DIM_0+1)];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dim != SPATIAL_DIM_0) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Spatial dimension %d should be %d", dim, SPATIAL_DIM_0);
  if (dim != SPATIAL_DIM_1) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Spatial dimension %d should be %d", dim, SPATIAL_DIM_1);
  for(PetscInt d = 0; d <= dim; ++d) {
    numDof[0*(dim+1)+d] = numDof_0[d];
    numDof[1*(dim+1)+d] = numDof_1[d];
  }
  for(PetscInt f = 0; f < numFields; ++f) {
    for(PetscInt d = 1; d < dim; ++d) {
      if ((numDof[f*(dim+1)+d] > 0) && !user->interpolate) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  if (user->bcType == DIRICHLET) {
    numBC = 1;
    ierr  = DMMeshGetStratumIS(dm, "marker", 1, &bcPoints[0]);CHKERRQ(ierr);
  }
  ierr = DMMeshCreateSection(dm, dim, numFields, numComp, numDof, numBC, bcFields, bcPoints, &section);CHKERRQ(ierr);
  ierr = DMMeshSetDefaultSection(dm, section);CHKERRQ(ierr);
  if (user->bcType == DIRICHLET) {
    ierr = ISDestroy(&bcPoints[0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupExactSolution"
PetscErrorCode SetupExactSolution(AppCtx *user) {
  PetscFunctionBegin;
  user->f0Funcs[0] = f0_u;
  user->f0Funcs[1] = f0_p;
  user->f1Funcs[0] = f1_u;
  user->f1Funcs[1] = f1_p;
  user->g0Funcs[0] = PETSC_NULL;
  user->g0Funcs[1] = PETSC_NULL;
  user->g0Funcs[2] = PETSC_NULL;
  user->g0Funcs[3] = PETSC_NULL;
  user->g1Funcs[0] = PETSC_NULL;
  user->g1Funcs[1] = PETSC_NULL;
  user->g1Funcs[2] = g1_pu;      // < q, \nabla\cdot v >
  user->g1Funcs[3] = PETSC_NULL;
  user->g2Funcs[0] = PETSC_NULL;
  user->g2Funcs[1] = g2_up;      // < \nabla\cdot v, p >
  user->g2Funcs[2] = PETSC_NULL;
  user->g2Funcs[3] = PETSC_NULL;
  user->g3Funcs[0] = g3_uu;      // < \nabla v, \nabla u + {\nabla u}^T >
  user->g3Funcs[1] = PETSC_NULL;
  user->g3Funcs[2] = PETSC_NULL;
  user->g3Funcs[3] = PETSC_NULL;
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeError"
PetscErrorCode ComputeError(Vec X, PetscReal *error, AppCtx *user) {
  PetscScalar   (**exactFuncs)(const PetscReal []) = user->exactFuncs;
  const PetscInt   debug         = user->debug;
  const PetscInt   dim           = user->dim;
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

    for(PetscInt field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      const PetscInt   numQuadPoints = user->q[field].numQuadPoints;
      const PetscReal *quadPoints    = user->q[field].quadPoints;
      const PetscReal *quadWeights   = user->q[field].quadWeights;
      const PetscInt   numBasisFuncs = user->q[field].numBasisFuncs;
      const PetscInt   numBasisComps = user->q[field].numComponents;
      const PetscReal *basis         = user->q[field].basis;

      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, numBasisFuncs*numBasisComps, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for(PetscInt q = 0; q < numQuadPoints; ++q) {
        for(PetscInt d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for(PetscInt e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        for(PetscInt fc = 0; fc < numBasisComps; ++fc) {
          const PetscScalar funcVal     = (*exactFuncs[comp+fc])(coords);
          PetscReal         interpolant = 0.0;
          for(int f = 0; f < numBasisFuncs; ++f) {
            const PetscInt fidx = f*numBasisComps+fc;
            interpolant += x[fieldOffset+fidx]*basis[q*numBasisFuncs*numBasisComps+fidx];
          }
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d field %d error %g\n", c, field, PetscSqr(interpolant - funcVal)*quadWeights[q]*detJ);CHKERRQ(ierr);}
          elemError += PetscSqr(interpolant - funcVal)*quadWeights[q]*detJ;
        }
      }
      comp        += numBasisComps;
      fieldOffset += numBasisFuncs*numBasisComps;
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
#define __FUNCT__ "DMComputeVertexFunction"
/*
  DMComputeVertexFunction - This calls a function with the coordinates of each vertex, and stores the result in a vector.

  Input Parameters:
+ dm - The DM
. mode - The insertion mode for values
. numComp - The number of components (functions)
- func - The coordinate functions to evaluate

  Output Parameter:
. X - vector
*/
PetscErrorCode DMComputeVertexFunction(DM dm, InsertMode mode, Vec X, PetscInt numComp, PetscScalar (**funcs)(const PetscReal []), AppCtx *user)
{
  Vec            localX, coordinates;
  PetscSection   section, cSection;
  PetscInt       vStart, vEnd;
  PetscScalar   *values;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMMeshGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMMeshGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMMeshGetCoordinateSection(dm, &cSection);CHKERRQ(ierr);
  ierr = DMMeshGetCoordinateVec(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscMalloc(numComp * sizeof(PetscScalar), &values);CHKERRQ(ierr);
  for(PetscInt v = vStart; v < vEnd; ++v) {
    PetscScalar *coords;

    ierr = VecGetValuesSection(coordinates, cSection, v, &coords);CHKERRQ(ierr);
    for(PetscInt c = 0; c < numComp; ++c) {
      values[c] = (*funcs[c])(coords);
    }
    ierr = VecSetValuesSection(localX, section, v, values, mode);CHKERRQ(ierr);
  }
  // Temporary bullshit
  {
    ALE::Obj<PETSC_MESH_TYPE> mesh;
    PetscScalar *coordsE;
    PetscInt     eStart = 0, eEnd = 0, dim;

    ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(cSection, vStart, &dim);CHKERRQ(ierr);
    if (mesh->depth() > 1) {ierr = DMMeshGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);}
    ierr = PetscMalloc(dim * sizeof(PetscScalar),&coordsE);CHKERRQ(ierr);
    ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> pV((int) pow(mesh->getSieve()->getMaxConeSize(), dim+1)+1, true);

    for(PetscInt e = eStart; e < eEnd; ++e) {
      mesh->getSieve()->cone(e, pV);
      const PetscInt *points = pV.getPoints();
      PetscScalar    *coordsA, *coordsB;

      if (pV.getSize() != 2) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Cone size %d for point %d should be 2", pV.getSize(), e);}
      ierr = VecGetValuesSection(coordinates, cSection, points[0], &coordsA);CHKERRQ(ierr);
      ierr = VecGetValuesSection(coordinates, cSection, points[1], &coordsB);CHKERRQ(ierr);
      for(PetscInt d = 0; d < dim; ++d) {
        coordsE[d] = 0.5*(coordsA[d] + coordsB[d]);
      }
      for(PetscInt c = 0; c < numComp; ++c) {
        values[c] = (*funcs[c])(coordsE);
      }
      ierr = VecSetValuesSection(localX, section, e, values, mode);CHKERRQ(ierr);
      pV.clear();
    }
    ierr = PetscFree(coordsE);CHKERRQ(ierr);
  }

  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cSection);CHKERRQ(ierr);
  if (user->showInitial) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Local function\n");CHKERRQ(ierr);
    for(int p = 0; p < user->numProcs; ++p) {
      if (p == user->rank) {ierr = VecView(localX, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  ierr = DMLocalToGlobalBegin(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localX, mode, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePressureNullSpace"
PetscErrorCode CreatePressureNullSpace(DM dm, AppCtx *user, MatNullSpace *nullSpace) {
  Vec            pressure, localP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dm, &pressure);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localP);CHKERRQ(ierr);
  ierr = VecSet(pressure, 0.0);CHKERRQ(ierr);
  // Put a constant in for all pressures
  // Could change this to project the constant function onto the pressure space (when that is finished)
  {
    PetscSection section;
    PetscInt     pStart, pEnd, p;
    PetscScalar *a;

    ierr = DMMeshGetDefaultSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = VecGetArray(localP, &a);CHKERRQ(ierr);
    for(p = pStart; p < pEnd; ++p) {
      PetscInt fDim, off, d;

      ierr = PetscSectionGetFieldDof(section, p, 1, &fDim);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(section, p, 1, &off);CHKERRQ(ierr);
      for(d = 0; d < fDim; ++d) {
        a[off+d] = 1.0;
      }
    }
    ierr = VecRestoreArray(localP, &a);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobalBegin(dm, localP, INSERT_VALUES, pressure);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localP, INSERT_VALUES, pressure);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localP);CHKERRQ(ierr);
  if (user->debug) {
    ierr = PetscPrintf(((PetscObject) dm)->comm, "Pressure Null Space\n");CHKERRQ(ierr);
    ierr = VecView(pressure, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceCreate(((PetscObject) dm)->comm, PETSC_FALSE, 1, &pressure, nullSpace);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &pressure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateResidualBatchCPU"
PetscErrorCode IntegrateResidualBatchCPU(PetscInt Ne, PetscInt numFields, PetscInt field, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscQuadrature quad[], void (*f0_func)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]), void (*f1_func)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]), PetscScalar elemVec[], AppCtx *user) {
  const PetscInt debug   = user->debug;
  const PetscInt dim     = SPATIAL_DIM_0;
  PetscInt       cOffset = 0;
  PetscInt       eOffset = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->integrateResCPUEvent,0,0,0,0);CHKERRQ(ierr);
  for(PetscInt e = 0; e < Ne; ++e) {
    const PetscReal  detJ = jacobianDeterminants[e];
    const PetscReal *invJ = &jacobianInverses[e*dim*dim];
    const PetscInt   Nq   = quad[field].numQuadPoints;
    PetscScalar      f0[NUM_QUADRATURE_POINTS_0*dim];
    PetscScalar      f1[NUM_QUADRATURE_POINTS_0*dim*dim];

    assert(Nq <= NUM_QUADRATURE_POINTS_0);
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
    }
    for(PetscInt q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      PetscScalar u[dim+1];
      PetscScalar gradU[dim*(dim+1)];
      PetscInt    fOffset            = 0;
      PetscInt    dOffset            = cOffset;

      for(PetscInt d = 0; d <= dim; ++d)        {u[d]     = 0.0;}
      for(PetscInt d = 0; d < dim*(dim+1); ++d) {gradU[d] = 0.0;}
      for(PetscInt f = 0; f < numFields; ++f) {
        const PetscInt   Nb       = quad[f].numBasisFuncs;
        const PetscInt   Ncomp    = quad[f].numComponents;
        const PetscReal *basis    = quad[f].basis;
        const PetscReal *basisDer = quad[f].basisDer;

        for(PetscInt b = 0; b < Nb; ++b) {
          for(int comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscScalar    realSpaceDer[dim];

            u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
            for(PetscInt d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for(PetscInt g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              gradU[(fOffset+comp)*dim+d] += coefficients[dOffset+cidx]*realSpaceDer[d];
            }
          }
        }
        if (debug > 1) {
          for(int comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, u[fOffset+comp]);CHKERRQ(ierr);
            for(PetscInt d = 0; d < dim; ++d) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, gradU[(fOffset+comp)*dim+d]);CHKERRQ(ierr);
            }
          }
        }
        fOffset += Ncomp;
        dOffset += Nb*Ncomp;
      }
      const PetscInt   Ncomp       = quad[field].numComponents;
      const PetscReal *quadWeights = quad[field].quadWeights;

      f0_func(u, gradU, &f0[q*Ncomp]);
      for(int i = 0; i < Ncomp; ++i) {
        f0[q*Ncomp+i] *= detJ*quadWeights[q];
      }
      f1_func(u, gradU, &f1[q*Ncomp*dim]);
      for(int i = 0; i < Ncomp*dim; ++i) {
        f1[q*Ncomp*dim+i] *= detJ*quadWeights[q];
      }
      if (debug > 1) {
        for(int c = 0; c < Ncomp; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "    f0[%d]: %g\n", c, f0[q*Ncomp+c]);CHKERRQ(ierr);
          for(int d = 0; d < dim; ++d) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]_%c: %g\n", c, 'x'+d, f1[(q*Ncomp + c)*dim+d]);CHKERRQ(ierr);
          }
        }
      }
      if (q == Nq-1) {cOffset = dOffset;}
    }
    for(PetscInt f = 0; f < numFields; ++f) {
      const PetscInt   Nq       = quad[f].numQuadPoints;
      const PetscInt   Nb       = quad[f].numBasisFuncs;
      const PetscInt   Ncomp    = quad[f].numComponents;
      const PetscReal *basis    = quad[f].basis;
      const PetscReal *basisDer = quad[f].basisDer;

      if (f == field) {
      for(PetscInt b = 0; b < Nb; ++b) {
        for(int comp = 0; comp < Ncomp; ++comp) {
          const PetscInt cidx = b*Ncomp+comp;

          elemVec[eOffset+cidx] = 0.0;
          for(PetscInt q = 0; q < Nq; ++q) {
            PetscScalar realSpaceDer[dim];

            elemVec[eOffset+cidx] += basis[q*Nb*Ncomp+cidx]*f0[q*Ncomp+comp];
            for(PetscInt d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for(PetscInt g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              elemVec[eOffset+cidx] += realSpaceDer[d]*f1[(q*Ncomp+comp)*dim+d];
            }
          }
        }
      }
      if (debug > 1) {
        for(PetscInt b = 0; b < Nb; ++b) {
          for(int comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    elemVec[%d,%d]: %g\n", b, comp, elemVec[eOffset+b*Ncomp+comp]);CHKERRQ(ierr);
          }
        }
      }
      }
      eOffset += Nb*Ncomp;
    }
  }
  //ierr = PetscLogFlops((((2+(2+2*dim)*dim)*Ncomp*Nb+(2+2)*dim*Ncomp)*Nq + (2+2*dim)*dim*Nq*Ncomp*Nb)*Ne);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->integrateResCPUEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
  FormFunctionLocal - Form the local residual F from the local input X

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. F  - Local output vector

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: FormJacobianLocal()
*/
PetscErrorCode FormFunctionLocal(DM dm, Vec X, Vec F, AppCtx *user)
{
  const PetscInt   debug = user->debug;
  const PetscInt   dim   = user->dim;
  PetscReal       *coords, *v0, *J, *invJ, *detJ;
  PetscScalar     *elemVec;
  PetscInt         cStart, cEnd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->residualEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscReal,&coords,dim,PetscReal,&v0,dim*dim,PetscReal,&J);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  const PetscInt numCells = cEnd - cStart;
  PetscInt       cellDof  = 0;
  PetscScalar   *u;

  for(PetscInt field = 0; field < numFields; ++field) {
    cellDof += user->q[field].numBasisFuncs*user->q[field].numComponents;
  }
  ierr = PetscMalloc4(numCells*cellDof,PetscScalar,&u,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;

    ierr = DMMeshComputeCellGeometry(dm, c, v0, J, &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);}
    ierr = DMMeshVecGetClosure(dm, X, c, &x);CHKERRQ(ierr);

    for(PetscInt i = 0; i < cellDof; ++i) {
      u[c*cellDof+i] = x[i];
    }
  }
  for(PetscInt field = 0; field < numFields; ++field) {
    const PetscInt numQuadPoints = user->q[field].numQuadPoints;
    const PetscInt numBasisFuncs = user->q[field].numBasisFuncs;
    void (*f0)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]) = user->f0Funcs[field];
    void (*f1)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]) = user->f1Funcs[field];
    // Conforming batches
    PetscInt blockSize  = numBasisFuncs*numQuadPoints;
    PetscInt numBlocks  = 1;
    PetscInt batchSize  = numBlocks * blockSize;
    PetscInt numBatches = user->numBatches;
    PetscInt numChunks  = numCells / (numBatches*batchSize);
    ierr = IntegrateResidualBatchCPU(numChunks*numBatches*batchSize, numFields, field, u, invJ, detJ, user->q, f0, f1, elemVec, user);CHKERRQ(ierr);
    // Remainder
    PetscInt numRemainder = numCells % (numBatches * batchSize);
    PetscInt offset       = numCells - numRemainder;
    ierr = IntegrateResidualBatchCPU(numRemainder, numFields, field, &u[offset*cellDof], &invJ[offset*dim*dim], &detJ[offset],
                                     user->q, f0, f1, &elemVec[offset*cellDof], user);CHKERRQ(ierr);
  }
  for(PetscInt c = cStart; c < cEnd; ++c) {
    if (debug) {ierr = DMPrintCellVector(c, "Residual", cellDof, &elemVec[c*cellDof]);CHKERRQ(ierr);}
    ierr = DMMeshVecSetClosure(dm, F, c, &elemVec[c*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree4(u,invJ,detJ,elemVec);CHKERRQ(ierr);
  ierr = PetscFree3(coords,v0,J);CHKERRQ(ierr);
  if (user->showResidual) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual:\n");CHKERRQ(ierr);
    for(int p = 0; p < user->numProcs; ++p) {
      if (p == user->rank) {ierr = VecView(F, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(user->residualEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateJacobianBatchCPU"
/*
Loop over batch of elements (e):
  Loop over element matrix entries (f,fc,g,gc --> i,j):
    Loop over quadrature points (q):
      Make u_q and gradU_q (loops over fields,Nb,Ncomp)
        elemMat[i,j] += \psi^{fc}_f(q) g0_{fc,gc}(u, \nabla u) \phi^{gc}_g(q)
                      + \psi^{fc}_f(q) \cdot g1_{fc,gc,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
                      + \nabla\psi^{fc}_f(q) \cdot g2_{fc,gc,df}(u, \nabla u) \phi^{gc}_g(q)
                      + \nabla\psi^{fc}_f(q) \cdot g3_{fc,gc,df,dg}(u, \nabla u) \nabla\phi^{gc}_g(q)
*/
PetscErrorCode IntegrateJacobianBatchCPU(PetscInt Ne, PetscInt numFields, PetscInt fieldI, PetscInt fieldJ, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscQuadrature quad[], void (*g0_func)(PetscScalar u[], const PetscScalar gradU[], PetscScalar g0[]), void (*g1_func)(PetscScalar u[], const PetscScalar gradU[], PetscScalar g1[]), void (*g2_func)(PetscScalar u[], const PetscScalar gradU[], PetscScalar g0[]), void (*g3_func)(PetscScalar u[], const PetscScalar gradU[], PetscScalar g1[]), PetscScalar elemMat[], AppCtx *user) {
  const PetscReal *basisI    = quad[fieldI].basis;
  const PetscReal *basisDerI = quad[fieldI].basisDer;
  const PetscReal *basisJ    = quad[fieldJ].basis;
  const PetscReal *basisDerJ = quad[fieldJ].basisDer;
  const PetscInt   debug   = user->debug;
  const PetscInt   dim     = SPATIAL_DIM_0;
  PetscInt         cellDof = 0; // Total number of dof on a cell
  PetscInt         cOffset = 0; // Offset into coefficients[] for element e
  PetscInt         eOffset = 0; // Offset into elemMat[] for element e
  PetscInt         offsetI = 0; // Offset into an element vector for fieldI
  PetscInt         offsetJ = 0; // Offset into an element vector for fieldJ
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(PetscInt field = 0; field < numFields; ++field) {
    if (field == fieldI) {offsetI = cellDof;}
    if (field == fieldJ) {offsetJ = cellDof;}
    cellDof += quad[field].numBasisFuncs*quad[field].numComponents;
  }
  ierr = PetscLogEventBegin(user->integrateJacCPUEvent,0,0,0,0);CHKERRQ(ierr);
  for(PetscInt e = 0; e < Ne; ++e) {
    const PetscReal  detJ    = jacobianDeterminants[e];
    const PetscReal *invJ    = &jacobianInverses[e*dim*dim];
    const PetscInt   Nb_i    = quad[fieldI].numBasisFuncs;
    const PetscInt   Ncomp_i = quad[fieldI].numComponents;
    const PetscInt   Nb_j    = quad[fieldJ].numBasisFuncs;
    const PetscInt   Ncomp_j = quad[fieldJ].numComponents;

    for(int f = 0; f < Nb_i; ++f) {
      for(int g = 0; g < Nb_j; ++g) {
        const PetscInt   Nq          = quad[fieldI].numQuadPoints;
        const PetscReal *quadWeights = quad[fieldI].quadWeights;

        for(PetscInt q = 0; q < Nq; ++q) {
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
          PetscScalar u[dim+1];
          PetscScalar gradU[dim*(dim+1)];
          PetscInt    fOffset            = 0;       // Offset into u[] for field_q (like offsetI)
          PetscInt    dOffset            = cOffset; // Offset into coefficients[] for field_q

          for(PetscInt d = 0; d <= dim; ++d)        {u[d]     = 0.0;}
          for(PetscInt d = 0; d < dim*(dim+1); ++d) {gradU[d] = 0.0;}
          for(PetscInt field_q = 0; field_q < numFields; ++field_q) {
            const PetscInt   Nb          = quad[field_q].numBasisFuncs;
            const PetscInt   Ncomp       = quad[field_q].numComponents;
            const PetscReal *basis       = quad[field_q].basis;
            const PetscReal *basisDer    = quad[field_q].basisDer;

            for(PetscInt b = 0; b < Nb; ++b) {
              for(int comp = 0; comp < Ncomp; ++comp) {
                const PetscInt cidx = b*Ncomp+comp;
                PetscScalar    realSpaceDer[dim];

                u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
                for(PetscInt d1 = 0; d1 < dim; ++d1) {
                  realSpaceDer[d1] = 0.0;
                  for(PetscInt d2 = 0; d2 < dim; ++d2) {
                    realSpaceDer[d1] += invJ[d2*dim+d1]*basisDer[(q*Nb*Ncomp+cidx)*dim+d2];
                  }
                  gradU[(fOffset+comp)*dim+d1] += coefficients[dOffset+cidx]*realSpaceDer[d1];
                }
              }
            }
            if (debug > 1) {
              for(int comp = 0; comp < Ncomp; ++comp) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, u[fOffset+comp]);CHKERRQ(ierr);
                for(PetscInt d = 0; d < dim; ++d) {
                  ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, gradU[(fOffset+comp)*dim+d]);CHKERRQ(ierr);
                }
              }
            }
            fOffset += Ncomp;
            dOffset += Nb*Ncomp;
          }
          PetscScalar g0[dim*dim];         // Ncomp_i*Ncomp_j
          PetscScalar g1[dim*dim*dim];     // Ncomp_i*Ncomp_j*dim
          PetscScalar g2[dim*dim*dim];     // Ncomp_i*Ncomp_j*dim
          PetscScalar g3[dim*dim*dim*dim]; // Ncomp_i*Ncomp_j*dim*dim

          assert(Ncomp_i <= dim);assert(Ncomp_j <= dim);
          ierr = PetscMemzero(g0, Ncomp_i*Ncomp_j         * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g1, Ncomp_i*Ncomp_j*dim     * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g2, Ncomp_i*Ncomp_j*dim     * sizeof(PetscScalar));CHKERRQ(ierr);
          ierr = PetscMemzero(g3, Ncomp_i*Ncomp_j*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
          if (g0_func) {
            g0_func(u, gradU, g0);
            for(int c = 0; c < Ncomp_i*Ncomp_j; ++c) {
              g0[c] *= detJ*quadWeights[q];
            }
          }
          if (g1_func) {
            g1_func(u, gradU, g1);
            for(int c = 0; c < Ncomp_i*Ncomp_j*dim; ++c) {
              g1[c] *= detJ*quadWeights[q];
            }
          }
          if (g2_func) {
            g2_func(u, gradU, g2);
            for(int c = 0; c < Ncomp_i*Ncomp_j*dim; ++c) {
              g2[c] *= detJ*quadWeights[q];
            }
          }
          if (g3_func) {
            g3_func(u, gradU, g3);
            for(int c = 0; c < Ncomp_i*Ncomp_j*dim*dim; ++c) {
              g3[c] *= detJ*quadWeights[q];
            }
          }

          for(int fc = 0; fc < Ncomp_i; ++fc) {
            const PetscInt fidx = f*Ncomp_i+fc; // Test function basis index
            const PetscInt i    = offsetI+fidx; // Element matrix row
            for(int gc = 0; gc < Ncomp_j; ++gc) {
              const PetscInt gidx = g*Ncomp_j+gc; // Trial function basis index
              const PetscInt j    = offsetJ+gidx; // Element matrix column
              PetscScalar    realSpaceDerI[dim];
              PetscScalar    realSpaceDerJ[dim];

              for(PetscInt d = 0; d < dim; ++d) {
                realSpaceDerI[d] = 0.0;
                realSpaceDerJ[d] = 0.0;
                for(PetscInt d2 = 0; d2 < dim; ++d2) {
                  realSpaceDerI[d] += invJ[d2*dim+d]*basisDerI[(q*Nb_i*Ncomp_i+fidx)*dim+d2];
                  realSpaceDerJ[d] += invJ[d2*dim+d]*basisDerJ[(q*Nb_j*Ncomp_j+gidx)*dim+d2];
                }
              }
              elemMat[eOffset+i*cellDof+j] += basisI[q*Nb_i*Ncomp_i+fidx]*g0[fc*Ncomp_j+gc]*basisJ[q*Nb_j*Ncomp_j+gidx];
              for(PetscInt d = 0; d < dim; ++d) {
                elemMat[eOffset+i*cellDof+j] += basisI[q*Nb_i*Ncomp_i+fidx]*g1[(fc*Ncomp_j+gc)*dim+d]*realSpaceDerJ[d];
                elemMat[eOffset+i*cellDof+j] += realSpaceDerI[d]*g2[(fc*Ncomp_j+gc)*dim+d]*basisJ[q*Nb_j*Ncomp_j+gidx];
                for(PetscInt d2 = 0; d2 < dim; ++d2) {
                  elemMat[eOffset+i*cellDof+j] += realSpaceDerI[d]*g3[((fc*Ncomp_j+gc)*dim+d)*dim+d2]*realSpaceDerJ[d2];
                }
              }
            }
          }
        }
      }
    }
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for(int fc = 0; fc < Ncomp_i; ++fc) {
        for(int f = 0; f < Nb_i; ++f) {
          const PetscInt i = offsetI + f*Ncomp_i+fc;
          for(int gc = 0; gc < Ncomp_j; ++gc) {
            for(int g = 0; g < Nb_j; ++g) {
              const PetscInt j = offsetJ + g*Ncomp_j+gc;
              //ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, elemMat[eOffset+i*cellDof+j]);CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_SELF, "%8.6g ", f, fc, g, gc, elemMat[eOffset+i*cellDof+j]);CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset += cellDof;
    eOffset += cellDof*cellDof;
  }
  ierr = PetscLogEventEnd(user->integrateJacCPUEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
  FormJacobianLocal - Form the local portion of the Jacobian matrix J from the local input X.

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. Jac  - Jacobian matrix

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: FormFunctionLocal()
*/
PetscErrorCode FormJacobianLocal(DM dm, Vec X, Mat Jac, AppCtx *user)
{
  const PetscInt debug = user->debug;
  const PetscInt dim   = user->dim;
  PetscReal     *v0, *J, *invJ, *detJ;
  PetscScalar   *elemMat;
  PetscInt       cStart, cEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->jacobianEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MatZeroEntries(Jac);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim,PetscReal,&v0,dim*dim,PetscReal,&J);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  const PetscInt numCells = cEnd - cStart;
  PetscInt       cellDof  = 0;
  PetscScalar   *u;

  for(PetscInt field = 0; field < numFields; ++field) {
    cellDof += user->q[field].numBasisFuncs*user->q[field].numComponents;
  }
  ierr = PetscMalloc4(numCells*cellDof,PetscScalar,&u,numCells*dim*dim,PetscReal,&invJ,numCells,PetscReal,&detJ,numCells*cellDof*cellDof,PetscScalar,&elemMat);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;

    ierr = DMMeshComputeCellGeometry(dm, c, v0, J, &invJ[c*dim*dim], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);}
    ierr = DMMeshVecGetClosure(dm, X, c, &x);CHKERRQ(ierr);

    for(int i = 0; i < cellDof; ++i) {
      u[c*cellDof+i] = x[i];
    }
  }
  ierr = PetscMemzero(elemMat, numCells*cellDof*cellDof * sizeof(PetscScalar));CHKERRQ(ierr);
  for(PetscInt fieldI = 0; fieldI < numFields; ++fieldI) {
    const PetscInt numQuadPoints = user->q[fieldI].numQuadPoints;
    const PetscInt numBasisFuncs = user->q[fieldI].numBasisFuncs;
    for(PetscInt fieldJ = 0; fieldJ < numFields; ++fieldJ) {
      void (*g0)(PetscScalar u[], const PetscScalar gradU[], PetscScalar g0[]) = user->g0Funcs[fieldI*numFields+fieldJ];
      void (*g1)(PetscScalar u[], const PetscScalar gradU[], PetscScalar g1[]) = user->g1Funcs[fieldI*numFields+fieldJ];
      void (*g2)(PetscScalar u[], const PetscScalar gradU[], PetscScalar g2[]) = user->g2Funcs[fieldI*numFields+fieldJ];
      void (*g3)(PetscScalar u[], const PetscScalar gradU[], PetscScalar g3[]) = user->g3Funcs[fieldI*numFields+fieldJ];
      // Conforming batches
      PetscInt blockSize  = numBasisFuncs*numQuadPoints;
      PetscInt numBlocks  = 1;
      PetscInt batchSize  = numBlocks * blockSize;
      PetscInt numBatches = user->numBatches;
      PetscInt numChunks  = numCells / (numBatches*batchSize);
      ierr = IntegrateJacobianBatchCPU(numChunks*numBatches*batchSize, numFields, fieldI, fieldJ, u, invJ, detJ, user->q, g0, g1, g2, g3, elemMat, user);CHKERRQ(ierr);
      // Remainder
      PetscInt numRemainder = numCells % (numBatches * batchSize);
      PetscInt offset       = numCells - numRemainder;
      ierr = IntegrateJacobianBatchCPU(numRemainder, numFields, fieldI, fieldJ, &u[offset*cellDof], &invJ[offset*dim*dim], &detJ[offset],
                                       user->q, g0, g1, g2, g3, &elemMat[offset*cellDof*cellDof], user);CHKERRQ(ierr);
    }
  }
  if (debug) {
    ALE::Obj<PETSC_MESH_TYPE> mesh;
    ierr = DMMeshGetMesh(dm, mesh);CHKERRQ(ierr);
    mesh->setDebug(debug);
    mesh->getSieve()->setDebug(debug);
  }
  for(PetscInt c = cStart; c < cEnd; ++c) {
    if (debug) {ierr = DMPrintCellMatrix(c, "Jacobian", cellDof, cellDof, &elemMat[c*cellDof*cellDof]);CHKERRQ(ierr);}
    ierr = DMMeshMatSetClosure(dm, Jac, c, &elemMat[c*cellDof*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree4(u,invJ,detJ,elemMat);CHKERRQ(ierr);
  ierr = PetscFree2(v0,J);CHKERRQ(ierr);

  /* Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd(). */
  ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (user->showJacobian) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Jacobian:\n");CHKERRQ(ierr);
    ierr = MatView(Jac, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(user->jacobianEvent,0,0,0,0);CHKERRQ(ierr);
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
  PetscInt       its;                  /* iterations for convergence */
  PetscReal      error;                /* L_2 error in the solution */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, user.dm);CHKERRQ(ierr);

  ierr = SetupExactSolution(&user);CHKERRQ(ierr);
  ierr = SetupQuadrature(&user);CHKERRQ(ierr);
  ierr = SetupSection(user.dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMCreateMatrix(user.dm, MATAIJ, &J);CHKERRQ(ierr);
  A    = J;
  ierr = SNESSetJacobian(snes, A, J, SNESDMMeshComputeJacobian, &user);CHKERRQ(ierr);
  ierr = CreatePressureNullSpace(user.dm, &user, &nullSpace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J, nullSpace);CHKERRQ(ierr);

  ierr = DMMeshSetLocalFunction(user.dm, (DMMeshLocalFunction1) FormFunctionLocal);CHKERRQ(ierr);
  ierr = DMMeshSetLocalJacobian(user.dm, (DMMeshLocalJacobian1) FormJacobianLocal);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, SNESDMMeshComputeFunction, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMMeshSetLocalFunction(user.dm, (DMMeshLocalFunction1) FormFunctionLocal);CHKERRQ(ierr);
  ierr = DMMeshSetLocalJacobian(user.dm, (DMMeshLocalJacobian1) FormJacobianLocal);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, SNESDMMeshComputeFunction, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMComputeVertexFunction(user.dm, INSERT_ALL_VALUES, u, numComponents, user.exactFuncs, &user);CHKERRQ(ierr);
  if (user.runType == RUN_FULL) {
    PetscScalar (*initialGuess[numComponents])(const PetscReal x[]);

    for(PetscInt c = 0; c < numComponents; ++c) {initialGuess[c] = zero;}
    ierr = DMComputeVertexFunction(user.dm, INSERT_VALUES, u, numComponents, initialGuess, &user);CHKERRQ(ierr);
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
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
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Vec          b;
      MatStructure flag;
      PetscBool    isNull;

      ierr = SNESDMMeshComputeJacobian(snes, u, &A, &A, &flag, &user);CHKERRQ(ierr);
      ierr = MatNullSpaceTest(nullSpace, J, &isNull);CHKERRQ(ierr);
      if (!isNull) {SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");}
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESDMMeshComputeFunction(snes, r, b, &user);CHKERRQ(ierr);
      ierr = MatMult(A, u, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n");CHKERRQ(ierr);
      ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", res);CHKERRQ(ierr);
    }
  }

  if (user.runType == RUN_FULL) {
    PetscViewer viewer;

    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
    /*ierr = PetscViewerSetType(viewer, PETSCVIEWERDRAW);CHKERRQ(ierr);
      ierr = PetscViewerDrawSetInfo(viewer, PETSC_NULL, "Solution", PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr); */
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "ex56_sol.vtk");CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(user.dm, viewer);CHKERRQ(ierr);
    ierr = VecView(u, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
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
