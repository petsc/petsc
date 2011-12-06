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
#define __FUNCT__ "CreateTensorQuadrature"
PetscErrorCode CreateTensorQuadrature(PetscInt dim, PetscInt numQuadPoints, const PetscReal quadPoints[], const PetscReal quadWeights[], PetscInt *numTensorQuadPoints, PetscReal **tensorQuadPoints, PetscReal **tensorQuadWeights) {
  PetscInt Nq  = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(PetscInt d = 0; d < dim; ++d) {
    Nq *= numQuadPoints;
  }
  ierr = PetscMalloc2(Nq*dim,PetscReal,tensorQuadPoints,Nq,PetscReal,tensorQuadWeights);CHKERRQ(ierr);
  switch(dim) {
  case 2:
    for(PetscInt q = 0; q < numQuadPoints; ++q) {
      for(PetscInt r = 0; r < numQuadPoints; ++r) {
        (*tensorQuadPoints)[(q*numQuadPoints+r)*dim+0] = quadPoints[q];
        (*tensorQuadPoints)[(q*numQuadPoints+r)*dim+1] = quadPoints[r];
        (*tensorQuadWeights)[q*numQuadPoints+r]        = quadWeights[q]*quadWeights[r];
      }
    }
    break;
  case 3:
    for(PetscInt q = 0; q < numQuadPoints; ++q) {
      for(PetscInt r = 0; r < numQuadPoints; ++r) {
        for(PetscInt s = 0; s < numQuadPoints; ++s) {
          (*tensorQuadPoints)[((q*numQuadPoints+r)*numQuadPoints+s)*dim+0] = quadPoints[q];
          (*tensorQuadPoints)[((q*numQuadPoints+r)*numQuadPoints+s)*dim+1] = quadPoints[r];
          (*tensorQuadPoints)[((q*numQuadPoints+r)*numQuadPoints+s)*dim+2] = quadPoints[s];
          (*tensorQuadWeights)[(q*numQuadPoints+r)*numQuadPoints+s]        = quadWeights[q]*quadWeights[r]*quadWeights[s];
        }
      }
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %d not supported", dim);
  }
  *numTensorQuadPoints = Nq;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTensorBasis"
  PetscErrorCode CreateTensorBasis(PetscInt dim, PetscInt numQuadPoints, PetscInt numBasisFuncs, PetscInt numBasisComps, const PetscReal basis[], const PetscReal basisDer[], PetscInt *numTensorBasisFuncs, PetscReal **tensorBasis, PetscReal **tensorBasisDer) {
  PetscInt Nq  = 1;
  PetscInt Nb  = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(PetscInt d = 0; d < dim; ++d) {
    Nq *= numQuadPoints;
  }
  for(PetscInt d = 0; d < dim; ++d) {
    Nb *= numBasisFuncs;
  }
  ierr = PetscMalloc2(Nb*numBasisComps*Nq,PetscReal,tensorBasis,Nb*numBasisComps*Nq*dim,PetscReal,tensorBasisDer);CHKERRQ(ierr);
  switch(dim) {
  case 2:
    for(PetscInt q = 0; q < numQuadPoints; ++q) {
      for(PetscInt r = 0; r < numQuadPoints; ++r) {
        for(PetscInt b1 = 0; b1 < numBasisFuncs; ++b1) {
          for(PetscInt b2 = 0; b2 < numBasisFuncs; ++b2) {
            for(PetscInt c = 0; c < numBasisComps; ++c) {
              (*tensorBasis)[((q*numQuadPoints+r)*Nb + (b1*numBasisFuncs+b2))*numBasisComps+c] = basis[(q*numBasisFuncs+b1)*numBasisComps+c]*basis[(r*numBasisFuncs+b2)*numBasisComps+c];
              (*tensorBasisDer)[(((q*numQuadPoints+r)*Nb + (b1*numBasisFuncs+b2))*numBasisComps+c)*dim+0] = basisDer[(q*numBasisFuncs+b1)*numBasisComps+c+0]*basis[(r*numBasisFuncs+b2)*numBasisComps+c];
              (*tensorBasisDer)[(((q*numQuadPoints+r)*Nb + (b1*numBasisFuncs+b2))*numBasisComps+c)*dim+1] = basis[(q*numBasisFuncs+b1)*numBasisComps+c]*basisDer[(r*numBasisFuncs+b2)*numBasisComps+c+0];
            }
          }
        }
      }
    }
    break;
  case 3:
    for(PetscInt q = 0; q < numQuadPoints; ++q) {
      for(PetscInt r = 0; r < numQuadPoints; ++r) {
        for(PetscInt s = 0; s < numQuadPoints; ++s) {
          for(PetscInt b1 = 0; b1 < numBasisFuncs; ++b1) {
            for(PetscInt b2 = 0; b2 < numBasisFuncs; ++b2) {
              for(PetscInt b3 = 0; b3 < numBasisFuncs; ++b3) {
                for(PetscInt c = 0; c < numBasisComps; ++c) {
                  (*tensorBasis)[(((q*numQuadPoints+r)*numQuadPoints+s)*Nb + ((b1*numBasisFuncs+b2)*numBasisFuncs+b3))*numBasisComps+c] =
                    basis[(q*numBasisFuncs+b1)*numBasisComps+c]*basis[(r*numBasisFuncs+b2)*numBasisComps+c]*basis[(s*numBasisFuncs+b3)*numBasisComps+c];
                  (*tensorBasisDer)[((((q*numQuadPoints+r)*numQuadPoints+s)*Nb + ((b1*numBasisFuncs+b2)*numBasisFuncs+b3))*numBasisComps+c)*dim+0] =
                    basisDer[(q*numBasisFuncs+b1)*numBasisComps+c+0]*basis[(r*numBasisFuncs+b2)*numBasisComps+c]*basis[(s*numBasisFuncs+b3)*numBasisComps+c];
                  (*tensorBasisDer)[((((q*numQuadPoints+r)*numQuadPoints+s)*Nb + ((b1*numBasisFuncs+b2)*numBasisFuncs+b3))*numBasisComps+c)*dim+0] =
                    basis[(q*numBasisFuncs+b1)*numBasisComps+c]*basisDer[(r*numBasisFuncs+b2)*numBasisComps+c+0]*basis[(s*numBasisFuncs+b3)*numBasisComps+c];
                  (*tensorBasisDer)[((((q*numQuadPoints+r)*numQuadPoints+s)*Nb + ((b1*numBasisFuncs+b2)*numBasisFuncs+b3))*numBasisComps+c)*dim+0] =
                    basis[(q*numBasisFuncs+b1)*numBasisComps+c]*basis[(r*numBasisFuncs+b2)*numBasisComps+c]*basisDer[(s*numBasisFuncs+b3)*numBasisComps+c+0];
                }
              }
            }
          }
        }
      }
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %d not supported", dim);
  }
  *numTensorBasisFuncs = Nb;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupQuadrature"
PetscErrorCode SetupQuadrature(AppCtx *user) {
  PetscInt       dim = user->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //user->q[0].numQuadPoints = NUM_QUADRATURE_POINTS_0;
  //user->q[0].quadPoints    = points_0;
  //user->q[0].quadWeights   = weights_0;
  ierr = CreateTensorQuadrature(dim, NUM_QUADRATURE_POINTS_0, points_0, weights_0, &user->q[0].numQuadPoints, (PetscReal**) &user->q[0].quadPoints, (PetscReal **) &user->q[0].quadWeights);CHKERRQ(ierr);
  //user->q[0].numBasisFuncs = NUM_BASIS_FUNCTIONS_0;
  user->q[0].numComponents = NUM_BASIS_COMPONENTS_0;
  //user->q[0].basis         = Basis_0;
  //user->q[0].basisDer      = BasisDerivatives_0;
  ierr = CreateTensorBasis(dim, NUM_QUADRATURE_POINTS_0, NUM_BASIS_FUNCTIONS_0, NUM_BASIS_COMPONENTS_0, Basis_0, BasisDerivatives_0, &user->q[0].numBasisFuncs, (PetscReal **) &user->q[0].basis, (PetscReal **) &user->q[0].basisDer);CHKERRQ(ierr);

  //user->q[1].numQuadPoints = NUM_QUADRATURE_POINTS_1;
  //user->q[1].quadPoints    = points_1;
  //user->q[1].quadWeights   = weights_1;
  ierr = CreateTensorQuadrature(dim, NUM_QUADRATURE_POINTS_1, points_1, weights_1, &user->q[1].numQuadPoints, (PetscReal **) &user->q[1].quadPoints, (PetscReal **) &user->q[1].quadWeights);CHKERRQ(ierr);
  //user->q[1].numBasisFuncs = NUM_BASIS_FUNCTIONS_1;
  user->q[1].numComponents = NUM_BASIS_COMPONENTS_1;
  //user->q[1].basis         = Basis_1;
  //user->q[1].basisDer      = BasisDerivatives_1;
  ierr = CreateTensorBasis(dim, NUM_QUADRATURE_POINTS_1, NUM_BASIS_FUNCTIONS_1, NUM_BASIS_COMPONENTS_1, Basis_1, BasisDerivatives_1, &user->q[1].numBasisFuncs, (PetscReal **) &user->q[1].basis, (PetscReal **) &user->q[1].basisDer);CHKERRQ(ierr);
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
  PetscInt      *numDof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // NEED flags to distinguish tensor product from regular disc
  if (1 != SPATIAL_DIM_0) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Spatial dimension %d should be %d", 1, SPATIAL_DIM_0);}
  if (1 != SPATIAL_DIM_1) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_ARG_SIZ, "Spatial dimension %d should be %d", 1, SPATIAL_DIM_1);}
  ierr = PetscMalloc(numFields*(dim+1) * sizeof(PetscInt), &numDof);CHKERRQ(ierr);
  numDof[0*(dim+1)+0] = numDof_0[0];
  numDof[1*(dim+1)+0] = numDof_1[0];
  numDof[0*(dim+1)+1] = numDof_0[1];
  numDof[1*(dim+1)+1] = numDof_1[1];
  for(PetscInt d = 2; d <= dim; ++d) {
    numDof[0*(dim+1)+d] = numDof[0*(dim+1)+d-1]*numDof[0*(dim+1)+d-1];
    numDof[1*(dim+1)+d] = numDof[1*(dim+1)+d-1]*numDof[1*(dim+1)+d-1];
  }
  for(PetscInt f = 0; f < numFields; ++f) {
    for(PetscInt d = 1; d < dim; ++d) {
      if ((numDof[f*(dim+1)+d] > 0) && !user->interpolate) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");}
    }
  }
  if (user->bcType == DIRICHLET) {
    numBC = 1;
    ierr  = DMMeshGetStratumIS(dm, "marker", 1, &bcPoints[0]);CHKERRQ(ierr);
  }
  ierr = DMMeshCreateSection(dm, dim, numFields, numComp, numDof, numBC, bcFields, bcPoints, &section);CHKERRQ(ierr);
  ierr = DMMeshSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = PetscFree(numDof);CHKERRQ(ierr);
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
    if (detJ <= 0.0) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);}
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
        ierr = DMMeshPrintCellVector(c, title, numBasisFuncs*numBasisComps, &x[fieldOffset]);CHKERRQ(ierr);
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
  PetscFunctionBegin;
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
  PetscFunctionBegin;
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
