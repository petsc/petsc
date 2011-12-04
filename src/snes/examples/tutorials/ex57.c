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
  {
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
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, user.dm);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
