static char help[] = "Stokes Problem with Temperature in 2d and 3d with simplicial finite elements.\n\
We solve the Stokes problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*
TODO for Mantle Convection:
  - Variable viscosity
  - Free-slip boundary condition on upper surface
  - Parse Citcom input

The isoviscous Stokes problem, which we discretize using the finite
element method on an unstructured mesh. The weak form equations are

  < \nabla v, \nabla u + {\nabla u}^T > - < \nabla\cdot v, p > + < v, f > = 0
  < q, \nabla\cdot v >                                                    = 0
  < \nabla t, \nabla T>                                                   = q_T

Boundary Conditions:

  -bc_type dirichlet   Dirichlet conditions on the entire boundary, coming from the exact solution functions

  -dm_plex_separate_marker
  -bc_type freeslip    Dirichlet conditions on the normal velocity at each boundary

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
integration. Therefore, the closure visitor in DMPlexVecGetClosure() reorders
the data so that each field is contiguous

  x'    = [u_{e_0} v_{e_0} u_{e_1} v_{e_1} u_{e_2} v_{e_2} u_{v_0} v_{v_0} u_{v_1} v_{v_1} u_{v_2} v_{v_2} p_{v_0} p_{v_1} p_{v_2}]

Likewise, DMPlexVecSetClosure() takes data partitioned by field, and correctly
puts it into the Sieve ordering.
*/

#include <petscdmplex.h>
#include <petscsnes.h>

/*------------------------------------------------------------------------------
  This code can be generated using 'bin/pythonscripts/PetscGenerateFEMQuadrature.py dim order dim 1 laplacian dim order 1 1 gradient dim order 1 1 identity src/snes/examples/tutorials/ex31.h'
 -----------------------------------------------------------------------------*/
#include "ex31.h"

typedef enum {DIRICHLET, FREE_SLIP} BCType;
typedef enum {RUN_FULL, RUN_TEST} RunType;
typedef enum {FORCING_CONSTANT, FORCING_LINEAR, FORCING_CUBIC} ForcingType;

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscFEM      fem;               /* REQUIRED to use DMPlexComputeResidualFEM() */
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
  void (*exactFuncs[NUM_BASIS_COMPONENTS_TOTAL])(const PetscReal x[], PetscScalar *u); /* The exact solution function u(x,y,z), v(x,y,z), p(x,y,z), and T(x,y,z) */
  BCType      bcType;              /* The type of boundary conditions */
  ForcingType forcingType;         /* The type of rhs */
} AppCtx;

void zero(const PetscReal coords[], PetscScalar *u)
{
  *u = 0.0;
}

/*
  In 2D, for constant forcing,

    f_x = f_y = 3

  we use the exact solution,

    u = x^2 + y^2
    v = 2 x^2 - 2xy
    p = x + y - 1
    T = x + y

  so that

    -\Delta u + \nabla p + f = <-4, -4> + <1, 1> + <3, 3> = 0
    \nabla \cdot u           = 2x - 2x                    = 0
    -\Delta T + q_T          = 0
*/
void quadratic_u_2d(const PetscReal x[], PetscScalar *u)
{
  *u = x[0]*x[0] + x[1]*x[1];
};

void quadratic_v_2d(const PetscReal x[], PetscScalar *v)
{
  *v = 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
};

void linear_p_2d(const PetscReal x[], PetscScalar *p)
{
  *p = x[0] + x[1] - 1.0;
};

void linear_T_2d(const PetscReal x[], PetscScalar *T)
{
  *T = x[0] + x[1];
};

/*
  In 2D, for linear forcing,

    f_x =  3 - 8y
    f_y = -5 + 8x

  we use the exact solution,

    u =  2 x (x-1) (1 - 2 y)
    v = -2 y (y-1) (1 - 2 x)
    p = x + y - 1
    T = x + y

  so that

    -\Delta u + \nabla p + f = <-4+8y, 4-8x> + <1, 1> + <3-8y, 8x-5> = 0
    \nabla \cdot u           = (4x-2) (1-2y) - (4y-2) (1-2x)         = 0
    -\Delta T + q_T          = 0
*/
void cubic_u_2d(const PetscReal x[], PetscScalar *u)
{
  *u = 2.0*x[0]*(x[0]-1.0)*(1.0 - 2.0*x[1]);
};

void cubic_v_2d(const PetscReal x[], PetscScalar *v)
{
  *v = -2.0*x[1]*(x[1]-1.0)*(1.0 - 2.0*x[0]);
};

/*
  Let \sigma = (\nabla u + \nabla u^T) = < \sigma_{ij} >, where \sigma_{ij} = \sigma_{ji}
  Then at the top and bottom (t = <1,0>),
    <\sigma_{00}, \sigma_{01}> = 0 so \sigma_{00} = A(x,y) y(1-y) \sigma_{01} = B(x,y) y(1-y)
  Using the left and right (t = <0,1>),
    <\sigma_{10}, \sigma_{11}> = 0 so \sigma_{10} = C(x,y) x(1-x) \sigma_{11} = D(x,y) x(1-x)
  Which means
    \sigma_{00} = A(x,y) y(1-y)        = 2 u_x
    \sigma_{01} = E(x,y) x(1-x) y(1-y) = u_y + v_x
    \sigma_{11} = D(x,y) x(1-x)        = 2 v_y
  Also we have
    u(x=0,1) = 0 ==> u = A'(x,y) x(1-x)
    v(y=0,1) = 0 ==> v = D'(x,y) y(1-y)
  Thus we need
    \int x - x^2 = x^2/2 - x^3/3 + C ==> 3 x^2 - 2 x^3 + 1 = 0 at x=0,1
  so that
    u =  (3 x^2 - 2 x^3 + 1) y(1-y)
    v = -(3 y^2 - 2 y^3 + 1) x(1-x)
    u_x =  6 x(1-x) y(1-y)
    v_y = -6 x(1-x) y(1-y)
    u_xx =  6 (1-2x) y(1-y)
    v_yy = -6 (1-2y) x(1-x)

  In 2D, for cubic forcing,

    f_x = -1 + 6 (1-2x) y(1-y)
    f_y = -1 - 6 (1-2y) x(1-x)

  we use the exact solution,

    u =  (3 x^2 - 2 x^3 + 1) y(1-y)
    v = -(3 y^2 - 2 y^3 + 1) x(1-x)
    p = x + y - 1
    T = x + y

  so that

    -\Delta u + \nabla p + f = <-6 (1-2x) y(1-y), 6 (1-2y) x(1-x)> + <1, 1> + <-1 + 6 (1-2x) y(1-y), -1 - 6 (1-2y) x(1-x)> = 0
    \nabla \cdot u           = 6 x(1-x) y(1-y) -6 (1-2y) x(1-x) = 0
    -\Delta T + q_T          = 0
*/
void quintic_u_2d(const PetscReal x[], PetscScalar *u)
{
  *u = (3.0*x[0]*x[0] - 2.0*x[0]*x[0]*x[0] + 1.0)*x[1]*(1.0-x[1]);
};

void quintic_v_2d(const PetscReal x[], PetscScalar *v)
{
  *v = -(3.0*x[1]*x[1] - 2.0*x[1]*x[1]*x[1] + 1.0)*x[0]*(1.0-x[0]);
};

void f0_u_constant(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[])
{
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) f0[comp] = 3.0;
}

void f0_u_linear_2d(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] =  3.0 - 8.0*x[1];
  f0[1] = -5.0 + 8.0*x[0];
}

void f0_u_cubic_2d(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = -1.0 + 6.0*(1.0 - 2.0*x[0])*x[1]*(1.0 - x[1]);
  f0[1] = -1.0 - 6.0*(1.0 - 2.0*x[1])*x[0]*(1.0 - x[0]);
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
   u[Ncomp]          = {p} */
void f1_u(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[])
{
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
void f0_p(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[])
{
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) f0[0] += gradU[d*dim+d];
}

void f1_p(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[])
{
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) f1[d] = 0.0;
}

void f0_T(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

void f1_T(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[])
{
  const PetscInt dim = SPATIAL_DIM_2;
  const PetscInt off = SPATIAL_DIM_0*NUM_BASIS_COMPONENTS_0+SPATIAL_DIM_1*NUM_BASIS_COMPONENTS_1;
  PetscInt       d;

  for (d = 0; d < dim; ++d) f1[d] = gradU[off+d];
}

/* < v_t, I t > */
void g0_TT(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

/* < q, \nabla\cdot v >
   NcompI = 1, NcompJ = dim */
void g1_pu(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g1[])
{
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
}

/* -< \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
void g2_up(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g2[])
{
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g3[])
{
  const PetscInt dim   = SPATIAL_DIM_0;
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;
  PetscInt       compI, d;

  for (compI = 0; compI < Ncomp; ++compI) {
    for (d = 0; d < dim; ++d) {
      g3[((compI*Ncomp+compI)*dim+d)*dim+d] = 1.0;
    }
  }
}

/* < \nabla t, \nabla T + {\nabla u}^T >
   This just gives \nabla T, give the perdiagonal for the transpose */
void g3_TT(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar g3[])
{
  const PetscInt dim   = SPATIAL_DIM_2;
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_2;
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
void quadratic_u_3d(const PetscReal x[], PetscScalar *u)
{
  *u = x[0]*x[0] + x[1]*x[1];
};

void quadratic_v_3d(const PetscReal x[], PetscScalar *v)
{
  *v = x[1]*x[1] + x[2]*x[2];
};

void quadratic_w_3d(const PetscReal x[], PetscScalar *w)
{
  *w = x[0]*x[0] + x[1]*x[1] - 2.0*(x[0] + x[1])*x[2];
};

void linear_p_3d(const PetscReal x[], PetscScalar *p)
{
  *p = x[0] + x[1] + x[2] - 1.5;
};

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char     *bcTypes[2]      = {"dirichlet", "freeslip"};
  const char     *forcingTypes[3] = {"constant", "linear", "cubic"};
  const char     *runTypes[2]     = {"full", "test"};
  PetscInt       bc, forcing, run;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug           = 0;
  options->runType         = RUN_FULL;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;
  options->bcType          = DIRICHLET;
  options->forcingType     = FORCING_CONSTANT;
  options->numBatches      = 1;
  options->numBlocks       = 1;
  options->jacobianMF      = PETSC_FALSE;
  options->showInitial     = PETSC_FALSE;
  options->showSolution    = PETSC_TRUE;

  options->fem.quad    = (PetscQuadrature*) &options->q;
  options->fem.f0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f0Funcs;
  options->fem.f1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f1Funcs;
  options->fem.g0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g0Funcs;
  options->fem.g1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g1Funcs;
  options->fem.g2Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g2Funcs;
  options->fem.g3Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g3Funcs;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Stokes Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex31.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex31.c", runTypes, 2, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);

  options->runType = (RunType) run;

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex31.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex31.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex31.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex31.c",bcTypes,2,bcTypes[options->bcType],&bc,NULL);CHKERRQ(ierr);

  options->bcType = (BCType) bc;
  forcing         = options->forcingType;

  ierr = PetscOptionsEList("-forcing_type","Type of forcing function","ex31.c",forcingTypes,3,forcingTypes[options->forcingType],&forcing,NULL);CHKERRQ(ierr);

  options->forcingType = (ForcingType) forcing;

  ierr = PetscOptionsInt("-gpu_batches", "The number of cell batches per kernel", "ex31.c", options->numBatches, &options->numBatches, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-gpu_blocks", "The number of concurrent blocks per kernel", "ex31.c", options->numBlocks, &options->numBlocks, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-jacobian_mf", "Calculate the action of the Jacobian on the fly", "ex31.c", options->jacobianMF, &options->jacobianMF, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex31.c", options->showInitial, &options->showInitial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex31.c", options->showSolution, &options->showSolution, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

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
  const char     *partitioner    = user->partitioner;
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
    ierr = DMPlexDistribute(*dm, partitioner, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr     = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr     = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PointOnBoundary_2D"
PetscErrorCode PointOnBoundary_2D(const PetscScalar coords[], PetscBool onBd[])
{
  const PetscInt  corner = 0, bottom = 1, right = 2, top = 3, left = 4;
  const PetscReal eps    = 1.0e-10;

  PetscFunctionBeginUser;
  onBd[bottom] = PetscAbsScalar(coords[1])       < eps ? PETSC_TRUE : PETSC_FALSE;
  onBd[right]  = PetscAbsScalar(coords[0] - 1.0) < eps ? PETSC_TRUE : PETSC_FALSE;
  onBd[top]    = PetscAbsScalar(coords[1] - 1.0) < eps ? PETSC_TRUE : PETSC_FALSE;
  onBd[left]   = PetscAbsScalar(coords[0])       < eps ? PETSC_TRUE : PETSC_FALSE;
  onBd[corner] = onBd[bottom] + onBd[right] + onBd[top] + onBd[left] > 1 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateBoundaryPointIS_Square"
PetscErrorCode CreateBoundaryPointIS_Square(DM dm, PetscInt *numBoundaries, PetscInt **numBoundaryConstraints, IS **boundaryPoints, IS **constraintIndices)
{
  MPI_Comm       comm;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar    *coords;
  PetscInt       vStart, vEnd;
  IS             bcPoints;
  const PetscInt *points;
  const PetscInt corner               = 0, bottom = 1, right = 2, top = 3, left = 4;
  PetscInt       numBoundaryPoints[5] = {0, 0, 0, 0, 0}, bd, numPoints, p;
  PetscInt       *bdPoints[5], *idx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  /* boundary 0: corners
     boundary 1: bottom
     boundary 2: right
     boundary 3: top
     boundary 4: left
  */
  *numBoundaries = 5;

  ierr = PetscMalloc(*numBoundaries * sizeof(PetscInt), numBoundaryConstraints);CHKERRQ(ierr);
  ierr = PetscMalloc(*numBoundaries * sizeof(IS), boundaryPoints);CHKERRQ(ierr);
  ierr = PetscMalloc(*numBoundaries * sizeof(IS), constraintIndices);CHKERRQ(ierr);

  /* Set number of constraints for each boundary */
  (*numBoundaryConstraints)[corner] = 2;
  (*numBoundaryConstraints)[bottom] = 1;
  (*numBoundaryConstraints)[right]  = 1;
  (*numBoundaryConstraints)[top]    = 1;
  (*numBoundaryConstraints)[left]   = 1;
  /* Set local constraint indices for each boundary */
  ierr   = PetscMalloc((*numBoundaryConstraints)[corner] * sizeof(PetscInt), &idx);CHKERRQ(ierr);
  idx[0] = 0; idx[1] = 1;
  ierr   = ISCreateGeneral(comm, (*numBoundaryConstraints)[corner], idx, PETSC_OWN_POINTER, &(*constraintIndices)[corner]);CHKERRQ(ierr);
  ierr   = PetscMalloc((*numBoundaryConstraints)[bottom] * sizeof(PetscInt), &idx);CHKERRQ(ierr);
  idx[0] = 1;
  ierr   = ISCreateGeneral(comm, (*numBoundaryConstraints)[bottom], idx, PETSC_OWN_POINTER, &(*constraintIndices)[bottom]);CHKERRQ(ierr);
  ierr   = PetscMalloc((*numBoundaryConstraints)[right] * sizeof(PetscInt), &idx);CHKERRQ(ierr);
  idx[0] = 0;
  ierr   = ISCreateGeneral(comm, (*numBoundaryConstraints)[right], idx, PETSC_OWN_POINTER, &(*constraintIndices)[right]);CHKERRQ(ierr);
  ierr   = PetscMalloc((*numBoundaryConstraints)[top] * sizeof(PetscInt), &idx);CHKERRQ(ierr);
  idx[0] = 1;
  ierr   = ISCreateGeneral(comm, (*numBoundaryConstraints)[top], idx, PETSC_OWN_POINTER, &(*constraintIndices)[top]);CHKERRQ(ierr);
  ierr   = PetscMalloc((*numBoundaryConstraints)[left] * sizeof(PetscInt), &idx);CHKERRQ(ierr);
  idx[0] = 0;
  ierr   = ISCreateGeneral(comm, (*numBoundaryConstraints)[left], idx, PETSC_OWN_POINTER, &(*constraintIndices)[left]);CHKERRQ(ierr);

  /* Count points on each boundary */
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMPlexGetStratumIS(dm, "marker", 1, &bcPoints);CHKERRQ(ierr);
  ierr = ISGetLocalSize(bcPoints, &numPoints);CHKERRQ(ierr);
  ierr = ISGetIndices(bcPoints, &points);CHKERRQ(ierr);
  for (p = 0; p < numPoints; ++p) {
    PetscBool onBd[5];
    PetscInt  off, bd;

    if ((points[p] >= vStart) && (points[p] < vEnd)) {
      ierr = PetscSectionGetOffset(coordSection, points[p], &off);CHKERRQ(ierr);
      ierr = PointOnBoundary_2D(&coords[off], onBd);CHKERRQ(ierr);
    } else {
      PetscInt *closure = NULL;
      PetscInt closureSize, q, r;

      ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      /* Compress out non-vertices */
      for (q = 0, r = 0; q < closureSize*2; q += 2) {
        if ((closure[q] >= vStart) && (closure[q] < vEnd)) {
          closure[r] = closure[q];
          ++r;
        }
      }
      closureSize = r;
      for (q = 0; q < closureSize; ++q) {
        ierr = PetscSectionGetOffset(coordSection, closure[q], &off);CHKERRQ(ierr);
        ierr = PointOnBoundary_2D(&coords[off], onBd);CHKERRQ(ierr);
        if (!onBd[corner]) break;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if (q == closureSize) SETERRQ1(comm, PETSC_ERR_PLIB, "Cannot handle face %d which has every vertex on a corner", points[p]);
    }

    for (bd = 0; bd < 5; ++bd) {
      if (onBd[bd]) {
        ++numBoundaryPoints[bd];
        break;
      }
    }
  }
  /* Set points on each boundary */
  for (bd = 0; bd < 5; ++bd) {
    ierr = PetscMalloc(numBoundaryPoints[bd] * sizeof(PetscInt), &bdPoints[bd]);CHKERRQ(ierr);
    numBoundaryPoints[bd] = 0;
  }
  for (p = 0; p < numPoints; ++p) {
    PetscBool onBd[5];
    PetscInt  off, bd;

    if ((points[p] >= vStart) && (points[p] < vEnd)) {
      ierr = PetscSectionGetOffset(coordSection, points[p], &off);CHKERRQ(ierr);
      ierr = PointOnBoundary_2D(&coords[off], onBd);CHKERRQ(ierr);
    } else {
      PetscInt *closure = NULL;
      PetscInt closureSize, q, r;

      ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      /* Compress out non-vertices */
      for (q = 0, r = 0; q < closureSize*2; q += 2) {
        if ((closure[q] >= vStart) && (closure[q] < vEnd)) {
          closure[r] = closure[q];
          ++r;
        }
      }
      closureSize = r;
      for (q = 0; q < closureSize; ++q) {
        ierr = PetscSectionGetOffset(coordSection, closure[q], &off);CHKERRQ(ierr);
        ierr = PointOnBoundary_2D(&coords[off], onBd);CHKERRQ(ierr);
        if (!onBd[corner]) break;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if (q == closureSize) SETERRQ1(comm, PETSC_ERR_PLIB, "Cannot handle face %d which has every vertex on a corner", points[p]);
    }

    for (bd = 0; bd < 5; ++bd) {
      if (onBd[bd]) {
        bdPoints[bd][numBoundaryPoints[bd]++] = points[p];
        break;
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = ISRestoreIndices(bcPoints, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPoints);CHKERRQ(ierr);
  for (bd = 0; bd < 5; ++bd) {
    ierr = ISCreateGeneral(comm, numBoundaryPoints[bd], bdPoints[bd], PETSC_OWN_POINTER, &(*boundaryPoints)[bd]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateBoundaryPointIS_Cube"
PetscErrorCode CreateBoundaryPointIS_Cube(DM dm, PetscInt *numBoundaries, PetscInt **numBoundaryConstraints, IS **boundaryPoints, IS **constraintIndices)
{
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Just lazy");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateBoundaryPointIS"
/* This will only work for the square/cube, but I think the interface is robust */
PetscErrorCode CreateBoundaryPointIS(DM dm, PetscInt *numBoundaries, PetscInt **numBoundaryConstraints, IS **boundaryPoints, IS **constraintIndices)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    CreateBoundaryPointIS_Square(dm, numBoundaries, numBoundaryConstraints, boundaryPoints, constraintIndices);CHKERRQ(ierr);
    break;
  case 3:
    CreateBoundaryPointIS_Cube(dm, numBoundaries, numBoundaryConstraints, boundaryPoints, constraintIndices);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "No boundary creatin routine for dimension %d", dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupQuadrature"
PetscErrorCode SetupQuadrature(AppCtx *user)
{
  PetscFunctionBeginUser;
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
  user->fem.quad[2].numQuadPoints = NUM_QUADRATURE_POINTS_2;
  user->fem.quad[2].quadPoints    = points_2;
  user->fem.quad[2].quadWeights   = weights_2;
  user->fem.quad[2].numBasisFuncs = NUM_BASIS_FUNCTIONS_2;
  user->fem.quad[2].numComponents = NUM_BASIS_COMPONENTS_2;
  user->fem.quad[2].basis         = Basis_2;
  user->fem.quad[2].basisDer      = BasisDerivatives_2;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
/*
  There is a problem here with uninterpolated meshes. The index in numDof[] is not dimension in this case,
  but sieve depth.
*/
PetscErrorCode SetupSection(DM dm, AppCtx *user)
{
  PetscSection   section;
  const PetscInt numFields           = NUM_FIELDS;
  PetscInt       dim                 = user->dim;
  PetscInt       numBC               = 0;
  PetscInt       numComp[NUM_FIELDS] = {NUM_BASIS_COMPONENTS_0, NUM_BASIS_COMPONENTS_1, NUM_BASIS_COMPONENTS_2};
  PetscInt       bcFields[2]         = {0, 2};
  IS             bcPoints[2]         = {NULL, NULL};
  PetscInt       numDof[NUM_FIELDS*(SPATIAL_DIM_0+1)];
  PetscInt       f, d;
  PetscBool      view;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (dim != SPATIAL_DIM_0) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Spatial dimension %d should be %d", dim, SPATIAL_DIM_0);
  if (dim != SPATIAL_DIM_1) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Spatial dimension %d should be %d", dim, SPATIAL_DIM_1);
  for (d = 0; d <= dim; ++d) {
    numDof[0*(dim+1)+d] = numDof_0[d];
    numDof[1*(dim+1)+d] = numDof_1[d];
    numDof[2*(dim+1)+d] = numDof_2[d];
  }
  for (f = 0; f < numFields; ++f) {
    for (d = 1; d < dim; ++d) {
      if ((numDof[f*(dim+1)+d] > 0) && !user->interpolate) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  if (user->bcType == FREE_SLIP) {
    PetscInt numBoundaries, b;
    PetscInt *numBoundaryConstraints;
    IS       *boundaryPoints, *constraintIndices;

    ierr = DMPlexCreateSectionInitial(dm, dim, numFields, numComp, numDof, &section);CHKERRQ(ierr);
    /* Velocity conditions */
    ierr = CreateBoundaryPointIS(dm, &numBoundaries, &numBoundaryConstraints, &boundaryPoints, &constraintIndices);CHKERRQ(ierr);
    for (b = 0; b < numBoundaries; ++b) {
      ierr = DMPlexCreateSectionBCDof(dm, 1, &bcFields[0], &boundaryPoints[b], numBoundaryConstraints[b], section);CHKERRQ(ierr);
    }
    /* Temperature conditions */
    ierr = DMPlexGetStratumIS(dm, "marker", 1, &bcPoints[0]);CHKERRQ(ierr);
    ierr = DMPlexCreateSectionBCDof(dm, 1, &bcFields[1], &bcPoints[0], PETSC_DETERMINE, section);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
    for (b = 0; b < numBoundaries; ++b) {
      ierr = DMPlexCreateSectionBCIndicesField(dm, bcFields[0], boundaryPoints[b], constraintIndices[b], section);CHKERRQ(ierr);
    }
    ierr = DMPlexCreateSectionBCIndicesField(dm, bcFields[1], bcPoints[0], NULL, section);CHKERRQ(ierr);
    ierr = DMPlexCreateSectionBCIndices(dm, section);CHKERRQ(ierr);
  } else {
    if (user->bcType == DIRICHLET) {
      numBC       = 2;
      ierr        = DMPlexGetStratumIS(dm, "marker", 1, &bcPoints[0]);CHKERRQ(ierr);
      bcPoints[1] = bcPoints[0];
      ierr        = PetscObjectReference((PetscObject) bcPoints[1]);CHKERRQ(ierr);
    }
    ierr = DMPlexCreateSection(dm, dim, numFields, numComp, numDof, numBC, bcFields, bcPoints, &section);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetFieldName(section, 0, "velocity");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 1, "pressure");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 2, "temperature");CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPoints[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPoints[1]);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-section_view", &view);CHKERRQ(ierr);
  if ((user->bcType == FREE_SLIP) && view) {
    PetscSection s, gs;

    ierr = DMGetDefaultSection(dm, &s);CHKERRQ(ierr);
    ierr = PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(dm, &gs);CHKERRQ(ierr);
    ierr = PetscSectionView(gs, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupExactSolution"
PetscErrorCode SetupExactSolution(DM dm, AppCtx *user)
{
  PetscFEM       *fem = &user->fem;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  switch (user->forcingType) {
  case FORCING_CONSTANT:
    if (user->bcType == FREE_SLIP) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Constant forcing is incompatible with freeslip boundary conditions");
    fem->f0Funcs[0] = f0_u_constant;
    break;
  case FORCING_LINEAR:
    switch (user->bcType) {
    case DIRICHLET:
    case FREE_SLIP:
      switch (user->dim) {
      case 2:
        fem->f0Funcs[0] = f0_u_linear_2d;
        break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary condition type %d", user->bcType);
    }
    break;
  case FORCING_CUBIC:
    switch (user->bcType) {
    case DIRICHLET:
    case FREE_SLIP:
      switch (user->dim) {
      case 2:
        fem->f0Funcs[0] = f0_u_cubic_2d;
        break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary condition type %d", user->bcType);
    }
    break;
  }
  fem->f0Funcs[1] = f0_p;
  fem->f0Funcs[2] = f0_T;
  fem->f1Funcs[0] = f1_u;
  fem->f1Funcs[1] = f1_p;
  fem->f1Funcs[2] = f1_T;
  fem->g0Funcs[0] = NULL;
  fem->g0Funcs[1] = NULL;
  fem->g0Funcs[2] = NULL;
  fem->g0Funcs[3] = NULL;
  fem->g0Funcs[4] = NULL;
  fem->g0Funcs[5] = NULL;
  fem->g0Funcs[6] = NULL;
  fem->g0Funcs[7] = NULL;
  fem->g0Funcs[8] = NULL;
  fem->g1Funcs[0] = NULL;
  fem->g1Funcs[1] = NULL;
  fem->g1Funcs[2] = NULL;
  fem->g1Funcs[3] = g1_pu;      /* < q, \nabla\cdot v > */
  fem->g1Funcs[4] = NULL;
  fem->g1Funcs[5] = NULL;
  fem->g1Funcs[6] = NULL;
  fem->g1Funcs[7] = NULL;
  fem->g1Funcs[8] = NULL;
  fem->g2Funcs[0] = NULL;
  fem->g2Funcs[1] = g2_up;      /* < \nabla\cdot v, p > */
  fem->g2Funcs[2] = NULL;
  fem->g2Funcs[3] = NULL;
  fem->g2Funcs[4] = NULL;
  fem->g2Funcs[5] = NULL;
  fem->g2Funcs[6] = NULL;
  fem->g2Funcs[7] = NULL;
  fem->g2Funcs[8] = NULL;
  fem->g3Funcs[0] = g3_uu;      /* < \nabla v, \nabla u + {\nabla u}^T > */
  fem->g3Funcs[1] = NULL;
  fem->g3Funcs[2] = NULL;
  fem->g3Funcs[3] = NULL;
  fem->g3Funcs[4] = NULL;
  fem->g3Funcs[5] = NULL;
  fem->g3Funcs[6] = NULL;
  fem->g3Funcs[7] = NULL;
  fem->g3Funcs[8] = g3_TT;      /* < \nabla t, \nabla T + {\nabla T}^T > */
  switch (user->forcingType) {
  case FORCING_CONSTANT:
    switch (user->bcType) {
    case DIRICHLET:
      switch (user->dim) {
      case 2:
        user->exactFuncs[0] = quadratic_u_2d;
        user->exactFuncs[1] = quadratic_v_2d;
        user->exactFuncs[2] = linear_p_2d;
        user->exactFuncs[3] = linear_T_2d;
        break;
      case 3:
        user->exactFuncs[0] = quadratic_u_3d;
        user->exactFuncs[1] = quadratic_v_3d;
        user->exactFuncs[2] = quadratic_w_3d;
        user->exactFuncs[3] = linear_p_3d;
        user->exactFuncs[4] = linear_T_2d;
        break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary condition type %d", user->bcType);
    }
    break;
  case FORCING_LINEAR:
    switch (user->bcType) {
    case DIRICHLET:
      switch (user->dim) {
      case 2:
        user->exactFuncs[0] = cubic_u_2d;
        user->exactFuncs[1] = cubic_v_2d;
        user->exactFuncs[2] = linear_p_2d;
        user->exactFuncs[3] = linear_T_2d;
        break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary condition type %d", user->bcType);
    }
    break;
  case FORCING_CUBIC:
    switch (user->bcType) {
    case DIRICHLET:
    case FREE_SLIP:
      switch (user->dim) {
      case 2:
        user->exactFuncs[0] = quintic_u_2d;
        user->exactFuncs[1] = quintic_v_2d;
        user->exactFuncs[2] = linear_p_2d;
        user->exactFuncs[3] = linear_T_2d;
        break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary condition type %d", user->bcType);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid forcing type %d", user->forcingType);
  }
  ierr = DMPlexSetFEMIntegration(dm, FEMIntegrateResidualBatch, FEMIntegrateJacobianActionBatch, FEMIntegrateJacobianBatch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateNullSpaces"
/*
  . field - The field whose diagonal block (of the Jacobian) has this null space
*/
PetscErrorCode CreateNullSpaces(DM dm, PetscInt field, MatNullSpace *nullSpace)
{
  AppCtx         *user;
  Vec            nullVec, localNullVec;
  PetscSection   section;
  PetscScalar    *a;
  PetscInt       pressure = field;
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetApplicationContext(dm, (void**) &user);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &nullVec);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localNullVec);CHKERRQ(ierr);
  ierr = VecSet(nullVec, 0.0);CHKERRQ(ierr);
  /* Put a constant in for all pressures */
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = VecGetArray(localNullVec, &a);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt fDim, off, d;

    ierr = PetscSectionGetFieldDof(section, p, pressure, &fDim);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldOffset(section, p, pressure, &off);CHKERRQ(ierr);
    for (d = 0; d < fDim; ++d) a[off+d] = 1.0;
  }
  ierr = VecRestoreArray(localNullVec, &a);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, localNullVec, INSERT_VALUES, nullVec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localNullVec, INSERT_VALUES, nullVec);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localNullVec);CHKERRQ(ierr);
  ierr = VecNormalize(nullVec, NULL);CHKERRQ(ierr);
  if (user->debug) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm), "Pressure Null Space\n");CHKERRQ(ierr);
    ierr = VecView(nullVec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &nullVec, nullSpace);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &nullVec);CHKERRQ(ierr);
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
  JacActionCtx   *ctx;
  DM             dm;
  Vec            dummy, localX, localY;
  PetscInt       N, n;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(J, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 3);
  ierr = MatShellGetContext(J, &ctx);CHKERRQ(ierr);
  dm   = ctx->dm;

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

  if (n != N) { /* X != localX */
    ierr = VecSet(localX, 0.0);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  } else {
    ierr   = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DMPlexComputeJacobianActionFEM(dm, J, localX, localY, ctx->user);CHKERRQ(ierr);
  if (n != N) {
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
      SETERRQ1(PetscObjectComm((PetscObject)J), PETSC_ERR_ARG_WRONG, "The difference with assembled multiply is too large %g", norm);
    }
    ierr = VecDestroy(&r);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  MPI_Comm       comm;
  SNES           snes;                 /* nonlinear solver */
  Vec            u,r;                  /* solution, residual vectors */
  Mat            A,J;                  /* Jacobian matrix */
  MatNullSpace   nullSpace = 0;            /* May be necessary for pressure */
  AppCtx         user;                 /* user-defined work context */
  JacActionCtx   userJ;                /* context for Jacobian MF action */
  PetscInt       its;                  /* iterations for convergence */
  PetscReal      error         = 0.0;  /* L_2 error in the solution */
  const PetscInt numComponents = NUM_BASIS_COMPONENTS_TOTAL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &user.dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, user.dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.dm, &user);CHKERRQ(ierr);

  ierr = SetupExactSolution(user.dm, &user);CHKERRQ(ierr);
  ierr = SetupQuadrature(&user);CHKERRQ(ierr);
  ierr = SetupSection(user.dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "solution");CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMSetMatType(user.dm,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.dm, &J);CHKERRQ(ierr);
  if (user.jacobianMF) {
    PetscInt M, m, N, n;

    ierr = MatGetSize(J, &M, &N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(J, &m, &n);CHKERRQ(ierr);
    ierr = MatCreate(comm, &A);CHKERRQ(ierr);
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
  ierr = DMSetNullSpaceConstructor(user.dm, 1, CreateNullSpaces);CHKERRQ(ierr);

  ierr = DMSNESSetFunctionLocal(user.dm,  (PetscErrorCode (*)(DM,Vec,Vec,void*))DMPlexComputeResidualFEM,&user);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(user.dm,  (PetscErrorCode (*)(DM,Vec,Mat,Mat,MatStructure*,void*))DMPlexComputeJacobianFEM,&user);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  {
    KSP               ksp; PC pc; Vec crd_vec;
    const PetscScalar *v;
    PetscInt          i,k,j,mlocal;
    PetscReal         *coords;

    ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(user.dm, &crd_vec);CHKERRQ(ierr);
    ierr = VecGetLocalSize(crd_vec,&mlocal);CHKERRQ(ierr);
    ierr = PetscMalloc(SPATIAL_DIM_0*mlocal*sizeof(*coords),&coords);CHKERRQ(ierr);
    ierr = VecGetArrayRead(crd_vec,&v);CHKERRQ(ierr);
    for (k=j=0; j<mlocal; j++) {
      for (i=0; i<SPATIAL_DIM_0; i++,k++) {
        coords[k] = PetscRealPart(v[k]);
      }
    }
    ierr = VecRestoreArrayRead(crd_vec,&v);CHKERRQ(ierr);
    ierr = PCSetCoordinates(pc, SPATIAL_DIM_0, mlocal, coords);CHKERRQ(ierr);
    ierr = PetscFree(coords);CHKERRQ(ierr);
  }

  ierr = DMPlexProjectFunction(user.dm, numComponents, user.exactFuncs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  if (user.showInitial) {ierr = DMVecViewLocal(user.dm, u, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
  if (user.runType == RUN_FULL) {
    PetscScalar (*initialGuess[numComponents])(const PetscReal x[]);
    PetscInt c;

    for (c = 0; c < numComponents; ++c) initialGuess[c] = zero;
    ierr = DMPlexProjectFunction(user.dm, numComponents, initialGuess, INSERT_VALUES, u);CHKERRQ(ierr);
    if (user.showInitial) {ierr = DMVecViewLocal(user.dm, u, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    if (user.debug) {
      ierr = PetscPrintf(comm, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of SNES iterations = %D\n", its);CHKERRQ(ierr);
    ierr = DMPlexComputeL2Diff(user.dm, user.q, user.exactFuncs, u, &error);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "L_2 Error: %.3g\n", error);CHKERRQ(ierr);
    if (user.showSolution) {
      ierr = PetscPrintf(comm, "Solution\n");CHKERRQ(ierr);
      ierr = VecChop(u, 3.0e-9);CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  } else {
    PetscReal res = 0.0;

    /* Check discretization error */
    ierr = PetscPrintf(comm, "Initial guess\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMPlexComputeL2Diff(user.dm, user.q, user.exactFuncs, u, &error);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "L_2 Error: %g\n", error);CHKERRQ(ierr);
    /* Check residual */
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Vec          b;
      MatStructure flag;
      MatNullSpace nullSpace2;
      PetscBool    isNull;

      ierr = CreateNullSpaces(user.dm, 1, &nullSpace2);CHKERRQ(ierr);
      ierr = MatNullSpaceTest(nullSpace2, J, &isNull);CHKERRQ(ierr);
      if (!isNull) SETERRQ(comm, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
      ierr = MatNullSpaceDestroy(&nullSpace2);CHKERRQ(ierr);

      ierr = SNESComputeJacobian(snes, u, &A, &A, &flag);CHKERRQ(ierr);
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
      ierr = MatMult(A, u, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "Au - b = Au + F(0)\n");CHKERRQ(ierr);
      ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
      ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "Linear L_2 Residual: %g\n", res);CHKERRQ(ierr);
    }
  }

  if (user.runType == RUN_FULL) {
    PetscViewer viewer;
    Vec         uLocal;
    const char *name;

    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "ex31_sol.vtk");CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);

    ierr = DMGetLocalVector(user.dm, &uLocal);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) u, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) uLocal, name);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(user.dm, u, INSERT_VALUES, uLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user.dm, u, INSERT_VALUES, uLocal);CHKERRQ(ierr);
    ierr = VecView(uLocal, viewer);CHKERRQ(ierr);
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
