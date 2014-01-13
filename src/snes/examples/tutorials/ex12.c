static char help[] = "Poisson Problem in 2d and 3d with simplicial finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#endif

#define NUM_FIELDS 1
PetscInt spatialDim = 0;

typedef enum {NEUMANN, DIRICHLET, NONE} BCType;
typedef enum {RUN_FULL, RUN_TEST, RUN_PERF} RunType;
typedef enum {COEFF_NONE, COEFF_ANALYTIC, COEFF_FIELD} CoeffType;

typedef struct {
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
  char          filename[2048];    /* The optional ExodusII file */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  PetscBool     refinementUniform; /* Uniformly refine the mesh */
  PetscInt      refinementRounds;  /* The number of uniform refinements */
  char          partitioner[2048]; /* The graph partitioner */
  /* Element definition */
  PetscFE       fe[NUM_FIELDS];
  PetscFE       feBd[NUM_FIELDS];
  PetscFE       feAux[1];
  /* Problem definition */
  void (*f0Funcs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[]); /* f0_u(x,y,z), and f0_p(x,y,z) */
  void (*f1Funcs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[]); /* f1_u(x,y,z), and f1_p(x,y,z) */
  void (*g0Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[]); /* g0_uu(x,y,z), g0_up(x,y,z), g0_pu(x,y,z), and g0_pp(x,y,z) */
  void (*g1Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g1[]); /* g1_uu(x,y,z), g1_up(x,y,z), g1_pu(x,y,z), and g1_pp(x,y,z) */
  void (*g2Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[]); /* g2_uu(x,y,z), g2_up(x,y,z), g2_pu(x,y,z), and g2_pp(x,y,z) */
  void (*g3Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[]); /* g3_uu(x,y,z), g3_up(x,y,z), g3_pu(x,y,z), and g3_pp(x,y,z) */
  void (**exactFuncs)(const PetscReal x[], PetscScalar *u, void *ctx); /* The exact solution function u(x,y,z), v(x,y,z), and p(x,y,z) */
  void (*f0BdFuncs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f0[]); /* f0_u(x,y,z), and f0_p(x,y,z) */
  void (*f1BdFuncs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f1[]); /* f1_u(x,y,z), and f1_p(x,y,z) */
  BCType    bcType;
  CoeffType variableCoefficient;
} AppCtx;

void zero(const PetscReal coords[], PetscScalar *u, void *ctx)
{
  *u = 0.0;
}

/*
  In 2D for Dirichlet conditions, we use exact solution:

    u = x^2 + y^2
    f = 4

  so that

    -\Delta u + f = -4 + 4 = 0

  For Neumann conditions, we have

    -\nabla u \cdot -\hat y |_{y=0} =  (2y)|_{y=0} =  0 (bottom)
    -\nabla u \cdot  \hat y |_{y=1} = -(2y)|_{y=1} = -2 (top)
    -\nabla u \cdot -\hat x |_{x=0} =  (2x)|_{x=0} =  0 (left)
    -\nabla u \cdot  \hat x |_{x=1} = -(2x)|_{x=1} = -2 (right)

  Which we can express as

    \nabla u \cdot  \hat n|_\Gamma = {2 x, 2 y} \cdot \hat n = 2 (x + y)
*/
void quadratic_u_2d(const PetscReal x[], PetscScalar *u, void *ctx)
{
  *u = x[0]*x[0] + x[1]*x[1];
}

void f0_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 4.0;
}

void f0_bd_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  PetscInt  d;
  for (d = 0, f0[0] = 0.0; d < spatialDim; ++d) f0[0] += -n[d]*2.0*x[d];
}

void f0_bd_zero(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

void f1_bd_zero(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f1[])
{
  const PetscInt Ncomp = spatialDim;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) f1[comp] = 0.0;
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) f1[d] = gradU[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) g3[d*spatialDim+d] = 1.0;
}

/*
  In 2D for Dirichlet conditions with a variable coefficient, we use exact solution:

    u  = x^2 + y^2
    f  = 6 (x + y)
    nu = (x + y)

  so that

    -\div \nu \grad u + f = -6 (x + y) + 6 (x + y) = 0
*/
void nu_2d(const PetscReal x[], PetscScalar *u, void *ctx)
{
  *u = x[0] + x[1];
}

void f0_analytic_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 6.0*(x[0] + x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_analytic_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) f1[d] = (x[0] + x[1])*gradU[d];
}
void f1_field_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) f1[d] = a[0]*gradU[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_analytic_uu(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) g3[d*spatialDim+d] = x[0] + x[1];
}
void g3_field_uu(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) g3[d*spatialDim+d] = a[0];
}

/*
  In 3D for Dirichlet conditions we use exact solution:

    u = x^2 + y^2 + z^2
    f = 6

  so that

    -\Delta u + f = -6 + 6 = 0

  For Neumann conditions, we have

    -\nabla u \cdot -\hat z |_{z=0} =  (2z)|_{z=0} =  0 (bottom)
    -\nabla u \cdot  \hat z |_{z=1} = -(2z)|_{z=1} = -2 (top)
    -\nabla u \cdot -\hat y |_{y=0} =  (2y)|_{y=0} =  0 (front)
    -\nabla u \cdot  \hat y |_{y=1} = -(2y)|_{y=1} = -2 (back)
    -\nabla u \cdot -\hat x |_{x=0} =  (2x)|_{x=0} =  0 (left)
    -\nabla u \cdot  \hat x |_{x=1} = -(2x)|_{x=1} = -2 (right)

  Which we can express as

    \nabla u \cdot  \hat n|_\Gamma = {2 x, 2 y, 2z} \cdot \hat n = 2 (x + y + z)
*/
void quadratic_u_3d(const PetscReal x[], PetscScalar *u, void *ctx)
{
  *u = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *bcTypes[3]  = {"neumann", "dirichlet", "none"};
  const char    *runTypes[3] = {"full", "test", "perf"};
  const char    *coeffTypes[3] = {"none", "analytic", "field"};
  PetscInt       bc, run, coeff;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug               = 0;
  options->runType             = RUN_FULL;
  options->dim                 = 2;
  options->filename[0]         = '\0';
  options->interpolate         = PETSC_FALSE;
  options->refinementLimit     = 0.0;
  options->refinementUniform   = PETSC_FALSE;
  options->refinementRounds    = 1;
  options->bcType              = DIRICHLET;
  options->variableCoefficient = COEFF_NONE;
  options->jacobianMF          = PETSC_FALSE;
  options->showInitial         = PETSC_FALSE;
  options->showSolution        = PETSC_FALSE;

  options->fem.f0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f0Funcs;
  options->fem.f1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f1Funcs;
  options->fem.g0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g0Funcs;
  options->fem.g1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g1Funcs;
  options->fem.g2Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g2Funcs;
  options->fem.g3Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g3Funcs;
  options->fem.f0BdFuncs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[])) &options->f0BdFuncs;
  options->fem.f1BdFuncs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[])) &options->f1BdFuncs;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex12.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex12.c", runTypes, 3, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);

  options->runType = (RunType) run;

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex12.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  ierr = PetscOptionsString("-f", "Exodus.II filename to read", "ex12.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_EXODUSII)
  if (flg) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "This option requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex12.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex12.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-refinement_uniform", "Uniformly refine the mesh", "ex52.c", options->refinementUniform, &options->refinementUniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-refinement_rounds", "The number of uniform refinements", "ex52.c", options->refinementRounds, &options->refinementRounds, NULL);CHKERRQ(ierr);
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex12.c",bcTypes,3,bcTypes[options->bcType],&bc,NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  coeff = options->variableCoefficient;
  ierr = PetscOptionsEList("-variable_coefficient","Type of variable coefficent","ex12.c",coeffTypes,3,coeffTypes[options->variableCoefficient],&coeff,NULL);CHKERRQ(ierr);
  options->variableCoefficient = (CoeffType) coeff;

  ierr = PetscOptionsBool("-jacobian_mf", "Calculate the action of the Jacobian on the fly", "ex12.c", options->jacobianMF, &options->jacobianMF, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex12.c", options->showInitial, &options->showInitial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex12.c", options->showSolution, &options->showSolution, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim               = user->dim;
  const char    *filename          = user->filename;
  PetscBool      interpolate       = user->interpolate;
  PetscReal      refinementLimit   = user->refinementLimit;
  PetscBool      refinementUniform = user->refinementUniform;
  PetscInt       refinementRounds  = user->refinementRounds;
  const char    *partitioner       = user->partitioner;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) {
    DMLabel label;

    ierr = DMPlexCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
    ierr = DMPlexGetLabel(*dm, "marker", &label);CHKERRQ(ierr);
    if (label) {ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);}
  } else {
#if defined(PETSC_HAVE_EXODUSII)
    int        CPU_word_size = 0, IO_word_size = 0, exoid;
    float       version;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    if (!rank) {
      exoid = ex_open(filename, EX_READ, &CPU_word_size, &IO_word_size, &version);
      if (exoid <= 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open(\"%s\",...) did not return a valid file ID", filename);
    } else exoid = -1;                 /* Not used */
    ierr = DMPlexCreateExodus(comm, exoid, interpolate, dm);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
    if (!rank) {ierr = ex_close(exoid);CHKERRQ(ierr);}
    /* Must have boundary marker for Dirichlet conditions */
#endif
  }
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      const char *name;

      ierr = PetscObjectGetName((PetscObject) *dm,         &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) refinedMesh,  name);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, partitioner, 0, NULL, &distributedMesh);CHKERRQ(ierr);
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
#define __FUNCT__ "SetupMaterialElement"
PetscErrorCode SetupMaterialElement(DM dm, AppCtx *user)
{
  const PetscInt  dim = user->dim;
  const char     *prefix = "mat_";
  PetscFE         fem;
  PetscQuadrature q;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        order, qorder;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (user->variableCoefficient != COEFF_FIELD) {user->fem.feAux = NULL; user->feAux[0] = NULL; PetscFunctionReturn(0);}
  /* Create space */
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix);CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(P, &order);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, prefix);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, PETSC_TRUE, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), &fem);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) fem, prefix);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(fem);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(fem, 1);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature, must agree with solution quadrature */
  ierr = PetscFEGetBasisSpace(user->fe[0], &P);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(P, &qorder);CHKERRQ(ierr);
  ierr = PetscDTGaussJacobiQuadrature(dim, qorder, -1.0, 1.0, &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fem, q);CHKERRQ(ierr);
  user->feAux[0]  = fem;
  user->fem.feAux = user->feAux;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupBdElement"
PetscErrorCode SetupBdElement(DM dm, AppCtx *user)
{
  const PetscInt  dim    = user->dim-1;
  const char     *prefix = "bd_";
  PetscFE         fem;
  PetscQuadrature q;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        order;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (user->bcType != NEUMANN) {user->fem.feBd = NULL; user->feBd[0] = NULL; PetscFunctionReturn(0);}
  /* Create space */
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix);CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(P, &order);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, prefix);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, PETSC_TRUE, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), &fem);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) fem, prefix);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(fem);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(fem, 1);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature */
  ierr = PetscDTGaussJacobiQuadrature(dim, order, -1.0, 1.0, &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fem, q);CHKERRQ(ierr);
  user->feBd[0] = fem;
  user->fem.feBd = user->feBd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyElement"
PetscErrorCode DestroyElement(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFEDestroy(&user->fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user->feBd[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user->feAux[0]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupExactSolution"
PetscErrorCode SetupExactSolution(DM dm, AppCtx *user)
{
  PetscFEM *fem = &user->fem;

  PetscFunctionBeginUser;
  switch (user->variableCoefficient) {
  case COEFF_NONE:
    fem->f0Funcs[0] = f0_u;
    fem->f1Funcs[0] = f1_u;
    fem->g0Funcs[0] = NULL;
    fem->g1Funcs[0] = NULL;
    fem->g2Funcs[0] = NULL;
    fem->g3Funcs[0] = g3_uu;      /* < \nabla v, \nabla u > */
    break;
  case COEFF_ANALYTIC:
    fem->f0Funcs[0] = f0_analytic_u;
    fem->f1Funcs[0] = f1_analytic_u;
    fem->g0Funcs[0] = NULL;
    fem->g1Funcs[0] = NULL;
    fem->g2Funcs[0] = NULL;
    fem->g3Funcs[0] = g3_analytic_uu;
    break;
  case COEFF_FIELD:
    fem->f0Funcs[0] = f0_analytic_u;
    fem->f1Funcs[0] = f1_field_u;
    fem->g0Funcs[0] = NULL;
    fem->g1Funcs[0] = NULL;
    fem->g2Funcs[0] = NULL;
    fem->g3Funcs[0] = g3_analytic_uu /*g3_field_uu*/;
    break;
  default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid variable coefficient type %d", user->variableCoefficient);
  }
  fem->f0BdFuncs[0] = f0_bd_zero;
  fem->f1BdFuncs[0] = f1_bd_zero;
  switch (user->dim) {
  case 2:
    user->exactFuncs[0] = quadratic_u_2d;
    if (user->bcType == NEUMANN) fem->f0BdFuncs[0] = f0_bd_u;
    break;
  case 3:
    user->exactFuncs[0] = quadratic_u_3d;
    if (user->bcType == NEUMANN) fem->f0BdFuncs[0] = f0_bd_u;
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
PetscErrorCode SetupSection(DM dm, AppCtx *user)
{
  PetscSection    section;
  DMLabel         label;
  PetscInt        dim         = user->dim;
  const char     *bdLabel     = user->bcType == NEUMANN   ? "boundary" : "marker";
  PetscInt        numBC       = user->bcType == DIRICHLET ? 1 : 0;
  PetscInt        bcFields[1] = {0};
  IS              bcPoints[1] = {NULL};
  PetscInt        numComp[1];
  const PetscInt *numDof;
  PetscBool       has;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscFEGetNumComponents(user->fe[0], &numComp[0]);CHKERRQ(ierr);
  ierr = PetscFEGetNumDof(user->fe[0], &numDof);CHKERRQ(ierr);
  ierr = DMPlexHasLabel(dm, bdLabel, &has);CHKERRQ(ierr);
  if (!has) {
    ierr = DMPlexCreateLabel(dm, bdLabel);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, label);CHKERRQ(ierr);
  }
  ierr = DMPlexGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
  if (user->bcType == DIRICHLET) {ierr  = DMPlexGetStratumIS(dm, bdLabel, 1, &bcPoints[0]);CHKERRQ(ierr);}
  ierr = DMPlexCreateSection(dm, dim, NUM_FIELDS, numComp, numDof, numBC, bcFields, bcPoints, &section);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 0, "potential");CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  if (user->bcType == DIRICHLET) {ierr = ISDestroy(&bcPoints[0]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupMaterialSection"
PetscErrorCode SetupMaterialSection(DM dm, AppCtx *user)
{
  PetscSection    section;
  PetscInt        dim   = user->dim;
  PetscInt        numBC = 0;
  PetscInt        numComp[1];
  const PetscInt *numDof;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  if (user->variableCoefficient != COEFF_FIELD) PetscFunctionReturn(0);
  ierr = PetscFEGetNumComponents(user->feAux[0], &numComp[0]);CHKERRQ(ierr);
  ierr = PetscFEGetNumDof(user->feAux[0], &numDof);CHKERRQ(ierr);
  ierr = DMPlexCreateSection(dm, dim, 1, numComp, numDof, numBC, NULL, NULL, &section);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupMaterial"
PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  void (*matFuncs[1])(const PetscReal x[], PetscScalar *u, void *ctx) = {nu_2d};
  Vec            nu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (user->variableCoefficient != COEFF_FIELD) PetscFunctionReturn(0);
  ierr = DMCreateLocalVector(dmAux, &nu);CHKERRQ(ierr);
  ierr = DMPlexProjectFunctionLocal(dmAux, user->feAux, matFuncs, NULL, INSERT_ALL_VALUES, nu);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) nu);CHKERRQ(ierr);
  ierr = VecDestroy(&nu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;          /* Problem specification */
  DM             dmAux;       /* Material specification */
  SNES           snes;        /* nonlinear solver */
  Vec            u,r;         /* solution, residual vectors */
  Mat            A,J;         /* Jacobian matrix */
  MatNullSpace   nullSpace;   /* May be necessary for Neumann conditions */
  AppCtx         user;        /* user-defined work context */
  JacActionCtx   userJ;       /* context for Jacobian MF action */
  PetscInt       its;         /* iterations for convergence */
  PetscReal      error = 0.0; /* L_2 error in the solution */
  PetscInt       numComponents;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);

  ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, dmAux);CHKERRQ(ierr);
  ierr = SetupElement(dm, &user);CHKERRQ(ierr);
  ierr = SetupBdElement(dm, &user);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(user.fe[0], &numComponents);CHKERRQ(ierr);
  ierr = PetscMalloc(NUM_FIELDS * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs);CHKERRQ(ierr);
  user.fem.bcFuncs = user.exactFuncs;
  user.fem.bcCtxs = NULL;
  ierr = SetupExactSolution(dm, &user);CHKERRQ(ierr);
  ierr = SetupSection(dm, &user);CHKERRQ(ierr);
  ierr = SetupMaterialElement(dmAux, &user);CHKERRQ(ierr);
  ierr = SetupMaterialSection(dmAux, &user);CHKERRQ(ierr);
  ierr = SetupMaterial(dm, dmAux, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dmAux);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMSetMatType(dm,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  if (user.jacobianMF) {
    PetscInt M, m, N, n;

    ierr = MatGetSize(J, &M, &N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(J, &m, &n);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetSizes(A, m, n, M, N);CHKERRQ(ierr);
    ierr = MatSetType(A, MATSHELL);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
#if 0
    ierr = MatShellSetOperation(A, MATOP_MULT, (void (*)(void))FormJacobianAction);CHKERRQ(ierr);
#endif

    userJ.dm   = dm;
    userJ.J    = J;
    userJ.user = &user;

    ierr = DMCreateLocalVector(dm, &userJ.u);CHKERRQ(ierr);
    ierr = DMPlexProjectFunctionLocal(dm, user.fe, user.exactFuncs, NULL, INSERT_BC_VALUES, userJ.u);CHKERRQ(ierr);
    ierr = MatShellSetContext(A, &userJ);CHKERRQ(ierr);
  } else {
    A = J;
  }
  if (user.bcType == NEUMANN) {
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullSpace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J, nullSpace);CHKERRQ(ierr);
    if (A != J) {
      ierr = MatSetNullSpace(A, nullSpace);CHKERRQ(ierr);
    }
  }

  ierr = DMSNESSetFunctionLocal(dm,  (PetscErrorCode (*)(DM,Vec,Vec,void*)) DMPlexComputeResidualFEM, &user);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(dm,  (PetscErrorCode (*)(DM,Vec,Mat,Mat,MatStructure*,void*)) DMPlexComputeJacobianFEM, &user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, NULL, NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMPlexProjectFunction(dm, user.fe, user.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  if (user.showInitial) {
    Vec lv;
    ierr = DMGetLocalVector(dm, &lv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, u, INSERT_VALUES, lv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, u, INSERT_VALUES, lv);CHKERRQ(ierr);
    ierr = DMPrintLocalVec(dm, "Local function", 1.0e-10, lv);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &lv);CHKERRQ(ierr);
  }
  if (user.runType == RUN_FULL) {
    void (*initialGuess[numComponents])(const PetscReal x[], PetscScalar *, void *ctx);
    PetscInt c;

    for (c = 0; c < numComponents; ++c) initialGuess[c] = zero;
    ierr = DMPlexProjectFunction(dm, user.fe, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %D\n", its);CHKERRQ(ierr);
    ierr = DMPlexComputeL2Diff(dm, user.fe, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
    else                 {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);}
    if (user.showSolution) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution\n");CHKERRQ(ierr);
      ierr = VecChop(u, 3.0e-9);CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  } else if (user.runType == RUN_PERF) {
    PetscReal res = 0.0;

    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
  } else {
    PetscReal res = 0.0;

    /* Check discretization error */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMPlexComputeL2Diff(dm, user.fe, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
    else                 {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);}
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
      MatStructure flag;

      ierr = SNESComputeJacobian(snes, u, &A, &A, &flag);CHKERRQ(ierr);
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

  if (user.bcType == NEUMANN) {
    ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  }
  if (user.jacobianMF) {
    ierr = VecDestroy(&userJ.u);CHKERRQ(ierr);
  }
  if (A != J) {ierr = MatDestroy(&A);CHKERRQ(ierr);}
  ierr = DestroyElement(&user);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
