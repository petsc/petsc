static char help[] = "Poisson Problem in 2d and 3d with simplicial finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports discretized auxiliary fields (conductivity) as well as\n\
multilevel nonlinear solvers.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscviewerhdf5.h>

typedef enum {NEUMANN, DIRICHLET, NONE} BCType;
typedef enum {RUN_FULL, RUN_EXACT, RUN_TEST, RUN_PERF} RunType;
typedef enum {COEFF_NONE, COEFF_ANALYTIC, COEFF_FIELD, COEFF_NONLINEAR} CoeffType;

typedef struct {
  PetscInt       debug;             /* The debugging level */
  RunType        runType;           /* Whether to run tests, or solve the full problem */
  PetscBool      jacobianMF;        /* Whether to calculate the Jacobian action on the fly */
  PetscLogEvent  createMeshEvent;
  PetscBool      showInitial, showSolution, restart, check;
  /* Domain and mesh definition */
  PetscInt       dim;               /* The topological mesh dimension */
  char           filename[2048];    /* The optional ExodusII file */
  PetscBool      interpolate;       /* Generate intermediate mesh elements */
  PetscReal      refinementLimit;   /* The largest allowable cell volume */
  PetscBool      viewHierarchy;     /* Whether to view the hierarchy */
  PetscBool      simplex;           /* Simplicial mesh */
  /* Problem definition */
  BCType         bcType;
  CoeffType      variableCoefficient;
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  /* Solver */
  PC            pcmg;              /* This is needed for error monitoring */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
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
static PetscErrorCode quadratic_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = x[0]*x[0] + x[1]*x[1];
  return 0;
}

void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 4.0;
}

void f0_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
             PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += -n[d]*2.0*x[d];
}

void f0_bd_zero(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

void f1_bd_zero(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f1[])
{
  PetscInt comp;
  for (comp = 0; comp < dim; ++comp) f1[comp] = 0.0;
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

/*
  In 2D for Dirichlet conditions with a variable coefficient, we use exact solution:

    u  = x^2 + y^2
    f  = 6 (x + y)
    nu = (x + y)

  so that

    -\div \nu \grad u + f = -6 (x + y) + 6 (x + y) = 0
*/
static PetscErrorCode nu_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = x[0] + x[1];
  return 0;
}

void f0_analytic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 6.0*(x[0] + x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_analytic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = (x[0] + x[1])*u_x[d];
}

void f1_field_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = a[0]*u_x[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_analytic_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = x[0] + x[1];
}

void g3_field_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = a[0];
}

/*
  In 2D for Dirichlet conditions with a nonlinear coefficient (p-Laplacian with p = 4), we use exact solution:

    u  = x^2 + y^2
    f  = 16 (x^2 + y^2)
    nu = 1/2 |grad u|^2

  so that

    -\div \nu \grad u + f = -16 (x^2 + y^2) + 16 (x^2 + y^2) = 0
*/
void f0_analytic_nonlinear_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 16.0*(x[0]*x[0] + x[1]*x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_analytic_nonlinear_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscScalar nu = 0.0;
  PetscInt    d;
  for (d = 0; d < dim; ++d) nu += u_x[d]*u_x[d];
  for (d = 0; d < dim; ++d) f1[d] = 0.5*nu*u_x[d];
}

/*
  grad (u + eps w) - grad u = eps grad w

  1/2 |grad (u + eps w)|^2 grad (u + eps w) - 1/2 |grad u|^2 grad u
= 1/2 (|grad u|^2 + 2 eps <grad u,grad w>) (grad u + eps grad w) - 1/2 |grad u|^2 grad u
= 1/2 (eps |grad u|^2 grad w + 2 eps <grad u,grad w> grad u)
= eps (1/2 |grad u|^2 grad w + grad u <grad u,grad w>)
*/
void g3_analytic_nonlinear_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscScalar nu = 0.0;
  PetscInt    d, e;
  for (d = 0; d < dim; ++d) nu += u_x[d]*u_x[d];
  for (d = 0; d < dim; ++d) {
    g3[d*dim+d] = 0.5*nu;
    for (e = 0; e < dim; ++e) {
      g3[d*dim+e] += u_x[d]*u_x[e];
    }
  }
}

/*
  In 3D for Dirichlet conditions we use exact solution:

    u = 2/3 (x^2 + y^2 + z^2)
    f = 4

  so that

    -\Delta u + f = -2/3 * 6 + 4 = 0

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
static PetscErrorCode quadratic_u_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = 2.0*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])/3.0;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *bcTypes[3]  = {"neumann", "dirichlet", "none"};
  const char    *runTypes[4] = {"full", "exact", "test", "perf"};
  const char    *coeffTypes[4] = {"none", "analytic", "field", "nonlinear"};
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
  options->bcType              = DIRICHLET;
  options->variableCoefficient = COEFF_NONE;
  options->jacobianMF          = PETSC_FALSE;
  options->showInitial         = PETSC_FALSE;
  options->showSolution        = PETSC_FALSE;
  options->restart             = PETSC_FALSE;
  options->check               = PETSC_FALSE;
  options->viewHierarchy       = PETSC_FALSE;
  options->simplex             = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex12.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "ex12.c", runTypes, 3, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);

  options->runType = (RunType) run;

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex12.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Exodus.II filename to read", "ex12.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_EXODUSII)
  if (flg) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "This option requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex12.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex12.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex12.c",bcTypes,3,bcTypes[options->bcType],&bc,NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  coeff = options->variableCoefficient;
  ierr = PetscOptionsEList("-variable_coefficient","Type of variable coefficent","ex12.c",coeffTypes,4,coeffTypes[options->variableCoefficient],&coeff,NULL);CHKERRQ(ierr);
  options->variableCoefficient = (CoeffType) coeff;

  ierr = PetscOptionsBool("-jacobian_mf", "Calculate the action of the Jacobian on the fly", "ex12.c", options->jacobianMF, &options->jacobianMF, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex12.c", options->showInitial, &options->showInitial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex12.c", options->showSolution, &options->showSolution, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-restart", "Read in the mesh and solution from a file", "ex12.c", options->restart, &options->restart, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check", "Compare with default integration routines", "ex12.c", options->check, &options->check, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_view_hierarchy", "View the coarsened hierarchy", "ex12.c", options->viewHierarchy, &options->viewHierarchy, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex12.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateBCLabel"
static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  const char    *filename        = user->filename;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) {
    if (user->simplex) {
      ierr = DMPlexCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
    }
    else {
      PetscInt cells[3] = {1, 1, 1}; /* coarse mesh is one cell; refine from there */

      ierr = DMPlexCreateHexBoxMesh(comm, dim, cells,  DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, dm);CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
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
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  if (user->bcType == NEUMANN) {
    DMLabel   label;

    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, label);CHKERRQ(ierr);
  } else if (user->bcType == DIRICHLET) {
    PetscBool hasLabel;

    ierr = DMHasLabel(*dm,"marker",&hasLabel);CHKERRQ(ierr);
    if (!hasLabel) {ierr = CreateBCLabel(*dm, "marker");CHKERRQ(ierr);}
  }
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  if (user->viewHierarchy) {
    DM       cdm = *dm;
    PetscInt i   = 0;
    char     buf[256];

    while (cdm) {
      ierr = DMSetUp(cdm);CHKERRQ(ierr);
      ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
      ++i;
    }
    cdm = *dm;
    while (cdm) {
      PetscViewer       viewer;
      PetscBool   isHDF5, isVTK;

      --i;
      ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer,PETSCVIEWERHDF5);CHKERRQ(ierr);
      ierr = PetscViewerSetOptionsPrefix(viewer,"hierarchy_");CHKERRQ(ierr);
      ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&isHDF5);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&isVTK);CHKERRQ(ierr);
      if (isHDF5) {
        ierr = PetscSNPrintf(buf, 256, "ex12-%d.h5", i);CHKERRQ(ierr);
      }
      else if (isVTK) {
        ierr = PetscSNPrintf(buf, 256, "ex12-%d.vtu", i);CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
      }
      else {
        ierr = PetscSNPrintf(buf, 256, "ex12-%d", i);CHKERRQ(ierr);
      }
      ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer,buf);CHKERRQ(ierr);
      ierr = DMView(cdm, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupProblem"
static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  switch (user->variableCoefficient) {
  case COEFF_NONE:
    ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
    break;
  case COEFF_ANALYTIC:
    ierr = PetscDSSetResidual(prob, 0, f0_analytic_u, f1_analytic_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_analytic_uu);CHKERRQ(ierr);
    break;
  case COEFF_FIELD:
    ierr = PetscDSSetResidual(prob, 0, f0_analytic_u, f1_field_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_field_uu);CHKERRQ(ierr);
    break;
  case COEFF_NONLINEAR:
    ierr = PetscDSSetResidual(prob, 0, f0_analytic_nonlinear_u, f1_analytic_nonlinear_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_analytic_nonlinear_uu);CHKERRQ(ierr);
    break;
  default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid variable coefficient type %d", user->variableCoefficient);
  }
  switch (user->dim) {
  case 2:
    user->exactFuncs[0] = quadratic_u_2d;
    if (user->bcType == NEUMANN) {ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u, f1_bd_zero);CHKERRQ(ierr);}
    break;
  case 3:
    user->exactFuncs[0] = quadratic_u_3d;
    if (user->bcType == NEUMANN) {ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u, f1_bd_zero);CHKERRQ(ierr);}
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupMaterial"
static PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*matFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {nu_2d};
  Vec            nu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dmAux, &nu);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, matFuncs, NULL, INSERT_ALL_VALUES, nu);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) nu);CHKERRQ(ierr);
  ierr = VecDestroy(&nu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupDiscretization"
static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm   = dm;
  const PetscInt dim   = user->dim;
  const PetscInt id    = 1;
  PetscFE        feAux = NULL;
  PetscFE        feBd  = NULL;
  PetscFE        feCh  = NULL;
  PetscFE        fe;
  PetscDS        prob;
  PetscBool      simplex = user->simplex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  if (user->bcType == NEUMANN) {
    ierr = PetscFECreateDefault(dm, dim-1, 1, simplex, "bd_", -1, &feBd);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) feBd, "potential");CHKERRQ(ierr);
  }
  if (user->variableCoefficient == COEFF_FIELD) {
    PetscQuadrature q;

    ierr = PetscFECreateDefault(dm, dim, 1, simplex, "mat_", -1, &feAux);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscFESetQuadrature(feAux, q);CHKERRQ(ierr);
  }
  if (user->check) {ierr = PetscFECreateDefault(dm, dim, 1, simplex, "ch_", -1, &feCh);CHKERRQ(ierr);}
  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    DM coordDM;

    ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscDSSetBdDiscretization(prob, 0, (PetscObject) feBd);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(cdm,&coordDM);CHKERRQ(ierr);
    if (feAux) {
      DM      dmAux;
      PetscDS probAux;

      ierr = DMClone(cdm, &dmAux);CHKERRQ(ierr);
      ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
      ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
      ierr = PetscDSSetDiscretization(probAux, 0, (PetscObject) feAux);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
      ierr = SetupMaterial(cdm, dmAux, user);CHKERRQ(ierr);
      ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
    }
    if (feCh) {
      DM      dmCh;
      PetscDS probCh;

      ierr = DMClone(cdm, &dmCh);CHKERRQ(ierr);
      ierr = DMSetCoordinateDM(dmCh, coordDM);CHKERRQ(ierr);
      ierr = DMGetDS(dmCh, &probCh);CHKERRQ(ierr);
      ierr = PetscDSSetDiscretization(probCh, 0, (PetscObject) feCh);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dmCh", (PetscObject) dmCh);CHKERRQ(ierr);
      ierr = DMDestroy(&dmCh);CHKERRQ(ierr);
    }
    if (user->bcType == DIRICHLET) {
      PetscBool hasLabel;

      ierr = DMHasLabel(cdm, "marker", &hasLabel);CHKERRQ(ierr);
      if (!hasLabel) {ierr = CreateBCLabel(cdm, "marker");CHKERRQ(ierr);}
    }
    ierr = SetupProblem(cdm, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(cdm, user->bcType == DIRICHLET ? PETSC_TRUE : PETSC_FALSE, "wall", user->bcType == DIRICHLET ? "marker" : "boundary", 0, 0, NULL, (void (*)()) user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feBd);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feAux);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feCh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "petsc/private/petscimpl.h"

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorError"
/*@C
  KSPMonitorError - Outputs the error at each iteration of an iterative solver.

  Collective on KSP

  Input Parameters:
+ ksp   - the KSP
. its   - iteration number
. rnorm - 2-norm, preconditioned residual value (may be estimated).
- ctx   - monitor context

  Level: intermediate

.keywords: KSP, default, monitor, residual
.seealso: KSPMonitorSet(), KSPMonitorTrueResidualNorm(), KSPMonitorDefault()
@*/
static PetscErrorCode KSPMonitorError(KSP ksp, PetscInt its, PetscReal rnorm, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             dm;
  Vec            du, r;
  PetscInt       level = 0;
  PetscBool      hasLevel;
  PetscViewer    viewer;
  char           buf[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  /* Calculate solution */
  {
    PC        pc = user->pcmg; /* The MG PC */
    DM        fdm,  cdm;
    KSP       fksp, cksp;
    Vec       fu,   cu;
    PetscInt  levels, l;

    ierr = KSPBuildSolution(ksp, NULL, &du);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataGetInt((PetscObject) ksp, PetscMGLevelId, level, hasLevel);CHKERRQ(ierr);
    ierr = PCMGGetLevels(pc, &levels);CHKERRQ(ierr);
    ierr = PCMGGetSmoother(pc, levels-1, &fksp);CHKERRQ(ierr);
    ierr = KSPBuildSolution(fksp, NULL, &fu);CHKERRQ(ierr);
    for (l = levels-1; l > level; --l) {
      Mat R;
      Vec s;

      ierr = PCMGGetSmoother(pc, l-1, &cksp);CHKERRQ(ierr);
      ierr = KSPGetDM(cksp, &cdm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(cdm, &cu);CHKERRQ(ierr);
      ierr = PCMGGetRestriction(pc, l, &R);CHKERRQ(ierr);
      ierr = PCMGGetRScale(pc, l, &s);CHKERRQ(ierr);
      ierr = MatRestrict(R, fu, cu);CHKERRQ(ierr);
      ierr = VecPointwiseMult(cu, cu, s);CHKERRQ(ierr);
      if (l < levels-1) {ierr = DMRestoreGlobalVector(fdm, &fu);CHKERRQ(ierr);}
      fdm  = cdm;
      fu   = cu;
    }
    if (levels-1 > level) {
      ierr = VecAXPY(du, 1.0, cu);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(cdm, &cu);CHKERRQ(ierr);
    }
  }
  /* Calculate error */
  ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, user->exactFuncs, NULL, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,du);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "solution error");CHKERRQ(ierr);
  /* View error */
  ierr = PetscSNPrintf(buf, 256, "ex12-%D.h5", level);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, buf, FILE_MODE_APPEND, &viewer);CHKERRQ(ierr);
  ierr = VecView(r, viewer);CHKERRQ(ierr);
  /* Cleanup */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMonitorError"
/*@C
  SNESMonitorError - Outputs the error at each iteration of an iterative solver.

  Collective on SNES

  Input Parameters:
+ snes  - the SNES
. its   - iteration number
. rnorm - 2-norm of residual
- ctx   - user context

  Level: intermediate

.keywords: SNES, nonlinear, default, monitor, norm
.seealso: SNESMonitorDefault(), SNESMonitorSet(), SNESMonitorSolution()
@*/
static PetscErrorCode SNESMonitorError(SNES snes, PetscInt its, PetscReal rnorm, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             dm;
  Vec            u, r;
  PetscInt       level;
  PetscBool      hasLevel;
  PetscViewer    viewer;
  char           buf[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  /* Calculate error */
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "solution error");CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, user->exactFuncs, NULL, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
  ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
  /* View error */
  ierr = PetscObjectComposedDataGetInt((PetscObject) snes, PetscMGLevelId, level, hasLevel);CHKERRQ(ierr);
  ierr = PetscSNPrintf(buf, 256, "ex12-%D.h5", level);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, buf, FILE_MODE_APPEND, &viewer);CHKERRQ(ierr);
  ierr = VecView(r, viewer);CHKERRQ(ierr);
  /* Cleanup */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;          /* Problem specification */
  SNES           snes;        /* nonlinear solver */
  Vec            u,r;         /* solution, residual vectors */
  Mat            A,J;         /* Jacobian matrix */
  MatNullSpace   nullSpace;   /* May be necessary for Neumann conditions */
  AppCtx         user;        /* user-defined work context */
  JacActionCtx   userJ;       /* context for Jacobian MF action */
  PetscInt       its;         /* iterations for convergence */
  PetscReal      error = 0.0; /* L_2 error in the solution */
  PetscBool      isFAS;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = PetscMalloc(1 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
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
    ierr = DMProjectFunctionLocal(dm, 0.0, user.exactFuncs, NULL, INSERT_BC_VALUES, userJ.u);CHKERRQ(ierr);
    ierr = MatShellSetContext(A, &userJ);CHKERRQ(ierr);
  } else {
    A = J;
  }
  if (user.bcType == NEUMANN) {
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullSpace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(A, nullSpace);CHKERRQ(ierr);
  }

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, NULL, NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  if (user.restart) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewer viewer;

    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERHDF5);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, user.filename);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
    ierr = VecLoad(u, viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#endif
  }
  if (user.showInitial) {
    Vec lv;
    ierr = DMGetLocalVector(dm, &lv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, u, INSERT_VALUES, lv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, u, INSERT_VALUES, lv);CHKERRQ(ierr);
    ierr = DMPrintLocalVec(dm, "Local function", 1.0e-10, lv);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &lv);CHKERRQ(ierr);
  }
  if (user.viewHierarchy) {
    SNES      lsnes;
    KSP       ksp;
    PC        pc;
    PetscInt  numLevels, l;
    PetscBool isMG;

    ierr = PetscObjectTypeCompare((PetscObject) snes, SNESFAS, &isFAS);CHKERRQ(ierr);
    if (isFAS) {
      ierr = SNESFASGetLevels(snes, &numLevels);CHKERRQ(ierr);
      for (l = 0; l < numLevels; ++l) {
        ierr = SNESFASGetCycleSNES(snes, l, &lsnes);CHKERRQ(ierr);
        ierr = SNESMonitorSet(lsnes, SNESMonitorError, &user, NULL);CHKERRQ(ierr);
      }
    } else {
      ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject) pc, PCMG, &isMG);CHKERRQ(ierr);
      if (isMG) {
        user.pcmg = pc;
        ierr = PCMGGetLevels(pc, &numLevels);CHKERRQ(ierr);
        for (l = 0; l < numLevels; ++l) {
          ierr = PCMGGetSmootherDown(pc, l, &ksp);CHKERRQ(ierr);
          ierr = KSPMonitorSet(ksp, KSPMonitorError, &user, NULL);CHKERRQ(ierr);
        }
      }
    }
  }
  if (user.runType == RUN_FULL || user.runType == RUN_EXACT) {
    PetscErrorCode (*initialGuess[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {zero};

    if (user.runType == RUN_FULL) {
      ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    }
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %D\n", its);CHKERRQ(ierr);
    ierr = DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
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
    ierr = DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
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

      ierr = SNESComputeJacobian(snes, u, A, A);CHKERRQ(ierr);
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
  ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);

  if (user.bcType == NEUMANN) {ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);}
  if (user.jacobianMF) {ierr = VecDestroy(&userJ.u);CHKERRQ(ierr);}
  if (A != J) {ierr = MatDestroy(&A);CHKERRQ(ierr);}
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
