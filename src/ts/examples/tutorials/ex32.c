//TODO: I don't think the boundary conditions are correct on subsequent solves since they depend on time

//TODO: Output iterations as HDF5 and visualize

static char help[] = "The Heat Equation in 2d and 3d with simplicial finite elements.\n\
We solve the heat equation in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports discretized auxiliary fields (conductivity) as well as\n\
multilevel nonlinear solvers.\n\n\n";

#include <petscdmplex.h>
#include <petscts.h>
#include <petscds.h>
#include <petscviewerhdf5.h>

PetscInt spatialDim = 0;

typedef enum {NEUMANN, DIRICHLET, NONE} BCType;
typedef enum {COEFF_NONE, COEFF_ANALYTIC, COEFF_FIELD, COEFF_NONLINEAR} CoeffType;

typedef struct {
  PetscInt      debug;             /* The debugging level */
  PetscLogEvent createMeshEvent;
  PetscBool     restart;
  PetscViewer   checkpoint;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  char          filename[2048];    /* The optional ExodusII file */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
  /* Problem definition */
  BCType        bcType;
  CoeffType     variableCoefficient;
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return 0;
}

/*
  In 2D for Dirichlet conditions, we use exact solution:

    u = 2t + x^2 + y^2
    f = 2

  so that

    u_t - \Delta u + f = 2 - 4 + 2 = 0

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
  *u = 2.0*time + x[0]*x[0] + x[1]*x[1];
  return 0;
}

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = u_t[0] + 2.0;
}

static void f0_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += -n[d]*2.0*x[d];
}

static void f0_bd_zero(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

static void f1_bd_zero(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f1[])
{
  PetscInt comp;
  for (comp = 0; comp < dim; ++comp) f1[comp] = 0.0;
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/* < v, u > */
static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
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

static void f0_analytic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 6.0*(x[0] + x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
static void f1_analytic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;

  for (d = 0; d < spatialDim; ++d) f1[d] = (x[0] + x[1])*u_x[d];
}

static void f1_field_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = a[0]*u_x[d];
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
static void g3_analytic_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = x[0] + x[1];
}

static void g3_field_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
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
static void f0_analytic_nonlinear_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                    PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 16.0*(x[0]*x[0] + x[1]*x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
static void f1_analytic_nonlinear_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                    PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscScalar nu = 0.0;
  PetscInt    d;
  for (d = 0; d < spatialDim; ++d) nu += u_x[d]*u_x[d];
  for (d = 0; d < spatialDim; ++d) f1[d] = 0.5*nu*u_x[d];
}

/*
  grad (u + eps w) - grad u = eps grad w

  1/2 |grad (u + eps w)|^2 grad (u + eps w) - 1/2 |grad u|^2 grad u
= 1/2 (|grad u|^2 + 2 eps <grad u,grad w>) (grad u + eps grad w) - 1/2 |grad u|^2 grad u
= 1/2 (eps |grad u|^2 grad w + 2 eps <grad u,grad w> grad u)
= eps (1/2 |grad u|^2 grad w + grad u <grad u,grad w>)
*/
static void g3_analytic_nonlinear_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscScalar nu = 0.0;
  PetscInt    d, e;
  for (d = 0; d < spatialDim; ++d) nu += u_x[d]*u_x[d];
  for (d = 0; d < spatialDim; ++d) {
    g3[d*spatialDim+d] = 0.5*nu;
    for (e = 0; e < spatialDim; ++e) {
      g3[d*spatialDim+e] += u_x[d]*u_x[e];
    }
  }
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
static PetscErrorCode quadratic_u_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *bcTypes[3]  = {"neumann", "dirichlet", "none"};
  const char    *coeffTypes[4] = {"none", "analytic", "field", "nonlinear"};
  PetscInt       bc, coeff;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug               = 0;
  options->dim                 = 2;
  options->filename[0]         = '\0';
  options->interpolate         = PETSC_FALSE;
  options->refinementLimit     = 0.0;
  options->bcType              = DIRICHLET;
  options->variableCoefficient = COEFF_NONE;
  options->restart             = PETSC_FALSE;
  options->checkpoint          = NULL;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex32.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex32.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  ierr = PetscOptionsString("-f", "Exodus.II filename to read", "ex32.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_EXODUSII)
  if (flg) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "This option requires ExodusII support. Reconfigure using --download-exodusii");
#endif
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex32.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex32.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex32.c",bcTypes,3,bcTypes[options->bcType],&bc,NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  coeff = options->variableCoefficient;
  ierr = PetscOptionsEList("-variable_coefficient","Type of variable coefficent","ex32.c",coeffTypes,4,coeffTypes[options->variableCoefficient],&coeff,NULL);CHKERRQ(ierr);
  options->variableCoefficient = (CoeffType) coeff;

  ierr = PetscOptionsBool("-restart", "Read in the mesh and solution from a file", "ex32.c", options->restart, &options->restart, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);

  if (options->restart) {
    ierr = PetscViewerCreate(comm, &options->checkpoint);CHKERRQ(ierr);
    ierr = PetscViewerSetType(options->checkpoint, PETSCVIEWERHDF5);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(options->checkpoint, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(options->checkpoint, options->filename);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
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
    ierr = DMPlexCreateBoxMesh(comm, dim, 2, interpolate, dm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  } else if (user->checkpoint) {
    ierr = DMCreate(comm, dm);CHKERRQ(ierr);
    ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
    ierr = DMLoad(*dm, user->checkpoint);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  } else {
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
    /* Must have boundary marker for Dirichlet conditions */
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
    DMLabel label;

    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, label);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupProblem"
PetscErrorCode SetupProblem(DM dm, AppCtx *user)
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
PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*matFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {nu_2d};
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
PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm   = dm;
  const PetscInt dim   = user->dim;
  const PetscInt id    = 1;
  PetscFE        feAux = NULL;
  PetscFE        feBd  = NULL;
  PetscFE        fe;
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  if (user->bcType == NEUMANN) {
    ierr = PetscFECreateDefault(dm, dim-1, 1, PETSC_TRUE, "bd_", -1, &feBd);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) feBd, "potential");CHKERRQ(ierr);
  }
  if (user->variableCoefficient == COEFF_FIELD) {
    PetscQuadrature q;

    ierr = PetscFECreateDefault(dm, dim, 1, PETSC_TRUE, "mat_", -1, &feAux);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscFESetQuadrature(feAux, q);CHKERRQ(ierr);
  }
  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscDSSetBdDiscretization(prob, 0, (PetscObject) feBd);CHKERRQ(ierr);
    if (feAux) {
      DM      dmAux;
      PetscDS probAux;

      ierr = DMClone(cdm, &dmAux);CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(cdm, dmAux);CHKERRQ(ierr);
      ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
      ierr = PetscDSSetDiscretization(probAux, 0, (PetscObject) feAux);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
      ierr = SetupMaterial(cdm, dmAux, user);CHKERRQ(ierr);
      ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
    }
    ierr = SetupProblem(cdm, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(cdm, user->bcType == DIRICHLET ? PETSC_TRUE : PETSC_FALSE, "wall", user->bcType == NEUMANN ? "boundary" : "marker", 0, 0, NULL, (void (*)()) user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feBd);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckMassMatrix"
PetscErrorCode CheckMassMatrix(TS ts, AppCtx *user)
{
  DM             dm, dm_mass;
  PetscDS        prob_mass;
  PetscFE        fe;
  Mat            M;
  Vec            x, y, z;
  PetscReal      norm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, user->dim, 1, PETSC_TRUE, "mass_", -1, &fe);CHKERRQ(ierr);
  ierr = DMClone(dm, &dm_mass);CHKERRQ(ierr);
  ierr = DMSetNumFields(dm_mass, 1);CHKERRQ(ierr);
  ierr = PetscDSCreate(PetscObjectComm((PetscObject) ts), &prob_mass);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob_mass, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob_mass, 0, 0, g0_uu, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMSetDS(dm_mass, prob_mass);CHKERRQ(ierr);
  ierr = PetscDSDestroy(&prob_mass);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm_mass, &M);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(M, "M_");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm_mass, &x);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm_mass, x, M, M, NULL);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M, (PetscObject) M, "-mat_view");CHKERRQ(ierr);

  ierr = VecSet(x, 1.0);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm_mass, &y);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm_mass, &z);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianActionFEM(dm_mass, x, x, y, NULL);CHKERRQ(ierr);
  ierr = VecViewFromOptions(y, NULL, "-mass_y_vec_view");CHKERRQ(ierr);
  ierr = MatMult(M, x, z);CHKERRQ(ierr);
  ierr = VecAXPY(z, -1.0, y);CHKERRQ(ierr);
  ierr = VecViewFromOptions(y, NULL, "-mass_diff_vec_view");CHKERRQ(ierr);
  ierr = VecNorm(z, NORM_INFINITY, &norm);CHKERRQ(ierr);
  if (norm > 1.0e-10) SETERRQ(PetscObjectComm((PetscObject) ts), PETSC_ERR_PLIB, "Incorrect matrix-free action computed");

  ierr = DMRestoreGlobalVector(dm_mass, &x);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm_mass, &y);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm_mass, &z);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = DMDestroy(&dm_mass);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;            /* Problem specification */
  TS             ts;            /* timestepper */
  Vec            u,r;           /* solution, residual vectors */
  AppCtx         user;          /* user-defined work context */
  PetscReal      error   = 0.0; /* L_2 error in the solution */
  PetscReal      t       = 0.0;
  void          *ctxs[1] = {&t};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = PetscMalloc(1 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "temperature");CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  if (user.bcType == NEUMANN) {
    PetscObject  temp;
    MatNullSpace nullSpace;

    ierr = DMGetField(dm, 0, &temp);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_TRUE, 0, NULL, &nullSpace);CHKERRQ(ierr);
    ierr = PetscObjectCompose(temp, "nullspace", (PetscObject) nullSpace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  }

  ierr = DMTSSetIFunctionLocal(dm,  (PetscErrorCode (*)(DM,PetscReal,Vec,Vec,Vec,void*)) DMPlexTSComputeIFunctionFEM, &user);CHKERRQ(ierr);
  //ierr = DMTSSetIJacobianLocal(dm,  (PetscErrorCode (*)(DM,Vec,Mat,Mat,void*)) DMPlexTSComputeIJacobianFEM, &user);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetDuration(ts, 1, 2.0);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts, 0.0, 0.01);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, ctxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  if (user.checkpoint) {
#if defined(PETSC_HAVE_HDF5)
    ierr = PetscViewerHDF5PushGroup(user.checkpoint, "/fields");CHKERRQ(ierr);
    ierr = VecLoad(u, user.checkpoint);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(user.checkpoint);CHKERRQ(ierr);
#endif
  }
  ierr = PetscViewerDestroy(&user.checkpoint);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-init_vec_view");CHKERRQ(ierr);
  ierr = CheckMassMatrix(ts, &user);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u, user.exactFuncs, ctxs);CHKERRQ(ierr);
  {
    PetscErrorCode (*initialGuess[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {zero};

    ierr = DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    if (user.debug) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
      ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = TSSolve(ts, u);CHKERRQ(ierr);
    ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
    ierr = DMComputeL2Diff(dm, t, user.exactFuncs, ctxs, u, &error);CHKERRQ(ierr);
    if (error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
    else                 {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);}
    ierr = VecViewFromOptions(u, NULL, "-final_vec_view");CHKERRQ(ierr);
  }
  ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
