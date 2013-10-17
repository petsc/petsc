static char help[] = "Check that a particular FIAT-style header gives accurate function representations\n\n";

#include <petscdmplex.h>
#include <petscdmda.h>
#include <petscfe.h>

typedef struct {
  PetscFEM  fem;               /* REQUIRED to use DMPlexComputeResidualFEM() */
  PetscInt  debug;             /* The debugging level */
  /* Domain and mesh definition */
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool simplex;           /* Flag for simplex or tensor product mesh */
  PetscBool interpolate;       /* Generate intermediate mesh elements */
  PetscReal refinementLimit;   /* The largest allowable cell volume */
  /* Element definition */
  PetscInt  qorder;            /* Order of the quadrature */
  PetscInt  numComponents;     /* Number of field components */
  PetscFE   fe;                /* The finite element */
  /* Testing space */
  PetscInt  porder;            /* Order of polynomials to test */
} AppCtx;

static int spdim = 1;

void constant(const PetscReal coords[], PetscScalar *u)
{
  PetscInt d;
  for (d = 0; d < spdim; ++d) u[d] = 1.0;
}

void linear(const PetscReal coords[], PetscScalar *u)
{
  PetscInt d;
  for (d = 0; d < spdim; ++d) u[d] = coords[d];
}

void quadratic(const PetscReal coords[], PetscScalar *u)
{
  if (spdim > 2)      {u[0] = coords[0]*coords[1]; u[1] = coords[1]*coords[2]; u[2] = coords[2]*coords[0];}
  else if (spdim > 1) {u[0] = coords[0]*coords[0]; u[1] = coords[0]*coords[1];}
  else if (spdim > 0) {u[0] = coords[0]*coords[0];}
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->simplex         = PETSC_TRUE;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;
  options->qorder          = 0;
  options->numComponents   = 1;
  options->porder          = 0;

  ierr = PetscOptionsBegin(comm, "", "Projection Test Options", "DMPlex");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex3.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex3.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Flag for simplices or hexhedra", "ex3.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex3.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex3.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qorder", "The quadrature order", "ex3.c", options->qorder, &options->qorder, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_comp", "The number of field components", "ex3.c", options->numComponents, &options->numComponents, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-porder", "The order of polynomials to test", "ex3.c", options->porder, &options->porder, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  spdim = options->dim;
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  const char    *partitioner     = "chaco";
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (user->simplex) {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    ierr = DMPlexCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
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
    ierr = PetscObjectSetName((PetscObject) *dm, "Simplical Mesh");CHKERRQ(ierr);
  } else {
    switch (user->dim) {
    case 2:
      ierr = DMDACreate2d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX, -3, -3, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, dm);CHKERRQ(ierr);
      ierr = DMDASetVertexCoordinates(*dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot create structured mesh of dimension %d", dim);
    }
    ierr = PetscObjectSetName((PetscObject) *dm, "Hexahedral Mesh");CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupElement"
PetscErrorCode SetupElement(DM dm, AppCtx *user)
{
  PetscFE         fem;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscReal      *B, *D;
  PetscQuadrature q;
  const PetscInt  dim   = user->dim;
  PetscInt        order, numBasisFunc, numBasisComp;
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
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, user->simplex, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create quadrature */
  ierr = PetscDTGaussJacobiQuadrature(dim, PetscMax(user->qorder, order), -1.0, 1.0, &q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), &fem);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(fem, user->numComponents);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fem, q);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  ierr = PetscFEGetDimension(fem, &numBasisFunc);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &numBasisComp);CHKERRQ(ierr);
  ierr = PetscFEGetDefaultTabulation(fem, &B, &D, NULL);CHKERRQ(ierr);
  user->fe = fem;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
PetscErrorCode SetupSection(DM dm, AppCtx *user)
{
  PetscSection    section;
  const PetscInt  numFields   = 1;
  PetscInt        dim         = user->dim;
  PetscInt        numBC       = 0;
  PetscInt        bcFields[1] = {0};
  IS              bcPoints[1] = {NULL};
  const PetscInt *numFieldDof[1];
  PetscInt        numComp[1];
  PetscInt       *numDof;
  PetscInt        f, d;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetNumComponents(user->fe, &numComp[0]);CHKERRQ(ierr);
  ierr = PetscFEGetNumDof(user->fe, &numFieldDof[0]);CHKERRQ(ierr);
  ierr = PetscMalloc(1*(dim+1) * sizeof(PetscInt), &numDof);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    numDof[0*(dim+1)+d] = numFieldDof[0][d];
  }
  for (f = 0; f < numFields; ++f) {
    for (d = 1; d < dim; ++d) {
      if ((numDof[f*(dim+1)+d] > 0) && !user->interpolate) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  if (user->simplex) {
    ierr = DMPlexCreateSection(dm, dim, numFields, numComp, numDof, numBC, bcFields, bcPoints, &section);CHKERRQ(ierr);
  } else {
    ierr = DMDACreateSection(dm, numComp, numDof, NULL, &section);CHKERRQ(ierr);
  }
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = PetscFree(numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckFunctions"
PetscErrorCode CheckFunctions(DM dm, PetscInt order, Vec u, AppCtx *user)
{
  void          (*exactFuncs[3]) (const PetscReal x[], PetscScalar *u);
  MPI_Comm        comm;
  PetscInt        dim  = user->dim, Nc;
  PetscQuadrature fq;
  PetscReal       error, tol = 1.0e-10;
  PetscBool       isPlex, isDA;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMDA,   &isDA);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(user->fe, &fq);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(user->fe, &Nc);CHKERRQ(ierr);
  /* Setup functions to approximate */
  switch (order) {
  case 0:
    exactFuncs[0] = constant;
    break;
  case 1:
    exactFuncs[0] = linear;
    break;
  case 2:
    exactFuncs[0] = quadratic;
    break;
  default:
    SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for dimension %d order %d", dim, order);
  }
  /* Project function into FE function space */
  if (isPlex) {
    ierr = DMPlexProjectFunction(dm, &user->fe, exactFuncs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = DMDAProjectFunction(dm, &user->fe, exactFuncs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM projection routine for this type of DM");
  /* Compare approximation to exact in L_2 */
  if (isPlex) {
    ierr = DMPlexComputeL2Diff(dm, &user->fe, exactFuncs, u, &error);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = DMDAComputeL2Diff(dm, &user->fe, exactFuncs, u, &error);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM projection routine for this type of DM");
  if (error > tol) {
    ierr = PetscPrintf(comm, "Tests FAIL for order %d at tolerance %g error %g\n", order, tol, error);CHKERRQ(ierr);
    /* SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Input FIAT tabulation cannot resolve functions of order %d, error %g > %g", order, error, tol); */
  } else {
    ierr = PetscPrintf(comm, "Tests pass for order %d at tolerance %g\n", order, tol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  Vec            u;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SetupElement(dm, &user);CHKERRQ(ierr);
  ierr = SetupSection(dm, &user);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = CheckFunctions(dm, user.porder, u, &user);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user.fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
