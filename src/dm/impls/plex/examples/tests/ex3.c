static char help[] = "Check that a DM can accurately represent and interpolate functions of a given polynomial order\n\n";

#include <petscdmplex.h>
#include <petscdm.h>
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

/* u = 1 */
void constant(const PetscReal coords[], PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < spdim; ++d) u[d] = ((PetscReal *) ctx)[d];
}
void constantDer(const PetscReal coords[], const PetscReal n[], PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < spdim; ++d) u[d] = 0.0;
}

/* u = x */
void linear(const PetscReal coords[], PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < spdim; ++d) u[d] = coords[d];
}
void linearDer(const PetscReal coords[], const PetscReal n[], PetscScalar *u, void *ctx)
{
  PetscInt d, e;
  for (d = 0; d < spdim; ++d) {
    u[d] = 0.0;
    for (e = 0; e < spdim; ++e) u[d] += (d == e ? 1.0 : 0.0) * n[e];
  }
}

/* u = x^2 or u = (x^2, xy) or u = (xy, yz, zx) */
void quadratic(const PetscReal coords[], PetscScalar *u, void *ctx)
{
  if (spdim > 2)      {u[0] = coords[0]*coords[1]; u[1] = coords[1]*coords[2]; u[2] = coords[2]*coords[0];}
  else if (spdim > 1) {u[0] = coords[0]*coords[0]; u[1] = coords[0]*coords[1];}
  else if (spdim > 0) {u[0] = coords[0]*coords[0];}
}
void quadraticDer(const PetscReal coords[], const PetscReal n[], PetscScalar *u, void *ctx)
{
  if (spdim > 2)      {u[0] = coords[1]*n[0] + coords[0]*n[1]; u[1] = coords[2]*n[1] + coords[1]*n[2]; u[2] = coords[2]*n[0] + coords[0]*n[2];}
  else if (spdim > 1) {u[0] = 2.0*coords[0]*n[0]; u[1] = coords[1]*n[0] + coords[0]*n[1];}
  else if (spdim > 0) {u[0] = 2.0*coords[0]*n[0];}
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
  options->interpolate     = PETSC_TRUE;
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
      ierr = DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, -3, -3, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, dm);CHKERRQ(ierr);
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
#define __FUNCT__ "SetupSection"
PetscErrorCode SetupSection(DM dm, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (user->simplex) {
    ierr = DMSetNumFields(dm, 1);CHKERRQ(ierr);
    ierr = DMSetField(dm, 0, (PetscObject) user->fe);CHKERRQ(ierr);
  } else {
    PetscSection    section;
    const PetscInt *numDof;
    PetscInt        numComp;

    ierr = PetscFEGetNumComponents(user->fe, &numComp);CHKERRQ(ierr);
    ierr = PetscFEGetNumDof(user->fe, &numDof);CHKERRQ(ierr);
    ierr = DMDACreateSection(dm, &numComp, numDof, NULL, &section);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckFunctions"
PetscErrorCode CheckFunctions(DM dm, PetscInt order, Vec u, AppCtx *user)
{
  void          (*exactFuncs[1]) (const PetscReal x[], PetscScalar *u, void *ctx);
  void          (*exactFuncDers[1]) (const PetscReal x[], const PetscReal n[], PetscScalar *u, void *ctx);
  PetscReal       n[3]         = {1.0, 1.0, 1.0};
  PetscReal       constants[3] = {1.0, 2.0, 3.0};
  void           *exactCtxs[3] = {NULL, NULL, NULL};
  MPI_Comm        comm;
  PetscInt        dim  = user->dim;
  PetscReal       error, errorDer, tol = 1.0e-10;
  PetscBool       isPlex, isDA;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMDA,   &isDA);CHKERRQ(ierr);
  /* Setup functions to approximate */
  switch (order) {
  case 0:
    exactFuncs[0]    = constant;
    exactFuncDers[0] = constantDer;
    exactCtxs[0]     = &constants[0];
    exactCtxs[1]     = &constants[1];
    exactCtxs[2]     = &constants[2];
    break;
  case 1:
    exactFuncs[0]    = linear;
    exactFuncDers[0] = linearDer;
    break;
  case 2:
    exactFuncs[0]    = quadratic;
    exactFuncDers[0] = quadraticDer;
    break;
  default:
    SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for dimension %d order %d", dim, order);
  }
  /* Project function into FE function space */
  if (isPlex) {
    ierr = DMPlexProjectFunction(dm, &user->fe, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = DMDAProjectFunction(dm, &user->fe, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM projection routine for this type of DM");
  /* Compare approximation to exact in L_2 */
  if (isPlex) {
    ierr = DMPlexComputeL2Diff(dm, &user->fe, exactFuncs, exactCtxs, u, &error);CHKERRQ(ierr);
    ierr = DMPlexComputeL2GradientDiff(dm, &user->fe, exactFuncDers, exactCtxs, u, n, &errorDer);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = DMDAComputeL2Diff(dm, &user->fe, exactFuncs, exactCtxs, u, &error);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM L_2 difference routine for this type of DM");
  /* Report result */
  if (error > tol) {
    ierr = PetscPrintf(comm, "Tests FAIL for order %d at tolerance %g error %g\n", order, tol, error);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(comm, "Tests pass for order %d at tolerance %g\n", order, tol);CHKERRQ(ierr);
  }
  if (errorDer > tol) {
    ierr = PetscPrintf(comm, "Tests FAIL for order %d derivatives at tolerance %g error %g\n", order, tol, errorDer);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(comm, "Tests pass for order %d derivatives at tolerance %g\n", order, tol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckInterpolation"
PetscErrorCode CheckInterpolation(DM dm, PetscBool checkRestrict, PetscInt order, AppCtx *user)
{
  void          (*exactFuncs[1]) (const PetscReal x[], PetscScalar *u, void *ctx);
  void          (*exactFuncDers[1]) (const PetscReal x[], const PetscReal n[], PetscScalar *u, void *ctx);
  PetscReal       n[3]         = {1.0, 1.0, 1.0};
  PetscReal       constants[3] = {1.0, 2.0, 3.0};
  void           *exactCtxs[3] = {NULL, NULL, NULL};
  DM              rdm, idm, fdm;
  Mat             I;
  Vec             iu, fu, scaling;
  MPI_Comm        comm;
  PetscInt        dim  = user->dim;
  PetscReal       error, errorDer, tol = 1.0e-10;
  PetscBool       isPlex, isDA;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMDA,   &isDA);CHKERRQ(ierr);
  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMRefine(dm, comm, &rdm);CHKERRQ(ierr);
  ierr = SetupSection(rdm, user);CHKERRQ(ierr);
  /* Setup functions to approximate */
  switch (order) {
  case 0:
    exactFuncs[0]    = constant;
    exactFuncDers[0] = constantDer;
    exactCtxs[0]     = &constants[0];
    exactCtxs[1]     = &constants[1];
    exactCtxs[2]     = &constants[2];
    break;
  case 1:
    exactFuncs[0]    = linear;
    exactFuncDers[0] = linearDer;
    break;
  case 2:
    exactFuncs[0]    = quadratic;
    exactFuncDers[0] = quadraticDer;
    break;
  default:
    SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for dimension %d order %d", dim, order);
  }
  idm  = checkRestrict ? rdm :  dm;
  fdm  = checkRestrict ?  dm : rdm;
  ierr = DMGetGlobalVector(idm, &iu);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(fdm, &fu);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(rdm, user);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dm, rdm, &I, &scaling);CHKERRQ(ierr);
  /* Project function into initial FE function space */
  if (isPlex) {
    ierr = DMPlexProjectFunction(idm, &user->fe, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iu);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = DMDAProjectFunction(idm, &user->fe, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iu);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM projection routine for this type of DM");
  /* Interpolate function into final FE function space */
  if (checkRestrict) {ierr = MatRestrict(I, iu, fu);CHKERRQ(ierr);ierr = VecPointwiseMult(fu, scaling, fu);CHKERRQ(ierr);}
  else               {ierr = MatInterpolate(I, iu, fu);CHKERRQ(ierr);}
#if 0
  ierr = PetscPrintf(PETSC_COMM_WORLD, checkRestrict ? "Fine vec\n"   : "Coarse vec\n");
  ierr = VecView(iu, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, checkRestrict ? "Coarse vec\n" : "Fine vec\n");
  ierr = VecView(fu, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  /* Compare approximation to exact in L_2 */
  if (isPlex) {
    ierr = DMPlexComputeL2Diff(fdm, &user->fe, exactFuncs, exactCtxs, fu, &error);CHKERRQ(ierr);
    ierr = DMPlexComputeL2GradientDiff(fdm, &user->fe, exactFuncDers, exactCtxs, fu, n, &errorDer);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = DMDAComputeL2Diff(fdm, &user->fe, exactFuncs, exactCtxs, fu, &error);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM L_2 difference routine for this type of DM");
  /* Report result */
  if (error > tol) {
    ierr = PetscPrintf(comm, "Tests FAIL for order %d at tolerance %g error %g\n", order, tol, error);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(comm, "Tests pass for order %d at tolerance %g\n", order, tol);CHKERRQ(ierr);
  }
  if (errorDer > tol) {
    ierr = PetscPrintf(comm, "Tests FAIL for order %d derivatives at tolerance %g error %g\n", order, tol, errorDer);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(comm, "Tests pass for order %d derivatives at tolerance %g\n", order, tol);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(idm, &iu);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(fdm, &fu);CHKERRQ(ierr);
  ierr = MatDestroy(&I);CHKERRQ(ierr);
  ierr = VecDestroy(&scaling);CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
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
  ierr = PetscFECreateDefault(dm, user.dim, user.numComponents, user.simplex, NULL, user.qorder, &user.fe);CHKERRQ(ierr);
  user.fem.fe = &user.fe;
  ierr = SetupSection(dm, &user);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = CheckFunctions(dm, user.porder, u, &user);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  if (user.dim == 2 && user.simplex == PETSC_TRUE) {
    ierr = CheckInterpolation(dm, PETSC_FALSE, user.porder, &user);CHKERRQ(ierr);
    ierr = CheckInterpolation(dm, PETSC_TRUE,  user.porder, &user);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&user.fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
