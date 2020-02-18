static char help[] = "Check that a DM can accurately represent and interpolate functions of a given polynomial order\n\n";

#include <petscdmplex.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscksp.h>
#include <petscsnes.h>

typedef struct {
  PetscInt  debug;             /* The debugging level */
  /* Domain and mesh definition */
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool simplex;           /* Flag for simplex or tensor product mesh */
  PetscBool refcell;           /* Make the mesh only a reference cell */
  PetscBool useDA;             /* Flag DMDA tensor product mesh */
  PetscBool interpolate;       /* Generate intermediate mesh elements */
  PetscReal refinementLimit;   /* The largest allowable cell volume */
  PetscBool shearCoords;       /* Flag for shear transform */
  PetscBool nonaffineCoords;   /* Flag for non-affine transform */
  /* Element definition */
  PetscInt  qorder;            /* Order of the quadrature */
  PetscInt  numComponents;     /* Number of field components */
  PetscFE   fe;                /* The finite element */
  /* Testing space */
  PetscInt  porder;            /* Order of polynomials to test */
  PetscBool convergence;       /* Test for order of convergence */
  PetscBool convRefine;        /* Test for convergence using refinement, otherwise use coarsening */
  PetscBool constraints;       /* Test local constraints */
  PetscBool tree;              /* Test tree routines */
  PetscBool testFEjacobian;    /* Test finite element Jacobian assembly */
  PetscBool testFVgrad;        /* Test finite difference gradient routine */
  PetscBool testInjector;      /* Test finite element injection routines */
  PetscInt  treeCell;          /* Cell to refine in tree test */
  PetscReal constants[3];      /* Constant values for each dimension */
} AppCtx;

/* u = 1 */
PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx   *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = user->constants[d];
  return 0;
}
PetscErrorCode constantDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx   *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = 0.0;
  return 0;
}

/* u = x */
PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = coords[d];
  return 0;
}
PetscErrorCode linearDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d, e;
  for (d = 0; d < dim; ++d) {
    u[d] = 0.0;
    for (e = 0; e < dim; ++e) u[d] += (d == e ? 1.0 : 0.0) * n[e];
  }
  return 0;
}

/* u = x^2 or u = (x^2, xy) or u = (xy, yz, zx) */
PetscErrorCode quadratic(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  if (user->dim > 2)      {u[0] = coords[0]*coords[1]; u[1] = coords[1]*coords[2]; u[2] = coords[2]*coords[0];}
  else if (user->dim > 1) {u[0] = coords[0]*coords[0]; u[1] = coords[0]*coords[1];}
  else if (user->dim > 0) {u[0] = coords[0]*coords[0];}
  return 0;
}
PetscErrorCode quadraticDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  if (user->dim > 2)      {u[0] = coords[1]*n[0] + coords[0]*n[1]; u[1] = coords[2]*n[1] + coords[1]*n[2]; u[2] = coords[2]*n[0] + coords[0]*n[2];}
  else if (user->dim > 1) {u[0] = 2.0*coords[0]*n[0]; u[1] = coords[1]*n[0] + coords[0]*n[1];}
  else if (user->dim > 0) {u[0] = 2.0*coords[0]*n[0];}
  return 0;
}

/* u = x^3 or u = (x^3, x^2y) or u = (x^2y, y^2z, z^2x) */
PetscErrorCode cubic(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  if (user->dim > 2)      {u[0] = coords[0]*coords[0]*coords[1]; u[1] = coords[1]*coords[1]*coords[2]; u[2] = coords[2]*coords[2]*coords[0];}
  else if (user->dim > 1) {u[0] = coords[0]*coords[0]*coords[0]; u[1] = coords[0]*coords[0]*coords[1];}
  else if (user->dim > 0) {u[0] = coords[0]*coords[0]*coords[0];}
  return 0;
}
PetscErrorCode cubicDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  if (user->dim > 2)      {u[0] = 2.0*coords[0]*coords[1]*n[0] + coords[0]*coords[0]*n[1]; u[1] = 2.0*coords[1]*coords[2]*n[1] + coords[1]*coords[1]*n[2]; u[2] = 2.0*coords[2]*coords[0]*n[2] + coords[2]*coords[2]*n[0];}
  else if (user->dim > 1) {u[0] = 3.0*coords[0]*coords[0]*n[0]; u[1] = 2.0*coords[0]*coords[1]*n[0] + coords[0]*coords[0]*n[1];}
  else if (user->dim > 0) {u[0] = 3.0*coords[0]*coords[0]*n[0];}
  return 0;
}

/* u = tanh(x) */
PetscErrorCode trig(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx   *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = PetscTanhReal(coords[d] - 0.5);
  return 0;
}
PetscErrorCode trigDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx   *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = 1.0/PetscSqr(PetscCoshReal(coords[d] - 0.5)) * n[d];
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 3;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug           = 0;
  options->dim             = 2;
  options->simplex         = PETSC_TRUE;
  options->refcell         = PETSC_FALSE;
  options->useDA           = PETSC_TRUE;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;
  options->shearCoords     = PETSC_FALSE;
  options->nonaffineCoords = PETSC_FALSE;
  options->qorder          = 0;
  options->numComponents   = PETSC_DEFAULT;
  options->porder          = 0;
  options->convergence     = PETSC_FALSE;
  options->convRefine      = PETSC_TRUE;
  options->constraints     = PETSC_FALSE;
  options->tree            = PETSC_FALSE;
  options->treeCell        = 0;
  options->testFEjacobian  = PETSC_FALSE;
  options->testFVgrad      = PETSC_FALSE;
  options->testInjector    = PETSC_FALSE;
  options->constants[0]    = 1.0;
  options->constants[1]    = 2.0;
  options->constants[2]    = 3.0;

  ierr = PetscOptionsBegin(comm, "", "Projection Test Options", "DMPlex");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-debug", "The debugging level", "ex3.c", options->debug, &options->debug, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex3.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Flag for simplices or hexhedra", "ex3.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-refcell", "Make the mesh only the reference cell", "ex3.c", options->refcell, &options->refcell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_da", "Flag for DMDA mesh", "ex3.c", options->useDA, &options->useDA, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex3.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex3.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-shear_coords", "Transform coordinates with a shear", "ex3.c", options->shearCoords, &options->shearCoords, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-non_affine_coords", "Transform coordinates with a non-affine transform", "ex3.c", options->nonaffineCoords, &options->nonaffineCoords, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-qorder", "The quadrature order", "ex3.c", options->qorder, &options->qorder, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_comp", "The number of field components", "ex3.c", options->numComponents, &options->numComponents, NULL,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-porder", "The order of polynomials to test", "ex3.c", options->porder, &options->porder, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-convergence", "Check the convergence rate", "ex3.c", options->convergence, &options->convergence, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-conv_refine", "Use refinement for the convergence rate", "ex3.c", options->convRefine, &options->convRefine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-constraints", "Test local constraints (serial only)", "ex3.c", options->constraints, &options->constraints, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tree", "Test tree routines", "ex3.c", options->tree, &options->tree, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-tree_cell", "cell to refine in tree test", "ex3.c", options->treeCell, &options->treeCell, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_fe_jacobian", "Test finite element Jacobian assembly", "ex3.c", options->testFEjacobian, &options->testFEjacobian, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_fv_grad", "Test finite volume gradient reconstruction", "ex3.c", options->testFVgrad, &options->testFVgrad, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_injector","Test finite element injection", "ex3.c", options->testInjector, &options->testInjector,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-constants","Set the constant values", "ex3.c", options->constants, &n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  options->numComponents = options->numComponents < 0 ? options->dim : options->numComponents;

  PetscFunctionReturn(0);
}

static PetscErrorCode TransformCoordinates(DM dm, AppCtx *user)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       vStart, vEnd, v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (user->nonaffineCoords) {
    /* x' = r^(1/p) (x/r), y' = r^(1/p) (y/r), z' = z */
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt  dof, off;
      PetscReal p = 4.0, r;

      ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      switch (dof) {
      case 2:
        r             = PetscSqr(PetscRealPart(coords[off+0])) + PetscSqr(PetscRealPart(coords[off+1]));
        coords[off+0] = r == 0.0 ? 0.0 : PetscPowReal(r, (1 - p)/(2*p))*coords[off+0];
        coords[off+1] = r == 0.0 ? 0.0 : PetscPowReal(r, (1 - p)/(2*p))*coords[off+1];
        break;
      case 3:
        r             = PetscSqr(PetscRealPart(coords[off+0])) + PetscSqr(PetscRealPart(coords[off+1]));
        coords[off+0] = r == 0.0 ? 0.0 : PetscPowReal(r, (1 - p)/(2*p))*coords[off+0];
        coords[off+1] = r == 0.0 ? 0.0 : PetscPowReal(r, (1 - p)/(2*p))*coords[off+1];
        coords[off+2] = coords[off+2];
        break;
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  }
  if (user->shearCoords) {
    /* x' = x + m y + m z, y' = y + m z,  z' = z */
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt  dof, off;
      PetscReal m = 1.0;

      ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      switch (dof) {
      case 2:
        coords[off+0] = coords[off+0] + m*coords[off+1];
        coords[off+1] = coords[off+1];
        break;
      case 3:
        coords[off+0] = coords[off+0] + m*coords[off+1] + m*coords[off+2];
        coords[off+1] = coords[off+1] + m*coords[off+2];
        coords[off+2] = coords[off+2];
        break;
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (user->refcell) {
    ierr = DMPlexCreateReferenceCell(comm, dim, user->simplex, dm);CHKERRQ(ierr);
  } else if (user->simplex || !user->useDA) {
    DM refinedMesh = NULL;

    ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, NULL, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
    /* Refine mesh using a volume constraint */
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_TRUE);CHKERRQ(ierr);
  } else {
    if (user->constraints || user->tree || !user->useDA) {
      PetscInt cells[3] = {2, 2, 2};

      ierr = PetscOptionsGetInt(NULL,NULL,"-da_grid_x",&cells[0],NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetInt(NULL,NULL,"-da_grid_y",&cells[1],NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetInt(NULL,NULL,"-da_grid_z",&cells[2],NULL);CHKERRQ(ierr);
      ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
    } else {
      switch (user->dim) {
      case 2:
        ierr = DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 2, 2, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, dm);CHKERRQ(ierr);
        ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
        ierr = DMSetUp(*dm);CHKERRQ(ierr);
        ierr = DMDASetVertexCoordinates(*dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot create structured mesh of dimension %d", dim);
      }
      ierr = PetscObjectSetName((PetscObject) *dm, "Hexahedral Mesh");CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectTypeCompare((PetscObject)*dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (isPlex) {
    PetscPartitioner part;
    DM               distributedMesh = NULL;

    if (user->tree) {
      DM refTree;
      DM ncdm = NULL;

      ierr = DMPlexCreateDefaultReferenceTree(comm,user->dim,user->simplex,&refTree);CHKERRQ(ierr);
      ierr = DMPlexSetReferenceTree(*dm,refTree);CHKERRQ(ierr);
      ierr = DMDestroy(&refTree);CHKERRQ(ierr);
      ierr = DMPlexTreeRefineCell(*dm,user->treeCell,&ncdm);CHKERRQ(ierr);
      if (ncdm) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm = ncdm;
        ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
      }
    } else {
      ierr = DMPlexSetRefinementUniform(*dm, PETSC_TRUE);CHKERRQ(ierr);
    }
    /* Distribute mesh over processes */
    ierr = DMPlexGetPartitioner(*dm,&part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
    if (user->simplex) {
      ierr = PetscObjectSetName((PetscObject) *dm, "Simplicial Mesh");CHKERRQ(ierr);
    } else {
      ierr = PetscObjectSetName((PetscObject) *dm, "Hexahedral Mesh");CHKERRQ(ierr);
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = TransformCoordinates(*dm, user);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm,NULL,"-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void simple_mass(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d, e;
  for (d = 0, e = 0; d < dim; d++, e+=dim+1) {
    g0[e] = 1.;
  }
}

/* < \nabla v, 1/2(\nabla u + {\nabla u}^T) > */
static void symmetric_gradient_inner_product(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar C[])
{
  PetscInt compI, compJ, d, e;

  for (compI = 0; compI < dim; ++compI) {
    for (compJ = 0; compJ < dim; ++compJ) {
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; e++) {
          if (d == e && d == compI && d == compJ) {
            C[((compI*dim+compJ)*dim+d)*dim+e] = 1.0;
          } else if ((d == compJ && e == compI) || (d == e && compI == compJ)) {
            C[((compI*dim+compJ)*dim+d)*dim+e] = 0.5;
          } else {
            C[((compI*dim+compJ)*dim+d)*dim+e] = 0.0;
          }
        }
      }
    }
  }
}

static PetscErrorCode SetupSection(DM dm, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!user->simplex && user->constraints) {
    /* test local constraints */
    DM            coordDM;
    PetscInt      fStart, fEnd, f, vStart, vEnd, v;
    PetscInt      edgesx = 2, vertsx;
    PetscInt      edgesy = 2, vertsy;
    PetscMPIInt   size;
    PetscInt      numConst;
    PetscSection  aSec;
    PetscInt     *anchors;
    PetscInt      offset;
    IS            aIS;
    MPI_Comm      comm = PetscObjectComm((PetscObject)dm);

    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    if (size > 1) SETERRQ(comm,PETSC_ERR_SUP,"Local constraint test can only be performed in serial");

    /* we are going to test constraints by using them to enforce periodicity
     * in one direction, and comparing to the existing method of enforcing
     * periodicity */

    /* first create the coordinate section so that it does not clone the
     * constraints */
    ierr = DMGetCoordinateDM(dm,&coordDM);CHKERRQ(ierr);

    /* create the constrained-to-anchor section */
    ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_SELF,&aSec);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(aSec,PetscMin(fStart,vStart),PetscMax(fEnd,vEnd));CHKERRQ(ierr);

    /* define the constraints */
    ierr = PetscOptionsGetInt(NULL,NULL,"-da_grid_x",&edgesx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-da_grid_y",&edgesy,NULL);CHKERRQ(ierr);
    vertsx = edgesx + 1;
    vertsy = edgesy + 1;
    numConst = vertsy + edgesy;
    ierr = PetscMalloc1(numConst,&anchors);CHKERRQ(ierr);
    offset = 0;
    for (v = vStart + edgesx; v < vEnd; v+= vertsx) {
      ierr = PetscSectionSetDof(aSec,v,1);CHKERRQ(ierr);
      anchors[offset++] = v - edgesx;
    }
    for (f = fStart + edgesx * vertsy + edgesx * edgesy; f < fEnd; f++) {
      ierr = PetscSectionSetDof(aSec,f,1);CHKERRQ(ierr);
      anchors[offset++] = f - edgesx * edgesy;
    }
    ierr = PetscSectionSetUp(aSec);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,numConst,anchors,PETSC_OWN_POINTER,&aIS);CHKERRQ(ierr);

    ierr = DMPlexSetAnchors(dm,aSec,aIS);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&aSec);CHKERRQ(ierr);
    ierr = ISDestroy(&aIS);CHKERRQ(ierr);
  }
  ierr = DMSetNumFields(dm, 1);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) user->fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  if (!user->simplex && user->constraints) {
    /* test getting local constraint matrix that matches section */
    PetscSection aSec;
    IS           aIS;

    ierr = DMPlexGetAnchors(dm,&aSec,&aIS);CHKERRQ(ierr);
    if (aSec) {
      PetscDS         ds;
      PetscSection    cSec, section;
      PetscInt        cStart, cEnd, c, numComp;
      Mat             cMat, mass;
      Vec             local;
      const PetscInt *anchors;

      ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
      /* this creates the matrix and preallocates the matrix structure: we
       * just have to fill in the values */
      ierr = DMGetDefaultConstraints(dm,&cSec,&cMat);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(cSec,&cStart,&cEnd);CHKERRQ(ierr);
      ierr = ISGetIndices(aIS,&anchors);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(user->fe, &numComp);CHKERRQ(ierr);
      for (c = cStart; c < cEnd; c++) {
        PetscInt cDof;

        /* is this point constrained? (does it have an anchor?) */
        ierr = PetscSectionGetDof(aSec,c,&cDof);CHKERRQ(ierr);
        if (cDof) {
          PetscInt cOff, a, aDof, aOff, j;
          if (cDof != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Found %d anchor points: should be just one",cDof);

          /* find the anchor point */
          ierr = PetscSectionGetOffset(aSec,c,&cOff);CHKERRQ(ierr);
          a    = anchors[cOff];

          /* find the constrained dofs (row in constraint matrix) */
          ierr = PetscSectionGetDof(cSec,c,&cDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(cSec,c,&cOff);CHKERRQ(ierr);

          /* find the anchor dofs (column in constraint matrix) */
          ierr = PetscSectionGetDof(section,a,&aDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(section,a,&aOff);CHKERRQ(ierr);

          if (cDof != aDof) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point and anchor have different number of dofs: %d, %d\n",cDof,aDof);
          if (cDof % numComp) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point dofs not divisible by field components: %d, %d\n",cDof,numComp);

          /* put in a simple equality constraint */
          for (j = 0; j < cDof; j++) {
            ierr = MatSetValue(cMat,cOff+j,aOff+j,1.,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
      ierr = MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);

      /* Now that we have constructed the constraint matrix, any FE matrix
       * that we construct will apply the constraints during construction */

      ierr = DMCreateMatrix(dm,&mass);CHKERRQ(ierr);
      /* get a dummy local variable to serve as the solution */
      ierr = DMGetLocalVector(dm,&local);CHKERRQ(ierr);
      ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
      /* set the jacobian to be the mass matrix */
      ierr = PetscDSSetJacobian(ds, 0, 0, simple_mass, NULL,  NULL, NULL);CHKERRQ(ierr);
      /* build the mass matrix */
      ierr = DMPlexSNESComputeJacobianFEM(dm,local,mass,mass,NULL);CHKERRQ(ierr);
      ierr = MatView(mass,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = MatDestroy(&mass);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(dm,&local);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFEJacobian(DM dm, AppCtx *user)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (isPlex) {
    Vec          local;
    const Vec    *vecs;
    Mat          E;
    MatNullSpace sp;
    PetscBool    isNullSpace, hasConst;
    PetscInt     n, i;
    Vec          res = NULL, localX, localRes;
    PetscDS      ds;

    if (user->numComponents != user->dim) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "The number of components %d must be equal to the dimension %d for this test", user->numComponents, user->dim);
    ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds,0,0,NULL,NULL,NULL,symmetric_gradient_inner_product);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm,&E);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local);CHKERRQ(ierr);
    ierr = DMPlexSNESComputeJacobianFEM(dm,local,E,E,NULL);CHKERRQ(ierr);
    ierr = DMPlexCreateRigidBody(dm,&sp);CHKERRQ(ierr);
    ierr = MatNullSpaceGetVecs(sp,&hasConst,&n,&vecs);CHKERRQ(ierr);
    if (n) {ierr = VecDuplicate(vecs[0],&res);CHKERRQ(ierr);}
    ierr = DMCreateLocalVector(dm,&localX);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(dm,&localRes);CHKERRQ(ierr);
    for (i = 0; i < n; i++) { /* also test via matrix-free Jacobian application */
      PetscReal resNorm;

      ierr = VecSet(localRes,0.);CHKERRQ(ierr);
      ierr = VecSet(localX,0.);CHKERRQ(ierr);
      ierr = VecSet(local,0.);CHKERRQ(ierr);
      ierr = VecSet(res,0.);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dm,vecs[i],INSERT_VALUES,localX);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dm,vecs[i],INSERT_VALUES,localX);CHKERRQ(ierr);
      ierr = DMPlexComputeJacobianAction(dm,NULL,0,0,local,NULL,localX,localRes,NULL);CHKERRQ(ierr);
      ierr = DMLocalToGlobalBegin(dm,localRes,ADD_VALUES,res);CHKERRQ(ierr);
      ierr = DMLocalToGlobalEnd(dm,localRes,ADD_VALUES,res);CHKERRQ(ierr);
      ierr = VecNorm(res,NORM_2,&resNorm);CHKERRQ(ierr);
      if (resNorm > PETSC_SMALL) {
        ierr = PetscPrintf(PetscObjectComm((PetscObject)dm),"Symmetric gradient action null space vector %D residual: %E\n",i,resNorm);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(&localRes);CHKERRQ(ierr);
    ierr = VecDestroy(&localX);CHKERRQ(ierr);
    ierr = VecDestroy(&res);CHKERRQ(ierr);
    ierr = MatNullSpaceTest(sp,E,&isNullSpace);CHKERRQ(ierr);
    if (isNullSpace) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)dm),"Symmetric gradient null space: PASS\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)dm),"Symmetric gradient null space: FAIL\n");CHKERRQ(ierr);
    }
    ierr = MatNullSpaceDestroy(&sp);CHKERRQ(ierr);
    ierr = MatDestroy(&E);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestInjector(DM dm, AppCtx *user)
{
  DM             refTree;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
  if (refTree) {
    Mat inj;

    ierr = DMPlexComputeInjectorReferenceTree(refTree,&inj);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)inj,"Reference Tree Injector");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatView(inj,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&inj);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFVGrad(DM dm, AppCtx *user)
{
  MPI_Comm          comm;
  DM                dmRedist, dmfv, dmgrad, dmCell, refTree;
  PetscFV           fv;
  PetscInt          nvecs, v, cStart, cEnd, cEndInterior;
  PetscMPIInt       size;
  Vec               cellgeom, grad, locGrad;
  const PetscScalar *cgeom;
  PetscReal         allVecMaxDiff = 0., fvTol = 100. * PETSC_MACHINE_EPSILON;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)dm);
  /* duplicate DM, give dup. a FV discretization */
  ierr = DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  dmRedist = NULL;
  if (size > 1) {
    ierr = DMPlexDistributeOverlap(dm,1,NULL,&dmRedist);CHKERRQ(ierr);
  }
  if (!dmRedist) {
    dmRedist = dm;
    ierr = PetscObjectReference((PetscObject)dmRedist);CHKERRQ(ierr);
  }
  ierr = PetscFVCreate(comm,&fv);CHKERRQ(ierr);
  ierr = PetscFVSetType(fv,PETSCFVLEASTSQUARES);CHKERRQ(ierr);
  ierr = PetscFVSetNumComponents(fv,user->numComponents);CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(fv,user->dim);CHKERRQ(ierr);
  ierr = PetscFVSetFromOptions(fv);CHKERRQ(ierr);
  ierr = PetscFVSetUp(fv);CHKERRQ(ierr);
  ierr = DMPlexConstructGhostCells(dmRedist,NULL,NULL,&dmfv);CHKERRQ(ierr);
  ierr = DMDestroy(&dmRedist);CHKERRQ(ierr);
  ierr = DMSetNumFields(dmfv,1);CHKERRQ(ierr);
  ierr = DMSetField(dmfv, 0, NULL, (PetscObject) fv);CHKERRQ(ierr);
  ierr = DMCreateDS(dmfv);CHKERRQ(ierr);
  ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
  if (refTree) {ierr = DMCopyDisc(dmfv,refTree);CHKERRQ(ierr);}
  ierr = DMPlexSNESGetGradientDM(dmfv, fv, &dmgrad);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmfv,0,&cStart,&cEnd);CHKERRQ(ierr);
  nvecs = user->dim * (user->dim+1) / 2;
  ierr = DMPlexSNESGetGeometryFVM(dmfv,NULL,&cellgeom,NULL);CHKERRQ(ierr);
  ierr = VecGetDM(cellgeom,&dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellgeom,&cgeom);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmgrad,&grad);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmgrad,&locGrad);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dmgrad,&cEndInterior,NULL);CHKERRQ(ierr);
  cEndInterior = (cEndInterior < 0) ? cEnd: cEndInterior;
  for (v = 0; v < nvecs; v++) {
    Vec               locX;
    PetscInt          c;
    PetscScalar       trueGrad[3][3] = {{0.}};
    const PetscScalar *gradArray;
    PetscReal         maxDiff, maxDiffGlob;

    ierr = DMGetLocalVector(dmfv,&locX);CHKERRQ(ierr);
    /* get the local projection of the rigid body mode */
    for (c = cStart; c < cEnd; c++) {
      PetscFVCellGeom *cg;
      PetscScalar     cx[3] = {0.,0.,0.};

      ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
      if (v < user->dim) {
        cx[v] = 1.;
      } else {
        PetscInt w = v - user->dim;

        cx[(w + 1) % user->dim] =  cg->centroid[(w + 2) % user->dim];
        cx[(w + 2) % user->dim] = -cg->centroid[(w + 1) % user->dim];
      }
      ierr = DMPlexVecSetClosure(dmfv,NULL,locX,c,cx,INSERT_ALL_VALUES);CHKERRQ(ierr);
    }
    /* TODO: this isn't in any header */
    ierr = DMPlexReconstructGradientsFVM(dmfv,locX,grad);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmgrad,grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmgrad,grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGrad,&gradArray);CHKERRQ(ierr);
    /* compare computed gradient to exact gradient */
    if (v >= user->dim) {
      PetscInt w = v - user->dim;

      trueGrad[(w + 1) % user->dim][(w + 2) % user->dim] =  1.;
      trueGrad[(w + 2) % user->dim][(w + 1) % user->dim] = -1.;
    }
    maxDiff = 0.;
    for (c = cStart; c < cEndInterior; c++) {
      PetscScalar *compGrad;
      PetscInt    i, j, k;
      PetscReal   FrobDiff = 0.;

      ierr = DMPlexPointLocalRead(dmgrad, c, gradArray, &compGrad);CHKERRQ(ierr);

      for (i = 0, k = 0; i < user->dim; i++) {
        for (j = 0; j < user->dim; j++, k++) {
          PetscScalar diff = compGrad[k] - trueGrad[i][j];
          FrobDiff += PetscRealPart(diff * PetscConj(diff));
        }
      }
      FrobDiff = PetscSqrtReal(FrobDiff);
      maxDiff  = PetscMax(maxDiff,FrobDiff);
    }
    ierr = MPI_Allreduce(&maxDiff,&maxDiffGlob,1,MPIU_REAL,MPIU_MAX,comm);CHKERRQ(ierr);
    allVecMaxDiff = PetscMax(allVecMaxDiff,maxDiffGlob);
    ierr = VecRestoreArrayRead(locGrad,&gradArray);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmfv,&locX);CHKERRQ(ierr);
  }
  if (allVecMaxDiff < fvTol) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm),"Finite volume gradient reconstruction: PASS\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm),"Finite volume gradient reconstruction: FAIL at tolerance %g with max difference %g\n",fvTol,allVecMaxDiff);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dmgrad,&locGrad);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmgrad,&grad);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellgeom,&cgeom);CHKERRQ(ierr);
  ierr = DMDestroy(&dmfv);CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeError(DM dm, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *),
                                   PetscErrorCode (**exactFuncDers)(PetscInt, PetscReal, const PetscReal[], const PetscReal[], PetscInt, PetscScalar *, void *),
                                   void **exactCtxs, PetscReal *error, PetscReal *errorDer, AppCtx *user)
{
  Vec            u;
  PetscReal      n[3] = {1.0, 1.0, 1.0};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Project function into FE function space */
  ierr = DMProjectFunction(dm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-projection_view");CHKERRQ(ierr);
  /* Compare approximation to exact in L_2 */
  ierr = DMComputeL2Diff(dm, 0.0, exactFuncs, exactCtxs, u, error);CHKERRQ(ierr);
  ierr = DMComputeL2GradientDiff(dm, 0.0, exactFuncDers, exactCtxs, u, n, errorDer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckFunctions(DM dm, PetscInt order, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1]) (PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx);
  void            *exactCtxs[3];
  MPI_Comm         comm;
  PetscReal        error, errorDer, tol = PETSC_SMALL;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  /* Setup functions to approximate */
  switch (order) {
  case 0:
    exactFuncs[0]    = constant;
    exactFuncDers[0] = constantDer;
    break;
  case 1:
    exactFuncs[0]    = linear;
    exactFuncDers[0] = linearDer;
    break;
  case 2:
    exactFuncs[0]    = quadratic;
    exactFuncDers[0] = quadraticDer;
    break;
  case 3:
    exactFuncs[0]    = cubic;
    exactFuncDers[0] = cubicDer;
    break;
  default:
    SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for dimension %d order %d", user->dim, order);
  }
  ierr = ComputeError(dm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user);CHKERRQ(ierr);
  /* Report result */
  if (error > tol)    {ierr = PetscPrintf(comm, "Function tests FAIL for order %D at tolerance %g error %g\n", order, (double)tol,(double) error);CHKERRQ(ierr);}
  else                {ierr = PetscPrintf(comm, "Function tests pass for order %D at tolerance %g\n", order, (double)tol);CHKERRQ(ierr);}
  if (errorDer > tol) {ierr = PetscPrintf(comm, "Function tests FAIL for order %D derivatives at tolerance %g error %g\n", order, (double)tol, (double)errorDer);CHKERRQ(ierr);}
  else                {ierr = PetscPrintf(comm, "Function tests pass for order %D derivatives at tolerance %g\n", order, (double)tol);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckInterpolation(DM dm, PetscBool checkRestrict, PetscInt order, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1]) (PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1]) (PetscInt, PetscReal, const PetscReal x[], const PetscReal n[], PetscInt, PetscScalar *u, void *ctx);
  PetscReal       n[3]         = {1.0, 1.0, 1.0};
  void           *exactCtxs[3];
  DM              rdm, idm, fdm;
  Mat             Interp;
  Vec             iu, fu, scaling;
  MPI_Comm        comm;
  PetscInt        dim  = user->dim;
  PetscReal       error, errorDer, tol = PETSC_SMALL;
  PetscBool       isPlex;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  ierr = DMRefine(dm, comm, &rdm);CHKERRQ(ierr);
  ierr = DMSetCoarseDM(rdm, dm);CHKERRQ(ierr);
  ierr = DMPlexSetRegularRefinement(rdm, user->convRefine);CHKERRQ(ierr);
  if (user->tree && isPlex) {
    DM refTree;
    ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
    ierr = DMPlexSetReferenceTree(rdm,refTree);CHKERRQ(ierr);
  }
  if (!user->simplex && !user->constraints) {ierr = DMDASetVertexCoordinates(rdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);}
  ierr = SetupSection(rdm, user);CHKERRQ(ierr);
  /* Setup functions to approximate */
  switch (order) {
  case 0:
    exactFuncs[0]    = constant;
    exactFuncDers[0] = constantDer;
    break;
  case 1:
    exactFuncs[0]    = linear;
    exactFuncDers[0] = linearDer;
    break;
  case 2:
    exactFuncs[0]    = quadratic;
    exactFuncDers[0] = quadraticDer;
    break;
  case 3:
    exactFuncs[0]    = cubic;
    exactFuncDers[0] = cubicDer;
    break;
  default:
    SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for dimension %D order %D", dim, order);
  }
  idm  = checkRestrict ? rdm :  dm;
  fdm  = checkRestrict ?  dm : rdm;
  ierr = DMGetGlobalVector(idm, &iu);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(fdm, &fu);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(rdm, user);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dm, rdm, &Interp, &scaling);CHKERRQ(ierr);
  /* Project function into initial FE function space */
  ierr = DMProjectFunction(idm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iu);CHKERRQ(ierr);
  /* Interpolate function into final FE function space */
  if (checkRestrict) {ierr = MatRestrict(Interp, iu, fu);CHKERRQ(ierr);ierr = VecPointwiseMult(fu, scaling, fu);CHKERRQ(ierr);}
  else               {ierr = MatInterpolate(Interp, iu, fu);CHKERRQ(ierr);}
  /* Compare approximation to exact in L_2 */
  ierr = DMComputeL2Diff(fdm, 0.0, exactFuncs, exactCtxs, fu, &error);CHKERRQ(ierr);
  ierr = DMComputeL2GradientDiff(fdm, 0.0, exactFuncDers, exactCtxs, fu, n, &errorDer);CHKERRQ(ierr);
  /* Report result */
  if (error > tol)    {ierr = PetscPrintf(comm, "Interpolation tests FAIL for order %D at tolerance %g error %g\n", order, (double)tol, (double)error);CHKERRQ(ierr);}
  else                {ierr = PetscPrintf(comm, "Interpolation tests pass for order %D at tolerance %g\n", order, (double)tol);CHKERRQ(ierr);}
  if (errorDer > tol) {ierr = PetscPrintf(comm, "Interpolation tests FAIL for order %D derivatives at tolerance %g error %g\n", order, (double)tol, (double)errorDer);CHKERRQ(ierr);}
  else                {ierr = PetscPrintf(comm, "Interpolation tests pass for order %D derivatives at tolerance %g\n", order, (double)tol);CHKERRQ(ierr);}
  ierr = DMRestoreGlobalVector(idm, &iu);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(fdm, &fu);CHKERRQ(ierr);
  ierr = MatDestroy(&Interp);CHKERRQ(ierr);
  ierr = VecDestroy(&scaling);CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckConvergence(DM dm, PetscInt Nr, AppCtx *user)
{
  DM               odm = dm, rdm = NULL, cdm = NULL;
  PetscErrorCode (*exactFuncs[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {trig};
  PetscErrorCode (*exactFuncDers[1]) (PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx) = {trigDer};
  void            *exactCtxs[3];
  PetscInt         r, c, cStart, cEnd;
  PetscReal        errorOld, errorDerOld, error, errorDer, rel, len, lenOld;
  double           p;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  if (!user->convergence) PetscFunctionReturn(0);
  exactCtxs[0] = user;
  exactCtxs[1] = user;
  exactCtxs[2] = user;
  ierr = PetscObjectReference((PetscObject) odm);CHKERRQ(ierr);
  if (!user->convRefine) {
    for (r = 0; r < Nr; ++r) {
      ierr = DMRefine(odm, PetscObjectComm((PetscObject) dm), &rdm);CHKERRQ(ierr);
      ierr = DMDestroy(&odm);CHKERRQ(ierr);
      odm  = rdm;
    }
    ierr = SetupSection(odm, user);CHKERRQ(ierr);
  }
  ierr = ComputeError(odm, exactFuncs, exactFuncDers, exactCtxs, &errorOld, &errorDerOld, user);CHKERRQ(ierr);
  if (user->convRefine) {
    for (r = 0; r < Nr; ++r) {
      ierr = DMRefine(odm, PetscObjectComm((PetscObject) dm), &rdm);CHKERRQ(ierr);
      if (!user->simplex && user->useDA) {ierr = DMDASetVertexCoordinates(rdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);}
      ierr = SetupSection(rdm, user);CHKERRQ(ierr);
      ierr = ComputeError(rdm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user);CHKERRQ(ierr);
      p    = PetscLog2Real(errorOld/error);
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Function   convergence rate at refinement %D: %.2f\n", r, (double)p);CHKERRQ(ierr);
      p    = PetscLog2Real(errorDerOld/errorDer);
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Derivative convergence rate at refinement %D: %.2f\n", r, (double)p);CHKERRQ(ierr);
      ierr = DMDestroy(&odm);CHKERRQ(ierr);
      odm         = rdm;
      errorOld    = error;
      errorDerOld = errorDer;
    }
  } else {
    /* ierr = ComputeLongestEdge(dm, &lenOld);CHKERRQ(ierr); */
    ierr = DMPlexGetHeightStratum(odm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    lenOld = cEnd - cStart;
    for (c = 0; c < Nr; ++c) {
      ierr = DMCoarsen(odm, PetscObjectComm((PetscObject) dm), &cdm);CHKERRQ(ierr);
      if (!user->simplex && user->useDA) {ierr = DMDASetVertexCoordinates(cdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);}
      ierr = SetupSection(cdm, user);CHKERRQ(ierr);
      ierr = ComputeError(cdm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user);CHKERRQ(ierr);
      /* ierr = ComputeLongestEdge(cdm, &len);CHKERRQ(ierr); */
      ierr = DMPlexGetHeightStratum(cdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
      len  = cEnd - cStart;
      rel  = error/errorOld;
      p    = PetscLogReal(rel) / PetscLogReal(lenOld / len);
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Function   convergence rate at coarsening %D: %.2f\n", c, (double)p);CHKERRQ(ierr);
      rel  = errorDer/errorDerOld;
      p    = PetscLogReal(rel) / PetscLogReal(lenOld / len);
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Derivative convergence rate at coarsening %D: %.2f\n", c, (double)p);CHKERRQ(ierr);
      ierr = DMDestroy(&odm);CHKERRQ(ierr);
      odm         = cdm;
      errorOld    = error;
      errorDerOld = errorDer;
      lenOld      = len;
    }
  }
  ierr = DMDestroy(&odm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_WORLD, user.dim, user.numComponents, user.simplex, NULL, user.qorder, &user.fe);CHKERRQ(ierr);
  ierr = SetupSection(dm, &user);CHKERRQ(ierr);
  if (user.testFEjacobian) {ierr = TestFEJacobian(dm, &user);CHKERRQ(ierr);}
  if (user.testFVgrad) {ierr = TestFVGrad(dm, &user);CHKERRQ(ierr);}
  if (user.testInjector) {ierr = TestInjector(dm, &user);CHKERRQ(ierr);}
  ierr = CheckFunctions(dm, user.porder, &user);CHKERRQ(ierr);
  {
    PetscDualSpace dsp;
    PetscInt       k;

    ierr = PetscFEGetDualSpace(user.fe, &dsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDeRahm(dsp, &k);CHKERRQ(ierr);
    if (user.dim == 2 && user.simplex == PETSC_TRUE && user.tree == PETSC_FALSE && k == 0) {
      ierr = CheckInterpolation(dm, PETSC_FALSE, user.porder, &user);CHKERRQ(ierr);
      ierr = CheckInterpolation(dm, PETSC_TRUE,  user.porder, &user);CHKERRQ(ierr);
    }
  }
  ierr = CheckConvergence(dm, 3, &user);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user.fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 1
    requires: triangle

  # 2D P_1 on a triangle
  test:
    suffix: p1_2d_0
    requires: triangle
    args: -petscspace_degree 1 -qorder 1 -convergence
  test:
    suffix: p1_2d_1
    requires: triangle
    args: -petscspace_degree 1 -qorder 1 -porder 1
  test:
    suffix: p1_2d_2
    requires: triangle
    args: -petscspace_degree 1 -qorder 1 -porder 2
  test:
    suffix: p1_2d_3
    requires: triangle pragmatic
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -convergence -conv_refine 0
    filter: grep -v DEBUG
  test:
    suffix: p1_2d_4
    requires: triangle pragmatic
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p1_2d_5
    requires: triangle pragmatic
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 2 -conv_refine 0

  # 3D P_1 on a tetrahedron
  test:
    suffix: p1_3d_0
    requires: ctetgen
    args: -dim 3 -petscspace_degree 1 -qorder 1 -convergence
  test:
    suffix: p1_3d_1
    requires: ctetgen
    args: -dim 3 -petscspace_degree 1 -qorder 1 -porder 1
  test:
    suffix: p1_3d_2
    requires: ctetgen
    args: -dim 3 -petscspace_degree 1 -qorder 1 -porder 2
  test:
    suffix: p1_3d_3
    requires: ctetgen pragmatic
    args: -dim 3 -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -convergence -conv_refine 0
    filter: grep -v DEBUG
  test:
    suffix: p1_3d_4
    requires: ctetgen pragmatic
    args: -dim 3 -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p1_3d_5
    requires: ctetgen pragmatic
    args: -dim 3 -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 2 -conv_refine 0

  # 2D P_2 on a triangle
  test:
    suffix: p2_2d_0
    requires: triangle
    args: -petscspace_degree 2 -qorder 2 -convergence
  test:
    suffix: p2_2d_1
    requires: triangle
    args: -petscspace_degree 2 -qorder 2 -porder 1
  test:
    suffix: p2_2d_2
    requires: triangle
    args: -petscspace_degree 2 -qorder 2 -porder 2
  test:
    suffix: p2_2d_3
    requires: triangle pragmatic
    args: -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -convergence -conv_refine 0
    filter: grep -v DEBUG
  test:
    suffix: p2_2d_4
    requires: triangle pragmatic
    args: -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p2_2d_5
    requires: triangle pragmatic
    args: -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder 2 -conv_refine 0

  # 3D P_2 on a tetrahedron
  test:
    suffix: p2_3d_0
    requires: ctetgen
    args: -dim 3 -petscspace_degree 2 -qorder 2 -convergence
  test:
    suffix: p2_3d_1
    requires: ctetgen
    args: -dim 3 -petscspace_degree 2 -qorder 2 -porder 1
  test:
    suffix: p2_3d_2
    requires: ctetgen
    args: -dim 3 -petscspace_degree 2 -qorder 2 -porder 2
  test:
    suffix: p2_3d_3
    requires: ctetgen pragmatic
    args: -dim 3 -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -convergence -conv_refine 0
    filter: grep -v DEBUG
  test:
    suffix: p2_3d_4
    requires: ctetgen pragmatic
    args: -dim 3 -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p2_3d_5
    requires: ctetgen pragmatic
    args: -dim 3 -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder 2 -conv_refine 0

  # 2D Q_1 on a quadrilaterial DA
  test:
    suffix: q1_2d_da_0
    requires: mpi_type_get_envelope broken
    args: -simplex 0 -petscspace_degree 1 -qorder 1 -convergence
  test:
    suffix: q1_2d_da_1
    requires: mpi_type_get_envelope broken
    args: -simplex 0 -petscspace_degree 1 -qorder 1 -porder 1
  test:
    suffix: q1_2d_da_2
    requires: mpi_type_get_envelope broken
    args: -simplex 0 -petscspace_degree 1 -qorder 1 -porder 2

  # 2D Q_1 on a quadrilaterial Plex
  test:
    suffix: q1_2d_plex_0
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -qorder 1 -convergence
  test:
    suffix: q1_2d_plex_1
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -qorder 1 -porder 1
  test:
    suffix: q1_2d_plex_2
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -qorder 1 -porder 2
  test:
    suffix: q1_2d_plex_3
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -qorder 1 -porder 1 -shear_coords
  test:
    suffix: q1_2d_plex_4
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -qorder 1 -porder 2 -shear_coords
  test:
    suffix: q1_2d_plex_5
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -petscspace_type tensor -qorder 1 -porder 0 -non_affine_coords -convergence
  test:
    suffix: q1_2d_plex_6
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -petscspace_type tensor -qorder 1 -porder 1 -non_affine_coords -convergence
  test:
    suffix: q1_2d_plex_7
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -petscspace_type tensor -qorder 1 -porder 2 -non_affine_coords -convergence

  # 2D Q_2 on a quadrilaterial
  test:
    suffix: q2_2d_plex_0
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -convergence
  test:
    suffix: q2_2d_plex_1
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -porder 1
  test:
    suffix: q2_2d_plex_2
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -porder 2
  test:
    suffix: q2_2d_plex_3
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -porder 1 -shear_coords
  test:
    suffix: q2_2d_plex_4
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -porder 2 -shear_coords
  test:
    suffix: q2_2d_plex_5
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -petscspace_type tensor -qorder 2 -porder 0 -non_affine_coords -convergence
  test:
    suffix: q2_2d_plex_6
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -petscspace_type tensor -qorder 2 -porder 1 -non_affine_coords -convergence
  test:
    suffix: q2_2d_plex_7
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -petscspace_type tensor -qorder 2 -porder 2 -non_affine_coords -convergence


  # 2D P_3 on a triangle
  test:
    suffix: p3_2d_0
    requires: triangle !single
    args: -petscspace_degree 3 -qorder 3 -convergence
  test:
    suffix: p3_2d_1
    requires: triangle !single
    args: -petscspace_degree 3 -qorder 3 -porder 1
  test:
    suffix: p3_2d_2
    requires: triangle !single
    args: -petscspace_degree 3 -qorder 3 -porder 2
  test:
    suffix: p3_2d_3
    requires: triangle !single
    args: -petscspace_degree 3 -qorder 3 -porder 3
  test:
    suffix: p3_2d_4
    requires: triangle pragmatic
    args: -petscspace_degree 3 -qorder 3 -dm_plex_hash_location -convergence -conv_refine 0
    filter: grep -v DEBUG
  test:
    suffix: p3_2d_5
    requires: triangle pragmatic
    args: -petscspace_degree 3 -qorder 3 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p3_2d_6
    requires: triangle pragmatic
    args: -petscspace_degree 3 -qorder 3 -dm_plex_hash_location -porder 3 -conv_refine 0

  # 2D Q_3 on a quadrilaterial
  test:
    suffix: q3_2d_0
    requires: mpi_type_get_envelope !single
    args: -use_da 0 -simplex 0 -petscspace_degree 3 -qorder 3 -convergence
  test:
    suffix: q3_2d_1
    requires: mpi_type_get_envelope !single
    args: -use_da 0 -simplex 0 -petscspace_degree 3 -qorder 3 -porder 1
  test:
    suffix: q3_2d_2
    requires: mpi_type_get_envelope !single
    args: -use_da 0 -simplex 0 -petscspace_degree 3 -qorder 3 -porder 2
  test:
    suffix: q3_2d_3
    requires: mpi_type_get_envelope !single
    args: -use_da 0 -simplex 0 -petscspace_degree 3 -qorder 3 -porder 3

  # 2D P_1disc on a triangle/quadrilateral
  test:
    suffix: p1d_2d_0
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -convergence
  test:
    suffix: p1d_2d_1
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -porder 1
  test:
    suffix: p1d_2d_2
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -porder 2
  test:
    suffix: p1d_2d_3
    requires: triangle
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -convergence
    filter: sed  -e "s/convergence rate at refinement 0: 2/convergence rate at refinement 0: 1.9/g"
  test:
    suffix: p1d_2d_4
    requires: triangle
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -porder 1
  test:
    suffix: p1d_2d_5
    requires: triangle
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -porder 2

  # 2D BDM_1 on a triangle
  test:
    suffix: bdm1_2d_0
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_type bdm \
          -num_comp 2 -qorder 1 -convergence
  test:
    suffix: bdm1_2d_1
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_type bdm \
          -num_comp 2 -qorder 1 -porder 1
  test:
    suffix: bdm1_2d_2
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_type bdm \
          -num_comp 2 -qorder 1 -porder 2

  # 2D BDM_1 on a quadrilateral
  test:
    suffix: bdm1q_2d_0
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_type bdm \
          -use_da 0 -simplex 0 -num_comp 2 -qorder 1 -convergence
  test:
    suffix: bdm1q_2d_1
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_type bdm \
          -use_da 0 -simplex 0 -num_comp 2 -qorder 1 -porder 1
  test:
    suffix: bdm1q_2d_2
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_type bdm \
          -use_da 0 -simplex 0 -num_comp 2 -qorder 1 -porder 2

  # Test high order quadrature
  test:
    suffix: p1_quad_2
    requires: triangle
    args: -petscspace_degree 1 -qorder 2 -porder 1
  test:
    suffix: p1_quad_5
    requires: triangle
    args: -petscspace_degree 1 -qorder 5 -porder 1
  test:
    suffix: p2_quad_3
    requires: triangle
    args: -petscspace_degree 2 -qorder 3 -porder 2
  test:
    suffix: p2_quad_5
    requires: triangle
    args: -petscspace_degree 2 -qorder 5 -porder 2
  test:
    suffix: q1_quad_2
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -qorder 2 -porder 1
  test:
    suffix: q1_quad_5
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 1 -qorder 5 -porder 1
  test:
    suffix: q2_quad_3
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 3 -porder 1
  test:
    suffix: q2_quad_5
    requires: mpi_type_get_envelope
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 5 -porder 1


  # Nonconforming tests
  test:
    suffix: constraints
    args: -simplex 0 -petscspace_type tensor -petscspace_degree 1 -qorder 0 -constraints
  test:
    suffix: nonconforming_tensor_2
    nsize: 4
    args: -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -simplex 0 -dim 2 -dm_plex_max_projection_height 1 -petscspace_type tensor -petscspace_degree 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL
  test:
    suffix: nonconforming_tensor_3
    nsize: 4
    args: -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 0 -dim 3 -dm_plex_max_projection_height 2 -petscspace_type tensor -petscspace_degree 1 -qorder 1 -dm_view ascii::ASCII_INFO_DETAIL
  test:
    suffix: nonconforming_tensor_2_fv
    nsize: 4
    args: -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 0 -dim 2 -num_comp 2
  test:
    suffix: nonconforming_tensor_3_fv
    nsize: 4
    args: -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 0 -dim 3 -num_comp 3
  test:
    suffix: nonconforming_tensor_2_hi
    requires: !single
    nsize: 4
    args: -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 0 -dim 2 -dm_plex_max_projection_height 1 -petscspace_type tensor -petscspace_degree 4 -qorder 4
  test:
    suffix: nonconforming_tensor_3_hi
    requires: !single skip
    nsize: 4
    args: -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 0 -dim 3 -dm_plex_max_projection_height 2 -petscspace_type tensor -petscspace_degree 4 -qorder 4
  test:
    suffix: nonconforming_simplex_2
    requires: triangle
    nsize: 4
    args: -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -simplex 1 -dim 2 -dm_plex_max_projection_height 1 -petscspace_degree 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL
  test:
    suffix: nonconforming_simplex_2_hi
    requires: triangle !single
    nsize: 4
    args: -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 1 -dim 2 -dm_plex_max_projection_height 1 -petscspace_degree 4 -qorder 4
  test:
    suffix: nonconforming_simplex_2_fv
    requires: triangle
    nsize: 4
    args: -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 1 -dim 2 -num_comp 2
  test:
    suffix: nonconforming_simplex_3
    requires: ctetgen
    nsize: 4
    args: -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -simplex 1 -dim 3 -dm_plex_max_projection_height 2 -petscspace_degree 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL
  test:
    suffix: nonconforming_simplex_3_hi
    requires: ctetgen skip
    nsize: 4
    args: -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 1 -dim 3 -dm_plex_max_projection_height 2 -petscspace_degree 4 -qorder 4
  test:
    suffix: nonconforming_simplex_3_fv
    requires: ctetgen
    nsize: 4
    args: -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 1 -dim 3 -num_comp 3

TEST*/

/*
   # 2D Q_2 on a quadrilaterial Plex
  test:
    suffix: q2_2d_plex_0
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -convergence
  test:
    suffix: q2_2d_plex_1
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -porder 1
  test:
    suffix: q2_2d_plex_2
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -porder 2
  test:
    suffix: q2_2d_plex_3
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -porder 1 -shear_coords
  test:
    suffix: q2_2d_plex_4
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -qorder 2 -porder 2 -shear_coords
  test:
    suffix: q2_2d_plex_5
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -petscspace_poly_tensor 1 -qorder 2 -porder 0 -non_affine_coords
  test:
    suffix: q2_2d_plex_6
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -petscspace_poly_tensor 1 -qorder 2 -porder 1 -non_affine_coords
  test:
    suffix: q2_2d_plex_7
    args: -use_da 0 -simplex 0 -petscspace_degree 2 -petscspace_poly_tensor 1 -qorder 2 -porder 2 -non_affine_coords

  test:
    suffix: p1d_2d_6
    requires: pragmatic
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -convergence -conv_refine 0
  test:
    suffix: p1d_2d_7
    requires: pragmatic
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p1d_2d_8
    requires: pragmatic
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 2 -conv_refine 0
*/
