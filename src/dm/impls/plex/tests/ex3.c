static char help[] = "Check that a DM can accurately represent and interpolate functions of a given polynomial order\n\n";

#include <petscdmplex.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscksp.h>
#include <petscsnes.h>

typedef struct {
  /* Domain and mesh definition */
  PetscBool useDA;             /* Flag DMDA tensor product mesh */
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
  for (d = 0; d < dim; ++d) u[d] = user->constants[d];
  return 0;
}
PetscErrorCode constantDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
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
  if (dim > 2)      {u[0] = coords[0]*coords[1]; u[1] = coords[1]*coords[2]; u[2] = coords[2]*coords[0];}
  else if (dim > 1) {u[0] = coords[0]*coords[0]; u[1] = coords[0]*coords[1];}
  else if (dim > 0) {u[0] = coords[0]*coords[0];}
  return 0;
}
PetscErrorCode quadraticDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  if (dim > 2)      {u[0] = coords[1]*n[0] + coords[0]*n[1]; u[1] = coords[2]*n[1] + coords[1]*n[2]; u[2] = coords[2]*n[0] + coords[0]*n[2];}
  else if (dim > 1) {u[0] = 2.0*coords[0]*n[0]; u[1] = coords[1]*n[0] + coords[0]*n[1];}
  else if (dim > 0) {u[0] = 2.0*coords[0]*n[0];}
  return 0;
}

/* u = x^3 or u = (x^3, x^2y) or u = (x^2y, y^2z, z^2x) */
PetscErrorCode cubic(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  if (dim > 2)      {u[0] = coords[0]*coords[0]*coords[1]; u[1] = coords[1]*coords[1]*coords[2]; u[2] = coords[2]*coords[2]*coords[0];}
  else if (dim > 1) {u[0] = coords[0]*coords[0]*coords[0]; u[1] = coords[0]*coords[0]*coords[1];}
  else if (dim > 0) {u[0] = coords[0]*coords[0]*coords[0];}
  return 0;
}
PetscErrorCode cubicDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  if (dim > 2)      {u[0] = 2.0*coords[0]*coords[1]*n[0] + coords[0]*coords[0]*n[1]; u[1] = 2.0*coords[1]*coords[2]*n[1] + coords[1]*coords[1]*n[2]; u[2] = 2.0*coords[2]*coords[0]*n[2] + coords[2]*coords[2]*n[0];}
  else if (dim > 1) {u[0] = 3.0*coords[0]*coords[0]*n[0]; u[1] = 2.0*coords[0]*coords[1]*n[0] + coords[0]*coords[0]*n[1];}
  else if (dim > 0) {u[0] = 3.0*coords[0]*coords[0]*n[0];}
  return 0;
}

/* u = tanh(x) */
PetscErrorCode trig(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = PetscTanhReal(coords[d] - 0.5);
  return 0;
}
PetscErrorCode trigDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 1.0/PetscSqr(PetscCoshReal(coords[d] - 0.5)) * n[d];
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 3;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->useDA           = PETSC_FALSE;
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
  CHKERRQ(PetscOptionsBool("-use_da", "Flag for DMDA mesh", "ex3.c", options->useDA, &options->useDA, NULL));
  CHKERRQ(PetscOptionsBool("-shear_coords", "Transform coordinates with a shear", "ex3.c", options->shearCoords, &options->shearCoords, NULL));
  CHKERRQ(PetscOptionsBool("-non_affine_coords", "Transform coordinates with a non-affine transform", "ex3.c", options->nonaffineCoords, &options->nonaffineCoords, NULL));
  CHKERRQ(PetscOptionsBoundedInt("-qorder", "The quadrature order", "ex3.c", options->qorder, &options->qorder, NULL,0));
  CHKERRQ(PetscOptionsBoundedInt("-num_comp", "The number of field components", "ex3.c", options->numComponents, &options->numComponents, NULL,PETSC_DEFAULT));
  CHKERRQ(PetscOptionsBoundedInt("-porder", "The order of polynomials to test", "ex3.c", options->porder, &options->porder, NULL,0));
  CHKERRQ(PetscOptionsBool("-convergence", "Check the convergence rate", "ex3.c", options->convergence, &options->convergence, NULL));
  CHKERRQ(PetscOptionsBool("-conv_refine", "Use refinement for the convergence rate", "ex3.c", options->convRefine, &options->convRefine, NULL));
  CHKERRQ(PetscOptionsBool("-constraints", "Test local constraints (serial only)", "ex3.c", options->constraints, &options->constraints, NULL));
  CHKERRQ(PetscOptionsBool("-tree", "Test tree routines", "ex3.c", options->tree, &options->tree, NULL));
  CHKERRQ(PetscOptionsBoundedInt("-tree_cell", "cell to refine in tree test", "ex3.c", options->treeCell, &options->treeCell, NULL,0));
  CHKERRQ(PetscOptionsBool("-test_fe_jacobian", "Test finite element Jacobian assembly", "ex3.c", options->testFEjacobian, &options->testFEjacobian, NULL));
  CHKERRQ(PetscOptionsBool("-test_fv_grad", "Test finite volume gradient reconstruction", "ex3.c", options->testFVgrad, &options->testFVgrad, NULL));
  CHKERRQ(PetscOptionsBool("-test_injector","Test finite element injection", "ex3.c", options->testInjector, &options->testInjector,NULL));
  CHKERRQ(PetscOptionsRealArray("-constants","Set the constant values", "ex3.c", options->constants, &n,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode TransformCoordinates(DM dm, AppCtx *user)
{
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       vStart, vEnd, v;

  PetscFunctionBeginUser;
  if (user->nonaffineCoords) {
    /* x' = r^(1/p) (x/r), y' = r^(1/p) (y/r), z' = z */
    CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    CHKERRQ(VecGetArray(coordinates, &coords));
    for (v = vStart; v < vEnd; ++v) {
      PetscInt  dof, off;
      PetscReal p = 4.0, r;

      CHKERRQ(PetscSectionGetDof(coordSection, v, &dof));
      CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
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
    CHKERRQ(VecRestoreArray(coordinates, &coords));
  }
  if (user->shearCoords) {
    /* x' = x + m y + m z, y' = y + m z,  z' = z */
    CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    CHKERRQ(VecGetArray(coordinates, &coords));
    for (v = vStart; v < vEnd; ++v) {
      PetscInt  dof, off;
      PetscReal m = 1.0;

      CHKERRQ(PetscSectionGetDof(coordSection, v, &dof));
      CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
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
    CHKERRQ(VecRestoreArray(coordinates, &coords));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim = 2;
  PetscBool      simplex;

  PetscFunctionBeginUser;
  if (user->useDA) {
    switch (dim) {
    case 2:
      CHKERRQ(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 2, 2, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, dm));
      CHKERRQ(DMSetFromOptions(*dm));
      CHKERRQ(DMSetUp(*dm));
      CHKERRQ(DMDASetVertexCoordinates(*dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot create structured mesh of dimension %d", dim);
    }
    CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Hexahedral Mesh"));
  } else {
    CHKERRQ(DMCreate(comm, dm));
    CHKERRQ(DMSetType(*dm, DMPLEX));
    CHKERRQ(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(*dm));

    CHKERRQ(DMGetDimension(*dm, &dim));
    CHKERRQ(DMPlexIsSimplex(*dm, &simplex));
    CHKERRMPI(MPI_Bcast(&simplex, 1, MPIU_BOOL, 0, comm));
    if (user->tree) {
      DM refTree, ncdm = NULL;

      CHKERRQ(DMPlexCreateDefaultReferenceTree(comm,dim,simplex,&refTree));
      CHKERRQ(DMViewFromOptions(refTree,NULL,"-reftree_dm_view"));
      CHKERRQ(DMPlexSetReferenceTree(*dm,refTree));
      CHKERRQ(DMDestroy(&refTree));
      CHKERRQ(DMPlexTreeRefineCell(*dm,user->treeCell,&ncdm));
      if (ncdm) {
        CHKERRQ(DMDestroy(dm));
        *dm = ncdm;
        CHKERRQ(DMPlexSetRefinementUniform(*dm, PETSC_FALSE));
      }
      CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) *dm, "tree_"));
      CHKERRQ(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
      CHKERRQ(DMSetFromOptions(*dm));
      CHKERRQ(DMViewFromOptions(*dm,NULL,"-dm_view"));
    } else {
      CHKERRQ(DMPlexSetRefinementUniform(*dm, PETSC_TRUE));
    }
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) *dm, "dist_"));
    CHKERRQ(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(*dm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) *dm, NULL));
    if (simplex) CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Simplicial Mesh"));
    else         CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Hexahedral Mesh"));
  }
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(TransformCoordinates(*dm, user));
  CHKERRQ(DMViewFromOptions(*dm,NULL,"-dm_view"));
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
  PetscFunctionBeginUser;
  if (user->constraints) {
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

    CHKERRMPI(MPI_Comm_size(comm,&size));
    PetscCheckFalse(size > 1,comm,PETSC_ERR_SUP,"Local constraint test can only be performed in serial");

    /* we are going to test constraints by using them to enforce periodicity
     * in one direction, and comparing to the existing method of enforcing
     * periodicity */

    /* first create the coordinate section so that it does not clone the
     * constraints */
    CHKERRQ(DMGetCoordinateDM(dm,&coordDM));

    /* create the constrained-to-anchor section */
    CHKERRQ(DMPlexGetDepthStratum(dm,0,&vStart,&vEnd));
    CHKERRQ(DMPlexGetDepthStratum(dm,1,&fStart,&fEnd));
    CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF,&aSec));
    CHKERRQ(PetscSectionSetChart(aSec,PetscMin(fStart,vStart),PetscMax(fEnd,vEnd)));

    /* define the constraints */
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-da_grid_x",&edgesx,NULL));
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-da_grid_y",&edgesy,NULL));
    vertsx = edgesx + 1;
    vertsy = edgesy + 1;
    numConst = vertsy + edgesy;
    CHKERRQ(PetscMalloc1(numConst,&anchors));
    offset = 0;
    for (v = vStart + edgesx; v < vEnd; v+= vertsx) {
      CHKERRQ(PetscSectionSetDof(aSec,v,1));
      anchors[offset++] = v - edgesx;
    }
    for (f = fStart + edgesx * vertsy + edgesx * edgesy; f < fEnd; f++) {
      CHKERRQ(PetscSectionSetDof(aSec,f,1));
      anchors[offset++] = f - edgesx * edgesy;
    }
    CHKERRQ(PetscSectionSetUp(aSec));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,numConst,anchors,PETSC_OWN_POINTER,&aIS));

    CHKERRQ(DMPlexSetAnchors(dm,aSec,aIS));
    CHKERRQ(PetscSectionDestroy(&aSec));
    CHKERRQ(ISDestroy(&aIS));
  }
  CHKERRQ(DMSetNumFields(dm, 1));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) user->fe));
  CHKERRQ(DMCreateDS(dm));
  if (user->constraints) {
    /* test getting local constraint matrix that matches section */
    PetscSection aSec;
    IS           aIS;

    CHKERRQ(DMPlexGetAnchors(dm,&aSec,&aIS));
    if (aSec) {
      PetscDS         ds;
      PetscSection    cSec, section;
      PetscInt        cStart, cEnd, c, numComp;
      Mat             cMat, mass;
      Vec             local;
      const PetscInt *anchors;

      CHKERRQ(DMGetLocalSection(dm,&section));
      /* this creates the matrix and preallocates the matrix structure: we
       * just have to fill in the values */
      CHKERRQ(DMGetDefaultConstraints(dm,&cSec,&cMat,NULL));
      CHKERRQ(PetscSectionGetChart(cSec,&cStart,&cEnd));
      CHKERRQ(ISGetIndices(aIS,&anchors));
      CHKERRQ(PetscFEGetNumComponents(user->fe, &numComp));
      for (c = cStart; c < cEnd; c++) {
        PetscInt cDof;

        /* is this point constrained? (does it have an anchor?) */
        CHKERRQ(PetscSectionGetDof(aSec,c,&cDof));
        if (cDof) {
          PetscInt cOff, a, aDof, aOff, j;
          PetscCheckFalse(cDof != 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Found %d anchor points: should be just one",cDof);

          /* find the anchor point */
          CHKERRQ(PetscSectionGetOffset(aSec,c,&cOff));
          a    = anchors[cOff];

          /* find the constrained dofs (row in constraint matrix) */
          CHKERRQ(PetscSectionGetDof(cSec,c,&cDof));
          CHKERRQ(PetscSectionGetOffset(cSec,c,&cOff));

          /* find the anchor dofs (column in constraint matrix) */
          CHKERRQ(PetscSectionGetDof(section,a,&aDof));
          CHKERRQ(PetscSectionGetOffset(section,a,&aOff));

          PetscCheckFalse(cDof != aDof,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point and anchor have different number of dofs: %d, %d",cDof,aDof);
          PetscCheckFalse(cDof % numComp,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point dofs not divisible by field components: %d, %d",cDof,numComp);

          /* put in a simple equality constraint */
          for (j = 0; j < cDof; j++) {
            CHKERRQ(MatSetValue(cMat,cOff+j,aOff+j,1.,INSERT_VALUES));
          }
        }
      }
      CHKERRQ(MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY));
      CHKERRQ(ISRestoreIndices(aIS,&anchors));

      /* Now that we have constructed the constraint matrix, any FE matrix
       * that we construct will apply the constraints during construction */

      CHKERRQ(DMCreateMatrix(dm,&mass));
      /* get a dummy local variable to serve as the solution */
      CHKERRQ(DMGetLocalVector(dm,&local));
      CHKERRQ(DMGetDS(dm,&ds));
      /* set the jacobian to be the mass matrix */
      CHKERRQ(PetscDSSetJacobian(ds, 0, 0, simple_mass, NULL,  NULL, NULL));
      /* build the mass matrix */
      CHKERRQ(DMPlexSNESComputeJacobianFEM(dm,local,mass,mass,NULL));
      CHKERRQ(MatView(mass,PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(MatDestroy(&mass));
      CHKERRQ(DMRestoreLocalVector(dm,&local));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFEJacobian(DM dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  if (!user->useDA) {
    Vec          local;
    const Vec    *vecs;
    Mat          E;
    MatNullSpace sp;
    PetscBool    isNullSpace, hasConst;
    PetscInt     dim, n, i;
    Vec          res = NULL, localX, localRes;
    PetscDS      ds;

    CHKERRQ(DMGetDimension(dm, &dim));
    PetscCheckFalse(user->numComponents != dim,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "The number of components %d must be equal to the dimension %d for this test", user->numComponents, dim);
    CHKERRQ(DMGetDS(dm,&ds));
    CHKERRQ(PetscDSSetJacobian(ds,0,0,NULL,NULL,NULL,symmetric_gradient_inner_product));
    CHKERRQ(DMCreateMatrix(dm,&E));
    CHKERRQ(DMGetLocalVector(dm,&local));
    CHKERRQ(DMPlexSNESComputeJacobianFEM(dm,local,E,E,NULL));
    CHKERRQ(DMPlexCreateRigidBody(dm,0,&sp));
    CHKERRQ(MatNullSpaceGetVecs(sp,&hasConst,&n,&vecs));
    if (n) CHKERRQ(VecDuplicate(vecs[0],&res));
    CHKERRQ(DMCreateLocalVector(dm,&localX));
    CHKERRQ(DMCreateLocalVector(dm,&localRes));
    for (i = 0; i < n; i++) { /* also test via matrix-free Jacobian application */
      PetscReal resNorm;

      CHKERRQ(VecSet(localRes,0.));
      CHKERRQ(VecSet(localX,0.));
      CHKERRQ(VecSet(local,0.));
      CHKERRQ(VecSet(res,0.));
      CHKERRQ(DMGlobalToLocalBegin(dm,vecs[i],INSERT_VALUES,localX));
      CHKERRQ(DMGlobalToLocalEnd(dm,vecs[i],INSERT_VALUES,localX));
      CHKERRQ(DMSNESComputeJacobianAction(dm,local,localX,localRes,NULL));
      CHKERRQ(DMLocalToGlobalBegin(dm,localRes,ADD_VALUES,res));
      CHKERRQ(DMLocalToGlobalEnd(dm,localRes,ADD_VALUES,res));
      CHKERRQ(VecNorm(res,NORM_2,&resNorm));
      if (resNorm > PETSC_SMALL) {
        CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dm),"Symmetric gradient action null space vector %D residual: %E\n",i,resNorm));
      }
    }
    CHKERRQ(VecDestroy(&localRes));
    CHKERRQ(VecDestroy(&localX));
    CHKERRQ(VecDestroy(&res));
    CHKERRQ(MatNullSpaceTest(sp,E,&isNullSpace));
    if (isNullSpace) {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dm),"Symmetric gradient null space: PASS\n"));
    } else {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dm),"Symmetric gradient null space: FAIL\n"));
    }
    CHKERRQ(MatNullSpaceDestroy(&sp));
    CHKERRQ(MatDestroy(&E));
    CHKERRQ(DMRestoreLocalVector(dm,&local));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestInjector(DM dm, AppCtx *user)
{
  DM             refTree;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetReferenceTree(dm,&refTree));
  if (refTree) {
    Mat inj;

    CHKERRQ(DMPlexComputeInjectorReferenceTree(refTree,&inj));
    CHKERRQ(PetscObjectSetName((PetscObject)inj,"Reference Tree Injector"));
    CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    if (rank == 0) {
      CHKERRQ(MatView(inj,PETSC_VIEWER_STDOUT_SELF));
    }
    CHKERRQ(MatDestroy(&inj));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFVGrad(DM dm, AppCtx *user)
{
  MPI_Comm          comm;
  DM                dmRedist, dmfv, dmgrad, dmCell, refTree;
  PetscFV           fv;
  PetscInt          dim, nvecs, v, cStart, cEnd, cEndInterior;
  PetscMPIInt       size;
  Vec               cellgeom, grad, locGrad;
  const PetscScalar *cgeom;
  PetscReal         allVecMaxDiff = 0., fvTol = 100. * PETSC_MACHINE_EPSILON;

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)dm);
  CHKERRQ(DMGetDimension(dm, &dim));
  /* duplicate DM, give dup. a FV discretization */
  CHKERRQ(DMSetBasicAdjacency(dm,PETSC_TRUE,PETSC_FALSE));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  dmRedist = NULL;
  if (size > 1) {
    CHKERRQ(DMPlexDistributeOverlap(dm,1,NULL,&dmRedist));
  }
  if (!dmRedist) {
    dmRedist = dm;
    CHKERRQ(PetscObjectReference((PetscObject)dmRedist));
  }
  CHKERRQ(PetscFVCreate(comm,&fv));
  CHKERRQ(PetscFVSetType(fv,PETSCFVLEASTSQUARES));
  CHKERRQ(PetscFVSetNumComponents(fv,user->numComponents));
  CHKERRQ(PetscFVSetSpatialDimension(fv,dim));
  CHKERRQ(PetscFVSetFromOptions(fv));
  CHKERRQ(PetscFVSetUp(fv));
  CHKERRQ(DMPlexConstructGhostCells(dmRedist,NULL,NULL,&dmfv));
  CHKERRQ(DMDestroy(&dmRedist));
  CHKERRQ(DMSetNumFields(dmfv,1));
  CHKERRQ(DMSetField(dmfv, 0, NULL, (PetscObject) fv));
  CHKERRQ(DMCreateDS(dmfv));
  CHKERRQ(DMPlexGetReferenceTree(dm,&refTree));
  if (refTree) CHKERRQ(DMCopyDisc(dmfv,refTree));
  CHKERRQ(DMPlexGetGradientDM(dmfv, fv, &dmgrad));
  CHKERRQ(DMPlexGetHeightStratum(dmfv,0,&cStart,&cEnd));
  nvecs = dim * (dim+1) / 2;
  CHKERRQ(DMPlexGetGeometryFVM(dmfv,NULL,&cellgeom,NULL));
  CHKERRQ(VecGetDM(cellgeom,&dmCell));
  CHKERRQ(VecGetArrayRead(cellgeom,&cgeom));
  CHKERRQ(DMGetGlobalVector(dmgrad,&grad));
  CHKERRQ(DMGetLocalVector(dmgrad,&locGrad));
  CHKERRQ(DMPlexGetGhostCellStratum(dmgrad,&cEndInterior,NULL));
  cEndInterior = (cEndInterior < 0) ? cEnd: cEndInterior;
  for (v = 0; v < nvecs; v++) {
    Vec               locX;
    PetscInt          c;
    PetscScalar       trueGrad[3][3] = {{0.}};
    const PetscScalar *gradArray;
    PetscReal         maxDiff, maxDiffGlob;

    CHKERRQ(DMGetLocalVector(dmfv,&locX));
    /* get the local projection of the rigid body mode */
    for (c = cStart; c < cEnd; c++) {
      PetscFVCellGeom *cg;
      PetscScalar     cx[3] = {0.,0.,0.};

      CHKERRQ(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
      if (v < dim) {
        cx[v] = 1.;
      } else {
        PetscInt w = v - dim;

        cx[(w + 1) % dim] =  cg->centroid[(w + 2) % dim];
        cx[(w + 2) % dim] = -cg->centroid[(w + 1) % dim];
      }
      CHKERRQ(DMPlexVecSetClosure(dmfv,NULL,locX,c,cx,INSERT_ALL_VALUES));
    }
    /* TODO: this isn't in any header */
    CHKERRQ(DMPlexReconstructGradientsFVM(dmfv,locX,grad));
    CHKERRQ(DMGlobalToLocalBegin(dmgrad,grad,INSERT_VALUES,locGrad));
    CHKERRQ(DMGlobalToLocalEnd(dmgrad,grad,INSERT_VALUES,locGrad));
    CHKERRQ(VecGetArrayRead(locGrad,&gradArray));
    /* compare computed gradient to exact gradient */
    if (v >= dim) {
      PetscInt w = v - dim;

      trueGrad[(w + 1) % dim][(w + 2) % dim] =  1.;
      trueGrad[(w + 2) % dim][(w + 1) % dim] = -1.;
    }
    maxDiff = 0.;
    for (c = cStart; c < cEndInterior; c++) {
      PetscScalar *compGrad;
      PetscInt    i, j, k;
      PetscReal   FrobDiff = 0.;

      CHKERRQ(DMPlexPointLocalRead(dmgrad, c, gradArray, &compGrad));

      for (i = 0, k = 0; i < dim; i++) {
        for (j = 0; j < dim; j++, k++) {
          PetscScalar diff = compGrad[k] - trueGrad[i][j];
          FrobDiff += PetscRealPart(diff * PetscConj(diff));
        }
      }
      FrobDiff = PetscSqrtReal(FrobDiff);
      maxDiff  = PetscMax(maxDiff,FrobDiff);
    }
    CHKERRMPI(MPI_Allreduce(&maxDiff,&maxDiffGlob,1,MPIU_REAL,MPIU_MAX,comm));
    allVecMaxDiff = PetscMax(allVecMaxDiff,maxDiffGlob);
    CHKERRQ(VecRestoreArrayRead(locGrad,&gradArray));
    CHKERRQ(DMRestoreLocalVector(dmfv,&locX));
  }
  if (allVecMaxDiff < fvTol) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dm),"Finite volume gradient reconstruction: PASS\n"));
  } else {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dm),"Finite volume gradient reconstruction: FAIL at tolerance %g with max difference %g\n",fvTol,allVecMaxDiff));
  }
  CHKERRQ(DMRestoreLocalVector(dmgrad,&locGrad));
  CHKERRQ(DMRestoreGlobalVector(dmgrad,&grad));
  CHKERRQ(VecRestoreArrayRead(cellgeom,&cgeom));
  CHKERRQ(DMDestroy(&dmfv));
  CHKERRQ(PetscFVDestroy(&fv));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeError(DM dm, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *),
                                   PetscErrorCode (**exactFuncDers)(PetscInt, PetscReal, const PetscReal[], const PetscReal[], PetscInt, PetscScalar *, void *),
                                   void **exactCtxs, PetscReal *error, PetscReal *errorDer, AppCtx *user)
{
  Vec            u;
  PetscReal      n[3] = {1.0, 1.0, 1.0};

  PetscFunctionBeginUser;
  CHKERRQ(DMGetGlobalVector(dm, &u));
  /* Project function into FE function space */
  CHKERRQ(DMProjectFunction(dm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u));
  CHKERRQ(VecViewFromOptions(u, NULL, "-projection_view"));
  /* Compare approximation to exact in L_2 */
  CHKERRQ(DMComputeL2Diff(dm, 0.0, exactFuncs, exactCtxs, u, error));
  CHKERRQ(DMComputeL2GradientDiff(dm, 0.0, exactFuncDers, exactCtxs, u, n, errorDer));
  CHKERRQ(DMRestoreGlobalVector(dm, &u));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckFunctions(DM dm, PetscInt order, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1]) (PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx);
  void            *exactCtxs[3];
  MPI_Comm         comm;
  PetscReal        error, errorDer, tol = PETSC_SMALL;

  PetscFunctionBeginUser;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
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
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for order %d", order);
  }
  CHKERRQ(ComputeError(dm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user));
  /* Report result */
  if (error > tol)    CHKERRQ(PetscPrintf(comm, "Function tests FAIL for order %D at tolerance %g error %g\n", order, (double)tol,(double) error));
  else                CHKERRQ(PetscPrintf(comm, "Function tests pass for order %D at tolerance %g\n", order, (double)tol));
  if (errorDer > tol) CHKERRQ(PetscPrintf(comm, "Function tests FAIL for order %D derivatives at tolerance %g error %g\n", order, (double)tol, (double)errorDer));
  else                CHKERRQ(PetscPrintf(comm, "Function tests pass for order %D derivatives at tolerance %g\n", order, (double)tol));
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
  PetscInt        dim;
  PetscReal       error, errorDer, tol = PETSC_SMALL;

  PetscFunctionBeginUser;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMRefine(dm, comm, &rdm));
  CHKERRQ(DMSetCoarseDM(rdm, dm));
  CHKERRQ(DMPlexSetRegularRefinement(rdm, user->convRefine));
  if (user->tree) {
    DM refTree;
    CHKERRQ(DMPlexGetReferenceTree(dm,&refTree));
    CHKERRQ(DMPlexSetReferenceTree(rdm,refTree));
  }
  if (user->useDA) CHKERRQ(DMDASetVertexCoordinates(rdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  CHKERRQ(SetupSection(rdm, user));
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
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not determine functions to test for dimension %D order %D", dim, order);
  }
  idm  = checkRestrict ? rdm :  dm;
  fdm  = checkRestrict ?  dm : rdm;
  CHKERRQ(DMGetGlobalVector(idm, &iu));
  CHKERRQ(DMGetGlobalVector(fdm, &fu));
  CHKERRQ(DMSetApplicationContext(dm, user));
  CHKERRQ(DMSetApplicationContext(rdm, user));
  CHKERRQ(DMCreateInterpolation(dm, rdm, &Interp, &scaling));
  /* Project function into initial FE function space */
  CHKERRQ(DMProjectFunction(idm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iu));
  /* Interpolate function into final FE function space */
  if (checkRestrict) {CHKERRQ(MatRestrict(Interp, iu, fu));CHKERRQ(VecPointwiseMult(fu, scaling, fu));}
  else               CHKERRQ(MatInterpolate(Interp, iu, fu));
  /* Compare approximation to exact in L_2 */
  CHKERRQ(DMComputeL2Diff(fdm, 0.0, exactFuncs, exactCtxs, fu, &error));
  CHKERRQ(DMComputeL2GradientDiff(fdm, 0.0, exactFuncDers, exactCtxs, fu, n, &errorDer));
  /* Report result */
  if (error > tol)    CHKERRQ(PetscPrintf(comm, "Interpolation tests FAIL for order %D at tolerance %g error %g\n", order, (double)tol, (double)error));
  else                CHKERRQ(PetscPrintf(comm, "Interpolation tests pass for order %D at tolerance %g\n", order, (double)tol));
  if (errorDer > tol) CHKERRQ(PetscPrintf(comm, "Interpolation tests FAIL for order %D derivatives at tolerance %g error %g\n", order, (double)tol, (double)errorDer));
  else                CHKERRQ(PetscPrintf(comm, "Interpolation tests pass for order %D derivatives at tolerance %g\n", order, (double)tol));
  CHKERRQ(DMRestoreGlobalVector(idm, &iu));
  CHKERRQ(DMRestoreGlobalVector(fdm, &fu));
  CHKERRQ(MatDestroy(&Interp));
  CHKERRQ(VecDestroy(&scaling));
  CHKERRQ(DMDestroy(&rdm));
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

  PetscFunctionBeginUser;
  if (!user->convergence) PetscFunctionReturn(0);
  exactCtxs[0] = user;
  exactCtxs[1] = user;
  exactCtxs[2] = user;
  CHKERRQ(PetscObjectReference((PetscObject) odm));
  if (!user->convRefine) {
    for (r = 0; r < Nr; ++r) {
      CHKERRQ(DMRefine(odm, PetscObjectComm((PetscObject) dm), &rdm));
      CHKERRQ(DMDestroy(&odm));
      odm  = rdm;
    }
    CHKERRQ(SetupSection(odm, user));
  }
  CHKERRQ(ComputeError(odm, exactFuncs, exactFuncDers, exactCtxs, &errorOld, &errorDerOld, user));
  if (user->convRefine) {
    for (r = 0; r < Nr; ++r) {
      CHKERRQ(DMRefine(odm, PetscObjectComm((PetscObject) dm), &rdm));
      if (user->useDA) CHKERRQ(DMDASetVertexCoordinates(rdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
      CHKERRQ(SetupSection(rdm, user));
      CHKERRQ(ComputeError(rdm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user));
      p    = PetscLog2Real(errorOld/error);
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "Function   convergence rate at refinement %D: %.2f\n", r, (double)p));
      p    = PetscLog2Real(errorDerOld/errorDer);
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "Derivative convergence rate at refinement %D: %.2f\n", r, (double)p));
      CHKERRQ(DMDestroy(&odm));
      odm         = rdm;
      errorOld    = error;
      errorDerOld = errorDer;
    }
  } else {
    /* CHKERRQ(ComputeLongestEdge(dm, &lenOld)); */
    CHKERRQ(DMPlexGetHeightStratum(odm, 0, &cStart, &cEnd));
    lenOld = cEnd - cStart;
    for (c = 0; c < Nr; ++c) {
      CHKERRQ(DMCoarsen(odm, PetscObjectComm((PetscObject) dm), &cdm));
      if (user->useDA) CHKERRQ(DMDASetVertexCoordinates(cdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
      CHKERRQ(SetupSection(cdm, user));
      CHKERRQ(ComputeError(cdm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user));
      /* CHKERRQ(ComputeLongestEdge(cdm, &len)); */
      CHKERRQ(DMPlexGetHeightStratum(cdm, 0, &cStart, &cEnd));
      len  = cEnd - cStart;
      rel  = error/errorOld;
      p    = PetscLogReal(rel) / PetscLogReal(lenOld / len);
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "Function   convergence rate at coarsening %D: %.2f\n", c, (double)p));
      rel  = errorDer/errorDerOld;
      p    = PetscLogReal(rel) / PetscLogReal(lenOld / len);
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "Derivative convergence rate at coarsening %D: %.2f\n", c, (double)p));
      CHKERRQ(DMDestroy(&odm));
      odm         = cdm;
      errorOld    = error;
      errorDerOld = errorDer;
      lenOld      = len;
    }
  }
  CHKERRQ(DMDestroy(&odm));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  PetscInt       dim = 2;
  PetscBool      simplex = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  if (!user.useDA) {
    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  }
  CHKERRQ(DMPlexMetricSetFromOptions(dm));
  user.numComponents = user.numComponents < 0 ? dim : user.numComponents;
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_WORLD, dim, user.numComponents, simplex, NULL, user.qorder, &user.fe));
  CHKERRQ(SetupSection(dm, &user));
  if (user.testFEjacobian) CHKERRQ(TestFEJacobian(dm, &user));
  if (user.testFVgrad) CHKERRQ(TestFVGrad(dm, &user));
  if (user.testInjector) CHKERRQ(TestInjector(dm, &user));
  CHKERRQ(CheckFunctions(dm, user.porder, &user));
  {
    PetscDualSpace dsp;
    PetscInt       k;

    CHKERRQ(PetscFEGetDualSpace(user.fe, &dsp));
    CHKERRQ(PetscDualSpaceGetDeRahm(dsp, &k));
    if (dim == 2 && simplex == PETSC_TRUE && user.tree == PETSC_FALSE && k == 0) {
      CHKERRQ(CheckInterpolation(dm, PETSC_FALSE, user.porder, &user));
      CHKERRQ(CheckInterpolation(dm, PETSC_TRUE,  user.porder, &user));
    }
  }
  CHKERRQ(CheckConvergence(dm, 3, &user));
  CHKERRQ(PetscFEDestroy(&user.fe));
  CHKERRQ(DMDestroy(&dm));
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
    requires: triangle mmg
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -convergence -conv_refine 0
  test:
    suffix: p1_2d_4
    requires: triangle mmg
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p1_2d_5
    requires: triangle mmg
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 2 -conv_refine 0

  # 3D P_1 on a tetrahedron
  test:
    suffix: p1_3d_0
    requires: ctetgen
    args: -dm_plex_dim 3 -petscspace_degree 1 -qorder 1 -convergence
  test:
    suffix: p1_3d_1
    requires: ctetgen
    args: -dm_plex_dim 3 -petscspace_degree 1 -qorder 1 -porder 1
  test:
    suffix: p1_3d_2
    requires: ctetgen
    args: -dm_plex_dim 3 -petscspace_degree 1 -qorder 1 -porder 2
  test:
    suffix: p1_3d_3
    requires: ctetgen mmg
    args: -dm_plex_dim 3 -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -convergence -conv_refine 0
  test:
    suffix: p1_3d_4
    requires: ctetgen mmg
    args: -dm_plex_dim 3 -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p1_3d_5
    requires: ctetgen mmg
    args: -dm_plex_dim 3 -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 2 -conv_refine 0

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
    requires: triangle mmg
    args: -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -convergence -conv_refine 0
  test:
    suffix: p2_2d_4
    requires: triangle mmg
    args: -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p2_2d_5
    requires: triangle mmg
    args: -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder 2 -conv_refine 0

  # 3D P_2 on a tetrahedron
  test:
    suffix: p2_3d_0
    requires: ctetgen
    args: -dm_plex_dim 3 -petscspace_degree 2 -qorder 2 -convergence
  test:
    suffix: p2_3d_1
    requires: ctetgen
    args: -dm_plex_dim 3 -petscspace_degree 2 -qorder 2 -porder 1
  test:
    suffix: p2_3d_2
    requires: ctetgen
    args: -dm_plex_dim 3 -petscspace_degree 2 -qorder 2 -porder 2
  test:
    suffix: p2_3d_3
    requires: ctetgen mmg
    args: -dm_plex_dim 3 -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -convergence -conv_refine 0
  test:
    suffix: p2_3d_4
    requires: ctetgen mmg
    args: -dm_plex_dim 3 -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p2_3d_5
    requires: ctetgen mmg
    args: -dm_plex_dim 3 -petscspace_degree 2 -qorder 2 -dm_plex_hash_location -porder 2 -conv_refine 0

  # 2D Q_1 on a quadrilaterial DA
  test:
    suffix: q1_2d_da_0
    requires: broken
    args: -use_da 1 -petscspace_degree 1 -qorder 1 -convergence
  test:
    suffix: q1_2d_da_1
    requires: broken
    args: -use_da 1 -petscspace_degree 1 -qorder 1 -porder 1
  test:
    suffix: q1_2d_da_2
    requires: broken
    args: -use_da 1 -petscspace_degree 1 -qorder 1 -porder 2

  # 2D Q_1 on a quadrilaterial Plex
  test:
    suffix: q1_2d_plex_0
    args: -dm_plex_simplex 0 -petscspace_degree 1 -qorder 1 -convergence
  test:
    suffix: q1_2d_plex_1
    args: -dm_plex_simplex 0 -petscspace_degree 1 -qorder 1 -porder 1
  test:
    suffix: q1_2d_plex_2
    args: -dm_plex_simplex 0 -petscspace_degree 1 -qorder 1 -porder 2
  test:
    suffix: q1_2d_plex_3
    args: -dm_plex_simplex 0 -petscspace_degree 1 -qorder 1 -porder 1 -shear_coords
  test:
    suffix: q1_2d_plex_4
    args: -dm_plex_simplex 0 -petscspace_degree 1 -qorder 1 -porder 2 -shear_coords
  test:
    suffix: q1_2d_plex_5
    args: -dm_plex_simplex 0 -petscspace_degree 1 -petscspace_type tensor -qorder 1 -porder 0 -non_affine_coords -convergence
  test:
    suffix: q1_2d_plex_6
    args: -dm_plex_simplex 0 -petscspace_degree 1 -petscspace_type tensor -qorder 1 -porder 1 -non_affine_coords -convergence
  test:
    suffix: q1_2d_plex_7
    args: -dm_plex_simplex 0 -petscspace_degree 1 -petscspace_type tensor -qorder 1 -porder 2 -non_affine_coords -convergence

  # 2D Q_2 on a quadrilaterial
  test:
    suffix: q2_2d_plex_0
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -convergence
  test:
    suffix: q2_2d_plex_1
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder 1
  test:
    suffix: q2_2d_plex_2
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder 2
  test:
    suffix: q2_2d_plex_3
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder 1 -shear_coords
  test:
    suffix: q2_2d_plex_4
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder 2 -shear_coords
  test:
    suffix: q2_2d_plex_5
    args: -dm_plex_simplex 0 -petscspace_degree 2 -petscspace_type tensor -qorder 2 -porder 0 -non_affine_coords -convergence
  test:
    suffix: q2_2d_plex_6
    args: -dm_plex_simplex 0 -petscspace_degree 2 -petscspace_type tensor -qorder 2 -porder 1 -non_affine_coords -convergence
  test:
    suffix: q2_2d_plex_7
    args: -dm_plex_simplex 0 -petscspace_degree 2 -petscspace_type tensor -qorder 2 -porder 2 -non_affine_coords -convergence

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
    requires: triangle mmg
    args: -petscspace_degree 3 -qorder 3 -dm_plex_hash_location -convergence -conv_refine 0
  test:
    suffix: p3_2d_5
    requires: triangle mmg
    args: -petscspace_degree 3 -qorder 3 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p3_2d_6
    requires: triangle mmg
    args: -petscspace_degree 3 -qorder 3 -dm_plex_hash_location -porder 3 -conv_refine 0

  # 2D Q_3 on a quadrilaterial
  test:
    suffix: q3_2d_0
    requires: !single
    args: -dm_plex_simplex 0 -petscspace_degree 3 -qorder 3 -convergence
  test:
    suffix: q3_2d_1
    requires: !single
    args: -dm_plex_simplex 0 -petscspace_degree 3 -qorder 3 -porder 1
  test:
    suffix: q3_2d_2
    requires: !single
    args: -dm_plex_simplex 0 -petscspace_degree 3 -qorder 3 -porder 2
  test:
    suffix: q3_2d_3
    requires: !single
    args: -dm_plex_simplex 0 -petscspace_degree 3 -qorder 3 -porder 3

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
    args: -dm_plex_simplex 0 -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -convergence
    filter: sed  -e "s/convergence rate at refinement 0: 2/convergence rate at refinement 0: 1.9/g"
  test:
    suffix: p1d_2d_4
    requires: triangle
    args: -dm_plex_simplex 0 -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -porder 1
  test:
    suffix: p1d_2d_5
    requires: triangle
    args: -dm_plex_simplex 0 -petscspace_degree 1 -petscdualspace_lagrange_continuity 0 -qorder 1 -porder 2

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
          -petscdualspace_lagrange_tensor 1 \
          -dm_plex_simplex 0 -num_comp 2 -qorder 1 -convergence
  test:
    suffix: bdm1q_2d_1
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_type bdm \
          -petscdualspace_lagrange_tensor 1 \
          -dm_plex_simplex 0 -num_comp 2 -qorder 1 -porder 1
  test:
    suffix: bdm1q_2d_2
    requires: triangle
    args: -petscspace_degree 1 -petscdualspace_type bdm \
          -petscdualspace_lagrange_tensor 1 \
          -dm_plex_simplex 0 -num_comp 2 -qorder 1 -porder 2

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
    args: -dm_plex_simplex 0 -petscspace_degree 1 -qorder 2 -porder 1
  test:
    suffix: q1_quad_5
    args: -dm_plex_simplex 0 -petscspace_degree 1 -qorder 5 -porder 1
  test:
    suffix: q2_quad_3
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 3 -porder 1
  test:
    suffix: q2_quad_5
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 5 -porder 1

  # Nonconforming tests
  test:
    suffix: constraints
    args: -dm_coord_space 0 -dm_plex_simplex 0 -petscspace_type tensor -petscspace_degree 1 -qorder 0 -constraints
  test:
    suffix: nonconforming_tensor_2
    nsize: 4
    args: -dist_dm_distribute -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -dm_plex_simplex 0 -dm_plex_max_projection_height 1 -petscspace_type tensor -petscspace_degree 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL
  test:
    suffix: nonconforming_tensor_3
    nsize: 4
    args: -dist_dm_distribute -test_fe_jacobian -petscpartitioner_type simple -tree -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_plex_max_projection_height 2 -petscspace_type tensor -petscspace_degree 1 -qorder 1 -dm_view ascii::ASCII_INFO_DETAIL
  test:
    suffix: nonconforming_tensor_2_fv
    nsize: 4
    args: -dist_dm_distribute -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -dm_plex_simplex 0 -num_comp 2
  test:
    suffix: nonconforming_tensor_3_fv
    nsize: 4
    args: -dist_dm_distribute -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -num_comp 3
  test:
    suffix: nonconforming_tensor_2_hi
    requires: !single
    nsize: 4
    args: -dist_dm_distribute -test_fe_jacobian -petscpartitioner_type simple -tree -dm_plex_simplex 0 -dm_plex_max_projection_height 1 -petscspace_type tensor -petscspace_degree 4 -qorder 4
  test:
    suffix: nonconforming_tensor_3_hi
    requires: !single skip
    nsize: 4
    args: -dist_dm_distribute -test_fe_jacobian -petscpartitioner_type simple -tree -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_plex_max_projection_height 2 -petscspace_type tensor -petscspace_degree 4 -qorder 4
  test:
    suffix: nonconforming_simplex_2
    requires: triangle
    nsize: 4
    args: -dist_dm_distribute -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -dm_plex_max_projection_height 1 -petscspace_degree 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL
  test:
    suffix: nonconforming_simplex_2_hi
    requires: triangle !single
    nsize: 4
    args: -dist_dm_distribute -test_fe_jacobian -petscpartitioner_type simple -tree -dm_plex_max_projection_height 1 -petscspace_degree 4 -qorder 4
  test:
    suffix: nonconforming_simplex_2_fv
    requires: triangle
    nsize: 4
    args: -dist_dm_distribute -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -num_comp 2
  test:
    suffix: nonconforming_simplex_3
    requires: ctetgen
    nsize: 4
    args: -dist_dm_distribute -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -dm_plex_dim 3 -dm_plex_max_projection_height 2 -petscspace_degree 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL
  test:
    suffix: nonconforming_simplex_3_hi
    requires: ctetgen skip
    nsize: 4
    args: -dist_dm_distribute -test_fe_jacobian -petscpartitioner_type simple -tree -dm_plex_dim 3 -dm_plex_max_projection_height 2 -petscspace_degree 4 -qorder 4
  test:
    suffix: nonconforming_simplex_3_fv
    requires: ctetgen
    nsize: 4
    args: -dist_dm_distribute -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -dm_plex_dim 3 -num_comp 3

  # 3D WXY on a triangular prism
  test:
    suffix: wxy_0
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism -qorder 2 -porder 0 \
          -petscspace_type sum \
          -petscspace_variables 3 \
          -petscspace_components 3 \
          -petscspace_sum_spaces 2 \
          -petscspace_sum_concatenate false \
          -sumcomp_0_petscspace_variables 3 \
          -sumcomp_0_petscspace_components 3 \
          -sumcomp_0_petscspace_degree 1 \
          -sumcomp_1_petscspace_variables 3 \
          -sumcomp_1_petscspace_components 3 \
          -sumcomp_1_petscspace_type wxy \
          -petscdualspace_refcell triangular_prism \
          -petscdualspace_form_degree 0 \
          -petscdualspace_order 1 \
          -petscdualspace_components 3

TEST*/

/*
   # 2D Q_2 on a quadrilaterial Plex
  test:
    suffix: q2_2d_plex_0
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -convergence
  test:
    suffix: q2_2d_plex_1
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder 1
  test:
    suffix: q2_2d_plex_2
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder 2
  test:
    suffix: q2_2d_plex_3
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder 1 -shear_coords
  test:
    suffix: q2_2d_plex_4
    args: -dm_plex_simplex 0 -petscspace_degree 2 -qorder 2 -porder 2 -shear_coords
  test:
    suffix: q2_2d_plex_5
    args: -dm_plex_simplex 0 -petscspace_degree 2 -petscspace_poly_tensor 1 -qorder 2 -porder 0 -non_affine_coords
  test:
    suffix: q2_2d_plex_6
    args: -dm_plex_simplex 0 -petscspace_degree 2 -petscspace_poly_tensor 1 -qorder 2 -porder 1 -non_affine_coords
  test:
    suffix: q2_2d_plex_7
    args: -dm_plex_simplex 0 -petscspace_degree 2 -petscspace_poly_tensor 1 -qorder 2 -porder 2 -non_affine_coords

  test:
    suffix: p1d_2d_6
    requires: mmg
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -convergence -conv_refine 0
  test:
    suffix: p1d_2d_7
    requires: mmg
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 1 -conv_refine 0
  test:
    suffix: p1d_2d_8
    requires: mmg
    args: -petscspace_degree 1 -qorder 1 -dm_plex_hash_location -porder 2 -conv_refine 0
*/
