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
  PetscBool useDA;             /* Flag DMDA tensor product mesh */
  PetscBool interpolate;       /* Generate intermediate mesh elements */
  PetscReal refinementLimit;   /* The largest allowable cell volume */
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
  AppCtx *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = user->constants[d];
  return 0;
}
PetscErrorCode constantDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = 0.0;
  return 0;
}

/* u = x */
PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = coords[d];
  return 0;
}
PetscErrorCode linearDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  PetscInt d, e;
  for (d = 0; d < user->dim; ++d) {
    u[d] = 0.0;
    for (e = 0; e < user->dim; ++e) u[d] += (d == e ? 1.0 : 0.0) * n[e];
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
  AppCtx *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = tanh(coords[d] - 0.5);
  return 0;
}
PetscErrorCode trigDer(PetscInt dim, PetscReal time, const PetscReal coords[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *) ctx;
  PetscInt d;
  for (d = 0; d < user->dim; ++d) u[d] = 1.0/PetscSqr(cosh(coords[d] - 0.5)) * n[d];
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->simplex         = PETSC_TRUE;
  options->useDA           = PETSC_TRUE;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;
  options->qorder          = 0;
  options->numComponents   = 1;
  options->porder          = 0;
  options->convergence     = PETSC_FALSE;
  options->convRefine      = PETSC_TRUE;
  options->constraints     = PETSC_FALSE;
  options->tree            = PETSC_FALSE;
  options->treeCell        = 0;
  options->testFEjacobian  = PETSC_FALSE;
  options->testFVgrad      = PETSC_FALSE;
  options->testInjector    = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Projection Test Options", "DMPlex");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex3.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex3.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Flag for simplices or hexhedra", "ex3.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_da", "Flag for DMDA mesh", "ex3.c", options->useDA, &options->useDA, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex3.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex3.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qorder", "The quadrature order", "ex3.c", options->qorder, &options->qorder, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_comp", "The number of field components", "ex3.c", options->numComponents, &options->numComponents, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-porder", "The order of polynomials to test", "ex3.c", options->porder, &options->porder, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-convergence", "Check the convergence rate", "ex3.c", options->convergence, &options->convergence, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-conv_refine", "Use refinement for the convergence rate", "ex3.c", options->convRefine, &options->convRefine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-constraints", "Test local constraints (serial only)", "ex3.c", options->constraints, &options->constraints, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tree", "Test tree routines", "ex3.c", options->tree, &options->tree, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tree_cell", "cell to refine in tree test", "ex3.c", options->treeCell, &options->treeCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_fe_jacobian", "Test finite element Jacobian assembly", "ex3.c", options->testFEjacobian, &options->testFEjacobian, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_fv_grad", "Test finite volume gradient reconstruction", "ex3.c", options->testFVgrad, &options->testFVgrad, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_injector","Test finite element injection", "ex3.c", options->testInjector, &options->testInjector,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (user->simplex) {
    DM refinedMesh     = NULL;

    ierr = DMPlexCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
    /* Refine mesh using a volume constraint */
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
      ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, dm);CHKERRQ(ierr);
    } else {
      switch (user->dim) {
      case 2:
        ierr = DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, -2, -2, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, dm);CHKERRQ(ierr);
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
    DM distributedMesh = NULL;
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
    }
    else {
      ierr = DMPlexSetRefinementUniform(*dm, PETSC_TRUE);CHKERRQ(ierr);
    }
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
    if (user->simplex) {
      ierr = PetscObjectSetName((PetscObject) *dm, "Simplicial Mesh");CHKERRQ(ierr);
    }
    else {
      ierr = PetscObjectSetName((PetscObject) *dm, "Hexahedral Mesh");CHKERRQ(ierr);
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm,NULL,"-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "simple_mass"
static void simple_mass(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g0[])
{
  PetscInt d, e;
  for (d = 0, e = 0; d < dim; d++, e+=dim+1) {
    g0[e] = 1.;
  }
}

/* < \nabla v, 1/2(\nabla u + {\nabla u}^T) > */
#undef __FUNCT__
#define __FUNCT__ "symmetric_gradient_inner_product"
static void symmetric_gradient_inner_product(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                             PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar C[])
{
  PetscInt compI, compJ, d, e;

  for (compI = 0; compI < dim; ++compI) {
    for (compJ = 0; compJ < dim; ++compJ) {
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; e++) {
          if (d == e && d == compI && d == compJ) {
            C[((compI*dim+compJ)*dim+d)*dim+e] = 1.0;
          }
          else if ((d == compJ && e == compI) || (d == e && compI == compJ)) {
            C[((compI*dim+compJ)*dim+d)*dim+e] = 0.5;
          }
          else {
            C[((compI*dim+compJ)*dim+d)*dim+e] = 0.0;
          }
        }
      }
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
static PetscErrorCode SetupSection(DM dm, AppCtx *user)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
    MPI_Comm comm = PetscObjectComm((PetscObject)dm);

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
  ierr = DMSetField(dm, 0, (PetscObject) user->fe);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (!isPlex) {
      PetscSection    section;
      const PetscInt *numDof;
      PetscInt        numComp;

      ierr = PetscFEGetNumComponents(user->fe, &numComp);CHKERRQ(ierr);
      ierr = PetscFEGetNumDof(user->fe, &numDof);CHKERRQ(ierr);
      ierr = DMDACreateSection(dm, &numComp, numDof, NULL, &section);CHKERRQ(ierr);
      ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
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

      ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
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
#if 0
      {
        /* compare this to periodicity with DMDA: this is broken right now
         * because DMCreateMatrix() doesn't respect the default section that I
         * set */
        DM              dmda;
        PetscSection    section;
        const PetscInt *numDof;
        PetscInt        numComp;

                                                              /* periodic x */
        ierr = DMDACreate2d(PetscObjectComm((PetscObject)dm), DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, -2, -2, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, &dmda);CHKERRQ(ierr);
        ierr = DMDASetVertexCoordinates(dmda, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);


        ierr = PetscFEGetNumComponents(user->fe, &numComp);CHKERRQ(ierr);
        ierr = PetscFEGetNumDof(user->fe, &numDof);CHKERRQ(ierr);
        ierr = DMDACreateSection(dmda, &numComp, numDof, NULL, &section);CHKERRQ(ierr);
        ierr = DMSetDefaultSection(dmda, section);CHKERRQ(ierr);
        ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
        ierr = DMCreateMatrix(dmda,&mass);CHKERRQ(ierr);
        /* there isn't a DMDA equivalent of DMPlexSNESComputeJacobianFEM()
         * right now, but we can at least verify the nonzero structure */
        ierr = MatView(mass,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = MatDestroy(&mass);CHKERRQ(ierr);
        ierr = DMDestroy(&dmda);CHKERRQ(ierr);
      }
#endif
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TestFEJacobian"
static PetscErrorCode TestFEJacobian(DM dm, AppCtx *user)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (isPlex) {
    Vec          local;
    Mat          E;
    MatNullSpace sp;
    PetscBool    isNullSpace;
    PetscDS ds;

    if (user->numComponents != user->dim) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "The number of components %d must be equal to the dimension %d for this test", user->numComponents, user->dim);
    ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds,0,0,NULL,NULL,NULL,symmetric_gradient_inner_product);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm,&E);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local);CHKERRQ(ierr);
    ierr = DMPlexSNESComputeJacobianFEM(dm,local,E,E,NULL);CHKERRQ(ierr);
    ierr = DMPlexCreateRigidBody(dm,&sp);CHKERRQ(ierr);
    ierr = MatNullSpaceTest(sp,E,&isNullSpace);CHKERRQ(ierr);
    if (isNullSpace) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)dm),"Symmetric gradient null space: PASS\n");CHKERRQ(ierr);
    }
    else {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)dm),"Symmetric gradient null space: FAIL\n");CHKERRQ(ierr);
    }
    ierr = MatNullSpaceDestroy(&sp);CHKERRQ(ierr);
    ierr = MatDestroy(&E);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TestInjector"
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

#undef __FUNCT__
#define __FUNCT__ "TestFVGrad"
static PetscErrorCode TestFVGrad(DM dm, AppCtx *user)
{
  MPI_Comm comm;
  DM dmRedist, dmfv, dmgrad, dmCell, refTree;
  PetscFV fv;
  PetscInt nvecs, v, cStart, cEnd, cEndInterior;
  PetscMPIInt size;
  Vec cellgeom, grad, locGrad;
  const PetscScalar *cgeom;
  PetscReal allVecMaxDiff = 0., fvTol = 100. * PETSC_MACHINE_EPSILON;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)dm);
  /* duplicate DM, give dup. a FV discretization */
  ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_FALSE);CHKERRQ(ierr);
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
  ierr = DMSetField(dmfv, 0, (PetscObject) fv);CHKERRQ(ierr);
  ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
  if (refTree) {
    PetscDS ds;
    ierr = DMGetDS(dmfv,&ds);CHKERRQ(ierr);
    ierr = DMSetDS(refTree,ds);CHKERRQ(ierr);
  }
  ierr = DMPlexSNESGetGradientDM(dmfv, fv, &dmgrad);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmfv,0,&cStart,&cEnd);CHKERRQ(ierr);
  nvecs = user->dim * (user->dim+1) / 2;
  ierr = DMPlexSNESGetGeometryFVM(dmfv,NULL,&cellgeom,NULL);CHKERRQ(ierr);
  ierr = VecGetDM(cellgeom,&dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellgeom,&cgeom);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmgrad,&grad);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmgrad,&locGrad);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dmgrad,&cEndInterior,NULL,NULL,NULL);CHKERRQ(ierr);
  cEndInterior = (cEndInterior < 0) ? cEnd: cEndInterior;
  for (v = 0; v < nvecs; v++) {
    Vec locX;
    PetscInt c;
    PetscScalar trueGrad[3][3] = {{0.}};
    const PetscScalar *gradArray;
    PetscReal maxDiff, maxDiffGlob;

    ierr = DMGetLocalVector(dmfv,&locX);CHKERRQ(ierr);
    /* get the local projection of the rigid body mode */
    for (c = cStart; c < cEnd; c++) {
      PetscFVCellGeom *cg;
      PetscScalar cx[3] = {0.,0.,0.};

      ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
      if (v < user->dim) {
        cx[v] = 1.;
      }
      else {
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
      PetscInt i, j, k;
      PetscReal FrobDiff = 0.;

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
  }
  else {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm),"Finite volume gradient reconstruction: FAIL at tolerance %g with max difference %g\n",fvTol,allVecMaxDiff);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dmgrad,&locGrad);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmgrad,&grad);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellgeom,&cgeom);CHKERRQ(ierr);
  ierr = DMDestroy(&dmfv);CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeError"
static PetscErrorCode ComputeError(DM dm, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *),
                                   PetscErrorCode (**exactFuncDers)(PetscInt, PetscReal, const PetscReal[], const PetscReal[], PetscInt, PetscScalar *, void *),
                                   void **exactCtxs, PetscReal *error, PetscReal *errorDer, AppCtx *user)
{
  Vec            u;
  PetscReal      n[3] = {1.0, 1.0, 1.0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Project function into FE function space */
  ierr = DMProjectFunction(dm, 0.0, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  /* Compare approximation to exact in L_2 */
  ierr = DMComputeL2Diff(dm, 0.0, exactFuncs, exactCtxs, u, error);CHKERRQ(ierr);
  ierr = DMComputeL2GradientDiff(dm, 0.0, exactFuncDers, exactCtxs, u, n, errorDer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckFunctions"
static PetscErrorCode CheckFunctions(DM dm, PetscInt order, AppCtx *user)
{
  PetscErrorCode (*exactFuncs[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscErrorCode (*exactFuncDers[1]) (PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nf, PetscScalar *u, void *ctx);
  void            *exactCtxs[3];
  MPI_Comm         comm;
  PetscReal        error, errorDer, tol = PETSC_SMALL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  user->constants[0] = 1.0;
  user->constants[1] = 2.0;
  user->constants[2] = 3.0;
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

#undef __FUNCT__
#define __FUNCT__ "CheckInterpolation"
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
  PetscReal       error, errorDer, tol = 1.0e-10;
  PetscBool       isPlex, isDA;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  exactCtxs[0]       = user;
  exactCtxs[1]       = user;
  exactCtxs[2]       = user;
  user->constants[0] = 1.0;
  user->constants[1] = 2.0;
  user->constants[2] = 3.0;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMDA,   &isDA);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "CheckConvergence"
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

  PetscFunctionBegin;
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
      if (!user->simplex) {ierr = DMDASetVertexCoordinates(rdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);}
      ierr = SetupSection(rdm, user);CHKERRQ(ierr);
      ierr = ComputeError(rdm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user);CHKERRQ(ierr);
      p    = PetscLog2Real(errorOld/error);
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Function   convergence rate at refinement %D: %.2g\n", r, (double)p);CHKERRQ(ierr);
      p    = PetscLog2Real(errorDerOld/errorDer);
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Derivative convergence rate at refinement %D: %.2g\n", r, (double)p);CHKERRQ(ierr);
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
      if (!user->simplex) {ierr = DMDASetVertexCoordinates(cdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);}
      ierr = SetupSection(cdm, user);CHKERRQ(ierr);
      ierr = ComputeError(cdm, exactFuncs, exactFuncDers, exactCtxs, &error, &errorDer, user);CHKERRQ(ierr);
      /* ierr = ComputeLongestEdge(cdm, &len);CHKERRQ(ierr); */
      ierr = DMPlexGetHeightStratum(cdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
      len  = cEnd - cStart;
      rel  = error/errorOld;
      p    = PetscLogReal(rel) / PetscLogReal(lenOld / len);
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Function   convergence rate at coarsening %D: %.2g\n", c, (double)p);CHKERRQ(ierr);
      rel  = errorDer/errorDerOld;
      p    = PetscLogReal(rel) / PetscLogReal(lenOld / len);
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Derivative convergence rate at coarsening %D: %.2g\n", c, (double)p);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, user.dim, user.numComponents, user.simplex, NULL, user.qorder, &user.fe);CHKERRQ(ierr);
  ierr = SetupSection(dm, &user);CHKERRQ(ierr);
  if (user.testFEjacobian) {
    ierr = TestFEJacobian(dm, &user);CHKERRQ(ierr);
  }
  if (user.testFVgrad) {
    ierr = TestFVGrad(dm, &user);CHKERRQ(ierr);
  }
  if (user.testInjector) {
    ierr = TestInjector(dm, &user);CHKERRQ(ierr);
  }
  ierr = CheckFunctions(dm, user.porder, &user);CHKERRQ(ierr);
  if (user.dim == 2 && user.simplex == PETSC_TRUE && user.tree == PETSC_FALSE) {
    ierr = CheckInterpolation(dm, PETSC_FALSE, user.porder, &user);CHKERRQ(ierr);
    ierr = CheckInterpolation(dm, PETSC_TRUE,  user.porder, &user);CHKERRQ(ierr);
  }
  ierr = CheckConvergence(dm, 3, &user);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user.fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
