static char help[] = "Check that a DM can accurately represent and interpolate functions of a given polynomial order\n\n";

#include <petscdmplex.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscksp.h>

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
  PetscBool constraints;       /* Test local constraints */
  PetscBool tree;              /* Test tree routines */
  PetscInt  treeCell;          /* Cell to refine in tree test */
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

/* u = x^3 or u = (x^3, x^2y) or u = (x^2y, y^2z, z^2x) */
void cubic(const PetscReal coords[], PetscScalar *u, void *ctx)
{
  if (spdim > 2)      {u[0] = coords[0]*coords[0]*coords[1]; u[1] = coords[1]*coords[1]*coords[2]; u[2] = coords[2]*coords[2]*coords[0];}
  else if (spdim > 1) {u[0] = coords[0]*coords[0]*coords[0]; u[1] = coords[0]*coords[0]*coords[1];}
  else if (spdim > 0) {u[0] = coords[0]*coords[0]*coords[0];}
}
void cubicDer(const PetscReal coords[], const PetscReal n[], PetscScalar *u, void *ctx)
{
  if (spdim > 2)      {u[0] = 2.0*coords[0]*coords[1]*n[0] + coords[0]*coords[0]*n[1]; u[1] = 2.0*coords[1]*coords[2]*n[1] + coords[1]*coords[1]*n[2]; u[2] = 2.0*coords[2]*coords[0]*n[2] + coords[2]*coords[2]*n[0];}
  else if (spdim > 1) {u[0] = 3.0*coords[0]*coords[0]*n[0]; u[1] = 2.0*coords[0]*coords[1]*n[0] + coords[0]*coords[0]*n[1];}
  else if (spdim > 0) {u[0] = 3.0*coords[0]*coords[0]*n[0];}
}

/* u = sin(x) */
void trig(const PetscReal coords[], PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < spdim; ++d) u[d] = tanh(coords[d] - 0.5);
}
void trigDer(const PetscReal coords[], const PetscReal n[], PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < spdim; ++d) u[d] = 1.0/PetscSqr(cosh(coords[d] - 0.5)) * n[d];
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
  options->constraints     = PETSC_FALSE;
  options->tree            = PETSC_FALSE;
  options->treeCell        = 0;

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
  ierr = PetscOptionsBool("-constraints", "Test local constraints (serial only)", "ex3.c", options->constraints, &options->constraints, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tree", "Test tree routines", "ex3.c", options->tree, &options->tree, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tree_cell", "cell to refine in tree test", "ex3.c", options->treeCell, &options->treeCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  spdim = options->numComponents = options->dim;
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

      ierr = PetscOptionsGetInt(NULL,"-da_grid_x",&cells[0],NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetInt(NULL,"-da_grid_y",&cells[1],NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetInt(NULL,"-da_grid_z",&cells[2],NULL);CHKERRQ(ierr);
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
static void simple_mass(const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar g0[])
{
  PetscInt d, e;
  for (d = 0, e = 0; d < spdim; d++, e+=spdim+1) {
    g0[e] = 1.;
  }
}

/* < \nabla v, 1/2(\nabla u + {\nabla u}^T) > */
#undef __FUNCT__
#define __FUNCT__ "symmetric_gradient_inner_product"
void symmetric_gradient_inner_product (const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], const PetscReal x[], PetscScalar C[])
{
  const PetscInt dim   = spdim;
  const PetscInt Ncomp = spdim;
  PetscInt       compI, compJ, d, e;

  for (compI = 0; compI < Ncomp; ++compI) {
    for (compJ = 0; compJ < Ncomp; ++compJ) {
      for (d = 0; d < dim; ++d) {
        for (e = 0; e < dim; e++) {
          if (d == e && d == compI && d == compJ) {
            C[((compI*Ncomp+compJ)*dim+d)*dim+e] = 1.0;
          }
          else if ((d == compJ && e == compI) || (d == e && compI == compJ)) {
            C[((compI*Ncomp+compJ)*dim+d)*dim+e] = 0.5;
          }
          else {
            C[((compI*Ncomp+compJ)*dim+d)*dim+e] = 0.0;
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
    ierr = PetscOptionsGetInt(NULL,"-da_grid_x",&edgesx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,"-da_grid_y",&edgesy,NULL);CHKERRQ(ierr);
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
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (user->tree && isPlex) {
    Vec          local;
    Mat          E;
    MatNullSpace sp;
    PetscBool    isNullSpace;
    PetscDS ds;

    ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds,0,0,NULL,NULL,NULL,symmetric_gradient_inner_product);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm,&E);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local);CHKERRQ(ierr);
    ierr = DMPlexSNESComputeJacobianFEM(dm,local,E,E,NULL);CHKERRQ(ierr);
    ierr = DMPlexCreateRigidBody(dm,&sp);CHKERRQ(ierr);
    ierr = MatNullSpaceTest(sp,E,&isNullSpace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&sp);CHKERRQ(ierr);
    ierr = MatDestroy(&E);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeError_Plex"
static PetscErrorCode ComputeError_Plex(DM dm, void (**exactFuncs)(const PetscReal[], PetscScalar *, void *), void (**exactFuncDers)(const PetscReal[], const PetscReal[], PetscScalar *, void *),
                                        void **exactCtxs, PetscReal *error, PetscReal *errorDer, AppCtx *user)
{
  Vec            u;
  PetscReal      n[3] = {1.0, 1.0, 1.0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Project function into FE function space */
  ierr = DMPlexProjectFunction(dm, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  /* Compare approximation to exact in L_2 */
  ierr = DMPlexComputeL2Diff(dm, exactFuncs, exactCtxs, u, error);CHKERRQ(ierr);
  ierr = DMPlexComputeL2GradientDiff(dm, exactFuncDers, exactCtxs, u, n, errorDer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeError_DA"
static PetscErrorCode ComputeError_DA(DM dm, void (**exactFuncs)(const PetscReal[], PetscScalar *, void *), void (**exactFuncDers)(const PetscReal[], const PetscReal[], PetscScalar *, void *),
                                      void **exactCtxs, PetscReal *error, PetscReal *errorDer, AppCtx *user)
{
  Vec            u;
  PetscReal      n[3] = {1.0, 1.0, 1.0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Project function into FE function space */
  ierr = DMDAProjectFunction(dm, exactFuncs, exactCtxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  /* Compare approximation to exact in L_2 */
  ierr = DMDAComputeL2Diff(dm, exactFuncs, exactCtxs, u, error);CHKERRQ(ierr);
  ierr = DMDAComputeL2GradientDiff(dm, exactFuncDers, exactCtxs, u, n, errorDer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeError"
static PetscErrorCode ComputeError(DM dm, void (**exactFuncs)(const PetscReal[], PetscScalar *, void *), void (**exactFuncDers)(const PetscReal[], const PetscReal[], PetscScalar *, void *),
                                   void **exactCtxs, PetscReal *error, PetscReal *errorDer, AppCtx *user)
{
  PetscBool      isPlex, isDA;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMDA,   &isDA);CHKERRQ(ierr);
  if (isPlex) {
    ierr = ComputeError_Plex(dm, exactFuncs, exactFuncDers, exactCtxs, error, errorDer, user);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = ComputeError_DA(dm, exactFuncs, exactFuncDers, exactCtxs, error, errorDer, user);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM projection routine for this type of DM");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckFunctions"
static PetscErrorCode CheckFunctions(DM dm, PetscInt order, AppCtx *user)
{
  void          (*exactFuncs[1]) (const PetscReal x[], PetscScalar *u, void *ctx);
  void          (*exactFuncDers[1]) (const PetscReal x[], const PetscReal n[], PetscScalar *u, void *ctx);
  PetscReal       constants[3] = {1.0, 2.0, 3.0};
  void           *exactCtxs[3] = {NULL, NULL, NULL};
  MPI_Comm        comm;
  PetscReal       error, errorDer, tol = 1.0e-10;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
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
  void          (*exactFuncs[1]) (const PetscReal x[], PetscScalar *u, void *ctx);
  void          (*exactFuncDers[1]) (const PetscReal x[], const PetscReal n[], PetscScalar *u, void *ctx);
  PetscReal       n[3]         = {1.0, 1.0, 1.0};
  PetscReal       constants[3] = {1.0, 2.0, 3.0};
  void           *exactCtxs[3] = {NULL, NULL, NULL};
  DM              rdm, idm, fdm;
  Mat             Interp;
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
  ierr = DMRefine(dm, comm, &rdm);CHKERRQ(ierr);
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
  if (isPlex) {
    ierr = DMPlexProjectFunction(idm, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iu);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = DMDAProjectFunction(idm, exactFuncs, exactCtxs, INSERT_ALL_VALUES, iu);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM projection routine for this type of DM");
  /* Interpolate function into final FE function space */
  if (checkRestrict) {ierr = MatRestrict(Interp, iu, fu);CHKERRQ(ierr);ierr = VecPointwiseMult(fu, scaling, fu);CHKERRQ(ierr);}
  else               {ierr = MatInterpolate(Interp, iu, fu);CHKERRQ(ierr);}
  /* Compare approximation to exact in L_2 */
  if (isPlex) {
    ierr = DMPlexComputeL2Diff(fdm, exactFuncs, exactCtxs, fu, &error);CHKERRQ(ierr);
    ierr = DMPlexComputeL2GradientDiff(fdm, exactFuncDers, exactCtxs, fu, n, &errorDer);CHKERRQ(ierr);
  } else if (isDA) {
    ierr = DMDAComputeL2Diff(fdm, exactFuncs, exactCtxs, fu, &error);CHKERRQ(ierr);
    ierr = DMDAComputeL2GradientDiff(dm, exactFuncDers, exactCtxs, fu, n, &errorDer);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No FEM L_2 difference routine for this type of DM");
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
  DM             odm = dm, rdm = NULL;
  void         (*exactFuncs[1]) (const PetscReal x[], PetscScalar *u, void *ctx) = {trig};
  void         (*exactFuncDers[1]) (const PetscReal x[], const PetscReal n[], PetscScalar *u, void *ctx) = {trigDer};
  void          *exactCtxs[3] = {NULL, NULL, NULL};
  PetscInt       r;
  PetscReal      errorOld, errorDerOld, error, errorDer;
  double         p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!user->convergence) PetscFunctionReturn(0);
  ierr = PetscObjectReference((PetscObject) odm);CHKERRQ(ierr);
  ierr = ComputeError(odm, exactFuncs, exactFuncDers, exactCtxs, &errorOld, &errorDerOld, user);CHKERRQ(ierr);
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
