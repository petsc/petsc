#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#include <petscblaslapack.h>
#include <petsc/private/hashsetij.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/petscfvimpl.h>

PetscBool Clementcite = PETSC_FALSE;
const char ClementCitation[] = "@article{clement1975approximation,\n"
                               "  title   = {Approximation by finite element functions using local regularization},\n"
                               "  author  = {Philippe Cl{\\'e}ment},\n"
                               "  journal = {Revue fran{\\c{c}}aise d'automatique, informatique, recherche op{\\'e}rationnelle. Analyse num{\\'e}rique},\n"
                               "  volume  = {9},\n"
                               "  number  = {R2},\n"
                               "  pages   = {77--84},\n"
                               "  year    = {1975}\n}\n";

static PetscErrorCode DMPlexConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex));
  if (isPlex) {
    *plex = dm;
    CHKERRQ(PetscObjectReference((PetscObject) dm));
  } else {
    CHKERRQ(PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex));
    if (!*plex) {
      CHKERRQ(DMConvert(dm, DMPLEX, plex));
      CHKERRQ(PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex));
      if (copy) {
        DMSubDomainHookLink link;

        CHKERRQ(DMCopyAuxiliaryVec(dm, *plex));
        /* Run the subdomain hook (this will copy the DMSNES/DMTS) */
        for (link = dm->subdomainhook; link; link = link->next) {
          if (link->ddhook) CHKERRQ((*link->ddhook)(dm, *plex, link->ctx));
        }
      }
    } else {
      CHKERRQ(PetscObjectReference((PetscObject) *plex));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscContainerUserDestroy_PetscFEGeom (void *ctx)
{
  PetscFEGeom *geom = (PetscFEGeom *) ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscFEGeomDestroy(&geom));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetFEGeom(DMField coordField, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetId((PetscObject)quad,&id));
  CHKERRQ(PetscSNPrintf(composeStr, 32, "DMPlexGetFEGeom_%x\n", id));
  CHKERRQ(PetscObjectQuery((PetscObject) pointIS, composeStr, (PetscObject *) &container));
  if (container) {
    CHKERRQ(PetscContainerGetPointer(container, (void **) geom));
  } else {
    CHKERRQ(DMFieldCreateFEGeom(coordField, pointIS, quad, faceData, geom));
    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF,&container));
    CHKERRQ(PetscContainerSetPointer(container, (void *) *geom));
    CHKERRQ(PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom));
    CHKERRQ(PetscObjectCompose((PetscObject) pointIS, composeStr, (PetscObject) container));
    CHKERRQ(PetscContainerDestroy(&container));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexRestoreFEGeom(DMField coordField, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  PetscFunctionBegin;
  *geom = NULL;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetScale - Get the scale for the specified fundamental unit

  Not collective

  Input Parameters:
+ dm   - the DM
- unit - The SI unit

  Output Parameter:
. scale - The value used to scale all quantities with this unit

  Level: advanced

.seealso: DMPlexSetScale(), PetscUnit
@*/
PetscErrorCode DMPlexGetScale(DM dm, PetscUnit unit, PetscReal *scale)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(scale, 3);
  *scale = mesh->scale[unit];
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetScale - Set the scale for the specified fundamental unit

  Not collective

  Input Parameters:
+ dm   - the DM
. unit - The SI unit
- scale - The value used to scale all quantities with this unit

  Level: advanced

.seealso: DMPlexGetScale(), PetscUnit
@*/
PetscErrorCode DMPlexSetScale(DM dm, PetscUnit unit, PetscReal scale)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->scale[unit] = scale;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexProjectRigidBody_Private(PetscInt dim, PetscReal t, const PetscReal X[], PetscInt Nc, PetscScalar *mode, void *ctx)
{
  const PetscInt eps[3][3][3] = {{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}, {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}}, {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}};
  PetscInt *ctxInt  = (PetscInt *) ctx;
  PetscInt  dim2    = ctxInt[0];
  PetscInt  d       = ctxInt[1];
  PetscInt  i, j, k = dim > 2 ? d - dim : d;

  PetscFunctionBegin;
  PetscCheck(dim == dim2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input dimension %D does not match context dimension %D", dim, dim2);
  for (i = 0; i < dim; i++) mode[i] = 0.;
  if (d < dim) {
    mode[d] = 1.; /* Translation along axis d */
  } else {
    for (i = 0; i < dim; i++) {
      for (j = 0; j < dim; j++) {
        mode[j] += eps[i][j][k]*X[i]; /* Rotation about axis d */
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateRigidBody - For the default global section, create rigid body modes by function space interpolation

  Collective on dm

  Input Parameters:
+ dm - the DM
- field - The field number for the rigid body space, or 0 for the default

  Output Parameter:
. sp - the null space

  Note: This is necessary to provide a suitable coarse space for algebraic multigrid

  Level: advanced

.seealso: MatNullSpaceCreate(), PCGAMG
@*/
PetscErrorCode DMPlexCreateRigidBody(DM dm, PetscInt field, MatNullSpace *sp)
{
  PetscErrorCode (**func)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *);
  MPI_Comm          comm;
  Vec               mode[6];
  PetscSection      section, globalSection;
  PetscInt          dim, dimEmbed, Nf, n, m, mmin, d, i, j;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &dimEmbed));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  PetscCheckFalse(Nf && (field < 0 || field >= Nf),comm, PETSC_ERR_ARG_OUTOFRANGE, "Field %D is not in [0, Nf)", field, Nf);
  if (dim == 1 && Nf < 2) {
    CHKERRQ(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, sp));
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetGlobalSection(dm, &globalSection));
  CHKERRQ(PetscSectionGetConstrainedStorageSize(globalSection, &n));
  CHKERRQ(PetscCalloc1(Nf, &func));
  m    = (dim*(dim+1))/2;
  CHKERRQ(VecCreate(comm, &mode[0]));
  CHKERRQ(VecSetType(mode[0], dm->vectype));
  CHKERRQ(VecSetSizes(mode[0], n, PETSC_DETERMINE));
  CHKERRQ(VecSetUp(mode[0]));
  CHKERRQ(VecGetSize(mode[0], &n));
  mmin = PetscMin(m, n);
  func[field] = DMPlexProjectRigidBody_Private;
  for (i = 1; i < m; ++i) CHKERRQ(VecDuplicate(mode[0], &mode[i]));
  for (d = 0; d < m; d++) {
    PetscInt ctx[2];
    void    *voidctx = (void *) (&ctx[0]);

    ctx[0] = dimEmbed;
    ctx[1] = d;
    CHKERRQ(DMProjectFunction(dm, 0.0, func, &voidctx, INSERT_VALUES, mode[d]));
  }
  /* Orthonormalize system */
  for (i = 0; i < mmin; ++i) {
    PetscScalar dots[6];

    CHKERRQ(VecNormalize(mode[i], NULL));
    CHKERRQ(VecMDot(mode[i], mmin-i-1, mode+i+1, dots+i+1));
    for (j = i+1; j < mmin; ++j) {
      dots[j] *= -1.0;
      CHKERRQ(VecAXPY(mode[j], dots[j], mode[i]));
    }
  }
  CHKERRQ(MatNullSpaceCreate(comm, PETSC_FALSE, mmin, mode, sp));
  for (i = 0; i < m; ++i) CHKERRQ(VecDestroy(&mode[i]));
  CHKERRQ(PetscFree(func));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateRigidBodies - For the default global section, create rigid body modes by function space interpolation

  Collective on dm

  Input Parameters:
+ dm    - the DM
. nb    - The number of bodies
. label - The DMLabel marking each domain
. nids  - The number of ids per body
- ids   - An array of the label ids in sequence for each domain

  Output Parameter:
. sp - the null space

  Note: This is necessary to provide a suitable coarse space for algebraic multigrid

  Level: advanced

.seealso: MatNullSpaceCreate()
@*/
PetscErrorCode DMPlexCreateRigidBodies(DM dm, PetscInt nb, DMLabel label, const PetscInt nids[], const PetscInt ids[], MatNullSpace *sp)
{
  MPI_Comm       comm;
  PetscSection   section, globalSection;
  Vec           *mode;
  PetscScalar   *dots;
  PetscInt       dim, dimEmbed, n, m, b, d, i, j, off;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &dimEmbed));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetGlobalSection(dm, &globalSection));
  CHKERRQ(PetscSectionGetConstrainedStorageSize(globalSection, &n));
  m    = nb * (dim*(dim+1))/2;
  CHKERRQ(PetscMalloc2(m, &mode, m, &dots));
  CHKERRQ(VecCreate(comm, &mode[0]));
  CHKERRQ(VecSetSizes(mode[0], n, PETSC_DETERMINE));
  CHKERRQ(VecSetUp(mode[0]));
  for (i = 1; i < m; ++i) CHKERRQ(VecDuplicate(mode[0], &mode[i]));
  for (b = 0, off = 0; b < nb; ++b) {
    for (d = 0; d < m/nb; ++d) {
      PetscInt         ctx[2];
      PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *) = DMPlexProjectRigidBody_Private;
      void            *voidctx = (void *) (&ctx[0]);

      ctx[0] = dimEmbed;
      ctx[1] = d;
      CHKERRQ(DMProjectFunctionLabel(dm, 0.0, label, nids[b], &ids[off], 0, NULL, &func, &voidctx, INSERT_VALUES, mode[d]));
      off   += nids[b];
    }
  }
  /* Orthonormalize system */
  for (i = 0; i < m; ++i) {
    PetscScalar dots[6];

    CHKERRQ(VecNormalize(mode[i], NULL));
    CHKERRQ(VecMDot(mode[i], m-i-1, mode+i+1, dots+i+1));
    for (j = i+1; j < m; ++j) {
      dots[j] *= -1.0;
      CHKERRQ(VecAXPY(mode[j], dots[j], mode[i]));
    }
  }
  CHKERRQ(MatNullSpaceCreate(comm, PETSC_FALSE, m, mode, sp));
  for (i = 0; i< m; ++i) CHKERRQ(VecDestroy(&mode[i]));
  CHKERRQ(PetscFree2(mode, dots));
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetMaxProjectionHeight - In DMPlexProjectXXXLocal() functions, the projected values of a basis function's dofs
  are computed by associating the basis function with one of the mesh points in its transitively-closed support, and
  evaluating the dual space basis of that point.  A basis function is associated with the point in its
  transitively-closed support whose mesh height is highest (w.r.t. DAG height), but not greater than the maximum
  projection height, which is set with this function.  By default, the maximum projection height is zero, which means
  that only mesh cells are used to project basis functions.  A height of one, for example, evaluates a cell-interior
  basis functions using its cells dual space basis, but all other basis functions with the dual space basis of a face.

  Input Parameters:
+ dm - the DMPlex object
- height - the maximum projection height >= 0

  Level: advanced

.seealso: DMPlexGetMaxProjectionHeight(), DMProjectFunctionLocal(), DMProjectFunctionLabelLocal()
@*/
PetscErrorCode DMPlexSetMaxProjectionHeight(DM dm, PetscInt height)
{
  DM_Plex *plex = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  plex->maxProjectionHeight = height;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetMaxProjectionHeight - Get the maximum height (w.r.t. DAG) of mesh points used to evaluate dual bases in
  DMPlexProjectXXXLocal() functions.

  Input Parameters:
. dm - the DMPlex object

  Output Parameters:
. height - the maximum projection height

  Level: intermediate

.seealso: DMPlexSetMaxProjectionHeight(), DMProjectFunctionLocal(), DMProjectFunctionLabelLocal()
@*/
PetscErrorCode DMPlexGetMaxProjectionHeight(DM dm, PetscInt *height)
{
  DM_Plex *plex = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *height = plex->maxProjectionHeight;
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal    alpha; /* The first Euler angle, and in 2D the only one */
  PetscReal    beta;  /* The second Euler angle */
  PetscReal    gamma; /* The third Euler angle */
  PetscInt     dim;   /* The dimension of R */
  PetscScalar *R;     /* The rotation matrix, transforming a vector in the local basis to the global basis */
  PetscScalar *RT;    /* The transposed rotation matrix, transforming a vector in the global basis to the local basis */
} RotCtx;

/*
  Note: Following https://en.wikipedia.org/wiki/Euler_angles, we will specify Euler angles by extrinsic rotations, meaning that
  we rotate with respect to a fixed initial coordinate system, the local basis (x-y-z). The global basis (X-Y-Z) is reached as follows:
  $ The XYZ system rotates about the z axis by alpha. The X axis is now at angle alpha with respect to the x axis.
  $ The XYZ system rotates again about the x axis by beta. The Z axis is now at angle beta with respect to the z axis.
  $ The XYZ system rotates a third time about the z axis by gamma.
*/
static PetscErrorCode DMPlexBasisTransformSetUp_Rotation_Internal(DM dm, void *ctx)
{
  RotCtx        *rc  = (RotCtx *) ctx;
  PetscInt       dim = rc->dim;
  PetscReal      c1, s1, c2, s2, c3, s3;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc2(PetscSqr(dim), &rc->R, PetscSqr(dim), &rc->RT));
  switch (dim) {
  case 2:
    c1 = PetscCosReal(rc->alpha);s1 = PetscSinReal(rc->alpha);
    rc->R[0] =  c1;rc->R[1] = s1;
    rc->R[2] = -s1;rc->R[3] = c1;
    CHKERRQ(PetscArraycpy(rc->RT, rc->R, PetscSqr(dim)));
    DMPlex_Transpose2D_Internal(rc->RT);
    break;
  case 3:
    c1 = PetscCosReal(rc->alpha);s1 = PetscSinReal(rc->alpha);
    c2 = PetscCosReal(rc->beta); s2 = PetscSinReal(rc->beta);
    c3 = PetscCosReal(rc->gamma);s3 = PetscSinReal(rc->gamma);
    rc->R[0] =  c1*c3 - c2*s1*s3;rc->R[1] =  c3*s1    + c1*c2*s3;rc->R[2] = s2*s3;
    rc->R[3] = -c1*s3 - c2*c3*s1;rc->R[4] =  c1*c2*c3 - s1*s3;   rc->R[5] = c3*s2;
    rc->R[6] =  s1*s2;           rc->R[7] = -c1*s2;              rc->R[8] = c2;
    CHKERRQ(PetscArraycpy(rc->RT, rc->R, PetscSqr(dim)));
    DMPlex_Transpose3D_Internal(rc->RT);
    break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransformDestroy_Rotation_Internal(DM dm, void *ctx)
{
  RotCtx        *rc = (RotCtx *) ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscFree2(rc->R, rc->RT));
  CHKERRQ(PetscFree(rc));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransformGetMatrix_Rotation_Internal(DM dm, const PetscReal x[], PetscBool l2g, const PetscScalar **A, void *ctx)
{
  RotCtx *rc = (RotCtx *) ctx;

  PetscFunctionBeginHot;
  PetscValidPointer(ctx, 5);
  if (l2g) {*A = rc->R;}
  else     {*A = rc->RT;}
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexBasisTransformApplyReal_Internal(DM dm, const PetscReal x[], PetscBool l2g, PetscInt dim, const PetscReal *y, PetscReal *z, void *ctx)
{
  PetscFunctionBegin;
  #if defined(PETSC_USE_COMPLEX)
  switch (dim) {
    case 2:
    {
      PetscScalar yt[2] = {y[0], y[1]}, zt[2] = {0.0,0.0};

      CHKERRQ(DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, yt, zt, ctx));
      z[0] = PetscRealPart(zt[0]); z[1] = PetscRealPart(zt[1]);
    }
    break;
    case 3:
    {
      PetscScalar yt[3] = {y[0], y[1], y[2]}, zt[3] = {0.0,0.0,0.0};

      CHKERRQ(DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, yt, zt, ctx));
      z[0] = PetscRealPart(zt[0]); z[1] = PetscRealPart(zt[1]); z[2] = PetscRealPart(zt[2]);
    }
    break;
  }
  #else
  CHKERRQ(DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, y, z, ctx));
  #endif
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexBasisTransformApply_Internal(DM dm, const PetscReal x[], PetscBool l2g, PetscInt dim, const PetscScalar *y, PetscScalar *z, void *ctx)
{
  const PetscScalar *A;

  PetscFunctionBeginHot;
  CHKERRQ((*dm->transformGetMatrix)(dm, x, l2g, &A, ctx));
  switch (dim) {
  case 2: DMPlex_Mult2D_Internal(A, 1, y, z);break;
  case 3: DMPlex_Mult3D_Internal(A, 1, y, z);break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransformField_Internal(DM dm, DM tdm, Vec tv, PetscInt p, PetscInt f, PetscBool l2g, PetscScalar *a)
{
  PetscSection       ts;
  const PetscScalar *ta, *tva;
  PetscInt           dof;

  PetscFunctionBeginHot;
  CHKERRQ(DMGetLocalSection(tdm, &ts));
  CHKERRQ(PetscSectionGetFieldDof(ts, p, f, &dof));
  CHKERRQ(VecGetArrayRead(tv, &ta));
  CHKERRQ(DMPlexPointLocalFieldRead(tdm, p, f, ta, &tva));
  if (l2g) {
    switch (dof) {
    case 4: DMPlex_Mult2D_Internal(tva, 1, a, a);break;
    case 9: DMPlex_Mult3D_Internal(tva, 1, a, a);break;
    }
  } else {
    switch (dof) {
    case 4: DMPlex_MultTranspose2D_Internal(tva, 1, a, a);break;
    case 9: DMPlex_MultTranspose3D_Internal(tva, 1, a, a);break;
    }
  }
  CHKERRQ(VecRestoreArrayRead(tv, &ta));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransformFieldTensor_Internal(DM dm, DM tdm, Vec tv, PetscInt pf, PetscInt f, PetscInt pg, PetscInt g, PetscBool l2g, PetscInt lda, PetscScalar *a)
{
  PetscSection       s, ts;
  const PetscScalar *ta, *tvaf, *tvag;
  PetscInt           fdof, gdof, fpdof, gpdof;

  PetscFunctionBeginHot;
  CHKERRQ(DMGetLocalSection(dm, &s));
  CHKERRQ(DMGetLocalSection(tdm, &ts));
  CHKERRQ(PetscSectionGetFieldDof(s, pf, f, &fpdof));
  CHKERRQ(PetscSectionGetFieldDof(s, pg, g, &gpdof));
  CHKERRQ(PetscSectionGetFieldDof(ts, pf, f, &fdof));
  CHKERRQ(PetscSectionGetFieldDof(ts, pg, g, &gdof));
  CHKERRQ(VecGetArrayRead(tv, &ta));
  CHKERRQ(DMPlexPointLocalFieldRead(tdm, pf, f, ta, &tvaf));
  CHKERRQ(DMPlexPointLocalFieldRead(tdm, pg, g, ta, &tvag));
  if (l2g) {
    switch (fdof) {
    case 4: DMPlex_MatMult2D_Internal(tvaf, gpdof, lda, a, a);break;
    case 9: DMPlex_MatMult3D_Internal(tvaf, gpdof, lda, a, a);break;
    }
    switch (gdof) {
    case 4: DMPlex_MatMultTransposeLeft2D_Internal(tvag, fpdof, lda, a, a);break;
    case 9: DMPlex_MatMultTransposeLeft3D_Internal(tvag, fpdof, lda, a, a);break;
    }
  } else {
    switch (fdof) {
    case 4: DMPlex_MatMultTranspose2D_Internal(tvaf, gpdof, lda, a, a);break;
    case 9: DMPlex_MatMultTranspose3D_Internal(tvaf, gpdof, lda, a, a);break;
    }
    switch (gdof) {
    case 4: DMPlex_MatMultLeft2D_Internal(tvag, fpdof, lda, a, a);break;
    case 9: DMPlex_MatMultLeft3D_Internal(tvag, fpdof, lda, a, a);break;
    }
  }
  CHKERRQ(VecRestoreArrayRead(tv, &ta));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexBasisTransformPoint_Internal(DM dm, DM tdm, Vec tv, PetscInt p, PetscBool fieldActive[], PetscBool l2g, PetscScalar *a)
{
  PetscSection    s;
  PetscSection    clSection;
  IS              clPoints;
  const PetscInt *clp;
  PetscInt       *points = NULL;
  PetscInt        Nf, f, Np, cp, dof, d = 0;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalSection(dm, &s));
  CHKERRQ(PetscSectionGetNumFields(s, &Nf));
  CHKERRQ(DMPlexGetCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp));
  for (f = 0; f < Nf; ++f) {
    for (cp = 0; cp < Np*2; cp += 2) {
      CHKERRQ(PetscSectionGetFieldDof(s, points[cp], f, &dof));
      if (!dof) continue;
      if (fieldActive[f]) CHKERRQ(DMPlexBasisTransformField_Internal(dm, tdm, tv, points[cp], f, l2g, &a[d]));
      d += dof;
    }
  }
  CHKERRQ(DMPlexRestoreCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexBasisTransformPointTensor_Internal(DM dm, DM tdm, Vec tv, PetscInt p, PetscBool l2g, PetscInt lda, PetscScalar *a)
{
  PetscSection    s;
  PetscSection    clSection;
  IS              clPoints;
  const PetscInt *clp;
  PetscInt       *points = NULL;
  PetscInt        Nf, f, g, Np, cpf, cpg, fdof, gdof, r, c = 0;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalSection(dm, &s));
  CHKERRQ(PetscSectionGetNumFields(s, &Nf));
  CHKERRQ(DMPlexGetCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp));
  for (f = 0, r = 0; f < Nf; ++f) {
    for (cpf = 0; cpf < Np*2; cpf += 2) {
      CHKERRQ(PetscSectionGetFieldDof(s, points[cpf], f, &fdof));
      for (g = 0, c = 0; g < Nf; ++g) {
        for (cpg = 0; cpg < Np*2; cpg += 2) {
          CHKERRQ(PetscSectionGetFieldDof(s, points[cpg], g, &gdof));
          CHKERRQ(DMPlexBasisTransformFieldTensor_Internal(dm, tdm, tv, points[cpf], f, points[cpg], g, l2g, lda, &a[r*lda+c]));
          c += gdof;
        }
      }
      PetscCheck(c == lda,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of columns %D should be %D", c, lda);
      r += fdof;
    }
  }
  PetscCheck(r == lda,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of rows %D should be %D", c, lda);
  CHKERRQ(DMPlexRestoreCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransform_Internal(DM dm, Vec lv, PetscBool l2g)
{
  DM                 tdm;
  Vec                tv;
  PetscSection       ts, s;
  const PetscScalar *ta;
  PetscScalar       *a, *va;
  PetscInt           pStart, pEnd, p, Nf, f;

  PetscFunctionBegin;
  CHKERRQ(DMGetBasisTransformDM_Internal(dm, &tdm));
  CHKERRQ(DMGetBasisTransformVec_Internal(dm, &tv));
  CHKERRQ(DMGetLocalSection(tdm, &ts));
  CHKERRQ(DMGetLocalSection(dm, &s));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(PetscSectionGetNumFields(s, &Nf));
  CHKERRQ(VecGetArray(lv, &a));
  CHKERRQ(VecGetArrayRead(tv, &ta));
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < Nf; ++f) {
      CHKERRQ(DMPlexPointLocalFieldRef(dm, p, f, a, &va));
      CHKERRQ(DMPlexBasisTransformField_Internal(dm, tdm, tv, p, f, l2g, va));
    }
  }
  CHKERRQ(VecRestoreArray(lv, &a));
  CHKERRQ(VecRestoreArrayRead(tv, &ta));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToLocalBasis - Transform the values in the given local vector from the global basis to the local basis

  Input Parameters:
+ dm - The DM
- lv - A local vector with values in the global basis

  Output Parameters:
. lv - A local vector with values in the local basis

  Note: This method is only intended to be called inside DMGlobalToLocal(). It is unlikely that a user will have a local vector full of coefficients for the global basis unless they are reimplementing GlobalToLocal.

  Level: developer

.seealso: DMPlexLocalToGlobalBasis(), DMGetLocalSection(), DMPlexCreateBasisRotation()
@*/
PetscErrorCode DMPlexGlobalToLocalBasis(DM dm, Vec lv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lv, VEC_CLASSID, 2);
  CHKERRQ(DMPlexBasisTransform_Internal(dm, lv, PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@
  DMPlexLocalToGlobalBasis - Transform the values in the given local vector from the local basis to the global basis

  Input Parameters:
+ dm - The DM
- lv - A local vector with values in the local basis

  Output Parameters:
. lv - A local vector with values in the global basis

  Note: This method is only intended to be called inside DMGlobalToLocal(). It is unlikely that a user would want a local vector full of coefficients for the global basis unless they are reimplementing GlobalToLocal.

  Level: developer

.seealso: DMPlexGlobalToLocalBasis(), DMGetLocalSection(), DMPlexCreateBasisRotation()
@*/
PetscErrorCode DMPlexLocalToGlobalBasis(DM dm, Vec lv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lv, VEC_CLASSID, 2);
  CHKERRQ(DMPlexBasisTransform_Internal(dm, lv, PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateBasisRotation - Create an internal transformation from the global basis, used to specify boundary conditions
    and global solutions, to a local basis, appropriate for discretization integrals and assembly.

  Input Parameters:
+ dm    - The DM
. alpha - The first Euler angle, and in 2D the only one
. beta  - The second Euler angle
- gamma - The third Euler angle

  Note: Following https://en.wikipedia.org/wiki/Euler_angles, we will specify Euler angles by extrinsic rotations, meaning that
  we rotate with respect to a fixed initial coordinate system, the local basis (x-y-z). The global basis (X-Y-Z) is reached as follows:
  $ The XYZ system rotates about the z axis by alpha. The X axis is now at angle alpha with respect to the x axis.
  $ The XYZ system rotates again about the x axis by beta. The Z axis is now at angle beta with respect to the z axis.
  $ The XYZ system rotates a third time about the z axis by gamma.

  Level: developer

.seealso: DMPlexGlobalToLocalBasis(), DMPlexLocalToGlobalBasis()
@*/
PetscErrorCode DMPlexCreateBasisRotation(DM dm, PetscReal alpha, PetscReal beta, PetscReal gamma)
{
  RotCtx        *rc;
  PetscInt       cdim;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(PetscMalloc1(1, &rc));
  dm->transformCtx       = rc;
  dm->transformSetUp     = DMPlexBasisTransformSetUp_Rotation_Internal;
  dm->transformDestroy   = DMPlexBasisTransformDestroy_Rotation_Internal;
  dm->transformGetMatrix = DMPlexBasisTransformGetMatrix_Rotation_Internal;
  rc->dim   = cdim;
  rc->alpha = alpha;
  rc->beta  = beta;
  rc->gamma = gamma;
  CHKERRQ((*dm->transformSetUp)(dm, dm->transformCtx));
  CHKERRQ(DMConstructBasisTransform_Internal(dm));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexInsertBoundaryValuesEssential - Insert boundary values into a local vector using a function of the coordinates

  Input Parameters:
+ dm     - The DM, with a PetscDS that matches the problem being constrained
. time   - The time
. field  - The field to constrain
. Nc     - The number of constrained field components, or 0 for all components
. comps  - An array of constrained component numbers, or NULL for all components
. label  - The DMLabel defining constrained points
. numids - The number of DMLabel ids for constrained points
. ids    - An array of ids for constrained points
. func   - A pointwise function giving boundary values
- ctx    - An optional user context for bcFunc

  Output Parameter:
. locX   - A local vector to receives the boundary values

  Level: developer

.seealso: DMPlexInsertBoundaryValuesEssentialField(), DMPlexInsertBoundaryValuesEssentialBdField(), DMAddBoundary()
@*/
PetscErrorCode DMPlexInsertBoundaryValuesEssential(DM dm, PetscReal time, PetscInt field, PetscInt Nc, const PetscInt comps[], DMLabel label, PetscInt numids, const PetscInt ids[], PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void *ctx, Vec locX)
{
  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void            **ctxs;
  PetscInt          numFields;

  PetscFunctionBegin;
  CHKERRQ(DMGetNumFields(dm, &numFields));
  CHKERRQ(PetscCalloc2(numFields,&funcs,numFields,&ctxs));
  funcs[field] = func;
  ctxs[field]  = ctx;
  CHKERRQ(DMProjectFunctionLabelLocal(dm, time, label, numids, ids, Nc, comps, funcs, ctxs, INSERT_BC_VALUES, locX));
  CHKERRQ(PetscFree2(funcs,ctxs));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexInsertBoundaryValuesEssentialField - Insert boundary values into a local vector using a function of the coordinates and field data

  Input Parameters:
+ dm     - The DM, with a PetscDS that matches the problem being constrained
. time   - The time
. locU   - A local vector with the input solution values
. field  - The field to constrain
. Nc     - The number of constrained field components, or 0 for all components
. comps  - An array of constrained component numbers, or NULL for all components
. label  - The DMLabel defining constrained points
. numids - The number of DMLabel ids for constrained points
. ids    - An array of ids for constrained points
. func   - A pointwise function giving boundary values
- ctx    - An optional user context for bcFunc

  Output Parameter:
. locX   - A local vector to receives the boundary values

  Level: developer

.seealso: DMPlexInsertBoundaryValuesEssential(), DMPlexInsertBoundaryValuesEssentialBdField(), DMAddBoundary()
@*/
PetscErrorCode DMPlexInsertBoundaryValuesEssentialField(DM dm, PetscReal time, Vec locU, PetscInt field, PetscInt Nc, const PetscInt comps[], DMLabel label, PetscInt numids, const PetscInt ids[],
                                                        void (*func)(PetscInt, PetscInt, PetscInt,
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                     PetscReal, const PetscReal[], PetscInt, const PetscScalar[],
                                                                     PetscScalar[]),
                                                        void *ctx, Vec locX)
{
  void (**funcs)(PetscInt, PetscInt, PetscInt,
                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                 PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  void            **ctxs;
  PetscInt          numFields;

  PetscFunctionBegin;
  CHKERRQ(DMGetNumFields(dm, &numFields));
  CHKERRQ(PetscCalloc2(numFields,&funcs,numFields,&ctxs));
  funcs[field] = func;
  ctxs[field]  = ctx;
  CHKERRQ(DMProjectFieldLabelLocal(dm, time, label, numids, ids, Nc, comps, locU, funcs, INSERT_BC_VALUES, locX));
  CHKERRQ(PetscFree2(funcs,ctxs));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexInsertBoundaryValuesEssentialBdField - Insert boundary values into a local vector using a function of the coodinates and boundary field data

  Collective on dm

  Input Parameters:
+ dm     - The DM, with a PetscDS that matches the problem being constrained
. time   - The time
. locU   - A local vector with the input solution values
. field  - The field to constrain
. Nc     - The number of constrained field components, or 0 for all components
. comps  - An array of constrained component numbers, or NULL for all components
. label  - The DMLabel defining constrained points
. numids - The number of DMLabel ids for constrained points
. ids    - An array of ids for constrained points
. func   - A pointwise function giving boundary values, the calling sequence is given in DMProjectBdFieldLabelLocal()
- ctx    - An optional user context for bcFunc

  Output Parameter:
. locX   - A local vector to receive the boundary values

  Level: developer

.seealso: DMProjectBdFieldLabelLocal(), DMPlexInsertBoundaryValuesEssential(), DMPlexInsertBoundaryValuesEssentialField(), DMAddBoundary()
@*/
PetscErrorCode DMPlexInsertBoundaryValuesEssentialBdField(DM dm, PetscReal time, Vec locU, PetscInt field, PetscInt Nc, const PetscInt comps[], DMLabel label, PetscInt numids, const PetscInt ids[],
                                                          void (*func)(PetscInt, PetscInt, PetscInt,
                                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                       PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[],
                                                                       PetscScalar[]),
                                                          void *ctx, Vec locX)
{
  void (**funcs)(PetscInt, PetscInt, PetscInt,
                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                 const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                 PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);
  void            **ctxs;
  PetscInt          numFields;

  PetscFunctionBegin;
  CHKERRQ(DMGetNumFields(dm, &numFields));
  CHKERRQ(PetscCalloc2(numFields,&funcs,numFields,&ctxs));
  funcs[field] = func;
  ctxs[field]  = ctx;
  CHKERRQ(DMProjectBdFieldLabelLocal(dm, time, label, numids, ids, Nc, comps, locU, funcs, INSERT_BC_VALUES, locX));
  CHKERRQ(PetscFree2(funcs,ctxs));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexInsertBoundaryValuesRiemann - Insert boundary values into a local vector

  Input Parameters:
+ dm     - The DM, with a PetscDS that matches the problem being constrained
. time   - The time
. faceGeometry - A vector with the FVM face geometry information
. cellGeometry - A vector with the FVM cell geometry information
. Grad         - A vector with the FVM cell gradient information
. field  - The field to constrain
. Nc     - The number of constrained field components, or 0 for all components
. comps  - An array of constrained component numbers, or NULL for all components
. label  - The DMLabel defining constrained points
. numids - The number of DMLabel ids for constrained points
. ids    - An array of ids for constrained points
. func   - A pointwise function giving boundary values
- ctx    - An optional user context for bcFunc

  Output Parameter:
. locX   - A local vector to receives the boundary values

  Note: This implementation currently ignores the numcomps/comps argument from DMAddBoundary()

  Level: developer

.seealso: DMPlexInsertBoundaryValuesEssential(), DMPlexInsertBoundaryValuesEssentialField(), DMAddBoundary()
@*/
PetscErrorCode DMPlexInsertBoundaryValuesRiemann(DM dm, PetscReal time, Vec faceGeometry, Vec cellGeometry, Vec Grad, PetscInt field, PetscInt Nc, const PetscInt comps[], DMLabel label, PetscInt numids, const PetscInt ids[],
                                                 PetscErrorCode (*func)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*), void *ctx, Vec locX)
{
  PetscDS            prob;
  PetscSF            sf;
  DM                 dmFace, dmCell, dmGrad;
  const PetscScalar *facegeom, *cellgeom = NULL, *grad;
  const PetscInt    *leaves;
  PetscScalar       *x, *fx;
  PetscInt           dim, nleaves, loc, fStart, fEnd, pdim, i;
  PetscErrorCode     ierru = 0;

  PetscFunctionBegin;
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL));
  nleaves = PetscMax(0, nleaves);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(VecGetDM(faceGeometry, &dmFace));
  CHKERRQ(VecGetArrayRead(faceGeometry, &facegeom));
  if (cellGeometry) {
    CHKERRQ(VecGetDM(cellGeometry, &dmCell));
    CHKERRQ(VecGetArrayRead(cellGeometry, &cellgeom));
  }
  if (Grad) {
    PetscFV fv;

    CHKERRQ(PetscDSGetDiscretization(prob, field, (PetscObject *) &fv));
    CHKERRQ(VecGetDM(Grad, &dmGrad));
    CHKERRQ(VecGetArrayRead(Grad, &grad));
    CHKERRQ(PetscFVGetNumComponents(fv, &pdim));
    CHKERRQ(DMGetWorkArray(dm, pdim, MPIU_SCALAR, &fx));
  }
  CHKERRQ(VecGetArray(locX, &x));
  for (i = 0; i < numids; ++i) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    CHKERRQ(DMLabelGetStratumIS(label, ids[i], &faceIS));
    if (!faceIS) continue; /* No points with that id on this process */
    CHKERRQ(ISGetLocalSize(faceIS, &numFaces));
    CHKERRQ(ISGetIndices(faceIS, &faces));
    for (f = 0; f < numFaces; ++f) {
      const PetscInt         face = faces[f], *cells;
      PetscFVFaceGeom        *fg;

      if ((face < fStart) || (face >= fEnd)) continue; /* Refinement adds non-faces to labels */
      CHKERRQ(PetscFindInt(face, nleaves, (PetscInt *) leaves, &loc));
      if (loc >= 0) continue;
      CHKERRQ(DMPlexPointLocalRead(dmFace, face, facegeom, &fg));
      CHKERRQ(DMPlexGetSupport(dm, face, &cells));
      if (Grad) {
        PetscFVCellGeom       *cg;
        PetscScalar           *cx, *cgrad;
        PetscScalar           *xG;
        PetscReal              dx[3];
        PetscInt               d;

        CHKERRQ(DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cg));
        CHKERRQ(DMPlexPointLocalRead(dm, cells[0], x, &cx));
        CHKERRQ(DMPlexPointLocalRead(dmGrad, cells[0], grad, &cgrad));
        CHKERRQ(DMPlexPointLocalFieldRef(dm, cells[1], field, x, &xG));
        DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, fg->centroid, dx);
        for (d = 0; d < pdim; ++d) fx[d] = cx[d] + DMPlex_DotD_Internal(dim, &cgrad[d*dim], dx);
        CHKERRQ((*func)(time, fg->centroid, fg->normal, fx, xG, ctx));
      } else {
        PetscScalar *xI;
        PetscScalar *xG;

        CHKERRQ(DMPlexPointLocalRead(dm, cells[0], x, &xI));
        CHKERRQ(DMPlexPointLocalFieldRef(dm, cells[1], field, x, &xG));
        ierru = (*func)(time, fg->centroid, fg->normal, xI, xG, ctx);
        if (ierru) {
          CHKERRQ(ISRestoreIndices(faceIS, &faces));
          CHKERRQ(ISDestroy(&faceIS));
          goto cleanup;
        }
      }
    }
    CHKERRQ(ISRestoreIndices(faceIS, &faces));
    CHKERRQ(ISDestroy(&faceIS));
  }
  cleanup:
  CHKERRQ(VecRestoreArray(locX, &x));
  if (Grad) {
    CHKERRQ(DMRestoreWorkArray(dm, pdim, MPIU_SCALAR, &fx));
    CHKERRQ(VecRestoreArrayRead(Grad, &grad));
  }
  if (cellGeometry) CHKERRQ(VecRestoreArrayRead(cellGeometry, &cellgeom));
  CHKERRQ(VecRestoreArrayRead(faceGeometry, &facegeom));
  CHKERRQ(ierru);
  PetscFunctionReturn(0);
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 0.0;
  return 0;
}

PetscErrorCode DMPlexInsertBoundaryValues_Plex(DM dm, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscObject    isZero;
  PetscDS        prob;
  PetscInt       numBd, b;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSGetNumBoundary(prob, &numBd));
  CHKERRQ(PetscObjectQuery((PetscObject) locX, "__Vec_bc_zero__", &isZero));
  for (b = 0; b < numBd; ++b) {
    PetscWeakForm           wf;
    DMBoundaryConditionType type;
    const char             *name;
    DMLabel                 label;
    PetscInt                field, Nc;
    const PetscInt         *comps;
    PetscObject             obj;
    PetscClassId            id;
    void                  (*bvfunc)(void);
    PetscInt                numids;
    const PetscInt         *ids;
    void                   *ctx;

    CHKERRQ(PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, &bvfunc, NULL, &ctx));
    if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
    CHKERRQ(DMGetField(dm, field, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      switch (type) {
        /* for FEM, there is no insertion to be done for non-essential boundary conditions */
      case DM_BC_ESSENTIAL:
        {
          PetscSimplePointFunc func = (PetscSimplePointFunc) bvfunc;

          if (isZero) func = zero;
          CHKERRQ(DMPlexLabelAddCells(dm,label));
          CHKERRQ(DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func, ctx, locX));
          CHKERRQ(DMPlexLabelClearCells(dm,label));
        }
        break;
      case DM_BC_ESSENTIAL_FIELD:
        {
          PetscPointFunc func = (PetscPointFunc) bvfunc;

          CHKERRQ(DMPlexLabelAddCells(dm,label));
          CHKERRQ(DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func, ctx, locX));
          CHKERRQ(DMPlexLabelClearCells(dm,label));
        }
        break;
      default: break;
      }
    } else if (id == PETSCFV_CLASSID) {
      {
        PetscErrorCode (*func)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*) = (PetscErrorCode (*)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*)) bvfunc;

        if (!faceGeomFVM) continue;
        CHKERRQ(DMPlexInsertBoundaryValuesRiemann(dm, time, faceGeomFVM, cellGeomFVM, gradFVM, field, Nc, comps, label, numids, ids, func, ctx, locX));
      }
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexInsertTimeDerivativeBoundaryValues_Plex(DM dm, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscObject    isZero;
  PetscDS        prob;
  PetscInt       numBd, b;

  PetscFunctionBegin;
  if (!locX) PetscFunctionReturn(0);
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSGetNumBoundary(prob, &numBd));
  CHKERRQ(PetscObjectQuery((PetscObject) locX, "__Vec_bc_zero__", &isZero));
  for (b = 0; b < numBd; ++b) {
    PetscWeakForm           wf;
    DMBoundaryConditionType type;
    const char             *name;
    DMLabel                 label;
    PetscInt                field, Nc;
    const PetscInt         *comps;
    PetscObject             obj;
    PetscClassId            id;
    PetscInt                numids;
    const PetscInt         *ids;
    void                  (*bvfunc)(void);
    void                   *ctx;

    CHKERRQ(PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, NULL, &bvfunc, &ctx));
    if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
    CHKERRQ(DMGetField(dm, field, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      switch (type) {
        /* for FEM, there is no insertion to be done for non-essential boundary conditions */
      case DM_BC_ESSENTIAL:
        {
          PetscSimplePointFunc func_t = (PetscSimplePointFunc) bvfunc;

          if (isZero) func_t = zero;
          CHKERRQ(DMPlexLabelAddCells(dm,label));
          CHKERRQ(DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func_t, ctx, locX));
          CHKERRQ(DMPlexLabelClearCells(dm,label));
        }
        break;
      case DM_BC_ESSENTIAL_FIELD:
        {
          PetscPointFunc func_t = (PetscPointFunc) bvfunc;

          CHKERRQ(DMPlexLabelAddCells(dm,label));
          CHKERRQ(DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func_t, ctx, locX));
          CHKERRQ(DMPlexLabelClearCells(dm,label));
        }
        break;
      default: break;
      }
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexInsertBoundaryValues - Puts coefficients which represent boundary values into the local solution vector

  Not Collective

  Input Parameters:
+ dm - The DM
. insertEssential - Should I insert essential (e.g. Dirichlet) or inessential (e.g. Neumann) boundary conditions
. time - The time
. faceGeomFVM - Face geometry data for FV discretizations
. cellGeomFVM - Cell geometry data for FV discretizations
- gradFVM - Gradient reconstruction data for FV discretizations

  Output Parameters:
. locX - Solution updated with boundary values

  Level: intermediate

.seealso: DMProjectFunctionLabelLocal(), DMAddBoundary()
@*/
PetscErrorCode DMPlexInsertBoundaryValues(DM dm, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 3);
  if (faceGeomFVM) {PetscValidHeaderSpecific(faceGeomFVM, VEC_CLASSID, 5);}
  if (cellGeomFVM) {PetscValidHeaderSpecific(cellGeomFVM, VEC_CLASSID, 6);}
  if (gradFVM)     {PetscValidHeaderSpecific(gradFVM, VEC_CLASSID, 7);}
  CHKERRQ(PetscTryMethod(dm,"DMPlexInsertBoundaryValues_C",(DM,PetscBool,Vec,PetscReal,Vec,Vec,Vec),(dm,insertEssential,locX,time,faceGeomFVM,cellGeomFVM,gradFVM)));
  PetscFunctionReturn(0);
}

/*@
  DMPlexInsertTimeDerivativeBoundaryValues - Puts coefficients which represent boundary values of the time derivative into the local solution vector

  Input Parameters:
+ dm - The DM
. insertEssential - Should I insert essential (e.g. Dirichlet) or inessential (e.g. Neumann) boundary conditions
. time - The time
. faceGeomFVM - Face geometry data for FV discretizations
. cellGeomFVM - Cell geometry data for FV discretizations
- gradFVM - Gradient reconstruction data for FV discretizations

  Output Parameters:
. locX_t - Solution updated with boundary values

  Level: developer

.seealso: DMProjectFunctionLabelLocal()
@*/
PetscErrorCode DMPlexInsertTimeDerivativeBoundaryValues(DM dm, PetscBool insertEssential, Vec locX_t, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (locX_t)      {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 3);}
  if (faceGeomFVM) {PetscValidHeaderSpecific(faceGeomFVM, VEC_CLASSID, 5);}
  if (cellGeomFVM) {PetscValidHeaderSpecific(cellGeomFVM, VEC_CLASSID, 6);}
  if (gradFVM)     {PetscValidHeaderSpecific(gradFVM, VEC_CLASSID, 7);}
  CHKERRQ(PetscTryMethod(dm,"DMPlexInsertTimeDerviativeBoundaryValues_C",(DM,PetscBool,Vec,PetscReal,Vec,Vec,Vec),(dm,insertEssential,locX_t,time,faceGeomFVM,cellGeomFVM,gradFVM)));
  PetscFunctionReturn(0);
}

PetscErrorCode DMComputeL2Diff_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  Vec              localX;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalVector(dm, &localX));
  CHKERRQ(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, localX, time, NULL, NULL, NULL));
  CHKERRQ(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX));
  CHKERRQ(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX));
  CHKERRQ(DMPlexComputeL2DiffLocal(dm, time, funcs, ctxs, localX, diff));
  CHKERRQ(DMRestoreLocalVector(dm, &localX));
  PetscFunctionReturn(0);
}

/*@C
  DMComputeL2DiffLocal - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h.

  Collective on dm

  Input Parameters:
+ dm     - The DM
. time   - The time
. funcs  - The functions to evaluate for each field component
. ctxs   - Optional array of contexts to pass to each function, or NULL.
- localX - The coefficient vector u_h, a local vector

  Output Parameter:
. diff - The diff ||u - u_h||_2

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2FieldDiff(), DMComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeL2DiffLocal(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec localX, PetscReal *diff)
{
  const PetscInt   debug = ((DM_Plex*)dm->data)->printL2;
  DM               tdm;
  Vec              tv;
  PetscSection     section;
  PetscQuadrature  quad;
  PetscFEGeom      fegeom;
  PetscScalar     *funcVal, *interpolant;
  PetscReal       *coords, *gcoords;
  PetscReal        localDiff = 0.0;
  const PetscReal *quadWeights;
  PetscInt         dim, coordDim, numFields, numComponents = 0, qNc, Nq, cellHeight, cStart, cEnd, c, field, fieldOffset;
  PetscBool        transform;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &coordDim));
  fegeom.dimEmbed = coordDim;
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  CHKERRQ(DMGetBasisTransformDM_Internal(dm, &tdm));
  CHKERRQ(DMGetBasisTransformVec_Internal(dm, &tv));
  CHKERRQ(DMHasBasisTransform(dm, &transform));
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    CHKERRQ(DMGetField(dm, field, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      CHKERRQ(PetscFEGetQuadrature(fe, &quad));
      CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      CHKERRQ(PetscFVGetQuadrature(fv, &quad));
      CHKERRQ(PetscFVGetNumComponents(fv, &Nc));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
    numComponents += Nc;
  }
  CHKERRQ(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights));
  PetscCheckFalse((qNc != 1) && (qNc != numComponents),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  CHKERRQ(PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ));
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;
    PetscInt     qc = 0;

    CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
    CHKERRQ(DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x));

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      CHKERRQ(DMGetField(dm, field, NULL, &obj));
      CHKERRQ(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID)      {CHKERRQ(PetscFEGetNumComponents((PetscFE) obj, &Nc));CHKERRQ(PetscFEGetDimension((PetscFE) obj, &Nb));}
      else if (id == PETSCFV_CLASSID) {CHKERRQ(PetscFVGetNumComponents((PetscFV) obj, &Nc));Nb = 1;}
      else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
      if (debug) {
        char title[1024];
        CHKERRQ(PetscSNPrintf(title, 1023, "Solution for Field %D", field));
        CHKERRQ(DMPrintCellVector(c, title, Nb, &x[fieldOffset]));
      }
      for (q = 0; q < Nq; ++q) {
        PetscFEGeom qgeom;

        qgeom.dimEmbed = fegeom.dimEmbed;
        qgeom.J        = &fegeom.J[q*coordDim*coordDim];
        qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
        qgeom.detJ     = &fegeom.detJ[q];
        PetscCheck(fegeom.detJ[q] > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, point %D", (double)fegeom.detJ[q], c, q);
        if (transform) {
          gcoords = &coords[coordDim*Nq];
          CHKERRQ(DMPlexBasisTransformApplyReal_Internal(dm, &coords[coordDim*q], PETSC_TRUE, coordDim, &coords[coordDim*q], gcoords, dm->transformCtx));
        } else {
          gcoords = &coords[coordDim*q];
        }
        CHKERRQ(PetscArrayzero(funcVal,Nc));
        ierr = (*funcs[field])(coordDim, time, gcoords, Nc, funcVal, ctx);
        if (ierr) {
          CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
          CHKERRQ(DMRestoreLocalVector(dm, &localX));
          CHKERRQ(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
          CHKERRQ(ierr);
        }
        if (transform) CHKERRQ(DMPlexBasisTransformApply_Internal(dm, &coords[coordDim*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx));
        if (id == PETSCFE_CLASSID)      CHKERRQ(PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], &qgeom, q, interpolant));
        else if (id == PETSCFV_CLASSID) CHKERRQ(PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant));
        else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
        for (fc = 0; fc < Nc; ++fc) {
          const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
          if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "    elem %" PetscInt_FMT " field %" PetscInt_FMT ",%" PetscInt_FMT " point %g %g %g diff %g (%g, %g)\n", c, field, fc, (double)(coordDim > 0 ? coords[coordDim*q] : 0.), (double)(coordDim > 1 ? coords[coordDim*q+1] : 0.),(double)(coordDim > 2 ? coords[coordDim*q+2] : 0.), (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q]), PetscRealPart(interpolant[fc]), PetscRealPart(funcVal[fc])));
          elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
        }
      }
      fieldOffset += Nb;
      qc += Nc;
    }
    CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
    if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  elem %" PetscInt_FMT " diff %g\n", c, (double)elemDiff));
    localDiff += elemDiff;
  }
  CHKERRQ(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
  CHKERRMPI(MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

PetscErrorCode DMComputeL2GradientDiff_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, const PetscReal n[], PetscReal *diff)
{
  const PetscInt   debug = ((DM_Plex*)dm->data)->printL2;
  DM               tdm;
  PetscSection     section;
  PetscQuadrature  quad;
  Vec              localX, tv;
  PetscScalar     *funcVal, *interpolant;
  const PetscReal *quadWeights;
  PetscFEGeom      fegeom;
  PetscReal       *coords, *gcoords;
  PetscReal        localDiff = 0.0;
  PetscInt         dim, coordDim, qNc = 0, Nq = 0, numFields, numComponents = 0, cStart, cEnd, c, field, fieldOffset;
  PetscBool        transform;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &coordDim));
  fegeom.dimEmbed = coordDim;
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  CHKERRQ(DMGetLocalVector(dm, &localX));
  CHKERRQ(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX));
  CHKERRQ(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX));
  CHKERRQ(DMGetBasisTransformDM_Internal(dm, &tdm));
  CHKERRQ(DMGetBasisTransformVec_Internal(dm, &tv));
  CHKERRQ(DMHasBasisTransform(dm, &transform));
  for (field = 0; field < numFields; ++field) {
    PetscFE  fe;
    PetscInt Nc;

    CHKERRQ(DMGetField(dm, field, NULL, (PetscObject *) &fe));
    CHKERRQ(PetscFEGetQuadrature(fe, &quad));
    CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
    numComponents += Nc;
  }
  CHKERRQ(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights));
  PetscCheckFalse((qNc != 1) && (qNc != numComponents),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  /* CHKERRQ(DMProjectFunctionLocal(dm, fe, funcs, INSERT_BC_VALUES, localX)); */
  CHKERRQ(PetscMalloc6(numComponents,&funcVal,coordDim*Nq,&coords,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ,numComponents*coordDim,&interpolant,Nq,&fegeom.detJ));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;
    PetscInt     qc = 0;

    CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
    CHKERRQ(DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x));

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscFE          fe;
      void * const     ctx = ctxs ? ctxs[field] : NULL;
      PetscInt         Nb, Nc, q, fc;

      CHKERRQ(DMGetField(dm, field, NULL, (PetscObject *) &fe));
      CHKERRQ(PetscFEGetDimension(fe, &Nb));
      CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
      if (debug) {
        char title[1024];
        CHKERRQ(PetscSNPrintf(title, 1023, "Solution for Field %" PetscInt_FMT, field));
        CHKERRQ(DMPrintCellVector(c, title, Nb, &x[fieldOffset]));
      }
      for (q = 0; q < Nq; ++q) {
        PetscFEGeom qgeom;

        qgeom.dimEmbed = fegeom.dimEmbed;
        qgeom.J        = &fegeom.J[q*coordDim*coordDim];
        qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
        qgeom.detJ     = &fegeom.detJ[q];
        PetscCheck(fegeom.detJ[q] > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT ", quadrature points %" PetscInt_FMT, (double)fegeom.detJ[q], c, q);
        if (transform) {
          gcoords = &coords[coordDim*Nq];
          CHKERRQ(DMPlexBasisTransformApplyReal_Internal(dm, &coords[coordDim*q], PETSC_TRUE, coordDim, &coords[coordDim*q], gcoords, dm->transformCtx));
        } else {
          gcoords = &coords[coordDim*q];
        }
        CHKERRQ(PetscArrayzero(funcVal,Nc));
        ierr = (*funcs[field])(coordDim, time, gcoords, n, Nc, funcVal, ctx);
        if (ierr) {
          CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
          CHKERRQ(DMRestoreLocalVector(dm, &localX));
          CHKERRQ(PetscFree6(funcVal,coords,fegeom.J,fegeom.invJ,interpolant,fegeom.detJ));
          CHKERRQ(ierr);
        }
        if (transform) CHKERRQ(DMPlexBasisTransformApply_Internal(dm, &coords[coordDim*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx));
        CHKERRQ(PetscFEInterpolateGradient_Static(fe, 1, &x[fieldOffset], &qgeom, q, interpolant));
        /* Overwrite with the dot product if the normal is given */
        if (n) {
          for (fc = 0; fc < Nc; ++fc) {
            PetscScalar sum = 0.0;
            PetscInt    d;
            for (d = 0; d < dim; ++d) sum += interpolant[fc*dim+d]*n[d];
            interpolant[fc] = sum;
          }
        }
        for (fc = 0; fc < Nc; ++fc) {
          const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
          if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "    elem %D fieldDer %D,%D diff %g\n", c, field, fc, (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q])));
          elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
        }
      }
      fieldOffset += Nb;
      qc          += Nc;
    }
    CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
    if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  elem %D diff %g\n", c, (double)elemDiff));
    localDiff += elemDiff;
  }
  CHKERRQ(PetscFree6(funcVal,coords,fegeom.J,fegeom.invJ,interpolant,fegeom.detJ));
  CHKERRQ(DMRestoreLocalVector(dm, &localX));
  CHKERRMPI(MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

PetscErrorCode DMComputeL2FieldDiff_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  const PetscInt   debug = ((DM_Plex*)dm->data)->printL2;
  DM               tdm;
  DMLabel          depthLabel;
  PetscSection     section;
  Vec              localX, tv;
  PetscReal       *localDiff;
  PetscInt         dim, depth, dE, Nf, f, Nds, s;
  PetscBool        transform;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &dE));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetLocalVector(dm, &localX));
  CHKERRQ(DMGetBasisTransformDM_Internal(dm, &tdm));
  CHKERRQ(DMGetBasisTransformVec_Internal(dm, &tv));
  CHKERRQ(DMHasBasisTransform(dm, &transform));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMLabelGetNumValues(depthLabel, &depth));

  CHKERRQ(VecSet(localX, 0.0));
  CHKERRQ(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX));
  CHKERRQ(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX));
  CHKERRQ(DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX));
  CHKERRQ(DMGetNumDS(dm, &Nds));
  CHKERRQ(PetscCalloc1(Nf, &localDiff));
  for (s = 0; s < Nds; ++s) {
    PetscDS          ds;
    DMLabel          label;
    IS               fieldIS, pointIS;
    const PetscInt  *fields, *points = NULL;
    PetscQuadrature  quad;
    const PetscReal *quadPoints, *quadWeights;
    PetscFEGeom      fegeom;
    PetscReal       *coords, *gcoords;
    PetscScalar     *funcVal, *interpolant;
    PetscBool        isCohesive;
    PetscInt         qNc, Nq, totNc, cStart = 0, cEnd, c, dsNf;

    CHKERRQ(DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds));
    CHKERRQ(ISGetIndices(fieldIS, &fields));
    CHKERRQ(PetscDSIsCohesive(ds, &isCohesive));
    CHKERRQ(PetscDSGetNumFields(ds, &dsNf));
    CHKERRQ(PetscDSGetTotalComponents(ds, &totNc));
    CHKERRQ(PetscDSGetQuadrature(ds, &quad));
    CHKERRQ(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
    PetscCheckFalse((qNc != 1) && (qNc != totNc),PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, totNc);
    CHKERRQ(PetscCalloc6(totNc, &funcVal, totNc, &interpolant, dE*(Nq+1), &coords,Nq, &fegeom.detJ, dE*dE*Nq, &fegeom.J, dE*dE*Nq, &fegeom.invJ));
    if (!label) {
      CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
    } else {
      CHKERRQ(DMLabelGetStratumIS(label, 1, &pointIS));
      CHKERRQ(ISGetLocalSize(pointIS, &cEnd));
      CHKERRQ(ISGetIndices(pointIS, &points));
    }
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt cell  = points ? points[c] : c;
      PetscScalar    *x    = NULL;
      const PetscInt *cone;
      PetscInt        qc   = 0, fOff = 0, dep;

      CHKERRQ(DMLabelGetValue(depthLabel, cell, &dep));
      if (dep != depth-1) continue;
      if (isCohesive) {
        CHKERRQ(DMPlexGetCone(dm, cell, &cone));
        CHKERRQ(DMPlexComputeCellGeometryFEM(dm, cone[0], quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      } else {
        CHKERRQ(DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      }
      CHKERRQ(DMPlexVecGetClosure(dm, NULL, localX, cell, NULL, &x));
      for (f = 0; f < dsNf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        void * const ctx = ctxs ? ctxs[fields[f]] : NULL;
        PetscInt     Nb, Nc, q, fc;
        PetscReal    elemDiff = 0.0;
        PetscBool    cohesive;

        CHKERRQ(PetscDSGetCohesive(ds, f, &cohesive));
        if (isCohesive && !cohesive) continue;
        CHKERRQ(PetscDSGetDiscretization(ds, f, &obj));
        CHKERRQ(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID)      {CHKERRQ(PetscFEGetNumComponents((PetscFE) obj, &Nc));CHKERRQ(PetscFEGetDimension((PetscFE) obj, &Nb));}
        else if (id == PETSCFV_CLASSID) {CHKERRQ(PetscFVGetNumComponents((PetscFV) obj, &Nc));Nb = 1;}
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", fields[f]);
        if (debug) {
          char title[1024];
          CHKERRQ(PetscSNPrintf(title, 1023, "Solution for Field %D", fields[f]));
          CHKERRQ(DMPrintCellVector(cell, title, Nb, &x[fOff]));
        }
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*dE*dE];
          qgeom.invJ     = &fegeom.invJ[q*dE*dE];
          qgeom.detJ     = &fegeom.detJ[q];
          PetscCheck(fegeom.detJ[q] > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %D, quadrature point %D", (double)fegeom.detJ[q], cell, q);
          if (transform) {
            gcoords = &coords[dE*Nq];
            CHKERRQ(DMPlexBasisTransformApplyReal_Internal(dm, &coords[dE*q], PETSC_TRUE, dE, &coords[dE*q], gcoords, dm->transformCtx));
          } else {
            gcoords = &coords[dE*q];
          }
          for (fc = 0; fc < Nc; ++fc) funcVal[fc] = 0.;
          ierr = (*funcs[fields[f]])(dE, time, gcoords, Nc, funcVal, ctx);
          if (ierr) {
            CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, localX, cell, NULL, &x));
            CHKERRQ(DMRestoreLocalVector(dm, &localX));
            CHKERRQ(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
            CHKERRQ(ierr);
          }
          if (transform) CHKERRQ(DMPlexBasisTransformApply_Internal(dm, &coords[dE*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx));
          /* Call once for each face, except for lagrange field */
          if (id == PETSCFE_CLASSID)      CHKERRQ(PetscFEInterpolate_Static((PetscFE) obj, &x[fOff], &qgeom, q, interpolant));
          else if (id == PETSCFV_CLASSID) CHKERRQ(PetscFVInterpolate_Static((PetscFV) obj, &x[fOff], q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", fields[f]);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
            if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "    cell %D field %D,%D point %g %g %g diff %g\n", cell, fields[f], fc, (double)(dE > 0 ? coords[dE*q] : 0.), (double)(dE > 1 ? coords[dE*q+1] : 0.),(double)(dE > 2 ? coords[dE*q+2] : 0.), (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q])));
            elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
          }
        }
        fOff += Nb;
        qc   += Nc;
        localDiff[fields[f]] += elemDiff;
        if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  cell %D field %D cum diff %g\n", cell, fields[f], (double)localDiff[fields[f]]));
      }
      CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, localX, cell, NULL, &x));
    }
    if (label) {
      CHKERRQ(ISRestoreIndices(pointIS, &points));
      CHKERRQ(ISDestroy(&pointIS));
    }
    CHKERRQ(ISRestoreIndices(fieldIS, &fields));
    CHKERRQ(PetscFree6(funcVal, interpolant, coords, fegeom.detJ, fegeom.J, fegeom.invJ));
  }
  CHKERRQ(DMRestoreLocalVector(dm, &localX));
  CHKERRMPI(MPIU_Allreduce(localDiff, diff, Nf, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
  CHKERRQ(PetscFree(localDiff));
  for (f = 0; f < Nf; ++f) diff[f] = PetscSqrtReal(diff[f]);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeL2DiffVec - This function computes the cellwise L_2 difference between a function u and an FEM interpolant solution u_h, and stores it in a Vec.

  Collective on dm

  Input Parameters:
+ dm    - The DM
. time  - The time
. funcs - The functions to evaluate for each field component: NULL means that component does not contribute to error calculation
. ctxs  - Optional array of contexts to pass to each function, or NULL.
- X     - The coefficient vector u_h

  Output Parameter:
. D - A Vec which holds the difference ||u - u_h||_2 for each cell

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2Diff(), DMPlexComputeL2FieldDiff(), DMComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeL2DiffVec(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, Vec D)
{
  PetscSection     section;
  PetscQuadrature  quad;
  Vec              localX;
  PetscFEGeom      fegeom;
  PetscScalar     *funcVal, *interpolant;
  PetscReal       *coords;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, coordDim, numFields, numComponents = 0, qNc, Nq, cStart, cEnd, c, field, fieldOffset;

  PetscFunctionBegin;
  CHKERRQ(VecSet(D, 0.0));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &coordDim));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  CHKERRQ(DMGetLocalVector(dm, &localX));
  CHKERRQ(DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX));
  CHKERRQ(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX));
  CHKERRQ(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX));
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    CHKERRQ(DMGetField(dm, field, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      CHKERRQ(PetscFEGetQuadrature(fe, &quad));
      CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      CHKERRQ(PetscFVGetQuadrature(fv, &quad));
      CHKERRQ(PetscFVGetNumComponents(fv, &Nc));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
    numComponents += Nc;
  }
  CHKERRQ(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCheckFalse((qNc != 1) && (qNc != numComponents),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  CHKERRQ(PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscScalar  elemDiff = 0.0;
    PetscInt     qc = 0;

    CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
    CHKERRQ(DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x));

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      CHKERRQ(DMGetField(dm, field, NULL, &obj));
      CHKERRQ(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID)      {CHKERRQ(PetscFEGetNumComponents((PetscFE) obj, &Nc));CHKERRQ(PetscFEGetDimension((PetscFE) obj, &Nb));}
      else if (id == PETSCFV_CLASSID) {CHKERRQ(PetscFVGetNumComponents((PetscFV) obj, &Nc));Nb = 1;}
      else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
      if (funcs[field]) {
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*coordDim*coordDim];
          qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
          qgeom.detJ     = &fegeom.detJ[q];
          PetscCheck(fegeom.detJ[q] > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, quadrature points %D", (double)fegeom.detJ[q], c, q);
          CHKERRQ((*funcs[field])(coordDim, time, &coords[q*coordDim], Nc, funcVal, ctx));
#if defined(needs_fix_with_return_code_argument)
          if (ierr) {
            PetscErrorCode ierr;
            CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
            CHKERRQ(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
            CHKERRQ(DMRestoreLocalVector(dm, &localX));
          }
#endif
          if (id == PETSCFE_CLASSID)      CHKERRQ(PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], &qgeom, q, interpolant));
          else if (id == PETSCFV_CLASSID) CHKERRQ(PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
            elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
          }
        }
      }
      fieldOffset += Nb;
      qc          += Nc;
    }
    CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
    CHKERRQ(VecSetValue(D, c - cStart, elemDiff, INSERT_VALUES));
  }
  CHKERRQ(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
  CHKERRQ(DMRestoreLocalVector(dm, &localX));
  CHKERRQ(VecSqrtAbs(D));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeClementInterpolant - This function computes the L2 projection of the cellwise values of a function u onto P1, and stores it in a Vec.

  Collective on dm

  Input Parameters:
+ dm - The DM
- locX  - The coefficient vector u_h

  Output Parameter:
. locC - A Vec which holds the Clement interpolant of the function

  Notes:
  u_h(v_i) = \sum_{T_i \in support(v_i)} |T_i| u_h(T_i) / \sum_{T_i \in support(v_i)} |T_i| where |T_i| is the cell volume

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2Diff(), DMPlexComputeL2FieldDiff(), DMComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeClementInterpolant(DM dm, Vec locX, Vec locC)
{
  PetscInt         debug = ((DM_Plex *) dm->data)->printFEM;
  DM               dmc;
  PetscQuadrature  quad;
  PetscScalar     *interpolant, *valsum;
  PetscFEGeom      fegeom;
  PetscReal       *coords;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, cdim, Nf, f, Nc = 0, Nq, qNc, cStart, cEnd, vStart, vEnd, v;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(ClementCitation, &Clementcite));
  CHKERRQ(VecGetDM(locC, &dmc));
  CHKERRQ(VecSet(locC, 0.0));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  fegeom.dimEmbed = cdim;
  CHKERRQ(DMGetNumFields(dm, &Nf));
  PetscCheck(Nf > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fields is zero!");
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     fNc;

    CHKERRQ(DMGetField(dm, f, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      CHKERRQ(PetscFEGetQuadrature(fe, &quad));
      CHKERRQ(PetscFEGetNumComponents(fe, &fNc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      CHKERRQ(PetscFVGetQuadrature(fv, &quad));
      CHKERRQ(PetscFVGetNumComponents(fv, &fNc));
    } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
    Nc += fNc;
  }
  CHKERRQ(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCheck(qNc == 1,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D > 1", qNc);
  CHKERRQ(PetscMalloc6(Nc*2, &valsum, Nc, &interpolant, cdim*Nq, &coords, Nq, &fegeom.detJ, cdim*cdim*Nq, &fegeom.J, cdim*cdim*Nq, &fegeom.invJ));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar volsum = 0.0;
    PetscInt   *star   = NULL;
    PetscInt    starSize, st, fc;

    CHKERRQ(PetscArrayzero(valsum, Nc));
    CHKERRQ(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    for (st = 0; st < starSize*2; st += 2) {
      const PetscInt cell = star[st];
      PetscScalar   *val  = &valsum[Nc];
      PetscScalar   *x    = NULL;
      PetscReal      vol  = 0.0;
      PetscInt       foff = 0;

      if ((cell < cStart) || (cell >= cEnd)) continue;
      CHKERRQ(DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      CHKERRQ(DMPlexVecGetClosure(dm, NULL, locX, cell, NULL, &x));
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscInt     Nb, fNc, q;

        CHKERRQ(PetscArrayzero(val, Nc));
        CHKERRQ(DMGetField(dm, f, NULL, &obj));
        CHKERRQ(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID)      {CHKERRQ(PetscFEGetNumComponents((PetscFE) obj, &fNc));CHKERRQ(PetscFEGetDimension((PetscFE) obj, &Nb));}
        else if (id == PETSCFV_CLASSID) {CHKERRQ(PetscFVGetNumComponents((PetscFV) obj, &fNc));Nb = 1;}
        else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
        for (q = 0; q < Nq; ++q) {
          const PetscReal wt = quadWeights[q]*fegeom.detJ[q];
          PetscFEGeom     qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*cdim*cdim];
          qgeom.invJ     = &fegeom.invJ[q*cdim*cdim];
          qgeom.detJ     = &fegeom.detJ[q];
          PetscCheck(fegeom.detJ[q] > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, quadrature points %D", (double) fegeom.detJ[q], cell, q);
          if (id == PETSCFE_CLASSID) CHKERRQ(PetscFEInterpolate_Static((PetscFE) obj, &x[foff], &qgeom, q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
          for (fc = 0; fc < fNc; ++fc) val[foff+fc] += interpolant[fc]*wt;
          vol += wt;
        }
        foff += Nb;
      }
      CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x));
      for (fc = 0; fc < Nc; ++fc) valsum[fc] += val[fc];
      volsum += vol;
      if (debug) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Vertex %" PetscInt_FMT " Cell %" PetscInt_FMT " value: [", v, cell));
        for (fc = 0; fc < Nc; ++fc) {
          if (fc) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, ", "));
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%g", (double) PetscRealPart(val[fc])));
        }
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "]\n"));
      }
    }
    for (fc = 0; fc < Nc; ++fc) valsum[fc] /= volsum;
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    CHKERRQ(DMPlexVecSetClosure(dmc, NULL, locC, v, valsum, INSERT_VALUES));
  }
  CHKERRQ(PetscFree6(valsum, interpolant, coords, fegeom.detJ, fegeom.J, fegeom.invJ));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeGradientClementInterpolant - This function computes the L2 projection of the cellwise gradient of a function u onto P1, and stores it in a Vec.

  Collective on dm

  Input Parameters:
+ dm - The DM
- locX  - The coefficient vector u_h

  Output Parameter:
. locC - A Vec which holds the Clement interpolant of the gradient

  Notes:
  \nabla u_h(v_i) = \sum_{T_i \in support(v_i)} |T_i| \nabla u_h(T_i) / \sum_{T_i \in support(v_i)} |T_i| where |T_i| is the cell volume

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2Diff(), DMPlexComputeL2FieldDiff(), DMComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeGradientClementInterpolant(DM dm, Vec locX, Vec locC)
{
  DM_Plex         *mesh  = (DM_Plex *) dm->data;
  PetscInt         debug = mesh->printFEM;
  DM               dmC;
  PetscQuadrature  quad;
  PetscScalar     *interpolant, *gradsum;
  PetscFEGeom      fegeom;
  PetscReal       *coords;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, coordDim, numFields, numComponents = 0, qNc, Nq, cStart, cEnd, vStart, vEnd, v, field, fieldOffset;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(ClementCitation, &Clementcite));
  CHKERRQ(VecGetDM(locC, &dmC));
  CHKERRQ(VecSet(locC, 0.0));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &coordDim));
  fegeom.dimEmbed = coordDim;
  CHKERRQ(DMGetNumFields(dm, &numFields));
  PetscCheck(numFields,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fields is zero!");
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    CHKERRQ(DMGetField(dm, field, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      CHKERRQ(PetscFEGetQuadrature(fe, &quad));
      CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      CHKERRQ(PetscFVGetQuadrature(fv, &quad));
      CHKERRQ(PetscFVGetNumComponents(fv, &Nc));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
    numComponents += Nc;
  }
  CHKERRQ(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCheckFalse((qNc != 1) && (qNc != numComponents),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  CHKERRQ(PetscMalloc6(coordDim*numComponents*2,&gradsum,coordDim*numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar volsum = 0.0;
    PetscInt   *star   = NULL;
    PetscInt    starSize, st, d, fc;

    CHKERRQ(PetscArrayzero(gradsum, coordDim*numComponents));
    CHKERRQ(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    for (st = 0; st < starSize*2; st += 2) {
      const PetscInt cell = star[st];
      PetscScalar   *grad = &gradsum[coordDim*numComponents];
      PetscScalar   *x    = NULL;
      PetscReal      vol  = 0.0;

      if ((cell < cStart) || (cell >= cEnd)) continue;
      CHKERRQ(DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      CHKERRQ(DMPlexVecGetClosure(dm, NULL, locX, cell, NULL, &x));
      for (field = 0, fieldOffset = 0; field < numFields; ++field) {
        PetscObject  obj;
        PetscClassId id;
        PetscInt     Nb, Nc, q, qc = 0;

        CHKERRQ(PetscArrayzero(grad, coordDim*numComponents));
        CHKERRQ(DMGetField(dm, field, NULL, &obj));
        CHKERRQ(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID)      {CHKERRQ(PetscFEGetNumComponents((PetscFE) obj, &Nc));CHKERRQ(PetscFEGetDimension((PetscFE) obj, &Nb));}
        else if (id == PETSCFV_CLASSID) {CHKERRQ(PetscFVGetNumComponents((PetscFV) obj, &Nc));Nb = 1;}
        else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*coordDim*coordDim];
          qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
          qgeom.detJ     = &fegeom.detJ[q];
          PetscCheck(fegeom.detJ[q] > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, quadrature points %D", (double)fegeom.detJ[q], cell, q);
          if (id == PETSCFE_CLASSID)      CHKERRQ(PetscFEInterpolateGradient_Static((PetscFE) obj, 1, &x[fieldOffset], &qgeom, q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+qc];

            for (d = 0; d < coordDim; ++d) grad[fc*coordDim+d] += interpolant[fc*dim+d]*wt*fegeom.detJ[q];
          }
          vol += quadWeights[q*qNc]*fegeom.detJ[q];
        }
        fieldOffset += Nb;
        qc          += Nc;
      }
      CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x));
      for (fc = 0; fc < numComponents; ++fc) {
        for (d = 0; d < coordDim; ++d) {
          gradsum[fc*coordDim+d] += grad[fc*coordDim+d];
        }
      }
      volsum += vol;
      if (debug) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Vertex %" PetscInt_FMT " Cell %" PetscInt_FMT " gradient: [", v, cell));
        for (fc = 0; fc < numComponents; ++fc) {
          for (d = 0; d < coordDim; ++d) {
            if (fc || d > 0) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, ", "));
            CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%g", (double)PetscRealPart(grad[fc*coordDim+d])));
          }
        }
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "]\n"));
      }
    }
    for (fc = 0; fc < numComponents; ++fc) {
      for (d = 0; d < coordDim; ++d) gradsum[fc*coordDim+d] /= volsum;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    CHKERRQ(DMPlexVecSetClosure(dmC, NULL, locC, v, gradsum, INSERT_VALUES));
  }
  CHKERRQ(PetscFree6(gradsum,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeIntegral_Internal(DM dm, Vec X, PetscInt cStart, PetscInt cEnd, PetscScalar *cintegral, void *user)
{
  DM                 dmAux = NULL;
  PetscDS            prob,    probAux = NULL;
  PetscSection       section, sectionAux;
  Vec                locX,    locA;
  PetscInt           dim, numCells = cEnd - cStart, c, f;
  PetscBool          useFVM = PETSC_FALSE;
  /* DS */
  PetscInt           Nf,    totDim,    *uOff, *uOff_x, numConstants;
  PetscInt           NfAux, totDimAux, *aOff;
  PetscScalar       *u, *a;
  const PetscScalar *constants;
  /* Geometry */
  PetscFEGeom       *cgeomFEM;
  DM                 dmGrad;
  PetscQuadrature    affineQuad = NULL;
  Vec                cellGeometryFVM = NULL, faceGeometryFVM = NULL, locGrad = NULL;
  PetscFVCellGeom   *cgeomFVM;
  const PetscScalar *lgrad;
  PetscInt           maxDegree;
  DMField            coordField;
  IS                 cellIS;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  /* Determine which discretizations we have */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFV_CLASSID) useFVM = PETSC_TRUE;
  }
  /* Get local solution with boundary values */
  CHKERRQ(DMGetLocalVector(dm, &locX));
  CHKERRQ(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, 0.0, NULL, NULL, NULL));
  CHKERRQ(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
  CHKERRQ(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
  /* Read DS information */
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(PetscDSGetComponentOffsets(prob, &uOff));
  CHKERRQ(PetscDSGetComponentDerivativeOffsets(prob, &uOff_x));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,numCells,cStart,1,&cellIS));
  CHKERRQ(PetscDSGetConstants(prob, &numConstants, &constants));
  /* Read Auxiliary DS information */
  CHKERRQ(DMGetAuxiliaryVec(dm, NULL, 0, 0, &locA));
  if (locA) {
    CHKERRQ(VecGetDM(locA, &dmAux));
    CHKERRQ(DMGetDS(dmAux, &probAux));
    CHKERRQ(PetscDSGetNumFields(probAux, &NfAux));
    CHKERRQ(DMGetLocalSection(dmAux, &sectionAux));
    CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
    CHKERRQ(PetscDSGetComponentOffsets(probAux, &aOff));
  }
  /* Allocate data  arrays */
  CHKERRQ(PetscCalloc1(numCells*totDim, &u));
  if (dmAux) CHKERRQ(PetscMalloc1(numCells*totDimAux, &a));
  /* Read out geometry */
  CHKERRQ(DMGetCoordinateField(dm,&coordField));
  CHKERRQ(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
  if (maxDegree <= 1) {
    CHKERRQ(DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad));
    if (affineQuad) {
      CHKERRQ(DMFieldCreateFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&cgeomFEM));
    }
  }
  if (useFVM) {
    PetscFV   fv = NULL;
    Vec       grad;
    PetscInt  fStart, fEnd;
    PetscBool compGrad;

    for (f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;

      CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
      CHKERRQ(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFV_CLASSID) {fv = (PetscFV) obj; break;}
    }
    CHKERRQ(PetscFVGetComputeGradients(fv, &compGrad));
    CHKERRQ(PetscFVSetComputeGradients(fv, PETSC_TRUE));
    CHKERRQ(DMPlexComputeGeometryFVM(dm, &cellGeometryFVM, &faceGeometryFVM));
    CHKERRQ(DMPlexComputeGradientFVM(dm, fv, faceGeometryFVM, cellGeometryFVM, &dmGrad));
    CHKERRQ(PetscFVSetComputeGradients(fv, compGrad));
    CHKERRQ(VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM));
    /* Reconstruct and limit cell gradients */
    CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    CHKERRQ(DMGetGlobalVector(dmGrad, &grad));
    CHKERRQ(DMPlexReconstructGradients_Internal(dm, fv, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad));
    /* Communicate gradient values */
    CHKERRQ(DMGetLocalVector(dmGrad, &locGrad));
    CHKERRQ(DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad));
    CHKERRQ(DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad));
    CHKERRQ(DMRestoreGlobalVector(dmGrad, &grad));
    /* Handle non-essential (e.g. outflow) boundary values */
    CHKERRQ(DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, 0.0, faceGeometryFVM, cellGeometryFVM, locGrad));
    CHKERRQ(VecGetArrayRead(locGrad, &lgrad));
  }
  /* Read out data from inputs */
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscInt     i;

    CHKERRQ(DMPlexVecGetClosure(dm, section, locX, c, NULL, &x));
    for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
    CHKERRQ(DMPlexVecRestoreClosure(dm, section, locX, c, NULL, &x));
    if (dmAux) {
      CHKERRQ(DMPlexVecGetClosure(dmAux, sectionAux, locA, c, NULL, &x));
      for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
      CHKERRQ(DMPlexVecRestoreClosure(dmAux, sectionAux, locA, c, NULL, &x));
    }
  }
  /* Do integration for each field */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

    CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE         fe = (PetscFE) obj;
      PetscQuadrature q;
      PetscFEGeom     *chunkGeom = NULL;
      PetscInt        Nq, Nb;

      CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
      CHKERRQ(PetscFEGetQuadrature(fe, &q));
      CHKERRQ(PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL));
      CHKERRQ(PetscFEGetDimension(fe, &Nb));
      blockSize = Nb*Nq;
      batchSize = numBlocks * blockSize;
      CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      if (!affineQuad) {
        CHKERRQ(DMFieldCreateFEGeom(coordField,cellIS,q,PETSC_FALSE,&cgeomFEM));
      }
      CHKERRQ(PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom));
      CHKERRQ(PetscFEIntegrate(prob, f, Ne, chunkGeom, u, probAux, a, cintegral));
      CHKERRQ(PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&chunkGeom));
      CHKERRQ(PetscFEIntegrate(prob, f, Nr, chunkGeom, &u[offset*totDim], probAux, &a[offset*totDimAux], &cintegral[offset*Nf]));
      CHKERRQ(PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&chunkGeom));
      if (!affineQuad) {
        CHKERRQ(PetscFEGeomDestroy(&cgeomFEM));
      }
    } else if (id == PETSCFV_CLASSID) {
      PetscInt       foff;
      PetscPointFunc obj_func;
      PetscScalar    lint;

      CHKERRQ(PetscDSGetObjective(prob, f, &obj_func));
      CHKERRQ(PetscDSGetFieldOffset(prob, f, &foff));
      if (obj_func) {
        for (c = 0; c < numCells; ++c) {
          PetscScalar *u_x;

          CHKERRQ(DMPlexPointLocalRead(dmGrad, c, lgrad, &u_x));
          obj_func(dim, Nf, NfAux, uOff, uOff_x, &u[totDim*c+foff], NULL, u_x, aOff, NULL, &a[totDimAux*c], NULL, NULL, 0.0, cgeomFVM[c].centroid, numConstants, constants, &lint);
          cintegral[c*Nf+f] += PetscRealPart(lint)*cgeomFVM[c].volume;
        }
      }
    } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
  }
  /* Cleanup data arrays */
  if (useFVM) {
    CHKERRQ(VecRestoreArrayRead(locGrad, &lgrad));
    CHKERRQ(VecRestoreArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM));
    CHKERRQ(DMRestoreLocalVector(dmGrad, &locGrad));
    CHKERRQ(VecDestroy(&faceGeometryFVM));
    CHKERRQ(VecDestroy(&cellGeometryFVM));
    CHKERRQ(DMDestroy(&dmGrad));
  }
  if (dmAux) CHKERRQ(PetscFree(a));
  CHKERRQ(PetscFree(u));
  /* Cleanup */
  if (affineQuad) {
    CHKERRQ(PetscFEGeomDestroy(&cgeomFEM));
  }
  CHKERRQ(PetscQuadratureDestroy(&affineQuad));
  CHKERRQ(ISDestroy(&cellIS));
  CHKERRQ(DMRestoreLocalVector(dm, &locX));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeIntegralFEM - Form the integral over the domain from the global input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. X  - Global input vector
- user - The user context

  Output Parameter:
. integral - Integral for each field

  Level: developer

.seealso: DMPlexSNESComputeResidualFEM()
@*/
PetscErrorCode DMPlexComputeIntegralFEM(DM dm, Vec X, PetscScalar *integral, void *user)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  PetscScalar   *cintegral, *lintegral;
  PetscInt       Nf, f, cellHeight, cStart, cEnd, cell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidPointer(integral, 3);
  CHKERRQ(PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd));
  /* TODO Introduce a loop over large chunks (right now this is a single chunk) */
  CHKERRQ(PetscCalloc2(Nf, &lintegral, (cEnd-cStart)*Nf, &cintegral));
  CHKERRQ(DMPlexComputeIntegral_Internal(dm, X, cStart, cEnd, cintegral, user));
  /* Sum up values */
  for (cell = cStart; cell < cEnd; ++cell) {
    const PetscInt c = cell - cStart;

    if (mesh->printFEM > 1) CHKERRQ(DMPrintCellVector(cell, "Cell Integral", Nf, &cintegral[c*Nf]));
    for (f = 0; f < Nf; ++f) lintegral[f] += cintegral[c*Nf+f];
  }
  CHKERRMPI(MPIU_Allreduce(lintegral, integral, Nf, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject) dm)));
  if (mesh->printFEM) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "Integral:"));
    for (f = 0; f < Nf; ++f) CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), " %g", (double) PetscRealPart(integral[f])));
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "\n"));
  }
  CHKERRQ(PetscFree2(lintegral, cintegral));
  CHKERRQ(PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeCellwiseIntegralFEM - Form the vector of cellwise integrals F from the global input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. X  - Global input vector
- user - The user context

  Output Parameter:
. integral - Cellwise integrals for each field

  Level: developer

.seealso: DMPlexSNESComputeResidualFEM()
@*/
PetscErrorCode DMPlexComputeCellwiseIntegralFEM(DM dm, Vec X, Vec F, void *user)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DM             dmF;
  PetscSection   sectionF;
  PetscScalar   *cintegral, *af;
  PetscInt       Nf, f, cellHeight, cStart, cEnd, cell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  CHKERRQ(PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd));
  /* TODO Introduce a loop over large chunks (right now this is a single chunk) */
  CHKERRQ(PetscCalloc1((cEnd-cStart)*Nf, &cintegral));
  CHKERRQ(DMPlexComputeIntegral_Internal(dm, X, cStart, cEnd, cintegral, user));
  /* Put values in F*/
  CHKERRQ(VecGetDM(F, &dmF));
  CHKERRQ(DMGetLocalSection(dmF, &sectionF));
  CHKERRQ(VecGetArray(F, &af));
  for (cell = cStart; cell < cEnd; ++cell) {
    const PetscInt c = cell - cStart;
    PetscInt       dof, off;

    if (mesh->printFEM > 1) CHKERRQ(DMPrintCellVector(cell, "Cell Integral", Nf, &cintegral[c*Nf]));
    CHKERRQ(PetscSectionGetDof(sectionF, cell, &dof));
    CHKERRQ(PetscSectionGetOffset(sectionF, cell, &off));
    PetscCheck(dof == Nf,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of cell dofs %D != %D", dof, Nf);
    for (f = 0; f < Nf; ++f) af[off+f] = cintegral[c*Nf+f];
  }
  CHKERRQ(VecRestoreArray(F, &af));
  CHKERRQ(PetscFree(cintegral));
  CHKERRQ(PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeBdIntegral_Internal(DM dm, Vec locX, IS pointIS,
                                                       void (*func)(PetscInt, PetscInt, PetscInt,
                                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                    PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                       PetscScalar *fintegral, void *user)
{
  DM                 plex = NULL, plexA = NULL;
  DMEnclosureType    encAux;
  PetscDS            prob, probAux = NULL;
  PetscSection       section, sectionAux = NULL;
  Vec                locA = NULL;
  DMField            coordField;
  PetscInt           Nf,        totDim,        *uOff, *uOff_x;
  PetscInt           NfAux = 0, totDimAux = 0, *aOff = NULL;
  PetscScalar       *u, *a = NULL;
  const PetscScalar *constants;
  PetscInt           numConstants, f;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscSectionGetNumFields(section, &Nf));
  /* Determine which discretizations we have */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    PetscCheck(id != PETSCFV_CLASSID,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Not supported for FVM (field %D)", f);
  }
  /* Read DS information */
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(PetscDSGetComponentOffsets(prob, &uOff));
  CHKERRQ(PetscDSGetComponentDerivativeOffsets(prob, &uOff_x));
  CHKERRQ(PetscDSGetConstants(prob, &numConstants, &constants));
  /* Read Auxiliary DS information */
  CHKERRQ(DMGetAuxiliaryVec(dm, NULL, 0, 0, &locA));
  if (locA) {
    DM dmAux;

    CHKERRQ(VecGetDM(locA, &dmAux));
    CHKERRQ(DMGetEnclosureRelation(dmAux, dm, &encAux));
    CHKERRQ(DMConvert(dmAux, DMPLEX, &plexA));
    CHKERRQ(DMGetDS(dmAux, &probAux));
    CHKERRQ(PetscDSGetNumFields(probAux, &NfAux));
    CHKERRQ(DMGetLocalSection(dmAux, &sectionAux));
    CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
    CHKERRQ(PetscDSGetComponentOffsets(probAux, &aOff));
  }
  /* Integrate over points */
  {
    PetscFEGeom    *fgeom, *chunkGeom = NULL;
    PetscInt        maxDegree;
    PetscQuadrature qGeom = NULL;
    const PetscInt *points;
    PetscInt        numFaces, face, Nq, field;
    PetscInt        numChunks, chunkSize, chunk, Nr, offset;

    CHKERRQ(ISGetLocalSize(pointIS, &numFaces));
    CHKERRQ(ISGetIndices(pointIS, &points));
    CHKERRQ(PetscCalloc2(numFaces*totDim, &u, locA ? numFaces*totDimAux : 0, &a));
    CHKERRQ(DMFieldGetDegree(coordField, pointIS, NULL, &maxDegree));
    for (field = 0; field < Nf; ++field) {
      PetscFE fe;

      CHKERRQ(PetscDSGetDiscretization(prob, field, (PetscObject *) &fe));
      if (maxDegree <= 1) CHKERRQ(DMFieldCreateDefaultQuadrature(coordField, pointIS, &qGeom));
      if (!qGeom) {
        CHKERRQ(PetscFEGetFaceQuadrature(fe, &qGeom));
        CHKERRQ(PetscObjectReference((PetscObject) qGeom));
      }
      CHKERRQ(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
      CHKERRQ(DMPlexGetFEGeom(coordField, pointIS, qGeom, PETSC_TRUE, &fgeom));
      for (face = 0; face < numFaces; ++face) {
        const PetscInt point = points[face], *support;
        PetscScalar    *x    = NULL;
        PetscInt       i;

        CHKERRQ(DMPlexGetSupport(dm, point, &support));
        CHKERRQ(DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x));
        for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
        CHKERRQ(DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x));
        if (locA) {
          PetscInt subp;
          CHKERRQ(DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp));
          CHKERRQ(DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x));
          for (i = 0; i < totDimAux; ++i) a[f*totDimAux+i] = x[i];
          CHKERRQ(DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x));
        }
      }
      /* Get blocking */
      {
        PetscQuadrature q;
        PetscInt        numBatches, batchSize, numBlocks, blockSize;
        PetscInt        Nq, Nb;

        CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
        CHKERRQ(PetscFEGetQuadrature(fe, &q));
        CHKERRQ(PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL));
        CHKERRQ(PetscFEGetDimension(fe, &Nb));
        blockSize = Nb*Nq;
        batchSize = numBlocks * blockSize;
        chunkSize = numBatches*batchSize;
        CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
        numChunks = numFaces / chunkSize;
        Nr        = numFaces % chunkSize;
        offset    = numFaces - Nr;
      }
      /* Do integration for each field */
      for (chunk = 0; chunk < numChunks; ++chunk) {
        CHKERRQ(PetscFEGeomGetChunk(fgeom, chunk*chunkSize, (chunk+1)*chunkSize, &chunkGeom));
        CHKERRQ(PetscFEIntegrateBd(prob, field, func, chunkSize, chunkGeom, u, probAux, a, fintegral));
        CHKERRQ(PetscFEGeomRestoreChunk(fgeom, 0, offset, &chunkGeom));
      }
      CHKERRQ(PetscFEGeomGetChunk(fgeom, offset, numFaces, &chunkGeom));
      CHKERRQ(PetscFEIntegrateBd(prob, field, func, Nr, chunkGeom, &u[offset*totDim], probAux, a ? &a[offset*totDimAux] : NULL, &fintegral[offset*Nf]));
      CHKERRQ(PetscFEGeomRestoreChunk(fgeom, offset, numFaces, &chunkGeom));
      /* Cleanup data arrays */
      CHKERRQ(DMPlexRestoreFEGeom(coordField, pointIS, qGeom, PETSC_TRUE, &fgeom));
      CHKERRQ(PetscQuadratureDestroy(&qGeom));
      CHKERRQ(PetscFree2(u, a));
      CHKERRQ(ISRestoreIndices(pointIS, &points));
    }
  }
  if (plex)  CHKERRQ(DMDestroy(&plex));
  if (plexA) CHKERRQ(DMDestroy(&plexA));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeBdIntegral - Form the integral over the specified boundary from the global input X using pointwise functions specified by the user

  Input Parameters:
+ dm      - The mesh
. X       - Global input vector
. label   - The boundary DMLabel
. numVals - The number of label values to use, or PETSC_DETERMINE for all values
. vals    - The label values to use, or PETSC_NULL for all values
. func    - The function to integrate along the boundary
- user    - The user context

  Output Parameter:
. integral - Integral for each field

  Level: developer

.seealso: DMPlexComputeIntegralFEM(), DMPlexComputeBdResidualFEM()
@*/
PetscErrorCode DMPlexComputeBdIntegral(DM dm, Vec X, DMLabel label, PetscInt numVals, const PetscInt vals[],
                                       void (*func)(PetscInt, PetscInt, PetscInt,
                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                    PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                       PetscScalar *integral, void *user)
{
  Vec            locX;
  PetscSection   section;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim, Nf, f, v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidPointer(label, 3);
  if (vals) PetscValidPointer(vals, 5);
  PetscValidPointer(integral, 7);
  CHKERRQ(PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMLabelGetStratumIS(depthLabel, dim-1, &facetIS));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscSectionGetNumFields(section, &Nf));
  /* Get local solution with boundary values */
  CHKERRQ(DMGetLocalVector(dm, &locX));
  CHKERRQ(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, 0.0, NULL, NULL, NULL));
  CHKERRQ(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
  CHKERRQ(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
  /* Loop over label values */
  CHKERRQ(PetscArrayzero(integral, Nf));
  for (v = 0; v < numVals; ++v) {
    IS           pointIS;
    PetscInt     numFaces, face;
    PetscScalar *fintegral;

    CHKERRQ(DMLabelGetStratumIS(label, vals[v], &pointIS));
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a priori if one is a superset of the other */
      CHKERRQ(ISIntersect_Caching_Internal(facetIS, pointIS, &isectIS));
      CHKERRQ(ISDestroy(&pointIS));
      pointIS = isectIS;
    }
    CHKERRQ(ISGetLocalSize(pointIS, &numFaces));
    CHKERRQ(PetscCalloc1(numFaces*Nf, &fintegral));
    CHKERRQ(DMPlexComputeBdIntegral_Internal(dm, locX, pointIS, func, fintegral, user));
    /* Sum point contributions into integral */
    for (f = 0; f < Nf; ++f) for (face = 0; face < numFaces; ++face) integral[f] += fintegral[face*Nf+f];
    CHKERRQ(PetscFree(fintegral));
    CHKERRQ(ISDestroy(&pointIS));
  }
  CHKERRQ(DMRestoreLocalVector(dm, &locX));
  CHKERRQ(ISDestroy(&facetIS));
  CHKERRQ(PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeInterpolatorNested - Form the local portion of the interpolation matrix I from the coarse DM to a uniformly refined DM.

  Input Parameters:
+ dmc  - The coarse mesh
. dmf  - The fine mesh
. isRefined - Flag indicating regular refinement, rather than the same topology
- user - The user context

  Output Parameter:
. In  - The interpolation matrix

  Level: developer

.seealso: DMPlexComputeInterpolatorGeneral(), DMPlexComputeJacobianFEM()
@*/
PetscErrorCode DMPlexComputeInterpolatorNested(DM dmc, DM dmf, PetscBool isRefined, Mat In, void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dmc->data;
  const char       *name  = "Interpolator";
  PetscFE          *feRef;
  PetscFV          *fvRef;
  PetscSection      fsection, fglobalSection;
  PetscSection      csection, cglobalSection;
  PetscScalar      *elemMat;
  PetscInt          dim, Nf, f, fieldI, fieldJ, offsetI, offsetJ, cStart, cEnd, c;
  PetscInt          cTotDim=0, rTotDim = 0;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_InterpolatorFEM,dmc,dmf,0,0));
  CHKERRQ(DMGetDimension(dmf, &dim));
  CHKERRQ(DMGetLocalSection(dmf, &fsection));
  CHKERRQ(DMGetGlobalSection(dmf, &fglobalSection));
  CHKERRQ(DMGetLocalSection(dmc, &csection));
  CHKERRQ(DMGetGlobalSection(dmc, &cglobalSection));
  CHKERRQ(PetscSectionGetNumFields(fsection, &Nf));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dmc, 0, &cStart, &cEnd));
  CHKERRQ(PetscCalloc2(Nf, &feRef, Nf, &fvRef));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj, objc;
    PetscClassId id, idc;
    PetscInt     rNb = 0, Nc = 0, cNb = 0;

    CHKERRQ(DMGetField(dmf, f, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      if (isRefined) {
        CHKERRQ(PetscFERefine(fe, &feRef[f]));
      } else {
        CHKERRQ(PetscObjectReference((PetscObject) fe));
        feRef[f] = fe;
      }
      CHKERRQ(PetscFEGetDimension(feRef[f], &rNb));
      CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      if (isRefined) {
        CHKERRQ(PetscFVRefine(fv, &fvRef[f]));
      } else {
        CHKERRQ(PetscObjectReference((PetscObject) fv));
        fvRef[f] = fv;
      }
      CHKERRQ(PetscFVGetDualSpace(fvRef[f], &Q));
      CHKERRQ(PetscDualSpaceGetDimension(Q, &rNb));
      CHKERRQ(PetscFVGetDualSpace(fv, &Q));
      CHKERRQ(PetscFVGetNumComponents(fv, &Nc));
    }
    CHKERRQ(DMGetField(dmc, f, NULL, &objc));
    CHKERRQ(PetscObjectGetClassId(objc, &idc));
    if (idc == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) objc;

      CHKERRQ(PetscFEGetDimension(fe, &cNb));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      CHKERRQ(PetscFVGetDualSpace(fv, &Q));
      CHKERRQ(PetscDualSpaceGetDimension(Q, &cNb));
    }
    rTotDim += rNb;
    cTotDim += cNb;
  }
  CHKERRQ(PetscMalloc1(rTotDim*cTotDim,&elemMat));
  CHKERRQ(PetscArrayzero(elemMat, rTotDim*cTotDim));
  for (fieldI = 0, offsetI = 0; fieldI < Nf; ++fieldI) {
    PetscDualSpace   Qref;
    PetscQuadrature  f;
    const PetscReal *qpoints, *qweights;
    PetscReal       *points;
    PetscInt         npoints = 0, Nc, Np, fpdim, i, k, p, d;

    /* Compose points from all dual basis functionals */
    if (feRef[fieldI]) {
      CHKERRQ(PetscFEGetDualSpace(feRef[fieldI], &Qref));
      CHKERRQ(PetscFEGetNumComponents(feRef[fieldI], &Nc));
    } else {
      CHKERRQ(PetscFVGetDualSpace(fvRef[fieldI], &Qref));
      CHKERRQ(PetscFVGetNumComponents(fvRef[fieldI], &Nc));
    }
    CHKERRQ(PetscDualSpaceGetDimension(Qref, &fpdim));
    for (i = 0; i < fpdim; ++i) {
      CHKERRQ(PetscDualSpaceGetFunctional(Qref, i, &f));
      CHKERRQ(PetscQuadratureGetData(f, NULL, NULL, &Np, NULL, NULL));
      npoints += Np;
    }
    CHKERRQ(PetscMalloc1(npoints*dim,&points));
    for (i = 0, k = 0; i < fpdim; ++i) {
      CHKERRQ(PetscDualSpaceGetFunctional(Qref, i, &f));
      CHKERRQ(PetscQuadratureGetData(f, NULL, NULL, &Np, &qpoints, NULL));
      for (p = 0; p < Np; ++p, ++k) for (d = 0; d < dim; ++d) points[k*dim+d] = qpoints[p*dim+d];
    }

    for (fieldJ = 0, offsetJ = 0; fieldJ < Nf; ++fieldJ) {
      PetscObject  obj;
      PetscClassId id;
      PetscInt     NcJ = 0, cpdim = 0, j, qNc;

      CHKERRQ(DMGetField(dmc, fieldJ, NULL, &obj));
      CHKERRQ(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) {
        PetscFE           fe = (PetscFE) obj;
        PetscTabulation T  = NULL;

        /* Evaluate basis at points */
        CHKERRQ(PetscFEGetNumComponents(fe, &NcJ));
        CHKERRQ(PetscFEGetDimension(fe, &cpdim));
        /* For now, fields only interpolate themselves */
        if (fieldI == fieldJ) {
          PetscCheck(Nc == NcJ,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %D does not match coarse field %D", Nc, NcJ);
          CHKERRQ(PetscFECreateTabulation(fe, 1, npoints, points, 0, &T));
          for (i = 0, k = 0; i < fpdim; ++i) {
            CHKERRQ(PetscDualSpaceGetFunctional(Qref, i, &f));
            CHKERRQ(PetscQuadratureGetData(f, NULL, &qNc, &Np, NULL, &qweights));
            PetscCheck(qNc == NcJ,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %D does not match coarse field %D", qNc, NcJ);
            for (p = 0; p < Np; ++p, ++k) {
              for (j = 0; j < cpdim; ++j) {
                /*
                   cTotDim:            Total columns in element interpolation matrix, sum of number of dual basis functionals in each field
                   offsetI, offsetJ:   Offsets into the larger element interpolation matrix for different fields
                   fpdim, i, cpdim, j: Dofs for fine and coarse grids, correspond to dual space basis functionals
                   qNC, Nc, Ncj, c:    Number of components in this field
                   Np, p:              Number of quad points in the fine grid functional i
                   k:                  i*Np + p, overall point number for the interpolation
                */
                for (c = 0; c < Nc; ++c) elemMat[(offsetI + i)*cTotDim + offsetJ + j] += T->T[0][k*cpdim*NcJ+j*Nc+c]*qweights[p*qNc+c];
              }
            }
          }
          CHKERRQ(PetscTabulationDestroy(&T));
        }
      } else if (id == PETSCFV_CLASSID) {
        PetscFV        fv = (PetscFV) obj;

        /* Evaluate constant function at points */
        CHKERRQ(PetscFVGetNumComponents(fv, &NcJ));
        cpdim = 1;
        /* For now, fields only interpolate themselves */
        if (fieldI == fieldJ) {
          PetscCheck(Nc == NcJ,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %D does not match coarse field %D", Nc, NcJ);
          for (i = 0, k = 0; i < fpdim; ++i) {
            CHKERRQ(PetscDualSpaceGetFunctional(Qref, i, &f));
            CHKERRQ(PetscQuadratureGetData(f, NULL, &qNc, &Np, NULL, &qweights));
            PetscCheck(qNc == NcJ,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %D does not match coarse field %D", qNc, NcJ);
            for (p = 0; p < Np; ++p, ++k) {
              for (j = 0; j < cpdim; ++j) {
                for (c = 0; c < Nc; ++c) elemMat[(offsetI + i)*cTotDim + offsetJ + j] += 1.0*qweights[p*qNc+c];
              }
            }
          }
        }
      }
      offsetJ += cpdim;
    }
    offsetI += fpdim;
    CHKERRQ(PetscFree(points));
  }
  if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(0, name, rTotDim, cTotDim, elemMat));
  /* Preallocate matrix */
  {
    Mat          preallocator;
    PetscScalar *vals;
    PetscInt    *cellCIndices, *cellFIndices;
    PetscInt     locRows, locCols, cell;

    CHKERRQ(MatGetLocalSize(In, &locRows, &locCols));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject) In), &preallocator));
    CHKERRQ(MatSetType(preallocator, MATPREALLOCATOR));
    CHKERRQ(MatSetSizes(preallocator, locRows, locCols, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRQ(MatSetUp(preallocator));
    CHKERRQ(PetscCalloc3(rTotDim*cTotDim, &vals,cTotDim,&cellCIndices,rTotDim,&cellFIndices));
    for (cell = cStart; cell < cEnd; ++cell) {
      if (isRefined) {
        CHKERRQ(DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, cell, cellCIndices, cellFIndices));
        CHKERRQ(MatSetValues(preallocator, rTotDim, cellFIndices, cTotDim, cellCIndices, vals, INSERT_VALUES));
      } else {
        CHKERRQ(DMPlexMatSetClosureGeneral(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, preallocator, cell, vals, INSERT_VALUES));
      }
    }
    CHKERRQ(PetscFree3(vals,cellCIndices,cellFIndices));
    CHKERRQ(MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatPreallocatorPreallocate(preallocator, PETSC_TRUE, In));
    CHKERRQ(MatDestroy(&preallocator));
  }
  /* Fill matrix */
  CHKERRQ(MatZeroEntries(In));
  for (c = cStart; c < cEnd; ++c) {
    if (isRefined) {
      CHKERRQ(DMPlexMatSetClosureRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, In, c, elemMat, INSERT_VALUES));
    } else {
      CHKERRQ(DMPlexMatSetClosureGeneral(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, In, c, elemMat, INSERT_VALUES));
    }
  }
  for (f = 0; f < Nf; ++f) CHKERRQ(PetscFEDestroy(&feRef[f]));
  CHKERRQ(PetscFree2(feRef,fvRef));
  CHKERRQ(PetscFree(elemMat));
  CHKERRQ(MatAssemblyBegin(In, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(In, MAT_FINAL_ASSEMBLY));
  if (mesh->printFEM > 1) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)In), "%s:\n", name));
    CHKERRQ(MatChop(In, 1.0e-10));
    CHKERRQ(MatView(In, NULL));
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_InterpolatorFEM,dmc,dmf,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeMassMatrixNested(DM dmc, DM dmf, Mat mass, void *user)
{
  SETERRQ(PetscObjectComm((PetscObject) dmc), PETSC_ERR_SUP, "Laziness");
}

/*@
  DMPlexComputeInterpolatorGeneral - Form the local portion of the interpolation matrix I from the coarse DM to a non-nested fine DM.

  Input Parameters:
+ dmf  - The fine mesh
. dmc  - The coarse mesh
- user - The user context

  Output Parameter:
. In  - The interpolation matrix

  Level: developer

.seealso: DMPlexComputeInterpolatorNested(), DMPlexComputeJacobianFEM()
@*/
PetscErrorCode DMPlexComputeInterpolatorGeneral(DM dmc, DM dmf, Mat In, void *user)
{
  DM_Plex       *mesh = (DM_Plex *) dmf->data;
  const char    *name = "Interpolator";
  PetscDS        prob;
  PetscSection   fsection, csection, globalFSection, globalCSection;
  PetscHSetIJ    ht;
  PetscLayout    rLayout;
  PetscInt      *dnz, *onz;
  PetscInt       locRows, rStart, rEnd;
  PetscReal     *x, *v0, *J, *invJ, detJ;
  PetscReal     *v0c, *Jc, *invJc, detJc;
  PetscScalar   *elemMat;
  PetscInt       dim, Nf, field, totDim, cStart, cEnd, cell, ccell;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_InterpolatorFEM,dmc,dmf,0,0));
  CHKERRQ(DMGetCoordinateDim(dmc, &dim));
  CHKERRQ(DMGetDS(dmc, &prob));
  CHKERRQ(PetscDSGetWorkspace(prob, &x, NULL, NULL, NULL, NULL));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ));
  CHKERRQ(PetscMalloc3(dim,&v0c,dim*dim,&Jc,dim*dim,&invJc));
  CHKERRQ(DMGetLocalSection(dmf, &fsection));
  CHKERRQ(DMGetGlobalSection(dmf, &globalFSection));
  CHKERRQ(DMGetLocalSection(dmc, &csection));
  CHKERRQ(DMGetGlobalSection(dmc, &globalCSection));
  CHKERRQ(DMPlexGetHeightStratum(dmf, 0, &cStart, &cEnd));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(PetscMalloc1(totDim, &elemMat));

  CHKERRQ(MatGetLocalSize(In, &locRows, NULL));
  CHKERRQ(PetscLayoutCreate(PetscObjectComm((PetscObject) In), &rLayout));
  CHKERRQ(PetscLayoutSetLocalSize(rLayout, locRows));
  CHKERRQ(PetscLayoutSetBlockSize(rLayout, 1));
  CHKERRQ(PetscLayoutSetUp(rLayout));
  CHKERRQ(PetscLayoutGetRange(rLayout, &rStart, &rEnd));
  CHKERRQ(PetscLayoutDestroy(&rLayout));
  CHKERRQ(PetscCalloc2(locRows,&dnz,locRows,&onz));
  CHKERRQ(PetscHSetIJCreate(&ht));
  for (field = 0; field < Nf; ++field) {
    PetscObject      obj;
    PetscClassId     id;
    PetscDualSpace   Q = NULL;
    PetscQuadrature  f;
    const PetscReal *qpoints;
    PetscInt         Nc, Np, fpdim, i, d;

    CHKERRQ(PetscDSGetDiscretization(prob, field, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      CHKERRQ(PetscFEGetDualSpace(fe, &Q));
      CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      CHKERRQ(PetscFVGetDualSpace(fv, &Q));
      Nc   = 1;
    }
    CHKERRQ(PetscDualSpaceGetDimension(Q, &fpdim));
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt *findices,   *cindices;
      PetscInt  numFIndices, numCIndices;

      CHKERRQ(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      CHKERRQ(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
      PetscCheck(numFIndices == fpdim,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fine indices %D != %D dual basis vecs", numFIndices, fpdim);
      for (i = 0; i < fpdim; ++i) {
        Vec             pointVec;
        PetscScalar    *pV;
        PetscSF         coarseCellSF = NULL;
        const PetscSFNode *coarseCells;
        PetscInt        numCoarseCells, q, c;

        /* Get points from the dual basis functional quadrature */
        CHKERRQ(PetscDualSpaceGetFunctional(Q, i, &f));
        CHKERRQ(PetscQuadratureGetData(f, NULL, NULL, &Np, &qpoints, NULL));
        CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, Np*dim, &pointVec));
        CHKERRQ(VecSetBlockSize(pointVec, dim));
        CHKERRQ(VecGetArray(pointVec, &pV));
        for (q = 0; q < Np; ++q) {
          const PetscReal xi0[3] = {-1., -1., -1.};

          /* Transform point to real space */
          CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
          for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
        }
        CHKERRQ(VecRestoreArray(pointVec, &pV));
        /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
        /* OPT: Pack all quad points from fine cell */
        CHKERRQ(DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF));
        CHKERRQ(PetscSFViewFromOptions(coarseCellSF, NULL, "-interp_sf_view"));
        /* Update preallocation info */
        CHKERRQ(PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells));
        PetscCheck(numCoarseCells == Np,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
        {
          PetscHashIJKey key;
          PetscBool      missing;

          key.i = findices[i];
          if (key.i >= 0) {
            /* Get indices for coarse elements */
            for (ccell = 0; ccell < numCoarseCells; ++ccell) {
              CHKERRQ(DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
              for (c = 0; c < numCIndices; ++c) {
                key.j = cindices[c];
                if (key.j < 0) continue;
                CHKERRQ(PetscHSetIJQueryAdd(ht, key, &missing));
                if (missing) {
                  if ((key.j >= rStart) && (key.j < rEnd)) ++dnz[key.i-rStart];
                  else                                     ++onz[key.i-rStart];
                }
              }
              CHKERRQ(DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
            }
          }
        }
        CHKERRQ(PetscSFDestroy(&coarseCellSF));
        CHKERRQ(VecDestroy(&pointVec));
      }
      CHKERRQ(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
    }
  }
  CHKERRQ(PetscHSetIJDestroy(&ht));
  CHKERRQ(MatXAIJSetPreallocation(In, 1, dnz, onz, NULL, NULL));
  CHKERRQ(MatSetOption(In, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  CHKERRQ(PetscFree2(dnz,onz));
  for (field = 0; field < Nf; ++field) {
    PetscObject       obj;
    PetscClassId      id;
    PetscDualSpace    Q = NULL;
    PetscTabulation T = NULL;
    PetscQuadrature   f;
    const PetscReal  *qpoints, *qweights;
    PetscInt          Nc, qNc, Np, fpdim, i, d;

    CHKERRQ(PetscDSGetDiscretization(prob, field, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      CHKERRQ(PetscFEGetDualSpace(fe, &Q));
      CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
      CHKERRQ(PetscFECreateTabulation(fe, 1, 1, x, 0, &T));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      CHKERRQ(PetscFVGetDualSpace(fv, &Q));
      Nc   = 1;
    } else SETERRQ(PetscObjectComm((PetscObject)dmc),PETSC_ERR_ARG_WRONG,"Unknown discretization type for field %D",field);
    CHKERRQ(PetscDualSpaceGetDimension(Q, &fpdim));
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt *findices,   *cindices;
      PetscInt  numFIndices, numCIndices;

      CHKERRQ(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      CHKERRQ(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
      PetscCheck(numFIndices == fpdim,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fine indices %D != %D dual basis vecs", numFIndices, fpdim);
      for (i = 0; i < fpdim; ++i) {
        Vec             pointVec;
        PetscScalar    *pV;
        PetscSF         coarseCellSF = NULL;
        const PetscSFNode *coarseCells;
        PetscInt        numCoarseCells, cpdim, q, c, j;

        /* Get points from the dual basis functional quadrature */
        CHKERRQ(PetscDualSpaceGetFunctional(Q, i, &f));
        CHKERRQ(PetscQuadratureGetData(f, NULL, &qNc, &Np, &qpoints, &qweights));
        PetscCheck(qNc == Nc,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %D does not match coarse field %D", qNc, Nc);
        CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, Np*dim, &pointVec));
        CHKERRQ(VecSetBlockSize(pointVec, dim));
        CHKERRQ(VecGetArray(pointVec, &pV));
        for (q = 0; q < Np; ++q) {
          const PetscReal xi0[3] = {-1., -1., -1.};

          /* Transform point to real space */
          CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
          for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
        }
        CHKERRQ(VecRestoreArray(pointVec, &pV));
        /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
        /* OPT: Read this out from preallocation information */
        CHKERRQ(DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF));
        /* Update preallocation info */
        CHKERRQ(PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells));
        PetscCheck(numCoarseCells == Np,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
        CHKERRQ(VecGetArray(pointVec, &pV));
        for (ccell = 0; ccell < numCoarseCells; ++ccell) {
          PetscReal pVReal[3];
          const PetscReal xi0[3] = {-1., -1., -1.};

          CHKERRQ(DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
          /* Transform points from real space to coarse reference space */
          CHKERRQ(DMPlexComputeCellGeometryFEM(dmc, coarseCells[ccell].index, NULL, v0c, Jc, invJc, &detJc));
          for (d = 0; d < dim; ++d) pVReal[d] = PetscRealPart(pV[ccell*dim+d]);
          CoordinatesRealToRef(dim, dim, xi0, v0c, invJc, pVReal, x);

          if (id == PETSCFE_CLASSID) {
            PetscFE fe = (PetscFE) obj;

            /* Evaluate coarse basis on contained point */
            CHKERRQ(PetscFEGetDimension(fe, &cpdim));
            CHKERRQ(PetscFEComputeTabulation(fe, 1, x, 0, T));
            CHKERRQ(PetscArrayzero(elemMat, cpdim));
            /* Get elemMat entries by multiplying by weight */
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[j] += T->T[0][j*Nc + c]*qweights[ccell*qNc + c];
            }
          } else {
            cpdim = 1;
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[j] += 1.0*qweights[ccell*qNc + c];
            }
          }
          /* Update interpolator */
          if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat));
          PetscCheck(numCIndices == cpdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %D != %D", numCIndices, cpdim);
          CHKERRQ(MatSetValues(In, 1, &findices[i], numCIndices, cindices, elemMat, INSERT_VALUES));
          CHKERRQ(DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
        }
        CHKERRQ(VecRestoreArray(pointVec, &pV));
        CHKERRQ(PetscSFDestroy(&coarseCellSF));
        CHKERRQ(VecDestroy(&pointVec));
      }
      CHKERRQ(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
    }
    if (id == PETSCFE_CLASSID) CHKERRQ(PetscTabulationDestroy(&T));
  }
  CHKERRQ(PetscFree3(v0,J,invJ));
  CHKERRQ(PetscFree3(v0c,Jc,invJc));
  CHKERRQ(PetscFree(elemMat));
  CHKERRQ(MatAssemblyBegin(In, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(In, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogEventEnd(DMPLEX_InterpolatorFEM,dmc,dmf,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeMassMatrixGeneral - Form the local portion of the mass matrix M from the coarse DM to a non-nested fine DM.

  Input Parameters:
+ dmf  - The fine mesh
. dmc  - The coarse mesh
- user - The user context

  Output Parameter:
. mass  - The mass matrix

  Level: developer

.seealso: DMPlexComputeMassMatrixNested(), DMPlexComputeInterpolatorNested(), DMPlexComputeInterpolatorGeneral(), DMPlexComputeJacobianFEM()
@*/
PetscErrorCode DMPlexComputeMassMatrixGeneral(DM dmc, DM dmf, Mat mass, void *user)
{
  DM_Plex       *mesh = (DM_Plex *) dmf->data;
  const char    *name = "Mass Matrix";
  PetscDS        prob;
  PetscSection   fsection, csection, globalFSection, globalCSection;
  PetscHSetIJ    ht;
  PetscLayout    rLayout;
  PetscInt      *dnz, *onz;
  PetscInt       locRows, rStart, rEnd;
  PetscReal     *x, *v0, *J, *invJ, detJ;
  PetscReal     *v0c, *Jc, *invJc, detJc;
  PetscScalar   *elemMat;
  PetscInt       dim, Nf, field, totDim, cStart, cEnd, cell, ccell;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinateDim(dmc, &dim));
  CHKERRQ(DMGetDS(dmc, &prob));
  CHKERRQ(PetscDSGetWorkspace(prob, &x, NULL, NULL, NULL, NULL));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ));
  CHKERRQ(PetscMalloc3(dim,&v0c,dim*dim,&Jc,dim*dim,&invJc));
  CHKERRQ(DMGetLocalSection(dmf, &fsection));
  CHKERRQ(DMGetGlobalSection(dmf, &globalFSection));
  CHKERRQ(DMGetLocalSection(dmc, &csection));
  CHKERRQ(DMGetGlobalSection(dmc, &globalCSection));
  CHKERRQ(DMPlexGetHeightStratum(dmf, 0, &cStart, &cEnd));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(PetscMalloc1(totDim, &elemMat));

  CHKERRQ(MatGetLocalSize(mass, &locRows, NULL));
  CHKERRQ(PetscLayoutCreate(PetscObjectComm((PetscObject) mass), &rLayout));
  CHKERRQ(PetscLayoutSetLocalSize(rLayout, locRows));
  CHKERRQ(PetscLayoutSetBlockSize(rLayout, 1));
  CHKERRQ(PetscLayoutSetUp(rLayout));
  CHKERRQ(PetscLayoutGetRange(rLayout, &rStart, &rEnd));
  CHKERRQ(PetscLayoutDestroy(&rLayout));
  CHKERRQ(PetscCalloc2(locRows,&dnz,locRows,&onz));
  CHKERRQ(PetscHSetIJCreate(&ht));
  for (field = 0; field < Nf; ++field) {
    PetscObject      obj;
    PetscClassId     id;
    PetscQuadrature  quad;
    const PetscReal *qpoints;
    PetscInt         Nq, Nc, i, d;

    CHKERRQ(PetscDSGetDiscretization(prob, field, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) CHKERRQ(PetscFEGetQuadrature((PetscFE) obj, &quad));
    else                       CHKERRQ(PetscFVGetQuadrature((PetscFV) obj, &quad));
    CHKERRQ(PetscQuadratureGetData(quad, NULL, &Nc, &Nq, &qpoints, NULL));
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      Vec                pointVec;
      PetscScalar       *pV;
      PetscSF            coarseCellSF = NULL;
      const PetscSFNode *coarseCells;
      PetscInt           numCoarseCells, q, c;
      PetscInt          *findices,   *cindices;
      PetscInt           numFIndices, numCIndices;

      CHKERRQ(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      CHKERRQ(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
      /* Get points from the quadrature */
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, Nq*dim, &pointVec));
      CHKERRQ(VecSetBlockSize(pointVec, dim));
      CHKERRQ(VecGetArray(pointVec, &pV));
      for (q = 0; q < Nq; ++q) {
        const PetscReal xi0[3] = {-1., -1., -1.};

        /* Transform point to real space */
        CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
        for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
      }
      CHKERRQ(VecRestoreArray(pointVec, &pV));
      /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
      CHKERRQ(DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF));
      CHKERRQ(PetscSFViewFromOptions(coarseCellSF, NULL, "-interp_sf_view"));
      /* Update preallocation info */
      CHKERRQ(PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells));
      PetscCheck(numCoarseCells == Nq,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
      {
        PetscHashIJKey key;
        PetscBool      missing;

        for (i = 0; i < numFIndices; ++i) {
          key.i = findices[i];
          if (key.i >= 0) {
            /* Get indices for coarse elements */
            for (ccell = 0; ccell < numCoarseCells; ++ccell) {
              CHKERRQ(DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
              for (c = 0; c < numCIndices; ++c) {
                key.j = cindices[c];
                if (key.j < 0) continue;
                CHKERRQ(PetscHSetIJQueryAdd(ht, key, &missing));
                if (missing) {
                  if ((key.j >= rStart) && (key.j < rEnd)) ++dnz[key.i-rStart];
                  else                                     ++onz[key.i-rStart];
                }
              }
              CHKERRQ(DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
            }
          }
        }
      }
      CHKERRQ(PetscSFDestroy(&coarseCellSF));
      CHKERRQ(VecDestroy(&pointVec));
      CHKERRQ(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
    }
  }
  CHKERRQ(PetscHSetIJDestroy(&ht));
  CHKERRQ(MatXAIJSetPreallocation(mass, 1, dnz, onz, NULL, NULL));
  CHKERRQ(MatSetOption(mass, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  CHKERRQ(PetscFree2(dnz,onz));
  for (field = 0; field < Nf; ++field) {
    PetscObject       obj;
    PetscClassId      id;
    PetscTabulation T, Tfine;
    PetscQuadrature   quad;
    const PetscReal  *qpoints, *qweights;
    PetscInt          Nq, Nc, i, d;

    CHKERRQ(PetscDSGetDiscretization(prob, field, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      CHKERRQ(PetscFEGetQuadrature((PetscFE) obj, &quad));
      CHKERRQ(PetscFEGetCellTabulation((PetscFE) obj, 1, &Tfine));
      CHKERRQ(PetscFECreateTabulation((PetscFE) obj, 1, 1, x, 0, &T));
    } else {
      CHKERRQ(PetscFVGetQuadrature((PetscFV) obj, &quad));
    }
    CHKERRQ(PetscQuadratureGetData(quad, NULL, &Nc, &Nq, &qpoints, &qweights));
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      Vec                pointVec;
      PetscScalar       *pV;
      PetscSF            coarseCellSF = NULL;
      const PetscSFNode *coarseCells;
      PetscInt           numCoarseCells, cpdim, q, c, j;
      PetscInt          *findices,   *cindices;
      PetscInt           numFIndices, numCIndices;

      CHKERRQ(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      CHKERRQ(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
      /* Get points from the quadrature */
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, Nq*dim, &pointVec));
      CHKERRQ(VecSetBlockSize(pointVec, dim));
      CHKERRQ(VecGetArray(pointVec, &pV));
      for (q = 0; q < Nq; ++q) {
        const PetscReal xi0[3] = {-1., -1., -1.};

        /* Transform point to real space */
        CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
        for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
      }
      CHKERRQ(VecRestoreArray(pointVec, &pV));
      /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
      CHKERRQ(DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF));
      /* Update matrix */
      CHKERRQ(PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells));
      PetscCheck(numCoarseCells == Nq,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
      CHKERRQ(VecGetArray(pointVec, &pV));
      for (ccell = 0; ccell < numCoarseCells; ++ccell) {
        PetscReal pVReal[3];
        const PetscReal xi0[3] = {-1., -1., -1.};

        CHKERRQ(DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
        /* Transform points from real space to coarse reference space */
        CHKERRQ(DMPlexComputeCellGeometryFEM(dmc, coarseCells[ccell].index, NULL, v0c, Jc, invJc, &detJc));
        for (d = 0; d < dim; ++d) pVReal[d] = PetscRealPart(pV[ccell*dim+d]);
        CoordinatesRealToRef(dim, dim, xi0, v0c, invJc, pVReal, x);

        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          /* Evaluate coarse basis on contained point */
          CHKERRQ(PetscFEGetDimension(fe, &cpdim));
          CHKERRQ(PetscFEComputeTabulation(fe, 1, x, 0, T));
          /* Get elemMat entries by multiplying by weight */
          for (i = 0; i < numFIndices; ++i) {
            CHKERRQ(PetscArrayzero(elemMat, cpdim));
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[j] += T->T[0][j*Nc + c]*Tfine->T[0][(ccell*numFIndices + i)*Nc + c]*qweights[ccell*Nc + c]*detJ;
            }
            /* Update interpolator */
            if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat));
            PetscCheck(numCIndices == cpdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %D != %D", numCIndices, cpdim);
            CHKERRQ(MatSetValues(mass, 1, &findices[i], numCIndices, cindices, elemMat, ADD_VALUES));
          }
        } else {
          cpdim = 1;
          for (i = 0; i < numFIndices; ++i) {
            CHKERRQ(PetscArrayzero(elemMat, cpdim));
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[j] += 1.0*1.0*qweights[ccell*Nc + c]*detJ;
            }
            /* Update interpolator */
            if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat));
            CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Nq: %D %D Nf: %D %D Nc: %D %D\n", ccell, Nq, i, numFIndices, j, numCIndices));
            PetscCheck(numCIndices == cpdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %D != %D", numCIndices, cpdim);
            CHKERRQ(MatSetValues(mass, 1, &findices[i], numCIndices, cindices, elemMat, ADD_VALUES));
          }
        }
        CHKERRQ(DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
      }
      CHKERRQ(VecRestoreArray(pointVec, &pV));
      CHKERRQ(PetscSFDestroy(&coarseCellSF));
      CHKERRQ(VecDestroy(&pointVec));
      CHKERRQ(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
    }
    if (id == PETSCFE_CLASSID) CHKERRQ(PetscTabulationDestroy(&T));
  }
  CHKERRQ(PetscFree3(v0,J,invJ));
  CHKERRQ(PetscFree3(v0c,Jc,invJc));
  CHKERRQ(PetscFree(elemMat));
  CHKERRQ(MatAssemblyBegin(mass, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mass, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeInjectorFEM - Compute a mapping from coarse unknowns to fine unknowns

  Input Parameters:
+ dmc  - The coarse mesh
- dmf  - The fine mesh
- user - The user context

  Output Parameter:
. sc   - The mapping

  Level: developer

.seealso: DMPlexComputeInterpolatorNested(), DMPlexComputeJacobianFEM()
@*/
PetscErrorCode DMPlexComputeInjectorFEM(DM dmc, DM dmf, VecScatter *sc, void *user)
{
  PetscDS        prob;
  PetscFE       *feRef;
  PetscFV       *fvRef;
  Vec            fv, cv;
  IS             fis, cis;
  PetscSection   fsection, fglobalSection, csection, cglobalSection;
  PetscInt      *cmap, *cellCIndices, *cellFIndices, *cindices, *findices;
  PetscInt       cTotDim, fTotDim = 0, Nf, f, field, cStart, cEnd, c, dim, d, startC, endC, offsetC, offsetF, m;
  PetscBool     *needAvg;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_InjectorFEM,dmc,dmf,0,0));
  CHKERRQ(DMGetDimension(dmf, &dim));
  CHKERRQ(DMGetLocalSection(dmf, &fsection));
  CHKERRQ(DMGetGlobalSection(dmf, &fglobalSection));
  CHKERRQ(DMGetLocalSection(dmc, &csection));
  CHKERRQ(DMGetGlobalSection(dmc, &cglobalSection));
  CHKERRQ(PetscSectionGetNumFields(fsection, &Nf));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dmc, 0, &cStart, &cEnd));
  CHKERRQ(DMGetDS(dmc, &prob));
  CHKERRQ(PetscCalloc3(Nf,&feRef,Nf,&fvRef,Nf,&needAvg));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     fNb = 0, Nc = 0;

    CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE    fe = (PetscFE) obj;
      PetscSpace sp;
      PetscInt   maxDegree;

      CHKERRQ(PetscFERefine(fe, &feRef[f]));
      CHKERRQ(PetscFEGetDimension(feRef[f], &fNb));
      CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
      CHKERRQ(PetscFEGetBasisSpace(fe, &sp));
      CHKERRQ(PetscSpaceGetDegree(sp, NULL, &maxDegree));
      if (!maxDegree) needAvg[f] = PETSC_TRUE;
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      CHKERRQ(PetscFVRefine(fv, &fvRef[f]));
      CHKERRQ(PetscFVGetDualSpace(fvRef[f], &Q));
      CHKERRQ(PetscDualSpaceGetDimension(Q, &fNb));
      CHKERRQ(PetscFVGetNumComponents(fv, &Nc));
      needAvg[f] = PETSC_TRUE;
    }
    fTotDim += fNb;
  }
  CHKERRQ(PetscDSGetTotalDimension(prob, &cTotDim));
  CHKERRQ(PetscMalloc1(cTotDim,&cmap));
  for (field = 0, offsetC = 0, offsetF = 0; field < Nf; ++field) {
    PetscFE        feC;
    PetscFV        fvC;
    PetscDualSpace QF, QC;
    PetscInt       order = -1, NcF, NcC, fpdim, cpdim;

    if (feRef[field]) {
      CHKERRQ(PetscDSGetDiscretization(prob, field, (PetscObject *) &feC));
      CHKERRQ(PetscFEGetNumComponents(feC, &NcC));
      CHKERRQ(PetscFEGetNumComponents(feRef[field], &NcF));
      CHKERRQ(PetscFEGetDualSpace(feRef[field], &QF));
      CHKERRQ(PetscDualSpaceGetOrder(QF, &order));
      CHKERRQ(PetscDualSpaceGetDimension(QF, &fpdim));
      CHKERRQ(PetscFEGetDualSpace(feC, &QC));
      CHKERRQ(PetscDualSpaceGetDimension(QC, &cpdim));
    } else {
      CHKERRQ(PetscDSGetDiscretization(prob, field, (PetscObject *) &fvC));
      CHKERRQ(PetscFVGetNumComponents(fvC, &NcC));
      CHKERRQ(PetscFVGetNumComponents(fvRef[field], &NcF));
      CHKERRQ(PetscFVGetDualSpace(fvRef[field], &QF));
      CHKERRQ(PetscDualSpaceGetDimension(QF, &fpdim));
      CHKERRQ(PetscFVGetDualSpace(fvC, &QC));
      CHKERRQ(PetscDualSpaceGetDimension(QC, &cpdim));
    }
    PetscCheck(NcF == NcC,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %D does not match coarse field %D", NcF, NcC);
    for (c = 0; c < cpdim; ++c) {
      PetscQuadrature  cfunc;
      const PetscReal *cqpoints, *cqweights;
      PetscInt         NqcC, NpC;
      PetscBool        found = PETSC_FALSE;

      CHKERRQ(PetscDualSpaceGetFunctional(QC, c, &cfunc));
      CHKERRQ(PetscQuadratureGetData(cfunc, NULL, &NqcC, &NpC, &cqpoints, &cqweights));
      PetscCheck(NqcC == NcC,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of quadrature components %D must match number of field components %D", NqcC, NcC);
      PetscCheckFalse(NpC != 1 && feRef[field],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not know how to do injection for moments");
      for (f = 0; f < fpdim; ++f) {
        PetscQuadrature  ffunc;
        const PetscReal *fqpoints, *fqweights;
        PetscReal        sum = 0.0;
        PetscInt         NqcF, NpF;

        CHKERRQ(PetscDualSpaceGetFunctional(QF, f, &ffunc));
        CHKERRQ(PetscQuadratureGetData(ffunc, NULL, &NqcF, &NpF, &fqpoints, &fqweights));
        PetscCheck(NqcF == NcF,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of quadrature components %D must match number of field components %D", NqcF, NcF);
        if (NpC != NpF) continue;
        for (d = 0; d < dim; ++d) sum += PetscAbsReal(cqpoints[d] - fqpoints[d]);
        if (sum > 1.0e-9) continue;
        for (d = 0; d < NcC; ++d) sum += PetscAbsReal(cqweights[d]*fqweights[d]);
        if (sum < 1.0e-9) continue;
        cmap[offsetC+c] = offsetF+f;
        found = PETSC_TRUE;
        break;
      }
      if (!found) {
        /* TODO We really want the average here, but some asshole put VecScatter in the interface */
        if (fvRef[field] || (feRef[field] && order == 0)) {
          cmap[offsetC+c] = offsetF+0;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not locate matching functional for injection");
      }
    }
    offsetC += cpdim;
    offsetF += fpdim;
  }
  for (f = 0; f < Nf; ++f) {CHKERRQ(PetscFEDestroy(&feRef[f]));CHKERRQ(PetscFVDestroy(&fvRef[f]));}
  CHKERRQ(PetscFree3(feRef,fvRef,needAvg));

  CHKERRQ(DMGetGlobalVector(dmf, &fv));
  CHKERRQ(DMGetGlobalVector(dmc, &cv));
  CHKERRQ(VecGetOwnershipRange(cv, &startC, &endC));
  CHKERRQ(PetscSectionGetConstrainedStorageSize(cglobalSection, &m));
  CHKERRQ(PetscMalloc2(cTotDim,&cellCIndices,fTotDim,&cellFIndices));
  CHKERRQ(PetscMalloc1(m,&cindices));
  CHKERRQ(PetscMalloc1(m,&findices));
  for (d = 0; d < m; ++d) cindices[d] = findices[d] = -1;
  for (c = cStart; c < cEnd; ++c) {
    CHKERRQ(DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, c, cellCIndices, cellFIndices));
    for (d = 0; d < cTotDim; ++d) {
      if ((cellCIndices[d] < startC) || (cellCIndices[d] >= endC)) continue;
      PetscCheckFalse((findices[cellCIndices[d]-startC] >= 0) && (findices[cellCIndices[d]-startC] != cellFIndices[cmap[d]]),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse dof %D maps to both %D and %D", cindices[cellCIndices[d]-startC], findices[cellCIndices[d]-startC], cellFIndices[cmap[d]]);
      cindices[cellCIndices[d]-startC] = cellCIndices[d];
      findices[cellCIndices[d]-startC] = cellFIndices[cmap[d]];
    }
  }
  CHKERRQ(PetscFree(cmap));
  CHKERRQ(PetscFree2(cellCIndices,cellFIndices));

  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, m, cindices, PETSC_OWN_POINTER, &cis));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, m, findices, PETSC_OWN_POINTER, &fis));
  CHKERRQ(VecScatterCreate(cv, cis, fv, fis, sc));
  CHKERRQ(ISDestroy(&cis));
  CHKERRQ(ISDestroy(&fis));
  CHKERRQ(DMRestoreGlobalVector(dmf, &fv));
  CHKERRQ(DMRestoreGlobalVector(dmc, &cv));
  CHKERRQ(PetscLogEventEnd(DMPLEX_InjectorFEM,dmc,dmf,0,0));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetCellFields - Retrieve the field values values for a chunk of cells

  Input Parameters:
+ dm     - The DM
. cellIS - The cells to include
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
- locA   - A local vector with auxiliary fields, or NULL

  Output Parameters:
+ u   - The field coefficients
. u_t - The fields derivative coefficients
- a   - The auxiliary field coefficients

  Level: developer

.seealso: DMPlexGetFaceFields()
@*/
PetscErrorCode DMPlexGetCellFields(DM dm, IS cellIS, Vec locX, Vec locX_t, Vec locA, PetscScalar **u, PetscScalar **u_t, PetscScalar **a)
{
  DM              plex, plexA = NULL;
  DMEnclosureType encAux;
  PetscSection    section, sectionAux;
  PetscDS         prob;
  const PetscInt *cells;
  PetscInt        cStart, cEnd, numCells, totDim, totDimAux, c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 3);
  if (locX_t) {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 4);}
  if (locA)   {PetscValidHeaderSpecific(locA, VEC_CLASSID, 5);}
  PetscValidPointer(u, 6);
  PetscValidPointer(u_t, 7);
  PetscValidPointer(a, 8);
  CHKERRQ(DMPlexConvertPlex(dm, &plex, PETSC_FALSE));
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &prob));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  if (locA) {
    DM      dmAux;
    PetscDS probAux;

    CHKERRQ(VecGetDM(locA, &dmAux));
    CHKERRQ(DMGetEnclosureRelation(dmAux, dm, &encAux));
    CHKERRQ(DMPlexConvertPlex(dmAux, &plexA, PETSC_FALSE));
    CHKERRQ(DMGetLocalSection(dmAux, &sectionAux));
    CHKERRQ(DMGetDS(dmAux, &probAux));
    CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
  }
  numCells = cEnd - cStart;
  CHKERRQ(DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, u));
  if (locX_t) CHKERRQ(DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, u_t)); else {*u_t = NULL;}
  if (locA)   CHKERRQ(DMGetWorkArray(dm, numCells*totDimAux, MPIU_SCALAR, a)); else {*a = NULL;}
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL, *x_t = NULL, *ul = *u, *ul_t = *u_t, *al = *a;
    PetscInt       i;

    CHKERRQ(DMPlexVecGetClosure(plex, section, locX, cell, NULL, &x));
    for (i = 0; i < totDim; ++i) ul[cind*totDim+i] = x[i];
    CHKERRQ(DMPlexVecRestoreClosure(plex, section, locX, cell, NULL, &x));
    if (locX_t) {
      CHKERRQ(DMPlexVecGetClosure(plex, section, locX_t, cell, NULL, &x_t));
      for (i = 0; i < totDim; ++i) ul_t[cind*totDim+i] = x_t[i];
      CHKERRQ(DMPlexVecRestoreClosure(plex, section, locX_t, cell, NULL, &x_t));
    }
    if (locA) {
      PetscInt subcell;
      CHKERRQ(DMGetEnclosurePoint(plexA, dm, encAux, cell, &subcell));
      CHKERRQ(DMPlexVecGetClosure(plexA, sectionAux, locA, subcell, NULL, &x));
      for (i = 0; i < totDimAux; ++i) al[cind*totDimAux+i] = x[i];
      CHKERRQ(DMPlexVecRestoreClosure(plexA, sectionAux, locA, subcell, NULL, &x));
    }
  }
  CHKERRQ(DMDestroy(&plex));
  if (locA) CHKERRQ(DMDestroy(&plexA));
  CHKERRQ(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreCellFields - Restore the field values values for a chunk of cells

  Input Parameters:
+ dm     - The DM
. cellIS - The cells to include
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
- locA   - A local vector with auxiliary fields, or NULL

  Output Parameters:
+ u   - The field coefficients
. u_t - The fields derivative coefficients
- a   - The auxiliary field coefficients

  Level: developer

.seealso: DMPlexGetFaceFields()
@*/
PetscErrorCode DMPlexRestoreCellFields(DM dm, IS cellIS, Vec locX, Vec locX_t, Vec locA, PetscScalar **u, PetscScalar **u_t, PetscScalar **a)
{
  PetscFunctionBegin;
  CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, u));
  if (locX_t) CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, u_t));
  if (locA)   CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, a));
  PetscFunctionReturn(0);
}

/*
  Get the auxiliary field vectors for the negative side (s = 0) and positive side (s = 1) of the interfaace
*/
static PetscErrorCode DMPlexGetHybridAuxFields(DM dm, DM dmAux[], PetscDS dsAux[], IS cellIS, Vec locA[], PetscScalar *a[])
{
  DM              plexA[2];
  DMEnclosureType encAux[2];
  PetscSection    sectionAux[2];
  const PetscInt *cells;
  PetscInt        cStart, cEnd, numCells, c, s, totDimAux[2];

  PetscFunctionBegin;
  PetscValidPointer(locA, 5);
  if (!locA[0] || !locA[1]) PetscFunctionReturn(0);
  PetscValidPointer(dmAux, 2);
  PetscValidPointer(dsAux, 3);
  PetscValidPointer(a, 6);
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  numCells = cEnd - cStart;
  for (s = 0; s < 2; ++s) {
    PetscValidHeaderSpecific(dmAux[s], DM_CLASSID, 2);
    PetscValidHeaderSpecific(dsAux[s], PETSCDS_CLASSID, 3);
    PetscValidHeaderSpecific(locA[s], VEC_CLASSID, 5);
    CHKERRQ(DMPlexConvertPlex(dmAux[s], &plexA[s], PETSC_FALSE));
    CHKERRQ(DMGetEnclosureRelation(dmAux[s], dm, &encAux[s]));
    CHKERRQ(DMGetLocalSection(dmAux[s], &sectionAux[s]));
    CHKERRQ(PetscDSGetTotalDimension(dsAux[s], &totDimAux[s]));
    CHKERRQ(DMGetWorkArray(dmAux[s], numCells*totDimAux[s], MPIU_SCALAR, &a[s]));
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt  cell = cells ? cells[c] : c;
    const PetscInt  cind = c - cStart;
    const PetscInt *cone, *ornt;

    CHKERRQ(DMPlexGetCone(dm, cell, &cone));
    CHKERRQ(DMPlexGetConeOrientation(dm, cell, &ornt));
    PetscCheckFalse(ornt[0],PETSC_COMM_SELF, PETSC_ERR_SUP, "Face %D in hybrid cell %D has orientation %D != 0", cone[0], cell, ornt[0]);
    for (s = 0; s < 2; ++s) {
      const PetscInt *support;
      PetscScalar    *x = NULL, *al = a[s];
      const PetscInt  tdA = totDimAux[s];
      PetscInt        ssize, scell;
      PetscInt        subface, Na, i;

      CHKERRQ(DMPlexGetSupport(dm, cone[s], &support));
      CHKERRQ(DMPlexGetSupportSize(dm, cone[s], &ssize));
      PetscCheckFalse(ssize != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D from cell %D has support size %D != 2", cone[s], cell, ssize);
      if      (support[0] == cell) scell = support[1];
      else if (support[1] == cell) scell = support[0];
      else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D does not have cell %D in its support", cone[s], cell);

      CHKERRQ(DMGetEnclosurePoint(plexA[s], dm, encAux[s], scell, &subface));
      CHKERRQ(DMPlexVecGetClosure(plexA[s], sectionAux[s], locA[s], subface, &Na, &x));
      for (i = 0; i < Na; ++i) al[cind*tdA+i] = x[i];
      CHKERRQ(DMPlexVecRestoreClosure(plexA[s], sectionAux[s], locA[s], subface, &Na, &x));
    }
  }
  for (s = 0; s < 2; ++s) CHKERRQ(DMDestroy(&plexA[s]));
  CHKERRQ(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexRestoreHybridAuxFields(DM dmAux[], PetscDS dsAux[], IS cellIS, Vec locA[], PetscScalar *a[])
{
  PetscFunctionBegin;
  if (!locA[0] || !locA[1]) PetscFunctionReturn(0);
  CHKERRQ(DMRestoreWorkArray(dmAux[0], 0, MPIU_SCALAR, &a[0]));
  CHKERRQ(DMRestoreWorkArray(dmAux[1], 0, MPIU_SCALAR, &a[1]));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetFaceFields - Retrieve the field values values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
. faceGeometry - A local vector with face geometry
. cellGeometry - A local vector with cell geometry
- locaGrad - A local vector with field gradients, or NULL

  Output Parameters:
+ Nface - The number of faces with field values
. uL - The field values at the left side of the face
- uR - The field values at the right side of the face

  Level: developer

.seealso: DMPlexGetCellFields()
@*/
PetscErrorCode DMPlexGetFaceFields(DM dm, PetscInt fStart, PetscInt fEnd, Vec locX, Vec locX_t, Vec faceGeometry, Vec cellGeometry, Vec locGrad, PetscInt *Nface, PetscScalar **uL, PetscScalar **uR)
{
  DM                 dmFace, dmCell, dmGrad = NULL;
  PetscSection       section;
  PetscDS            prob;
  DMLabel            ghostLabel;
  const PetscScalar *facegeom, *cellgeom, *x, *lgrad;
  PetscBool         *isFE;
  PetscInt           dim, Nf, f, Nc, numFaces = fEnd - fStart, iface, face;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 4);
  if (locX_t) {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 5);}
  PetscValidHeaderSpecific(faceGeometry, VEC_CLASSID, 6);
  PetscValidHeaderSpecific(cellGeometry, VEC_CLASSID, 7);
  if (locGrad) {PetscValidHeaderSpecific(locGrad, VEC_CLASSID, 8);}
  PetscValidPointer(uL, 10);
  PetscValidPointer(uR, 11);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscDSGetTotalComponents(prob, &Nc));
  CHKERRQ(PetscMalloc1(Nf, &isFE));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else                            {isFE[f] = PETSC_FALSE;}
  }
  CHKERRQ(DMGetLabel(dm, "ghost", &ghostLabel));
  CHKERRQ(VecGetArrayRead(locX, &x));
  CHKERRQ(VecGetDM(faceGeometry, &dmFace));
  CHKERRQ(VecGetArrayRead(faceGeometry, &facegeom));
  CHKERRQ(VecGetDM(cellGeometry, &dmCell));
  CHKERRQ(VecGetArrayRead(cellGeometry, &cellgeom));
  if (locGrad) {
    CHKERRQ(VecGetDM(locGrad, &dmGrad));
    CHKERRQ(VecGetArrayRead(locGrad, &lgrad));
  }
  CHKERRQ(DMGetWorkArray(dm, numFaces*Nc, MPIU_SCALAR, uL));
  CHKERRQ(DMGetWorkArray(dm, numFaces*Nc, MPIU_SCALAR, uR));
  /* Right now just eat the extra work for FE (could make a cell loop) */
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscFVCellGeom       *cgL, *cgR;
    PetscScalar           *xL, *xR, *gL, *gR;
    PetscScalar           *uLl = *uL, *uRl = *uR;
    PetscInt               ghost, nsupp, nchild;

    CHKERRQ(DMLabelGetValue(ghostLabel, face, &ghost));
    CHKERRQ(DMPlexGetSupportSize(dm, face, &nsupp));
    CHKERRQ(DMPlexGetTreeChildren(dm, face, &nchild, NULL));
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
    CHKERRQ(DMPlexPointLocalRead(dmFace, face, facegeom, &fg));
    CHKERRQ(DMPlexGetSupport(dm, face, &cells));
    CHKERRQ(DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL));
    CHKERRQ(DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR));
    for (f = 0; f < Nf; ++f) {
      PetscInt off;

      CHKERRQ(PetscDSGetComponentOffset(prob, f, &off));
      if (isFE[f]) {
        const PetscInt *cone;
        PetscInt        comp, coneSizeL, coneSizeR, faceLocL, faceLocR, ldof, rdof, d;

        xL = xR = NULL;
        CHKERRQ(PetscSectionGetFieldComponents(section, f, &comp));
        CHKERRQ(DMPlexVecGetClosure(dm, section, locX, cells[0], &ldof, (PetscScalar **) &xL));
        CHKERRQ(DMPlexVecGetClosure(dm, section, locX, cells[1], &rdof, (PetscScalar **) &xR));
        CHKERRQ(DMPlexGetCone(dm, cells[0], &cone));
        CHKERRQ(DMPlexGetConeSize(dm, cells[0], &coneSizeL));
        for (faceLocL = 0; faceLocL < coneSizeL; ++faceLocL) if (cone[faceLocL] == face) break;
        CHKERRQ(DMPlexGetCone(dm, cells[1], &cone));
        CHKERRQ(DMPlexGetConeSize(dm, cells[1], &coneSizeR));
        for (faceLocR = 0; faceLocR < coneSizeR; ++faceLocR) if (cone[faceLocR] == face) break;
        PetscCheckFalse(faceLocL == coneSizeL && faceLocR == coneSizeR,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %D in cone of cell %D or cell %D", face, cells[0], cells[1]);
        /* Check that FEM field has values in the right cell (sometimes its an FV ghost cell) */
        /* TODO: this is a hack that might not be right for nonconforming */
        if (faceLocL < coneSizeL) {
          CHKERRQ(PetscFEEvaluateFaceFields_Internal(prob, f, faceLocL, xL, &uLl[iface*Nc+off]));
          if (rdof == ldof && faceLocR < coneSizeR) CHKERRQ(PetscFEEvaluateFaceFields_Internal(prob, f, faceLocR, xR, &uRl[iface*Nc+off]));
          else              {for (d = 0; d < comp; ++d) uRl[iface*Nc+off+d] = uLl[iface*Nc+off+d];}
        }
        else {
          CHKERRQ(PetscFEEvaluateFaceFields_Internal(prob, f, faceLocR, xR, &uRl[iface*Nc+off]));
          CHKERRQ(PetscSectionGetFieldComponents(section, f, &comp));
          for (d = 0; d < comp; ++d) uLl[iface*Nc+off+d] = uRl[iface*Nc+off+d];
        }
        CHKERRQ(DMPlexVecRestoreClosure(dm, section, locX, cells[0], &ldof, (PetscScalar **) &xL));
        CHKERRQ(DMPlexVecRestoreClosure(dm, section, locX, cells[1], &rdof, (PetscScalar **) &xR));
      } else {
        PetscFV  fv;
        PetscInt numComp, c;

        CHKERRQ(PetscDSGetDiscretization(prob, f, (PetscObject *) &fv));
        CHKERRQ(PetscFVGetNumComponents(fv, &numComp));
        CHKERRQ(DMPlexPointLocalFieldRead(dm, cells[0], f, x, &xL));
        CHKERRQ(DMPlexPointLocalFieldRead(dm, cells[1], f, x, &xR));
        if (dmGrad) {
          PetscReal dxL[3], dxR[3];

          CHKERRQ(DMPlexPointLocalRead(dmGrad, cells[0], lgrad, &gL));
          CHKERRQ(DMPlexPointLocalRead(dmGrad, cells[1], lgrad, &gR));
          DMPlex_WaxpyD_Internal(dim, -1, cgL->centroid, fg->centroid, dxL);
          DMPlex_WaxpyD_Internal(dim, -1, cgR->centroid, fg->centroid, dxR);
          for (c = 0; c < numComp; ++c) {
            uLl[iface*Nc+off+c] = xL[c] + DMPlex_DotD_Internal(dim, &gL[c*dim], dxL);
            uRl[iface*Nc+off+c] = xR[c] + DMPlex_DotD_Internal(dim, &gR[c*dim], dxR);
          }
        } else {
          for (c = 0; c < numComp; ++c) {
            uLl[iface*Nc+off+c] = xL[c];
            uRl[iface*Nc+off+c] = xR[c];
          }
        }
      }
    }
    ++iface;
  }
  *Nface = iface;
  CHKERRQ(VecRestoreArrayRead(locX, &x));
  CHKERRQ(VecRestoreArrayRead(faceGeometry, &facegeom));
  CHKERRQ(VecRestoreArrayRead(cellGeometry, &cellgeom));
  if (locGrad) {
    CHKERRQ(VecRestoreArrayRead(locGrad, &lgrad));
  }
  CHKERRQ(PetscFree(isFE));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreFaceFields - Restore the field values values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
. faceGeometry - A local vector with face geometry
. cellGeometry - A local vector with cell geometry
- locaGrad - A local vector with field gradients, or NULL

  Output Parameters:
+ Nface - The number of faces with field values
. uL - The field values at the left side of the face
- uR - The field values at the right side of the face

  Level: developer

.seealso: DMPlexGetFaceFields()
@*/
PetscErrorCode DMPlexRestoreFaceFields(DM dm, PetscInt fStart, PetscInt fEnd, Vec locX, Vec locX_t, Vec faceGeometry, Vec cellGeometry, Vec locGrad, PetscInt *Nface, PetscScalar **uL, PetscScalar **uR)
{
  PetscFunctionBegin;
  CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, uL));
  CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, uR));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetFaceGeometry - Retrieve the geometric values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. faceGeometry - A local vector with face geometry
- cellGeometry - A local vector with cell geometry

  Output Parameters:
+ Nface - The number of faces with field values
. fgeom - The extract the face centroid and normal
- vol   - The cell volume

  Level: developer

.seealso: DMPlexGetCellFields()
@*/
PetscErrorCode DMPlexGetFaceGeometry(DM dm, PetscInt fStart, PetscInt fEnd, Vec faceGeometry, Vec cellGeometry, PetscInt *Nface, PetscFVFaceGeom **fgeom, PetscReal **vol)
{
  DM                 dmFace, dmCell;
  DMLabel            ghostLabel;
  const PetscScalar *facegeom, *cellgeom;
  PetscInt           dim, numFaces = fEnd - fStart, iface, face;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(faceGeometry, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(cellGeometry, VEC_CLASSID, 5);
  PetscValidPointer(fgeom, 7);
  PetscValidPointer(vol, 8);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetLabel(dm, "ghost", &ghostLabel));
  CHKERRQ(VecGetDM(faceGeometry, &dmFace));
  CHKERRQ(VecGetArrayRead(faceGeometry, &facegeom));
  CHKERRQ(VecGetDM(cellGeometry, &dmCell));
  CHKERRQ(VecGetArrayRead(cellGeometry, &cellgeom));
  CHKERRQ(PetscMalloc1(numFaces, fgeom));
  CHKERRQ(DMGetWorkArray(dm, numFaces*2, MPIU_SCALAR, vol));
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscFVCellGeom       *cgL, *cgR;
    PetscFVFaceGeom       *fgeoml = *fgeom;
    PetscReal             *voll   = *vol;
    PetscInt               ghost, d, nchild, nsupp;

    CHKERRQ(DMLabelGetValue(ghostLabel, face, &ghost));
    CHKERRQ(DMPlexGetSupportSize(dm, face, &nsupp));
    CHKERRQ(DMPlexGetTreeChildren(dm, face, &nchild, NULL));
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
    CHKERRQ(DMPlexPointLocalRead(dmFace, face, facegeom, &fg));
    CHKERRQ(DMPlexGetSupport(dm, face, &cells));
    CHKERRQ(DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL));
    CHKERRQ(DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR));
    for (d = 0; d < dim; ++d) {
      fgeoml[iface].centroid[d] = fg->centroid[d];
      fgeoml[iface].normal[d]   = fg->normal[d];
    }
    voll[iface*2+0] = cgL->volume;
    voll[iface*2+1] = cgR->volume;
    ++iface;
  }
  *Nface = iface;
  CHKERRQ(VecRestoreArrayRead(faceGeometry, &facegeom));
  CHKERRQ(VecRestoreArrayRead(cellGeometry, &cellgeom));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreFaceGeometry - Restore the field values values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. faceGeometry - A local vector with face geometry
- cellGeometry - A local vector with cell geometry

  Output Parameters:
+ Nface - The number of faces with field values
. fgeom - The extract the face centroid and normal
- vol   - The cell volume

  Level: developer

.seealso: DMPlexGetFaceFields()
@*/
PetscErrorCode DMPlexRestoreFaceGeometry(DM dm, PetscInt fStart, PetscInt fEnd, Vec faceGeometry, Vec cellGeometry, PetscInt *Nface, PetscFVFaceGeom **fgeom, PetscReal **vol)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(*fgeom));
  CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_REAL, vol));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESGetFEGeom(DMField coordField, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetId((PetscObject)quad,&id));
  CHKERRQ(PetscSNPrintf(composeStr, 32, "DMSNESGetFEGeom_%x\n", id));
  CHKERRQ(PetscObjectQuery((PetscObject) pointIS, composeStr, (PetscObject *) &container));
  if (container) {
    CHKERRQ(PetscContainerGetPointer(container, (void **) geom));
  } else {
    CHKERRQ(DMFieldCreateFEGeom(coordField, pointIS, quad, faceData, geom));
    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF,&container));
    CHKERRQ(PetscContainerSetPointer(container, (void *) *geom));
    CHKERRQ(PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom));
    CHKERRQ(PetscObjectCompose((PetscObject) pointIS, composeStr, (PetscObject) container));
    CHKERRQ(PetscContainerDestroy(&container));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESRestoreFEGeom(DMField coordField, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  PetscFunctionBegin;
  *geom = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeResidual_Patch_Internal(DM dm, PetscSection section, IS cellIS, PetscReal t, Vec locX, Vec locX_t, Vec locF, void *user)
{
  DM_Plex         *mesh       = (DM_Plex *) dm->data;
  const char      *name       = "Residual";
  DM               dmAux      = NULL;
  DMLabel          ghostLabel = NULL;
  PetscDS          prob       = NULL;
  PetscDS          probAux    = NULL;
  PetscBool        useFEM     = PETSC_FALSE;
  PetscBool        isImplicit = (locX_t || t == PETSC_MIN_REAL) ? PETSC_TRUE : PETSC_FALSE;
  DMField          coordField = NULL;
  Vec              locA;
  PetscScalar     *u = NULL, *u_t, *a, *uL = NULL, *uR = NULL;
  IS               chunkIS;
  const PetscInt  *cells;
  PetscInt         cStart, cEnd, numCells;
  PetscInt         Nf, f, totDim, totDimAux, numChunks, cellChunkSize, chunk, fStart, fEnd;
  PetscInt         maxDegree = PETSC_MAX_INT;
  PetscFormKey key;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0));
  /* FEM+FVM */
  /* 1: Get sizes from dm and dmAux */
  CHKERRQ(DMGetLabel(dm, "ghost", &ghostLabel));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(DMGetAuxiliaryVec(dm, NULL, 0, 0, &locA));
  if (locA) {
    CHKERRQ(VecGetDM(locA, &dmAux));
    CHKERRQ(DMGetDS(dmAux, &probAux));
    CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
  }
  /* 2: Get geometric data */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscBool    fimp;

    CHKERRQ(PetscDSGetImplicit(prob, f, &fimp));
    if (isImplicit != fimp) continue;
    CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {useFEM = PETSC_TRUE;}
    PetscCheck(id != PETSCFV_CLASSID,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Use of FVM with PCPATCH not yet implemented");
  }
  if (useFEM) {
    CHKERRQ(DMGetCoordinateField(dm, &coordField));
    CHKERRQ(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      CHKERRQ(DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad));
      if (affineQuad) {
        CHKERRQ(DMSNESGetFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
      }
    } else {
      CHKERRQ(PetscCalloc2(Nf,&quads,Nf,&geoms));
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscBool    fimp;

        CHKERRQ(PetscDSGetImplicit(prob, f, &fimp));
        if (isImplicit != fimp) continue;
        CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
        CHKERRQ(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          CHKERRQ(PetscFEGetQuadrature(fe, &quads[f]));
          CHKERRQ(PetscObjectReference((PetscObject)quads[f]));
          CHKERRQ(DMSNESGetFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
        }
      }
    }
  }
  /* Loop over chunks */
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  if (useFEM) CHKERRQ(ISCreate(PETSC_COMM_SELF, &chunkIS));
  numCells      = cEnd - cStart;
  numChunks     = 1;
  cellChunkSize = numCells/numChunks;
  numChunks     = PetscMin(1,numCells);
  key.label     = NULL;
  key.value     = 0;
  key.part      = 0;
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscScalar     *elemVec, *fluxL = NULL, *fluxR = NULL;
    PetscReal       *vol = NULL;
    PetscFVFaceGeom *fgeom = NULL;
    PetscInt         cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;
    PetscInt         numFaces = 0;

    /* Extract field coefficients */
    if (useFEM) {
      CHKERRQ(ISGetPointSubrange(chunkIS, cS, cE, cells));
      CHKERRQ(DMPlexGetCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a));
      CHKERRQ(DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
      CHKERRQ(PetscArrayzero(elemVec, numCells*totDim));
    }
    /* TODO We will interlace both our field coefficients (u, u_t, uL, uR, etc.) and our output (elemVec, fL, fR). I think this works */
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;
      PetscBool    fimp;
      PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

      key.field = f;
      CHKERRQ(PetscDSGetImplicit(prob, f, &fimp));
      if (isImplicit != fimp) continue;
      CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
      CHKERRQ(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) {
        PetscFE         fe = (PetscFE) obj;
        PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
        PetscFEGeom    *chunkGeom = NULL;
        PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
        PetscInt        Nq, Nb;

        CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
        CHKERRQ(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
        CHKERRQ(PetscFEGetDimension(fe, &Nb));
        blockSize = Nb;
        batchSize = numBlocks * blockSize;
        CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
        numChunks = numCells / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numCells % (numBatches*batchSize);
        offset    = numCells - Nr;
        /* Integrate FE residual to get elemVec (need fields at quadrature points) */
        /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
        CHKERRQ(PetscFEGeomGetChunk(geom,0,offset,&chunkGeom));
        CHKERRQ(PetscFEIntegrateResidual(prob, key, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec));
        CHKERRQ(PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom));
        CHKERRQ(PetscFEIntegrateResidual(prob, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]));
        CHKERRQ(PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom));
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        Ne = numFaces;
        /* Riemann solve over faces (need fields at face centroids) */
        /*   We need to evaluate FE fields at those coordinates */
        CHKERRQ(PetscFVIntegrateRHSFunction(fv, prob, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR));
      } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
    }
    /* Loop over domain */
    if (useFEM) {
      /* Add elemVec to locX */
      for (c = cS; c < cE; ++c) {
        const PetscInt cell = cells ? cells[c] : c;
        const PetscInt cind = c - cStart;

        if (mesh->printFEM > 1) CHKERRQ(DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]));
        if (ghostLabel) {
          PetscInt ghostVal;

          CHKERRQ(DMLabelGetValue(ghostLabel,cell,&ghostVal));
          if (ghostVal > 0) continue;
        }
        CHKERRQ(DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES));
      }
    }
    /* Handle time derivative */
    if (locX_t) {
      PetscScalar *x_t, *fa;

      CHKERRQ(VecGetArray(locF, &fa));
      CHKERRQ(VecGetArray(locX_t, &x_t));
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     pdim, d;

        CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
        CHKERRQ(PetscObjectGetClassId(obj, &id));
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        CHKERRQ(PetscFVGetNumComponents(fv, &pdim));
        for (c = cS; c < cE; ++c) {
          const PetscInt cell = cells ? cells[c] : c;
          PetscScalar   *u_t, *r;

          if (ghostLabel) {
            PetscInt ghostVal;

            CHKERRQ(DMLabelGetValue(ghostLabel, cell, &ghostVal));
            if (ghostVal > 0) continue;
          }
          CHKERRQ(DMPlexPointLocalFieldRead(dm, cell, f, x_t, &u_t));
          CHKERRQ(DMPlexPointLocalFieldRef(dm, cell, f, fa, &r));
          for (d = 0; d < pdim; ++d) r[d] += u_t[d];
        }
      }
      CHKERRQ(VecRestoreArray(locX_t, &x_t));
      CHKERRQ(VecRestoreArray(locF, &fa));
    }
    if (useFEM) {
      CHKERRQ(DMPlexRestoreCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a));
      CHKERRQ(DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
    }
  }
  if (useFEM) CHKERRQ(ISDestroy(&chunkIS));
  CHKERRQ(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  /* TODO Could include boundary residual here (see DMPlexComputeResidual_Internal) */
  if (useFEM) {
    if (maxDegree <= 1) {
      CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
      CHKERRQ(PetscQuadratureDestroy(&affineQuad));
    } else {
      for (f = 0; f < Nf; ++f) {
        CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
        CHKERRQ(PetscQuadratureDestroy(&quads[f]));
      }
      CHKERRQ(PetscFree2(quads,geoms));
    }
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*
  We always assemble JacP, and if the matrix is different from Jac and two different sets of point functions are provided, we also assemble Jac

  X   - The local solution vector
  X_t - The local solution time derivative vector, or NULL
*/
PetscErrorCode DMPlexComputeJacobian_Patch_Internal(DM dm, PetscSection section, PetscSection globalSection, IS cellIS,
                                                    PetscReal t, PetscReal X_tShift, Vec X, Vec X_t, Mat Jac, Mat JacP, void *ctx)
{
  DM_Plex         *mesh  = (DM_Plex *) dm->data;
  const char      *name = "Jacobian", *nameP = "JacobianPre";
  DM               dmAux = NULL;
  PetscDS          prob,   probAux = NULL;
  PetscSection     sectionAux = NULL;
  Vec              A;
  DMField          coordField;
  PetscFEGeom     *cgeomFEM;
  PetscQuadrature  qGeom = NULL;
  Mat              J = Jac, JP = JacP;
  PetscScalar     *work, *u = NULL, *u_t = NULL, *a = NULL, *elemMat = NULL, *elemMatP = NULL, *elemMatD = NULL;
  PetscBool        hasJac, hasPrec, hasDyn, assembleJac, *isFE, hasFV = PETSC_FALSE;
  const PetscInt  *cells;
  PetscFormKey key;
  PetscInt         Nf, fieldI, fieldJ, maxDegree, numCells, cStart, cEnd, numChunks, chunkSize, chunk, totDim, totDimAux = 0, sz, wsz, off = 0, offCell = 0;

  PetscFunctionBegin;
  CHKERRQ(ISGetLocalSize(cellIS, &numCells));
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  CHKERRQ(PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(DMGetAuxiliaryVec(dm, NULL, 0, 0, &A));
  if (A) {
    CHKERRQ(VecGetDM(A, &dmAux));
    CHKERRQ(DMGetLocalSection(dmAux, &sectionAux));
    CHKERRQ(DMGetDS(dmAux, &probAux));
  }
  /* Get flags */
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(DMGetWorkArray(dm, Nf, MPIU_BOOL, &isFE));
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscObject  disc;
    PetscClassId id;
    CHKERRQ(PetscDSGetDiscretization(prob, fieldI, &disc));
    CHKERRQ(PetscObjectGetClassId(disc, &id));
    if (id == PETSCFE_CLASSID)      {isFE[fieldI] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {hasFV = PETSC_TRUE; isFE[fieldI] = PETSC_FALSE;}
  }
  CHKERRQ(PetscDSHasJacobian(prob, &hasJac));
  CHKERRQ(PetscDSHasJacobianPreconditioner(prob, &hasPrec));
  CHKERRQ(PetscDSHasDynamicJacobian(prob, &hasDyn));
  assembleJac = hasJac && hasPrec && (Jac != JacP) ? PETSC_TRUE : PETSC_FALSE;
  hasDyn      = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  if (hasFV) CHKERRQ(MatSetOption(JP, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE)); /* No allocated space for FV stuff, so ignore the zero entries */
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  if (probAux) CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
  /* Compute batch sizes */
  if (isFE[0]) {
    PetscFE         fe;
    PetscQuadrature q;
    PetscInt        numQuadPoints, numBatches, batchSize, numBlocks, blockSize, Nb;

    CHKERRQ(PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe));
    CHKERRQ(PetscFEGetQuadrature(fe, &q));
    CHKERRQ(PetscQuadratureGetData(q, NULL, NULL, &numQuadPoints, NULL, NULL));
    CHKERRQ(PetscFEGetDimension(fe, &Nb));
    CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
    blockSize = Nb*numQuadPoints;
    batchSize = numBlocks  * blockSize;
    chunkSize = numBatches * batchSize;
    numChunks = numCells / chunkSize + numCells % chunkSize;
    CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
  } else {
    chunkSize = numCells;
    numChunks = 1;
  }
  /* Get work space */
  wsz  = (((X?1:0) + (X_t?1:0) + (dmAux?1:0))*totDim + ((hasJac?1:0) + (hasPrec?1:0) + (hasDyn?1:0))*totDim*totDim)*chunkSize;
  CHKERRQ(DMGetWorkArray(dm, wsz, MPIU_SCALAR, &work));
  CHKERRQ(PetscArrayzero(work, wsz));
  off      = 0;
  u        = X       ? (sz = chunkSize*totDim,        off += sz, work+off-sz) : NULL;
  u_t      = X_t     ? (sz = chunkSize*totDim,        off += sz, work+off-sz) : NULL;
  a        = dmAux   ? (sz = chunkSize*totDimAux,     off += sz, work+off-sz) : NULL;
  elemMat  = hasJac  ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  elemMatP = hasPrec ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  elemMatD = hasDyn  ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  PetscCheck(off == wsz,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error is workspace size %D should be %D", off, wsz);
  /* Setup geometry */
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  CHKERRQ(DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree));
  if (maxDegree <= 1) CHKERRQ(DMFieldCreateDefaultQuadrature(coordField, cellIS, &qGeom));
  if (!qGeom) {
    PetscFE fe;

    CHKERRQ(PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe));
    CHKERRQ(PetscFEGetQuadrature(fe, &qGeom));
    CHKERRQ(PetscObjectReference((PetscObject) qGeom));
  }
  CHKERRQ(DMSNESGetFEGeom(coordField, cellIS, qGeom, PETSC_FALSE, &cgeomFEM));
  /* Compute volume integrals */
  if (assembleJac) CHKERRQ(MatZeroEntries(J));
  CHKERRQ(MatZeroEntries(JP));
  key.label = NULL;
  key.value = 0;
  key.part  = 0;
  for (chunk = 0; chunk < numChunks; ++chunk, offCell += chunkSize) {
    const PetscInt   Ncell = PetscMin(chunkSize, numCells - offCell);
    PetscInt         c;

    /* Extract values */
    for (c = 0; c < Ncell; ++c) {
      const PetscInt cell = cells ? cells[c+offCell] : c+offCell;
      PetscScalar   *x = NULL,  *x_t = NULL;
      PetscInt       i;

      if (X) {
        CHKERRQ(DMPlexVecGetClosure(dm, section, X, cell, NULL, &x));
        for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
        CHKERRQ(DMPlexVecRestoreClosure(dm, section, X, cell, NULL, &x));
      }
      if (X_t) {
        CHKERRQ(DMPlexVecGetClosure(dm, section, X_t, cell, NULL, &x_t));
        for (i = 0; i < totDim; ++i) u_t[c*totDim+i] = x_t[i];
        CHKERRQ(DMPlexVecRestoreClosure(dm, section, X_t, cell, NULL, &x_t));
      }
      if (dmAux) {
        CHKERRQ(DMPlexVecGetClosure(dmAux, sectionAux, A, cell, NULL, &x));
        for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
        CHKERRQ(DMPlexVecRestoreClosure(dmAux, sectionAux, A, cell, NULL, &x));
      }
    }
    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscFE fe;
      CHKERRQ(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf + fieldJ;
        if (hasJac)  CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN,     key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMat));
        if (hasPrec) CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMatP));
        if (hasDyn)  CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMatD));
      }
      /* For finite volume, add the identity */
      if (!isFE[fieldI]) {
        PetscFV  fv;
        PetscInt eOffset = 0, Nc, fc, foff;

        CHKERRQ(PetscDSGetFieldOffset(prob, fieldI, &foff));
        CHKERRQ(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fv));
        CHKERRQ(PetscFVGetNumComponents(fv, &Nc));
        for (c = 0; c < chunkSize; ++c, eOffset += totDim*totDim) {
          for (fc = 0; fc < Nc; ++fc) {
            const PetscInt i = foff + fc;
            if (hasJac)  {elemMat [eOffset+i*totDim+i] = 1.0;}
            if (hasPrec) {elemMatP[eOffset+i*totDim+i] = 1.0;}
          }
        }
      }
    }
    /*   Add contribution from X_t */
    if (hasDyn) {for (c = 0; c < chunkSize*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];}
    /* Insert values into matrix */
    for (c = 0; c < Ncell; ++c) {
      const PetscInt cell = cells ? cells[c+offCell] : c+offCell;
      if (mesh->printFEM > 1) {
        if (hasJac)  CHKERRQ(DMPrintCellMatrix(cell, name,  totDim, totDim, &elemMat[(c-cStart)*totDim*totDim]));
        if (hasPrec) CHKERRQ(DMPrintCellMatrix(cell, nameP, totDim, totDim, &elemMatP[(c-cStart)*totDim*totDim]));
      }
      if (assembleJac) CHKERRQ(DMPlexMatSetClosure(dm, section, globalSection, Jac, cell, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES));
      CHKERRQ(DMPlexMatSetClosure(dm, section, globalSection, JP, cell, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES));
    }
  }
  /* Cleanup */
  CHKERRQ(DMSNESRestoreFEGeom(coordField, cellIS, qGeom, PETSC_FALSE, &cgeomFEM));
  CHKERRQ(PetscQuadratureDestroy(&qGeom));
  if (hasFV) CHKERRQ(MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  CHKERRQ(DMRestoreWorkArray(dm, Nf, MPIU_BOOL, &isFE));
  CHKERRQ(DMRestoreWorkArray(dm, ((1 + (X_t?1:0) + (dmAux?1:0))*totDim + ((hasJac?1:0) + (hasPrec?1:0) + (hasDyn?1:0))*totDim*totDim)*chunkSize, MPIU_SCALAR, &work));
  /* Compute boundary integrals */
  /* CHKERRQ(DMPlexComputeBdJacobian_Internal(dm, X, X_t, t, X_tShift, Jac, JacP, ctx)); */
  /* Assemble matrix */
  if (assembleJac) {CHKERRQ(MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY));CHKERRQ(MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY));}
  CHKERRQ(MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY));CHKERRQ(MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

/******** FEM Assembly Function ********/

static PetscErrorCode DMConvertPlex_Internal(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex));
  if (isPlex) {
    *plex = dm;
    CHKERRQ(PetscObjectReference((PetscObject) dm));
  } else {
    CHKERRQ(PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex));
    if (!*plex) {
      CHKERRQ(DMConvert(dm,DMPLEX,plex));
      CHKERRQ(PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex));
      if (copy) {
        CHKERRQ(DMCopyAuxiliaryVec(dm, *plex));
      }
    } else {
      CHKERRQ(PetscObjectReference((PetscObject) *plex));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetGeometryFVM - Return precomputed geometric data

  Collective on DM

  Input Parameter:
. dm - The DM

  Output Parameters:
+ facegeom - The values precomputed from face geometry
. cellgeom - The values precomputed from cell geometry
- minRadius - The minimum radius over the mesh of an inscribed sphere in a cell

  Level: developer

.seealso: DMTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexGetGeometryFVM(DM dm, Vec *facegeom, Vec *cellgeom, PetscReal *minRadius)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMConvertPlex_Internal(dm,&plex,PETSC_TRUE));
  CHKERRQ(DMPlexGetDataFVM(plex, NULL, cellgeom, facegeom, NULL));
  if (minRadius) CHKERRQ(DMPlexGetMinRadius(plex, minRadius));
  CHKERRQ(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetGradientDM - Return gradient data layout

  Collective on DM

  Input Parameters:
+ dm - The DM
- fv - The PetscFV

  Output Parameter:
. dmGrad - The layout for gradient values

  Level: developer

.seealso: DMPlexGetGeometryFVM()
@*/
PetscErrorCode DMPlexGetGradientDM(DM dm, PetscFV fv, DM *dmGrad)
{
  DM             plex;
  PetscBool      computeGradients;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(fv,PETSCFV_CLASSID,2);
  PetscValidPointer(dmGrad,3);
  CHKERRQ(PetscFVGetComputeGradients(fv, &computeGradients));
  if (!computeGradients) {*dmGrad = NULL; PetscFunctionReturn(0);}
  CHKERRQ(DMConvertPlex_Internal(dm,&plex,PETSC_TRUE));
  CHKERRQ(DMPlexGetDataFVM(plex, fv, NULL, NULL, dmGrad));
  CHKERRQ(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeBdResidual_Single_Internal(DM dm, PetscReal t, PetscWeakForm wf, PetscFormKey key, Vec locX, Vec locX_t, Vec locF, DMField coordField, IS facetIS)
{
  DM_Plex         *mesh = (DM_Plex *) dm->data;
  DM               plex = NULL, plexA = NULL;
  DMEnclosureType  encAux;
  PetscDS          prob, probAux = NULL;
  PetscSection     section, sectionAux = NULL;
  Vec              locA = NULL;
  PetscScalar     *u = NULL, *u_t = NULL, *a = NULL, *elemVec = NULL;
  PetscInt         totDim, totDimAux = 0;

  PetscFunctionBegin;
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &locA));
  if (locA) {
    DM dmAux;

    CHKERRQ(VecGetDM(locA, &dmAux));
    CHKERRQ(DMGetEnclosureRelation(dmAux, dm, &encAux));
    CHKERRQ(DMConvert(dmAux, DMPLEX, &plexA));
    CHKERRQ(DMGetDS(plexA, &probAux));
    CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
    CHKERRQ(DMGetLocalSection(plexA, &sectionAux));
  }
  {
    PetscFEGeom     *fgeom;
    PetscInt         maxDegree;
    PetscQuadrature  qGeom = NULL;
    IS               pointIS;
    const PetscInt  *points;
    PetscInt         numFaces, face, Nq;

    CHKERRQ(DMLabelGetStratumIS(key.label, key.value, &pointIS));
    if (!pointIS) goto end; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a priori if one is a superset of the other */
      CHKERRQ(ISIntersect_Caching_Internal(facetIS,pointIS,&isectIS));
      CHKERRQ(ISDestroy(&pointIS));
      pointIS = isectIS;
    }
    CHKERRQ(ISGetLocalSize(pointIS,&numFaces));
    CHKERRQ(ISGetIndices(pointIS,&points));
    CHKERRQ(PetscMalloc4(numFaces*totDim, &u, locX_t ? numFaces*totDim : 0, &u_t, numFaces*totDim, &elemVec, locA ? numFaces*totDimAux : 0, &a));
    CHKERRQ(DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      CHKERRQ(DMFieldCreateDefaultQuadrature(coordField,pointIS,&qGeom));
    }
    if (!qGeom) {
      PetscFE fe;

      CHKERRQ(PetscDSGetDiscretization(prob, key.field, (PetscObject *) &fe));
      CHKERRQ(PetscFEGetFaceQuadrature(fe, &qGeom));
      CHKERRQ(PetscObjectReference((PetscObject)qGeom));
    }
    CHKERRQ(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
    CHKERRQ(DMSNESGetFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom));
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;
      PetscScalar   *x     = NULL;
      PetscInt       i;

      CHKERRQ(DMPlexGetSupport(dm, point, &support));
      CHKERRQ(DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x));
      for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
      CHKERRQ(DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x));
      if (locX_t) {
        CHKERRQ(DMPlexVecGetClosure(plex, section, locX_t, support[0], NULL, &x));
        for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
        CHKERRQ(DMPlexVecRestoreClosure(plex, section, locX_t, support[0], NULL, &x));
      }
      if (locA) {
        PetscInt subp;

        CHKERRQ(DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp));
        CHKERRQ(DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x));
        for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
        CHKERRQ(DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x));
      }
    }
    CHKERRQ(PetscArrayzero(elemVec, numFaces*totDim));
    {
      PetscFE         fe;
      PetscInt        Nb;
      PetscFEGeom     *chunkGeom = NULL;
      /* Conforming batches */
      PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscInt        Nr, offset;

      CHKERRQ(PetscDSGetDiscretization(prob, key.field, (PetscObject *) &fe));
      CHKERRQ(PetscFEGetDimension(fe, &Nb));
      CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
      /* TODO: documentation is unclear about what is going on with these numbers: how should Nb / Nq factor in ? */
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      CHKERRQ(PetscFEGeomGetChunk(fgeom,0,offset,&chunkGeom));
      CHKERRQ(PetscFEIntegrateBdResidual(prob, wf, key, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec));
      CHKERRQ(PetscFEGeomRestoreChunk(fgeom, 0, offset, &chunkGeom));
      CHKERRQ(PetscFEGeomGetChunk(fgeom,offset,numFaces,&chunkGeom));
      CHKERRQ(PetscFEIntegrateBdResidual(prob, wf, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, &elemVec[offset*totDim]));
      CHKERRQ(PetscFEGeomRestoreChunk(fgeom,offset,numFaces,&chunkGeom));
    }
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;

      if (mesh->printFEM > 1) CHKERRQ(DMPrintCellVector(point, "BdResidual", totDim, &elemVec[face*totDim]));
      CHKERRQ(DMPlexGetSupport(plex, point, &support));
      CHKERRQ(DMPlexVecSetClosure(plex, NULL, locF, support[0], &elemVec[face*totDim], ADD_ALL_VALUES));
    }
    CHKERRQ(DMSNESRestoreFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom));
    CHKERRQ(PetscQuadratureDestroy(&qGeom));
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
    CHKERRQ(PetscFree4(u, u_t, elemVec, a));
  }
  end:
  CHKERRQ(DMDestroy(&plex));
  CHKERRQ(DMDestroy(&plexA));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdResidualSingle(DM dm, PetscReal t, PetscWeakForm wf, PetscFormKey key, Vec locX, Vec locX_t, Vec locF)
{
  DMField        coordField;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMLabelGetStratumIS(depthLabel, dim-1, &facetIS));
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  CHKERRQ(DMPlexComputeBdResidual_Single_Internal(dm, t, wf, key, locX, locX_t, locF, coordField, facetIS));
  CHKERRQ(ISDestroy(&facetIS));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdResidual_Internal(DM dm, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  PetscDS        prob;
  PetscInt       numBd, bd;
  DMField        coordField = NULL;
  IS             facetIS    = NULL;
  DMLabel        depthLabel;
  PetscInt       dim;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMLabelGetStratumIS(depthLabel,dim - 1,&facetIS));
  CHKERRQ(PetscDSGetNumBoundary(prob, &numBd));
  for (bd = 0; bd < numBd; ++bd) {
    PetscWeakForm           wf;
    DMBoundaryConditionType type;
    DMLabel                 label;
    const PetscInt         *values;
    PetscInt                field, numValues, v;
    PetscObject             obj;
    PetscClassId            id;
    PetscFormKey            key;

    CHKERRQ(PetscDSGetBoundary(prob, bd, &wf, &type, NULL, &label, &numValues, &values, &field, NULL, NULL, NULL, NULL, NULL));
    CHKERRQ(PetscDSGetDiscretization(prob, field, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    if (!facetIS) {
      DMLabel  depthLabel;
      PetscInt dim;

      CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
      CHKERRQ(DMGetDimension(dm, &dim));
      CHKERRQ(DMLabelGetStratumIS(depthLabel, dim - 1, &facetIS));
    }
    CHKERRQ(DMGetCoordinateField(dm, &coordField));
    for (v = 0; v < numValues; ++v) {
      key.label = label;
      key.value = values[v];
      key.field = field;
      key.part  = 0;
      CHKERRQ(DMPlexComputeBdResidual_Single_Internal(dm, t, wf, key, locX, locX_t, locF, coordField, facetIS));
    }
  }
  CHKERRQ(ISDestroy(&facetIS));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeResidual_Internal(DM dm, PetscFormKey key, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  DM_Plex         *mesh       = (DM_Plex *) dm->data;
  const char      *name       = "Residual";
  DM               dmAux      = NULL;
  DM               dmGrad     = NULL;
  DMLabel          ghostLabel = NULL;
  PetscDS          ds         = NULL;
  PetscDS          dsAux      = NULL;
  PetscSection     section    = NULL;
  PetscBool        useFEM     = PETSC_FALSE;
  PetscBool        useFVM     = PETSC_FALSE;
  PetscBool        isImplicit = (locX_t || time == PETSC_MIN_REAL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFV          fvm        = NULL;
  PetscFVCellGeom *cgeomFVM   = NULL;
  PetscFVFaceGeom *fgeomFVM   = NULL;
  DMField          coordField = NULL;
  Vec              locA, cellGeometryFVM = NULL, faceGeometryFVM = NULL, grad, locGrad = NULL;
  PetscScalar     *u = NULL, *u_t, *a, *uL, *uR;
  IS               chunkIS;
  const PetscInt  *cells;
  PetscInt         cStart, cEnd, numCells;
  PetscInt         Nf, f, totDim, totDimAux, numChunks, cellChunkSize, faceChunkSize, chunk, fStart, fEnd;
  PetscInt         maxDegree = PETSC_MAX_INT;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0));
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* TODO The FVM geometry is over-manipulated. Make the precalc functions return exactly what we need */
  /* FEM+FVM */
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  /* 1: Get sizes from dm and dmAux */
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetLabel(dm, "ghost", &ghostLabel));
  CHKERRQ(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds));
  CHKERRQ(PetscDSGetNumFields(ds, &Nf));
  CHKERRQ(PetscDSGetTotalDimension(ds, &totDim));
  CHKERRQ(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &locA));
  if (locA) {
    PetscInt subcell;
    CHKERRQ(VecGetDM(locA, &dmAux));
    CHKERRQ(DMGetEnclosurePoint(dmAux, dm, DM_ENC_UNKNOWN, cStart, &subcell));
    CHKERRQ(DMGetCellDS(dmAux, subcell, &dsAux));
    CHKERRQ(PetscDSGetTotalDimension(dsAux, &totDimAux));
  }
  /* 2: Get geometric data */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscBool    fimp;

    CHKERRQ(PetscDSGetImplicit(ds, f, &fimp));
    if (isImplicit != fimp) continue;
    CHKERRQ(PetscDSGetDiscretization(ds, f, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {useFEM = PETSC_TRUE;}
    if (id == PETSCFV_CLASSID) {useFVM = PETSC_TRUE; fvm = (PetscFV) obj;}
  }
  if (useFEM) {
    CHKERRQ(DMGetCoordinateField(dm, &coordField));
    CHKERRQ(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      CHKERRQ(DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad));
      if (affineQuad) {
        CHKERRQ(DMSNESGetFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
      }
    } else {
      CHKERRQ(PetscCalloc2(Nf,&quads,Nf,&geoms));
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscBool    fimp;

        CHKERRQ(PetscDSGetImplicit(ds, f, &fimp));
        if (isImplicit != fimp) continue;
        CHKERRQ(PetscDSGetDiscretization(ds, f, &obj));
        CHKERRQ(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          CHKERRQ(PetscFEGetQuadrature(fe, &quads[f]));
          CHKERRQ(PetscObjectReference((PetscObject)quads[f]));
          CHKERRQ(DMSNESGetFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
        }
      }
    }
  }
  if (useFVM) {
    CHKERRQ(DMPlexGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL));
    CHKERRQ(VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM));
    CHKERRQ(VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM));
    /* Reconstruct and limit cell gradients */
    CHKERRQ(DMPlexGetGradientDM(dm, fvm, &dmGrad));
    if (dmGrad) {
      CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
      CHKERRQ(DMGetGlobalVector(dmGrad, &grad));
      CHKERRQ(DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad));
      /* Communicate gradient values */
      CHKERRQ(DMGetLocalVector(dmGrad, &locGrad));
      CHKERRQ(DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad));
      CHKERRQ(DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad));
      CHKERRQ(DMRestoreGlobalVector(dmGrad, &grad));
    }
    /* Handle non-essential (e.g. outflow) boundary values */
    CHKERRQ(DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, time, faceGeometryFVM, cellGeometryFVM, locGrad));
  }
  /* Loop over chunks */
  if (useFEM) CHKERRQ(ISCreate(PETSC_COMM_SELF, &chunkIS));
  numCells      = cEnd - cStart;
  numChunks     = 1;
  cellChunkSize = numCells/numChunks;
  faceChunkSize = (fEnd - fStart)/numChunks;
  numChunks     = PetscMin(1,numCells);
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscScalar     *elemVec, *fluxL, *fluxR;
    PetscReal       *vol;
    PetscFVFaceGeom *fgeom;
    PetscInt         cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;
    PetscInt         fS = fStart+chunk*faceChunkSize, fE = PetscMin(fS+faceChunkSize, fEnd), numFaces = 0, face;

    /* Extract field coefficients */
    if (useFEM) {
      CHKERRQ(ISGetPointSubrange(chunkIS, cS, cE, cells));
      CHKERRQ(DMPlexGetCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a));
      CHKERRQ(DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
      CHKERRQ(PetscArrayzero(elemVec, numCells*totDim));
    }
    if (useFVM) {
      CHKERRQ(DMPlexGetFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR));
      CHKERRQ(DMPlexGetFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol));
      CHKERRQ(DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL));
      CHKERRQ(DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR));
      CHKERRQ(PetscArrayzero(fluxL, numFaces*totDim));
      CHKERRQ(PetscArrayzero(fluxR, numFaces*totDim));
    }
    /* TODO We will interlace both our field coefficients (u, u_t, uL, uR, etc.) and our output (elemVec, fL, fR). I think this works */
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;
      PetscBool    fimp;
      PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

      key.field = f;
      CHKERRQ(PetscDSGetImplicit(ds, f, &fimp));
      if (isImplicit != fimp) continue;
      CHKERRQ(PetscDSGetDiscretization(ds, f, &obj));
      CHKERRQ(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) {
        PetscFE         fe = (PetscFE) obj;
        PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
        PetscFEGeom    *chunkGeom = NULL;
        PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
        PetscInt        Nq, Nb;

        CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
        CHKERRQ(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
        CHKERRQ(PetscFEGetDimension(fe, &Nb));
        blockSize = Nb;
        batchSize = numBlocks * blockSize;
        CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
        numChunks = numCells / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numCells % (numBatches*batchSize);
        offset    = numCells - Nr;
        /* Integrate FE residual to get elemVec (need fields at quadrature points) */
        /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
        CHKERRQ(PetscFEGeomGetChunk(geom,0,offset,&chunkGeom));
        CHKERRQ(PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, u_t, dsAux, a, t, elemVec));
        CHKERRQ(PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom));
        CHKERRQ(PetscFEIntegrateResidual(ds, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]));
        CHKERRQ(PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom));
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        Ne = numFaces;
        /* Riemann solve over faces (need fields at face centroids) */
        /*   We need to evaluate FE fields at those coordinates */
        CHKERRQ(PetscFVIntegrateRHSFunction(fv, ds, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR));
      } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
    }
    /* Loop over domain */
    if (useFEM) {
      /* Add elemVec to locX */
      for (c = cS; c < cE; ++c) {
        const PetscInt cell = cells ? cells[c] : c;
        const PetscInt cind = c - cStart;

        if (mesh->printFEM > 1) CHKERRQ(DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]));
        if (ghostLabel) {
          PetscInt ghostVal;

          CHKERRQ(DMLabelGetValue(ghostLabel,cell,&ghostVal));
          if (ghostVal > 0) continue;
        }
        CHKERRQ(DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES));
      }
    }
    if (useFVM) {
      PetscScalar *fa;
      PetscInt     iface;

      CHKERRQ(VecGetArray(locF, &fa));
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     foff, pdim;

        CHKERRQ(PetscDSGetDiscretization(ds, f, &obj));
        CHKERRQ(PetscDSGetFieldOffset(ds, f, &foff));
        CHKERRQ(PetscObjectGetClassId(obj, &id));
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        CHKERRQ(PetscFVGetNumComponents(fv, &pdim));
        /* Accumulate fluxes to cells */
        for (face = fS, iface = 0; face < fE; ++face) {
          const PetscInt *scells;
          PetscScalar    *fL = NULL, *fR = NULL;
          PetscInt        ghost, d, nsupp, nchild;

          CHKERRQ(DMLabelGetValue(ghostLabel, face, &ghost));
          CHKERRQ(DMPlexGetSupportSize(dm, face, &nsupp));
          CHKERRQ(DMPlexGetTreeChildren(dm, face, &nchild, NULL));
          if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
          CHKERRQ(DMPlexGetSupport(dm, face, &scells));
          CHKERRQ(DMLabelGetValue(ghostLabel,scells[0],&ghost));
          if (ghost <= 0) CHKERRQ(DMPlexPointLocalFieldRef(dm, scells[0], f, fa, &fL));
          CHKERRQ(DMLabelGetValue(ghostLabel,scells[1],&ghost));
          if (ghost <= 0) CHKERRQ(DMPlexPointLocalFieldRef(dm, scells[1], f, fa, &fR));
          for (d = 0; d < pdim; ++d) {
            if (fL) fL[d] -= fluxL[iface*totDim+foff+d];
            if (fR) fR[d] += fluxR[iface*totDim+foff+d];
          }
          ++iface;
        }
      }
      CHKERRQ(VecRestoreArray(locF, &fa));
    }
    /* Handle time derivative */
    if (locX_t) {
      PetscScalar *x_t, *fa;

      CHKERRQ(VecGetArray(locF, &fa));
      CHKERRQ(VecGetArray(locX_t, &x_t));
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     pdim, d;

        CHKERRQ(PetscDSGetDiscretization(ds, f, &obj));
        CHKERRQ(PetscObjectGetClassId(obj, &id));
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        CHKERRQ(PetscFVGetNumComponents(fv, &pdim));
        for (c = cS; c < cE; ++c) {
          const PetscInt cell = cells ? cells[c] : c;
          PetscScalar   *u_t, *r;

          if (ghostLabel) {
            PetscInt ghostVal;

            CHKERRQ(DMLabelGetValue(ghostLabel, cell, &ghostVal));
            if (ghostVal > 0) continue;
          }
          CHKERRQ(DMPlexPointLocalFieldRead(dm, cell, f, x_t, &u_t));
          CHKERRQ(DMPlexPointLocalFieldRef(dm, cell, f, fa, &r));
          for (d = 0; d < pdim; ++d) r[d] += u_t[d];
        }
      }
      CHKERRQ(VecRestoreArray(locX_t, &x_t));
      CHKERRQ(VecRestoreArray(locF, &fa));
    }
    if (useFEM) {
      CHKERRQ(DMPlexRestoreCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a));
      CHKERRQ(DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
    }
    if (useFVM) {
      CHKERRQ(DMPlexRestoreFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR));
      CHKERRQ(DMPlexRestoreFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol));
      CHKERRQ(DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL));
      CHKERRQ(DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR));
      if (dmGrad) CHKERRQ(DMRestoreLocalVector(dmGrad, &locGrad));
    }
  }
  if (useFEM) CHKERRQ(ISDestroy(&chunkIS));
  CHKERRQ(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));

  if (useFEM) {
    CHKERRQ(DMPlexComputeBdResidual_Internal(dm, locX, locX_t, t, locF, user));

    if (maxDegree <= 1) {
      CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
      CHKERRQ(PetscQuadratureDestroy(&affineQuad));
    } else {
      for (f = 0; f < Nf; ++f) {
        CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
        CHKERRQ(PetscQuadratureDestroy(&quads[f]));
      }
      CHKERRQ(PetscFree2(quads,geoms));
    }
  }

  /* FEM */
  /* 1: Get sizes from dm and dmAux */
  /* 2: Get geometric data */
  /* 3: Handle boundary values */
  /* 4: Loop over domain */
  /*   Extract coefficients */
  /* Loop over fields */
  /*   Set tiling for FE*/
  /*   Integrate FE residual to get elemVec */
  /*     Loop over subdomain */
  /*       Loop over quad points */
  /*         Transform coords to real space */
  /*         Evaluate field and aux fields at point */
  /*         Evaluate residual at point */
  /*         Transform residual to real space */
  /*       Add residual to elemVec */
  /* Loop over domain */
  /*   Add elemVec to locX */

  /* FVM */
  /* Get geometric data */
  /* If using gradients */
  /*   Compute gradient data */
  /*   Loop over domain faces */
  /*     Count computational faces */
  /*     Reconstruct cell gradient */
  /*   Loop over domain cells */
  /*     Limit cell gradients */
  /* Handle boundary values */
  /* Loop over domain faces */
  /*   Read out field, centroid, normal, volume for each side of face */
  /* Riemann solve over faces */
  /* Loop over domain faces */
  /*   Accumulate fluxes to cells */
  /* TODO Change printFEM to printDisc here */
  if (mesh->printFEM) {
    Vec         locFbc;
    PetscInt    pStart, pEnd, p, maxDof;
    PetscScalar *zeroes;

    CHKERRQ(VecDuplicate(locF,&locFbc));
    CHKERRQ(VecCopy(locF,locFbc));
    CHKERRQ(PetscSectionGetChart(section,&pStart,&pEnd));
    CHKERRQ(PetscSectionGetMaxDof(section,&maxDof));
    CHKERRQ(PetscCalloc1(maxDof,&zeroes));
    for (p = pStart; p < pEnd; p++) {
      CHKERRQ(VecSetValuesSection(locFbc,section,p,zeroes,INSERT_BC_VALUES));
    }
    CHKERRQ(PetscFree(zeroes));
    CHKERRQ(DMPrintLocalVec(dm, name, mesh->printTol, locFbc));
    CHKERRQ(VecDestroy(&locFbc));
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*
  1) Allow multiple kernels for BdResidual for hybrid DS

  DONE 2) Get out dsAux for either side at the same time as cohesive cell dsAux

  DONE 3) Change DMGetCellFields() to get different aux data a[] for each side
     - I think I just need to replace a[] with the closure from each face

  4) Run both kernels for each non-hybrid field with correct dsAux, and then hybrid field as before
*/
PetscErrorCode DMPlexComputeResidual_Hybrid_Internal(DM dm, PetscFormKey key[], IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  DM_Plex         *mesh       = (DM_Plex *) dm->data;
  const char      *name       = "Hybrid Residual";
  DM               dmAux[3]   = {NULL, NULL, NULL};
  DMLabel          ghostLabel = NULL;
  PetscDS          ds         = NULL;
  PetscDS          dsAux[3]   = {NULL, NULL, NULL};
  Vec              locA[3]    = {NULL, NULL, NULL};
  PetscSection     section    = NULL;
  DMField          coordField = NULL;
  PetscScalar     *u = NULL, *u_t, *a[3];
  PetscScalar     *elemVec;
  IS               chunkIS;
  const PetscInt  *cells;
  PetscInt        *faces;
  PetscInt         cStart, cEnd, numCells;
  PetscInt         Nf, f, totDim, totDimAux[3], numChunks, cellChunkSize, chunk;
  PetscInt         maxDegree = PETSC_MAX_INT;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;

  PetscFunctionBegin;
  if ((key[0].label == key[1].label) && (key[0].value == key[1].value) && (key[0].part == key[1].part)) {
    const char *name;
    CHKERRQ(PetscObjectGetName((PetscObject) key[0].label, &name));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Form keys for each side of a cohesive surface must be different (%s, %D, %D)", name, key[0].value, key[0].part);
  }
  CHKERRQ(PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0));
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* FEM */
  CHKERRQ(ISGetLocalSize(cellIS, &numCells));
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  /* 1: Get sizes from dm and dmAux */
  CHKERRQ(DMGetSection(dm, &section));
  CHKERRQ(DMGetLabel(dm, "ghost", &ghostLabel));
  CHKERRQ(DMGetCellDS(dm, cStart, &ds));
  CHKERRQ(PetscDSGetNumFields(ds, &Nf));
  CHKERRQ(PetscDSGetTotalDimension(ds, &totDim));
  CHKERRQ(DMGetAuxiliaryVec(dm, key[2].label, key[2].value, key[2].part, &locA[2]));
  if (locA[2]) {
    CHKERRQ(VecGetDM(locA[2], &dmAux[2]));
    CHKERRQ(DMGetCellDS(dmAux[2], cStart, &dsAux[2]));
    CHKERRQ(PetscDSGetTotalDimension(dsAux[2], &totDimAux[2]));
    {
      const PetscInt *cone;
      PetscInt        c;

      CHKERRQ(DMPlexGetCone(dm, cStart, &cone));
      for (c = 0; c < 2; ++c) {
        const PetscInt *support;
        PetscInt ssize, s;

        CHKERRQ(DMPlexGetSupport(dm, cone[c], &support));
        CHKERRQ(DMPlexGetSupportSize(dm, cone[c], &ssize));
        PetscCheck(ssize == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D from cell %D has support size %D != 2", cone[c], cStart, ssize);
        if      (support[0] == cStart) s = 1;
        else if (support[1] == cStart) s = 0;
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D does not have cell %D in its support", cone[c], cStart);
        CHKERRQ(DMGetAuxiliaryVec(dm, key[c].label, key[c].value, key[c].part, &locA[c]));
        if (locA[c]) CHKERRQ(VecGetDM(locA[c], &dmAux[c]));
        else         {dmAux[c] = dmAux[2];}
        CHKERRQ(DMGetCellDS(dmAux[c], support[s], &dsAux[c]));
        CHKERRQ(PetscDSGetTotalDimension(dsAux[c], &totDimAux[c]));
      }
    }
  }
  /* 2: Setup geometric data */
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  CHKERRQ(DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree));
  if (maxDegree > 1) {
    CHKERRQ(PetscCalloc2(Nf, &quads, Nf, &geoms));
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      CHKERRQ(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      if (fe) {
        CHKERRQ(PetscFEGetQuadrature(fe, &quads[f]));
        CHKERRQ(PetscObjectReference((PetscObject) quads[f]));
      }
    }
  }
  /* Loop over chunks */
  cellChunkSize = numCells;
  numChunks     = !numCells ? 0 : PetscCeilReal(((PetscReal) numCells)/cellChunkSize);
  CHKERRQ(PetscCalloc1(2*cellChunkSize, &faces));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, cellChunkSize, faces, PETSC_USE_POINTER, &chunkIS));
  /* Extract field coefficients */
  /* NOTE This needs the end cap faces to have identical orientations */
  CHKERRQ(DMPlexGetCellFields(dm, cellIS, locX, locX_t, locA[2], &u, &u_t, &a[2]));
  CHKERRQ(DMPlexGetHybridAuxFields(dm, dmAux, dsAux, cellIS, locA, a));
  CHKERRQ(DMGetWorkArray(dm, cellChunkSize*totDim, MPIU_SCALAR, &elemVec));
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscInt cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;

    CHKERRQ(PetscMemzero(elemVec, cellChunkSize*totDim * sizeof(PetscScalar)));
    /* Get faces */
    for (c = cS; c < cE; ++c) {
      const PetscInt  cell = cells ? cells[c] : c;
      const PetscInt *cone;
      CHKERRQ(DMPlexGetCone(dm, cell, &cone));
      faces[(c-cS)*2+0] = cone[0];
      faces[(c-cS)*2+1] = cone[1];
    }
    CHKERRQ(ISGeneralSetIndices(chunkIS, cellChunkSize, faces, PETSC_USE_POINTER));
    /* Get geometric data */
    if (maxDegree <= 1) {
      if (!affineQuad) CHKERRQ(DMFieldCreateDefaultQuadrature(coordField, chunkIS, &affineQuad));
      if (affineQuad)  CHKERRQ(DMSNESGetFEGeom(coordField, chunkIS, affineQuad, PETSC_TRUE, &affineGeom));
    } else {
      for (f = 0; f < Nf; ++f) {
        if (quads[f]) CHKERRQ(DMSNESGetFEGeom(coordField, chunkIS, quads[f], PETSC_TRUE, &geoms[f]));
      }
    }
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscFE         fe;
      PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
      PetscFEGeom    *chunkGeom = NULL, *remGeom = NULL;
      PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
      PetscInt        numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset, Nq, Nb;
      PetscBool       isCohesiveField;

      CHKERRQ(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      if (!fe) continue;
      CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
      CHKERRQ(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
      CHKERRQ(PetscFEGetDimension(fe, &Nb));
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      CHKERRQ(PetscFEGeomGetChunk(geom,0,offset,&chunkGeom));
      CHKERRQ(PetscFEGeomGetChunk(geom,offset,numCells,&remGeom));
      CHKERRQ(PetscDSGetCohesive(ds, f, &isCohesiveField));
      chunkGeom->isCohesive = remGeom->isCohesive = PETSC_TRUE;
      key[0].field = f;
      key[1].field = f;
      key[2].field = f;
      CHKERRQ(PetscFEIntegrateHybridResidual(ds, key[0], 0, Ne, chunkGeom, u, u_t, dsAux[0], a[0], t, elemVec));
      CHKERRQ(PetscFEIntegrateHybridResidual(ds, key[0], 0, Nr, remGeom,  &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[0], &a[0][offset*totDimAux[0]], t, &elemVec[offset*totDim]));
      CHKERRQ(PetscFEIntegrateHybridResidual(ds, key[1], 1, Ne, chunkGeom, u, u_t, dsAux[1], a[1], t, elemVec));
      CHKERRQ(PetscFEIntegrateHybridResidual(ds, key[1], 1, Nr, remGeom,  &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[1], &a[1][offset*totDimAux[1]], t, &elemVec[offset*totDim]));
      CHKERRQ(PetscFEIntegrateHybridResidual(ds, key[2], 2, Ne, chunkGeom, u, u_t, dsAux[2], a[2], t, elemVec));
      CHKERRQ(PetscFEIntegrateHybridResidual(ds, key[2], 2, Nr, remGeom,  &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[2], &a[2][offset*totDimAux[2]], t, &elemVec[offset*totDim]));
      CHKERRQ(PetscFEGeomRestoreChunk(geom,offset,numCells,&remGeom));
      CHKERRQ(PetscFEGeomRestoreChunk(geom,0,offset,&chunkGeom));
    }
    /* Add elemVec to locX */
    for (c = cS; c < cE; ++c) {
      const PetscInt cell = cells ? cells[c] : c;
      const PetscInt cind = c - cStart;

      if (mesh->printFEM > 1) CHKERRQ(DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]));
      if (ghostLabel) {
        PetscInt ghostVal;

        CHKERRQ(DMLabelGetValue(ghostLabel,cell,&ghostVal));
        if (ghostVal > 0) continue;
      }
      CHKERRQ(DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES));
    }
  }
  CHKERRQ(DMPlexRestoreCellFields(dm, cellIS, locX, locX_t, locA[2], &u, &u_t, &a[2]));
  CHKERRQ(DMPlexRestoreHybridAuxFields(dmAux, dsAux, cellIS, locA, a));
  CHKERRQ(DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
  CHKERRQ(PetscFree(faces));
  CHKERRQ(ISDestroy(&chunkIS));
  CHKERRQ(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  if (maxDegree <= 1) {
    CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
    CHKERRQ(PetscQuadratureDestroy(&affineQuad));
  } else {
    for (f = 0; f < Nf; ++f) {
      if (geoms) CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
      if (quads) CHKERRQ(PetscQuadratureDestroy(&quads[f]));
    }
    CHKERRQ(PetscFree2(quads,geoms));
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobian_Single_Internal(DM dm, PetscReal t, PetscWeakForm wf, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt fieldI, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, DMField coordField, IS facetIS)
{
  DM_Plex        *mesh = (DM_Plex *) dm->data;
  DM              plex = NULL, plexA = NULL, tdm;
  DMEnclosureType encAux;
  PetscDS         prob, probAux = NULL;
  PetscSection    section, sectionAux = NULL;
  PetscSection    globalSection;
  Vec             locA = NULL, tv;
  PetscScalar    *u = NULL, *u_t = NULL, *a = NULL, *elemMat = NULL;
  PetscInt        v;
  PetscInt        Nf, totDim, totDimAux = 0;
  PetscBool       transform;

  PetscFunctionBegin;
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  CHKERRQ(DMHasBasisTransform(dm, &transform));
  CHKERRQ(DMGetBasisTransformDM_Internal(dm, &tdm));
  CHKERRQ(DMGetBasisTransformVec_Internal(dm, &tv));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(DMGetAuxiliaryVec(dm, label, values[0], 0, &locA));
  if (locA) {
    DM dmAux;

    CHKERRQ(VecGetDM(locA, &dmAux));
    CHKERRQ(DMGetEnclosureRelation(dmAux, dm, &encAux));
    CHKERRQ(DMConvert(dmAux, DMPLEX, &plexA));
    CHKERRQ(DMGetDS(plexA, &probAux));
    CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
    CHKERRQ(DMGetLocalSection(plexA, &sectionAux));
  }

  CHKERRQ(DMGetGlobalSection(dm, &globalSection));
  for (v = 0; v < numValues; ++v) {
    PetscFEGeom     *fgeom;
    PetscInt         maxDegree;
    PetscQuadrature  qGeom = NULL;
    IS               pointIS;
    const PetscInt  *points;
    PetscFormKey key;
    PetscInt         numFaces, face, Nq;

    key.label = label;
    key.value = values[v];
    key.part  = 0;
    CHKERRQ(DMLabelGetStratumIS(label, values[v], &pointIS));
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a prior if one is a superset of the other */
      CHKERRQ(ISIntersect_Caching_Internal(facetIS,pointIS,&isectIS));
      CHKERRQ(ISDestroy(&pointIS));
      pointIS = isectIS;
    }
    CHKERRQ(ISGetLocalSize(pointIS, &numFaces));
    CHKERRQ(ISGetIndices(pointIS, &points));
    CHKERRQ(PetscMalloc4(numFaces*totDim, &u, locX_t ? numFaces*totDim : 0, &u_t, numFaces*totDim*totDim, &elemMat, locA ? numFaces*totDimAux : 0, &a));
    CHKERRQ(DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      CHKERRQ(DMFieldCreateDefaultQuadrature(coordField,pointIS,&qGeom));
    }
    if (!qGeom) {
      PetscFE fe;

      CHKERRQ(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
      CHKERRQ(PetscFEGetFaceQuadrature(fe, &qGeom));
      CHKERRQ(PetscObjectReference((PetscObject)qGeom));
    }
    CHKERRQ(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
    CHKERRQ(DMSNESGetFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom));
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;
      PetscScalar   *x     = NULL;
      PetscInt       i;

      CHKERRQ(DMPlexGetSupport(dm, point, &support));
      CHKERRQ(DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x));
      for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
      CHKERRQ(DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x));
      if (locX_t) {
        CHKERRQ(DMPlexVecGetClosure(plex, section, locX_t, support[0], NULL, &x));
        for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
        CHKERRQ(DMPlexVecRestoreClosure(plex, section, locX_t, support[0], NULL, &x));
      }
      if (locA) {
        PetscInt subp;
        CHKERRQ(DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp));
        CHKERRQ(DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x));
        for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
        CHKERRQ(DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x));
      }
    }
    CHKERRQ(PetscArrayzero(elemMat, numFaces*totDim*totDim));
    {
      PetscFE         fe;
      PetscInt        Nb;
      /* Conforming batches */
      PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscFEGeom    *chunkGeom = NULL;
      PetscInt        fieldJ, Nr, offset;

      CHKERRQ(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
      CHKERRQ(PetscFEGetDimension(fe, &Nb));
      CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      CHKERRQ(PetscFEGeomGetChunk(fgeom,0,offset,&chunkGeom));
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf+fieldJ;
        CHKERRQ(PetscFEIntegrateBdJacobian(prob, wf, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat));
      }
      CHKERRQ(PetscFEGeomGetChunk(fgeom,offset,numFaces,&chunkGeom));
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf+fieldJ;
        CHKERRQ(PetscFEIntegrateBdJacobian(prob, wf, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, X_tShift, &elemMat[offset*totDim*totDim]));
      }
      CHKERRQ(PetscFEGeomRestoreChunk(fgeom,offset,numFaces,&chunkGeom));
    }
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;

      /* Transform to global basis before insertion in Jacobian */
      CHKERRQ(DMPlexGetSupport(plex, point, &support));
      if (transform) CHKERRQ(DMPlexBasisTransformPointTensor_Internal(dm, tdm, tv, support[0], PETSC_TRUE, totDim, &elemMat[face*totDim*totDim]));
      if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(point, "BdJacobian", totDim, totDim, &elemMat[face*totDim*totDim]));
      CHKERRQ(DMPlexMatSetClosure(plex, section, globalSection, JacP, support[0], &elemMat[face*totDim*totDim], ADD_VALUES));
    }
    CHKERRQ(DMSNESRestoreFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom));
    CHKERRQ(PetscQuadratureDestroy(&qGeom));
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
    CHKERRQ(PetscFree4(u, u_t, elemMat, a));
  }
  if (plex)  CHKERRQ(DMDestroy(&plex));
  if (plexA) CHKERRQ(DMDestroy(&plexA));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobianSingle(DM dm, PetscReal t, PetscWeakForm wf, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt field, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP)
{
  DMField        coordField;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMLabelGetStratumIS(depthLabel, dim-1, &facetIS));
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  CHKERRQ(DMPlexComputeBdJacobian_Single_Internal(dm, t, wf, label, numValues, values, field, locX, locX_t, X_tShift, Jac, JacP, coordField, facetIS));
  CHKERRQ(ISDestroy(&facetIS));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobian_Internal(DM dm, Vec locX, Vec locX_t, PetscReal t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  PetscDS          prob;
  PetscInt         dim, numBd, bd;
  DMLabel          depthLabel;
  DMField          coordField = NULL;
  IS               facetIS;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMLabelGetStratumIS(depthLabel, dim-1, &facetIS));
  CHKERRQ(PetscDSGetNumBoundary(prob, &numBd));
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  for (bd = 0; bd < numBd; ++bd) {
    PetscWeakForm           wf;
    DMBoundaryConditionType type;
    DMLabel                 label;
    const PetscInt         *values;
    PetscInt                fieldI, numValues;
    PetscObject             obj;
    PetscClassId            id;

    CHKERRQ(PetscDSGetBoundary(prob, bd, &wf, &type, NULL, &label, &numValues, &values, &fieldI, NULL, NULL, NULL, NULL, NULL));
    CHKERRQ(PetscDSGetDiscretization(prob, fieldI, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    CHKERRQ(DMPlexComputeBdJacobian_Single_Internal(dm, t, wf, label, numValues, values, fieldI, locX, locX_t, X_tShift, Jac, JacP, coordField, facetIS));
  }
  CHKERRQ(ISDestroy(&facetIS));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeJacobian_Internal(DM dm, PetscFormKey key, IS cellIS, PetscReal t, PetscReal X_tShift, Vec X, Vec X_t, Mat Jac, Mat JacP,void *user)
{
  DM_Plex        *mesh  = (DM_Plex *) dm->data;
  const char     *name  = "Jacobian";
  DM              dmAux = NULL, plex, tdm;
  DMEnclosureType encAux;
  Vec             A, tv;
  DMField         coordField;
  PetscDS         prob, probAux = NULL;
  PetscSection    section, globalSection, sectionAux;
  PetscScalar    *elemMat, *elemMatP, *elemMatD, *u, *u_t, *a = NULL;
  const PetscInt *cells;
  PetscInt        Nf, fieldI, fieldJ;
  PetscInt        totDim, totDimAux, cStart, cEnd, numCells, c;
  PetscBool       hasJac, hasPrec, hasDyn, hasFV = PETSC_FALSE, transform;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0));
  CHKERRQ(ISGetLocalSize(cellIS, &numCells));
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  CHKERRQ(DMHasBasisTransform(dm, &transform));
  CHKERRQ(DMGetBasisTransformDM_Internal(dm, &tdm));
  CHKERRQ(DMGetBasisTransformVec_Internal(dm, &tv));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetGlobalSection(dm, &globalSection));
  CHKERRQ(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &prob));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(PetscDSHasJacobian(prob, &hasJac));
  CHKERRQ(PetscDSHasJacobianPreconditioner(prob, &hasPrec));
  /* user passed in the same matrix, avoid double contributions and
     only assemble the Jacobian */
  if (hasJac && Jac == JacP) hasPrec = PETSC_FALSE;
  CHKERRQ(PetscDSHasDynamicJacobian(prob, &hasDyn));
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  CHKERRQ(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &A));
  if (A) {
    CHKERRQ(VecGetDM(A, &dmAux));
    CHKERRQ(DMGetEnclosureRelation(dmAux, dm, &encAux));
    CHKERRQ(DMConvert(dmAux, DMPLEX, &plex));
    CHKERRQ(DMGetLocalSection(plex, &sectionAux));
    CHKERRQ(DMGetDS(dmAux, &probAux));
    CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
  }
  CHKERRQ(PetscMalloc5(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,hasJac ? numCells*totDim*totDim : 0,&elemMat,hasPrec ? numCells*totDim*totDim : 0, &elemMatP,hasDyn ? numCells*totDim*totDim : 0, &elemMatD));
  if (dmAux) CHKERRQ(PetscMalloc1(numCells*totDimAux, &a));
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL,  *x_t = NULL;
    PetscInt       i;

    CHKERRQ(DMPlexVecGetClosure(dm, section, X, cell, NULL, &x));
    for (i = 0; i < totDim; ++i) u[cind*totDim+i] = x[i];
    CHKERRQ(DMPlexVecRestoreClosure(dm, section, X, cell, NULL, &x));
    if (X_t) {
      CHKERRQ(DMPlexVecGetClosure(dm, section, X_t, cell, NULL, &x_t));
      for (i = 0; i < totDim; ++i) u_t[cind*totDim+i] = x_t[i];
      CHKERRQ(DMPlexVecRestoreClosure(dm, section, X_t, cell, NULL, &x_t));
    }
    if (dmAux) {
      PetscInt subcell;
      CHKERRQ(DMGetEnclosurePoint(dmAux, dm, encAux, cell, &subcell));
      CHKERRQ(DMPlexVecGetClosure(plex, sectionAux, A, subcell, NULL, &x));
      for (i = 0; i < totDimAux; ++i) a[cind*totDimAux+i] = x[i];
      CHKERRQ(DMPlexVecRestoreClosure(plex, sectionAux, A, subcell, NULL, &x));
    }
  }
  if (hasJac)  CHKERRQ(PetscArrayzero(elemMat,  numCells*totDim*totDim));
  if (hasPrec) CHKERRQ(PetscArrayzero(elemMatP, numCells*totDim*totDim));
  if (hasDyn)  CHKERRQ(PetscArrayzero(elemMatD, numCells*totDim*totDim));
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscClassId    id;
    PetscFE         fe;
    PetscQuadrature qGeom = NULL;
    PetscInt        Nb;
    /* Conforming batches */
    PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt        Nr, offset, Nq;
    PetscInt        maxDegree;
    PetscFEGeom     *cgeomFEM, *chunkGeom = NULL, *remGeom = NULL;

    CHKERRQ(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
    CHKERRQ(PetscObjectGetClassId((PetscObject) fe, &id));
    if (id == PETSCFV_CLASSID) {hasFV = PETSC_TRUE; continue;}
    CHKERRQ(PetscFEGetDimension(fe, &Nb));
    CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
    CHKERRQ(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      CHKERRQ(DMFieldCreateDefaultQuadrature(coordField,cellIS,&qGeom));
    }
    if (!qGeom) {
      CHKERRQ(PetscFEGetQuadrature(fe,&qGeom));
      CHKERRQ(PetscObjectReference((PetscObject)qGeom));
    }
    CHKERRQ(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
    CHKERRQ(DMSNESGetFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM));
    blockSize = Nb;
    batchSize = numBlocks * blockSize;
    CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    CHKERRQ(PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom));
    CHKERRQ(PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&remGeom));
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      key.field = fieldI*Nf+fieldJ;
      if (hasJac) {
        CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat));
        CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]));
      }
      if (hasPrec) {
        CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatP));
        CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatP[offset*totDim*totDim]));
      }
      if (hasDyn) {
        CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatD));
        CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]));
      }
    }
    CHKERRQ(PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&remGeom));
    CHKERRQ(PetscFEGeomRestoreChunk(cgeomFEM,0,offset,&chunkGeom));
    CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM));
    CHKERRQ(PetscQuadratureDestroy(&qGeom));
  }
  /*   Add contribution from X_t */
  if (hasDyn) {for (c = 0; c < numCells*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];}
  if (hasFV) {
    PetscClassId id;
    PetscFV      fv;
    PetscInt     offsetI, NcI, NbI = 1, fc, f;

    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      CHKERRQ(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fv));
      CHKERRQ(PetscDSGetFieldOffset(prob, fieldI, &offsetI));
      CHKERRQ(PetscObjectGetClassId((PetscObject) fv, &id));
      if (id != PETSCFV_CLASSID) continue;
      /* Put in the identity */
      CHKERRQ(PetscFVGetNumComponents(fv, &NcI));
      for (c = cStart; c < cEnd; ++c) {
        const PetscInt cind    = c - cStart;
        const PetscInt eOffset = cind*totDim*totDim;
        for (fc = 0; fc < NcI; ++fc) {
          for (f = 0; f < NbI; ++f) {
            const PetscInt i = offsetI + f*NcI+fc;
            if (hasPrec) {
              if (hasJac) {elemMat[eOffset+i*totDim+i] = 1.0;}
              elemMatP[eOffset+i*totDim+i] = 1.0;
            } else {elemMat[eOffset+i*totDim+i] = 1.0;}
          }
        }
      }
    }
    /* No allocated space for FV stuff, so ignore the zero entries */
    CHKERRQ(MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  }
  /* Insert values into matrix */
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;

    /* Transform to global basis before insertion in Jacobian */
    if (transform) CHKERRQ(DMPlexBasisTransformPointTensor_Internal(dm, tdm, tv, cell, PETSC_TRUE, totDim, &elemMat[cind*totDim*totDim]));
    if (hasPrec) {
      if (hasJac) {
        if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
        CHKERRQ(DMPlexMatSetClosure(dm, section, globalSection, Jac, cell, &elemMat[cind*totDim*totDim], ADD_VALUES));
      }
      if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMatP[cind*totDim*totDim]));
      CHKERRQ(DMPlexMatSetClosure(dm, section, globalSection, JacP, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES));
    } else {
      if (hasJac) {
        if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
        CHKERRQ(DMPlexMatSetClosure(dm, section, globalSection, JacP, cell, &elemMat[cind*totDim*totDim], ADD_VALUES));
      }
    }
  }
  CHKERRQ(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  if (hasFV) CHKERRQ(MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  CHKERRQ(PetscFree5(u,u_t,elemMat,elemMatP,elemMatD));
  if (dmAux) {
    CHKERRQ(PetscFree(a));
    CHKERRQ(DMDestroy(&plex));
  }
  /* Compute boundary integrals */
  CHKERRQ(DMPlexComputeBdJacobian_Internal(dm, X, X_t, t, X_tShift, Jac, JacP, user));
  /* Assemble matrix */
  if (hasJac && hasPrec) {
    CHKERRQ(MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeJacobian_Hybrid_Internal(DM dm, PetscFormKey key[], IS cellIS, PetscReal t, PetscReal X_tShift, Vec locX, Vec locX_t, Mat Jac, Mat JacP, void *user)
{
  DM_Plex         *mesh          = (DM_Plex *) dm->data;
  const char      *name          = "Hybrid Jacobian";
  DM               dmAux[3]      = {NULL, NULL, NULL};
  DMLabel          ghostLabel    = NULL;
  DM               plex          = NULL;
  DM               plexA         = NULL;
  PetscDS          ds            = NULL;
  PetscDS          dsAux[3]      = {NULL, NULL, NULL};
  Vec              locA[3]       = {NULL, NULL, NULL};
  PetscSection     section       = NULL;
  PetscSection     sectionAux[3] = {NULL, NULL, NULL};
  DMField          coordField    = NULL;
  PetscScalar     *u = NULL, *u_t, *a[3];
  PetscScalar     *elemMat, *elemMatP;
  PetscSection     globalSection;
  IS               chunkIS;
  const PetscInt  *cells;
  PetscInt        *faces;
  PetscInt         cStart, cEnd, numCells;
  PetscInt         Nf, fieldI, fieldJ, totDim, totDimAux[3], numChunks, cellChunkSize, chunk;
  PetscInt         maxDegree = PETSC_MAX_INT;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;
  PetscBool        hasBdJac, hasBdPrec;

  PetscFunctionBegin;
  if ((key[0].label == key[1].label) && (key[0].value == key[1].value) && (key[0].part == key[1].part)) {
    const char *name;
    CHKERRQ(PetscObjectGetName((PetscObject) key[0].label, &name));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Form keys for each side of a cohesive surface must be different (%s, %D, %D)", name, key[0].value, key[0].part);
  }
  CHKERRQ(PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0));
  CHKERRQ(ISGetLocalSize(cellIS, &numCells));
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  CHKERRQ(DMGetSection(dm, &section));
  CHKERRQ(DMGetGlobalSection(dm, &globalSection));
  CHKERRQ(DMGetLabel(dm, "ghost", &ghostLabel));
  CHKERRQ(DMGetCellDS(dm, cStart, &ds));
  CHKERRQ(PetscDSGetNumFields(ds, &Nf));
  CHKERRQ(PetscDSGetTotalDimension(ds, &totDim));
  CHKERRQ(PetscDSHasBdJacobian(ds, &hasBdJac));
  CHKERRQ(PetscDSHasBdJacobianPreconditioner(ds, &hasBdPrec));
  CHKERRQ(DMGetAuxiliaryVec(dm, key[2].label, key[2].value, key[2].part, &locA[2]));
  if (locA[2]) {
    CHKERRQ(VecGetDM(locA[2], &dmAux[2]));
    CHKERRQ(DMConvert(dmAux[2], DMPLEX, &plexA));
    CHKERRQ(DMGetSection(dmAux[2], &sectionAux[2]));
    CHKERRQ(DMGetCellDS(dmAux[2], cStart, &dsAux[2]));
    CHKERRQ(PetscDSGetTotalDimension(dsAux[2], &totDimAux[2]));
    {
      const PetscInt *cone;
      PetscInt        c;

      CHKERRQ(DMPlexGetCone(dm, cStart, &cone));
      for (c = 0; c < 2; ++c) {
        const PetscInt *support;
        PetscInt ssize, s;

        CHKERRQ(DMPlexGetSupport(dm, cone[c], &support));
        CHKERRQ(DMPlexGetSupportSize(dm, cone[c], &ssize));
        PetscCheck(ssize == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D from cell %D has support size %D != 2", cone[c], cStart, ssize);
        if      (support[0] == cStart) s = 1;
        else if (support[1] == cStart) s = 0;
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D does not have cell %D in its support", cone[c], cStart);
        CHKERRQ(DMGetAuxiliaryVec(dm, key[c].label, key[c].value, key[c].part, &locA[c]));
        if (locA[c]) CHKERRQ(VecGetDM(locA[c], &dmAux[c]));
        else         {dmAux[c] = dmAux[2];}
        CHKERRQ(DMGetCellDS(dmAux[c], support[s], &dsAux[c]));
        CHKERRQ(PetscDSGetTotalDimension(dsAux[c], &totDimAux[c]));
      }
    }
  }
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  CHKERRQ(DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree));
  if (maxDegree > 1) {
    PetscInt f;
    CHKERRQ(PetscCalloc2(Nf, &quads, Nf, &geoms));
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      CHKERRQ(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      if (fe) {
        CHKERRQ(PetscFEGetQuadrature(fe, &quads[f]));
        CHKERRQ(PetscObjectReference((PetscObject) quads[f]));
      }
    }
  }
  cellChunkSize = numCells;
  numChunks     = !numCells ? 0 : PetscCeilReal(((PetscReal) numCells)/cellChunkSize);
  CHKERRQ(PetscCalloc1(2*cellChunkSize, &faces));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, cellChunkSize, faces, PETSC_USE_POINTER, &chunkIS));
  CHKERRQ(DMPlexGetCellFields(dm, cellIS, locX, locX_t, locA[2], &u, &u_t, &a[2]));
  CHKERRQ(DMPlexGetHybridAuxFields(dm, dmAux, dsAux, cellIS, locA, a));
  CHKERRQ(DMGetWorkArray(dm, hasBdJac  ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMat));
  CHKERRQ(DMGetWorkArray(dm, hasBdPrec ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMatP));
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscInt cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;

    if (hasBdJac)  CHKERRQ(PetscMemzero(elemMat,  numCells*totDim*totDim * sizeof(PetscScalar)));
    if (hasBdPrec) CHKERRQ(PetscMemzero(elemMatP, numCells*totDim*totDim * sizeof(PetscScalar)));
    /* Get faces */
    for (c = cS; c < cE; ++c) {
      const PetscInt  cell = cells ? cells[c] : c;
      const PetscInt *cone;
      CHKERRQ(DMPlexGetCone(plex, cell, &cone));
      faces[(c-cS)*2+0] = cone[0];
      faces[(c-cS)*2+1] = cone[1];
    }
    CHKERRQ(ISGeneralSetIndices(chunkIS, cellChunkSize, faces, PETSC_USE_POINTER));
    if (maxDegree <= 1) {
      if (!affineQuad) CHKERRQ(DMFieldCreateDefaultQuadrature(coordField, chunkIS, &affineQuad));
      if (affineQuad)  CHKERRQ(DMSNESGetFEGeom(coordField, chunkIS, affineQuad, PETSC_TRUE, &affineGeom));
    } else {
      PetscInt f;
      for (f = 0; f < Nf; ++f) {
        if (quads[f]) CHKERRQ(DMSNESGetFEGeom(coordField, chunkIS, quads[f], PETSC_TRUE, &geoms[f]));
      }
    }

    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscFE         feI;
      PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[fieldI];
      PetscFEGeom    *chunkGeom = NULL, *remGeom = NULL;
      PetscQuadrature quad = affineQuad ? affineQuad : quads[fieldI];
      PetscInt        numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset, Nq, Nb;
      PetscBool       isCohesiveField;

      CHKERRQ(PetscDSGetDiscretization(ds, fieldI, (PetscObject *) &feI));
      if (!feI) continue;
      CHKERRQ(PetscFEGetTileSizes(feI, NULL, &numBlocks, NULL, &numBatches));
      CHKERRQ(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
      CHKERRQ(PetscFEGetDimension(feI, &Nb));
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      CHKERRQ(PetscFESetTileSizes(feI, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      CHKERRQ(PetscFEGeomGetChunk(geom,0,offset,&chunkGeom));
      CHKERRQ(PetscFEGeomGetChunk(geom,offset,numCells,&remGeom));
      CHKERRQ(PetscDSGetCohesive(ds, fieldI, &isCohesiveField));
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        PetscFE feJ;

        CHKERRQ(PetscDSGetDiscretization(ds, fieldJ, (PetscObject *) &feJ));
        if (!feJ) continue;
        key[0].field = fieldI*Nf+fieldJ;
        key[1].field = fieldI*Nf+fieldJ;
        key[2].field = fieldI*Nf+fieldJ;
        if (hasBdJac) {
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[0], 0, Ne, chunkGeom, u, u_t, dsAux[0], a[0], t, X_tShift, elemMat));
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[0], 0, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[0], &a[0][offset*totDimAux[0]], t, X_tShift, &elemMat[offset*totDim*totDim]));
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[1], 1, Ne, chunkGeom, u, u_t, dsAux[1], a[1], t, X_tShift, elemMat));
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[1], 1, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[1], &a[1][offset*totDimAux[1]], t, X_tShift, &elemMat[offset*totDim*totDim]));
        }
        if (hasBdPrec) {
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[0], 0, Ne, chunkGeom, u, u_t, dsAux[0], a[0], t, X_tShift, elemMatP));
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[0], 0, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[0], &a[0][offset*totDimAux[0]], t, X_tShift, &elemMatP[offset*totDim*totDim]));
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[1], 1, Ne, chunkGeom, u, u_t, dsAux[1], a[1], t, X_tShift, elemMatP));
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[1], 1, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[1], &a[1][offset*totDimAux[1]], t, X_tShift, &elemMatP[offset*totDim*totDim]));
        }
        if (hasBdJac) {
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[2], 2, Ne, chunkGeom, u, u_t, dsAux[2], a[2], t, X_tShift, elemMat));
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[2], 2, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[2], &a[2][offset*totDimAux[2]], t, X_tShift, &elemMat[offset*totDim*totDim]));
        }
        if (hasBdPrec) {
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[2], 2, Ne, chunkGeom, u, u_t, dsAux[2], a[2], t, X_tShift, elemMatP));
          CHKERRQ(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[2], 2, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[2], &a[2][offset*totDimAux[2]], t, X_tShift, &elemMatP[offset*totDim*totDim]));
        }
      }
      CHKERRQ(PetscFEGeomRestoreChunk(geom,offset,numCells,&remGeom));
      CHKERRQ(PetscFEGeomRestoreChunk(geom,0,offset,&chunkGeom));
    }
    /* Insert values into matrix */
    for (c = cS; c < cE; ++c) {
      const PetscInt cell = cells ? cells[c] : c;
      const PetscInt cind = c - cS;

      if (hasBdPrec) {
        if (hasBdJac) {
          if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
          CHKERRQ(DMPlexMatSetClosure(plex, section, globalSection, Jac, cell, &elemMat[cind*totDim*totDim], ADD_VALUES));
        }
        if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMatP[cind*totDim*totDim]));
        CHKERRQ(DMPlexMatSetClosure(plex, section, globalSection, JacP, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES));
      } else if (hasBdJac) {
        if (mesh->printFEM > 1) CHKERRQ(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
        CHKERRQ(DMPlexMatSetClosure(plex, section, globalSection, JacP, cell, &elemMat[cind*totDim*totDim], ADD_VALUES));
      }
    }
  }
  CHKERRQ(DMPlexRestoreCellFields(dm, cellIS, locX, locX_t, locA[2], &u, &u_t, &a[2]));
  CHKERRQ(DMPlexRestoreHybridAuxFields(dmAux, dsAux, cellIS, locA, a));
  CHKERRQ(DMRestoreWorkArray(dm, hasBdJac  ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMat));
  CHKERRQ(DMRestoreWorkArray(dm, hasBdPrec ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMatP));
  CHKERRQ(PetscFree(faces));
  CHKERRQ(ISDestroy(&chunkIS));
  CHKERRQ(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  if (maxDegree <= 1) {
    CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
    CHKERRQ(PetscQuadratureDestroy(&affineQuad));
  } else {
    PetscInt f;
    for (f = 0; f < Nf; ++f) {
      if (geoms) CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE, &geoms[f]));
      if (quads) CHKERRQ(PetscQuadratureDestroy(&quads[f]));
    }
    CHKERRQ(PetscFree2(quads,geoms));
  }
  if (dmAux[2]) CHKERRQ(DMDestroy(&plexA));
  CHKERRQ(DMDestroy(&plex));
  /* Assemble matrix */
  if (hasBdJac && hasBdPrec) {
    CHKERRQ(MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*
  DMPlexComputeJacobian_Action_Internal - Form the local portion of the Jacobian action Z = J(X) Y at the local solution X using pointwise functions specified by the user.

  Input Parameters:
+ dm     - The mesh
. key    - The PetscWeakFormKey indcating where integration should happen
. cellIS - The cells to integrate over
. t      - The time
. X_tShift - The multiplier for the Jacobian with repsect to X_t
. X      - Local solution vector
. X_t    - Time-derivative of the local solution vector
. Y      - Local input vector
- user   - the user context

  Output Parameter:
. Z - Local output vector

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.
*/
PetscErrorCode DMPlexComputeJacobian_Action_Internal(DM dm, PetscFormKey key, IS cellIS, PetscReal t, PetscReal X_tShift, Vec X, Vec X_t, Vec Y, Vec Z, void *user)
{
  DM_Plex        *mesh  = (DM_Plex *) dm->data;
  const char     *name  = "Jacobian";
  DM              dmAux = NULL, plex, plexAux = NULL;
  DMEnclosureType encAux;
  Vec             A;
  DMField         coordField;
  PetscDS         prob, probAux = NULL;
  PetscQuadrature quad;
  PetscSection    section, globalSection, sectionAux;
  PetscScalar    *elemMat, *elemMatD, *u, *u_t, *a = NULL, *y, *z;
  const PetscInt *cells;
  PetscInt        Nf, fieldI, fieldJ;
  PetscInt        totDim, totDimAux = 0, cStart, cEnd, numCells, c;
  PetscBool       hasDyn;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  if (!cellIS) {
    PetscInt depth;

    CHKERRQ(DMPlexGetDepth(plex, &depth));
    CHKERRQ(DMGetStratumIS(plex, "dim", depth, &cellIS));
    if (!cellIS) CHKERRQ(DMGetStratumIS(plex, "depth", depth, &cellIS));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject) cellIS));
  }
  CHKERRQ(ISGetLocalSize(cellIS, &numCells));
  CHKERRQ(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetGlobalSection(dm, &globalSection));
  CHKERRQ(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &prob));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscDSGetTotalDimension(prob, &totDim));
  CHKERRQ(PetscDSHasDynamicJacobian(prob, &hasDyn));
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  CHKERRQ(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &A));
  if (A) {
    CHKERRQ(VecGetDM(A, &dmAux));
    CHKERRQ(DMGetEnclosureRelation(dmAux, dm, &encAux));
    CHKERRQ(DMConvert(dmAux, DMPLEX, &plexAux));
    CHKERRQ(DMGetLocalSection(plexAux, &sectionAux));
    CHKERRQ(DMGetDS(dmAux, &probAux));
    CHKERRQ(PetscDSGetTotalDimension(probAux, &totDimAux));
  }
  CHKERRQ(VecSet(Z, 0.0));
  CHKERRQ(PetscMalloc6(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,numCells*totDim*totDim,&elemMat,hasDyn ? numCells*totDim*totDim : 0, &elemMatD,numCells*totDim,&y,totDim,&z));
  if (dmAux) CHKERRQ(PetscMalloc1(numCells*totDimAux, &a));
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL,  *x_t = NULL;
    PetscInt       i;

    CHKERRQ(DMPlexVecGetClosure(plex, section, X, cell, NULL, &x));
    for (i = 0; i < totDim; ++i) u[cind*totDim+i] = x[i];
    CHKERRQ(DMPlexVecRestoreClosure(plex, section, X, cell, NULL, &x));
    if (X_t) {
      CHKERRQ(DMPlexVecGetClosure(plex, section, X_t, cell, NULL, &x_t));
      for (i = 0; i < totDim; ++i) u_t[cind*totDim+i] = x_t[i];
      CHKERRQ(DMPlexVecRestoreClosure(plex, section, X_t, cell, NULL, &x_t));
    }
    if (dmAux) {
      PetscInt subcell;
      CHKERRQ(DMGetEnclosurePoint(dmAux, dm, encAux, cell, &subcell));
      CHKERRQ(DMPlexVecGetClosure(plexAux, sectionAux, A, subcell, NULL, &x));
      for (i = 0; i < totDimAux; ++i) a[cind*totDimAux+i] = x[i];
      CHKERRQ(DMPlexVecRestoreClosure(plexAux, sectionAux, A, subcell, NULL, &x));
    }
    CHKERRQ(DMPlexVecGetClosure(plex, section, Y, cell, NULL, &x));
    for (i = 0; i < totDim; ++i) y[cind*totDim+i] = x[i];
    CHKERRQ(DMPlexVecRestoreClosure(plex, section, Y, cell, NULL, &x));
  }
  CHKERRQ(PetscArrayzero(elemMat, numCells*totDim*totDim));
  if (hasDyn)  CHKERRQ(PetscArrayzero(elemMatD, numCells*totDim*totDim));
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscFE  fe;
    PetscInt Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset, Nq;
    PetscQuadrature qGeom = NULL;
    PetscInt    maxDegree;
    PetscFEGeom *cgeomFEM, *chunkGeom = NULL, *remGeom = NULL;

    CHKERRQ(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
    CHKERRQ(PetscFEGetQuadrature(fe, &quad));
    CHKERRQ(PetscFEGetDimension(fe, &Nb));
    CHKERRQ(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
    CHKERRQ(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
    if (maxDegree <= 1) CHKERRQ(DMFieldCreateDefaultQuadrature(coordField,cellIS,&qGeom));
    if (!qGeom) {
      CHKERRQ(PetscFEGetQuadrature(fe,&qGeom));
      CHKERRQ(PetscObjectReference((PetscObject)qGeom));
    }
    CHKERRQ(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
    CHKERRQ(DMSNESGetFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM));
    blockSize = Nb;
    batchSize = numBlocks * blockSize;
    CHKERRQ(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    CHKERRQ(PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom));
    CHKERRQ(PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&remGeom));
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      key.field = fieldI*Nf + fieldJ;
      CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat));
      CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]));
      if (hasDyn) {
        CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatD));
        CHKERRQ(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]));
      }
    }
    CHKERRQ(PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&remGeom));
    CHKERRQ(PetscFEGeomRestoreChunk(cgeomFEM,0,offset,&chunkGeom));
    CHKERRQ(DMSNESRestoreFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM));
    CHKERRQ(PetscQuadratureDestroy(&qGeom));
  }
  if (hasDyn) {
    for (c = 0; c < numCells*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt     cell = cells ? cells[c] : c;
    const PetscInt     cind = c - cStart;
    const PetscBLASInt M = totDim, one = 1;
    const PetscScalar  a = 1.0, b = 0.0;

    PetscStackCallBLAS("BLASgemv", BLASgemv_("N", &M, &M, &a, &elemMat[cind*totDim*totDim], &M, &y[cind*totDim], &one, &b, z, &one));
    if (mesh->printFEM > 1) {
      CHKERRQ(DMPrintCellMatrix(c, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
      CHKERRQ(DMPrintCellVector(c, "Y",  totDim, &y[cind*totDim]));
      CHKERRQ(DMPrintCellVector(c, "Z",  totDim, z));
    }
    CHKERRQ(DMPlexVecSetClosure(dm, section, Z, cell, z, ADD_VALUES));
  }
  CHKERRQ(PetscFree6(u,u_t,elemMat,elemMatD,y,z));
  if (mesh->printFEM) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)Z), "Z:\n"));
    CHKERRQ(VecView(Z, NULL));
  }
  CHKERRQ(PetscFree(a));
  CHKERRQ(ISDestroy(&cellIS));
  CHKERRQ(DMDestroy(&plexAux));
  CHKERRQ(DMDestroy(&plex));
  CHKERRQ(PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}
