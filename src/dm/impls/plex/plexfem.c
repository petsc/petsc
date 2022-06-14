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
  PetscCall(PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex));
  if (isPlex) {
    *plex = dm;
    PetscCall(PetscObjectReference((PetscObject) dm));
  } else {
    PetscCall(PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex));
    if (!*plex) {
      PetscCall(DMConvert(dm, DMPLEX, plex));
      PetscCall(PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex));
      if (copy) {
        DMSubDomainHookLink link;

        PetscCall(DMCopyAuxiliaryVec(dm, *plex));
        /* Run the subdomain hook (this will copy the DMSNES/DMTS) */
        for (link = dm->subdomainhook; link; link = link->next) {
          if (link->ddhook) PetscCall((*link->ddhook)(dm, *plex, link->ctx));
        }
      }
    } else {
      PetscCall(PetscObjectReference((PetscObject) *plex));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscContainerUserDestroy_PetscFEGeom (void *ctx)
{
  PetscFEGeom *geom = (PetscFEGeom *) ctx;

  PetscFunctionBegin;
  PetscCall(PetscFEGeomDestroy(&geom));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetFEGeom(DMField coordField, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)quad,&id));
  PetscCall(PetscSNPrintf(composeStr, 32, "DMPlexGetFEGeom_%" PetscInt64_FMT "\n", id));
  PetscCall(PetscObjectQuery((PetscObject) pointIS, composeStr, (PetscObject *) &container));
  if (container) {
    PetscCall(PetscContainerGetPointer(container, (void **) geom));
  } else {
    PetscCall(DMFieldCreateFEGeom(coordField, pointIS, quad, faceData, geom));
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF,&container));
    PetscCall(PetscContainerSetPointer(container, (void *) *geom));
    PetscCall(PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom));
    PetscCall(PetscObjectCompose((PetscObject) pointIS, composeStr, (PetscObject) container));
    PetscCall(PetscContainerDestroy(&container));
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

.seealso: `DMPlexSetScale()`, `PetscUnit`
@*/
PetscErrorCode DMPlexGetScale(DM dm, PetscUnit unit, PetscReal *scale)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidRealPointer(scale, 3);
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

.seealso: `DMPlexGetScale()`, `PetscUnit`
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
  PetscCheck(dim == dim2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input dimension %" PetscInt_FMT " does not match context dimension %" PetscInt_FMT, dim, dim2);
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

.seealso: `MatNullSpaceCreate()`, `PCGAMG`
@*/
PetscErrorCode DMPlexCreateRigidBody(DM dm, PetscInt field, MatNullSpace *sp)
{
  PetscErrorCode (**func)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *);
  MPI_Comm          comm;
  Vec               mode[6];
  PetscSection      section, globalSection;
  PetscInt          dim, dimEmbed, Nf, n, m, mmin, d, i, j;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dimEmbed));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCheck(!Nf || !(field < 0 || field >= Nf),comm, PETSC_ERR_ARG_OUTOFRANGE, "Field %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", field, Nf);
  if (dim == 1 && Nf < 2) {
    PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, sp));
    PetscFunctionReturn(0);
  }
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &globalSection));
  PetscCall(PetscSectionGetConstrainedStorageSize(globalSection, &n));
  PetscCall(PetscCalloc1(Nf, &func));
  m    = (dim*(dim+1))/2;
  PetscCall(VecCreate(comm, &mode[0]));
  PetscCall(VecSetType(mode[0], dm->vectype));
  PetscCall(VecSetSizes(mode[0], n, PETSC_DETERMINE));
  PetscCall(VecSetUp(mode[0]));
  PetscCall(VecGetSize(mode[0], &n));
  mmin = PetscMin(m, n);
  func[field] = DMPlexProjectRigidBody_Private;
  for (i = 1; i < m; ++i) PetscCall(VecDuplicate(mode[0], &mode[i]));
  for (d = 0; d < m; d++) {
    PetscInt ctx[2];
    void    *voidctx = (void *) (&ctx[0]);

    ctx[0] = dimEmbed;
    ctx[1] = d;
    PetscCall(DMProjectFunction(dm, 0.0, func, &voidctx, INSERT_VALUES, mode[d]));
  }
  /* Orthonormalize system */
  for (i = 0; i < mmin; ++i) {
    PetscScalar dots[6];

    PetscCall(VecNormalize(mode[i], NULL));
    PetscCall(VecMDot(mode[i], mmin-i-1, mode+i+1, dots+i+1));
    for (j = i+1; j < mmin; ++j) {
      dots[j] *= -1.0;
      PetscCall(VecAXPY(mode[j], dots[j], mode[i]));
    }
  }
  PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, mmin, mode, sp));
  for (i = 0; i < m; ++i) PetscCall(VecDestroy(&mode[i]));
  PetscCall(PetscFree(func));
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

.seealso: `MatNullSpaceCreate()`
@*/
PetscErrorCode DMPlexCreateRigidBodies(DM dm, PetscInt nb, DMLabel label, const PetscInt nids[], const PetscInt ids[], MatNullSpace *sp)
{
  MPI_Comm       comm;
  PetscSection   section, globalSection;
  Vec           *mode;
  PetscScalar   *dots;
  PetscInt       dim, dimEmbed, n, m, b, d, i, j, off;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dimEmbed));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &globalSection));
  PetscCall(PetscSectionGetConstrainedStorageSize(globalSection, &n));
  m    = nb * (dim*(dim+1))/2;
  PetscCall(PetscMalloc2(m, &mode, m, &dots));
  PetscCall(VecCreate(comm, &mode[0]));
  PetscCall(VecSetSizes(mode[0], n, PETSC_DETERMINE));
  PetscCall(VecSetUp(mode[0]));
  for (i = 1; i < m; ++i) PetscCall(VecDuplicate(mode[0], &mode[i]));
  for (b = 0, off = 0; b < nb; ++b) {
    for (d = 0; d < m/nb; ++d) {
      PetscInt         ctx[2];
      PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *) = DMPlexProjectRigidBody_Private;
      void            *voidctx = (void *) (&ctx[0]);

      ctx[0] = dimEmbed;
      ctx[1] = d;
      PetscCall(DMProjectFunctionLabel(dm, 0.0, label, nids[b], &ids[off], 0, NULL, &func, &voidctx, INSERT_VALUES, mode[d]));
      off   += nids[b];
    }
  }
  /* Orthonormalize system */
  for (i = 0; i < m; ++i) {
    PetscScalar dots[6];

    PetscCall(VecNormalize(mode[i], NULL));
    PetscCall(VecMDot(mode[i], m-i-1, mode+i+1, dots+i+1));
    for (j = i+1; j < m; ++j) {
      dots[j] *= -1.0;
      PetscCall(VecAXPY(mode[j], dots[j], mode[i]));
    }
  }
  PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, m, mode, sp));
  for (i = 0; i< m; ++i) PetscCall(VecDestroy(&mode[i]));
  PetscCall(PetscFree2(mode, dots));
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

.seealso: `DMPlexGetMaxProjectionHeight()`, `DMProjectFunctionLocal()`, `DMProjectFunctionLabelLocal()`
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

.seealso: `DMPlexSetMaxProjectionHeight()`, `DMProjectFunctionLocal()`, `DMProjectFunctionLabelLocal()`
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
  PetscCall(PetscMalloc2(PetscSqr(dim), &rc->R, PetscSqr(dim), &rc->RT));
  switch (dim) {
  case 2:
    c1 = PetscCosReal(rc->alpha);s1 = PetscSinReal(rc->alpha);
    rc->R[0] =  c1;rc->R[1] = s1;
    rc->R[2] = -s1;rc->R[3] = c1;
    PetscCall(PetscArraycpy(rc->RT, rc->R, PetscSqr(dim)));
    DMPlex_Transpose2D_Internal(rc->RT);
    break;
  case 3:
    c1 = PetscCosReal(rc->alpha);s1 = PetscSinReal(rc->alpha);
    c2 = PetscCosReal(rc->beta); s2 = PetscSinReal(rc->beta);
    c3 = PetscCosReal(rc->gamma);s3 = PetscSinReal(rc->gamma);
    rc->R[0] =  c1*c3 - c2*s1*s3;rc->R[1] =  c3*s1    + c1*c2*s3;rc->R[2] = s2*s3;
    rc->R[3] = -c1*s3 - c2*c3*s1;rc->R[4] =  c1*c2*c3 - s1*s3;   rc->R[5] = c3*s2;
    rc->R[6] =  s1*s2;           rc->R[7] = -c1*s2;              rc->R[8] = c2;
    PetscCall(PetscArraycpy(rc->RT, rc->R, PetscSqr(dim)));
    DMPlex_Transpose3D_Internal(rc->RT);
    break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported", dim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransformDestroy_Rotation_Internal(DM dm, void *ctx)
{
  RotCtx        *rc = (RotCtx *) ctx;

  PetscFunctionBegin;
  PetscCall(PetscFree2(rc->R, rc->RT));
  PetscCall(PetscFree(rc));
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

      PetscCall(DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, yt, zt, ctx));
      z[0] = PetscRealPart(zt[0]); z[1] = PetscRealPart(zt[1]);
    }
    break;
    case 3:
    {
      PetscScalar yt[3] = {y[0], y[1], y[2]}, zt[3] = {0.0,0.0,0.0};

      PetscCall(DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, yt, zt, ctx));
      z[0] = PetscRealPart(zt[0]); z[1] = PetscRealPart(zt[1]); z[2] = PetscRealPart(zt[2]);
    }
    break;
  }
  #else
  PetscCall(DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, y, z, ctx));
  #endif
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexBasisTransformApply_Internal(DM dm, const PetscReal x[], PetscBool l2g, PetscInt dim, const PetscScalar *y, PetscScalar *z, void *ctx)
{
  const PetscScalar *A;

  PetscFunctionBeginHot;
  PetscCall((*dm->transformGetMatrix)(dm, x, l2g, &A, ctx));
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
  PetscCall(DMGetLocalSection(tdm, &ts));
  PetscCall(PetscSectionGetFieldDof(ts, p, f, &dof));
  PetscCall(VecGetArrayRead(tv, &ta));
  PetscCall(DMPlexPointLocalFieldRead(tdm, p, f, ta, &tva));
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
  PetscCall(VecRestoreArrayRead(tv, &ta));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransformFieldTensor_Internal(DM dm, DM tdm, Vec tv, PetscInt pf, PetscInt f, PetscInt pg, PetscInt g, PetscBool l2g, PetscInt lda, PetscScalar *a)
{
  PetscSection       s, ts;
  const PetscScalar *ta, *tvaf, *tvag;
  PetscInt           fdof, gdof, fpdof, gpdof;

  PetscFunctionBeginHot;
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMGetLocalSection(tdm, &ts));
  PetscCall(PetscSectionGetFieldDof(s, pf, f, &fpdof));
  PetscCall(PetscSectionGetFieldDof(s, pg, g, &gpdof));
  PetscCall(PetscSectionGetFieldDof(ts, pf, f, &fdof));
  PetscCall(PetscSectionGetFieldDof(ts, pg, g, &gdof));
  PetscCall(VecGetArrayRead(tv, &ta));
  PetscCall(DMPlexPointLocalFieldRead(tdm, pf, f, ta, &tvaf));
  PetscCall(DMPlexPointLocalFieldRead(tdm, pg, g, ta, &tvag));
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
  PetscCall(VecRestoreArrayRead(tv, &ta));
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
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscSectionGetNumFields(s, &Nf));
  PetscCall(DMPlexGetCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp));
  for (f = 0; f < Nf; ++f) {
    for (cp = 0; cp < Np*2; cp += 2) {
      PetscCall(PetscSectionGetFieldDof(s, points[cp], f, &dof));
      if (!dof) continue;
      if (fieldActive[f]) PetscCall(DMPlexBasisTransformField_Internal(dm, tdm, tv, points[cp], f, l2g, &a[d]));
      d += dof;
    }
  }
  PetscCall(DMPlexRestoreCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp));
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
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscSectionGetNumFields(s, &Nf));
  PetscCall(DMPlexGetCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp));
  for (f = 0, r = 0; f < Nf; ++f) {
    for (cpf = 0; cpf < Np*2; cpf += 2) {
      PetscCall(PetscSectionGetFieldDof(s, points[cpf], f, &fdof));
      for (g = 0, c = 0; g < Nf; ++g) {
        for (cpg = 0; cpg < Np*2; cpg += 2) {
          PetscCall(PetscSectionGetFieldDof(s, points[cpg], g, &gdof));
          PetscCall(DMPlexBasisTransformFieldTensor_Internal(dm, tdm, tv, points[cpf], f, points[cpg], g, l2g, lda, &a[r*lda+c]));
          c += gdof;
        }
      }
      PetscCheck(c == lda,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of columns %" PetscInt_FMT " should be %" PetscInt_FMT, c, lda);
      r += fdof;
    }
  }
  PetscCheck(r == lda,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of rows %" PetscInt_FMT " should be %" PetscInt_FMT, c, lda);
  PetscCall(DMPlexRestoreCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp));
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
  PetscCall(DMGetBasisTransformDM_Internal(dm, &tdm));
  PetscCall(DMGetBasisTransformVec_Internal(dm, &tv));
  PetscCall(DMGetLocalSection(tdm, &ts));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  PetscCall(PetscSectionGetNumFields(s, &Nf));
  PetscCall(VecGetArray(lv, &a));
  PetscCall(VecGetArrayRead(tv, &ta));
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < Nf; ++f) {
      PetscCall(DMPlexPointLocalFieldRef(dm, p, f, a, &va));
      PetscCall(DMPlexBasisTransformField_Internal(dm, tdm, tv, p, f, l2g, va));
    }
  }
  PetscCall(VecRestoreArray(lv, &a));
  PetscCall(VecRestoreArrayRead(tv, &ta));
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

.seealso: `DMPlexLocalToGlobalBasis()`, `DMGetLocalSection()`, `DMPlexCreateBasisRotation()`
@*/
PetscErrorCode DMPlexGlobalToLocalBasis(DM dm, Vec lv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lv, VEC_CLASSID, 2);
  PetscCall(DMPlexBasisTransform_Internal(dm, lv, PETSC_FALSE));
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

.seealso: `DMPlexGlobalToLocalBasis()`, `DMGetLocalSection()`, `DMPlexCreateBasisRotation()`
@*/
PetscErrorCode DMPlexLocalToGlobalBasis(DM dm, Vec lv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lv, VEC_CLASSID, 2);
  PetscCall(DMPlexBasisTransform_Internal(dm, lv, PETSC_TRUE));
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

.seealso: `DMPlexGlobalToLocalBasis()`, `DMPlexLocalToGlobalBasis()`
@*/
PetscErrorCode DMPlexCreateBasisRotation(DM dm, PetscReal alpha, PetscReal beta, PetscReal gamma)
{
  RotCtx        *rc;
  PetscInt       cdim;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(PetscMalloc1(1, &rc));
  dm->transformCtx       = rc;
  dm->transformSetUp     = DMPlexBasisTransformSetUp_Rotation_Internal;
  dm->transformDestroy   = DMPlexBasisTransformDestroy_Rotation_Internal;
  dm->transformGetMatrix = DMPlexBasisTransformGetMatrix_Rotation_Internal;
  rc->dim   = cdim;
  rc->alpha = alpha;
  rc->beta  = beta;
  rc->gamma = gamma;
  PetscCall((*dm->transformSetUp)(dm, dm->transformCtx));
  PetscCall(DMConstructBasisTransform_Internal(dm));
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

.seealso: `DMPlexInsertBoundaryValuesEssentialField()`, `DMPlexInsertBoundaryValuesEssentialBdField()`, `DMAddBoundary()`
@*/
PetscErrorCode DMPlexInsertBoundaryValuesEssential(DM dm, PetscReal time, PetscInt field, PetscInt Nc, const PetscInt comps[], DMLabel label, PetscInt numids, const PetscInt ids[], PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void *ctx, Vec locX)
{
  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void            **ctxs;
  PetscInt          numFields;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(dm, &numFields));
  PetscCall(PetscCalloc2(numFields,&funcs,numFields,&ctxs));
  funcs[field] = func;
  ctxs[field]  = ctx;
  PetscCall(DMProjectFunctionLabelLocal(dm, time, label, numids, ids, Nc, comps, funcs, ctxs, INSERT_BC_VALUES, locX));
  PetscCall(PetscFree2(funcs,ctxs));
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

.seealso: `DMPlexInsertBoundaryValuesEssential()`, `DMPlexInsertBoundaryValuesEssentialBdField()`, `DMAddBoundary()`
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
  PetscCall(DMGetNumFields(dm, &numFields));
  PetscCall(PetscCalloc2(numFields,&funcs,numFields,&ctxs));
  funcs[field] = func;
  ctxs[field]  = ctx;
  PetscCall(DMProjectFieldLabelLocal(dm, time, label, numids, ids, Nc, comps, locU, funcs, INSERT_BC_VALUES, locX));
  PetscCall(PetscFree2(funcs,ctxs));
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

.seealso: `DMProjectBdFieldLabelLocal()`, `DMPlexInsertBoundaryValuesEssential()`, `DMPlexInsertBoundaryValuesEssentialField()`, `DMAddBoundary()`
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
  PetscCall(DMGetNumFields(dm, &numFields));
  PetscCall(PetscCalloc2(numFields,&funcs,numFields,&ctxs));
  funcs[field] = func;
  ctxs[field]  = ctx;
  PetscCall(DMProjectBdFieldLabelLocal(dm, time, label, numids, ids, Nc, comps, locU, funcs, INSERT_BC_VALUES, locX));
  PetscCall(PetscFree2(funcs,ctxs));
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

.seealso: `DMPlexInsertBoundaryValuesEssential()`, `DMPlexInsertBoundaryValuesEssentialField()`, `DMAddBoundary()`
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
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL));
  nleaves = PetscMax(0, nleaves);
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(VecGetDM(faceGeometry, &dmFace));
  PetscCall(VecGetArrayRead(faceGeometry, &facegeom));
  if (cellGeometry) {
    PetscCall(VecGetDM(cellGeometry, &dmCell));
    PetscCall(VecGetArrayRead(cellGeometry, &cellgeom));
  }
  if (Grad) {
    PetscFV fv;

    PetscCall(PetscDSGetDiscretization(prob, field, (PetscObject *) &fv));
    PetscCall(VecGetDM(Grad, &dmGrad));
    PetscCall(VecGetArrayRead(Grad, &grad));
    PetscCall(PetscFVGetNumComponents(fv, &pdim));
    PetscCall(DMGetWorkArray(dm, pdim, MPIU_SCALAR, &fx));
  }
  PetscCall(VecGetArray(locX, &x));
  for (i = 0; i < numids; ++i) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    PetscCall(DMLabelGetStratumIS(label, ids[i], &faceIS));
    if (!faceIS) continue; /* No points with that id on this process */
    PetscCall(ISGetLocalSize(faceIS, &numFaces));
    PetscCall(ISGetIndices(faceIS, &faces));
    for (f = 0; f < numFaces; ++f) {
      const PetscInt         face = faces[f], *cells;
      PetscFVFaceGeom        *fg;

      if ((face < fStart) || (face >= fEnd)) continue; /* Refinement adds non-faces to labels */
      PetscCall(PetscFindInt(face, nleaves, (PetscInt *) leaves, &loc));
      if (loc >= 0) continue;
      PetscCall(DMPlexPointLocalRead(dmFace, face, facegeom, &fg));
      PetscCall(DMPlexGetSupport(dm, face, &cells));
      if (Grad) {
        PetscFVCellGeom       *cg;
        PetscScalar           *cx, *cgrad;
        PetscScalar           *xG;
        PetscReal              dx[3];
        PetscInt               d;

        PetscCall(DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cg));
        PetscCall(DMPlexPointLocalRead(dm, cells[0], x, &cx));
        PetscCall(DMPlexPointLocalRead(dmGrad, cells[0], grad, &cgrad));
        PetscCall(DMPlexPointLocalFieldRef(dm, cells[1], field, x, &xG));
        DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, fg->centroid, dx);
        for (d = 0; d < pdim; ++d) fx[d] = cx[d] + DMPlex_DotD_Internal(dim, &cgrad[d*dim], dx);
        PetscCall((*func)(time, fg->centroid, fg->normal, fx, xG, ctx));
      } else {
        PetscScalar *xI;
        PetscScalar *xG;

        PetscCall(DMPlexPointLocalRead(dm, cells[0], x, &xI));
        PetscCall(DMPlexPointLocalFieldRef(dm, cells[1], field, x, &xG));
        ierru = (*func)(time, fg->centroid, fg->normal, xI, xG, ctx);
        if (ierru) {
          PetscCall(ISRestoreIndices(faceIS, &faces));
          PetscCall(ISDestroy(&faceIS));
          goto cleanup;
        }
      }
    }
    PetscCall(ISRestoreIndices(faceIS, &faces));
    PetscCall(ISDestroy(&faceIS));
  }
  cleanup:
  PetscCall(VecRestoreArray(locX, &x));
  if (Grad) {
    PetscCall(DMRestoreWorkArray(dm, pdim, MPIU_SCALAR, &fx));
    PetscCall(VecRestoreArrayRead(Grad, &grad));
  }
  if (cellGeometry) PetscCall(VecRestoreArrayRead(cellGeometry, &cellgeom));
  PetscCall(VecRestoreArrayRead(faceGeometry, &facegeom));
  PetscCall(ierru);
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
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSGetNumBoundary(prob, &numBd));
  PetscCall(PetscObjectQuery((PetscObject) locX, "__Vec_bc_zero__", &isZero));
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

    PetscCall(PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, &bvfunc, NULL, &ctx));
    if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
    PetscCall(DMGetField(dm, field, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      switch (type) {
        /* for FEM, there is no insertion to be done for non-essential boundary conditions */
      case DM_BC_ESSENTIAL:
        {
          PetscSimplePointFunc func = (PetscSimplePointFunc) bvfunc;

          if (isZero) func = zero;
          PetscCall(DMPlexLabelAddCells(dm,label));
          PetscCall(DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func, ctx, locX));
          PetscCall(DMPlexLabelClearCells(dm,label));
        }
        break;
      case DM_BC_ESSENTIAL_FIELD:
        {
          PetscPointFunc func = (PetscPointFunc) bvfunc;

          PetscCall(DMPlexLabelAddCells(dm,label));
          PetscCall(DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func, ctx, locX));
          PetscCall(DMPlexLabelClearCells(dm,label));
        }
        break;
      default: break;
      }
    } else if (id == PETSCFV_CLASSID) {
      {
        PetscErrorCode (*func)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*) = (PetscErrorCode (*)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*)) bvfunc;

        if (!faceGeomFVM) continue;
        PetscCall(DMPlexInsertBoundaryValuesRiemann(dm, time, faceGeomFVM, cellGeomFVM, gradFVM, field, Nc, comps, label, numids, ids, func, ctx, locX));
      }
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
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
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSGetNumBoundary(prob, &numBd));
  PetscCall(PetscObjectQuery((PetscObject) locX, "__Vec_bc_zero__", &isZero));
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

    PetscCall(PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, NULL, &bvfunc, &ctx));
    if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
    PetscCall(DMGetField(dm, field, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      switch (type) {
        /* for FEM, there is no insertion to be done for non-essential boundary conditions */
      case DM_BC_ESSENTIAL:
        {
          PetscSimplePointFunc func_t = (PetscSimplePointFunc) bvfunc;

          if (isZero) func_t = zero;
          PetscCall(DMPlexLabelAddCells(dm,label));
          PetscCall(DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func_t, ctx, locX));
          PetscCall(DMPlexLabelClearCells(dm,label));
        }
        break;
      case DM_BC_ESSENTIAL_FIELD:
        {
          PetscPointFunc func_t = (PetscPointFunc) bvfunc;

          PetscCall(DMPlexLabelAddCells(dm,label));
          PetscCall(DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func_t, ctx, locX));
          PetscCall(DMPlexLabelClearCells(dm,label));
        }
        break;
      default: break;
      }
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
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

.seealso: `DMProjectFunctionLabelLocal()`, `DMAddBoundary()`
@*/
PetscErrorCode DMPlexInsertBoundaryValues(DM dm, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 3);
  if (faceGeomFVM) {PetscValidHeaderSpecific(faceGeomFVM, VEC_CLASSID, 5);}
  if (cellGeomFVM) {PetscValidHeaderSpecific(cellGeomFVM, VEC_CLASSID, 6);}
  if (gradFVM)     {PetscValidHeaderSpecific(gradFVM, VEC_CLASSID, 7);}
  PetscTryMethod(dm,"DMPlexInsertBoundaryValues_C",(DM,PetscBool,Vec,PetscReal,Vec,Vec,Vec),(dm,insertEssential,locX,time,faceGeomFVM,cellGeomFVM,gradFVM));
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

.seealso: `DMProjectFunctionLabelLocal()`
@*/
PetscErrorCode DMPlexInsertTimeDerivativeBoundaryValues(DM dm, PetscBool insertEssential, Vec locX_t, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (locX_t)      {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 3);}
  if (faceGeomFVM) {PetscValidHeaderSpecific(faceGeomFVM, VEC_CLASSID, 5);}
  if (cellGeomFVM) {PetscValidHeaderSpecific(cellGeomFVM, VEC_CLASSID, 6);}
  if (gradFVM)     {PetscValidHeaderSpecific(gradFVM, VEC_CLASSID, 7);}
  PetscTryMethod(dm,"DMPlexInsertTimeDerviativeBoundaryValues_C",(DM,PetscBool,Vec,PetscReal,Vec,Vec,Vec),(dm,insertEssential,locX_t,time,faceGeomFVM,cellGeomFVM,gradFVM));
  PetscFunctionReturn(0);
}

PetscErrorCode DMComputeL2Diff_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  Vec              localX;

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, localX, time, NULL, NULL, NULL));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX));
  PetscCall(DMPlexComputeL2DiffLocal(dm, time, funcs, ctxs, localX, diff));
  PetscCall(DMRestoreLocalVector(dm, &localX));
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

.seealso: `DMProjectFunction()`, `DMComputeL2FieldDiff()`, `DMComputeL2GradientDiff()`
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

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &coordDim));
  fegeom.dimEmbed = coordDim;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &numFields));
  PetscCall(DMGetBasisTransformDM_Internal(dm, &tdm));
  PetscCall(DMGetBasisTransformVec_Internal(dm, &tv));
  PetscCall(DMHasBasisTransform(dm, &transform));
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    PetscCall(DMGetField(dm, field, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      PetscCall(PetscFEGetQuadrature(fe, &quad));
      PetscCall(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      PetscCall(PetscFVGetQuadrature(fv, &quad));
      PetscCall(PetscFVGetNumComponents(fv, &Nc));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
    numComponents += Nc;
  }
  PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights));
  PetscCheck(!(qNc != 1) || !(qNc != numComponents),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %" PetscInt_FMT " != %" PetscInt_FMT " field components", qNc, numComponents);
  PetscCall(PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;
    PetscInt     qc = 0;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
    PetscCall(DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x));

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      PetscCall(DMGetField(dm, field, NULL, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID)      {PetscCall(PetscFEGetNumComponents((PetscFE) obj, &Nc));PetscCall(PetscFEGetDimension((PetscFE) obj, &Nb));}
      else if (id == PETSCFV_CLASSID) {PetscCall(PetscFVGetNumComponents((PetscFV) obj, &Nc));Nb = 1;}
      else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
      if (debug) {
        char title[1024];
        PetscCall(PetscSNPrintf(title, 1023, "Solution for Field %" PetscInt_FMT, field));
        PetscCall(DMPrintCellVector(c, title, Nb, &x[fieldOffset]));
      }
      for (q = 0; q < Nq; ++q) {
        PetscFEGeom    qgeom;
        PetscErrorCode ierr;

        qgeom.dimEmbed = fegeom.dimEmbed;
        qgeom.J        = &fegeom.J[q*coordDim*coordDim];
        qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
        qgeom.detJ     = &fegeom.detJ[q];
        PetscCheck(fegeom.detJ[q] > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT ", point %" PetscInt_FMT, (double)fegeom.detJ[q], c, q);
        if (transform) {
          gcoords = &coords[coordDim*Nq];
          PetscCall(DMPlexBasisTransformApplyReal_Internal(dm, &coords[coordDim*q], PETSC_TRUE, coordDim, &coords[coordDim*q], gcoords, dm->transformCtx));
        } else {
          gcoords = &coords[coordDim*q];
        }
        PetscCall(PetscArrayzero(funcVal,Nc));
        ierr = (*funcs[field])(coordDim, time, gcoords, Nc, funcVal, ctx);
        if (ierr) {
          PetscCall(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
          PetscCall(DMRestoreLocalVector(dm, &localX));
          PetscCall(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
        }
        if (transform) PetscCall(DMPlexBasisTransformApply_Internal(dm, &coords[coordDim*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx));
        if (id == PETSCFE_CLASSID)      PetscCall(PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], &qgeom, q, interpolant));
        else if (id == PETSCFV_CLASSID) PetscCall(PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant));
        else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
        for (fc = 0; fc < Nc; ++fc) {
          const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
          if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "    elem %" PetscInt_FMT " field %" PetscInt_FMT ",%" PetscInt_FMT " point %g %g %g diff %g (%g, %g)\n", c, field, fc, (double)(coordDim > 0 ? coords[coordDim*q] : 0.), (double)(coordDim > 1 ? coords[coordDim*q+1] : 0.),(double)(coordDim > 2 ? coords[coordDim*q+2] : 0.), (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q]), (double)PetscRealPart(interpolant[fc]), (double)PetscRealPart(funcVal[fc])));
          elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
        }
      }
      fieldOffset += Nb;
      qc += Nc;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
    if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  elem %" PetscInt_FMT " diff %g\n", c, (double)elemDiff));
    localDiff += elemDiff;
  }
  PetscCall(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
  PetscCall(MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
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

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &coordDim));
  fegeom.dimEmbed = coordDim;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &numFields));
  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX));
  PetscCall(DMGetBasisTransformDM_Internal(dm, &tdm));
  PetscCall(DMGetBasisTransformVec_Internal(dm, &tv));
  PetscCall(DMHasBasisTransform(dm, &transform));
  for (field = 0; field < numFields; ++field) {
    PetscFE  fe;
    PetscInt Nc;

    PetscCall(DMGetField(dm, field, NULL, (PetscObject *) &fe));
    PetscCall(PetscFEGetQuadrature(fe, &quad));
    PetscCall(PetscFEGetNumComponents(fe, &Nc));
    numComponents += Nc;
  }
  PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights));
  PetscCheck(!(qNc != 1) || !(qNc != numComponents),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %" PetscInt_FMT " != %" PetscInt_FMT " field components", qNc, numComponents);
  /* PetscCall(DMProjectFunctionLocal(dm, fe, funcs, INSERT_BC_VALUES, localX)); */
  PetscCall(PetscMalloc6(numComponents,&funcVal,coordDim*Nq,&coords,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ,numComponents*coordDim,&interpolant,Nq,&fegeom.detJ));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;
    PetscInt     qc = 0;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
    PetscCall(DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x));

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscFE          fe;
      void * const     ctx = ctxs ? ctxs[field] : NULL;
      PetscInt         Nb, Nc, q, fc;

      PetscCall(DMGetField(dm, field, NULL, (PetscObject *) &fe));
      PetscCall(PetscFEGetDimension(fe, &Nb));
      PetscCall(PetscFEGetNumComponents(fe, &Nc));
      if (debug) {
        char title[1024];
        PetscCall(PetscSNPrintf(title, 1023, "Solution for Field %" PetscInt_FMT, field));
        PetscCall(DMPrintCellVector(c, title, Nb, &x[fieldOffset]));
      }
      for (q = 0; q < Nq; ++q) {
        PetscFEGeom    qgeom;
        PetscErrorCode ierr;

        qgeom.dimEmbed = fegeom.dimEmbed;
        qgeom.J        = &fegeom.J[q*coordDim*coordDim];
        qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
        qgeom.detJ     = &fegeom.detJ[q];
        PetscCheck(fegeom.detJ[q] > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT ", quadrature points %" PetscInt_FMT, (double)fegeom.detJ[q], c, q);
        if (transform) {
          gcoords = &coords[coordDim*Nq];
          PetscCall(DMPlexBasisTransformApplyReal_Internal(dm, &coords[coordDim*q], PETSC_TRUE, coordDim, &coords[coordDim*q], gcoords, dm->transformCtx));
        } else {
          gcoords = &coords[coordDim*q];
        }
        PetscCall(PetscArrayzero(funcVal,Nc));
        ierr = (*funcs[field])(coordDim, time, gcoords, n, Nc, funcVal, ctx);
        if (ierr) {
          PetscCall(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
          PetscCall(DMRestoreLocalVector(dm, &localX));
          PetscCall(PetscFree6(funcVal,coords,fegeom.J,fegeom.invJ,interpolant,fegeom.detJ));
        }
        if (transform) PetscCall(DMPlexBasisTransformApply_Internal(dm, &coords[coordDim*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx));
        PetscCall(PetscFEInterpolateGradient_Static(fe, 1, &x[fieldOffset], &qgeom, q, interpolant));
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
          if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "    elem %" PetscInt_FMT " fieldDer %" PetscInt_FMT ",%" PetscInt_FMT " diff %g\n", c, field, fc, (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q])));
          elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
        }
      }
      fieldOffset += Nb;
      qc          += Nc;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
    if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  elem %" PetscInt_FMT " diff %g\n", c, (double)elemDiff));
    localDiff += elemDiff;
  }
  PetscCall(PetscFree6(funcVal,coords,fegeom.J,fegeom.invJ,interpolant,fegeom.detJ));
  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscCall(MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
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

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(DMGetBasisTransformDM_Internal(dm, &tdm));
  PetscCall(DMGetBasisTransformVec_Internal(dm, &tv));
  PetscCall(DMHasBasisTransform(dm, &transform));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMLabelGetNumValues(depthLabel, &depth));

  PetscCall(VecSet(localX, 0.0));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX));
  PetscCall(DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX));
  PetscCall(DMGetNumDS(dm, &Nds));
  PetscCall(PetscCalloc1(Nf, &localDiff));
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

    PetscCall(DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds));
    PetscCall(ISGetIndices(fieldIS, &fields));
    PetscCall(PetscDSIsCohesive(ds, &isCohesive));
    PetscCall(PetscDSGetNumFields(ds, &dsNf));
    PetscCall(PetscDSGetTotalComponents(ds, &totNc));
    PetscCall(PetscDSGetQuadrature(ds, &quad));
    PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
    PetscCheck(!(qNc != 1) || !(qNc != totNc),PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Quadrature components %" PetscInt_FMT " != %" PetscInt_FMT " field components", qNc, totNc);
    PetscCall(PetscCalloc6(totNc, &funcVal, totNc, &interpolant, dE*(Nq+1), &coords,Nq, &fegeom.detJ, dE*dE*Nq, &fegeom.J, dE*dE*Nq, &fegeom.invJ));
    if (!label) {
      PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
    } else {
      PetscCall(DMLabelGetStratumIS(label, 1, &pointIS));
      PetscCall(ISGetLocalSize(pointIS, &cEnd));
      PetscCall(ISGetIndices(pointIS, &points));
    }
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt cell  = points ? points[c] : c;
      PetscScalar    *x    = NULL;
      const PetscInt *cone;
      PetscInt        qc   = 0, fOff = 0, dep;

      PetscCall(DMLabelGetValue(depthLabel, cell, &dep));
      if (dep != depth-1) continue;
      if (isCohesive) {
        PetscCall(DMPlexGetCone(dm, cell, &cone));
        PetscCall(DMPlexComputeCellGeometryFEM(dm, cone[0], quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      } else {
        PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      }
      PetscCall(DMPlexVecGetClosure(dm, NULL, localX, cell, NULL, &x));
      for (f = 0; f < dsNf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        void * const ctx = ctxs ? ctxs[fields[f]] : NULL;
        PetscInt     Nb, Nc, q, fc;
        PetscReal    elemDiff = 0.0;
        PetscBool    cohesive;

        PetscCall(PetscDSGetCohesive(ds, f, &cohesive));
        if (isCohesive && !cohesive) continue;
        PetscCall(PetscDSGetDiscretization(ds, f, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID)      {PetscCall(PetscFEGetNumComponents((PetscFE) obj, &Nc));PetscCall(PetscFEGetDimension((PetscFE) obj, &Nb));}
        else if (id == PETSCFV_CLASSID) {PetscCall(PetscFVGetNumComponents((PetscFV) obj, &Nc));Nb = 1;}
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, fields[f]);
        if (debug) {
          char title[1024];
          PetscCall(PetscSNPrintf(title, 1023, "Solution for Field %" PetscInt_FMT, fields[f]));
          PetscCall(DMPrintCellVector(cell, title, Nb, &x[fOff]));
        }
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom    qgeom;
          PetscErrorCode ierr;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*dE*dE];
          qgeom.invJ     = &fegeom.invJ[q*dE*dE];
          qgeom.detJ     = &fegeom.detJ[q];
          PetscCheck(fegeom.detJ[q] > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %" PetscInt_FMT ", quadrature point %" PetscInt_FMT, (double)fegeom.detJ[q], cell, q);
          if (transform) {
            gcoords = &coords[dE*Nq];
            PetscCall(DMPlexBasisTransformApplyReal_Internal(dm, &coords[dE*q], PETSC_TRUE, dE, &coords[dE*q], gcoords, dm->transformCtx));
          } else {
            gcoords = &coords[dE*q];
          }
          for (fc = 0; fc < Nc; ++fc) funcVal[fc] = 0.;
          ierr = (*funcs[fields[f]])(dE, time, gcoords, Nc, funcVal, ctx);
          if (ierr) {
            PetscCall(DMPlexVecRestoreClosure(dm, NULL, localX, cell, NULL, &x));
            PetscCall(DMRestoreLocalVector(dm, &localX));
            PetscCall(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
          }
          if (transform) PetscCall(DMPlexBasisTransformApply_Internal(dm, &coords[dE*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx));
          /* Call once for each face, except for lagrange field */
          if (id == PETSCFE_CLASSID)      PetscCall(PetscFEInterpolate_Static((PetscFE) obj, &x[fOff], &qgeom, q, interpolant));
          else if (id == PETSCFV_CLASSID) PetscCall(PetscFVInterpolate_Static((PetscFV) obj, &x[fOff], q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, fields[f]);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
            if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "    cell %" PetscInt_FMT " field %" PetscInt_FMT ",%" PetscInt_FMT " point %g %g %g diff %g\n", cell, fields[f], fc, (double)(dE > 0 ? coords[dE*q] : 0.), (double)(dE > 1 ? coords[dE*q+1] : 0.),(double)(dE > 2 ? coords[dE*q+2] : 0.), (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q])));
            elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
          }
        }
        fOff += Nb;
        qc   += Nc;
        localDiff[fields[f]] += elemDiff;
        if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  cell %" PetscInt_FMT " field %" PetscInt_FMT " cum diff %g\n", cell, fields[f], (double)localDiff[fields[f]]));
      }
      PetscCall(DMPlexVecRestoreClosure(dm, NULL, localX, cell, NULL, &x));
    }
    if (label) {
      PetscCall(ISRestoreIndices(pointIS, &points));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISRestoreIndices(fieldIS, &fields));
    PetscCall(PetscFree6(funcVal, interpolant, coords, fegeom.detJ, fegeom.J, fegeom.invJ));
  }
  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscCall(MPIU_Allreduce(localDiff, diff, Nf, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
  PetscCall(PetscFree(localDiff));
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

.seealso: `DMProjectFunction()`, `DMComputeL2Diff()`, `DMPlexComputeL2FieldDiff()`, `DMComputeL2GradientDiff()`
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
  PetscCall(VecSet(D, 0.0));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &coordDim));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &numFields));
  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX));
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    PetscCall(DMGetField(dm, field, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      PetscCall(PetscFEGetQuadrature(fe, &quad));
      PetscCall(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      PetscCall(PetscFVGetQuadrature(fv, &quad));
      PetscCall(PetscFVGetNumComponents(fv, &Nc));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
    numComponents += Nc;
  }
  PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCheck(!(qNc != 1) || !(qNc != numComponents),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %" PetscInt_FMT " != %" PetscInt_FMT " field components", qNc, numComponents);
  PetscCall(PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscScalar  elemDiff = 0.0;
    PetscInt     qc = 0;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
    PetscCall(DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x));

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      PetscCall(DMGetField(dm, field, NULL, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID)      {PetscCall(PetscFEGetNumComponents((PetscFE) obj, &Nc));PetscCall(PetscFEGetDimension((PetscFE) obj, &Nb));}
      else if (id == PETSCFV_CLASSID) {PetscCall(PetscFVGetNumComponents((PetscFV) obj, &Nc));Nb = 1;}
      else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
      if (funcs[field]) {
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*coordDim*coordDim];
          qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
          qgeom.detJ     = &fegeom.detJ[q];
          PetscCheck(fegeom.detJ[q] > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT ", quadrature points %" PetscInt_FMT, (double)fegeom.detJ[q], c, q);
          PetscCall((*funcs[field])(coordDim, time, &coords[q*coordDim], Nc, funcVal, ctx));
#if defined(needs_fix_with_return_code_argument)
          if (ierr) {
            PetscCall(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
            PetscCall(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
            PetscCall(DMRestoreLocalVector(dm, &localX));
          }
#endif
          if (id == PETSCFE_CLASSID)      PetscCall(PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], &qgeom, q, interpolant));
          else if (id == PETSCFV_CLASSID) PetscCall(PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
            elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
          }
        }
      }
      fieldOffset += Nb;
      qc          += Nc;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x));
    PetscCall(VecSetValue(D, c - cStart, elemDiff, INSERT_VALUES));
  }
  PetscCall(PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscCall(VecSqrtAbs(D));
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

.seealso: `DMProjectFunction()`, `DMComputeL2Diff()`, `DMPlexComputeL2FieldDiff()`, `DMComputeL2GradientDiff()`
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
  PetscCall(PetscCitationsRegister(ClementCitation, &Clementcite));
  PetscCall(VecGetDM(locC, &dmc));
  PetscCall(VecSet(locC, 0.0));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  fegeom.dimEmbed = cdim;
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCheck(Nf > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fields is zero!");
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     fNc;

    PetscCall(DMGetField(dm, f, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      PetscCall(PetscFEGetQuadrature(fe, &quad));
      PetscCall(PetscFEGetNumComponents(fe, &fNc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      PetscCall(PetscFVGetQuadrature(fv, &quad));
      PetscCall(PetscFVGetNumComponents(fv, &fNc));
    } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, f);
    Nc += fNc;
  }
  PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCheck(qNc == 1,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %" PetscInt_FMT " > 1", qNc);
  PetscCall(PetscMalloc6(Nc*2, &valsum, Nc, &interpolant, cdim*Nq, &coords, Nq, &fegeom.detJ, cdim*cdim*Nq, &fegeom.J, cdim*cdim*Nq, &fegeom.invJ));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar volsum = 0.0;
    PetscInt   *star   = NULL;
    PetscInt    starSize, st, fc;

    PetscCall(PetscArrayzero(valsum, Nc));
    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    for (st = 0; st < starSize*2; st += 2) {
      const PetscInt cell = star[st];
      PetscScalar   *val  = &valsum[Nc];
      PetscScalar   *x    = NULL;
      PetscReal      vol  = 0.0;
      PetscInt       foff = 0;

      if ((cell < cStart) || (cell >= cEnd)) continue;
      PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      PetscCall(DMPlexVecGetClosure(dm, NULL, locX, cell, NULL, &x));
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscInt     Nb, fNc, q;

        PetscCall(PetscArrayzero(val, Nc));
        PetscCall(DMGetField(dm, f, NULL, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID)      {PetscCall(PetscFEGetNumComponents((PetscFE) obj, &fNc));PetscCall(PetscFEGetDimension((PetscFE) obj, &Nb));}
        else if (id == PETSCFV_CLASSID) {PetscCall(PetscFVGetNumComponents((PetscFV) obj, &fNc));Nb = 1;}
        else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, f);
        for (q = 0; q < Nq; ++q) {
          const PetscReal wt = quadWeights[q]*fegeom.detJ[q];
          PetscFEGeom     qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*cdim*cdim];
          qgeom.invJ     = &fegeom.invJ[q*cdim*cdim];
          qgeom.detJ     = &fegeom.detJ[q];
          PetscCheck(fegeom.detJ[q] > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT ", quadrature points %" PetscInt_FMT, (double) fegeom.detJ[q], cell, q);
          if (id == PETSCFE_CLASSID) PetscCall(PetscFEInterpolate_Static((PetscFE) obj, &x[foff], &qgeom, q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, f);
          for (fc = 0; fc < fNc; ++fc) val[foff+fc] += interpolant[fc]*wt;
          vol += wt;
        }
        foff += Nb;
      }
      PetscCall(DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x));
      for (fc = 0; fc < Nc; ++fc) valsum[fc] += val[fc];
      volsum += vol;
      if (debug) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Vertex %" PetscInt_FMT " Cell %" PetscInt_FMT " value: [", v, cell));
        for (fc = 0; fc < Nc; ++fc) {
          if (fc) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", (double) PetscRealPart(val[fc])));
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "]\n"));
      }
    }
    for (fc = 0; fc < Nc; ++fc) valsum[fc] /= volsum;
    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    PetscCall(DMPlexVecSetClosure(dmc, NULL, locC, v, valsum, INSERT_VALUES));
  }
  PetscCall(PetscFree6(valsum, interpolant, coords, fegeom.detJ, fegeom.J, fegeom.invJ));
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

.seealso: `DMProjectFunction()`, `DMComputeL2Diff()`, `DMPlexComputeL2FieldDiff()`, `DMComputeL2GradientDiff()`
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
  PetscCall(PetscCitationsRegister(ClementCitation, &Clementcite));
  PetscCall(VecGetDM(locC, &dmC));
  PetscCall(VecSet(locC, 0.0));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &coordDim));
  fegeom.dimEmbed = coordDim;
  PetscCall(DMGetNumFields(dm, &numFields));
  PetscCheck(numFields,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fields is zero!");
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    PetscCall(DMGetField(dm, field, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      PetscCall(PetscFEGetQuadrature(fe, &quad));
      PetscCall(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      PetscCall(PetscFVGetQuadrature(fv, &quad));
      PetscCall(PetscFVGetNumComponents(fv, &Nc));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
    numComponents += Nc;
  }
  PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCheck(!(qNc != 1) || !(qNc != numComponents),PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %" PetscInt_FMT " != %" PetscInt_FMT " field components", qNc, numComponents);
  PetscCall(PetscMalloc6(coordDim*numComponents*2,&gradsum,coordDim*numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar volsum = 0.0;
    PetscInt   *star   = NULL;
    PetscInt    starSize, st, d, fc;

    PetscCall(PetscArrayzero(gradsum, coordDim*numComponents));
    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    for (st = 0; st < starSize*2; st += 2) {
      const PetscInt cell = star[st];
      PetscScalar   *grad = &gradsum[coordDim*numComponents];
      PetscScalar   *x    = NULL;
      PetscReal      vol  = 0.0;

      if ((cell < cStart) || (cell >= cEnd)) continue;
      PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      PetscCall(DMPlexVecGetClosure(dm, NULL, locX, cell, NULL, &x));
      for (field = 0, fieldOffset = 0; field < numFields; ++field) {
        PetscObject  obj;
        PetscClassId id;
        PetscInt     Nb, Nc, q, qc = 0;

        PetscCall(PetscArrayzero(grad, coordDim*numComponents));
        PetscCall(DMGetField(dm, field, NULL, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID)      {PetscCall(PetscFEGetNumComponents((PetscFE) obj, &Nc));PetscCall(PetscFEGetDimension((PetscFE) obj, &Nb));}
        else if (id == PETSCFV_CLASSID) {PetscCall(PetscFVGetNumComponents((PetscFV) obj, &Nc));Nb = 1;}
        else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*coordDim*coordDim];
          qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
          qgeom.detJ     = &fegeom.detJ[q];
          PetscCheck(fegeom.detJ[q] > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT ", quadrature points %" PetscInt_FMT, (double)fegeom.detJ[q], cell, q);
          if (id == PETSCFE_CLASSID)      PetscCall(PetscFEInterpolateGradient_Static((PetscFE) obj, 1, &x[fieldOffset], &qgeom, q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+qc];

            for (d = 0; d < coordDim; ++d) grad[fc*coordDim+d] += interpolant[fc*dim+d]*wt*fegeom.detJ[q];
          }
          vol += quadWeights[q*qNc]*fegeom.detJ[q];
        }
        fieldOffset += Nb;
        qc          += Nc;
      }
      PetscCall(DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x));
      for (fc = 0; fc < numComponents; ++fc) {
        for (d = 0; d < coordDim; ++d) {
          gradsum[fc*coordDim+d] += grad[fc*coordDim+d];
        }
      }
      volsum += vol;
      if (debug) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Vertex %" PetscInt_FMT " Cell %" PetscInt_FMT " gradient: [", v, cell));
        for (fc = 0; fc < numComponents; ++fc) {
          for (d = 0; d < coordDim; ++d) {
            if (fc || d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", (double)PetscRealPart(grad[fc*coordDim+d])));
          }
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "]\n"));
      }
    }
    for (fc = 0; fc < numComponents; ++fc) {
      for (d = 0; d < coordDim; ++d) gradsum[fc*coordDim+d] /= volsum;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    PetscCall(DMPlexVecSetClosure(dmC, NULL, locC, v, gradsum, INSERT_VALUES));
  }
  PetscCall(PetscFree6(gradsum,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
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
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetNumFields(dm, &Nf));
  /* Determine which discretizations we have */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(PetscDSGetDiscretization(prob, f, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFV_CLASSID) useFVM = PETSC_TRUE;
  }
  /* Get local solution with boundary values */
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, 0.0, NULL, NULL, NULL));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
  /* Read DS information */
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscDSGetComponentOffsets(prob, &uOff));
  PetscCall(PetscDSGetComponentDerivativeOffsets(prob, &uOff_x));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,numCells,cStart,1,&cellIS));
  PetscCall(PetscDSGetConstants(prob, &numConstants, &constants));
  /* Read Auxiliary DS information */
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &locA));
  if (locA) {
    PetscCall(VecGetDM(locA, &dmAux));
    PetscCall(DMGetDS(dmAux, &probAux));
    PetscCall(PetscDSGetNumFields(probAux, &NfAux));
    PetscCall(DMGetLocalSection(dmAux, &sectionAux));
    PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
    PetscCall(PetscDSGetComponentOffsets(probAux, &aOff));
  }
  /* Allocate data  arrays */
  PetscCall(PetscCalloc1(numCells*totDim, &u));
  if (dmAux) PetscCall(PetscMalloc1(numCells*totDimAux, &a));
  /* Read out geometry */
  PetscCall(DMGetCoordinateField(dm,&coordField));
  PetscCall(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
  if (maxDegree <= 1) {
    PetscCall(DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad));
    if (affineQuad) {
      PetscCall(DMFieldCreateFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&cgeomFEM));
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

      PetscCall(PetscDSGetDiscretization(prob, f, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFV_CLASSID) {fv = (PetscFV) obj; break;}
    }
    PetscCall(PetscFVGetComputeGradients(fv, &compGrad));
    PetscCall(PetscFVSetComputeGradients(fv, PETSC_TRUE));
    PetscCall(DMPlexComputeGeometryFVM(dm, &cellGeometryFVM, &faceGeometryFVM));
    PetscCall(DMPlexComputeGradientFVM(dm, fv, faceGeometryFVM, cellGeometryFVM, &dmGrad));
    PetscCall(PetscFVSetComputeGradients(fv, compGrad));
    PetscCall(VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM));
    /* Reconstruct and limit cell gradients */
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(DMGetGlobalVector(dmGrad, &grad));
    PetscCall(DMPlexReconstructGradients_Internal(dm, fv, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad));
    /* Communicate gradient values */
    PetscCall(DMGetLocalVector(dmGrad, &locGrad));
    PetscCall(DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad));
    PetscCall(DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad));
    PetscCall(DMRestoreGlobalVector(dmGrad, &grad));
    /* Handle non-essential (e.g. outflow) boundary values */
    PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, 0.0, faceGeometryFVM, cellGeometryFVM, locGrad));
    PetscCall(VecGetArrayRead(locGrad, &lgrad));
  }
  /* Read out data from inputs */
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscInt     i;

    PetscCall(DMPlexVecGetClosure(dm, section, locX, c, NULL, &x));
    for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
    PetscCall(DMPlexVecRestoreClosure(dm, section, locX, c, NULL, &x));
    if (dmAux) {
      PetscCall(DMPlexVecGetClosure(dmAux, sectionAux, locA, c, NULL, &x));
      for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
      PetscCall(DMPlexVecRestoreClosure(dmAux, sectionAux, locA, c, NULL, &x));
    }
  }
  /* Do integration for each field */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

    PetscCall(PetscDSGetDiscretization(prob, f, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE         fe = (PetscFE) obj;
      PetscQuadrature q;
      PetscFEGeom     *chunkGeom = NULL;
      PetscInt        Nq, Nb;

      PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
      PetscCall(PetscFEGetQuadrature(fe, &q));
      PetscCall(PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL));
      PetscCall(PetscFEGetDimension(fe, &Nb));
      blockSize = Nb*Nq;
      batchSize = numBlocks * blockSize;
      PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      if (!affineQuad) {
        PetscCall(DMFieldCreateFEGeom(coordField,cellIS,q,PETSC_FALSE,&cgeomFEM));
      }
      PetscCall(PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom));
      PetscCall(PetscFEIntegrate(prob, f, Ne, chunkGeom, u, probAux, a, cintegral));
      PetscCall(PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&chunkGeom));
      PetscCall(PetscFEIntegrate(prob, f, Nr, chunkGeom, &u[offset*totDim], probAux, &a[offset*totDimAux], &cintegral[offset*Nf]));
      PetscCall(PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&chunkGeom));
      if (!affineQuad) {
        PetscCall(PetscFEGeomDestroy(&cgeomFEM));
      }
    } else if (id == PETSCFV_CLASSID) {
      PetscInt       foff;
      PetscPointFunc obj_func;
      PetscScalar    lint;

      PetscCall(PetscDSGetObjective(prob, f, &obj_func));
      PetscCall(PetscDSGetFieldOffset(prob, f, &foff));
      if (obj_func) {
        for (c = 0; c < numCells; ++c) {
          PetscScalar *u_x;

          PetscCall(DMPlexPointLocalRead(dmGrad, c, lgrad, &u_x));
          obj_func(dim, Nf, NfAux, uOff, uOff_x, &u[totDim*c+foff], NULL, u_x, aOff, NULL, &a[totDimAux*c], NULL, NULL, 0.0, cgeomFVM[c].centroid, numConstants, constants, &lint);
          cintegral[c*Nf+f] += PetscRealPart(lint)*cgeomFVM[c].volume;
        }
      }
    } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, f);
  }
  /* Cleanup data arrays */
  if (useFVM) {
    PetscCall(VecRestoreArrayRead(locGrad, &lgrad));
    PetscCall(VecRestoreArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM));
    PetscCall(DMRestoreLocalVector(dmGrad, &locGrad));
    PetscCall(VecDestroy(&faceGeometryFVM));
    PetscCall(VecDestroy(&cellGeometryFVM));
    PetscCall(DMDestroy(&dmGrad));
  }
  if (dmAux) PetscCall(PetscFree(a));
  PetscCall(PetscFree(u));
  /* Cleanup */
  if (affineQuad) {
    PetscCall(PetscFEGeomDestroy(&cgeomFEM));
  }
  PetscCall(PetscQuadratureDestroy(&affineQuad));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(DMRestoreLocalVector(dm, &locX));
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

.seealso: `DMPlexSNESComputeResidualFEM()`
@*/
PetscErrorCode DMPlexComputeIntegralFEM(DM dm, Vec X, PetscScalar *integral, void *user)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  PetscScalar   *cintegral, *lintegral;
  PetscInt       Nf, f, cellHeight, cStart, cEnd, cell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidScalarPointer(integral, 3);
  PetscCall(PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd));
  /* TODO Introduce a loop over large chunks (right now this is a single chunk) */
  PetscCall(PetscCalloc2(Nf, &lintegral, (cEnd-cStart)*Nf, &cintegral));
  PetscCall(DMPlexComputeIntegral_Internal(dm, X, cStart, cEnd, cintegral, user));
  /* Sum up values */
  for (cell = cStart; cell < cEnd; ++cell) {
    const PetscInt c = cell - cStart;

    if (mesh->printFEM > 1) PetscCall(DMPrintCellVector(cell, "Cell Integral", Nf, &cintegral[c*Nf]));
    for (f = 0; f < Nf; ++f) lintegral[f] += cintegral[c*Nf+f];
  }
  PetscCall(MPIU_Allreduce(lintegral, integral, Nf, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject) dm)));
  if (mesh->printFEM) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) dm), "Integral:"));
    for (f = 0; f < Nf; ++f) PetscCall(PetscPrintf(PetscObjectComm((PetscObject) dm), " %g", (double) PetscRealPart(integral[f])));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) dm), "\n"));
  }
  PetscCall(PetscFree2(lintegral, cintegral));
  PetscCall(PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0));
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

.seealso: `DMPlexSNESComputeResidualFEM()`
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
  PetscCall(PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd));
  /* TODO Introduce a loop over large chunks (right now this is a single chunk) */
  PetscCall(PetscCalloc1((cEnd-cStart)*Nf, &cintegral));
  PetscCall(DMPlexComputeIntegral_Internal(dm, X, cStart, cEnd, cintegral, user));
  /* Put values in F*/
  PetscCall(VecGetDM(F, &dmF));
  PetscCall(DMGetLocalSection(dmF, &sectionF));
  PetscCall(VecGetArray(F, &af));
  for (cell = cStart; cell < cEnd; ++cell) {
    const PetscInt c = cell - cStart;
    PetscInt       dof, off;

    if (mesh->printFEM > 1) PetscCall(DMPrintCellVector(cell, "Cell Integral", Nf, &cintegral[c*Nf]));
    PetscCall(PetscSectionGetDof(sectionF, cell, &dof));
    PetscCall(PetscSectionGetOffset(sectionF, cell, &off));
    PetscCheck(dof == Nf,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of cell dofs %" PetscInt_FMT " != %" PetscInt_FMT, dof, Nf);
    for (f = 0; f < Nf; ++f) af[off+f] = cintegral[c*Nf+f];
  }
  PetscCall(VecRestoreArray(F, &af));
  PetscCall(PetscFree(cintegral));
  PetscCall(PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0));
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
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &Nf));
  /* Determine which discretizations we have */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(PetscDSGetDiscretization(prob, f, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    PetscCheck(id != PETSCFV_CLASSID,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Not supported for FVM (field %" PetscInt_FMT ")", f);
  }
  /* Read DS information */
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscDSGetComponentOffsets(prob, &uOff));
  PetscCall(PetscDSGetComponentDerivativeOffsets(prob, &uOff_x));
  PetscCall(PetscDSGetConstants(prob, &numConstants, &constants));
  /* Read Auxiliary DS information */
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &locA));
  if (locA) {
    DM dmAux;

    PetscCall(VecGetDM(locA, &dmAux));
    PetscCall(DMGetEnclosureRelation(dmAux, dm, &encAux));
    PetscCall(DMConvert(dmAux, DMPLEX, &plexA));
    PetscCall(DMGetDS(dmAux, &probAux));
    PetscCall(PetscDSGetNumFields(probAux, &NfAux));
    PetscCall(DMGetLocalSection(dmAux, &sectionAux));
    PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
    PetscCall(PetscDSGetComponentOffsets(probAux, &aOff));
  }
  /* Integrate over points */
  {
    PetscFEGeom    *fgeom, *chunkGeom = NULL;
    PetscInt        maxDegree;
    PetscQuadrature qGeom = NULL;
    const PetscInt *points;
    PetscInt        numFaces, face, Nq, field;
    PetscInt        numChunks, chunkSize, chunk, Nr, offset;

    PetscCall(ISGetLocalSize(pointIS, &numFaces));
    PetscCall(ISGetIndices(pointIS, &points));
    PetscCall(PetscCalloc2(numFaces*totDim, &u, locA ? numFaces*totDimAux : 0, &a));
    PetscCall(DMFieldGetDegree(coordField, pointIS, NULL, &maxDegree));
    for (field = 0; field < Nf; ++field) {
      PetscFE fe;

      PetscCall(PetscDSGetDiscretization(prob, field, (PetscObject *) &fe));
      if (maxDegree <= 1) PetscCall(DMFieldCreateDefaultQuadrature(coordField, pointIS, &qGeom));
      if (!qGeom) {
        PetscCall(PetscFEGetFaceQuadrature(fe, &qGeom));
        PetscCall(PetscObjectReference((PetscObject) qGeom));
      }
      PetscCall(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
      PetscCall(DMPlexGetFEGeom(coordField, pointIS, qGeom, PETSC_TRUE, &fgeom));
      for (face = 0; face < numFaces; ++face) {
        const PetscInt point = points[face], *support;
        PetscScalar    *x    = NULL;
        PetscInt       i;

        PetscCall(DMPlexGetSupport(dm, point, &support));
        PetscCall(DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x));
        for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
        PetscCall(DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x));
        if (locA) {
          PetscInt subp;
          PetscCall(DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp));
          PetscCall(DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x));
          for (i = 0; i < totDimAux; ++i) a[f*totDimAux+i] = x[i];
          PetscCall(DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x));
        }
      }
      /* Get blocking */
      {
        PetscQuadrature q;
        PetscInt        numBatches, batchSize, numBlocks, blockSize;
        PetscInt        Nq, Nb;

        PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
        PetscCall(PetscFEGetQuadrature(fe, &q));
        PetscCall(PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL));
        PetscCall(PetscFEGetDimension(fe, &Nb));
        blockSize = Nb*Nq;
        batchSize = numBlocks * blockSize;
        chunkSize = numBatches*batchSize;
        PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
        numChunks = numFaces / chunkSize;
        Nr        = numFaces % chunkSize;
        offset    = numFaces - Nr;
      }
      /* Do integration for each field */
      for (chunk = 0; chunk < numChunks; ++chunk) {
        PetscCall(PetscFEGeomGetChunk(fgeom, chunk*chunkSize, (chunk+1)*chunkSize, &chunkGeom));
        PetscCall(PetscFEIntegrateBd(prob, field, func, chunkSize, chunkGeom, u, probAux, a, fintegral));
        PetscCall(PetscFEGeomRestoreChunk(fgeom, 0, offset, &chunkGeom));
      }
      PetscCall(PetscFEGeomGetChunk(fgeom, offset, numFaces, &chunkGeom));
      PetscCall(PetscFEIntegrateBd(prob, field, func, Nr, chunkGeom, &u[offset*totDim], probAux, a ? &a[offset*totDimAux] : NULL, &fintegral[offset*Nf]));
      PetscCall(PetscFEGeomRestoreChunk(fgeom, offset, numFaces, &chunkGeom));
      /* Cleanup data arrays */
      PetscCall(DMPlexRestoreFEGeom(coordField, pointIS, qGeom, PETSC_TRUE, &fgeom));
      PetscCall(PetscQuadratureDestroy(&qGeom));
      PetscCall(PetscFree2(u, a));
      PetscCall(ISRestoreIndices(pointIS, &points));
    }
  }
  if (plex)  PetscCall(DMDestroy(&plex));
  if (plexA) PetscCall(DMDestroy(&plexA));
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

.seealso: `DMPlexComputeIntegralFEM()`, `DMPlexComputeBdResidualFEM()`
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
  if (vals) PetscValidIntPointer(vals, 5);
  PetscValidScalarPointer(integral, 7);
  PetscCall(PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMLabelGetStratumIS(depthLabel, dim-1, &facetIS));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &Nf));
  /* Get local solution with boundary values */
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, 0.0, NULL, NULL, NULL));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
  /* Loop over label values */
  PetscCall(PetscArrayzero(integral, Nf));
  for (v = 0; v < numVals; ++v) {
    IS           pointIS;
    PetscInt     numFaces, face;
    PetscScalar *fintegral;

    PetscCall(DMLabelGetStratumIS(label, vals[v], &pointIS));
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a priori if one is a superset of the other */
      PetscCall(ISIntersect_Caching_Internal(facetIS, pointIS, &isectIS));
      PetscCall(ISDestroy(&pointIS));
      pointIS = isectIS;
    }
    PetscCall(ISGetLocalSize(pointIS, &numFaces));
    PetscCall(PetscCalloc1(numFaces*Nf, &fintegral));
    PetscCall(DMPlexComputeBdIntegral_Internal(dm, locX, pointIS, func, fintegral, user));
    /* Sum point contributions into integral */
    for (f = 0; f < Nf; ++f) for (face = 0; face < numFaces; ++face) integral[f] += fintegral[face*Nf+f];
    PetscCall(PetscFree(fintegral));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(ISDestroy(&facetIS));
  PetscCall(PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0));
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

.seealso: `DMPlexComputeInterpolatorGeneral()`, `DMPlexComputeJacobianFEM()`
@*/
PetscErrorCode DMPlexComputeInterpolatorNested(DM dmc, DM dmf, PetscBool isRefined, Mat In, void *user)
{
  DM_Plex     *mesh  = (DM_Plex *) dmc->data;
  const char  *name  = "Interpolator";
  PetscFE     *feRef;
  PetscFV     *fvRef;
  PetscSection fsection, fglobalSection;
  PetscSection csection, cglobalSection;
  PetscScalar *elemMat;
  PetscInt     dim, Nf, f, fieldI, fieldJ, offsetI, offsetJ, cStart, cEnd, c;
  PetscInt     cTotDim = 0, rTotDim = 0;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_InterpolatorFEM,dmc,dmf,0,0));
  PetscCall(DMGetDimension(dmf, &dim));
  PetscCall(DMGetLocalSection(dmf, &fsection));
  PetscCall(DMGetGlobalSection(dmf, &fglobalSection));
  PetscCall(DMGetLocalSection(dmc, &csection));
  PetscCall(DMGetGlobalSection(dmc, &cglobalSection));
  PetscCall(PetscSectionGetNumFields(fsection, &Nf));
  PetscCall(DMPlexGetSimplexOrBoxCells(dmc, 0, &cStart, &cEnd));
  PetscCall(PetscCalloc2(Nf, &feRef, Nf, &fvRef));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj, objc;
    PetscClassId id, idc;
    PetscInt     rNb = 0, Nc = 0, cNb = 0;

    PetscCall(DMGetField(dmf, f, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      if (isRefined) {
        PetscCall(PetscFERefine(fe, &feRef[f]));
      } else {
        PetscCall(PetscObjectReference((PetscObject) fe));
        feRef[f] = fe;
      }
      PetscCall(PetscFEGetDimension(feRef[f], &rNb));
      PetscCall(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      if (isRefined) {
        PetscCall(PetscFVRefine(fv, &fvRef[f]));
      } else {
        PetscCall(PetscObjectReference((PetscObject) fv));
        fvRef[f] = fv;
      }
      PetscCall(PetscFVGetDualSpace(fvRef[f], &Q));
      PetscCall(PetscDualSpaceGetDimension(Q, &rNb));
      PetscCall(PetscFVGetDualSpace(fv, &Q));
      PetscCall(PetscFVGetNumComponents(fv, &Nc));
    }
    PetscCall(DMGetField(dmc, f, NULL, &objc));
    PetscCall(PetscObjectGetClassId(objc, &idc));
    if (idc == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) objc;

      PetscCall(PetscFEGetDimension(fe, &cNb));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      PetscCall(PetscFVGetDualSpace(fv, &Q));
      PetscCall(PetscDualSpaceGetDimension(Q, &cNb));
    }
    rTotDim += rNb;
    cTotDim += cNb;
  }
  PetscCall(PetscMalloc1(rTotDim*cTotDim,&elemMat));
  PetscCall(PetscArrayzero(elemMat, rTotDim*cTotDim));
  for (fieldI = 0, offsetI = 0; fieldI < Nf; ++fieldI) {
    PetscDualSpace   Qref;
    PetscQuadrature  f;
    const PetscReal *qpoints, *qweights;
    PetscReal       *points;
    PetscInt         npoints = 0, Nc, Np, fpdim, i, k, p, d;

    /* Compose points from all dual basis functionals */
    if (feRef[fieldI]) {
      PetscCall(PetscFEGetDualSpace(feRef[fieldI], &Qref));
      PetscCall(PetscFEGetNumComponents(feRef[fieldI], &Nc));
    } else {
      PetscCall(PetscFVGetDualSpace(fvRef[fieldI], &Qref));
      PetscCall(PetscFVGetNumComponents(fvRef[fieldI], &Nc));
    }
    PetscCall(PetscDualSpaceGetDimension(Qref, &fpdim));
    for (i = 0; i < fpdim; ++i) {
      PetscCall(PetscDualSpaceGetFunctional(Qref, i, &f));
      PetscCall(PetscQuadratureGetData(f, NULL, NULL, &Np, NULL, NULL));
      npoints += Np;
    }
    PetscCall(PetscMalloc1(npoints*dim,&points));
    for (i = 0, k = 0; i < fpdim; ++i) {
      PetscCall(PetscDualSpaceGetFunctional(Qref, i, &f));
      PetscCall(PetscQuadratureGetData(f, NULL, NULL, &Np, &qpoints, NULL));
      for (p = 0; p < Np; ++p, ++k) for (d = 0; d < dim; ++d) points[k*dim+d] = qpoints[p*dim+d];
    }

    for (fieldJ = 0, offsetJ = 0; fieldJ < Nf; ++fieldJ) {
      PetscObject  obj;
      PetscClassId id;
      PetscInt     NcJ = 0, cpdim = 0, j, qNc;

      PetscCall(DMGetField(dmc, fieldJ, NULL, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) {
        PetscFE           fe = (PetscFE) obj;
        PetscTabulation T  = NULL;

        /* Evaluate basis at points */
        PetscCall(PetscFEGetNumComponents(fe, &NcJ));
        PetscCall(PetscFEGetDimension(fe, &cpdim));
        /* For now, fields only interpolate themselves */
        if (fieldI == fieldJ) {
          PetscCheck(Nc == NcJ,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %" PetscInt_FMT " does not match coarse field %" PetscInt_FMT, Nc, NcJ);
          PetscCall(PetscFECreateTabulation(fe, 1, npoints, points, 0, &T));
          for (i = 0, k = 0; i < fpdim; ++i) {
            PetscCall(PetscDualSpaceGetFunctional(Qref, i, &f));
            PetscCall(PetscQuadratureGetData(f, NULL, &qNc, &Np, NULL, &qweights));
            PetscCheck(qNc == NcJ,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %" PetscInt_FMT " does not match coarse field %" PetscInt_FMT, qNc, NcJ);
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
          PetscCall(PetscTabulationDestroy(&T));
        }
      } else if (id == PETSCFV_CLASSID) {
        PetscFV        fv = (PetscFV) obj;

        /* Evaluate constant function at points */
        PetscCall(PetscFVGetNumComponents(fv, &NcJ));
        cpdim = 1;
        /* For now, fields only interpolate themselves */
        if (fieldI == fieldJ) {
          PetscCheck(Nc == NcJ,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %" PetscInt_FMT " does not match coarse field %" PetscInt_FMT, Nc, NcJ);
          for (i = 0, k = 0; i < fpdim; ++i) {
            PetscCall(PetscDualSpaceGetFunctional(Qref, i, &f));
            PetscCall(PetscQuadratureGetData(f, NULL, &qNc, &Np, NULL, &qweights));
            PetscCheck(qNc == NcJ,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %" PetscInt_FMT " does not match coarse field %" PetscInt_FMT, qNc, NcJ);
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
    PetscCall(PetscFree(points));
  }
  if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(0, name, rTotDim, cTotDim, elemMat));
  /* Preallocate matrix */
  {
    Mat          preallocator;
    PetscScalar *vals;
    PetscInt    *cellCIndices, *cellFIndices;
    PetscInt     locRows, locCols, cell;

    PetscCall(MatGetLocalSize(In, &locRows, &locCols));
    PetscCall(MatCreate(PetscObjectComm((PetscObject) In), &preallocator));
    PetscCall(MatSetType(preallocator, MATPREALLOCATOR));
    PetscCall(MatSetSizes(preallocator, locRows, locCols, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetUp(preallocator));
    PetscCall(PetscCalloc3(rTotDim*cTotDim, &vals,cTotDim,&cellCIndices,rTotDim,&cellFIndices));
    for (cell = cStart; cell < cEnd; ++cell) {
      if (isRefined) {
        PetscCall(DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, cell, cellCIndices, cellFIndices));
        PetscCall(MatSetValues(preallocator, rTotDim, cellFIndices, cTotDim, cellCIndices, vals, INSERT_VALUES));
      } else {
        PetscCall(DMPlexMatSetClosureGeneral(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, preallocator, cell, vals, INSERT_VALUES));
      }
    }
    PetscCall(PetscFree3(vals,cellCIndices,cellFIndices));
    PetscCall(MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY));
    PetscCall(MatPreallocatorPreallocate(preallocator, PETSC_TRUE, In));
    PetscCall(MatDestroy(&preallocator));
  }
  /* Fill matrix */
  PetscCall(MatZeroEntries(In));
  for (c = cStart; c < cEnd; ++c) {
    if (isRefined) {
      PetscCall(DMPlexMatSetClosureRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, In, c, elemMat, INSERT_VALUES));
    } else {
      PetscCall(DMPlexMatSetClosureGeneral(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, In, c, elemMat, INSERT_VALUES));
    }
  }
  for (f = 0; f < Nf; ++f) PetscCall(PetscFEDestroy(&feRef[f]));
  PetscCall(PetscFree2(feRef,fvRef));
  PetscCall(PetscFree(elemMat));
  PetscCall(MatAssemblyBegin(In, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(In, MAT_FINAL_ASSEMBLY));
  if (mesh->printFEM > 1) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)In), "%s:\n", name));
    PetscCall(MatChop(In, 1.0e-10));
    PetscCall(MatView(In, NULL));
  }
  PetscCall(PetscLogEventEnd(DMPLEX_InterpolatorFEM,dmc,dmf,0,0));
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

.seealso: `DMPlexComputeInterpolatorNested()`, `DMPlexComputeJacobianFEM()`
@*/
PetscErrorCode DMPlexComputeInterpolatorGeneral(DM dmc, DM dmf, Mat In, void *user)
{
  DM_Plex     *mesh = (DM_Plex *) dmf->data;
  const char  *name = "Interpolator";
  PetscDS      prob;
  Mat          interp;
  PetscSection fsection, globalFSection;
  PetscSection csection, globalCSection;
  PetscInt     locRows, locCols;
  PetscReal   *x, *v0, *J, *invJ, detJ;
  PetscReal   *v0c, *Jc, *invJc, detJc;
  PetscScalar *elemMat;
  PetscInt     dim, Nf, field, totDim, cStart, cEnd, cell, ccell, s;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_InterpolatorFEM,dmc,dmf,0,0));
  PetscCall(DMGetCoordinateDim(dmc, &dim));
  PetscCall(DMGetDS(dmc, &prob));
  PetscCall(PetscDSGetWorkspace(prob, &x, NULL, NULL, NULL, NULL));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ));
  PetscCall(PetscMalloc3(dim,&v0c,dim*dim,&Jc,dim*dim,&invJc));
  PetscCall(DMGetLocalSection(dmf, &fsection));
  PetscCall(DMGetGlobalSection(dmf, &globalFSection));
  PetscCall(DMGetLocalSection(dmc, &csection));
  PetscCall(DMGetGlobalSection(dmc, &globalCSection));
  PetscCall(DMPlexGetSimplexOrBoxCells(dmf, 0, &cStart, &cEnd));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscMalloc1(totDim, &elemMat));

  PetscCall(MatGetLocalSize(In, &locRows, &locCols));
  PetscCall(MatCreate(PetscObjectComm((PetscObject) In), &interp));
  PetscCall(MatSetType(interp, MATPREALLOCATOR));
  PetscCall(MatSetSizes(interp, locRows, locCols, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetUp(interp));
  for (s = 0; s < 2; ++s) {
    for (field = 0; field < Nf; ++field) {
      PetscObject      obj;
      PetscClassId     id;
      PetscDualSpace   Q = NULL;
      PetscTabulation  T = NULL;
      PetscQuadrature  f;
      const PetscReal *qpoints, *qweights;
      PetscInt         Nc, qNc, Np, fpdim, off, i, d;

      PetscCall(PetscDSGetFieldOffset(prob, field, &off));
      PetscCall(PetscDSGetDiscretization(prob, field, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) {
        PetscFE fe = (PetscFE) obj;

        PetscCall(PetscFEGetDualSpace(fe, &Q));
        PetscCall(PetscFEGetNumComponents(fe, &Nc));
        if (s) PetscCall(PetscFECreateTabulation(fe, 1, 1, x, 0, &T));
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        PetscCall(PetscFVGetDualSpace(fv, &Q));
        Nc = 1;
      } else SETERRQ(PetscObjectComm((PetscObject) dmc), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
      PetscCall(PetscDualSpaceGetDimension(Q, &fpdim));
      /* For each fine grid cell */
      for (cell = cStart; cell < cEnd; ++cell) {
        PetscInt *findices,   *cindices;
        PetscInt  numFIndices, numCIndices;

        PetscCall(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
        PetscCall(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
        PetscCheck(numFIndices == totDim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fine indices %" PetscInt_FMT " != %" PetscInt_FMT " dual basis vecs", numFIndices, totDim);
        for (i = 0; i < fpdim; ++i) {
          Vec                pointVec;
          PetscScalar       *pV;
          PetscSF            coarseCellSF = NULL;
          const PetscSFNode *coarseCells;
          PetscInt           numCoarseCells, cpdim, row = findices[i+off], q, c, j;

          /* Get points from the dual basis functional quadrature */
          PetscCall(PetscDualSpaceGetFunctional(Q, i, &f));
          PetscCall(PetscQuadratureGetData(f, NULL, &qNc, &Np, &qpoints, &qweights));
          PetscCheck(qNc == Nc,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %" PetscInt_FMT " does not match coarse field %" PetscInt_FMT, qNc, Nc);
          PetscCall(VecCreateSeq(PETSC_COMM_SELF, Np*dim, &pointVec));
          PetscCall(VecSetBlockSize(pointVec, dim));
          PetscCall(VecGetArray(pointVec, &pV));
          for (q = 0; q < Np; ++q) {
            const PetscReal xi0[3] = {-1., -1., -1.};

            /* Transform point to real space */
            CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
            for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
          }
          PetscCall(VecRestoreArray(pointVec, &pV));
          /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
          /* OPT: Read this out from preallocation information */
          PetscCall(DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF));
          /* Update preallocation info */
          PetscCall(PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells));
          PetscCheck(numCoarseCells == Np, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not all closure points located");
          PetscCall(VecGetArray(pointVec, &pV));
          for (ccell = 0; ccell < numCoarseCells; ++ccell) {
            PetscReal pVReal[3];
            const PetscReal xi0[3] = {-1., -1., -1.};

            PetscCall(DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
            if (id == PETSCFE_CLASSID) PetscCall(PetscFEGetDimension((PetscFE) obj, &cpdim));
            else                       cpdim = 1;

            if (s) {
              /* Transform points from real space to coarse reference space */
              PetscCall(DMPlexComputeCellGeometryFEM(dmc, coarseCells[ccell].index, NULL, v0c, Jc, invJc, &detJc));
              for (d = 0; d < dim; ++d) pVReal[d] = PetscRealPart(pV[ccell*dim+d]);
              CoordinatesRealToRef(dim, dim, xi0, v0c, invJc, pVReal, x);

              if (id == PETSCFE_CLASSID) {
                /* Evaluate coarse basis on contained point */
                PetscCall(PetscFEComputeTabulation((PetscFE) obj, 1, x, 0, T));
                PetscCall(PetscArrayzero(elemMat, cpdim));
                /* Get elemMat entries by multiplying by weight */
                for (j = 0; j < cpdim; ++j) {
                  for (c = 0; c < Nc; ++c) elemMat[j] += T->T[0][j*Nc + c]*qweights[ccell*qNc + c];
                }
              } else {
                for (j = 0; j < cpdim; ++j) {
                  for (c = 0; c < Nc; ++c) elemMat[j] += 1.0*qweights[ccell*qNc + c];
                }
              }
              if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat));
            }
            /* Update interpolator */
            PetscCheck(numCIndices == totDim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %" PetscInt_FMT " != %" PetscInt_FMT, numCIndices, totDim);
            PetscCall(MatSetValues(interp, 1, &row, cpdim, &cindices[off], elemMat, INSERT_VALUES));
            PetscCall(DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
          }
          PetscCall(VecRestoreArray(pointVec, &pV));
          PetscCall(PetscSFDestroy(&coarseCellSF));
          PetscCall(VecDestroy(&pointVec));
        }
        PetscCall(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      }
      if (s && id == PETSCFE_CLASSID) PetscCall(PetscTabulationDestroy(&T));
    }
    if (!s) {
      PetscCall(MatAssemblyBegin(interp, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(interp, MAT_FINAL_ASSEMBLY));
      PetscCall(MatPreallocatorPreallocate(interp, PETSC_TRUE, In));
      PetscCall(MatDestroy(&interp));
      interp = In;
    }
  }
  PetscCall(PetscFree3(v0, J, invJ));
  PetscCall(PetscFree3(v0c, Jc, invJc));
  PetscCall(PetscFree(elemMat));
  PetscCall(MatAssemblyBegin(In, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(In, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogEventEnd(DMPLEX_InterpolatorFEM,dmc,dmf,0,0));
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

.seealso: `DMPlexComputeMassMatrixNested()`, `DMPlexComputeInterpolatorNested()`, `DMPlexComputeInterpolatorGeneral()`, `DMPlexComputeJacobianFEM()`
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
  PetscCall(DMGetCoordinateDim(dmc, &dim));
  PetscCall(DMGetDS(dmc, &prob));
  PetscCall(PetscDSGetWorkspace(prob, &x, NULL, NULL, NULL, NULL));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ));
  PetscCall(PetscMalloc3(dim,&v0c,dim*dim,&Jc,dim*dim,&invJc));
  PetscCall(DMGetLocalSection(dmf, &fsection));
  PetscCall(DMGetGlobalSection(dmf, &globalFSection));
  PetscCall(DMGetLocalSection(dmc, &csection));
  PetscCall(DMGetGlobalSection(dmc, &globalCSection));
  PetscCall(DMPlexGetHeightStratum(dmf, 0, &cStart, &cEnd));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscMalloc1(totDim, &elemMat));

  PetscCall(MatGetLocalSize(mass, &locRows, NULL));
  PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject) mass), &rLayout));
  PetscCall(PetscLayoutSetLocalSize(rLayout, locRows));
  PetscCall(PetscLayoutSetBlockSize(rLayout, 1));
  PetscCall(PetscLayoutSetUp(rLayout));
  PetscCall(PetscLayoutGetRange(rLayout, &rStart, &rEnd));
  PetscCall(PetscLayoutDestroy(&rLayout));
  PetscCall(PetscCalloc2(locRows,&dnz,locRows,&onz));
  PetscCall(PetscHSetIJCreate(&ht));
  for (field = 0; field < Nf; ++field) {
    PetscObject      obj;
    PetscClassId     id;
    PetscQuadrature  quad;
    const PetscReal *qpoints;
    PetscInt         Nq, Nc, i, d;

    PetscCall(PetscDSGetDiscretization(prob, field, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) PetscCall(PetscFEGetQuadrature((PetscFE) obj, &quad));
    else                       PetscCall(PetscFVGetQuadrature((PetscFV) obj, &quad));
    PetscCall(PetscQuadratureGetData(quad, NULL, &Nc, &Nq, &qpoints, NULL));
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      Vec                pointVec;
      PetscScalar       *pV;
      PetscSF            coarseCellSF = NULL;
      const PetscSFNode *coarseCells;
      PetscInt           numCoarseCells, q, c;
      PetscInt          *findices,   *cindices;
      PetscInt           numFIndices, numCIndices;

      PetscCall(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      PetscCall(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
      /* Get points from the quadrature */
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, Nq*dim, &pointVec));
      PetscCall(VecSetBlockSize(pointVec, dim));
      PetscCall(VecGetArray(pointVec, &pV));
      for (q = 0; q < Nq; ++q) {
        const PetscReal xi0[3] = {-1., -1., -1.};

        /* Transform point to real space */
        CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
        for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
      }
      PetscCall(VecRestoreArray(pointVec, &pV));
      /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
      PetscCall(DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF));
      PetscCall(PetscSFViewFromOptions(coarseCellSF, NULL, "-interp_sf_view"));
      /* Update preallocation info */
      PetscCall(PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells));
      PetscCheck(numCoarseCells == Nq,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
      {
        PetscHashIJKey key;
        PetscBool      missing;

        for (i = 0; i < numFIndices; ++i) {
          key.i = findices[i];
          if (key.i >= 0) {
            /* Get indices for coarse elements */
            for (ccell = 0; ccell < numCoarseCells; ++ccell) {
              PetscCall(DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
              for (c = 0; c < numCIndices; ++c) {
                key.j = cindices[c];
                if (key.j < 0) continue;
                PetscCall(PetscHSetIJQueryAdd(ht, key, &missing));
                if (missing) {
                  if ((key.j >= rStart) && (key.j < rEnd)) ++dnz[key.i-rStart];
                  else                                     ++onz[key.i-rStart];
                }
              }
              PetscCall(DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
            }
          }
        }
      }
      PetscCall(PetscSFDestroy(&coarseCellSF));
      PetscCall(VecDestroy(&pointVec));
      PetscCall(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
    }
  }
  PetscCall(PetscHSetIJDestroy(&ht));
  PetscCall(MatXAIJSetPreallocation(mass, 1, dnz, onz, NULL, NULL));
  PetscCall(MatSetOption(mass, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  PetscCall(PetscFree2(dnz,onz));
  for (field = 0; field < Nf; ++field) {
    PetscObject       obj;
    PetscClassId      id;
    PetscTabulation T, Tfine;
    PetscQuadrature   quad;
    const PetscReal  *qpoints, *qweights;
    PetscInt          Nq, Nc, i, d;

    PetscCall(PetscDSGetDiscretization(prob, field, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscCall(PetscFEGetQuadrature((PetscFE) obj, &quad));
      PetscCall(PetscFEGetCellTabulation((PetscFE) obj, 1, &Tfine));
      PetscCall(PetscFECreateTabulation((PetscFE) obj, 1, 1, x, 0, &T));
    } else {
      PetscCall(PetscFVGetQuadrature((PetscFV) obj, &quad));
    }
    PetscCall(PetscQuadratureGetData(quad, NULL, &Nc, &Nq, &qpoints, &qweights));
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      Vec                pointVec;
      PetscScalar       *pV;
      PetscSF            coarseCellSF = NULL;
      const PetscSFNode *coarseCells;
      PetscInt           numCoarseCells, cpdim, q, c, j;
      PetscInt          *findices,   *cindices;
      PetscInt           numFIndices, numCIndices;

      PetscCall(DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
      PetscCall(DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ));
      /* Get points from the quadrature */
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, Nq*dim, &pointVec));
      PetscCall(VecSetBlockSize(pointVec, dim));
      PetscCall(VecGetArray(pointVec, &pV));
      for (q = 0; q < Nq; ++q) {
        const PetscReal xi0[3] = {-1., -1., -1.};

        /* Transform point to real space */
        CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
        for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
      }
      PetscCall(VecRestoreArray(pointVec, &pV));
      /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
      PetscCall(DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF));
      /* Update matrix */
      PetscCall(PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells));
      PetscCheck(numCoarseCells == Nq,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
      PetscCall(VecGetArray(pointVec, &pV));
      for (ccell = 0; ccell < numCoarseCells; ++ccell) {
        PetscReal pVReal[3];
        const PetscReal xi0[3] = {-1., -1., -1.};

        PetscCall(DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
        /* Transform points from real space to coarse reference space */
        PetscCall(DMPlexComputeCellGeometryFEM(dmc, coarseCells[ccell].index, NULL, v0c, Jc, invJc, &detJc));
        for (d = 0; d < dim; ++d) pVReal[d] = PetscRealPart(pV[ccell*dim+d]);
        CoordinatesRealToRef(dim, dim, xi0, v0c, invJc, pVReal, x);

        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          /* Evaluate coarse basis on contained point */
          PetscCall(PetscFEGetDimension(fe, &cpdim));
          PetscCall(PetscFEComputeTabulation(fe, 1, x, 0, T));
          /* Get elemMat entries by multiplying by weight */
          for (i = 0; i < numFIndices; ++i) {
            PetscCall(PetscArrayzero(elemMat, cpdim));
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[j] += T->T[0][j*Nc + c]*Tfine->T[0][(ccell*numFIndices + i)*Nc + c]*qweights[ccell*Nc + c]*detJ;
            }
            /* Update interpolator */
            if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat));
            PetscCheck(numCIndices == cpdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %" PetscInt_FMT " != %" PetscInt_FMT, numCIndices, cpdim);
            PetscCall(MatSetValues(mass, 1, &findices[i], numCIndices, cindices, elemMat, ADD_VALUES));
          }
        } else {
          cpdim = 1;
          for (i = 0; i < numFIndices; ++i) {
            PetscCall(PetscArrayzero(elemMat, cpdim));
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[j] += 1.0*1.0*qweights[ccell*Nc + c]*detJ;
            }
            /* Update interpolator */
            if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "Nq: %" PetscInt_FMT " %" PetscInt_FMT " Nf: %" PetscInt_FMT " %" PetscInt_FMT " Nc: %" PetscInt_FMT " %" PetscInt_FMT "\n", ccell, Nq, i, numFIndices, j, numCIndices));
            PetscCheck(numCIndices == cpdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %" PetscInt_FMT " != %" PetscInt_FMT, numCIndices, cpdim);
            PetscCall(MatSetValues(mass, 1, &findices[i], numCIndices, cindices, elemMat, ADD_VALUES));
          }
        }
        PetscCall(DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL));
      }
      PetscCall(VecRestoreArray(pointVec, &pV));
      PetscCall(PetscSFDestroy(&coarseCellSF));
      PetscCall(VecDestroy(&pointVec));
      PetscCall(DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL));
    }
    if (id == PETSCFE_CLASSID) PetscCall(PetscTabulationDestroy(&T));
  }
  PetscCall(PetscFree3(v0,J,invJ));
  PetscCall(PetscFree3(v0c,Jc,invJc));
  PetscCall(PetscFree(elemMat));
  PetscCall(MatAssemblyBegin(mass, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mass, MAT_FINAL_ASSEMBLY));
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

.seealso: `DMPlexComputeInterpolatorNested()`, `DMPlexComputeJacobianFEM()`
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
  PetscCall(PetscLogEventBegin(DMPLEX_InjectorFEM,dmc,dmf,0,0));
  PetscCall(DMGetDimension(dmf, &dim));
  PetscCall(DMGetLocalSection(dmf, &fsection));
  PetscCall(DMGetGlobalSection(dmf, &fglobalSection));
  PetscCall(DMGetLocalSection(dmc, &csection));
  PetscCall(DMGetGlobalSection(dmc, &cglobalSection));
  PetscCall(PetscSectionGetNumFields(fsection, &Nf));
  PetscCall(DMPlexGetSimplexOrBoxCells(dmc, 0, &cStart, &cEnd));
  PetscCall(DMGetDS(dmc, &prob));
  PetscCall(PetscCalloc3(Nf,&feRef,Nf,&fvRef,Nf,&needAvg));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     fNb = 0, Nc = 0;

    PetscCall(PetscDSGetDiscretization(prob, f, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE    fe = (PetscFE) obj;
      PetscSpace sp;
      PetscInt   maxDegree;

      PetscCall(PetscFERefine(fe, &feRef[f]));
      PetscCall(PetscFEGetDimension(feRef[f], &fNb));
      PetscCall(PetscFEGetNumComponents(fe, &Nc));
      PetscCall(PetscFEGetBasisSpace(fe, &sp));
      PetscCall(PetscSpaceGetDegree(sp, NULL, &maxDegree));
      if (!maxDegree) needAvg[f] = PETSC_TRUE;
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      PetscCall(PetscFVRefine(fv, &fvRef[f]));
      PetscCall(PetscFVGetDualSpace(fvRef[f], &Q));
      PetscCall(PetscDualSpaceGetDimension(Q, &fNb));
      PetscCall(PetscFVGetNumComponents(fv, &Nc));
      needAvg[f] = PETSC_TRUE;
    }
    fTotDim += fNb;
  }
  PetscCall(PetscDSGetTotalDimension(prob, &cTotDim));
  PetscCall(PetscMalloc1(cTotDim,&cmap));
  for (field = 0, offsetC = 0, offsetF = 0; field < Nf; ++field) {
    PetscFE        feC;
    PetscFV        fvC;
    PetscDualSpace QF, QC;
    PetscInt       order = -1, NcF, NcC, fpdim, cpdim;

    if (feRef[field]) {
      PetscCall(PetscDSGetDiscretization(prob, field, (PetscObject *) &feC));
      PetscCall(PetscFEGetNumComponents(feC, &NcC));
      PetscCall(PetscFEGetNumComponents(feRef[field], &NcF));
      PetscCall(PetscFEGetDualSpace(feRef[field], &QF));
      PetscCall(PetscDualSpaceGetOrder(QF, &order));
      PetscCall(PetscDualSpaceGetDimension(QF, &fpdim));
      PetscCall(PetscFEGetDualSpace(feC, &QC));
      PetscCall(PetscDualSpaceGetDimension(QC, &cpdim));
    } else {
      PetscCall(PetscDSGetDiscretization(prob, field, (PetscObject *) &fvC));
      PetscCall(PetscFVGetNumComponents(fvC, &NcC));
      PetscCall(PetscFVGetNumComponents(fvRef[field], &NcF));
      PetscCall(PetscFVGetDualSpace(fvRef[field], &QF));
      PetscCall(PetscDualSpaceGetDimension(QF, &fpdim));
      PetscCall(PetscFVGetDualSpace(fvC, &QC));
      PetscCall(PetscDualSpaceGetDimension(QC, &cpdim));
    }
    PetscCheck(NcF == NcC,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %" PetscInt_FMT " does not match coarse field %" PetscInt_FMT, NcF, NcC);
    for (c = 0; c < cpdim; ++c) {
      PetscQuadrature  cfunc;
      const PetscReal *cqpoints, *cqweights;
      PetscInt         NqcC, NpC;
      PetscBool        found = PETSC_FALSE;

      PetscCall(PetscDualSpaceGetFunctional(QC, c, &cfunc));
      PetscCall(PetscQuadratureGetData(cfunc, NULL, &NqcC, &NpC, &cqpoints, &cqweights));
      PetscCheck(NqcC == NcC,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of quadrature components %" PetscInt_FMT " must match number of field components %" PetscInt_FMT, NqcC, NcC);
      PetscCheck(NpC == 1 || !feRef[field],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not know how to do injection for moments");
      for (f = 0; f < fpdim; ++f) {
        PetscQuadrature  ffunc;
        const PetscReal *fqpoints, *fqweights;
        PetscReal        sum = 0.0;
        PetscInt         NqcF, NpF;

        PetscCall(PetscDualSpaceGetFunctional(QF, f, &ffunc));
        PetscCall(PetscQuadratureGetData(ffunc, NULL, &NqcF, &NpF, &fqpoints, &fqweights));
        PetscCheck(NqcF == NcF,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of quadrature components %" PetscInt_FMT " must match number of field components %" PetscInt_FMT, NqcF, NcF);
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
  for (f = 0; f < Nf; ++f) {PetscCall(PetscFEDestroy(&feRef[f]));PetscCall(PetscFVDestroy(&fvRef[f]));}
  PetscCall(PetscFree3(feRef,fvRef,needAvg));

  PetscCall(DMGetGlobalVector(dmf, &fv));
  PetscCall(DMGetGlobalVector(dmc, &cv));
  PetscCall(VecGetOwnershipRange(cv, &startC, &endC));
  PetscCall(PetscSectionGetConstrainedStorageSize(cglobalSection, &m));
  PetscCall(PetscMalloc2(cTotDim,&cellCIndices,fTotDim,&cellFIndices));
  PetscCall(PetscMalloc1(m,&cindices));
  PetscCall(PetscMalloc1(m,&findices));
  for (d = 0; d < m; ++d) cindices[d] = findices[d] = -1;
  for (c = cStart; c < cEnd; ++c) {
    PetscCall(DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, c, cellCIndices, cellFIndices));
    for (d = 0; d < cTotDim; ++d) {
      if ((cellCIndices[d] < startC) || (cellCIndices[d] >= endC)) continue;
      PetscCheck(!(findices[cellCIndices[d]-startC] >= 0) || !(findices[cellCIndices[d]-startC] != cellFIndices[cmap[d]]),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse dof %" PetscInt_FMT " maps to both %" PetscInt_FMT " and %" PetscInt_FMT, cindices[cellCIndices[d]-startC], findices[cellCIndices[d]-startC], cellFIndices[cmap[d]]);
      cindices[cellCIndices[d]-startC] = cellCIndices[d];
      findices[cellCIndices[d]-startC] = cellFIndices[cmap[d]];
    }
  }
  PetscCall(PetscFree(cmap));
  PetscCall(PetscFree2(cellCIndices,cellFIndices));

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, cindices, PETSC_OWN_POINTER, &cis));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, findices, PETSC_OWN_POINTER, &fis));
  PetscCall(VecScatterCreate(cv, cis, fv, fis, sc));
  PetscCall(ISDestroy(&cis));
  PetscCall(ISDestroy(&fis));
  PetscCall(DMRestoreGlobalVector(dmf, &fv));
  PetscCall(DMRestoreGlobalVector(dmc, &cv));
  PetscCall(PetscLogEventEnd(DMPLEX_InjectorFEM,dmc,dmf,0,0));
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

.seealso: `DMPlexGetFaceFields()`
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
  PetscCall(DMPlexConvertPlex(dm, &plex, PETSC_FALSE));
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &prob));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  if (locA) {
    DM      dmAux;
    PetscDS probAux;

    PetscCall(VecGetDM(locA, &dmAux));
    PetscCall(DMGetEnclosureRelation(dmAux, dm, &encAux));
    PetscCall(DMPlexConvertPlex(dmAux, &plexA, PETSC_FALSE));
    PetscCall(DMGetLocalSection(dmAux, &sectionAux));
    PetscCall(DMGetDS(dmAux, &probAux));
    PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
  }
  numCells = cEnd - cStart;
  PetscCall(DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, u));
  if (locX_t) PetscCall(DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, u_t)); else {*u_t = NULL;}
  if (locA)   PetscCall(DMGetWorkArray(dm, numCells*totDimAux, MPIU_SCALAR, a)); else {*a = NULL;}
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL, *x_t = NULL, *ul = *u, *ul_t = *u_t, *al = *a;
    PetscInt       i;

    PetscCall(DMPlexVecGetClosure(plex, section, locX, cell, NULL, &x));
    for (i = 0; i < totDim; ++i) ul[cind*totDim+i] = x[i];
    PetscCall(DMPlexVecRestoreClosure(plex, section, locX, cell, NULL, &x));
    if (locX_t) {
      PetscCall(DMPlexVecGetClosure(plex, section, locX_t, cell, NULL, &x_t));
      for (i = 0; i < totDim; ++i) ul_t[cind*totDim+i] = x_t[i];
      PetscCall(DMPlexVecRestoreClosure(plex, section, locX_t, cell, NULL, &x_t));
    }
    if (locA) {
      PetscInt subcell;
      PetscCall(DMGetEnclosurePoint(plexA, dm, encAux, cell, &subcell));
      PetscCall(DMPlexVecGetClosure(plexA, sectionAux, locA, subcell, NULL, &x));
      for (i = 0; i < totDimAux; ++i) al[cind*totDimAux+i] = x[i];
      PetscCall(DMPlexVecRestoreClosure(plexA, sectionAux, locA, subcell, NULL, &x));
    }
  }
  PetscCall(DMDestroy(&plex));
  if (locA) PetscCall(DMDestroy(&plexA));
  PetscCall(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
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

.seealso: `DMPlexGetFaceFields()`
@*/
PetscErrorCode DMPlexRestoreCellFields(DM dm, IS cellIS, Vec locX, Vec locX_t, Vec locA, PetscScalar **u, PetscScalar **u_t, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, u));
  if (locX_t) PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, u_t));
  if (locA)   PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, a));
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
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  numCells = cEnd - cStart;
  for (s = 0; s < 2; ++s) {
    PetscValidHeaderSpecific(dmAux[s], DM_CLASSID, 2);
    PetscValidHeaderSpecific(dsAux[s], PETSCDS_CLASSID, 3);
    PetscValidHeaderSpecific(locA[s], VEC_CLASSID, 5);
    PetscCall(DMPlexConvertPlex(dmAux[s], &plexA[s], PETSC_FALSE));
    PetscCall(DMGetEnclosureRelation(dmAux[s], dm, &encAux[s]));
    PetscCall(DMGetLocalSection(dmAux[s], &sectionAux[s]));
    PetscCall(PetscDSGetTotalDimension(dsAux[s], &totDimAux[s]));
    PetscCall(DMGetWorkArray(dmAux[s], numCells*totDimAux[s], MPIU_SCALAR, &a[s]));
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt  cell = cells ? cells[c] : c;
    const PetscInt  cind = c - cStart;
    const PetscInt *cone, *ornt;

    PetscCall(DMPlexGetCone(dm, cell, &cone));
    PetscCall(DMPlexGetConeOrientation(dm, cell, &ornt));
    for (s = 0; s < 2; ++s) {
      const PetscInt *support;
      PetscScalar    *x = NULL, *al = a[s];
      const PetscInt  tdA = totDimAux[s];
      PetscInt        ssize, scell;
      PetscInt        subface, Na, i;

      PetscCall(DMPlexGetSupport(dm, cone[s], &support));
      PetscCall(DMPlexGetSupportSize(dm, cone[s], &ssize));
      PetscCheck(ssize == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " from cell %" PetscInt_FMT " has support size %" PetscInt_FMT " != 2", cone[s], cell, ssize);
      if      (support[0] == cell) scell = support[1];
      else if (support[1] == cell) scell = support[0];
      else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " does not have cell %" PetscInt_FMT " in its support", cone[s], cell);

      PetscCall(DMGetEnclosurePoint(plexA[s], dm, encAux[s], scell, &subface));
      PetscCall(DMPlexVecGetClosure(plexA[s], sectionAux[s], locA[s], subface, &Na, &x));
      for (i = 0; i < Na; ++i) al[cind*tdA+i] = x[i];
      PetscCall(DMPlexVecRestoreClosure(plexA[s], sectionAux[s], locA[s], subface, &Na, &x));
    }
  }
  for (s = 0; s < 2; ++s) PetscCall(DMDestroy(&plexA[s]));
  PetscCall(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexRestoreHybridAuxFields(DM dmAux[], PetscDS dsAux[], IS cellIS, Vec locA[], PetscScalar *a[])
{
  PetscFunctionBegin;
  if (!locA[0] || !locA[1]) PetscFunctionReturn(0);
  PetscCall(DMRestoreWorkArray(dmAux[0], 0, MPIU_SCALAR, &a[0]));
  PetscCall(DMRestoreWorkArray(dmAux[1], 0, MPIU_SCALAR, &a[1]));
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

.seealso: `DMPlexGetCellFields()`
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscDSGetTotalComponents(prob, &Nc));
  PetscCall(PetscMalloc1(Nf, &isFE));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(PetscDSGetDiscretization(prob, f, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else                            {isFE[f] = PETSC_FALSE;}
  }
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(VecGetArrayRead(locX, &x));
  PetscCall(VecGetDM(faceGeometry, &dmFace));
  PetscCall(VecGetArrayRead(faceGeometry, &facegeom));
  PetscCall(VecGetDM(cellGeometry, &dmCell));
  PetscCall(VecGetArrayRead(cellGeometry, &cellgeom));
  if (locGrad) {
    PetscCall(VecGetDM(locGrad, &dmGrad));
    PetscCall(VecGetArrayRead(locGrad, &lgrad));
  }
  PetscCall(DMGetWorkArray(dm, numFaces*Nc, MPIU_SCALAR, uL));
  PetscCall(DMGetWorkArray(dm, numFaces*Nc, MPIU_SCALAR, uR));
  /* Right now just eat the extra work for FE (could make a cell loop) */
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscFVCellGeom       *cgL, *cgR;
    PetscScalar           *xL, *xR, *gL, *gR;
    PetscScalar           *uLl = *uL, *uRl = *uR;
    PetscInt               ghost, nsupp, nchild;

    PetscCall(DMLabelGetValue(ghostLabel, face, &ghost));
    PetscCall(DMPlexGetSupportSize(dm, face, &nsupp));
    PetscCall(DMPlexGetTreeChildren(dm, face, &nchild, NULL));
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
    PetscCall(DMPlexPointLocalRead(dmFace, face, facegeom, &fg));
    PetscCall(DMPlexGetSupport(dm, face, &cells));
    PetscCall(DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL));
    PetscCall(DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR));
    for (f = 0; f < Nf; ++f) {
      PetscInt off;

      PetscCall(PetscDSGetComponentOffset(prob, f, &off));
      if (isFE[f]) {
        const PetscInt *cone;
        PetscInt        comp, coneSizeL, coneSizeR, faceLocL, faceLocR, ldof, rdof, d;

        xL = xR = NULL;
        PetscCall(PetscSectionGetFieldComponents(section, f, &comp));
        PetscCall(DMPlexVecGetClosure(dm, section, locX, cells[0], &ldof, (PetscScalar **) &xL));
        PetscCall(DMPlexVecGetClosure(dm, section, locX, cells[1], &rdof, (PetscScalar **) &xR));
        PetscCall(DMPlexGetCone(dm, cells[0], &cone));
        PetscCall(DMPlexGetConeSize(dm, cells[0], &coneSizeL));
        for (faceLocL = 0; faceLocL < coneSizeL; ++faceLocL) if (cone[faceLocL] == face) break;
        PetscCall(DMPlexGetCone(dm, cells[1], &cone));
        PetscCall(DMPlexGetConeSize(dm, cells[1], &coneSizeR));
        for (faceLocR = 0; faceLocR < coneSizeR; ++faceLocR) if (cone[faceLocR] == face) break;
        PetscCheck(faceLocL != coneSizeL || faceLocR != coneSizeR,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %" PetscInt_FMT " in cone of cell %" PetscInt_FMT " or cell %" PetscInt_FMT, face, cells[0], cells[1]);
        /* Check that FEM field has values in the right cell (sometimes its an FV ghost cell) */
        /* TODO: this is a hack that might not be right for nonconforming */
        if (faceLocL < coneSizeL) {
          PetscCall(PetscFEEvaluateFaceFields_Internal(prob, f, faceLocL, xL, &uLl[iface*Nc+off]));
          if (rdof == ldof && faceLocR < coneSizeR) PetscCall(PetscFEEvaluateFaceFields_Internal(prob, f, faceLocR, xR, &uRl[iface*Nc+off]));
          else              {for (d = 0; d < comp; ++d) uRl[iface*Nc+off+d] = uLl[iface*Nc+off+d];}
        }
        else {
          PetscCall(PetscFEEvaluateFaceFields_Internal(prob, f, faceLocR, xR, &uRl[iface*Nc+off]));
          PetscCall(PetscSectionGetFieldComponents(section, f, &comp));
          for (d = 0; d < comp; ++d) uLl[iface*Nc+off+d] = uRl[iface*Nc+off+d];
        }
        PetscCall(DMPlexVecRestoreClosure(dm, section, locX, cells[0], &ldof, (PetscScalar **) &xL));
        PetscCall(DMPlexVecRestoreClosure(dm, section, locX, cells[1], &rdof, (PetscScalar **) &xR));
      } else {
        PetscFV  fv;
        PetscInt numComp, c;

        PetscCall(PetscDSGetDiscretization(prob, f, (PetscObject *) &fv));
        PetscCall(PetscFVGetNumComponents(fv, &numComp));
        PetscCall(DMPlexPointLocalFieldRead(dm, cells[0], f, x, &xL));
        PetscCall(DMPlexPointLocalFieldRead(dm, cells[1], f, x, &xR));
        if (dmGrad) {
          PetscReal dxL[3], dxR[3];

          PetscCall(DMPlexPointLocalRead(dmGrad, cells[0], lgrad, &gL));
          PetscCall(DMPlexPointLocalRead(dmGrad, cells[1], lgrad, &gR));
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
  PetscCall(VecRestoreArrayRead(locX, &x));
  PetscCall(VecRestoreArrayRead(faceGeometry, &facegeom));
  PetscCall(VecRestoreArrayRead(cellGeometry, &cellgeom));
  if (locGrad) {
    PetscCall(VecRestoreArrayRead(locGrad, &lgrad));
  }
  PetscCall(PetscFree(isFE));
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

.seealso: `DMPlexGetFaceFields()`
@*/
PetscErrorCode DMPlexRestoreFaceFields(DM dm, PetscInt fStart, PetscInt fEnd, Vec locX, Vec locX_t, Vec faceGeometry, Vec cellGeometry, Vec locGrad, PetscInt *Nface, PetscScalar **uL, PetscScalar **uR)
{
  PetscFunctionBegin;
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, uL));
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, uR));
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

.seealso: `DMPlexGetCellFields()`
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(VecGetDM(faceGeometry, &dmFace));
  PetscCall(VecGetArrayRead(faceGeometry, &facegeom));
  PetscCall(VecGetDM(cellGeometry, &dmCell));
  PetscCall(VecGetArrayRead(cellGeometry, &cellgeom));
  PetscCall(PetscMalloc1(numFaces, fgeom));
  PetscCall(DMGetWorkArray(dm, numFaces*2, MPIU_SCALAR, vol));
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscFVCellGeom       *cgL, *cgR;
    PetscFVFaceGeom       *fgeoml = *fgeom;
    PetscReal             *voll   = *vol;
    PetscInt               ghost, d, nchild, nsupp;

    PetscCall(DMLabelGetValue(ghostLabel, face, &ghost));
    PetscCall(DMPlexGetSupportSize(dm, face, &nsupp));
    PetscCall(DMPlexGetTreeChildren(dm, face, &nchild, NULL));
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
    PetscCall(DMPlexPointLocalRead(dmFace, face, facegeom, &fg));
    PetscCall(DMPlexGetSupport(dm, face, &cells));
    PetscCall(DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL));
    PetscCall(DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR));
    for (d = 0; d < dim; ++d) {
      fgeoml[iface].centroid[d] = fg->centroid[d];
      fgeoml[iface].normal[d]   = fg->normal[d];
    }
    voll[iface*2+0] = cgL->volume;
    voll[iface*2+1] = cgR->volume;
    ++iface;
  }
  *Nface = iface;
  PetscCall(VecRestoreArrayRead(faceGeometry, &facegeom));
  PetscCall(VecRestoreArrayRead(cellGeometry, &cellgeom));
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

.seealso: `DMPlexGetFaceFields()`
@*/
PetscErrorCode DMPlexRestoreFaceGeometry(DM dm, PetscInt fStart, PetscInt fEnd, Vec faceGeometry, Vec cellGeometry, PetscInt *Nface, PetscFVFaceGeom **fgeom, PetscReal **vol)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*fgeom));
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_REAL, vol));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESGetFEGeom(DMField coordField, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)quad,&id));
  PetscCall(PetscSNPrintf(composeStr, 32, "DMSNESGetFEGeom_%" PetscInt64_FMT "\n", id));
  PetscCall(PetscObjectQuery((PetscObject) pointIS, composeStr, (PetscObject *) &container));
  if (container) {
    PetscCall(PetscContainerGetPointer(container, (void **) geom));
  } else {
    PetscCall(DMFieldCreateFEGeom(coordField, pointIS, quad, faceData, geom));
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF,&container));
    PetscCall(PetscContainerSetPointer(container, (void *) *geom));
    PetscCall(PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom));
    PetscCall(PetscObjectCompose((PetscObject) pointIS, composeStr, (PetscObject) container));
    PetscCall(PetscContainerDestroy(&container));
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
  PetscCall(PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0));
  /* FEM+FVM */
  /* 1: Get sizes from dm and dmAux */
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &locA));
  if (locA) {
    PetscCall(VecGetDM(locA, &dmAux));
    PetscCall(DMGetDS(dmAux, &probAux));
    PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
  }
  /* 2: Get geometric data */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscBool    fimp;

    PetscCall(PetscDSGetImplicit(prob, f, &fimp));
    if (isImplicit != fimp) continue;
    PetscCall(PetscDSGetDiscretization(prob, f, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {useFEM = PETSC_TRUE;}
    PetscCheck(id != PETSCFV_CLASSID,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Use of FVM with PCPATCH not yet implemented");
  }
  if (useFEM) {
    PetscCall(DMGetCoordinateField(dm, &coordField));
    PetscCall(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      PetscCall(DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad));
      if (affineQuad) {
        PetscCall(DMSNESGetFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
      }
    } else {
      PetscCall(PetscCalloc2(Nf,&quads,Nf,&geoms));
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscBool    fimp;

        PetscCall(PetscDSGetImplicit(prob, f, &fimp));
        if (isImplicit != fimp) continue;
        PetscCall(PetscDSGetDiscretization(prob, f, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          PetscCall(PetscFEGetQuadrature(fe, &quads[f]));
          PetscCall(PetscObjectReference((PetscObject)quads[f]));
          PetscCall(DMSNESGetFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
        }
      }
    }
  }
  /* Loop over chunks */
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  if (useFEM) PetscCall(ISCreate(PETSC_COMM_SELF, &chunkIS));
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
      PetscCall(ISGetPointSubrange(chunkIS, cS, cE, cells));
      PetscCall(DMPlexGetCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a));
      PetscCall(DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
      PetscCall(PetscArrayzero(elemVec, numCells*totDim));
    }
    /* TODO We will interlace both our field coefficients (u, u_t, uL, uR, etc.) and our output (elemVec, fL, fR). I think this works */
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;
      PetscBool    fimp;
      PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

      key.field = f;
      PetscCall(PetscDSGetImplicit(prob, f, &fimp));
      if (isImplicit != fimp) continue;
      PetscCall(PetscDSGetDiscretization(prob, f, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) {
        PetscFE         fe = (PetscFE) obj;
        PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
        PetscFEGeom    *chunkGeom = NULL;
        PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
        PetscInt        Nq, Nb;

        PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
        PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
        PetscCall(PetscFEGetDimension(fe, &Nb));
        blockSize = Nb;
        batchSize = numBlocks * blockSize;
        PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
        numChunks = numCells / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numCells % (numBatches*batchSize);
        offset    = numCells - Nr;
        /* Integrate FE residual to get elemVec (need fields at quadrature points) */
        /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
        PetscCall(PetscFEGeomGetChunk(geom,0,offset,&chunkGeom));
        PetscCall(PetscFEIntegrateResidual(prob, key, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec));
        PetscCall(PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom));
        PetscCall(PetscFEIntegrateResidual(prob, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]));
        PetscCall(PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom));
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        Ne = numFaces;
        /* Riemann solve over faces (need fields at face centroids) */
        /*   We need to evaluate FE fields at those coordinates */
        PetscCall(PetscFVIntegrateRHSFunction(fv, prob, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR));
      } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, f);
    }
    /* Loop over domain */
    if (useFEM) {
      /* Add elemVec to locX */
      for (c = cS; c < cE; ++c) {
        const PetscInt cell = cells ? cells[c] : c;
        const PetscInt cind = c - cStart;

        if (mesh->printFEM > 1) PetscCall(DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]));
        if (ghostLabel) {
          PetscInt ghostVal;

          PetscCall(DMLabelGetValue(ghostLabel,cell,&ghostVal));
          if (ghostVal > 0) continue;
        }
        PetscCall(DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES));
      }
    }
    /* Handle time derivative */
    if (locX_t) {
      PetscScalar *x_t, *fa;

      PetscCall(VecGetArray(locF, &fa));
      PetscCall(VecGetArray(locX_t, &x_t));
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     pdim, d;

        PetscCall(PetscDSGetDiscretization(prob, f, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        PetscCall(PetscFVGetNumComponents(fv, &pdim));
        for (c = cS; c < cE; ++c) {
          const PetscInt cell = cells ? cells[c] : c;
          PetscScalar   *u_t, *r;

          if (ghostLabel) {
            PetscInt ghostVal;

            PetscCall(DMLabelGetValue(ghostLabel, cell, &ghostVal));
            if (ghostVal > 0) continue;
          }
          PetscCall(DMPlexPointLocalFieldRead(dm, cell, f, x_t, &u_t));
          PetscCall(DMPlexPointLocalFieldRef(dm, cell, f, fa, &r));
          for (d = 0; d < pdim; ++d) r[d] += u_t[d];
        }
      }
      PetscCall(VecRestoreArray(locX_t, &x_t));
      PetscCall(VecRestoreArray(locF, &fa));
    }
    if (useFEM) {
      PetscCall(DMPlexRestoreCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a));
      PetscCall(DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
    }
  }
  if (useFEM) PetscCall(ISDestroy(&chunkIS));
  PetscCall(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  /* TODO Could include boundary residual here (see DMPlexComputeResidual_Internal) */
  if (useFEM) {
    if (maxDegree <= 1) {
      PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
      PetscCall(PetscQuadratureDestroy(&affineQuad));
    } else {
      for (f = 0; f < Nf; ++f) {
        PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
        PetscCall(PetscQuadratureDestroy(&quads[f]));
      }
      PetscCall(PetscFree2(quads,geoms));
    }
  }
  PetscCall(PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0));
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
  PetscCall(ISGetLocalSize(cellIS, &numCells));
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &A));
  if (A) {
    PetscCall(VecGetDM(A, &dmAux));
    PetscCall(DMGetLocalSection(dmAux, &sectionAux));
    PetscCall(DMGetDS(dmAux, &probAux));
  }
  /* Get flags */
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(DMGetWorkArray(dm, Nf, MPIU_BOOL, &isFE));
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscObject  disc;
    PetscClassId id;
    PetscCall(PetscDSGetDiscretization(prob, fieldI, &disc));
    PetscCall(PetscObjectGetClassId(disc, &id));
    if (id == PETSCFE_CLASSID)      {isFE[fieldI] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {hasFV = PETSC_TRUE; isFE[fieldI] = PETSC_FALSE;}
  }
  PetscCall(PetscDSHasJacobian(prob, &hasJac));
  PetscCall(PetscDSHasJacobianPreconditioner(prob, &hasPrec));
  PetscCall(PetscDSHasDynamicJacobian(prob, &hasDyn));
  assembleJac = hasJac && hasPrec && (Jac != JacP) ? PETSC_TRUE : PETSC_FALSE;
  hasDyn      = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  if (hasFV) PetscCall(MatSetOption(JP, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE)); /* No allocated space for FV stuff, so ignore the zero entries */
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  if (probAux) PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
  /* Compute batch sizes */
  if (isFE[0]) {
    PetscFE         fe;
    PetscQuadrature q;
    PetscInt        numQuadPoints, numBatches, batchSize, numBlocks, blockSize, Nb;

    PetscCall(PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe));
    PetscCall(PetscFEGetQuadrature(fe, &q));
    PetscCall(PetscQuadratureGetData(q, NULL, NULL, &numQuadPoints, NULL, NULL));
    PetscCall(PetscFEGetDimension(fe, &Nb));
    PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
    blockSize = Nb*numQuadPoints;
    batchSize = numBlocks  * blockSize;
    chunkSize = numBatches * batchSize;
    numChunks = numCells / chunkSize + numCells % chunkSize;
    PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
  } else {
    chunkSize = numCells;
    numChunks = 1;
  }
  /* Get work space */
  wsz  = (((X?1:0) + (X_t?1:0) + (dmAux?1:0))*totDim + ((hasJac?1:0) + (hasPrec?1:0) + (hasDyn?1:0))*totDim*totDim)*chunkSize;
  PetscCall(DMGetWorkArray(dm, wsz, MPIU_SCALAR, &work));
  PetscCall(PetscArrayzero(work, wsz));
  off      = 0;
  u        = X       ? (sz = chunkSize*totDim,        off += sz, work+off-sz) : NULL;
  u_t      = X_t     ? (sz = chunkSize*totDim,        off += sz, work+off-sz) : NULL;
  a        = dmAux   ? (sz = chunkSize*totDimAux,     off += sz, work+off-sz) : NULL;
  elemMat  = hasJac  ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  elemMatP = hasPrec ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  elemMatD = hasDyn  ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  PetscCheck(off == wsz,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error is workspace size %" PetscInt_FMT " should be %" PetscInt_FMT, off, wsz);
  /* Setup geometry */
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree));
  if (maxDegree <= 1) PetscCall(DMFieldCreateDefaultQuadrature(coordField, cellIS, &qGeom));
  if (!qGeom) {
    PetscFE fe;

    PetscCall(PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe));
    PetscCall(PetscFEGetQuadrature(fe, &qGeom));
    PetscCall(PetscObjectReference((PetscObject) qGeom));
  }
  PetscCall(DMSNESGetFEGeom(coordField, cellIS, qGeom, PETSC_FALSE, &cgeomFEM));
  /* Compute volume integrals */
  if (assembleJac) PetscCall(MatZeroEntries(J));
  PetscCall(MatZeroEntries(JP));
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
        PetscCall(DMPlexVecGetClosure(dm, section, X, cell, NULL, &x));
        for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
        PetscCall(DMPlexVecRestoreClosure(dm, section, X, cell, NULL, &x));
      }
      if (X_t) {
        PetscCall(DMPlexVecGetClosure(dm, section, X_t, cell, NULL, &x_t));
        for (i = 0; i < totDim; ++i) u_t[c*totDim+i] = x_t[i];
        PetscCall(DMPlexVecRestoreClosure(dm, section, X_t, cell, NULL, &x_t));
      }
      if (dmAux) {
        PetscCall(DMPlexVecGetClosure(dmAux, sectionAux, A, cell, NULL, &x));
        for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
        PetscCall(DMPlexVecRestoreClosure(dmAux, sectionAux, A, cell, NULL, &x));
      }
    }
    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscFE fe;
      PetscCall(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf + fieldJ;
        if (hasJac)  PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN,     key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMat));
        if (hasPrec) PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMatP));
        if (hasDyn)  PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMatD));
      }
      /* For finite volume, add the identity */
      if (!isFE[fieldI]) {
        PetscFV  fv;
        PetscInt eOffset = 0, Nc, fc, foff;

        PetscCall(PetscDSGetFieldOffset(prob, fieldI, &foff));
        PetscCall(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fv));
        PetscCall(PetscFVGetNumComponents(fv, &Nc));
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
        if (hasJac)  PetscCall(DMPrintCellMatrix(cell, name,  totDim, totDim, &elemMat[(c-cStart)*totDim*totDim]));
        if (hasPrec) PetscCall(DMPrintCellMatrix(cell, nameP, totDim, totDim, &elemMatP[(c-cStart)*totDim*totDim]));
      }
      if (assembleJac) PetscCall(DMPlexMatSetClosure(dm, section, globalSection, Jac, cell, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES));
      PetscCall(DMPlexMatSetClosure(dm, section, globalSection, JP, cell, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES));
    }
  }
  /* Cleanup */
  PetscCall(DMSNESRestoreFEGeom(coordField, cellIS, qGeom, PETSC_FALSE, &cgeomFEM));
  PetscCall(PetscQuadratureDestroy(&qGeom));
  if (hasFV) PetscCall(MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  PetscCall(DMRestoreWorkArray(dm, Nf, MPIU_BOOL, &isFE));
  PetscCall(DMRestoreWorkArray(dm, ((1 + (X_t?1:0) + (dmAux?1:0))*totDim + ((hasJac?1:0) + (hasPrec?1:0) + (hasDyn?1:0))*totDim*totDim)*chunkSize, MPIU_SCALAR, &work));
  /* Compute boundary integrals */
  /* PetscCall(DMPlexComputeBdJacobian_Internal(dm, X, X_t, t, X_tShift, Jac, JacP, ctx)); */
  /* Assemble matrix */
  if (assembleJac) {PetscCall(MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY));}
  PetscCall(MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}

/******** FEM Assembly Function ********/

static PetscErrorCode DMConvertPlex_Internal(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex));
  if (isPlex) {
    *plex = dm;
    PetscCall(PetscObjectReference((PetscObject) dm));
  } else {
    PetscCall(PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex));
    if (!*plex) {
      PetscCall(DMConvert(dm,DMPLEX,plex));
      PetscCall(PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex));
      if (copy) {
        PetscCall(DMCopyAuxiliaryVec(dm, *plex));
      }
    } else {
      PetscCall(PetscObjectReference((PetscObject) *plex));
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

.seealso: `DMTSSetRHSFunctionLocal()`
@*/
PetscErrorCode DMPlexGetGeometryFVM(DM dm, Vec *facegeom, Vec *cellgeom, PetscReal *minRadius)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMConvertPlex_Internal(dm,&plex,PETSC_TRUE));
  PetscCall(DMPlexGetDataFVM(plex, NULL, cellgeom, facegeom, NULL));
  if (minRadius) PetscCall(DMPlexGetMinRadius(plex, minRadius));
  PetscCall(DMDestroy(&plex));
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

.seealso: `DMPlexGetGeometryFVM()`
@*/
PetscErrorCode DMPlexGetGradientDM(DM dm, PetscFV fv, DM *dmGrad)
{
  DM             plex;
  PetscBool      computeGradients;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(fv,PETSCFV_CLASSID,2);
  PetscValidPointer(dmGrad,3);
  PetscCall(PetscFVGetComputeGradients(fv, &computeGradients));
  if (!computeGradients) {*dmGrad = NULL; PetscFunctionReturn(0);}
  PetscCall(DMConvertPlex_Internal(dm,&plex,PETSC_TRUE));
  PetscCall(DMPlexGetDataFVM(plex, fv, NULL, NULL, dmGrad));
  PetscCall(DMDestroy(&plex));
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
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &locA));
  if (locA) {
    DM dmAux;

    PetscCall(VecGetDM(locA, &dmAux));
    PetscCall(DMGetEnclosureRelation(dmAux, dm, &encAux));
    PetscCall(DMConvert(dmAux, DMPLEX, &plexA));
    PetscCall(DMGetDS(plexA, &probAux));
    PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
    PetscCall(DMGetLocalSection(plexA, &sectionAux));
  }
  {
    PetscFEGeom     *fgeom;
    PetscInt         maxDegree;
    PetscQuadrature  qGeom = NULL;
    IS               pointIS;
    const PetscInt  *points;
    PetscInt         numFaces, face, Nq;

    PetscCall(DMLabelGetStratumIS(key.label, key.value, &pointIS));
    if (!pointIS) goto end; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a priori if one is a superset of the other */
      PetscCall(ISIntersect_Caching_Internal(facetIS,pointIS,&isectIS));
      PetscCall(ISDestroy(&pointIS));
      pointIS = isectIS;
    }
    PetscCall(ISGetLocalSize(pointIS,&numFaces));
    PetscCall(ISGetIndices(pointIS,&points));
    PetscCall(PetscMalloc4(numFaces*totDim, &u, locX_t ? numFaces*totDim : 0, &u_t, numFaces*totDim, &elemVec, locA ? numFaces*totDimAux : 0, &a));
    PetscCall(DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      PetscCall(DMFieldCreateDefaultQuadrature(coordField,pointIS,&qGeom));
    }
    if (!qGeom) {
      PetscFE fe;

      PetscCall(PetscDSGetDiscretization(prob, key.field, (PetscObject *) &fe));
      PetscCall(PetscFEGetFaceQuadrature(fe, &qGeom));
      PetscCall(PetscObjectReference((PetscObject)qGeom));
    }
    PetscCall(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
    PetscCall(DMSNESGetFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom));
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;
      PetscScalar   *x     = NULL;
      PetscInt       i;

      PetscCall(DMPlexGetSupport(dm, point, &support));
      PetscCall(DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x));
      for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
      PetscCall(DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x));
      if (locX_t) {
        PetscCall(DMPlexVecGetClosure(plex, section, locX_t, support[0], NULL, &x));
        for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
        PetscCall(DMPlexVecRestoreClosure(plex, section, locX_t, support[0], NULL, &x));
      }
      if (locA) {
        PetscInt subp;

        PetscCall(DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp));
        PetscCall(DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x));
        for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
        PetscCall(DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x));
      }
    }
    PetscCall(PetscArrayzero(elemVec, numFaces*totDim));
    {
      PetscFE         fe;
      PetscInt        Nb;
      PetscFEGeom     *chunkGeom = NULL;
      /* Conforming batches */
      PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscInt        Nr, offset;

      PetscCall(PetscDSGetDiscretization(prob, key.field, (PetscObject *) &fe));
      PetscCall(PetscFEGetDimension(fe, &Nb));
      PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
      /* TODO: documentation is unclear about what is going on with these numbers: how should Nb / Nq factor in ? */
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      PetscCall(PetscFEGeomGetChunk(fgeom,0,offset,&chunkGeom));
      PetscCall(PetscFEIntegrateBdResidual(prob, wf, key, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec));
      PetscCall(PetscFEGeomRestoreChunk(fgeom, 0, offset, &chunkGeom));
      PetscCall(PetscFEGeomGetChunk(fgeom,offset,numFaces,&chunkGeom));
      PetscCall(PetscFEIntegrateBdResidual(prob, wf, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, &elemVec[offset*totDim]));
      PetscCall(PetscFEGeomRestoreChunk(fgeom,offset,numFaces,&chunkGeom));
    }
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;

      if (mesh->printFEM > 1) PetscCall(DMPrintCellVector(point, "BdResidual", totDim, &elemVec[face*totDim]));
      PetscCall(DMPlexGetSupport(plex, point, &support));
      PetscCall(DMPlexVecSetClosure(plex, NULL, locF, support[0], &elemVec[face*totDim], ADD_ALL_VALUES));
    }
    PetscCall(DMSNESRestoreFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom));
    PetscCall(PetscQuadratureDestroy(&qGeom));
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
    PetscCall(PetscFree4(u, u_t, elemVec, a));
  }
  end:
  PetscCall(DMDestroy(&plex));
  PetscCall(DMDestroy(&plexA));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdResidualSingle(DM dm, PetscReal t, PetscWeakForm wf, PetscFormKey key, Vec locX, Vec locX_t, Vec locF)
{
  DMField        coordField;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMLabelGetStratumIS(depthLabel, dim-1, &facetIS));
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMPlexComputeBdResidual_Single_Internal(dm, t, wf, key, locX, locX_t, locF, coordField, facetIS));
  PetscCall(ISDestroy(&facetIS));
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
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMLabelGetStratumIS(depthLabel,dim - 1,&facetIS));
  PetscCall(PetscDSGetNumBoundary(prob, &numBd));
  for (bd = 0; bd < numBd; ++bd) {
    PetscWeakForm           wf;
    DMBoundaryConditionType type;
    DMLabel                 label;
    const PetscInt         *values;
    PetscInt                field, numValues, v;
    PetscObject             obj;
    PetscClassId            id;
    PetscFormKey            key;

    PetscCall(PetscDSGetBoundary(prob, bd, &wf, &type, NULL, &label, &numValues, &values, &field, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscDSGetDiscretization(prob, field, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    if (!facetIS) {
      DMLabel  depthLabel;
      PetscInt dim;

      PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
      PetscCall(DMGetDimension(dm, &dim));
      PetscCall(DMLabelGetStratumIS(depthLabel, dim - 1, &facetIS));
    }
    PetscCall(DMGetCoordinateField(dm, &coordField));
    for (v = 0; v < numValues; ++v) {
      key.label = label;
      key.value = values[v];
      key.field = field;
      key.part  = 0;
      PetscCall(DMPlexComputeBdResidual_Single_Internal(dm, t, wf, key, locX, locX_t, locF, coordField, facetIS));
    }
  }
  PetscCall(ISDestroy(&facetIS));
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
  if (!cellIS) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0));
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* TODO The FVM geometry is over-manipulated. Make the precalc functions return exactly what we need */
  /* FEM+FVM */
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  /* 1: Get sizes from dm and dmAux */
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));
  PetscCall(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &locA));
  if (locA) {
    PetscInt subcell;
    PetscCall(VecGetDM(locA, &dmAux));
    PetscCall(DMGetEnclosurePoint(dmAux, dm, DM_ENC_UNKNOWN, cells ? cells[cStart] : cStart, &subcell));
    PetscCall(DMGetCellDS(dmAux, subcell, &dsAux));
    PetscCall(PetscDSGetTotalDimension(dsAux, &totDimAux));
  }
  /* 2: Get geometric data */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscBool    fimp;

    PetscCall(PetscDSGetImplicit(ds, f, &fimp));
    if (isImplicit != fimp) continue;
    PetscCall(PetscDSGetDiscretization(ds, f, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {useFEM = PETSC_TRUE;}
    if (id == PETSCFV_CLASSID) {useFVM = PETSC_TRUE; fvm = (PetscFV) obj;}
  }
  if (useFEM) {
    PetscCall(DMGetCoordinateField(dm, &coordField));
    PetscCall(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      PetscCall(DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad));
      if (affineQuad) {
        PetscCall(DMSNESGetFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
      }
    } else {
      PetscCall(PetscCalloc2(Nf,&quads,Nf,&geoms));
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscBool    fimp;

        PetscCall(PetscDSGetImplicit(ds, f, &fimp));
        if (isImplicit != fimp) continue;
        PetscCall(PetscDSGetDiscretization(ds, f, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          PetscCall(PetscFEGetQuadrature(fe, &quads[f]));
          PetscCall(PetscObjectReference((PetscObject)quads[f]));
          PetscCall(DMSNESGetFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
        }
      }
    }
  }
  if (useFVM) {
    PetscCall(DMPlexGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL));
    PetscCall(VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM));
    PetscCall(VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM));
    /* Reconstruct and limit cell gradients */
    PetscCall(DMPlexGetGradientDM(dm, fvm, &dmGrad));
    if (dmGrad) {
      PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
      PetscCall(DMGetGlobalVector(dmGrad, &grad));
      PetscCall(DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad));
      /* Communicate gradient values */
      PetscCall(DMGetLocalVector(dmGrad, &locGrad));
      PetscCall(DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad));
      PetscCall(DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad));
      PetscCall(DMRestoreGlobalVector(dmGrad, &grad));
    }
    /* Handle non-essential (e.g. outflow) boundary values */
    PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, time, faceGeometryFVM, cellGeometryFVM, locGrad));
  }
  /* Loop over chunks */
  if (useFEM) PetscCall(ISCreate(PETSC_COMM_SELF, &chunkIS));
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
      PetscCall(ISGetPointSubrange(chunkIS, cS, cE, cells));
      PetscCall(DMPlexGetCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a));
      PetscCall(DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
      PetscCall(PetscArrayzero(elemVec, numCells*totDim));
    }
    if (useFVM) {
      PetscCall(DMPlexGetFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR));
      PetscCall(DMPlexGetFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol));
      PetscCall(DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL));
      PetscCall(DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR));
      PetscCall(PetscArrayzero(fluxL, numFaces*totDim));
      PetscCall(PetscArrayzero(fluxR, numFaces*totDim));
    }
    /* TODO We will interlace both our field coefficients (u, u_t, uL, uR, etc.) and our output (elemVec, fL, fR). I think this works */
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;
      PetscBool    fimp;
      PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

      key.field = f;
      PetscCall(PetscDSGetImplicit(ds, f, &fimp));
      if (isImplicit != fimp) continue;
      PetscCall(PetscDSGetDiscretization(ds, f, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFE_CLASSID) {
        PetscFE         fe = (PetscFE) obj;
        PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
        PetscFEGeom    *chunkGeom = NULL;
        PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
        PetscInt        Nq, Nb;

        PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
        PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
        PetscCall(PetscFEGetDimension(fe, &Nb));
        blockSize = Nb;
        batchSize = numBlocks * blockSize;
        PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
        numChunks = numCells / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numCells % (numBatches*batchSize);
        offset    = numCells - Nr;
        /* Integrate FE residual to get elemVec (need fields at quadrature points) */
        /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
        PetscCall(PetscFEGeomGetChunk(geom,0,offset,&chunkGeom));
        PetscCall(PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, u_t, dsAux, a, t, elemVec));
        PetscCall(PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom));
        PetscCall(PetscFEIntegrateResidual(ds, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]));
        PetscCall(PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom));
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        Ne = numFaces;
        /* Riemann solve over faces (need fields at face centroids) */
        /*   We need to evaluate FE fields at those coordinates */
        PetscCall(PetscFVIntegrateRHSFunction(fv, ds, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR));
      } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, f);
    }
    /* Loop over domain */
    if (useFEM) {
      /* Add elemVec to locX */
      for (c = cS; c < cE; ++c) {
        const PetscInt cell = cells ? cells[c] : c;
        const PetscInt cind = c - cStart;

        if (mesh->printFEM > 1) PetscCall(DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]));
        if (ghostLabel) {
          PetscInt ghostVal;

          PetscCall(DMLabelGetValue(ghostLabel,cell,&ghostVal));
          if (ghostVal > 0) continue;
        }
        PetscCall(DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES));
      }
    }
    if (useFVM) {
      PetscScalar *fa;
      PetscInt     iface;

      PetscCall(VecGetArray(locF, &fa));
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     foff, pdim;

        PetscCall(PetscDSGetDiscretization(ds, f, &obj));
        PetscCall(PetscDSGetFieldOffset(ds, f, &foff));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        PetscCall(PetscFVGetNumComponents(fv, &pdim));
        /* Accumulate fluxes to cells */
        for (face = fS, iface = 0; face < fE; ++face) {
          const PetscInt *scells;
          PetscScalar    *fL = NULL, *fR = NULL;
          PetscInt        ghost, d, nsupp, nchild;

          PetscCall(DMLabelGetValue(ghostLabel, face, &ghost));
          PetscCall(DMPlexGetSupportSize(dm, face, &nsupp));
          PetscCall(DMPlexGetTreeChildren(dm, face, &nchild, NULL));
          if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
          PetscCall(DMPlexGetSupport(dm, face, &scells));
          PetscCall(DMLabelGetValue(ghostLabel,scells[0],&ghost));
          if (ghost <= 0) PetscCall(DMPlexPointLocalFieldRef(dm, scells[0], f, fa, &fL));
          PetscCall(DMLabelGetValue(ghostLabel,scells[1],&ghost));
          if (ghost <= 0) PetscCall(DMPlexPointLocalFieldRef(dm, scells[1], f, fa, &fR));
          for (d = 0; d < pdim; ++d) {
            if (fL) fL[d] -= fluxL[iface*totDim+foff+d];
            if (fR) fR[d] += fluxR[iface*totDim+foff+d];
          }
          ++iface;
        }
      }
      PetscCall(VecRestoreArray(locF, &fa));
    }
    /* Handle time derivative */
    if (locX_t) {
      PetscScalar *x_t, *fa;

      PetscCall(VecGetArray(locF, &fa));
      PetscCall(VecGetArray(locX_t, &x_t));
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     pdim, d;

        PetscCall(PetscDSGetDiscretization(ds, f, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        PetscCall(PetscFVGetNumComponents(fv, &pdim));
        for (c = cS; c < cE; ++c) {
          const PetscInt cell = cells ? cells[c] : c;
          PetscScalar   *u_t, *r;

          if (ghostLabel) {
            PetscInt ghostVal;

            PetscCall(DMLabelGetValue(ghostLabel, cell, &ghostVal));
            if (ghostVal > 0) continue;
          }
          PetscCall(DMPlexPointLocalFieldRead(dm, cell, f, x_t, &u_t));
          PetscCall(DMPlexPointLocalFieldRef(dm, cell, f, fa, &r));
          for (d = 0; d < pdim; ++d) r[d] += u_t[d];
        }
      }
      PetscCall(VecRestoreArray(locX_t, &x_t));
      PetscCall(VecRestoreArray(locF, &fa));
    }
    if (useFEM) {
      PetscCall(DMPlexRestoreCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a));
      PetscCall(DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
    }
    if (useFVM) {
      PetscCall(DMPlexRestoreFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR));
      PetscCall(DMPlexRestoreFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol));
      PetscCall(DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL));
      PetscCall(DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR));
      if (dmGrad) PetscCall(DMRestoreLocalVector(dmGrad, &locGrad));
    }
  }
  if (useFEM) PetscCall(ISDestroy(&chunkIS));
  PetscCall(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));

  if (useFEM) {
    PetscCall(DMPlexComputeBdResidual_Internal(dm, locX, locX_t, t, locF, user));

    if (maxDegree <= 1) {
      PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
      PetscCall(PetscQuadratureDestroy(&affineQuad));
    } else {
      for (f = 0; f < Nf; ++f) {
        PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
        PetscCall(PetscQuadratureDestroy(&quads[f]));
      }
      PetscCall(PetscFree2(quads,geoms));
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

    PetscCall(VecDuplicate(locF,&locFbc));
    PetscCall(VecCopy(locF,locFbc));
    PetscCall(PetscSectionGetChart(section,&pStart,&pEnd));
    PetscCall(PetscSectionGetMaxDof(section,&maxDof));
    PetscCall(PetscCalloc1(maxDof,&zeroes));
    for (p = pStart; p < pEnd; p++) {
      PetscCall(VecSetValuesSection(locFbc,section,p,zeroes,INSERT_BC_VALUES));
    }
    PetscCall(PetscFree(zeroes));
    PetscCall(DMPrintLocalVec(dm, name, mesh->printTol, locFbc));
    PetscCall(VecDestroy(&locFbc));
  }
  PetscCall(PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0));
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
    PetscCall(PetscObjectGetName((PetscObject) key[0].label, &name));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Form keys for each side of a cohesive surface must be different (%s, %" PetscInt_FMT ", %" PetscInt_FMT ")", name, key[0].value, key[0].part);
  }
  PetscCall(PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0));
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* FEM */
  PetscCall(ISGetLocalSize(cellIS, &numCells));
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  /* 1: Get sizes from dm and dmAux */
  PetscCall(DMGetSection(dm, &section));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));
  PetscCall(DMGetAuxiliaryVec(dm, key[2].label, key[2].value, key[2].part, &locA[2]));
  if (locA[2]) {
    const PetscInt cellStart = cells ? cells[cStart] : cStart;

    PetscCall(VecGetDM(locA[2], &dmAux[2]));
    PetscCall(DMGetCellDS(dmAux[2], cellStart, &dsAux[2]));
    PetscCall(PetscDSGetTotalDimension(dsAux[2], &totDimAux[2]));
    {
      const PetscInt *cone;
      PetscInt        c;

      PetscCall(DMPlexGetCone(dm, cellStart, &cone));
      for (c = 0; c < 2; ++c) {
        const PetscInt *support;
        PetscInt ssize, s;

        PetscCall(DMPlexGetSupport(dm, cone[c], &support));
        PetscCall(DMPlexGetSupportSize(dm, cone[c], &ssize));
        PetscCheck(ssize == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " from cell %" PetscInt_FMT " has support size %" PetscInt_FMT " != 2", cone[c], cellStart, ssize);
        if      (support[0] == cellStart) s = 1;
        else if (support[1] == cellStart) s = 0;
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " does not have cell %" PetscInt_FMT " in its support", cone[c], cellStart);
        PetscCall(DMGetAuxiliaryVec(dm, key[c].label, key[c].value, key[c].part, &locA[c]));
        if (locA[c]) PetscCall(VecGetDM(locA[c], &dmAux[c]));
        else         {dmAux[c] = dmAux[2];}
        PetscCall(DMGetCellDS(dmAux[c], support[s], &dsAux[c]));
        PetscCall(PetscDSGetTotalDimension(dsAux[c], &totDimAux[c]));
      }
    }
  }
  /* 2: Setup geometric data */
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree));
  if (maxDegree > 1) {
    PetscCall(PetscCalloc2(Nf, &quads, Nf, &geoms));
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      PetscCall(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      if (fe) {
        PetscCall(PetscFEGetQuadrature(fe, &quads[f]));
        PetscCall(PetscObjectReference((PetscObject) quads[f]));
      }
    }
  }
  /* Loop over chunks */
  cellChunkSize = numCells;
  numChunks     = !numCells ? 0 : PetscCeilReal(((PetscReal) numCells)/cellChunkSize);
  PetscCall(PetscCalloc1(1*cellChunkSize, &faces));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 1*cellChunkSize, faces, PETSC_USE_POINTER, &chunkIS));
  /* Extract field coefficients */
  /* NOTE This needs the end cap faces to have identical orientations */
  PetscCall(DMPlexGetCellFields(dm, cellIS, locX, locX_t, locA[2], &u, &u_t, &a[2]));
  PetscCall(DMPlexGetHybridAuxFields(dm, dmAux, dsAux, cellIS, locA, a));
  PetscCall(DMGetWorkArray(dm, cellChunkSize*totDim, MPIU_SCALAR, &elemVec));
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscInt cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;

    PetscCall(PetscMemzero(elemVec, cellChunkSize*totDim * sizeof(PetscScalar)));
    /* Get faces */
    for (c = cS; c < cE; ++c) {
      const PetscInt  cell = cells ? cells[c] : c;
      const PetscInt *cone;
      PetscCall(DMPlexGetCone(dm, cell, &cone));
      faces[0*cellChunkSize+(c-cS)] = cone[0];
      /*faces[1*cellChunkSize+(c-cS)] = cone[1];*/
    }
    PetscCall(ISGeneralSetIndices(chunkIS, 1*cellChunkSize, faces, PETSC_USE_POINTER));
    /* Get geometric data */
    if (maxDegree <= 1) {
      if (!affineQuad) PetscCall(DMFieldCreateDefaultQuadrature(coordField, chunkIS, &affineQuad));
      if (affineQuad)  PetscCall(DMSNESGetFEGeom(coordField, chunkIS, affineQuad, PETSC_TRUE, &affineGeom));
    } else {
      for (f = 0; f < Nf; ++f) {
        if (quads[f]) PetscCall(DMSNESGetFEGeom(coordField, chunkIS, quads[f], PETSC_TRUE, &geoms[f]));
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

      PetscCall(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      if (!fe) continue;
      PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
      PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
      PetscCall(PetscFEGetDimension(fe, &Nb));
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      PetscCall(PetscFEGeomGetChunk(geom,0,offset,&chunkGeom));
      PetscCall(PetscFEGeomGetChunk(geom,offset,numCells,&remGeom));
      PetscCall(PetscDSGetCohesive(ds, f, &isCohesiveField));
      chunkGeom->isCohesive = remGeom->isCohesive = PETSC_TRUE;
      key[0].field = f;
      key[1].field = f;
      key[2].field = f;
      PetscCall(PetscFEIntegrateHybridResidual(ds, key[0], 0, Ne, chunkGeom, u, u_t, dsAux[0], a[0], t, elemVec));
      PetscCall(PetscFEIntegrateHybridResidual(ds, key[0], 0, Nr, remGeom,  &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[0], &a[0][offset*totDimAux[0]], t, &elemVec[offset*totDim]));
      PetscCall(PetscFEIntegrateHybridResidual(ds, key[1], 1, Ne, chunkGeom, u, u_t, dsAux[1], a[1], t, elemVec));
      PetscCall(PetscFEIntegrateHybridResidual(ds, key[1], 1, Nr, remGeom,  &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[1], &a[1][offset*totDimAux[1]], t, &elemVec[offset*totDim]));
      PetscCall(PetscFEIntegrateHybridResidual(ds, key[2], 2, Ne, chunkGeom, u, u_t, dsAux[2], a[2], t, elemVec));
      PetscCall(PetscFEIntegrateHybridResidual(ds, key[2], 2, Nr, remGeom,  &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[2], &a[2][offset*totDimAux[2]], t, &elemVec[offset*totDim]));
      PetscCall(PetscFEGeomRestoreChunk(geom,offset,numCells,&remGeom));
      PetscCall(PetscFEGeomRestoreChunk(geom,0,offset,&chunkGeom));
    }
    /* Add elemVec to locX */
    for (c = cS; c < cE; ++c) {
      const PetscInt cell = cells ? cells[c] : c;
      const PetscInt cind = c - cStart;

      if (mesh->printFEM > 1) PetscCall(DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]));
      if (ghostLabel) {
        PetscInt ghostVal;

        PetscCall(DMLabelGetValue(ghostLabel,cell,&ghostVal));
        if (ghostVal > 0) continue;
      }
      PetscCall(DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES));
    }
  }
  PetscCall(DMPlexRestoreCellFields(dm, cellIS, locX, locX_t, locA[2], &u, &u_t, &a[2]));
  PetscCall(DMPlexRestoreHybridAuxFields(dmAux, dsAux, cellIS, locA, a));
  PetscCall(DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec));
  PetscCall(PetscFree(faces));
  PetscCall(ISDestroy(&chunkIS));
  PetscCall(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  if (maxDegree <= 1) {
    PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
    PetscCall(PetscQuadratureDestroy(&affineQuad));
  } else {
    for (f = 0; f < Nf; ++f) {
      if (geoms) PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]));
      if (quads) PetscCall(PetscQuadratureDestroy(&quads[f]));
    }
    PetscCall(PetscFree2(quads,geoms));
  }
  PetscCall(PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0));
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
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMHasBasisTransform(dm, &transform));
  PetscCall(DMGetBasisTransformDM_Internal(dm, &tdm));
  PetscCall(DMGetBasisTransformVec_Internal(dm, &tv));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(DMGetAuxiliaryVec(dm, label, values[0], 0, &locA));
  if (locA) {
    DM dmAux;

    PetscCall(VecGetDM(locA, &dmAux));
    PetscCall(DMGetEnclosureRelation(dmAux, dm, &encAux));
    PetscCall(DMConvert(dmAux, DMPLEX, &plexA));
    PetscCall(DMGetDS(plexA, &probAux));
    PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
    PetscCall(DMGetLocalSection(plexA, &sectionAux));
  }

  PetscCall(DMGetGlobalSection(dm, &globalSection));
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
    PetscCall(DMLabelGetStratumIS(label, values[v], &pointIS));
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a prior if one is a superset of the other */
      PetscCall(ISIntersect_Caching_Internal(facetIS,pointIS,&isectIS));
      PetscCall(ISDestroy(&pointIS));
      pointIS = isectIS;
    }
    PetscCall(ISGetLocalSize(pointIS, &numFaces));
    PetscCall(ISGetIndices(pointIS, &points));
    PetscCall(PetscMalloc4(numFaces*totDim, &u, locX_t ? numFaces*totDim : 0, &u_t, numFaces*totDim*totDim, &elemMat, locA ? numFaces*totDimAux : 0, &a));
    PetscCall(DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      PetscCall(DMFieldCreateDefaultQuadrature(coordField,pointIS,&qGeom));
    }
    if (!qGeom) {
      PetscFE fe;

      PetscCall(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
      PetscCall(PetscFEGetFaceQuadrature(fe, &qGeom));
      PetscCall(PetscObjectReference((PetscObject)qGeom));
    }
    PetscCall(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
    PetscCall(DMSNESGetFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom));
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;
      PetscScalar   *x     = NULL;
      PetscInt       i;

      PetscCall(DMPlexGetSupport(dm, point, &support));
      PetscCall(DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x));
      for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
      PetscCall(DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x));
      if (locX_t) {
        PetscCall(DMPlexVecGetClosure(plex, section, locX_t, support[0], NULL, &x));
        for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
        PetscCall(DMPlexVecRestoreClosure(plex, section, locX_t, support[0], NULL, &x));
      }
      if (locA) {
        PetscInt subp;
        PetscCall(DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp));
        PetscCall(DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x));
        for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
        PetscCall(DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x));
      }
    }
    PetscCall(PetscArrayzero(elemMat, numFaces*totDim*totDim));
    {
      PetscFE         fe;
      PetscInt        Nb;
      /* Conforming batches */
      PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscFEGeom    *chunkGeom = NULL;
      PetscInt        fieldJ, Nr, offset;

      PetscCall(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
      PetscCall(PetscFEGetDimension(fe, &Nb));
      PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      PetscCall(PetscFEGeomGetChunk(fgeom,0,offset,&chunkGeom));
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf+fieldJ;
        PetscCall(PetscFEIntegrateBdJacobian(prob, wf, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat));
      }
      PetscCall(PetscFEGeomGetChunk(fgeom,offset,numFaces,&chunkGeom));
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf+fieldJ;
        PetscCall(PetscFEIntegrateBdJacobian(prob, wf, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, X_tShift, &elemMat[offset*totDim*totDim]));
      }
      PetscCall(PetscFEGeomRestoreChunk(fgeom,offset,numFaces,&chunkGeom));
    }
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;

      /* Transform to global basis before insertion in Jacobian */
      PetscCall(DMPlexGetSupport(plex, point, &support));
      if (transform) PetscCall(DMPlexBasisTransformPointTensor_Internal(dm, tdm, tv, support[0], PETSC_TRUE, totDim, &elemMat[face*totDim*totDim]));
      if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(point, "BdJacobian", totDim, totDim, &elemMat[face*totDim*totDim]));
      PetscCall(DMPlexMatSetClosure(plex, section, globalSection, JacP, support[0], &elemMat[face*totDim*totDim], ADD_VALUES));
    }
    PetscCall(DMSNESRestoreFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom));
    PetscCall(PetscQuadratureDestroy(&qGeom));
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
    PetscCall(PetscFree4(u, u_t, elemMat, a));
  }
  if (plex)  PetscCall(DMDestroy(&plex));
  if (plexA) PetscCall(DMDestroy(&plexA));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobianSingle(DM dm, PetscReal t, PetscWeakForm wf, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt field, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP)
{
  DMField        coordField;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMLabelGetStratumIS(depthLabel, dim-1, &facetIS));
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMPlexComputeBdJacobian_Single_Internal(dm, t, wf, label, numValues, values, field, locX, locX_t, X_tShift, Jac, JacP, coordField, facetIS));
  PetscCall(ISDestroy(&facetIS));
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
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMLabelGetStratumIS(depthLabel, dim-1, &facetIS));
  PetscCall(PetscDSGetNumBoundary(prob, &numBd));
  PetscCall(DMGetCoordinateField(dm, &coordField));
  for (bd = 0; bd < numBd; ++bd) {
    PetscWeakForm           wf;
    DMBoundaryConditionType type;
    DMLabel                 label;
    const PetscInt         *values;
    PetscInt                fieldI, numValues;
    PetscObject             obj;
    PetscClassId            id;

    PetscCall(PetscDSGetBoundary(prob, bd, &wf, &type, NULL, &label, &numValues, &values, &fieldI, NULL, NULL, NULL, NULL, NULL));
    PetscCall(PetscDSGetDiscretization(prob, fieldI, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    PetscCall(DMPlexComputeBdJacobian_Single_Internal(dm, t, wf, label, numValues, values, fieldI, locX, locX_t, X_tShift, Jac, JacP, coordField, facetIS));
  }
  PetscCall(ISDestroy(&facetIS));
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
  PetscBool       hasJac = PETSC_FALSE, hasPrec = PETSC_FALSE, hasDyn, hasFV = PETSC_FALSE, transform;

  PetscFunctionBegin;
  if (!cellIS) goto end;
  PetscCall(PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscCall(ISGetLocalSize(cellIS, &numCells));
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(DMHasBasisTransform(dm, &transform));
  PetscCall(DMGetBasisTransformDM_Internal(dm, &tdm));
  PetscCall(DMGetBasisTransformVec_Internal(dm, &tv));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &globalSection));
  PetscCall(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &prob));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscDSHasJacobian(prob, &hasJac));
  PetscCall(PetscDSHasJacobianPreconditioner(prob, &hasPrec));
  /* user passed in the same matrix, avoid double contributions and
     only assemble the Jacobian */
  if (hasJac && Jac == JacP) hasPrec = PETSC_FALSE;
  PetscCall(PetscDSHasDynamicJacobian(prob, &hasDyn));
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &A));
  if (A) {
    PetscCall(VecGetDM(A, &dmAux));
    PetscCall(DMGetEnclosureRelation(dmAux, dm, &encAux));
    PetscCall(DMConvert(dmAux, DMPLEX, &plex));
    PetscCall(DMGetLocalSection(plex, &sectionAux));
    PetscCall(DMGetDS(dmAux, &probAux));
    PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
  }
  PetscCall(PetscMalloc5(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,hasJac ? numCells*totDim*totDim : 0,&elemMat,hasPrec ? numCells*totDim*totDim : 0, &elemMatP,hasDyn ? numCells*totDim*totDim : 0, &elemMatD));
  if (dmAux) PetscCall(PetscMalloc1(numCells*totDimAux, &a));
  PetscCall(DMGetCoordinateField(dm, &coordField));
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL,  *x_t = NULL;
    PetscInt       i;

    PetscCall(DMPlexVecGetClosure(dm, section, X, cell, NULL, &x));
    for (i = 0; i < totDim; ++i) u[cind*totDim+i] = x[i];
    PetscCall(DMPlexVecRestoreClosure(dm, section, X, cell, NULL, &x));
    if (X_t) {
      PetscCall(DMPlexVecGetClosure(dm, section, X_t, cell, NULL, &x_t));
      for (i = 0; i < totDim; ++i) u_t[cind*totDim+i] = x_t[i];
      PetscCall(DMPlexVecRestoreClosure(dm, section, X_t, cell, NULL, &x_t));
    }
    if (dmAux) {
      PetscInt subcell;
      PetscCall(DMGetEnclosurePoint(dmAux, dm, encAux, cell, &subcell));
      PetscCall(DMPlexVecGetClosure(plex, sectionAux, A, subcell, NULL, &x));
      for (i = 0; i < totDimAux; ++i) a[cind*totDimAux+i] = x[i];
      PetscCall(DMPlexVecRestoreClosure(plex, sectionAux, A, subcell, NULL, &x));
    }
  }
  if (hasJac)  PetscCall(PetscArrayzero(elemMat,  numCells*totDim*totDim));
  if (hasPrec) PetscCall(PetscArrayzero(elemMatP, numCells*totDim*totDim));
  if (hasDyn)  PetscCall(PetscArrayzero(elemMatD, numCells*totDim*totDim));
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

    PetscCall(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
    PetscCall(PetscObjectGetClassId((PetscObject) fe, &id));
    if (id == PETSCFV_CLASSID) {hasFV = PETSC_TRUE; continue;}
    PetscCall(PetscFEGetDimension(fe, &Nb));
    PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
    PetscCall(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
    if (maxDegree <= 1) {
      PetscCall(DMFieldCreateDefaultQuadrature(coordField,cellIS,&qGeom));
    }
    if (!qGeom) {
      PetscCall(PetscFEGetQuadrature(fe,&qGeom));
      PetscCall(PetscObjectReference((PetscObject)qGeom));
    }
    PetscCall(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
    PetscCall(DMSNESGetFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM));
    blockSize = Nb;
    batchSize = numBlocks * blockSize;
    PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    PetscCall(PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom));
    PetscCall(PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&remGeom));
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      key.field = fieldI*Nf+fieldJ;
      if (hasJac) {
        PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat));
        PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]));
      }
      if (hasPrec) {
        PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatP));
        PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatP[offset*totDim*totDim]));
      }
      if (hasDyn) {
        PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatD));
        PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]));
      }
    }
    PetscCall(PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&remGeom));
    PetscCall(PetscFEGeomRestoreChunk(cgeomFEM,0,offset,&chunkGeom));
    PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM));
    PetscCall(PetscQuadratureDestroy(&qGeom));
  }
  /*   Add contribution from X_t */
  if (hasDyn) {for (c = 0; c < numCells*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];}
  if (hasFV) {
    PetscClassId id;
    PetscFV      fv;
    PetscInt     offsetI, NcI, NbI = 1, fc, f;

    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscCall(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fv));
      PetscCall(PetscDSGetFieldOffset(prob, fieldI, &offsetI));
      PetscCall(PetscObjectGetClassId((PetscObject) fv, &id));
      if (id != PETSCFV_CLASSID) continue;
      /* Put in the identity */
      PetscCall(PetscFVGetNumComponents(fv, &NcI));
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
    PetscCall(MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  }
  /* Insert values into matrix */
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;

    /* Transform to global basis before insertion in Jacobian */
    if (transform) PetscCall(DMPlexBasisTransformPointTensor_Internal(dm, tdm, tv, cell, PETSC_TRUE, totDim, &elemMat[cind*totDim*totDim]));
    if (hasPrec) {
      if (hasJac) {
        if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
        PetscCall(DMPlexMatSetClosure(dm, section, globalSection, Jac, cell, &elemMat[cind*totDim*totDim], ADD_VALUES));
      }
      if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMatP[cind*totDim*totDim]));
      PetscCall(DMPlexMatSetClosure(dm, section, globalSection, JacP, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES));
    } else {
      if (hasJac) {
        if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
        PetscCall(DMPlexMatSetClosure(dm, section, globalSection, JacP, cell, &elemMat[cind*totDim*totDim], ADD_VALUES));
      }
    }
  }
  PetscCall(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  if (hasFV) PetscCall(MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  PetscCall(PetscFree5(u,u_t,elemMat,elemMatP,elemMatD));
  if (dmAux) {
    PetscCall(PetscFree(a));
    PetscCall(DMDestroy(&plex));
  }
  /* Compute boundary integrals */
  PetscCall(DMPlexComputeBdJacobian_Internal(dm, X, X_t, t, X_tShift, Jac, JacP, user));
  /* Assemble matrix */
end:
  {
    PetscBool assOp = hasJac && hasPrec ? PETSC_TRUE : PETSC_FALSE, gassOp;

    PetscCallMPI(MPI_Allreduce(&assOp, &gassOp, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) dm)));
    if (hasJac && hasPrec) {
      PetscCall(MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY));
    }
  }
  PetscCall(MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0));
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
    PetscCall(PetscObjectGetName((PetscObject) key[0].label, &name));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Form keys for each side of a cohesive surface must be different (%s, %" PetscInt_FMT ", %" PetscInt_FMT ")", name, key[0].value, key[0].part);
  }
  PetscCall(PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscCall(ISGetLocalSize(cellIS, &numCells));
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMGetSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &globalSection));
  PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
  PetscCall(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));
  PetscCall(PetscDSHasBdJacobian(ds, &hasBdJac));
  PetscCall(PetscDSHasBdJacobianPreconditioner(ds, &hasBdPrec));
  PetscCall(DMGetAuxiliaryVec(dm, key[2].label, key[2].value, key[2].part, &locA[2]));
  if (locA[2]) {
    const PetscInt cellStart = cells ? cells[cStart] : cStart;

    PetscCall(VecGetDM(locA[2], &dmAux[2]));
    PetscCall(DMConvert(dmAux[2], DMPLEX, &plexA));
    PetscCall(DMGetSection(dmAux[2], &sectionAux[2]));
    PetscCall(DMGetCellDS(dmAux[2], cellStart, &dsAux[2]));
    PetscCall(PetscDSGetTotalDimension(dsAux[2], &totDimAux[2]));
    {
      const PetscInt *cone;
      PetscInt        c;

      PetscCall(DMPlexGetCone(dm, cellStart, &cone));
      for (c = 0; c < 2; ++c) {
        const PetscInt *support;
        PetscInt ssize, s;

        PetscCall(DMPlexGetSupport(dm, cone[c], &support));
        PetscCall(DMPlexGetSupportSize(dm, cone[c], &ssize));
        PetscCheck(ssize == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " from cell %" PetscInt_FMT " has support size %" PetscInt_FMT " != 2", cone[c], cellStart, ssize);
        if      (support[0] == cellStart) s = 1;
        else if (support[1] == cellStart) s = 0;
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " does not have cell %" PetscInt_FMT " in its support", cone[c], cellStart);
        PetscCall(DMGetAuxiliaryVec(dm, key[c].label, key[c].value, key[c].part, &locA[c]));
        if (locA[c]) PetscCall(VecGetDM(locA[c], &dmAux[c]));
        else         {dmAux[c] = dmAux[2];}
        PetscCall(DMGetCellDS(dmAux[c], support[s], &dsAux[c]));
        PetscCall(PetscDSGetTotalDimension(dsAux[c], &totDimAux[c]));
      }
    }
  }
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree));
  if (maxDegree > 1) {
    PetscInt f;
    PetscCall(PetscCalloc2(Nf, &quads, Nf, &geoms));
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      PetscCall(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      if (fe) {
        PetscCall(PetscFEGetQuadrature(fe, &quads[f]));
        PetscCall(PetscObjectReference((PetscObject) quads[f]));
      }
    }
  }
  cellChunkSize = numCells;
  numChunks     = !numCells ? 0 : PetscCeilReal(((PetscReal) numCells)/cellChunkSize);
  PetscCall(PetscCalloc1(1*cellChunkSize, &faces));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 1*cellChunkSize, faces, PETSC_USE_POINTER, &chunkIS));
  PetscCall(DMPlexGetCellFields(dm, cellIS, locX, locX_t, locA[2], &u, &u_t, &a[2]));
  PetscCall(DMPlexGetHybridAuxFields(dm, dmAux, dsAux, cellIS, locA, a));
  PetscCall(DMGetWorkArray(dm, hasBdJac  ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMat));
  PetscCall(DMGetWorkArray(dm, hasBdPrec ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMatP));
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscInt cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;

    if (hasBdJac)  PetscCall(PetscMemzero(elemMat,  numCells*totDim*totDim * sizeof(PetscScalar)));
    if (hasBdPrec) PetscCall(PetscMemzero(elemMatP, numCells*totDim*totDim * sizeof(PetscScalar)));
    /* Get faces */
    for (c = cS; c < cE; ++c) {
      const PetscInt  cell = cells ? cells[c] : c;
      const PetscInt *cone;
      PetscCall(DMPlexGetCone(plex, cell, &cone));
      faces[0*cellChunkSize+(c-cS)] = cone[0];
      /*faces[2*cellChunkSize+(c-cS)] = cone[1];*/
    }
    PetscCall(ISGeneralSetIndices(chunkIS, 1*cellChunkSize, faces, PETSC_USE_POINTER));
    if (maxDegree <= 1) {
      if (!affineQuad) PetscCall(DMFieldCreateDefaultQuadrature(coordField, chunkIS, &affineQuad));
      if (affineQuad)  PetscCall(DMSNESGetFEGeom(coordField, chunkIS, affineQuad, PETSC_TRUE, &affineGeom));
    } else {
      PetscInt f;
      for (f = 0; f < Nf; ++f) {
        if (quads[f]) PetscCall(DMSNESGetFEGeom(coordField, chunkIS, quads[f], PETSC_TRUE, &geoms[f]));
      }
    }

    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscFE         feI;
      PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[fieldI];
      PetscFEGeom    *chunkGeom = NULL, *remGeom = NULL;
      PetscQuadrature quad = affineQuad ? affineQuad : quads[fieldI];
      PetscInt        numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset, Nq, Nb;
      PetscBool       isCohesiveField;

      PetscCall(PetscDSGetDiscretization(ds, fieldI, (PetscObject *) &feI));
      if (!feI) continue;
      PetscCall(PetscFEGetTileSizes(feI, NULL, &numBlocks, NULL, &numBatches));
      PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
      PetscCall(PetscFEGetDimension(feI, &Nb));
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      PetscCall(PetscFESetTileSizes(feI, blockSize, numBlocks, batchSize, numBatches));
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      PetscCall(PetscFEGeomGetChunk(geom,0,offset,&chunkGeom));
      PetscCall(PetscFEGeomGetChunk(geom,offset,numCells,&remGeom));
      PetscCall(PetscDSGetCohesive(ds, fieldI, &isCohesiveField));
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        PetscFE feJ;

        PetscCall(PetscDSGetDiscretization(ds, fieldJ, (PetscObject *) &feJ));
        if (!feJ) continue;
        key[0].field = fieldI*Nf+fieldJ;
        key[1].field = fieldI*Nf+fieldJ;
        key[2].field = fieldI*Nf+fieldJ;
        if (hasBdJac) {
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[0], 0, Ne, chunkGeom, u, u_t, dsAux[0], a[0], t, X_tShift, elemMat));
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[0], 0, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[0], &a[0][offset*totDimAux[0]], t, X_tShift, &elemMat[offset*totDim*totDim]));
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[1], 1, Ne, chunkGeom, u, u_t, dsAux[1], a[1], t, X_tShift, elemMat));
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[1], 1, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[1], &a[1][offset*totDimAux[1]], t, X_tShift, &elemMat[offset*totDim*totDim]));
        }
        if (hasBdPrec) {
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[0], 0, Ne, chunkGeom, u, u_t, dsAux[0], a[0], t, X_tShift, elemMatP));
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[0], 0, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[0], &a[0][offset*totDimAux[0]], t, X_tShift, &elemMatP[offset*totDim*totDim]));
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[1], 1, Ne, chunkGeom, u, u_t, dsAux[1], a[1], t, X_tShift, elemMatP));
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[1], 1, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[1], &a[1][offset*totDimAux[1]], t, X_tShift, &elemMatP[offset*totDim*totDim]));
        }
        if (hasBdJac) {
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[2], 2, Ne, chunkGeom, u, u_t, dsAux[2], a[2], t, X_tShift, elemMat));
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN, key[2], 2, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[2], &a[2][offset*totDimAux[2]], t, X_tShift, &elemMat[offset*totDim*totDim]));
        }
        if (hasBdPrec) {
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[2], 2, Ne, chunkGeom, u, u_t, dsAux[2], a[2], t, X_tShift, elemMatP));
          PetscCall(PetscFEIntegrateHybridJacobian(ds, PETSCFE_JACOBIAN_PRE, key[2], 2, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[2], &a[2][offset*totDimAux[2]], t, X_tShift, &elemMatP[offset*totDim*totDim]));
        }
      }
      PetscCall(PetscFEGeomRestoreChunk(geom,offset,numCells,&remGeom));
      PetscCall(PetscFEGeomRestoreChunk(geom,0,offset,&chunkGeom));
    }
    /* Insert values into matrix */
    for (c = cS; c < cE; ++c) {
      const PetscInt cell = cells ? cells[c] : c;
      const PetscInt cind = c - cS;

      if (hasBdPrec) {
        if (hasBdJac) {
          if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
          PetscCall(DMPlexMatSetClosure(plex, section, globalSection, Jac, cell, &elemMat[cind*totDim*totDim], ADD_VALUES));
        }
        if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMatP[cind*totDim*totDim]));
        PetscCall(DMPlexMatSetClosure(plex, section, globalSection, JacP, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES));
      } else if (hasBdJac) {
        if (mesh->printFEM > 1) PetscCall(DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
        PetscCall(DMPlexMatSetClosure(plex, section, globalSection, JacP, cell, &elemMat[cind*totDim*totDim], ADD_VALUES));
      }
    }
  }
  PetscCall(DMPlexRestoreCellFields(dm, cellIS, locX, locX_t, locA[2], &u, &u_t, &a[2]));
  PetscCall(DMPlexRestoreHybridAuxFields(dmAux, dsAux, cellIS, locA, a));
  PetscCall(DMRestoreWorkArray(dm, hasBdJac  ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMat));
  PetscCall(DMRestoreWorkArray(dm, hasBdPrec ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMatP));
  PetscCall(PetscFree(faces));
  PetscCall(ISDestroy(&chunkIS));
  PetscCall(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  if (maxDegree <= 1) {
    PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom));
    PetscCall(PetscQuadratureDestroy(&affineQuad));
  } else {
    PetscInt f;
    for (f = 0; f < Nf; ++f) {
      if (geoms) PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE, &geoms[f]));
      if (quads) PetscCall(PetscQuadratureDestroy(&quads[f]));
    }
    PetscCall(PetscFree2(quads,geoms));
  }
  if (dmAux[2]) PetscCall(DMDestroy(&plexA));
  PetscCall(DMDestroy(&plex));
  /* Assemble matrix */
  if (hasBdJac && hasBdPrec) {
    PetscCall(MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0));
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
  PetscCall(PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  if (!cellIS) {
    PetscInt depth;

    PetscCall(DMPlexGetDepth(plex, &depth));
    PetscCall(DMGetStratumIS(plex, "dim", depth, &cellIS));
    if (!cellIS) PetscCall(DMGetStratumIS(plex, "depth", depth, &cellIS));
  } else {
    PetscCall(PetscObjectReference((PetscObject) cellIS));
  }
  PetscCall(ISGetLocalSize(cellIS, &numCells));
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &globalSection));
  PetscCall(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &prob));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscDSGetTotalDimension(prob, &totDim));
  PetscCall(PetscDSHasDynamicJacobian(prob, &hasDyn));
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(DMGetAuxiliaryVec(dm, key.label, key.value, key.part, &A));
  if (A) {
    PetscCall(VecGetDM(A, &dmAux));
    PetscCall(DMGetEnclosureRelation(dmAux, dm, &encAux));
    PetscCall(DMConvert(dmAux, DMPLEX, &plexAux));
    PetscCall(DMGetLocalSection(plexAux, &sectionAux));
    PetscCall(DMGetDS(dmAux, &probAux));
    PetscCall(PetscDSGetTotalDimension(probAux, &totDimAux));
  }
  PetscCall(VecSet(Z, 0.0));
  PetscCall(PetscMalloc6(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,numCells*totDim*totDim,&elemMat,hasDyn ? numCells*totDim*totDim : 0, &elemMatD,numCells*totDim,&y,totDim,&z));
  if (dmAux) PetscCall(PetscMalloc1(numCells*totDimAux, &a));
  PetscCall(DMGetCoordinateField(dm, &coordField));
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL,  *x_t = NULL;
    PetscInt       i;

    PetscCall(DMPlexVecGetClosure(plex, section, X, cell, NULL, &x));
    for (i = 0; i < totDim; ++i) u[cind*totDim+i] = x[i];
    PetscCall(DMPlexVecRestoreClosure(plex, section, X, cell, NULL, &x));
    if (X_t) {
      PetscCall(DMPlexVecGetClosure(plex, section, X_t, cell, NULL, &x_t));
      for (i = 0; i < totDim; ++i) u_t[cind*totDim+i] = x_t[i];
      PetscCall(DMPlexVecRestoreClosure(plex, section, X_t, cell, NULL, &x_t));
    }
    if (dmAux) {
      PetscInt subcell;
      PetscCall(DMGetEnclosurePoint(dmAux, dm, encAux, cell, &subcell));
      PetscCall(DMPlexVecGetClosure(plexAux, sectionAux, A, subcell, NULL, &x));
      for (i = 0; i < totDimAux; ++i) a[cind*totDimAux+i] = x[i];
      PetscCall(DMPlexVecRestoreClosure(plexAux, sectionAux, A, subcell, NULL, &x));
    }
    PetscCall(DMPlexVecGetClosure(plex, section, Y, cell, NULL, &x));
    for (i = 0; i < totDim; ++i) y[cind*totDim+i] = x[i];
    PetscCall(DMPlexVecRestoreClosure(plex, section, Y, cell, NULL, &x));
  }
  PetscCall(PetscArrayzero(elemMat, numCells*totDim*totDim));
  if (hasDyn)  PetscCall(PetscArrayzero(elemMatD, numCells*totDim*totDim));
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

    PetscCall(PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe));
    PetscCall(PetscFEGetQuadrature(fe, &quad));
    PetscCall(PetscFEGetDimension(fe, &Nb));
    PetscCall(PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches));
    PetscCall(DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree));
    if (maxDegree <= 1) PetscCall(DMFieldCreateDefaultQuadrature(coordField,cellIS,&qGeom));
    if (!qGeom) {
      PetscCall(PetscFEGetQuadrature(fe,&qGeom));
      PetscCall(PetscObjectReference((PetscObject)qGeom));
    }
    PetscCall(PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL));
    PetscCall(DMSNESGetFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM));
    blockSize = Nb;
    batchSize = numBlocks * blockSize;
    PetscCall(PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches));
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    PetscCall(PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom));
    PetscCall(PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&remGeom));
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      key.field = fieldI*Nf + fieldJ;
      PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat));
      PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]));
      if (hasDyn) {
        PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatD));
        PetscCall(PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]));
      }
    }
    PetscCall(PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&remGeom));
    PetscCall(PetscFEGeomRestoreChunk(cgeomFEM,0,offset,&chunkGeom));
    PetscCall(DMSNESRestoreFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM));
    PetscCall(PetscQuadratureDestroy(&qGeom));
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
      PetscCall(DMPrintCellMatrix(c, name, totDim, totDim, &elemMat[cind*totDim*totDim]));
      PetscCall(DMPrintCellVector(c, "Y",  totDim, &y[cind*totDim]));
      PetscCall(DMPrintCellVector(c, "Z",  totDim, z));
    }
    PetscCall(DMPlexVecSetClosure(dm, section, Z, cell, z, ADD_VALUES));
  }
  PetscCall(PetscFree6(u,u_t,elemMat,elemMatD,y,z));
  if (mesh->printFEM) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)Z), "Z:\n"));
    PetscCall(VecView(Z, NULL));
  }
  PetscCall(ISRestorePointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(PetscFree(a));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(DMDestroy(&plexAux));
  PetscCall(DMDestroy(&plex));
  PetscCall(PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0));
  PetscFunctionReturn(0);
}
