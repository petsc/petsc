#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#include <petsc/private/hashsetij.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/petscfvimpl.h>

static PetscErrorCode DMPlexConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  if (isPlex) {
    *plex = dm;
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex);CHKERRQ(ierr);
    if (!*plex) {
      ierr = DMConvert(dm, DMPLEX, plex);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex);CHKERRQ(ierr);
      if (copy) {
        DMSubDomainHookLink link;

        ierr = DMCopyAuxiliaryVec(dm, *plex);CHKERRQ(ierr);
        /* Run the subdomain hook (this will copy the DMSNES/DMTS) */
        for (link = dm->subdomainhook; link; link = link->next) {
          if (link->ddhook) {ierr = (*link->ddhook)(dm, *plex, link->ctx);CHKERRQ(ierr);}
        }
      }
    } else {
      ierr = PetscObjectReference((PetscObject) *plex);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscContainerUserDestroy_PetscFEGeom (void *ctx)
{
  PetscFEGeom *geom = (PetscFEGeom *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFEGeomDestroy(&geom);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetFEGeom(DMField coordField, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetId((PetscObject)quad,&id);CHKERRQ(ierr);
  ierr = PetscSNPrintf(composeStr, 32, "DMPlexGetFEGeom_%x\n", id);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) pointIS, composeStr, (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container, (void **) geom);CHKERRQ(ierr);
  } else {
    ierr = DMFieldCreateFEGeom(coordField, pointIS, quad, faceData, geom);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container, (void *) *geom);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) pointIS, composeStr, (PetscObject) container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
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

  Input Arguments:
+ dm   - the DM
- unit - The SI unit

  Output Argument:
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

  Input Arguments:
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
  if (dim != dim2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input dimension %D does not match context dimension %D", dim, dim2);
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

  Input Arguments:
+ dm - the DM
- field - The field number for the rigid body space, or 0 for the default

  Output Argument:
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  if (Nf && (field < 0 || field >= Nf)) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Field %D is not in [0, Nf)", field, Nf);
  if (dim == 1 && Nf < 2) {
    ierr = MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, sp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(globalSection, &n);CHKERRQ(ierr);
  ierr = PetscCalloc1(Nf, &func);CHKERRQ(ierr);
  m    = (dim*(dim+1))/2;
  ierr = VecCreate(comm, &mode[0]);CHKERRQ(ierr);
  ierr = VecSetSizes(mode[0], n, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(mode[0]);CHKERRQ(ierr);
  ierr = VecGetSize(mode[0], &n);CHKERRQ(ierr);
  mmin = PetscMin(m, n);
  func[field] = DMPlexProjectRigidBody_Private;
  for (i = 1; i < m; ++i) {ierr = VecDuplicate(mode[0], &mode[i]);CHKERRQ(ierr);}
  for (d = 0; d < m; d++) {
    PetscInt ctx[2];
    void    *voidctx = (void *) (&ctx[0]);

    ctx[0] = dimEmbed;
    ctx[1] = d;
    ierr = DMProjectFunction(dm, 0.0, func, &voidctx, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
  }
  /* Orthonormalize system */
  for (i = 0; i < mmin; ++i) {
    PetscScalar dots[6];

    ierr = VecNormalize(mode[i], NULL);CHKERRQ(ierr);
    ierr = VecMDot(mode[i], mmin-i-1, mode+i+1, dots+i+1);CHKERRQ(ierr);
    for (j = i+1; j < mmin; ++j) {
      dots[j] *= -1.0;
      ierr = VecAXPY(mode[j], dots[j], mode[i]);CHKERRQ(ierr);
    }
  }
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, mmin, mode, sp);CHKERRQ(ierr);
  for (i = 0; i < m; ++i) {ierr = VecDestroy(&mode[i]);CHKERRQ(ierr);}
  ierr = PetscFree(func);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateRigidBodies - For the default global section, create rigid body modes by function space interpolation

  Collective on dm

  Input Arguments:
+ dm    - the DM
. nb    - The number of bodies
. label - The DMLabel marking each domain
. nids  - The number of ids per body
- ids   - An array of the label ids in sequence for each domain

  Output Argument:
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(globalSection, &n);CHKERRQ(ierr);
  m    = nb * (dim*(dim+1))/2;
  ierr = PetscMalloc2(m, &mode, m, &dots);CHKERRQ(ierr);
  ierr = VecCreate(comm, &mode[0]);CHKERRQ(ierr);
  ierr = VecSetSizes(mode[0], n, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(mode[0]);CHKERRQ(ierr);
  for (i = 1; i < m; ++i) {ierr = VecDuplicate(mode[0], &mode[i]);CHKERRQ(ierr);}
  for (b = 0, off = 0; b < nb; ++b) {
    for (d = 0; d < m/nb; ++d) {
      PetscInt         ctx[2];
      PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *) = DMPlexProjectRigidBody_Private;
      void            *voidctx = (void *) (&ctx[0]);

      ctx[0] = dimEmbed;
      ctx[1] = d;
      ierr = DMProjectFunctionLabel(dm, 0.0, label, nids[b], &ids[off], 0, NULL, &func, &voidctx, INSERT_VALUES, mode[d]);CHKERRQ(ierr);
      off   += nids[b];
    }
  }
  /* Orthonormalize system */
  for (i = 0; i < m; ++i) {
    PetscScalar dots[6];

    ierr = VecNormalize(mode[i], NULL);CHKERRQ(ierr);
    ierr = VecMDot(mode[i], m-i-1, mode+i+1, dots+i+1);CHKERRQ(ierr);
    for (j = i+1; j < m; ++j) {
      dots[j] *= -1.0;
      ierr = VecAXPY(mode[j], dots[j], mode[i]);CHKERRQ(ierr);
    }
  }
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, m, mode, sp);CHKERRQ(ierr);
  for (i = 0; i< m; ++i) {ierr = VecDestroy(&mode[i]);CHKERRQ(ierr);}
  ierr = PetscFree2(mode, dots);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(PetscSqr(dim), &rc->R, PetscSqr(dim), &rc->RT);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    c1 = PetscCosReal(rc->alpha);s1 = PetscSinReal(rc->alpha);
    rc->R[0] =  c1;rc->R[1] = s1;
    rc->R[2] = -s1;rc->R[3] = c1;
    ierr = PetscArraycpy(rc->RT, rc->R, PetscSqr(dim));CHKERRQ(ierr);
    DMPlex_Transpose2D_Internal(rc->RT);
    break;
  case 3:
    c1 = PetscCosReal(rc->alpha);s1 = PetscSinReal(rc->alpha);
    c2 = PetscCosReal(rc->beta); s2 = PetscSinReal(rc->beta);
    c3 = PetscCosReal(rc->gamma);s3 = PetscSinReal(rc->gamma);
    rc->R[0] =  c1*c3 - c2*s1*s3;rc->R[1] =  c3*s1    + c1*c2*s3;rc->R[2] = s2*s3;
    rc->R[3] = -c1*s3 - c2*c3*s1;rc->R[4] =  c1*c2*c3 - s1*s3;   rc->R[5] = c3*s2;
    rc->R[6] =  s1*s2;           rc->R[7] = -c1*s2;              rc->R[8] = c2;
    ierr = PetscArraycpy(rc->RT, rc->R, PetscSqr(dim));CHKERRQ(ierr);
    DMPlex_Transpose3D_Internal(rc->RT);
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransformDestroy_Rotation_Internal(DM dm, void *ctx)
{
  RotCtx        *rc = (RotCtx *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(rc->R, rc->RT);CHKERRQ(ierr);
  ierr = PetscFree(rc);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  #if defined(PETSC_USE_COMPLEX)
  switch (dim) {
    case 2:
    {
      PetscScalar yt[2], zt[2] = {0.0,0.0};

      yt[0] = y[0]; yt[1] = y[1];
      ierr = DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, yt, zt, ctx);CHKERRQ(ierr);
      z[0] = PetscRealPart(zt[0]); z[1] = PetscRealPart(zt[1]);
    }
    break;
    case 3:
    {
      PetscScalar yt[3], zt[3] = {0.0,0.0,0.0};

      yt[0] = y[0]; yt[1] = y[1]; yt[2] = y[2];
      ierr = DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, yt, zt, ctx);CHKERRQ(ierr);
      z[0] = PetscRealPart(zt[0]); z[1] = PetscRealPart(zt[1]); z[2] = PetscRealPart(zt[2]);
    }
    break;
  }
  #else
  ierr = DMPlexBasisTransformApply_Internal(dm, x, l2g, dim, y, z, ctx);CHKERRQ(ierr);
  #endif
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexBasisTransformApply_Internal(DM dm, const PetscReal x[], PetscBool l2g, PetscInt dim, const PetscScalar *y, PetscScalar *z, void *ctx)
{
  const PetscScalar *A;
  PetscErrorCode     ierr;

  PetscFunctionBeginHot;
  ierr = (*dm->transformGetMatrix)(dm, x, l2g, &A, ctx);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBeginHot;
  ierr = DMGetLocalSection(tdm, &ts);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldDof(ts, p, f, &dof);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tv, &ta);CHKERRQ(ierr);
  ierr = DMPlexPointLocalFieldRead(tdm, p, f, ta, (void *) &tva);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(tv, &ta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexBasisTransformFieldTensor_Internal(DM dm, DM tdm, Vec tv, PetscInt pf, PetscInt f, PetscInt pg, PetscInt g, PetscBool l2g, PetscInt lda, PetscScalar *a)
{
  PetscSection       s, ts;
  const PetscScalar *ta, *tvaf, *tvag;
  PetscInt           fdof, gdof, fpdof, gpdof;
  PetscErrorCode     ierr;

  PetscFunctionBeginHot;
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = DMGetLocalSection(tdm, &ts);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldDof(s, pf, f, &fpdof);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldDof(s, pg, g, &gpdof);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldDof(ts, pf, f, &fdof);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldDof(ts, pg, g, &gdof);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tv, &ta);CHKERRQ(ierr);
  ierr = DMPlexPointLocalFieldRead(tdm, pf, f, ta, (void *) &tvaf);CHKERRQ(ierr);
  ierr = DMPlexPointLocalFieldRead(tdm, pg, g, ta, (void *) &tvag);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(tv, &ta);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    for (cp = 0; cp < Np*2; cp += 2) {
      ierr = PetscSectionGetFieldDof(s, points[cp], f, &dof);CHKERRQ(ierr);
      if (!dof) continue;
      if (fieldActive[f]) {ierr = DMPlexBasisTransformField_Internal(dm, tdm, tv, points[cp], f, l2g, &a[d]);CHKERRQ(ierr);}
      d += dof;
    }
  }
  ierr = DMPlexRestoreCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp);CHKERRQ(ierr);
  for (f = 0, r = 0; f < Nf; ++f) {
    for (cpf = 0; cpf < Np*2; cpf += 2) {
      ierr = PetscSectionGetFieldDof(s, points[cpf], f, &fdof);CHKERRQ(ierr);
      for (g = 0, c = 0; g < Nf; ++g) {
        for (cpg = 0; cpg < Np*2; cpg += 2) {
          ierr = PetscSectionGetFieldDof(s, points[cpg], g, &gdof);CHKERRQ(ierr);
          ierr = DMPlexBasisTransformFieldTensor_Internal(dm, tdm, tv, points[cpf], f, points[cpg], g, l2g, lda, &a[r*lda+c]);CHKERRQ(ierr);
          c += gdof;
        }
      }
      if (c != lda) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of columns %D should be %D", c, lda);
      r += fdof;
    }
  }
  if (r != lda) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of rows %D should be %D", c, lda);
  ierr = DMPlexRestoreCompressedClosure(dm, s, p, &Np, &points, &clSection, &clPoints, &clp);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMGetLocalSection(tdm, &ts);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
  ierr = VecGetArray(lv, &a);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tv, &ta);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < Nf; ++f) {
      ierr = DMPlexPointLocalFieldRef(dm, p, f, a, (void *) &va);CHKERRQ(ierr);
      ierr = DMPlexBasisTransformField_Internal(dm, tdm, tv, p, f, l2g, va);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(lv, &a);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tv, &ta);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lv, VEC_CLASSID, 2);
  ierr = DMPlexBasisTransform_Internal(dm, lv, PETSC_FALSE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(lv, VEC_CLASSID, 2);
  ierr = DMPlexBasisTransform_Internal(dm, lv, PETSC_TRUE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &rc);CHKERRQ(ierr);
  dm->transformCtx       = rc;
  dm->transformSetUp     = DMPlexBasisTransformSetUp_Rotation_Internal;
  dm->transformDestroy   = DMPlexBasisTransformDestroy_Rotation_Internal;
  dm->transformGetMatrix = DMPlexBasisTransformGetMatrix_Rotation_Internal;
  rc->dim   = cdim;
  rc->alpha = alpha;
  rc->beta  = beta;
  rc->gamma = gamma;
  ierr = (*dm->transformSetUp)(dm, dm->transformCtx);CHKERRQ(ierr);
  ierr = DMConstructBasisTransform_Internal(dm);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  ierr = PetscCalloc2(numFields,&funcs,numFields,&ctxs);CHKERRQ(ierr);
  funcs[field] = func;
  ctxs[field]  = ctx;
  ierr = DMProjectFunctionLabelLocal(dm, time, label, numids, ids, Nc, comps, funcs, ctxs, INSERT_BC_VALUES, locX);CHKERRQ(ierr);
  ierr = PetscFree2(funcs,ctxs);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  ierr = PetscCalloc2(numFields,&funcs,numFields,&ctxs);CHKERRQ(ierr);
  funcs[field] = func;
  ctxs[field]  = ctx;
  ierr = DMProjectFieldLabelLocal(dm, time, label, numids, ids, Nc, comps, locU, funcs, INSERT_BC_VALUES, locX);CHKERRQ(ierr);
  ierr = PetscFree2(funcs,ctxs);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  ierr = PetscCalloc2(numFields,&funcs,numFields,&ctxs);CHKERRQ(ierr);
  funcs[field] = func;
  ctxs[field]  = ctx;
  ierr = DMProjectBdFieldLabelLocal(dm, time, label, numids, ids, Nc, comps, locU, funcs, INSERT_BC_VALUES, locX);CHKERRQ(ierr);
  ierr = PetscFree2(funcs,ctxs);CHKERRQ(ierr);
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
  PetscErrorCode     ierr, ierru = 0;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL);CHKERRQ(ierr);
  nleaves = PetscMax(0, nleaves);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  if (cellGeometry) {
    ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  }
  if (Grad) {
    PetscFV fv;

    ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fv);CHKERRQ(ierr);
    ierr = VecGetDM(Grad, &dmGrad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Grad, &grad);CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, pdim, MPIU_SCALAR, &fx);CHKERRQ(ierr);
  }
  ierr = VecGetArray(locX, &x);CHKERRQ(ierr);
  for (i = 0; i < numids; ++i) {
    IS              faceIS;
    const PetscInt *faces;
    PetscInt        numFaces, f;

    ierr = DMLabelGetStratumIS(label, ids[i], &faceIS);CHKERRQ(ierr);
    if (!faceIS) continue; /* No points with that id on this process */
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      const PetscInt         face = faces[f], *cells;
      PetscFVFaceGeom        *fg;

      if ((face < fStart) || (face >= fEnd)) continue; /* Refinement adds non-faces to labels */
      ierr = PetscFindInt(face, nleaves, (PetscInt *) leaves, &loc);CHKERRQ(ierr);
      if (loc >= 0) continue;
      ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
      if (Grad) {
        PetscFVCellGeom       *cg;
        PetscScalar           *cx, *cgrad;
        PetscScalar           *xG;
        PetscReal              dx[3];
        PetscInt               d;

        ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cg);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dm, cells[0], x, &cx);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmGrad, cells[0], grad, &cgrad);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRef(dm, cells[1], field, x, &xG);CHKERRQ(ierr);
        DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, fg->centroid, dx);
        for (d = 0; d < pdim; ++d) fx[d] = cx[d] + DMPlex_DotD_Internal(dim, &cgrad[d*dim], dx);
        ierru = (*func)(time, fg->centroid, fg->normal, fx, xG, ctx);
        if (ierru) {
          ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
          ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
          goto cleanup;
        }
      } else {
        PetscScalar       *xI;
        PetscScalar       *xG;

        ierr = DMPlexPointLocalRead(dm, cells[0], x, &xI);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRef(dm, cells[1], field, x, &xG);CHKERRQ(ierr);
        ierru = (*func)(time, fg->centroid, fg->normal, xI, xG, ctx);
        if (ierru) {
          ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
          ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
          goto cleanup;
        }
      }
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  cleanup:
  ierr = VecRestoreArray(locX, &x);CHKERRQ(ierr);
  if (Grad) {
    ierr = DMRestoreWorkArray(dm, pdim, MPIU_SCALAR, &fx);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Grad, &grad);CHKERRQ(ierr);
  }
  if (cellGeometry) {ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);}
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) locX, "__Vec_bc_zero__", &isZero);CHKERRQ(ierr);
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

    ierr = PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, &bvfunc, NULL, &ctx);CHKERRQ(ierr);
    if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
    ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      switch (type) {
        /* for FEM, there is no insertion to be done for non-essential boundary conditions */
      case DM_BC_ESSENTIAL:
        {
          PetscSimplePointFunc func = (PetscSimplePointFunc) bvfunc;

          if (isZero) func = zero;
          ierr = DMPlexLabelAddCells(dm,label);CHKERRQ(ierr);
          ierr = DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func, ctx, locX);CHKERRQ(ierr);
          ierr = DMPlexLabelClearCells(dm,label);CHKERRQ(ierr);
        }
        break;
      case DM_BC_ESSENTIAL_FIELD:
        {
          PetscPointFunc func = (PetscPointFunc) bvfunc;

          ierr = DMPlexLabelAddCells(dm,label);CHKERRQ(ierr);
          ierr = DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func, ctx, locX);CHKERRQ(ierr);
          ierr = DMPlexLabelClearCells(dm,label);CHKERRQ(ierr);
        }
        break;
      default: break;
      }
    } else if (id == PETSCFV_CLASSID) {
      {
        PetscErrorCode (*func)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*) = (PetscErrorCode (*)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*)) bvfunc;

        if (!faceGeomFVM) continue;
        ierr = DMPlexInsertBoundaryValuesRiemann(dm, time, faceGeomFVM, cellGeomFVM, gradFVM, field, Nc, comps, label, numids, ids, func, ctx, locX);CHKERRQ(ierr);
      }
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexInsertTimeDerivativeBoundaryValues_Plex(DM dm, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscObject    isZero;
  PetscDS        prob;
  PetscInt       numBd, b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!locX) PetscFunctionReturn(0);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) locX, "__Vec_bc_zero__", &isZero);CHKERRQ(ierr);
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

    ierr = PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, NULL, &bvfunc, &ctx);CHKERRQ(ierr);
    if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
    ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      switch (type) {
        /* for FEM, there is no insertion to be done for non-essential boundary conditions */
      case DM_BC_ESSENTIAL:
        {
          PetscSimplePointFunc func_t = (PetscSimplePointFunc) bvfunc;

          if (isZero) func_t = zero;
          ierr = DMPlexLabelAddCells(dm,label);CHKERRQ(ierr);
          ierr = DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func_t, ctx, locX);CHKERRQ(ierr);
          ierr = DMPlexLabelClearCells(dm,label);CHKERRQ(ierr);
        }
        break;
      case DM_BC_ESSENTIAL_FIELD:
        {
          PetscPointFunc func_t = (PetscPointFunc) bvfunc;

          ierr = DMPlexLabelAddCells(dm,label);CHKERRQ(ierr);
          ierr = DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func_t, ctx, locX);CHKERRQ(ierr);
          ierr = DMPlexLabelClearCells(dm,label);CHKERRQ(ierr);
        }
        break;
      default: break;
      }
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexInsertBoundaryValues - Puts coefficients which represent boundary values into the local solution vector

  Input Parameters:
+ dm - The DM
. insertEssential - Should I insert essential (e.g. Dirichlet) or inessential (e.g. Neumann) boundary conditions
. time - The time
. faceGeomFVM - Face geometry data for FV discretizations
. cellGeomFVM - Cell geometry data for FV discretizations
- gradFVM - Gradient reconstruction data for FV discretizations

  Output Parameters:
. locX - Solution updated with boundary values

  Level: developer

.seealso: DMProjectFunctionLabelLocal()
@*/
PetscErrorCode DMPlexInsertBoundaryValues(DM dm, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 2);
  if (faceGeomFVM) {PetscValidHeaderSpecific(faceGeomFVM, VEC_CLASSID, 4);}
  if (cellGeomFVM) {PetscValidHeaderSpecific(cellGeomFVM, VEC_CLASSID, 5);}
  if (gradFVM)     {PetscValidHeaderSpecific(gradFVM, VEC_CLASSID, 6);}
  ierr = PetscTryMethod(dm,"DMPlexInsertBoundaryValues_C",(DM,PetscBool,Vec,PetscReal,Vec,Vec,Vec),(dm,insertEssential,locX,time,faceGeomFVM,cellGeomFVM,gradFVM));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexInsertTimeDerivativeBoundaryValues - Puts coefficients which represent boundary values of the time derviative into the local solution vector

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (locX_t)      {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 2);}
  if (faceGeomFVM) {PetscValidHeaderSpecific(faceGeomFVM, VEC_CLASSID, 4);}
  if (cellGeomFVM) {PetscValidHeaderSpecific(cellGeomFVM, VEC_CLASSID, 5);}
  if (gradFVM)     {PetscValidHeaderSpecific(gradFVM, VEC_CLASSID, 6);}
  ierr = PetscTryMethod(dm,"DMPlexInsertTimeDerviativeBoundaryValues_C",(DM,PetscBool,Vec,PetscReal,Vec,Vec,Vec),(dm,insertEssential,locX_t,time,faceGeomFVM,cellGeomFVM,gradFVM));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMComputeL2Diff_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  Vec              localX;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, localX, time, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMPlexComputeL2DiffLocal(dm, time, funcs, ctxs, localX, diff);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
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
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &coordDim);CHKERRQ(ierr);
  fegeom.dimEmbed = coordDim;
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetQuadrature(fv, &quad);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
    numComponents += Nc;
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights);CHKERRQ(ierr);
  if ((qNc != 1) && (qNc != numComponents)) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  ierr = PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;
    PetscInt     qc = 0;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);ierr = PetscFEGetDimension((PetscFE) obj, &Nb);CHKERRQ(ierr);}
      else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);Nb = 1;}
      else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %D", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, Nb, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < Nq; ++q) {
        PetscFEGeom qgeom;

        qgeom.dimEmbed = fegeom.dimEmbed;
        qgeom.J        = &fegeom.J[q*coordDim*coordDim];
        qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
        qgeom.detJ     = &fegeom.detJ[q];
        if (fegeom.detJ[q] <= 0.0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, point %D", (double)fegeom.detJ[q], c, q);
        if (transform) {
          gcoords = &coords[coordDim*Nq];
          ierr = DMPlexBasisTransformApplyReal_Internal(dm, &coords[coordDim*q], PETSC_TRUE, coordDim, &coords[coordDim*q], gcoords, dm->transformCtx);CHKERRQ(ierr);
        } else {
          gcoords = &coords[coordDim*q];
        }
        ierr = (*funcs[field])(coordDim, time, gcoords, Nc, funcVal, ctx);
        if (ierr) {
          PetscErrorCode ierr2;
          ierr2 = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr2);
          ierr2 = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr2);
          ierr2 = PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr2);
          CHKERRQ(ierr);
        }
        if (transform) {ierr = DMPlexBasisTransformApply_Internal(dm, &coords[coordDim*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx);CHKERRQ(ierr);}
        if (id == PETSCFE_CLASSID)      {ierr = PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], &qgeom, q, interpolant);CHKERRQ(ierr);}
        else if (id == PETSCFV_CLASSID) {ierr = PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant);CHKERRQ(ierr);}
        else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
        for (fc = 0; fc < Nc; ++fc) {
          const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %D field %D,%D point %g %g %g diff %g\n", c, field, fc, (double)(coordDim > 0 ? coords[coordDim*q] : 0.), (double)(coordDim > 1 ? coords[coordDim*q+1] : 0.),(double)(coordDim > 2 ? coords[coordDim*q+2] : 0.), (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q]));CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
        }
      }
      fieldOffset += Nb;
      qc += Nc;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %D diff %g\n", c, (double)elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr  = PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr);
  ierr  = MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
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
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &coordDim);CHKERRQ(ierr);
  fegeom.dimEmbed = coordDim;
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscFE  fe;
    PetscInt Nc;

    ierr = DMGetField(dm, field, NULL, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    numComponents += Nc;
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights);CHKERRQ(ierr);
  if ((qNc != 1) && (qNc != numComponents)) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  /* ierr = DMProjectFunctionLocal(dm, fe, funcs, INSERT_BC_VALUES, localX);CHKERRQ(ierr); */
  ierr = PetscMalloc6(numComponents,&funcVal,coordDim*Nq,&coords,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ,numComponents*coordDim,&interpolant,Nq,&fegeom.detJ);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;
    PetscInt     qc = 0;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscFE          fe;
      void * const     ctx = ctxs ? ctxs[field] : NULL;
      PetscInt         Nb, Nc, q, fc;

      ierr = DMGetField(dm, field, NULL, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %D", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, Nb, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < Nq; ++q) {
        PetscFEGeom qgeom;

        qgeom.dimEmbed = fegeom.dimEmbed;
        qgeom.J        = &fegeom.J[q*coordDim*coordDim];
        qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
        qgeom.detJ     = &fegeom.detJ[q];
        if (fegeom.detJ[q] <= 0.0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, quadrature points %D", (double)fegeom.detJ[q], c, q);
        if (transform) {
          gcoords = &coords[coordDim*Nq];
          ierr = DMPlexBasisTransformApplyReal_Internal(dm, &coords[coordDim*q], PETSC_TRUE, coordDim, &coords[coordDim*q], gcoords, dm->transformCtx);CHKERRQ(ierr);
        } else {
          gcoords = &coords[coordDim*q];
        }
        ierr = (*funcs[field])(coordDim, time, gcoords, n, Nc, funcVal, ctx);
        if (ierr) {
          PetscErrorCode ierr2;
          ierr2 = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr2);
          ierr2 = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr2);
          ierr2 = PetscFree6(funcVal,coords,fegeom.J,fegeom.invJ,interpolant,fegeom.detJ);CHKERRQ(ierr2);
          CHKERRQ(ierr);
        }
        if (transform) {ierr = DMPlexBasisTransformApply_Internal(dm, &coords[coordDim*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx);CHKERRQ(ierr);}
        ierr = PetscFEInterpolateGradient_Static(fe, 1, &x[fieldOffset], &qgeom, q, interpolant);CHKERRQ(ierr);
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
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %D fieldDer %D,%D diff %g\n", c, field, fc, (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q]));CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
        }
      }
      fieldOffset += Nb;
      qc          += Nc;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %D diff %g\n", c, (double)elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr  = PetscFree6(funcVal,coords,fegeom.J,fegeom.invJ,interpolant,fegeom.detJ);CHKERRQ(ierr);
  ierr  = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr  = MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
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
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dE);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(depthLabel, &depth);CHKERRQ(ierr);

  ierr = VecSet(localX, 0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGetNumDS(dm, &Nds);CHKERRQ(ierr);
  ierr = PetscCalloc1(Nf, &localDiff);CHKERRQ(ierr);
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
    PetscBool        isHybrid;
    PetscInt         qNc, Nq, totNc, cStart = 0, cEnd, c, dsNf;

    ierr = DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds);CHKERRQ(ierr);
    ierr = ISGetIndices(fieldIS, &fields);CHKERRQ(ierr);
    ierr = PetscDSGetHybrid(ds, &isHybrid);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(ds, &dsNf);CHKERRQ(ierr);
    ierr = PetscDSGetTotalComponents(ds, &totNc);CHKERRQ(ierr);
    ierr = PetscDSGetQuadrature(ds, &quad);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if ((qNc != 1) && (qNc != totNc)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, totNc);
    ierr = PetscCalloc6(totNc, &funcVal, totNc, &interpolant, dE*(Nq+1), &coords,Nq, &fegeom.detJ, dE*dE*Nq, &fegeom.J, dE*dE*Nq, &fegeom.invJ);CHKERRQ(ierr);
    if (!label) {
      ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    } else {
      ierr = DMLabelGetStratumIS(label, 1, &pointIS);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pointIS, &cEnd);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    }
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt cell = points ? points[c] : c;
      PetscScalar   *x    = NULL;
      PetscInt       qc   = 0, fOff = 0, dep, fStart = isHybrid ? dsNf-1 : 0;

      ierr = DMLabelGetValue(depthLabel, cell, &dep);CHKERRQ(ierr);
      if (dep != depth-1) continue;
      if (isHybrid) {
        const PetscInt *cone;

        ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(dm, cone[0], quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ);CHKERRQ(ierr);
      } else {
        ierr = DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ);CHKERRQ(ierr);
      }
      ierr = DMPlexVecGetClosure(dm, NULL, localX, cell, NULL, &x);CHKERRQ(ierr);
      for (f = fStart; f < dsNf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        void * const ctx = ctxs ? ctxs[fields[f]] : NULL;
        PetscInt     Nb, Nc, q, fc;
        PetscReal    elemDiff = 0.0;

        ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);ierr = PetscFEGetDimension((PetscFE) obj, &Nb);CHKERRQ(ierr);}
        else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);Nb = 1;}
        else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", fields[f]);
        if (debug) {
          char title[1024];
          ierr = PetscSNPrintf(title, 1023, "Solution for Field %D", fields[f]);CHKERRQ(ierr);
          ierr = DMPrintCellVector(cell, title, Nb, &x[fOff]);CHKERRQ(ierr);
        }
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*dE*dE];
          qgeom.invJ     = &fegeom.invJ[q*dE*dE];
          qgeom.detJ     = &fegeom.detJ[q];
          if (fegeom.detJ[q] <= 0.0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %D, quadrature point %D", (double)fegeom.detJ[q], cell, q);
          if (transform) {
            gcoords = &coords[dE*Nq];
            ierr = DMPlexBasisTransformApplyReal_Internal(dm, &coords[dE*q], PETSC_TRUE, dE, &coords[dE*q], gcoords, dm->transformCtx);CHKERRQ(ierr);
          } else {
            gcoords = &coords[dE*q];
          }
          ierr = (*funcs[fields[f]])(dE, time, gcoords, Nc, funcVal, ctx);
          if (ierr) {
            PetscErrorCode ierr2;
            ierr2 = DMPlexVecRestoreClosure(dm, NULL, localX, cell, NULL, &x);CHKERRQ(ierr2);
            ierr2 = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr2);
            ierr2 = PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr2);
            CHKERRQ(ierr);
          }
          if (transform) {ierr = DMPlexBasisTransformApply_Internal(dm, &coords[dE*q], PETSC_FALSE, Nc, funcVal, funcVal, dm->transformCtx);CHKERRQ(ierr);}
          /* Call once for each face, except for lagrange field */
          if (id == PETSCFE_CLASSID)      {ierr = PetscFEInterpolate_Static((PetscFE) obj, &x[fOff], &qgeom, q, interpolant);CHKERRQ(ierr);}
          else if (id == PETSCFV_CLASSID) {ierr = PetscFVInterpolate_Static((PetscFV) obj, &x[fOff], q, interpolant);CHKERRQ(ierr);}
          else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", fields[f]);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
            if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    cell %D field %D,%D point %g %g %g diff %g\n", cell, fields[f], fc, (double)(dE > 0 ? coords[dE*q] : 0.), (double)(dE > 1 ? coords[dE*q+1] : 0.),(double)(dE > 2 ? coords[dE*q+2] : 0.), (double)(PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q]));CHKERRQ(ierr);}
            elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
          }
        }
        fOff += Nb;
        qc   += Nc;
        localDiff[fields[f]] += elemDiff;
        if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  cell %D field %D cum diff %g\n", cell, fields[f], (double)localDiff[fields[f]]);CHKERRQ(ierr);}
      }
      ierr = DMPlexVecRestoreClosure(dm, NULL, localX, cell, NULL, &x);CHKERRQ(ierr);
    }
    if (label) {
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(fieldIS, &fields);CHKERRQ(ierr);
    ierr = PetscFree6(funcVal, interpolant, coords, fegeom.detJ, fegeom.J, fegeom.invJ);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(localDiff, diff, Nf, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
  ierr = PetscFree(localDiff);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecSet(D, 0.0);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &coordDim);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dm, time, funcs, ctxs, INSERT_BC_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetQuadrature(fv, &quad);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
    numComponents += Nc;
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if ((qNc != 1) && (qNc != numComponents)) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  ierr = PetscMalloc6(numComponents,&funcVal,numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscScalar  elemDiff = 0.0;
    PetscInt     qc = 0;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, fieldOffset = 0; field < numFields; ++field) {
      PetscObject  obj;
      PetscClassId id;
      void * const ctx = ctxs ? ctxs[field] : NULL;
      PetscInt     Nb, Nc, q, fc;

      ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);ierr = PetscFEGetDimension((PetscFE) obj, &Nb);CHKERRQ(ierr);}
      else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);Nb = 1;}
      else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
      if (funcs[field]) {
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*coordDim*coordDim];
          qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
          qgeom.detJ     = &fegeom.detJ[q];
          if (fegeom.detJ[q] <= 0.0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, quadrature points %D", (double)fegeom.detJ[q], c, q);
          ierr = (*funcs[field])(coordDim, time, &coords[q*coordDim], Nc, funcVal, ctx);
          if (ierr) {
            PetscErrorCode ierr2;
            ierr2 = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr2);
            ierr2 = PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr2);
            ierr2 = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr2);
            CHKERRQ(ierr);
          }
          if (id == PETSCFE_CLASSID)      {ierr = PetscFEInterpolate_Static((PetscFE) obj, &x[fieldOffset], &qgeom, q, interpolant);CHKERRQ(ierr);}
          else if (id == PETSCFV_CLASSID) {ierr = PetscFVInterpolate_Static((PetscFV) obj, &x[fieldOffset], q, interpolant);CHKERRQ(ierr);}
          else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+(qNc == 1 ? 0 : qc+fc)];
            elemDiff += PetscSqr(PetscRealPart(interpolant[fc] - funcVal[fc]))*wt*fegeom.detJ[q];
          }
        }
      }
      fieldOffset += Nb;
      qc          += Nc;
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    ierr = VecSetValue(D, c - cStart, elemDiff, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(funcVal,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = VecSqrtAbs(D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexComputeGradientClementInterpolant - This function computes the L2 projection of the cellwise gradient of a function u onto P1, and stores it in a Vec.

  Collective on dm

  Input Parameters:
+ dm - The DM
- LocX  - The coefficient vector u_h

  Output Parameter:
. locC - A Vec which holds the Clement interpolant of the gradient

  Notes:
    Add citation to (Clement, 1975) and definition of the interpolant
  \nabla u_h(v_i) = \sum_{T_i \in support(v_i)} |T_i| \nabla u_h(T_i) / \sum_{T_i \in support(v_i)} |T_i| where |T_i| is the cell volume

  Level: developer

.seealso: DMProjectFunction(), DMComputeL2Diff(), DMPlexComputeL2FieldDiff(), DMComputeL2GradientDiff()
@*/
PetscErrorCode DMPlexComputeGradientClementInterpolant(DM dm, Vec locX, Vec locC)
{
  DM_Plex         *mesh  = (DM_Plex *) dm->data;
  PetscInt         debug = mesh->printFEM;
  DM               dmC;
  PetscSection     section;
  PetscQuadrature  quad;
  PetscScalar     *interpolant, *gradsum;
  PetscFEGeom      fegeom;
  PetscReal       *coords;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, coordDim, numFields, numComponents = 0, qNc, Nq, cStart, cEnd, vStart, vEnd, v, field, fieldOffset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(locC, &dmC);CHKERRQ(ierr);
  ierr = VecSet(locC, 0.0);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &coordDim);CHKERRQ(ierr);
  fegeom.dimEmbed = coordDim;
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetQuadrature(fv, &quad);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
    numComponents += Nc;
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if ((qNc != 1) && (qNc != numComponents)) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  ierr = PetscMalloc6(coordDim*numComponents*2,&gradsum,coordDim*numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar volsum = 0.0;
    PetscInt   *star = NULL;
    PetscInt    starSize, st, d, fc;

    ierr = PetscArrayzero(gradsum, coordDim*numComponents);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (st = 0; st < starSize*2; st += 2) {
      const PetscInt cell = star[st];
      PetscScalar   *grad = &gradsum[coordDim*numComponents];
      PetscScalar   *x    = NULL;
      PetscReal      vol  = 0.0;

      if ((cell < cStart) || (cell >= cEnd)) continue;
      ierr = DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(dm, NULL, locX, cell, NULL, &x);CHKERRQ(ierr);
      for (field = 0, fieldOffset = 0; field < numFields; ++field) {
        PetscObject  obj;
        PetscClassId id;
        PetscInt     Nb, Nc, q, qc = 0;

        ierr = PetscArrayzero(grad, coordDim*numComponents);CHKERRQ(ierr);
        ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);ierr = PetscFEGetDimension((PetscFE) obj, &Nb);CHKERRQ(ierr);}
        else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);Nb = 1;}
        else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
        for (q = 0; q < Nq; ++q) {
          PetscFEGeom qgeom;

          qgeom.dimEmbed = fegeom.dimEmbed;
          qgeom.J        = &fegeom.J[q*coordDim*coordDim];
          qgeom.invJ     = &fegeom.invJ[q*coordDim*coordDim];
          qgeom.detJ     = &fegeom.detJ[q];
          if (fegeom.detJ[q] <= 0.0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, quadrature points %D", (double)fegeom.detJ[q], cell, q);
          if (ierr) {
            PetscErrorCode ierr2;
            ierr2 = DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x);CHKERRQ(ierr2);
            ierr2 = DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr2);
            ierr2 = PetscFree6(gradsum,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr2);
            CHKERRQ(ierr);
          }
          if (id == PETSCFE_CLASSID)      {ierr = PetscFEInterpolateGradient_Static((PetscFE) obj, 1, &x[fieldOffset], &qgeom, q, interpolant);CHKERRQ(ierr);}
          else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+qc+fc];

            for (d = 0; d < coordDim; ++d) grad[fc*coordDim+d] += interpolant[fc*dim+d]*wt*fegeom.detJ[q];
          }
          vol += quadWeights[q*qNc]*fegeom.detJ[q];
        }
        fieldOffset += Nb;
        qc          += Nc;
      }
      ierr = DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x);CHKERRQ(ierr);
      for (fc = 0; fc < numComponents; ++fc) {
        for (d = 0; d < coordDim; ++d) {
          gradsum[fc*coordDim+d] += grad[fc*coordDim+d];
        }
      }
      volsum += vol;
      if (debug) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D gradient: [", cell);CHKERRQ(ierr);
        for (fc = 0; fc < numComponents; ++fc) {
          for (d = 0; d < coordDim; ++d) {
            if (fc || d > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");CHKERRQ(ierr);}
            ierr = PetscPrintf(PETSC_COMM_SELF, "%g", (double)PetscRealPart(grad[fc*coordDim+d]));CHKERRQ(ierr);
          }
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "]\n");CHKERRQ(ierr);
      }
    }
    for (fc = 0; fc < numComponents; ++fc) {
      for (d = 0; d < coordDim; ++d) gradsum[fc*coordDim+d] /= volsum;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    ierr = DMPlexVecSetClosure(dmC, NULL, locC, v, gradsum, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(gradsum,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  /* Determine which discretizations we have */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFV_CLASSID) useFVM = PETSC_TRUE;
  }
  /* Get local solution with boundary values */
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, 0.0, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  /* Read DS information */
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,numCells,cStart,1,&cellIS);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  /* Read Auxiliary DS information */
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
  }
  /* Allocate data  arrays */
  ierr = PetscCalloc1(numCells*totDim, &u);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  /* Read out geometry */
  ierr = DMGetCoordinateField(dm,&coordField);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
  if (maxDegree <= 1) {
    ierr = DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad);CHKERRQ(ierr);
    if (affineQuad) {
      ierr = DMFieldCreateFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&cgeomFEM);CHKERRQ(ierr);
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

      ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFV_CLASSID) {fv = (PetscFV) obj; break;}
    }
    ierr = PetscFVGetComputeGradients(fv, &compGrad);CHKERRQ(ierr);
    ierr = PetscFVSetComputeGradients(fv, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMPlexComputeGeometryFVM(dm, &cellGeometryFVM, &faceGeometryFVM);CHKERRQ(ierr);
    ierr = DMPlexComputeGradientFVM(dm, fv, faceGeometryFVM, cellGeometryFVM, &dmGrad);CHKERRQ(ierr);
    ierr = PetscFVSetComputeGradients(fv, compGrad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);
    /* Reconstruct and limit cell gradients */
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dmGrad, &grad);CHKERRQ(ierr);
    ierr = DMPlexReconstructGradients_Internal(dm, fv, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad);CHKERRQ(ierr);
    /* Communicate gradient values */
    ierr = DMGetLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmGrad, &grad);CHKERRQ(ierr);
    /* Handle non-essential (e.g. outflow) boundary values */
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, 0.0, faceGeometryFVM, cellGeometryFVM, locGrad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGrad, &lgrad);CHKERRQ(ierr);
  }
  /* Read out data from inputs */
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscInt     i;

    ierr = DMPlexVecGetClosure(dm, section, locX, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, locX, c, NULL, &x);CHKERRQ(ierr);
    if (dmAux) {
      ierr = DMPlexVecGetClosure(dmAux, sectionAux, locA, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, locA, c, NULL, &x);CHKERRQ(ierr);
    }
  }
  /* Do integration for each field */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE         fe = (PetscFE) obj;
      PetscQuadrature q;
      PetscFEGeom     *chunkGeom = NULL;
      PetscInt        Nq, Nb;

      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      blockSize = Nb*Nq;
      batchSize = numBlocks * blockSize;
      ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      if (!affineQuad) {
        ierr = DMFieldCreateFEGeom(coordField,cellIS,q,PETSC_FALSE,&cgeomFEM);CHKERRQ(ierr);
      }
      ierr = PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrate(prob, f, Ne, chunkGeom, u, probAux, a, cintegral);CHKERRQ(ierr);
      ierr = PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrate(prob, f, Nr, chunkGeom, &u[offset*totDim], probAux, &a[offset*totDimAux], &cintegral[offset*Nf]);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&chunkGeom);CHKERRQ(ierr);
      if (!affineQuad) {
        ierr = PetscFEGeomDestroy(&cgeomFEM);CHKERRQ(ierr);
      }
    } else if (id == PETSCFV_CLASSID) {
      PetscInt       foff;
      PetscPointFunc obj_func;
      PetscScalar    lint;

      ierr = PetscDSGetObjective(prob, f, &obj_func);CHKERRQ(ierr);
      ierr = PetscDSGetFieldOffset(prob, f, &foff);CHKERRQ(ierr);
      if (obj_func) {
        for (c = 0; c < numCells; ++c) {
          PetscScalar *u_x;

          ierr = DMPlexPointLocalRead(dmGrad, c, lgrad, &u_x);CHKERRQ(ierr);
          obj_func(dim, Nf, NfAux, uOff, uOff_x, &u[totDim*c+foff], NULL, u_x, aOff, NULL, &a[totDimAux*c], NULL, NULL, 0.0, cgeomFVM[c].centroid, numConstants, constants, &lint);
          cintegral[c*Nf+f] += PetscRealPart(lint)*cgeomFVM[c].volume;
        }
      }
    } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
  }
  /* Cleanup data arrays */
  if (useFVM) {
    ierr = VecRestoreArrayRead(locGrad, &lgrad);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);
    ierr = VecDestroy(&faceGeometryFVM);CHKERRQ(ierr);
    ierr = VecDestroy(&cellGeometryFVM);CHKERRQ(ierr);
    ierr = DMDestroy(&dmGrad);CHKERRQ(ierr);
  }
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
  ierr = PetscFree(u);CHKERRQ(ierr);
  /* Cleanup */
  if (affineQuad) {
    ierr = PetscFEGeomDestroy(&cgeomFEM);CHKERRQ(ierr);
  }
  ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
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

.seealso: DMPlexComputeResidualFEM()
@*/
PetscErrorCode DMPlexComputeIntegralFEM(DM dm, Vec X, PetscScalar *integral, void *user)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  PetscScalar   *cintegral, *lintegral;
  PetscInt       Nf, f, cellHeight, cStart, cEnd, cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidPointer(integral, 3);
  ierr = PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  /* TODO Introduce a loop over large chunks (right now this is a single chunk) */
  ierr = PetscCalloc2(Nf, &lintegral, (cEnd-cStart)*Nf, &cintegral);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegral_Internal(dm, X, cStart, cEnd, cintegral, user);CHKERRQ(ierr);
  /* Sum up values */
  for (cell = cStart; cell < cEnd; ++cell) {
    const PetscInt c = cell - cStart;

    if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, "Cell Integral", Nf, &cintegral[c*Nf]);CHKERRQ(ierr);}
    for (f = 0; f < Nf; ++f) lintegral[f] += cintegral[c*Nf+f];
  }
  ierr = MPIU_Allreduce(lintegral, integral, Nf, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
  if (mesh->printFEM) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Integral:");CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), " %g", (double) PetscRealPart(integral[f]));CHKERRQ(ierr);}
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "\n");CHKERRQ(ierr);
  }
  ierr = PetscFree2(lintegral, cintegral);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);
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

.seealso: DMPlexComputeResidualFEM()
@*/
PetscErrorCode DMPlexComputeCellwiseIntegralFEM(DM dm, Vec X, Vec F, void *user)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  DM             dmF;
  PetscSection   sectionF;
  PetscScalar   *cintegral, *af;
  PetscInt       Nf, f, cellHeight, cStart, cEnd, cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  ierr = PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  /* TODO Introduce a loop over large chunks (right now this is a single chunk) */
  ierr = PetscCalloc1((cEnd-cStart)*Nf, &cintegral);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegral_Internal(dm, X, cStart, cEnd, cintegral, user);CHKERRQ(ierr);
  /* Put values in F*/
  ierr = VecGetDM(F, &dmF);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmF, &sectionF);CHKERRQ(ierr);
  ierr = VecGetArray(F, &af);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    const PetscInt c = cell - cStart;
    PetscInt       dof, off;

    if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, "Cell Integral", Nf, &cintegral[c*Nf]);CHKERRQ(ierr);}
    ierr = PetscSectionGetDof(sectionF, cell, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionF, cell, &off);CHKERRQ(ierr);
    if (dof != Nf) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of cell dofs %D != %D", dof, Nf);
    for (f = 0; f < Nf; ++f) af[off+f] = cintegral[c*Nf+f];
  }
  ierr = VecRestoreArray(F, &af);CHKERRQ(ierr);
  ierr = PetscFree(cintegral);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  /* Determine which discretizations we have */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFV_CLASSID) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Not supported for FVM (field %D)", f);
  }
  /* Read DS information */
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  /* Read Auxiliary DS information */
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &locA);CHKERRQ(ierr);
  if (locA) {
    DM dmAux;

    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plexA);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
  }
  /* Integrate over points */
  {
    PetscFEGeom    *fgeom, *chunkGeom = NULL;
    PetscInt        maxDegree;
    PetscQuadrature qGeom = NULL;
    const PetscInt *points;
    PetscInt        numFaces, face, Nq, field;
    PetscInt        numChunks, chunkSize, chunk, Nr, offset;

    ierr = ISGetLocalSize(pointIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = PetscCalloc2(numFaces*totDim, &u, locA ? numFaces*totDimAux : 0, &a);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField, pointIS, NULL, &maxDegree);CHKERRQ(ierr);
    for (field = 0; field < Nf; ++field) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
      if (maxDegree <= 1) {ierr = DMFieldCreateDefaultQuadrature(coordField, pointIS, &qGeom);CHKERRQ(ierr);}
      if (!qGeom) {
        ierr = PetscFEGetFaceQuadrature(fe, &qGeom);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject) qGeom);CHKERRQ(ierr);
      }
      ierr = PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = DMPlexGetFEGeom(coordField, pointIS, qGeom, PETSC_TRUE, &fgeom);CHKERRQ(ierr);
      for (face = 0; face < numFaces; ++face) {
        const PetscInt point = points[face], *support;
        PetscScalar    *x    = NULL;
        PetscInt       i;

        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
        if (locA) {
          PetscInt subp;
          ierr = DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp);CHKERRQ(ierr);
          ierr = DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
          for (i = 0; i < totDimAux; ++i) a[f*totDimAux+i] = x[i];
          ierr = DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
        }
      }
      /* Get blocking */
      {
        PetscQuadrature q;
        PetscInt        numBatches, batchSize, numBlocks, blockSize;
        PetscInt        Nq, Nb;

        ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
        ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
        blockSize = Nb*Nq;
        batchSize = numBlocks * blockSize;
        chunkSize = numBatches*batchSize;
        ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
        numChunks = numFaces / chunkSize;
        Nr        = numFaces % chunkSize;
        offset    = numFaces - Nr;
      }
      /* Do integration for each field */
      for (chunk = 0; chunk < numChunks; ++chunk) {
        ierr = PetscFEGeomGetChunk(fgeom, chunk*chunkSize, (chunk+1)*chunkSize, &chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateBd(prob, field, func, chunkSize, chunkGeom, u, probAux, a, fintegral);CHKERRQ(ierr);
        ierr = PetscFEGeomRestoreChunk(fgeom, 0, offset, &chunkGeom);CHKERRQ(ierr);
      }
      ierr = PetscFEGeomGetChunk(fgeom, offset, numFaces, &chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrateBd(prob, field, func, Nr, chunkGeom, &u[offset*totDim], probAux, a ? &a[offset*totDimAux] : NULL, &fintegral[offset*Nf]);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(fgeom, offset, numFaces, &chunkGeom);CHKERRQ(ierr);
      /* Cleanup data arrays */
      ierr = DMPlexRestoreFEGeom(coordField, pointIS, qGeom, PETSC_TRUE, &fgeom);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
      ierr = PetscFree2(u, a);CHKERRQ(ierr);
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    }
  }
  if (plex)  {ierr = DMDestroy(&plex);CHKERRQ(ierr);}
  if (plexA) {ierr = DMDestroy(&plexA);CHKERRQ(ierr);}
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
. func    = The function to integrate along the boundary
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidPointer(label, 3);
  if (vals) PetscValidPointer(vals, 5);
  PetscValidPointer(integral, 6);
  ierr = PetscLogEventBegin(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, dim-1, &facetIS);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  /* Get local solution with boundary values */
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, 0.0, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  /* Loop over label values */
  ierr = PetscArrayzero(integral, Nf);CHKERRQ(ierr);
  for (v = 0; v < numVals; ++v) {
    IS           pointIS;
    PetscInt     numFaces, face;
    PetscScalar *fintegral;

    ierr = DMLabelGetStratumIS(label, vals[v], &pointIS);CHKERRQ(ierr);
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a priori if one is a superset of the other */
      ierr = ISIntersect_Caching_Internal(facetIS, pointIS, &isectIS);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      pointIS = isectIS;
    }
    ierr = ISGetLocalSize(pointIS, &numFaces);CHKERRQ(ierr);
    ierr = PetscCalloc1(numFaces*Nf, &fintegral);CHKERRQ(ierr);
    ierr = DMPlexComputeBdIntegral_Internal(dm, locX, pointIS, func, fintegral, user);CHKERRQ(ierr);
    /* Sum point contributions into integral */
    for (f = 0; f < Nf; ++f) for (face = 0; face < numFaces; ++face) integral[f] += fintegral[face*Nf+f];
    ierr = PetscFree(fintegral);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_IntegralFEM,dm,0,0,0);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_InterpolatorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(dmf, &dim);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmf, &fsection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmf, &fglobalSection);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmc, &csection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmc, &cglobalSection);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(fsection, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dmc, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscCalloc2(Nf, &feRef, Nf, &fvRef);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj, objc;
    PetscClassId id, idc;
    PetscInt     rNb = 0, Nc = 0, cNb = 0;

    ierr = DMGetField(dmf, f, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      if (isRefined) {
        ierr = PetscFERefine(fe, &feRef[f]);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject) fe);CHKERRQ(ierr);
        feRef[f] = fe;
      }
      ierr = PetscFEGetDimension(feRef[f], &rNb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      if (isRefined) {
        ierr = PetscFVRefine(fv, &fvRef[f]);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject) fv);CHKERRQ(ierr);
        fvRef[f] = fv;
      }
      ierr = PetscFVGetDualSpace(fvRef[f], &Q);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(Q, &rNb);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fv, &Q);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    }
    ierr = DMGetField(dmc, f, NULL, &objc);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(objc, &idc);CHKERRQ(ierr);
    if (idc == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) objc;

      ierr = PetscFEGetDimension(fe, &cNb);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      ierr = PetscFVGetDualSpace(fv, &Q);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(Q, &cNb);CHKERRQ(ierr);
    }
    rTotDim += rNb;
    cTotDim += cNb;
  }
  ierr = PetscMalloc1(rTotDim*cTotDim,&elemMat);CHKERRQ(ierr);
  ierr = PetscArrayzero(elemMat, rTotDim*cTotDim);CHKERRQ(ierr);
  for (fieldI = 0, offsetI = 0; fieldI < Nf; ++fieldI) {
    PetscDualSpace   Qref;
    PetscQuadrature  f;
    const PetscReal *qpoints, *qweights;
    PetscReal       *points;
    PetscInt         npoints = 0, Nc, Np, fpdim, i, k, p, d;

    /* Compose points from all dual basis functionals */
    if (feRef[fieldI]) {
      ierr = PetscFEGetDualSpace(feRef[fieldI], &Qref);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(feRef[fieldI], &Nc);CHKERRQ(ierr);
    } else {
      ierr = PetscFVGetDualSpace(fvRef[fieldI], &Qref);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fvRef[fieldI], &Nc);CHKERRQ(ierr);
    }
    ierr = PetscDualSpaceGetDimension(Qref, &fpdim);CHKERRQ(ierr);
    for (i = 0; i < fpdim; ++i) {
      ierr = PetscDualSpaceGetFunctional(Qref, i, &f);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(f, NULL, NULL, &Np, NULL, NULL);CHKERRQ(ierr);
      npoints += Np;
    }
    ierr = PetscMalloc1(npoints*dim,&points);CHKERRQ(ierr);
    for (i = 0, k = 0; i < fpdim; ++i) {
      ierr = PetscDualSpaceGetFunctional(Qref, i, &f);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(f, NULL, NULL, &Np, &qpoints, NULL);CHKERRQ(ierr);
      for (p = 0; p < Np; ++p, ++k) for (d = 0; d < dim; ++d) points[k*dim+d] = qpoints[p*dim+d];
    }

    for (fieldJ = 0, offsetJ = 0; fieldJ < Nf; ++fieldJ) {
      PetscObject  obj;
      PetscClassId id;
      PetscInt     NcJ = 0, cpdim = 0, j, qNc;

      ierr = DMGetField(dmc, fieldJ, NULL, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        PetscFE           fe = (PetscFE) obj;
        PetscTabulation T  = NULL;

        /* Evaluate basis at points */
        ierr = PetscFEGetNumComponents(fe, &NcJ);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &cpdim);CHKERRQ(ierr);
        /* For now, fields only interpolate themselves */
        if (fieldI == fieldJ) {
          if (Nc != NcJ) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %D does not match coarse field %D", Nc, NcJ);
          ierr = PetscFECreateTabulation(fe, 1, npoints, points, 0, &T);CHKERRQ(ierr);
          for (i = 0, k = 0; i < fpdim; ++i) {
            ierr = PetscDualSpaceGetFunctional(Qref, i, &f);CHKERRQ(ierr);
            ierr = PetscQuadratureGetData(f, NULL, &qNc, &Np, NULL, &qweights);CHKERRQ(ierr);
            if (qNc != NcJ) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %D does not match coarse field %D", qNc, NcJ);
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
          ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);CHKERRQ(ierr);
        }
      } else if (id == PETSCFV_CLASSID) {
        PetscFV        fv = (PetscFV) obj;

        /* Evaluate constant function at points */
        ierr = PetscFVGetNumComponents(fv, &NcJ);CHKERRQ(ierr);
        cpdim = 1;
        /* For now, fields only interpolate themselves */
        if (fieldI == fieldJ) {
          if (Nc != NcJ) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %D does not match coarse field %D", Nc, NcJ);
          for (i = 0, k = 0; i < fpdim; ++i) {
            ierr = PetscDualSpaceGetFunctional(Qref, i, &f);CHKERRQ(ierr);
            ierr = PetscQuadratureGetData(f, NULL, &qNc, &Np, NULL, &qweights);CHKERRQ(ierr);
            if (qNc != NcJ) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %D does not match coarse field %D", qNc, NcJ);
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
    ierr = PetscFree(points);CHKERRQ(ierr);
  }
  if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(0, name, rTotDim, cTotDim, elemMat);CHKERRQ(ierr);}
  /* Preallocate matrix */
  {
    Mat          preallocator;
    PetscScalar *vals;
    PetscInt    *cellCIndices, *cellFIndices;
    PetscInt     locRows, locCols, cell;

    ierr = MatGetLocalSize(In, &locRows, &locCols);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject) In), &preallocator);CHKERRQ(ierr);
    ierr = MatSetType(preallocator, MATPREALLOCATOR);CHKERRQ(ierr);
    ierr = MatSetSizes(preallocator, locRows, locCols, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetUp(preallocator);CHKERRQ(ierr);
    ierr = PetscCalloc3(rTotDim*cTotDim, &vals,cTotDim,&cellCIndices,rTotDim,&cellFIndices);CHKERRQ(ierr);
    for (cell = cStart; cell < cEnd; ++cell) {
      if (isRefined) {
        ierr = DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, cell, cellCIndices, cellFIndices);CHKERRQ(ierr);
        ierr = MatSetValues(preallocator, rTotDim, cellFIndices, cTotDim, cellCIndices, vals, INSERT_VALUES);CHKERRQ(ierr);
      } else {
        ierr = DMPlexMatSetClosureGeneral(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, preallocator, cell, vals, INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree3(vals,cellCIndices,cellFIndices);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatPreallocatorPreallocate(preallocator, PETSC_TRUE, In);CHKERRQ(ierr);
    ierr = MatDestroy(&preallocator);CHKERRQ(ierr);
  }
  /* Fill matrix */
  ierr = MatZeroEntries(In);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    if (isRefined) {
      ierr = DMPlexMatSetClosureRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, In, c, elemMat, INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = DMPlexMatSetClosureGeneral(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, In, c, elemMat, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&feRef[f]);CHKERRQ(ierr);}
  ierr = PetscFree2(feRef,fvRef);CHKERRQ(ierr);
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(In, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(In, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (mesh->printFEM) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)In), "%s:\n", name);CHKERRQ(ierr);
    ierr = MatChop(In, 1.0e-10);CHKERRQ(ierr);
    ierr = MatView(In, NULL);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_InterpolatorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_InterpolatorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dmc, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(dmc, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(prob, &x, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0c,dim*dim,&Jc,dim*dim,&invJc);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmf, &fsection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmf, &globalFSection);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmc, &csection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmc, &globalCSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmf, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscMalloc1(totDim, &elemMat);CHKERRQ(ierr);

  ierr = MatGetLocalSize(In, &locRows, NULL);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject) In), &rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(rLayout, locRows);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(rLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rLayout, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rLayout);CHKERRQ(ierr);
  ierr = PetscCalloc2(locRows,&dnz,locRows,&onz);CHKERRQ(ierr);
  ierr = PetscHSetIJCreate(&ht);CHKERRQ(ierr);
  for (field = 0; field < Nf; ++field) {
    PetscObject      obj;
    PetscClassId     id;
    PetscDualSpace   Q = NULL;
    PetscQuadrature  f;
    const PetscReal *qpoints;
    PetscInt         Nc, Np, fpdim, i, d;

    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetDualSpace(fe, &Q);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetDualSpace(fv, &Q);CHKERRQ(ierr);
      Nc   = 1;
    }
    ierr = PetscDualSpaceGetDimension(Q, &fpdim);CHKERRQ(ierr);
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt *findices,   *cindices;
      PetscInt  numFIndices, numCIndices;

      ierr = DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      if (numFIndices != fpdim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fine indices %D != %D dual basis vecs", numFIndices, fpdim);
      for (i = 0; i < fpdim; ++i) {
        Vec             pointVec;
        PetscScalar    *pV;
        PetscSF         coarseCellSF = NULL;
        const PetscSFNode *coarseCells;
        PetscInt        numCoarseCells, q, c;

        /* Get points from the dual basis functional quadrature */
        ierr = PetscDualSpaceGetFunctional(Q, i, &f);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(f, NULL, NULL, &Np, &qpoints, NULL);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, Np*dim, &pointVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(pointVec, dim);CHKERRQ(ierr);
        ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
        for (q = 0; q < Np; ++q) {
          const PetscReal xi0[3] = {-1., -1., -1.};

          /* Transform point to real space */
          CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
          for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
        }
        ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
        /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
        /* OPT: Pack all quad points from fine cell */
        ierr = DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF);CHKERRQ(ierr);
        ierr = PetscSFViewFromOptions(coarseCellSF, NULL, "-interp_sf_view");CHKERRQ(ierr);
        /* Update preallocation info */
        ierr = PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells);CHKERRQ(ierr);
        if (numCoarseCells != Np) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
        {
          PetscHashIJKey key;
          PetscBool      missing;

          key.i = findices[i];
          if (key.i >= 0) {
            /* Get indices for coarse elements */
            for (ccell = 0; ccell < numCoarseCells; ++ccell) {
              ierr = DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL);CHKERRQ(ierr);
              for (c = 0; c < numCIndices; ++c) {
                key.j = cindices[c];
                if (key.j < 0) continue;
                ierr = PetscHSetIJQueryAdd(ht, key, &missing);CHKERRQ(ierr);
                if (missing) {
                  if ((key.j >= rStart) && (key.j < rEnd)) ++dnz[key.i-rStart];
                  else                                     ++onz[key.i-rStart];
                }
              }
              ierr = DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL);CHKERRQ(ierr);
            }
          }
        }
        ierr = PetscSFDestroy(&coarseCellSF);CHKERRQ(ierr);
        ierr = VecDestroy(&pointVec);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscHSetIJDestroy(&ht);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(In, 1, dnz, onz, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetOption(In, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);
  for (field = 0; field < Nf; ++field) {
    PetscObject       obj;
    PetscClassId      id;
    PetscDualSpace    Q = NULL;
    PetscTabulation T = NULL;
    PetscQuadrature   f;
    const PetscReal  *qpoints, *qweights;
    PetscInt          Nc, qNc, Np, fpdim, i, d;

    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetDualSpace(fe, &Q);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
      ierr = PetscFECreateTabulation(fe, 1, 1, x, 0, &T);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetDualSpace(fv, &Q);CHKERRQ(ierr);
      Nc   = 1;
    } else SETERRQ1(PetscObjectComm((PetscObject)dmc),PETSC_ERR_ARG_WRONG,"Unknown discretization type for field %D",field);
    ierr = PetscDualSpaceGetDimension(Q, &fpdim);CHKERRQ(ierr);
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt *findices,   *cindices;
      PetscInt  numFIndices, numCIndices;

      ierr = DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      if (numFIndices != fpdim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fine indices %D != %D dual basis vecs", numFIndices, fpdim);
      for (i = 0; i < fpdim; ++i) {
        Vec             pointVec;
        PetscScalar    *pV;
        PetscSF         coarseCellSF = NULL;
        const PetscSFNode *coarseCells;
        PetscInt        numCoarseCells, cpdim, q, c, j;

        /* Get points from the dual basis functional quadrature */
        ierr = PetscDualSpaceGetFunctional(Q, i, &f);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(f, NULL, &qNc, &Np, &qpoints, &qweights);CHKERRQ(ierr);
        if (qNc != Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in quadrature %D does not match coarse field %D", qNc, Nc);
        ierr = VecCreateSeq(PETSC_COMM_SELF, Np*dim, &pointVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(pointVec, dim);CHKERRQ(ierr);
        ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
        for (q = 0; q < Np; ++q) {
          const PetscReal xi0[3] = {-1., -1., -1.};

          /* Transform point to real space */
          CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
          for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
        }
        ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
        /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
        /* OPT: Read this out from preallocation information */
        ierr = DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF);CHKERRQ(ierr);
        /* Update preallocation info */
        ierr = PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells);CHKERRQ(ierr);
        if (numCoarseCells != Np) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
        ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
        for (ccell = 0; ccell < numCoarseCells; ++ccell) {
          PetscReal pVReal[3];
          const PetscReal xi0[3] = {-1., -1., -1.};

          ierr = DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL);CHKERRQ(ierr);
          /* Transform points from real space to coarse reference space */
          ierr = DMPlexComputeCellGeometryFEM(dmc, coarseCells[ccell].index, NULL, v0c, Jc, invJc, &detJc);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) pVReal[d] = PetscRealPart(pV[ccell*dim+d]);
          CoordinatesRealToRef(dim, dim, xi0, v0c, invJc, pVReal, x);

          if (id == PETSCFE_CLASSID) {
            PetscFE fe = (PetscFE) obj;

            /* Evaluate coarse basis on contained point */
            ierr = PetscFEGetDimension(fe, &cpdim);CHKERRQ(ierr);
            ierr = PetscFEComputeTabulation(fe, 1, x, 0, T);CHKERRQ(ierr);
            ierr = PetscArrayzero(elemMat, cpdim);CHKERRQ(ierr);
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
          if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat);CHKERRQ(ierr);}
          if (numCIndices != cpdim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %D != %D", numCIndices, cpdim);
          ierr = MatSetValues(In, 1, &findices[i], numCIndices, cindices, elemMat, INSERT_VALUES);CHKERRQ(ierr);
          ierr = DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL);CHKERRQ(ierr);
        }
        ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&coarseCellSF);CHKERRQ(ierr);
        ierr = VecDestroy(&pointVec);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL);CHKERRQ(ierr);
    }
    if (id == PETSCFE_CLASSID) {ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);}
  }
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  ierr = PetscFree3(v0c,Jc,invJc);CHKERRQ(ierr);
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(In, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(In, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_InterpolatorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(dmc, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(dmc, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(prob, &x, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0c,dim*dim,&Jc,dim*dim,&invJc);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmf, &fsection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmf, &globalFSection);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmc, &csection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmc, &globalCSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmf, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscMalloc1(totDim, &elemMat);CHKERRQ(ierr);

  ierr = MatGetLocalSize(mass, &locRows, NULL);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject) mass), &rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(rLayout, locRows);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(rLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rLayout, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rLayout);CHKERRQ(ierr);
  ierr = PetscCalloc2(locRows,&dnz,locRows,&onz);CHKERRQ(ierr);
  ierr = PetscHSetIJCreate(&ht);CHKERRQ(ierr);
  for (field = 0; field < Nf; ++field) {
    PetscObject      obj;
    PetscClassId     id;
    PetscQuadrature  quad;
    const PetscReal *qpoints;
    PetscInt         Nq, Nc, i, d;

    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {ierr = PetscFEGetQuadrature((PetscFE) obj, &quad);CHKERRQ(ierr);}
    else                       {ierr = PetscFVGetQuadrature((PetscFV) obj, &quad);CHKERRQ(ierr);}
    ierr = PetscQuadratureGetData(quad, NULL, &Nc, &Nq, &qpoints, NULL);CHKERRQ(ierr);
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      Vec                pointVec;
      PetscScalar       *pV;
      PetscSF            coarseCellSF = NULL;
      const PetscSFNode *coarseCells;
      PetscInt           numCoarseCells, q, c;
      PetscInt          *findices,   *cindices;
      PetscInt           numFIndices, numCIndices;

      ierr = DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      /* Get points from the quadrature */
      ierr = VecCreateSeq(PETSC_COMM_SELF, Nq*dim, &pointVec);CHKERRQ(ierr);
      ierr = VecSetBlockSize(pointVec, dim);CHKERRQ(ierr);
      ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
      for (q = 0; q < Nq; ++q) {
        const PetscReal xi0[3] = {-1., -1., -1.};

        /* Transform point to real space */
        CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
        for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
      }
      ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
      /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
      ierr = DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF);CHKERRQ(ierr);
      ierr = PetscSFViewFromOptions(coarseCellSF, NULL, "-interp_sf_view");CHKERRQ(ierr);
      /* Update preallocation info */
      ierr = PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells);CHKERRQ(ierr);
      if (numCoarseCells != Nq) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
      {
        PetscHashIJKey key;
        PetscBool      missing;

        for (i = 0; i < numFIndices; ++i) {
          key.i = findices[i];
          if (key.i >= 0) {
            /* Get indices for coarse elements */
            for (ccell = 0; ccell < numCoarseCells; ++ccell) {
              ierr = DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL);CHKERRQ(ierr);
              for (c = 0; c < numCIndices; ++c) {
                key.j = cindices[c];
                if (key.j < 0) continue;
                ierr = PetscHSetIJQueryAdd(ht, key, &missing);CHKERRQ(ierr);
                if (missing) {
                  if ((key.j >= rStart) && (key.j < rEnd)) ++dnz[key.i-rStart];
                  else                                     ++onz[key.i-rStart];
                }
              }
              ierr = DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL);CHKERRQ(ierr);
            }
          }
        }
      }
      ierr = PetscSFDestroy(&coarseCellSF);CHKERRQ(ierr);
      ierr = VecDestroy(&pointVec);CHKERRQ(ierr);
      ierr = DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscHSetIJDestroy(&ht);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(mass, 1, dnz, onz, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetOption(mass, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);
  for (field = 0; field < Nf; ++field) {
    PetscObject       obj;
    PetscClassId      id;
    PetscTabulation T, Tfine;
    PetscQuadrature   quad;
    const PetscReal  *qpoints, *qweights;
    PetscInt          Nq, Nc, i, d;

    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      ierr = PetscFEGetQuadrature((PetscFE) obj, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetCellTabulation((PetscFE) obj, 1, &Tfine);CHKERRQ(ierr);
      ierr = PetscFECreateTabulation((PetscFE) obj, 1, 1, x, 0, &T);CHKERRQ(ierr);
    } else {
      ierr = PetscFVGetQuadrature((PetscFV) obj, &quad);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureGetData(quad, NULL, &Nc, &Nq, &qpoints, &qweights);CHKERRQ(ierr);
    /* For each fine grid cell */
    for (cell = cStart; cell < cEnd; ++cell) {
      Vec                pointVec;
      PetscScalar       *pV;
      PetscSF            coarseCellSF = NULL;
      const PetscSFNode *coarseCells;
      PetscInt           numCoarseCells, cpdim, q, c, j;
      PetscInt          *findices,   *cindices;
      PetscInt           numFIndices, numCIndices;

      ierr = DMPlexGetClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFEM(dmf, cell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      /* Get points from the quadrature */
      ierr = VecCreateSeq(PETSC_COMM_SELF, Nq*dim, &pointVec);CHKERRQ(ierr);
      ierr = VecSetBlockSize(pointVec, dim);CHKERRQ(ierr);
      ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
      for (q = 0; q < Nq; ++q) {
        const PetscReal xi0[3] = {-1., -1., -1.};

        /* Transform point to real space */
        CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], x);
        for (d = 0; d < dim; ++d) pV[q*dim+d] = x[d];
      }
      ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
      /* Get set of coarse cells that overlap points (would like to group points by coarse cell) */
      ierr = DMLocatePoints(dmc, pointVec, DM_POINTLOCATION_NEAREST, &coarseCellSF);CHKERRQ(ierr);
      /* Update matrix */
      ierr = PetscSFGetGraph(coarseCellSF, NULL, &numCoarseCells, NULL, &coarseCells);CHKERRQ(ierr);
      if (numCoarseCells != Nq) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not all closure points located");
      ierr = VecGetArray(pointVec, &pV);CHKERRQ(ierr);
      for (ccell = 0; ccell < numCoarseCells; ++ccell) {
        PetscReal pVReal[3];
        const PetscReal xi0[3] = {-1., -1., -1.};


        ierr = DMPlexGetClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL);CHKERRQ(ierr);
        /* Transform points from real space to coarse reference space */
        ierr = DMPlexComputeCellGeometryFEM(dmc, coarseCells[ccell].index, NULL, v0c, Jc, invJc, &detJc);CHKERRQ(ierr);
        for (d = 0; d < dim; ++d) pVReal[d] = PetscRealPart(pV[ccell*dim+d]);
        CoordinatesRealToRef(dim, dim, xi0, v0c, invJc, pVReal, x);

        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          /* Evaluate coarse basis on contained point */
          ierr = PetscFEGetDimension(fe, &cpdim);CHKERRQ(ierr);
          ierr = PetscFEComputeTabulation(fe, 1, x, 0, T);CHKERRQ(ierr);
          /* Get elemMat entries by multiplying by weight */
          for (i = 0; i < numFIndices; ++i) {
            ierr = PetscArrayzero(elemMat, cpdim);CHKERRQ(ierr);
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[j] += T->T[0][j*Nc + c]*Tfine->T[0][(ccell*numFIndices + i)*Nc + c]*qweights[ccell*Nc + c]*detJ;
            }
            /* Update interpolator */
            if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat);CHKERRQ(ierr);}
            if (numCIndices != cpdim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %D != %D", numCIndices, cpdim);
            ierr = MatSetValues(mass, 1, &findices[i], numCIndices, cindices, elemMat, ADD_VALUES);CHKERRQ(ierr);
          }
        } else {
          cpdim = 1;
          for (i = 0; i < numFIndices; ++i) {
            ierr = PetscArrayzero(elemMat, cpdim);CHKERRQ(ierr);
            for (j = 0; j < cpdim; ++j) {
              for (c = 0; c < Nc; ++c) elemMat[j] += 1.0*1.0*qweights[ccell*Nc + c]*detJ;
            }
            /* Update interpolator */
            if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, 1, numCIndices, elemMat);CHKERRQ(ierr);}
            ierr = PetscPrintf(PETSC_COMM_SELF, "Nq: %D %D Nf: %D %D Nc: %D %D\n", ccell, Nq, i, numFIndices, j, numCIndices);CHKERRQ(ierr);
            if (numCIndices != cpdim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of element matrix columns %D != %D", numCIndices, cpdim);
            ierr = MatSetValues(mass, 1, &findices[i], numCIndices, cindices, elemMat, ADD_VALUES);CHKERRQ(ierr);
          }
        }
        ierr = DMPlexRestoreClosureIndices(dmc, csection, globalCSection, coarseCells[ccell].index, PETSC_FALSE, &numCIndices, &cindices, NULL, NULL);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(pointVec, &pV);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&coarseCellSF);CHKERRQ(ierr);
      ierr = VecDestroy(&pointVec);CHKERRQ(ierr);
      ierr = DMPlexRestoreClosureIndices(dmf, fsection, globalFSection, cell, PETSC_FALSE, &numFIndices, &findices, NULL, NULL);CHKERRQ(ierr);
    }
    if (id == PETSCFE_CLASSID) {ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);}
  }
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  ierr = PetscFree3(v0c,Jc,invJc);CHKERRQ(ierr);
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mass, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mass, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_InjectorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(dmf, &dim);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmf, &fsection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmf, &fglobalSection);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmc, &csection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmc, &cglobalSection);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(fsection, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dmc, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetDS(dmc, &prob);CHKERRQ(ierr);
  ierr = PetscCalloc3(Nf,&feRef,Nf,&fvRef,Nf,&needAvg);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     fNb = 0, Nc = 0;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE    fe = (PetscFE) obj;
      PetscSpace sp;
      PetscInt   maxDegree;

      ierr = PetscFERefine(fe, &feRef[f]);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(feRef[f], &fNb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
      ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
      ierr = PetscSpaceGetDegree(sp, NULL, &maxDegree);CHKERRQ(ierr);
      if (!maxDegree) needAvg[f] = PETSC_TRUE;
    } else if (id == PETSCFV_CLASSID) {
      PetscFV        fv = (PetscFV) obj;
      PetscDualSpace Q;

      ierr = PetscFVRefine(fv, &fvRef[f]);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fvRef[f], &Q);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(Q, &fNb);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
      needAvg[f] = PETSC_TRUE;
    }
    fTotDim += fNb;
  }
  ierr = PetscDSGetTotalDimension(prob, &cTotDim);CHKERRQ(ierr);
  ierr = PetscMalloc1(cTotDim,&cmap);CHKERRQ(ierr);
  for (field = 0, offsetC = 0, offsetF = 0; field < Nf; ++field) {
    PetscFE        feC;
    PetscFV        fvC;
    PetscDualSpace QF, QC;
    PetscInt       order = -1, NcF, NcC, fpdim, cpdim;

    if (feRef[field]) {
      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &feC);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(feC, &NcC);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(feRef[field], &NcF);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(feRef[field], &QF);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetOrder(QF, &order);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(QF, &fpdim);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(feC, &QC);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(QC, &cpdim);CHKERRQ(ierr);
    } else {
      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fvC);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fvC, &NcC);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fvRef[field], &NcF);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fvRef[field], &QF);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(QF, &fpdim);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fvC, &QC);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(QC, &cpdim);CHKERRQ(ierr);
    }
    if (NcF != NcC) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of components in fine space field %D does not match coarse field %D", NcF, NcC);
    for (c = 0; c < cpdim; ++c) {
      PetscQuadrature  cfunc;
      const PetscReal *cqpoints, *cqweights;
      PetscInt         NqcC, NpC;
      PetscBool        found = PETSC_FALSE;

      ierr = PetscDualSpaceGetFunctional(QC, c, &cfunc);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(cfunc, NULL, &NqcC, &NpC, &cqpoints, &cqweights);CHKERRQ(ierr);
      if (NqcC != NcC) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of quadrature components %D must match number of field components %D", NqcC, NcC);
      if (NpC != 1 && feRef[field]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Do not know how to do injection for moments");
      for (f = 0; f < fpdim; ++f) {
        PetscQuadrature  ffunc;
        const PetscReal *fqpoints, *fqweights;
        PetscReal        sum = 0.0;
        PetscInt         NqcF, NpF;

        ierr = PetscDualSpaceGetFunctional(QF, f, &ffunc);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(ffunc, NULL, &NqcF, &NpF, &fqpoints, &fqweights);CHKERRQ(ierr);
        if (NqcF != NcF) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of quadrature components %D must match number of field components %D", NqcF, NcF);
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
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&feRef[f]);CHKERRQ(ierr);ierr = PetscFVDestroy(&fvRef[f]);CHKERRQ(ierr);}
  ierr = PetscFree3(feRef,fvRef,needAvg);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmf, &fv);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmc, &cv);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(cv, &startC, &endC);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(cglobalSection, &m);CHKERRQ(ierr);
  ierr = PetscMalloc2(cTotDim,&cellCIndices,fTotDim,&cellFIndices);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&cindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&findices);CHKERRQ(ierr);
  for (d = 0; d < m; ++d) cindices[d] = findices[d] = -1;
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexMatGetClosureIndicesRefined(dmf, fsection, fglobalSection, dmc, csection, cglobalSection, c, cellCIndices, cellFIndices);CHKERRQ(ierr);
    for (d = 0; d < cTotDim; ++d) {
      if ((cellCIndices[d] < startC) || (cellCIndices[d] >= endC)) continue;
      if ((findices[cellCIndices[d]-startC] >= 0) && (findices[cellCIndices[d]-startC] != cellFIndices[cmap[d]])) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse dof %D maps to both %D and %D", cindices[cellCIndices[d]-startC], findices[cellCIndices[d]-startC], cellFIndices[cmap[d]]);
      cindices[cellCIndices[d]-startC] = cellCIndices[d];
      findices[cellCIndices[d]-startC] = cellFIndices[cmap[d]];
    }
  }
  ierr = PetscFree(cmap);CHKERRQ(ierr);
  ierr = PetscFree2(cellCIndices,cellFIndices);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF, m, cindices, PETSC_OWN_POINTER, &cis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, m, findices, PETSC_OWN_POINTER, &fis);CHKERRQ(ierr);
  ierr = VecScatterCreate(cv, cis, fv, fis, sc);CHKERRQ(ierr);
  ierr = ISDestroy(&cis);CHKERRQ(ierr);
  ierr = ISDestroy(&fis);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmf, &fv);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmc, &cv);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_InjectorFEM,dmc,dmf,0,0);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 4);
  if (locX_t) {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 5);}
  if (locA)   {PetscValidHeaderSpecific(locA, VEC_CLASSID, 6);}
  PetscValidPointer(u, 7);
  PetscValidPointer(u_t, 8);
  PetscValidPointer(a, 9);
  ierr = DMPlexConvertPlex(dm, &plex, PETSC_FALSE);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  if (locA) {
    DM      dmAux;
    PetscDS probAux;

    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMPlexConvertPlex(dmAux, &plexA, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  numCells = cEnd - cStart;
  ierr = DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, u);CHKERRQ(ierr);
  if (locX_t) {ierr = DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, u_t);CHKERRQ(ierr);} else {*u_t = NULL;}
  if (locA)   {ierr = DMGetWorkArray(dm, numCells*totDimAux, MPIU_SCALAR, a);CHKERRQ(ierr);} else {*a = NULL;}
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL, *x_t = NULL, *ul = *u, *ul_t = *u_t, *al = *a;
    PetscInt       i;

    ierr = DMPlexVecGetClosure(plex, section, locX, cell, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) ul[cind*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(plex, section, locX, cell, NULL, &x);CHKERRQ(ierr);
    if (locX_t) {
      ierr = DMPlexVecGetClosure(plex, section, locX_t, cell, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) ul_t[cind*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(plex, section, locX_t, cell, NULL, &x_t);CHKERRQ(ierr);
    }
    if (locA) {
      PetscInt subcell;
      ierr = DMGetEnclosurePoint(plexA, dm, encAux, cell, &subcell);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plexA, sectionAux, locA, subcell, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) al[cind*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plexA, sectionAux, locA, subcell, NULL, &x);CHKERRQ(ierr);
    }
  }
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  if (locA) {ierr = DMDestroy(&plexA);CHKERRQ(ierr);}
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMRestoreWorkArray(dm, 0, MPIU_SCALAR, u);CHKERRQ(ierr);
  if (locX_t) {ierr = DMRestoreWorkArray(dm, 0, MPIU_SCALAR, u_t);CHKERRQ(ierr);}
  if (locA)   {ierr = DMRestoreWorkArray(dm, 0, MPIU_SCALAR, a);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetHybridAuxFields(DM dmAux, PetscDS dsAux[], IS cellIS, Vec locA, PetscScalar *a[])
{
  DM              plexA;
  PetscSection    sectionAux;
  const PetscInt *cells;
  PetscInt        cStart, cEnd, numCells, c, totDimAux[2];
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!locA) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(dmAux, DM_CLASSID, 1);
  PetscValidPointer(dsAux, 2);
  PetscValidHeaderSpecific(locA, VEC_CLASSID, 4);
  PetscValidPointer(a, 5);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMPlexConvertPlex(dmAux, &plexA, PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmAux, &sectionAux);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscDSGetTotalDimension(dsAux[0], &totDimAux[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dmAux, numCells*totDimAux[0], MPIU_SCALAR, &a[0]);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(dsAux[1], &totDimAux[1]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dmAux, numCells*totDimAux[1], MPIU_SCALAR, &a[1]);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt  cell = cells ? cells[c] : c;
    const PetscInt  cind = c - cStart;
    const PetscInt *cone, *ornt;
    PetscInt        c;

    ierr = DMPlexGetCone(dmAux, cell, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dmAux, cell, &ornt);CHKERRQ(ierr);
    for (c = 0; c < 2; ++c) {
      PetscScalar   *x = NULL, *al = a[c];
      const PetscInt tdA = totDimAux[c];
      PetscInt       Na, i;

      if (ornt[c]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_SUP, "Face %D in hybrid cell %D has orientation %D != 0", cone[c], cell, ornt[c]);
      ierr = DMPlexVecGetClosure(plexA, sectionAux, locA, cone[c], &Na, &x);CHKERRQ(ierr);
      for (i = 0; i < Na; ++i) al[cind*tdA+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plexA, sectionAux, locA, cone[c], &Na, &x);CHKERRQ(ierr);
    }
  }
  ierr = DMDestroy(&plexA);CHKERRQ(ierr);
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexRestoreHybridAuxFields(DM dmAux, PetscDS dsAux[], IS cellIS, Vec locA, PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!locA) PetscFunctionReturn(0);
  ierr = DMRestoreWorkArray(dmAux, 0, MPIU_SCALAR, &a[0]);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dmAux, 0, MPIU_SCALAR, &a[1]);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 4);
  if (locX_t) {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 5);}
  PetscValidHeaderSpecific(faceGeometry, VEC_CLASSID, 6);
  PetscValidHeaderSpecific(cellGeometry, VEC_CLASSID, 7);
  if (locGrad) {PetscValidHeaderSpecific(locGrad, VEC_CLASSID, 8);}
  PetscValidPointer(uL, 9);
  PetscValidPointer(uR, 10);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nf, &isFE);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else                            {isFE[f] = PETSC_FALSE;}
  }
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  if (locGrad) {
    ierr = VecGetDM(locGrad, &dmGrad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGrad, &lgrad);CHKERRQ(ierr);
  }
  ierr = DMGetWorkArray(dm, numFaces*Nc, MPIU_SCALAR, uL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numFaces*Nc, MPIU_SCALAR, uR);CHKERRQ(ierr);
  /* Right now just eat the extra work for FE (could make a cell loop) */
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscFVCellGeom       *cgL, *cgR;
    PetscScalar           *xL, *xR, *gL, *gR;
    PetscScalar           *uLl = *uL, *uRl = *uR;
    PetscInt               ghost, nsupp, nchild;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
    ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
    ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscInt off;

      ierr = PetscDSGetComponentOffset(prob, f, &off);CHKERRQ(ierr);
      if (isFE[f]) {
        const PetscInt *cone;
        PetscInt        comp, coneSizeL, coneSizeR, faceLocL, faceLocR, ldof, rdof, d;

        xL = xR = NULL;
        ierr = PetscSectionGetFieldComponents(section, f, &comp);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(dm, section, locX, cells[0], &ldof, (PetscScalar **) &xL);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(dm, section, locX, cells[1], &rdof, (PetscScalar **) &xR);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, cells[0], &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeSize(dm, cells[0], &coneSizeL);CHKERRQ(ierr);
        for (faceLocL = 0; faceLocL < coneSizeL; ++faceLocL) if (cone[faceLocL] == face) break;
        ierr = DMPlexGetCone(dm, cells[1], &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeSize(dm, cells[1], &coneSizeR);CHKERRQ(ierr);
        for (faceLocR = 0; faceLocR < coneSizeR; ++faceLocR) if (cone[faceLocR] == face) break;
        if (faceLocL == coneSizeL && faceLocR == coneSizeR) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %D in cone of cell %D or cell %D", face, cells[0], cells[1]);
        /* Check that FEM field has values in the right cell (sometimes its an FV ghost cell) */
        /* TODO: this is a hack that might not be right for nonconforming */
        if (faceLocL < coneSizeL) {
          ierr = PetscFEEvaluateFaceFields_Internal(prob, f, faceLocL, xL, &uLl[iface*Nc+off]);CHKERRQ(ierr);
          if (rdof == ldof && faceLocR < coneSizeR) {ierr = PetscFEEvaluateFaceFields_Internal(prob, f, faceLocR, xR, &uRl[iface*Nc+off]);CHKERRQ(ierr);}
          else              {for (d = 0; d < comp; ++d) uRl[iface*Nc+off+d] = uLl[iface*Nc+off+d];}
        }
        else {
          ierr = PetscFEEvaluateFaceFields_Internal(prob, f, faceLocR, xR, &uRl[iface*Nc+off]);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldComponents(section, f, &comp);CHKERRQ(ierr);
          for (d = 0; d < comp; ++d) uLl[iface*Nc+off+d] = uRl[iface*Nc+off+d];
        }
        ierr = DMPlexVecRestoreClosure(dm, section, locX, cells[0], &ldof, (PetscScalar **) &xL);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(dm, section, locX, cells[1], &rdof, (PetscScalar **) &xR);CHKERRQ(ierr);
      } else {
        PetscFV  fv;
        PetscInt numComp, c;

        ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fv);CHKERRQ(ierr);
        ierr = PetscFVGetNumComponents(fv, &numComp);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(dm, cells[0], f, x, &xL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(dm, cells[1], f, x, &xR);CHKERRQ(ierr);
        if (dmGrad) {
          PetscReal dxL[3], dxR[3];

          ierr = DMPlexPointLocalRead(dmGrad, cells[0], lgrad, &gL);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRead(dmGrad, cells[1], lgrad, &gR);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  if (locGrad) {
    ierr = VecRestoreArrayRead(locGrad, &lgrad);CHKERRQ(ierr);
  }
  ierr = PetscFree(isFE);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMRestoreWorkArray(dm, 0, MPIU_SCALAR, uL);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, 0, MPIU_SCALAR, uR);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(faceGeometry, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(cellGeometry, VEC_CLASSID, 5);
  PetscValidPointer(fgeom, 6);
  PetscValidPointer(vol, 7);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFaces, fgeom);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numFaces*2, MPIU_SCALAR, vol);CHKERRQ(ierr);
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscFVCellGeom       *cgL, *cgR;
    PetscFVFaceGeom       *fgeoml = *fgeom;
    PetscReal             *voll   = *vol;
    PetscInt               ghost, d, nchild, nsupp;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
    ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
    ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) {
      fgeoml[iface].centroid[d] = fg->centroid[d];
      fgeoml[iface].normal[d]   = fg->normal[d];
    }
    voll[iface*2+0] = cgL->volume;
    voll[iface*2+1] = cgR->volume;
    ++iface;
  }
  *Nface = iface;
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*fgeom);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, 0, MPIU_REAL, vol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESGetFEGeom(DMField coordField, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetId((PetscObject)quad,&id);CHKERRQ(ierr);
  ierr = PetscSNPrintf(composeStr, 32, "DMSNESGetFEGeom_%x\n", id);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) pointIS, composeStr, (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container, (void **) geom);CHKERRQ(ierr);
  } else {
    ierr = DMFieldCreateFEGeom(coordField, pointIS, quad, faceData, geom);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container, (void *) *geom);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) pointIS, composeStr, (PetscObject) container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
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
  PetscHashFormKey key;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  /* FEM+FVM */
  /* 1: Get sizes from dm and dmAux */
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  /* 2: Get geometric data */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscBool    fimp;

    ierr = PetscDSGetImplicit(prob, f, &fimp);CHKERRQ(ierr);
    if (isImplicit != fimp) continue;
    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {useFEM = PETSC_TRUE;}
    if (id == PETSCFV_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Use of FVM with PCPATCH not yet implemented");
  }
  if (useFEM) {
    ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad);CHKERRQ(ierr);
      if (affineQuad) {
        ierr = DMSNESGetFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscCalloc2(Nf,&quads,Nf,&geoms);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscBool    fimp;

        ierr = PetscDSGetImplicit(prob, f, &fimp);CHKERRQ(ierr);
        if (isImplicit != fimp) continue;
        ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          ierr = PetscFEGetQuadrature(fe, &quads[f]);CHKERRQ(ierr);
          ierr = PetscObjectReference((PetscObject)quads[f]);CHKERRQ(ierr);
          ierr = DMSNESGetFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);
        }
      }
    }
  }
  /* Loop over chunks */
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  if (useFEM) {ierr = ISCreate(PETSC_COMM_SELF, &chunkIS);CHKERRQ(ierr);}
  numCells      = cEnd - cStart;
  numChunks     = 1;
  cellChunkSize = numCells/numChunks;
  numChunks     = PetscMin(1,numCells);
  key.label     = NULL;
  key.value     = 0;
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscScalar     *elemVec, *fluxL = NULL, *fluxR = NULL;
    PetscReal       *vol = NULL;
    PetscFVFaceGeom *fgeom = NULL;
    PetscInt         cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;
    PetscInt         numFaces = 0;

    /* Extract field coefficients */
    if (useFEM) {
      ierr = ISGetPointSubrange(chunkIS, cS, cE, cells);CHKERRQ(ierr);
      ierr = DMPlexGetCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
      ierr = PetscArrayzero(elemVec, numCells*totDim);CHKERRQ(ierr);
    }
    /* TODO We will interlace both our field coefficients (u, u_t, uL, uR, etc.) and our output (elemVec, fL, fR). I think this works */
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;
      PetscBool    fimp;
      PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

      key.field = f;
      ierr = PetscDSGetImplicit(prob, f, &fimp);CHKERRQ(ierr);
      if (isImplicit != fimp) continue;
      ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        PetscFE         fe = (PetscFE) obj;
        PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
        PetscFEGeom    *chunkGeom = NULL;
        PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
        PetscInt        Nq, Nb;

        ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
        blockSize = Nb;
        batchSize = numBlocks * blockSize;
        ierr      = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
        numChunks = numCells / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numCells % (numBatches*batchSize);
        offset    = numCells - Nr;
        /* Integrate FE residual to get elemVec (need fields at quadrature points) */
        /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
        ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateResidual(prob, key, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec);CHKERRQ(ierr);
        ierr = PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateResidual(prob, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
        ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        Ne = numFaces;
        /* Riemann solve over faces (need fields at face centroids) */
        /*   We need to evaluate FE fields at those coordinates */
        ierr = PetscFVIntegrateRHSFunction(fv, prob, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR);CHKERRQ(ierr);
      } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
    }
    /* Loop over domain */
    if (useFEM) {
      /* Add elemVec to locX */
      for (c = cS; c < cE; ++c) {
        const PetscInt cell = cells ? cells[c] : c;
        const PetscInt cind = c - cStart;

        if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]);CHKERRQ(ierr);}
        if (ghostLabel) {
          PetscInt ghostVal;

          ierr = DMLabelGetValue(ghostLabel,cell,&ghostVal);CHKERRQ(ierr);
          if (ghostVal > 0) continue;
        }
        ierr = DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
      }
    }
    /* Handle time derivative */
    if (locX_t) {
      PetscScalar *x_t, *fa;

      ierr = VecGetArray(locF, &fa);CHKERRQ(ierr);
      ierr = VecGetArray(locX_t, &x_t);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     pdim, d;

        ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
        for (c = cS; c < cE; ++c) {
          const PetscInt cell = cells ? cells[c] : c;
          PetscScalar   *u_t, *r;

          if (ghostLabel) {
            PetscInt ghostVal;

            ierr = DMLabelGetValue(ghostLabel, cell, &ghostVal);CHKERRQ(ierr);
            if (ghostVal > 0) continue;
          }
          ierr = DMPlexPointLocalFieldRead(dm, cell, f, x_t, &u_t);CHKERRQ(ierr);
          ierr = DMPlexPointLocalFieldRef(dm, cell, f, fa, &r);CHKERRQ(ierr);
          for (d = 0; d < pdim; ++d) r[d] += u_t[d];
        }
      }
      ierr = VecRestoreArray(locX_t, &x_t);CHKERRQ(ierr);
      ierr = VecRestoreArray(locF, &fa);CHKERRQ(ierr);
    }
    if (useFEM) {
      ierr = DMPlexRestoreCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
    }
  }
  if (useFEM) {ierr = ISDestroy(&chunkIS);CHKERRQ(ierr);}
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  /* TODO Could include boundary residual here (see DMPlexComputeResidual_Internal) */
  if (useFEM) {
    if (maxDegree <= 1) {
      ierr = DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
    } else {
      for (f = 0; f < Nf; ++f) {
        ierr = DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);
        ierr = PetscQuadratureDestroy(&quads[f]);CHKERRQ(ierr);
      }
      ierr = PetscFree2(quads,geoms);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  We always assemble JacP, and if the matrix is different from Jac and two different sets of point functions are provided, we also assemble Jac

  X   - The local solution vector
  X_t - The local solution time derviative vector, or NULL
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
  PetscBool        hasJac, hasPrec, hasDyn, assembleJac, isMatIS, isMatISP, *isFE, hasFV = PETSC_FALSE;
  const PetscInt  *cells;
  PetscHashFormKey key;
  PetscInt         Nf, fieldI, fieldJ, maxDegree, numCells, cStart, cEnd, numChunks, chunkSize, chunk, totDim, totDimAux = 0, sz, wsz, off = 0, offCell = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(cellIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &A);CHKERRQ(ierr);
  if (A) {
    ierr = VecGetDM(A, &dmAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
  }
  /* Get flags */
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, Nf, MPIU_BOOL, &isFE);CHKERRQ(ierr);
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscObject  disc;
    PetscClassId id;
    ierr = PetscDSGetDiscretization(prob, fieldI, &disc);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(disc, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[fieldI] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {hasFV = PETSC_TRUE; isFE[fieldI] = PETSC_FALSE;}
  }
  ierr = PetscDSHasJacobian(prob, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(prob, &hasPrec);CHKERRQ(ierr);
  ierr = PetscDSHasDynamicJacobian(prob, &hasDyn);CHKERRQ(ierr);
  assembleJac = hasJac && hasPrec && (Jac != JacP) ? PETSC_TRUE : PETSC_FALSE;
  hasDyn      = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscObjectTypeCompare((PetscObject) Jac,  MATIS, &isMatIS);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  /* Setup input data and temp arrays (should be DMGetWorkArray) */
  if (isMatISP || isMatISP) {ierr = DMPlexGetSubdomainSection(dm, &globalSection);CHKERRQ(ierr);}
  if (isMatIS)  {ierr = MatISGetLocalMat(Jac,  &J);CHKERRQ(ierr);}
  if (isMatISP) {ierr = MatISGetLocalMat(JacP, &JP);CHKERRQ(ierr);}
  if (hasFV)    {ierr = MatSetOption(JP, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);} /* No allocated space for FV stuff, so ignore the zero entries */
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  if (probAux) {ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);}
  /* Compute batch sizes */
  if (isFE[0]) {
    PetscFE         fe;
    PetscQuadrature q;
    PetscInt        numQuadPoints, numBatches, batchSize, numBlocks, blockSize, Nb;

    ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q, NULL, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    blockSize = Nb*numQuadPoints;
    batchSize = numBlocks  * blockSize;
    chunkSize = numBatches * batchSize;
    numChunks = numCells / chunkSize + numCells % chunkSize;
    ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
  } else {
    chunkSize = numCells;
    numChunks = 1;
  }
  /* Get work space */
  wsz  = (((X?1:0) + (X_t?1:0) + (dmAux?1:0))*totDim + ((hasJac?1:0) + (hasPrec?1:0) + (hasDyn?1:0))*totDim*totDim)*chunkSize;
  ierr = DMGetWorkArray(dm, wsz, MPIU_SCALAR, &work);CHKERRQ(ierr);
  ierr = PetscArrayzero(work, wsz);CHKERRQ(ierr);
  off      = 0;
  u        = X       ? (sz = chunkSize*totDim,        off += sz, work+off-sz) : NULL;
  u_t      = X_t     ? (sz = chunkSize*totDim,        off += sz, work+off-sz) : NULL;
  a        = dmAux   ? (sz = chunkSize*totDimAux,     off += sz, work+off-sz) : NULL;
  elemMat  = hasJac  ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  elemMatP = hasPrec ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  elemMatD = hasDyn  ? (sz = chunkSize*totDim*totDim, off += sz, work+off-sz) : NULL;
  if (off != wsz) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error is workspace size %D should be %D", off, wsz);
  /* Setup geometry */
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree);CHKERRQ(ierr);
  if (maxDegree <= 1) {ierr = DMFieldCreateDefaultQuadrature(coordField, cellIS, &qGeom);CHKERRQ(ierr);}
  if (!qGeom) {
    PetscFE fe;

    ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &qGeom);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) qGeom);CHKERRQ(ierr);
  }
  ierr = DMSNESGetFEGeom(coordField, cellIS, qGeom, PETSC_FALSE, &cgeomFEM);CHKERRQ(ierr);
  /* Compute volume integrals */
  if (assembleJac) {ierr = MatZeroEntries(J);CHKERRQ(ierr);}
  ierr = MatZeroEntries(JP);CHKERRQ(ierr);
  key.label = NULL;
  key.value = 0;
  for (chunk = 0; chunk < numChunks; ++chunk, offCell += chunkSize) {
    const PetscInt   Ncell = PetscMin(chunkSize, numCells - offCell);
    PetscInt         c;

    /* Extract values */
    for (c = 0; c < Ncell; ++c) {
      const PetscInt cell = cells ? cells[c+offCell] : c+offCell;
      PetscScalar   *x = NULL,  *x_t = NULL;
      PetscInt       i;

      if (X) {
        ierr = DMPlexVecGetClosure(dm, section, X, cell, NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
        ierr = DMPlexVecRestoreClosure(dm, section, X, cell, NULL, &x);CHKERRQ(ierr);
      }
      if (X_t) {
        ierr = DMPlexVecGetClosure(dm, section, X_t, cell, NULL, &x_t);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u_t[c*totDim+i] = x_t[i];
        ierr = DMPlexVecRestoreClosure(dm, section, X_t, cell, NULL, &x_t);CHKERRQ(ierr);
      }
      if (dmAux) {
        ierr = DMPlexVecGetClosure(dmAux, sectionAux, A, cell, NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
        ierr = DMPlexVecRestoreClosure(dmAux, sectionAux, A, cell, NULL, &x);CHKERRQ(ierr);
      }
    }
    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscFE fe;
      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf + fieldJ;
        if (hasJac)  {ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN,     key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);}
        if (hasPrec) {ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMatP);CHKERRQ(ierr);}
        if (hasDyn)  {ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Ncell, cgeomFEM, u, u_t, probAux, a, t, X_tShift, elemMatD);CHKERRQ(ierr);}
      }
      /* For finite volume, add the identity */
      if (!isFE[fieldI]) {
        PetscFV  fv;
        PetscInt eOffset = 0, Nc, fc, foff;

        ierr = PetscDSGetFieldOffset(prob, fieldI, &foff);CHKERRQ(ierr);
        ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fv);CHKERRQ(ierr);
        ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
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
        if (hasJac)  {ierr = DMPrintCellMatrix(cell, name,  totDim, totDim, &elemMat[(c-cStart)*totDim*totDim]);CHKERRQ(ierr);}
        if (hasPrec) {ierr = DMPrintCellMatrix(cell, nameP, totDim, totDim, &elemMatP[(c-cStart)*totDim*totDim]);CHKERRQ(ierr);}
      }
      if (assembleJac) {ierr = DMPlexMatSetClosure(dm, section, globalSection, Jac, cell, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);}
      ierr = DMPlexMatSetClosure(dm, section, globalSection, JP, cell, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
    }
  }
  /* Cleanup */
  ierr = DMSNESRestoreFEGeom(coordField, cellIS, qGeom, PETSC_FALSE, &cgeomFEM);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
  if (hasFV) {ierr = MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE);CHKERRQ(ierr);}
  ierr = DMRestoreWorkArray(dm, Nf, MPIU_BOOL, &isFE);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, ((1 + (X_t?1:0) + (dmAux?1:0))*totDim + ((hasJac?1:0) + (hasPrec?1:0) + (hasDyn?1:0))*totDim*totDim)*chunkSize, MPIU_SCALAR, &work);CHKERRQ(ierr);
  /* Compute boundary integrals */
  /* ierr = DMPlexComputeBdJacobian_Internal(dm, X, X_t, t, X_tShift, Jac, JacP, ctx);CHKERRQ(ierr); */
  /* Assemble matrix */
  if (assembleJac) {ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);}
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/******** FEM Assembly Function ********/

static PetscErrorCode DMConvertPlex_Internal(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  if (isPlex) {
    *plex = dm;
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex);CHKERRQ(ierr);
    if (!*plex) {
      ierr = DMConvert(dm,DMPLEX,plex);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex);CHKERRQ(ierr);
      if (copy) {
        ierr = DMCopyAuxiliaryVec(dm, *plex);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscObjectReference((PetscObject) *plex);CHKERRQ(ierr);
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

.seealso: DMPlexTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexGetGeometryFVM(DM dm, Vec *facegeom, Vec *cellgeom, PetscReal *minRadius)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMConvertPlex_Internal(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(plex, NULL, cellgeom, facegeom, NULL);CHKERRQ(ierr);
  if (minRadius) {ierr = DMPlexGetMinRadius(plex, minRadius);CHKERRQ(ierr);}
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
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

.seealso: DMPlexSNESGetGeometryFVM()
@*/
PetscErrorCode DMPlexGetGradientDM(DM dm, PetscFV fv, DM *dmGrad)
{
  DM             plex;
  PetscBool      computeGradients;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(fv,PETSCFV_CLASSID,2);
  PetscValidPointer(dmGrad,3);
  ierr = PetscFVGetComputeGradients(fv, &computeGradients);CHKERRQ(ierr);
  if (!computeGradients) {*dmGrad = NULL; PetscFunctionReturn(0);}
  ierr = DMConvertPlex_Internal(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(plex, fv, NULL, NULL, dmGrad);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeBdResidual_Single_Internal(DM dm, PetscReal t, PetscWeakForm wf, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt field, Vec locX, Vec locX_t, Vec locF, DMField coordField, IS facetIS)
{
  DM_Plex         *mesh = (DM_Plex *) dm->data;
  DM               plex = NULL, plexA = NULL;
  DMEnclosureType  encAux;
  PetscDS          prob, probAux = NULL;
  PetscSection     section, sectionAux = NULL;
  Vec              locA = NULL;
  PetscScalar     *u = NULL, *u_t = NULL, *a = NULL, *elemVec = NULL;
  PetscInt         v;
  PetscInt         totDim, totDimAux = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &locA);CHKERRQ(ierr);
  if (locA) {
    DM dmAux;

    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plexA);CHKERRQ(ierr);
    ierr = DMGetDS(plexA, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(plexA, &sectionAux);CHKERRQ(ierr);
  }
  for (v = 0; v < numValues; ++v) {
    PetscFEGeom     *fgeom;
    PetscInt         maxDegree;
    PetscQuadrature  qGeom = NULL;
    IS               pointIS;
    const PetscInt  *points;
    PetscHashFormKey key;
    PetscInt         numFaces, face, Nq;

    key.label = label;
    key.value = values[v];
    key.field = field;
    ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a priori if one is a superset of the other */
      ierr = ISIntersect_Caching_Internal(facetIS,pointIS,&isectIS);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      pointIS = isectIS;
    }
    ierr = ISGetLocalSize(pointIS,&numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS,&points);CHKERRQ(ierr);
    ierr = PetscMalloc4(numFaces*totDim, &u, locX_t ? numFaces*totDim : 0, &u_t, numFaces*totDim, &elemVec, locA ? numFaces*totDimAux : 0, &a);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,pointIS,&qGeom);CHKERRQ(ierr);
    }
    if (!qGeom) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetFaceQuadrature(fe, &qGeom);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)qGeom);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
    ierr = DMSNESGetFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom);CHKERRQ(ierr);
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;
      PetscScalar   *x     = NULL;
      PetscInt       i;

      ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
      if (locX_t) {
        ierr = DMPlexVecGetClosure(plex, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plex, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
      }
      if (locA) {
        PetscInt subp;

        ierr = DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
      }
    }
    ierr = PetscArrayzero(elemVec, numFaces*totDim);CHKERRQ(ierr);
    {
      PetscFE         fe;
      PetscInt        Nb;
      PetscFEGeom     *chunkGeom = NULL;
      /* Conforming batches */
      PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscInt        Nr, offset;

      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      /* TODO: documentation is unclear about what is going on with these numbers: how should Nb / Nq factor in ? */
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      ierr = PetscFEGeomGetChunk(fgeom,0,offset,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrateBdResidual(prob, wf, key, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(fgeom, 0, offset, &chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEGeomGetChunk(fgeom,offset,numFaces,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrateBdResidual(prob, wf, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, &elemVec[offset*totDim]);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(fgeom,offset,numFaces,&chunkGeom);CHKERRQ(ierr);
    }
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;

      if (mesh->printFEM > 1) {ierr = DMPrintCellVector(point, "BdResidual", totDim, &elemVec[face*totDim]);CHKERRQ(ierr);}
      ierr = DMPlexGetSupport(plex, point, &support);CHKERRQ(ierr);
      ierr = DMPlexVecSetClosure(plex, NULL, locF, support[0], &elemVec[face*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
    }
    ierr = DMSNESRestoreFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = PetscFree4(u, u_t, elemVec, a);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = DMDestroy(&plexA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdResidualSingle(DM dm, PetscReal t, PetscWeakForm wf, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt field, Vec locX, Vec locX_t, Vec locF)
{
  DMField        coordField;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, dim-1, &facetIS);CHKERRQ(ierr);
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMPlexComputeBdResidual_Single_Internal(dm, t, wf, label, numValues, values, field, locX, locX_t, locF, coordField, facetIS);CHKERRQ(ierr);
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel,dim - 1,&facetIS);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
  for (bd = 0; bd < numBd; ++bd) {
    PetscWeakForm           wf;
    DMBoundaryConditionType type;
    DMLabel                 label;
    const PetscInt         *values;
    PetscInt                field, numValues;
    PetscObject             obj;
    PetscClassId            id;

    ierr = PetscDSGetBoundary(prob, bd, &wf, &type, NULL, &label, &numValues, &values, &field, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    if (!facetIS) {
      DMLabel  depthLabel;
      PetscInt dim;

      ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
      ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(depthLabel, dim - 1, &facetIS);CHKERRQ(ierr);
    }
    ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
    ierr = DMPlexComputeBdResidual_Single_Internal(dm, t, wf, label, numValues, values, field, locX, locX_t, locF, coordField, facetIS);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeResidual_Internal(DM dm, PetscHashFormKey key, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* TODO The FVM geometry is over-manipulated. Make the precalc functions return exactly what we need */
  /* FEM+FVM */
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  /* 1: Get sizes from dm and dmAux */
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &locA);CHKERRQ(ierr);
  if (locA) {
    PetscInt subcell;
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosurePoint(dmAux, dm, DM_ENC_UNKNOWN, cStart, &subcell);CHKERRQ(ierr);
    ierr = DMGetCellDS(dmAux, subcell, &dsAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
  }
  /* 2: Get geometric data */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscBool    fimp;

    ierr = PetscDSGetImplicit(ds, f, &fimp);CHKERRQ(ierr);
    if (isImplicit != fimp) continue;
    ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {useFEM = PETSC_TRUE;}
    if (id == PETSCFV_CLASSID) {useFVM = PETSC_TRUE; fvm = (PetscFV) obj;}
  }
  if (useFEM) {
    ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad);CHKERRQ(ierr);
      if (affineQuad) {
        ierr = DMSNESGetFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscCalloc2(Nf,&quads,Nf,&geoms);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscBool    fimp;

        ierr = PetscDSGetImplicit(ds, f, &fimp);CHKERRQ(ierr);
        if (isImplicit != fimp) continue;
        ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          ierr = PetscFEGetQuadrature(fe, &quads[f]);CHKERRQ(ierr);
          ierr = PetscObjectReference((PetscObject)quads[f]);CHKERRQ(ierr);
          ierr = DMSNESGetFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);
        }
      }
    }
  }
  if (useFVM) {
    ierr = DMPlexGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);
    /* Reconstruct and limit cell gradients */
    ierr = DMPlexGetGradientDM(dm, fvm, &dmGrad);CHKERRQ(ierr);
    if (dmGrad) {
      ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dmGrad, &grad);CHKERRQ(ierr);
      ierr = DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad);CHKERRQ(ierr);
      /* Communicate gradient values */
      ierr = DMGetLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dmGrad, &grad);CHKERRQ(ierr);
    }
    /* Handle non-essential (e.g. outflow) boundary values */
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, time, faceGeometryFVM, cellGeometryFVM, locGrad);CHKERRQ(ierr);
  }
  /* Loop over chunks */
  if (useFEM) {ierr = ISCreate(PETSC_COMM_SELF, &chunkIS);CHKERRQ(ierr);}
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
      ierr = ISGetPointSubrange(chunkIS, cS, cE, cells);CHKERRQ(ierr);
      ierr = DMPlexGetCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
      ierr = PetscArrayzero(elemVec, numCells*totDim);CHKERRQ(ierr);
    }
    if (useFVM) {
      ierr = DMPlexGetFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR);CHKERRQ(ierr);
      ierr = DMPlexGetFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);
      ierr = PetscArrayzero(fluxL, numFaces*totDim);CHKERRQ(ierr);
      ierr = PetscArrayzero(fluxR, numFaces*totDim);CHKERRQ(ierr);
    }
    /* TODO We will interlace both our field coefficients (u, u_t, uL, uR, etc.) and our output (elemVec, fL, fR). I think this works */
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;
      PetscBool    fimp;
      PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

      key.field = f;
      ierr = PetscDSGetImplicit(ds, f, &fimp);CHKERRQ(ierr);
      if (isImplicit != fimp) continue;
      ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        PetscFE         fe = (PetscFE) obj;
        PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
        PetscFEGeom    *chunkGeom = NULL;
        PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
        PetscInt        Nq, Nb;

        ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
        blockSize = Nb;
        batchSize = numBlocks * blockSize;
        ierr      = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
        numChunks = numCells / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numCells % (numBatches*batchSize);
        offset    = numCells - Nr;
        /* Integrate FE residual to get elemVec (need fields at quadrature points) */
        /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
        ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, u_t, dsAux, a, t, elemVec);CHKERRQ(ierr);
        ierr = PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateResidual(ds, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
        ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        Ne = numFaces;
        /* Riemann solve over faces (need fields at face centroids) */
        /*   We need to evaluate FE fields at those coordinates */
        ierr = PetscFVIntegrateRHSFunction(fv, ds, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR);CHKERRQ(ierr);
      } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
    }
    /* Loop over domain */
    if (useFEM) {
      /* Add elemVec to locX */
      for (c = cS; c < cE; ++c) {
        const PetscInt cell = cells ? cells[c] : c;
        const PetscInt cind = c - cStart;

        if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]);CHKERRQ(ierr);}
        if (ghostLabel) {
          PetscInt ghostVal;

          ierr = DMLabelGetValue(ghostLabel,cell,&ghostVal);CHKERRQ(ierr);
          if (ghostVal > 0) continue;
        }
        ierr = DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
      }
    }
    if (useFVM) {
      PetscScalar *fa;
      PetscInt     iface;

      ierr = VecGetArray(locF, &fa);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     foff, pdim;

        ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
        ierr = PetscDSGetFieldOffset(ds, f, &foff);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
        /* Accumulate fluxes to cells */
        for (face = fS, iface = 0; face < fE; ++face) {
          const PetscInt *scells;
          PetscScalar    *fL = NULL, *fR = NULL;
          PetscInt        ghost, d, nsupp, nchild;

          ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
          ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
          ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
          if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
          ierr = DMPlexGetSupport(dm, face, &scells);CHKERRQ(ierr);
          ierr = DMLabelGetValue(ghostLabel,scells[0],&ghost);CHKERRQ(ierr);
          if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, scells[0], f, fa, &fL);CHKERRQ(ierr);}
          ierr = DMLabelGetValue(ghostLabel,scells[1],&ghost);CHKERRQ(ierr);
          if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, scells[1], f, fa, &fR);CHKERRQ(ierr);}
          for (d = 0; d < pdim; ++d) {
            if (fL) fL[d] -= fluxL[iface*totDim+foff+d];
            if (fR) fR[d] += fluxR[iface*totDim+foff+d];
          }
          ++iface;
        }
      }
      ierr = VecRestoreArray(locF, &fa);CHKERRQ(ierr);
    }
    /* Handle time derivative */
    if (locX_t) {
      PetscScalar *x_t, *fa;

      ierr = VecGetArray(locF, &fa);CHKERRQ(ierr);
      ierr = VecGetArray(locX_t, &x_t);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     pdim, d;

        ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
        for (c = cS; c < cE; ++c) {
          const PetscInt cell = cells ? cells[c] : c;
          PetscScalar   *u_t, *r;

          if (ghostLabel) {
            PetscInt ghostVal;

            ierr = DMLabelGetValue(ghostLabel, cell, &ghostVal);CHKERRQ(ierr);
            if (ghostVal > 0) continue;
          }
          ierr = DMPlexPointLocalFieldRead(dm, cell, f, x_t, &u_t);CHKERRQ(ierr);
          ierr = DMPlexPointLocalFieldRef(dm, cell, f, fa, &r);CHKERRQ(ierr);
          for (d = 0; d < pdim; ++d) r[d] += u_t[d];
        }
      }
      ierr = VecRestoreArray(locX_t, &x_t);CHKERRQ(ierr);
      ierr = VecRestoreArray(locF, &fa);CHKERRQ(ierr);
    }
    if (useFEM) {
      ierr = DMPlexRestoreCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
    }
    if (useFVM) {
      ierr = DMPlexRestoreFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR);CHKERRQ(ierr);
      ierr = DMPlexRestoreFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);
      if (dmGrad) {ierr = DMRestoreLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);}
    }
  }
  if (useFEM) {ierr = ISDestroy(&chunkIS);CHKERRQ(ierr);}
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);

  if (useFEM) {
    ierr = DMPlexComputeBdResidual_Internal(dm, locX, locX_t, t, locF, user);CHKERRQ(ierr);

    if (maxDegree <= 1) {
      ierr = DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
    } else {
      for (f = 0; f < Nf; ++f) {
        ierr = DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);
        ierr = PetscQuadratureDestroy(&quads[f]);CHKERRQ(ierr);
      }
      ierr = PetscFree2(quads,geoms);CHKERRQ(ierr);
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

    ierr = VecDuplicate(locF,&locFbc);CHKERRQ(ierr);
    ierr = VecCopy(locF,locFbc);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(section,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetMaxDof(section,&maxDof);CHKERRQ(ierr);
    ierr = PetscCalloc1(maxDof,&zeroes);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      ierr = VecSetValuesSection(locFbc,section,p,zeroes,INSERT_BC_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(zeroes);CHKERRQ(ierr);
    ierr = DMPrintLocalVec(dm, name, mesh->printTol, locFbc);CHKERRQ(ierr);
    ierr = VecDestroy(&locFbc);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  1) Allow multiple kernels for BdResidual for hybrid DS

  DONE 2) Get out dsAux for either side at the same time as cohesive cell dsAux

  DONE 3) Change DMGetCellFields() to get different aux data a[] for each side
     - I think I just need to replace a[] with the closure from each face

  4) Run both kernels for each non-hybrid field with correct dsAux, and then hybrid field as before
*/
PetscErrorCode DMPlexComputeResidual_Hybrid_Internal(DM dm, PetscHashFormKey key[], IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  DM_Plex         *mesh       = (DM_Plex *) dm->data;
  const char      *name       = "Hybrid Residual";
  DM               dmAux      = NULL;
  DMLabel          ghostLabel = NULL;
  PetscDS          ds         = NULL;
  PetscDS          dsAux[3]   = {NULL, NULL, NULL};
  PetscSection     section    = NULL;
  DMField          coordField = NULL;
  Vec              locA;
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* FEM */
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  /* 1: Get sizes from dm and dmAux */
  ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cStart, &ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetCellDS(dmAux, cStart, &dsAux[2]);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux[2], &totDimAux[2]);CHKERRQ(ierr);
    {
      const PetscInt *cone;
      PetscInt        c;

      ierr = DMPlexGetCone(dm, cStart, &cone);CHKERRQ(ierr);
      for (c = 0; c < 2; ++c) {
        const PetscInt *support;
        PetscInt ssize, s;

        ierr = DMPlexGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, cone[c], &ssize);CHKERRQ(ierr);
        if (ssize != 2) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D from cell %D has support size %D != 2", cone[c], cStart, ssize);
        if      (support[0] == cStart) s = 1;
        else if (support[1] == cStart) s = 0;
        else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D does not have cell %D in its support", cone[c], cStart);
        ierr = DMGetCellDS(dmAux, support[s], &dsAux[c]);CHKERRQ(ierr);
        ierr = PetscDSGetTotalDimension(dsAux[c], &totDimAux[c]);CHKERRQ(ierr);
      }
    }
  }
  /* 2: Setup geometric data */
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree);CHKERRQ(ierr);
  if (maxDegree > 1) {
    ierr = PetscCalloc2(Nf, &quads, Nf, &geoms);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(ds, f, (PetscObject *) &fe);CHKERRQ(ierr);
      if (fe) {
        ierr = PetscFEGetQuadrature(fe, &quads[f]);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject) quads[f]);CHKERRQ(ierr);
      }
    }
  }
  /* Loop over chunks */
  numCells      = cEnd - cStart;
  cellChunkSize = numCells;
  numChunks     = !numCells ? 0 : PetscCeilReal(((PetscReal) numCells)/cellChunkSize);
  ierr = PetscCalloc1(2*cellChunkSize, &faces);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, cellChunkSize, faces, PETSC_USE_POINTER, &chunkIS);CHKERRQ(ierr);
  /* Extract field coefficients */
  /* NOTE This needs the end cap faces to have identical orientations */
  ierr = DMPlexGetCellFields(dm, cellIS, locX, locX_t, locA, &u, &u_t, &a[2]);CHKERRQ(ierr);
  ierr = DMPlexGetHybridAuxFields(dm, dsAux, cellIS, locA, a);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, cellChunkSize*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscInt cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;

    ierr = PetscMemzero(elemVec, cellChunkSize*totDim * sizeof(PetscScalar));CHKERRQ(ierr);
    /* Get faces */
    for (c = cS; c < cE; ++c) {
      const PetscInt  cell = cells ? cells[c] : c;
      const PetscInt *cone;
      ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
      faces[(c-cS)*2+0] = cone[0];
      faces[(c-cS)*2+1] = cone[1];
    }
    ierr = ISGeneralSetIndices(chunkIS, cellChunkSize, faces, PETSC_USE_POINTER);CHKERRQ(ierr);
    /* Get geometric data */
    if (maxDegree <= 1) {
      if (!affineQuad) {ierr = DMFieldCreateDefaultQuadrature(coordField, chunkIS, &affineQuad);CHKERRQ(ierr);}
      if (affineQuad)  {ierr = DMSNESGetFEGeom(coordField, chunkIS, affineQuad, PETSC_TRUE, &affineGeom);CHKERRQ(ierr);}
    } else {
      for (f = 0; f < Nf; ++f) {
        if (quads[f]) {ierr = DMSNESGetFEGeom(coordField, chunkIS, quads[f], PETSC_TRUE, &geoms[f]);CHKERRQ(ierr);}
      }
    }
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscFE         fe;
      PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
      PetscFEGeom    *chunkGeom = NULL;
      PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
      PetscInt        numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset, Nq, Nb;

      ierr = PetscDSGetDiscretization(ds, f, (PetscObject *) &fe);CHKERRQ(ierr);
      if (!fe) continue;
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      ierr      = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      if (f == Nf-1) {
        key[2].field = f;
        ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateHybridResidual(ds, key[2], Ne, chunkGeom, u, u_t, dsAux[2], a[2], t, elemVec);CHKERRQ(ierr);
        ierr = PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateHybridResidual(ds, key[2], Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[2], &a[2][offset*totDimAux[2]], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
        ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
      } else {
        key[0].field = f;
        key[1].field = f;
        ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateHybridResidual(ds, key[0], Ne, chunkGeom, u, u_t, dsAux[0], a[0], t, elemVec);CHKERRQ(ierr);
        ierr = PetscFEIntegrateHybridResidual(ds, key[1], Ne, chunkGeom, u, u_t, dsAux[1], a[1], t, elemVec);CHKERRQ(ierr);
        ierr = PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateHybridResidual(ds, key[0], Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[0], &a[0][offset*totDimAux[0]], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
        ierr = PetscFEIntegrateHybridResidual(ds, key[1], Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux[1], &a[1][offset*totDimAux[1]], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
        ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
      }
    }
    /* Add elemVec to locX */
    for (c = cS; c < cE; ++c) {
      const PetscInt cell = cells ? cells[c] : c;
      const PetscInt cind = c - cStart;

      if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]);CHKERRQ(ierr);}
      if (ghostLabel) {
        PetscInt ghostVal;

        ierr = DMLabelGetValue(ghostLabel,cell,&ghostVal);CHKERRQ(ierr);
        if (ghostVal > 0) continue;
      }
      ierr = DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexRestoreCellFields(dm, cellIS, locX, locX_t, locA, &u, &u_t, &a[2]);CHKERRQ(ierr);
  ierr = DMPlexRestoreHybridAuxFields(dm, dsAux, cellIS, locA, a);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
  ierr = PetscFree(faces);CHKERRQ(ierr);
  ierr = ISDestroy(&chunkIS);CHKERRQ(ierr);
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  if (maxDegree <= 1) {
    ierr = DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
  } else {
    for (f = 0; f < Nf; ++f) {
      if (geoms) {ierr = DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);}
      if (quads) {ierr = PetscQuadratureDestroy(&quads[f]);CHKERRQ(ierr);}
    }
    ierr = PetscFree2(quads,geoms);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobian_Single_Internal(DM dm, PetscReal t, PetscWeakForm wf, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt fieldI, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, DMField coordField, IS facetIS)
{
  DM_Plex        *mesh = (DM_Plex *) dm->data;
  DM              plex = NULL, plexA = NULL, tdm;
  DMEnclosureType encAux;
  PetscDS         prob, probAux = NULL;
  PetscSection    section, sectionAux = NULL;
  PetscSection    globalSection, subSection = NULL;
  Vec             locA = NULL, tv;
  PetscScalar    *u = NULL, *u_t = NULL, *a = NULL, *elemMat = NULL;
  PetscInt        v;
  PetscInt        Nf, totDim, totDimAux = 0;
  PetscBool       isMatISP, transform;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &locA);CHKERRQ(ierr);
  if (locA) {
    DM dmAux;

    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plexA);CHKERRQ(ierr);
    ierr = DMGetDS(plexA, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(plexA, &sectionAux);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  if (isMatISP) {ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);}
  for (v = 0; v < numValues; ++v) {
    PetscFEGeom     *fgeom;
    PetscInt         maxDegree;
    PetscQuadrature  qGeom = NULL;
    IS               pointIS;
    const PetscInt  *points;
    PetscHashFormKey key;
    PetscInt         numFaces, face, Nq;

    key.label = label;
    key.value = values[v];
    ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a prior if one is a superset of the other */
      ierr = ISIntersect_Caching_Internal(facetIS,pointIS,&isectIS);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      pointIS = isectIS;
    }
    ierr = ISGetLocalSize(pointIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = PetscMalloc4(numFaces*totDim, &u, locX_t ? numFaces*totDim : 0, &u_t, numFaces*totDim*totDim, &elemMat, locA ? numFaces*totDimAux : 0, &a);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,pointIS,&qGeom);CHKERRQ(ierr);
    }
    if (!qGeom) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetFaceQuadrature(fe, &qGeom);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)qGeom);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
    ierr = DMSNESGetFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom);CHKERRQ(ierr);
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;
      PetscScalar   *x     = NULL;
      PetscInt       i;

      ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
      if (locX_t) {
        ierr = DMPlexVecGetClosure(plex, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plex, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
      }
      if (locA) {
        PetscInt subp;
        ierr = DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
      }
    }
    ierr = PetscArrayzero(elemMat, numFaces*totDim*totDim);CHKERRQ(ierr);
    {
      PetscFE         fe;
      PetscInt        Nb;
      /* Conforming batches */
      PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscFEGeom    *chunkGeom = NULL;
      PetscInt        fieldJ, Nr, offset;

      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      ierr = PetscFEGeomGetChunk(fgeom,0,offset,&chunkGeom);CHKERRQ(ierr);
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf+fieldJ;
        ierr = PetscFEIntegrateBdJacobian(prob, wf, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
      }
      ierr = PetscFEGeomGetChunk(fgeom,offset,numFaces,&chunkGeom);CHKERRQ(ierr);
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        key.field = fieldI*Nf+fieldJ;
        ierr = PetscFEIntegrateBdJacobian(prob, wf, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
      }
      ierr = PetscFEGeomRestoreChunk(fgeom,offset,numFaces,&chunkGeom);CHKERRQ(ierr);
    }
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;

      /* Transform to global basis before insertion in Jacobian */
      ierr = DMPlexGetSupport(plex, point, &support);CHKERRQ(ierr);
      if (transform) {ierr = DMPlexBasisTransformPointTensor_Internal(dm, tdm, tv, support[0], PETSC_TRUE, totDim, &elemMat[face*totDim*totDim]);CHKERRQ(ierr);}
      if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(point, "BdJacobian", totDim, totDim, &elemMat[face*totDim*totDim]);CHKERRQ(ierr);}
      if (!isMatISP) {
        ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, support[0], &elemMat[face*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      } else {
        Mat lJ;

        ierr = MatISGetLocalMat(JacP, &lJ);CHKERRQ(ierr);
        ierr = DMPlexMatSetClosure(plex, section, subSection, lJ, support[0], &elemMat[face*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = DMSNESRestoreFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = PetscFree4(u, u_t, elemMat, a);CHKERRQ(ierr);
  }
  if (plex)  {ierr = DMDestroy(&plex);CHKERRQ(ierr);}
  if (plexA) {ierr = DMDestroy(&plexA);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobianSingle(DM dm, PetscReal t, PetscWeakForm wf, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt field, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP)
{
  DMField        coordField;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, dim-1, &facetIS);CHKERRQ(ierr);
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMPlexComputeBdJacobian_Single_Internal(dm, t, wf, label, numValues, values, field, locX, locX_t, X_tShift, Jac, JacP, coordField, facetIS);CHKERRQ(ierr);
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobian_Internal(DM dm, Vec locX, Vec locX_t, PetscReal t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  PetscDS          prob;
  PetscInt         dim, numBd, bd;
  DMLabel          depthLabel;
  DMField          coordField = NULL;
  IS               facetIS;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, dim-1, &facetIS);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  for (bd = 0; bd < numBd; ++bd) {
    PetscWeakForm           wf;
    DMBoundaryConditionType type;
    DMLabel                 label;
    const PetscInt         *values;
    PetscInt                fieldI, numValues;
    PetscObject             obj;
    PetscClassId            id;

    ierr = PetscDSGetBoundary(prob, bd, &wf, &type, NULL, &label, &numValues, &values, &fieldI, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(prob, fieldI, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    ierr = DMPlexComputeBdJacobian_Single_Internal(dm, t, wf, label, numValues, values, fieldI, locX, locX_t, X_tShift, Jac, JacP, coordField, facetIS);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeJacobian_Internal(DM dm, PetscHashFormKey key, IS cellIS, PetscReal t, PetscReal X_tShift, Vec X, Vec X_t, Mat Jac, Mat JacP,void *user)
{
  DM_Plex        *mesh  = (DM_Plex *) dm->data;
  const char     *name  = "Jacobian";
  DM              dmAux = NULL, plex, tdm;
  DMEnclosureType encAux;
  Vec             A, tv;
  DMField         coordField;
  PetscDS         prob, probAux = NULL;
  PetscSection    section, globalSection, subSection, sectionAux;
  PetscScalar    *elemMat, *elemMatP, *elemMatD, *u, *u_t, *a = NULL;
  const PetscInt *cells;
  PetscInt        Nf, fieldI, fieldJ;
  PetscInt        totDim, totDimAux, cStart, cEnd, numCells, c;
  PetscBool       isMatIS, isMatISP, hasJac, hasPrec, hasDyn, hasFV = PETSC_FALSE, transform;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cellIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  if (isMatISP) {ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);}
  ierr = ISGetLocalSize(cellIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(prob, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(prob, &hasPrec);CHKERRQ(ierr);
  /* user passed in the same matrix, avoid double contributions and
     only assemble the Jacobian */
  if (hasJac && Jac == JacP) hasPrec = PETSC_FALSE;
  ierr = PetscDSHasDynamicJacobian(prob, &hasDyn);CHKERRQ(ierr);
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &A);CHKERRQ(ierr);
  if (A) {
    ierr = VecGetDM(A, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetLocalSection(plex, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = PetscMalloc5(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,hasJac ? numCells*totDim*totDim : 0,&elemMat,hasPrec ? numCells*totDim*totDim : 0, &elemMatP,hasDyn ? numCells*totDim*totDim : 0, &elemMatD);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL,  *x_t = NULL;
    PetscInt       i;

    ierr = DMPlexVecGetClosure(dm, section, X, cell, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[cind*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, cell, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, cell, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[cind*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, cell, NULL, &x_t);CHKERRQ(ierr);
    }
    if (dmAux) {
      PetscInt subcell;
      ierr = DMGetEnclosurePoint(dmAux, dm, encAux, cell, &subcell);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plex, sectionAux, A, subcell, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[cind*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plex, sectionAux, A, subcell, NULL, &x);CHKERRQ(ierr);
    }
  }
  if (hasJac)  {ierr = PetscArrayzero(elemMat,  numCells*totDim*totDim);CHKERRQ(ierr);}
  if (hasPrec) {ierr = PetscArrayzero(elemMatP, numCells*totDim*totDim);CHKERRQ(ierr);}
  if (hasDyn)  {ierr = PetscArrayzero(elemMatD, numCells*totDim*totDim);CHKERRQ(ierr);}
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

    ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId((PetscObject) fe, &id);CHKERRQ(ierr);
    if (id == PETSCFV_CLASSID) {hasFV = PETSC_TRUE; continue;}
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,cellIS,&qGeom);CHKERRQ(ierr);
    }
    if (!qGeom) {
      ierr = PetscFEGetQuadrature(fe,&qGeom);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)qGeom);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
    ierr = DMSNESGetFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM);CHKERRQ(ierr);
    blockSize = Nb;
    batchSize = numBlocks * blockSize;
    ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    ierr = PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom);CHKERRQ(ierr);
    ierr = PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&remGeom);CHKERRQ(ierr);
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      key.field = fieldI*Nf+fieldJ;
      if (hasJac) {
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
      }
      if (hasPrec) {
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatP);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatP[offset*totDim*totDim]);CHKERRQ(ierr);
      }
      if (hasDyn) {
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatD);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&remGeom);CHKERRQ(ierr);
    ierr = PetscFEGeomRestoreChunk(cgeomFEM,0,offset,&chunkGeom);CHKERRQ(ierr);
    ierr = DMSNESRestoreFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
  }
  /*   Add contribution from X_t */
  if (hasDyn) {for (c = 0; c < numCells*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];}
  if (hasFV) {
    PetscClassId id;
    PetscFV      fv;
    PetscInt     offsetI, NcI, NbI = 1, fc, f;

    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fv);CHKERRQ(ierr);
      ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId((PetscObject) fv, &id);CHKERRQ(ierr);
      if (id != PETSCFV_CLASSID) continue;
      /* Put in the identity */
      ierr = PetscFVGetNumComponents(fv, &NcI);CHKERRQ(ierr);
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
    ierr = MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
  }
  /* Insert values into matrix */
  isMatIS = PETSC_FALSE;
  if (hasPrec && hasJac) {
    ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatIS);CHKERRQ(ierr);
  }
  if (isMatIS && !subSection) {
    ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;

    /* Transform to global basis before insertion in Jacobian */
    if (transform) {ierr = DMPlexBasisTransformPointTensor_Internal(dm, tdm, tv, cell, PETSC_TRUE, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
    if (hasPrec) {
      if (hasJac) {
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatIS) {
          ierr = DMPlexMatSetClosure(dm, section, globalSection, Jac, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(Jac,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      }
      if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMatP[cind*totDim*totDim]);CHKERRQ(ierr);}
      if (!isMatISP) {
        ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      } else {
        Mat lJ;

        ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
        ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      }
    } else {
      if (hasJac) {
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatISP) {
          ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  if (hasFV) {ierr = MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE);CHKERRQ(ierr);}
  ierr = PetscFree5(u,u_t,elemMat,elemMatP,elemMatD);CHKERRQ(ierr);
  if (dmAux) {
    ierr = PetscFree(a);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  /* Compute boundary integrals */
  ierr = DMPlexComputeBdJacobian_Internal(dm, X, X_t, t, X_tShift, Jac, JacP, user);CHKERRQ(ierr);
  /* Assemble matrix */
  if (hasJac && hasPrec) {
    ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeJacobian_Hybrid_Internal(DM dm, IS cellIS, PetscReal t, PetscReal X_tShift, Vec locX, Vec locX_t, Mat Jac, Mat JacP, void *user)
{
  DM_Plex         *mesh       = (DM_Plex *) dm->data;
  const char      *name       = "Hybrid Jacobian";
  DM               dmAux      = NULL;
  DM               plex       = NULL;
  DM               plexA      = NULL;
  DMLabel          ghostLabel = NULL;
  PetscDS          prob       = NULL;
  PetscDS          probAux    = NULL;
  PetscSection     section    = NULL;
  DMField          coordField = NULL;
  Vec              locA;
  PetscScalar     *u = NULL, *u_t, *a = NULL;
  PetscScalar     *elemMat, *elemMatP;
  PetscSection     globalSection, subSection, sectionAux;
  IS               chunkIS;
  const PetscInt  *cells;
  PetscInt        *faces;
  PetscHashFormKey key;
  PetscInt         cStart, cEnd, numCells;
  PetscInt         Nf, fieldI, fieldJ, totDim, totDimAux, numChunks, cellChunkSize, chunk;
  PetscInt         maxDegree = PETSC_MAX_INT;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;
  PetscBool        isMatIS = PETSC_FALSE, isMatISP = PETSC_FALSE, hasBdJac, hasBdPrec;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cellIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cStart, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSHasBdJacobian(prob, &hasBdJac);CHKERRQ(ierr);
  ierr = PetscDSHasBdJacobianPreconditioner(prob, &hasBdPrec);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  if (isMatISP) {ierr = DMPlexGetSubdomainSection(plex, &subSection);CHKERRQ(ierr);}
  if (hasBdPrec && hasBdJac) {ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatIS);CHKERRQ(ierr);}
  if (isMatIS && !subSection) {ierr = DMPlexGetSubdomainSection(plex, &subSection);CHKERRQ(ierr);}
  ierr = DMGetAuxiliaryVec(dm, NULL, 0, &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plexA);CHKERRQ(ierr);
    ierr = DMGetSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetCellDS(dmAux, cStart, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree);CHKERRQ(ierr);
  if (maxDegree > 1) {
    PetscInt f;
    ierr = PetscCalloc2(Nf,&quads,Nf,&geoms);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
      if (fe) {
        ierr = PetscFEGetQuadrature(fe, &quads[f]);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject) quads[f]);CHKERRQ(ierr);
      }
    }
  }
  cellChunkSize = numCells;
  numChunks     = !numCells ? 0 : PetscCeilReal(((PetscReal) numCells)/cellChunkSize);
  ierr = PetscCalloc1(2*cellChunkSize, &faces);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, cellChunkSize, faces, PETSC_USE_POINTER, &chunkIS);CHKERRQ(ierr);
  ierr = DMPlexGetCellFields(dm, cellIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, hasBdJac  ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMat);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, hasBdPrec ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMatP);CHKERRQ(ierr);
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscInt cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;

    if (hasBdJac)  {ierr = PetscMemzero(elemMat,  numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);}
    if (hasBdPrec) {ierr = PetscMemzero(elemMatP, numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);}
    /* Get faces */
    for (c = cS; c < cE; ++c) {
      const PetscInt  cell = cells ? cells[c] : c;
      const PetscInt *cone;
      ierr = DMPlexGetCone(plex, cell, &cone);CHKERRQ(ierr);
      faces[(c-cS)*2+0] = cone[0];
      faces[(c-cS)*2+1] = cone[1];
    }
    ierr = ISGeneralSetIndices(chunkIS, cellChunkSize, faces, PETSC_USE_POINTER);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      if (!affineQuad) {ierr = DMFieldCreateDefaultQuadrature(coordField, chunkIS, &affineQuad);CHKERRQ(ierr);}
      if (affineQuad)  {ierr = DMSNESGetFEGeom(coordField, chunkIS, affineQuad, PETSC_TRUE, &affineGeom);CHKERRQ(ierr);}
    } else {
      PetscInt f;
      for (f = 0; f < Nf; ++f) {
        if (quads[f]) {ierr = DMSNESGetFEGeom(coordField, chunkIS, quads[f], PETSC_TRUE, &geoms[f]);CHKERRQ(ierr);}
      }
    }

    key.label = NULL;
    key.value = 0;
    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscFE         feI;
      PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[fieldI];
      PetscFEGeom    *chunkGeom = NULL, *remGeom = NULL;
      PetscQuadrature quad = affineQuad ? affineQuad : quads[fieldI];
      PetscInt        numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset, Nq, Nb;

      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &feI);CHKERRQ(ierr);
      if (!feI) continue;
      ierr = PetscFEGetTileSizes(feI, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(feI, &Nb);CHKERRQ(ierr);
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      ierr      = PetscFESetTileSizes(feI, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEGeomGetChunk(geom,offset,numCells,&remGeom);CHKERRQ(ierr);
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        PetscFE feJ;

        ierr = PetscDSGetDiscretization(prob, fieldJ, (PetscObject *) &feJ);CHKERRQ(ierr);
        if (!feJ) continue;
        key.field = fieldI*Nf+fieldJ;
        if (hasBdJac) {
          ierr = PetscFEIntegrateHybridJacobian(prob, PETSCFE_JACOBIAN, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
          ierr = PetscFEIntegrateHybridJacobian(prob, PETSCFE_JACOBIAN, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
        }
        if (hasBdPrec) {
          ierr = PetscFEIntegrateHybridJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatP);CHKERRQ(ierr);
          ierr = PetscFEIntegrateHybridJacobian(prob, PETSCFE_JACOBIAN_PRE, key, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatP[offset*totDim*totDim]);CHKERRQ(ierr);
        }
      }
      ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&remGeom);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
    }
    /* Insert values into matrix */
    for (c = cS; c < cE; ++c) {
      const PetscInt cell = cells ? cells[c] : c;
      const PetscInt cind = c - cS;

      if (hasBdPrec) {
        if (hasBdJac) {
          if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
          if (!isMatIS) {
            ierr = DMPlexMatSetClosure(plex, section, globalSection, Jac, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
          } else {
            Mat lJ;

            ierr = MatISGetLocalMat(Jac,&lJ);CHKERRQ(ierr);
            ierr = DMPlexMatSetClosure(plex, section, subSection, lJ, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
          }
        }
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMatP[cind*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatISP) {
          ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(plex, section, subSection, lJ, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      } else if (hasBdJac) {
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatISP) {
          ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(plex, section, subSection, lJ, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMPlexRestoreCellFields(dm, cellIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, hasBdJac  ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMat);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, hasBdPrec ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMatP);CHKERRQ(ierr);
  ierr = PetscFree(faces);CHKERRQ(ierr);
  ierr = ISDestroy(&chunkIS);CHKERRQ(ierr);
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  if (maxDegree <= 1) {
    ierr = DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
  } else {
    PetscInt f;
    for (f = 0; f < Nf; ++f) {
      if (geoms) {ierr = DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE, &geoms[f]);CHKERRQ(ierr);}
      if (quads) {ierr = PetscQuadratureDestroy(&quads[f]);CHKERRQ(ierr);}
    }
    ierr = PetscFree2(quads,geoms);CHKERRQ(ierr);
  }
  if (dmAux) {ierr = DMDestroy(&plexA);CHKERRQ(ierr);}
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  /* Assemble matrix */
  if (hasBdJac && hasBdPrec) {
    ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
