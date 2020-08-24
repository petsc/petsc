#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>
#include <petsc/private/hashmapij.h>

const char * const DMPlexInterpolatedFlags[] = {"none", "partial", "mixed", "full", "DMPlexInterpolatedFlag", "DMPLEX_INTERPOLATED_", NULL};

/* HashIJKL */

#include <petsc/private/hashmap.h>

typedef struct _PetscHashIJKLKey { PetscInt i, j, k, l; } PetscHashIJKLKey;

#define PetscHashIJKLKeyHash(key) \
  PetscHashCombine(PetscHashCombine(PetscHashInt((key).i),PetscHashInt((key).j)), \
                   PetscHashCombine(PetscHashInt((key).k),PetscHashInt((key).l)))

#define PetscHashIJKLKeyEqual(k1,k2) \
  (((k1).i==(k2).i) ? ((k1).j==(k2).j) ? ((k1).k==(k2).k) ? ((k1).l==(k2).l) : 0 : 0 : 0)

PETSC_HASH_MAP(HashIJKL, PetscHashIJKLKey, PetscInt, PetscHashIJKLKeyHash, PetscHashIJKLKeyEqual, -1)

static PetscSFNode _PetscInvalidSFNode = {-1, -1};

typedef struct _PetscHashIJKLRemoteKey { PetscSFNode i, j, k, l; } PetscHashIJKLRemoteKey;

#define PetscHashIJKLRemoteKeyHash(key) \
  PetscHashCombine(PetscHashCombine(PetscHashInt((key).i.rank + (key).i.index),PetscHashInt((key).j.rank + (key).j.index)), \
                   PetscHashCombine(PetscHashInt((key).k.rank + (key).k.index),PetscHashInt((key).l.rank + (key).l.index)))

#define PetscHashIJKLRemoteKeyEqual(k1,k2) \
  (((k1).i.rank==(k2).i.rank) ? ((k1).i.index==(k2).i.index) ? ((k1).j.rank==(k2).j.rank) ? ((k1).j.index==(k2).j.index) ? ((k1).k.rank==(k2).k.rank) ? ((k1).k.index==(k2).k.index) ? ((k1).l.rank==(k2).l.rank) ? ((k1).l.index==(k2).l.index) : 0 : 0 : 0 : 0 : 0 : 0 : 0)

PETSC_HASH_MAP(HashIJKLRemote, PetscHashIJKLRemoteKey, PetscSFNode, PetscHashIJKLRemoteKeyHash, PetscHashIJKLRemoteKeyEqual, _PetscInvalidSFNode)

static PetscErrorCode PetscSortSFNode(PetscInt n, PetscSFNode A[])
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 1; i < n; ++i) {
    PetscSFNode x = A[i];
    PetscInt    j;

    for (j = i-1; j >= 0; --j) {
      if ((A[j].rank > x.rank) || (A[j].rank == x.rank && A[j].index > x.index)) break;
      A[j+1] = A[j];
    }
    A[j+1] = x;
  }
  PetscFunctionReturn(0);
}

/*
  DMPlexGetRawFaces_Internal - Gets groups of vertices that correspond to faces for the given cone
*/
PetscErrorCode DMPlexGetRawFaces_Internal(DM dm, DMPolytopeType ct, const PetscInt cone[], PetscInt *numFaces, const DMPolytopeType *faceTypes[], const PetscInt *faceSizes[], const PetscInt *faces[])
{
  DMPolytopeType *typesTmp;
  PetscInt       *sizesTmp, *facesTmp;
  PetscInt        maxConeSize, maxSupportSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cone) PetscValidIntPointer(cone, 3);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  if (faceTypes) {ierr = DMGetWorkArray(dm, PetscMax(maxConeSize, maxSupportSize),           MPIU_INT, &typesTmp);CHKERRQ(ierr);}
  if (faceSizes) {ierr = DMGetWorkArray(dm, PetscMax(maxConeSize, maxSupportSize),           MPIU_INT, &sizesTmp);CHKERRQ(ierr);}
  if (faces)     {ierr = DMGetWorkArray(dm, PetscSqr(PetscMax(maxConeSize, maxSupportSize)), MPIU_INT, &facesTmp);CHKERRQ(ierr);}
  switch (ct) {
    case DM_POLYTOPE_POINT:
      if (numFaces) *numFaces = 0;
      break;
    case DM_POLYTOPE_SEGMENT:
      if (numFaces) *numFaces = 2;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_POINT; typesTmp[1] = DM_POLYTOPE_POINT;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 1; sizesTmp[1] = 1;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
      if (numFaces) *numFaces = 2;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_POINT; typesTmp[1] = DM_POLYTOPE_POINT;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 1; sizesTmp[1] = 1;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_TRIANGLE:
      if (numFaces) *numFaces = 3;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_SEGMENT; typesTmp[1] = DM_POLYTOPE_SEGMENT; typesTmp[2] = DM_POLYTOPE_SEGMENT;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 2; sizesTmp[1] = 2; sizesTmp[2] = 2;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[0];
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      /* Vertices follow right hand rule */
      if (numFaces) *numFaces = 4;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_SEGMENT; typesTmp[1] = DM_POLYTOPE_SEGMENT; typesTmp[2] = DM_POLYTOPE_SEGMENT; typesTmp[3] = DM_POLYTOPE_SEGMENT;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 2; sizesTmp[1] = 2; sizesTmp[2] = 2; sizesTmp[3] = 2;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[3];
        facesTmp[6] = cone[3]; facesTmp[7] = cone[0];
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
      if (numFaces) *numFaces = 4;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_SEGMENT; typesTmp[1] = DM_POLYTOPE_SEGMENT; typesTmp[2] = DM_POLYTOPE_POINT_PRISM_TENSOR; typesTmp[3] = DM_POLYTOPE_POINT_PRISM_TENSOR;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 2; sizesTmp[1] = 2; sizesTmp[2] = 2; sizesTmp[3] = 2;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[2]; facesTmp[3] = cone[3];
        facesTmp[4] = cone[0]; facesTmp[5] = cone[2];
        facesTmp[6] = cone[1]; facesTmp[7] = cone[3];
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_TETRAHEDRON:
      /* Vertices of first face follow right hand rule and normal points away from last vertex */
      if (numFaces) *numFaces = 4;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_TRIANGLE; typesTmp[1] = DM_POLYTOPE_TRIANGLE; typesTmp[2] = DM_POLYTOPE_TRIANGLE; typesTmp[3] = DM_POLYTOPE_TRIANGLE;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 3; sizesTmp[1] = 3; sizesTmp[2] = 3; sizesTmp[3] = 3;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2];
        facesTmp[3] = cone[0]; facesTmp[4]  = cone[3]; facesTmp[5]  = cone[1];
        facesTmp[6] = cone[0]; facesTmp[7]  = cone[2]; facesTmp[8]  = cone[3];
        facesTmp[9] = cone[2]; facesTmp[10] = cone[1]; facesTmp[11] = cone[3];
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_HEXAHEDRON:
      /*  7--------6
         /|       /|
        / |      / |
       4--------5  |
       |  |     |  |
       |  |     |  |
       |  1--------2
       | /      | /
       |/       |/
       0--------3
       */
      if (numFaces) *numFaces = 6;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_QUADRILATERAL; typesTmp[1] = DM_POLYTOPE_QUADRILATERAL; typesTmp[2] = DM_POLYTOPE_QUADRILATERAL;
        typesTmp[3] = DM_POLYTOPE_QUADRILATERAL; typesTmp[4] = DM_POLYTOPE_QUADRILATERAL; typesTmp[5] = DM_POLYTOPE_QUADRILATERAL;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 4; sizesTmp[1] = 4; sizesTmp[2] = 4; sizesTmp[3] = 4; sizesTmp[4] = 4; sizesTmp[5] = 4;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2]; facesTmp[3]  = cone[3]; /* Bottom */
        facesTmp[4]  = cone[4]; facesTmp[5]  = cone[5]; facesTmp[6]  = cone[6]; facesTmp[7]  = cone[7]; /* Top */
        facesTmp[8]  = cone[0]; facesTmp[9]  = cone[3]; facesTmp[10] = cone[5]; facesTmp[11] = cone[4]; /* Front */
        facesTmp[12] = cone[2]; facesTmp[13] = cone[1]; facesTmp[14] = cone[7]; facesTmp[15] = cone[6]; /* Back */
        facesTmp[16] = cone[3]; facesTmp[17] = cone[2]; facesTmp[18] = cone[6]; facesTmp[19] = cone[5]; /* Right */
        facesTmp[20] = cone[0]; facesTmp[21] = cone[4]; facesTmp[22] = cone[7]; facesTmp[23] = cone[1]; /* Left */
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_TRI_PRISM:
      if (numFaces) *numFaces = 5;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_TRIANGLE; typesTmp[1] = DM_POLYTOPE_TRIANGLE;
        typesTmp[2] = DM_POLYTOPE_QUADRILATERAL; typesTmp[3] = DM_POLYTOPE_QUADRILATERAL; typesTmp[4] = DM_POLYTOPE_QUADRILATERAL;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 3; sizesTmp[1] = 3;
        sizesTmp[2] = 4; sizesTmp[3] = 4; sizesTmp[4] = 4;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2];                         /* Bottom */
        facesTmp[3]  = cone[3]; facesTmp[4]  = cone[4]; facesTmp[5]  = cone[5];                         /* Top */
        facesTmp[6]  = cone[0]; facesTmp[7]  = cone[2]; facesTmp[8]  = cone[4]; facesTmp[9]  = cone[3]; /* Back left */
        facesTmp[10] = cone[2]; facesTmp[11] = cone[1]; facesTmp[12] = cone[5]; facesTmp[13] = cone[4]; /* Front */
        facesTmp[14] = cone[1]; facesTmp[15] = cone[0]; facesTmp[16] = cone[3]; facesTmp[17] = cone[5]; /* Back right */
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
      if (numFaces)     *numFaces = 5;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_TRIANGLE; typesTmp[1] = DM_POLYTOPE_TRIANGLE;
        typesTmp[2] = DM_POLYTOPE_SEG_PRISM_TENSOR; typesTmp[3] = DM_POLYTOPE_SEG_PRISM_TENSOR; typesTmp[4] = DM_POLYTOPE_SEG_PRISM_TENSOR;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 3; sizesTmp[1] = 3;
        sizesTmp[2] = 4; sizesTmp[3] = 4; sizesTmp[4] = 4;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2];                         /* Bottom */
        facesTmp[3]  = cone[3]; facesTmp[4]  = cone[4]; facesTmp[5]  = cone[5];                         /* Top */
        facesTmp[6]  = cone[0]; facesTmp[7]  = cone[1]; facesTmp[8]  = cone[3]; facesTmp[9]  = cone[4]; /* Back left */
        facesTmp[10] = cone[1]; facesTmp[11] = cone[2]; facesTmp[12] = cone[4]; facesTmp[13] = cone[5]; /* Back right */
        facesTmp[14] = cone[2]; facesTmp[15] = cone[0]; facesTmp[16] = cone[5]; facesTmp[17] = cone[3]; /* Front */
        *faces = facesTmp;
      }
      break;
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
      /*  7--------6
         /|       /|
        / |      / |
       4--------5  |
       |  |     |  |
       |  |     |  |
       |  3--------2
       | /      | /
       |/       |/
       0--------1
       */
      if (numFaces) *numFaces = 6;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_QUADRILATERAL;    typesTmp[1] = DM_POLYTOPE_QUADRILATERAL;    typesTmp[2] = DM_POLYTOPE_SEG_PRISM_TENSOR;
        typesTmp[3] = DM_POLYTOPE_SEG_PRISM_TENSOR; typesTmp[4] = DM_POLYTOPE_SEG_PRISM_TENSOR; typesTmp[5] = DM_POLYTOPE_SEG_PRISM_TENSOR;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 4; sizesTmp[1] = 4; sizesTmp[2] = 4; sizesTmp[3] = 4; sizesTmp[4] = 4; sizesTmp[5] = 4;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2]; facesTmp[3]  = cone[3]; /* Bottom */
        facesTmp[4]  = cone[4]; facesTmp[5]  = cone[5]; facesTmp[6]  = cone[6]; facesTmp[7]  = cone[7]; /* Top */
        facesTmp[8]  = cone[0]; facesTmp[9]  = cone[1]; facesTmp[10] = cone[4]; facesTmp[11] = cone[5]; /* Front */
        facesTmp[12] = cone[1]; facesTmp[13] = cone[2]; facesTmp[14] = cone[5]; facesTmp[15] = cone[6]; /* Right */
        facesTmp[16] = cone[2]; facesTmp[17] = cone[3]; facesTmp[18] = cone[6]; facesTmp[19] = cone[7]; /* Back */
        facesTmp[20] = cone[3]; facesTmp[21] = cone[0]; facesTmp[22] = cone[7]; facesTmp[23] = cone[4]; /* Left */
        *faces = facesTmp;
      }
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No face description for cell type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexRestoreRawFaces_Internal(DM dm, DMPolytopeType ct, const PetscInt cone[], PetscInt *numFaces, const DMPolytopeType *faceTypes[], const PetscInt *faceSizes[], const PetscInt *faces[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (faceTypes) {ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, (void *) faceTypes);CHKERRQ(ierr);}
  if (faceSizes) {ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, (void *) faceSizes);CHKERRQ(ierr);}
  if (faces)     {ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, (void *) faces);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* This interpolates faces for cells at some stratum */
static PetscErrorCode DMPlexInterpolateFaces_Internal(DM dm, PetscInt cellDepth, DM idm)
{
  DMLabel        ctLabel;
  PetscHashIJKL  faceTable;
  PetscInt       faceTypeNum[DM_NUM_POLYTOPES];
  PetscInt       depth, d, Np, cStart, cEnd, c, fStart, fEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscHashIJKLCreate(&faceTable);CHKERRQ(ierr);
  ierr = PetscArrayzero(faceTypeNum, DM_NUM_POLYTOPES);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, cellDepth, &cStart, &cEnd);CHKERRQ(ierr);
  /* Number new faces and save face vertices in hash table */
  ierr = DMPlexGetDepthStratum(dm, depth > cellDepth ? cellDepth : 0, NULL, &fStart);CHKERRQ(ierr);
  fEnd = fStart;
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt       *cone, *faceSizes, *faces;
    const DMPolytopeType *faceTypes;
    DMPolytopeType        ct;
    PetscInt              numFaces, cf, foff = 0;

    ierr = DMPlexGetCellType(dm, c, &ct);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces);CHKERRQ(ierr);
    for (cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
      const PetscInt       faceSize = faceSizes[cf];
      const DMPolytopeType faceType = faceTypes[cf];
      const PetscInt      *face     = &faces[foff];
      PetscHashIJKLKey     key;
      PetscHashIter        iter;
      PetscBool            missing;

      if (faceSize > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %D > 4", faceSize);
      key.i = face[0];
      key.j = faceSize > 1 ? face[1] : PETSC_MAX_INT;
      key.k = faceSize > 2 ? face[2] : PETSC_MAX_INT;
      key.l = faceSize > 3 ? face[3] : PETSC_MAX_INT;
      ierr = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
      ierr = PetscHashIJKLPut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
      if (missing) {
        ierr = PetscHashIJKLIterSet(faceTable, iter, fEnd++);CHKERRQ(ierr);
        ++faceTypeNum[faceType];
      }
    }
    ierr = DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces);CHKERRQ(ierr);
  }
  /* We need to number faces contiguously among types */
  {
    PetscInt faceTypeStart[DM_NUM_POLYTOPES], ct, numFT = 0;

    for (ct = 0; ct < DM_NUM_POLYTOPES; ++ct) {if (faceTypeNum[ct]) ++numFT; faceTypeStart[ct] = 0;}
    if (numFT > 1) {
      ierr = PetscHashIJKLClear(faceTable);CHKERRQ(ierr);
      faceTypeStart[0] = fStart;
      for (ct = 1; ct < DM_NUM_POLYTOPES; ++ct) faceTypeStart[ct] = faceTypeStart[ct-1] + faceTypeNum[ct-1];
      for (c = cStart; c < cEnd; ++c) {
        const PetscInt       *cone, *faceSizes, *faces;
        const DMPolytopeType *faceTypes;
        DMPolytopeType        ct;
        PetscInt              numFaces, cf, foff = 0;

        ierr = DMPlexGetCellType(dm, c, &ct);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces);CHKERRQ(ierr);
        for (cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
          const PetscInt       faceSize = faceSizes[cf];
          const DMPolytopeType faceType = faceTypes[cf];
          const PetscInt      *face     = &faces[foff];
          PetscHashIJKLKey     key;
          PetscHashIter        iter;
          PetscBool            missing;

          if (faceSize > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %D > 4", faceSize);
          key.i = face[0];
          key.j = faceSize > 1 ? face[1] : PETSC_MAX_INT;
          key.k = faceSize > 2 ? face[2] : PETSC_MAX_INT;
          key.l = faceSize > 3 ? face[3] : PETSC_MAX_INT;
          ierr = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
          ierr = PetscHashIJKLPut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
          if (missing) {ierr = PetscHashIJKLIterSet(faceTable, iter, faceTypeStart[faceType]++);CHKERRQ(ierr);}
        }
        ierr = DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces);CHKERRQ(ierr);
      }
      for (ct = 1; ct < DM_NUM_POLYTOPES; ++ct) {
        if (faceTypeStart[ct] != faceTypeStart[ct-1] + faceTypeNum[ct]) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent numbering for cell type %s, %D != %D + %D", DMPolytopeTypes[ct], faceTypeStart[ct], faceTypeStart[ct-1], faceTypeNum[ct]);
      }
    }
  }
  /* Add new points, always at the end of the numbering */
  ierr = DMPlexGetChart(dm, NULL, &Np);CHKERRQ(ierr);
  ierr = DMPlexSetChart(idm, 0, Np + (fEnd - fStart));CHKERRQ(ierr);
  /* Set cone sizes */
  /*   Must create the celltype label here so that we do not automatically try to compute the types */
  ierr = DMCreateLabel(idm, "celltype");CHKERRQ(ierr);
  ierr = DMPlexGetCellTypeLabel(idm, &ctLabel);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    DMPolytopeType ct;
    PetscInt       coneSize, pStart, pEnd, p;

    if (d == cellDepth) continue;
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(idm, p, coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
      ierr = DMPlexSetCellType(idm, p, ct);CHKERRQ(ierr);
    }
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt       *cone, *faceSizes, *faces;
    const DMPolytopeType *faceTypes;
    DMPolytopeType        ct;
    PetscInt              numFaces, cf, foff = 0;

    ierr = DMPlexGetCellType(dm, c, &ct);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces);CHKERRQ(ierr);
    ierr = DMPlexSetCellType(idm, c, ct);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(idm, c, numFaces);CHKERRQ(ierr);
    for (cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
      const PetscInt       faceSize = faceSizes[cf];
      const DMPolytopeType faceType = faceTypes[cf];
      const PetscInt      *face     = &faces[foff];
      PetscHashIJKLKey     key;
      PetscHashIter        iter;
      PetscBool            missing;
      PetscInt             f;

      if (faceSize > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %D > 4", faceSize);
      key.i = face[0];
      key.j = faceSize > 1 ? face[1] : PETSC_MAX_INT;
      key.k = faceSize > 2 ? face[2] : PETSC_MAX_INT;
      key.l = faceSize > 3 ? face[3] : PETSC_MAX_INT;
      ierr = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
      ierr = PetscHashIJKLPut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
      if (missing) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing face (cell %D, lf %D)", c, cf);
      ierr = PetscHashIJKLIterGet(faceTable, iter, &f);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(idm, f, faceSize);CHKERRQ(ierr);
      ierr = DMPlexSetCellType(idm, f, faceType);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces);CHKERRQ(ierr);
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  /* Initialize cones so we do not need the bash table to tell us that a cone has been set */
  {
    PetscSection cs;
    PetscInt    *cones, csize;

    ierr = DMPlexGetConeSection(idm, &cs);CHKERRQ(ierr);
    ierr = DMPlexGetCones(idm, &cones);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(cs, &csize);CHKERRQ(ierr);
    for (c = 0; c < csize; ++c) cones[c] = -1;
  }
  /* Set cones */
  for (d = 0; d <= depth; ++d) {
    const PetscInt *cone;
    PetscInt        pStart, pEnd, p;

    if (d == cellDepth) continue;
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexSetCone(idm, p, cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(idm, p, cone);CHKERRQ(ierr);
    }
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt       *cone, *faceSizes, *faces;
    const DMPolytopeType *faceTypes;
    DMPolytopeType        ct;
    PetscInt              numFaces, cf, foff = 0;

    ierr = DMPlexGetCellType(dm, c, &ct);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces);CHKERRQ(ierr);
    for (cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
      DMPolytopeType   faceType = faceTypes[cf];
      const PetscInt   faceSize = faceSizes[cf];
      const PetscInt  *face     = &faces[foff];
      const PetscInt  *fcone;
      PetscHashIJKLKey key;
      PetscHashIter    iter;
      PetscBool        missing;
      PetscInt         f;

      if (faceSize > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %D > 4", faceSize);
      key.i = face[0];
      key.j = faceSize > 1 ? face[1] : PETSC_MAX_INT;
      key.k = faceSize > 2 ? face[2] : PETSC_MAX_INT;
      key.l = faceSize > 3 ? face[3] : PETSC_MAX_INT;
      ierr = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
      ierr = PetscHashIJKLPut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
      ierr = PetscHashIJKLIterGet(faceTable, iter, &f);CHKERRQ(ierr);
      ierr = DMPlexInsertCone(idm, c, cf, f);CHKERRQ(ierr);
      ierr = DMPlexGetCone(idm, f, &fcone);CHKERRQ(ierr);
      if (fcone[0] < 0) {ierr = DMPlexSetCone(idm, f, face);CHKERRQ(ierr);}
      /* TODO This should be unnecessary since we have autoamtic orientation */
      {
        /* when matching hybrid faces in 3D, only few cases are possible.
           Face traversal however can no longer follow the usual convention, this seems a serious issue to me */
        PetscInt        tquad_map[4][4] = { {PETSC_MIN_INT,            0,PETSC_MIN_INT,PETSC_MIN_INT},
                                            {           -1,PETSC_MIN_INT,PETSC_MIN_INT,PETSC_MIN_INT},
                                            {PETSC_MIN_INT,PETSC_MIN_INT,PETSC_MIN_INT,            1},
                                            {PETSC_MIN_INT,PETSC_MIN_INT,           -2,PETSC_MIN_INT} };
        PetscInt        i, i2, j;
        const PetscInt *cone;
        PetscInt        coneSize, ornt;

        /* Orient face: Do not allow reverse orientation at the first vertex */
        ierr = DMPlexGetConeSize(idm, f, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(idm, f, &cone);CHKERRQ(ierr);
        if (coneSize != faceSize) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of face vertices %D for face %D should be %D", coneSize, f, faceSize);
        /* - First find the initial vertex */
        for (i = 0; i < faceSize; ++i) if (face[0] == cone[i]) break;
        /* If we want to compare tensor faces to regular faces, we have to flip them and take the else branch here */
        if (faceType == DM_POLYTOPE_SEG_PRISM_TENSOR) {
          /* find the second vertex */
          for (i2 = 0; i2 < faceSize; ++i2) if (face[1] == cone[i2]) break;
          switch (faceSize) {
          case 2:
            ornt = i ? -2 : 0;
            break;
          case 4:
            ornt = tquad_map[i][i2];
            break;
          default: SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unhandled face size %D for face %D in cell %D", faceSize, f, c);
          }
        } else {
          /* Try forward comparison */
          for (j = 0; j < faceSize; ++j) if (face[j] != cone[(i+j)%faceSize]) break;
          if (j == faceSize) {
            if ((faceSize == 2) && (i == 1)) ornt = -2;
            else                             ornt = i;
          } else {
            /* Try backward comparison */
            for (j = 0; j < faceSize; ++j) if (face[j] != cone[(i+faceSize-j)%faceSize]) break;
            if (j == faceSize) {
              if (i == 0) ornt = -faceSize;
              else        ornt = -i;
            } else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not determine orientation of face %D in cell %D", f, c);
          }
        }
        ierr = DMPlexInsertConeOrientation(idm, c, cf, ornt);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces);CHKERRQ(ierr);
  }
  ierr = PetscHashIJKLDestroy(&faceTable);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMPlexStratify(idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SortRmineRremoteByRemote_Private(PetscSF sf, PetscInt *rmine1[], PetscInt *rremote1[])
{
  PetscInt            nleaves;
  PetscInt            nranks;
  const PetscMPIInt  *ranks=NULL;
  const PetscInt     *roffset=NULL, *rmine=NULL, *rremote=NULL;
  PetscInt            n, o, r;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote);CHKERRQ(ierr);
  nleaves = roffset[nranks];
  ierr = PetscMalloc2(nleaves, rmine1, nleaves, rremote1);CHKERRQ(ierr);
  for (r=0; r<nranks; r++) {
    /* simultaneously sort rank-wise portions of rmine & rremote by values in rremote
       - to unify order with the other side */
    o = roffset[r];
    n = roffset[r+1] - o;
    ierr = PetscArraycpy(&(*rmine1)[o], &rmine[o], n);CHKERRQ(ierr);
    ierr = PetscArraycpy(&(*rremote1)[o], &rremote[o], n);CHKERRQ(ierr);
    ierr = PetscSortIntWithArray(n, &(*rremote1)[o], &(*rmine1)[o]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexOrientInterface_Internal(DM dm)
{
  /* Here we only compare first 2 points of the cone. Full cone size would lead to stronger self-checking. */
  PetscInt          masterCone[2];
  PetscInt          (*roots)[2], (*leaves)[2];
  PetscMPIInt       (*rootsRanks)[2], (*leavesRanks)[2];

  PetscSF           sf=NULL;
  const PetscInt    *locals=NULL;
  const PetscSFNode *remotes=NULL;
  PetscInt           nroots, nleaves, p, c;
  PetscInt           nranks, n, o, r;
  const PetscMPIInt *ranks=NULL;
  const PetscInt    *roffset=NULL;
  PetscInt          *rmine1=NULL, *rremote1=NULL; /* rmine and rremote copies simultaneously sorted by rank and rremote */
  const PetscInt    *cone=NULL;
  PetscInt           coneSize, ind0;
  MPI_Comm           comm;
  PetscMPIInt        rank,size;
  PetscInt           debug = 0;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &locals, &remotes);CHKERRQ(ierr);
  if (nroots < 0) PetscFunctionReturn(0);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, NULL, NULL);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-before_fix_dm_view");CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {ierr = DMPlexCheckPointSF(dm);CHKERRQ(ierr);}
  ierr = SortRmineRremoteByRemote_Private(sf, &rmine1, &rremote1);CHKERRQ(ierr);
  ierr = PetscMalloc4(nroots, &roots, nroots, &leaves, nroots, &rootsRanks, nroots, &leavesRanks);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  for (p = 0; p < nroots; ++p) {
    ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
    if (coneSize < 2) {
      for (c = 0; c < 2; c++) {
        roots[p][c] = -1;
        rootsRanks[p][c] = -1;
      }
      continue;
    }
    /* Translate all points to root numbering */
    for (c = 0; c < 2; c++) {
      ierr = PetscFindInt(cone[c], nleaves, locals, &ind0);CHKERRQ(ierr);
      if (ind0 < 0) {
        roots[p][c] = cone[c];
        rootsRanks[p][c] = rank;
      } else {
        roots[p][c] = remotes[ind0].index;
        rootsRanks[p][c] = remotes[ind0].rank;
      }
    }
  }
  for (p = 0; p < nroots; ++p) {
    for (c = 0; c < 2; c++) {
      leaves[p][c] = -2;
      leavesRanks[p][c] = -2;
    }
  }
  ierr = PetscSFBcastBegin(sf, MPIU_2INT, roots, leaves);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPI_2INT, rootsRanks, leavesRanks);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_2INT, roots, leaves);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPI_2INT, rootsRanks, leavesRanks);CHKERRQ(ierr);
  if (debug) {ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);}
  if (debug && rank == 0) {ierr = PetscSynchronizedPrintf(comm, "Referenced roots\n");CHKERRQ(ierr);}
  for (p = 0; p < nroots; ++p) {
    if (leaves[p][0] < 0) continue;
    ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
    if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]  %4D: cone=[%4D %4D] roots=[(%d,%4D) (%d,%4D)] leaves=[(%d,%4D) (%d,%4D)]", rank, p, cone[0], cone[1], rootsRanks[p][0], roots[p][0], rootsRanks[p][1], roots[p][1], leavesRanks[p][0], leaves[p][0], leavesRanks[p][1], leaves[p][1]);CHKERRQ(ierr);}
    if ((leaves[p][0] != roots[p][0]) || (leaves[p][1] != roots[p][1]) || (leavesRanks[p][0] != rootsRanks[p][0]) || (leavesRanks[p][1] != rootsRanks[p][1])) {
      /* Translate these two leaves to my cone points; masterCone means desired order p's cone points */
      for (c = 0; c < 2; c++) {
        if (leavesRanks[p][c] == rank) {
          /* A local leave is just taken as it is */
          masterCone[c] = leaves[p][c];
          continue;
        }
        /* Find index of rank leavesRanks[p][c] among remote ranks */
        /* No need for PetscMPIIntCast because these integers were originally cast from PetscMPIInt. */
        ierr = PetscFindMPIInt((PetscMPIInt)leavesRanks[p][c], nranks, ranks, &r);CHKERRQ(ierr);
        if (PetscUnlikely(r < 0)) SETERRQ7(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D cone[%D]=%D root (%d,%D) leave (%d,%D): leave rank not found among remote ranks",p,c,cone[c],rootsRanks[p][c],roots[p][c],leavesRanks[p][c],leaves[p][c]);
        if (PetscUnlikely(ranks[r] < 0 || ranks[r] >= size)) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_PLIB, "p=%D c=%D commsize=%d: ranks[%D] = %d makes no sense",p,c,size,r,ranks[r]);
        /* Find point leaves[p][c] among remote points aimed at rank leavesRanks[p][c] */
        o = roffset[r];
        n = roffset[r+1] - o;
        ierr = PetscFindInt(leaves[p][c], n, &rremote1[o], &ind0);CHKERRQ(ierr);
        if (PetscUnlikely(ind0 < 0)) SETERRQ7(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D cone[%D]=%D root (%d,%D) leave (%d,%D): corresponding remote point not found - it seems there is missing connection in point SF!",p,c,cone[c],rootsRanks[p][c],roots[p][c],leavesRanks[p][c],leaves[p][c]);
        /* Get the corresponding local point */
        masterCone[c] = rmine1[o+ind0];CHKERRQ(ierr);
      }
      if (debug) {ierr = PetscSynchronizedPrintf(comm, " masterCone=[%4D %4D]\n", masterCone[0], masterCone[1]);CHKERRQ(ierr);}
      /* Set the desired order of p's cone points and fix orientations accordingly */
      /* Vaclav's note: Here we only compare first 2 points of the cone. Full cone size would lead to stronger self-checking. */
      ierr = DMPlexOrientCell(dm, p, 2, masterCone);CHKERRQ(ierr);
    } else if (debug) {ierr = PetscSynchronizedPrintf(comm, " ==\n");CHKERRQ(ierr);}
  }
  if (debug) {
    ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  }
  ierr = DMViewFromOptions(dm, NULL, "-after_fix_dm_view");CHKERRQ(ierr);
  ierr = PetscFree4(roots, leaves, rootsRanks, leavesRanks);CHKERRQ(ierr);
  ierr = PetscFree2(rmine1, rremote1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IntArrayViewFromOptions(MPI_Comm comm, const char opt[], const char name[], const char idxname[], const char valname[], PetscInt n, const PetscInt a[])
{
  PetscInt       idx;
  PetscMPIInt    rank;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL, NULL, opt, &flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(comm, "[%d]%s:\n", rank, name);CHKERRQ(ierr);
  for (idx = 0; idx < n; ++idx) {ierr = PetscSynchronizedPrintf(comm, "[%d]%s %D %s %D\n", rank, idxname, idx, valname, a[idx]);CHKERRQ(ierr);}
  ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SFNodeArrayViewFromOptions(MPI_Comm comm, const char opt[], const char name[], const char idxname[], PetscInt n, const PetscSFNode a[])
{
  PetscInt       idx;
  PetscMPIInt    rank;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL, NULL, opt, &flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(comm, "[%d]%s:\n", rank, name);CHKERRQ(ierr);
  if (idxname) {
    for (idx = 0; idx < n; ++idx) {ierr = PetscSynchronizedPrintf(comm, "[%d]%s %D rank %D index %D\n", rank, idxname, idx, a[idx].rank, a[idx].index);CHKERRQ(ierr);}
  } else {
    for (idx = 0; idx < n; ++idx) {ierr = PetscSynchronizedPrintf(comm, "[%d]rank %D index %D\n", rank, a[idx].rank, a[idx].index);CHKERRQ(ierr);}
  }
  ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMapToLocalPoint(DM dm, PetscHMapIJ remotehash, PetscSFNode remotePoint, PetscInt *localPoint)
{
  PetscSF         sf;
  const PetscInt *locals;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, NULL, &locals, NULL);CHKERRQ(ierr);
  if (remotePoint.rank == rank) {
    *localPoint = remotePoint.index;
  } else {
    PetscHashIJKey key;
    PetscInt       l;

    key.i = remotePoint.index;
    key.j = remotePoint.rank;
    ierr = PetscHMapIJGet(remotehash, key, &l);CHKERRQ(ierr);
    if (l >= 0) {
      *localPoint = locals[l];
    } else PetscFunctionReturn(1);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMapToGlobalPoint(DM dm, PetscInt localPoint, PetscSFNode *remotePoint)
{
  PetscSF            sf;
  const PetscInt    *locals, *rootdegree;
  const PetscSFNode *remotes;
  PetscInt           Nl, l;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, &Nl, &locals, &remotes);CHKERRQ(ierr);
  if (Nl < 0) goto owned;
  ierr = PetscSFComputeDegreeBegin(sf, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf, &rootdegree);CHKERRQ(ierr);
  if (rootdegree[localPoint]) goto owned;
  ierr = PetscFindInt(localPoint, Nl, locals, &l);CHKERRQ(ierr);
  if (l < 0) PetscFunctionReturn(1);
  *remotePoint = remotes[l];
  PetscFunctionReturn(0);
  owned:
  remotePoint->rank  = rank;
  remotePoint->index = localPoint;
  PetscFunctionReturn(0);
}


static PetscErrorCode DMPlexPointIsShared(DM dm, PetscInt p, PetscBool *isShared)
{
  PetscSF         sf;
  const PetscInt *locals, *rootdegree;
  PetscInt        Nl, idx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *isShared = PETSC_FALSE;
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, NULL, &Nl, &locals, NULL);CHKERRQ(ierr);
  if (Nl < 0) PetscFunctionReturn(0);
  ierr = PetscFindInt(p, Nl, locals, &idx);CHKERRQ(ierr);
  if (idx >= 0) {*isShared = PETSC_TRUE; PetscFunctionReturn(0);}
  ierr = PetscSFComputeDegreeBegin(sf, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf, &rootdegree);CHKERRQ(ierr);
  if (rootdegree[p] > 0) *isShared = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexConeIsShared(DM dm, PetscInt p, PetscBool *isShared)
{
  const PetscInt *cone;
  PetscInt        coneSize, c;
  PetscBool       cShared = PETSC_TRUE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (c = 0; c < coneSize; ++c) {
    PetscBool pointShared;

    ierr = DMPlexPointIsShared(dm, cone[c], &pointShared);CHKERRQ(ierr);
    cShared = (PetscBool) (cShared && pointShared);
  }
  *isShared = coneSize ? cShared : PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetConeMinimum(DM dm, PetscInt p, PetscSFNode *cpmin)
{
  const PetscInt *cone;
  PetscInt        coneSize, c;
  PetscSFNode     cmin = {PETSC_MAX_INT, PETSC_MAX_INT}, missing = {-1, -1};
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (c = 0; c < coneSize; ++c) {
    PetscSFNode rcp;

    ierr = DMPlexMapToGlobalPoint(dm, cone[c], &rcp);
    if (ierr) {
      cmin = missing;
    } else {
      cmin = (rcp.rank < cmin.rank) || (rcp.rank == cmin.rank && rcp.index < cmin.index) ? rcp : cmin;
    }
  }
  *cpmin = coneSize ? cmin : missing;
  PetscFunctionReturn(0);
}

/*
  Each shared face has an entry in the candidates array:
    (-1, coneSize-1), {(global cone point)}
  where the set is missing the point p which we use as the key for the face
*/
static PetscErrorCode DMPlexAddSharedFace_Private(DM dm, PetscSection candidateSection, PetscSFNode candidates[], PetscHMapIJ faceHash, PetscInt p, PetscBool debug)
{
  MPI_Comm        comm;
  const PetscInt *support;
  PetscInt        supportSize, s, off = 0, idx = 0, overlap, cellHeight, height;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetOverlap(dm, &overlap);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetPointHeight(dm, p, &height);CHKERRQ(ierr);
  if (!overlap && height <= cellHeight+1) {
    /* cells can't be shared for non-overlapping meshes */
    if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Skipping face %D to avoid adding cell to hashmap since this is nonoverlapping mesh\n", rank, p);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
  if (candidates) {ierr = PetscSectionGetOffset(candidateSection, p, &off);CHKERRQ(ierr);}
  for (s = 0; s < supportSize; ++s) {
    const PetscInt  face = support[s];
    const PetscInt *cone;
    PetscSFNode     cpmin={-1,-1}, rp={-1,-1};
    PetscInt        coneSize, c, f;
    PetscBool       isShared = PETSC_FALSE;
    PetscHashIJKey  key;

    /* Only add point once */
    if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Support face %D\n", rank, face);CHKERRQ(ierr);}
    key.i = p;
    key.j = face;
    ierr = PetscHMapIJGet(faceHash, key, &f);CHKERRQ(ierr);
    if (f >= 0) continue;
    ierr = DMPlexConeIsShared(dm, face, &isShared);CHKERRQ(ierr);
    ierr = DMPlexGetConeMinimum(dm, face, &cpmin);CHKERRQ(ierr);
    ierr = DMPlexMapToGlobalPoint(dm, p, &rp);CHKERRQ(ierr);
    if (debug) {
      ierr = PetscSynchronizedPrintf(comm, "[%d]      Face point %D is shared: %d\n", rank, face, (int) isShared);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "[%d]      Global point (%D, %D) Min Cone Point (%D, %D)\n", rank, rp.rank, rp.index, cpmin.rank, cpmin.index);CHKERRQ(ierr);
    }
    if (isShared && (rp.rank == cpmin.rank && rp.index == cpmin.index)) {
      ierr = PetscHMapIJSet(faceHash, key, p);CHKERRQ(ierr);
      if (candidates) {
        if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Adding shared face %D at idx %D\n[%d]     ", rank, face, idx, rank);CHKERRQ(ierr);}
        ierr = DMPlexGetConeSize(dm, face, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, face, &cone);CHKERRQ(ierr);
        candidates[off+idx].rank    = -1;
        candidates[off+idx++].index = coneSize-1;
        candidates[off+idx].rank    = rank;
        candidates[off+idx++].index = face;
        for (c = 0; c < coneSize; ++c) {
          const PetscInt cp = cone[c];

          if (cp == p) continue;
          ierr = DMPlexMapToGlobalPoint(dm, cp, &candidates[off+idx]);CHKERRQ(ierr);
          if (debug) {ierr = PetscSynchronizedPrintf(comm, " (%D,%D)", candidates[off+idx].rank, candidates[off+idx].index);CHKERRQ(ierr);}
          ++idx;
        }
        if (debug) {ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);}
      } else {
        /* Add cone size to section */
        if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Scheduling shared face %D\n", rank, face);CHKERRQ(ierr);}
        ierr = DMPlexGetConeSize(dm, face, &coneSize);CHKERRQ(ierr);
        ierr = PetscHMapIJSet(faceHash, key, p);CHKERRQ(ierr);
        ierr = PetscSectionAddDof(candidateSection, p, coneSize+1);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexInterpolatePointSF - Insert interpolated points in the overlap into the PointSF in parallel, following local interpolation

  Collective on dm

  Input Parameters:
+ dm      - The interpolated DM
- pointSF - The initial SF without interpolated points

  Output Parameter:
. pointSF - The SF including interpolated points

  Level: developer

   Note: All debugging for this process can be turned on with the options: -dm_interp_pre_view -petscsf_interp_pre_view -petscsection_interp_candidate_view -petscsection_interp_candidate_remote_view -petscsection_interp_claim_view -petscsf_interp_pre_view -dmplex_interp_debug

.seealso: DMPlexInterpolate(), DMPlexUninterpolate()
@*/
PetscErrorCode DMPlexInterpolatePointSF(DM dm, PetscSF pointSF)
{
  MPI_Comm           comm;
  PetscHMapIJ        remoteHash;
  PetscHMapI         claimshash;
  PetscSection       candidateSection, candidateRemoteSection, claimSection;
  PetscSFNode       *candidates, *candidatesRemote, *claims;
  const PetscInt    *localPoints, *rootdegree;
  const PetscSFNode *remotePoints;
  PetscInt           ov, Nr, r, Nl, l;
  PetscInt           candidatesSize, candidatesRemoteSize, claimsSize;
  PetscBool          flg, debug = PETSC_FALSE;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(pointSF, PETSCSF_CLASSID, 3);
  ierr = DMPlexIsDistributed(dm, &flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  /* Set initial SF so that lower level queries work */
  ierr = DMSetPointSF(dm, pointSF);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetOverlap(dm, &ov);CHKERRQ(ierr);
  if (ov) SETERRQ(comm, PETSC_ERR_SUP, "Interpolation of overlapped DMPlex not implemented yet");
  ierr = PetscOptionsHasName(NULL, ((PetscObject) dm)->prefix, "-dmplex_interp_debug", &debug);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_interp_pre_view");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) pointSF, NULL, "-petscsf_interp_pre_view");CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_InterpolateSF,dm,0,0,0);CHKERRQ(ierr);
  /* Step 0: Precalculations */
  ierr = PetscSFGetGraph(pointSF, &Nr, &Nl, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (Nr < 0) SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "This DMPlex is distributed but input PointSF has no graph set");
  ierr = PetscHMapIJCreate(&remoteHash);CHKERRQ(ierr);
  for (l = 0; l < Nl; ++l) {
    PetscHashIJKey key;
    key.i = remotePoints[l].index;
    key.j = remotePoints[l].rank;
    ierr = PetscHMapIJSet(remoteHash, key, l);CHKERRQ(ierr);
  }
  /*   Compute root degree to identify shared points */
  ierr = PetscSFComputeDegreeBegin(pointSF, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(pointSF, &rootdegree);CHKERRQ(ierr);
  ierr = IntArrayViewFromOptions(comm, "-interp_root_degree_view", "Root degree", "point", "degree", Nr, rootdegree);CHKERRQ(ierr);
  /*
  1) Loop over each leaf point $p$ at depth $d$ in the SF
  \item Get set $F(p)$ of faces $f$ in the support of $p$ for which
  \begin{itemize}
    \item all cone points of $f$ are shared
    \item $p$ is the cone point with smallest canonical number
  \end{itemize}
  \item Send $F(p)$ and the cone of each face to the active root point $r(p)$
  \item At the root, if at least two faces with a given cone are present, including a local face, mark the face as shared \label{alg:rootStep} and choose the root face
  \item Send the root face from the root back to all leaf process
  \item Leaf processes add the shared face to the SF
  */
  /* Step 1: Construct section+SFNode array
       The section has entries for all shared faces for which we have a leaf point in the cone
       The array holds candidate shared faces, each face is refered to by the leaf point */
  ierr = PetscSectionCreate(comm, &candidateSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(candidateSection, 0, Nr);CHKERRQ(ierr);
  {
    PetscHMapIJ faceHash;

    ierr = PetscHMapIJCreate(&faceHash);CHKERRQ(ierr);
    for (l = 0; l < Nl; ++l) {
      const PetscInt p = localPoints[l];

      if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]  First pass leaf point %D\n", rank, p);CHKERRQ(ierr);}
      ierr = DMPlexAddSharedFace_Private(dm, candidateSection, NULL, faceHash, p, debug);CHKERRQ(ierr);
    }
    ierr = PetscHMapIJClear(faceHash);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(candidateSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(candidateSection, &candidatesSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(candidatesSize, &candidates);CHKERRQ(ierr);
    for (l = 0; l < Nl; ++l) {
      const PetscInt p = localPoints[l];

      if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]  Second pass leaf point %D\n", rank, p);CHKERRQ(ierr);}
      ierr = DMPlexAddSharedFace_Private(dm, candidateSection, candidates, faceHash, p, debug);CHKERRQ(ierr);
    }
    ierr = PetscHMapIJDestroy(&faceHash);CHKERRQ(ierr);
    if (debug) {ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);}
  }
  ierr = PetscObjectSetName((PetscObject) candidateSection, "Candidate Section");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) candidateSection, NULL, "-petscsection_interp_candidate_view");CHKERRQ(ierr);
  ierr = SFNodeArrayViewFromOptions(comm, "-petscsection_interp_candidate_view", "Candidates", NULL, candidatesSize, candidates);CHKERRQ(ierr);
  /* Step 2: Gather candidate section / array pair into the root partition via inverse(multi(pointSF)). */
  /*   Note that this section is indexed by offsets into leaves, not by point number */
  {
    PetscSF   sfMulti, sfInverse, sfCandidates;
    PetscInt *remoteOffsets;

    ierr = PetscSFGetMultiSF(pointSF, &sfMulti);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF(sfMulti, &sfInverse);CHKERRQ(ierr);
    ierr = PetscSectionCreate(comm, &candidateRemoteSection);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfInverse, candidateSection, &remoteOffsets, candidateRemoteSection);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sfInverse, candidateSection, remoteOffsets, candidateRemoteSection, &sfCandidates);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(candidateRemoteSection, &candidatesRemoteSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(candidatesRemoteSize, &candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sfCandidates, MPIU_2INT, candidates, candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfCandidates, MPIU_2INT, candidates, candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfInverse);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfCandidates);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject) candidateRemoteSection, "Remote Candidate Section");CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) candidateRemoteSection, NULL, "-petscsection_interp_candidate_remote_view");CHKERRQ(ierr);
    ierr = SFNodeArrayViewFromOptions(comm, "-petscsection_interp_candidate_remote_view", "Remote Candidates", NULL, candidatesRemoteSize, candidatesRemote);CHKERRQ(ierr);
  }
  /* Step 3: At the root, if at least two faces with a given cone are present, including a local face, mark the face as shared and choose the root face */
  {
    PetscHashIJKLRemote faceTable;
    PetscInt            idx, idx2;

    ierr = PetscHashIJKLRemoteCreate(&faceTable);CHKERRQ(ierr);
    /* There is a section point for every leaf attached to a given root point */
    for (r = 0, idx = 0, idx2 = 0; r < Nr; ++r) {
      PetscInt deg;

      for (deg = 0; deg < rootdegree[r]; ++deg, ++idx) {
        PetscInt offset, dof, d;

        ierr = PetscSectionGetDof(candidateRemoteSection, idx, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(candidateRemoteSection, idx, &offset);CHKERRQ(ierr);
        /* dof may include many faces from the remote process */
        for (d = 0; d < dof; ++d) {
          const PetscInt         hidx  = offset+d;
          const PetscInt         Np    = candidatesRemote[hidx].index+1;
          const PetscSFNode      rface = candidatesRemote[hidx+1];
          const PetscSFNode     *fcone = &candidatesRemote[hidx+2];
          PetscSFNode            fcp0;
          const PetscSFNode      pmax  = {PETSC_MAX_INT, PETSC_MAX_INT};
          const PetscInt        *join  = NULL;
          PetscHashIJKLRemoteKey key;
          PetscHashIter          iter;
          PetscBool              missing;
          PetscInt               points[1024], p, joinSize;

          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Checking face (%D, %D) at (%D, %D, %D) with cone size %D\n", rank, rface.rank, rface.index, r, idx, d, Np);CHKERRQ(ierr);}
          if (Np > 4) SETERRQ6(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle face (%D, %D) at (%D, %D, %D) with %D cone points", rface.rank, rface.index, r, idx, d, Np);
          fcp0.rank  = rank;
          fcp0.index = r;
          d += Np;
          /* Put remote face in hash table */
          key.i = fcp0;
          key.j = fcone[0];
          key.k = Np > 2 ? fcone[1] : pmax;
          key.l = Np > 3 ? fcone[2] : pmax;
          ierr = PetscSortSFNode(Np, (PetscSFNode *) &key);CHKERRQ(ierr);
          ierr = PetscHashIJKLRemotePut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
          if (missing) {
            if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Setting remote face (%D, %D)\n", rank, rface.index, rface.rank);CHKERRQ(ierr);}
            ierr = PetscHashIJKLRemoteIterSet(faceTable, iter, rface);CHKERRQ(ierr);
          } else {
            PetscSFNode oface;

            ierr = PetscHashIJKLRemoteIterGet(faceTable, iter, &oface);CHKERRQ(ierr);
            if ((rface.rank < oface.rank) || (rface.rank == oface.rank && rface.index < oface.index)) {
              if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Replacing with remote face (%D, %D)\n", rank, rface.index, rface.rank);CHKERRQ(ierr);}
              ierr = PetscHashIJKLRemoteIterSet(faceTable, iter, rface);CHKERRQ(ierr);
            }
          }
          /* Check for local face */
          points[0] = r;
          for (p = 1; p < Np; ++p) {
            ierr = DMPlexMapToLocalPoint(dm, remoteHash, fcone[p-1], &points[p]);
            if (ierr) break; /* We got a point not in our overlap */
            if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Checking local candidate %D\n", rank, points[p]);CHKERRQ(ierr);}
          }
          if (ierr) continue;
          ierr = DMPlexGetJoin(dm, Np, points, &joinSize, &join);CHKERRQ(ierr);
          if (joinSize == 1) {
            PetscSFNode lface;
            PetscSFNode oface;

            /* Always replace with local face */
            lface.rank  = rank;
            lface.index = join[0];
            ierr = PetscHashIJKLRemoteIterGet(faceTable, iter, &oface);CHKERRQ(ierr);
            if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Replacing (%D, %D) with local face (%D, %D)\n", rank, oface.index, oface.rank, lface.index, lface.rank);CHKERRQ(ierr);}
            ierr = PetscHashIJKLRemoteIterSet(faceTable, iter, lface);CHKERRQ(ierr);
          }
          ierr = DMPlexRestoreJoin(dm, Np, points, &joinSize, &join);CHKERRQ(ierr);
        }
      }
      /* Put back faces for this root */
      for (deg = 0; deg < rootdegree[r]; ++deg, ++idx2) {
        PetscInt offset, dof, d;

        ierr = PetscSectionGetDof(candidateRemoteSection, idx2, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(candidateRemoteSection, idx2, &offset);CHKERRQ(ierr);
        /* dof may include many faces from the remote process */
        for (d = 0; d < dof; ++d) {
          const PetscInt         hidx  = offset+d;
          const PetscInt         Np    = candidatesRemote[hidx].index+1;
          const PetscSFNode     *fcone = &candidatesRemote[hidx+2];
          PetscSFNode            fcp0;
          const PetscSFNode      pmax  = {PETSC_MAX_INT, PETSC_MAX_INT};
          PetscHashIJKLRemoteKey key;
          PetscHashIter          iter;
          PetscBool              missing;

          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Entering face at (%D, %D)\n", rank, r, idx);CHKERRQ(ierr);}
          if (Np > 4) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle faces with %D cone points", Np);
          fcp0.rank  = rank;
          fcp0.index = r;
          d += Np;
          /* Find remote face in hash table */
          key.i = fcp0;
          key.j = fcone[0];
          key.k = Np > 2 ? fcone[1] : pmax;
          key.l = Np > 3 ? fcone[2] : pmax;
          ierr = PetscSortSFNode(Np, (PetscSFNode *) &key);CHKERRQ(ierr);
          if (debug) {ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    key (%D, %D) (%D, %D) (%D, %D) (%D, %D)\n", rank, key.i.rank, key.i.index, key.j.rank, key.j.index, key.k.rank, key.k.index, key.l.rank, key.l.index);CHKERRQ(ierr);}
          ierr = PetscHashIJKLRemotePut(faceTable, key, &iter, &missing);CHKERRQ(ierr);
          if (missing) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Root %D Idx %D ought to have an assoicated face", r, idx2);
          else        {ierr = PetscHashIJKLRemoteIterGet(faceTable, iter, &candidatesRemote[hidx]);CHKERRQ(ierr);}
        }
      }
    }
    if (debug) {ierr = PetscSynchronizedFlush(PetscObjectComm((PetscObject) dm), NULL);CHKERRQ(ierr);}
    ierr = PetscHashIJKLRemoteDestroy(&faceTable);CHKERRQ(ierr);
  }
  /* Step 4: Push back owned faces */
  {
    PetscSF      sfMulti, sfClaims, sfPointNew;
    PetscSFNode *remotePointsNew;
    PetscInt    *remoteOffsets, *localPointsNew;
    PetscInt     pStart, pEnd, r, NlNew, p;

    /* 4) Push claims back to receiver via the MultiSF and derive new pointSF mapping on receiver */
    ierr = PetscSFGetMultiSF(pointSF, &sfMulti);CHKERRQ(ierr);
    ierr = PetscSectionCreate(comm, &claimSection);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfMulti, candidateRemoteSection, &remoteOffsets, claimSection);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sfMulti, candidateRemoteSection, remoteOffsets, claimSection, &sfClaims);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(claimSection, &claimsSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(claimsSize, &claims);CHKERRQ(ierr);
    for (p = 0; p < claimsSize; ++p) claims[p].rank = -1;
    ierr = PetscSFBcastBegin(sfClaims, MPIU_2INT, candidatesRemote, claims);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfClaims, MPIU_2INT, candidatesRemote, claims);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfClaims);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) claimSection, "Claim Section");CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) claimSection, NULL, "-petscsection_interp_claim_view");CHKERRQ(ierr);
    ierr = SFNodeArrayViewFromOptions(comm, "-petscsection_interp_claim_view", "Claims", NULL, claimsSize, claims);CHKERRQ(ierr);
    /* Step 5) Walk the original section of local supports and add an SF entry for each updated item */
    /* TODO I should not have to do a join here since I already put the face and its cone in the candidate section */
    ierr = PetscHMapICreate(&claimshash);CHKERRQ(ierr);
    for (r = 0; r < Nr; ++r) {
      PetscInt dof, off, d;

      if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]  Checking root for claims %D\n", rank, r);CHKERRQ(ierr);}
      ierr = PetscSectionGetDof(candidateSection, r, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(candidateSection, r, &off);CHKERRQ(ierr);
      for (d = 0; d < dof;) {
        if (claims[off+d].rank >= 0) {
          const PetscInt  faceInd = off+d;
          const PetscInt  Np      = candidates[off+d].index;
          const PetscInt *join    = NULL;
          PetscInt        joinSize, points[1024], c;

          if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Found claim for remote point (%D, %D)\n", rank, claims[faceInd].rank, claims[faceInd].index);CHKERRQ(ierr);}
          points[0] = r;
          if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]      point %D\n", rank, points[0]);CHKERRQ(ierr);}
          for (c = 0, d += 2; c < Np; ++c, ++d) {
            ierr = DMPlexMapToLocalPoint(dm, remoteHash, candidates[off+d], &points[c+1]);CHKERRQ(ierr);
            if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]      point %D\n", rank, points[c+1]);CHKERRQ(ierr);}
          }
          ierr = DMPlexGetJoin(dm, Np+1, points, &joinSize, &join);CHKERRQ(ierr);
          if (joinSize == 1) {
            if (claims[faceInd].rank == rank) {
              if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Ignoring local face %D for non-remote partner\n", rank, join[0]);CHKERRQ(ierr);}
            } else {
              if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Found local face %D\n", rank, join[0]);CHKERRQ(ierr);}
              ierr = PetscHMapISet(claimshash, join[0], faceInd);CHKERRQ(ierr);
            }
          } else {
            if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    Failed to find face\n", rank);CHKERRQ(ierr);}
          }
          ierr = DMPlexRestoreJoin(dm, Np+1, points, &joinSize, &join);CHKERRQ(ierr);
        } else {
          if (debug) {ierr = PetscSynchronizedPrintf(comm, "[%d]    No claim for point %D\n", rank, r);CHKERRQ(ierr);}
          d += claims[off+d].index+1;
        }
      }
    }
    if (debug) {ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);}
    /* Step 6) Create new pointSF from hashed claims */
    ierr = PetscHMapIGetSize(claimshash, &NlNew);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nl + NlNew, &localPointsNew);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nl + NlNew, &remotePointsNew);CHKERRQ(ierr);
    for (l = 0; l < Nl; ++l) {
      localPointsNew[l] = localPoints[l];
      remotePointsNew[l].index = remotePoints[l].index;
      remotePointsNew[l].rank  = remotePoints[l].rank;
    }
    p = Nl;
    ierr = PetscHMapIGetKeys(claimshash, &p, localPointsNew);CHKERRQ(ierr);
    /* We sort new points, and assume they are numbered after all existing points */
    ierr = PetscSortInt(NlNew, &localPointsNew[Nl]);CHKERRQ(ierr);
    for (p = Nl; p < Nl + NlNew; ++p) {
      PetscInt off;
      ierr = PetscHMapIGet(claimshash, localPointsNew[p], &off);CHKERRQ(ierr);
      if (claims[off].rank < 0 || claims[off].index < 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid claim for local point %D, (%D, %D)", localPointsNew[p], claims[off].rank, claims[off].index);
      remotePointsNew[p] = claims[off];
    }
    ierr = PetscSFCreate(comm, &sfPointNew);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfPointNew, pEnd-pStart, Nl+NlNew, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sfPointNew);CHKERRQ(ierr);
    ierr = DMSetPointSF(dm, sfPointNew);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) sfPointNew, NULL, "-petscsf_interp_view");CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfPointNew);CHKERRQ(ierr);
    ierr = PetscHMapIDestroy(&claimshash);CHKERRQ(ierr);
  }
  ierr = PetscHMapIJDestroy(&remoteHash);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&candidateSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&candidateRemoteSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&claimSection);CHKERRQ(ierr);
  ierr = PetscFree(candidates);CHKERRQ(ierr);
  ierr = PetscFree(candidatesRemote);CHKERRQ(ierr);
  ierr = PetscFree(claims);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_InterpolateSF,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexInterpolate - Take in a cell-vertex mesh and return one with all intermediate faces, edges, etc.

  Collective on dm

  Input Parameters:
+ dm - The DMPlex object with only cells and vertices
- dmInt - The interpolated DM

  Output Parameter:
. dmInt - The complete DMPlex object

  Level: intermediate

  Notes:
    It does not copy over the coordinates.

  Developer Notes:
    It sets plex->interpolated = DMPLEX_INTERPOLATED_FULL.

.seealso: DMPlexUninterpolate(), DMPlexCreateFromCellListPetsc(), DMPlexCopyCoordinates()
@*/
PetscErrorCode DMPlexInterpolate(DM dm, DM *dmInt)
{
  DMPlexInterpolatedFlag interpolated;
  DM             idm, odm = dm;
  PetscSF        sfPoint;
  PetscInt       depth, dim, d;
  const char    *name;
  PetscBool      flg=PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmInt, 2);
  ierr = PetscLogEventBegin(DMPLEX_Interpolate,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsInterpolated(dm, &interpolated);CHKERRQ(ierr);
  if (interpolated == DMPLEX_INTERPOLATED_PARTIAL) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated == DMPLEX_INTERPOLATED_FULL) {
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    idm  = dm;
  } else {
    for (d = 1; d < dim; ++d) {
      /* Create interpolated mesh */
      ierr = DMCreate(PetscObjectComm((PetscObject)dm), &idm);CHKERRQ(ierr);
      ierr = DMSetType(idm, DMPLEX);CHKERRQ(ierr);
      ierr = DMSetDimension(idm, dim);CHKERRQ(ierr);
      if (depth > 0) {
        ierr = DMPlexInterpolateFaces_Internal(odm, 1, idm);CHKERRQ(ierr);
        ierr = DMGetPointSF(odm, &sfPoint);CHKERRQ(ierr);
        {
          /* TODO: We need to systematically fix cases of distributed Plexes with no graph set */
          PetscInt nroots;
          ierr = PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
          if (nroots >= 0) {ierr = DMPlexInterpolatePointSF(idm, sfPoint);CHKERRQ(ierr);}
        }
      }
      if (odm != dm) {ierr = DMDestroy(&odm);CHKERRQ(ierr);}
      odm = idm;
    }
    ierr = PetscObjectGetName((PetscObject) dm,  &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) idm,  name);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
    ierr = DMCopyLabels(dm, idm, PETSC_COPY_VALUES, PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dm_plex_interpolate_orient_interfaces", &flg, NULL);CHKERRQ(ierr);
    if (flg) {ierr = DMPlexOrientInterface_Internal(idm);CHKERRQ(ierr);}
  }
  {
    PetscBool            isper;
    const PetscReal      *maxCell, *L;
    const DMBoundaryType *bd;

    ierr = DMGetPeriodicity(dm,&isper,&maxCell,&L,&bd);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(idm,isper,maxCell,L,bd);CHKERRQ(ierr);
  }
  /* This function makes the mesh fully interpolated on all ranks */
  {
    DM_Plex *plex = (DM_Plex *) idm->data;
    plex->interpolated = plex->interpolatedCollective = DMPLEX_INTERPOLATED_FULL;
  }
  *dmInt = idm;
  ierr = PetscLogEventEnd(DMPLEX_Interpolate,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCopyCoordinates - Copy coordinates from one mesh to another with the same vertices

  Collective on dmA

  Input Parameter:
. dmA - The DMPlex object with initial coordinates

  Output Parameter:
. dmB - The DMPlex object with copied coordinates

  Level: intermediate

  Note: This is typically used when adding pieces other than vertices to a mesh

.seealso: DMCopyLabels(), DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMGetCoordinateSection()
@*/
PetscErrorCode DMPlexCopyCoordinates(DM dmA, DM dmB)
{
  Vec            coordinatesA, coordinatesB;
  VecType        vtype;
  PetscSection   coordSectionA, coordSectionB;
  PetscScalar   *coordsA, *coordsB;
  PetscInt       spaceDim, Nf, vStartA, vStartB, vEndA, vEndB, coordSizeB, v, d;
  PetscInt       cStartA, cEndA, cStartB, cEndB, cS, cE, cdim;
  PetscBool      lc = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  if (dmA == dmB) PetscFunctionReturn(0);
  ierr = DMGetCoordinateDim(dmA, &cdim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dmB, cdim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dmA, 0, &vStartA, &vEndA);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dmB, 0, &vStartB, &vEndB);CHKERRQ(ierr);
  if ((vEndA-vStartA) != (vEndB-vStartB)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of vertices in first DM %d != %d in the second DM", vEndA-vStartA, vEndB-vStartB);
  ierr = DMPlexGetHeightStratum(dmA, 0, &cStartA, &cEndA);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmB, 0, &cStartB, &cEndB);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmA, &coordSectionA);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmB, &coordSectionB);CHKERRQ(ierr);
  if (coordSectionA == coordSectionB) PetscFunctionReturn(0);
  ierr = PetscSectionGetNumFields(coordSectionA, &Nf);CHKERRQ(ierr);
  if (!Nf) PetscFunctionReturn(0);
  if (Nf > 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of coordinate fields must be 1, not %D", Nf);
  if (!coordSectionB) {
    PetscInt dim;

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) coordSectionA), &coordSectionB);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dmA, &dim);CHKERRQ(ierr);
    ierr = DMSetCoordinateSection(dmB, dim, coordSectionB);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject) coordSectionB);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetNumFields(coordSectionB, 1);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(coordSectionA, 0, &spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSectionB, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSectionA, &cS, &cE);CHKERRQ(ierr);
  if (cStartA <= cS && cS < cEndA) { /* localized coordinates */
    if ((cEndA-cStartA) != (cEndB-cStartB)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of cells in first DM %D != %D in the second DM", cEndA-cStartA, cEndB-cStartB);
    cS = cS - cStartA + cStartB;
    cE = vEndB;
    lc = PETSC_TRUE;
  } else {
    cS = vStartB;
    cE = vEndB;
  }
  ierr = PetscSectionSetChart(coordSectionB, cS, cE);CHKERRQ(ierr);
  for (v = vStartB; v < vEndB; ++v) {
    ierr = PetscSectionSetDof(coordSectionB, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSectionB, v, 0, spaceDim);CHKERRQ(ierr);
  }
  if (lc) { /* localized coordinates */
    PetscInt c;

    for (c = cS-cStartB; c < cEndB-cStartB; c++) {
      PetscInt dof;

      ierr = PetscSectionGetDof(coordSectionA, c + cStartA, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(coordSectionB, c + cStartB, dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(coordSectionB, c + cStartB, 0, dof);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(coordSectionB);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSectionB, &coordSizeB);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmA, &coordinatesA);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinatesB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinatesB, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinatesB, coordSizeB, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinatesA, &d);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinatesB, d);CHKERRQ(ierr);
  ierr = VecGetType(coordinatesA, &vtype);CHKERRQ(ierr);
  ierr = VecSetType(coordinatesB, vtype);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesA, &coordsA);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesB, &coordsB);CHKERRQ(ierr);
  for (v = 0; v < vEndB-vStartB; ++v) {
    PetscInt offA, offB;

    ierr = PetscSectionGetOffset(coordSectionA, v + vStartA, &offA);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(coordSectionB, v + vStartB, &offB);CHKERRQ(ierr);
    for (d = 0; d < spaceDim; ++d) {
      coordsB[offB+d] = coordsA[offA+d];
    }
  }
  if (lc) { /* localized coordinates */
    PetscInt c;

    for (c = cS-cStartB; c < cEndB-cStartB; c++) {
      PetscInt dof, offA, offB;

      ierr = PetscSectionGetOffset(coordSectionA, c + cStartA, &offA);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSectionB, c + cStartB, &offB);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(coordSectionA, c + cStartA, &dof);CHKERRQ(ierr);
      ierr = PetscArraycpy(coordsB + offB,coordsA + offA,dof);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(coordinatesA, &coordsA);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinatesB, &coordsB);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmB, coordinatesB);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinatesB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexUninterpolate - Take in a mesh with all intermediate faces, edges, etc. and return a cell-vertex mesh

  Collective on dm

  Input Parameter:
. dm - The complete DMPlex object

  Output Parameter:
. dmUnint - The DMPlex object with only cells and vertices

  Level: intermediate

  Notes:
    It does not copy over the coordinates.

  Developer Notes:
    It sets plex->interpolated = DMPLEX_INTERPOLATED_NONE.

.seealso: DMPlexInterpolate(), DMPlexCreateFromCellListPetsc(), DMPlexCopyCoordinates()
@*/
PetscErrorCode DMPlexUninterpolate(DM dm, DM *dmUnint)
{
  DMPlexInterpolatedFlag interpolated;
  DM             udm;
  PetscInt       dim, vStart, vEnd, cStart, cEnd, c, maxConeSize = 0, *cone;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmUnint, 2);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsInterpolated(dm, &interpolated);CHKERRQ(ierr);
  if (interpolated == DMPLEX_INTERPOLATED_PARTIAL) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated == DMPLEX_INTERPOLATED_NONE || dim <= 1) {
    /* in case dim <= 1 just keep the DMPLEX_INTERPOLATED_FULL flag */
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    *dmUnint = dm;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &udm);CHKERRQ(ierr);
  ierr = DMSetType(udm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(udm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(udm, cStart, vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) ++coneSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(udm, c, coneSize);CHKERRQ(ierr);
    maxConeSize = PetscMax(maxConeSize, coneSize);
  }
  ierr = DMSetUp(udm);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxConeSize, &cone);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) cone[coneSize++] = p;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMPlexSetCone(udm, c, cone);CHKERRQ(ierr);
  }
  ierr = PetscFree(cone);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(udm);CHKERRQ(ierr);
  ierr = DMPlexStratify(udm);CHKERRQ(ierr);
  /* Reduce SF */
  {
    PetscSF            sfPoint, sfPointUn;
    const PetscSFNode *remotePoints;
    const PetscInt    *localPoints;
    PetscSFNode       *remotePointsUn;
    PetscInt          *localPointsUn;
    PetscInt           vEnd, numRoots, numLeaves, l;
    PetscInt           numLeavesUn = 0, n = 0;
    PetscErrorCode     ierr;

    /* Get original SF information */
    ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
    ierr = DMGetPointSF(udm, &sfPointUn);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, NULL, &vEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
    /* Allocate space for cells and vertices */
    for (l = 0; l < numLeaves; ++l) if (localPoints[l] < vEnd) numLeavesUn++;
    /* Fill in leaves */
    if (vEnd >= 0) {
      ierr = PetscMalloc1(numLeavesUn, &remotePointsUn);CHKERRQ(ierr);
      ierr = PetscMalloc1(numLeavesUn, &localPointsUn);CHKERRQ(ierr);
      for (l = 0; l < numLeaves; l++) {
        if (localPoints[l] < vEnd) {
          localPointsUn[n]        = localPoints[l];
          remotePointsUn[n].rank  = remotePoints[l].rank;
          remotePointsUn[n].index = remotePoints[l].index;
          ++n;
        }
      }
      if (n != numLeavesUn) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent number of leaves %d != %d", n, numLeavesUn);
      ierr = PetscSFSetGraph(sfPointUn, vEnd, numLeavesUn, localPointsUn, PETSC_OWN_POINTER, remotePointsUn, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
  }
  {
    PetscBool            isper;
    const PetscReal      *maxCell, *L;
    const DMBoundaryType *bd;

    ierr = DMGetPeriodicity(dm,&isper,&maxCell,&L,&bd);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(udm,isper,maxCell,L,bd);CHKERRQ(ierr);
  }
  /* This function makes the mesh fully uninterpolated on all ranks */
  {
    DM_Plex *plex = (DM_Plex *) udm->data;
    plex->interpolated = plex->interpolatedCollective = DMPLEX_INTERPOLATED_NONE;
  }
  *dmUnint = udm;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexIsInterpolated_Internal(DM dm, DMPlexInterpolatedFlag *interpolated)
{
  PetscInt       coneSize, depth, dim, h, p, pStart, pEnd;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  if (depth == dim) {
    *interpolated = DMPLEX_INTERPOLATED_FULL;
    if (!dim) goto finish;

    /* Check points at height = dim are vertices (have no cones) */
    ierr = DMPlexGetHeightStratum(dm, dim, &pStart, &pEnd);CHKERRQ(ierr);
    for (p=pStart; p<pEnd; p++) {
      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      if (coneSize) {
        *interpolated = DMPLEX_INTERPOLATED_PARTIAL;
        goto finish;
      }
    }

    /* Check points at height < dim have cones */
    for (h=0; h<dim; h++) {
      ierr = DMPlexGetHeightStratum(dm, h, &pStart, &pEnd);CHKERRQ(ierr);
      for (p=pStart; p<pEnd; p++) {
        ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
        if (!coneSize) {
          *interpolated = DMPLEX_INTERPOLATED_PARTIAL;
          goto finish;
        }
      }
    }
  } else if (depth == 1) {
    *interpolated = DMPLEX_INTERPOLATED_NONE;
  } else {
    *interpolated = DMPLEX_INTERPOLATED_PARTIAL;
  }
finish:
  PetscFunctionReturn(0);
}

/*@
  DMPlexIsInterpolated - Find out to what extent the DMPlex is topologically interpolated.

  Not Collective

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. interpolated - Flag whether the DM is interpolated

  Level: intermediate

  Notes:
  Unlike DMPlexIsInterpolatedCollective(), this is NOT collective
  so the results can be different on different ranks in special cases.
  However, DMPlexInterpolate() guarantees the result is the same on all.

  Unlike DMPlexIsInterpolatedCollective(), this cannot return DMPLEX_INTERPOLATED_MIXED.

  Developer Notes:
  Initially, plex->interpolated = DMPLEX_INTERPOLATED_INVALID.

  If plex->interpolated == DMPLEX_INTERPOLATED_INVALID, DMPlexIsInterpolated_Internal() is called.
  It checks the actual topology and sets plex->interpolated on each rank separately to one of
  DMPLEX_INTERPOLATED_NONE, DMPLEX_INTERPOLATED_PARTIAL or DMPLEX_INTERPOLATED_FULL.

  If plex->interpolated != DMPLEX_INTERPOLATED_INVALID, this function just returns plex->interpolated.

  DMPlexInterpolate() sets plex->interpolated = DMPLEX_INTERPOLATED_FULL,
  and DMPlexUninterpolate() sets plex->interpolated = DMPLEX_INTERPOLATED_NONE.

.seealso: DMPlexInterpolate(), DMPlexIsInterpolatedCollective()
@*/
PetscErrorCode DMPlexIsInterpolated(DM dm, DMPlexInterpolatedFlag *interpolated)
{
  DM_Plex        *plex = (DM_Plex *) dm->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(interpolated,2);
  if (plex->interpolated < 0) {
    ierr = DMPlexIsInterpolated_Internal(dm, &plex->interpolated);CHKERRQ(ierr);
  } else if (PetscDefined (USE_DEBUG)) {
    DMPlexInterpolatedFlag flg;

    ierr = DMPlexIsInterpolated_Internal(dm, &flg);CHKERRQ(ierr);
    if (flg != plex->interpolated) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stashed DMPlexInterpolatedFlag %s is inconsistent with current %s", DMPlexInterpolatedFlags[plex->interpolated], DMPlexInterpolatedFlags[flg]);
  }
  *interpolated = plex->interpolated;
  PetscFunctionReturn(0);
}

/*@
  DMPlexIsInterpolatedCollective - Find out to what extent the DMPlex is topologically interpolated (in collective manner).

  Collective

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. interpolated - Flag whether the DM is interpolated

  Level: intermediate

  Notes:
  Unlike DMPlexIsInterpolated(), this is collective so the results are guaranteed to be the same on all ranks.

  This function will return DMPLEX_INTERPOLATED_MIXED if the results of DMPlexIsInterpolated() are different on different ranks.

  Developer Notes:
  Initially, plex->interpolatedCollective = DMPLEX_INTERPOLATED_INVALID.

  If plex->interpolatedCollective == DMPLEX_INTERPOLATED_INVALID, this function calls DMPlexIsInterpolated() which sets plex->interpolated.
  MPI_Allreduce() is then called and collectively consistent flag plex->interpolatedCollective is set and returned;
  if plex->interpolated varies on different ranks, plex->interpolatedCollective = DMPLEX_INTERPOLATED_MIXED,
  otherwise sets plex->interpolatedCollective = plex->interpolated.

  If plex->interpolatedCollective != DMPLEX_INTERPOLATED_INVALID, this function just returns plex->interpolatedCollective.

.seealso: DMPlexInterpolate(), DMPlexIsInterpolated()
@*/
PetscErrorCode DMPlexIsInterpolatedCollective(DM dm, DMPlexInterpolatedFlag *interpolated)
{
  DM_Plex        *plex = (DM_Plex *) dm->data;
  PetscBool       debug=PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(interpolated,2);
  ierr = PetscOptionsGetBool(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_plex_is_interpolated_collective_debug", &debug, NULL);CHKERRQ(ierr);
  if (plex->interpolatedCollective < 0) {
    DMPlexInterpolatedFlag  min, max;
    MPI_Comm                comm;

    ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
    ierr = DMPlexIsInterpolated(dm, &plex->interpolatedCollective);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&plex->interpolatedCollective, &min, 1, MPIU_ENUM, MPI_MIN, comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&plex->interpolatedCollective, &max, 1, MPIU_ENUM, MPI_MAX, comm);CHKERRQ(ierr);
    if (min != max) plex->interpolatedCollective = DMPLEX_INTERPOLATED_MIXED;
    if (debug) {
      PetscMPIInt rank;

      ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "[%d] interpolated=%s interpolatedCollective=%s\n", rank, DMPlexInterpolatedFlags[plex->interpolated], DMPlexInterpolatedFlags[plex->interpolatedCollective]);CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(comm, PETSC_STDOUT);CHKERRQ(ierr);
    }
  }
  *interpolated = plex->interpolatedCollective;
  PetscFunctionReturn(0);
}
