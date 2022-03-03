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

PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PETSC_HASH_MAP(HashIJKL, PetscHashIJKLKey, PetscInt, PetscHashIJKLKeyHash, PetscHashIJKLKeyEqual, -1))

static PetscSFNode _PetscInvalidSFNode = {-1, -1};

typedef struct _PetscHashIJKLRemoteKey { PetscSFNode i, j, k, l; } PetscHashIJKLRemoteKey;

#define PetscHashIJKLRemoteKeyHash(key) \
  PetscHashCombine(PetscHashCombine(PetscHashInt((key).i.rank + (key).i.index),PetscHashInt((key).j.rank + (key).j.index)), \
                   PetscHashCombine(PetscHashInt((key).k.rank + (key).k.index),PetscHashInt((key).l.rank + (key).l.index)))

#define PetscHashIJKLRemoteKeyEqual(k1,k2) \
  (((k1).i.rank==(k2).i.rank) ? ((k1).i.index==(k2).i.index) ? ((k1).j.rank==(k2).j.rank) ? ((k1).j.index==(k2).j.index) ? ((k1).k.rank==(k2).k.rank) ? ((k1).k.index==(k2).k.index) ? ((k1).l.rank==(k2).l.rank) ? ((k1).l.index==(k2).l.index) : 0 : 0 : 0 : 0 : 0 : 0 : 0)

PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PETSC_HASH_MAP(HashIJKLRemote, PetscHashIJKLRemoteKey, PetscSFNode, PetscHashIJKLRemoteKeyHash, PetscHashIJKLRemoteKeyEqual, _PetscInvalidSFNode))

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cone) PetscValidIntPointer(cone, 3);
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
  if (faceTypes) CHKERRQ(DMGetWorkArray(dm, PetscMax(maxConeSize, maxSupportSize),           MPIU_INT, &typesTmp));
  if (faceSizes) CHKERRQ(DMGetWorkArray(dm, PetscMax(maxConeSize, maxSupportSize),           MPIU_INT, &sizesTmp));
  if (faces)     CHKERRQ(DMGetWorkArray(dm, PetscSqr(PetscMax(maxConeSize, maxSupportSize)), MPIU_INT, &facesTmp));
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
    case DM_POLYTOPE_PYRAMID:
      /*
       4----
       |\-\ \-----
       | \ -\     \
       |  1--\-----2
       | /    \   /
       |/      \ /
       0--------3
       */
      if (numFaces) *numFaces = 5;
      if (faceTypes) {
        typesTmp[0] = DM_POLYTOPE_QUADRILATERAL;
        typesTmp[1] = DM_POLYTOPE_TRIANGLE; typesTmp[2] = DM_POLYTOPE_TRIANGLE; typesTmp[3] = DM_POLYTOPE_TRIANGLE; typesTmp[4] = DM_POLYTOPE_TRIANGLE;
        *faceTypes = typesTmp;
      }
      if (faceSizes) {
        sizesTmp[0] = 4;
        sizesTmp[1] = 3; sizesTmp[2] = 3; sizesTmp[3] = 3; sizesTmp[4] = 3;
        *faceSizes = sizesTmp;
      }
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2]; facesTmp[3]  = cone[3]; /* Bottom */
        facesTmp[4]  = cone[0]; facesTmp[5]  = cone[3]; facesTmp[6]  = cone[4];                         /* Front */
        facesTmp[7]  = cone[3]; facesTmp[8]  = cone[2]; facesTmp[9]  = cone[4];                         /* Right */
        facesTmp[10] = cone[2]; facesTmp[11] = cone[1]; facesTmp[12] = cone[4];                         /* Back */
        facesTmp[13] = cone[1]; facesTmp[14] = cone[0]; facesTmp[15] = cone[4];                         /* Left */
        *faces = facesTmp;
      }
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No face description for cell type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexRestoreRawFaces_Internal(DM dm, DMPolytopeType ct, const PetscInt cone[], PetscInt *numFaces, const DMPolytopeType *faceTypes[], const PetscInt *faceSizes[], const PetscInt *faces[])
{
  PetscFunctionBegin;
  if (faceTypes) CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_INT, (void *) faceTypes));
  if (faceSizes) CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_INT, (void *) faceSizes));
  if (faces)     CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_INT, (void *) faces));
  PetscFunctionReturn(0);
}

/* This interpolates faces for cells at some stratum */
static PetscErrorCode DMPlexInterpolateFaces_Internal(DM dm, PetscInt cellDepth, DM idm)
{
  DMLabel        ctLabel;
  PetscHashIJKL  faceTable;
  PetscInt       faceTypeNum[DM_NUM_POLYTOPES];
  PetscInt       depth, d, pStart, Np, cStart, cEnd, c, fStart, fEnd;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(PetscHashIJKLCreate(&faceTable));
  CHKERRQ(PetscArrayzero(faceTypeNum, DM_NUM_POLYTOPES));
  CHKERRQ(DMPlexGetDepthStratum(dm, cellDepth, &cStart, &cEnd));
  /* Number new faces and save face vertices in hash table */
  CHKERRQ(DMPlexGetDepthStratum(dm, depth > cellDepth ? cellDepth : 0, NULL, &fStart));
  fEnd = fStart;
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt       *cone, *faceSizes, *faces;
    const DMPolytopeType *faceTypes;
    DMPolytopeType        ct;
    PetscInt              numFaces, cf, foff = 0;

    CHKERRQ(DMPlexGetCellType(dm, c, &ct));
    CHKERRQ(DMPlexGetCone(dm, c, &cone));
    CHKERRQ(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
    for (cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
      const PetscInt       faceSize = faceSizes[cf];
      const DMPolytopeType faceType = faceTypes[cf];
      const PetscInt      *face     = &faces[foff];
      PetscHashIJKLKey     key;
      PetscHashIter        iter;
      PetscBool            missing;

      PetscCheck(faceSize <= 4,PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %" PetscInt_FMT " > 4", faceSize);
      key.i = face[0];
      key.j = faceSize > 1 ? face[1] : PETSC_MAX_INT;
      key.k = faceSize > 2 ? face[2] : PETSC_MAX_INT;
      key.l = faceSize > 3 ? face[3] : PETSC_MAX_INT;
      CHKERRQ(PetscSortInt(faceSize, (PetscInt *) &key));
      CHKERRQ(PetscHashIJKLPut(faceTable, key, &iter, &missing));
      if (missing) {
        CHKERRQ(PetscHashIJKLIterSet(faceTable, iter, fEnd++));
        ++faceTypeNum[faceType];
      }
    }
    CHKERRQ(DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
  }
  /* We need to number faces contiguously among types */
  {
    PetscInt faceTypeStart[DM_NUM_POLYTOPES], ct, numFT = 0;

    for (ct = 0; ct < DM_NUM_POLYTOPES; ++ct) {if (faceTypeNum[ct]) ++numFT; faceTypeStart[ct] = 0;}
    if (numFT > 1) {
      CHKERRQ(PetscHashIJKLClear(faceTable));
      faceTypeStart[0] = fStart;
      for (ct = 1; ct < DM_NUM_POLYTOPES; ++ct) faceTypeStart[ct] = faceTypeStart[ct-1] + faceTypeNum[ct-1];
      for (c = cStart; c < cEnd; ++c) {
        const PetscInt       *cone, *faceSizes, *faces;
        const DMPolytopeType *faceTypes;
        DMPolytopeType        ct;
        PetscInt              numFaces, cf, foff = 0;

        CHKERRQ(DMPlexGetCellType(dm, c, &ct));
        CHKERRQ(DMPlexGetCone(dm, c, &cone));
        CHKERRQ(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
        for (cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
          const PetscInt       faceSize = faceSizes[cf];
          const DMPolytopeType faceType = faceTypes[cf];
          const PetscInt      *face     = &faces[foff];
          PetscHashIJKLKey     key;
          PetscHashIter        iter;
          PetscBool            missing;

          PetscCheck(faceSize <= 4,PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %" PetscInt_FMT " > 4", faceSize);
          key.i = face[0];
          key.j = faceSize > 1 ? face[1] : PETSC_MAX_INT;
          key.k = faceSize > 2 ? face[2] : PETSC_MAX_INT;
          key.l = faceSize > 3 ? face[3] : PETSC_MAX_INT;
          CHKERRQ(PetscSortInt(faceSize, (PetscInt *) &key));
          CHKERRQ(PetscHashIJKLPut(faceTable, key, &iter, &missing));
          if (missing) CHKERRQ(PetscHashIJKLIterSet(faceTable, iter, faceTypeStart[faceType]++));
        }
        CHKERRQ(DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
      }
      for (ct = 1; ct < DM_NUM_POLYTOPES; ++ct) {
        PetscCheckFalse(faceTypeStart[ct] != faceTypeStart[ct-1] + faceTypeNum[ct],PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent numbering for cell type %s, %D != %D + %D", DMPolytopeTypes[ct], faceTypeStart[ct], faceTypeStart[ct-1], faceTypeNum[ct]);
      }
    }
  }
  /* Add new points, always at the end of the numbering */
  CHKERRQ(DMPlexGetChart(dm, &pStart, &Np));
  CHKERRQ(DMPlexSetChart(idm, pStart, Np + (fEnd - fStart)));
  /* Set cone sizes */
  /*   Must create the celltype label here so that we do not automatically try to compute the types */
  CHKERRQ(DMCreateLabel(idm, "celltype"));
  CHKERRQ(DMPlexGetCellTypeLabel(idm, &ctLabel));
  for (d = 0; d <= depth; ++d) {
    DMPolytopeType ct;
    PetscInt       coneSize, pStart, pEnd, p;

    if (d == cellDepth) continue;
    CHKERRQ(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
      CHKERRQ(DMPlexSetConeSize(idm, p, coneSize));
      CHKERRQ(DMPlexGetCellType(dm, p, &ct));
      CHKERRQ(DMPlexSetCellType(idm, p, ct));
    }
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt       *cone, *faceSizes, *faces;
    const DMPolytopeType *faceTypes;
    DMPolytopeType        ct;
    PetscInt              numFaces, cf, foff = 0;

    CHKERRQ(DMPlexGetCellType(dm, c, &ct));
    CHKERRQ(DMPlexGetCone(dm, c, &cone));
    CHKERRQ(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
    CHKERRQ(DMPlexSetCellType(idm, c, ct));
    CHKERRQ(DMPlexSetConeSize(idm, c, numFaces));
    for (cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
      const PetscInt       faceSize = faceSizes[cf];
      const DMPolytopeType faceType = faceTypes[cf];
      const PetscInt      *face     = &faces[foff];
      PetscHashIJKLKey     key;
      PetscHashIter        iter;
      PetscBool            missing;
      PetscInt             f;

      PetscCheckFalse(faceSize > 4,PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %D > 4", faceSize);
      key.i = face[0];
      key.j = faceSize > 1 ? face[1] : PETSC_MAX_INT;
      key.k = faceSize > 2 ? face[2] : PETSC_MAX_INT;
      key.l = faceSize > 3 ? face[3] : PETSC_MAX_INT;
      CHKERRQ(PetscSortInt(faceSize, (PetscInt *) &key));
      CHKERRQ(PetscHashIJKLPut(faceTable, key, &iter, &missing));
      PetscCheck(!missing,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing face (cell %D, lf %D)", c, cf);
      CHKERRQ(PetscHashIJKLIterGet(faceTable, iter, &f));
      CHKERRQ(DMPlexSetConeSize(idm, f, faceSize));
      CHKERRQ(DMPlexSetCellType(idm, f, faceType));
    }
    CHKERRQ(DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
  }
  CHKERRQ(DMSetUp(idm));
  /* Initialize cones so we do not need the bash table to tell us that a cone has been set */
  {
    PetscSection cs;
    PetscInt    *cones, csize;

    CHKERRQ(DMPlexGetConeSection(idm, &cs));
    CHKERRQ(DMPlexGetCones(idm, &cones));
    CHKERRQ(PetscSectionGetStorageSize(cs, &csize));
    for (c = 0; c < csize; ++c) cones[c] = -1;
  }
  /* Set cones */
  for (d = 0; d <= depth; ++d) {
    const PetscInt *cone;
    PetscInt        pStart, pEnd, p;

    if (d == cellDepth) continue;
    CHKERRQ(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      CHKERRQ(DMPlexGetCone(dm, p, &cone));
      CHKERRQ(DMPlexSetCone(idm, p, cone));
      CHKERRQ(DMPlexGetConeOrientation(dm, p, &cone));
      CHKERRQ(DMPlexSetConeOrientation(idm, p, cone));
    }
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt       *cone, *faceSizes, *faces;
    const DMPolytopeType *faceTypes;
    DMPolytopeType        ct;
    PetscInt              numFaces, cf, foff = 0;

    CHKERRQ(DMPlexGetCellType(dm, c, &ct));
    CHKERRQ(DMPlexGetCone(dm, c, &cone));
    CHKERRQ(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
    for (cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
      DMPolytopeType   faceType = faceTypes[cf];
      const PetscInt   faceSize = faceSizes[cf];
      const PetscInt  *face     = &faces[foff];
      const PetscInt  *fcone;
      PetscHashIJKLKey key;
      PetscHashIter    iter;
      PetscBool        missing;
      PetscInt         f;

      PetscCheckFalse(faceSize > 4,PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %D > 4", faceSize);
      key.i = face[0];
      key.j = faceSize > 1 ? face[1] : PETSC_MAX_INT;
      key.k = faceSize > 2 ? face[2] : PETSC_MAX_INT;
      key.l = faceSize > 3 ? face[3] : PETSC_MAX_INT;
      CHKERRQ(PetscSortInt(faceSize, (PetscInt *) &key));
      CHKERRQ(PetscHashIJKLPut(faceTable, key, &iter, &missing));
      CHKERRQ(PetscHashIJKLIterGet(faceTable, iter, &f));
      CHKERRQ(DMPlexInsertCone(idm, c, cf, f));
      CHKERRQ(DMPlexGetCone(idm, f, &fcone));
      if (fcone[0] < 0) CHKERRQ(DMPlexSetCone(idm, f, face));
      {
        const PetscInt *cone;
        PetscInt        coneSize, ornt;

        CHKERRQ(DMPlexGetConeSize(idm, f, &coneSize));
        CHKERRQ(DMPlexGetCone(idm, f, &cone));
        PetscCheckFalse(coneSize != faceSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of face vertices %D for face %D should be %D", coneSize, f, faceSize);
        /* Notice that we have to use vertices here because the lower dimensional faces have not been created yet */
        CHKERRQ(DMPolytopeGetVertexOrientation(faceType, cone, face, &ornt));
        CHKERRQ(DMPlexInsertConeOrientation(idm, c, cf, ornt));
      }
    }
    CHKERRQ(DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
  }
  CHKERRQ(PetscHashIJKLDestroy(&faceTable));
  CHKERRQ(DMPlexSymmetrize(idm));
  CHKERRQ(DMPlexStratify(idm));
  PetscFunctionReturn(0);
}

static PetscErrorCode SortRmineRremoteByRemote_Private(PetscSF sf, PetscInt *rmine1[], PetscInt *rremote1[])
{
  PetscInt            nleaves;
  PetscInt            nranks;
  const PetscMPIInt  *ranks=NULL;
  const PetscInt     *roffset=NULL, *rmine=NULL, *rremote=NULL;
  PetscInt            n, o, r;

  PetscFunctionBegin;
  CHKERRQ(PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote));
  nleaves = roffset[nranks];
  CHKERRQ(PetscMalloc2(nleaves, rmine1, nleaves, rremote1));
  for (r=0; r<nranks; r++) {
    /* simultaneously sort rank-wise portions of rmine & rremote by values in rremote
       - to unify order with the other side */
    o = roffset[r];
    n = roffset[r+1] - o;
    CHKERRQ(PetscArraycpy(&(*rmine1)[o], &rmine[o], n));
    CHKERRQ(PetscArraycpy(&(*rremote1)[o], &rremote[o], n));
    CHKERRQ(PetscSortIntWithArray(n, &(*rremote1)[o], &(*rmine1)[o]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexOrientInterface_Internal(DM dm)
{
  PetscSF            sf;
  const PetscInt    *locals;
  const PetscSFNode *remotes;
  const PetscMPIInt *ranks;
  const PetscInt    *roffset;
  PetscInt          *rmine1, *rremote1; /* rmine and rremote copies simultaneously sorted by rank and rremote */
  PetscInt           nroots, p, nleaves, nranks, r, maxConeSize = 0;
  PetscInt         (*roots)[4],      (*leaves)[4], mainCone[4];
  PetscMPIInt      (*rootsRanks)[4], (*leavesRanks)[4];
  MPI_Comm           comm;
  PetscMPIInt        rank, size;
  PetscInt           debug = 0;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-before_orient_interface_dm_view"));
  if (PetscDefined(USE_DEBUG)) CHKERRQ(DMPlexCheckPointSF(dm));
  CHKERRQ(PetscSFGetGraph(sf, &nroots, &nleaves, &locals, &remotes));
  if (nroots < 0) PetscFunctionReturn(0);
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRQ(SortRmineRremoteByRemote_Private(sf, &rmine1, &rremote1));
  for (p = 0; p < nleaves; ++p) {
    PetscInt coneSize;
    CHKERRQ(DMPlexGetConeSize(dm, locals[p], &coneSize));
    maxConeSize = PetscMax(maxConeSize, coneSize);
  }
  PetscCheckFalse(maxConeSize > 4,comm, PETSC_ERR_SUP, "This method does not support cones of size %D", maxConeSize);
  CHKERRQ(PetscMalloc4(nroots, &roots, nroots, &leaves, nroots, &rootsRanks, nroots, &leavesRanks));
  for (p = 0; p < nroots; ++p) {
    const PetscInt *cone;
    PetscInt        coneSize, c, ind0;

    CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
    CHKERRQ(DMPlexGetCone(dm, p, &cone));
    /* Ignore vertices */
    if (coneSize < 2) {
      for (c = 0; c < 4; c++) {
        roots[p][c]      = -1;
        rootsRanks[p][c] = -1;
      }
      continue;
    }
    /* Translate all points to root numbering */
    for (c = 0; c < PetscMin(coneSize, 4); c++) {
      CHKERRQ(PetscFindInt(cone[c], nleaves, locals, &ind0));
      if (ind0 < 0) {
        roots[p][c]      = cone[c];
        rootsRanks[p][c] = rank;
      } else {
        roots[p][c]      = remotes[ind0].index;
        rootsRanks[p][c] = remotes[ind0].rank;
      }
    }
    for (c = coneSize; c < 4; c++) {
      roots[p][c]      = -1;
      rootsRanks[p][c] = -1;
    }
  }
  for (p = 0; p < nroots; ++p) {
    PetscInt c;
    for (c = 0; c < 4; c++) {
      leaves[p][c]      = -2;
      leavesRanks[p][c] = -2;
    }
  }
  CHKERRQ(PetscSFBcastBegin(sf, MPIU_4INT, roots, leaves, MPI_REPLACE));
  CHKERRQ(PetscSFBcastBegin(sf, MPI_4INT, rootsRanks, leavesRanks, MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf, MPIU_4INT, roots, leaves, MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf, MPI_4INT, rootsRanks, leavesRanks, MPI_REPLACE));
  if (debug) {
    CHKERRQ(PetscSynchronizedFlush(comm, NULL));
    if (!rank) CHKERRQ(PetscSynchronizedPrintf(comm, "Referenced roots\n"));
  }
  CHKERRQ(PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, NULL, NULL));
  for (p = 0; p < nroots; ++p) {
    DMPolytopeType  ct;
    const PetscInt *cone;
    PetscInt        coneSize, c, ind0, o;

    if (leaves[p][0] < 0) continue; /* Ignore vertices */
    CHKERRQ(DMPlexGetCellType(dm, p, &ct));
    CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
    CHKERRQ(DMPlexGetCone(dm, p, &cone));
    if (debug) {
      CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]  %4D: cone=[%4D %4D %4D %4D] roots=[(%d,%4D) (%d,%4D) (%d,%4D) (%d,%4D)] leaves=[(%d,%4D) (%d,%4D) (%d,%4D) (%d,%4D)]",
                                      rank, p, cone[0], cone[1], cone[2], cone[3],
                                      rootsRanks[p][0], roots[p][0], rootsRanks[p][1], roots[p][1], rootsRanks[p][2], roots[p][2], rootsRanks[p][3], roots[p][3],
                                      leavesRanks[p][0], leaves[p][0], leavesRanks[p][1], leaves[p][1], leavesRanks[p][2], leaves[p][2], leavesRanks[p][3], leaves[p][3]));
    }
    if (leavesRanks[p][0] != rootsRanks[p][0] || leaves[p][0] != roots[p][0] ||
        leavesRanks[p][1] != rootsRanks[p][1] || leaves[p][1] != roots[p][1] ||
        leavesRanks[p][2] != rootsRanks[p][2] || leaves[p][2] != roots[p][2] ||
        leavesRanks[p][3] != rootsRanks[p][3] || leaves[p][3] != roots[p][3]) {
      /* Translate these leaves to my cone points; mainCone means desired order p's cone points */
      for (c = 0; c < PetscMin(coneSize, 4); ++c) {
        PetscInt rS, rN;

        if (leavesRanks[p][c] == rank) {
          /* A local leaf is just taken as it is */
          mainCone[c] = leaves[p][c];
          continue;
        }
        /* Find index of rank leavesRanks[p][c] among remote ranks */
        /* No need for PetscMPIIntCast because these integers were originally cast from PetscMPIInt. */
        CHKERRQ(PetscFindMPIInt((PetscMPIInt) leavesRanks[p][c], nranks, ranks, &r));
        PetscCheckFalse(r < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D cone[%D]=%D root (%d,%D) leaf (%d,%D): leaf rank not found among remote ranks", p, c, cone[c], rootsRanks[p][c], roots[p][c], leavesRanks[p][c], leaves[p][c]);
        PetscCheckFalse(ranks[r] < 0 || ranks[r] >= size,PETSC_COMM_SELF, PETSC_ERR_PLIB, "p=%D c=%D commsize=%d: ranks[%D] = %d makes no sense", p, c, size, r, ranks[r]);
        /* Find point leaves[p][c] among remote points aimed at rank leavesRanks[p][c] */
        rS = roffset[r];
        rN = roffset[r+1] - rS;
        CHKERRQ(PetscFindInt(leaves[p][c], rN, &rremote1[rS], &ind0));
        PetscCheckFalse(ind0 < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D cone[%D]=%D root (%d,%D) leave (%d,%D): corresponding remote point not found - it seems there is missing connection in point SF!", p, c, cone[c], rootsRanks[p][c], roots[p][c], leavesRanks[p][c], leaves[p][c]);
        /* Get the corresponding local point */
        mainCone[c] = rmine1[rS + ind0];
      }
      if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, " mainCone=[%4D %4D %4D %4D]\n", mainCone[0], mainCone[1], mainCone[2], mainCone[3]));
      /* Set the desired order of p's cone points and fix orientations accordingly */
      CHKERRQ(DMPolytopeGetOrientation(ct, cone, mainCone, &o));
      CHKERRQ(DMPlexOrientPoint(dm, p, o));
    } else if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, " ==\n"));
  }
  if (debug) {
    CHKERRQ(PetscSynchronizedFlush(comm, NULL));
    CHKERRMPI(MPI_Barrier(comm));
  }
  CHKERRQ(DMViewFromOptions(dm, NULL, "-after_orient_interface_dm_view"));
  CHKERRQ(PetscFree4(roots, leaves, rootsRanks, leavesRanks));
  CHKERRQ(PetscFree2(rmine1, rremote1));
  PetscFunctionReturn(0);
}

static PetscErrorCode IntArrayViewFromOptions(MPI_Comm comm, const char opt[], const char name[], const char idxname[], const char valname[], PetscInt n, const PetscInt a[])
{
  PetscInt       idx;
  PetscMPIInt    rank;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHasName(NULL, NULL, opt, &flg));
  if (!flg) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]%s:\n", rank, name));
  for (idx = 0; idx < n; ++idx) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]%s %D %s %D\n", rank, idxname, idx, valname, a[idx]));
  CHKERRQ(PetscSynchronizedFlush(comm, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SFNodeArrayViewFromOptions(MPI_Comm comm, const char opt[], const char name[], const char idxname[], PetscInt n, const PetscSFNode a[])
{
  PetscInt       idx;
  PetscMPIInt    rank;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHasName(NULL, NULL, opt, &flg));
  if (!flg) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]%s:\n", rank, name));
  if (idxname) {
    for (idx = 0; idx < n; ++idx) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]%s %D rank %D index %D\n", rank, idxname, idx, a[idx].rank, a[idx].index));
  } else {
    for (idx = 0; idx < n; ++idx) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]rank %D index %D\n", rank, a[idx].rank, a[idx].index));
  }
  CHKERRQ(PetscSynchronizedFlush(comm, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMapToLocalPoint(DM dm, PetscHMapIJ remotehash, PetscSFNode remotePoint, PetscInt *localPoint, PetscBool *mapFailed)
{
  PetscSF         sf;
  const PetscInt *locals;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(PetscSFGetGraph(sf, NULL, NULL, &locals, NULL));
  if (mapFailed) *mapFailed = PETSC_FALSE;
  if (remotePoint.rank == rank) {
    *localPoint = remotePoint.index;
  } else {
    PetscHashIJKey key;
    PetscInt       l;

    key.i = remotePoint.index;
    key.j = remotePoint.rank;
    CHKERRQ(PetscHMapIJGet(remotehash, key, &l));
    if (l >= 0) {
      *localPoint = locals[l];
    } else if (mapFailed) *mapFailed = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMapToGlobalPoint(DM dm, PetscInt localPoint, PetscSFNode *remotePoint, PetscBool *mapFailed)
{
  PetscSF            sf;
  const PetscInt    *locals, *rootdegree;
  const PetscSFNode *remotes;
  PetscInt           Nl, l;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  if (mapFailed) *mapFailed = PETSC_FALSE;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(PetscSFGetGraph(sf, NULL, &Nl, &locals, &remotes));
  if (Nl < 0) goto owned;
  CHKERRQ(PetscSFComputeDegreeBegin(sf, &rootdegree));
  CHKERRQ(PetscSFComputeDegreeEnd(sf, &rootdegree));
  if (rootdegree[localPoint]) goto owned;
  CHKERRQ(PetscFindInt(localPoint, Nl, locals, &l));
  if (l < 0) {if (mapFailed) *mapFailed = PETSC_TRUE;}
  else *remotePoint = remotes[l];
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

  PetscFunctionBegin;
  *isShared = PETSC_FALSE;
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(PetscSFGetGraph(sf, NULL, &Nl, &locals, NULL));
  if (Nl < 0) PetscFunctionReturn(0);
  CHKERRQ(PetscFindInt(p, Nl, locals, &idx));
  if (idx >= 0) {*isShared = PETSC_TRUE; PetscFunctionReturn(0);}
  CHKERRQ(PetscSFComputeDegreeBegin(sf, &rootdegree));
  CHKERRQ(PetscSFComputeDegreeEnd(sf, &rootdegree));
  if (rootdegree[p] > 0) *isShared = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexConeIsShared(DM dm, PetscInt p, PetscBool *isShared)
{
  const PetscInt *cone;
  PetscInt        coneSize, c;
  PetscBool       cShared = PETSC_TRUE;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
  CHKERRQ(DMPlexGetCone(dm, p, &cone));
  for (c = 0; c < coneSize; ++c) {
    PetscBool pointShared;

    CHKERRQ(DMPlexPointIsShared(dm, cone[c], &pointShared));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
  CHKERRQ(DMPlexGetCone(dm, p, &cone));
  for (c = 0; c < coneSize; ++c) {
    PetscSFNode rcp;
    PetscBool   mapFailed;

    CHKERRQ(DMPlexMapToGlobalPoint(dm, cone[c], &rcp, &mapFailed));
    if (mapFailed) {
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMPlexGetOverlap(dm, &overlap));
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetPointHeight(dm, p, &height));
  if (!overlap && height <= cellHeight+1) {
    /* cells can't be shared for non-overlapping meshes */
    if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    Skipping face %D to avoid adding cell to hashmap since this is nonoverlapping mesh\n", rank, p));
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMPlexGetSupportSize(dm, p, &supportSize));
  CHKERRQ(DMPlexGetSupport(dm, p, &support));
  if (candidates) CHKERRQ(PetscSectionGetOffset(candidateSection, p, &off));
  for (s = 0; s < supportSize; ++s) {
    const PetscInt  face = support[s];
    const PetscInt *cone;
    PetscSFNode     cpmin={-1,-1}, rp={-1,-1};
    PetscInt        coneSize, c, f;
    PetscBool       isShared = PETSC_FALSE;
    PetscHashIJKey  key;

    /* Only add point once */
    if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    Support face %D\n", rank, face));
    key.i = p;
    key.j = face;
    CHKERRQ(PetscHMapIJGet(faceHash, key, &f));
    if (f >= 0) continue;
    CHKERRQ(DMPlexConeIsShared(dm, face, &isShared));
    CHKERRQ(DMPlexGetConeMinimum(dm, face, &cpmin));
    CHKERRQ(DMPlexMapToGlobalPoint(dm, p, &rp, NULL));
    if (debug) {
      CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]      Face point %D is shared: %d\n", rank, face, (int) isShared));
      CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]      Global point (%D, %D) Min Cone Point (%D, %D)\n", rank, rp.rank, rp.index, cpmin.rank, cpmin.index));
    }
    if (isShared && (rp.rank == cpmin.rank && rp.index == cpmin.index)) {
      CHKERRQ(PetscHMapIJSet(faceHash, key, p));
      if (candidates) {
        if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    Adding shared face %D at idx %D\n[%d]     ", rank, face, idx, rank));
        CHKERRQ(DMPlexGetConeSize(dm, face, &coneSize));
        CHKERRQ(DMPlexGetCone(dm, face, &cone));
        candidates[off+idx].rank    = -1;
        candidates[off+idx++].index = coneSize-1;
        candidates[off+idx].rank    = rank;
        candidates[off+idx++].index = face;
        for (c = 0; c < coneSize; ++c) {
          const PetscInt cp = cone[c];

          if (cp == p) continue;
          CHKERRQ(DMPlexMapToGlobalPoint(dm, cp, &candidates[off+idx], NULL));
          if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, " (%D,%D)", candidates[off+idx].rank, candidates[off+idx].index));
          ++idx;
        }
        if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "\n"));
      } else {
        /* Add cone size to section */
        if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    Scheduling shared face %D\n", rank, face));
        CHKERRQ(DMPlexGetConeSize(dm, face, &coneSize));
        CHKERRQ(PetscHMapIJSet(faceHash, key, p));
        CHKERRQ(PetscSectionAddDof(candidateSection, p, coneSize+1));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(pointSF, PETSCSF_CLASSID, 2);
  CHKERRQ(DMPlexIsDistributed(dm, &flg));
  if (!flg) PetscFunctionReturn(0);
  /* Set initial SF so that lower level queries work */
  CHKERRQ(DMSetPointSF(dm, pointSF));
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMPlexGetOverlap(dm, &ov));
  PetscCheck(!ov,comm, PETSC_ERR_SUP, "Interpolation of overlapped DMPlex not implemented yet");
  CHKERRQ(PetscOptionsHasName(NULL, ((PetscObject) dm)->prefix, "-dmplex_interp_debug", &debug));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_interp_pre_view"));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) pointSF, NULL, "-petscsf_interp_pre_view"));
  CHKERRQ(PetscLogEventBegin(DMPLEX_InterpolateSF,dm,0,0,0));
  /* Step 0: Precalculations */
  CHKERRQ(PetscSFGetGraph(pointSF, &Nr, &Nl, &localPoints, &remotePoints));
  PetscCheckFalse(Nr < 0,comm, PETSC_ERR_ARG_WRONGSTATE, "This DMPlex is distributed but input PointSF has no graph set");
  CHKERRQ(PetscHMapIJCreate(&remoteHash));
  for (l = 0; l < Nl; ++l) {
    PetscHashIJKey key;
    key.i = remotePoints[l].index;
    key.j = remotePoints[l].rank;
    CHKERRQ(PetscHMapIJSet(remoteHash, key, l));
  }
  /*   Compute root degree to identify shared points */
  CHKERRQ(PetscSFComputeDegreeBegin(pointSF, &rootdegree));
  CHKERRQ(PetscSFComputeDegreeEnd(pointSF, &rootdegree));
  CHKERRQ(IntArrayViewFromOptions(comm, "-interp_root_degree_view", "Root degree", "point", "degree", Nr, rootdegree));
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
  CHKERRQ(PetscSectionCreate(comm, &candidateSection));
  CHKERRQ(PetscSectionSetChart(candidateSection, 0, Nr));
  {
    PetscHMapIJ faceHash;

    CHKERRQ(PetscHMapIJCreate(&faceHash));
    for (l = 0; l < Nl; ++l) {
      const PetscInt p = localPoints[l];

      if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]  First pass leaf point %D\n", rank, p));
      CHKERRQ(DMPlexAddSharedFace_Private(dm, candidateSection, NULL, faceHash, p, debug));
    }
    CHKERRQ(PetscHMapIJClear(faceHash));
    CHKERRQ(PetscSectionSetUp(candidateSection));
    CHKERRQ(PetscSectionGetStorageSize(candidateSection, &candidatesSize));
    CHKERRQ(PetscMalloc1(candidatesSize, &candidates));
    for (l = 0; l < Nl; ++l) {
      const PetscInt p = localPoints[l];

      if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]  Second pass leaf point %D\n", rank, p));
      CHKERRQ(DMPlexAddSharedFace_Private(dm, candidateSection, candidates, faceHash, p, debug));
    }
    CHKERRQ(PetscHMapIJDestroy(&faceHash));
    if (debug) CHKERRQ(PetscSynchronizedFlush(comm, NULL));
  }
  CHKERRQ(PetscObjectSetName((PetscObject) candidateSection, "Candidate Section"));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) candidateSection, NULL, "-petscsection_interp_candidate_view"));
  CHKERRQ(SFNodeArrayViewFromOptions(comm, "-petscsection_interp_candidate_view", "Candidates", NULL, candidatesSize, candidates));
  /* Step 2: Gather candidate section / array pair into the root partition via inverse(multi(pointSF)). */
  /*   Note that this section is indexed by offsets into leaves, not by point number */
  {
    PetscSF   sfMulti, sfInverse, sfCandidates;
    PetscInt *remoteOffsets;

    CHKERRQ(PetscSFGetMultiSF(pointSF, &sfMulti));
    CHKERRQ(PetscSFCreateInverseSF(sfMulti, &sfInverse));
    CHKERRQ(PetscSectionCreate(comm, &candidateRemoteSection));
    CHKERRQ(PetscSFDistributeSection(sfInverse, candidateSection, &remoteOffsets, candidateRemoteSection));
    CHKERRQ(PetscSFCreateSectionSF(sfInverse, candidateSection, remoteOffsets, candidateRemoteSection, &sfCandidates));
    CHKERRQ(PetscSectionGetStorageSize(candidateRemoteSection, &candidatesRemoteSize));
    CHKERRQ(PetscMalloc1(candidatesRemoteSize, &candidatesRemote));
    CHKERRQ(PetscSFBcastBegin(sfCandidates, MPIU_2INT, candidates, candidatesRemote,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sfCandidates, MPIU_2INT, candidates, candidatesRemote,MPI_REPLACE));
    CHKERRQ(PetscSFDestroy(&sfInverse));
    CHKERRQ(PetscSFDestroy(&sfCandidates));
    CHKERRQ(PetscFree(remoteOffsets));

    CHKERRQ(PetscObjectSetName((PetscObject) candidateRemoteSection, "Remote Candidate Section"));
    CHKERRQ(PetscObjectViewFromOptions((PetscObject) candidateRemoteSection, NULL, "-petscsection_interp_candidate_remote_view"));
    CHKERRQ(SFNodeArrayViewFromOptions(comm, "-petscsection_interp_candidate_remote_view", "Remote Candidates", NULL, candidatesRemoteSize, candidatesRemote));
  }
  /* Step 3: At the root, if at least two faces with a given cone are present, including a local face, mark the face as shared and choose the root face */
  {
    PetscHashIJKLRemote faceTable;
    PetscInt            idx, idx2;

    CHKERRQ(PetscHashIJKLRemoteCreate(&faceTable));
    /* There is a section point for every leaf attached to a given root point */
    for (r = 0, idx = 0, idx2 = 0; r < Nr; ++r) {
      PetscInt deg;

      for (deg = 0; deg < rootdegree[r]; ++deg, ++idx) {
        PetscInt offset, dof, d;

        CHKERRQ(PetscSectionGetDof(candidateRemoteSection, idx, &dof));
        CHKERRQ(PetscSectionGetOffset(candidateRemoteSection, idx, &offset));
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
          PetscBool              missing,mapToLocalPointFailed = PETSC_FALSE;
          PetscInt               points[1024], p, joinSize;

          if (debug) CHKERRQ(PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Checking face (%D, %D) at (%D, %D, %D) with cone size %D\n", rank, rface.rank, rface.index, r, idx, d, Np));
          PetscCheckFalse(Np > 4,PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle face (%D, %D) at (%D, %D, %D) with %D cone points", rface.rank, rface.index, r, idx, d, Np);
          fcp0.rank  = rank;
          fcp0.index = r;
          d += Np;
          /* Put remote face in hash table */
          key.i = fcp0;
          key.j = fcone[0];
          key.k = Np > 2 ? fcone[1] : pmax;
          key.l = Np > 3 ? fcone[2] : pmax;
          CHKERRQ(PetscSortSFNode(Np, (PetscSFNode *) &key));
          CHKERRQ(PetscHashIJKLRemotePut(faceTable, key, &iter, &missing));
          if (missing) {
            if (debug) CHKERRQ(PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Setting remote face (%D, %D)\n", rank, rface.index, rface.rank));
            CHKERRQ(PetscHashIJKLRemoteIterSet(faceTable, iter, rface));
          } else {
            PetscSFNode oface;

            CHKERRQ(PetscHashIJKLRemoteIterGet(faceTable, iter, &oface));
            if ((rface.rank < oface.rank) || (rface.rank == oface.rank && rface.index < oface.index)) {
              if (debug) CHKERRQ(PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Replacing with remote face (%D, %D)\n", rank, rface.index, rface.rank));
              CHKERRQ(PetscHashIJKLRemoteIterSet(faceTable, iter, rface));
            }
          }
          /* Check for local face */
          points[0] = r;
          for (p = 1; p < Np; ++p) {
            CHKERRQ(DMPlexMapToLocalPoint(dm, remoteHash, fcone[p-1], &points[p], &mapToLocalPointFailed));
            if (mapToLocalPointFailed) break; /* We got a point not in our overlap */
            if (debug) CHKERRQ(PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Checking local candidate %D\n", rank, points[p]));
          }
          if (mapToLocalPointFailed) continue;
          CHKERRQ(DMPlexGetJoin(dm, Np, points, &joinSize, &join));
          if (joinSize == 1) {
            PetscSFNode lface;
            PetscSFNode oface;

            /* Always replace with local face */
            lface.rank  = rank;
            lface.index = join[0];
            CHKERRQ(PetscHashIJKLRemoteIterGet(faceTable, iter, &oface));
            if (debug) CHKERRQ(PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Replacing (%D, %D) with local face (%D, %D)\n", rank, oface.index, oface.rank, lface.index, lface.rank));
            CHKERRQ(PetscHashIJKLRemoteIterSet(faceTable, iter, lface));
          }
          CHKERRQ(DMPlexRestoreJoin(dm, Np, points, &joinSize, &join));
        }
      }
      /* Put back faces for this root */
      for (deg = 0; deg < rootdegree[r]; ++deg, ++idx2) {
        PetscInt offset, dof, d;

        CHKERRQ(PetscSectionGetDof(candidateRemoteSection, idx2, &dof));
        CHKERRQ(PetscSectionGetOffset(candidateRemoteSection, idx2, &offset));
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

          if (debug) CHKERRQ(PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]  Entering face at (%D, %D)\n", rank, r, idx));
          PetscCheckFalse(Np > 4,PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle faces with %D cone points", Np);
          fcp0.rank  = rank;
          fcp0.index = r;
          d += Np;
          /* Find remote face in hash table */
          key.i = fcp0;
          key.j = fcone[0];
          key.k = Np > 2 ? fcone[1] : pmax;
          key.l = Np > 3 ? fcone[2] : pmax;
          CHKERRQ(PetscSortSFNode(Np, (PetscSFNode *) &key));
          if (debug) CHKERRQ(PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d]    key (%D, %D) (%D, %D) (%D, %D) (%D, %D)\n", rank, key.i.rank, key.i.index, key.j.rank, key.j.index, key.k.rank, key.k.index, key.l.rank, key.l.index));
          CHKERRQ(PetscHashIJKLRemotePut(faceTable, key, &iter, &missing));
          PetscCheck(!missing,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Root %D Idx %D ought to have an associated face", r, idx2);
          else        CHKERRQ(PetscHashIJKLRemoteIterGet(faceTable, iter, &candidatesRemote[hidx]));
        }
      }
    }
    if (debug) CHKERRQ(PetscSynchronizedFlush(PetscObjectComm((PetscObject) dm), NULL));
    CHKERRQ(PetscHashIJKLRemoteDestroy(&faceTable));
  }
  /* Step 4: Push back owned faces */
  {
    PetscSF      sfMulti, sfClaims, sfPointNew;
    PetscSFNode *remotePointsNew;
    PetscInt    *remoteOffsets, *localPointsNew;
    PetscInt     pStart, pEnd, r, NlNew, p;

    /* 4) Push claims back to receiver via the MultiSF and derive new pointSF mapping on receiver */
    CHKERRQ(PetscSFGetMultiSF(pointSF, &sfMulti));
    CHKERRQ(PetscSectionCreate(comm, &claimSection));
    CHKERRQ(PetscSFDistributeSection(sfMulti, candidateRemoteSection, &remoteOffsets, claimSection));
    CHKERRQ(PetscSFCreateSectionSF(sfMulti, candidateRemoteSection, remoteOffsets, claimSection, &sfClaims));
    CHKERRQ(PetscSectionGetStorageSize(claimSection, &claimsSize));
    CHKERRQ(PetscMalloc1(claimsSize, &claims));
    for (p = 0; p < claimsSize; ++p) claims[p].rank = -1;
    CHKERRQ(PetscSFBcastBegin(sfClaims, MPIU_2INT, candidatesRemote, claims,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sfClaims, MPIU_2INT, candidatesRemote, claims,MPI_REPLACE));
    CHKERRQ(PetscSFDestroy(&sfClaims));
    CHKERRQ(PetscFree(remoteOffsets));
    CHKERRQ(PetscObjectSetName((PetscObject) claimSection, "Claim Section"));
    CHKERRQ(PetscObjectViewFromOptions((PetscObject) claimSection, NULL, "-petscsection_interp_claim_view"));
    CHKERRQ(SFNodeArrayViewFromOptions(comm, "-petscsection_interp_claim_view", "Claims", NULL, claimsSize, claims));
    /* Step 5) Walk the original section of local supports and add an SF entry for each updated item */
    /* TODO I should not have to do a join here since I already put the face and its cone in the candidate section */
    CHKERRQ(PetscHMapICreate(&claimshash));
    for (r = 0; r < Nr; ++r) {
      PetscInt dof, off, d;

      if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]  Checking root for claims %D\n", rank, r));
      CHKERRQ(PetscSectionGetDof(candidateSection, r, &dof));
      CHKERRQ(PetscSectionGetOffset(candidateSection, r, &off));
      for (d = 0; d < dof;) {
        if (claims[off+d].rank >= 0) {
          const PetscInt  faceInd = off+d;
          const PetscInt  Np      = candidates[off+d].index;
          const PetscInt *join    = NULL;
          PetscInt        joinSize, points[1024], c;

          if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    Found claim for remote point (%D, %D)\n", rank, claims[faceInd].rank, claims[faceInd].index));
          points[0] = r;
          if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]      point %D\n", rank, points[0]));
          for (c = 0, d += 2; c < Np; ++c, ++d) {
            CHKERRQ(DMPlexMapToLocalPoint(dm, remoteHash, candidates[off+d], &points[c+1], NULL));
            if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]      point %D\n", rank, points[c+1]));
          }
          CHKERRQ(DMPlexGetJoin(dm, Np+1, points, &joinSize, &join));
          if (joinSize == 1) {
            if (claims[faceInd].rank == rank) {
              if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    Ignoring local face %D for non-remote partner\n", rank, join[0]));
            } else {
              if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    Found local face %D\n", rank, join[0]));
              CHKERRQ(PetscHMapISet(claimshash, join[0], faceInd));
            }
          } else {
            if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    Failed to find face\n", rank));
          }
          CHKERRQ(DMPlexRestoreJoin(dm, Np+1, points, &joinSize, &join));
        } else {
          if (debug) CHKERRQ(PetscSynchronizedPrintf(comm, "[%d]    No claim for point %D\n", rank, r));
          d += claims[off+d].index+1;
        }
      }
    }
    if (debug) CHKERRQ(PetscSynchronizedFlush(comm, NULL));
    /* Step 6) Create new pointSF from hashed claims */
    CHKERRQ(PetscHMapIGetSize(claimshash, &NlNew));
    CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
    CHKERRQ(PetscMalloc1(Nl + NlNew, &localPointsNew));
    CHKERRQ(PetscMalloc1(Nl + NlNew, &remotePointsNew));
    for (l = 0; l < Nl; ++l) {
      localPointsNew[l] = localPoints[l];
      remotePointsNew[l].index = remotePoints[l].index;
      remotePointsNew[l].rank  = remotePoints[l].rank;
    }
    p = Nl;
    CHKERRQ(PetscHMapIGetKeys(claimshash, &p, localPointsNew));
    /* We sort new points, and assume they are numbered after all existing points */
    CHKERRQ(PetscSortInt(NlNew, &localPointsNew[Nl]));
    for (p = Nl; p < Nl + NlNew; ++p) {
      PetscInt off;
      CHKERRQ(PetscHMapIGet(claimshash, localPointsNew[p], &off));
      PetscCheckFalse(claims[off].rank < 0 || claims[off].index < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid claim for local point %D, (%D, %D)", localPointsNew[p], claims[off].rank, claims[off].index);
      remotePointsNew[p] = claims[off];
    }
    CHKERRQ(PetscSFCreate(comm, &sfPointNew));
    CHKERRQ(PetscSFSetGraph(sfPointNew, pEnd-pStart, Nl+NlNew, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER));
    CHKERRQ(PetscSFSetUp(sfPointNew));
    CHKERRQ(DMSetPointSF(dm, sfPointNew));
    CHKERRQ(PetscObjectViewFromOptions((PetscObject) sfPointNew, NULL, "-petscsf_interp_view"));
    CHKERRQ(PetscSFDestroy(&sfPointNew));
    CHKERRQ(PetscHMapIDestroy(&claimshash));
  }
  CHKERRQ(PetscHMapIJDestroy(&remoteHash));
  CHKERRQ(PetscSectionDestroy(&candidateSection));
  CHKERRQ(PetscSectionDestroy(&candidateRemoteSection));
  CHKERRQ(PetscSectionDestroy(&claimSection));
  CHKERRQ(PetscFree(candidates));
  CHKERRQ(PetscFree(candidatesRemote));
  CHKERRQ(PetscFree(claims));
  CHKERRQ(PetscLogEventEnd(DMPLEX_InterpolateSF,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmInt, 2);
  CHKERRQ(PetscLogEventBegin(DMPLEX_Interpolate,dm,0,0,0));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsInterpolated(dm, &interpolated));
  PetscCheckFalse(interpolated == DMPLEX_INTERPOLATED_PARTIAL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated == DMPLEX_INTERPOLATED_FULL) {
    CHKERRQ(PetscObjectReference((PetscObject) dm));
    idm  = dm;
  } else {
    for (d = 1; d < dim; ++d) {
      /* Create interpolated mesh */
      CHKERRQ(DMCreate(PetscObjectComm((PetscObject)dm), &idm));
      CHKERRQ(DMSetType(idm, DMPLEX));
      CHKERRQ(DMSetDimension(idm, dim));
      if (depth > 0) {
        CHKERRQ(DMPlexInterpolateFaces_Internal(odm, 1, idm));
        CHKERRQ(DMGetPointSF(odm, &sfPoint));
        {
          /* TODO: We need to systematically fix cases of distributed Plexes with no graph set */
          PetscInt nroots;
          CHKERRQ(PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL));
          if (nroots >= 0) CHKERRQ(DMPlexInterpolatePointSF(idm, sfPoint));
        }
      }
      if (odm != dm) CHKERRQ(DMDestroy(&odm));
      odm = idm;
    }
    CHKERRQ(PetscObjectGetName((PetscObject) dm,  &name));
    CHKERRQ(PetscObjectSetName((PetscObject) idm,  name));
    CHKERRQ(DMPlexCopyCoordinates(dm, idm));
    CHKERRQ(DMCopyLabels(dm, idm, PETSC_COPY_VALUES, PETSC_FALSE, DM_COPY_LABELS_FAIL));
    CHKERRQ(PetscOptionsGetBool(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dm_plex_interpolate_orient_interfaces", &flg, NULL));
    if (flg) CHKERRQ(DMPlexOrientInterface_Internal(idm));
  }
  /* This function makes the mesh fully interpolated on all ranks */
  {
    DM_Plex *plex = (DM_Plex *) idm->data;
    plex->interpolated = plex->interpolatedCollective = DMPLEX_INTERPOLATED_FULL;
  }
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, idm));
  *dmInt = idm;
  CHKERRQ(PetscLogEventEnd(DMPLEX_Interpolate,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  if (dmA == dmB) PetscFunctionReturn(0);
  CHKERRQ(DMGetCoordinateDim(dmA, &cdim));
  CHKERRQ(DMSetCoordinateDim(dmB, cdim));
  CHKERRQ(DMPlexGetDepthStratum(dmA, 0, &vStartA, &vEndA));
  CHKERRQ(DMPlexGetDepthStratum(dmB, 0, &vStartB, &vEndB));
  PetscCheckFalse((vEndA-vStartA) != (vEndB-vStartB),PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of vertices in first DM %d != %d in the second DM", vEndA-vStartA, vEndB-vStartB);
  /* Copy over discretization if it exists */
  {
    DM                 cdmA, cdmB;
    PetscDS            dsA, dsB;
    PetscObject        objA, objB;
    PetscClassId       idA, idB;
    const PetscScalar *constants;
    PetscInt            cdim, Nc;

    CHKERRQ(DMGetCoordinateDM(dmA, &cdmA));
    CHKERRQ(DMGetCoordinateDM(dmB, &cdmB));
    CHKERRQ(DMGetField(cdmA, 0, NULL, &objA));
    CHKERRQ(DMGetField(cdmB, 0, NULL, &objB));
    CHKERRQ(PetscObjectGetClassId(objA, &idA));
    CHKERRQ(PetscObjectGetClassId(objB, &idB));
    if ((idA == PETSCFE_CLASSID) && (idA != idB)) {
      CHKERRQ(DMSetField(cdmB, 0, NULL, objA));
      CHKERRQ(DMCreateDS(cdmB));
      CHKERRQ(DMGetDS(cdmA, &dsA));
      CHKERRQ(DMGetDS(cdmB, &dsB));
      CHKERRQ(PetscDSGetCoordinateDimension(dsA, &cdim));
      CHKERRQ(PetscDSSetCoordinateDimension(dsB, cdim));
      CHKERRQ(PetscDSGetConstants(dsA, &Nc, &constants));
      CHKERRQ(PetscDSSetConstants(dsB, Nc, (PetscScalar *) constants));
    }
  }
  CHKERRQ(DMPlexGetHeightStratum(dmA, 0, &cStartA, &cEndA));
  CHKERRQ(DMPlexGetHeightStratum(dmB, 0, &cStartB, &cEndB));
  CHKERRQ(DMGetCoordinateSection(dmA, &coordSectionA));
  CHKERRQ(DMGetCoordinateSection(dmB, &coordSectionB));
  if (coordSectionA == coordSectionB) PetscFunctionReturn(0);
  CHKERRQ(PetscSectionGetNumFields(coordSectionA, &Nf));
  if (!Nf) PetscFunctionReturn(0);
  PetscCheckFalse(Nf > 1,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of coordinate fields must be 1, not %D", Nf);
  if (!coordSectionB) {
    PetscInt dim;

    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) coordSectionA), &coordSectionB));
    CHKERRQ(DMGetCoordinateDim(dmA, &dim));
    CHKERRQ(DMSetCoordinateSection(dmB, dim, coordSectionB));
    CHKERRQ(PetscObjectDereference((PetscObject) coordSectionB));
  }
  CHKERRQ(PetscSectionSetNumFields(coordSectionB, 1));
  CHKERRQ(PetscSectionGetFieldComponents(coordSectionA, 0, &spaceDim));
  CHKERRQ(PetscSectionSetFieldComponents(coordSectionB, 0, spaceDim));
  CHKERRQ(PetscSectionGetChart(coordSectionA, &cS, &cE));
  if (cStartA <= cS && cS < cEndA) { /* localized coordinates */
    PetscCheckFalse((cEndA-cStartA) != (cEndB-cStartB),PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of cells in first DM %D != %D in the second DM", cEndA-cStartA, cEndB-cStartB);
    cS = cS - cStartA + cStartB;
    cE = vEndB;
    lc = PETSC_TRUE;
  } else {
    cS = vStartB;
    cE = vEndB;
  }
  CHKERRQ(PetscSectionSetChart(coordSectionB, cS, cE));
  for (v = vStartB; v < vEndB; ++v) {
    CHKERRQ(PetscSectionSetDof(coordSectionB, v, spaceDim));
    CHKERRQ(PetscSectionSetFieldDof(coordSectionB, v, 0, spaceDim));
  }
  if (lc) { /* localized coordinates */
    PetscInt c;

    for (c = cS-cStartB; c < cEndB-cStartB; c++) {
      PetscInt dof;

      CHKERRQ(PetscSectionGetDof(coordSectionA, c + cStartA, &dof));
      CHKERRQ(PetscSectionSetDof(coordSectionB, c + cStartB, dof));
      CHKERRQ(PetscSectionSetFieldDof(coordSectionB, c + cStartB, 0, dof));
    }
  }
  CHKERRQ(PetscSectionSetUp(coordSectionB));
  CHKERRQ(PetscSectionGetStorageSize(coordSectionB, &coordSizeB));
  CHKERRQ(DMGetCoordinatesLocal(dmA, &coordinatesA));
  CHKERRQ(VecCreate(PETSC_COMM_SELF, &coordinatesB));
  CHKERRQ(PetscObjectSetName((PetscObject) coordinatesB, "coordinates"));
  CHKERRQ(VecSetSizes(coordinatesB, coordSizeB, PETSC_DETERMINE));
  CHKERRQ(VecGetBlockSize(coordinatesA, &d));
  CHKERRQ(VecSetBlockSize(coordinatesB, d));
  CHKERRQ(VecGetType(coordinatesA, &vtype));
  CHKERRQ(VecSetType(coordinatesB, vtype));
  CHKERRQ(VecGetArray(coordinatesA, &coordsA));
  CHKERRQ(VecGetArray(coordinatesB, &coordsB));
  for (v = 0; v < vEndB-vStartB; ++v) {
    PetscInt offA, offB;

    CHKERRQ(PetscSectionGetOffset(coordSectionA, v + vStartA, &offA));
    CHKERRQ(PetscSectionGetOffset(coordSectionB, v + vStartB, &offB));
    for (d = 0; d < spaceDim; ++d) {
      coordsB[offB+d] = coordsA[offA+d];
    }
  }
  if (lc) { /* localized coordinates */
    PetscInt c;

    for (c = cS-cStartB; c < cEndB-cStartB; c++) {
      PetscInt dof, offA, offB;

      CHKERRQ(PetscSectionGetOffset(coordSectionA, c + cStartA, &offA));
      CHKERRQ(PetscSectionGetOffset(coordSectionB, c + cStartB, &offB));
      CHKERRQ(PetscSectionGetDof(coordSectionA, c + cStartA, &dof));
      CHKERRQ(PetscArraycpy(coordsB + offB,coordsA + offA,dof));
    }
  }
  CHKERRQ(VecRestoreArray(coordinatesA, &coordsA));
  CHKERRQ(VecRestoreArray(coordinatesB, &coordsB));
  CHKERRQ(DMSetCoordinatesLocal(dmB, coordinatesB));
  CHKERRQ(VecDestroy(&coordinatesB));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmUnint, 2);
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsInterpolated(dm, &interpolated));
  PetscCheckFalse(interpolated == DMPLEX_INTERPOLATED_PARTIAL,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated == DMPLEX_INTERPOLATED_NONE || dim <= 1) {
    /* in case dim <= 1 just keep the DMPLEX_INTERPOLATED_FULL flag */
    CHKERRQ(PetscObjectReference((PetscObject) dm));
    *dmUnint = dm;
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), &udm));
  CHKERRQ(DMSetType(udm, DMPLEX));
  CHKERRQ(DMSetDimension(udm, dim));
  CHKERRQ(DMPlexSetChart(udm, cStart, vEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) ++coneSize;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    CHKERRQ(DMPlexSetConeSize(udm, c, coneSize));
    maxConeSize = PetscMax(maxConeSize, coneSize);
  }
  CHKERRQ(DMSetUp(udm));
  CHKERRQ(PetscMalloc1(maxConeSize, &cone));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) cone[coneSize++] = p;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    CHKERRQ(DMPlexSetCone(udm, c, cone));
  }
  CHKERRQ(PetscFree(cone));
  CHKERRQ(DMPlexSymmetrize(udm));
  CHKERRQ(DMPlexStratify(udm));
  /* Reduce SF */
  {
    PetscSF            sfPoint, sfPointUn;
    const PetscSFNode *remotePoints;
    const PetscInt    *localPoints;
    PetscSFNode       *remotePointsUn;
    PetscInt          *localPointsUn;
    PetscInt           vEnd, numRoots, numLeaves, l;
    PetscInt           numLeavesUn = 0, n = 0;

    /* Get original SF information */
    CHKERRQ(DMGetPointSF(dm, &sfPoint));
    CHKERRQ(DMGetPointSF(udm, &sfPointUn));
    CHKERRQ(DMPlexGetDepthStratum(dm, 0, NULL, &vEnd));
    CHKERRQ(PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints));
    /* Allocate space for cells and vertices */
    for (l = 0; l < numLeaves; ++l) if (localPoints[l] < vEnd) numLeavesUn++;
    /* Fill in leaves */
    if (vEnd >= 0) {
      CHKERRQ(PetscMalloc1(numLeavesUn, &remotePointsUn));
      CHKERRQ(PetscMalloc1(numLeavesUn, &localPointsUn));
      for (l = 0; l < numLeaves; l++) {
        if (localPoints[l] < vEnd) {
          localPointsUn[n]        = localPoints[l];
          remotePointsUn[n].rank  = remotePoints[l].rank;
          remotePointsUn[n].index = remotePoints[l].index;
          ++n;
        }
      }
      PetscCheckFalse(n != numLeavesUn,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent number of leaves %d != %d", n, numLeavesUn);
      CHKERRQ(PetscSFSetGraph(sfPointUn, vEnd, numLeavesUn, localPointsUn, PETSC_OWN_POINTER, remotePointsUn, PETSC_OWN_POINTER));
    }
  }
  /* This function makes the mesh fully uninterpolated on all ranks */
  {
    DM_Plex *plex = (DM_Plex *) udm->data;
    plex->interpolated = plex->interpolatedCollective = DMPLEX_INTERPOLATED_NONE;
  }
  CHKERRQ(DMPlexCopy_Internal(dm, PETSC_TRUE, udm));
  *dmUnint = udm;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexIsInterpolated_Internal(DM dm, DMPlexInterpolatedFlag *interpolated)
{
  PetscInt       coneSize, depth, dim, h, p, pStart, pEnd;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMGetDimension(dm, &dim));

  if (depth == dim) {
    *interpolated = DMPLEX_INTERPOLATED_FULL;
    if (!dim) goto finish;

    /* Check points at height = dim are vertices (have no cones) */
    CHKERRQ(DMPlexGetHeightStratum(dm, dim, &pStart, &pEnd));
    for (p=pStart; p<pEnd; p++) {
      CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
      if (coneSize) {
        *interpolated = DMPLEX_INTERPOLATED_PARTIAL;
        goto finish;
      }
    }

    /* Check points at height < dim have cones */
    for (h=0; h<dim; h++) {
      CHKERRQ(DMPlexGetHeightStratum(dm, h, &pStart, &pEnd));
      for (p=pStart; p<pEnd; p++) {
        CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(interpolated,2);
  if (plex->interpolated < 0) {
    CHKERRQ(DMPlexIsInterpolated_Internal(dm, &plex->interpolated));
  } else if (PetscDefined (USE_DEBUG)) {
    DMPlexInterpolatedFlag flg;

    CHKERRQ(DMPlexIsInterpolated_Internal(dm, &flg));
    PetscCheckFalse(flg != plex->interpolated,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stashed DMPlexInterpolatedFlag %s is inconsistent with current %s", DMPlexInterpolatedFlags[plex->interpolated], DMPlexInterpolatedFlags[flg]);
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(interpolated,2);
  CHKERRQ(PetscOptionsGetBool(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_plex_is_interpolated_collective_debug", &debug, NULL));
  if (plex->interpolatedCollective < 0) {
    DMPlexInterpolatedFlag  min, max;
    MPI_Comm                comm;

    CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
    CHKERRQ(DMPlexIsInterpolated(dm, &plex->interpolatedCollective));
    CHKERRMPI(MPI_Allreduce(&plex->interpolatedCollective, &min, 1, MPIU_ENUM, MPI_MIN, comm));
    CHKERRMPI(MPI_Allreduce(&plex->interpolatedCollective, &max, 1, MPIU_ENUM, MPI_MAX, comm));
    if (min != max) plex->interpolatedCollective = DMPLEX_INTERPOLATED_MIXED;
    if (debug) {
      PetscMPIInt rank;

      CHKERRMPI(MPI_Comm_rank(comm, &rank));
      CHKERRQ(PetscSynchronizedPrintf(comm, "[%d] interpolated=%s interpolatedCollective=%s\n", rank, DMPlexInterpolatedFlags[plex->interpolated], DMPlexInterpolatedFlags[plex->interpolatedCollective]));
      CHKERRQ(PetscSynchronizedFlush(comm, PETSC_STDOUT));
    }
  }
  *interpolated = plex->interpolatedCollective;
  PetscFunctionReturn(0);
}
