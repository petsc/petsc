#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>
#include <petsc/private/hashmapij.h>

const char *const DMPlexInterpolatedFlags[] = {"none", "partial", "mixed", "full", "DMPlexInterpolatedFlag", "DMPLEX_INTERPOLATED_", NULL};

/* HMapIJKL */

#include <petsc/private/hashmapijkl.h>

static PetscSFNode _PetscInvalidSFNode = {-1, -1};

typedef struct _PetscHMapIJKLRemoteKey {
  PetscSFNode i, j, k, l;
} PetscHMapIJKLRemoteKey;

#define PetscHMapIJKLRemoteKeyHash(key) \
  PetscHashCombine(PetscHashCombine(PetscHashInt((key).i.rank + (key).i.index), PetscHashInt((key).j.rank + (key).j.index)), PetscHashCombine(PetscHashInt((key).k.rank + (key).k.index), PetscHashInt((key).l.rank + (key).l.index)))

#define PetscHMapIJKLRemoteKeyEqual(k1, k2) \
  (((k1).i.rank == (k2).i.rank) ? ((k1).i.index == (k2).i.index) ? ((k1).j.rank == (k2).j.rank) ? ((k1).j.index == (k2).j.index) ? ((k1).k.rank == (k2).k.rank) ? ((k1).k.index == (k2).k.index) ? ((k1).l.rank == (k2).l.rank) ? ((k1).l.index == (k2).l.index) : 0 : 0 : 0 : 0 : 0 : 0 : 0)

PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PETSC_HASH_MAP(HMapIJKLRemote, PetscHMapIJKLRemoteKey, PetscSFNode, PetscHMapIJKLRemoteKeyHash, PetscHMapIJKLRemoteKeyEqual, _PetscInvalidSFNode))

  static PetscErrorCode PetscSortSFNode(PetscInt n, PetscSFNode A[])
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 1; i < n; ++i) {
    PetscSFNode x = A[i];
    PetscInt    j;

    for (j = i - 1; j >= 0; --j) {
      if ((A[j].rank > x.rank) || (A[j].rank == x.rank && A[j].index > x.index)) break;
      A[j + 1] = A[j];
    }
    A[j + 1] = x;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexGetRawFaces_Internal - Gets groups of vertices that correspond to faces for the given cone
*/
PetscErrorCode DMPlexGetRawFaces_Internal(DM dm, DMPolytopeType ct, const PetscInt cone[], PetscInt *numFaces, const DMPolytopeType *faceTypes[], const PetscInt *faceSizes[], const PetscInt *faces[])
{
  DMPolytopeType *typesTmp = NULL;
  PetscInt       *sizesTmp = NULL, *facesTmp = NULL;
  PetscInt       *tmp;
  PetscInt        maxConeSize, maxSupportSize, maxSize;
  PetscInt        getSize = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cone) PetscAssertPointer(cone, 3);
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
  maxSize = PetscMax(maxConeSize, maxSupportSize);
  if (faceTypes) getSize += maxSize;
  if (faceSizes) getSize += maxSize;
  if (faces) getSize += PetscSqr(maxSize);
  PetscCall(DMGetWorkArray(dm, getSize, MPIU_INT, &tmp));
  if (faceTypes) {
    typesTmp = (DMPolytopeType *)tmp;
    tmp += maxSize;
  }
  if (faceSizes) {
    sizesTmp = tmp;
    tmp += maxSize;
  }
  if (faces) facesTmp = tmp;
  switch (ct) {
  case DM_POLYTOPE_POINT:
    if (numFaces) *numFaces = 0;
    if (faceTypes) *faceTypes = typesTmp;
    if (faceSizes) *faceSizes = sizesTmp;
    if (faces) *faces = facesTmp;
    break;
  case DM_POLYTOPE_SEGMENT:
    if (numFaces) *numFaces = 2;
    if (faceTypes) {
      typesTmp[0] = DM_POLYTOPE_POINT;
      typesTmp[1] = DM_POLYTOPE_POINT;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 1;
      sizesTmp[1] = 1;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0] = cone[0];
      facesTmp[1] = cone[1];
      *faces      = facesTmp;
    }
    break;
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
    if (numFaces) *numFaces = 2;
    if (faceTypes) {
      typesTmp[0] = DM_POLYTOPE_POINT;
      typesTmp[1] = DM_POLYTOPE_POINT;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 1;
      sizesTmp[1] = 1;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0] = cone[0];
      facesTmp[1] = cone[1];
      *faces      = facesTmp;
    }
    break;
  case DM_POLYTOPE_TRIANGLE:
    if (numFaces) *numFaces = 3;
    if (faceTypes) {
      typesTmp[0] = DM_POLYTOPE_SEGMENT;
      typesTmp[1] = DM_POLYTOPE_SEGMENT;
      typesTmp[2] = DM_POLYTOPE_SEGMENT;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 2;
      sizesTmp[1] = 2;
      sizesTmp[2] = 2;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0] = cone[0];
      facesTmp[1] = cone[1];
      facesTmp[2] = cone[1];
      facesTmp[3] = cone[2];
      facesTmp[4] = cone[2];
      facesTmp[5] = cone[0];
      *faces      = facesTmp;
    }
    break;
  case DM_POLYTOPE_QUADRILATERAL:
    /* Vertices follow right hand rule */
    if (numFaces) *numFaces = 4;
    if (faceTypes) {
      typesTmp[0] = DM_POLYTOPE_SEGMENT;
      typesTmp[1] = DM_POLYTOPE_SEGMENT;
      typesTmp[2] = DM_POLYTOPE_SEGMENT;
      typesTmp[3] = DM_POLYTOPE_SEGMENT;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 2;
      sizesTmp[1] = 2;
      sizesTmp[2] = 2;
      sizesTmp[3] = 2;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0] = cone[0];
      facesTmp[1] = cone[1];
      facesTmp[2] = cone[1];
      facesTmp[3] = cone[2];
      facesTmp[4] = cone[2];
      facesTmp[5] = cone[3];
      facesTmp[6] = cone[3];
      facesTmp[7] = cone[0];
      *faces      = facesTmp;
    }
    break;
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
    if (numFaces) *numFaces = 4;
    if (faceTypes) {
      typesTmp[0] = DM_POLYTOPE_SEGMENT;
      typesTmp[1] = DM_POLYTOPE_SEGMENT;
      typesTmp[2] = DM_POLYTOPE_POINT_PRISM_TENSOR;
      typesTmp[3] = DM_POLYTOPE_POINT_PRISM_TENSOR;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 2;
      sizesTmp[1] = 2;
      sizesTmp[2] = 2;
      sizesTmp[3] = 2;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0] = cone[0];
      facesTmp[1] = cone[1];
      facesTmp[2] = cone[2];
      facesTmp[3] = cone[3];
      facesTmp[4] = cone[0];
      facesTmp[5] = cone[2];
      facesTmp[6] = cone[1];
      facesTmp[7] = cone[3];
      *faces      = facesTmp;
    }
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    /* Vertices of first face follow right hand rule and normal points away from last vertex */
    if (numFaces) *numFaces = 4;
    if (faceTypes) {
      typesTmp[0] = DM_POLYTOPE_TRIANGLE;
      typesTmp[1] = DM_POLYTOPE_TRIANGLE;
      typesTmp[2] = DM_POLYTOPE_TRIANGLE;
      typesTmp[3] = DM_POLYTOPE_TRIANGLE;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 3;
      sizesTmp[1] = 3;
      sizesTmp[2] = 3;
      sizesTmp[3] = 3;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0]  = cone[0];
      facesTmp[1]  = cone[1];
      facesTmp[2]  = cone[2];
      facesTmp[3]  = cone[0];
      facesTmp[4]  = cone[3];
      facesTmp[5]  = cone[1];
      facesTmp[6]  = cone[0];
      facesTmp[7]  = cone[2];
      facesTmp[8]  = cone[3];
      facesTmp[9]  = cone[2];
      facesTmp[10] = cone[1];
      facesTmp[11] = cone[3];
      *faces       = facesTmp;
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
      typesTmp[0] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[1] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[2] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[3] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[4] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[5] = DM_POLYTOPE_QUADRILATERAL;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 4;
      sizesTmp[1] = 4;
      sizesTmp[2] = 4;
      sizesTmp[3] = 4;
      sizesTmp[4] = 4;
      sizesTmp[5] = 4;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0]  = cone[0];
      facesTmp[1]  = cone[1];
      facesTmp[2]  = cone[2];
      facesTmp[3]  = cone[3]; /* Bottom */
      facesTmp[4]  = cone[4];
      facesTmp[5]  = cone[5];
      facesTmp[6]  = cone[6];
      facesTmp[7]  = cone[7]; /* Top */
      facesTmp[8]  = cone[0];
      facesTmp[9]  = cone[3];
      facesTmp[10] = cone[5];
      facesTmp[11] = cone[4]; /* Front */
      facesTmp[12] = cone[2];
      facesTmp[13] = cone[1];
      facesTmp[14] = cone[7];
      facesTmp[15] = cone[6]; /* Back */
      facesTmp[16] = cone[3];
      facesTmp[17] = cone[2];
      facesTmp[18] = cone[6];
      facesTmp[19] = cone[5]; /* Right */
      facesTmp[20] = cone[0];
      facesTmp[21] = cone[4];
      facesTmp[22] = cone[7];
      facesTmp[23] = cone[1]; /* Left */
      *faces       = facesTmp;
    }
    break;
  case DM_POLYTOPE_TRI_PRISM:
    if (numFaces) *numFaces = 5;
    if (faceTypes) {
      typesTmp[0] = DM_POLYTOPE_TRIANGLE;
      typesTmp[1] = DM_POLYTOPE_TRIANGLE;
      typesTmp[2] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[3] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[4] = DM_POLYTOPE_QUADRILATERAL;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 3;
      sizesTmp[1] = 3;
      sizesTmp[2] = 4;
      sizesTmp[3] = 4;
      sizesTmp[4] = 4;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0]  = cone[0];
      facesTmp[1]  = cone[1];
      facesTmp[2]  = cone[2]; /* Bottom */
      facesTmp[3]  = cone[3];
      facesTmp[4]  = cone[4];
      facesTmp[5]  = cone[5]; /* Top */
      facesTmp[6]  = cone[0];
      facesTmp[7]  = cone[2];
      facesTmp[8]  = cone[4];
      facesTmp[9]  = cone[3]; /* Back left */
      facesTmp[10] = cone[2];
      facesTmp[11] = cone[1];
      facesTmp[12] = cone[5];
      facesTmp[13] = cone[4]; /* Front */
      facesTmp[14] = cone[1];
      facesTmp[15] = cone[0];
      facesTmp[16] = cone[3];
      facesTmp[17] = cone[5]; /* Back right */
      *faces       = facesTmp;
    }
    break;
  case DM_POLYTOPE_TRI_PRISM_TENSOR:
    if (numFaces) *numFaces = 5;
    if (faceTypes) {
      typesTmp[0] = DM_POLYTOPE_TRIANGLE;
      typesTmp[1] = DM_POLYTOPE_TRIANGLE;
      typesTmp[2] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      typesTmp[3] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      typesTmp[4] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 3;
      sizesTmp[1] = 3;
      sizesTmp[2] = 4;
      sizesTmp[3] = 4;
      sizesTmp[4] = 4;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0]  = cone[0];
      facesTmp[1]  = cone[1];
      facesTmp[2]  = cone[2]; /* Bottom */
      facesTmp[3]  = cone[3];
      facesTmp[4]  = cone[4];
      facesTmp[5]  = cone[5]; /* Top */
      facesTmp[6]  = cone[0];
      facesTmp[7]  = cone[1];
      facesTmp[8]  = cone[3];
      facesTmp[9]  = cone[4]; /* Back left */
      facesTmp[10] = cone[1];
      facesTmp[11] = cone[2];
      facesTmp[12] = cone[4];
      facesTmp[13] = cone[5]; /* Back right */
      facesTmp[14] = cone[2];
      facesTmp[15] = cone[0];
      facesTmp[16] = cone[5];
      facesTmp[17] = cone[3]; /* Front */
      *faces       = facesTmp;
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
      typesTmp[0] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[1] = DM_POLYTOPE_QUADRILATERAL;
      typesTmp[2] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      typesTmp[3] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      typesTmp[4] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      typesTmp[5] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 4;
      sizesTmp[1] = 4;
      sizesTmp[2] = 4;
      sizesTmp[3] = 4;
      sizesTmp[4] = 4;
      sizesTmp[5] = 4;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0]  = cone[0];
      facesTmp[1]  = cone[1];
      facesTmp[2]  = cone[2];
      facesTmp[3]  = cone[3]; /* Bottom */
      facesTmp[4]  = cone[4];
      facesTmp[5]  = cone[5];
      facesTmp[6]  = cone[6];
      facesTmp[7]  = cone[7]; /* Top */
      facesTmp[8]  = cone[0];
      facesTmp[9]  = cone[1];
      facesTmp[10] = cone[4];
      facesTmp[11] = cone[5]; /* Front */
      facesTmp[12] = cone[1];
      facesTmp[13] = cone[2];
      facesTmp[14] = cone[5];
      facesTmp[15] = cone[6]; /* Right */
      facesTmp[16] = cone[2];
      facesTmp[17] = cone[3];
      facesTmp[18] = cone[6];
      facesTmp[19] = cone[7]; /* Back */
      facesTmp[20] = cone[3];
      facesTmp[21] = cone[0];
      facesTmp[22] = cone[7];
      facesTmp[23] = cone[4]; /* Left */
      *faces       = facesTmp;
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
      typesTmp[1] = DM_POLYTOPE_TRIANGLE;
      typesTmp[2] = DM_POLYTOPE_TRIANGLE;
      typesTmp[3] = DM_POLYTOPE_TRIANGLE;
      typesTmp[4] = DM_POLYTOPE_TRIANGLE;
      *faceTypes  = typesTmp;
    }
    if (faceSizes) {
      sizesTmp[0] = 4;
      sizesTmp[1] = 3;
      sizesTmp[2] = 3;
      sizesTmp[3] = 3;
      sizesTmp[4] = 3;
      *faceSizes  = sizesTmp;
    }
    if (faces) {
      facesTmp[0]  = cone[0];
      facesTmp[1]  = cone[1];
      facesTmp[2]  = cone[2];
      facesTmp[3]  = cone[3]; /* Bottom */
      facesTmp[4]  = cone[0];
      facesTmp[5]  = cone[3];
      facesTmp[6]  = cone[4]; /* Front */
      facesTmp[7]  = cone[3];
      facesTmp[8]  = cone[2];
      facesTmp[9]  = cone[4]; /* Right */
      facesTmp[10] = cone[2];
      facesTmp[11] = cone[1];
      facesTmp[12] = cone[4]; /* Back */
      facesTmp[13] = cone[1];
      facesTmp[14] = cone[0];
      facesTmp[15] = cone[4]; /* Left */
      *faces       = facesTmp;
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No face description for cell type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexRestoreRawFaces_Internal(DM dm, DMPolytopeType ct, const PetscInt cone[], PetscInt *numFaces, const DMPolytopeType *faceTypes[], const PetscInt *faceSizes[], const PetscInt *faces[])
{
  PetscFunctionBegin;
  if (faceTypes) PetscCall(DMRestoreWorkArray(dm, 0, MPIU_INT, (void *)faceTypes));
  else if (faceSizes) PetscCall(DMRestoreWorkArray(dm, 0, MPIU_INT, (void *)faceSizes));
  else if (faces) PetscCall(DMRestoreWorkArray(dm, 0, MPIU_INT, (void *)faces));
  if (faceTypes) *faceTypes = NULL;
  if (faceSizes) *faceSizes = NULL;
  if (faces) *faces = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This interpolates faces for cells at some stratum */
static PetscErrorCode DMPlexInterpolateFaces_Internal(DM dm, PetscInt cellDepth, DM idm)
{
  DMLabel       ctLabel;
  PetscHMapIJKL faceTable;
  PetscInt      faceTypeNum[DM_NUM_POLYTOPES];
  PetscInt      depth, pStart, Np, cStart, cEnd, fStart, fEnd, vStart, vEnd;
  PetscInt      cntFaces, *facesId, minCone;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(PetscHMapIJKLCreate(&faceTable));
  PetscCall(PetscArrayzero(faceTypeNum, DM_NUM_POLYTOPES));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetDepthStratum(dm, cellDepth, &cStart, &cEnd));
  // If the range incorporates the vertices, it means we have a non-manifold topology, so choose just cells
  if (cStart <= vStart && cEnd >= vEnd) cEnd = vStart;
  // Number new faces and save face vertices in hash table
  //   If depth > cellDepth, meaning we are interpolating faces, put the new (d-1)-faces after them
  //   otherwise, we are interpolating cells, so put the faces after the vertices
  PetscCall(DMPlexGetDepthStratum(dm, depth > cellDepth ? cellDepth : 0, NULL, &fStart));
  fEnd = fStart;

  minCone  = PETSC_INT_MAX;
  cntFaces = 0;
  for (PetscInt c = cStart; c < cEnd; ++c) {
    const PetscInt *cone;
    DMPolytopeType  ct;
    PetscInt        numFaces = 0, coneSize;

    PetscCall(DMPlexGetCellType(dm, c, &ct));
    PetscCall(DMPlexGetCone(dm, c, &cone));
    PetscCall(DMPlexGetConeSize(dm, c, &coneSize));
    for (PetscInt j = 0; j < coneSize; j++) minCone = PetscMin(cone[j], minCone);
    // Ignore faces since they are interpolated
    if (ct != DM_POLYTOPE_SEGMENT && ct != DM_POLYTOPE_POINT_PRISM_TENSOR) PetscCall(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, NULL, NULL, NULL));
    cntFaces += numFaces;
  }
  // Encode so that we can use 0 as an excluded value, instead of PETSC_INT_MAX
  minCone = -(minCone - 1);

  PetscCall(PetscMalloc1(cntFaces, &facesId));

  cntFaces = 0;
  for (PetscInt c = cStart; c < cEnd; ++c) {
    const PetscInt       *cone, *faceSizes, *faces;
    const DMPolytopeType *faceTypes;
    DMPolytopeType        ct;
    PetscInt              numFaces, foff = 0;

    PetscCall(DMPlexGetCellType(dm, c, &ct));
    PetscCall(DMPlexGetCone(dm, c, &cone));
    // Ignore faces since they are interpolated
    if (ct != DM_POLYTOPE_SEGMENT && ct != DM_POLYTOPE_POINT_PRISM_TENSOR) {
      PetscCall(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
    } else {
      numFaces = 0;
    }
    for (PetscInt cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
      const PetscInt       faceSize = faceSizes[cf];
      const DMPolytopeType faceType = faceTypes[cf];
      const PetscInt      *face     = &faces[foff];
      PetscHashIJKLKey     key;
      PetscHashIter        iter;
      PetscBool            missing;

      PetscCheck(faceSize <= 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not support faces of size %" PetscInt_FMT " > 4", faceSize);
      key.i = face[0] + minCone;
      key.j = faceSize > 1 ? face[1] + minCone : 0;
      key.k = faceSize > 2 ? face[2] + minCone : 0;
      key.l = faceSize > 3 ? face[3] + minCone : 0;
      PetscCall(PetscSortInt(faceSize, (PetscInt *)&key));
      PetscCall(PetscHMapIJKLPut(faceTable, key, &iter, &missing));
      if (missing) {
        facesId[cntFaces] = fEnd;
        PetscCall(PetscHMapIJKLIterSet(faceTable, iter, fEnd++));
        ++faceTypeNum[faceType];
      } else PetscCall(PetscHMapIJKLIterGet(faceTable, iter, &facesId[cntFaces]));
      cntFaces++;
    }
    if (ct != DM_POLYTOPE_SEGMENT && ct != DM_POLYTOPE_POINT_PRISM_TENSOR) PetscCall(DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
  }
  /* We need to number faces contiguously among types */
  {
    PetscInt faceTypeStart[DM_NUM_POLYTOPES], ct, numFT = 0;

    for (ct = 0; ct < DM_NUM_POLYTOPES; ++ct) {
      if (faceTypeNum[ct]) ++numFT;
      faceTypeStart[ct] = 0;
    }
    if (numFT > 1) {
      PetscCall(PetscHMapIJKLClear(faceTable));
      faceTypeStart[0] = fStart;
      for (ct = 1; ct < DM_NUM_POLYTOPES; ++ct) faceTypeStart[ct] = faceTypeStart[ct - 1] + faceTypeNum[ct - 1];
      cntFaces = 0;
      for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscInt       *cone, *faceSizes, *faces;
        const DMPolytopeType *faceTypes;
        DMPolytopeType        ct;
        PetscInt              numFaces, foff = 0;

        PetscCall(DMPlexGetCellType(dm, c, &ct));
        PetscCall(DMPlexGetCone(dm, c, &cone));
        if (ct != DM_POLYTOPE_SEGMENT && ct != DM_POLYTOPE_POINT_PRISM_TENSOR) {
          PetscCall(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
        } else {
          numFaces = 0;
        }
        for (PetscInt cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
          const PetscInt       faceSize = faceSizes[cf];
          const DMPolytopeType faceType = faceTypes[cf];
          const PetscInt      *face     = &faces[foff];
          PetscHashIJKLKey     key;
          PetscHashIter        iter;
          PetscBool            missing;

          key.i = face[0] + minCone;
          key.j = faceSize > 1 ? face[1] + minCone : 0;
          key.k = faceSize > 2 ? face[2] + minCone : 0;
          key.l = faceSize > 3 ? face[3] + minCone : 0;
          PetscCall(PetscSortInt(faceSize, (PetscInt *)&key));
          PetscCall(PetscHMapIJKLPut(faceTable, key, &iter, &missing));
          if (missing) {
            facesId[cntFaces] = faceTypeStart[faceType];
            PetscCall(PetscHMapIJKLIterSet(faceTable, iter, faceTypeStart[faceType]++));
          } else PetscCall(PetscHMapIJKLIterGet(faceTable, iter, &facesId[cntFaces]));
          cntFaces++;
        }
        if (ct != DM_POLYTOPE_SEGMENT && ct != DM_POLYTOPE_POINT_PRISM_TENSOR) PetscCall(DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
      }
      for (ct = 1; ct < DM_NUM_POLYTOPES; ++ct) {
        PetscCheck(faceTypeStart[ct] == faceTypeStart[ct - 1] + faceTypeNum[ct], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent numbering for cell type %s, %" PetscInt_FMT " != %" PetscInt_FMT " + %" PetscInt_FMT, DMPolytopeTypes[ct], faceTypeStart[ct], faceTypeStart[ct - 1], faceTypeNum[ct]);
      }
    }
  }
  PetscCall(PetscHMapIJKLDestroy(&faceTable));

  // Add new points, perhaps inserting into the numbering
  PetscCall(DMPlexGetChart(dm, &pStart, &Np));
  PetscCall(DMPlexSetChart(idm, pStart, Np + (fEnd - fStart)));
  // Set cone sizes
  //   Must create the celltype label here so that we do not automatically try to compute the types
  PetscCall(DMCreateLabel(idm, "celltype"));
  PetscCall(DMPlexGetCellTypeLabel(idm, &ctLabel));
  for (PetscInt d = 0; d <= depth; ++d) {
    DMPolytopeType ct;
    PetscInt       coneSize, pStart, pEnd, poff = 0;

    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    // Check for non-manifold condition
    if (d == cellDepth) {
      if (pEnd == cEnd) continue;
      else pStart = vEnd;
    }
    // Account for insertion
    if (pStart >= fStart) poff = fEnd - fStart;
    for (PetscInt p = pStart; p < pEnd; ++p) {
      PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
      PetscCall(DMPlexSetConeSize(idm, p + poff, coneSize));
      PetscCall(DMPlexGetCellType(dm, p, &ct));
      PetscCall(DMPlexSetCellType(idm, p + poff, ct));
    }
  }
  cntFaces = 0;
  for (PetscInt c = cStart; c < cEnd; ++c) {
    const PetscInt       *cone, *faceSizes;
    const DMPolytopeType *faceTypes;
    DMPolytopeType        ct;
    PetscInt              numFaces, poff = 0;

    PetscCall(DMPlexGetCellType(dm, c, &ct));
    PetscCall(DMPlexGetCone(dm, c, &cone));
    if (c >= fStart) poff = fEnd - fStart;
    if (ct == DM_POLYTOPE_SEGMENT || ct == DM_POLYTOPE_POINT_PRISM_TENSOR) {
      PetscCall(DMPlexSetCellType(idm, c + poff, ct));
      PetscCall(DMPlexSetConeSize(idm, c + poff, 2));
      continue;
    }
    PetscCall(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, NULL));
    PetscCall(DMPlexSetCellType(idm, c + poff, ct));
    PetscCall(DMPlexSetConeSize(idm, c + poff, numFaces));
    for (PetscInt cf = 0; cf < numFaces; ++cf) {
      const PetscInt f        = facesId[cntFaces];
      DMPolytopeType faceType = faceTypes[cf];
      const PetscInt faceSize = faceSizes[cf];
      PetscCall(DMPlexSetConeSize(idm, f, faceSize));
      PetscCall(DMPlexSetCellType(idm, f, faceType));
      cntFaces++;
    }
    PetscCall(DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, NULL));
  }
  PetscCall(DMSetUp(idm));
  // Initialize cones so we do not need the bash table to tell us that a cone has been set
  {
    PetscSection cs;
    PetscInt    *cones, csize;

    PetscCall(DMPlexGetConeSection(idm, &cs));
    PetscCall(DMPlexGetCones(idm, &cones));
    PetscCall(PetscSectionGetStorageSize(cs, &csize));
    for (PetscInt c = 0; c < csize; ++c) cones[c] = -1;
  }
  // Set cones
  {
    PetscInt *icone;
    PetscInt  maxConeSize;

    PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, NULL));
    PetscCall(PetscMalloc1(maxConeSize, &icone));
    for (PetscInt d = 0; d <= depth; ++d) {
      const PetscInt *cone;
      PetscInt        pStart, pEnd, poff = 0, coneSize;

      PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
      // Check for non-manifold condition
      if (d == cellDepth) {
        if (pEnd == cEnd) continue;
        else pStart = vEnd;
      }
      // Account for insertion
      if (pStart >= fStart) poff = fEnd - fStart;
      for (PetscInt p = pStart; p < pEnd; ++p) {
        PetscCall(DMPlexGetCone(dm, p, &cone));
        PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
        for (PetscInt cp = 0; cp < coneSize; ++cp) icone[cp] = cone[cp] + (cone[cp] >= fStart ? fEnd - fStart : 0);
        PetscCall(DMPlexSetCone(idm, p + poff, icone));
        PetscCall(DMPlexGetConeOrientation(dm, p, &cone));
        PetscCall(DMPlexSetConeOrientation(idm, p + poff, cone));
      }
    }
    cntFaces = 0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
      const PetscInt       *cone, *faceSizes, *faces;
      const DMPolytopeType *faceTypes;
      DMPolytopeType        ct;
      PetscInt              coneSize, numFaces, foff = 0, poff = 0;

      PetscCall(DMPlexGetCellType(dm, c, &ct));
      PetscCall(DMPlexGetCone(dm, c, &cone));
      PetscCall(DMPlexGetConeSize(dm, c, &coneSize));
      if (c >= fStart) poff = fEnd - fStart;
      if (ct == DM_POLYTOPE_SEGMENT || ct == DM_POLYTOPE_POINT_PRISM_TENSOR) {
        for (PetscInt cp = 0; cp < coneSize; ++cp) icone[cp] = cone[cp] + (cone[cp] >= fStart ? fEnd - fStart : 0);
        PetscCall(DMPlexSetCone(idm, c + poff, icone));
        PetscCall(DMPlexGetConeOrientation(dm, c, &cone));
        PetscCall(DMPlexSetConeOrientation(idm, c + poff, cone));
        continue;
      }
      PetscCall(DMPlexGetRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
      for (PetscInt cf = 0; cf < numFaces; foff += faceSizes[cf], ++cf) {
        DMPolytopeType  faceType = faceTypes[cf];
        const PetscInt  faceSize = faceSizes[cf];
        const PetscInt  f        = facesId[cntFaces];
        const PetscInt *face     = &faces[foff];
        const PetscInt *fcone;

        PetscCall(DMPlexInsertCone(idm, c, cf, f));
        PetscCall(DMPlexGetCone(idm, f, &fcone));
        if (fcone[0] < 0) PetscCall(DMPlexSetCone(idm, f, face));
        {
          const PetscInt *fcone2;
          PetscInt        ornt;

          PetscCall(DMPlexGetConeSize(idm, f, &coneSize));
          PetscCall(DMPlexGetCone(idm, f, &fcone2));
          PetscCheck(coneSize == faceSize, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of face vertices %" PetscInt_FMT " for face %" PetscInt_FMT " should be %" PetscInt_FMT, coneSize, f, faceSize);
          /* Notice that we have to use vertices here because the lower dimensional faces have not been created yet */
          PetscCall(DMPolytopeGetVertexOrientation(faceType, fcone2, face, &ornt));
          PetscCall(DMPlexInsertConeOrientation(idm, c + poff, cf, ornt));
        }
        cntFaces++;
      }
      PetscCall(DMPlexRestoreRawFaces_Internal(dm, ct, cone, &numFaces, &faceTypes, &faceSizes, &faces));
    }
    PetscCall(PetscFree(icone));
  }
  PetscCall(PetscFree(facesId));
  PetscCall(DMPlexSymmetrize(idm));
  PetscCall(DMPlexStratify(idm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SortRmineRremoteByRemote_Private(PetscSF sf, PetscInt *rmine1[], PetscInt *rremote1[])
{
  PetscInt           nleaves;
  PetscMPIInt        nranks;
  const PetscMPIInt *ranks   = NULL;
  const PetscInt    *roffset = NULL, *rmine = NULL, *rremote = NULL;
  PetscInt           n, o;

  PetscFunctionBegin;
  PetscCall(PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, &rmine, &rremote));
  nleaves = roffset[nranks];
  PetscCall(PetscMalloc2(nleaves, rmine1, nleaves, rremote1));
  for (PetscMPIInt r = 0; r < nranks; r++) {
    /* simultaneously sort rank-wise portions of rmine & rremote by values in rremote
       - to unify order with the other side */
    o = roffset[r];
    n = roffset[r + 1] - o;
    PetscCall(PetscArraycpy(&(*rmine1)[o], &rmine[o], n));
    PetscCall(PetscArraycpy(&(*rremote1)[o], &rremote[o], n));
    PetscCall(PetscSortIntWithArray(n, &(*rremote1)[o], &(*rmine1)[o]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexOrientInterface_Internal(DM dm)
{
  PetscSF            sf;
  const PetscInt    *locals;
  const PetscSFNode *remotes;
  const PetscMPIInt *ranks;
  const PetscInt    *roffset;
  PetscInt          *rmine1, *rremote1; /* rmine and rremote copies simultaneously sorted by rank and rremote */
  PetscInt           nroots, p, nleaves, maxConeSize = 0;
  PetscMPIInt        nranks, r;
  PetscInt(*roots)[4], (*leaves)[4], mainCone[4];
  PetscMPIInt(*rootsRanks)[4], (*leavesRanks)[4];
  MPI_Comm    comm;
  PetscMPIInt rank, size;
  PetscInt    debug = 0;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(DMViewFromOptions(dm, NULL, "-before_orient_interface_dm_view"));
  if (PetscDefined(USE_DEBUG)) PetscCall(DMPlexCheckPointSF(dm, sf, PETSC_FALSE));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &locals, &remotes));
  if (nroots < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSFSetUp(sf));
  PetscCall(SortRmineRremoteByRemote_Private(sf, &rmine1, &rremote1));
  for (p = 0; p < nleaves; ++p) {
    PetscInt coneSize;
    PetscCall(DMPlexGetConeSize(dm, locals[p], &coneSize));
    maxConeSize = PetscMax(maxConeSize, coneSize);
  }
  PetscCheck(maxConeSize <= 4, comm, PETSC_ERR_SUP, "This method does not support cones of size %" PetscInt_FMT, maxConeSize);
  PetscCall(PetscMalloc4(nroots, &roots, nroots, &leaves, nroots, &rootsRanks, nroots, &leavesRanks));
  for (p = 0; p < nroots; ++p) {
    const PetscInt *cone;
    PetscInt        coneSize, c, ind0;

    PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
    PetscCall(DMPlexGetCone(dm, p, &cone));
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
      PetscCall(PetscFindInt(cone[c], nleaves, locals, &ind0));
      if (ind0 < 0) {
        roots[p][c]      = cone[c];
        rootsRanks[p][c] = rank;
      } else {
        roots[p][c]      = remotes[ind0].index;
        rootsRanks[p][c] = (PetscMPIInt)remotes[ind0].rank;
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
  PetscCall(PetscSFBcastBegin(sf, MPIU_4INT, roots, leaves, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(sf, MPI_4INT, rootsRanks, leavesRanks, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_4INT, roots, leaves, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPI_4INT, rootsRanks, leavesRanks, MPI_REPLACE));
  if (debug) {
    PetscCall(PetscSynchronizedFlush(comm, NULL));
    if (rank == 0) PetscCall(PetscSynchronizedPrintf(comm, "Referenced roots\n"));
  }
  PetscCall(PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, NULL, NULL));
  for (p = 0; p < nroots; ++p) {
    DMPolytopeType  ct;
    const PetscInt *cone;
    PetscInt        coneSize, c, ind0, o, ir;

    if (leaves[p][0] < 0) continue; /* Ignore vertices */
    PetscCall(DMPlexGetCellType(dm, p, &ct));
    PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
    PetscCall(DMPlexGetCone(dm, p, &cone));
    if (debug) {
      PetscCall(PetscSynchronizedPrintf(comm, "[%d]  %4" PetscInt_FMT ": cone=[%4" PetscInt_FMT " %4" PetscInt_FMT " %4" PetscInt_FMT " %4" PetscInt_FMT "] roots=[(%d,%4" PetscInt_FMT ") (%d,%4" PetscInt_FMT ") (%d,%4" PetscInt_FMT ") (%d,%4" PetscInt_FMT ")] leaves=[(%d,%4" PetscInt_FMT ") (%d,%4" PetscInt_FMT ") (%d,%4" PetscInt_FMT ") (%d,%4" PetscInt_FMT ")]", rank, p, cone[0], cone[1], cone[2], cone[3], rootsRanks[p][0], roots[p][0], rootsRanks[p][1], roots[p][1], rootsRanks[p][2], roots[p][2], rootsRanks[p][3], roots[p][3], leavesRanks[p][0], leaves[p][0], leavesRanks[p][1], leaves[p][1], leavesRanks[p][2], leaves[p][2], leavesRanks[p][3], leaves[p][3]));
    }
    if (leavesRanks[p][0] != rootsRanks[p][0] || leaves[p][0] != roots[p][0] || leavesRanks[p][1] != rootsRanks[p][1] || leaves[p][1] != roots[p][1] || leavesRanks[p][2] != rootsRanks[p][2] || leaves[p][2] != roots[p][2] || leavesRanks[p][3] != rootsRanks[p][3] || leaves[p][3] != roots[p][3]) {
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
        PetscCall(PetscFindMPIInt((PetscMPIInt)leavesRanks[p][c], nranks, ranks, &ir));
        PetscCall(PetscMPIIntCast(ir, &r));
        PetscCheck(r >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " cone[%" PetscInt_FMT "]=%" PetscInt_FMT " root (%d,%" PetscInt_FMT ") leaf (%d,%" PetscInt_FMT "): leaf rank not found among remote ranks", p, c, cone[c], rootsRanks[p][c], roots[p][c], leavesRanks[p][c], leaves[p][c]);
        PetscCheck(ranks[r] >= 0 && ranks[r] < size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "p=%" PetscInt_FMT " c=%" PetscInt_FMT " commsize=%d: ranks[%d] = %d makes no sense", p, c, size, r, ranks[r]);
        /* Find point leaves[p][c] among remote points aimed at rank leavesRanks[p][c] */
        rS = roffset[r];
        rN = roffset[r + 1] - rS;
        PetscCall(PetscFindInt(leaves[p][c], rN, &rremote1[rS], &ind0));
        PetscCheck(ind0 >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " cone[%" PetscInt_FMT "]=%" PetscInt_FMT " root (%d,%" PetscInt_FMT ") leave (%d,%" PetscInt_FMT "): corresponding remote point not found - it seems there is missing connection in point SF!", p, c, cone[c], rootsRanks[p][c], roots[p][c], leavesRanks[p][c], leaves[p][c]);
        /* Get the corresponding local point */
        mainCone[c] = rmine1[rS + ind0];
      }
      if (debug) PetscCall(PetscSynchronizedPrintf(comm, " mainCone=[%4" PetscInt_FMT " %4" PetscInt_FMT " %4" PetscInt_FMT " %4" PetscInt_FMT "]\n", mainCone[0], mainCone[1], mainCone[2], mainCone[3]));
      /* Set the desired order of p's cone points and fix orientations accordingly */
      PetscCall(DMPolytopeGetOrientation(ct, cone, mainCone, &o));
      PetscCall(DMPlexOrientPoint(dm, p, o));
    } else if (debug) PetscCall(PetscSynchronizedPrintf(comm, " ==\n"));
  }
  if (debug) {
    PetscCall(PetscSynchronizedFlush(comm, NULL));
    PetscCallMPI(MPI_Barrier(comm));
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-after_orient_interface_dm_view"));
  PetscCall(PetscFree4(roots, leaves, rootsRanks, leavesRanks));
  PetscCall(PetscFree2(rmine1, rremote1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IntArrayViewFromOptions(MPI_Comm comm, const char opt[], const char name[], const char idxname[], const char valname[], PetscInt n, const PetscInt a[])
{
  PetscInt    idx;
  PetscMPIInt rank;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHasName(NULL, NULL, opt, &flg));
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSynchronizedPrintf(comm, "[%d]%s:\n", rank, name));
  for (idx = 0; idx < n; ++idx) PetscCall(PetscSynchronizedPrintf(comm, "[%d]%s %" PetscInt_FMT " %s %" PetscInt_FMT "\n", rank, idxname, idx, valname, a[idx]));
  PetscCall(PetscSynchronizedFlush(comm, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SFNodeArrayViewFromOptions(MPI_Comm comm, const char opt[], const char name[], const char idxname[], PetscInt n, const PetscSFNode a[])
{
  PetscInt    idx;
  PetscMPIInt rank;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHasName(NULL, NULL, opt, &flg));
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSynchronizedPrintf(comm, "[%d]%s:\n", rank, name));
  if (idxname) {
    for (idx = 0; idx < n; ++idx) PetscCall(PetscSynchronizedPrintf(comm, "[%d]%s %" PetscInt_FMT " rank %d index %" PetscInt_FMT "\n", rank, idxname, idx, (PetscMPIInt)a[idx].rank, a[idx].index));
  } else {
    for (idx = 0; idx < n; ++idx) PetscCall(PetscSynchronizedPrintf(comm, "[%d]rank %d index %" PetscInt_FMT "\n", rank, (PetscMPIInt)a[idx].rank, a[idx].index));
  }
  PetscCall(PetscSynchronizedFlush(comm, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexMapToLocalPoint(DM dm, PetscHMapIJ remotehash, PetscSFNode remotePoint, PetscInt *localPoint, PetscBool *mapFailed)
{
  PetscSF         sf;
  const PetscInt *locals;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, NULL, &locals, NULL));
  if (mapFailed) *mapFailed = PETSC_FALSE;
  if (remotePoint.rank == rank) {
    *localPoint = remotePoint.index;
  } else {
    PetscHashIJKey key;
    PetscInt       l;

    key.i = remotePoint.index;
    key.j = remotePoint.rank;
    PetscCall(PetscHMapIJGet(remotehash, key, &l));
    if (l >= 0) {
      *localPoint = locals[l];
    } else if (mapFailed) *mapFailed = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &locals, &remotes));
  if (Nl < 0) goto owned;
  PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegree));
  PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegree));
  if (rootdegree[localPoint]) goto owned;
  PetscCall(PetscFindInt(localPoint, Nl, locals, &l));
  if (l < 0) {
    if (mapFailed) *mapFailed = PETSC_TRUE;
  } else *remotePoint = remotes[l];
  PetscFunctionReturn(PETSC_SUCCESS);
owned:
  remotePoint->rank  = rank;
  remotePoint->index = localPoint;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexPointIsShared(DM dm, PetscInt p, PetscBool *isShared)
{
  PetscSF         sf;
  const PetscInt *locals, *rootdegree;
  PetscInt        Nl, idx;

  PetscFunctionBegin;
  *isShared = PETSC_FALSE;
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &locals, NULL));
  if (Nl < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFindInt(p, Nl, locals, &idx));
  if (idx >= 0) {
    *isShared = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegree));
  PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegree));
  if (rootdegree[p] > 0) *isShared = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexConeIsShared(DM dm, PetscInt p, PetscBool *isShared)
{
  const PetscInt *cone;
  PetscInt        coneSize, c;
  PetscBool       cShared = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
  PetscCall(DMPlexGetCone(dm, p, &cone));
  for (c = 0; c < coneSize; ++c) {
    PetscBool pointShared;

    PetscCall(DMPlexPointIsShared(dm, cone[c], &pointShared));
    cShared = (PetscBool)(cShared && pointShared);
  }
  *isShared = coneSize ? cShared : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexGetConeMinimum(DM dm, PetscInt p, PetscSFNode *cpmin)
{
  const PetscInt *cone;
  PetscInt        coneSize, c;
  PetscSFNode     cmin = {PETSC_INT_MAX, PETSC_MPI_INT_MAX}, missing = {-1, -1};

  PetscFunctionBegin;
  PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
  PetscCall(DMPlexGetCone(dm, p, &cone));
  for (c = 0; c < coneSize; ++c) {
    PetscSFNode rcp;
    PetscBool   mapFailed;

    PetscCall(DMPlexMapToGlobalPoint(dm, cone[c], &rcp, &mapFailed));
    if (mapFailed) {
      cmin = missing;
    } else {
      cmin = (rcp.rank < cmin.rank) || (rcp.rank == cmin.rank && rcp.index < cmin.index) ? rcp : cmin;
    }
  }
  *cpmin = coneSize ? cmin : missing;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMPlexGetOverlap(dm, &overlap));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetPointHeight(dm, p, &height));
  if (!overlap && height <= cellHeight + 1) {
    /* cells can't be shared for non-overlapping meshes */
    if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    Skipping face %" PetscInt_FMT " to avoid adding cell to hashmap since this is nonoverlapping mesh\n", rank, p));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
  PetscCall(DMPlexGetSupport(dm, p, &support));
  if (candidates) PetscCall(PetscSectionGetOffset(candidateSection, p, &off));
  for (s = 0; s < supportSize; ++s) {
    const PetscInt  face = support[s];
    const PetscInt *cone;
    PetscSFNode     cpmin = {-1, -1}, rp = {-1, -1};
    PetscInt        coneSize, c, f;
    PetscBool       isShared = PETSC_FALSE;
    PetscHashIJKey  key;

    /* Only add point once */
    if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    Support face %" PetscInt_FMT "\n", rank, face));
    key.i = p;
    key.j = face;
    PetscCall(PetscHMapIJGet(faceHash, key, &f));
    if (f >= 0) continue;
    PetscCall(DMPlexConeIsShared(dm, face, &isShared));
    PetscCall(DMPlexGetConeMinimum(dm, face, &cpmin));
    PetscCall(DMPlexMapToGlobalPoint(dm, p, &rp, NULL));
    if (debug) {
      PetscCall(PetscSynchronizedPrintf(comm, "[%d]      Face point %" PetscInt_FMT " is shared: %d\n", rank, face, (int)isShared));
      PetscCall(PetscSynchronizedPrintf(comm, "[%d]      Global point (%d, %" PetscInt_FMT ") Min Cone Point (%d, %" PetscInt_FMT ")\n", rank, (PetscMPIInt)rp.rank, rp.index, (PetscMPIInt)cpmin.rank, cpmin.index));
    }
    if (isShared && (rp.rank == cpmin.rank && rp.index == cpmin.index)) {
      PetscCall(PetscHMapIJSet(faceHash, key, p));
      if (candidates) {
        if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    Adding shared face %" PetscInt_FMT " at idx %" PetscInt_FMT "\n[%d]     ", rank, face, idx, rank));
        PetscCall(DMPlexGetConeSize(dm, face, &coneSize));
        PetscCall(DMPlexGetCone(dm, face, &cone));
        candidates[off + idx].rank    = -1;
        candidates[off + idx++].index = coneSize - 1;
        candidates[off + idx].rank    = rank;
        candidates[off + idx++].index = face;
        for (c = 0; c < coneSize; ++c) {
          const PetscInt cp = cone[c];

          if (cp == p) continue;
          PetscCall(DMPlexMapToGlobalPoint(dm, cp, &candidates[off + idx], NULL));
          if (debug) PetscCall(PetscSynchronizedPrintf(comm, " (%d,%" PetscInt_FMT ")", (PetscMPIInt)candidates[off + idx].rank, candidates[off + idx].index));
          ++idx;
        }
        if (debug) PetscCall(PetscSynchronizedPrintf(comm, "\n"));
      } else {
        /* Add cone size to section */
        if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    Scheduling shared face %" PetscInt_FMT "\n", rank, face));
        PetscCall(DMPlexGetConeSize(dm, face, &coneSize));
        PetscCall(PetscHMapIJSet(faceHash, key, p));
        PetscCall(PetscSectionAddDof(candidateSection, p, coneSize + 1));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexInterpolatePointSF - Insert interpolated points in the overlap into the `PointSF` in parallel, following local interpolation

  Collective

  Input Parameters:
+ dm      - The interpolated `DMPLEX`
- pointSF - The initial `PetscSF` without interpolated points

  Level: developer

  Note:
  Debugging for this process can be turned on with the options: `-dm_interp_pre_view` `-petscsf_interp_pre_view` `-petscsection_interp_candidate_view` `-petscsection_interp_candidate_remote_view` `-petscsection_interp_claim_view` `-petscsf_interp_pre_view` `-dmplex_interp_debug`

.seealso: `DMPLEX`, `DMPlexInterpolate()`, `DMPlexUninterpolate()`
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
  PetscCall(DMPlexIsDistributed(dm, &flg));
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);
  /* Set initial SF so that lower level queries work */
  PetscCall(DMSetPointSF(dm, pointSF));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMPlexGetOverlap(dm, &ov));
  PetscCheck(!ov, comm, PETSC_ERR_SUP, "Interpolation of overlapped DMPlex not implemented yet");
  PetscCall(PetscOptionsHasName(NULL, ((PetscObject)dm)->prefix, "-dmplex_interp_debug", &debug));
  PetscCall(PetscObjectViewFromOptions((PetscObject)dm, NULL, "-dm_interp_pre_view"));
  PetscCall(PetscObjectViewFromOptions((PetscObject)pointSF, NULL, "-petscsf_interp_pre_view"));
  PetscCall(PetscLogEventBegin(DMPLEX_InterpolateSF, dm, 0, 0, 0));
  /* Step 0: Precalculations */
  PetscCall(PetscSFGetGraph(pointSF, &Nr, &Nl, &localPoints, &remotePoints));
  PetscCheck(Nr >= 0, comm, PETSC_ERR_ARG_WRONGSTATE, "This DMPlex is distributed but input PointSF has no graph set");
  PetscCall(PetscHMapIJCreate(&remoteHash));
  for (l = 0; l < Nl; ++l) {
    PetscHashIJKey key;

    key.i = remotePoints[l].index;
    key.j = remotePoints[l].rank;
    PetscCall(PetscHMapIJSet(remoteHash, key, l));
  }
  /*   Compute root degree to identify shared points */
  PetscCall(PetscSFComputeDegreeBegin(pointSF, &rootdegree));
  PetscCall(PetscSFComputeDegreeEnd(pointSF, &rootdegree));
  PetscCall(IntArrayViewFromOptions(comm, "-interp_root_degree_view", "Root degree", "point", "degree", Nr, rootdegree));
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
       The array holds candidate shared faces, each face is referred to by the leaf point */
  PetscCall(PetscSectionCreate(comm, &candidateSection));
  PetscCall(PetscSectionSetChart(candidateSection, 0, Nr));
  {
    PetscHMapIJ faceHash;

    PetscCall(PetscHMapIJCreate(&faceHash));
    for (l = 0; l < Nl; ++l) {
      const PetscInt p = localPoints[l];

      if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]  First pass leaf point %" PetscInt_FMT "\n", rank, p));
      PetscCall(DMPlexAddSharedFace_Private(dm, candidateSection, NULL, faceHash, p, debug));
    }
    PetscCall(PetscHMapIJClear(faceHash));
    PetscCall(PetscSectionSetUp(candidateSection));
    PetscCall(PetscSectionGetStorageSize(candidateSection, &candidatesSize));
    PetscCall(PetscMalloc1(candidatesSize, &candidates));
    for (l = 0; l < Nl; ++l) {
      const PetscInt p = localPoints[l];

      if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]  Second pass leaf point %" PetscInt_FMT "\n", rank, p));
      PetscCall(DMPlexAddSharedFace_Private(dm, candidateSection, candidates, faceHash, p, debug));
    }
    PetscCall(PetscHMapIJDestroy(&faceHash));
    if (debug) PetscCall(PetscSynchronizedFlush(comm, NULL));
  }
  PetscCall(PetscObjectSetName((PetscObject)candidateSection, "Candidate Section"));
  PetscCall(PetscObjectViewFromOptions((PetscObject)candidateSection, NULL, "-petscsection_interp_candidate_view"));
  PetscCall(SFNodeArrayViewFromOptions(comm, "-petscsection_interp_candidate_view", "Candidates", NULL, candidatesSize, candidates));
  /* Step 2: Gather candidate section / array pair into the root partition via inverse(multi(pointSF)). */
  /*   Note that this section is indexed by offsets into leaves, not by point number */
  {
    PetscSF   sfMulti, sfInverse, sfCandidates;
    PetscInt *remoteOffsets;

    PetscCall(PetscSFGetMultiSF(pointSF, &sfMulti));
    PetscCall(PetscSFCreateInverseSF(sfMulti, &sfInverse));
    PetscCall(PetscSectionCreate(comm, &candidateRemoteSection));
    PetscCall(PetscSFDistributeSection(sfInverse, candidateSection, &remoteOffsets, candidateRemoteSection));
    PetscCall(PetscSFCreateSectionSF(sfInverse, candidateSection, remoteOffsets, candidateRemoteSection, &sfCandidates));
    PetscCall(PetscSectionGetStorageSize(candidateRemoteSection, &candidatesRemoteSize));
    PetscCall(PetscMalloc1(candidatesRemoteSize, &candidatesRemote));
    PetscCall(PetscSFBcastBegin(sfCandidates, MPIU_SF_NODE, candidates, candidatesRemote, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sfCandidates, MPIU_SF_NODE, candidates, candidatesRemote, MPI_REPLACE));
    PetscCall(PetscSFDestroy(&sfInverse));
    PetscCall(PetscSFDestroy(&sfCandidates));
    PetscCall(PetscFree(remoteOffsets));

    PetscCall(PetscObjectSetName((PetscObject)candidateRemoteSection, "Remote Candidate Section"));
    PetscCall(PetscObjectViewFromOptions((PetscObject)candidateRemoteSection, NULL, "-petscsection_interp_candidate_remote_view"));
    PetscCall(SFNodeArrayViewFromOptions(comm, "-petscsection_interp_candidate_remote_view", "Remote Candidates", NULL, candidatesRemoteSize, candidatesRemote));
  }
  /* Step 3: At the root, if at least two faces with a given cone are present, including a local face, mark the face as shared and choose the root face */
  {
    PetscHMapIJKLRemote faceTable;
    PetscInt            idx, idx2;

    PetscCall(PetscHMapIJKLRemoteCreate(&faceTable));
    /* There is a section point for every leaf attached to a given root point */
    for (r = 0, idx = 0, idx2 = 0; r < Nr; ++r) {
      PetscInt deg;

      for (deg = 0; deg < rootdegree[r]; ++deg, ++idx) {
        PetscInt offset, dof, d;

        PetscCall(PetscSectionGetDof(candidateRemoteSection, idx, &dof));
        PetscCall(PetscSectionGetOffset(candidateRemoteSection, idx, &offset));
        /* dof may include many faces from the remote process */
        for (d = 0; d < dof; ++d) {
          const PetscInt         hidx  = offset + d;
          const PetscInt         Np    = candidatesRemote[hidx].index + 1;
          const PetscSFNode      rface = candidatesRemote[hidx + 1];
          const PetscSFNode     *fcone = &candidatesRemote[hidx + 2];
          PetscSFNode            fcp0;
          const PetscSFNode      pmax = {-1, -1};
          const PetscInt        *join = NULL;
          PetscHMapIJKLRemoteKey key;
          PetscHashIter          iter;
          PetscBool              missing, mapToLocalPointFailed = PETSC_FALSE;
          PetscInt               points[1024], p, joinSize;

          if (debug)
            PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "[%d]  Checking face (%d, %" PetscInt_FMT ") at (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") with cone size %" PetscInt_FMT "\n", rank, (PetscMPIInt)rface.rank,
                                              rface.index, r, idx, d, Np));

          PetscCheck(Np <= 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle face (%d, %" PetscInt_FMT ") at (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") with %" PetscInt_FMT " cone points", (PetscMPIInt)rface.rank, rface.index, r, idx, d, Np);
          fcp0.rank  = rank;
          fcp0.index = r;
          d += Np;
          /* Put remote face in hash table */
          key.i = fcp0;
          key.j = fcone[0];
          key.k = Np > 2 ? fcone[1] : pmax;
          key.l = Np > 3 ? fcone[2] : pmax;
          PetscCall(PetscSortSFNode(Np, (PetscSFNode *)&key));
          PetscCall(PetscHMapIJKLRemotePut(faceTable, key, &iter, &missing));
          if (missing) {
            if (debug) PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "[%d]  Setting remote face (%" PetscInt_FMT ", %d)\n", rank, rface.index, (PetscMPIInt)rface.rank));
            PetscCall(PetscHMapIJKLRemoteIterSet(faceTable, iter, rface));
          } else {
            PetscSFNode oface;

            PetscCall(PetscHMapIJKLRemoteIterGet(faceTable, iter, &oface));
            if ((rface.rank < oface.rank) || (rface.rank == oface.rank && rface.index < oface.index)) {
              if (debug) PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "[%d]  Replacing with remote face (%" PetscInt_FMT ", %d)\n", rank, rface.index, (PetscMPIInt)rface.rank));
              PetscCall(PetscHMapIJKLRemoteIterSet(faceTable, iter, rface));
            }
          }
          /* Check for local face */
          points[0] = r;
          for (p = 1; p < Np; ++p) {
            PetscCall(DMPlexMapToLocalPoint(dm, remoteHash, fcone[p - 1], &points[p], &mapToLocalPointFailed));
            if (mapToLocalPointFailed) break; /* We got a point not in our overlap */
            if (debug) PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "[%d]  Checking local candidate %" PetscInt_FMT "\n", rank, points[p]));
          }
          if (mapToLocalPointFailed) continue;
          PetscCall(DMPlexGetJoin(dm, Np, points, &joinSize, &join));
          if (joinSize == 1) {
            PetscSFNode lface;
            PetscSFNode oface;

            /* Always replace with local face */
            lface.rank  = rank;
            lface.index = join[0];
            PetscCall(PetscHMapIJKLRemoteIterGet(faceTable, iter, &oface));
            if (debug)
              PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "[%d]  Replacing (%" PetscInt_FMT ", %d) with local face (%" PetscInt_FMT ", %d)\n", rank, oface.index, (PetscMPIInt)oface.rank, lface.index, (PetscMPIInt)lface.rank));
            PetscCall(PetscHMapIJKLRemoteIterSet(faceTable, iter, lface));
          }
          PetscCall(DMPlexRestoreJoin(dm, Np, points, &joinSize, &join));
        }
      }
      /* Put back faces for this root */
      for (deg = 0; deg < rootdegree[r]; ++deg, ++idx2) {
        PetscInt offset, dof, d;

        PetscCall(PetscSectionGetDof(candidateRemoteSection, idx2, &dof));
        PetscCall(PetscSectionGetOffset(candidateRemoteSection, idx2, &offset));
        /* dof may include many faces from the remote process */
        for (d = 0; d < dof; ++d) {
          const PetscInt         hidx  = offset + d;
          const PetscInt         Np    = candidatesRemote[hidx].index + 1;
          const PetscSFNode     *fcone = &candidatesRemote[hidx + 2];
          PetscSFNode            fcp0;
          const PetscSFNode      pmax = {-1, -1};
          PetscHMapIJKLRemoteKey key;
          PetscHashIter          iter;
          PetscBool              missing;

          if (debug) PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "[%d]  Entering face at (%" PetscInt_FMT ", %" PetscInt_FMT ")\n", rank, r, idx));
          PetscCheck(Np <= 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle faces with %" PetscInt_FMT " cone points", Np);
          fcp0.rank  = rank;
          fcp0.index = r;
          d += Np;
          /* Find remote face in hash table */
          key.i = fcp0;
          key.j = fcone[0];
          key.k = Np > 2 ? fcone[1] : pmax;
          key.l = Np > 3 ? fcone[2] : pmax;
          PetscCall(PetscSortSFNode(Np, (PetscSFNode *)&key));
          if (debug)
            PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)dm), "[%d]    key (%d, %" PetscInt_FMT ") (%d, %" PetscInt_FMT ") (%d, %" PetscInt_FMT ") (%d, %" PetscInt_FMT ")\n", rank, (PetscMPIInt)key.i.rank, key.i.index,
                                              (PetscMPIInt)key.j.rank, key.j.index, (PetscMPIInt)key.k.rank, key.k.index, (PetscMPIInt)key.l.rank, key.l.index));
          PetscCall(PetscHMapIJKLRemotePut(faceTable, key, &iter, &missing));
          PetscCheck(!missing, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Root %" PetscInt_FMT " Idx %" PetscInt_FMT " ought to have an associated face", r, idx2);
          PetscCall(PetscHMapIJKLRemoteIterGet(faceTable, iter, &candidatesRemote[hidx]));
        }
      }
    }
    if (debug) PetscCall(PetscSynchronizedFlush(PetscObjectComm((PetscObject)dm), NULL));
    PetscCall(PetscHMapIJKLRemoteDestroy(&faceTable));
  }
  /* Step 4: Push back owned faces */
  {
    PetscSF      sfMulti, sfClaims, sfPointNew;
    PetscSFNode *remotePointsNew;
    PetscInt    *remoteOffsets, *localPointsNew;
    PetscInt     pStart, pEnd, r, NlNew, p;

    /* 4) Push claims back to receiver via the MultiSF and derive new pointSF mapping on receiver */
    PetscCall(PetscSFGetMultiSF(pointSF, &sfMulti));
    PetscCall(PetscSectionCreate(comm, &claimSection));
    PetscCall(PetscSFDistributeSection(sfMulti, candidateRemoteSection, &remoteOffsets, claimSection));
    PetscCall(PetscSFCreateSectionSF(sfMulti, candidateRemoteSection, remoteOffsets, claimSection, &sfClaims));
    PetscCall(PetscSectionGetStorageSize(claimSection, &claimsSize));
    PetscCall(PetscMalloc1(claimsSize, &claims));
    for (p = 0; p < claimsSize; ++p) claims[p].rank = -1;
    PetscCall(PetscSFBcastBegin(sfClaims, MPIU_SF_NODE, candidatesRemote, claims, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sfClaims, MPIU_SF_NODE, candidatesRemote, claims, MPI_REPLACE));
    PetscCall(PetscSFDestroy(&sfClaims));
    PetscCall(PetscFree(remoteOffsets));
    PetscCall(PetscObjectSetName((PetscObject)claimSection, "Claim Section"));
    PetscCall(PetscObjectViewFromOptions((PetscObject)claimSection, NULL, "-petscsection_interp_claim_view"));
    PetscCall(SFNodeArrayViewFromOptions(comm, "-petscsection_interp_claim_view", "Claims", NULL, claimsSize, claims));
    /* Step 5) Walk the original section of local supports and add an SF entry for each updated item */
    /* TODO I should not have to do a join here since I already put the face and its cone in the candidate section */
    PetscCall(PetscHMapICreate(&claimshash));
    for (r = 0; r < Nr; ++r) {
      PetscInt dof, off, d;

      if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]  Checking root for claims %" PetscInt_FMT "\n", rank, r));
      PetscCall(PetscSectionGetDof(candidateSection, r, &dof));
      PetscCall(PetscSectionGetOffset(candidateSection, r, &off));
      for (d = 0; d < dof;) {
        if (claims[off + d].rank >= 0) {
          const PetscInt  faceInd = off + d;
          const PetscInt  Np      = candidates[off + d].index;
          const PetscInt *join    = NULL;
          PetscInt        joinSize, points[1024], c;

          if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    Found claim for remote point (%d, %" PetscInt_FMT ")\n", rank, (PetscMPIInt)claims[faceInd].rank, claims[faceInd].index));
          points[0] = r;
          if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]      point %" PetscInt_FMT "\n", rank, points[0]));
          for (c = 0, d += 2; c < Np; ++c, ++d) {
            PetscCall(DMPlexMapToLocalPoint(dm, remoteHash, candidates[off + d], &points[c + 1], NULL));
            if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]      point %" PetscInt_FMT "\n", rank, points[c + 1]));
          }
          PetscCall(DMPlexGetJoin(dm, Np + 1, points, &joinSize, &join));
          if (joinSize == 1) {
            if (claims[faceInd].rank == rank) {
              if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    Ignoring local face %" PetscInt_FMT " for non-remote partner\n", rank, join[0]));
            } else {
              if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    Found local face %" PetscInt_FMT "\n", rank, join[0]));
              PetscCall(PetscHMapISet(claimshash, join[0], faceInd));
            }
          } else {
            if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    Failed to find face\n", rank));
          }
          PetscCall(DMPlexRestoreJoin(dm, Np + 1, points, &joinSize, &join));
        } else {
          if (debug) PetscCall(PetscSynchronizedPrintf(comm, "[%d]    No claim for point %" PetscInt_FMT "\n", rank, r));
          d += claims[off + d].index + 1;
        }
      }
    }
    if (debug) PetscCall(PetscSynchronizedFlush(comm, NULL));
    /* Step 6) Create new pointSF from hashed claims */
    PetscCall(PetscHMapIGetSize(claimshash, &NlNew));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(PetscMalloc1(Nl + NlNew, &localPointsNew));
    PetscCall(PetscMalloc1(Nl + NlNew, &remotePointsNew));
    for (l = 0; l < Nl; ++l) {
      localPointsNew[l]        = localPoints[l];
      remotePointsNew[l].index = remotePoints[l].index;
      remotePointsNew[l].rank  = remotePoints[l].rank;
    }
    p = Nl;
    PetscCall(PetscHMapIGetKeys(claimshash, &p, localPointsNew));
    /* We sort new points, and assume they are numbered after all existing points */
    PetscCall(PetscSortInt(NlNew, PetscSafePointerPlusOffset(localPointsNew, Nl)));
    for (p = Nl; p < Nl + NlNew; ++p) {
      PetscInt off;
      PetscCall(PetscHMapIGet(claimshash, localPointsNew[p], &off));
      PetscCheck(claims[off].rank >= 0 && claims[off].index >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid claim for local point %" PetscInt_FMT ", (%d, %" PetscInt_FMT ")", localPointsNew[p], (PetscMPIInt)claims[off].rank, claims[off].index);
      remotePointsNew[p] = claims[off];
    }
    PetscCall(PetscSFCreate(comm, &sfPointNew));
    PetscCall(PetscSFSetGraph(sfPointNew, pEnd - pStart, Nl + NlNew, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER));
    PetscCall(PetscSFSetUp(sfPointNew));
    PetscCall(DMSetPointSF(dm, sfPointNew));
    PetscCall(PetscObjectViewFromOptions((PetscObject)sfPointNew, NULL, "-petscsf_interp_view"));
    if (PetscDefined(USE_DEBUG)) PetscCall(DMPlexCheckPointSF(dm, sfPointNew, PETSC_FALSE));
    PetscCall(PetscSFDestroy(&sfPointNew));
    PetscCall(PetscHMapIDestroy(&claimshash));
  }
  PetscCall(PetscHMapIJDestroy(&remoteHash));
  PetscCall(PetscSectionDestroy(&candidateSection));
  PetscCall(PetscSectionDestroy(&candidateRemoteSection));
  PetscCall(PetscSectionDestroy(&claimSection));
  PetscCall(PetscFree(candidates));
  PetscCall(PetscFree(candidatesRemote));
  PetscCall(PetscFree(claims));
  PetscCall(PetscLogEventEnd(DMPLEX_InterpolateSF, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexInterpolate - Take in a cell-vertex mesh and return one with all intermediate faces, edges, etc.

  Collective

  Input Parameter:
. dm - The `DMPLEX` object with only cells and vertices

  Output Parameter:
. dmInt - The complete `DMPLEX` object

  Level: intermediate

  Note:
  Labels and coordinates are copied.

  Developer Notes:
  It sets plex->interpolated = `DMPLEX_INTERPOLATED_FULL`.

.seealso: `DMPLEX`, `DMPlexUninterpolate()`, `DMPlexCreateFromCellListPetsc()`, `DMPlexCopyCoordinates()`
@*/
PetscErrorCode DMPlexInterpolate(DM dm, DM *dmInt)
{
  DMPlexInterpolatedFlag interpolated;
  DM                     idm, odm = dm;
  PetscSF                sfPoint;
  PetscInt               depth, dim, d;
  const char            *name;
  PetscBool              flg = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(dmInt, 2);
  PetscCall(PetscLogEventBegin(DMPLEX_Interpolate, dm, 0, 0, 0));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsInterpolated(dm, &interpolated));
  PetscCheck(interpolated != DMPLEX_INTERPOLATED_PARTIAL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated == DMPLEX_INTERPOLATED_FULL) {
    PetscCall(PetscObjectReference((PetscObject)dm));
    idm = dm;
  } else {
    PetscBool nonmanifold = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, dm->hdr.prefix, "-dm_plex_stratify_celltype", &nonmanifold, NULL));
    if (nonmanifold) {
      do {
        const char *prefix;
        PetscInt    pStart, pEnd, pdepth;
        PetscBool   done = PETSC_TRUE;

        // Find a point which is not correctly interpolated
        PetscCall(DMPlexGetChart(odm, &pStart, &pEnd));
        for (PetscInt p = pStart; p < pEnd; ++p) {
          DMPolytopeType  ct;
          const PetscInt *cone;
          PetscInt        coneSize, cdepth;

          PetscCall(DMPlexGetPointDepth(odm, p, &pdepth));
          PetscCall(DMPlexGetCellType(odm, p, &ct));
          // Check against celltype
          if (pdepth != DMPolytopeTypeGetDim(ct)) {
            done = PETSC_FALSE;
            break;
          }
          // Check against boundary
          PetscCall(DMPlexGetCone(odm, p, &cone));
          PetscCall(DMPlexGetConeSize(odm, p, &coneSize));
          for (PetscInt c = 0; c < coneSize; ++c) {
            PetscCall(DMPlexGetPointDepth(odm, cone[c], &cdepth));
            if (cdepth != pdepth - 1) {
              done = PETSC_FALSE;
              p    = pEnd;
              break;
            }
          }
        }
        if (done) break;
        /* Create interpolated mesh */
        PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), &idm));
        PetscCall(DMSetType(idm, DMPLEX));
        PetscCall(DMSetDimension(idm, dim));
        PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
        PetscCall(PetscObjectSetOptionsPrefix((PetscObject)idm, prefix));
        if (depth > 0) {
          PetscCall(DMPlexInterpolateFaces_Internal(odm, pdepth, idm));
          PetscCall(DMGetPointSF(odm, &sfPoint));
          if (PetscDefined(USE_DEBUG)) PetscCall(DMPlexCheckPointSF(odm, sfPoint, PETSC_FALSE));
          {
            /* TODO: We need to systematically fix cases of distributed Plexes with no graph set */
            PetscInt nroots;
            PetscCall(PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL));
            if (nroots >= 0) PetscCall(DMPlexInterpolatePointSF(idm, sfPoint));
          }
        }
        if (odm != dm) PetscCall(DMDestroy(&odm));
        odm = idm;
      } while (1);
    } else {
      for (d = 1; d < dim; ++d) {
        const char *prefix;

        /* Create interpolated mesh */
        PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), &idm));
        PetscCall(DMSetType(idm, DMPLEX));
        PetscCall(DMSetDimension(idm, dim));
        PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
        PetscCall(PetscObjectSetOptionsPrefix((PetscObject)idm, prefix));
        if (depth > 0) {
          PetscCall(DMPlexInterpolateFaces_Internal(odm, 1, idm));
          PetscCall(DMGetPointSF(odm, &sfPoint));
          if (PetscDefined(USE_DEBUG)) PetscCall(DMPlexCheckPointSF(odm, sfPoint, PETSC_FALSE));
          {
            /* TODO: We need to systematically fix cases of distributed Plexes with no graph set */
            PetscInt nroots;
            PetscCall(PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL));
            if (nroots >= 0) PetscCall(DMPlexInterpolatePointSF(idm, sfPoint));
          }
        }
        if (odm != dm) PetscCall(DMDestroy(&odm));
        odm = idm;
      }
    }
    PetscCall(PetscObjectGetName((PetscObject)dm, &name));
    PetscCall(PetscObjectSetName((PetscObject)idm, name));
    PetscCall(DMPlexCopyCoordinates(dm, idm));
    PetscCall(DMCopyLabels(dm, idm, PETSC_COPY_VALUES, PETSC_FALSE, DM_COPY_LABELS_FAIL));
    PetscCall(PetscOptionsGetBool(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dm_plex_interpolate_orient_interfaces", &flg, NULL));
    if (flg) PetscCall(DMPlexOrientInterface_Internal(idm));
  }
  /* This function makes the mesh fully interpolated on all ranks */
  {
    DM_Plex *plex      = (DM_Plex *)idm->data;
    plex->interpolated = plex->interpolatedCollective = DMPLEX_INTERPOLATED_FULL;
  }
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, PETSC_TRUE, idm));
  *dmInt = idm;
  PetscCall(PetscLogEventEnd(DMPLEX_Interpolate, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexCopyCoordinates - Copy coordinates from one mesh to another with the same vertices

  Collective

  Input Parameter:
. dmA - The `DMPLEX` object with initial coordinates

  Output Parameter:
. dmB - The `DMPLEX` object with copied coordinates

  Level: intermediate

  Notes:
  This is typically used when adding pieces other than vertices to a mesh

  This function does not copy localized coordinates.

.seealso: `DMPLEX`, `DMCopyLabels()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetCoordinateDM()`, `DMGetCoordinateSection()`
@*/
PetscErrorCode DMPlexCopyCoordinates(DM dmA, DM dmB)
{
  Vec          coordinatesA, coordinatesB;
  VecType      vtype;
  PetscSection coordSectionA, coordSectionB;
  PetscScalar *coordsA, *coordsB;
  PetscInt     spaceDim, Nf, vStartA, vStartB, vEndA, vEndB, coordSizeB, v, d;
  PetscInt     cStartA, cEndA, cStartB, cEndB, cS, cE, cdim;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  if (dmA == dmB) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetCoordinateDim(dmA, &cdim));
  PetscCall(DMSetCoordinateDim(dmB, cdim));
  PetscCall(DMPlexGetDepthStratum(dmA, 0, &vStartA, &vEndA));
  PetscCall(DMPlexGetDepthStratum(dmB, 0, &vStartB, &vEndB));
  PetscCheck((vEndA - vStartA) == (vEndB - vStartB), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of vertices in first DM %" PetscInt_FMT " != %" PetscInt_FMT " in the second DM", vEndA - vStartA, vEndB - vStartB);
  /* Copy over discretization if it exists */
  {
    DM                 cdmA, cdmB;
    PetscDS            dsA, dsB;
    PetscObject        objA, objB;
    PetscClassId       idA, idB;
    const PetscScalar *constants;
    PetscInt           cdim, Nc;

    PetscCall(DMGetCoordinateDM(dmA, &cdmA));
    PetscCall(DMGetCoordinateDM(dmB, &cdmB));
    PetscCall(DMGetField(cdmA, 0, NULL, &objA));
    PetscCall(DMGetField(cdmB, 0, NULL, &objB));
    PetscCall(PetscObjectGetClassId(objA, &idA));
    PetscCall(PetscObjectGetClassId(objB, &idB));
    if ((idA == PETSCFE_CLASSID) && (idA != idB)) {
      PetscCall(DMSetField(cdmB, 0, NULL, objA));
      PetscCall(DMCreateDS(cdmB));
      PetscCall(DMGetDS(cdmA, &dsA));
      PetscCall(DMGetDS(cdmB, &dsB));
      PetscCall(PetscDSGetCoordinateDimension(dsA, &cdim));
      PetscCall(PetscDSSetCoordinateDimension(dsB, cdim));
      PetscCall(PetscDSGetConstants(dsA, &Nc, &constants));
      PetscCall(PetscDSSetConstants(dsB, Nc, (PetscScalar *)constants));
    }
  }
  PetscCall(DMPlexGetHeightStratum(dmA, 0, &cStartA, &cEndA));
  PetscCall(DMPlexGetHeightStratum(dmB, 0, &cStartB, &cEndB));
  PetscCall(DMGetCoordinateSection(dmA, &coordSectionA));
  PetscCall(DMGetCoordinateSection(dmB, &coordSectionB));
  if (coordSectionA == coordSectionB) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSectionGetNumFields(coordSectionA, &Nf));
  if (!Nf) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(Nf <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of coordinate fields must be 1, not %" PetscInt_FMT, Nf);
  if (!coordSectionB) {
    PetscInt dim;

    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)coordSectionA), &coordSectionB));
    PetscCall(DMGetCoordinateDim(dmA, &dim));
    PetscCall(DMSetCoordinateSection(dmB, dim, coordSectionB));
    PetscCall(PetscObjectDereference((PetscObject)coordSectionB));
  }
  PetscCall(PetscSectionSetNumFields(coordSectionB, 1));
  PetscCall(PetscSectionGetFieldComponents(coordSectionA, 0, &spaceDim));
  PetscCall(PetscSectionSetFieldComponents(coordSectionB, 0, spaceDim));
  PetscCall(PetscSectionGetChart(coordSectionA, &cS, &cE));
  cS = vStartB;
  cE = vEndB;
  PetscCall(PetscSectionSetChart(coordSectionB, cS, cE));
  for (v = vStartB; v < vEndB; ++v) {
    PetscCall(PetscSectionSetDof(coordSectionB, v, spaceDim));
    PetscCall(PetscSectionSetFieldDof(coordSectionB, v, 0, spaceDim));
  }
  PetscCall(PetscSectionSetUp(coordSectionB));
  PetscCall(PetscSectionGetStorageSize(coordSectionB, &coordSizeB));
  PetscCall(DMGetCoordinatesLocal(dmA, &coordinatesA));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinatesB));
  PetscCall(PetscObjectSetName((PetscObject)coordinatesB, "coordinates"));
  PetscCall(VecSetSizes(coordinatesB, coordSizeB, PETSC_DETERMINE));
  PetscCall(VecGetBlockSize(coordinatesA, &d));
  PetscCall(VecSetBlockSize(coordinatesB, d));
  PetscCall(VecGetType(coordinatesA, &vtype));
  PetscCall(VecSetType(coordinatesB, vtype));
  PetscCall(VecGetArray(coordinatesA, &coordsA));
  PetscCall(VecGetArray(coordinatesB, &coordsB));
  for (v = 0; v < vEndB - vStartB; ++v) {
    PetscInt offA, offB;

    PetscCall(PetscSectionGetOffset(coordSectionA, v + vStartA, &offA));
    PetscCall(PetscSectionGetOffset(coordSectionB, v + vStartB, &offB));
    for (d = 0; d < spaceDim; ++d) coordsB[offB + d] = coordsA[offA + d];
  }
  PetscCall(VecRestoreArray(coordinatesA, &coordsA));
  PetscCall(VecRestoreArray(coordinatesB, &coordsB));
  PetscCall(DMSetCoordinatesLocal(dmB, coordinatesB));
  PetscCall(VecDestroy(&coordinatesB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexUninterpolate - Take in a mesh with all intermediate faces, edges, etc. and return a cell-vertex mesh

  Collective

  Input Parameter:
. dm - The complete `DMPLEX` object

  Output Parameter:
. dmUnint - The `DMPLEX` object with only cells and vertices

  Level: intermediate

  Note:
  It does not copy over the coordinates.

  Developer Notes:
  Sets plex->interpolated = `DMPLEX_INTERPOLATED_NONE`.

.seealso: `DMPLEX`, `DMPlexInterpolate()`, `DMPlexCreateFromCellListPetsc()`, `DMPlexCopyCoordinates()`
@*/
PetscErrorCode DMPlexUninterpolate(DM dm, DM *dmUnint)
{
  DMPlexInterpolatedFlag interpolated;
  DM                     udm;
  PetscInt               dim, vStart, vEnd, cStart, cEnd, c, maxConeSize = 0, *cone;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(dmUnint, 2);
  PetscCall(PetscLogEventBegin(DMPLEX_Uninterpolate, dm, 0, 0, 0));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsInterpolated(dm, &interpolated));
  PetscCheck(interpolated != DMPLEX_INTERPOLATED_PARTIAL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not for partially interpolated meshes");
  if (interpolated == DMPLEX_INTERPOLATED_NONE || dim <= 1) {
    /* in case dim <= 1 just keep the DMPLEX_INTERPOLATED_FULL flag */
    PetscCall(PetscObjectReference((PetscObject)dm));
    *dmUnint = dm;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), &udm));
  PetscCall(DMSetType(udm, DMPLEX));
  PetscCall(DMSetDimension(udm, dim));
  PetscCall(DMPlexSetChart(udm, cStart, vEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize * 2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) ++coneSize;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    PetscCall(DMPlexSetConeSize(udm, c, coneSize));
    maxConeSize = PetscMax(maxConeSize, coneSize);
  }
  PetscCall(DMSetUp(udm));
  PetscCall(PetscMalloc1(maxConeSize, &cone));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize * 2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) cone[coneSize++] = p;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    PetscCall(DMPlexSetCone(udm, c, cone));
  }
  PetscCall(PetscFree(cone));
  PetscCall(DMPlexSymmetrize(udm));
  PetscCall(DMPlexStratify(udm));
  /* Reduce SF */
  {
    PetscSF            sfPoint, sfPointUn;
    const PetscSFNode *remotePoints;
    const PetscInt    *localPoints;
    PetscSFNode       *remotePointsUn;
    PetscInt          *localPointsUn;
    PetscInt           numRoots, numLeaves, l;
    PetscInt           numLeavesUn = 0, n = 0;

    /* Get original SF information */
    PetscCall(DMGetPointSF(dm, &sfPoint));
    if (PetscDefined(USE_DEBUG)) PetscCall(DMPlexCheckPointSF(dm, sfPoint, PETSC_FALSE));
    PetscCall(DMGetPointSF(udm, &sfPointUn));
    PetscCall(PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints));
    if (numRoots >= 0) {
      /* Allocate space for cells and vertices */
      for (l = 0; l < numLeaves; ++l) {
        const PetscInt p = localPoints[l];

        if ((vStart <= p && p < vEnd) || (cStart <= p && p < cEnd)) numLeavesUn++;
      }
      /* Fill in leaves */
      PetscCall(PetscMalloc1(numLeavesUn, &remotePointsUn));
      PetscCall(PetscMalloc1(numLeavesUn, &localPointsUn));
      for (l = 0; l < numLeaves; l++) {
        const PetscInt p = localPoints[l];

        if ((vStart <= p && p < vEnd) || (cStart <= p && p < cEnd)) {
          localPointsUn[n]        = p;
          remotePointsUn[n].rank  = remotePoints[l].rank;
          remotePointsUn[n].index = remotePoints[l].index;
          ++n;
        }
      }
      PetscCheck(n == numLeavesUn, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent number of leaves %" PetscInt_FMT " != %" PetscInt_FMT, n, numLeavesUn);
      PetscCall(PetscSFSetGraph(sfPointUn, cEnd - cStart + vEnd - vStart, numLeavesUn, localPointsUn, PETSC_OWN_POINTER, remotePointsUn, PETSC_OWN_POINTER));
    }
  }
  /* This function makes the mesh fully uninterpolated on all ranks */
  {
    DM_Plex *plex      = (DM_Plex *)udm->data;
    plex->interpolated = plex->interpolatedCollective = DMPLEX_INTERPOLATED_NONE;
  }
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, PETSC_TRUE, udm));
  if (PetscDefined(USE_DEBUG)) PetscCall(DMPlexCheckPointSF(udm, NULL, PETSC_FALSE));
  *dmUnint = udm;
  PetscCall(PetscLogEventEnd(DMPLEX_Uninterpolate, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexIsInterpolated_Internal(DM dm, DMPlexInterpolatedFlag *interpolated)
{
  PetscInt coneSize, depth, dim, h, p, pStart, pEnd;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetDimension(dm, &dim));

  if (depth == dim) {
    *interpolated = DMPLEX_INTERPOLATED_FULL;
    if (!dim) goto finish;

    /* Check points at height = dim are vertices (have no cones) */
    PetscCall(DMPlexGetHeightStratum(dm, dim, &pStart, &pEnd));
    for (p = pStart; p < pEnd; p++) {
      PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
      if (coneSize) {
        *interpolated = DMPLEX_INTERPOLATED_PARTIAL;
        goto finish;
      }
    }

    /* Check points at height < dim have cones */
    for (h = 0; h < dim; h++) {
      PetscCall(DMPlexGetHeightStratum(dm, h, &pStart, &pEnd));
      for (p = pStart; p < pEnd; p++) {
        PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexIsInterpolated - Find out to what extent the `DMPLEX` is topologically interpolated.

  Not Collective

  Input Parameter:
. dm - The `DMPLEX` object

  Output Parameter:
. interpolated - Flag whether the `DM` is interpolated

  Level: intermediate

  Notes:
  Unlike `DMPlexIsInterpolatedCollective()`, this is NOT collective
  so the results can be different on different ranks in special cases.
  However, `DMPlexInterpolate()` guarantees the result is the same on all.

  Unlike `DMPlexIsInterpolatedCollective()`, this cannot return `DMPLEX_INTERPOLATED_MIXED`.

  Developer Notes:
  Initially, plex->interpolated = `DMPLEX_INTERPOLATED_INVALID`.

  If plex->interpolated == `DMPLEX_INTERPOLATED_INVALID`, `DMPlexIsInterpolated_Internal()` is called.
  It checks the actual topology and sets plex->interpolated on each rank separately to one of
  `DMPLEX_INTERPOLATED_NONE`, `DMPLEX_INTERPOLATED_PARTIAL` or `DMPLEX_INTERPOLATED_FULL`.

  If plex->interpolated != `DMPLEX_INTERPOLATED_INVALID`, this function just returns plex->interpolated.

  `DMPlexInterpolate()` sets plex->interpolated = `DMPLEX_INTERPOLATED_FULL`,
  and DMPlexUninterpolate() sets plex->interpolated = `DMPLEX_INTERPOLATED_NONE`.

.seealso: `DMPLEX`, `DMPlexInterpolate()`, `DMPlexIsInterpolatedCollective()`
@*/
PetscErrorCode DMPlexIsInterpolated(DM dm, DMPlexInterpolatedFlag *interpolated)
{
  DM_Plex *plex = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(interpolated, 2);
  if (plex->interpolated < 0) {
    PetscCall(DMPlexIsInterpolated_Internal(dm, &plex->interpolated));
  } else if (PetscDefined(USE_DEBUG)) {
    DMPlexInterpolatedFlag flg;

    PetscCall(DMPlexIsInterpolated_Internal(dm, &flg));
    PetscCheck(plex->tr || flg == plex->interpolated, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stashed DMPlexInterpolatedFlag %s is inconsistent with current %s", DMPlexInterpolatedFlags[plex->interpolated], DMPlexInterpolatedFlags[flg]);
  }
  *interpolated = plex->interpolated;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexIsInterpolatedCollective - Find out to what extent the `DMPLEX` is topologically interpolated (in collective manner).

  Collective

  Input Parameter:
. dm - The `DMPLEX` object

  Output Parameter:
. interpolated - Flag whether the `DM` is interpolated

  Level: intermediate

  Notes:
  Unlike `DMPlexIsInterpolated()`, this is collective so the results are guaranteed to be the same on all ranks.

  This function will return `DMPLEX_INTERPOLATED_MIXED` if the results of `DMPlexIsInterpolated()` are different on different ranks.

  Developer Notes:
  Initially, plex->interpolatedCollective = `DMPLEX_INTERPOLATED_INVALID`.

  If plex->interpolatedCollective == `DMPLEX_INTERPOLATED_INVALID`, this function calls `DMPlexIsInterpolated()` which sets plex->interpolated.
  `MPI_Allreduce()` is then called and collectively consistent flag plex->interpolatedCollective is set and returned;
  if plex->interpolated varies on different ranks, plex->interpolatedCollective = `DMPLEX_INTERPOLATED_MIXED`,
  otherwise sets plex->interpolatedCollective = plex->interpolated.

  If plex->interpolatedCollective != `DMPLEX_INTERPOLATED_INVALID`, this function just returns plex->interpolatedCollective.

.seealso: `DMPLEX`, `DMPlexInterpolate()`, `DMPlexIsInterpolated()`
@*/
PetscErrorCode DMPlexIsInterpolatedCollective(DM dm, DMPlexInterpolatedFlag *interpolated)
{
  DM_Plex  *plex  = (DM_Plex *)dm->data;
  PetscBool debug = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(interpolated, 2);
  PetscCall(PetscOptionsGetBool(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dm_plex_is_interpolated_collective_debug", &debug, NULL));
  if (plex->interpolatedCollective < 0) {
    DMPlexInterpolatedFlag min, max;
    MPI_Comm               comm;

    PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
    PetscCall(DMPlexIsInterpolated(dm, &plex->interpolatedCollective));
    PetscCallMPI(MPIU_Allreduce(&plex->interpolatedCollective, &min, 1, MPIU_ENUM, MPI_MIN, comm));
    PetscCallMPI(MPIU_Allreduce(&plex->interpolatedCollective, &max, 1, MPIU_ENUM, MPI_MAX, comm));
    if (min != max) plex->interpolatedCollective = DMPLEX_INTERPOLATED_MIXED;
    if (debug) {
      PetscMPIInt rank;

      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscCall(PetscSynchronizedPrintf(comm, "[%d] interpolated=%s interpolatedCollective=%s\n", rank, DMPlexInterpolatedFlags[plex->interpolated], DMPlexInterpolatedFlags[plex->interpolatedCollective]));
      PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
    }
  }
  *interpolated = plex->interpolatedCollective;
  PetscFunctionReturn(PETSC_SUCCESS);
}
