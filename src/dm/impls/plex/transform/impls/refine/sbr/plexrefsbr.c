#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/
#include <petscsf.h>

PetscBool SBRcite = PETSC_FALSE;
const char SBRCitation[] = "@article{PlazaCarey2000,\n"
                          "  title   = {Local refinement of simplicial grids based on the skeleton},\n"
                          "  journal = {Applied Numerical Mathematics},\n"
                          "  author  = {A. Plaza and Graham F. Carey},\n"
                          "  volume  = {32},\n"
                          "  number  = {3},\n"
                          "  pages   = {195--218},\n"
                          "  doi     = {10.1016/S0168-9274(99)00022-7},\n"
                          "  year    = {2000}\n}\n";

typedef struct _p_PointQueue *PointQueue;
struct _p_PointQueue {
  PetscInt  size;   /* Size of the storage array */
  PetscInt *points; /* Array of mesh points */
  PetscInt  front;  /* Index of the front of the queue */
  PetscInt  back;   /* Index of the back of the queue */
  PetscInt  num;    /* Number of enqueued points */
};

static PetscErrorCode PointQueueCreate(PetscInt size, PointQueue *queue)
{
  PointQueue     q;

  PetscFunctionBegin;
  PetscCheck(size >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Queue size %D must be non-negative", size);
  PetscCall(PetscCalloc1(1, &q));
  q->size = size;
  PetscCall(PetscMalloc1(q->size, &q->points));
  q->num   = 0;
  q->front = 0;
  q->back  = q->size-1;
  *queue = q;
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueDestroy(PointQueue *queue)
{
  PointQueue     q = *queue;

  PetscFunctionBegin;
  PetscCall(PetscFree(q->points));
  PetscCall(PetscFree(q));
  *queue = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueEnsureSize(PointQueue queue)
{
  PetscFunctionBegin;
  if (queue->num < queue->size) PetscFunctionReturn(0);
  queue->size *= 2;
  PetscCall(PetscRealloc(queue->size * sizeof(PetscInt), &queue->points));
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueEnqueue(PointQueue queue, PetscInt p)
{
  PetscFunctionBegin;
  PetscCall(PointQueueEnsureSize(queue));
  queue->back = (queue->back + 1) % queue->size;
  queue->points[queue->back] = p;
  ++queue->num;
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueDequeue(PointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheck(queue->num,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot dequeue from an empty queue");
  *p = queue->points[queue->front];
  queue->front = (queue->front + 1) % queue->size;
  --queue->num;
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode PointQueueFront(PointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheck(queue->num,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot get the front of an empty queue");
  *p = queue->points[queue->front];
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueBack(PointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheck(queue->num,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot get the back of an empty queue");
  *p = queue->points[queue->back];
  PetscFunctionReturn(0);
}
#endif

static inline PetscBool PointQueueEmpty(PointQueue queue)
{
  if (!queue->num) return PETSC_TRUE;
  return PETSC_FALSE;
}

static PetscErrorCode SBRGetEdgeLen_Private(DMPlexTransform tr, PetscInt edge, PetscReal *len)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *) tr->data;
  DM                dm;
  PetscInt          off;

  PetscFunctionBeginHot;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(PetscSectionGetOffset(sbr->secEdgeLen, edge, &off));
  if (sbr->edgeLen[off] <= 0.0) {
    DM                 cdm;
    Vec                coordsLocal;
    const PetscScalar *coords;
    const PetscInt    *cone;
    PetscScalar       *cA, *cB;
    PetscInt           coneSize, cdim;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMPlexGetCone(dm, edge, &cone));
    PetscCall(DMPlexGetConeSize(dm, edge, &coneSize));
    PetscCheck(coneSize == 2,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Edge %D cone size must be 2, not %D", edge, coneSize);
    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
    PetscCall(VecGetArrayRead(coordsLocal, &coords));
    PetscCall(DMPlexPointLocalRead(cdm, cone[0], coords, &cA));
    PetscCall(DMPlexPointLocalRead(cdm, cone[1], coords, &cB));
    sbr->edgeLen[off] = DMPlex_DistD_Internal(cdim, cA, cB);
    PetscCall(VecRestoreArrayRead(coordsLocal, &coords));
  }
  *len = sbr->edgeLen[off];
  PetscFunctionReturn(0);
}

/* Mark local edges that should be split */
/* TODO This will not work in 3D */
static PetscErrorCode SBRSplitLocalEdges_Private(DMPlexTransform tr, PointQueue queue)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *) tr->data;
  DM                dm;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  while (!PointQueueEmpty(queue)) {
    PetscInt        p = -1;
    const PetscInt *support;
    PetscInt        supportSize, s;

    PetscCall(PointQueueDequeue(queue, &p));
    PetscCall(DMPlexGetSupport(dm, p, &support));
    PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
    for (s = 0; s < supportSize; ++s) {
      const PetscInt  cell = support[s];
      const PetscInt *cone;
      PetscInt        coneSize, c;
      PetscInt        cval, eval, maxedge;
      PetscReal       len, maxlen;

      PetscCall(DMLabelGetValue(sbr->splitPoints, cell, &cval));
      if (cval == 2) continue;
      PetscCall(DMPlexGetCone(dm, cell, &cone));
      PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
      PetscCall(SBRGetEdgeLen_Private(tr, cone[0], &maxlen));
      maxedge = cone[0];
      for (c = 1; c < coneSize; ++c) {
        PetscCall(SBRGetEdgeLen_Private(tr, cone[c], &len));
        if (len > maxlen) {maxlen = len; maxedge = cone[c];}
      }
      PetscCall(DMLabelGetValue(sbr->splitPoints, maxedge, &eval));
      if (eval != 1) {
        PetscCall(DMLabelSetValue(sbr->splitPoints, maxedge, 1));
        PetscCall(PointQueueEnqueue(queue, maxedge));
      }
      PetscCall(DMLabelSetValue(sbr->splitPoints, cell, 2));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SBRInitializeComm(DMPlexTransform tr, PetscSF pointSF)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *) tr->data;
  DM                dm;
  DMLabel           splitPoints = sbr->splitPoints;
  PetscInt         *splitArray  = sbr->splitArray;
  const PetscInt   *degree;
  const PetscInt   *points;
  PetscInt          Nl, l, pStart, pEnd, p, val;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  /* Add in leaves */
  PetscCall(PetscSFGetGraph(pointSF, NULL, &Nl, &points, NULL));
  for (l = 0; l < Nl; ++l) {
    PetscCall(DMLabelGetValue(splitPoints, points[l], &val));
    if (val > 0) splitArray[points[l]] = val;
  }
  /* Add in shared roots */
  PetscCall(PetscSFComputeDegreeBegin(pointSF, &degree));
  PetscCall(PetscSFComputeDegreeEnd(pointSF, &degree));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    if (degree[p]) {
      PetscCall(DMLabelGetValue(splitPoints, p, &val));
      if (val > 0) splitArray[p] = val;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SBRFinalizeComm(DMPlexTransform tr, PetscSF pointSF, PointQueue queue)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *) tr->data;
  DM                dm;
  DMLabel           splitPoints = sbr->splitPoints;
  PetscInt         *splitArray  = sbr->splitArray;
  const PetscInt   *degree;
  const PetscInt   *points;
  PetscInt          Nl, l, pStart, pEnd, p, val;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  /* Read out leaves */
  PetscCall(PetscSFGetGraph(pointSF, NULL, &Nl, &points, NULL));
  for (l = 0; l < Nl; ++l) {
    const PetscInt p    = points[l];
    const PetscInt cval = splitArray[p];

    if (cval) {
      PetscCall(DMLabelGetValue(splitPoints, p, &val));
      if (val <= 0) {
        PetscCall(DMLabelSetValue(splitPoints, p, cval));
        PetscCall(PointQueueEnqueue(queue, p));
      }
    }
  }
  /* Read out shared roots */
  PetscCall(PetscSFComputeDegreeBegin(pointSF, &degree));
  PetscCall(PetscSFComputeDegreeEnd(pointSF, &degree));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    if (degree[p]) {
      const PetscInt cval = splitArray[p];

      if (cval) {
        PetscCall(DMLabelGetValue(splitPoints, p, &val));
        if (val <= 0) {
          PetscCall(DMLabelSetValue(splitPoints, p, cval));
          PetscCall(PointQueueEnqueue(queue, p));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  The 'splitPoints' label marks mesh points to be divided. It marks edges with 1, triangles with 2, and tetrahedra with 3.
  Then the refinement type is calculated as follows:

    vertex:                   0
    edge unsplit:             1
    edge split:               2
    triangle unsplit:         3
    triangle split all edges: 4
    triangle split edges 0 1: 5
    triangle split edges 1 0: 6
    triangle split edges 1 2: 7
    triangle split edges 2 1: 8
    triangle split edges 2 0: 9
    triangle split edges 0 2: 10
    triangle split edge 0:    11
    triangle split edge 1:    12
    triangle split edge 2:    13
*/
typedef enum {RT_VERTEX, RT_EDGE, RT_EDGE_SPLIT, RT_TRIANGLE, RT_TRIANGLE_SPLIT, RT_TRIANGLE_SPLIT_01, RT_TRIANGLE_SPLIT_10, RT_TRIANGLE_SPLIT_12, RT_TRIANGLE_SPLIT_21, RT_TRIANGLE_SPLIT_20, RT_TRIANGLE_SPLIT_02, RT_TRIANGLE_SPLIT_0, RT_TRIANGLE_SPLIT_1, RT_TRIANGLE_SPLIT_2} RefinementType;

static PetscErrorCode DMPlexTransformSetUp_SBR(DMPlexTransform tr)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *) tr->data;
  DM                dm;
  DMLabel           active;
  PetscSF           pointSF;
  PointQueue        queue = NULL;
  IS                refineIS;
  const PetscInt   *refineCells;
  PetscMPIInt       size;
  PetscInt          pStart, pEnd, p, eStart, eEnd, e, edgeLenSize, Nc, c;
  PetscBool         empty;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Split Points", &sbr->splitPoints));
  /* Create edge lengths */
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &sbr->secEdgeLen));
  PetscCall(PetscSectionSetChart(sbr->secEdgeLen, eStart, eEnd));
  for (e = eStart; e < eEnd; ++e) {
    PetscCall(PetscSectionSetDof(sbr->secEdgeLen, e, 1));
  }
  PetscCall(PetscSectionSetUp(sbr->secEdgeLen));
  PetscCall(PetscSectionGetStorageSize(sbr->secEdgeLen, &edgeLenSize));
  PetscCall(PetscCalloc1(edgeLenSize, &sbr->edgeLen));
  /* Add edges of cells that are marked for refinement to edge queue */
  PetscCall(DMPlexTransformGetActive(tr, &active));
  PetscCheck(active,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_WRONGSTATE, "DMPlexTransform must have an adaptation label in order to use SBR algorithm");
  PetscCall(PointQueueCreate(1024, &queue));
  PetscCall(DMLabelGetStratumIS(active, DM_ADAPT_REFINE, &refineIS));
  PetscCall(DMLabelGetStratumSize(active, DM_ADAPT_REFINE, &Nc));
  if (refineIS) PetscCall(ISGetIndices(refineIS, &refineCells));
  for (c = 0; c < Nc; ++c) {
    const PetscInt cell = refineCells[c];
    PetscInt       depth;

    PetscCall(DMPlexGetPointDepth(dm, cell, &depth));
    if (depth == 1) {
      PetscCall(DMLabelSetValue(sbr->splitPoints, cell, 1));
      PetscCall(PointQueueEnqueue(queue, cell));
    } else {
      PetscInt *closure = NULL;
      PetscInt  Ncl, cl;

      PetscCall(DMLabelSetValue(sbr->splitPoints, cell, depth));
      PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
      for (cl = 0; cl < Ncl; cl += 2) {
        const PetscInt edge = closure[cl];

        if (edge >= eStart && edge < eEnd) {
          PetscCall(DMLabelSetValue(sbr->splitPoints, edge, 1));
          PetscCall(PointQueueEnqueue(queue, edge));
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    }
  }
  if (refineIS) PetscCall(ISRestoreIndices(refineIS, &refineCells));
  PetscCall(ISDestroy(&refineIS));
  /* Setup communication */
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  PetscCall(DMGetPointSF(dm, &pointSF));
  if (size > 1) {
    PetscInt pStart, pEnd;

    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(PetscCalloc1(pEnd-pStart, &sbr->splitArray));
  }
  /* While edge queue is not empty: */
  empty = PointQueueEmpty(queue);
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &empty, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject) dm)));
  while (!empty) {
    PetscCall(SBRSplitLocalEdges_Private(tr, queue));
    /* Communicate marked edges
         An easy implementation is to allocate an array the size of the number of points. We put the splitPoints marks into the
         array, and then call PetscSFReduce()+PetscSFBcast() to make the marks consistent.

         TODO: We could use in-place communication with a different SF
           We use MPI_SUM for the Reduce, and check the result against the rootdegree. If sum >= rootdegree+1, then the edge has
           already been marked. If not, it might have been handled on the process in this round, but we add it anyway.

           In order to update the queue with the new edges from the label communication, we use BcastAnOp(MPI_SUM), so that new
           values will have 1+0=1 and old values will have 1+1=2. Loop over these, resetting the values to 1, and adding any new
           edge to the queue.
    */
    if (size > 1) {
      PetscCall(SBRInitializeComm(tr, pointSF));
      PetscCall(PetscSFReduceBegin(pointSF, MPIU_INT, sbr->splitArray, sbr->splitArray, MPI_MAX));
      PetscCall(PetscSFReduceEnd(pointSF, MPIU_INT, sbr->splitArray, sbr->splitArray, MPI_MAX));
      PetscCall(PetscSFBcastBegin(pointSF, MPIU_INT, sbr->splitArray, sbr->splitArray,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(pointSF, MPIU_INT, sbr->splitArray, sbr->splitArray,MPI_REPLACE));
      PetscCall(SBRFinalizeComm(tr, pointSF, queue));
    }
    empty = PointQueueEmpty(queue);
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &empty, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject) dm)));
  }
  PetscCall(PetscFree(sbr->splitArray));
  /* Calculate refineType for each cell */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    DMLabel        trType = tr->trType;
    DMPolytopeType ct;
    PetscInt       val;

    PetscCall(DMPlexGetCellType(dm, p, &ct));
    switch (ct) {
      case DM_POLYTOPE_POINT:
        PetscCall(DMLabelSetValue(trType, p, RT_VERTEX));break;
      case DM_POLYTOPE_SEGMENT:
        PetscCall(DMLabelGetValue(sbr->splitPoints, p, &val));
        if (val == 1) PetscCall(DMLabelSetValue(trType, p, RT_EDGE_SPLIT));
        else          PetscCall(DMLabelSetValue(trType, p, RT_EDGE));
        break;
      case DM_POLYTOPE_TRIANGLE:
        PetscCall(DMLabelGetValue(sbr->splitPoints, p, &val));
        if (val == 2) {
          const PetscInt *cone;
          PetscReal       lens[3];
          PetscInt        vals[3], i;

          PetscCall(DMPlexGetCone(dm, p, &cone));
          for (i = 0; i < 3; ++i) {
            PetscCall(DMLabelGetValue(sbr->splitPoints, cone[i], &vals[i]));
            vals[i] = vals[i] < 0 ? 0 : vals[i];
            PetscCall(SBRGetEdgeLen_Private(tr, cone[i], &lens[i]));
          }
          if (vals[0] && vals[1] && vals[2]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT));
          else if (vals[0] && vals[1])       PetscCall(DMLabelSetValue(trType, p, lens[0] > lens[1] ? RT_TRIANGLE_SPLIT_01 : RT_TRIANGLE_SPLIT_10));
          else if (vals[1] && vals[2])       PetscCall(DMLabelSetValue(trType, p, lens[1] > lens[2] ? RT_TRIANGLE_SPLIT_12 : RT_TRIANGLE_SPLIT_21));
          else if (vals[2] && vals[0])       PetscCall(DMLabelSetValue(trType, p, lens[2] > lens[0] ? RT_TRIANGLE_SPLIT_20 : RT_TRIANGLE_SPLIT_02));
          else if (vals[0])                  PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_0));
          else if (vals[1])                  PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_1));
          else if (vals[2])                  PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_2));
          else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %D does not fit any refinement type (%D, %D, %D)", p, vals[0], vals[1], vals[2]);
        } else PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE));
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle points of type %s", DMPolytopeTypes[ct]);
    }
    PetscCall(DMLabelGetValue(sbr->splitPoints, p, &val));
  }
  /* Cleanup */
  PetscCall(PointQueueDestroy(&queue));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_SBR(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscInt         rt;

  PetscFunctionBeginHot;
  PetscCall(DMLabelGetValue(tr->trType, sp, &rt));
  *rnew = r;
  *onew = o;
  switch (rt) {
    case RT_TRIANGLE_SPLIT_01:
    case RT_TRIANGLE_SPLIT_10:
    case RT_TRIANGLE_SPLIT_12:
    case RT_TRIANGLE_SPLIT_21:
    case RT_TRIANGLE_SPLIT_20:
    case RT_TRIANGLE_SPLIT_02:
      switch (tct) {
        case DM_POLYTOPE_SEGMENT:  break;
        case DM_POLYTOPE_TRIANGLE: break;
        default: break;
      }
      break;
    case RT_TRIANGLE_SPLIT_0:
    case RT_TRIANGLE_SPLIT_1:
    case RT_TRIANGLE_SPLIT_2:
      switch (tct) {
        case DM_POLYTOPE_SEGMENT:
          break;
        case DM_POLYTOPE_TRIANGLE:
          *onew = so < 0 ? -(o+1)  : o;
          *rnew = so < 0 ? (r+1)%2 : r;
          break;
        default: break;
      }
      break;
    case RT_EDGE_SPLIT:
    case RT_TRIANGLE_SPLIT:
      PetscCall(DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew));
      break;
    default: PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
  }
  PetscFunctionReturn(0);
}

/* Add 1 edge inside this triangle, making 2 new triangles.
 2
 |\
 | \
 |  \
 |   \
 |    1
 |     \
 |  B   \
 2       1
 |      / \
 | ____/   0
 |/    A    \
 0-----0-----1
*/
static PetscErrorCode SBRGetTriangleSplitSingle(PetscInt o, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  const PetscInt       *arr     = DMPolytopeTypeGetArrangment(DM_POLYTOPE_TRIANGLE, o);
  static DMPolytopeType triT1[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS1[] = {1, 2};
  static PetscInt       triC1[] = {DM_POLYTOPE_POINT, 2, 0, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0,    0,
                                   DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0,    0};
  static PetscInt       triO1[] = {0, 0,
                                   0,  0, -1,
                                   0,  0,  0};

  PetscFunctionBeginHot;
  /* To get the other divisions, we reorient the triangle */
  triC1[2]  = arr[0*2];
  triC1[7]  = arr[1*2];
  triC1[11] = arr[0*2];
  triC1[15] = arr[1*2];
  triC1[22] = arr[1*2];
  triC1[26] = arr[2*2];
  *Nt = 2; *target = triT1; *size = triS1; *cone = triC1; *ornt = triO1;
  PetscFunctionReturn(0);
}

/* Add 2 edges inside this triangle, making 3 new triangles.
 RT_TRIANGLE_SPLIT_12
 2
 |\
 | \
 |  \
 0   \
 |    1
 |     \
 |  B   \
 2-------1
 |   C  / \
 1 ____/   0
 |/    A    \
 0-----0-----1
 RT_TRIANGLE_SPLIT_10
 2
 |\
 | \
 |  \
 0   \
 |    1
 |     \
 |  A   \
 2       1
 |      /|\
 1 ____/ / 0
 |/ C   / B \
 0-----0-----1
 RT_TRIANGLE_SPLIT_20
 2
 |\
 | \
 |  \
 0   \
 |    \
 |     \
 |      \
 2   A   1
 |\       \
 1 ---\    \
 |B \_C----\\
 0-----0-----1
 RT_TRIANGLE_SPLIT_21
 2
 |\
 | \
 |  \
 0   \
 |    \
 |  B  \
 |      \
 2-------1
 |\     C \
 1 ---\    \
 |  A  ----\\
 0-----0-----1
 RT_TRIANGLE_SPLIT_01
 2
 |\
 |\\
 || \
 | \ \
 |  | \
 |  |  \
 |  |   \
 2   \ C 1
 |  A | / \
 |    | |B \
 |     \/   \
 0-----0-----1
 RT_TRIANGLE_SPLIT_02
 2
 |\
 |\\
 || \
 | \ \
 |  | \
 |  |  \
 |  |   \
 2 C \   1
 |\   |   \
 | \__|  A \
 | B  \\    \
 0-----0-----1
*/
static PetscErrorCode SBRGetTriangleSplitDouble(PetscInt o, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  PetscInt              e0, e1;
  const PetscInt       *arr     = DMPolytopeTypeGetArrangment(DM_POLYTOPE_TRIANGLE, o);
  static DMPolytopeType triT2[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS2[] = {2, 3};
  static PetscInt       triC2[] = {DM_POLYTOPE_POINT, 2, 0, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                   DM_POLYTOPE_POINT, 1, 1,    0, DM_POLYTOPE_POINT, 1, 2, 0,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0,    0,
                                   DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0,    1,
                                   DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    1};
  static PetscInt       triO2[] = {0, 0,
                                   0, 0,
                                   0,  0, -1,
                                   0,  0, -1,
                                   0,  0,  0};

  PetscFunctionBeginHot;
  /* To get the other divisions, we reorient the triangle */
  triC2[2]  = arr[0*2]; triC2[3] = arr[0*2+1] ? 1 : 0;
  triC2[7]  = arr[1*2];
  triC2[11] = arr[1*2];
  triC2[15] = arr[2*2];
  /* Swap the first two edges if the triangle is reversed */
  e0 = o < 0 ? 23: 19;
  e1 = o < 0 ? 19: 23;
  triC2[e0] = arr[0*2]; triC2[e0+1] = 0;
  triC2[e1] = arr[1*2]; triC2[e1+1] = o < 0 ? 1 : 0;
  triO2[6]  = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, -1, arr[2*2+1]);
  /* Swap the first two edges if the triangle is reversed */
  e0 = o < 0 ? 34: 30;
  e1 = o < 0 ? 30: 34;
  triC2[e0] = arr[1*2]; triC2[e0+1] = o < 0 ? 0 : 1;
  triC2[e1] = arr[2*2]; triC2[e1+1] = o < 0 ? 1 : 0;
  triO2[9]  = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, -1, arr[2*2+1]);
  /* Swap the last two edges if the triangle is reversed */
  triC2[41] = arr[2*2]; triC2[42] = o < 0 ? 0 : 1;
  triC2[45] = o < 0 ? 1 : 0;
  triC2[48] = o < 0 ? 0 : 1;
  triO2[11] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, 0, arr[1*2+1]);
  triO2[12] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, 0, arr[2*2+1]);
  *Nt = 2; *target = triT2; *size = triS2; *cone = triC2; *ornt = triO2;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellTransform_SBR(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMLabel        trType = tr->trType;
  PetscInt       val;

  PetscFunctionBeginHot;
  PetscCheck(p >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point argument is invalid");
  PetscCall(DMLabelGetValue(trType, p, &val));
  if (rt) *rt = val;
  switch (source) {
    case DM_POLYTOPE_POINT:
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    case DM_POLYTOPE_QUADRILATERAL:
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
    case DM_POLYTOPE_TETRAHEDRON:
    case DM_POLYTOPE_HEXAHEDRON:
    case DM_POLYTOPE_TRI_PRISM:
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
    case DM_POLYTOPE_PYRAMID:
      PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
      break;
    case DM_POLYTOPE_SEGMENT:
      if (val == RT_EDGE) PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
      else                PetscCall(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
      break;
    case DM_POLYTOPE_TRIANGLE:
      switch (val) {
        case RT_TRIANGLE_SPLIT_0: PetscCall(SBRGetTriangleSplitSingle(2, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT_1: PetscCall(SBRGetTriangleSplitSingle(0, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT_2: PetscCall(SBRGetTriangleSplitSingle(1, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT_21: PetscCall(SBRGetTriangleSplitDouble(-3, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT_10: PetscCall(SBRGetTriangleSplitDouble(-2, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT_02: PetscCall(SBRGetTriangleSplitDouble(-1, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT_12: PetscCall(SBRGetTriangleSplitDouble( 0, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT_20: PetscCall(SBRGetTriangleSplitDouble( 1, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT_01: PetscCall(SBRGetTriangleSplitDouble( 2, Nt, target, size, cone, ornt));break;
        case RT_TRIANGLE_SPLIT: PetscCall(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt)); break;
        default: PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
      }
      break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetFromOptions_SBR(PetscOptionItems *PetscOptionsObject, DMPlexTransform tr)
{
  PetscInt       cells[256], n = 256, i;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 2);
  PetscOptionsHeadBegin(PetscOptionsObject,"DMPlex Options");
  PetscCall(PetscOptionsIntArray("-dm_plex_transform_sbr_ref_cell", "Mark cells for refinement", "", cells, &n, &flg));
  if (flg) {
    DMLabel active;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active));
    for (i = 0; i < n; ++i) PetscCall(DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE));
    PetscCall(DMPlexTransformSetActive(tr, active));
    PetscCall(DMLabelDestroy(&active));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformView_SBR(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    PetscCall(PetscObjectGetName((PetscObject) tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SBR refinement %s\n", name ? name : ""));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(DMLabelView(tr->trType, viewer));
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_SBR(DMPlexTransform tr)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *) tr->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(sbr->edgeLen));
  PetscCall(PetscSectionDestroy(&sbr->secEdgeLen));
  PetscCall(DMLabelDestroy(&sbr->splitPoints));
  PetscCall(PetscFree(tr->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_SBR(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view           = DMPlexTransformView_SBR;
  tr->ops->setfromoptions = DMPlexTransformSetFromOptions_SBR;
  tr->ops->setup          = DMPlexTransformSetUp_SBR;
  tr->ops->destroy        = DMPlexTransformDestroy_SBR;
  tr->ops->celltransform  = DMPlexTransformCellTransform_SBR;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_SBR;
  tr->ops->mapcoordinates = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_SBR(DMPlexTransform tr)
{
  DMPlexRefine_SBR *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNewLog(tr, &f));
  tr->data = f;

  PetscCall(DMPlexTransformInitialize_SBR(tr));
  PetscCall(PetscCitationsRegister(SBRCitation, &SBRcite));
  PetscFunctionReturn(0);
}
