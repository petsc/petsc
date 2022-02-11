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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(size < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Queue size %D must be non-negative", size);
  ierr = PetscCalloc1(1, &q);CHKERRQ(ierr);
  q->size = size;
  ierr = PetscMalloc1(q->size, &q->points);CHKERRQ(ierr);
  q->num   = 0;
  q->front = 0;
  q->back  = q->size-1;
  *queue = q;
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueDestroy(PointQueue *queue)
{
  PointQueue     q = *queue;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(q->points);CHKERRQ(ierr);
  ierr = PetscFree(q);CHKERRQ(ierr);
  *queue = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueEnsureSize(PointQueue queue)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (queue->num < queue->size) PetscFunctionReturn(0);
  queue->size *= 2;
  ierr = PetscRealloc(queue->size * sizeof(PetscInt), &queue->points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueEnqueue(PointQueue queue, PetscInt p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PointQueueEnsureSize(queue);CHKERRQ(ierr);
  queue->back = (queue->back + 1) % queue->size;
  queue->points[queue->back] = p;
  ++queue->num;
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueDequeue(PointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheckFalse(!queue->num,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot dequeue from an empty queue");
  *p = queue->points[queue->front];
  queue->front = (queue->front + 1) % queue->size;
  --queue->num;
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode PointQueueFront(PointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheckFalse(!queue->num,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot get the front of an empty queue");
  *p = queue->points[queue->front];
  PetscFunctionReturn(0);
}

static PetscErrorCode PointQueueBack(PointQueue queue, PetscInt *p)
{
  PetscFunctionBegin;
  PetscCheckFalse(!queue->num,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot get the back of an empty queue");
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
  PetscErrorCode    ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(sbr->secEdgeLen, edge, &off);CHKERRQ(ierr);
  if (sbr->edgeLen[off] <= 0.0) {
    DM                 cdm;
    Vec                coordsLocal;
    const PetscScalar *coords;
    const PetscInt    *cone;
    PetscScalar       *cA, *cB;
    PetscInt           coneSize, cdim;

    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, edge, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, edge, &coneSize);CHKERRQ(ierr);
    PetscCheckFalse(coneSize != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Edge %D cone size must be 2, not %D", edge, coneSize);
    ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordsLocal, &coords);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(cdm, cone[0], coords, &cA);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(cdm, cone[1], coords, &cB);CHKERRQ(ierr);
    sbr->edgeLen[off] = DMPlex_DistD_Internal(cdim, cA, cB);
    ierr = VecRestoreArrayRead(coordsLocal, &coords);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  while (!PointQueueEmpty(queue)) {
    PetscInt        p = -1;
    const PetscInt *support;
    PetscInt        supportSize, s;

    ierr = PointQueueDequeue(queue, &p);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      const PetscInt  cell = support[s];
      const PetscInt *cone;
      PetscInt        coneSize, c;
      PetscInt        cval, eval, maxedge;
      PetscReal       len, maxlen;

      ierr = DMLabelGetValue(sbr->splitPoints, cell, &cval);CHKERRQ(ierr);
      if (cval == 2) continue;
      ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
      ierr = SBRGetEdgeLen_Private(tr, cone[0], &maxlen);CHKERRQ(ierr);
      maxedge = cone[0];
      for (c = 1; c < coneSize; ++c) {
        ierr = SBRGetEdgeLen_Private(tr, cone[c], &len);CHKERRQ(ierr);
        if (len > maxlen) {maxlen = len; maxedge = cone[c];}
      }
      ierr = DMLabelGetValue(sbr->splitPoints, maxedge, &eval);CHKERRQ(ierr);
      if (eval != 1) {
        ierr = DMLabelSetValue(sbr->splitPoints, maxedge, 1);CHKERRQ(ierr);
        ierr = PointQueueEnqueue(queue, maxedge);CHKERRQ(ierr);
      }
      ierr = DMLabelSetValue(sbr->splitPoints, cell, 2);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  /* Add in leaves */
  ierr = PetscSFGetGraph(pointSF, NULL, &Nl, &points, NULL);CHKERRQ(ierr);
  for (l = 0; l < Nl; ++l) {
    ierr = DMLabelGetValue(splitPoints, points[l], &val);CHKERRQ(ierr);
    if (val > 0) splitArray[points[l]] = val;
  }
  /* Add in shared roots */
  ierr = PetscSFComputeDegreeBegin(pointSF, &degree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(pointSF, &degree);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    if (degree[p]) {
      ierr = DMLabelGetValue(splitPoints, p, &val);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  /* Read out leaves */
  ierr = PetscSFGetGraph(pointSF, NULL, &Nl, &points, NULL);CHKERRQ(ierr);
  for (l = 0; l < Nl; ++l) {
    const PetscInt p    = points[l];
    const PetscInt cval = splitArray[p];

    if (cval) {
      ierr = DMLabelGetValue(splitPoints, p, &val);CHKERRQ(ierr);
      if (val <= 0) {
        ierr = DMLabelSetValue(splitPoints, p, cval);CHKERRQ(ierr);
        ierr = PointQueueEnqueue(queue, p);CHKERRQ(ierr);
      }
    }
  }
  /* Read out shared roots */
  ierr = PetscSFComputeDegreeBegin(pointSF, &degree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(pointSF, &degree);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    if (degree[p]) {
      const PetscInt cval = splitArray[p];

      if (cval) {
        ierr = DMLabelGetValue(splitPoints, p, &val);CHKERRQ(ierr);
        if (val <= 0) {
          ierr = DMLabelSetValue(splitPoints, p, cval);CHKERRQ(ierr);
          ierr = PointQueueEnqueue(queue, p);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Split Points", &sbr->splitPoints);CHKERRQ(ierr);
  /* Create edge lengths */
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &sbr->secEdgeLen);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sbr->secEdgeLen, eStart, eEnd);CHKERRQ(ierr);
  for (e = eStart; e < eEnd; ++e) {
    ierr = PetscSectionSetDof(sbr->secEdgeLen, e, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sbr->secEdgeLen);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(sbr->secEdgeLen, &edgeLenSize);CHKERRQ(ierr);
  ierr = PetscCalloc1(edgeLenSize, &sbr->edgeLen);CHKERRQ(ierr);
  /* Add edges of cells that are marked for refinement to edge queue */
  ierr = DMPlexTransformGetActive(tr, &active);CHKERRQ(ierr);
  PetscCheckFalse(!active,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_WRONGSTATE, "DMPlexTransform must have an adaptation label in order to use SBR algorithm");
  ierr = PointQueueCreate(1024, &queue);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(active, DM_ADAPT_REFINE, &refineIS);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(active, DM_ADAPT_REFINE, &Nc);CHKERRQ(ierr);
  if (refineIS) {ierr = ISGetIndices(refineIS, &refineCells);CHKERRQ(ierr);}
  for (c = 0; c < Nc; ++c) {
    const PetscInt cell = refineCells[c];
    PetscInt       depth;

    ierr = DMPlexGetPointDepth(dm, cell, &depth);CHKERRQ(ierr);
    if (depth == 1) {
      ierr = DMLabelSetValue(sbr->splitPoints, cell, 1);CHKERRQ(ierr);
      ierr = PointQueueEnqueue(queue, cell);CHKERRQ(ierr);
    } else {
      PetscInt *closure = NULL;
      PetscInt  Ncl, cl;

      ierr = DMLabelSetValue(sbr->splitPoints, cell, depth);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < Ncl; cl += 2) {
        const PetscInt edge = closure[cl];

        if (edge >= eStart && edge < eEnd) {
          ierr = DMLabelSetValue(sbr->splitPoints, edge, 1);CHKERRQ(ierr);
          ierr = PointQueueEnqueue(queue, edge);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
    }
  }
  if (refineIS) {ierr = ISRestoreIndices(refineIS, &refineCells);CHKERRQ(ierr);}
  ierr = ISDestroy(&refineIS);CHKERRQ(ierr);
  /* Setup communication */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  if (size > 1) {
    PetscInt pStart, pEnd;

    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscCalloc1(pEnd-pStart, &sbr->splitArray);CHKERRQ(ierr);
  }
  /* While edge queue is not empty: */
  empty = PointQueueEmpty(queue);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &empty, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
  while (!empty) {
    ierr = SBRSplitLocalEdges_Private(tr, queue);CHKERRQ(ierr);
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
      ierr = SBRInitializeComm(tr, pointSF);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(pointSF, MPIU_INT, sbr->splitArray, sbr->splitArray, MPI_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(pointSF, MPIU_INT, sbr->splitArray, sbr->splitArray, MPI_MAX);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(pointSF, MPIU_INT, sbr->splitArray, sbr->splitArray,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(pointSF, MPIU_INT, sbr->splitArray, sbr->splitArray,MPI_REPLACE);CHKERRQ(ierr);
      ierr = SBRFinalizeComm(tr, pointSF, queue);CHKERRQ(ierr);
    }
    empty = PointQueueEmpty(queue);
    ierr = MPI_Allreduce(MPI_IN_PLACE, &empty, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
  }
  ierr = PetscFree(sbr->splitArray);CHKERRQ(ierr);
  /* Calculate refineType for each cell */
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    DMLabel        trType = tr->trType;
    DMPolytopeType ct;
    PetscInt       val;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    switch (ct) {
      case DM_POLYTOPE_POINT:
        ierr = DMLabelSetValue(trType, p, RT_VERTEX);CHKERRQ(ierr);break;
      case DM_POLYTOPE_SEGMENT:
        ierr = DMLabelGetValue(sbr->splitPoints, p, &val);CHKERRQ(ierr);
        if (val == 1) {ierr = DMLabelSetValue(trType, p, RT_EDGE_SPLIT);CHKERRQ(ierr);}
        else          {ierr = DMLabelSetValue(trType, p, RT_EDGE);CHKERRQ(ierr);}
        break;
      case DM_POLYTOPE_TRIANGLE:
        ierr = DMLabelGetValue(sbr->splitPoints, p, &val);CHKERRQ(ierr);
        if (val == 2) {
          const PetscInt *cone;
          PetscReal       lens[3];
          PetscInt        vals[3], i;

          ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
          for (i = 0; i < 3; ++i) {
            ierr = DMLabelGetValue(sbr->splitPoints, cone[i], &vals[i]);CHKERRQ(ierr);
            vals[i] = vals[i] < 0 ? 0 : vals[i];
            ierr = SBRGetEdgeLen_Private(tr, cone[i], &lens[i]);CHKERRQ(ierr);
          }
          if (vals[0] && vals[1] && vals[2]) {ierr = DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT);CHKERRQ(ierr);}
          else if (vals[0] && vals[1])       {ierr = DMLabelSetValue(trType, p, lens[0] > lens[1] ? RT_TRIANGLE_SPLIT_01 : RT_TRIANGLE_SPLIT_10);CHKERRQ(ierr);}
          else if (vals[1] && vals[2])       {ierr = DMLabelSetValue(trType, p, lens[1] > lens[2] ? RT_TRIANGLE_SPLIT_12 : RT_TRIANGLE_SPLIT_21);CHKERRQ(ierr);}
          else if (vals[2] && vals[0])       {ierr = DMLabelSetValue(trType, p, lens[2] > lens[0] ? RT_TRIANGLE_SPLIT_20 : RT_TRIANGLE_SPLIT_02);CHKERRQ(ierr);}
          else if (vals[0])                  {ierr = DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_0);CHKERRQ(ierr);}
          else if (vals[1])                  {ierr = DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_1);CHKERRQ(ierr);}
          else if (vals[2])                  {ierr = DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_2);CHKERRQ(ierr);}
          else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %D does not fit any refinement type (%D, %D, %D)", p, vals[0], vals[1], vals[2]);
        } else {ierr = DMLabelSetValue(trType, p, RT_TRIANGLE);CHKERRQ(ierr);}
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle points of type %s", DMPolytopeTypes[ct]);
    }
    ierr = DMLabelGetValue(sbr->splitPoints, p, &val);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = PointQueueDestroy(&queue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_SBR(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscInt         rt;
  PetscErrorCode   ierr;

  PetscFunctionBeginHot;
  ierr = DMLabelGetValue(tr->trType, sp, &rt);CHKERRQ(ierr);
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
      ierr = DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew);CHKERRQ(ierr);
      break;
    default: ierr = DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscCheckFalse(p < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point argument is invalid");
  ierr = DMLabelGetValue(trType, p, &val);CHKERRQ(ierr);
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
      ierr = DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);
      break;
    case DM_POLYTOPE_SEGMENT:
      if (val == RT_EDGE) {ierr = DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);}
      else                {ierr = DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);}
      break;
    case DM_POLYTOPE_TRIANGLE:
      switch (val) {
        case RT_TRIANGLE_SPLIT_0: ierr = SBRGetTriangleSplitSingle(2, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT_1: ierr = SBRGetTriangleSplitSingle(0, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT_2: ierr = SBRGetTriangleSplitSingle(1, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT_21: ierr = SBRGetTriangleSplitDouble(-3, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT_10: ierr = SBRGetTriangleSplitDouble(-2, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT_02: ierr = SBRGetTriangleSplitDouble(-1, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT_12: ierr = SBRGetTriangleSplitDouble( 0, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT_20: ierr = SBRGetTriangleSplitDouble( 1, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT_01: ierr = SBRGetTriangleSplitDouble( 2, Nt, target, size, cone, ornt);CHKERRQ(ierr);break;
        case RT_TRIANGLE_SPLIT: ierr = DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr); break;
        default: ierr = DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 2);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPlex Options");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-dm_plex_transform_sbr_ref_cell", "Mark cells for refinement", "", cells, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    DMLabel active;

    ierr = DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {ierr = DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE);CHKERRQ(ierr);}
    ierr = DMPlexTransformSetActive(tr, active);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&active);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformView_SBR(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    ierr = PetscObjectGetName((PetscObject) tr, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "SBR refinement %s\n", name ? name : "");CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = DMLabelView(tr->trType, viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_SBR(DMPlexTransform tr)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *) tr->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sbr->edgeLen);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sbr->secEdgeLen);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&sbr->splitPoints);CHKERRQ(ierr);
  ierr = PetscFree(tr->data);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ierr = PetscNewLog(tr, &f);CHKERRQ(ierr);
  tr->data = f;

  ierr = DMPlexTransformInitialize_SBR(tr);CHKERRQ(ierr);
  ierr = PetscCitationsRegister(SBRCitation, &SBRcite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
