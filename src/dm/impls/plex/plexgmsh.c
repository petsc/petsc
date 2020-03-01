#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>

typedef struct {
  PetscViewer    viewer;
  int            fileFormat;
  int            dataSize;
  PetscBool      binary;
  PetscBool      byteSwap;
  size_t         wlen;
  void           *wbuf;
  size_t         slen;
  void           *sbuf;
} GmshFile;

static PetscErrorCode GmshBufferGet(GmshFile *gmsh, size_t count, size_t eltsize, void *buf)
{
  size_t         size = count * eltsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (gmsh->wlen < size) {
    ierr = PetscFree(gmsh->wbuf);CHKERRQ(ierr);
    ierr = PetscMalloc(size, &gmsh->wbuf);CHKERRQ(ierr);
    gmsh->wlen = size;
  }
  *(void**)buf = size ? gmsh->wbuf : NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshBufferSizeGet(GmshFile *gmsh, size_t count, void *buf)
{
  size_t         dataSize = (size_t)gmsh->dataSize;
  size_t         size = count * dataSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (gmsh->slen < size) {
    ierr = PetscFree(gmsh->sbuf);CHKERRQ(ierr);
    ierr = PetscMalloc(size, &gmsh->sbuf);CHKERRQ(ierr);
    gmsh->slen = size;
  }
  *(void**)buf = size ? gmsh->sbuf : NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshRead(GmshFile *gmsh, void *buf, PetscInt count, PetscDataType dtype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscViewerRead(gmsh->viewer, buf, count, NULL, dtype);CHKERRQ(ierr);
  if (gmsh->byteSwap) {ierr = PetscByteSwap(buf, dtype, count);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadString(GmshFile *gmsh, char *buf, PetscInt count)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscViewerRead(gmsh->viewer, buf, count, NULL, PETSC_STRING);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshMatch(PETSC_UNUSED GmshFile *gmsh, const char Section[], char line[PETSC_MAX_PATH_LEN], PetscBool *match)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscStrcmp(line, Section, match);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshExpect(GmshFile *gmsh, const char Section[], char line[PETSC_MAX_PATH_LEN])
{
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshMatch(gmsh, Section, line, &match);CHKERRQ(ierr);
  if (!match) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file, expecting %s",Section);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadSection(GmshFile *gmsh, char line[PETSC_MAX_PATH_LEN])
{
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (PETSC_TRUE) {
    ierr = GmshReadString(gmsh, line, 1);CHKERRQ(ierr);
    ierr = GmshMatch(gmsh, "$Comments", line, &match);CHKERRQ(ierr);
    if (!match) break;
    while (PETSC_TRUE) {
      ierr = GmshReadString(gmsh, line, 1);CHKERRQ(ierr);
      ierr = GmshMatch(gmsh, "$EndComments", line, &match);CHKERRQ(ierr);
      if (match) break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadEndSection(GmshFile *gmsh, const char EndSection[], char line[PETSC_MAX_PATH_LEN])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = GmshReadString(gmsh, line, 1);CHKERRQ(ierr);
  ierr = GmshExpect(gmsh, EndSection, line);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadSize(GmshFile *gmsh, PetscInt *buf, PetscInt count)
{
  PetscInt       i;
  size_t         dataSize = (size_t)gmsh->dataSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dataSize == sizeof(PetscInt)) {
    ierr = GmshRead(gmsh, buf, count, PETSC_INT);CHKERRQ(ierr);
  } else  if (dataSize == sizeof(int)) {
    int *ibuf = NULL;
    ierr = GmshBufferSizeGet(gmsh, count, &ibuf);CHKERRQ(ierr);
    ierr = GmshRead(gmsh, ibuf, count, PETSC_ENUM);CHKERRQ(ierr);
    for (i = 0; i < count; ++i) buf[i] = (PetscInt)ibuf[i];
  } else  if (dataSize == sizeof(long)) {
    long *ibuf = NULL;
    ierr = GmshBufferSizeGet(gmsh, count, &ibuf);CHKERRQ(ierr);
    ierr = GmshRead(gmsh, ibuf, count, PETSC_LONG);CHKERRQ(ierr);
    for (i = 0; i < count; ++i) buf[i] = (PetscInt)ibuf[i];
  } else if (dataSize == sizeof(PetscInt64)) {
    PetscInt64 *ibuf = NULL;
    ierr = GmshBufferSizeGet(gmsh, count, &ibuf);CHKERRQ(ierr);
    ierr = GmshRead(gmsh, ibuf, count, PETSC_INT64);CHKERRQ(ierr);
    for (i = 0; i < count; ++i) buf[i] = (PetscInt)ibuf[i];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadInt(GmshFile *gmsh, int *buf, PetscInt count)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = GmshRead(gmsh, buf, count, PETSC_ENUM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadDouble(GmshFile *gmsh, double *buf, PetscInt count)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = GmshRead(gmsh, buf, count, PETSC_DOUBLE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt id;       /* Entity number */
  PetscInt dim;      /* Entity dimension */
  double   bbox[6];  /* Bounding box */
  PetscInt numTags;  /* Size of tag array */
  int      tags[4];  /* Tag array */
} GmshEntity;

typedef struct {
  GmshEntity *entity[4];
  PetscHMapI  entityMap[4];
} GmshEntities;

static PetscErrorCode GmshEntitiesCreate(PetscInt count[4], GmshEntities **entities)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(entities);CHKERRQ(ierr);
  for (dim = 0; dim < 4; ++dim) {
    ierr = PetscCalloc1(count[dim], &(*entities)->entity[dim]);CHKERRQ(ierr);
    ierr = PetscHMapICreate(&(*entities)->entityMap[dim]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshEntitiesAdd(GmshEntities *entities, PetscInt index, PetscInt dim, PetscInt eid, GmshEntity** entity)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscHMapISet(entities->entityMap[dim], eid, index);CHKERRQ(ierr);
  entities->entity[dim][index].dim = dim;
  entities->entity[dim][index].id  = eid;
  if (entity) *entity = &entities->entity[dim][index];
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshEntitiesGet(GmshEntities *entities, PetscInt dim, PetscInt eid, GmshEntity** entity)
{
  PetscInt       index;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHMapIGet(entities->entityMap[dim], eid, &index);CHKERRQ(ierr);
  *entity = &entities->entity[dim][index];
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshEntitiesDestroy(GmshEntities **entities)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*entities) PetscFunctionReturn(0);
  for (dim = 0; dim < 4; ++dim) {
    ierr = PetscFree((*entities)->entity[dim]);CHKERRQ(ierr);
    ierr = PetscHMapIDestroy(&(*entities)->entityMap[dim]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*entities));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt id;       /* Entity number */
  PetscInt dim;      /* Entity dimension */
  PetscInt cellType; /* Cell type */
  PetscInt numNodes; /* Size of node array */
  PetscInt nodes[8]; /* Node array */
  PetscInt numTags;  /* Size of tag array */
  int      tags[4];  /* Tag array */
} GmshElement;

static PetscErrorCode DMPlexCreateGmsh_ReadNodes_v22(GmshFile *gmsh, int shift, PetscInt *numVertices, double **gmsh_nodes)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      byteSwap = gmsh->byteSwap;
  char           line[PETSC_MAX_PATH_LEN];
  int            v, num, nid, snum;
  double         *coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
  snum = sscanf(line, "%d", &num);
  if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
  ierr = PetscMalloc1(num*3, &coordinates);CHKERRQ(ierr);
  *numVertices = num;
  *gmsh_nodes = coordinates;
  for (v = 0; v < num; ++v) {
    double *xyz = coordinates + v*3;
    ierr = PetscViewerRead(viewer, &nid, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(&nid, PETSC_ENUM, 1);CHKERRQ(ierr);}
    if (nid != v+shift) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unexpected node number %d should be %d", nid, v+shift);
    ierr = PetscViewerRead(viewer, xyz, 3, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(xyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/* Gmsh elements can be of any dimension/co-dimension, so we need to traverse the
   file contents multiple times to figure out the true number of cells and facets
   in the given mesh. To make this more efficient we read the file contents only
   once and store them in memory, while determining the true number of cells. */
static PetscErrorCode DMPlexCreateGmsh_ReadElements_v22(GmshFile* gmsh, PETSC_UNUSED int shift, PetscInt *numCells, GmshElement **gmsh_elems)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      binary = gmsh->binary;
  PetscBool      byteSwap = gmsh->byteSwap;
  char           line[PETSC_MAX_PATH_LEN];
  GmshElement   *elements;
  int            i, c, p, num, ibuf[1+4+512], snum;
  int            cellType, dim, numNodes, numNodesIgnore, numElem, numTags;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
  snum = sscanf(line, "%d", &num);
  if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
  ierr = PetscMalloc1(num, &elements);CHKERRQ(ierr);
  *numCells = num;
  *gmsh_elems = elements;
  for (c = 0; c < num;) {
    ierr = PetscViewerRead(viewer, ibuf, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, 3);CHKERRQ(ierr);}
    if (binary) {
      cellType = ibuf[0];
      numElem = ibuf[1];
      numTags = ibuf[2];
    } else {
      elements[c].id = ibuf[0];
      cellType = ibuf[1];
      numTags = ibuf[2];
      numElem = 1;
    }
    /* http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format */
    numNodesIgnore = 0;
    switch (cellType) {
    case 1: /* 2-node line */
      dim = 1;
      numNodes = 2;
      break;
    case 2: /* 3-node triangle */
      dim = 2;
      numNodes = 3;
      break;
    case 3: /* 4-node quadrangle */
      dim = 2;
      numNodes = 4;
      break;
    case 4: /* 4-node tetrahedron */
      dim  = 3;
      numNodes = 4;
      break;
    case 5: /* 8-node hexahedron */
      dim = 3;
      numNodes = 8;
      break;
    case 6: /* 6-node wedge */
      dim = 3;
      numNodes = 6;
      break;
    case 8: /* 3-node 2nd order line */
      dim = 1;
      numNodes = 2;
      numNodesIgnore = 1;
      break;
    case 9: /* 6-node 2nd order triangle */
      dim = 2;
      numNodes = 3;
      numNodesIgnore = 3;
      break;
    case 10: /* 9-node 2nd order quadrangle */
      dim = 2;
      numNodes = 4;
      numNodesIgnore = 5;
      break;
    case 11: /* 10-node 2nd order tetrahedron */
      dim  = 3;
      numNodes = 4;
      numNodesIgnore = 6;
      break;
    case 12: /* 27-node 2nd order hexhedron */
      dim  = 3;
      numNodes = 8;
      numNodesIgnore = 19;
      break;
    case 13: /* 18-node 2nd wedge */
      dim = 3;
      numNodes = 6;
      numNodesIgnore = 12;
      break;
    case 15: /* 1-node vertex */
      dim = 0;
      numNodes = 1;
      break;
    case 7: /* 5-node pyramid */
    case 14: /* 14-node 2nd order pyramid */
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
    }
    if (binary) {
      const int nint = 1 + numTags + numNodes + numNodesIgnore;
      /* Loop over element blocks */
      for (i = 0; i < numElem; ++i, ++c) {
        ierr = PetscViewerRead(viewer, ibuf, nint, NULL, PETSC_ENUM);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, nint);CHKERRQ(ierr);}
        elements[c].dim = dim;
        elements[c].numNodes = numNodes;
        elements[c].numTags = numTags;
        elements[c].id = ibuf[0];
        elements[c].cellType = cellType;
        for (p = 0; p < numTags;  p++) elements[c].tags[p]  = ibuf[1 + p];
        for (p = 0; p < numNodes; p++) elements[c].nodes[p] = ibuf[1 + numTags + p];
      }
    } else {
      const int nint = numTags + numNodes + numNodesIgnore;
      elements[c].dim = dim;
      elements[c].numNodes = numNodes;
      elements[c].numTags = numTags;
      elements[c].cellType = cellType;
      ierr = PetscViewerRead(viewer, ibuf, nint, NULL, PETSC_ENUM);CHKERRQ(ierr);
      for (p = 0; p < numTags;  p++) elements[c].tags[p]  = ibuf[p];
      for (p = 0; p < numNodes; p++) elements[c].nodes[p] = ibuf[numTags + p];
      c++;
    }
  }
  PetscFunctionReturn(0);
}

/*
$Entities
  numPoints(unsigned long) numCurves(unsigned long)
    numSurfaces(unsigned long) numVolumes(unsigned long)
  // points
  tag(int) boxMinX(double) boxMinY(double) boxMinZ(double)
    boxMaxX(double) boxMaxY(double) boxMaxZ(double)
    numPhysicals(unsigned long) phyisicalTag[...](int)
  ...
  // curves
  tag(int) boxMinX(double) boxMinY(double) boxMinZ(double)
     boxMaxX(double) boxMaxY(double) boxMaxZ(double)
     numPhysicals(unsigned long) physicalTag[...](int)
     numBREPVert(unsigned long) tagBREPVert[...](int)
  ...
  // surfaces
  tag(int) boxMinX(double) boxMinY(double) boxMinZ(double)
    boxMaxX(double) boxMaxY(double) boxMaxZ(double)
    numPhysicals(unsigned long) physicalTag[...](int)
    numBREPCurve(unsigned long) tagBREPCurve[...](int)
  ...
  // volumes
  tag(int) boxMinX(double) boxMinY(double) boxMinZ(double)
    boxMaxX(double) boxMaxY(double) boxMaxZ(double)
    numPhysicals(unsigned long) physicalTag[...](int)
    numBREPSurfaces(unsigned long) tagBREPSurfaces[...](int)
  ...
$EndEntities
*/
static PetscErrorCode DMPlexCreateGmsh_ReadEntities_v40(GmshFile *gmsh, GmshEntities **entities)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      byteSwap = gmsh->byteSwap;
  long           index, num, lbuf[4];
  int            dim, eid, numTags, *ibuf, t;
  PetscInt       count[4], i;
  GmshEntity     *entity = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, lbuf, 4, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(lbuf, PETSC_LONG, 4);CHKERRQ(ierr);}
  for (i = 0; i < 4; ++i) count[i] = lbuf[i];
  ierr = GmshEntitiesCreate(count, entities);CHKERRQ(ierr);
  for (dim = 0; dim < 4; ++dim) {
    for (index = 0; index < count[dim]; ++index) {
      ierr = PetscViewerRead(viewer, &eid, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(&eid, PETSC_ENUM, 1);CHKERRQ(ierr);}
      ierr = GmshEntitiesAdd(*entities, (PetscInt)index, dim, eid, &entity);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, entity->bbox, 6, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(entity->bbox, PETSC_DOUBLE, 6);CHKERRQ(ierr);}
      ierr = PetscViewerRead(viewer, &num, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(&num, PETSC_LONG, 1);CHKERRQ(ierr);}
      ierr = GmshBufferGet(gmsh, num, sizeof(int), &ibuf);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, ibuf, num, NULL, PETSC_ENUM);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, num);CHKERRQ(ierr);}
      entity->numTags = numTags = (int) PetscMin(num, 4);
      for (t = 0; t < numTags; ++t) entity->tags[t] = ibuf[t];
      if (dim == 0) continue;
      ierr = PetscViewerRead(viewer, &num, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(&num, PETSC_LONG, 1);CHKERRQ(ierr);}
      ierr = GmshBufferGet(gmsh, num, sizeof(int), &ibuf);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, ibuf, num, NULL, PETSC_ENUM);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, num);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

/*
$Nodes
  numEntityBlocks(unsigned long) numNodes(unsigned long)
  tagEntity(int) dimEntity(int) typeNode(int) numNodes(unsigned long)
    tag(int) x(double) y(double) z(double)
    ...
  ...
$EndNodes
*/
static PetscErrorCode DMPlexCreateGmsh_ReadNodes_v40(GmshFile *gmsh, int shift, PetscInt *numVertices, double **gmsh_nodes)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      byteSwap = gmsh->byteSwap;
  long           block, node, v, numEntityBlocks, numTotalNodes, numNodes;
  int            info[3], nid;
  double         *coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, &numEntityBlocks, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(&numEntityBlocks, PETSC_LONG, 1);CHKERRQ(ierr);}
  ierr = PetscViewerRead(viewer, &numTotalNodes, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(&numTotalNodes, PETSC_LONG, 1);CHKERRQ(ierr);}
  ierr = PetscMalloc1(numTotalNodes*3, &coordinates);CHKERRQ(ierr);
  *numVertices = numTotalNodes;
  *gmsh_nodes = coordinates;
  for (v = 0, block = 0; block < numEntityBlocks; ++block) {
    ierr = PetscViewerRead(viewer, info, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
    ierr = PetscViewerRead(viewer, &numNodes, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(&numNodes, PETSC_LONG, 1);CHKERRQ(ierr);}
    if (gmsh->binary) {
      size_t nbytes = sizeof(int) + 3*sizeof(double);
      char   *cbuf = NULL; /* dummy value to prevent warning from compiler about possible unitilized value */
      ierr = GmshBufferGet(gmsh, numNodes, nbytes, &cbuf);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, cbuf, numNodes*nbytes, NULL, PETSC_CHAR);CHKERRQ(ierr);
      for (node = 0; node < numNodes; ++node, ++v) {
        char   *cnid = cbuf + node*nbytes, *cxyz = cnid + sizeof(int);
        double *xyz = coordinates + v*3;
        if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(cnid, PETSC_ENUM, 1);CHKERRQ(ierr);}
        if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(cxyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
        ierr = PetscMemcpy(&nid, cnid, sizeof(int));CHKERRQ(ierr);
        ierr = PetscMemcpy(xyz, cxyz, 3*sizeof(double));CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(&nid, PETSC_ENUM, 1);CHKERRQ(ierr);}
        if (byteSwap) {ierr = PetscByteSwap(xyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
        if ((long)nid != v+shift) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unexpected node number %d should be %ld", nid, v+shift);
      }
    } else {
      for (node = 0; node < numNodes; ++node, ++v) {
        double *xyz = coordinates + v*3;
        ierr = PetscViewerRead(viewer, &nid, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(&nid, PETSC_ENUM, 1);CHKERRQ(ierr);}
        if ((long)nid != v+shift) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unexpected node number %d should be %ld", nid, v+shift);
        ierr = PetscViewerRead(viewer, xyz, 3, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(xyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
$Elements
  numEntityBlocks(unsigned long) numElements(unsigned long)
  tagEntity(int) dimEntity(int) typeEle(int) numElements(unsigned long)
    tag(int) numVert[...](int)
    ...
  ...
$EndElements
*/
static PetscErrorCode DMPlexCreateGmsh_ReadElements_v40(GmshFile *gmsh, PETSC_UNUSED int shift, GmshEntities *entities, PetscInt *numCells, GmshElement **gmsh_elems)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      byteSwap = gmsh->byteSwap;
  long           c, block, numEntityBlocks, numTotalElements, elem, numElements;
  int            p, info[3], *ibuf = NULL;
  int            eid, dim, numTags, *tags, cellType, numNodes;
  GmshEntity     *entity = NULL;
  GmshElement    *elements;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, &numEntityBlocks, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(&numEntityBlocks, PETSC_LONG, 1);CHKERRQ(ierr);}
  ierr = PetscViewerRead(viewer, &numTotalElements, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(&numTotalElements, PETSC_LONG, 1);CHKERRQ(ierr);}
  ierr = PetscCalloc1(numTotalElements, &elements);CHKERRQ(ierr);
  *numCells = numTotalElements;
  *gmsh_elems = elements;
  for (c = 0, block = 0; block < numEntityBlocks; ++block) {
    ierr = PetscViewerRead(viewer, info, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(info, PETSC_ENUM, 3);CHKERRQ(ierr);}
    eid = info[0]; dim = info[1]; cellType = info[2];
    ierr = GmshEntitiesGet(entities, dim, eid, &entity);CHKERRQ(ierr);
    numTags = entity->numTags;
    tags = entity->tags;
    switch (cellType) {
    case 1: /* 2-node line */
      numNodes = 2;
      break;
    case 2: /* 3-node triangle */
      numNodes = 3;
      break;
    case 3: /* 4-node quadrangle */
      numNodes = 4;
      break;
    case 4: /* 4-node tetrahedron */
      numNodes = 4;
      break;
    case 5: /* 8-node hexahedron */
      numNodes = 8;
      break;
    case 6: /* 6-node wedge */
      numNodes = 6;
      break;
    case 15: /* 1-node vertex */
      numNodes = 1;
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
    }
    ierr = PetscViewerRead(viewer, &numElements, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(&numElements, PETSC_LONG, 1);CHKERRQ(ierr);}
    ierr = GmshBufferGet(gmsh, (1+numNodes)*numElements, sizeof(int), &ibuf);CHKERRQ(ierr);
    ierr = PetscViewerRead(viewer, ibuf, (1+numNodes)*numElements, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, (1+numNodes)*numElements);CHKERRQ(ierr);}
    for (elem = 0; elem < numElements; ++elem, ++c) {
      int *id = ibuf + elem*(1+numNodes), *nodes = id + 1;
      GmshElement *element = elements + c;
      element->dim = dim;
      element->cellType = cellType;
      element->numNodes = numNodes;
      element->numTags = numTags;
      element->id = id[0];
      for (p = 0; p < numNodes; p++) element->nodes[p] = nodes[p];
      for (p = 0; p < numTags;  p++) element->tags[p]  = tags[p];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateGmsh_ReadPeriodic_v40(GmshFile *gmsh, int shift, PetscInt slaveMap[], PetscBT bt)
{
  PetscViewer    viewer = gmsh->viewer;
  int            fileFormat = gmsh->fileFormat;
  PetscBool      binary = gmsh->binary;
  PetscBool      byteSwap = gmsh->byteSwap;
  int            numPeriodic, snum, i;
  char           line[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fileFormat == 22 || !binary) {
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d", &numPeriodic);
    if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
  } else {
    ierr = PetscViewerRead(viewer, &numPeriodic, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(&numPeriodic, PETSC_ENUM, 1);CHKERRQ(ierr);}
  }
  for (i = 0; i < numPeriodic; i++) {
    int    ibuf[3], slaveDim = -1, slaveTag = -1, masterTag = -1, slaveNode, masterNode;
    long   j, nNodes;
    double affine[16];

    if (fileFormat == 22 || !binary) {
      ierr = PetscViewerRead(viewer, line, 3, NULL, PETSC_STRING);CHKERRQ(ierr);
      snum = sscanf(line, "%d %d %d", &slaveDim, &slaveTag, &masterTag);
      if (snum != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    } else {
      ierr = PetscViewerRead(viewer, ibuf, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, 3);CHKERRQ(ierr);}
      slaveDim = ibuf[0]; slaveTag = ibuf[1]; masterTag = ibuf[2];
    }
    (void)slaveDim; (void)slaveTag; (void)masterTag; /* unused */

    if (fileFormat == 22 || !binary) {
      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
      snum = sscanf(line, "%ld", &nNodes);
      if (snum != 1) { /* discard transformation and try again */
        ierr = PetscViewerRead(viewer, line, -PETSC_MAX_PATH_LEN, NULL, PETSC_STRING);CHKERRQ(ierr);
        ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
        snum = sscanf(line, "%ld", &nNodes);
        if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
      }
    } else {
      ierr = PetscViewerRead(viewer, &nNodes, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(&nNodes, PETSC_LONG, 1);CHKERRQ(ierr);}
      if (nNodes == -1) { /* discard transformation and try again */
        ierr = PetscViewerRead(viewer, affine, 16, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
        ierr = PetscViewerRead(viewer, &nNodes, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(&nNodes, PETSC_LONG, 1);CHKERRQ(ierr);}
      }
    }

    for (j = 0; j < nNodes; j++) {
      if (fileFormat == 22 || !binary) {
        ierr = PetscViewerRead(viewer, line, 2, NULL, PETSC_STRING);CHKERRQ(ierr);
        snum = sscanf(line, "%d %d", &slaveNode, &masterNode);
        if (snum != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
      } else {
        ierr = PetscViewerRead(viewer, ibuf, 2, NULL, PETSC_ENUM);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, 2);CHKERRQ(ierr);}
        slaveNode = ibuf[0]; masterNode = ibuf[1];
      }
      slaveMap[slaveNode - shift] = masterNode - shift;
      ierr = PetscBTSet(bt, slaveNode - shift);CHKERRQ(ierr);
      ierr = PetscBTSet(bt, masterNode - shift);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
$Entities
  numPoints(size_t) numCurves(size_t)
    numSurfaces(size_t) numVolumes(size_t)
  pointTag(int) X(double) Y(double) Z(double)
    numPhysicalTags(size_t) physicalTag(int) ...
  ...
  curveTag(int) minX(double) minY(double) minZ(double)
    maxX(double) maxY(double) maxZ(double)
    numPhysicalTags(size_t) physicalTag(int) ...
    numBoundingPoints(size_t) pointTag(int) ...
  ...
  surfaceTag(int) minX(double) minY(double) minZ(double)
    maxX(double) maxY(double) maxZ(double)
    numPhysicalTags(size_t) physicalTag(int) ...
    numBoundingCurves(size_t) curveTag(int) ...
  ...
  volumeTag(int) minX(double) minY(double) minZ(double)
    maxX(double) maxY(double) maxZ(double)
    numPhysicalTags(size_t) physicalTag(int) ...
    numBoundngSurfaces(size_t) surfaceTag(int) ...
  ...
$EndEntities
*/
static PetscErrorCode DMPlexCreateGmsh_ReadEntities_v41(GmshFile *gmsh, GmshEntities **entities)
{
  PetscInt       count[4], index, numTags, i;
  int            dim, eid, *tags = NULL;
  GmshEntity     *entity = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadSize(gmsh, count, 4);CHKERRQ(ierr);
  ierr = GmshEntitiesCreate(count, entities);CHKERRQ(ierr);
  for (dim = 0; dim < 4; ++dim) {
    for (index = 0; index < count[dim]; ++index) {
      ierr = GmshReadInt(gmsh, &eid, 1);CHKERRQ(ierr);
      ierr = GmshEntitiesAdd(*entities, (PetscInt)index, dim, eid, &entity);CHKERRQ(ierr);
      ierr = GmshReadDouble(gmsh, entity->bbox, (dim == 0) ? 3 : 6);CHKERRQ(ierr);
      ierr = GmshReadSize(gmsh, &numTags, 1);CHKERRQ(ierr);
      ierr = GmshBufferGet(gmsh, numTags, sizeof(int), &tags);CHKERRQ(ierr);
      ierr = GmshReadInt(gmsh, tags, numTags);CHKERRQ(ierr);
      entity->numTags = PetscMin(numTags, 4);
      for (i = 0; i < entity->numTags; ++i) entity->tags[i] = tags[i];
      if (dim == 0) continue;
      ierr = GmshReadSize(gmsh, &numTags, 1);CHKERRQ(ierr);
      ierr = GmshBufferGet(gmsh, numTags, sizeof(int), &tags);CHKERRQ(ierr);
      ierr = GmshReadInt(gmsh, tags, numTags);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
$Nodes
  numEntityBlocks(size_t) numNodes(size_t)
    minNodeTag(size_t) maxNodeTag(size_t)
  entityDim(int) entityTag(int) parametric(int; 0 or 1) numNodesBlock(size_t)
    nodeTag(size_t)
    ...
    x(double) y(double) z(double)
       < u(double; if parametric and entityDim = 1 or entityDim = 2) >
       < v(double; if parametric and entityDim = 2) >
    ...
  ...
$EndNodes
*/
static PetscErrorCode DMPlexCreateGmsh_ReadNodes_v41(GmshFile *gmsh, int shift, PetscInt *numVertices, double **gmsh_nodes)
{
  int            info[3];
  PetscInt       sizes[4], numEntityBlocks, numNodes, numNodesBlock = 0, *nodeTag = NULL, block, node, i;
  double         *coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadSize(gmsh, sizes, 4);CHKERRQ(ierr);
  numEntityBlocks = sizes[0]; numNodes = sizes[1];
  ierr = PetscMalloc1(numNodes*3, &coordinates);CHKERRQ(ierr);
  *numVertices = numNodes;
  *gmsh_nodes = coordinates;
  for (block = 0, node = 0; block < numEntityBlocks; ++block, node += numNodesBlock) {
    ierr = GmshReadInt(gmsh, info, 3);CHKERRQ(ierr);
    ierr = GmshReadSize(gmsh, &numNodesBlock, 1);CHKERRQ(ierr);
    ierr = GmshBufferGet(gmsh, numNodesBlock, sizeof(PetscInt), &nodeTag);CHKERRQ(ierr);
    ierr = GmshReadSize(gmsh, nodeTag, numNodesBlock);CHKERRQ(ierr);
    for (i = 0; i < numNodesBlock; ++i) if (nodeTag[i] != node+i+shift) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unexpected node number %d should be %d", nodeTag[i], node+i+shift);
    if (info[2] != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Parametric coordinates not supported");
    ierr = GmshReadDouble(gmsh, coordinates+node*3, numNodesBlock*3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
$Elements
  numEntityBlocks(size_t) numElements(size_t)
    minElementTag(size_t) maxElementTag(size_t)
  entityDim(int) entityTag(int) elementType(int; see below) numElementsBlock(size_t)
    elementTag(size_t) nodeTag(size_t) ...
    ...
  ...
$EndElements
*/
static PetscErrorCode DMPlexCreateGmsh_ReadElements_v41(GmshFile *gmsh, PETSC_UNUSED int shift, GmshEntities *entities, PetscInt *numCells, GmshElement **gmsh_elems)
{
  int            info[3], eid, dim, cellType, *tags;
  PetscInt       sizes[4], *ibuf = NULL, numEntityBlocks, numElements, numBlockElements, numNodes, numTags, block, elem, c, p;
  GmshEntity     *entity = NULL;
  GmshElement    *elements;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadSize(gmsh, sizes, 4);CHKERRQ(ierr);
  numEntityBlocks = sizes[0]; numElements = sizes[1];
  ierr = PetscCalloc1(numElements, &elements);CHKERRQ(ierr);
  *numCells = numElements;
  *gmsh_elems = elements;
  for (c = 0, block = 0; block < numEntityBlocks; ++block) {
    ierr = GmshReadInt(gmsh, info, 3);CHKERRQ(ierr);
    dim = info[0]; eid = info[1]; cellType = info[2];
    ierr = GmshEntitiesGet(entities, dim, eid, &entity);CHKERRQ(ierr);
    numTags = entity->numTags;
    tags = entity->tags;
    switch (cellType) {
    case 1: /* 2-node line */
      numNodes = 2;
      break;
    case 2: /* 3-node triangle */
      numNodes = 3;
      break;
    case 3: /* 4-node quadrangle */
      numNodes = 4;
      break;
    case 4: /* 4-node tetrahedron */
      numNodes = 4;
      break;
    case 5: /* 8-node hexahedron */
      numNodes = 8;
      break;
    case 6: /* 6-node wedge */
      numNodes = 6;
      break;
    case 15: /* 1-node vertex */
      numNodes = 1;
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
    }
    ierr = GmshReadSize(gmsh, &numBlockElements, 1);CHKERRQ(ierr);
    ierr = GmshBufferGet(gmsh, (1+numNodes)*numBlockElements, sizeof(PetscInt), &ibuf);CHKERRQ(ierr);
    ierr = GmshReadSize(gmsh, ibuf, (1+numNodes)*numBlockElements);CHKERRQ(ierr);
    for (elem = 0; elem < numBlockElements; ++elem, ++c) {
      GmshElement *element = elements + c;
      PetscInt *id = ibuf + elem*(1+numNodes), *nodes = id + 1;
      element->id       = id[0];
      element->dim      = dim;
      element->cellType = cellType;
      element->numNodes = numNodes;
      element->numTags  = numTags;
      for (p = 0; p < numNodes; p++) element->nodes[p] = nodes[p];
      for (p = 0; p < numTags;  p++) element->tags[p]  = tags[p];
    }
  }
  PetscFunctionReturn(0);
}

/*
$Periodic
  numPeriodicLinks(size_t)
 entityDim(int) entityTag(int) entityTagMaster(int)
  numAffine(size_t) value(double) ...
  numCorrespondingNodes(size_t)
    nodeTag(size_t) nodeTagMaster(size_t)
    ...
  ...
$EndPeriodic
*/
static PetscErrorCode DMPlexCreateGmsh_ReadPeriodic_v41(GmshFile *gmsh, int shift, PetscInt slaveMap[], PetscBT bt)
{
  int            info[3];
  PetscInt       numPeriodicLinks, numAffine, numCorrespondingNodes, *nodeTags = NULL, link, node;
  double         dbuf[16];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadSize(gmsh, &numPeriodicLinks, 1);CHKERRQ(ierr);
  for (link = 0; link < numPeriodicLinks; ++link) {
    ierr = GmshReadInt(gmsh, info, 3);CHKERRQ(ierr);
    ierr = GmshReadSize(gmsh, &numAffine, 1);CHKERRQ(ierr);
    ierr = GmshReadDouble(gmsh, dbuf, numAffine);CHKERRQ(ierr);
    ierr = GmshReadSize(gmsh, &numCorrespondingNodes, 1);CHKERRQ(ierr);
    ierr = GmshBufferGet(gmsh, numCorrespondingNodes, sizeof(PetscInt), &nodeTags);CHKERRQ(ierr);
    ierr = GmshReadSize(gmsh, nodeTags, numCorrespondingNodes*2);CHKERRQ(ierr);
    for (node = 0; node < numCorrespondingNodes; ++node) {
      PetscInt slaveNode  = nodeTags[node*2+0] - shift;
      PetscInt masterNode = nodeTags[node*2+1] - shift;
      slaveMap[slaveNode] = masterNode;
      ierr = PetscBTSet(bt, slaveNode);CHKERRQ(ierr);
      ierr = PetscBTSet(bt, masterNode);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
$MeshFormat // same as MSH version 2
  version(ASCII double; currently 4.1)
  file-type(ASCII int; 0 for ASCII mode, 1 for binary mode)
  data-size(ASCII int; sizeof(size_t))
  < int with value one; only in binary mode, to detect endianness >
$EndMeshFormat
*/
static PetscErrorCode DMPlexCreateGmsh_ReadMeshFormat(GmshFile *gmsh)
{
  char           line[PETSC_MAX_PATH_LEN];
  int            snum, fileType, fileFormat, dataSize, checkEndian;
  float          version;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadString(gmsh, line, 3);CHKERRQ(ierr);
  snum = sscanf(line, "%f %d %d", &version, &fileType, &dataSize);
  if (snum != 3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse Gmsh file header: %s", line);
  if (version < 2.2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file version %3.1f must be at least 2.2", (double)version);
  if ((int)version == 3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file version %3.1f not supported", (double)version);
  if (version > 4.1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file version %3.1f must be at most 4.1", (double)version);
  if (gmsh->binary && !fileType) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Viewer is binary but Gmsh file is ASCII");
  if (!gmsh->binary && fileType) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Viewer is ASCII but Gmsh file is binary");
  fileFormat = (int)(version*10); /* XXX Should use (int)roundf(version*10) ? */
  if (fileFormat <= 40 && dataSize != sizeof(double)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data size %d is not valid for a Gmsh file", dataSize);
  if (fileFormat >= 41 && dataSize != sizeof(int) && dataSize != sizeof(PetscInt64)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data size %d is not valid for a Gmsh file", dataSize);
  gmsh->fileFormat = fileFormat;
  gmsh->dataSize = dataSize;
  gmsh->byteSwap = PETSC_FALSE;
  if (gmsh->binary) {
    ierr = GmshReadInt(gmsh, &checkEndian, 1);CHKERRQ(ierr);
    if (checkEndian != 1) {
      ierr = PetscByteSwap(&checkEndian, PETSC_ENUM, 1);CHKERRQ(ierr);
      if (checkEndian != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to detect endianness in Gmsh file header: %s", line);
      gmsh->byteSwap = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

/*
PhysicalNames
  numPhysicalNames(ASCII int)
  dimension(ASCII int) physicalTag(ASCII int) "name"(127 characters max)
  ...
$EndPhysicalNames
*/
static PetscErrorCode DMPlexCreateGmsh_ReadPhysicalNames(GmshFile *gmsh)
{
  char           line[PETSC_MAX_PATH_LEN], name[128+2], *p, *q;
  int            snum, numRegions, region, dim, tag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadString(gmsh, line, 1);CHKERRQ(ierr);
  snum = sscanf(line, "%d", &numRegions);
  if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
  for (region = 0; region < numRegions; ++region) {
    ierr = GmshReadString(gmsh, line, 2);CHKERRQ(ierr);
    snum = sscanf(line, "%d %d", &dim, &tag);
    if (snum != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = GmshReadString(gmsh, line, -(PetscInt)sizeof(line));CHKERRQ(ierr);
    ierr = PetscStrchr(line, '"', &p);CHKERRQ(ierr);
    if (!p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscStrrchr(line, '"', &q);CHKERRQ(ierr);
    if (q == p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscStrncpy(name, p+1, (size_t)(q-p-1));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateGmshFromFile - Create a DMPlex mesh from a Gmsh file

+ comm        - The MPI communicator
. filename    - Name of the Gmsh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Level: beginner

.seealso: DMPlexCreateFromFile(), DMPlexCreateGmsh(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateGmshFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscViewer     viewer;
  PetscMPIInt     rank;
  int             fileType;
  PetscViewerType vtype;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  /* Determine Gmsh file type (ASCII or binary) from file header */
  if (!rank) {
    GmshFile    gmsh_, *gmsh = &gmsh_;
    char        line[PETSC_MAX_PATH_LEN];
    int         snum;
    float       version;

    ierr = PetscArrayzero(gmsh,1);CHKERRQ(ierr);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &gmsh->viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(gmsh->viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(gmsh->viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(gmsh->viewer, filename);CHKERRQ(ierr);
    /* Read only the first two lines of the Gmsh file */
    ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    ierr = GmshExpect(gmsh, "$MeshFormat", line);CHKERRQ(ierr);
    ierr = GmshReadString(gmsh, line, 2);CHKERRQ(ierr);
    snum = sscanf(line, "%f %d", &version, &fileType);
    if (snum != 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse Gmsh file header: %s", line);
    if (version < 2.2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file version %3.1f must be at least 2.2", (double)version);
    if ((int)version == 3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file version %3.1f not supported", (double)version);
    if (version > 4.1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file version %3.1f must be at most 4.1", (double)version);
    ierr = PetscViewerDestroy(&gmsh->viewer);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(&fileType, 1, MPI_INT, 0, comm);CHKERRQ(ierr);
  vtype = (fileType == 0) ? PETSCVIEWERASCII : PETSCVIEWERBINARY;

  /* Create appropriate viewer and build plex */
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, vtype);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = DMPlexCreateGmsh(comm, viewer, interpolate, dm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateGmsh - Create a DMPlex mesh from a Gmsh file viewer

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. viewer - The Viewer associated with a Gmsh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format

  Level: beginner

.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateGmsh(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  PetscViewer    parentviewer = NULL;
  double        *coordsIn = NULL;
  GmshEntities  *entities = NULL;
  GmshElement   *gmsh_elem = NULL;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscBT        periodicV = NULL, periodicC = NULL;
  PetscScalar   *coords;
  PetscInt       dim = 0, embedDim = -1, coordSize, c, v, d, cell, *periodicMap = NULL, *periodicMapI = NULL, *hybridMap = NULL;
  PetscInt       numVertices = 0, numCells = 0, trueNumCells = 0;
  int            i, shift = 1;
  PetscMPIInt    rank;
  PetscBool      binary, zerobase = PETSC_FALSE, usemarker = PETSC_FALSE;
  PetscBool      enable_hybrid = interpolate, periodic = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPlex Gmsh options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_hybrid", "Generate hybrid cell bounds", "DMPlexCreateGmsh", enable_hybrid, &enable_hybrid, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_periodic","Read Gmsh periodic section", "DMPlexCreateGmsh", periodic, &periodic, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_use_marker", "Generate marker label", "DMPlexCreateGmsh", usemarker, &usemarker, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_zero_base", "Read Gmsh file with zero base indices", "DMPlexCreateGmsh", zerobase, &zerobase, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_plex_gmsh_spacedim", "Embedding space dimension", "DMPlexCreateGmsh", embedDim, &embedDim, NULL,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (zerobase) shift = 0;

  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_CreateGmsh,*dm,0,0,0);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &binary);CHKERRQ(ierr);

  /* Binary viewers read on all ranks, get subviewer to read only in rank 0 */
  if (binary) {
    parentviewer = viewer;
    ierr = PetscViewerGetSubViewer(parentviewer, PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
  }

  if (!rank) {
    GmshFile  gmsh_, *gmsh = &gmsh_;
    char      line[PETSC_MAX_PATH_LEN];
    PetscBool match;

    ierr = PetscArrayzero(gmsh,1);CHKERRQ(ierr);
    gmsh->viewer = viewer;
    gmsh->binary = binary;

    /* Read mesh format */
    ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    ierr = GmshExpect(gmsh, "$MeshFormat", line);CHKERRQ(ierr);
    ierr = DMPlexCreateGmsh_ReadMeshFormat(gmsh);CHKERRQ(ierr);
    ierr = GmshReadEndSection(gmsh, "$EndMeshFormat", line);CHKERRQ(ierr);

    /* OPTIONAL Read physical names */
    ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    ierr = GmshMatch(gmsh,"$PhysicalNames", line, &match);CHKERRQ(ierr);
    if (match) {
      ierr = DMPlexCreateGmsh_ReadPhysicalNames(gmsh);CHKERRQ(ierr);
      ierr = GmshReadEndSection(gmsh, "$EndPhysicalNames", line);CHKERRQ(ierr);
      /* Initial read for entity section */
      ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    }

    /* Read entities */
    if (gmsh->fileFormat >= 40) {
      ierr = GmshExpect(gmsh, "$Entities", line);CHKERRQ(ierr);
      switch (gmsh->fileFormat) {
      case 41: ierr = DMPlexCreateGmsh_ReadEntities_v41(gmsh, &entities);CHKERRQ(ierr); break;
      default: ierr = DMPlexCreateGmsh_ReadEntities_v40(gmsh, &entities);CHKERRQ(ierr); break;
      }
      ierr = GmshReadEndSection(gmsh, "$EndEntities", line);CHKERRQ(ierr);
      /* Initial read for nodes section */
      ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    }

    /* Read nodes */
    ierr = GmshExpect(gmsh, "$Nodes", line);CHKERRQ(ierr);
    switch (gmsh->fileFormat) {
    case 41: ierr = DMPlexCreateGmsh_ReadNodes_v41(gmsh, shift, &numVertices, &coordsIn);CHKERRQ(ierr); break;
    case 40: ierr = DMPlexCreateGmsh_ReadNodes_v40(gmsh, shift, &numVertices, &coordsIn);CHKERRQ(ierr); break;
    default: ierr = DMPlexCreateGmsh_ReadNodes_v22(gmsh, shift, &numVertices, &coordsIn);CHKERRQ(ierr); break;
    }
    ierr = GmshReadEndSection(gmsh, "$EndNodes", line);CHKERRQ(ierr);

    /* Read elements */
    ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    ierr = GmshExpect(gmsh, "$Elements", line);CHKERRQ(ierr);
    switch (gmsh->fileFormat) {
    case 41: ierr = DMPlexCreateGmsh_ReadElements_v41(gmsh, shift, entities, &numCells, &gmsh_elem);CHKERRQ(ierr); break;
    case 40: ierr = DMPlexCreateGmsh_ReadElements_v40(gmsh, shift, entities, &numCells, &gmsh_elem);CHKERRQ(ierr); break;
    default: ierr = DMPlexCreateGmsh_ReadElements_v22(gmsh, shift, &numCells, &gmsh_elem);CHKERRQ(ierr); break;
    }
    ierr = GmshReadEndSection(gmsh, "$EndElements", line);CHKERRQ(ierr);

    /* OPTIONAL Read periodic section */
    if (periodic) {
      ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
      ierr = GmshMatch(gmsh, "$Periodic", line, &periodic);CHKERRQ(ierr);
    }
    if (periodic) {
      PetscInt pVert, *periodicMapT, *aux;

      ierr = PetscMalloc1(numVertices, &periodicMapT);CHKERRQ(ierr);
      ierr = PetscBTCreate(numVertices, &periodicV);CHKERRQ(ierr);
      for (i = 0; i < numVertices; i++) periodicMapT[i] = i;

      ierr = GmshExpect(gmsh, "$Periodic", line);CHKERRQ(ierr);
      switch (gmsh->fileFormat) {
      case 41: ierr = DMPlexCreateGmsh_ReadPeriodic_v41(gmsh, shift, periodicMapT, periodicV);CHKERRQ(ierr); break;
      default: ierr = DMPlexCreateGmsh_ReadPeriodic_v40(gmsh, shift, periodicMapT, periodicV);CHKERRQ(ierr); break;
      }
      ierr = GmshReadEndSection(gmsh, "$EndPeriodic", line);CHKERRQ(ierr);

      /* we may have slaves of slaves */
      for (i = 0; i < numVertices; i++) {
        while (periodicMapT[periodicMapT[i]] != periodicMapT[i]) {
          periodicMapT[i] = periodicMapT[periodicMapT[i]];
        }
      }
      /* periodicMap : from old to new numbering (periodic vertices excluded)
         periodicMapI: from new to old numbering */
      ierr = PetscMalloc1(numVertices, &periodicMap);CHKERRQ(ierr);
      ierr = PetscMalloc1(numVertices, &periodicMapI);CHKERRQ(ierr);
      ierr = PetscMalloc1(numVertices, &aux);CHKERRQ(ierr);
      for (i = 0, pVert = 0; i < numVertices; i++) {
        if (periodicMapT[i] != i) {
          pVert++;
        } else {
          aux[i] = i - pVert;
          periodicMapI[i - pVert] = i;
        }
      }
      for (i = 0 ; i < numVertices; i++) {
        periodicMap[i] = aux[periodicMapT[i]];
      }
      ierr = PetscFree(periodicMapT);CHKERRQ(ierr);
      ierr = PetscFree(aux);CHKERRQ(ierr);
      /* remove periodic vertices */
      numVertices = numVertices - pVert;
    }

    ierr = GmshEntitiesDestroy(&entities);CHKERRQ(ierr);
    ierr = PetscFree(gmsh->wbuf);CHKERRQ(ierr);
    ierr = PetscFree(gmsh->sbuf);CHKERRQ(ierr);
  }

  if (parentviewer) {
    ierr = PetscViewerRestoreSubViewer(parentviewer, PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
  }

  if (!rank) {
    PetscBool hybrid   = PETSC_FALSE;
    PetscInt  cellType = -1;

    for (trueNumCells = 0, c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim > dim) {dim = gmsh_elem[c].dim; trueNumCells = 0; hybrid = PETSC_FALSE; cellType = -1;}
      if (gmsh_elem[c].dim < dim) continue;
      if (cellType == -1) cellType = gmsh_elem[c].cellType;
      /* different cell type indicate an hybrid mesh in PLEX */
      if (cellType != gmsh_elem[c].cellType) hybrid = PETSC_TRUE;
      /* wedges always indicate an hybrid mesh in PLEX */
      if (cellType == 6 || cellType == 13) hybrid = PETSC_TRUE;
      trueNumCells++;
    }
    /* Renumber cells for hybrid grids */
    if (hybrid && enable_hybrid) {
      PetscInt hc1 = 0, hc2 = 0, *hybridCells1 = NULL, *hybridCells2 = NULL;
      PetscInt cell, tn, *tp;
      int      n1 = 0,n2 = 0;

      ierr = PetscMalloc1(trueNumCells, &hybridCells1);CHKERRQ(ierr);
      ierr = PetscMalloc1(trueNumCells, &hybridCells2);CHKERRQ(ierr);
      for (cell = 0, c = 0; c < numCells; ++c) {
        if (gmsh_elem[c].dim == dim) {
          if (!n1) n1 = gmsh_elem[c].cellType;
          else if (!n2 && n1 != gmsh_elem[c].cellType) n2 = gmsh_elem[c].cellType;

          if      (gmsh_elem[c].cellType == n1) hybridCells1[hc1++] = cell;
          else if (gmsh_elem[c].cellType == n2) hybridCells2[hc2++] = cell;
          else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle more than 2 cell types");
          cell++;
        }
      }

      switch (n1) {
      case 2: /* triangles */
      case 9:
        switch (n2) {
        case 0: /* single type mesh */
        case 3: /* quads */
        case 10:
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      case 3: /* quadrilateral */
      case 10:
        switch (n2) {
        case 0: /* single type mesh */
        case 2: /* swap since we list simplices first */
        case 9:
          tn  = hc1;
          hc1 = hc2;
          hc2 = tn;

          tp           = hybridCells1;
          hybridCells1 = hybridCells2;
          hybridCells2 = tp;
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      case 4: /* tetrahedra */
      case 11:
        switch (n2) {
        case 0: /* single type mesh */
        case 6: /* wedges */
        case 13:
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      case 5: /* hexahedra */
      case 12:
        switch (n2) {
        case 0: /* single type mesh */
        case 6: /* wedges */
        case 13:
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      case 6: /* wedge */
      case 13:
        switch (n2) {
        case 0: /* single type mesh */
        case 4: /* tetrahedra: swap since we list simplices first */
        case 11:
        case 5: /* hexahedra */
        case 12:
          tn  = hc1;
          hc1 = hc2;
          hc2 = tn;

          tp           = hybridCells1;
          hybridCells1 = hybridCells2;
          hybridCells2 = tp;
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      default:
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
      }
      ierr = PetscMalloc1(trueNumCells, &hybridMap);CHKERRQ(ierr);
      for (cell = 0; cell < hc1; cell++) hybridMap[hybridCells1[cell]] = cell;
      for (cell = 0; cell < hc2; cell++) hybridMap[hybridCells2[cell]] = cell + hc1;
      ierr = PetscFree(hybridCells1);CHKERRQ(ierr);
      ierr = PetscFree(hybridCells2);CHKERRQ(ierr);
    }

  }

  /* Allocate the cell-vertex mesh */
  /*   We do not want this label automatically computed, instead we compute it here */
  ierr = DMCreateLabel(*dm, "celltype");CHKERRQ(ierr);
  ierr = DMPlexSetChart(*dm, 0, trueNumCells+numVertices);CHKERRQ(ierr);
  for (cell = 0, c = 0; c < numCells; ++c) {
    if (gmsh_elem[c].dim == dim) {
      ierr = DMPlexSetConeSize(*dm, hybridMap ? hybridMap[cell] : cell, gmsh_elem[c].numNodes);CHKERRQ(ierr);
      ierr = DMPlexSetCellType(*dm, hybridMap ? hybridMap[cell] : cell, DMPolytopeTypeFromGmsh(gmsh_elem[c].cellType));CHKERRQ(ierr);
      cell++;
    }
  }
  for (v = trueNumCells; v < trueNumCells+numVertices; ++v) {ierr = DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT);CHKERRQ(ierr);}
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  /* Add cell-vertex connections */
  for (cell = 0, c = 0; c < numCells; ++c) {
    if (gmsh_elem[c].dim == dim) {
      PetscInt pcone[8], corner;
      for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
        const PetscInt cc = gmsh_elem[c].nodes[corner] - shift;
        pcone[corner] = (periodicMap ? periodicMap[cc] : cc) + trueNumCells;
      }
      if (dim == 3) {
        /* Tetrahedra are inverted */
        if (gmsh_elem[c].cellType == 4 || gmsh_elem[c].cellType == 11) {
          PetscInt tmp = pcone[0];
          pcone[0] = pcone[1];
          pcone[1] = tmp;
        }
        /* Hexahedra are inverted */
        if (gmsh_elem[c].cellType == 5 || gmsh_elem[c].cellType == 12) {
          PetscInt tmp = pcone[1];
          pcone[1] = pcone[3];
          pcone[3] = tmp;
        }
        /* Prisms are inverted */
        if (gmsh_elem[c].cellType == 6 || gmsh_elem[c].cellType == 13) {
          PetscInt tmp;

          tmp      = pcone[1];
          pcone[1] = pcone[2];
          pcone[2] = tmp;
        }
      }
      ierr = DMPlexSetCone(*dm, hybridMap ? hybridMap[cell] : cell, pcone);CHKERRQ(ierr);
      cell++;
    }
  }
  ierr = MPI_Bcast(&dim, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }

  if (usemarker && !interpolate && dim > 1) SETERRQ(comm,PETSC_ERR_SUP,"Cannot create marker label without interpolation");
  if (!rank && usemarker) {
    PetscInt f, fStart, fEnd;

    ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(*dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      PetscInt suppSize;

      ierr = DMPlexGetSupportSize(*dm, f, &suppSize);CHKERRQ(ierr);
      if (suppSize == 1) {
        PetscInt *cone = NULL, coneSize, p;

        ierr = DMPlexGetTransitiveClosure(*dm, f, PETSC_TRUE, &coneSize, &cone);CHKERRQ(ierr);
        for (p = 0; p < coneSize; p += 2) {
          ierr = DMSetLabelValue(*dm, "marker", cone[p], 1);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreTransitiveClosure(*dm, f, PETSC_TRUE, &coneSize, &cone);CHKERRQ(ierr);
      }
    }
  }

  if (!rank) {
    PetscInt vStart, vEnd;

    ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    for (cell = 0, c = 0; c < numCells; ++c) {

      /* Create face sets */
      if (interpolate && gmsh_elem[c].dim == dim-1) {
        const PetscInt *join;
        PetscInt        joinSize, pcone[8], corner;
        /* Find the relevant facet with vertex joins */
        for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
          const PetscInt cc = gmsh_elem[c].nodes[corner] - shift;
          pcone[corner] = (periodicMap ? periodicMap[cc] : cc) + vStart;
        }
        ierr = DMPlexGetFullJoin(*dm, gmsh_elem[c].numNodes, pcone, &joinSize, &join);CHKERRQ(ierr);
        if (joinSize != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for GMsh element %d (Plex cell %D)", gmsh_elem[c].id, c);
        ierr = DMSetLabelValue(*dm, "Face Sets", join[0], gmsh_elem[c].tags[0]);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(*dm, gmsh_elem[c].numNodes, (const PetscInt *) pcone, &joinSize, &join);CHKERRQ(ierr);
      }

      /* Create cell sets */
      if (gmsh_elem[c].dim == dim) {
        if (gmsh_elem[c].numTags > 0) {
          ierr = DMSetLabelValue(*dm, "Cell Sets", hybridMap ? hybridMap[cell] : cell, gmsh_elem[c].tags[0]);CHKERRQ(ierr);
        }
        cell++;
      }

      /* Create vertex sets */
      if (gmsh_elem[c].dim == 0) {
        if (gmsh_elem[c].numTags > 0) {
          const PetscInt cc = gmsh_elem[c].nodes[0] - shift;
          const PetscInt vid = (periodicMap ? periodicMap[cc] : cc) + vStart;
          ierr = DMSetLabelValue(*dm, "Vertex Sets", vid, gmsh_elem[c].tags[0]);CHKERRQ(ierr);
        }
      }
    }
  }

  /* Create coordinates */
  if (embedDim < 0) embedDim = dim;
  ierr = DMSetCoordinateDim(*dm, embedDim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, embedDim);CHKERRQ(ierr);
  if (periodicMap) { /* we need to localize coordinates on cells */
    ierr = PetscSectionSetChart(coordSection, 0, trueNumCells + numVertices);CHKERRQ(ierr);
  } else {
    ierr = PetscSectionSetChart(coordSection, trueNumCells, trueNumCells + numVertices);CHKERRQ(ierr);
  }
  for (v = trueNumCells; v < trueNumCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, embedDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, embedDim);CHKERRQ(ierr);
  }
  if (periodicMap) {
    ierr = PetscBTCreate(trueNumCells, &periodicC);CHKERRQ(ierr);
    for (cell = 0, c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim == dim) {
        PetscInt  corner;
        PetscBool pc = PETSC_FALSE;
        for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
          pc = (PetscBool)(pc || PetscBTLookup(periodicV, gmsh_elem[c].nodes[corner] - shift));
        }
        if (pc) {
          PetscInt dof = gmsh_elem[c].numNodes*embedDim;
          PetscInt ucell = hybridMap ? hybridMap[cell] : cell;
          ierr = PetscBTSet(periodicC, ucell);CHKERRQ(ierr);
          ierr = PetscSectionSetDof(coordSection, ucell, dof);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldDof(coordSection, ucell, 0, dof);CHKERRQ(ierr);
        }
        cell++;
      }
    }
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, embedDim);CHKERRQ(ierr);
  ierr = VecSetType(coordinates, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (periodicMap) {
    PetscInt off;

    for (cell = 0, c = 0; c < numCells; ++c) {
      PetscInt pcone[8], corner;
      if (gmsh_elem[c].dim == dim) {
        PetscInt ucell = hybridMap ? hybridMap[cell] : cell;
        if (PetscUnlikely(PetscBTLookup(periodicC, ucell))) {
          for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
            pcone[corner] = gmsh_elem[c].nodes[corner] - shift;
          }
          if (dim == 3) {
            /* Tetrahedra are inverted */
            if (gmsh_elem[c].cellType == 4 || gmsh_elem[c].cellType == 11) {
              PetscInt tmp = pcone[0];
              pcone[0] = pcone[1];
              pcone[1] = tmp;
            }
            /* Hexahedra are inverted */
            if (gmsh_elem[c].cellType == 5 || gmsh_elem[c].cellType == 12) {
              PetscInt tmp = pcone[1];
              pcone[1] = pcone[3];
              pcone[3] = tmp;
            }
            /* Prisms are inverted */
            if (gmsh_elem[c].cellType == 6 || gmsh_elem[c].cellType == 13) {
              PetscInt tmp;

              tmp      = pcone[1];
              pcone[1] = pcone[2];
              pcone[2] = tmp;
            }
          }
          ierr = PetscSectionGetOffset(coordSection, ucell, &off);CHKERRQ(ierr);
          for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
            v = pcone[corner];
            for (d = 0; d < embedDim; ++d) {
              coords[off++] = (PetscReal) coordsIn[v*3+d];
            }
          }
        }
        cell++;
      }
    }
    for (v = 0; v < numVertices; ++v) {
      ierr = PetscSectionGetOffset(coordSection, v + trueNumCells, &off);CHKERRQ(ierr);
      for (d = 0; d < embedDim; ++d) {
        coords[off+d] = (PetscReal) coordsIn[periodicMapI[v]*3+d];
      }
    }
  } else {
    for (v = 0; v < numVertices; ++v) {
      for (d = 0; d < embedDim; ++d) {
        coords[v*embedDim+d] = (PetscReal) coordsIn[v*3+d];
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);

  periodic = periodicMap ? PETSC_TRUE : PETSC_FALSE;
  ierr = MPI_Bcast(&periodic, 1, MPIU_BOOL, 0, comm);CHKERRQ(ierr);
  ierr = DMSetPeriodicity(*dm, periodic, NULL, NULL, NULL);CHKERRQ(ierr);

  ierr = PetscFree(coordsIn);CHKERRQ(ierr);
  ierr = PetscFree(gmsh_elem);CHKERRQ(ierr);
  ierr = PetscFree(hybridMap);CHKERRQ(ierr);
  ierr = PetscFree(periodicMap);CHKERRQ(ierr);
  ierr = PetscFree(periodicMapI);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&periodicV);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&periodicC);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(DMPLEX_CreateGmsh,*dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
