#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>

typedef struct {
  int            cellType;
  DMPolytopeType polytope;
  int            dim;
  int            numVerts;
  int            order;
  int            numNodes;
} GmshCellInfo;

#define DM_POLYTOPE_VERTEX  DM_POLYTOPE_POINT
#define DM_POLYTOPE_PYRAMID DM_POLYTOPE_UNKNOWN

static const GmshCellInfo GmshCellTable[] = {
  { 15, DM_POLYTOPE_VERTEX,         0,  1,   0,    1},

  {  1, DM_POLYTOPE_SEGMENT,        1,  2,   1,    2},
  {  8, DM_POLYTOPE_SEGMENT,        1,  2,   2,    3},
  { 26, DM_POLYTOPE_SEGMENT,        1,  2,   3,    4},
  { 27, DM_POLYTOPE_SEGMENT,        1,  2,   4,    5},
  { 28, DM_POLYTOPE_SEGMENT,        1,  2,   5,    6},
  { 62, DM_POLYTOPE_SEGMENT,        1,  2,   6,    7},
  { 63, DM_POLYTOPE_SEGMENT,        1,  2,   7,    8},
  { 64, DM_POLYTOPE_SEGMENT,        1,  2,   8,    9},
  { 65, DM_POLYTOPE_SEGMENT,        1,  2,   9,   10},
  { 66, DM_POLYTOPE_SEGMENT,        1,  2,  10,   11},

  {  2, DM_POLYTOPE_TRIANGLE,       2,  3,   1,    3},
  {  9, DM_POLYTOPE_TRIANGLE,       2,  3,   2,    6},
  { 21, DM_POLYTOPE_TRIANGLE,       2,  3,   3,   10},
  { 23, DM_POLYTOPE_TRIANGLE,       2,  3,   4,   15},
  { 25, DM_POLYTOPE_TRIANGLE,       2,  3,   5,   21},
  { 42, DM_POLYTOPE_TRIANGLE,       2,  3,   6,   28},
  { 43, DM_POLYTOPE_TRIANGLE,       2,  3,   7,   36},
  { 44, DM_POLYTOPE_TRIANGLE,       2,  3,   8,   45},
  { 45, DM_POLYTOPE_TRIANGLE,       2,  3,   9,   55},
  { 46, DM_POLYTOPE_TRIANGLE,       2,  3,  10,   66},

  {  3, DM_POLYTOPE_QUADRILATERAL,  2,  4,   1,    4},
  { 10, DM_POLYTOPE_QUADRILATERAL,  2,  4,   2,    9},
  { 36, DM_POLYTOPE_QUADRILATERAL,  2,  4,   3,   16},
  { 37, DM_POLYTOPE_QUADRILATERAL,  2,  4,   4,   25},
  { 38, DM_POLYTOPE_QUADRILATERAL,  2,  4,   5,   36},
  { 47, DM_POLYTOPE_QUADRILATERAL,  2,  4,   6,   49},
  { 48, DM_POLYTOPE_QUADRILATERAL,  2,  4,   7,   64},
  { 49, DM_POLYTOPE_QUADRILATERAL,  2,  4,   8,   81},
  { 50, DM_POLYTOPE_QUADRILATERAL,  2,  4,   9,  100},
  { 51, DM_POLYTOPE_QUADRILATERAL,  2,  4,  10,  121},

  {  4, DM_POLYTOPE_TETRAHEDRON,    3,  4,   1,    4},
  { 11, DM_POLYTOPE_TETRAHEDRON,    3,  4,   2,   10},
  { 29, DM_POLYTOPE_TETRAHEDRON,    3,  4,   3,   20},
  { 30, DM_POLYTOPE_TETRAHEDRON,    3,  4,   4,   35},
  { 31, DM_POLYTOPE_TETRAHEDRON,    3,  4,   5,   56},
  { 71, DM_POLYTOPE_TETRAHEDRON,    3,  4,   6,   84},
  { 72, DM_POLYTOPE_TETRAHEDRON,    3,  4,   7,  120},
  { 73, DM_POLYTOPE_TETRAHEDRON,    3,  4,   8,  165},
  { 74, DM_POLYTOPE_TETRAHEDRON,    3,  4,   9,  220},
  { 75, DM_POLYTOPE_TETRAHEDRON,    3,  4,  10,  286},

  {  5, DM_POLYTOPE_HEXAHEDRON,     3,  8,   1,    8},
  { 12, DM_POLYTOPE_HEXAHEDRON,     3,  8,   2,   27},
  { 92, DM_POLYTOPE_HEXAHEDRON,     3,  8,   3,   64},
  { 93, DM_POLYTOPE_HEXAHEDRON,     3,  8,   4,  125},
  { 94, DM_POLYTOPE_HEXAHEDRON,     3,  8,   5,  216},
  { 95, DM_POLYTOPE_HEXAHEDRON,     3,  8,   6,  343},
  { 96, DM_POLYTOPE_HEXAHEDRON,     3,  8,   7,  512},
  { 97, DM_POLYTOPE_HEXAHEDRON,     3,  8,   8,  729},
  { 98, DM_POLYTOPE_HEXAHEDRON,     3,  8,   9, 1000},

  {  6, DM_POLYTOPE_TRI_PRISM,      3,  6,   1,    6},
  { 13, DM_POLYTOPE_TRI_PRISM,      3,  6,   2,   18},
  { 90, DM_POLYTOPE_TRI_PRISM,      3,  6,   3,   40},
  { 91, DM_POLYTOPE_TRI_PRISM,      3,  6,   4,   75},
  {106, DM_POLYTOPE_TRI_PRISM,      3,  6,   5,  126},
  {107, DM_POLYTOPE_TRI_PRISM,      3,  6,   6,  196},
  {108, DM_POLYTOPE_TRI_PRISM,      3,  6,   7,  288},
  {109, DM_POLYTOPE_TRI_PRISM,      3,  6,   8,  405},
  {110, DM_POLYTOPE_TRI_PRISM,      3,  6,   9,  550},

  {  7, DM_POLYTOPE_PYRAMID,        3,  5,   1,    5},
  { 14, DM_POLYTOPE_PYRAMID,        3,  5,   2,   14},
  {118, DM_POLYTOPE_PYRAMID,        3,  5,   3,   30},
  {119, DM_POLYTOPE_PYRAMID,        3,  5,   4,   55},
  {120, DM_POLYTOPE_PYRAMID,        3,  5,   5,   91},
  {121, DM_POLYTOPE_PYRAMID,        3,  5,   6,  140},
  {122, DM_POLYTOPE_PYRAMID,        3,  5,   7,  204},
  {123, DM_POLYTOPE_PYRAMID,        3,  5,   8,  285},
  {124, DM_POLYTOPE_PYRAMID,        3,  5,   9,  385},

#if 0
  { 20, DM_POLYTOPE_TRIANGLE,       2,  3,   3,    9},
  { 16, DM_POLYTOPE_QUADRILATERAL,  2,  4,   2,    8},
  { 17, DM_POLYTOPE_HEXAHEDRON,     3,  8,   2,   20},
  { 18, DM_POLYTOPE_TRI_PRISM,      3,  6,   2,   15},
  { 19, DM_POLYTOPE_PYRAMID,        3,  5,   2,   13},
#endif
};

static GmshCellInfo GmshCellMap[150];

static PetscErrorCode GmshCellInfoSetUp(void)
{
  size_t           i, n;
  static PetscBool called = PETSC_FALSE;

  if (called) return 0;
  PetscFunctionBegin;
  called = PETSC_TRUE;
  n = sizeof(GmshCellMap)/sizeof(GmshCellMap[0]);
  for (i = 0; i < n; ++i) {
    GmshCellMap[i].cellType = -1;
    GmshCellMap[i].polytope = DM_POLYTOPE_UNKNOWN;
  }
  n = sizeof(GmshCellTable)/sizeof(GmshCellTable[0]);
  for (i = 0; i < n; ++i) GmshCellMap[GmshCellTable[i].cellType] = GmshCellTable[i];
  PetscFunctionReturn(0);
}

#define GmshCellTypeCheck(ct) 0; do { \
    const int _ct_ = (int)ct; \
    if (_ct_ < 0 || _ct_ >= (int)(sizeof(GmshCellMap)/sizeof(GmshCellMap[0]))) \
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid Gmsh element type %d", _ct_); \
    if (GmshCellMap[_ct_].cellType != _ct_) \
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported Gmsh element type %d", _ct_); \
    if (GmshCellMap[_ct_].polytope == DM_POLYTOPE_UNKNOWN) \
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported Gmsh element type %d", _ct_); \
  } while(0)


typedef struct {
  PetscViewer  viewer;
  int          fileFormat;
  int          dataSize;
  PetscBool    binary;
  PetscBool    byteSwap;
  size_t       wlen;
  void        *wbuf;
  size_t       slen;
  void        *sbuf;
  PetscInt    *nbuf;
  PetscInt     nodeStart;
  PetscInt     nodeEnd;
  PetscInt    *nodeMap;
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
  if (!match) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file, expecting %s",Section);
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
  PetscInt id;       /* Entity ID */
  PetscInt dim;      /* Dimension */
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

typedef struct {
  PetscInt *id;   /* Node IDs */
  double   *xyz;  /* Coordinates */
} GmshNodes;

static PetscErrorCode GmshNodesCreate(PetscInt count, GmshNodes **nodes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(nodes);CHKERRQ(ierr);
  ierr = PetscMalloc1(count*1, &(*nodes)->id);CHKERRQ(ierr);
  ierr = PetscMalloc1(count*3, &(*nodes)->xyz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshNodesDestroy(GmshNodes **nodes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!*nodes) PetscFunctionReturn(0);
  ierr = PetscFree((*nodes)->id);CHKERRQ(ierr);
  ierr = PetscFree((*nodes)->xyz);CHKERRQ(ierr);
  ierr = PetscFree((*nodes));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt id;       /* Element ID */
  PetscInt dim;      /* Dimension */
  PetscInt cellType; /* Cell type */
  PetscInt numVerts; /* Size of vertex array */
  PetscInt numNodes; /* Size of node array */
  PetscInt *nodes;   /* Vertex/Node array */
  PetscInt numTags;  /* Size of tag array */
  int      tags[4];  /* Tag array */
} GmshElement;

static PetscErrorCode GmshElementsCreate(PetscInt count, GmshElement **elements)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(count, elements);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshElementsDestroy(GmshElement **elements)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*elements) PetscFunctionReturn(0);
  ierr = PetscFree(*elements);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt       dim;
  GmshEntities  *entities;
  PetscInt       numNodes;
  GmshNodes     *nodelist;
  PetscInt       numElems;
  GmshElement   *elements;
  PetscInt       numVerts;
  PetscInt       numCells;
  PetscInt      *periodMap;
  PetscInt      *vertexMap;
  PetscSegBuffer segbuf;
} GmshMesh;

static PetscErrorCode GmshMeshCreate(GmshMesh **mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(mesh);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscInt), 0, &(*mesh)->segbuf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshMeshDestroy(GmshMesh **mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*mesh) PetscFunctionReturn(0);
  ierr = GmshEntitiesDestroy(&(*mesh)->entities);CHKERRQ(ierr);
  ierr = GmshNodesDestroy(&(*mesh)->nodelist);CHKERRQ(ierr);
  ierr = GmshElementsDestroy(&(*mesh)->elements);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->periodMap);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->vertexMap);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&(*mesh)->segbuf);CHKERRQ(ierr);
  ierr = PetscFree((*mesh));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadNodes_v22(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      byteSwap = gmsh->byteSwap;
  char           line[PETSC_MAX_PATH_LEN];
  int            n, num, nid, snum;
  GmshNodes      *nodes;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
  snum = sscanf(line, "%d", &num);
  if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
  ierr = GmshNodesCreate(num, &nodes);CHKERRQ(ierr);
  mesh->numNodes = num;
  mesh->nodelist = nodes;
  for (n = 0; n < num; ++n) {
    double *xyz = nodes->xyz + n*3;
    ierr = PetscViewerRead(viewer, &nid, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
    ierr = PetscViewerRead(viewer, xyz, 3, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(&nid, PETSC_ENUM, 1);CHKERRQ(ierr);}
    if (byteSwap) {ierr = PetscByteSwap(xyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
    nodes->id[n] = nid;
  }
  PetscFunctionReturn(0);
}

/* Gmsh elements can be of any dimension/co-dimension, so we need to traverse the
   file contents multiple times to figure out the true number of cells and facets
   in the given mesh. To make this more efficient we read the file contents only
   once and store them in memory, while determining the true number of cells. */
static PetscErrorCode GmshReadElements_v22(GmshFile* gmsh, GmshMesh *mesh)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      binary = gmsh->binary;
  PetscBool      byteSwap = gmsh->byteSwap;
  char           line[PETSC_MAX_PATH_LEN];
  int            i, c, p, num, ibuf[1+4+1000], snum;
  int            cellType, numElem, numVerts, numNodes, numTags;
  GmshElement   *elements;
  PetscInt      *nodeMap = gmsh->nodeMap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
  snum = sscanf(line, "%d", &num);
  if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
  ierr = GmshElementsCreate(num, &elements);CHKERRQ(ierr);
  mesh->numElems = num;
  mesh->elements = elements;
  for (c = 0; c < num;) {
    ierr = PetscViewerRead(viewer, ibuf, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, 3);CHKERRQ(ierr);}

    cellType = binary ? ibuf[0] : ibuf[1];
    numElem  = binary ? ibuf[1] : 1;
    numTags  = ibuf[2];

    ierr = GmshCellTypeCheck(cellType);CHKERRQ(ierr);
    numVerts = GmshCellMap[cellType].numVerts;
    numNodes = GmshCellMap[cellType].numNodes;

    for (i = 0; i < numElem; ++i, ++c) {
      GmshElement *element = elements + c;
      const int off = binary ? 0 : 1, nint = 1 + numTags + numNodes - off;
      const int *id = ibuf, *nodes = ibuf + 1 + numTags, *tags = ibuf + 1;
      ierr = PetscViewerRead(viewer, ibuf+off, nint, NULL, PETSC_ENUM);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(ibuf+off, PETSC_ENUM, nint);CHKERRQ(ierr);}
      element->id  = id[0];
      element->dim = GmshCellMap[cellType].dim;
      element->cellType = cellType;
      element->numVerts = numVerts;
      element->numNodes = numNodes;
      element->numTags  = PetscMin(numTags, 4);
      ierr = PetscSegBufferGet(mesh->segbuf, (size_t)element->numNodes, &element->nodes);CHKERRQ(ierr);
      for (p = 0; p < element->numNodes; p++) element->nodes[p] = nodeMap[nodes[p]];
      for (p = 0; p < element->numTags;  p++) element->tags[p]  = tags[p];
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
static PetscErrorCode GmshReadEntities_v40(GmshFile *gmsh, GmshMesh *mesh)
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
  ierr = GmshEntitiesCreate(count, &mesh->entities);CHKERRQ(ierr);
  for (dim = 0; dim < 4; ++dim) {
    for (index = 0; index < count[dim]; ++index) {
      ierr = PetscViewerRead(viewer, &eid, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(&eid, PETSC_ENUM, 1);CHKERRQ(ierr);}
      ierr = GmshEntitiesAdd(mesh->entities, (PetscInt)index, dim, eid, &entity);CHKERRQ(ierr);
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
static PetscErrorCode GmshReadNodes_v40(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      byteSwap = gmsh->byteSwap;
  long           block, node, n, numEntityBlocks, numTotalNodes, numNodes;
  int            info[3], nid;
  GmshNodes      *nodes;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, &numEntityBlocks, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(&numEntityBlocks, PETSC_LONG, 1);CHKERRQ(ierr);}
  ierr = PetscViewerRead(viewer, &numTotalNodes, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(&numTotalNodes, PETSC_LONG, 1);CHKERRQ(ierr);}
  ierr = GmshNodesCreate(numTotalNodes, &nodes);CHKERRQ(ierr);
  mesh->numNodes = numTotalNodes;
  mesh->nodelist = nodes;
  for (n = 0, block = 0; block < numEntityBlocks; ++block) {
    ierr = PetscViewerRead(viewer, info, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
    ierr = PetscViewerRead(viewer, &numNodes, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(&numNodes, PETSC_LONG, 1);CHKERRQ(ierr);}
    if (gmsh->binary) {
      size_t nbytes = sizeof(int) + 3*sizeof(double);
      char   *cbuf = NULL; /* dummy value to prevent warning from compiler about possible unitilized value */
      ierr = GmshBufferGet(gmsh, numNodes, nbytes, &cbuf);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, cbuf, numNodes*nbytes, NULL, PETSC_CHAR);CHKERRQ(ierr);
      for (node = 0; node < numNodes; ++node, ++n) {
        char   *cnid = cbuf + node*nbytes, *cxyz = cnid + sizeof(int);
        double *xyz = nodes->xyz + n*3;
        if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(cnid, PETSC_ENUM, 1);CHKERRQ(ierr);}
        if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(cxyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
        ierr = PetscMemcpy(&nid, cnid, sizeof(int));CHKERRQ(ierr);
        ierr = PetscMemcpy(xyz, cxyz, 3*sizeof(double));CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(&nid, PETSC_ENUM, 1);CHKERRQ(ierr);}
        if (byteSwap) {ierr = PetscByteSwap(xyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
        nodes->id[n] = nid;
      }
    } else {
      for (node = 0; node < numNodes; ++node, ++n) {
        double *xyz = nodes->xyz + n*3;
        ierr = PetscViewerRead(viewer, &nid, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
        ierr = PetscViewerRead(viewer, xyz, 3, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(&nid, PETSC_ENUM, 1);CHKERRQ(ierr);}
        if (byteSwap) {ierr = PetscByteSwap(xyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
        nodes->id[n] = nid;
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
static PetscErrorCode GmshReadElements_v40(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      byteSwap = gmsh->byteSwap;
  long           c, block, numEntityBlocks, numTotalElements, elem, numElements;
  int            p, info[3], *ibuf = NULL;
  int            eid, dim, cellType, numVerts, numNodes, numTags;
  GmshEntity     *entity = NULL;
  GmshElement    *elements;
  PetscInt       *nodeMap = gmsh->nodeMap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerRead(viewer, &numEntityBlocks, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(&numEntityBlocks, PETSC_LONG, 1);CHKERRQ(ierr);}
  ierr = PetscViewerRead(viewer, &numTotalElements, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
  if (byteSwap) {ierr = PetscByteSwap(&numTotalElements, PETSC_LONG, 1);CHKERRQ(ierr);}
  ierr = GmshElementsCreate(numTotalElements, &elements);CHKERRQ(ierr);
  mesh->numElems = numTotalElements;
  mesh->elements = elements;
  for (c = 0, block = 0; block < numEntityBlocks; ++block) {
    ierr = PetscViewerRead(viewer, info, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(info, PETSC_ENUM, 3);CHKERRQ(ierr);}
    eid = info[0]; dim = info[1]; cellType = info[2];
    ierr = GmshEntitiesGet(mesh->entities, dim, eid, &entity);CHKERRQ(ierr);
    ierr = GmshCellTypeCheck(cellType);CHKERRQ(ierr);
    numVerts = GmshCellMap[cellType].numVerts;
    numNodes = GmshCellMap[cellType].numNodes;
    numTags  = entity->numTags;
    ierr = PetscViewerRead(viewer, &numElements, 1, NULL, PETSC_LONG);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(&numElements, PETSC_LONG, 1);CHKERRQ(ierr);}
    ierr = GmshBufferGet(gmsh, (1+numNodes)*numElements, sizeof(int), &ibuf);CHKERRQ(ierr);
    ierr = PetscViewerRead(viewer, ibuf, (1+numNodes)*numElements, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, (1+numNodes)*numElements);CHKERRQ(ierr);}
    for (elem = 0; elem < numElements; ++elem, ++c) {
      GmshElement *element = elements + c;
      const int *id = ibuf + elem*(1+numNodes), *nodes = id + 1;
      element->id  = id[0];
      element->dim = dim;
      element->cellType = cellType;
      element->numVerts = numVerts;
      element->numNodes = numNodes;
      element->numTags  = numTags;
      ierr = PetscSegBufferGet(mesh->segbuf, (size_t)element->numNodes, &element->nodes);CHKERRQ(ierr);
      for (p = 0; p < element->numNodes; p++) element->nodes[p] = nodeMap[nodes[p]];
      for (p = 0; p < element->numTags;  p++) element->tags[p]  = entity->tags[p];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadPeriodic_v40(GmshFile *gmsh, PetscInt periodicMap[])
{
  PetscViewer    viewer = gmsh->viewer;
  int            fileFormat = gmsh->fileFormat;
  PetscBool      binary = gmsh->binary;
  PetscBool      byteSwap = gmsh->byteSwap;
  int            numPeriodic, snum, i;
  char           line[PETSC_MAX_PATH_LEN];
  PetscInt       *nodeMap = gmsh->nodeMap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fileFormat == 22 || !binary) {
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d", &numPeriodic);
    if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
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
      if (snum != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
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
        if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
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
        if (snum != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
      } else {
        ierr = PetscViewerRead(viewer, ibuf, 2, NULL, PETSC_ENUM);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, 2);CHKERRQ(ierr);}
        slaveNode = ibuf[0]; masterNode = ibuf[1];
      }
      slaveNode  = (int) nodeMap[slaveNode];
      masterNode = (int) nodeMap[masterNode];
      periodicMap[slaveNode] = masterNode;
    }
  }
  PetscFunctionReturn(0);
}

/* http://gmsh.info/dev/doc/texinfo/gmsh.html#MSH-file-format
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
static PetscErrorCode GmshReadEntities_v41(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscInt       count[4], index, numTags, i;
  int            dim, eid, *tags = NULL;
  GmshEntity     *entity = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadSize(gmsh, count, 4);CHKERRQ(ierr);
  ierr = GmshEntitiesCreate(count, &mesh->entities);CHKERRQ(ierr);
  for (dim = 0; dim < 4; ++dim) {
    for (index = 0; index < count[dim]; ++index) {
      ierr = GmshReadInt(gmsh, &eid, 1);CHKERRQ(ierr);
      ierr = GmshEntitiesAdd(mesh->entities, (PetscInt)index, dim, eid, &entity);CHKERRQ(ierr);
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

/* http://gmsh.info/dev/doc/texinfo/gmsh.html#MSH-file-format
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
static PetscErrorCode GmshReadNodes_v41(GmshFile *gmsh, GmshMesh *mesh)
{
  int            info[3];
  PetscInt       sizes[4], numEntityBlocks, numNodes, numNodesBlock = 0, block, node;
  GmshNodes      *nodes;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadSize(gmsh, sizes, 4);CHKERRQ(ierr);
  numEntityBlocks = sizes[0]; numNodes = sizes[1];
  ierr = GmshNodesCreate(numNodes, &nodes);CHKERRQ(ierr);
  mesh->numNodes = numNodes;
  mesh->nodelist = nodes;
  for (block = 0, node = 0; block < numEntityBlocks; ++block, node += numNodesBlock) {
    ierr = GmshReadInt(gmsh, info, 3);CHKERRQ(ierr);
    if (info[2] != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Parametric coordinates not supported");
    ierr = GmshReadSize(gmsh, &numNodesBlock, 1);CHKERRQ(ierr);
    ierr = GmshReadSize(gmsh, nodes->id+node, numNodesBlock);CHKERRQ(ierr);
    ierr = GmshReadDouble(gmsh, nodes->xyz+node*3, numNodesBlock*3);CHKERRQ(ierr);
  }
  gmsh->nodeStart = sizes[2];
  gmsh->nodeEnd   = sizes[3]+1;
  PetscFunctionReturn(0);
}

/* http://gmsh.info/dev/doc/texinfo/gmsh.html#MSH-file-format
$Elements
  numEntityBlocks(size_t) numElements(size_t)
    minElementTag(size_t) maxElementTag(size_t)
  entityDim(int) entityTag(int) elementType(int; see below) numElementsBlock(size_t)
    elementTag(size_t) nodeTag(size_t) ...
    ...
  ...
$EndElements
*/
static PetscErrorCode GmshReadElements_v41(GmshFile *gmsh, GmshMesh *mesh)
{
  int            info[3], eid, dim, cellType;
  PetscInt       sizes[4], *ibuf = NULL, numEntityBlocks, numElements, numBlockElements, numVerts, numNodes, numTags, block, elem, c, p;
  GmshEntity     *entity = NULL;
  GmshElement    *elements;
  PetscInt       *nodeMap = gmsh->nodeMap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadSize(gmsh, sizes, 4);CHKERRQ(ierr);
  numEntityBlocks = sizes[0]; numElements = sizes[1];
  ierr = GmshElementsCreate(numElements, &elements);CHKERRQ(ierr);
  mesh->numElems = numElements;
  mesh->elements = elements;
  for (c = 0, block = 0; block < numEntityBlocks; ++block) {
    ierr = GmshReadInt(gmsh, info, 3);CHKERRQ(ierr);
    dim = info[0]; eid = info[1]; cellType = info[2];
    ierr = GmshEntitiesGet(mesh->entities, dim, eid, &entity);CHKERRQ(ierr);
    ierr = GmshCellTypeCheck(cellType);CHKERRQ(ierr);
    numVerts = GmshCellMap[cellType].numVerts;
    numNodes = GmshCellMap[cellType].numNodes;
    numTags  = entity->numTags;
    ierr = GmshReadSize(gmsh, &numBlockElements, 1);CHKERRQ(ierr);
    ierr = GmshBufferGet(gmsh, (1+numNodes)*numBlockElements, sizeof(PetscInt), &ibuf);CHKERRQ(ierr);
    ierr = GmshReadSize(gmsh, ibuf, (1+numNodes)*numBlockElements);CHKERRQ(ierr);
    for (elem = 0; elem < numBlockElements; ++elem, ++c) {
      GmshElement *element = elements + c;
      const PetscInt *id = ibuf + elem*(1+numNodes), *nodes = id + 1;
      element->id  = id[0];
      element->dim = dim;
      element->cellType = cellType;
      element->numVerts = numVerts;
      element->numNodes = numNodes;
      element->numTags  = numTags;
      ierr = PetscSegBufferGet(mesh->segbuf, (size_t)element->numNodes, &element->nodes);CHKERRQ(ierr);
      for (p = 0; p < element->numNodes; p++) element->nodes[p] = nodeMap[nodes[p]];
      for (p = 0; p < element->numTags;  p++) element->tags[p]  = entity->tags[p];
    }
  }
  PetscFunctionReturn(0);
}

/* http://gmsh.info/dev/doc/texinfo/gmsh.html#MSH-file-format
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
static PetscErrorCode GmshReadPeriodic_v41(GmshFile *gmsh, PetscInt periodicMap[])
{
  int            info[3];
  double         dbuf[16];
  PetscInt       numPeriodicLinks, numAffine, numCorrespondingNodes, *nodeTags = NULL, link, node;
  PetscInt       *nodeMap = gmsh->nodeMap;
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
      PetscInt slaveNode  = nodeMap[nodeTags[node*2+0]];
      PetscInt masterNode = nodeMap[nodeTags[node*2+1]];
      periodicMap[slaveNode] = masterNode;
    }
  }
  PetscFunctionReturn(0);
}

/* http://gmsh.info/dev/doc/texinfo/gmsh.html#MSH-file-format
$MeshFormat // same as MSH version 2
  version(ASCII double; currently 4.1)
  file-type(ASCII int; 0 for ASCII mode, 1 for binary mode)
  data-size(ASCII int; sizeof(size_t))
  < int with value one; only in binary mode, to detect endianness >
$EndMeshFormat
*/
static PetscErrorCode GmshReadMeshFormat(GmshFile *gmsh)
{
  char           line[PETSC_MAX_PATH_LEN];
  int            snum, fileType, fileFormat, dataSize, checkEndian;
  float          version;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadString(gmsh, line, 3);CHKERRQ(ierr);
  snum = sscanf(line, "%f %d %d", &version, &fileType, &dataSize);
  if (snum != 3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to parse Gmsh file header: %s", line);
  if (version < 2.2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at least 2.2", (double)version);
  if ((int)version == 3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f not supported", (double)version);
  if (version > 4.1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at most 4.1", (double)version);
  if (gmsh->binary && !fileType) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Viewer is binary but Gmsh file is ASCII");
  if (!gmsh->binary && fileType) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Viewer is ASCII but Gmsh file is binary");
  fileFormat = (int)(version*10); /* XXX Should use (int)roundf(version*10) ? */
  if (fileFormat <= 40 && dataSize != sizeof(double)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Data size %d is not valid for a Gmsh file", dataSize);
  if (fileFormat >= 41 && dataSize != sizeof(int) && dataSize != sizeof(PetscInt64)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Data size %d is not valid for a Gmsh file", dataSize);
  gmsh->fileFormat = fileFormat;
  gmsh->dataSize = dataSize;
  gmsh->byteSwap = PETSC_FALSE;
  if (gmsh->binary) {
    ierr = GmshReadInt(gmsh, &checkEndian, 1);CHKERRQ(ierr);
    if (checkEndian != 1) {
      ierr = PetscByteSwap(&checkEndian, PETSC_ENUM, 1);CHKERRQ(ierr);
      if (checkEndian != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to detect endianness in Gmsh file header: %s", line);
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
static PetscErrorCode GmshReadPhysicalNames(GmshFile *gmsh)
{
  char           line[PETSC_MAX_PATH_LEN], name[128+2], *p, *q;
  int            snum, numRegions, region, dim, tag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadString(gmsh, line, 1);CHKERRQ(ierr);
  snum = sscanf(line, "%d", &numRegions);
  if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
  for (region = 0; region < numRegions; ++region) {
    ierr = GmshReadString(gmsh, line, 2);CHKERRQ(ierr);
    snum = sscanf(line, "%d %d", &dim, &tag);
    if (snum != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    ierr = GmshReadString(gmsh, line, -(PetscInt)sizeof(line));CHKERRQ(ierr);
    ierr = PetscStrchr(line, '"', &p);CHKERRQ(ierr);
    if (!p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    ierr = PetscStrrchr(line, '"', &q);CHKERRQ(ierr);
    if (q == p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    ierr = PetscStrncpy(name, p+1, (size_t)(q-p-1));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadEntities(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (gmsh->fileFormat) {
  case 41: ierr = GmshReadEntities_v41(gmsh, mesh);CHKERRQ(ierr); break;
  default: ierr = GmshReadEntities_v40(gmsh, mesh);CHKERRQ(ierr); break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadNodes(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (gmsh->fileFormat) {
  case 41: ierr = GmshReadNodes_v41(gmsh, mesh);CHKERRQ(ierr); break;
  case 40: ierr = GmshReadNodes_v40(gmsh, mesh);CHKERRQ(ierr); break;
  default: ierr = GmshReadNodes_v22(gmsh, mesh);CHKERRQ(ierr); break;
  }

  { /* Gmsh v2.2/v4.0 does not provide min/max node tags */
    if (mesh->numNodes > 0 && gmsh->nodeEnd >= gmsh->nodeStart) {
      PetscInt  tagMin = PETSC_MAX_INT, tagMax = PETSC_MIN_INT, n;
      GmshNodes *nodes = mesh->nodelist;
      for (n = 0; n < mesh->numNodes; ++n) {
        const PetscInt tag = nodes->id[n];
        tagMin = PetscMin(tag, tagMin);
        tagMax = PetscMax(tag, tagMax);
      }
      gmsh->nodeStart = tagMin;
      gmsh->nodeEnd   = tagMax+1;
    }
  }

  { /* Support for sparse node tags */
    PetscInt  n, t;
    GmshNodes *nodes = mesh->nodelist;
    ierr = PetscMalloc1(gmsh->nodeEnd - gmsh->nodeStart, &gmsh->nbuf);CHKERRQ(ierr);
    for (t = 0; t < gmsh->nodeEnd - gmsh->nodeStart; ++t) gmsh->nbuf[t] = PETSC_MIN_INT;
    gmsh->nodeMap = gmsh->nbuf - gmsh->nodeStart;
    for (n = 0; n < mesh->numNodes; ++n) {
      const PetscInt tag = nodes->id[n];
      if (gmsh->nodeMap[tag] >= 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Repeated node tag %D", tag);
      gmsh->nodeMap[tag] = n;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadElements(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (gmsh->fileFormat) {
  case 41: ierr = GmshReadElements_v41(gmsh, mesh);CHKERRQ(ierr); break;
  case 40: ierr = GmshReadElements_v40(gmsh, mesh);CHKERRQ(ierr); break;
  default: ierr = GmshReadElements_v22(gmsh, mesh);CHKERRQ(ierr); break;
  }

  { /* Reorder elements by codimension and polytope type */
    PetscInt    ne = mesh->numElems;
    GmshElement *elements = mesh->elements;
    PetscInt    keymap[DM_NUM_POLYTOPES], nk = 0;
    PetscInt    offset[DM_NUM_POLYTOPES+1], e, k;

    for (k = 0; k < DM_NUM_POLYTOPES; ++k) keymap[k] = PETSC_MIN_INT;
    ierr = PetscMemzero(offset,sizeof(offset));CHKERRQ(ierr);

    keymap[DM_POLYTOPE_TETRAHEDRON]   = nk++;
    keymap[DM_POLYTOPE_HEXAHEDRON]    = nk++;
    keymap[DM_POLYTOPE_TRI_PRISM]     = nk++;
    keymap[DM_POLYTOPE_PYRAMID]       = nk++;
    keymap[DM_POLYTOPE_TRIANGLE]      = nk++;
    keymap[DM_POLYTOPE_QUADRILATERAL] = nk++;
    keymap[DM_POLYTOPE_SEGMENT]       = nk++;
    keymap[DM_POLYTOPE_VERTEX]        = nk++;
    keymap[DM_POLYTOPE_UNKNOWN]       = nk++;

    ierr = GmshElementsCreate(mesh->numElems, &mesh->elements);CHKERRQ(ierr);
#define key(eid) keymap[GmshCellMap[elements[(eid)].cellType].polytope]
    for (e = 0; e < ne; ++e) offset[1+key(e)]++;
    for (k = 1; k < nk; ++k) offset[k] += offset[k-1];
    for (e = 0; e < ne; ++e) mesh->elements[offset[key(e)]++] = elements[e];
#undef key
    ierr = GmshElementsDestroy(&elements);CHKERRQ(ierr);

    mesh->dim = mesh->numElems ? mesh->elements[0].dim : 0;
  }

  {
    PetscBT  vtx;
    PetscInt dim = mesh->dim, e, n, v;

    ierr = PetscBTCreate(mesh->numNodes, &vtx);CHKERRQ(ierr);

    /* Compute number of cells and set of vertices */
    mesh->numCells = 0;
    for (e = 0; e < mesh->numElems; ++e) {
      GmshElement *elem = mesh->elements + e;
      if (elem->dim == dim && dim > 0) mesh->numCells++;
      for (v = 0; v < elem->numVerts; v++) {
        ierr = PetscBTSet(vtx, elem->nodes[v]);CHKERRQ(ierr);
      }
    }

    /* Compute numbering for vertices */
    mesh->numVerts = 0;
    ierr = PetscMalloc1(mesh->numNodes, &mesh->vertexMap);CHKERRQ(ierr);
    for (n = 0; n < mesh->numNodes; ++n)
      mesh->vertexMap[n] = PetscBTLookup(vtx, n) ? mesh->numVerts++ : PETSC_MIN_INT;

    ierr = PetscBTDestroy(&vtx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadPeriodic(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(mesh->numNodes, &mesh->periodMap);CHKERRQ(ierr);
  for (n = 0; n < mesh->numNodes; ++n) mesh->periodMap[n] = n;
  switch (gmsh->fileFormat) {
  case 41: ierr = GmshReadPeriodic_v41(gmsh, mesh->periodMap);CHKERRQ(ierr); break;
  default: ierr = GmshReadPeriodic_v40(gmsh, mesh->periodMap);CHKERRQ(ierr); break;
  }

  /* Find canonical master nodes */
  for (n = 0; n < mesh->numNodes; ++n)
    while (mesh->periodMap[n] != mesh->periodMap[mesh->periodMap[n]])
      mesh->periodMap[n] = mesh->periodMap[mesh->periodMap[n]];

  /* Renumber vertices (filter out slaves) */
  mesh->numVerts = 0;
  for (n = 0; n < mesh->numNodes; ++n)
    if (mesh->vertexMap[n] >= 0)   /* is vertex */
      if (mesh->periodMap[n] == n) /* is master */
        mesh->vertexMap[n] = mesh->numVerts++;
  for (n = 0; n < mesh->numNodes; ++n)
    if (mesh->vertexMap[n] >= 0)   /* is vertex */
      if (mesh->periodMap[n] != n) /* is slave  */
        mesh->vertexMap[n] = mesh->vertexMap[mesh->periodMap[n]];
  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE DMPolytopeType DMPolytopeTypeFromGmsh(PetscInt cellType)
{
  return GmshCellMap[cellType].polytope;
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
    GmshFile    gmsh[1];
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
    if (snum != 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to parse Gmsh file header: %s", line);
    if (version < 2.2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at least 2.2", (double)version);
    if ((int)version == 3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f not supported", (double)version);
    if (version > 4.1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at most 4.1", (double)version);
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
  GmshMesh      *mesh = NULL;
  PetscViewer    parentviewer = NULL;
  PetscBT        periodicVerts = NULL;
  PetscBT        periodicCells = NULL;
  PetscSection   coordSection;
  Vec            coordinates;
  double        *coordsIn;
  PetscScalar   *coords;
  PetscInt       dim = 0, coordDim = -1;
  PetscInt       numNodes = 0, numElems = 0, numVerts = 0, numCells = 0;
  PetscInt       coordSize, *vertexMapInv, cell, cone[8], e, n, v, d;
  PetscBool      binary, usemarker = PETSC_FALSE;
  PetscBool      hybrid = interpolate, periodic = PETSC_TRUE;
  PetscBool      hasTetra = PETSC_FALSE;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPlex Gmsh options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_hybrid", "Generate hybrid cell bounds", "DMPlexCreateGmsh", hybrid, &hybrid, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_periodic","Read Gmsh periodic section", "DMPlexCreateGmsh", periodic, &periodic, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_use_marker", "Generate marker label", "DMPlexCreateGmsh", usemarker, &usemarker, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_plex_gmsh_spacedim", "Embedding space dimension", "DMPlexCreateGmsh", coordDim, &coordDim, NULL,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = GmshCellInfoSetUp();CHKERRQ(ierr);

  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_CreateGmsh,*dm,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &binary);CHKERRQ(ierr);

  /* Binary viewers read on all ranks, get subviewer to read only in rank 0 */
  if (binary) {
    parentviewer = viewer;
    ierr = PetscViewerGetSubViewer(parentviewer, PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
  }

  if (!rank) {
    GmshFile  gmsh[1];
    char      line[PETSC_MAX_PATH_LEN];
    PetscBool match;

    ierr = PetscArrayzero(gmsh,1);CHKERRQ(ierr);
    gmsh->viewer = viewer;
    gmsh->binary = binary;

    ierr = GmshMeshCreate(&mesh);CHKERRQ(ierr);

    /* Read mesh format */
    ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    ierr = GmshExpect(gmsh, "$MeshFormat", line);CHKERRQ(ierr);
    ierr = GmshReadMeshFormat(gmsh);CHKERRQ(ierr);
    ierr = GmshReadEndSection(gmsh, "$EndMeshFormat", line);CHKERRQ(ierr);

    /* OPTIONAL Read physical names */
    ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    ierr = GmshMatch(gmsh, "$PhysicalNames", line, &match);CHKERRQ(ierr);
    if (match) {
      ierr = GmshExpect(gmsh, "$PhysicalNames", line);CHKERRQ(ierr);
      ierr = GmshReadPhysicalNames(gmsh);CHKERRQ(ierr);
      ierr = GmshReadEndSection(gmsh, "$EndPhysicalNames", line);CHKERRQ(ierr);
      /* Initial read for entity section */
      ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    }

    /* Read entities */
    if (gmsh->fileFormat >= 40) {
      ierr = GmshExpect(gmsh, "$Entities", line);CHKERRQ(ierr);
      ierr = GmshReadEntities(gmsh, mesh);CHKERRQ(ierr);
      ierr = GmshReadEndSection(gmsh, "$EndEntities", line);CHKERRQ(ierr);
      /* Initial read for nodes section */
      ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    }

    /* Read nodes */
    ierr = GmshExpect(gmsh, "$Nodes", line);CHKERRQ(ierr);
    ierr = GmshReadNodes(gmsh, mesh);CHKERRQ(ierr);
    ierr = GmshReadEndSection(gmsh, "$EndNodes", line);CHKERRQ(ierr);

    /* Read elements */
    ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
    ierr = GmshExpect(gmsh, "$Elements", line);CHKERRQ(ierr);
    ierr = GmshReadElements(gmsh, mesh);CHKERRQ(ierr);
    ierr = GmshReadEndSection(gmsh, "$EndElements", line);CHKERRQ(ierr);

    /* Read periodic section (OPTIONAL) */
    if (periodic) {
      ierr = GmshReadSection(gmsh, line);CHKERRQ(ierr);
      ierr = GmshMatch(gmsh, "$Periodic", line, &periodic);CHKERRQ(ierr);
    }
    if (periodic) {
      ierr = GmshExpect(gmsh, "$Periodic", line);CHKERRQ(ierr);
      ierr = GmshReadPeriodic(gmsh, mesh);CHKERRQ(ierr);
      ierr = GmshReadEndSection(gmsh, "$EndPeriodic", line);CHKERRQ(ierr);
    }

    ierr = PetscFree(gmsh->wbuf);CHKERRQ(ierr);
    ierr = PetscFree(gmsh->sbuf);CHKERRQ(ierr);
    ierr = PetscFree(gmsh->nbuf);CHKERRQ(ierr);

    dim       = mesh->dim;
    numNodes  = mesh->numNodes;
    numElems  = mesh->numElems;
    numVerts  = mesh->numVerts;
    numCells  = mesh->numCells;
  }

  if (parentviewer) {
    ierr = PetscViewerRestoreSubViewer(parentviewer, PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
  }

  ierr = MPI_Bcast(&dim, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&periodic, 1, MPIU_BOOL, 0, comm);CHKERRQ(ierr);

  /* Flag presence of tetrahedra to special case wedges */
  for (cell = 0; cell < numCells; ++cell) {
    GmshElement *elem = mesh->elements + cell;
    DMPolytopeType ctype = DMPolytopeTypeFromGmsh(elem->cellType);
    if (ctype == DM_POLYTOPE_TETRAHEDRON) hasTetra = PETSC_TRUE;
  }

  /* We do not want this label automatically computed, instead we fill it here */
  ierr = DMCreateLabel(*dm, "celltype");CHKERRQ(ierr);

  /* Allocate the cell-vertex mesh */
  ierr = DMPlexSetChart(*dm, 0, numCells+numVerts);CHKERRQ(ierr);
  for (cell = 0; cell < numCells; ++cell) {
    GmshElement *elem = mesh->elements + cell;
    DMPolytopeType ctype = DMPolytopeTypeFromGmsh(elem->cellType);
    if (hybrid && hasTetra && ctype == DM_POLYTOPE_TRI_PRISM) ctype = DM_POLYTOPE_TRI_PRISM_TENSOR;
    ierr = DMPlexSetConeSize(*dm, cell, elem->numVerts);CHKERRQ(ierr);
    ierr = DMPlexSetCellType(*dm, cell, ctype);CHKERRQ(ierr);
  }
  for (v = numCells; v < numCells+numVerts; ++v) {
    ierr = DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT);CHKERRQ(ierr);
  }
  ierr = DMSetUp(*dm);CHKERRQ(ierr);

  /* Add cell-vertex connections */
  for (cell = 0; cell < numCells; ++cell) {
    GmshElement *elem = mesh->elements + cell;
    for (v = 0; v < elem->numVerts; ++v) {
      const PetscInt nn = elem->nodes[v];
      const PetscInt vv = mesh->vertexMap[nn];
      cone[v] = numCells + vv;
    }
    ierr = DMPlexReorderCell(*dm, cell, cone);CHKERRQ(ierr);
    ierr = DMPlexSetCone(*dm, cell, cone);CHKERRQ(ierr);
  }

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
    for (cell = 0, e = 0; e < numElems; ++e) {
      GmshElement *elem = mesh->elements + e;

      /* Create cell sets */
      if (elem->dim == dim && dim > 0) {
        if (elem->numTags > 0) {
          ierr = DMSetLabelValue(*dm, "Cell Sets", cell, elem->tags[0]);CHKERRQ(ierr);
        }
        cell++;
      }

      /* Create face sets */
      if (interpolate && elem->dim == dim-1) {
        PetscInt        joinSize;
        const PetscInt *join = NULL;
        /* Find the relevant facet with vertex joins */
        for (v = 0; v < elem->numVerts; ++v) {
          const PetscInt nn = elem->nodes[v];
          const PetscInt vv = mesh->vertexMap[nn];
          cone[v] = vStart + vv;
        }
        ierr = DMPlexGetFullJoin(*dm, elem->numVerts, cone, &joinSize, &join);CHKERRQ(ierr);
        if (joinSize != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Could not determine Plex facet for Gmsh element %D (Plex cell %D)", elem->id, e);
        ierr = DMSetLabelValue(*dm, "Face Sets", join[0], elem->tags[0]);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(*dm, elem->numVerts, cone, &joinSize, &join);CHKERRQ(ierr);
      }

      /* Create vertex sets */
      if (elem->dim == 0) {
        if (elem->numTags > 0) {
          const PetscInt nn = elem->nodes[0];
          const PetscInt vv = mesh->vertexMap[nn];
          ierr = DMSetLabelValue(*dm, "Vertex Sets", vStart + vv, elem->tags[0]);CHKERRQ(ierr);
        }
      }
    }
  }

  if (periodic) {
    ierr = PetscBTCreate(numVerts, &periodicVerts);CHKERRQ(ierr);
    for (n = 0; n < numNodes; ++n) {
      if (mesh->vertexMap[n] >= 0) {
        if (PetscUnlikely(mesh->periodMap[n] != n)) {
          PetscInt m = mesh->periodMap[n];
          ierr= PetscBTSet(periodicVerts, mesh->vertexMap[n]);CHKERRQ(ierr);
          ierr= PetscBTSet(periodicVerts, mesh->vertexMap[m]);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscBTCreate(numCells, &periodicCells);CHKERRQ(ierr);
    for (cell = 0; cell < numCells; ++cell) {
      GmshElement *elem = mesh->elements + cell;
      for (v = 0; v < elem->numVerts; ++v) {
        PetscInt nn = elem->nodes[v];
        PetscInt vv = mesh->vertexMap[nn];
        if (PetscUnlikely(PetscBTLookup(periodicVerts, vv))) {
          ierr = PetscBTSet(periodicCells, cell);CHKERRQ(ierr); break;
        }
      }
    }
  }

  /* Create coordinates */
  if (coordDim < 0) coordDim = dim;
  ierr = DMSetCoordinateDim(*dm, coordDim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, coordDim);CHKERRQ(ierr);
  if (periodic) { /* we need to localize coordinates on cells */
    ierr = PetscSectionSetChart(coordSection, 0, numCells+numVerts);CHKERRQ(ierr);
  } else {
    ierr = PetscSectionSetChart(coordSection, numCells, numCells+numVerts);CHKERRQ(ierr);
  }
  if (periodic) {
    for (cell = 0; cell < numCells; ++cell) {
      if (PetscUnlikely(PetscBTLookup(periodicCells, cell))) {
        GmshElement *elem = mesh->elements + cell;
        PetscInt dof = elem->numVerts * coordDim;
        ierr = PetscSectionSetDof(coordSection, cell, dof);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(coordSection, cell, 0, dof);CHKERRQ(ierr);
      }
    }
  }
  for (v = numCells; v < numCells+numVerts; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, coordDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, coordDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, coordDim);CHKERRQ(ierr);
  ierr = VecSetType(coordinates, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  coordsIn = mesh ? mesh->nodelist->xyz : NULL;
  if (periodic) {
    for (cell = 0; cell < numCells; ++cell) {
      if (PetscUnlikely(PetscBTLookup(periodicCells, cell))) {
        GmshElement *elem = mesh->elements + cell;
        PetscInt off, node;
        for (v = 0; v < elem->numVerts; ++v)
          cone[v] = elem->nodes[v];
        ierr = DMPlexReorderCell(*dm, cell, cone);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, cell, &off);CHKERRQ(ierr);
        for (v = 0; v < elem->numVerts; ++v)
          for (node = cone[v], d = 0; d < coordDim; ++d)
            coords[off++] = (PetscReal) coordsIn[node*3+d];
      }
    }
  }
  ierr = PetscMalloc1(numVerts, &vertexMapInv);CHKERRQ(ierr);
  for (n = 0; n < numNodes; n++)
    if (mesh->vertexMap[n] >= 0)
      vertexMapInv[mesh->vertexMap[n]] = n;
  for (v = 0; v < numVerts; ++v) {
    PetscInt off, node = vertexMapInv[v];
    ierr = PetscSectionGetOffset(coordSection, numCells + v, &off);CHKERRQ(ierr);
    for (d = 0; d < coordDim; ++d)
      coords[off+d] = (PetscReal) coordsIn[node*3+d];
  }
  ierr = PetscFree(vertexMapInv);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = DMSetPeriodicity(*dm, periodic, NULL, NULL, NULL);CHKERRQ(ierr);

  ierr = GmshMeshDestroy(&mesh);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&periodicVerts);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&periodicCells);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(DMPLEX_CreateGmsh,*dm,NULL,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
