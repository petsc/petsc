#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>

#include <../src/dm/impls/plex/gmshlex.h>

#define GMSH_LEXORDER_ITEM(T, p)                                   \
static int *GmshLexOrder_##T##_##p(void)                           \
{                                                                  \
  static int Gmsh_LexOrder_##T##_##p[GmshNumNodes_##T(p)] = {-1};  \
  int *lex = Gmsh_LexOrder_##T##_##p;                              \
  if (lex[0] == -1) (void)GmshLexOrder_##T(p, lex, 0);             \
  return lex;                                                      \
}

#define GMSH_LEXORDER_LIST(T) \
GMSH_LEXORDER_ITEM(T,  1)     \
GMSH_LEXORDER_ITEM(T,  2)     \
GMSH_LEXORDER_ITEM(T,  3)     \
GMSH_LEXORDER_ITEM(T,  4)     \
GMSH_LEXORDER_ITEM(T,  5)     \
GMSH_LEXORDER_ITEM(T,  6)     \
GMSH_LEXORDER_ITEM(T,  7)     \
GMSH_LEXORDER_ITEM(T,  8)     \
GMSH_LEXORDER_ITEM(T,  9)     \
GMSH_LEXORDER_ITEM(T, 10)

GMSH_LEXORDER_ITEM(VTX, 0)
GMSH_LEXORDER_LIST(SEG)
GMSH_LEXORDER_LIST(TRI)
GMSH_LEXORDER_LIST(QUA)
GMSH_LEXORDER_LIST(TET)
GMSH_LEXORDER_LIST(HEX)
GMSH_LEXORDER_LIST(PRI)
GMSH_LEXORDER_LIST(PYR)

typedef enum {
  GMSH_VTX = 0,
  GMSH_SEG = 1,
  GMSH_TRI = 2,
  GMSH_QUA = 3,
  GMSH_TET = 4,
  GMSH_HEX = 5,
  GMSH_PRI = 6,
  GMSH_PYR = 7,
  GMSH_NUM_POLYTOPES = 8
} GmshPolytopeType;

typedef struct {
  int   cellType;
  int   polytope;
  int   dim;
  int   order;
  int   numVerts;
  int   numNodes;
  int* (*lexorder)(void);
} GmshCellInfo;

#define GmshCellEntry(cellType, polytope, dim, order) \
  {cellType, GMSH_##polytope, dim, order, \
   GmshNumNodes_##polytope(1), \
   GmshNumNodes_##polytope(order), \
   GmshLexOrder_##polytope##_##order}

static const GmshCellInfo GmshCellTable[] = {
  GmshCellEntry( 15, VTX, 0,  0),

  GmshCellEntry(  1, SEG, 1,  1),
  GmshCellEntry(  8, SEG, 1,  2),
  GmshCellEntry( 26, SEG, 1,  3),
  GmshCellEntry( 27, SEG, 1,  4),
  GmshCellEntry( 28, SEG, 1,  5),
  GmshCellEntry( 62, SEG, 1,  6),
  GmshCellEntry( 63, SEG, 1,  7),
  GmshCellEntry( 64, SEG, 1,  8),
  GmshCellEntry( 65, SEG, 1,  9),
  GmshCellEntry( 66, SEG, 1, 10),

  GmshCellEntry(  2, TRI, 2,  1),
  GmshCellEntry(  9, TRI, 2,  2),
  GmshCellEntry( 21, TRI, 2,  3),
  GmshCellEntry( 23, TRI, 2,  4),
  GmshCellEntry( 25, TRI, 2,  5),
  GmshCellEntry( 42, TRI, 2,  6),
  GmshCellEntry( 43, TRI, 2,  7),
  GmshCellEntry( 44, TRI, 2,  8),
  GmshCellEntry( 45, TRI, 2,  9),
  GmshCellEntry( 46, TRI, 2, 10),

  GmshCellEntry(  3, QUA, 2,  1),
  GmshCellEntry( 10, QUA, 2,  2),
  GmshCellEntry( 36, QUA, 2,  3),
  GmshCellEntry( 37, QUA, 2,  4),
  GmshCellEntry( 38, QUA, 2,  5),
  GmshCellEntry( 47, QUA, 2,  6),
  GmshCellEntry( 48, QUA, 2,  7),
  GmshCellEntry( 49, QUA, 2,  8),
  GmshCellEntry( 50, QUA, 2,  9),
  GmshCellEntry( 51, QUA, 2, 10),

  GmshCellEntry(  4, TET, 3,  1),
  GmshCellEntry( 11, TET, 3,  2),
  GmshCellEntry( 29, TET, 3,  3),
  GmshCellEntry( 30, TET, 3,  4),
  GmshCellEntry( 31, TET, 3,  5),
  GmshCellEntry( 71, TET, 3,  6),
  GmshCellEntry( 72, TET, 3,  7),
  GmshCellEntry( 73, TET, 3,  8),
  GmshCellEntry( 74, TET, 3,  9),
  GmshCellEntry( 75, TET, 3, 10),

  GmshCellEntry(  5, HEX, 3,  1),
  GmshCellEntry( 12, HEX, 3,  2),
  GmshCellEntry( 92, HEX, 3,  3),
  GmshCellEntry( 93, HEX, 3,  4),
  GmshCellEntry( 94, HEX, 3,  5),
  GmshCellEntry( 95, HEX, 3,  6),
  GmshCellEntry( 96, HEX, 3,  7),
  GmshCellEntry( 97, HEX, 3,  8),
  GmshCellEntry( 98, HEX, 3,  9),
  GmshCellEntry( -1, HEX, 3, 10),

  GmshCellEntry(  6, PRI, 3,  1),
  GmshCellEntry( 13, PRI, 3,  2),
  GmshCellEntry( 90, PRI, 3,  3),
  GmshCellEntry( 91, PRI, 3,  4),
  GmshCellEntry(106, PRI, 3,  5),
  GmshCellEntry(107, PRI, 3,  6),
  GmshCellEntry(108, PRI, 3,  7),
  GmshCellEntry(109, PRI, 3,  8),
  GmshCellEntry(110, PRI, 3,  9),
  GmshCellEntry( -1, PRI, 3, 10),

  GmshCellEntry(  7, PYR, 3,  1),
  GmshCellEntry( 14, PYR, 3,  2),
  GmshCellEntry(118, PYR, 3,  3),
  GmshCellEntry(119, PYR, 3,  4),
  GmshCellEntry(120, PYR, 3,  5),
  GmshCellEntry(121, PYR, 3,  6),
  GmshCellEntry(122, PYR, 3,  7),
  GmshCellEntry(123, PYR, 3,  8),
  GmshCellEntry(124, PYR, 3,  9),
  GmshCellEntry( -1, PYR, 3, 10)

#if 0
  {20, GMSH_TRI, 2, 3, 3,  9, NULL},
  {16, GMSH_QUA, 2, 2, 4,  8, NULL},
  {17, GMSH_HEX, 3, 2, 8, 20, NULL},
  {18, GMSH_PRI, 3, 2, 6, 15, NULL},
  {19, GMSH_PYR, 3, 2, 5, 13, NULL},
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
    GmshCellMap[i].polytope = -1;
  }
  n = sizeof(GmshCellTable)/sizeof(GmshCellTable[0]);
  for (i = 0; i < n; ++i) {
    if (GmshCellTable[i].cellType <= 0) continue;
    GmshCellMap[GmshCellTable[i].cellType] = GmshCellTable[i];
  }
  PetscFunctionReturn(0);
}

#define GmshCellTypeCheck(ct) 0; do { \
    const int _ct_ = (int)ct; \
    if (_ct_ < 0 || _ct_ >= (int)(sizeof(GmshCellMap)/sizeof(GmshCellMap[0]))) \
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid Gmsh element type %d", _ct_); \
    if (GmshCellMap[_ct_].cellType != _ct_) \
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported Gmsh element type %d", _ct_); \
    if (GmshCellMap[_ct_].polytope == -1) \
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported Gmsh element type %d", _ct_); \
  } while (0)

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
  PetscInt       order;
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
  PetscInt       numRegions;
  PetscInt      *regionTags;
  char         **regionNames;
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
  PetscInt       r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*mesh) PetscFunctionReturn(0);
  ierr = GmshEntitiesDestroy(&(*mesh)->entities);CHKERRQ(ierr);
  ierr = GmshNodesDestroy(&(*mesh)->nodelist);CHKERRQ(ierr);
  ierr = GmshElementsDestroy(&(*mesh)->elements);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->periodMap);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->vertexMap);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&(*mesh)->segbuf);CHKERRQ(ierr);
  for (r = 0; r < (*mesh)->numRegions; ++r) {ierr = PetscFree((*mesh)->regionNames[r]);CHKERRQ(ierr);}
  ierr = PetscFree2((*mesh)->regionTags, (*mesh)->regionNames);CHKERRQ(ierr);
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
    int    ibuf[3], correspondingDim = -1, correspondingTag = -1, primaryTag = -1, correspondingNode, primaryNode;
    long   j, nNodes;
    double affine[16];

    if (fileFormat == 22 || !binary) {
      ierr = PetscViewerRead(viewer, line, 3, NULL, PETSC_STRING);CHKERRQ(ierr);
      snum = sscanf(line, "%d %d %d", &correspondingDim, &correspondingTag, &primaryTag);
      if (snum != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    } else {
      ierr = PetscViewerRead(viewer, ibuf, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
      if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, 3);CHKERRQ(ierr);}
      correspondingDim = ibuf[0]; correspondingTag = ibuf[1]; primaryTag = ibuf[2];
    }
    (void)correspondingDim; (void)correspondingTag; (void)primaryTag; /* unused */

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
        snum = sscanf(line, "%d %d", &correspondingNode, &primaryNode);
        if (snum != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
      } else {
        ierr = PetscViewerRead(viewer, ibuf, 2, NULL, PETSC_ENUM);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, 2);CHKERRQ(ierr);}
        correspondingNode = ibuf[0]; primaryNode = ibuf[1];
      }
      correspondingNode  = (int) nodeMap[correspondingNode];
      primaryNode = (int) nodeMap[primaryNode];
      periodicMap[correspondingNode] = primaryNode;
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
  entityDim(int) entityTag(int) entityTagPrimary(int)
  numAffine(size_t) value(double) ...
  numCorrespondingNodes(size_t)
    nodeTag(size_t) nodeTagPrimary(size_t)
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
      PetscInt correspondingNode = nodeMap[nodeTags[node*2+0]];
      PetscInt primaryNode = nodeMap[nodeTags[node*2+1]];
      periodicMap[correspondingNode] = primaryNode;
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
  fileFormat = (int)roundf(version*10);
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
static PetscErrorCode GmshReadPhysicalNames(GmshFile *gmsh, GmshMesh *mesh)
{
  char           line[PETSC_MAX_PATH_LEN], name[128+2], *p, *q;
  int            snum, region, dim, tag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GmshReadString(gmsh, line, 1);CHKERRQ(ierr);
  snum = sscanf(line, "%d", &region);
  mesh->numRegions = region;
  if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
  ierr = PetscMalloc2(mesh->numRegions, &mesh->regionTags, mesh->numRegions, &mesh->regionNames);CHKERRQ(ierr);
  for (region = 0; region < mesh->numRegions; ++region) {
    ierr = GmshReadString(gmsh, line, 2);CHKERRQ(ierr);
    snum = sscanf(line, "%d %d", &dim, &tag);
    if (snum != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    ierr = GmshReadString(gmsh, line, -(PetscInt)sizeof(line));CHKERRQ(ierr);
    ierr = PetscStrchr(line, '"', &p);CHKERRQ(ierr);
    if (!p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    ierr = PetscStrrchr(line, '"', &q);CHKERRQ(ierr);
    if (q == p) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    ierr = PetscStrncpy(name, p+1, (size_t)(q-p-1));CHKERRQ(ierr);
    mesh->regionTags[region] = tag;
    ierr = PetscStrallocpy(name, &mesh->regionNames[region]);CHKERRQ(ierr);
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
    PetscInt    keymap[GMSH_NUM_POLYTOPES], nk = 0;
    PetscInt    offset[GMSH_NUM_POLYTOPES+1], e, k;

    for (k = 0; k < GMSH_NUM_POLYTOPES; ++k) keymap[k] = PETSC_MIN_INT;
    ierr = PetscMemzero(offset,sizeof(offset));CHKERRQ(ierr);

    keymap[GMSH_TET] = nk++;
    keymap[GMSH_HEX] = nk++;
    keymap[GMSH_PRI] = nk++;
    keymap[GMSH_PYR] = nk++;
    keymap[GMSH_TRI] = nk++;
    keymap[GMSH_QUA] = nk++;
    keymap[GMSH_SEG] = nk++;
    keymap[GMSH_VTX] = nk++;

    ierr = GmshElementsCreate(mesh->numElems, &mesh->elements);CHKERRQ(ierr);
#define key(eid) keymap[GmshCellMap[elements[(eid)].cellType].polytope]
    for (e = 0; e < ne; ++e) offset[1+key(e)]++;
    for (k = 1; k < nk; ++k) offset[k] += offset[k-1];
    for (e = 0; e < ne; ++e) mesh->elements[offset[key(e)]++] = elements[e];
#undef key
    ierr = GmshElementsDestroy(&elements);CHKERRQ(ierr);
  }

  { /* Mesh dimension and order */
    GmshElement *elem = mesh->numElems ? mesh->elements : NULL;
    mesh->dim   = elem ? GmshCellMap[elem->cellType].dim   : 0;
    mesh->order = elem ? GmshCellMap[elem->cellType].order : 0;
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

  /* Find canonical primary nodes */
  for (n = 0; n < mesh->numNodes; ++n)
    while (mesh->periodMap[n] != mesh->periodMap[mesh->periodMap[n]])
      mesh->periodMap[n] = mesh->periodMap[mesh->periodMap[n]];

  /* Renumber vertices (filter out correspondings) */
  mesh->numVerts = 0;
  for (n = 0; n < mesh->numNodes; ++n)
    if (mesh->vertexMap[n] >= 0)   /* is vertex */
      if (mesh->periodMap[n] == n) /* is primary */
        mesh->vertexMap[n] = mesh->numVerts++;
  for (n = 0; n < mesh->numNodes; ++n)
    if (mesh->vertexMap[n] >= 0)   /* is vertex */
      if (mesh->periodMap[n] != n) /* is corresponding  */
        mesh->vertexMap[n] = mesh->vertexMap[mesh->periodMap[n]];
  PetscFunctionReturn(0);
}

#define DM_POLYTOPE_VERTEX  DM_POLYTOPE_POINT
#define DM_POLYTOPE_PYRAMID DM_POLYTOPE_UNKNOWN
static const DMPolytopeType DMPolytopeMap[] = {
  /* GMSH_VTX */ DM_POLYTOPE_VERTEX,
  /* GMSH_SEG */ DM_POLYTOPE_SEGMENT,
  /* GMSH_TRI */ DM_POLYTOPE_TRIANGLE,
  /* GMSH_QUA */ DM_POLYTOPE_QUADRILATERAL,
  /* GMSH_TET */ DM_POLYTOPE_TETRAHEDRON,
  /* GMSH_HEX */ DM_POLYTOPE_HEXAHEDRON,
  /* GMSH_PRI */ DM_POLYTOPE_TRI_PRISM,
  /* GMSH_PYR */ DM_POLYTOPE_PYRAMID,
  DM_POLYTOPE_UNKNOWN
};

PETSC_STATIC_INLINE DMPolytopeType DMPolytopeTypeFromGmsh(PetscInt cellType)
{
  return DMPolytopeMap[GmshCellMap[cellType].polytope];
}

static PetscErrorCode GmshCreateFE(MPI_Comm comm, const char prefix[], PetscBool isSimplex, PetscBool continuity, PetscDTNodeType nodeType, PetscInt dim, PetscInt Nc, PetscInt k, PetscFE *fem)
{
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscQuadrature q, fq;
  PetscBool       isTensor = isSimplex ? PETSC_FALSE : PETSC_TRUE;
  PetscBool       endpoint = PETSC_TRUE;
  char            name[32];
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create space */
  ierr = PetscSpaceCreate(comm, &P);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, isTensor);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, k, PETSC_DETERMINE);CHKERRQ(ierr);
  if (prefix) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix);CHKERRQ(ierr);
    ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) P, NULL);CHKERRQ(ierr);
    ierr = PetscSpaceGetDegree(P, &k, NULL);CHKERRQ(ierr);
  }
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(comm, &Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, isTensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetContinuity(Q, continuity);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetNodeType(Q, nodeType, endpoint, 0);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, k);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  if (prefix) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, prefix);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, NULL);CHKERRQ(ierr);
  }
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create quadrature */
  if (isSimplex) {
    ierr = PetscDTStroudConicalQuadrature(dim,   1, k+1, -1, +1, &q);CHKERRQ(ierr);
    ierr = PetscDTStroudConicalQuadrature(dim-1, 1, k+1, -1, +1, &fq);CHKERRQ(ierr);
  } else {
    ierr = PetscDTGaussTensorQuadrature(dim,   1, k+1, -1, +1, &q);CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1, 1, k+1, -1, +1, &fq);CHKERRQ(ierr);
  }
  /* Create finite element */
  ierr = PetscFECreate(comm, fem);CHKERRQ(ierr);
  ierr = PetscFESetType(*fem, PETSCFEBASIC);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fem, Nc);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fem, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fem, Q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(*fem, q);CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(*fem, fq);CHKERRQ(ierr);
  if (prefix) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix);CHKERRQ(ierr);
    ierr = PetscFESetFromOptions(*fem);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) *fem, NULL);CHKERRQ(ierr);
  }
  ierr = PetscFESetUp(*fem);CHKERRQ(ierr);
  /* Cleanup */
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fq);CHKERRQ(ierr);
  /* Set finite element name */
  ierr = PetscSNPrintf(name, sizeof(name), "%s%D", isSimplex? "P" : "Q", k);CHKERRQ(ierr);
  ierr = PetscFESetName(*fem, name);CHKERRQ(ierr);
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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);

  /* Determine Gmsh file type (ASCII or binary) from file header */
  if (rank == 0) {
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
  ierr = MPI_Bcast(&fileType, 1, MPI_INT, 0, comm);CHKERRMPI(ierr);
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
  DM             cdm;
  PetscSection   coordSection;
  Vec            coordinates;
  DMLabel        cellSets = NULL, faceSets = NULL, vertSets = NULL, marker = NULL, *regionSets;
  PetscInt       dim = 0, coordDim = -1, order = 0;
  PetscInt       numNodes = 0, numElems = 0, numVerts = 0, numCells = 0;
  PetscInt       cell, cone[8], e, n, v, d;
  PetscBool      binary, usemarker = PETSC_FALSE, useregions = PETSC_FALSE;
  PetscBool      hybrid = interpolate, periodic = PETSC_TRUE;
  PetscBool      highOrder = PETSC_TRUE, highOrderSet, project = PETSC_FALSE;
  PetscBool      isSimplex = PETSC_FALSE, isHybrid = PETSC_FALSE, hasTetra = PETSC_FALSE;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPlex Gmsh options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_hybrid", "Generate hybrid cell bounds", "DMPlexCreateGmsh", hybrid, &hybrid, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_periodic","Read Gmsh periodic section", "DMPlexCreateGmsh", periodic, &periodic, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_highorder","Generate high-order coordinates", "DMPlexCreateGmsh", highOrder, &highOrder, &highOrderSet);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_project", "Project high-order coordinates to a different space", "DMPlexCreateGmsh", project, &project, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_use_marker", "Generate marker label", "DMPlexCreateGmsh", usemarker, &usemarker, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_gmsh_use_regions", "Generate labels with region names", "DMPlexCreateGmsh", useregions, &useregions, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_plex_gmsh_spacedim", "Embedding space dimension", "DMPlexCreateGmsh", coordDim, &coordDim, NULL, PETSC_DECIDE);CHKERRQ(ierr);
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

  if (rank == 0) {
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
      ierr = GmshReadPhysicalNames(gmsh, mesh);CHKERRQ(ierr);
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
    order     = mesh->order;
    numNodes  = mesh->numNodes;
    numElems  = mesh->numElems;
    numVerts  = mesh->numVerts;
    numCells  = mesh->numCells;

    {
      GmshElement *elemA = mesh->numCells > 0 ? mesh->elements : NULL;
      GmshElement *elemB = elemA ? elemA + mesh->numCells - 1  : NULL;
      int ptA = elemA ? GmshCellMap[elemA->cellType].polytope : -1;
      int ptB = elemB ? GmshCellMap[elemB->cellType].polytope : -1;
      isSimplex = (ptA == GMSH_QUA || ptA == GMSH_HEX) ? PETSC_FALSE : PETSC_TRUE;
      isHybrid  = (ptA == ptB) ? PETSC_FALSE : PETSC_TRUE;
      hasTetra  = (ptA == GMSH_TET) ? PETSC_TRUE : PETSC_FALSE;
    }
  }

  if (parentviewer) {
    ierr = PetscViewerRestoreSubViewer(parentviewer, PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
  }

  {
    int buf[6];

    buf[0] = (int)dim;
    buf[1] = (int)order;
    buf[2] = periodic;
    buf[3] = isSimplex;
    buf[4] = isHybrid;
    buf[5] = hasTetra;

    ierr = MPI_Bcast(buf, 6, MPI_INT, 0, comm);CHKERRMPI(ierr);

    dim       = buf[0];
    order     = buf[1];
    periodic  = buf[2] ? PETSC_TRUE : PETSC_FALSE;
    isSimplex = buf[3] ? PETSC_TRUE : PETSC_FALSE;
    isHybrid  = buf[4] ? PETSC_TRUE : PETSC_FALSE;
    hasTetra  = buf[5] ? PETSC_TRUE : PETSC_FALSE;
  }

  if (!highOrderSet) highOrder = (order > 1) ? PETSC_TRUE : PETSC_FALSE;
  if (highOrder && isHybrid) SETERRQ(comm, PETSC_ERR_SUP, "No support for discretization on hybrid meshes yet");

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
  if (rank == 0 && usemarker) {
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
          ierr = DMSetLabelValue_Fast(*dm, &marker, "marker", cone[p], 1);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreTransitiveClosure(*dm, f, PETSC_TRUE, &coneSize, &cone);CHKERRQ(ierr);
      }
    }
  }

  if (rank == 0) {
    const PetscInt Nr = useregions ? mesh->numRegions : 0;
    PetscInt       vStart, vEnd;

    ierr = PetscCalloc1(Nr, &regionSets);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    for (cell = 0, e = 0; e < numElems; ++e) {
      GmshElement *elem = mesh->elements + e;

      /* Create cell sets */
      if (elem->dim == dim && dim > 0) {
        if (elem->numTags > 0) {
          const PetscInt tag = elem->tags[0];
          PetscInt       r;

          ierr = DMSetLabelValue_Fast(*dm, &cellSets, "Cell Sets", cell, tag);CHKERRQ(ierr);
          for (r = 0; r < Nr; ++r) {
            if (mesh->regionTags[r] == tag) {ierr = DMSetLabelValue_Fast(*dm, &regionSets[r], mesh->regionNames[r], cell, tag);CHKERRQ(ierr);}
          }
        }
        cell++;
      }

      /* Create face sets */
      if (interpolate && elem->dim == dim-1) {
        PetscInt        joinSize;
        const PetscInt *join = NULL;
        const PetscInt  tag = elem->tags[0];
        PetscInt        r;

        /* Find the relevant facet with vertex joins */
        for (v = 0; v < elem->numVerts; ++v) {
          const PetscInt nn = elem->nodes[v];
          const PetscInt vv = mesh->vertexMap[nn];
          cone[v] = vStart + vv;
        }
        ierr = DMPlexGetFullJoin(*dm, elem->numVerts, cone, &joinSize, &join);CHKERRQ(ierr);
        if (joinSize != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Could not determine Plex facet for Gmsh element %D (Plex cell %D)", elem->id, e);
        ierr = DMSetLabelValue_Fast(*dm, &faceSets, "Face Sets", join[0], tag);CHKERRQ(ierr);
        for (r = 0; r < Nr; ++r) {
          if (mesh->regionTags[r] == tag) {ierr = DMSetLabelValue_Fast(*dm, &regionSets[r], mesh->regionNames[r], join[0], tag);CHKERRQ(ierr);}
        }
        ierr = DMPlexRestoreJoin(*dm, elem->numVerts, cone, &joinSize, &join);CHKERRQ(ierr);
      }

      /* Create vertex sets */
      if (elem->dim == 0) {
        if (elem->numTags > 0) {
          const PetscInt nn = elem->nodes[0];
          const PetscInt vv = mesh->vertexMap[nn];
          const PetscInt tag = elem->tags[0];
          PetscInt       r;

          ierr = DMSetLabelValue_Fast(*dm, &vertSets, "Vertex Sets", vStart + vv, tag);CHKERRQ(ierr);
          for (r = 0; r < Nr; ++r) {
            if (mesh->regionTags[r] == tag) {ierr = DMSetLabelValue_Fast(*dm, &regionSets[r], mesh->regionNames[r], vStart + vv, tag);CHKERRQ(ierr);}
          }
        }
      }
    }
    ierr = PetscFree(regionSets);CHKERRQ(ierr);
  }

  { /* Create Cell/Face/Vertex Sets labels at all processes */
    enum {n = 4};
    PetscBool flag[n];

    flag[0] = cellSets ? PETSC_TRUE : PETSC_FALSE;
    flag[1] = faceSets ? PETSC_TRUE : PETSC_FALSE;
    flag[2] = vertSets ? PETSC_TRUE : PETSC_FALSE;
    flag[3] = marker   ? PETSC_TRUE : PETSC_FALSE;
    ierr = MPI_Bcast(flag, n, MPIU_BOOL, 0, comm);CHKERRMPI(ierr);
    if (flag[0]) {ierr = DMCreateLabel(*dm, "Cell Sets");CHKERRQ(ierr);}
    if (flag[1]) {ierr = DMCreateLabel(*dm, "Face Sets");CHKERRQ(ierr);}
    if (flag[2]) {ierr = DMCreateLabel(*dm, "Vertex Sets");CHKERRQ(ierr);}
    if (flag[3]) {ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);}
  }

  if (periodic) {
    ierr = PetscBTCreate(numVerts, &periodicVerts);CHKERRQ(ierr);
    for (n = 0; n < numNodes; ++n) {
      if (mesh->vertexMap[n] >= 0) {
        if (PetscUnlikely(mesh->periodMap[n] != n)) {
          PetscInt m = mesh->periodMap[n];
          ierr = PetscBTSet(periodicVerts, mesh->vertexMap[n]);CHKERRQ(ierr);
          ierr = PetscBTSet(periodicVerts, mesh->vertexMap[m]);CHKERRQ(ierr);
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

  /* Setup coordinate DM */
  if (coordDim < 0) coordDim = dim;
  ierr = DMSetCoordinateDim(*dm, coordDim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*dm, &cdm);CHKERRQ(ierr);
  if (highOrder) {
    PetscFE         fe;
    PetscBool       continuity = periodic ? PETSC_FALSE : PETSC_TRUE;
    PetscDTNodeType nodeType   = PETSCDTNODES_EQUISPACED;

    if (isSimplex) continuity = PETSC_FALSE; /* XXX FIXME Requires DMPlexSetClosurePermutationLexicographic() */

    ierr = GmshCreateFE(comm, NULL, isSimplex, continuity, nodeType, dim, coordDim, order, &fe);CHKERRQ(ierr);
    ierr = PetscFEViewFromOptions(fe, NULL, "-dm_plex_gmsh_fe_view");CHKERRQ(ierr);
    ierr = DMSetField(cdm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    ierr = DMCreateDS(cdm);CHKERRQ(ierr);
  }

  /* Create coordinates */
  if (highOrder) {

    PetscInt     maxDof = GmshNumNodes_HEX(order)*coordDim;
    double       *coords = mesh ? mesh->nodelist->xyz : NULL;
    PetscSection section;
    PetscScalar  *cellCoords;

    ierr = DMSetLocalSection(cdm, NULL);CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
    ierr = PetscSectionClone(coordSection, &section);CHKERRQ(ierr);
    ierr = DMPlexSetClosurePermutationTensor(cdm, 0, section);CHKERRQ(ierr); /* XXX Implement DMPlexSetClosurePermutationLexicographic() */

    ierr = DMCreateLocalVector(cdm, &coordinates);CHKERRQ(ierr);
    ierr = PetscMalloc1(maxDof, &cellCoords);CHKERRQ(ierr);
    for (cell = 0; cell < numCells; ++cell) {
      GmshElement *elem = mesh->elements + cell;
      const int *lexorder = GmshCellMap[elem->cellType].lexorder();
      for (n = 0; n < elem->numNodes; ++n) {
        const PetscInt node = elem->nodes[lexorder[n]];
        for (d = 0; d < coordDim; ++d)
          cellCoords[n*coordDim+d] = (PetscReal) coords[node*3+d];
      }
      ierr = DMPlexVecSetClosure(cdm, section, coordinates, cell, cellCoords, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
    ierr = PetscFree(cellCoords);CHKERRQ(ierr);

  } else {

    PetscInt    *nodeMap;
    double      *coords = mesh ? mesh->nodelist->xyz : NULL;
    PetscScalar *pointCoords;

    ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
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

    ierr = DMCreateLocalVector(cdm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &pointCoords);CHKERRQ(ierr);
    if (periodic) {
      for (cell = 0; cell < numCells; ++cell) {
        if (PetscUnlikely(PetscBTLookup(periodicCells, cell))) {
          GmshElement *elem = mesh->elements + cell;
          PetscInt off, node;
          for (v = 0; v < elem->numVerts; ++v)
            cone[v] = elem->nodes[v];
          ierr = DMPlexReorderCell(cdm, cell, cone);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(coordSection, cell, &off);CHKERRQ(ierr);
          for (v = 0; v < elem->numVerts; ++v)
            for (node = cone[v], d = 0; d < coordDim; ++d)
              pointCoords[off++] = (PetscReal) coords[node*3+d];
        }
      }
    }
    ierr = PetscMalloc1(numVerts, &nodeMap);CHKERRQ(ierr);
    for (n = 0; n < numNodes; n++)
      if (mesh->vertexMap[n] >= 0)
        nodeMap[mesh->vertexMap[n]] = n;
    for (v = 0; v < numVerts; ++v) {
      PetscInt off, node = nodeMap[v];
      ierr = PetscSectionGetOffset(coordSection, numCells + v, &off);CHKERRQ(ierr);
      for (d = 0; d < coordDim; ++d)
        pointCoords[off+d] = (PetscReal) coords[node*3+d];
    }
    ierr = PetscFree(nodeMap);CHKERRQ(ierr);
    ierr = VecRestoreArray(coordinates, &pointCoords);CHKERRQ(ierr);

  }

  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, coordDim);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = DMSetPeriodicity(*dm, periodic, NULL, NULL, NULL);CHKERRQ(ierr);

  ierr = GmshMeshDestroy(&mesh);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&periodicVerts);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&periodicCells);CHKERRQ(ierr);

  if (highOrder && project)  {
    PetscFE         fe;
    const char      prefix[]   = "dm_plex_gmsh_project_";
    PetscBool       continuity = periodic ? PETSC_FALSE : PETSC_TRUE;
    PetscDTNodeType nodeType   = PETSCDTNODES_GAUSSJACOBI;

    if (isSimplex) continuity = PETSC_FALSE; /* XXX FIXME Requires DMPlexSetClosurePermutationLexicographic() */

    ierr = GmshCreateFE(comm, prefix, isSimplex, continuity, nodeType, dim, coordDim, order, &fe);CHKERRQ(ierr);
    ierr = PetscFEViewFromOptions(fe, NULL, "-dm_plex_gmsh_project_fe_view");CHKERRQ(ierr);
    ierr = DMProjectCoordinates(*dm, fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(DMPLEX_CreateGmsh,*dm,NULL,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
