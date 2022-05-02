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

static int *GmshLexOrder_QUA_2_Serendipity(void)
{
  static int Gmsh_LexOrder_QUA_2_Serendipity[9] = {-1};
  int *lex = Gmsh_LexOrder_QUA_2_Serendipity;
  if (lex[0] == -1) {
    /* Vertices */
    lex[0] = 0; lex[2] = 1; lex[8] = 2; lex[6] = 3;
    /* Edges */
    lex[1] = 4; lex[5] = 5; lex[7] = 6; lex[3] = 7;
    /* Cell */
    lex[4] = -8;
  }
  return lex;
}

static int *GmshLexOrder_HEX_2_Serendipity(void)
{
  static int Gmsh_LexOrder_HEX_2_Serendipity[27] = {-1};
  int *lex = Gmsh_LexOrder_HEX_2_Serendipity;
  if (lex[0] == -1) {
    /* Vertices */
    lex[ 0] =   0; lex[ 2] =   1; lex[ 8] =   2; lex[ 6] =   3;
    lex[18] =   4; lex[20] =   5; lex[26] =   6; lex[24] =   7;
    /* Edges */
    lex[ 1] =   8; lex[ 3] =   9; lex[ 9] =  10; lex[ 5] =  11;
    lex[11] =  12; lex[ 7] =  13; lex[17] =  14; lex[15] =  15;
    lex[19] =  16; lex[21] =  17; lex[23] =  18; lex[25] =  19;
    /* Faces */
    lex[ 4] = -20; lex[10] = -21; lex[12] = -22; lex[14] = -23;
    lex[16] = -24; lex[22] = -25;
    /* Cell */
    lex[13] = -26;
  }
  return lex;
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
  {16, GMSH_QUA, 2, 2, 4, 8, GmshLexOrder_QUA_2_Serendipity},
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
  {17, GMSH_HEX, 3, 2, 8, 20, GmshLexOrder_HEX_2_Serendipity},
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
  n = PETSC_STATIC_ARRAY_LENGTH(GmshCellMap);
  for (i = 0; i < n; ++i) {
    GmshCellMap[i].cellType = -1;
    GmshCellMap[i].polytope = -1;
  }
  n = PETSC_STATIC_ARRAY_LENGTH(GmshCellTable);
  for (i = 0; i < n; ++i) {
    if (GmshCellTable[i].cellType <= 0) continue;
    GmshCellMap[GmshCellTable[i].cellType] = GmshCellTable[i];
  }
  PetscFunctionReturn(0);
}

#define GmshCellTypeCheck(ct) PetscMacroReturnStandard(                                        \
    const int _ct_ = (int)ct;                                                                  \
    PetscCheck(_ct_ >= 0 && _ct_ < (int)PETSC_STATIC_ARRAY_LENGTH(GmshCellMap), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid Gmsh element type %d", _ct_); \
    PetscCheck(GmshCellMap[_ct_].cellType == _ct_, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported Gmsh element type %d", _ct_); \
    PetscCheck(GmshCellMap[_ct_].polytope != -1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported Gmsh element type %d", _ct_); \
  )

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

  PetscFunctionBegin;
  if (gmsh->wlen < size) {
    PetscCall(PetscFree(gmsh->wbuf));
    PetscCall(PetscMalloc(size, &gmsh->wbuf));
    gmsh->wlen = size;
  }
  *(void**)buf = size ? gmsh->wbuf : NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshBufferSizeGet(GmshFile *gmsh, size_t count, void *buf)
{
  size_t         dataSize = (size_t)gmsh->dataSize;
  size_t         size = count * dataSize;

  PetscFunctionBegin;
  if (gmsh->slen < size) {
    PetscCall(PetscFree(gmsh->sbuf));
    PetscCall(PetscMalloc(size, &gmsh->sbuf));
    gmsh->slen = size;
  }
  *(void**)buf = size ? gmsh->sbuf : NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshRead(GmshFile *gmsh, void *buf, PetscInt count, PetscDataType dtype)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerRead(gmsh->viewer, buf, count, NULL, dtype));
  if (gmsh->byteSwap) PetscCall(PetscByteSwap(buf, dtype, count));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadString(GmshFile *gmsh, char *buf, PetscInt count)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerRead(gmsh->viewer, buf, count, NULL, PETSC_STRING));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshMatch(PETSC_UNUSED GmshFile *gmsh, const char Section[], char line[PETSC_MAX_PATH_LEN], PetscBool *match)
{
  PetscFunctionBegin;
  PetscCall(PetscStrcmp(line, Section, match));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshExpect(GmshFile *gmsh, const char Section[], char line[PETSC_MAX_PATH_LEN])
{
  PetscBool      match;

  PetscFunctionBegin;
  PetscCall(GmshMatch(gmsh, Section, line, &match));
  PetscCheck(match,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file, expecting %s",Section);
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadSection(GmshFile *gmsh, char line[PETSC_MAX_PATH_LEN])
{
  PetscBool      match;

  PetscFunctionBegin;
  while (PETSC_TRUE) {
    PetscCall(GmshReadString(gmsh, line, 1));
    PetscCall(GmshMatch(gmsh, "$Comments", line, &match));
    if (!match) break;
    while (PETSC_TRUE) {
      PetscCall(GmshReadString(gmsh, line, 1));
      PetscCall(GmshMatch(gmsh, "$EndComments", line, &match));
      if (match) break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadEndSection(GmshFile *gmsh, const char EndSection[], char line[PETSC_MAX_PATH_LEN])
{
  PetscFunctionBegin;
  PetscCall(GmshReadString(gmsh, line, 1));
  PetscCall(GmshExpect(gmsh, EndSection, line));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadSize(GmshFile *gmsh, PetscInt *buf, PetscInt count)
{
  PetscInt       i;
  size_t         dataSize = (size_t)gmsh->dataSize;

  PetscFunctionBegin;
  if (dataSize == sizeof(PetscInt)) {
    PetscCall(GmshRead(gmsh, buf, count, PETSC_INT));
  } else  if (dataSize == sizeof(int)) {
    int *ibuf = NULL;
    PetscCall(GmshBufferSizeGet(gmsh, count, &ibuf));
    PetscCall(GmshRead(gmsh, ibuf, count, PETSC_ENUM));
    for (i = 0; i < count; ++i) buf[i] = (PetscInt)ibuf[i];
  } else  if (dataSize == sizeof(long)) {
    long *ibuf = NULL;
    PetscCall(GmshBufferSizeGet(gmsh, count, &ibuf));
    PetscCall(GmshRead(gmsh, ibuf, count, PETSC_LONG));
    for (i = 0; i < count; ++i) buf[i] = (PetscInt)ibuf[i];
  } else if (dataSize == sizeof(PetscInt64)) {
    PetscInt64 *ibuf = NULL;
    PetscCall(GmshBufferSizeGet(gmsh, count, &ibuf));
    PetscCall(GmshRead(gmsh, ibuf, count, PETSC_INT64));
    for (i = 0; i < count; ++i) buf[i] = (PetscInt)ibuf[i];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadInt(GmshFile *gmsh, int *buf, PetscInt count)
{
  PetscFunctionBegin;
  PetscCall(GmshRead(gmsh, buf, count, PETSC_ENUM));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadDouble(GmshFile *gmsh, double *buf, PetscInt count)
{
  PetscFunctionBegin;
  PetscCall(GmshRead(gmsh, buf, count, PETSC_DOUBLE));
  PetscFunctionReturn(0);
}

#define GMSH_MAX_TAGS 4

typedef struct {
  PetscInt id;      /* Entity ID */
  PetscInt dim;     /* Dimension */
  double   bbox[6]; /* Bounding box */
  PetscInt numTags;             /* Size of tag array */
  int      tags[GMSH_MAX_TAGS]; /* Tag array */
} GmshEntity;

typedef struct {
  GmshEntity *entity[4];
  PetscHMapI  entityMap[4];
} GmshEntities;

static PetscErrorCode GmshEntitiesCreate(PetscInt count[4], GmshEntities **entities)
{
  PetscInt       dim;

  PetscFunctionBegin;
  PetscCall(PetscNew(entities));
  for (dim = 0; dim < 4; ++dim) {
    PetscCall(PetscCalloc1(count[dim], &(*entities)->entity[dim]));
    PetscCall(PetscHMapICreate(&(*entities)->entityMap[dim]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshEntitiesDestroy(GmshEntities **entities)
{
  PetscInt       dim;

  PetscFunctionBegin;
  if (!*entities) PetscFunctionReturn(0);
  for (dim = 0; dim < 4; ++dim) {
    PetscCall(PetscFree((*entities)->entity[dim]));
    PetscCall(PetscHMapIDestroy(&(*entities)->entityMap[dim]));
  }
  PetscCall(PetscFree((*entities)));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshEntitiesAdd(GmshEntities *entities, PetscInt index, PetscInt dim, PetscInt eid, GmshEntity** entity)
{
  PetscFunctionBegin;
  PetscCall(PetscHMapISet(entities->entityMap[dim], eid, index));
  entities->entity[dim][index].dim = dim;
  entities->entity[dim][index].id  = eid;
  if (entity) *entity = &entities->entity[dim][index];
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshEntitiesGet(GmshEntities *entities, PetscInt dim, PetscInt eid, GmshEntity** entity)
{
  PetscInt       index;

  PetscFunctionBegin;
  PetscCall(PetscHMapIGet(entities->entityMap[dim], eid, &index));
  *entity = &entities->entity[dim][index];
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt *id;  /* Node IDs */
  double   *xyz; /* Coordinates */
  PetscInt *tag; /* Physical tag */
} GmshNodes;

static PetscErrorCode GmshNodesCreate(PetscInt count, GmshNodes **nodes)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(nodes));
  PetscCall(PetscMalloc1(count*1, &(*nodes)->id));
  PetscCall(PetscMalloc1(count*3, &(*nodes)->xyz));
  PetscCall(PetscMalloc1(count*GMSH_MAX_TAGS, &(*nodes)->tag));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshNodesDestroy(GmshNodes **nodes)
{
  PetscFunctionBegin;
  if (!*nodes) PetscFunctionReturn(0);
  PetscCall(PetscFree((*nodes)->id));
  PetscCall(PetscFree((*nodes)->xyz));
  PetscCall(PetscFree((*nodes)->tag));
  PetscCall(PetscFree((*nodes)));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt id;       /* Element ID */
  PetscInt dim;      /* Dimension */
  PetscInt cellType; /* Cell type */
  PetscInt numVerts; /* Size of vertex array */
  PetscInt numNodes; /* Size of node array */
  PetscInt *nodes;   /* Vertex/Node array */
  PetscInt numTags;             /* Size of physical tag array */
  int      tags[GMSH_MAX_TAGS]; /* Physical tag array */
} GmshElement;

static PetscErrorCode GmshElementsCreate(PetscInt count, GmshElement **elements)
{
  PetscFunctionBegin;
  PetscCall(PetscCalloc1(count, elements));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshElementsDestroy(GmshElement **elements)
{
  PetscFunctionBegin;
  if (!*elements) PetscFunctionReturn(0);
  PetscCall(PetscFree(*elements));
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
  PetscFunctionBegin;
  PetscCall(PetscNew(mesh));
  PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 0, &(*mesh)->segbuf));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshMeshDestroy(GmshMesh **mesh)
{
  PetscInt       r;

  PetscFunctionBegin;
  if (!*mesh) PetscFunctionReturn(0);
  PetscCall(GmshEntitiesDestroy(&(*mesh)->entities));
  PetscCall(GmshNodesDestroy(&(*mesh)->nodelist));
  PetscCall(GmshElementsDestroy(&(*mesh)->elements));
  PetscCall(PetscFree((*mesh)->periodMap));
  PetscCall(PetscFree((*mesh)->vertexMap));
  PetscCall(PetscSegBufferDestroy(&(*mesh)->segbuf));
  for (r = 0; r < (*mesh)->numRegions; ++r) PetscCall(PetscFree((*mesh)->regionNames[r]));
  PetscCall(PetscFree2((*mesh)->regionTags, (*mesh)->regionNames));
  PetscCall(PetscFree((*mesh)));
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadNodes_v22(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscViewer    viewer = gmsh->viewer;
  PetscBool      byteSwap = gmsh->byteSwap;
  char           line[PETSC_MAX_PATH_LEN];
  int            n, t, num, nid, snum;
  GmshNodes      *nodes;

  PetscFunctionBegin;
  PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
  snum = sscanf(line, "%d", &num);
  PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
  PetscCall(GmshNodesCreate(num, &nodes));
  mesh->numNodes = num;
  mesh->nodelist = nodes;
  for (n = 0; n < num; ++n) {
    double *xyz = nodes->xyz + n*3;
    PetscCall(PetscViewerRead(viewer, &nid, 1, NULL, PETSC_ENUM));
    PetscCall(PetscViewerRead(viewer, xyz, 3, NULL, PETSC_DOUBLE));
    if (byteSwap) PetscCall(PetscByteSwap(&nid, PETSC_ENUM, 1));
    if (byteSwap) PetscCall(PetscByteSwap(xyz, PETSC_DOUBLE, 3));
    nodes->id[n] = nid;
    for (t = 0; t < GMSH_MAX_TAGS; ++t) nodes->tag[n*GMSH_MAX_TAGS+t] = -1;
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

  PetscFunctionBegin;
  PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
  snum = sscanf(line, "%d", &num);
  PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
  PetscCall(GmshElementsCreate(num, &elements));
  mesh->numElems = num;
  mesh->elements = elements;
  for (c = 0; c < num;) {
    PetscCall(PetscViewerRead(viewer, ibuf, 3, NULL, PETSC_ENUM));
    if (byteSwap) PetscCall(PetscByteSwap(ibuf, PETSC_ENUM, 3));

    cellType = binary ? ibuf[0] : ibuf[1];
    numElem  = binary ? ibuf[1] : 1;
    numTags  = ibuf[2];

    PetscCall(GmshCellTypeCheck(cellType));
    numVerts = GmshCellMap[cellType].numVerts;
    numNodes = GmshCellMap[cellType].numNodes;

    for (i = 0; i < numElem; ++i, ++c) {
      GmshElement *element = elements + c;
      const int off = binary ? 0 : 1, nint = 1 + numTags + numNodes - off;
      const int *id = ibuf, *nodes = ibuf + 1 + numTags, *tags = ibuf + 1;
      PetscCall(PetscViewerRead(viewer, ibuf+off, nint, NULL, PETSC_ENUM));
      if (byteSwap) PetscCall(PetscByteSwap(ibuf+off, PETSC_ENUM, nint));
      element->id  = id[0];
      element->dim = GmshCellMap[cellType].dim;
      element->cellType = cellType;
      element->numVerts = numVerts;
      element->numNodes = numNodes;
      element->numTags  = PetscMin(numTags, GMSH_MAX_TAGS);
      PetscCall(PetscSegBufferGet(mesh->segbuf, (size_t)element->numNodes, &element->nodes));
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

  PetscFunctionBegin;
  PetscCall(PetscViewerRead(viewer, lbuf, 4, NULL, PETSC_LONG));
  if (byteSwap) PetscCall(PetscByteSwap(lbuf, PETSC_LONG, 4));
  for (i = 0; i < 4; ++i) count[i] = lbuf[i];
  PetscCall(GmshEntitiesCreate(count, &mesh->entities));
  for (dim = 0; dim < 4; ++dim) {
    for (index = 0; index < count[dim]; ++index) {
      PetscCall(PetscViewerRead(viewer, &eid, 1, NULL, PETSC_ENUM));
      if (byteSwap) PetscCall(PetscByteSwap(&eid, PETSC_ENUM, 1));
      PetscCall(GmshEntitiesAdd(mesh->entities, (PetscInt)index, dim, eid, &entity));
      PetscCall(PetscViewerRead(viewer, entity->bbox, 6, NULL, PETSC_DOUBLE));
      if (byteSwap) PetscCall(PetscByteSwap(entity->bbox, PETSC_DOUBLE, 6));
      PetscCall(PetscViewerRead(viewer, &num, 1, NULL, PETSC_LONG));
      if (byteSwap) PetscCall(PetscByteSwap(&num, PETSC_LONG, 1));
      PetscCall(GmshBufferGet(gmsh, num, sizeof(int), &ibuf));
      PetscCall(PetscViewerRead(viewer, ibuf, num, NULL, PETSC_ENUM));
      if (byteSwap) PetscCall(PetscByteSwap(ibuf, PETSC_ENUM, num));
      entity->numTags = numTags = (int) PetscMin(num, GMSH_MAX_TAGS);
      for (t = 0; t < numTags; ++t) entity->tags[t] = ibuf[t];
      if (dim == 0) continue;
      PetscCall(PetscViewerRead(viewer, &num, 1, NULL, PETSC_LONG));
      if (byteSwap) PetscCall(PetscByteSwap(&num, PETSC_LONG, 1));
      PetscCall(GmshBufferGet(gmsh, num, sizeof(int), &ibuf));
      PetscCall(PetscViewerRead(viewer, ibuf, num, NULL, PETSC_ENUM));
      if (byteSwap) PetscCall(PetscByteSwap(ibuf, PETSC_ENUM, num));
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
  long           block, node, n, t, numEntityBlocks, numTotalNodes, numNodes;
  int            info[3], nid;
  GmshNodes      *nodes;

  PetscFunctionBegin;
  PetscCall(PetscViewerRead(viewer, &numEntityBlocks, 1, NULL, PETSC_LONG));
  if (byteSwap) PetscCall(PetscByteSwap(&numEntityBlocks, PETSC_LONG, 1));
  PetscCall(PetscViewerRead(viewer, &numTotalNodes, 1, NULL, PETSC_LONG));
  if (byteSwap) PetscCall(PetscByteSwap(&numTotalNodes, PETSC_LONG, 1));
  PetscCall(GmshNodesCreate(numTotalNodes, &nodes));
  mesh->numNodes = numTotalNodes;
  mesh->nodelist = nodes;
  for (n = 0, block = 0; block < numEntityBlocks; ++block) {
    PetscCall(PetscViewerRead(viewer, info, 3, NULL, PETSC_ENUM));
    PetscCall(PetscViewerRead(viewer, &numNodes, 1, NULL, PETSC_LONG));
    if (byteSwap) PetscCall(PetscByteSwap(&numNodes, PETSC_LONG, 1));
    if (gmsh->binary) {
      size_t nbytes = sizeof(int) + 3*sizeof(double);
      char   *cbuf = NULL; /* dummy value to prevent warning from compiler about possible unitilized value */
      PetscCall(GmshBufferGet(gmsh, numNodes, nbytes, &cbuf));
      PetscCall(PetscViewerRead(viewer, cbuf, numNodes*nbytes, NULL, PETSC_CHAR));
      for (node = 0; node < numNodes; ++node, ++n) {
        char   *cnid = cbuf + node*nbytes, *cxyz = cnid + sizeof(int);
        double *xyz = nodes->xyz + n*3;
        if (!PetscBinaryBigEndian()) PetscCall(PetscByteSwap(cnid, PETSC_ENUM, 1));
        if (!PetscBinaryBigEndian()) PetscCall(PetscByteSwap(cxyz, PETSC_DOUBLE, 3));
        PetscCall(PetscMemcpy(&nid, cnid, sizeof(int)));
        PetscCall(PetscMemcpy(xyz, cxyz, 3*sizeof(double)));
        if (byteSwap) PetscCall(PetscByteSwap(&nid, PETSC_ENUM, 1));
        if (byteSwap) PetscCall(PetscByteSwap(xyz, PETSC_DOUBLE, 3));
        nodes->id[n] = nid;
        for (t = 0; t < GMSH_MAX_TAGS; ++t) nodes->tag[n*GMSH_MAX_TAGS+t] = -1;
      }
    } else {
      for (node = 0; node < numNodes; ++node, ++n) {
        double *xyz = nodes->xyz + n*3;
        PetscCall(PetscViewerRead(viewer, &nid, 1, NULL, PETSC_ENUM));
        PetscCall(PetscViewerRead(viewer, xyz, 3, NULL, PETSC_DOUBLE));
        if (byteSwap) PetscCall(PetscByteSwap(&nid, PETSC_ENUM, 1));
        if (byteSwap) PetscCall(PetscByteSwap(xyz, PETSC_DOUBLE, 3));
        nodes->id[n] = nid;
        for (t = 0; t < GMSH_MAX_TAGS; ++t) nodes->tag[n*GMSH_MAX_TAGS+t] = -1;
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

  PetscFunctionBegin;
  PetscCall(PetscViewerRead(viewer, &numEntityBlocks, 1, NULL, PETSC_LONG));
  if (byteSwap) PetscCall(PetscByteSwap(&numEntityBlocks, PETSC_LONG, 1));
  PetscCall(PetscViewerRead(viewer, &numTotalElements, 1, NULL, PETSC_LONG));
  if (byteSwap) PetscCall(PetscByteSwap(&numTotalElements, PETSC_LONG, 1));
  PetscCall(GmshElementsCreate(numTotalElements, &elements));
  mesh->numElems = numTotalElements;
  mesh->elements = elements;
  for (c = 0, block = 0; block < numEntityBlocks; ++block) {
    PetscCall(PetscViewerRead(viewer, info, 3, NULL, PETSC_ENUM));
    if (byteSwap) PetscCall(PetscByteSwap(info, PETSC_ENUM, 3));
    eid = info[0]; dim = info[1]; cellType = info[2];
    PetscCall(GmshEntitiesGet(mesh->entities, dim, eid, &entity));
    PetscCall(GmshCellTypeCheck(cellType));
    numVerts = GmshCellMap[cellType].numVerts;
    numNodes = GmshCellMap[cellType].numNodes;
    numTags  = entity->numTags;
    PetscCall(PetscViewerRead(viewer, &numElements, 1, NULL, PETSC_LONG));
    if (byteSwap) PetscCall(PetscByteSwap(&numElements, PETSC_LONG, 1));
    PetscCall(GmshBufferGet(gmsh, (1+numNodes)*numElements, sizeof(int), &ibuf));
    PetscCall(PetscViewerRead(viewer, ibuf, (1+numNodes)*numElements, NULL, PETSC_ENUM));
    if (byteSwap) PetscCall(PetscByteSwap(ibuf, PETSC_ENUM, (1+numNodes)*numElements));
    for (elem = 0; elem < numElements; ++elem, ++c) {
      GmshElement *element = elements + c;
      const int *id = ibuf + elem*(1+numNodes), *nodes = id + 1;
      element->id  = id[0];
      element->dim = dim;
      element->cellType = cellType;
      element->numVerts = numVerts;
      element->numNodes = numNodes;
      element->numTags  = numTags;
      PetscCall(PetscSegBufferGet(mesh->segbuf, (size_t)element->numNodes, &element->nodes));
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

  PetscFunctionBegin;
  if (fileFormat == 22 || !binary) {
    PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
    snum = sscanf(line, "%d", &numPeriodic);
    PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
  } else {
    PetscCall(PetscViewerRead(viewer, &numPeriodic, 1, NULL, PETSC_ENUM));
    if (byteSwap) PetscCall(PetscByteSwap(&numPeriodic, PETSC_ENUM, 1));
  }
  for (i = 0; i < numPeriodic; i++) {
    int    ibuf[3], correspondingDim = -1, correspondingTag = -1, primaryTag = -1, correspondingNode, primaryNode;
    long   j, nNodes;
    double affine[16];

    if (fileFormat == 22 || !binary) {
      PetscCall(PetscViewerRead(viewer, line, 3, NULL, PETSC_STRING));
      snum = sscanf(line, "%d %d %d", &correspondingDim, &correspondingTag, &primaryTag);
      PetscCheck(snum == 3,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    } else {
      PetscCall(PetscViewerRead(viewer, ibuf, 3, NULL, PETSC_ENUM));
      if (byteSwap) PetscCall(PetscByteSwap(ibuf, PETSC_ENUM, 3));
      correspondingDim = ibuf[0]; correspondingTag = ibuf[1]; primaryTag = ibuf[2];
    }
    (void)correspondingDim; (void)correspondingTag; (void)primaryTag; /* unused */

    if (fileFormat == 22 || !binary) {
      PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
      snum = sscanf(line, "%ld", &nNodes);
      if (snum != 1) { /* discard transformation and try again */
        PetscCall(PetscViewerRead(viewer, line, -PETSC_MAX_PATH_LEN, NULL, PETSC_STRING));
        PetscCall(PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING));
        snum = sscanf(line, "%ld", &nNodes);
        PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
      }
    } else {
      PetscCall(PetscViewerRead(viewer, &nNodes, 1, NULL, PETSC_LONG));
      if (byteSwap) PetscCall(PetscByteSwap(&nNodes, PETSC_LONG, 1));
      if (nNodes == -1) { /* discard transformation and try again */
        PetscCall(PetscViewerRead(viewer, affine, 16, NULL, PETSC_DOUBLE));
        PetscCall(PetscViewerRead(viewer, &nNodes, 1, NULL, PETSC_LONG));
        if (byteSwap) PetscCall(PetscByteSwap(&nNodes, PETSC_LONG, 1));
      }
    }

    for (j = 0; j < nNodes; j++) {
      if (fileFormat == 22 || !binary) {
        PetscCall(PetscViewerRead(viewer, line, 2, NULL, PETSC_STRING));
        snum = sscanf(line, "%d %d", &correspondingNode, &primaryNode);
        PetscCheck(snum == 2,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
      } else {
        PetscCall(PetscViewerRead(viewer, ibuf, 2, NULL, PETSC_ENUM));
        if (byteSwap) PetscCall(PetscByteSwap(ibuf, PETSC_ENUM, 2));
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

  PetscFunctionBegin;
  PetscCall(GmshReadSize(gmsh, count, 4));
  PetscCall(GmshEntitiesCreate(count, &mesh->entities));
  for (dim = 0; dim < 4; ++dim) {
    for (index = 0; index < count[dim]; ++index) {
      PetscCall(GmshReadInt(gmsh, &eid, 1));
      PetscCall(GmshEntitiesAdd(mesh->entities, (PetscInt)index, dim, eid, &entity));
      PetscCall(GmshReadDouble(gmsh, entity->bbox, (dim == 0) ? 3 : 6));
      PetscCall(GmshReadSize(gmsh, &numTags, 1));
      PetscCall(GmshBufferGet(gmsh, numTags, sizeof(int), &tags));
      PetscCall(GmshReadInt(gmsh, tags, numTags));
      PetscCheck(numTags <= GMSH_MAX_TAGS, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "PETSc currently supports up to 4 tags per entity, not %" PetscInt_FMT, numTags);
      entity->numTags = numTags;
      for (i = 0; i < entity->numTags; ++i) entity->tags[i] = tags[i];
      if (dim == 0) continue;
      PetscCall(GmshReadSize(gmsh, &numTags, 1));
      PetscCall(GmshBufferGet(gmsh, numTags, sizeof(int), &tags));
      PetscCall(GmshReadInt(gmsh, tags, numTags));
      /* Currently, we do not save the ids for the bounding entities */
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
  int            info[3], dim, eid, parametric;
  PetscInt       sizes[4], numEntityBlocks, numTags, t, numNodes, numNodesBlock = 0, block, node, n;
  GmshEntity     *entity = NULL;
  GmshNodes      *nodes;

  PetscFunctionBegin;
  PetscCall(GmshReadSize(gmsh, sizes, 4));
  numEntityBlocks = sizes[0]; numNodes = sizes[1];
  PetscCall(GmshNodesCreate(numNodes, &nodes));
  mesh->numNodes = numNodes;
  mesh->nodelist = nodes;
  for (block = 0, node = 0; block < numEntityBlocks; ++block, node += numNodesBlock) {
    PetscCall(GmshReadInt(gmsh, info, 3));
    dim = info[0]; eid = info[1]; parametric = info[2];
    PetscCall(GmshEntitiesGet(mesh->entities, dim, eid, &entity));
    numTags = entity->numTags;
    PetscCheck(!parametric, PETSC_COMM_SELF, PETSC_ERR_SUP, "Parametric coordinates not supported");
    PetscCall(GmshReadSize(gmsh, &numNodesBlock, 1));
    PetscCall(GmshReadSize(gmsh, nodes->id+node, numNodesBlock));
    PetscCall(GmshReadDouble(gmsh, nodes->xyz+node*3, numNodesBlock*3));
    for (n = 0; n < numNodesBlock; ++n) {
      PetscInt *tags = &nodes->tag[node*GMSH_MAX_TAGS];

      for (t = 0; t < numTags; ++t) tags[n*GMSH_MAX_TAGS+t] = entity->tags[t];
      for (t = numTags; t < GMSH_MAX_TAGS; ++t) tags[n*GMSH_MAX_TAGS+t] = -1;
    }
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

  PetscFunctionBegin;
  PetscCall(GmshReadSize(gmsh, sizes, 4));
  numEntityBlocks = sizes[0]; numElements = sizes[1];
  PetscCall(GmshElementsCreate(numElements, &elements));
  mesh->numElems = numElements;
  mesh->elements = elements;
  for (c = 0, block = 0; block < numEntityBlocks; ++block) {
    PetscCall(GmshReadInt(gmsh, info, 3));
    dim = info[0]; eid = info[1]; cellType = info[2];
    PetscCall(GmshEntitiesGet(mesh->entities, dim, eid, &entity));
    PetscCall(GmshCellTypeCheck(cellType));
    numVerts = GmshCellMap[cellType].numVerts;
    numNodes = GmshCellMap[cellType].numNodes;
    numTags  = entity->numTags;
    PetscCall(GmshReadSize(gmsh, &numBlockElements, 1));
    PetscCall(GmshBufferGet(gmsh, (1+numNodes)*numBlockElements, sizeof(PetscInt), &ibuf));
    PetscCall(GmshReadSize(gmsh, ibuf, (1+numNodes)*numBlockElements));
    for (elem = 0; elem < numBlockElements; ++elem, ++c) {
      GmshElement *element = elements + c;
      const PetscInt *id = ibuf + elem*(1+numNodes), *nodes = id + 1;
      element->id  = id[0];
      element->dim = dim;
      element->cellType = cellType;
      element->numVerts = numVerts;
      element->numNodes = numNodes;
      element->numTags  = numTags;
      PetscCall(PetscSegBufferGet(mesh->segbuf, (size_t)element->numNodes, &element->nodes));
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

  PetscFunctionBegin;
  PetscCall(GmshReadSize(gmsh, &numPeriodicLinks, 1));
  for (link = 0; link < numPeriodicLinks; ++link) {
    PetscCall(GmshReadInt(gmsh, info, 3));
    PetscCall(GmshReadSize(gmsh, &numAffine, 1));
    PetscCall(GmshReadDouble(gmsh, dbuf, numAffine));
    PetscCall(GmshReadSize(gmsh, &numCorrespondingNodes, 1));
    PetscCall(GmshBufferGet(gmsh, numCorrespondingNodes, sizeof(PetscInt), &nodeTags));
    PetscCall(GmshReadSize(gmsh, nodeTags, numCorrespondingNodes*2));
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

  PetscFunctionBegin;
  PetscCall(GmshReadString(gmsh, line, 3));
  snum = sscanf(line, "%f %d %d", &version, &fileType, &dataSize);
  PetscCheck(snum == 3,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to parse Gmsh file header: %s", line);
  PetscCheck(version >= 2.2,PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at least 2.2", (double)version);
  PetscCheck((int)version != 3,PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f not supported", (double)version);
  PetscCheck(version <= 4.1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at most 4.1", (double)version);
  PetscCheck(!gmsh->binary || fileType,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Viewer is binary but Gmsh file is ASCII");
  PetscCheck(gmsh->binary || !fileType,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Viewer is ASCII but Gmsh file is binary");
  fileFormat = (int)roundf(version*10);
  PetscCheck(fileFormat > 40 || dataSize == sizeof(double),PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Data size %d is not valid for a Gmsh file", dataSize);
  PetscCheck(fileFormat < 41 || dataSize == sizeof(int) || dataSize == sizeof(PetscInt64),PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Data size %d is not valid for a Gmsh file", dataSize);
  gmsh->fileFormat = fileFormat;
  gmsh->dataSize = dataSize;
  gmsh->byteSwap = PETSC_FALSE;
  if (gmsh->binary) {
    PetscCall(GmshReadInt(gmsh, &checkEndian, 1));
    if (checkEndian != 1) {
      PetscCall(PetscByteSwap(&checkEndian, PETSC_ENUM, 1));
      PetscCheck(checkEndian == 1,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to detect endianness in Gmsh file header: %s", line);
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
  char           line[PETSC_MAX_PATH_LEN], name[128+2], *p, *q, *r;
  int            snum, region, dim, tag;

  PetscFunctionBegin;
  PetscCall(GmshReadString(gmsh, line, 1));
  snum = sscanf(line, "%d", &region);
  mesh->numRegions = region;
  PetscCheck(snum == 1,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
  PetscCall(PetscMalloc2(mesh->numRegions, &mesh->regionTags, mesh->numRegions, &mesh->regionNames));
  for (region = 0; region < mesh->numRegions; ++region) {
    PetscCall(GmshReadString(gmsh, line, 2));
    snum = sscanf(line, "%d %d", &dim, &tag);
    PetscCheck(snum == 2,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    PetscCall(GmshReadString(gmsh, line, -(PetscInt)sizeof(line)));
    PetscCall(PetscStrchr(line, '"', &p));
    PetscCheck(p, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    PetscCall(PetscStrrchr(line, '"', &q));
    PetscCheck(q != p, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "File is not a valid Gmsh file");
    PetscCall(PetscStrrchr(line, ':', &r));
    if (p != r) q = r;
    PetscCall(PetscStrncpy(name, p+1, (size_t)(q-p-1)));
    mesh->regionTags[region] = tag;
    PetscCall(PetscStrallocpy(name, &mesh->regionNames[region]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadEntities(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscFunctionBegin;
  switch (gmsh->fileFormat) {
  case 41: PetscCall(GmshReadEntities_v41(gmsh, mesh)); break;
  default: PetscCall(GmshReadEntities_v40(gmsh, mesh)); break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadNodes(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscFunctionBegin;
  switch (gmsh->fileFormat) {
  case 41: PetscCall(GmshReadNodes_v41(gmsh, mesh)); break;
  case 40: PetscCall(GmshReadNodes_v40(gmsh, mesh)); break;
  default: PetscCall(GmshReadNodes_v22(gmsh, mesh)); break;
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
    PetscCall(PetscMalloc1(gmsh->nodeEnd - gmsh->nodeStart, &gmsh->nbuf));
    for (t = 0; t < gmsh->nodeEnd - gmsh->nodeStart; ++t) gmsh->nbuf[t] = PETSC_MIN_INT;
    gmsh->nodeMap = gmsh->nbuf - gmsh->nodeStart;
    for (n = 0; n < mesh->numNodes; ++n) {
      const PetscInt tag = nodes->id[n];
      PetscCheck(gmsh->nodeMap[tag] < 0,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Repeated node tag %" PetscInt_FMT, tag);
      gmsh->nodeMap[tag] = n;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadElements(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscFunctionBegin;
  switch (gmsh->fileFormat) {
  case 41: PetscCall(GmshReadElements_v41(gmsh, mesh)); break;
  case 40: PetscCall(GmshReadElements_v40(gmsh, mesh)); break;
  default: PetscCall(GmshReadElements_v22(gmsh, mesh)); break;
  }

  { /* Reorder elements by codimension and polytope type */
    PetscInt    ne = mesh->numElems;
    GmshElement *elements = mesh->elements;
    PetscInt    keymap[GMSH_NUM_POLYTOPES], nk = 0;
    PetscInt    offset[GMSH_NUM_POLYTOPES+1], e, k;

    for (k = 0; k < GMSH_NUM_POLYTOPES; ++k) keymap[k] = PETSC_MIN_INT;
    PetscCall(PetscMemzero(offset,sizeof(offset)));

    keymap[GMSH_TET] = nk++;
    keymap[GMSH_HEX] = nk++;
    keymap[GMSH_PRI] = nk++;
    keymap[GMSH_PYR] = nk++;
    keymap[GMSH_TRI] = nk++;
    keymap[GMSH_QUA] = nk++;
    keymap[GMSH_SEG] = nk++;
    keymap[GMSH_VTX] = nk++;

    PetscCall(GmshElementsCreate(mesh->numElems, &mesh->elements));
#define key(eid) keymap[GmshCellMap[elements[(eid)].cellType].polytope]
    for (e = 0; e < ne; ++e) offset[1+key(e)]++;
    for (k = 1; k < nk; ++k) offset[k] += offset[k-1];
    for (e = 0; e < ne; ++e) mesh->elements[offset[key(e)]++] = elements[e];
#undef key
    PetscCall(GmshElementsDestroy(&elements));
  }

  { /* Mesh dimension and order */
    GmshElement *elem = mesh->numElems ? mesh->elements : NULL;
    mesh->dim   = elem ? GmshCellMap[elem->cellType].dim   : 0;
    mesh->order = elem ? GmshCellMap[elem->cellType].order : 0;
  }

  {
    PetscBT  vtx;
    PetscInt dim = mesh->dim, e, n, v;

    PetscCall(PetscBTCreate(mesh->numNodes, &vtx));

    /* Compute number of cells and set of vertices */
    mesh->numCells = 0;
    for (e = 0; e < mesh->numElems; ++e) {
      GmshElement *elem = mesh->elements + e;
      if (elem->dim == dim && dim > 0) mesh->numCells++;
      for (v = 0; v < elem->numVerts; v++) {
        PetscCall(PetscBTSet(vtx, elem->nodes[v]));
      }
    }

    /* Compute numbering for vertices */
    mesh->numVerts = 0;
    PetscCall(PetscMalloc1(mesh->numNodes, &mesh->vertexMap));
    for (n = 0; n < mesh->numNodes; ++n)
      mesh->vertexMap[n] = PetscBTLookup(vtx, n) ? mesh->numVerts++ : PETSC_MIN_INT;

    PetscCall(PetscBTDestroy(&vtx));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode GmshReadPeriodic(GmshFile *gmsh, GmshMesh *mesh)
{
  PetscInt       n;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(mesh->numNodes, &mesh->periodMap));
  for (n = 0; n < mesh->numNodes; ++n) mesh->periodMap[n] = n;
  switch (gmsh->fileFormat) {
  case 41: PetscCall(GmshReadPeriodic_v41(gmsh, mesh->periodMap)); break;
  default: PetscCall(GmshReadPeriodic_v40(gmsh, mesh->periodMap)); break;
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

static inline DMPolytopeType DMPolytopeTypeFromGmsh(PetscInt cellType)
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

  PetscFunctionBegin;
  /* Create space */
  PetscCall(PetscSpaceCreate(comm, &P));
  PetscCall(PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL));
  PetscCall(PetscSpacePolynomialSetTensor(P, isTensor));
  PetscCall(PetscSpaceSetNumComponents(P, Nc));
  PetscCall(PetscSpaceSetNumVariables(P, dim));
  PetscCall(PetscSpaceSetDegree(P, k, PETSC_DETERMINE));
  if (prefix) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) P, prefix));
    PetscCall(PetscSpaceSetFromOptions(P));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) P, NULL));
    PetscCall(PetscSpaceGetDegree(P, &k, NULL));
  }
  PetscCall(PetscSpaceSetUp(P));
  /* Create dual space */
  PetscCall(PetscDualSpaceCreate(comm, &Q));
  PetscCall(PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscDualSpaceLagrangeSetTensor(Q, isTensor));
  PetscCall(PetscDualSpaceLagrangeSetContinuity(Q, continuity));
  PetscCall(PetscDualSpaceLagrangeSetNodeType(Q, nodeType, endpoint, 0));
  PetscCall(PetscDualSpaceSetNumComponents(Q, Nc));
  PetscCall(PetscDualSpaceSetOrder(Q, k));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, isSimplex), &K));
  PetscCall(PetscDualSpaceSetDM(Q, K));
  PetscCall(DMDestroy(&K));
  if (prefix) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) Q, prefix));
    PetscCall(PetscDualSpaceSetFromOptions(Q));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) Q, NULL));
  }
  PetscCall(PetscDualSpaceSetUp(Q));
  /* Create quadrature */
  if (isSimplex) {
    PetscCall(PetscDTStroudConicalQuadrature(dim,   1, k+1, -1, +1, &q));
    PetscCall(PetscDTStroudConicalQuadrature(dim-1, 1, k+1, -1, +1, &fq));
  } else {
    PetscCall(PetscDTGaussTensorQuadrature(dim,   1, k+1, -1, +1, &q));
    PetscCall(PetscDTGaussTensorQuadrature(dim-1, 1, k+1, -1, +1, &fq));
  }
  /* Create finite element */
  PetscCall(PetscFECreate(comm, fem));
  PetscCall(PetscFESetType(*fem, PETSCFEBASIC));
  PetscCall(PetscFESetNumComponents(*fem, Nc));
  PetscCall(PetscFESetBasisSpace(*fem, P));
  PetscCall(PetscFESetDualSpace(*fem, Q));
  PetscCall(PetscFESetQuadrature(*fem, q));
  PetscCall(PetscFESetFaceQuadrature(*fem, fq));
  if (prefix) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix));
    PetscCall(PetscFESetFromOptions(*fem));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *fem, NULL));
  }
  PetscCall(PetscFESetUp(*fem));
  /* Cleanup */
  PetscCall(PetscSpaceDestroy(&P));
  PetscCall(PetscDualSpaceDestroy(&Q));
  PetscCall(PetscQuadratureDestroy(&q));
  PetscCall(PetscQuadratureDestroy(&fq));
  /* Set finite element name */
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s%" PetscInt_FMT, isSimplex? "P" : "Q", k));
  PetscCall(PetscFESetName(*fem, name));
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

.seealso: `DMPlexCreateFromFile()`, `DMPlexCreateGmsh()`, `DMPlexCreate()`
@*/
PetscErrorCode DMPlexCreateGmshFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscViewer     viewer;
  PetscMPIInt     rank;
  int             fileType;
  PetscViewerType vtype;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  /* Determine Gmsh file type (ASCII or binary) from file header */
  if (rank == 0) {
    GmshFile    gmsh[1];
    char        line[PETSC_MAX_PATH_LEN];
    int         snum;
    float       version;

    PetscCall(PetscArrayzero(gmsh,1));
    PetscCall(PetscViewerCreate(PETSC_COMM_SELF, &gmsh->viewer));
    PetscCall(PetscViewerSetType(gmsh->viewer, PETSCVIEWERASCII));
    PetscCall(PetscViewerFileSetMode(gmsh->viewer, FILE_MODE_READ));
    PetscCall(PetscViewerFileSetName(gmsh->viewer, filename));
    /* Read only the first two lines of the Gmsh file */
    PetscCall(GmshReadSection(gmsh, line));
    PetscCall(GmshExpect(gmsh, "$MeshFormat", line));
    PetscCall(GmshReadString(gmsh, line, 2));
    snum = sscanf(line, "%f %d", &version, &fileType);
    PetscCheck(snum == 2,PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to parse Gmsh file header: %s", line);
    PetscCheck(version >= 2.2,PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at least 2.2", (double)version);
    PetscCheck((int)version != 3,PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f not supported", (double)version);
    PetscCheck(version <= 4.1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Gmsh file version %3.1f must be at most 4.1", (double)version);
    PetscCall(PetscViewerDestroy(&gmsh->viewer));
  }
  PetscCallMPI(MPI_Bcast(&fileType, 1, MPI_INT, 0, comm));
  vtype = (fileType == 0) ? PETSCVIEWERASCII : PETSCVIEWERBINARY;

  /* Create appropriate viewer and build plex */
  PetscCall(PetscViewerCreate(comm, &viewer));
  PetscCall(PetscViewerSetType(viewer, vtype));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer, filename));
  PetscCall(DMPlexCreateGmsh(comm, viewer, interpolate, dm));
  PetscCall(PetscViewerDestroy(&viewer));
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

.seealso: `DMPLEX`, `DMCreate()`
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
  PetscBool      binary, usemarker = PETSC_FALSE, useregions = PETSC_FALSE, markvertices = PETSC_FALSE;
  PetscBool      hybrid = interpolate, periodic = PETSC_TRUE;
  PetscBool      highOrder = PETSC_TRUE, highOrderSet, project = PETSC_FALSE;
  PetscBool      isSimplex = PETSC_FALSE, isHybrid = PETSC_FALSE, hasTetra = PETSC_FALSE;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscObjectOptionsBegin((PetscObject)viewer);
  PetscOptionsHeadBegin(PetscOptionsObject,"DMPlex Gmsh options");
  PetscCall(PetscOptionsBool("-dm_plex_gmsh_hybrid", "Generate hybrid cell bounds", "DMPlexCreateGmsh", hybrid, &hybrid, NULL));
  PetscCall(PetscOptionsBool("-dm_plex_gmsh_periodic","Read Gmsh periodic section", "DMPlexCreateGmsh", periodic, &periodic, NULL));
  PetscCall(PetscOptionsBool("-dm_plex_gmsh_highorder","Generate high-order coordinates", "DMPlexCreateGmsh", highOrder, &highOrder, &highOrderSet));
  PetscCall(PetscOptionsBool("-dm_plex_gmsh_project", "Project high-order coordinates to a different space", "DMPlexCreateGmsh", project, &project, NULL));
  PetscCall(PetscOptionsBool("-dm_plex_gmsh_use_marker", "Generate marker label", "DMPlexCreateGmsh", usemarker, &usemarker, NULL));
  PetscCall(PetscOptionsBool("-dm_plex_gmsh_use_regions", "Generate labels with region names", "DMPlexCreateGmsh", useregions, &useregions, NULL));
  PetscCall(PetscOptionsBool("-dm_plex_gmsh_mark_vertices", "Add vertices to generated labels", "DMPlexCreateGmsh", markvertices, &markvertices, NULL));
  PetscCall(PetscOptionsBoundedInt("-dm_plex_gmsh_spacedim", "Embedding space dimension", "DMPlexCreateGmsh", coordDim, &coordDim, NULL, PETSC_DECIDE));
  PetscOptionsHeadEnd();
  PetscOptionsEnd();

  PetscCall(GmshCellInfoSetUp());

  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(PetscLogEventBegin(DMPLEX_CreateGmsh,*dm,NULL,NULL,NULL));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &binary));

  /* Binary viewers read on all ranks, get subviewer to read only in rank 0 */
  if (binary) {
    parentviewer = viewer;
    PetscCall(PetscViewerGetSubViewer(parentviewer, PETSC_COMM_SELF, &viewer));
  }

  if (rank == 0) {
    GmshFile  gmsh[1];
    char      line[PETSC_MAX_PATH_LEN];
    PetscBool match;

    PetscCall(PetscArrayzero(gmsh,1));
    gmsh->viewer = viewer;
    gmsh->binary = binary;

    PetscCall(GmshMeshCreate(&mesh));

    /* Read mesh format */
    PetscCall(GmshReadSection(gmsh, line));
    PetscCall(GmshExpect(gmsh, "$MeshFormat", line));
    PetscCall(GmshReadMeshFormat(gmsh));
    PetscCall(GmshReadEndSection(gmsh, "$EndMeshFormat", line));

    /* OPTIONAL Read physical names */
    PetscCall(GmshReadSection(gmsh, line));
    PetscCall(GmshMatch(gmsh, "$PhysicalNames", line, &match));
    if (match) {
      PetscCall(GmshExpect(gmsh, "$PhysicalNames", line));
      PetscCall(GmshReadPhysicalNames(gmsh, mesh));
      PetscCall(GmshReadEndSection(gmsh, "$EndPhysicalNames", line));
      /* Initial read for entity section */
      PetscCall(GmshReadSection(gmsh, line));
    }

    /* Read entities */
    if (gmsh->fileFormat >= 40) {
      PetscCall(GmshExpect(gmsh, "$Entities", line));
      PetscCall(GmshReadEntities(gmsh, mesh));
      PetscCall(GmshReadEndSection(gmsh, "$EndEntities", line));
      /* Initial read for nodes section */
      PetscCall(GmshReadSection(gmsh, line));
    }

    /* Read nodes */
    PetscCall(GmshExpect(gmsh, "$Nodes", line));
    PetscCall(GmshReadNodes(gmsh, mesh));
    PetscCall(GmshReadEndSection(gmsh, "$EndNodes", line));

    /* Read elements */
    PetscCall(GmshReadSection(gmsh, line));
    PetscCall(GmshExpect(gmsh, "$Elements", line));
    PetscCall(GmshReadElements(gmsh, mesh));
    PetscCall(GmshReadEndSection(gmsh, "$EndElements", line));

    /* Read periodic section (OPTIONAL) */
    if (periodic) {
      PetscCall(GmshReadSection(gmsh, line));
      PetscCall(GmshMatch(gmsh, "$Periodic", line, &periodic));
    }
    if (periodic) {
      PetscCall(GmshExpect(gmsh, "$Periodic", line));
      PetscCall(GmshReadPeriodic(gmsh, mesh));
      PetscCall(GmshReadEndSection(gmsh, "$EndPeriodic", line));
    }

    PetscCall(PetscFree(gmsh->wbuf));
    PetscCall(PetscFree(gmsh->sbuf));
    PetscCall(PetscFree(gmsh->nbuf));

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
    PetscCall(PetscViewerRestoreSubViewer(parentviewer, PETSC_COMM_SELF, &viewer));
  }

  {
    int buf[6];

    buf[0] = (int)dim;
    buf[1] = (int)order;
    buf[2] = periodic;
    buf[3] = isSimplex;
    buf[4] = isHybrid;
    buf[5] = hasTetra;

    PetscCallMPI(MPI_Bcast(buf, 6, MPI_INT, 0, comm));

    dim       = buf[0];
    order     = buf[1];
    periodic  = buf[2] ? PETSC_TRUE : PETSC_FALSE;
    isSimplex = buf[3] ? PETSC_TRUE : PETSC_FALSE;
    isHybrid  = buf[4] ? PETSC_TRUE : PETSC_FALSE;
    hasTetra  = buf[5] ? PETSC_TRUE : PETSC_FALSE;
  }

  if (!highOrderSet) highOrder = (order > 1) ? PETSC_TRUE : PETSC_FALSE;
  PetscCheck(!highOrder || !isHybrid,comm, PETSC_ERR_SUP, "No support for discretization on hybrid meshes yet");

  /* We do not want this label automatically computed, instead we fill it here */
  PetscCall(DMCreateLabel(*dm, "celltype"));

  /* Allocate the cell-vertex mesh */
  PetscCall(DMPlexSetChart(*dm, 0, numCells+numVerts));
  for (cell = 0; cell < numCells; ++cell) {
    GmshElement *elem = mesh->elements + cell;
    DMPolytopeType ctype = DMPolytopeTypeFromGmsh(elem->cellType);
    if (hybrid && hasTetra && ctype == DM_POLYTOPE_TRI_PRISM) ctype = DM_POLYTOPE_TRI_PRISM_TENSOR;
    PetscCall(DMPlexSetConeSize(*dm, cell, elem->numVerts));
    PetscCall(DMPlexSetCellType(*dm, cell, ctype));
  }
  for (v = numCells; v < numCells+numVerts; ++v) {
    PetscCall(DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT));
  }
  PetscCall(DMSetUp(*dm));

  /* Add cell-vertex connections */
  for (cell = 0; cell < numCells; ++cell) {
    GmshElement *elem = mesh->elements + cell;
    for (v = 0; v < elem->numVerts; ++v) {
      const PetscInt nn = elem->nodes[v];
      const PetscInt vv = mesh->vertexMap[nn];
      cone[v] = numCells + vv;
    }
    PetscCall(DMPlexReorderCell(*dm, cell, cone));
    PetscCall(DMPlexSetCone(*dm, cell, cone));
  }

  PetscCall(DMSetDimension(*dm, dim));
  PetscCall(DMPlexSymmetrize(*dm));
  PetscCall(DMPlexStratify(*dm));
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm  = idm;
  }

  /* Create the label "marker" over the whole boundary */
  PetscCheck(!usemarker || interpolate || dim <= 1,comm,PETSC_ERR_SUP,"Cannot create marker label without interpolation");
  if (rank == 0 && usemarker) {
    PetscInt f, fStart, fEnd;

    PetscCall(DMCreateLabel(*dm, "marker"));
    PetscCall(DMPlexGetHeightStratum(*dm, 1, &fStart, &fEnd));
    for (f = fStart; f < fEnd; ++f) {
      PetscInt suppSize;

      PetscCall(DMPlexGetSupportSize(*dm, f, &suppSize));
      if (suppSize == 1) {
        PetscInt *cone = NULL, coneSize, p;

        PetscCall(DMPlexGetTransitiveClosure(*dm, f, PETSC_TRUE, &coneSize, &cone));
        for (p = 0; p < coneSize; p += 2) {
          PetscCall(DMSetLabelValue_Fast(*dm, &marker, "marker", cone[p], 1));
        }
        PetscCall(DMPlexRestoreTransitiveClosure(*dm, f, PETSC_TRUE, &coneSize, &cone));
      }
    }
  }

  if (rank == 0) {
    const PetscInt Nr = useregions ? mesh->numRegions : 0;
    PetscInt       vStart, vEnd;

    PetscCall(PetscCalloc1(Nr, &regionSets));
    PetscCall(DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd));
    for (cell = 0, e = 0; e < numElems; ++e) {
      GmshElement *elem = mesh->elements + e;

      /* Create cell sets */
      if (elem->dim == dim && dim > 0) {
        if (elem->numTags > 0) {
          const PetscInt tag = elem->tags[0];
          PetscInt       r;

          if (!Nr) PetscCall(DMSetLabelValue_Fast(*dm, &cellSets, "Cell Sets", cell, tag));
          for (r = 0; r < Nr; ++r) {
            if (mesh->regionTags[r] == tag) PetscCall(DMSetLabelValue_Fast(*dm, &regionSets[r], mesh->regionNames[r], cell, tag));
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
        PetscCall(DMPlexGetFullJoin(*dm, elem->numVerts, cone, &joinSize, &join));
        PetscCheck(joinSize == 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Could not determine Plex facet for Gmsh element %" PetscInt_FMT " (Plex cell %" PetscInt_FMT ")", elem->id, e);
        if (!Nr) PetscCall(DMSetLabelValue_Fast(*dm, &faceSets, "Face Sets", join[0], tag));
        for (r = 0; r < Nr; ++r) {
          if (mesh->regionTags[r] == tag) PetscCall(DMSetLabelValue_Fast(*dm, &regionSets[r], mesh->regionNames[r], join[0], tag));
        }
        PetscCall(DMPlexRestoreJoin(*dm, elem->numVerts, cone, &joinSize, &join));
      }

      /* Create vertex sets */
      if (elem->dim == 0) {
        if (elem->numTags > 0) {
          const PetscInt nn = elem->nodes[0];
          const PetscInt vv = mesh->vertexMap[nn];
          const PetscInt tag = elem->tags[0];
          PetscInt       r;

          if (!Nr) PetscCall(DMSetLabelValue_Fast(*dm, &vertSets, "Vertex Sets", vStart + vv, tag));
          for (r = 0; r < Nr; ++r) {
            if (mesh->regionTags[r] == tag) PetscCall(DMSetLabelValue_Fast(*dm, &regionSets[r], mesh->regionNames[r], vStart + vv, tag));
          }
        }
      }
    }
    if (markvertices) {
      for (v = 0; v < numNodes; ++v) {
        const PetscInt  vv   = mesh->vertexMap[v];
        const PetscInt *tags = &mesh->nodelist->tag[v*GMSH_MAX_TAGS];
        PetscInt        r, t;

        for (t = 0; t < GMSH_MAX_TAGS; ++t) {
          const PetscInt tag = tags[t];

          if (tag == -1) continue;
          if (!Nr) PetscCall(DMSetLabelValue_Fast(*dm, &vertSets, "Vertex Sets", vStart + vv, tag));
          for (r = 0; r < Nr; ++r) {
            if (mesh->regionTags[r] == tag) PetscCall(DMSetLabelValue_Fast(*dm, &regionSets[r], mesh->regionNames[r], vStart + vv, tag));
          }
        }
      }
    }
    PetscCall(PetscFree(regionSets));
  }

  { /* Create Cell/Face/Vertex Sets labels at all processes */
    enum {n = 4};
    PetscBool flag[n];

    flag[0] = cellSets ? PETSC_TRUE : PETSC_FALSE;
    flag[1] = faceSets ? PETSC_TRUE : PETSC_FALSE;
    flag[2] = vertSets ? PETSC_TRUE : PETSC_FALSE;
    flag[3] = marker   ? PETSC_TRUE : PETSC_FALSE;
    PetscCallMPI(MPI_Bcast(flag, n, MPIU_BOOL, 0, comm));
    if (flag[0]) PetscCall(DMCreateLabel(*dm, "Cell Sets"));
    if (flag[1]) PetscCall(DMCreateLabel(*dm, "Face Sets"));
    if (flag[2]) PetscCall(DMCreateLabel(*dm, "Vertex Sets"));
    if (flag[3]) PetscCall(DMCreateLabel(*dm, "marker"));
  }

  if (periodic) {
    PetscCall(PetscBTCreate(numVerts, &periodicVerts));
    for (n = 0; n < numNodes; ++n) {
      if (mesh->vertexMap[n] >= 0) {
        if (PetscUnlikely(mesh->periodMap[n] != n)) {
          PetscInt m = mesh->periodMap[n];
          PetscCall(PetscBTSet(periodicVerts, mesh->vertexMap[n]));
          PetscCall(PetscBTSet(periodicVerts, mesh->vertexMap[m]));
        }
      }
    }
    PetscCall(PetscBTCreate(numCells, &periodicCells));
    for (cell = 0; cell < numCells; ++cell) {
      GmshElement *elem = mesh->elements + cell;
      for (v = 0; v < elem->numVerts; ++v) {
        PetscInt nn = elem->nodes[v];
        PetscInt vv = mesh->vertexMap[nn];
        if (PetscUnlikely(PetscBTLookup(periodicVerts, vv))) {
          PetscCall(PetscBTSet(periodicCells, cell)); break;
        }
      }
    }
  }

  /* Setup coordinate DM */
  if (coordDim < 0) coordDim = dim;
  PetscCall(DMSetCoordinateDim(*dm, coordDim));
  PetscCall(DMGetCoordinateDM(*dm, &cdm));
  if (highOrder) {
    PetscFE         fe;
    PetscBool       continuity = periodic ? PETSC_FALSE : PETSC_TRUE;
    PetscDTNodeType nodeType   = PETSCDTNODES_EQUISPACED;

    if (isSimplex) continuity = PETSC_FALSE; /* XXX FIXME Requires DMPlexSetClosurePermutationLexicographic() */

    PetscCall(GmshCreateFE(comm, NULL, isSimplex, continuity, nodeType, dim, coordDim, order, &fe));
    PetscCall(PetscFEViewFromOptions(fe, NULL, "-dm_plex_gmsh_fe_view"));
    PetscCall(DMSetField(cdm, 0, NULL, (PetscObject) fe));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMCreateDS(cdm));
  }

  /* Create coordinates */
  if (highOrder) {

    PetscInt     maxDof = GmshNumNodes_HEX(order)*coordDim;
    double       *coords = mesh ? mesh->nodelist->xyz : NULL;
    PetscSection section;
    PetscScalar  *cellCoords;

    PetscCall(DMSetLocalSection(cdm, NULL));
    PetscCall(DMGetLocalSection(cdm, &coordSection));
    PetscCall(PetscSectionClone(coordSection, &section));
    PetscCall(DMPlexSetClosurePermutationTensor(cdm, 0, section)); /* XXX Implement DMPlexSetClosurePermutationLexicographic() */

    PetscCall(DMCreateLocalVector(cdm, &coordinates));
    PetscCall(PetscMalloc1(maxDof, &cellCoords));
    for (cell = 0; cell < numCells; ++cell) {
      GmshElement *elem = mesh->elements + cell;
      const int *lexorder = GmshCellMap[elem->cellType].lexorder();
      int s = 0;
      for (n = 0; n < elem->numNodes; ++n) {
        while (lexorder[n+s] < 0) ++s;
        const PetscInt node = elem->nodes[lexorder[n+s]];
        for (d = 0; d < coordDim; ++d) cellCoords[(n+s)*coordDim+d] = (PetscReal) coords[node*3+d];
      }
      if (s) {
        /* For the coordinate mapping we weight vertices by -1/4 and edges by 1/2, which we get from Q_2 interpolation */
        PetscReal quaCenterWeights[9]  = {-0.25, 0.5, -0.25, 0.5, 0.0, 0.5, -0.25, 0.5, -0.25};
        /* For the coordinate mapping we weight vertices by -1/4 and edges by 1/2, which we get from Q_2 interpolation */
        PetscReal hexBottomWeights[27] = {-0.25, 0.5,  -0.25, 0.5,  0.0, 0.5,  -0.25, 0.5,  -0.25,
                                           0.0,  0.0,   0.0,  0.0,  0.0, 0.0,   0.0,  0.0,   0.0,
                                           0.0,  0.0,   0.0,  0.0,  0.0, 0.0,   0.0,  0.0,   0.0};
        PetscReal hexFrontWeights[27]  = {-0.25, 0.5,  -0.25, 0.0,  0.0, 0.0,   0.0,  0.0,   0.0,
                                           0.5,  0.0,   0.5,  0.0,  0.0, 0.0,   0.0,  0.0,   0.0,
                                          -0.25, 0.5,  -0.25, 0.0,  0.0, 0.0,   0.0,  0.0,   0.0};
        PetscReal hexLeftWeights[27]   = {-0.25, 0.0,   0.0,  0.5,  0.0, 0.0,  -0.25, 0.0,   0.0,
                                           0.5,  0.0,   0.0,  0.0,  0.0, 0.0,   0.5,  0.0,   0.0,
                                          -0.25, 0.0,   0.0,  0.5,  0.0, 0.0,  -0.25, 0.0,   0.0};
        PetscReal hexRightWeights[27]  = { 0.0,  0.0,  -0.25, 0.0,  0.0, 0.5,   0.0,  0.0,  -0.25,
                                           0.0,  0.0,   0.5,  0.0,  0.0, 0.0,   0.0,  0.0,   0.5,
                                           0.0,  0.0,  -0.25, 0.0,  0.0, 0.5,   0.0,  0.0,  -0.25};
        PetscReal hexBackWeights[27]   = { 0.0,  0.0,   0.0,  0.0,  0.0, 0.0,  -0.25, 0.5,  -0.25,
                                           0.0,  0.0,   0.0,  0.0,  0.0, 0.0,   0.5,  0.0,   0.5,
                                           0.0,  0.0,   0.0,  0.0,  0.0, 0.0,  -0.25, 0.5,  -0.25};
        PetscReal hexTopWeights[27]    = { 0.0,  0.0,   0.0,  0.0,  0.0, 0.0,   0.0,  0.0,   0.0,
                                           0.0,  0.0,   0.0,  0.0,  0.0, 0.0,   0.0,  0.0,   0.0,
                                          -0.25, 0.5,  -0.25, 0.5,  0.0, 0.5,  -0.25, 0.5,  -0.25};
        PetscReal hexCenterWeights[27] = {-0.25, 0.25, -0.25, 0.25, 0.0, 0.25, -0.25, 0.25, -0.25,
                                           0.25, 0.0,   0.25, 0.0,  0.0, 0.0,   0.25, 0.0,   0.25,
                                          -0.25, 0.25, -0.25, 0.25, 0.0, 0.25, -0.25, 0.25, -0.25};
        PetscReal  *sdWeights2[9]      = {NULL, NULL, NULL, NULL, quaCenterWeights, NULL, NULL, NULL, NULL};
        PetscReal  *sdWeights3[27]     = {NULL, NULL, NULL, NULL, hexBottomWeights, NULL, NULL, NULL, NULL,
                                          NULL, hexFrontWeights, NULL, hexLeftWeights, hexCenterWeights, hexRightWeights, NULL, hexBackWeights, NULL,
                                          NULL, NULL, NULL, NULL, hexTopWeights,    NULL, NULL, NULL, NULL};
        PetscReal **sdWeights[4]       = {NULL, NULL, sdWeights2, sdWeights3};

        /* Missing entries in serendipity cell, only works for 8-node quad and 20-node hex */
        for (n = 0; n < elem->numNodes+s; ++n) {
          if (lexorder[n] >= 0) continue;
          for (d = 0; d < coordDim; ++d) cellCoords[n*coordDim+d] = 0.0;
          for (int bn = 0; bn < elem->numNodes+s; ++bn) {
            if (lexorder[bn] < 0) continue;
            const PetscReal *weights = sdWeights[coordDim][n];
            const PetscInt   bnode   = elem->nodes[lexorder[bn]];
            for (d = 0; d < coordDim; ++d) cellCoords[n*coordDim+d] += weights[bn] * (PetscReal) coords[bnode*3+d];
          }
        }
      }
      PetscCall(DMPlexVecSetClosure(cdm, section, coordinates, cell, cellCoords, INSERT_VALUES));
    }
    PetscCall(PetscSectionDestroy(&section));
    PetscCall(PetscFree(cellCoords));

  } else {

    PetscInt    *nodeMap;
    double      *coords = mesh ? mesh->nodelist->xyz : NULL;
    PetscScalar *pointCoords;

    PetscCall(DMGetLocalSection(cdm, &coordSection));
    PetscCall(PetscSectionSetNumFields(coordSection, 1));
    PetscCall(PetscSectionSetFieldComponents(coordSection, 0, coordDim));
    if (periodic) { /* we need to localize coordinates on cells */
      PetscCall(PetscSectionSetChart(coordSection, 0, numCells+numVerts));
    } else {
      PetscCall(PetscSectionSetChart(coordSection, numCells, numCells+numVerts));
    }
    if (periodic) {
      for (cell = 0; cell < numCells; ++cell) {
        if (PetscUnlikely(PetscBTLookup(periodicCells, cell))) {
          GmshElement *elem = mesh->elements + cell;
          PetscInt dof = elem->numVerts * coordDim;
          PetscCall(PetscSectionSetDof(coordSection, cell, dof));
          PetscCall(PetscSectionSetFieldDof(coordSection, cell, 0, dof));
        }
      }
    }
    for (v = numCells; v < numCells+numVerts; ++v) {
      PetscCall(PetscSectionSetDof(coordSection, v, coordDim));
      PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, coordDim));
    }
    PetscCall(PetscSectionSetUp(coordSection));

    PetscCall(DMCreateLocalVector(cdm, &coordinates));
    PetscCall(VecGetArray(coordinates, &pointCoords));
    if (periodic) {
      for (cell = 0; cell < numCells; ++cell) {
        if (PetscUnlikely(PetscBTLookup(periodicCells, cell))) {
          GmshElement *elem = mesh->elements + cell;
          PetscInt off, node;
          for (v = 0; v < elem->numVerts; ++v)
            cone[v] = elem->nodes[v];
          PetscCall(DMPlexReorderCell(cdm, cell, cone));
          PetscCall(PetscSectionGetOffset(coordSection, cell, &off));
          for (v = 0; v < elem->numVerts; ++v)
            for (node = cone[v], d = 0; d < coordDim; ++d)
              pointCoords[off++] = (PetscReal) coords[node*3+d];
        }
      }
    }
    PetscCall(PetscMalloc1(numVerts, &nodeMap));
    for (n = 0; n < numNodes; n++)
      if (mesh->vertexMap[n] >= 0)
        nodeMap[mesh->vertexMap[n]] = n;
    for (v = 0; v < numVerts; ++v) {
      PetscInt off, node = nodeMap[v];
      PetscCall(PetscSectionGetOffset(coordSection, numCells + v, &off));
      for (d = 0; d < coordDim; ++d)
        pointCoords[off+d] = (PetscReal) coords[node*3+d];
    }
    PetscCall(PetscFree(nodeMap));
    PetscCall(VecRestoreArray(coordinates, &pointCoords));

  }

  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetBlockSize(coordinates, coordDim));
  PetscCall(DMSetCoordinatesLocal(*dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
  PetscCall(DMSetPeriodicity(*dm, periodic, NULL, NULL, NULL));

  PetscCall(GmshMeshDestroy(&mesh));
  PetscCall(PetscBTDestroy(&periodicVerts));
  PetscCall(PetscBTDestroy(&periodicCells));

  if (highOrder && project)  {
    PetscFE         fe;
    const char      prefix[]   = "dm_plex_gmsh_project_";
    PetscBool       continuity = periodic ? PETSC_FALSE : PETSC_TRUE;
    PetscDTNodeType nodeType   = PETSCDTNODES_GAUSSJACOBI;

    if (isSimplex) continuity = PETSC_FALSE; /* XXX FIXME Requires DMPlexSetClosurePermutationLexicographic() */

    PetscCall(GmshCreateFE(comm, prefix, isSimplex, continuity, nodeType, dim, coordDim, order, &fe));
    PetscCall(PetscFEViewFromOptions(fe, NULL, "-dm_plex_gmsh_project_fe_view"));
    PetscCall(DMProjectCoordinates(*dm, fe));
    PetscCall(PetscFEDestroy(&fe));
  }

  PetscCall(PetscLogEventEnd(DMPLEX_CreateGmsh,*dm,NULL,NULL,NULL));
  PetscFunctionReturn(0);
}
