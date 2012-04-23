#include <petsc-private/compleximpl.h>   /*I      "petscdmcomplex.h"   I*/

/*================================ Start Converted TetGen Objects ================================*/
/*  Geometric predicates                                                       */
/*                                                                             */
/*  Return one of the values +1, 0, and -1 on basic geometric questions such   */
/*  as the orientation of point sets, in-circle, and in-sphere tests.  They    */
/*  are basic units for implmenting geometric algorithms.  TetGen uses two 3D  */
/*  geometric predicates: the orientation and in-sphere tests.                 */
/*                                                                             */
/*  Orientation test:  let a, b, c be a sequence of 3 non-collinear points in  */
/*  R^3.  They defines a unique hypeplane H.  Let H+ and H- be the two spaces  */
/*  separated by H, which are defined as follows (using the left-hand rule):   */
/*  make a fist using your left hand in such a way that your fingers follow    */
/*  the order of a, b and c, then your thumb is pointing to H+.  Given any     */
/*  point d in R^3, the orientation test returns +1 if d lies in H+, -1 if d   */
/*  lies in H-, or 0 if d lies on H.                                           */
/*                                                                             */
/*  In-sphere test:  let a, b, c, d be 4 non-coplanar points in R^3.  They     */
/*  defines a unique circumsphere S.  Given any point e in R^3, the in-sphere  */
/*  test returns +1 if e lies inside S, or -1 if e lies outside S, or 0 if e   */
/*  lies on S.                                                                 */
/*                                                                             */
/*  The following routines use arbitrary precision floating-point arithmetic.  */
/*  They are provided by J. R. Schewchuk in public domain (http:www.cs.cmu.    */
/*  edu/~quake/robust.html). The source code are in "predicates.cxx".          */
PetscReal TetGenExactInit();
PetscReal TetGenOrient3D(PetscReal *pa, PetscReal *pb, PetscReal *pc, PetscReal *pd);
PetscReal TetGenInsphere(PetscReal *pa, PetscReal *pb, PetscReal *pc, PetscReal *pd, PetscReal *pe);

/*  Labels that signify whether a record consists primarily of pointers */
/*    or of floating-point words.  Used for data alignment. */
typedef enum {POINTER, FLOATINGPOINT} wordtype;

/*  Labels that signify the type of a vertex. */
typedef enum {UNUSEDVERTEX, DUPLICATEDVERTEX, NACUTEVERTEX, ACUTEVERTEX, FREESEGVERTEX, FREESUBVERTEX, FREEVOLVERTEX, DEADVERTEX = -32768} verttype;

/*  Labels that signify the result of triangle-triangle intersection test. */
typedef enum {DISJOINT, INTERSECT, SHAREVERTEX, SHAREEDGE, SHAREFACE, TOUCHEDGE, TOUCHFACE, INTERVERT, INTEREDGE, INTERFACE, INTERTET,
              TRIEDGEINT, EDGETRIINT, COLLISIONFACE, INTERSUBSEG, INTERSUBFACE, BELOWHULL2} interresult;

/*  Labels that signify the result of point location. */
typedef enum {INTETRAHEDRON, ONFACE, ONEDGE, ONVERTEX, OUTSIDE, ENCSEGMENT} locateresult;

/*  Labels that signify the result of direction finding. */
typedef enum {ACROSSEDGE, ACROSSFACE, LEFTCOLLINEAR, RIGHTCOLLINEAR, TOPCOLLINEAR, BELOWHULL} finddirectionresult;

/*  Labels that signify the type of a subface/subsegment. */
typedef enum {NSHARP, SHARP} shestype;

/*  For efficiency, a variety of data structures are allocated in bulk. */
/*    The following constants determine how many of each structure is allocated at once. */
enum {VERPERBLOCK = 4092, SUBPERBLOCK = 4092, ELEPERBLOCK = 8188};

/*  Labels that signify two edge rings of a triangle (see Muecke's thesis). */
enum {CCW = 0, CW = 1};

/*  Used for the point location scheme of Mucke, Saias, and Zhu, to decide how large a random sample of tetrahedra to inspect. */
enum {SAMPLEFACTOR = 11};

/*  Arraypool                                                                  */
/*                                                                             */
/*  Each arraypool contains an array of pointers to a number of blocks.  Each  */
/*  block contains the same fixed number of objects.  Each index of the array  */
/*  addesses a particular object in the pool.  The most significant bits add-  */
/*  ress the index of the block containing the object. The less significant    */
/*  bits address this object within the block.                                 */
/*                                                                             */
/*  'objectbytes' is the size of one object in blocks; 'log2objectsperblock'   */
/*  is the base-2 logarithm of 'objectsperblock'; 'objects' counts the number  */
/*  of allocated objects; 'totalmemory' is the totoal memorypool in bytes.     */
typedef struct {
  int objectbytes;
  int objectsperblock;
  int log2objectsperblock;
  int toparraylen;
  char **toparray;
  long objects;
  unsigned long totalmemory;
} ArrayPool;

/*  fastlookup() -- A fast, unsafe operation. Return the pointer to the object */
/*    with a given index.  Note: The object's block must have been allocated, */
/*    i.e., by the function newindex(). */
#define fastlookup(pool, index) \
  (void *) ((pool)->toparray[(index) >> (pool)->log2objectsperblock] + \
            ((index) & ((pool)->objectsperblock - 1)) * (pool)->objectbytes)

/*  Memorypool                                                                 */
/*                                                                             */
/*  A type used to allocate memory.                                            */
/*                                                                             */
/*  firstblock is the first block of items. nowblock is the block from which   */
/*    items are currently being allocated. nextitem points to the next slab    */
/*    of free memory for an item. deaditemstack is the head of a linked list   */
/*    (stack) of deallocated items that can be recycled.  unallocateditems is  */
/*    the number of items that remain to be allocated from nowblock.           */
/*                                                                             */
/*  Traversal is the process of walking through the entire list of items, and  */
/*    is separate from allocation.  Note that a traversal will visit items on  */
/*    the "deaditemstack" stack as well as live items.  pathblock points to    */
/*    the block currently being traversed.  pathitem points to the next item   */
/*    to be traversed.  pathitemsleft is the number of items that remain to    */
/*    be traversed in pathblock.                                               */
/*                                                                             */
/*  itemwordtype is set to POINTER or FLOATINGPOINT, and is used to suggest    */
/*    what sort of word the record is primarily made up of.  alignbytes        */
/*    determines how new records should be aligned in memory.  itembytes and   */
/*    itemwords are the length of a record in bytes (after rounding up) and    */
/*    words.  itemsperblock is the number of items allocated at once in a      */
/*    single block.  items is the number of currently allocated items.         */
/*    maxitems is the maximum number of items that have been allocated at      */
/*    once; it is the current number of items plus the number of records kept  */
/*    on deaditemstack.                                                        */
typedef struct {
  void **firstblock, **nowblock;
  void *nextitem;
  void *deaditemstack;
  void **pathblock;
  void *pathitem;
  wordtype itemwordtype;
  int  alignbytes;
  int  itembytes, itemwords;
  int  itemsperblock;
  long items, maxitems;
  int  unallocateditems;
  int  pathitemsleft;
} MemoryPool;

/*  Queue                                                                      */
/*                                                                             */
/*  A 'queue' is a FIFO data structure.                                        */
typedef struct {
  MemoryPool *mp;
  void      **head, **tail;
  int         linkitembytes;
  int         linkitems; /*  Not counting 'head' and 'tail'. */
} Queue;

/*  A function: int cmp(const T &, const T &),  is said to realize a */
/*    linear order on the type T if there is a linear order <= on T such */
/*    that for all x and y in T satisfy the following relation: */
/*                  -1  if x < y. */
/*    comp(x, y) =   0  if x is equivalent to y. */
/*                  +1  if x > y. */
/*  A 'compfunc' is a pointer to a linear-order function. */
typedef int (*compfunc) (const void *, const void *);

/*  An array of items with automatically reallocation of memory.               */
/*                                                                             */
/*  'base' is the starting address of the array.  'itembytes' is the size of   */
/*    each item in byte.                                                       */
/*                                                                             */
/*  'items' is the number of items stored in list.  'maxitems' indicates how   */
/*    many items can be stored in this list. 'expandsize' is the increasing    */
/*    size (items) when the list is full.                                      */
/*                                                                             */
/*  The index of list always starts from zero, i.e., for a list L contains     */
/*    n elements, the first element is L[0], and the last element is L[n-1].   */
typedef struct {
  char *base;
  int  itembytes;
  int  items, maxitems, expandsize;
  compfunc comp;
} List;

/* This replaces tetgenmesh */
/* The tetrahedron data structure.  Fields of a tetrahedron contains:
   - a list of four adjoining tetrahedra;
   - a list of four vertices;
   - a list of four subfaces (optional, used for -p switch);
   - a list of user-defined floating-point attributes (optional);
   - a volume constraint (optional, used for -a switch);
   - an integer of element marker (optional, used for -n switch);
   - a pointer to a list of high-ordered nodes (optional, -o2 switch); */
typedef PetscReal **tetrahedron;
/* The shellface data structure.  Fields of a shellface contains:
   - a list of three adjoining subfaces;
   - a list of three vertices;
   - a list of two adjoining tetrahedra;
   - a list of three adjoining subsegments;
   - a pointer to a badface containing it (used for -q);
   - an area constraint (optional, used for -q);
   - an integer for boundary marker;
   - an integer for type: SHARPSEGMENT, NONSHARPSEGMENT, ...;
   - an integer for pbc group (optional, if in->pbcgrouplist exists); */
typedef PetscReal **shellface;
/* The point data structure.  It is actually an array of PetscReals:
   - x, y and z coordinates;
   - a list of user-defined point attributes (optional);
   - a list of PetscReals of a user-defined metric tensor (optional);
   - a pointer to a simplex (tet, tri, edge, or vertex);
   - a pointer to a parent (or duplicate) point;
   - a pointer to a tet in background mesh (optional);
   - a pointer to another pbc point (optional);
   - an integer for boundary marker;
   - an integer for verttype: INPUTVERTEX, FREEVERTEX, ...; */
typedef PetscReal  *point;
/* /////////////////////////////////////////////////////////////////////////// */
/*                                                                             */
/*  Fast lookup tables for mesh manipulation primitives.                       */
/*                                                                             */
/*  Mesh manipulation primitives (given below) are basic operations on mesh    */
/*  data structures. They answer basic queries on mesh handles, such as "what  */
/*  is the origin (or destination, or apex) of the face?", "what is the next   */
/*  (or previous) edge in the edge ring?", and "what is the next face in the   */
/*  face ring?", and so on.                                                    */
/*                                                                             */
/*  The implementation of teste basic queries can take advangtage of the fact  */
/*  that the mesh data structures additionally store geometric informations.   */
/*  For example, we have ordered the 4 vertices (from 0 to 3) and the 4 faces  */
/*  (from 0 to 3) of a tetrahedron,  and for each face of the tetrahedron, a   */
/*  sequence of vertices has stipulated,  therefore the origin of any face of  */
/*  the tetrahedron can be quickly determined by a table 'locver2org', which   */
/*  takes the index of the face and the edge version as inputs.  A list of     */
/*  fast lookup tables are defined below. They're just like global variables.  */
/*  These tables are initialized at the runtime.                               */
/*                                                                             */
/* /////////////////////////////////////////////////////////////////////////// */
/*  Table 've' takes an edge version as input, returns the next edge version in the same edge ring. */
/*    For enext() primitive, uses 'ver' as the index. */
static int ve[6] = {2, 5, 4, 1, 0, 3};

/*  Tables 'vo', 'vd' and 'va' take an edge version, return the positions of */
/*    the origin, destination and apex in the triangle. */
/*    For org(), dest() and apex() primitives, uses 'ver' as the index. */
static int vo[6] = { 0, 1, 1, 2, 2, 0 };
static int vd[6] = { 1, 0, 2, 1, 0, 2 };
static int va[6] = { 2, 2, 0, 0, 1, 1 };

/*  The following tables are for tetrahedron primitives (operate on trifaces). */
/*    For org(), dest() and apex() primitives, uses 'loc' as the first index and 'ver' as the second index. */
static int locver2org[4][6]  = {{0, 1, 1, 2, 2, 0},
                                {0, 3, 3, 1, 1, 0},
                                {1, 3, 3, 2, 2, 1},
                                {2, 3, 3, 0, 0, 2}};
static int locver2dest[4][6] = {{1, 0, 2, 1, 0, 2},
                                {3, 0, 1, 3, 0, 1},
                                {3, 1, 2, 3, 1, 2},
                                {3, 2, 0, 3, 2, 0}};
static int locver2apex[4][6] = {{2, 2, 0, 0, 1, 1},
                                {1, 1, 0, 0, 3, 3},
                                {2, 2, 1, 1, 3, 3},
                                {0, 0, 2, 2, 3, 3}};

/*  For oppo() primitives, uses 'loc' as the index. */
static int loc2oppo[4] = {3, 2, 0, 1};

/*  For fnext() primitives, uses 'loc' as the first index and 'ver' as */
/*    the second index,  returns an array containing a new 'loc' and a */
/*    new 'ver'. Note: Only valid for 'ver' equals one of {0, 2, 4}. */
static int locver2nextf[4][6][2] = {{{1, 5}, {-1, -1}, {2, 5}, {-1, -1}, {3, 5}, {-1, -1}},
                                    {{3, 3}, {-1, -1}, {2, 1}, {-1, -1}, {0, 1}, {-1, -1}},
                                    {{1, 3}, {-1, -1}, {3, 1}, {-1, -1}, {0, 3}, {-1, -1}},
                                    {{2, 3}, {-1, -1}, {1, 1}, {-1, -1}, {0, 5}, {-1, -1}}};

/*  The edge number (from 0 to 5) of a tet is defined as follows: */
/*    0 - (v0, v1), 1 - (v1, v2), 2 - (v2, v0) */
/*    3 - (v3, v0), 4 - (v3, v1), 5 - (v3, v2). */
static int locver2edge[4][6] = {{0, 0, 1, 1, 2, 2},
                                {3, 3, 4, 4, 0, 0},
                                {4, 4, 5, 5, 1, 1},
                                {5, 5, 3, 3, 2, 2}};
static int edge2locver[6][2] = {{0, 0},  /*  0  v0 -> v1 (a -> b) */
                                {0, 2},  /*  1  v1 -> v2 (b -> c) */
                                {0, 4},  /*  2  v2 -> v0 (c -> a) */
                                {1, 0},  /*  3  v0 -> v3 (a -> d) */
                                {1, 2},  /*  4  v1 -> v3 (b -> d */
                                {2, 2}}; /*  5  v2 -> v3 (c -> d); */

/*  The map from a given face ('loc') to the other three faces in the tet. */
/*    and the map from a given face's edge ('loc', 'ver') to other two */
/*    faces in the tet opposite to this edge. (used in speeding the Bowyer- */
/*    Watson cavity construction). */
static int locpivot[4][3] = {{1, 2, 3},
                             {0, 2, 3},
                             {0, 1, 3},
                             {0, 1, 2}};
static int locverpivot[4][6][2] = {{{2, 3}, {2, 3}, {1, 3}, {1, 3}, {1, 2}, {1, 2}},
                                   {{0, 2}, {0, 2}, {0, 3}, {0, 3}, {2, 3}, {2, 3}},
                                   {{0, 3}, {0, 3}, {0, 1}, {0, 1}, {1, 3}, {1, 3}},
                                   {{0, 1}, {0, 1}, {0, 2}, {0, 2}, {1, 2}, {1, 2}}};

/*  For enumerating three edges of a triangle. */
/* static int plus1mod3[3]  = {1, 2, 0}; */
/* static int minus1mod3[3] = {2, 0, 1}; */

/*  A 'triface' represents a face of a tetrahedron and an oriented edge of     */
/*  the face simultaneously.  It has a pointer 'tet' to a tetrahedron, an      */
/*  integer 'loc' (range from 0 to 3) as the face index, and an integer 'ver'  */
/*  (range from 0 to 5) as the edge version. A face of the tetrahedron can be  */
/*  uniquely determined by the pair (tet, loc), and an oriented edge of this   */
/*  face can be uniquly determined by the triple (tet, loc, ver).  Therefore,  */
/*  different usages of one triface are possible.  If we only use the pair     */
/*  (tet, loc), it refers to a face, and if we add the 'ver' additionally to   */
/*  the pair, it is an oriented edge of this face.                             */
typedef struct {
  tetrahedron* tet;
  int loc, ver;
} triface;

/*  A 'face' represents a subface and an oriented edge of it simultaneously.   */
/*  It has a pointer 'sh' to a subface, an integer 'shver'(range from 0 to 5)  */
/*  as the edge version.  The pair (sh, shver) determines a unique oriented    */
/*  edge of this subface.  A 'face' is also used to represent a subsegment,    */
/*  in this case, 'sh' points to the subsegment, and 'shver' indicates the     */
/*  one of two orientations of this subsegment, hence, it only can be 0 or 1.  */
typedef struct {
  shellface *sh;
  int shver;
} face;

/*  A multiple usages structure. Despite of its name, a 'badface' can be used  */
/*  to represent the following objects:                                        */
/*    - a face of a tetrahedron which is (possibly) non-Delaunay;              */
/*    - an encroached subsegment or subface;                                   */
/*    - a bad-quality tetrahedron, i.e, has too large radius-edge ratio;       */
/*    - a sliver, i.e., has good radius-edge ratio but nearly zero volume;     */
/*    - a degenerate tetrahedron (see routine checkdegetet()).                 */
/*    - a recently flipped face (saved for undoing the flip later).            */
/*                                                                             */
/*  It has the following fields:  'tt' holds a tetrahedron; 'ss' holds a sub-  */
/*  segment or subface; 'cent' is the circumcent of 'tt' or 'ss', 'key' is a   */
/*  special value depending on the use, it can be either the square of the     */
/*  radius-edge ratio of 'tt' or the flipped type of 'tt';  'forg', 'fdest',   */
/*  'fapex', and 'foppo' are vertices saved for checking the object in 'tt'    */
/*  or 'ss' is still the same when it was stored; 'noppo' is the fifth vertex  */
/*  of a degenerate point set.  'previtem' and 'nextitem' implement a double   */
/*  link for managing many basfaces.                                           */
typedef struct _s_badface {
    triface tt;
    face ss;
    PetscReal key;
    PetscReal cent[3];
    point forg, fdest, fapex, foppo;
    point noppo;
    struct _s_badface *previtem, *nextitem;
} badface;

/*  A pbcdata stores data of a periodic boundary condition defined on a pair   */
/*  of facets or segments. Let f1 and f2 define a pbcgroup. 'fmark' saves the  */
/*  facet markers of f1 and f2;  'ss' contains two subfaces belong to f1 and   */
/*  f2, respectively.  Let s1 and s2 define a segment pbcgroup. 'segid' are    */
/*  the segment ids of s1 and s2; 'ss' contains two segments belong to s1 and  */
/*  s2, respectively. 'transmat' are two transformation matrices. transmat[0]  */
/*  transforms a point of f1 (or s1) into a point of f2 (or s2),  transmat[1]  */
/*  does the inverse.                                                          */
typedef struct {
  int fmark[2];
  int segid[2];
  face ss[2];
  PetscReal transmat[2][4][4];
} pbcdata;

typedef struct {
  /*  Pointer to the input data (a set of nodes, a PLC, or a mesh). */
  PLC *in;

  /*  Pointer to the options (and filenames). */
  TetGenOpts *b;

  /*  Pointer to a background mesh (contains size specification map). */
  /*  tetgenmesh *bgm; */

  /*  Variables used to allocate and access memory for tetrahedra, subfaces */
  /*    subsegments, points, encroached subfaces, encroached subsegments, */
  /*    bad-quality tetrahedra, and so on. */
  MemoryPool *tetrahedrons;
  MemoryPool *subfaces;
  MemoryPool *subsegs;
  MemoryPool *points;
  MemoryPool *badsubsegs;
  MemoryPool *badsubfaces;
  MemoryPool *badtetrahedrons;
  MemoryPool *tet2segpool, *tet2subpool;

  /*  Pointer to the 'tetrahedron' that occupies all of "outer space". */
  tetrahedron *dummytet;
  tetrahedron *dummytetbase; /*  Keep base address so we can free it later. */

  /*  Pointer to the omnipresent subface.  Referenced by any tetrahedron, */
  /*    or subface that isn't connected to a subface at that location. */
  shellface *dummysh;
  shellface *dummyshbase;    /*  Keep base address so we can free it later. */

  /*  Entry to find the binary tree nodes (-u option). */
  ArrayPool *btreenode_list;
  /*  The maximum size of a btree node (number after -u option) is */
  int max_btreenode_size; /*  <= b->max_btreenode_size. */
  /*  The maximum btree depth (for bookkeeping). */
  int max_btree_depth;

  /*  Arrays used by Bowyer-Watson algorithm. */
  ArrayPool *cavetetlist, *cavebdrylist, *caveoldtetlist;
  ArrayPool *caveshlist, *caveshbdlist;
  /*  Stacks used by the boundary recovery algorithm. */
  ArrayPool *subsegstack, *subfacstack;

  /*  Two handles used in constrained facet recovery. */
  triface firsttopface, firstbotface;

  /*  An array for registering elementary flips. */
  ArrayPool *elemfliplist;

  /*  An array of fixed edges for facet recovering by flips. */
  ArrayPool *fixededgelist;

  /*  A point above the plane in which the facet currently being used lies. */
  /*    It is used as a reference point for TetGenOrient3D(). */
  point *facetabovepointarray, abovepoint, dummypoint;

  /*  Array (size = numberoftetrahedra * 6) for storing high-order nodes of */
  /*    tetrahedra (only used when -o2 switch is selected). */
  point *highordertable;

  /*  Arrays for storing and searching pbc data. 'subpbcgrouptable', (size */
  /*    is numberofpbcgroups) for pbcgroup of subfaces. 'segpbcgrouptable', */
  /*    a list for pbcgroup of segments. Because a segment can have several */
  /*    pbcgroup incident on it, its size is unknown on input, it will be */
  /*    found in 'createsegpbcgrouptable()'. */
  pbcdata *subpbcgrouptable;
  List *segpbcgrouptable;
  /*  A map for searching the pbcgroups of a given segment. 'idx2segpglist' */
  /*    (size = number of input segments + 1), and 'segpglist'. */
  int *idx2segpglist, *segpglist;

  /*  Queues that maintain the bad (badly-shaped or too large) tetrahedra. */
  /*    The tails are pointers to the pointers that have to be filled in to */
  /*    enqueue an item.  The queues are ordered from 63 (highest priority) */
  /*    to 0 (lowest priority). */
  badface *subquefront[3], **subquetail[3];
  badface *tetquefront[64], *tetquetail[64];
  int nextnonemptyq[64];
  int firstnonemptyq, recentq;

  /*  Pointer to a recently visited tetrahedron. Improves point location */
  /*    if proximate points are inserted sequentially. */
  triface recenttet;

  PetscReal xmax, xmin, ymax, ymin, zmax, zmin;         /*  Bounding box of points. */
  PetscReal longest;                          /*  The longest possible edge length. */
  PetscReal lengthlimit;                     /*  The limiting length of a new edge. */
  long hullsize;                           /*  Number of faces of convex hull. */
  long insegments;                               /*  Number of input segments. */
  long meshedges;                             /*  Number of output mesh edges. */
  int steinerleft;                  /*  Number of Steiner points not yet used. */
  int sizeoftensor;                     /*  Number of PetscReals per metric tensor. */
  int pointmtrindex;           /*  Index to find the metric tensor of a point. */
  int point2simindex;         /*  Index to find a simplex adjacent to a point. */
  int pointmarkindex;            /*  Index to find boundary marker of a point. */
  int point2pbcptindex;              /*  Index to find a pbc point to a point. */
  int highorderindex;    /*  Index to find extra nodes for highorder elements. */
  int elemattribindex;          /*  Index to find attributes of a tetrahedron. */
  int volumeboundindex;       /*  Index to find volume bound of a tetrahedron. */
  int elemmarkerindex;              /*  Index to find marker of a tetrahedron. */
  int shmarkindex;             /*  Index to find boundary marker of a subface. */
  int areaboundindex;               /*  Index to find area bound of a subface. */
  int checksubfaces;                   /*  Are there subfaces in the mesh yet? */
  int checksubsegs;                     /*  Are there subsegs in the mesh yet? */
  int checkpbcs;                   /*  Are there periodic boundary conditions? */
  int varconstraint;     /*  Are there variant (node, seg, facet) constraints? */
  int nonconvex;                               /*  Is current mesh non-convex? */
  int dupverts;                             /*  Are there duplicated vertices? */
  int unuverts;                                 /*  Are there unused vertices? */
  int relverts;                          /*  The number of relocated vertices. */
  int suprelverts;            /*  The number of suppressed relocated vertices. */
  int collapverts;             /*  The number of collapsed relocated vertices. */
  int unsupverts;                     /*  The number of unsuppressed vertices. */
  int smoothsegverts;                     /*  The number of smoothed vertices. */
  int jettisoninverts;            /*  The number of jettisoned input vertices. */
  long samples;               /*  Number of random samples for point location. */
  unsigned long randomseed;                    /*  Current random number seed. */
  PetscReal macheps;                                       /*  The machine epsilon. */
  PetscReal cosmaxdihed, cosmindihed;    /*  The cosine values of max/min dihedral. */
  PetscReal minfaceang, minfacetdihed;     /*  The minimum input (dihedral) angles. */
  int maxcavfaces, maxcavverts;            /*  The size of the largest cavity. */
  PetscBool b_steinerflag;

  /*  Algorithm statistical counters. */
  long ptloc_count, ptloc_max_count;
  long orient3dcount;
  long inspherecount, insphere_sos_count;
  long flip14count, flip26count, flipn2ncount;
  long flip22count;
  long inserthullcount;
  long maxbowatcavsize, totalbowatcavsize, totaldeadtets;
  long across_face_count, across_edge_count, across_max_count;
  long maxcavsize, maxregionsize;
  long ndelaunayedgecount, cavityexpcount;
  long opt_tet_peels, opt_face_flips, opt_edge_flips;

  long abovecount;                     /*  Number of abovepoints calculation. */
  long bowatvolcount, bowatsubcount, bowatsegcount;       /*  Bowyer-Watsons. */
  long updvolcount, updsubcount, updsegcount;   /*  Bow-Wat cavities updates. */
  long failvolcount, failsubcount, failsegcount;           /*  Bow-Wat fails. */
  long outbowatcircumcount;    /*  Number of circumcenters outside Bowat-cav. */
  long r1count, r2count, r3count;        /*  Numbers of edge splitting rules. */
  long cdtenforcesegpts;                /*  Number of CDT enforcement points. */
  long rejsegpts, rejsubpts, rejtetpts;        /*  Number of rejected points. */
  long optcount[10];            /*  Numbers of various optimizing operations. */
  long flip23s, flip32s, flip22s, flip44s;     /*  Number of flips performed. */
} TetGenMesh;

/*================================= End Converted TetGen Objects =================================*/

/* Forward Declarations */
static PetscErrorCode MemoryPoolCreate(int, int, wordtype, int, MemoryPool **);
static PetscErrorCode MemoryPoolAlloc(MemoryPool *, void **);
static PetscErrorCode MemoryPoolDealloc(MemoryPool *, void *);
static PetscErrorCode MemoryPoolDestroy(MemoryPool **);
static PetscErrorCode ArrayPoolDestroy(ArrayPool **);
static PetscErrorCode ListDestroy(List **);
extern PetscErrorCode TetGenMeshPointTraverse(TetGenMesh *, point *);
extern PetscErrorCode TetGenMeshShellFaceTraverse(TetGenMesh *, MemoryPool *, shellface **);
extern PetscErrorCode TetGenMeshTetrahedronTraverse(TetGenMesh *, tetrahedron **);
extern PetscErrorCode TetGenMeshGetNextSFace(TetGenMesh *, face *, face *);
extern PetscErrorCode TetGenMeshSplitSubEdge_arraypool(TetGenMesh *, point, face *, ArrayPool *, ArrayPool *);
extern PetscErrorCode TetGenMeshCheck4FixedEdge(TetGenMesh *, point, point, PetscBool *);
extern PetscErrorCode TetGenMeshGetFacetAbovePoint(TetGenMesh *, face *);
extern PetscErrorCode TetGenMeshSInsertVertex(TetGenMesh *, point, face *, face *, PetscBool, PetscBool, locateresult *);
extern PetscErrorCode TetGenMeshInsertVertexBW(TetGenMesh *, point, triface *, PetscBool, PetscBool, PetscBool, PetscBool, locateresult *);
extern PetscErrorCode TetGenMeshJettisonNodes(TetGenMesh *);
extern PetscErrorCode TetGenMeshTallEncSegs(TetGenMesh *, point, int, List **, PetscBool *);
extern PetscErrorCode TetGenMeshTallEncSubs(TetGenMesh *, point, int, List **, PetscBool *);
extern PetscErrorCode TetGenMeshCheckTet4Opt(TetGenMesh *, triface *, PetscBool, PetscBool *);
extern PetscErrorCode TetGenMeshCheckTet4BadQual(TetGenMesh *, triface *, PetscBool, PetscBool *);
extern PetscErrorCode TetGenMeshCheckSeg4Encroach(TetGenMesh *, face *, point, point *, PetscBool, PetscBool *);
extern PetscErrorCode TetGenMeshCheckSub4Encroach(TetGenMesh *, face *, point, PetscBool, PetscBool *);

/*=========================== Start Converted TetGen Inline Functions ============================*/
/*  Some macros for convenience */
#define Div2  >> 1
#define Mod2  & 01
/*  NOTE: These bit operators should only be used in macros below. */
/*  Get orient(Range from 0 to 2) from face version(Range from 0 to 5). */
#define Orient(V)   ((V) Div2)
/*  Determine edge ring(0 or 1) from face version(Range from 0 to 5). */
#define EdgeRing(V) ((V) Mod2)

/*** Begin of primitives for points ***/
PETSC_STATIC_INLINE int pointmark(TetGenMesh *m, point pt) {
  return ((int *) (pt))[m->pointmarkindex];
}

PETSC_STATIC_INLINE void setpointmark(TetGenMesh *m, point pt, int value) {
  ((int *) (pt))[m->pointmarkindex] = value;
}

/*  These two primitives set and read the type of the point. */
/*  The last significant bit of this integer is used by pinfect/puninfect. */
PETSC_STATIC_INLINE verttype pointtype(TetGenMesh *m, point pt) {
  return (verttype) (((int *) (pt))[m->pointmarkindex + 1] >> (int) 1);
}

PETSC_STATIC_INLINE void setpointtype(TetGenMesh *m, point pt, verttype value) {
  ((int *) (pt))[m->pointmarkindex + 1] = ((int) value << 1) + (((int *) (pt))[m->pointmarkindex + 1] & (int) 1);
}

/*  pinfect(), puninfect(), pinfected() -- primitives to flag or unflag a point */
/*    The last bit of the integer '[pointindex+1]' is flaged. */
PETSC_STATIC_INLINE void pinfect(TetGenMesh *m, point pt) {
  ((int *) (pt))[m->pointmarkindex + 1] |= (int) 1;
}

PETSC_STATIC_INLINE void puninfect(TetGenMesh *m, point pt) {
  ((int *) (pt))[m->pointmarkindex + 1] &= ~(int) 1;
}

PETSC_STATIC_INLINE PetscBool pinfected(TetGenMesh *m, point pt) {
  return (((int *) (pt))[m->pointmarkindex + 1] & (int) 1) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

/*  These following primitives set and read a pointer to a tetrahedron */
/*    a subface/subsegment, a point, or a tet of background mesh. */
PETSC_STATIC_INLINE tetrahedron point2tet(TetGenMesh *m, point pt) {
  return ((tetrahedron *) (pt))[m->point2simindex];
}

PETSC_STATIC_INLINE void setpoint2tet(TetGenMesh *m, point pt, tetrahedron value) {
  ((tetrahedron *) (pt))[m->point2simindex] = value;
}

PETSC_STATIC_INLINE shellface point2sh(TetGenMesh *m, point pt) {
  return (shellface) ((tetrahedron *) (pt))[m->point2simindex + 1];
}

PETSC_STATIC_INLINE void setpoint2sh(TetGenMesh *m, point pt, shellface value) {
  ((tetrahedron *) (pt))[m->point2simindex + 1] = (tetrahedron) value;
}

PETSC_STATIC_INLINE shellface point2seg(TetGenMesh *m, point pt) {
  return (shellface) ((tetrahedron *) (pt))[m->point2simindex + 2];
}

PETSC_STATIC_INLINE void setpoint2seg(TetGenMesh *m, point pt, shellface value) {
  ((tetrahedron *) (pt))[m->point2simindex + 2] = (tetrahedron) value;
}

PETSC_STATIC_INLINE point point2ppt(TetGenMesh *m, point pt) {
  return (point) ((tetrahedron *) (pt))[m->point2simindex + 3];
}

PETSC_STATIC_INLINE void setpoint2ppt(TetGenMesh *m, point pt, point value) {
  ((tetrahedron *) (pt))[m->point2simindex + 3] = (tetrahedron) value;
}

PETSC_STATIC_INLINE tetrahedron point2bgmtet(TetGenMesh *m, point pt) {
  return ((tetrahedron *) (pt))[m->point2simindex + 4];
}

PETSC_STATIC_INLINE void setpoint2bgmtet(TetGenMesh *m, point pt, tetrahedron value) {
  ((tetrahedron *) (pt))[m->point2simindex + 4] = value;
}

/*  These primitives set and read a pointer to its pbc point. */
PETSC_STATIC_INLINE point point2pbcpt(TetGenMesh *m, point pt) {
  return (point) ((tetrahedron *) (pt))[m->point2pbcptindex];
}

PETSC_STATIC_INLINE void setpoint2pbcpt(TetGenMesh *m, point pt, point value) {
  ((tetrahedron *) (pt))[m->point2pbcptindex] = (tetrahedron) value;
}
/*** End of primitives for points ***/

/*** Begin of primitives for tetrahedra ***/
/*  Each tetrahedron contains four pointers to its neighboring tetrahedra, */
/*    with face indices.  To save memory, both information are kept in a */
/*    single pointer. To make this possible, all tetrahedra are aligned to */
/*    eight-byte boundaries, so that the last three bits of each pointer are */
/*    zeros. A face index (in the range 0 to 3) is compressed into the last */
/*    two bits of each pointer by the function 'encode()'.  The function */
/*    'decode()' decodes a pointer, extracting a face index and a pointer to */
/*    the beginning of a tetrahedron. */
PETSC_STATIC_INLINE void decode(tetrahedron ptr, triface *t) {
  t->loc = (int) ((PETSC_UINTPTR_T) (ptr) & (PETSC_UINTPTR_T) 3);
  t->tet = (tetrahedron *) ((PETSC_UINTPTR_T) (ptr) & ~(PETSC_UINTPTR_T) 7);
}

PETSC_STATIC_INLINE tetrahedron encode(triface *t) {
  return (tetrahedron) ((PETSC_UINTPTR_T) t->tet | (PETSC_UINTPTR_T) t->loc);
}

/*  sym() finds the abutting tetrahedron on the same face. */
PETSC_STATIC_INLINE void sym(triface *t1, triface *t2) {
  tetrahedron ptr = t1->tet[t1->loc];
  decode(ptr, t2);
}

PETSC_STATIC_INLINE void symself(triface *t) {
  tetrahedron ptr = t->tet[t->loc];
  decode(ptr, t);
}

/*  Bond two tetrahedra together at their faces. */
PETSC_STATIC_INLINE void bond(TetGenMesh *m, triface *t1, triface *t2) {
  t1->tet[t1->loc] = encode(t2);
  t2->tet[t2->loc] = encode(t1);
}

/*  Dissolve a bond (from one side).  Note that the other tetrahedron will */
/*    still think it is connected to this tetrahedron.  Usually, however, */
/*    the other tetrahedron is being deleted entirely, or bonded to another */
/*    tetrahedron, so it doesn't matter. */
PETSC_STATIC_INLINE void dissolve(TetGenMesh *m, triface *t) {
  t->tet[t->loc] = (tetrahedron) m->dummytet;
}

/*  These primitives determine or set the origin, destination, apex or */
/*    opposition of a tetrahedron with respect to 'loc' and 'ver'. */
PETSC_STATIC_INLINE point org(triface *t) {
  return (point) t->tet[locver2org[t->loc][t->ver] + 4];
}

PETSC_STATIC_INLINE point dest(triface *t) {
  return (point) t->tet[locver2dest[t->loc][t->ver] + 4];
}

PETSC_STATIC_INLINE point apex(triface *t) {
  return (point) t->tet[locver2apex[t->loc][t->ver] + 4];
}

PETSC_STATIC_INLINE point oppo(triface *t) {
  return (point) t->tet[loc2oppo[t->loc] + 4];
}

PETSC_STATIC_INLINE void setorg(triface *t, point pointptr) {
  t->tet[locver2org[t->loc][t->ver] + 4] = (tetrahedron) pointptr;
}

PETSC_STATIC_INLINE void setdest(triface *t, point pointptr) {
  t->tet[locver2dest[t->loc][t->ver] + 4] = (tetrahedron) pointptr;
}

PETSC_STATIC_INLINE void setapex(triface *t, point pointptr) {
  t->tet[locver2apex[t->loc][t->ver] + 4] = (tetrahedron) pointptr;
}

PETSC_STATIC_INLINE void setoppo(triface *t, point pointptr) {
  t->tet[loc2oppo[t->loc] + 4] = (tetrahedron) pointptr;
}

/*  These primitives were drived from Mucke's triangle-edge data structure */
/*    to change face-edge relation in a tetrahedron (esym, enext and enext2) */
/*    or between two tetrahedra (fnext). */

/*  If e0 = e(i, j), e1 = e(j, i), that is e0 and e1 are the two directions */
/*    of the same undirected edge of a face. e0.sym() = e1 and vice versa. */
PETSC_STATIC_INLINE void esym(triface *t1, triface *t2) {
  t2->tet = t1->tet;
  t2->loc = t1->loc;
  t2->ver = t1->ver + (EdgeRing(t1->ver) ? -1 : 1);
}

PETSC_STATIC_INLINE void esymself(triface *t) {
  t->ver += (EdgeRing(t->ver) ? -1 : 1);
}

/*  If e0 and e1 are both in the same edge ring of a face, e1 = e0.enext(). */
PETSC_STATIC_INLINE void enext(triface *t1, triface *t2) {
  t2->tet = t1->tet;
  t2->loc = t1->loc;
  t2->ver = ve[t1->ver];
}

PETSC_STATIC_INLINE void enextself(triface *t) {
  t->ver = ve[t->ver];
}

/*  enext2() is equal to e2 = e0.enext().enext() */
PETSC_STATIC_INLINE void enext2(triface *t1, triface *t2) {
  t2->tet = t1->tet;
  t2->loc = t1->loc;
  t2->ver = ve[ve[t1->ver]];
}

PETSC_STATIC_INLINE void enext2self(triface *t) {
  t->ver = ve[ve[t->ver]];
}

/*  If f0 and f1 are both in the same face ring of a face, f1 = f0.fnext(). */
/*    If f1 exists, return true. Otherwise, return false, i.e., f0 is a boundary or hull face. */
PETSC_STATIC_INLINE PetscBool fnext(TetGenMesh *m, triface *t1, triface *t2)
{
  /*  Get the next face. */
  t2->loc = locver2nextf[t1->loc][t1->ver][0];
  /*  Is the next face in the same tet? */
  if (t2->loc != -1) {
    /*  It's in the same tet. Get the edge version. */
    t2->ver = locver2nextf[t1->loc][t1->ver][1];
    t2->tet = t1->tet;
  } else {
    /*  The next face is in the neigbhour of 't1'. */
    sym(t1, t2);
    if (t2->tet != m->dummytet) {
      /*  Find the corresponding edge in t2. */
      point torg;
      int tloc, tver, i;
      t2->ver = 0;
      torg = org(t1);
      for (i = 0; (i < 3) && (org(t2) != torg); i++) {
        enextself(t2);
      }
      /*  Go to the next face in t2. */
      tloc = t2->loc;
      tver = t2->ver;
      t2->loc = locver2nextf[tloc][tver][0];
      t2->ver = locver2nextf[tloc][tver][1];
    }
  }
  return t2->tet != m->dummytet ? PETSC_TRUE : PETSC_FALSE;
}

PETSC_STATIC_INLINE PetscBool fnextself(TetGenMesh *m, triface *t1)
{
  triface t2 = {PETSC_NULL, 0, 0};

  /*  Get the next face. */
  t2.loc = locver2nextf[t1->loc][t1->ver][0];
  /*  Is the next face in the same tet? */
  if (t2.loc != -1) {
    /*  It's in the same tet. Get the edge version. */
    t2.ver = locver2nextf[t1->loc][t1->ver][1];
    t1->loc = t2.loc;
    t1->ver = t2.ver;
  } else {
    /*  The next face is in the neigbhour of 't1'. */
    sym(t1, &t2);
    if (t2.tet != m->dummytet) {
      /*  Find the corresponding edge in t2. */
      point torg;
      int i;
      t2.ver = 0;
      torg = org(t1);
      for (i = 0; (i < 3) && (org(&t2) != torg); i++) {
        enextself(&t2);
      }
      t1->loc = locver2nextf[t2.loc][t2.ver][0];
      t1->ver = locver2nextf[t2.loc][t2.ver][1];
      t1->tet = t2.tet;
    }
  }
  return t2.tet != m->dummytet ? PETSC_TRUE : PETSC_FALSE;
}

/*  Given a face t1, find the face f2 in the adjacent tet. If t2 is not */
/*    a dummytet, then t1 and t2 refer to the same edge. Moreover, t2's */
/*    edge must be in 0th edge ring, e.g., t2->ver is one of {0, 2, 4}. */
/*    No matter what edge version t1 is. */

PETSC_STATIC_INLINE void symedge(TetGenMesh *m, triface *t1, triface *t2)
{
  decode(t1->tet[t1->loc], t2);
  if (t2->tet != m->dummytet) {
    /*  Search the edge of t1 in t2. */
    point tapex = apex(t1);
    if ((point) (t2->tet[locver2apex[t2->loc][0] + 4]) == tapex) {
      t2->ver = 0;
    } else if ((point) (t2->tet[locver2apex[t2->loc][2] + 4]) == tapex) {
      t2->ver = 2;
    } else {
      /* assert((point) (t2->tet[locver2apex[t2->loc][4] + 4]) == tapex); */
      t2->ver = 4;
    }
  }
}

PETSC_STATIC_INLINE void symedgeself(TetGenMesh *m, triface *t)
{
  tetrahedron ptr;
  point tapex;

  ptr = t->tet[t->loc];
  tapex = apex(t);
  decode(ptr, t);
  if (t->tet != m->dummytet) {
    /*  Search the edge of t1 in t2. */
    if ((point) (t->tet[locver2apex[t->loc][0] + 4]) == tapex) {
      t->ver = 0;
    } else if ((point) (t->tet[locver2apex[t->loc][2] + 4]) == tapex) {
      t->ver = 2;
    } else {
      /* assert((point) (t->tet[locver2apex[t->loc][4] + 4]) == tapex); */
      t->ver = 4;
    }
  }
}

/*  Given a face t1, find the next face t2 in the face ring, t1 and t2 */
/*    are in two different tetrahedra. If the next face is a hull face, t2 is dummytet. */
PETSC_STATIC_INLINE void tfnext(TetGenMesh *m, triface *t1, triface *t2)
{
  int *iptr;

  if ((t1->ver & 1) == 0) {
    t2->tet = t1->tet;
    iptr = locver2nextf[t1->loc][t1->ver];
    t2->loc = iptr[0];
    t2->ver = iptr[1];
    symedgeself(m, t2);  /*  t2->tet may be dummytet. */
  } else {
    symedge(m, t1, t2);
    if (t2->tet != m->dummytet) {
      iptr = locver2nextf[t2->loc][t2->ver];
      t2->loc = iptr[0];
      t2->ver = iptr[1];
    }
  }
}

PETSC_STATIC_INLINE void tfnextself(TetGenMesh *m, triface *t)
{
  int *iptr;

  if ((t->ver & 1) == 0) {
    iptr = locver2nextf[t->loc][t->ver];
    t->loc = iptr[0];
    t->ver = iptr[1];
    symedgeself(m, t); /*  t->tet may be dummytet. */
  } else {
    symedgeself(m, t);
    if (t->tet != m->dummytet) {
      iptr = locver2nextf[t->loc][t->ver];
      t->loc = iptr[0];
      t->ver = iptr[1];
    }
  }
}

/*  enextfnext() and enext2fnext() are combination primitives of enext(), enext2() and fnext(). */
PETSC_STATIC_INLINE void enextfnext(TetGenMesh *m, triface *t1, triface *t2) {
  enext(t1, t2);
  fnextself(m, t2);
}

PETSC_STATIC_INLINE void enextfnextself(TetGenMesh *m, triface *t) {
  enextself(t);
  fnextself(m, t);
}

PETSC_STATIC_INLINE void enext2fnext(TetGenMesh *m, triface *t1, triface *t2) {
  enext2(t1, t2);
  fnextself(m, t2);
}

PETSC_STATIC_INLINE void enext2fnextself(TetGenMesh *m, triface *t) {
  enext2self(t);
  fnextself(m, t);
}

/*  Check or set a tetrahedron's attributes. */
PETSC_STATIC_INLINE PetscReal elemattribute(TetGenMesh *m, tetrahedron *ptr, int attnum) {
  return ((PetscReal *) (ptr))[m->elemattribindex + attnum];
}

PETSC_STATIC_INLINE void setelemattribute(TetGenMesh *m, tetrahedron *ptr, int attnum, PetscReal value){
  ((PetscReal *) (ptr))[m->elemattribindex + attnum] = value;
}

/*  Check or set a tetrahedron's maximum volume bound. */
PETSC_STATIC_INLINE PetscReal volumebound(TetGenMesh *m, tetrahedron *ptr) {
  return ((PetscReal *) (ptr))[m->volumeboundindex];
}

PETSC_STATIC_INLINE void setvolumebound(TetGenMesh *m, tetrahedron* ptr, PetscReal value) {
  ((PetscReal *) (ptr))[m->volumeboundindex] = value;
}

/*  Check or set a tetrahedron's marker. */
PETSC_STATIC_INLINE int getelemmarker(TetGenMesh *m, tetrahedron* ptr) {
  return ((int *) (ptr))[m->elemmarkerindex];
}

PETSC_STATIC_INLINE void setelemmarker(TetGenMesh *m, tetrahedron* ptr, int value) {
  ((int *) (ptr))[m->elemmarkerindex] = value;
}

/*  infect(), infected(), uninfect() -- primitives to flag or unflag a */
/*    tetrahedron. The last bit of the element marker is flagged (1) */
/*    or unflagged (0). */
PETSC_STATIC_INLINE void infect(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] |= (int) 1;
}

PETSC_STATIC_INLINE void uninfect(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] &= ~(int) 1;
}

/*  Test a tetrahedron for viral infection. */
PETSC_STATIC_INLINE PetscBool infected(TetGenMesh *m, triface *t) {
  return (((int *) (t->tet))[m->elemmarkerindex] & (int) 1) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

/*  marktest(), marktested(), unmarktest() -- primitives to flag or unflag a */
/*    tetrahedron.  The last second bit of the element marker is marked (1) */
/*    or unmarked (0). */
/*  One needs them in forming Bowyer-Watson cavity, to mark a tetrahedron if */
/*    it has been checked (for Delaunay case) so later check can be avoided. */
PETSC_STATIC_INLINE void marktest(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] |= (int) 2;
}

PETSC_STATIC_INLINE void unmarktest(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] &= ~(int) 2;
}

PETSC_STATIC_INLINE PetscBool marktested(TetGenMesh *m, triface *t) {
  return (((int *) (t->tet))[m->elemmarkerindex] & (int) 2) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

/*  markface(), unmarkface(), facemarked() -- primitives to flag or unflag a */
/*    face of a tetrahedron.  From the last 3rd to 6th bits are used for face markers, e.g., the last third bit corresponds to loc = 0. */
/*  One use of the face marker is in flip algorithm. Each queued face (check for locally Delaunay) is marked. */
PETSC_STATIC_INLINE void markface(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] |= (int) (4<<(t)->loc);
}

PETSC_STATIC_INLINE void unmarkface(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] &= ~(int) (4<<(t)->loc);
}

PETSC_STATIC_INLINE PetscBool facemarked(TetGenMesh *m, triface *t) {
  return (((int *) (t->tet))[m->elemmarkerindex] & (int) (4<<(t)->loc)) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

/*  markedge(), unmarkedge(), edgemarked() -- primitives to flag or unflag an edge of a tetrahedron.  From the last 7th to 12th bits are used for */
/*    edge markers, e.g., the last 7th bit corresponds to the 0th edge, etc. */
/*  Remark: The last 7th bit is marked by 2^6 = 64. */
PETSC_STATIC_INLINE void markedge(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] |= (int) (64<<locver2edge[(t)->loc][(t)->ver]);
}

PETSC_STATIC_INLINE void unmarkedge(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] &= ~(int) (64<<locver2edge[(t)->loc][(t)->ver]);
}

PETSC_STATIC_INLINE PetscBool edgemarked(TetGenMesh *m, triface *t) {
  return (((int *) (t->tet))[m->elemmarkerindex] & (int) (64<<locver2edge[(t)->loc][(t)->ver])) != 0 ? PETSC_TRUE : PETSC_FALSE;
}
/*** End of primitives for tetrahedra ***/

/*** Begin of primitives for subfaces/subsegments ***/
/*  Each subface contains three pointers to its neighboring subfaces, with */
/*    edge versions.  To save memory, both information are kept in a single */
/*    pointer. To make this possible, all subfaces are aligned to eight-byte */
/*    boundaries, so that the last three bits of each pointer are zeros. An */
/*    edge version (in the range 0 to 5) is compressed into the last three */
/*    bits of each pointer by 'sencode()'.  'sdecode()' decodes a pointer, */
/*    extracting an edge version and a pointer to the beginning of a subface. */
PETSC_STATIC_INLINE void sdecode(shellface sptr, face *s) {
  s->shver = (int) ((PETSC_UINTPTR_T) (sptr) & (PETSC_UINTPTR_T) 7);
  s->sh    = (shellface *) ((PETSC_UINTPTR_T) (sptr) & ~ (PETSC_UINTPTR_T) 7);
}

PETSC_STATIC_INLINE shellface sencode(face *s) {
  return (shellface) ((PETSC_UINTPTR_T) s->sh | (PETSC_UINTPTR_T) s->shver);
}

/*  spivot() finds the other subface (from this subface) that shares the */
/*    same edge. */
PETSC_STATIC_INLINE void spivot(face *s1, face *s2) {
  shellface sptr = s1->sh[Orient(s1->shver)];
  sdecode(sptr, s2);
}

PETSC_STATIC_INLINE void spivotself(face *s) {
  shellface sptr = s->sh[Orient(s->shver)];
  sdecode(sptr, s);
}

/*  sbond() bonds two subfaces together, i.e., after bonding, both faces */
/*    are pointing to each other. */
PETSC_STATIC_INLINE void sbond(face *s1, face *s2) {
  s1->sh[Orient(s1->shver)] = sencode(s2);
  s2->sh[Orient(s2->shver)] = sencode(s1);
}

/*  sbond1() only bonds s2 to s1, i.e., after bonding, s1 is pointing to s2, but s2 is not pointing to s1. */
PETSC_STATIC_INLINE void sbond1(face *s1, face *s2) {
  s1->sh[Orient(s1->shver)] = sencode(s2);
}

/*  Dissolve a subface bond (from one side).  Note that the other subface will still think it's connected to this subface. */
PETSC_STATIC_INLINE void sdissolve(TetGenMesh *m, face *s) {
  s->sh[Orient(s->shver)] = (shellface) m->dummysh;
}

/*  These primitives determine or set the origin, destination, or apex of a subface with respect to the edge version. */
PETSC_STATIC_INLINE point sorg(face *s) {
  return (point) s->sh[3 + vo[s->shver]];
}

PETSC_STATIC_INLINE point sdest(face *s) {
  return (point) s->sh[3 + vd[s->shver]];
}

PETSC_STATIC_INLINE point sapex(face *s) {
  return (point) s->sh[3 + va[s->shver]];
}

PETSC_STATIC_INLINE void setsorg(face *s, point pointptr) {
  s->sh[3 + vo[s->shver]] = (shellface) pointptr;
}

PETSC_STATIC_INLINE void setsdest(face *s, point pointptr) {
  s->sh[3 + vd[s->shver]] = (shellface) pointptr;
}

PETSC_STATIC_INLINE void setsapex(face *s, point pointptr) {
  s->sh[3 + va[s->shver]] = (shellface) pointptr;
}

/*  These primitives were drived from Mucke[2]'s triangle-edge data structure */
/*    to change face-edge relation in a subface (sesym, senext and senext2). */
PETSC_STATIC_INLINE void sesym(face *s1, face *s2) {
  s2->sh    = s1->sh;
  s2->shver = s1->shver + (EdgeRing(s1->shver) ? -1 : 1);
}

PETSC_STATIC_INLINE void sesymself(face *s) {
  s->shver += (EdgeRing(s->shver) ? -1 : 1);
}

PETSC_STATIC_INLINE void senext(face *s1, face *s2) {
  s2->sh    = s1->sh;
  s2->shver = ve[s1->shver];
}

PETSC_STATIC_INLINE void senextself(face *s) {
  s->shver = ve[s->shver];
}

PETSC_STATIC_INLINE void senext2(face *s1, face *s2) {
  s2->sh    = s1->sh;
  s2->shver = ve[ve[s1->shver]];
}

PETSC_STATIC_INLINE void senext2self(face *s) {
  s->shver = ve[ve[s->shver]];
}

/*  If f0 and f1 are both in the same face ring, then f1 = f0.fnext(), */
PETSC_STATIC_INLINE void sfnext(TetGenMesh *m, face *s1, face *s2) {
  TetGenMeshGetNextSFace(m, s1, s2);
}

PETSC_STATIC_INLINE void sfnextself(TetGenMesh *m, face *s) {
  TetGenMeshGetNextSFace(m, s, PETSC_NULL);
}

/*  These primitives read or set a pointer of the badface structure.  The pointer is stored sh[11]. */
PETSC_STATIC_INLINE badface* shell2badface(face *s) {
  return (badface*) s->sh[11];
}

PETSC_STATIC_INLINE void setshell2badface(face *s, badface* value) {
  s->sh[11] = (shellface) value;
}

/*  Check or set a subface's maximum area bound. */
PETSC_STATIC_INLINE PetscReal areabound(TetGenMesh *m, face *s) {
  return ((PetscReal *) (s->sh))[m->areaboundindex];
}

PETSC_STATIC_INLINE void setareabound(TetGenMesh *m, face *s, PetscReal value) {
  ((PetscReal *) (s->sh))[m->areaboundindex] = value;
}

/*  These two primitives read or set a shell marker.  Shell markers are used */
/*    to hold user boundary information. */
/*  The last two bits of the int ((int *) ((s).sh))[shmarkindex] are used */
/*    by sinfect() and smarktest(). */
PETSC_STATIC_INLINE int shellmark(TetGenMesh *m, face *s) {
  return (((int *) ((s)->sh))[m->shmarkindex]) >> (int) 2;
}

PETSC_STATIC_INLINE void setshellmark(TetGenMesh *m, face *s, int value) {
  ((int *) ((s)->sh))[m->shmarkindex] = (value << (int) 2) + ((((int *) ((s)->sh))[m->shmarkindex]) & (int) 3);
}

/*  These two primitives set or read the type of the subface or subsegment. */
PETSC_STATIC_INLINE shestype shelltype(TetGenMesh *m, face *s) {
  return (shestype) ((int *) (s->sh))[m->shmarkindex + 1];
}

PETSC_STATIC_INLINE void setshelltype(TetGenMesh *m, face *s, shestype value) {
  ((int *) (s->sh))[m->shmarkindex + 1] = (int) value;
}

/*  These two primitives set or read the pbc group of the subface. */
PETSC_STATIC_INLINE int shellpbcgroup(TetGenMesh *m, face *s) {
  return ((int *) (s->sh))[m->shmarkindex + 2];
}

PETSC_STATIC_INLINE void setshellpbcgroup(TetGenMesh *m, face *s, int value) {
  ((int *) (s->sh))[m->shmarkindex + 2] = value;
}

/*  sinfect(), sinfected(), suninfect() -- primitives to flag or unflag a */
/*    subface. The last bit of ((int *) ((s).sh))[shmarkindex] is flaged. */
PETSC_STATIC_INLINE void sinfect(TetGenMesh *m, face *s) {
  ((int *) ((s)->sh))[m->shmarkindex] = (((int *) ((s)->sh))[m->shmarkindex] | (int) 1);
  /*  s->sh[6] = (shellface) ((unsigned long) s->sh[6] | (unsigned long) 4l); */
}

PETSC_STATIC_INLINE void suninfect(TetGenMesh *m, face *s) {
  ((int *) ((s)->sh))[m->shmarkindex] = (((int *) ((s)->sh))[m->shmarkindex] & ~(int) 1);
  /*  s->sh[6] = (shellface)((unsigned long) s->sh[6] & ~(unsigned long) 4l); */
}

/*  Test a subface for viral infection. */
PETSC_STATIC_INLINE PetscBool sinfected(TetGenMesh *m, face *s) {
  return (((int *) ((s)->sh))[m->shmarkindex] & (int) 1) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

/*  smarktest(), smarktested(), sunmarktest() -- primitives to flag or unflag */
/*    a subface. The last 2nd bit of ((int *) ((s).sh))[shmarkindex] is flaged. */
#define smarktest(s) ((int *) ((s)->sh))[m->shmarkindex] = (((int *)((s)->sh))[m->shmarkindex] | (int) 2)

#define sunmarktest(s) ((int *) ((s)->sh))[m->shmarkindex] = (((int *)((s)->sh))[m->shmarkindex] & ~(int) 2)

#define smarktested(s) ((((int *) ((s)->sh))[m->shmarkindex] & (int) 2) != 0)
/*** End of primitives for subfaces/subsegments ***/

/*** Begin of primitives for interacting between tetrahedra and subfaces ***/
/*  tspivot() finds a subface abutting on this tetrahdera. */
PETSC_STATIC_INLINE void tspivot(TetGenMesh *m, triface *t, face *s) {
  if ((t)->tet[9]) {
    sdecode(((shellface *) (t)->tet[9])[(t)->loc], s);
  } else {
    (s)->sh = m->dummysh;
  }
}

/*  stpivot() finds a tetrahedron abutting a subface. */
PETSC_STATIC_INLINE void stpivot(TetGenMesh *m, face *s, triface *t) {
  tetrahedron ptr = (tetrahedron) s->sh[6 + EdgeRing(s->shver)];
  decode(ptr, t);
}

/*  tsbond() bond a tetrahedron to a subface. */
PETSC_STATIC_INLINE PetscErrorCode tsbond(TetGenMesh *m, triface *t, face *s) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(t)->tet[9]) {
    int i;
    /*  Allocate space for this tet. */
    ierr = MemoryPoolAlloc(m->tet2subpool, (void **) &(t)->tet[9]);CHKERRQ(ierr);
    /*  NULL all fields in this space. */
    for(i = 0; i < 4; i++) {
      ((shellface *) (t)->tet[9])[i] = (shellface) m->dummysh;
    }
  }
  /*  Bond t <==> s. */
  ((shellface *) (t)->tet[9])[(t)->loc] = sencode(s);
  /* t.tet[8 + t.loc] = (tetrahedron) sencode(s); */
  s->sh[6 + EdgeRing(s->shver)] = (shellface) encode(t);
  PetscFunctionReturn(0);
}

/*  tsdissolve() dissolve a bond (from the tetrahedron side). */
PETSC_STATIC_INLINE void tsdissolve(TetGenMesh *m, triface *t) {
  if ((t)->tet[9]) {
    ((shellface *) (t)->tet[9])[(t)->loc] = (shellface) m->dummysh;
  }
  /*  t.tet[8 + t.loc] = (tetrahedron) dummysh; */
}

/*  stdissolve() dissolve a bond (from the subface side). */
PETSC_STATIC_INLINE void stdissolve(TetGenMesh *m, face *s) {
  s->sh[6 + EdgeRing(s->shver)] = (shellface) m->dummytet;
}
/*** End of primitives for interacting between tetrahedra and subfaces ***/

/*** Begin of primitives for interacting between subfaces and subsegs ***/
/*  sspivot() finds a subsegment abutting a subface. */
PETSC_STATIC_INLINE void sspivot(TetGenMesh *m, face *s, face *edge) {
  shellface sptr = (shellface) s->sh[8 + Orient(s->shver)];
  sdecode(sptr, edge);
}

/*  ssbond() bond a subface to a subsegment. */
PETSC_STATIC_INLINE void ssbond(TetGenMesh *m, face *s, face *edge) {
  s->sh[8 + Orient(s->shver)] = sencode(edge);
  edge->sh[0] = sencode(s);
}

/*  ssdisolve() dissolve a bond (from the subface side) */
PETSC_STATIC_INLINE void ssdissolve(TetGenMesh *m, face *s) {
  s->sh[8 + Orient(s->shver)] = (shellface) m->dummysh;
}
/*** End of primitives for interacting between subfaces and subsegs ***/

/*** Begin of primitives for interacting between tet and subsegs ***/
PETSC_STATIC_INLINE void tsspivot1(TetGenMesh *m, triface *t, face *s)
{
  if ((t)->tet[8]) {
    sdecode(((shellface *) (t)->tet[8])[locver2edge[(t)->loc][(t)->ver]], s);
  } else {
    (s)->sh = m->dummysh;
  }
}

/*  Only bond/dissolve at tet's side, but not vice versa. */
PETSC_STATIC_INLINE PetscErrorCode tssbond1(TetGenMesh *m, triface *t, face *s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(t)->tet[8]) {
    int i;
    /*  Allocate space for this tet. */
    ierr = MemoryPoolAlloc(m->tet2segpool, (void **) &(t)->tet[8]);CHKERRQ(ierr);
    /*  NULL all fields in this space. */
    for(i = 0; i < 6; i++) {
      ((shellface *) (t)->tet[8])[i] = (shellface) m->dummysh;
    }
  }
  /*  Bond the segment. */
  ((shellface *) (t)->tet[8])[locver2edge[(t)->loc][(t)->ver]] = sencode((s));
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void tssdissolve1(TetGenMesh *m, triface *t)
{
  if ((t)->tet[8]) {
    ((shellface *) (t)->tet[8])[locver2edge[(t)->loc][(t)->ver]] = (shellface) m->dummysh;
  }
}
/*** End of primitives for interacting between tet and subsegs ***/

/*** Begin of advanced primitives ***/

/*  adjustedgering() adjusts the edge version so that it belongs to the */
/*    indicated edge ring.  The 'direction' only can be 0(CCW) or 1(CW). */
/*    If the edge is not in the wanted edge ring, reverse it. */
PETSC_STATIC_INLINE void adjustedgering_triface(triface *t, int direction) {
  if (EdgeRing(t->ver) != direction) {
    esymself(t);
  }
}
PETSC_STATIC_INLINE void adjustedgering_face(face *s, int direction) {
  if (EdgeRing(s->shver) != direction) {
    sesymself(s);
  }
}

/*  isdead() returns TRUE if the tetrahedron or subface has been dealloced. */
PETSC_STATIC_INLINE PetscBool isdead_triface(triface *t) {
  if (!t->tet) {
    return PETSC_TRUE;
  } else {
    return t->tet[4] ? PETSC_FALSE : PETSC_TRUE;
  }
}
PETSC_STATIC_INLINE PetscBool isdead_face(face *s) {
  if (!s->sh) {
    return PETSC_TRUE;
  } else {
    return s->sh[3] ? PETSC_FALSE : PETSC_TRUE;
  }
}

/*  isfacehaspoint() returns TRUE if the 'testpoint' is one of the vertices of the tetface 't' subface 's'. */
PETSC_STATIC_INLINE PetscBool isfacehaspoint_triface(triface *t, point testpoint) {
  return ((org(t) == testpoint) || (dest(t) == testpoint) || (apex(t) == testpoint)) ? PETSC_TRUE : PETSC_FALSE;
}
PETSC_STATIC_INLINE PetscBool isfacehaspoint_face(face* s, point testpoint) {
  return (s->sh[3] == (shellface) testpoint) || (s->sh[4] == (shellface) testpoint) || (s->sh[5] == (shellface) testpoint) ? PETSC_TRUE : PETSC_FALSE;
}

/*  isfacehasedge() returns TRUE if the edge (given by its two endpoints) is one of the three edges of the subface 's'. */
PETSC_STATIC_INLINE PetscBool isfacehasedge(face* s, point tend1, point tend2) {
  return (isfacehaspoint_face(s, tend1) && isfacehaspoint_face(s, tend2)) ? PETSC_TRUE : PETSC_FALSE;
}

/*  issymexist() returns TRUE if the adjoining tetrahedron is not 'duumytet'. */
PETSC_STATIC_INLINE PetscBool issymexist(TetGenMesh *m, triface* t) {
  tetrahedron *ptr = (tetrahedron *) ((unsigned long)(t->tet[t->loc]) & ~(unsigned long)7l);
  return ptr != m->dummytet ? PETSC_TRUE : PETSC_FALSE;
}

/*  dot() returns the dot product: v1 dot v2. */
PETSC_STATIC_INLINE PetscReal dot(PetscReal *v1, PetscReal *v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}
/*  cross() computes the cross product: n = v1 cross v2. */
PETSC_STATIC_INLINE void cross(PetscReal *v1, PetscReal *v2, PetscReal *n)
{
  n[0] =   v1[1] * v2[2] - v2[1] * v1[2];
  n[1] = -(v1[0] * v2[2] - v2[0] * v1[2]);
  n[2] =   v1[0] * v2[1] - v2[0] * v1[1];
}
/*  distance() computes the Euclidean distance between two points. */
PETSC_STATIC_INLINE PetscReal distance(PetscReal *p1, PetscReal *p2)
{
  return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) +
              (p2[1] - p1[1]) * (p2[1] - p1[1]) +
              (p2[2] - p1[2]) * (p2[2] - p1[2]));
}

/*  Linear algebra operators. */
#define NORM2(x, y, z) ((x) * (x) + (y) * (y) + (z) * (z))

#define DIST(p1, p2) sqrt(NORM2((p2)[0] - (p1)[0], (p2)[1] - (p1)[1], (p2)[2] - (p1)[2]))

#define DOT(v1, v2) ((v1)[0] * (v2)[0] + (v1)[1] * (v2)[1] + (v1)[2] * (v2)[2])

#define CROSS(v1, v2, n) (n)[0] =   (v1)[1] * (v2)[2] - (v2)[1] * (v1)[2];\
  (n)[1] = -((v1)[0] * (v2)[2] - (v2)[0] * (v1)[2]);\
  (n)[2] =   (v1)[0] * (v2)[1] - (v2)[0] * (v1)[1]

#define SETVECTOR3(V, a0, a1, a2) (V)[0] = (a0); (V)[1] = (a1); (V)[2] = (a2)

#define SWAP2(a0, a1, tmp) (tmp) = (a0); (a0) = (a1); (a1) = (tmp)
/*** End of advanced primitives ***/

/*============================ End Converted TetGen Inline Functions =============================*/

/*=============================== Start Converted TetGen Functions ===============================*/
#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCreate"
/* tetgenmesh::tetgenmesh() */
PetscErrorCode TetGenMeshCreate(TetGenMesh **mesh)
{
  TetGenMesh    *m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(TetGenMesh, &m);CHKERRQ(ierr);
  /* m->bgm = PETSC_NULL; */
  m->in  = PETSC_NULL;
  m->b   = PETSC_NULL;

  m->tetrahedrons    = PETSC_NULL;
  m->subfaces        = PETSC_NULL;
  m->subsegs         = PETSC_NULL;
  m->points          = PETSC_NULL;
  m->badsubsegs      = PETSC_NULL;
  m->badsubfaces     = PETSC_NULL;
  m->badtetrahedrons = PETSC_NULL;
  m->tet2segpool     = PETSC_NULL;
  m->tet2subpool     = PETSC_NULL;

  m->firsttopface.tet = PETSC_NULL; m->firsttopface.loc = 0; m->firsttopface.ver = 0;
  m->firstbotface.tet = PETSC_NULL; m->firstbotface.loc = 0; m->firstbotface.ver = 0;
  m->recenttet.tet    = PETSC_NULL; m->recenttet.loc    = 0; m->recenttet.ver    = 0;

  m->dummytet     = PETSC_NULL;
  m->dummytetbase = PETSC_NULL;
  m->dummysh      = PETSC_NULL;
  m->dummyshbase  = PETSC_NULL;

  m->facetabovepointarray = PETSC_NULL;
  m->abovepoint       = PETSC_NULL;
  m->dummypoint       = PETSC_NULL;
  m->btreenode_list   = PETSC_NULL;
  m->highordertable   = PETSC_NULL;
  m->subpbcgrouptable = PETSC_NULL;
  m->segpbcgrouptable = PETSC_NULL;
  m->idx2segpglist    = PETSC_NULL;
  m->segpglist        = PETSC_NULL;

  m->cavetetlist    = PETSC_NULL;
  m->cavebdrylist   = PETSC_NULL;
  m->caveoldtetlist = PETSC_NULL;
  m->caveshlist = m->caveshbdlist = PETSC_NULL;
  m->subsegstack = m->subfacstack = PETSC_NULL;

  m->elemfliplist  = PETSC_NULL;
  m->fixededgelist = PETSC_NULL;

  m->xmax = m->xmin = m->ymax = m->ymin = m->zmax = m->zmin = 0.0;
  m->longest = 0.0;
  m->hullsize = 0l;
  m->insegments = 0l;
  m->meshedges = 0l;
  m->pointmtrindex = 0;
  m->pointmarkindex = 0;
  m->point2simindex = 0;
  m->point2pbcptindex = 0;
  m->highorderindex = 0;
  m->elemattribindex = 0;
  m->volumeboundindex = 0;
  m->shmarkindex = 0;
  m->areaboundindex = 0;
  m->checksubfaces = 0;
  m->checksubsegs = 0;
  m->checkpbcs = 0;
  m->varconstraint = 0;
  m->nonconvex = 0;
  m->dupverts = 0;
  m->unuverts = 0;
  m->relverts = 0;
  m->suprelverts = 0;
  m->collapverts = 0;
  m->unsupverts = 0;
  m->jettisoninverts = 0;
  m->samples = 0l;
  m->randomseed = 1l;
  m->macheps = 0.0;
  m->minfaceang = m->minfacetdihed = PETSC_PI;
  m->b_steinerflag = PETSC_FALSE;

  m->ptloc_count = m->ptloc_max_count = 0l;
  m->orient3dcount = 0l;
  m->inspherecount = m->insphere_sos_count = 0l;
  m->flip14count = m->flip26count = m->flipn2ncount = 0l;
  m->flip22count = 0l;
  m->inserthullcount = 0l;
  m->maxbowatcavsize = m->totalbowatcavsize = m->totaldeadtets = 0l;
  m->across_face_count = m->across_edge_count = m->across_max_count = 0l;
  m->maxcavsize = m->maxregionsize = 0l;
  m->ndelaunayedgecount = m->cavityexpcount = 0l;
  m->opt_tet_peels = m->opt_face_flips = m->opt_edge_flips = 0l;

  m->maxcavfaces = m->maxcavverts = 0;
  m->abovecount = 0l;
  m->bowatvolcount = m->bowatsubcount = m->bowatsegcount = 0l;
  m->updvolcount = m->updsubcount = m->updsegcount = 0l;
  m->outbowatcircumcount = 0l;
  m->failvolcount = m->failsubcount = m->failsegcount = 0l;
  m->r1count = m->r2count = m->r3count = 0l;
  m->cdtenforcesegpts = 0l;
  m->rejsegpts = m->rejsubpts = m->rejtetpts = 0l;
  m->flip23s = m->flip32s = m->flip22s = m->flip44s = 0l;
  *mesh = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDestroy"
/* tetgenmesh::~tetgenmesh() */
PetscErrorCode TetGenMeshDestroy(TetGenMesh **mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*mesh) {PetscFunctionReturn(0);}
  /* (*mesh)->bgm = PETSC_NULL; */
  (*mesh)->in  = PETSC_NULL;
  (*mesh)->b   = PETSC_NULL;
  ierr = MemoryPoolDestroy(&(*mesh)->tetrahedrons);CHKERRQ(ierr);
  ierr = MemoryPoolDestroy(&(*mesh)->subfaces);CHKERRQ(ierr);
  ierr = MemoryPoolDestroy(&(*mesh)->subsegs);CHKERRQ(ierr);
  ierr = MemoryPoolDestroy(&(*mesh)->points);CHKERRQ(ierr);
  ierr = MemoryPoolDestroy(&(*mesh)->tet2segpool);CHKERRQ(ierr);
  ierr = MemoryPoolDestroy(&(*mesh)->tet2subpool);CHKERRQ(ierr);

  ierr = PetscFree((*mesh)->dummytetbase);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->dummyshbase);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->facetabovepointarray);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->dummypoint);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->highordertable);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->subpbcgrouptable);CHKERRQ(ierr);
  ierr = ListDestroy(&(*mesh)->segpbcgrouptable);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->idx2segpglist);CHKERRQ(ierr);
  ierr = PetscFree((*mesh)->segpglist);CHKERRQ(ierr);

  ierr = ArrayPoolDestroy(&(*mesh)->cavetetlist);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&(*mesh)->cavebdrylist);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&(*mesh)->caveoldtetlist);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&(*mesh)->subsegstack);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&(*mesh)->subfacstack);CHKERRQ(ierr);
  ierr = PetscFree(*mesh);CHKERRQ(ierr);
  *mesh = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenLUDecomp"
/*  TetGenLUDecomp()    Compute the LU decomposition of a matrix.              */
/*                                                                             */
/*  Compute the LU decomposition of a (non-singular) square matrix A using     */
/*  partial pivoting and implicit row exchanges.  The result is:               */
/*      A = P * L * U,                                                         */
/*  where P is a permutation matrix, L is unit lower triangular, and U is      */
/*  upper triangular.  The factored form of A is used in combination with      */
/*  'TetGenLUSolve()' to solve linear equations: Ax = b, or invert a matrix.        */
/*                                                                             */
/*  The inputs are a square matrix 'lu[N..n+N-1][N..n+N-1]', it's size is 'n'. */
/*  On output, 'lu' is replaced by the LU decomposition of a rowwise permuta-  */
/*  tion of itself, 'ps[N..n+N-1]' is an output vector that records the row    */
/*  permutation effected by the partial pivoting, effectively,  'ps' array     */
/*  tells the user what the permutation matrix P is; 'd' is output as +1/-1    */
/*  depending on whether the number of row interchanges was even or odd,       */
/*  respectively.                                                              */
/*                                                                             */
/*  Return true if the LU decomposition is successfully computed, otherwise,   */
/*  return false in case that A is a singular matrix.                          */
PetscBool TetGenLUDecomp(PetscReal lu[4][4], int n, int* ps, PetscReal *d, int N)
{
  PetscReal scales[4];
  PetscReal pivot, biggest, mult, tempf;
  int pivotindex = 0;
  int i, j, k;

  *d = 1.0;                                      /*  No row interchanges yet. */

  for (i = N; i < n + N; i++) {                             /*  For each row. */
    /*  Find the largest element in each row for row equilibration */
    biggest = 0.0;
    for (j = N; j < n + N; j++)
      if (biggest < (tempf = fabs(lu[i][j])))
        biggest  = tempf;
    if (biggest != 0.0)
      scales[i] = 1.0 / biggest;
    else {
      scales[i] = 0.0;
      return PETSC_FALSE;                            /*  Zero row: singular matrix. */
    }
    ps[i] = i;                                 /*  Initialize pivot sequence. */
  }

  for (k = N; k < n + N - 1; k++) {                      /*  For each column. */
    /*  Find the largest element in each column to pivot around. */
    biggest = 0.0;
    for (i = k; i < n + N; i++) {
      if (biggest < (tempf = fabs(lu[ps[i]][k]) * scales[ps[i]])) {
        biggest = tempf;
        pivotindex = i;
      }
    }
    if (biggest == 0.0) {
      return PETSC_FALSE;                         /*  Zero column: singular matrix. */
    }
    if (pivotindex != k) {                         /*  Update pivot sequence. */
      j = ps[k];
      ps[k] = ps[pivotindex];
      ps[pivotindex] = j;
      *d = -(*d);                          /*  ...and change the parity of d. */
    }

    /*  Pivot, eliminating an extra variable  each time */
    pivot = lu[ps[k]][k];
    for (i = k + 1; i < n + N; i++) {
      lu[ps[i]][k] = mult = lu[ps[i]][k] / pivot;
      if (mult != 0.0) {
        for (j = k + 1; j < n + N; j++)
          lu[ps[i]][j] -= mult * lu[ps[k]][j];
      }
    }
  }

  /*  (lu[ps[n + N - 1]][n + N - 1] == 0.0) ==> A is singular. */
  return lu[ps[n + N - 1]][n + N - 1] != 0.0 ? PETSC_TRUE : PETSC_FALSE;
}

#undef __FUNCT__
#define __FUNCT__ "TetGenLUSolve"
/*  TetGenLUSolve()    Solves the linear equation:  Ax = b,  after the matrix A     */
/*                has been decomposed into the lower and upper triangular      */
/*                matrices L and U, where A = LU.                              */
/*                                                                             */
/*  'lu[N..n+N-1][N..n+N-1]' is input, not as the matrix 'A' but rather as     */
/*  its LU decomposition, computed by the routine 'TetGenLUDecomp'; 'ps[N..n+N-1]'   */
/*  is input as the permutation vector returned by 'TetGenLUDecomp';  'b[N..n+N-1]'  */
/*  is input as the right-hand side vector, and returns with the solution      */
/*  vector. 'lu', 'n', and 'ps' are not modified by this routine and can be    */
/*  left in place for successive calls with different right-hand sides 'b'.    */
void TetGenLUSolve(PetscReal lu[4][4], int n, int *ps, PetscReal *b, int N)
{
  int i, j;
  PetscReal X[4], dot;

  for (i = N; i < n + N; i++) X[i] = 0.0;

  /*  Vector reduction using U triangular matrix. */
  for (i = N; i < n + N; i++) {
    dot = 0.0;
    for (j = N; j < i + N; j++)
      dot += lu[ps[i]][j] * X[j];
    X[i] = b[ps[i]] - dot;
  }

  /*  Back substitution, in L triangular matrix. */
  for (i = n + N - 1; i >= N; i--) {
    dot = 0.0;
    for (j = i + 1; j < n + N; j++)
      dot += lu[ps[i]][j] * X[j];
    X[i] = (X[i] - dot) / lu[ps[i]][i];
  }

  for (i = N; i < n + N; i++) b[i] = X[i];
}

#undef __FUNCT__
#define __FUNCT__ "interiorangle"
/*  interiorangle()    Return the interior angle (0 - 2 * PI) between vectors  */
/*                     o->p1 and o->p2.                                        */
/*                                                                             */
/*  'n' is the normal of the plane containing face (o, p1, p2).  The interior  */
/*  angle is the total angle rotating from o->p1 around n to o->p2.  Exchange  */
/*  the position of p1 and p2 will get the complement angle of the other one.  */
/*  i.e., interiorangle(o, p1, p2) = 2 * PI - interiorangle(o, p2, p1).  Set   */
/*  'n' be NULL if you only want the interior angle between 0 - PI.            */
static PetscReal interiorangle(PetscReal* o, PetscReal* p1, PetscReal* p2, PetscReal* n)
{
  PetscReal v1[3], v2[3], np[3];
  PetscReal theta, costheta, lenlen;
  PetscReal ori, len1, len2;

  /*  Get the interior angle (0 - PI) between o->p1, and o->p2. */
  v1[0] = p1[0] - o[0];
  v1[1] = p1[1] - o[1];
  v1[2] = p1[2] - o[2];
  v2[0] = p2[0] - o[0];
  v2[1] = p2[1] - o[1];
  v2[2] = p2[2] - o[2];
  len1 = sqrt(dot(v1, v1));
  len2 = sqrt(dot(v2, v2));
  lenlen = len1 * len2;
#ifdef PETSC_USE_DEBUG
  if (lenlen == 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
  costheta = dot(v1, v2) / lenlen;
  if (costheta > 1.0) {
    costheta = 1.0; /*  Roundoff. */
  } else if (costheta < -1.0) {
    costheta = -1.0; /*  Roundoff. */
  }
  theta = acos(costheta);
  if (n) {
    /*  Get a point above the face (o, p1, p2); */
    np[0] = o[0] + n[0];
    np[1] = o[1] + n[1];
    np[2] = o[2] + n[2];
    /*  Adjust theta (0 - 2 * PI). */
    ori = TetGenOrient3D(p1, o, np, p2);
    if (ori > 0.0) {
      theta = 2 * PETSC_PI - theta;
    }
  }
  return theta;
}

#undef __FUNCT__
#define __FUNCT__ "TetGenOptsInitialize"
PetscErrorCode TetGenOptsInitialize(TetGenOpts *t)
{
  PetscFunctionBegin;
  t->plc = 0;
  t->quality = 0;
  t->refine = 0;
  t->coarse = 0;
  t->metric = 0;
  t->minratio = 2.0;
  t->goodratio = 0.0;
  t->minangle = 20.0;
  t->goodangle = 0.0;
  t->maxdihedral = 165.0;
  t->mindihedral = 5.0;
  t->varvolume = 0;
  t->fixedvolume = 0;
  t->maxvolume = -1.0;
  t->regionattrib = 0;
  t->insertaddpoints = 0;
  t->diagnose = 0;
  t->offcenter = 0;
  t->conformdel = 0;
  t->alpha1 = sqrt(2.0);
  t->alpha2 = 1.0;
  t->alpha3 = 0.6;
  t->zeroindex = 0;
  t->btree = 1;
  t->max_btreenode_size = 100;
  t->facesout = 0;
  t->edgesout = 0;
  t->neighout = 0;
  t->voroout = 0;
  t->meditview = 0;
  t->gidview = 0;
  t->geomview = 0;
  t->vtkview = 0;
  t->optlevel = 3;
  t->optpasses = 3;
  t->order = 1;
  t->nojettison = 0;
  t->nobound = 0;
  t->nonodewritten = 0;
  t->noelewritten = 0;
  t->nofacewritten = 0;
  t->noiterationnum = 0;
  t->nobisect = 0;
  t->noflip = 0;
  t->steiner = -1;
  t->fliprepair = 1;
  t->nomerge = 0;
  t->docheck = 0;
  t->quiet = 0;
  t->verbose = 0;
  t->useshelles = 0;
  t->maxflipedgelinksize = 10;
  t->epsilon = 1.0e-8;
  t->epsilon2 = 1.0e-5;
  t->object = TETGEN_OBJECT_NONE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PLCCreate"
/* tetgenio:: initialize() */
PetscErrorCode PLCCreate(PLC **plc)
{
  PLC           *p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(PLC, &p);CHKERRQ(ierr);
  p->firstnumber = 0; /*  Default item index is numbered from Zero. */
  p->mesh_dim = 3; /*  Default mesh dimension is 3. */
  p->useindex = PETSC_TRUE;

  p->pointlist = PETSC_NULL;
  p->pointattributelist = PETSC_NULL;
  p->pointmtrlist = PETSC_NULL;
  p->pointmarkerlist = PETSC_NULL;
  p->numberofpoints = 0;
  p->numberofpointattributes = 0;
  p->numberofpointmtrs = 0;

  p->tetrahedronlist = PETSC_NULL;
  p->tetrahedronattributelist = PETSC_NULL;
  p->tetrahedronvolumelist = PETSC_NULL;
  p->neighborlist = PETSC_NULL;
  p->numberoftetrahedra = 0;
  p->numberofcorners = 4; /*  Default is 4 nodes per element. */
  p->numberoftetrahedronattributes = 0;

  p->trifacelist = PETSC_NULL;
  p->adjtetlist = PETSC_NULL;
  p->trifacemarkerlist = PETSC_NULL;
  p->numberoftrifaces = 0;

  p->facetlist = PETSC_NULL;
  p->facetmarkerlist = PETSC_NULL;
  p->numberoffacets = 0;

  p->edgelist = PETSC_NULL;
  p->edgemarkerlist = PETSC_NULL;
  p->numberofedges = 0;

  p->holelist = PETSC_NULL;
  p->numberofholes = 0;

  p->regionlist = PETSC_NULL;
  p->numberofregions = 0;

  p->facetconstraintlist = PETSC_NULL;
  p->numberoffacetconstraints = 0;
  p->segmentconstraintlist = PETSC_NULL;
  p->numberofsegmentconstraints = 0;

  p->pbcgrouplist = PETSC_NULL;
  p->numberofpbcgroups = 0;

  p->vpointlist = PETSC_NULL;
  p->vedgelist = PETSC_NULL;
  p->vfacetlist = PETSC_NULL;
  p->vcelllist = PETSC_NULL;
  p->numberofvpoints = 0;
  p->numberofvedges = 0;
  p->numberofvfacets = 0;
  p->numberofvcells = 0;

  p->tetunsuitable = PETSC_NULL;
  *plc = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PLCDestroy"
/* tetgenio:: deinitialize() */
PetscErrorCode PLCDestroy(PLC **p)
{
  PLC           *plc = *p;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*p) {PetscFunctionReturn(0);}
  ierr = PetscFree(plc->pointlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->pointattributelist);CHKERRQ(ierr);
  ierr = PetscFree(plc->pointmtrlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->pointmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->tetrahedronlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->tetrahedronattributelist);CHKERRQ(ierr);
  ierr = PetscFree(plc->tetrahedronvolumelist);CHKERRQ(ierr);
  ierr = PetscFree(plc->neighborlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->trifacelist);CHKERRQ(ierr);
  ierr = PetscFree(plc->adjtetlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->trifacemarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->edgelist);CHKERRQ(ierr);
  ierr = PetscFree(plc->edgemarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->facetmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->holelist);CHKERRQ(ierr);
  ierr = PetscFree(plc->regionlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->facetconstraintlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->segmentconstraintlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->vpointlist);CHKERRQ(ierr);
  ierr = PetscFree(plc->vedgelist);CHKERRQ(ierr);
  for (i = 0; i < plc->numberoffacets && plc->facetlist; i++) {
    facet *f = &plc->facetlist[i];

    for (j = 0; j < f->numberofpolygons; j++) {
      polygon *p = &f->polygonlist[j];

      ierr = PetscFree(p->vertexlist);CHKERRQ(ierr);
    }
    ierr = PetscFree(f->polygonlist);CHKERRQ(ierr);
    ierr = PetscFree(f->holelist);CHKERRQ(ierr);
  }
  ierr = PetscFree(plc->facetlist);CHKERRQ(ierr);
  for(i = 0; i < plc->numberofpbcgroups; i++) {
    pbcgroup *pg = &(plc->pbcgrouplist[i]);

    ierr = PetscFree(pg->pointpairlist);CHKERRQ(ierr);
  }
  ierr = PetscFree(plc->pbcgrouplist);CHKERRQ(ierr);
  for(i = 0; i < plc->numberofvfacets; i++) {
    ierr = PetscFree(plc->vfacetlist[i].elist);CHKERRQ(ierr);
  }
  ierr = PetscFree(plc->vfacetlist);CHKERRQ(ierr);
  for(i = 0; i < plc->numberofvcells; i++) {
    ierr = PetscFree(plc->vcelllist[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(plc->vcelllist);CHKERRQ(ierr);
  ierr = PetscFree(plc);CHKERRQ(ierr);
  *p = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListCreate"
/*  listinit()    Initialize a list for storing a data type.                   */
/*                                                                             */
/*  Determine the size of each item, set the maximum size allocated at onece,  */
/*  set the expand size in case the list is full, and set the linear order     */
/*  function if it is provided (default is NULL).                              */
/* tetgenmesh::list::list() and tetgenmesh::list::listinit() */
static PetscErrorCode ListCreate(int itbytes, compfunc pcomp, int mitems, int exsize, List **newl)
{
  List          *l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(List), &l);CHKERRQ(ierr);
  l->itembytes  = itbytes;
  l->comp       = pcomp;
  l->maxitems   = mitems < 0 ? 256 : mitems;
  l->expandsize = exsize < 0 ? 128 : exsize;
  l->items      = 0;
  ierr = PetscMalloc(l->maxitems * l->itembytes, &l->base);CHKERRQ(ierr);
  *newl = l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListAppend"
/*  append()    Add a new item at the end of the list.                         */
/*                                                                             */
/*  A new space at the end of this list will be allocated for storing the new  */
/*  item. If the memory is not sufficient, reallocation will be performed. If  */
/*  'appitem' is not NULL, the contents of this pointer will be copied to the  */
/*  new allocated space.  Returns the pointer to the new allocated space.      */
/* tetgenmesh::list::append() */
static PetscErrorCode ListAppend(List *l, void *appitem, void **newspace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Do we have enough space? */
  if (l->items == l->maxitems) {
    char *newbase;

    ierr = PetscMalloc((l->maxitems + l->expandsize) * l->itembytes, &newbase);CHKERRQ(ierr);
    ierr = PetscMemcpy(newbase, l->base, l->maxitems * l->itembytes);CHKERRQ(ierr);
    ierr = PetscFree(l->base);CHKERRQ(ierr);
    l->base      = newbase;
    l->maxitems += l->expandsize;
  }
  if (appitem) {
    ierr = PetscMemcpy(l->base + l->items * l->itembytes, appitem, l->itembytes);CHKERRQ(ierr);
  }
  l->items++;
  if (newspace) {*newspace = (void *) (l->base + (l->items - 1) * l->itembytes);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListInsert"
/*  insert()    Insert an item before 'pos' (range from 0 to items - 1).       */
/*                                                                             */
/*  A new space will be inserted at the position 'pos', that is, items lie     */
/*  after pos (including the item at pos) will be moved one space downwords.   */
/*  If 'insitem' is not NULL, its contents will be copied into the new         */
/*  inserted space. Return a pointer to the new inserted space.                */
/* tetgenmesh::list::insert() */
static PetscErrorCode ListInsert(List *l, int pos, void *insitem, void **newspace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pos >= l->items) {
    PetscFunctionReturn(ListAppend(l, insitem, newspace));
  }
  /*  Do we have enough space. */
  if (l->items == l->maxitems) {
    char *newbase;

    ierr = PetscMalloc((l->maxitems + l->expandsize) * l->itembytes, &newbase);CHKERRQ(ierr);
    ierr = PetscMemcpy(newbase, l->base, l->maxitems * l->itembytes);CHKERRQ(ierr);
    ierr = PetscFree(l->base);CHKERRQ(ierr);
    l->base      = newbase;
    l->maxitems += l->expandsize;
  }
  /*  Do block move. */
  ierr = PetscMemmove(l->base + (pos + 1) * l->itembytes, l->base + pos * l->itembytes, (l->items - pos) * l->itembytes);CHKERRQ(ierr);
  /*  Insert the item. */
  if (insitem) {
    ierr = PetscMemcpy(l->base + pos * l->itembytes, insitem, l->itembytes);CHKERRQ(ierr);
  }
  l->items++;
  if (newspace) {*newspace = (void *) (l->base + pos * l->itembytes);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListDelete"
/*  del()    Delete an item at 'pos' (range from 0 to items - 1).              */
/*                                                                             */
/*  The space at 'pos' will be overlapped by other item. If 'order' is 1, the  */
/*  remaining items of the list have the same order as usual, i.e., items lie  */
/*  after pos will be moved one space upwords. If 'order' is 0, the last item  */
/*  of the list will be moved up to pos.                                       */
/* tetgenmesh::list::del() */
static PetscErrorCode ListDelete(List *l, int pos, int order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  If 'pos' is the last item of the list, nothing need to do. */
  if (pos >= 0 && pos < l->items - 1) {
    if (order == 1) {
      /*  Do block move. */
      ierr = PetscMemmove(l->base + pos * l->itembytes, l->base + (pos + 1) * l->itembytes, (l->items - pos - 1) * l->itembytes);CHKERRQ(ierr);
    } else {
      /*  Use the last item to overlap the del item. */
      ierr = PetscMemcpy(l->base + pos * l->itembytes, l->base + (l->items - 1) * l->itembytes, l->itembytes);CHKERRQ(ierr);
    }
  }
  if (l->items > 0) {
    l->items--;
  }
  PetscFunctionReturn(0);
}

#if 0 /* Currently unused */
#undef __FUNCT__
#define __FUNCT__ "ListHasItem"
/*  hasitem()    Search in this list to find if 'checkitem' exists.            */
/*                                                                             */
/*  This routine assumes that a linear order function has been set.  It loops  */
/*  through the entire list, compares each item to 'checkitem'. If it exists,  */
/*  return its position (between 0 to items - 1), otherwise, return -1.        */
/* tetgenmesh::list::hasitem() */
static PetscErrorCode ListHasItem(List *l, void *checkitem, int *idx)
{
  int i, id = -1;

  PetscFunctionBegin;
  for(i = 0; i < l->items; i++) {
    if (l->comp) {
      if ((*l->comp)((void *) (l->base + i * l->itembytes), checkitem) == 0) {
        id = i;
        break;
      }
    }
  }
  *idx = id;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "ListLength"
/* tetgenmesh::list::len() */
static PetscErrorCode ListLength(List *l, int *len)
{
  PetscFunctionBegin;
  *len = l->items;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListItem"
/* tetgenmesh::list::operator[]() */
static PetscErrorCode ListItem(List *l, int i, void **item)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* *item = l->base + i * l->itembytes; */
  ierr = PetscMemcpy(item, l->base + i * l->itembytes, l->itembytes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListSetItem"
/* tetgenmesh::list::operator[]() */
static PetscErrorCode ListSetItem(List *l, int i, void *item)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(l->base + i * l->itembytes, item, l->itembytes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListClear"
/* tetgenmesh::list::clear() */
static PetscErrorCode ListClear(List *l)
{
  PetscFunctionBegin;
  l->items = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListDestroy"
/* tetgenmesh::list::~list() */
static PetscErrorCode ListDestroy(List **l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*l) PetscFunctionReturn(0);
  ierr = PetscFree((*l)->base);CHKERRQ(ierr);
  ierr = PetscFree((*l));CHKERRQ(ierr);
  *l   = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QueueCreate"
/* tetgenmesh::queue::queue() */
static PetscErrorCode QueueCreate(int bytecount, int itemcount, Queue **newq)
{
  Queue         *q;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(Queue), &q);CHKERRQ(ierr);
  q->linkitembytes = bytecount;
  ierr = MemoryPoolCreate(bytecount + sizeof(void *), itemcount < 0 ? 256 : itemcount, POINTER, 0, &q->mp);CHKERRQ(ierr);
  ierr = MemoryPoolAlloc(q->mp, (void **) &q->head);CHKERRQ(ierr);
  ierr = MemoryPoolAlloc(q->mp, (void **) &q->tail);CHKERRQ(ierr);
  *q->head     = (void *) q->tail;
  *q->tail     = PETSC_NULL;
  q->linkitems = 0;
  *newq = q;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QueueLength"
/* tetgenmesh::queue::len() */
static PetscErrorCode QueueLength(Queue *q, int *len)
{
  PetscFunctionBegin;
  if (len) {*len = q->linkitems;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QueuePush"
/* tetgenmesh::queue::push() */
static PetscErrorCode QueuePush(Queue *q, void *newitem, void **next)
{
  void **newnode = q->tail;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (newitem) {
    ierr = PetscMemcpy(newnode + 1, newitem, q->linkitembytes);CHKERRQ(ierr);
  }
  ierr = MemoryPoolAlloc(q->mp, (void **) &q->tail);CHKERRQ(ierr);
  *q->tail = PETSC_NULL;
  *newnode = (void *) q->tail;
  q->linkitems++;
  if (next) {*next = (void *) (newnode + 1);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QueuePop"
/* tetgenmesh::queue::pop() */
static PetscErrorCode QueuePop(Queue *q, void **next)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (q->linkitems > 0) {
    void **deadnode = (void **) *q->head;
    *q->head = *deadnode;
    ierr = MemoryPoolDealloc(q->mp, (void *) deadnode);CHKERRQ(ierr);
    q->linkitems--;
    if (next) {*next = (void *) (deadnode + 1);}
  } else {
    if (next) {*next = PETSC_NULL;}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QueueDestroy"
/* tetgenmesh::queue::~queue() */
static PetscErrorCode QueueDestroy(Queue **q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*q) {PetscFunctionReturn(0);}
  ierr = MemoryPoolDestroy(&(*q)->mp);CHKERRQ(ierr);
  ierr = PetscFree(*q);CHKERRQ(ierr);
  *q = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolAlloc"
/* tetgenmesh::memorypool::alloc() */
static PetscErrorCode MemoryPoolAlloc(MemoryPool *m, void **item)
{
  void           *newitem;
  void          **newblock;
  PETSC_UINTPTR_T alignptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*  First check the linked list of dead items.  If the list is not */
  /*    empty, allocate an item from the list rather than a fresh one. */
  if (m->deaditemstack) {
    newitem = m->deaditemstack;                     /*  Take first item in list. */
    m->deaditemstack = * (void **) m->deaditemstack;
  } else {
    /*  Check if there are any free items left in the current block. */
    if (m->unallocateditems == 0) {
      /*  Check if another block must be allocated. */
      if (!*m->nowblock) {
        /*  Allocate a new block of items, pointed to by the previous block. */
        ierr = PetscMalloc(m->itemsperblock * m->itembytes + sizeof(void *) + m->alignbytes, &newblock);CHKERRQ(ierr);
        *m->nowblock = (void *) newblock;
        /*  The next block pointer is NULL. */
        *newblock = PETSC_NULL;
      }
      /*  Move to the new block. */
      m->nowblock = (void **) *m->nowblock;
      /*  Find the first item in the block. */
      /*    Increment by the size of (void *). */
      /*  alignptr = (unsigned long) (nowblock + 1); */
      alignptr = (PETSC_UINTPTR_T) (m->nowblock + 1);
      /*  Align the item on an `alignbytes'-byte boundary. */
      /*  nextitem = (void *) */
      /*    (alignptr + (unsigned long) alignbytes - */
      /*     (alignptr % (unsigned long) alignbytes)); */
      m->nextitem = (void *) (alignptr + (PETSC_UINTPTR_T) m->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->alignbytes));
      /*  There are lots of unallocated items left in this block. */
      m->unallocateditems = m->itemsperblock;
    }
    /*  Allocate a new item. */
    newitem = m->nextitem;
    /*  Advance `nextitem' pointer to next free item in block. */
    if (m->itemwordtype == POINTER) {
      m->nextitem = (void *) ((void **)     m->nextitem + m->itemwords);
    } else {
      m->nextitem = (void *) ((PetscReal *) m->nextitem + m->itemwords);
    }
    m->unallocateditems--;
    m->maxitems++;
  }
  m->items++;
  *item = newitem;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolDealloc"
/* tetgenmesh::memorypool::dealloc() */
static PetscErrorCode MemoryPoolDealloc(MemoryPool *m, void *dyingitem)
{
  PetscFunctionBegin;
  /*  Push freshly killed item onto stack. */
  *((void **) dyingitem) = m->deaditemstack;
  m->deaditemstack = dyingitem;
  m->items--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolTraversalInit"
/*  traversalinit()   Prepare to traverse the entire list of items.            */
/*                                                                             */
/*  This routine is used in conjunction with traverse().                       */
/* tetgenmesh::memorypool::traversalinit() */
static PetscErrorCode MemoryPoolTraversalInit(MemoryPool *m)
{
  PETSC_UINTPTR_T alignptr;

  PetscFunctionBegin;
  /*  Begin the traversal in the first block. */
  m->pathblock = m->firstblock;
  /*  Find the first item in the block.  Increment by the size of (void *). */
  alignptr = (PETSC_UINTPTR_T) (m->pathblock + 1);
  /*  Align with item on an `alignbytes'-byte boundary. */
  m->pathitem = (void *) (alignptr + (PETSC_UINTPTR_T) m->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->alignbytes));
  /*  Set the number of items left in the current block. */
  m->pathitemsleft = m->itemsperblock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolTraverse"
/*  traverse()   Find the next item in the list.                               */
/*                                                                             */
/*  This routine is used in conjunction with traversalinit().  Be forewarned   */
/*  that this routine successively returns all items in the list, including    */
/*  deallocated ones on the deaditemqueue. It's up to you to figure out which  */
/*  ones are actually dead.  It can usually be done more space-efficiently by  */
/*  a routine that knows something about the structure of the item.            */
/* tetgenmesh::memorypool::traverse() */
static PetscErrorCode MemoryPoolTraverse(MemoryPool *m, void **next)
{
  void           *newitem;
  PETSC_UINTPTR_T alignptr;

  PetscFunctionBegin;
  /*  Stop upon exhausting the list of items. */
  if (m->pathitem == m->nextitem) {
    *next = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  /*  Check whether any untraversed items remain in the current block. */
  if (m->pathitemsleft == 0) {
    /*  Find the next block. */
    m->pathblock = (void **) *m->pathblock;
    /*  Find the first item in the block.  Increment by the size of (void *). */
    alignptr = (PETSC_UINTPTR_T) (m->pathblock + 1);
    /*  Align with item on an `alignbytes'-byte boundary. */
    m->pathitem = (void *) (alignptr + (PETSC_UINTPTR_T) m->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->alignbytes));
    /*  Set the number of items left in the current block. */
    m->pathitemsleft = m->itemsperblock;
  }
  newitem = m->pathitem;
  /*  Find the next item in the block. */
  if (m->itemwordtype == POINTER) {
    m->pathitem = (void *) ((void **)     m->pathitem + m->itemwords);
  } else {
    m->pathitem = (void *) ((PetscReal *) m->pathitem + m->itemwords);
  }
  m->pathitemsleft--;
  *next = newitem;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolRestart"
/* tetgenmesh::memorypool::restart() */
static PetscErrorCode MemoryPoolRestart(MemoryPool *m)
{
  PETSC_UINTPTR_T alignptr;

  PetscFunctionBegin;
  m->items    = 0;
  m->maxitems = 0;
  /*  Set the currently active block. */
  m->nowblock = m->firstblock;
  /*  Find the first item in the pool.  Increment by the size of (void *). */
  /*  alignptr = (unsigned long) (nowblock + 1); */
  alignptr = (PETSC_UINTPTR_T) (m->nowblock + 1);
  /*  Align the item on an `alignbytes'-byte boundary. */
  /*  nextitem = (void *) */
  /*    (alignptr + (unsigned long) alignbytes - */
  /*     (alignptr % (unsigned long) alignbytes)); */
  m->nextitem = (void *) (alignptr + (PETSC_UINTPTR_T) m->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->alignbytes));
  /*  There are lots of unallocated items left in this block. */
  m->unallocateditems = m->itemsperblock;
  /*  The stack of deallocated items is empty. */
  m->deaditemstack = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolCreate"
/*  poolinit()    Initialize a pool of memory for allocation of items.         */
/*                                                                             */
/*  A `pool' is created whose records have size at least `bytecount'.  Items   */
/*  will be allocated in `itemcount'-item blocks.  Each item is assumed to be  */
/*  a collection of words, and either pointers or floating-point values are    */
/*  assumed to be the "primary" word type.  (The "primary" word type is used   */
/*  to determine alignment of items.)  If `alignment' isn't zero, all items    */
/*  will be `alignment'-byte aligned in memory.  `alignment' must be either a  */
/*  multiple or a factor of the primary word size;  powers of two are safe.    */
/*  `alignment' is normally used to create a few unused bits at the bottom of  */
/*  each item's pointer, in which information may be stored.                   */
/* tetgenmesh::memorypool::memorypool() and tetgenmesh::memorypool::poolinit() */
static PetscErrorCode MemoryPoolCreate(int bytecount, int itemcount, wordtype wtype, int alignment, MemoryPool **mp)
{
  MemoryPool    *m;
  int            wordsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(MemoryPool), &m);CHKERRQ(ierr);
  /*  Initialize values in the pool. */
  m->itemwordtype = wtype;
  wordsize        = (m->itemwordtype == POINTER) ? sizeof(void *) : sizeof(PetscReal);
  /*  Find the proper alignment, which must be at least as large as: */
  /*    - The parameter `alignment'. */
  /*    - The primary word type, to avoid unaligned accesses. */
  /*    - sizeof(void *), so the stack of dead items can be maintained */
  /*        without unaligned accesses. */
  if (alignment > wordsize) {
    m->alignbytes = alignment;
  } else {
    m->alignbytes = wordsize;
  }
  if ((int) sizeof(void *) > m->alignbytes) {
    m->alignbytes = (int) sizeof(void *);
  }
  m->itemwords = ((bytecount + m->alignbytes - 1) /  m->alignbytes) * (m->alignbytes / wordsize);
  m->itembytes = m->itemwords * wordsize;
  m->itemsperblock = itemcount;

  /*  Allocate a block of items.  Space for `itemsperblock' items and one */
  /*    pointer (to point to the next block) are allocated, as well as space */
  /*    to ensure alignment of the items. */
  ierr =  PetscMalloc(m->itemsperblock * m->itembytes + sizeof(void *) + m->alignbytes, &m->firstblock);CHKERRQ(ierr);
  /*  Set the next block pointer to NULL. */
  *(m->firstblock) = PETSC_NULL;
  ierr = MemoryPoolRestart(m);CHKERRQ(ierr);
  *mp = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolDestroy"
/* tetgenmesh::memorypool::~memorypool() */
static PetscErrorCode MemoryPoolDestroy(MemoryPool **m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*m) {PetscFunctionReturn(0);}
  while((*m)->firstblock) {
    (*m)->nowblock = (void **) *((*m)->firstblock);
    ierr = PetscFree((*m)->firstblock);CHKERRQ(ierr);
    (*m)->firstblock = (*m)->nowblock;
  }
  ierr = PetscFree(*m);CHKERRQ(ierr);
  *m   = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolRestart"
/*  restart()    Deallocate all objects in this pool.                          */
/*                                                                             */
/*  The pool returns to a fresh state, like after it was initialized, except   */
/*  that no memory is freed to the operating system.  Rather, the previously   */
/*  allocated blocks are ready to be used.                                     */
/* tetgenmesh::arraypool::restart() */
static PetscErrorCode ArrayPoolRestart(ArrayPool *a)
{
  PetscFunctionBegin;
  a->objects = 0l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolCreate"
/*  poolinit()    Initialize an arraypool for allocation of objects.           */
/*                                                                             */
/*  Before the pool may be used, it must be initialized by this procedure.     */
/*  After initialization, memory can be allocated and freed in this pool.      */
/* tetgenmesh::arraypool::arraypool() and tetgenmesh::arraypool::poolinit() */
static PetscErrorCode ArrayPoolCreate(int sizeofobject, int log2objperblk, ArrayPool **ap)
{
  ArrayPool     *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(ArrayPool), &a);CHKERRQ(ierr);
  /*  Each object must be at least one byte long. */
  a->objectbytes         = sizeofobject > 1 ? sizeofobject : 1;
  a->log2objectsperblock = log2objperblk;
  /*  Compute the number of objects in each block. */
  a->objectsperblock = ((int) 1) << a->log2objectsperblock;
  /*  No memory has been allocated. */
  a->totalmemory = 0l;
  /*  The top array has not been allocated yet. */
  a->toparray    = PETSC_NULL;
  a->toparraylen = 0;
  /*  Ready all indices to be allocated. */
  ierr = ArrayPoolRestart(a);CHKERRQ(ierr);
  *ap = a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolGetBlock"
/*  getblock()    Return (and perhaps create) the block containing the object  */
/*                with a given index.                                          */
/*                                                                             */
/*  This function takes care of allocating or resizing the top array if nece-  */
/*  ssary, and of allocating the block if it hasn't yet been allocated.        */
/*                                                                             */
/*  Return a pointer to the beginning of the block (NOT the object).           */
/* tetgenmesh::arraypool::getblock() */
static PetscErrorCode ArrayPoolGetBlock(ArrayPool *a, int objectindex, char **blk)
{
  char **newarray;
  char *block;
  int newsize;
  int topindex;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Compute the index in the top array (upper bits). */
  topindex = objectindex >> a->log2objectsperblock;
  /*  Does the top array need to be allocated or resized? */
  if (!a->toparray) {
    /*  Allocate the top array big enough to hold 'topindex', and NULL out its contents. */
    newsize = topindex + 128;
    ierr = PetscMalloc(newsize * sizeof(char *), &a->toparray);CHKERRQ(ierr);
    a->toparraylen = newsize;
    for(i = 0; i < newsize; i++) {
      a->toparray[i] = PETSC_NULL;
    }
    /*  Account for the memory. */
    a->totalmemory = newsize * (unsigned long) sizeof(char *);
  } else if (topindex >= a->toparraylen) {
    /*  Resize the top array, making sure it holds 'topindex'. */
    newsize = 3 * a->toparraylen;
    if (topindex >= newsize) {
      newsize = topindex + 128;
    }
    /*  Allocate the new array, copy the contents, NULL out the rest, and free the old array. */
    ierr = PetscMalloc(newsize * sizeof(char *), &newarray);CHKERRQ(ierr);
    for(i = 0; i < a->toparraylen; i++) {
      newarray[i] = a->toparray[i];
    }
    for(i = a->toparraylen; i < newsize; i++) {
      newarray[i] = PETSC_NULL;
    }
    ierr = PetscFree(a->toparray);CHKERRQ(ierr);
    /*  Account for the memory. */
    a->totalmemory += (newsize - a->toparraylen) * sizeof(char *);
    a->toparray     = newarray;
    a->toparraylen  = newsize;
  }
  /*  Find the block, or learn that it hasn't been allocated yet. */
  block = a->toparray[topindex];
  if (!block) {
    /*  Allocate a block at this index. */
    ierr = PetscMalloc(a->objectsperblock * a->objectbytes, &block);CHKERRQ(ierr);
    a->toparray[topindex] = block;
    /*  Account for the memory. */
    a->totalmemory += a->objectsperblock * a->objectbytes;
  }
  /*  Return a pointer to the block. */
  *blk = block;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolNewIndex"
/*  newindex()    Allocate space for a fresh object from the pool.             */
/* tetgenmesh::arraypool::newindex() */
static PetscErrorCode ArrayPoolNewIndex(ArrayPool *a, void **newptr, int *idx)
{
  char          *block = PETSC_NULL;
  void          *newobject;
  int            newindex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Allocate an object at index 'firstvirgin'. */
  ierr = ArrayPoolGetBlock(a, a->objects, &block);CHKERRQ(ierr);
  newindex  = a->objects;
  newobject = (void *) (block + (a->objects & (a->objectsperblock - 1)) * a->objectbytes);
  a->objects++;
  /*  If 'newptr' is not NULL, use it to return a pointer to the object. */
  if (newptr) {*newptr = newobject;}
  if (idx)    {*idx    = newindex;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolDestroy"
/* tetgenmesh::arraypool::~arraypool() */
static PetscErrorCode ArrayPoolDestroy(ArrayPool **a)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*a) {PetscFunctionReturn(0);}
  /*  Has anything been allocated at all? */
  if ((*a)->toparray) {
    /*  Walk through the top array. */
    for(i = 0; i < (*a)->toparraylen; ++i) {
      /*  Check every pointer; NULLs may be scattered randomly. */
      if ((*a)->toparray[i]) {
        /*  Free an allocated block. */
        ierr = PetscFree((*a)->toparray[i]);CHKERRQ(ierr);
      }
    }
    /*  Free the top array. */
    ierr = PetscFree((*a)->toparray);CHKERRQ(ierr);
  }
  /*  The top array is no longer allocated. */
  (*a)->toparray    = PETSC_NULL;
  (*a)->toparraylen = 0;
  (*a)->objects     = 0;
  (*a)->totalmemory = 0;
  *a                = PETSC_NULL;
  PetscFunctionReturn(0);
}

/*  prim_cxx ///////////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetNextSFace"
/*  getnextsface()    Finds the next subface in the face ring.                 */
/*                                                                             */
/*  For saving space in the data structure of subface, there only exists one   */
/*  face ring around a segment (see programming manual).  This routine imple-  */
/*  ments the double face ring as desired in Muecke's data structure.          */
/* tetgenmesh::getnextsface() */
PetscErrorCode TetGenMeshGetNextSFace(TetGenMesh *m, face* s1, face* s2)
{
  face neighsh = {PETSC_NULL, 0}, spinsh = {PETSC_NULL, 0};
  face testseg = {PETSC_NULL, 0};

  PetscFunctionBegin;
  sspivot(m, s1, &testseg);
  if (testseg.sh != m->dummysh) {
    testseg.shver = 0;
    if (sorg(&testseg) == sorg(s1)) {
      spivot(s1, &neighsh);
    } else {
      spinsh = *s1;
      do {
        neighsh = spinsh;
        spivotself(&spinsh);
      } while (spinsh.sh != s1->sh);
    }
  } else {
    spivot(s1, &neighsh);
  }
  if (sorg(&neighsh) != sorg(s1)) {
    sesymself(&neighsh);
  }
  if (s2) {
    *s2 = neighsh;
  } else {
    *s1 = neighsh;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFindEdge_triface"
/*  findedge()    Find an edge in the given tet or subface.                    */
/*                                                                             */
/*  The edge is given in two points 'eorg' and 'edest'.  It is assumed that    */
/*  the edge must exist in the given handle (tetrahedron or subface).  This    */
/*  routine sets the right edge version for the input handle.                  */
/* tetgenmesh::findedge() */
PetscErrorCode TetGenMeshFindEdge_triface(TetGenMesh *m, triface *tface, point eorg, point edest)
{
  PetscInt       i;

  PetscFunctionBegin;
  for(i = 0; i < 3; i++) {
    if (org(tface) == eorg) {
      if (dest(tface) == edest) {
        /*  Edge is found, return. */
        PetscFunctionReturn(0);
      }
    } else {
      if (org(tface) == edest) {
        if (dest(tface) == eorg) {
          /*  Edge is found, invert the direction and return. */
          esymself(tface);
          PetscFunctionReturn(0);
        }
      }
    }
    enextself(tface);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find an edge in tet");
}
#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFindEdge_face"
PetscErrorCode TetGenMeshFindEdge_face(TetGenMesh *m, face *sface, point eorg, point edest)
{
  PetscInt       i;

  PetscFunctionBegin;
  for(i = 0; i < 3; i++) {
    if (sorg(sface) == eorg) {
      if (sdest(sface) == edest) {
        /*  Edge is found, return. */
        PetscFunctionReturn(0);
      }
    } else {
      if (sorg(sface) == edest) {
        if (sdest(sface) == eorg) {
          /*  Edge is found, invert the direction and return. */
          sesymself(sface);
          PetscFunctionReturn(0);
        }
      }
    }
    senextself(sface);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find an edge in tet");
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTssPivot"
/*  tsspivot()    Finds a subsegment abutting on a tetrahderon's edge.         */
/*                                                                             */
/*  The edge is represented in the primary edge of 'checkedge'. If there is a  */
/*  subsegment bonded at this edge, it is returned in handle 'checkseg', the   */
/*  edge direction of 'checkseg' is conformed to 'checkedge'. If there isn't,  */
/*  set 'checkseg.sh = dummysh' to indicate it is not a subsegment.            */
/*                                                                             */
/*  To find whether an edge of a tetrahedron is a subsegment or not. First we  */
/*  need find a subface around this edge to see if it contains a subsegment.   */
/*  The reason is there is no direct connection between a tetrahedron and its  */
/*  adjoining subsegments.                                                     */
/* tetgenmesh::tsspivot() */
PetscErrorCode TetGenMeshTssPivot(TetGenMesh *m, triface* checkedge, face* checkseg)
{
  triface spintet  = {PETSC_NULL, 0, 0};
  face    parentsh = {PETSC_NULL, 0};
  point tapex;
  int hitbdry;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  spintet = *checkedge;
  tapex = apex(checkedge);
  hitbdry = 0;
  do {
    tspivot(m, &spintet, &parentsh);
    /*  Does spintet have a (non-fake) subface attached? */
    if ((parentsh.sh != m->dummysh) && (sapex(&parentsh))) {
      /*  Find a subface! Find the edge in it. */
      ierr = TetGenMeshFindEdge_face(m, &parentsh, org(checkedge), dest(checkedge));CHKERRQ(ierr);
      sspivot(m, &parentsh, checkseg);
      if (checkseg->sh != m->dummysh) {
        /*  Find a subsegment! Correct its edge direction before return. */
        if (sorg(checkseg) != org(checkedge)) {
          sesymself(checkseg);
        }
      }
      PetscFunctionReturn(0);
    }
    if (!fnextself(m, &spintet)) {
      hitbdry++;
      if (hitbdry < 2) {
        esym(checkedge, &spintet);
        if (!fnextself(m, &spintet)) {
          hitbdry++;
        }
      }
    }
  } while ((apex(&spintet) != tapex) && (hitbdry < 2));
  /*  Not find. */
  checkseg->sh = m->dummysh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSstPivot"
/*  sstpivot()    Finds a tetrahedron abutting a subsegment.                   */
/*                                                                             */
/*  This is the inverse operation of 'tsspivot()'.  One subsegment shared by   */
/*  arbitrary number of tetrahedron, the returned tetrahedron is not unique.   */
/*  The edge direction of the returned tetrahedron is conformed to the given   */
/*  subsegment.                                                                */
/* tetgenmesh::sstpivot() */
PetscErrorCode TetGenMeshSstPivot(TetGenMesh *m, face* checkseg, triface* retedge)
{
  face parentsh = {PETSC_NULL, 0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Get the subface which holds the subsegment. */
  sdecode(checkseg->sh[0], &parentsh);
#ifdef PETSC_USE_DEBUG
    if (parentsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Shell edge should not be null");
#endif
  /*  Get a tetraheron to which the subface attches. */
  stpivot(m, &parentsh, retedge);
  if (retedge->tet == m->dummytet) {
    sesymself(&parentsh);
    stpivot(m, &parentsh, retedge);
#ifdef PETSC_USE_DEBUG
    if (retedge->tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Tet should not be null");
#endif
  }
  /*  Correct the edge direction before return. */
  ierr = TetGenMeshFindEdge_triface(m, retedge, sorg(checkseg), sdest(checkseg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPoint2TetOrg"
/*  point2tetorg(), point2shorg(), point2segorg()                              */
/*                                                                             */
/*  Return a tet, a subface, or a subsegment whose origin is the given point.  */
/*  These routines assume the maps between points to tets (subfaces, segments  */
/*  ) have been built and maintained.                                          */
/* tetgenmesh::point2tetorg() */
PetscErrorCode TetGenMeshPoint2TetOrg(TetGenMesh *m, point pa, triface *searchtet)
{
  int i;

  PetscFunctionBegin;
  /*  Search a tet whose origin is pa. */
  decode(point2tet(m, pa), searchtet);
  if (!searchtet->tet) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad tet pointer.\n", pointmark(m, pa));
  for(i = 4; i < 8; i++) {
    if ((point) searchtet->tet[i] == pa) {
      /*  Found. Set pa as its origin. */
      switch (i) {
        case 4: searchtet->loc = 0; searchtet->ver = 0; break;
        case 5: searchtet->loc = 0; searchtet->ver = 2; break;
        case 6: searchtet->loc = 0; searchtet->ver = 4; break;
        case 7: searchtet->loc = 1; searchtet->ver = 2; break;
      }
      break;
    }
  }
  if (i == 8) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad tet pointer.\n", pointmark(m, pa));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPoint2ShOrg"
/*  point2tetorg(), point2shorg(), point2segorg()                              */
/*                                                                             */
/*  Return a tet, a subface, or a subsegment whose origin is the given point.  */
/*  These routines assume the maps between points to tets (subfaces, segments  */
/*  ) have been built and maintained.                                          */
/* tetgenmesh::point2shorg() */
PetscErrorCode TetGenMeshPoint2ShOrg(TetGenMesh *m, point pa, face *searchsh)
{
  PetscFunctionBegin;
  sdecode(point2sh(m, pa), searchsh);
  if (!searchsh->sh) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad sub pointer.\n", pointmark(m, pa));
  if (((point) searchsh->sh[3]) == pa) {
    searchsh->shver = 0;
  } else if (((point) searchsh->sh[4]) == pa) {
    searchsh->shver = 2;
  } else if (((point) searchsh->sh[5]) == pa) {
    searchsh->shver = 4;
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad sub pointer.\n", pointmark(m, pa));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPoint2SegOrg"
/*  point2tetorg(), point2shorg(), point2segorg()                              */
/*                                                                             */
/*  Return a tet, a subface, or a subsegment whose origin is the given point.  */
/*  These routines assume the maps between points to tets (subfaces, segments  */
/*  ) have been built and maintained.                                          */
/* tetgenmesh::point2segorg() */
PetscErrorCode TetGenMeshPoint2SegOrg(TetGenMesh *m, point pa, face *searchsh)
{
  PetscFunctionBegin;
  sdecode(point2seg(m, pa), searchsh);
  if (!searchsh->sh) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad seg pointer.\n", pointmark(m, pa));
  if (((point) searchsh->sh[3]) == pa) {
    searchsh->shver = 0;
  } else if (((point) searchsh->sh[4]) == pa) {
    searchsh->shver = 1;
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad seg pointer.\n", pointmark(m, pa));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetSubsegFarOrg"
/*  getsubsegfarorg()    Get the origin of the parent segment of a subseg.     */
/* tetgenmesh::getsubsegfarorg() */
PetscErrorCode TetGenMeshGetSubsegFarOrg(TetGenMesh *m, face *sseg, point *p)
{
  face prevseg = {PETSC_NULL, 0};
  point checkpt;

  PetscFunctionBegin;
  checkpt = sorg(sseg);
  senext2(sseg, &prevseg);
  spivotself(&prevseg);
  /*  Search dorg along the original direction of sseg. */
  while(prevseg.sh != m->dummysh) {
    prevseg.shver = 0;
    if (sdest(&prevseg) != checkpt) sesymself(&prevseg);
    checkpt = sorg(&prevseg);
    senext2self(&prevseg);
    spivotself(&prevseg);
  }
  if (p) {*p = checkpt;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetSubsegFarDest"
/*  getsubsegfardest()    Get the dest. of the parent segment of a subseg.     */
/* tetgenmesh::getsubsegfardest() */
PetscErrorCode TetGenMeshGetSubsegFarDest(TetGenMesh *m, face *sseg, point *p)
{
  face nextseg = {PETSC_NULL, 0};
  point checkpt;

  PetscFunctionBegin;
  checkpt = sdest(sseg);
  senext(sseg, &nextseg);
  spivotself(&nextseg);
  /*  Search dorg along the destinational direction of sseg. */
  while (nextseg.sh != m->dummysh) {
    nextseg.shver = 0;
    if (sorg(&nextseg) != checkpt) sesymself(&nextseg);
    checkpt = sdest(&nextseg);
    senextself(&nextseg);
    spivotself(&nextseg);
  }
  if (p) {*p = checkpt;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPrintTet"
/* tetgenmesh::printtet() */
PetscErrorCode TetGenMeshPrintTet(TetGenMesh *m, triface *tface, PetscBool showPointer)
{
  TetGenOpts    *b = m->b;
  triface        tmpface = {PETSC_NULL, 0, 0}, prtface = {PETSC_NULL, 0, 0};
  shellface     *shells;
  point          tmppt;
  face           checksh = {PETSC_NULL, 0};
  int            facecount;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (showPointer) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Tetra x%lx with loc(%i) and ver(%i):", (PETSC_UINTPTR_T) tface->tet, tface->loc, tface->ver);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Tetra with loc(%i) and ver(%i):", tface->loc, tface->ver);CHKERRQ(ierr);
  }
  if (infected(m, tface)) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " (infected)");CHKERRQ(ierr);
  }
  if (marktested(m, tface)) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " (marked)");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);

  tmpface = *tface;
  facecount = 0;
  while(facecount < 4) {
    tmpface.loc = facecount;
    sym(&tmpface, &prtface);
    if (prtface.tet == m->dummytet) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [%i] Outer space.\n", facecount);CHKERRQ(ierr);
    } else {
      if (!isdead_triface(&prtface)) {
        if (showPointer) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      [%i] x%lx  loc(%i).", facecount, (PETSC_UINTPTR_T) prtface.tet, prtface.loc);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      [%i] loc(%i).", facecount, prtface.loc);CHKERRQ(ierr);
        }
        if (infected(m, &prtface)) {
          ierr = PetscPrintf(PETSC_COMM_SELF, " (infected)");CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [%i] NULL\n", facecount);CHKERRQ(ierr);
      }
    }
    facecount ++;
  }

  tmppt = org(tface);
  if (!tmppt) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Org [%i] NULL\n", locver2org[tface->loc][tface->ver]);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Org [%i] (%.12g,%.12g,%.12g) %d\n",
                       locver2org[tface->loc][tface->ver], tmppt[0], tmppt[1], tmppt[2], pointmark(m, tmppt));CHKERRQ(ierr);
  }
  tmppt = dest(tface);
  if(tmppt == (point) NULL) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Dest[%i] NULL\n", locver2dest[tface->loc][tface->ver]);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Dest[%i] (%.12g,%.12g,%.12g) %d\n",
                       locver2dest[tface->loc][tface->ver], tmppt[0], tmppt[1], tmppt[2], pointmark(m, tmppt));CHKERRQ(ierr);
  }
  tmppt = apex(tface);
  if (!tmppt) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Apex[%i] NULL\n", locver2apex[tface->loc][tface->ver]);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Apex[%i] (%.12g,%.12g,%.12g) %d\n",
                       locver2apex[tface->loc][tface->ver], tmppt[0], tmppt[1], tmppt[2], pointmark(m, tmppt));CHKERRQ(ierr);
  }
  tmppt = oppo(tface);
  if (!tmppt) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Oppo[%i] NULL\n", loc2oppo[tface->loc]);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Oppo[%i] (%.12g,%.12g,%.12g) %d\n",
                       loc2oppo[tface->loc], tmppt[0], tmppt[1], tmppt[2], pointmark(m, tmppt));CHKERRQ(ierr);
  }

  if (b->useshelles) {
    if (tface->tet[8]) {
      shells = (shellface *) tface->tet[8];
      for (facecount = 0; facecount < 6; facecount++) {
        sdecode(shells[facecount], &checksh);
        if (checksh.sh != m->dummysh) {
          if (showPointer) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] x%lx %d.", facecount, (PETSC_UINTPTR_T) checksh.sh, checksh.shver);CHKERRQ(ierr);
          } else {
            ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] %d.", facecount, checksh.shver);CHKERRQ(ierr);
          }
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] NULL.", facecount);CHKERRQ(ierr);
        }
        if (locver2edge[tface->loc][tface->ver] == facecount) {
          ierr = PetscPrintf(PETSC_COMM_SELF, " (*)");CHKERRQ(ierr);  /*  It is the current edge. */
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }
    }
    if (tface->tet[9]) {
      shells = (shellface *) tface->tet[9];
      for (facecount = 0; facecount < 4; facecount++) {
        sdecode(shells[facecount], &checksh);
        if (checksh.sh != m->dummysh) {
          if (showPointer) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] x%lx %d.", facecount, (PETSC_UINTPTR_T) checksh.sh, checksh.shver);CHKERRQ(ierr);
          } else {
            ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] %d.", facecount, checksh.shver);CHKERRQ(ierr);
          }
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] NULL.", facecount);CHKERRQ(ierr);
        }
        if (tface->loc == facecount) {
          ierr = PetscPrintf(PETSC_COMM_SELF, " (*)");CHKERRQ(ierr);  /*  It is the current face. */
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPrintSh"
/*  printsh()    Print out the details of a subface or subsegment on screen.   */
/*                                                                             */
/*  It's also used when the highest level of verbosity (`-VVV') is specified.  */
/* tetgenmesh::printsh() */
PetscErrorCode TetGenMeshPrintSh(TetGenMesh *m, face *sface, PetscBool showPointer)
{
  face prtsh = {PETSC_NULL, 0};
  triface prttet = {PETSC_NULL, 0, 0};
  point printpoint;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sapex(sface)) {
    if (showPointer) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "subface x%lx, ver %d, mark %d:", (PETSC_UINTPTR_T) (sface->sh), sface->shver, shellmark(m, sface));CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF, "subface ver %d, mark %d:", sface->shver, shellmark(m, sface));CHKERRQ(ierr);
    }
  } else {
    if (showPointer) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "Subsegment x%lx, ver %d, mark %d:", (PETSC_UINTPTR_T) (sface->sh), sface->shver, shellmark(m, sface));CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF, "Subsegment ver %d, mark %d:", sface->shver, shellmark(m, sface));CHKERRQ(ierr);
    }
  }
  if (sinfected(m, sface)) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " (infected)");CHKERRQ(ierr);
  }
  if (smarktested(sface)) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " (marked)");CHKERRQ(ierr);
  }
  if (shell2badface(sface)) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " (queued)");CHKERRQ(ierr);
  }
  if (sapex(sface)) {
    if (shelltype(m, sface) == SHARP) {
      ierr = PetscPrintf(PETSC_COMM_SELF, " (sharp)");CHKERRQ(ierr);
    }
  } else {
    if (shelltype(m, sface) == SHARP) {
      ierr = PetscPrintf(PETSC_COMM_SELF, " (sharp)");CHKERRQ(ierr);
    }
  }
  if (m->checkpbcs) {
    if (shellpbcgroup(m, sface) >= 0) {
      ierr = PetscPrintf(PETSC_COMM_SELF, " (pbc %d)", shellpbcgroup(m, sface));CHKERRQ(ierr);
    }
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);

  sdecode(sface->sh[0], &prtsh);
  if (prtsh.sh == m->dummysh) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      [0] = No shell\n");CHKERRQ(ierr);
  } else {
    if (showPointer) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [0] = x%lx  %d\n", (PETSC_UINTPTR_T)(prtsh.sh), prtsh.shver);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [0] = %d\n", prtsh.shver);CHKERRQ(ierr);
    }
  }
  sdecode(sface->sh[1], &prtsh);
  if (prtsh.sh == m->dummysh) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      [1] = No shell\n");CHKERRQ(ierr);
  } else {
    if (showPointer) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [1] = x%lx  %d\n", (PETSC_UINTPTR_T)(prtsh.sh), prtsh.shver);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [1] =  %d\n", prtsh.shver);CHKERRQ(ierr);
    }
  }
  sdecode(sface->sh[2], &prtsh);
  if (prtsh.sh == m->dummysh) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      [2] = No shell\n");CHKERRQ(ierr);
  } else {
    if (showPointer) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [2] = x%lx  %d\n", (PETSC_UINTPTR_T)(prtsh.sh), prtsh.shver);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [2] =  %d\n", prtsh.shver);CHKERRQ(ierr);
    }
  }

  printpoint = sorg(sface);
  if (!printpoint) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Org [%d] = NULL\n", vo[sface->shver]);CHKERRQ(ierr);
  } else {
    if (showPointer) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      Org [%d] = x%lx  (%.12g,%.12g,%.12g) %d\n", vo[sface->shver], (PETSC_UINTPTR_T)(printpoint), printpoint[0], printpoint[1], printpoint[2], pointmark(m, printpoint));CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      Org [%d] = (%.12g,%.12g,%.12g) %d\n", vo[sface->shver], printpoint[0], printpoint[1], printpoint[2], pointmark(m, printpoint));CHKERRQ(ierr);
    }
  }
  printpoint = sdest(sface);
  if (!printpoint) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Dest[%d] = NULL\n", vd[sface->shver]);CHKERRQ(ierr);
  } else {
    if (showPointer) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      Dest[%d] = x%lx  (%.12g,%.12g,%.12g) %d\n", vd[sface->shver], (PETSC_UINTPTR_T)(printpoint), printpoint[0], printpoint[1], printpoint[2], pointmark(m, printpoint));CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      Dest[%d] = (%.12g,%.12g,%.12g) %d\n", vd[sface->shver], printpoint[0], printpoint[1], printpoint[2], pointmark(m, printpoint));CHKERRQ(ierr);
    }
  }

  if (sapex(sface)) {
    printpoint = sapex(sface);
    if (!printpoint) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      Apex[%d] = NULL\n", va[sface->shver]);CHKERRQ(ierr);
    } else {
      if (showPointer) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      Apex[%d] = x%lx  (%.12g,%.12g,%.12g) %d\n", va[sface->shver], (PETSC_UINTPTR_T)(printpoint), printpoint[0], printpoint[1], printpoint[2], pointmark(m, printpoint));CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      Apex[%d] = (%.12g,%.12g,%.12g) %d\n", va[sface->shver], printpoint[0], printpoint[1], printpoint[2], pointmark(m, printpoint));CHKERRQ(ierr);
      }
    }
    decode(sface->sh[6], &prttet);
    if (prttet.tet == m->dummytet) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [6] = Outer space\n");CHKERRQ(ierr);
    } else {
      if (showPointer) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [6] = x%lx  %d\n", (PETSC_UINTPTR_T)(prttet.tet), prttet.loc);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [6] = %d\n", prttet.loc);CHKERRQ(ierr);
      }
    }
    decode(sface->sh[7], &prttet);
    if (prttet.tet == m->dummytet) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [7] = Outer space\n");CHKERRQ(ierr);
    } else {
      if (showPointer) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [7] = x%lx  %d\n", (PETSC_UINTPTR_T)(prttet.tet), prttet.loc);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [7] = %d\n", prttet.loc);CHKERRQ(ierr);
      }
    }

    sdecode(sface->sh[8], &prtsh);
    if (prtsh.sh == m->dummysh) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [8] = No subsegment\n");CHKERRQ(ierr);
    } else {
      if (showPointer) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [8] = x%lx  %d\n", (PETSC_UINTPTR_T)(prtsh.sh), prtsh.shver);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [8] = %d\n", prtsh.shver);CHKERRQ(ierr);
      }
    }
    sdecode(sface->sh[9], &prtsh);
    if (prtsh.sh == m->dummysh) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [9] = No subsegment\n");CHKERRQ(ierr);
    } else {
      if (showPointer) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [9] = x%lx  %d\n", (PETSC_UINTPTR_T)(prtsh.sh), prtsh.shver);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [9] = %d\n", prtsh.shver);CHKERRQ(ierr);
      }
    }
    sdecode(sface->sh[10], &prtsh);
    if (prtsh.sh == m->dummysh) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [10]= No subsegment\n");CHKERRQ(ierr);
    } else {
      if (showPointer) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [10]= x%lx  %d\n", (PETSC_UINTPTR_T)(prtsh.sh), prtsh.shver);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [10]= %d\n", prtsh.shver);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRandomChoice"
/* tetgenmesh::randomnation() */
PetscErrorCode TetGenMeshRandomChoice(TetGenMesh *m, unsigned int choices, int *choice)
{
  unsigned long newrandom;

  PetscFunctionBegin;
  if (choices >= 714025l) {
    newrandom     = (m->randomseed * 1366l + 150889l) % 714025l;
    m->randomseed = (newrandom * 1366l + 150889l) % 714025l;
    newrandom     = newrandom * (choices / 714025l) + m->randomseed;
    if (newrandom >= choices) {
      newrandom -= choices;
    }
  } else {
    m->randomseed = (m->randomseed * 1366l + 150889l) % 714025l;
    newrandom = m->randomseed % choices;
  }
  *choice = newrandom;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDistance2"
/*  distance2()    Returns the square "distance" of a tetrahedron to point p.  */
/* tetgenmesh::distance2() */
PetscErrorCode TetGenMeshDistance2(TetGenMesh *m, tetrahedron *tetptr, point p, PetscReal *dist)
{
  point p1, p2, p3, p4;
  PetscReal dx, dy, dz;

  PetscFunctionBegin;
  p1 = (point) tetptr[4];
  p2 = (point) tetptr[5];
  p3 = (point) tetptr[6];
  p4 = (point) tetptr[7];

  dx = p[0] - 0.25 * (p1[0] + p2[0] + p3[0] + p4[0]);
  dy = p[1] - 0.25 * (p1[1] + p2[1] + p3[1] + p4[1]);
  dz = p[2] - 0.25 * (p1[2] + p2[2] + p3[2] + p4[2]);

  *dist = dx * dx + dy * dy + dz * dz;
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  prim_cxx ///////////////////////////////////////////////////////////////// */

/*  mempool_cxx ////////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDummyInit"
/*  dummyinit()    Initialize the tetrahedron that fills "outer space" and     */
/*                 the omnipresent subface.                                    */
/*                                                                             */
/*  The tetrahedron that fills "outer space" called 'dummytet', is pointed to  */
/*  by every tetrahedron and subface on a boundary (be it outer or inner) of   */
/*  the tetrahedralization. Also, 'dummytet' points to one of the tetrahedron  */
/*  on the convex hull(until the holes and concavities are carved), making it  */
/*  possible to find a starting tetrahedron for point location.                */
/*                                                                             */
/*  The omnipresent subface,'dummysh', is pointed to by every tetrahedron or   */
/*  subface that doesn't have a full complement of real subface to point to.   */
/* tetgenmesh::dummyinit() */
PetscErrorCode TetGenMeshDummyInit(TetGenMesh *m, int tetwords, int shwords)
{
  TetGenOpts    *b = m->b;
  unsigned long  alignptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Set up 'dummytet', the 'tetrahedron' that occupies "outer space". */
  ierr = PetscMalloc(tetwords * sizeof(tetrahedron) + m->tetrahedrons->alignbytes, &m->dummytetbase);CHKERRQ(ierr);
  /*  Align 'dummytet' on a 'tetrahedrons->alignbytes'-byte boundary. */
  alignptr = (unsigned long) m->dummytetbase;
  m->dummytet = (tetrahedron *) (alignptr + (unsigned long) m->tetrahedrons->alignbytes
                                 - (alignptr % (unsigned long) m->tetrahedrons->alignbytes));
  /*  Initialize the four adjoining tetrahedra to be "outer space". These */
  /*    will eventually be changed by various bonding operations, but their */
  /*    values don't really matter, as long as they can legally be */
  /*    dereferenced. */
  m->dummytet[0] = (tetrahedron) m->dummytet;
  m->dummytet[1] = (tetrahedron) m->dummytet;
  m->dummytet[2] = (tetrahedron) m->dummytet;
  m->dummytet[3] = (tetrahedron) m->dummytet;
  /*  Four null vertex points. */
  m->dummytet[4] = PETSC_NULL;
  m->dummytet[5] = PETSC_NULL;
  m->dummytet[6] = PETSC_NULL;
  m->dummytet[7] = PETSC_NULL;

  if (b->useshelles) {
    /*  Set up 'dummysh', the omnipresent "subface" pointed to by any */
    /*    tetrahedron side or subface end that isn't attached to a real */
    /*    subface. */
    ierr = PetscMalloc(shwords * sizeof(shellface) + m->subfaces->alignbytes, &m->dummyshbase);CHKERRQ(ierr);
    /*  Align 'dummysh' on a 'subfaces->alignbytes'-byte boundary. */
    alignptr = (unsigned long) m->dummyshbase;
    m->dummysh = (shellface *) (alignptr + (unsigned long) m->subfaces->alignbytes
                                - (alignptr % (unsigned long) m->subfaces->alignbytes));
    /*  Initialize the three adjoining subfaces to be the omnipresent */
    /*    subface. These will eventually be changed by various bonding */
    /*    operations, but their values don't really matter, as long as they */
    /*    can legally be dereferenced. */
    m->dummysh[0] = (shellface) m->dummysh;
    m->dummysh[1] = (shellface) m->dummysh;
    m->dummysh[2] = (shellface) m->dummysh;
    /*  Three null vertex points. */
    m->dummysh[3] = PETSC_NULL;
    m->dummysh[4] = PETSC_NULL;
    m->dummysh[5] = PETSC_NULL;
    /*  Initialize the two adjoining tetrahedra to be "outer space". */
    m->dummysh[6] = (shellface) m->dummytet;
    m->dummysh[7] = (shellface) m->dummytet;
    /*  Initialize the three adjoining subsegments to be "out boundary". */
    m->dummysh[8]  = (shellface) m->dummysh;
    m->dummysh[9]  = (shellface) m->dummysh;
    m->dummysh[10] = (shellface) m->dummysh;
    /*  Initialize the pointer to badface structure. */
    m->dummysh[11] = PETSC_NULL;
    /*  Initialize the four adjoining subfaces of 'dummytet' to be the */
    /*    omnipresent subface. */
    m->dummytet[8 ] = PETSC_NULL;
    m->dummytet[9 ] = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInitializePools"
/*  initializepools()    Calculate the sizes of the point, tetrahedron, and    */
/*                       subface. Initialize their memory pools.               */
/*                                                                             */
/*  This routine also computes the indices 'pointmarkindex', 'point2simindex', */
/*  and 'point2pbcptindex' used to find values within each point;  computes    */
/*  indices 'highorderindex', 'elemattribindex', and 'volumeboundindex' used   */
/*  to find values within each tetrahedron.                                    */
/*                                                                             */
/*  There are two types of boundary elements, which are subfaces and subsegs,  */
/*  they are stored in seperate pools. However, the data structures of them    */
/*  are the same.  A subsegment can be regarded as a degenerate subface, i.e., */
/*  one of its three corners is not used. We set the apex of it be 'NULL' to   */
/*  distinguish it's a subsegment.                                             */
/* tetgenmesh::initializepools() */
PetscErrorCode TetGenMeshInitializePools(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  wordtype       wtype;
  int            pointsize, elesize, shsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Default checkpbc = 0; */
  if ((b->plc || b->refine) && (in->pbcgrouplist)) {
    m->checkpbcs = 1;
  }
  /*  Default varconstraint = 0; */
  if (in->segmentconstraintlist || in->facetconstraintlist) {
    m->varconstraint = 1;
  }

  /*  The index within each point at which its metric tensor is found. It is */
  /*    saved directly after the list of point attributes. */
  m->pointmtrindex = 3 + in->numberofpointattributes;
  /*  Decide the size (1, 3, or 6) of the metric tensor. */
  if (b->metric) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    /*  For '-m' option. A tensor field is provided (*.mtr or *.b.mtr file). */
    if (bgm) {
      /*  A background mesh is allocated. It may not exist though. */
      sizeoftensor = bgm->in ? bgm->in->numberofpointmtrs : in->numberofpointmtrs;
    } else {
      /*  No given background mesh - Itself is a background mesh. */
      sizeoftensor = in->numberofpointmtrs;
    }
    /*  Make sure sizeoftensor is at least 1. */
    sizeoftensor = (sizeoftensor > 0) ? sizeoftensor : 1;
#endif
  } else {
    /*  For '-q' option. Make sure to have space for saving a scalar value. */
    m->sizeoftensor = b->quality ? 1 : 0;
  }
  /*  The index within each point at which an element pointer is found, where */
  /*    the index is measured in pointers. Ensure the index is aligned to a */
  /*    sizeof(tetrahedron)-byte address. */
  m->point2simindex = ((m->pointmtrindex + m->sizeoftensor) * sizeof(PetscReal) + sizeof(tetrahedron) - 1) / sizeof(tetrahedron);
  if (b->plc || b->refine || b->voroout) {
    /*  Increase the point size by four pointers, which are: */
    /*    - a pointer to a tet, read by point2tet(); */
    /*    - a pointer to a subface, read by point2sh(); */
    /*    - a pointer to a subsegment, read by point2seg(); */
    /*    - a pointer to a parent point, read by point2ppt()). */
    if (b->metric) {
      /*  Increase one pointer to a tet of the background mesh. */
      pointsize = (m->point2simindex + 5) * sizeof(tetrahedron);
    } else {
      pointsize = (m->point2simindex + 4) * sizeof(tetrahedron);
    }
    /*  The index within each point at which a pbc point is found. */
    m->point2pbcptindex = (pointsize + sizeof(tetrahedron) - 1) / sizeof(tetrahedron);
    if (m->checkpbcs) {
      /*  Increase the size by one pointer to a corresponding pbc point, */
      /*    read by point2pbcpt(). */
      pointsize = (m->point2pbcptindex + 1) * sizeof(tetrahedron);
    }
  } else {
    /*  Increase the point size by FOUR pointer, which are: */
    /*    - a pointer to a tet, read by point2tet(); */
    /*    - a pointer to a subface, read by point2sh(); -- !! Unused !! */
    /*    - a pointer to a subsegment, read by point2seg(); -- !! Unused !! */
    /*    - a pointer to a parent point, read by point2ppt()). -- Used by btree. */
    pointsize = (m->point2simindex + 4) * sizeof(tetrahedron);
  }
  /*  The index within each point at which the boundary marker is found, */
  /*    Ensure the point marker is aligned to a sizeof(int)-byte address. */
  m->pointmarkindex = (pointsize + sizeof(int) - 1) / sizeof(int);
  /*  Now point size is the ints (inidcated by pointmarkindex) plus: */
  /*    - an integer for boundary marker; */
  /*    - an integer for vertex type; */
  /* pointsize = (pointmarkindex + 2) * sizeof(int);  Wrong for 64 bit. */
  pointsize = (m->pointmarkindex + 2) * sizeof(tetrahedron);
  /*  Decide the wordtype used in vertex pool. */
  wtype = (sizeof(PetscReal) >= sizeof(tetrahedron)) ? FLOATINGPOINT : POINTER;
  /*  Initialize the pool of vertices. */
  ierr = MemoryPoolCreate(pointsize, VERPERBLOCK, wtype, 0, &m->points);CHKERRQ(ierr);

  if (b->useshelles) { /* For abovepoint() */
    ierr = PetscMalloc(pointsize, &m->dummypoint);CHKERRQ(ierr);
  }

  /*  The number of bytes occupied by a tetrahedron.  There are four pointers */
  /*    to other tetrahedra, four pointers to corners, and possibly four */
  /*    pointers to subfaces (or six pointers to subsegments (used in */
  /*    segment recovery only)). */
  elesize = (8 + b->useshelles * 2) * sizeof(tetrahedron);
  /*  If Voronoi diagram is wanted, make sure we have additional space. */
  if (b->voroout) {
    elesize = (8 + 4) * sizeof(tetrahedron);
  }
  /*  The index within each element at which its attributes are found, where */
  /*    the index is measured in PetscReals. */
  m->elemattribindex = (elesize + sizeof(PetscReal) - 1) / sizeof(PetscReal);
  /*  The index within each element at which the maximum voulme bound is */
  /*    found, where the index is measured in PetscReals.  Note that if the */
  /*    `b->regionattrib' flag is set, an additional attribute will be added. */
  m->volumeboundindex = m->elemattribindex + in->numberoftetrahedronattributes + (b->regionattrib > 0);
  /*  If element attributes or an constraint are needed, increase the number */
  /*    of bytes occupied by an element. */
  if (b->varvolume) {
    elesize = (m->volumeboundindex + 1) * sizeof(PetscReal);
  } else if (in->numberoftetrahedronattributes + b->regionattrib > 0) {
    elesize = m->volumeboundindex * sizeof(PetscReal);
  }
  /*  If element neighbor graph is requested (-n switch), an additional */
  /*    integer is allocated for each element. */
  /*  elemmarkerindex = (elesize + sizeof(int) - 1) / sizeof(int); */
  m->elemmarkerindex = (elesize + sizeof(int) - 1) / sizeof(int);
  /*  if (b->neighout || b->voroout) { */
    /*  elesize = (elemmarkerindex + 1) * sizeof(int); */
    /*  Allocate one slot for the element marker. The actual need isa size */
    /*    of an integer. We allocate enough space (a pointer) for alignment */
    /*    for 64 bit system. Thanks Liu Yang (LORIA/INRIA) for reporting */
    /*    this problem. */
    elesize = elesize + sizeof(tetrahedron);
  /*  } */
  /*  If -o2 switch is used, an additional pointer pointed to the list of */
  /*    higher order nodes is allocated for each element. */
  m->highorderindex = (elesize + sizeof(tetrahedron) - 1) / sizeof(tetrahedron);
  if (b->order == 2) {
    elesize = (m->highorderindex + 1) * sizeof(tetrahedron);
  }
  /*  Having determined the memory size of an element, initialize the pool. */
  ierr = MemoryPoolCreate(elesize, ELEPERBLOCK, POINTER, 8, &m->tetrahedrons);CHKERRQ(ierr);

  if (b->useshelles) {
    /*  The number of bytes occupied by a subface.  The list of pointers */
    /*    stored in a subface are: three to other subfaces, three to corners, */
    /*    three to subsegments, two to tetrahedra, and one to a badface. */
    shsize = 12 * sizeof(shellface);
    /*  The index within each subface at which the maximum area bound is */
    /*    found, where the index is measured in PetscReals. */
    m->areaboundindex = (shsize + sizeof(PetscReal) - 1) / sizeof(PetscReal);
    /*  If -q switch is in use, increase the number of bytes occupied by */
    /*    a subface for saving maximum area bound. */
    if (b->quality && m->varconstraint) {
      shsize = (m->areaboundindex + 1) * sizeof(PetscReal);
    } else {
      shsize = m->areaboundindex * sizeof(PetscReal);
    }
    /*  The index within subface at which the facet marker is found. Ensure */
    /*    the marker is aligned to a sizeof(int)-byte address. */
    m->shmarkindex = (shsize + sizeof(int) - 1) / sizeof(int);
    /*  Increase the number of bytes by two or three integers, one for facet */
    /*    marker, one for shellface type, and optionally one for pbc group. */
    shsize = (m->shmarkindex + 2 + m->checkpbcs) * sizeof(int);
    /*  Initialize the pool of subfaces. Each subface record is eight-byte */
    /*    aligned so it has room to store an edge version (from 0 to 5) in */
    /*    the least three bits. */
    ierr = MemoryPoolCreate(shsize, SUBPERBLOCK, POINTER, 8, &m->subfaces);CHKERRQ(ierr);
    /*  Initialize the pool of subsegments. The subsegment's record is same */
    /*    with subface. */
    ierr = MemoryPoolCreate(shsize, SUBPERBLOCK, POINTER, 8, &m->subsegs);CHKERRQ(ierr);
    /*  Initialize the pool for tet-subseg connections. */
    ierr = MemoryPoolCreate(6*sizeof(shellface), SUBPERBLOCK, POINTER, 0, &m->tet2segpool);CHKERRQ(ierr);
    /*  Initialize the pool for tet-subface connections. */
    ierr = MemoryPoolCreate(4*sizeof(shellface), SUBPERBLOCK, POINTER, 0, &m->tet2subpool);CHKERRQ(ierr);
    /*  Initialize arraypools for segment & facet recovery. */
    ierr = ArrayPoolCreate(sizeof(face), 10, &m->subsegstack);CHKERRQ(ierr);
    ierr = ArrayPoolCreate(sizeof(face), 10, &m->subfacstack);CHKERRQ(ierr);
    /*  Initialize the "outer space" tetrahedron and omnipresent subface. */
    ierr = TetGenMeshDummyInit(m, m->tetrahedrons->itemwords, m->subfaces->itemwords);CHKERRQ(ierr);
  } else {
    /*  Initialize the "outer space" tetrahedron. */
    ierr = TetGenMeshDummyInit(m, m->tetrahedrons->itemwords, 0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakePoint2TetMap"
/*  makepoint2tetmap()    Construct a mapping from points to tetrahedra.       */
/*                                                                             */
/*  Traverses all the tetrahedra,  provides each corner of each tetrahedron    */
/*  with a pointer to that tetrahedera.  Some pointers will be overwritten by  */
/*  other pointers because each point may be a corner of several tetrahedra,   */
/*  but in the end every point will point to a tetrahedron that contains it.   */
/* tetgenmesh::makepoint2tetmap() */
PetscErrorCode TetGenMeshMakePoint2TetMap(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  triface tetloop = {PETSC_NULL, 0, 0};
  point pointptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constructing mapping from points to tetrahedra.\n");

  /*  Initialize the point2tet field of each point. */
  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  ierr = TetGenMeshPointTraverse(m, &pointptr);CHKERRQ(ierr);
  while(pointptr) {
    setpoint2tet(m, pointptr, PETSC_NULL);
    ierr = TetGenMeshPointTraverse(m, &pointptr);CHKERRQ(ierr);
  }

  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    /*  Check all four points of the tetrahedron. */
    tetloop.loc = 0;
    pointptr = org(&tetloop);
    setpoint2tet(m, pointptr, encode(&tetloop));
    pointptr = dest(&tetloop);
    setpoint2tet(m, pointptr, encode(&tetloop));
    pointptr = apex(&tetloop);
    setpoint2tet(m, pointptr, encode(&tetloop));
    pointptr = oppo(&tetloop);
    setpoint2tet(m, pointptr, encode(&tetloop));
    /*  Get the next tetrahedron in the list. */
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakePoint2SegMap"
/* tetgenmesh::makepoint2segmap() */
PetscErrorCode TetGenMeshMakePoint2SegMap(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  face segloop = {PETSC_NULL, 0};
  point *ppt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constructing mapping from points to segments.\n");

  segloop.shver = 0;
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  while(segloop.sh) {
    ppt = (point *) &(segloop.sh[3]);
    setpoint2seg(m, ppt[0], sencode(&segloop));
    setpoint2seg(m, ppt[1], sencode(&segloop));
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakeIndex2PointMap"
/*  makeindex2pointmap()    Create a map from index to vertices.               */
/*                                                                             */
/*  'idx2verlist' returns the created map.  Traverse all vertices, a pointer   */
/*  to each vertex is set into the array.  The pointer to the first vertex is  */
/*  saved in 'idx2verlist[0]'.  Don't forget to minus 'in->firstnumber' when   */
/*  to get the vertex form its index.                                          */
/* tetgenmesh::makeindex2pointmap() */
PetscErrorCode TetGenMeshMakeIndex2PointMap(TetGenMesh *m, point **idx2verlist)
{
  TetGenOpts    *b  = m->b;
  point pointloop;
  int idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constructing mapping from indices to points.\n");

  ierr = PetscMalloc(m->points->items * sizeof(point), idx2verlist);CHKERRQ(ierr);
  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  idx  = 0;
  while(pointloop) {
    (*idx2verlist)[idx] = pointloop;
    idx++;
    ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakeSubfaceMap"
/*  makesegmentmap(), makesubfacemap(), maketetrahedronmap()                   */
/*                                                                             */
/*  Create a map from vertex indices to segments, subfaces, and tetrahedra     */
/*  sharing at the same vertices.                                              */
/*                                                                             */
/*  The map is stored in two arrays: 'idx2___list' and '___sperverlist', they  */
/*  form a sparse matrix whose size is (n+1)x(n+1), where n is the number of   */
/*  segments, subfaces, or tetrahedra. 'idx2___list' contains row information  */
/*  and '___sperverlist' contains all non-zero elements.  The i-th entry of    */
/*  'idx2___list' is the starting position of i-th row's non-zero elements in  */
/*  '___sperverlist'.  The number of elements of i-th row is (i+1)-th entry    */
/*  minus i-th entry of 'idx2___list'.                                         */
/*                                                                             */
/*  NOTE: These two arrays will be created inside this routine, don't forget   */
/*  to free them after using.                                                  */
/* tetgenmesh::makesubfacemap() */
PetscErrorCode TetGenMeshMakeSubfaceMap(TetGenMesh *m, int **index2facelist, shellface ***facespervertexlist)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  shellface *shloop;
  shellface **facesperverlist;
  int i, j, k;
  int *idx2facelist;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constructing mapping from points to subfaces.\n");

  /*  Create and initialize 'idx2facelist'. */
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &idx2facelist);CHKERRQ(ierr);
  for (i = 0; i < m->points->items + 1; i++) idx2facelist[i] = 0;

  /*  Loop the set of subfaces once, counter the number of subfaces sharing each vertex. */
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &shloop);CHKERRQ(ierr);
  while(shloop) {
    /*  Increment the number of sharing segments for each endpoint. */
    for(i = 0; i < 3; i++) {
      j = pointmark(m, (point) shloop[3 + i]) - in->firstnumber;
      idx2facelist[j]++;
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &shloop);CHKERRQ(ierr);
  }

  /*  Calculate the total length of array 'facesperverlist'. */
  j = idx2facelist[0];
  idx2facelist[0] = 0;  /*  Array starts from 0 element. */
  for(i = 0; i < m->points->items; i++) {
    k = idx2facelist[i + 1];
    idx2facelist[i + 1] = idx2facelist[i] + j;
    j = k;
  }
  /*  The total length is in the last unit of idx2facelist. */
  ierr = PetscMalloc(idx2facelist[i] * sizeof(shellface *), &facesperverlist);CHKERRQ(ierr);
  /*  Loop the set of segments again, set the info. of segments per vertex. */
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &shloop);CHKERRQ(ierr);
  while(shloop) {
    for(i = 0; i < 3; i++) {
      j = pointmark(m, (point) shloop[3 + i]) - in->firstnumber;
      facesperverlist[idx2facelist[j]] = shloop;
      idx2facelist[j]++;
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &shloop);CHKERRQ(ierr);
  }
  /*  Contents in 'idx2facelist' are shifted, now shift them back. */
  for(i = m->points->items - 1; i >= 0; i--) {
    idx2facelist[i + 1] = idx2facelist[i];
  }
  idx2facelist[0] = 0;
  *index2facelist     = idx2facelist;
  *facespervertexlist = facesperverlist;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakePoint"
/* tetgenmesh::makepoint() */
PetscErrorCode TetGenMeshMakePoint(TetGenMesh *m, point *pnewpoint)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  int            ptmark, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MemoryPoolAlloc(m->points, (void **) pnewpoint);CHKERRQ(ierr);
  /*  Initialize three coordinates. */
  (*pnewpoint)[0] = 0.0;
  (*pnewpoint)[1] = 0.0;
  (*pnewpoint)[2] = 0.0;
  /*  Initialize the list of user-defined attributes. */
  for(i = 0; i < in->numberofpointattributes; i++) {
    (*pnewpoint)[3 + i] = 0.0;
  }
  /*  Initialize the metric tensor. */
  for(i = 0; i < m->sizeoftensor; i++) {
    (*pnewpoint)[m->pointmtrindex + i] = 0.0;
  }
  if (b->plc || b->refine) {
    /*  Initialize the point-to-simplex filed. */
    setpoint2tet(m, *pnewpoint, PETSC_NULL);
    setpoint2sh(m, *pnewpoint, PETSC_NULL);
    setpoint2seg(m, *pnewpoint, PETSC_NULL);
    setpoint2ppt(m, *pnewpoint, PETSC_NULL);
    if (b->metric) {
      setpoint2bgmtet(m, *pnewpoint, PETSC_NULL);
    }
    if (m->checkpbcs) {
      /*  Initialize the other pointer to its pbc point. */
      setpoint2pbcpt(m, *pnewpoint, PETSC_NULL);
    }
  }
  /*  Initialize the point marker (starting from in->firstnumber). */
  ptmark = (int) m->points->items - (in->firstnumber == 1 ? 0 : 1);
  setpointmark(m, *pnewpoint, ptmark);
  /*  Initialize the point type. */
  setpointtype(m, *pnewpoint, UNUSEDVERTEX);
  /*  Clear the point flags. */
  puninfect(m, *pnewpoint);
  /* punmarktest(*pnewpoint); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakeShellFace"
/*  makeshellface()    Create a new shellface with version zero. Used for both subfaces and seusegments.  */
/* tetgenmesh::makeshellface() */
PetscErrorCode TetGenMeshMakeShellFace(TetGenMesh *m, MemoryPool *pool, face *newface)
{
  TetGenOpts    *b  = m->b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MemoryPoolAlloc(pool, (void **) &newface->sh);CHKERRQ(ierr);
  /* Initialize the three adjoining subfaces to be the omnipresent subface. */
  newface->sh[0] = (shellface) m->dummysh;
  newface->sh[1] = (shellface) m->dummysh;
  newface->sh[2] = (shellface) m->dummysh;
  /*  Three NULL vertices. */
  newface->sh[3] = PETSC_NULL;
  newface->sh[4] = PETSC_NULL;
  newface->sh[5] = PETSC_NULL;
  /*  Initialize the two adjoining tetrahedra to be "outer space". */
  newface->sh[6] = (shellface) m->dummytet;
  newface->sh[7] = (shellface) m->dummytet;
  /*  Initialize the three adjoining subsegments to be the omnipresent */
  /*    subsegments. */
  newface->sh [8] = (shellface) m->dummysh;
  newface->sh [9] = (shellface) m->dummysh;
  newface->sh[10] = (shellface) m->dummysh;
  /*  Initialize the pointer to badface structure. */
  newface->sh[11] = PETSC_NULL;
  if (b->quality && m->varconstraint) {
    /*  Initialize the maximum area bound. */
    setareabound(m, newface, 0.0);
  }
  /*  Clear the infection and marktest bits. */
  suninfect(m, newface);
  sunmarktest(newface);
  /*  Set the boundary marker to zero. */
  setshellmark(m, newface, 0);
  /*  Set the type. */
  setshelltype(m, newface, NSHARP);
  if (m->checkpbcs) {
    /*  Set the pbcgroup be ivalid. */
    setshellpbcgroup(m, newface, -1);
  }
  /*  Initialize the version to be Zero. */
  newface->shver = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakeTetrahedron"
/* tetgenmesh::maketetrahedron() */
PetscErrorCode TetGenMeshMakeTetrahedron(TetGenMesh *m, triface *newtet)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MemoryPoolAlloc(m->tetrahedrons, (void **) &newtet->tet);CHKERRQ(ierr);
  /*  Initialize the four adjoining tetrahedra to be "outer space". */
  newtet->tet[0] = (tetrahedron) m->dummytet;
  newtet->tet[1] = (tetrahedron) m->dummytet;
  newtet->tet[2] = (tetrahedron) m->dummytet;
  newtet->tet[3] = (tetrahedron) m->dummytet;
  /*  Four NULL vertices. */
  newtet->tet[4] = PETSC_NULL;
  newtet->tet[5] = PETSC_NULL;
  newtet->tet[6] = PETSC_NULL;
  newtet->tet[7] = PETSC_NULL;
  /*  Initialize the four adjoining subfaces to be the omnipresent subface. */
  if (b->useshelles) {
    newtet->tet[8 ] = PETSC_NULL;
    newtet->tet[9 ] = PETSC_NULL;
  }
  for(i = 0; i < in->numberoftetrahedronattributes; i++) {
    setelemattribute(m, newtet->tet, i, 0.0);
  }
  if (b->varvolume) {
    setvolumebound(m, newtet->tet, -1.0);
  }
  /*  Initialize the marker (for flags). */
  setelemmarker(m, newtet->tet, 0);
  /*  Initialize the location and version to be Zero. */
  newtet->loc = 0;
  newtet->ver = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPointTraverse"
/* tetgenmesh::pointtraverse() */
PetscErrorCode TetGenMeshPointTraverse(TetGenMesh *m, point *next)
{
  point          newpoint;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  do {
    ierr = MemoryPoolTraverse(m->points, (void **) &newpoint);CHKERRQ(ierr);
    if (!newpoint) {
      *next = PETSC_NULL;
      PetscFunctionReturn(0);
    }
  } while (pointtype(m, newpoint) == DEADVERTEX);            /*  Skip dead ones. */
  *next = newpoint;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshShellFaceTraverse"
/* tetgenmesh::shellfacetraverse() */
PetscErrorCode TetGenMeshShellFaceTraverse(TetGenMesh *m, MemoryPool *pool, shellface **next)
{
  shellface     *newshellface;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  do {
    ierr = MemoryPoolTraverse(pool, (void **) &newshellface);CHKERRQ(ierr);
    if (!newshellface) {
      *next = PETSC_NULL;
      PetscFunctionReturn(0);
    }
  } while (!newshellface[3]);            /*  Skip dead ones. */
  *next = newshellface;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBadFaceTraverse"
/* tetgenmesh::badfacetraverse() */
PetscErrorCode TetGenMeshBadFaceTraverse(TetGenMesh *m, MemoryPool *pool, badface **next)
{
  badface       *newsh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  do {
    ierr = MemoryPoolTraverse(pool, (void **) &newsh);CHKERRQ(ierr);
    if (!newsh) {
      *next = PETSC_NULL;
      PetscFunctionReturn(0);
    }
  } while (!newsh->forg);            /*  Skip dead ones. */
  *next = newsh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTetrahedronTraverse"
/* tetgenmesh::tetrahedrontraverse() */
PetscErrorCode TetGenMeshTetrahedronTraverse(TetGenMesh *m, tetrahedron **next)
{
  tetrahedron   *newtetrahedron;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  do {
    ierr = MemoryPoolTraverse(m->tetrahedrons, (void **) &newtetrahedron);CHKERRQ(ierr);
    if (!newtetrahedron) {
      *next = PETSC_NULL;
      PetscFunctionReturn(0);
    }
  } while (!newtetrahedron[7]);            /*  Skip dead ones. */
  *next = newtetrahedron;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPointDealloc"
/* tetgenmesh::pointdealloc() */
PetscErrorCode TetGenMeshPointDealloc(TetGenMesh *m, point dyingpoint)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Mark the point as dead. This  makes it possible to detect dead points */
  /*    when traversing the list of all points. */
  setpointtype(m, dyingpoint, DEADVERTEX);
  ierr = MemoryPoolDealloc(m->points, dyingpoint);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshShellFaceDealloc"
/* tetgenmesh::shellfacedealloc() */
PetscErrorCode TetGenMeshShellFaceDealloc(TetGenMesh *m, MemoryPool *pool, shellface *dyingsh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Set shellface's vertices to NULL. This makes it possible to detect dead */
  /*    shellfaces when traversing the list of all shellfaces. */
  dyingsh[3] = PETSC_NULL;
  dyingsh[4] = PETSC_NULL;
  dyingsh[5] = PETSC_NULL;
  ierr = MemoryPoolDealloc(pool, dyingsh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBadFaceDealloc"
/* tetgenmesh::badfacedealloc() */
PetscErrorCode TetGenMeshBadFaceDealloc(TetGenMesh *m, MemoryPool *pool, badface *dying)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Set badface's forg to NULL. This makes it possible to detect dead */
  /*    ones when traversing the list of all items. */
  dying->forg = PETSC_NULL;
  ierr = MemoryPoolDealloc(pool, dying);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTetrahedronDealloc"
/* tetgenmesh::tetrahedrondealloc() */
PetscErrorCode TetGenMeshTetrahedronDealloc(TetGenMesh *m, tetrahedron *dyingtetrahedron)
{
  TetGenOpts    *b = m->b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Set tetrahedron's vertices to NULL. This makes it possible to detect */
  /*    dead tetrahedra when traversing the list of all tetrahedra. */
  dyingtetrahedron[4] = PETSC_NULL;
  /*  dyingtetrahedron[5] = (tetrahedron) NULL; */
  /*  dyingtetrahedron[6] = (tetrahedron) NULL; */
  dyingtetrahedron[7] = PETSC_NULL;

  if (b->useshelles) {
    /*  Dealloc the space to subfaces/subsegments. */
    if (dyingtetrahedron[8]) {
      ierr = MemoryPoolDealloc(m->tet2segpool, dyingtetrahedron[8]);CHKERRQ(ierr);
    }
    if (dyingtetrahedron[9]) {
      ierr = MemoryPoolDealloc(m->tet2subpool, dyingtetrahedron[9]);CHKERRQ(ierr);
    }
  }
  ierr = MemoryPoolDealloc(m->tetrahedrons, dyingtetrahedron);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  mempool_cxx ////////////////////////////////////////////////////////////// */

/*  geom_cxx ///////////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCircumsphere"
/*  circumsphere()    Calculate the smallest circumsphere (center and radius)  */
/*                    of the given three or four points.                       */
/*                                                                             */
/*  The circumsphere of four points (a tetrahedron) is unique if they are not  */
/*  degenerate. If 'pd = NULL', the smallest circumsphere of three points is   */
/*  the diametral sphere of the triangle if they are not degenerate.           */
/*                                                                             */
/*  Return TRUE if the input points are not degenerate and the circumcenter    */
/*  and circumradius are returned in 'cent' and 'radius' respectively if they  */
/*  are not NULLs. Otherwise, return FALSE indicated the points are degenrate. */
/* tetgenmesh::circumsphere() */
PetscErrorCode TetGenMeshCircumsphere(TetGenMesh *m, PetscReal* pa, PetscReal* pb, PetscReal* pc, PetscReal* pd, PetscReal* cent, PetscReal* radius, PetscBool *notDegenerate)
{
  PetscReal A[4][4], rhs[4], D;
  int indx[4];

  PetscFunctionBegin;
  /*  Compute the coefficient matrix A (3x3). */
  A[0][0] = pb[0] - pa[0];
  A[0][1] = pb[1] - pa[1];
  A[0][2] = pb[2] - pa[2];
  A[1][0] = pc[0] - pa[0];
  A[1][1] = pc[1] - pa[1];
  A[1][2] = pc[2] - pa[2];
  if (pd) {
    A[2][0] = pd[0] - pa[0];
    A[2][1] = pd[1] - pa[1];
    A[2][2] = pd[2] - pa[2];
  } else {
    cross(A[0], A[1], A[2]);
  }

  /*  Compute the right hand side vector b (3x1). */
  rhs[0] = 0.5 * dot(A[0], A[0]);
  rhs[1] = 0.5 * dot(A[1], A[1]);
  if (pd) {
    rhs[2] = 0.5 * dot(A[2], A[2]);
  } else {
    rhs[2] = 0.0;
  }

  /*  Solve the 3 by 3 equations use LU decomposition with partial pivoting */
  /*    and backward and forward substitute.. */
  if (!TetGenLUDecomp(A, 3, indx, &D, 0)) {
    if (radius) {*radius = 0.0;}
    if (notDegenerate) {*notDegenerate = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }
  TetGenLUSolve(A, 3, indx, rhs, 0);
  if (cent) {
    cent[0] = pa[0] + rhs[0];
    cent[1] = pa[1] + rhs[1];
    cent[2] = pa[2] + rhs[2];
  }
  if (radius) {
    *radius = sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);
  }
  if (notDegenerate) {*notDegenerate = PETSC_TRUE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFaceNormal"
/*  facenormal()    Calculate the normal of a face given by three points.      */
/*                                                                             */
/*  In general, the face normal can be calculate by the cross product of any   */
/*  pair of the three edge vectors.  However, if the three points are nearly   */
/*  collinear, the rounding error may harm the result. To choose a good pair   */
/*  of vectors is helpful to reduce the error.                                 */
/* tetgenmesh::facenormal() */
PetscErrorCode TetGenMeshFaceNormal(TetGenMesh *m, PetscReal* pa, PetscReal* pb, PetscReal* pc, PetscReal* n, PetscReal* nlen)
{
  PetscReal v1[3], v2[3];

  PetscFunctionBegin;
  v1[0] = pb[0] - pa[0];
  v1[1] = pb[1] - pa[1];
  v1[2] = pb[2] - pa[2];
  v2[0] = pc[0] - pa[0];
  v2[1] = pc[1] - pa[1];
  v2[2] = pc[2] - pa[2];

  cross(v1, v2, n);
  if (nlen) {
    *nlen = sqrt(dot(n, n));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFaceNormal2"
/*  facenormal2()    Calculate the normal of the face.                          */
/*                                                                             */
/*  The normal of the face abc can be calculated by the cross product of 2 of  */
/*  its 3 edge vectors.  A better choice of two edge vectors will reduce the   */
/*  numerical error during the calculation.  Burdakov proved that the optimal  */
/*  basis problem is equivalent to the minimum spanning tree problem with the  */
/*  edge length be the functional, see Burdakov, "A greedy algorithm for the   */
/*  optimal basis problem", BIT 37:3 (1997), 591-599. If 'pivot' > 0, the two  */
/*  short edges in abc are chosen for the calculation.                         */
/* tetgenmesh::facenormal2() */
PetscErrorCode TetGenMeshFaceNormal2(TetGenMesh *m, point pa, point pb, point pc, PetscReal *n, int pivot)
{
  PetscReal v1[3], v2[3], v3[3], *pv1, *pv2;
  PetscReal L1, L2, L3;

  PetscFunctionBegin;
  v1[0] = pb[0] - pa[0];  /*  edge vector v1: a->b */
  v1[1] = pb[1] - pa[1];
  v1[2] = pb[2] - pa[2];
  v2[0] = pa[0] - pc[0];  /*  edge vector v2: c->a */
  v2[1] = pa[1] - pc[1];
  v2[2] = pa[2] - pc[2];

  /*  Default, normal is calculated by: v1 x (-v2) (see Fig. fnormal). */
  if (pivot > 0) {
    /*  Choose edge vectors by Burdakov's algorithm. */
    v3[0] = pc[0] - pb[0];  /*  edge vector v3: b->c */
    v3[1] = pc[1] - pb[1];
    v3[2] = pc[2] - pb[2];
    L1 = DOT(v1, v1);
    L2 = DOT(v2, v2);
    L3 = DOT(v3, v3);
    /*  Sort the three edge lengths. */
    if (L1 < L2) {
      if (L2 < L3) {
        pv1 = v1; pv2 = v2; /*  n = v1 x (-v2). */
      } else {
        pv1 = v3; pv2 = v1; /*  n = v3 x (-v1). */
      }
    } else {
      if (L1 < L3) {
        pv1 = v1; pv2 = v2; /*  n = v1 x (-v2). */
      } else {
        pv1 = v2; pv2 = v3; /*  n = v2 x (-v3). */
      }
    }
  } else {
    pv1 = v1; pv2 = v2; /*  n = v1 x (-v2). */
  }

  /*  Calculate the face normal. */
  CROSS(pv1, pv2, n);
  /*  Inverse the direction; */
  n[0] = -n[0];
  n[1] = -n[1];
  n[2] = -n[2];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshProjPt2Face"
/*  projpt2face()    Return the projection point from a point to a face.       */
/* tetgenmesh::projpt2face() */
PetscErrorCode TetGenMeshProjPt2Face(TetGenMesh *m, PetscReal* p, PetscReal* f1, PetscReal* f2, PetscReal* f3, PetscReal* prj)
{
  PetscReal fnormal[3], v1[3];
  PetscReal len, dist;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Get the unit face normal. */
  /*  facenormal(f1, f2, f3, fnormal, &len); */
  ierr = TetGenMeshFaceNormal2(m, f1, f2, f3, fnormal, 1);CHKERRQ(ierr);
  len = sqrt(fnormal[0]*fnormal[0] + fnormal[1]*fnormal[1] + fnormal[2]*fnormal[2]);
  fnormal[0] /= len;
  fnormal[1] /= len;
  fnormal[2] /= len;
  /*  Get the vector v1 = |p - f1|. */
  v1[0] = p[0] - f1[0];
  v1[1] = p[1] - f1[1];
  v1[2] = p[2] - f1[2];
  /*  Get the project distance. */
  dist = dot(fnormal, v1);

  /*  Get the project point. */
  prj[0] = p[0] - dist * fnormal[0];
  prj[1] = p[1] - dist * fnormal[1];
  prj[2] = p[2] - dist * fnormal[2];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTriEdge2D"
/* tetgenmesh::tri_edge_2d() */
PetscErrorCode TetGenMeshTriEdge2D(TetGenMesh *m, point A, point B, point C, point P, point Q, point R, int level, int *types, int *pos, int *isIntersect)
{
  TetGenOpts    *b = m->b;
  point U[3] = {PETSC_NULL, PETSC_NULL, PETSC_NULL}, V[3] = {PETSC_NULL, PETSC_NULL, PETSC_NULL};  /*  The permuted vectors of points. */
  int pu[3] = {0, 0, 0}, pv[3] = {0, 0, 0};  /*  The original positions of points. */
  PetscReal sA, sB, sC;
  PetscReal s1, s2, s3, s4;
  int z1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!R) {
    PetscReal n[3], len;
    /*  Calculate a lift point, saved in dummypoint. */
    ierr = TetGenMeshFaceNormal2(m, A, B, C, n, 1);CHKERRQ(ierr);
    len = sqrt(DOT(n, n));
    n[0] /= len;
    n[1] /= len;
    n[2] /= len;
    len = DIST(A, B);
    len += DIST(B, C);
    len += DIST(C, A);
    len /= 3.0;
    R = m->dummypoint;
    R[0] = A[0] + len * n[0];
    R[1] = A[1] + len * n[1];
    R[2] = A[2] + len * n[2];
  }

  /*  Test A's, B's, and C's orientations wrt plane PQR. */
  sA = TetGenOrient3D(P, Q, R, A);
  sB = TetGenOrient3D(P, Q, R, B);
  sC = TetGenOrient3D(P, Q, R, C);
  m->orient3dcount+=3;

  PetscInfo6(b->in, "      Tri-edge-2d (%d %d %d)-(%d %d)-(%d) (%c%c%c)", pointmark(m, A), pointmark(m, B), pointmark(m, C), pointmark(m, P), pointmark(m, Q), pointmark(m, R));
  PetscInfo3(b->in, "        (%c%c%c)", sA > 0 ? '+' : (sA < 0 ? '-' : '0'), sB>0 ? '+' : (sB<0 ? '-' : '0'), sC>0 ? '+' : (sC<0 ? '-' : '0'));
  /*  triedgcopcount++; */

  if (sA < 0) {
    if (sB < 0) {
      if (sC < 0) { /*  (---). */
        if (isIntersect) {*isIntersect = 0;}
        PetscFunctionReturn(0);
      } else {
        if (sC > 0) { /*  (--+). */
          /*  All points are in the right positions. */
          SETVECTOR3(U, A, B, C);  /*  I3 */
          SETVECTOR3(V, P, Q, R);  /*  I2 */
          SETVECTOR3(pu, 0, 1, 2);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 0;
        } else { /*  (--0). */
          SETVECTOR3(U, A, B, C);  /*  I3 */
          SETVECTOR3(V, P, Q, R);  /*  I2 */
          SETVECTOR3(pu, 0, 1, 2);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 1;
        }
      }
    } else {
      if (sB > 0) {
        if (sC < 0) { /*  (-+-). */
          SETVECTOR3(U, C, A, B);  /*  PT = ST */
          SETVECTOR3(V, P, Q, R);  /*  I2 */
          SETVECTOR3(pu, 2, 0, 1);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 0;
        } else {
          if (sC > 0) { /*  (-++). */
            SETVECTOR3(U, B, C, A);  /*  PT = ST x ST */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 1, 2, 0);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 0;
          } else { /*  (-+0). */
            SETVECTOR3(U, C, A, B);  /*  PT = ST */
            SETVECTOR3(V, P, Q, R);  /*  I2 */
            SETVECTOR3(pu, 2, 0, 1);
            SETVECTOR3(pv, 0, 1, 2);
            z1 = 2;
          }
        }
      } else {
        if (sC < 0) { /*  (-0-). */
          SETVECTOR3(U, C, A, B);  /*  PT = ST */
          SETVECTOR3(V, P, Q, R);  /*  I2 */
          SETVECTOR3(pu, 2, 0, 1);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 1;
        } else {
          if (sC > 0) { /*  (-0+). */
            SETVECTOR3(U, B, C, A);  /*  PT = ST x ST */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 1, 2, 0);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 2;
          } else { /*  (-00). */
            SETVECTOR3(U, B, C, A);  /*  PT = ST x ST */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 1, 2, 0);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 3; 
          }
        }
      }
    }
  } else {
    if (sA > 0) {
      if (sB < 0) {
        if (sC < 0) { /*  (+--). */
          SETVECTOR3(U, B, C, A);  /*  PT = ST x ST */
          SETVECTOR3(V, P, Q, R);  /*  I2 */
          SETVECTOR3(pu, 1, 2, 0);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 0;
        } else {
          if (sC > 0) { /*  (+-+). */
            SETVECTOR3(U, C, A, B);  /*  PT = ST */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 2, 0, 1);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 0;
          } else { /*  (+-0). */
            SETVECTOR3(U, C, A, B);  /*  PT = ST */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 2, 0, 1);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 2;
          }
        }
      } else {
        if (sB > 0) {
          if (sC < 0) { /*  (++-). */
            SETVECTOR3(U, A, B, C);  /*  I3 */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 0, 1, 2);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 0;
          } else {
            if (sC > 0) { /*  (+++). */
              if (isIntersect) {*isIntersect = 0;}
              PetscFunctionReturn(0);
            } else { /*  (++0). */
              SETVECTOR3(U, A, B, C);  /*  I3 */
              SETVECTOR3(V, Q, P, R);  /*  PL = SL */
              SETVECTOR3(pu, 0, 1, 2);
              SETVECTOR3(pv, 1, 0, 2);
              z1 = 1;
            }
          }
        } else { /*  (+0#) */
          if (sC < 0) { /*  (+0-). */
            SETVECTOR3(U, B, C, A);  /*  PT = ST x ST */
            SETVECTOR3(V, P, Q, R);  /*  I2 */
            SETVECTOR3(pu, 1, 2, 0);
            SETVECTOR3(pv, 0, 1, 2);
            z1 = 2;
          } else {
            if (sC > 0) { /*  (+0+). */
              SETVECTOR3(U, C, A, B);  /*  PT = ST */
              SETVECTOR3(V, Q, P, R);  /*  PL = SL */
              SETVECTOR3(pu, 2, 0, 1);
              SETVECTOR3(pv, 1, 0, 2);
              z1 = 1;
            } else { /*  (+00). */
              SETVECTOR3(U, B, C, A);  /*  PT = ST x ST */
              SETVECTOR3(V, P, Q, R);  /*  I2 */
              SETVECTOR3(pu, 1, 2, 0);
              SETVECTOR3(pv, 0, 1, 2);
              z1 = 3;
            }
          }
        }
      }
    } else {
      if (sB < 0) {
        if (sC < 0) { /*  (0--). */
          SETVECTOR3(U, B, C, A);  /*  PT = ST x ST */
          SETVECTOR3(V, P, Q, R);  /*  I2 */
          SETVECTOR3(pu, 1, 2, 0);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 1;
        } else {
          if (sC > 0) { /*  (0-+). */
            SETVECTOR3(U, A, B, C);  /*  I3 */
            SETVECTOR3(V, P, Q, R);  /*  I2 */
            SETVECTOR3(pu, 0, 1, 2);
            SETVECTOR3(pv, 0, 1, 2);
            z1 = 2;
          } else { /*  (0-0). */
            SETVECTOR3(U, C, A, B);  /*  PT = ST */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 2, 0, 1);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 3;
          }
        }
      } else {
        if (sB > 0) {
          if (sC < 0) { /*  (0+-). */
            SETVECTOR3(U, A, B, C);  /*  I3 */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 0, 1, 2);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 2;
          } else {
            if (sC > 0) { /*  (0++). */
              SETVECTOR3(U, B, C, A);  /*  PT = ST x ST */
              SETVECTOR3(V, Q, P, R);  /*  PL = SL */
              SETVECTOR3(pu, 1, 2, 0);
              SETVECTOR3(pv, 1, 0, 2);
              z1 = 1;
            } else { /*  (0+0). */
              SETVECTOR3(U, C, A, B);  /*  PT = ST */
              SETVECTOR3(V, P, Q, R);  /*  I2 */
              SETVECTOR3(pu, 2, 0, 1);
              SETVECTOR3(pv, 0, 1, 2);
              z1 = 3;
            }
          }
        } else { /*  (00#) */
          if (sC < 0) { /*  (00-). */
            SETVECTOR3(U, A, B, C);  /*  I3 */
            SETVECTOR3(V, Q, P, R);  /*  PL = SL */
            SETVECTOR3(pu, 0, 1, 2);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 3;
          } else {
            if (sC > 0) { /*  (00+). */
              SETVECTOR3(U, A, B, C);  /*  I3 */
              SETVECTOR3(V, P, Q, R);  /*  I2 */
              SETVECTOR3(pu, 0, 1, 2);
              SETVECTOR3(pv, 0, 1, 2);
              z1 = 3;
            } else { /*  (000) */
              /*  Not possible unless ABC is degenerate. */
              z1 = 4;
            }
          }
        }
      }
    }
  }

  s1 = TetGenOrient3D(U[0], U[2], R, V[1]);  /*  A, C, R, Q */
  s2 = TetGenOrient3D(U[1], U[2], R, V[0]);  /*  B, C, R, P */
  m->orient3dcount+=2;

  PetscInfo7(b->in, "      Tri-edge-2d (%d %d %d)-(%d %d %d) (%d) (%c%c)\n", pointmark(m, U[0]), pointmark(m, U[1]), pointmark(m, U[2]), pointmark(m, V[0]),
             pointmark(m, V[1]), pointmark(m, V[2]), z1);
  PetscInfo2(b->in, "        (%c%c)\n", s1>0 ? '+' : (s1<0 ? '-' : '0'), s2>0 ? '+' : (s2<0 ? '-' : '0'));
  if (z1 == 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

  if (s1 > 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }
  if (s2 < 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }

  if (level == 0) {
    if (isIntersect) {*isIntersect = 1;} /*  They are intersected. */
    PetscFunctionReturn(0);
  }

  if (z1 == 1) {
    if (s1 == 0) {  /*  (0###) */
      /*  C = Q. */
      types[0] = (int) SHAREVERTEX;
      pos[0] = pu[2]; /*  C */
      pos[1] = pv[1]; /*  Q */
      types[1] = (int) DISJOINT;
    } else {
      if (s2 == 0) { /*  (#0##) */
        /*  C = P. */
        types[0] = (int) SHAREVERTEX;
        pos[0] = pu[2]; /*  C */
        pos[1] = pv[0]; /*  P */
        types[1] = (int) DISJOINT;
      } else { /*  (-+##) */
        /*  C in [P, Q]. */
        types[0] = (int) INTERVERT;
        pos[0] = pu[2]; /*  C */
        pos[1] = pv[0]; /*  [P, Q] */
        types[1] = (int) DISJOINT;
      }
    }
    if (isIntersect) {*isIntersect = 1;}
    PetscFunctionReturn(0);
  }

  s3 = TetGenOrient3D(U[0], U[2], R, V[0]);  /*  A, C, R, P */
  s4 = TetGenOrient3D(U[1], U[2], R, V[1]);  /*  B, C, R, Q */
  m->orient3dcount+=2;

  if (z1 == 0) {  /*  (tritri-03) */
    if (s1 < 0) {
      if (s3 > 0) {
        if (s2 <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        if (s4 > 0) {
          /*  [P, Q] overlaps [k, l] (-+++). */
          types[0] = (int) INTEREDGE;
          pos[0] = pu[2]; /*  [C, A] */
          pos[1] = pv[0]; /*  [P, Q] */
          types[1] = (int) TOUCHFACE;
          pos[2] = 3;     /*  [A, B, C] */
          pos[3] = pv[1]; /*  Q */
        } else {
          if (s4 == 0) {
            /*  Q = l, [P, Q] contains [k, l] (-++0). */
            types[0] = (int) INTEREDGE;
            pos[0] = pu[2]; /*  [C, A] */
            pos[1] = pv[0]; /*  [P, Q] */
            types[1] = (int) TOUCHEDGE;
            pos[2] = pu[1]; /*  [B, C] */
            pos[3] = pv[1]; /*  Q */
          } else { /*  s4 < 0 */
            /*  [P, Q] contains [k, l] (-++-). */
            types[0] = (int) INTEREDGE;
            pos[0] = pu[2]; /*  [C, A] */
            pos[1] = pv[0]; /*  [P, Q] */
            types[1] = (int) INTEREDGE;
            pos[2] = pu[1]; /*  [B, C] */
            pos[3] = pv[0]; /*  [P, Q] */
          }
        }
      } else {
        if (s3 == 0) {
          if (s2 <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          if (s4 > 0) {
            /*  P = k, [P, Q] in [k, l] (-+0+). */
            types[0] = (int) TOUCHEDGE;
            pos[0] = pu[2]; /*  [C, A] */
            pos[1] = pv[0]; /*  P */
            types[1] = (int) TOUCHFACE;
            pos[2] = 3;     /*  [A, B, C] */
            pos[3] = pv[1]; /*  Q */
          } else {
            if (s4 == 0) {
              /*  [P, Q] = [k, l] (-+00). */
              types[0] = (int) TOUCHEDGE;
              pos[0] = pu[2]; /*  [C, A] */
              pos[1] = pv[0]; /*  P */
              types[1] = (int) TOUCHEDGE;
              pos[2] = pu[1]; /*  [B, C] */
              pos[3] = pv[1]; /*  Q */
            } else {
              /*  P = k, [P, Q] contains [k, l] (-+0-). */
              types[0] = (int) TOUCHEDGE;
              pos[0] = pu[2]; /*  [C, A] */
              pos[1] = pv[0]; /*  P */
              types[1] = (int) INTEREDGE;
              pos[2] = pu[1]; /*  [B, C] */
              pos[3] = pv[0]; /*  [P, Q] */
            }
          }
        } else { /*  s3 < 0 */
          if (s2 > 0) {
            if (s4 > 0) {
              /*  [P, Q] in [k, l] (-+-+). */
              types[0] = (int) TOUCHFACE;
              pos[0] = 3;     /*  [A, B, C] */
              pos[1] = pv[0]; /*  P */
              types[1] = (int) TOUCHFACE;
              pos[2] = 3;     /*  [A, B, C] */
              pos[3] = pv[1]; /*  Q */
            } else {
              if (s4 == 0) {
                /*  Q = l, [P, Q] in [k, l] (-+-0). */
                types[0] = (int) TOUCHFACE;
                pos[0] = 3;     /*  [A, B, C] */
                pos[1] = pv[0]; /*  P */
                types[1] = (int) TOUCHEDGE;
                pos[2] = pu[1]; /*  [B, C] */
                pos[3] = pv[1]; /*  Q */
              } else { /*  s4 < 0 */
                /*  [P, Q] overlaps [k, l] (-+--). */
                types[0] = (int) TOUCHFACE;
                pos[0] = 3;     /*  [A, B, C] */
                pos[1] = pv[0]; /*  P */
                types[1] = (int) INTEREDGE;
                pos[2] = pu[1]; /*  [B, C] */
                pos[3] = pv[0]; /*  [P, Q] */
              }
            }
          } else { /*  s2 == 0 */
            /*  P = l (#0##). */
            types[0] = (int) TOUCHEDGE;
            pos[0] = pu[1]; /*  [B, C] */
            pos[1] = pv[0]; /*  P */
            types[1] = (int) DISJOINT;
          }
        }
      }
    } else { /*  s1 == 0 */
      /*  Q = k (0####) */
      types[0] = (int) TOUCHEDGE;
      pos[0] = pu[2]; /*  [C, A] */
      pos[1] = pv[1]; /*  Q */
      types[1] = (int) DISJOINT;
    }
  } else if (z1 == 2) {  /*  (tritri-23) */
    if (s1 < 0) {
      if (s3 > 0) {
        if (s2 <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        if (s4 > 0) {
          /*  [P, Q] overlaps [A, l] (-+++). */
          types[0] = (int) INTERVERT;
          pos[0] = pu[0]; /*  A */
          pos[1] = pv[0]; /*  [P, Q] */
          types[1] = (int) TOUCHFACE;
          pos[2] = 3;     /*  [A, B, C] */
          pos[3] = pv[1]; /*  Q */
        } else {
          if (s4 == 0) {
            /*  Q = l, [P, Q] contains [A, l] (-++0). */
            types[0] = (int) INTERVERT;
            pos[0] = pu[0]; /*  A */
            pos[1] = pv[0]; /*  [P, Q] */
            types[1] = (int) TOUCHEDGE;
            pos[2] = pu[1]; /*  [B, C] */
            pos[3] = pv[1]; /*  Q */
          } else { /*  s4 < 0 */
            /*  [P, Q] contains [A, l] (-++-). */
            types[0] = (int) INTERVERT;
            pos[0] = pu[0]; /*  A */
            pos[1] = pv[0]; /*  [P, Q] */
            types[1] = (int) INTEREDGE;
            pos[2] = pu[1]; /*  [B, C] */
            pos[3] = pv[0]; /*  [P, Q] */
          }
        }
      } else {
        if (s3 == 0) {
          if (s2 <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          if (s4 > 0) {
            /*  P = A, [P, Q] in [A, l] (-+0+). */
            types[0] = (int) SHAREVERTEX;
            pos[0] = pu[0]; /*  A */
            pos[1] = pv[0]; /*  P */
            types[1] = (int) TOUCHFACE;
            pos[2] = 3;     /*  [A, B, C] */
            pos[3] = pv[1]; /*  Q */
          } else {
            if (s4 == 0) {
              /*  [P, Q] = [A, l] (-+00). */
              types[0] = (int) SHAREVERTEX;
              pos[0] = pu[0]; /*  A */
              pos[1] = pv[0]; /*  P */
              types[1] = (int) TOUCHEDGE;
              pos[2] = pu[1]; /*  [B, C] */
              pos[3] = pv[1]; /*  Q */
            } else { /*  s4 < 0 */
              /*  Q = l, [P, Q] in [A, l] (-+0-). */
              types[0] = (int) SHAREVERTEX;
              pos[0] = pu[0]; /*  A */
              pos[1] = pv[0]; /*  P */
              types[1] = (int) INTEREDGE;
              pos[2] = pu[1]; /*  [B, C] */
              pos[3] = pv[0]; /*  [P, Q] */
            }
          }
        } else { /*  s3 < 0 */
          if (s2 > 0) {
            if (s4 > 0) {
              /*  [P, Q] in [A, l] (-+-+). */
              types[0] = (int) TOUCHFACE;
              pos[0] = 3;     /*  [A, B, C] */
              pos[1] = pv[0]; /*  P */
              types[0] = (int) TOUCHFACE;
              pos[0] = 3;     /*  [A, B, C] */
              pos[1] = pv[1]; /*  Q */
            } else {
              if (s4 == 0) {
                /*  Q = l, [P, Q] in [A, l] (-+-0). */
                types[0] = (int) TOUCHFACE;
                pos[0] = 3;     /*  [A, B, C] */
                pos[1] = pv[0]; /*  P */
                types[0] = (int) TOUCHEDGE;
                pos[0] = pu[1]; /*  [B, C] */
                pos[1] = pv[1]; /*  Q */
              } else { /*  s4 < 0 */
                /*  [P, Q] overlaps [A, l] (-+--). */
                types[0] = (int) TOUCHFACE;
                pos[0] = 3;     /*  [A, B, C] */
                pos[1] = pv[0]; /*  P */
                types[0] = (int) INTEREDGE;
                pos[0] = pu[1]; /*  [B, C] */
                pos[1] = pv[0]; /*  [P, Q] */
              }
            }
          } else { /*  s2 == 0 */
            /*  P = l (#0##). */
            types[0] = (int) TOUCHEDGE;
            pos[0] = pu[1]; /*  [B, C] */
            pos[1] = pv[0]; /*  P */
            types[1] = (int) DISJOINT;
          }
        }
      }
    } else { /*  s1 == 0 */
      /*  Q = A (0###). */
      types[0] = (int) SHAREVERTEX;
      pos[0] = pu[0]; /*  A */
      pos[1] = pv[1]; /*  Q */
      types[1] = (int) DISJOINT;
    }
  } else if (z1 == 3) {  /*  (tritri-33) */
    if (s1 < 0) {
      if (s3 > 0) {
        if (s2 <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        if (s4 > 0) {
          /*  [P, Q] overlaps [A, B] (-+++). */
          types[0] = (int) INTERVERT;
          pos[0] = pu[0]; /*  A */
          pos[1] = pv[0]; /*  [P, Q] */
          types[1] = (int) TOUCHEDGE;
          pos[2] = pu[0]; /*  [A, B] */
          pos[3] = pv[1]; /*  Q */
        } else {
          if (s4 == 0) {
            /*  Q = B, [P, Q] contains [A, B] (-++0). */
            types[0] = (int) INTERVERT;
            pos[0] = pu[0]; /*  A */
            pos[1] = pv[0]; /*  [P, Q] */
            types[1] = (int) SHAREVERTEX;
            pos[2] = pu[1]; /*  B */
            pos[3] = pv[1]; /*  Q */
          } else { /*  s4 < 0 */
            /*  [P, Q] contains [A, B] (-++-). */
            types[0] = (int) INTERVERT;
            pos[0] = pu[0]; /*  A */
            pos[1] = pv[0]; /*  [P, Q] */
            types[1] = (int) INTERVERT;
            pos[2] = pu[1]; /*  B */
            pos[3] = pv[0]; /*  [P, Q] */
          }
        }
      } else {
        if (s3 == 0) {
          if (s2 <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          if (s4 > 0) {
            /*  P = A, [P, Q] in [A, B] (-+0+). */
            types[0] = (int) SHAREVERTEX;
            pos[0] = pu[0]; /*  A */
            pos[1] = pv[0]; /*  P */
            types[1] = (int) TOUCHEDGE;
            pos[2] = pu[0]; /*  [A, B] */
            pos[3] = pv[1]; /*  Q */
          } else {
            if (s4 == 0) {
              /*  [P, Q] = [A, B] (-+00). */
              types[0] = (int) SHAREEDGE;
              pos[0] = pu[0]; /*  [A, B] */
              pos[1] = pv[0]; /*  [P, Q] */
              types[1] = (int) DISJOINT;
            } else { /*  s4 < 0 */
              /*  P= A, [P, Q] in [A, B] (-+0-). */
              types[0] = (int) SHAREVERTEX;
              pos[0] = pu[0]; /*  A */
              pos[1] = pv[0]; /*  P */
              types[1] = (int) INTERVERT;
              pos[2] = pu[1]; /*  B */
              pos[3] = pv[0]; /*  [P, Q] */
            }
          }
        } else { /*  s3 < 0 */
          if (s2 > 0) {
            if (s4 > 0) {
              /*  [P, Q] in [A, B] (-+-+). */
              types[0] = (int) TOUCHEDGE;
              pos[0] = pu[0]; /*  [A, B] */
              pos[1] = pv[0]; /*  P */
              types[1] = (int) TOUCHEDGE;
              pos[2] = pu[0]; /*  [A, B] */
              pos[3] = pv[1]; /*  Q */
            } else {
              if (s4 == 0) {
                /*  Q = B, [P, Q] in [A, B] (-+-0). */
                types[0] = (int) TOUCHEDGE;
                pos[0] = pu[0]; /*  [A, B] */
                pos[1] = pv[0]; /*  P */
                types[1] = (int) SHAREVERTEX;
                pos[2] = pu[1]; /*  B */
                pos[3] = pv[1]; /*  Q */
              } else { /*  s4 < 0 */
                /*  [P, Q] overlaps [A, B] (-+--). */
                types[0] = (int) TOUCHEDGE;
                pos[0] = pu[0]; /*  [A, B] */
                pos[1] = pv[0]; /*  P */
                types[1] = (int) INTERVERT;
                pos[2] = pu[1]; /*  B */
                pos[3] = pv[0]; /*  [P, Q] */
              }
            }
          } else { /*  s2 == 0 */
            /*  P = B (#0##). */
            types[0] = (int) SHAREVERTEX;
            pos[0] = pu[1]; /*  B */
            pos[1] = pv[0]; /*  P */
            types[1] = (int) DISJOINT;
          }
        }
      }
    } else { /*  s1 == 0 */
      /*  Q = A (0###). */
      types[0] = (int) SHAREVERTEX;
      pos[0] = pu[0]; /*  A */
      pos[1] = pv[1]; /*  Q */
      types[1] = (int) DISJOINT;
    }
  }
  if (isIntersect) {*isIntersect = 1;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTriEdgeTest"
/*  tri_edge_test()    Triangle-edge intersection test.                        */
/*                                                                             */
/*  This routine takes a triangle T (with vertices A, B, C) and an edge E (P,  */
/*  Q) in 3D, and tests if they intersect each other.  Return 1 if they are    */
/*  intersected, i.e., T \cap E is not empty, otherwise, return 0.             */
/*                                                                             */
/*  If the point 'R' is not NULL, it lies strictly above the plane defined by  */
/*  A, B, C. It is used in test when T and E are coplanar.                     */
/*                                                                             */
/*  If T1 and T2 intersect each other (return 1), they may intersect in diff-  */
/*  erent ways. If 'level' > 0, their intersection type will be reported in    */
/*  combinations of 'types' and 'pos'.                                         */
/* tetgenmesh::tri_edge_test() */
PetscErrorCode TetGenMeshTriEdgeTest(TetGenMesh *m, point A, point B, point C, point P, point Q, point R, int level, int *types, int *pos, int *isIntersect)
{
  TetGenOpts    *b = m->b;
  point U[3], V[3]; /* , Ptmp; */
  int pu[3], pv[3]; /* , itmp; */
  PetscReal sP, sQ, s1, s2, s3;
  int z1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Test the locations of P and Q with respect to ABC. */
  sP = TetGenOrient3D(A, B, C, P);
  sQ = TetGenOrient3D(A, B, C, Q);
  m->orient3dcount+=2;

  PetscInfo7(b->in, "      Tri-edge (%d %d %d)-(%d %d) (%c%c).\n", pointmark(m, A),
             pointmark(m, B), pointmark(m, C), pointmark(m, P), pointmark(m, Q),
             sP>0 ? '+' : (sP<0 ? '-' : '0'), sQ>0 ? '+' : (sQ<0 ? '-' : '0'));
  /*  triedgcount++; */

  if (sP < 0) {
    if (sQ < 0) { /*  (--) disjoint */
      if (isIntersect) {*isIntersect = 0;}
      PetscFunctionReturn(0);
    } else {
      if (sQ > 0) { /*  (-+) */
        SETVECTOR3(U, A, B, C);
        SETVECTOR3(V, P, Q, R);
        SETVECTOR3(pu, 0, 1, 2);
        SETVECTOR3(pv, 0, 1, 2);
        z1 = 0;
      } else { /*  (-0) */
        SETVECTOR3(U, A, B, C);
        SETVECTOR3(V, P, Q, R);
        SETVECTOR3(pu, 0, 1, 2);
        SETVECTOR3(pv, 0, 1, 2);
        z1 = 1;
      }
    }
  } else {
    if (sP > 0) { /*  (+-) */
      if (sQ < 0) {
        SETVECTOR3(U, A, B, C);
        SETVECTOR3(V, Q, P, R);  /*  P and Q are flipped. */
        SETVECTOR3(pu, 0, 1, 2);
        SETVECTOR3(pv, 1, 0, 2);
        z1 = 0;
      } else {
        if (sQ > 0) { /*  (++) disjoint */
          if (isIntersect) {*isIntersect = 0;}
          PetscFunctionReturn(0);
        } else { /*  (+0) */
          SETVECTOR3(U, B, A, C); /*  A and B are flipped. */
          SETVECTOR3(V, P, Q, R);
          SETVECTOR3(pu, 1, 0, 2);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 1;
        }
      }
    } else { /*  sP == 0 */
      if (sQ < 0) { /*  (0-) */
        SETVECTOR3(U, A, B, C);
        SETVECTOR3(V, Q, P, R);  /*  P and Q are flipped. */
        SETVECTOR3(pu, 0, 1, 2);
        SETVECTOR3(pv, 1, 0, 2);
        z1 = 1;
      } else {
        if (sQ > 0) { /*  (0+) */
          SETVECTOR3(U, B, A, C);  /*  A and B are flipped. */
          SETVECTOR3(V, Q, P, R);  /*  P and Q are flipped. */
          SETVECTOR3(pu, 1, 0, 2);
          SETVECTOR3(pv, 1, 0, 2);
          z1 = 1;
        } else { /*  (00) */
          /*  A, B, C, P, and Q are coplanar. */
          z1 = 2;
        }
      }
    }
  }

  if (z1 == 2) {
    int isInter;
    /*  The triangle and the edge are coplanar. */
    ierr = TetGenMeshTriEdge2D(m, A, B, C, P, Q, R, level, types, pos, &isInter);CHKERRQ(ierr);
    if (isIntersect) {*isIntersect = isInter;}
    PetscFunctionReturn(0);
  }

  s1 = TetGenOrient3D(U[0], U[1], V[0], V[1]); m->orient3dcount++;
  if (s1 < 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }

  s2 = TetGenOrient3D(U[1], U[2], V[0], V[1]); m->orient3dcount++;
  if (s2 < 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }

  s3 = TetGenOrient3D(U[2], U[0], V[0], V[1]); m->orient3dcount++;
  if (s3 < 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }

  PetscInfo5(b->in, "      Tri-edge (%d %d %d)-(%d %d).\n", pointmark(m, U[0]), pointmark(m, U[1]), pointmark(m, U[2]), pointmark(m, V[0]), pointmark(m, V[1]));
  PetscInfo3(b->in, "        (%c%c%c).\n", s1>0 ? '+' : (s1<0 ? '-' : '0'), s2>0 ? '+' : (s2<0 ? '-' : '0'), s3>0 ? '+' : (s3<0 ? '-' : '0'));

  if (level == 0) {
    if (isIntersect) {*isIntersect = 1;} /*  The are intersected. */
    PetscFunctionReturn(0);
  }

  types[1] = (int) DISJOINT; /*  No second intersection point. */

  if (z1 == 0) {
    if (s1 > 0) {
      if (s2 > 0) {
        if (s3 > 0) { /*  (+++) */
          /*  [P, Q] passes interior of [A, B, C]. */
          types[0] = (int) INTERFACE;
          pos[0] = 3;  /*  interior of [A, B, C] */
          pos[1] = 0;  /*  [P, Q] */
        } else { /*  s3 == 0 (++0) */
          /*  [P, Q] intersects [C, A]. */
          types[0] = (int) INTEREDGE;
          pos[0] = pu[2];  /*  [C, A] */
          pos[1] = 0;  /*  [P, Q] */
        }
      } else { /*  s2 == 0 */
        if (s3 > 0) { /*  (+0+) */
          /*  [P, Q] intersects [B, C]. */
          types[0] = (int) INTEREDGE;
          pos[0] = pu[1];  /*  [B, C] */
          pos[1] = 0;  /*  [P, Q] */
        } else { /*  s3 == 0 (+00) */
          /*  [P, Q] passes C. */
          types[0] = (int) INTERVERT;
          pos[0] = pu[2];  /*  C */
          pos[1] = 0;  /*  [P, Q] */
        }
      }
    } else { /*  s1 == 0 */
      if (s2 > 0) {
        if (s3 > 0) { /*  (0++) */
          /*  [P, Q] intersects [A, B]. */
          types[0] = (int) INTEREDGE;
          pos[0] = pu[0];  /*  [A, B] */
          pos[1] = 0;  /*  [P, Q] */
        } else { /*  s3 == 0 (0+0) */
          /*  [P, Q] passes A. */
          types[0] = (int) INTERVERT;
          pos[0] = pu[0];  /*  A */
          pos[1] = 0;  /*  [P, Q] */
        }
      } else { /*  s2 == 0 */
        if (s3 > 0) { /*  (00+) */
          /*  [P, Q] passes B. */
          types[0] = (int) INTERVERT;
          pos[0] = pu[1];  /*  B */
          pos[1] = 0;  /*  [P, Q] */
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Impossible");
      }
    }
  } else { /*  z1 == 1 */
    if (s1 > 0) {
      if (s2 > 0) {
        if (s3 > 0) { /*  (+++) */
          /*  Q lies in [A, B, C]. */
          types[0] = (int) TOUCHFACE;
          pos[0] = 0; /*  [A, B, C] */
          pos[1] = pv[1]; /*  Q */
        } else { /*  s3 == 0 (++0) */
          /*  Q lies on [C, A]. */
          types[0] = (int) TOUCHEDGE;
          pos[0] = pu[2]; /*  [C, A] */
          pos[1] = pv[1]; /*  Q */
        }
      } else { /*  s2 == 0 */
        if (s3 > 0) { /*  (+0+) */
          /*  Q lies on [B, C]. */
          types[0] = (int) TOUCHEDGE;
          pos[0] = pu[1]; /*  [B, C] */
          pos[1] = pv[1]; /*  Q */
        } else { /*  s3 == 0 (+00) */
          /*  Q = C. */
          types[0] = (int) SHAREVERTEX;
          pos[0] = pu[2]; /*  C */
          pos[1] = pv[1]; /*  Q */
        }
      }
    } else { /*  s1 == 0 */
      if (s2 > 0) {
        if (s3 > 0) { /*  (0++) */
          /*  Q lies on [A, B]. */
          types[0] = (int) TOUCHEDGE;
          pos[0] = pu[0]; /*  [A, B] */
          pos[1] = pv[1]; /*  Q */
        } else { /*  s3 == 0 (0+0) */
          /*  Q = A. */
          types[0] = (int) SHAREVERTEX;
          pos[0] = pu[0]; /*  A */
          pos[1] = pv[1]; /*  Q */
        }
      } else { /*  s2 == 0 */
        if (s3 > 0) { /*  (00+) */
          /*  Q = B. */
          types[0] = (int) SHAREVERTEX;
          pos[0] = pu[1]; /*  B */
          pos[1] = pv[1]; /*  Q */
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Impossible");
      }
    }
  }

  if (isIntersect) {*isIntersect = 1;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInCircle3D"
/*  incircle3d()    3D in-circle test.                                         */
/*                                                                             */
/*  Return a negative value if pd is inside the circumcircle of the triangle   */
/*  pa, pb, and pc.                                                            */
/* tetgenmesh::incirlce3d() */
PetscErrorCode TetGenMeshInCircle3D(TetGenMesh *m, point pa, point pb, point pc, point pd, PetscReal *signTest)
{
  TetGenOpts    *b = m->b;
  PetscReal area2[2], n1[3], n2[3], c[3];
  PetscReal sign, r, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Calculate the areas of the two triangles [a, b, c] and [b, a, d]. */
  ierr = TetGenMeshFaceNormal2(m, pa, pb, pc, n1, 1);CHKERRQ(ierr);
  area2[0] = DOT(n1, n1);
  ierr = TetGenMeshFaceNormal2(m, pb, pa, pd, n2, 1);CHKERRQ(ierr);
  area2[1] = DOT(n2, n2);

  if (area2[0] > area2[1]) {
    /*  Choose [a, b, c] as the base triangle. */
    ierr = TetGenMeshCircumsphere(m, pa, pb, pc, PETSC_NULL, c, &r, PETSC_NULL);CHKERRQ(ierr);
    d = DIST(c, pd);
  } else {
    /*  Choose [b, a, d] as the base triangle. */
    if (area2[1] > 0) {
      ierr = TetGenMeshCircumsphere(m, pb, pa, pd, PETSC_NULL, c, &r, PETSC_NULL);CHKERRQ(ierr);
      d = DIST(c, pc);
    } else {
      /*  The four points are collinear. This case only happens on the boundary. */
      return 0; /*  Return "not inside". */
    }
  }

  sign = d - r;
  if (fabs(sign) / r < b->epsilon) {
    sign = 0;
  }

  *signTest = sign;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInSphereS"
/*  insphere_s()    Insphere test with symbolic perturbation.                  */
/*                                                                             */
/*  Given four points pa, pb, pc, and pd, test if the point pe lies inside or  */
/*  outside the circumscirbed sphere of the four points.  Here we assume that  */
/*  the orientation of the sequence {pa, pb, pc, pd} is negative (NOT zero),   */
/*  i.e., pd lies at the negative side of the plane defined by pa, pb, and pc. */
/*                                                                             */
/*  Return a positive value (> 0) if pe lies outside, a negative value (< 0)   */
/*  if pe lies inside the sphere, the returned value will not be zero.         */
/* tetgenmesh::insphere_s() */
PetscErrorCode TetGenMeshInSphereS(TetGenMesh *m, PetscReal* pa, PetscReal* pb, PetscReal* pc, PetscReal* pd, PetscReal* pe, PetscReal *isOutside)
{
  PetscReal sign;
  /*  Symbolic perturbation. */
  point pt[5], swappt;
  PetscReal oriA, oriB;
  int swaps, count;
  int n, i;

  PetscFunctionBegin;
  m->inspherecount++;
  sign = TetGenInsphere(pa, pb, pc, pd, pe);
  if (sign != 0.0) {
    *isOutside = sign;
    PetscFunctionReturn(0);
  }
  m->insphere_sos_count++;

  pt[0] = pa;
  pt[1] = pb;
  pt[2] = pc;
  pt[3] = pd;
  pt[4] = pe;

  /*  Sort the five points such that their indices are in the increasing */
  /*    order. An optimized bubble sort algorithm is used, i.e., it has */
  /*    the worst case O(n^2) runtime, but it is usually much faster. */
  swaps = 0; /*  Record the total number of swaps. */
  n = 5;
  do {
    count = 0;
    n = n - 1;
    for (i = 0; i < n; i++) {
      if (pointmark(m, pt[i]) > pointmark(m, pt[i+1])) {
        swappt = pt[i]; pt[i] = pt[i+1]; pt[i+1] = swappt;
        count++;
      }
    }
    swaps += count;
  } while (count > 0); /*  Continue if some points are swapped. */

  oriA = TetGenOrient3D(pt[1], pt[2], pt[3], pt[4]);
  if (oriA != 0.0) {
    /*  Flip the sign if there are odd number of swaps. */
    if ((swaps % 2) != 0) oriA = -oriA;
    *isOutside = oriA;
    PetscFunctionReturn(0);
  }

  oriB = -TetGenOrient3D(pt[0], pt[2], pt[3], pt[4]);
  if (oriB == 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
  /*  Flip the sign if there are odd number of swaps. */
  if ((swaps % 2) != 0) oriB = -oriB;
  *isOutside = oriB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshIsCollinear"
/*  iscollinear()    Check if three points are approximately collinear.        */
/*                                                                             */
/*  'eps' is a relative error tolerance.  The collinearity is determined by    */
/*  the value q = cos(theta), where theta is the angle between two vectors     */
/*  A->B and A->C.  They're collinear if 1.0 - q <= epspp.                     */
/* tetgenmesh::iscollinear() */
PetscErrorCode TetGenMeshIsCollinear(TetGenMesh *m, PetscReal *A, PetscReal *B, PetscReal *C, PetscReal eps, PetscBool *co)
{
  PetscReal abx, aby, abz;
  PetscReal acx, acy, acz;
  PetscReal Lv, Lw, dd;
  PetscReal d, q;

  PetscFunctionBegin;
  /*  Limit of two closed points. */
  q = m->longest * eps;
  q *= q;

  abx = A[0] - B[0];
  aby = A[1] - B[1];
  abz = A[2] - B[2];
  acx = A[0] - C[0];
  acy = A[1] - C[1];
  acz = A[2] - C[2];
  Lv = abx * abx + aby * aby + abz * abz;
  /*  Is AB (nearly) indentical? */
  if (Lv < q) {
    *co = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  Lw = acx * acx + acy * acy + acz * acz;
  /*  Is AC (nearly) indentical? */
  if (Lw < q) {
    *co = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  dd = abx * acx + aby * acy + abz * acz;

  d = (dd * dd) / (Lv * Lw);
  if (d > 1.0) d = 1.0; /*  Rounding. */
  q = 1.0 - sqrt(d); /*  Notice 0 < q < 1.0. */

  *co = q <= eps ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshIsCoplanar"
/*  iscoplanar()    Check if four points are approximately coplanar.           */
/*                                                                             */
/*  'vol6' is six times of the signed volume of the tetrahedron formed by the  */
/*  four points. 'eps' is the relative error tolerance.  The coplanarity is    */
/*  determined by the value: q = fabs(vol6) / L^3,  where L is the average     */
/*  edge length of the tet. They're coplanar if q <= eps.                      */
/* tetgenmesh::iscoplanar() */
PetscErrorCode TetGenMeshIsCoplanar(TetGenMesh *mesh, PetscReal *k, PetscReal *l, PetscReal *m, PetscReal *n, PetscReal vol6, PetscReal eps, PetscBool *co)
{
  PetscReal L, q;
  PetscReal x, y, z;

  PetscFunctionBegin;
  if (vol6 == 0.0) {
    *co = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  x = k[0] - l[0];
  y = k[1] - l[1];
  z = k[2] - l[2];
  L = sqrt(x * x + y * y + z * z);
  x = l[0] - m[0];
  y = l[1] - m[1];
  z = l[2] - m[2];
  L += sqrt(x * x + y * y + z * z);
  x = m[0] - k[0];
  y = m[1] - k[1];
  z = m[2] - k[2];
  L += sqrt(x * x + y * y + z * z);
  x = k[0] - n[0];
  y = k[1] - n[1];
  z = k[2] - n[2];
  L += sqrt(x * x + y * y + z * z);
  x = l[0] - n[0];
  y = l[1] - n[1];
  z = l[2] - n[2];
  L += sqrt(x * x + y * y + z * z);
  x = m[0] - n[0];
  y = m[1] - n[1];
  z = m[2] - n[2];
  L += sqrt(x * x + y * y + z * z);
#ifdef PETSC_USE_DEBUG
  if (L <= 0.0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Length %g should be positive", L);
#endif
  L /= 6.0;
  q = fabs(vol6) / (L * L * L);

  *co = q <= eps ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTetAllNormal"
/*  tetallnormal()    Get the in-noramls of the four faces of a given tet.     */
/*                                                                             */
/*  Let tet be abcd. N[4][3] returns the four normals, which are: N[0] cbd,    */
/*  N[1] acd, N[2] bad, N[3] abc. These normals are unnormalized.              */
/* tetgenmesh::tetallnormal() */
PetscErrorCode TetGenMeshTetAllNormal(TetGenMesh *m, point pa, point pb, point pc, point pd, PetscReal N[4][3], PetscReal *volume)
{
  PetscReal A[4][4], rhs[4], D;
  int       indx[4];
  int       i, j;

  PetscFunctionBegin;
  /*  get the entries of A[3][3]. */
  for(i = 0; i < 3; i++) A[0][i] = pa[i] - pd[i];  /*  d->a vec */
  for(i = 0; i < 3; i++) A[1][i] = pb[i] - pd[i];  /*  d->b vec */
  for(i = 0; i < 3; i++) A[2][i] = pc[i] - pd[i];  /*  d->c vec */
  /*  Compute the inverse of matrix A, to get 3 normals of the 4 faces. */
  TetGenLUDecomp(A, 3, indx, &D, 0);     /*  Decompose the matrix just once. */
  if (volume) {
    /*  Get the volume of the tet. */
    *volume = fabs((A[indx[0]][0] * A[indx[1]][1] * A[indx[2]][2])) / 6.0;
  }
  for(j = 0; j < 3; j++) {
    for(i = 0; i < 3; i++) rhs[i] = 0.0;
    rhs[j] = 1.0;  /*  Positive means the inside direction */
    TetGenLUSolve(A, 3, indx, rhs, 0);
    for (i = 0; i < 3; i++) N[j][i] = rhs[i];
  }
  /*  Get the fourth normal by summing up the first three. */
  for(i = 0; i < 3; i++) N[3][i] = - N[0][i] - N[1][i] - N[2][i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTetAllDihedral"
/*  tetalldihedral()    Get all (six) dihedral angles of a tet.                */
/*                                                                             */
/*  The tet is given by its four corners a, b, c, and d. If 'cosdd' is not     */
/*  NULL, it returns the cosines of the 6 dihedral angles, the corresponding   */
/*  edges are: ab, bc, ca, ad, bd, and cd. If 'cosmaxd' (or 'cosmind') is not  */
/*  NULL, it returns the cosine of the maximal (or minimal) dihedral angle.    */
/* tetgenmesh::tetalldihedral() */
PetscErrorCode TetGenMeshTetAllDihedral(TetGenMesh *m, point pa, point pb, point pc, point pd, PetscReal *cosdd, PetscReal *cosmaxd, PetscReal *cosmind)
{
  PetscReal N[4][3], vol, cosd, len;
  int f1 = 0, f2 = 0, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  vol = 0; /*  Check if the tet is valid or not. */

  /*  Get four normals of faces of the tet. */
  ierr = TetGenMeshTetAllNormal(m, pa, pb, pc, pd, N, &vol);CHKERRQ(ierr);

  if (vol == 0.0) {
    /*  This tet is not valid. */
    if (cosdd != NULL) {
      for (i = 0; i < 6; i++) {
        cosdd[i] = -1.0; /*  180 degree. */
      }
    }
    /*  This tet has zero volume. */
    if (cosmaxd != NULL) {
      *cosmaxd = -1.0; /*  180 degree. */
    }
    if (cosmind != NULL) {
      *cosmind = 1.0; /*  0 degree. */
    }
    PetscFunctionReturn(0);
  }

  /*  Normalize the normals. */
  for (i = 0; i < 4; i++) {
    len = sqrt(dot(N[i], N[i]));
    if (len != 0.0) {
      for (j = 0; j < 3; j++) N[i][j] /= len;
    }
  }

  for (i = 0; i < 6; i++) {
    switch (i) {
    case 0: f1 = 2; f2 = 3; break; /*  edge ab. */
    case 1: f1 = 0; f2 = 3; break; /*  edge bc. */
    case 2: f1 = 1; f2 = 3; break; /*  edge ca. */
    case 3: f1 = 1; f2 = 2; break; /*  edge ad. */
    case 4: f1 = 2; f2 = 0; break; /*  edge bd. */
    case 5: f1 = 0; f2 = 1; break; /*  edge cd. */
    }
    cosd = -dot(N[f1], N[f2]);
    if (cosdd) cosdd[i] = cosd;
    if (i == 0) {
      if (cosmaxd) *cosmaxd = cosd;
      if (cosmind) *cosmind = cosd;
    } else {
      if (cosmaxd) *cosmaxd = cosd < *cosmaxd ? cosd : *cosmaxd;
      if (cosmind) *cosmind = cosd > *cosmind ? cosd : *cosmind;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFaceDihedral"
/*  facedihedral()    Return the dihedral angle (in radian) between two        */
/*                    adjoining faces.                                         */
/*                                                                             */
/*  'pa', 'pb' are the shared edge of these two faces, 'pc1', and 'pc2' are    */
/*  apexes of these two faces.  Return the angle (between 0 to 2*pi) between   */
/*  the normal of face (pa, pb, pc1) and normal of face (pa, pb, pc2).         */
/* tetgenmesh::facedihedral() */
PetscErrorCode TetGenMeshFaceDihedral(TetGenMesh *m, PetscReal* pa, PetscReal* pb, PetscReal* pc1, PetscReal* pc2, PetscReal *angle)
{
  PetscReal n1[3], n2[3];
  PetscReal n1len, n2len;
  PetscReal costheta, ori;
  PetscReal theta;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TetGenMeshFaceNormal(m, pa, pb, pc1, n1, &n1len);CHKERRQ(ierr);
  ierr = TetGenMeshFaceNormal(m, pa, pb, pc2, n2, &n2len);CHKERRQ(ierr);
  costheta = dot(n1, n2) / (n1len * n2len);
  /*  Be careful rounding error! */
  if (costheta > 1.0) {
    costheta = 1.0;
  } else if (costheta < -1.0) {
    costheta = -1.0;
  }
  theta = acos(costheta);
  ori   = TetGenOrient3D(pa, pb, pc1, pc2);
  if (ori > 0.0) {
    theta = 2 * PETSC_PI - theta;
  }

  *angle = theta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTetAspectRatio"
/*  tetaspectratio()    Calculate the aspect ratio of the tetrahedron.         */
/*                                                                             */
/*  The aspect ratio of a tet is R/h, where R is the circumradius and h is     */
/*  the shortest height of the tet.                                            */
/* tetgenmesh::tetaspectratio() */
PetscErrorCode TetGenMeshTetAspectRatio(TetGenMesh *m, point pa, point pb, point pc, point pd, PetscReal *ratio)
{
  PetscReal vda[3], vdb[3], vdc[3];
  PetscReal N[4][3], A[4][4], rhs[4], D;
  PetscReal H[4], volume, radius2, minheightinv;
  int indx[4];
  int i, j;

  PetscFunctionBegin;
  /*  Set the matrix A = [vda, vdb, vdc]^T. */
  for(i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
  for(i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
  for(i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];
  /*  Lu-decompose the matrix A. */
  TetGenLUDecomp(A, 3, indx, &D, 0);
  /*  Get the volume of abcd. */
  volume = (A[indx[0]][0] * A[indx[1]][1] * A[indx[2]][2]) / 6.0;
  /*  Check if it is zero. */
  if (volume == 0.0) {
    if (ratio) {*ratio = 1.0e+200;} /*  A degenerate tet. */
    PetscFunctionReturn(0);
  }
  /*  if (volume < 0.0) volume = -volume; */
  /*  Check the radiu-edge ratio of the tet. */
  rhs[0] = 0.5 * dot(vda, vda);
  rhs[1] = 0.5 * dot(vdb, vdb);
  rhs[2] = 0.5 * dot(vdc, vdc);
  TetGenLUSolve(A, 3, indx, rhs, 0);
  /*  Get the circumcenter. */
  /*  for (i = 0; i < 3; i++) circumcent[i] = pd[i] + rhs[i]; */
  /*  Get the square of the circumradius. */
  radius2 = dot(rhs, rhs);

  /*  Compute the 4 face normals (N[0], ..., N[3]). */
  for(j = 0; j < 3; j++) {
    for(i = 0; i < 3; i++) rhs[i] = 0.0;
    rhs[j] = 1.0;  /*  Positive means the inside direction */
    TetGenLUSolve(A, 3, indx, rhs, 0);
    for(i = 0; i < 3; i++) N[j][i] = rhs[i];
  }
  /*  Get the fourth normal by summing up the first three. */
  for(i = 0; i < 3; i++) N[3][i] = - N[0][i] - N[1][i] - N[2][i];
  /*  Normalized the normals. */
  for(i = 0; i < 4; i++) {
    /*  H[i] is the inverse of the height of its corresponding face. */
    H[i] = sqrt(dot(N[i], N[i]));
  }
  /*  Get the radius of the inscribed sphere. */
  /*  insradius = 1.0 / (H[0] + H[1] + H[2] + H[3]); */
  /*  Get the biggest H[i] (corresponding to the smallest height). */
  minheightinv = H[0];
  for(i = 1; i < 3; i++) {
    if (H[i] > minheightinv) minheightinv = H[i];
  }
  if (ratio) {*ratio = sqrt(radius2) * minheightinv;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPreciseLocate"
/*  preciselocate()    Find a simplex containing a given point.                */
/*                                                                             */
/*  This routine implements the simple Walk-through point location algorithm.  */
/*  Begins its search from 'searchtet', assume there is a line segment L from  */
/*  a vertex of 'searchtet' to the query point 'searchpt', and simply walk     */
/*  towards 'searchpt' by traversing all faces intersected by L.               */
/*                                                                             */
/*  On completion, 'searchtet' is a tetrahedron that contains 'searchpt'. The  */
/*  returned value indicates one of the following cases:                       */
/*    - ONVERTEX, the search point lies on the origin of 'searchtet'.          */
/*    - ONEDGE, the search point lies on an edge of 'searchtet'.               */
/*    - ONFACE, the search point lies on a face of 'searchtet'.                */
/*    - INTET, the search point lies in the interior of 'searchtet'.           */
/*    - OUTSIDE, the search point lies outside the mesh. 'searchtet' is a      */
/*      hull tetrahedron whose base face is visible by the search point.       */
/*                                                                             */
/*  WARNING: This routine is designed for convex triangulations, and will not  */
/*  generally work after the holes and concavities have been carved.           */
/*                                                                             */
/*  If 'maxtetnumber' > 0, stop the searching process if the number of passed  */
/*  tets is larger than it and return OUTSIDE.                                 */
/* tetgenmesh::preciselocate() */
PetscErrorCode TetGenMeshPreciseLocate(TetGenMesh *m, point searchpt, triface *searchtet, long maxtetnumber, locateresult *result)
{
  TetGenOpts    *b  = m->b;
  triface backtracetet = {PETSC_NULL, 0, 0};
  triface walkthroface = {PETSC_NULL, 0, 0};
  point forg, fdest, fapex, toppo;
  PetscReal ori1, ori2, ori3, ori4;
  long tetnumber;
  int side;

  PetscFunctionBegin;
  if (isdead_triface(searchtet)) searchtet->tet = m->dummytet;
  if (searchtet->tet == m->dummytet) {
    searchtet->loc = 0;
    symself(searchtet);
  }
  /*  'searchtet' should be a valid tetrahedron now. */
#ifdef PETSC_USE_DEBUG
  if (searchtet->tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif

  searchtet->ver = 0; /*  Keep in CCW edge ring. */
  /*  Find a face of 'searchtet' such that the 'searchpt' lies strictly */
  /*    above it.  Such face should always exist. */
  for(searchtet->loc = 0; searchtet->loc < 4; searchtet->loc++) {
    forg = org(searchtet);
    fdest = dest(searchtet);
    fapex = apex(searchtet);
    ori1 = TetGenOrient3D(forg, fdest, fapex, searchpt);
    if (ori1 < 0.0) break;
  }
#ifdef PETSC_USE_DEBUG
  if (searchtet->loc >= 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif

  backtracetet = *searchtet; /*  Initialize backtracetet. */

  /*  Define 'tetnumber' for exit the loop when it's running endless. */
  tetnumber = 0l;
  while ((maxtetnumber > 0l) && (tetnumber <= maxtetnumber)) {
    m->ptloc_count++;  /*  Algorithimic count. */
    /*  Check if we are reaching the boundary of the triangulation. */
    if (searchtet->tet == m->dummytet) {
      *searchtet = backtracetet;
      if (result) {*result = OUTSIDE;}
      PetscFunctionReturn(0);
    }
    /*  Initialize the face for returning the walk-through face. */
    walkthroface.tet = PETSC_NULL;
    /*  Adjust the edge ring, so that 'ori1 < 0.0' holds. */
    searchtet->ver = 0;
    /*  'toppo' remains unchange for the following orientation tests. */
    toppo = oppo(searchtet);
    /*  Check the three sides of 'searchtet' to find the face through which */
    /*    we can walk next. */
    for(side = 0; side < 3; side++) {
      forg = org(searchtet);
      fdest = dest(searchtet);
      ori2 = TetGenOrient3D(forg, fdest, toppo, searchpt);
      if (ori2 == 0.0) {
        /*  They are coplanar, check if 'searchpt' lies inside, or on an edge, */
        /*    or coindes with a vertex of face (forg, fdest, toppo). */
        fapex = apex(searchtet);
        ori3 = TetGenOrient3D(fdest, fapex, toppo, searchpt);
        if (ori3 < 0.0) {
          /*  Outside the face (fdest, fapex, toppo), walk through it. */
          enextself(searchtet);
          fnext(m, searchtet, &walkthroface);
          break;
        }
        ori4 = TetGenOrient3D(fapex, forg, toppo, searchpt);
        if (ori4 < 0.0) {
          /*  Outside the face (fapex, forg, toppo), walk through it. */
          enext2self(searchtet);
          fnext(m, searchtet, &walkthroface);
          break;
        }
        /*  Remember, ori1 < 0.0, which means that 'searchpt' will not on edge */
        /*    (forg, fdest) or on vertex forg or fdest. */
        /*  The rest possible cases are: */
        /*    (1) 'searchpt' lies on edge (fdest, toppo); */
        /*    (2) 'searchpt' lies on edge (toppo, forg); */
        /*    (3) 'searchpt' coincident with toppo; */
        /*    (4) 'searchpt' lies inside face (forg, fdest, toppo). */
        fnextself(m, searchtet);
        if (ori3 == 0.0) {
          if (ori4 == 0.0) {
            /*  Case (4). */
            enext2self(searchtet);
            if (result) {*result = ONVERTEX;}
            PetscFunctionReturn(0);
          } else {
            /*  Case (1). */
            enextself(searchtet);
            if (result) {*result = ONEDGE;}
            PetscFunctionReturn(0);
          }
        }
        if (ori4 == 0.0) {
          /*  Case (2). */
          enext2self(searchtet);
          if (result) {*result = ONEDGE;}
          PetscFunctionReturn(0);
        }
        /*  Case (4). */
        if (result) {*result = ONFACE;}
        PetscFunctionReturn(0);
      } else if (ori2 < 0.0) {
        /*  Outside the face (forg, fdest, toppo), walk through it. */
        fnext(m, searchtet, &walkthroface);
        break;
      }
      /*  Go to check next side. */
      enextself(searchtet);
    }
    if (side == 3) {
      /*  Found! Inside tetrahedron. */
      if (result) {*result = INTETRAHEDRON;}
      PetscFunctionReturn(0);
    }
    /*  We walk through the face 'walkthroface' and continue the searching. */
    /*  Store the face handle in 'backtracetet' before we take the real walk. */
    /*    So we are able to restore the handle to 'searchtet' if we are */
    /*    reaching the outer boundary. */
    backtracetet = walkthroface;
    sym(&walkthroface, searchtet);
    tetnumber++;
  }

  PetscInfo1(b->in, "Warning:  Point location stopped after searching %ld tets.\n", maxtetnumber);
  if (result) {*result = OUTSIDE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRandomSample"
/*  randomsample()    Randomly sample the tetrahedra for point loation.        */
/*                                                                             */
/*  This routine implements Muecke's Jump-and-walk point location algorithm.   */
/*  It improves the simple walk-through by "jumping" to a good starting point  */
/*  via random sampling.  Searching begins from one of handles:  the input     */
/*  'searchtet', a recently encountered tetrahedron 'recenttet',  or from one  */
/*  chosen from a random sample.  The choice is made by determining which one  */
/*  's origin is closest to the point we are searcing for.  Having chosen the  */
/*  starting tetrahedron, the simple Walk-through algorithm is executed.       */
/* tetgenmesh::randomsample() */
PetscErrorCode TetGenMeshRandomSample(TetGenMesh *m, point searchpt, triface *searchtet)
{
  tetrahedron *firsttet, *tetptr;
  void **sampleblock;
  long sampleblocks, samplesperblock;
  int  samplenum;
  long tetblocks, i, j;
  PETSC_UINTPTR_T alignptr;
  PetscReal searchdist, dist;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  'searchtet' should be a valid tetrahedron. */
  if (isdead_triface(searchtet)) {
    searchtet->tet = m->dummytet;
  }
  if (searchtet->tet == m->dummytet) {
    /*  This is an 'Outer Space' handle, get a hull tetrahedron. */
    searchtet->loc = 0;
    symself(searchtet);
  }

  /*  Note 'searchtet' may be dead (chnaged in constrainedcavity2()). */
  if (!isdead_triface(searchtet)) {
    /*  Get the distance from the suggested starting tet to the point we seek. */
    ierr = TetGenMeshDistance2(m, searchtet->tet, searchpt, &searchdist);CHKERRQ(ierr);
  } else {
    searchdist = m->longest * m->longest;
  }

  /*  If a recently encountered tetrahedron has been recorded and has not */
  /*    been deallocated, test it as a good starting point. */
  if (!isdead_triface(&m->recenttet) && (m->recenttet.tet != searchtet->tet)) {
    ierr = TetGenMeshDistance2(m, m->recenttet.tet, searchpt, &dist);CHKERRQ(ierr);
    if (dist < searchdist) {
      *searchtet = m->recenttet;
      searchdist = dist;
    }
  }

  /*  Select "good" candidate using k random samples, taking the closest one. */
  /*    The number of random samples taken is proportional to the fourth root */
  /*    of the number of tetrahedra in the mesh. The next bit of code assumes */
  /*    that the number of tetrahedra increases monotonically. */
  while(SAMPLEFACTOR * m->samples * m->samples * m->samples * m->samples < m->tetrahedrons->items) {
    m->samples++;
  }
  /*  Find how much blocks in current tet pool. */
  tetblocks = (m->tetrahedrons->maxitems + ELEPERBLOCK - 1) / ELEPERBLOCK;
  /*  Find the average samles per block. Each block at least have 1 sample. */
  samplesperblock = 1 + (m->samples / tetblocks);
  sampleblocks = m->samples / samplesperblock;
  sampleblock = m->tetrahedrons->firstblock;
  for(i = 0; i < sampleblocks; i++) {
    alignptr = (PETSC_UINTPTR_T) (sampleblock + 1);
    firsttet = (tetrahedron *) (alignptr + (PETSC_UINTPTR_T) m->tetrahedrons->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->tetrahedrons->alignbytes));
    for(j = 0; j < samplesperblock; j++) {
      if (i == tetblocks - 1) {
        /*  This is the last block. */
        ierr = TetGenMeshRandomChoice(m, (int) (m->tetrahedrons->maxitems - (i * ELEPERBLOCK)), &samplenum);CHKERRQ(ierr);
      } else {
        ierr = TetGenMeshRandomChoice(m, ELEPERBLOCK, &samplenum);CHKERRQ(ierr);
      }
      tetptr = (tetrahedron *) (firsttet + (samplenum * m->tetrahedrons->itemwords));
      if (tetptr[4]) {
        ierr = TetGenMeshDistance2(m, tetptr, searchpt, &dist);CHKERRQ(ierr);
        if (dist < searchdist) {
          searchtet->tet = tetptr;
          searchdist = dist;
        }
      }
    }
    sampleblock = (void **) *sampleblock;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshLocate"
/*  locate()    Find a simplex containing a given point.                       */
/* tetgenmesh::locate() */
PetscErrorCode TetGenMeshLocate(TetGenMesh *m, point searchpt, triface *searchtet, locateresult *result)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Randomly sample for a good starting tet. */
  ierr = TetGenMeshRandomSample(m, searchpt, searchtet);CHKERRQ(ierr);
  /*  Call simple walk-through to locate the point. */
  ierr = TetGenMeshPreciseLocate(m, searchpt, searchtet, m->tetrahedrons->items, result);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshLocate2"
/*  locate2()    Find a simplex containing a given point.                      */
/*                                                                             */
/*  Another implementation of the Walk-through point location algorithm.       */
/*  See the comments of preciselocate().                                       */
/* tetgenmesh::locate2() */
PetscErrorCode TetGenMeshLocate2(TetGenMesh *m, point searchpt, triface *searchtet, ArrayPool *histtetarray, locateresult *result)
{
  triface neightet = {PETSC_NULL, 0, 0}, backtracetet = {PETSC_NULL, 0, 0}, *parytet;
  point torg, tdest, tapex, toppo, ntoppo;
  enum {ORGMOVE, DESTMOVE, APEXMOVE} nextmove;
  PetscReal ori, oriorg, oridest, oriapex;
  PetscReal searchdist, dist;
  locateresult loc;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (searchtet->tet == m->dummytet) {
    /*  A hull tet. Choose the neighbor of its base face. */
    searchtet->loc = 0;
    symself(searchtet);
  }

  /*  Stay in the 0th edge ring. */
  searchtet->ver = 0;

  /*  Let searchtet be the face such that 'searchpt' lies above to it. */
  for(searchtet->loc = 0; searchtet->loc < 4; searchtet->loc++) {
    torg = org(searchtet);
    tdest = dest(searchtet);
    tapex = apex(searchtet);
    ori = TetGenOrient3D(torg, tdest, tapex, searchpt); m->orient3dcount++;
    if (ori < 0.0) break;
  }
  if (!(searchtet->loc < 4)) {
    /*  Either 'searchtet' is a very flat tet, or the 'searchpt' lies in */
    /*    infinity, or both of them. Return OUTSIDE. */
    return OUTSIDE;
  }

  if (histtetarray) {
    /*  Remember all the tets we've visited. */
    if (histtetarray->objects != 0l) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    infect(m, searchtet);
    ierr = ArrayPoolNewIndex(histtetarray, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
    *parytet = *searchtet;
  }

  loc = OUTSIDE; /*  Set a default return value. */

  /*  Walk through tetrahedra to locate the point. */
  while(1) {

    m->ptloc_count++;  /*  Algorithimic count. */

    toppo = oppo(searchtet);

    /*  Check if the vertex is we seek. */
    if (toppo == searchpt) {
      /*  Adjust the origin of searchtet to be searchpt. */
      fnextself(m, searchtet);
      esymself(searchtet);
      enext2self(searchtet);
      loc = ONVERTEX; /*  return ONVERTEX; */
      break;
    }

    /*  We enter from serarchtet's base face. There are three other faces in */
    /*    searchtet (all connecting to toppo), which one is the exit? */
    oriorg = TetGenOrient3D(tdest, tapex, toppo, searchpt);
    oridest = TetGenOrient3D(tapex, torg, toppo, searchpt);
    oriapex = TetGenOrient3D(torg, tdest, toppo, searchpt);
    m->orient3dcount+=3;

    /*  Now decide which face to move. It is possible there are more than one */
    /*    faces are viable moves. Use the opposite points of thier neighbors */
    /*    to discriminate, i.e., we choose the face whose opposite point has */
    /*    the shortest distance to searchpt. */
    if (oriorg < 0) {
      if (oridest < 0) {
        if (oriapex < 0) {
          /*  Any of the three faces is a viable move. */
          nextmove = ORGMOVE;
          enextfnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            searchdist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            searchdist = NORM2(m->xmax - m->xmin, m->ymax - m->ymin, m->zmax - m->zmin);
          }
          enext2fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            dist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            dist = searchdist;
          }
          if (dist < searchdist) {
            nextmove = DESTMOVE;
            searchdist = dist;
          }
          fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            dist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            dist = searchdist;
          }
          if (dist < searchdist) {
            nextmove = APEXMOVE;
            searchdist = dist;
          }
        } else {
          /*  Two faces, opposite to origin and destination, are viable. */
          nextmove = ORGMOVE;
          enextfnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            searchdist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            searchdist = NORM2(m->xmax - m->xmin, m->ymax - m->ymin, m->zmax - m->zmin);
          }
          enext2fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            dist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            dist = searchdist;
          }
          if (dist < searchdist) {
            nextmove = DESTMOVE;
            searchdist = dist;
          }
        }
      } else {
        if (oriapex < 0) {
          /*  Two faces, opposite to origin and apex, are viable. */
          nextmove = ORGMOVE;
          enextfnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            searchdist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            searchdist = NORM2(m->xmax - m->xmin, m->ymax - m->ymin, m->zmax - m->zmin);
          }
          fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            dist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            dist = searchdist;
          }
          if (dist < searchdist) {
            nextmove = APEXMOVE;
            searchdist = dist;
          }
        } else {
          /*  Only the face opposite to origin is viable. */
          nextmove = ORGMOVE;
        }
      }
    } else {
      if (oridest < 0) {
        if (oriapex < 0) {
          /*  Two faces, opposite to destination and apex, are viable. */
          nextmove = DESTMOVE;
          enext2fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            searchdist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            searchdist = NORM2(m->xmax - m->xmin, m->ymax - m->ymin, m->zmax - m->zmin);
          }
          fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            ntoppo = oppo(&neightet);
            dist = NORM2(searchpt[0] - ntoppo[0], searchpt[1] - ntoppo[1], searchpt[2] - ntoppo[2]);
          } else {
            dist = searchdist;
          }
          if (dist < searchdist) {
            nextmove = APEXMOVE;
            searchdist = dist;
          }
        } else {
          /*  Only the face opposite to destination is viable. */
          nextmove = DESTMOVE;
        }
      } else {
        if (oriapex < 0) {
          /*  Only the face opposite to apex is viable. */
          nextmove = APEXMOVE;
        } else {
          /*  The point we seek must be on the boundary of or inside this */
          /*    tetrahedron. Check for boundary cases. */
          if (oriorg == 0) {
            /*  Go to the face opposite to origin. */
            enextfnextself(m, searchtet);
            if (oridest == 0) {
              enextself(searchtet); /*  edge apex->oppo */
              if (oriapex == 0) {
                enextself(searchtet); /*  oppo is duplicated with p. */
                loc = ONVERTEX; /*  return ONVERTEX; */
                break;
              }
              loc = ONEDGE; /*  return ONEDGE; */
              break;
            }
            if (oriapex == 0) {
              enext2self(searchtet);
              loc = ONEDGE; /*  return ONEDGE; */
              break;
            }
            loc = ONFACE; /*  return ONFACE; */
            break;
          }
          if (oridest == 0) {
            /*  Go to the face opposite to destination. */
            enext2fnextself(m, searchtet);
            if (oriapex == 0) {
              enextself(searchtet);
              loc = ONEDGE; /*  return ONEDGE; */
              break;
            }
            loc = ONFACE; /*  return ONFACE; */
            break;
          }
          if (oriapex == 0) {
            /*  Go to the face opposite to apex */
            fnextself(m, searchtet);
            loc = ONFACE; /*  return ONFACE; */
            break;
          }
          loc = INTETRAHEDRON; /*  return INTETRAHEDRON; */
          break;
        }
      }
    }

    /*  Move to the selected face. */
    if (nextmove == ORGMOVE) {
      enextfnextself(m, searchtet);
    } else if (nextmove == DESTMOVE) {
      enext2fnextself(m, searchtet);
    } else {
      fnextself(m, searchtet);
    }
    /*  Move to the adjacent tetrahedron (maybe a hull tetrahedron). */
    backtracetet = *searchtet;
    symself(searchtet);
    if (searchtet->tet == m->dummytet) {
      *searchtet = backtracetet;
      loc = OUTSIDE; /*  return OUTSIDE; */
      break;
    }

    if (histtetarray) {
      /*  Check if we have run into a loop. */
      if (infected(m, searchtet)) {
        /*  We have visited this tet. A potential loop is found. */
        loc = OUTSIDE;
        break;
      } else {
        /*  Remember this tet. */
        infect(m, searchtet);
        ierr = ArrayPoolNewIndex(histtetarray, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
        *parytet = *searchtet;
      }
    }

    /*  Retreat the three vertices of the base face. */
    searchtet->ver = 0;
    torg = org(searchtet);
    tdest = dest(searchtet);
    tapex = apex(searchtet);

  } /*  while (true) */

  if (histtetarray) {
    /*  Unmark the visited tets. */
    for(i = 0; i < (int) histtetarray->objects; i++) {
      parytet = (triface *) fastlookup(histtetarray, i);
      uninfect(m, parytet);
    }
    ierr = ArrayPoolRestart(histtetarray);CHKERRQ(ierr);
  }

  if (result) {*result = loc;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshLocateSub"
/*  locatesub()    Find a point in the surface mesh of a facet.                */
/*                                                                             */
/*  Searching begins from the input 'searchsh', it should be a handle on the   */
/*  convex hull of the facet triangulation.                                    */
/*                                                                             */
/*  If 'stopatseg' is nonzero, the search will stop if it tries to walk        */
/*  through a subsegment, and will return OUTSIDE.                             */
/*                                                                             */
/*  On completion, 'searchsh' is a subface that contains 'searchpt'.           */
/*    - Returns ONVERTEX if the point lies on an existing vertex. 'searchsh'   */
/*      is a handle whose origin is the existing vertex.                       */
/*    - Returns ONEDGE if the point lies on a mesh edge.  'searchsh' is a      */
/*      handle whose primary edge is the edge on which the point lies.         */
/*    - Returns ONFACE if the point lies strictly within a subface.            */
/*      'searchsh' is a handle on which the point lies.                        */
/*    - Returns OUTSIDE if the point lies outside the triangulation.           */
/*                                                                             */
/*  WARNING: This routine is designed for convex triangulations, and will not  */
/*  not generally work after the holes and concavities have been carved.       */
/* tetgenmesh::locatesub() */
PetscErrorCode TetGenMeshLocateSub(TetGenMesh *m, point searchpt, face *searchsh, int stopatseg, PetscReal epspp, locateresult *result)
{
  TetGenOpts    *b  = m->b;
  face backtracksh = {PETSC_NULL, 0}, spinsh = {PETSC_NULL, 0}, checkedge = {PETSC_NULL, 0};
  point forg, fdest, fapex;
  PetscReal orgori, destori;
  PetscReal ori, sign;
  int moveleft, i;
  PetscBool      isCoplanar;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (searchsh->sh == m->dummysh) {
    searchsh->shver = 0;
    spivotself(searchsh);
    if (searchsh->sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
  }
  /*  Find the sign to simulate that abovepoint is 'above' the facet. */
  adjustedgering_face(searchsh, CCW);
  forg  = sorg(searchsh);
  fdest = sdest(searchsh);
  fapex = sapex(searchsh);
  ori = TetGenOrient3D(forg, fdest, fapex, m->abovepoint);
  sign = ori > 0.0 ? -1 : 1;

  /*  Orient 'searchsh' so that 'searchpt' is below it (i.e., searchpt has */
  /*    CCW orientation with respect to searchsh in plane).  Such edge */
  /*    should always exist. Save it as (forg, fdest). */
  for(i = 0; i < 3; i++) {
    forg  = sorg(searchsh);
    fdest = sdest(searchsh);
    ori   = TetGenOrient3D(forg, fdest, m->abovepoint, searchpt) * sign;
    if (ori > 0.0) break;
    senextself(searchsh);
  }
  if (i >= 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

  while (1) {
    fapex = sapex(searchsh);
    /*  Check whether the apex is the point we seek. */
    if (fapex[0] == searchpt[0] && fapex[1] == searchpt[1] && fapex[2] == searchpt[2]) {
      senext2self(searchsh);
      if (result) *result = ONVERTEX;
      PetscFunctionReturn(0);
    }
    /*  Does the point lie on the other side of the line defined by the */
    /*    triangle edge opposite the triangle's destination? */
    destori = TetGenOrient3D(forg, fapex, m->abovepoint, searchpt) * sign;
    if (epspp > 0.0) {
      ierr = TetGenMeshIsCoplanar(m, forg, fapex, m->abovepoint, searchpt, destori, epspp, &isCoplanar);CHKERRQ(ierr);
      if (isCoplanar) {
        destori = 0.0;
      }
    }
    /*  Does the point lie on the other side of the line defined by the */
    /*    triangle edge opposite the triangle's origin? */
    orgori = TetGenOrient3D(fapex, fdest, m->abovepoint, searchpt) * sign;
    if (epspp > 0.0) {
      ierr = TetGenMeshIsCoplanar(m, fapex, fdest, m->abovepoint, searchpt, orgori, epspp, &isCoplanar);CHKERRQ(ierr);
      if (isCoplanar) {
        orgori = 0.0;
      }
    }
    if (destori > 0.0) {
      moveleft = 1;
    } else {
      if (orgori > 0.0) {
        moveleft = 0;
      } else {
        /*  The point must be on the boundary of or inside this triangle. */
        if (destori == 0.0) {
          senext2self(searchsh);
          if (result) *result = ONEDGE;
          PetscFunctionReturn(0);
        }
        if (orgori == 0.0) {
          senextself(searchsh);
          if (result) *result = ONEDGE;
          PetscFunctionReturn(0);
        }
        if (result) *result = ONFACE;
        PetscFunctionReturn(0);
      }
    }
    /*  Move to another triangle.  Leave a trace `backtracksh' in case */
    /*    walking off a boundary of the triangulation. */
    if (moveleft) {
      senext2(searchsh, &backtracksh);
      fdest = fapex;
    } else {
      senext(searchsh, &backtracksh);
      forg = fapex;
    }
    /*  Check if we meet a segment. */
    sspivot(m, &backtracksh, &checkedge);
    if (checkedge.sh != m->dummysh) {
      if (stopatseg) {
        /*  The flag indicates we should not cross a segment. Stop. */
        *searchsh = backtracksh;
        if (result) *result = OUTSIDE;
        PetscFunctionReturn(0);
      }
      /*  Try to walk through a segment. We need to find a coplanar subface */
      /*    sharing this segment to get into. */
      spinsh = backtracksh;
      do {
        spivotself(&spinsh);
        if (spinsh.sh == backtracksh.sh) {
          /*  Turn back, no coplanar subface is found. */
          break;
        }
        /*  Are they belong to the same facet. */
        if (shellmark(m, &spinsh) == shellmark(m, &backtracksh)) {
          /*  Find a coplanar subface. Walk into it. */
          *searchsh = spinsh;
          break;
        }
        /*  Are they (nearly) coplanar? */
        ori = TetGenOrient3D(forg, fdest, sapex(&backtracksh), sapex(&spinsh));
        ierr = TetGenMeshIsCoplanar(m, forg, fdest, sapex(&backtracksh), sapex(&spinsh), ori, b->epsilon, &isCoplanar);CHKERRQ(ierr);
        if (isCoplanar) {
          /*  Find a coplanar subface. Walk into it. */
          *searchsh = spinsh;
          break;
        }
      } while (spinsh.sh != backtracksh.sh);
    } else {
      spivot(&backtracksh, searchsh);
    }
    /*  Check for walking right out of the triangulation. */
    if ((searchsh->sh == m->dummysh) || (searchsh->sh == backtracksh.sh)) {
      /*  Go back to the last triangle. */
      *searchsh = backtracksh;
      if (result) *result = OUTSIDE;
      PetscFunctionReturn(0);
    }
    /*  To keep the same orientation wrt abovepoint. */
    if (sorg(searchsh) != forg) sesymself(searchsh);
    if ((sorg(searchsh) != forg) || (sdest(searchsh) != fdest)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
  }
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  geom_cxx ///////////////////////////////////////////////////////////////// */

/*  flip_cxx ///////////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshEnqueueFlipFace"
/*  enqueueflipface(), enqueueflipedge()    Queue a face (or an edge).         */
/*                                                                             */
/*  The face (or edge) may be non-locally Delaunay. It is queued for process-  */
/*  ing in flip() (or flipsub()). The vertices of the face (edge) are stored   */
/*  seperatly to ensure the face (or edge) is still the same one when we save  */
/*  it since other flips will cause this face (or edge) be changed or dead.    */
/* tetgenmesh::enqueueflipface() */
PetscErrorCode TetGenMeshEnqueueFlipFace(TetGenMesh *m, triface *checkface, Queue *flipqueue)
{
  badface       *queface;
  triface        symface = {PETSC_NULL, 0, 0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sym(checkface, &symface);
  if (symface.tet != m->dummytet) {
    ierr = QueuePush(flipqueue, PETSC_NULL, (void **) &queface);CHKERRQ(ierr);
    queface->tt    = *checkface;
    queface->foppo = oppo(&symface);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshEnqueueFlipEdge"
/*  enqueueflipface(), enqueueflipedge()    Queue a face (or an edge).         */
/*                                                                             */
/*  The face (or edge) may be non-locally Delaunay. It is queued for process-  */
/*  ing in flip() (or flipsub()). The vertices of the face (edge) are stored   */
/*  seperatly to ensure the face (or edge) is still the same one when we save  */
/*  it since other flips will cause this face (or edge) be changed or dead.    */
/* tetgenmesh::enqueueflipedge() */
PetscErrorCode TetGenMeshEnqueueFlipEdge(TetGenMesh *m, face *checkedge, Queue *flipqueue)
{
  badface       *queface;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = QueuePush(flipqueue, PETSC_NULL, (void **) &queface);CHKERRQ(ierr);
  queface->ss    = *checkedge;
  queface->forg  = sorg(checkedge);
  queface->fdest = sdest(checkedge);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFlip22Sub"
/*  flip22sub()    Perform a 2-to-2 flip on a subface edge.                    */
/*                                                                             */
/*  The flip edge is given by subface 'flipedge'.  Let it is abc, where ab is  */
/*  the flipping edge.  The other subface is bad,  where a, b, c, d form a     */
/*  convex quadrilateral.  ab is not a subsegment.                             */
/*                                                                             */
/*  A 2-to-2 subface flip is to change two subfaces abc and bad to another     */
/*  two subfaces dca and cdb.  Hence, edge ab has been removed and dc becomes  */
/*  an edge. If a point e is above abc, this flip is equal to rotate abc and   */
/*  bad counterclockwise using right-hand rule with thumb points to e. It is   */
/*  important to know that the edge rings of the flipped subfaces dca and cdb  */
/*  are keeping the same orientation as their original subfaces. So they have  */
/*  the same orientation with respect to the lift point of this facet.         */
/*                                                                             */
/*  During rotating, the face rings of the four edges bc, ca, ad, and de need  */
/*  be re-connected. If the edge is not a subsegment, then its face ring has   */
/*  only two faces, a sbond() will bond them together. If it is a subsegment,  */
/*  one should use sbond1() twice to bond two different handles to the rotat-  */
/*  ing subface, one is predecssor (-casin), another is successor (-casout).   */
/*                                                                             */
/*  If 'flipqueue' is not NULL, it returns four edges bc, ca, ad, de, which    */
/*  may be non-Delaunay.                                                       */
/* tetgenmesh::flip22sub() */
PetscErrorCode TetGenMeshFlip22Sub(TetGenMesh *m, face *flipedge, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  face abc = {PETSC_NULL, 0}, bad = {PETSC_NULL, 0};
  face oldbc = {PETSC_NULL, 0}, oldca = {PETSC_NULL, 0}, oldad = {PETSC_NULL, 0}, olddb = {PETSC_NULL, 0};
  face bccasin = {PETSC_NULL, 0}, bccasout = {PETSC_NULL, 0}, cacasin = {PETSC_NULL, 0}, cacasout = {PETSC_NULL, 0};
  face adcasin = {PETSC_NULL, 0}, adcasout = {PETSC_NULL, 0}, dbcasin = {PETSC_NULL, 0}, dbcasout = {PETSC_NULL, 0};
  face bc = {PETSC_NULL, 0}, ca = {PETSC_NULL, 0}, ad = {PETSC_NULL, 0}, db = {PETSC_NULL, 0};
  face spinsh = {PETSC_NULL, 0};
  point pa, pb, pc, pd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  abc = *flipedge;
  spivot(&abc, &bad);
  if (sorg(&bad) != sdest(&abc)) {
    sesymself(&bad);
  }
  pa = sorg(&abc);
  pb = sdest(&abc);
  pc = sapex(&abc);
  pd = sapex(&bad);

  PetscInfo4(b->in, "    Flip subedge (%d, %d) to (%d, %d).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), pointmark(m, pd));

  /*  Unmark the flipped subfaces (used in mesh refinement). 2009-08-17. */
  sunmarktest(&abc);
  sunmarktest(&bad);

  /*  Save the old configuration outside the quadrilateral. */
  senext(&abc, &oldbc);
  senext2(&abc, &oldca);
  senext(&bad, &oldad);
  senext2(&bad, &olddb);
  /*  Get the outside connection. Becareful if there is a subsegment on the */
  /*    quadrilateral, two casings (casin and casout) are needed to save for */
  /*    keeping the face link. */
  spivot(&oldbc, &bccasout);
  sspivot(m, &oldbc, &bc);
  if (bc.sh != m->dummysh) {
    /*  'bc' is a subsegment. */
    if (bccasout.sh != m->dummysh) {
      if (oldbc.sh != bccasout.sh) {
        /*  'oldbc' is not self-bonded. */
        spinsh = bccasout;
        do {
          bccasin = spinsh;
          spivotself(&spinsh);
        } while (spinsh.sh != oldbc.sh);
      } else {
        bccasout.sh = m->dummysh;
      }
    }
    ssdissolve(m, &oldbc);
  }
  spivot(&oldca, &cacasout);
  sspivot(m, &oldca, &ca);
  if (ca.sh != m->dummysh) {
    /*  'ca' is a subsegment. */
    if (cacasout.sh != m->dummysh) {
      if (oldca.sh != cacasout.sh) {
        /*  'oldca' is not self-bonded. */
        spinsh = cacasout;
        do {
          cacasin = spinsh;
          spivotself(&spinsh);
        } while (spinsh.sh != oldca.sh);
      } else {
        cacasout.sh = m->dummysh;
      }
    }
    ssdissolve(m, &oldca);
  }
  spivot(&oldad, &adcasout);
  sspivot(m, &oldad, &ad);
  if (ad.sh != m->dummysh) {
    /*  'ad' is a subsegment. */
    if (adcasout.sh != m->dummysh) {
      if (oldad.sh != adcasout.sh) {
        /*  'adcasout' is not self-bonded. */
        spinsh = adcasout;
        do {
          adcasin = spinsh;
          spivotself(&spinsh);
        } while (spinsh.sh != oldad.sh);
      } else {
        adcasout.sh = m->dummysh;
      }
    }
    ssdissolve(m, &oldad);
  }
  spivot(&olddb, &dbcasout);
  sspivot(m, &olddb, &db);
  if (db.sh != m->dummysh) {
    /*  'db' is a subsegment. */
    if (dbcasout.sh != m->dummysh) {
      if (olddb.sh != dbcasout.sh) {
        /*  'dbcasout' is not self-bonded. */
        spinsh = dbcasout;
        do {
          dbcasin = spinsh;
          spivotself(&spinsh);
        } while (spinsh.sh != olddb.sh);
      } else {
        dbcasout.sh = m->dummysh;
      }
    }
    ssdissolve(m, &olddb);
  }

  /*  Rotate abc and bad one-quarter turn counterclockwise. */
  if (ca.sh != m->dummysh) {
    if (cacasout.sh != m->dummysh) {
      sbond1(&cacasin, &oldbc);
      sbond1(&oldbc, &cacasout);
    } else {
      /*  Bond 'oldbc' to itself. */
      sdissolve(m, &oldbc); /*  sbond(oldbc, oldbc); */
      /*  Make sure that dummysh always correctly bonded. */
      m->dummysh[0] = sencode(&oldbc);
    }
    ssbond(m, &oldbc, &ca);
  } else {
    sbond(&oldbc, &cacasout);
  }
  if (ad.sh != m->dummysh) {
    if (adcasout.sh != m->dummysh) {
      sbond1(&adcasin, &oldca);
      sbond1(&oldca, &adcasout);
    } else {
      /*  Bond 'oldca' to itself. */
      sdissolve(m, &oldca); /*  sbond(oldca, oldca); */
      /*  Make sure that dummysh always correctly bonded. */
      m->dummysh[0] = sencode(&oldca);
    }
    ssbond(m, &oldca, &ad);
  } else {
    sbond(&oldca, &adcasout);
  }
  if (db.sh != m->dummysh) {
    if (dbcasout.sh != m->dummysh) {
      sbond1(&dbcasin, &oldad);
      sbond1(&oldad, &dbcasout);
    } else {
      /*  Bond 'oldad' to itself. */
      sdissolve(m, &oldad); /*  sbond(oldad, oldad); */
      /*  Make sure that dummysh always correctly bonded. */
      m->dummysh[0] = sencode(&oldad);
    }
    ssbond(m, &oldad, &db);
  } else {
    sbond(&oldad, &dbcasout);
  }
  if (bc.sh != m->dummysh) {
    if (bccasout.sh != m->dummysh) {
      sbond1(&bccasin, &olddb);
      sbond1(&olddb, &bccasout);
    } else {
      /*  Bond 'olddb' to itself. */
      sdissolve(m, &olddb); /*  sbond(olddb, olddb); */
      /*  Make sure that dummysh always correctly bonded. */
      m->dummysh[0] = sencode(&olddb);
    }
    ssbond(m, &olddb, &bc);
  } else {
    sbond(&olddb, &bccasout);
  }

  /*  New vertex assignments for the rotated subfaces. */
  setsorg(&abc, pd);  /*  Update abc to dca. */
  setsdest(&abc, pc);
  setsapex(&abc, pa);
  setsorg(&bad, pc);  /*  Update bad to cdb. */
  setsdest(&bad, pd);
  setsapex(&bad, pb);

  /*  Update the point-to-subface map. */
  /*  Comemnt: After the flip, abc becomes dca, bad becodes cdb.  */
  setpoint2sh(m, pa, sencode(&abc)); /*  dca */
  setpoint2sh(m, pb, sencode(&bad)); /*  cdb */
  setpoint2sh(m, pc, sencode(&bad));
  setpoint2sh(m, pd, sencode(&bad));

  if (flipqueue) {
    ierr = TetGenMeshEnqueueFlipEdge(m, &bccasout, flipqueue);CHKERRQ(ierr);
    ierr = TetGenMeshEnqueueFlipEdge(m, &cacasout, flipqueue);CHKERRQ(ierr);
    ierr = TetGenMeshEnqueueFlipEdge(m, &adcasout, flipqueue);CHKERRQ(ierr);
    ierr = TetGenMeshEnqueueFlipEdge(m, &dbcasout, flipqueue);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFlip23"
/*  flip23()    Perform a 2-to-3 flip.                                         */
/*                                                                             */
/*  On input, 'flipface' represents the face will be flipped.  Let it is abc,  */
/*  the two tetrahedra sharing abc are abcd, bace. abc is not a subface.       */
/*                                                                             */
/*  A 2-to-3 flip is to change two tetrahedra abcd, bace to three tetrahedra   */
/*  edab, edbc, and edca.  As a result, face abc has been removed and three    */
/*  new faces eda, edb and edc have been created.                              */
/*                                                                             */
/*  On completion, 'flipface' returns edab.  If 'flipqueue' is not NULL, all   */
/*  possibly non-Delaunay faces are added into it.                             */
/* tetgenmesh::flip23() */
PetscErrorCode TetGenMeshFlip23(TetGenMesh *m, triface *flipface, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  triface abcd = {PETSC_NULL, 0, 0}, bace = {PETSC_NULL, 0, 0};                                     /*  Old configuration. */
  triface oldabd = {PETSC_NULL, 0, 0}, oldbcd = {PETSC_NULL, 0, 0}, oldcad = {PETSC_NULL, 0, 0};
  triface abdcasing = {PETSC_NULL, 0, 0}, bcdcasing = {PETSC_NULL, 0, 0}, cadcasing = {PETSC_NULL, 0, 0};
  triface oldbae = {PETSC_NULL, 0, 0}, oldcbe = {PETSC_NULL, 0, 0}, oldace = {PETSC_NULL, 0, 0};
  triface baecasing = {PETSC_NULL, 0, 0}, cbecasing = {PETSC_NULL, 0, 0}, acecasing = {PETSC_NULL, 0, 0};
  triface worktet = {PETSC_NULL, 0, 0};
  face abdsh = {PETSC_NULL, 0}, bcdsh = {PETSC_NULL, 0}, cadsh = {PETSC_NULL, 0};                   /*  The six subfaces on the CH. */
  face baesh = {PETSC_NULL, 0}, cbesh = {PETSC_NULL, 0}, acesh = {PETSC_NULL, 0};
  face abseg = {PETSC_NULL, 0}, bcseg = {PETSC_NULL, 0}, caseg = {PETSC_NULL, 0};                   /*  The nine segs on the CH. */
  face adseg = {PETSC_NULL, 0}, bdseg = {PETSC_NULL, 0}, cdseg = {PETSC_NULL, 0};
  face aeseg = {PETSC_NULL, 0}, beseg = {PETSC_NULL, 0}, ceseg = {PETSC_NULL, 0};
  triface edab = {PETSC_NULL, 0, 0}, edbc = {PETSC_NULL, 0, 0}, edca = {PETSC_NULL, 0, 0};          /*  New configuration. */
  point pa, pb, pc, pd, pe;
  PetscReal attrib, volume;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  abcd = *flipface;
  adjustedgering_triface(&abcd, CCW); /*  abcd represents edge ab. */
  pa = org(&abcd);
  pb = dest(&abcd);
  pc = apex(&abcd);
  pd = oppo(&abcd);
  /*  sym(abcd, bace); */
  /*  findedge(&bace, dest(abcd), org(abcd));  bace represents edge ba. */
  sym(&abcd, &bace);
  bace.ver = 0; /*  CCW. */
  for(i = 0; (i < 3) && (org(&bace) != pb); i++) {
    enextself(&bace);
  }
  pe = oppo(&bace);

  PetscInfo5(b->in, "    Do T23 on face (%d, %d, %d) %d, %d.\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), pointmark(m, pd), pointmark(m, pe));
  m->flip23s++;

  /*  Storing the old configuration outside the convex hull. */
  fnext(m, &abcd, &oldabd);
  enextfnext(m, &abcd, &oldbcd);
  enext2fnext(m, &abcd, &oldcad);
  fnext(m, &bace, &oldbae);
  enext2fnext(m, &bace, &oldcbe);
  enextfnext(m, &bace, &oldace);
  sym(&oldabd, &abdcasing);
  sym(&oldbcd, &bcdcasing);
  sym(&oldcad, &cadcasing);
  sym(&oldbae, &baecasing);
  sym(&oldcbe, &cbecasing);
  sym(&oldace, &acecasing);
  if (m->checksubfaces) {
    tspivot(m, &oldabd, &abdsh);
    tspivot(m, &oldbcd, &bcdsh);
    tspivot(m, &oldcad, &cadsh);
    tspivot(m, &oldbae, &baesh);
    tspivot(m, &oldcbe, &cbesh);
    tspivot(m, &oldace, &acesh);
  }
  if (m->checksubsegs) {
    tsspivot1(m, &abcd, &abseg);
    enext(&abcd, &worktet);
    tsspivot1(m, &worktet, &bcseg);
    enext2(&abcd, &worktet);
    tsspivot1(m, &worktet, &caseg);
    enext2(&oldabd, &worktet);
    tsspivot1(m, &worktet, &adseg);
    enext2(&oldbcd, &worktet);
    tsspivot1(m, &worktet, &bdseg);
    enext2(&oldcad, &worktet);
    tsspivot1(m, &worktet, &cdseg);
    enext(&oldbae, &worktet);
    tsspivot1(m, &worktet, &aeseg);
    enext(&oldcbe, &worktet);
    tsspivot1(m, &worktet, &beseg);
    enext(&oldace, &worktet);
    tsspivot1(m, &worktet, &ceseg);
  }

  /*  Creating the new configuration inside the convex hull. */
  edab.tet = abcd.tet; /*  Update abcd to be edab. */
  setorg (&edab, pe);
  setdest(&edab, pd);
  setapex(&edab, pa);
  setoppo(&edab, pb);
  edbc.tet = bace.tet; /*  Update bace to be edbc. */
  setorg (&edbc, pe);
  setdest(&edbc, pd);
  setapex(&edbc, pb);
  setoppo(&edbc, pc);
  ierr = TetGenMeshMakeTetrahedron(m, &edca);CHKERRQ(ierr); /*  Create edca. */
  setorg (&edca, pe);
  setdest(&edca, pd);
  setapex(&edca, pc);
  setoppo(&edca, pa);
  /*  Set the element attributes of the new tetrahedron 'edca'. */
  for(i = 0; i < in->numberoftetrahedronattributes; i++) {
    attrib = elemattribute(m, abcd.tet, i);
    setelemattribute(m, edca.tet, i, attrib);
  }
  /*  Set the volume constraint of the new tetrahedron 'edca' if the -ra */
  /*    switches are not used together. In -ra case, the various volume */
  /*    constraints can be spreaded very far. */
  if (b->varvolume && !b->refine) {
    volume = volumebound(m, abcd.tet);
    setvolumebound(m, edca.tet, volume);
  }

  /*  Clear old bonds in edab(was abcd) and edbc(was bace). */
  for(i = 0; i < 4; i ++) {
    edab.tet[i] = (tetrahedron) m->dummytet;
  }
  for(i = 0; i < 4; i ++) {
    edbc.tet[i] = (tetrahedron) m->dummytet;
  }
  /*  Bond the faces inside the convex hull. */
  edab.loc = 0;
  edca.loc = 1;
  bond(m, &edab, &edca);
  edab.loc = 1;
  edbc.loc = 0;
  bond(m, &edab, &edbc);
  edbc.loc = 1;
  edca.loc = 0;
  bond(m, &edbc, &edca);
  /*  Bond the faces on the convex hull. */
  edab.loc = 2;
  bond(m, &edab, &abdcasing);
  edab.loc = 3;
  bond(m, &edab, &baecasing);
  edbc.loc = 2;
  bond(m, &edbc, &bcdcasing);
  edbc.loc = 3;
  bond(m, &edbc, &cbecasing);
  edca.loc = 2;
  bond(m, &edca, &cadcasing);
  edca.loc = 3;
  bond(m, &edca, &acecasing);
  /*  There may exist subfaces that need to be bonded to new configuarton. */
  if (m->checksubfaces) {
    /*  Clear old flags in edab(was abcd) and edbc(was bace). */
    for(i = 0; i < 4; i ++) {
      edab.loc = i;
      tsdissolve(m, &edab);
      edbc.loc = i;
      tsdissolve(m, &edbc);
    }
    if (abdsh.sh != m->dummysh) {
      edab.loc = 2;
      tsbond(m, &edab, &abdsh);
    }
    if (baesh.sh != m->dummysh) {
      edab.loc = 3;
      tsbond(m, &edab, &baesh);
    }
    if (bcdsh.sh != m->dummysh) {
      edbc.loc = 2;
      tsbond(m, &edbc, &bcdsh);
    }
    if (cbesh.sh != m->dummysh) {
      edbc.loc = 3;
      tsbond(m, &edbc, &cbesh);
    }
    if (cadsh.sh != m->dummysh) {
      edca.loc = 2;
      tsbond(m, &edca, &cadsh);
    }
    if (acesh.sh != m->dummysh) {
      edca.loc = 3;
      tsbond(m, &edca, &acesh);
    }
  }
  if (m->checksubsegs) {
    for(i = 0; i < 6; i++) {
      edab.loc = edge2locver[i][0];
      edab.ver = edge2locver[i][1];
      tssdissolve1(m, &edab);
    }
    for(i = 0; i < 6; i++) {
      edbc.loc = edge2locver[i][0];
      edbc.ver = edge2locver[i][1];
      tssdissolve1(m, &edbc);
    }
    edab.loc = edab.ver = 0;
    edbc.loc = edab.ver = 0;
    edca.loc = edab.ver = 0;
    /*  Operate in tet edab (5 edges). */
    enext(&edab, &worktet);
    tssbond1(m, &worktet, &adseg);
    enext2(&edab, &worktet);
    tssbond1(m, &worktet, &aeseg);
    fnext(m, &edab, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &bdseg);
    enextself(&worktet);
    tssbond1(m, &worktet, &beseg);
    enextfnext(m, &edab, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &abseg);
    /*  Operate in tet edbc (5 edges) */
    enext(&edbc, &worktet);
    tssbond1(m, &worktet, &bdseg);
    enext2(&edbc, &worktet);
    tssbond1(m, &worktet, &beseg);
    fnext(m, &edbc, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &cdseg);
    enextself(&worktet);
    tssbond1(m, &worktet, &ceseg);
    enextfnext(m, &edbc, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &bcseg);
    /*  Operate in tet edca (5 edges) */
    enext(&edca, &worktet);
    tssbond1(m, &worktet, &cdseg);
    enext2(&edca, &worktet);
    tssbond1(m, &worktet, &ceseg);
    fnext(m, &edca, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &adseg);
    enextself(&worktet);
    tssbond1(m, &worktet, &aeseg);
    enextfnext(m, &edca, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &caseg);
  }

  edab.loc = 0;
  edbc.loc = 0;
  edca.loc = 0;
  if (b->verbose > 3) {
    PetscInfo(b->in, "    Updating edab "); ierr = TetGenMeshPrintTet(m, &edab, PETSC_FALSE);CHKERRQ(ierr);
    PetscInfo(b->in, "    Updating edbc "); ierr = TetGenMeshPrintTet(m, &edbc, PETSC_FALSE);CHKERRQ(ierr);
    PetscInfo(b->in, "    Creating edca "); ierr = TetGenMeshPrintTet(m, &edca, PETSC_FALSE);CHKERRQ(ierr);
  }

  /*  Update point-to-tet map. */
  setpoint2tet(m, pa, encode(&edab));
  setpoint2tet(m, pb, encode(&edab));
  setpoint2tet(m, pc, encode(&edbc));
  setpoint2tet(m, pd, encode(&edab));
  setpoint2tet(m, pe, encode(&edab));

  if (flipqueue) {
    enextfnext(m, &edab, &abdcasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &abdcasing, flipqueue);CHKERRQ(ierr);
    enext2fnext(m, &edab, &baecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &baecasing, flipqueue);CHKERRQ(ierr);
    enextfnext(m, &edbc, &bcdcasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &bcdcasing, flipqueue);CHKERRQ(ierr);
    enext2fnext(m, &edbc, &cbecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &cbecasing, flipqueue);CHKERRQ(ierr);
    enextfnext(m, &edca, &cadcasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &cadcasing, flipqueue);CHKERRQ(ierr);
    enext2fnext(m, &edca, &acecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &acecasing, flipqueue);CHKERRQ(ierr);
  }

  /*  Save a live handle in 'recenttet'. */
  m->recenttet = edbc;
  /*  Set the return handle be edab. */
  *flipface = edab;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFlip32"
/*  flip32()    Perform a 3-to-2 flip.                                         */
/*                                                                             */
/*  On input, 'flipface' represents the face will be flipped.  Let it is eda,  */
/*  where edge ed is locally non-convex. Three tetrahedra sharing ed are edab, */
/*  edbc, and edca.  ed is not a subsegment.                                   */
/*                                                                             */
/*  A 3-to-2 flip is to change the three tetrahedra edab, edbc, and edca into  */
/*  another two tetrahedra abcd and bace.  As a result, the edge ed has been   */
/*  removed and the face abc has been created.                                 */
/*                                                                             */
/*  On completion, 'flipface' returns abcd.  If 'flipqueue' is not NULL, all   */
/*  possibly non-Delaunay faces are added into it.                             */
/* tetgenmesh::flip32() */
PetscErrorCode TetGenMeshFlip32(TetGenMesh *m, triface *flipface, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  triface edab = {PETSC_NULL, 0, 0}, edbc = {PETSC_NULL, 0, 0}, edca = {PETSC_NULL, 0, 0};                /*  Old configuration. */
  triface oldabd = {PETSC_NULL, 0, 0}, oldbcd = {PETSC_NULL, 0, 0}, oldcad = {PETSC_NULL, 0, 0};
  triface abdcasing = {PETSC_NULL, 0, 0}, bcdcasing = {PETSC_NULL, 0, 0}, cadcasing = {PETSC_NULL, 0, 0};
  triface oldbae = {PETSC_NULL, 0, 0}, oldcbe = {PETSC_NULL, 0, 0}, oldace = {PETSC_NULL, 0, 0};
  triface baecasing = {PETSC_NULL, 0, 0}, cbecasing = {PETSC_NULL, 0, 0}, acecasing = {PETSC_NULL, 0, 0};
  triface worktet = {PETSC_NULL, 0, 0};
  face abdsh = {PETSC_NULL, 0}, bcdsh = {PETSC_NULL, 0}, cadsh = {PETSC_NULL, 0};
  face baesh = {PETSC_NULL, 0}, cbesh = {PETSC_NULL, 0}, acesh = {PETSC_NULL, 0};
  face abseg = {PETSC_NULL, 0}, bcseg = {PETSC_NULL, 0}, caseg = {PETSC_NULL, 0};                         /*  The nine segs on the CH. */
  face adseg = {PETSC_NULL, 0}, bdseg = {PETSC_NULL, 0}, cdseg = {PETSC_NULL, 0};
  face aeseg = {PETSC_NULL, 0}, beseg = {PETSC_NULL, 0}, ceseg = {PETSC_NULL, 0};
  triface abcd = {PETSC_NULL, 0, 0}, bace = {PETSC_NULL, 0, 0};                                           /*  New configuration. */
  point pa, pb, pc, pd, pe;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  edab = *flipface;
  adjustedgering_triface(&edab, CCW);
  pa = apex(&edab);
  pb = oppo(&edab);
  pd = dest(&edab);
  pe = org(&edab);
  fnext(m, &edab, &edbc);
  symself(&edbc);
  edbc.ver = 0;
  for (i = 0; (i < 3) && (org(&edbc) != pe); i++) {
    enextself(&edbc);
  }
  pc = oppo(&edbc);
  fnext(m, &edbc, &edca);
  symself(&edca);
  edca.ver = 0;
  for (i = 0; (i < 3) && (org(&edca) != pe); i++) {
    enextself(&edca);
  }

  PetscInfo5(b->in, "    Do T32 on edge (%d, %d) %d, %d, %d.\n", pointmark(m, pe), pointmark(m, pd), pointmark(m, pa), pointmark(m, pb), pointmark(m, pc));
  m->flip32s++;

  /*  Storing the old configuration outside the convex hull. */
  enextfnext(m, &edab, &oldabd);
  enext2fnext(m, &edab, &oldbae);
  enextfnext(m, &edbc, &oldbcd);
  enext2fnext(m, &edbc, &oldcbe);
  enextfnext(m, &edca, &oldcad);
  enext2fnext(m, &edca, &oldace);
  sym(&oldabd, &abdcasing);
  sym(&oldbcd, &bcdcasing);
  sym(&oldcad, &cadcasing);
  sym(&oldbae, &baecasing);
  sym(&oldcbe, &cbecasing);
  sym(&oldace, &acecasing);
  if (m->checksubfaces) {
    tspivot(m, &oldabd, &abdsh);
    tspivot(m, &oldbcd, &bcdsh);
    tspivot(m, &oldcad, &cadsh);
    tspivot(m, &oldbae, &baesh);
    tspivot(m, &oldcbe, &cbesh);
    tspivot(m, &oldace, &acesh);
  }
  if (m->checksubsegs) {
    enext(&edab, &worktet);
    tsspivot1(m, &worktet, &adseg);
    enext2(&edab, &worktet);
    tsspivot1(m, &worktet, &aeseg);
    enext(&edbc, &worktet);
    tsspivot1(m, &worktet, &bdseg);
    enext2(&edbc, &worktet);
    tsspivot1(m, &worktet, &beseg);
    enext(&edca, &worktet);
    tsspivot1(m, &worktet, &cdseg);
    enext2(&edca, &worktet);
    tsspivot1(m, &worktet, &ceseg);
    enextfnext(m, &edab, &worktet);
    enextself(&worktet);
    tsspivot1(m, &worktet, &abseg);
    enextfnext(m, &edbc, &worktet);
    enextself(&worktet);
    tsspivot1(m, &worktet, &bcseg);
    enextfnext(m, &edca, &worktet);
    enextself(&worktet);
    tsspivot1(m, &worktet, &caseg);
  }

  /*  Creating the new configuration inside the convex hull. */
  abcd.tet = edab.tet; /*  Update edab to be abcd. */
  setorg (&abcd, pa);
  setdest(&abcd, pb);
  setapex(&abcd, pc);
  setoppo(&abcd, pd);
  bace.tet = edbc.tet; /*  Update edbc to be bace. */
  setorg (&bace, pb);
  setdest(&bace, pa);
  setapex(&bace, pc);
  setoppo(&bace, pe);
  /*  Dealloc a redundant tetrahedron (edca). */
  ierr = TetGenMeshTetrahedronDealloc(m, edca.tet);CHKERRQ(ierr);

  /*  Clear the old bonds in abcd (was edab) and bace (was edbc). */
  for(i = 0; i < 4; i ++) {
    abcd.tet[i] = (tetrahedron) m->dummytet;
  }
  for(i = 0; i < 4; i ++) {
    bace.tet[i] = (tetrahedron) m->dummytet;
  }
  /*  Bond the inside face of the convex hull. */
  abcd.loc = 0;
  bace.loc = 0;
  bond(m, &abcd, &bace);
  /*  Bond the outside faces of the convex hull. */
  abcd.loc = 1;
  bond(m, &abcd, &abdcasing);
  abcd.loc = 2;
  bond(m, &abcd, &bcdcasing);
  abcd.loc = 3;
  bond(m, &abcd, &cadcasing);
  bace.loc = 1;
  bond(m, &bace, &baecasing);
  bace.loc = 3;
  bond(m, &bace, &cbecasing);
  bace.loc = 2;
  bond(m, &bace, &acecasing);
  if (m->checksubfaces) {
    /*  Clear old bonds in abcd(was edab) and bace(was edbc). */
    for(i = 0; i < 4; i ++) {
      abcd.loc = i;
      tsdissolve(m, &abcd);
    }
    for(i = 0; i < 4; i ++) {
      bace.loc = i;
      tsdissolve(m, &bace);
    }
    if (abdsh.sh != m->dummysh) {
      abcd.loc = 1;
      tsbond(m, &abcd, &abdsh);
    }
    if (bcdsh.sh != m->dummysh) {
      abcd.loc = 2;
      tsbond(m, &abcd, &bcdsh);
    }
    if (cadsh.sh != m->dummysh) {
      abcd.loc = 3;
      tsbond(m, &abcd, &cadsh);
    }
    if (baesh.sh != m->dummysh) {
      bace.loc = 1;
      tsbond(m, &bace, &baesh);
    }
    if (cbesh.sh != m->dummysh) {
      bace.loc = 3;
      tsbond(m, &bace, &cbesh);
    }
    if (acesh.sh != m->dummysh) {
      bace.loc = 2;
      tsbond(m, &bace, &acesh);
    }
  }
  if (m->checksubsegs) {
    for (i = 0; i < 6; i++) {
      abcd.loc = edge2locver[i][0];
      abcd.ver = edge2locver[i][1];
      tssdissolve1(m, &abcd);
    }
    for (i = 0; i < 6; i++) {
      bace.loc = edge2locver[i][0];
      bace.ver = edge2locver[i][1];
      tssdissolve1(m, &bace);
    }
    abcd.loc = abcd.ver = 0;
    bace.loc = bace.ver = 0;
    tssbond1(m, &abcd, &abseg);     /*  1 */
    enext(&abcd, &worktet);
    tssbond1(m, &worktet, &bcseg);  /*  2 */
    enext2(&abcd, &worktet);
    tssbond1(m, &worktet, &caseg);  /*  3 */
    fnext(m, &abcd, &worktet);
    enext2self(&worktet);
    tssbond1(m, &worktet, &adseg);  /*  4 */
    enextfnext(m, &abcd, &worktet);
    enext2self(&worktet);
    tssbond1(m, &worktet, &bdseg);  /*  5 */
    enext2fnext(m, &abcd, &worktet);
    enext2self(&worktet);
    tssbond1(m, &worktet, &cdseg);  /*  6 */
    tssbond1(m, &bace, &abseg);
    enext2(&bace, &worktet);
    tssbond1(m, &worktet, &bcseg);
    enext(&bace, &worktet);
    tssbond1(m, &worktet, &caseg);
    fnext(m, &bace, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &aeseg);  /*  7 */
    enext2fnext(m, &bace, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &beseg);  /*  8 */
    enextfnext(m, &bace, &worktet);
    enextself(&worktet);
    tssbond1(m, &worktet, &ceseg);  /*  9 */
  }

  abcd.loc = 0;
  bace.loc = 0;
  if (b->verbose > 3) {
    PetscInfo(b->in, "    Updating abcd "); ierr = TetGenMeshPrintTet(m, &abcd, PETSC_FALSE);CHKERRQ(ierr);
    PetscInfo(b->in, "    Updating bace "); ierr = TetGenMeshPrintTet(m, &bace, PETSC_FALSE);CHKERRQ(ierr);
    PetscInfo(b->in, "    Deleting edca "); /*  ierr = TetGenMeshPrintTet(m, &edca, PETSC_FALSE);CHKERRQ(ierr); */
  }

  /*  Update point-to-tet map. */
  setpoint2tet(m, pa, encode(&abcd));
  setpoint2tet(m, pb, encode(&abcd));
  setpoint2tet(m, pc, encode(&abcd));
  setpoint2tet(m, pd, encode(&abcd));
  setpoint2tet(m, pe, encode(&bace));

  if (flipqueue) {
    fnext(m, &abcd, &abdcasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &abdcasing, flipqueue);CHKERRQ(ierr);
    fnext(m, &bace, &baecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &baecasing, flipqueue);CHKERRQ(ierr);
    enextfnext(m, &abcd, &bcdcasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &bcdcasing, flipqueue);CHKERRQ(ierr);
    enextfnext(m, &bace, &cbecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &cbecasing, flipqueue);CHKERRQ(ierr);
    enext2fnext(m, &abcd, &cadcasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &cadcasing, flipqueue);CHKERRQ(ierr);
    enext2fnext(m, &bace, &acecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &acecasing, flipqueue);CHKERRQ(ierr);
  }

  /*  Save a live handle in 'recenttet'. */
  m->recenttet = abcd;
  /*  Set the return handle be abcd. */
  *flipface = abcd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFlip22"
/*  flip22()    Perform a 2-to-2 (or 4-to-4) flip.                             */
/*                                                                             */
/*  On input, 'flipface' represents the face will be flipped.  Let it is abe,  */
/*  ab is the flipable edge, the two tetrahedra sharing abe are abce and bade, */
/*  hence a, b, c and d are coplanar. If abc, bad are interior faces, the two  */
/*  tetrahedra opposite to e are bacf and abdf.  ab is not a subsegment.       */
/*                                                                             */
/*  A 2-to-2 flip is to change two tetrahedra abce and bade into another two   */
/*  tetrahedra dcae and cdbe. If bacf and abdf exist, they're changed to cdaf  */
/*  and dcbf, thus a 4-to-4 flip.  As a result, two or four tetrahedra have    */
/*  rotated counterclockwise (using right-hand rule with thumb points to e):   */
/*  abce->dcae, bade->cdbe, and bacf->cdaf, abdf->dcbf.                        */
/*                                                                             */
/*  If abc and bad are subfaces, a 2-to-2 flip is performed simultaneously by  */
/*  calling routine flip22sub(), hence abc->dca, bad->cdb.  The edge rings of  */
/*  the flipped subfaces dca and cdb have the same orientation as abc and bad. */
/*  Hence, they have the same orientation as other subfaces of the facet with  */
/*  respect to the lift point of this facet.                                   */
/*                                                                             */
/*  On completion, 'flipface' holds edge dc of tetrahedron dcae. 'flipqueue'   */
/*  contains all possibly non-Delaunay faces if it is not NULL.                */
/* tetgenmesh::flip22() */
PetscErrorCode TetGenMeshFlip22(TetGenMesh *m, triface *flipface, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  triface abce = {PETSC_NULL, 0, 0}, bade = {PETSC_NULL, 0, 0};
  triface oldbce = {PETSC_NULL, 0, 0}, oldcae = {PETSC_NULL, 0, 0}, oldade = {PETSC_NULL, 0, 0}, olddbe = {PETSC_NULL, 0, 0};
  triface bcecasing = {PETSC_NULL, 0, 0}, caecasing = {PETSC_NULL, 0, 0}, adecasing = {PETSC_NULL, 0, 0}, dbecasing = {PETSC_NULL, 0, 0};
  face bcesh = {PETSC_NULL, 0}, caesh = {PETSC_NULL, 0}, adesh = {PETSC_NULL, 0}, dbesh = {PETSC_NULL, 0};
  triface bacf = {PETSC_NULL, 0, 0}, abdf = {PETSC_NULL, 0, 0};
  triface oldacf = {PETSC_NULL, 0, 0}, oldcbf = {PETSC_NULL, 0, 0}, oldbdf = {PETSC_NULL, 0, 0}, olddaf = {PETSC_NULL, 0, 0};
  triface acfcasing = {PETSC_NULL, 0, 0}, cbfcasing = {PETSC_NULL, 0, 0}, bdfcasing = {PETSC_NULL, 0, 0}, dafcasing = {PETSC_NULL, 0, 0};
  triface worktet = {PETSC_NULL, 0, 0};
  face acfsh = {PETSC_NULL, 0}, cbfsh = {PETSC_NULL, 0}, bdfsh = {PETSC_NULL, 0}, dafsh = {PETSC_NULL, 0};
  face abc = {PETSC_NULL, 0}, bad = {PETSC_NULL, 0};
  face adseg = {PETSC_NULL, 0}, dbseg = {PETSC_NULL, 0}, bcseg = {PETSC_NULL, 0}, caseg = {PETSC_NULL, 0};  /*  Coplanar segs. */
  face aeseg = {PETSC_NULL, 0}, deseg = {PETSC_NULL, 0}, beseg = {PETSC_NULL, 0}, ceseg = {PETSC_NULL, 0};  /*  Above segs. */
  face afseg = {PETSC_NULL, 0}, dfseg = {PETSC_NULL, 0}, bfseg = {PETSC_NULL, 0}, cfseg = {PETSC_NULL, 0};  /*  Below segs. */
  point pa, pb, pc, pd, pe, pf = PETSC_NULL;
  int mirrorflag, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  adjustedgering_triface(flipface, CCW); /*  'flipface' is bae. */
  fnext(m, flipface, &abce);
  esymself(&abce);
  adjustedgering_triface(flipface, CW); /*  'flipface' is abe. */
  fnext(m, flipface, &bade);
#ifdef PETSC_USE_DEBUG
  if (bade.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
  esymself(&bade);
  pa = org(&abce);
  pb = dest(&abce);
  pc = apex(&abce);
  pd = apex(&bade);
  pe = oppo(&bade);
#ifdef PETSC_USE_DEBUG
  if (oppo(&abce) != pe) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
  sym(&abce, &bacf);
  mirrorflag = bacf.tet != m->dummytet ? PETSC_TRUE : PETSC_FALSE;
  if (mirrorflag) {
    /*  findedge(&bacf, pb, pa); */
    bacf.ver = 0;
    for(i = 0; (i < 3) && (org(&bacf) != pb); i++) {
      enextself(&bacf);
    }
    sym(&bade, &abdf);
#ifdef PETSC_USE_DEBUG
    if (abdf.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
    /*  findedge(&abdf, pa, pb); */
    abdf.ver = 0;
    for (i = 0; (i < 3) && (org(&abdf) != pa); i++) {
      enextself(&abdf);
    }
    pf = oppo(&bacf);
#ifdef PETSC_USE_DEBUG
    if (oppo(&abdf) != pf) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
  }

  PetscInfo5(b->in, "    Flip edge (%d, %d) to (%d, %d) %s.\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), pointmark(m, pd), mirrorflag ? "T44" : "T22");
  mirrorflag ? m->flip44s++ : m->flip22s++;

  /*  Save the old configuration at the convex hull. */
  enextfnext(m, &abce, &oldbce);
  enext2fnext(m, &abce, &oldcae);
  enextfnext(m, &bade, &oldade);
  enext2fnext(m, &bade, &olddbe);
  sym(&oldbce, &bcecasing);
  sym(&oldcae, &caecasing);
  sym(&oldade, &adecasing);
  sym(&olddbe, &dbecasing);
  if (m->checksubfaces) {
    tspivot(m, &oldbce, &bcesh);
    tspivot(m, &oldcae, &caesh);
    tspivot(m, &oldade, &adesh);
    tspivot(m, &olddbe, &dbesh);
    tspivot(m, &abce, &abc);
    tspivot(m, &bade, &bad);
  }
  if (m->checksubsegs) {
    /*  Coplanar segs: a->d->b->c. */
    enext(&bade, &worktet);
    tsspivot1(m, &worktet, &adseg);
    enext2(&bade, &worktet);
    tsspivot1(m, &worktet, &dbseg);
    enext(&abce, &worktet);
    tsspivot1(m, &worktet, &bcseg);
    enext2(&abce, &worktet);
    tsspivot1(m, &worktet, &caseg);
    /*  Above segs: a->e, d->e, b->e, c->e. */
    fnext(m, &bade, &worktet);
    enextself(&worktet);
    tsspivot1(m, &worktet, &aeseg);
    enextfnext(m, &bade, &worktet);
    enextself(&worktet);
    tsspivot1(m, &worktet, &deseg);
    enext2fnext(m, &bade, &worktet);
    enextself(&worktet);
    tsspivot1(m, &worktet, &beseg);
    enextfnext(m, &abce, &worktet);
    enextself(&worktet);
    tsspivot1(m, &worktet, &ceseg);
  }
  if (mirrorflag) {
    enextfnext(m, &bacf, &oldacf);
    enext2fnext(m, &bacf, &oldcbf);
    enextfnext(m, &abdf, &oldbdf);
    enext2fnext(m, &abdf, &olddaf);
    sym(&oldacf, &acfcasing);
    sym(&oldcbf, &cbfcasing);
    sym(&oldbdf, &bdfcasing);
    sym(&olddaf, &dafcasing);
    if (m->checksubfaces) {
      tspivot(m, &oldacf, &acfsh);
      tspivot(m, &oldcbf, &cbfsh);
      tspivot(m, &oldbdf, &bdfsh);
      tspivot(m, &olddaf, &dafsh);
    }
    if (m->checksubsegs) {
      /*  Below segs: a->f, d->f, b->f, c->f. */
      fnext(m, &abdf, &worktet);
      enext2self(&worktet);
      tsspivot1(m, &worktet, &afseg);
      enext2fnext(m, &abdf, &worktet);
      enext2self(&worktet);
      tsspivot1(m, &worktet, &dfseg);
      enextfnext(m, &abdf, &worktet);
      enext2self(&worktet);
      tsspivot1(m, &worktet, &bfseg);
      enextfnext(m, &bacf, &worktet);
      enextself(&worktet);
      tsspivot1(m, &worktet, &cfseg);
    }
  }

  /*  Rotate abce, bade one-quarter turn counterclockwise. */
  bond(m, &oldbce, &caecasing);
  bond(m, &oldcae, &adecasing);
  bond(m, &oldade, &dbecasing);
  bond(m, &olddbe, &bcecasing);
  if (m->checksubfaces) {
    /*  Check for subfaces and rebond them to the rotated tets. */
    if (caesh.sh == m->dummysh) {
      tsdissolve(m, &oldbce);
    } else {
      tsbond(m, &oldbce, &caesh);
    }
    if (adesh.sh == m->dummysh) {
      tsdissolve(m, &oldcae);
    } else {
      tsbond(m, &oldcae, &adesh);
    }
    if (dbesh.sh == m->dummysh) {
      tsdissolve(m, &oldade);
    } else {
      tsbond(m, &oldade, &dbesh);
    }
    if (bcesh.sh == m->dummysh) {
      tsdissolve(m, &olddbe);
    } else {
      tsbond(m, &olddbe, &bcesh);
    }
  }
  if (m->checksubsegs) {
    /*  5 edges in abce are changed. */
    enext(&abce, &worktet);  /*  fit b->c into c->a. */
    if (caseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &caseg);
    }
    enext2(&abce, &worktet); /*  fit c->a into a->d. */
    if (adseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &adseg);
    }
    fnext(m, &abce, &worktet); /*  fit b->e into c->e. */
    enextself(&worktet);
    if (ceseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &ceseg);
    }
    enextfnext(m, &abce, &worktet); /*  fit c->e into a->e. */
    enextself(&worktet);
    if (aeseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &aeseg);
    }
    enext2fnext(m, &abce, &worktet); /*  fit a->e into d->e. */
    enextself(&worktet);
    if (deseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &deseg);
    }
    /*  5 edges in bade are changed. */
    enext(&bade, &worktet); /*  fit a->d into d->b. */
    if (dbseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &dbseg);
    }
    enext2(&bade, &worktet); /*  fit d->b into b->c. */
    if (bcseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &bcseg);
    }
    fnext(m, &bade, &worktet); /*  fit a->e into d->e. */
    enextself(&worktet);
    if (deseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &deseg);
    }
    enextfnext(m, &bade, &worktet); /*  fit d->e into b->e. */
    enextself(&worktet);
    if (beseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &beseg);
    }
    enext2fnext(m, &bade, &worktet); /*  fit b->e into c->e. */
    enextself(&worktet);
    if (ceseg.sh == m->dummysh) {
      tssdissolve1(m, &worktet);
    } else {
      tssbond1(m, &worktet, &ceseg);
    }
  }
  if (mirrorflag) {
    /*  Rotate bacf, abdf one-quarter turn counterclockwise. */
    bond(m, &oldcbf, &acfcasing);
    bond(m, &oldacf, &dafcasing);
    bond(m, &olddaf, &bdfcasing);
    bond(m, &oldbdf, &cbfcasing);
    if (m->checksubfaces) {
      /*  Check for subfaces and rebond them to the rotated tets. */
      if (acfsh.sh == m->dummysh) {
        tsdissolve(m, &oldcbf);
      } else {
        tsbond(m, &oldcbf, &acfsh);
      }
      if (dafsh.sh == m->dummysh) {
        tsdissolve(m, &oldacf);
      } else {
        tsbond(m, &oldacf, &dafsh);
      }
      if (bdfsh.sh == m->dummysh) {
        tsdissolve(m, &olddaf);
      } else {
        tsbond(m, &olddaf, &bdfsh);
      }
      if (cbfsh.sh == m->dummysh) {
        tsdissolve(m, &oldbdf);
      } else {
        tsbond(m, &oldbdf, &cbfsh);
      }
    }
    if (m->checksubsegs) {
      /*  5 edges in bacf are changed. */
      enext2(&bacf, &worktet); /*  fit b->c into c->a. */
      if (caseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &caseg);
      }
      enext(&bacf, &worktet); /*  fit c->a into a->d. */
      if (adseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &adseg);
      }
      fnext(m, &bacf, &worktet); /*  fit b->f into c->f. */
      enext2self(&worktet);
      if (cfseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &cfseg);
      }
      enext2fnext(m, &bacf, &worktet); /*  fit c->f into a->f. */
      enext2self(&worktet);
      if (afseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &afseg);
      }
      enextfnext(m, &bacf, &worktet); /*  fit a->f into d->f. */
      enext2self(&worktet);
      if (dfseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &dfseg);
      }
      /*  5 edges in abdf are changed. */
      enext2(&abdf, &worktet); /*  fit a->d into d->b. */
      if (dbseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &dbseg);
      }
      enext(&abdf, &worktet); /*  fit d->b into b->c. */
      if (bcseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &bcseg);
      }
      fnext(m, &abdf, &worktet); /*  fit a->f into d->f. */
      enext2self(&worktet);
      if (dfseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &dfseg);
      }
      enext2fnext(m, &abdf, &worktet); /*  fit d->f into b->f. */
      enext2self(&worktet);
      if (bfseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &bfseg);
      }
      enextfnext(m, &abdf, &worktet); /*  fit b->f into c->f. */
      enext2self(&worktet);
      if (cfseg.sh == m->dummysh) {
        tssdissolve1(m, &worktet);
      } else {
        tssbond1(m, &worktet, &cfseg);
      }
    }
  }

  /*  New vertex assignments for the rotated tetrahedra. */
  setorg(&abce, pd); /*  Update abce to dcae */
  setdest(&abce, pc);
  setapex(&abce, pa);
  setorg(&bade, pc); /*  Update bade to cdbe */
  setdest(&bade, pd);
  setapex(&bade, pb);
  if (mirrorflag) {
    setorg(&bacf, pc); /*  Update bacf to cdaf */
    setdest(&bacf, pd);
    setapex(&bacf, pa);
    setorg(&abdf, pd); /*  Update abdf to dcbf */
    setdest(&abdf, pc);
    setapex(&abdf, pb);
  }

  /*  Update point-to-tet map. */
  setpoint2tet(m, pa, encode(&abce));
  setpoint2tet(m, pb, encode(&bade));
  setpoint2tet(m, pc, encode(&abce));
  setpoint2tet(m, pd, encode(&bade));
  setpoint2tet(m, pe, encode(&abce));
  if (mirrorflag) {
    setpoint2tet(m, pf, encode(&bacf));
  }

  /*  Are there subfaces need to be flipped? */
  if (m->checksubfaces && abc.sh != m->dummysh) {
#ifdef PETSC_USE_DEBUG
    if (bad.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
    /*  Adjust the edge be ab, so the rotation of subfaces is according with */
    /*    the rotation of tetrahedra. */
    ierr = TetGenMeshFindEdge_face(m, &abc, pa, pb);CHKERRQ(ierr);
    /*  Flip an edge of two subfaces, ignore non-Delaunay edges. */
    ierr = TetGenMeshFlip22Sub(m, &abc, PETSC_NULL);CHKERRQ(ierr);
  }

  if (b->verbose > 3) {
    PetscInfo(b->in, "    Updating abce "); ierr = TetGenMeshPrintTet(m, &abce, PETSC_FALSE);CHKERRQ(ierr);
    PetscInfo(b->in, "    Updating bade "); ierr = TetGenMeshPrintTet(m, &bade, PETSC_FALSE);CHKERRQ(ierr);
    if (mirrorflag) {
      PetscInfo(b->in, "    Updating bacf "); ierr = TetGenMeshPrintTet(m, &bacf, PETSC_FALSE);CHKERRQ(ierr);
      PetscInfo(b->in, "    Updating abdf "); ierr = TetGenMeshPrintTet(m, &abdf, PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  if (flipqueue) {
    enextfnext(m, &abce, &bcecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &bcecasing, flipqueue);CHKERRQ(ierr);
    enext2fnext(m, &abce, &caecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &caecasing, flipqueue);CHKERRQ(ierr);
    enextfnext(m, &bade, &adecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &adecasing, flipqueue);CHKERRQ(ierr);
    enext2fnext(m, &bade, &dbecasing);
    ierr = TetGenMeshEnqueueFlipFace(m, &dbecasing, flipqueue);CHKERRQ(ierr);
    if (mirrorflag) {
      enextfnext(m, &bacf, &acfcasing);
      ierr = TetGenMeshEnqueueFlipFace(m, &acfcasing, flipqueue);CHKERRQ(ierr);
      enext2fnext(m, &bacf, &cbfcasing);
      ierr = TetGenMeshEnqueueFlipFace(m, &cbfcasing, flipqueue);CHKERRQ(ierr);
      enextfnext(m, &abdf, &bdfcasing);
      ierr = TetGenMeshEnqueueFlipFace(m, &bdfcasing, flipqueue);CHKERRQ(ierr);
      enext2fnext(m, &abdf, &dafcasing);
      ierr = TetGenMeshEnqueueFlipFace(m, &dafcasing, flipqueue);CHKERRQ(ierr);
    }
    /*  The two new faces dcae (abce), cdbe (bade) may still not be locally */
    /*    Delaunay, and may need be flipped (flip23).  On the other hand, in */
    /*    conforming Delaunay algorithm, two new subfaces dca (abc), and cdb */
    /*    (bad) may be non-conforming Delaunay, they need be queued if they */
    /*    are locally Delaunay but non-conforming Delaunay. */
    ierr = TetGenMeshEnqueueFlipFace(m, &abce, flipqueue);CHKERRQ(ierr);
    ierr = TetGenMeshEnqueueFlipFace(m, &bade, flipqueue);CHKERRQ(ierr);
  }

  /*  Save a live handle in 'recenttet'. */
  m->recenttet = abce;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshLawson3D"
/*  lawson3d()    Perform 3D Lawson flips on non-Delaunay faces/edges.         */
/* tetgenmesh::lawson3d() */
PetscErrorCode TetGenMeshLawson3D(TetGenMesh *m, Queue *flipqueue, long *numFlips)
{
  TetGenOpts    *b  = m->b;
  badface *qface;
  triface flipface = {PETSC_NULL, 0, 0}, symface = {PETSC_NULL, 0, 0}, flipedge = {PETSC_NULL, 0, 0};
  triface neighface = {PETSC_NULL, 0, 0}, symneighface = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0}, checkseg = {PETSC_NULL, 0};
  face neighsh = {PETSC_NULL, 0}, symneighsh = {PETSC_NULL, 0};
  point pa, pb, pc, pd, pe;
  point end1, end2;
  PetscReal sign, ori1, ori2, ori3;
  PetscReal ori4, len, vol;
  long flipcount;
  int copflag;
  int llen, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = QueueLength(flipqueue, &llen);CHKERRQ(ierr);
  PetscInfo1(b->in, "    Lawson flip: %ld faces.\n", llen);
  flipcount = m->flip23s + m->flip32s + m->flip22s + m->flip44s;

  /*  Loop until the queue is empty. */
  while(llen) {
    ierr = QueuePop(flipqueue, (void **) &qface);CHKERRQ(ierr);
    flipface = qface->tt;
    if (isdead_triface(&flipface)) continue;
    if (flipface.tet == m->dummytet) continue;
    /*  Do not flip it if it is a subface. */
    tspivot(m, &flipface, &checksh);
    if (checksh.sh != m->dummysh) continue;

    sym(&flipface, &symface);
    /*  Only do check when the adjacent tet exists and it's not a "fake" tet. */
    if ((symface.tet != m->dummytet) && (oppo(&symface) == qface->foppo)) {
      flipface.ver = 0; /*  CCW. */
      pa = org(&flipface);
      pb = dest(&flipface);
      pc = apex(&flipface);
      pd = oppo(&flipface);
      pe = oppo(&symface);
      ierr = TetGenMeshInSphereS(m, pb, pa, pc, pd, pe, &sign);CHKERRQ(ierr);
      if (sign == 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

      if (sign > 0.0) {
        /*  flipface is not locally Delaunay. Try to flip it. */
        ori1 = TetGenOrient3D(pa, pb, pd, pe);
        ori2 = TetGenOrient3D(pb, pc, pd, pe);
        ori3 = TetGenOrient3D(pc, pa, pd, pe);

        flipedge = flipface;  /*  Initialize flipedge. */
        copflag = 0;

        /*  Find a suitable flip. */
        if (ori1 > 0) {
          if (ori2 > 0) {
            if (ori3 > 0) { /*  (+++) */
              /*  A 2-to-3 flip is found. */
              /*  Do not flip it if it is a subface. */
              /*  tspivot(flipface, checksh); */
              /*  if (checksh.sh == dummysh) { */
                /*  Do not flip it if it will create a tet spanning two */
                /*    "coplanar" subfaces. We treat this case as either */
                /*    a 2-to-2 or a 4-to-4 flip. */
                for(i = 0; i < 3; i++) {
                  ierr = TetGenMeshTssPivot(m, &flipface, &checkseg);CHKERRQ(ierr);
                  if (checkseg.sh == m->dummysh) {
                    fnext(m, &flipface, &neighface);
                    tspivot(m, &neighface, &neighsh);
                    if (neighsh.sh != m->dummysh) {
                      /*  Check if there exist another subface. */
                      symedge(m, &flipface, &symface);
                      fnext(m, &symface, &symneighface);
                      tspivot(m, &symneighface, &symneighsh);
                      if (symneighsh.sh != m->dummysh) {
                        /*  Do not flip this face. Try to do a 2-to-2 or a */
                        /*    4-to-4 flip instead. */
                        flipedge = flipface;
                        copflag = 1;
                        break;
                      }
                    }
                  }
                  enextself(&flipface);
                }
                if (i == 3) {
                  /*  Do not flip if it will create a nearly degenerate tet */
                  /*    at a segment. Once we created such a tet, it may */
                  /*    prevent you to split the segment later. An example */
                  /*    is in dump-.lua */
                  for(i = 0; i < 3; i++) {
                    ierr = TetGenMeshTssPivot(m, &flipface, &checkseg);CHKERRQ(ierr);
                    if (checkseg.sh != m->dummysh) {
                      end1 = (point) checkseg.sh[3];
                      end2 = (point) checkseg.sh[4];
                      ori4 = TetGenOrient3D(end1, end2, pd, pe);
                      len = distance(end1, end2);
                      vol = len * len * len;
                      /*  Is it nearly degnerate? */
                      if ((fabs(ori4) / vol) < b->epsilon) {
                        flipedge = flipface;
                        copflag = 0;
                        break;
                      }
                    }
                    enextself(&flipface);
                  }
                  if (i == 3) {
                    ierr = TetGenMeshFlip23(m, &flipface, flipqueue);CHKERRQ(ierr);
                    continue;
                  }
                }
              /*  } */
            } else {
              if (ori3 < 0) { /*  (++-) */
                /*  Try to flip edge [c, a]. */
                flipedge.ver = 4;
                copflag = 0;
              } else { /*  (++0) */
                /*  A 2-to-2 or 4-to-4 flip at edge [c, a]. */
                flipedge.ver = 4;
                copflag = 1;
              }
            }
          } else {
            if (ori2 < 0) {
              if (ori3 > 0) { /*  (+-+) */
                /*  Try to flip edge [b, c]. */
                flipedge.ver = 2;
                copflag = 0;
              } else {
                if (ori3 < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]"); /*  (+--) */
                else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]"); /*  (+-0) */
              }
            } else { /*  ori2 == 0 */
              if (ori3 > 0) { /*  (+0+) */
                /*  A 2-to-2 or 4-to-4 flip at edge [b, c]. */
                flipedge.ver = 2;
                copflag = 1;
              } else {
                if (ori3 < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]"); /*  (+0-) */
                else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]"); /*  (+00) */
              }
            }
          }
        } else {
          if (ori1 < 0) {
            if (ori2 > 0) {
              if (ori3 > 0) { /*  (-++) */
                /*  Try to flip edge [a, b]. */
                flipedge.ver = 0;
                copflag = 0;
              } else {
                if (ori3 < 0) { /*  (-+-) */
                  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]");
                } else { /*  (-+0) */
                  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]");
                }
              }
            } else {
              if (ori2 < 0) {
                if (ori3 > 0) { /*  (--+) */
                  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]");
                } else {
                  if (ori3 < 0) { /*  (---) */
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]");
                  } else { /*  (--0) */
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]");
                  }
                }
              } else { /*  ori2 == 0 */
                if (ori3 > 0) { /*  (-0+) */
                  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]");
                } else {
                  if (ori3 < 0) { /*  (-0-) */
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]");
                  } else { /*  (-00) */
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not possible when pe is inside the circumsphere of the tet [pa, pb, pc, pd]");
                  }
                }
              }
            }
          } else { /*  ori1 == 0 */
            if (ori2 > 0) {
              if (ori3 > 0) { /*  (0++) */
                /*  A 2-to-2 or 4-to-4 flip at edge [a, b]. */
                flipedge.ver = 0;
                copflag = 1;
              } else {
                if (ori3 < 0) { /*  (0+-) */
                  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                } else { /*  (0+0) */
                  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                }
              }
            } else {
              if (ori2 < 0) {
                if (ori3 > 0) { /*  (0-+) */
                  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                } else {
                  if (ori3 < 0) { /*  (0--) */
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                  } else { /*  (0-0) */
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                  }
                }
              } else {
                if (ori3 > 0) { /*  (00+) */
                  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                } else {
                  if (ori3 < 0) { /*  (00-) */
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                  } else { /*  (000) */
                    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                  }
                }
              }
            }
          }
        }

        /*  An edge (flipedge) is going to be flipped. */
        /*  Do not flip it it is a subsegment. */
        ierr = TetGenMeshTssPivot(m, &flipedge, &checkseg);CHKERRQ(ierr);
        if (checkseg.sh == m->dummysh) {
          symedge(m, &flipedge, &symface);
          if (copflag == 0) {
            /*  Check if a 3-to-2 flip is possible. */
            tfnext(m, &flipedge, &neighface);
            if (neighface.tet != m->dummytet) {
              /*  Check if neighface is a subface. */
              tspivot(m, &neighface, &neighsh);
              if (neighsh.sh == m->dummysh) {
                tfnext(m, &symface, &symneighface);
                if (neighface.tet == symneighface.tet) {
                  /*  symneighface should not be a subface. Check it. */
                  tspivot(m, &symneighface, &symneighsh);
                  if (symneighsh.sh != m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                  /*  Found a 3-to-2 flip. */
                  ierr = TetGenMeshFlip32(m, &flipedge, flipqueue);CHKERRQ(ierr);
                }
              } else {
                /*  neighsh is a subface. Check a potential 4-to-4 flip. */
                tfnext(m, &symface, &symneighface);
                tspivot(m, &symneighface, &symneighsh);
                if (symneighsh.sh != m->dummysh) {
                  if (oppo(&neighface) == oppo(&symneighface)) {
                    /*  Found a 4-to-4 flip. */
                    ierr = TetGenMeshFlip22(m, &flipedge, flipqueue);CHKERRQ(ierr);
                  }
                }
              }
            } else {
              /*  neightface is a hull face. Since flipedge is not a segment */
              /*    and this edge is locally non-convex. */
              tfnext(m, &symface, &symneighface);
              /*  symneighface should also be a hull face. */
              if (symneighface.tet == m->dummytet) {
                /*  Force a 2-to-2 flip (recovery of Delaunay). */
                ierr = TetGenMeshFlip22(m, &flipedge, flipqueue);CHKERRQ(ierr);
              }
            }
          } else {
            /*  Check if a 2-to-2 or 4-to-4 flip is possible. */
            tfnext(m, &flipedge, &neighface);
            tfnext(m, &symface, &symneighface);
            if (neighface.tet != m->dummytet) {
              if (symneighface.tet != m->dummytet) {
                if (oppo(&neighface) == oppo(&symneighface)) {
                  /*  Found a 4-to-4 flip. */
                  ierr = TetGenMeshFlip22(m, &flipedge, flipqueue);CHKERRQ(ierr);
                }
              }
            } else {
              if (symneighface.tet == m->dummytet) {
                /*  Found a 2-to-2 flip. */
                ierr = TetGenMeshFlip22(m, &flipedge, flipqueue);CHKERRQ(ierr);
              }
            }
          }
        }

      } /*  if (sign > 0) */
    }
    ierr = QueueLength(flipqueue, &llen);CHKERRQ(ierr);
  } /*  while (!flipqueue->empty()) */

  flipcount = m->flip23s + m->flip32s + m->flip22s + m->flip44s - flipcount;
  PetscInfo1(b->in, "    %ld flips.\n", flipcount);

  if (numFlips) {*numFlips = flipcount;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshLawson"
/*  lawson()    Perform lawson flips on non-Delaunay edges.                    */
/*                                                                             */
/*  Assumpation:  Current triangulation T contains non-Delaunay edges (after   */
/*  inserting a point or performing a flip). Non-Delaunay edges are queued in  */
/*  'facequeue'. Returns the total number of flips done during this call.      */
/* tetgenmesh::lawson() */
PetscErrorCode TetGenMeshLawson(TetGenMesh *m, Queue *flipqueue, long *numFlips)
{
  TetGenOpts    *b  = m->b;
  badface *qedge;
  face flipedge = {PETSC_NULL, 0}, symedge = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  point pa, pb, pc, pd;
  PetscReal vab[3], vac[3], vad[3];
  PetscReal dot1, dot2, lac, lad;
  PetscReal sign, ori;
  int edgeflips, maxflips;
  int len, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = QueueLength(flipqueue, &len);CHKERRQ(ierr);
  PetscInfo1(b->in, "    Lawson flip: %ld edges.\n", len);

  if (b->diagnose) {
    maxflips = (int) ((len + 1l) * 3l);
    maxflips *= maxflips;
  } else {
    maxflips = -1;
  }
  edgeflips = 0;

  while(len > 0 && maxflips != 0) {
    ierr = QueuePop(flipqueue, (void **) &qedge);CHKERRQ(ierr);
    ierr = QueueLength(flipqueue, &len);CHKERRQ(ierr);
    flipedge = qedge->ss;
    if (flipedge.sh == m->dummysh) continue;
    if ((sorg(&flipedge) != qedge->forg) || (sdest(&flipedge) != qedge->fdest)) continue;
    sspivot(m, &flipedge, &checkseg);
    if (checkseg.sh != m->dummysh) continue;  /*  Can't flip a subsegment. */
    spivot(&flipedge, &symedge);
    if (symedge.sh == m->dummysh) continue; /*  Can't flip a hull edge. */
    pa = sorg(&flipedge);
    pb = sdest(&flipedge);
    pc = sapex(&flipedge);
    pd = sapex(&symedge);
    /*  Choose the triangle abc or abd as the base depending on the angle1 */
    /*    (Vac, Vab) and angle2 (Vad, Vab). */
    for(i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
    for(i = 0; i < 3; i++) vac[i] = pc[i] - pa[i];
    for(i = 0; i < 3; i++) vad[i] = pd[i] - pa[i];
    dot1 = dot(vac, vab);
    dot2 = dot(vad, vab);
    dot1 *= dot1;
    dot2 *= dot2;
    lac = dot(vac, vac);
    lad = dot(vad, vad);
    if (lad * dot1 <= lac * dot2) {
      /*  angle1 is closer to 90 than angle2, choose abc (flipedge). */
      m->abovepoint = m->facetabovepointarray[shellmark(m, &flipedge)];
      if (!m->abovepoint) {
        ierr = TetGenMeshGetFacetAbovePoint(m, &flipedge);CHKERRQ(ierr);
      }
      sign = TetGenInsphere(pa, pb, pc, m->abovepoint, pd);
      ori  = TetGenOrient3D(pa, pb, pc, m->abovepoint);
    } else {
      /*  angle2 is closer to 90 than angle1, choose abd (symedge). */
      m->abovepoint = m->facetabovepointarray[shellmark(m, &symedge)];
      if (!m->abovepoint) {
        ierr = TetGenMeshGetFacetAbovePoint(m, &symedge);CHKERRQ(ierr);
      }
      sign = TetGenInsphere(pa, pb, pd, m->abovepoint, pc);
      ori  = TetGenOrient3D(pa, pb, pd, m->abovepoint);
    }
    /*  Correct the sign. */
    sign = ori > 0.0 ? sign : -sign;
    if (sign > 0.0) {
      /*  Flip the non-Delaunay edge. */
      ierr = TetGenMeshFlip22Sub(m, &flipedge, flipqueue);CHKERRQ(ierr);
      edgeflips++;
      if (maxflips > 0) maxflips--;
    }
    ierr = QueueLength(flipqueue, &len);CHKERRQ(ierr);
  }

  if (!maxflips) {
    PetscInfo(b->in, "Warning:  Maximal number of flips reached !\n");
  }
  PetscInfo1(b->in, "  Total %d flips.\n", edgeflips);

  if (numFlips) {*numFlips = edgeflips;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRemoveEdgeByFlip32"
/* removeedgebyflip32()    Remove an edge by a 3-to-2 flip.                  */
/*                                                                           */
/* 'abtetlist' contains 3 tets sharing ab. Imaging that ab is perpendicular  */
/* to the screen, where a lies in front of and b lies behind it. The 3 tets  */
/* of the list are: [0]abce, [1]abdc, and [2]abed, respectively.             */
/* Comment: the edge ab is in CW edge ring of the three faces: abc, abd, and */
/* abe. (2009-06-29)                                                         */
/*                                                                           */
/* This routine forms two new tets that ab is not an edge of them. Save them */
/* in 'newtetlist', [0]dcea, [1]cdeb. Note that the new tets may not valid   */
/* if one of them get inverted. return false if so.                          */
/*                                                                           */
/* If 'key' != NULL.  The old tets are replaced by the new tets only if the  */
/* local mesh quality is improved. Current 'key' = cos(\theta), where \theta */
/* is the maximum dihedral angle in the old tets.                            */
/*                                                                           */
/* If the edge is flipped, 'newtetlist' returns the two new tets. The three  */
/* tets in 'abtetlist' are NOT deleted.  The caller has the right to either  */
/* delete them or reverse the operation.                                     */
/* tetgenmesh::removeedgebyflip32() */
PetscErrorCode TetGenMeshRemoveEdgeByFlip32(TetGenMesh *m, PetscReal *key, triface *abtetlist, triface *newtetlist, Queue *flipque, PetscBool *isFlipped)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  triface dcea = {PETSC_NULL, 0, 0}, cdeb = {PETSC_NULL, 0, 0}; /* new configuration. */
  triface newfront = {PETSC_NULL, 0, 0}, oldfront = {PETSC_NULL, 0, 0}, adjfront = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0};
  point pa, pb, pc, pd, pe;
  PetscReal ori, cosmaxd, d1, d2;
  PetscReal attrib, volume;
  PetscBool doflip;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pa = org(&abtetlist[0]);
  pb = dest(&abtetlist[0]);
  pc = apex(&abtetlist[0]);
  pd = apex(&abtetlist[1]);
  pe = apex(&abtetlist[2]);

  ori = TetGenOrient3D(pd, pc, pe, pa);
  if (ori < 0.0) {
    ori = TetGenOrient3D(pc, pd, pe, pb);
  }
  doflip = (ori < 0.0) ? PETSC_TRUE : PETSC_FALSE; /* Can ab be flipped away? */

  /* Does the caller ensure a valid configuration? */
  if (doflip && key) {
    if (*key > -1.0) {
      /* Test if the new tets reduce the maximal dihedral angle. */
      ierr = TetGenMeshTetAllDihedral(m, pd, pc, pe, pa, PETSC_NULL, &d1, PETSC_NULL);CHKERRQ(ierr);
      ierr = TetGenMeshTetAllDihedral(m, pc, pd, pe, pb, PETSC_NULL, &d2, PETSC_NULL);CHKERRQ(ierr);
      cosmaxd = d1 < d2 ? d1 : d2; /* Choose the bigger angle. */
      doflip = (*key < cosmaxd) ? PETSC_TRUE : PETSC_FALSE; /* Can local quality be improved? */
      /* Return the key */
      *key = cosmaxd;
    }
  }

  /* Comment: This edge must not be fixed. It has been checked before.*/
  if (doflip && m->elemfliplist) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    /* Regist this flip. */
    ierr = TetGenMeshRegisterElemFlip(m, T32, pa, pb, dummypoint, pc, pd, pe, &isFlipped);CHKERRQ(ierr);
    if (!isFlipped) {
      /* Detected a potential flip loop. Don't do it. */
      if (isFlipped) {*isFlipped = PETSC_FALSE;}
      PetscFunctionReturn(0);
    }
#endif
  }

  if (doflip) {
    /* Create the new tets. */
    ierr = TetGenMeshMakeTetrahedron(m, &dcea);CHKERRQ(ierr);
    setorg(&dcea, pd);
    setdest(&dcea, pc);
    setapex(&dcea, pe);
    setoppo(&dcea, pa);
    ierr = TetGenMeshMakeTetrahedron(m, &cdeb);CHKERRQ(ierr);
    setorg(&cdeb, pc);
    setdest(&cdeb, pd);
    setapex(&cdeb, pe);
    setoppo(&cdeb, pb);
    /* Transfer the element attributes. */
    for(i = 0; i < in->numberoftetrahedronattributes; i++) {
      attrib = elemattribute(m, abtetlist[0].tet, i);
      setelemattribute(m, dcea.tet, i, attrib);
      setelemattribute(m, cdeb.tet, i, attrib);
    }
    /* Transfer the volume constraints. */
    if (b->varvolume && !b->refine) {
      volume = volumebound(m, abtetlist[0].tet);
      setvolumebound(m, dcea.tet, volume);
      setvolumebound(m, cdeb.tet, volume);
    }
    /* Return two new tets. */
    newtetlist[0] = dcea;
    newtetlist[1] = cdeb;
    /* Glue the two new tets. */
    bond(m, &dcea, &cdeb);
    /* Substitute the two new tets into the old three-tets cavity. */
    for(i = 0; i < 3; i++) {
      fnext(m, &dcea, &newfront); /* face dca, cea, eda. */
      esym(&abtetlist[(i + 1) % 3], &oldfront);
      enextfnextself(m, &oldfront);
      /* Get the adjacent tet at the face (may be a dummytet). */
      sym(&oldfront, &adjfront);
      bond(m, &newfront, &adjfront);
      if (m->checksubfaces) {
        tspivot(m, &oldfront, &checksh);
        if (checksh.sh != m->dummysh) {
          tsbond(m, &newfront, &checksh);
        }
      }
      if (flipque) {
        ierr = TetGenMeshEnqueueFlipFace(m, &newfront, flipque);CHKERRQ(ierr);
      }
      enext2self(&dcea);
    }
    for(i = 0; i < 3; i++) {
      fnext(m, &cdeb, &newfront); /* face cdb, deb, ecb. */
      esym(&abtetlist[(i + 1) % 3], &oldfront);
      enext2fnextself(m, &oldfront);
      /* Get the adjacent tet at the face (may be a dummytet). */
      sym(&oldfront, &adjfront);
      bond(m, &newfront, &adjfront);
      if (m->checksubfaces) {
        tspivot(m, &oldfront, &checksh);
        if (checksh.sh != m->dummysh) {
          tsbond(m, &newfront, &checksh);
        }
      }
      if (flipque) {
        ierr = TetGenMeshEnqueueFlipFace(m, &newfront, flipque);CHKERRQ(ierr);
      }
      enextself(&cdeb);
    }
    if (isFlipped) {*isFlipped = PETSC_TRUE;}
    PetscFunctionReturn(0);
  } /* if (doflip) */

  if (isFlipped) {*isFlipped = PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRemoveEdgeByTranNM"
/* removeedgebytranNM()    Remove an edge by transforming n-to-m tets.       */
/*                                                                           */
/* This routine attempts to remove a given edge (ab) by transforming the set */
/* T of tets surrounding ab into another set T' of tets.  T and T' have the  */
/* same outer faces and ab is not an edge of T' anymore. Let |T|=n, and |T'| */
/* =m, it is actually a n-to-m flip for n > 3.  The relation between n and m */
/* depends on the method, ours is found below.                               */
/*                                                                           */
/* 'abtetlist' contains n tets sharing ab. Imaging that ab is perpendicular  */
/* to the screen, where a lies in front of and b lies behind it.  Let the    */
/* projections of the n apexes onto screen in clockwise order are: p_0, ...  */
/* p_n-1, respectively. The tets in the list are: [0]abp_0p_n-1,[1]abp_1p_0, */
/* ..., [n-1]abp_n-1p_n-2, respectively.                                     */
/*                                                                           */
/* The principle of the approach is: Recursively reduce the link of ab by    */
/* using flip23 until only three faces remain, hence a flip32 can be applied */
/* to remove ab. For a given face a.b.p_0, check a flip23 can be applied on  */
/* it, i.e, edge p_1.p_n-1 crosses it. NOTE*** We do the flip even p_1.p_n-1 */
/* intersects with a.b (they are coplanar). If so, a degenerate tet (a.b.p_1.*/
/* p_n-1) is temporarily created, but it will be eventually removed by the   */
/* final flip32. This relaxation splits a flip44 into flip23 + flip32. *NOTE */
/* Now suppose a.b.p_0 gets flipped, p_0 is not on the link of ab anymore.   */
/* The link is then reduced (by 1). 2 of the 3 new tets, p_n-1.p_1.p_0.a and */
/* p_1.p_n-1.p_0.b, will be part of the new configuration.  The left new tet,*/
/* a.b.p_1.p_n-1, goes into the new link of ab. A recurrence can be applied. */
/*                                                                           */
/* If 'e1' and 'e2' are not NULLs, they specify an wanted edge to appear in  */
/* the new tet configuration. In such case, only do flip23 if edge e1<->e2   */
/* can be recovered. It is used in removeedgebycombNM().                     */
/*                                                                           */
/* If ab gets removed. 'newtetlist' contains m new tets.  By using the above */
/* approach, the pairs (n, m) can be easily enumerated.  For example, (3, 2),*/
/* (4, 4), (5, 6), (6, 8), (7, 10), (8, 12), (9, 14), (10, 16),  and so on.  */
/* It is easy to deduce, that m = (n - 2) * 2, when n >= 3.  The n tets in   */
/* 'abtetlist' are NOT deleted in this routine. The caller has the right to  */
/* either delete them or reverse this operation.                             */
/* tetgenmesh::removeedgebytranNM() */
PetscErrorCode TetGenMeshRemoveEdgeByTranNM(TetGenMesh *m, PetscReal *key, PetscInt n, triface *abtetlist, triface *newtetlist, point e1, point e2, Queue *flipque, PetscBool *isFlipped)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  triface tmpabtetlist[21]; /* Temporary max 20 tets configuration. */
  triface newfront = {PETSC_NULL, 0, 0}, oldfront = {PETSC_NULL, 0, 0}, adjfront = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0};
  point pa, pb, p[21];
  PetscReal ori, cosmaxd, d1, d2;
  PetscReal tmpkey;
  PetscReal attrib, volume;
  PetscBool doflip, copflag, success;
  PetscInt i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Maximum 20 tets. */
  if (n >= 20) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This routine cannot handle %d > 20 tets", n);
  /* Two points a and b are fixed. */
  pa = org(&abtetlist[0]);
  pb = dest(&abtetlist[0]);
  /* The points p_0, p_1, ..., p_n-1 are permuted in each new configuration. */
  /*   These permutations can be easily done in the following loop. */
  /* Loop through all the possible new tets configurations. Stop on finding */
  /*   a valid new tet configuration which also immproves the quality value. */
  for(i = 0; i < n; i++) {
    /* Get other n points for the current configuration. */
    for(j = 0; j < n; j++) {
      p[j] = apex(&abtetlist[(i + j) % n]);
    }
    /* Is there a wanted edge? */
    if (e1 && e2) {
      /* Yes. Skip this face if p[1]<->p[n-1] is not the edge. */
      if (!(((p[1] == e1) && (p[n - 1] == e2)) ||
	    ((p[1] == e2) && (p[n - 1] == e1)))) continue;
    }
    /* Test if face a.b.p_0 can be flipped (by flip23), ie, to check if the */
    /*   edge p_n-1.p_1 crosses face a.b.p_0 properly. */
    /* Note. It is possible that face a.b.p_0 has type flip44, ie, a,b,p_1, */
    /*   and p_n-1 are coplanar. A trick is to split the flip44 into two */
    /*   steps: frist a flip23, then a flip32. The first step creates a */
    /*   degenerate tet (vol=0) which will be removed by the second flip. */
    ori = TetGenOrient3D(pa, pb, p[1], p[n - 1]);
    copflag = (ori == 0.0) ? PETSC_TRUE : PETSC_FALSE; /* Are they coplanar? */
    if (ori >= 0.0) {
      /* Accept the coplanar case which supports flip44. */
      ori = TetGenOrient3D(pb, p[0], p[1], p[n - 1]);
      if (ori > 0.0) {
        ori = TetGenOrient3D(p[0], pa, p[1], p[n - 1]);
      }
    }
    /* Is face abc flipable? */
    if (ori > 0.0) {
      /* A valid (2-to-3) flip (or 4-to-4 flip) is found. */
      copflag ? m->flip44s++ : m->flip23s++;
      doflip = PETSC_TRUE;
      if (key) {
        if (*key > -1.0) {
          /* Test if the new tets reduce the maximal dihedral angle. Only 2 */
          /*   tets, p_n-1.p_1.p_0.a and p_1.p_n-1.p_0.b, need to be tested */
          /*   The left one a.b.p_n-1.p_1 goes into the new link of ab. */
          ierr = TetGenMeshTetAllDihedral(m, p[n - 1], p[1], p[0], pa, PETSC_NULL, &d1, PETSC_NULL);CHKERRQ(ierr);
          ierr = TetGenMeshTetAllDihedral(m, p[1], p[n - 1], p[0], pb, PETSC_NULL, &d2, PETSC_NULL);CHKERRQ(ierr);
          cosmaxd = d1 < d2 ? d1 : d2; /* Choose the bigger angle. */
          doflip = *key < cosmaxd ? PETSC_TRUE : PETSC_FALSE; /* Can the local quality be improved? */
        }
      }
      if (doflip && m->elemfliplist) {
#if 1
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
        /* Comment: The flipping face must be not fixed. This case has been */
        /*   tested during collecting the face ring of this edge. */
        /* Do not flip this face if it has been registered before. */
        if (!registerelemflip(T23, pa, pb, p[0], p[1], p[n-1], m->dummypoint)) {
          doflip = PETSC_FALSE; /* Do not flip this face. */
        }
#endif
      }
      if (doflip) {
        tmpkey = key ? *key : -1.0;
        /* Create the two new tets. */
        ierr = TetGenMeshMakeTetrahedron(m, &(newtetlist[0]));CHKERRQ(ierr);
        setorg(&newtetlist[0], p[n - 1]);
        setdest(&newtetlist[0], p[1]);
        setapex(&newtetlist[0], p[0]);
        setoppo(&newtetlist[0], pa);
        ierr = TetGenMeshMakeTetrahedron(m, &(newtetlist[1]));CHKERRQ(ierr);
        setorg(&newtetlist[1], p[1]);
        setdest(&newtetlist[1], p[n - 1]);
        setapex(&newtetlist[1], p[0]);
        setoppo(&newtetlist[1], pb);
        /* Create the n - 1 temporary new tets (the new Star(ab)). */
        ierr = TetGenMeshMakeTetrahedron(m, &(tmpabtetlist[0]));CHKERRQ(ierr);
        setorg(&tmpabtetlist[0], pa);
        setdest(&tmpabtetlist[0], pb);
        setapex(&tmpabtetlist[0], p[n - 1]);
        setoppo(&tmpabtetlist[0], p[1]);
        for(j = 1; j < n - 1; j++) {
          ierr = TetGenMeshMakeTetrahedron(m, &(tmpabtetlist[j]));CHKERRQ(ierr);
          setorg(&tmpabtetlist[j], pa);
          setdest(&tmpabtetlist[j], pb);
          setapex(&tmpabtetlist[j], p[j]);
          setoppo(&tmpabtetlist[j], p[j + 1]);
        }
        /* Transfer the element attributes. */
        for(j = 0; j < in->numberoftetrahedronattributes; j++) {
          attrib = elemattribute(m, abtetlist[0].tet, j);
          setelemattribute(m, newtetlist[0].tet, j, attrib);
          setelemattribute(m, newtetlist[1].tet, j, attrib);
          for(k = 0; k < n - 1; k++) {
            setelemattribute(m, tmpabtetlist[k].tet, j, attrib);
          }
        }
        /* Transfer the volume constraints. */
        if (b->varvolume && !b->refine) {
          volume = volumebound(m, abtetlist[0].tet);
          setvolumebound(m, newtetlist[0].tet, volume);
          setvolumebound(m, newtetlist[1].tet, volume);
          for(k = 0; k < n - 1; k++) {
            setvolumebound(m, tmpabtetlist[k].tet, volume);
          }
        }
        /* Glue the new tets at their internal faces: 2 + (n - 1). */
        bond(m, &newtetlist[0], &newtetlist[1]); /* p_n-1.p_1.p_0. */
        fnext(m, &newtetlist[0], &newfront);
        enext2fnext(m, &tmpabtetlist[0], &adjfront);
        bond(m, &newfront, &adjfront); /* p_n-1.p_1.a. */
        fnext(m, &newtetlist[1], &newfront);
        enextfnext(m, &tmpabtetlist[0], &adjfront);
        bond(m, &newfront, &adjfront); /* p_n-1.p_1.b. */
        /* Glue n - 1 internal faces around ab. */
        for(j = 0; j < n - 1; j++) {
          fnext(m, &tmpabtetlist[j], &newfront);
          bond(m, &newfront, &tmpabtetlist[(j + 1) % (n - 1)]); /* a.b.p_j+1 */
        }
        /* Substitute the old tets with the new tets by connecting the new */
        /*   tets to the adjacent tets in the mesh. There are n * 2 (outer) */
        /*   faces of the new tets need to be operated. */
        /* Note, after the substitution, the old tets still have pointers to */
        /*   their adjacent tets in the mesh.  These pointers can be re-used */
        /*   to inverse the substitution. */
        for(j = 0; j < n; j++) {
          /* Get an old tet: [0]a.b.p_0.p_n-1 or [j]a.b.p_j.p_j-1, (j > 0). */
          oldfront = abtetlist[(i + j) % n];
          esymself(&oldfront);
          enextfnextself(m, &oldfront);
          /* Get an adjacent tet at face: [0]a.p_0.p_n-1 or [j]a.p_j.p_j-1. */
          sym(&oldfront, &adjfront); /* adjfront may be dummy. */
          /* Get the corresponding face from the new tets. */
          if (j == 0) {
            enext2fnext(m, &newtetlist[0], &newfront); /* a.p_0.n_n-1 */
          } else if (j == 1) {
            enextfnext(m, &newtetlist[0], &newfront); /* a.p_1.p_0 */
          } else { /* j >= 2. */
            enext2fnext(m, &tmpabtetlist[j - 1], &newfront); /* a.p_j.p_j-1 */
          }
          bond(m, &newfront, &adjfront);
          if (m->checksubfaces) {
            tspivot(m, &oldfront, &checksh);
            if (checksh.sh != m->dummysh) {
              tsbond(m, &newfront, &checksh);
            }
          }
          if (flipque) {
            /* Only queue the faces of the two new tets. */
            if (j < 2) {ierr = TetGenMeshEnqueueFlipFace(m, &newfront, flipque);CHKERRQ(ierr);}
          }
        }
        for(j = 0; j < n; j++) {
          /* Get an old tet: [0]a.b.p_0.p_n-1 or [j]a.b.p_j.p_j-1, (j > 0). */
          oldfront = abtetlist[(i + j) % n];
          esymself(&oldfront);
          enext2fnextself(m, &oldfront);
          /* Get an adjacent tet at face: [0]b.p_0.p_n-1 or [j]b.p_j.p_j-1. */
          sym(&oldfront, &adjfront); /* adjfront may be dummy. */
          /* Get the corresponding face from the new tets. */
          if (j == 0) {
            enextfnext(m, &newtetlist[1], &newfront); /* b.p_0.n_n-1 */
          } else if (j == 1) {
            enext2fnext(m, &newtetlist[1], &newfront); /* b.p_1.p_0 */
          } else { /* j >= 2. */
            enextfnext(m, &tmpabtetlist[j - 1], &newfront); /* b.p_j.p_j-1 */
          }
          bond(m, &newfront, &adjfront);
          if (m->checksubfaces) {
            tspivot(m, &oldfront, &checksh);
            if (checksh.sh != m->dummysh) {
              tsbond(m, &newfront, &checksh);
            }
          }
          if (flipque) {
            /* Only queue the faces of the two new tets. */
            if (j < 2) {ierr = TetGenMeshEnqueueFlipFace(m, &newfront, flipque);CHKERRQ(ierr);}
          }
        }
        /* Adjust the faces in the temporary new tets at ab for recursively */
        /*   processing on the n-1 tets.(See the description at beginning) */
        for(j = 0; j < n - 1; j++) {
          fnextself(m, &tmpabtetlist[j]);
        }
        if (n > 4) {
          ierr = TetGenMeshRemoveEdgeByTranNM(m, &tmpkey, n-1, tmpabtetlist, &(newtetlist[2]), PETSC_NULL, PETSC_NULL, flipque, &success);CHKERRQ(ierr);
        } else { /* assert(n == 4); */
          ierr = TetGenMeshRemoveEdgeByFlip32(m, &tmpkey, tmpabtetlist, &(newtetlist[2]), flipque, &success);CHKERRQ(ierr);
        }
        /* No matter it was success or not, delete the temporary tets. */
        for(j = 0; j < n - 1; j++) {
          ierr = TetGenMeshTetrahedronDealloc(m, tmpabtetlist[j].tet);CHKERRQ(ierr);
        }
        if (success) {
          /* The new configuration is good. */
          /* Do not delete the old tets. */
          /* for (j = 0; j < n; j++) { */
          /*   tetrahedrondealloc(abtetlist[j].tet); */
          /* } */
          /* Save the minimal improved quality value. */
          if (key) {
            *key = (tmpkey < cosmaxd ? tmpkey : cosmaxd);
          }
          if (isFlipped) {*isFlipped = PETSC_TRUE;}
          PetscFunctionReturn(0);
        } else {
          /* The new configuration is bad, substitue back the old tets. */
          if (m->elemfliplist) {
            /* Remove the last registered 2-to-3 flip. */
            m->elemfliplist->objects--;
          }
          for(j = 0; j < n; j++) {
            oldfront = abtetlist[(i + j) % n];
            esymself(&oldfront);
            enextfnextself(m, &oldfront); /* [0]a.p_0.p_n-1, [j]a.p_j.p_j-1. */
            sym(&oldfront, &adjfront); /* adjfront may be dummy. */
            bond(m, &oldfront, &adjfront);
            if (m->checksubfaces) {
              tspivot(m, &oldfront, &checksh);
              if (checksh.sh != m->dummysh) {
                tsbond(m, &oldfront, &checksh);
              }
            }
          }
          for(j = 0; j < n; j++) {
            oldfront = abtetlist[(i + j) % n];
            esymself(&oldfront);
            enext2fnextself(m, &oldfront); /* [0]b.p_0.p_n-1, [j]b.p_j.p_j-1. */
            sym(&oldfront, &adjfront); /* adjfront may be dummy */
            bond(m, &oldfront, &adjfront);
            if (m->checksubfaces) {
              tspivot(m, &oldfront, &checksh);
              if (checksh.sh != m->dummysh) {
                tsbond(m, &oldfront, &checksh);
              }
            }
          }
          /* Delete the new tets. */
          ierr = TetGenMeshTetrahedronDealloc(m, newtetlist[0].tet);CHKERRQ(ierr);
          ierr = TetGenMeshTetrahedronDealloc(m, newtetlist[1].tet);CHKERRQ(ierr);
          /* If tmpkey has been modified, then the failure was not due to */
          /*   unflipable configuration, but the non-improvement. */
          if (key && (tmpkey < *key)) {
            *key = tmpkey;
            if (isFlipped) {*isFlipped = PETSC_FALSE;}
            PetscFunctionReturn(0);
          }
        } /* if (success) */
      } /* if (doflip) */
    } /* if (ori > 0.0) */
  } /* for (i = 0; i < n; i++) */

  if (isFlipped) {*isFlipped = PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRemoveEdgeByCombNM"
/* removeedgebycombNM()    Remove an edge by combining two flipNMs.          */
/*                                                                           */
/* Given a set T of tets surrounding edge ab. The premise is that ab can not */
/* be removed by a flipNM. This routine attempts to remove ab by two flipNMs,*/
/* i.e., first find and flip an edge af (or bf) by flipNM, then flip ab by   */
/* flipNM. If it succeeds, two sets T(ab) and T(af) of tets are replaced by  */
/* a new set T' and both ab and af are not edges in T' anymore.              */
/*                                                                           */
/* 'abtetlist' contains n tets sharing ab. Imaging that ab is perpendicular  */
/* to the screen, such that a lies in front of and b lies behind it. Let the */
/* projections of the n apexes on the screen in clockwise order are: p_0,...,*/
/* p_n-1, respectively. So the list of tets are: [0]abp_0p_n-1, [1]abp_1p_0, */
/* ..., [n-1]abp_n-1p_n-2, respectively.                                     */
/*                                                                           */
/* The principle of the approach is: for a face a.b.p_0, check if edge b.p_0 */
/* is of type N32 (or N44). If it is, then try to do a flipNM on it. If the  */
/* flip is successful, then try to do another flipNM on a.b.  If one of the  */
/* two flipNMs fails, restore the old tets as they have never been flipped.  */
/* Then try the next face a.b.p_1.  The process can be looped for all faces  */
/* having ab. Stop if ab is removed or all faces have been visited. Note in  */
/* the above description only b.p_0 is considered, a.p_0 is done by swapping */
/* the position of a and b.                                                  */
/*                                                                           */
/* Similar operations have been described in [Joe,1995].  My approach checks */
/* more cases for finding flips than Joe's.  For instance, the cases (1)-(7) */
/* of Joe only consider abf for finding a flip (T23/T32).  My approach looks */
/* all faces at ab for finding flips. Moreover, the flipNM can flip an edge  */
/* whose star may have more than 3 tets while Joe's only works on 3-tet case.*/
/*                                                                           */
/* If ab is removed, 'newtetlist' contains the new tets. Two sets 'abtetlist'*/
/* (n tets) and 'bftetlist' (n1 tets) have been replaced.  The number of new */
/* tets can be calculated by follows: the 1st flip transforms n1 tets into   */
/* (n1 - 2) * 2 new tets, however,one of the new tets goes into the new link */
/* of ab, i.e., the reduced tet number in Star(ab) is n - 1;  the 2nd flip   */
/* transforms n - 1 tets into (n - 3) * 2 new tets. Hence the number of new  */
/* tets are: m = ((n1 - 2) * 2 - 1) + (n - 3) * 2.  The old tets are NOT del-*/
/* eted. The caller has the right to delete them or reverse the operation.   */
/* tetgenmesh::removeedgebycombNM() */
PetscErrorCode TetGenMeshRemoveEdgeByCombNM(TetGenMesh *m, PetscReal *key, PetscInt n, triface *abtetlist, PetscInt *n1, triface *bftetlist, triface *newtetlist, Queue *flipque, PetscBool *isCombined)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  triface tmpabtetlist[21];
  triface newfront = {PETSC_NULL, 0, 0}, oldfront = {PETSC_NULL, 0, 0}, adjfront = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0};
  point pa, pb, p[21];
  PetscReal ori, tmpkey, tmpkey2;
  PetscReal attrib, volume;
  PetscBool doflip, success;
  PetscInt twice, count;
  PetscInt i, j, k, m1;
  long bakflipcount; /* Used for elemfliplist. */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Maximal 20 tets in Star(ab). */
  if (n >= 20) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This routine cannot handle %d > 20 tets", n);

  /* Do the following procedure twice, one for flipping edge b.p_0 and the */
  /*   other for p_0.a which is symmetric to the first. */
  twice = 0;
  do {
    /* Two points a and b are fixed. */
    pa = org(&abtetlist[0]);
    pb = dest(&abtetlist[0]);
    /* The points p_0, ..., p_n-1 are permuted in the following loop. */
    for(i = 0; i < n; i++) {
      /* Get the n points for the current configuration. */
      for(j = 0; j < n; j++) {
        p[j] = apex(&abtetlist[(i + j) % n]);
      }
      /* Check if b.p_0 is of type N32 or N44. */
      ori = TetGenOrient3D(pb, p[0], p[1], p[n - 1]);
      if ((ori > 0) && key) {
        /* b.p_0 is not N32. However, it is possible that the tet b.p_0.p_1. */
        /*   p_n-1 has worse quality value than the key. In such case, also */
        /*   try to flip b.p_0. */
        ierr = TetGenMeshTetAllDihedral(m, pb, p[0], p[n - 1], p[1], PETSC_NULL, &tmpkey, PETSC_NULL);CHKERRQ(ierr);
        if (tmpkey < *key) ori = 0.0;
      }
      if (m->fixededgelist && (ori <= 0.0)) {
        PetscBool isFixed;
        /* b.p_0 is either N32 or N44. Do not flip a fixed edge. */
        ierr = TetGenMeshCheck4FixedEdge(m, pb, p[0], &isFixed);CHKERRQ(ierr);
        if (isFixed) {
          ori = 1.0; /* Do not flip this edge. Skip it. */
        }
      }
      if (ori <= 0.0) {
        /* b.p_0 is either N32 or N44. Try the 1st flipNM. */
        bftetlist[0] = abtetlist[i];
        enextself(&bftetlist[0]);/* go to edge b.p_0. */
        adjustedgering_triface(&bftetlist[0], CW); /* edge p_0.b. */
        if (apex(&bftetlist[0]) != pa) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        /* Form Star(b.p_0). */
        doflip = PETSC_TRUE;
        *n1 = 0;
        do {
          /* Is the list full? */
          if (*n1 == 20) break;
          if (m->checksubfaces) {
            /* Stop if a subface appears. */
            tspivot(m, &bftetlist[*n1], &checksh);
            if (checksh.sh != m->dummysh) {
              doflip = PETSC_FALSE; break;
            }
          }
          /* Get the next tet at p_0.b. */
          if (!fnext(m, &bftetlist[*n1], &bftetlist[(*n1) + 1])) {
            /* Meet a boundary face. Do not flip. */
            doflip = PETSC_FALSE; break;
          }
          (*n1)++;
        } while (apex(&bftetlist[*n1]) != pa);
        /* 2 < n1 <= b->maxflipedgelinksize. */
        if (doflip) {
          success = PETSC_FALSE;
          tmpkey = -1.0;  /* = acos(pi). */
          if (key) tmpkey = *key;
          m1 = 0;
          if (*n1 == 3) {
            /* Three tets case. Try flip32. */
            ierr = TetGenMeshRemoveEdgeByFlip32(m, &tmpkey, bftetlist, newtetlist, flipque, &success);CHKERRQ(ierr);
            m1 = 2;
          } else if ((*n1 > 3) && (*n1 <= b->maxflipedgelinksize)) {
            /* Four or more tets case. Try flipNM. */
            ierr = TetGenMeshRemoveEdgeByTranNM(m, &tmpkey, *n1, bftetlist, newtetlist, p[1], p[n - 1], flipque, &success);CHKERRQ(ierr);
            /* If success, the number of new tets. */
            m1 = ((*n1) - 2) * 2;
          } else {
            PetscInfo1(b->in, "  !! Unhandled case: n1 = %d.\n", *n1);
          }
          if (success) {
            /* b.p_0 is flipped. The link of ab is reduced (by 1), i.e., p_0 */
            /*   is not on the link of ab. Two old tets a.b.p_0.p_n-1 and */
            /*   a.b.p_1.p_0 have been removed from the Star(ab) and one new */
            /*   tet t = a.b.p_1.p_n-1 belongs to Star(ab).  */
            /* Find t in the 'newtetlist' and remove it from the list. */
            setpointmark(m, pa, -pointmark(m, pa) - 1);
            setpointmark(m, pb, -pointmark(m, pb) - 1);
            if (m1 <= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            for(j = 0; j < m1; j++) {
              tmpabtetlist[0] = newtetlist[j];
              /* Does it has ab? */
              count = 0;
              for(k = 0; k < 4; k++) {
                if (pointmark(m, (point) (tmpabtetlist[0].tet[4+k])) < 0) count++;
              }
              if (count == 2) {
                /* It is. Adjust t to be the edge ab. */
                for(tmpabtetlist[0].loc = 0; tmpabtetlist[0].loc < 4; tmpabtetlist[0].loc++) {
                  if ((oppo(&tmpabtetlist[0]) != pa) && (oppo(&tmpabtetlist[0]) != pb)) break;
                }
                /* The face of t must contain ab. */
                if (tmpabtetlist[0].loc >= 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                ierr = TetGenMeshFindEdge_triface(m, &(tmpabtetlist[0]), pa, pb);CHKERRQ(ierr);
                break;
              }
            }
            if (j >= m1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong"); /* The tet must exist. */
            /* Remove t from list. Fill t's position by the last tet. */
            newtetlist[j] = newtetlist[m1 - 1];
            setpointmark(m, pa, -(pointmark(m, pa) + 1));
            setpointmark(m, pb, -(pointmark(m, pb) + 1));
            /* Create the temporary Star(ab) for the next flipNM. */
            adjustedgering_triface(&tmpabtetlist[0], CCW);
            if (org(&tmpabtetlist[0]) != pa) {
              fnextself(m, &tmpabtetlist[0]);
              esymself(&tmpabtetlist[0]);
            }
            /* SELF_CHECK: Make sure current edge is a->b. */
            if (org(&tmpabtetlist[0]) != pa) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            if (dest(&tmpabtetlist[0]) != pb) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            if (apex(&tmpabtetlist[0]) != p[n - 1]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            if (oppo(&tmpabtetlist[0]) != p[1]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            /* There are n - 2 left temporary tets. */
            for(j = 1; j < n - 1; j++) {
              ierr = TetGenMeshMakeTetrahedron(m, &(tmpabtetlist[j]));CHKERRQ(ierr);
              setorg(&tmpabtetlist[j], pa);
              setdest(&tmpabtetlist[j], pb);
              setapex(&tmpabtetlist[j], p[j]);
              setoppo(&tmpabtetlist[j], p[j + 1]);
            }
            /* Transfer the element attributes. */
            for(j = 0; j < in->numberoftetrahedronattributes; j++) {
              attrib = elemattribute(m, abtetlist[0].tet, j);
              for(k = 0; k < n - 1; k++) {
                setelemattribute(m, tmpabtetlist[k].tet, j, attrib);
              }
            }
            /* Transfer the volume constraints. */
            if (b->varvolume && !b->refine) {
              volume = volumebound(m, abtetlist[0].tet);
              for (k = 0; k < n - 1; k++) {
                setvolumebound(m, tmpabtetlist[k].tet, volume);
              }
            }
            /* Glue n - 1 internal faces of Star(ab). */
            for(j = 0; j < n - 1; j++) {
              fnext(m, &tmpabtetlist[j], &newfront);
              bond(m, &newfront, &tmpabtetlist[(j + 1) % (n - 1)]); /* a.b.p_j+1 */
            }
            /* Substitute the old tets with the new tets by connecting the */
            /*   new tets to the adjacent tets in the mesh. There are (n-2) */
            /*   * 2 (outer) faces of the new tets need to be operated. */
            /* Note that the old tets still have the pointers to their */
            /*   adjacent tets in the mesh.  These pointers can be re-used */
            /*   to inverse the substitution. */
            for(j = 2; j < n; j++) {
              /* Get an old tet: [j]a.b.p_j.p_j-1, (j > 1). */
              oldfront = abtetlist[(i + j) % n];
              esymself(&oldfront);
              enextfnextself(m, &oldfront);
              /* Get an adjacent tet at face: [j]a.p_j.p_j-1. */
              sym(&oldfront, &adjfront); /* adjfront may be dummy. */
              /* Get the corresponding face from the new tets. */
              /* j >= 2. */
              enext2fnext(m, &tmpabtetlist[j - 1], &newfront); /* a.p_j.p_j-1 */
              bond(m, &newfront, &adjfront);
              if (m->checksubfaces) {
                tspivot(m, &oldfront, &checksh);
                if (checksh.sh != m->dummysh) {
                  tsbond(m, &newfront, &checksh);
                }
              }
            }
            for(j = 2; j < n; j++) {
              /* Get an old tet: [j]a.b.p_j.p_j-1, (j > 2). */
              oldfront = abtetlist[(i + j) % n];
              esymself(&oldfront);
              enext2fnextself(m, &oldfront);
              /* Get an adjacent tet at face: [j]b.p_j.p_j-1. */
              sym(&oldfront, &adjfront); /* adjfront may be dummy. */
              /* Get the corresponding face from the new tets. */
              /* j >= 2. */
              enextfnext(m, &tmpabtetlist[j - 1], &newfront); /* b.p_j.p_j-1 */
              bond(m, &newfront, &adjfront);
              if (m->checksubfaces) {
                tspivot(m, &oldfront, &checksh);
                if (checksh.sh != m->dummysh) {
                  tsbond(m, &newfront, &checksh);
                }
              }
            }
            /* Adjust the faces in the temporary new tets at ab for */
            /*   recursively processing on the n-1 tets. */
            for(j = 0; j < n - 1; j++) {
              fnextself(m, &tmpabtetlist[j]);
            }
            tmpkey2 = -1;
            if (key) tmpkey2 = *key;
            if (m->elemfliplist) {
              /* Remember the current registered flips. */
              bakflipcount = m->elemfliplist->objects;
            }
            if ((n - 1) == 3) {
              ierr = TetGenMeshRemoveEdgeByFlip32(m, &tmpkey2, tmpabtetlist, &(newtetlist[m1 - 1]), flipque, &success);CHKERRQ(ierr);
            } else { /* assert((n - 1) >= 4); */
              ierr = TetGenMeshRemoveEdgeByTranNM(m, &tmpkey2, n - 1, tmpabtetlist, &(newtetlist[m1 - 1]), PETSC_NULL, PETSC_NULL, flipque, &success);CHKERRQ(ierr);
            }
            /* No matter it was success or not, delete the temporary tets. */
            for(j = 0; j < n - 1; j++) {
              ierr = TetGenMeshTetrahedronDealloc(m, tmpabtetlist[j].tet);CHKERRQ(ierr);
            }
            if (success) {
              /* The new configuration is good.  */
              /* Do not delete the old tets. */
              /* for (j = 0; j < n; j++) { */
              /*   tetrahedrondealloc(abtetlist[j].tet); */
              /* } */
              /* Return the bigger dihedral in the two sets of new tets. */
              if (key) {
                *key = tmpkey2 < tmpkey ? tmpkey2 : tmpkey;
              }
              if (isCombined) {*isCombined = PETSC_TRUE;}
              PetscFunctionReturn(0);
            } else {
              /* The new configuration is bad, substitue back the old tets. */
              if (m->elemfliplist) {
                /* Restore the registered flips. */
                m->elemfliplist->objects = bakflipcount;
              }
              for(j = 0; j < n; j++) {
                oldfront = abtetlist[(i + j) % n];
                esymself(&oldfront);
                enextfnextself(m, &oldfront); /* [0]a.p_0.p_n-1, [j]a.p_j.p_j-1. */
                sym(&oldfront, &adjfront); /* adjfront may be dummy. */
                bond(m, &oldfront, &adjfront);
                if (m->checksubfaces) {
                  tspivot(m, &oldfront, &checksh);
                  if (checksh.sh != m->dummysh) {
                    tsbond(m, &oldfront, &checksh);
                  }
                }
              }
              for(j = 0; j < n; j++) {
                oldfront = abtetlist[(i + j) % n];
                esymself(&oldfront);
                enext2fnextself(m, &oldfront); /* [0]b.p_0.p_n-1, [j]b.p_j.p_j-1. */
                sym(&oldfront, &adjfront); /* adjfront may be dummy */
                bond(m, &oldfront, &adjfront);
                if (m->checksubfaces) {
                  tspivot(m, &oldfront, &checksh);
                  if (checksh.sh != m->dummysh) {
                    tsbond(m, &oldfront, &checksh);
                  }
                }
              }
              /* Substitute back the old tets of the first flip. */
              for(j = 0; j < *n1; j++) {
                oldfront = bftetlist[j];
                esymself(&oldfront);
                enextfnextself(m, &oldfront);
                sym(&oldfront, &adjfront); /* adjfront may be dummy. */
                bond(m, &oldfront, &adjfront);
                if (m->checksubfaces) {
                  tspivot(m, &oldfront, &checksh);
                  if (checksh.sh != m->dummysh) {
                    tsbond(m, &oldfront, &checksh);
                  }
                }
              }
              for(j = 0; j < *n1; j++) {
                oldfront = bftetlist[j];
                esymself(&oldfront);
                enext2fnextself(m, &oldfront); /* [0]b.p_0.p_n-1, [j]b.p_j.p_j-1. */
                sym(&oldfront, &adjfront); /* adjfront may be dummy */
                bond(m, &oldfront, &adjfront);
                if (m->checksubfaces) {
                  tspivot(m, &oldfront, &checksh);
                  if (checksh.sh != m->dummysh) {
                    tsbond(m, &oldfront, &checksh);
                  }
                }
              }
              /* Delete the new tets of the first flip. Note that one new */
              /*   tet has already been removed from the list. */
              for(j = 0; j < m1 - 1; j++) {
                ierr = TetGenMeshTetrahedronDealloc(m, newtetlist[j].tet);CHKERRQ(ierr);
              }
            } /* if (success) */
          } /* if (success) */
        } /* if (doflip) */
      } /* if (ori <= 0.0) */
    } /* for (i = 0; i < n; i++) */
    /* Inverse a and b and the tets configuration. */
    for(i = 0; i < n; i++) newtetlist[i] = abtetlist[i];
    for(i = 0; i < n; i++) {
      oldfront = newtetlist[n - i - 1];
      esymself(&oldfront);
      fnextself(m, &oldfront);
      abtetlist[i] = oldfront;
    }
    twice++;
  } while (twice < 2);

  if (isCombined) {*isCombined = PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSplitSubEdge_queue"
/*  splitsubedge()    Insert a point on an edge of the surface mesh.           */
/*                                                                             */
/*  The splitting edge is given by 'splitsh'. Assume its three corners are a,  */
/*  b, c, where ab is the edge will be split. ab may be a subsegment.          */
/*                                                                             */
/*  To split edge ab is to split all subfaces conatining ab. If ab is not a    */
/*  subsegment, there are only two subfaces need be split, otherwise, there    */
/*  may have any number of subfaces need be split. Each splitting subface abc  */
/*  is shrunk to avc, a new subface vbc is created.  It is important to keep   */
/*  the orientations of edge rings of avc and vbc be the same as abc's. If ab  */
/*  is a subsegment, it is shrunk to av and a new subsegment vb is created.    */
/*                                                                             */
/*  If there are tetrahedra adjoining to the splitting subfaces, they should   */
/*  be split before calling this routine, so the connection between the new    */
/*  tetrahedra and the new subfaces can be correctly set.                      */
/*                                                                             */
/*  On completion, 'splitsh' returns avc.  If 'flipqueue' is not NULL, it      */
/*  returns all edges which may be non-Delaunay.                               */
/* tetgenmesh::splitsubedge() */
PetscErrorCode TetGenMeshSplitSubEdge_queue(TetGenMesh *m, point newpoint, face *splitsh, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  triface abcd = {PETSC_NULL, 0, 0}, bace = {PETSC_NULL, 0, 0}, vbcd = {PETSC_NULL, 0, 0}, bvce = {PETSC_NULL, 0, 0};
  face startabc = {PETSC_NULL, 0}, spinabc = {PETSC_NULL, 0}, spinsh = {PETSC_NULL, 0};
  face oldbc = {PETSC_NULL, 0}, bccasin = {PETSC_NULL, 0}, bccasout = {PETSC_NULL, 0};
  face ab = {PETSC_NULL, 0}, bc = {PETSC_NULL, 0};
  face avc = {PETSC_NULL, 0}, vbc = {PETSC_NULL, 0}, vbc1 = {PETSC_NULL, 0};
  face av = {PETSC_NULL, 0}, vb = {PETSC_NULL, 0};
  point pa, pb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  startabc = *splitsh;
  /*  Is there a subsegment? */
  sspivot(m, &startabc, &ab);
  if (ab.sh != m->dummysh) {
    ab.shver = 0;
    if (sorg(&startabc) != sorg(&ab)) {
      sesymself(&startabc);
    }
  }
  pa = sorg(&startabc);
  pb = sdest(&startabc);

  PetscInfo4(b->in, "  Inserting point %d on subedge (%d, %d) %s.\n", pointmark(m, newpoint), pointmark(m, pa), pointmark(m, pb), (ab.sh != m->dummysh ? "(seg)" : " "));

  /*  Spin arround ab, split every subface containing ab. */
  spinabc = startabc;
  do {
    /*  Adjust spinabc be edge ab. */
    if (sorg(&spinabc) != pa) {
      sesymself(&spinabc);
    }
    /*  Unmark the face for splitting (used for refinement) 2009-08-17. */
    sunmarktest(&spinabc);
    /*  Save old configuration at edge bc, if bc has a subsegment, save the */
    /*    face link of it and dissolve it from bc. */
    senext(&spinabc, &oldbc);
    spivot(&oldbc, &bccasout);
    sspivot(m, &oldbc, &bc);
    if (bc.sh != m->dummysh) {
      if (bccasout.sh != m->dummysh) {
        /*  'spinabc' is not self-bonded. */
        spinsh = bccasout;
        do {
          bccasin = spinsh;
          spivotself(&spinsh);
        } while (spinsh.sh != oldbc.sh);
      } else {
        bccasout.sh = m->dummysh;
      }
      ssdissolve(m, &oldbc);
    }
    /*  Create a new subface. */
    ierr = TetGenMeshMakeShellFace(m, m->subfaces, &vbc);CHKERRQ(ierr);
    /*  Split abc. */
    avc = spinabc;  /*  Update 'abc' to 'avc'. */
    setsdest(&avc, newpoint);
    /*  Make 'vbc' be in the same edge ring as 'avc'. */
    vbc.shver = avc.shver;
    setsorg(&vbc, newpoint); /*  Set 'vbc'. */
    setsdest(&vbc, pb);
    setsapex(&vbc, sapex(&avc));
    if (b->quality && m->varconstraint) {
      /*  Copy the area bound into the new subface. */
      setareabound(m, &vbc, areabound(m, &avc));
    }
    /*  Copy the shell marker and shell type into the new subface. */
    setshellmark(m, &vbc, shellmark(m, &avc));
    setshelltype(m, &vbc, shelltype(m, &avc));
    if (m->checkpbcs) {
      /*  Copy the pbcgroup into the new subface. */
      setshellpbcgroup(m, &vbc, shellpbcgroup(m, &avc));
    }
    /*  Set the connection between updated and new subfaces. */
    senext2self(&vbc);
    sbond(&vbc, &oldbc);
    /*  Set the connection between new subface and casings. */
    senext2self(&vbc);
    if (bc.sh != m->dummysh) {
      if (bccasout.sh != m->dummysh) {
        /*  Insert 'vbc' into face link. */
        sbond1(&bccasin, &vbc);
        sbond1(&vbc, &bccasout);
      } else {
        /*  Bond 'vbc' to itself. */
        sdissolve(m, &vbc); /*  sbond(vbc, vbc); */
      }
      ssbond(m, &vbc, &bc);
    } else {
      sbond(&vbc, &bccasout);
    }
    /*  Go to next subface at edge ab. */
    spivotself(&spinabc);
    if (spinabc.sh == m->dummysh) {
      break; /*  'ab' is a hull edge. */
    }
  } while (spinabc.sh != startabc.sh);

  /*  Get the new subface vbc above the updated subface avc (= startabc). */
  senext(&startabc, &oldbc);
  spivot(&oldbc, &vbc);
  if (sorg(&vbc) == newpoint) {
    sesymself(&vbc);
  }
#ifdef PETSC_USE_DEBUG
  if (sorg(&vbc) != sdest(&oldbc) || sdest(&vbc) != sorg(&oldbc)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
  senextself(&vbc);
  /*  Set the face link for the new created subfaces around edge vb. */
  spinabc = startabc;
  do {
    /*  Go to the next subface at edge av. */
    spivotself(&spinabc);
    if (spinabc.sh == m->dummysh) {
      break; /*  'ab' is a hull edge. */
    }
    if (sorg(&spinabc) != pa) {
      sesymself(&spinabc);
    }
    /*  Get the new subface vbc1 above the updated subface avc (= spinabc). */
    senext(&spinabc, &oldbc);
    spivot(&oldbc, &vbc1);
    if (sorg(&vbc1) == newpoint) {
      sesymself(&vbc1);
    }
#ifdef PETSC_USE_DEBUG
    if (sorg(&vbc1) != sdest(&oldbc) || sdest(&vbc1) != sorg(&oldbc)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
    senextself(&vbc1);
    /*  Set the connection: vbc->vbc1. */
    sbond1(&vbc, &vbc1);
    /*  For the next connection. */
    vbc = vbc1;
  } while (spinabc.sh != startabc.sh);

  /*  Split ab if it is a subsegment. */
  if (ab.sh != m->dummysh) {
    /*  Unmark the segment for mesh optimization. 2009-08-17. */
    sunmarktest(&ab);
    /*  Update subsegment ab to av. */
    av = ab;
    setsdest(&av, newpoint);
    /*  Create a new subsegment vb. */
    ierr = TetGenMeshMakeShellFace(m, m->subsegs, &vb);CHKERRQ(ierr);
    setsorg(&vb, newpoint);
    setsdest(&vb, pb);
    /*  vb gets the same mark and segment type as av. */
    setshellmark(m, &vb, shellmark(m, &av));
    setshelltype(m, &vb, shelltype(m, &av));
    if (b->quality && m->varconstraint) {
      /*  Copy the area bound into the new subsegment. */
      setareabound(m, &vb, areabound(m, &av));
    }
    /*  Save the old connection at ab (re-use the handles oldbc, bccasout). */
    senext(&av, &oldbc);
    spivot(&oldbc, &bccasout);
    /*  Bond av and vb (bonded at their "fake" edges). */
    senext2(&vb, &bccasin);
    sbond(&bccasin, &oldbc);
    if (bccasout.sh != m->dummysh) {
      /*  There is a subsegment connecting with ab at b. It will connect */
      /*    to vb at b after splitting. */
      bccasout.shver = 0;
      if (sorg(&bccasout) != pb) sesymself(&bccasout);
#ifdef PETSC_USE_DEBUG
      if (sorg(&bccasout) != pb) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
      senext2self(&bccasout);
      senext(&vb, &bccasin);
      sbond(&bccasin, &bccasout);
    }
    /*  Bond all new subfaces (vbc) to vb. */
    spinabc = startabc;
    do {
      /*  Adjust spinabc be edge av. */
      if (sorg(&spinabc) != pa) {
        sesymself(&spinabc);
      }
      /*  Get new subface vbc above the updated subface avc (= spinabc). */
      senext(&spinabc, &oldbc);
      spivot(&oldbc, &vbc);
      if (sorg(&vbc) == newpoint) {
        sesymself(&vbc);
      }
      senextself(&vbc);
      /*  Bond the new subface and the new subsegment. */
      ssbond(m, &vbc, &vb);
      /*  Go to the next. */
      spivotself(&spinabc);
      if (spinabc.sh == m->dummysh) {
        break; /*  There's only one facet at the segment.rr */
      }
    } while (spinabc.sh != startabc.sh);
  }

  /*  Bond the new subfaces to new tetrahedra if they exist.  New tetrahedra */
  /*    should have been created before calling this routine. */
  spinabc = startabc;
  do {
    /*  Adjust spinabc be edge av. */
    if (sorg(&spinabc) != pa) {
      sesymself(&spinabc);
    }
    /*  Get new subface vbc above the updated subface avc (= spinabc). */
    senext(&spinabc, &oldbc);
    spivot(&oldbc, &vbc);
    if (sorg(&vbc) == newpoint) {
      sesymself(&vbc);
    }
    senextself(&vbc);
    /*  Get the adjacent tetrahedra at 'spinabc'. */
    stpivot(m, &spinabc, &abcd);
    if (abcd.tet != m->dummytet) {
      ierr = TetGenMeshFindEdge_triface(m, &abcd, sorg(&spinabc), sdest(&spinabc));CHKERRQ(ierr);
      enextfnext(m, &abcd, &vbcd);
      fnextself(m, &vbcd);
#ifdef PETSC_USE_DEBUG
      if (vbcd.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
      tsbond(m, &vbcd, &vbc);
      sym(&vbcd, &bvce);
      sesymself(&vbc);
      tsbond(m, &bvce, &vbc);
    } else {
      /*  One side is empty, check the other side. */
      sesymself(&spinabc);
      stpivot(m, &spinabc, &bace);
      if (bace.tet != m->dummytet) {
        ierr = TetGenMeshFindEdge_triface(m, &bace, sorg(&spinabc), sdest(&spinabc));CHKERRQ(ierr);
        enext2fnext(m, &bace, &bvce);
        fnextself(m, &bvce);
#ifdef PETSC_USE_DEBUG
        if (bvce.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
        sesymself(&vbc);
        tsbond(m, &bvce, &vbc);
      }
    }
    /*  Go to the next. */
    spivotself(&spinabc);
    if (spinabc.sh == m->dummysh) {
      break; /*  'ab' is a hull edge. */
    }
  } while (spinabc.sh != startabc.sh);

  if (b->verbose > 3) {
    spinabc = startabc;
    do {
      /*  Adjust spinabc be edge av. */
      if (sorg(&spinabc) != pa) {
        sesymself(&spinabc);
      }
      PetscInfo(b->in, "    Updating abc:\n");
      ierr = TetGenMeshPrintSh(m, &spinabc, PETSC_FALSE);CHKERRQ(ierr);
      /*  Get new subface vbc above the updated subface avc (= spinabc). */
      senext(&spinabc, &oldbc);
      spivot(&oldbc, &vbc);
      if (sorg(&vbc) == newpoint) {
        sesymself(&vbc);
      }
      senextself(&vbc);
      PetscInfo(b->in, "    Creating vbc:\n");
      ierr = TetGenMeshPrintSh(m, &vbc, PETSC_FALSE);CHKERRQ(ierr);
      /*  Go to the next. */
      spivotself(&spinabc);
      if (spinabc.sh == m->dummysh) {
        break; /*  'ab' is a hull edge. */
      }
    } while (spinabc.sh != startabc.sh);
  }

  if (flipqueue) {
    spinabc = startabc;
    do {
      /*  Adjust spinabc be edge av. */
      if (sorg(&spinabc) != pa) {
        sesymself(&spinabc);
      }
      senext2(&spinabc, &oldbc); /*  Re-use oldbc. */
      ierr = TetGenMeshEnqueueFlipEdge(m, &oldbc, flipqueue);CHKERRQ(ierr);
      /*  Get new subface vbc above the updated subface avc (= spinabc). */
      senext(&spinabc, &oldbc);
      spivot(&oldbc, &vbc);
      if (sorg(&vbc) == newpoint) {
        sesymself(&vbc);
      }
      senextself(&vbc);
      senext(&vbc, &oldbc); /*  Re-use oldbc. */
      ierr = TetGenMeshEnqueueFlipEdge(m, &oldbc, flipqueue);CHKERRQ(ierr);
      /*  Go to the next. */
      spivotself(&spinabc);
      if (spinabc.sh == m->dummysh) {
        break; /*  'ab' is a hull edge. */
      }
    } while (spinabc.sh != startabc.sh);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSplitTetEdge"
/*  splittetedge()    Insert a point on an edge of the mesh.                   */
/*                                                                             */
/*  The edge is given by 'splittet'. Assume its four corners are a, b, n1 and  */
/*  n2, where ab is the edge will be split. Around ab may exist any number of  */
/*  tetrahedra. For convenience, they're ordered in a sequence following the   */
/*  right-hand rule with your thumb points from a to b. Let the vertex set of  */
/*  these tetrahedra be {a, b, n1, n2, ..., n(i)}. NOTE the tetrahedra around  */
/*  ab may not connect to each other (can only happen when ab is a subsegment, */
/*  hence some faces abn(i) are subfaces).  If ab is a subsegment, abn1 must   */
/*  be a subface.                                                              */
/*                                                                             */
/*  To split edge ab by a point v is to split all tetrahedra containing ab by  */
/*  v.  More specifically, for each such tetrahedron, an1n2b, it is shrunk to  */
/*  an1n2v, and a new tetrahedra bn2n1v is created. If ab is a subsegment, or  */
/*  some faces of the splitting tetrahedra are subfaces, they must be split    */
/*  either by calling routine 'splitsubedge()'.                                */
/*                                                                             */
/*  On completion, 'splittet' returns avn1n2.  If 'flipqueue' is not NULL, it  */
/*  returns all faces which may become non-Delaunay after this operation.      */
/* tetgenmesh::splittetedge() */
PetscErrorCode TetGenMeshSplitTetEdge(TetGenMesh *m, point newpoint, triface *splittet, Queue *flipqueue, PetscBool *isSplit)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  triface *bots, *newtops;
  triface oldtop = {PETSC_NULL, 0, 0}, topcasing = {PETSC_NULL, 0, 0};
  triface spintet = {PETSC_NULL, 0, 0}, tmpbond0 = {PETSC_NULL, 0, 0}, tmpbond1 = {PETSC_NULL, 0, 0};
  face abseg = {PETSC_NULL, 0}, splitsh = {PETSC_NULL, 0}, topsh = {PETSC_NULL, 0}, spinsh = {PETSC_NULL, 0};
  triface worktet = {PETSC_NULL, 0, 0};
  face n1n2seg = {PETSC_NULL, 0}, n2vseg = {PETSC_NULL, 0}, n1vseg = {PETSC_NULL, 0};
  point pa, pb, n1, n2;
  PetscReal attrib, volume;
  int wrapcount, hitbdry;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (m->checksubfaces) {
    /*  Is there a subsegment need to be split together? */
    ierr = TetGenMeshTssPivot(m, splittet, &abseg);CHKERRQ(ierr);
    if (abseg.sh != m->dummysh) {
      abseg.shver = 0;
      /*  Orient the edge direction of 'splittet' be abseg. */
      if (org(splittet) != sorg(&abseg)) {
        esymself(splittet);
      }
    }
  }
  spintet = *splittet;
  pa = org(&spintet);
  pb = dest(&spintet);

  PetscInfo3(b->in, "  Inserting point %d on edge (%d, %d).\n", pointmark(m, newpoint), pointmark(m, pa), pointmark(m, pb));

  /*  Collect the tetrahedra containing the splitting edge (ab). */
  n1 = apex(&spintet);
  hitbdry = 0;
  wrapcount = 1;
  if (m->checksubfaces && abseg.sh != m->dummysh) {
    /*  It may happen that some tetrahedra containing ab (a subsegment) are */
    /*    completely disconnected with others. If it happens, use the face */
    /*    link of ab to cross the boundary. */
    while(1) {
      if (!fnextself(m, &spintet)) {
        /*  Meet a boundary, walk through it. */
        hitbdry ++;
        tspivot(m, &spintet, &spinsh);
#ifdef PETSC_USE_DEBUG
        if (spinsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Shell edge should not be null");
#endif
        ierr = TetGenMeshFindEdge_face(m, &spinsh, pa, pb);CHKERRQ(ierr);
        sfnextself(m, &spinsh);
        stpivot(m, &spinsh, &spintet);
#ifdef PETSC_USE_DEBUG
        if (spintet.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Tet should not be null");
#endif
        ierr = TetGenMeshFindEdge_triface(m, &spintet, pa, pb);CHKERRQ(ierr);
        /*  Remember this position (hull face) in 'splittet'. */
        *splittet = spintet;
        /*  Split two hull faces increase the hull size; */
        m->hullsize += 2;
      }
      if (apex(&spintet) == n1) break;
      wrapcount ++;
    }
    if (hitbdry > 0) {
      wrapcount -= hitbdry;
    }
  } else {
    /*  All the tetrahedra containing ab are connected together. If there */
    /*    are subfaces, 'splitsh' keeps one of them. */
    splitsh.sh = m->dummysh;
    while (hitbdry < 2) {
      if (m->checksubfaces && splitsh.sh == m->dummysh) {
        tspivot(m, &spintet, &splitsh);
      }
      if (fnextself(m, &spintet)) {
        if (apex(&spintet) == n1) break;
        wrapcount++;
      } else {
        hitbdry ++;
        if (hitbdry < 2) {
          esym(splittet, &spintet);
        }
      }
    }
    if (hitbdry > 0) {
      /*  ab is on the hull. */
      wrapcount -= 1;
      /*  'spintet' now is a hull face, inverse its edge direction. */
      esym(&spintet, splittet);
      /*  Split two hull faces increases the number of hull faces. */
      m->hullsize += 2;
    }
  }

  /*  Make arrays of updating (bot, oldtop) and new (newtop) tetrahedra. */
  ierr = PetscMalloc2(wrapcount,triface,&bots,wrapcount,triface,&newtops);CHKERRQ(ierr);
  /*  Spin around ab, gather tetrahedra and set up new tetrahedra. */
  spintet = *splittet;
  for (i = 0; i < wrapcount; i++) {
    /*  Get 'bots[i] = an1n2b'. */
    enext2fnext(m, &spintet, &bots[i]);
    esymself(&bots[i]);
    /*  Create 'newtops[i]'. */
    ierr = TetGenMeshMakeTetrahedron(m, &(newtops[i]));CHKERRQ(ierr);
    /*  Go to the next. */
    fnextself(m, &spintet);
    if (m->checksubfaces && abseg.sh != m->dummysh) {
      if (!issymexist(m, &spintet)) {
        /*  We meet a hull face, walk through it. */
        tspivot(m, &spintet, &spinsh);
        if (spinsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        ierr = TetGenMeshFindEdge_face(m, &spinsh, pa, pb);CHKERRQ(ierr);
        sfnextself(m, &spinsh);
        stpivot(m, &spinsh, &spintet);
        if (spintet.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        ierr = TetGenMeshFindEdge_triface(m, &spintet, pa, pb);CHKERRQ(ierr);
      }
    }
  }

  /*  Set the vertices of updated and new tetrahedra. */
  for(i = 0; i < wrapcount; i++) {
    /*  Update 'bots[i] = an1n2v'. */
    setoppo(&bots[i], newpoint);
    /*  Set 'newtops[i] = bn2n1v'. */
    n1 = dest(&bots[i]);
    n2 = apex(&bots[i]);
    /*  Set 'newtops[i]'. */
    setorg(&newtops[i], pb);
    setdest(&newtops[i], n2);
    setapex(&newtops[i], n1);
    setoppo(&newtops[i], newpoint);
    /*  Set the element attributes of a new tetrahedron. */
    for(j = 0; j < in->numberoftetrahedronattributes; j++) {
      attrib = elemattribute(m, bots[i].tet, j);
      setelemattribute(m, newtops[i].tet, j, attrib);
    }
    if (b->varvolume) {
      /*  Set the area constraint of a new tetrahedron. */
      volume = volumebound(m, bots[i].tet);
      setvolumebound(m, newtops[i].tet, volume);
    }
    /*  Make sure no inversed tetrahedron has been created. */
    volume = TetGenOrient3D(pa, n1, n2, newpoint);
    if (volume >= 0.0) {
      /* printf("Internal error in splittetedge(): volume = %.12g.\n", volume); */
      break;
    }
    volume = TetGenOrient3D(pb, n2, n1, newpoint);
    if (volume >= 0.0) {
      /* printf("Internal error in splittetedge(): volume = %.12g.\n", volume); */
      break;
    }
  }

  if (i < wrapcount) {
    /*  Do not insert this point. It will result inverted or degenerated tet. */
    /*  Restore have updated tets in "bots". */
    for(; i >= 0; i--) {
      setoppo(&bots[i], pb);
    }
    /*  Deallocate tets in "newtops". */
    for (i = 0; i < wrapcount; i++) {
      ierr = TetGenMeshTetrahedronDealloc(m, newtops[i].tet);CHKERRQ(ierr);
    }
    ierr = PetscFree2(bots,newtops);CHKERRQ(ierr);
    if (isSplit) {*isSplit = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  /*  Bond newtops to topcasings and bots. */
  for (i = 0; i < wrapcount; i++) {
    /*  Get 'oldtop = n1n2va' from 'bots[i]'. */
    enextfnext(m, &bots[i], &oldtop);
    sym(&oldtop, &topcasing);
    bond(m, &newtops[i], &topcasing);
    if (m->checksubfaces) {
      tspivot(m, &oldtop, &topsh);
      if (topsh.sh != m->dummysh) {
        tsdissolve(m, &oldtop);
        tsbond(m, &newtops[i], &topsh);
      }
    }
    enextfnext(m, &newtops[i], &tmpbond0);
    bond(m, &oldtop, &tmpbond0);
  }
  /*  Bond between newtops. */
  fnext(m, &newtops[0], &tmpbond0);
  enext2fnext(m, &bots[0], &spintet);
  for(i = 1; i < wrapcount; i ++) {
    if (issymexist(m, &spintet)) {
      enext2fnext(m, &newtops[i], &tmpbond1);
      bond(m, &tmpbond0, &tmpbond1);
    }
    fnext(m, &newtops[i], &tmpbond0);
    enext2fnext(m, &bots[i], &spintet);
  }
  /*  Bond the last to the first if no boundary. */
  if (issymexist(m, &spintet)) {
    enext2fnext(m, &newtops[0], &tmpbond1);
    bond(m, &tmpbond0, &tmpbond1);
  }
  if (m->checksubsegs) {
    for(i = 0; i < wrapcount; i++) {
      enextfnext(m, &bots[i], &worktet); /*  edge n1->n2. */
      tsspivot1(m, &worktet, &n1n2seg);
      if (n1n2seg.sh != m->dummysh) {
        enext(&newtops[i], &tmpbond0);
        tssbond1(m, &tmpbond0, &n1n2seg);
      }
      enextself(&worktet); /*  edge n2->v ==> n2->b */
      tsspivot1(m, &worktet, &n2vseg);
      if (n2vseg.sh != m->dummysh) {
        tssdissolve1(m, &worktet);
        tssbond1(m, &newtops[i], &n2vseg);
      }
      enextself(&worktet); /*  edge v->n1 ==> b->n1 */
      tsspivot1(m, &worktet, &n1vseg);
      if (n1vseg.sh != m->dummysh) {
        tssdissolve1(m, &worktet);
        enext2(&newtops[i], &tmpbond0);
        tssbond1(m, &tmpbond0, &n1vseg);
      }
    }
  }

  /*  Is there exist subfaces and subsegment need to be split? */
  if (m->checksubfaces) {
    if (abseg.sh != m->dummysh) {
      /*  A subsegment needs be split. */
      spivot(&abseg, &splitsh);
      if (splitsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    }
    if (splitsh.sh != m->dummysh) {
      /*  Split subfaces (and subsegment). */
      ierr = TetGenMeshFindEdge_face(m, &splitsh, pa, pb);CHKERRQ(ierr);
      ierr = TetGenMeshSplitSubEdge_queue(m, newpoint, &splitsh, PETSC_NULL);CHKERRQ(ierr);
    }
  }

  if (b->verbose > 3) {
    for(i = 0; i < wrapcount; i++) {
      printf("    Updating bots[%i] ", i);
      ierr = TetGenMeshPrintTet(m, &(bots[i]), PETSC_FALSE);CHKERRQ(ierr);
      printf("    Creating newtops[%i] ", i);
      ierr = TetGenMeshPrintTet(m, &(newtops[i]), PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  if (flipqueue) {
    for(i = 0; i < wrapcount; i++) {
      ierr = TetGenMeshEnqueueFlipFace(m, &bots[i], flipqueue);CHKERRQ(ierr);
      ierr = TetGenMeshEnqueueFlipFace(m, &newtops[i], flipqueue);CHKERRQ(ierr);
    }
  }

  /*  Set the return handle be avn1n2.  It is got by transforming from */
  /*    'bots[0]' (which is an1n2v). */
  fnext(m, &bots[0], &spintet); /*  spintet is an1vn2. */
  esymself(&spintet); /*  spintet is n1avn2. */
  enextself(&spintet); /*  spintet is avn1n2. */
  *splittet = spintet;

  ierr = PetscFree2(bots,newtops);CHKERRQ(ierr);
  if (isSplit) {*isSplit = PETSC_TRUE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormStarPolyhedron"
/*  formstarpolyhedron()    Get the star ployhedron of a point 'pt'.           */
/*                                                                             */
/*  The polyhedron P is formed by faces of tets having 'pt' as a vertex.  If   */
/*  'complete' is TRUE, P is the complete star of 'pt'. Otherwise, P is boun-  */
/*  ded by subfaces, i.e. P is only part of the star of 'pt'.                  */
/*                                                                             */
/*  'tetlist' T returns the tets, it has one of such tets on input. Moreover,  */
/*  if t is in T, then oppo(t) = p.  Topologically, T is the star of p;  and   */
/*  the faces of T is the link of p. 'verlist' V returns the vertices of T.    */
/* tetgenmesh::formstarpolyhedron() */
PetscErrorCode TetGenMeshFormStarPolyhedron(TetGenMesh *m, point pt, List* tetlist, List* verlist, PetscBool complete)
{
  triface starttet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  face    checksh  = {PETSC_NULL, 0};
  point ver[3];
  int len, idx, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Get a tet t containing p. */
  ierr = ListItem(tetlist, 0, (void **) &starttet);CHKERRQ(ierr);
  /*  Let oppo(t) = p. */
  for(starttet.loc = 0; starttet.loc < 4; starttet.loc++) {
    if (oppo(&starttet) == pt) break;
  }
  if (starttet.loc >= 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate opposing vertex");
  /*  Add t into T. */
  ierr = ListSetItem(tetlist, 0, &starttet);CHKERRQ(ierr);
  infect(m, &starttet);
  if (verlist) {
    /*  Add three verts of t into V. */
    ver[0] = org(&starttet);
    ver[1] = dest(&starttet);
    ver[2] = apex(&starttet);
    for(i = 0; i < 3; i++) {
      /*  Mark the vert by inversing the index of the vert. */
      idx = pointmark(m, ver[i]);
      setpointmark(m, ver[i], -idx - 1); /*  -1 to distinguish the zero. */
      ierr = ListAppend(verlist, &(ver[i]), PETSC_NULL);CHKERRQ(ierr);
    }
  }

  /*  Find other tets by a broadth-first search. */
  ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
    starttet.ver = 0;
    for(j = 0; j < 3; j++) {
      fnext(m, &starttet, &neightet);
      tspivot(m, &neightet, &checksh);
      /*  Should we cross a subface. */
      if ((checksh.sh == m->dummysh) || complete) {
        /*  Get the neighbor n. */
        symself(&neightet);
        if ((neightet.tet != m->dummytet) && !infected(m, &neightet)) {
          /*  Let oppo(n) = p. */
          for(neightet.loc = 0; neightet.loc < 4; neightet.loc++) {
            if (oppo(&neightet) == pt) break;
          }
          if (neightet.loc >= 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate opposing vertex");
          /*  Add n into T. */
          infect(m, &neightet);
          ierr = ListAppend(tetlist, &neightet, PETSC_NULL);CHKERRQ(ierr);
          if (verlist) {
            /*  Add the apex vertex in n into V. */
            ver[0] = org(&starttet);
            ver[1] = dest(&starttet);
            ierr = TetGenMeshFindEdge_triface(m, &neightet, ver[0], ver[1]);CHKERRQ(ierr);
            ver[2] = apex(&neightet);
            idx = pointmark(m, ver[2]);
            if (idx >= 0) {
              setpointmark(m, ver[2], -idx - 1);
              ierr = ListAppend(verlist, &(ver[2]), PETSC_NULL);CHKERRQ(ierr);
            }
          }
        }
      }
      enextself(&starttet);
    }
    ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
  }

  /*  Uninfect tets. */
  ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
    uninfect(m, &starttet);
  }
  if (verlist) {
    /*  Uninfect vertices. */
    ierr = ListLength(verlist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(verlist, i, (void **) &ver[0]);CHKERRQ(ierr);
      idx = pointmark(m, ver[0]);
      setpointmark(m, ver[0], -(idx + 1));
    }
  }
  PetscFunctionReturn(0);
}

/*  Terminology: BC(p) and CBC(p), B(p) and C(p).                              */
/*                                                                             */
/*  Given an arbitrary point p,  the Bowyer-Watson cavity BC(p) is formed by   */
/*  tets whose circumspheres containing p.  The outer faces of BC(p) form a    */
/*  polyhedron B(p).                                                           */
/*                                                                             */
/*  If p is on a facet F, the constrained Bowyer-Watson cavity CBC(p) on F is  */
/*  formed by subfaces of F whose circumspheres containing p. The outer edges  */
/*  of CBC(p) form a polygon C(p).  B(p) is separated into two parts by C(p),  */
/*  denoted as B_1(p) and B_2(p), one of them may be empty (F is on the hull). */
/*                                                                             */
/*  If p is on a segment S which is shared by n facets.  There exist n C(p)s,  */
/*  each one is a non-closed polygon (without S). B(p) is split into n parts,  */
/*  each of them is denoted as B_i(p), some B_i(p) may be empty.               */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormBowatCavitySub"
/*  formbowatcavitysub()    Form CBC(p) and C(p) on a facet F.                 */
/*                                                                             */
/*  Parameters: bp = p, bpseg = S, sublist = CBC(p), subceillist = C(p).       */
/*                                                                             */
/*  CBC(p) contains at least one subface on input; S may be NULL which means   */
/*  that p is inside a facet. On output, all subfaces of CBC(p) are infected,  */
/*  and the edge rings are oriented to the same halfspace.                     */
/* tetgenmesh::formbowatcavitysub() */
PetscErrorCode TetGenMeshFormBowatCavitySub(TetGenMesh *m, point bp, face *bpseg, List *sublist, List *subceillist)
{
  TetGenOpts    *b  = m->b;
  triface adjtet = {PETSC_NULL, 0, 0};
  face startsh = {PETSC_NULL, 0}, neighsh = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  point pa, pb, pc, pd;
  PetscReal sign;
  int len, len2, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Form CBC(p) and C(p) by a broadth-first searching. */
  ierr = ListLength(sublist,  &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(sublist, i, (void **) &startsh);CHKERRQ(ierr); /*  startsh = f. */
    /*  Look for three neighbors of f. */
    for(j = 0; j < 3; j++) {
      sspivot(m, &startsh, &checkseg);
      if (checkseg.sh == m->dummysh) {
        /*  Get its neighbor n. */
        spivot(&startsh, &neighsh);
        /*  Is n already in CBC(p)? */
        if (!sinfected(m, &neighsh)) {
          stpivot(m, &neighsh, &adjtet);
          if (adjtet.tet == m->dummytet) {
            sesymself(&neighsh);
            stpivot(m, &neighsh, &adjtet);
          }
          /*  For positive orientation that TetGenInsphere() test requires. */
          adjustedgering_triface(&adjtet, CW);
          pa = org(&adjtet);
          pb = dest(&adjtet);
          pc = apex(&adjtet);
          pd = oppo(&adjtet);
          sign = TetGenInsphere(pa, pb, pc, pd, bp);
          if (sign >= 0.0) {
            /*  Orient edge ring of n according to that of f. */
            if (sorg(&neighsh) != sdest(&startsh)) sesymself(&neighsh);
            /*  Collect it into CBC(p). */
            sinfect(m, &neighsh);
            ierr = ListAppend(sublist, &neighsh, PETSC_NULL);CHKERRQ(ierr);
          } else {
            ierr = ListAppend(subceillist, &startsh, PETSC_NULL);CHKERRQ(ierr); /*  Found an edge of C(p). */
          }
        }
      } else {
        /*  Do not cross a segment. */
        if (bpseg) {
          if (checkseg.sh != bpseg->sh) {
            ierr = ListAppend(subceillist, &startsh, PETSC_NULL);CHKERRQ(ierr); /*  Found an edge of C(p). */
          }
        } else {
          ierr = ListAppend(subceillist, &startsh, PETSC_NULL);CHKERRQ(ierr); /*  Found an edge of C(p). */
        }
      }
      senextself(&startsh);
    }
    ierr = ListLength(sublist,  &len);CHKERRQ(ierr);
  }

  ierr = ListLength(sublist,     &len);CHKERRQ(ierr);
  ierr = ListLength(subceillist, &len2);CHKERRQ(ierr);
  PetscInfo3(b->in, "    Collect CBC(%d): %d subfaces, %d edges.\n", pointmark(m, bp), len, len2);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormBowatCavityQuad"
/*  formbowatcavityquad()    Form BC_i(p) and B_i(p) in a quadrant.            */
/*                                                                             */
/*  Parameters: bp = p, tetlist = BC_i(p), ceillist = B_i(p).                  */
/*                                                                             */
/*  BC_i(p) contains at least one tet on input. On finish, all tets collected  */
/*  in BC_i(p) are infected. B_i(p) may not closed when p is on segment or in  */
/*  facet. C(p) must be formed before this routine.  Check the infect flag of  */
/*  a subface to identify the unclosed side of B_i(p).  These sides will be    */
/*  closed by new subfaces of C(p)s.                                           */
/* tetgenmesh::formbowatcavityquad() */
PetscErrorCode TetGenMeshFormBowatCavityQuad(TetGenMesh *m, point bp, List *tetlist, List *ceillist)
{
  TetGenOpts    *b  = m->b;
  triface starttet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0};
  point pa, pb, pc, pd;
  PetscReal sign;
  int len, len2, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Form BC_i(p) and B_i(p) by a broadth-first searching. */
  ierr = ListLength(tetlist,  &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
    for(starttet.loc = 0; starttet.loc < 4; starttet.loc++) {
      /*  Try to collect the neighbor of the face (f). */
      tspivot(m, &starttet, &checksh);
      if (checksh.sh == m->dummysh) {
        /*  Get its neighbor n. */
        sym(&starttet, &neightet);
        /*  Is n already in BC_i(p)? */
        if (!infected(m, &neightet)) {
          /*  For positive orientation that TetGenInsphere() test requires. */
          adjustedgering_triface(&neightet, CW);
          pa = org(&neightet);
          pb = dest(&neightet);
          pc = apex(&neightet);
          pd = oppo(&neightet);
          sign = TetGenInsphere(pa, pb, pc, pd, bp);
          if (sign >= 0.0) {
            /*  Collect it into BC_i(p). */
            infect(m, &neightet);
            ierr = ListAppend(tetlist, &neightet, PETSC_NULL);CHKERRQ(ierr);
          } else {
            ierr = ListAppend(ceillist, &starttet, PETSC_NULL);CHKERRQ(ierr); /*  Found a face of B_i(p). */
          }
        }
      } else {
        /*  Do not cross a boundary face. */
        if (!sinfected(m, &checksh)) {
          ierr = ListAppend(ceillist, &starttet, PETSC_NULL);CHKERRQ(ierr); /*  Found a face of B_i(p). */
        }
      }
    }
    ierr = ListLength(tetlist,  &len);CHKERRQ(ierr);
  }

  ierr = ListLength(tetlist,  &len);CHKERRQ(ierr);
  ierr = ListLength(ceillist, &len2);CHKERRQ(ierr);
  PetscInfo3(b->in, "    Collect BC_i(%d): %d tets, %d faces.\n", pointmark(m, bp), len, len2);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormBowatCavity"
/*  formbowatcavity()    Form BC(p), B(p), CBC(p)s, and C(p)s.                 */
/*                                                                             */
/*  If 'bpseg'(S) != NULL, p is on segment S, else, p is on facet containing   */
/*  'bpsh' (F).  'n' returns the number of quadrants in BC(p). 'nmax' is the   */
/*  maximum pre-allocated array length for the lists.                          */
/* tetgenmesh::formbowatcavity() */
PetscErrorCode TetGenMeshFormBowatCavity(TetGenMesh *m, point bp, face *bpseg, face *bpsh, int *n, int *nmax, List **sublists, List **subceillists, List **tetlists, List **ceillists)
{
  List *sublist;
  triface adjtet = {PETSC_NULL, 0, 0};
  face startsh = {PETSC_NULL, 0}, spinsh = {PETSC_NULL, 0};
  point pa, pb;
  int len, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *n = 0;
  if (bpseg) {
    /*  p is on segment S. */
    bpseg->shver = 0;
    pa = sorg(bpseg);
    pb = sdest(bpseg);
    /*  Count the number of facets sharing at S. */
    spivot(bpseg, &startsh);
    spinsh = startsh;
    do {
      (*n)++; /*  spinshlist->append(&spinsh); */
      spivotself(&spinsh);
    } while (spinsh.sh != startsh.sh);
    /*  *n is the number of quadrants around S. */
    if (*n > *nmax) {
      /*  Reallocate arrays. Should not happen very often. */
      ierr = PetscFree(tetlists);CHKERRQ(ierr);
      ierr = PetscFree(ceillists);CHKERRQ(ierr);
      ierr = PetscFree(sublists);CHKERRQ(ierr);
      ierr = PetscFree(subceillists);CHKERRQ(ierr);
      ierr = PetscMalloc(*n * sizeof(List *), &tetlists);CHKERRQ(ierr);
      ierr = PetscMalloc(*n * sizeof(List *), &ceillists);CHKERRQ(ierr);
      ierr = PetscMalloc(*n * sizeof(List *), &sublists);CHKERRQ(ierr);
      ierr = PetscMalloc(*n * sizeof(List *), &subceillists);CHKERRQ(ierr);
      *nmax = *n;
    }
    /*  Form CBC(p)s and C(p)s. */
    spinsh = startsh;
    for(i = 0; i < *n; i++) {
      ierr = ListCreate(sizeof(face), PETSC_NULL, 256, PETSC_DECIDE, &sublists[i]);CHKERRQ(ierr);
      ierr = ListCreate(sizeof(face), PETSC_NULL, 256, PETSC_DECIDE, &subceillists[i]);CHKERRQ(ierr);
      /*  Set a subface f to start search. */
      startsh = spinsh;
      /*  Let f face to the quadrant of interest (used in forming BC(p)). */
      ierr = TetGenMeshFindEdge_face(m, &startsh, pa, pb);CHKERRQ(ierr);
      sinfect(m, &startsh);
      ierr = ListAppend(sublists[i], &startsh, PETSC_NULL);CHKERRQ(ierr);
      ierr = TetGenMeshFormBowatCavitySub(m, bp, bpseg, sublists[i], subceillists[i]);CHKERRQ(ierr);
      /*  Go to the next facet. */
      spivotself(&spinsh);
    }
  } else if (sublists) {
    /*  p is on a facet. */
    *n = 2;
    /*  Form CBC(p) and C(p). */
    ierr = ListCreate(sizeof(face), PETSC_NULL, 256, PETSC_DECIDE, &sublists[0]);CHKERRQ(ierr);
    ierr = ListCreate(sizeof(face), PETSC_NULL, 256, PETSC_DECIDE, &subceillists[0]);CHKERRQ(ierr);
    sinfect(m, bpsh);
    ierr = ListAppend(sublists[0], bpsh, PETSC_NULL);CHKERRQ(ierr);
    ierr = TetGenMeshFormBowatCavitySub(m, bp, PETSC_NULL, sublists[0], subceillists[0]);CHKERRQ(ierr);
  } else {
    /*  p is inside a tet. */
    *n = 1;
  }

  /*  Form BC_i(p) and B_i(p). */
  for(i = 0; i < *n; i++) {
    ierr = ListCreate(sizeof(triface), PETSC_NULL, 256, PETSC_DECIDE, &tetlists[i]);CHKERRQ(ierr);
    ierr = ListCreate(sizeof(triface), PETSC_NULL, 256, PETSC_DECIDE, &ceillists[i]);CHKERRQ(ierr);
    if (sublists) {
      /*  There are C(p)s. */
      sublist = ((!bpseg) ? sublists[0] : sublists[i]);
      /*  Add all adjacent tets of C_i(p) into BC_i(p). */
      ierr = ListLength(sublist, &len);CHKERRQ(ierr);
      for(j = 0; j < len; j++) {
        ierr = ListItem(sublist, j, (void **) &startsh);CHKERRQ(ierr);
        /*  Adjust the side facing to the right quadrant for C(p). */
        if ((!bpseg) && (i == 1)) sesymself(&startsh);
        stpivot(m, &startsh, &adjtet);
        if (adjtet.tet != m->dummytet) {
          if (!infected(m, &adjtet)) {
            infect(m, &adjtet);
            ierr = ListAppend(tetlists[i], &adjtet, PETSC_NULL);CHKERRQ(ierr);
          }
        }
      }
      if (bpseg) {
        /*  The quadrant is bounded by another facet. */
        sublist = ((i < *n - 1) ? sublists[i + 1] : sublists[0]);
        ierr = ListLength(sublist, &len);CHKERRQ(ierr);
        for (j = 0; j < len; j++) {
          ierr = ListItem(sublist, j, (void **) &startsh);CHKERRQ(ierr);
          /*  Adjust the side facing to the right quadrant for C(p). */
          sesymself(&startsh);
          stpivot(m, &startsh, &adjtet);
          if (adjtet.tet != m->dummytet) {
            if (!infected(m, &adjtet)) {
              infect(m, &adjtet);
              ierr = ListAppend(tetlists[i], &adjtet, PETSC_NULL);CHKERRQ(ierr);
            }
          }
        }
      }
    }
    /*  It is possible that BC_i(p) is empty. */
    ierr = ListLength(tetlists[i], &len);CHKERRQ(ierr);
    if (len == 0) continue;
    /*  Collect the rest of tets of BC_i(p) and form B_i(p). */
    /*  if (b->conformdel) { */
      /*  formbowatcavitysegquad(bp, tetlists[i], ceillists[i]); */
    /*  } else { */
    ierr = TetGenMeshFormBowatCavityQuad(m, bp, tetlists[i], ceillists[i]);CHKERRQ(ierr);
    /*  } */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshReleaseBowatCavity"
/*  releasebowatcavity()    Undo and free the memory allocated in routine      */
/*                          formbowatcavity().                                 */
/* tetgenmesh::releasebowatcavity() */
PetscErrorCode TetGenMeshReleaseBowatCavity(TetGenMesh *m, face *bpseg, int n, List **sublists, List **subceillist, List **tetlists, List **ceillists)
{
  triface oldtet = {PETSC_NULL, 0, 0};
  face oldsh = {PETSC_NULL, 0};
  int len, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sublists) {
    /*  Release CBC(p)s. */
    for(i = 0; i < n; i++) {
      /*  Uninfect subfaces of CBC(p). */
      ierr = ListLength(sublists[i], &len);CHKERRQ(ierr);
      for(j = 0; j < len; j++) {
        ierr = ListItem(sublists[i], j, (void **) &oldsh);CHKERRQ(ierr);
#ifdef PETSC_USE_DEBUG
        if (!sinfected(m, &oldsh)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
        suninfect(m, &oldsh);
      }
      ierr = ListDestroy(&sublists[i]);CHKERRQ(ierr);
      ierr = ListDestroy(&subceillist[i]);CHKERRQ(ierr);
      sublists[i] = PETSC_NULL;
      subceillist[i] = PETSC_NULL;
      if (!bpseg) break;
    }
  }
  /*  Release BC(p). */
  for(i = 0; i < n; i++) {
    /*  Uninfect tets of BC_i(p). */
    ierr = ListLength(tetlists[i], &len);CHKERRQ(ierr);
    for(j = 0; j < len; j++) {
      ierr = ListItem(tetlists[i], j, (void **) &oldtet);CHKERRQ(ierr);
#ifdef PETSC_USE_DEBUG
      if (!infected(m, &oldtet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
      uninfect(m, &oldtet);
    }
    ierr = ListDestroy(&tetlists[i]);CHKERRQ(ierr);
    ierr = ListDestroy(&ceillists[i]);CHKERRQ(ierr);
    tetlists[i] = PETSC_NULL;
    ceillists[i] = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshValidateBowatCavityQuad"
/*  validatebowatcavityquad()    Valid B_i(p).                                 */
/*                                                                             */
/*  B_i(p) is valid if all faces of B_i(p) are visible by p, else B_i(p) is    */
/*  invalid.  Each tet of BC_i(p) which has such a face is marked (uninfect).  */
/*  They will be removed in updatebowatcavityquad().                           */
/*                                                                             */
/*  Return TRUE if B(p) is valid, else, return FALSE.                          */
/* tetgenmesh::validatebowatcavityquad() */
PetscErrorCode TetGenMeshValidateBowatCavityQuad(TetGenMesh *m, point bp, List *ceillist, PetscReal maxcosd, PetscBool *isValid)
{
  triface ceiltet = {PETSC_NULL, 0, 0};
  point pa, pb, pc;
  PetscReal ori, cosd;
  int len, remcount, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Check the validate of B(p), cut tets having invisible faces. */
  remcount = 0;
  ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(ceillist, i, (void **) &ceiltet);CHKERRQ(ierr);
    if (infected(m, &ceiltet)) {
      adjustedgering_triface(&ceiltet, CCW);
      pa = org(&ceiltet);
      pb = dest(&ceiltet);
      pc = apex(&ceiltet);
      ori = TetGenOrient3D(pa, pb, pc, bp);
      if (ori >= 0.0) {
        /*  Found an invisible face. */
        uninfect(m, &ceiltet);
        remcount++;
        continue;
      }
      /*  If a non-trival 'maxcosd' is given. */
      if (maxcosd > -1.0) {
        /*  Get the maximal dihedral angle of tet abcp. */
        ierr = TetGenMeshTetAllDihedral(m, pa, pb, pc, bp, PETSC_NULL, &cosd, PETSC_NULL);CHKERRQ(ierr);
        /*  Do not form the tet if the maximal dihedral angle is not reduced. */
        if (cosd < maxcosd) {
          uninfect(m, &ceiltet);
          remcount++;
        }
      }
    }
  }
  if (isValid) {*isValid = (remcount == 0) ? PETSC_TRUE : PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshUpdateBowatCavityQuad"
/*  updatebowatcavityquad()    Update BC_i(p) and reform B_i(p).               */
/*                                                                             */
/*  B_i(p) is invalid and some tets in BC_i(p) have been marked to be removed  */
/*  in validatebowatcavityquad().  This routine actually remove the cut tets   */
/*  of BC_i(p) and re-form the B_i(p).                                         */
/* tetgenmesh::updatebowatcavityquad() */
PetscErrorCode TetGenMeshUpdateBowatCavityQuad(TetGenMesh *m, List *tetlist, List *ceillist)
{
  TetGenOpts    *b  = m->b;
  triface cavtet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0};
  int len, len2, remcount, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  remcount = 0;
  ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(tetlist, i, (void **) &cavtet);CHKERRQ(ierr);
    if (!infected(m, &cavtet)) {
      ierr = ListDelete(tetlist, i, 1);CHKERRQ(ierr);
      remcount++;
      i--;
    }
    ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
  }

  /*  Are there tets have been cut in BC_i(p)? */
  if (remcount > 0) {
    /*  Re-form B_i(p). */
    ierr = ListClear(ceillist);CHKERRQ(ierr);
    ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(tetlist, i, (void **) &cavtet);CHKERRQ(ierr);
      for(cavtet.loc = 0; cavtet.loc < 4; cavtet.loc++) {
        tspivot(m, &cavtet, &checksh);
        if (checksh.sh == m->dummysh) {
          sym(&cavtet, &neightet);
          if (!infected(m, &neightet)) {
            ierr = ListAppend(ceillist, &cavtet, PETSC_NULL);CHKERRQ(ierr); /*  Found a face of B_i(p). */
          }
        } else {
          /*  Do not cross a boundary face. */
          if (!sinfected(m, &checksh)) {
            ierr = ListAppend(ceillist, &cavtet, PETSC_NULL);CHKERRQ(ierr); /*  Found a face of B_i(p). */
          }
        }
      }
    }
    ierr = ListLength(tetlist,  &len);CHKERRQ(ierr);
    ierr = ListLength(ceillist, &len2);CHKERRQ(ierr);
    PetscInfo2(b->in, "    Update BC_i(p): %d tets, %d faces.\n", len, len2);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshUpdateBowatCavitySub"
/*  updatebowatcavitysub()    Check and update CBC(p) and C(p).                */
/*                                                                             */
/*  A CBC(p) is valid if all its subfaces are inside or on the hull of BC(p).  */
/*  A subface s of CBC(p) is invalid if it is in one of the two cases:         */
/*    (1) s is completely outside BC(p);                                       */
/*    (2) s has two adjacent tets but only one of them is in BC(p);            */
/*  s is removed from CBC(p) if it is invalid. If there is an adjacent tet of  */
/*  s which is in BC(p), it gets removed from BC(p) too. If CBC(p) is updated, */
/*  C(p) is re-formed.                                                         */
/*                                                                             */
/*  A C(p) is valid if all its edges are on the hull of BC(p).  An edge e of   */
/*  C(p) may be inside BC(p) if e is a segment and belongs to only one facet.  */
/*  To correct C(p), a tet of BC(p) which shields e gets removed.              */
/*                                                                             */
/*  If BC(p) is formed with locally non-Delaunay check (b->conformdel > 0).    */
/*  A boundary-consistent check is needed for non-segment edges of C(p). Let   */
/*  e be such an edge, the subface f contains e and outside C(p) may belong    */
/*  to B(p) due to the non-coplanarity of the facet definition.  The tet of    */
/*  BC(p) containing f gets removed to avoid creating a degenerate new tet.    */
/*                                                                             */
/*  'cutcount' accumulates the total number of cuttets(not only by this call). */
/* tetgenmesh::updatebowatcavitysub() */
PetscErrorCode TetGenMeshUpdateBowatCavitySub(TetGenMesh *m, List *sublist, List *subceillist, int *cutcount)
{
  TetGenOpts    *b  = m->b;
  triface adjtet = {PETSC_NULL, 0, 0}, rotface = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0}, neighsh = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  point pa, pb, pc;
  PetscReal ori1, ori2;
  int remcount;
  int len, len2, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  remcount = 0;
  /*  Check the validity of CBC(p). */
  ierr = ListLength(sublist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(sublist, i, (void **) &checksh);CHKERRQ(ierr);
    /*  Check two adjacent tets of s. */
    for(j = 0; j < 2; j++) {
      stpivot(m, &checksh, &adjtet);
      if (adjtet.tet != m->dummytet) {
        if (!infected(m, &adjtet)) {
          /*  Could be either case (1) or (2). */
          suninfect(m, &checksh); /*  s survives. */
          /*  If the sym. adjtet exists, it should remove from BC(p) too. */
          sesymself(&checksh);
          stpivot(m, &checksh, &adjtet);
          if (adjtet.tet != m->dummytet) {
            if (infected(m, &adjtet)) {
              /*  Found an adj. tet in BC(p), remove it. */
              uninfect(m, &adjtet);
              (*cutcount)++;
            }
          }
          /*  Remove s from C(p). */
          ierr = ListDelete(sublist, i, 1);CHKERRQ(ierr);
          i--;
          remcount++;
          break;
        }
      }
      sesymself(&checksh);
    }
  }
  if (remcount > 0) {
    /*  Some subfaces have been removed from the cavity. */
    if (m->checkpbcs) {
      /*  Check if the facet has a PBC defined. */
      ierr = ListItem(sublist, 0, (void **) &checksh);CHKERRQ(ierr);
      if (shellpbcgroup(m, &checksh) >= 0) {
        /*  Yes, A PBC facet. Remove all subfaces -- Do not insert the point. */
        ierr = ListLength(sublist, &len);CHKERRQ(ierr);
        for(i = 0; i < len; i++) {
          ierr = ListItem(sublist, i, (void **) &checksh);CHKERRQ(ierr);
          suninfect(m, &checksh);
          /*  Remove both side tets from the cavity. */
          for(j = 0; j < 2; j++) {
            stpivot(m, &checksh, &adjtet);
            if (adjtet.tet != m->dummytet) {
              if (infected(m, &adjtet)) {
                uninfect(m, &adjtet);
                (*cutcount)++;
              }
            }
            sesymself(&checksh);
          }
        }
        ierr = ListLength(sublist, &len);CHKERRQ(ierr);
        remcount += len;
        ierr = ListClear(sublist);CHKERRQ(ierr);
      }
    }
    PetscInfo1(b->in, "    Removed %d subfaces from CBC(p).\n", remcount);
    /*  Re-generate C(p). */
    ierr = ListClear(subceillist);CHKERRQ(ierr);
    ierr = ListLength(sublist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(sublist, i, (void **) &checksh);CHKERRQ(ierr);
      for(j = 0; j < 3; j++) {
        spivot(&checksh, &neighsh);
        if (!sinfected(m, &neighsh)) {
          ierr = ListAppend(subceillist, &checksh, PETSC_NULL);CHKERRQ(ierr);
        }
        senextself(&checksh);
      }
    }
    ierr = ListLength(sublist,     &len);CHKERRQ(ierr);
    ierr = ListLength(subceillist, &len2);CHKERRQ(ierr);
    PetscInfo2(b->in, "    Update CBC(p): %d subs, %d edges.\n", len, len2);
  }

  /*  Check the validity of C(p). */
  ierr = ListLength(subceillist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(subceillist, i, (void **) &checksh);CHKERRQ(ierr);
    sspivot(m, &checksh, &checkseg);
    if (checkseg.sh != m->dummysh) {
      /*  A segment. Check if it is inside BC(p). */
      stpivot(m, &checksh, &adjtet);
      if (adjtet.tet == m->dummytet) {
        sesym(&checksh, &neighsh);
        stpivot(m, &neighsh, &adjtet);
      }
      ierr = TetGenMeshFindEdge_triface(m, &adjtet, sorg(&checkseg), sdest(&checkseg));CHKERRQ(ierr);
      adjustedgering_triface(&adjtet, CCW);
      fnext(m, &adjtet, &rotface); /*  It's the same tet. */
      /*  Rotate rotface (f), stop on either of the following cases: */
      /*    (a) meet a subface, or */
      /*    (b) enter an uninfected tet, or */
      /*    (c) rewind back to adjtet. */
      do {
        if (!infected(m, &rotface)) break; /*  case (b) */
        tspivot(m, &rotface, &neighsh);
        if (neighsh.sh != m->dummysh) break; /*  case (a) */
        /*  Go to the next tet of the facing ring. */
        fnextself(m, &rotface);
      } while (apex(&rotface) != apex(&adjtet));
      /*  Is it case (c)? */
      if (apex(&rotface) == apex(&adjtet)) {
        /*  The segment is enclosed by BC(p), invalid cavity. */
        pa = org(&adjtet);
        pb = dest(&adjtet);
        pc = apex(&adjtet);
        /*  Find the shield tet and cut it. Notice that the shield tet may */
        /*    not be unique when there are four coplanar points, ie., */
        /*    ori1 * ori2 == 0.0. In such case, choose either of them. */
        fnext(m, &adjtet, &rotface);
        do {
          fnextself(m, &rotface);
          if (!infected(m, &rotface)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          ori1 = TetGenOrient3D(pa, pb, pc, apex(&rotface));
          ori2 = TetGenOrient3D(pa, pb, pc, oppo(&rotface));
        } while (ori1 * ori2 > 0.0);
        /*  Cut this tet from BC(p). */
        uninfect(m, &rotface);
        (*cutcount)++;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTrimBowatCavity"
/*  trimbowatcavity()    Validate B(p), CBC(p)s and C(p)s, update BC(p).       */
/*                                                                             */
/*  A B(p) is valid if all its faces are visible by p. If a face f of B(p) is  */
/*  found invisible by p, the tet of BC(p) containing f gets removed and B(p)  */
/*  is refromed. The new B(p) may still contain invisible faces by p. Iterat-  */
/*  ively do the above procedure until B(p) is satisfied.                      */
/*                                                                             */
/*  A CBC(p) is valid if each subface of CBC(p) is either on the hull of BC(p) */
/*  or completely inside BC(p). If a subface s of CBC(p) is not valid, it is   */
/*  removed from CBC(p) and C(p) is reformed. If there exists a tet t of BC(p) */
/*  containg s, t is removed from BC(p). The process for validating BC(p) and  */
/*  B(p) is re-excuted.                                                        */
/*                                                                             */
/*  A C(p) is valid if each edge of C(p) is on the hull of BC(p). If an edge   */
/*  e of C(p) is invalid (e should be a subsegment which only belong to one    */
/*  facet), a tet of BC(p) which contains e and has two other faces shielding  */
/*  e is removed. The process for validating BC(p) and B(p) is re-excuted.     */
/*                                                                             */
/*  If either BC(p) or CBC(p) becomes empty. No valid BC(p) is found, return   */
/*  FALSE. else, return TRUE.                                                  */
/* tetgenmesh::trimbowatcavity() */
PetscErrorCode TetGenMeshTrimBowatCavity(TetGenMesh *m, point bp, face *bpseg, int n, List **sublists, List **subceillists, List **tetlists, List **ceillists, PetscReal maxcosd, PetscBool *isValid)
{
  PetscBool valflag;
  int oldnum, cutnum, cutcount;
  int            len, i;
  PetscBool      isGood;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cutnum = 0; /*  Count the total number of cut-off tets of BC(p). */
  valflag = PETSC_TRUE;

  do {
    /*  Validate BC(p), B(p). */
    for(i = 0; i < n && valflag; i++) {
      ierr = ListLength(tetlists[i], &oldnum);CHKERRQ(ierr);
      /*  Iteratively validate BC_i(p) and B_i(p). */
      ierr = TetGenMeshValidateBowatCavityQuad(m, bp, ceillists[i], maxcosd, &isGood);CHKERRQ(ierr);
      while(!isGood) {
        /*  Update BC_i(p) and B_i(p). */
        ierr = TetGenMeshUpdateBowatCavityQuad(m, tetlists[i], ceillists[i]);CHKERRQ(ierr);
        ierr = ListLength(tetlists[i], &len);CHKERRQ(ierr);
        valflag = len > 0 ? PETSC_TRUE : PETSC_FALSE;
      }
      ierr = ListLength(tetlists[i], &len);CHKERRQ(ierr);
      cutnum += (oldnum - len);
    }
    if (valflag && sublists) {
      /*  Validate CBC(p), C(p). */
      cutcount = 0;
      for(i = 0; i < n; i++) {
        ierr = TetGenMeshUpdateBowatCavitySub(m, sublists[i], subceillists[i], &cutcount);CHKERRQ(ierr);
        /*  Only do once if p is on a facet. */
        if (!bpseg) break;
      }
      /*  Are there cut tets? */
      if (cutcount > 0) {
        /*  Squeeze all cut tets in BC(p), keep valflag once it gets FLASE. */
        for (i = 0; i < n; i++) {
          ierr = ListLength(tetlists[i], &len);CHKERRQ(ierr);
          if (len > 0) {
            ierr = TetGenMeshUpdateBowatCavityQuad(m, tetlists[i], ceillists[i]);CHKERRQ(ierr);
            if (valflag) {
              ierr = ListLength(tetlists[i], &len);CHKERRQ(ierr);
              valflag = len > 0 ? PETSC_TRUE : PETSC_FALSE;
            }
          }
        }
        cutnum += cutcount;
        /*  Go back to valid the updated BC(p). */
        continue;
      }
    }
    break; /*  Leave the while-loop. */
  } while(1);

  /*  Check if any CBC(p) becomes non-empty. */
  if (valflag && sublists) {
    for(i = 0; i < n && valflag; i++) {
      ierr = ListLength(sublists[i], &len);CHKERRQ(ierr);
      valflag = (len > 0) ? PETSC_TRUE : PETSC_FALSE;
      if (!bpseg) break;
    }
  }

  if (valflag && (cutnum > 0)) {
    /*  Accumulate counters. */
    if (bpseg) {
      m->updsegcount++;
    } else if (sublists) {
      m->updsubcount++;
    } else {
      m->updvolcount++;
    }
  }

  if (!valflag) {
    /*  Accumulate counters. */
    if (bpseg) {
      m->failsegcount++;
    } else if (sublists) {
      m->failsubcount++;
    } else {
      m->failvolcount++;
    }
  }

  if (isValid) {*isValid = valflag;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBowatInsertSite"
/*  bowatinsertsite()    Insert a point using the Bowyer-Watson method.        */
/*                                                                             */
/*  Parameters: 'bp' = p, 'splitseg' = S, 'n' = the number of quadrants,       */
/*  'sublists', an array of CBC_i(p)s, 'subceillists', an array of C_i(p)s,    */
/*  'tetlists', an array of BC_i(p)s, 'ceillists', an array of B_i(p)s.        */
/*                                                                             */
/*  If p is inside the mesh domain, then S = NULL, n = 1, CBC(p) and C(p) are  */
/*    NULLs. 'tetlists[0]' = BC(p), 'ceillists[0]' = B(p).                     */
/*  If p is on a facet F, then S = NULL, n = 2, and 'subceillists[0]' = C(p),  */
/*   'subceillists[1]' is not needed (set it to NULL). B_1(p) and B_2(p) are   */
/*   in 'ceillists[0]' and 'ceillists[1]'.                                     */
/*  If p is on a segment S, then F(S) is a list of subfaces around S, and n =  */
/*    len(F(S)), there are n C_i(p)s and B_i(p)s supplied in 'subceillists[i]' */
/*    and 'ceillists[i]'.                                                      */
/*                                                                             */
/*  If 'verlist' != NULL, it returns a list of vertices which connect to p.    */
/*    This vertices are used for interpolating size of p.                      */
/*                                                                             */
/*  If 'flipque' != NULL, it returns a list of internal faces of new tets in   */
/*    BC(p), faces on C(p)s are excluded. These faces may be locally non-      */
/*    Delaunay and will be flipped if they are flippable. Such non-Delaunay    */
/*    faces may exist when p is inserted to split an encroaching segment.      */
/*                                                                             */
/*  'chkencseg', 'chkencsub', and 'chkbadtet' are flags that indicate whether  */
/*  or not there should be checks for the creation of encroached subsegments,  */
/*  subfaces, or bad quality tets. If 'chkencseg' = TRUE, the encroached sub-  */
/*  segments are added to the list of subsegments to be split.                 */
/*                                                                             */
/*  On return, 'ceillists' returns Star(p).                                    */
/* tetgenmesh::bowatinsertsite() */
PetscErrorCode TetGenMeshBowatInsertSite(TetGenMesh *m, point bp, face *splitseg, int n, List **sublists, List **subceillists, List **tetlists, List **ceillists, List *verlist, Queue *flipque, PetscBool chkencseg, PetscBool chkencsub, PetscBool chkbadtet)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  List *ceillist, *subceillist;
  triface oldtet = {PETSC_NULL, 0, 0}, newtet = {PETSC_NULL, 0, 0}, newface = {PETSC_NULL, 0, 0}, rotface = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  face oldsh = {PETSC_NULL, 0}, newsh = {PETSC_NULL, 0}, newedge = {PETSC_NULL, 0}, checksh = {PETSC_NULL, 0};
  face spinsh = {PETSC_NULL, 0}, casingin = {PETSC_NULL, 0}, casingout = {PETSC_NULL, 0};
  face *apsegshs, *pbsegshs;
  face apseg = {PETSC_NULL, 0}, pbseg = {PETSC_NULL, 0}, checkseg = {PETSC_NULL, 0};
  point pa, pb, pc;
  PetscReal attrib, volume;
  int len, len2, idx, i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo4(b->in, "    Insert point %d (%.12g, %.12g, %.12g)", pointmark(m, bp), bp[0], bp[1], bp[2]);
  if (splitseg) {
    PetscInfo(b->in, " on segment.\n");
    m->bowatsegcount++;
  } else {
    if (subceillists) {
      PetscInfo(b->in, " on facet.\n");
      m->bowatsubcount++;
    } else {
      PetscInfo(b->in, " in volume.\n");
      m->bowatvolcount++;
    }
  }

  /*  Create new tets to fill B(p). */
  for(k = 0; k < n; k++) {
    /*  Create new tets from each B_i(p). */
    ceillist = ceillists[k];
    ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(ceillist, i, (void **) &oldtet);CHKERRQ(ierr);
      adjustedgering_triface(&oldtet, CCW);
      pa = org(&oldtet);
      pb = dest(&oldtet);
      pc = apex(&oldtet);
      ierr = TetGenMeshMakeTetrahedron(m, &newtet);CHKERRQ(ierr);
      setorg(&newtet, pa);
      setdest(&newtet, pb);
      setapex(&newtet, pc);
      setoppo(&newtet, bp);
      for(j = 0; j < in->numberoftetrahedronattributes; j++) {
        attrib = elemattribute(m, oldtet.tet, j);
        setelemattribute(m, newtet.tet, j, attrib);
      }
      if (b->varvolume) {
        volume = volumebound(m, oldtet.tet);
        if (volume > 0.0) {
          if (!b->fixedvolume && b->refine) {
            /*  '-r -a' switches and a .vol file case. Enlarge the maximum */
            /*    volume constraint for the new tets. Hence the new points */
            /*    only spread near the original constrained tet. */
            volume *= 1.2;
          }
        }
        setvolumebound(m, newtet.tet, volume);
      }
      sym(&oldtet, &neightet);
      tspivot(m, &oldtet, &checksh);
      if (neightet.tet != m->dummytet) {
        bond(m, &newtet, &neightet);
      }
      if (checksh.sh != m->dummysh) {
        tsbond(m, &newtet, &checksh);
      }
      if (verlist) {
        /*  Collect vertices connecting to p. */
        idx = pointmark(m, pa);
        if (idx >= 0) {
          setpointmark(m, pa, -idx - 1);
          ierr = ListAppend(verlist, &pa, PETSC_NULL);CHKERRQ(ierr);
        }
        idx = pointmark(m, pb);
        if (idx >= 0) {
          setpointmark(m, pb, -idx - 1);
          ierr = ListAppend(verlist, &pb, PETSC_NULL);CHKERRQ(ierr);
        }
        idx = pointmark(m, pc);
        if (idx >= 0) {
          setpointmark(m, pc, -idx - 1);
          ierr = ListAppend(verlist, &pc, PETSC_NULL);CHKERRQ(ierr);
        }
      }
      /*  Replace the tet by the newtet for checking the quality. */
      ierr = ListSetItem(ceillist, i, (void **) &newtet);CHKERRQ(ierr);
    }
  }
  if (verlist) {
    /*  Uninfect collected vertices. */
    ierr = ListLength(verlist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(verlist, i, (void **) &pa);CHKERRQ(ierr);
      idx = pointmark(m, pa);
      setpointmark(m, pa, -(idx + 1));
    }
  }

  /*  Connect new tets of B(p). Not all faces of new tets can be connected, */
  /*    e.g., if there are empty B_i(p)s. */
  for(k = 0; k < n; k++) {
    ceillist = ceillists[k];
    ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(ceillist, i, (void **) &newtet);CHKERRQ(ierr);
      newtet.ver = 0;
      for(j = 0; j < 3; j++) {
        fnext(m, &newtet, &newface);
        sym(&newface, &neightet);
        if (neightet.tet == m->dummytet) {
          /*  Find the neighbor face by rotating the faces at edge ab. */
          esym(&newtet, &rotface);
          pa = org(&rotface);
          pb = dest(&rotface);
          while (fnextself(m, &rotface));
          /*  Do we meet a boundary face? */
          tspivot(m, &rotface, &checksh);
          if (checksh.sh != m->dummysh) {
            /*  Walk through the boundary and continue to rotate faces. */
            do {
              ierr = TetGenMeshFindEdge_face(m, &checksh, pa, pb);CHKERRQ(ierr);
              sfnextself(m, &checksh);
              if ((sorg(&checksh) != pa) || (sdest(&checksh) != pb)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
              stpivot(m, &checksh, &rotface);
              if (infected(m, &rotface)) {
                /*  Meet an old tet of B_i(p). This side is on the hull and */
                /*    will be connected to a new subface created in C(p). */
                break;
              }
              ierr = TetGenMeshFindEdge_triface(m, &rotface, pa, pb);CHKERRQ(ierr);
              while (fnextself(m, &rotface));
              tspivot(m, &rotface, &checksh);
            } while (checksh.sh != m->dummysh);
          }
          /*  The rotface has edge ab, but it may not have newpt. */
          if (apex(&rotface) == apex(&newface)) {
            /*  Bond the two tets together. */
            bond(m, &newface, &rotface);
            /*  Queue (uniquely) this face if 'flipque' is given. */
            if (flipque) {
              ierr = TetGenMeshEnqueueFlipFace(m, &newface, flipque);CHKERRQ(ierr);
            }
          }
        }
        enextself(&newtet);
      }
    }
  }

  if (subceillists) {
    /*  There are C(p)s. */
    if (splitseg) {
      /*  S (ab) is split by p. */
      splitseg->shver = 0;
      pa = sorg(splitseg);
      pb = sdest(splitseg);
      /*  Allcate two arrays for saving the subface rings of the two new */
      /*    segments a->p and p->b. */
      ierr = PetscMalloc(n * sizeof(face), &apsegshs);CHKERRQ(ierr);
      ierr = PetscMalloc(n * sizeof(face), &pbsegshs);CHKERRQ(ierr);
    }

    /*  For each C_k(p), do the following: */
    /*    (1) Create new subfaces to fill C_k(p), insert them into B(p); */
    /*    (2) Connect new subfaces to each other; */
    for(k = 0; k < n; k++) {
      subceillist = subceillists[k];

      /*  Check if 'hullsize' should be updated. */
      ierr = ListItem(subceillist, 0, (void **) &oldsh);CHKERRQ(ierr);
      stpivot(m, &oldsh, &neightet);
      if (neightet.tet != m->dummytet) {
        sesymself(&oldsh);
        stpivot(m, &oldsh, &neightet);
      }
      if (neightet.tet == m->dummytet) {
        /*  The hull size changes. */
        ierr = ListLength(subceillist, &len);CHKERRQ(ierr);
        ierr = ListLength(sublists[k], &len2);CHKERRQ(ierr);
        m->hullsize += (len - len2);
      }

      /*  (1) Create new subfaces to fill C_k(p), insert them into B(p). */
      ierr = ListLength(subceillist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(subceillist, i, (void **) &oldsh);CHKERRQ(ierr);
        ierr = TetGenMeshMakeShellFace(m, m->subfaces, &newsh);CHKERRQ(ierr);
        setsorg(&newsh, sorg(&oldsh));
        setsdest(&newsh, sdest(&oldsh));
        setsapex(&newsh, bp);
        if (b->quality && m->varconstraint) {
          setareabound(m, &newsh, areabound(m, &oldsh));
        }
        setshellmark(m, &newsh, shellmark(m, &oldsh));
        setshelltype(m, &newsh, shelltype(m, &oldsh));
        if (m->checkpbcs) {
          setshellpbcgroup(m, &newsh, shellpbcgroup(m, &oldsh));
        }
        /*  Replace oldsh by newsh at the edge. */
        spivot(&oldsh, &casingout);
        sspivot(m, &oldsh, &checkseg);
        if (checkseg.sh != m->dummysh) {
          /*  A segment. Insert s into the face ring, ie, s_in -> s -> s_out. */
          if (casingout.sh != m->dummysh) { /*  if (oldsh.sh != casingout.sh) { */
            /*  s is not bonded to itself. */
            spinsh = casingout;
            do {
              casingin = spinsh;
              spivotself(&spinsh);
            } while (sapex(&spinsh) != sapex(&oldsh));
            if (casingin.sh == oldsh.sh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            /*  Bond s_in -> s -> s_out (and dissolve s_in -> s_old -> s_out). */
            sbond1(&casingin, &newsh);
            sbond1(&newsh, &casingout);
          } else {
            /*  Bond newsh -> newsh. */
            sdissolve(m, &newsh); /*  sbond(newsh, newsh); */
          }
          /*  Bond the segment. */
          ssbond(m, &newsh, &checkseg);
        } else {
          /*  Bond s <-> s_out (and dissolve s_out -> s_old). */
          sbond(&newsh, &casingout);
        }

        /*  Insert newsh into B(p). Use the coonections of oldsh. */
        stpivot(m, &oldsh, &neightet);
        if (neightet.tet == m->dummytet) {
          sesymself(&oldsh);
          sesymself(&newsh); /*  Keep the same orientation as oldsh. */
          stpivot(m, &oldsh, &neightet);
        }
        if (!infected(m, &neightet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        /*  Set on the rotating edge. */
        ierr = TetGenMeshFindEdge_triface(m, &neightet, sorg(&oldsh), sdest(&oldsh));CHKERRQ(ierr);
        /*  Choose the rotating direction (to the inside of B(p)). */
        adjustedgering_triface(&neightet, CCW);
        rotface = neightet;
        /*  Rotate face. Stop at a non-infected tet t (not in B(p)) or a */
        /*    hull face f (on B(p)). Get the neighbor n of t or f.  n is */
        /*    a new tet that has just been created to fill B(p). */
        do {
          fnextself(m, &rotface);
          sym(&rotface, &neightet);
          if (neightet.tet == m->dummytet) {
            tspivot(m, &rotface, &checksh);
            if (checksh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            stpivot(m, &checksh, &newtet);
            break;
          } else if (!infected(m, &neightet)) {
            sym(&neightet, &newtet);
            break;
          }
        } while (1);
        if (newtet.tet == rotface.tet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        /*  Set the rotating edge of n. */
        ierr = TetGenMeshFindEdge_triface(m, &newtet, sorg(&oldsh), sdest(&oldsh));CHKERRQ(ierr);
        /*  Choose the rotating direction (to the inside of B(p)). */
        adjustedgering_triface(&newtet, CCW);
        fnext(m, &newtet, &newface);
        if (apex(&newface) != bp) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        /*  newsh has already been oriented toward n. */
        tsbond(m, &newface, &newsh);
        sym(&newface, &neightet); /*  'neightet' maybe outside. */
        sesymself(&newsh);
        tsbond(m, &neightet, &newsh); /*  Bond them anyway. */

        /*  Replace oldsh by newsh in list. */
        ierr = ListSetItem(subceillist, i, (void **) &newsh);CHKERRQ(ierr);
      }

      /*  (2) Connect new subfaces to each other. */
      ierr = ListLength(subceillist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        /*  Get a face cdp. */
        ierr = ListItem(subceillist, i, (void **) &newsh);CHKERRQ(ierr);
        /*  Get a new tet containing cdp. */
        stpivot(m, &newsh, &newtet);
        if (newtet.tet == m->dummytet) {
          sesymself(&newsh);
          stpivot(m, &newsh, &newtet);
        }
        for(j = 0; j < 2; j++) {
          if (j == 0) {
            senext(&newsh, &newedge); /*  edge dp. */
          } else {
            senext2(&newsh, &newedge); /*  edge pc. */
            sesymself(&newedge); /*  edge cp. */
          }
          if (splitseg) {
            /*  Don not operate on newedge if it is ap or pb. */
            if (sorg(&newedge) == pa) {
              apsegshs[k] = newedge;
              continue;
            } else if (sorg(&newedge) == pb) {
              pbsegshs[k] = newedge;
              continue;
            }
          }
          /*  There should no segment inside the cavity. Check it. */
          sspivot(m, &newedge, &checkseg);
          if (checkseg.sh != m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          spivot(&newedge, &casingout);
          if (casingout.sh == m->dummysh) {
            rotface = newtet;
            ierr = TetGenMeshFindEdge_triface(m, &rotface, sorg(&newedge), sdest(&newedge));CHKERRQ(ierr);
            /*  Rotate newtet until meeting a new subface which contains */
            /*    newedge. It must exist since newedge is not a seg. */
            adjustedgering_triface(&rotface, CCW);
            do {
              fnextself(m, &rotface);
              tspivot(m, &rotface, &checksh);
              if (checksh.sh != m->dummysh) break;
            } while (1);
            ierr = TetGenMeshFindEdge_face(m, &checksh, sorg(&newedge), sdest(&newedge));CHKERRQ(ierr);
            sbond(&newedge, &checksh);
          }
        }
      }
      /*  Only do once if p is on a facet. */
      if (!splitseg) break;
    } /*  for (k = 0; k < n; k++) */

    if (splitseg) {
      /*  Update a->b to be a->p. */
      apseg = *splitseg;
      setsdest(&apseg, bp);
      /*  Create a new subsegment p->b. */
      ierr = TetGenMeshMakeShellFace(m, m->subsegs, &pbseg);CHKERRQ(ierr);
      setsorg(&pbseg, bp);
      setsdest(&pbseg, pb);
      /*  p->b gets the same mark and segment type as a->p. */
      setshellmark(m, &pbseg, shellmark(m, &apseg));
      setshelltype(m, &pbseg, shelltype(m, &apseg));
      if (b->quality && m->varconstraint) {
        /*  Copy the area bound into the new subsegment. */
        setareabound(m, &pbseg, areabound(m, &apseg));
      }
      senext(&apseg, &checkseg);
      /*  Get the old connection at b of a->b. */
      spivot(&checkseg, &casingout);
      /*  Bond a->p and p->b together. */
      senext2(&pbseg, &casingin);
      sbond(&casingin, &checkseg);
      if (casingout.sh != m->dummysh) {
        /*  There is a subsegment connect at b of p->b. */
        casingout.shver = 0;
#ifdef PETSC_USE_DEBUG
        if (sorg(&casingout) != pb) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
        senext2self(&casingout);
        senext(&pbseg, &casingin);
        sbond(&casingin, &casingout);
      }

      /*  Bond all new subfaces to a->p and p->b. */
      for(i = 0; i < n; i++) {
        spinsh = apsegshs[i];
        ierr = TetGenMeshFindEdge_face(m, &spinsh, pa, bp);CHKERRQ(ierr);
        ssbond(m, &spinsh, &apseg);
        spinsh = pbsegshs[i];
        ierr = TetGenMeshFindEdge_face(m, &spinsh, bp, pb);CHKERRQ(ierr);
        ssbond(m, &spinsh, &pbseg);
      }
      /*  Bond all subfaces share at a->p together. */
      for(i = 0; i < n; i++) {
        spinsh = apsegshs[i];
        if (i < (n - 1)) {
          casingout = apsegshs[i + 1];
        } else {
          casingout = apsegshs[0];
        }
        sbond1(&spinsh, &casingout);
      }
      /*  Bond all subfaces share at p->b together. */
      for(i = 0; i < n; i++) {
        spinsh = pbsegshs[i];
        if (i < (n - 1)) {
          casingout = pbsegshs[i + 1];
        } else {
          casingout = pbsegshs[0];
        }
        sbond1(&spinsh, &casingout);
      }
      ierr = PetscFree(apsegshs);CHKERRQ(ierr);
      ierr = PetscFree(pbsegshs);CHKERRQ(ierr);

      /*  Check for newly encroached subsegments if the flag is set. */
      if (chkencseg) {
        /*  Check if a->p and p->b are encroached by other vertices. */
        ierr = TetGenMeshCheckSeg4Encroach(m, &apseg, PETSC_NULL, PETSC_NULL, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
        ierr = TetGenMeshCheckSeg4Encroach(m, &pbseg, PETSC_NULL, PETSC_NULL, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
        /*  Check if the adjacent segments are encroached by p. */
        ierr = TetGenMeshTallEncSegs(m, bp, n, ceillists, PETSC_NULL);CHKERRQ(ierr);
      }
    } /*  if (splitseg != (face *) NULL) */

    /*  Delete subfaces of old CBC_i(p)s. */
    for(k = 0; k < n; k++) {
      ierr = ListLength(sublists[k], &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(sublists[k], i, (void **) &oldsh);CHKERRQ(ierr);
        ierr = TetGenMeshShellFaceDealloc(m, m->subfaces, oldsh.sh);CHKERRQ(ierr);
      }
      /*  Clear the list so that the subs will not get unmarked later in */
      /*    routine releasebowatcavity() which only frees the memory. */
      ierr = ListClear(sublists[k]);CHKERRQ(ierr);
      /*  Only do once if p is on a facet. */
      if (!splitseg) break;
    }

    /*  Check for newly encroached subfaces if the flag is set. */
    if (chkencsub) {
      /*  Check if new subfaces of C_i(p) are encroached by other vertices. */
      for(k = 0; k < n; k++) {
        subceillist = subceillists[k];
        ierr = ListLength(subceillist, &len);CHKERRQ(ierr);
        for(i = 0; i < len; i++) {
          ierr = ListItem(subceillist, i, (void **) &newsh);CHKERRQ(ierr);
          ierr = TetGenMeshCheckSub4Encroach(m, &newsh, PETSC_NULL, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
        }
        /*  Only do once if p is on a facet. */
        if (!splitseg) break;
      }
      /*  Check if the adjacent subfaces are encroached by p. */
      ierr = TetGenMeshTallEncSubs(m, bp, n, ceillists, PETSC_NULL);CHKERRQ(ierr);
    }
  } /*  if (subceillists != (list **) NULL) */

  /*  Delete tets of old BC_i(p)s. */
  for(k = 0; k < n; k++) {
    ierr = ListLength(tetlists[k], &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(tetlists[k], i, (void **) &oldtet);CHKERRQ(ierr);
      ierr = TetGenMeshTetrahedronDealloc(m, oldtet.tet);CHKERRQ(ierr);
    }
    /*  Clear the list so that the tets will not get unmarked later in */
    /*    routine releasebowatcavity() which only frees the memory. */
    ierr = ListClear(tetlists[k]);CHKERRQ(ierr);
  }

  /*  check for bad quality tets if the flags is set. */
  if (chkbadtet) {
    for(k = 0; k < n; k++) {
      ceillist = ceillists[k];
      ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(ceillist, i, (void **) &newtet);CHKERRQ(ierr);
        ierr = TetGenMeshCheckTet4BadQual(m, &newtet, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      }
    }
  }

  if (flipque) {
    /*  Newly created internal faces of BC(p) (excluding faces on C(p)s) are */
    /*    in 'flipque'.  Some of these faces may be locally non-Delaunay due */
    /*    to the existence of non-constrained tets. check and fix them. */
    ierr = TetGenMeshLawson3D(m, flipque, PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  flip_cxx ///////////////////////////////////////////////////////////////// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakeSegmentMap"
/*  Create a map from vertex indices to segments, subfaces, and tetrahedra     */
/*  sharing at the same vertices.                                              */
/*                                                                             */
/*  The map is stored in two arrays: 'idx2___list' and '___sperverlist', they  */
/*  form a sparse matrix whose size is (n+1)x(n+1), where n is the number of   */
/*  segments, subfaces, or tetrahedra. 'idx2___list' contains row information  */
/*  and '___sperverlist' contains all non-zero elements.  The i-th entry of    */
/*  'idx2___list' is the starting position of i-th row's non-zero elements in  */
/*  '___sperverlist'.  The number of elements of i-th row is (i+1)-th entry    */
/*  minus i-th entry of 'idx2___list'.                                         */
/*                                                                             */
/*  NOTE: These two arrays will be created inside this routine, don't forget   */
/*  to free them after using.                                                  */
/* tetgenmesh::makesegmentmap() */
PetscErrorCode TetGenMeshMakeSegmentMap(TetGenMesh *m, int **idx2seglistPtr, shellface ***segsperverlistPtr)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  int           *idx2seglist;
  shellface    **segsperverlist;
  shellface     *shloop;
  int            i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constructing mapping from points to segments.\n");
  /*  Create and initialize 'idx2seglist'. */
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &idx2seglist);CHKERRQ(ierr);
  for(i = 0; i < m->points->items + 1; i++) idx2seglist[i] = 0;
  /*  Loop the set of segments once, counter the number of segments sharing each vertex. */
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &shloop);CHKERRQ(ierr);
  while(shloop) {
    /*  Increment the number of sharing segments for each endpoint. */
    for(i = 0; i < 2; i++) {
      j = pointmark(m, (point) shloop[3 + i]) - in->firstnumber;
      idx2seglist[j]++;
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &shloop);CHKERRQ(ierr);
  }
  /*  Calculate the total length of array 'facesperverlist'. */
  j = idx2seglist[0];
  idx2seglist[0] = 0;  /*  Array starts from 0 element. */
  for(i = 0; i < m->points->items; i++) {
    k = idx2seglist[i + 1];
    idx2seglist[i + 1] = idx2seglist[i] + j;
    j = k;
  }
  /*  The total length is in the last unit of idx2seglist. */
  ierr = PetscMalloc(idx2seglist[i] * sizeof(shellface*), &segsperverlist);CHKERRQ(ierr);
  /*  Loop the set of segments again, set the info. of segments per vertex. */
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &shloop);CHKERRQ(ierr);
  while(shloop) {
    for(i = 0; i < 2; i++) {
      j = pointmark(m, (point) shloop[3 + i]) - in->firstnumber;
      segsperverlist[idx2seglist[j]] = shloop;
      idx2seglist[j]++;
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &shloop);CHKERRQ(ierr);
  }
  /*  Contents in 'idx2seglist' are shifted, now shift them back. */
  for(i = m->points->items - 1; i >= 0; i--) {
    idx2seglist[i + 1] = idx2seglist[i];
  }
  idx2seglist[0] = 0;
  *idx2seglistPtr    = idx2seglist;
  *segsperverlistPtr = segsperverlist;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakeTetrahedronMap"
/* tetgenmesh::maketetrahedronmap() */
PetscErrorCode TetGenMeshMakeTetrahedronMap(TetGenMesh *m, int **index2tetlist, tetrahedron ***tetspervertexlist)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  tetrahedron   *tetloop;
  tetrahedron  **tetsperverlist;
  int           *idx2tetlist;
  int            i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constructing mapping from points to tetrahedra.\n");

  /*  Create and initialize 'idx2tetlist'. */
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &idx2tetlist);CHKERRQ(ierr);
  for(i = 0; i < m->points->items + 1; i++) idx2tetlist[i] = 0;

  /*  Loop the set of tetrahedra once, counter the number of tetrahedra */
  /*    sharing each vertex. */
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop);CHKERRQ(ierr);
  while(tetloop) {
    /*  Increment the number of sharing tetrahedra for each endpoint. */
    for(i = 0; i < 4; i++) {
      j = pointmark(m, (point) tetloop[4 + i]) - in->firstnumber;
      idx2tetlist[j]++;
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop);CHKERRQ(ierr);
  }

  /*  Calculate the total length of array 'tetsperverlist'. */
  j = idx2tetlist[0];
  idx2tetlist[0] = 0;  /*  Array starts from 0 element. */
  for(i = 0; i < m->points->items; i++) {
    k = idx2tetlist[i + 1];
    idx2tetlist[i + 1] = idx2tetlist[i] + j;
    j = k;
  }
  /*  The total length is in the last unit of idx2tetlist. */
  ierr = PetscMalloc(idx2tetlist[i] * sizeof(tetrahedron *), &tetsperverlist);CHKERRQ(ierr);
  /*  Loop the set of tetrahedra again, set the info. of tet. per vertex. */
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop);CHKERRQ(ierr);
  while(tetloop) {
    for(i = 0; i < 4; i++) {
      j = pointmark(m, (point) tetloop[4 + i]) - in->firstnumber;
      tetsperverlist[idx2tetlist[j]] = tetloop;
      idx2tetlist[j]++;
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop);CHKERRQ(ierr);
  }
  /*  Contents in 'idx2tetlist' are shifted, now shift them back. */
  for(i = m->points->items - 1; i >= 0; i--) {
    idx2tetlist[i + 1] = idx2tetlist[i];
  }
  idx2tetlist[0] = 0;
  *index2tetlist     = idx2tetlist;
  *tetspervertexlist = tetsperverlist;
  PetscFunctionReturn(0);
}

/*  delaunay_cxx ///////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBTreeSort"
/*  btree_sort()    Sort vertices using a binary space partition (bsp) tree.   */
/* tetgenmesh::btree_sort() */
PetscErrorCode TetGenMeshBTreeSort(TetGenMesh *m, point *vertexarray, int arraysize, int axis, PetscReal bxmin, PetscReal bxmax, PetscReal bymin, PetscReal bymax, PetscReal bzmin, PetscReal bzmax, int depth)
{
  TetGenOpts    *b  = m->b;
  point *leftarray, *rightarray;
  point **pptary, swapvert;
  PetscReal split;
  PetscBool lflag, rflag;
  int i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo2(b->in, "  Depth %d, %d verts.\n", depth, arraysize);
  PetscInfo7(b->in, "  Bbox (%g, %g, %g),(%g, %g, %g). %s-axis\n", bxmin, bymin, bzmin, bxmax, bymax, bzmax, axis == 0 ? "x" : (axis == 1 ? "y" : "z"));

  if (depth > m->max_btree_depth) {
    m->max_btree_depth = depth;
  }

  if (axis == 0) {
    /*  Split along x-axis. */
    split = 0.5 * (bxmin + bxmax);
  } else if (axis == 1) {
    /*  Split along y-axis. */
    split = 0.5 * (bymin + bymax);
  } else {
    /*  Split along z-axis. */
    split = 0.5 * (bzmin + bzmax);
  }

  i = 0;
  j = arraysize - 1;

  /*  Partition the vertices into left- and right-arraies. */
  do {
    for (; i < arraysize; i++) {
      if (vertexarray[i][axis] >= split) {
        break;
      }
    }
    for (; j >= 0; j--) {
      if (vertexarray[j][axis] < split) {
        break;
      }
    }
    /*  Is the partition finished? */
    if (i == (j + 1)) {
      break;
    }
    /*  Swap i-th and j-th vertices. */
    swapvert = vertexarray[i];
    vertexarray[i] = vertexarray[j];
    vertexarray[j] = swapvert;
    /*  Continue patitioning the array; */
  } while (1);

  PetscInfo2(b->in, "    leftsize = %d, rightsize = %d\n", i, arraysize - i);
  lflag = rflag = PETSC_FALSE;

  /*  if (depth < max_tree_depth) { */
    if (i > b->max_btreenode_size) {
      /*  Recursively partition the left array (length = i). */
      if (axis == 0) { /*  x */
        ierr = TetGenMeshBTreeSort(m, vertexarray, i, (axis + 1) % 3, bxmin, split, bymin, bymax, bzmin, bzmax, depth + 1);CHKERRQ(ierr);
      } else if (axis == 1) { /*  y */
        ierr = TetGenMeshBTreeSort(m, vertexarray, i, (axis + 1) % 3, bxmin, bxmax, bymin, split, bzmin, bzmax, depth + 1);CHKERRQ(ierr);
      } else { /*  z */
        ierr = TetGenMeshBTreeSort(m, vertexarray, i, (axis + 1) % 3, bxmin, bxmax, bymin, bymax, bzmin, split, depth + 1);CHKERRQ(ierr);
      }
    } else {
      lflag = PETSC_TRUE;
    }
    if ((arraysize - i) > b->max_btreenode_size) {
      /*  Recursively partition the right array (length = arraysize - i). */
      if (axis == 0) { /*  x */
        ierr = TetGenMeshBTreeSort(m, &(vertexarray[i]), arraysize - i, (axis + 1) % 3, split, bxmax, bymin, bymax, bzmin, bzmax, depth + 1);CHKERRQ(ierr);
      } else if (axis == 1) { /*  y */
        ierr = TetGenMeshBTreeSort(m, &(vertexarray[i]), arraysize - i, (axis + 1) % 3, bxmin, bxmax, split, bymax, bzmin, bzmax, depth + 1);CHKERRQ(ierr);
      } else { /*  z */
        ierr = TetGenMeshBTreeSort(m, &(vertexarray[i]), arraysize - i, (axis + 1) % 3, bxmin, bxmax, bymin, bymax, split, bzmax, depth + 1);CHKERRQ(ierr);
      }
    } else {
      rflag = PETSC_TRUE;
    }

  if (lflag && (i > 0)) {
    /*  Remember the maximal length of the partitions. */
    if (i > m->max_btreenode_size) {
      m->max_btreenode_size = i;
    }
    /*  Allocate space for the left array (use the first entry to save */
    /*    the length of this array). */
    ierr = PetscMalloc((i + 1) * sizeof(point), &leftarray);CHKERRQ(ierr);
    leftarray[0] = (point) (PETSC_UINTPTR_T) i; /*  The array length. */
    /*  Put all points in this array. */
    for(k = 0; k < i; k++) {
      leftarray[k + 1] = vertexarray[k];
      setpoint2ppt(m, leftarray[k + 1], (point) leftarray);
    }
    /*  Save this array in list. */
    ierr = ArrayPoolNewIndex(m->btreenode_list, (void **) &pptary, PETSC_NULL);CHKERRQ(ierr);
    *pptary = leftarray;
  }

  /*  Get the length of the right array. */
  j = arraysize - i;
  if (rflag && (j > 0)) {
    if (j > m->max_btreenode_size) {
      m->max_btreenode_size = j;
    }
    /*  Allocate space for the right array (use the first entry to save */
    /*    the length of this array). */
    ierr = PetscMalloc((j + 1) * sizeof(point), &rightarray);CHKERRQ(ierr);
    rightarray[0] = (point) (PETSC_UINTPTR_T) j; /*  The array length. */
    /*  Put all points in this array. */
    for (k = 0; k < j; k++) {
      rightarray[k + 1] = vertexarray[i + k];
      setpoint2ppt(m, rightarray[k + 1], (point) rightarray);
    }
    /*  Save this array in list. */
    ierr = ArrayPoolNewIndex(m->btreenode_list, (void **) &pptary, PETSC_NULL);CHKERRQ(ierr);
    *pptary = rightarray;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBTreeInsert"
/*  btree_insert()    Add a vertex into a tree node.                           */
/* tetgenmesh::btree_insert() */
PetscErrorCode TetGenMeshBTreeInsert(TetGenMesh *m, point insertpt)
{
  point *ptary;
  long arylen; /*  The array lenhgth is saved in ptary[0]. */

  PetscFunctionBegin;
  /*  Get the tree node (save in this point). */
  ptary = (point *) point2ppt(m, insertpt);
  /*  Get the current array length. */
  arylen = (long) ptary[0];
  /*  Insert the point into the node. */
  ptary[arylen + 1] = insertpt;
  /*  Increase the array length by 1. */
  ptary[0] = (point) (arylen + 1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBTreeSearch"
/*  btree_search()    Search a near point for an inserting point.              */
/* tetgenmesh::btree_search() */
PetscErrorCode TetGenMeshBTreeSearch(TetGenMesh *m, point insertpt, triface *searchtet)
{
  TetGenOpts    *b  = m->b;
  point *ptary;
  point nearpt, candpt;
  PetscReal dist2, mindist2;
  int ptsamples, ptidx;
  long arylen;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Get the tree node (save in this point). */
  ptary = (point *) point2ppt(m, insertpt);
  /*  Get the current array length. */
  arylen = (long) ptary[0];

  if (arylen == 0) {
    searchtet->tet = PETSC_NULL;
    PetscFunctionReturn(0);
  }

  if (arylen < 10) {
    ptsamples = arylen;
  } else {
    ptsamples = 10; /*  Take at least 10 samples. */
    /*    The number of random samples taken is proportional to the third root */
    /*    of the number of points in the cell. */
    while (ptsamples * ptsamples * ptsamples < arylen) {
      ptsamples++;
    }
  }

  /*  Select "good" candidate using k random samples, taking the closest one. */
  mindist2 = 1.79769E+308; /*  The largest double value (8 byte). */
  nearpt = PETSC_NULL;

  for(i = 0; i < ptsamples; i++) {
    ierr = TetGenMeshRandomChoice(m, arylen, &ptidx);CHKERRQ(ierr);
    candpt = ptary[ptidx + 1];
    dist2 = (candpt[0] - insertpt[0]) * (candpt[0] - insertpt[0])
          + (candpt[1] - insertpt[1]) * (candpt[1] - insertpt[1])
          + (candpt[2] - insertpt[2]) * (candpt[2] - insertpt[2]);
    if (dist2 < mindist2) {
      mindist2 = dist2;
      nearpt = candpt;
    }
  }

  PetscInfo2(b->in, "    Get point %d (cell size %ld).\n", pointmark(m, nearpt), arylen);

  decode(point2tet(m, nearpt), searchtet);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshOrderVertices"
/*  ordervertices()    Order the vertices for incremental inserting.           */
/*                                                                             */
/*  We assume the vertices have been sorted by a binary tree.                  */
/* tetgenmesh::ordervertices() */
PetscErrorCode TetGenMeshOrderVertices(TetGenMesh *m, point *vertexarray, int arraysize)
{
  point **ipptary, **jpptary, *swappptary;
  point *ptary;
  long arylen;
  int index, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  First pick one vertex from each tree node. */
  for(i = 0; i < (int) m->btreenode_list->objects; i++) {
    ipptary = (point **) fastlookup(m->btreenode_list, i);
    ptary = *ipptary;
    vertexarray[i] = ptary[1]; /*  Skip the first entry. */
  }

  index = i;
  /*  Then put all other points in the array node by node. */
  for(i = (int) m->btreenode_list->objects - 1; i >= 0; i--) {
    /*  Randomly pick a tree node. */
    ierr = TetGenMeshRandomChoice(m, i + 1, &j);CHKERRQ(ierr);
    /*  Save the i-th node. */
    ipptary = (point **) fastlookup(m->btreenode_list, i);
    /*  Get the j-th node. */
    jpptary = (point **) fastlookup(m->btreenode_list, j);
    /*  Order the points in the node. */
    ptary = *jpptary;
    arylen = (long) ptary[0];
    for(j = 2; j <= arylen; j++) { /*  Skip the first point. */
      vertexarray[index] = ptary[j];
      index++;
    }
    /*  Clear this tree node. */
    ptary[0] = (point) 0;
    /*  Swap i-th node to j-th node. */
    swappptary = *ipptary;
    *ipptary = *jpptary; /*  [i] <= [j] */
    *jpptary = swappptary; /*  [j] <= [i] */
  }

  /*  Make sure we've done correctly. */
  if (index != arraysize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %d should match array size %d", index, arraysize);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInsertVertexBW"
/*  insertvertexbw()    Insert a vertex using the Boywer-Watson algorithm.     */
/*                                                                             */
/*  The point p will be first located in T. 'searchtet' is a suggested start-  */
/*  tetrahedron, it can be NULL. Note that p may lies outside T. In such case, */
/*  the convex hull of T will be updated to include p as a vertex.             */
/*                                                                             */
/*  If 'bwflag' is TRUE, the Bowyer-Watson algorithm is used to recover the    */
/*  Delaunayness of T. Otherwise, do nothing with regard to the Delaunayness   */
/*  T (T may be non-Delaunay after this function).                             */
/*                                                                             */
/*  If 'visflag' is TRUE, force to check the visibility of the boundary faces  */
/*  of cavity. This is needed when T is not Delaunay.                          */
/*                                                                             */
/*  If 'noencflag' is TRUE, only insert the new point p if it does not cause   */
/*  any existing (sub)segment be non-Delaunay. This option only is checked     */
/*  when the global variable 'checksubsegs' is set.                            */
/* tetgenmesh::insertvertexbw() */
PetscErrorCode TetGenMeshInsertVertexBW(TetGenMesh *m, point insertpt, triface *searchtet, PetscBool bwflag, PetscBool visflag, PetscBool noencsegflag, PetscBool noencsubflag, locateresult *result)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  triface neightet = {PETSC_NULL, 0, 0}, spintet = {PETSC_NULL, 0, 0}, newtet = {PETSC_NULL, 0, 0}, neineitet = {PETSC_NULL, 0, 0};
  triface *cavetet, *parytet, *parytet1;
  face checksh  = {PETSC_NULL, 0}, *pssub;
  face checkseg = {PETSC_NULL, 0}, *paryseg;
  point pa, pb, pc, *ppt;
  locateresult loc;
  PetscReal attrib, volume;
  PetscReal sign, ori;
  long tetcount;
  PetscBool enqflag;
  int hitbdry;
  int i, j;
  ArrayPool *swaplist; /*  for updating cavity. */
  long updatecount;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo1(b->in, "    Insert point %d\n", pointmark(m, insertpt));

  tetcount = m->ptloc_count;
  updatecount = 0;

  /*  Locate the point. */
  if (!searchtet->tet) {
    if (m->btreenode_list) { /*  default option */
      /*  Use bsp-tree to select a starting tetrahedron. */
      ierr = TetGenMeshBTreeSearch(m, insertpt, searchtet);CHKERRQ(ierr);
    } else { /*  -u0 option */
      /*  Randomly select a starting tetrahedron. */
      ierr = TetGenMeshRandomSample(m, insertpt, searchtet);CHKERRQ(ierr);
    }
    ierr = TetGenMeshPreciseLocate(m, insertpt, searchtet, m->tetrahedrons->items, &loc);CHKERRQ(ierr);
  } else {
    /*  Start from 'searchtet'. */
    ierr = TetGenMeshLocate2(m, insertpt, searchtet, PETSC_NULL, &loc);CHKERRQ(ierr);
  }

  PetscInfo1(b->in, "    Walk distance (# tets): %ld\n", m->ptloc_count - tetcount);

  if (m->ptloc_max_count < (m->ptloc_count - tetcount)) {
    m->ptloc_max_count = (m->ptloc_count - tetcount);
  }

  PetscInfo5(b->in, "    Located (%d) tet (%d, %d, %d, %d).\n", (int) loc, pointmark(m, org(searchtet)), pointmark(m, dest(searchtet)), pointmark(m, apex(searchtet)), pointmark(m, oppo(searchtet)));

  if (loc == ONVERTEX) {
    /*  The point already exists. Mark it and do nothing on it. */
    if (b->object != TETGEN_OBJECT_STL) {
      PetscInfo2(b->in, "Warning:  Point #%d is duplicated with Point #%d. Ignored!\n", pointmark(m, insertpt), pointmark(m, org(searchtet)));
    }
    setpoint2ppt(m, insertpt, org(searchtet));
    setpointtype(m, insertpt, DUPLICATEDVERTEX);
    m->dupverts++;
    if (result) {*result = loc;}
    PetscFunctionReturn(0);
  }

  tetcount = 0l;  /*  The number of deallocated tets. */

  /*  Create the initial boundary of the cavity. */
  if (loc == INTETRAHEDRON) {
    /*  Add four boundary faces of this tet into list. */
    neightet.tet = searchtet->tet;
    for(neightet.loc = 0; neightet.loc < 4; neightet.loc++) {
      ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      *parytet = neightet;
    }
    infect(m, searchtet);
    ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
    *parytet = *searchtet;
    tetcount++;
    m->flip14count++;
  } else if (loc == ONFACE) {
    /*  Add at most six boundary faces into list. */
    neightet.tet = searchtet->tet;
    for(i = 0; i < 3; i++) {
      neightet.loc = locpivot[searchtet->loc][i];
      ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      *parytet = neightet;
    }
    infect(m, searchtet);
    ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
    *parytet = *searchtet;
    tetcount++;
    decode(searchtet->tet[searchtet->loc], &spintet);
    if (spintet.tet != m->dummytet) {
      neightet.tet = spintet.tet;
      for(i = 0; i < 3; i++) {
        neightet.loc = locpivot[spintet.loc][i];
        ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
        *parytet = neightet;
      }
      infect(m, &spintet);
      ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      *parytet = spintet;
      tetcount++;
    } else {
      /*  Split a hull face into three hull faces. */
      m->hullsize += 2;
    }
    m->flip26count++;
  } else if (loc == ONEDGE) {
    /*  Add all adjacent boundary tets into list. */
    spintet = *searchtet;
    pc = apex(&spintet);
    hitbdry = 0;
    do {
      tetcount++;
      neightet.tet = spintet.tet;
      neightet.loc = locverpivot[spintet.loc][spintet.ver][0];
      ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      *parytet = neightet;
      neightet.loc = locverpivot[spintet.loc][spintet.ver][1];
      ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      *parytet = neightet;
      infect(m, &spintet);
      ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      *parytet = spintet;
      /*  Go to the next tet (may be dummytet). */
      tfnext(m, &spintet, &neightet);
      if (neightet.tet == m->dummytet) {
        hitbdry++;
        if (hitbdry == 2) break;
        esym(searchtet, &spintet); /*  Go to another direction. */
        tfnext(m, &spintet, &neightet);
        if (neightet.tet == m->dummytet) break;
      }
      spintet = neightet;
    } while (apex(&spintet) != pc);
    /*  Update hull size if it is a hull edge. */
    if (hitbdry > 0) {
      /*  Split a hull edge deletes two hull faces, adds four new hull faces. */
      m->hullsize += 2;
    }
    m->flipn2ncount++;
  } else if (loc == OUTSIDE) {
    /*  p lies outside the convex hull. Enlarge the convex hull by including p. */
    PetscInfo(b->in, "    Insert a hull vertex.\n");
    /*  'searchtet' refers to a hull face which is visible by p. */
    adjustedgering_triface(searchtet, CW);
    /*  Create the first tet t (from f and p). */
    ierr = TetGenMeshMakeTetrahedron(m, &newtet);CHKERRQ(ierr);
    setorg (&newtet, org(searchtet));
    setdest(&newtet, dest(searchtet));
    setapex(&newtet, apex(searchtet));
    setoppo(&newtet, insertpt);
    for(i = 0; i < in->numberoftetrahedronattributes; i++) {
      attrib = elemattribute(m, searchtet->tet, i);
      setelemattribute(m, newtet.tet, i, attrib);
    }
    if (b->varvolume) {
      volume = volumebound(m, searchtet->tet);
      setvolumebound(m, newtet.tet, volume);
    }
    /*  Connect t to T. */
    bond(m, &newtet, searchtet);
    /*  Removed a hull face, added three "new hull faces". */
    m->hullsize += 2;

    /*  Add a cavity boundary face. */
    ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
    *parytet = newtet;
    /*  Add a cavity tet. */
    infect(m, &newtet);
    ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
    *parytet = newtet;
    tetcount++;

    /*  Add three "new hull faces" into list (re-use cavebdrylist). */
    newtet.ver = 0;
    for(i = 0; i < 3; i++) {
      fnext(m, &newtet, &neightet);
      ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      *parytet = neightet;
      enextself(&newtet);
    }

    /*  Find all actual new hull faces. */
    for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
      /*  Get a queued "new hull face". */
      parytet = (triface *) fastlookup(m->cavebdrylist, i);
      /*  Every "new hull face" must have p as its apex. */
      if (apex(parytet) != insertpt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      if ((parytet->ver & 1) != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong"); /*  It's CW edge ring. */
      /*  Check if it is still a hull face. */
      sym(parytet, &neightet);
      if (neightet.tet == m->dummytet) {
        /*  Yes, get its adjacent hull face (at its edge). */
        esym(parytet, &neightet);
        while (1) {
          fnextself(m, &neightet);
          /*  Does its adjacent tet exist? */
          sym(&neightet, &neineitet);
          if (neineitet.tet == m->dummytet) break;
          symedgeself(m, &neightet);
        }
        /*  neightet is an adjacent hull face. */
        pc = apex(&neightet);
        if (pc != insertpt) {
          /*  Check if p is visible by the hull face ('neightet'). */
          pa = org(&neightet);
          pb = dest(&neightet);
          ori = TetGenOrient3D(pa, pb, pc, insertpt); m->orient3dcount++;
          if (ori < 0) {
            /*  Create a new tet adjacent to neightet. */
            ierr = TetGenMeshMakeTetrahedron(m, &newtet);CHKERRQ(ierr);
            setorg (&newtet, pa);
            setdest(&newtet, pb);
            setapex(&newtet, pc);
            setoppo(&newtet, insertpt);
            for(j = 0; j < in->numberoftetrahedronattributes; j++) {
              attrib = elemattribute(m, neightet.tet, j);
              setelemattribute(m, newtet.tet, j, attrib);
            }
            if (b->varvolume) {
              volume = volumebound(m, neightet.tet);
              setvolumebound(m, newtet.tet, volume);
            }
            bond(m, &newtet, &neightet);
            fnext(m, &newtet, &neineitet);
            bond(m, &neineitet, parytet);
            /*  Comment: We removed two hull faces, and added two "new hull */
            /*    faces", hence hullsize remains unchanged. */
            /*  Add a cavity boundary face. */
            ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
            *parytet1 = newtet;
            /*  Add a cavity tet. */
            infect(m, &newtet);
            ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
            *parytet1 = newtet;
            tetcount++;
            /*  Add two "new hull faces" into list. */
            enextself(&newtet);
            for(j = 0; j < 2; j++) {
              fnext(m, &newtet, &neineitet);
              ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
              *parytet1 = neineitet;
              enextself(&newtet);
            }
          }
        } else {
          /*  Two hull faces matched. Bond the two adjacent tets. */
          bond(m, parytet, &neightet);
          m->hullsize -= 2;
        }
      } /*  if (neightet.tet == dummytet) */
    } /*  i */
    ierr = ArrayPoolRestart(m->cavebdrylist);CHKERRQ(ierr);
    m->inserthullcount++;
  }

  if (!bwflag) {
    if (result) {*result = loc;}
    PetscFunctionReturn(0);
  }

  /*  Form the Boywer-Watson cavity. */
  for (i = 0; i < (int) m->cavetetlist->objects; i++) {
    /*  Get a cavity boundary face. */
    parytet = (triface *) fastlookup(m->cavetetlist, i);
    if (parytet->tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    if (!infected(m, parytet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong"); /*  The tet is inside the cavity. */
    enqflag = PETSC_FALSE;
    /*  Get the adjacent tet. */
    sym(parytet, &neightet);
    if (neightet.tet != m->dummytet) {
      if (!infected(m, &neightet)) {
        if (!marktested(m, &neightet)) {
          ppt = (point *) &(neightet.tet[4]);
          ierr = TetGenMeshInSphereS(m, ppt[0], ppt[1], ppt[2], ppt[3], insertpt, &sign);CHKERRQ(ierr);
          enqflag = (sign < 0.0) ? PETSC_TRUE : PETSC_FALSE;
          /*  Avoid redundant insphere tests. */
          marktest(m, &neightet);
        }
      } else {
        enqflag = PETSC_TRUE;
      }
    }
    if (enqflag) { /*  Found a tet in the cavity. */
      if (!infected(m, &neightet)) { /*  Avoid to add it multiple times. */
        /*  Put other three faces in check list. */
        neineitet.tet = neightet.tet;
        for(j = 0; j < 3; j++) {
          neineitet.loc = locpivot[neightet.loc][j];
          ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
          *parytet1 = neineitet;
        }
        infect(m, &neightet);
        ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
        *parytet1 = neightet;
        tetcount++;
      }
    } else {
      /*  Found a boundary face of the cavity. */
      if (neightet.tet == m->dummytet) {
        /*  Check for a possible flat tet (see m27.node, use -J option). */
        pa = org(parytet);
        pb = dest(parytet);
        pc = apex(parytet);
        ori = TetGenOrient3D(pa, pb, pc, insertpt);
        if (ori != 0) {
          ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
          *parytet1 = *parytet;
          /*  futureflip = flippush(futureflip, parytet, insertpt); */
        }
      } else {
        ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
        *parytet1 = *parytet;
      }
    }
  } /*  i */

  PetscInfo2(b->in, "    Cavity formed: %ld tets, %ld faces.\n", tetcount, m->cavebdrylist->objects);

  m->totaldeadtets += tetcount;
  m->totalbowatcavsize += m->cavebdrylist->objects;
  if (m->maxbowatcavsize < (long) m->cavebdrylist->objects) {
    m->maxbowatcavsize = m->cavebdrylist->objects;
  }

  if (m->checksubsegs || noencsegflag) {
    /*  Check if some (sub)segments are inside the cavity. */
    for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
      parytet = (triface *) fastlookup(m->caveoldtetlist, i);
      for (j = 0; j < 6; j++) {
        parytet->loc = edge2locver[j][0];
        parytet->ver = edge2locver[j][1];
        tsspivot1(m, parytet, &checkseg);
        if ((checkseg.sh != m->dummysh) && !sinfected(m, &checkseg)) {
          /*  Check if this segment is inside the cavity. */
          spintet = *parytet;
          pa = apex(&spintet);
          enqflag = PETSC_TRUE;
          hitbdry = 0;
          while (1) {
            tfnextself(m, &spintet);
            if (spintet.tet == m->dummytet) {
              hitbdry++;
              if (hitbdry == 2) break;
              esym(parytet, &spintet);
              tfnextself(m, &spintet);
              if (spintet.tet == m->dummytet) break;
            }
            if (!infected(m, &spintet)) {
              enqflag = PETSC_FALSE; break; /*  It is not inside. */
            }
            if (apex(&spintet) == pa) break;
          }
          if (enqflag) {
            PetscInfo2(b->in, "      Queue a missing segment (%d, %d).\n", pointmark(m, sorg(&checkseg)), pointmark(m, sdest(&checkseg)));
            sinfect(m, &checkseg);  /*  Only save it once. */
            ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &paryseg, PETSC_NULL);CHKERRQ(ierr);
            *paryseg = checkseg;
          }
        }
      }
    }
  }

  if (noencsegflag && (m->subsegstack->objects > 0)) {
    /*  Found encroached subsegments! Do not insert this point. */
    for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
      parytet = (triface *) fastlookup(m->caveoldtetlist, i);
      uninfect(m, parytet);
      unmarktest(m, parytet);
    }
    /*  Unmark cavity neighbor tets (outside the cavity). */
    for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
      parytet = (triface *) fastlookup(m->cavebdrylist, i);
      sym(parytet, &neightet);
      if (neightet.tet != m->dummytet) {
        unmarktest(m, &neightet);
      }
    }
    ierr = ArrayPoolRestart(m->cavetetlist);CHKERRQ(ierr);
    ierr = ArrayPoolRestart(m->cavebdrylist);CHKERRQ(ierr);
    ierr = ArrayPoolRestart(m->caveoldtetlist);CHKERRQ(ierr);
    if (result) {*result = ENCSEGMENT;}
    PetscFunctionReturn(0);
  }

  if (m->checksubfaces || noencsubflag) {
    /*  Check if some subfaces are inside the cavity. */
    for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
      parytet = (triface *) fastlookup(m->caveoldtetlist, i);
      neightet.tet = parytet->tet;
      for(neightet.loc = 0; neightet.loc < 4; neightet.loc++) {
        tspivot(m, &neightet, &checksh);
        if (checksh.sh != m->dummysh) {
          sym(&neightet, &neineitet);
          /*  Do not check it if it is a hull tet. */
          if (neineitet.tet != m->dummytet) {
            if (infected(m, &neineitet)) {
              PetscInfo3(b->in, "      Queue a missing subface (%d, %d, %d).\n", pointmark(m, sorg(&checksh)), pointmark(m, sdest(&checksh)), pointmark(m, sapex(&checksh)));
              tsdissolve(m, &neineitet); /*  Disconnect a tet-sub bond. */
              stdissolve(m, &checksh); /*  Disconnect the sub-tet bond. */
              sesymself(&checksh);
              stdissolve(m, &checksh);
              /*  Add the missing subface into list. */
              ierr = ArrayPoolNewIndex(m->subfacstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
              *pssub = checksh;
            }
          }
        }
      }
    }
  }

  if (noencsubflag && (m->subfacstack->objects > 0)) {
    /*  Found encroached subfaces! Do not insert this point. */
  }

  if (visflag) {
    /*  If T is not a Delaunay triangulation, the formed cavity may not be */
    /*    star-shaped (fig/dump-cavity-case8). Validation is needed. */
    ierr = ArrayPoolRestart(m->cavetetlist);CHKERRQ(ierr);
    for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
      cavetet = (triface *) fastlookup(m->cavebdrylist, i);
      if (infected(m, cavetet)) {
        sym(cavetet, &neightet);
        if (neightet.tet == m->dummytet || !infected(m, &neightet)) {
          if (neightet.tet != m->dummytet) {
            cavetet->ver = 4; /*  CCW edge ring. */
            pa = dest(cavetet);
            pb = org(cavetet);
            pc = apex(cavetet);
            ori = TetGenOrient3D(pa, pb, pc, insertpt); m->orient3dcount++;
            if (ori == 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            enqflag = (ori > 0.0) ? PETSC_TRUE : PETSC_FALSE;
          } else {
            enqflag = PETSC_TRUE; /*  A hull face. */
          }
          if (enqflag) {
            /*  This face is valid, save it. */
            ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
            *parytet = *cavetet;
          } else {
            PetscInfo4(b->in, "    Cut tet (%d, %d, %d, %d)\n", pointmark(m, pb), pointmark(m, pa), pointmark(m, pc), pointmark(m, oppo(cavetet)));
            uninfect(m, cavetet);
            unmarktest(m, cavetet);
            if (neightet.tet != m->dummytet) {
              unmarktest(m, &neightet);
            }
            updatecount++;
            /*  Add three new faces to find new boundaries. */
            for(j = 0; j < 3; j++) {
              fnext(m, cavetet, &neineitet);
              sym(&neineitet, &neightet);
              if (neightet.tet != m->dummytet) {
                if (infected(m, &neightet)) {
                  neightet.ver = 4;
                  ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
                  *parytet = neightet;
                } else {
                  unmarktest(m, &neightet);
                }
              }
              enextself(cavetet);
            }
          }
        } else {
          /*  This face is not on the cavity boundary anymore. */
          unmarktest(m, cavetet);
        }
      } else {
        if (marktested(m, cavetet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      }
    }
    if (updatecount > 0) {
      /*  Update the cavity boundary faces (fig/dump-cavity-case9). */
      ierr = ArrayPoolRestart(m->cavebdrylist);CHKERRQ(ierr);
      for(i = 0; i < (int) m->cavetetlist->objects; i++) {
        cavetet = (triface *) fastlookup(m->cavetetlist, i);
        /*  'cavetet' was boundary face of the cavity. */
        if (infected(m, cavetet)) {
          sym(cavetet, &neightet);
          if ((neightet.tet != m->dummytet) || !infected(m, &neightet)) {
            /*  It is a cavity boundary face. */
            ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
            *parytet = *cavetet;
          } else {
            /*  Not a cavity boundary face. */
            unmarktest(m, cavetet);
          }
        } else {
          if (marktested(m, cavetet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        }
      }
      /*  Update the list of old tets. */
      ierr = ArrayPoolRestart(m->cavetetlist);CHKERRQ(ierr);
      for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
        cavetet = (triface *) fastlookup(m->caveoldtetlist, i);
        if (infected(m, cavetet)) {
          ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = *cavetet;
        }
      }
      if ((int) m->cavetetlist->objects >= i) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      /*  Swap 'm->cavetetlist' and 'm->caveoldtetlist'. */
      swaplist = m->caveoldtetlist;
      m->caveoldtetlist = m->cavetetlist;
      m->cavetetlist = swaplist;
      PetscInfo2(b->in, "    Size of the updated cavity: %d faces %d tets.\n", (int) m->cavebdrylist->objects, (int) m->caveoldtetlist->objects);
    }
  }

  /*  Re-use this list for new cavity faces. */
  ierr = ArrayPoolRestart(m->cavetetlist);CHKERRQ(ierr);

  /*  Create new tetrahedra in the Bowyer-Watson cavity and Connect them. */
  for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
    parytet = (triface *) fastlookup(m->cavebdrylist, i);
    if (!infected(m, parytet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong"); /*  The tet is inside the cavity. */
    parytet->ver = 0; /*  In CCW edge ring. */
    ierr = TetGenMeshMakeTetrahedron(m, &newtet);CHKERRQ(ierr);
    setorg (&newtet, org(parytet));
    setdest(&newtet, dest(parytet));
    setapex(&newtet, apex(parytet));
    setoppo(&newtet, insertpt);
    for(j = 0; j < in->numberoftetrahedronattributes; j++) {
      attrib = elemattribute(m, parytet->tet, j);
      setelemattribute(m, newtet.tet, j, attrib);
    }
    if (b->varvolume) {
      volume = volumebound(m, parytet->tet);
      setvolumebound(m, newtet.tet, volume);
    }
    /*  Bond the new tet to the adjacent tet outside the cavity. */
    sym(parytet, &neightet);
    if (neightet.tet != m->dummytet) {
      /*  The tet was marked (to avoid redundant insphere tests). */
      unmarktest(m, &neightet);
      bond(m, &newtet, &neightet);
    } else {
      /*  Bond newtet to dummytet. */
      m->dummytet[0] = encode(&newtet);
    }
    /*  mark the other three faces of this tet as "open". */
    neightet.tet = newtet.tet;
    for(j = 0; j < 3; j++) {
      neightet.tet[locpivot[0][j]] = PETSC_NULL;
    }
    /*  Let the oldtet knows newtet (for connecting adjacent new tets). */
    parytet->tet[parytet->loc] = encode(&newtet);
    if (m->checksubsegs) {
      /*  newtet and parytet share at the same edge. */
      for(j = 0; j < 3; j++) {
        tsspivot1(m, parytet, &checkseg);
        if (checkseg.sh != m->dummysh) {
          if (sinfected(m, &checkseg)) {
            /*  This subsegment is not missing. Unmark it. */
            PetscInfo2(b->in, "      Dequeue a segment (%d, %d).\n", pointmark(m, sorg(&checkseg)), pointmark(m, sdest(&checkseg)));
            suninfect(m, &checkseg); /*  Dequeue a non-missing segment. */
          }
          tssbond1(m, &newtet, &checkseg);
        }
        enextself(parytet);
        enextself(&newtet);
      }
    }
    if (m->checksubfaces) {
      /*  Bond subface to the new tet. */
      tspivot(m, parytet, &checksh);
      if (checksh.sh != m->dummysh) {
        tsbond(m, &newtet, &checksh);
        /*  The other-side-connection of checksh should be no change. */
      }
    }
  } /*  i */

  /*  Set a handle for speeding point location. */
  m->recenttet = newtet;
  setpoint2tet(m, insertpt, encode(&newtet));

  /*  Connect adjacent new tetrahedra together. Here we utilize the connections */
  /*    of the old cavity tets to find the new adjacent tets. */
  for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
    parytet = (triface *) fastlookup(m->cavebdrylist, i);
    decode(parytet->tet[parytet->loc], &newtet);
    /*  assert(org(newtet) == org(*parytet));  PETSC_USE_DEBUG */
    /*  assert((newtet.ver & 1) == 0);  in CCW edge ring. */
    for(j = 0; j < 3; j++) {
      fnext(m, &newtet, &neightet); /*  Go to the "open" face. */
      if (neightet.tet[neightet.loc] == PETSC_NULL) {
        spintet = *parytet;
        while (1) {
          fnextself(m, &spintet);
          symedgeself(m, &spintet);
          if (spintet.tet == m->dummytet) break;
          if (!infected(m, &spintet)) break;
        }
        if (spintet.tet != m->dummytet) {
          /*  'spintet' is the adjacent tet of the cavity. */
          fnext(m, &spintet, &neineitet);
          if (neineitet.tet[neineitet.loc]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          bond(m, &neightet, &neineitet);
        } else {
          /*  This side is a hull face. */
          neightet.tet[neightet.loc] = (tetrahedron) m->dummytet;
          m->dummytet[0] = encode(&neightet);
        }
      }
      setpoint2tet(m, org(&newtet), encode(&newtet));
      enextself(&newtet);
      enextself(parytet);
    }
  }

  /*  Delete the old cavity tets. */
  for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
    parytet = (triface *) fastlookup(m->caveoldtetlist, i);
    ierr = TetGenMeshTetrahedronDealloc(m, parytet->tet);CHKERRQ(ierr);
  }

  /*  Set the point type. */
  if (pointtype(m, insertpt) == UNUSEDVERTEX) {
    setpointtype(m, insertpt, FREEVOLVERTEX);
  }

  if (m->btreenode_list) {
    ierr = TetGenMeshBTreeInsert(m, insertpt);CHKERRQ(ierr);
  }

  ierr = ArrayPoolRestart(m->cavetetlist);CHKERRQ(ierr);
  ierr = ArrayPoolRestart(m->cavebdrylist);CHKERRQ(ierr);
  ierr = ArrayPoolRestart(m->caveoldtetlist);CHKERRQ(ierr);

  if (result) {*result = loc;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDelaunayIncrFlip"
/* tetgenmesh::incrflipdelaunay() */
PetscErrorCode TetGenMeshDelaunayIncrFlip(TetGenMesh *m, triface *oldtet, point *insertarray, long arraysize, PetscBool jump, PetscBool merge, PetscReal eps, Queue *flipque)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  triface        newtet = {PETSC_NULL, 0, 0}, searchtet = {PETSC_NULL, 0, 0};
  point          swappt, lastpt;
  locateresult   loc;
  PetscReal      det = 0.0, attrib, volume;
  int            i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  The initial tetrahedralization T only has one tet formed by 4 affinely */
  /*    linear independent vertices of the point set V = 'insertarray'. The */
  /*    first point a = insertarray[0]. */

  /*  Get the second point b, that is not identical or very close to a. */
  for(i = 1; i < arraysize; i++) {
    det = distance(insertarray[0], insertarray[i]);
    if (det > (m->longest * eps)) break;
  }
  if (i == arraysize) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All points seem to be identical,");
  else {
    /*  Swap to move b from index i to index 1. */
    swappt         = insertarray[i];
    insertarray[i] = insertarray[1];
    insertarray[1] = swappt;
  }
  /*  Get the third point c, that is not collinear with a and b. */
  for(i++; i < arraysize; i++) {
    PetscBool co;

    ierr = TetGenMeshIsCollinear(m, insertarray[0], insertarray[1], insertarray[i], eps, &co);CHKERRQ(ierr);
    if (!co) break;
  }
  if (i == arraysize) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All points seem to be collinear.");
  else {
    /*  Swap to move c from index i to index 2. */
    swappt         = insertarray[i];
    insertarray[i] = insertarray[2];
    insertarray[2] = swappt;
  }
  /*  Get the fourth point d, that is not coplanar with a, b, and c. */
  for(i++; i < arraysize; i++) {
    PetscBool co;

    det = TetGenOrient3D(insertarray[0], insertarray[1], insertarray[2], insertarray[i]);
    if (det == 0.0) continue;
    ierr = TetGenMeshIsCoplanar(m, insertarray[0], insertarray[1], insertarray[2], insertarray[i], det, eps, &co);CHKERRQ(ierr);
    if (!co) break;
  }
  if (i == arraysize) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All points seem to be coplanar.");
  else {
    /*  Swap to move d from index i to index 3. */
    swappt         = insertarray[i];
    insertarray[i] = insertarray[3];
    insertarray[3] = swappt;
    lastpt = insertarray[3];
    /*  The index of the next inserting point is 4. */
    i = 4;
  }

  if (det > 0.0) {
    /*  For keeping the positive orientation. */
    swappt         = insertarray[0];
    insertarray[0] = insertarray[1];
    insertarray[1] = swappt;
  }

  /*  Create the initial tet. */
  PetscInfo4(b->in, "    Create the first tet (%d, %d, %d, %d).\n", pointmark(m, insertarray[0]), pointmark(m, insertarray[1]), pointmark(m, insertarray[2]), pointmark(m, lastpt));
  ierr = TetGenMeshMakeTetrahedron(m, &newtet);CHKERRQ(ierr);
  setorg(&newtet, insertarray[0]);
  setdest(&newtet, insertarray[1]);
  setapex(&newtet, insertarray[2]);
  setoppo(&newtet, lastpt);
  if (oldtet) {
    for (j = 0; j < in->numberoftetrahedronattributes; j++) {
      attrib = elemattribute(m, oldtet->tet, j);
      setelemattribute(m, newtet.tet, j, attrib);
    }
    if (b->varvolume) {
      volume = volumebound(m, oldtet->tet);
      setvolumebound(m, newtet.tet, volume);
    }
  }
  /*  Set vertex type be FREEVOLVERTEX if it has no type yet. */
  if (pointtype(m, insertarray[0]) == UNUSEDVERTEX) {
    setpointtype(m, insertarray[0], FREEVOLVERTEX);
  }
  if (pointtype(m, insertarray[1]) == UNUSEDVERTEX) {
    setpointtype(m, insertarray[1], FREEVOLVERTEX);
  }
  if (pointtype(m, insertarray[2]) == UNUSEDVERTEX) {
    setpointtype(m, insertarray[2], FREEVOLVERTEX);
  }
  if (pointtype(m, lastpt) == UNUSEDVERTEX) {
    setpointtype(m, lastpt, FREEVOLVERTEX);
  }
  /*  Bond to 'dummytet' for point location. */
  m->dummytet[0] = encode(&newtet);
  m->recenttet   = newtet;
  /*  Update the point-to-tet map. */
  setpoint2tet(m, insertarray[0], encode(&newtet));
  setpoint2tet(m, insertarray[1], encode(&newtet));
  setpoint2tet(m, insertarray[2], encode(&newtet));
  setpoint2tet(m, lastpt,         encode(&newtet));
  if (b->verbose > 3) {
    PetscInfo(b->in, "    Creating tetra ");
    ierr = TetGenMeshPrintTet(m, &newtet, PETSC_FALSE);CHKERRQ(ierr);
  }
  /*  At init, all faces of this tet are hull faces. */
  m->hullsize = 4;

  PetscInfo(b->in, "    Incrementally inserting points.\n");
  /*  Insert the rest of points, one by one. */
  for (; i < arraysize; i++) {
    if (jump) {
      searchtet.tet = PETSC_NULL;
    } else {
      searchtet     = m->recenttet;
    }
    ierr = TetGenMeshInsertVertexBW(m, insertarray[i], &searchtet, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &loc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDelaunizeVertices"
/*  delaunizevertices()    Form a Delaunay tetrahedralization.                 */
/*                                                                             */
/*  Given a point set V (saved in 'points').  The Delaunay tetrahedralization  */
/*  D of V is created by incrementally inserting vertices. Returns the number  */
/*  of triangular faces bounding the convex hull of D.                         */
/* tetgenmesh::delaunizevertices() */
PetscErrorCode TetGenMeshDelaunizeVertices(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  point         *insertarray;
  long           arraysize;
  int            i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Constructing Delaunay tetrahedralization.\n");

  if (b->btree) {
    ierr = ArrayPoolCreate(sizeof(point*), 10, &m->btreenode_list);CHKERRQ(ierr);
    m->max_btreenode_size = 0;
    m->max_btree_depth    = 0;
  }

  if (!m->cavetetlist) {
    ierr = ArrayPoolCreate(sizeof(triface), 10, &m->cavetetlist);CHKERRQ(ierr);
    ierr = ArrayPoolCreate(sizeof(triface), 10, &m->cavebdrylist);CHKERRQ(ierr);
    ierr = ArrayPoolCreate(sizeof(triface), 10, &m->caveoldtetlist);CHKERRQ(ierr);
  }

  /*  Prepare the array of points for inserting. */
  arraysize = m->points->items;
  ierr = PetscMalloc(arraysize * sizeof(point), &insertarray);CHKERRQ(ierr);

  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  if (b->btree) { /*  -u option. */
    /*  Use the input order. */
    for(i = 0; i < arraysize; i++) {
      ierr = TetGenMeshPointTraverse(m, &insertarray[i]);CHKERRQ(ierr);
    }
    PetscInfo(b->in, "  Sorting vertices by a bsp-tree.\n");
    /*  Sort the points using a binary tree recursively. */
    ierr = TetGenMeshBTreeSort(m, insertarray, in->numberofpoints, 0, m->xmin, m->xmax, m->ymin, m->ymax, m->zmin, m->zmax, 0);CHKERRQ(ierr);
    PetscInfo1(b->in, "  Number of tree nodes: %ld.\n", m->btreenode_list->objects);
    PetscInfo1(b->in, "  Maximum tree node size: %d.\n", m->max_btreenode_size);
    PetscInfo1(b->in, "  Maximum tree depth: %d.\n", m->max_btree_depth);
    /*  Order the sorted points. */
    ierr = TetGenMeshOrderVertices(m, insertarray, in->numberofpoints);CHKERRQ(ierr);
  } else {
    PetscInfo(b->in, "  Permuting vertices.\n");
    /*  Randomize the point order. */
    for(i = 0; i < arraysize; i++) {
      ierr = TetGenMeshRandomChoice(m, i+1, &j);CHKERRQ(ierr); /*  0 <= j <= i */
      insertarray[i] = insertarray[j];
      ierr = TetGenMeshPointTraverse(m, &insertarray[j]);CHKERRQ(ierr);
    }
  }

  PetscInfo(b->in, "  Incrementally inserting vertices.\n");
  /*  Form the DT by incremental flip Delaunay algorithm. */
  ierr = TetGenMeshDelaunayIncrFlip(m, PETSC_NULL, insertarray, arraysize, PETSC_TRUE, b->plc ? PETSC_TRUE : PETSC_FALSE, 0.0, PETSC_NULL);CHKERRQ(ierr);

  if (b->btree) {
    point **pptary;

    for(i = 0; i < (int) m->btreenode_list->objects; i++) {
      pptary = (point **) fastlookup(m->btreenode_list, i);
      ierr = PetscFree(*pptary);CHKERRQ(ierr);
    }
    ierr = PetscFree(m->btreenode_list);CHKERRQ(ierr);
    m->btreenode_list = PETSC_NULL;
  }
  ierr = PetscFree(insertarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  delaunay_cxx ///////////////////////////////////////////////////////////// */

/*  surface_cxx ////////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSInsertVertex"
/*  sinsertvertex()    Insert a vertex into a triangulation of a facet.        */
/*                                                                             */
/*  The new point (p) will be located. Searching from 'splitsh'. If 'splitseg' */
/*  is not NULL, p is on a segment, no search is needed.                       */
/*                                                                             */
/*  If 'cflag' is not TRUE, the triangulation may be not convex. Don't insert  */
/*  p if it is found in outside.                                               */
/*                                                                             */
/*  Comment: This routine assumes the 'abovepoint' of this facet has been set, */
/*  i.e., the routine getabovepoint() has been executed before it is called.   */
/* tetgenmesh::sinsertvertex() */
PetscErrorCode TetGenMeshSInsertVertex(TetGenMesh *m, point insertpt, face *splitsh, face *splitseg, PetscBool bwflag, PetscBool cflag, locateresult *result)
{
  TetGenOpts    *b  = m->b;
  face *abfaces, *parysh, *pssub;
  face neighsh = {PETSC_NULL, 0}, newsh = {PETSC_NULL, 0}, casout = {PETSC_NULL, 0}, casin = {PETSC_NULL, 0};
  face aseg = {PETSC_NULL, 0}, bseg = {PETSC_NULL, 0}, aoutseg = {PETSC_NULL, 0}, boutseg = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  triface neightet = {PETSC_NULL, 0, 0};
  point pa, pb, pc, *ppt;
  locateresult loc;
  PetscReal sign, ori, area;
  int n = 0, s, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (splitseg) {
    spivot(splitseg, splitsh);
    loc = ONEDGE;
  } else {
    /*  Locate the point, '1' means the flag stop-at-segment is on. */
    ierr = TetGenMeshLocateSub(m, insertpt, splitsh, 1, 0, &loc);CHKERRQ(ierr);
  }

  /*  Return if p lies on a vertex. */
  if (loc == ONVERTEX) {
    if (result) {*result = loc;}
    PetscFunctionReturn(0);
  }

  if (loc == OUTSIDE && !cflag) {
    /*  Return if 'cflag' is not set. */
    if (result) {*result = loc;}
    PetscFunctionReturn(0);
  }

  if (loc == ONEDGE) {
    if (!splitseg) {
      /*  Do not split a segment. */
      sspivot(m, splitsh, &checkseg);
      if (checkseg.sh != m->dummysh) {
        if (result) {*result = loc;}
        PetscFunctionReturn(0);
      }
      /*  Check if this edge is on the hull. */
      spivot(splitsh, &neighsh);
      if (neighsh.sh == m->dummysh) {
        /*  A convex hull edge. The new point is on the hull. */
        loc = OUTSIDE;
      }
    }
  }

  if (b->verbose > 1) {
    pa = sorg(splitsh);
    pb = sdest(splitsh);
    pc = sapex(splitsh);
    PetscInfo5(b->in, "    Insert point %d (%d, %d, %d) loc %d\n", pointmark(m, insertpt), pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), (int) loc);
  }

  /*  Does 'insertpt' lie on a segment? */
  if (splitseg) {
    splitseg->shver = 0;
    pa = sorg(splitseg);
    /*  Count the number of faces at segment [a, b]. */
    n = 0;
    neighsh = *splitsh;
    do {
      spivotself(&neighsh);
      n++;
    } while ((neighsh.sh != m->dummysh) && (neighsh.sh != splitsh->sh));
    /*  n is at least 1. */
    ierr = PetscMalloc(n * sizeof(face), &abfaces);CHKERRQ(ierr);
    /*  Collect faces at seg [a, b]. */
    abfaces[0] = *splitsh;
    if (sorg(&abfaces[0]) != pa) sesymself(&abfaces[0]);
    for (i = 1; i < n; i++) {
      spivot(&abfaces[i - 1], &abfaces[i]);
      if (sorg(&abfaces[i]) != pa) sesymself(&abfaces[i]);
    }
  }

  /*  Initialize the cavity. */
  if (loc == ONEDGE) {
    smarktest(splitsh);
    ierr = ArrayPoolNewIndex(m->caveshlist, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
    *parysh = *splitsh;
    if (splitseg) {
      for(i = 1; i < n; i++) {
        smarktest(&abfaces[i]);
        ierr = ArrayPoolNewIndex(m->caveshlist, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
        *parysh = abfaces[i];
      }
    } else {
      spivot(splitsh, &neighsh);
      if (neighsh.sh != m->dummysh) {
        smarktest(&neighsh);
        ierr = ArrayPoolNewIndex(m->caveshlist, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
        *parysh = neighsh;
      }
    }
  } else if (loc == ONFACE) {
    smarktest(splitsh);
    ierr = ArrayPoolNewIndex(m->caveshlist, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
    *parysh = *splitsh;
  } else { /*  loc == OUTSIDE; */
    /*  This is only possible when T is convex. */
    if (!cflag) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot have point otside the convex hull");
    /*  Adjust 'abovepoint' to be above the 'splitsh'. 2009-07-21. */
    ori = TetGenOrient3D(sorg(splitsh), sdest(splitsh), sapex(splitsh), m->abovepoint);
    if (ori == 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    if (ori > 0) {
      sesymself(splitsh);
    }
    /*  Assume p is on top of the edge ('splitsh'). Find a right-most edge which is visible by p. */
    neighsh = *splitsh;
    while (1) {
      senext2self(&neighsh);
      spivot(&neighsh, &casout);
      if (casout.sh == m->dummysh) {
        /*  A convex hull edge. Is it visible by p. */
        pa = sorg(&neighsh);
        pb = sdest(&neighsh);
        ori = TetGenOrient3D(pa, pb, m->abovepoint, insertpt);
        if (ori < 0) {
          *splitsh = neighsh; /*  Update 'splitsh'. */
        } else {
          break; /*  'splitsh' is the right-most visible edge. */
        }
      } else {
        if (sorg(&casout) != sdest(&neighsh)) sesymself(&casout);
        neighsh = casout;
      }
    }
    /*  Create new triangles for all visible edges of p (from right to left). */
    casin.sh = m->dummysh;  /*  No adjacent face at right. */
    pa = sorg(splitsh);
    pb = sdest(splitsh);
    while (1) {
      /*  Create a new subface on top of the (visible) edge. */
      ierr = TetGenMeshMakeShellFace(m, m->subfaces, &newsh);CHKERRQ(ierr);
      /*  setshvertices(newsh, pb, pa, insertpt); */
      setsorg(&newsh, pb);
      setsdest(&newsh, pa);
      setsapex(&newsh, insertpt);
      setshellmark(m, &newsh, shellmark(m, splitsh));
      if (b->quality && m->varconstraint) {
        area = areabound(m, splitsh);
        setareabound(m, &newsh, area);
      }
      /*  Connect the new subface to the bottom subfaces. */
      sbond1(&newsh, splitsh);
      sbond1(splitsh, &newsh);
      /*  Connect the new subface to its right-adjacent subface. */
      if (casin.sh != m->dummysh) {
        senext(&newsh, &casout);
        sbond1(&casout, &casin);
        sbond1(&casin, &casout);
      }
      /*  The left-adjacent subface has not been created yet. */
      senext2(&newsh, &casin);
      /*  Add the new face into list. */
      smarktest(&newsh);
      ierr = ArrayPoolNewIndex(m->caveshlist, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
      *parysh = newsh;
      /*  Move to the convex hull edge at the left of 'splitsh'. */
      neighsh = *splitsh;
      while (1) {
        senextself(&neighsh);
        spivot(&neighsh, &casout);
        if (casout.sh == m->dummysh) {
          *splitsh = neighsh;
          break;
        }
        if (sorg(&casout) != sdest(&neighsh)) sesymself(&casout);
        neighsh = casout;
      }
      /*  A convex hull edge. Is it visible by p. */
      pa = sorg(splitsh);
      pb = sdest(splitsh);
      ori = TetGenOrient3D(pa, pb, m->abovepoint, insertpt);
      if (ori >= 0) break;
    }
  }

  /*  Form the Bowyer-Watson cavity. */
  for(i = 0; i < (int) m->caveshlist->objects; i++) {
    parysh = (face *) fastlookup(m->caveshlist, i);
    for(j = 0; j < 3; j++) {
      sspivot(m, parysh, &checkseg);
      if (checkseg.sh == m->dummysh) {
        spivot(parysh, &neighsh);
        if (neighsh.sh != m->dummysh) {
          if (!smarktested(&neighsh)) {
            if (bwflag) {
              pa = sorg(&neighsh);
              pb = sdest(&neighsh);
              pc = sapex(&neighsh);
              ierr = TetGenMeshInCircle3D(m, pa, pb, pc, insertpt, &sign);CHKERRQ(ierr);
              if (sign < 0) {
                smarktest(&neighsh);
                ierr = ArrayPoolNewIndex(m->caveshlist, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
                *pssub = neighsh;
              }
            } else {
              sign = 1; /*  A boundary edge. */
            }
          } else {
            sign = -1; /*  Not a boundary edge. */
          }
        } else {
          if (loc == OUTSIDE) {
            /*  It is a boundary edge if it does not contain insertp. */
            if ((sorg(parysh)==insertpt) || (sdest(parysh)==insertpt)) {
              sign = -1; /*  Not a boundary edge. */
            } else {
              sign = 1; /*  A boundary edge. */
            }
          } else {
            sign = 1; /*  A boundary edge. */
          }
        }
      } else {
        sign = 1; /*  A segment! */
      }
      if (sign >= 0) {
        /*  Add a boundary edge. */
        ierr = ArrayPoolNewIndex(m->caveshbdlist, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
        *pssub = *parysh;
      }
      senextself(parysh);
    }
  }

  /*  Creating new subfaces. */
  for(i = 0; i < (int) m->caveshbdlist->objects; i++) {
    parysh = (face *) fastlookup(m->caveshbdlist, i);
    sspivot(m, parysh, &checkseg);
    if ((parysh->shver & 01) != 0) sesymself(parysh);
    pa = sorg(parysh);
    pb = sdest(parysh);
    /*  Create a new subface. */
    ierr = TetGenMeshMakeShellFace(m, m->subfaces, &newsh);CHKERRQ(ierr);
    /*  setshvertices(newsh, pa, pb, insertpt); */
    setsorg(&newsh, pa);
    setsdest(&newsh, pb);
    setsapex(&newsh, insertpt);
    setshellmark(m, &newsh, shellmark(m, parysh));
    if (b->quality && m->varconstraint) {
      area = areabound(m, parysh);
      setareabound(m, &newsh, area);
    }
    /*  Connect newsh to outer subfaces. */
    spivot(parysh, &casout);
    if (casout.sh != m->dummysh) {
      if (casout.sh != parysh->sh) { /*  It is not self-bonded. */
        casin = casout;
        if (checkseg.sh != m->dummysh) {
          spivot(&casin, &neighsh);
          while (neighsh.sh != parysh->sh) {
            casin = neighsh;
            spivot(&casin, &neighsh);
          }
        }
        sbond1(&newsh, &casout);
        sbond1(&casin, &newsh);
      } else {
        /*  This side is empty. */
      }
    } else {
      /*  This is a hull side. Save it in dummysh[0] (it will be used by the routine locatesub()). 2009-07-20. */
      m->dummysh[0] = sencode(&newsh);
    }
    if (checkseg.sh != m->dummysh) {
      ssbond(m, &newsh, &checkseg);
    }
    /*  Connect oldsh <== newsh (for connecting adjacent new subfaces). */
    sbond1(parysh, &newsh);
  }

  /*  Set a handle for searching. */
  /*  recentsh = newsh; */

  /*  Connect adjacent new subfaces together. */
  for(i = 0; i < (int) m->caveshbdlist->objects; i++) {
    /*  Get an old subface at edge [a, b]. */
    parysh = (face *) fastlookup(m->caveshbdlist, i);
    sspivot(m, parysh, &checkseg);
    spivot(parysh, &newsh); /*  The new subface [a, b, p]. */
    senextself(&newsh); /*  At edge [b, p]. */
    spivot(&newsh, &neighsh);
    if (neighsh.sh == m->dummysh) {
      /*  Find the adjacent new subface at edge [b, p]. */
      pb = sdest(parysh);
      neighsh = *parysh;
      while (1) {
        senextself(&neighsh);
        spivotself(&neighsh);
        if (neighsh.sh == m->dummysh) break;
        if (!smarktested(&neighsh)) break;
        if (sdest(&neighsh) != pb) sesymself(&neighsh);
      }
      if (neighsh.sh != m->dummysh) {
        /*  Now 'neighsh' is a new subface at edge [b, #]. */
        if (sorg(&neighsh) != pb) sesymself(&neighsh);
        if (sorg(&neighsh) != pb) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        if (sapex(&neighsh) != insertpt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        senext2self(&neighsh); /*  Go to the open edge [p, b]. */
        spivot(&neighsh, &casout); /*  SELF_CHECK */
        if (casout.sh != m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        sbond(&newsh, &neighsh);
      } else {
        if (loc != OUTSIDE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        /*  It is a hull edge. 2009-07-21 */
        m->dummysh[0] = sencode(&newsh);
      }
    }
    spivot(parysh, &newsh); /*  The new subface [a, b, p]. */
    senext2self(&newsh); /*  At edge [p, a]. */
    spivot(&newsh, &neighsh);
    if (neighsh.sh == m->dummysh) {
      /*  Find the adjacent new subface at edge [p, a]. */
      pa = sorg(parysh);
      neighsh = *parysh;
      while (1) {
        senext2self(&neighsh);
        spivotself(&neighsh);
        if (neighsh.sh == m->dummysh) break;
        if (!smarktested(&neighsh)) break;
        if (sorg(&neighsh) != pa) sesymself(&neighsh);
      }
      if (neighsh.sh != m->dummysh) {
        /*  Now 'neighsh' is a new subface at edge [#, a]. */
        if (sdest(&neighsh) != pa) sesymself(&neighsh);
        if (sdest(&neighsh) != pa) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        if (sapex(&neighsh) != insertpt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        senextself(&neighsh); /*  Go to the open edge [a, p]. */
        spivot(&neighsh, &casout); /*  SELF_CHECK */
        if (casout.sh != m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        sbond(&newsh, &neighsh);
      } else {
        if (loc != OUTSIDE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        /*  It is a hull edge. 2009-07-21 */
        m->dummysh[0] = sencode(&newsh);
      }
    }
  }

  if (splitseg) {
    /*  Split the segment [a, b]. */
    aseg = *splitseg;
    pa = sorg(&aseg);
    pb = sdest(&aseg);
    PetscInfo3(b->in, "    Split seg (%d, %d) by %d.\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, insertpt));
    /*  Insert the new point p. */
    ierr = TetGenMeshMakeShellFace(m, m->subsegs, &bseg);CHKERRQ(ierr);
    /*  setshvertices(bseg, insertpt, pb, NULL); */
    setsorg(&bseg, insertpt);
    setsdest(&bseg, pb);
    setsapex(&bseg, PETSC_NULL);
    setsdest(&aseg, insertpt);
    setshellmark(m, &bseg, shellmark(m, &aseg));
    /*  This is done outside this routine (at where newpt was created). */
    /*  setpoint2sh(insertpt, sencode(aseg)); */
    if (b->quality && m->varconstraint) {
      setareabound(m, &bseg, areabound(m, &aseg));
    }
    /*  Update the point-to-seg map. */
    setpoint2seg(m, pb, sencode(&bseg));
    setpoint2seg(m, insertpt, sencode(&bseg));
    /*  Connect [p, b]<->[b, #]. */
    senext(&aseg, &aoutseg);
    spivotself(&aoutseg);
    if (aoutseg.sh != m->dummysh) {
      senext(&bseg, &boutseg);
      sbond(&boutseg, &aoutseg);
    }
    /*  Connect [a, p] <-> [p, b]. */
    senext(&aseg, &aoutseg);
    senext2(&bseg, &boutseg);
    sbond(&aoutseg, &boutseg);
    /*  Connect subsegs [a, p] and [p, b] to the true new subfaces. */
    for(i = 0; i < n; i++) {
      spivot(&abfaces[i], &newsh); /*  The faked new subface. */
      if (sorg(&newsh) != pa) sesymself(&newsh);
      senext2(&newsh, &neighsh); /*  The edge [p, a] in newsh */
      spivot(&neighsh, &casout);
      ssbond(m, &casout, &aseg);
      senext(&newsh, &neighsh); /*  The edge [b, p] in newsh */
      spivot(&neighsh, &casout);
      ssbond(m, &casout, &bseg);
    }
    if (n > 1) {
      /*  Create the two face rings at [a, p] and [p, b]. */
      for(i = 0; i < n; i++) {
        spivot(&abfaces[i], &newsh); /*  The faked new subface. */
        if (sorg(&newsh) != pa) sesymself(&newsh);
        spivot(&abfaces[(i + 1) % n], &neighsh); /*  The next faked new subface. */
        if (sorg(&neighsh) != pa) sesymself(&neighsh);
        senext2(&newsh, &casout); /*  The edge [p, a] in newsh. */
        senext2(&neighsh, &casin); /*  The edge [p, a] in neighsh. */
        spivotself(&casout);
        spivotself(&casin);
        sbond1(&casout, &casin); /*  Let the i's face point to (i+1)'s face. */
        senext(&newsh, &casout); /*  The edge [b, p] in newsh. */
        senext(&neighsh, &casin); /*  The edge [b, p] in neighsh. */
        spivotself(&casout);
        spivotself(&casin);
        sbond1(&casout, &casin);
      }
    } else {
      /*  Only one subface contains this segment. */
      /*  assert(n == 1); */
      spivot(&abfaces[0], &newsh);  /*  The faked new subface. */
      if (sorg(&newsh) != pa) sesymself(&newsh);
      senext2(&newsh, &casout); /*  The edge [p, a] in newsh. */
      spivotself(&casout);
      sdissolve(m, &casout); /*  Disconnect to faked subface. */
      senext(&newsh, &casout); /*  The edge [b, p] in newsh. */
      spivotself(&casout);
      sdissolve(m, &casout); /*  Disconnect to faked subface. */
    }
    /*  Delete the faked new subfaces. */
    for(i = 0; i < n; i++) {
      spivot(&abfaces[i], &newsh); /*  The faked new subface. */
      ierr = TetGenMeshShellFaceDealloc(m, m->subfaces, newsh.sh);CHKERRQ(ierr);
    }
    if (m->checksubsegs) {
      /*  Add two subsegs into stack (for recovery). */
      if (!sinfected(m, &aseg)) {
        ierr = TetGenMeshRandomChoice(m, m->subsegstack->objects + 1, &s);CHKERRQ(ierr);
        ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
        *parysh = * (face *) fastlookup(m->subsegstack, s);
        sinfect(m, &aseg);
        parysh = (face *) fastlookup(m->subsegstack, s);
        *parysh = aseg;
      }
      if (sinfected(m, &bseg)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      ierr = TetGenMeshRandomChoice(m, m->subsegstack->objects + 1, &s);CHKERRQ(ierr);
      ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
      *parysh = * (face *) fastlookup(m->subsegstack, s);
      sinfect(m, &bseg);
      parysh = (face *) fastlookup(m->subsegstack, s);
      *parysh = bseg;
    }
    ierr = PetscFree(abfaces);CHKERRQ(ierr);
  }

  if (m->checksubfaces) {
    /*  Add all new subfaces into list. */
    for(i = 0; i < (int) m->caveshbdlist->objects; i++) {
      /*  Get an old subface at edge [a, b]. */
      parysh = (face *) fastlookup(m->caveshbdlist, i);
      spivot(parysh, &newsh); /*  The new subface [a, b, p]. */
      /*  Some new subfaces may get deleted (when 'splitseg' is a segment). */
      if (!isdead_face(&newsh)) {
        PetscInfo3(b->in, "      Queue a new subface (%d, %d, %d).\n", pointmark(m, sorg(&newsh)), pointmark(m, sdest(&newsh)), pointmark(m, sapex(&newsh)));
        ierr = ArrayPoolNewIndex(m->subfacstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
        *pssub = newsh;
      }
    }
  }

  /*  Update the point-to-subface map. */
  for(i = 0; i < (int) m->caveshbdlist->objects; i++) {
    /*  Get an old subface at edge [a, b]. */
    parysh = (face *) fastlookup(m->caveshbdlist, i);
    spivot(parysh, &newsh); /*  The new subface [a, b, p]. */
    /*  Some new subfaces may get deleted (when 'splitseg' is a segment). */
    if (!isdead_face(&newsh)) {
      ppt = (point *) &(newsh.sh[3]);
      for(j = 0; j < 3; j++) {
        setpoint2sh(m, ppt[j], sencode(&newsh));
      }
    }
  }

  /*  Delete the old subfaces. */
  for(i = 0; i < (int) m->caveshlist->objects; i++) {
    parysh = (face *) fastlookup(m->caveshlist, i);
    if (m->checksubfaces) {
      /*  Disconnect in the neighbor tets. */
      for(j = 0; j < 2; j++) {
        stpivot(m, parysh, &neightet);
        if (neightet.tet != m->dummytet) {
          tsdissolve(m, &neightet);
        }
        sesymself(parysh);
      }
    }
    ierr = TetGenMeshShellFaceDealloc(m, m->subfaces, parysh->sh);CHKERRQ(ierr);
  }

  /*  Clean the working lists. */
  ierr = ArrayPoolRestart(m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolRestart(m->caveshbdlist);CHKERRQ(ierr);

  if (result) {*result = loc;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormStarPolygon"
/*  formstarpolygon()    Form the star polygon of a point in facet.            */
/*                                                                             */
/*  The polygon P is formed by all coplanar subfaces having 'pt' as a vertex.  */
/*  P is bounded by segments, e.g, if no segments, P is the full star of pt.   */
/*                                                                             */
/*  'trilist' T returns the subfaces, it has one of such subfaces on input.    */
/*  In addition, if f is in T, then sapex(f) = p. 'vertlist' V are verts of P. */
/*  Topologically, T is the star of p; V and the edges of T are the link of p. */
/* tetgenmesh::formstarpolygon() */
PetscErrorCode TetGenMeshFormStarPolygon(TetGenMesh *m, point pt, List *trilist, List *vertlist)
{
  face steinsh  = {PETSC_NULL, 0}, lnextsh = {PETSC_NULL, 0}, rnextsh = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  point pa, pb, pc, pd;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Get a subface f containing p. */
  ierr = ListItem(trilist, 0, (void **) &steinsh);CHKERRQ(ierr);
  steinsh.shver = 0; /*  CCW */
  /*  Let sapex(f) be p. */
  for(i = 0; i < 3; i++) {
    if (sapex(&steinsh) == pt) break;
    senextself(&steinsh);
  }
  if (i >= 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
  /*  Add the edge f into list. */
  ierr = ListSetItem(trilist, 0, &steinsh);CHKERRQ(ierr);
  pa = sorg(&steinsh);
  pb = sdest(&steinsh);
  if (vertlist) {
    /*  Add two verts a, b into V, */
    ierr = ListAppend(vertlist, &pa, PETSC_NULL);CHKERRQ(ierr);
    ierr = ListAppend(vertlist, &pb, PETSC_NULL);CHKERRQ(ierr);
  }

  /*  Rotate edge pa to the left (CW) until meet pb or a segment. */
  lnextsh = steinsh;
  pc = pa;
  do {
    senext2self(&lnextsh);
    if (sorg(&lnextsh) != pt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    sspivot(m, &lnextsh, &checkseg);
    if (checkseg.sh != m->dummysh) break; /*  Do not cross a segment. */
    /*  Get neighbor subface n (must exist). */
    spivotself(&lnextsh);
    if (lnextsh.sh == m->dummysh) break; /*  It's a hull edge. */
    /*  Go to the edge ca opposite to p. */
    if (sdest(&lnextsh) != pt) sesymself(&lnextsh);
    if (sdest(&lnextsh) != pt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    senext2self(&lnextsh);
    /*  Add n (at edge ca) to T. */
    ierr = ListAppend(trilist, &lnextsh, PETSC_NULL);CHKERRQ(ierr);
    /*  Add edge ca to E. */
    pc = sorg(&lnextsh);
    if (pc == pb) break; /*  Rotate back. */
    if (vertlist) {
      /*  Add vert c into V. */
      ierr = ListAppend(vertlist, &pc, PETSC_NULL);CHKERRQ(ierr);
    }
  } while (1);

  if (pc != pb) {
    /*  Rotate edge bp to the right (CCW) until meet a segment. */
    rnextsh = steinsh;
    do {
      senextself(&rnextsh);
      if (sdest(&rnextsh) != pt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      sspivot(m, &rnextsh, &checkseg);
      if (checkseg.sh != m->dummysh) break; /*  Do not cross a segment. */
      /*  Get neighbor subface n (must exist). */
      spivotself(&rnextsh);
      if (rnextsh.sh == m->dummysh) break; /*  It's a hull edge. */
      /*  Go to the edge bd opposite to p. */
      if (sorg(&rnextsh) != pt) sesymself(&rnextsh);
      if (sorg(&rnextsh) != pt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      senextself(&rnextsh);
      /*  Add n (at edge bd) to T. */
      ierr = ListAppend(trilist, &rnextsh, PETSC_NULL);CHKERRQ(ierr);
      /*  Add edge bd to E. */
      pd = sdest(&rnextsh);
      if (pd == pa) break; /*  Rotate back. */
      if (vertlist) {
        /*  Add vert d into V. */
        ierr = ListAppend(vertlist, &pd, PETSC_NULL);CHKERRQ(ierr);
      }
    } while (1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetFacetAbovePoint"
/*  getfacetabovepoint()    Get a point above a plane pass through a facet.    */
/*                                                                             */
/*  The calculcated point is saved in 'facetabovepointarray'. The 'abovepoint' */
/*  is set on return.                                                          */
/* tetgenmesh::getfacetabovepoint() */
PetscErrorCode TetGenMeshGetFacetAbovePoint(TetGenMesh *m, face *facetsh)
{
  TetGenOpts    *b  = m->b;
  List *verlist, *trilist, *tetlist;
  triface adjtet = {PETSC_NULL, 0, 0};
  point p1, p2, p3, pa;
  /*  enum locateresult loc; */
  PetscReal smallcos, cosa;
  PetscReal largevol, volume;
  PetscReal v1[3], v2[3], len;
  int llen, smallidx, largeidx;
  int shmark;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  m->abovecount++;
  /*  Initialize working lists. */
  ierr = ListCreate(sizeof(point *), PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &verlist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(face),    PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &trilist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(triface), PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &tetlist);CHKERRQ(ierr);

  /*  Get three pivotal points p1, p2, and p3 in the facet as a base triangle */
  /*    which is non-trivil and has good base angle (close to 90 degree). */

  /*  p1 is chosen as the one which has the smallest index in pa, pb, pc. */
  p1 = sorg(facetsh);
  pa = sdest(facetsh);
  if (pointmark(m, pa) < pointmark(m, p1)) p1 = pa;
  pa = sapex(facetsh);
  if (pointmark(m, pa) < pointmark(m, p1)) p1 = pa;
  /*  Form the star polygon of p1. */
  ierr = ListAppend(trilist, facetsh, PETSC_NULL);CHKERRQ(ierr);
  ierr = TetGenMeshFormStarPolygon(m, p1, trilist, verlist);CHKERRQ(ierr);

  /*  Get the second pivotal point p2. */
  ierr = ListItem(verlist, 0, (void **) &p2);CHKERRQ(ierr);
  /*  Get vector v1 = p1->p2. */
  for(i = 0; i < 3; i++) v1[i] = p2[i] - p1[i];
  len = sqrt(dot(v1, v1));
  if (len <= 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");  /*  p2 != p1. */
  for(i = 0; i < 3; i++) v1[i] /= len;

  /*  Get the third pivotal point p3. p3 is chosen as the one in 'verlist' */
  /*    which forms an angle with v1 closer to 90 degree than others do. */
  smallcos = 1.0; /*  The cosine value of 0 degree. */
  smallidx = 1;   /*  Default value. */
  ierr = ListLength(verlist, &llen);CHKERRQ(ierr);
  for(i = 1; i < llen; i++) {
    ierr = ListItem(verlist, i, (void **) &p3);CHKERRQ(ierr);
    for(j = 0; j < 3; j++) v2[j] = p3[j] - p1[j];
    len = sqrt(dot(v2, v2));
    if (len > 0.0) { /*  v2 is not too small. */
      cosa = fabs(dot(v1, v2)) / len;
      if (cosa < smallcos) {
        smallidx = i;
        smallcos = cosa;
      }
    }
  }
  if (smallcos >= 1.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");  /*  p1->p3 != p1->p2. */
  ierr = ListItem(verlist, smallidx, (void **) &p3);CHKERRQ(ierr);
  ierr = ListClear(verlist);CHKERRQ(ierr);

  if (m->tetrahedrons->items > 0l) {
    /*  Get a tet having p1 as a vertex. */
    ierr = TetGenMeshPoint2TetOrg(m, p1, &adjtet);CHKERRQ(ierr);
    if (org(&adjtet) != p1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    if (adjtet.tet != m->dummytet) {
      /*  Get the star polyhedron of p1. */
      ierr = ListAppend(tetlist, &adjtet, PETSC_NULL);CHKERRQ(ierr);
      ierr = TetGenMeshFormStarPolyhedron(m, p1, tetlist, verlist, PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  /*  Get the abovepoint in 'verlist'. It is the one form the largest valid */
  /*    volumw with the base triangle over other points in 'verlist. */
  largevol = 0.0;
  largeidx = 0;
  ierr = ListLength(verlist, &llen);CHKERRQ(ierr);
  for(i = 0; i < llen; i++) {
    PetscBool isCoplanar;

    ierr = ListItem(verlist, i, (void **) &pa);CHKERRQ(ierr);
    volume = TetGenOrient3D(p1, p2, p3, pa);
    ierr = TetGenMeshIsCoplanar(m, p1, p2, p3, pa, volume, b->epsilon * 1e+2, &isCoplanar);CHKERRQ(ierr);
    if (!isCoplanar) {
      if (fabs(volume) > largevol) {
        largevol = fabs(volume);
        largeidx = i;
      }
    }
  }

  /*  Do we have the abovepoint? */
  if (largevol > 0.0) {
    ierr = ListItem(verlist, largeidx, (void **) &m->abovepoint);CHKERRQ(ierr);
    PetscInfo2(b->in, "    Chosen abovepoint %d for facet %d.\n", pointmark(m, m->abovepoint), shellmark(m, facetsh));
  } else {
    /*  Calculate an abovepoint for this facet. */
    ierr = TetGenMeshFaceNormal(m, p1, p2, p3, v1, &len);CHKERRQ(ierr);
    if (len != 0.0) for (i = 0; i < 3; i++) v1[i] /= len;
    /*  Take the average edge length of the bounding box. */
    len = (0.5*(m->xmax - m->xmin) + 0.5*(m->ymax - m->ymin) + 0.5*(m->zmax - m->zmin)) / 3.0;
    /*  Temporarily create a point. It will be removed by jettison(); */
    ierr = TetGenMeshMakePoint(m, &m->abovepoint);CHKERRQ(ierr);
    setpointtype(m, m->abovepoint, UNUSEDVERTEX);
    m->unuverts++;
    for(i = 0; i < 3; i++) m->abovepoint[i] = p1[i] + len * v1[i];
    PetscInfo2(b->in, "    Calculated abovepoint %d for facet %d.\n", pointmark(m, m->abovepoint), shellmark(m, facetsh));
  }
  /*  Save the abovepoint in 'facetabovepointarray'. */
  shmark = shellmark(m, facetsh);
  m->facetabovepointarray[shmark] = m->abovepoint;
  ierr = ListDestroy(&trilist);CHKERRQ(ierr);
  ierr = ListDestroy(&tetlist);CHKERRQ(ierr);
  ierr = ListDestroy(&verlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshIncrFlipDelaunaySub"
/*  incrflipdelaunaysub()    Create a DT from a 3D coplanar point set using    */
/*                           the incremental flip algorithm.                   */
/*                                                                             */
/*  Let T be the current Delaunay triangulation (of vertices of a facet F).    */
/*  'shmark', the index of F in 'in->facetlist' (starts from 1).               */
/* tetgenmesh::incrflipdelaunaysub() */
PetscErrorCode TetGenMeshIncrFlipDelaunaySub(TetGenMesh *m, int shmark, PetscReal eps, List *ptlist, int holes, PetscReal *holelist, Queue *flipque, PetscBool *result)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  face newsh = {PETSC_NULL, 0}, startsh = {PETSC_NULL, 0};
  point *insertarray;
  point swappt;
  pbcdata *pd;
  locateresult loc;
  PetscReal det, area;
  PetscBool aboveflag;
  int arraysize;
  int epscount;
  int fmarker;
  int idx, i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Get the point array (saved in 'ptlist'). */
  insertarray = (point *) ptlist->base;
  ierr = ListLength(ptlist, &arraysize);CHKERRQ(ierr);
  if (arraysize < 3) {
    if (result) {*result = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  /*  Do calculation of 'abovepoint' if number of points > 3. */
  aboveflag = (arraysize > 3) ? PETSC_TRUE : PETSC_FALSE;

  /*  The initial triangulation T only has one triangle formed by 3 not */
  /*    cillinear points of the set V = 'insertarray'. The first point: */
  /*    a = insertarray[0]. */

  epscount = 0;
  while(1) {
    for(i = 1; i < arraysize; i++) {
      det = distance(insertarray[0], insertarray[i]);
      if (det > (m->longest * eps)) break;
    }
    if (i < arraysize) {
      /*  Swap to move b from index i to index 1. */
      swappt = insertarray[i];
      insertarray[i] = insertarray[1];
      insertarray[1] = swappt;
    }
    /*  Get the third point c, that is not collinear with a and b. */
    for (i++; i < arraysize; i++) {
      PetscBool isCollinear;
      ierr = TetGenMeshIsCollinear(m, insertarray[0], insertarray[1], insertarray[i], eps, &isCollinear);CHKERRQ(ierr);
      if (!isCollinear) break;
    }
    if (i < arraysize) {
      /*  Swap to move c from index i to index 2. */
      swappt = insertarray[i];
      insertarray[i] = insertarray[2];
      insertarray[2] = swappt;
      i = 3; /*  The next inserting point. */
    } else {
      /*  The set of vertices is not good (or nearly degenerate). */
      if ((eps == 0.0) || (epscount > 3)) {
        PetscInfo4(b->in, "Warning:  Discard an invalid facet #%d (%d, %d, %d, ...) looks like a line.\n",
                   shmark, pointmark(m, insertarray[0]), pointmark(m, insertarray[1]), pointmark(m, insertarray[2]));
        if (result) {*result = PETSC_FALSE;}
        PetscFunctionReturn(0);
      }
      /*  Decrease the eps, and continue to try. */
      eps *= 1e-2;
      epscount++;
      continue;
    }
    break;
  } /*  while (true); */

  /*  Create the initial triangle. */
  ierr = TetGenMeshMakeShellFace(m, m->subfaces, &newsh);CHKERRQ(ierr);
  setsorg(&newsh, insertarray[0]);
  setsdest(&newsh, insertarray[1]);
  setsapex(&newsh, insertarray[2]);
  /*  Remeber the facet it belongs to. */
  setshellmark(m, &newsh, shmark);
  /*  Set vertex type be FREESUBVERTEX if it has no type yet. */
  if (pointtype(m, insertarray[0]) == FREEVOLVERTEX) {
    setpointtype(m, insertarray[0], FREESUBVERTEX);
  }
  if (pointtype(m, insertarray[1]) == FREEVOLVERTEX) {
    setpointtype(m, insertarray[1], FREESUBVERTEX);
  }
  if (pointtype(m, insertarray[2]) == FREEVOLVERTEX) {
    setpointtype(m, insertarray[2], FREESUBVERTEX);
  }
  /*  Let 'dummysh' point to it (for point location). */
  m->dummysh[0] = sencode(&newsh);

  /*  Update the point-to-subface map. */
  for(i = 0; i < 3; i++) {
    setpoint2sh(m, insertarray[i], sencode(&newsh));
  }

  /*  Are there area constraints? */
  if (b->quality && in->facetconstraintlist) {
    idx = in->facetmarkerlist[shmark - 1]; /*  The actual facet marker. */
    for(k = 0; k < in->numberoffacetconstraints; k++) {
      fmarker = (int) in->facetconstraintlist[k * 2];
      if (fmarker == idx) {
        area = in->facetconstraintlist[k * 2 + 1];
        setareabound(m, &newsh, area);
        break;
      }
    }
  }

  /*  Are there pbc conditions? */
  if (m->checkpbcs) {
    idx = in->facetmarkerlist[shmark - 1]; /*  The actual facet marker. */
    for (k = 0; k < in->numberofpbcgroups; k++) {
      pd = &m->subpbcgrouptable[k];
      for(j = 0; j < 2; j++) {
        if (pd->fmark[j] == idx) {
          setshellpbcgroup(m, &newsh, k);
          pd->ss[j] = newsh;
        }
      }
    }
  }

  if (aboveflag) {
    /*  Compute the 'abovepoint' for TetGenOrient3D(). */
    m->abovepoint = m->facetabovepointarray[shmark];
    if (!m->abovepoint) {
      ierr = TetGenMeshGetFacetAbovePoint(m, &newsh);CHKERRQ(ierr);
    }
  }

  if (holes > 0) {
    /*  Project hole points onto the plane containing the facet. */
    PetscReal prj[3];
    for(k = 0; k < holes; k++) {
      ierr = TetGenMeshProjPt2Face(m, &holelist[k * 3], insertarray[0], insertarray[1], insertarray[2], prj);CHKERRQ(ierr);
      for(j = 0; j < 3; j++) holelist[k * 3 + j] = prj[j];
    }
  }

  /*  Incrementally insert the rest of points into T. */
  for(; i < arraysize; i++) {
    /*  Insert p_i. */
    startsh.sh = m->dummysh;
    ierr = TetGenMeshSInsertVertex(m, insertarray[i], &startsh, PETSC_NULL, PETSC_TRUE, PETSC_TRUE, &loc);CHKERRQ(ierr);
    /*  The point-to-subface map has been updated. */
    /*  Set p_i's type FREESUBVERTEX if it has no type yet. */
    if (pointtype(m, insertarray[i]) == FREEVOLVERTEX) {
      setpointtype(m, insertarray[i], FREESUBVERTEX);
    }
  }

  if (result) {*result = PETSC_TRUE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFindDirectionSub"
/*  finddirectionsub()    Find the first subface in a facet on the path from   */
/*                        one point to another.                                */
/*                                                                             */
/*  Finds the subface in the facet that intersects a line segment drawn from   */
/*  the origin of `searchsh' to the point `tend', and returns the result in    */
/*  `searchsh'.  The origin of `searchsh' does not change,  even though the    */
/*  subface returned may differ from the one passed in.                        */
/*                                                                             */
/*  The return value notes whether the destination or apex of the found face   */
/*  is collinear with the two points in question.                              */
/* tetgenmesh::finddirectionsub() */
PetscErrorCode TetGenMeshFindDirectionSub(TetGenMesh *m, face *searchsh, point tend, finddirectionresult *result)
{
  face checksh = {PETSC_NULL, 0};
  point startpoint, leftpoint, rightpoint;
  PetscReal leftccw, rightccw;
  PetscReal ori, sign;
  int leftflag, rightflag;

  PetscFunctionBegin;
  startpoint = sorg(searchsh);
  /*  Find the sign to simulate that abovepoint is 'above' the facet. */
  adjustedgering_face(searchsh, CCW);
  /*  Make sure 'startpoint' is the origin. */
  if (sorg(searchsh) != startpoint) senextself(searchsh);
  rightpoint = sdest(searchsh);
  leftpoint  = sapex(searchsh);
  ori = TetGenOrient3D(startpoint, rightpoint, leftpoint, m->abovepoint);
  sign = ori > 0.0 ? -1 : 1;

  /*  Is `tend' to the left? */
  ori = TetGenOrient3D(tend, startpoint, m->abovepoint, leftpoint);
  leftccw  = ori * sign;
  leftflag = leftccw > 0.0;
  /*  Is `tend' to the right? */
  ori = TetGenOrient3D(startpoint, tend, m->abovepoint, rightpoint);
  rightccw  = ori * sign;
  rightflag = rightccw > 0.0;
  if (leftflag && rightflag) {
    /*  `searchsh' faces directly away from `tend'.  We could go left or */
    /*    right.  Ask whether it's a triangle or a boundary on the left. */
    senext2(searchsh, &checksh);
    spivotself(&checksh);
    if (checksh.sh == m->dummysh) {
      leftflag = 0;
    } else {
      rightflag = 0;
    }
  }
  while (leftflag) {
    /*  Turn left until satisfied. */
    senext2self(searchsh);
    spivotself(searchsh);
    if (searchsh->sh == m->dummysh) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error in finddirectionsub():  Unable to find a subface leading from %d to %d.\n",pointmark(m,startpoint),pointmark(m,tend));
    if (sorg(searchsh) != startpoint) sesymself(searchsh);
    if (sorg(searchsh) != startpoint) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    leftpoint = sapex(searchsh);
    rightccw  = leftccw;
    ori = TetGenOrient3D(tend, startpoint, m->abovepoint, leftpoint);
    leftccw  = ori * sign;
    leftflag = leftccw > 0.0;
  }
  while (rightflag) {
    /*  Turn right until satisfied. */
    spivotself(searchsh);
    if (searchsh->sh == m->dummysh) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error in finddirectionsub():  Unable to find a subface leading from %d to %d.\n",pointmark(m, startpoint), pointmark(m, tend));
    if (sdest(searchsh) != startpoint) sesymself(searchsh);
    if (sdest(searchsh) != startpoint) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    senextself(searchsh);
    rightpoint = sdest(searchsh);
    leftccw = rightccw;
    ori = TetGenOrient3D(startpoint, tend, m->abovepoint, rightpoint);
    rightccw = ori * sign;
    rightflag = rightccw > 0.0;
  }
  if (leftccw == 0.0) {
    *result = LEFTCOLLINEAR;
  } else if (rightccw == 0.0) {
    *result = RIGHTCOLLINEAR;
  } else {
    *result = ACROSSEDGE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInsertSubseg"
/*  insertsubseg()    Create a subsegment and insert it between two subfaces.  */
/*                                                                             */
/*  The new subsegment ab is inserted at the edge of subface 'tri'.  If ab is  */
/*  not a hull edge, it is inserted between two subfaces.  If 'tri' is a hull  */
/*  face, the initial face ring of ab will be set only one face which is self- */
/*  bonded.  The final face ring will be constructed in 'unifysegments()'.     */
/* tetgenmesh::insertsubseg() */
PetscErrorCode TetGenMeshInsertSubseg(TetGenMesh *m, face *tri)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  face oppotri = {PETSC_NULL, 0};
  face newsubseg = {PETSC_NULL, 0};
  point pa, pb;
  PetscReal len;
  int e1, e2;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Check if there's already a subsegment here. */
  sspivot(m, tri, &newsubseg);
  if (newsubseg.sh == m->dummysh) {
    /*  Make new subsegment and initialize its vertices. */
    ierr = TetGenMeshMakeShellFace(m, m->subsegs, &newsubseg);CHKERRQ(ierr);
    pa = sorg(tri);
    pb = sdest(tri);
    setsorg(&newsubseg, pa);
    setsdest(&newsubseg, pb);
    /*  Are there length constraints? */
    if (b->quality && (in->segmentconstraintlist)) {
      for(i = 0; i < in->numberofsegmentconstraints; i++) {
        e1 = (int) in->segmentconstraintlist[i * 3];
        e2 = (int) in->segmentconstraintlist[i * 3 + 1];
        if (((pointmark(m, pa) == e1) && (pointmark(m, pb) == e2)) || ((pointmark(m, pa) == e2) && (pointmark(m, pb) == e1))) {
          len = in->segmentconstraintlist[i * 3 + 2];
          setareabound(m, &newsubseg, len);
          break;
        }
      }
    }
    /*  Bond new subsegment to the two subfaces it is sandwiched between. */
    ssbond(m, tri, &newsubseg);
    spivot(tri, &oppotri);
    /*  'oppotri' might be "out space". */
    if (oppotri.sh != m->dummysh) {
      ssbond(m, &oppotri, &newsubseg);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshScoutSegmentSub"
/*  scoutsegmentsub()    Scout the first triangle on the path from one point   */
/*                       to another, and check for completion (reaching the    */
/*                       second point), a collinear point,or the intersection  */
/*                       of two segments.                                      */
/*                                                                             */
/*  Returns true if the entire segment is successfully inserted, and false if  */
/*  the job must be finished by constrainededge().                             */
/* tetgenmesh::scoutsegmentsub() */
PetscErrorCode TetGenMeshScoutSegmentSub(TetGenMesh *m, face* searchsh, point tend, PetscBool *isInserted)
{
  face crosssub = {PETSC_NULL, 0}, crosssubseg = {PETSC_NULL, 0};
  point leftpoint, rightpoint;
  finddirectionresult collinear;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TetGenMeshFindDirectionSub(m, searchsh, tend, &collinear);CHKERRQ(ierr);
  rightpoint = sdest(searchsh);
  leftpoint = sapex(searchsh);
  if (rightpoint == tend || leftpoint == tend) {
    /*  The segment is already an edge. */
    if (leftpoint == tend) {
      senext2self(searchsh);
    }
    /*  Insert a subsegment. */
    ierr = TetGenMeshInsertSubseg(m, searchsh);CHKERRQ(ierr);
    if (isInserted) {*isInserted = PETSC_TRUE;}
    PetscFunctionReturn(0);
  } else if (collinear == LEFTCOLLINEAR) {
    /*  We've collided with a vertex between the segment's endpoints. */
    /*  Make the collinear vertex be the triangle's origin. */
    senextself(searchsh); /*  lprevself(*searchtri); */
    /*  Insert a subsegment. */
    ierr = TetGenMeshInsertSubseg(m, searchsh);CHKERRQ(ierr);
    /*  Insert the remainder of the segment. */
    ierr = TetGenMeshScoutSegmentSub(m, searchsh, tend, isInserted);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (collinear == RIGHTCOLLINEAR) {
    /*  We've collided with a vertex between the segment's endpoints. */
    /*  Insert a subsegment. */
    ierr = TetGenMeshInsertSubseg(m, searchsh);CHKERRQ(ierr);
    /*  Make the collinear vertex be the triangle's origin. */
    senextself(searchsh); /*  lnextself(*searchtri); */
    /*  Insert the remainder of the segment. */
    ierr = TetGenMeshScoutSegmentSub(m, searchsh, tend, isInserted);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    senext(searchsh, &crosssub); /*  lnext(*searchtri, crosstri); */
    /*  Check for a crossing segment. */
    sspivot(m, &crosssub, &crosssubseg);
    if (crosssubseg.sh != m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    if (isInserted) {*isInserted = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFlipEdgeRecursive"
/*  flipedgerecursive()    Flip an edge.                                       */
/*                                                                             */
/*  This is a support routine for inserting segments into a CDT.               */
/*                                                                             */
/*  Let 'flipedge' be ab, and two triangles abc, abd share at it.  ab may not  */
/*  flipable if the four vertices a, b, c, and d are non-convex. If it is the  */
/*  case, recursively flip ad or bd. Return when ab is flipped.                */
/* tetgenmesh::flipedgerecursive() */
PetscErrorCode TetGenMeshFlipEdgeRecursive(TetGenMesh *m, face *flipedge, Queue *flipqueue)
{
  face fixupsh = {PETSC_NULL, 0};
  point pa, pb, pc, pd;
  PetscReal oria, orib;
  PetscBool doflip;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pa = sorg(flipedge);
  pb = sdest(flipedge);
  pc = sapex(flipedge);
  do {
    spivot(flipedge, &fixupsh);
    pd = sapex(&fixupsh);
    oria = TetGenOrient3D(pc, pd, m->abovepoint, pa);
    orib = TetGenOrient3D(pc, pd, m->abovepoint, pb);
    doflip = (oria * orib < 0.0) ? PETSC_TRUE : PETSC_FALSE;
    if (doflip) {
      /*  Flip the edge (a, b) away. */
      ierr = TetGenMeshFlip22Sub(m, flipedge, flipqueue);CHKERRQ(ierr);
      /*  Fix flipedge on edge e (c, d). */
      ierr = TetGenMeshFindEdge_face(m, flipedge, pc, pd);CHKERRQ(ierr);
    } else {
      /*  ab is unflipable. Get the next edge (bd, or da) to flip. */
      if (sorg(&fixupsh) != pb) sesymself(&fixupsh);
      if (sdest(&fixupsh) != pa) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      if (fabs(oria) > fabs(orib)) {
        /*  acd has larger area. Choose da. */
        senextself(&fixupsh);
      } else {
        /*  bcd has larger area. Choose bd. */
        senext2self(&fixupsh);
      }
      /*  Flip the edge. */
      ierr = TetGenMeshFlipEdgeRecursive(m, &fixupsh, flipqueue);CHKERRQ(ierr);
    }
  } while (!doflip);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshConstrainedEdge"
/*  constrainededge()    Force a segment into a CDT.                           */
/*                                                                             */
/*  The segment s is recovered by flipping away the edges it intersects, and   */
/*  triangulating the polygons that form on each side of it.                   */
/*                                                                             */
/*  Generates a single subsegment connecting `tstart' to `tend'. The triangle  */
/*  `startsh' has `tstart' as its origin.                                      */
/* tetgenmesh::constrainededge() */
PetscErrorCode TetGenMeshConstrainedEdge(TetGenMesh *m, face *startsh, point tend, Queue *flipqueue)
{
  point tstart, tright, tleft;
  PetscReal rori, lori;
  PetscBool collision;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tstart = sorg(startsh);
  do {
    /*  Loop edges oppo to tstart until find one crosses the segment. */
    do {
      tright = sdest(startsh);
      tleft = sapex(startsh);
      /*  Is edge (tright, tleft) corss the segment. */
      rori = TetGenOrient3D(tstart, tright, m->abovepoint, tend);
      collision = (rori == 0.0) ? PETSC_TRUE : PETSC_FALSE;
      if (collision) break; /*  tright is on the segment. */
      lori = TetGenOrient3D(tstart, tleft, m->abovepoint, tend);
      collision = (lori == 0.0) ? PETSC_TRUE : PETSC_FALSE;
      if (collision) { /*   tleft is on the segment. */
        senext2self(startsh);
        break;
      }
      if (rori * lori < 0.0) break; /*  Find the crossing edge. */
      /*  Both points are at one side of the segment. */
      ierr = TetGenMeshFindDirectionSub(m, startsh, tend, PETSC_NULL);CHKERRQ(ierr);
    } while (PETSC_TRUE);
    if (collision) break;
    /*  Get the neighbor face at edge e (tright, tleft). */
    senextself(startsh);
    /*  Flip the crossing edge. */
    ierr = TetGenMeshFlipEdgeRecursive(m, startsh, flipqueue);CHKERRQ(ierr);
    /*  After flip, sorg(*startsh) == tstart. */
    if (sorg(startsh) != tstart) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
  } while (sdest(startsh) != tend);

  /*  Insert a subsegment to make the segment permanent. */
  ierr = TetGenMeshInsertSubseg(m, startsh);CHKERRQ(ierr);
  /*  If there was a collision with an interceding vertex, install another */
  /*    segment connecting that vertex with endpoint2. */
  if (collision) {
    PetscBool isInsert;
    /*  Insert the remainder of the segment. */
    ierr = TetGenMeshScoutSegmentSub(m, startsh, tend, &isInsert);CHKERRQ(ierr);
    if (!isInsert) {
      ierr = TetGenMeshConstrainedEdge(m, startsh, tend, flipqueue);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRecoverSegment"
/*  recoversegment()    Recover a segment in the surface triangulation.        */
/* tetgenmesh::recoversegment() */
PetscErrorCode TetGenMeshRecoverSegment(TetGenMesh *m, point tstart, point tend, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  face searchsh = {PETSC_NULL, 0};
  PetscBool isInsert;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo2(b->in, "    Insert seg (%d, %d).\n", pointmark(m, tstart), pointmark(m, tend));

  /*  Find a triangle whose origin is the segment's first endpoint. */
  ierr = TetGenMeshPoint2ShOrg(m, tstart, &searchsh);CHKERRQ(ierr);
  /*  Scout the segment and insert it if it is found. */
  ierr = TetGenMeshScoutSegmentSub(m, &searchsh, tend, &isInsert);CHKERRQ(ierr);
  if (isInsert) {
    /*  The segment was easily inserted. */
    PetscFunctionReturn(0);
  }
  /*  Insert the segment into the triangulation by flips. */
  ierr = TetGenMeshConstrainedEdge(m, &searchsh, tend, flipqueue);CHKERRQ(ierr);
  /*  Some edges may need flipping. */
  ierr = TetGenMeshLawson(m, flipqueue, PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInfectHullSub"
/*  infecthullsub()    Virally infect all of the triangles of the convex hull  */
/*                     that are not protected by subsegments.                  */
/* tetgenmesh::infecthullsub() */
PetscErrorCode TetGenMeshInfectHullSub(TetGenMesh *m, MemoryPool* viri)
{
  face hulltri = {PETSC_NULL, 0}, nexttri = {PETSC_NULL, 0}, starttri = {PETSC_NULL, 0};
  face hullsubseg = {PETSC_NULL, 0};
  shellface **deadshellface;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Find a triangle handle on the hull. */
  hulltri.sh = m->dummysh;
  hulltri.shver = 0;
  spivotself(&hulltri);
  adjustedgering_face(&hulltri, CCW);
  /*  Remember where we started so we know when to stop. */
  starttri = hulltri;
  /*  Go once counterclockwise around the convex hull. */
  do {
    /*  Ignore triangles that are already infected. */
    if (!sinfected(m, &hulltri)) {
      /*  Is the triangle protected by a subsegment? */
      sspivot(m, &hulltri, &hullsubseg);
      if (hullsubseg.sh == m->dummysh) {
        /*  The triangle is not protected; infect it. */
        if (!sinfected(m, &hulltri)) {
          sinfect(m, &hulltri);
          ierr = MemoryPoolAlloc(viri, (void **) &deadshellface);CHKERRQ(ierr);
          *deadshellface = hulltri.sh;
        }
      }
    }
    /*  To find the next hull edge, go clockwise around the next vertex. */
    senextself(&hulltri);
    spivot(&hulltri, &nexttri);
    while (nexttri.sh != m->dummysh) {
      if (sorg(&nexttri) != sdest(&hulltri)) {
        sesymself(&nexttri);
      }
      senext(&nexttri, &hulltri);
      spivot(&hulltri, &nexttri);
    }
  } while ((hulltri.sh != starttri.sh) || (hulltri.shver != starttri.shver));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPlagueSub"
/*  plaguesub()    Spread the virus from all infected triangles to any         */
/*                 neighbors not protected by subsegments.  Delete all         */
/*                 infected triangles.                                         */
/*                                                                             */
/*  This is the procedure that actually creates holes and concavities.         */
/* tetgenmesh::plaguesub() */
PetscErrorCode TetGenMeshPlagueSub(TetGenMesh *m, MemoryPool* viri)
{
  face testtri = {PETSC_NULL, 0}, neighbor = {PETSC_NULL, 0}, ghostsh = {PETSC_NULL, 0};
  face neighborsubseg = {PETSC_NULL, 0};
  shellface **virusloop;
  shellface **deadshellface;
  point *ppt;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Loop through all the infected triangles, spreading the virus to */
  /*    their neighbors, then to their neighbors' neighbors. */
  ierr = MemoryPoolTraversalInit(viri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtri.sh = *virusloop;
    /*  Check each of the triangle's three neighbors. */
    for(i = 0; i < 3; i++) {
      /*  Find the neighbor. */
      spivot(&testtri, &neighbor);
      /*  Check for a subsegment between the triangle and its neighbor. */
      sspivot(m, &testtri, &neighborsubseg);
      /*  Check if the neighbor is nonexistent or already infected. */
      if ((neighbor.sh == m->dummysh) || sinfected(m, &neighbor)) {
        if (neighborsubseg.sh != m->dummysh) {
          /*  There is a subsegment separating the triangle from its */
          /*    neighbor, but both triangles are dying, so the subsegment */
          /*    dies too. */
          ierr = TetGenMeshShellFaceDealloc(m, m->subsegs, neighborsubseg.sh);CHKERRQ(ierr);
          if (neighbor.sh != m->dummysh) {
            /*  Make sure the subsegment doesn't get deallocated again */
            /*    later when the infected neighbor is visited. */
            ssdissolve(m, &neighbor);
          }
        }
      } else {                   /*  The neighbor exists and is not infected. */
        if (neighborsubseg.sh == m->dummysh) {
          /*  There is no subsegment protecting the neighbor, so the */
          /*    neighbor becomes infected. */
          sinfect(m, &neighbor);
          /*  Ensure that the neighbor's neighbors will be infected. */
          ierr = MemoryPoolAlloc(viri, (void **) &deadshellface);CHKERRQ(ierr);
          *deadshellface = neighbor.sh;
        } else {               /*  The neighbor is protected by a subsegment. */
          /*  Remove this triangle from the subsegment. */
          ssbond(m, &neighbor, &neighborsubseg);
          /*  Update the point-to-subface map. 2009-07-21. */
          ppt = (point *) &(neighbor.sh[3]);
          for(j = 0; j < 3; j++) {
            setpoint2sh(m, ppt[j], sencode(&neighbor));
          }
        }
      }
      senextself(&testtri);
    }
    ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  }

  ghostsh.sh = m->dummysh; /*  A handle of outer space. */
  ierr = MemoryPoolTraversalInit(viri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtri.sh = *virusloop;
    /*  Record changes in the number of boundary edges, and disconnect */
    /*    dead triangles from their neighbors. */
    for(i = 0; i < 3; i++) {
      spivot(&testtri, &neighbor);
      if (neighbor.sh != m->dummysh) {
        /*  Disconnect the triangle from its neighbor. */
        /*  sdissolve(neighbor); */
        sbond(&neighbor, &ghostsh);
      }
      senextself(&testtri);
    }
    /*  Return the dead triangle to the pool of triangles. */
    ierr = TetGenMeshShellFaceDealloc(m, m->subfaces, testtri.sh);CHKERRQ(ierr);
    ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  }
  /*  Empty the virus pool. */
  ierr = MemoryPoolRestart(viri);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCarveHolesSub"
/*  carveholessub()    Find the holes and infect them.  Find the area          */
/*                     constraints and infect them.  Infect the convex hull.   */
/*                     Spread the infection and kill triangles.  Spread the    */
/*                     area constraints.                                       */
/*                                                                             */
/*  This routine mainly calls other routines to carry out all these functions. */
/* tetgenmesh::carveholessub() */
PetscErrorCode TetGenMeshCarveHolesSub(TetGenMesh *m, int holes, PetscReal *holelist, MemoryPool *viri)
{
  face searchtri = {PETSC_NULL, 0};
  shellface **holetri;
  locateresult intersect;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Mark as infected any unprotected triangles on the boundary. */
  /*    This is one way by which concavities are created. */
  ierr = TetGenMeshInfectHullSub(m, viri);CHKERRQ(ierr);

  if (holes > 0) {
    /*  Infect each triangle in which a hole lies. */
    for(i = 0; i < 3 * holes; i += 3) {
      /*  Ignore holes that aren't within the bounds of the mesh. */
      if ((holelist[i + 0] >= m->xmin) && (holelist[i + 0] <= m->xmax) &&
          (holelist[i + 1] >= m->ymin) && (holelist[i + 1] <= m->ymax) &&
          (holelist[i + 2] >= m->zmin) && (holelist[i + 2] <= m->zmax)) {
        /*  Start searching from some triangle on the outer boundary. */
        searchtri.sh = m->dummysh;
        /*  Find a triangle that contains the hole. */
        ierr = TetGenMeshLocateSub(m, &holelist[i], &searchtri, 0, 0.0, &intersect);CHKERRQ(ierr);
        if ((intersect != OUTSIDE) && (!sinfected(m, &searchtri))) {
          /*  Infect the triangle.  This is done by marking the triangle */
          /*    as infected and including the triangle in the virus pool. */
          sinfect(m, &searchtri);
          ierr = MemoryPoolAlloc(viri, (void **) &holetri);CHKERRQ(ierr);
          *holetri = searchtri.sh;
        }
      }
    }
  }

  if (viri->items > 0) {
    /*  Carve the holes and concavities. */
    ierr = TetGenMeshPlagueSub(m, viri);CHKERRQ(ierr);
  }
  /*  The virus pool should be empty now. */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTriangulate"
/*  triangulate()    Triangulate a PSLG into a CDT.                            */
/*                                                                             */
/*  A Planar Straight Line Graph (PSLG) P is actually a 2D polygonal region,   */
/*  possibly contains holes, segments and vertices in its interior. P is tri-  */
/*  angulated into a set of _subfaces_ forming a CDT of P.                     */
/*                                                                             */
/*  The vertices and segments of P are found in 'ptlist' and 'conlist', resp-  */
/*  ectively. 'holelist' contains a list of hole points. 'shmark' will be set  */
/*  to all subfaces of P.                                                      */
/*                                                                             */
/*  The CDT is created directly in the pools 'subfaces' and 'subsegs'. It can  */
/*  be retrived by a broadth-first searching starting from 'dummysh[0]'(debug  */
/*  function 'outsurfmesh()' does it).                                         */
/* tetgenmesh::triangulate() */
PetscErrorCode TetGenMeshTriangulate(TetGenMesh *m, int shmark, PetscReal eps, List *ptlist, List *conlist, int holes, PetscReal *holelist, MemoryPool *viri, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  face newsh = {PETSC_NULL, 0};
  int len, len2, i;
  PetscBool isFlipped;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ListLength(ptlist,  &len);CHKERRQ(ierr);
  ierr = ListLength(conlist, &len2);CHKERRQ(ierr);
  PetscInfo4(b->in, "    %d vertices, %d segments, %d holes, shmark: %d.\n", len, len2, holes, shmark);

  /*  Create the DT of V by the 2D incremental flip algorithm. */
  ierr = TetGenMeshIncrFlipDelaunaySub(m, shmark, eps, ptlist, holes, holelist, flipqueue, &isFlipped);CHKERRQ(ierr);
  if (isFlipped) {
    /*  Recover boundary edges. */
    ierr = ListLength(ptlist, &len);CHKERRQ(ierr);
    if (len > 3) {
      /*  Insert segments into the DT. */
      ierr = ListLength(conlist, &len2);CHKERRQ(ierr);
      for(i = 0; i < len2; i++) {
        point cons2[2];
        ierr = ListItem(conlist, i, (void **) cons2);CHKERRQ(ierr);
        ierr = TetGenMeshRecoverSegment(m, cons2[0], cons2[1], flipqueue);CHKERRQ(ierr);
      }
      /*  Carve holes and concavities. */
      ierr = TetGenMeshCarveHolesSub(m, holes, holelist, viri);CHKERRQ(ierr);
    } else if (len == 3) {
      /*  Insert 3 segments directly. */
      newsh.sh    = m->dummysh;
      newsh.shver = 0;
      spivotself(&newsh);
      for(i = 0; i < 3; i++) {
        ierr = TetGenMeshInsertSubseg(m, &newsh);CHKERRQ(ierr);
        senextself(&newsh);
      }
    } else if (len == 2) {
      /*  This facet is actually a segment. It is not support by the mesh data */
      /*    strcuture. Hence the segment will not be maintained in the mesh. */
      /*    However, during segment recovery, the segment can be processed. */
      point cons2[2];
      ierr = ListItem(conlist, 0, (void **) cons2);CHKERRQ(ierr);
      ierr = TetGenMeshMakeShellFace(m, m->subsegs, &newsh);CHKERRQ(ierr);
      setsorg(&newsh, cons2[0]);
      setsdest(&newsh, cons2[1]);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshUnifySegments"
/*  unifysegments()    Unify identical segments and build facet connections.   */
/*                                                                             */
/*  After creating the surface mesh. Each facet has its own segments.  There   */
/*  are duplicated segments between adjacent facets.  This routine has three   */
/*  purposes:                                                                  */
/*    (1) identify the set of segments which have the same endpoints and       */
/*        unify them into one segment, remove redundant ones;                  */
/*    (2) create the face rings of the unified segments, hence setup the       */
/*        connections between facets; and                                      */
/*    (3) set a unique marker (1-based) for each segment.                      */
/*  On finish, each segment is unique and the face ring around it (right-hand  */
/*  rule) is constructed. The connections between facets-facets are setup.     */
/* tetgenmesh::unifysegments() */
PetscErrorCode TetGenMeshUnifySegments(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  List *sfacelist;
  shellface **facesperverlist;
  face subsegloop = {PETSC_NULL, 0}, testseg = {PETSC_NULL, 0};
  face sface = {PETSC_NULL, 0}, sface1 = {PETSC_NULL, 0}, sface2 = {PETSC_NULL, 0};
  point torg, tdest;
  PetscReal da1, da2;
  int *idx2facelist;
  int segmarker;
  int len, idx, k, m1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Unifying segments.\n");

  /*  Compute a mapping from indices of vertices to subfaces. */
  ierr = TetGenMeshMakeSubfaceMap(m, &idx2facelist, &facesperverlist);CHKERRQ(ierr);
  /*  Initialize 'sfacelist' for constructing the face link of each segment. */
  ierr = ListCreate(sizeof(face), PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &sfacelist);CHKERRQ(ierr);
  segmarker = 1;
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &subsegloop.sh);CHKERRQ(ierr);
  while(subsegloop.sh) {
    subsegloop.shver = 0; /*  For sure. */
    torg = sorg(&subsegloop);
    tdest = sdest(&subsegloop);
    idx = pointmark(m, torg) - in->firstnumber;
    /*  Loop through the set of subfaces containing 'torg'.  Get all the */
    /*    subfaces containing the edge (torg, tdest). Save and order them */
    /*    in 'sfacelist', the ordering is defined by the right-hand rule */
    /*    with thumb points from torg to tdest. */
    for(k = idx2facelist[idx]; k < idx2facelist[idx + 1]; k++) {
      sface.sh = facesperverlist[k];
      sface.shver = 0;
      /*  sface may be died due to the removing of duplicated subfaces. */
      if (!isdead_face(&sface) && isfacehasedge(&sface, torg, tdest)) {
        /*  'sface' contains this segment. */
        ierr = TetGenMeshFindEdge_face(m, &sface, torg, tdest);CHKERRQ(ierr);
        /*  Save it in 'sfacelist'. */
        ierr = ListLength(sfacelist, &len);CHKERRQ(ierr);
        if (len < 2) {
          ierr = ListAppend(sfacelist, &sface, PETSC_NULL);CHKERRQ(ierr);
        } else {
          for(m1 = 0; m1 < len - 1; m1++) {
            ierr = ListItem(sfacelist, m1,   (void **) &sface1);CHKERRQ(ierr);
            ierr = ListItem(sfacelist, m1+1, (void **) &sface2);CHKERRQ(ierr);
            ierr = TetGenMeshFaceDihedral(m, torg, tdest, sapex(&sface1), sapex(&sface),  &da1);CHKERRQ(ierr);
            ierr = TetGenMeshFaceDihedral(m, torg, tdest, sapex(&sface1), sapex(&sface2), &da2);CHKERRQ(ierr);
            if (da1 < da2) {
              break;  /*  Insert it after m. */
            }
          }
          ierr = ListInsert(sfacelist, m1+1, &sface, PETSC_NULL);CHKERRQ(ierr);
        }
      }
    }
    ierr = ListLength(sfacelist, &len);CHKERRQ(ierr);
    PetscInfo3(b->in, "    Identifying %d segments of (%d  %d).\n", len, pointmark(m, torg), pointmark(m, tdest));
    /*  Set the connection between this segment and faces containing it, */
    /*    at the same time, remove redundant segments. */
    for(k = 0; k < len; k++) {
      ierr = ListItem(sfacelist, k, (void **) &sface);CHKERRQ(ierr);
      sspivot(m, &sface, &testseg);
      /*  If 'testseg' is not 'subsegloop', it is a redundant segment that */
      /*    needs be removed. BE CAREFUL it may already be removed. Do not */
      /*    remove it twice, i.e., do test 'isdead()' together. */
      if ((testseg.sh != subsegloop.sh) && !isdead_face(&testseg)) {
        ierr = TetGenMeshShellFaceDealloc(m, m->subsegs, testseg.sh);CHKERRQ(ierr);
      }
      /*  'ssbond' bonds the subface and the segment together, and dissloves */
      /*    the old bond as well. */
      ssbond(m, &sface, &subsegloop);
    }
    /*  Set connection between these faces. */
    ierr = ListItem(sfacelist, 0, (void **) &sface);CHKERRQ(ierr);
    ierr = ListLength(sfacelist, &len);CHKERRQ(ierr);
    if (len > 1) {
      for(k = 1; k <= len; k++) {
        if (k < len) {
          ierr = ListItem(sfacelist, k, (void **) &sface1);CHKERRQ(ierr);
        } else {
          ierr = ListItem(sfacelist, 0, (void **) &sface1);CHKERRQ(ierr); /*  Form a face loop. */
        }
        /*  Comment: For detecting invalid PLC, here we could check if the */
        /*    two subfaces "sface" and "sface1" are identical (skipped). */
        PetscInfo6(b->in, "    Bond subfaces (%d, %d, %d) and (%d, %d, %d).\n", pointmark(m, torg), pointmark(m, tdest), pointmark(m, sapex(&sface)),
                   pointmark(m, torg), pointmark(m, tdest), pointmark(m, sapex(&sface1)));
        sbond1(&sface, &sface1);
        sface = sface1;
      }
    } else {
      /*  This segment belongs to only on subface. */
      sdissolve(m, &sface);
    }
    /*  Set the unique segment marker into the unified segment. */
    setshellmark(m, &subsegloop, segmarker);
    /*  Increase the marker. */
    segmarker++;
    /*  Clear the working list. */
    ierr = ListClear(sfacelist);CHKERRQ(ierr);
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &subsegloop.sh);CHKERRQ(ierr);
  }
  ierr = PetscFree(idx2facelist);CHKERRQ(ierr);
  ierr = PetscFree(facesperverlist);CHKERRQ(ierr);
  ierr = ListDestroy(&sfacelist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMergeFacets"
/*  mergefacets()    Merge adjacent facets to be one facet if they are         */
/*                   coplanar and have the same boundary marker.               */
/*                                                                             */
/*  Segments between two merged facets will be removed from the mesh.  If all  */
/*  segments around a vertex have been removed, change its vertex type to be   */
/*  FREESUBVERTEX. Edge flips will be performed to ensure the Delaunayness of  */
/*  the triangulation of merged facets.                                        */
/* tetgenmesh::mergefacets() */
PetscErrorCode TetGenMeshMergeFacets(TetGenMesh *m, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  face parentsh = {PETSC_NULL, 0}, neighsh = {PETSC_NULL, 0}, neineighsh = {PETSC_NULL, 0};
  face segloop = {PETSC_NULL, 0};
  point eorg, edest;
  PetscReal ori;
  PetscBool mergeflag, pbcflag;
  int* segspernodelist;
  int fidx1, fidx2;
  int len, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Merging coplanar facets.\n");
  /*  Create and initialize 'segspernodelist'. */
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &segspernodelist);CHKERRQ(ierr);
  for(i = 0; i < m->points->items + 1; i++) segspernodelist[i] = 0;

  /*  Loop the segments, counter the number of segments sharing each vertex. */
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  while(segloop.sh) {
    /*  Increment the number of sharing segments for each endpoint. */
    for(i = 0; i < 2; i++) {
      j = pointmark(m, (point) segloop.sh[3 + i]);
      segspernodelist[j]++;
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  }

  /*  Loop the segments, find out dead segments. */
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  while(segloop.sh) {
    eorg = sorg(&segloop);
    edest = sdest(&segloop);
    spivot(&segloop, &parentsh);
    if (parentsh.sh != m->dummysh) {
      /*  This segment is not dangling. */
      spivot(&parentsh, &neighsh);
      if (neighsh.sh != m->dummysh) {
        /*  This segment belongs to at least two facets. */
        spivot(&neighsh, &neineighsh);
        if ((parentsh.sh != neighsh.sh) && (parentsh.sh == neineighsh.sh)) {
          /*  Exactly two subfaces at this segment. */
          fidx1 = shellmark(m, &parentsh) - 1;
          fidx2 = shellmark(m, &neighsh) - 1;
          pbcflag = PETSC_FALSE;
          if (m->checkpbcs) {
            pbcflag = (shellpbcgroup(m, &parentsh) >= 0) || (shellpbcgroup(m, &neighsh) >= 0) ? PETSC_TRUE : PETSC_FALSE;
          }
          /*  Possibly merge them if they are not in the same facet. */
          if ((fidx1 != fidx2) && !pbcflag) {
            /*  Test if they are coplanar. */
            ori = TetGenOrient3D(eorg, edest, sapex(&parentsh), sapex(&neighsh));
            if (ori != 0.0) {
              PetscBool isCoplanar;

              ierr = TetGenMeshIsCoplanar(m, eorg, edest, sapex(&parentsh), sapex(&neighsh), ori, b->epsilon, &isCoplanar);CHKERRQ(ierr);
              if (isCoplanar) {
                ori = 0.0; /*  They are assumed as coplanar. */
              }
            }
            if (ori == 0.0) {
              mergeflag = (!in->facetmarkerlist || in->facetmarkerlist[fidx1] == in->facetmarkerlist[fidx2]) ? PETSC_TRUE : PETSC_FALSE;
              if (mergeflag) {
                /*  This segment becomes dead. */
                PetscInfo2(b->in, "  Removing segment (%d, %d).\n", pointmark(m, eorg), pointmark(m, edest));
                ssdissolve(m, &parentsh);
                ssdissolve(m, &neighsh);
                ierr = TetGenMeshShellFaceDealloc(m, m->subsegs, segloop.sh);CHKERRQ(ierr);
                j = pointmark(m, eorg);
                segspernodelist[j]--;
                if (segspernodelist[j] == 0) {
                  setpointtype(m, eorg, FREESUBVERTEX);
                }
                j = pointmark(m, edest);
                segspernodelist[j]--;
                if (segspernodelist[j] == 0) {
                  setpointtype(m, edest, FREESUBVERTEX);
                }
                /*  Add 'parentsh' to queue checking for flip. */
                ierr = TetGenMeshEnqueueFlipEdge(m, &parentsh, flipqueue);CHKERRQ(ierr);
              }
            }
          }
        }
      } /*  neighsh.sh != dummysh */
    } /*  parentsh.sh != dummysh */
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  }

  ierr = QueueLength(flipqueue, &len);CHKERRQ(ierr);
  if (len > 0) {
    /*  Restore the Delaunay property in the facet triangulation. */
    ierr = TetGenMeshLawson(m, flipqueue, PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscFree(segspernodelist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMeshSurface"
/*  meshsurface()    Create the surface mesh of a PLC.                         */
/*                                                                             */
/*  Let X be the PLC, the surface mesh S of X consists of triangulated facets. */
/*  S is created mainly in the following steps:                                */
/*                                                                             */
/*  (1) Form the CDT of each facet of X separately (by routine triangulate()). */
/*  After it is done, the subfaces of each facet are connected to each other,  */
/*  however there is no connection between facets yet.  Notice each facet has  */
/*  its own segments, some of them are duplicated.                             */
/*                                                                             */
/*  (2) Remove the redundant segments created in step (1) (by routine unify-   */
/*  segment()). The subface ring of each segment is created,  the connection   */
/*  between facets are established as well.                                    */
/*                                                                             */
/*  The return value indicates the number of segments of X.                    */
/* tetgenmesh::meshsurface() */
PetscErrorCode TetGenMeshMeshSurface(TetGenMesh *m, long *numSegments)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  List *ptlist, *conlist;
  Queue *flipqueue;
  facet *f;
  polygon *p;
  MemoryPool *viri;
  point *idx2verlist;
  point tstart, tend, *cons;
  int *worklist;
  int end1, end2;
  int len, shmark, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Creating surface mesh.\n");

  /*  Compute a mapping from indices to points. */
  ierr = TetGenMeshMakeIndex2PointMap(m, &idx2verlist);CHKERRQ(ierr);
  /*   Compute a mapping from points to tets for computing abovepoints. */
  /*  makepoint2tetmap(); */
  /*  Initialize 'facetabovepointarray'. */
  ierr = PetscMalloc((in->numberoffacets + 1) * sizeof(point), &m->facetabovepointarray);CHKERRQ(ierr);
  for(i = 0; i < in->numberoffacets + 1; i++) {
    m->facetabovepointarray[i] = PETSC_NULL;
  }
  if (m->checkpbcs) {
    /*  Initialize the global array 'subpbcgrouptable'. */
    /*  createsubpbcgrouptable(); */
  }

  /*  Initialize working lists. */
  ierr = MemoryPoolCreate(sizeof(shellface *), 1024, POINTER, 0, &viri);CHKERRQ(ierr);
  ierr = QueueCreate(sizeof(badface), PETSC_DECIDE, &flipqueue);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(point *), PETSC_NULL, 256, PETSC_DECIDE, &ptlist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(point *)*2, PETSC_NULL, 256, PETSC_DECIDE, &conlist);CHKERRQ(ierr);
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &worklist);CHKERRQ(ierr);
  for (i = 0; i < m->points->items + 1; i++) worklist[i] = 0;
  ierr = ArrayPoolCreate(sizeof(face), 10, &m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &m->caveshbdlist);CHKERRQ(ierr);

  /*  Loop the facet list, triangulate each facet. On finish, all subfaces */
  /*    are in 'subfaces', all segments are in 'subsegs'. Notice: there're */
  /*    redundant segments.  Remember: All facet indices count from 1. */
  for(shmark = 1; shmark <= in->numberoffacets; shmark++) {
    /*  Get a facet F. */
    f = &in->facetlist[shmark - 1];

    /*  Process the duplicated points first, they are marked with type */
    /*    DUPLICATEDVERTEX by incrflipdelaunay().  Let p and q are dup. */
    /*    and the index of p is larger than q's, p is substituted by q. */
    /*    In a STL mesh, duplicated points are implicitly included. */
    if ((b->object == TETGEN_OBJECT_STL) || m->dupverts) {
      /*  Loop all polygons of this facet. */
      for(i = 0; i < f->numberofpolygons; i++) {
        p = &(f->polygonlist[i]);
        /*  Loop other vertices of this polygon. */
        for(j = 0; j < p->numberofvertices; j++) {
          end1 = p->vertexlist[j];
          tstart = idx2verlist[end1 - in->firstnumber];
          if (pointtype(m, tstart) == DUPLICATEDVERTEX) {
            /*  Reset the index of vertex-j. */
            tend = point2ppt(m, tstart);
            end2 = pointmark(m, tend);
            p->vertexlist[j] = end2;
          }
        }
      }
    }

    /*  Loop polygons of F, get the set V of vertices and S of segments. */
    for(i = 0; i < f->numberofpolygons; i++) {
      /*  Get a polygon. */
      p = &(f->polygonlist[i]);
      /*  Get the first vertex. */
      end1 = p->vertexlist[0];
      if ((end1 < in->firstnumber) || (end1 >= in->firstnumber + in->numberofpoints)) {
        PetscInfo3(b->in, "Warning:  Invalid the 1st vertex %d of polygon %d in facet %d.\n", end1, i + 1, shmark);
        continue; /*  Skip this polygon. */
      }
      tstart = idx2verlist[end1 - in->firstnumber];
      /*  Add tstart to V if it haven't been added yet. */
      if (worklist[end1] == 0) {
        ierr = ListAppend(ptlist, &tstart, PETSC_NULL);CHKERRQ(ierr);
        worklist[end1] = 1;
      }
      /*  Loop other vertices of this polygon. */
      for(j = 1; j <= p->numberofvertices; j++) {
        /*  get a vertex. */
        if (j < p->numberofvertices) {
          end2 = p->vertexlist[j];
        } else {
          end2 = p->vertexlist[0];  /*  Form a loop from last to first. */
        }
        if ((end2 < in->firstnumber) || (end2 >= in->firstnumber + in->numberofpoints)) {
          PetscInfo3(b->in, "Warning:  Invalid vertex %d in polygon %d in facet %d.\n", end2, i + 1, shmark);
        } else {
          if (end1 != end2) {
            /*  'end1' and 'end2' form a segment. */
            tend = idx2verlist[end2 - in->firstnumber];
            /*  Add tstart to V if it haven't been added yet. */
            if (worklist[end2] == 0) {
              ierr = ListAppend(ptlist, &tend, PETSC_NULL);CHKERRQ(ierr);
              worklist[end2] = 1;
            }
            /*  Save the segment in S (conlist). */
            ierr = ListAppend(conlist, PETSC_NULL, (void **) &cons);CHKERRQ(ierr);
            cons[0] = tstart;
            cons[1] = tend;
            /*  Set the start for next continuous segment. */
            end1   = end2;
            tstart = tend;
          } else {
            /*  Two identical vertices represent an isolated vertex of F. */
            if (p->numberofvertices > 2) {
              /*  This may be an error in the input, anyway, we can continue */
              /*    by simply skipping this segment. */
              PetscInfo2(b->in, "Warning:  Polygon %d has two identical verts in facet %d.\n", i + 1, shmark);
            }
            /*  Ignore this vertex. */
          }
        }
        /*  Is the polygon degenerate (a segment or a vertex)? */
        if (p->numberofvertices == 2) break;
      }
    }
    /*  Unmark vertices. */
    ierr = ListLength(ptlist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(ptlist, i, (void **) &tstart);CHKERRQ(ierr);
      end1 = pointmark(m, tstart);
      if (worklist[end1] != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Vertex %d mark %d should be 1", end1, worklist[end1]);
      worklist[end1] = 0;
    }

    /*  Create a CDT of F. */
    ierr = TetGenMeshTriangulate(m, shmark, b->epsilon * 1e+2, ptlist, conlist, f->numberofholes, f->holelist, viri, flipqueue);CHKERRQ(ierr);
    /*  Clear working lists. */
    ierr = ListClear(ptlist);CHKERRQ(ierr);
    ierr = ListClear(conlist);CHKERRQ(ierr);
    ierr = MemoryPoolRestart(viri);CHKERRQ(ierr);
  }

  ierr = ArrayPoolDestroy(&m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&m->caveshbdlist);CHKERRQ(ierr);
  m->caveshlist   = PETSC_NULL;
  m->caveshbdlist = PETSC_NULL;

  /*  Unify segments in 'subsegs', remove redundant segments.  Face links of segments are also built. */
  ierr = TetGenMeshUnifySegments(m);CHKERRQ(ierr);

  /*  Remember the number of input segments (for output). */
  m->insegments = m->subsegs->items;

  if (m->checkpbcs) {
    /*  Create the global array 'segpbcgrouptable'. */
    /*  createsegpbcgrouptable(); */
  }

  if (b->object == TETGEN_OBJECT_STL) {
    /*  Remove redundant vertices (for .stl input mesh). */
    ierr = TetGenMeshJettisonNodes(m);CHKERRQ(ierr);
  }

  if (!b->nomerge && !b->nobisect && !m->checkpbcs) {
    /*  No '-M' switch - merge adjacent facets if they are coplanar. */
    ierr = TetGenMeshMergeFacets(m, flipqueue);CHKERRQ(ierr);
  }

  /*  Create the point-to-segment map. */
  ierr = TetGenMeshMakePoint2SegMap(m);CHKERRQ(ierr);

  ierr = PetscFree(idx2verlist);CHKERRQ(ierr);
  ierr = PetscFree(worklist);CHKERRQ(ierr);
  ierr = ListDestroy(&ptlist);CHKERRQ(ierr);
  ierr = ListDestroy(&conlist);CHKERRQ(ierr);
  ierr = QueueDestroy(&flipqueue);CHKERRQ(ierr);
  ierr = MemoryPoolDestroy(&viri);CHKERRQ(ierr);

  if (numSegments) {*numSegments = m->subsegs->items;}
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  surface_cxx ////////////////////////////////////////////////////////////// */

/*  constrained_cxx ////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMarkAcuteVertices"
/*  A vertex v is called acute if there are two segments sharing at v forming  */
/*  an acute angle (i.e. smaller than 90 degree).                              */
/*                                                                             */
/*  This routine finds all acute vertices in the PLC and marks them as point-  */
/*  type ACUTEVERTEX. The other vertices of segments which are non-acute will  */
/*  be marked as NACUTEVERTEX.  Vertices which are not endpoints of segments   */
/*  (such as DUPLICATEDVERTEX, UNUSEDVERTEX, etc) are not infected.            */
/*                                                                             */
/*  NOTE: This routine should be called before Steiner points are introduced.  */
/*  That is, no point has type like FREESEGVERTEX, etc.                        */
/* tetgenmesh::markacutevertices() */
PetscErrorCode TetGenMeshMarkAcuteVertices(TetGenMesh *m, PetscReal acuteangle)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  shellface **segsperverlist;
  face segloop = {PETSC_NULL, 0}, nextseg = {PETSC_NULL, 0};
  point pointloop, edest, eapex;
  PetscReal cosbound, anglearc;
  PetscReal v1[3], v2[3], L, D;
  PetscBool isacute;
  int *idx2seglist;
  int acutecount;
  int idx, i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Marking acute vertices.\n");
  anglearc   = acuteangle * PETSC_PI / 180.0;
  cosbound   = cos(anglearc);
  acutecount = 0;
  /*  Constructing a map from vertex to segments. */
  ierr = TetGenMeshMakeSegmentMap(m, &idx2seglist, &segsperverlist);CHKERRQ(ierr);
  /*  Loop over the set of vertices. */
  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  while(pointloop) {
    idx = pointmark(m, pointloop) - in->firstnumber;
    /*  Only do test if p is an endpoint of some segments. */
    if (idx2seglist[idx + 1] > idx2seglist[idx]) {
      /*  Init p to be non-acute. */
      setpointtype(m, pointloop, NACUTEVERTEX);
      isacute = PETSC_FALSE;
      /*  Loop through all segments sharing at p. */
      for(i = idx2seglist[idx]; i < idx2seglist[idx + 1] && !isacute; i++) {
        segloop.sh = segsperverlist[i];
        /*  segloop.shver = 0; */
        if (sorg(&segloop) != pointloop) {sesymself(&segloop);}
        edest = sdest(&segloop);
        for(j = i + 1; j < idx2seglist[idx + 1] && !isacute; j++) {
          nextseg.sh = segsperverlist[j];
          /*  nextseg.shver = 0; */
          if (sorg(&nextseg) != pointloop) {sesymself(&nextseg);}
          eapex = sdest(&nextseg);
          /*  Check the angle formed by segs (p, edest) and (p, eapex). */
          for(k = 0; k < 3; k++) {
            v1[k] = edest[k] - pointloop[k];
            v2[k] = eapex[k] - pointloop[k];
          }
          L = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
          for(k = 0; k < 3; k++) v1[k] /= L;
          L = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
          for(k = 0; k < 3; k++) v2[k] /= L;
          D = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
          /*  Is D acute? */
          isacute = D >= cosbound ? PETSC_TRUE : PETSC_FALSE;
        }
      }
      if (isacute) {
        /*  Mark p to be acute. */
        setpointtype(m, pointloop, ACUTEVERTEX);
        acutecount++;
      }
    }
    ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  }
  ierr = PetscFree(idx2seglist);CHKERRQ(ierr);
  ierr = PetscFree(segsperverlist);CHKERRQ(ierr);
  PetscInfo1(b->in, "  %d acute vertices.\n", acutecount);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFindDirection2"
/*  finddirection()    Find the tet on the path from one point to another.     */
/*                                                                             */
/*  The path starts from 'searchtet''s origin and ends at 'endpt'. On finish,  */
/*  'searchtet' contains a tet on the path, its origin does not change.        */
/*                                                                             */
/*  The return value indicates one of the following cases (let 'searchtet' be  */
/*  abcd, a is the origin of the path):                                        */
/*    - ACROSSVERT, edge ab is collinear with the path;                        */
/*    - ACROSSEDGE, edge bc intersects with the path;                          */
/*    - ACROSSFACE, face bcd intersects with the path.                         */
/*                                                                             */
/*  WARNING: This routine is designed for convex triangulations, and will not  */
/*  generally work after the holes and concavities have been carved.           */
/*    - BELOWHULL2, the mesh is non-convex and the searching for the path has  */
/*                  got stucked at a non-convex boundary face.                 */
/* tetgenmesh::finddirection2() */
PetscErrorCode TetGenMeshFindDirection2(TetGenMesh *m, triface* searchtet, point endpt, interresult *result)
{
  TetGenOpts    *b  = m->b;
  triface neightet = {PETSC_NULL, 0, 0};
  point pa, pb, pc, pd, pn;
  enum {HMOVE, RMOVE, LMOVE} nextmove;
  /* enum {HCOPLANE, RCOPLANE, LCOPLANE, NCOPLANE} cop; */
  PetscReal hori, rori, lori;
  PetscReal dmin, dist;

  PetscFunctionBegin;
  if ((!searchtet->tet) || (searchtet->tet == m->dummytet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
  /*  The origin is fixed. */
  pa = org(searchtet);
  if (searchtet->ver & 01) {
    /*  Switch to the 0th edge ring. */
    esymself(searchtet);
    enextself(searchtet);
  }
  pb = dest(searchtet);
  if (pb == endpt) {
    /*  pa->pb is the search edge. */
    if (result) {*result = INTERVERT;}
    PetscFunctionReturn(0);
  }
  pc = apex(searchtet);
  if (pc == endpt) {
    /*  pa->pc is the search edge. */
    enext2self(searchtet);
    esymself(searchtet);
    if (result) {*result = INTERVERT;}
    PetscFunctionReturn(0);
  }

  /*  Walk through tets at pa until the right one is found. */
  while (1) {
    pd = oppo(searchtet);

    PetscInfo5(b->in, "      From tet (%d, %d, %d, %d) to %d.\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), pointmark(m, pd), pointmark(m, endpt));

    /*  Check whether the opposite vertex is 'endpt'. */
    if (pd == endpt) {
      /*  pa->pd is the search edge. */
      fnextself(m, searchtet);
      enext2self(searchtet);
      esymself(searchtet);
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }

    /*  Now assume that the base face abc coincides with the horizon plane, */
    /*    and d lies above the horizon.  The search point 'endpt' may lie */
    /*    above or below the horizon.  We test the orientations of 'endpt' */
    /*    with respect to three planes: abc (horizon), bad (right plane), */
    /*    and acd (left plane). */
    hori = TetGenOrient3D(pa, pb, pc, endpt);
    rori = TetGenOrient3D(pb, pa, pd, endpt);
    lori = TetGenOrient3D(pa, pc, pd, endpt);
    m->orient3dcount += 3;

    /*  Now decide the tet to move.  It is possible there are more than one */
    /*    tet are viable moves. Use the opposite points of thier neighbors */
    /*    to discriminate, i.e., we choose the tet whose opposite point has */
    /*    the shortest distance to 'endpt'. */
    if (hori > 0) {
      if (rori > 0) {
        if (lori > 0) {
          /*  Any of the three neighbors is a viable move. */
          nextmove = HMOVE;
          sym(searchtet, &neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dmin = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dmin = NORM2(m->xmax - m->xmin, m->ymax - m->ymin, m->zmax - m->zmin);
          }
          fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dist = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dist = dmin;
          }
          if (dist < dmin) {
            nextmove = RMOVE;
            dmin = dist;
          }
          enext2fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dist = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dist = dmin;
          }
          if (dist < dmin) {
            nextmove = LMOVE;
            dmin = dist;
          }
        } else {
          /*  Two tets, below horizon and below right, are viable. */
          nextmove = HMOVE;
          sym(searchtet, &neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dmin = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dmin = NORM2(m->xmax - m->xmin, m->ymax - m->ymin, m->zmax - m->zmin);
          }
          fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dist = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dist = dmin;
          }
          if (dist < dmin) {
            nextmove = RMOVE;
            dmin = dist;
          }
        }
      } else {
        if (lori > 0) {
          /*  Two tets, below horizon and below left, are viable. */
          nextmove = HMOVE;
          sym(searchtet, &neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dmin = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dmin = NORM2(m->xmax - m->xmin, m->ymax - m->ymin, m->zmax - m->zmin);
          }
          enext2fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dist = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dist = dmin;
          }
          if (dist < dmin) {
            nextmove = LMOVE;
            dmin = dist;
          }
        } else {
          /*  The tet below horizon is chosen. */
          nextmove = HMOVE;
        }
      }
    } else {
      if (rori > 0) {
        if (lori > 0) {
          /*  Two tets, below right and below left, are viable. */
          nextmove = RMOVE;
          fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dmin = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dmin = NORM2(m->xmax - m->xmin, m->ymax - m->ymin, m->zmax - m->zmin);
          }
          enext2fnext(m, searchtet, &neightet);
          symself(&neightet);
          if (neightet.tet != m->dummytet) {
            pn = oppo(&neightet);
            dist = NORM2(endpt[0] - pn[0], endpt[1] - pn[1], endpt[2] - pn[2]);
          } else {
            dist = dmin;
          }
          if (dist < dmin) {
            nextmove = LMOVE;
            dmin = dist;
          }
        } else {
          /*  The tet below right is chosen. */
          nextmove = RMOVE;
        }
      } else {
        if (lori > 0) {
          /*  The tet below left is chosen. */
          nextmove = LMOVE;
        } else {
          /*  'endpt' lies either on the plane(s) or across face bcd. */
          if (hori == 0) {
            if (rori == 0) {
              /*  pa->'endpt' is COLLINEAR with pa->pb. */
              if (result) {*result = INTERVERT;}
              PetscFunctionReturn(0);
            }
            if (lori == 0) {
              /*  pa->'endpt' is COLLINEAR with pa->pc. */
              enext2self(searchtet);
              esymself(searchtet);
              if (result) {*result = INTERVERT;}
              PetscFunctionReturn(0);
            }
            /*  pa->'endpt' crosses the edge pb->pc. */
            /*  enextself(*searchtet); */
            /*  return INTEREDGE; */
            /* cop = HCOPLANE; */
            break;
          }
          if (rori == 0) {
            if (lori == 0) {
              /*  pa->'endpt' is COLLINEAR with pa->pd. */
              fnextself(m, searchtet); /*  face abd. */
              enext2self(searchtet);
              esymself(searchtet);
              if (result) {*result = INTERVERT;}
              PetscFunctionReturn(0);
            }
            /*  pa->'endpt' crosses the edge pb->pd. */
            /* cop = RCOPLANE; */
            break;
          }
          if (lori == 0) {
            /*  pa->'endpt' crosses the edge pc->pd. */
            /* cop = LCOPLANE; */
            break;
          }
          /*  pa->'endpt' crosses the face bcd. */
          /* cop = NCOPLANE; */
          break;
        }
      }
    }

    /*  Move to the next tet, fix pa as its origin. */
    if (nextmove == RMOVE) {
      tfnextself(m, searchtet);
    } else if (nextmove == LMOVE) {
      enext2self(searchtet);
      tfnextself(m, searchtet);
      enextself(searchtet);
    } else { /*  HMOVE */
      symedgeself(m, searchtet);
      enextself(searchtet);
    }
    /*  Assume convex case, we should not move to outside. */
    if (searchtet->tet == m->dummytet) {
      /*  This should only happen when the domain is non-convex. */
      if (result) {*result = BELOWHULL2;}
      PetscFunctionReturn(0);
    }
    if (org(searchtet) != pa) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    pb = dest(searchtet);
    pc = apex(searchtet);

  } /*  while (1) */

  /*  Either case INTEREDGE or INTERFACE. */
  /*  Now decide the degenerate cases. */
  if (hori == 0) {
    if (rori == 0) {
      /*  pa->'endpt' is COLLINEAR with pa->pb. */
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }
    if (lori == 0) {
      /*  pa->'endpt' is COLLINEAR with pa->pc. */
      enext2self(searchtet);
      esymself(searchtet);
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }
    /*  pa->'endpt' crosses the edge pb->pc. */
    if (result) {*result = INTEREDGE;}
    PetscFunctionReturn(0);
  }
  if (rori == 0) {
    if (lori == 0) {
      /*  pa->'endpt' is COLLINEAR with pa->pd. */
      fnextself(m, searchtet); /*  face abd. */
      enext2self(searchtet);
      esymself(searchtet);
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }
    /*  pa->'endpt' crosses the edge pb->pd. */
    fnextself(m, searchtet); /*  face abd. */
    esymself(searchtet);
    enextself(searchtet);
    if (result) {*result = INTEREDGE;}
    PetscFunctionReturn(0);
  }
  if (lori == 0) {
    /*  pa->'endpt' crosses the edge pc->pd. */
    enext2fnextself(m, searchtet);  /*  face cad */
    esymself(searchtet);
    if (result) {*result = INTEREDGE;}
    PetscFunctionReturn(0);
  }
  /*  pa->'endpt' crosses the face bcd. */
  if (result) {*result = INTERFACE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFindDirection3"
/*  finddirection3()    Used when finddirection2() returns BELOWHULL2.         */
/* tetgenmesh::finddirection3() */
PetscErrorCode TetGenMeshFindDirection3(TetGenMesh *m, triface *searchtet, point endpt, interresult *result)
{
  TetGenOpts    *b  = m->b;
  ArrayPool *startetlist;
  triface *parytet, oppoface = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  point startpt, pa, pb, pc;
  interresult dir;
  int types[2], poss[4];
  int pos = 0, i, j;
  int isIntersect;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ArrayPoolCreate(sizeof(triface), 8, &startetlist);CHKERRQ(ierr);
  startpt = org(searchtet);
  infect(m, searchtet);
  ierr = ArrayPoolNewIndex(startetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
  *parytet = *searchtet;

  PetscInfo2(b->in, "      Search path (%d, %d) under non-convexity.\n", pointmark(m, startpt), pointmark(m, endpt));

  for(i = 0; i < (int) startetlist->objects; i++) {
    parytet = (triface *) fastlookup(startetlist, i);
    *searchtet = *parytet;
    /*  assert(org(*searchtet) == startpt); */
    adjustedgering_triface(searchtet, CCW);
    if (org(searchtet) != startpt) {
      enextself(searchtet);
      if (org(searchtet) != startpt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    }
    /*  Go to the opposite face of startpt. */
    enextfnext(m, searchtet, &oppoface);
    esymself(&oppoface);
    pa = org(&oppoface);
    pb = dest(&oppoface);
    pc = apex(&oppoface);
    /*  Check if face [a, b, c] intersects the searching path. */
    ierr = TetGenMeshTriEdgeTest(m, pa, pb, pc, startpt, endpt, NULL, 1, types, poss, &isIntersect);CHKERRQ(ierr);
    if (isIntersect) {
      /*  They intersect. Get the type of intersection. */
      dir = (interresult) types[0];
      pos = poss[0];
      break;
    } else {
      dir = DISJOINT;
    }
    /*  Get the neighbor tets. */
    for(j = 0; j < 3; j++) {
      if (j == 0) {
        symedge(m, searchtet, &neightet);
      } else if (j == 1) {
        fnext(m, searchtet, &neightet);
        symedgeself(m, &neightet);
      } else {
        enext2fnext(m, searchtet, &neightet);
        symedgeself(m, &neightet);
      }
      if (neightet.tet != m->dummytet) {
        if (!infected(m, &neightet)) {
          if (org(&neightet) != startpt) esymself(&neightet);
          infect(m, &neightet);
          ierr = ArrayPoolNewIndex(startetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = neightet;
        }
      }
    }
  }

  for(i = 0; i < (int) startetlist->objects; i++) {
    parytet = (triface *) fastlookup(startetlist, i);
    uninfect(m, parytet);
  }
  ierr = ArrayPoolDestroy(&startetlist);CHKERRQ(ierr);

  if (dir == INTERVERT) {
    /*  This path passing a vertex of the face [a, b, c]. */
    if (pos == 0) {
      /*  The path acrosses pa. */
      enext2self(searchtet);
      esymself(searchtet);
    } else if (pos == 1) {
      /*  The path acrosses pa. */
    } else { /*  pos == 2 */
      /*  The path acrosses pc. */
      fnextself(m, searchtet);
      enext2self(searchtet);
      esymself(searchtet);
    }
    if (result) {*result = INTERVERT;}
    PetscFunctionReturn(0);
  }
  if (dir == INTEREDGE) {
    /*  This path passing an edge of the face [a, b, c]. */
    if (pos == 0) {
      /*  The path intersects [pa, pb]. */
    } else if (pos == 1) {
      /*  The path intersects [pb, pc]. */
      fnextself(m, searchtet);
      enext2self(searchtet);
      esymself(searchtet);
    } else { /*  pos == 2 */
      /*  The path intersects [pc, pa]. */
      enext2fnextself(m, searchtet);
      esymself(searchtet);
    }
    if (result) {*result = INTEREDGE;}
    PetscFunctionReturn(0);
  }
  if (dir == INTERFACE) {
    if (result) {*result = INTERFACE;}
    PetscFunctionReturn(0);
  }

  /*  The path does not intersect any tet at pa. */
  if (result) {*result = BELOWHULL2;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshScoutSegment2"
/*  scoutsegment()    Look for a given segment in the tetrahedralization T.    */
/*                                                                             */
/*  Search an edge in the tetrahedralization that matches the given segmment.  */
/*  If such an edge exists, the segment is 'locked' at the edge. 'searchtet'   */
/*  returns this (constrained) edge. Otherwise, the segment is missing.        */
/*                                                                             */
/*  The returned value indicates one of the following cases:                   */
/*    - SHAREEDGE, the segment exists and is inserted in T;                    */
/*    - INTERVERT, the segment intersects a vertex ('refpt').                  */
/*    - INTEREDGE, the segment intersects an edge (in 'searchtet').            */
/*    - INTERFACE, the segment crosses a face (in 'searchtet').                */
/*                                                                             */
/*  If the returned value is INTEREDGE or INTERFACE, i.e., the segment is      */
/*  missing, 'refpt' returns the reference point for splitting thus segment,   */
/*  'searchtet' returns a tet containing the 'refpt'.                          */
/* tetgenmesh::scoutsegment2() */
PetscErrorCode TetGenMeshScoutSegment2(TetGenMesh *m, face *sseg, triface *searchtet, point *refpt, interresult *result)
{
  TetGenOpts    *b  = m->b;
  triface neightet = {PETSC_NULL, 0, 0}, reftet = {PETSC_NULL, 0, 0};
  face    checkseg = {PETSC_NULL, 0};
  point startpt, endpt;
  point pa, pb, pc, pd;
  interresult dir;
  PetscReal angmax, ang;
  long facecount;
  int hitbdry;
  int types[2], poss[4];
  int pos, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Is 'searchtet' a valid handle? */
  if ((!searchtet->tet) || (searchtet->tet == m->dummytet)) {
    startpt = sorg(sseg);
    ierr = TetGenMeshPoint2TetOrg(m, startpt, searchtet);CHKERRQ(ierr);
  } else {
    startpt = sorg(sseg);
  }
  if (org(searchtet) != startpt) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
  endpt = sdest(sseg);

  PetscInfo2(b->in, "    Scout seg (%d, %d).\n", pointmark(m, startpt), pointmark(m, endpt));

  ierr = TetGenMeshFindDirection2(m, searchtet, endpt, &dir);CHKERRQ(ierr);

  if (dir == INTERVERT) {
    pd = dest(searchtet);
    if (pd == endpt) {
      /*  Found! Insert the segment. */
      tsspivot1(m, searchtet, &checkseg);
      if (checkseg.sh == m->dummysh) {
        neightet = *searchtet;
        hitbdry = 0;
        do {
          tssbond1(m, &neightet, sseg);
          tfnextself(m, &neightet);
          if (neightet.tet == m->dummytet) {
            hitbdry++;
            if (hitbdry == 2) break;
            esym(searchtet, &neightet);
            tfnextself(m, &neightet);
            if (neightet.tet == m->dummytet) break;
          }
        } while (neightet.tet != searchtet->tet);
      } else {
        /*  Collision! This can happy during facet recovery. */
        /*  See fig/dump-cavity-case19, -case20. */
        if (checkseg.sh != sseg->sh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      }
      /*  The job is done. */
      if (result) {*result = SHAREEDGE;}
      PetscFunctionReturn(0);
    } else {
      /*  A point is on the path. */
      *refpt = pd;
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }
  }

  PetscInfo2(b->in, "    Scout ref point of seg (%d, %d).\n", pointmark(m, startpt), pointmark(m, endpt));
  facecount = m->across_face_count;

  enextfnextself(m, searchtet); /*  Go to the opposite face. */
  symedgeself(m, searchtet); /*  Enter the adjacent tet. */

  pa = org(searchtet);
  angmax = interiorangle(pa, startpt, endpt, PETSC_NULL);
  *refpt = pa;
  pb = dest(searchtet);
  ang = interiorangle(pb, startpt, endpt, PETSC_NULL);
  if (ang > angmax) {
    angmax = ang;
    *refpt = pb;
  }

  /*  Check whether two segments are intersecting. */
  if (dir == INTEREDGE) {
    tsspivot1(m, searchtet, &checkseg);
    if (checkseg.sh != m->dummysh) {
      ierr = TetGenMeshGetSubsegFarOrg(m, sseg, &startpt);CHKERRQ(ierr);
      ierr = TetGenMeshGetSubsegFarDest(m, sseg, &endpt);CHKERRQ(ierr);
      ierr = TetGenMeshGetSubsegFarOrg(m, &checkseg, &pa);CHKERRQ(ierr);
      ierr = TetGenMeshGetSubsegFarDest(m, &checkseg, &pb);CHKERRQ(ierr);
      SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid PLC. Two segments intersect.\n  1st: (%d, %d), 2nd: (%d, %d).\n", pointmark(m, startpt), pointmark(m, endpt), pointmark(m, pa), pointmark(m, pb));
    }
    m->across_edge_count++;
  }

  pc = apex(searchtet);
  ang = interiorangle(pc, startpt, endpt, PETSC_NULL);
  if (ang > angmax) {
    angmax = ang;
    *refpt = pc;
  }
  reftet = *searchtet; /*  Save the tet containing the refpt. */

  /*  Search intersecting faces along the segment. */
  while(1) {
    pd = oppo(searchtet);

    PetscInfo5(b->in, "      Passing face (%d, %d, %d, %d), dir(%d).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), pointmark(m, pd), (int) dir);
    m->across_face_count++;

    /*  Stop if we meet 'endpt'. */
    if (pd == endpt) break;

    ang = interiorangle(pd, startpt, endpt, PETSC_NULL);
    if (ang > angmax) {
      angmax = ang;
      *refpt = pd;
      reftet = *searchtet;
    }

    /*  Find a face intersecting the segment. */
    if (dir == INTERFACE) {
      /*  One of the three oppo faces in 'searchtet' intersects the segment. */
      neightet.tet = searchtet->tet;
      neightet.ver = 0;
      for(i = 0; i < 3; i++) {
        int isIntersect;

        neightet.loc = locpivot[searchtet->loc][i];
        pa = org(&neightet);
        pb = dest(&neightet);
        pc = apex(&neightet);
        pd = oppo(&neightet); /*  The above point. */
        ierr = TetGenMeshTriEdgeTest(m, pa, pb, pc, startpt, endpt, pd, 1, types, poss, &isIntersect);CHKERRQ(ierr);
        if (isIntersect) {
          dir = (interresult) types[0];
          pos = poss[0];
          break;
        } else {
          dir = DISJOINT;
          pos = 0;
        }
      }
      if (dir == DISJOINT) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    } else { /*  dir == ACROSSEDGE */
      /*  Check the two opposite faces (of the edge) in 'searchtet'. */
      neightet = *searchtet;
      neightet.ver = 0;
      for(i = 0; i < 2; i++) {
        int isIntersect;

        neightet.loc = locverpivot[searchtet->loc][searchtet->ver][i];
        pa = org(&neightet);
        pb = dest(&neightet);
        pc = apex(&neightet);
        pd = oppo(&neightet); /*  The above point. */
        ierr = TetGenMeshTriEdgeTest(m, pa, pb, pc, startpt, endpt, pd, 1, types, poss, &isIntersect);CHKERRQ(ierr);
        if (isIntersect) {
          dir = (interresult) types[0];
          pos = poss[0];
          break;
        } else {
          dir = DISJOINT;
          pos = 0;
        }
      }
      if (dir == DISJOINT) {
        /*  No intersection. Go to the next tet. */
        dir = INTEREDGE;
        tfnextself(m, searchtet);
        continue;
      }
    }

    if (dir == INTERVERT) {
      /*  This segment passing a vertex. Choose it and return. */
      for(i = 0; i < pos; i++) {
        enextself(&neightet);
      }
      pd = org(&neightet);
      if (b->verbose > 2) {
        angmax = interiorangle(pd, startpt, endpt, PETSC_NULL);
      }
      *refpt = pd;
      break;
    }
    if (dir == INTEREDGE) {
      /*  Get the edge intersects with the segment. */
      for(i = 0; i < pos; i++) {
        enextself(&neightet);
      }
    }
    /*  Go to the next tet. */
    symedge(m, &neightet, searchtet);

    if (dir == INTEREDGE) {
      /*  Check whether two segments are intersecting. */
      tsspivot1(m, searchtet, &checkseg);
      if (checkseg.sh != m->dummysh) {
        ierr = TetGenMeshGetSubsegFarOrg(m, sseg, &startpt);CHKERRQ(ierr);
        ierr = TetGenMeshGetSubsegFarDest(m, sseg, &endpt);CHKERRQ(ierr);
        ierr = TetGenMeshGetSubsegFarOrg(m, &checkseg, &pa);CHKERRQ(ierr);
        ierr = TetGenMeshGetSubsegFarDest(m, &checkseg, &pb);CHKERRQ(ierr);
        SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid PLC! Two segments intersect.\n    1st: (%d, %d), 2nd: (%d, %d).\n", pointmark(m, startpt), pointmark(m, endpt), pointmark(m, pa), pointmark(m, pb));
      }
      m->across_edge_count++;
    }
  } /*  while (1) */

  /*  dir is either ACROSSVERT, or ACROSSEDGE, or ACROSSFACE. */
  PetscInfo3(b->in, "      Refpt %d (%g), visited %ld faces.\n", pointmark(m, *refpt), angmax / PETSC_PI * 180.0, m->across_face_count - facecount);
  if (m->across_face_count - facecount > m->across_max_count) {
    m->across_max_count = m->across_face_count - facecount;
  }

  *searchtet = reftet;
  if (result) {*result = dir;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDelaunizeSegments2"
/*  delaunizesegments()    Recover segments in a Delaunay tetrahedralization.  */
/* tetgenmesh::delaunizesegments2() */
PetscErrorCode TetGenMeshDelaunizeSegments2(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  triface searchtet = {PETSC_NULL, 0, 0};
  face    splitsh   = {PETSC_NULL, 0};
  face *psseg, sseg = {PETSC_NULL, 0};
  point refpt, newpt;
  interresult dir;
  PetscBool visflag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Delaunizing segments.\n");

  /*  Loop until 'subsegstack' is empty. */
  while(m->subsegstack->objects > 0l) {
    /*  seglist is used as a stack. */
    m->subsegstack->objects--;
    psseg = (face *) fastlookup(m->subsegstack, m->subsegstack->objects);
    sseg = *psseg;

    if (!sinfected(m, &sseg)) continue; /*  Not a missing segment. */
    suninfect(m, &sseg);

    /*  Insert the segment. */
    searchtet.tet = PETSC_NULL;
    ierr = TetGenMeshScoutSegment2(m, &sseg, &searchtet, &refpt, &dir);CHKERRQ(ierr);

    if (dir != SHAREEDGE) {
      /*  The segment is missing, split it. */
      spivot(&sseg, &splitsh);
      if (dir != INTERVERT) {
        /*  Create the new point. */
        ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
#if 1
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
        getsegmentsplitpoint3(&sseg, refpt, newpt);
#endif
        setpointtype(m, newpt, FREESEGVERTEX);
        setpoint2sh(m, newpt, sencode(&sseg));
        /*  Split the segment by newpt. */
        ierr = TetGenMeshSInsertVertex(m, newpt, &splitsh, &sseg, PETSC_TRUE, PETSC_FALSE, PETSC_NULL);CHKERRQ(ierr);
        /*  Insert newpt into the DT. If 'checksubfaces == 1' the current */
        /*    mesh is constrained Delaunay (but may not Delaunay). */
        visflag = (m->checksubfaces == 1) ? PETSC_TRUE : PETSC_FALSE;
        ierr = TetGenMeshInsertVertexBW(m, newpt, &searchtet, PETSC_TRUE, visflag, PETSC_FALSE, PETSC_FALSE, PETSC_NULL);CHKERRQ(ierr);
      } else {
        point pa, pb;
        ierr = TetGenMeshGetSubsegFarOrg(m, &sseg, &pa);CHKERRQ(ierr);
        ierr = TetGenMeshGetSubsegFarDest(m, &sseg, &pb);CHKERRQ(ierr);
        SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid PLC! A point and a segment intersect.\n  Point: %d. Segment: (%d, %d).\n", pointmark(m, refpt), pointmark(m, pa), pointmark(m, pb));
      }
    }
  }

  PetscInfo1(b->in, "  %ld protecting points.\n", m->r1count + m->r2count + m->r3count);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshScoutSubface"
/*  scoutsubface()    Look for a given subface in the tetrahedralization T.    */
/*                                                                             */
/*  'ssub' is the subface, denoted as abc. If abc exists in T, it is 'locked'  */
/*  at the place where the two tets sharing at it.                             */
/*                                                                             */
/*  'convexflag' indicates the current mesh is convex (1) or non-convex (0).   */
/*                                                                             */
/*  The returned value indicates one of the following cases:                   */
/*    - SHAREFACE, abc exists and is inserted;                                 */
/*    - TOUCHEDGE, a vertex (the origin of 'searchtet') lies on ab.            */
/*    - EDGETRIINT, all three edges of abc are missing.                        */
/*    - ACROSSTET, a tet (in 'searchtet') crosses the facet containg abc.      */
/*                                                                             */
/*  If the retunred value is ACROSSTET, the subface is missing.  'searchtet'   */
/*  returns a tet which shares the same edge as 'pssub'.                       */
/* tetgenmesh::scoutsubface() */
PetscErrorCode TetGenMeshScoutSubface(TetGenMesh *m, face *pssub, triface *searchtet, int convexflag, interresult *result)
{
  TetGenOpts    *b  = m->b;
  triface spintet = {PETSC_NULL, 0, 0};
  face    checksh = {PETSC_NULL, 0};
  point pa, pb, pc, pd;
  interresult dir;
  int hitbdry;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((!searchtet->tet) || (searchtet->tet == m->dummytet)) {
    /*  Search an edge of 'ssub' in tetrahedralization. */
    pssub->shver = 0;
    for(i = 0; i < 3; i++) {
      pa = sorg(pssub);
      pb = sdest(pssub);
      /*  Get a tet whose origin is pa. */
      ierr = TetGenMeshPoint2TetOrg(m, pa, searchtet);CHKERRQ(ierr);
      /*  Search the edge from pa->pb. */
      ierr = TetGenMeshFindDirection2(m, searchtet, pb, &dir);CHKERRQ(ierr);
      if (dir == INTERVERT) {
        if (dest(searchtet) == pb) {
          /*  Found the edge. Break the loop. */
          break;
        } else {
          /*  A vertex lies on the search edge. Return it. */
          enextself(searchtet);
          if (result) {*result = TOUCHEDGE;}
          PetscFunctionReturn(0);
        }
      } else if (dir == BELOWHULL2) {
        if (convexflag > 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        /*  The domain is non-convex, and we got stucked at a boundary face. */
        ierr = TetGenMeshPoint2TetOrg(m, pa, searchtet);CHKERRQ(ierr);
        ierr = TetGenMeshFindDirection3(m, searchtet, pb, &dir);CHKERRQ(ierr);
        if (dir == INTERVERT) {
          if (dest(searchtet) == pb) {
            /*  Found the edge. Break the loop. */
            break;
          } else {
            /*  A vertex lies on the search edge. Return it. */
            enextself(searchtet);
            if (result) {*result = TOUCHEDGE;}
            PetscFunctionReturn(0);
          }
        }
      }
      senextself(pssub);
    }
    if (i == 3) {
      /*  None of the three edges exists. */
      if (result) {*result = EDGETRIINT;} /*  ab intersects the face in 'searchtet'. */
      PetscFunctionReturn(0);
    }
  } else {
    /*  'searchtet' holds the current edge of 'pssub'. */
    pa = org(searchtet);
    pb = dest(searchtet);
  }

  pc = sapex(pssub);

  PetscInfo4(b->in, "    Scout subface (%d, %d, %d) (%ld).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), m->subfacstack->objects);

  /*  Searchtet holds edge pa->pb. Search a face with apex pc. */
  spintet = *searchtet;
  pd = apex(&spintet);
  hitbdry = 0;
  while (1) {
    if (pd == pc) {
      /*  Found! Insert the subface. */
      tspivot(m, &spintet, &checksh); /*  SELF_CHECK */
      if (checksh.sh == m->dummysh) {
        /*  Comment: here we know that spintet and pssub refer to the same */
        /*    edge and the same DIRECTION: pa->pb. */
        if ((spintet.ver & 1) == 1) {
          /*  Stay in CCW edge ring. */
          esymself(&spintet);
        }
        if (sorg(pssub) != org(&spintet)) {
          sesymself(pssub);
        }
        tsbond(m, &spintet, pssub);
        symself(&spintet);
        if (spintet.tet != m->dummytet) {
          tspivot(m, &spintet, &checksh); /*  SELF_CHECK */
          if (checksh.sh != m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          sesymself(pssub);
          tsbond(m, &spintet, pssub);
        }
        if (result) {*result = SHAREFACE;}
        PetscFunctionReturn(0);
      } else {
        *searchtet = spintet;
        if (checksh.sh != pssub->sh) {
          /*  Another subface is laready inserted. */
          /*  Comment: This is possible when there are faked tets. */
          if (result) {*result = COLLISIONFACE;}
          PetscFunctionReturn(0);
        } else {
          /*  The subface has already been inserted (when you do check). */
          if (result) {*result = SHAREFACE;}
          PetscFunctionReturn(0);
        }
      }
    }
    if (!fnextself(m, &spintet)) {
      hitbdry++;
      if (hitbdry == 2) break;
      esym(searchtet, &spintet);
      if (!fnextself(m, &spintet)) break;
    }
    pd = apex(&spintet);
    if (pd == apex(searchtet)) break;
  }
  if (result) {*result = INTERTET;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshScoutCrossTet"
/*  scoutcrosstet()    Scout a tetrahedron across a facet.                     */
/*                                                                             */
/*  A subface (abc) of the facet (F) is given in 'pssub', 'searchtet' holds    */
/*  the edge ab, it is the tet starting the search.  'facpoints' contains all  */
/*  points which are co-facet with a, b, and c.                                */
/*                                                                             */
/*  The subface (abc) was produced by a 2D CDT algorithm under the Assumption  */
/*  that F is flat. In real data, however, F may not be strictly flat.  Hence  */
/*  a tet (abde) that crosses abc may be in one of the two cases: (i) abde     */
/*  intersects F in its interior, or (ii) abde intersects F on its boundary.   */
/*  In case (i) F (or part of it) is missing in DT and needs to be recovered.  */
/*  In (ii) F is not missing, the surface mesh of F needs to be adjusted.      */
/*                                                                             */
/*  This routine distinguishes the two cases by the returned value, which is   */
/*    - INTERTET, if it is case (i), 'searchtet' is abde, d and e lies below   */
/*      and above abc, respectively, neither d nor e is dummypoint; or         */
/*    - INTERFACE, if it is case (ii), 'searchtet' is abde, where the face     */
/*      abd intersects abc, i.e., d is co-facet with abc, e may be co-facet    */
/*      with abc or dummypoint.                                                */
/* tetgenmesh::scoutcrosstet() */
PetscErrorCode TetGenMeshScoutCrossTet(TetGenMesh *m, face *pssub, triface *searchtet, ArrayPool *facpoints, interresult *result)
{
  TetGenOpts    *b  = m->b;
  triface spintet = {PETSC_NULL, 0, 0}, crossface = {PETSC_NULL, 0, 0};
  point pa, pb, pc, pd, pe;
  PetscReal ori, ori1, len, n[3];
  PetscReal r, dr, drmin = 0.0;
  PetscBool cofacetflag;
  int hitbdry;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (facpoints) {
    /*  Infect all vertices of the facet. */
    for(i = 0; i < (int) facpoints->objects; i++) {
      pd = * (point *) fastlookup(facpoints, i);
      pinfect(m, pd);
    }
  }

  /*  Search an edge crossing the facet containing abc. */
  if (searchtet->ver & 01) {
    esymself(searchtet); /*  Adjust to 0th edge ring. */
    sesymself(pssub);
  }

  pa = sorg(pssub);
  pb = sdest(pssub);
  pc = sapex(pssub);

  /*  'searchtet' refers to edge pa->pb. */
  if (org(searchtet)  != pa) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
  if (dest(searchtet) != pb) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

  /*  Search an apex lies below the subface. Note that such apex may not */
  /*    exist which indicates there is a co-facet apex. */
  cofacetflag = PETSC_FALSE;
  pd = apex(searchtet);
  spintet = *searchtet;
  hitbdry = 0;
  while (1) {
    ori = TetGenOrient3D(pa, pb, pc, pd);
    if ((ori != 0) && pinfected(m, pd)) {
      ori = 0; /*  Force d be co-facet with abc. */
    }
    if (ori > 0) {
      break; /*  Found a lower point (the apex of spintet). */
    }
    /*  Go to the next face. */
    if (!fnextself(m, &spintet)) {
      hitbdry++;
      if (hitbdry == 2) {
        cofacetflag = PETSC_TRUE; break; /*  Not found. */
      }
      esym(searchtet, &spintet);
      if (!fnextself(m, &spintet)) {
        cofacetflag = PETSC_TRUE; break; /*  Not found. */
      }
    }
    pd = apex(&spintet);
    if (pd == apex(searchtet)) {
      cofacetflag = PETSC_TRUE; break; /*  Not found. */
    }
  }

  if (!cofacetflag) {
    if (hitbdry > 0) {
      /*  The edge direction is reversed, which means we have to reverse */
      /*    the face rotation direction to find the crossing edge d->e. */
      esymself(&spintet);
    }
    /*  Keep the edge a->b be in the CCW edge ring of spintet. */
    if (spintet.ver & 1) {
      symedgeself(m, &spintet);
      if (spintet.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    }
    /*  Search a tet whose apex->oppo crosses the face [a, b, c]. */
    /*    -- spintet is a face [a, b, d]. */
    /*    -- the apex (d) of spintet is below [a, b, c]. */
    while (1) {
      pe = oppo(&spintet);
      ori = TetGenOrient3D(pa, pb, pc, pe);
      if ((ori != 0) && pinfected(m, pe)) {
        ori = 0; /*  Force it to be a coplanar point. */
      }
      if (ori == 0) {
        cofacetflag = PETSC_TRUE;
        break; /*  Found a co-facet point. */
      }
      if (ori < 0) {
        *searchtet = spintet;
        break;  /*  Found. edge [d, e]. */
      }
      /*  Go to the next tet. */
      tfnextself(m, &spintet);
      if (spintet.tet == m->dummytet) {
        cofacetflag = PETSC_TRUE;
        break; /*  There is a co-facet point. */
      }
    }
    /*  Now if "cofacetflag != true", searchtet contains a cross tet (abde), */
    /*    where d and e lie below and above abc, respectively, and */
    /*    TetGenOrient3D(a, b, d, e) < 0. */
  }

  if (cofacetflag) {
    /*  There are co-facet points. Calculate a point above the subface. */
    ierr = TetGenMeshFaceNormal2(m, pa, pb, pc, n, 1);CHKERRQ(ierr);
    len = sqrt(DOT(n, n));
    n[0] /= len;
    n[1] /= len;
    n[2] /= len;
    len = DIST(pa, pb);
    len += DIST(pb, pc);
    len += DIST(pc, pa);
    len /= 3.0;
    m->dummypoint[0] = pa[0] + len * n[0];
    m->dummypoint[1] = pa[1] + len * n[1];
    m->dummypoint[2] = pa[2] + len * n[2];
    /*  Search a co-facet point d, s.t. (i) [a, b, d] intersects [a, b, c], */
    /*    AND (ii) a, b, c, d has the closet circumradius of [a, b, c]. */
    /*  NOTE: (ii) is needed since there may be several points satisfy (i). */
    /*    For an example, see file2.poly. */
    ierr = TetGenMeshCircumsphere(m, pa, pb, pc, PETSC_NULL, n, &r, PETSC_NULL);CHKERRQ(ierr);
    crossface.tet = PETSC_NULL;
    pe = apex(searchtet);
    spintet = *searchtet;
    hitbdry = 0;
    while (1) {
      pd = apex(&spintet);
      ori = TetGenOrient3D(pa, pb, pc, pd);
      if ((ori == 0) || pinfected(m, pd)) {
        ori1 = TetGenOrient3D(pa, pb, m->dummypoint, pd);
        if (ori1 > 0) {
          /*  [a, b, d] intersects with [a, b, c]. */
          if (pinfected(m, pd)) {
            len = DIST(n, pd);
            dr = fabs(len - r);
            if (crossface.tet == PETSC_NULL) {
              /*  This is the first cross face. */
              crossface = spintet;
              drmin = dr;
            } else {
              if (dr < drmin) {
                crossface = spintet;
                drmin = dr;
              }
            }
          } else {
            if (ori != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            /*  Found a coplanar but not co-facet point (pd). */
            SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error:  Invalid PLC! A point and a subface intersect\n  Point %d. Subface (#%d) (%d, %d, %d)\n",
                     pointmark(m, pd), shellmark(m, pssub), pointmark(m, pa), pointmark(m, pb), pointmark(m, pc));
          }
        }
      }
      /*  Go to the next face. */
      if (!fnextself(m, &spintet)) {
        hitbdry++;
        if (hitbdry == 2) break;
        esym(searchtet, &spintet);
        if (!fnextself(m, &spintet)) break;
      }
      if (apex(&spintet) == pe) {
        break;
      }
    }
    if(crossface.tet == PETSC_NULL) {
      if (!crossface.tet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not handled yet");
    }
    *searchtet = crossface;
    m->dummypoint[0] = m->dummypoint[1] = m->dummypoint[2] = 0;
  }

  if (cofacetflag) {
    PetscInfo4(b->in, "    Found a co-facet face (%d, %d, %d) op (%d).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, apex(searchtet)), pointmark(m, oppo(searchtet)));
    if (facpoints) {
      /*  Unmark all facet vertices. */
      for(i = 0; i < (int) facpoints->objects; i++) {
        pd = * (point *) fastlookup(facpoints, i);
        puninfect(m, pd);
      }
    }
    /*  Comment: Now no vertex is infected. */
    if (result) {*result = INTERFACE;}
  } else {
    /*  Return a crossing tet. */
    PetscInfo4(b->in, "    Found a crossing tet (%d, %d, %d, %d).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, apex(searchtet)), pointmark(m, pe));
    /*  Comment: if facpoints != NULL, co-facet vertices are stll infected. */
    /*    They will be uninfected in formcavity(); */
    if (result) {*result = INTERTET;} /*  abc intersects the volume of 'searchtet'. */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRecoverSubfaceByFlips"
/*  recoversubfacebyflips()   Recover a subface by flips in the surface mesh.  */
/*                                                                             */
/*  A subface [a, b, c] ('pssub') intersects with a face [a, b, d] ('cross-    */
/*  face'), where a, b, c, and d belong to the same facet.  It indicates that  */
/*  the face [a, b, d] should appear in the surface mesh.                      */
/*                                                                             */
/*  This routine recovers [a, b, d] in the surface mesh through a sequence of  */
/*  2-to-2 flips. No Steiner points is needed. 'pssub' returns [a, b, d].      */
/*                                                                             */
/*  If 'facfaces' is not NULL, all flipped subfaces are queued for recovery.   */
/* tetgenmesh::recoversubfacebyflips() */
PetscErrorCode TetGenMeshRecoverSubfaceByFlips(TetGenMesh *m, face *pssub, triface *crossface, ArrayPool *facfaces)
{
  triface neightet = {PETSC_NULL, 0, 0};
  face flipfaces[2], *parysh;
  face checkseg = {PETSC_NULL, 0};
  point pa, pb, pc, pd, pe;
  PetscReal ori, len, n[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Initialize faces */
  flipfaces[0] = flipfaces[1] = checkseg;
  /*  Get the missing subface is [a, b, c]. */
  pa = sorg(pssub);
  pb = sdest(pssub);
  pc = sapex(pssub);

  /*  The crossface is [a, b, d, e]. */
  /*  assert(org(*crossface) == pa); */
  /*  assert(dest(*crossface) == pb); */
  pd = apex(crossface);
  pe = m->dummypoint; /*  oppo(*crossface); */

  if (pe == m->dummypoint) {
    /*  Calculate a point above the faces. */
    ierr = TetGenMeshFaceNormal2(m, pa, pb, pd, n, 1);CHKERRQ(ierr);
    len = sqrt(DOT(n, n));
    n[0] /= len;
    n[1] /= len;
    n[2] /= len;
    len = DIST(pa, pb);
    len += DIST(pb, pd);
    len += DIST(pd, pa);
    len /= 3.0;
    pe[0] = pa[0] + len * n[0];
    pe[1] = pa[1] + len * n[1];
    pe[2] = pa[2] + len * n[2];
  }

  /*  Adjust face [a, b, c], so that edge [b, c] crosses edge [a, d]. */
  ori = TetGenOrient3D(pb, pc, pe, pd);
  if (ori == 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

  if (ori > 0) {
    /*  Swap a and b. */
    sesymself(pssub);
    esymself(crossface); /*  symedgeself(*crossface); */
    pa = sorg(pssub);
    pb = sdest(pssub);
    if (pe == m->dummypoint) {
      pe[0] = pe[1] = pe[2] = 0;
    }
    pe = m->dummypoint; /*  oppo(*crossface); */
  }

  while (1) {
    /*  Flip edge [b, c] to edge [a, d]. */
    senext(pssub, &flipfaces[0]);
    sspivot(m, &flipfaces[0], &checkseg);
    if (checkseg.sh != m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    spivot(&flipfaces[0], &flipfaces[1]);

    stpivot(m, &flipfaces[1], &neightet);
    if (neightet.tet != m->dummytet) {
      /*  A recovered subface, clean sub<==>tet connections. */
      tsdissolve(m, &neightet);
      symself(&neightet);
      tsdissolve(m, &neightet);
      stdissolve(m, &flipfaces[1]);
      sesymself(&flipfaces[1]);
      stdissolve(m, &flipfaces[1]);
      sesymself(&flipfaces[1]);
      /*  flipfaces[1] refers to edge [b, c] (either b->c or c->b). */
    }

    ierr = TetGenMeshFlip22Sub(m, &(flipfaces[0]), PETSC_NULL);CHKERRQ(ierr);
    m->flip22count++;

    /*  Comment: now flipfaces[0] is [d, a, b], flipfaces[1] is [a, d, c]. */

    /*  Add them into list (make ensure that they must be recovered). */
    ierr = ArrayPoolNewIndex(facfaces, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
    *parysh = flipfaces[0];
    ierr = ArrayPoolNewIndex(facfaces, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
    *parysh = flipfaces[1];

    /*  Find the edge [a, b]. */
    senext(&flipfaces[0], pssub);
    if (sorg(pssub)  != pa) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    if (sdest(pssub) != pb) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

    pc = sapex(pssub);
    if (pc == pd) break;

    if (pe == m->dummypoint) {
      /*  Calculate a point above the faces. */
      ierr = TetGenMeshFaceNormal2(m, pa, pb, pd, n, 1);CHKERRQ(ierr);
      len = sqrt(DOT(n, n));
      n[0] /= len;
      n[1] /= len;
      n[2] /= len;
      len = DIST(pa, pb);
      len += DIST(pb, pd);
      len += DIST(pd, pa);
      len /= 3.0;
      pe[0] = pa[0] + len * n[0];
      pe[1] = pa[1] + len * n[1];
      pe[2] = pa[2] + len * n[2];
    }

    while(1) {
      ori = TetGenOrient3D(pb, pc, pe, pd);
      if (ori == 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
      if (ori > 0) {
        senext2self(pssub);
        spivotself(pssub);
        if (sorg(pssub) != pa) sesymself(pssub);
        pb = sdest(pssub);
        pc = sapex(pssub);
        continue;
      }
      break;
    }
  }

  if (pe == m->dummypoint) {
    pe[0] = pe[1] = pe[2] = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormCavity"
/*  formcavity()    Form the cavity of a missing region.                       */
/*                                                                             */
/*  A missing region R is a set of co-facet (co-palanr) subfaces. 'pssub' is   */
/*  a missing subface [a, b, c]. 'crosstets' contains only one tet, [a, b, d,  */
/*  e], where d and e lie below and above [a, b, c], respectively.  Other      */
/*  crossing tets are sought from this tet and saved in 'crosstets'.           */
/*                                                                             */
/*  The cavity C is divided into two parts by R,one at top and one at bottom.  */
/*  'topfaces' and 'botfaces' return the upper and lower boundary faces of C.  */
/*  'toppoints' contains vertices of 'crosstets' in the top part of C, and so  */
/*  does 'botpoints'. Both 'toppoints' and 'botpoints' contain vertices of R.  */
/*                                                                             */
/*  NOTE: 'toppoints' may contain points which are not vertices of any top     */
/*  faces, and so may 'botpoints'. Such points may belong to other facets and  */
/*  need to be present after the recovery of this cavity (P1029.poly).         */
/*                                                                             */
/*  A pair of boundary faces: 'firsttopface' and 'firstbotface', are saved.    */
/*  They share the same edge in the boundary of the missing region.            */
/*                                                                             */
/*  'facpoints' contains all vertices of the facet containing R.  They are     */
/*  used for searching the crossing tets. On input all vertices are infected.  */
/*  They are uninfected after the cavity is formed.                            */
/* tetgenmesh::formcavity() */
PetscErrorCode TetGenMeshFormCavity(TetGenMesh *m, face *pssub, ArrayPool *crosstets, ArrayPool *topfaces, ArrayPool *botfaces, ArrayPool *toppoints, ArrayPool *botpoints, ArrayPool *facpoints, ArrayPool *facfaces)
{
  TetGenOpts    *b  = m->b;
  ArrayPool *crossedges;
  triface *parytet, crosstet = {PETSC_NULL, 0, 0}, spintet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0}, faketet = {PETSC_NULL, 0, 0};
  face neighsh  = {PETSC_NULL, 0}, checksh = {PETSC_NULL, 0}, *parysh;
  face checkseg = {PETSC_NULL, 0};
  point pa, pb, pc, pf, pg;
  point pd, pe;
  point *ppt;
  int i, j;
  /*  For triangle-edge test. */
  interresult dir;
  int isIntersect;
  int types[2], poss[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Get the missing subface abc. */
  pa = sorg(pssub);
  pb = sdest(pssub);
  pc = sapex(pssub);

  /*  Comment: Now all facet vertices are infected. */

  /*  Get a crossing tet abde. */
  parytet = (triface *) fastlookup(crosstets, 0); /*  face abd. */
  /*  The edge de crosses the facet. d lies below abc. */
  enext2fnext(m, parytet, &crosstet);
  enext2self(&crosstet);
  esymself(&crosstet); /*  the edge d->e at face [d,e,a] */
  infect(m, &crosstet);
  *parytet = crosstet; /*  Save it in list. */

  /*  Temporarily re-use 'topfaces' for storing crossing edges. */
  crossedges = topfaces;
  ierr = ArrayPoolNewIndex(crossedges, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
  *parytet = crosstet;

  /*  Collect all crossing tets.  Each cross tet is saved in the standard */
  /*    form deab, where de is a corrsing edge, TetGenOrient3D(d,e,a,b) < 0. */
  /*  NOTE: hull tets may be collected. See fig/dump-cavity-case2a(b).lua. */
  /*    Make sure that neither d nor e is dummypoint. */
  for(i = 0; i < (int) crossedges->objects; i++) {
    crosstet = * (triface *) fastlookup(crossedges, i);
    /*  It may already be tested. */
    if (!edgemarked(m, &crosstet)) {
      /*  Collect all tets sharing at the edge. */
      pg = apex(&crosstet);
      spintet = crosstet;
      while (1) {
        /*  Mark this edge as tested. */
        markedge(m, &spintet);
        if (!infected(m, &spintet)) {
          infect(m, &spintet);
          ierr = ArrayPoolNewIndex(crosstets, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = spintet;
        }
        /*  Go to the neighbor tet. */
        tfnextself(m, &spintet);
        if (spintet.tet != m->dummytet) {
          /*  Check the validity of the PLC. */
          tspivot(m, &spintet, &checksh);
          if (checksh.sh != m->dummysh) SETERRQ8(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error:  Invalid PLC! Two subfaces intersect.\n  1st (#%4d): (%d, %d, %d)\n  2nd (#%4d): (%d, %d, %d)\n",
                     shellmark(m, pssub), pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), shellmark(m, &checksh),
                     pointmark(m, sorg(&checksh)), pointmark(m, sdest(&checksh)), pointmark(m, sapex(&checksh)));
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not handled yet"); /*  Encounter a boundary face. */
        if (apex(&spintet) == pg) break;
      }
      /*  Detect new cross edges. */
      /*  Comment: A crossing edge must intersect one missing subface of */
      /*    this facet. We do edge-face tests. */
      pd = org(&spintet);
      pe = dest(&spintet);
      while (1) {
        /*  Remember: spintet is edge d->e, d lies below [a, b, c]. */
        pf = apex(&spintet);
        /*  if (pf != dummypoint) {  Do not grab a hull edge. */
        if (!pinfected(m, pf)) {
            for(j = 0; j < (int) facfaces->objects; j++) {
              parysh = (face *) fastlookup(facfaces, j);
              pa = sorg(parysh);
              pb = sdest(parysh);
              pc = sapex(parysh);
              /*  Check if pd->pf crosses the facet. */
              ierr = TetGenMeshTriEdgeTest(m, pa, pb, pc, pd, pf, PETSC_NULL, 1, types, poss, &isIntersect);CHKERRQ(ierr);
              if (isIntersect) {
                dir = (interresult) types[0];
                if ((dir == INTEREDGE) || (dir == INTERFACE)) {
                  /*  The edge d->f corsses the facet. */
                  enext2fnext(m, &spintet, &neightet);
                  esymself(&neightet); /*  d->f. */
                  /*  pd must lie below the subface. */
                  break;
                }
              }
              /*  Check if pe->pf crosses the facet. */
              ierr = TetGenMeshTriEdgeTest(m, pa, pb, pc, pe, pf, PETSC_NULL, 1, types, poss, &isIntersect);CHKERRQ(ierr);
              if (isIntersect) {
                dir = (interresult) types[0];
                if ((dir == INTEREDGE) || (dir == INTERFACE)) {
                  /*  The edge f->e crosses the face. */
                  enextfnext(m, &spintet, &neightet);
                  esymself(&neightet); /*  f->e. */
                  /*  pf must lie below the subface. */
                  break;
                }
              }
            }
            /*  There must exist a crossing edge. */
            if (j >= (int) facfaces->objects) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            if (!edgemarked(m, &neightet)) {
              /*  Add a new cross edge. */
              ierr = ArrayPoolNewIndex(crossedges, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
              *parytet = neightet;
            }
          }
        /*  } */
        tfnextself(m, &spintet);
        if (spintet.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not handled yet");/*  Encounter a boundary face. */
        if (apex(&spintet) == pg) break;
      }
    }
  }

  /*  Unmark all facet vertices. */
  for(i = 0; i < (int) facpoints->objects; i++) {
    ppt = (point *) fastlookup(facpoints, i);
    puninfect(m, *ppt);
  }

  /*  Comments: Now no vertex is marked. Next we will mark vertices which  */
  /*    belong to the top and bottom boundary faces of the cavity and put */
  /*    them in 'toppopints' and 'botpoints', respectively. */

  /*  All cross tets are found. Unmark cross edges. */
  for(i = 0; i < (int) crossedges->objects; i++) {
    crosstet = * (triface *) fastlookup(crossedges, i);
    if (edgemarked(m, &crosstet)) {
      /*  Add the vertices of the cross edge [d, e] in lists. It must be */
      /*    that d lies below the facet (i.e., its a bottom vertex). */
      /*    Note that a cross edge contains no dummypoint. */
      pf = org(&crosstet);
      /*  assert(pf != dummypoint);  SELF_CHECK */
      if (!pinfected(m, pf)) {
        pinfect(m, pf);
        ierr = ArrayPoolNewIndex(botpoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); /*  Add a bottom vertex. */
        *ppt = pf;
      }
      pf = dest(&crosstet);
      /*  assert(pf != dummypoint);  SELF_CHECK */
      if (!pinfected(m, pf)) {
        pinfect(m, pf);
        ierr = ArrayPoolNewIndex(toppoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); /*  Add a top vertex. */
        *ppt = pf;
      }
      /*  Unmark this edge in all tets containing it. */
      pg = apex(&crosstet);
      spintet = crosstet;
      while (1) {
        if (!edgemarked(m, &spintet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        unmarkedge(m, &spintet);
        tfnextself(m, &spintet); /*  Go to the neighbor tet. */
        if (spintet.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not handled yet");
        if (apex(&spintet) == pg) break;
      }
    }
  }

  PetscInfo2(b->in, "    Formed cavity: %ld (%ld) cross tets (edges).\n", crosstets->objects, crossedges->objects);
  ierr = ArrayPoolRestart(crossedges);CHKERRQ(ierr);

  /*  Find a pair of cavity boundary faces from the top and bottom sides of */
  /*    the facet each, and they share the same edge. Save them in the */
  /*    global variables: firsttopface, firstbotface. They will be used in */
  /*    fillcavity() for gluing top and bottom new tets. */
  for(i = 0; i < (int) crosstets->objects; i++) {
    crosstet = * (triface *) fastlookup(crosstets, i);
    enextfnext(m, &crosstet, &spintet);
    enextself(&spintet);
    symedge(m, &spintet, &neightet);
    /*  if (!infected(neightet)) { */
    if ((neightet.tet == m->dummytet) || !infected(m, &neightet)) {
      /*  A top face. */
      if (neightet.tet == m->dummytet) {
        /*  Create a fake tet to hold the boundary face. */
        ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  /*  Create a faked tet. */
        setorg(&faketet, org(&spintet));
        setdest(&faketet, dest(&spintet));
        setapex(&faketet, apex(&spintet));
        setoppo(&faketet, m->dummypoint);
        bond(m, &faketet, &spintet);
        tspivot(m, &spintet, &checksh);
        if (checksh.sh != m->dummysh) {
          sesymself(&checksh);
          tsbond(m, &faketet, &checksh);
        }
        for(j = 0; j < 3; j++) { /*  Bond segments. */
          tsspivot1(m, &spintet, &checkseg);
          if (checkseg.sh != m->dummysh) {
            tssbond1(m, &faketet, &checkseg);
          }
          enextself(&spintet);
          enextself(&faketet);
        }
        m->firsttopface = faketet;
      } else {
        m->firsttopface = neightet;
      }
    } else {
      continue; /*  Go to the next cross tet. */
    }
    enext2fnext(m, &crosstet, &spintet);
    enext2self(&spintet);
    symedge(m, &spintet, &neightet);
    /*  if (!infected(neightet)) { */
    if ((neightet.tet == m->dummytet) || !infected(m, &neightet)) {
      /*  A bottom face. */
      if (neightet.tet == m->dummytet) {
        /*  Create a fake tet to hold the boundary face. */
        ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  /*  Create a faked tet. */
        setorg(&faketet, org(&spintet));
        setdest(&faketet, dest(&spintet));
        setapex(&faketet, apex(&spintet));
        setoppo(&faketet, m->dummypoint);
        bond(m, &spintet, &faketet);
        tspivot(m, &spintet, &checksh);
        if (checksh.sh != m->dummysh) {
          sesymself(&checksh);
          tsbond(m, &faketet, &checksh);
        }
        for(j = 0; j < 3; j++) { /*  Bond segments. */
          tsspivot1(m, &spintet, &checkseg);
          if (checkseg.sh != m->dummysh) {
            tssbond1(m, &faketet, &checkseg);
          }
          enextself(&spintet);
          enextself(&faketet);
        }
        m->firstbotface = faketet;
      } else {
        m->firstbotface = neightet;
      }
    } else {
      continue;
    }
    break;
  }
  if (i >= (int) crosstets->objects) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

  /*  Collect the top and bottom faces and the middle vertices. Since all top */
  /*    and bottom vertices have been marked in above. Unmarked vertices are */
  /*    middle vertices. */
  /*  NOTE 1: Hull tets may be collected. Process them as normal one. */
  /*    (see fig/dump-cavity-case2.lua.) */
  /*  NOTE 2: Some previously recovered subfaces may be completely */
  /*    contained in a cavity (see fig/dump-cavity-case6.lua). In such case, */
  /*    we create two faked tets to hold this subface, one at each side. */
  /*    The faked tets will be removed in fillcavity(). */
  for(i = 0; i < (int) crosstets->objects; i++) {
    crosstet = * (triface *) fastlookup(crosstets, i);
    enextfnext(m, &crosstet, &spintet);
    enextself(&spintet);
    symedge(m, &spintet, &neightet);
    /*  if (!infected(neightet)) { */
    if ((neightet.tet == m->dummytet) || !infected(m, &neightet)) {
      /*  A top face. */
      ierr = ArrayPoolNewIndex(topfaces, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      if (neightet.tet == m->dummytet) {
        /*  Create a fake tet to hold the boundary face. */
        ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  /*  Create a faked tet. */
        setorg(&faketet, org(&spintet));
        setdest(&faketet, dest(&spintet));
        setapex(&faketet, apex(&spintet));
        setoppo(&faketet, m->dummypoint);
        bond(m, &spintet, &faketet);
        tspivot(m, &spintet, &checksh);
        if (checksh.sh != m->dummysh) {
          sesymself(&checksh);
          tsbond(m, &faketet, &checksh);
        }
        for(j = 0; j < 3; j++) { /*  Bond segments. */
          tsspivot1(m, &spintet, &checkseg);
          if (checkseg.sh != m->dummysh) {
            tssbond1(m, &faketet, &checkseg);
          }
          enextself(&spintet);
          enextself(&faketet);
        }
        *parytet = faketet;
      } else {
        *parytet = neightet;
      }
    } else {
      if ((neightet.tet != m->dummytet) && infected(m, &neightet)) {
        /*  Check if this side is a subface. */
        tspivot(m, &spintet, &neighsh);
        if (neighsh.sh != m->dummysh) {
          /*  Found a subface (inside the cavity)! */
          ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  /*  Create a faked tet. */
          setorg(&faketet, org(&spintet));
          setdest(&faketet, dest(&spintet));
          setapex(&faketet, apex(&spintet));
          setoppo(&faketet, m->dummypoint);
          marktest(m, &faketet);  /*  To distinguish it from other faked tets. */
          sesymself(&neighsh);
          tsbond(m, &faketet, &neighsh); /*  Let it hold the subface. */
          for(j = 0; j < 3; j++) { /*  Bond segments. */
            tsspivot1(m, &spintet, &checkseg);
            if (checkseg.sh != m->dummysh) {
              tssbond1(m, &faketet, &checkseg);
            }
            enextself(&spintet);
            enextself(&faketet);
          }
          /*  Add a top face (at faked tet). */
          ierr = ArrayPoolNewIndex(topfaces, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = faketet;
        }
      }
    }
    enext2fnext(m, &crosstet, &spintet);
    enext2self(&spintet);
    symedge(m, &spintet, &neightet);
    /*  if (!infected(neightet)) { */
    if ((neightet.tet == m->dummytet) || !infected(m, &neightet)) {
      /*  A bottom face. */
      ierr = ArrayPoolNewIndex(botfaces, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      if (neightet.tet == m->dummytet) {
        /*  Create a fake tet to hold the boundary face. */
        ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  /*  Create a faked tet. */
        setorg(&faketet, org(&spintet));
        setdest(&faketet, dest(&spintet));
        setapex(&faketet, apex(&spintet));
        setoppo(&faketet, m->dummypoint);
        bond(m, &spintet, &faketet);
        tspivot(m, &spintet, &checksh);
        if (checksh.sh != m->dummysh) {
          sesymself(&checksh);
          tsbond(m, &faketet, &checksh);
        }
        for(j = 0; j < 3; j++) { /*  Bond segments. */
          tsspivot1(m, &spintet, &checkseg);
          if (checkseg.sh != m->dummysh) {
            tssbond1(m, &faketet, &checkseg);
          }
          enextself(&spintet);
          enextself(&faketet);
        }
        *parytet = faketet;
      } else {
        *parytet = neightet;
      }
    } else {
      if ((neightet.tet != m->dummytet) && infected(m, &neightet)) {
        tspivot(m, &spintet, &neighsh);
        if (neighsh.sh != m->dummysh) {
          /*  Found a subface (inside the cavity)! */
          ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  /*  Create a faked tet. */
          setorg(&faketet, org(&spintet));
          setdest(&faketet, dest(&spintet));
          setapex(&faketet, apex(&spintet));
          setoppo(&faketet, m->dummypoint);
          marktest(m, &faketet);  /*  To distinguish it from other faked tets. */
          sesymself(&neighsh);
          tsbond(m, &faketet, &neighsh); /*  Let it hold the subface. */
          for(j = 0; j < 3; j++) { /*  Bond segments. */
            tsspivot1(m, &spintet, &checkseg);
            if (checkseg.sh != m->dummysh) {
              tssbond1(m, &faketet, &checkseg);
            }
            enextself(&spintet);
            enextself(&faketet);
          }
          /*  Add a bottom face (at faked tet). */
          ierr = ArrayPoolNewIndex(botfaces, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = faketet;
        }
      }
    }
    /*  Add middle vertices if there are (skip dummypoint). */
    pf = org(&spintet);
    if (!pinfected(m, pf)) {
      pinfect(m, pf);
      ierr = ArrayPoolNewIndex(botpoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); /*  Add a bottom vertex. */
      *ppt = pf;
      ierr = ArrayPoolNewIndex(toppoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); /*  Add a top vertex. */
      *ppt = pf;
    }
    pf = dest(&spintet);
    if (!pinfected(m, pf)) {
      pinfect(m, pf);
      ierr = ArrayPoolNewIndex(botpoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); /*  Add a bottom vertex. */
      *ppt = pf;
      ierr = ArrayPoolNewIndex(toppoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); /*  Add a top vertex. */
      *ppt = pf;
    }
  }

  /*  Unmark all collected top, bottom, and middle vertices. */
  for(i = 0; i < (int) toppoints->objects; i++) {
    ppt = (point *) fastlookup(toppoints, i);
    puninfect(m, *ppt);
  }
  for(i = 0; i < (int) botpoints->objects; i++) {
    ppt = (point *) fastlookup(botpoints, i);
    puninfect(m, *ppt);
  }
  /*  Comments: Now no vertex is marked. */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDelaunizeCavity"
/*  delaunizecavity()    Fill a cavity by Delaunay tetrahedra.                 */
/*                                                                             */
/*  The tetrahedralizing cavity is the half (top or bottom part) of the whole  */
/*  cavity.  The boundary faces of the half cavity are given in 'cavfaces',    */
/*  the bounday faces of the internal facet are not given.  These faces will   */
/*  be recovered later in fillcavity().                                        */
/*                                                                             */
/*  This routine first constructs the DT of the vertices by the Bowyer-Watson  */
/*  algorithm.  Then it identifies the boundary faces of the cavity in DT.     */
/*  The DT is returned in 'newtets'.                                           */
/* tetgenmesh::delaunizecavity() */
PetscErrorCode TetGenMeshDelaunizeCavity(TetGenMesh *m, ArrayPool *cavpoints, ArrayPool *cavfaces, ArrayPool *cavshells, ArrayPool *newtets, ArrayPool *crosstets, ArrayPool *misfaces, PetscBool *result)
{
#if 0
  triface *parytet, searchtet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0}, spintet = {PETSC_NULL, 0, 0}, *parytet1;
  triface newtet   = {PETSC_NULL, 0, 0}, faketet = {PETSC_NULL, 0, 0};
  face    checksh  = {PETSC_NULL, 0}, tmpsh = {PETSC_NULL, 0}, *parysh;
  face    checkseg = {PETSC_NULL, 0};
  point pa, pb, pc, pd, pt[3], *parypt;
  interresult dir;
  PetscReal ori;
  int i, j, k;
  PetscErrorCode ierr;
#endif
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFillCavity"
/*  fillcavity()    Fill new tets into the cavity.                             */
/*                                                                             */
/*  The new tets are stored in two disjoint sets(which share the same facet).  */
/*  'topfaces' and 'botfaces' are the boundaries of these two sets, respect-   */
/*  ively. 'midfaces' is empty on input, and will store faces in the facet.    */
/* tetgenmesh::fillcavity() */
PetscErrorCode TetGenMeshFillCavity(TetGenMesh *m, ArrayPool *topshells, ArrayPool *botshells, ArrayPool *midfaces, ArrayPool *facpoints, PetscBool *result)
{
#if 0
  ArrayPool *cavshells;
  triface *parytet, bdrytet = {PETSC_NULL, 0, 0}, toptet = {PETSC_NULL, 0, 0}, bottet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0}, midface = {PETSC_NULL, 0, 0}, spintet = {PETSC_NULL, 0, 0};
  face checksh  = {PETSC_NULL, 0}, *parysh;
  face checkseg = {PETSC_NULL, 0};
  point pa, pb, pc, pf, pg;
  PetscReal ori, len, n[3];
  PetscBool mflag, bflag;
  int i, j, k;
  PetscErrorCode ierr;
#endif
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCarveCavity"
/*  carvecavity()    Delete old tets and outer new tets of the cavity.         */
/* tetgenmesh::carvecavity() */
PetscErrorCode TetGenMeshCarveCavity(TetGenMesh *m, ArrayPool *crosstets, ArrayPool *topnewtets, ArrayPool *botnewtets)
{
  ArrayPool *newtets;
  triface *parytet, *pnewtet, neightet = {PETSC_NULL, 0, 0};
  int i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Delete the old tets in cavity. */
  for(i = 0; i < (int) crosstets->objects; i++) {
    parytet = (triface *) fastlookup(crosstets, i);
    ierr = TetGenMeshTetrahedronDealloc(m, parytet->tet);CHKERRQ(ierr);
  }
  ierr = ArrayPoolRestart(crosstets);CHKERRQ(ierr); /*  crosstets will be re-used. */

  /*  Collect infected new tets in cavity. */
  for(k = 0; k < 2; k++) {
    newtets = (k == 0 ? topnewtets : botnewtets);
    if (newtets) {
      for(i = 0; i < (int) newtets->objects; i++) {
        parytet = (triface *) fastlookup(newtets, i);
        if (infected(m, parytet)) {
          ierr = ArrayPoolNewIndex(crosstets, (void **) &pnewtet, PETSC_NULL);CHKERRQ(ierr);
          *pnewtet = *parytet;
        }
      }
    }
  }
  /*  Collect all new tets in cavity. */
  for(i = 0; i < (int) crosstets->objects; i++) {
    parytet = (triface *) fastlookup(crosstets, i);
    if (i == 0) {
      m->recenttet = *parytet; /*  Remember a live handle. */
    }
    for(j = 0; j < 4; j++) {
      decode(parytet->tet[j], &neightet);
      if (marktested(m, &neightet)) { /*  Is it a new tet? */
        if (!infected(m, &neightet)) {
          /*  Find an interior tet. */
          if (neightet.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          infect(m, &neightet);
          ierr = ArrayPoolNewIndex(crosstets, (void **) &pnewtet, PETSC_NULL);CHKERRQ(ierr);
          *pnewtet = neightet;
        }
      }
    }
  }

  /*  Delete outer new tets (those new tets which are not infected). */
  for(k = 0; k < 2; k++) {
    newtets = (k == 0 ? topnewtets : botnewtets);
    if (newtets != NULL) {
      for(i = 0; i < (int) newtets->objects; i++) {
        parytet = (triface *) fastlookup(newtets, i);
        if (infected(m, parytet)) {
          /*  This is an interior tet. */
          uninfect(m, parytet);
          unmarktest(m, parytet);
        } else {
          /*  An outer tet. Delete it. */
          ierr = TetGenMeshTetrahedronDealloc(m, parytet->tet);CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = ArrayPoolRestart(crosstets);CHKERRQ(ierr);
  ierr = ArrayPoolRestart(topnewtets);CHKERRQ(ierr);
  if (botnewtets) {
    ierr = ArrayPoolRestart(botnewtets);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRestoreCavity"
/*  restorecavity()    Reconnect old tets and delete new tets of the cavity.   */
/* tetgenmesh::restorecavity() */
PetscErrorCode TetGenMeshRestoreCavity(TetGenMesh *m, ArrayPool *crosstets, ArrayPool *topnewtets, ArrayPool *botnewtets)
{
  triface *parytet, neightet = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0};
  point *ppt;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Reconnect crossing tets to cavity boundary. */
  for(i = 0; i < (int) crosstets->objects; i++) {
    parytet = (triface *) fastlookup(crosstets, i);
    if (!infected(m, parytet)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    if (i == 0) {
      m->recenttet = *parytet; /*  Remember a live handle. */
    }
    parytet->ver = 0;
    for(parytet->loc = 0; parytet->loc < 4; parytet->loc++) {
      sym(parytet, &neightet);
      /*  The neighbor may be a deleted faked tet. */
      if (isdead_triface(&neightet) || (neightet.tet == m->dummytet)) {
        dissolve(m, parytet);  /*  Detach a faked tet. */
        /*  Remember a boundary tet. */
        m->dummytet[0] = encode(parytet);
      } else if (!infected(m, &neightet)) {
        bond(m, parytet, &neightet);
        tspivot(m, parytet, &checksh);
        if (checksh.sh != m->dummysh) {
          tsbond(m, parytet, &checksh);
        }
      }
    }
    /*  Update the point-to-tet map. */
    parytet->loc = 0;
    ppt = (point *) &(parytet->tet[4]);
    for(j = 0; j < 4; j++) {
      setpoint2tet(m, ppt[j], encode(parytet));
    }
  }

  /*  Uninfect all crossing tets. */
  for(i = 0; i < (int) crosstets->objects; i++) {
    parytet = (triface *) fastlookup(crosstets, i);
    uninfect(m, parytet);
  }

  /*  Delete new tets. */
  for(i = 0; i < (int) topnewtets->objects; i++) {
    parytet = (triface *) fastlookup(topnewtets, i);
    ierr = TetGenMeshTetrahedronDealloc(m, parytet->tet);CHKERRQ(ierr);
  }

  if (botnewtets) {
    for(i = 0; i < (int) botnewtets->objects; i++) {
      parytet = (triface *) fastlookup(botnewtets, i);
      ierr = TetGenMeshTetrahedronDealloc(m, parytet->tet);CHKERRQ(ierr);
    }
  }

  ierr = ArrayPoolRestart(crosstets);CHKERRQ(ierr);
  ierr = ArrayPoolRestart(topnewtets);CHKERRQ(ierr);
  if (botnewtets) {
    ierr = ArrayPoolRestart(botnewtets);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshConstrainedFacets"
/*  constrainedfacets()    Recover subfaces saved in 'subfacestack'.           */
/* tetgenmesh::constrainedfacets2() */
PetscErrorCode TetGenMeshConstrainedFacets(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  ArrayPool *crosstets, *topnewtets, *botnewtets;
  ArrayPool *topfaces, *botfaces, *midfaces;
  ArrayPool *topshells, *botshells, *facfaces;
  ArrayPool *toppoints, *botpoints, *facpoints;
  triface *parytet, searchtet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  face *pssub, ssub = {PETSC_NULL, 0}, neighsh = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  point *ppt, pt, newpt;
  interresult dir;
  PetscBool success, delaunayflag;
  long bakflip22count;
  long cavitycount;
  int facetcount;
  int bakhullsize;
  int s, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constraining facets.\n");

  /*  Initialize arrays. */
  ierr = ArrayPoolCreate(sizeof(triface), 10, &crosstets);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &topnewtets);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &botnewtets);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &topfaces);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &botfaces);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &midfaces);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(point), 8, &toppoints);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(point), 8, &botpoints);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(point), 8, &facpoints);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &facfaces);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &topshells);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &botshells);CHKERRQ(ierr);

  bakflip22count = m->flip22count;
  cavitycount = 0;
  facetcount = 0;

  /*  Loop until 'subfacstack' is empty. */
  while (m->subfacstack->objects > 0l) {
    m->subfacstack->objects--;
    pssub = (face *) fastlookup(m->subfacstack, m->subfacstack->objects);
    ssub = *pssub;

    if (!ssub.sh[3]) continue; /*  Skip a dead subface. */

    stpivot(m, &ssub, &neightet);
    if (neightet.tet == m->dummytet) {
      sesymself(&ssub);
      stpivot(m, &ssub, &neightet);
    }

    if (neightet.tet == m->dummytet) {
      /*  Find an unrecovered subface. */
      smarktest(&ssub);
      ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
      *pssub = ssub;
      /*  Get all subfaces and vertices of the same facet. */
      for(i = 0; i < (int) facfaces->objects; i++) {
        ssub = * (face *) fastlookup(facfaces, i);
        for (j = 0; j < 3; j++) {
          sspivot(m, &ssub, &checkseg);
          if (checkseg.sh == m->dummysh) {
            spivot(&ssub, &neighsh);
            if (neighsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack should not be empty");
            if (!smarktested(&neighsh)) {
              /*  It may be already recovered. */
              stpivot(m, &neighsh, &neightet);
              if (neightet.tet == m->dummytet) {
                sesymself(&neighsh);
                stpivot(m, &neighsh, &neightet);
              }
              if (neightet.tet == m->dummytet) {
                /*  Add it into list. */
                smarktest(&neighsh);
                ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
                *pssub = neighsh;
              }
            }
          }
          pt = sorg(&ssub);
          if (!pinfected(m, pt)) {
            pinfect(m, pt);
            ierr = ArrayPoolNewIndex(facpoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr);
            *ppt = pt;
          }
          senextself(&ssub);
        } /*  j */
      } /*  i */
      /*  Have found all facet subfaces (vertices). Uninfect them. */
      for(i = 0; i < (int) facfaces->objects; i++) {
        pssub = (face *) fastlookup(facfaces, i);
        sunmarktest(pssub);
      }
      for (i = 0; i < (int) facpoints->objects; i++) {
        ppt = (point *) fastlookup(facpoints, i);
        puninfect(m, *ppt);
      }
      PetscInfo3(b->in, "  Recover facet #%d: %ld subfaces, %ld vertices.\n", facetcount + 1, facfaces->objects, facpoints->objects);
      facetcount++;

      /*  Loop until 'facfaces' is empty. */
      while(facfaces->objects > 0l) {
        /*  Get the last subface of this array. */
        facfaces->objects--;
        pssub = (face *) fastlookup(facfaces, facfaces->objects);
        ssub = *pssub;

        stpivot(m, &ssub, &neightet);
        if (neightet.tet == m->dummytet) {
          sesymself(&ssub);
          stpivot(m, &ssub, &neightet);
        }

        if (neightet.tet != m->dummytet) continue; /*  Not a missing subface. */

        /*  Insert the subface. */
        searchtet.tet = PETSC_NULL;
        ierr = TetGenMeshScoutSubface(m, &ssub, &searchtet, 1, &dir);CHKERRQ(ierr);
        if (dir == SHAREFACE) continue; /*  The subface is inserted. */
        if (dir == COLLISIONFACE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack should not be empty");

        /*  Not exist. Push the subface back into stack. */
        ierr = TetGenMeshRandomChoice(m, facfaces->objects + 1, &s);CHKERRQ(ierr);
        ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
        *pssub = * (face *) fastlookup(facfaces, s);
        * (face *) fastlookup(facfaces, s) = ssub;

        if (dir == EDGETRIINT) continue; /*  All three edges are missing. */

        /*  Search for a crossing tet. */
        ierr = TetGenMeshScoutCrossTet(m, &ssub, &searchtet, facpoints, &dir);CHKERRQ(ierr);

        if (dir == INTERTET) {
          /*  Recover subfaces by local retetrahedralization. */
          cavitycount++;
          bakhullsize = m->hullsize;
          m->checksubsegs = m->checksubfaces = 0;
          ierr = ArrayPoolNewIndex(crosstets, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = searchtet;
          /*  Form a cavity of crossing tets. */
          ierr = TetGenMeshFormCavity(m, &ssub, crosstets, topfaces, botfaces, toppoints, botpoints, facpoints, facfaces);CHKERRQ(ierr);
          delaunayflag = PETSC_TRUE;
          /*  Tetrahedralize the top part. Re-use 'midfaces'. */
          ierr = TetGenMeshDelaunizeCavity(m, toppoints, topfaces, topshells, topnewtets, crosstets, midfaces, &success);CHKERRQ(ierr);
          if (success) {
            /*  Tetrahedralize the bottom part. Re-use 'midfaces'. */
            ierr = TetGenMeshDelaunizeCavity(m, botpoints, botfaces, botshells, botnewtets, crosstets, midfaces, &success);CHKERRQ(ierr);
            if (success) {
              /*  Fill the cavity with new tets. */
              ierr = TetGenMeshFillCavity(m, topshells, botshells, midfaces, facpoints, &success);CHKERRQ(ierr);
              if (success) {
                /*  Delete old tets and outer new tets. */
                ierr = TetGenMeshCarveCavity(m, crosstets, topnewtets, botnewtets);CHKERRQ(ierr);
              }
            } else {
              delaunayflag = PETSC_FALSE;
            }
          } else {
            delaunayflag = PETSC_FALSE;
          }
          if (!success) {
            /*  Restore old tets and delete new tets. */
            ierr = TetGenMeshRestoreCavity(m, crosstets, topnewtets, botnewtets);CHKERRQ(ierr);
          }
          if (!delaunayflag) {
            PetscInfo(b->in, "TetGen had some debugging code here");
          }
          m->hullsize = bakhullsize;
          m->checksubsegs = m->checksubfaces = 1;
        } else if (dir == INTERFACE) {
          /*  Recover subfaces by flipping edges in surface mesh. */
          ierr = TetGenMeshRecoverSubfaceByFlips(m, &ssub, &searchtet, facfaces);CHKERRQ(ierr);
          success = PETSC_TRUE;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong"); /*  dir == TOUCHFACE */
        if (!success) break;
      } /*  while */

      if (facfaces->objects > 0l) {
        /*  Found a non-Delaunay edge, split it (or a segment close to it). */
        /*  Create a new point at the middle of this edge, its coordinates */
        /*    were saved in dummypoint in 'fillcavity()'. */
        ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
        for(i = 0; i < 3; i++) newpt[i] = m->dummypoint[i];
        setpointtype(m, newpt, FREESUBVERTEX);
        setpoint2sh(m, newpt, sencode(&ssub));
        m->dummypoint[0] = m->dummypoint[1] = m->dummypoint[2] = 0;
        /*  Insert the new point. Starting search it from 'ssub'. */
        ierr = TetGenMeshSplitSubEdge_arraypool(m, newpt, &ssub, facfaces, facpoints);CHKERRQ(ierr);
        ierr = ArrayPoolRestart(facfaces);CHKERRQ(ierr);
      }
      /*  Clear the list of facet vertices. */
      ierr = ArrayPoolRestart(facpoints);CHKERRQ(ierr);

      /*  Some subsegments may be queued, recover them. */
      if (m->subsegstack->objects > 0l) {
        b->verbose--; /*  Suppress the message output. */
        ierr = TetGenMeshDelaunizeSegments2(m);CHKERRQ(ierr);
        b->verbose++;
      }
      /*  Now the mesh should be constrained Delaunay. */
    } /*  if (neightet.tet == NULL) */
  }

  PetscInfo2(b->in, "  %ld subedge flips.\n  %ld cavities remeshed.\n", m->flip22count - bakflip22count, cavitycount);

  /*  Delete arrays. */
  ierr = ArrayPoolDestroy(&crosstets);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&topnewtets);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&botnewtets);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&topfaces);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&botfaces);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&midfaces);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&toppoints);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&botpoints);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&facpoints);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&facfaces);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&topshells);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&botshells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSplitSubEdge"
/*  splitsubedge()    Split a non-Delaunay edge (not a segment) in the         */
/*                    surface mesh of a facet.                                 */
/*                                                                             */
/*  The new point 'newpt' will be inserted in the tetrahedral mesh if it does  */
/*  not cause any existing (sub)segments become non-Delaunay.  Otherwise, the  */
/*  new point is not inserted and one of such subsegments will be split.       */
/*                                                                             */
/*  Next,the actual inserted new point is also inserted into the surface mesh. */
/*  Non-Delaunay segments and newly created subfaces are queued for recovery.  */
/* tetgenmesh::splitsubedge() */
PetscErrorCode TetGenMeshSplitSubEdge_arraypool(TetGenMesh *m, point newpt, face *searchsh, ArrayPool *facfaces, ArrayPool *facpoints)
{
  triface searchtet = {PETSC_NULL, 0, 0};
  face *psseg, sseg = {PETSC_NULL, 0};
  point pa, pb;
  locateresult loc;
  int s, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Try to insert the point. Do not insert if it will encroach any segment (noencsegflag is TRUE). Queue encroacged subfaces. */
  if (m->subsegstack->objects != 0l) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack should be empty");
  searchtet = m->recenttet; /*  Start search it from recentet */
  /*  Always insert this point, missing segments are queued. 2009-06-11. */
  ierr = TetGenMeshInsertVertexBW(m, newpt, &searchtet, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &loc);CHKERRQ(ierr);

  if (loc == ENCSEGMENT) {
    /*  Some segments are encroached. Randomly pick one to split. */
    if (m->subsegstack->objects == 0l) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack should not be empty");
    ierr = TetGenMeshRandomChoice(m, m->subsegstack->objects, &s);CHKERRQ(ierr);
    psseg = (face *) fastlookup(m->subsegstack, s);
    sseg  = *psseg;
    pa    = sorg(&sseg);
    pb    = sdest(&sseg);
    for(i = 0; i < 3; i++) {newpt[i] = 0.5 * (pa[i] + pb[i]);}
    setpointtype(m, newpt, FREESEGVERTEX);
    setpoint2sh(m, newpt, sencode(&sseg));
    /*  Uninfect all queued segments. */
    for(i = 0; i < (int) m->subsegstack->objects; i++) {
      psseg = (face *) fastlookup(m->subsegstack, i);
      suninfect(m, psseg);
    }
    /*  Clear the queue. */
    ierr = ArrayPoolRestart(m->subsegstack);CHKERRQ(ierr);
    /*  Split the segment. Two subsegments are queued. */
    ierr = TetGenMeshSInsertVertex(m, newpt, searchsh, &sseg, PETSC_TRUE, PETSC_FALSE, PETSC_NULL);CHKERRQ(ierr);
    /*  Insert the point. Missing segments are queued. */
    searchtet = m->recenttet; /*  Start search it from recentet */
    ierr = TetGenMeshInsertVertexBW(m, newpt, &searchtet, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, PETSC_NULL);CHKERRQ(ierr);
  } else {
    /*  Set the abovepoint of f for point location. */
    m->abovepoint = m->facetabovepointarray[shellmark(m, searchsh)];
    if (!m->abovepoint) {
      ierr = TetGenMeshGetFacetAbovePoint(m, searchsh);CHKERRQ(ierr);
    }
    /*  Insert the new point on facet. New subfaces are queued for reocvery. */
    ierr = TetGenMeshSInsertVertex(m, newpt, searchsh, PETSC_NULL, PETSC_TRUE, PETSC_FALSE, &loc);CHKERRQ(ierr);
    if (loc == OUTSIDE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshConstrainedFacets2"
/*  constrainedfacets()    Recover subfaces saved in 'subfacestack'.           */
/* tetgenmesh::constrainedfacets2() */
PetscErrorCode TetGenMeshConstrainedFacets2(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  ArrayPool *crosstets, *topnewtets, *botnewtets;
  ArrayPool *topfaces, *botfaces, *midfaces;
  ArrayPool *topshells, *botshells, *facfaces;
  ArrayPool *toppoints, *botpoints, *facpoints;
  triface *parytet, searchtet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  face *pssub, ssub = {PETSC_NULL, 0}, neighsh = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  point *ppt, pt, newpt;
  interresult dir;
  PetscBool success, delaunayflag;
  long bakflip22count;
  long cavitycount;
  int facetcount;
  int bakhullsize;
  int s, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constraining facets.\n");

  /*  Initialize arrays. */
  ierr = ArrayPoolCreate(sizeof(triface), 10, &crosstets);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &topnewtets);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &botnewtets);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &topfaces);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &botfaces);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 10, &midfaces);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(point), 8, &toppoints);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(point), 8, &botpoints);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(point), 8, &facpoints);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &facfaces);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &topshells);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &botshells);CHKERRQ(ierr);

  bakflip22count = m->flip22count;
  cavitycount = 0;
  facetcount = 0;

  /*  Loop until 'subfacstack' is empty. */
  while(m->subfacstack->objects > 0l) {
    m->subfacstack->objects--;
    pssub = (face *) fastlookup(m->subfacstack, m->subfacstack->objects);
    ssub = *pssub;

    if (!ssub.sh[3]) continue; /*  Skip a dead subface. */

    stpivot(m, &ssub, &neightet);
    if (neightet.tet == m->dummytet) {
      sesymself(&ssub);
      stpivot(m, &ssub, &neightet);
    }

    if (neightet.tet == m->dummytet) {
      /*  Find an unrecovered subface. */
      smarktest(&ssub);
      ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
      *pssub = ssub;
      /*  Get all subfaces and vertices of the same facet. */
      for(i = 0; i < (int) facfaces->objects; i++) {
        ssub = * (face *) fastlookup(facfaces, i);
        for(j = 0; j < 3; j++) {
          sspivot(m, &ssub, &checkseg);
          if (checkseg.sh == m->dummysh) {
            spivot(&ssub, &neighsh);
            if (neighsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            if (!smarktested(&neighsh)) {
              /*  It may be already recovered. */
              stpivot(m, &neighsh, &neightet);
              if (neightet.tet == m->dummytet) {
                sesymself(&neighsh);
                stpivot(m, &neighsh, &neightet);
              }
              if (neightet.tet == m->dummytet) {
                /*  Add it into list. */
                smarktest(&neighsh);
                ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
                *pssub = neighsh;
              }
            }
          }
          pt = sorg(&ssub);
          if (!pinfected(m, pt)) {
            pinfect(m, pt);
            ierr = ArrayPoolNewIndex(facpoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr);
            *ppt = pt;
          }
          senextself(&ssub);
        } /*  j */
      } /*  i */
      /*  Have found all facet subfaces (vertices). Uninfect them. */
      for(i = 0; i < (int) facfaces->objects; i++) {
        pssub = (face *) fastlookup(facfaces, i);
        sunmarktest(pssub);
      }
      for(i = 0; i < (int) facpoints->objects; i++) {
        ppt = (point *) fastlookup(facpoints, i);
        puninfect(m, *ppt);
      }
      PetscInfo3(b->in, "  Recover facet #%d: %ld subfaces, %ld vertices.\n", facetcount + 1, facfaces->objects, facpoints->objects);
      facetcount++;

      /*  Loop until 'facfaces' is empty. */
      while(facfaces->objects > 0l) {
        /*  Get the last subface of this array. */
        facfaces->objects--;
        pssub = (face *) fastlookup(facfaces, facfaces->objects);
        ssub = *pssub;

        stpivot(m, &ssub, &neightet);
        if (neightet.tet == m->dummytet) {
          sesymself(&ssub);
          stpivot(m, &ssub, &neightet);
        }

        if (neightet.tet != m->dummytet) continue; /*  Not a missing subface. */

        /*  Insert the subface. */
        searchtet.tet = PETSC_NULL;
        ierr = TetGenMeshScoutSubface(m, &ssub, &searchtet, 1, &dir);CHKERRQ(ierr);
        if (dir == SHAREFACE) continue; /*  The subface is inserted. */
        if (dir == COLLISIONFACE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

        /*  Not exist. Push the subface back into stack. */
        ierr = TetGenMeshRandomChoice(m, facfaces->objects + 1, &s);CHKERRQ(ierr);
        ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
        *pssub = * (face *) fastlookup(facfaces, s);
        * (face *) fastlookup(facfaces, s) = ssub;

        if (dir == EDGETRIINT) continue; /*  All three edges are missing. */

        /*  Search for a crossing tet. */
        ierr = TetGenMeshScoutCrossTet(m, &ssub, &searchtet, facpoints, &dir);CHKERRQ(ierr);

        if (dir == INTERTET) {
          /*  Recover subfaces by local retetrahedralization. */
          cavitycount++;
          bakhullsize = m->hullsize;
          m->checksubsegs = m->checksubfaces = 0;
          ierr = ArrayPoolNewIndex(crosstets, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = searchtet;
          /*  Form a cavity of crossing tets. */
          ierr = TetGenMeshFormCavity(m, &ssub, crosstets, topfaces, botfaces, toppoints, botpoints, facpoints, facfaces);CHKERRQ(ierr);
          delaunayflag = PETSC_TRUE;
          /*  Tetrahedralize the top part. Re-use 'midfaces'. */
          ierr = TetGenMeshDelaunizeCavity(m, toppoints, topfaces, topshells, topnewtets, crosstets, midfaces, &success);CHKERRQ(ierr);
          if (success) {
            /*  Tetrahedralize the bottom part. Re-use 'midfaces'. */
            ierr = TetGenMeshDelaunizeCavity(m, botpoints, botfaces, botshells, botnewtets, crosstets, midfaces, &success);CHKERRQ(ierr);
            if (success) {
              /*  Fill the cavity with new tets. */
              ierr = TetGenMeshFillCavity(m, topshells, botshells, midfaces, facpoints, &success);CHKERRQ(ierr);
              if (success) {
                /*  Delete old tets and outer new tets. */
                ierr = TetGenMeshCarveCavity(m, crosstets, topnewtets, botnewtets);CHKERRQ(ierr);
              }
            } else {
              delaunayflag = PETSC_FALSE;
            }
          } else {
            delaunayflag = PETSC_FALSE;
          }
          if (!success) {
            /*  Restore old tets and delete new tets. */
            ierr = TetGenMeshRestoreCavity(m, crosstets, topnewtets, botnewtets);CHKERRQ(ierr);
          }
          if (!delaunayflag) {
            PetscInfo(b->in, "TetGen had some debugging code here");
          }
          m->hullsize = bakhullsize;
          m->checksubsegs = m->checksubfaces = 1;
        } else if (dir == INTERFACE) {
          /*  Recover subfaces by flipping edges in surface mesh. */
          ierr = TetGenMeshRecoverSubfaceByFlips(m, &ssub, &searchtet, facfaces);CHKERRQ(ierr);
          success = PETSC_TRUE;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong"); /*  dir == TOUCHFACE */
        if (!success) break;
      } /*  while */

      if (facfaces->objects > 0l) {
        /*  Found a non-Delaunay edge, split it (or a segment close to it). */
        /*  Create a new point at the middle of this edge, its coordinates */
        /*    were saved in dummypoint in 'fillcavity()'. */
        ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
        for (i = 0; i < 3; i++) newpt[i] = m->dummypoint[i];
        setpointtype(m, newpt, FREESUBVERTEX);
        setpoint2sh(m, newpt, sencode(&ssub));
        m->dummypoint[0] = m->dummypoint[1] = m->dummypoint[2] = 0;
        /*  Insert the new point. Starting search it from 'ssub'. */
        ierr = TetGenMeshSplitSubEdge_arraypool(m, newpt, &ssub, facfaces, facpoints);CHKERRQ(ierr);
        ierr = ArrayPoolRestart(facfaces);CHKERRQ(ierr);
      }
      /*  Clear the list of facet vertices. */
      ierr = ArrayPoolRestart(facpoints);CHKERRQ(ierr);

      /*  Some subsegments may be queued, recover them. */
      if (m->subsegstack->objects > 0l) {
        b->verbose--; /*  Suppress the message output. */
        ierr = TetGenMeshDelaunizeSegments2(m);CHKERRQ(ierr);
        b->verbose++;
      }
      /*  Now the mesh should be constrained Delaunay. */
    } /*  if (neightet.tet == NULL) */
  }

  PetscInfo2(b->in, "  %ld subedge flips  %ld cavities remeshed.\n", m->flip22count - bakflip22count, cavitycount);

  /*  Delete arrays. */
  ierr = ArrayPoolDestroy(&crosstets);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&topnewtets);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&botnewtets);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&topfaces);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&botfaces);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&midfaces);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&toppoints);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&botpoints);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&facpoints);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&facfaces);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&topshells);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&botshells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormSkeleton"
/*  formskeleton()    Form a constrained tetrahedralization.                   */
/*                                                                             */
/*  The segments and facets of a PLS will be recovered.                        */
/* tetgenmesh::formskeleton() */
PetscErrorCode TetGenMeshFormSkeleton(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  triface searchtet = {PETSC_NULL, 0, 0};
  face *pssub, ssub = {PETSC_NULL, 0};
  int s, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Recovering boundaries.\n");
  ierr = ArrayPoolCreate(sizeof(face), 10, &m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &m->caveshbdlist);CHKERRQ(ierr);
  /*  Put all segments into the list. */
  if (b->nojettison == 1) {  /*  '-J' option (for debug) */
    /*  The sequential order. */
    ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
    for(i = 0; i < m->subsegs->items; i++) {
      ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &ssub.sh);CHKERRQ(ierr);
      sinfect(m, &ssub);  /*  Only save it once. */
      ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
      *pssub = ssub;
    }
  } else {
    /*  Randomly order the segments. */
    ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
    for(i = 0; i < m->subsegs->items; i++) {
      ierr = TetGenMeshRandomChoice(m, i + 1, &s);CHKERRQ(ierr);
      /*  Move the s-th seg to the i-th. */
      ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
      *pssub = * (face *) fastlookup(m->subsegstack, s);
      /*  Put i-th seg to be the s-th. */
      ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &ssub.sh);CHKERRQ(ierr);
      sinfect(m, &ssub);  /*  Only save it once. */
      pssub = (face *) fastlookup(m->subsegstack, s);
      *pssub = ssub;
    }
  }
  /*  Segments will be introduced. */
  m->checksubsegs = 1;
  /*  Recover segments. */
  ierr = TetGenMeshDelaunizeSegments2(m);CHKERRQ(ierr);
  /*  Randomly order the subfaces. */
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  for(i = 0; i < m->subfaces->items; i++) {
    ierr = TetGenMeshRandomChoice(m, i + 1, &s);CHKERRQ(ierr);
    /*  Move the s-th subface to the i-th. */
    ierr = ArrayPoolNewIndex(m->subfacstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
    *pssub = * (face *) fastlookup(m->subfacstack, s);
    /*  Put i-th subface to be the s-th. */
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &ssub.sh);CHKERRQ(ierr);
    pssub = (face *) fastlookup(m->subfacstack, s);
    *pssub = ssub;
  }

  /*  Subfaces will be introduced. */
  m->checksubfaces = 1;
  /*  Recover facets. */
  ierr = TetGenMeshConstrainedFacets2(m);CHKERRQ(ierr);

  ierr = ArrayPoolDestroy(&m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&m->caveshbdlist);CHKERRQ(ierr);

  /*  Detach all segments from tets. */
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &searchtet.tet);CHKERRQ(ierr);
  while(searchtet.tet) {
    if (searchtet.tet[8]) {
      for(i = 0; i < 6; i++) {
        searchtet.loc = edge2locver[i][0];
        searchtet.ver = edge2locver[i][1];
        tssdissolve1(m, &searchtet);
      }
      searchtet.tet[8] = PETSC_NULL;
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &searchtet.tet);CHKERRQ(ierr);
  }
  /*  Now no segment is bonded to tets. */
  m->checksubsegs = 0;
  /*  Delete the memory. */
  ierr = MemoryPoolRestart(m->tet2segpool);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInfectHull"
/*  infecthull()    Virally infect all of the tetrahedra of the convex hull    */
/*                  that are not protected by subfaces.  Where there are       */
/*                  subfaces, set boundary markers as appropriate.             */
/*                                                                             */
/*  Memorypool 'viri' is used to return all the infected tetrahedra.           */
/* tetgenmesh::infecthull() */
PetscErrorCode TetGenMeshInfectHull(TetGenMesh *m, MemoryPool *viri)
{
  TetGenOpts    *b  = m->b;
  triface tetloop = {PETSC_NULL, 0, 0}, tsymtet = {PETSC_NULL, 0, 0};
  tetrahedron **deadtet;
  face hullface = {PETSC_NULL, 0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Marking concavities for elimination.\n");
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    /*  Is this tetrahedron on the hull? */
    for(tetloop.loc = 0; tetloop.loc < 4; tetloop.loc++) {
      sym(&tetloop, &tsymtet);
      if (tsymtet.tet == m->dummytet) {
        /*  Is the tetrahedron protected by a subface? */
        tspivot(m, &tetloop, &hullface);
        if (hullface.sh == m->dummysh) {
          /*  The tetrahedron is not protected; infect it. */
          if (!infected(m, &tetloop)) {
            infect(m, &tetloop);
            ierr = MemoryPoolAlloc(viri, (void **) &deadtet);CHKERRQ(ierr);
            *deadtet = tetloop.tet;
            break;  /*  Go and get next tet. */
          }
        } else {
          /*  The tetrahedron is protected; set boundary markers if appropriate. */
          if (shellmark(m, &hullface) == 0) {
            setshellmark(m, &hullface, 1);
          }
        }
      }
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPlague"
/*  plague()    Spread the virus from all infected tets to any neighbors not   */
/*              protected by subfaces.                                         */
/*                                                                             */
/*  This routine identifies all the tetrahedra that will die, and marks them   */
/*  as infected.  They are marked to ensure that each tetrahedron is added to  */
/*  the virus pool only once, so the procedure will terminate. 'viri' returns  */
/*  all infected tetrahedra which are outside the domian.                      */
/* tetgenmesh::plague() */
PetscErrorCode TetGenMeshPlague(TetGenMesh *m, MemoryPool *viri)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  tetrahedron **virusloop;
  tetrahedron **deadtet;
  triface testtet = {PETSC_NULL, 0, 0}, neighbor = {PETSC_NULL, 0, 0};
  face neighsh = {PETSC_NULL, 0}, testseg = {PETSC_NULL, 0};
  face spinsh = {PETSC_NULL, 0}, casingin = {PETSC_NULL, 0}, casingout = {PETSC_NULL, 0};
  int firstdadsub;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Marking neighbors of marked tetrahedra.\n");
  firstdadsub = 0;
  /*  Loop through all the infected tetrahedra, spreading the virus to */
  /*    their neighbors, then to their neighbors' neighbors. */
  ierr = MemoryPoolTraversalInit(viri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtet.tet = *virusloop;
    /*  Temporarily uninfect this tetrahedron, not necessary. */
    uninfect(m, &testtet);
    /*  Check each of the tetrahedron's four neighbors. */
    for(testtet.loc = 0; testtet.loc < 4; testtet.loc++) {
      /*  Find the neighbor. */
      sym(&testtet, &neighbor);
      /*  Check for a shell between the tetrahedron and its neighbor. */
      tspivot(m, &testtet, &neighsh);
      /*  Check if the neighbor is nonexistent or already infected. */
      if ((neighbor.tet == m->dummytet) || infected(m, &neighbor)) {
        if (neighsh.sh != m->dummysh) {
          /*  There is a subface separating the tetrahedron from its neighbor, */
          /*    but both tetrahedra are dying, so the subface dies too. */
          /*  Before deallocte this subface, dissolve the connections between */
          /*    other subfaces, subsegments and tetrahedra. */
          neighsh.shver = 0;
          if (!firstdadsub) {
            firstdadsub = 1; /*  Report the problem once. */
            PetscInfo3(b->in, "Warning:  Detecting an open face (%d, %d, %d).\n", pointmark(m, sorg(&neighsh)), pointmark(m, sdest(&neighsh)), pointmark(m, sapex(&neighsh)));
          }
          /*  For keep the same enext() direction. */
          ierr = TetGenMeshFindEdge_triface(m, &testtet, sorg(&neighsh), sdest(&neighsh));CHKERRQ(ierr);
          for (i = 0; i < 3; i++) {
            sspivot(m, &neighsh, &testseg);
            if (testseg.sh != m->dummysh) {
              /*  A subsegment is found at this side, dissolve this subface */
              /*    from the face link of this subsegment. */
              testseg.shver = 0;
              spinsh = neighsh;
              if (sorg(&spinsh) != sorg(&testseg)) {
                sesymself(&spinsh);
              }
              spivot(&spinsh, &casingout);
              if ((casingout.sh == spinsh.sh) || (casingout.sh == m->dummysh)) {
                /*  This is a trivial face link, only 'neighsh' itself, */
                /*    the subsegment at this side is also died. */
                ierr = TetGenMeshShellFaceDealloc(m, m->subsegs, testseg.sh);CHKERRQ(ierr);
              } else {
                spinsh = casingout;
                do {
                  casingin = spinsh;
                  spivotself(&spinsh);
                } while (spinsh.sh != neighsh.sh);
                /*  Set the link casingin->casingout. */
                sbond1(&casingin, &casingout);
                /*  Bond the subsegment anyway. */
                ssbond(m, &casingin, &testseg);
              }
            }
            senextself(&neighsh);
            enextself(&testtet);
          }
          if (neighbor.tet != m->dummytet) {
            /*  Make sure the subface doesn't get deallocated again later */
            /*    when the infected neighbor is visited. */
            tsdissolve(m, &neighbor);
          }
          /*  This subface has been separated. */
          if (in->mesh_dim > 2) {
            ierr = TetGenMeshShellFaceDealloc(m, m->subfaces, neighsh.sh);CHKERRQ(ierr);
          } else {
            /*  Dimension is 2. keep it for output. */
            /*  Dissolve tets at both sides of this subface. */
            stdissolve(m, &neighsh);
            sesymself(&neighsh);
            stdissolve(m, &neighsh);
          }
        }
      } else {                   /*  The neighbor exists and is not infected. */
        if (neighsh.sh == m->dummysh) {
          /*  There is no subface protecting the neighbor, infect it. */
          infect(m, &neighbor);
          /*  Ensure that the neighbor's neighbors will be infected. */
          ierr = MemoryPoolAlloc(viri, (void **) &deadtet);CHKERRQ(ierr);
          *deadtet = neighbor.tet;
        } else {               /*  The neighbor is protected by a subface. */
          /*  Remove this tetrahedron from the subface. */
          stdissolve(m, &neighsh);
          /*  The subface becomes a boundary.  Set markers accordingly. */
          if (shellmark(m, &neighsh) == 0) {
            setshellmark(m, &neighsh, 1);
          }
          /*  This side becomes hull. Update the handle in dummytet. */
          m->dummytet[0] = encode(&neighbor);
        }
      }
    }
    /*  Remark the tetrahedron as infected, so it doesn't get added to the */
    /*    virus pool again. */
    infect(m, &testtet);
    ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRegionPlague"
/*  regionplague()    Spread regional attributes and/or volume constraints     */
/*                    (from a .poly file) throughout the mesh.                 */
/*                                                                             */
/*  This procedure operates in two phases.  The first phase spreads an attri-  */
/*  bute and/or a volume constraint through a (facet-bounded) region.  The     */
/*  second phase uninfects all infected tetrahedra, returning them to normal.  */
/* tetgenmesh::regionplague() */
PetscErrorCode TetGenMeshRegionPlague(TetGenMesh *m, MemoryPool *regionviri, PetscReal attribute, PetscReal volume)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  tetrahedron **virusloop;
  tetrahedron **regiontet;
  triface testtet = {PETSC_NULL, 0, 0}, neighbor = {PETSC_NULL, 0, 0};
  face neighsh = {PETSC_NULL, 0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Marking neighbors of marked tetrahedra.\n");
  /*  Loop through all the infected tetrahedra, spreading the attribute */
  /*    and/or volume constraint to their neighbors, then to their neighbors' */
  /*    neighbors. */
  ierr = MemoryPoolTraversalInit(regionviri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(regionviri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtet.tet = *virusloop;
    /*  Temporarily uninfect this tetrahedron, not necessary. */
    uninfect(m, &testtet);
    if (b->regionattrib) {
      /*  Set an attribute. */
      setelemattribute(m, testtet.tet, in->numberoftetrahedronattributes, attribute);
    }
    if (b->varvolume) {
      /*  Set a volume constraint. */
      setvolumebound(m, testtet.tet, volume);
    }
    /*  Check each of the tetrahedron's four neighbors. */
    for(testtet.loc = 0; testtet.loc < 4; testtet.loc++) {
      /*  Find the neighbor. */
      sym(&testtet, &neighbor);
      /*  Check for a subface between the tetrahedron and its neighbor. */
      tspivot(m, &testtet, &neighsh);
      /*  Make sure the neighbor exists, is not already infected, and */
      /*    isn't protected by a subface, or is protected by a nonsolid */
      /*    subface. */
      if ((neighbor.tet != m->dummytet) && !infected(m, &neighbor) && (neighsh.sh == m->dummysh)) {
        /*  Infect the neighbor. */
        infect(m, &neighbor);
        /*  Ensure that the neighbor's neighbors will be infected. */
        ierr = MemoryPoolAlloc(regionviri, (void **) &regiontet);CHKERRQ(ierr);
        *regiontet = neighbor.tet;
      }
    }
    /*  Remark the tetrahedron as infected, so it doesn't get added to the */
    /*    virus pool again. */
    infect(m, &testtet);
    ierr = MemoryPoolTraverse(regionviri, (void **) &virusloop);CHKERRQ(ierr);
  }

  /*  Uninfect all tetrahedra. */
  PetscInfo(b->in, "  Unmarking marked tetrahedra.\n");
  ierr = MemoryPoolTraversalInit(regionviri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(regionviri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtet.tet = *virusloop;
    uninfect(m, &testtet);
    ierr = MemoryPoolTraverse(regionviri, (void **) &virusloop);CHKERRQ(ierr);
  }
  /*  Empty the virus pool. */
  ierr = MemoryPoolRestart(regionviri);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRemoveHoleTets"
/*  removeholetets()    Remove tetrahedra which are outside the domain.        */
/* tetgenmesh::removeholetets() */
PetscErrorCode TetGenMeshRemoveHoleTets(TetGenMesh *m, MemoryPool *viri)
{
  TetGenOpts    *b  = m->b;
  tetrahedron **virusloop;
  triface testtet = {PETSC_NULL, 0, 0}, neighbor = {PETSC_NULL, 0, 0};
  point checkpt;
  int *tetspernodelist;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Deleting marked tetrahedra.\n");
  /*  Create and initialize 'tetspernodelist'. */
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &tetspernodelist);CHKERRQ(ierr);
  for(i = 0; i < m->points->items + 1; i++) tetspernodelist[i] = 0;

  /*  Loop the tetrahedra list, counter the number of tets sharing each node. */
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &testtet.tet);CHKERRQ(ierr);
  while(testtet.tet) {
    /*  Increment the number of sharing tets for each endpoint. */
    for(i = 0; i < 4; i++) {
      j = pointmark(m, (point) testtet.tet[4 + i]);
      tetspernodelist[j]++;
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &testtet.tet);CHKERRQ(ierr);
  }

  ierr = MemoryPoolTraversalInit(viri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtet.tet = *virusloop;
    /*  Record changes in the number of boundary faces, and disconnect */
    /*    dead tetrahedra from their neighbors. */
    for(testtet.loc = 0; testtet.loc < 4; testtet.loc++) {
      sym(&testtet, &neighbor);
      if (neighbor.tet == m->dummytet) {
        /*  There is no neighboring tetrahedron on this face, so this face */
        /*    is a boundary face.  This tetrahedron is being deleted, so this */
        /*    boundary face is deleted. */
        m->hullsize--;
      } else {
        /*  Disconnect the tetrahedron from its neighbor. */
        dissolve(m, &neighbor);
        /*  There is a neighboring tetrahedron on this face, so this face */
        /*    becomes a boundary face when this tetrahedron is deleted. */
        m->hullsize++;
      }
    }
    /*  Check the four corners of this tet if they're isolated. */
    for(i = 0; i < 4; i++) {
      checkpt = (point) testtet.tet[4 + i];
      j = pointmark(m, checkpt);
      tetspernodelist[j]--;
      if (tetspernodelist[j] == 0) {
        /*  If it is added volume vertex or '-j' is not used, delete it. */
        if ((pointtype(m, checkpt) == FREEVOLVERTEX) || !b->nojettison) {
          setpointtype(m, checkpt, UNUSEDVERTEX);
          m->unuverts++;
        }
      }
    }
    /*  Return the dead tetrahedron to the pool of tetrahedra. */
    ierr = TetGenMeshTetrahedronDealloc(m, testtet.tet);CHKERRQ(ierr);
    ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  }
  ierr = PetscFree(tetspernodelist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCarveHoles"
/*  carveholes()    Find the holes and infect them.  Find the volume           */
/*                  constraints and infect them.  Infect the convex hull.      */
/*                  Spread the infection and kill tetrahedra.  Spread the      */
/*                  volume constraints.                                        */
/*                                                                             */
/*  This routine mainly calls other routines to carry out all these functions. */
/* tetgenmesh::carveholes() */
PetscErrorCode TetGenMeshCarveHoles(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  MemoryPool *holeviri, *regionviri;
  tetrahedron *tptr, **holetet, **regiontet;
  triface searchtet = {PETSC_NULL, 0, 0}, *holetets, *regiontets;
  locateresult intersect;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Removing exterior tetrahedra.\n");
  if (in->numberofholes > 0) {
    PetscInfo(b->in, "  Marking holes for elimination.\n");
  }

  /*  Initialize a pool of viri to be used for holes, concavities. */
  ierr = MemoryPoolCreate(sizeof(tetrahedron *), 1024, POINTER, 0, &holeviri);CHKERRQ(ierr);
  /*  Mark as infected any unprotected tetrahedra on the boundary. */
  ierr = TetGenMeshInfectHull(m, holeviri);CHKERRQ(ierr);

  if (in->numberofholes > 0) {
    /*  Allocate storage for the tetrahedra in which hole points fall. */
    ierr = PetscMalloc(in->numberofholes * sizeof(triface), &holetets);CHKERRQ(ierr);
    /*  Infect each tetrahedron in which a hole lies. */
    for(i = 0; i < 3 * in->numberofholes; i += 3) {
      /*  Ignore holes that aren't within the bounds of the mesh. */
      if ((in->holelist[i + 0] >= m->xmin) && (in->holelist[i + 0] <= m->xmax) &&
          (in->holelist[i + 1] >= m->ymin) && (in->holelist[i + 1] <= m->ymax) &&
          (in->holelist[i + 2] >= m->zmin) && (in->holelist[i + 2] <= m->zmax)) {
        searchtet.tet = m->dummytet;
        /*  Find a tetrahedron that contains the hole. */
        ierr = TetGenMeshLocate(m, &in->holelist[i], &searchtet, &intersect);CHKERRQ(ierr);
        if ((intersect != OUTSIDE) && (!infected(m, &searchtet))) {
          /*  Record the tetrahedron for processing carve hole. */
          holetets[i / 3] = searchtet;
        }
      }
    }
    /*  Infect the hole tetrahedron.  This is done by marking the tet as */
    /*    infected and including the tetrahedron in the virus pool. */
    for(i = 0; i < in->numberofholes; i++) {
      infect(m, &holetets[i]);
      ierr = MemoryPoolAlloc(holeviri, (void **) &holetet);CHKERRQ(ierr);
      *holetet = holetets[i].tet;
    }
    /*  Free up memory. */
    ierr = PetscFree(holetets);CHKERRQ(ierr);
  }

  /*  Mark as infected all tets of the holes and concavities. */
  ierr = TetGenMeshPlague(m, holeviri);CHKERRQ(ierr);
  /*  The virus pool contains all outside tets now. */

  /*  Is -A switch in use. */
  if (b->regionattrib) {
    /*  Assign every tetrahedron a regional attribute of zero. */
    ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
    ierr = TetGenMeshTetrahedronTraverse(m, &tptr);CHKERRQ(ierr);
    while(tptr) {
      setelemattribute(m, tptr, in->numberoftetrahedronattributes, 0.0);
      ierr = TetGenMeshTetrahedronTraverse(m, &tptr);CHKERRQ(ierr);
    }
  }

  if (in->numberofregions > 0) {
    if (b->regionattrib) {
      if (b->varvolume) {
        PetscInfo(b->in, "Spreading regional attributes and volume constraints.\n");
      } else {
        PetscInfo(b->in, "Spreading regional attributes.\n");
      }
    } else {
      PetscInfo(b->in, "Spreading regional volume constraints.\n");
    }
    /*  Allocate storage for the tetrahedra in which region points fall. */
    ierr = PetscMalloc(in->numberofregions * sizeof(triface), &regiontets);CHKERRQ(ierr);
    /*  Find the starting tetrahedron for each region. */
    for(i = 0; i < in->numberofregions; i++) {
      regiontets[i].tet = m->dummytet;
      /*  Ignore region points that aren't within the bounds of the mesh. */
      if ((in->regionlist[5 * i + 0] >= m->xmin) && (in->regionlist[5 * i + 0] <= m->xmax) &&
          (in->regionlist[5 * i + 1] >= m->ymin) && (in->regionlist[5 * i + 1] <= m->ymax) &&
          (in->regionlist[5 * i + 2] >= m->zmin) && (in->regionlist[5 * i + 2] <= m->zmax)) {
        searchtet.tet = m->dummytet;
        /*  Find a tetrahedron that contains the region point. */
        ierr = TetGenMeshLocate(m, &in->regionlist[5 * i], &searchtet, &intersect);CHKERRQ(ierr);
        if ((intersect != OUTSIDE) && (!infected(m, &searchtet))) {
          /*  Record the tetrahedron for processing after the */
          /*    holes have been carved. */
          regiontets[i] = searchtet;
        }
      }
    }
    /*  Initialize a pool to be used for regional attrs, and/or regional */
    /*    volume constraints. */
    ierr = MemoryPoolCreate(sizeof(tetrahedron *), 1024, POINTER, 0, &regionviri);CHKERRQ(ierr);
    /*  Find and set all regions. */
    for(i = 0; i < in->numberofregions; i++) {
      if (regiontets[i].tet != m->dummytet) {
        /*  Make sure the tetrahedron under consideration still exists. */
        /*    It may have been eaten by the virus. */
        if (!isdead_triface(&(regiontets[i]))) {
          /*  Put one tetrahedron in the virus pool. */
          infect(m, &regiontets[i]);
          ierr = MemoryPoolAlloc(regionviri, (void **) &regiontet);CHKERRQ(ierr);
          *regiontet = regiontets[i].tet;
          /*  Apply one region's attribute and/or volume constraint. */
          ierr = TetGenMeshRegionPlague(m, regionviri, in->regionlist[5 * i + 3], in->regionlist[5 * i + 4]);CHKERRQ(ierr);
          /*  The virus pool should be empty now. */
        }
      }
    }
    /*  Free up memory. */
    ierr = PetscFree(regiontets);CHKERRQ(ierr);
    ierr = MemoryPoolDestroy(&regionviri);CHKERRQ(ierr);
  }

  /*  Now acutually remove the outside and hole tets. */
  ierr = TetGenMeshRemoveHoleTets(m, holeviri);CHKERRQ(ierr);
  /*  The mesh is nonconvex now. */
  m->nonconvex = 1;

  /*  Update the point-to-tet map. */
  ierr = TetGenMeshMakePoint2TetMap(m);CHKERRQ(ierr);

  if (b->regionattrib) {
    if (b->regionattrib > 1) {
      /*  -AA switch. Assign each tet a region number (> 0). */
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
      assignregionattribs();
#endif
    }
    /*  Note the fact that each tetrahedron has an additional attribute. */
    in->numberoftetrahedronattributes++;
  }

  /*  Free up memory. */
  ierr = MemoryPoolDestroy(&holeviri);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  constrained_cxx ////////////////////////////////////////////////////////// */

/*  steiner_cxx ////////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCheck4FixedEdge"
/* check4fixededge()    Check if the given edge [a, b] is a fixed edge.      */
/*                                                                           */
/* A fixed edge is saved in the "fixededgelist". Return TRUE if [a, b] has   */
/* already existed in the list, otherwise, return FALSE.                     */
/* tetgenmesh::check4fixededge() */
PetscErrorCode TetGenMeshCheck4FixedEdge(TetGenMesh *m, point pa, point pb, PetscBool *isFixed)
{
  TetGenOpts *b  = m->b;
  point      *ppt;
  PetscInt    i;

  PetscFunctionBegin;
  pinfect(m, pa);
  pinfect(m, pb);
  for(i = 0; i < m->fixededgelist->objects; ++i) {
    ppt = (point *) fastlookup(m->fixededgelist, i);
    if (pinfected(m, ppt[0]) && pinfected(m, ppt[1])) {
      PetscInfo2(b->in, "    Edge (%d, %d) is fixed.\n", pointmark(m, pa), pointmark(m, pb));
      break; /* This edge already exists. */
    }
  }
  puninfect(m, pa);
  puninfect(m, pb);
  if (isFixed) {*isFixed = i < m->fixededgelist->objects ? PETSC_TRUE : PETSC_FALSE;}
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  steiner_cxx ////////////////////////////////////////////////////////////// */

/*  reconstruct_cxx ////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTransferNodes"
/*  transfernodes()    Transfer nodes from 'io->pointlist' to 'this->points'.  */
/*                                                                             */
/*  Initializing 'this->points'.  Transferring all points from 'in->pointlist' */
/*  into it. All points are indexed (start from in->firstnumber).  Each point  */
/*  is initialized be UNUSEDVERTEX.  The bounding box (xmin, xmax, ymin, ymax, */
/*  zmin, zmax) and the diameter (longest) of the point set are calculated.    */
/* tetgenmesh::transfernodes() */
PetscErrorCode TetGenMeshTransferNodes(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  point          pointloop;
  PetscReal      x, y, z;
  int            coordindex, attribindex, mtrindex;
  int            i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Read the points. */
  coordindex  = 0;
  attribindex = 0;
  mtrindex    = 0;
  for(i = 0; i < in->numberofpoints; i++) {
    ierr = TetGenMeshMakePoint(m, &pointloop);CHKERRQ(ierr);
    /*  Read the point coordinates. */
    x = pointloop[0] = in->pointlist[coordindex++];
    y = pointloop[1] = in->pointlist[coordindex++];
    z = pointloop[2] = in->pointlist[coordindex++];
    /*  Read the point attributes. */
    for(j = 0; j < in->numberofpointattributes; j++) {
      pointloop[3 + j] = in->pointattributelist[attribindex++];
    }
    /*  Read the point metric tensor. */
    for(j = 0; j < in->numberofpointmtrs; j++) {
      pointloop[m->pointmtrindex + j] = in->pointmtrlist[mtrindex++];
    }
    /*  Determine the smallest and largests x, y and z coordinates. */
    if (i == 0) {
      m->xmin = m->xmax = x;
      m->ymin = m->ymax = y;
      m->zmin = m->zmax = z;
    } else {
      m->xmin = (x < m->xmin) ? x : m->xmin;
      m->xmax = (x > m->xmax) ? x : m->xmax;
      m->ymin = (y < m->ymin) ? y : m->ymin;
      m->ymax = (y > m->ymax) ? y : m->ymax;
      m->zmin = (z < m->zmin) ? z : m->zmin;
      m->zmax = (z > m->zmax) ? z : m->zmax;
    }
  }
  /*  'longest' is the largest possible edge length formed by input vertices. */
  x = m->xmax - m->xmin;
  y = m->ymax - m->ymin;
  z = m->zmax - m->zmin;
  m->longest = sqrt(x * x + y * y + z * z);
  if (m->longest == 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The point set is trivial.\n");
  /*  Two identical points are distinguished by 'lengthlimit'. */
  m->lengthlimit = m->longest * b->epsilon * 1e+2;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshReconstructMesh"
/*  reconstructmesh()    Reconstruct a tetrahedral mesh.                       */
/*                                                                             */
/*  The list of tetrahedra will be read from 'in->tetrahedronlist'. If 'in->   */
/*  trifacelist' is not empty, boundary faces (faces with a non-zero marker)   */
/*  from this list will be inserted into the mesh. In addition, this routine   */
/*  automatically detects boundary faces (subfaces): all hull faces will be    */
/*  recognized as subfaces, internal faces between two tetrahedra which have   */
/*  different region attributes will also be recognized as subfaces.           */
/*                                                                             */
/*  Subsegments will be identified after subfaces are reconstructed. Edges at  */
/*  the intersections of non-coplanar subfaces are recognized as subsegments.  */
/*  Edges between two coplanar subfaces with different boundary markers are    */
/*  also recognized as subsegments.                                            */
/*                                                                             */
/*  The facet index of each subface will be set automatically after we have    */
/*  recovered subfaces and subsegments.  That is, the set of subfaces, which   */
/*  are coplanar and have the same boundary marker will be recognized as a     */
/*  facet and has a unique index, stored as the facet marker in each subface   */
/*  of the set, the real boundary marker of each subface will be found in      */
/*  'in->facetmarkerlist' by the index.  Facet index will be used in Delaunay  */
/*  refinement for detecting two incident facets.                              */
/*                                                                             */
/*  Points which are not corners of tetrahedra will be inserted into the mesh. */
/*  Return the number of faces on the hull after the reconstruction.           */
/* tetgenmesh::reconstructmesh() */
PetscErrorCode TetGenMeshReconstructMesh(TetGenMesh *m, long *numFaces)
{
  TetGenOpts *b  = m->b;
  PLC        *in = m->in;
  tetrahedron **tetsperverlist;
  shellface **facesperverlist;
  triface tetloop = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0}, neineightet = {PETSC_NULL, 0, 0};
  face subloop = {PETSC_NULL, 0}, neighsh = {PETSC_NULL, 0}, neineighsh = {PETSC_NULL, 0};
  face sface1 = {PETSC_NULL, 0}, sface2 = {PETSC_NULL, 0};
  face subseg = {PETSC_NULL, 0};
  point *idx2verlist;
  point torg, tdest, tapex, toppo;
  point norg, napex;
  List *neighshlist, *markerlist;
  PetscReal sign, attrib, volume;
  PetscReal da1, da2;
  PetscBool bondflag, insertsegflag, isCoplanar;
  int *idx2tetlist;
  int *idx2facelist;
  int *worklist;
  int facetidx, marker;
  int iorg, idest, iapex, ioppo;
  int pivot, ipivot, isum;
  int maxbandwidth;
  int len, index, i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Reconstructing mesh.\n");

  /*  Create a map from index to points. */
  ierr = TetGenMeshMakeIndex2PointMap(m, &idx2verlist);CHKERRQ(ierr);

  /*  Create the tetrahedra. */
  for(i = 0; i < in->numberoftetrahedra; i++) {
    /*  Create a new tetrahedron and set its four corners, make sure that */
    /*    four corners form a positive orientation. */
    ierr = TetGenMeshMakeTetrahedron(m, &tetloop);CHKERRQ(ierr);
    index = i * in->numberofcorners;
    /*  Although there may be 10 nodes, we only read the first 4. */
    iorg  = in->tetrahedronlist[index + 0] - in->firstnumber;
    idest = in->tetrahedronlist[index + 1] - in->firstnumber;
    iapex = in->tetrahedronlist[index + 2] - in->firstnumber;
    ioppo = in->tetrahedronlist[index + 3] - in->firstnumber;
    torg  = idx2verlist[iorg];
    tdest = idx2verlist[idest];
    tapex = idx2verlist[iapex];
    toppo = idx2verlist[ioppo];
    sign  = TetGenOrient3D(torg, tdest, tapex, toppo);
    if (sign > 0.0) {
      norg = torg; torg = tdest; tdest = norg;
    } else if (sign == 0.0) {
      PetscInfo1(b->in, "Warning:  Tet %d is degenerate.\n", i + in->firstnumber);
    }
    setorg(&tetloop, torg);
    setdest(&tetloop, tdest);
    setapex(&tetloop, tapex);
    setoppo(&tetloop, toppo);
    /*  Temporarily set the vertices be type FREEVOLVERTEX, to indicate that */
    /*    they belong to the mesh.  These types may be changed later. */
    setpointtype(m, torg, FREEVOLVERTEX);
    setpointtype(m, tdest, FREEVOLVERTEX);
    setpointtype(m, tapex, FREEVOLVERTEX);
    setpointtype(m, toppo, FREEVOLVERTEX);
    /*  Set element attributes if they exist. */
    for(j = 0; j < in->numberoftetrahedronattributes; j++) {
      index  = i * in->numberoftetrahedronattributes;
      attrib = in->tetrahedronattributelist[index + j];
      setelemattribute(m, tetloop.tet, j, attrib);
    }
    /*  If -a switch is used (with no number follows) Set a volume */
    /*    constraint if it exists. */
    if (b->varvolume) {
      if (in->tetrahedronvolumelist) {
        volume = in->tetrahedronvolumelist[i];
      } else {
        volume = -1.0;
      }
      setvolumebound(m, tetloop.tet, volume);
    }
  }

  /*  Set the connection between tetrahedra. */
  m->hullsize = 0l;
  /*  Create a map from nodes to tetrahedra. */
  ierr = TetGenMeshMakeTetrahedronMap(m, &idx2tetlist, &tetsperverlist);CHKERRQ(ierr);
  /*  Initialize the worklist. */
  ierr = PetscMalloc(m->points->items * sizeof(int), &worklist);CHKERRQ(ierr);
  for(i = 0; i < m->points->items; i++) worklist[i] = 0;
  maxbandwidth = 0;

  /*  Loop all tetrahedra, bond two tetrahedra if they share a common face. */
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    /*  Loop the four sides of the tetrahedron. */
    for(tetloop.loc = 0; tetloop.loc < 4; tetloop.loc++) {
      sym(&tetloop, &neightet);
      if (neightet.tet != m->dummytet) continue; /*  This side has finished. */
      torg = org(&tetloop);
      tdest = dest(&tetloop);
      tapex = apex(&tetloop);
      iorg = pointmark(m, torg) - in->firstnumber;
      idest = pointmark(m, tdest) - in->firstnumber;
      iapex = pointmark(m, tapex) - in->firstnumber;
      worklist[iorg] = 1;
      worklist[idest] = 1;
      worklist[iapex] = 1;
      /*  Pick the vertex which has the lowest degree. */
      if ((idx2tetlist[iorg + 1] - idx2tetlist[iorg]) > (idx2tetlist[idest + 1] - idx2tetlist[idest])) {
        if ((idx2tetlist[idest + 1] - idx2tetlist[idest]) > (idx2tetlist[iapex + 1] - idx2tetlist[iapex])) {
          pivot = iapex;
        } else {
          pivot = idest;
        }
      } else {
        if ((idx2tetlist[iorg + 1] - idx2tetlist[iorg]) > (idx2tetlist[iapex + 1] - idx2tetlist[iapex])) {
          pivot = iapex;
        } else {
          pivot = iorg;
        }
      }
      if ((idx2tetlist[pivot + 1] - idx2tetlist[pivot]) > maxbandwidth) {
        maxbandwidth = idx2tetlist[pivot + 1] - idx2tetlist[pivot];
      }
      bondflag = PETSC_FALSE;
      /*  Search its neighbor in the adjacent tets of the pivoted vertex. */
      for(j = idx2tetlist[pivot]; j < idx2tetlist[pivot + 1] && !bondflag; j++) {
        /*  Quickly check if this tet contains the neighbor. */
        isum = 0;
        for(k = 0; k < 4; k++) {
          norg = (point) tetsperverlist[j][4 + k];
          ipivot = pointmark(m, norg) - in->firstnumber;
          isum += worklist[ipivot];
        }
        if (isum != 3) continue;
        if (tetsperverlist[j] == tetloop.tet) continue; /*  Skip myself. */
        /*  This tet contains its neighbor, find the face and bond them. */
        neightet.tet = tetsperverlist[j];
        for(neightet.loc = 0; neightet.loc < 4; neightet.loc++) {
          norg = oppo(&neightet);
          ipivot = pointmark(m, norg) - in->firstnumber;
          if (worklist[ipivot] == 0) {
            /*  Find! Bond them together and break the loop. */
#ifdef PETSC_USE_DEBUG
            sym(&neightet, &neineightet);
            if (neineightet.tet != m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
            bond(m, &tetloop, &neightet);
            bondflag = PETSC_TRUE;
            break;
          }
        }
      }
      if (!bondflag) {
        m->hullsize++;  /*  It's a hull face. */
        /*  Bond this side to outer space. */
        m->dummytet[0] = encode(&tetloop);
        if ((in->pointmarkerlist) && !b->coarse) {
          /*  Set its three corners's markers be boundary (hull) vertices. */
          if (in->pointmarkerlist[iorg] == 0) {
            in->pointmarkerlist[iorg] = 1;
          }
          if (in->pointmarkerlist[idest] == 0) {
            in->pointmarkerlist[idest] = 1;
          }
          if (in->pointmarkerlist[iapex] == 0) {
            in->pointmarkerlist[iapex] = 1;
          }
        }
      }
      worklist[iorg] = 0;
      worklist[idest] = 0;
      worklist[iapex] = 0;
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  }

  PetscInfo1(b->in, "  Maximal vertex degree = %d.\n", maxbandwidth);

  /*  Subfaces will be inserted into the mesh. It has two phases: */
  /*    (1) Insert subfaces provided by user (in->trifacelist); */
  /*    (2) Create subfaces for hull faces (if they're not subface yet) and */
  /*        interior faces which separate two different materials. */

  /*  Phase (1). Is there a list of user-provided subfaces? */
  if (in->trifacelist) {
    /*  Recover subfaces from 'in->trifacelist'. */
    for(i = 0; i < in->numberoftrifaces; i++) {
      index = i * 3;
      iorg = in->trifacelist[index] - in->firstnumber;
      idest = in->trifacelist[index + 1] - in->firstnumber;
      iapex = in->trifacelist[index + 2] - in->firstnumber;
      /*  Look for the location of this subface. */
      worklist[iorg] = 1;
      worklist[idest] = 1;
      worklist[iapex] = 1;
      /*  Pick the vertex which has the lowest degree. */
      if ((idx2tetlist[iorg + 1] - idx2tetlist[iorg]) > (idx2tetlist[idest + 1] - idx2tetlist[idest])) {
        if ((idx2tetlist[idest + 1] - idx2tetlist[idest]) > (idx2tetlist[iapex + 1] - idx2tetlist[iapex])) {
          pivot = iapex;
        } else {
          pivot = idest;
        }
      } else {
        if ((idx2tetlist[iorg + 1] - idx2tetlist[iorg]) > (idx2tetlist[iapex + 1] - idx2tetlist[iapex])) {
          pivot = iapex;
        } else {
          pivot = iorg;
        }
      }
      bondflag = PETSC_FALSE;
      /*  Search its neighbor in the adjacent tets of torg. */
      for (j = idx2tetlist[pivot]; j < idx2tetlist[pivot + 1] && !bondflag; j++) {
        /*  Quickly check if this tet contains the neighbor. */
        isum = 0;
        for(k = 0; k < 4; k++) {
          norg = (point) tetsperverlist[j][4 + k];
          ipivot = pointmark(m, norg) - in->firstnumber;
          isum += worklist[ipivot];
        }
        if (isum != 3) continue;
        neightet.tet = tetsperverlist[j];
        for(neightet.loc = 0; neightet.loc < 4; neightet.loc++) {
          norg = oppo(&neightet);
          ipivot = pointmark(m, norg) - in->firstnumber;
          if (worklist[ipivot] == 0) {
            bondflag = PETSC_TRUE;  /*  Find! */
            break;
          }
        }
      }
      if (bondflag) {
        /*  Create a new subface and insert it into the mesh. */
        ierr = TetGenMeshMakeShellFace(m, m->subfaces, &subloop);CHKERRQ(ierr);
        torg = idx2verlist[iorg];
        tdest = idx2verlist[idest];
        tapex = idx2verlist[iapex];
        setsorg(&subloop, torg);
        setsdest(&subloop, tdest);
        setsapex(&subloop, tapex);
        /*  Set the vertices be FREESUBVERTEX to indicate they belong to a */
        /*    facet of the domain.  They may be changed later. */
        setpointtype(m, torg, FREESUBVERTEX);
        setpointtype(m, tdest, FREESUBVERTEX);
        setpointtype(m, tapex, FREESUBVERTEX);
        if (in->trifacemarkerlist) {
          setshellmark(m, &subloop, in->trifacemarkerlist[i]);
        }
        adjustedgering_triface(&neightet, CCW);
        ierr = TetGenMeshFindEdge_face(m, &subloop, org(&neightet), dest(&neightet));CHKERRQ(ierr);
        tsbond(m, &neightet, &subloop);
        sym(&neightet, &neineightet);
        if (neineightet.tet != m->dummytet) {
          sesymself(&subloop);
          tsbond(m, &neineightet, &subloop);
        }
      } else {
        PetscInfo1(b->in, "Warning:  Subface %d is discarded.\n", i + in->firstnumber);
      }
      worklist[iorg] = 0;
      worklist[idest] = 0;
      worklist[iapex] = 0;
    }
  }

  /*  Phase (2). Indentify subfaces from the mesh. */
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    /*  Loop the four sides of the tetrahedron. */
    for(tetloop.loc = 0; tetloop.loc < 4; tetloop.loc++) {
      tspivot(m, &tetloop, &subloop);
      if (subloop.sh != m->dummysh) continue;
      bondflag = PETSC_FALSE;
      sym(&tetloop, &neightet);
      if (neightet.tet == m->dummytet) {
        /*  It's a hull face. Insert a subface at here. */
        bondflag = PETSC_TRUE;
      } else {
        /*  It's an interior face. Insert a subface if two tetrahedra have */
        /*    different attributes (i.e., they belong to two regions). */
        if (in->numberoftetrahedronattributes > 0) {
          if (elemattribute(m, neightet.tet, in->numberoftetrahedronattributes - 1) != elemattribute(m, tetloop.tet, in->numberoftetrahedronattributes - 1)) {
            bondflag = PETSC_TRUE;
          }
        }
      }
      if (bondflag) {
        adjustedgering_triface(&tetloop, CCW);
        ierr = TetGenMeshMakeShellFace(m, m->subfaces, &subloop);CHKERRQ(ierr);
        torg = org(&tetloop);
        tdest = dest(&tetloop);
        tapex = apex(&tetloop);
        setsorg(&subloop, torg);
        setsdest(&subloop, tdest);
        setsapex(&subloop, tapex);
        /*  Set the vertices be FREESUBVERTEX to indicate they belong to a */
        /*    facet of the domain.  They may be changed later. */
        setpointtype(m, torg, FREESUBVERTEX);
        setpointtype(m, tdest, FREESUBVERTEX);
        setpointtype(m, tapex, FREESUBVERTEX);
        /* Mark inserted subfaces with default boundary marker */
        setshellmark(m, &subloop, 1);
        tsbond(m, &tetloop, &subloop);
        if (neightet.tet != m->dummytet) {
          sesymself(&subloop);
          tsbond(m, &neightet, &subloop);
        }
      }
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  }

  /*  Set the connection between subfaces. A subsegment may have more than */
  /*    two subfaces sharing it, 'neighshlist' stores all subfaces sharing */
  /*    one edge. */
  ierr = ListCreate(sizeof(face), PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &neighshlist);CHKERRQ(ierr);
  /*  Create a map from nodes to subfaces. */
  ierr = TetGenMeshMakeSubfaceMap(m, &idx2facelist, &facesperverlist);CHKERRQ(ierr);

  /*  Loop over the set of subfaces, setup the connection between subfaces. */
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
  while(subloop.sh) {
    for(i = 0; i < 3; i++) {
      spivot(&subloop, &neighsh);
      if (neighsh.sh == m->dummysh) {
        /*  This side is 'empty', operate on it. */
        torg = sorg(&subloop);
        tdest = sdest(&subloop);
        tapex = sapex(&subloop);
        ierr = ListAppend(neighshlist, &subloop, PETSC_NULL);CHKERRQ(ierr);
        iorg = pointmark(m, torg) - in->firstnumber;
        /*  Search its neighbor in the adjacent list of torg. */
        for(j = idx2facelist[iorg]; j < idx2facelist[iorg + 1]; j++) {
          neighsh.sh = facesperverlist[j];
          if (neighsh.sh == subloop.sh) continue;
          neighsh.shver = 0;
          if (isfacehasedge(&neighsh, torg, tdest)) {
            ierr = TetGenMeshFindEdge_face(m, &neighsh, torg, tdest);CHKERRQ(ierr);
            /*  Insert 'neighsh' into 'neighshlist'. */
            ierr = ListLength(neighshlist, &len);CHKERRQ(ierr);
            if (len < 2) {
              ierr = ListAppend(neighshlist, &neighsh, PETSC_NULL);CHKERRQ(ierr);
            } else {
              for(index = 0; index < len - 1; index++) {
                ierr = ListItem(neighshlist, index,   (void **) &sface1);CHKERRQ(ierr);
                ierr = ListItem(neighshlist, index+1, (void **) &sface2);CHKERRQ(ierr);
                ierr = TetGenMeshFaceDihedral(m, torg, tdest, sapex(&sface1), sapex(&neighsh), &da1);CHKERRQ(ierr);
                ierr = TetGenMeshFaceDihedral(m, torg, tdest, sapex(&sface1), sapex(&sface2), &da2);CHKERRQ(ierr);
                if (da1 < da2) {
                  break;  /*  Insert it after index. */
                }
              }
              ierr = ListInsert(neighshlist, index + 1, &neighsh, PETSC_NULL);CHKERRQ(ierr);
            }
          }
        }
        /*  Bond the subfaces in 'neighshlist'. */
        ierr = ListLength(neighshlist, &len);CHKERRQ(ierr);
        if (len > 1) {
          ierr = ListItem(neighshlist, 0, (void **) &neighsh);CHKERRQ(ierr);
          for(j = 1; j <= len; j++) {
            if (j < len) {
              ierr = ListItem(neighshlist, j, (void **) &neineighsh);CHKERRQ(ierr);
            } else {
              ierr = ListItem(neighshlist, 0, (void **) &neineighsh);CHKERRQ(ierr);
            }
            sbond1(&neighsh, &neineighsh);
            neighsh = neineighsh;
          }
        } else {
          /*  No neighbor subface be found, bond 'subloop' to itself. */
          sdissolve(m, &subloop); /*  sbond(subloop, subloop); */
        }
        ierr = ListClear(neighshlist);CHKERRQ(ierr);
      }
      senextself(&subloop);
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
  }

  /*  Segments will be introudced. Each segment has a unique marker (1-based). */
  marker = 1;
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
  while(subloop.sh) {
    for(i = 0; i < 3; i++) {
      sspivot(m, &subloop, &subseg);
      if (subseg.sh == m->dummysh) {
        /*  This side has no subsegment bonded, check it. */
        torg = sorg(&subloop);
        tdest = sdest(&subloop);
        tapex = sapex(&subloop);
        spivot(&subloop, &neighsh);
        spivot(&neighsh, &neineighsh);
        insertsegflag = PETSC_FALSE;
        if (subloop.sh == neighsh.sh || subloop.sh != neineighsh.sh) {
          /*  This side is either self-bonded or more than two subfaces, */
          /*    insert a subsegment at this side. */
          insertsegflag = PETSC_TRUE;
        } else {
          /*  Only two subfaces case. */
#ifdef PETSC_USE_DEBUG
          if (subloop.sh == neighsh.sh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
          napex = sapex(&neighsh);
          sign = TetGenOrient3D(torg, tdest, tapex, napex);
          ierr = TetGenMeshIsCoplanar(m, torg, tdest, tapex, napex, sign, b->epsilon, &isCoplanar);CHKERRQ(ierr);
          if (isCoplanar) {
            /*  Although they are coplanar, we still need to check if they */
            /*    have the same boundary marker. */
            insertsegflag = (shellmark(m, &subloop) != shellmark(m, &neighsh)) ? PETSC_TRUE : PETSC_FALSE;
          } else {
            /*  Non-coplanar. */
            insertsegflag = PETSC_TRUE;
          }
        }
        if (insertsegflag) {
          /*  Create a subsegment at this side. */
          ierr = TetGenMeshMakeShellFace(m, m->subsegs, &subseg);CHKERRQ(ierr);
          setsorg(&subseg, torg);
          setsdest(&subseg, tdest);
          /*  The two vertices have been marked as FREESUBVERTEX. Now mark */
          /*    them as NACUTEVERTEX. */
          setpointtype(m, torg, NACUTEVERTEX);
          setpointtype(m, tdest, NACUTEVERTEX);
          setshellmark(m, &subseg, marker);
          marker++;
          /*  Bond all subfaces to this subsegment. */
          neighsh = subloop;
          do {
            ssbond(m, &neighsh, &subseg);
            spivotself(&neighsh);
            if (neighsh.sh == m->dummysh) {
              break; /*  Only one facet case. */
            }
          } while (neighsh.sh != subloop.sh);
        }
      }
      senextself(&subloop);
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
  }

  /*  Remember the number of input segments. */
  m->insegments = m->subsegs->items;
  /*  Find the acute vertices and set them be type ACUTEVERTEX. */

  /*  Indentify facets and set the facet marker (1-based) for subfaces. */
  ierr = ListCreate(sizeof(int), PETSC_NULL, 256, PETSC_DECIDE, &markerlist);CHKERRQ(ierr);

  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
  while(subloop.sh) {
    /*  Only operate on uninfected subface, after operating, infect it. */
    if (!sinfected(m, &subloop)) {
      /*  A new facet is found. */
      marker = shellmark(m, &subloop);
      ierr = ListAppend(markerlist, &marker, PETSC_NULL);CHKERRQ(ierr);
      ierr = ListLength(markerlist, &facetidx);CHKERRQ(ierr); /*  'facetidx' starts from 1. */
      setshellmark(m, &subloop, facetidx);
      sinfect(m, &subloop);
      ierr = ListAppend(neighshlist, &subloop, PETSC_NULL);CHKERRQ(ierr);
      /*  Find out all subfaces of this facet (bounded by subsegments). */
      ierr = ListLength(neighshlist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(neighshlist, i, (void **) &neighsh);CHKERRQ(ierr);
        for(j = 0; j < 3; j++) {
          sspivot(m, &neighsh, &subseg);
          if (subseg.sh == m->dummysh) {
            spivot(&neighsh, &neineighsh);
            if (!sinfected(m, &neineighsh)) {
              /*  'neineighsh' is in the same facet as 'subloop'. */
#ifdef PETSC_USE_DEBUG
              if (shellmark(m, &neineighsh) != marker) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
              setshellmark(m, &neineighsh, facetidx);
              sinfect(m, &neineighsh);
              ierr = ListAppend(neighshlist, &neineighsh, PETSC_NULL);CHKERRQ(ierr);
            }
          }
          senextself(&neighsh);
        }
      }
      ierr = ListClear(neighshlist);CHKERRQ(ierr);
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
  }
  /*  Uninfect all subfaces. */
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
  while(subloop.sh) {
#ifdef PETSC_USE_DEBUG
    if (!sinfected(m, &subloop)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
    suninfect(m, &subloop);
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
  }
  /*  Save the facet markers in 'in->facetmarkerlist'. */
  ierr = ListLength(markerlist, &in->numberoffacets);CHKERRQ(ierr);
  ierr = PetscMalloc(in->numberoffacets * sizeof(int), &in->facetmarkerlist);CHKERRQ(ierr);
  for(i = 0; i < in->numberoffacets; i++) {
    ierr = ListItem(markerlist, i, (void **) &marker);CHKERRQ(ierr);
    in->facetmarkerlist[i] = marker;
  }
  /*  Initialize the 'facetabovepointlist'. */
  ierr = PetscMalloc((in->numberoffacets + 1) * sizeof(point), &m->facetabovepointarray);CHKERRQ(ierr);
  for(i = 0; i < in->numberoffacets + 1; i++) {
    m->facetabovepointarray[i] = PETSC_NULL;
  }

  /*  The mesh contains boundary now. */
  m->checksubfaces = 1;
  /*  The mesh is nonconvex now. */
  m->nonconvex = 1;

  ierr = ListDestroy(&markerlist);CHKERRQ(ierr);
  ierr = ListDestroy(&neighshlist);CHKERRQ(ierr);
  ierr = PetscFree(worklist);CHKERRQ(ierr);
  ierr = PetscFree(idx2tetlist);CHKERRQ(ierr);
  ierr = PetscFree(tetsperverlist);CHKERRQ(ierr);
  ierr = PetscFree(idx2facelist);CHKERRQ(ierr);
  ierr = PetscFree(facesperverlist);CHKERRQ(ierr);
  ierr = PetscFree(idx2verlist);CHKERRQ(ierr);

  if (numFaces) {*numFaces = m->hullsize;}
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  reconstruct_cxx ////////////////////////////////////////////////////////// */

/*  refine_cxx /////////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMarkSharpSegments"
/*  marksharpsegments()    Mark sharp segments.                                */
/*                                                                             */
/*  A segment s is called sharp if it is in one of the two cases:              */
/*   (1) There is a segment s' intersecting with s.  The internal angle (*)    */
/*       between s and s' is acute.                                            */
/*   (2) There are two facets f1 and f2 intersecting at s.  The internal       */
/*       dihedral angle (*) between f1 and f2 is acute.                        */
/*  This routine finds the sharp segments and marked them as type SHARP.       */
/*                                                                             */
/*  The minimum angle between segments (minfaceang) and the minimum dihedral   */
/*  angle between facets (minfacetdihed) are calulcated.                       */
/*                                                                             */
/*  (*) The internal angle (or dihedral) bewteen two features means the angle  */
/*  inside the mesh domain.                                                    */
/* tetgenmesh::marksharpsegments() */
PetscErrorCode TetGenMeshMarkSharpSegments(TetGenMesh *m, PetscReal sharpangle)
{
  TetGenOpts *b  = m->b;
  triface adjtet = {PETSC_NULL, 0, 0};
  face startsh = {PETSC_NULL, 0}, spinsh = {PETSC_NULL, 0}, neighsh = {PETSC_NULL, 0};
  face segloop = {PETSC_NULL, 0}, prevseg = {PETSC_NULL, 0}, nextseg = {PETSC_NULL, 0};
  point eorg, edest;
  PetscReal ang, smallang;
  PetscBool issharp;
  int sharpsegcount;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Marking sharp segments.\n");

  smallang = sharpangle * PETSC_PI / 180.;
  sharpsegcount = 0;
  eorg = edest = PETSC_NULL; /*  To avoid compiler warnings. */

  /*  A segment s may have been split into many subsegments. Operate the one */
  /*    which contains the origin of s. Then mark the rest of subsegments. */
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  while(segloop.sh) {
    segloop.shver = 0;
    senext2(&segloop, &prevseg);
    spivotself(&prevseg);
    if (prevseg.sh == m->dummysh) {
      /*  Operate on this seg s. */
      issharp = PETSC_FALSE;
      spivot(&segloop, &startsh);
      if (startsh.sh != m->dummysh) {
        /*  First check if two facets form an acute dihedral angle at s. */
        eorg = sorg(&segloop);
        edest = sdest(&segloop);
        spinsh = startsh;
        do {
          if (sorg(&spinsh) != eorg) {
            sesymself(&spinsh);
          }
          /*  Only do test when the spinsh is faceing inward. */
          stpivot(m, &spinsh, &adjtet);
          if (adjtet.tet != m->dummytet) {
            /*  Get the subface on the adjacent facet. */
            spivot(&spinsh, &neighsh);
            /*  Do not calculate if it is self-bonded. */
            if ((neighsh.sh != m->dummysh) && (neighsh.sh != spinsh.sh)) {
              /*  Calculate the dihedral angle between the two subfaces. */
              ierr = TetGenMeshFaceDihedral(m, eorg, edest, sapex(&spinsh), sapex(&neighsh), &ang);CHKERRQ(ierr);
              /*  Only do check if a sharp angle has not been found. */
              if (!issharp) issharp = (ang < smallang) ? PETSC_TRUE : PETSC_FALSE;
              /*  Remember the smallest facet dihedral angle. */
              m->minfacetdihed = m->minfacetdihed < ang ? m->minfacetdihed : ang;
            }
          }
          /*  Go to the next facet. */
          spivotself(&spinsh);
          if (spinsh.sh == m->dummysh) break; /*  A single subface case. */
        } while (spinsh.sh != startsh.sh);
        /*  if (!issharp) { */
          /*  Second check if s forms an acute angle with another seg. */
          spinsh = startsh;
          do {
            if (sorg(&spinsh) != eorg) {
              sesymself(&spinsh);
            }
            /*  Calculate the angle between s and s' of this facet. */
            neighsh = spinsh;
            /*  Rotate edges around 'eorg' until meeting another seg s'. Such */
            /*    seg (s') must exist since the facet is segment-bounded. */
            /*    The sum of the angles of faces at 'eorg' gives the internal */
            /*    angle between the two segments. */
            ang = 0.0;
            do {
              ang += interiorangle(eorg, sdest(&neighsh), sapex(&neighsh), PETSC_NULL);
              senext2self(&neighsh);
              sspivot(m, &neighsh, &nextseg);
              if (nextseg.sh != m->dummysh) break;
              /*  Go to the next coplanar subface. */
              spivotself(&neighsh);
              if (neighsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
              if (sorg(&neighsh) != eorg) {
                sesymself(&neighsh);
              }
            } while (1);
            /*  Only do check if a sharp angle has not been found. */
            if (!issharp) issharp = (ang < smallang) ? PETSC_TRUE : PETSC_FALSE;
            /*  Remember the smallest input face angle. */
            m->minfaceang = m->minfaceang < ang ? m->minfaceang : ang;
            /*  Go to the next facet. */
            spivotself(&spinsh);
            if (spinsh.sh == m->dummysh) break; /*  A single subface case. */
          } while (spinsh.sh != startsh.sh);
        /*  } */
      }
      if (issharp) {
        setshelltype(m, &segloop, SHARP);
        /*  Set the type for all subsegments at forwards. */
        edest = sdest(&segloop);
        senext(&segloop, &nextseg);
        spivotself(&nextseg);
        while (nextseg.sh != m->dummysh) {
          setshelltype(m, &nextseg, SHARP);
          /*  Adjust the direction of nextseg. */
          nextseg.shver = 0;
          if (sorg(&nextseg) != edest) {
            sesymself(&nextseg);
          }
          if (sorg(&nextseg) != edest) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          edest = sdest(&nextseg);
          /*  Go the next connected subsegment at edest. */
          senextself(&nextseg);
          spivotself(&nextseg);
        }
        sharpsegcount++;
      }
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  }

  /*  So far we have marked all segments which have an acute dihedral angle */
  /*    or whose ORIGINs have an acute angle. In the un-marked subsegments, */
  /*    there are possible ones whose DESTINATIONs have an acute angle. */
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  while(segloop.sh) {
    /*  Only operate if s is non-sharp and contains the dest. */
    segloop.shver = 0;
    senext(&segloop, &nextseg);
    spivotself(&nextseg);
    /*  if ((nextseg.sh == dummysh) && (shelltype(segloop) != SHARP)) { */
    if (nextseg.sh == m->dummysh) {
      /*  issharp = false; */
      issharp = (shelltype(m, &segloop) == SHARP) ? PETSC_TRUE : PETSC_FALSE;
      spivot(&segloop, &startsh);
      if (startsh.sh != m->dummysh) {
        /*  Check if s forms an acute angle with another seg. */
        eorg = sdest(&segloop);
        spinsh = startsh;
        do {
          if (sorg(&spinsh) != eorg) {
            sesymself(&spinsh);
          }
          /*  Calculate the angle between s and s' of this facet. */
          neighsh = spinsh;
          ang = 0.0;
          do {
            ang += interiorangle(eorg, sdest(&neighsh), sapex(&neighsh), PETSC_NULL);
            senext2self(&neighsh);
            sspivot(m, &neighsh, &nextseg);
            if (nextseg.sh != m->dummysh) break;
            /*  Go to the next coplanar subface. */
            spivotself(&neighsh);
            if (neighsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            if (sorg(&neighsh) != eorg) {
              sesymself(&neighsh);
            }
          } while (1);
          /*  Only do check if a sharp angle has not been found. */
          if (!issharp) issharp = (ang < smallang) ? PETSC_TRUE : PETSC_FALSE;
          /*  Remember the smallest input face angle. */
          m->minfaceang = m->minfaceang < ang ? m->minfaceang : ang;
          /*  Go to the next facet. */
          spivotself(&spinsh);
          if (spinsh.sh == m->dummysh) break; /*  A single subface case. */
        } while (spinsh.sh != startsh.sh);
      }
      if (issharp) {
        setshelltype(m, &segloop, SHARP);
        /*  Set the type for all subsegments at backwards. */
        eorg = sorg(&segloop);
        senext2(&segloop, &prevseg);
        spivotself(&prevseg);
        while (prevseg.sh != m->dummysh) {
          setshelltype(m, &prevseg, SHARP);
          /*  Adjust the direction of prevseg. */
          prevseg.shver = 0;
          if (sdest(&prevseg) != eorg) {
            sesymself(&prevseg);
          }
          if (sdest(&prevseg) != eorg) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          eorg = sorg(&prevseg);
          /*  Go to the next connected subsegment at eorg. */
          senext2self(&prevseg);
          spivotself(&prevseg);
        }
        sharpsegcount++;
      }
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  }

  if (sharpsegcount > 0) {
    PetscInfo1(b->in, "  %d sharp segments.\n", sharpsegcount);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDecideFeaturePointSizes"
/*  decidefeaturepointsizes()    Decide the sizes for all feature points.      */
/*                                                                             */
/*  A feature point is a point on a sharp segment. Every feature point p will  */
/*  be assigned a positive size which is the radius of the protecting ball.    */
/*                                                                             */
/*  The size of a feature point may be specified by one of the following ways: */
/*    (1) directly specifying on an input vertex (by using .mtr file);         */
/*    (2) imposing a fixed maximal volume constraint ('-a__' option);          */
/*    (3) imposing a maximal volume constraint in a region ('-a' option);      */
/*    (4) imposing a maximal area constraint on a facet (in .var file);        */
/*    (5) imposing a maximal length constraint on a segment (in .var file);    */
/*    (6) combining (1) - (5).                                                 */
/*    (7) automatically deriving a size if none of (1) - (6) is available.     */
/*  In case (7),the size of p is set to be the smallest edge length among all  */
/*  edges connecting at p.  The final size of p is the minimum of (1) - (7).   */
/* tetgenmesh::decidefeaturepointsizes() */
PetscErrorCode TetGenMeshDecideFeaturePointSizes(TetGenMesh *m)
{
  TetGenOpts *b  = m->b;
  PLC        *in = m->in;
  List *tetlist, *verlist;
  shellface **segsperverlist;
  triface starttet = {PETSC_NULL, 0, 0};
  face shloop = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0}, prevseg = {PETSC_NULL, 0}, nextseg = {PETSC_NULL, 0}, testseg = {PETSC_NULL, 0};
  point ploop, adjpt, e1, e2;
  PetscReal lfs_0, len, vol, maxlen = 0.0, varlen;
  PetscBool isfeature;
  int *idx2seglist;
  int featurecount;
  int llen, idx, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Deciding feature-point sizes.\n");

  /*  Constructing a map from vertices to segments. */
  ierr = TetGenMeshMakeSegmentMap(m, &idx2seglist, &segsperverlist);CHKERRQ(ierr);
  /*  Initialize working lists. */
  ierr = ListCreate(sizeof(triface), PETSC_NULL, 256, PETSC_DECIDE, &tetlist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(point *), PETSC_NULL, 256, PETSC_DECIDE, &verlist);CHKERRQ(ierr);

  if (b->fixedvolume) {
    /*  A fixed volume constraint is imposed. This gives an upper bound of */
    /*    the maximal radius of the protect ball of a vertex. */
    maxlen = pow(6.0 * b->maxvolume, 1.0/3.0);
  }

  /*  First only assign a size of p if p is not a Steiner point. The size of */
  /*    a Steiner point will be interpolated later from the endpoints of the */
  /*    segment on which it lies. */
  featurecount = 0;
  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  ierr = TetGenMeshPointTraverse(m, &ploop);CHKERRQ(ierr);
  while(ploop) {
    if (pointtype(m, ploop) != FREESEGVERTEX) {
      /*  Is p a feature point? */
      isfeature = PETSC_FALSE;
      idx = pointmark(m, ploop) - in->firstnumber;
      for (i = idx2seglist[idx]; i < idx2seglist[idx + 1] && !isfeature; i++) {
        checkseg.sh = segsperverlist[i];
        isfeature = (shelltype(m, &checkseg) == SHARP) ? PETSC_TRUE : PETSC_FALSE;
      }
      /*  Decide the size of p if it is on a sharp segment. */
      if (isfeature) {
        /*  Find a tet containing p; */
        ierr = TetGenMeshSstPivot(m, &checkseg, &starttet);CHKERRQ(ierr);
        /*  Form star(p). */
        ierr = ListAppend(tetlist, &starttet, PETSC_NULL);CHKERRQ(ierr);
        ierr = TetGenMeshFormStarPolyhedron(m, ploop, tetlist, verlist, PETSC_TRUE);CHKERRQ(ierr);
        /*  Decide the size for p if no input size is given on input. */
        if (ploop[m->pointmtrindex] == 0.0) {
          /*  Calculate lfs_0(p). */
          lfs_0 = m->longest;
          ierr = ListLength(verlist, &llen);CHKERRQ(ierr);
          for(i = 0; i < llen; i++) {
            ierr = ListItem(verlist, i, (void **) &adjpt);CHKERRQ(ierr);
            if (pointtype(m, adjpt) == FREESEGVERTEX) {
              /*  A Steiner point q. Find the seg it lies on. */
              sdecode(point2seg(m, adjpt), &checkseg);
              if (checkseg.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
              checkseg.shver = 0;
              /*  Find the origin of this seg. */
              prevseg = checkseg;
              e1 = sorg(&prevseg);
              do {
                senext2(&prevseg, &testseg);
                spivotself(&testseg);
                if (testseg.sh == m->dummysh) break;
                /*  Go to the previous subseg. */
                prevseg = testseg;
                /*  Adjust the direction of the previous subsegment. */
                prevseg.shver = 0;
                if (sdest(&prevseg) != e1) {
                  sesymself(&prevseg);
                }
                if (sdest(&prevseg) != e1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                e1 = sorg(&prevseg);
              } while (1);
              /*  Find the dest of this seg. */
              nextseg = checkseg;
              e2 = sdest(&nextseg);
              do {
                senext(&nextseg, &testseg);
                spivotself(&testseg);
                if (testseg.sh == m->dummysh) break;
                /*  Go to the next subseg. */
                nextseg = testseg;
                /*  Adjust the direction of the nextseg. */
                nextseg.shver = 0;
                if (sorg(&nextseg) != e2) {
                  sesymself(&nextseg);
                }
                if (sorg(&nextseg) != e2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
                e2 = sdest(&nextseg);
              } while (1);
              /*  e1 = sorg(prevseg); */
              /*  e2 = sdest(nextseg); */
              /*  Check if p is the origin or the dest of this seg. */
              if (ploop == e1) {
                /*  Set q to be the dest of this seg. */
                adjpt = e2;
              } else if (ploop == e2) {
                /*  Set q to be the org of this seg. */
                adjpt = e1;
              }
            }
            len = distance(ploop, adjpt);
            if (lfs_0 > len) lfs_0 = len;
          }
          ploop[m->pointmtrindex] = lfs_0;
        }
        if (b->fixedvolume) {
          /*  A fixed volume constraint is imposed. Adjust H(p) <= maxlen. */
          if (ploop[m->pointmtrindex] > maxlen) {
            ploop[m->pointmtrindex] = maxlen;
          }
        }
        if (b->varvolume) {
          /*  Variant volume constraints are imposed. Adjust H(p) <= varlen. */
          ierr = ListLength(tetlist, &llen);CHKERRQ(ierr);
          for(i = 0; i < llen; i++) {
            ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
            vol = volumebound(m, starttet.tet);
            if (vol > 0.0) {
              varlen = pow(6 * vol, 1.0/3.0);
              if (ploop[m->pointmtrindex] > varlen) {
                ploop[m->pointmtrindex] = varlen;
              }
            }
          }
        }
        /*  Clear working lists. */
        ierr = ListClear(tetlist);CHKERRQ(ierr);
        ierr = ListClear(verlist);CHKERRQ(ierr);
        featurecount++;
      } else {
        /*  NO feature point, set the size of p be zero. */
        ploop[m->pointmtrindex] = 0.0;
      }
    } /*  if (pointtype(ploop) != FREESEGVERTEX) { */
    ierr = TetGenMeshPointTraverse(m, &ploop);CHKERRQ(ierr);
  }

  PetscInfo1(b->in, "  %d feature points.\n", featurecount);

  if (!b->refine) {
    /*  Second only assign sizes for all Steiner points. A Steiner point p */
    /*    inserted on a sharp segment s is assigned a size by interpolating */
    /*    the sizes of the original endpoints of s. */
    featurecount = 0;
    ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
    ierr = TetGenMeshPointTraverse(m, &ploop);CHKERRQ(ierr);
    while(ploop) {
      if (pointtype(m, ploop) == FREESEGVERTEX) {
        if (ploop[m->pointmtrindex] == 0.0) {
          sdecode(point2seg(m, ploop), &checkseg);
          if (checkseg.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
          if (shelltype(m, &checkseg) == SHARP) {
            checkseg.shver = 0;
            /*  Find the origin of this seg. */
            prevseg = checkseg;
            e1 = sorg(&prevseg);
            do {
              senext2(&prevseg, &testseg);
              spivotself(&testseg);
              if (testseg.sh == m->dummysh) break;
              prevseg = testseg; /*  Go the previous subseg. */
              /*  Adjust the direction of this subsegmnt. */
              prevseg.shver = 0;
              if (sdest(&prevseg) != e1) {
                sesymself(&prevseg);
              }
              if (sdest(&prevseg) != e1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
              e1 = sorg(&prevseg);
            } while (1);
            /*  Find the dest of this seg. */
            nextseg = checkseg;
            e2 = sdest(&nextseg);
            do {
              senext(&nextseg, &testseg);
              spivotself(&testseg);
              if (testseg.sh == m->dummysh) break;
              nextseg = testseg; /*  Go the next subseg. */
              /*  Adjust the direction of this subsegment. */
              nextseg.shver = 0;
              if (sorg(&nextseg) != e2) {
                sesymself(&nextseg);
              }
              if (sorg(&nextseg) != e2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
              e2 = sdest(&nextseg);
            } while (1);
            /*  e1 = sorg(prevseg); */
            /*  e2 = sdest(nextseg); */
            len = distance(e1, e2);
            lfs_0 = distance(e1, ploop);
            /*  The following assert() happens when -Y option is used. */
            if (b->nobisect == 0) {
              if (lfs_0 >= len) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            }
            ploop[m->pointmtrindex] = e1[m->pointmtrindex] + (lfs_0 / len) * (e2[m->pointmtrindex] - e1[m->pointmtrindex]);
            featurecount++;
          } else {
            /*  NO feature point, set the size of p be zero. */
            ploop[m->pointmtrindex] = 0.0;
          } /*  if (shelltype(checkseg) == SHARP) */
        } /*  if (ploop[m->pointmtrindex] == 0.0) */
      } /*  if (pointtype(ploop) != FREESEGVERTEX) */
      ierr = TetGenMeshPointTraverse(m, &ploop);CHKERRQ(ierr);
    }
    if (featurecount > 0) {
      PetscInfo1(b->in, "  %d Steiner feature points.\n", featurecount);
    }
  }

  if (m->varconstraint) {
    /*  A .var file exists. Adjust feature sizes. */
    if (in->facetconstraintlist) {
      /*  Have facet area constrains. */
      ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
      ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &shloop.sh);CHKERRQ(ierr);
      while(shloop.sh) {
        varlen = areabound(m, &shloop);
        if (varlen > 0.0) {
          /*  Check if the three corners are feature points. */
          varlen = sqrt(varlen);
          for(j = 0; j < 3; j++) {
            ploop = (point) shloop.sh[3 + j];
            isfeature = PETSC_FALSE;
            idx = pointmark(m, ploop) - in->firstnumber;
            for (i = idx2seglist[idx]; i < idx2seglist[idx + 1] && !isfeature; i++) {
              checkseg.sh = segsperverlist[i];
              isfeature = (shelltype(m, &checkseg) == SHARP) ? PETSC_TRUE : PETSC_FALSE;
            }
            if (isfeature) {
              if (ploop[m->pointmtrindex] <= 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
              if (ploop[m->pointmtrindex] > varlen) {
                ploop[m->pointmtrindex] = varlen;
              }
            }
          } /*  for (j = 0; j < 3; j++) */
        }
        ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &shloop.sh);CHKERRQ(ierr);
      }
    }
    if (in->segmentconstraintlist) {
      /*  Have facet area constrains. */
      ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
      ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &shloop.sh);CHKERRQ(ierr);
      while(shloop.sh) {
        varlen = areabound(m, &shloop);
        if (varlen > 0.0) {
          /*  Check if the two endpoints are feature points. */
          for(j = 0; j < 2; j++) {
            ploop = (point) shloop.sh[3 + j];
            isfeature = PETSC_FALSE;
            idx = pointmark(m, ploop) - in->firstnumber;
            for(i = idx2seglist[idx]; i < idx2seglist[idx + 1] && !isfeature; i++) {
              checkseg.sh = segsperverlist[i];
              isfeature = (shelltype(m, &checkseg) == SHARP) ? PETSC_TRUE : PETSC_FALSE;
            }
            if (isfeature) {
              if (ploop[m->pointmtrindex] <= 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
              if (ploop[m->pointmtrindex] > varlen) {
                ploop[m->pointmtrindex] = varlen;
              }
            }
          } /*  for (j = 0; j < 2; j++) */
        }
        ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &shloop.sh);CHKERRQ(ierr);
      }
    }
  } /*  if (varconstraint) */

  ierr = PetscFree(segsperverlist);CHKERRQ(ierr);
  ierr = PetscFree(idx2seglist);CHKERRQ(ierr);
  ierr = ListDestroy(&tetlist);CHKERRQ(ierr);
  ierr = ListDestroy(&verlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshEnqueueEncSub"
/*  enqueueencsub()    Add an encroached subface into the queue.               */
/* tetgenmesh::enqueueencsub() */
PetscErrorCode TetGenMeshEnqueueEncSub(TetGenMesh *m, face *testsub, point encpt, int quenumber, PetscReal *cent)
{
  TetGenOpts *b  = m->b;
  badface *encsub;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!smarktested(testsub)) {
    if (!shell2badface(testsub)) {
      ierr = MemoryPoolAlloc(m->badsubfaces, (void **) &encsub);CHKERRQ(ierr);
      encsub->ss = *testsub;
      encsub->forg = sorg(testsub);
      encsub->fdest = sdest(testsub);
      encsub->fapex = sapex(testsub);
      encsub->foppo = (point) encpt;
      for(i = 0; i < 3; i++) encsub->cent[i] = cent[i];
      encsub->nextitem = PETSC_NULL;
      /*  Set the pointer of 'encsubseg' into 'testsub'.  It has two purposes: */
      /*    (1) We can regonize it is encroached; (2) It is uniquely queued. */
      setshell2badface(&encsub->ss, encsub);
      /*  Add the subface to the end of a queue (quenumber = 2, high priority). */
      *m->subquetail[quenumber] = encsub;
      /*  Maintain a pointer to the NULL pointer at the end of the queue. */
      m->subquetail[quenumber] = &encsub->nextitem;
      PetscInfo4(b->in, "    Queuing subface (%d, %d, %d) [%d].\n", pointmark(m, encsub->forg), pointmark(m, encsub->fdest), pointmark(m, encsub->fapex), quenumber);
    }
  } else {
    PetscInfo3(b->in, "    Ignore an encroached subface (%d, %d, %d).\n", pointmark(m, sorg(testsub)), pointmark(m, sdest(testsub)), pointmark(m, sapex(testsub)));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDequeueEncSub"
/*  dequeueencsub()    Remove an enc-subface from the front of the queue.      */
/* tetgenmesh::dequeueencsub() */
PetscErrorCode TetGenMeshDequeueEncSub(TetGenMesh *m, int *pquenumber, badface **subface)
{
  badface *result;
  int quenumber;

  PetscFunctionBegin;
  /*  Look for a nonempty queue. */
  for(quenumber = 2; quenumber >= 0; quenumber--) {
    result = m->subquefront[quenumber];
    if (result) {
      /*  Remove the badface from the queue. */
      m->subquefront[quenumber] = result->nextitem;
      /*  Maintain a pointer to the NULL pointer at the end of the queue. */
      if (!m->subquefront[quenumber]) {
        m->subquetail[quenumber] = &m->subquefront[quenumber];
      }
      *pquenumber = quenumber;
      *subface    = result;
      PetscFunctionReturn(0);
    }
  }
  *pquenumber = -1;
  *subface    = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshEnqueueBadTet"
/*  enqueuebadtet()    Add a tetrahedron into the queue.                       */
/* tetgenmesh::enqueuebadtet() */
PetscErrorCode TetGenMeshEnqueueBadTet(TetGenMesh *m, triface *testtet, PetscReal ratio2, PetscReal *cent)
{
  TetGenOpts *b  = m->b;
  badface *newbadtet;
  int queuenumber;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Allocate space for the bad tetrahedron. */
  ierr = MemoryPoolAlloc(m->badtetrahedrons, (void **) &newbadtet);CHKERRQ(ierr);
  newbadtet->tt = *testtet;
  newbadtet->key = ratio2;
  if (cent) {
    for(i = 0; i < 3; i++) newbadtet->cent[i] = cent[i];
  } else {
    for(i = 0; i < 3; i++) newbadtet->cent[i] = 0.0;
  }
  newbadtet->forg = org(testtet);
  newbadtet->fdest = dest(testtet);
  newbadtet->fapex = apex(testtet);
  newbadtet->foppo = oppo(testtet);
  newbadtet->nextitem = PETSC_NULL;
  /*  Determine the appropriate queue to put the bad tetrahedron into. */
  if (ratio2 > b->goodratio) {
    /*  queuenumber = (int) ((ratio2 - b->goodratio) / 0.5); */
    queuenumber = (int) (64.0 - 64.0 / ratio2);
    /*  'queuenumber' may overflow (negative) caused by a very large ratio. */
    if ((queuenumber > 63) || (queuenumber < 0)) {
      queuenumber = 63;
    }
  } else {
    /*  It's not a bad ratio; put the tet in the lowest-priority queue. */
    queuenumber = 0;
  }

  /*  Are we inserting into an empty queue? */
  if (!m->tetquefront[queuenumber]) {
    /*  Yes. Will this become the highest-priority queue? */
    if (queuenumber > m->firstnonemptyq) {
      /*  Yes, this is the highest-priority queue. */
      m->nextnonemptyq[queuenumber] = m->firstnonemptyq;
      m->firstnonemptyq = queuenumber;
    } else {
      /*  No. Find the queue with next higher priority. */
      i = queuenumber + 1;
      while(!m->tetquefront[i]) {
        i++;
      }
      /*  Mark the newly nonempty queue as following a higher-priority queue. */
      m->nextnonemptyq[queuenumber] = m->nextnonemptyq[i];
      m->nextnonemptyq[i] = queuenumber;
    }
    /*  Put the bad tetrahedron at the beginning of the (empty) queue. */
    m->tetquefront[queuenumber] = newbadtet;
  } else {
    /*  Add the bad tetrahedron to the end of an already nonempty queue. */
    m->tetquetail[queuenumber]->nextitem = newbadtet;
  }
  /*  Maintain a pointer to the last tetrahedron of the queue. */
  m->tetquetail[queuenumber] = newbadtet;

  PetscInfo6(b->in, "    Queueing bad tet: (%d, %d, %d, %d), ratio %g, qnum %d.\n",
             pointmark(m, newbadtet->forg), pointmark(m, newbadtet->fdest), pointmark(m, newbadtet->fapex), pointmark(m, newbadtet->foppo), sqrt(ratio2), queuenumber);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTopBadTetra"
/* tetgenmesh::topbadtetra() */
PetscErrorCode TetGenMeshTopBadTetra(TetGenMesh *m, badface **badTet)
{
  PetscFunctionBegin;
  /*  Keep a record of which queue was accessed in case dequeuebadtetra() is called later. */
  m->recentq = m->firstnonemptyq;
  /*  If no queues are nonempty, return NULL. */
  if (m->firstnonemptyq < 0) {
    *badTet = PETSC_NULL;
  } else {
    /*  Return the first tetrahedron of the highest-priority queue. */
    *badTet = m->tetquefront[m->firstnonemptyq];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDequeueBadTet"
/*  dequeuebadtet()    Remove a tetrahedron from the front of the queue.       */
/* tetgenmesh::dequeuebadtet() */
PetscErrorCode TetGenMeshDequeueBadTet(TetGenMesh *m)
{
  badface *deadbadtet;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  If queues were empty last time topbadtetra() was called, do nothing. */
  if (m->recentq >= 0) {
    /*  Find the tetrahedron last returned by topbadtetra(). */
    deadbadtet = m->tetquefront[m->recentq];
    /*  Remove the tetrahedron from the queue. */
    m->tetquefront[m->recentq] = deadbadtet->nextitem;
    /*  If this queue is now empty, update the list of nonempty queues. */
    if (deadbadtet == m->tetquetail[m->recentq]) {
      /*  Was this the highest-priority queue? */
      if (m->firstnonemptyq == m->recentq) {
        /*  Yes; find the queue with next lower priority. */
        m->firstnonemptyq = m->nextnonemptyq[m->firstnonemptyq];
      } else {
        /*  No; find the queue with next higher priority. */
        i = m->recentq + 1;
        while(!m->tetquefront[i]) {
          i++;
        }
        m->nextnonemptyq[i] = m->nextnonemptyq[m->recentq];
      }
    }
    /*  Return the bad tetrahedron to the pool. */
    ierr = TetGenMeshBadFaceDealloc(m, m->badtetrahedrons, deadbadtet);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCheckSeg4Encroach"
/*  checkseg4encroach()    Check a subsegment to see if it is encroached.      */
/*                                                                             */
/*  A segment s is encroached if there is a vertex lies inside or on its dia-  */
/*  metral circumsphere, i.e., s faces an angle theta > 90 degrees.            */
/*                                                                             */
/*  If 'testpt' (p) != NULL, only test if 'testseg' (s) is encroached by it,   */
/*  else, check all apexes of faces around s. Return TRUE if s is encroached.  */
/*  If and 'enqflag' is TRUE, add it into 'badsubsegs' if s is encroached.     */
/*                                                                             */
/*  If 'prefpt' != NULL, it returns the reference point (defined in my paper)  */
/*  if it exists.  This point is will be used to split s.                      */
/* tetgenmesh::checkseg4encroach() */
PetscErrorCode TetGenMeshCheckSeg4Encroach(TetGenMesh *m, face *testseg, point testpt, point *prefpt, PetscBool enqflag, PetscBool *isEncroached)
{
  TetGenOpts *b  = m->b;
  badface *encsubseg;
  triface starttet = {PETSC_NULL, 0, 0}, spintet = {PETSC_NULL, 0, 0};
  point eorg, edest, eapex, encpt;
  PetscReal cent[3], radius, dist, diff;
  PetscReal maxradius;
  PetscBool enq;
  int hitbdry;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  enq = PETSC_FALSE;
  eorg = sorg(testseg);
  edest = sdest(testseg);
  cent[0] = 0.5 * (eorg[0] + edest[0]);
  cent[1] = 0.5 * (eorg[1] + edest[1]);
  cent[2] = 0.5 * (eorg[2] + edest[2]);
  radius = distance(cent, eorg);

  if (m->varconstraint && (areabound(m, testseg) > 0.0)) {
    enq = ((2.0 * radius) > areabound(m, testseg)) ? PETSC_TRUE : PETSC_FALSE;
  }

  if (!enq) {
    maxradius = 0.0;
    if (!testpt) {
      /*  Check if it is encroached by traversing all faces containing it. */
      ierr = TetGenMeshSstPivot(m, testseg, &starttet);CHKERRQ(ierr);
      eapex = apex(&starttet);
      spintet = starttet;
      hitbdry = 0;
      do {
        dist = distance(cent, apex(&spintet));
        diff = dist - radius;
        if (fabs(diff) / radius <= b->epsilon) diff = 0.0; /*  Rounding. */
        if (diff <= 0.0) {
          /*  s is encroached. */
          enq = PETSC_TRUE;
          if (prefpt) {
            /*  Find the reference point. */
            encpt = apex(&spintet);
            ierr = TetGenMeshCircumsphere(m, eorg, edest, encpt, PETSC_NULL, PETSC_NULL, &dist, PETSC_NULL);CHKERRQ(ierr);
            if (dist > maxradius) {
              /*  Rememebr this point. */
              *prefpt = encpt;
              maxradius = dist;
            }
          } else {
            break;
          }
        }
        if (!fnextself(m, &spintet)) {
          hitbdry++;
          if (hitbdry < 2) {
            esym(&starttet, &spintet);
            if (!fnextself(m, &spintet)) {
              hitbdry++;
            }
          }
        }
      } while(apex(&spintet) != eapex && (hitbdry < 2));
    } else {
      /*  Only check if 'testseg' is encroached by 'testpt'. */
      dist = distance(cent, testpt);
      diff = dist - radius;
      if (fabs(diff) / radius <= b->epsilon) diff = 0.0; /*  Rounding. */
      enq = (diff <= 0.0) ? PETSC_TRUE : PETSC_FALSE;
    }
  }

  if (enq && enqflag) {
    /*  This segment is encroached and should be repaired. */
    if (!smarktested(testseg)) {
      if (!shell2badface(testseg)) { /*  Is it not queued yet? */
        PetscInfo2(b->in, "    Queuing encroaching subsegment (%d, %d).\n", pointmark(m, eorg), pointmark(m, edest));
        ierr = MemoryPoolAlloc(m->badsubsegs, (void **) &encsubseg);CHKERRQ(ierr);
        encsubseg->ss = *testseg;
        encsubseg->forg = eorg;
        encsubseg->fdest = edest;
        encsubseg->foppo = PETSC_NULL; /*  Not used. */
        /*  Set the pointer of 'encsubseg' into 'testseg'.  It has two purposes: */
        /*    (1) We can regonize it is encroached; (2) It is uniquely queued. */
        setshell2badface(&encsubseg->ss, encsubseg);
      }
    } else {
      /*  This segment has been rejected for splitting. Do not queue it. */
      PetscInfo2(b->in, "    Ignore a rejected encroaching subsegment (%d, %d).\n", pointmark(m, eorg), pointmark(m, edest));
    }
  }
  if (isEncroached) {*isEncroached = enq;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCheckSub4Encroach"
/*  checksub4encroach()    Check a subface to see if it is encroached.         */
/*                                                                             */
/*  A subface f is encroached if there is a vertex inside or on its diametral  */
/*  circumsphere.                                                              */
/*                                                                             */
/*  If 'testpt (p) != NULL', test if 'testsub' (f) is encroached by it, else,  */
/*  test if f is encroached by one of the two opposites of the adjacent tets.  */
/*  Return TRUE if f is encroached and queue it if 'enqflag' is set.           */
/* tetgenmesh::checksub4encroach() */
PetscErrorCode TetGenMeshCheckSub4Encroach(TetGenMesh *m, face *testsub, point testpt, PetscBool enqflag, PetscBool *isEncroached)
{
  TetGenOpts *b  = m->b;
  triface abuttet = {PETSC_NULL, 0, 0};
  point pa, pb, pc, encpt;
  PetscReal A[4][4], rhs[4], D;
  PetscReal cent[3], area;
  PetscReal radius, dist, diff;
  PetscBool enq;
  int indx[4];
  int quenumber;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  enq = PETSC_FALSE;
  radius = 0.0;
  encpt = PETSC_NULL;

  pa = sorg(testsub);
  pb = sdest(testsub);
  pc = sapex(testsub);

  /*  Compute the coefficient matrix A (3x3). */
  A[0][0] = pb[0] - pa[0];
  A[0][1] = pb[1] - pa[1];
  A[0][2] = pb[2] - pa[2]; /*  vector V1 (pa->pb) */
  A[1][0] = pc[0] - pa[0];
  A[1][1] = pc[1] - pa[1];
  A[1][2] = pc[2] - pa[2]; /*  vector V2 (pa->pc) */
  cross(A[0], A[1], A[2]); /*  vector V3 (V1 X V2) */

  if (m->varconstraint && (areabound(m, testsub) > 0.0)) {
    /*  Check if the subface has too big area. */
    area = 0.5 * sqrt(dot(A[2], A[2]));
    enq = (area > areabound(m, testsub)) ? PETSC_TRUE : PETSC_FALSE;
    if (enq) {
      quenumber = 2; /*  A queue of subfaces having too big area. */
    }
  }

  /*  Compute the right hand side vector b (3x1). */
  rhs[0] = 0.5 * dot(A[0], A[0]);
  rhs[1] = 0.5 * dot(A[1], A[1]);
  rhs[2] = 0.0;
  /*  Solve the 3 by 3 equations use LU decomposition with partial pivoting */
  /*    and backward and forward substitute.. */
  if (TetGenLUDecomp(A, 3, indx, &D, 0)) {
    TetGenLUSolve(A, 3, indx, rhs, 0);
    cent[0] = pa[0] + rhs[0];
    cent[1] = pa[1] + rhs[1];
    cent[2] = pa[2] + rhs[2];
    radius = sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);
  }

  if (!enq) {
    /*  Check if the subface is encroached. */
    if (!testpt) {
      stpivot(m, testsub, &abuttet);
      if (abuttet.tet != m->dummytet) {
        dist = distance(cent, oppo(&abuttet));
        diff = dist - radius;
        if (fabs(diff) / radius <= b->epsilon) diff = 0.0; /*  Rounding. */
        enq = (diff <= 0.0) ? PETSC_TRUE : PETSC_FALSE;
        if (enq) encpt = oppo(&abuttet);
      }
      if (!enq) {
        sesymself(testsub);
        stpivot(m, testsub, &abuttet);
        if (abuttet.tet != m->dummytet) {
          dist = distance(cent, oppo(&abuttet));
          diff = dist - radius;
          if (fabs(diff) / radius <= b->epsilon) diff = 0.0; /*  Rounding. */
          enq = (diff <= 0.0) ? PETSC_TRUE : PETSC_FALSE;
          if (enq) encpt = oppo(&abuttet);
        }
      }
    } else {
      dist = distance(cent, testpt);
      diff = dist - radius;
      if (fabs(diff) / radius <= b->epsilon) diff = 0.0; /*  Rounding. */
      enq = (diff <= 0.0) ? PETSC_TRUE : PETSC_FALSE;
    }
    if (enq) {
      quenumber = 0; /*  A queue of encroached subfaces. */
    }
  }

  if (enq && enqflag) {
    ierr = TetGenMeshEnqueueEncSub(m, testsub, encpt, quenumber, cent);CHKERRQ(ierr);
  }

  if (isEncroached) {*isEncroached = enq;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCheckTet4BadQual"
/*  checktet4badqual()    Test a tetrahedron for quality measures.             */
/*                                                                             */
/*  Tests a tetrahedron to see if it satisfies the minimum ratio condition     */
/*  and the maximum volume condition. Tetrahedra that aren't upto spec are     */
/*  added to the bad tetrahedron queue.                                        */
/* tetgenmesh::checktet4badqual() */
PetscErrorCode TetGenMeshCheckTet4BadQual(TetGenMesh *m, triface *testtet, PetscBool enqflag, PetscBool *isBad)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  point pa, pb, pc, pd, pe1, pe2;
  PetscReal vda[3], vdb[3], vdc[3];
  PetscReal vab[3], vbc[3], vca[3];
  PetscReal N[4][3], A[4][4], rhs[4], D;
  PetscReal elen[6], circumcent[3];
  PetscReal bicent[3], offcent[3];
  PetscReal volume, L, cosd;
  PetscReal radius2, smlen2, ratio2;
  PetscReal dist, sdist, split;
  PetscBool enq;
  int indx[4];
  int sidx, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pa = (point) testtet->tet[4];
  pb = (point) testtet->tet[5];
  pc = (point) testtet->tet[6];
  pd = (point) testtet->tet[7];

  /*  Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c. */
  /*  Set the matrix A = [vda, vdb, vdc]^T. */
  for(i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
  for(i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
  for(i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];
  /*  Get the rest edge vectors */
  for(i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
  for(i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
  for(i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

  /*  Lu-decompose the matrix A. */
  TetGenLUDecomp(A, 3, indx, &D, 0);
  /*  Get the volume of abcd. */
  volume = (A[indx[0]][0] * A[indx[1]][1] * A[indx[2]][2]) / 6.0;
  if (volume < 0.0) volume = -volume;
  /*  Check the radiu-edge ratio of the tet. */
  rhs[0] = 0.5 * dot(vda, vda);
  rhs[1] = 0.5 * dot(vdb, vdb);
  rhs[2] = 0.5 * dot(vdc, vdc);
  TetGenLUSolve(A, 3, indx, rhs, 0);
  /*  Get the circumcenter. */
  for(i = 0; i < 3; i++) circumcent[i] = pd[i] + rhs[i];
  /*  Get the square of the circumradius. */
  radius2 = dot(rhs, rhs);
  /*  Find the square of the shortest edge length. */
  elen[0] = dot(vda, vda);
  elen[1] = dot(vdb, vdb);
  elen[2] = dot(vdc, vdc);
  elen[3] = dot(vab, vab);
  elen[4] = dot(vbc, vbc);
  elen[5] = dot(vca, vca);
  smlen2 = elen[0]; sidx = 0;
  for(i = 1; i < 6; i++) {
    if (smlen2 > elen[i]) {smlen2 = elen[i]; sidx = i;}
  }
  /*  Calculate the square of radius-edge ratio. */
  ratio2 = radius2 / smlen2;
  /*  Check whether the ratio is smaller than permitted. */
  enq = (ratio2 > b->goodratio) ? PETSC_TRUE : PETSC_FALSE;
  if (!enq) {
    /*  abcd has good ratio. */
    /*  ratio2 = 0.0; */
    /*  if (b->offcenter) { */
      /*  Test if it is a sliver. */
      /*  Compute the 4 face normals (N[0], ..., N[3]). */
      for(j = 0; j < 3; j++) {
        for(i = 0; i < 3; i++) rhs[i] = 0.0;
        rhs[j] = 1.0;  /*  Positive means the inside direction */
        TetGenLUSolve(A, 3, indx, rhs, 0);
        for(i = 0; i < 3; i++) N[j][i] = rhs[i];
      }
      /*  Get the fourth normal by summing up the first three. */
      for(i = 0; i < 3; i++) N[3][i] = - N[0][i] - N[1][i] - N[2][i];
      /*  Normalized the normals. */
      for(i = 0; i < 4; i++) {
        L = sqrt(dot(N[i], N[i]));
        if (L > 0.0) {
          for (j = 0; j < 3; j++) N[i][j] /= L;
        }
      }
      /*  N[0] is the normal of face bcd. Test the dihedral angles at edge */
      /*    cd, bd, and bc to see if they are too small or too big. */
      for(i = 1; i < 4 && !enq; i++) {
        cosd = -dot(N[0], N[i]); /*  Edge cd, bd, bc. */
        enq = (cosd > m->cosmindihed) ? PETSC_TRUE : PETSC_FALSE;
      }
      if (!enq) {
        for(i = 2; i < 4 && !enq; i++) {
          cosd = -dot(N[1], N[i]); /*  Edge ad, ac */
          enq = (cosd > m->cosmindihed) ? PETSC_TRUE : PETSC_FALSE;
        }
        if (!enq) {
          cosd = -dot(N[2], N[3]); /*  Edge ab */
          enq = (cosd > m->cosmindihed) ? PETSC_TRUE : PETSC_FALSE;
        }
      }
    /*  } */
  } else if (b->offcenter) {
    /*  abcd has bad-quality. Use off-center instead of circumcenter. */
    switch (sidx) {
    case 0: /*  edge da. */
      pe1 = pd; pe2 = pa; break;
    case 1: /*  edge db. */
      pe1 = pd; pe2 = pb; break;
    case 2: /*  edge dc. */
      pe1 = pd; pe2 = pc; break;
    case 3: /*  edge ab. */
      pe1 = pa; pe2 = pb; break;
    case 4: /*  edge bc. */
      pe1 = pb; pe2 = pc; break;
    case 5: /*  edge ca. */
      pe1 = pc; pe2 = pa; break;
    default:
      pe1 = pe2 = PETSC_NULL; /*  Avoid a compile warning. */
    }
    /*  The shortest edge is e1->e2. */
    for (i = 0; i < 3; i++) bicent[i] = 0.5 * (pe1[i] + pe2[i]);
    dist = distance(bicent, circumcent);
    /*  sdist = sqrt(smlen2) * sin(PI / 3.0);  A icoso-triangle. */
    /*  The following formulae is from  */
    sdist = b->alpha3 * (b->minratio+sqrt(b->goodratio-0.25))* sqrt(smlen2);
    split = sdist / dist;
    if (split > 1.0) split = 1.0;
    /*  Get the off-center. */
    for (i = 0; i < 3; i++) {
      offcent[i] = bicent[i] + split * (circumcent[i] - bicent[i]);
    }
  }

  if (!enq && (b->varvolume || b->fixedvolume)) {
    /*  Check if the tet has too big volume. */
    enq = (b->fixedvolume && (volume > b->maxvolume)) ? PETSC_TRUE : PETSC_FALSE;
    if (!enq && b->varvolume) {
      enq = ((volume > volumebound(m, testtet->tet)) && (volumebound(m, testtet->tet) > 0.0)) ? PETSC_TRUE : PETSC_FALSE;
    }
  }

  if (!enq) {
    /*  Check if the user-defined sizing function is satisfied. */
    if (b->metric) {
      if (in->tetunsuitable) {
        /*  Execute the user-defined meshing sizing evaluation. */
        pa = (point) testtet->tet[4];
        pb = (point) testtet->tet[5];
        pc = (point) testtet->tet[6];
        pd = (point) testtet->tet[7];
        enq = (*(in->tetunsuitable))(pa, pb, pc, pd, elen, volume);
      } else {
        /*  assert(b->alpha1 > 0.0); */
        sdist = sqrt(radius2) / b->alpha1;
        for(i = 0; i < 4; i++) {
          pa = (point) testtet->tet[4 + i];
          /*  Get the indicated size of p. */
          dist = pa[m->pointmtrindex]; /*  dist = b->alpha1 * pa[m->pointmtrindex]; */
          enq = ((dist < sdist) && (dist > 0.0)) ? PETSC_TRUE : PETSC_FALSE;
          if (enq) break; /*  It is bad wrt. a node constraint. */
          /*  *** Experiment ! Stop test if c is inside H(a). */
          /*  if ((dist > 0.0) && (dist > sdist)) break; */
        }
      }
      /*  *** Experiment ! */
      /*  enq = (i == 4);  Does c lies outside all sparse-ball? */
    } /*  if (b->metric) */
  }

  if (enq && enqflag) {
    if (b->offcenter && (ratio2 > b->goodratio)) {
      for(i = 0; i < 3; i++) circumcent[i] = offcent[i];
    }
    ierr = TetGenMeshEnqueueBadTet(m, testtet, ratio2, circumcent);CHKERRQ(ierr);
  }
  if (isBad) {*isBad = enq;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshAcceptSegPt"
/*  acceptsegpt()    Check if a segment point can be inserted or not.          */
/*                                                                             */
/*  Segment(ab) is indicated to be split by a point p (\in ab). This routine   */
/*  decides whether p can be inserted or not.                                  */
/*                                                                             */
/*  p can not be inserted either the '-Y' option is used and ab is a hull      */
/*  segment or '-YY' option is used.                                           */
/*                                                                             */
/*  p can be inserted if it is in one of the following cases:                  */
/*    (1) if L = |a - b| is too long wrt the edge constraint; or               */
/*    (2) if |x - p| > \alpha_2 H(x) for x = a, b; or                          */
/*    (3) if 'refpt' != NULL.                                                  */
/* tetgenmesh::acceptsegpt() */
PetscErrorCode TetGenMeshAcceptSegPt(TetGenMesh *m, point segpt, point refpt, face *splitseg, PetscBool *isInserted)
{
  TetGenOpts *b  = m->b;
  point p[2];
  PetscReal L, lfs;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  This segment must have not been checked (and rejected) yet. */
  if (smarktested(splitseg)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

  if (b->nobisect == 1) {
    /*  '-Y'. It can not be split if it is on the hull. */
    triface spintet;
    point pc;

    ierr = TetGenMeshSstPivot(m, splitseg, &spintet);CHKERRQ(ierr);
    if (spintet.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
    pc = apex(&spintet);
    do {
      if (!fnextself(m, &spintet)) {
        /*  Meet a boundary face - s is on the hull. */
        if (isInserted) {*isInserted = PETSC_FALSE;}
        PetscFunctionReturn(0);
      }
    } while (pc != apex(&spintet));
  } else if (b->nobisect > 1) {
    /*  '-YY'. Do not split it. */
    if (isInserted) {*isInserted = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  p[0] = sorg(splitseg);
  p[1] = sdest(splitseg);
  if (m->varconstraint && (areabound(m, splitseg) > 0)) {
    lfs = areabound(m, splitseg);
    L = distance(p[0], p[1]);
    if (L > lfs) {
      if (isInserted) {*isInserted = PETSC_TRUE;} /*  case (1) */
      PetscFunctionReturn(0);
    }
  }

  j = 0; /*  Use j to count the number of inside balls. */
  for(i = 0; i < 2; i++) {
    /*  Check if p is inside the protect ball of q. */
    if (p[i][m->pointmtrindex] > 0.0) {
      lfs = b->alpha2 * p[i][m->pointmtrindex];
      L = distance(p[i], segpt);
      if (L < lfs) j++; /*  p is inside ball. */
    }
  }
  if (j == 0) {
    if (isInserted) {*isInserted = PETSC_TRUE;} /*  case (3) */
    PetscFunctionReturn(0);
  }

  /*  If 'refpt' != NULL, force p to be inserted. */
  if (refpt) {
    m->cdtenforcesegpts++;
    if (isInserted) {*isInserted = PETSC_TRUE;}
    PetscFunctionReturn(0);
  }

  /*  Do not split it. */
  m->rejsegpts++;
  if (isInserted) {*isInserted = PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshAcceptFacPt"
/*  acceptfacpt()    Check if a facet point can be inserted or not.            */
/*                                                                             */
/*  'subceillist' is CBC(p). 'verlist' (V) is empty on input, it returns the   */
/*  set of vertices of CBC(p).                                                 */
/*                                                                             */
/*  p can not be inserted either the '-Y' option is used and the facet is on   */
/*  the hull or '-YY' option is used.                                          */
/*                                                                             */
/*  p can be inserted if |p - v| > \alpha_2 H(v), for all v \in V.             */
/* tetgenmesh::acceptfacpt() */
PetscErrorCode TetGenMeshAcceptFacPt(TetGenMesh *m, point facpt, List *subceillist, List *verlist, PetscBool *isInserted)
{
  TetGenOpts *b  = m->b;
  face testsh;
  point p[2], ploop;
  PetscReal L, lfs;
  int len, idx, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (b->nobisect == 1) {
    /*  '-Y'. p can not be inserted if CBC(p) is on the hull. */
    triface testtet = {PETSC_NULL, 0};

    ierr = ListItem(subceillist, 0, (void **) &testsh);CHKERRQ(ierr);
    stpivot(m, &testsh, &testtet);
    if (testtet.tet != m->dummytet) {
      sesymself(&testsh);
      stpivot(m, &testsh, &testtet);
    }
    if (testtet.tet == m->dummytet) {
      if (isInserted) {*isInserted = PETSC_FALSE;}
      PetscFunctionReturn(0);
    }
  } else if (b->nobisect > 1) {
    /*  '-YY'. Do not split s. */
    if (isInserted) {*isInserted = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  /*  Collect the vertices of CBC(p), save them in V. */
  ierr = ListLength(subceillist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(subceillist, i, (void **) &testsh);CHKERRQ(ierr);
    p[0] = sorg(&testsh);
    p[1] = sdest(&testsh);
    for(j = 0; j < 2; j++) {
      idx = pointmark(m, p[j]);
      if (idx >= 0) {
        setpointmark(m, p[j], -idx - 1);
        ierr = ListAppend(verlist, &(p[j]), PETSC_NULL);CHKERRQ(ierr);
      }
    }
  }

  j = 0; /*  Use j to count the number of inside balls. */
  ierr = ListLength(verlist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(verlist, i, (void **) &ploop);CHKERRQ(ierr);
    /*  Uninfect q. */
    idx = pointmark(m, ploop);
    setpointmark(m, ploop, -(idx + 1));
    /*  Check if p is inside the protect ball of q. */
    if (ploop[m->pointmtrindex] > 0.0) {
      lfs = b->alpha2 * ploop[m->pointmtrindex];
      L = distance(ploop, facpt);
      if (L < lfs) j++; /*  p is inside ball. */
    }
  }
  ierr = ListClear(verlist);CHKERRQ(ierr);

  if (j == 0) {
    if (isInserted) {*isInserted = PETSC_TRUE;} /*  case (3). */
    PetscFunctionReturn(0);
  }

  m->rejsubpts++;
  if (isInserted) {*isInserted = PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshAcceptVolPt"
/*  acceptvolpt()    Check if a volume point can be inserted or not.           */
/*                                                                             */
/*  'ceillist' is B(p).  'verlist' (V) is empty on input, it returns the set   */
/*  of vertices of B(p).                                                       */
/*                                                                             */
/*  p can be split if |p - v| > \alpha_2 H(v), for all v \in V.                */
/* tetgenmesh::acceptvolpt() */
PetscErrorCode TetGenMeshAcceptVolPt(TetGenMesh *m, point volpt, List *ceillist, List *verlist, PetscBool *isInserted)
{
  TetGenOpts *b  = m->b;
  triface testtet;
  point p[3], ploop;
  PetscReal L, lfs;
  int len, idx, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Collect the vertices of CBC(p), save them in V. */
  ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(ceillist, i, (void **) &testtet);CHKERRQ(ierr);
    p[0] = org(&testtet);
    p[1] = dest(&testtet);
    p[2] = apex(&testtet);
    for(j = 0; j < 3; j++) {
      idx = pointmark(m, p[j]);
      if (idx >= 0) {
        setpointmark(m, p[j], -idx - 1);
        ierr = ListAppend(verlist, &(p[j]), PETSC_NULL);CHKERRQ(ierr);
      }
    }
  }

  j = 0; /*  Use j to counte the number of inside balls. */
  ierr = ListLength(verlist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(verlist, i, (void **) &ploop);CHKERRQ(ierr);
    /*  Uninfect q. */
    idx = pointmark(m, ploop);
    setpointmark(m, ploop, -(idx + 1));
    /*  Check if p is inside the protect ball of q. */
    if (ploop[m->pointmtrindex] > 0.0) {
      lfs = b->alpha2 * ploop[m->pointmtrindex];
      L = distance(ploop, volpt);
      if (L < lfs) j++; /*  p is inside the protect ball. */
    }
  }
  ierr = ListClear(verlist);CHKERRQ(ierr);

  if (j == 0) {
    if (isInserted) {*isInserted = PETSC_TRUE;} /*  case (2). */
    PetscFunctionReturn(0);
  }

  m->rejtetpts++;
  if (isInserted) {*isInserted = PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetSplitPoint"
/*  getsplitpoint()    Get the inserting point in a segment.                   */
/* tetgenmesh::getsplitpoint() */
PetscErrorCode TetGenMeshGetSplitPoint(TetGenMesh *m, point e1, point e2, point refpt, point newpt)
{
  TetGenOpts *b  = m->b;
  point ei, ej;
  PetscReal split, L, d1, d2;
  PetscBool acutea, acuteb;
  int i;

  PetscFunctionBegin;
  if (refpt) {
    /*  Use the CDT rules to split the segment. */
    acutea = (pointtype(m, e1) == ACUTEVERTEX) ? PETSC_TRUE : PETSC_FALSE;
    acuteb = (pointtype(m, e2) == ACUTEVERTEX) ? PETSC_TRUE : PETSC_FALSE;
    if (acutea ^ acuteb) {
      /*  Only one endpoint is acute. Use rule-2 or rule-3. */
      ei = acutea ? e1 : e2;
      ej = acutea ? e2 : e1;
      L = distance(ei, ej);
      /*  Apply rule-2. */
      d1 = distance(ei, refpt);
      split = d1 / L;
      for(i = 0; i < 3; i++) newpt[i] = ei[i] + split * (ej[i] - ei[i]);
      /*  Check if rule-3 is needed. */
      d2 = distance(refpt, newpt);
      if (d2 > (L - d1)) {
        /*  Apply rule-3. */
        if ((d1 - d2) > (0.5 * d1)) {
          split = (d1 - d2) / L;
        } else {
          split = 0.5 * d1 / L;
        }
        for (i = 0; i < 3; i++) newpt[i] = ei[i] + split * (ej[i] - ei[i]);
        PetscInfo(b->in, "    Found by rule-3:");
        m->r3count++;
      } else {
        PetscInfo(b->in, "    Found by rule-2:");
        m->r2count++;
      }
      PetscInfo2(b->in, " center %d, split = %.12g.\n", pointmark(m, ei), split);
    } else {
      /*  Both endpoints are acute or not. Split it at the middle. */
      for(i = 0; i < 3; i++) newpt[i] = 0.5 * (e1[i] + e2[i]);
    }
  } else {
    /*  Split the segment at its midpoint. */
    for(i = 0; i < 3; i++) newpt[i] = 0.5 * (e1[i] + e2[i]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSetNewPointSize"
/*  setnewpointsize()    Set the size for a new point.                         */
/*                                                                             */
/*  The size of the new point p is interpolated either from a background mesh  */
/*  (b->bgmesh) or from the two input endpoints.                               */
/* tetgenmesh::setnewpointsize() */
PetscErrorCode TetGenMeshSetNewPointSize(TetGenMesh *m, point newpt, point e1, point e2)
{
  TetGenOpts *b  = m->b;

  PetscFunctionBegin;
  if (b->metric) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    /*  Interpolate the point size in a background mesh. */
    triface bgmtet = {PETSC_NULL, 0, 0};
    /*  Get a tet in background mesh for locating p. */
    decode(point2bgmtet(m, e1), &bgmtet);
    p1interpolatebgm(newpt, &bgmtet, PETSC_NULL);
#endif
  } else {
    if (e2) {
      /*  Interpolate the size between the two endpoints. */
      PetscReal split, l, d;
      l = distance(e1, e2);
      d = distance(e1, newpt);
      split = d / l;
#ifdef PETSC_USE_DEBUG
      /*  Check if e1 and e2 are endpoints of a sharp segment. */
      if (e1[m->pointmtrindex] <= 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Null point");
      if (e2[m->pointmtrindex] <= 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Null point");
#endif
      newpt[m->pointmtrindex] = (1.0 - split) * e1[m->pointmtrindex] + split * e2[m->pointmtrindex];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSplitEncSeg"
/*  splitencseg()    Split an enc-seg and recover the Delaunayness by flips.   */
/* tetgenmesh::splitencseg() */
PetscErrorCode TetGenMeshSplitEncSeg(TetGenMesh *m, point newpt, face *splitseg, List *tetlist, List *sublist, List *verlist, Queue *flipque, PetscBool chkencsub, PetscBool chkbadtet, PetscBool optflag, PetscBool *isSplit)
{
  List *mytetlist;
  Queue *myflipque;
  triface starttet = {PETSC_NULL, 0, 0};
  face startsh = {PETSC_NULL, 0}, spinsh = {PETSC_NULL, 0}, checksh = {PETSC_NULL, 0};
  int            len, i;
  PetscBool      isSp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (optflag) {
    ierr = ListCreate(sizeof(triface), PETSC_NULL, 1024, PETSC_DECIDE, &mytetlist);CHKERRQ(ierr);
    ierr = QueueCreate(sizeof(badface), PETSC_DECIDE, &myflipque);CHKERRQ(ierr);
    tetlist = mytetlist;
    flipque = myflipque;
  }

  /*  Use the base orientation (important in this routine). */
  splitseg->shver = 0;
  /*  Insert p, this should always success. */
  ierr = TetGenMeshSstPivot(m, splitseg, &starttet);CHKERRQ(ierr);
  ierr = TetGenMeshSplitTetEdge(m, newpt, &starttet, flipque, &isSp);CHKERRQ(ierr);
  if (isSp) {
    /*  Remove locally non-Delaunay faces by flipping. */
    ierr = TetGenMeshLawson3D(m, flipque, PETSC_NULL);CHKERRQ(ierr);
  } else {
    if (optflag) {
      ierr = ListDestroy(&mytetlist);CHKERRQ(ierr);
      ierr = QueueDestroy(&myflipque);CHKERRQ(ierr);
    }
    if (isSplit) {*isSplit = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  if (!optflag) {
    /*  Check the two new subsegs to see if they're encroached (not by p). */
    for(i = 0; i < 2; i++) {
      /* if (!shell2badface(*splitseg)) { */
      ierr = TetGenMeshCheckSeg4Encroach(m, splitseg, PETSC_NULL, PETSC_NULL, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      /* } */
      if (i == 1) break; /*  Two new segs have been checked. */
      senextself(splitseg);
      spivotself(splitseg);
#ifdef PETSC_USE_DEBUG
      if (!splitseg->sh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
#endif
      splitseg->shver = 0;
    }
    /*  Check the new subfaces to see if they're encroached (not by p). */
    if (chkencsub) {
      spivot(splitseg, &startsh);
      spinsh = startsh;
      do {
        ierr = ListAppend(sublist, &spinsh, PETSC_NULL);CHKERRQ(ierr);
        ierr = TetGenMeshFormStarPolygon(m, newpt, sublist, verlist);CHKERRQ(ierr);
        ierr = ListLength(sublist, &len);CHKERRQ(ierr);
        for(i = 0; i < len; i++) {
          ierr = ListItem(sublist, i, (void **) &checksh);CHKERRQ(ierr);
          /* if (!shell2badface(checksh)) { */
          ierr = TetGenMeshCheckSub4Encroach(m, &checksh, PETSC_NULL, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
          /* } */
        }
        ierr = ListClear(sublist);CHKERRQ(ierr);
        if (verlist) {ierr = ListClear(verlist);CHKERRQ(ierr);}
        spivotself(&spinsh);
        if (spinsh.sh == m->dummysh) {
          break; /*  There's only one facet having this segment. */
        }
      } while (spinsh.sh != startsh.sh);
    }
  } /*  if (!optflag) */

  /*  Collect the new tets connecting at p. */
  ierr = TetGenMeshSstPivot(m, splitseg, &starttet);CHKERRQ(ierr);
  ierr = ListAppend(tetlist, &starttet, PETSC_NULL);CHKERRQ(ierr);
  ierr = TetGenMeshFormStarPolyhedron(m, newpt, tetlist, verlist, PETSC_TRUE);CHKERRQ(ierr);

  if (!optflag) {
    /*  Check if p encroaches adjacent segments. */
    ierr = TetGenMeshTallEncSegs(m, newpt, 1, &tetlist, PETSC_NULL);CHKERRQ(ierr);
    if (chkencsub) {
      /*  Check if p encroaches adjacent subfaces. */
      ierr = TetGenMeshTallEncSubs(m, newpt, 1, &tetlist, PETSC_NULL);CHKERRQ(ierr);
    }
    if (chkbadtet) {
      /*  Check if there are new bad quality tets at p. */
      ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
        ierr = TetGenMeshCheckTet4BadQual(m, &starttet, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      }
    }
    ierr = ListClear(tetlist);CHKERRQ(ierr);
  } else {
    /*  Check if new tets are non-optimal. */
    ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
      ierr = TetGenMeshCheckTet4Opt(m, &starttet, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = ListDestroy(&mytetlist);CHKERRQ(ierr);
    ierr = QueueDestroy(&myflipque);CHKERRQ(ierr);
  }

  if (isSplit) {*isSplit = PETSC_TRUE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTallEncSegs"
/*  tallencsegs()    Check for encroached segments and save them in list.      */
/*                                                                             */
/*  If 'testpt' (p) != NULL, only check if segments are encroached by p, else, */
/*  check all the nearby mesh vertices.                                        */
/*                                                                             */
/*  If 'ceillists' (B_i(p)) != NULL, there are 'n' B_i(p)s, only check the     */
/*  segments which are on B_i(p)s, else, check the entire list of segments     */
/*  (in the pool 'this->subsegs').                                             */
/* tetgenmesh::tallencsegs() */
PetscErrorCode TetGenMeshTallEncSegs(TetGenMesh *m, point testpt, int n, List **ceillists, PetscBool *isEncroached)
{
  List *ceillist;
  triface ceiltet = {PETSC_NULL, 0, 0};
  face checkseg = {PETSC_NULL, 0};
  int enccount; /* long oldencnum; */
  int len, i, j, k;
  PetscBool      isEnc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Remember the current number of encroached segments. */
  /*  oldencnum = badsubsegs->items; */

  /*  Count the number of encroached segments. */
  enccount = 0;

  if (ceillists) {
    for(k = 0; k < n; k++) {
      ceillist = ceillists[k];
      /*  Check the segments on B_i(p). */
      ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(ceillist, i, (void **) &ceiltet);CHKERRQ(ierr);
        ceiltet.ver = 0;
        for(j = 0; j < 3; j++) {
          ierr = TetGenMeshTssPivot(m, &ceiltet, &checkseg);CHKERRQ(ierr);
          if (checkseg.sh != m->dummysh) {
            /*  Found a segment. Test it if it isn't in enc-list. */
            /*  if (!shell2badface(checkseg)) { */
              ierr = TetGenMeshCheckSeg4Encroach(m, &checkseg, testpt, PETSC_NULL, PETSC_TRUE, &isEnc);CHKERRQ(ierr);
              if (isEnc) {
                enccount++;
              }
            /*  } */
          }
          enextself(&ceiltet);
        }
      }
    }
  } else {
    /*  Check the entire list of segments. */
    ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &checkseg.sh);CHKERRQ(ierr);
    while(checkseg.sh) {
      /*  Test it if it isn't in enc-list. */
      /*  if (!shell2badface(checkseg)) { */
      ierr = TetGenMeshCheckSeg4Encroach(m, &checkseg, testpt, PETSC_NULL, PETSC_TRUE, &isEnc);CHKERRQ(ierr);
        if (isEnc) {
          enccount++;
        }
      /*  } */
      ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &checkseg.sh);CHKERRQ(ierr);
    }
  }

  /*  return (badsubsegs->items > oldencnum); */
  if (isEncroached) {*isEncroached = (enccount > 0) ? PETSC_TRUE: PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTallEncSubs"
/*  tallencsubs()    Find all encroached subfaces and save them in list.       */
/*                                                                             */
/*  If 'testpt' (p) != NULL, only check if subfaces are encroached by p, else, */
/*  check the adjacent vertices of subfaces.                                   */
/*                                                                             */
/*  If 'ceillists' (B_i(p)) != NULL, there are 'n' B_i(p)s, only check the     */
/*  subfaces which are on B_i(p)s, else, check the entire list of subfaces     */
/*  (in the pool 'this->subfaces').                                            */
/* tetgenmesh::tallencsubs() */
PetscErrorCode TetGenMeshTallEncSubs(TetGenMesh *m, point testpt, int n, List **ceillists, PetscBool *isEncroached)
{
  List *ceillist;
  triface ceiltet = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0};
  int enccount; /* long oldencnum; */
  int len, i, k;
  PetscBool      isEnc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Remember the current number of encroached segments. */
  /*  oldencnum = badsubfaces->items; */

  enccount = 0; /*  Count the number encroached subfaces. */

  if (ceillists) {
    for(k = 0; k < n; k++) {
      ceillist = ceillists[k];
      /*  Check the subfaces on B_i(p). */
      ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(ceillist, i, (void **) &ceiltet);CHKERRQ(ierr);
        tspivot(m, &ceiltet, &checksh);
        if (checksh.sh != m->dummysh) {
          /*  Found a subface. Test it if it isn't in enc-list. */
          /* if (!shell2badface(checksh)) { */
            ierr = TetGenMeshCheckSub4Encroach(m, &checksh, testpt, PETSC_TRUE, &isEnc);CHKERRQ(ierr);
            if (isEnc) {
              enccount++;
            }
          /* } */
        }
      }
    }
  } else {
    /*  Check the entire list of subfaces. */
    ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &checksh.sh);CHKERRQ(ierr);
    while(checksh.sh) {
      /*  Test it if it isn't in enc-list. */
      /*  if (!shell2badface(checksh)) { */
        ierr = TetGenMeshCheckSub4Encroach(m, &checksh, testpt, PETSC_TRUE, &isEnc);CHKERRQ(ierr);
        if (isEnc) {
          enccount++;
        }
      /*  } */
      ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &checksh.sh);CHKERRQ(ierr);
    }
  }

  /* return (badsubfaces->items > oldencnum); */
  if (isEncroached) {*isEncroached = (enccount > 0) ? PETSC_TRUE: PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTallBadTetrahedrons"
/*  tallbadtetrahedrons()    Queue all the bad-quality tetrahedra in the mesh. */
/* tetgenmesh::tallbadtetrahedrons() */
PetscErrorCode TetGenMeshTallBadTetrahedrons(TetGenMesh *m)
{
  triface tetloop = {PETSC_NULL, 0, 0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    ierr = TetGenMeshCheckTet4BadQual(m, &tetloop, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRepairEncSegs"
/*  repairencsegs()    Repair (split) all the encroached segments.             */
/*                                                                             */
/*  Each encroached segment is repaired by splitting it - inserting a vertex   */
/*  at or near its midpoint.  Newly inserted vertices may encroach upon other  */
/*  subsegments, these are also repaired.                                      */
/*                                                                             */
/*  'chkencsub' and 'chkbadtet' are two flags that specify whether one should  */
/*  take note of new encroaced subfaces and bad quality tets that result from  */
/*  inserting vertices to repair encroached subsegments.                       */
/* tetgenmesh::repairencsegs() */
PetscErrorCode TetGenMeshRepairEncSegs(TetGenMesh *m, PetscBool chkencsub, PetscBool chkbadtet)
{
  TetGenOpts    *b = m->b;
  List **tetlists, **ceillists;
  List **sublists, **subceillists;
  List *tetlist, *sublist;
  Queue *flipque;
  badface *encloop;
  face splitseg = {PETSC_NULL, 0};
  point newpt, refpt;
  point e1, e2;
  int nmax, n;
  PetscBool      isInserted, isValid;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = 0;
  nmax = 128;
  if (!b->fliprepair) {
    ierr = PetscMalloc(nmax * sizeof(List *), &tetlists);CHKERRQ(ierr);
    ierr = PetscMalloc(nmax * sizeof(List *), &ceillists);CHKERRQ(ierr);
    ierr = PetscMalloc(nmax * sizeof(List *), &sublists);CHKERRQ(ierr);
    ierr = PetscMalloc(nmax * sizeof(List *), &subceillists);CHKERRQ(ierr);
  } else {
    ierr = ListCreate(sizeof(triface), PETSC_NULL, 1024, PETSC_DECIDE, &tetlist);CHKERRQ(ierr);
    ierr = ListCreate(sizeof(face),    PETSC_NULL,  256, PETSC_DECIDE, &sublist);CHKERRQ(ierr);
    ierr = QueueCreate(sizeof(badface), PETSC_DECIDE, &flipque);CHKERRQ(ierr);
  }

  /*  Loop until the pool 'badsubsegs' is empty. Note that steinerleft == -1 */
  /*    if an unlimited number of Steiner points is allowed. */
  while((m->badsubsegs->items > 0) && (m->steinerleft != 0)) {
    ierr = MemoryPoolTraversalInit(m->badsubsegs);CHKERRQ(ierr);
    ierr = TetGenMeshBadFaceTraverse(m, m->badsubsegs, &encloop);CHKERRQ(ierr);
    while(encloop && (m->steinerleft != 0)) {
      /*  Get an encroached subsegment s. */
      splitseg = encloop->ss;
      /*  Clear the in-queue flag in s. */
      setshell2badface(&splitseg, PETSC_NULL);
      if ((sorg(&splitseg) == encloop->forg) && (sdest(&splitseg) == encloop->fdest)) {
        PetscInfo2(b->in, "  Get an enc-seg (%d, %d)\n", pointmark(m, encloop->forg), pointmark(m, encloop->fdest));
        refpt = PETSC_NULL;
        if (b->conformdel) {
          /*  Look for a reference point. */
          ierr = TetGenMeshCheckSeg4Encroach(m, &splitseg, PETSC_NULL, &refpt, PETSC_FALSE, PETSC_NULL);CHKERRQ(ierr);
        }
        /*  Create the new point p (at the middle of s). */
        ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
        ierr = TetGenMeshGetSplitPoint(m, encloop->forg, encloop->fdest, refpt, newpt);CHKERRQ(ierr);
        setpointtype(m, newpt, FREESEGVERTEX);
        setpoint2seg(m, newpt, sencode(&splitseg));
        /*  Decide whether p can be inserted or not. */
        ierr = TetGenMeshAcceptSegPt(m, newpt, refpt, &splitseg, &isInserted);CHKERRQ(ierr);
        if (isInserted) {
          /*  Save the endpoints of the seg for size interpolation. */
          e1 = sorg(&splitseg);
          if (shelltype(m, &splitseg) == SHARP) {
            e2 = sdest(&splitseg);
          } else {
            e2 = PETSC_NULL; /*  No need to do size interoplation. */
          }
          if (!b->fliprepair) {
            /*  Form BC(p), B(p), CBC(p)s, and C(p)s. */
            ierr = TetGenMeshFormBowatCavity(m, newpt, &splitseg, PETSC_NULL, &n, &nmax, sublists, subceillists, tetlists, ceillists);CHKERRQ(ierr);
            /*  Validate/update BC(p), B(p), CBC(p)s, and C(p)s. */
            ierr = TetGenMeshTrimBowatCavity(m, newpt, &splitseg, n, sublists, subceillists, tetlists, ceillists, -1.0, &isValid);CHKERRQ(ierr);
            if (isValid) {
              ierr = TetGenMeshBowatInsertSite(m, newpt, &splitseg, n, sublists, subceillists, tetlists, ceillists, PETSC_NULL, flipque, PETSC_TRUE, chkencsub, chkbadtet);CHKERRQ(ierr);
              ierr = TetGenMeshSetNewPointSize(m, newpt, e1, e2);CHKERRQ(ierr);
              if (m->steinerleft > 0) m->steinerleft--;
            } else {
              /*  p did not insert for invalid B(p). */
              ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
            }
            /*  Free the memory allocated in formbowatcavity(). */
            ierr = TetGenMeshReleaseBowatCavity(m, &splitseg, n, sublists, subceillists, tetlists, ceillists);CHKERRQ(ierr);
          } else {
            PetscBool isSplit;

            ierr = TetGenMeshSplitEncSeg(m, newpt, &splitseg, tetlist, sublist, PETSC_NULL, flipque, chkencsub, chkbadtet, PETSC_FALSE, &isSplit);CHKERRQ(ierr);
            if (isSplit) {
              ierr = TetGenMeshSetNewPointSize(m, newpt, e1, e2);CHKERRQ(ierr);
              if (m->steinerleft > 0) m->steinerleft--;
            } else {
              /*  Fail to split the segment. It MUST be caused by a very flat */
              /*    tet connected at the splitting segment. We do not handle */
              /*    this case yet. Hopefully, the later repairs will remove */
              /*    the flat tet and hence the segment can be split later. */
              ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
            }
          }
        } else {
          /*  This segment can not be split for not meeting the rules in */
          /*    acceptsegpt(). Mark it to avoid re-checking it later. */
          smarktest(&splitseg);
          /*  p did not accept for insertion. */
          ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
        } /*  if (checkseg4splitting(newpt, &splitseg)) */
      } /*  if ((encloop->forg == pa) && (encloop->fdest == pb)) */
      ierr = TetGenMeshBadFaceDealloc(m, m->badsubsegs, encloop);CHKERRQ(ierr); /*  Remove this entry from list. */
      ierr = TetGenMeshBadFaceTraverse(m, m->badsubsegs, &encloop);CHKERRQ(ierr); /*  Get the next enc-segment. */
    } /*  while ((encloop != (badface *) NULL) && (steinerleft != 0)) */
  } /*  while ((badsubsegs->items > 0) && (steinerleft != 0)) */

  if (!b->fliprepair) {
    ierr = PetscFree(tetlists);CHKERRQ(ierr);
    ierr = PetscFree(ceillists);CHKERRQ(ierr);
    ierr = PetscFree(sublists);CHKERRQ(ierr);
    ierr = PetscFree(subceillists);CHKERRQ(ierr);
  } else {
    ierr = ListDestroy(&tetlist);CHKERRQ(ierr);
    ierr = ListDestroy(&sublist);CHKERRQ(ierr);
    ierr = QueueDestroy(&flipque);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRepairEncSubs"
/*  repairencsubs()    Repair (split) all the encroached subfaces.             */
/*                                                                             */
/*  Each encroached subface is repaired by splitting it - inserting a vertex   */
/*  at or near its circumcenter.  Newly inserted vertices may encroach upon    */
/*  other subfaces, these are also repaired.                                   */
/*                                                                             */
/*  'chkbadtet' is a flag that specifies whether one should take note of new   */
/*  bad quality tets that result from inserted vertices.                       */
/* tetgenmesh::repairencsubs() */
PetscErrorCode TetGenMeshRepairEncSubs(TetGenMesh *m, PetscBool chkbadtet)
{
  TetGenOpts    *b = m->b;
  List *tetlists[2], *ceillists[2];
  List *sublist, *subceillist;
  List *verlist;
  badface *encloop;
  face splitsub = {PETSC_NULL, 0};
  point newpt, e1;
  locateresult loc;
  PetscReal normal[3], len;
  PetscBool reject;
  long oldencsegnum;
  int quenumber, n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = 0;
  sublist = PETSC_NULL;
  subceillist = PETSC_NULL;
  ierr = ListCreate(sizeof(point *), PETSC_NULL, 256, PETSC_DECIDE, &verlist);CHKERRQ(ierr);

  /*  Loop until the pool 'badsubfaces' is empty. Note that steinerleft == -1 */
  /*    if an unlimited number of Steiner points is allowed. */
  while((m->badsubfaces->items > 0) && (m->steinerleft != 0)) {
    /*  Get an encroached subface f. */
    ierr = TetGenMeshDequeueEncSub(m, &quenumber, &encloop);CHKERRQ(ierr);
    splitsub = encloop->ss;
    /*  Clear the in-queue flag of f. */
    setshell2badface(&splitsub, PETSC_NULL);
    /*  f may not be the same one when it was determined to be encroached. */
    if (!isdead_face(&splitsub) && (sorg(&splitsub) == encloop->forg) && (sdest(&splitsub) == encloop->fdest) && (sapex(&splitsub) == encloop->fapex)) {
      PetscInfo4(b->in, "    Dequeuing ensub (%d, %d, %d) [%d].\n", pointmark(m, encloop->forg), pointmark(m, encloop->fdest), pointmark(m, encloop->fapex), quenumber);
      /*  Create a new point p at the circumcenter of f. */
      ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
      for(i = 0; i < 3; i++) newpt[i] = encloop->cent[i];
      setpointtype(m, newpt, FREESUBVERTEX);
      setpoint2sh(m, newpt, sencode(&splitsub));
      /*  Set the abovepoint of f for point location. */
      m->abovepoint = m->facetabovepointarray[shellmark(m, &splitsub)];
      if (!m->abovepoint) {
        /*  getfacetabovepoint(&splitsub); */
        /*  Calculate an abovepoint in dummypoint. */
        ierr = TetGenMeshFaceNormal2(m, encloop->forg, encloop->fdest, encloop->fapex, normal, 1);CHKERRQ(ierr);
        len = sqrt(DOT(normal, normal));
        normal[0] /= len;
        normal[1] /= len;
        normal[2] /= len;
        len = DIST(encloop->forg, encloop->fdest);
        len += DIST(encloop->fdest, encloop->fapex);
        len += DIST(encloop->fapex, encloop->forg);
        len /= 3.0;
        m->dummypoint[0] = encloop->forg[0] + len * normal[0];
        m->dummypoint[1] = encloop->forg[1] + len * normal[1];
        m->dummypoint[2] = encloop->forg[2] + len * normal[2];
        m->abovepoint = m->dummypoint;
      }
      /*  Locate p, start from f, stop at segment (1), use a tolerance to */
      /*    detect ONVERTEX or OUTSIDE case. Update f on return. */
      ierr = TetGenMeshLocateSub(m, newpt, &splitsub, 1, b->epsilon * 1e+2, &loc);CHKERRQ(ierr);
      if ((loc != ONVERTEX) && (loc != OUTSIDE)) {
        /*  Form BC(p), B(p), CBC(p) and C(p). */
        ierr = TetGenMeshFormBowatCavity(m, newpt, PETSC_NULL, &splitsub, &n, PETSC_NULL, &sublist, &subceillist, tetlists, ceillists);CHKERRQ(ierr);
        /*  Check for encroached subsegments (on B(p)). */
        oldencsegnum = m->badsubsegs->items;
        ierr = TetGenMeshTallEncSegs(m, newpt, 2, ceillists, &reject);CHKERRQ(ierr);
        if (reject && (oldencsegnum == m->badsubsegs->items)) {
          /*  'newpt' encroaches upon some subsegments. But none of them can */
          /*     be split. So this subface can't be split as well. Mark it to */
          /*     avoid re-checking it later. */
          smarktest(&encloop->ss);
        }
        /*  Execute point accept rule if p does not encroach upon any segment. */
        if (!reject) {
          ierr = TetGenMeshAcceptFacPt(m, newpt, subceillist, verlist, &reject);CHKERRQ(ierr);
          reject = !reject ? PETSC_TRUE : PETSC_FALSE;
          if (reject) {
            /*  'newpt' lies in some protecting balls. This subface can't be */
            /*     split. Mark it to avoid re-checking it later. */
            smarktest(&encloop->ss);
          }
        }
        if (!reject) {
          /*  Validate/update cavity. */
          ierr = TetGenMeshTrimBowatCavity(m, newpt, PETSC_NULL, n, &sublist, &subceillist, tetlists, ceillists, -1.0, &reject);CHKERRQ(ierr);
          reject = !reject ? PETSC_TRUE : PETSC_FALSE;
        }
        if (!reject) {
          /*  CBC(p) should include s, so that s can be removed after CBC(p) */
          /*    is remeshed. However, if there are locally non-Delaunay faces */
          /*    and encroached subsegments, s may not be collected in CBC(p). */
          /*    p should not be inserted in such case. */
          reject = !sinfected(m, &encloop->ss) ? PETSC_TRUE : PETSC_FALSE;
        }
        if (!reject) {
          /*  Save a point for size interpolation. */
          e1 = sorg(&splitsub);
          ierr = TetGenMeshBowatInsertSite(m, newpt, PETSC_NULL, n, &sublist, &subceillist, tetlists, ceillists, PETSC_NULL, PETSC_NULL, PETSC_TRUE, PETSC_TRUE, chkbadtet);CHKERRQ(ierr);
          ierr = TetGenMeshSetNewPointSize(m, newpt, e1, PETSC_NULL);CHKERRQ(ierr);
          if (m->steinerleft > 0) m->steinerleft--;
        } else {
          /*  p is rejected for the one of the following reasons: */
          /*    (1) BC(p) is not valid. */
          /*    (2) s does not in CBC(p). */
          /*    (3) p encroaches upon some segments (queued); or */
          /*    (4) p is rejected by point accepting rule, or */
          /*    (5) due to the rejection of symp (the PBC). */
          ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
        } /*  if (!reject) */
        /*  Release the cavity and free the memory. */
        ierr = TetGenMeshReleaseBowatCavity(m, PETSC_NULL, n, &sublist, &subceillist, tetlists, ceillists);CHKERRQ(ierr);
        if (reject) {
          /*  Are there queued encroached subsegments. */
          if (m->badsubsegs->items > 0) {
            /*  Repair enc-subsegments. */
            /* oldptnum = m->points->items; */
            ierr = TetGenMeshRepairEncSegs(m, PETSC_TRUE, chkbadtet);CHKERRQ(ierr);
          }
        }
      } else {
        /*  Don't insert p for one of the following reasons: */
        /*    (1) Locate on an existing vertex; or */
        /*    (2) locate outside the domain. */
        /*  Case (1) should not be possible. If such vertex v exists, it is */
        /*    the circumcenter of f, ie., f is non-Delaunay. Either f was got */
        /*    split before by v, but survived after v was inserted, or the */
        /*    same for a f' which is nearly co-circular with f.  Whatsoever, */
        /*    there are encroached segs by v, but the routine tallencsegs() */
        /*    did not find them out. */
        if (loc == ONVERTEX) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_PLIB, "During repairing encroached subface (%d, %d, %d)\n  New point %d is coincident with an existing vertex %d\n",
                                      pointmark(m, encloop->forg), pointmark(m, encloop->fdest), pointmark(m, encloop->fapex), pointmark(m, newpt), pointmark(m, sorg(&splitsub)));
        if (loc != OUTSIDE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        /*  The circumcenter lies outside of the facet. Mark it to avoid */
        /*    rechecking it later. */
        smarktest(&encloop->ss);
        /*  Case (2) can happen when thers is a segment s which is close to f */
        /*    and is non-conforming Delaunay. The circumcenter of f encroaches */
        /*    upon s, but the circumcenter of s is rejected for insertion. */
        ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
      } /*  if ((loc != ONVERTEX) && (loc != OUTSIDE)) */
    }
    /*  Remove this entry from list. */
    ierr = TetGenMeshBadFaceDealloc(m, m->badsubfaces, encloop);CHKERRQ(ierr);
  } /*  while ((badsubfaces->items > 0) && (steinerleft != 0)) */

  ierr = ListDestroy(&verlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRepairBadTets"
/*  repairbadtets()    Repair all bad-quality tetrahedra.                      */
/*                                                                             */
/*  All bad-quality tets are stored in pool 'badtetrahedrons'.  Each bad tet   */
/*  is repaired by inserting a point at or near its circumcenter. However, if  */
/*  this point encroaches any subsegment or subface, it is not inserted. Ins-  */
/*  tead the encroached segment and subface are split.  Newly inserted points  */
/*  may create other bad-quality tets, these are also repaired.                */
/* tetgenmesh::repairbadtets() */
PetscErrorCode TetGenMeshRepairBadTets(TetGenMesh *m)
{
  TetGenOpts    *b = m->b;
  List *tetlist, *ceillist;
  List *verlist;
  ArrayPool *histtetarray;
  badface *badtet;
  triface starttet = {PETSC_NULL, 0, 0};
  point newpt, e1;
  locateresult loc;
  PetscBool reject;
  long oldptnum;
  int len, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ListCreate(sizeof(triface), PETSC_NULL, 1024, PETSC_DECIDE, &tetlist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(triface), PETSC_NULL, 1024, PETSC_DECIDE, &ceillist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(point *), PETSC_NULL,  256, PETSC_DECIDE, &verlist);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(triface), 8, &histtetarray);CHKERRQ(ierr);

  /*  Loop until pool 'badtetrahedrons' is empty. Note that steinerleft == -1 */
  /*    if an unlimited number of Steiner points is allowed. */
  while((m->badtetrahedrons->items > 0) && (m->steinerleft != 0)) {
    /*  Get a bad-quality tet t. */
    ierr = TetGenMeshTopBadTetra(m, &badtet);CHKERRQ(ierr);
    /*  Make sure that the tet is still the same one when it was tested. */
    /*    Subsequent transformations may have made it a different tet. */
    if (badtet && !isdead_triface(&badtet->tt) &&
        org(&badtet->tt)  == badtet->forg  && dest(&badtet->tt) == badtet->fdest &&
        apex(&badtet->tt) == badtet->fapex && oppo(&badtet->tt) == badtet->foppo) {
      PetscInfo4(b->in, "    Dequeuing btet (%d, %d, %d, %d).\n", pointmark(m, badtet->forg), pointmark(m, badtet->fdest), pointmark(m, badtet->fapex), pointmark(m, badtet->foppo));
      /*  Create the new point p (at the circumcenter of t). */
      ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
      for(i = 0; i < 3; i++) newpt[i] = badtet->cent[i];
      setpointtype(m, newpt, FREEVOLVERTEX);
      /*  Locate p. */
      starttet = badtet->tt;
      /* loc = preciselocate(newpt, &starttet, tetrahedrons->items); */
      ierr = TetGenMeshLocate2(m, newpt, &starttet, histtetarray, &loc);CHKERRQ(ierr);
      PetscInfo1(b->in, "    loc = %d.\n", (int) loc);
      if ((loc != ONVERTEX) && (loc != OUTSIDE)) {
        /*  For BC(p) and B(p). */
        infect(m, &starttet);
        ierr = ListAppend(tetlist, &starttet, PETSC_NULL);CHKERRQ(ierr);
        ierr = TetGenMeshFormBowatCavityQuad(m, newpt, tetlist, ceillist);CHKERRQ(ierr);
        /*  Check for encroached subsegments. */
        ierr = TetGenMeshTallEncSegs(m, newpt, 1, &ceillist, &reject);CHKERRQ(ierr);
        if (!reject) {
          /*  Check for encroached subfaces. */
          ierr = TetGenMeshTallEncSubs(m, newpt, 1, &ceillist, &reject);CHKERRQ(ierr);
        }
        /*  Execute point accepting rule if p does not encroach upon any subsegment and subface. */
        if (!reject) {
          ierr = TetGenMeshAcceptVolPt(m, newpt, ceillist, verlist, &reject);CHKERRQ(ierr);
          reject = !reject ? PETSC_TRUE : PETSC_FALSE;
        }
        if (!reject) {
          ierr = TetGenMeshTrimBowatCavity(m, newpt, PETSC_NULL, 1, PETSC_NULL, PETSC_NULL, &tetlist, &ceillist, -1.0, &reject);CHKERRQ(ierr);
          reject = !reject ? PETSC_TRUE : PETSC_FALSE;
        }
        if (!reject) {
          /*  BC(p) should include t, so that t can be removed after BC(p) is */
          /*    remeshed. However, if there are locally non-Delaunay faces */
          /*    and encroached subsegments/subfaces, t may not be collected */
          /*    in BC(p). p should not be inserted in such case. */
          reject = !infected(m, &badtet->tt) ? PETSC_TRUE : PETSC_FALSE;
          if (reject) m->outbowatcircumcount++;
        }
        if (!reject) {
          /*  Save a point for size interpolation. */
          e1 = org(&starttet);
          /*  Insert p. */
          ierr = TetGenMeshBowatInsertSite(m, newpt, PETSC_NULL, 1, PETSC_NULL, PETSC_NULL, &tetlist, &ceillist, PETSC_NULL, PETSC_NULL, PETSC_FALSE, PETSC_FALSE, PETSC_TRUE);CHKERRQ(ierr);
          ierr = TetGenMeshSetNewPointSize(m, newpt, e1, PETSC_NULL);CHKERRQ(ierr);
          if (m->steinerleft > 0) m->steinerleft--;
        } else {
          /*  p is rejected for one of the following reasons: */
          /*    (1) BC(p) is not valid. */
          /*    (2) t does not in BC(p). */
          /*    (3) p encroaches upon some segments; */
          /*    (4) p encroaches upon some subfaces; */
          /*    (5) p is rejected by the point accepting rule. */
          ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
          /*  Uninfect tets of BC(p). */
          ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
          for(i = 0; i < len; i++) {
            ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
            uninfect(m, &starttet);
          }
        }
        ierr = ListClear(tetlist);CHKERRQ(ierr);
        ierr = ListClear(ceillist);CHKERRQ(ierr);
        /*  Split encroached subsegments/subfaces if there are. */
        if (reject) {
          oldptnum = m->points->items;
          if (m->badsubsegs->items > 0) {
            ierr = TetGenMeshRepairEncSegs(m, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
          }
          if (m->badsubfaces->items > 0) {
            ierr = TetGenMeshRepairEncSubs(m, PETSC_TRUE);CHKERRQ(ierr);
          }
          if (m->points->items > oldptnum) {
            /*  Some encroaching subsegments/subfaces got split. Re-queue the */
            /*    tet if it is still alive. */
            starttet = badtet->tt;
            if (!isdead_triface(&starttet)) {
              ierr = TetGenMeshCheckTet4BadQual(m, &starttet, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
            }
          }
        }
      } else {
        /*  Do not insert p. The reason may be one of: */
        /*    (1) p is coincident (ONVERTEX) with an existing vertex; or */
        /*    (2) p is outside (OUTSIDE) the mesh. */
        /*  Case (1) should not be possible. If such vertex v exists, it is */
        /*    the circumcenter of t, ie., t is non-Delaunay. Either t was got */
        /*    split before by v, but survived after v was inserted, or the */
        /*    same for a t' which is nearly co-spherical with t.  Whatsoever, */
        /*    there are encroached segments or subfaces by v but the routines */
        /*    tallencsegs() or tallencsubs() did not find them out. */
        /*  Case (2) can happen when there is a segment s (or subface f) which */
        /*    is close to f and is non-conforming Delaunay.  The circumcenter */
        /*    of t encroaches upon s (or f), but the circumcenter of s (or f) */
        /*    is rejected for insertion. */
        ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
      } /*  if ((loc != ONVERTEX) && (loc != OUTSIDE)) */
    } /*  if (!isdead(&badtet->tt) && org(badtet->tt) == badtet->forg && */
    /*  Remove the tet from the queue. */
    ierr = TetGenMeshDequeueBadTet(m);CHKERRQ(ierr);
  } /*  while ((badtetrahedrons->items > 0) && (steinerleft != 0)) */

  ierr = ListDestroy(&tetlist);CHKERRQ(ierr);
  ierr = ListDestroy(&ceillist);CHKERRQ(ierr);
  ierr = ListDestroy(&verlist);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&histtetarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshEnforceQuality"
/*  enforcequality()    Refine the mesh.                                       */
/* tetgenmesh::enforcequality() */
PetscErrorCode TetGenMeshEnforceQuality(TetGenMesh *m)
{
  TetGenOpts    *b = m->b;
  long total, vertcount;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Adding Steiner points to enforce quality.\n");

  total = vertcount = 0l;
  if (b->conformdel) {
    m->r2count = m->r3count = 0l;
  }

  /*  If both '-D' and '-r' options are used. */
  if (b->conformdel && b->refine) {
    ierr = TetGenMeshMarkAcuteVertices(m, 65.0);CHKERRQ(ierr);
  }
  /*  If '-m' is not used. */
  if (!b->metric) {
    /*  Find and mark all sharp segments. */
    ierr = TetGenMeshMarkSharpSegments(m, 65.0);CHKERRQ(ierr);
    /*  Decide the sizes for feature points. */
    ierr = TetGenMeshDecideFeaturePointSizes(m);CHKERRQ(ierr);
  }

  /*  Initialize the pool of encroached subsegments. */
  ierr = MemoryPoolCreate(sizeof(badface), SUBPERBLOCK, POINTER, 0, &m->badsubsegs);CHKERRQ(ierr);
  /*  Looking for encroached subsegments. */
  ierr = TetGenMeshTallEncSegs(m, PETSC_NULL, 0, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (m->badsubsegs->items > 0) {
    PetscInfo(b->in, "  Splitting encroached subsegments.\n");
  }
  vertcount = m->points->items;
  /*  Fix encroached segments without noting any enc subfaces. */
  ierr = TetGenMeshRepairEncSegs(m, PETSC_FALSE, PETSC_FALSE);CHKERRQ(ierr);
  PetscInfo1(b->in, "  %ld split points.\n", m->points->items - vertcount);
  total += m->points->items - vertcount;

  /*  Initialize the pool of encroached subfaces. */
  ierr = MemoryPoolCreate(sizeof(badface), SUBPERBLOCK, POINTER, 0, &m->badsubfaces);CHKERRQ(ierr);
  /*  Initialize the priority queues of badfaces. */
  for(i = 0; i < 3; i++) m->subquefront[i] = PETSC_NULL;
  for(i = 0; i < 3; i++) m->subquetail[i]  = &m->subquefront[i];
  /*  Looking for encroached subfaces. */
  ierr = TetGenMeshTallEncSubs(m, PETSC_NULL, 0, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (m->badsubfaces->items > 0) {
    PetscInfo(b->in, "  Splitting encroached subfaces.\n");
  }
  vertcount = m->points->items;
  /*  Fix encroached subfaces without noting bad tetrahedra. */
  ierr = TetGenMeshRepairEncSubs(m, PETSC_FALSE);CHKERRQ(ierr);
  PetscInfo1(b->in, "  %ld split points.\n", m->points->items - vertcount);
  total += m->points->items - vertcount;
  /*  At this point, the mesh should be conforming Delaunay if no input */
  /*    angle is smaller than 90 degree. */

  /*  Next, fix bad quality tetrahedra. */
  if ((b->minratio > 0.0) || b->varvolume || b->fixedvolume) {
    /*  Initialize the pool of bad tets */
    ierr = MemoryPoolCreate(sizeof(badface), ELEPERBLOCK, POINTER, 0, &m->badtetrahedrons);CHKERRQ(ierr);
    /*  Initialize the priority queues of bad tets. */
    for(i = 0; i < 64; i++) m->tetquefront[i] = PETSC_NULL;
    m->firstnonemptyq = -1;
    m->recentq = -1;
    /*  Looking for bad quality tets. */
    m->cosmaxdihed = cos(b->maxdihedral * PETSC_PI / 180.0);
    m->cosmindihed = cos(b->mindihedral * PETSC_PI / 180.0);
    ierr = TetGenMeshTallBadTetrahedrons(m);CHKERRQ(ierr);
    if (m->badtetrahedrons->items > 0) {
      PetscInfo(b->in, "  Splitting bad tetrahedra.\n");
    }
    vertcount = m->points->items;
    ierr = TetGenMeshRepairBadTets(m);CHKERRQ(ierr);
    PetscInfo1(b->in, "  %ld refinement points.\n", m->points->items - vertcount);
    total += m->points->items - vertcount;
    ierr = MemoryPoolDestroy(&m->badtetrahedrons);CHKERRQ(ierr);
  }

  PetscInfo1(b->in, "  Totally added %ld points.\n", total);
  ierr = MemoryPoolDestroy(&m->badsubfaces);CHKERRQ(ierr);
  ierr = MemoryPoolDestroy(&m->badsubsegs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  refine_cxx /////////////////////////////////////////////////////////////// */

/*  optimize_cxx ///////////////////////////////////////////////////////////// */
/*                                                                        //// */
/*                                                                        //// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCheckTet4Ill"
/*  checktet4ill()    Check a tet to see if it is illegal.                     */
/*                                                                             */
/*  A tet is "illegal" if it spans on one input facet.  Save the tet in queue  */
/*  if it is illegal and the flag 'enqflag' is set.                            */
/*                                                                             */
/*  Note: Such case can happen when the input facet has non-coplanar vertices  */
/*  and the Delaunay tetrahedralization of the vertices may creat such tets.   */
/* tetgenmesh::checktet4ill() */
PetscErrorCode TetGenMeshCheckTet4Ill(TetGenMesh *m, triface* testtet, PetscBool enqflag, PetscBool *isIllegal)
{
  TetGenOpts    *b = m->b;
  badface *newbadtet;
  triface checktet = {PETSC_NULL, 0, 0};
  face checksh1 = {PETSC_NULL, 0}, checksh2 = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  PetscBool illflag;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  illflag = PETSC_FALSE;
  for (testtet->loc = 0; testtet->loc < 4; testtet->loc++) {
    tspivot(m, testtet, &checksh1);
    if (checksh1.sh != m->dummysh) {
      testtet->ver = 0;
      ierr = TetGenMeshFindEdge_face(m, &checksh1, org(testtet), dest(testtet));CHKERRQ(ierr);
      for(i = 0; i < 3; i++) {
        fnext(m, testtet, &checktet);
        tspivot(m, &checktet, &checksh2);
        if (checksh2.sh != m->dummysh) {
          /*  Two subfaces share this edge. */
          sspivot(m, &checksh1, &checkseg);
          if (checkseg.sh == m->dummysh) {
            /*  The four corners of the tet are on one facet. Illegal! Try to */
            /*    flip the opposite edge of the current one. */
            enextfnextself(m, testtet);
            enextself(testtet);
            illflag = PETSC_TRUE;
            break;
          }
        }
        enextself(testtet);
        senextself(&checksh1);
      }
    }
    if (illflag) break;
  }

  if (illflag && enqflag) {
    /*  Allocate space for the bad tetrahedron. */
    ierr = MemoryPoolAlloc(m->badtetrahedrons, (void **) &newbadtet);CHKERRQ(ierr);
    newbadtet->tt = *testtet;
    newbadtet->key = -1.0; /*  = 180 degree. */
    for(i = 0; i < 3; i++) newbadtet->cent[i] = 0.0;
    newbadtet->forg = org(testtet);
    newbadtet->fdest = dest(testtet);
    newbadtet->fapex = apex(testtet);
    newbadtet->foppo = oppo(testtet);
    newbadtet->nextitem = PETSC_NULL;
    PetscInfo4(b->in, "    Queueing illtet: (%d, %d, %d, %d).\n", pointmark(m, newbadtet->forg), pointmark(m, newbadtet->fdest),
               pointmark(m, newbadtet->fapex), pointmark(m, newbadtet->foppo));
  }

  if (isIllegal) {*isIllegal = illflag;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCheckTet4Opt"
/*  checktet4opt()    Check a tet to see if it needs to be optimized.          */
/*                                                                             */
/*  A tet t needs to be optimized if it fails to certain quality measures.     */
/*  The only quality measure currently used is the maximal dihedral angle at   */
/*  edges. The desired maximal dihedral angle is 'b->maxdihedal' (set by the   */
/*  '-qqq' option.                                                             */
/*                                                                             */
/*  A tet may have one, two, or three big dihedral angles. Examples: Let the   */
/*  tet t = abcd, and its four corners are nearly co-planar. Then t has one    */
/*  big dihedral angle if d is very close to the edge ab; t has three big      */
/*  dihedral angles if d's projection on the face abc is also inside abc, i.e. */
/*  the shape of t likes a hat; finally, t has two big dihedral angles if d's  */
/*  projection onto abc is outside abc.                                        */
/* tetgenmesh::checktet4opt() */
PetscErrorCode TetGenMeshCheckTet4Opt(TetGenMesh *m, triface *testtet, PetscBool enqflag, PetscBool *doOpt)
{
  TetGenOpts    *b = m->b;
  badface *newbadtet;
  point pa, pb, pc, pd;
  PetscReal N[4][3], len;
  PetscReal cosd = 0.0;
  int count;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pa = (point) testtet->tet[4];
  pb = (point) testtet->tet[5];
  pc = (point) testtet->tet[6];
  pd = (point) testtet->tet[7];
  /*  Compute the 4 face normals: N[0] cbd, N[1] acd, N[2] bad, N[3] abc. */
  ierr = TetGenMeshTetAllNormal(m, pa, pb, pc, pd, N, PETSC_NULL);CHKERRQ(ierr);
  /*  Normalize the normals. */
  for(i = 0; i < 4; i++) {
    len = sqrt(dot(N[i], N[i]));
    if (len != 0.0) {
      for(j = 0; j < 3; j++) N[i][j] /= len;
    }
  }
  count = 0;

  /*  Find all large dihedral angles. */
  for(i = 0; i < 6; i++) {
    /*  Locate the edge i and calculate the dihedral angle at the edge. */
    testtet->loc = 0;
    testtet->ver = 0;
    switch (i) {
    case 0: /*  edge ab */
      cosd = -dot(N[2], N[3]);
      break;
    case 1: /*  edge cd */
      enextfnextself(m, testtet);
      enextself(testtet);
      cosd = -dot(N[0], N[1]);
      break;
    case 2: /*  edge bd */
      enextfnextself(m, testtet);
      enext2self(testtet);
      cosd = -dot(N[0], N[2]);
      break;
    case 3: /*  edge bc */
      enextself(testtet);
      cosd = -dot(N[0], N[3]);
      break;
    case 4: /*  edge ad */
      enext2fnextself(m, testtet);
      enextself(testtet);
      cosd = -dot(N[1], N[2]);
      break;
    case 5: /*  edge ac */
      enext2self(testtet);
      cosd = -dot(N[1], N[3]);
      break;
    }
    if (cosd < m->cosmaxdihed) {
      /*  A bigger dihedral angle. */
      count++;
      if (enqflag) {
        /*  Allocate space for the bad tetrahedron. */
        ierr = MemoryPoolAlloc(m->badtetrahedrons, (void **) &newbadtet);CHKERRQ(ierr);
        newbadtet->tt = *testtet;
        newbadtet->key = cosd;
        for(j = 0; j < 3; j++) newbadtet->cent[j] = 0.0;
        newbadtet->forg  = org(testtet);
        newbadtet->fdest = dest(testtet);
        newbadtet->fapex = apex(testtet);
        newbadtet->foppo = oppo(testtet);
        newbadtet->nextitem = PETSC_NULL;
        PetscInfo5(b->in, "    Queueing tet: (%d, %d, %d, %d), dihed %g (degree).\n", pointmark(m, newbadtet->forg), pointmark(m, newbadtet->fdest),
                   pointmark(m, newbadtet->fapex), pointmark(m, newbadtet->foppo), acos(cosd) * 180.0 / PETSC_PI);
      }
    }
  }

  if (doOpt) {*doOpt = count > 0 ? PETSC_TRUE : PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRemoveEdge"
/*  removeedge()    Remove an edge                                             */
/*                                                                             */
/*  'remedge' is a tet (abcd) having the edge ab wanted to be removed.  Local  */
/*  reconnecting operations are used to remove edge ab.  The following opera-  */
/*  tion will be tryed.                                                        */
/*                                                                             */
/*  If ab is on the hull, and abc and abd are both hull faces. Then ab can be  */
/*  removed by stripping abcd from the mesh. However, if ab is a segemnt, do   */
/*  the operation only if 'b->optlevel' > 1 and 'b->nobisect == 0'.            */
/*                                                                             */
/*  If ab is an internal edge, there are n tets contains it.  Then ab can be   */
/*  removed if there exists another m tets which can replace the n tets with-  */
/*  out changing the boundary of the n tets.                                   */
/*                                                                             */
/*  If 'optflag' is set.  The value 'remedge->key' means cos(theta), where     */
/*  'theta' is the maximal dishedral angle at ab. In this case, even if the    */
/*  n-to-m flip exists, it will not be performed if the maximum dihedral of    */
/*  the new tets is larger than 'theta'.                                       */
/* tetgenmesh::removeedge() */
PetscErrorCode TetGenMeshRemoveEdge(TetGenMesh *m, badface* remedge, PetscBool optflag, PetscBool *isRemoved)
{
  TetGenOpts    *b = m->b;
  triface abcd = {PETSC_NULL, 0, 0}, badc = {PETSC_NULL, 0, 0};  /*  Tet configuration at edge ab. */
  triface baccasing = {PETSC_NULL, 0, 0}, abdcasing = {PETSC_NULL, 0, 0};
  triface abtetlist[21];  /*  Old configuration at ab, save maximum 20 tets. */
  triface bftetlist[21];  /*  Old configuration at bf, save maximum 20 tets. */
  triface newtetlist[90]; /*  New configuration after removing ab. */
  face checksh = {PETSC_NULL, 0};
  PetscReal key;
  PetscBool remflag, subflag;
  int n, n1, m1, i, j, k;
  triface newtet = {PETSC_NULL, 0, 0};
  point *ppt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  First try to strip abcd from the mesh. This needs to check either ab */
  /*    or cd is on the hull. Try to strip it whichever is true. */
  abcd = remedge->tt;
  adjustedgering_triface(&abcd, CCW);
  k = 0;
  do {
    sym(&abcd, &baccasing);
    /*  Is the tet on the hull? */
    if (baccasing.tet == m->dummytet) {
      fnext(m, &abcd, &badc);
      sym(&badc, &abdcasing);
      if (abdcasing.tet == m->dummytet) {
        /*  Strip the tet from the mesh -> ab is removed as well. */
#if 1
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
        if (removetetbypeeloff(&abcd, newtetlist)) {
          PetscInfo(b->in, "    Stripped tet from the mesh.\n");
          m->optcount[0]++;
          m->opt_tet_peels++;
          /*  edge is removed. Test new tets for further optimization. */
          for(i = 0; i < 2; i++) {
            if (optflag) {
              ierr = TetGenMeshCheckTet4Opt(m, &(newtetlist[i]), PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
            } else {
              ierr = TetGenMeshCheckTet4Ill(m, &(newtetlist[i]), PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
            }
          }
          /*  Update the point-to-tet map */
          for(i = 0; i < 2; i++) {
            newtet = newtetlist[i];
            ppt = (point *) &(newtet.tet[4]);
            for (j = 0; j < 4; j++) {
              setpoint2tet(m, ppt[j], encode(&newtet));
            }
          }
          if (isRemoved) {*isRemoved = PETSC_TRUE;}
          PetscFunctionReturn(0);
        }
#endif
      }
    }
    /*  Check if the oppsite edge cd is on the hull. */
    enext2fnextself(m, &abcd);
    enext2self(&abcd);
    esymself(&abcd); /*  --> cdab */
    k++;
  } while (k < 2);

  /*  Get the tets configuration at ab. Collect maximum 10 tets. */
  subflag = PETSC_FALSE;
  abcd = remedge->tt;
  adjustedgering_triface(&abcd, CW);
  n = 0;
  abtetlist[n] = abcd;
  do {
    /*  Is the list full? */
    if (n == 20) break;
    /*  Stop if a subface appears. */
    tspivot(m, &abtetlist[n], &checksh);
    if (checksh.sh != m->dummysh) {
      /*  ab is either a segment or a facet edge. The latter case is not */
      /*    handled yet! An edge flip is needed. */
      subflag = PETSC_TRUE; break; /*  return false; */
    }
    /*  Get the next tet at ab. */
    fnext(m, &abtetlist[n], &abtetlist[n + 1]);
    n++;
  } while (apex(&abtetlist[n]) != apex(&abcd));

  remflag = PETSC_FALSE;
  key = remedge->key;

  if (subflag && optflag) {
    /*  Faces are not flipable. Return. */
    if (isRemoved) {*isRemoved = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  /*  2 < n < 20. */
  if (n == 3) {
    /*  There are three tets at ab. Try to do a flip32 at ab. */
    ierr = TetGenMeshRemoveEdgeByFlip32(m, &key, abtetlist, newtetlist, PETSC_NULL, &remflag);CHKERRQ(ierr);
  } else if ((n > 3) && (n <= b->maxflipedgelinksize)) {
    /*  Four tets case. Try to do edge transformation. */
    ierr = TetGenMeshRemoveEdgeByTranNM(m, &key,n,abtetlist,newtetlist,PETSC_NULL,PETSC_NULL,PETSC_NULL, &remflag);CHKERRQ(ierr);
  } else {
    PetscInfo1(b->in, "  !! Unhandled case: n = %d.\n", n);
  }
  if (remflag) {
    m->optcount[n]++;
    /*  Delete the old tets. */
    for(i = 0; i < n; i++) {
      ierr = TetGenMeshTetrahedronDealloc(m, abtetlist[i].tet);CHKERRQ(ierr);
    }
    m1 = (n - 2) * 2; /*  The number of new tets. */
    if (b->verbose > 1) {
      if (optflag) {
        PetscInfo4(b->in, "  Done flip %d-to-%d Qual: %g -> %g.", n, m1, acos(remedge->key) / PETSC_PI * 180.0, acos(key) / PETSC_PI * 180.0);
      } else {
        PetscInfo2(b->in, "  Done flip %d-to-%d.\n", n, m1);
      }
    }
  }

  if (!remflag && (key == remedge->key) && (n <= b->maxflipedgelinksize)) {
    /*  Try to do a combination of flips. */
    n1 = 0;
    ierr = TetGenMeshRemoveEdgeByCombNM(m, &key, n, abtetlist, &n1, bftetlist, newtetlist, PETSC_NULL, &remflag);CHKERRQ(ierr);
    if (remflag) {
      m->optcount[9]++;
      /*  Delete the old tets. */
      for(i = 0; i < n; i++) {
        ierr = TetGenMeshTetrahedronDealloc(m, abtetlist[i].tet);CHKERRQ(ierr);
      }
      for(i = 0; i < n1; i++) {
        if (!isdead_triface(&(bftetlist[i]))) {
          ierr = TetGenMeshTetrahedronDealloc(m, bftetlist[i].tet);CHKERRQ(ierr);
        }
      }
      m1 = ((n1 - 2) * 2 - 1) + (n - 3) * 2; /*  The number of new tets. */
      if (optflag) {
        PetscInfo6(b->in, "  Done flip %d-to-%d (n-1=%d, n1=%d) Qual: %g -> %g.\n", n+n1-2, m1, n-1, n1, acos(remedge->key) / PETSC_PI * 180.0, acos(key) / PETSC_PI * 180.0);
      } else {
        PetscInfo4(b->in, "  Done flip %d-to-%d (n-1=%d, n1=%d).\n", n+n1-2, m1, n-1, n1);
      }
    }
  }

  if (remflag) {
    /*  edge is removed. Test new tets for further optimization. */
    for(i = 0; i < m1; i++) {
      if (optflag) {
        ierr = TetGenMeshCheckTet4Opt(m, &(newtetlist[i]), PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      } else {
        ierr = TetGenMeshCheckTet4Ill(m, &(newtetlist[i]), PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      }
    }
    /*  Update the point-to-tet map */
    for(i = 0; i < m1; i++) {
      newtet = newtetlist[i];
      ppt = (point *) &(newtet.tet[4]);
      for (j = 0; j < 4; j++) {
        setpoint2tet(m, ppt[j], encode(&newtet));
      }
    }
    m->opt_edge_flips++;
  }

  if (isRemoved) {*isRemoved = remflag;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSmoothPoint"
/* tetgenmesh::smoothpoint() */
PetscErrorCode TetGenMeshSmoothPoint(TetGenMesh *m, point smthpt, point e1, point e2, List *starlist, PetscBool invtori, PetscReal *key, PetscBool *isSmooth)
{
  TetGenOpts    *b  = m->b;
  triface starttet = {PETSC_NULL, 0, 0};
  point pa, pb, pc;
  PetscReal fcent[3], startpt[3], nextpt[3], bestpt[3];
  PetscReal iniTmax, oldTmax, newTmax;
  PetscReal ori, aspT, aspTmax = 0.0, imprate;
  PetscReal cosd, maxcosd;
  PetscBool segflag, randflag;
  int numdirs;
  int len, iter, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Is p a segment vertex? */
  segflag = e1 ? PETSC_TRUE : PETSC_FALSE;
  /*  Decide the number of moving directions. */
  ierr = ListLength(starlist, &len);CHKERRQ(ierr);
  numdirs = segflag ? 2 : len;
  randflag = numdirs > 10 ? PETSC_TRUE : PETSC_FALSE;
  if (randflag) {
    numdirs = 10; /*  Maximum 10 directions. */
  }

  /*  Calculate the initial object value (the largest aspect ratio). */
  for(i = 0; i < len; i++) {
    ierr = ListItem(starlist, i, (void **) &starttet);CHKERRQ(ierr);
    adjustedgering_triface(&starttet, !invtori ? CCW : CW);
    pa = org(&starttet);
    pb = dest(&starttet);
    pc = apex(&starttet);
    ierr = TetGenMeshTetAspectRatio(m, pa, pb, pc, smthpt, &aspT);CHKERRQ(ierr);
    if (i == 0) {
      aspTmax = aspT;
    } else {
      aspTmax = aspT > aspTmax ? aspT : aspTmax;
    }
  }
  iniTmax = aspTmax;

  PetscInfo5(b->in, "    Smooth %s point %d (%g, %g, %g).\n", segflag ? "seg" : "vol", pointmark(m, smthpt), smthpt[0], smthpt[1], smthpt[2]);
  PetscInfo1(b->in, "    Initial max L/h = %g.\n", iniTmax);
  for(i = 0; i < 3; i++) {
    bestpt[i] = startpt[i] = smthpt[i];
  }

  /*  Do iteration until the new aspTmax does not decrease. */
  newTmax = iniTmax;
  iter = 0;
  while(1) {
    /*  Find the best next location. */
    oldTmax = newTmax;
    for(i = 0; i < numdirs; i++) {
      /*  Calculate the moved point (saved in 'nextpt'). */
      if (!segflag) {
        if (randflag) {
          /*  Randomly pick a direction. */
          ierr = TetGenMeshRandomChoice(m, len, &j);CHKERRQ(ierr);
        } else {
          j = i;
        }
        ierr = ListItem(starlist, j, (void **) &starttet);CHKERRQ(ierr);
        adjustedgering_triface(&starttet, !invtori ? CCW : CW);
        pa = org(&starttet);
        pb = dest(&starttet);
        pc = apex(&starttet);
        for(j = 0; j < 3; j++) {
          fcent[j] = (pa[j] + pb[j] + pc[j]) / 3.0;
        }
      } else {
        for(j = 0; j < 3; j++) {
          fcent[j] = (i == 0 ? e1[j] : e2[j]);
        }
      }
      for(j = 0; j < 3; j++) {
        nextpt[j] = startpt[j] + 0.01 * (fcent[j] - startpt[j]);
      }
      /*  Get the largest object value for the new location. */
      for(j = 0; j < len; j++) {
        ierr = ListItem(starlist, j, (void **) &starttet);CHKERRQ(ierr);
        adjustedgering_triface(&starttet, !invtori ? CCW : CW);
        pa = org(&starttet);
        pb = dest(&starttet);
        pc = apex(&starttet);
        ori = TetGenOrient3D(pa, pb, pc, nextpt);
        if (ori < 0.0) {
          ierr = TetGenMeshTetAspectRatio(m, pa, pb, pc, nextpt, &aspT);CHKERRQ(ierr);
          if (j == 0) {
            aspTmax = aspT;
          } else {
            aspTmax = aspT > aspTmax ? aspT : aspTmax;
          }
        } else {
          /*  An invalid new tet. Discard this point. */
          aspTmax = newTmax;
        } /*  if (ori < 0.0) */
        /*  Stop looping when the object value is bigger than before. */
        if (aspTmax >= newTmax) break;
      } /*  for (j = 0; j < starlist->len(); j++) */
      if (aspTmax < newTmax) {
        /*  Save the improved object value and the location. */
        newTmax = aspTmax;
        for(j = 0; j < 3; j++) bestpt[j] = nextpt[j];
      }
    } /*  for (i = 0; i < starlist->len(); i++) */
    /*  Does the object value improved much? */
    imprate = fabs(oldTmax - newTmax) / oldTmax;
    if (imprate < 1e-3) break;
    /*  Yes, move p to the new location and continue. */
    for (j = 0; j < 3; j++) startpt[j] = bestpt[j];
    iter++;
  } /*  while (true) */

  if (iter > 0) {
    /*  The point is moved. */
    if (key) {
      /*  Check if the quality is improved by the smoothed point. */
      maxcosd = 0.0; /*  = cos(90). */
      for(j = 0; j < len; j++) {
        ierr = ListItem(starlist, j, (void **) &starttet);CHKERRQ(ierr);
        adjustedgering_triface(&starttet, !invtori ? CCW : CW);
        pa = org(&starttet);
        pb = dest(&starttet);
        pc = apex(&starttet);
        ierr = TetGenMeshTetAllDihedral(m, pa, pb, pc, startpt, PETSC_NULL, &cosd, PETSC_NULL);CHKERRQ(ierr);
        if (cosd < *key) {
          /*  This quality will not be improved. Stop. */
          iter = 0; break;
        } else {
          /*  Remeber the worst quality value (of the new configuration). */
          maxcosd = maxcosd < cosd ? maxcosd : cosd;
        }
      }
      if (iter > 0) *key = maxcosd;
    }
  }

  if (iter > 0) {
    if (segflag) m->smoothsegverts++;
    for(i = 0; i < 3; i++) smthpt[i] = startpt[i];
    PetscInfo5(b->in, "    Move to new location (%g, %g, %g).\n    Final max L/h = %g. (%d iterations)\n", smthpt[0], smthpt[1], smthpt[2], newTmax, iter);
    if (key) {PetscInfo1(b->in, "    Max. dihed = %g (degree).\n", acos(*key) / PETSC_PI * 180.0);}
  } else {
    PetscInfo(b->in, "    Not smoothed.\n");
  }
  if (isSmooth) {*isSmooth = iter > 0 ? PETSC_TRUE : PETSC_FALSE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSmoothSliver"
/*  smoothsliver()    Remove a sliver by smoothing a vertex of it.             */
/*                                                                             */
/*  The 'slivtet' represents a sliver abcd, and ab is the current edge which   */
/*  has a large dihedral angle (close to 180 degree).                          */
/* tetgenmesh::smoothsliver() */
PetscErrorCode TetGenMeshSmoothSliver(TetGenMesh *m, badface *remedge, List *starlist, PetscBool *isSmooth)
{
  PLC           *in = m->in;
  triface checktet = {PETSC_NULL, 0, 0};
  point smthpt;
  PetscBool smthed;
  int idx, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Find a Steiner volume point and smooth it. */
  smthed = PETSC_FALSE;
  for(i = 0; i < 4 && !smthed; i++) {
    smthpt = (point) remedge->tt.tet[4 + i];
    /*  Is it a volume point? */
    if (pointtype(m, smthpt) == FREEVOLVERTEX) {
      /*  Is it a Steiner point? */
      idx = pointmark(m, smthpt) - in->firstnumber;
      if (!(idx < in->numberofpoints)) {
        /*  Smooth a Steiner volume point. */
        ierr = ListAppend(starlist, &(remedge->tt.tet), PETSC_NULL);CHKERRQ(ierr);
        ierr = TetGenMeshFormStarPolyhedron(m, smthpt, starlist, PETSC_NULL, PETSC_FALSE);CHKERRQ(ierr);
        ierr = TetGenMeshSmoothPoint(m, smthpt,PETSC_NULL,PETSC_NULL,starlist,PETSC_FALSE,&remedge->key, &smthed);CHKERRQ(ierr);
        /*  If it is smoothed. Queue new bad tets. */
        if (smthed) {
          int len;

          ierr = ListLength(starlist, &len);CHKERRQ(ierr);
          for(j = 0; j < len; j++) {
            ierr = ListItem(starlist, j, (void **) &checktet);CHKERRQ(ierr);
            ierr = TetGenMeshCheckTet4Opt(m, &checktet, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
          }
        }
        ierr = ListClear(starlist);CHKERRQ(ierr);
      }
    }
  }

  return smthed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSplitSliver"
/*  splitsliver()    Remove a sliver by inserting a point.                     */
/*                                                                             */
/*  The 'remedge->tt' represents a sliver abcd, ab is the current edge which   */
/*  has a large dihedral angle (close to 180 degree).                          */
/* tetgenmesh::splitsliver() */
PetscErrorCode TetGenMeshSplitSliver(TetGenMesh *m, badface *remedge, List *tetlist, List *ceillist, PetscBool *isSplit)
{
  TetGenOpts    *b = m->b;
  triface starttet = {PETSC_NULL, 0, 0};
  face checkseg = {PETSC_NULL, 0};
  point newpt, pt[4];
  PetscBool isValid, remflag;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Let 'remedge->tt' be the edge [a, b]. */
  starttet = remedge->tt;

  /*  Go to the opposite edge [c, d]. */
  adjustedgering_triface(&starttet, CCW);
  enextfnextself(m, &starttet);
  enextself(&starttet);

  /*  Check if cd is a segment. */
  ierr = TetGenMeshTssPivot(m, &starttet, &checkseg);CHKERRQ(ierr);
  if (b->nobisect == 0) {
    if (checkseg.sh != m->dummysh) {
      int len;

      /*  cd is a segment. The seg will be split. */
      checkseg.shver = 0;
      pt[0] = sorg(&checkseg);
      pt[1] = sdest(&checkseg);
      ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
      ierr = TetGenMeshGetSplitPoint(m, pt[0], pt[1], PETSC_NULL, newpt);CHKERRQ(ierr);
      setpointtype(m, newpt, FREESEGVERTEX);
      setpoint2seg(m, newpt, sencode(&checkseg));
      /*  Insert p, this should always success. */
      ierr = TetGenMeshSstPivot(m, &checkseg, &starttet);CHKERRQ(ierr);
      ierr = TetGenMeshSplitTetEdge(m, newpt, &starttet, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
      /*  Collect the new tets connecting at p. */
      ierr = TetGenMeshSstPivot(m, &checkseg, &starttet);CHKERRQ(ierr);
      ierr = ListAppend(ceillist, &starttet, PETSC_NULL);CHKERRQ(ierr);
      ierr = TetGenMeshFormStarPolyhedron(m, newpt, ceillist, PETSC_NULL, PETSC_TRUE);CHKERRQ(ierr);
      ierr = TetGenMeshSetNewPointSize(m, newpt, pt[0], PETSC_NULL);CHKERRQ(ierr);
      if (m->steinerleft > 0) m->steinerleft--;
      /*  Smooth p. */
      ierr = TetGenMeshSmoothPoint(m, newpt, pt[0], pt[1], ceillist, PETSC_FALSE, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
      /*  Queue new slivers. */
      ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(ceillist, i, (void **) &starttet);CHKERRQ(ierr);
        ierr = TetGenMeshCheckTet4Opt(m, &starttet, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      }
      ierr = ListClear(ceillist);CHKERRQ(ierr);
      if (isSplit) {*isSplit = PETSC_TRUE;}
      PetscFunctionReturn(0);
    }
  }

  /*  Create the new point p (at the circumcenter of t). */
  ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
  pt[0] = org(&starttet);
  pt[1] = dest(&starttet);
  for (i = 0; i < 3; i++) {
    newpt[i] = 0.5 * (pt[0][i] + pt[1][i]);
  }
  setpointtype(m, newpt, FREEVOLVERTEX);

  /*  Form the Bowyer-Watson cavity of p. */
  remflag = PETSC_FALSE;
  infect(m, &starttet);
  ierr = ListAppend(tetlist, &starttet, PETSC_NULL);CHKERRQ(ierr);
  ierr = TetGenMeshFormBowatCavityQuad(m, newpt, tetlist, ceillist);CHKERRQ(ierr);
  ierr = TetGenMeshTrimBowatCavity(m, newpt, PETSC_NULL, 1, PETSC_NULL, PETSC_NULL, &tetlist, &ceillist, -1.0, &isValid);CHKERRQ(ierr);
  if (isValid) {
    PetscBool isSmooth;
    /*  Smooth p. */
    ierr = TetGenMeshSmoothPoint(m, newpt, PETSC_NULL, PETSC_NULL, ceillist, PETSC_FALSE, &remedge->key, &isSmooth);CHKERRQ(ierr);
    if (isSmooth) {
      int len;
      /*  Insert p. */
      ierr = TetGenMeshBowatInsertSite(m, newpt, PETSC_NULL, 1, PETSC_NULL, PETSC_NULL, &tetlist, &ceillist, PETSC_NULL, PETSC_NULL, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE);CHKERRQ(ierr);
      ierr = TetGenMeshSetNewPointSize(m, newpt, pt[0], PETSC_NULL);CHKERRQ(ierr);
      if (m->steinerleft > 0) m->steinerleft--;
      /*  Queue new slivers. */
      ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(ceillist, i, (void **) &starttet);CHKERRQ(ierr);
        ierr = TetGenMeshCheckTet4Opt(m, &starttet, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      }
      remflag = PETSC_TRUE;
    } /*  if (smoothpoint) */
  } /*  if (trimbowatcavity) */

  if (!remflag) {
    int len;
    /*  p is rejected for BC(p) is not valid. */
    ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
    /*  Uninfect tets of BC(p). */
    ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
      uninfect(m, &starttet);
    }
  }
  ierr = ListClear(tetlist);CHKERRQ(ierr);
  ierr = ListClear(ceillist);CHKERRQ(ierr);

  if (isSplit) {*isSplit = remflag;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTallSlivers"
/*  tallslivers()    Queue all the slivers in the mesh.                        */
/* tetgenmesh::tallslivers() */
PetscErrorCode TetGenMeshTallSlivers(TetGenMesh *m, PetscBool optflag)
{
  triface        tetloop = {PETSC_NULL, 0, 0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    if (optflag) {
      ierr = TetGenMeshCheckTet4Opt(m, &tetloop, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
    } else {
      ierr = TetGenMeshCheckTet4Ill(m, &tetloop, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshOptimize"
/*  Available mesh optimizing operations are: (1) multiple edge flips (3-to-2, */
/*  4-to-4, 5-to-6, etc), (2) free vertex deletion, (3) new vertex insertion.  */
/*  (1) is mandatory, while (2) and (3) are optionally.                        */
/*                                                                             */
/*  The variable 'b->optlevel' (set after '-s') determines the use of these    */
/*  operations. If it is: 0, do no optimization; 1, only do (1) operation; 2,  */
/*  do (1) and (2) operations; 3, do all operations. Deault, b->optlvel = 2.   */
/* tetgenmesh::optimizemesh2() */
PetscErrorCode TetGenMeshOptimize(TetGenMesh *m, PetscBool optflag)
{
  TetGenOpts    *b  = m->b;
  /*  Cosines of the six dihedral angles of the tet [a, b, c, d]. */
  /*    From cosdd[0] to cosdd[5]: ab, bc, ca, ad, bd, cd. */
  PetscReal      cosdd[6];
  List          *splittetlist, *tetlist, *ceillist;
  badface       *remtet, *newbadtet;
  PetscReal      objdihed, cosobjdihed;
  long           oldflipcount = 0, newflipcount = 0, oldpointcount, slivercount, optpasscount = 0;
  int            iter, len, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (optflag) {
    if (m->b_steinerflag) {
      /*  This routine is called from removesteiners2(); */
    } else {
      PetscInfo(b->in, "Optimizing mesh.\n");
    }
  } else {
    PetscInfo(b->in, "Repairing mesh.\n");
  }

  if (optflag) {
    if (m->b_steinerflag) {
      /*  This routine is called from removesteiners2(); */
      m->cosmaxdihed = cos(179.0 * PETSC_PI / 180.0);
      m->cosmindihed = cos(1.0 * PETSC_PI / 180.0);
      /*  The radian of the maximum dihedral angle. */
      /* maxdihed = 179.0 / 180.0 * PETSC_PI; */
    } else {
      m->cosmaxdihed = cos(b->maxdihedral * PETSC_PI / 180.0);
      m->cosmindihed = cos(b->mindihedral * PETSC_PI / 180.0);
      /*  The radian of the maximum dihedral angle. */
      /* maxdihed = b->maxdihedral / 180.0 * PETSC_PI; */
      /*  A sliver has an angle large than 'objdihed' will be split. */
      objdihed = b->maxdihedral + 5.0;
      if (objdihed < 175.0) objdihed = 175.0;
      objdihed = objdihed / 180.0 * PETSC_PI;
      cosobjdihed = cos(objdihed);
    }
  }

  /*  Initialize the pool of bad tets. */
  ierr = MemoryPoolCreate(sizeof(badface), ELEPERBLOCK, POINTER, 0, &m->badtetrahedrons);CHKERRQ(ierr);
  /*  Looking for non-optimal tets. */
  ierr = TetGenMeshTallSlivers(m, optflag);CHKERRQ(ierr);

  oldpointcount = m->points->items;
  m->opt_tet_peels = m->opt_face_flips = m->opt_edge_flips = 0l;
  m->smoothsegverts = 0l;

  if (optflag) {PetscInfo1(b->in, "  level = %d.\n", b->optlevel);}

  /*  Start the mesh optimization iteration. */
  do {
    if (optflag) {PetscInfo2(b->in, "  level = %d pass %d.\n", b->optlevel, optpasscount);}

    /*  Improve the mesh quality by flips. */
    iter = 0;
    do {
      oldflipcount = newflipcount;
      /*  Loop in the list of bad tets. */
      ierr = MemoryPoolTraversalInit(m->badtetrahedrons);CHKERRQ(ierr);
      ierr = TetGenMeshBadFaceTraverse(m, m->badtetrahedrons, &remtet);CHKERRQ(ierr);
      while(remtet) {
        if (!isdead_triface(&remtet->tt) && (org(&remtet->tt) == remtet->forg) &&
            (dest(&remtet->tt) == remtet->fdest) &&
            (apex(&remtet->tt) == remtet->fapex) &&
            (oppo(&remtet->tt) == remtet->foppo)) {
          PetscBool isRemoved;
          PetscInfo5(b->in, "    Repair tet (%d, %d, %d, %d) %g (degree).\n", pointmark(m, remtet->forg), pointmark(m, remtet->fdest),
                     pointmark(m, remtet->fapex), pointmark(m, remtet->foppo), acos(remtet->key) / PETSC_PI * 180.0);
          ierr = TetGenMeshRemoveEdge(m, remtet, optflag, &isRemoved);CHKERRQ(ierr);
          if (isRemoved) {
            /*  Remove the badtet from the list. */
            ierr = TetGenMeshBadFaceDealloc(m, m->badtetrahedrons, remtet);CHKERRQ(ierr);
          }
        } else {
          /*  Remove the badtet from the list. */
          ierr = TetGenMeshBadFaceDealloc(m, m->badtetrahedrons, remtet);CHKERRQ(ierr);
        }
        ierr = TetGenMeshBadFaceTraverse(m, m->badtetrahedrons, &remtet);CHKERRQ(ierr);
      }
      iter++;
      if (iter > 10) break; /*  Stop at 10th iterations. */
      /*  Count the total number of flips. */
      newflipcount = m->opt_tet_peels + m->opt_face_flips + m->opt_edge_flips;
      /*  Continue if there are bad tets and new flips. */
    } while ((m->badtetrahedrons->items > 0) && (newflipcount > oldflipcount));

    if (m->b_steinerflag) {
      /*  This routine was called from removesteiner2(). Do not repair the bad tets by splitting. */
      ierr = MemoryPoolRestart(m->badtetrahedrons);CHKERRQ(ierr);
    }

    if ((m->badtetrahedrons->items > 0l) && optflag  && (b->optlevel > 2)) {
      /*  Get a list of slivers and try to split them. */
      ierr = ListCreate(sizeof(badface), PETSC_NULL, 256, PETSC_DECIDE, &splittetlist);CHKERRQ(ierr);
      ierr = ListCreate(sizeof(triface), PETSC_NULL, 256, PETSC_DECIDE, &tetlist);CHKERRQ(ierr);
      ierr = ListCreate(sizeof(triface), PETSC_NULL, 256, PETSC_DECIDE, &ceillist);CHKERRQ(ierr);

      /*  Form a list of slivers to be split and clean the pool. */
      ierr = MemoryPoolTraversalInit(m->badtetrahedrons);CHKERRQ(ierr);
      ierr = TetGenMeshBadFaceTraverse(m, m->badtetrahedrons, &remtet);CHKERRQ(ierr);
      while(remtet) {
        ierr = ListAppend(splittetlist, remtet, PETSC_NULL);CHKERRQ(ierr);
        ierr = TetGenMeshBadFaceTraverse(m, m->badtetrahedrons, &remtet);CHKERRQ(ierr);
      }
      /*  Clean the pool of bad tets. */
      ierr = MemoryPoolRestart(m->badtetrahedrons);CHKERRQ(ierr);
      ierr = ListLength(splittetlist, &len);CHKERRQ(ierr);
      slivercount = 0;
      for(i = 0; i < len; i++) {
        badface remtet2;

        remtet = &remtet2;
        ierr = ListItem(splittetlist, i, (void **) &remtet2);CHKERRQ(ierr);
        if (!isdead_triface(&remtet->tt) && org(&remtet->tt) == remtet->forg &&
            dest(&remtet->tt) == remtet->fdest &&
            apex(&remtet->tt) == remtet->fapex &&
            oppo(&remtet->tt) == remtet->foppo) {
          /*  Calculate the six dihedral angles of this tet. */
          adjustedgering_triface(&remtet->tt, CCW);
          remtet->forg  = org(&remtet->tt);
          remtet->fdest = dest(&remtet->tt);
          remtet->fapex = apex(&remtet->tt);
          remtet->foppo = oppo(&remtet->tt);
          ierr = TetGenMeshTetAllDihedral(m, remtet->forg, remtet->fdest, remtet->fapex, remtet->foppo, cosdd, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
          /*  Is it a large angle? */
          if (cosdd[0] < cosobjdihed) {
            PetscBool couldSmooth;

            slivercount++;
            remtet->key = cosdd[0];
            PetscInfo5(b->in, "    Split tet (%d, %d, %d, %d) %g (degree).\n", pointmark(m, remtet->forg), pointmark(m, remtet->fdest),
                       pointmark(m, remtet->fapex), pointmark(m, remtet->foppo), acos(remtet->key) / PETSC_PI * 180.0);
            /*  Queue this tet. */
            ierr = MemoryPoolAlloc(m->badtetrahedrons, (void **) &newbadtet);CHKERRQ(ierr);
            *newbadtet = *remtet;
            /*  Try to remove this tet. */
            ierr = TetGenMeshSmoothSliver(m, remtet, tetlist, &couldSmooth);CHKERRQ(ierr);
            if (!couldSmooth) {
              ierr = TetGenMeshSplitSliver(m, remtet, tetlist, ceillist, PETSC_NULL);CHKERRQ(ierr);
            }
          }
        }
      } /*  i */

      ierr = ListDestroy(&splittetlist);CHKERRQ(ierr);
      ierr = ListDestroy(&tetlist);CHKERRQ(ierr);
      ierr = ListDestroy(&ceillist);CHKERRQ(ierr);
    }

    optpasscount++;
  } while ((m->badtetrahedrons->items > 0) && (optpasscount < b->optpasses));

  if (m->opt_tet_peels > 0l) {
    PetscInfo1(b->in, "  %ld tet removals.\n", m->opt_tet_peels);
  }
  if (m->opt_face_flips > 0l) {
    PetscInfo1(b->in, "  %ld face flips.\n", m->opt_face_flips);
  }
  if (m->opt_edge_flips > 0l) {
    PetscInfo1(b->in, "  %ld edge flips.\n", m->opt_edge_flips);
  }
  if ((m->points->items - oldpointcount) > 0l) {
    if (m->smoothsegverts > 0) {
      PetscInfo2(b->in, "  %ld point insertions (%d on segment)\n", m->points->items - oldpointcount, m->smoothsegverts);
    } else {
      PetscInfo1(b->in, "  %ld point insertions", m->points->items - oldpointcount);
    }
  }

  ierr = MemoryPoolDestroy(&m->badtetrahedrons);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*                                                                        //// */
/*                                                                        //// */
/*  optimize_cxx ///////////////////////////////////////////////////////////// */

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshJettisonNodes"
/*  Unused points are those input points which are outside the mesh domain or  */
/*  have no connection (isolated) to the mesh.  Duplicated points exist for    */
/*  example if the input PLC is read from a .stl mesh file (marked during the  */
/*  Delaunay tetrahedralization step. This routine remove these points from    */
/*  points list. All existing points are reindexed.                            */
/* tetgenmesh::jettisonnodes() */
PetscErrorCode TetGenMeshJettisonNodes(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  point          pointloop;
  int            oldidx = 0, newidx = 0, remcount = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  while(pointloop) {
    if ((pointtype(m, pointloop) == DUPLICATEDVERTEX) || (pointtype(m, pointloop) == UNUSEDVERTEX)) {
      /*  It is a duplicated point, delete it. */
      ierr = TetGenMeshPointDealloc(m, pointloop);CHKERRQ(ierr);
      remcount++;
    } else {
      /*  Re-index it. */
      setpointmark(m, pointloop, newidx + in->firstnumber);
      if (in->pointmarkerlist) {
        if (oldidx < in->numberofpoints) {
          /*  Re-index the point marker as well. */
          in->pointmarkerlist[newidx] = in->pointmarkerlist[oldidx];
        }
      }
      newidx++;
    }
    oldidx++;
    if (oldidx == in->numberofpoints) {
      /*  Update the numbe of input points (Because some were removed). */
      in->numberofpoints -= remcount;
      /*  Remember this number for output original input nodes. */
      m->jettisoninverts = remcount;
    }
    ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  }
  PetscInfo1(b->in, "  %d duplicated vertices have been removed.\n", m->dupverts);
  PetscInfo1(b->in, "  %d unused vertices have been removed.\n", m->unuverts);
  m->dupverts = 0;
  m->unuverts = 0;
  /*  The following line ensures that dead items in the pool of nodes cannot */
  /*    be allocated for the new created nodes. This ensures that the input */
  /*    nodes will occur earlier in the output files, and have lower indices. */
  m->points->deaditemstack = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshOutputNodes"
/* tetgenmesh::outnodes() */
PetscErrorCode TetGenMeshOutputNodes(TetGenMesh *m, PLC *out)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  shellface subptr;
  triface adjtet = {PETSC_NULL, 0, 0};
  face subloop = {PETSC_NULL, 0};
  point pointloop;
  point *extralist, ep[3];
  int nextras, bmark, shmark, marker;
  int coordindex, attribindex;
  int pointnumber, firstindex;
  int index, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Writing nodes.\n");
  nextras    = in->numberofpointattributes;
  bmark      = !b->nobound && in->pointmarkerlist;
  ierr = PetscMalloc(m->points->items * 3 * sizeof(PetscReal), &out->pointlist);CHKERRQ(ierr);
  if (nextras > 0) {
    ierr = PetscMalloc(m->points->items * nextras * sizeof(PetscReal), &out->pointattributelist);CHKERRQ(ierr);
  }
  if (bmark) {
    ierr = PetscMalloc(m->points->items * sizeof(int), &out->pointmarkerlist);CHKERRQ(ierr);
  }
  out->numberofpoints          = m->points->items;
  out->numberofpointattributes = nextras;
  marker      = 0;
  coordindex  = 0;
  attribindex = 0;
  if (bmark && (b->plc || b->refine)) {
    /*  Initialize the point2tet field of each point. */
    ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
    ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
    while(pointloop) {
      setpoint2tet(m, pointloop, (tetrahedron) PETSC_NULL);
      ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
    }
    /*  Make a map point-to-subface. Hence a boundary point will get the facet marker from that facet where it lies on. */
    ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
    while(subloop.sh) {
      subloop.shver = 0;
      /*  Check all three points of the subface. */
      for(i = 0; i < 3; i++) {
        pointloop = (point) subloop.sh[3 + i];
        setpoint2tet(m, pointloop, (tetrahedron) sencode(&subloop));
      }
      if (b->order == 2) {
        /*  '-o2' switch. Set markers for quadratic nodes of this subface. */
        stpivot(m, &subloop, &adjtet);
        if (adjtet.tet == m->dummytet) {
          sesymself(&subloop);
          stpivot(m, &subloop, &adjtet);
        }
        if (adjtet.tet == m->dummytet) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid adjacency");
        extralist = (point *) adjtet.tet[m->highorderindex];
        switch (adjtet.loc) {
        case 0:
          ep[0] = extralist[0];
          ep[1] = extralist[1];
          ep[2] = extralist[2];
          break;
        case 1:
          ep[0] = extralist[0];
          ep[1] = extralist[4];
          ep[2] = extralist[3];
          break;
        case 2:
          ep[0] = extralist[1];
          ep[1] = extralist[5];
          ep[2] = extralist[4];
          break;
        case 3:
          ep[0] = extralist[2];
          ep[1] = extralist[3];
          ep[2] = extralist[5];
          break;
        default: break;
        }
        for(i = 0; i < 3; i++) {
          setpoint2tet(m, ep[i], (tetrahedron) sencode(&subloop));
        }
      }
      ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
    }
  }
  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  firstindex  = b->zeroindex ? 0 : in->firstnumber; /* Determine the first index (0 or 1) */
  pointnumber = firstindex;
  index       = 0;
  while(pointloop) {
    if (bmark) {
      /*  Default the vertex has a zero marker. */
      marker = 0;
      /*  Is it an input vertex? */
      if (index < in->numberofpoints) {
        /*  Input point's marker is directly copied to output. */
        marker = in->pointmarkerlist[index];
      }
      /*  Is it a boundary vertex has marker zero? */
      if ((marker == 0) && (b->plc || b->refine)) {
        subptr = (shellface) point2tet(m, pointloop);
        if (subptr) {
          /*  Default a boundary vertex has marker 1. */
          marker = 1;
          if (in->facetmarkerlist) {
            /*  The vertex gets the marker from the facet it lies on. */
            sdecode(subptr, &subloop);
            shmark = shellmark(m, &subloop);
            marker = in->facetmarkerlist[shmark - 1];
          }
        }
      }
    }
    /*  x, y, and z coordinates. */
    out->pointlist[coordindex++] = pointloop[0];
    out->pointlist[coordindex++] = pointloop[1];
    out->pointlist[coordindex++] = pointloop[2];
    /*  Point attributes. */
    for(i = 0; i < nextras; i++) {
      /*  Output an attribute. */
      out->pointattributelist[attribindex++] = pointloop[3 + i];
    }
    if (bmark) {
      /*  Output the boundary marker. */
      out->pointmarkerlist[index] = marker;
    }
    ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
    pointnumber++;
    index++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshNumberEdges"
/* tetgenmesh::numberedges() */
PetscErrorCode TetGenMeshNumberEdges(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  triface tetloop = {PETSC_NULL, 0, 0}, worktet = {PETSC_NULL, 0, 0}, spintet = {PETSC_NULL, 0, 0};
  int hitbdry, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!b->plc && !b->refine) {
    /*  Using the Euler formula (V-E+F-T=1) to get the total number of edges. */
    long faces = (4l * m->tetrahedrons->items + m->hullsize) / 2l;
    m->meshedges = m->points->items + faces - m->tetrahedrons->items - 1l;
    PetscFunctionReturn(0);
  }

  m->meshedges = 0l;
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    /*  Count the number of Voronoi faces. Look at the six edges of each */
    /*    tetrahedron. Count the edge only if the tetrahedron's pointer is */
    /*    smaller than those of all other tetrahedra that share the edge. */
    worktet.tet = tetloop.tet;
    for(i = 0; i < 6; i++) {
      worktet.loc = edge2locver[i][0];
      worktet.ver = edge2locver[i][1];
      adjustedgering_triface(&worktet, CW);
      spintet = worktet;
      hitbdry = 0;
      while(hitbdry < 2) {
        if (fnextself(m, &spintet)) {
          if (apex(&spintet) == apex(&worktet)) break;
          if (spintet.tet < worktet.tet) break;
        } else {
          hitbdry++;
          if (hitbdry < 2) {
            esym(&worktet, &spintet);
            fnextself(m, &spintet); /*  In the same tet. */
	  }
        }
      }
      /*  Count this edge if no adjacent tets are smaller than this tet. */
      if (spintet.tet >= worktet.tet) {
        m->meshedges++;
      }
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshOutputElements"
/* tetgenmesh::outelements() */
PetscErrorCode TetGenMeshOutputElements(TetGenMesh *m, PLC *out)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  tetrahedron* tptr;
  triface worktet = {PETSC_NULL, 0, 0}, spintet = {PETSC_NULL, 0, 0};
  int *tlist = PETSC_NULL;
  PetscReal *talist = PETSC_NULL;
  int firstindex, shift;
  int pointindex;
  int attribindex;
  point p1, p2, p3, p4;
  point *extralist;
  int elementnumber;
  int eextras;
  int hitbdry, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Writing elements.\n");
  eextras = in->numberoftetrahedronattributes;
  ierr = PetscMalloc((m->tetrahedrons->items * (b->order == 1 ? 4 : 10)) * sizeof(int), &out->tetrahedronlist);CHKERRQ(ierr);
  /*  Allocate memory for output tetrahedron attributes if necessary. */
  if (eextras > 0) {
    ierr = PetscMalloc(m->tetrahedrons->items * eextras * sizeof(PetscReal), &out->tetrahedronattributelist);CHKERRQ(ierr);
  }
  out->numberoftetrahedra = m->tetrahedrons->items;
  out->numberofcorners    = b->order == 1 ? 4 : 10;
  out->numberoftetrahedronattributes = eextras;
  tlist  = out->tetrahedronlist;
  talist = out->tetrahedronattributelist;
  pointindex  = 0;
  attribindex = 0;
  /*  Determine the first index (0 or 1). */
  firstindex = b->zeroindex ? 0 : in->firstnumber;
  shift      = 0; /*  Default no shiftment. */
  if ((in->firstnumber == 1) && (firstindex == 0)) {
    shift = 1; /*  Shift the output indices by 1. */
  }
  /*  Count the total edge numbers. */
  m->meshedges = 0l;
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tptr);CHKERRQ(ierr);
  elementnumber = firstindex; /*  in->firstnumber; */
  while(tptr) {
    if (b->noelewritten == 2) {
      /*  Reverse the orientation, such that TetGenOrient3D() > 0. */
      p1 = (point) tptr[5];
      p2 = (point) tptr[4];
    } else {
      p1 = (point) tptr[4];
      p2 = (point) tptr[5];
    }
    p3 = (point) tptr[6];
    p4 = (point) tptr[7];
    tlist[pointindex++] = pointmark(m, p1) - shift;
    tlist[pointindex++] = pointmark(m, p2) - shift;
    tlist[pointindex++] = pointmark(m, p3) - shift;
    tlist[pointindex++] = pointmark(m, p4) - shift;
    if (b->order == 2) {
      extralist = (point *) tptr[m->highorderindex];
      tlist[pointindex++] = pointmark(m, extralist[0]) - shift;
      tlist[pointindex++] = pointmark(m, extralist[1]) - shift;
      tlist[pointindex++] = pointmark(m, extralist[2]) - shift;
      tlist[pointindex++] = pointmark(m, extralist[3]) - shift;
      tlist[pointindex++] = pointmark(m, extralist[4]) - shift;
      tlist[pointindex++] = pointmark(m, extralist[5]) - shift;
    }
    for(i = 0; i < eextras; i++) {
      talist[attribindex++] = elemattribute(m, tptr, i);
    }
    if (b->neighout) {
      /*  Remember the index of this element. */
      setelemmarker(m, tptr, elementnumber);
    }
    /*  Count the number of Voronoi faces. Look at the six edges of each */
    /*    tetrahedron. Count the edge only if the tetrahedron's pointer is */
    /*    smaller than those of all other tetrahedra that share the edge. */
    worktet.tet = tptr;
    for(i = 0; i < 6; i++) {
      worktet.loc = edge2locver[i][0];
      worktet.ver = edge2locver[i][1];
      adjustedgering_triface(&worktet, CW);
      spintet = worktet;
      hitbdry = 0;
      while(hitbdry < 2) {
        if (fnextself(m, &spintet)) {
          if (apex(&spintet) == apex(&worktet)) break;
          if (spintet.tet < worktet.tet) break;
        } else {
          hitbdry++;
          if (hitbdry < 2) {
            esym(&worktet, &spintet);
            fnextself(m, &spintet); /*  In the same tet. */
	  }
        }
      }
      /*  Count this edge if no adjacent tets are smaller than this tet. */
      if (spintet.tet >= worktet.tet) {
        m->meshedges++;
      }
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tptr);CHKERRQ(ierr);
    elementnumber++;
  }
  if (b->neighout) {
    /*  Set the outside element marker. */
    setelemmarker(m, m->dummytet, -1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshOutputSubfaces"
/*  The boundary faces are exist in 'subfaces'. For listing triangle vertices  */
/*  in the same sense for all triangles in the mesh, the direction determined  */
/*  by right-hand rule is pointer to the inside of the volume.                 */
/* tetgenmesh::outsubfaces() */
PetscErrorCode TetGenMeshOutputSubfaces(TetGenMesh *m, PLC *out)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  int *elist  = PETSC_NULL;
  int *emlist = PETSC_NULL;
  int index = 0, index1 = 0, index2 = 0;
  triface abuttingtet = {PETSC_NULL, 0, 0};
  face faceloop = {PETSC_NULL, 0};
  point torg, tdest, tapex;
  int bmark, faceid = 0, marker = 0;
  int firstindex, shift;
  int neigh1 = 0, neigh2 = 0;
  int facenumber;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "Writing faces.\n");
  bmark = !b->nobound && in->facetmarkerlist;
  /*  Allocate memory for 'trifacelist'. */
  ierr = PetscMalloc(m->subfaces->items * 3 * sizeof(int), &out->trifacelist);CHKERRQ(ierr);
  if (bmark) {
    ierr = PetscMalloc(m->subfaces->items * sizeof(int), &out->trifacemarkerlist);CHKERRQ(ierr);
  }
  if (b->neighout > 1) { /* '-nn' switch. */
    ierr = PetscMalloc(m->subfaces->items * 2 * sizeof(int), &out->adjtetlist);CHKERRQ(ierr);
  }
  out->numberoftrifaces = m->subfaces->items;
  elist  = out->trifacelist;
  emlist = out->trifacemarkerlist;
  /*  Determine the first index (0 or 1). */
  firstindex = b->zeroindex ? 0 : in->firstnumber;
  shift = 0; /*  Default no shiftment. */
  if ((in->firstnumber == 1) && (firstindex == 0)) {
    shift = 1; /*  Shift the output indices by 1. */
  }
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &faceloop.sh);CHKERRQ(ierr);
  facenumber = firstindex;
  while(faceloop.sh) {
    stpivot(m, &faceloop, &abuttingtet);
    if (abuttingtet.tet == m->dummytet) {
      sesymself(&faceloop);
      stpivot(m, &faceloop, &abuttingtet);
    }
    if (abuttingtet.tet != m->dummytet) {
      /*  If there is a tetrahedron containing this subface, orient it so */
      /*    that the normal of this face points to inside of the volume by */
      /*    right-hand rule. */
      adjustedgering_triface(&abuttingtet, CCW);
      torg  = org(&abuttingtet);
      tdest = dest(&abuttingtet);
      tapex = apex(&abuttingtet);
    } else {
      /*  This may happen when only a surface mesh be generated. */
      torg  = sorg(&faceloop);
      tdest = sdest(&faceloop);
      tapex = sapex(&faceloop);
    }
    if (bmark) {
      faceid = shellmark(m, &faceloop) - 1;
      marker = in->facetmarkerlist[faceid];
    }
    if (b->neighout > 1) {
      /*  '-nn' switch. Output adjacent tets indices. */
      neigh1 = -1;
      stpivot(m, &faceloop, &abuttingtet);
      if (abuttingtet.tet != m->dummytet) {
        neigh1 = getelemmarker(m, abuttingtet.tet);
      }
      neigh2 = -1;
      sesymself(&faceloop);
      stpivot(m, &faceloop, &abuttingtet);
      if (abuttingtet.tet != m->dummytet) {
        neigh2 = getelemmarker(m, abuttingtet.tet);
      }
    }
    /*  Output three vertices of this face; */
    elist[index++] = pointmark(m, torg) - shift;
    elist[index++] = pointmark(m, tdest) - shift;
    elist[index++] = pointmark(m, tapex) - shift;
    if (bmark) {
      emlist[index1++] = marker;
    }
    if (b->neighout > 1) {
      out->adjtetlist[index2++] = neigh1;
      out->adjtetlist[index2++] = neigh2;
    }
    facenumber++;
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &faceloop.sh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenTetrahedralize"
/*
  TetGenTetrahedralize - Interface for using TetGen's library to generate Delaunay tetrahedralizations, constrained
                         Delaunay tetrahedralizations, quality tetrahedral meshes.

  Input Parameters:
+ t   - The TetGenOpts object with all the option information
- in  - A PLC you want to tetrahedralize, or a previously generated tetrahedral mesh you want to refine

  Output Parameter:
. out - A PLC for storing the generated tetrahedral mesh

  Note:
  We have omitted the bgmin parameter, which would contain a background mesh which defines a mesh size "distruction"
  function, since I cannot figure out what it would do.

  This is roughly the sequence of actions:
$ - Initialize constants and parse the command line.
$ - Read the vertices from a file and either
$   - tetrahedralize them (no -r), or
$   - read an old mesh from files and reconstruct it (-r).
$ - Insert the PLC segments and facets (-p).
$ - Read the holes (-p), regional attributes (-pA), and regional volume
$   constraints (-pa).  Carve the holes and concavities, and spread the
$   regional attributes and volume constraints.
$ - Enforce the constraints on minimum quality bound (-q) and maximum
$   volume (-a). Also enforce the conforming Delaunay property (-q and -a).
$ - Promote the mesh's linear tetrahedra to higher order elements (-o).
$ - Write the output files and print the statistics.
$ - Check the consistency and Delaunay property of the mesh (-C).
*/
PetscErrorCode TetGenTetrahedralize(TetGenOpts *b, PLC *in, PLC *out)
{
  TetGenMesh    *m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TetGenMeshCreate(&m);CHKERRQ(ierr);
  m->b           = b;
  m->in          = in;
  m->macheps     = TetGenExactInit();
  m->steinerleft = b->steiner;
  if (b->metric) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m->bgm = new tetgenmesh();
    m->bgm->b = b;
    m->bgm->in = bgmin;
    m->bgm->macheps = TetGenExactInit();
#endif
  }
  ierr = TetGenMeshInitializePools(m);CHKERRQ(ierr);
  ierr = TetGenMeshTransferNodes(m);CHKERRQ(ierr);

  /*  PetscLogEventBegin(DelaunayOrReconstruct) */
  if (b->refine) {
    ierr = TetGenMeshReconstructMesh(m, PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = TetGenMeshDelaunizeVertices(m);CHKERRQ(ierr);
    if (!m->hullsize) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The input point set does not span a 3D subspace.");
  }
  /*  PetscLogEventEnd(DelaunayOrReconstruct) */

  /*  PetscLogEventBegin(BackgroundMeshReconstruct) */
  if (b->metric) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    if (bgmin) {
      m->bgm->initializepools();
      m->bgm->transfernodes();
      m->bgm->reconstructmesh();
    } else {
      m->bgm->in = in;
      m->bgm->initializepools();
      m->duplicatebgmesh();
    }
#endif
  }
  /*  PetscLogEventEnd(BackgroundMeshReconstruct) */

  /*  PetscLogEventBegin(BdRecoveryOrIntersection) */
  if (b->useshelles && !b->refine) {
    ierr = TetGenMeshMeshSurface(m, PETSC_NULL);CHKERRQ(ierr);
    if (b->diagnose != 1) {
      ierr = TetGenMeshMarkAcuteVertices(m, 60.0);CHKERRQ(ierr);
      ierr = TetGenMeshFormSkeleton(m);CHKERRQ(ierr);
    } else {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
      ierr = TetGenMeshDetectInterfaces(m);CHKERRQ(ierr);
#endif
    }
  }
  /*  PetscLogEventEnd(BdRecoveryOrIntersection) */

  /*  PetscLogEventBegin(Holes) */
  if (b->plc && !(b->diagnose == 1)) {
    ierr = TetGenMeshCarveHoles(m);CHKERRQ(ierr);
  }
  /*  PetscLogEventEnd(Holes) */

  /*  PetscLogEventBegin(Repair) */
  if ((b->plc || b->refine) && !(b->diagnose == 1)) {
    ierr = TetGenMeshOptimize(m, PETSC_FALSE);CHKERRQ(ierr);
  }
  /*  PetscLogEventEnd(Repair) */

  /*  PetscLogEventBegin(SteinerRemoval) */
  if ((b->plc && b->nobisect) && !(b->diagnose == 1)) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m.removesteiners2();
#endif
  }
  /*  PetscLogEventEnd(SteinerRemoval) */

  /*  PetscLogEventBegin(ConstrainedPoints) */
  if (b->insertaddpoints) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    if (addin && addin->numberofpoints > 0) {
      m.insertconstrainedpoints(addin);
    }
#endif
  }
  /*  PetscLogEventEnd(ConstrainedPoints) */

  /*  PetscLogEventBegin(SizeInterpolation) */
  if (b->metric) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m.interpolatesizemap();
#endif
  }
  /*  PetscLogEventEnd(SizeInterpolation) */

#if 0 /* Removed by TetGen */
  /*  PetscLogEventBegin(MeshCoarsen) */
  if (b->coarse) {
    m.removesteiners2(PETSC_TRUE);
  }
  /*  PetscLogEventEnd(MeshCoarsen) */
#endif

  /*  PetscLogEventBegin(Quality) */
  if (b->quality) {
    ierr = TetGenMeshEnforceQuality(m);CHKERRQ(ierr);
  }
  /*  PetscLogEventEnd(Quality) */

  /*  PetscLogEventBegin(Optimize) */
  if (b->quality && (b->optlevel > 0)) {
    ierr = TetGenMeshOptimize(m, PETSC_TRUE);CHKERRQ(ierr);
  }
  /*  PetscLogEventEnd(Optimize) */

  if (!b->nojettison && ((m->dupverts > 0) || (m->unuverts > 0) || (b->refine && (in->numberofcorners == 10)))) {
    ierr = TetGenMeshJettisonNodes(m);CHKERRQ(ierr);
  }

  if (b->order > 1) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m.highorder();
#endif
  }

  out->firstnumber = in->firstnumber;
  out->mesh_dim    = in->mesh_dim;

  if (b->nonodewritten || b->noiterationnum) {
    PetscInfo(b->in, "NOT writing a .node file.\n");
  } else {
    if (b->diagnose == 1) {
      if (m->subfaces->items > 0l) {
        /*  Only output when self-intersecting faces exist. */
        ierr = TetGenMeshOutputNodes(m, out);CHKERRQ(ierr);
      }
    } else {
      ierr = TetGenMeshOutputNodes(m, out);CHKERRQ(ierr);
      if (b->metric) { /* if (b->quality && b->metric) { */
#if 1
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
        m.outmetrics(out);
#endif
      }
    }
  }

  if (b->noelewritten == 1) {
    PetscInfo(b->in, "NOT writing an .ele file.\n");
    ierr = TetGenMeshNumberEdges(m);CHKERRQ(ierr);
  } else {
    if (!(b->diagnose == 1)) {
      if (m->tetrahedrons->items > 0l) {
        ierr = TetGenMeshOutputElements(m, out);CHKERRQ(ierr);
      }
    }
  }

  if (b->nofacewritten) {
    PetscInfo(b->in, "NOT writing an .face file.\n");
  } else {
    if (b->facesout) {
      if (m->tetrahedrons->items > 0l) {
#if 1
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
        m.outfaces(out);  /*  Output all faces. */
#endif
      }
    } else {
      if (b->diagnose == 1) {
        if (m->subfaces->items > 0l) {
          ierr = TetGenMeshOutputSubfaces(m, out);CHKERRQ(ierr); /* Only output self-intersecting faces. */
        }
      } else if (b->plc || b->refine) {
        if (m->subfaces->items > 0l) {
          ierr = TetGenMeshOutputSubfaces(m, out);CHKERRQ(ierr); /* Output boundary faces. */
        }
      } else {
        if (m->tetrahedrons->items > 0l) {
#if 1
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
          m.outhullfaces(out); /*  Output convex hull faces. */
#endif
        }
      }
    }
  }

#if 0 /* Removed by TetGen */
  if (m.checkpbcs) {
    m.outpbcnodes(out);
  }
#endif

#if 0 /* No output */
  if (b->edgesout) {
    if (b->edgesout > 1) {
      m.outedges(out); /*  -ee, output all mesh edges */
    } else {
      m.outsubsegments(out); /*  -e, only output subsegments. */
    }
  }

  if (!out && b->plc &&
      ((b->object == TETGEN_OBJECT_OFF) ||
       (b->object == TETGEN_OBJECT_PLY) ||
       (b->object == TETGEN_OBJECT_STL))) {
    m.outsmesh(b->outfilename);
  }

  if (!out && b->meditview) {
    m.outmesh2medit(b->outfilename);
  }

  if (!out && b->gidview) {
    m.outmesh2gid(b->outfilename);
  }

  if (!out && b->geomview) {
    m.outmesh2off(b->outfilename);
  }

  if (!out && b->vtkview) {
    m.outmesh2vtk(b->outfilename);
  }

  if (b->neighout) {
    m.outneighbors(out);
  }

  if (b->voroout) {
    m.outvoronoi(out);
  }
#endif

  if (b->docheck) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m.checkmesh();
    if (m.checksubfaces) {
      m.checkshells();
    }
    if (b->docheck > 1) {
      if (m.checkdelaunay(0.0, PETSC_NULL) > 0) {
        assert(0);
      }
      if (b->docheck > 2) {
        if (b->quality || b->refine) {
          m.checkconforming();
        }
      }
    }
#endif
  }

  /* Make into a viewer */
  if (!b->quiet) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m.statistics();
#endif
  }
  if (b->metric) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    delete m.bgm;
#endif
  }
  ierr = TetGenMeshDestroy(&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*================================ End Converted TetGen Functions ================================*/

#undef __FUNCT__
#define __FUNCT__ "TetGenCheckOpts"
PetscErrorCode TetGenCheckOpts(TetGenOpts *t)
{
  PetscFunctionBegin;
  t->plc        = t->plc || t->diagnose;
  t->useshelles = t->plc || t->refine || t->coarse || t->quality;
  t->goodratio  = t->minratio;
  t->goodratio *= t->goodratio;
  if (t->plc && t->refine) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Switch -r cannot use together with -p.");
  if (t->refine && (t->plc || t->noiterationnum)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Switches -p, -d, and -I cannot use together with -r.\n");
  if (t->diagnose && (t->quality || t->insertaddpoints || (t->order == 2) || t->neighout || t->docheck)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Switches -q, -i, -o2, -n, and -C cannot use together with -d.\n");

  /* Be careful not to allocate space for element area constraints that will never be assigned any value (other than the default -1.0). */
  if (!t->refine && !t->plc) {
    t->varvolume = 0;
  }
  /* Be careful not to add an extra attribute to each element unless the input supports it (PLC in, but not refining a preexisting mesh). */
  if (t->refine || !t->plc) {
    t->regionattrib = 0;
  }
  /* If '-a' or '-aa' is in use, enable '-q' option too. */
  if (t->fixedvolume || t->varvolume) {
    if (t->quality == 0) {
      t->quality = 1;
    }
  }
  /* Calculate the goodangle for testing bad subfaces. */
  t->goodangle = cos(t->minangle * PETSC_PI / 180.0);
  t->goodangle *= t->goodangle;
  PetscFunctionReturn(0);
}
