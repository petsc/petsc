static char help[] = "Run C version of TetGen to construct and refine a mesh\n\n";

#include <petscdmcomplex.h>
#include <private/compleximpl.h>

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscInt      debug;             /* The debugging level */
  PetscLogEvent createMeshEvent;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscBool     interpolate;       /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;   /* The largest allowable cell volume */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_FALSE;
  options->refinementLimit = 0.0;

  ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMESH");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex62.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex62.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex62.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex62.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",          DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

/*================================ Start Converted TetGen Objects ================================*/
// Geometric predicates                                                      //
//                                                                           //
// Return one of the values +1, 0, and -1 on basic geometric questions such  //
// as the orientation of point sets, in-circle, and in-sphere tests.  They   //
// are basic units for implmenting geometric algorithms.  TetGen uses two 3D //
// geometric predicates: the orientation and in-sphere tests.                //
//                                                                           //
// Orientation test:  let a, b, c be a sequence of 3 non-collinear points in //
// R^3.  They defines a unique hypeplane H.  Let H+ and H- be the two spaces //
// separated by H, which are defined as follows (using the left-hand rule):  //
// make a fist using your left hand in such a way that your fingers follow   //
// the order of a, b and c, then your thumb is pointing to H+.  Given any    //
// point d in R^3, the orientation test returns +1 if d lies in H+, -1 if d  //
// lies in H-, or 0 if d lies on H.                                          //
//                                                                           //
// In-sphere test:  let a, b, c, d be 4 non-coplanar points in R^3.  They    //
// defines a unique circumsphere S.  Given any point e in R^3, the in-sphere //
// test returns +1 if e lies inside S, or -1 if e lies outside S, or 0 if e  //
// lies on S.                                                                //
//                                                                           //
// The following routines use arbitrary precision floating-point arithmetic. //
// They are provided by J. R. Schewchuk in public domain (http://www.cs.cmu. //
// edu/~quake/robust.html). The source code are in "predicates.cxx".         //
PetscReal exactinit();
PetscReal orient3d(PetscReal *pa, PetscReal *pb, PetscReal *pc, PetscReal *pd);
PetscReal insphere(PetscReal *pa, PetscReal *pb, PetscReal *pc, PetscReal *pd, PetscReal *pe);

// Labels that signify whether a record consists primarily of pointers
//   or of floating-point words.  Used for data alignment.
typedef enum {POINTER, FLOATINGPOINT} wordtype;

// Labels that signify the type of a vertex.
typedef enum {UNUSEDVERTEX, DUPLICATEDVERTEX, NACUTEVERTEX, ACUTEVERTEX, FREESEGVERTEX, FREESUBVERTEX, FREEVOLVERTEX, DEADVERTEX = -32768} verttype;

// Labels that signify the result of triangle-triangle intersection test.
typedef enum {DISJOINT, INTERSECT, SHAREVERTEX, SHAREEDGE, SHAREFACE, TOUCHEDGE, TOUCHFACE, INTERVERT, INTEREDGE, INTERFACE, INTERTET,
              TRIEDGEINT, EDGETRIINT, COLLISIONFACE, INTERSUBSEG, INTERSUBFACE, BELOWHULL2} interresult;

// Labels that signify the result of point location.
typedef enum {INTETRAHEDRON, ONFACE, ONEDGE, ONVERTEX, OUTSIDE, ENCSEGMENT} locateresult;

// Labels that signify the result of direction finding.
typedef enum {ACROSSEDGE, ACROSSFACE, LEFTCOLLINEAR, RIGHTCOLLINEAR, TOPCOLLINEAR, BELOWHULL} finddirectionresult;

// Labels that signify the type of a subface/subsegment.
typedef enum {NSHARP, SHARP} shestype;

// For efficiency, a variety of data structures are allocated in bulk.
//   The following constants determine how many of each structure is allocated at once.
enum {VERPERBLOCK = 4092, SUBPERBLOCK = 4092, ELEPERBLOCK = 8188};

// Labels that signify two edge rings of a triangle (see Muecke's thesis).
enum {CCW = 0, CW = 1};

/* Replaces tetgenbehavior x*/
typedef enum {NONE, NODES, POLY, OFF, PLY, STL, MEDIT, VTK, MESH} objecttype;
typedef struct {
  DM  in; /* Eventually make this a PetscObject */
  int plc;                                                 // '-p' switch, 0.
  int quality;                                             // '-q' switch, 0.
  int refine;                                              // '-r' switch, 0.
  int coarse;                                              // '-R' switch, 0.
  int metric;                                              // '-m' switch, 0.
  int varvolume;                            // '-a' switch without number, 0.
  int fixedvolume;                             // '-a' switch with number, 0.
  int insertaddpoints;                                     // '-i' switch, 0.
  int regionattrib;                                        // '-A' switch, 0.
  int conformdel;                                          // '-D' switch, 0.
  int diagnose;                                            // '-d' switch, 0.
  int zeroindex;                                           // '-z' switch, 0.
  int btree;                                                        // -u, 1.
  int max_btreenode_size;                            // number after -u, 100.
  int optlevel;                     // number specified after '-s' switch, 3.
  int optpasses;                   // number specified after '-ss' switch, 3.
  int order;                // element order, specified after '-o' switch, 1.
  int facesout;                                            // '-f' switch, 0.
  int edgesout;                                            // '-e' switch, 0.
  int neighout;                                            // '-n' switch, 0.
  int voroout;                                             // '-v',switch, 0.
  int meditview;                                           // '-g' switch, 0.
  int gidview;                                             // '-G' switch, 0.
  int geomview;                                            // '-O' switch, 0.
  int vtkview;                                             // '-K' switch, 0.
  int nobound;                                             // '-B' switch, 0.
  int nonodewritten;                                       // '-N' switch, 0.
  int noelewritten;                                        // '-E' switch, 0.
  int nofacewritten;                                       // '-F' switch, 0.
  int noiterationnum;                                      // '-I' switch, 0.
  int nomerge;                                             // '-M',switch, 0.
  int nobisect;             // count of how often '-Y' switch is selected, 0.
  int noflip;                        // do not perform flips. '-X' switch. 0.
  int nojettison;        // do not jettison redundants nodes. '-J' switch. 0.
  int steiner;                                // number after '-S' switch. 0.
  int fliprepair;                                          // '-X' switch, 1.
  int offcenter;                                           // '-R' switch, 0.
  int docheck;                                             // '-C' switch, 0.
  int quiet;                                               // '-Q' switch, 0.
  int verbose;              // count of how often '-V' switch is selected, 0.
  int useshelles;               // '-p', '-r', '-q', '-d', or '-R' switch, 0.
  int maxflipedgelinksize;        // The maximum flippable edge link size 10.
  PetscReal minratio;                            // number after '-q' switch, 2.0.
  PetscReal goodratio;                  // number calculated from 'minratio', 0.0.
  PetscReal minangle;                                // minimum angle bound, 20.0.
  PetscReal goodangle;                         // cosine squared of minangle, 0.0.
  PetscReal maxvolume;                          // number after '-a' switch, -1.0.
  PetscReal mindihedral;                        // number after '-qq' switch, 5.0.
  PetscReal maxdihedral;                     // number after '-qqq' switch, 165.0.
  PetscReal alpha1;                          // number after '-m' switch, sqrt(2).
  PetscReal alpha2;                             // number after '-mm' switch, 1.0.
  PetscReal alpha3;                            // number after '-mmm' switch, 0.6.
  PetscReal epsilon;                          // number after '-T' switch, 1.0e-8.
  PetscReal epsilon2;                        // number after '-TT' switch, 1.0e-5.
  objecttype object;            // determined by -p, or -r switch. NONE.
} TetGenOpts;

// A callback function for mesh refinement.
typedef PetscBool (*TetSizeFunc)(PetscReal*, PetscReal*, PetscReal*, PetscReal*, PetscReal*, PetscReal);

// The polygon data structure.  A "polygon" describes a simple polygon
//   (no holes). It is not necessarily convex.  Each polygon contains a
//   number of corners (points) and the same number of sides (edges).
// Note that the points of the polygon must be given in either counter-
//   clockwise or clockwise order and they form a ring, so every two
//   consective points forms an edge of the polygon.
typedef struct {
  int *vertexlist;
  int numberofvertices;
} polygon;

// The facet data structure.  A "facet" describes a facet. Each facet is
//   a polygonal region possibly with holes, edges, and points in it.
typedef struct {
  polygon *polygonlist;
  int numberofpolygons;
  PetscReal *holelist;
  int numberofholes;
} facet;

// A 'voroedge' is an edge of the Voronoi diagram. It corresponds to a
//   Delaunay face.  Each voroedge is either a line segment connecting
//   two Voronoi vertices or a ray starting from a Voronoi vertex to an
//   "infinite vertex".  'v1' and 'v2' are two indices pointing to the
//   list of Voronoi vertices. 'v1' must be non-negative, while 'v2' may
//   be -1 if it is a ray, in this case, the unit normal of this ray is
//   given in 'vnormal'.
typedef struct {
  int v1, v2;
  PetscReal vnormal[3];
} voroedge;

// A 'vorofacet' is an facet of the Voronoi diagram. It corresponds to a
//   Delaunay edge.  Each Voronoi facet is a convex polygon formed by a
//   list of Voronoi edges, it may not be closed.  'c1' and 'c2' are two
//   indices pointing into the list of Voronoi cells, i.e., the two cells
//   share this facet.  'elist' is an array of indices pointing into the
//   list of Voronoi edges, 'elist[0]' saves the number of Voronoi edges
//   (including rays) of this facet.
typedef struct {
  int c1, c2;
  int *elist;
} vorofacet;

// The periodic boundary condition group data structure.  A "pbcgroup"
//   contains the definition of a pbc and the list of pbc point pairs.
//   'fmark1' and 'fmark2' are the facetmarkers of the two pbc facets f1
//   and f2, respectively. 'transmat' is the transformation matrix which
//   maps a point in f1 into f2.  An array of pbc point pairs are saved
//   in 'pointpairlist'. The first point pair is at indices [0] and [1],
//   followed by remaining pairs. Two integers per pair.
typedef struct {
  int fmark1, fmark2;
  PetscReal transmat[4][4];
  int numberofpointpairs;
  int *pointpairlist;
} pbcgroup;

/* This replaces tetgenio */
typedef struct {
  // Items are numbered starting from 'firstnumber' (0 or 1), default is 0.
  int firstnumber;

  // Dimension of the mesh (2 or 3), default is 3.
  int mesh_dim;

  // Does the lines in .node file contain index or not, default is TRUE.
  PetscBool useindex;

  // 'pointlist':  An array of point coordinates.  The first point's x
  //   coordinate is at index [0] and its y coordinate at index [1], its
  //   z coordinate is at index [2], followed by the coordinates of the
  //   remaining points.  Each point occupies three PetscReals.
  // 'pointattributelist':  An array of point attributes.  Each point's
  //   attributes occupy 'numberofpointattributes' PetscReals.
  // 'pointmtrlist': An array of metric tensors at points. Each point's
  //   tensor occupies 'numberofpointmtr' PetscReals.
  // `pointmarkerlist':  An array of point markers; one int per point.
  PetscReal *pointlist;
  PetscReal *pointattributelist;
  PetscReal *pointmtrlist;
  int *pointmarkerlist;
  int numberofpoints;
  int numberofpointattributes;
  int numberofpointmtrs;

  // `elementlist':  An array of element (triangle or tetrahedron) corners.
  //   The first element's first corner is at index [0], followed by its
  //   other corners in counterclockwise order, followed by any other
  //   nodes if the element represents a nonlinear element.  Each element
  //   occupies `numberofcorners' ints.
  // `elementattributelist':  An array of element attributes.  Each
  //   element's attributes occupy `numberofelementattributes' PetscReals.
  // `elementconstraintlist':  An array of constraints, i.e. triangle's
  //   area or tetrahedron's volume; one PetscReal per element.  Input only.
  // `neighborlist':  An array of element neighbors; 3 or 4 ints per
  //   element.  Output only.
  int *tetrahedronlist;
  PetscReal *tetrahedronattributelist;
  PetscReal *tetrahedronvolumelist;
  int *neighborlist;
  int numberoftetrahedra;
  int numberofcorners;
  int numberoftetrahedronattributes;

  // `facetlist':  An array of facets.  Each entry is a structure of facet.
  // `facetmarkerlist':  An array of facet markers; one int per facet.
  facet *facetlist;
  int *facetmarkerlist;
  int numberoffacets;

  // `holelist':  An array of holes.  The first hole's x, y and z
  //   coordinates  are at indices [0], [1] and [2], followed by the
  //   remaining holes. Three PetscReals per hole.
  PetscReal *holelist;
  int numberofholes;

  // `regionlist': An array of regional attributes and volume constraints.
  //   The first constraint's x, y and z coordinates are at indices [0],
  //   [1] and [2], followed by the regional attribute at index [3], foll-
  //   owed by the maximum volume at index [4]. Five PetscReals per constraint.
  // Note that each regional attribute is used only if you select the `A'
  //   switch, and each volume constraint is used only if you select the
  //   `a' switch (with no number following).
  PetscReal *regionlist;
  int numberofregions;

  // `facetconstraintlist': An array of facet maximal area constraints.
  //   Two PetscReals per constraint. The first (at index [0]) is the facet
  //   marker (cast it to int), the second (at index [1]) is its maximum
  //   area bound.
  PetscReal *facetconstraintlist;
  int numberoffacetconstraints;

  // `segmentconstraintlist': An array of segment max. length constraints.
  //   Three PetscReals per constraint. The first two (at indcies [0] and [1])
  //   are the indices of the endpoints of the segment, the third (at index
  //   [2]) is its maximum length bound.
  PetscReal *segmentconstraintlist;
  int numberofsegmentconstraints;

  // 'pbcgrouplist':  An array of periodic boundary condition groups.
  pbcgroup *pbcgrouplist;
  int numberofpbcgroups;

  // `trifacelist':  An array of triangular face endpoints.  The first
  //   face's endpoints are at indices [0], [1] and [2], followed by the
  //   remaining faces.  Three ints per face.
  // `adjtetlist':  An array of adjacent tetrahedra to the faces of
  //   trifacelist. Each face has at most two adjacent tets, the first
  //   face's adjacent tets are at [0], [1]. Two ints per face. A '-1'
  //   indicates outside (no adj. tet). This list is output when '-nn'
  //   switch is used.
  // `trifacemarkerlist':  An array of face markers; one int per face.
  int *trifacelist;
  int *adjtetlist;
  int *trifacemarkerlist;
  int numberoftrifaces;

  // `edgelist':  An array of edge endpoints.  The first edge's endpoints
  //   are at indices [0] and [1], followed by the remaining edges.  Two
  //   ints per edge.
  // `edgemarkerlist':  An array of edge markers; one int per edge.
  int *edgelist;
  int *edgemarkerlist;
  int numberofedges;

  // 'vpointlist':  An array of Voronoi vertex coordinates (like pointlist).
  // 'vedgelist':  An array of Voronoi edges.  Each entry is a 'voroedge'.
  // 'vfacetlist':  An array of Voronoi facets. Each entry is a 'vorofacet'.
  // 'vcelllist':  An array of Voronoi cells.  Each entry is an array of
  //   indices pointing into 'vfacetlist'. The 0th entry is used to store
  //   the length of this array.
  PetscReal *vpointlist;
  voroedge *vedgelist;
  vorofacet *vfacetlist;
  int **vcelllist;
  int numberofvpoints;
  int numberofvedges;
  int numberofvfacets;
  int numberofvcells;

  // A callback function.
  TetSizeFunc tetunsuitable;
} PLC;

// Arraypool                                                                 //
//                                                                           //
// Each arraypool contains an array of pointers to a number of blocks.  Each //
// block contains the same fixed number of objects.  Each index of the array //
// addesses a particular object in the pool.  The most significant bits add- //
// ress the index of the block containing the object. The less significant   //
// bits address this object within the block.                                //
//                                                                           //
// 'objectbytes' is the size of one object in blocks; 'log2objectsperblock'  //
// is the base-2 logarithm of 'objectsperblock'; 'objects' counts the number //
// of allocated objects; 'totalmemory' is the totoal memorypool in bytes.    //
typedef struct {
  int objectbytes;
  int objectsperblock;
  int log2objectsperblock;
  int toparraylen;
  char **toparray;
  long objects;
  unsigned long totalmemory;
} ArrayPool;

// fastlookup() -- A fast, unsafe operation. Return the pointer to the object
//   with a given index.  Note: The object's block must have been allocated,
//   i.e., by the function newindex().
#define fastlookup(pool, index) \
  (void *) ((pool)->toparray[(index) >> (pool)->log2objectsperblock] + \
            ((index) & ((pool)->objectsperblock - 1)) * (pool)->objectbytes)

// Memorypool                                                                //
//                                                                           //
// A type used to allocate memory.                                           //
//                                                                           //
// firstblock is the first block of items. nowblock is the block from which  //
//   items are currently being allocated. nextitem points to the next slab   //
//   of free memory for an item. deaditemstack is the head of a linked list  //
//   (stack) of deallocated items that can be recycled.  unallocateditems is //
//   the number of items that remain to be allocated from nowblock.          //
//                                                                           //
// Traversal is the process of walking through the entire list of items, and //
//   is separate from allocation.  Note that a traversal will visit items on //
//   the "deaditemstack" stack as well as live items.  pathblock points to   //
//   the block currently being traversed.  pathitem points to the next item  //
//   to be traversed.  pathitemsleft is the number of items that remain to   //
//   be traversed in pathblock.                                              //
//                                                                           //
// itemwordtype is set to POINTER or FLOATINGPOINT, and is used to suggest   //
//   what sort of word the record is primarily made up of.  alignbytes       //
//   determines how new records should be aligned in memory.  itembytes and  //
//   itemwords are the length of a record in bytes (after rounding up) and   //
//   words.  itemsperblock is the number of items allocated at once in a     //
//   single block.  items is the number of currently allocated items.        //
//   maxitems is the maximum number of items that have been allocated at     //
//   once; it is the current number of items plus the number of records kept //
//   on deaditemstack.                                                       //
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

// Queue                                                                     //
//                                                                           //
// A 'queue' is a FIFO data structure.                                       //
typedef struct {
  MemoryPool *mp;
  void      **head, **tail;
  int         linkitembytes;
  int         linkitems; // Not counting 'head' and 'tail'.
} Queue;

// A function: int cmp(const T &, const T &),  is said to realize a
//   linear order on the type T if there is a linear order <= on T such
//   that for all x and y in T satisfy the following relation:
//                 -1  if x < y.
//   comp(x, y) =   0  if x is equivalent to y.
//                 +1  if x > y.
// A 'compfunc' is a pointer to a linear-order function.
typedef int (*compfunc) (const void *, const void *);

// An array of items with automatically reallocation of memory.              //
//                                                                           //
// 'base' is the starting address of the array.  'itembytes' is the size of  //
//   each item in byte.                                                      //
//                                                                           //
// 'items' is the number of items stored in list.  'maxitems' indicates how  //
//   many items can be stored in this list. 'expandsize' is the increasing   //
//   size (items) when the list is full.                                     //
//                                                                           //
// The index of list always starts from zero, i.e., for a list L contains    //
//   n elements, the first element is L[0], and the last element is L[n-1].  //
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
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Fast lookup tables for mesh manipulation primitives.                      //
//                                                                           //
// Mesh manipulation primitives (given below) are basic operations on mesh   //
// data structures. They answer basic queries on mesh handles, such as "what //
// is the origin (or destination, or apex) of the face?", "what is the next  //
// (or previous) edge in the edge ring?", and "what is the next face in the  //
// face ring?", and so on.                                                   //
//                                                                           //
// The implementation of teste basic queries can take advangtage of the fact //
// that the mesh data structures additionally store geometric informations.  //
// For example, we have ordered the 4 vertices (from 0 to 3) and the 4 faces //
// (from 0 to 3) of a tetrahedron,  and for each face of the tetrahedron, a  //
// sequence of vertices has stipulated,  therefore the origin of any face of //
// the tetrahedron can be quickly determined by a table 'locver2org', which  //
// takes the index of the face and the edge version as inputs.  A list of    //
// fast lookup tables are defined below. They're just like global variables. //
// These tables are initialized at the runtime.                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
// Table 've' takes an edge version as input, returns the next edge version in the same edge ring.
//   For enext() primitive, uses 'ver' as the index.
static int ve[6] = {2, 5, 4, 1, 0, 3};

// Tables 'vo', 'vd' and 'va' take an edge version, return the positions of
//   the origin, destination and apex in the triangle.
//   For org(), dest() and apex() primitives, uses 'ver' as the index.
static int vo[6] = { 0, 1, 1, 2, 2, 0 };
static int vd[6] = { 1, 0, 2, 1, 0, 2 };
static int va[6] = { 2, 2, 0, 0, 1, 1 };

// The following tables are for tetrahedron primitives (operate on trifaces).
//   For org(), dest() and apex() primitives, uses 'loc' as the first index and 'ver' as the second index.
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

// For oppo() primitives, uses 'loc' as the index.
static int loc2oppo[4] = {3, 2, 0, 1};

// For fnext() primitives, uses 'loc' as the first index and 'ver' as
//   the second index,  returns an array containing a new 'loc' and a
//   new 'ver'. Note: Only valid for 'ver' equals one of {0, 2, 4}.
static int locver2nextf[4][6][2] = {{{1, 5}, {-1, -1}, {2, 5}, {-1, -1}, {3, 5}, {-1, -1}},
                                    {{3, 3}, {-1, -1}, {2, 1}, {-1, -1}, {0, 1}, {-1, -1}},
                                    {{1, 3}, {-1, -1}, {3, 1}, {-1, -1}, {0, 3}, {-1, -1}},
                                    {{2, 3}, {-1, -1}, {1, 1}, {-1, -1}, {0, 5}, {-1, -1}}};

// The edge number (from 0 to 5) of a tet is defined as follows:
//   0 - (v0, v1), 1 - (v1, v2), 2 - (v2, v0)
//   3 - (v3, v0), 4 - (v3, v1), 5 - (v3, v2).
static int locver2edge[4][6] = {{0, 0, 1, 1, 2, 2},
                                {3, 3, 4, 4, 0, 0},
                                {4, 4, 5, 5, 1, 1},
                                {5, 5, 3, 3, 2, 2}};
static int edge2locver[6][2] = {{0, 0},  // 0  v0 -> v1 (a -> b)
                                {0, 2},  // 1  v1 -> v2 (b -> c)
                                {0, 4},  // 2  v2 -> v0 (c -> a)
                                {1, 0},  // 3  v0 -> v3 (a -> d)
                                {1, 2},  // 4  v1 -> v3 (b -> d
                                {2, 2}}; // 5  v2 -> v3 (c -> d);

// The map from a given face ('loc') to the other three faces in the tet.
//   and the map from a given face's edge ('loc', 'ver') to other two
//   faces in the tet opposite to this edge. (used in speeding the Bowyer-
//   Watson cavity construction).
static int locpivot[4][3] = {{1, 2, 3},
                             {0, 2, 3},
                             {0, 1, 3},
                             {0, 1, 2}};
static int locverpivot[4][6][2] = {{{2, 3}, {2, 3}, {1, 3}, {1, 3}, {1, 2}, {1, 2}},
                                   {{0, 2}, {0, 2}, {0, 3}, {0, 3}, {2, 3}, {2, 3}},
                                   {{0, 3}, {0, 3}, {0, 1}, {0, 1}, {1, 3}, {1, 3}},
                                   {{0, 1}, {0, 1}, {0, 2}, {0, 2}, {1, 2}, {1, 2}}};

// For enumerating three edges of a triangle.
static int plus1mod3[3]  = {1, 2, 0};
static int minus1mod3[3] = {2, 0, 1};

// A 'triface' represents a face of a tetrahedron and an oriented edge of    //
// the face simultaneously.  It has a pointer 'tet' to a tetrahedron, an     //
// integer 'loc' (range from 0 to 3) as the face index, and an integer 'ver' //
// (range from 0 to 5) as the edge version. A face of the tetrahedron can be //
// uniquely determined by the pair (tet, loc), and an oriented edge of this  //
// face can be uniquly determined by the triple (tet, loc, ver).  Therefore, //
// different usages of one triface are possible.  If we only use the pair    //
// (tet, loc), it refers to a face, and if we add the 'ver' additionally to  //
// the pair, it is an oriented edge of this face.                            //
typedef struct {
  tetrahedron* tet;
  int loc, ver;
} triface;

// A 'face' represents a subface and an oriented edge of it simultaneously.  //
// It has a pointer 'sh' to a subface, an integer 'shver'(range from 0 to 5) //
// as the edge version.  The pair (sh, shver) determines a unique oriented   //
// edge of this subface.  A 'face' is also used to represent a subsegment,   //
// in this case, 'sh' points to the subsegment, and 'shver' indicates the    //
// one of two orientations of this subsegment, hence, it only can be 0 or 1. //
typedef struct {
  shellface *sh;
  int shver;
} face;

// A multiple usages structure. Despite of its name, a 'badface' can be used //
// to represent the following objects:                                       //
//   - a face of a tetrahedron which is (possibly) non-Delaunay;             //
//   - an encroached subsegment or subface;                                  //
//   - a bad-quality tetrahedron, i.e, has too large radius-edge ratio;      //
//   - a sliver, i.e., has good radius-edge ratio but nearly zero volume;    //
//   - a degenerate tetrahedron (see routine checkdegetet()).                //
//   - a recently flipped face (saved for undoing the flip later).           //
//                                                                           //
// It has the following fields:  'tt' holds a tetrahedron; 'ss' holds a sub- //
// segment or subface; 'cent' is the circumcent of 'tt' or 'ss', 'key' is a  //
// special value depending on the use, it can be either the square of the    //
// radius-edge ratio of 'tt' or the flipped type of 'tt';  'forg', 'fdest',  //
// 'fapex', and 'foppo' are vertices saved for checking the object in 'tt'   //
// or 'ss' is still the same when it was stored; 'noppo' is the fifth vertex //
// of a degenerate point set.  'previtem' and 'nextitem' implement a double  //
// link for managing many basfaces.                                          //
typedef struct _s_badface {
    triface tt;
    face ss;
    PetscReal key;
    PetscReal cent[3];
    point forg, fdest, fapex, foppo;
    point noppo;
    struct _s_badface *previtem, *nextitem;
} badface;

// A pbcdata stores data of a periodic boundary condition defined on a pair  //
// of facets or segments. Let f1 and f2 define a pbcgroup. 'fmark' saves the //
// facet markers of f1 and f2;  'ss' contains two subfaces belong to f1 and  //
// f2, respectively.  Let s1 and s2 define a segment pbcgroup. 'segid' are   //
// the segment ids of s1 and s2; 'ss' contains two segments belong to s1 and //
// s2, respectively. 'transmat' are two transformation matrices. transmat[0] //
// transforms a point of f1 (or s1) into a point of f2 (or s2),  transmat[1] //
// does the inverse.                                                         //
typedef struct {
  int fmark[2];
  int segid[2];
  face ss[2];
  PetscReal transmat[2][4][4];
} pbcdata;

typedef struct {
  // Pointer to the input data (a set of nodes, a PLC, or a mesh).
  PLC *in;

  // Pointer to the options (and filenames).
  TetGenOpts *b;

  // Pointer to a background mesh (contains size specification map).
  // tetgenmesh *bgm;

  // Variables used to allocate and access memory for tetrahedra, subfaces
  //   subsegments, points, encroached subfaces, encroached subsegments,
  //   bad-quality tetrahedra, and so on.
  MemoryPool *tetrahedrons;
  MemoryPool *subfaces;
  MemoryPool *subsegs;
  MemoryPool *points;
  MemoryPool *badsubsegs;
  MemoryPool *badsubfaces;
  MemoryPool *badtetrahedrons;
  MemoryPool *tet2segpool, *tet2subpool;

  // Pointer to the 'tetrahedron' that occupies all of "outer space".
  tetrahedron *dummytet;
  tetrahedron *dummytetbase; // Keep base address so we can free it later.

  // Pointer to the omnipresent subface.  Referenced by any tetrahedron,
  //   or subface that isn't connected to a subface at that location.
  shellface *dummysh;
  shellface *dummyshbase;    // Keep base address so we can free it later.

  // Entry to find the binary tree nodes (-u option).
  ArrayPool *btreenode_list;
  // The maximum size of a btree node (number after -u option) is
  int max_btreenode_size; // <= b->max_btreenode_size.
  // The maximum btree depth (for bookkeeping).
  int max_btree_depth;

  // Arrays used by Bowyer-Watson algorithm.
  ArrayPool *cavetetlist, *cavebdrylist, *caveoldtetlist;
  ArrayPool *caveshlist, *caveshbdlist;
  // Stacks used by the boundary recovery algorithm.
  ArrayPool *subsegstack, *subfacstack;

  // Two handles used in constrained facet recovery.
  triface firsttopface, firstbotface;

  // An array for registering elementary flips.
  ArrayPool *elemfliplist;

  // An array of fixed edges for facet recovering by flips.
  ArrayPool *fixededgelist;

  // A point above the plane in which the facet currently being used lies.
  //   It is used as a reference point for orient3d().
  point *facetabovepointarray, abovepoint, dummypoint;

  // Array (size = numberoftetrahedra * 6) for storing high-order nodes of
  //   tetrahedra (only used when -o2 switch is selected).
  point *highordertable;

  // Arrays for storing and searching pbc data. 'subpbcgrouptable', (size
  //   is numberofpbcgroups) for pbcgroup of subfaces. 'segpbcgrouptable',
  //   a list for pbcgroup of segments. Because a segment can have several
  //   pbcgroup incident on it, its size is unknown on input, it will be
  //   found in 'createsegpbcgrouptable()'.
  pbcdata *subpbcgrouptable;
  List *segpbcgrouptable;
  // A map for searching the pbcgroups of a given segment. 'idx2segpglist'
  //   (size = number of input segments + 1), and 'segpglist'.
  int *idx2segpglist, *segpglist;

  // Queues that maintain the bad (badly-shaped or too large) tetrahedra.
  //   The tails are pointers to the pointers that have to be filled in to
  //   enqueue an item.  The queues are ordered from 63 (highest priority)
  //   to 0 (lowest priority).
  badface *subquefront[3], **subquetail[3];
  badface *tetquefront[64], *tetquetail[64];
  int nextnonemptyq[64];
  int firstnonemptyq, recentq;

  // Pointer to a recently visited tetrahedron. Improves point location
  //   if proximate points are inserted sequentially.
  triface recenttet;

  PetscReal xmax, xmin, ymax, ymin, zmax, zmin;         // Bounding box of points.
  PetscReal longest;                          // The longest possible edge length.
  PetscReal lengthlimit;                     // The limiting length of a new edge.
  long hullsize;                           // Number of faces of convex hull.
  long insegments;                               // Number of input segments.
  long meshedges;                             // Number of output mesh edges.
  int steinerleft;                  // Number of Steiner points not yet used.
  int sizeoftensor;                     // Number of PetscReals per metric tensor.
  int pointmtrindex;           // Index to find the metric tensor of a point.
  int point2simindex;         // Index to find a simplex adjacent to a point.
  int pointmarkindex;            // Index to find boundary marker of a point.
  int point2pbcptindex;              // Index to find a pbc point to a point.
  int highorderindex;    // Index to find extra nodes for highorder elements.
  int elemattribindex;          // Index to find attributes of a tetrahedron.
  int volumeboundindex;       // Index to find volume bound of a tetrahedron.
  int elemmarkerindex;              // Index to find marker of a tetrahedron.
  int shmarkindex;             // Index to find boundary marker of a subface.
  int areaboundindex;               // Index to find area bound of a subface.
  int checksubfaces;                   // Are there subfaces in the mesh yet?
  int checksubsegs;                     // Are there subsegs in the mesh yet?
  int checkpbcs;                   // Are there periodic boundary conditions?
  int varconstraint;     // Are there variant (node, seg, facet) constraints?
  int nonconvex;                               // Is current mesh non-convex?
  int dupverts;                             // Are there duplicated vertices?
  int unuverts;                                 // Are there unused vertices?
  int relverts;                          // The number of relocated vertices.
  int suprelverts;            // The number of suppressed relocated vertices.
  int collapverts;             // The number of collapsed relocated vertices.
  int unsupverts;                     // The number of unsuppressed vertices.
  int smoothsegverts;                     // The number of smoothed vertices.
  int jettisoninverts;            // The number of jettisoned input vertices.
  long samples;               // Number of random samples for point location.
  unsigned long randomseed;                    // Current random number seed.
  PetscReal macheps;                                       // The machine epsilon.
  PetscReal cosmaxdihed, cosmindihed;    // The cosine values of max/min dihedral.
  PetscReal minfaceang, minfacetdihed;     // The minimum input (dihedral) angles.
  int maxcavfaces, maxcavverts;            // The size of the largest cavity.
  PetscBool b_steinerflag;

  // Algorithm statistical counters.
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

  long abovecount;                     // Number of abovepoints calculation.
  long bowatvolcount, bowatsubcount, bowatsegcount;       // Bowyer-Watsons.
  long updvolcount, updsubcount, updsegcount;   // Bow-Wat cavities updates.
  long failvolcount, failsubcount, failsegcount;           // Bow-Wat fails.
  long outbowatcircumcount;    // Number of circumcenters outside Bowat-cav.
  long r1count, r2count, r3count;        // Numbers of edge splitting rules.
  long cdtenforcesegpts;                // Number of CDT enforcement points.
  long rejsegpts, rejsubpts, rejtetpts;        // Number of rejected points.
  long optcount[10];            // Numbers of various optimizing operations.
  long flip23s, flip32s, flip22s, flip44s;     // Number of flips performed.
} TetGenMesh;

/*================================= End Converted TetGen Objects =================================*/

/* Forward Declarations */
extern PetscErrorCode MemoryPoolCreate(int, int, wordtype, int, MemoryPool **);
extern PetscErrorCode MemoryPoolAlloc(MemoryPool *, void **);
extern PetscErrorCode MemoryPoolDealloc(MemoryPool *, void *);
extern PetscErrorCode MemoryPoolDestroy(MemoryPool **);
extern PetscErrorCode ArrayPoolDestroy(ArrayPool **);
extern PetscErrorCode ListDestroy(List **);
extern PetscErrorCode TetGenMeshPointTraverse(TetGenMesh *, point *);
extern PetscErrorCode TetGenMeshShellFaceTraverse(TetGenMesh *, MemoryPool *, shellface **);
extern PetscErrorCode TetGenMeshTetrahedronTraverse(TetGenMesh *, tetrahedron **);
extern PetscErrorCode TetGenMeshGetNextSFace(TetGenMesh *, face *, face *);
extern PetscErrorCode TetGenMeshGetFacetAbovePoint(TetGenMesh *, face *);
extern PetscErrorCode TetGenMeshSplitSubEdge(TetGenMesh *, point, face *, ArrayPool *, ArrayPool *);
extern PetscErrorCode TetGenMeshSInsertVertex(TetGenMesh *, point, face *, face *, PetscBool, PetscBool, locateresult *);
extern PetscErrorCode TetGenMeshInsertVertexBW(TetGenMesh *, point, triface *, PetscBool, PetscBool, PetscBool, PetscBool, locateresult *);
extern PetscErrorCode TetGenMeshJettisonNodes(TetGenMesh *);

/*=========================== Start Converted TetGen Inline Functions ============================*/
// Some macros for convenience
#define Div2  >> 1
#define Mod2  & 01
// NOTE: These bit operators should only be used in macros below.
// Get orient(Range from 0 to 2) from face version(Range from 0 to 5).
#define Orient(V)   ((V) Div2)
// Determine edge ring(0 or 1) from face version(Range from 0 to 5).
#define EdgeRing(V) ((V) Mod2)

/*** Begin of primitives for points ***/
PETSC_STATIC_INLINE int pointmark(TetGenMesh *m, point pt) {
  return ((int *) (pt))[m->pointmarkindex];
}

PETSC_STATIC_INLINE void setpointmark(TetGenMesh *m, point pt, int value) {
  ((int *) (pt))[m->pointmarkindex] = value;
}

// These two primitives set and read the type of the point.
// The last significant bit of this integer is used by pinfect/puninfect.
PETSC_STATIC_INLINE verttype pointtype(TetGenMesh *m, point pt) {
  return (verttype) (((int *) (pt))[m->pointmarkindex + 1] >> (int) 1);
}

PETSC_STATIC_INLINE void setpointtype(TetGenMesh *m, point pt, verttype value) {
  ((int *) (pt))[m->pointmarkindex + 1] = ((int) value << 1) + (((int *) (pt))[m->pointmarkindex + 1] & (int) 1);
}

// pinfect(), puninfect(), pinfected() -- primitives to flag or unflag a point
//   The last bit of the integer '[pointindex+1]' is flaged.
PETSC_STATIC_INLINE void pinfect(TetGenMesh *m, point pt) {
  ((int *) (pt))[m->pointmarkindex + 1] |= (int) 1;
}

PETSC_STATIC_INLINE void puninfect(TetGenMesh *m, point pt) {
  ((int *) (pt))[m->pointmarkindex + 1] &= ~(int) 1;
}

PETSC_STATIC_INLINE PetscBool pinfected(TetGenMesh *m, point pt) {
  return (((int *) (pt))[m->pointmarkindex + 1] & (int) 1) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

// These following primitives set and read a pointer to a tetrahedron
//   a subface/subsegment, a point, or a tet of background mesh.
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

// These primitives set and read a pointer to its pbc point.
PETSC_STATIC_INLINE point point2pbcpt(TetGenMesh *m, point pt) {
  return (point) ((tetrahedron *) (pt))[m->point2pbcptindex];
}

PETSC_STATIC_INLINE void setpoint2pbcpt(TetGenMesh *m, point pt, point value) {
  ((tetrahedron *) (pt))[m->point2pbcptindex] = (tetrahedron) value;
}
/*** End of primitives for points ***/

/*** Begin of primitives for tetrahedra ***/
// Each tetrahedron contains four pointers to its neighboring tetrahedra,
//   with face indices.  To save memory, both information are kept in a
//   single pointer. To make this possible, all tetrahedra are aligned to
//   eight-byte boundaries, so that the last three bits of each pointer are
//   zeros. A face index (in the range 0 to 3) is compressed into the last
//   two bits of each pointer by the function 'encode()'.  The function
//   'decode()' decodes a pointer, extracting a face index and a pointer to
//   the beginning of a tetrahedron.
PETSC_STATIC_INLINE void decode(tetrahedron ptr, triface *t) {
  t->loc = (int) ((PETSC_UINTPTR_T) (ptr) & (PETSC_UINTPTR_T) 3);
  t->tet = (tetrahedron *) ((PETSC_UINTPTR_T) (ptr) & ~(PETSC_UINTPTR_T) 7);
}

PETSC_STATIC_INLINE tetrahedron encode(triface *t) {
  return (tetrahedron) ((PETSC_UINTPTR_T) t->tet | (PETSC_UINTPTR_T) t->loc);
}

// sym() finds the abutting tetrahedron on the same face.
PETSC_STATIC_INLINE void sym(triface *t1, triface *t2) {
  tetrahedron ptr = t1->tet[t1->loc];
  decode(ptr, t2);
}

PETSC_STATIC_INLINE void symself(triface *t) {
  tetrahedron ptr = t->tet[t->loc];
  decode(ptr, t);
}

// Bond two tetrahedra together at their faces.
PETSC_STATIC_INLINE void bond(TetGenMesh *m, triface *t1, triface *t2) {
  t1->tet[t1->loc] = encode(t2);
  t2->tet[t2->loc] = encode(t1);
}

// Dissolve a bond (from one side).  Note that the other tetrahedron will
//   still think it is connected to this tetrahedron.  Usually, however,
//   the other tetrahedron is being deleted entirely, or bonded to another
//   tetrahedron, so it doesn't matter.
PETSC_STATIC_INLINE void dissolve(TetGenMesh *m, triface *t) {
  t->tet[t->loc] = (tetrahedron) m->dummytet;
}

// These primitives determine or set the origin, destination, apex or
//   opposition of a tetrahedron with respect to 'loc' and 'ver'.
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

// These primitives were drived from Mucke's triangle-edge data structure
//   to change face-edge relation in a tetrahedron (esym, enext and enext2)
//   or between two tetrahedra (fnext).

// If e0 = e(i, j), e1 = e(j, i), that is e0 and e1 are the two directions
//   of the same undirected edge of a face. e0.sym() = e1 and vice versa.
PETSC_STATIC_INLINE void esym(triface *t1, triface *t2) {
  t2->tet = t1->tet;
  t2->loc = t1->loc;
  t2->ver = t1->ver + (EdgeRing(t1->ver) ? -1 : 1);
}

PETSC_STATIC_INLINE void esymself(triface *t) {
  t->ver += (EdgeRing(t->ver) ? -1 : 1);
}

// If e0 and e1 are both in the same edge ring of a face, e1 = e0.enext().
PETSC_STATIC_INLINE void enext(triface *t1, triface *t2) {
  t2->tet = t1->tet;
  t2->loc = t1->loc;
  t2->ver = ve[t1->ver];
}

PETSC_STATIC_INLINE void enextself(triface *t) {
  t->ver = ve[t->ver];
}

// enext2() is equal to e2 = e0.enext().enext()
PETSC_STATIC_INLINE void enext2(triface *t1, triface *t2) {
  t2->tet = t1->tet;
  t2->loc = t1->loc;
  t2->ver = ve[ve[t1->ver]];
}

PETSC_STATIC_INLINE void enext2self(triface *t) {
  t->ver = ve[ve[t->ver]];
}

// If f0 and f1 are both in the same face ring of a face, f1 = f0.fnext().
//   If f1 exists, return true. Otherwise, return false, i.e., f0 is a boundary or hull face.
PETSC_STATIC_INLINE PetscBool fnext(TetGenMesh *m, triface *t1, triface *t2)
{
  // Get the next face.
  t2->loc = locver2nextf[t1->loc][t1->ver][0];
  // Is the next face in the same tet?
  if (t2->loc != -1) {
    // It's in the same tet. Get the edge version.
    t2->ver = locver2nextf[t1->loc][t1->ver][1];
    t2->tet = t1->tet;
  } else {
    // The next face is in the neigbhour of 't1'.
    sym(t1, t2);
    if (t2->tet != m->dummytet) {
      // Find the corresponding edge in t2.
      point torg;
      int tloc, tver, i;
      t2->ver = 0;
      torg = org(t1);
      for (i = 0; (i < 3) && (org(t2) != torg); i++) {
        enextself(t2);
      }
      // Go to the next face in t2.
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
  triface t2;

  // Get the next face.
  t2.loc = locver2nextf[t1->loc][t1->ver][0];
  // Is the next face in the same tet?
  if (t2.loc != -1) {
    // It's in the same tet. Get the edge version.
    t2.ver = locver2nextf[t1->loc][t1->ver][1];
    t1->loc = t2.loc;
    t1->ver = t2.ver;
  } else {
    // The next face is in the neigbhour of 't1'.
    sym(t1, &t2);
    if (t2.tet != m->dummytet) {
      // Find the corresponding edge in t2.
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

// Given a face t1, find the face f2 in the adjacent tet. If t2 is not
//   a dummytet, then t1 and t2 refer to the same edge. Moreover, t2's
//   edge must be in 0th edge ring, e.g., t2->ver is one of {0, 2, 4}.
//   No matter what edge version t1 is.

PETSC_STATIC_INLINE void symedge(TetGenMesh *m, triface *t1, triface *t2)
{
  decode(t1->tet[t1->loc], t2);
  if (t2->tet != m->dummytet) {
    // Search the edge of t1 in t2.
    point tapex = apex(t1);
    if ((point) (t2->tet[locver2apex[t2->loc][0] + 4]) == tapex) {
      t2->ver = 0;
    } else if ((point) (t2->tet[locver2apex[t2->loc][2] + 4]) == tapex) {
      t2->ver = 2;
    } else {
      //assert((point) (t2->tet[locver2apex[t2->loc][4] + 4]) == tapex);
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
    // Search the edge of t1 in t2.
    if ((point) (t->tet[locver2apex[t->loc][0] + 4]) == tapex) {
      t->ver = 0;
    } else if ((point) (t->tet[locver2apex[t->loc][2] + 4]) == tapex) {
      t->ver = 2;
    } else {
      //assert((point) (t->tet[locver2apex[t->loc][4] + 4]) == tapex);
      t->ver = 4;
    }
  }
}

// Given a face t1, find the next face t2 in the face ring, t1 and t2
//   are in two different tetrahedra. If the next face is a hull face, t2 is dummytet.
PETSC_STATIC_INLINE void tfnext(TetGenMesh *m, triface *t1, triface *t2)
{
  int *iptr;

  if ((t1->ver & 1) == 0) {
    t2->tet = t1->tet;
    iptr = locver2nextf[t1->loc][t1->ver];
    t2->loc = iptr[0];
    t2->ver = iptr[1];
    symedgeself(m, t2);  // t2->tet may be dummytet.
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
    symedgeself(m, t); // t->tet may be dummytet.
  } else {
    symedgeself(m, t);
    if (t->tet != m->dummytet) {
      iptr = locver2nextf[t->loc][t->ver];
      t->loc = iptr[0];
      t->ver = iptr[1];
    }
  }
}

// enextfnext() and enext2fnext() are combination primitives of enext(), enext2() and fnext().
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

// Check or set a tetrahedron's attributes.
PETSC_STATIC_INLINE PetscReal elemattribute(TetGenMesh *m, tetrahedron *ptr, int attnum) {
  return ((PetscReal *) (ptr))[m->elemattribindex + attnum];
}

PETSC_STATIC_INLINE void setelemattribute(TetGenMesh *m, tetrahedron *ptr, int attnum, PetscReal value){
  ((PetscReal *) (ptr))[m->elemattribindex + attnum] = value;
}

// Check or set a tetrahedron's maximum volume bound.
PETSC_STATIC_INLINE PetscReal volumebound(TetGenMesh *m, tetrahedron *ptr) {
  return ((PetscReal *) (ptr))[m->volumeboundindex];
}

PETSC_STATIC_INLINE void setvolumebound(TetGenMesh *m, tetrahedron* ptr, PetscReal value) {
  ((PetscReal *) (ptr))[m->volumeboundindex] = value;
}

// Check or set a tetrahedron's marker.
PETSC_STATIC_INLINE int getelemmarker(TetGenMesh *m, tetrahedron* ptr) {
  return ((int *) (ptr))[m->elemmarkerindex];
}

PETSC_STATIC_INLINE void setelemmarker(TetGenMesh *m, tetrahedron* ptr, int value) {
  ((int *) (ptr))[m->elemmarkerindex] = value;
}

// infect(), infected(), uninfect() -- primitives to flag or unflag a
//   tetrahedron. The last bit of the element marker is flagged (1)
//   or unflagged (0).
PETSC_STATIC_INLINE void infect(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] |= (int) 1;
}

PETSC_STATIC_INLINE void uninfect(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] &= ~(int) 1;
}

// Test a tetrahedron for viral infection.
PETSC_STATIC_INLINE PetscBool infected(TetGenMesh *m, triface *t) {
  return (((int *) (t->tet))[m->elemmarkerindex] & (int) 1) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

// marktest(), marktested(), unmarktest() -- primitives to flag or unflag a
//   tetrahedron.  The last second bit of the element marker is marked (1)
//   or unmarked (0).
// One needs them in forming Bowyer-Watson cavity, to mark a tetrahedron if
//   it has been checked (for Delaunay case) so later check can be avoided.
PETSC_STATIC_INLINE void marktest(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] |= (int) 2;
}

PETSC_STATIC_INLINE void unmarktest(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] &= ~(int) 2;
}

PETSC_STATIC_INLINE PetscBool marktested(TetGenMesh *m, triface *t) {
  return (((int *) (t->tet))[m->elemmarkerindex] & (int) 2) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

// markface(), unmarkface(), facemarked() -- primitives to flag or unflag a
//   face of a tetrahedron.  From the last 3rd to 6th bits are used for face markers, e.g., the last third bit corresponds to loc = 0.
// One use of the face marker is in flip algorithm. Each queued face (check for locally Delaunay) is marked.
PETSC_STATIC_INLINE void markface(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] |= (int) (4<<(t)->loc);
}

PETSC_STATIC_INLINE void unmarkface(TetGenMesh *m, triface *t) {
  ((int *) (t->tet))[m->elemmarkerindex] &= ~(int) (4<<(t)->loc);
}

PETSC_STATIC_INLINE PetscBool facemarked(TetGenMesh *m, triface *t) {
  return (((int *) (t->tet))[m->elemmarkerindex] & (int) (4<<(t)->loc)) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

// markedge(), unmarkedge(), edgemarked() -- primitives to flag or unflag an edge of a tetrahedron.  From the last 7th to 12th bits are used for
//   edge markers, e.g., the last 7th bit corresponds to the 0th edge, etc.
// Remark: The last 7th bit is marked by 2^6 = 64.
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
// Each subface contains three pointers to its neighboring subfaces, with
//   edge versions.  To save memory, both information are kept in a single
//   pointer. To make this possible, all subfaces are aligned to eight-byte
//   boundaries, so that the last three bits of each pointer are zeros. An
//   edge version (in the range 0 to 5) is compressed into the last three
//   bits of each pointer by 'sencode()'.  'sdecode()' decodes a pointer,
//   extracting an edge version and a pointer to the beginning of a subface.
PETSC_STATIC_INLINE void sdecode(shellface sptr, face *s) {
  s->shver = (int) ((PETSC_UINTPTR_T) (sptr) & (PETSC_UINTPTR_T) 7);
  s->sh    = (shellface *) ((PETSC_UINTPTR_T) (sptr) & ~ (PETSC_UINTPTR_T) 7);
}

PETSC_STATIC_INLINE shellface sencode(face *s) {
  return (shellface) ((PETSC_UINTPTR_T) s->sh | (PETSC_UINTPTR_T) s->shver);
}

// spivot() finds the other subface (from this subface) that shares the
//   same edge.
PETSC_STATIC_INLINE void spivot(face *s1, face *s2) {
  shellface sptr = s1->sh[Orient(s1->shver)];
  sdecode(sptr, s2);
}

PETSC_STATIC_INLINE void spivotself(face *s) {
  shellface sptr = s->sh[Orient(s->shver)];
  sdecode(sptr, s);
}

// sbond() bonds two subfaces together, i.e., after bonding, both faces
//   are pointing to each other.
PETSC_STATIC_INLINE void sbond(face *s1, face *s2) {
  s1->sh[Orient(s1->shver)] = sencode(s2);
  s2->sh[Orient(s2->shver)] = sencode(s1);
}

// sbond1() only bonds s2 to s1, i.e., after bonding, s1 is pointing to s2, but s2 is not pointing to s1.
PETSC_STATIC_INLINE void sbond1(face *s1, face *s2) {
  s1->sh[Orient(s1->shver)] = sencode(s2);
}

// Dissolve a subface bond (from one side).  Note that the other subface will still think it's connected to this subface.
PETSC_STATIC_INLINE void sdissolve(TetGenMesh *m, face *s) {
  s->sh[Orient(s->shver)] = (shellface) m->dummysh;
}

// These primitives determine or set the origin, destination, or apex of a subface with respect to the edge version.
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

// These primitives were drived from Mucke[2]'s triangle-edge data structure
//   to change face-edge relation in a subface (sesym, senext and senext2).
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

// If f0 and f1 are both in the same face ring, then f1 = f0.fnext(),
PETSC_STATIC_INLINE void sfnext(TetGenMesh *m, face *s1, face *s2) {
  TetGenMeshGetNextSFace(m, s1, s2);
}

PETSC_STATIC_INLINE void sfnextself(TetGenMesh *m, face *s) {
  TetGenMeshGetNextSFace(m, s, PETSC_NULL);
}

// These primitives read or set a pointer of the badface structure.  The pointer is stored sh[11].
PETSC_STATIC_INLINE badface* shell2badface(face *s) {
  return (badface*) s->sh[11];
}

PETSC_STATIC_INLINE void setshell2badface(face *s, badface* value) {
  s->sh[11] = (shellface) value;
}

// Check or set a subface's maximum area bound.
PETSC_STATIC_INLINE PetscReal areabound(TetGenMesh *m, face *s) {
  return ((PetscReal *) (s->sh))[m->areaboundindex];
}

PETSC_STATIC_INLINE void setareabound(TetGenMesh *m, face *s, PetscReal value) {
  ((PetscReal *) (s->sh))[m->areaboundindex] = value;
}

// These two primitives read or set a shell marker.  Shell markers are used
//   to hold user boundary information.
// The last two bits of the int ((int *) ((s).sh))[shmarkindex] are used
//   by sinfect() and smarktest().
PETSC_STATIC_INLINE int shellmark(TetGenMesh *m, face *s) {
  return (((int *) ((s)->sh))[m->shmarkindex]) >> (int) 2;
}

PETSC_STATIC_INLINE void setshellmark(TetGenMesh *m, face *s, int value) {
  ((int *) ((s)->sh))[m->shmarkindex] = (value << (int) 2) + ((((int *) ((s)->sh))[m->shmarkindex]) & (int) 3);
}

// These two primitives set or read the type of the subface or subsegment.
PETSC_STATIC_INLINE shestype shelltype(TetGenMesh *m, face *s) {
  return (shestype) ((int *) (s->sh))[m->shmarkindex + 1];
}

PETSC_STATIC_INLINE void setshelltype(TetGenMesh *m, face *s, shestype value) {
  ((int *) (s->sh))[m->shmarkindex + 1] = (int) value;
}

// These two primitives set or read the pbc group of the subface.
PETSC_STATIC_INLINE int shellpbcgroup(TetGenMesh *m, face *s) {
  return ((int *) (s->sh))[m->shmarkindex + 2];
}

PETSC_STATIC_INLINE void setshellpbcgroup(TetGenMesh *m, face *s, int value) {
  ((int *) (s->sh))[m->shmarkindex + 2] = value;
}

// sinfect(), sinfected(), suninfect() -- primitives to flag or unflag a
//   subface. The last bit of ((int *) ((s).sh))[shmarkindex] is flaged.
PETSC_STATIC_INLINE void sinfect(TetGenMesh *m, face *s) {
  ((int *) ((s)->sh))[m->shmarkindex] = (((int *) ((s)->sh))[m->shmarkindex] | (int) 1);
  // s->sh[6] = (shellface) ((unsigned long) s->sh[6] | (unsigned long) 4l);
}

PETSC_STATIC_INLINE void suninfect(TetGenMesh *m, face *s) {
  ((int *) ((s)->sh))[m->shmarkindex] = (((int *) ((s)->sh))[m->shmarkindex] & ~(int) 1);
  // s->sh[6] = (shellface)((unsigned long) s->sh[6] & ~(unsigned long) 4l);
}

// Test a subface for viral infection.
PETSC_STATIC_INLINE PetscBool sinfected(TetGenMesh *m, face *s) {
  return (((int *) ((s)->sh))[m->shmarkindex] & (int) 1) != 0 ? PETSC_TRUE : PETSC_FALSE;
}

// smarktest(), smarktested(), sunmarktest() -- primitives to flag or unflag
//   a subface. The last 2nd bit of ((int *) ((s).sh))[shmarkindex] is flaged.
#define smarktest(s) ((int *) ((s)->sh))[m->shmarkindex] = (((int *)((s)->sh))[m->shmarkindex] | (int) 2)

#define sunmarktest(s) ((int *) ((s)->sh))[m->shmarkindex] = (((int *)((s)->sh))[m->shmarkindex] & ~(int) 2)

#define smarktested(s) ((((int *) ((s)->sh))[m->shmarkindex] & (int) 2) != 0)
/*** End of primitives for subfaces/subsegments ***/

/*** Begin of primitives for interacting between tetrahedra and subfaces ***/
// tspivot() finds a subface abutting on this tetrahdera.
PETSC_STATIC_INLINE void tspivot(TetGenMesh *m, triface *t, face *s) {
  if ((t)->tet[9]) {
    sdecode(((shellface *) (t)->tet[9])[(t)->loc], s);
  } else {
    (s)->sh = m->dummysh;
  }
}

// stpivot() finds a tetrahedron abutting a subface.
PETSC_STATIC_INLINE void stpivot(TetGenMesh *m, face *s, triface *t) {
  tetrahedron ptr = (tetrahedron) s->sh[6 + EdgeRing(s->shver)];
  decode(ptr, t);
}

// tsbond() bond a tetrahedron to a subface.
PETSC_STATIC_INLINE void tsbond(TetGenMesh *m, triface *t, face *s) {
  if (!(t)->tet[9]) {
    int i;
    PetscErrorCode ierr;
    // Allocate space for this tet.
    ierr = MemoryPoolAlloc(m->tet2subpool, (void **) &(t)->tet[9]);
    // NULL all fields in this space.
    for(i = 0; i < 4; i++) {
      ((shellface *) (t)->tet[9])[i] = (shellface) m->dummysh;
    }
  }
  // Bond t <==> s.
  ((shellface *) (t)->tet[9])[(t)->loc] = sencode(s);
  //t.tet[8 + t.loc] = (tetrahedron) sencode(s);
  s->sh[6 + EdgeRing(s->shver)] = (shellface) encode(t);
}

// tsdissolve() dissolve a bond (from the tetrahedron side).
PETSC_STATIC_INLINE void tsdissolve(TetGenMesh *m, triface *t) {
  if ((t)->tet[9]) {
    ((shellface *) (t)->tet[9])[(t)->loc] = (shellface) m->dummysh;
  }
  // t.tet[8 + t.loc] = (tetrahedron) dummysh;
}

// stdissolve() dissolve a bond (from the subface side).
PETSC_STATIC_INLINE void stdissolve(TetGenMesh *m, face *s) {
  s->sh[6 + EdgeRing(s->shver)] = (shellface) m->dummytet;
}
/*** End of primitives for interacting between tetrahedra and subfaces ***/

/*** Begin of primitives for interacting between subfaces and subsegs ***/
// sspivot() finds a subsegment abutting a subface.
PETSC_STATIC_INLINE void sspivot(TetGenMesh *m, face *s, face *edge) {
  shellface sptr = (shellface) s->sh[8 + Orient(s->shver)];
  sdecode(sptr, edge);
}

// ssbond() bond a subface to a subsegment.
PETSC_STATIC_INLINE void ssbond(TetGenMesh *m, face *s, face *edge) {
  s->sh[8 + Orient(s->shver)] = sencode(edge);
  edge->sh[0] = sencode(s);
}

// ssdisolve() dissolve a bond (from the subface side)
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

// Only bond/dissolve at tet's side, but not vice versa.
PETSC_STATIC_INLINE void tssbond1(TetGenMesh *m, triface *t, face *s)
{
  if (!(t)->tet[8]) {
    int i;
    PetscErrorCode ierr;
    // Allocate space for this tet.
    ierr = MemoryPoolAlloc(m->tet2segpool, (void **) &(t)->tet[8]);
    // NULL all fields in this space.
    for(i = 0; i < 6; i++) {
      ((shellface *) (t)->tet[8])[i] = (shellface) m->dummysh;
    }
  }
  // Bond the segment.
  ((shellface *) (t)->tet[8])[locver2edge[(t)->loc][(t)->ver]] = sencode((s));
}

PETSC_STATIC_INLINE void tssdissolve1(TetGenMesh *m, triface *t)
{
  if ((t)->tet[8]) {
    ((shellface *) (t)->tet[8])[locver2edge[(t)->loc][(t)->ver]] = (shellface) m->dummysh;
  }
}
/*** End of primitives for interacting between tet and subsegs ***/

/*** Begin of advanced primitives ***/

// adjustedgering() adjusts the edge version so that it belongs to the
//   indicated edge ring.  The 'direction' only can be 0(CCW) or 1(CW).
//   If the edge is not in the wanted edge ring, reverse it.
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

// isdead() returns TRUE if the tetrahedron or subface has been dealloced.
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

// isfacehaspoint() returns TRUE if the 'testpoint' is one of the vertices of the tetface 't' subface 's'.
PETSC_STATIC_INLINE PetscBool isfacehaspoint_triface(triface *t, point testpoint) {
  return ((org(t) == testpoint) || (dest(t) == testpoint) || (apex(t) == testpoint)) ? PETSC_TRUE : PETSC_FALSE;
}
PETSC_STATIC_INLINE PetscBool isfacehaspoint_face(face* s, point testpoint) {
  return (s->sh[3] == (shellface) testpoint) || (s->sh[4] == (shellface) testpoint) || (s->sh[5] == (shellface) testpoint) ? PETSC_TRUE : PETSC_FALSE;
}

// isfacehasedge() returns TRUE if the edge (given by its two endpoints) is one of the three edges of the subface 's'.
PETSC_STATIC_INLINE PetscBool isfacehasedge(face* s, point tend1, point tend2) {
  return (isfacehaspoint_face(s, tend1) && isfacehaspoint_face(s, tend2)) ? PETSC_TRUE : PETSC_FALSE;
}

// issymexist() returns TRUE if the adjoining tetrahedron is not 'duumytet'.
PETSC_STATIC_INLINE PetscBool issymexist(TetGenMesh *m, triface* t) {
  tetrahedron *ptr = (tetrahedron *) ((unsigned long)(t->tet[t->loc]) & ~(unsigned long)7l);
  return ptr != m->dummytet ? PETSC_TRUE : PETSC_FALSE;
}

// dot() returns the dot product: v1 dot v2.
PETSC_STATIC_INLINE PetscReal dot(PetscReal *v1, PetscReal *v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}
// cross() computes the cross product: n = v1 cross v2.
PETSC_STATIC_INLINE void cross(PetscReal *v1, PetscReal *v2, PetscReal *n)
{
  n[0] =   v1[1] * v2[2] - v2[1] * v1[2];
  n[1] = -(v1[0] * v2[2] - v2[0] * v1[2]);
  n[2] =   v1[0] * v2[1] - v2[0] * v1[1];
}
// distance() computes the Euclidean distance between two points.
PETSC_STATIC_INLINE PetscReal distance(PetscReal *p1, PetscReal *p2)
{
  return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) +
              (p2[1] - p1[1]) * (p2[1] - p1[1]) +
              (p2[2] - p1[2]) * (p2[2] - p1[2]));
}

// Linear algebra operators.
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
#define __FUNCT__ "lu_decmp"
// lu_decmp()    Compute the LU decomposition of a matrix.                   //
//                                                                           //
// Compute the LU decomposition of a (non-singular) square matrix A using    //
// partial pivoting and implicit row exchanges.  The result is:              //
//     A = P * L * U,                                                        //
// where P is a permutation matrix, L is unit lower triangular, and U is     //
// upper triangular.  The factored form of A is used in combination with     //
// 'lu_solve()' to solve linear equations: Ax = b, or invert a matrix.       //
//                                                                           //
// The inputs are a square matrix 'lu[N..n+N-1][N..n+N-1]', it's size is 'n'.//
// On output, 'lu' is replaced by the LU decomposition of a rowwise permuta- //
// tion of itself, 'ps[N..n+N-1]' is an output vector that records the row   //
// permutation effected by the partial pivoting, effectively,  'ps' array    //
// tells the user what the permutation matrix P is; 'd' is output as +1/-1   //
// depending on whether the number of row interchanges was even or odd,      //
// respectively.                                                             //
//                                                                           //
// Return true if the LU decomposition is successfully computed, otherwise,  //
// return false in case that A is a singular matrix.                         //
PetscBool lu_decmp(PetscReal lu[4][4], int n, int* ps, PetscReal *d, int N)
{
  PetscReal scales[4];
  PetscReal pivot, biggest, mult, tempf;
  int pivotindex = 0;
  int i, j, k;

  *d = 1.0;                                      // No row interchanges yet.

  for (i = N; i < n + N; i++) {                             // For each row.
    // Find the largest element in each row for row equilibration
    biggest = 0.0;
    for (j = N; j < n + N; j++)
      if (biggest < (tempf = fabs(lu[i][j])))
        biggest  = tempf;
    if (biggest != 0.0)
      scales[i] = 1.0 / biggest;
    else {
      scales[i] = 0.0;
      return PETSC_FALSE;                            // Zero row: singular matrix.
    }
    ps[i] = i;                                 // Initialize pivot sequence.
  }

  for (k = N; k < n + N - 1; k++) {                      // For each column.
    // Find the largest element in each column to pivot around.
    biggest = 0.0;
    for (i = k; i < n + N; i++) {
      if (biggest < (tempf = fabs(lu[ps[i]][k]) * scales[ps[i]])) {
        biggest = tempf;
        pivotindex = i;
      }
    }
    if (biggest == 0.0) {
      return PETSC_FALSE;                         // Zero column: singular matrix.
    }
    if (pivotindex != k) {                         // Update pivot sequence.
      j = ps[k];
      ps[k] = ps[pivotindex];
      ps[pivotindex] = j;
      *d = -(*d);                          // ...and change the parity of d.
    }

    // Pivot, eliminating an extra variable  each time
    pivot = lu[ps[k]][k];
    for (i = k + 1; i < n + N; i++) {
      lu[ps[i]][k] = mult = lu[ps[i]][k] / pivot;
      if (mult != 0.0) {
        for (j = k + 1; j < n + N; j++)
          lu[ps[i]][j] -= mult * lu[ps[k]][j];
      }
    }
  }

  // (lu[ps[n + N - 1]][n + N - 1] == 0.0) ==> A is singular.
  return lu[ps[n + N - 1]][n + N - 1] != 0.0 ? PETSC_TRUE : PETSC_FALSE;
}

#undef __FUNCT__
#define __FUNCT__ "lu_solve"
// lu_solve()    Solves the linear equation:  Ax = b,  after the matrix A    //
//               has been decomposed into the lower and upper triangular     //
//               matrices L and U, where A = LU.                             //
//                                                                           //
// 'lu[N..n+N-1][N..n+N-1]' is input, not as the matrix 'A' but rather as    //
// its LU decomposition, computed by the routine 'lu_decmp'; 'ps[N..n+N-1]'  //
// is input as the permutation vector returned by 'lu_decmp';  'b[N..n+N-1]' //
// is input as the right-hand side vector, and returns with the solution     //
// vector. 'lu', 'n', and 'ps' are not modified by this routine and can be   //
// left in place for successive calls with different right-hand sides 'b'.   //
void lu_solve(PetscReal lu[4][4], int n, int *ps, PetscReal *b, int N)
{
  int i, j;
  PetscReal X[4], dot;

  for (i = N; i < n + N; i++) X[i] = 0.0;

  // Vector reduction using U triangular matrix.
  for (i = N; i < n + N; i++) {
    dot = 0.0;
    for (j = N; j < i + N; j++)
      dot += lu[ps[i]][j] * X[j];
    X[i] = b[ps[i]] - dot;
  }

  // Back substitution, in L triangular matrix.
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
// interiorangle()    Return the interior angle (0 - 2 * PI) between vectors //
//                    o->p1 and o->p2.                                       //
//                                                                           //
// 'n' is the normal of the plane containing face (o, p1, p2).  The interior //
// angle is the total angle rotating from o->p1 around n to o->p2.  Exchange //
// the position of p1 and p2 will get the complement angle of the other one. //
// i.e., interiorangle(o, p1, p2) = 2 * PI - interiorangle(o, p2, p1).  Set  //
// 'n' be NULL if you only want the interior angle between 0 - PI.           //
PetscReal interiorangle(PetscReal* o, PetscReal* p1, PetscReal* p2, PetscReal* n)
{
  PetscReal v1[3], v2[3], np[3];
  PetscReal theta, costheta, lenlen;
  PetscReal ori, len1, len2;

  // Get the interior angle (0 - PI) between o->p1, and o->p2.
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
  if (lenlen == 0.0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
#endif
  costheta = dot(v1, v2) / lenlen;
  if (costheta > 1.0) {
    costheta = 1.0; // Roundoff.
  } else if (costheta < -1.0) {
    costheta = -1.0; // Roundoff.
  }
  theta = acos(costheta);
  if (n) {
    // Get a point above the face (o, p1, p2);
    np[0] = o[0] + n[0];
    np[1] = o[1] + n[1];
    np[2] = o[2] + n[2];
    // Adjust theta (0 - 2 * PI).
    ori = orient3d(p1, o, np, p2);
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
  t->object = NONE;
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
  p->firstnumber = 0; // Default item index is numbered from Zero.
  p->mesh_dim = 3; // Default mesh dimension is 3.
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
  p->numberofcorners = 4; // Default is 4 nodes per element.
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
  for (i = 0; i < plc->numberoffacets; i++) {
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
// listinit()    Initialize a list for storing a data type.                  //
//                                                                           //
// Determine the size of each item, set the maximum size allocated at onece, //
// set the expand size in case the list is full, and set the linear order    //
// function if it is provided (default is NULL).                             //
/* tetgenmesh::list::list() and tetgenmesh::list::listinit() */
PetscErrorCode ListCreate(int itbytes, compfunc pcomp, int mitems, int exsize, List **newl)
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
// append()    Add a new item at the end of the list.                        //
//                                                                           //
// A new space at the end of this list will be allocated for storing the new //
// item. If the memory is not sufficient, reallocation will be performed. If //
// 'appitem' is not NULL, the contents of this pointer will be copied to the //
// new allocated space.  Returns the pointer to the new allocated space.     //
/* tetgenmesh::list::append() */
PetscErrorCode ListAppend(List *l, void *appitem, void **newspace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Do we have enough space?
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
// insert()    Insert an item before 'pos' (range from 0 to items - 1).      //
//                                                                           //
// A new space will be inserted at the position 'pos', that is, items lie    //
// after pos (including the item at pos) will be moved one space downwords.  //
// If 'insitem' is not NULL, its contents will be copied into the new        //
// inserted space. Return a pointer to the new inserted space.               //
/* tetgenmesh::list::insert() */
PetscErrorCode ListInsert(List *l, int pos, void *insitem, void **newspace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pos >= l->items) {
    PetscFunctionReturn(ListAppend(l, insitem, newspace));
  }
  // Do we have enough space.
  if (l->items == l->maxitems) {
    char *newbase;

    ierr = PetscMalloc((l->maxitems + l->expandsize) * l->itembytes, &newbase);CHKERRQ(ierr);
    ierr = PetscMemcpy(newbase, l->base, l->maxitems * l->itembytes);CHKERRQ(ierr);
    ierr = PetscFree(l->base);CHKERRQ(ierr);
    l->base      = newbase;
    l->maxitems += l->expandsize;
  }
  // Do block move.
  ierr = PetscMemmove(l->base + (pos + 1) * l->itembytes, l->base + pos * l->itembytes, (l->items - pos) * l->itembytes);CHKERRQ(ierr);
  // Insert the item.
  if (insitem) {
    ierr = PetscMemcpy(l->base + pos * l->itembytes, insitem, l->itembytes);CHKERRQ(ierr);
  }
  l->items++;
  if (newspace) {*newspace = (void *) (l->base + pos * l->itembytes);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListDelete"
// del()    Delete an item at 'pos' (range from 0 to items - 1).             //
//                                                                           //
// The space at 'pos' will be overlapped by other item. If 'order' is 1, the //
// remaining items of the list have the same order as usual, i.e., items lie //
// after pos will be moved one space upwords. If 'order' is 0, the last item //
// of the list will be moved up to pos.                                      //
/* tetgenmesh::list::del() */
PetscErrorCode ListDelete(List *l, int pos, int order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // If 'pos' is the last item of the list, nothing need to do.
  if (pos >= 0 && pos < l->items - 1) {
    if (order == 1) {
      // Do block move.
      ierr = PetscMemmove(l->base + pos * l->itembytes, l->base + (pos + 1) * l->itembytes, (l->items - pos - 1) * l->itembytes);CHKERRQ(ierr);
    } else {
      // Use the last item to overlap the del item.
      ierr = PetscMemcpy(l->base + pos * l->itembytes, l->base + (l->items - 1) * l->itembytes, l->itembytes);CHKERRQ(ierr);
    }
  }
  if (l->items > 0) {
    l->items--;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListHasItem"
// hasitem()    Search in this list to find if 'checkitem' exists.           //
//                                                                           //
// This routine assumes that a linear order function has been set.  It loops //
// through the entire list, compares each item to 'checkitem'. If it exists, //
// return its position (between 0 to items - 1), otherwise, return -1.       //
/* tetgenmesh::list::hasitem() */
PetscErrorCode ListHasItem(List *l, void *checkitem, int *idx)
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

#undef __FUNCT__
#define __FUNCT__ "ListLength"
/* tetgenmesh::list::len() */
PetscErrorCode ListLength(List *l, int *len)
{
  PetscFunctionBegin;
  *len = l->items;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListItem"
/* tetgenmesh::list::operator[]() */
PetscErrorCode ListItem(List *l, int i, void **item)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(item, l->base + i * l->itembytes, l->itembytes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListSetItem"
/* tetgenmesh::list::operator[]() */
PetscErrorCode ListSetItem(List *l, int i, void **item)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(l->base + i * l->itembytes, item, l->itembytes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListClear"
/* tetgenmesh::list::clear() */
PetscErrorCode ListClear(List *l)
{
  PetscFunctionBegin;
  l->items = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ListDestroy"
/* tetgenmesh::list::~list() */
PetscErrorCode ListDestroy(List **l)
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
PetscErrorCode QueueCreate(int bytecount, int itemcount, Queue **newq)
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
PetscErrorCode QueueLength(Queue *q, int *len)
{
  PetscFunctionBegin;
  if (len) {*len = q->linkitems;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QueuePush"
/* tetgenmesh::queue::push() */
PetscErrorCode QueuePush(Queue *q, void *newitem, void **next)
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
PetscErrorCode QueuePop(Queue *q, void **next)
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
PetscErrorCode QueueDestroy(Queue **q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MemoryPoolDestroy(&(*q)->mp);CHKERRQ(ierr);
  ierr = PetscFree(*q);CHKERRQ(ierr);
  *q = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolAlloc"
/* tetgenmesh::memorypool::alloc() */
PetscErrorCode MemoryPoolAlloc(MemoryPool *m, void **item)
{
  void           *newitem;
  void          **newblock;
  PETSC_UINTPTR_T alignptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  // First check the linked list of dead items.  If the list is not
  //   empty, allocate an item from the list rather than a fresh one.
  if (m->deaditemstack) {
    newitem = m->deaditemstack;                     // Take first item in list.
    m->deaditemstack = * (void **) m->deaditemstack;
  } else {
    // Check if there are any free items left in the current block.
    if (m->unallocateditems == 0) {
      // Check if another block must be allocated.
      if (!*m->nowblock) {
        // Allocate a new block of items, pointed to by the previous block.
        ierr = PetscMalloc(m->itemsperblock * m->itembytes + sizeof(void *) + m->alignbytes, &newblock);CHKERRQ(ierr);
        *m->nowblock = (void *) newblock;
        // The next block pointer is NULL.
        *newblock = PETSC_NULL;
      }
      // Move to the new block.
      m->nowblock = (void **) *m->nowblock;
      // Find the first item in the block.
      //   Increment by the size of (void *).
      // alignptr = (unsigned long) (nowblock + 1);
      alignptr = (PETSC_UINTPTR_T) (m->nowblock + 1);
      // Align the item on an `alignbytes'-byte boundary.
      // nextitem = (void *)
      //   (alignptr + (unsigned long) alignbytes -
      //    (alignptr % (unsigned long) alignbytes));
      m->nextitem = (void *) (alignptr + (PETSC_UINTPTR_T) m->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->alignbytes));
      // There are lots of unallocated items left in this block.
      m->unallocateditems = m->itemsperblock;
    }
    // Allocate a new item.
    newitem = m->nextitem;
    // Advance `nextitem' pointer to next free item in block.
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
PetscErrorCode MemoryPoolDealloc(MemoryPool *m, void *dyingitem)
{
  PetscFunctionBegin;
  // Push freshly killed item onto stack.
  *((void **) dyingitem) = m->deaditemstack;
  m->deaditemstack = dyingitem;
  m->items--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolTraversalInit"
// traversalinit()   Prepare to traverse the entire list of items.           //
//                                                                           //
// This routine is used in conjunction with traverse().                      //
/* tetgenmesh::memorypool::traversalinit() */
PetscErrorCode MemoryPoolTraversalInit(MemoryPool *m)
{
  PETSC_UINTPTR_T alignptr;

  PetscFunctionBegin;
  // Begin the traversal in the first block.
  m->pathblock = m->firstblock;
  // Find the first item in the block.  Increment by the size of (void *).
  alignptr = (PETSC_UINTPTR_T) (m->pathblock + 1);
  // Align with item on an `alignbytes'-byte boundary.
  m->pathitem = (void *) (alignptr + (PETSC_UINTPTR_T) m->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->alignbytes));
  // Set the number of items left in the current block.
  m->pathitemsleft = m->itemsperblock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolTraverse"
// traverse()   Find the next item in the list.                              //
//                                                                           //
// This routine is used in conjunction with traversalinit().  Be forewarned  //
// that this routine successively returns all items in the list, including   //
// deallocated ones on the deaditemqueue. It's up to you to figure out which //
// ones are actually dead.  It can usually be done more space-efficiently by //
// a routine that knows something about the structure of the item.           //
/* tetgenmesh::memorypool::traverse() */
PetscErrorCode MemoryPoolTraverse(MemoryPool *m, void **next)
{
  void           *newitem;
  PETSC_UINTPTR_T alignptr;

  PetscFunctionBegin;
  // Stop upon exhausting the list of items.
  if (m->pathitem == m->nextitem) {
    *next = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  // Check whether any untraversed items remain in the current block.
  if (m->pathitemsleft == 0) {
    // Find the next block.
    m->pathblock = (void **) *m->pathblock;
    // Find the first item in the block.  Increment by the size of (void *).
    alignptr = (PETSC_UINTPTR_T) (m->pathblock + 1);
    // Align with item on an `alignbytes'-byte boundary.
    m->pathitem = (void *) (alignptr + (PETSC_UINTPTR_T) m->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->alignbytes));
    // Set the number of items left in the current block.
    m->pathitemsleft = m->itemsperblock;
  }
  newitem = m->pathitem;
  // Find the next item in the block.
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
PetscErrorCode MemoryPoolRestart(MemoryPool *m)
{
  PETSC_UINTPTR_T alignptr;

  PetscFunctionBegin;
  m->items    = 0;
  m->maxitems = 0;
  // Set the currently active block.
  m->nowblock = m->firstblock;
  // Find the first item in the pool.  Increment by the size of (void *).
  // alignptr = (unsigned long) (nowblock + 1);
  alignptr = (PETSC_UINTPTR_T) (m->nowblock + 1);
  // Align the item on an `alignbytes'-byte boundary.
  // nextitem = (void *)
  //   (alignptr + (unsigned long) alignbytes -
  //    (alignptr % (unsigned long) alignbytes));
  m->nextitem = (void *) (alignptr + (PETSC_UINTPTR_T) m->alignbytes - (alignptr % (PETSC_UINTPTR_T) m->alignbytes));
  // There are lots of unallocated items left in this block.
  m->unallocateditems = m->itemsperblock;
  // The stack of deallocated items is empty.
  m->deaditemstack = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolCreate"
// poolinit()    Initialize a pool of memory for allocation of items.        //
//                                                                           //
// A `pool' is created whose records have size at least `bytecount'.  Items  //
// will be allocated in `itemcount'-item blocks.  Each item is assumed to be //
// a collection of words, and either pointers or floating-point values are   //
// assumed to be the "primary" word type.  (The "primary" word type is used  //
// to determine alignment of items.)  If `alignment' isn't zero, all items   //
// will be `alignment'-byte aligned in memory.  `alignment' must be either a //
// multiple or a factor of the primary word size;  powers of two are safe.   //
// `alignment' is normally used to create a few unused bits at the bottom of //
// each item's pointer, in which information may be stored.                  //
/* tetgenmesh::memorypool::memorypool() and tetgenmesh::memorypool::poolinit() */
PetscErrorCode MemoryPoolCreate(int bytecount, int itemcount, wordtype wtype, int alignment, MemoryPool **mp)
{
  MemoryPool    *m;
  int            wordsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(MemoryPool), &m);CHKERRQ(ierr);
  // Initialize values in the pool.
  m->itemwordtype = wtype;
  wordsize        = (m->itemwordtype == POINTER) ? sizeof(void *) : sizeof(PetscReal);
  // Find the proper alignment, which must be at least as large as:
  //   - The parameter `alignment'.
  //   - The primary word type, to avoid unaligned accesses.
  //   - sizeof(void *), so the stack of dead items can be maintained
  //       without unaligned accesses.
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

  // Allocate a block of items.  Space for `itemsperblock' items and one
  //   pointer (to point to the next block) are allocated, as well as space
  //   to ensure alignment of the items.
  ierr =  PetscMalloc(m->itemsperblock * m->itembytes + sizeof(void *) + m->alignbytes, &m->firstblock);CHKERRQ(ierr);
  // Set the next block pointer to NULL.
  *(m->firstblock) = PETSC_NULL;
  ierr = MemoryPoolRestart(m);CHKERRQ(ierr);
  *mp = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MemoryPoolDestroy"
/* tetgenmesh::memorypool::~memorypool() */
PetscErrorCode MemoryPoolDestroy(MemoryPool **m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
// restart()    Deallocate all objects in this pool.                         //
//                                                                           //
// The pool returns to a fresh state, like after it was initialized, except  //
// that no memory is freed to the operating system.  Rather, the previously  //
// allocated blocks are ready to be used.                                    //
/* tetgenmesh::arraypool::restart() */
PetscErrorCode ArrayPoolRestart(ArrayPool *a)
{
  PetscFunctionBegin;
  a->objects = 0l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolCreate"
// poolinit()    Initialize an arraypool for allocation of objects.          //
//                                                                           //
// Before the pool may be used, it must be initialized by this procedure.    //
// After initialization, memory can be allocated and freed in this pool.     //
/* tetgenmesh::arraypool::arraypool() and tetgenmesh::arraypool::poolinit() */
PetscErrorCode ArrayPoolCreate(int sizeofobject, int log2objperblk, ArrayPool **ap)
{
  ArrayPool     *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(ArrayPool), &a);CHKERRQ(ierr);
  // Each object must be at least one byte long.
  a->objectbytes         = sizeofobject > 1 ? sizeofobject : 1;
  a->log2objectsperblock = log2objperblk;
  // Compute the number of objects in each block.
  a->objectsperblock = ((int) 1) << a->log2objectsperblock;
  // No memory has been allocated.
  a->totalmemory = 0l;
  // The top array has not been allocated yet.
  a->toparray    = PETSC_NULL;
  a->toparraylen = 0;
  // Ready all indices to be allocated.
  ierr = ArrayPoolRestart(a);CHKERRQ(ierr);
  *ap = a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolGetBlock"
// getblock()    Return (and perhaps create) the block containing the object //
//               with a given index.                                         //
//                                                                           //
// This function takes care of allocating or resizing the top array if nece- //
// ssary, and of allocating the block if it hasn't yet been allocated.       //
//                                                                           //
// Return a pointer to the beginning of the block (NOT the object).          //
/* tetgenmesh::arraypool::getblock() */
PetscErrorCode ArrayPoolGetBlock(ArrayPool *a, int objectindex, char **blk)
{
  char **newarray;
  char *block;
  int newsize;
  int topindex;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Compute the index in the top array (upper bits).
  topindex = objectindex >> a->log2objectsperblock;
  // Does the top array need to be allocated or resized?
  if (!a->toparray) {
    // Allocate the top array big enough to hold 'topindex', and NULL out its contents.
    newsize = topindex + 128;
    ierr = PetscMalloc(newsize * sizeof(char *), &a->toparray);CHKERRQ(ierr);
    a->toparraylen = newsize;
    for(i = 0; i < newsize; i++) {
      a->toparray[i] = PETSC_NULL;
    }
    // Account for the memory.
    a->totalmemory = newsize * (unsigned long) sizeof(char *);
  } else if (topindex >= a->toparraylen) {
    // Resize the top array, making sure it holds 'topindex'.
    newsize = 3 * a->toparraylen;
    if (topindex >= newsize) {
      newsize = topindex + 128;
    }
    // Allocate the new array, copy the contents, NULL out the rest, and free the old array.
    ierr = PetscMalloc(newsize * sizeof(char *), &newarray);CHKERRQ(ierr);
    for(i = 0; i < a->toparraylen; i++) {
      newarray[i] = a->toparray[i];
    }
    for(i = a->toparraylen; i < newsize; i++) {
      newarray[i] = PETSC_NULL;
    }
    ierr = PetscFree(a->toparray);CHKERRQ(ierr);
    // Account for the memory.
    a->totalmemory += (newsize - a->toparraylen) * sizeof(char *);
    a->toparray     = newarray;
    a->toparraylen  = newsize;
  }
  // Find the block, or learn that it hasn't been allocated yet.
  block = a->toparray[topindex];
  if (!block) {
    // Allocate a block at this index.
    ierr = PetscMalloc(a->objectsperblock * a->objectbytes, &block);CHKERRQ(ierr);
    a->toparray[topindex] = block;
    // Account for the memory.
    a->totalmemory += a->objectsperblock * a->objectbytes;
  }
  // Return a pointer to the block.
  *blk = block;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolNewIndex"
// newindex()    Allocate space for a fresh object from the pool.            //
/* tetgenmesh::arraypool::newindex() */
PetscErrorCode ArrayPoolNewIndex(ArrayPool *a, void **newptr, int *idx)
{
  char          *block;
  void          *newobject;
  int            newindex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Allocate an object at index 'firstvirgin'.
  ierr = ArrayPoolGetBlock(a, a->objects, &block);CHKERRQ(ierr);
  newindex  = a->objects;
  newobject = (void *) (block + (a->objects & (a->objectsperblock - 1)) * a->objectbytes);
  a->objects++;
  // If 'newptr' is not NULL, use it to return a pointer to the object.
  if (newptr) {*newptr = newobject;}
  if (idx)    {*idx    = newindex;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ArrayPoolDestroy"
/* tetgenmesh::arraypool::~arraypool() */
PetscErrorCode ArrayPoolDestroy(ArrayPool **a)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Has anything been allocated at all?
  if ((*a)->toparray) {
    // Walk through the top array.
    for(i = 0; i < (*a)->toparraylen; ++i) {
      // Check every pointer; NULLs may be scattered randomly.
      if ((*a)->toparray[i]) {
        // Free an allocated block.
        ierr = PetscFree((*a)->toparray[i]);CHKERRQ(ierr);
      }
    }
    // Free the top array.
    ierr = PetscFree((*a)->toparray);CHKERRQ(ierr);
  }
  // The top array is no longer allocated.
  (*a)->toparray    = PETSC_NULL;
  (*a)->toparraylen = 0;
  (*a)->objects     = 0;
  (*a)->totalmemory = 0;
  PetscFunctionReturn(0);
}

//// prim_cxx /////////////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetNextSFace"
// getnextsface()    Finds the next subface in the face ring.                //
//                                                                           //
// For saving space in the data structure of subface, there only exists one  //
// face ring around a segment (see programming manual).  This routine imple- //
// ments the double face ring as desired in Muecke's data structure.         //
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
// findedge()    Find an edge in the given tet or subface.                   //
//                                                                           //
// The edge is given in two points 'eorg' and 'edest'.  It is assumed that   //
// the edge must exist in the given handle (tetrahedron or subface).  This   //
// routine sets the right edge version for the input handle.                 //
/* tetgenmesh::findedge() */
PetscErrorCode TetGenMeshFindEdge_triface(TetGenMesh *m, triface *tface, point eorg, point edest)
{
  PetscInt       i;

  PetscFunctionBegin;
  for(i = 0; i < 3; i++) {
    if (org(tface) == eorg) {
      if (dest(tface) == edest) {
        // Edge is found, return.
        PetscFunctionReturn(0);
      }
    } else {
      if (org(tface) == edest) {
        if (dest(tface) == eorg) {
          // Edge is found, invert the direction and return.
          esymself(tface);
          PetscFunctionReturn(0);
        }
      }
    }
    enextself(tface);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find an edge in tet");
}
PetscErrorCode TetGenMeshFindEdge_face(TetGenMesh *m, face *sface, point eorg, point edest)
{
  PetscInt       i;

  PetscFunctionBegin;
  for(i = 0; i < 3; i++) {
    if (sorg(sface) == eorg) {
      if (sdest(sface) == edest) {
        // Edge is found, return.
        PetscFunctionReturn(0);
      }
    } else {
      if (sorg(sface) == edest) {
        if (sdest(sface) == eorg) {
          // Edge is found, invert the direction and return.
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
// tsspivot()    Finds a subsegment abutting on a tetrahderon's edge.        //
//                                                                           //
// The edge is represented in the primary edge of 'checkedge'. If there is a //
// subsegment bonded at this edge, it is returned in handle 'checkseg', the  //
// edge direction of 'checkseg' is conformed to 'checkedge'. If there isn't, //
// set 'checkseg.sh = dummysh' to indicate it is not a subsegment.           //
//                                                                           //
// To find whether an edge of a tetrahedron is a subsegment or not. First we //
// need find a subface around this edge to see if it contains a subsegment.  //
// The reason is there is no direct connection between a tetrahedron and its //
// adjoining subsegments.                                                    //
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
    // Does spintet have a (non-fake) subface attached?
    if ((parentsh.sh != m->dummysh) && (sapex(&parentsh))) {
      // Find a subface! Find the edge in it.
      ierr = TetGenMeshFindEdge_face(m, &parentsh, org(checkedge), dest(checkedge));CHKERRQ(ierr);
      sspivot(m, &parentsh, checkseg);
      if (checkseg->sh != m->dummysh) {
        // Find a subsegment! Correct its edge direction before return.
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
  // Not find.
  checkseg->sh = m->dummysh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSstPivot"
// sstpivot()    Finds a tetrahedron abutting a subsegment.                  //
//                                                                           //
// This is the inverse operation of 'tsspivot()'.  One subsegment shared by  //
// arbitrary number of tetrahedron, the returned tetrahedron is not unique.  //
// The edge direction of the returned tetrahedron is conformed to the given  //
// subsegment.                                                               //
/* tetgenmesh::sstpivot() */
PetscErrorCode TetGenMeshSstPivot(TetGenMesh *m, face* checkseg, triface* retedge)
{
  face parentsh = {PETSC_NULL, 0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Get the subface which holds the subsegment.
  sdecode(checkseg->sh[0], &parentsh);
#ifdef PETSC_USE_DEBUG
    if (parentsh.sh == m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Shell edge should not be null");}
#endif
  // Get a tetraheron to which the subface attches.
  stpivot(m, &parentsh, retedge);
  if (retedge->tet == m->dummytet) {
    sesymself(&parentsh);
    stpivot(m, &parentsh, retedge);
#ifdef PETSC_USE_DEBUG
    if (retedge->tet == m->dummytet) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Tet should not be null");}
#endif
  }
  // Correct the edge direction before return.
  ierr = TetGenMeshFindEdge_triface(m, retedge, sorg(checkseg), sdest(checkseg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPoint2TetOrg"
// point2tetorg(), point2shorg(), point2segorg()                             //
//                                                                           //
// Return a tet, a subface, or a subsegment whose origin is the given point. //
// These routines assume the maps between points to tets (subfaces, segments //
// ) have been built and maintained.                                         //
/* tetgenmesh::point2tetorg() */
PetscErrorCode TetGenMeshPoint2TetOrg(TetGenMesh *m, point pa, triface *searchtet)
{
  int i;

  PetscFunctionBegin;
  // Search a tet whose origin is pa.
  decode(point2tet(m, pa), searchtet);
  if (!searchtet->tet) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad tet pointer.\n", pointmark(m, pa));}
  for(i = 4; i < 8; i++) {
    if ((point) searchtet->tet[i] == pa) {
      // Found. Set pa as its origin.
      switch (i) {
        case 4: searchtet->loc = 0; searchtet->ver = 0; break;
        case 5: searchtet->loc = 0; searchtet->ver = 2; break;
        case 6: searchtet->loc = 0; searchtet->ver = 4; break;
        case 7: searchtet->loc = 1; searchtet->ver = 2; break;
      }
      break;
    }
  }
  if (i == 8) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad tet pointer.\n", pointmark(m, pa));}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPoint2ShOrg"
// point2tetorg(), point2shorg(), point2segorg()                             //
//                                                                           //
// Return a tet, a subface, or a subsegment whose origin is the given point. //
// These routines assume the maps between points to tets (subfaces, segments //
// ) have been built and maintained.                                         //
/* tetgenmesh::point2shorg() */
PetscErrorCode TetGenMeshPoint2ShOrg(TetGenMesh *m, point pa, face *searchsh)
{
  PetscFunctionBegin;
  sdecode(point2sh(m, pa), searchsh);
  if (!searchsh->sh) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad sub pointer.\n", pointmark(m, pa));
  }
  if (((point) searchsh->sh[3]) == pa) {
    searchsh->shver = 0;
  } else if (((point) searchsh->sh[4]) == pa) {
    searchsh->shver = 2;
  } else if (((point) searchsh->sh[5]) == pa) {
    searchsh->shver = 4;
  } else {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad sub pointer.\n", pointmark(m, pa));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPoint2SegOrg"
// point2tetorg(), point2shorg(), point2segorg()                             //
//                                                                           //
// Return a tet, a subface, or a subsegment whose origin is the given point. //
// These routines assume the maps between points to tets (subfaces, segments //
// ) have been built and maintained.                                         //
/* tetgenmesh::point2segorg() */
PetscErrorCode TetGenMeshPoint2SegOrg(TetGenMesh *m, point pa, face *searchsh)
{
  PetscFunctionBegin;
  sdecode(point2seg(m, pa), searchsh);
  if (!searchsh->sh) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad seg pointer.\n", pointmark(m, pa));
  }
  if (((point) searchsh->sh[3]) == pa) {
    searchsh->shver = 0;
  } else if (((point) searchsh->sh[4]) == pa) {
    searchsh->shver = 1;
  } else {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error: %d contains bad seg pointer.\n", pointmark(m, pa));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetSubsegFarOrg"
// getsubsegfarorg()    Get the origin of the parent segment of a subseg.    //
/* tetgenmesh::getsubsegfarorg() */
PetscErrorCode TetGenMeshGetSubsegFarOrg(TetGenMesh *m, face *sseg, point *p)
{
  face prevseg = {PETSC_NULL, 0};
  point checkpt;

  PetscFunctionBegin;
  checkpt = sorg(sseg);
  senext2(sseg, &prevseg);
  spivotself(&prevseg);
  // Search dorg along the original direction of sseg.
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
// getsubsegfardest()    Get the dest. of the parent segment of a subseg.    //
/* tetgenmesh::getsubsegfardest() */
PetscErrorCode TetGenMeshGetSubsegFarDest(TetGenMesh *m, face *sseg, point *p)
{
  face nextseg = {PETSC_NULL, 0};
  point checkpt;

  PetscFunctionBegin;
  checkpt = sdest(sseg);
  senext(sseg, &nextseg);
  spivotself(&nextseg);
  // Search dorg along the destinational direction of sseg.
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
PetscErrorCode TetGenMeshPrintTet(TetGenMesh *m, triface *tface)
{
  TetGenOpts    *b = m->b;
  triface        tmpface = {PETSC_NULL, 0, 0}, prtface = {PETSC_NULL, 0, 0};
  shellface     *shells;
  point          tmppt;
  face           checksh = {PETSC_NULL, 0};
  int            facecount;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Tetra x%lx with loc(%i) and ver(%i):", (PETSC_UINTPTR_T) tface->tet, tface->loc, tface->ver);
  if (infected(m, tface)) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " (infected)");
  }
  if (marktested(m, tface)) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " (marked)");
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "\n");

  tmpface = *tface;
  facecount = 0;
  while(facecount < 4) {
    tmpface.loc = facecount;
    sym(&tmpface, &prtface);
    if (prtface.tet == m->dummytet) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "      [%i] Outer space.\n", facecount);
    } else {
      if (!isdead_triface(&prtface)) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [%i] x%lx  loc(%i).", facecount, (PETSC_UINTPTR_T) prtface.tet, prtface.loc);
        if (infected(m, &prtface)) {
          ierr = PetscPrintf(PETSC_COMM_SELF, " (infected)");
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF, "      [%i] NULL\n", facecount);
      }
    }
    facecount ++;
  }

  tmppt = org(tface);
  if (!tmppt) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Org [%i] NULL\n", locver2org[tface->loc][tface->ver]);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Org [%i] x%lx (%.12g,%.12g,%.12g) %d\n",
                       locver2org[tface->loc][tface->ver], (PETSC_UINTPTR_T) tmppt, tmppt[0], tmppt[1], tmppt[2], pointmark(m, tmppt));
  }
  tmppt = dest(tface);
  if(tmppt == (point) NULL) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Dest[%i] NULL\n", locver2dest[tface->loc][tface->ver]);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Dest[%i] x%lx (%.12g,%.12g,%.12g) %d\n",
                       locver2dest[tface->loc][tface->ver], (PETSC_UINTPTR_T) tmppt, tmppt[0], tmppt[1], tmppt[2], pointmark(m, tmppt));
  }
  tmppt = apex(tface);
  if (!tmppt) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Apex[%i] NULL\n", locver2apex[tface->loc][tface->ver]);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Apex[%i] x%lx (%.12g,%.12g,%.12g) %d\n",
                       locver2apex[tface->loc][tface->ver], (PETSC_UINTPTR_T) tmppt, tmppt[0], tmppt[1], tmppt[2], pointmark(m, tmppt));
  }
  tmppt = oppo(tface);
  if (!tmppt) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Oppo[%i] NULL\n", loc2oppo[tface->loc]);
  } else {
    ierr = PetscPrintf(PETSC_COMM_SELF, "      Oppo[%i] x%lx (%.12g,%.12g,%.12g) %d\n",
                       loc2oppo[tface->loc], (PETSC_UINTPTR_T) tmppt, tmppt[0], tmppt[1], tmppt[2], pointmark(m, tmppt));
  }

  if (b->useshelles) {
    if (tface->tet[8]) {
      shells = (shellface *) tface->tet[8];
      for (facecount = 0; facecount < 6; facecount++) {
        sdecode(shells[facecount], &checksh);
        if (checksh.sh != m->dummysh) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] x%lx %d.", facecount, (PETSC_UINTPTR_T) checksh.sh, checksh.shver);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] NULL.", facecount);
        }
        if (locver2edge[tface->loc][tface->ver] == facecount) {
          ierr = PetscPrintf(PETSC_COMM_SELF, " (*)");  // It is the current edge.
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");
      }
    }
    if (tface->tet[9]) {
      shells = (shellface *) tface->tet[9];
      for (facecount = 0; facecount < 4; facecount++) {
        sdecode(shells[facecount], &checksh);
        if (checksh.sh != m->dummysh) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] x%lx %d.", facecount, (PETSC_UINTPTR_T) checksh.sh, checksh.shver);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      [%d] NULL.", facecount);
        }
        if (tface->loc == facecount) {
          ierr = PetscPrintf(PETSC_COMM_SELF, " (*)");  // It is the current face.
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");
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

////                                                                       ////
////                                                                       ////
//// prim_cxx /////////////////////////////////////////////////////////////////

//// mempool_cxx //////////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDummyInit"
// dummyinit()    Initialize the tetrahedron that fills "outer space" and    //
//                the omnipresent subface.                                   //
//                                                                           //
// The tetrahedron that fills "outer space" called 'dummytet', is pointed to //
// by every tetrahedron and subface on a boundary (be it outer or inner) of  //
// the tetrahedralization. Also, 'dummytet' points to one of the tetrahedron //
// on the convex hull(until the holes and concavities are carved), making it //
// possible to find a starting tetrahedron for point location.               //
//                                                                           //
// The omnipresent subface,'dummysh', is pointed to by every tetrahedron or  //
// subface that doesn't have a full complement of real subface to point to.  //
/* tetgenmesh::dummyinit() */
PetscErrorCode TetGenMeshDummyInit(TetGenMesh *m, int tetwords, int shwords)
{
  TetGenOpts    *b = m->b;
  unsigned long  alignptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Set up 'dummytet', the 'tetrahedron' that occupies "outer space".
  ierr = PetscMalloc(tetwords * sizeof(tetrahedron) + m->tetrahedrons->alignbytes, &m->dummytetbase);CHKERRQ(ierr);
  // Align 'dummytet' on a 'tetrahedrons->alignbytes'-byte boundary.
  alignptr = (unsigned long) m->dummytetbase;
  m->dummytet = (tetrahedron *) (alignptr + (unsigned long) m->tetrahedrons->alignbytes
                                 - (alignptr % (unsigned long) m->tetrahedrons->alignbytes));
  // Initialize the four adjoining tetrahedra to be "outer space". These
  //   will eventually be changed by various bonding operations, but their
  //   values don't really matter, as long as they can legally be
  //   dereferenced.
  m->dummytet[0] = (tetrahedron) m->dummytet;
  m->dummytet[1] = (tetrahedron) m->dummytet;
  m->dummytet[2] = (tetrahedron) m->dummytet;
  m->dummytet[3] = (tetrahedron) m->dummytet;
  // Four null vertex points.
  m->dummytet[4] = PETSC_NULL;
  m->dummytet[5] = PETSC_NULL;
  m->dummytet[6] = PETSC_NULL;
  m->dummytet[7] = PETSC_NULL;

  if (b->useshelles) {
    // Set up 'dummysh', the omnipresent "subface" pointed to by any
    //   tetrahedron side or subface end that isn't attached to a real
    //   subface.
    ierr = PetscMalloc(shwords * sizeof(shellface) + m->subfaces->alignbytes, &m->dummyshbase);CHKERRQ(ierr);
    // Align 'dummysh' on a 'subfaces->alignbytes'-byte boundary.
    alignptr = (unsigned long) m->dummyshbase;
    m->dummysh = (shellface *) (alignptr + (unsigned long) m->subfaces->alignbytes
                                - (alignptr % (unsigned long) m->subfaces->alignbytes));
    // Initialize the three adjoining subfaces to be the omnipresent
    //   subface. These will eventually be changed by various bonding
    //   operations, but their values don't really matter, as long as they
    //   can legally be dereferenced.
    m->dummysh[0] = (shellface) m->dummysh;
    m->dummysh[1] = (shellface) m->dummysh;
    m->dummysh[2] = (shellface) m->dummysh;
    // Three null vertex points.
    m->dummysh[3] = PETSC_NULL;
    m->dummysh[4] = PETSC_NULL;
    m->dummysh[5] = PETSC_NULL;
    // Initialize the two adjoining tetrahedra to be "outer space".
    m->dummysh[6] = (shellface) m->dummytet;
    m->dummysh[7] = (shellface) m->dummytet;
    // Initialize the three adjoining subsegments to be "out boundary".
    m->dummysh[8]  = (shellface) m->dummysh;
    m->dummysh[9]  = (shellface) m->dummysh;
    m->dummysh[10] = (shellface) m->dummysh;
    // Initialize the pointer to badface structure.
    m->dummysh[11] = PETSC_NULL;
    // Initialize the four adjoining subfaces of 'dummytet' to be the
    //   omnipresent subface.
    m->dummytet[8 ] = PETSC_NULL;
    m->dummytet[9 ] = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInitializePools"
// initializepools()    Calculate the sizes of the point, tetrahedron, and   //
//                      subface. Initialize their memory pools.              //
//                                                                           //
// This routine also computes the indices 'pointmarkindex', 'point2simindex',//
// and 'point2pbcptindex' used to find values within each point;  computes   //
// indices 'highorderindex', 'elemattribindex', and 'volumeboundindex' used  //
// to find values within each tetrahedron.                                   //
//                                                                           //
// There are two types of boundary elements, which are subfaces and subsegs, //
// they are stored in seperate pools. However, the data structures of them   //
// are the same.  A subsegment can be regarded as a degenerate subface, i.e.,//
// one of its three corners is not used. We set the apex of it be 'NULL' to  //
// distinguish it's a subsegment.                                            //
/* tetgenmesh::initializepools() */
PetscErrorCode TetGenMeshInitializePools(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  PLC           *in = m->in;
  wordtype       wtype;
  int            pointsize, elesize, shsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Default checkpbc = 0;
  if ((b->plc || b->refine) && (in->pbcgrouplist)) {
    m->checkpbcs = 1;
  }
  // Default varconstraint = 0;
  if (in->segmentconstraintlist || in->facetconstraintlist) {
    m->varconstraint = 1;
  }

  // The index within each point at which its metric tensor is found. It is
  //   saved directly after the list of point attributes.
  m->pointmtrindex = 3 + in->numberofpointattributes;
  // Decide the size (1, 3, or 6) of the metric tensor.
  if (b->metric) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    // For '-m' option. A tensor field is provided (*.mtr or *.b.mtr file).
    if (bgm) {
      // A background mesh is allocated. It may not exist though.
      sizeoftensor = bgm->in ? bgm->in->numberofpointmtrs : in->numberofpointmtrs;
    } else {
      // No given background mesh - Itself is a background mesh.
      sizeoftensor = in->numberofpointmtrs;
    }
    // Make sure sizeoftensor is at least 1.
    sizeoftensor = (sizeoftensor > 0) ? sizeoftensor : 1;
#endif
  } else {
    // For '-q' option. Make sure to have space for saving a scalar value.
    m->sizeoftensor = b->quality ? 1 : 0;
  }
  // The index within each point at which an element pointer is found, where
  //   the index is measured in pointers. Ensure the index is aligned to a
  //   sizeof(tetrahedron)-byte address.
  m->point2simindex = ((m->pointmtrindex + m->sizeoftensor) * sizeof(PetscReal) + sizeof(tetrahedron) - 1) / sizeof(tetrahedron);
  if (b->plc || b->refine || b->voroout) {
    // Increase the point size by four pointers, which are:
    //   - a pointer to a tet, read by point2tet();
    //   - a pointer to a subface, read by point2sh();
    //   - a pointer to a subsegment, read by point2seg();
    //   - a pointer to a parent point, read by point2ppt()).
    if (b->metric) {
      // Increase one pointer to a tet of the background mesh.
      pointsize = (m->point2simindex + 5) * sizeof(tetrahedron);
    } else {
      pointsize = (m->point2simindex + 4) * sizeof(tetrahedron);
    }
    // The index within each point at which a pbc point is found.
    m->point2pbcptindex = (pointsize + sizeof(tetrahedron) - 1) / sizeof(tetrahedron);
    if (m->checkpbcs) {
      // Increase the size by one pointer to a corresponding pbc point,
      //   read by point2pbcpt().
      pointsize = (m->point2pbcptindex + 1) * sizeof(tetrahedron);
    }
  } else {
    // Increase the point size by FOUR pointer, which are:
    //   - a pointer to a tet, read by point2tet();
    //   - a pointer to a subface, read by point2sh(); -- !! Unused !!
    //   - a pointer to a subsegment, read by point2seg(); -- !! Unused !!
    //   - a pointer to a parent point, read by point2ppt()). -- Used by btree.
    pointsize = (m->point2simindex + 4) * sizeof(tetrahedron);
  }
  // The index within each point at which the boundary marker is found,
  //   Ensure the point marker is aligned to a sizeof(int)-byte address.
  m->pointmarkindex = (pointsize + sizeof(int) - 1) / sizeof(int);
  // Now point size is the ints (inidcated by pointmarkindex) plus:
  //   - an integer for boundary marker;
  //   - an integer for vertex type;
  //pointsize = (pointmarkindex + 2) * sizeof(int); // Wrong for 64 bit.
  pointsize = (m->pointmarkindex + 2) * sizeof(tetrahedron);
  // Decide the wordtype used in vertex pool.
  wtype = (sizeof(PetscReal) >= sizeof(tetrahedron)) ? FLOATINGPOINT : POINTER;
  // Initialize the pool of vertices.
  ierr = MemoryPoolCreate(pointsize, VERPERBLOCK, wtype, 0, &m->points);CHKERRQ(ierr);

  if (b->useshelles) { /* For abovepoint() */
    ierr = PetscMalloc(sizeof(pointsize), &m->dummypoint);CHKERRQ(ierr);
  }

  // The number of bytes occupied by a tetrahedron.  There are four pointers
  //   to other tetrahedra, four pointers to corners, and possibly four
  //   pointers to subfaces (or six pointers to subsegments (used in
  //   segment recovery only)).
  elesize = (8 + b->useshelles * 2) * sizeof(tetrahedron);
  // If Voronoi diagram is wanted, make sure we have additional space.
  if (b->voroout) {
    elesize = (8 + 4) * sizeof(tetrahedron);
  }
  // The index within each element at which its attributes are found, where
  //   the index is measured in PetscReals.
  m->elemattribindex = (elesize + sizeof(PetscReal) - 1) / sizeof(PetscReal);
  // The index within each element at which the maximum voulme bound is
  //   found, where the index is measured in PetscReals.  Note that if the
  //   `b->regionattrib' flag is set, an additional attribute will be added.
  m->volumeboundindex = m->elemattribindex + in->numberoftetrahedronattributes + (b->regionattrib > 0);
  // If element attributes or an constraint are needed, increase the number
  //   of bytes occupied by an element.
  if (b->varvolume) {
    elesize = (m->volumeboundindex + 1) * sizeof(PetscReal);
  } else if (in->numberoftetrahedronattributes + b->regionattrib > 0) {
    elesize = m->volumeboundindex * sizeof(PetscReal);
  }
  // If element neighbor graph is requested (-n switch), an additional
  //   integer is allocated for each element.
  // elemmarkerindex = (elesize + sizeof(int) - 1) / sizeof(int);
  m->elemmarkerindex = (elesize + sizeof(int) - 1) / sizeof(int);
  // if (b->neighout || b->voroout) {
    // elesize = (elemmarkerindex + 1) * sizeof(int);
    // Allocate one slot for the element marker. The actual need isa size
    //   of an integer. We allocate enough space (a pointer) for alignment
    //   for 64 bit system. Thanks Liu Yang (LORIA/INRIA) for reporting
    //   this problem.
    elesize = elesize + sizeof(tetrahedron);
  // }
  // If -o2 switch is used, an additional pointer pointed to the list of
  //   higher order nodes is allocated for each element.
  m->highorderindex = (elesize + sizeof(tetrahedron) - 1) / sizeof(tetrahedron);
  if (b->order == 2) {
    elesize = (m->highorderindex + 1) * sizeof(tetrahedron);
  }
  // Having determined the memory size of an element, initialize the pool.
  ierr = MemoryPoolCreate(elesize, ELEPERBLOCK, POINTER, 8, &m->tetrahedrons);CHKERRQ(ierr);

  if (b->useshelles) {
    // The number of bytes occupied by a subface.  The list of pointers
    //   stored in a subface are: three to other subfaces, three to corners,
    //   three to subsegments, two to tetrahedra, and one to a badface.
    shsize = 12 * sizeof(shellface);
    // The index within each subface at which the maximum area bound is
    //   found, where the index is measured in PetscReals.
    m->areaboundindex = (shsize + sizeof(PetscReal) - 1) / sizeof(PetscReal);
    // If -q switch is in use, increase the number of bytes occupied by
    //   a subface for saving maximum area bound.
    if (b->quality && m->varconstraint) {
      shsize = (m->areaboundindex + 1) * sizeof(PetscReal);
    } else {
      shsize = m->areaboundindex * sizeof(PetscReal);
    }
    // The index within subface at which the facet marker is found. Ensure
    //   the marker is aligned to a sizeof(int)-byte address.
    m->shmarkindex = (shsize + sizeof(int) - 1) / sizeof(int);
    // Increase the number of bytes by two or three integers, one for facet
    //   marker, one for shellface type, and optionally one for pbc group.
    shsize = (m->shmarkindex + 2 + m->checkpbcs) * sizeof(int);
    // Initialize the pool of subfaces. Each subface record is eight-byte
    //   aligned so it has room to store an edge version (from 0 to 5) in
    //   the least three bits.
    ierr = MemoryPoolCreate(shsize, SUBPERBLOCK, POINTER, 8, &m->subfaces);CHKERRQ(ierr);
    // Initialize the pool of subsegments. The subsegment's record is same
    //   with subface.
    ierr = MemoryPoolCreate(shsize, SUBPERBLOCK, POINTER, 8, &m->subsegs);CHKERRQ(ierr);
    // Initialize the pool for tet-subseg connections.
    ierr = MemoryPoolCreate(6*sizeof(shellface), SUBPERBLOCK, POINTER, 0, &m->tet2segpool);CHKERRQ(ierr);
    // Initialize the pool for tet-subface connections.
    ierr = MemoryPoolCreate(4*sizeof(shellface), SUBPERBLOCK, POINTER, 0, &m->tet2subpool);CHKERRQ(ierr);
    // Initialize arraypools for segment & facet recovery.
    ierr = ArrayPoolCreate(sizeof(face), 10, &m->subsegstack);CHKERRQ(ierr);
    ierr = ArrayPoolCreate(sizeof(face), 10, &m->subfacstack);CHKERRQ(ierr);
    // Initialize the "outer space" tetrahedron and omnipresent subface.
    ierr = TetGenMeshDummyInit(m, m->tetrahedrons->itemwords, m->subfaces->itemwords);CHKERRQ(ierr);
  } else {
    // Initialize the "outer space" tetrahedron.
    ierr = TetGenMeshDummyInit(m, m->tetrahedrons->itemwords, 0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakePoint2TetMap"
// makepoint2tetmap()    Construct a mapping from points to tetrahedra.      //
//                                                                           //
// Traverses all the tetrahedra,  provides each corner of each tetrahedron   //
// with a pointer to that tetrahedera.  Some pointers will be overwritten by //
// other pointers because each point may be a corner of several tetrahedra,  //
// but in the end every point will point to a tetrahedron that contains it.  //
/* tetgenmesh::makepoint2tetmap() */
PetscErrorCode TetGenMeshMakePoint2TetMap(TetGenMesh *m)
{
  TetGenOpts    *b  = m->b;
  triface tetloop = {PETSC_NULL, 0, 0};
  point pointptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo(b->in, "  Constructing mapping from points to tetrahedra.\n");

  // Initialize the point2tet field of each point.
  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  ierr = TetGenMeshPointTraverse(m, &pointptr);CHKERRQ(ierr);
  while(pointptr) {
    setpoint2tet(m, pointptr, PETSC_NULL);
    ierr = TetGenMeshPointTraverse(m, &pointptr);CHKERRQ(ierr);
  }

  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    // Check all four points of the tetrahedron.
    tetloop.loc = 0;
    pointptr = org(&tetloop);
    setpoint2tet(m, pointptr, encode(&tetloop));
    pointptr = dest(&tetloop);
    setpoint2tet(m, pointptr, encode(&tetloop));
    pointptr = apex(&tetloop);
    setpoint2tet(m, pointptr, encode(&tetloop));
    pointptr = oppo(&tetloop);
    setpoint2tet(m, pointptr, encode(&tetloop));
    // Get the next tetrahedron in the list.
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
// makeindex2pointmap()    Create a map from index to vertices.              //
//                                                                           //
// 'idx2verlist' returns the created map.  Traverse all vertices, a pointer  //
// to each vertex is set into the array.  The pointer to the first vertex is //
// saved in 'idx2verlist[0]'.  Don't forget to minus 'in->firstnumber' when  //
// to get the vertex form its index.                                         //
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
// makesegmentmap(), makesubfacemap(), maketetrahedronmap()                  //
//                                                                           //
// Create a map from vertex indices to segments, subfaces, and tetrahedra    //
// sharing at the same vertices.                                             //
//                                                                           //
// The map is stored in two arrays: 'idx2___list' and '___sperverlist', they //
// form a sparse matrix whose size is (n+1)x(n+1), where n is the number of  //
// segments, subfaces, or tetrahedra. 'idx2___list' contains row information //
// and '___sperverlist' contains all non-zero elements.  The i-th entry of   //
// 'idx2___list' is the starting position of i-th row's non-zero elements in //
// '___sperverlist'.  The number of elements of i-th row is (i+1)-th entry   //
// minus i-th entry of 'idx2___list'.                                        //
//                                                                           //
// NOTE: These two arrays will be created inside this routine, don't forget  //
// to free them after using.                                                 //
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

  // Create and initialize 'idx2facelist'.
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &idx2facelist);CHKERRQ(ierr);
  for (i = 0; i < m->points->items + 1; i++) idx2facelist[i] = 0;

  // Loop the set of subfaces once, counter the number of subfaces sharing each vertex.
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &shloop);CHKERRQ(ierr);
  while(shloop) {
    // Increment the number of sharing segments for each endpoint.
    for(i = 0; i < 3; i++) {
      j = pointmark(m, (point) shloop[3 + i]) - in->firstnumber;
      idx2facelist[j]++;
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &shloop);CHKERRQ(ierr);
  }

  // Calculate the total length of array 'facesperverlist'.
  j = idx2facelist[0];
  idx2facelist[0] = 0;  // Array starts from 0 element.
  for(i = 0; i < m->points->items; i++) {
    k = idx2facelist[i + 1];
    idx2facelist[i + 1] = idx2facelist[i] + j;
    j = k;
  }
  // The total length is in the last unit of idx2facelist.
  ierr = PetscMalloc(idx2facelist[i] * sizeof(shellface *), &facesperverlist);CHKERRQ(ierr);
  // Loop the set of segments again, set the info. of segments per vertex.
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
  // Contents in 'idx2facelist' are shifted, now shift them back.
  for(i = m->points->items - 1; i >= 0; i--) {
    idx2facelist[i + 1] = idx2facelist[i];
  }
  idx2facelist[0] = 0;
  *index2facelist = idx2facelist;
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
  // Initialize three coordinates.
  (*pnewpoint)[0] = 0.0;
  (*pnewpoint)[1] = 0.0;
  (*pnewpoint)[2] = 0.0;
  // Initialize the list of user-defined attributes.
  for(i = 0; i < in->numberofpointattributes; i++) {
    (*pnewpoint)[3 + i] = 0.0;
  }
  // Initialize the metric tensor.
  for(i = 0; i < m->sizeoftensor; i++) {
    (*pnewpoint)[m->pointmtrindex + i] = 0.0;
  }
  if (b->plc || b->refine) {
    // Initialize the point-to-simplex filed.
    setpoint2tet(m, *pnewpoint, PETSC_NULL);
    setpoint2sh(m, *pnewpoint, PETSC_NULL);
    setpoint2seg(m, *pnewpoint, PETSC_NULL);
    setpoint2ppt(m, *pnewpoint, PETSC_NULL);
    if (b->metric) {
      setpoint2bgmtet(m, *pnewpoint, PETSC_NULL);
    }
    if (m->checkpbcs) {
      // Initialize the other pointer to its pbc point.
      setpoint2pbcpt(m, *pnewpoint, PETSC_NULL);
    }
  }
  // Initialize the point marker (starting from in->firstnumber).
  ptmark = (int) m->points->items - (in->firstnumber == 1 ? 0 : 1);
  setpointmark(m, *pnewpoint, ptmark);
  // Initialize the point type.
  setpointtype(m, *pnewpoint, UNUSEDVERTEX);
  // Clear the point flags.
  puninfect(m, *pnewpoint);
  //punmarktest(*pnewpoint);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakeShellFace"
// makeshellface()    Create a new shellface with version zero. Used for both subfaces and seusegments. //
/* tetgenmesh::makeshellface() */
PetscErrorCode TetGenMeshMakeShellFace(TetGenMesh *m, MemoryPool *pool, face *newface)
{
  TetGenOpts    *b  = m->b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MemoryPoolAlloc(pool, (void **) &newface->sh);CHKERRQ(ierr);
  //Initialize the three adjoining subfaces to be the omnipresent subface.
  newface->sh[0] = (shellface) m->dummysh;
  newface->sh[1] = (shellface) m->dummysh;
  newface->sh[2] = (shellface) m->dummysh;
  // Three NULL vertices.
  newface->sh[3] = PETSC_NULL;
  newface->sh[4] = PETSC_NULL;
  newface->sh[5] = PETSC_NULL;
  // Initialize the two adjoining tetrahedra to be "outer space".
  newface->sh[6] = (shellface) m->dummytet;
  newface->sh[7] = (shellface) m->dummytet;
  // Initialize the three adjoining subsegments to be the omnipresent
  //   subsegments.
  newface->sh [8] = (shellface) m->dummysh;
  newface->sh [9] = (shellface) m->dummysh;
  newface->sh[10] = (shellface) m->dummysh;
  // Initialize the pointer to badface structure.
  newface->sh[11] = PETSC_NULL;
  if (b->quality && m->varconstraint) {
    // Initialize the maximum area bound.
    setareabound(m, newface, 0.0);
  }
  // Clear the infection and marktest bits.
  suninfect(m, newface);
  sunmarktest(newface);
  // Set the boundary marker to zero.
  setshellmark(m, newface, 0);
  // Set the type.
  setshelltype(m, newface, NSHARP);
  if (m->checkpbcs) {
    // Set the pbcgroup be ivalid.
    setshellpbcgroup(m, newface, -1);
  }
  // Initialize the version to be Zero.
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
  // Initialize the four adjoining tetrahedra to be "outer space".
  newtet->tet[0] = (tetrahedron) m->dummytet;
  newtet->tet[1] = (tetrahedron) m->dummytet;
  newtet->tet[2] = (tetrahedron) m->dummytet;
  newtet->tet[3] = (tetrahedron) m->dummytet;
  // Four NULL vertices.
  newtet->tet[4] = PETSC_NULL;
  newtet->tet[5] = PETSC_NULL;
  newtet->tet[6] = PETSC_NULL;
  newtet->tet[7] = PETSC_NULL;
  // Initialize the four adjoining subfaces to be the omnipresent subface.
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
  // Initialize the marker (for flags).
  setelemmarker(m, newtet->tet, 0);
  // Initialize the location and version to be Zero.
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
  } while (pointtype(m, newpoint) == DEADVERTEX);            // Skip dead ones.
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
  } while (!newshellface[3]);            // Skip dead ones.
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
  } while (!newsh->forg);            // Skip dead ones.
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
  } while (!newtetrahedron[7]);            // Skip dead ones.
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
  // Mark the point as dead. This  makes it possible to detect dead points
  //   when traversing the list of all points.
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
  // Set shellface's vertices to NULL. This makes it possible to detect dead
  //   shellfaces when traversing the list of all shellfaces.
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
  // Set badface's forg to NULL. This makes it possible to detect dead
  //   ones when traversing the list of all items.
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
  // Set tetrahedron's vertices to NULL. This makes it possible to detect
  //   dead tetrahedra when traversing the list of all tetrahedra.
  dyingtetrahedron[4] = PETSC_NULL;
  // dyingtetrahedron[5] = (tetrahedron) NULL;
  // dyingtetrahedron[6] = (tetrahedron) NULL;
  dyingtetrahedron[7] = PETSC_NULL;

  if (b->useshelles) {
    // Dealloc the space to subfaces/subsegments.
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

////                                                                       ////
////                                                                       ////
//// mempool_cxx //////////////////////////////////////////////////////////////

//// geom_cxx /////////////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCircumsphere"
// circumsphere()    Calculate the smallest circumsphere (center and radius) //
//                   of the given three or four points.                      //
//                                                                           //
// The circumsphere of four points (a tetrahedron) is unique if they are not //
// degenerate. If 'pd = NULL', the smallest circumsphere of three points is  //
// the diametral sphere of the triangle if they are not degenerate.          //
//                                                                           //
// Return TRUE if the input points are not degenerate and the circumcenter   //
// and circumradius are returned in 'cent' and 'radius' respectively if they //
// are not NULLs. Otherwise, return FALSE indicated the points are degenrate.//
/* tetgenmesh::circumsphere() */
PetscErrorCode TetGenMeshCircumsphere(TetGenMesh *m, PetscReal* pa, PetscReal* pb, PetscReal* pc, PetscReal* pd, PetscReal* cent, PetscReal* radius, PetscBool *notDegenerate)
{
  PetscReal A[4][4], rhs[4], D;
  int indx[4];

  PetscFunctionBegin;
  // Compute the coefficient matrix A (3x3).
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

  // Compute the right hand side vector b (3x1).
  rhs[0] = 0.5 * dot(A[0], A[0]);
  rhs[1] = 0.5 * dot(A[1], A[1]);
  if (pd) {
    rhs[2] = 0.5 * dot(A[2], A[2]);
  } else {
    rhs[2] = 0.0;
  }

  // Solve the 3 by 3 equations use LU decomposition with partial pivoting
  //   and backward and forward substitute..
  if (!lu_decmp(A, 3, indx, &D, 0)) {
    if (radius) *radius = 0.0;
    *notDegenerate = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  lu_solve(A, 3, indx, rhs, 0);
  if (cent) {
    cent[0] = pa[0] + rhs[0];
    cent[1] = pa[1] + rhs[1];
    cent[2] = pa[2] + rhs[2];
  }
  if (radius) {
    *radius = sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);
  }
  *notDegenerate = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFaceNormal"
// facenormal()    Calculate the normal of a face given by three points.     //
//                                                                           //
// In general, the face normal can be calculate by the cross product of any  //
// pair of the three edge vectors.  However, if the three points are nearly  //
// collinear, the rounding error may harm the result. To choose a good pair  //
// of vectors is helpful to reduce the error.                                //
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
#define __FUNCT__ "TetGenMeshFaceNormal"
// facenormal()    Calculate the normal of the face.                         //
//                                                                           //
// The normal of the face abc can be calculated by the cross product of 2 of //
// its 3 edge vectors.  A better choice of two edge vectors will reduce the  //
// numerical error during the calculation.  Burdakov proved that the optimal //
// basis problem is equivalent to the minimum spanning tree problem with the //
// edge length be the functional, see Burdakov, "A greedy algorithm for the  //
// optimal basis problem", BIT 37:3 (1997), 591-599. If 'pivot' > 0, the two //
// short edges in abc are chosen for the calculation.                        //
/* tetgenmesh::facenormal2() */
PetscErrorCode TetGenMeshFaceNormal2(TetGenMesh *m, point pa, point pb, point pc, PetscReal *n, int pivot)
{
  PetscReal v1[3], v2[3], v3[3], *pv1, *pv2;
  PetscReal L1, L2, L3;

  PetscFunctionBegin;
  v1[0] = pb[0] - pa[0];  // edge vector v1: a->b
  v1[1] = pb[1] - pa[1];
  v1[2] = pb[2] - pa[2];
  v2[0] = pa[0] - pc[0];  // edge vector v2: c->a
  v2[1] = pa[1] - pc[1];
  v2[2] = pa[2] - pc[2];

  // Default, normal is calculated by: v1 x (-v2) (see Fig. fnormal).
  if (pivot > 0) {
    // Choose edge vectors by Burdakov's algorithm.
    v3[0] = pc[0] - pb[0];  // edge vector v3: b->c
    v3[1] = pc[1] - pb[1];
    v3[2] = pc[2] - pb[2];
    L1 = DOT(v1, v1);
    L2 = DOT(v2, v2);
    L3 = DOT(v3, v3);
    // Sort the three edge lengths.
    if (L1 < L2) {
      if (L2 < L3) {
        pv1 = v1; pv2 = v2; // n = v1 x (-v2).
      } else {
        pv1 = v3; pv2 = v1; // n = v3 x (-v1).
      }
    } else {
      if (L1 < L3) {
        pv1 = v1; pv2 = v2; // n = v1 x (-v2).
      } else {
        pv1 = v2; pv2 = v3; // n = v2 x (-v3).
      }
    }
  } else {
    pv1 = v1; pv2 = v2; // n = v1 x (-v2).
  }

  // Calculate the face normal.
  CROSS(pv1, pv2, n);
  // Inverse the direction;
  n[0] = -n[0];
  n[1] = -n[1];
  n[2] = -n[2];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshProjPt2Face"
// projpt2face()    Return the projection point from a point to a face.      //
/* tetgenmesh::projpt2face() */
PetscErrorCode TetGenMeshProjPt2Face(TetGenMesh *m, PetscReal* p, PetscReal* f1, PetscReal* f2, PetscReal* f3, PetscReal* prj)
{
  PetscReal fnormal[3], v1[3];
  PetscReal len, dist;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Get the unit face normal.
  // facenormal(f1, f2, f3, fnormal, &len);
  ierr = TetGenMeshFaceNormal2(m, f1, f2, f3, fnormal, 1);CHKERRQ(ierr);
  len = sqrt(fnormal[0]*fnormal[0] + fnormal[1]*fnormal[1] + fnormal[2]*fnormal[2]);
  fnormal[0] /= len;
  fnormal[1] /= len;
  fnormal[2] /= len;
  // Get the vector v1 = |p - f1|.
  v1[0] = p[0] - f1[0];
  v1[1] = p[1] - f1[1];
  v1[2] = p[2] - f1[2];
  // Get the project distance.
  dist = dot(fnormal, v1);

  // Get the project point.
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
  point U[3], V[3];  // The permuted vectors of points.
  int pu[3], pv[3];  // The original positions of points.
  PetscReal sA, sB, sC;
  PetscReal s1, s2, s3, s4;
  int z1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!R) {
    PetscReal n[3], len;
    // Calculate a lift point, saved in dummypoint.
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

  // Test A's, B's, and C's orientations wrt plane PQR.
  sA = orient3d(P, Q, R, A);
  sB = orient3d(P, Q, R, B);
  sC = orient3d(P, Q, R, C);
  m->orient3dcount+=3;

  PetscInfo6(b->in, "      Tri-edge-2d (%d %d %d)-(%d %d)-(%d) (%c%c%c)", pointmark(m, A), pointmark(m, B), pointmark(m, C), pointmark(m, P), pointmark(m, Q), pointmark(m, R));
  PetscInfo3(b->in, "        (%c%c%c)", sA > 0 ? '+' : (sA < 0 ? '-' : '0'), sB>0 ? '+' : (sB<0 ? '-' : '0'), sC>0 ? '+' : (sC<0 ? '-' : '0'));
  // triedgcopcount++;

  if (sA < 0) {
    if (sB < 0) {
      if (sC < 0) { // (---).
        if (isIntersect) {*isIntersect = 0;}
        PetscFunctionReturn(0);
      } else {
        if (sC > 0) { // (--+).
          // All points are in the right positions.
          SETVECTOR3(U, A, B, C);  // I3
          SETVECTOR3(V, P, Q, R);  // I2
          SETVECTOR3(pu, 0, 1, 2);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 0;
        } else { // (--0).
          SETVECTOR3(U, A, B, C);  // I3
          SETVECTOR3(V, P, Q, R);  // I2
          SETVECTOR3(pu, 0, 1, 2);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 1;
        }
      }
    } else {
      if (sB > 0) {
        if (sC < 0) { // (-+-).
          SETVECTOR3(U, C, A, B);  // PT = ST
          SETVECTOR3(V, P, Q, R);  // I2
          SETVECTOR3(pu, 2, 0, 1);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 0;
        } else {
          if (sC > 0) { // (-++).
            SETVECTOR3(U, B, C, A);  // PT = ST x ST
            SETVECTOR3(V, Q, P, R);  // PL = SL
            SETVECTOR3(pu, 1, 2, 0);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 0;
          } else { // (-+0).
            SETVECTOR3(U, C, A, B);  // PT = ST
            SETVECTOR3(V, P, Q, R);  // I2
            SETVECTOR3(pu, 2, 0, 1);
            SETVECTOR3(pv, 0, 1, 2);
            z1 = 2;
          }
        }
      } else {
        if (sC < 0) { // (-0-).
          SETVECTOR3(U, C, A, B);  // PT = ST
          SETVECTOR3(V, P, Q, R);  // I2
          SETVECTOR3(pu, 2, 0, 1);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 1;
        } else {
          if (sC > 0) { // (-0+).
            SETVECTOR3(U, B, C, A);  // PT = ST x ST
            SETVECTOR3(V, Q, P, R);  // PL = SL
            SETVECTOR3(pu, 1, 2, 0);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 2;
          } else { // (-00).
            SETVECTOR3(U, B, C, A);  // PT = ST x ST
            SETVECTOR3(V, Q, P, R);  // PL = SL
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
        if (sC < 0) { // (+--).
          SETVECTOR3(U, B, C, A);  // PT = ST x ST
          SETVECTOR3(V, P, Q, R);  // I2
          SETVECTOR3(pu, 1, 2, 0);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 0;
        } else {
          if (sC > 0) { // (+-+).
            SETVECTOR3(U, C, A, B);  // PT = ST
            SETVECTOR3(V, Q, P, R);  // PL = SL
            SETVECTOR3(pu, 2, 0, 1);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 0;
          } else { // (+-0).
            SETVECTOR3(U, C, A, B);  // PT = ST
            SETVECTOR3(V, Q, P, R);  // PL = SL
            SETVECTOR3(pu, 2, 0, 1);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 2;
          }
        }
      } else {
        if (sB > 0) {
          if (sC < 0) { // (++-).
            SETVECTOR3(U, A, B, C);  // I3
            SETVECTOR3(V, Q, P, R);  // PL = SL
            SETVECTOR3(pu, 0, 1, 2);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 0;
          } else {
            if (sC > 0) { // (+++).
              if (isIntersect) {*isIntersect = 0;}
              PetscFunctionReturn(0);
            } else { // (++0).
              SETVECTOR3(U, A, B, C);  // I3
              SETVECTOR3(V, Q, P, R);  // PL = SL
              SETVECTOR3(pu, 0, 1, 2);
              SETVECTOR3(pv, 1, 0, 2);
              z1 = 1;
            }
          }
        } else { // (+0#)
          if (sC < 0) { // (+0-).
            SETVECTOR3(U, B, C, A);  // PT = ST x ST
            SETVECTOR3(V, P, Q, R);  // I2
            SETVECTOR3(pu, 1, 2, 0);
            SETVECTOR3(pv, 0, 1, 2);
            z1 = 2;
          } else {
            if (sC > 0) { // (+0+).
              SETVECTOR3(U, C, A, B);  // PT = ST
              SETVECTOR3(V, Q, P, R);  // PL = SL
              SETVECTOR3(pu, 2, 0, 1);
              SETVECTOR3(pv, 1, 0, 2);
              z1 = 1;
            } else { // (+00).
              SETVECTOR3(U, B, C, A);  // PT = ST x ST
              SETVECTOR3(V, P, Q, R);  // I2
              SETVECTOR3(pu, 1, 2, 0);
              SETVECTOR3(pv, 0, 1, 2);
              z1 = 3;
            }
          }
        }
      }
    } else {
      if (sB < 0) {
        if (sC < 0) { // (0--).
          SETVECTOR3(U, B, C, A);  // PT = ST x ST
          SETVECTOR3(V, P, Q, R);  // I2
          SETVECTOR3(pu, 1, 2, 0);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 1;
        } else {
          if (sC > 0) { // (0-+).
            SETVECTOR3(U, A, B, C);  // I3
            SETVECTOR3(V, P, Q, R);  // I2
            SETVECTOR3(pu, 0, 1, 2);
            SETVECTOR3(pv, 0, 1, 2);
            z1 = 2;
          } else { // (0-0).
            SETVECTOR3(U, C, A, B);  // PT = ST
            SETVECTOR3(V, Q, P, R);  // PL = SL
            SETVECTOR3(pu, 2, 0, 1);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 3;
          }
        }
      } else {
        if (sB > 0) {
          if (sC < 0) { // (0+-).
            SETVECTOR3(U, A, B, C);  // I3
            SETVECTOR3(V, Q, P, R);  // PL = SL
            SETVECTOR3(pu, 0, 1, 2);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 2;
          } else {
            if (sC > 0) { // (0++).
              SETVECTOR3(U, B, C, A);  // PT = ST x ST
              SETVECTOR3(V, Q, P, R);  // PL = SL
              SETVECTOR3(pu, 1, 2, 0);
              SETVECTOR3(pv, 1, 0, 2);
              z1 = 1;
            } else { // (0+0).
              SETVECTOR3(U, C, A, B);  // PT = ST
              SETVECTOR3(V, P, Q, R);  // I2
              SETVECTOR3(pu, 2, 0, 1);
              SETVECTOR3(pv, 0, 1, 2);
              z1 = 3;
            }
          }
        } else { // (00#)
          if (sC < 0) { // (00-).
            SETVECTOR3(U, A, B, C);  // I3
            SETVECTOR3(V, Q, P, R);  // PL = SL
            SETVECTOR3(pu, 0, 1, 2);
            SETVECTOR3(pv, 1, 0, 2);
            z1 = 3;
          } else {
            if (sC > 0) { // (00+).
              SETVECTOR3(U, A, B, C);  // I3
              SETVECTOR3(V, P, Q, R);  // I2
              SETVECTOR3(pu, 0, 1, 2);
              SETVECTOR3(pv, 0, 1, 2);
              z1 = 3;
            } else { // (000)
              // Not possible unless ABC is degenerate.
              z1 = 4;
            }
          }
        }
      }
    }
  }

  s1 = orient3d(U[0], U[2], R, V[1]);  // A, C, R, Q
  s2 = orient3d(U[1], U[2], R, V[0]);  // B, C, R, P
  m->orient3dcount+=2;

  PetscInfo7(b->in, "      Tri-edge-2d (%d %d %d)-(%d %d %d) (%d) (%c%c)\n", pointmark(m, U[0]), pointmark(m, U[1]), pointmark(m, U[2]), pointmark(m, V[0]),
             pointmark(m, V[1]), pointmark(m, V[2]), z1);
  PetscInfo2(b->in, "        (%c%c)\n", s1>0 ? '+' : (s1<0 ? '-' : '0'), s2>0 ? '+' : (s2<0 ? '-' : '0'));
  if (z1 == 4) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}

  if (s1 > 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }
  if (s2 < 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }

  if (level == 0) {
    if (isIntersect) {*isIntersect = 1;} // They are intersected.
    PetscFunctionReturn(0);
  }

  if (z1 == 1) {
    if (s1 == 0) {  // (0###)
      // C = Q.
      types[0] = (int) SHAREVERTEX;
      pos[0] = pu[2]; // C
      pos[1] = pv[1]; // Q
      types[1] = (int) DISJOINT;
    } else {
      if (s2 == 0) { // (#0##)
        // C = P.
        types[0] = (int) SHAREVERTEX;
        pos[0] = pu[2]; // C
        pos[1] = pv[0]; // P
        types[1] = (int) DISJOINT;
      } else { // (-+##)
        // C in [P, Q].
        types[0] = (int) INTERVERT;
        pos[0] = pu[2]; // C
        pos[1] = pv[0]; // [P, Q]
        types[1] = (int) DISJOINT;
      }
    }
    if (isIntersect) {*isIntersect = 1;}
    PetscFunctionReturn(0);
  }

  s3 = orient3d(U[0], U[2], R, V[0]);  // A, C, R, P
  s4 = orient3d(U[1], U[2], R, V[1]);  // B, C, R, Q
  m->orient3dcount+=2;

  if (z1 == 0) {  // (tritri-03)
    if (s1 < 0) {
      if (s3 > 0) {
        if (s2 <= 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        if (s4 > 0) {
          // [P, Q] overlaps [k, l] (-+++).
          types[0] = (int) INTEREDGE;
          pos[0] = pu[2]; // [C, A]
          pos[1] = pv[0]; // [P, Q]
          types[1] = (int) TOUCHFACE;
          pos[2] = 3;     // [A, B, C]
          pos[3] = pv[1]; // Q
        } else {
          if (s4 == 0) {
            // Q = l, [P, Q] contains [k, l] (-++0).
            types[0] = (int) INTEREDGE;
            pos[0] = pu[2]; // [C, A]
            pos[1] = pv[0]; // [P, Q]
            types[1] = (int) TOUCHEDGE;
            pos[2] = pu[1]; // [B, C]
            pos[3] = pv[1]; // Q
          } else { // s4 < 0
            // [P, Q] contains [k, l] (-++-).
            types[0] = (int) INTEREDGE;
            pos[0] = pu[2]; // [C, A]
            pos[1] = pv[0]; // [P, Q]
            types[1] = (int) INTEREDGE;
            pos[2] = pu[1]; // [B, C]
            pos[3] = pv[0]; // [P, Q]
          }
        }
      } else {
        if (s3 == 0) {
          if (s2 <= 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
          if (s4 > 0) {
            // P = k, [P, Q] in [k, l] (-+0+).
            types[0] = (int) TOUCHEDGE;
            pos[0] = pu[2]; // [C, A]
            pos[1] = pv[0]; // P
            types[1] = (int) TOUCHFACE;
            pos[2] = 3;     // [A, B, C]
            pos[3] = pv[1]; // Q
          } else {
            if (s4 == 0) {
              // [P, Q] = [k, l] (-+00).
              types[0] = (int) TOUCHEDGE;
              pos[0] = pu[2]; // [C, A]
              pos[1] = pv[0]; // P
              types[1] = (int) TOUCHEDGE;
              pos[2] = pu[1]; // [B, C]
              pos[3] = pv[1]; // Q
            } else {
              // P = k, [P, Q] contains [k, l] (-+0-).
              types[0] = (int) TOUCHEDGE;
              pos[0] = pu[2]; // [C, A]
              pos[1] = pv[0]; // P
              types[1] = (int) INTEREDGE;
              pos[2] = pu[1]; // [B, C]
              pos[3] = pv[0]; // [P, Q]
            }
          }
        } else { // s3 < 0
          if (s2 > 0) {
            if (s4 > 0) {
              // [P, Q] in [k, l] (-+-+).
              types[0] = (int) TOUCHFACE;
              pos[0] = 3;     // [A, B, C]
              pos[1] = pv[0]; // P
              types[1] = (int) TOUCHFACE;
              pos[2] = 3;     // [A, B, C]
              pos[3] = pv[1]; // Q
            } else {
              if (s4 == 0) {
                // Q = l, [P, Q] in [k, l] (-+-0).
                types[0] = (int) TOUCHFACE;
                pos[0] = 3;     // [A, B, C]
                pos[1] = pv[0]; // P
                types[1] = (int) TOUCHEDGE;
                pos[2] = pu[1]; // [B, C]
                pos[3] = pv[1]; // Q
              } else { // s4 < 0
                // [P, Q] overlaps [k, l] (-+--).
                types[0] = (int) TOUCHFACE;
                pos[0] = 3;     // [A, B, C]
                pos[1] = pv[0]; // P
                types[1] = (int) INTEREDGE;
                pos[2] = pu[1]; // [B, C]
                pos[3] = pv[0]; // [P, Q]
              }
            }
          } else { // s2 == 0
            // P = l (#0##).
            types[0] = (int) TOUCHEDGE;
            pos[0] = pu[1]; // [B, C]
            pos[1] = pv[0]; // P
            types[1] = (int) DISJOINT;
          }
        }
      }
    } else { // s1 == 0
      // Q = k (0####)
      types[0] = (int) TOUCHEDGE;
      pos[0] = pu[2]; // [C, A]
      pos[1] = pv[1]; // Q
      types[1] = (int) DISJOINT;
    }
  } else if (z1 == 2) {  // (tritri-23)
    if (s1 < 0) {
      if (s3 > 0) {
        if (s2 <= 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        if (s4 > 0) {
          // [P, Q] overlaps [A, l] (-+++).
          types[0] = (int) INTERVERT;
          pos[0] = pu[0]; // A
          pos[1] = pv[0]; // [P, Q]
          types[1] = (int) TOUCHFACE;
          pos[2] = 3;     // [A, B, C]
          pos[3] = pv[1]; // Q
        } else {
          if (s4 == 0) {
            // Q = l, [P, Q] contains [A, l] (-++0).
            types[0] = (int) INTERVERT;
            pos[0] = pu[0]; // A
            pos[1] = pv[0]; // [P, Q]
            types[1] = (int) TOUCHEDGE;
            pos[2] = pu[1]; // [B, C]
            pos[3] = pv[1]; // Q
          } else { // s4 < 0
            // [P, Q] contains [A, l] (-++-).
            types[0] = (int) INTERVERT;
            pos[0] = pu[0]; // A
            pos[1] = pv[0]; // [P, Q]
            types[1] = (int) INTEREDGE;
            pos[2] = pu[1]; // [B, C]
            pos[3] = pv[0]; // [P, Q]
          }
        }
      } else {
        if (s3 == 0) {
          if (s2 <= 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
          if (s4 > 0) {
            // P = A, [P, Q] in [A, l] (-+0+).
            types[0] = (int) SHAREVERTEX;
            pos[0] = pu[0]; // A
            pos[1] = pv[0]; // P
            types[1] = (int) TOUCHFACE;
            pos[2] = 3;     // [A, B, C]
            pos[3] = pv[1]; // Q
          } else {
            if (s4 == 0) {
              // [P, Q] = [A, l] (-+00).
              types[0] = (int) SHAREVERTEX;
              pos[0] = pu[0]; // A
              pos[1] = pv[0]; // P
              types[1] = (int) TOUCHEDGE;
              pos[2] = pu[1]; // [B, C]
              pos[3] = pv[1]; // Q
            } else { // s4 < 0
              // Q = l, [P, Q] in [A, l] (-+0-).
              types[0] = (int) SHAREVERTEX;
              pos[0] = pu[0]; // A
              pos[1] = pv[0]; // P
              types[1] = (int) INTEREDGE;
              pos[2] = pu[1]; // [B, C]
              pos[3] = pv[0]; // [P, Q]
            }
          }
        } else { // s3 < 0
          if (s2 > 0) {
            if (s4 > 0) {
              // [P, Q] in [A, l] (-+-+).
              types[0] = (int) TOUCHFACE;
              pos[0] = 3;     // [A, B, C]
              pos[1] = pv[0]; // P
              types[0] = (int) TOUCHFACE;
              pos[0] = 3;     // [A, B, C]
              pos[1] = pv[1]; // Q
            } else {
              if (s4 == 0) {
                // Q = l, [P, Q] in [A, l] (-+-0).
                types[0] = (int) TOUCHFACE;
                pos[0] = 3;     // [A, B, C]
                pos[1] = pv[0]; // P
                types[0] = (int) TOUCHEDGE;
                pos[0] = pu[1]; // [B, C]
                pos[1] = pv[1]; // Q
              } else { // s4 < 0
                // [P, Q] overlaps [A, l] (-+--).
                types[0] = (int) TOUCHFACE;
                pos[0] = 3;     // [A, B, C]
                pos[1] = pv[0]; // P
                types[0] = (int) INTEREDGE;
                pos[0] = pu[1]; // [B, C]
                pos[1] = pv[0]; // [P, Q]
              }
            }
          } else { // s2 == 0
            // P = l (#0##).
            types[0] = (int) TOUCHEDGE;
            pos[0] = pu[1]; // [B, C]
            pos[1] = pv[0]; // P
            types[1] = (int) DISJOINT;
          }
        }
      }
    } else { // s1 == 0
      // Q = A (0###).
      types[0] = (int) SHAREVERTEX;
      pos[0] = pu[0]; // A
      pos[1] = pv[1]; // Q
      types[1] = (int) DISJOINT;
    }
  } else if (z1 == 3) {  // (tritri-33)
    if (s1 < 0) {
      if (s3 > 0) {
        if (s2 <= 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        if (s4 > 0) {
          // [P, Q] overlaps [A, B] (-+++).
          types[0] = (int) INTERVERT;
          pos[0] = pu[0]; // A
          pos[1] = pv[0]; // [P, Q]
          types[1] = (int) TOUCHEDGE;
          pos[2] = pu[0]; // [A, B]
          pos[3] = pv[1]; // Q
        } else {
          if (s4 == 0) {
            // Q = B, [P, Q] contains [A, B] (-++0).
            types[0] = (int) INTERVERT;
            pos[0] = pu[0]; // A
            pos[1] = pv[0]; // [P, Q]
            types[1] = (int) SHAREVERTEX;
            pos[2] = pu[1]; // B
            pos[3] = pv[1]; // Q
          } else { // s4 < 0
            // [P, Q] contains [A, B] (-++-).
            types[0] = (int) INTERVERT;
            pos[0] = pu[0]; // A
            pos[1] = pv[0]; // [P, Q]
            types[1] = (int) INTERVERT;
            pos[2] = pu[1]; // B
            pos[3] = pv[0]; // [P, Q]
          }
        }
      } else {
        if (s3 == 0) {
          if (s2 <= 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
          if (s4 > 0) {
            // P = A, [P, Q] in [A, B] (-+0+).
            types[0] = (int) SHAREVERTEX;
            pos[0] = pu[0]; // A
            pos[1] = pv[0]; // P
            types[1] = (int) TOUCHEDGE;
            pos[2] = pu[0]; // [A, B]
            pos[3] = pv[1]; // Q
          } else {
            if (s4 == 0) {
              // [P, Q] = [A, B] (-+00).
              types[0] = (int) SHAREEDGE;
              pos[0] = pu[0]; // [A, B]
              pos[1] = pv[0]; // [P, Q]
              types[1] = (int) DISJOINT;
            } else { // s4 < 0
              // P= A, [P, Q] in [A, B] (-+0-).
              types[0] = (int) SHAREVERTEX;
              pos[0] = pu[0]; // A
              pos[1] = pv[0]; // P
              types[1] = (int) INTERVERT;
              pos[2] = pu[1]; // B
              pos[3] = pv[0]; // [P, Q]
            }
          }
        } else { // s3 < 0
          if (s2 > 0) {
            if (s4 > 0) {
              // [P, Q] in [A, B] (-+-+).
              types[0] = (int) TOUCHEDGE;
              pos[0] = pu[0]; // [A, B]
              pos[1] = pv[0]; // P
              types[1] = (int) TOUCHEDGE;
              pos[2] = pu[0]; // [A, B]
              pos[3] = pv[1]; // Q
            } else {
              if (s4 == 0) {
                // Q = B, [P, Q] in [A, B] (-+-0).
                types[0] = (int) TOUCHEDGE;
                pos[0] = pu[0]; // [A, B]
                pos[1] = pv[0]; // P
                types[1] = (int) SHAREVERTEX;
                pos[2] = pu[1]; // B
                pos[3] = pv[1]; // Q
              } else { // s4 < 0
                // [P, Q] overlaps [A, B] (-+--).
                types[0] = (int) TOUCHEDGE;
                pos[0] = pu[0]; // [A, B]
                pos[1] = pv[0]; // P
                types[1] = (int) INTERVERT;
                pos[2] = pu[1]; // B
                pos[3] = pv[0]; // [P, Q]
              }
            }
          } else { // s2 == 0
            // P = B (#0##).
            types[0] = (int) SHAREVERTEX;
            pos[0] = pu[1]; // B
            pos[1] = pv[0]; // P
            types[1] = (int) DISJOINT;
          }
        }
      }
    } else { // s1 == 0
      // Q = A (0###).
      types[0] = (int) SHAREVERTEX;
      pos[0] = pu[0]; // A
      pos[1] = pv[1]; // Q
      types[1] = (int) DISJOINT;
    }
  }
  if (isIntersect) {*isIntersect = 1;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTriEdgeTest"
// tri_edge_test()    Triangle-edge intersection test.                       //
//                                                                           //
// This routine takes a triangle T (with vertices A, B, C) and an edge E (P, //
// Q) in 3D, and tests if they intersect each other.  Return 1 if they are   //
// intersected, i.e., T \cap E is not empty, otherwise, return 0.            //
//                                                                           //
// If the point 'R' is not NULL, it lies strictly above the plane defined by //
// A, B, C. It is used in test when T and E are coplanar.                    //
//                                                                           //
// If T1 and T2 intersect each other (return 1), they may intersect in diff- //
// erent ways. If 'level' > 0, their intersection type will be reported in   //
// combinations of 'types' and 'pos'.                                        //
/* tetgenmesh::tri_edge_test() */
PetscErrorCode TetGenMeshTriEdgeTest(TetGenMesh *m, point A, point B, point C, point P, point Q, point R, int level, int *types, int *pos, int *isIntersect)
{
  TetGenOpts    *b = m->b;
  point U[3], V[3]; //, Ptmp;
  int pu[3], pv[3]; //, itmp;
  PetscReal sP, sQ, s1, s2, s3;
  int z1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Test the locations of P and Q with respect to ABC.
  sP = orient3d(A, B, C, P);
  sQ = orient3d(A, B, C, Q);
  m->orient3dcount+=2;

  PetscInfo7(b->in, "      Tri-edge (%d %d %d)-(%d %d) (%c%c).\n", pointmark(m, A),
             pointmark(m, B), pointmark(m, C), pointmark(m, P), pointmark(m, Q),
             sP>0 ? '+' : (sP<0 ? '-' : '0'), sQ>0 ? '+' : (sQ<0 ? '-' : '0'));
  // triedgcount++;

  if (sP < 0) {
    if (sQ < 0) { // (--) disjoint
      if (isIntersect) {*isIntersect = 0;}
      PetscFunctionReturn(0);
    } else {
      if (sQ > 0) { // (-+)
        SETVECTOR3(U, A, B, C);
        SETVECTOR3(V, P, Q, R);
        SETVECTOR3(pu, 0, 1, 2);
        SETVECTOR3(pv, 0, 1, 2);
        z1 = 0;
      } else { // (-0)
        SETVECTOR3(U, A, B, C);
        SETVECTOR3(V, P, Q, R);
        SETVECTOR3(pu, 0, 1, 2);
        SETVECTOR3(pv, 0, 1, 2);
        z1 = 1;
      }
    }
  } else {
    if (sP > 0) { // (+-)
      if (sQ < 0) {
        SETVECTOR3(U, A, B, C);
        SETVECTOR3(V, Q, P, R);  // P and Q are flipped.
        SETVECTOR3(pu, 0, 1, 2);
        SETVECTOR3(pv, 1, 0, 2);
        z1 = 0;
      } else {
        if (sQ > 0) { // (++) disjoint
          if (isIntersect) {*isIntersect = 0;}
          PetscFunctionReturn(0);
        } else { // (+0)
          SETVECTOR3(U, B, A, C); // A and B are flipped.
          SETVECTOR3(V, P, Q, R);
          SETVECTOR3(pu, 1, 0, 2);
          SETVECTOR3(pv, 0, 1, 2);
          z1 = 1;
        }
      }
    } else { // sP == 0
      if (sQ < 0) { // (0-)
        SETVECTOR3(U, A, B, C);
        SETVECTOR3(V, Q, P, R);  // P and Q are flipped.
        SETVECTOR3(pu, 0, 1, 2);
        SETVECTOR3(pv, 1, 0, 2);
        z1 = 1;
      } else {
        if (sQ > 0) { // (0+)
          SETVECTOR3(U, B, A, C);  // A and B are flipped.
          SETVECTOR3(V, Q, P, R);  // P and Q are flipped.
          SETVECTOR3(pu, 1, 0, 2);
          SETVECTOR3(pv, 1, 0, 2);
          z1 = 1;
        } else { // (00)
          // A, B, C, P, and Q are coplanar.
          z1 = 2;
        }
      }
    }
  }

  if (z1 == 2) {
    int isInter;
    // The triangle and the edge are coplanar.
    ierr = TetGenMeshTriEdge2D(m, A, B, C, P, Q, R, level, types, pos, &isInter);CHKERRQ(ierr);
    if (isIntersect) {*isIntersect = isInter;}
    PetscFunctionReturn(0);
  }

  s1 = orient3d(U[0], U[1], V[0], V[1]); m->orient3dcount++;
  if (s1 < 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }

  s2 = orient3d(U[1], U[2], V[0], V[1]); m->orient3dcount++;
  if (s2 < 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }

  s3 = orient3d(U[2], U[0], V[0], V[1]); m->orient3dcount++;
  if (s3 < 0) {
    if (isIntersect) {*isIntersect = 0;}
    PetscFunctionReturn(0);
  }

  PetscInfo5(b->in, "      Tri-edge (%d %d %d)-(%d %d).\n", pointmark(m, U[0]), pointmark(m, U[1]), pointmark(m, U[2]), pointmark(m, V[0]), pointmark(m, V[1]));
  PetscInfo3(b->in, "        (%c%c%c).\n", s1>0 ? '+' : (s1<0 ? '-' : '0'), s2>0 ? '+' : (s2<0 ? '-' : '0'), s3>0 ? '+' : (s3<0 ? '-' : '0'));

  if (level == 0) {
    if (isIntersect) {*isIntersect = 1;} // The are intersected.
    PetscFunctionReturn(0);
  }

  types[1] = (int) DISJOINT; // No second intersection point.

  if (z1 == 0) {
    if (s1 > 0) {
      if (s2 > 0) {
        if (s3 > 0) { // (+++)
          // [P, Q] passes interior of [A, B, C].
          types[0] = (int) INTERFACE;
          pos[0] = 3;  // interior of [A, B, C]
          pos[1] = 0;  // [P, Q]
        } else { // s3 == 0 (++0)
          // [P, Q] intersects [C, A].
          types[0] = (int) INTEREDGE;
          pos[0] = pu[2];  // [C, A]
          pos[1] = 0;  // [P, Q]
        }
      } else { // s2 == 0
        if (s3 > 0) { // (+0+)
          // [P, Q] intersects [B, C].
          types[0] = (int) INTEREDGE;
          pos[0] = pu[1];  // [B, C]
          pos[1] = 0;  // [P, Q]
        } else { // s3 == 0 (+00)
          // [P, Q] passes C.
          types[0] = (int) INTERVERT;
          pos[0] = pu[2];  // C
          pos[1] = 0;  // [P, Q]
        }
      }
    } else { // s1 == 0
      if (s2 > 0) {
        if (s3 > 0) { // (0++)
          // [P, Q] intersects [A, B].
          types[0] = (int) INTEREDGE;
          pos[0] = pu[0];  // [A, B]
          pos[1] = 0;  // [P, Q]
        } else { // s3 == 0 (0+0)
          // [P, Q] passes A.
          types[0] = (int) INTERVERT;
          pos[0] = pu[0];  // A
          pos[1] = 0;  // [P, Q]
        }
      } else { // s2 == 0
        if (s3 > 0) { // (00+)
          // [P, Q] passes B.
          types[0] = (int) INTERVERT;
          pos[0] = pu[1];  // B
          pos[1] = 0;  // [P, Q]
        } else { // s3 == 0 (000)
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Impossible");
        }
      }
    }
  } else { // z1 == 1
    if (s1 > 0) {
      if (s2 > 0) {
        if (s3 > 0) { // (+++)
          // Q lies in [A, B, C].
          types[0] = (int) TOUCHFACE;
          pos[0] = 0; // [A, B, C]
          pos[1] = pv[1]; // Q
        } else { // s3 == 0 (++0)
          // Q lies on [C, A].
          types[0] = (int) TOUCHEDGE;
          pos[0] = pu[2]; // [C, A]
          pos[1] = pv[1]; // Q
        }
      } else { // s2 == 0
        if (s3 > 0) { // (+0+)
          // Q lies on [B, C].
          types[0] = (int) TOUCHEDGE;
          pos[0] = pu[1]; // [B, C]
          pos[1] = pv[1]; // Q
        } else { // s3 == 0 (+00)
          // Q = C.
          types[0] = (int) SHAREVERTEX;
          pos[0] = pu[2]; // C
          pos[1] = pv[1]; // Q
        }
      }
    } else { // s1 == 0
      if (s2 > 0) {
        if (s3 > 0) { // (0++)
          // Q lies on [A, B].
          types[0] = (int) TOUCHEDGE;
          pos[0] = pu[0]; // [A, B]
          pos[1] = pv[1]; // Q
        } else { // s3 == 0 (0+0)
          // Q = A.
          types[0] = (int) SHAREVERTEX;
          pos[0] = pu[0]; // A
          pos[1] = pv[1]; // Q
        }
      } else { // s2 == 0
        if (s3 > 0) { // (00+)
          // Q = B.
          types[0] = (int) SHAREVERTEX;
          pos[0] = pu[1]; // B
          pos[1] = pv[1]; // Q
        } else { // s3 == 0 (000)
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Impossible");
        }
      }
    }
  }

  if (isIntersect) {*isIntersect = 1;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInCircle3D"
// incircle3d()    3D in-circle test.                                        //
//                                                                           //
// Return a negative value if pd is inside the circumcircle of the triangle  //
// pa, pb, and pc.                                                           //
/* tetgenmesh::incirlce3d() */
PetscErrorCode TetGenMeshInCircle3D(TetGenMesh *m, point pa, point pb, point pc, point pd, PetscReal *signTest)
{
  TetGenOpts    *b = m->b;
  PetscReal area2[2], n1[3], n2[3], c[3];
  PetscReal sign, r, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Calculate the areas of the two triangles [a, b, c] and [b, a, d].
  ierr = TetGenMeshFaceNormal2(m, pa, pb, pc, n1, 1);CHKERRQ(ierr);
  area2[0] = DOT(n1, n1);
  ierr = TetGenMeshFaceNormal2(m, pb, pa, pd, n2, 1);CHKERRQ(ierr);
  area2[1] = DOT(n2, n2);

  if (area2[0] > area2[1]) {
    // Choose [a, b, c] as the base triangle.
    ierr = TetGenMeshCircumsphere(m, pa, pb, pc, PETSC_NULL, c, &r, PETSC_NULL);CHKERRQ(ierr);
    d = DIST(c, pd);
  } else {
    // Choose [b, a, d] as the base triangle.
    if (area2[1] > 0) {
      ierr = TetGenMeshCircumsphere(m, pb, pa, pd, PETSC_NULL, c, &r, PETSC_NULL);CHKERRQ(ierr);
      d = DIST(c, pc);
    } else {
      // The four points are collinear. This case only happens on the boundary.
      return 0; // Return "not inside".
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
// insphere_s()    Insphere test with symbolic perturbation.                 //
//                                                                           //
// Given four points pa, pb, pc, and pd, test if the point pe lies inside or //
// outside the circumscirbed sphere of the four points.  Here we assume that //
// the orientation of the sequence {pa, pb, pc, pd} is negative (NOT zero),  //
// i.e., pd lies at the negative side of the plane defined by pa, pb, and pc.//
//                                                                           //
// Return a positive value (> 0) if pe lies outside, a negative value (< 0)  //
// if pe lies inside the sphere, the returned value will not be zero.        //
/* tetgenmesh::insphere_s() */
PetscErrorCode TetGenMeshInSphereS(TetGenMesh *m, PetscReal* pa, PetscReal* pb, PetscReal* pc, PetscReal* pd, PetscReal* pe, PetscReal *isOutside)
{
  PetscReal sign;
  // Symbolic perturbation.
  point pt[5], swappt;
  PetscReal oriA, oriB;
  int swaps, count;
  int n, i;

  PetscFunctionBegin;
  m->inspherecount++;
  sign = insphere(pa, pb, pc, pd, pe);
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

  // Sort the five points such that their indices are in the increasing
  //   order. An optimized bubble sort algorithm is used, i.e., it has
  //   the worst case O(n^2) runtime, but it is usually much faster.
  swaps = 0; // Record the total number of swaps.
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
  } while (count > 0); // Continue if some points are swapped.

  oriA = orient3d(pt[1], pt[2], pt[3], pt[4]);
  if (oriA != 0.0) {
    // Flip the sign if there are odd number of swaps.
    if ((swaps % 2) != 0) oriA = -oriA;
    *isOutside = oriA;
    PetscFunctionReturn(0);
  }

  oriB = -orient3d(pt[0], pt[2], pt[3], pt[4]);
  if (oriB == 0.0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
  // Flip the sign if there are odd number of swaps.
  if ((swaps % 2) != 0) oriB = -oriB;
  *isOutside = oriB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshIsCollinear"
// iscollinear()    Check if three points are approximately collinear.       //
//                                                                           //
// 'eps' is a relative error tolerance.  The collinearity is determined by   //
// the value q = cos(theta), where theta is the angle between two vectors    //
// A->B and A->C.  They're collinear if 1.0 - q <= epspp.                    //
/* tetgenmesh::iscollinear() */
PetscErrorCode TetGenMeshIsCollinear(TetGenMesh *m, PetscReal *A, PetscReal *B, PetscReal *C, PetscReal eps, PetscBool *co)
{
  PetscReal abx, aby, abz;
  PetscReal acx, acy, acz;
  PetscReal Lv, Lw, dd;
  PetscReal d, q;

  PetscFunctionBegin;
  // Limit of two closed points.
  q = m->longest * eps;
  q *= q;

  abx = A[0] - B[0];
  aby = A[1] - B[1];
  abz = A[2] - B[2];
  acx = A[0] - C[0];
  acy = A[1] - C[1];
  acz = A[2] - C[2];
  Lv = abx * abx + aby * aby + abz * abz;
  // Is AB (nearly) indentical?
  if (Lv < q) {
    *co = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  Lw = acx * acx + acy * acy + acz * acz;
  // Is AC (nearly) indentical?
  if (Lw < q) {
    *co = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  dd = abx * acx + aby * acy + abz * acz;

  d = (dd * dd) / (Lv * Lw);
  if (d > 1.0) d = 1.0; // Rounding.
  q = 1.0 - sqrt(d); // Notice 0 < q < 1.0.

  *co = q <= eps ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshIsCoplanar"
// iscoplanar()    Check if four points are approximately coplanar.          //
//                                                                           //
// 'vol6' is six times of the signed volume of the tetrahedron formed by the //
// four points. 'eps' is the relative error tolerance.  The coplanarity is   //
// determined by the value: q = fabs(vol6) / L^3,  where L is the average    //
// edge length of the tet. They're coplanar if q <= eps.                     //
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
  if (L <= 0.0) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Length %g should be positive", L);}
#endif
  L /= 6.0;
  q = fabs(vol6) / (L * L * L);

  *co = q <= eps ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTetAllNormal"
// tetallnormal()    Get the in-noramls of the four faces of a given tet.    //
//                                                                           //
// Let tet be abcd. N[4][3] returns the four normals, which are: N[0] cbd,   //
// N[1] acd, N[2] bad, N[3] abc. These normals are unnormalized.             //
/* tetgenmesh::tetallnormal() */
PetscErrorCode TetGenMeshTetAllNormal(TetGenMesh *m, point pa, point pb, point pc, point pd, PetscReal N[4][3], PetscReal *volume)
{
  PetscReal A[4][4], rhs[4], D;
  int       indx[4];
  int       i, j;

  PetscFunctionBegin;
  // get the entries of A[3][3].
  for(i = 0; i < 3; i++) A[0][i] = pa[i] - pd[i];  // d->a vec
  for(i = 0; i < 3; i++) A[1][i] = pb[i] - pd[i];  // d->b vec
  for(i = 0; i < 3; i++) A[2][i] = pc[i] - pd[i];  // d->c vec
  // Compute the inverse of matrix A, to get 3 normals of the 4 faces.
  lu_decmp(A, 3, indx, &D, 0);     // Decompose the matrix just once.
  if (volume) {
    // Get the volume of the tet.
    *volume = fabs((A[indx[0]][0] * A[indx[1]][1] * A[indx[2]][2])) / 6.0;
  }
  for(j = 0; j < 3; j++) {
    for(i = 0; i < 3; i++) rhs[i] = 0.0;
    rhs[j] = 1.0;  // Positive means the inside direction
    lu_solve(A, 3, indx, rhs, 0);
    for (i = 0; i < 3; i++) N[j][i] = rhs[i];
  }
  // Get the fourth normal by summing up the first three.
  for(i = 0; i < 3; i++) N[3][i] = - N[0][i] - N[1][i] - N[2][i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTetAllDihedral"
// tetalldihedral()    Get all (six) dihedral angles of a tet.               //
//                                                                           //
// The tet is given by its four corners a, b, c, and d. If 'cosdd' is not    //
// NULL, it returns the cosines of the 6 dihedral angles, the corresponding  //
// edges are: ab, bc, ca, ad, bd, and cd. If 'cosmaxd' (or 'cosmind') is not //
// NULL, it returns the cosine of the maximal (or minimal) dihedral angle.   //
/* tetgenmesh::tetalldihedral() */
PetscErrorCode TetGenMeshTetAllDihedral(TetGenMesh *m, point pa, point pb, point pc, point pd, PetscReal *cosdd, PetscReal *cosmaxd, PetscReal *cosmind)
{
  PetscReal N[4][3], vol, cosd, len;
  int f1, f2, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  vol = 0; // Check if the tet is valid or not.

  // Get four normals of faces of the tet.
  ierr = TetGenMeshTetAllNormal(m, pa, pb, pc, pd, N, &vol);CHKERRQ(ierr);

  if (vol == 0.0) {
    // This tet is not valid.
    if (cosdd != NULL) {
      for (i = 0; i < 6; i++) {
        cosdd[i] = -1.0; // 180 degree.
      }
    }
    // This tet has zero volume.
    if (cosmaxd != NULL) {
      *cosmaxd = -1.0; // 180 degree.
    }
    if (cosmind != NULL) {
      *cosmind = 1.0; // 0 degree.
    }
    PetscFunctionReturn(0);
  }

  // Normalize the normals.
  for (i = 0; i < 4; i++) {
    len = sqrt(dot(N[i], N[i]));
    if (len != 0.0) {
      for (j = 0; j < 3; j++) N[i][j] /= len;
    }
  }

  for (i = 0; i < 6; i++) {
    switch (i) {
    case 0: f1 = 2; f2 = 3; break; // edge ab.
    case 1: f1 = 0; f2 = 3; break; // edge bc.
    case 2: f1 = 1; f2 = 3; break; // edge ca.
    case 3: f1 = 1; f2 = 2; break; // edge ad.
    case 4: f1 = 2; f2 = 0; break; // edge bd.
    case 5: f1 = 0; f2 = 1; break; // edge cd.
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
// facedihedral()    Return the dihedral angle (in radian) between two       //
//                   adjoining faces.                                        //
//                                                                           //
// 'pa', 'pb' are the shared edge of these two faces, 'pc1', and 'pc2' are   //
// apexes of these two faces.  Return the angle (between 0 to 2*pi) between  //
// the normal of face (pa, pb, pc1) and normal of face (pa, pb, pc2).        //
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
  // Be careful rounding error!
  if (costheta > 1.0) {
    costheta = 1.0;
  } else if (costheta < -1.0) {
    costheta = -1.0;
  }
  theta = acos(costheta);
  ori   = orient3d(pa, pb, pc1, pc2);
  if (ori > 0.0) {
    theta = 2 * PETSC_PI - theta;
  }

  *angle = theta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTetAspectRatio"
// tetaspectratio()    Calculate the aspect ratio of the tetrahedron.        //
//                                                                           //
// The aspect ratio of a tet is R/h, where R is the circumradius and h is    //
// the shortest height of the tet.                                           //
/* tetgenmesh::tetaspectratio() */
PetscErrorCode TetGenMeshTetAspectRatio(TetGenMesh *m, point pa, point pb, point pc, point pd, PetscReal *ratio)
{
  PetscReal vda[3], vdb[3], vdc[3];
  PetscReal N[4][3], A[4][4], rhs[4], D;
  PetscReal H[4], volume, radius2, minheightinv;
  int indx[4];
  int i, j;

  PetscFunctionBegin;
  // Set the matrix A = [vda, vdb, vdc]^T.
  for(i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
  for(i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
  for(i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];
  // Lu-decompose the matrix A.
  lu_decmp(A, 3, indx, &D, 0);
  // Get the volume of abcd.
  volume = (A[indx[0]][0] * A[indx[1]][1] * A[indx[2]][2]) / 6.0;
  // Check if it is zero.
  if (volume == 0.0) {
    if (ratio) {*ratio = 1.0e+200;} // A degenerate tet.
    PetscFunctionReturn(0);
  }
  // if (volume < 0.0) volume = -volume;
  // Check the radiu-edge ratio of the tet.
  rhs[0] = 0.5 * dot(vda, vda);
  rhs[1] = 0.5 * dot(vdb, vdb);
  rhs[2] = 0.5 * dot(vdc, vdc);
  lu_solve(A, 3, indx, rhs, 0);
  // Get the circumcenter.
  // for (i = 0; i < 3; i++) circumcent[i] = pd[i] + rhs[i];
  // Get the square of the circumradius.
  radius2 = dot(rhs, rhs);

  // Compute the 4 face normals (N[0], ..., N[3]).
  for(j = 0; j < 3; j++) {
    for(i = 0; i < 3; i++) rhs[i] = 0.0;
    rhs[j] = 1.0;  // Positive means the inside direction
    lu_solve(A, 3, indx, rhs, 0);
    for(i = 0; i < 3; i++) N[j][i] = rhs[i];
  }
  // Get the fourth normal by summing up the first three.
  for(i = 0; i < 3; i++) N[3][i] = - N[0][i] - N[1][i] - N[2][i];
  // Normalized the normals.
  for(i = 0; i < 4; i++) {
    // H[i] is the inverse of the height of its corresponding face.
    H[i] = sqrt(dot(N[i], N[i]));
  }
  // Get the radius of the inscribed sphere.
  // insradius = 1.0 / (H[0] + H[1] + H[2] + H[3]);
  // Get the biggest H[i] (corresponding to the smallest height).
  minheightinv = H[0];
  for(i = 1; i < 3; i++) {
    if (H[i] > minheightinv) minheightinv = H[i];
  }
  if (ratio) {*ratio = sqrt(radius2) * minheightinv;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshPreciseLocate"
// preciselocate()    Find a simplex containing a given point.               //
//                                                                           //
// This routine implements the simple Walk-through point location algorithm. //
// Begins its search from 'searchtet', assume there is a line segment L from //
// a vertex of 'searchtet' to the query point 'searchpt', and simply walk    //
// towards 'searchpt' by traversing all faces intersected by L.              //
//                                                                           //
// On completion, 'searchtet' is a tetrahedron that contains 'searchpt'. The //
// returned value indicates one of the following cases:                      //
//   - ONVERTEX, the search point lies on the origin of 'searchtet'.         //
//   - ONEDGE, the search point lies on an edge of 'searchtet'.              //
//   - ONFACE, the search point lies on a face of 'searchtet'.               //
//   - INTET, the search point lies in the interior of 'searchtet'.          //
//   - OUTSIDE, the search point lies outside the mesh. 'searchtet' is a     //
//     hull tetrahedron whose base face is visible by the search point.      //
//                                                                           //
// WARNING: This routine is designed for convex triangulations, and will not //
// generally work after the holes and concavities have been carved.          //
//                                                                           //
// If 'maxtetnumber' > 0, stop the searching process if the number of passed //
// tets is larger than it and return OUTSIDE.                                //
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
  // 'searchtet' should be a valid tetrahedron now.
#ifdef PETSC_USE_DEBUG
  if (searchtet->tet == m->dummytet) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
#endif

  searchtet->ver = 0; // Keep in CCW edge ring.
  // Find a face of 'searchtet' such that the 'searchpt' lies strictly
  //   above it.  Such face should always exist.
  for(searchtet->loc = 0; searchtet->loc < 4; searchtet->loc++) {
    forg = org(searchtet);
    fdest = dest(searchtet);
    fapex = apex(searchtet);
    ori1 = orient3d(forg, fdest, fapex, searchpt);
    if (ori1 < 0.0) break;
  }
#ifdef PETSC_USE_DEBUG
  if (searchtet->loc >= 4) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
#endif

  backtracetet = *searchtet; // Initialize backtracetet.

  // Define 'tetnumber' for exit the loop when it's running endless.
  tetnumber = 0l;
  while ((maxtetnumber > 0l) && (tetnumber <= maxtetnumber)) {
    m->ptloc_count++;  // Algorithimic count.
    // Check if we are reaching the boundary of the triangulation.
    if (searchtet->tet == m->dummytet) {
      *searchtet = backtracetet;
      if (result) {*result = OUTSIDE;}
      PetscFunctionReturn(0);
    }
    // Initialize the face for returning the walk-through face.
    walkthroface.tet = PETSC_NULL;
    // Adjust the edge ring, so that 'ori1 < 0.0' holds.
    searchtet->ver = 0;
    // 'toppo' remains unchange for the following orientation tests.
    toppo = oppo(searchtet);
    // Check the three sides of 'searchtet' to find the face through which
    //   we can walk next.
    for(side = 0; side < 3; side++) {
      forg = org(searchtet);
      fdest = dest(searchtet);
      ori2 = orient3d(forg, fdest, toppo, searchpt);
      if (ori2 == 0.0) {
        // They are coplanar, check if 'searchpt' lies inside, or on an edge,
        //   or coindes with a vertex of face (forg, fdest, toppo).
        fapex = apex(searchtet);
        ori3 = orient3d(fdest, fapex, toppo, searchpt);
        if (ori3 < 0.0) {
          // Outside the face (fdest, fapex, toppo), walk through it.
          enextself(searchtet);
          fnext(m, searchtet, &walkthroface);
          break;
        }
        ori4 = orient3d(fapex, forg, toppo, searchpt);
        if (ori4 < 0.0) {
          // Outside the face (fapex, forg, toppo), walk through it.
          enext2self(searchtet);
          fnext(m, searchtet, &walkthroface);
          break;
        }
        // Remember, ori1 < 0.0, which means that 'searchpt' will not on edge
        //   (forg, fdest) or on vertex forg or fdest.
        // The rest possible cases are:
        //   (1) 'searchpt' lies on edge (fdest, toppo);
        //   (2) 'searchpt' lies on edge (toppo, forg);
        //   (3) 'searchpt' coincident with toppo;
        //   (4) 'searchpt' lies inside face (forg, fdest, toppo).
        fnextself(m, searchtet);
        if (ori3 == 0.0) {
          if (ori4 == 0.0) {
            // Case (4).
            enext2self(searchtet);
            if (result) {*result = ONVERTEX;}
            PetscFunctionReturn(0);
          } else {
            // Case (1).
            enextself(searchtet);
            if (result) {*result = ONEDGE;}
            PetscFunctionReturn(0);
          }
        }
        if (ori4 == 0.0) {
          // Case (2).
          enext2self(searchtet);
          if (result) {*result = ONEDGE;}
          PetscFunctionReturn(0);
        }
        // Case (4).
        if (result) {*result = ONFACE;}
        PetscFunctionReturn(0);
      } else if (ori2 < 0.0) {
        // Outside the face (forg, fdest, toppo), walk through it.
        fnext(m, searchtet, &walkthroface);
        break;
      }
      // Go to check next side.
      enextself(searchtet);
    }
    if (side == 3) {
      // Found! Inside tetrahedron.
      if (result) {*result = INTETRAHEDRON;}
      PetscFunctionReturn(0);
    }
    // We walk through the face 'walkthroface' and continue the searching.
    // Store the face handle in 'backtracetet' before we take the real walk.
    //   So we are able to restore the handle to 'searchtet' if we are
    //   reaching the outer boundary.
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
// randomsample()    Randomly sample the tetrahedra for point loation.       //
//                                                                           //
// This routine implements Muecke's Jump-and-walk point location algorithm.  //
// It improves the simple walk-through by "jumping" to a good starting point //
// via random sampling.  Searching begins from one of handles:  the input    //
// 'searchtet', a recently encountered tetrahedron 'recenttet',  or from one //
// chosen from a random sample.  The choice is made by determining which one //
// 's origin is closest to the point we are searcing for.  Having chosen the //
// starting tetrahedron, the simple Walk-through algorithm is executed.      //
/* tetgenmesh::randomsample() */
PetscErrorCode TetGenMeshRandomSample(TetGenMesh *m, point searchpt, triface *searchtet)
{
  TetGenOpts    *b  = m->b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshLocate"
// locate()    Find a simplex containing a given point.                      //
/* tetgenmesh::locate() */
PetscErrorCode TetGenMeshLocate(TetGenMesh *m, point searchpt, triface *searchtet, locateresult *result)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Randomly sample for a good starting tet.
  ierr = TetGenMeshRandomSample(m, searchpt, searchtet);CHKERRQ(ierr);
  // Call simple walk-through to locate the point.
  ierr = TetGenMeshPreciseLocate(m, searchpt, searchtet, m->tetrahedrons->items, result);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshLocate2"
// locate2()    Find a simplex containing a given point.                     //
//                                                                           //
// Another implementation of the Walk-through point location algorithm.      //
// See the comments of preciselocate().                                      //
/* tetgenmesh::locate2() */
PetscErrorCode TetGenMeshLocate2(TetGenMesh *m, point searchpt, triface *searchtet, ArrayPool *histtetarray, locateresult *result)
{
  TetGenOpts    *b  = m->b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshLocateSub"
// locatesub()    Find a point in the surface mesh of a facet.               //
//                                                                           //
// Searching begins from the input 'searchsh', it should be a handle on the  //
// convex hull of the facet triangulation.                                   //
//                                                                           //
// If 'stopatseg' is nonzero, the search will stop if it tries to walk       //
// through a subsegment, and will return OUTSIDE.                            //
//                                                                           //
// On completion, 'searchsh' is a subface that contains 'searchpt'.          //
//   - Returns ONVERTEX if the point lies on an existing vertex. 'searchsh'  //
//     is a handle whose origin is the existing vertex.                      //
//   - Returns ONEDGE if the point lies on a mesh edge.  'searchsh' is a     //
//     handle whose primary edge is the edge on which the point lies.        //
//   - Returns ONFACE if the point lies strictly within a subface.           //
//     'searchsh' is a handle on which the point lies.                       //
//   - Returns OUTSIDE if the point lies outside the triangulation.          //
//                                                                           //
// WARNING: This routine is designed for convex triangulations, and will not //
// not generally work after the holes and concavities have been carved.      //
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
    if (searchsh->sh == m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
  }
  // Find the sign to simulate that abovepoint is 'above' the facet.
  adjustedgering_face(searchsh, CCW);
  forg  = sorg(searchsh);
  fdest = sdest(searchsh);
  fapex = sapex(searchsh);
  ori = orient3d(forg, fdest, fapex, m->abovepoint);
  sign = ori > 0.0 ? -1 : 1;

  // Orient 'searchsh' so that 'searchpt' is below it (i.e., searchpt has
  //   CCW orientation with respect to searchsh in plane).  Such edge
  //   should always exist. Save it as (forg, fdest).
  for(i = 0; i < 3; i++) {
    forg  = sorg(searchsh);
    fdest = sdest(searchsh);
    ori   = orient3d(forg, fdest, m->abovepoint, searchpt) * sign;
    if (ori > 0.0) break;
    senextself(searchsh);
  }
  if (i < 3) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}

  while (1) {
    fapex = sapex(searchsh);
    // Check whether the apex is the point we seek.
    if (fapex[0] == searchpt[0] && fapex[1] == searchpt[1] && fapex[2] == searchpt[2]) {
      senext2self(searchsh);
      if (result) *result = ONVERTEX;
      PetscFunctionReturn(0);
    }
    // Does the point lie on the other side of the line defined by the
    //   triangle edge opposite the triangle's destination?
    destori = orient3d(forg, fapex, m->abovepoint, searchpt) * sign;
    if (epspp > 0.0) {
      ierr = TetGenMeshIsCoplanar(m, forg, fapex, m->abovepoint, searchpt, destori, epspp, &isCoplanar);CHKERRQ(ierr);
      if (isCoplanar) {
        destori = 0.0;
      }
    }
    // Does the point lie on the other side of the line defined by the
    //   triangle edge opposite the triangle's origin?
    orgori = orient3d(fapex, fdest, m->abovepoint, searchpt) * sign;
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
        // The point must be on the boundary of or inside this triangle.
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
    // Move to another triangle.  Leave a trace `backtracksh' in case
    //   walking off a boundary of the triangulation.
    if (moveleft) {
      senext2(searchsh, &backtracksh);
      fdest = fapex;
    } else {
      senext(searchsh, &backtracksh);
      forg = fapex;
    }
    // Check if we meet a segment.
    sspivot(m, &backtracksh, &checkedge);
    if (checkedge.sh != m->dummysh) {
      if (stopatseg) {
        // The flag indicates we should not cross a segment. Stop.
        *searchsh = backtracksh;
        if (result) *result = OUTSIDE;
        PetscFunctionReturn(0);
      }
      // Try to walk through a segment. We need to find a coplanar subface
      //   sharing this segment to get into.
      spinsh = backtracksh;
      do {
        spivotself(&spinsh);
        if (spinsh.sh == backtracksh.sh) {
          // Turn back, no coplanar subface is found.
          break;
        }
        // Are they belong to the same facet.
        if (shellmark(m, &spinsh) == shellmark(m, &backtracksh)) {
          // Find a coplanar subface. Walk into it.
          *searchsh = spinsh;
          break;
        }
        // Are they (nearly) coplanar?
        ori = orient3d(forg, fdest, sapex(&backtracksh), sapex(&spinsh));
        ierr = TetGenMeshIsCoplanar(m, forg, fdest, sapex(&backtracksh), sapex(&spinsh), ori, b->epsilon, &isCoplanar);CHKERRQ(ierr);
        if (isCoplanar) {
          // Find a coplanar subface. Walk into it.
          *searchsh = spinsh;
          break;
        }
      } while (spinsh.sh != backtracksh.sh);
    } else {
      spivot(&backtracksh, searchsh);
    }
    // Check for walking right out of the triangulation.
    if ((searchsh->sh == m->dummysh) || (searchsh->sh == backtracksh.sh)) {
      // Go back to the last triangle.
      *searchsh = backtracksh;
      if (result) *result = OUTSIDE;
      PetscFunctionReturn(0);
    }
    // To keep the same orientation wrt abovepoint.
    if (sorg(searchsh) != forg) sesymself(searchsh);
    if ((sorg(searchsh) != forg) || (sdest(searchsh) != fdest)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
  }
  PetscFunctionReturn(0);
}

////                                                                       ////
////                                                                       ////
//// geom_cxx /////////////////////////////////////////////////////////////////

//// flip_cxx /////////////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshEnqueueFlipFace"
// enqueueflipface(), enqueueflipedge()    Queue a face (or an edge).        //
//                                                                           //
// The face (or edge) may be non-locally Delaunay. It is queued for process- //
// ing in flip() (or flipsub()). The vertices of the face (edge) are stored  //
// seperatly to ensure the face (or edge) is still the same one when we save //
// it since other flips will cause this face (or edge) be changed or dead.   //
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
// enqueueflipface(), enqueueflipedge()    Queue a face (or an edge).        //
//                                                                           //
// The face (or edge) may be non-locally Delaunay. It is queued for process- //
// ing in flip() (or flipsub()). The vertices of the face (edge) are stored  //
// seperatly to ensure the face (or edge) is still the same one when we save //
// it since other flips will cause this face (or edge) be changed or dead.   //
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
// flip22sub()    Perform a 2-to-2 flip on a subface edge.                   //
//                                                                           //
// The flip edge is given by subface 'flipedge'.  Let it is abc, where ab is //
// the flipping edge.  The other subface is bad,  where a, b, c, d form a    //
// convex quadrilateral.  ab is not a subsegment.                            //
//                                                                           //
// A 2-to-2 subface flip is to change two subfaces abc and bad to another    //
// two subfaces dca and cdb.  Hence, edge ab has been removed and dc becomes //
// an edge. If a point e is above abc, this flip is equal to rotate abc and  //
// bad counterclockwise using right-hand rule with thumb points to e. It is  //
// important to know that the edge rings of the flipped subfaces dca and cdb //
// are keeping the same orientation as their original subfaces. So they have //
// the same orientation with respect to the lift point of this facet.        //
//                                                                           //
// During rotating, the face rings of the four edges bc, ca, ad, and de need //
// be re-connected. If the edge is not a subsegment, then its face ring has  //
// only two faces, a sbond() will bond them together. If it is a subsegment, //
// one should use sbond1() twice to bond two different handles to the rotat- //
// ing subface, one is predecssor (-casin), another is successor (-casout).  //
//                                                                           //
// If 'flipqueue' is not NULL, it returns four edges bc, ca, ad, de, which   //
// may be non-Delaunay.                                                      //
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

  // Unmark the flipped subfaces (used in mesh refinement). 2009-08-17.
  sunmarktest(&abc);
  sunmarktest(&bad);

  // Save the old configuration outside the quadrilateral.
  senext(&abc, &oldbc);
  senext2(&abc, &oldca);
  senext(&bad, &oldad);
  senext2(&bad, &olddb);
  // Get the outside connection. Becareful if there is a subsegment on the
  //   quadrilateral, two casings (casin and casout) are needed to save for
  //   keeping the face link.
  spivot(&oldbc, &bccasout);
  sspivot(m, &oldbc, &bc);
  if (bc.sh != m->dummysh) {
    // 'bc' is a subsegment.
    if (bccasout.sh != m->dummysh) {
      if (oldbc.sh != bccasout.sh) {
        // 'oldbc' is not self-bonded.
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
    // 'ca' is a subsegment.
    if (cacasout.sh != m->dummysh) {
      if (oldca.sh != cacasout.sh) {
        // 'oldca' is not self-bonded.
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
    // 'ad' is a subsegment.
    if (adcasout.sh != m->dummysh) {
      if (oldad.sh != adcasout.sh) {
        // 'adcasout' is not self-bonded.
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
    // 'db' is a subsegment.
    if (dbcasout.sh != m->dummysh) {
      if (olddb.sh != dbcasout.sh) {
        // 'dbcasout' is not self-bonded.
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

  // Rotate abc and bad one-quarter turn counterclockwise.
  if (ca.sh != m->dummysh) {
    if (cacasout.sh != m->dummysh) {
      sbond1(&cacasin, &oldbc);
      sbond1(&oldbc, &cacasout);
    } else {
      // Bond 'oldbc' to itself.
      sdissolve(m, &oldbc); // sbond(oldbc, oldbc);
      // Make sure that dummysh always correctly bonded.
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
      // Bond 'oldca' to itself.
      sdissolve(m, &oldca); // sbond(oldca, oldca);
      // Make sure that dummysh always correctly bonded.
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
      // Bond 'oldad' to itself.
      sdissolve(m, &oldad); // sbond(oldad, oldad);
      // Make sure that dummysh always correctly bonded.
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
      // Bond 'olddb' to itself.
      sdissolve(m, &olddb); // sbond(olddb, olddb);
      // Make sure that dummysh always correctly bonded.
      m->dummysh[0] = sencode(&olddb);
    }
    ssbond(m, &olddb, &bc);
  } else {
    sbond(&olddb, &bccasout);
  }

  // New vertex assignments for the rotated subfaces.
  setsorg(&abc, pd);  // Update abc to dca.
  setsdest(&abc, pc);
  setsapex(&abc, pa);
  setsorg(&bad, pc);  // Update bad to cdb.
  setsdest(&bad, pd);
  setsapex(&bad, pb);

  // Update the point-to-subface map.
  // Comemnt: After the flip, abc becomes dca, bad becodes cdb. 
  setpoint2sh(m, pa, sencode(&abc)); // dca
  setpoint2sh(m, pb, sencode(&bad)); // cdb
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
#define __FUNCT__ "TetGenMeshLawson"
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
    if (checkseg.sh != m->dummysh) continue;  // Can't flip a subsegment.
    spivot(&flipedge, &symedge);
    if (symedge.sh == m->dummysh) continue; // Can't flip a hull edge.
    pa = sorg(&flipedge);
    pb = sdest(&flipedge);
    pc = sapex(&flipedge);
    pd = sapex(&symedge);
    // Choose the triangle abc or abd as the base depending on the angle1
    //   (Vac, Vab) and angle2 (Vad, Vab).
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
      // angle1 is closer to 90 than angle2, choose abc (flipedge).
      m->abovepoint = m->facetabovepointarray[shellmark(m, &flipedge)];
      if (!m->abovepoint) {
        ierr = TetGenMeshGetFacetAbovePoint(m, &flipedge);CHKERRQ(ierr);
      }
      sign = insphere(pa, pb, pc, m->abovepoint, pd);
      ori  = orient3d(pa, pb, pc, m->abovepoint);
    } else {
      // angle2 is closer to 90 than angle1, choose abd (symedge).
      m->abovepoint = m->facetabovepointarray[shellmark(m, &symedge)];
      if (!m->abovepoint) {
        ierr = TetGenMeshGetFacetAbovePoint(m, &symedge);CHKERRQ(ierr);
      }
      sign = insphere(pa, pb, pd, m->abovepoint, pc);
      ori  = orient3d(pa, pb, pd, m->abovepoint);
    }
    // Correct the sign.
    sign = ori > 0.0 ? sign : -sign;
    if (sign > 0.0) {
      // Flip the non-Delaunay edge.
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
#define __FUNCT__ "TetGenMeshSplitTetEdge"
// splittetedge()    Insert a point on an edge of the mesh.                  //
//                                                                           //
// The edge is given by 'splittet'. Assume its four corners are a, b, n1 and //
// n2, where ab is the edge will be split. Around ab may exist any number of //
// tetrahedra. For convenience, they're ordered in a sequence following the  //
// right-hand rule with your thumb points from a to b. Let the vertex set of //
// these tetrahedra be {a, b, n1, n2, ..., n(i)}. NOTE the tetrahedra around //
// ab may not connect to each other (can only happen when ab is a subsegment,//
// hence some faces abn(i) are subfaces).  If ab is a subsegment, abn1 must  //
// be a subface.                                                             //
//                                                                           //
// To split edge ab by a point v is to split all tetrahedra containing ab by //
// v.  More specifically, for each such tetrahedron, an1n2b, it is shrunk to //
// an1n2v, and a new tetrahedra bn2n1v is created. If ab is a subsegment, or //
// some faces of the splitting tetrahedra are subfaces, they must be split   //
// either by calling routine 'splitsubedge()'.                               //
//                                                                           //
// On completion, 'splittet' returns avn1n2.  If 'flipqueue' is not NULL, it //
// returns all faces which may become non-Delaunay after this operation.     //
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
    // Is there a subsegment need to be split together?
    ierr = TetGenMeshTssPivot(m, splittet, &abseg);CHKERRQ(ierr);
    if (abseg.sh != m->dummysh) {
      abseg.shver = 0;
      // Orient the edge direction of 'splittet' be abseg.
      if (org(splittet) != sorg(&abseg)) {
        esymself(splittet);
      }
    }
  }
  spintet = *splittet;
  pa = org(&spintet);
  pb = dest(&spintet);

  PetscInfo3(b->in, "  Inserting point %d on edge (%d, %d).\n", pointmark(m, newpoint), pointmark(m, pa), pointmark(m, pb));

  // Collect the tetrahedra containing the splitting edge (ab).
  n1 = apex(&spintet);
  hitbdry = 0;
  wrapcount = 1;
  if (m->checksubfaces && abseg.sh != m->dummysh) {
    // It may happen that some tetrahedra containing ab (a subsegment) are
    //   completely disconnected with others. If it happens, use the face
    //   link of ab to cross the boundary.
    while(1) {
      if (!fnextself(m, &spintet)) {
        // Meet a boundary, walk through it.
        hitbdry ++;
        tspivot(m, &spintet, &spinsh);
#ifdef PETSC_USE_DEBUG
        if (spinsh.sh == m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Shell edge should not be null");}
#endif
        ierr = TetGenMeshFindEdge_face(m, &spinsh, pa, pb);CHKERRQ(ierr);
        sfnextself(m, &spinsh);
        stpivot(m, &spinsh, &spintet);
#ifdef SELF_CHECK
        if (spintet.tet == m->dummytet) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Tet should not be null");}
#endif
        ierr = TetGenMeshFindEdge_triface(m, &spintet, pa, pb);CHKERRQ(ierr);
        // Remember this position (hull face) in 'splittet'.
        *splittet = spintet;
        // Split two hull faces increase the hull size;
        m->hullsize += 2;
      }
      if (apex(&spintet) == n1) break;
      wrapcount ++;
    }
    if (hitbdry > 0) {
      wrapcount -= hitbdry;
    }
  } else {
    // All the tetrahedra containing ab are connected together. If there
    //   are subfaces, 'splitsh' keeps one of them.
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
      // ab is on the hull.
      wrapcount -= 1;
      // 'spintet' now is a hull face, inverse its edge direction.
      esym(&spintet, splittet);
      // Split two hull faces increases the number of hull faces.
      m->hullsize += 2;
    }
  }

  // Make arrays of updating (bot, oldtop) and new (newtop) tetrahedra.
  ierr = PetscMalloc2(wrapcount,triface,&bots,wrapcount,triface,&newtops);CHKERRQ(ierr);
  // Spin around ab, gather tetrahedra and set up new tetrahedra.
  spintet = *splittet;
  for (i = 0; i < wrapcount; i++) {
    // Get 'bots[i] = an1n2b'.
    enext2fnext(m, &spintet, &bots[i]);
    esymself(&bots[i]);
    // Create 'newtops[i]'.
    ierr = TetGenMeshMakeTetrahedron(m, &(newtops[i]));CHKERRQ(ierr);
    // Go to the next.
    fnextself(m, &spintet);
    if (m->checksubfaces && abseg.sh != m->dummysh) {
      if (!issymexist(m, &spintet)) {
        // We meet a hull face, walk through it.
        tspivot(m, &spintet, &spinsh);
        if (spinsh.sh == m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        ierr = TetGenMeshFindEdge_face(m, &spinsh, pa, pb);CHKERRQ(ierr);
        sfnextself(m, &spinsh);
        stpivot(m, &spinsh, &spintet);
        if (spintet.tet == m->dummytet) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        ierr = TetGenMeshFindEdge_triface(m, &spintet, pa, pb);CHKERRQ(ierr);
      }
    }
  }

  // Set the vertices of updated and new tetrahedra.
  for(i = 0; i < wrapcount; i++) {
    // Update 'bots[i] = an1n2v'.
    setoppo(&bots[i], newpoint);
    // Set 'newtops[i] = bn2n1v'.
    n1 = dest(&bots[i]);
    n2 = apex(&bots[i]);
    // Set 'newtops[i]'.
    setorg(&newtops[i], pb);
    setdest(&newtops[i], n2);
    setapex(&newtops[i], n1);
    setoppo(&newtops[i], newpoint);
    // Set the element attributes of a new tetrahedron.
    for(j = 0; j < in->numberoftetrahedronattributes; j++) {
      attrib = elemattribute(m, bots[i].tet, j);
      setelemattribute(m, newtops[i].tet, j, attrib);
    }
    if (b->varvolume) {
      // Set the area constraint of a new tetrahedron.
      volume = volumebound(m, bots[i].tet);
      setvolumebound(m, newtops[i].tet, volume);
    }
    // Make sure no inversed tetrahedron has been created.
    volume = orient3d(pa, n1, n2, newpoint);
    if (volume >= 0.0) {
      //printf("Internal error in splittetedge(): volume = %.12g.\n", volume);
      break;
    }
    volume = orient3d(pb, n2, n1, newpoint);
    if (volume >= 0.0) {
      //printf("Internal error in splittetedge(): volume = %.12g.\n", volume);
      break;
    }
  }

  if (i < wrapcount) {
    // Do not insert this point. It will result inverted or degenerated tet.
    // Restore have updated tets in "bots".
    for(; i >= 0; i--) {
      setoppo(&bots[i], pb);
    }
    // Deallocate tets in "newtops".
    for (i = 0; i < wrapcount; i++) {
      ierr = TetGenMeshTetrahedronDealloc(m, newtops[i].tet);CHKERRQ(ierr);
    }
    ierr = PetscFree2(bots,newtops);CHKERRQ(ierr);
    if (isSplit) {*isSplit = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  // Bond newtops to topcasings and bots.
  for (i = 0; i < wrapcount; i++) {
    // Get 'oldtop = n1n2va' from 'bots[i]'.
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
  // Bond between newtops.
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
  // Bond the last to the first if no boundary.
  if (issymexist(m, &spintet)) {
    enext2fnext(m, &newtops[0], &tmpbond1);
    bond(m, &tmpbond0, &tmpbond1);
  }
  if (m->checksubsegs) {
    for(i = 0; i < wrapcount; i++) {
      enextfnext(m, &bots[i], &worktet); // edge n1->n2.
      tsspivot1(m, &worktet, &n1n2seg);
      if (n1n2seg.sh != m->dummysh) {
        enext(&newtops[i], &tmpbond0);
        tssbond1(m, &tmpbond0, &n1n2seg);
      }
      enextself(&worktet); // edge n2->v ==> n2->b
      tsspivot1(m, &worktet, &n2vseg);
      if (n2vseg.sh != m->dummysh) {
        tssdissolve1(m, &worktet);
        tssbond1(m, &newtops[i], &n2vseg);
      }
      enextself(&worktet); // edge v->n1 ==> b->n1
      tsspivot1(m, &worktet, &n1vseg);
      if (n1vseg.sh != m->dummysh) {
        tssdissolve1(m, &worktet);
        enext2(&newtops[i], &tmpbond0);
        tssbond1(m, &tmpbond0, &n1vseg);
      }
    }
  }

  // Is there exist subfaces and subsegment need to be split?
  if (m->checksubfaces) {
    if (abseg.sh != m->dummysh) {
      // A subsegment needs be split.
      spivot(&abseg, &splitsh);
      if (splitsh.sh == m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    }
    if (splitsh.sh != m->dummysh) {
      // Split subfaces (and subsegment).
      ierr = TetGenMeshFindEdge_face(m, &splitsh, pa, pb);CHKERRQ(ierr);
      ierr = TetGenMeshSplitSubEdge(m, newpoint, &splitsh, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    }
  }

  if (b->verbose > 3) {
    for(i = 0; i < wrapcount; i++) {
      printf("    Updating bots[%i] ", i);
      ierr = TetGenMeshPrintTet(m, &(bots[i]));CHKERRQ(ierr);
      printf("    Creating newtops[%i] ", i);
      ierr = TetGenMeshPrintTet(m, &(newtops[i]));CHKERRQ(ierr);
    }
  }

  if (flipqueue) {
    for(i = 0; i < wrapcount; i++) {
      ierr = TetGenMeshEnqueueFlipFace(m, &bots[i], flipqueue);CHKERRQ(ierr);
      ierr = TetGenMeshEnqueueFlipFace(m, &newtops[i], flipqueue);CHKERRQ(ierr);
    }
  }

  // Set the return handle be avn1n2.  It is got by transforming from
  //   'bots[0]' (which is an1n2v).
  fnext(m, &bots[0], &spintet); // spintet is an1vn2.
  esymself(&spintet); // spintet is n1avn2.
  enextself(&spintet); // spintet is avn1n2.
  *splittet = spintet;

  ierr = PetscFree2(bots,newtops);CHKERRQ(ierr);
  if (isSplit) {*isSplit = PETSC_TRUE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormStarPolyhedron"
// formstarpolyhedron()    Get the star ployhedron of a point 'pt'.          //
//                                                                           //
// The polyhedron P is formed by faces of tets having 'pt' as a vertex.  If  //
// 'complete' is TRUE, P is the complete star of 'pt'. Otherwise, P is boun- //
// ded by subfaces, i.e. P is only part of the star of 'pt'.                 //
//                                                                           //
// 'tetlist' T returns the tets, it has one of such tets on input. Moreover, //
// if t is in T, then oppo(t) = p.  Topologically, T is the star of p;  and  //
// the faces of T is the link of p. 'verlist' V returns the vertices of T.   //
/* tetgenmesh::formstarpolyhedron() */
PetscErrorCode TetGenMeshFormStarPolyhedron(TetGenMesh *m, point pt, List* tetlist, List* verlist, PetscBool complete)
{
  triface starttet = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  face    checksh  = {PETSC_NULL, 0};
  point ver[3];
  int len, idx, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Get a tet t containing p.
  ierr = ListItem(tetlist, 0, (void **) &starttet);CHKERRQ(ierr);
  // Let oppo(t) = p.
  for(starttet.loc = 0; starttet.loc < 4; starttet.loc++) {
    if (oppo(&starttet) == pt) break;
  }
  if (starttet.loc >= 4) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate opposing vertex");}
  // Add t into T.
  ierr = ListSetItem(tetlist, 0, (void **) &starttet);CHKERRQ(ierr);
  infect(m, &starttet);
  if (verlist) {
    // Add three verts of t into V.
    ver[0] = org(&starttet);
    ver[1] = dest(&starttet);
    ver[2] = apex(&starttet);
    for(i = 0; i < 3; i++) {
      // Mark the vert by inversing the index of the vert.
      idx = pointmark(m, ver[i]);
      setpointmark(m, ver[i], -idx - 1); // -1 to distinguish the zero.
      ierr = ListAppend(verlist, &(ver[i]), PETSC_NULL);CHKERRQ(ierr);
    }
  }

  // Find other tets by a broadth-first search.
  ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
    starttet.ver = 0;
    for(j = 0; j < 3; j++) {
      fnext(m, &starttet, &neightet);
      tspivot(m, &neightet, &checksh);
      // Should we cross a subface.
      if ((checksh.sh == m->dummysh) || complete) {
        // Get the neighbor n.
        symself(&neightet);
        if ((neightet.tet != m->dummytet) && !infected(m, &neightet)) {
          // Let oppo(n) = p.
          for(neightet.loc = 0; neightet.loc < 4; neightet.loc++) {
            if (oppo(&neightet) == pt) break;
          }
          if (neightet.loc >= 4) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate opposing vertex");}
          // Add n into T.
          infect(m, &neightet);
          ierr = ListAppend(tetlist, &neightet, PETSC_NULL);CHKERRQ(ierr);
          if (verlist) {
            // Add the apex vertex in n into V.
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
  }

  // Uninfect tets.
  ierr = ListLength(tetlist, &len);CHKERRQ(ierr);
  for(i = 0; i < len; i++) {
    ierr = ListItem(tetlist, i, (void **) &starttet);CHKERRQ(ierr);
    uninfect(m, &starttet);
  }
  if (verlist) {
    // Uninfect vertices.
    ierr = ListLength(verlist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(verlist, i, (void **) &ver[0]);CHKERRQ(ierr);
      idx = pointmark(m, ver[0]);
      setpointmark(m, ver[0], -(idx + 1));
    }
  }
  PetscFunctionReturn(0);
}

////                                                                       ////
////                                                                       ////
//// flip_cxx /////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTransferNodes"
// transfernodes()    Transfer nodes from 'io->pointlist' to 'this->points'. //
//                                                                           //
// Initializing 'this->points'.  Transferring all points from 'in->pointlist'//
// into it. All points are indexed (start from in->firstnumber).  Each point //
// is initialized be UNUSEDVERTEX.  The bounding box (xmin, xmax, ymin, ymax,//
// zmin, zmax) and the diameter (longest) of the point set are calculated.   //
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
  // Read the points.
  coordindex  = 0;
  attribindex = 0;
  mtrindex    = 0;
  for(i = 0; i < in->numberofpoints; i++) {
    ierr = TetGenMeshMakePoint(m, &pointloop);CHKERRQ(ierr);
    // Read the point coordinates.
    x = pointloop[0] = in->pointlist[coordindex++];
    y = pointloop[1] = in->pointlist[coordindex++];
    z = pointloop[2] = in->pointlist[coordindex++];
    // Read the point attributes.
    for(j = 0; j < in->numberofpointattributes; j++) {
      pointloop[3 + j] = in->pointattributelist[attribindex++];
    }
    // Read the point metric tensor.
    for(j = 0; j < in->numberofpointmtrs; j++) {
      pointloop[m->pointmtrindex + j] = in->pointmtrlist[mtrindex++];
    }
    // Determine the smallest and largests x, y and z coordinates.
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
  // 'longest' is the largest possible edge length formed by input vertices.
  x = m->xmax - m->xmin;
  y = m->ymax - m->ymin;
  z = m->zmax - m->zmin;
  m->longest = sqrt(x * x + y * y + z * z);
  if (m->longest == 0.0) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The point set is trivial.\n");
  }
  // Two identical points are distinguished by 'lengthlimit'.
  m->lengthlimit = m->longest * b->epsilon * 1e+2;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMakeSegmentMap"
// Create a map from vertex indices to segments, subfaces, and tetrahedra    //
// sharing at the same vertices.                                             //
//                                                                           //
// The map is stored in two arrays: 'idx2___list' and '___sperverlist', they //
// form a sparse matrix whose size is (n+1)x(n+1), where n is the number of  //
// segments, subfaces, or tetrahedra. 'idx2___list' contains row information //
// and '___sperverlist' contains all non-zero elements.  The i-th entry of   //
// 'idx2___list' is the starting position of i-th row's non-zero elements in //
// '___sperverlist'.  The number of elements of i-th row is (i+1)-th entry   //
// minus i-th entry of 'idx2___list'.                                        //
//                                                                           //
// NOTE: These two arrays will be created inside this routine, don't forget  //
// to free them after using.                                                 //
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
  // Create and initialize 'idx2seglist'.
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &idx2seglist);CHKERRQ(ierr);
  for(i = 0; i < m->points->items + 1; i++) idx2seglist[i] = 0;
  // Loop the set of segments once, counter the number of segments sharing each vertex.
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &shloop);CHKERRQ(ierr);
  while(shloop) {
    // Increment the number of sharing segments for each endpoint.
    for(i = 0; i < 2; i++) {
      j = pointmark(m, (point) shloop[3 + i]) - in->firstnumber;
      idx2seglist[j]++;
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &shloop);CHKERRQ(ierr);
  }
  // Calculate the total length of array 'facesperverlist'.
  j = idx2seglist[0];
  idx2seglist[0] = 0;  // Array starts from 0 element.
  for(i = 0; i < m->points->items; i++) {
    k = idx2seglist[i + 1];
    idx2seglist[i + 1] = idx2seglist[i] + j;
    j = k;
  }
  // The total length is in the last unit of idx2seglist.
  ierr = PetscMalloc(idx2seglist[i] * sizeof(shellface*), &segsperverlist);CHKERRQ(ierr);
  // Loop the set of segments again, set the info. of segments per vertex.
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
  // Contents in 'idx2seglist' are shifted, now shift them back.
  for(i = m->points->items - 1; i >= 0; i--) {
    idx2seglist[i + 1] = idx2seglist[i];
  }
  idx2seglist[0] = 0;
  *idx2seglistPtr    = idx2seglist;
  *segsperverlistPtr = segsperverlist;
  PetscFunctionReturn(0);
}

//// delaunay_cxx /////////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBTreeSort"
// btree_sort()    Sort vertices using a binary space partition (bsp) tree.  //
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
    // Split along x-axis.
    split = 0.5 * (bxmin + bxmax);
  } else if (axis == 1) {
    // Split along y-axis.
    split = 0.5 * (bymin + bymax);
  } else {
    // Split along z-axis.
    split = 0.5 * (bzmin + bzmax);
  }

  i = 0;
  j = arraysize - 1;

  // Partition the vertices into left- and right-arraies.
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
    // Is the partition finished?
    if (i == (j + 1)) {
      break;
    }
    // Swap i-th and j-th vertices.
    swapvert = vertexarray[i];
    vertexarray[i] = vertexarray[j];
    vertexarray[j] = swapvert;
    // Continue patitioning the array;
  } while (1);

  PetscInfo2(b->in, "    leftsize = %d, rightsize = %d\n", i, arraysize - i);
  lflag = rflag = PETSC_FALSE;

  // if (depth < max_tree_depth) {
    if (i > b->max_btreenode_size) {
      // Recursively partition the left array (length = i).
      if (axis == 0) { // x
        ierr = TetGenMeshBTreeSort(m, vertexarray, i, (axis + 1) % 3, bxmin, split, bymin, bymax, bzmin, bzmax, depth + 1);CHKERRQ(ierr);
      } else if (axis == 1) { // y
        ierr = TetGenMeshBTreeSort(m, vertexarray, i, (axis + 1) % 3, bxmin, bxmax, bymin, split, bzmin, bzmax, depth + 1);CHKERRQ(ierr);
      } else { // z
        ierr = TetGenMeshBTreeSort(m, vertexarray, i, (axis + 1) % 3, bxmin, bxmax, bymin, bymax, bzmin, split, depth + 1);CHKERRQ(ierr);
      }
    } else {
      lflag = PETSC_TRUE;
    }
    if ((arraysize - i) > b->max_btreenode_size) {
      // Recursively partition the right array (length = arraysize - i).
      if (axis == 0) { // x
        ierr = TetGenMeshBTreeSort(m, &(vertexarray[i]), arraysize - i, (axis + 1) % 3, split, bxmax, bymin, bymax, bzmin, bzmax, depth + 1);CHKERRQ(ierr);
      } else if (axis == 1) { // y
        ierr = TetGenMeshBTreeSort(m, &(vertexarray[i]), arraysize - i, (axis + 1) % 3, bxmin, bxmax, split, bymax, bzmin, bzmax, depth + 1);CHKERRQ(ierr);
      } else { // z
        ierr = TetGenMeshBTreeSort(m, &(vertexarray[i]), arraysize - i, (axis + 1) % 3, bxmin, bxmax, bymin, bymax, split, bzmax, depth + 1);CHKERRQ(ierr);
      }
    } else {
      rflag = PETSC_TRUE;
    }

  if (lflag && (i > 0)) {
    // Remember the maximal length of the partitions.
    if (i > m->max_btreenode_size) {
      m->max_btreenode_size = i;
    }
    // Allocate space for the left array (use the first entry to save
    //   the length of this array).
    ierr = PetscMalloc((i + 1) * sizeof(point), &leftarray);CHKERRQ(ierr);
    leftarray[0] = (point) i; // The array length.
    // Put all points in this array.
    for(k = 0; k < i; k++) {
      leftarray[k + 1] = vertexarray[k];
      setpoint2ppt(m, leftarray[k + 1], (point) leftarray);
    }
    // Save this array in list.
    ierr = ArrayPoolNewIndex(m->btreenode_list, (void **) &pptary, PETSC_NULL);CHKERRQ(ierr);
    *pptary = leftarray;
  }

  // Get the length of the right array.
  j = arraysize - i;
  if (rflag && (j > 0)) {
    if (j > m->max_btreenode_size) {
      m->max_btreenode_size = j;
    }
    // Allocate space for the right array (use the first entry to save
    //   the length of this array).
    ierr = PetscMalloc((j + 1) * sizeof(point), &rightarray);CHKERRQ(ierr);
    rightarray[0] = (point) j; // The array length.
    // Put all points in this array.
    for (k = 0; k < j; k++) {
      rightarray[k + 1] = vertexarray[i + k];
      setpoint2ppt(m, rightarray[k + 1], (point) rightarray);
    }
    // Save this array in list.
    ierr = ArrayPoolNewIndex(m->btreenode_list, (void **) &pptary, PETSC_NULL);CHKERRQ(ierr);
    *pptary = rightarray;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBTreeInsert"
// btree_insert()    Add a vertex into a tree node.                          //
/* tetgenmesh::btree_insert() */
PetscErrorCode TetGenMeshBTreeInsert(TetGenMesh *m, point insertpt)
{
  point *ptary;
  long arylen; // The array lenhgth is saved in ptary[0].

  PetscFunctionBegin;
  // Get the tree node (save in this point).
  ptary = (point *) point2ppt(m, insertpt);
  // Get the current array length.
  arylen = (long) ptary[0];
  // Insert the point into the node.
  ptary[arylen + 1] = insertpt;
  // Increase the array length by 1.
  ptary[0] = (point) (arylen + 1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshBTreeSearch"
// btree_search()    Search a near point for an inserting point.             //
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
  // Get the tree node (save in this point).
  ptary = (point *) point2ppt(m, insertpt);
  // Get the current array length.
  arylen = (long) ptary[0];

  if (arylen == 0) {
    searchtet->tet = PETSC_NULL;
    PetscFunctionReturn(0);
  }

  if (arylen < 10) {
    ptsamples = arylen;
  } else {
    ptsamples = 10; // Take at least 10 samples.
    //   The number of random samples taken is proportional to the third root
    //   of the number of points in the cell.
    while (ptsamples * ptsamples * ptsamples < arylen) {
      ptsamples++;
    }
  }

  // Select "good" candidate using k random samples, taking the closest one.
  mindist2 = 1.79769E+308; // The largest double value (8 byte).
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
// ordervertices()    Order the vertices for incremental inserting.          //
//                                                                           //
// We assume the vertices have been sorted by a binary tree.                 //
/* tetgenmesh::ordervertices() */
PetscErrorCode TetGenMeshOrderVertices(TetGenMesh *m, point *vertexarray, int arraysize)
{
  point **ipptary, **jpptary, *swappptary;
  point *ptary;
  long arylen;
  int index, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // First pick one vertex from each tree node.
  for(i = 0; i < (int) m->btreenode_list->objects; i++) {
    ipptary = (point **) fastlookup(m->btreenode_list, i);
    ptary = *ipptary;
    vertexarray[i] = ptary[1]; // Skip the first entry.
  }

  index = i;
  // Then put all other points in the array node by node.
  for(i = (int) m->btreenode_list->objects - 1; i >= 0; i--) {
    // Randomly pick a tree node.
    ierr = TetGenMeshRandomChoice(m, i + 1, &j);CHKERRQ(ierr);
    // Save the i-th node.
    ipptary = (point **) fastlookup(m->btreenode_list, i);
    // Get the j-th node.
    jpptary = (point **) fastlookup(m->btreenode_list, j);
    // Order the points in the node.
    ptary = *jpptary;
    arylen = (long) ptary[0];
    for(j = 2; j <= arylen; j++) { // Skip the first point.
      vertexarray[index] = ptary[j];
      index++;
    }
    // Clear this tree node.
    ptary[0] = (point) 0;
    // Swap i-th node to j-th node.
    swappptary = *ipptary;
    *ipptary = *jpptary; // [i] <= [j]
    *jpptary = swappptary; // [j] <= [i]
  }

  // Make sure we've done correctly.
  if (index != arraysize) {SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %d should match array size %d", index, arraysize);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInsertVertexBW"
// insertvertexbw()    Insert a vertex using the Boywer-Watson algorithm.    //
//                                                                           //
// The point p will be first located in T. 'searchtet' is a suggested start- //
// tetrahedron, it can be NULL. Note that p may lies outside T. In such case,//
// the convex hull of T will be updated to include p as a vertex.            //
//                                                                           //
// If 'bwflag' is TRUE, the Bowyer-Watson algorithm is used to recover the   //
// Delaunayness of T. Otherwise, do nothing with regard to the Delaunayness  //
// T (T may be non-Delaunay after this function).                            //
//                                                                           //
// If 'visflag' is TRUE, force to check the visibility of the boundary faces //
// of cavity. This is needed when T is not Delaunay.                         //
//                                                                           //
// If 'noencflag' is TRUE, only insert the new point p if it does not cause  //
// any existing (sub)segment be non-Delaunay. This option only is checked    //
// when the global variable 'checksubsegs' is set.                           //
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
  ArrayPool *swaplist; // for updating cavity.
  long updatecount;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo1(b->in, "    Insert point %d\n", pointmark(m, insertpt));

  tetcount = m->ptloc_count;
  updatecount = 0;

  // Locate the point.
  if (!searchtet->tet) {
    if (m->btreenode_list) { // default option
      // Use bsp-tree to select a starting tetrahedron.
      ierr = TetGenMeshBTreeSearch(m, insertpt, searchtet);CHKERRQ(ierr);
    } else { // -u0 option
      // Randomly select a starting tetrahedron.
      ierr = TetGenMeshRandomSample(m, insertpt, searchtet);CHKERRQ(ierr);
    }
    ierr = TetGenMeshPreciseLocate(m, insertpt, searchtet, m->tetrahedrons->items, &loc);CHKERRQ(ierr);
  } else {
    // Start from 'searchtet'.
    ierr = TetGenMeshLocate2(m, insertpt, searchtet, PETSC_NULL, &loc);CHKERRQ(ierr);
  }

  PetscInfo1(b->in, "    Walk distance (# tets): %ld\n", m->ptloc_count - tetcount);

  if (m->ptloc_max_count < (m->ptloc_count - tetcount)) {
    m->ptloc_max_count = (m->ptloc_count - tetcount);
  }

  PetscInfo5(b->in, "    Located (%d) tet (%d, %d, %d, %d).\n", (int) loc, pointmark(m, org(searchtet)), pointmark(m, dest(searchtet)), pointmark(m, apex(searchtet)), pointmark(m, oppo(searchtet)));

  if (loc == ONVERTEX) {
    // The point already exists. Mark it and do nothing on it.
    if (b->object != STL) {
      PetscInfo2(b->in, "Warning:  Point #%d is duplicated with Point #%d. Ignored!\n", pointmark(m, insertpt), pointmark(m, org(searchtet)));
    }
    setpoint2ppt(m, insertpt, org(searchtet));
    setpointtype(m, insertpt, DUPLICATEDVERTEX);
    m->dupverts++;
    if (result) {*result = loc;}
    PetscFunctionReturn(0);
  }

  tetcount = 0l;  // The number of deallocated tets.

  // Create the initial boundary of the cavity.
  if (loc == INTETRAHEDRON) {
    // Add four boundary faces of this tet into list.
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
    // Add at most six boundary faces into list.
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
      // Split a hull face into three hull faces.
      m->hullsize += 2;
    }
    m->flip26count++;
  } else if (loc == ONEDGE) {
    // Add all adjacent boundary tets into list.
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
      // Go to the next tet (may be dummytet).
      tfnext(m, &spintet, &neightet);
      if (neightet.tet == m->dummytet) {
        hitbdry++;
        if (hitbdry == 2) break;
        esym(searchtet, &spintet); // Go to another direction.
        tfnext(m, &spintet, &neightet);
        if (neightet.tet == m->dummytet) break;
      }
      spintet = neightet;
    } while (apex(&spintet) != pc);
    // Update hull size if it is a hull edge.
    if (hitbdry > 0) {
      // Split a hull edge deletes two hull faces, adds four new hull faces.
      m->hullsize += 2;
    }
    m->flipn2ncount++;
  } else if (loc == OUTSIDE) {
    // p lies outside the convex hull. Enlarge the convex hull by including p.
    PetscInfo(b->in, "    Insert a hull vertex.\n");
    // 'searchtet' refers to a hull face which is visible by p.
    adjustedgering_triface(searchtet, CW);
    // Create the first tet t (from f and p).
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
    // Connect t to T.
    bond(m, &newtet, searchtet);
    // Removed a hull face, added three "new hull faces".
    m->hullsize += 2;

    // Add a cavity boundary face.
    ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
    *parytet = newtet;
    // Add a cavity tet.
    infect(m, &newtet);
    ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
    *parytet = newtet;
    tetcount++;

    // Add three "new hull faces" into list (re-use cavebdrylist).
    newtet.ver = 0;
    for(i = 0; i < 3; i++) {
      fnext(m, &newtet, &neightet);
      ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      *parytet = neightet;
      enextself(&newtet);
    }

    // Find all actual new hull faces.
    for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
      // Get a queued "new hull face".
      parytet = (triface *) fastlookup(m->cavebdrylist, i);
      // Every "new hull face" must have p as its apex.
      if (apex(parytet) != insertpt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
      if ((parytet->ver & 1) != 1) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");} // It's CW edge ring.
      // Check if it is still a hull face.
      sym(parytet, &neightet);
      if (neightet.tet == m->dummytet) {
        // Yes, get its adjacent hull face (at its edge).
        esym(parytet, &neightet);
        while (1) {
          fnextself(m, &neightet);
          // Does its adjacent tet exist?
          sym(&neightet, &neineitet);
          if (neineitet.tet == m->dummytet) break;
          symedgeself(m, &neightet);
        }
        // neightet is an adjacent hull face.
        pc = apex(&neightet);
        if (pc != insertpt) {
          // Check if p is visible by the hull face ('neightet').
          pa = org(&neightet);
          pb = dest(&neightet);
          ori = orient3d(pa, pb, pc, insertpt); m->orient3dcount++;
          if (ori < 0) {
            // Create a new tet adjacent to neightet.
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
            // Comment: We removed two hull faces, and added two "new hull
            //   faces", hence hullsize remains unchanged.
            // Add a cavity boundary face.
            ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
            *parytet1 = newtet;
            // Add a cavity tet.
            infect(m, &newtet);
            ierr = ArrayPoolNewIndex(m->caveoldtetlist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
            *parytet1 = newtet;
            tetcount++;
            // Add two "new hull faces" into list.
            enextself(&newtet);
            for(j = 0; j < 2; j++) {
              fnext(m, &newtet, &neineitet);
              ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
              *parytet1 = neineitet;
              enextself(&newtet);
            }
          }
        } else {
          // Two hull faces matched. Bond the two adjacent tets.
          bond(m, parytet, &neightet);
          m->hullsize -= 2;
        }
      } // if (neightet.tet == dummytet)
    } // i
    ierr = ArrayPoolRestart(m->cavebdrylist);CHKERRQ(ierr);
    m->inserthullcount++;
  }

  if (!bwflag) {
    if (result) {*result = loc;}
    PetscFunctionReturn(0);
  }

  // Form the Boywer-Watson cavity.
  for (i = 0; i < (int) m->cavetetlist->objects; i++) {
    // Get a cavity boundary face.
    parytet = (triface *) fastlookup(m->cavetetlist, i);
    if (parytet->tet == m->dummytet) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    if (!infected(m, parytet)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");} // The tet is inside the cavity.
    enqflag = PETSC_FALSE;
    // Get the adjacent tet.
    sym(parytet, &neightet);
    if (neightet.tet != m->dummytet) {
      if (!infected(m, &neightet)) {
        if (!marktested(m, &neightet)) {
          ppt = (point *) &(neightet.tet[4]);
          ierr = TetGenMeshInSphereS(m, ppt[0], ppt[1], ppt[2], ppt[3], insertpt, &sign);CHKERRQ(ierr);
          enqflag = (sign < 0.0);
          // Avoid redundant insphere tests.
          marktest(m, &neightet);
        }
      } else {
        enqflag = PETSC_TRUE;
      }
    }
    if (enqflag) { // Found a tet in the cavity.
      if (!infected(m, &neightet)) { // Avoid to add it multiple times.
        // Put other three faces in check list.
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
      // Found a boundary face of the cavity.
      if (neightet.tet == m->dummytet) {
        // Check for a possible flat tet (see m27.node, use -J option).
        pa = org(parytet);
        pb = dest(parytet);
        pc = apex(parytet);
        ori = orient3d(pa, pb, pc, insertpt);
        if (ori != 0) {
          ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
          *parytet1 = *parytet;
          // futureflip = flippush(futureflip, parytet, insertpt);
        }
      } else {
        ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet1, PETSC_NULL);CHKERRQ(ierr);
        *parytet1 = *parytet;
      }
    }
  } // i

  PetscInfo2(b->in, "    Cavity formed: %ld tets, %ld faces.\n", tetcount, m->cavebdrylist->objects);

  m->totaldeadtets += tetcount;
  m->totalbowatcavsize += m->cavebdrylist->objects;
  if (m->maxbowatcavsize < (long) m->cavebdrylist->objects) {
    m->maxbowatcavsize = m->cavebdrylist->objects;
  }

  if (m->checksubsegs || noencsegflag) {
    // Check if some (sub)segments are inside the cavity.
    for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
      parytet = (triface *) fastlookup(m->caveoldtetlist, i);
      for (j = 0; j < 6; j++) {
        parytet->loc = edge2locver[j][0];
        parytet->ver = edge2locver[j][1];
        tsspivot1(m, parytet, &checkseg);
        if ((checkseg.sh != m->dummysh) && !sinfected(m, &checkseg)) {
          // Check if this segment is inside the cavity.
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
              enqflag = PETSC_FALSE; break; // It is not inside.
            }
            if (apex(&spintet) == pa) break;
          }
          if (enqflag) {
            PetscInfo2(b->in, "      Queue a missing segment (%d, %d).\n", pointmark(m, sorg(&checkseg)), pointmark(m, sdest(&checkseg)));
            sinfect(m, &checkseg);  // Only save it once.
            ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &paryseg, PETSC_NULL);CHKERRQ(ierr);
            *paryseg = checkseg;
          }
        }
      }
    }
  }

  if (noencsegflag && (m->subsegstack->objects > 0)) {
    // Found encroached subsegments! Do not insert this point.
    for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
      parytet = (triface *) fastlookup(m->caveoldtetlist, i);
      uninfect(m, parytet);
      unmarktest(m, parytet);
    }
    // Unmark cavity neighbor tets (outside the cavity).
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
    // Check if some subfaces are inside the cavity.
    for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
      parytet = (triface *) fastlookup(m->caveoldtetlist, i);
      neightet.tet = parytet->tet;
      for(neightet.loc = 0; neightet.loc < 4; neightet.loc++) {
        tspivot(m, &neightet, &checksh);
        if (checksh.sh != m->dummysh) {
          sym(&neightet, &neineitet);
          // Do not check it if it is a hull tet.
          if (neineitet.tet != m->dummytet) {
            if (infected(m, &neineitet)) {
              PetscInfo3(b->in, "      Queue a missing subface (%d, %d, %d).\n", pointmark(m, sorg(&checksh)), pointmark(m, sdest(&checksh)), pointmark(m, sapex(&checksh)));
              tsdissolve(m, &neineitet); // Disconnect a tet-sub bond.
              stdissolve(m, &checksh); // Disconnect the sub-tet bond.
              sesymself(&checksh);
              stdissolve(m, &checksh);
              // Add the missing subface into list.
              ierr = ArrayPoolNewIndex(m->subfacstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
              *pssub = checksh;
            }
          }
        }
      }
    }
  }

  if (noencsubflag && (m->subfacstack->objects > 0)) {
    // Found encroached subfaces! Do not insert this point.
  }

  if (visflag) {
    // If T is not a Delaunay triangulation, the formed cavity may not be
    //   star-shaped (fig/dump-cavity-case8). Validation is needed.
    ierr = ArrayPoolRestart(m->cavetetlist);CHKERRQ(ierr);
    for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
      cavetet = (triface *) fastlookup(m->cavebdrylist, i);
      if (infected(m, cavetet)) {
        sym(cavetet, &neightet);
        if (neightet.tet == m->dummytet || !infected(m, &neightet)) {
          if (neightet.tet != m->dummytet) {
            cavetet->ver = 4; // CCW edge ring.
            pa = dest(cavetet);
            pb = org(cavetet);
            pc = apex(cavetet);
            ori = orient3d(pa, pb, pc, insertpt); m->orient3dcount++;
            if (ori == 0.0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
            enqflag = (ori > 0.0);
          } else {
            enqflag = PETSC_TRUE; // A hull face.
          }
          if (enqflag) {
            // This face is valid, save it.
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
            // Add three new faces to find new boundaries.
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
          // This face is not on the cavity boundary anymore.
          unmarktest(m, cavetet);
        }
      } else {
        if (marktested(m, cavetet)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
      }
    }
    if (updatecount > 0) {
      // Update the cavity boundary faces (fig/dump-cavity-case9).
      ierr = ArrayPoolRestart(m->cavebdrylist);CHKERRQ(ierr);
      for(i = 0; i < (int) m->cavetetlist->objects; i++) {
        cavetet = (triface *) fastlookup(m->cavetetlist, i);
        // 'cavetet' was boundary face of the cavity.
        if (infected(m, cavetet)) {
          sym(cavetet, &neightet);
          if ((neightet.tet != m->dummytet) || !infected(m, &neightet)) {
            // It is a cavity boundary face.
            ierr = ArrayPoolNewIndex(m->cavebdrylist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
            *parytet = *cavetet;
          } else {
            // Not a cavity boundary face.
            unmarktest(m, cavetet);
          }
        } else {
          if (marktested(m, cavetet)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        }
      }
      // Update the list of old tets.
      ierr = ArrayPoolRestart(m->cavetetlist);CHKERRQ(ierr);
      for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
        cavetet = (triface *) fastlookup(m->caveoldtetlist, i);
        if (infected(m, cavetet)) {
          ierr = ArrayPoolNewIndex(m->cavetetlist, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = *cavetet;
        }
      }
      if ((int) m->cavetetlist->objects >= i) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
      // Swap 'm->cavetetlist' and 'm->caveoldtetlist'.
      swaplist = m->caveoldtetlist;
      m->caveoldtetlist = m->cavetetlist;
      m->cavetetlist = swaplist;
      PetscInfo2(b->in, "    Size of the updated cavity: %d faces %d tets.\n", (int) m->cavebdrylist->objects, (int) m->caveoldtetlist->objects);
    }
  }

  // Re-use this list for new cavity faces.
  ierr = ArrayPoolRestart(m->cavetetlist);CHKERRQ(ierr);

  // Create new tetrahedra in the Bowyer-Watson cavity and Connect them.
  for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
    parytet = (triface *) fastlookup(m->cavebdrylist, i);
    if (!infected(m, parytet)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");} // The tet is inside the cavity.
    parytet->ver = 0; // In CCW edge ring.
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
    // Bond the new tet to the adjacent tet outside the cavity.
    sym(parytet, &neightet);
    if (neightet.tet != m->dummytet) {
      // The tet was marked (to avoid redundant insphere tests).
      unmarktest(m, &neightet);
      bond(m, &newtet, &neightet);
    } else {
      // Bond newtet to dummytet.
      m->dummytet[0] = encode(&newtet);
    }
    // mark the other three faces of this tet as "open".
    neightet.tet = newtet.tet;
    for(j = 0; j < 3; j++) {
      neightet.tet[locpivot[0][j]] = PETSC_NULL;
    }
    // Let the oldtet knows newtet (for connecting adjacent new tets).
    parytet->tet[parytet->loc] = encode(&newtet);
    if (m->checksubsegs) {
      // newtet and parytet share at the same edge.
      for(j = 0; j < 3; j++) {
        tsspivot1(m, parytet, &checkseg);
        if (checkseg.sh != m->dummysh) {
          if (sinfected(m, &checkseg)) {
            // This subsegment is not missing. Unmark it.
            PetscInfo2(b->in, "      Dequeue a segment (%d, %d).\n", pointmark(m, sorg(&checkseg)), pointmark(m, sdest(&checkseg)));
            suninfect(m, &checkseg); // Dequeue a non-missing segment.
          }
          tssbond1(m, &newtet, &checkseg);
        }
        enextself(parytet);
        enextself(&newtet);
      }
    }
    if (m->checksubfaces) {
      // Bond subface to the new tet.
      tspivot(m, parytet, &checksh);
      if (checksh.sh != m->dummysh) {
        tsbond(m, &newtet, &checksh);
        // The other-side-connection of checksh should be no change.
      }
    }
  } // i

  // Set a handle for speeding point location.
  m->recenttet = newtet;
  setpoint2tet(m, insertpt, encode(&newtet));

  // Connect adjacent new tetrahedra together. Here we utilize the connections
  //   of the old cavity tets to find the new adjacent tets.
  for(i = 0; i < (int) m->cavebdrylist->objects; i++) {
    parytet = (triface *) fastlookup(m->cavebdrylist, i);
    decode(parytet->tet[parytet->loc], &newtet);
    // assert(org(newtet) == org(*parytet)); // SELF_CHECK
    // assert((newtet.ver & 1) == 0); // in CCW edge ring.
    for(j = 0; j < 3; j++) {
      fnext(m, &newtet, &neightet); // Go to the "open" face.
      if (neightet.tet[neightet.loc] == PETSC_NULL) {
        spintet = *parytet;
        while (1) {
          fnextself(m, &spintet);
          symedgeself(m, &spintet);
          if (spintet.tet == m->dummytet) break;
          if (!infected(m, &spintet)) break;
        }
        if (spintet.tet != m->dummytet) {
          // 'spintet' is the adjacent tet of the cavity.
          fnext(m, &spintet, &neineitet);
          if (neineitet.tet[neineitet.loc]) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
          bond(m, &neightet, &neineitet);
        } else {
          // This side is a hull face.
          neightet.tet[neightet.loc] = (tetrahedron) m->dummytet;
          m->dummytet[0] = encode(&neightet);
        }
      }
      setpoint2tet(m, org(&newtet), encode(&newtet));
      enextself(&newtet);
      enextself(parytet);
    }
  }

  // Delete the old cavity tets.
  for(i = 0; i < (int) m->caveoldtetlist->objects; i++) {
    parytet = (triface *) fastlookup(m->caveoldtetlist, i);
    ierr = TetGenMeshTetrahedronDealloc(m, parytet->tet);CHKERRQ(ierr);
  }

  // Set the point type.
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
  PetscReal      det, attrib, volume;
  int            i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // The initial tetrahedralization T only has one tet formed by 4 affinely
  //   linear independent vertices of the point set V = 'insertarray'. The
  //   first point a = insertarray[0].

  // Get the second point b, that is not identical or very close to a.
  for(i = 1; i < arraysize; i++) {
    det = distance(insertarray[0], insertarray[i]);
    if (det > (m->longest * eps)) break;
  }
  if (i == arraysize) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All points seem to be identical,");
  } else {
    // Swap to move b from index i to index 1.
    swappt         = insertarray[i];
    insertarray[i] = insertarray[1];
    insertarray[1] = swappt;
  }
  // Get the third point c, that is not collinear with a and b.
  for(i++; i < arraysize; i++) {
    PetscBool co;

    ierr = TetGenMeshIsCollinear(m, insertarray[0], insertarray[1], insertarray[i], eps, &co);CHKERRQ(ierr);
    if (!co) break;
  }
  if (i == arraysize) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All points seem to be collinear.");
  } else {
    // Swap to move c from index i to index 2.
    swappt         = insertarray[i];
    insertarray[i] = insertarray[2];
    insertarray[2] = swappt;
  }
  // Get the fourth point d, that is not coplanar with a, b, and c.
  for(i++; i < arraysize; i++) {
    PetscBool co;

    det = orient3d(insertarray[0], insertarray[1], insertarray[2], insertarray[i]);
    if (det == 0.0) continue;
    ierr = TetGenMeshIsCoplanar(m, insertarray[0], insertarray[1], insertarray[2], insertarray[i], det, eps, &co);CHKERRQ(ierr);
    if (!co) break;
  }
  if (i == arraysize) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "All points seem to be coplanar.");
  } else {
    // Swap to move d from index i to index 3.
    swappt         = insertarray[i];
    insertarray[i] = insertarray[3];
    insertarray[3] = swappt;
    lastpt = insertarray[3];
    // The index of the next inserting point is 4.
    i = 4;
  }

  if (det > 0.0) {
    // For keeping the positive orientation.
    swappt         = insertarray[0];
    insertarray[0] = insertarray[1];
    insertarray[1] = swappt;
  }

  // Create the initial tet.
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
  // Set vertex type be FREEVOLVERTEX if it has no type yet.
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
  // Bond to 'dummytet' for point location.
  m->dummytet[0] = encode(&newtet);
  m->recenttet   = newtet;
  // Update the point-to-tet map.
  setpoint2tet(m, insertarray[0], encode(&newtet));
  setpoint2tet(m, insertarray[1], encode(&newtet));
  setpoint2tet(m, insertarray[2], encode(&newtet));
  setpoint2tet(m, lastpt,         encode(&newtet));
  if (b->verbose > 3) {
    PetscInfo(b->in, "    Creating tetra ");
    ierr = TetGenMeshPrintTet(m, &newtet);CHKERRQ(ierr);
  }
  // At init, all faces of this tet are hull faces.
  m->hullsize = 4;

  PetscInfo(b->in, "    Incrementally inserting points.\n");
  // Insert the rest of points, one by one.
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
// delaunizevertices()    Form a Delaunay tetrahedralization.                //
//                                                                           //
// Given a point set V (saved in 'points').  The Delaunay tetrahedralization //
// D of V is created by incrementally inserting vertices. Returns the number //
// of triangular faces bounding the convex hull of D.                        //
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

  // Prepare the array of points for inserting.
  arraysize = m->points->items;
  ierr = PetscMalloc(arraysize * sizeof(point), &insertarray);CHKERRQ(ierr);

  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  if (b->btree) { // -u option.
    // Use the input order.
    for(i = 0; i < arraysize; i++) {
      ierr = TetGenMeshPointTraverse(m, &insertarray[i]);CHKERRQ(ierr);
    }
    PetscInfo(b->in, "  Sorting vertices by a bsp-tree.\n");
    // Sort the points using a binary tree recursively.
    ierr = TetGenMeshBTreeSort(m, insertarray, in->numberofpoints, 0, m->xmin, m->xmax, m->ymin, m->ymax, m->zmin, m->zmax, 0);CHKERRQ(ierr);
    PetscInfo1(b->in, "  Number of tree nodes: %ld.\n", m->btreenode_list->objects);
    PetscInfo1(b->in, "  Maximum tree node size: %d.\n", m->max_btreenode_size);
    PetscInfo1(b->in, "  Maximum tree depth: %d.\n", m->max_btree_depth);
    // Order the sorted points.
    ierr = TetGenMeshOrderVertices(m, insertarray, in->numberofpoints);CHKERRQ(ierr);
  } else {
    PetscInfo(b->in, "  Permuting vertices.\n");
    // Randomize the point order.
    for(i = 0; i < arraysize; i++) {
      ierr = TetGenMeshRandomChoice(m, i+1, &j);CHKERRQ(ierr); // 0 <= j <= i
      insertarray[i] = insertarray[j];
      ierr = TetGenMeshPointTraverse(m, &insertarray[j]);CHKERRQ(ierr);
    }
  }

  PetscInfo(b->in, "  Incrementally inserting vertices.\n");
  // Form the DT by incremental flip Delaunay algorithm.
  ierr = TetGenMeshDelaunayIncrFlip(m, PETSC_NULL, insertarray, arraysize, PETSC_TRUE, b->plc, 0.0, PETSC_NULL);CHKERRQ(ierr);

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

////                                                                       ////
////                                                                       ////
//// delaunay_cxx /////////////////////////////////////////////////////////////

//// surface_cxx //////////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSInsertVertex"
// sinsertvertex()    Insert a vertex into a triangulation of a facet.       //
//                                                                           //
// The new point (p) will be located. Searching from 'splitsh'. If 'splitseg'//
// is not NULL, p is on a segment, no search is needed.                      //
//                                                                           //
// If 'cflag' is not TRUE, the triangulation may be not convex. Don't insert //
// p if it is found in outside.                                              //
//                                                                           //
// Comment: This routine assumes the 'abovepoint' of this facet has been set,//
// i.e., the routine getabovepoint() has been executed before it is called.  //
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
  int n, s, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (splitseg) {
    spivot(splitseg, splitsh);
    loc = ONEDGE;
  } else {
    // Locate the point, '1' means the flag stop-at-segment is on.
    ierr = TetGenMeshLocateSub(m, insertpt, splitsh, 1, 0, &loc);CHKERRQ(ierr);
  }

  // Return if p lies on a vertex.
  if (loc == ONVERTEX) {
    if (result) {*result = loc;}
    PetscFunctionReturn(0);
  }

  if (loc == OUTSIDE && !cflag) {
    // Return if 'cflag' is not set.
    if (result) {*result = loc;}
    PetscFunctionReturn(0);
  }

  if (loc == ONEDGE) {
    if (!splitseg) {
      // Do not split a segment.
      sspivot(m, splitsh, &checkseg);
      if (checkseg.sh != m->dummysh) {
        if (result) {*result = loc;}
        PetscFunctionReturn(0);
      }
      // Check if this edge is on the hull.
      spivot(splitsh, &neighsh);
      if (neighsh.sh == m->dummysh) {
        // A convex hull edge. The new point is on the hull.
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

  // Does 'insertpt' lie on a segment?
  if (splitseg) {
    splitseg->shver = 0;
    pa = sorg(splitseg);
    // Count the number of faces at segment [a, b].
    n = 0;
    neighsh = *splitsh;
    do {
      spivotself(&neighsh);
      n++;
    } while ((neighsh.sh != m->dummysh) && (neighsh.sh != splitsh->sh));
    // n is at least 1.
    ierr = PetscMalloc(n * sizeof(face), &abfaces);CHKERRQ(ierr);
    // Collect faces at seg [a, b].
    abfaces[0] = *splitsh;
    if (sorg(&abfaces[0]) != pa) sesymself(&abfaces[0]);
    for (i = 1; i < n; i++) {
      spivot(&abfaces[i - 1], &abfaces[i]);
      if (sorg(&abfaces[i]) != pa) sesymself(&abfaces[i]);
    }
  }

  // Initialize the cavity.
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
  } else { // loc == OUTSIDE;
    // This is only possible when T is convex.
    if (!cflag) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot have point otside the convex hull");}
    // Adjust 'abovepoint' to be above the 'splitsh'. 2009-07-21.
    ori = orient3d(sorg(splitsh), sdest(splitsh), sapex(splitsh), m->abovepoint);
    if (ori == 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    if (ori > 0) {
      sesymself(splitsh);
    }
    // Assume p is on top of the edge ('splitsh'). Find a right-most edge which is visible by p.
    neighsh = *splitsh;
    while (1) {
      senext2self(&neighsh);
      spivot(&neighsh, &casout);
      if (casout.sh == m->dummysh) {
        // A convex hull edge. Is it visible by p.
        pa = sorg(&neighsh);
        pb = sdest(&neighsh);
        ori = orient3d(pa, pb, m->abovepoint, insertpt);
        if (ori < 0) {
          *splitsh = neighsh; // Update 'splitsh'.
        } else {
          break; // 'splitsh' is the right-most visible edge.
        }
      } else {
        if (sorg(&casout) != sdest(&neighsh)) sesymself(&casout);
        neighsh = casout;
      }
    }
    // Create new triangles for all visible edges of p (from right to left).
    casin.sh = m->dummysh;  // No adjacent face at right.
    pa = sorg(splitsh);
    pb = sdest(splitsh);
    while (1) {
      // Create a new subface on top of the (visible) edge.
      ierr = TetGenMeshMakeShellFace(m, m->subfaces, &newsh);CHKERRQ(ierr);
      // setshvertices(newsh, pb, pa, insertpt);
      setsorg(&newsh, pb);
      setsdest(&newsh, pa);
      setsapex(&newsh, insertpt);
      setshellmark(m, &newsh, shellmark(m, splitsh));
      if (b->quality && m->varconstraint) {
        area = areabound(m, splitsh);
        setareabound(m, &newsh, area);
      }
      // Connect the new subface to the bottom subfaces.
      sbond1(&newsh, splitsh);
      sbond1(splitsh, &newsh);
      // Connect the new subface to its right-adjacent subface.
      if (casin.sh != m->dummysh) {
        senext(&newsh, &casout);
        sbond1(&casout, &casin);
        sbond1(&casin, &casout);
      }
      // The left-adjacent subface has not been created yet.
      senext2(&newsh, &casin);
      // Add the new face into list.
      smarktest(&newsh);
      ierr = ArrayPoolNewIndex(m->caveshlist, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
      *parysh = newsh;
      // Move to the convex hull edge at the left of 'splitsh'.
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
      // A convex hull edge. Is it visible by p.
      pa = sorg(splitsh);
      pb = sdest(splitsh);
      ori = orient3d(pa, pb, m->abovepoint, insertpt);
      if (ori >= 0) break;
    }
  }

  // Form the Bowyer-Watson cavity.
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
              sign = 1; // A boundary edge.
            }
          } else {
            sign = -1; // Not a boundary edge.
          }
        } else {
          if (loc == OUTSIDE) {
            // It is a boundary edge if it does not contain insertp.
            if ((sorg(parysh)==insertpt) || (sdest(parysh)==insertpt)) {
              sign = -1; // Not a boundary edge.
            } else {
              sign = 1; // A boundary edge.
            }
          } else {
            sign = 1; // A boundary edge.
          }
        }
      } else {
        sign = 1; // A segment!
      }
      if (sign >= 0) {
        // Add a boundary edge.
        ierr = ArrayPoolNewIndex(m->caveshbdlist, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
        *pssub = *parysh;
      }
      senextself(parysh);
    }
  }

  // Creating new subfaces.
  for(i = 0; i < (int) m->caveshbdlist->objects; i++) {
    parysh = (face *) fastlookup(m->caveshbdlist, i);
    sspivot(m, parysh, &checkseg);
    if ((parysh->shver & 01) != 0) sesymself(parysh);
    pa = sorg(parysh);
    pb = sdest(parysh);
    // Create a new subface.
    ierr = TetGenMeshMakeShellFace(m, m->subfaces, &newsh);CHKERRQ(ierr);
    // setshvertices(newsh, pa, pb, insertpt);
    setsorg(&newsh, pa);
    setsdest(&newsh, pb);
    setsapex(&newsh, insertpt);
    setshellmark(m, &newsh, shellmark(m, parysh));
    if (b->quality && m->varconstraint) {
      area = areabound(m, parysh);
      setareabound(m, &newsh, area);
    }
    // Connect newsh to outer subfaces.
    spivot(parysh, &casout);
    if (casout.sh != m->dummysh) {
      if (casout.sh != parysh->sh) { // It is not self-bonded.
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
        // This side is empty.
      }
    } else {
      // This is a hull side. Save it in dummysh[0] (it will be used by the routine locatesub()). 2009-07-20.
      m->dummysh[0] = sencode(&newsh);
    }
    if (checkseg.sh != m->dummysh) {
      ssbond(m, &newsh, &checkseg);
    }
    // Connect oldsh <== newsh (for connecting adjacent new subfaces).
    sbond1(parysh, &newsh);
  }

  // Set a handle for searching.
  // recentsh = newsh;

  // Connect adjacent new subfaces together.
  for(i = 0; i < (int) m->caveshbdlist->objects; i++) {
    // Get an old subface at edge [a, b].
    parysh = (face *) fastlookup(m->caveshbdlist, i);
    sspivot(m, parysh, &checkseg);
    spivot(parysh, &newsh); // The new subface [a, b, p].
    senextself(&newsh); // At edge [b, p].
    spivot(&newsh, &neighsh);
    if (neighsh.sh == m->dummysh) {
      // Find the adjacent new subface at edge [b, p].
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
        // Now 'neighsh' is a new subface at edge [b, #].
        if (sorg(&neighsh) != pb) sesymself(&neighsh);
        if (sorg(&neighsh) != pb) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        if (sapex(&neighsh) != insertpt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        senext2self(&neighsh); // Go to the open edge [p, b].
        spivot(&neighsh, &casout); // SELF_CHECK
        if (casout.sh != m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        sbond(&newsh, &neighsh);
      } else {
        if (loc != OUTSIDE) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        // It is a hull edge. 2009-07-21
        m->dummysh[0] = sencode(&newsh);
      }
    }
    spivot(parysh, &newsh); // The new subface [a, b, p].
    senext2self(&newsh); // At edge [p, a].
    spivot(&newsh, &neighsh);
    if (neighsh.sh == m->dummysh) {
      // Find the adjacent new subface at edge [p, a].
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
        // Now 'neighsh' is a new subface at edge [#, a].
        if (sdest(&neighsh) != pa) sesymself(&neighsh);
        if (sdest(&neighsh) != pa) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        if (sapex(&neighsh) != insertpt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        senextself(&neighsh); // Go to the open edge [a, p].
        spivot(&neighsh, &casout); // SELF_CHECK
        if (casout.sh != m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        sbond(&newsh, &neighsh);
      } else {
        if (loc != OUTSIDE) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        // It is a hull edge. 2009-07-21
        m->dummysh[0] = sencode(&newsh);
      }
    }
  }

  if (splitseg) {
    // Split the segment [a, b].
    aseg = *splitseg;
    pa = sorg(&aseg);
    pb = sdest(&aseg);
    PetscInfo3(b->in, "    Split seg (%d, %d) by %d.\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, insertpt));
    // Insert the new point p.
    ierr = TetGenMeshMakeShellFace(m, m->subsegs, &bseg);CHKERRQ(ierr);
    // setshvertices(bseg, insertpt, pb, NULL);
    setsorg(&bseg, insertpt);
    setsdest(&bseg, pb);
    setsapex(&bseg, PETSC_NULL);
    setsdest(&aseg, insertpt);
    setshellmark(m, &bseg, shellmark(m, &aseg));
    // This is done outside this routine (at where newpt was created).
    // setpoint2sh(insertpt, sencode(aseg));
    if (b->quality && m->varconstraint) {
      setareabound(m, &bseg, areabound(m, &aseg));
    }
    // Update the point-to-seg map.
    setpoint2seg(m, pb, sencode(&bseg));
    setpoint2seg(m, insertpt, sencode(&bseg));
    // Connect [p, b]<->[b, #].
    senext(&aseg, &aoutseg);
    spivotself(&aoutseg);
    if (aoutseg.sh != m->dummysh) {
      senext(&bseg, &boutseg);
      sbond(&boutseg, &aoutseg);
    }
    // Connect [a, p] <-> [p, b].
    senext(&aseg, &aoutseg);
    senext2(&bseg, &boutseg);
    sbond(&aoutseg, &boutseg);
    // Connect subsegs [a, p] and [p, b] to the true new subfaces.
    for(i = 0; i < n; i++) {
      spivot(&abfaces[i], &newsh); // The faked new subface.
      if (sorg(&newsh) != pa) sesymself(&newsh);
      senext2(&newsh, &neighsh); // The edge [p, a] in newsh
      spivot(&neighsh, &casout);
      ssbond(m, &casout, &aseg);
      senext(&newsh, &neighsh); // The edge [b, p] in newsh
      spivot(&neighsh, &casout);
      ssbond(m, &casout, &bseg);
    }
    if (n > 1) {
      // Create the two face rings at [a, p] and [p, b].
      for(i = 0; i < n; i++) {
        spivot(&abfaces[i], &newsh); // The faked new subface.
        if (sorg(&newsh) != pa) sesymself(&newsh);
        spivot(&abfaces[(i + 1) % n], &neighsh); // The next faked new subface.
        if (sorg(&neighsh) != pa) sesymself(&neighsh);
        senext2(&newsh, &casout); // The edge [p, a] in newsh.
        senext2(&neighsh, &casin); // The edge [p, a] in neighsh.
        spivotself(&casout);
        spivotself(&casin);
        sbond1(&casout, &casin); // Let the i's face point to (i+1)'s face.
        senext(&newsh, &casout); // The edge [b, p] in newsh.
        senext(&neighsh, &casin); // The edge [b, p] in neighsh.
        spivotself(&casout);
        spivotself(&casin);
        sbond1(&casout, &casin);
      }
    } else {
      // Only one subface contains this segment.
      // assert(n == 1);
      spivot(&abfaces[0], &newsh);  // The faked new subface.
      if (sorg(&newsh) != pa) sesymself(&newsh);
      senext2(&newsh, &casout); // The edge [p, a] in newsh.
      spivotself(&casout);
      sdissolve(m, &casout); // Disconnect to faked subface.
      senext(&newsh, &casout); // The edge [b, p] in newsh.
      spivotself(&casout);
      sdissolve(m, &casout); // Disconnect to faked subface.
    }
    // Delete the faked new subfaces.
    for(i = 0; i < n; i++) {
      spivot(&abfaces[i], &newsh); // The faked new subface.
      ierr = TetGenMeshShellFaceDealloc(m, m->subfaces, newsh.sh);CHKERRQ(ierr);
    }
    if (m->checksubsegs) {
      // Add two subsegs into stack (for recovery).
      if (!sinfected(m, &aseg)) {
        ierr = TetGenMeshRandomChoice(m, m->subsegstack->objects + 1, &s);CHKERRQ(ierr);
        ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
        *parysh = * (face *) fastlookup(m->subsegstack, s);
        sinfect(m, &aseg);
        parysh = (face *) fastlookup(m->subsegstack, s);
        *parysh = aseg;
      }
      if (sinfected(m, &bseg)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
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
    // Add all new subfaces into list.
    for(i = 0; i < (int) m->caveshbdlist->objects; i++) {
      // Get an old subface at edge [a, b].
      parysh = (face *) fastlookup(m->caveshbdlist, i);
      spivot(parysh, &newsh); // The new subface [a, b, p].
      // Some new subfaces may get deleted (when 'splitseg' is a segment).
      if (!isdead_face(&newsh)) {
        PetscInfo3(b->in, "      Queue a new subface (%d, %d, %d).\n", pointmark(m, sorg(&newsh)), pointmark(m, sdest(&newsh)), pointmark(m, sapex(&newsh)));
        ierr = ArrayPoolNewIndex(m->subfacstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
        *pssub = newsh;
      }
    }
  }

  // Update the point-to-subface map.
  for(i = 0; i < (int) m->caveshbdlist->objects; i++) {
    // Get an old subface at edge [a, b].
    parysh = (face *) fastlookup(m->caveshbdlist, i);
    spivot(parysh, &newsh); // The new subface [a, b, p].
    // Some new subfaces may get deleted (when 'splitseg' is a segment).
    if (!isdead_face(&newsh)) {
      ppt = (point *) &(newsh.sh[3]);
      for(j = 0; j < 3; j++) {
        setpoint2sh(m, ppt[j], sencode(&newsh));
      }
    }
  }

  // Delete the old subfaces.
  for(i = 0; i < (int) m->caveshlist->objects; i++) {
    parysh = (face *) fastlookup(m->caveshlist, i);
    if (m->checksubfaces) {
      // Disconnect in the neighbor tets.
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

  // Clean the working lists.
  ierr = ArrayPoolRestart(m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolRestart(m->caveshbdlist);CHKERRQ(ierr);

  if (result) {*result = loc;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFormStarPolygon"
// formstarpolygon()    Form the star polygon of a point in facet.           //
//                                                                           //
// The polygon P is formed by all coplanar subfaces having 'pt' as a vertex. //
// P is bounded by segments, e.g, if no segments, P is the full star of pt.  //
//                                                                           //
// 'trilist' T returns the subfaces, it has one of such subfaces on input.   //
// In addition, if f is in T, then sapex(f) = p. 'vertlist' V are verts of P.//
// Topologically, T is the star of p; V and the edges of T are the link of p.//
/* tetgenmesh::formstarpolygon() */
PetscErrorCode TetGenMeshFormStarPolygon(TetGenMesh *m, point pt, List *trilist, List *vertlist)
{
  face steinsh  = {PETSC_NULL, 0}, lnextsh = {PETSC_NULL, 0}, rnextsh = {PETSC_NULL, 0};
  face checkseg = {PETSC_NULL, 0};
  point pa, pb, pc, pd;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Get a subface f containing p.
  ierr = ListItem(trilist, 0, (void **) &steinsh);CHKERRQ(ierr);
  steinsh.shver = 0; // CCW
  // Let sapex(f) be p.
  for(i = 0; i < 3; i++) {
    if (sapex(&steinsh) == pt) break;
    senextself(&steinsh);
  }
  if (i >= 3) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
  // Add the edge f into list.
  ierr = ListSetItem(trilist, 0, (void **) &steinsh);CHKERRQ(ierr);
  pa = sorg(&steinsh);
  pb = sdest(&steinsh);
  if (vertlist) {
    // Add two verts a, b into V,
    ierr = ListAppend(vertlist, &pa, PETSC_NULL);CHKERRQ(ierr);
    ierr = ListAppend(vertlist, &pb, PETSC_NULL);CHKERRQ(ierr);
  }

  // Rotate edge pa to the left (CW) until meet pb or a segment.
  lnextsh = steinsh;
  pc = pa;
  do {
    senext2self(&lnextsh);
    if (sorg(&lnextsh) != pt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    sspivot(m, &lnextsh, &checkseg);
    if (checkseg.sh != m->dummysh) break; // Do not cross a segment.
    // Get neighbor subface n (must exist).
    spivotself(&lnextsh);
    if (lnextsh.sh == m->dummysh) break; // It's a hull edge.
    // Go to the edge ca opposite to p.
    if (sdest(&lnextsh) != pt) sesymself(&lnextsh);
    if (sdest(&lnextsh) != pt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    senext2self(&lnextsh);
    // Add n (at edge ca) to T.
    ierr = ListAppend(trilist, &lnextsh, PETSC_NULL);CHKERRQ(ierr);
    // Add edge ca to E.
    pc = sorg(&lnextsh);
    if (pc == pb) break; // Rotate back.
    if (vertlist) {
      // Add vert c into V.
      ierr = ListAppend(vertlist, &pc, PETSC_NULL);CHKERRQ(ierr);
    }
  } while (1);

  if (pc != pb) {
    // Rotate edge bp to the right (CCW) until meet a segment.
    rnextsh = steinsh;
    do {
      senextself(&rnextsh);
      if (sdest(&rnextsh) != pt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
      sspivot(m, &rnextsh, &checkseg);
      if (checkseg.sh != m->dummysh) break; // Do not cross a segment.
      // Get neighbor subface n (must exist).
      spivotself(&rnextsh);
      if (rnextsh.sh == m->dummysh) break; // It's a hull edge.
      // Go to the edge bd opposite to p.
      if (sorg(&rnextsh) != pt) sesymself(&rnextsh);
      if (sorg(&rnextsh) != pt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
      senextself(&rnextsh);
      // Add n (at edge bd) to T.
      ierr = ListAppend(trilist, &rnextsh, PETSC_NULL);CHKERRQ(ierr);
      // Add edge bd to E.
      pd = sdest(&rnextsh);
      if (pd == pa) break; // Rotate back.
      if (vertlist) {
        // Add vert d into V.
        ierr = ListAppend(vertlist, &pd, PETSC_NULL);CHKERRQ(ierr);
      }
    } while (1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetFacetAbovePoint"
// getfacetabovepoint()    Get a point above a plane pass through a facet.   //
//                                                                           //
// The calculcated point is saved in 'facetabovepointarray'. The 'abovepoint'//
// is set on return.                                                         //
/* tetgenmesh::getfacetabovepoint() */
PetscErrorCode TetGenMeshGetFacetAbovePoint(TetGenMesh *m, face *facetsh)
{
  TetGenOpts    *b  = m->b;
  List *verlist, *trilist, *tetlist;
  triface adjtet = {PETSC_NULL, 0, 0};
  point p1, p2, p3, pa;
  // enum locateresult loc;
  PetscReal smallcos, cosa;
  PetscReal largevol, volume;
  PetscReal v1[3], v2[3], len;
  int llen, smallidx, largeidx;
  int shmark;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  m->abovecount++;
  // Initialize working lists.
  ierr = ListCreate(sizeof(point *), PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &verlist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(face),    PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &trilist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(triface), PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &tetlist);CHKERRQ(ierr);

  // Get three pivotal points p1, p2, and p3 in the facet as a base triangle
  //   which is non-trivil and has good base angle (close to 90 degree).

  // p1 is chosen as the one which has the smallest index in pa, pb, pc.
  p1 = sorg(facetsh);
  pa = sdest(facetsh);
  if (pointmark(m, pa) < pointmark(m, p1)) p1 = pa;
  pa = sapex(facetsh);
  if (pointmark(m, pa) < pointmark(m, p1)) p1 = pa;
  // Form the star polygon of p1.
  ierr = ListAppend(trilist, facetsh, PETSC_NULL);CHKERRQ(ierr);
  ierr = TetGenMeshFormStarPolygon(m, p1, trilist, verlist);CHKERRQ(ierr);

  // Get the second pivotal point p2.
  ierr = ListItem(verlist, 0, (void **) &p2);CHKERRQ(ierr);
  // Get vector v1 = p1->p2.
  for(i = 0; i < 3; i++) v1[i] = p2[i] - p1[i];
  len = sqrt(dot(v1, v1));
  if (len <= 0.0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}  // p2 != p1.
  for(i = 0; i < 3; i++) v1[i] /= len;

  // Get the third pivotal point p3. p3 is chosen as the one in 'verlist'
  //   which forms an angle with v1 closer to 90 degree than others do.
  smallcos = 1.0; // The cosine value of 0 degree.
  smallidx = 1;   // Default value.
  ierr = ListLength(verlist, &llen);CHKERRQ(ierr);
  for(i = 1; i < llen; i++) {
    ierr = ListItem(verlist, i, (void **) &p3);CHKERRQ(ierr);
    for(j = 0; j < 3; j++) v2[j] = p3[j] - p1[j];
    len = sqrt(dot(v2, v2));
    if (len > 0.0) { // v2 is not too small.
      cosa = fabs(dot(v1, v2)) / len;
      if (cosa < smallcos) {
        smallidx = i;
        smallcos = cosa;
      }
    }
  }
  if (smallcos >= 1.0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}  // p1->p3 != p1->p2.
  ierr = ListItem(verlist, smallidx, (void **) &p3);CHKERRQ(ierr);
  ierr = ListClear(verlist);CHKERRQ(ierr);

  if (m->tetrahedrons->items > 0l) {
    // Get a tet having p1 as a vertex.
    ierr = TetGenMeshPoint2TetOrg(m, p1, &adjtet);CHKERRQ(ierr);
    if (org(&adjtet) != p1) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    if (adjtet.tet != m->dummytet) {
      // Get the star polyhedron of p1.
      ierr = ListAppend(tetlist, &adjtet, PETSC_NULL);CHKERRQ(ierr);
      ierr = TetGenMeshFormStarPolyhedron(m, p1, tetlist, verlist, PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  // Get the abovepoint in 'verlist'. It is the one form the largest valid
  //   volumw with the base triangle over other points in 'verlist.
  largevol = 0.0;
  largeidx = 0;
  ierr = ListLength(verlist, &llen);CHKERRQ(ierr);
  for(i = 0; i < llen; i++) {
    PetscBool isCoplanar;

    ierr = ListItem(verlist, i, (void **) &pa);CHKERRQ(ierr);
    volume = orient3d(p1, p2, p3, pa);
    ierr = TetGenMeshIsCoplanar(m, p1, p2, p3, pa, volume, b->epsilon * 1e+2, &isCoplanar);CHKERRQ(ierr);
    if (!isCoplanar) {
      if (fabs(volume) > largevol) {
        largevol = fabs(volume);
        largeidx = i;
      }
    }
  }

  // Do we have the abovepoint?
  if (largevol > 0.0) {
    ierr = ListItem(verlist, largeidx, (void **) &m->abovepoint);CHKERRQ(ierr);
    PetscInfo2(b->in, "    Chosen abovepoint %d for facet %d.\n", pointmark(m, m->abovepoint), shellmark(m, facetsh));
  } else {
    // Calculate an abovepoint for this facet.
    ierr = TetGenMeshFaceNormal(m, p1, p2, p3, v1, &len);CHKERRQ(ierr);
    if (len != 0.0) for (i = 0; i < 3; i++) v1[i] /= len;
    // Take the average edge length of the bounding box.
    len = (0.5*(m->xmax - m->xmin) + 0.5*(m->ymax - m->ymin) + 0.5*(m->zmax - m->zmin)) / 3.0;
    // Temporarily create a point. It will be removed by jettison();
    ierr = TetGenMeshMakePoint(m, &m->abovepoint);CHKERRQ(ierr);
    setpointtype(m, m->abovepoint, UNUSEDVERTEX);
    m->unuverts++;
    for(i = 0; i < 3; i++) m->abovepoint[i] = p1[i] + len * v1[i];
    PetscInfo2(b->in, "    Calculated abovepoint %d for facet %d.\n", pointmark(m, m->abovepoint), shellmark(m, facetsh));
  }
  // Save the abovepoint in 'facetabovepointarray'.
  shmark = shellmark(m, facetsh);
  m->facetabovepointarray[shmark] = m->abovepoint;
  ierr = ListDestroy(&trilist);CHKERRQ(ierr);
  ierr = ListDestroy(&tetlist);CHKERRQ(ierr);
  ierr = ListDestroy(&verlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshIncrFlipDelaunaySub"
// incrflipdelaunaysub()    Create a DT from a 3D coplanar point set using   //
//                          the incremental flip algorithm.                  //
//                                                                           //
// Let T be the current Delaunay triangulation (of vertices of a facet F).   //
// 'shmark', the index of F in 'in->facetlist' (starts from 1).              //
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
  // Get the point array (saved in 'ptlist').
  insertarray = (point *) ptlist->base;
  ierr = ListLength(ptlist, &arraysize);CHKERRQ(ierr);
  if (arraysize < 3) {
    if (result) {*result = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  // Do calculation of 'abovepoint' if number of points > 3.
  aboveflag = (arraysize > 3) ? PETSC_TRUE : PETSC_FALSE;

  // The initial triangulation T only has one triangle formed by 3 not
  //   cillinear points of the set V = 'insertarray'. The first point:
  //   a = insertarray[0].

  epscount = 0;
  while(1) {
    for(i = 1; i < arraysize; i++) {
      det = distance(insertarray[0], insertarray[i]);
      if (det > (m->longest * eps)) break;
    }
    if (i < arraysize) {
      // Swap to move b from index i to index 1.
      swappt = insertarray[i];
      insertarray[i] = insertarray[1];
      insertarray[1] = swappt;
    }
    // Get the third point c, that is not collinear with a and b.
    for (i++; i < arraysize; i++) {
      PetscBool isCollinear;
      ierr = TetGenMeshIsCollinear(m, insertarray[0], insertarray[1], insertarray[i], eps, &isCollinear);CHKERRQ(ierr);
      if (!isCollinear) break;
    }
    if (i < arraysize) {
      // Swap to move c from index i to index 2.
      swappt = insertarray[i];
      insertarray[i] = insertarray[2];
      insertarray[2] = swappt;
      i = 3; // The next inserting point.
    } else {
      // The set of vertices is not good (or nearly degenerate).
      if ((eps == 0.0) || (epscount > 3)) {
        PetscInfo4(b->in, "Warning:  Discard an invalid facet #%d (%d, %d, %d, ...) looks like a line.\n",
                   shmark, pointmark(m, insertarray[0]), pointmark(m, insertarray[1]), pointmark(m, insertarray[2]));
        if (result) {*result = PETSC_FALSE;}
        PetscFunctionReturn(0);
      }
      // Decrease the eps, and continue to try.
      eps *= 1e-2;
      epscount++;
      continue;
    }
    break;
  } // while (true);

  // Create the initial triangle.
  ierr = TetGenMeshMakeShellFace(m, m->subfaces, &newsh);CHKERRQ(ierr);
  setsorg(&newsh, insertarray[0]);
  setsdest(&newsh, insertarray[1]);
  setsapex(&newsh, insertarray[2]);
  // Remeber the facet it belongs to.
  setshellmark(m, &newsh, shmark);
  // Set vertex type be FREESUBVERTEX if it has no type yet.
  if (pointtype(m, insertarray[0]) == FREEVOLVERTEX) {
    setpointtype(m, insertarray[0], FREESUBVERTEX);
  }
  if (pointtype(m, insertarray[1]) == FREEVOLVERTEX) {
    setpointtype(m, insertarray[1], FREESUBVERTEX);
  }
  if (pointtype(m, insertarray[2]) == FREEVOLVERTEX) {
    setpointtype(m, insertarray[2], FREESUBVERTEX);
  }
  // Let 'dummysh' point to it (for point location).
  m->dummysh[0] = sencode(&newsh);

  // Update the point-to-subface map.
  for(i = 0; i < 3; i++) {
    setpoint2sh(m, insertarray[i], sencode(&newsh));
  }

  // Are there area constraints?
  if (b->quality && in->facetconstraintlist) {
    idx = in->facetmarkerlist[shmark - 1]; // The actual facet marker.
    for(k = 0; k < in->numberoffacetconstraints; k++) {
      fmarker = (int) in->facetconstraintlist[k * 2];
      if (fmarker == idx) {
        area = in->facetconstraintlist[k * 2 + 1];
        setareabound(m, &newsh, area);
        break;
      }
    }
  }

  // Are there pbc conditions?
  if (m->checkpbcs) {
    idx = in->facetmarkerlist[shmark - 1]; // The actual facet marker.
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
    // Compute the 'abovepoint' for orient3d().
    m->abovepoint = m->facetabovepointarray[shmark];
    if (!m->abovepoint) {
      ierr = TetGenMeshGetFacetAbovePoint(m, &newsh);CHKERRQ(ierr);
    }
  }

  if (holes > 0) {
    // Project hole points onto the plane containing the facet.
    PetscReal prj[3];
    for(k = 0; k < holes; k++) {
      ierr = TetGenMeshProjPt2Face(m, &holelist[k * 3], insertarray[0], insertarray[1], insertarray[2], prj);CHKERRQ(ierr);
      for(j = 0; j < 3; j++) holelist[k * 3 + j] = prj[j];
    }
  }

  // Incrementally insert the rest of points into T.
  for(; i < arraysize; i++) {
    // Insert p_i.
    startsh.sh = m->dummysh;
    ierr = TetGenMeshSInsertVertex(m, insertarray[i], &startsh, PETSC_NULL, PETSC_TRUE, PETSC_TRUE, &loc);CHKERRQ(ierr);
    // The point-to-subface map has been updated.
    // Set p_i's type FREESUBVERTEX if it has no type yet.
    if (pointtype(m, insertarray[i]) == FREEVOLVERTEX) {
      setpointtype(m, insertarray[i], FREESUBVERTEX);
    }
  }

  if (result) {*result = PETSC_TRUE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFindDirectionSub"
// finddirectionsub()    Find the first subface in a facet on the path from  //
//                       one point to another.                               //
//                                                                           //
// Finds the subface in the facet that intersects a line segment drawn from  //
// the origin of `searchsh' to the point `tend', and returns the result in   //
// `searchsh'.  The origin of `searchsh' does not change,  even though the   //
// subface returned may differ from the one passed in.                       //
//                                                                           //
// The return value notes whether the destination or apex of the found face  //
// is collinear with the two points in question.                             //
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
  // Find the sign to simulate that abovepoint is 'above' the facet.
  adjustedgering_face(searchsh, CCW);
  // Make sure 'startpoint' is the origin.
  if (sorg(searchsh) != startpoint) senextself(searchsh);
  rightpoint = sdest(searchsh);
  leftpoint  = sapex(searchsh);
  ori = orient3d(startpoint, rightpoint, leftpoint, m->abovepoint);
  sign = ori > 0.0 ? -1 : 1;

  // Is `tend' to the left?
  ori = orient3d(tend, startpoint, m->abovepoint, leftpoint);
  leftccw  = ori * sign;
  leftflag = leftccw > 0.0;
  // Is `tend' to the right?
  ori = orient3d(startpoint, tend, m->abovepoint, rightpoint);
  rightccw  = ori * sign;
  rightflag = rightccw > 0.0;
  if (leftflag && rightflag) {
    // `searchsh' faces directly away from `tend'.  We could go left or
    //   right.  Ask whether it's a triangle or a boundary on the left.
    senext2(searchsh, &checksh);
    spivotself(&checksh);
    if (checksh.sh == m->dummysh) {
      leftflag = 0;
    } else {
      rightflag = 0;
    }
  }
  while (leftflag) {
    // Turn left until satisfied.
    senext2self(searchsh);
    spivotself(searchsh);
    if (searchsh->sh == m->dummysh) {
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error in finddirectionsub():  Unable to find a subface leading from %d to %d.\n",
               pointmark(m, startpoint), pointmark(m, tend));
    }
    if (sorg(searchsh) != startpoint) sesymself(searchsh);
    if (sorg(searchsh) != startpoint) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    leftpoint = sapex(searchsh);
    rightccw  = leftccw;
    ori = orient3d(tend, startpoint, m->abovepoint, leftpoint);
    leftccw  = ori * sign;
    leftflag = leftccw > 0.0;
  }
  while (rightflag) {
    // Turn right until satisfied.
    spivotself(searchsh);
    if (searchsh->sh == m->dummysh) {
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal error in finddirectionsub():  Unable to find a subface leading from %d to %d.\n",
               pointmark(m, startpoint), pointmark(m, tend));
    }
    if (sdest(searchsh) != startpoint) sesymself(searchsh);
    if (sdest(searchsh) != startpoint) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    senextself(searchsh);
    rightpoint = sdest(searchsh);
    leftccw = rightccw;
    ori = orient3d(startpoint, tend, m->abovepoint, rightpoint);
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
// insertsubseg()    Create a subsegment and insert it between two subfaces. //
//                                                                           //
// The new subsegment ab is inserted at the edge of subface 'tri'.  If ab is //
// not a hull edge, it is inserted between two subfaces.  If 'tri' is a hull //
// face, the initial face ring of ab will be set only one face which is self-//
// bonded.  The final face ring will be constructed in 'unifysegments()'.    //
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
  // Check if there's already a subsegment here.
  sspivot(m, tri, &newsubseg);
  if (newsubseg.sh == m->dummysh) {
    // Make new subsegment and initialize its vertices.
    ierr = TetGenMeshMakeShellFace(m, m->subsegs, &newsubseg);CHKERRQ(ierr);
    pa = sorg(tri);
    pb = sdest(tri);
    setsorg(&newsubseg, pa);
    setsdest(&newsubseg, pb);
    // Are there length constraints?
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
    // Bond new subsegment to the two subfaces it is sandwiched between.
    ssbond(m, tri, &newsubseg);
    spivot(tri, &oppotri);
    // 'oppotri' might be "out space".
    if (oppotri.sh != m->dummysh) {
      ssbond(m, &oppotri, &newsubseg);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshScoutSegmentSub"
// scoutsegmentsub()    Scout the first triangle on the path from one point  //
//                      to another, and check for completion (reaching the   //
//                      second point), a collinear point,or the intersection //
//                      of two segments.                                     //
//                                                                           //
// Returns true if the entire segment is successfully inserted, and false if //
// the job must be finished by constrainededge().                            //
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
    // The segment is already an edge.
    if (leftpoint == tend) {
      senext2self(searchsh);
    }
    // Insert a subsegment.
    ierr = TetGenMeshInsertSubseg(m, searchsh);CHKERRQ(ierr);
    if (isInserted) {*isInserted = PETSC_TRUE;}
    PetscFunctionReturn(0);
  } else if (collinear == LEFTCOLLINEAR) {
    // We've collided with a vertex between the segment's endpoints.
    // Make the collinear vertex be the triangle's origin.
    senextself(searchsh); // lprevself(*searchtri);
    // Insert a subsegment.
    ierr = TetGenMeshInsertSubseg(m, searchsh);CHKERRQ(ierr);
    // Insert the remainder of the segment.
    ierr = TetGenMeshScoutSegmentSub(m, searchsh, tend, isInserted);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (collinear == RIGHTCOLLINEAR) {
    // We've collided with a vertex between the segment's endpoints.
    // Insert a subsegment.
    ierr = TetGenMeshInsertSubseg(m, searchsh);CHKERRQ(ierr);
    // Make the collinear vertex be the triangle's origin.
    senextself(searchsh); // lnextself(*searchtri);
    // Insert the remainder of the segment.
    ierr = TetGenMeshScoutSegmentSub(m, searchsh, tend, isInserted);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    senext(searchsh, &crosssub); // lnext(*searchtri, crosstri);
    // Check for a crossing segment.
    sspivot(m, &crosssub, &crosssubseg);
    if (crosssubseg.sh != m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    if (isInserted) {*isInserted = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFlipEdgeRecursive"
// flipedgerecursive()    Flip an edge.                                      //
//                                                                           //
// This is a support routine for inserting segments into a CDT.              //
//                                                                           //
// Let 'flipedge' be ab, and two triangles abc, abd share at it.  ab may not //
// flipable if the four vertices a, b, c, and d are non-convex. If it is the //
// case, recursively flip ad or bd. Return when ab is flipped.               //
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
    oria = orient3d(pc, pd, m->abovepoint, pa);
    orib = orient3d(pc, pd, m->abovepoint, pb);
    doflip = (oria * orib < 0.0) ? PETSC_TRUE : PETSC_FALSE;
    if (doflip) {
      // Flip the edge (a, b) away.
      ierr = TetGenMeshFlip22Sub(m, flipedge, flipqueue);CHKERRQ(ierr);
      // Fix flipedge on edge e (c, d).
      ierr = TetGenMeshFindEdge_face(m, flipedge, pc, pd);CHKERRQ(ierr);
    } else {
      // ab is unflipable. Get the next edge (bd, or da) to flip.
      if (sorg(&fixupsh) != pb) sesymself(&fixupsh);
      if (sdest(&fixupsh) != pa) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
      if (fabs(oria) > fabs(orib)) {
        // acd has larger area. Choose da.
        senextself(&fixupsh);
      } else {
        // bcd has larger area. Choose bd.
        senext2self(&fixupsh);
      }
      // Flip the edge.
      ierr = TetGenMeshFlipEdgeRecursive(m, &fixupsh, flipqueue);CHKERRQ(ierr);
    }
  } while (!doflip);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshConstrainedEdge"
// constrainededge()    Force a segment into a CDT.                          //
//                                                                           //
// The segment s is recovered by flipping away the edges it intersects, and  //
// triangulating the polygons that form on each side of it.                  //
//                                                                           //
// Generates a single subsegment connecting `tstart' to `tend'. The triangle //
// `startsh' has `tstart' as its origin.                                     //
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
    // Loop edges oppo to tstart until find one crosses the segment.
    do {
      tright = sdest(startsh);
      tleft = sapex(startsh);
      // Is edge (tright, tleft) corss the segment.
      rori = orient3d(tstart, tright, m->abovepoint, tend);
      collision = (rori == 0.0) ? PETSC_TRUE : PETSC_FALSE;
      if (collision) break; // tright is on the segment.
      lori = orient3d(tstart, tleft, m->abovepoint, tend);
      collision = (lori == 0.0) ? PETSC_TRUE : PETSC_FALSE;
      if (collision) { //  tleft is on the segment.
        senext2self(startsh);
        break;
      }
      if (rori * lori < 0.0) break; // Find the crossing edge.
      // Both points are at one side of the segment.
      ierr = TetGenMeshFindDirectionSub(m, startsh, tend, PETSC_NULL);CHKERRQ(ierr);
    } while (PETSC_TRUE);
    if (collision) break;
    // Get the neighbor face at edge e (tright, tleft).
    senextself(startsh);
    // Flip the crossing edge.
    ierr = TetGenMeshFlipEdgeRecursive(m, startsh, flipqueue);CHKERRQ(ierr);
    // After flip, sorg(*startsh) == tstart.
    if (sorg(startsh) != tstart) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
  } while (sdest(startsh) != tend);

  // Insert a subsegment to make the segment permanent.
  ierr = TetGenMeshInsertSubseg(m, startsh);CHKERRQ(ierr);
  // If there was a collision with an interceding vertex, install another
  //   segment connecting that vertex with endpoint2.
  if (collision) {
    PetscBool isInsert;
    // Insert the remainder of the segment.
    ierr = TetGenMeshScoutSegmentSub(m, startsh, tend, &isInsert);CHKERRQ(ierr);
    if (!isInsert) {
      ierr = TetGenMeshConstrainedEdge(m, startsh, tend, flipqueue);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRecoverSegment"
// recoversegment()    Recover a segment in the surface triangulation.       //
/* tetgenmesh::recoversegment() */
PetscErrorCode TetGenMeshRecoverSegment(TetGenMesh *m, point tstart, point tend, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  face searchsh = {PETSC_NULL, 0};
  PetscBool isInsert;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscInfo2(b->in, "    Insert seg (%d, %d).\n", pointmark(m, tstart), pointmark(m, tend));

  // Find a triangle whose origin is the segment's first endpoint.
  ierr = TetGenMeshPoint2ShOrg(m, tstart, &searchsh);CHKERRQ(ierr);
  // Scout the segment and insert it if it is found.
  ierr = TetGenMeshScoutSegmentSub(m, &searchsh, tend, &isInsert);CHKERRQ(ierr);
  if (isInsert) {
    // The segment was easily inserted.
    PetscFunctionReturn(0);
  }
  // Insert the segment into the triangulation by flips.
  ierr = TetGenMeshConstrainedEdge(m, &searchsh, tend, flipqueue);CHKERRQ(ierr);
  // Some edges may need flipping.
  ierr = TetGenMeshLawson(m, flipqueue, PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInfectHullSub"
// infecthullsub()    Virally infect all of the triangles of the convex hull //
//                    that are not protected by subsegments.                 //
/* tetgenmesh::infecthullsub() */
PetscErrorCode TetGenMeshInfectHullSub(TetGenMesh *m, MemoryPool* viri)
{
  face hulltri = {PETSC_NULL, 0}, nexttri = {PETSC_NULL, 0}, starttri = {PETSC_NULL, 0};
  face hullsubseg = {PETSC_NULL, 0};
  shellface **deadshellface;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Find a triangle handle on the hull.
  hulltri.sh = m->dummysh;
  hulltri.shver = 0;
  spivotself(&hulltri);
  adjustedgering_face(&hulltri, CCW);
  // Remember where we started so we know when to stop.
  starttri = hulltri;
  // Go once counterclockwise around the convex hull.
  do {
    // Ignore triangles that are already infected.
    if (!sinfected(m, &hulltri)) {
      // Is the triangle protected by a subsegment?
      sspivot(m, &hulltri, &hullsubseg);
      if (hullsubseg.sh == m->dummysh) {
        // The triangle is not protected; infect it.
        if (!sinfected(m, &hulltri)) {
          sinfect(m, &hulltri);
          ierr = MemoryPoolAlloc(viri, (void **) &deadshellface);CHKERRQ(ierr);
          *deadshellface = hulltri.sh;
        }
      }
    }
    // To find the next hull edge, go clockwise around the next vertex.
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
// plaguesub()    Spread the virus from all infected triangles to any        //
//                neighbors not protected by subsegments.  Delete all        //
//                infected triangles.                                        //
//                                                                           //
// This is the procedure that actually creates holes and concavities.        //
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
  // Loop through all the infected triangles, spreading the virus to
  //   their neighbors, then to their neighbors' neighbors.
  ierr = MemoryPoolTraversalInit(viri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtri.sh = *virusloop;
    // Check each of the triangle's three neighbors.
    for(i = 0; i < 3; i++) {
      // Find the neighbor.
      spivot(&testtri, &neighbor);
      // Check for a subsegment between the triangle and its neighbor.
      sspivot(m, &testtri, &neighborsubseg);
      // Check if the neighbor is nonexistent or already infected.
      if ((neighbor.sh == m->dummysh) || sinfected(m, &neighbor)) {
        if (neighborsubseg.sh != m->dummysh) {
          // There is a subsegment separating the triangle from its
          //   neighbor, but both triangles are dying, so the subsegment
          //   dies too.
          ierr = TetGenMeshShellFaceDealloc(m, m->subsegs, neighborsubseg.sh);CHKERRQ(ierr);
          if (neighbor.sh != m->dummysh) {
            // Make sure the subsegment doesn't get deallocated again
            //   later when the infected neighbor is visited.
            ssdissolve(m, &neighbor);
          }
        }
      } else {                   // The neighbor exists and is not infected.
        if (neighborsubseg.sh == m->dummysh) {
          // There is no subsegment protecting the neighbor, so the
          //   neighbor becomes infected.
          sinfect(m, &neighbor);
          // Ensure that the neighbor's neighbors will be infected.
          ierr = MemoryPoolAlloc(viri, (void **) &deadshellface);CHKERRQ(ierr);
          *deadshellface = neighbor.sh;
        } else {               // The neighbor is protected by a subsegment.
          // Remove this triangle from the subsegment.
          ssbond(m, &neighbor, &neighborsubseg);
          // Update the point-to-subface map. 2009-07-21.
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

  ghostsh.sh = m->dummysh; // A handle of outer space.
  ierr = MemoryPoolTraversalInit(viri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtri.sh = *virusloop;
    // Record changes in the number of boundary edges, and disconnect
    //   dead triangles from their neighbors.
    for(i = 0; i < 3; i++) {
      spivot(&testtri, &neighbor);
      if (neighbor.sh != m->dummysh) {
        // Disconnect the triangle from its neighbor.
        // sdissolve(neighbor);
        sbond(&neighbor, &ghostsh);
      }
      senextself(&testtri);
    }
    // Return the dead triangle to the pool of triangles.
    ierr = TetGenMeshShellFaceDealloc(m, m->subfaces, testtri.sh);CHKERRQ(ierr);
    ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  }
  // Empty the virus pool.
  ierr = MemoryPoolRestart(viri);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCarveHolesSub"
// carveholessub()    Find the holes and infect them.  Find the area         //
//                    constraints and infect them.  Infect the convex hull.  //
//                    Spread the infection and kill triangles.  Spread the   //
//                    area constraints.                                      //
//                                                                           //
// This routine mainly calls other routines to carry out all these functions.//
/* tetgenmesh::carveholessub() */
PetscErrorCode TetGenMeshCarveHolesSub(TetGenMesh *m, int holes, PetscReal *holelist, MemoryPool *viri)
{
  face searchtri = {PETSC_NULL, 0};
  shellface **holetri;
  locateresult intersect;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Mark as infected any unprotected triangles on the boundary.
  //   This is one way by which concavities are created.
  ierr = TetGenMeshInfectHullSub(m, viri);CHKERRQ(ierr);

  if (holes > 0) {
    // Infect each triangle in which a hole lies.
    for(i = 0; i < 3 * holes; i += 3) {
      // Ignore holes that aren't within the bounds of the mesh.
      if ((holelist[i + 0] >= m->xmin) && (holelist[i + 0] <= m->xmax) &&
          (holelist[i + 1] >= m->ymin) && (holelist[i + 1] <= m->ymax) &&
          (holelist[i + 2] >= m->zmin) && (holelist[i + 2] <= m->zmax)) {
        // Start searching from some triangle on the outer boundary.
        searchtri.sh = m->dummysh;
        // Find a triangle that contains the hole.
        ierr = TetGenMeshLocateSub(m, &holelist[i], &searchtri, 0, 0.0, &intersect);CHKERRQ(ierr);
        if ((intersect != OUTSIDE) && (!sinfected(m, &searchtri))) {
          // Infect the triangle.  This is done by marking the triangle
          //   as infected and including the triangle in the virus pool.
          sinfect(m, &searchtri);
          ierr = MemoryPoolAlloc(viri, (void **) &holetri);CHKERRQ(ierr);
          *holetri = searchtri.sh;
        }
      }
    }
  }

  if (viri->items > 0) {
    // Carve the holes and concavities.
    ierr = TetGenMeshPlagueSub(m, viri);CHKERRQ(ierr);
  }
  // The virus pool should be empty now.
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshTriangulate"
// triangulate()    Triangulate a PSLG into a CDT.                           //
//                                                                           //
// A Planar Straight Line Graph (PSLG) P is actually a 2D polygonal region,  //
// possibly contains holes, segments and vertices in its interior. P is tri- //
// angulated into a set of _subfaces_ forming a CDT of P.                    //
//                                                                           //
// The vertices and segments of P are found in 'ptlist' and 'conlist', resp- //
// ectively. 'holelist' contains a list of hole points. 'shmark' will be set //
// to all subfaces of P.                                                     //
//                                                                           //
// The CDT is created directly in the pools 'subfaces' and 'subsegs'. It can //
// be retrived by a broadth-first searching starting from 'dummysh[0]'(debug //
// function 'outsurfmesh()' does it).                                        //
/* tetgenmesh::triangulate() */
PetscErrorCode TetGenMeshTriangulate(TetGenMesh *m, int shmark, PetscReal eps, List *ptlist, List *conlist, int holes, PetscReal *holelist, MemoryPool *viri, Queue *flipqueue)
{
  TetGenOpts    *b  = m->b;
  face newsh;
  point *cons;
  int len, len2, i;
  PetscBool isFlipped;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ListLength(ptlist,  &len);CHKERRQ(ierr);
  ierr = ListLength(conlist, &len2);CHKERRQ(ierr);
  PetscInfo4(b->in, "    %d vertices, %d segments, %d holes, shmark: %d.\n", len, len2, holes, shmark);

  // Create the DT of V by the 2D incremental flip algorithm.
  ierr = TetGenMeshIncrFlipDelaunaySub(m, shmark, eps, ptlist, holes, holelist, flipqueue, &isFlipped);CHKERRQ(ierr);
  if (isFlipped) {
    // Recover boundary edges.
    ierr = ListLength(ptlist, &len);CHKERRQ(ierr);
    if (len > 3) {
      // Insert segments into the DT.
      ierr = ListLength(conlist, &len2);CHKERRQ(ierr);
      for(i = 0; i < len2; i++) {
        ierr = ListItem(conlist, i, (void **) &cons);CHKERRQ(ierr);
        ierr = TetGenMeshRecoverSegment(m, cons[0], cons[1], flipqueue);CHKERRQ(ierr);
      }
      // Carve holes and concavities.
      ierr = TetGenMeshCarveHolesSub(m, holes, holelist, viri);CHKERRQ(ierr);
    } else if (len == 3) {
      // Insert 3 segments directly.
      newsh.sh    = m->dummysh;
      newsh.shver = 0;
      spivotself(&newsh);
      for(i = 0; i < 3; i++) {
        ierr = TetGenMeshInsertSubseg(m, &newsh);CHKERRQ(ierr);
        senextself(&newsh);
      }
    } else if (len == 2) {
      // This facet is actually a segment. It is not support by the mesh data
      //   strcuture. Hence the segment will not be maintained in the mesh.
      //   However, during segment recovery, the segment can be processed.
      ierr = ListItem(conlist, 0, (void **) &cons);CHKERRQ(ierr);
      ierr = TetGenMeshMakeShellFace(m, m->subsegs, &newsh);CHKERRQ(ierr);
      setsorg(&newsh, cons[0]);
      setsdest(&newsh, cons[1]);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshUnifySegments"
// unifysegments()    Unify identical segments and build facet connections.  //
//                                                                           //
// After creating the surface mesh. Each facet has its own segments.  There  //
// are duplicated segments between adjacent facets.  This routine has three  //
// purposes:                                                                 //
//   (1) identify the set of segments which have the same endpoints and      //
//       unify them into one segment, remove redundant ones;                 //
//   (2) create the face rings of the unified segments, hence setup the      //
//       connections between facets; and                                     //
//   (3) set a unique marker (1-based) for each segment.                     //
// On finish, each segment is unique and the face ring around it (right-hand //
// rule) is constructed. The connections between facets-facets are setup.    //
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

  // Compute a mapping from indices of vertices to subfaces.
  ierr = TetGenMeshMakeSubfaceMap(m, &idx2facelist, &facesperverlist);CHKERRQ(ierr);
  // Initialize 'sfacelist' for constructing the face link of each segment.
  ierr = ListCreate(sizeof(face), PETSC_NULL, PETSC_DECIDE, PETSC_DECIDE, &sfacelist);CHKERRQ(ierr);
  segmarker = 1;
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &subsegloop.sh);CHKERRQ(ierr);
  while(subsegloop.sh) {
    subsegloop.shver = 0; // For sure.
    torg = sorg(&subsegloop);
    tdest = sdest(&subsegloop);
    idx = pointmark(m, torg) - in->firstnumber;
    // Loop through the set of subfaces containing 'torg'.  Get all the
    //   subfaces containing the edge (torg, tdest). Save and order them
    //   in 'sfacelist', the ordering is defined by the right-hand rule
    //   with thumb points from torg to tdest.
    for(k = idx2facelist[idx]; k < idx2facelist[idx + 1]; k++) {
      sface.sh = facesperverlist[k];
      sface.shver = 0;
      // sface may be died due to the removing of duplicated subfaces.
      if (!isdead_face(&sface) && isfacehasedge(&sface, torg, tdest)) {
        // 'sface' contains this segment.
        ierr = TetGenMeshFindEdge_face(m, &sface, torg, tdest);CHKERRQ(ierr);
        // Save it in 'sfacelist'.
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
              break;  // Insert it after m.
            }
          }
          ierr = ListInsert(sfacelist, m1+1, &sface, PETSC_NULL);CHKERRQ(ierr);
        }
      }
    }
    ierr = ListLength(sfacelist, &len);CHKERRQ(ierr);
    PetscInfo3(b->in, "    Identifying %d segments of (%d  %d).\n", len, pointmark(m, torg), pointmark(m, tdest));
    // Set the connection between this segment and faces containing it,
    //   at the same time, remove redundant segments.
    for(k = 0; k < len; k++) {
      ierr = ListItem(sfacelist, k, (void **) &sface);CHKERRQ(ierr);
      sspivot(m, &sface, &testseg);
      // If 'testseg' is not 'subsegloop', it is a redundant segment that
      //   needs be removed. BE CAREFUL it may already be removed. Do not
      //   remove it twice, i.e., do test 'isdead()' together.
      if ((testseg.sh != subsegloop.sh) && !isdead_face(&testseg)) {
        ierr = TetGenMeshShellFaceDealloc(m, m->subsegs, testseg.sh);CHKERRQ(ierr);
      }
      // 'ssbond' bonds the subface and the segment together, and dissloves
      //   the old bond as well.
      ssbond(m, &sface, &subsegloop);
    }
    // Set connection between these faces.
    ierr = ListItem(sfacelist, 0, (void **) &sface);CHKERRQ(ierr);
    ierr = ListLength(sfacelist, &len);CHKERRQ(ierr);
    if (len > 1) {
      for(k = 1; k <= len; k++) {
        if (k < len) {
          ierr = ListItem(sfacelist, k, (void **) &sface1);CHKERRQ(ierr);
        } else {
          ierr = ListItem(sfacelist, 0, (void **) &sface1);CHKERRQ(ierr); // Form a face loop.
        }
        // Comment: For detecting invalid PLC, here we could check if the
        //   two subfaces "sface" and "sface1" are identical (skipped).
        PetscInfo6(b->in, "    Bond subfaces (%d, %d, %d) and (%d, %d, %d).\n", pointmark(m, torg), pointmark(m, tdest), pointmark(m, sapex(&sface)),
                   pointmark(m, torg), pointmark(m, tdest), pointmark(m, sapex(&sface1)));
        sbond1(&sface, &sface1);
        sface = sface1;
      }
    } else {
      // This segment belongs to only on subface.
      sdissolve(m, &sface);
    }
    // Set the unique segment marker into the unified segment.
    setshellmark(m, &subsegloop, segmarker);
    // Increase the marker.
    segmarker++;
    // Clear the working list.
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
// mergefacets()    Merge adjacent facets to be one facet if they are        //
//                  coplanar and have the same boundary marker.              //
//                                                                           //
// Segments between two merged facets will be removed from the mesh.  If all //
// segments around a vertex have been removed, change its vertex type to be  //
// FREESUBVERTEX. Edge flips will be performed to ensure the Delaunayness of //
// the triangulation of merged facets.                                       //
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
  // Create and initialize 'segspernodelist'.
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &segspernodelist);CHKERRQ(ierr);
  for(i = 0; i < m->points->items + 1; i++) segspernodelist[i] = 0;

  // Loop the segments, counter the number of segments sharing each vertex.
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  while(segloop.sh) {
    // Increment the number of sharing segments for each endpoint.
    for(i = 0; i < 2; i++) {
      j = pointmark(m, (point) segloop.sh[3 + i]);
      segspernodelist[j]++;
    }
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  }

  // Loop the segments, find out dead segments.
  ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
  ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  while(segloop.sh) {
    eorg = sorg(&segloop);
    edest = sdest(&segloop);
    spivot(&segloop, &parentsh);
    if (parentsh.sh != m->dummysh) {
      // This segment is not dangling.
      spivot(&parentsh, &neighsh);
      if (neighsh.sh != m->dummysh) {
        // This segment belongs to at least two facets.
        spivot(&neighsh, &neineighsh);
        if ((parentsh.sh != neighsh.sh) && (parentsh.sh == neineighsh.sh)) {
          // Exactly two subfaces at this segment.
          fidx1 = shellmark(m, &parentsh) - 1;
          fidx2 = shellmark(m, &neighsh) - 1;
          pbcflag = PETSC_FALSE;
          if (m->checkpbcs) {
            pbcflag = (shellpbcgroup(m, &parentsh) >= 0) || (shellpbcgroup(m, &neighsh) >= 0) ? PETSC_TRUE : PETSC_FALSE;
          }
          // Possibly merge them if they are not in the same facet.
          if ((fidx1 != fidx2) && !pbcflag) {
            // Test if they are coplanar.
            ori = orient3d(eorg, edest, sapex(&parentsh), sapex(&neighsh));
            if (ori != 0.0) {
              PetscBool isCoplanar;

              ierr = TetGenMeshIsCoplanar(m, eorg, edest, sapex(&parentsh), sapex(&neighsh), ori, b->epsilon, &isCoplanar);CHKERRQ(ierr);
              if (isCoplanar) {
                ori = 0.0; // They are assumed as coplanar.
              }
            }
            if (ori == 0.0) {
              mergeflag = (!in->facetmarkerlist || in->facetmarkerlist[fidx1] == in->facetmarkerlist[fidx2]) ? PETSC_TRUE : PETSC_FALSE;
              if (mergeflag) {
                // This segment becomes dead.
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
                // Add 'parentsh' to queue checking for flip.
                ierr = TetGenMeshEnqueueFlipEdge(m, &parentsh, flipqueue);CHKERRQ(ierr);
              }
            }
          }
        }
      } // neighsh.sh != dummysh
    } // parentsh.sh != dummysh
    ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &segloop.sh);CHKERRQ(ierr);
  }

  ierr = QueueLength(flipqueue, &len);CHKERRQ(ierr);
  if (len > 0) {
    // Restore the Delaunay property in the facet triangulation.
    ierr = TetGenMeshLawson(m, flipqueue, PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscFree(segspernodelist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMeshSurface"
// meshsurface()    Create the surface mesh of a PLC.                        //
//                                                                           //
// Let X be the PLC, the surface mesh S of X consists of triangulated facets.//
// S is created mainly in the following steps:                               //
//                                                                           //
// (1) Form the CDT of each facet of X separately (by routine triangulate()).//
// After it is done, the subfaces of each facet are connected to each other, //
// however there is no connection between facets yet.  Notice each facet has //
// its own segments, some of them are duplicated.                            //
//                                                                           //
// (2) Remove the redundant segments created in step (1) (by routine unify-  //
// segment()). The subface ring of each segment is created,  the connection  //
// between facets are established as well.                                   //
//                                                                           //
// The return value indicates the number of segments of X.                   //
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

  // Compute a mapping from indices to points.
  ierr = TetGenMeshMakeIndex2PointMap(m, &idx2verlist);CHKERRQ(ierr);
  // // Compute a mapping from points to tets for computing abovepoints.
  // makepoint2tetmap();
  // Initialize 'facetabovepointarray'.
  ierr = PetscMalloc((in->numberoffacets + 1) * sizeof(point), &m->facetabovepointarray);CHKERRQ(ierr);
  for(i = 0; i < in->numberoffacets + 1; i++) {
    m->facetabovepointarray[i] = PETSC_NULL;
  }
  if (m->checkpbcs) {
    // Initialize the global array 'subpbcgrouptable'.
    // createsubpbcgrouptable();
  }

  // Initialize working lists.
  ierr = MemoryPoolCreate(sizeof(shellface *), 1024, POINTER, 0, &viri);CHKERRQ(ierr);
  ierr = QueueCreate(sizeof(badface), PETSC_DECIDE, &flipqueue);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(point *), PETSC_NULL, 256, PETSC_DECIDE, &ptlist);CHKERRQ(ierr);
  ierr = ListCreate(sizeof(point *)*2, PETSC_NULL, 256, PETSC_DECIDE, &conlist);CHKERRQ(ierr);
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &worklist);CHKERRQ(ierr);
  for (i = 0; i < m->points->items + 1; i++) worklist[i] = 0;
  ierr = ArrayPoolCreate(sizeof(face), 10, &m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolCreate(sizeof(face), 10, &m->caveshbdlist);CHKERRQ(ierr);

  // Loop the facet list, triangulate each facet. On finish, all subfaces
  //   are in 'subfaces', all segments are in 'subsegs'. Notice: there're
  //   redundant segments.  Remember: All facet indices count from 1.
  for(shmark = 1; shmark <= in->numberoffacets; shmark++) {
    // Get a facet F.
    f = &in->facetlist[shmark - 1];

    // Process the duplicated points first, they are marked with type
    //   DUPLICATEDVERTEX by incrflipdelaunay().  Let p and q are dup.
    //   and the index of p is larger than q's, p is substituted by q.
    //   In a STL mesh, duplicated points are implicitly included.
    if ((b->object == STL) || m->dupverts) {
      // Loop all polygons of this facet.
      for(i = 0; i < f->numberofpolygons; i++) {
        p = &(f->polygonlist[i]);
        // Loop other vertices of this polygon.
        for(j = 0; j < p->numberofvertices; j++) {
          end1 = p->vertexlist[j];
          tstart = idx2verlist[end1 - in->firstnumber];
          if (pointtype(m, tstart) == DUPLICATEDVERTEX) {
            // Reset the index of vertex-j.
            tend = point2ppt(m, tstart);
            end2 = pointmark(m, tend);
            p->vertexlist[j] = end2;
          }
        }
      }
    }

    // Loop polygons of F, get the set V of vertices and S of segments.
    for(i = 0; i < f->numberofpolygons; i++) {
      // Get a polygon.
      p = &(f->polygonlist[i]);
      // Get the first vertex.
      end1 = p->vertexlist[0];
      if ((end1 < in->firstnumber) || (end1 >= in->firstnumber + in->numberofpoints)) {
        PetscInfo3(b->in, "Warning:  Invalid the 1st vertex %d of polygon %d in facet %d.\n", end1, i + 1, shmark);
        continue; // Skip this polygon.
      }
      tstart = idx2verlist[end1 - in->firstnumber];
      // Add tstart to V if it haven't been added yet.
      if (worklist[end1] == 0) {
        ierr = ListAppend(ptlist, &tstart, PETSC_NULL);CHKERRQ(ierr);
        worklist[end1] = 1;
      }
      // Loop other vertices of this polygon.
      for(j = 1; j <= p->numberofvertices; j++) {
        // get a vertex.
        if (j < p->numberofvertices) {
          end2 = p->vertexlist[j];
        } else {
          end2 = p->vertexlist[0];  // Form a loop from last to first.
        }
        if ((end2 < in->firstnumber) || (end2 >= in->firstnumber + in->numberofpoints)) {
          PetscInfo3(b->in, "Warning:  Invalid vertex %d in polygon %d in facet %d.\n", end2, i + 1, shmark);
        } else {
          if (end1 != end2) {
            // 'end1' and 'end2' form a segment.
            tend = idx2verlist[end2 - in->firstnumber];
            // Add tstart to V if it haven't been added yet.
            if (worklist[end2] == 0) {
              ierr = ListAppend(ptlist, &tend, PETSC_NULL);CHKERRQ(ierr);
              worklist[end2] = 1;
            }
            // Save the segment in S (conlist).
            ierr = ListAppend(conlist, PETSC_NULL, (void **) &cons);CHKERRQ(ierr);
            cons[0] = tstart;
            cons[1] = tend;
            // Set the start for next continuous segment.
            end1   = end2;
            tstart = tend;
          } else {
            // Two identical vertices represent an isolated vertex of F.
            if (p->numberofvertices > 2) {
              // This may be an error in the input, anyway, we can continue
              //   by simply skipping this segment.
              PetscInfo2(b->in, "Warning:  Polygon %d has two identical verts in facet %d.\n", i + 1, shmark);
            }
            // Ignore this vertex.
          }
        }
        // Is the polygon degenerate (a segment or a vertex)?
        if (p->numberofvertices == 2) break;
      }
    }
    // Unmark vertices.
    ierr = ListLength(ptlist, &len);CHKERRQ(ierr);
    for(i = 0; i < len; i++) {
      ierr = ListItem(ptlist, i, (void **) &tstart);CHKERRQ(ierr);
      end1 = pointmark(m, tstart);
      if (worklist[end1] != 1) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
      worklist[end1] = 0;
    }

    // Create a CDT of F.
    ierr = TetGenMeshTriangulate(m, shmark, b->epsilon * 1e+2, ptlist, conlist, f->numberofholes, f->holelist, viri, flipqueue);CHKERRQ(ierr);
    // Clear working lists.
    ierr = ListClear(ptlist);CHKERRQ(ierr);
    ierr = ListClear(conlist);CHKERRQ(ierr);
    ierr = MemoryPoolRestart(viri);CHKERRQ(ierr);
  }

  ierr = ArrayPoolDestroy(&m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&m->caveshbdlist);CHKERRQ(ierr);
  m->caveshlist   = PETSC_NULL;
  m->caveshbdlist = PETSC_NULL;

  // Unify segments in 'subsegs', remove redundant segments.  Face links of segments are also built.
  ierr = TetGenMeshUnifySegments(m);CHKERRQ(ierr);

  // Remember the number of input segments (for output).
  m->insegments = m->subsegs->items;

  if (m->checkpbcs) {
    // Create the global array 'segpbcgrouptable'.
    // createsegpbcgrouptable();
  }

  if (b->object == STL) {
    // Remove redundant vertices (for .stl input mesh).
    ierr = TetGenMeshJettisonNodes(m);CHKERRQ(ierr);
  }

  if (!b->nomerge && !b->nobisect && !m->checkpbcs) {
    // No '-M' switch - merge adjacent facets if they are coplanar.
    ierr = TetGenMeshMergeFacets(m, flipqueue);CHKERRQ(ierr);
  }

  // Create the point-to-segment map.
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

////                                                                       ////
////                                                                       ////
//// surface_cxx //////////////////////////////////////////////////////////////

//// constrained_cxx //////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshMarkAcuteVertices"
// A vertex v is called acute if there are two segments sharing at v forming //
// an acute angle (i.e. smaller than 90 degree).                             //
//                                                                           //
// This routine finds all acute vertices in the PLC and marks them as point- //
// type ACUTEVERTEX. The other vertices of segments which are non-acute will //
// be marked as NACUTEVERTEX.  Vertices which are not endpoints of segments  //
// (such as DUPLICATEDVERTEX, UNUSEDVERTEX, etc) are not infected.           //
//                                                                           //
// NOTE: This routine should be called before Steiner points are introduced. //
// That is, no point has type like FREESEGVERTEX, etc.                       //
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
  // Constructing a map from vertex to segments.
  ierr = TetGenMeshMakeSegmentMap(m, &idx2seglist, &segsperverlist);CHKERRQ(ierr);
  // Loop over the set of vertices.
  ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
  ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  while(pointloop) {
    idx = pointmark(m, pointloop) - in->firstnumber;
    // Only do test if p is an endpoint of some segments.
    if (idx2seglist[idx + 1] > idx2seglist[idx]) {
      // Init p to be non-acute.
      setpointtype(m, pointloop, NACUTEVERTEX);
      isacute = PETSC_FALSE;
      // Loop through all segments sharing at p.
      for(i = idx2seglist[idx]; i < idx2seglist[idx + 1] && !isacute; i++) {
        segloop.sh = segsperverlist[i];
        // segloop.shver = 0;
        if (sorg(&segloop) != pointloop) {sesymself(&segloop);}
        edest = sdest(&segloop);
        for(j = i + 1; j < idx2seglist[idx + 1] && !isacute; j++) {
          nextseg.sh = segsperverlist[j];
          // nextseg.shver = 0;
          if (sorg(&nextseg) != pointloop) {sesymself(&nextseg);}
          eapex = sdest(&nextseg);
          // Check the angle formed by segs (p, edest) and (p, eapex).
          for(k = 0; k < 3; k++) {
            v1[k] = edest[k] - pointloop[k];
            v2[k] = eapex[k] - pointloop[k];
          }
          L = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
          for(k = 0; k < 3; k++) v1[k] /= L;
          L = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
          for(k = 0; k < 3; k++) v2[k] /= L;
          D = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
          // Is D acute?
          isacute = D >= cosbound ? PETSC_TRUE : PETSC_FALSE;
        }
      }
      if (isacute) {
        // Mark p to be acute.
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
// finddirection()    Find the tet on the path from one point to another.    //
//                                                                           //
// The path starts from 'searchtet''s origin and ends at 'endpt'. On finish, //
// 'searchtet' contains a tet on the path, its origin does not change.       //
//                                                                           //
// The return value indicates one of the following cases (let 'searchtet' be //
// abcd, a is the origin of the path):                                       //
//   - ACROSSVERT, edge ab is collinear with the path;                       //
//   - ACROSSEDGE, edge bc intersects with the path;                         //
//   - ACROSSFACE, face bcd intersects with the path.                        //
//                                                                           //
// WARNING: This routine is designed for convex triangulations, and will not //
// generally work after the holes and concavities have been carved.          //
//   - BELOWHULL2, the mesh is non-convex and the searching for the path has //
//                 got stucked at a non-convex boundary face.                //
/* tetgenmesh::finddirection2() */
PetscErrorCode TetGenMeshFindDirection2(TetGenMesh *m, triface* searchtet, point endpt, interresult *result)
{
  TetGenOpts    *b  = m->b;
  triface neightet = {PETSC_NULL, 0, 0};
  point pa, pb, pc, pd, pn;
  enum {HMOVE, RMOVE, LMOVE} nextmove;
  enum {HCOPLANE, RCOPLANE, LCOPLANE, NCOPLANE} cop;
  PetscReal hori, rori, lori;
  PetscReal dmin, dist;

  PetscFunctionBegin;
  if ((!searchtet->tet) || (searchtet->tet == m->dummytet)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
  // The origin is fixed.
  pa = org(searchtet);
  if (searchtet->ver & 01) {
    // Switch to the 0th edge ring.
    esymself(searchtet);
    enextself(searchtet);
  }
  pb = dest(searchtet);
  if (pb == endpt) {
    // pa->pb is the search edge.
    if (result) {*result = INTERVERT;}
    PetscFunctionReturn(0);
  }
  pc = apex(searchtet);
  if (pc == endpt) {
    // pa->pc is the search edge.
    enext2self(searchtet);
    esymself(searchtet);
    if (result) {*result = INTERVERT;}
    PetscFunctionReturn(0);
  }

  // Walk through tets at pa until the right one is found.
  while (1) {
    pd = oppo(searchtet);

    PetscInfo5(b->in, "      From tet (%d, %d, %d, %d) to %d.\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), pointmark(m, pd), pointmark(m, endpt));

    // Check whether the opposite vertex is 'endpt'.
    if (pd == endpt) {
      // pa->pd is the search edge.
      fnextself(m, searchtet);
      enext2self(searchtet);
      esymself(searchtet);
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }

    // Now assume that the base face abc coincides with the horizon plane,
    //   and d lies above the horizon.  The search point 'endpt' may lie
    //   above or below the horizon.  We test the orientations of 'endpt'
    //   with respect to three planes: abc (horizon), bad (right plane),
    //   and acd (left plane).
    hori = orient3d(pa, pb, pc, endpt);
    rori = orient3d(pb, pa, pd, endpt);
    lori = orient3d(pa, pc, pd, endpt);
    m->orient3dcount += 3;

    // Now decide the tet to move.  It is possible there are more than one
    //   tet are viable moves. Use the opposite points of thier neighbors
    //   to discriminate, i.e., we choose the tet whose opposite point has
    //   the shortest distance to 'endpt'.
    if (hori > 0) {
      if (rori > 0) {
        if (lori > 0) {
          // Any of the three neighbors is a viable move.
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
          // Two tets, below horizon and below right, are viable.
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
          // Two tets, below horizon and below left, are viable.
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
          // The tet below horizon is chosen.
          nextmove = HMOVE;
        }
      }
    } else {
      if (rori > 0) {
        if (lori > 0) {
          // Two tets, below right and below left, are viable.
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
          // The tet below right is chosen.
          nextmove = RMOVE;
        }
      } else {
        if (lori > 0) {
          // The tet below left is chosen.
          nextmove = LMOVE;
        } else {
          // 'endpt' lies either on the plane(s) or across face bcd.
          if (hori == 0) {
            if (rori == 0) {
              // pa->'endpt' is COLLINEAR with pa->pb.
              if (result) {*result = INTERVERT;}
              PetscFunctionReturn(0);
            }
            if (lori == 0) {
              // pa->'endpt' is COLLINEAR with pa->pc.
              enext2self(searchtet);
              esymself(searchtet);
              if (result) {*result = INTERVERT;}
              PetscFunctionReturn(0);
            }
            // pa->'endpt' crosses the edge pb->pc.
            // enextself(*searchtet);
            // return INTEREDGE;
            cop = HCOPLANE;
            break;
          }
          if (rori == 0) {
            if (lori == 0) {
              // pa->'endpt' is COLLINEAR with pa->pd.
              fnextself(m, searchtet); // face abd.
              enext2self(searchtet);
              esymself(searchtet);
              if (result) {*result = INTERVERT;}
              PetscFunctionReturn(0);
            }
            // pa->'endpt' crosses the edge pb->pd.
            cop = RCOPLANE;
            break;
          }
          if (lori == 0) {
            // pa->'endpt' crosses the edge pc->pd.
            cop = LCOPLANE;
            break;
          }
          // pa->'endpt' crosses the face bcd.
          cop = NCOPLANE;
          break;
        }
      }
    }

    // Move to the next tet, fix pa as its origin.
    if (nextmove == RMOVE) {
      tfnextself(m, searchtet);
    } else if (nextmove == LMOVE) {
      enext2self(searchtet);
      tfnextself(m, searchtet);
      enextself(searchtet);
    } else { // HMOVE
      symedgeself(m, searchtet);
      enextself(searchtet);
    }
    // Assume convex case, we should not move to outside.
    if (searchtet->tet == m->dummytet) {
      // This should only happen when the domain is non-convex.
      if (result) {*result = BELOWHULL2;}
      PetscFunctionReturn(0);
    }
    if (org(searchtet) != pa) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    pb = dest(searchtet);
    pc = apex(searchtet);

  } // while (1)

  // Either case INTEREDGE or INTERFACE.
  // Now decide the degenerate cases.
  if (hori == 0) {
    if (rori == 0) {
      // pa->'endpt' is COLLINEAR with pa->pb.
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }
    if (lori == 0) {
      // pa->'endpt' is COLLINEAR with pa->pc.
      enext2self(searchtet);
      esymself(searchtet);
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }
    // pa->'endpt' crosses the edge pb->pc.
    if (result) {*result = INTEREDGE;}
    PetscFunctionReturn(0);
  }
  if (rori == 0) {
    if (lori == 0) {
      // pa->'endpt' is COLLINEAR with pa->pd.
      fnextself(m, searchtet); // face abd.
      enext2self(searchtet);
      esymself(searchtet);
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }
    // pa->'endpt' crosses the edge pb->pd.
    fnextself(m, searchtet); // face abd.
    esymself(searchtet);
    enextself(searchtet);
    if (result) {*result = INTEREDGE;}
    PetscFunctionReturn(0);
  }
  if (lori == 0) {
    // pa->'endpt' crosses the edge pc->pd.
    enext2fnextself(m, searchtet);  // face cad
    esymself(searchtet);
    if (result) {*result = INTEREDGE;}
    PetscFunctionReturn(0);
  }
  // pa->'endpt' crosses the face bcd.
  if (result) {*result = INTERFACE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshFindDirection3"
// finddirection3()    Used when finddirection2() returns BELOWHULL2.        //
/* tetgenmesh::finddirection3() */
PetscErrorCode TetGenMeshFindDirection3(TetGenMesh *m, triface *searchtet, point endpt, interresult *result)
{
  TetGenOpts    *b  = m->b;
  ArrayPool *startetlist;
  triface *parytet, oppoface = {PETSC_NULL, 0, 0}, neightet = {PETSC_NULL, 0, 0};
  point startpt, pa, pb, pc;
  interresult dir;
  int types[2], poss[4];
  int pos, i, j;
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
    // assert(org(*searchtet) == startpt);
    adjustedgering_triface(searchtet, CCW);
    if (org(searchtet) != startpt) {
      enextself(searchtet);
      if (org(searchtet) != startpt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    }
    // Go to the opposite face of startpt.
    enextfnext(m, searchtet, &oppoface);
    esymself(&oppoface);
    pa = org(&oppoface);
    pb = dest(&oppoface);
    pc = apex(&oppoface);
    // Check if face [a, b, c] intersects the searching path.
    ierr = TetGenMeshTriEdgeTest(m, pa, pb, pc, startpt, endpt, NULL, 1, types, poss, &isIntersect);CHKERRQ(ierr);
    if (isIntersect) {
      // They intersect. Get the type of intersection.
      dir = (interresult) types[0];
      pos = poss[0];
      break;
    } else {
      dir = DISJOINT;
    }
    // Get the neighbor tets.
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
    // This path passing a vertex of the face [a, b, c].
    if (pos == 0) {
      // The path acrosses pa.
      enext2self(searchtet);
      esymself(searchtet);
    } else if (pos == 1) {
      // The path acrosses pa.
    } else { // pos == 2
      // The path acrosses pc.
      fnextself(m, searchtet);
      enext2self(searchtet);
      esymself(searchtet);
    }
    if (result) {*result = INTERVERT;}
    PetscFunctionReturn(0);
  }
  if (dir == INTEREDGE) {
    // This path passing an edge of the face [a, b, c].
    if (pos == 0) {
      // The path intersects [pa, pb].
    } else if (pos == 1) {
      // The path intersects [pb, pc].
      fnextself(m, searchtet);
      enext2self(searchtet);
      esymself(searchtet);
    } else { // pos == 2
      // The path intersects [pc, pa].
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

  // The path does not intersect any tet at pa.
  if (result) {*result = BELOWHULL2;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshScoutSegment2"
// scoutsegment()    Look for a given segment in the tetrahedralization T.   //
//                                                                           //
// Search an edge in the tetrahedralization that matches the given segmment. //
// If such an edge exists, the segment is 'locked' at the edge. 'searchtet'  //
// returns this (constrained) edge. Otherwise, the segment is missing.       //
//                                                                           //
// The returned value indicates one of the following cases:                  //
//   - SHAREEDGE, the segment exists and is inserted in T;                   //
//   - INTERVERT, the segment intersects a vertex ('refpt').                 //
//   - INTEREDGE, the segment intersects an edge (in 'searchtet').           //
//   - INTERFACE, the segment crosses a face (in 'searchtet').               //
//                                                                           //
// If the returned value is INTEREDGE or INTERFACE, i.e., the segment is     //
// missing, 'refpt' returns the reference point for splitting thus segment,  //
// 'searchtet' returns a tet containing the 'refpt'.                         //
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
  // Is 'searchtet' a valid handle?
  if ((!searchtet->tet) || (searchtet->tet == m->dummytet)) {
    startpt = sorg(sseg);
    ierr = TetGenMeshPoint2TetOrg(m, startpt, searchtet);CHKERRQ(ierr);
  } else {
    startpt = sorg(sseg);
  }
  if (org(searchtet) != startpt) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
  endpt = sdest(sseg);

  PetscInfo2(b->in, "    Scout seg (%d, %d).\n", pointmark(m, startpt), pointmark(m, endpt));

  ierr = TetGenMeshFindDirection2(m, searchtet, endpt, &dir);CHKERRQ(ierr);

  if (dir == INTERVERT) {
    pd = dest(searchtet);
    if (pd == endpt) {
      // Found! Insert the segment.
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
        // Collision! This can happy during facet recovery.
        // See fig/dump-cavity-case19, -case20.
        if (checkseg.sh != sseg->sh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
      }
      // The job is done.
      if (result) {*result = SHAREEDGE;}
      PetscFunctionReturn(0);
    } else {
      // A point is on the path.
      *refpt = pd;
      if (result) {*result = INTERVERT;}
      PetscFunctionReturn(0);
    }
  }

  PetscInfo2(b->in, "    Scout ref point of seg (%d, %d).\n", pointmark(m, startpt), pointmark(m, endpt));
  facecount = m->across_face_count;

  enextfnextself(m, searchtet); // Go to the opposite face.
  symedgeself(m, searchtet); // Enter the adjacent tet.

  pa = org(searchtet);
  angmax = interiorangle(pa, startpt, endpt, PETSC_NULL);
  *refpt = pa;
  pb = dest(searchtet);
  ang = interiorangle(pb, startpt, endpt, PETSC_NULL);
  if (ang > angmax) {
    angmax = ang;
    *refpt = pb;
  }

  // Check whether two segments are intersecting.
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
  reftet = *searchtet; // Save the tet containing the refpt.

  // Search intersecting faces along the segment.
  while(1) {
    pd = oppo(searchtet);

    PetscInfo5(b->in, "      Passing face (%d, %d, %d, %d), dir(%d).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), pointmark(m, pd), (int) dir);
    m->across_face_count++;

    // Stop if we meet 'endpt'.
    if (pd == endpt) break;

    ang = interiorangle(pd, startpt, endpt, PETSC_NULL);
    if (ang > angmax) {
      angmax = ang;
      *refpt = pd;
      reftet = *searchtet;
    }

    // Find a face intersecting the segment.
    if (dir == INTERFACE) {
      // One of the three oppo faces in 'searchtet' intersects the segment.
      neightet.tet = searchtet->tet;
      neightet.ver = 0;
      for(i = 0; i < 3; i++) {
        int isIntersect;

        neightet.loc = locpivot[searchtet->loc][i];
        pa = org(&neightet);
        pb = dest(&neightet);
        pc = apex(&neightet);
        pd = oppo(&neightet); // The above point.
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
      if (dir == DISJOINT) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    } else { // dir == ACROSSEDGE
      // Check the two opposite faces (of the edge) in 'searchtet'.
      neightet = *searchtet;
      neightet.ver = 0;
      for(i = 0; i < 2; i++) {
        int isIntersect;

        neightet.loc = locverpivot[searchtet->loc][searchtet->ver][i];
        pa = org(&neightet);
        pb = dest(&neightet);
        pc = apex(&neightet);
        pd = oppo(&neightet); // The above point.
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
        // No intersection. Go to the next tet.
        dir = INTEREDGE;
        tfnextself(m, searchtet);
        continue;
      }
    }

    if (dir == INTERVERT) {
      // This segment passing a vertex. Choose it and return.
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
      // Get the edge intersects with the segment.
      for(i = 0; i < pos; i++) {
        enextself(&neightet);
      }
    }
    // Go to the next tet.
    symedge(m, &neightet, searchtet);

    if (dir == INTEREDGE) {
      // Check whether two segments are intersecting.
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
  } // while (1)

  // dir is either ACROSSVERT, or ACROSSEDGE, or ACROSSFACE.
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
// delaunizesegments()    Recover segments in a Delaunay tetrahedralization. //
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

  // Loop until 'subsegstack' is empty.
  while(m->subsegstack->objects > 0l) {
    // seglist is used as a stack.
    m->subsegstack->objects--;
    psseg = (face *) fastlookup(m->subsegstack, m->subsegstack->objects);
    sseg = *psseg;

    if (!sinfected(m, &sseg)) continue; // Not a missing segment.
    suninfect(m, &sseg);

    // Insert the segment.
    searchtet.tet = PETSC_NULL;
    ierr = TetGenMeshScoutSegment2(m, &sseg, &searchtet, &refpt, &dir);CHKERRQ(ierr);

    if (dir != SHAREEDGE) {
      // The segment is missing, split it.
      spivot(&sseg, &splitsh);
      if (dir != INTERVERT) {
        // Create the new point.
        ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
#if 1
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put code in");
#else
        getsegmentsplitpoint3(&sseg, refpt, newpt);
#endif
        setpointtype(m, newpt, FREESEGVERTEX);
        setpoint2sh(m, newpt, sencode(&sseg));
        // Split the segment by newpt.
        ierr = TetGenMeshSInsertVertex(m, newpt, &splitsh, &sseg, PETSC_TRUE, PETSC_FALSE, PETSC_NULL);CHKERRQ(ierr);
        // Insert newpt into the DT. If 'checksubfaces == 1' the current
        //   mesh is constrained Delaunay (but may not Delaunay).
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
// scoutsubface()    Look for a given subface in the tetrahedralization T.   //
//                                                                           //
// 'ssub' is the subface, denoted as abc. If abc exists in T, it is 'locked' //
// at the place where the two tets sharing at it.                            //
//                                                                           //
// 'convexflag' indicates the current mesh is convex (1) or non-convex (0).  //
//                                                                           //
// The returned value indicates one of the following cases:                  //
//   - SHAREFACE, abc exists and is inserted;                                //
//   - TOUCHEDGE, a vertex (the origin of 'searchtet') lies on ab.           //
//   - EDGETRIINT, all three edges of abc are missing.                       //
//   - ACROSSTET, a tet (in 'searchtet') crosses the facet containg abc.     //
//                                                                           //
// If the retunred value is ACROSSTET, the subface is missing.  'searchtet'  //
// returns a tet which shares the same edge as 'pssub'.                      //
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
    // Search an edge of 'ssub' in tetrahedralization.
    pssub->shver = 0;
    for(i = 0; i < 3; i++) {
      pa = sorg(pssub);
      pb = sdest(pssub);
      // Get a tet whose origin is pa.
      ierr = TetGenMeshPoint2TetOrg(m, pa, searchtet);CHKERRQ(ierr);
      // Search the edge from pa->pb.
      ierr = TetGenMeshFindDirection2(m, searchtet, pb, &dir);CHKERRQ(ierr);
      if (dir == INTERVERT) {
        if (dest(searchtet) == pb) {
          // Found the edge. Break the loop.
          break;
        } else {
          // A vertex lies on the search edge. Return it.
          enextself(searchtet);
          if (result) {*result = TOUCHEDGE;}
          PetscFunctionReturn(0);
        }
      } else if (dir == BELOWHULL2) {
        if (convexflag > 0) {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        }
        // The domain is non-convex, and we got stucked at a boundary face.
        ierr = TetGenMeshPoint2TetOrg(m, pa, searchtet);CHKERRQ(ierr);
        ierr = TetGenMeshFindDirection3(m, searchtet, pb, &dir);CHKERRQ(ierr);
        if (dir == INTERVERT) {
          if (dest(searchtet) == pb) {
            // Found the edge. Break the loop.
            break;
          } else {
            // A vertex lies on the search edge. Return it.
            enextself(searchtet);
            if (result) {*result = TOUCHEDGE;}
            PetscFunctionReturn(0);
          }
        }
      }
      senextself(pssub);
    }
    if (i == 3) {
      // None of the three edges exists.
      if (result) {*result = EDGETRIINT;} // ab intersects the face in 'searchtet'.
      PetscFunctionReturn(0);
    }
  } else {
    // 'searchtet' holds the current edge of 'pssub'.
    pa = org(searchtet);
    pb = dest(searchtet);
  }

  pc = sapex(pssub);

  PetscInfo4(b->in, "    Scout subface (%d, %d, %d) (%ld).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), m->subfacstack->objects);

  // Searchtet holds edge pa->pb. Search a face with apex pc.
  spintet = *searchtet;
  pd = apex(&spintet);
  hitbdry = 0;
  while (1) {
    if (pd == pc) {
      // Found! Insert the subface.
      tspivot(m, &spintet, &checksh); // SELF_CHECK
      if (checksh.sh == m->dummysh) {
        // Comment: here we know that spintet and pssub refer to the same
        //   edge and the same DIRECTION: pa->pb.
        if ((spintet.ver & 1) == 1) {
          // Stay in CCW edge ring.
          esymself(&spintet);
        }
        if (sorg(pssub) != org(&spintet)) {
          sesymself(pssub);
        }
        tsbond(m, &spintet, pssub);
        symself(&spintet);
        if (spintet.tet != m->dummytet) {
          tspivot(m, &spintet, &checksh); // SELF_CHECK
          if (checksh.sh != m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
          sesymself(pssub);
          tsbond(m, &spintet, pssub);
        }
        if (result) {*result = SHAREFACE;}
        PetscFunctionReturn(0);
      } else {
        *searchtet = spintet;
        if (checksh.sh != pssub->sh) {
          // Another subface is laready inserted.
          // Comment: This is possible when there are faked tets.
          if (result) {*result = COLLISIONFACE;}
          PetscFunctionReturn(0);
        } else {
          // The subface has already been inserted (when you do check).
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
// scoutcrosstet()    Scout a tetrahedron across a facet.                    //
//                                                                           //
// A subface (abc) of the facet (F) is given in 'pssub', 'searchtet' holds   //
// the edge ab, it is the tet starting the search.  'facpoints' contains all //
// points which are co-facet with a, b, and c.                               //
//                                                                           //
// The subface (abc) was produced by a 2D CDT algorithm under the Assumption //
// that F is flat. In real data, however, F may not be strictly flat.  Hence //
// a tet (abde) that crosses abc may be in one of the two cases: (i) abde    //
// intersects F in its interior, or (ii) abde intersects F on its boundary.  //
// In case (i) F (or part of it) is missing in DT and needs to be recovered. //
// In (ii) F is not missing, the surface mesh of F needs to be adjusted.     //
//                                                                           //
// This routine distinguishes the two cases by the returned value, which is  //
//   - INTERTET, if it is case (i), 'searchtet' is abde, d and e lies below  //
//     and above abc, respectively, neither d nor e is dummypoint; or        //
//   - INTERFACE, if it is case (ii), 'searchtet' is abde, where the face    //
//     abd intersects abc, i.e., d is co-facet with abc, e may be co-facet   //
//     with abc or dummypoint.                                               //
/* tetgenmesh::scoutcrosstet() */
PetscErrorCode TetGenMeshScoutCrossTet(TetGenMesh *m, face *pssub, triface *searchtet, ArrayPool *facpoints, interresult *result)
{
  TetGenOpts    *b  = m->b;
  triface spintet = {PETSC_NULL, 0, 0}, crossface = {PETSC_NULL, 0, 0};
  point pa, pb, pc, pd, pe;
  PetscReal ori, ori1, len, n[3];
  PetscReal r, dr, drmin;
  PetscBool cofacetflag;
  int hitbdry;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (facpoints) {
    // Infect all vertices of the facet.
    for(i = 0; i < (int) facpoints->objects; i++) {
      pd = * (point *) fastlookup(facpoints, i);
      pinfect(m, pd);
    }
  }

  // Search an edge crossing the facet containing abc.
  if (searchtet->ver & 01) {
    esymself(searchtet); // Adjust to 0th edge ring.
    sesymself(pssub);
  }

  pa = sorg(pssub);
  pb = sdest(pssub);
  pc = sapex(pssub);

  // 'searchtet' refers to edge pa->pb.
  if (org(searchtet)  != pa) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
  if (dest(searchtet) != pb) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}

  // Search an apex lies below the subface. Note that such apex may not
  //   exist which indicates there is a co-facet apex.
  cofacetflag = PETSC_FALSE;
  pd = apex(searchtet);
  spintet = *searchtet;
  hitbdry = 0;
  while (1) {
    ori = orient3d(pa, pb, pc, pd);
    if ((ori != 0) && pinfected(m, pd)) {
      ori = 0; // Force d be co-facet with abc.
    }
    if (ori > 0) {
      break; // Found a lower point (the apex of spintet).
    }
    // Go to the next face.
    if (!fnextself(m, &spintet)) {
      hitbdry++;
      if (hitbdry == 2) {
        cofacetflag = PETSC_TRUE; break; // Not found.
      }
      esym(searchtet, &spintet);
      if (!fnextself(m, &spintet)) {
        cofacetflag = PETSC_TRUE; break; // Not found.
      }
    }
    pd = apex(&spintet);
    if (pd == apex(searchtet)) {
      cofacetflag = PETSC_TRUE; break; // Not found.
    }
  }

  if (!cofacetflag) {
    if (hitbdry > 0) {
      // The edge direction is reversed, which means we have to reverse
      //   the face rotation direction to find the crossing edge d->e.
      esymself(&spintet);
    }
    // Keep the edge a->b be in the CCW edge ring of spintet.
    if (spintet.ver & 1) {
      symedgeself(m, &spintet);
      if (spintet.tet == m->dummytet) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    }
    // Search a tet whose apex->oppo crosses the face [a, b, c].
    //   -- spintet is a face [a, b, d].
    //   -- the apex (d) of spintet is below [a, b, c].
    while (1) {
      pe = oppo(&spintet);
      ori = orient3d(pa, pb, pc, pe);
      if ((ori != 0) && pinfected(m, pe)) {
        ori = 0; // Force it to be a coplanar point.
      }
      if (ori == 0) {
        cofacetflag = PETSC_TRUE;
        break; // Found a co-facet point.
      }
      if (ori < 0) {
        *searchtet = spintet;
        break;  // Found. edge [d, e].
      }
      // Go to the next tet.
      tfnextself(m, &spintet);
      if (spintet.tet == m->dummytet) {
        cofacetflag = PETSC_TRUE;
        break; // There is a co-facet point.
      }
    }
    // Now if "cofacetflag != true", searchtet contains a cross tet (abde),
    //   where d and e lie below and above abc, respectively, and
    //   orient3d(a, b, d, e) < 0.
  }

  if (cofacetflag) {
    // There are co-facet points. Calculate a point above the subface.
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
    // Search a co-facet point d, s.t. (i) [a, b, d] intersects [a, b, c],
    //   AND (ii) a, b, c, d has the closet circumradius of [a, b, c].
    // NOTE: (ii) is needed since there may be several points satisfy (i).
    //   For an example, see file2.poly.
    ierr = TetGenMeshCircumsphere(m, pa, pb, pc, PETSC_NULL, n, &r, PETSC_NULL);CHKERRQ(ierr);
    crossface.tet = PETSC_NULL;
    pe = apex(searchtet);
    spintet = *searchtet;
    hitbdry = 0;
    while (1) {
      pd = apex(&spintet);
      ori = orient3d(pa, pb, pc, pd);
      if ((ori == 0) || pinfected(m, pd)) {
        ori1 = orient3d(pa, pb, m->dummypoint, pd);
        if (ori1 > 0) {
          // [a, b, d] intersects with [a, b, c].
          if (pinfected(m, pd)) {
            len = DIST(n, pd);
            dr = fabs(len - r);
            if (crossface.tet == PETSC_NULL) {
              // This is the first cross face.
              crossface = spintet;
              drmin = dr;
            } else {
              if (dr < drmin) {
                crossface = spintet;
                drmin = dr;
              }
            }
          } else {
            if (ori != 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
            // Found a coplanar but not co-facet point (pd).
            SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error:  Invalid PLC! A point and a subface intersect\n  Point %d. Subface (#%d) (%d, %d, %d)\n",
                     pointmark(m, pd), shellmark(m, pssub), pointmark(m, pa), pointmark(m, pb), pointmark(m, pc));
          }
        }
      }
      // Go to the next face.
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
      if (!crossface.tet); {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not handled yet");}
    }
    *searchtet = crossface;
    m->dummypoint[0] = m->dummypoint[1] = m->dummypoint[2] = 0;
  }

  if (cofacetflag) {
    PetscInfo4(b->in, "    Found a co-facet face (%d, %d, %d) op (%d).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, apex(searchtet)), pointmark(m, oppo(searchtet)));
    if (facpoints) {
      // Unmark all facet vertices.
      for(i = 0; i < (int) facpoints->objects; i++) {
        pd = * (point *) fastlookup(facpoints, i);
        puninfect(m, pd);
      }
    }
    // Comment: Now no vertex is infected.
    if (result) {*result = INTERFACE;}
  } else {
    // Return a crossing tet.
    PetscInfo4(b->in, "    Found a crossing tet (%d, %d, %d, %d).\n", pointmark(m, pa), pointmark(m, pb), pointmark(m, apex(searchtet)), pointmark(m, pe));
    // Comment: if facpoints != NULL, co-facet vertices are stll infected.
    //   They will be uninfected in formcavity();
    if (result) {*result = INTERTET;} // abc intersects the volume of 'searchtet'.
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRecoverSubfaceByFlips"
// recoversubfacebyflips()   Recover a subface by flips in the surface mesh. //
//                                                                           //
// A subface [a, b, c] ('pssub') intersects with a face [a, b, d] ('cross-   //
// face'), where a, b, c, and d belong to the same facet.  It indicates that //
// the face [a, b, d] should appear in the surface mesh.                     //
//                                                                           //
// This routine recovers [a, b, d] in the surface mesh through a sequence of //
// 2-to-2 flips. No Steiner points is needed. 'pssub' returns [a, b, d].     //
//                                                                           //
// If 'facfaces' is not NULL, all flipped subfaces are queued for recovery.  //
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
  // Get the missing subface is [a, b, c].
  pa = sorg(pssub);
  pb = sdest(pssub);
  pc = sapex(pssub);

  // The crossface is [a, b, d, e].
  // assert(org(*crossface) == pa);
  // assert(dest(*crossface) == pb);
  pd = apex(crossface);
  pe = m->dummypoint; // oppo(*crossface);

  if (pe == m->dummypoint) {
    // Calculate a point above the faces.
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

  // Adjust face [a, b, c], so that edge [b, c] crosses edge [a, d].
  ori = orient3d(pb, pc, pe, pd);
  if (ori == 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}

  if (ori > 0) {
    // Swap a and b.
    sesymself(pssub);
    esymself(crossface); // symedgeself(*crossface);
    pa = sorg(pssub);
    pb = sdest(pssub);
    if (pe == m->dummypoint) {
      pe[0] = pe[1] = pe[2] = 0;
    }
    pe = m->dummypoint; // oppo(*crossface);
  }

  while (1) {
    // Flip edge [b, c] to edge [a, d].
    senext(pssub, &flipfaces[0]);
    sspivot(m, &flipfaces[0], &checkseg);
    if (checkseg.sh != m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    spivot(&flipfaces[0], &flipfaces[1]);

    stpivot(m, &flipfaces[1], &neightet);
    if (neightet.tet != m->dummytet) {
      // A recovered subface, clean sub<==>tet connections.
      tsdissolve(m, &neightet);
      symself(&neightet);
      tsdissolve(m, &neightet);
      stdissolve(m, &flipfaces[1]);
      sesymself(&flipfaces[1]);
      stdissolve(m, &flipfaces[1]);
      sesymself(&flipfaces[1]);
      // flipfaces[1] refers to edge [b, c] (either b->c or c->b).
    }

    ierr = TetGenMeshFlip22Sub(m, &(flipfaces[0]), PETSC_NULL);CHKERRQ(ierr);
    m->flip22count++;

    // Comment: now flipfaces[0] is [d, a, b], flipfaces[1] is [a, d, c].

    // Add them into list (make ensure that they must be recovered).
    ierr = ArrayPoolNewIndex(facfaces, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
    *parysh = flipfaces[0];
    ierr = ArrayPoolNewIndex(facfaces, (void **) &parysh, PETSC_NULL);CHKERRQ(ierr);
    *parysh = flipfaces[1];

    // Find the edge [a, b].
    senext(&flipfaces[0], pssub);
    if (sorg(pssub)  != pa) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    if (sdest(pssub) != pb) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}

    pc = sapex(pssub);
    if (pc == pd) break;

    if (pe == m->dummypoint) {
      // Calculate a point above the faces.
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
      ori = orient3d(pb, pc, pe, pd);
      if (ori == 0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
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
// formcavity()    Form the cavity of a missing region.                      //
//                                                                           //
// A missing region R is a set of co-facet (co-palanr) subfaces. 'pssub' is  //
// a missing subface [a, b, c]. 'crosstets' contains only one tet, [a, b, d, //
// e], where d and e lie below and above [a, b, c], respectively.  Other     //
// crossing tets are sought from this tet and saved in 'crosstets'.          //
//                                                                           //
// The cavity C is divided into two parts by R,one at top and one at bottom. //
// 'topfaces' and 'botfaces' return the upper and lower boundary faces of C. //
// 'toppoints' contains vertices of 'crosstets' in the top part of C, and so //
// does 'botpoints'. Both 'toppoints' and 'botpoints' contain vertices of R. //
//                                                                           //
// NOTE: 'toppoints' may contain points which are not vertices of any top    //
// faces, and so may 'botpoints'. Such points may belong to other facets and //
// need to be present after the recovery of this cavity (P1029.poly).        //
//                                                                           //
// A pair of boundary faces: 'firsttopface' and 'firstbotface', are saved.   //
// They share the same edge in the boundary of the missing region.           //
//                                                                           //
// 'facpoints' contains all vertices of the facet containing R.  They are    //
// used for searching the crossing tets. On input all vertices are infected. //
// They are uninfected after the cavity is formed.                           //
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
  // For triangle-edge test.
  interresult dir;
  int isIntersect;
  int types[2], poss[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Get the missing subface abc.
  pa = sorg(pssub);
  pb = sdest(pssub);
  pc = sapex(pssub);

  // Comment: Now all facet vertices are infected.

  // Get a crossing tet abde.
  parytet = (triface *) fastlookup(crosstets, 0); // face abd.
  // The edge de crosses the facet. d lies below abc.
  enext2fnext(m, parytet, &crosstet);
  enext2self(&crosstet);
  esymself(&crosstet); // the edge d->e at face [d,e,a]
  infect(m, &crosstet);
  *parytet = crosstet; // Save it in list.

  // Temporarily re-use 'topfaces' for storing crossing edges.
  crossedges = topfaces;
  ierr = ArrayPoolNewIndex(crossedges, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
  *parytet = crosstet;

  // Collect all crossing tets.  Each cross tet is saved in the standard
  //   form deab, where de is a corrsing edge, orient3d(d,e,a,b) < 0.
  // NOTE: hull tets may be collected. See fig/dump-cavity-case2a(b).lua.
  //   Make sure that neither d nor e is dummypoint.
  for(i = 0; i < (int) crossedges->objects; i++) {
    crosstet = * (triface *) fastlookup(crossedges, i);
    // It may already be tested.
    if (!edgemarked(m, &crosstet)) {
      // Collect all tets sharing at the edge.
      pg = apex(&crosstet);
      spintet = crosstet;
      while (1) {
        // Mark this edge as tested.
        markedge(m, &spintet);
        if (!infected(m, &spintet)) {
          infect(m, &spintet);
          ierr = ArrayPoolNewIndex(crosstets, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = spintet;
        }
        // Go to the neighbor tet.
        tfnextself(m, &spintet);
        if (spintet.tet != m->dummytet) {
          // Check the validity of the PLC.
          tspivot(m, &spintet, &checksh);
          if (checksh.sh != m->dummysh) {
            SETERRQ8(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error:  Invalid PLC! Two subfaces intersect.\n  1st (#%4d): (%d, %d, %d)\n  2nd (#%4d): (%d, %d, %d)\n",
                     shellmark(m, pssub), pointmark(m, pa), pointmark(m, pb), pointmark(m, pc), shellmark(m, &checksh),
                     pointmark(m, sorg(&checksh)), pointmark(m, sdest(&checksh)), pointmark(m, sapex(&checksh)));
          }
        } else {
          // Encounter a boundary face.
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not handled yet");
        }
        if (apex(&spintet) == pg) break;
      }
      // Detect new cross edges.
      // Comment: A crossing edge must intersect one missing subface of
      //   this facet. We do edge-face tests.
      pd = org(&spintet);
      pe = dest(&spintet);
      while (1) {
        // Remember: spintet is edge d->e, d lies below [a, b, c].
        pf = apex(&spintet);
        // if (pf != dummypoint) { // Do not grab a hull edge.
        if (!pinfected(m, pf)) {
            for(j = 0; j < (int) facfaces->objects; j++) {
              parysh = (face *) fastlookup(facfaces, j);
              pa = sorg(parysh);
              pb = sdest(parysh);
              pc = sapex(parysh);
              // Check if pd->pf crosses the facet.
              ierr = TetGenMeshTriEdgeTest(m, pa, pb, pc, pd, pf, PETSC_NULL, 1, types, poss, &isIntersect);CHKERRQ(ierr);
              if (isIntersect) {
                dir = (interresult) types[0];
                if ((dir == INTEREDGE) || (dir == INTERFACE)) {
                  // The edge d->f corsses the facet.
                  enext2fnext(m, &spintet, &neightet);
                  esymself(&neightet); // d->f.
                  // pd must lie below the subface.
                  break;
                }
              }
              // Check if pe->pf crosses the facet.
              ierr = TetGenMeshTriEdgeTest(m, pa, pb, pc, pe, pf, PETSC_NULL, 1, types, poss, &isIntersect);CHKERRQ(ierr);
              if (isIntersect) {
                dir = (interresult) types[0];
                if ((dir == INTEREDGE) || (dir == INTERFACE)) {
                  // The edge f->e crosses the face.
                  enextfnext(m, &spintet, &neightet);
                  esymself(&neightet); // f->e.
                  // pf must lie below the subface.
                  break;
                }
              }
            }
            // There must exist a crossing edge.
            if (j >= (int) facfaces->objects) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
            if (!edgemarked(m, &neightet)) {
              // Add a new cross edge.
              ierr = ArrayPoolNewIndex(crossedges, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
              *parytet = neightet;
            }
          }
        // }
        tfnextself(m, &spintet);
        if (spintet.tet == m->dummytet) {
          // Encounter a boundary face.
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not handled yet");
        }
        if (apex(&spintet) == pg) break;
      }
    }
  }

  // Unmark all facet vertices.
  for(i = 0; i < (int) facpoints->objects; i++) {
    ppt = (point *) fastlookup(facpoints, i);
    puninfect(m, *ppt);
  }

  // Comments: Now no vertex is marked. Next we will mark vertices which 
  //   belong to the top and bottom boundary faces of the cavity and put
  //   them in 'toppopints' and 'botpoints', respectively.

  // All cross tets are found. Unmark cross edges.
  for(i = 0; i < (int) crossedges->objects; i++) {
    crosstet = * (triface *) fastlookup(crossedges, i);
    if (edgemarked(m, &crosstet)) {
      // Add the vertices of the cross edge [d, e] in lists. It must be
      //   that d lies below the facet (i.e., its a bottom vertex).
      //   Note that a cross edge contains no dummypoint.
      pf = org(&crosstet);
      // assert(pf != dummypoint); // SELF_CHECK
      if (!pinfected(m, pf)) {
        pinfect(m, pf);
        ierr = ArrayPoolNewIndex(botpoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); // Add a bottom vertex.
        *ppt = pf;
      }
      pf = dest(&crosstet);
      // assert(pf != dummypoint); // SELF_CHECK
      if (!pinfected(m, pf)) {
        pinfect(m, pf);
        ierr = ArrayPoolNewIndex(toppoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); // Add a top vertex.
        *ppt = pf;
      }
      // Unmark this edge in all tets containing it.
      pg = apex(&crosstet);
      spintet = crosstet;
      while (1) {
        if (!edgemarked(m, &spintet)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
        unmarkedge(m, &spintet);
        tfnextself(m, &spintet); // Go to the neighbor tet.
        if (spintet.tet == m->dummytet) {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not handled yet");
        }
        if (apex(&spintet) == pg) break;
      }
    }
  }

  PetscInfo2(b->in, "    Formed cavity: %ld (%ld) cross tets (edges).\n", crosstets->objects, crossedges->objects);
  ierr = ArrayPoolRestart(crossedges);CHKERRQ(ierr);

  // Find a pair of cavity boundary faces from the top and bottom sides of
  //   the facet each, and they share the same edge. Save them in the
  //   global variables: firsttopface, firstbotface. They will be used in
  //   fillcavity() for gluing top and bottom new tets.
  for(i = 0; i < (int) crosstets->objects; i++) {
    crosstet = * (triface *) fastlookup(crosstets, i);
    enextfnext(m, &crosstet, &spintet);
    enextself(&spintet);
    symedge(m, &spintet, &neightet);
    // if (!infected(neightet)) {
    if ((neightet.tet == m->dummytet) || !infected(m, &neightet)) {
      // A top face.
      if (neightet.tet == m->dummytet) {
        // Create a fake tet to hold the boundary face.
        ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  // Create a faked tet.
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
        for(j = 0; j < 3; j++) { // Bond segments.
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
      continue; // Go to the next cross tet.
    }
    enext2fnext(m, &crosstet, &spintet);
    enext2self(&spintet);
    symedge(m, &spintet, &neightet);
    // if (!infected(neightet)) {
    if ((neightet.tet == m->dummytet) || !infected(m, &neightet)) {
      // A bottom face.
      if (neightet.tet == m->dummytet) {
        // Create a fake tet to hold the boundary face.
        ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  // Create a faked tet.
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
        for(j = 0; j < 3; j++) { // Bond segments.
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
  if (i >= (int) crosstets->objects) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}

  // Collect the top and bottom faces and the middle vertices. Since all top
  //   and bottom vertices have been marked in above. Unmarked vertices are
  //   middle vertices.
  // NOTE 1: Hull tets may be collected. Process them as normal one.
  //   (see fig/dump-cavity-case2.lua.)
  // NOTE 2: Some previously recovered subfaces may be completely
  //   contained in a cavity (see fig/dump-cavity-case6.lua). In such case,
  //   we create two faked tets to hold this subface, one at each side.
  //   The faked tets will be removed in fillcavity().
  for(i = 0; i < (int) crosstets->objects; i++) {
    crosstet = * (triface *) fastlookup(crosstets, i);
    enextfnext(m, &crosstet, &spintet);
    enextself(&spintet);
    symedge(m, &spintet, &neightet);
    // if (!infected(neightet)) {
    if ((neightet.tet == m->dummytet) || !infected(m, &neightet)) {
      // A top face.
      ierr = ArrayPoolNewIndex(topfaces, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      if (neightet.tet == m->dummytet) {
        // Create a fake tet to hold the boundary face.
        ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  // Create a faked tet.
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
        for(j = 0; j < 3; j++) { // Bond segments.
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
        // Check if this side is a subface.
        tspivot(m, &spintet, &neighsh);
        if (neighsh.sh != m->dummysh) {
          // Found a subface (inside the cavity)!
          ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  // Create a faked tet.
          setorg(&faketet, org(&spintet));
          setdest(&faketet, dest(&spintet));
          setapex(&faketet, apex(&spintet));
          setoppo(&faketet, m->dummypoint);
          marktest(m, &faketet);  // To distinguish it from other faked tets.
          sesymself(&neighsh);
          tsbond(m, &faketet, &neighsh); // Let it hold the subface.
          for(j = 0; j < 3; j++) { // Bond segments.
            tsspivot1(m, &spintet, &checkseg);
            if (checkseg.sh != m->dummysh) {
              tssbond1(m, &faketet, &checkseg);
            }
            enextself(&spintet);
            enextself(&faketet);
          }
          // Add a top face (at faked tet).
          ierr = ArrayPoolNewIndex(topfaces, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = faketet;
        }
      }
    }
    enext2fnext(m, &crosstet, &spintet);
    enext2self(&spintet);
    symedge(m, &spintet, &neightet);
    // if (!infected(neightet)) {
    if ((neightet.tet == m->dummytet) || !infected(m, &neightet)) {
      // A bottom face.
      ierr = ArrayPoolNewIndex(botfaces, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
      if (neightet.tet == m->dummytet) {
        // Create a fake tet to hold the boundary face.
        ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  // Create a faked tet.
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
        for(j = 0; j < 3; j++) { // Bond segments.
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
          // Found a subface (inside the cavity)!
          ierr = TetGenMeshMakeTetrahedron(m, &faketet);CHKERRQ(ierr);  // Create a faked tet.
          setorg(&faketet, org(&spintet));
          setdest(&faketet, dest(&spintet));
          setapex(&faketet, apex(&spintet));
          setoppo(&faketet, m->dummypoint);
          marktest(m, &faketet);  // To distinguish it from other faked tets.
          sesymself(&neighsh);
          tsbond(m, &faketet, &neighsh); // Let it hold the subface.
          for(j = 0; j < 3; j++) { // Bond segments.
            tsspivot1(m, &spintet, &checkseg);
            if (checkseg.sh != m->dummysh) {
              tssbond1(m, &faketet, &checkseg);
            }
            enextself(&spintet);
            enextself(&faketet);
          }
          // Add a bottom face (at faked tet).
          ierr = ArrayPoolNewIndex(botfaces, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = faketet;
        }
      }
    }
    // Add middle vertices if there are (skip dummypoint).
    pf = org(&spintet);
    if (!pinfected(m, pf)) {
      pinfect(m, pf);
      ierr = ArrayPoolNewIndex(botpoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); // Add a bottom vertex.
      *ppt = pf;
      ierr = ArrayPoolNewIndex(toppoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); // Add a top vertex.
      *ppt = pf;
    }
    pf = dest(&spintet);
    if (!pinfected(m, pf)) {
      pinfect(m, pf);
      ierr = ArrayPoolNewIndex(botpoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); // Add a bottom vertex.
      *ppt = pf;
      ierr = ArrayPoolNewIndex(toppoints, (void **) &ppt, PETSC_NULL);CHKERRQ(ierr); // Add a top vertex.
      *ppt = pf;
    }
  }

  // Unmark all collected top, bottom, and middle vertices.
  for(i = 0; i < (int) toppoints->objects; i++) {
    ppt = (point *) fastlookup(toppoints, i);
    puninfect(m, *ppt);
  }
  for(i = 0; i < (int) botpoints->objects; i++) {
    ppt = (point *) fastlookup(botpoints, i);
    puninfect(m, *ppt);
  }
  // Comments: Now no vertex is marked.
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshDelaunizeCavity"
// delaunizecavity()    Fill a cavity by Delaunay tetrahedra.                //
//                                                                           //
// The tetrahedralizing cavity is the half (top or bottom part) of the whole //
// cavity.  The boundary faces of the half cavity are given in 'cavfaces',   //
// the bounday faces of the internal facet are not given.  These faces will  //
// be recovered later in fillcavity().                                       //
//                                                                           //
// This routine first constructs the DT of the vertices by the Bowyer-Watson //
// algorithm.  Then it identifies the boundary faces of the cavity in DT.    //
// The DT is returned in 'newtets'.                                          //
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
// fillcavity()    Fill new tets into the cavity.                            //
//                                                                           //
// The new tets are stored in two disjoint sets(which share the same facet). //
// 'topfaces' and 'botfaces' are the boundaries of these two sets, respect-  //
// ively. 'midfaces' is empty on input, and will store faces in the facet.   //
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
// carvecavity()    Delete old tets and outer new tets of the cavity.        //
/* tetgenmesh::carvecavity() */
PetscErrorCode TetGenMeshCarveCavity(TetGenMesh *m, ArrayPool *crosstets, ArrayPool *topnewtets, ArrayPool *botnewtets)
{
  ArrayPool *newtets;
  triface *parytet, *pnewtet, neightet = {PETSC_NULL, 0, 0};
  int i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Delete the old tets in cavity.
  for(i = 0; i < (int) crosstets->objects; i++) {
    parytet = (triface *) fastlookup(crosstets, i);
    ierr = TetGenMeshTetrahedronDealloc(m, parytet->tet);CHKERRQ(ierr);
  }
  ierr = ArrayPoolRestart(crosstets);CHKERRQ(ierr); // crosstets will be re-used.

  // Collect infected new tets in cavity.
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
  // Collect all new tets in cavity.
  for(i = 0; i < (int) crosstets->objects; i++) {
    parytet = (triface *) fastlookup(crosstets, i);
    if (i == 0) {
      m->recenttet = *parytet; // Remember a live handle.
    }
    for(j = 0; j < 4; j++) {
      decode(parytet->tet[j], &neightet);
      if (marktested(m, &neightet)) { // Is it a new tet?
        if (!infected(m, &neightet)) {
          // Find an interior tet.
          if (neightet.tet == m->dummytet) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
          infect(m, &neightet);
          ierr = ArrayPoolNewIndex(crosstets, (void **) &pnewtet, PETSC_NULL);CHKERRQ(ierr);
          *pnewtet = neightet;
        }
      }
    }
  }

  // Delete outer new tets (those new tets which are not infected).
  for(k = 0; k < 2; k++) {
    newtets = (k == 0 ? topnewtets : botnewtets);
    if (newtets != NULL) {
      for(i = 0; i < (int) newtets->objects; i++) {
        parytet = (triface *) fastlookup(newtets, i);
        if (infected(m, parytet)) {
          // This is an interior tet.
          uninfect(m, parytet);
          unmarktest(m, parytet);
        } else {
          // An outer tet. Delete it.
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
// restorecavity()    Reconnect old tets and delete new tets of the cavity.  //
/* tetgenmesh::restorecavity() */
PetscErrorCode TetGenMeshRestoreCavity(TetGenMesh *m, ArrayPool *crosstets, ArrayPool *topnewtets, ArrayPool *botnewtets)
{
  triface *parytet, neightet = {PETSC_NULL, 0, 0};
  face checksh = {PETSC_NULL, 0};
  point *ppt;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Reconnect crossing tets to cavity boundary.
  for(i = 0; i < (int) crosstets->objects; i++) {
    parytet = (triface *) fastlookup(crosstets, i);
    if (!infected(m, parytet)) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");}
    if (i == 0) {
      m->recenttet = *parytet; // Remember a live handle.
    }
    parytet->ver = 0;
    for(parytet->loc = 0; parytet->loc < 4; parytet->loc++) {
      sym(parytet, &neightet);
      // The neighbor may be a deleted faked tet.
      if (isdead_triface(&neightet) || (neightet.tet == m->dummytet)) {
        dissolve(m, parytet);  // Detach a faked tet.
        // Remember a boundary tet.
        m->dummytet[0] = encode(parytet);
      } else if (!infected(m, &neightet)) {
        bond(m, parytet, &neightet);
        tspivot(m, parytet, &checksh);
        if (checksh.sh != m->dummysh) {
          tsbond(m, parytet, &checksh);
        }
      }
    }
    // Update the point-to-tet map.
    parytet->loc = 0;
    ppt = (point *) &(parytet->tet[4]);
    for(j = 0; j < 4; j++) {
      setpoint2tet(m, ppt[j], encode(parytet));
    }
  }

  // Uninfect all crossing tets.
  for(i = 0; i < (int) crosstets->objects; i++) {
    parytet = (triface *) fastlookup(crosstets, i);
    uninfect(m, parytet);
  }

  // Delete new tets.
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
// constrainedfacets()    Recover subfaces saved in 'subfacestack'.          //
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

  // Initialize arrays.
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

  // Loop until 'subfacstack' is empty.
  while (m->subfacstack->objects > 0l) {
    m->subfacstack->objects--;
    pssub = (face *) fastlookup(m->subfacstack, m->subfacstack->objects);
    ssub = *pssub;

    if (!ssub.sh[3]) continue; // Skip a dead subface.

    stpivot(m, &ssub, &neightet);
    if (neightet.tet == m->dummytet) {
      sesymself(&ssub);
      stpivot(m, &ssub, &neightet);
    }

    if (neightet.tet == m->dummytet) {
      // Find an unrecovered subface.
      smarktest(&ssub);
      ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
      *pssub = ssub;
      // Get all subfaces and vertices of the same facet.
      for(i = 0; i < (int) facfaces->objects; i++) {
        ssub = * (face *) fastlookup(facfaces, i);
        for (j = 0; j < 3; j++) {
          sspivot(m, &ssub, &checkseg);
          if (checkseg.sh == m->dummysh) {
            spivot(&ssub, &neighsh);
            if (neighsh.sh == m->dummysh) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack should not be empty");}
            if (!smarktested(&neighsh)) {
              // It may be already recovered.
              stpivot(m, &neighsh, &neightet);
              if (neightet.tet == m->dummytet) {
                sesymself(&neighsh);
                stpivot(m, &neighsh, &neightet);
              }
              if (neightet.tet == m->dummytet) {
                // Add it into list.
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
        } // j
      } // i
      // Have found all facet subfaces (vertices). Uninfect them.
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

      // Loop until 'facfaces' is empty.
      while(facfaces->objects > 0l) {
        // Get the last subface of this array.
        facfaces->objects--;
        pssub = (face *) fastlookup(facfaces, facfaces->objects);
        ssub = *pssub;

        stpivot(m, &ssub, &neightet);
        if (neightet.tet == m->dummytet) {
          sesymself(&ssub);
          stpivot(m, &ssub, &neightet);
        }

        if (neightet.tet != m->dummytet) continue; // Not a missing subface.

        // Insert the subface.
        searchtet.tet = PETSC_NULL;
        ierr = TetGenMeshScoutSubface(m, &ssub, &searchtet, 1, &dir);CHKERRQ(ierr);
        if (dir == SHAREFACE) continue; // The subface is inserted.
        if (dir == COLLISIONFACE) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack should not be empty");}

        // Not exist. Push the subface back into stack.
        ierr = TetGenMeshRandomChoice(m, facfaces->objects + 1, &s);CHKERRQ(ierr);
        ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
        *pssub = * (face *) fastlookup(facfaces, s);
        * (face *) fastlookup(facfaces, s) = ssub;

        if (dir == EDGETRIINT) continue; // All three edges are missing.

        // Search for a crossing tet.
        ierr = TetGenMeshScoutCrossTet(m, &ssub, &searchtet, facpoints, &dir);CHKERRQ(ierr);

        if (dir == INTERTET) {
          // Recover subfaces by local retetrahedralization.
          cavitycount++;
          bakhullsize = m->hullsize;
          m->checksubsegs = m->checksubfaces = 0;
          ierr = ArrayPoolNewIndex(crosstets, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = searchtet;
          // Form a cavity of crossing tets.
          ierr = TetGenMeshFormCavity(m, &ssub, crosstets, topfaces, botfaces, toppoints, botpoints, facpoints, facfaces);CHKERRQ(ierr);
          delaunayflag = PETSC_TRUE;
          // Tetrahedralize the top part. Re-use 'midfaces'.
          ierr = TetGenMeshDelaunizeCavity(m, toppoints, topfaces, topshells, topnewtets, crosstets, midfaces, &success);CHKERRQ(ierr);
          if (success) {
            // Tetrahedralize the bottom part. Re-use 'midfaces'.
            ierr = TetGenMeshDelaunizeCavity(m, botpoints, botfaces, botshells, botnewtets, crosstets, midfaces, &success);CHKERRQ(ierr);
            if (success) {
              // Fill the cavity with new tets.
              ierr = TetGenMeshFillCavity(m, topshells, botshells, midfaces, facpoints, &success);CHKERRQ(ierr);
              if (success) {
                // Delete old tets and outer new tets.
                ierr = TetGenMeshCarveCavity(m, crosstets, topnewtets, botnewtets);CHKERRQ(ierr);
              }
            } else {
              delaunayflag = PETSC_FALSE;
            }
          } else {
            delaunayflag = PETSC_FALSE;
          }
          if (!success) {
            // Restore old tets and delete new tets.
            ierr = TetGenMeshRestoreCavity(m, crosstets, topnewtets, botnewtets);CHKERRQ(ierr);
          }
          m->hullsize = bakhullsize;
          m->checksubsegs = m->checksubfaces = 1;
        } else if (dir == INTERFACE) {
          // Recover subfaces by flipping edges in surface mesh.
#if 1
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put code in");
#else
          recoversubfacebyflips(&ssub, &searchtet, facfaces);
#endif
          success = PETSC_TRUE;
        } else { // dir == TOUCHFACE
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        }
        if (!success) break;
      } // while

      if (facfaces->objects > 0l) {
        // Found a non-Delaunay edge, split it (or a segment close to it).
        // Create a new point at the middle of this edge, its coordinates
        //   were saved in dummypoint in 'fillcavity()'.
        ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
        for(i = 0; i < 3; i++) newpt[i] = m->dummypoint[i];
        setpointtype(m, newpt, FREESUBVERTEX);
        setpoint2sh(m, newpt, sencode(&ssub));
        m->dummypoint[0] = m->dummypoint[1] = m->dummypoint[2] = 0;
        // Insert the new point. Starting search it from 'ssub'.
        ierr = TetGenMeshSplitSubEdge(m, newpt, &ssub, facfaces, facpoints);CHKERRQ(ierr);
        ierr = ArrayPoolRestart(facfaces);CHKERRQ(ierr);
      }
      // Clear the list of facet vertices.
      ierr = ArrayPoolRestart(facpoints);CHKERRQ(ierr);

      // Some subsegments may be queued, recover them.
      if (m->subsegstack->objects > 0l) {
        b->verbose--; // Suppress the message output.
        ierr = TetGenMeshDelaunizeSegments2(m);CHKERRQ(ierr);
        b->verbose++;
      }
      // Now the mesh should be constrained Delaunay.
    } // if (neightet.tet == NULL)
  }

  PetscInfo2(b->in, "  %ld subedge flips.\n  %ld cavities remeshed.\n", m->flip22count - bakflip22count, cavitycount);

  // Delete arrays.
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
// splitsubedge()    Split a non-Delaunay edge (not a segment) in the        //
//                   surface mesh of a facet.                                //
//                                                                           //
// The new point 'newpt' will be inserted in the tetrahedral mesh if it does //
// not cause any existing (sub)segments become non-Delaunay.  Otherwise, the //
// new point is not inserted and one of such subsegments will be split.      //
//                                                                           //
// Next,the actual inserted new point is also inserted into the surface mesh.//
// Non-Delaunay segments and newly created subfaces are queued for recovery. //
/* tetgenmesh::splitsubedge() */
PetscErrorCode TetGenMeshSplitSubEdge(TetGenMesh *m, point newpt, face *searchsh, ArrayPool *facfaces, ArrayPool *facpoints)
{
  triface searchtet = {PETSC_NULL, 0, 0};
  face *psseg, sseg = {PETSC_NULL, 0};
  point pa, pb;
  locateresult loc;
  int s, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Try to insert the point. Do not insert if it will encroach any segment (noencsegflag is TRUE). Queue encroacged subfaces.
  if (m->subsegstack->objects != 0l) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack should be empty");}
  searchtet = m->recenttet; // Start search it from recentet
  // Always insert this point, missing segments are queued. 2009-06-11.
  ierr = TetGenMeshInsertVertexBW(m, newpt, &searchtet, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &loc);CHKERRQ(ierr);

  if (loc == ENCSEGMENT) {
    // Some segments are encroached. Randomly pick one to split.
    if (m->subsegstack->objects == 0l) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stack should not be empty");}
    ierr = TetGenMeshRandomChoice(m, m->subsegstack->objects, &s);CHKERRQ(ierr);
    psseg = (face *) fastlookup(m->subsegstack, s);
    sseg  = *psseg;
    pa    = sorg(&sseg);
    pb    = sdest(&sseg);
    for(i = 0; i < 3; i++) {newpt[i] = 0.5 * (pa[i] + pb[i]);}
    setpointtype(m, newpt, FREESEGVERTEX);
    setpoint2sh(m, newpt, sencode(&sseg));
    // Uninfect all queued segments.
    for(i = 0; i < (int) m->subsegstack->objects; i++) {
      psseg = (face *) fastlookup(m->subsegstack, i);
      suninfect(m, psseg);
    }
    // Clear the queue.
    ierr = ArrayPoolRestart(m->subsegstack);CHKERRQ(ierr);
    // Split the segment. Two subsegments are queued.
    ierr = TetGenMeshSInsertVertex(m, newpt, searchsh, &sseg, PETSC_TRUE, PETSC_FALSE, PETSC_NULL);CHKERRQ(ierr);
    // Insert the point. Missing segments are queued.
    searchtet = m->recenttet; // Start search it from recentet
    ierr = TetGenMeshInsertVertexBW(m, newpt, &searchtet, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, PETSC_NULL);CHKERRQ(ierr);
  } else {
    // Set the abovepoint of f for point location.
    m->abovepoint = m->facetabovepointarray[shellmark(m, searchsh)];
    if (!m->abovepoint) {
      ierr = TetGenMeshGetFacetAbovePoint(m, searchsh);CHKERRQ(ierr);
    }
    // Insert the new point on facet. New subfaces are queued for reocvery.
    ierr = TetGenMeshSInsertVertex(m, newpt, searchsh, PETSC_NULL, PETSC_TRUE, PETSC_FALSE, &loc);CHKERRQ(ierr);
    if (loc == OUTSIDE) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshConstrainedFacets2"
// constrainedfacets()    Recover subfaces saved in 'subfacestack'.          //
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

  // Initialize arrays.
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

  // Loop until 'subfacstack' is empty.
  while(m->subfacstack->objects > 0l) {
    m->subfacstack->objects--;
    pssub = (face *) fastlookup(m->subfacstack, m->subfacstack->objects);
    ssub = *pssub;

    if (!ssub.sh[3]) continue; // Skip a dead subface.

    stpivot(m, &ssub, &neightet);
    if (neightet.tet == m->dummytet) {
      sesymself(&ssub);
      stpivot(m, &ssub, &neightet);
    }

    if (neightet.tet == m->dummytet) {
      // Find an unrecovered subface.
      smarktest(&ssub);
      ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
      *pssub = ssub;
      // Get all subfaces and vertices of the same facet.
      for(i = 0; i < (int) facfaces->objects; i++) {
        ssub = * (face *) fastlookup(facfaces, i);
        for(j = 0; j < 3; j++) {
          sspivot(m, &ssub, &checkseg);
          if (checkseg.sh == m->dummysh) {
            spivot(&ssub, &neighsh);
            if (neighsh.sh == m->dummysh) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
            if (!smarktested(&neighsh)) {
              // It may be already recovered.
              stpivot(m, &neighsh, &neightet);
              if (neightet.tet == m->dummytet) {
                sesymself(&neighsh);
                stpivot(m, &neighsh, &neightet);
              }
              if (neightet.tet == m->dummytet) {
                // Add it into list.
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
        } // j
      } // i
      // Have found all facet subfaces (vertices). Uninfect them.
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

      // Loop until 'facfaces' is empty.
      while(facfaces->objects > 0l) {
        // Get the last subface of this array.
        facfaces->objects--;
        pssub = (face *) fastlookup(facfaces, facfaces->objects);
        ssub = *pssub;

        stpivot(m, &ssub, &neightet);
        if (neightet.tet == m->dummytet) {
          sesymself(&ssub);
          stpivot(m, &ssub, &neightet);
        }

        if (neightet.tet != m->dummytet) continue; // Not a missing subface.

        // Insert the subface.
        searchtet.tet = PETSC_NULL;
        ierr = TetGenMeshScoutSubface(m, &ssub, &searchtet, 1, &dir);CHKERRQ(ierr);
        if (dir == SHAREFACE) continue; // The subface is inserted.
        if (dir == COLLISIONFACE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");

        // Not exist. Push the subface back into stack.
        ierr = TetGenMeshRandomChoice(m, facfaces->objects + 1, &s);CHKERRQ(ierr);
        ierr = ArrayPoolNewIndex(facfaces, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
        *pssub = * (face *) fastlookup(facfaces, s);
        * (face *) fastlookup(facfaces, s) = ssub;

        if (dir == EDGETRIINT) continue; // All three edges are missing.

        // Search for a crossing tet.
        ierr = TetGenMeshScoutCrossTet(m, &ssub, &searchtet, facpoints, &dir);CHKERRQ(ierr);

        if (dir == INTERTET) {
          // Recover subfaces by local retetrahedralization.
          cavitycount++;
          bakhullsize = m->hullsize;
          m->checksubsegs = m->checksubfaces = 0;
          ierr = ArrayPoolNewIndex(crosstets, (void **) &parytet, PETSC_NULL);CHKERRQ(ierr);
          *parytet = searchtet;
          // Form a cavity of crossing tets.
          ierr = TetGenMeshFormCavity(m, &ssub, crosstets, topfaces, botfaces, toppoints, botpoints, facpoints, facfaces);CHKERRQ(ierr);
          delaunayflag = PETSC_TRUE;
          // Tetrahedralize the top part. Re-use 'midfaces'.
          ierr = TetGenMeshDelaunizeCavity(m, toppoints, topfaces, topshells, topnewtets, crosstets, midfaces, &success);CHKERRQ(ierr);
          if (success) {
            // Tetrahedralize the bottom part. Re-use 'midfaces'.
            ierr = TetGenMeshDelaunizeCavity(m, botpoints, botfaces, botshells, botnewtets, crosstets, midfaces, &success);CHKERRQ(ierr);
            if (success) {
              // Fill the cavity with new tets.
              ierr = TetGenMeshFillCavity(m, topshells, botshells, midfaces, facpoints, &success);CHKERRQ(ierr);
              if (success) {
                // Delete old tets and outer new tets.
                ierr = TetGenMeshCarveCavity(m, crosstets, topnewtets, botnewtets);CHKERRQ(ierr);
              }
            } else {
              delaunayflag = PETSC_FALSE;
            }
          } else {
            delaunayflag = PETSC_FALSE;
          }
          if (!success) {
            // Restore old tets and delete new tets.
            ierr = TetGenMeshRestoreCavity(m, crosstets, topnewtets, botnewtets);CHKERRQ(ierr);
          }
          m->hullsize = bakhullsize;
          m->checksubsegs = m->checksubfaces = 1;
        } else if (dir == INTERFACE) {
          // Recover subfaces by flipping edges in surface mesh.
          ierr = TetGenMeshRecoverSubfaceByFlips(m, &ssub, &searchtet, facfaces);CHKERRQ(ierr);
          success = PETSC_TRUE;
        } else { // dir == TOUCHFACE
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This is wrong");
        }
        if (!success) break;
      } // while

      if (facfaces->objects > 0l) {
        // Found a non-Delaunay edge, split it (or a segment close to it).
        // Create a new point at the middle of this edge, its coordinates
        //   were saved in dummypoint in 'fillcavity()'.
        ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
        for (i = 0; i < 3; i++) newpt[i] = m->dummypoint[i];
        setpointtype(m, newpt, FREESUBVERTEX);
        setpoint2sh(m, newpt, sencode(&ssub));
        m->dummypoint[0] = m->dummypoint[1] = m->dummypoint[2] = 0;
        // Insert the new point. Starting search it from 'ssub'.
        ierr = TetGenMeshSplitSubEdge(m, newpt, &ssub, facfaces, facpoints);CHKERRQ(ierr);
        ierr = ArrayPoolRestart(facfaces);CHKERRQ(ierr);
      }
      // Clear the list of facet vertices.
      ierr = ArrayPoolRestart(facpoints);CHKERRQ(ierr);

      // Some subsegments may be queued, recover them.
      if (m->subsegstack->objects > 0l) {
        b->verbose--; // Suppress the message output.
        ierr = TetGenMeshDelaunizeSegments2(m);CHKERRQ(ierr);
        b->verbose++;
      }
      // Now the mesh should be constrained Delaunay.
    } // if (neightet.tet == NULL)
  }

  PetscInfo2(b->in, "  %ld subedge flips  %ld cavities remeshed.\n", m->flip22count - bakflip22count, cavitycount);

  // Delete arrays.
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
// formskeleton()    Form a constrained tetrahedralization.                  //
//                                                                           //
// The segments and facets of a PLS will be recovered.                       //
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
  // Put all segments into the list.
  if (b->nojettison == 1) {  // '-J' option (for debug)
    // The sequential order.
    ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
    for(i = 0; i < m->subsegs->items; i++) {
      ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &ssub.sh);CHKERRQ(ierr);
      sinfect(m, &ssub);  // Only save it once.
      ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
      *pssub = ssub;
    }
  } else {
    // Randomly order the segments.
    ierr = MemoryPoolTraversalInit(m->subsegs);CHKERRQ(ierr);
    for(i = 0; i < m->subsegs->items; i++) {
      ierr = TetGenMeshRandomChoice(m, i + 1, &s);CHKERRQ(ierr);
      // Move the s-th seg to the i-th.
      ierr = ArrayPoolNewIndex(m->subsegstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
      *pssub = * (face *) fastlookup(m->subsegstack, s);
      // Put i-th seg to be the s-th.
      ierr = TetGenMeshShellFaceTraverse(m, m->subsegs, &ssub.sh);CHKERRQ(ierr);
      sinfect(m, &ssub);  // Only save it once.
      pssub = (face *) fastlookup(m->subsegstack, s);
      *pssub = ssub;
    }
  }
  // Segments will be introduced.
  m->checksubsegs = 1;
  // Recover segments.
  ierr = TetGenMeshDelaunizeSegments2(m);CHKERRQ(ierr);
  // Randomly order the subfaces.
  ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
  for(i = 0; i < m->subfaces->items; i++) {
    ierr = TetGenMeshRandomChoice(m, i + 1, &s);CHKERRQ(ierr);
    // Move the s-th subface to the i-th.
    ierr = ArrayPoolNewIndex(m->subfacstack, (void **) &pssub, PETSC_NULL);CHKERRQ(ierr);
    *pssub = * (face *) fastlookup(m->subfacstack, s);
    // Put i-th subface to be the s-th.
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &ssub.sh);CHKERRQ(ierr);
    pssub = (face *) fastlookup(m->subfacstack, s);
    *pssub = ssub;
  }

  // Subfaces will be introduced.
  m->checksubfaces = 1;
  // Recover facets.
  ierr = TetGenMeshConstrainedFacets2(m);CHKERRQ(ierr);

  ierr = ArrayPoolDestroy(&m->caveshlist);CHKERRQ(ierr);
  ierr = ArrayPoolDestroy(&m->caveshbdlist);CHKERRQ(ierr);

  // Detach all segments from tets.
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
  // Now no segment is bonded to tets.
  m->checksubsegs = 0;
  // Delete the memory.
  ierr = MemoryPoolRestart(m->tet2segpool);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshInfectHull"
// infecthull()    Virally infect all of the tetrahedra of the convex hull   //
//                 that are not protected by subfaces.  Where there are      //
//                 subfaces, set boundary markers as appropriate.            //
//                                                                           //
// Memorypool 'viri' is used to return all the infected tetrahedra.          //
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
    // Is this tetrahedron on the hull?
    for(tetloop.loc = 0; tetloop.loc < 4; tetloop.loc++) {
      sym(&tetloop, &tsymtet);
      if (tsymtet.tet == m->dummytet) {
        // Is the tetrahedron protected by a subface?
        tspivot(m, &tetloop, &hullface);
        if (hullface.sh == m->dummysh) {
          // The tetrahedron is not protected; infect it.
          if (!infected(m, &tetloop)) {
            infect(m, &tetloop);
            ierr = MemoryPoolAlloc(viri, (void **) &deadtet);CHKERRQ(ierr);
            *deadtet = tetloop.tet;
            break;  // Go and get next tet.
          }
        } else {
          // The tetrahedron is protected; set boundary markers if appropriate.
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
// plague()    Spread the virus from all infected tets to any neighbors not  //
//             protected by subfaces.                                        //
//                                                                           //
// This routine identifies all the tetrahedra that will die, and marks them  //
// as infected.  They are marked to ensure that each tetrahedron is added to //
// the virus pool only once, so the procedure will terminate. 'viri' returns //
// all infected tetrahedra which are outside the domian.                     //
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
  // Loop through all the infected tetrahedra, spreading the virus to
  //   their neighbors, then to their neighbors' neighbors.
  ierr = MemoryPoolTraversalInit(viri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtet.tet = *virusloop;
    // Temporarily uninfect this tetrahedron, not necessary.
    uninfect(m, &testtet);
    // Check each of the tetrahedron's four neighbors.
    for(testtet.loc = 0; testtet.loc < 4; testtet.loc++) {
      // Find the neighbor.
      sym(&testtet, &neighbor);
      // Check for a shell between the tetrahedron and its neighbor.
      tspivot(m, &testtet, &neighsh);
      // Check if the neighbor is nonexistent or already infected.
      if ((neighbor.tet == m->dummytet) || infected(m, &neighbor)) {
        if (neighsh.sh != m->dummysh) {
          // There is a subface separating the tetrahedron from its neighbor,
          //   but both tetrahedra are dying, so the subface dies too.
          // Before deallocte this subface, dissolve the connections between
          //   other subfaces, subsegments and tetrahedra.
          neighsh.shver = 0;
          if (!firstdadsub) {
            firstdadsub = 1; // Report the problem once.
            PetscInfo3(b->in, "Warning:  Detecting an open face (%d, %d, %d).\n", pointmark(m, sorg(&neighsh)), pointmark(m, sdest(&neighsh)), pointmark(m, sapex(&neighsh)));
          }
          // For keep the same enext() direction.
          ierr = TetGenMeshFindEdge_triface(m, &testtet, sorg(&neighsh), sdest(&neighsh));CHKERRQ(ierr);
          for (i = 0; i < 3; i++) {
            sspivot(m, &neighsh, &testseg);
            if (testseg.sh != m->dummysh) {
              // A subsegment is found at this side, dissolve this subface
              //   from the face link of this subsegment.
              testseg.shver = 0;
              spinsh = neighsh;
              if (sorg(&spinsh) != sorg(&testseg)) {
                sesymself(&spinsh);
              }
              spivot(&spinsh, &casingout);
              if ((casingout.sh == spinsh.sh) || (casingout.sh == m->dummysh)) {
                // This is a trivial face link, only 'neighsh' itself,
                //   the subsegment at this side is also died.
                ierr = TetGenMeshShellFaceDealloc(m, m->subsegs, testseg.sh);CHKERRQ(ierr);
              } else {
                spinsh = casingout;
                do {
                  casingin = spinsh;
                  spivotself(&spinsh);
                } while (spinsh.sh != neighsh.sh);
                // Set the link casingin->casingout.
                sbond1(&casingin, &casingout);
                // Bond the subsegment anyway.
                ssbond(m, &casingin, &testseg);
              }
            }
            senextself(&neighsh);
            enextself(&testtet);
          }
          if (neighbor.tet != m->dummytet) {
            // Make sure the subface doesn't get deallocated again later
            //   when the infected neighbor is visited.
            tsdissolve(m, &neighbor);
          }
          // This subface has been separated.
          if (in->mesh_dim > 2) {
            ierr = TetGenMeshShellFaceDealloc(m, m->subfaces, neighsh.sh);CHKERRQ(ierr);
          } else {
            // Dimension is 2. keep it for output.
            // Dissolve tets at both sides of this subface.
            stdissolve(m, &neighsh);
            sesymself(&neighsh);
            stdissolve(m, &neighsh);
          }
        }
      } else {                   // The neighbor exists and is not infected.
        if (neighsh.sh == m->dummysh) {
          // There is no subface protecting the neighbor, infect it.
          infect(m, &neighbor);
          // Ensure that the neighbor's neighbors will be infected.
          ierr = MemoryPoolAlloc(viri, (void **) &deadtet);CHKERRQ(ierr);
          *deadtet = neighbor.tet;
        } else {               // The neighbor is protected by a subface.
          // Remove this tetrahedron from the subface.
          stdissolve(m, &neighsh);
          // The subface becomes a boundary.  Set markers accordingly.
          if (shellmark(m, &neighsh) == 0) {
            setshellmark(m, &neighsh, 1);
          }
          // This side becomes hull. Update the handle in dummytet.
          m->dummytet[0] = encode(&neighbor);
        }
      }
    }
    // Remark the tetrahedron as infected, so it doesn't get added to the
    //   virus pool again.
    infect(m, &testtet);
    ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRegionPlague"
// regionplague()    Spread regional attributes and/or volume constraints    //
//                   (from a .poly file) throughout the mesh.                //
//                                                                           //
// This procedure operates in two phases.  The first phase spreads an attri- //
// bute and/or a volume constraint through a (facet-bounded) region.  The    //
// second phase uninfects all infected tetrahedra, returning them to normal. //
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
  // Loop through all the infected tetrahedra, spreading the attribute
  //   and/or volume constraint to their neighbors, then to their neighbors'
  //   neighbors.
  ierr = MemoryPoolTraversalInit(regionviri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(regionviri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtet.tet = *virusloop;
    // Temporarily uninfect this tetrahedron, not necessary.
    uninfect(m, &testtet);
    if (b->regionattrib) {
      // Set an attribute.
      setelemattribute(m, testtet.tet, in->numberoftetrahedronattributes, attribute);
    }
    if (b->varvolume) {
      // Set a volume constraint.
      setvolumebound(m, testtet.tet, volume);
    }
    // Check each of the tetrahedron's four neighbors.
    for(testtet.loc = 0; testtet.loc < 4; testtet.loc++) {
      // Find the neighbor.
      sym(&testtet, &neighbor);
      // Check for a subface between the tetrahedron and its neighbor.
      tspivot(m, &testtet, &neighsh);
      // Make sure the neighbor exists, is not already infected, and
      //   isn't protected by a subface, or is protected by a nonsolid
      //   subface.
      if ((neighbor.tet != m->dummytet) && !infected(m, &neighbor) && (neighsh.sh == m->dummysh)) {
        // Infect the neighbor.
        infect(m, &neighbor);
        // Ensure that the neighbor's neighbors will be infected.
        ierr = MemoryPoolAlloc(regionviri, (void **) &regiontet);CHKERRQ(ierr);
        *regiontet = neighbor.tet;
      }
    }
    // Remark the tetrahedron as infected, so it doesn't get added to the
    //   virus pool again.
    infect(m, &testtet);
    ierr = MemoryPoolTraverse(regionviri, (void **) &virusloop);CHKERRQ(ierr);
  }

  // Uninfect all tetrahedra.
  PetscInfo(b->in, "  Unmarking marked tetrahedra.\n");
  ierr = MemoryPoolTraversalInit(regionviri);CHKERRQ(ierr);
  ierr = MemoryPoolTraverse(regionviri, (void **) &virusloop);CHKERRQ(ierr);
  while(virusloop) {
    testtet.tet = *virusloop;
    uninfect(m, &testtet);
    ierr = MemoryPoolTraverse(regionviri, (void **) &virusloop);CHKERRQ(ierr);
  }
  // Empty the virus pool.
  ierr = MemoryPoolRestart(regionviri);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshRemoveHoleTets"
// removeholetets()    Remove tetrahedra which are outside the domain.       //
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
  // Create and initialize 'tetspernodelist'.
  ierr = PetscMalloc((m->points->items + 1) * sizeof(int), &tetspernodelist);CHKERRQ(ierr);
  for(i = 0; i < m->points->items + 1; i++) tetspernodelist[i] = 0;

  // Loop the tetrahedra list, counter the number of tets sharing each node.
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &testtet.tet);CHKERRQ(ierr);
  while(testtet.tet) {
    // Increment the number of sharing tets for each endpoint.
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
    // Record changes in the number of boundary faces, and disconnect
    //   dead tetrahedra from their neighbors.
    for(testtet.loc = 0; testtet.loc < 4; testtet.loc++) {
      sym(&testtet, &neighbor);
      if (neighbor.tet == m->dummytet) {
        // There is no neighboring tetrahedron on this face, so this face
        //   is a boundary face.  This tetrahedron is being deleted, so this
        //   boundary face is deleted.
        m->hullsize--;
      } else {
        // Disconnect the tetrahedron from its neighbor.
        dissolve(m, &neighbor);
        // There is a neighboring tetrahedron on this face, so this face
        //   becomes a boundary face when this tetrahedron is deleted.
        m->hullsize++;
      }
    }
    // Check the four corners of this tet if they're isolated.
    for(i = 0; i < 4; i++) {
      checkpt = (point) testtet.tet[4 + i];
      j = pointmark(m, checkpt);
      tetspernodelist[j]--;
      if (tetspernodelist[j] == 0) {
        // If it is added volume vertex or '-j' is not used, delete it.
        if ((pointtype(m, checkpt) == FREEVOLVERTEX) || !b->nojettison) {
          setpointtype(m, checkpt, UNUSEDVERTEX);
          m->unuverts++;
        }
      }
    }
    // Return the dead tetrahedron to the pool of tetrahedra.
    ierr = TetGenMeshTetrahedronDealloc(m, testtet.tet);CHKERRQ(ierr);
    ierr = MemoryPoolTraverse(viri, (void **) &virusloop);CHKERRQ(ierr);
  }
  ierr = PetscFree(tetspernodelist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCarveHoles"
// carveholes()    Find the holes and infect them.  Find the volume          //
//                 constraints and infect them.  Infect the convex hull.     //
//                 Spread the infection and kill tetrahedra.  Spread the     //
//                 volume constraints.                                       //
//                                                                           //
// This routine mainly calls other routines to carry out all these functions.//
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

  // Initialize a pool of viri to be used for holes, concavities.
  ierr = MemoryPoolCreate(sizeof(tetrahedron *), 1024, POINTER, 0, &holeviri);CHKERRQ(ierr);
  // Mark as infected any unprotected tetrahedra on the boundary.
  ierr = TetGenMeshInfectHull(m, holeviri);CHKERRQ(ierr);

  if (in->numberofholes > 0) {
    // Allocate storage for the tetrahedra in which hole points fall.
    ierr = PetscMalloc(in->numberofholes * sizeof(triface), &holetets);CHKERRQ(ierr);
    // Infect each tetrahedron in which a hole lies.
    for(i = 0; i < 3 * in->numberofholes; i += 3) {
      // Ignore holes that aren't within the bounds of the mesh.
      if ((in->holelist[i + 0] >= m->xmin) && (in->holelist[i + 0] <= m->xmax) &&
          (in->holelist[i + 1] >= m->ymin) && (in->holelist[i + 1] <= m->ymax) &&
          (in->holelist[i + 2] >= m->zmin) && (in->holelist[i + 2] <= m->zmax)) {
        searchtet.tet = m->dummytet;
        // Find a tetrahedron that contains the hole.
        ierr = TetGenMeshLocate(m, &in->holelist[i], &searchtet, &intersect);CHKERRQ(ierr);
        if ((intersect != OUTSIDE) && (!infected(m, &searchtet))) {
          // Record the tetrahedron for processing carve hole.
          holetets[i / 3] = searchtet;
        }
      }
    }
    // Infect the hole tetrahedron.  This is done by marking the tet as
    //   infected and including the tetrahedron in the virus pool.
    for(i = 0; i < in->numberofholes; i++) {
      infect(m, &holetets[i]);
      ierr = MemoryPoolAlloc(holeviri, (void **) &holetet);CHKERRQ(ierr);
      *holetet = holetets[i].tet;
    }
    // Free up memory.
    ierr = PetscFree(holetets);CHKERRQ(ierr);
  }

  // Mark as infected all tets of the holes and concavities.
  ierr = TetGenMeshPlague(m, holeviri);CHKERRQ(ierr);
  // The virus pool contains all outside tets now.

  // Is -A switch in use.
  if (b->regionattrib) {
    // Assign every tetrahedron a regional attribute of zero.
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
    // Allocate storage for the tetrahedra in which region points fall.
    ierr = PetscMalloc(in->numberofregions * sizeof(triface), &regiontets);CHKERRQ(ierr);
    // Find the starting tetrahedron for each region.
    for(i = 0; i < in->numberofregions; i++) {
      regiontets[i].tet = m->dummytet;
      // Ignore region points that aren't within the bounds of the mesh.
      if ((in->regionlist[5 * i + 0] >= m->xmin) && (in->regionlist[5 * i + 0] <= m->xmax) &&
          (in->regionlist[5 * i + 1] >= m->ymin) && (in->regionlist[5 * i + 1] <= m->ymax) &&
          (in->regionlist[5 * i + 2] >= m->zmin) && (in->regionlist[5 * i + 2] <= m->zmax)) {
        searchtet.tet = m->dummytet;
        // Find a tetrahedron that contains the region point.
        ierr = TetGenMeshLocate(m, &in->regionlist[5 * i], &searchtet, &intersect);CHKERRQ(ierr);
        if ((intersect != OUTSIDE) && (!infected(m, &searchtet))) {
          // Record the tetrahedron for processing after the
          //   holes have been carved.
          regiontets[i] = searchtet;
        }
      }
    }
    // Initialize a pool to be used for regional attrs, and/or regional
    //   volume constraints.
    ierr = MemoryPoolCreate(sizeof(tetrahedron *), 1024, POINTER, 0, &regionviri);CHKERRQ(ierr);
    // Find and set all regions.
    for(i = 0; i < in->numberofregions; i++) {
      if (regiontets[i].tet != m->dummytet) {
        // Make sure the tetrahedron under consideration still exists.
        //   It may have been eaten by the virus.
        if (!isdead_triface(&(regiontets[i]))) {
          // Put one tetrahedron in the virus pool.
          infect(m, &regiontets[i]);
          ierr = MemoryPoolAlloc(regionviri, (void **) &regiontet);CHKERRQ(ierr);
          *regiontet = regiontets[i].tet;
          // Apply one region's attribute and/or volume constraint.
          ierr = TetGenMeshRegionPlague(m, regionviri, in->regionlist[5 * i + 3], in->regionlist[5 * i + 4]);CHKERRQ(ierr);
          // The virus pool should be empty now.
        }
      }
    }
    // Free up memory.
    ierr = PetscFree(regiontets);CHKERRQ(ierr);
    ierr = MemoryPoolDestroy(&regionviri);CHKERRQ(ierr);
  }

  // Now acutually remove the outside and hole tets.
  ierr = TetGenMeshRemoveHoleTets(m, holeviri);CHKERRQ(ierr);
  // The mesh is nonconvex now.
  m->nonconvex = 1;

  // Update the point-to-tet map.
  ierr = TetGenMeshMakePoint2TetMap(m);CHKERRQ(ierr);

  if (b->regionattrib) {
    if (b->regionattrib > 1) {
      // -AA switch. Assign each tet a region number (> 0).
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
      assignregionattribs();
#endif
    }
    // Note the fact that each tetrahedron has an additional attribute.
    in->numberoftetrahedronattributes++;
  }

  // Free up memory.
  ierr = MemoryPoolDestroy(&holeviri);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

////                                                                       ////
////                                                                       ////
//// constrained_cxx //////////////////////////////////////////////////////////

//// reconstruct_cxx //////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

////                                                                       ////
////                                                                       ////
//// reconstruct_cxx //////////////////////////////////////////////////////////

//// refine_cxx ///////////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshGetSplitPoint"
// getsplitpoint()    Get the inserting point in a segment.                  //
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
    // Use the CDT rules to split the segment.
    acutea = (pointtype(m, e1) == ACUTEVERTEX) ? PETSC_TRUE : PETSC_FALSE;
    acuteb = (pointtype(m, e2) == ACUTEVERTEX) ? PETSC_TRUE : PETSC_FALSE;
    if (acutea ^ acuteb) {
      // Only one endpoint is acute. Use rule-2 or rule-3.
      ei = acutea ? e1 : e2;
      ej = acutea ? e2 : e1;
      L = distance(ei, ej);
      // Apply rule-2.
      d1 = distance(ei, refpt);
      split = d1 / L;
      for(i = 0; i < 3; i++) newpt[i] = ei[i] + split * (ej[i] - ei[i]);
      // Check if rule-3 is needed.
      d2 = distance(refpt, newpt);
      if (d2 > (L - d1)) {
        // Apply rule-3.
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
      // Both endpoints are acute or not. Split it at the middle.
      for(i = 0; i < 3; i++) newpt[i] = 0.5 * (e1[i] + e2[i]);
    }
  } else {
    // Split the segment at its midpoint.
    for(i = 0; i < 3; i++) newpt[i] = 0.5 * (e1[i] + e2[i]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshSetNewPointSize"
// setnewpointsize()    Set the size for a new point.                        //
//                                                                           //
// The size of the new point p is interpolated either from a background mesh //
// (b->bgmesh) or from the two input endpoints.                              //
/* tetgenmesh::setnewpointsize() */
PetscErrorCode TetGenMeshSetNewPointSize(TetGenMesh *m, point newpt, point e1, point e2)
{
  TetGenOpts *b  = m->b;

  PetscFunctionBegin;
  if (b->metric) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    // Interpolate the point size in a background mesh.
    triface bgmtet = {PETSC_NULL, 0, 0};
    // Get a tet in background mesh for locating p.
    decode(point2bgmtet(m, e1), &bgmtet);
    p1interpolatebgm(newpt, &bgmtet, PETSC_NULL);
#endif
  } else {
    if (e2) {
      // Interpolate the size between the two endpoints.
      PetscReal split, l, d;
      l = distance(e1, e2);
      d = distance(e1, newpt);
      split = d / l;
#ifdef PETSC_USE_DEBUG
      // Check if e1 and e2 are endpoints of a sharp segment.
      if (e1[m->pointmtrindex] <= 0.0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Null point");}
      if (e2[m->pointmtrindex] <= 0.0) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Null point");}
#endif
      newpt[m->pointmtrindex] = (1.0 - split) * e1[m->pointmtrindex] + split * e2[m->pointmtrindex];
    }
  }
  PetscFunctionReturn(0);
}

////                                                                       ////
////                                                                       ////
//// refine_cxx ///////////////////////////////////////////////////////////////

//// optimize_cxx /////////////////////////////////////////////////////////////
////                                                                       ////
////                                                                       ////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshCheckTet4Ill"
// checktet4ill()    Check a tet to see if it is illegal.                    //
//                                                                           //
// A tet is "illegal" if it spans on one input facet.  Save the tet in queue //
// if it is illegal and the flag 'enqflag' is set.                           //
//                                                                           //
// Note: Such case can happen when the input facet has non-coplanar vertices //
// and the Delaunay tetrahedralization of the vertices may creat such tets.  //
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
          // Two subfaces share this edge.
          sspivot(m, &checksh1, &checkseg);
          if (checkseg.sh == m->dummysh) {
            // The four corners of the tet are on one facet. Illegal! Try to
            //   flip the opposite edge of the current one.
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
    // Allocate space for the bad tetrahedron.
    ierr = MemoryPoolAlloc(m->badtetrahedrons, (void **) &newbadtet);CHKERRQ(ierr);
    newbadtet->tt = *testtet;
    newbadtet->key = -1.0; // = 180 degree.
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
// checktet4opt()    Check a tet to see if it needs to be optimized.         //
//                                                                           //
// A tet t needs to be optimized if it fails to certain quality measures.    //
// The only quality measure currently used is the maximal dihedral angle at  //
// edges. The desired maximal dihedral angle is 'b->maxdihedal' (set by the  //
// '-qqq' option.                                                            //
//                                                                           //
// A tet may have one, two, or three big dihedral angles. Examples: Let the  //
// tet t = abcd, and its four corners are nearly co-planar. Then t has one   //
// big dihedral angle if d is very close to the edge ab; t has three big     //
// dihedral angles if d's projection on the face abc is also inside abc, i.e.//
// the shape of t likes a hat; finally, t has two big dihedral angles if d's //
// projection onto abc is outside abc.                                       //
/* tetgenmesh::checktet4opt() */
PetscErrorCode TetGenMeshCheckTet4Opt(TetGenMesh *m, triface* testtet, PetscBool enqflag, PetscBool *doOpt)
{
  TetGenOpts    *b = m->b;
  badface *newbadtet;
  point pa, pb, pc, pd;
  PetscReal N[4][3], len;
  PetscReal cosd;
  int count;
  int i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pa = (point) testtet->tet[4];
  pb = (point) testtet->tet[5];
  pc = (point) testtet->tet[6];
  pd = (point) testtet->tet[7];
  // Compute the 4 face normals: N[0] cbd, N[1] acd, N[2] bad, N[3] abc.
  ierr = TetGenMeshTetAllNormal(m, pa, pb, pc, pd, N, PETSC_NULL);CHKERRQ(ierr);
  // Normalize the normals.
  for(i = 0; i < 4; i++) {
    len = sqrt(dot(N[i], N[i]));
    if (len != 0.0) {
      for(j = 0; j < 3; j++) N[i][j] /= len;
    }
  }
  count = 0;

  // Find all large dihedral angles.
  for(i = 0; i < 6; i++) {
    // Locate the edge i and calculate the dihedral angle at the edge.
    testtet->loc = 0;
    testtet->ver = 0;
    switch (i) {
    case 0: // edge ab
      cosd = -dot(N[2], N[3]);
      break;
    case 1: // edge cd
      enextfnextself(m, testtet);
      enextself(testtet);
      cosd = -dot(N[0], N[1]);
      break;
    case 2: // edge bd
      enextfnextself(m, testtet);
      enext2self(testtet);
      cosd = -dot(N[0], N[2]);
      break;
    case 3: // edge bc
      enextself(testtet);
      cosd = -dot(N[0], N[3]);
      break;
    case 4: // edge ad
      enext2fnextself(m, testtet);
      enextself(testtet);
      cosd = -dot(N[1], N[2]);
      break;
    case 5: // edge ac
      enext2self(testtet);
      cosd = -dot(N[1], N[3]);
      break;
    }
    if (cosd < m->cosmaxdihed) {
      // A bigger dihedral angle.
      count++;
      if (enqflag) {
        // Allocate space for the bad tetrahedron.
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
// removeedge()    Remove an edge                                            //
//                                                                           //
// 'remedge' is a tet (abcd) having the edge ab wanted to be removed.  Local //
// reconnecting operations are used to remove edge ab.  The following opera- //
// tion will be tryed.                                                       //
//                                                                           //
// If ab is on the hull, and abc and abd are both hull faces. Then ab can be //
// removed by stripping abcd from the mesh. However, if ab is a segemnt, do  //
// the operation only if 'b->optlevel' > 1 and 'b->nobisect == 0'.           //
//                                                                           //
// If ab is an internal edge, there are n tets contains it.  Then ab can be  //
// removed if there exists another m tets which can replace the n tets with- //
// out changing the boundary of the n tets.                                  //
//                                                                           //
// If 'optflag' is set.  The value 'remedge->key' means cos(theta), where    //
// 'theta' is the maximal dishedral angle at ab. In this case, even if the   //
// n-to-m flip exists, it will not be performed if the maximum dihedral of   //
// the new tets is larger than 'theta'.                                      //
/* tetgenmesh::removeedge() */
PetscErrorCode TetGenMeshRemoveEdge(TetGenMesh *m, badface* remedge, PetscBool optflag, PetscBool *isRemoved)
{
  TetGenOpts    *b = m->b;
  triface abcd = {PETSC_NULL, 0, 0}, badc = {PETSC_NULL, 0, 0};  // Tet configuration at edge ab.
  triface baccasing = {PETSC_NULL, 0, 0}, abdcasing = {PETSC_NULL, 0, 0};
  triface abtetlist[21];  // Old configuration at ab, save maximum 20 tets.
  triface bftetlist[21];  // Old configuration at bf, save maximum 20 tets.
  triface newtetlist[90]; // New configuration after removing ab.
  face checksh = {PETSC_NULL, 0};
  PetscReal key;
  PetscBool remflag, subflag;
  int n, n1, m1, i, j, k;
  triface newtet = {PETSC_NULL, 0, 0};
  point *ppt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // First try to strip abcd from the mesh. This needs to check either ab
  //   or cd is on the hull. Try to strip it whichever is true.
  abcd = remedge->tt;
  adjustedgering_triface(&abcd, CCW);
  k = 0;
  do {
    sym(&abcd, &baccasing);
    // Is the tet on the hull?
    if (baccasing.tet == m->dummytet) {
      fnext(m, &abcd, &badc);
      sym(&badc, &abdcasing);
      if (abdcasing.tet == m->dummytet) {
        // Strip the tet from the mesh -> ab is removed as well.
#if 1
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put code in");
#else
        if (removetetbypeeloff(&abcd, newtetlist)) {
          PetscInfo(b->in, "    Stripped tet from the mesh.\n");
          m->optcount[0]++;
          m->opt_tet_peels++;
          // edge is removed. Test new tets for further optimization.
          for(i = 0; i < 2; i++) {
            if (optflag) {
              ierr = TetGenMeshCheckTet4Opt(m, &(newtetlist[i]), PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
            } else {
              ierr = TetGenMeshCheckTet4Ill(m, &(newtetlist[i]), PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
            }
          }
          // Update the point-to-tet map
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
    // Check if the oppsite edge cd is on the hull.
    enext2fnextself(m, &abcd);
    enext2self(&abcd);
    esymself(&abcd); // --> cdab
    k++;
  } while (k < 2);

  // Get the tets configuration at ab. Collect maximum 10 tets.
  subflag = PETSC_FALSE;
  abcd = remedge->tt;
  adjustedgering_triface(&abcd, CW);
  n = 0;
  abtetlist[n] = abcd;
  do {
    // Is the list full?
    if (n == 20) break;
    // Stop if a subface appears.
    tspivot(m, &abtetlist[n], &checksh);
    if (checksh.sh != m->dummysh) {
      // ab is either a segment or a facet edge. The latter case is not
      //   handled yet! An edge flip is needed.
      subflag = PETSC_TRUE; break; // return false;
    }
    // Get the next tet at ab.
    fnext(m, &abtetlist[n], &abtetlist[n + 1]);
    n++;
  } while (apex(&abtetlist[n]) != apex(&abcd));

  remflag = PETSC_FALSE;
  key = remedge->key;

  if (subflag && optflag) {
    // Faces are not flipable. Return.
    if (isRemoved) {*isRemoved = PETSC_FALSE;}
    PetscFunctionReturn(0);
  }

  // 2 < n < 20.
  if (n == 3) {
    // There are three tets at ab. Try to do a flip32 at ab.
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put code in");
#else
    remflag = removeedgebyflip32(&key, abtetlist, newtetlist, PETSC_NULL);
#endif
  } else if ((n > 3) && (n <= b->maxflipedgelinksize)) {
    // Four tets case. Try to do edge transformation.
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put code in");
#else
    remflag = removeedgebytranNM(&key,n,abtetlist,newtetlist,PETSC_NULL,PETSC_NULL,PETSC_NULL);
#endif
  } else {
    PetscInfo1(b->in, "  !! Unhandled case: n = %d.\n", n);
  }
  if (remflag) {
    m->optcount[n]++;
    // Delete the old tets.
    for(i = 0; i < n; i++) {
      ierr = TetGenMeshTetrahedronDealloc(m, abtetlist[i].tet);CHKERRQ(ierr);
    }
    m1 = (n - 2) * 2; // The number of new tets.
    if (b->verbose > 1) {
      if (optflag) {
        PetscInfo4(b->in, "  Done flip %d-to-%d Qual: %g -> %g.", n, m1, acos(remedge->key) / PETSC_PI * 180.0, acos(key) / PETSC_PI * 180.0);
      } else {
        PetscInfo2(b->in, "  Done flip %d-to-%d.\n", n, m1);
      }
    }
  }

  if (!remflag && (key == remedge->key) && (n <= b->maxflipedgelinksize)) {
    // Try to do a combination of flips.
    n1 = 0;
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put code in");
#else
    remflag = removeedgebycombNM(&key, n, abtetlist, &n1, bftetlist, newtetlist, PETSC_NULL);
#endif
    if (remflag) {
      m->optcount[9]++;
      // Delete the old tets.
      for(i = 0; i < n; i++) {
        ierr = TetGenMeshTetrahedronDealloc(m, abtetlist[i].tet);CHKERRQ(ierr);
      }
      for(i = 0; i < n1; i++) {
        if (!isdead_triface(&(bftetlist[i]))) {
          ierr = TetGenMeshTetrahedronDealloc(m, bftetlist[i].tet);CHKERRQ(ierr);
        }
      }
      m1 = ((n1 - 2) * 2 - 1) + (n - 3) * 2; // The number of new tets.
      if (optflag) {
        PetscInfo6(b->in, "  Done flip %d-to-%d (n-1=%d, n1=%d) Qual: %g -> %g.\n", n+n1-2, m1, n-1, n1, acos(remedge->key) / PETSC_PI * 180.0, acos(key) / PETSC_PI * 180.0);
      } else {
        PetscInfo4(b->in, "  Done flip %d-to-%d (n-1=%d, n1=%d).\n", n+n1-2, m1, n-1, n1);
      }
    }
  }

  if (remflag) {
    // edge is removed. Test new tets for further optimization.
    for(i = 0; i < m1; i++) {
      if (optflag) {
        ierr = TetGenMeshCheckTet4Opt(m, &(newtetlist[i]), PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      } else {
        ierr = TetGenMeshCheckTet4Ill(m, &(newtetlist[i]), PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      }
    }
    // Update the point-to-tet map
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
  PetscReal ori, aspT, aspTmax, imprate;
  PetscReal cosd, maxcosd;
  PetscBool segflag, randflag;
  int numdirs;
  int len, iter, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Is p a segment vertex?
  segflag = e1 ? PETSC_TRUE : PETSC_FALSE;
  // Decide the number of moving directions.
  ierr = ListLength(starlist, &len);CHKERRQ(ierr);
  numdirs = segflag ? 2 : len;
  randflag = numdirs > 10 ? PETSC_TRUE : PETSC_FALSE;
  if (randflag) {
    numdirs = 10; // Maximum 10 directions.
  }

  // Calculate the initial object value (the largest aspect ratio).
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

  // Do iteration until the new aspTmax does not decrease.
  newTmax = iniTmax;
  iter = 0;
  while(1) {
    // Find the best next location.
    oldTmax = newTmax;
    for(i = 0; i < numdirs; i++) {
      // Calculate the moved point (saved in 'nextpt').
      if (!segflag) {
        if (randflag) {
          // Randomly pick a direction.
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
      // Get the largest object value for the new location.
      for(j = 0; j < len; j++) {
        ierr = ListItem(starlist, j, (void **) &starttet);CHKERRQ(ierr);
        adjustedgering_triface(&starttet, !invtori ? CCW : CW);
        pa = org(&starttet);
        pb = dest(&starttet);
        pc = apex(&starttet);
        ori = orient3d(pa, pb, pc, nextpt);
        if (ori < 0.0) {
          ierr = TetGenMeshTetAspectRatio(m, pa, pb, pc, nextpt, &aspT);CHKERRQ(ierr);
          if (j == 0) {
            aspTmax = aspT;
          } else {
            aspTmax = aspT > aspTmax ? aspT : aspTmax;
          }
        } else {
          // An invalid new tet. Discard this point.
          aspTmax = newTmax;
        } // if (ori < 0.0)
        // Stop looping when the object value is bigger than before.
        if (aspTmax >= newTmax) break;
      } // for (j = 0; j < starlist->len(); j++)
      if (aspTmax < newTmax) {
        // Save the improved object value and the location.
        newTmax = aspTmax;
        for(j = 0; j < 3; j++) bestpt[j] = nextpt[j];
      }
    } // for (i = 0; i < starlist->len(); i++)
    // Does the object value improved much?
    imprate = fabs(oldTmax - newTmax) / oldTmax;
    if (imprate < 1e-3) break;
    // Yes, move p to the new location and continue.
    for (j = 0; j < 3; j++) startpt[j] = bestpt[j];
    iter++;
  } // while (true)

  if (iter > 0) {
    // The point is moved.
    if (key) {
      // Check if the quality is improved by the smoothed point.
      maxcosd = 0.0; // = cos(90).
      for(j = 0; j < len; j++) {
        ierr = ListItem(starlist, j, (void **) &starttet);CHKERRQ(ierr);
        adjustedgering_triface(&starttet, !invtori ? CCW : CW);
        pa = org(&starttet);
        pb = dest(&starttet);
        pc = apex(&starttet);
        ierr = TetGenMeshTetAllDihedral(m, pa, pb, pc, startpt, PETSC_NULL, &cosd, PETSC_NULL);CHKERRQ(ierr);
        if (cosd < *key) {
          // This quality will not be improved. Stop.
          iter = 0; break;
        } else {
          // Remeber the worst quality value (of the new configuration).
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
// smoothsliver()    Remove a sliver by smoothing a vertex of it.            //
//                                                                           //
// The 'slivtet' represents a sliver abcd, and ab is the current edge which  //
// has a large dihedral angle (close to 180 degree).                         //
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
  // Find a Steiner volume point and smooth it.
  smthed = PETSC_FALSE;
  for(i = 0; i < 4 && !smthed; i++) {
    smthpt = (point) remedge->tt.tet[4 + i];
    // Is it a volume point?
    if (pointtype(m, smthpt) == FREEVOLVERTEX) {
      // Is it a Steiner point?
      idx = pointmark(m, smthpt) - in->firstnumber;
      if (!(idx < in->numberofpoints)) {
        // Smooth a Steiner volume point.
        ierr = ListAppend(starlist, &(remedge->tt.tet), PETSC_NULL);CHKERRQ(ierr);
        ierr = TetGenMeshFormStarPolyhedron(m, smthpt, starlist, PETSC_NULL, PETSC_FALSE);CHKERRQ(ierr);
        ierr = TetGenMeshSmoothPoint(m, smthpt,PETSC_NULL,PETSC_NULL,starlist,PETSC_FALSE,&remedge->key, &smthed);CHKERRQ(ierr);
        // If it is smoothed. Queue new bad tets.
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
// splitsliver()    Remove a sliver by inserting a point.                    //
//                                                                           //
// The 'remedge->tt' represents a sliver abcd, ab is the current edge which  //
// has a large dihedral angle (close to 180 degree).                         //
/* tetgenmesh::splitsliver() */
PetscErrorCode TetGenMeshSplitSliver(TetGenMesh *m, badface *remedge, List *tetlist, List *ceillist, PetscBool *isSplit)
{
  TetGenOpts    *b = m->b;
  triface starttet = {PETSC_NULL, 0, 0};
  face checkseg = {PETSC_NULL, 0};
  point newpt, pt[4];
  PetscBool remflag;
  int i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Let 'remedge->tt' be the edge [a, b].
  starttet = remedge->tt;

  // Go to the opposite edge [c, d].
  adjustedgering_triface(&starttet, CCW);
  enextfnextself(m, &starttet);
  enextself(&starttet);

  // Check if cd is a segment.
  ierr = TetGenMeshTssPivot(m, &starttet, &checkseg);CHKERRQ(ierr);
  if (b->nobisect == 0) {
    if (checkseg.sh != m->dummysh) {
      int len;

      // cd is a segment. The seg will be split.
      checkseg.shver = 0;
      pt[0] = sorg(&checkseg);
      pt[1] = sdest(&checkseg);
      ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
      ierr = TetGenMeshGetSplitPoint(m, pt[0], pt[1], PETSC_NULL, newpt);CHKERRQ(ierr);
      setpointtype(m, newpt, FREESEGVERTEX);
      setpoint2seg(m, newpt, sencode(&checkseg));
      // Insert p, this should always success.
      ierr = TetGenMeshSstPivot(m, &checkseg, &starttet);CHKERRQ(ierr);
      ierr = TetGenMeshSplitTetEdge(m, newpt, &starttet, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
      // Collect the new tets connecting at p.
      ierr = TetGenMeshSstPivot(m, &checkseg, &starttet);CHKERRQ(ierr);
      ierr = ListAppend(ceillist, &starttet, PETSC_NULL);CHKERRQ(ierr);
      ierr = TetGenMeshFormStarPolyhedron(m, newpt, ceillist, PETSC_NULL, PETSC_TRUE);CHKERRQ(ierr);
      ierr = TetGenMeshSetNewPointSize(m, newpt, pt[0], PETSC_NULL);CHKERRQ(ierr);
      if (m->steinerleft > 0) m->steinerleft--;
      // Smooth p.
      ierr = TetGenMeshSmoothPoint(m, newpt, pt[0], pt[1], ceillist, PETSC_FALSE, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
      // Queue new slivers.
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

  // Create the new point p (at the circumcenter of t).
  ierr = TetGenMeshMakePoint(m, &newpt);CHKERRQ(ierr);
  pt[0] = org(&starttet);
  pt[1] = dest(&starttet);
  for (i = 0; i < 3; i++) {
    newpt[i] = 0.5 * (pt[0][i] + pt[1][i]);
  }
  setpointtype(m, newpt, FREEVOLVERTEX);

  // Form the Bowyer-Watson cavity of p.
#if 1
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put code in");
#else
  remflag = PETSC_FALSE;
  infect(m, &starttet);
  ierr = ListAppend(tetlist, &starttet, PETSC_NULL);CHKERRQ(ierr);
  formbowatcavityquad(newpt, tetlist, ceillist);
  if (trimbowatcavity(newpt, PETSC_NULL, 1, PETSC_NULL, PETSC_NULL, &tetlist, &ceillist, -1.0)) {
    PetscBool isSmooth;
    // Smooth p.
    ierr = TetGenMeshSmoothPoint(m, newpt, PETSC_NULL, PETSC_NULL, ceillist, PETSC_FALSE, &remedge->key, &isSmooth);CHKERRQ(ierr);
    if (isSmooth) {
      int len;
      // Insert p.
      bowatinsertsite(newpt, PETSC_NULL, 1, PETSC_NULL, PETSC_NULL, &tetlist, &ceillist, PETSC_NULL,
                      PETSC_NULL, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE);
      ierr = TetGenMeshSetNewPointSize(m, newpt, pt[0], PETSC_NULL);CHKERRQ(ierr);
      if (m->steinerleft > 0) m->steinerleft--;
      // Queue new slivers.
      ierr = ListLength(ceillist, &len);CHKERRQ(ierr);
      for(i = 0; i < len; i++) {
        ierr = ListItem(ceillist, i, (void **) &starttet);CHKERRQ(ierr);
        ierr = TetGenMeshCheckTet4Opt(m, &starttet, PETSC_TRUE, PETSC_NULL);CHKERRQ(ierr);
      }
      remflag = PETSC_TRUE;
    } // if (smoothpoint)
  } // if (trimbowatcavity)
#endif

  if (!remflag) {
    int len;
    // p is rejected for BC(p) is not valid.
    ierr = TetGenMeshPointDealloc(m, newpt);CHKERRQ(ierr);
    // Uninfect tets of BC(p).
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
// tallslivers()    Queue all the slivers in the mesh.                       //
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
// Available mesh optimizing operations are: (1) multiple edge flips (3-to-2,//
// 4-to-4, 5-to-6, etc), (2) free vertex deletion, (3) new vertex insertion. //
// (1) is mandatory, while (2) and (3) are optionally.                       //
//                                                                           //
// The variable 'b->optlevel' (set after '-s') determines the use of these   //
// operations. If it is: 0, do no optimization; 1, only do (1) operation; 2, //
// do (1) and (2) operations; 3, do all operations. Deault, b->optlvel = 2.  //
/* tetgenmesh::optimizemesh2() */
PetscErrorCode TetGenMeshOptimize(TetGenMesh *m, PetscBool optflag)
{
  TetGenOpts    *b  = m->b;
  // Cosines of the six dihedral angles of the tet [a, b, c, d].
  //   From cosdd[0] to cosdd[5]: ab, bc, ca, ad, bd, cd.
  PetscReal      cosdd[6];
  List          *splittetlist, *tetlist, *ceillist;
  badface       *remtet, *newbadtet;
  PetscReal      maxdihed, objdihed, cosobjdihed;
  long           oldflipcount = 0, newflipcount = 0, oldpointcount, slivercount, optpasscount = 0;
  int            iter, len, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (optflag) {
    if (m->b_steinerflag) {
      // This routine is called from removesteiners2();
    } else {
      PetscInfo(b->in, "Optimizing mesh.\n");
    }
  } else {
    PetscInfo(b->in, "Repairing mesh.\n");
  }

  if (optflag) {
    if (m->b_steinerflag) {
      // This routine is called from removesteiners2();
      m->cosmaxdihed = cos(179.0 * PETSC_PI / 180.0);
      m->cosmindihed = cos(1.0 * PETSC_PI / 180.0);
      // The radian of the maximum dihedral angle.
      maxdihed = 179.0 / 180.0 * PETSC_PI;
    } else {
      m->cosmaxdihed = cos(b->maxdihedral * PETSC_PI / 180.0);
      m->cosmindihed = cos(b->mindihedral * PETSC_PI / 180.0);
      // The radian of the maximum dihedral angle.
      maxdihed = b->maxdihedral / 180.0 * PETSC_PI;
      // A sliver has an angle large than 'objdihed' will be split.
      objdihed = b->maxdihedral + 5.0;
      if (objdihed < 175.0) objdihed = 175.0;
      objdihed = objdihed / 180.0 * PETSC_PI;
      cosobjdihed = cos(objdihed);
    }
  }

  // Initialize the pool of bad tets.
  ierr = MemoryPoolCreate(sizeof(badface), ELEPERBLOCK, POINTER, 0, &m->badtetrahedrons);CHKERRQ(ierr);
  // Looking for non-optimal tets.
  ierr = TetGenMeshTallSlivers(m, optflag);CHKERRQ(ierr);

  oldpointcount = m->points->items;
  m->opt_tet_peels = m->opt_face_flips = m->opt_edge_flips = 0l;
  m->smoothsegverts = 0l;

  if (optflag) {PetscInfo1(b->in, "  level = %d.\n", b->optlevel);}

  // Start the mesh optimization iteration.
  do {
    PetscInfo2(b->in, "  level = %d pass %d.\n", b->optlevel, optpasscount);

    // Improve the mesh quality by flips.
    iter = 0;
    do {
      oldflipcount = newflipcount;
      // Loop in the list of bad tets.
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
            // Remove the badtet from the list.
            ierr = TetGenMeshBadFaceDealloc(m, m->badtetrahedrons, remtet);CHKERRQ(ierr);
          }
        } else {
          // Remove the badtet from the list.
          ierr = TetGenMeshBadFaceDealloc(m, m->badtetrahedrons, remtet);CHKERRQ(ierr);
        }
        ierr = TetGenMeshBadFaceTraverse(m, m->badtetrahedrons, &remtet);CHKERRQ(ierr);
      }
      iter++;
      if (iter > 10) break; // Stop at 10th iterations.
      // Count the total number of flips.
      newflipcount = m->opt_tet_peels + m->opt_face_flips + m->opt_edge_flips;
      // Continue if there are bad tets and new flips.
    } while ((m->badtetrahedrons->items > 0) && (newflipcount > oldflipcount));

    if (m->b_steinerflag) {
      // This routine was called from removesteiner2(). Do not repair the bad tets by splitting.
      ierr = MemoryPoolRestart(m->badtetrahedrons);CHKERRQ(ierr);
    }

    if ((m->badtetrahedrons->items > 0l) && optflag  && (b->optlevel > 2)) {
      // Get a list of slivers and try to split them.
      ierr = ListCreate(sizeof(badface), NULL, 256, PETSC_DECIDE, &splittetlist);CHKERRQ(ierr);
      ierr = ListCreate(sizeof(triface), NULL, 256, PETSC_DECIDE, &tetlist);CHKERRQ(ierr);
      ierr = ListCreate(sizeof(triface), NULL, 256, PETSC_DECIDE, &ceillist);CHKERRQ(ierr);

      // Form a list of slivers to be split and clean the pool.
      ierr = MemoryPoolTraversalInit(m->badtetrahedrons);CHKERRQ(ierr);
      ierr = TetGenMeshBadFaceTraverse(m, m->badtetrahedrons, &remtet);CHKERRQ(ierr);
      while(remtet) {
        ierr = ListAppend(splittetlist, remtet, PETSC_NULL);CHKERRQ(ierr);
        ierr = TetGenMeshBadFaceTraverse(m, m->badtetrahedrons, &remtet);CHKERRQ(ierr);
      }
      // Clean the pool of bad tets.
      ierr = MemoryPoolRestart(m->badtetrahedrons);CHKERRQ(ierr);
      ierr = ListLength(splittetlist, &len);CHKERRQ(ierr);
      slivercount = 0;
      for(i = 0; i < len; i++) {
        ierr = ListItem(splittetlist, i, (void **) &remtet);CHKERRQ(ierr);
        if (!isdead_triface(&remtet->tt) && org(&remtet->tt) == remtet->forg &&
            dest(&remtet->tt) == remtet->fdest &&
            apex(&remtet->tt) == remtet->fapex &&
            oppo(&remtet->tt) == remtet->foppo) {
          // Calculate the six dihedral angles of this tet.
          adjustedgering_triface(&remtet->tt, CCW);
          remtet->forg  = org(&remtet->tt);
          remtet->fdest = dest(&remtet->tt);
          remtet->fapex = apex(&remtet->tt);
          remtet->foppo = oppo(&remtet->tt);
          ierr = TetGenMeshTetAllDihedral(m, remtet->forg, remtet->fdest, remtet->fapex, remtet->foppo, cosdd, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
          // Is it a large angle?
          if (cosdd[0] < cosobjdihed) {
            PetscBool couldSmooth;

            slivercount++;
            remtet->key = cosdd[0];
            PetscInfo5(b->in, "    Split tet (%d, %d, %d, %d) %g (degree).\n", pointmark(m, remtet->forg), pointmark(m, remtet->fdest),
                       pointmark(m, remtet->fapex), pointmark(m, remtet->foppo), acos(remtet->key) / PETSC_PI * 180.0);
            // Queue this tet.
            ierr = MemoryPoolAlloc(m->badtetrahedrons, (void **) &newbadtet);CHKERRQ(ierr);
            *newbadtet = *remtet;
            // Try to remove this tet.
            ierr = TetGenMeshSmoothSliver(m, remtet, tetlist, &couldSmooth);CHKERRQ(ierr);
            if (!couldSmooth) {
              ierr = TetGenMeshSplitSliver(m, remtet, tetlist, ceillist, PETSC_NULL);CHKERRQ(ierr);
            }
          }
        }
      } // i

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

////                                                                       ////
////                                                                       ////
//// optimize_cxx /////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshJettisonNodes"
// Unused points are those input points which are outside the mesh domain or //
// have no connection (isolated) to the mesh.  Duplicated points exist for   //
// example if the input PLC is read from a .stl mesh file (marked during the //
// Delaunay tetrahedralization step. This routine remove these points from   //
// points list. All existing points are reindexed.                           //
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
      // It is a duplicated point, delete it.
      ierr = TetGenMeshPointDealloc(m, pointloop);CHKERRQ(ierr);
      remcount++;
    } else {
      // Re-index it.
      setpointmark(m, pointloop, newidx + in->firstnumber);
      if (in->pointmarkerlist) {
        if (oldidx < in->numberofpoints) {
          // Re-index the point marker as well.
          in->pointmarkerlist[newidx] = in->pointmarkerlist[oldidx];
        }
      }
      newidx++;
    }
    oldidx++;
    if (oldidx == in->numberofpoints) {
      // Update the numbe of input points (Because some were removed).
      in->numberofpoints -= remcount;
      // Remember this number for output original input nodes.
      m->jettisoninverts = remcount;
    }
    ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
  }
  PetscInfo1(b->in, "  %d duplicated vertices have been removed.\n", m->dupverts);
  PetscInfo1(b->in, "  %d unused vertices have been removed.\n", m->unuverts);
  m->dupverts = 0;
  m->unuverts = 0;
  // The following line ensures that dead items in the pool of nodes cannot
  //   be allocated for the new created nodes. This ensures that the input
  //   nodes will occur earlier in the output files, and have lower indices.
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
    // Initialize the point2tet field of each point.
    ierr = MemoryPoolTraversalInit(m->points);CHKERRQ(ierr);
    ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
    while(pointloop) {
      setpoint2tet(m, pointloop, (tetrahedron) PETSC_NULL);
      ierr = TetGenMeshPointTraverse(m, &pointloop);CHKERRQ(ierr);
    }
    // Make a map point-to-subface. Hence a boundary point will get the facet marker from that facet where it lies on.
    ierr = MemoryPoolTraversalInit(m->subfaces);CHKERRQ(ierr);
    ierr = TetGenMeshShellFaceTraverse(m, m->subfaces, &subloop.sh);CHKERRQ(ierr);
    while(subloop.sh) {
      subloop.shver = 0;
      // Check all three points of the subface.
      for(i = 0; i < 3; i++) {
        pointloop = (point) subloop.sh[3 + i];
        setpoint2tet(m, pointloop, (tetrahedron) sencode(&subloop));
      }
      if (b->order == 2) {
        // '-o2' switch. Set markers for quadratic nodes of this subface.
        stpivot(m, &subloop, &adjtet);
        if (adjtet.tet == m->dummytet) {
          sesymself(&subloop);
          stpivot(m, &subloop, &adjtet);
        }
        if (adjtet.tet == m->dummytet) {SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid adjacency");}
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
      // Default the vertex has a zero marker.
      marker = 0;
      // Is it an input vertex?
      if (index < in->numberofpoints) {
        // Input point's marker is directly copied to output.
        marker = in->pointmarkerlist[index];
      }
      // Is it a boundary vertex has marker zero?
      if ((marker == 0) && (b->plc || b->refine)) {
        subptr = (shellface) point2tet(m, pointloop);
        if (subptr) {
          // Default a boundary vertex has marker 1.
          marker = 1;
          if (in->facetmarkerlist) {
            // The vertex gets the marker from the facet it lies on.
            sdecode(subptr, &subloop);
            shmark = shellmark(m, &subloop);
            marker = in->facetmarkerlist[shmark - 1];
          }
        }
      }
    }
    // x, y, and z coordinates.
    out->pointlist[coordindex++] = pointloop[0];
    out->pointlist[coordindex++] = pointloop[1];
    out->pointlist[coordindex++] = pointloop[2];
    // Point attributes.
    for(i = 0; i < nextras; i++) {
      // Output an attribute.
      out->pointattributelist[attribindex++] = pointloop[3 + i];
    }
    if (bmark) {
      // Output the boundary marker.
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
    // Using the Euler formula (V-E+F-T=1) to get the total number of edges.
    long faces = (4l * m->tetrahedrons->items + m->hullsize) / 2l;
    m->meshedges = m->points->items + faces - m->tetrahedrons->items - 1l;
    PetscFunctionReturn(0);
  }

  m->meshedges = 0l;
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tetloop.tet);CHKERRQ(ierr);
  while(tetloop.tet) {
    // Count the number of Voronoi faces. Look at the six edges of each
    //   tetrahedron. Count the edge only if the tetrahedron's pointer is
    //   smaller than those of all other tetrahedra that share the edge.
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
            fnextself(m, &spintet); // In the same tet.
	  }
        }
      }
      // Count this edge if no adjacent tets are smaller than this tet.
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
  // Allocate memory for output tetrahedron attributes if necessary.
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
  // Determine the first index (0 or 1).
  firstindex = b->zeroindex ? 0 : in->firstnumber;
  shift      = 0; // Default no shiftment.
  if ((in->firstnumber == 1) && (firstindex == 0)) {
    shift = 1; // Shift the output indices by 1.
  }
  // Count the total edge numbers.
  m->meshedges = 0l;
  ierr = MemoryPoolTraversalInit(m->tetrahedrons);CHKERRQ(ierr);
  ierr = TetGenMeshTetrahedronTraverse(m, &tptr);CHKERRQ(ierr);
  elementnumber = firstindex; // in->firstnumber;
  while(tptr) {
    if (b->noelewritten == 2) {
      // Reverse the orientation, such that Orient3D() > 0.
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
      // Remember the index of this element.
      setelemmarker(m, tptr, elementnumber);
    }
    // Count the number of Voronoi faces. Look at the six edges of each
    //   tetrahedron. Count the edge only if the tetrahedron's pointer is
    //   smaller than those of all other tetrahedra that share the edge.
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
            fnextself(m, &spintet); // In the same tet.
	  }
        }
      }
      // Count this edge if no adjacent tets are smaller than this tet.
      if (spintet.tet >= worktet.tet) {
        m->meshedges++;
      }
    }
    ierr = TetGenMeshTetrahedronTraverse(m, &tptr);CHKERRQ(ierr);
    elementnumber++;
  }
  if (b->neighout) {
    // Set the outside element marker.
    setelemmarker(m, m->dummytet, -1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TetGenMeshOutputSubfaces"
// The boundary faces are exist in 'subfaces'. For listing triangle vertices //
// in the same sense for all triangles in the mesh, the direction determined //
// by right-hand rule is pointer to the inside of the volume.                //
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
  // Allocate memory for 'trifacelist'.
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
  // Determine the first index (0 or 1).
  firstindex = b->zeroindex ? 0 : in->firstnumber;
  shift = 0; // Default no shiftment.
  if ((in->firstnumber == 1) && (firstindex == 0)) {
    shift = 1; // Shift the output indices by 1.
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
      // If there is a tetrahedron containing this subface, orient it so
      //   that the normal of this face points to inside of the volume by
      //   right-hand rule.
      adjustedgering_triface(&abuttingtet, CCW);
      torg  = org(&abuttingtet);
      tdest = dest(&abuttingtet);
      tapex = apex(&abuttingtet);
    } else {
      // This may happen when only a surface mesh be generated.
      torg  = sorg(&faceloop);
      tdest = sdest(&faceloop);
      tapex = sapex(&faceloop);
    }
    if (bmark) {
      faceid = shellmark(m, &faceloop) - 1;
      marker = in->facetmarkerlist[faceid];
    }
    if (b->neighout > 1) {
      // '-nn' switch. Output adjacent tets indices.
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
    // Output three vertices of this face;
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
  m->macheps     = exactinit();
  m->steinerleft = b->steiner;
  if (b->metric) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m->bgm = new tetgenmesh();
    m->bgm->b = b;
    m->bgm->in = bgmin;
    m->bgm->macheps = exactinit();
#endif
  }
  ierr = TetGenMeshInitializePools(m);CHKERRQ(ierr);
  ierr = TetGenMeshTransferNodes(m);CHKERRQ(ierr);

  // PetscLogEventBegin(DelaunayOrReconstruct)
  if (b->refine) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m.reconstructmesh();
#endif
  } else {
    ierr = TetGenMeshDelaunizeVertices(m);CHKERRQ(ierr);
    if (!m->hullsize) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The input point set does not span a 3D subspace.");
    }
  }
  // PetscLogEventEnd(DelaunayOrReconstruct)

  // PetscLogEventBegin(BackgroundMeshReconstruct)
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
  // PetscLogEventEnd(BackgroundMeshReconstruct)

  // PetscLogEventBegin(BdRecoveryOrIntersection)
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
  // PetscLogEventEnd(BdRecoveryOrIntersection)

  // PetscLogEventBegin(Holes)
  if (b->plc && !(b->diagnose == 1)) {
    /* THIS IS BROKEN BECAUSE SOMEHOW TETS ARE NOT ASSOCIATED WITH BOUNDARY SEGMENTS ierr = TetGenMeshCarveHoles(m);CHKERRQ(ierr); */
  }
  // PetscLogEventEnd(Holes)

  // PetscLogEventBegin(Repair)
  if ((b->plc || b->refine) && !(b->diagnose == 1)) {
    ierr = TetGenMeshOptimize(m, PETSC_FALSE);CHKERRQ(ierr);
  }
  // PetscLogEventEnd(Repair)

  // PetscLogEventBegin(SteinerRemoval)
  if ((b->plc && b->nobisect) && !(b->diagnose == 1)) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m.removesteiners2();
#endif
  }
  // PetscLogEventEnd(SteinerRemoval)

  // PetscLogEventBegin(ConstrainedPoints)
  if (b->insertaddpoints) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    if (addin && addin->numberofpoints > 0) {
      m.insertconstrainedpoints(addin);
    }
#endif
  }
  // PetscLogEventEnd(ConstrainedPoints)

  // PetscLogEventBegin(SizeInterpolation)
  if (b->metric) {
#if 1
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put in code");
#else
    m.interpolatesizemap();
#endif
  }
  // PetscLogEventEnd(SizeInterpolation)

#if 0 /* Removed by TetGen */
  // PetscLogEventBegin(MeshCoarsen)
  if (b->coarse) {
    m.removesteiners2(PETSC_TRUE);
  }
  // PetscLogEventEnd(MeshCoarsen)
#endif

  // PetscLogEventBegin(Quality)
  if (b->quality) {
#if 1
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Put code in");
#else
    m.enforcequality();
#endif
  }
  // PetscLogEventEnd(Quality)

  // PetscLogEventBegin(Optimize)
  if (b->quality && (b->optlevel > 0)) {
    ierr = TetGenMeshOptimize(m, PETSC_TRUE);CHKERRQ(ierr);
  }
  // PetscLogEventEnd(Optimize)

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
        // Only output when self-intersecting faces exist.
        ierr = TetGenMeshOutputNodes(m, out);CHKERRQ(ierr);
      }
    } else {
      ierr = TetGenMeshOutputNodes(m, out);CHKERRQ(ierr);
      if (b->metric) { //if (b->quality && b->metric) {
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
        m.outfaces(out);  // Output all faces.
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
          m.outhullfaces(out); // Output convex hull faces.
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
      m.outedges(out); // -ee, output all mesh edges
    } else {
      m.outsubsegments(out); // -e, only output subsegments.
    }
  }

  if (!out && b->plc &&
      ((b->object == tetgenbehavior::OFF) ||
       (b->object == tetgenbehavior::PLY) ||
       (b->object == tetgenbehavior::STL))) {
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
#define __FUNCT__ "DMComplexGenerate_CTetgen"
PetscErrorCode DMComplexGenerate_CTetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm       comm = ((PetscObject) boundary)->comm;
  DM_Complex    *bd   = (DM_Complex *) boundary->data;
  const PetscInt dim  = 3;
  PLC           *in, *out;
  PetscInt       vStart, vEnd, v, fStart, fEnd, f;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PLCCreate(&in);CHKERRQ(ierr);
  ierr = PLCCreate(&out);CHKERRQ(ierr);
  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscScalar *array;

    ierr = PetscMalloc(in->numberofpoints*dim * sizeof(double), &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberofpoints     * sizeof(int),    &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = VecGetArray(bd->coordinates, &array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(bd->coordSection, v, &off);CHKERRQ(ierr);
      for(d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = array[off+d];
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", v, &in->pointmarkerlist[idx]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(bd->coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMComplexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);
  in->numberoffacets = fEnd - fStart;
  if (in->numberoffacets > 0) {
    ierr = PetscMalloc(in->numberoffacets * sizeof(facet), &in->facetlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in->numberoffacets * sizeof(int),   &in->facetmarkerlist);CHKERRQ(ierr);
    for(f = fStart; f < fEnd; ++f) {
      const PetscInt idx    = f - fStart;
      PetscInt      *points = PETSC_NULL, numPoints, p, numVertices = 0, v;

      in->facetlist[idx].numberofpolygons = 1;
      ierr = PetscMalloc(in->facetlist[idx].numberofpolygons * sizeof(polygon), &in->facetlist[idx].polygonlist);CHKERRQ(ierr);
      in->facetlist[idx].numberofholes    = 0;
      in->facetlist[idx].holelist         = PETSC_NULL;

      ierr = DMComplexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for(p = 0; p < numPoints; ++p) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) {
          points[numVertices++] = point;
        }
      }

      polygon *poly = in->facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      ierr = PetscMalloc(poly->numberofvertices * sizeof(int), &poly->vertexlist);CHKERRQ(ierr);
      for(v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      ierr = DMComplexGetLabelValue(boundary, "marker", f, &in->facetmarkerlist[idx]);CHKERRQ(ierr);
    }
  }
  if (!rank) {
    TetGenOpts t;

    ierr = TetGenOptsInitialize(&t);CHKERRQ(ierr);
    t.in        = boundary; /* Should go away */
    t.plc       = 1;
    t.quality   = 0; /* Change this */
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = 4; /* Change this */
    {
      t.plc        = t.plc || t.diagnose;
      t.useshelles = t.plc || t.refine || t.coarse || t.quality;
      t.goodratio  = t.minratio;
      t.goodratio *= t.goodratio;
      if (t.plc && t.refine) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Switch -r cannot use together with -p.");
      }
      if (t.refine && (t.plc || t.noiterationnum)) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Switches -p, -d, and -I cannot use together with -r.\n");
      }
      if (t.diagnose && (t.quality || t.insertaddpoints || (t.order == 2) || t.neighout || t.docheck)) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Switches -q, -i, -o2, -n, and -C cannot use together with -d.\n");
      }
      /* Be careful not to allocate space for element area constraints that will never be assigned any value (other than the default -1.0). */
      if (!t.refine && !t.plc) {
        t.varvolume = 0;
      }
      /* Be careful not to add an extra attribute to each element unless the input supports it (PLC in, but not refining a preexisting mesh). */
      if (t.refine || !t.plc) {
        t.regionattrib = 0;
      }
      /* If '-a' or '-aa' is in use, enable '-q' option too. */
      if (t.fixedvolume || t.varvolume) {
        if (t.quality == 0) {
          t.quality = 1;
        }
      }
      /* Calculate the goodangle for testing bad subfaces. */
      t.goodangle = cos(t.minangle * PETSC_PI / 180.0);
      t.goodangle *= t.goodangle;
    }
    ierr = TetGenTetrahedralize(&t, in, out);CHKERRQ(ierr);
  }
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(*dm, dim);CHKERRQ(ierr);
  {
    DM_Complex    *mesh        = (DM_Complex *) (*dm)->data;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    int           *cells       = out->tetrahedronlist;
    double        *meshCoords  = out->pointlist;
    PetscInt       coordSize, c;
    PetscScalar   *coords;

    ierr = DMComplexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      ierr = DMComplexSetConeSize(*dm, c, numCorners);CHKERRQ(ierr);
    }
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for(c = 0; c < numCells; ++c) {
      /* Should be numCorners, but c89 sucks shit */
      PetscInt cone[4] = {cells[c*numCorners+0]+numCells, cells[c*numCorners+1]+numCells, cells[c*numCorners+2]+numCells, cells[c*numCorners+3]+numCells};

      ierr = DMComplexSetCone(*dm, c, cone);CHKERRQ(ierr);
    }
    ierr = DMComplexSymmetrize(*dm);CHKERRQ(ierr);
    ierr = DMComplexStratify(*dm);CHKERRQ(ierr);
    if (interpolate) {
      DM        imesh;
      PetscInt *off;
      PetscInt  firstFace = numCells+numVertices, numFaces = 0, face, f, firstEdge, numEdges = 0, edge, e;

      SETERRQ(comm, PETSC_ERR_SUP, "Interpolation is not yet implemented in 3D");
      /* TODO: Rewrite algorithm here to do all meets with neighboring cells and return counts */
      /* Count faces using algorithm from CreateNeighborCSR */
      ierr = DMComplexCreateNeighborCSR(*dm, PETSC_NULL, &off, PETSC_NULL);CHKERRQ(ierr);
      if (off) {
        numFaces = off[numCells]/2;
        /* Account for boundary faces: \sum_c 4 - neighbors = 4*numCells - totalNeighbors */
        numFaces += 4*numCells - off[numCells];
      }
      firstEdge = firstFace+numFaces;
      /* Create interpolated mesh */
      ierr = DMCreate(comm, &imesh);CHKERRQ(ierr);
      ierr = DMSetType(imesh, DMCOMPLEX);CHKERRQ(ierr);
      ierr = DMComplexSetDimension(imesh, dim);CHKERRQ(ierr);
      ierr = DMComplexSetChart(imesh, 0, numCells+numVertices+numEdges);CHKERRQ(ierr);
      for(c = 0; c < numCells; ++c) {
        ierr = DMComplexSetConeSize(imesh, c, numCorners);CHKERRQ(ierr);
      }
      for(f = firstFace; f < firstFace+numFaces; ++f) {
        ierr = DMComplexSetConeSize(imesh, f, 3);CHKERRQ(ierr);
      }
      for(e = firstEdge; e < firstEdge+numEdges; ++e) {
        ierr = DMComplexSetConeSize(imesh, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(imesh);CHKERRQ(ierr);
      for(c = 0, face = firstFace; c < numCells; ++c) {
        const PetscInt *faces;
        PetscInt        numFaces, faceSize, f;

        ierr = DMComplexGetFaces(*dm, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (faceSize != 2) {SETERRQ1(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);}
        for(f = 0; f < numFaces; ++f) {
          PetscBool found = PETSC_FALSE;

          /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
          for(e = firstEdge; e < edge; ++e) {
            const PetscInt *cone;

            ierr = DMComplexGetCone(imesh, e, &cone);CHKERRQ(ierr);
            if (((faces[f*faceSize+0] == cone[0]) && (faces[f*faceSize+1] == cone[1])) ||
                ((faces[f*faceSize+0] == cone[1]) && (faces[f*faceSize+1] == cone[0]))) {
              found = PETSC_TRUE;
              break;
            }
          }
          if (!found) {
            ierr = DMComplexSetCone(imesh, edge, &faces[f*faceSize]);CHKERRQ(ierr);
            ++edge;
          }
          ierr = DMComplexInsertCone(imesh, c, f, e);CHKERRQ(ierr);
        }
      }
      if (edge != firstEdge+numEdges) {SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);}
      ierr = PetscFree(off);CHKERRQ(ierr);
      ierr = DMComplexSymmetrize(imesh);CHKERRQ(ierr);
      ierr = DMComplexStratify(imesh);CHKERRQ(ierr);
      mesh = (DM_Complex *) (imesh)->data;
      for(c = 0; c < numCells; ++c) {
        const PetscInt *cone, *faces;
        PetscInt        coneSize, coff, numFaces, faceSize, f;

        ierr = DMComplexGetConeSize(imesh, c, &coneSize);CHKERRQ(ierr);
        ierr = DMComplexGetCone(imesh, c, &cone);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
        ierr = DMComplexGetFaces(*dm, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
        if (coneSize != numFaces) {SETERRQ3(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numFaces);}
        for(f = 0; f < numFaces; ++f) {
          const PetscInt *econe;
          PetscInt        esize;

          ierr = DMComplexGetConeSize(imesh, cone[f], &esize);CHKERRQ(ierr);
          ierr = DMComplexGetCone(imesh, cone[f], &econe);CHKERRQ(ierr);
          if (esize != 2) {SETERRQ2(((PetscObject) imesh)->comm, PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[f]);}
          if ((faces[f*faceSize+0] == econe[0]) && (faces[f*faceSize+1] == econe[1])) {
            /* Correctly oriented */
            mesh->coneOrientations[coff+f] = 0;
          } else if ((faces[f*faceSize+0] == econe[1]) && (faces[f*faceSize+1] == econe[0])) {
            /* Start at index 1, and reverse orientation */
            mesh->coneOrientations[coff+f] = -(1+1);
          }
        }
      }
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = imesh;
    }
    ierr = PetscSectionSetChart(mesh->coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
    for(v = numCells; v < numCells+numVertices; ++v) {
      ierr = PetscSectionSetDof(mesh->coordSection, v, dim);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(mesh->coordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(mesh->coordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecSetSizes(mesh->coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(mesh->coordinates);CHKERRQ(ierr);
    ierr = VecGetArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      coords[v*dim+0] = meshCoords[v*dim+0];
      coords[v*dim+1] = meshCoords[v*dim+1];
      coords[v*dim+2] = meshCoords[v*dim+2];
    }
    ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
    for(v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMComplexSetLabelValue(*dm, "marker", v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for(e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMComplexJoinPoints(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);}
          ierr = DMComplexSetLabelValue(*dm, "marker", edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
        }
      }
      for(f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMComplexJoinPoints(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) {SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);}
          ierr = DMComplexSetLabelValue(*dm, "marker", faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateBoxMesh"
PetscErrorCode CreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool interpolate, DM *dm) {
  DM             boundary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(boundary,dim,2);
  ierr = DMSetType(boundary, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(boundary, dim-1);CHKERRQ(ierr);
  switch(dim) {
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};
    PetscInt  faces[3] = {1, 1, 1};

    ierr = DMComplexCreateCubeBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
  }
  ierr = DMComplexGenerate_CTetgen(boundary, interpolate, dm);CHKERRQ(ierr);
  ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  //const char    *partitioner     = user->partitioner;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = CreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);
#if 0
  {
    DM refinedMesh     = PETSC_NULL;
    DM distributedMesh = PETSC_NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMComplexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    /* Distribute mesh over processes */
    ierr = DMComplexDistribute(*dm, partitioner, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
#endif
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
