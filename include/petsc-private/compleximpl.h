#if !defined(_COMPLEXIMPL_H)
#define _COMPLEXIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmcomplex.h> /*I      "petscdmcomplex.h"    I*/
#include "petsc-private/dmimpl.h"

typedef struct Sieve_Label *SieveLabel;
struct Sieve_Label {
  char      *name;           /* Label name */
  PetscInt   numStrata;      /* Number of integer values */
  PetscInt  *stratumValues;  /* Value of each stratum */
  PetscInt  *stratumOffsets; /* Offset of each stratum */
  PetscInt  *stratumSizes;   /* Size of each stratum */
  PetscInt  *points;         /* Points for each stratum, sorted after setup */
  SieveLabel next;           /* Linked list */
};

typedef struct {
  PetscInt             dim;   /* Topological mesh dimension */
  PetscSF              sf;    /* SF for parallel point overlap */
  PetscSF              sfDefault; /* SF for parallel dof overlap using default section */

  /* Sieve */
  PetscSection         coneSection;      /* Layout of cones (inedges for DAG) */
  PetscInt             maxConeSize;      /* Cached for fast lookup */
  PetscInt            *cones;            /* Cone for each point */
  PetscInt            *coneOrientations; /* Orientation of each cone point, means cone traveral should start on point 'o', and if negative start on -(o+1) and go in reverse */
  PetscSection         supportSection;   /* Layout of cones (inedges for DAG) */
  PetscInt             maxSupportSize;   /* Cached for fast lookup */
  PetscInt            *supports;         /* Cone for each point */
  PetscSection         coordSection;     /* Layout for coordinates */
  Vec                  coordinates;      /* Coordinate values */
  PetscReal            refinementLimit;  /* Maximum volume for refined cell */

  PetscInt            *meetTmpA,    *meetTmpB;    /* Work space for meet operation */
  PetscInt            *joinTmpA,    *joinTmpB;    /* Work space for join operation */
  PetscInt            *closureTmpA, *closureTmpB; /* Work space for closure operation */
  PetscInt            *facesTmp;                  /* Work space for faces operation */

  /* Labels */
  SieveLabel           labels;         /* Linked list of labels */

  PetscSection            defaultSection;
  PetscSection            defaultGlobalSection;
  DMComplexLocalFunction1 lf;
  DMComplexLocalJacobian1 lj;

  /* Debugging */
  PetscBool               printSetValues;
} DM_Complex;

/****** TetGen Reimplementation ******/

/* Replaces tetgenbehavior x*/
typedef enum {TETGEN_OBJECT_NONE, TETGEN_OBJECT_NODES, TETGEN_OBJECT_POLY, TETGEN_OBJECT_OFF, TETGEN_OBJECT_PLY, TETGEN_OBJECT_STL, TETGEN_OBJECT_MEDIT, TETGEN_OBJECT_VTK, TETGEN_OBJECT_MESH} TetGenObjectType;

typedef struct {
  DM  in; /* Eventually make this a PetscObject */
  int plc;                                                 /* '-p' switch, 0. */
  int quality;                                             /* '-q' switch, 0. */
  int refine;                                              /* '-r' switch, 0. */
  int coarse;                                              /* '-R' switch, 0. */
  int metric;                                              /* '-m' switch, 0. */
  int varvolume;                            /* '-a' switch without number, 0. */
  int fixedvolume;                             /* '-a' switch with number, 0. */
  int insertaddpoints;                                     /* '-i' switch, 0. */
  int regionattrib;                                        /* '-A' switch, 0. */
  int conformdel;                                          /* '-D' switch, 0. */
  int diagnose;                                            /* '-d' switch, 0. */
  int zeroindex;                                           /* '-z' switch, 0. */
  int btree;                                                        /* -u, 1. */
  int max_btreenode_size;                            /* number after -u, 100. */
  int optlevel;                     /* number specified after '-s' switch, 3. */
  int optpasses;                   /* number specified after '-ss' switch, 3. */
  int order;                /* element order, specified after '-o' switch, 1. */
  int facesout;                                            /* '-f' switch, 0. */
  int edgesout;                                            /* '-e' switch, 0. */
  int neighout;                                            /* '-n' switch, 0. */
  int voroout;                                             /* '-v',switch, 0. */
  int meditview;                                           /* '-g' switch, 0. */
  int gidview;                                             /* '-G' switch, 0. */
  int geomview;                                            /* '-O' switch, 0. */
  int vtkview;                                             /* '-K' switch, 0. */
  int nobound;                                             /* '-B' switch, 0. */
  int nonodewritten;                                       /* '-N' switch, 0. */
  int noelewritten;                                        /* '-E' switch, 0. */
  int nofacewritten;                                       /* '-F' switch, 0. */
  int noiterationnum;                                      /* '-I' switch, 0. */
  int nomerge;                                             /* '-M',switch, 0. */
  int nobisect;             /* count of how often '-Y' switch is selected, 0. */
  int noflip;                        /* do not perform flips. '-X' switch. 0. */
  int nojettison;        /* do not jettison redundants nodes. '-J' switch. 0. */
  int steiner;                                /* number after '-S' switch. 0. */
  int fliprepair;                                          /* '-X' switch, 1. */
  int offcenter;                                           /* '-R' switch, 0. */
  int docheck;                                             /* '-C' switch, 0. */
  int quiet;                                               /* '-Q' switch, 0. */
  int verbose;              /* count of how often '-V' switch is selected, 0. */
  int useshelles;               /* '-p', '-r', '-q', '-d', or '-R' switch, 0. */
  int maxflipedgelinksize;        /* The maximum flippable edge link size 10. */
  PetscReal minratio;                            /* number after '-q' switch, 2.0. */
  PetscReal goodratio;                  /* number calculated from 'minratio', 0.0. */
  PetscReal minangle;                                /* minimum angle bound, 20.0. */
  PetscReal goodangle;                         /* cosine squared of minangle, 0.0. */
  PetscReal maxvolume;                          /* number after '-a' switch, -1.0. */
  PetscReal mindihedral;                        /* number after '-qq' switch, 5.0. */
  PetscReal maxdihedral;                     /* number after '-qqq' switch, 165.0. */
  PetscReal alpha1;                          /* number after '-m' switch, sqrt(2). */
  PetscReal alpha2;                             /* number after '-mm' switch, 1.0. */
  PetscReal alpha3;                            /* number after '-mmm' switch, 0.6. */
  PetscReal epsilon;                          /* number after '-T' switch, 1.0e-8. */
  PetscReal epsilon2;                        /* number after '-TT' switch, 1.0e-5. */
  TetGenObjectType object;            /* determined by -p, or -r switch. NONE. */
} TetGenOpts;

/* The polygon data structure.  A "polygon" describes a simple polygon  */
/*   (no holes). It is not necessarily convex.  Each polygon contains a */
/*   number of corners (points) and the same number of sides (edges).   */
/* Note that the points of the polygon must be given in either counter- */
/*   clockwise or clockwise order and they form a ring, so every two    */
/*   consective points forms an edge of the polygon.                    */
typedef struct {
  int *vertexlist;
  int numberofvertices;
} polygon;

/* The facet data structure.  A "facet" describes a facet. Each facet is */
/*   a polygonal region possibly with holes, edges, and points in it.    */
typedef struct {
  polygon *polygonlist;
  int numberofpolygons;
  PetscReal *holelist;
  int numberofholes;
} facet;

/* The periodic boundary condition group data structure.  A "pbcgroup"   */
/*   contains the definition of a pbc and the list of pbc point pairs.   */
/*   'fmark1' and 'fmark2' are the facetmarkers of the two pbc facets f1 */
/*   and f2, respectively. 'transmat' is the transformation matrix which */
/*   maps a point in f1 into f2.  An array of pbc point pairs are saved  */
/*   in 'pointpairlist'. The first point pair is at indices [0] and [1], */
/*   followed by remaining pairs. Two integers per pair.                 */
typedef struct {
  int fmark1, fmark2;
  PetscReal transmat[4][4];
  int numberofpointpairs;
  int *pointpairlist;
} pbcgroup;

/* A 'voroedge' is an edge of the Voronoi diagram. It corresponds to a   */
/*   Delaunay face.  Each voroedge is either a line segment connecting   */
/*   two Voronoi vertices or a ray starting from a Voronoi vertex to an  */
/*   "infinite vertex".  'v1' and 'v2' are two indices pointing to the   */
/*   list of Voronoi vertices. 'v1' must be non-negative, while 'v2' may */
/*   be -1 if it is a ray, in this case, the unit normal of this ray is  */
/*   given in 'vnormal'.                                                 */
typedef struct {
  int v1, v2;
  PetscReal vnormal[3];
} voroedge;

/* A 'vorofacet' is an facet of the Voronoi diagram. It corresponds to a  */
/*   Delaunay edge.  Each Voronoi facet is a convex polygon formed by a   */
/*   list of Voronoi edges, it may not be closed.  'c1' and 'c2' are two  */
/*   indices pointing into the list of Voronoi cells, i.e., the two cells */
/*   share this facet.  'elist' is an array of indices pointing into the  */
/*   list of Voronoi edges, 'elist[0]' saves the number of Voronoi edges  */
/*   (including rays) of this facet.                                      */
typedef struct {
  int c1, c2;
  int *elist;
} vorofacet;

/* A callback function for mesh refinement. */
typedef PetscBool (*TetSizeFunc)(PetscReal*, PetscReal*, PetscReal*, PetscReal*, PetscReal*, PetscReal);

/* This replaces tetgenio */
typedef struct {
  /* Items are numbered starting from 'firstnumber' (0 or 1), default is 0.*/
  int firstnumber;

  /* Dimension of the mesh (2 or 3), default is 3.*/
  int mesh_dim;

  /* Does the lines in .node file contain index or not, default is TRUE. */
  PetscBool useindex;

  /* 'pointlist':  An array of point coordinates.  The first point's x   */
  /*   coordinate is at index [0] and its y coordinate at index [1], its */
  /*   z coordinate is at index [2], followed by the coordinates of the  */
  /*   remaining points.  Each point occupies three PetscReals.          */
  /* 'pointattributelist':  An array of point attributes.  Each point's  */
  /*   attributes occupy 'numberofpointattributes' PetscReals.           */
  /* 'pointmtrlist': An array of metric tensors at points. Each point's  */
  /*   tensor occupies 'numberofpointmtr' PetscReals.                    */
  /* `pointmarkerlist':  An array of point markers; one int per point.   */
  PetscReal *pointlist;
  PetscReal *pointattributelist;
  PetscReal *pointmtrlist;
  int *pointmarkerlist;
  int numberofpoints;
  int numberofpointattributes;
  int numberofpointmtrs;

  /* `elementlist':  An array of element (triangle or tetrahedron) corners.  */
  /*   The first element's first corner is at index [0], followed by its     */
  /*   other corners in counterclockwise order, followed by any other        */
  /*   nodes if the element represents a nonlinear element.  Each element    */
  /*   occupies `numberofcorners' ints.                                      */
  /* `elementattributelist':  An array of element attributes.  Each          */
  /*   element's attributes occupy `numberofelementattributes' PetscReals.   */
  /* `elementconstraintlist':  An array of constraints, i.e. triangle's      */
  /*   area or tetrahedron's volume; one PetscReal per element.  Input only. */
  /* `neighborlist':  An array of element neighbors; 3 or 4 ints per         */
  /*   element.  Output only.                                                */
  int *tetrahedronlist;
  PetscReal *tetrahedronattributelist;
  PetscReal *tetrahedronvolumelist;
  int *neighborlist;
  int numberoftetrahedra;
  int numberofcorners;
  int numberoftetrahedronattributes;

  /* `facetlist':  An array of facets.  Each entry is a structure of facet. */
  /* `facetmarkerlist':  An array of facet markers; one int per facet.      */
  facet *facetlist;
  int *facetmarkerlist;
  int numberoffacets;

  /* `holelist':  An array of holes.  The first hole's x, y and z    */
  /*   coordinates  are at indices [0], [1] and [2], followed by the */
  /*   remaining holes. Three PetscReals per hole.                   */
  PetscReal *holelist;
  int numberofholes;

  /* `regionlist': An array of regional attributes and volume constraints.      */
  /*   The first constraint's x, y and z coordinates are at indices [0],        */
  /*   [1] and [2], followed by the regional attribute at index [3], foll-      */
  /*   owed by the maximum volume at index [4]. Five PetscReals per constraint. */
  /* Note that each regional attribute is used only if you select the `A'       */
  /*   switch, and each volume constraint is used only if you select the        */
  /*   `a' switch (with no number following).                                   */
  PetscReal *regionlist;
  int numberofregions;

  /* `facetconstraintlist': An array of facet maximal area constraints.     */
  /*   Two PetscReals per constraint. The first (at index [0]) is the facet */
  /*   marker (cast it to int), the second (at index [1]) is its maximum    */
  /*   area bound.                                                          */
  PetscReal *facetconstraintlist;
  int numberoffacetconstraints;

  /* `segmentconstraintlist': An array of segment max. length constraints.     */
  /*   Three PetscReals per constraint. The first two (at indcies [0] and [1]) */
  /*   are the indices of the endpoints of the segment, the third (at index    */
  /*   [2]) is its maximum length bound.                                       */
  PetscReal *segmentconstraintlist;
  int numberofsegmentconstraints;

  /* 'pbcgrouplist':  An array of periodic boundary condition groups. */
  pbcgroup *pbcgrouplist;
  int numberofpbcgroups;

  /* `trifacelist':  An array of triangular face endpoints.  The first   */
  /*   face's endpoints are at indices [0], [1] and [2], followed by the */
  /*   remaining faces.  Three ints per face.                            */
  /* `adjtetlist':  An array of adjacent tetrahedra to the faces of      */
  /*   trifacelist. Each face has at most two adjacent tets, the first   */
  /*   face's adjacent tets are at [0], [1]. Two ints per face. A '-1'   */
  /*   indicates outside (no adj. tet). This list is output when '-nn'   */
  /*   switch is used.                                                   */
  /* `trifacemarkerlist':  An array of face markers; one int per face.   */
  int *trifacelist;
  int *adjtetlist;
  int *trifacemarkerlist;
  int numberoftrifaces;

  /* `edgelist':  An array of edge endpoints.  The first edge's endpoints */
  /*   are at indices [0] and [1], followed by the remaining edges.  Two  */
  /*   ints per edge.                                                     */
  /* `edgemarkerlist':  An array of edge markers; one int per edge.       */
  int *edgelist;
  int *edgemarkerlist;
  int numberofedges;

  /* 'vpointlist':  An array of Voronoi vertex coordinates (like pointlist). */
  /* 'vedgelist':  An array of Voronoi edges.  Each entry is a 'voroedge'.   */
  /* 'vfacetlist':  An array of Voronoi facets. Each entry is a 'vorofacet'. */
  /* 'vcelllist':  An array of Voronoi cells.  Each entry is an array of     */
  /*   indices pointing into 'vfacetlist'. The 0th entry is used to store    */
  /*   the length of this array.                                             */
  PetscReal *vpointlist;
  voroedge *vedgelist;
  vorofacet *vfacetlist;
  int **vcelllist;
  int numberofvpoints;
  int numberofvedges;
  int numberofvfacets;
  int numberofvcells;

  /* A callback function. */
  TetSizeFunc tetunsuitable;
} PLC;

PetscErrorCode PLCCreate(PLC **);
PetscErrorCode PLCDestroy(PLC **);
PetscErrorCode TetGenOptsInitialize(TetGenOpts *);
PetscErrorCode TetGenCheckOpts(TetGenOpts *);
PetscErrorCode TetGenTetrahedralize(TetGenOpts *, PLC *, PLC *);

#endif /* _COMPLEXIMPL_H */
