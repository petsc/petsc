#if !defined(_NETWORKIMPL_H)
#define _NETWORKIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmnetwork.h> /*I      "petscdmnetwork.h"    I*/
#include <petsc/private/dmpleximpl.h>  /*I  "petscdmplex.h"  I*/

#define MAX_DATA_AT_POINT 36

#define MAX_COMPONENTS 16

typedef struct _p_DMNetworkComponentHeader *DMNetworkComponentHeader;
struct _p_DMNetworkComponentHeader {
  PetscInt index;    /* index for user input global edge and vertex */
  PetscInt subnetid; /* Id for subnetwork */
  PetscInt ndata;
  PetscInt size[MAX_DATA_AT_POINT];
  PetscInt key[MAX_DATA_AT_POINT];
  PetscInt offset[MAX_DATA_AT_POINT];
  PetscInt nvar[MAX_DATA_AT_POINT]; /* Number of variables */
  PetscInt offsetvarrel[MAX_DATA_AT_POINT]; /* offset from the first variable of the network point */
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

typedef struct _p_DMNetworkComponentValue *DMNetworkComponentValue;
struct _p_DMNetworkComponentValue {
  void* data[MAX_DATA_AT_POINT];
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

typedef struct {
  char     name[32-sizeof(PetscInt)];
  PetscInt size;
} DMNetworkComponent PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));


/* Indexing data structures for vertex and edges */
typedef struct {
  PetscSection                      DofSection;
  PetscSection                      GlobalDofSection;
  ISLocalToGlobalMapping            mapping;
  PetscSF                           sf;
} DMNetworkVertexInfo;

typedef struct {
  PetscSection                      DofSection;
  PetscSection                      GlobalDofSection;
  ISLocalToGlobalMapping            mapping;
  PetscSF                           sf;
} DMNetworkEdgeInfo;

typedef struct {
  PetscInt  Nvtx, nvtx;     /* Number of global/local vertices */
  PetscInt  Nedge,nedge;    /* Number of global/local edges */
  PetscInt  eStart, eEnd;   /* Range of edge numbers (start, end+1) */
  PetscInt  vStart, vEnd;   /* Range of vertex numbers (start, end+1) */
  PetscInt  *edgelist;      /* User provided list of edges. Each edge has the format [from to] where from and to are the vertices covering the edge */
  PetscInt  *vertices;      /* Vertices for this subnetwork. These are mapped to the vertex numbers for the whole network */
  PetscInt  *edges;         /* Edges for this subnetwork. These are mapped to the edge numbers for the whole network */
} DMSubnetwork;

typedef struct {
  PetscInt                          refct;       /* reference count */
  PetscInt                          NEdges,nEdges;        /* Number of global/local edges */
  PetscInt                          NVertices,nVertices; /* Number of global/local vertices */
  PetscInt                          pStart,pEnd; /* Start and end indices for topological points */
  PetscInt                          vStart,vEnd; /* Start and end indices for vertices */
  PetscInt                          eStart,eEnd; /* Start and end indices for edges */
  DM                                plex;        /* DM created from Plex */
  PetscSection                      DataSection; /* Section for managing parameter distribution */
  PetscSection                      DofSection;  /* Section for managing data distribution */
  PetscSection                      GlobalDofSection; /* Global Dof section */
  PetscBool                         distributecalled; /* Flag if DMNetworkDistribute() is called */
  PetscInt                          *vltog;           /* Maps vertex local ordering to global ordering, include ghost vertices */

  DMNetworkVertexInfo               vertex;
  DMNetworkEdgeInfo                 edge;

  PetscInt                          ncomponent; /* Number of components */
  DMNetworkComponent                component[MAX_COMPONENTS]; /* List of components */
  DMNetworkComponentHeader          header;
  DMNetworkComponentValue           cvalue;
  PetscInt                          dataheadersize;
  DMNetworkComponentGenericDataType *componentdataarray; /* Array to hold the data */

  PetscInt                          nsubnet;  /* Global number of subnetworks, including coupling subnetworks */
  PetscInt                          ncsubnet; /* Global number of coupling subnetworks */
  DMSubnetwork                      *subnet;  /* Subnetworks */
  PetscInt                          *subnetvtx; /* Maps local vertex to local subnetwork's vertex */

  PetscBool                         userEdgeJacobian,userVertexJacobian;  /* Global flag for using user's sub Jacobians */
  Mat                               *Je;  /* Pointer array to hold local sub Jacobians for edges, 3 elements for an edge */
  Mat                               *Jv;  /* Pointer array to hold local sub Jacobians for vertices, 1+2*nsupportedges for a vertex */
  PetscInt                          *Jvptr;   /* index of Jv for v-th vertex
                                              Jvpt[v-vStart]:    Jacobian(v,v)
                                              Jvpt[v-vStart]+2i+1: Jacobian(v,e[i]),   e[i]: i-th supporting edge
                                              Jvpt[v-vStart]+2i+2: Jacobian(v,vc[i]), vc[i]: i-th connected vertex
                                              */
} DM_Network;

#endif /* _NETWORKIMPL_H */
