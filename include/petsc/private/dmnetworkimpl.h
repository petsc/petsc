#if !defined(_NETWORKIMPL_H)
#define _NETWORKIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmnetwork.h> /*I      "petscdmnetwork.h"    I*/
#include <petsc/private/dmpleximpl.h>  /*I  "petscdmplex.h"  I*/
#include <petscctable.h>

/* The maximum number of components registered and maximum number of components per network point */
#define MAX_NETCOMPONENTS 36

typedef struct _p_DMNetworkComponentHeader *DMNetworkComponentHeader;
struct _p_DMNetworkComponentHeader {
  PetscInt index;    /* index for user input global edge and vertex */
  PetscInt subnetid; /* Id for subnetwork */
  PetscInt ndata;    /* number of components */
  PetscInt size[MAX_NETCOMPONENTS];
  PetscInt key[MAX_NETCOMPONENTS];
  PetscInt offset[MAX_NETCOMPONENTS];
  PetscInt nvar[MAX_NETCOMPONENTS]; /* Number of variables */
  PetscInt offsetvarrel[MAX_NETCOMPONENTS]; /* offset from the first variable of the network point */
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double),sizeof(PetscScalar)));

typedef struct _p_DMNetworkComponentValue *DMNetworkComponentValue;
struct _p_DMNetworkComponentValue {
  void* data[MAX_NETCOMPONENTS];
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

/*
  Shared vertex - a vertex in DMNetwork that is shared by 2 or more subnetworks. sv provides the mapping from the subnetwork vertices to the global DMNetwork vertex (merged network).
  sv is organized as (see SVtxCreate())
        sv(net[0],idx[0]) --> sv(net[1],idx[1])
                          --> sv(net[1],idx[1])
                          ...
                          --> sv(net[n-1],idx[n-1])
        and net[0] < net[1] < ... < net[n-1]
        where sv[0] has SVFROM type, sv[i], i>0, has SVTO type.
*/
typedef struct {
  PetscInt gidx;                /* global index of the shared vertices in dmplex */
  PetscInt n;                   /* number of subnetworks that share the common DMNetwork vertex */
  PetscInt *sv;                 /* array of size n: sv[2*i,2*i+1]=(net[i], idx[i]), i=0,...,n-1 */
} SVtx;
typedef enum {SVNONE=-1, SVFROM=0, SVTO=1} SVtxType;

typedef struct {
  PetscInt  Nvtx, nvtx;     /* Number of global/local vertices */
  PetscInt  Nedge,nedge;    /* Number of global/local edges */
  PetscInt  eStart, eEnd;   /* Range of edge numbers (start, end+1) */
  PetscInt  vStart, vEnd;   /* Range of vertex numbers (start, end+1) */
  PetscInt  *edgelist;      /* User provided list of edges. Each edge has the format [from to] where from and to are the vertices covering the edge in the subnet numbering */
  PetscInt  *vertices;      /* Vertices for this subnetwork. These are mapped to the vertex numbers for the whole network */
  PetscInt  *edges;         /* Edges for this subnetwork. These are mapped to the edge numbers for the whole network */
  char      name[32-sizeof(PetscInt)];
} DMSubnetwork;

typedef struct {
  PetscInt                          refct;               /* reference count */
  PetscInt                          NEdges,nEdges;       /* Number of global/local edges */
  PetscInt                          NVertices,nVertices; /* Number of global/local vertices */
  PetscInt                          pStart,pEnd;         /* Start and end indices for topological points */
  PetscInt                          vStart,vEnd;         /* Start and end indices for vertices */
  PetscInt                          eStart,eEnd;         /* Start and end indices for edges */
  DM                                plex;                /* DM created from Plex */
  PetscSection                      DataSection;         /* Section for managing parameter distribution */
  PetscSection                      DofSection;          /* Section for managing data distribution */
  PetscSection                      GlobalDofSection;    /* Global Dof section */
  PetscBool                         distributecalled;    /* Flag if DMNetworkDistribute() is called */
  PetscInt                          *vltog;              /* Maps vertex local ordering to global ordering, include ghost vertices */

  DMNetworkVertexInfo               vertex;
  DMNetworkEdgeInfo                 edge;

  PetscInt                          ncomponent; /* Number of components */
  DMNetworkComponent                component[MAX_NETCOMPONENTS]; /* List of components */
  DMNetworkComponentHeader          header;
  DMNetworkComponentValue           cvalue;
  PetscInt                          dataheadersize;
  DMNetworkComponentGenericDataType *componentdataarray; /* Array to hold the data */

  PetscInt                          nsubnet,Nsubnet; /* Local and global number of subnetworks */
  DMSubnetwork                      *subnet;         /* Subnetworks */
  PetscInt                          *subnetvtx;      /* Maps local vertex to local subnetwork's vertex */
  SVtx                              *svtx;           /* Array of vertices shared by subnetworks */
  PetscInt                          nsvtx,Nsvtx;     /* Local and global num of entries in svtx */
  PetscInt                          *svertices;      /* Array of local subnetwork vertices that are merged/shared */
  PetscInt                          *sedgelist;      /* Edge list of shared vertices */
  PetscTable                        svtable;         /* hash table for finding shared vertex info */

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
