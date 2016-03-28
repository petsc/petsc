#if !defined(_NETWORKIMPL_H)
#define _NETWORKIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmnetwork.h> /*I      "petscdmnetwork.h"    I*/
#include "petsc/private/dmimpl.h"

#define MAX_DATA_AT_POINT 14

typedef struct _p_DMNetworkComponentHeader *DMNetworkComponentHeader;
struct _p_DMNetworkComponentHeader {
  PetscInt ndata;
  PetscInt size[MAX_DATA_AT_POINT];
  PetscInt key[MAX_DATA_AT_POINT];
  PetscInt offset[MAX_DATA_AT_POINT];
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));

typedef struct _p_DMNetworkComponentValue *DMNetworkComponentValue;
struct _p_DMNetworkComponentValue {
  void* data[MAX_DATA_AT_POINT];
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));

typedef struct {
  char     name[32-sizeof(PetscInt)];
  PetscInt size;
} DMNetworkComponent PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));

typedef struct {
  PetscInt                          refct;  /* reference count */
  PetscInt                          NEdges; /* Number of global edges */
  PetscInt                          NNodes; /* Number of global nodes */
  PetscInt                          nEdges; /* Number of local edges */
  PetscInt                          nNodes; /* Number of local nodes */
  int                               *edges; /* Edge list */
  PetscInt                          pStart,pEnd; /* Start and end indices for topological points */
  PetscInt                          vStart,vEnd; /* Start and end indices for vertices */
  PetscInt                          eStart,eEnd; /* Start and end indices for edges */
  DM                                plex;     /* DM created from Plex */
  PetscSection                      DataSection; /* Section for managing parameter distribution */
  PetscSection                      DofSection;  /* Section for managing data distribution */
  PetscSection                      GlobalDofSection; /* Global Dof section */
  PetscInt                          ncomponent; /* Number of components */
  DMNetworkComponent                component[10]; /* List of components */
  DMNetworkComponentHeader          header;
  DMNetworkComponentValue           cvalue;
  PetscInt                          dataheadersize;
  DMNetworkComponentGenericDataType *componentdataarray; /* Array to hold the data */

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
