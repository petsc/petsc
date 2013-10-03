#if !defined(_CIRCUITIMPL_H)
#define _CIRCUITIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmcircuit.h> /*I      "petscdmcircuit.h"    I*/
#include "petsc-private/dmimpl.h"

#define MAX_DATA_AT_POINT 14

typedef struct _p_DMCircuitComponentHeader *DMCircuitComponentHeader;
struct _p_DMCircuitComponentHeader {
  PetscInt ndata; 
  PetscInt size[MAX_DATA_AT_POINT];
  PetscInt key[MAX_DATA_AT_POINT];
  PetscInt offset[MAX_DATA_AT_POINT];
};

typedef struct _p_DMCircuitComponentValue *DMCircuitComponentValue;
struct _p_DMCircuitComponentValue {
  void* data[MAX_DATA_AT_POINT];
};

typedef struct {
  char name[20];
  PetscInt size;
}DMCircuitComponent;

typedef struct {
  PetscInt                          refct;  /* reference count */
  PetscInt                          NEdges; /* Number of global edges */
  PetscInt                          NNodes; /* Number of global nodes */
  PetscInt                          nEdges; /* Number of local edges */
  PetscInt                          nNodes; /* Number of local nodes */
  PetscInt                          *edges; /* Edge list */
  PetscInt                          pStart,pEnd; /* Start and end indices for topological points */
  PetscInt                          vStart,vEnd; /* Start and end indices for vertices */
  PetscInt                          eStart,eEnd; /* Start and end indices for edges */
  DM                                plex;     /* DM created from Plex */
  PetscSection                      DataSection; /* Section for managing parameter distribution */
  PetscSection                      DofSection;  /* Section for managing data distribution */
  PetscSection                      GlobalDofSection; /* Global Dof section */
  PetscInt                          ncomponent; /* Number of components */
  DMCircuitComponent                component[10]; /* List of components */
  DMCircuitComponentHeader          header;  
  DMCircuitComponentValue           cvalue;
  PetscInt                          dataheadersize;
  DMCircuitComponentGenericDataType *componentdataarray; /* Array to hold the data */
} DM_Circuit;

#endif /* _CIRCUITIMPL_H */
