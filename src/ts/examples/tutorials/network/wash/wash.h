#ifndef WASH_H
#define WASH_H

#include <petscdmnetwork.h>
#include "pipe.h"

/* junction              */
/*-----------------------*/
struct _p_Junction{
  PetscInt	id;                   /* global index */
  PetscInt      isEnd;                /* -1: left end; 0: not an end; 1: right end */
  PetscInt      nedges_in,nedges_out; /* number of connected in/out edges */
  PetscReal     latitude, longitude;  /* GPS data */
};
typedef struct _p_Junction *Junction;

/* wash                   */
/*------------------------*/
struct _p_Wash
{
  MPI_Comm    comm;
  Junction    junction;
  PetscInt    nedge,nvertex,njunction; /* global number of components */
  Vec         localX,localXdot;        /* vectors used in local function evalutation */
  PetscInt    nnodes_loc;              /* num of global and local nodes */

  /* Pipe */
  Pipe        pipe;
  PetscScalar Q0,H0,QL,HL;    /* left and right boundary conditions for wash-network (not individual pipe) */
};
typedef struct _p_Wash *Wash;

extern PetscErrorCode WashNetworkCreate(MPI_Comm,PetscInt,Wash*,int**);
extern PetscErrorCode WashNetworkCleanUp(Wash,int*);
#endif
