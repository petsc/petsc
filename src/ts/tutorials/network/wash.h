#ifndef WASH_H
#define WASH_H

#include <petscdmnetwork.h>
#include "pipe.h"

typedef enum {
  NONE,
  JUNCTION  = 1,
  RESERVOIR = 2,
  VALVE     = 3,
  DEMAND    = 4,
  INFLOW    = 5,
  STAGE     = 6,
  TANK      = 7
} VertexType;

typedef struct {
  PetscInt    rid;  /*reservoir id*/
  PetscScalar hres; /*Reservoir water column*/
} Reservoir;

typedef struct {
  PetscInt    vid; /*valve id*/
  PetscScalar tau; /*valve aperture*/
  PetscScalar cdag;
  PetscScalar qout;
} Valve;

/* junction              */
/*-----------------------*/
struct _p_Junction {
  PetscInt   id;  /* global index */
  PetscInt   tag; /* external id */
  VertexType type;
  PetscInt   isEnd;                 /* -1: left end; 0: not an end; 1: right end */
  PetscInt   nedges_in, nedges_out; /* number of connected in/out edges */
  Mat       *jacobian;
  PetscReal  latitude, longitude; /* GPS data */

  /* boundary data structures */
  Reservoir reservoir;
  Valve     valve;
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double), sizeof(PetscScalar)));
typedef struct _p_Junction *Junction;

extern PetscErrorCode JunctionCreateJacobian(DM, PetscInt, Mat *, Mat *[]);
extern PetscErrorCode JunctionDestroyJacobian(DM, PetscInt, Junction);

/* wash                   */
/*------------------------*/
struct _p_Wash {
  MPI_Comm  comm;
  PetscInt  nedge, nvertex;    /* local number of components */
  PetscInt  Nedge, Nvertex;    /* global number of components */
  PetscInt *edgelist;          /* local edge list */
  Vec       localX, localXdot; /* vectors used in local function evaluation */
  PetscInt  nnodes_loc;        /* num of global and local nodes */

  /* Junction */
  Junction  junction;
  PetscInt *vtype;

  /* Pipe */
  Pipe        pipe;
  PetscScalar Q0, H0, QL, HL; /* left and right boundary conditions for wash-network (not individual pipe) */

  /* Events */
  PetscInt close_valve;
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double), sizeof(PetscScalar)));
typedef struct _p_Wash *Wash;

extern PetscErrorCode WashNetworkCreate(MPI_Comm, PetscInt, Wash *);
extern PetscErrorCode WashNetworkCleanUp(Wash);
#endif
