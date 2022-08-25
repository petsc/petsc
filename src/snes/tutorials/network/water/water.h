#ifndef WATER_H
#define WATER_H

#include <petscsnes.h>
#include <petscdmnetwork.h>

#define MAXLINE               1000
#define VERTEX_TYPE_RESERVOIR 0
#define VERTEX_TYPE_JUNCTION  1
#define VERTEX_TYPE_TANK      2
#define EDGE_TYPE_PIPE        0
#define EDGE_TYPE_PUMP        1
#define PIPE_STATUS_OPEN      0
#define PIPE_STATUS_CLOSED    1
#define PIPE_STATUS_CV        2

#define GPM_CFS 0.0022280023234587 /* Scaling constant for GPM to CFS conversion */

typedef struct {
  PetscInt compkey_edge;
  PetscInt compkey_vtx;
} AppCtx_Water;

typedef struct {
  PetscInt    id;         /* id */
  PetscScalar elev;       /* elevation (ft) */
  PetscScalar demand;     /* demand (gpm) */
  PetscInt    dempattern; /* demand pattern id */
} Junction;

typedef struct {
  PetscInt    id;          /* id */
  PetscScalar head;        /* head (ft) */
  PetscInt    headpattern; /* head pattern */
} Reservoir;

typedef struct {
  PetscInt    id;          /* id */
  PetscScalar elev;        /* elevation (ft) */
  PetscScalar initlvl;     /* initial level (ft) */
  PetscScalar minlvl;      /* minimum level (ft) */
  PetscScalar maxlvl;      /* maximum level (ft) */
  PetscScalar diam;        /* diameter (ft) */
  PetscScalar minvolume;   /* minimum volume (ft^3) */
  PetscInt    volumecurve; /* Volume curve id */
} Tank;

struct _p_VERTEX_Water {
  PetscInt  id;   /* id */
  PetscInt  type; /* vertex type (junction, reservoir) */
  Junction  junc; /* junction data */
  Reservoir res;  /* reservoir data */
  Tank      tank; /* tank data */
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double), sizeof(PetscScalar)));
typedef struct _p_VERTEX_Water *VERTEX_Water;

typedef struct {
  PetscInt    id;        /* id */
  PetscInt    node1;     /* From node */
  PetscInt    node2;     /* to node */
  PetscScalar length;    /* length (ft) */
  PetscScalar diam;      /* diameter (inches) */
  PetscScalar roughness; /* roughness (dimensionless) */
  PetscScalar minorloss; /* minor losses */
  char        stat[16];  /* Status */
  PetscInt    status;    /* Pipe status (see PIPE_STATUS_XXX definition on top) */
  PetscScalar n;         /* Exponent for h = kQ^n */
  PetscScalar k;
} Pipe;

typedef struct {
  PetscInt id;           /* id */
  PetscInt node1;        /* From node */
  PetscInt node2;        /* to node */
  char     param[16];    /* curve parameter (HEAD or ENERGY or EFFICIENCY) */
  PetscInt paramid;      /* Id of the curve parameter in the CURVE data */
  struct {               /* one point curve */
    PetscScalar flow[3]; /* flow (gpm) */
    PetscScalar head[3]; /* head (ft) */
    PetscInt    npt;     /* Number of given points */
  } headcurve;
  /* Parameters for pump headloss equation hL = h0 - rQ^n */
  PetscScalar h0;
  PetscScalar r;
  PetscScalar n;
} Pump;

struct _p_EDGE_Water {
  PetscInt id;   /* id */
  PetscInt type; /* edge type (pump, pipe) */
  Pipe     pipe; /* pipe data */
  Pump     pump; /* pump data */
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double), sizeof(PetscScalar)));
typedef struct _p_EDGE_Water *EDGE_Water;

/* EPANET top-level data structure */
struct _p_WATERDATA {
  PetscInt     nvertex;
  PetscInt     nedge;
  PetscInt     njunction;
  PetscInt     nreservoir;
  PetscInt     ntank;
  PetscInt     npipe;
  PetscInt     npump;
  VERTEX_Water vertex;
  EDGE_Water   edge;
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double), sizeof(PetscScalar)));
typedef struct _p_WATERDATA WATERDATA;

extern PetscErrorCode WaterReadData(WATERDATA *, char *);
extern PetscErrorCode GetListofEdges_Water(WATERDATA *, PetscInt *);
extern PetscErrorCode WaterSetInitialGuess(DM, Vec);
extern PetscErrorCode WaterFormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormFunction_Water(DM, Vec, Vec, PetscInt, PetscInt, const PetscInt *, const PetscInt *, void *);
extern PetscErrorCode SetInitialGuess_Water(DM, Vec, PetscInt, PetscInt, const PetscInt *, const PetscInt *, void *);
extern PetscScalar    Flow_Pipe(Pipe *, PetscScalar, PetscScalar);
extern PetscScalar    Flow_Pump(Pump *, PetscScalar, PetscScalar);
#endif
