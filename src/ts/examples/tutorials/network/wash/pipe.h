#ifndef PIPE_H
#define PIPE_H

#define GRAV 9.806
#define PIPE_CHARACTERISTIC 10000000.0

#include <petscts.h>
#include <petscdmda.h>


typedef struct {
  PetscScalar q;       /* flow rate */
  PetscScalar h;       /* pressure */
} PipeField;

typedef struct {
  PetscScalar Q0,H0;       /* boundary values in upstream */
  PetscScalar QL,HL;       /* boundary values in downstream */
} PipeBoundary;

/* pipe                 */
/*----------------------*/
struct _p_Pipe
{
  MPI_Comm     comm;
  PetscInt     id;
  DM           da;
  Vec          x;
  PetscInt     nnodes;   /* number of nodes in da discretization */
  PetscReal    length;   /* pipe length */
  PetscReal    a;        /* natural flow speed */
  PetscReal    fric;     /* friction */
  PetscReal    D;        /* diameter */
  PetscReal    A;        /* area of cross section */
  PetscReal    R;        
  PetscReal    rad;     
  PetscScalar  H0,QL;    /* left and right boundary conditions for H and Q */
  PipeBoundary boundary; /* boundary conditions for H and Q */
};
typedef struct _p_Pipe *Pipe;

extern PetscErrorCode PipeCreate(MPI_Comm,Pipe*);
extern PetscErrorCode PipeDestroy(Pipe*);
extern PetscErrorCode PipeSetParameters(Pipe,PetscReal,PetscInt,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode PipeSetUp(Pipe);

extern PetscErrorCode PipeComputeSteadyState(Pipe,PetscScalar,PetscScalar);
extern PetscErrorCode PipeIFunctionLocal(DMDALocalInfo*,PetscReal,PipeField*,PipeField*,PipeField*,Pipe);

#endif
