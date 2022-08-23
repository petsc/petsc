#ifndef PIPE_H
#define PIPE_H

#define GRAV                9.806
#define PIPE_CHARACTERISTIC 10000000.0

#include <petsc.h>

typedef struct {
  PetscScalar q; /* flow rate */
  PetscScalar h; /* pressure */
} PipeField;

typedef struct {
  PetscScalar Q0, H0; /* boundary values in upstream */
  PetscScalar QL, HL; /* boundary values in downstream */
} PipeBoundary;

/* pipe                 */
/*----------------------*/
struct _p_Pipe {
  /* identification variables */
  PetscInt id;
  PetscInt networkid; /* which network this pipe belongs */

  /* solver objects */
  Vec        x;
  PipeField *xold;
  PetscReal  dt;
  DM         da;
  PetscInt   nnodes; /* number of nodes in da discretization */
  Mat       *jacobian;

  /* physics */
  PetscReal    length; /* pipe length */
  PetscReal    a;      /* natural flow speed */
  PetscReal    fric;   /* friction */
  PetscReal    D;      /* diameter */
  PetscReal    A;      /* area of cross section */
  PetscReal    R;
  PetscReal    rad;
  PipeBoundary boundary; /* boundary conditions for H and Q */
} PETSC_ATTRIBUTEALIGNED(PetscMax(sizeof(double), sizeof(PetscScalar)));

typedef struct _p_Pipe *Pipe;

extern PetscErrorCode PipeCreate(MPI_Comm, Pipe *);
extern PetscErrorCode PipeDestroy(Pipe *);
extern PetscErrorCode PipeSetParameters(Pipe, PetscReal, PetscReal, PetscReal, PetscReal);
extern PetscErrorCode PipeSetUp(Pipe);
extern PetscErrorCode PipeCreateJacobian(Pipe, Mat *, Mat *[]);
extern PetscErrorCode PipeDestroyJacobian(Pipe);

extern PetscErrorCode PipeComputeSteadyState(Pipe, PetscScalar, PetscScalar);
extern PetscErrorCode PipeIFunctionLocal(DMDALocalInfo *, PetscReal, PipeField *, PipeField *, PipeField *, Pipe);
extern PetscErrorCode PipeIFunctionLocal_Lax(DMDALocalInfo *, PetscReal, PipeField *, PipeField *, PetscScalar *, Pipe);
extern PetscErrorCode PipeRHSFunctionLocal(DMDALocalInfo *, PetscReal, PipeField *, PetscScalar *, Pipe);
extern PetscErrorCode PipeMonitor(TS, PetscInt, PetscReal, Vec, void *);

extern PetscErrorCode PipeCreateJacobian(Pipe, Mat *, Mat *[]);
extern PetscErrorCode PipeDestroyJacobian(Pipe);
#endif
