#if !defined(PETSCGL_H)
#define PETSCGL_H

#include <petsc/private/tsimpl.h>

typedef enum {TSGLLEERROR_FORWARD,TSGLLEERROR_BACKWARD} TSGLLEErrorDirection;

typedef struct _TSGLLEScheme *TSGLLEScheme;
struct _TSGLLEScheme {
  PetscInt    p;                /* order of the method */
  PetscInt    q;                /* stage-order of the method */
  PetscInt    r;                /* number of items carried between stages */
  PetscInt    s;                /* number of stages */
  PetscScalar *c;               /* location of the stages */
  PetscScalar *a,*b,*u,*v;      /* tableau for the method */

  /* For use in rescale & modify */
  PetscScalar *alpha;             /* X_n(t_n) - X_{n-1}(t_n) = - alpha^T h^{p+1} x^{(p+1)}(t_n)        */
  PetscScalar *beta;              /*                 - beta^T h^{p+2} x^{(p+2)}(t_n)                   */
  PetscScalar *gamma;             /*                 - gamma^T h^{p+2} f' x^{(p+1)}(t_n)  + O(h^{p+3}) */

  /* Error estimates */
  /* h^{p+1}x^{(p+1)}(t_n)     ~= phi[0]*h*Ydot + psi[0]*X[1:] */
  /* h^{p+2}x^{(p+2)}(t_n)     ~= phi[1]*h*Ydot + psi[1]*X[1:] */
  /* h^{p+2}f' x^{(p+1)}(t_n)  ~= phi[2]*h*Ydot + psi[2]*X[1:] */
  PetscScalar *phi;             /* dim=[3][s] for estimating higher moments, see B,J,W 2007 */
  PetscScalar *psi;             /* dim=[3][r-1], [0 psi^T] of B,J,W 2007 */
  PetscScalar *stage_error;

  /* Desirable properties which enable extra optimizations */
  PetscBool stiffly_accurate;   /* Last row of [A U] is equal t first row of [B V]? */
  PetscBool fsal;               /* First Same As Last: X[1] = h*Ydot[s-1] (and stiffly accurate) */
};

typedef struct TS_GLLE {
  TSGLLEAcceptFunction Accept;    /* Decides whether to accept a given time step, given estimates of local truncation error */
  TSGLLEAdapt          adapt;

  /* These names are only stored so that they can be printed in TSView_GLLE() without making these schemes full-blown
   objects (the implementations I'm thinking of do not have state and I'm lazy). */
  char accept_name[256];

  /* specific to the family of GL method */
  PetscErrorCode (*EstimateHigherMoments)(TSGLLEScheme,PetscReal,Vec*,Vec*,Vec*); /* Provide local error estimates */
  PetscErrorCode (*CompleteStep)(TSGLLEScheme,PetscReal,TSGLLEScheme,PetscReal,Vec*,Vec*,Vec*);
  PetscErrorCode (*Destroy)(struct TS_GLLE*);
  PetscErrorCode (*View)(struct TS_GLLE*,PetscViewer);
  char       type_name[256];
  PetscInt   nschemes;
  TSGLLEScheme *schemes;

  Vec       *X;                 /* Items to carry between steps */
  Vec       *Xold;              /* Values of these items at the last step */
  Vec       W;                  /* = 1/(atol+rtol*|X0|), used for WRMS norm */
  Vec       *himom;             /* len=3, Estimates of h^{p+1}x^{(p+1)}, h^{p+2}x^{(p+2)}, h^{p+2}(df/dx) x^{(p+1)} */
  PetscReal wrms_atol,wrms_rtol;

  /* Stages (Y,Ydot) are computed sequentially */
  Vec       *Ydot;              /* Derivatives of stage vectors, must be stored */
  Vec       Y;                  /* Stage vector, only used while solving the stage so we don't need to store it */
  Vec       Z;                  /* Affine vector */
  PetscReal scoeff;             /* Ydot = Z + shift*Y; shift = scoeff/ts->time_step */
  PetscReal stage_time;         /* time at current stage */
  PetscInt  stage;              /* index of the stage we are currently solving for */

  /* Runtime options */
  PetscInt           current_scheme;
  PetscInt           max_order,min_order,start_order;
  PetscBool          extrapolate;   /* use extrapolation to produce initial Newton iterate? */
  TSGLLEErrorDirection error_direction; /* TSGLLEERROR_FORWARD or TSGLLEERROR_BACKWARD */

  PetscInt max_step_rejections;

  PetscBool setupcalled;
  void      *data;
} TS_GLLE;

#endif
