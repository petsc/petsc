#if !defined(__PETSCGL_H)
#define __PETSCGL_H

#include "private/tsimpl.h"

typedef enum {TSGLERROR_FORWARD,TSGLERROR_BACKWARD} TSGLErrorDirection;

typedef struct _TSGLScheme *TSGLScheme;
struct _TSGLScheme {
  PetscInt p;                   /* order of the method */
  PetscInt q;                   /* stage-order of the method */
  PetscInt r;                   /* number of items carried between stages */
  PetscInt s;                   /* number of stages */
  PetscReal *c;                 /* location of the stages */
  PetscReal *a,*b,*u,*v;        /* tableau for the method */

  /* For use in rescale & modify */
  PetscScalar *alpha;             /* X_n(t_n) - X_{n-1}(t_n) = - alpha^T h^{p+1} x^{(p+1)}(t_n)        */
  PetscScalar *beta;              /*                 - beta^T h^{p+2} x^{(p+2)}(t_n)                   */
  PetscScalar *gamma;             /*                 - gamma^T h^{p+2} f' x^{(p+1)}(t_n)  + O(h^{p+3}) */

  /* Error estimates */
  /* h^{p+1}x^{(p+1)}(t_n)     ~= phi[0]*h*Ydot + psi[0]*X[1:] */
  /* h^{p+2}x^{(p+2)}(t_n)     ~= phi[1]*h*Ydot + psi[1]*X[1:] */
  /* h^{p+2}f' x^{(p+1)}(t_n)  ~= phi[2]*h*Ydot + psi[2]*X[1:] */
  PetscScalar *phi;             /* dim=[3][r+s], in [[phi] [0] [psi]] in B,J,W 2007 */
  PetscScalar *stage_error;

  /* Desirable properties which enable extra optimizations */
  PetscTruth stiffly_accurate;  /* Last row of [A U] is equal t first row of [B V]? */
  PetscTruth fsal;              /* First Same As Last: X[1] = h*Ydot[s-1] (and stiffly accurate) */
};

typedef struct TS_GL {
  /* Control scheme */
  PetscErrorCode (*CtrlAcceptanceTest)(TS,PetscReal,PetscReal,const PetscReal[],PetscTruth*,void*);
  PetscErrorCode (*CtrlChooseNextScheme)(TS,PetscInt,const TSGLScheme[],PetscInt,PetscReal,const PetscReal[],PetscReal,PetscInt*,PetscReal*,void*);
  PetscErrorCode (*CtrlDestroy)(TS,void*);
  void *ctrlP;
  /* specific to the family of GL method */
  PetscErrorCode (*EstimateHigherMoments)(TSGLScheme,PetscReal,Vec*,Vec*,Vec*); /* Provide local error estimates */
  PetscErrorCode (*RescaleAndModify)(TSGLScheme,PetscReal,TSGLScheme,PetscReal,Vec*,Vec*,Vec*,Vec*);
  PetscErrorCode (*Destroy)(struct TS_GL*);
  PetscErrorCode (*View)(struct TS_GL*,PetscViewer);
  char     type_name[256];
  PetscInt nschemes;
  TSGLScheme *schemes;

  Vec *X;                       /* Items to carry between steps */
  Vec *Xold;                    /* Values of these items at the last step */
  Vec W;                        /* = 1/(atol+rtol*|X0|), used for WRMS norm */
  Vec *himom;                   /* len=3, Estimates of h^{p+1}x^{(p+1)}, h^{p+2}x^{(p+2)}, h^{p+2}(df/dx) x^{(p+1)} */
  PetscReal wrms_atol,wrms_rtol;

  /* Stages (Y,Ydot) are computed sequentially */
  Vec *Ydot;                    /* Derivatives of stage vectors, must be stored */
  Vec Y;                        /* Stage vector, only used while solving the stage so we don't need to store it */
  Vec Z;                        /* Affine vector */
  PetscReal shift;              /* Ydot = Z + shift*Y */
  PetscReal base_time;          /* physical time at start of current step */
  PetscInt  stage;              /* index of the stage we are currently solving for */

  /* Runtime options */
  PetscInt current_scheme;
  PetscInt max_order,min_order,start_order;
  PetscTruth extrapolate;           /* use extrapolation to produce initial Newton iterate? */
  TSGLErrorDirection error_direction; /* TSGLERROR_FORWARD or TSGLERROR_BACKWARD */

  void *data;
} TS_GL;

#endif
