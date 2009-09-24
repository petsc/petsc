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
  PetscReal *error1f;           /* forward-looking  estimation of h^{p+1}x^{(p+1)} */
  PetscReal *error1b;           /* backward-looking estimation of h^{p+1}x^{(p+1)} */
  PetscReal *error2f;           /* forward-looking  estimation of h^{p+2}x^{(p+2)} */
  PetscReal *error2b;           /* backward-looking estimation of h^{p+2}x^{(p+2)} */
  PetscTruth stiffly_accurate;  /* Last row of [A U] is equal t first row of [B V]? */
  PetscTruth fsal;              /* First Same As Last: X[1] = h*Ydot[s-1] */
};

struct _p_TSGL {
  PetscErrorCode (*EstimateError)(TSGLScheme,Vec*,Vec*,PetscReal*,PetscReal*);         /* Provide local error estimates */
  PetscErrorCode (*ChangeScheme)(TSGLScheme,TSGLScheme,PetscReal,PetscReal,Vec*,Vec*); /* Change step size and possibly order of scheme (within family) */
  PetscErrorCode (*Destroy)(TSGL);
  PetscErrorCode (*View)(TSGL,PetscViewer);
  PetscInt max_order;
  TSGLScheme *schemes;
};

typedef struct {
  char     type_name[256];
  TSGL     data;
  PetscInt current_order;

  Vec *X;                       /* Items to carry between steps */
  Vec *Xold;                    /* Values of these items at the last step */

  /* Stages (Y,Ydot) are computed sequentially */
  Vec *Ydot;                    /* Derivatives of stage vectors, must be stored */
  Vec Y;                        /* Stage vector, only used while solving the stage so we don't need to store it */
  Vec Z;                        /* Affine vector */
  PetscReal shift;              /* Ydot = Z + shift*Y */
  PetscReal base_time;          /* physical time at start of current step */
  PetscInt  stage;              /* index of the stage we are currently solving for */

  /* Runtime options */
  PetscInt max_order,min_order;
  PetscTruth extrapolate;           /* use extrapolation to produce initial Newton iterate? */
  TSGLErrorDirection error_direction; /* TSGLERROR_FORWARD or TSGLERROR_BACKWARD */
} TS_GL;

#endif
