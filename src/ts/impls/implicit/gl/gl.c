#define PETSCTS_DLL

/*
  Code for timestepping with diagonally implicit general linear methods

  Notes:
  This integrator can be applied to DAE.

  DIGL methods are a generalization of DIRKs.

  A  |  U
  -------
  B  |  V

  "Diagonally implicit" means that A is lower triangular.

  The method carries a multivector X = {x_1,x_2,...,x_r} between steps, x_1 is the solution.

  We solve the stages (Y,Y') sequentially:

      y_i = h sum_{j=1}^s (a_ij y'_j) + sum_{j=1}^r u_ij x_j,    i=1,...,s

  and then construct the pieces to carry to the next step

      xx_i = h sum_{j=1}^s b_ij y'_j  + sum_{j=1}^r v_ij x_j,    i=1,...,r

  Note that when the equations are cast in implicit form, we are using the stage equation to define y'_i
  in terms of y_i and known stuff (y_j for j<i and x_j for all j)


* Error estimation for step-size adaptivity

  GL methods admit a forward-looking local error estimator (can be evaluated before building X_{n+1})

      h^{p+1} x^{(p+1)}(t_n+h) \approx h \phi^T Y' + [0 \psi^T] X_n + \bigO(h^{p+2})

  and a backward-looking estimator (uses X_{n+1})

      h^{p+1} x^{(p+1)}(t_n+h) \approx h \tilde{\phi}^T Y' + [0 \tilde{\psi}^T] X_{n+1} + \bigO(h^{p+2})

*/

#include "gl.h"                /*I   "petscts.h"   I*/

static const char *TSGLErrorDirections[] = {"FORWARD","BACKWARD","TSGLErrorDirection","TSGLERROR_",0};
static PetscFList TSGLList = 0;

#undef __FUNCT__  
#define __FUNCT__ "TSGLSchemeCreate"
static PetscErrorCode TSGLSchemeCreate(PetscInt p,PetscInt q,PetscInt r,PetscInt s,const PetscReal *c,
                                       const PetscReal *a,const PetscReal *b,const PetscReal *u,const PetscReal *v,
                                       const PetscReal *error1f,const PetscReal *error1b,const PetscReal *error2f,const PetscReal *error2b,
                                       TSGLScheme *inscheme)
{
  TSGLScheme     scheme;
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (p < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Scheme order must be positive");
  if (r < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"At least one item must be carried between steps");
  if (s < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"At least one stage is required");
  PetscValidPointer(inscheme,4);
  *inscheme = 0;
  ierr = PetscMalloc(sizeof(struct _TSGLScheme),&scheme);CHKERRQ(ierr);
  scheme->p = p;
  scheme->q = q;
  scheme->r = r;
  scheme->s  = s;

  ierr = PetscMalloc5(s,PetscReal,&scheme->c,s*s,PetscReal,&scheme->a,r*s,PetscReal,&scheme->b,r*s,PetscReal,&scheme->u,r*r,PetscReal,&scheme->v);CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->c,c,s*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->a,a,s*s*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->b,b,r*s*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->u,u,s*r*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->v,v,r*r*sizeof(PetscReal));CHKERRQ(ierr);

  ierr = PetscMalloc4(r+s,PetscReal,&scheme->error1f,r+s,PetscReal,&scheme->error1b,r+s,PetscReal,&scheme->error2f,r+s,PetscReal,&scheme->error2b);CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->error1f,error1f,(r+s)*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->error1b,error1b,(r+s)*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->error2f,error2f,(r+s)*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->error2b,error2b,(r+s)*sizeof(PetscReal));CHKERRQ(ierr);

  scheme->stiffly_accurate = PETSC_TRUE;
  scheme->fsal = PETSC_TRUE;
  if (scheme->c[s-1] != 1.) {
    scheme->stiffly_accurate = PETSC_FALSE;
    scheme->fsal = PETSC_FALSE;
  }
  for (j=0; j<s; j++) {
    if (a[(s-1)*s+j] != b[j]) scheme->stiffly_accurate = PETSC_FALSE;
    if (r>1 && b[1*s+j] != (j<s-1)?0:1) scheme->fsal = PETSC_FALSE;
  }
  for (j=0; j<r; j++) {
    if (u[(s-1)*r+j] != v[j]) scheme->stiffly_accurate = PETSC_FALSE;
    if (r>1 && v[1*r+j] != 0) scheme->fsal = PETSC_FALSE;
  }

  *inscheme = scheme;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLSchemeDestroy"
static PetscErrorCode TSGLSchemeDestroy(TSGLScheme sc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree5(sc->c,sc->a,sc->b,sc->u,sc->v);CHKERRQ(ierr);
  ierr = PetscFree4(sc->error1f,sc->error1b,sc->error2f,sc->error2b);CHKERRQ(ierr);
  ierr = PetscFree(sc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLDestroy_Default"
static PetscErrorCode TSGLDestroy_Default(TS_GL *gl)
{
  PetscErrorCode ierr;
  PetscInt i;

  PetscFunctionBegin;
  gl->schemes++;                /* return to 0-indexed array */
  for (i=0; i<gl->nschemes; i++) {
    if (gl->schemes[i]) {ierr = TSGLSchemeDestroy(gl->schemes[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(gl->schemes);CHKERRQ(ierr);
  gl->schemes = 0;
  gl->nschemes = 0;
  ierr = PetscMemzero(gl->type_name,sizeof(gl->type_name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLSchemeView"
static PetscErrorCode TSGLSchemeView(TSGLScheme sc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"GL scheme p,q,r,s = %d,%d,%d,%d\n",sc->p,sc->q,sc->r,sc->s);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Stiffly accurate: %s\n",sc->stiffly_accurate?"yes":"no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"First same as last (FSAL): %s\n",sc->fsal?"yes":"no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Abscissas c = [");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<sc->s; i++) {
      ierr = PetscViewerASCIIPrintf(viewer," %8g",sc->c[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"]\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for TS_GL",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLCreate_DI"
static PetscErrorCode TSGLCreate_DI(TS ts)
{
  TS_GL *gl = (TS_GL*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  gl->Destroy = TSGLDestroy_Default;
  gl->nschemes = 2;
  ierr = PetscMalloc(gl->nschemes*sizeof(TSGLScheme),&gl->schemes);CHKERRQ(ierr);
  gl->schemes--;               /* Index schemes by their order, 1-based */

  if (0) {
    /* Implicit Euler */
    const PetscReal c[1]={1},a[1][1]={{1}},b[1][1]={{1}},u[1][1]={{1}},v[1][1]={{1}};
    const PetscReal error1f[2]={0,0},error1b[2]={0,0},error2f[2]={0,0},error2b[2]={0,0};
    ierr = TSGLSchemeCreate(1,1,1,1,c,*a,*b,*u,*v,error1f,error1b,error2f,error2b,&gl->schemes[1]);CHKERRQ(ierr);
  } else {
    /* p=q=1, r=s=2, A- and L-stable with error estimates of order 2 and 3
    * Listed in Butcher & Podhaisky 2006. On error estimation in general linear methods for stiff ODE.
    * irks(0.3,0,[.3,1],[1],1)
    */
    const PetscReal
      c[2]    = {3./10., 1.},
      a[2][2] = {{3./10., 0}, {7./10., 3./10.}},
      b[2][2] = {{7./10., 3./10.}, {0,1}},
      u[2][2] = {{1,0},{1,0}},
      v[2][2] = {{1,0},{0,0}},
      fphi[2] = {0,0},
      fpsi[2] = {0,0},
      bphi[2] = {0,0},
      bpsi[2] = {0,0};
    ierr = TSGLSchemeCreate(1,1,2,2,c,*a,*b,*u,*v,fphi,fpsi,bphi,bpsi,&gl->schemes[1]);CHKERRQ(ierr);
  }

  if (0) {
    /* p=q=2, r=s=3 */
    /* http://www.math.auckland.ac.nz/~hpod/atlas/i2a.html */
    const PetscReal c[3] = {1./3., 2./3., 1};
    const PetscReal a[3][3] = {{4./9.    ,0        , 0        },
                               {1.03752  ,4./9.    , 0        },
                               {0.767025 ,-0.38114 , 4./9.    }};
    const PetscReal b[3][3] = {{0.767025 ,-0.38114 , 0.444444 },
                               {0        ,0        , 1        },
                               {-1.03752 ,0.310864 , 0.638599 }};
    const PetscReal u[3][3] = {{1        ,-0.111111,-0.185185 },
                               {1        ,-0.815284,-0.839819 },
                               {1        ,0.169671 , 0.107948 }};
    const PetscReal v[3][3] = {{1        ,0.169671 , 0.107948 },
                               {0        ,0        , 0        },
                               {0        ,0.0880614, 0        }};
    ierr = TSGLSchemeCreate(2,2,3,3,c,*a,*b,*u,*v,0,0,0,0,&gl->schemes[2]);CHKERRQ(ierr);
  } else {
    /* p=q=2, r=3, s=2, A- and L-stable
    * Butcher & Jackiewicz 2003.
    * A new approach to error estimation for general linear methods.  Example 2.
    * I have not seen an estimator of h^{p+2} x^{(p+2)} for this method */
    const PetscReal c[2] = {0,1};
    const PetscReal a[2][2] = {{ 26./25.   , 0         },
                               { 69./175.  , 26./25.   }};
    const PetscReal b[3][2] = {{ 788./675. , 364./675. },
                               {-503./675. , 728./675. },
                               {-1178./675., 728./675. }};
    const PetscReal u[2][3] = {{ 1         ,-26./25.   , 0         },
                               { 1         ,-76./175.  ,-27./50.   }};
    const PetscReal v[3][3] = {{ 1         ,-53./75.   ,-53./1350. },
                               { 0         , 2./3.     ,-53./675.  },
                               { 0         , 2./3.     ,-53./675.  }};
    const PetscReal
      ferror1[5] = {-2,1,0,1,-2},
      berror1[5] = {1,-278./53.,0,225./53.,1},
      ferror2[5] = {0,0,0,0,0},
      berror2[5] = {0,0,0,0,0};
    ierr = TSGLSchemeCreate(2,2,3,2,c,*a,*b,*u,*v,ferror1,berror1,ferror2,berror2,&gl->schemes[2]);CHKERRQ(ierr);
  }
#if 0
  {
    const PetscReal c[4] = {1./3., 2./3., 1, 1};
    const PetscReal a[4][4] = {{9./40.   ,0        , 0        , 0     },
                               {0.37915  ,9./40.   , 0        , 0     },
                               {0.576331 ,0.157279 , 9./40.   , 0     },
                               {0.740628 ,0.864647 ,-0.404278 , 9./40.}};
    const PetscReal b[4][4] = {{0.740628 ,0.864647 ,-0.404278 , 0./40.},
                               {0        ,0        ,0         , 1     },
                               {-0.715262,-0.740829,0.350709  ,0.892767},
                               {-0.570461,-0.893763,0.381671  ,0.404828}};
    const PetscReal u[4][4] = {{1        ,-0.111111,-0.185185 },
                               {1        ,-0.815284,-0.839819 },
                               {1        ,0.169671 , 0.107948 }};
    const PetscReal v[4][4] = {{1        ,0.169671 , 0.107948 },
                               {0        ,0        , 0        },
                               {0        ,0.0880614, 0        }};
    ierr = TSGLSchemeCreate(3,3,4,4,c,*a,*b,*u,*v,ferror1,berror1,ferror2,berror2,&gl->schemes[3]);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLSetType"
PetscErrorCode PETSCTS_DLLEXPORT TSGLSetType(TS ts,const TSGLType type)
{
  PetscErrorCode ierr,(*r)(TS,const TSGLType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSGLSetType_C",(void(**)(void))&r);CHKERRQ(ierr);
  if (r) {
    ierr = (*r)(ts,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLSetType_GL"
PetscErrorCode PETSCTS_DLLEXPORT TSGLSetType_GL(TS ts,const TSGLType type)
{
  PetscErrorCode ierr,(*r)(TS);
  PetscTruth same;
  TS_GL *gl = (TS_GL*)ts->data;

  PetscFunctionBegin;
  if (gl->type_name[0]) {
    ierr = PetscStrcmp(gl->type_name,type,&same);CHKERRQ(ierr);
    if (same) PetscFunctionReturn(0);
    ierr = (*gl->Destroy)(gl);CHKERRQ(ierr);
  }

  ierr = PetscFListFind(TSGLList,((PetscObject)ts)->comm,type,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TS_GL type %s given",type);
  ierr = (*r)(ts);CHKERRQ(ierr);
  ierr = PetscStrcpy(gl->type_name,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLGetMaxSizes"
static PetscErrorCode TSGLGetMaxSizes(TS ts,PetscInt *max_r,PetscInt *max_s)
{
  TS_GL *gl = (TS_GL*)ts->data;

  PetscFunctionBegin;
  *max_r = gl->schemes[gl->nschemes]->r;
  *max_s = gl->schemes[gl->nschemes]->s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSStep_GL"
static PetscErrorCode TSStep_GL(TS ts,PetscInt *steps,PetscReal *ptime)
{
  PetscErrorCode ierr;
  PetscInt       i,k,max_steps = ts->max_steps,its,lits,max_r,max_s;
  TS_GL          *gl = (TS_GL*)ts->data;

  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

  ierr = TSGLGetMaxSizes(ts,&max_r,&max_s);CHKERRQ(ierr);
  ierr = VecCopy(ts->vec_sol,gl->X[0]);CHKERRQ(ierr);
  for (i=1; i<max_r; i++) {
    ierr = VecZeroEntries(gl->X[i]);CHKERRQ(ierr);
  }
  ierr = VecView(gl->X[0],0);CHKERRQ(ierr);

  for (k=0; k<max_steps; k++) {
    PetscInt j,r,s;
    PetscReal h;
    const PetscReal *c,*a,*b,*u,*v;
    Vec *X,*Ydot,Y;
    TSGLScheme scheme = gl->schemes[gl->current_order];

    r = scheme->r; s = scheme->s;
    c = scheme->c;
    a = scheme->a; u = scheme->u;
    b = scheme->b; v = scheme->v;
    h = ts->time_step;
    X = gl->X; Ydot = gl->Ydot; Y = gl->Y;

    if (ts->ptime + ts->time_step > ts->max_time) break;
    gl->base_time = ts->ptime;  /* save time at the start of this step */

    for (i=0; i<s; i++) {
      PetscReal shift = gl->shift = 1./(h*a[i*s+i]);
      gl->stage = i;
      ts->ptime = gl->base_time + c[i]*h;

      /*
      * Stage equation: Y = h A Y' + U X
      * We assume that A is lower-triangular so that we can solve the stages (Y,Y') sequentially
      * Build the affine vector z_i = -[1/(h a_ii)](h sum_j a_ij y'_j + sum_j u_ij x_j)
      * Then y'_i = z + 1/(h a_ii) y_i
      */
      ierr = VecZeroEntries(gl->Z);CHKERRQ(ierr);
      for (j=0; j<r; j++) {
        ierr = VecAXPY(gl->Z,-shift*u[i*r+j],X[j]);CHKERRQ(ierr);
      }
      for (j=0; j<i; j++) {
        ierr = VecAXPY(gl->Z,-shift*h*a[i*s+j],Ydot[j]);CHKERRQ(ierr);
      }
      /* Note: Z is used within function evaluation, Ydot = Z + shift*Y */

      /* Compute an estimate of Y to start Newton iteration */
      if (gl->extrapolate) {
        if (i==0) {
          /* Linear interpolation on the first stage */
          ierr = VecWAXPY(Y,c[i]*h,X[1],X[0]);CHKERRQ(ierr);
        } else {
          /* Linear interpolation from the last stage */
          ierr = VecAXPY(Y,(c[i]-c[i-1])*h,Ydot[i-1]);
        }
      } else if (i==0) {        /* Directly use solution from the last step, otherwise reuse the last stage (do nothing) */
        ierr = VecCopy(X[0],Y);CHKERRQ(ierr);
      }

      /* Solve this stage (Ydot[i] is computed during function evaluation) */
      ierr = SNESSolve(ts->snes,PETSC_NULL,Y);CHKERRQ(ierr);
      ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
      ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
      ts->nonlinear_its += its; ts->linear_its += lits;
    }

    X = gl->Xold;
    gl->Xold = gl->X;
    gl->X = X;

    if (gl->error_direction == TSGLERROR_FORWARD) {
      /* todo: build error and decide whether to accept step */
    }

    /* Build the new solution from (X,Ydot) */
    for (i=0; i<r; i++) {
      ierr = VecZeroEntries(X[i]);CHKERRQ(ierr);
      for (j=0; j<s; j++) {
        ierr = VecAXPY(X[i],h*b[i*s+j],Ydot[j]);CHKERRQ(ierr);
      }
      for (j=0; j<r; j++) {
        ierr = VecAXPY(X[i],v[i*r+j],gl->Xold[j]);CHKERRQ(ierr);
      }
    }

    if (gl->error_direction == TSGLERROR_BACKWARD) {
      /* todo: build error and decide whether to accept step */
    }

    /* Post the solution for the user, we could avoid this copy with a small bit of cleverness */
    ierr = VecCopy(gl->X[0],ts->vec_sol);CHKERRQ(ierr);

    ts->ptime = gl->base_time + h;
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

    /* todo: use error estimates to change step size and method order */
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_GL"
static PetscErrorCode TSDestroy_GL(TS ts)
{
  TS_GL          *gl = (TS_GL*)ts->data;
  PetscInt        max_r,max_s;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSGLGetMaxSizes(ts,&max_r,&max_s);CHKERRQ(ierr);
  ierr = VecDestroyVecs(gl->Xold,max_r);CHKERRQ(ierr);
  ierr = VecDestroyVecs(gl->X,max_r);CHKERRQ(ierr);
  ierr = VecDestroyVecs(gl->Ydot,max_s);CHKERRQ(ierr);
  ierr = VecDestroy(gl->Y);CHKERRQ(ierr);
  ierr = VecDestroy(gl->Z);CHKERRQ(ierr);
  ierr = (*gl->Destroy)(gl);CHKERRQ(ierr);
  ierr = PetscFree(gl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    This defines the nonlinear equation that is to be solved with SNES
    g(x) = f(t,x,z+shift*x) = 0
*/
#undef __FUNCT__  
#define __FUNCT__ "TSGLFunction"
static PetscErrorCode TSGLFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  TS              ts = (TS)ctx;
  TS_GL          *gl = (TS_GL*)ts->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecWAXPY(gl->Ydot[gl->stage],gl->shift,x,gl->Z);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts,ts->ptime,x,gl->Ydot[gl->stage],f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLJacobian"
static PetscErrorCode TSGLJacobian(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *str,void *ctx)
{
  TS              ts = (TS)ctx;
  TS_GL          *gl = (TS_GL*)ts->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* gl->Xdot will have already been computed in TSGLFunction */
  ierr = TSComputeIJacobian(ts,ts->ptime,x,gl->Ydot[gl->stage],gl->shift,A,B,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_GL"
static PetscErrorCode TSSetUp_GL(TS ts)
{
  TS_GL          *gl = (TS_GL*)ts->data;
  Vec             res;
  PetscInt        max_r,max_s;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSGLGetMaxSizes(ts,&max_r,&max_s);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,max_r,&gl->X);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,max_r,&gl->Xold);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,max_s,&gl->Ydot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&gl->Y);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&gl->Z);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&res);CHKERRQ(ierr);
  ierr = SNESSetFunction(ts->snes,res,&TSGLFunction,ts);CHKERRQ(ierr);
  ierr = VecDestroy(res);CHKERRQ(ierr); /* Give ownership to SNES */
  /* This is nasty.  SNESSetFromOptions() is usually called in TSSetFromOptions().  With -snes_mf_operator, it will
  replace A and we don't want to mess with that.  With -snes_mf, A and B will be replaced as well as the function and
  context.  Note that SNESSetFunction() normally has not been called before SNESSetFromOptions(), so when -snes_mf sets
  the Jacobian user context to snes->funP, it will actually be NULL.  This is not a problem because both snes->funP and
  snes->jacP should be the TS. */
  {
    Mat A,B;
    PetscErrorCode (*func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
    void *ctx;
    ierr = SNESGetJacobian(ts->snes,&A,&B,&func,&ctx);CHKERRQ(ierr);
    ierr = SNESSetJacobian(ts->snes,A?A:ts->A,B?B:ts->B,func?func:&TSGLJacobian,ctx?ctx:ts);CHKERRQ(ierr);
  }

  if (!gl->current_order) gl->current_order = gl->min_order;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_GL"
static PetscErrorCode TSSetFromOptions_GL(TS ts)
{
  TS_GL *gl = (TS_GL*)ts->data;
  char tname[256] = TSGL_DI;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("General Linear ODE solver options");CHKERRQ(ierr);
  {
    PetscTruth flg;
    ierr = PetscOptionsList("-ts_gl_type","Type of GL method","TSGLSetType",TSGLList,gl->type_name[0]?gl->type_name:tname,tname,sizeof(tname),&flg);CHKERRQ(ierr);
    if (flg || !gl->type_name[0]) {
      ierr = TSGLSetType(ts,tname);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-ts_gl_max_order","Maximum order to try","TSGLSetMaxOrder",gl->max_order,&gl->max_order,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_gl_min_order","Minimum order to try","TSGLSetMinOrder",gl->min_order,&gl->min_order,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-ts_gl_error_direction","Which direction to look when estimating error","TSGLSetErrorDirection",TSGLErrorDirections,(PetscEnum)gl->error_direction,(PetscEnum*)&gl->error_direction,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ts_gl_extrapolate","Extrapolate stage solution from previous solution (sometimes unstable)","TSGLSetExtrapolate",gl->extrapolate,&gl->extrapolate,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_GL"
static PetscErrorCode TSView_GL(TS ts,PetscViewer viewer)
{
  TS_GL          *gl = (TS_GL*)ts->data;
  PetscInt        i;
  PetscTruth      iascii;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  min order %D, max order %D, current order %D\n",gl->min_order,gl->max_order,gl->current_order);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Error estimation: %s\n",TSGLErrorDirections[gl->error_direction]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Extrapolation: %s\n",gl->extrapolate?"yes":"no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",gl->type_name[0]?gl->type_name:"(not yet set)");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Schemes within family (%d):\n",gl->nschemes);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i=1; i<=gl->nschemes; i++) {
      ierr = TSGLSchemeView(gl->schemes[i],viewer);CHKERRQ(ierr);
    }
    if (gl->View) {
      ierr = (*gl->View)(gl,viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for TS_GL",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}


#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define TSGLRegisterDynamic(a,b,c,d) TSGLRegister(a,b,c,0)
#else
#define TSGLRegisterDynamic(a,b,c,d) TSGLRegister(a,b,c,d)
#endif

#undef __FUNCT__  
#define __FUNCT__ "TSGLRegister"
PetscErrorCode PETSCTS_DLLEXPORT TSGLRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(TS))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&TSGLList,sname,fullname,(void(*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLRegisterAll"
static PetscErrorCode PETSCTS_DLLEXPORT TSGLRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGLRegisterDynamic(TSGL_DI,path,"TSGLCreate_DI",TSGLCreate_DI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSGLInitializePackage"
static PetscErrorCode PETSCTS_DLLEXPORT TSGLInitializePackage(const char path[])
{
  static PetscTruth TSGLPackageInitialized = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSGLPackageInitialized) PetscFunctionReturn(0);
  TSGLPackageInitialized = PETSC_TRUE;
  ierr = TSGLRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TS_GL - DAE solver using implicit General Linear methods

  These methods contain Runge-Kutta and multistep schemes as special cases.  These special cases have some fundamental
  limitations.  For example, diagonally implicit Runge-Kutta cannot have stage order greater than 1 which limits their
  applicability to very stiff systems.  Meanwhile, multistep methods cannot be A-stable for order greater than 2 and BDF
  are not 0-stable for order greater than 6.  GL methods can be A- and L-stable with arbitrarily high stage order and
  reliable error estimates for both 1 and 2 orders higher to facilitate adaptive step sizes and adaptive order schemes.
  All this is possible while preserving a singly diagonally implicit structure.

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_GL"
PetscErrorCode PETSCTS_DLLEXPORT TSCreate_GL(TS ts)
{
  TS_GL       *gl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = TSGLInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscNewLog(ts,TS_GL,&gl);CHKERRQ(ierr);
  ts->data = (void*)gl;

  ts->ops->destroy        = TSDestroy_GL;
  ts->ops->view           = TSView_GL;
  ts->ops->setup          = TSSetUp_GL;
  ts->ops->step           = TSStep_GL;
  ts->ops->setfromoptions = TSSetFromOptions_GL;

  ierr = SNESCreate(((PetscObject)ts)->comm,&ts->snes);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ts->snes,(PetscObject)ts,1);CHKERRQ(ierr);

  gl->min_order = 1;
  gl->max_order = 1;
  gl->extrapolate = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSGLSetType_C","TSGLSetType_GL",&TSGLSetType_GL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
