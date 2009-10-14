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
#include "petscblaslapack.h"

static const char *TSGLErrorDirections[] = {"FORWARD","BACKWARD","TSGLErrorDirection","TSGLERROR_",0};
static PetscFList TSGLList = 0;

/* This function is pure */
static PetscScalar Factorial(PetscInt n)
{
  PetscInt i;
  if (n < 12) {                 /* Can compute with 32-bit integers */
    PetscInt f = 1;
    for (i=2; i<=n; i++) f *= i;
    return (PetscScalar)f;
  } else {
    PetscScalar f = 1.;
    for (i=2; i<=n; i++) f *= (PetscScalar)i;
    return f;
  }
}

/* This function is pure */
static PetscScalar CPowF(PetscScalar c,PetscInt p)
{
  return PetscPowScalar(c,p)/Factorial(p);
}


#undef __FUNCT__  
#define __FUNCT__ "TSGLSchemeCreate"
static PetscErrorCode TSGLSchemeCreate(PetscInt p,PetscInt q,PetscInt r,PetscInt s,const PetscScalar *c,
                                       const PetscScalar *a,const PetscScalar *b,const PetscScalar *u,const PetscScalar *v,TSGLScheme *inscheme)
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
  scheme->p  = p;
  scheme->q  = q;
  scheme->r  = r;
  scheme->s  = s;

  ierr = PetscMalloc5(s,PetscScalar,&scheme->c,s*s,PetscScalar,&scheme->a,r*s,PetscScalar,&scheme->b,r*s,PetscScalar,&scheme->u,r*r,PetscScalar,&scheme->v);CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->c,c,s*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->a,a,s*s*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->b,b,r*s*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->u,u,s*r*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->v,v,r*r*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = PetscMalloc5(r,PetscScalar,&scheme->alpha,r,PetscScalar,&scheme->beta,r,PetscScalar,&scheme->gamma,3*(r+s),PetscScalar,&scheme->phi,r,PetscScalar,&scheme->stage_error);CHKERRQ(ierr);
  {
    PetscInt i,j,k,rs=r+s;
    PetscBLASInt m,n,one=1,*ipiv,lwork=4*((s+3)*3+3),info,rank,ldb;
    PetscScalar *ImV,*H,*work,*sing,rcond;
    PetscScalar *c=scheme->c,*a=scheme->a,*b=scheme->b,*u=scheme->u,*v=scheme->v;
    ierr = PetscMalloc5(PetscSqr(r),PetscScalar,&ImV,3*s,PetscScalar,&H,lwork,PetscScalar,&work,r+s,PetscScalar,&sing,r+s,PetscBLASInt,&ipiv);CHKERRQ(ierr);

    /* column-major input */
    for (i=0; i<r-1; i++) {
      for (j=0; j<r-1; j++) {
        ImV[i+j*r] = 1.0*(i==j) - v[(i+1)*r+j+1];
      }
    }
    /* Build right hand side for alpha (tp - glm.B(2:end,:)*(glm.c.^(p)./factorial(p))) */
    for (i=1; i<r; i++) {
      scheme->alpha[i] = 1./Factorial(p+1-i);
      for (j=0; j<s; j++) scheme->alpha[i] -= b[i*s+j]*CPowF(c[j],p);
    }
    m = PetscBLASIntCast(r-1);
    n = PetscBLASIntCast(r);
    LAPACKgesv_(&m,&one,ImV,&n,ipiv,scheme->alpha+1,&n,&info);
    if (info < 0) SETERRQ(PETSC_ERR_LIB,"Bad argument to GESV");
    if (info > 0) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

    /* Build right hand side for beta (tp1 - glm.B(2:end,:)*(glm.c.^(p+1)./factorial(p+1)) - e.alpha) */
    for (i=1; i<r; i++) {
      scheme->beta[i] = 1./Factorial(p+2-i) - scheme->alpha[i];
      for (j=0; j<s; j++) scheme->beta[i] -= b[i*s+j]*CPowF(c[j],p+1);
    }
    LAPACKgetrs_("No transpose",&m,&one,ImV,&n,ipiv,scheme->beta+1,&n,&info);
    if (info < 0) SETERRQ(PETSC_ERR_LIB,"Bad argument to GETRS");
    if (info > 0) SETERRQ(PETSC_ERR_LIB,"Should not happen");

    /* Build stage_error vector
           xi = glm.c.^(p+1)/factorial(p+1) - glm.A*glm.c.^p/factorial(p) + glm.U(:,2:end)*e.alpha;
    */
    for (i=0; i<s; i++) {
      scheme->stage_error[i] = CPowF(c[i],p+1);
      for (j=0; j<s; j++) scheme->stage_error[i] -= a[i*s+j]*CPowF(c[j],p);
      for (j=1; j<r; j++) scheme->stage_error[i] += u[i*r+j]*scheme->alpha[j];
    }

    /* alpha[0] (epsilon in B,J,W 2007)
           epsilon = 1/factorial(p+1) - B(1,:)*c.^p/factorial(p) + V(1,2:end)*e.alpha;
    */
    scheme->alpha[0] = 1./Factorial(p+1);
    for (j=0; j<s; j++) scheme->alpha[0] -= b[0*s+j]*CPowF(c[j],p);
    for (j=1; j<r; j++) scheme->alpha[0] += v[0*r+j]*scheme->alpha[j];

    /* right hand side for gamma (glm.B(2:end,:)*e.xi - e.epsilon*eye(s-1,1)) */
    for (i=1; i<r; i++) {
      scheme->gamma[i] = -(i==1)*scheme->alpha[0];
      for (j=0; j<s; j++) scheme->gamma[i] += b[i*s+j]*scheme->stage_error[j];
    }
    LAPACKgetrs_("No transpose",&m,&one,ImV,&n,ipiv,scheme->gamma+1,&n,&info);
    if (info < 0) SETERRQ(PETSC_ERR_LIB,"Bad argument to GETRS");
    if (info > 0) SETERRQ(PETSC_ERR_LIB,"Should not happen");

    /* beta[0] (rho in B,J,W 2007)
        e.rho = 1/factorial(p+2) - glm.B(1,:)*glm.c.^(p+1)/factorial(p+1) ...
            + glm.V(1,2:end)*e.beta;% - e.epsilon;
    % Note: The paper (B,J,W 2007) includes the last term in their definition
    * */
    scheme->beta[0] = 1./Factorial(p+2);
    for (j=0; j<s; j++) scheme->beta[0] -= b[0*s+j]*CPowF(c[j],p+1);
    for (j=1; j<r; j++) scheme->beta[0] += v[0*r+j]*scheme->beta[j];

    /* gamma[0] (sigma in B,J,W 2007)
    *   e.sigma = glm.B(1,:)*e.xi + glm.V(1,2:end)*e.gamma;
    * */
    scheme->gamma[0] = 0;
    for (j=0; j<s; j++) scheme->gamma[0] += b[0*s+j]*scheme->stage_error[j];
    for (j=1; j<r; j++) scheme->gamma[0] += v[0*s+j]*scheme->gamma[j];

    /* Assemble H
    *    % Determine the error estimators phi
       H = [[cpow(glm.c,p) + C*e.alpha] [cpow(glm.c,p+1) + C*e.beta] ...
               [e.xi - C*(e.gamma + 0*e.epsilon*eye(s-1,1))]]';
    % Paper has formula above without the 0, but the 0 must be left
    % out to satisfy the conditions they propose and to make the
    % example schemes work
    e.H = H;
    e.phi = (H \ [1 0 0;1 1 0;0 0 -1])';
    e.psi = -e.phi*C;
    * */
    for (j=0; j<s; j++) {
      H[0+j*3] = CPowF(c[j],p);
      H[1+j*3] = CPowF(c[j],p+1);
      H[2+j*3] = scheme->stage_error[j];
      for (k=1; k<r; k++) {
        H[0+j*3] += CPowF(c[j],k-1)*scheme->alpha[k];
        H[1+j*3] += CPowF(c[j],k-1)*scheme->beta[k];
        H[2+j*3] -= CPowF(c[j],k-1)*scheme->gamma[k];
      }
    }
    scheme->phi[0+0*rs] = 1;  scheme->phi[0+1*rs] = 0;  scheme->phi[0+2*rs] = 0;
    scheme->phi[1+0*rs] = 1;  scheme->phi[1+1*rs] = 1;  scheme->phi[1+2*rs] = 0;
    scheme->phi[2+0*rs] = 0;  scheme->phi[2+1*rs] = 0;  scheme->phi[2+2*rs] = -1;
    m = 3;
    n = PetscBLASIntCast(s);
    ldb = PetscBLASIntCast(rs);
    rcond = 1e-12;
    /* DGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, INFO ) */
    LAPACKgelss_(&m,&n,&m,H,&m,scheme->phi,&ldb,sing,&rcond,&rank,work,&lwork,&info);
    if (info < 0) SETERRQ(PETSC_ERR_LIB,"Bad argument to GELSS");
    if (info > 0) SETERRQ(PETSC_ERR_LIB,"SVD failed to converge");

    /* the other part of the error estimator, psi in B,J,W 2007 */
    scheme->phi[0*rs+s+0] = 0;
    scheme->phi[1*rs+s+0] = 0;
    scheme->phi[2*rs+s+0] = 0;
    for (j=1; j<r; j++) {
      scheme->phi[0*rs+s+j] = 0;
      scheme->phi[1*rs+s+j] = 0;
      scheme->phi[2*rs+s+j] = 0;
      for (k=0; k<s; k++) {
        scheme->phi[0*rs+s+j] -= CPowF(c[k],j-1)*scheme->phi[0*rs+k];
        scheme->phi[1*rs+s+j] -= CPowF(c[k],j-1)*scheme->phi[1*rs+k];
        scheme->phi[2*rs+s+j] -= CPowF(c[k],j-1)*scheme->phi[2*rs+k];
      }
    }
    ierr = PetscFree5(ImV,H,work,sing,ipiv);CHKERRQ(ierr);
  }
  /* Check which properties are satisfied */
  scheme->stiffly_accurate = PETSC_TRUE;
  if (scheme->c[s-1] != 1.) scheme->stiffly_accurate = PETSC_FALSE;
  for (j=0; j<s; j++) if (a[(s-1)*s+j] != b[j]) scheme->stiffly_accurate = PETSC_FALSE;
  for (j=0; j<r; j++) if (u[(s-1)*r+j] != v[j]) scheme->stiffly_accurate = PETSC_FALSE;
  scheme->fsal = scheme->stiffly_accurate; /* FSAL is stronger */
  for (j=0; j<s-1; j++) if (r>1 && b[1*s+j] != 0) scheme->fsal = PETSC_FALSE;
  if (b[1*s+r-1] != 1) scheme->fsal = PETSC_FALSE;
  for (j=0; j<r; j++) if (r>1 && v[1*r+j] != 0) scheme->fsal = PETSC_FALSE;

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
  ierr = PetscFree5(sc->alpha,sc->beta,sc->gamma,sc->phi,sc->stage_error);CHKERRQ(ierr);
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
#define __FUNCT__ "ViewTable_Private"
static PetscErrorCode ViewTable_Private(PetscViewer viewer,PetscInt m,PetscInt n,const PetscScalar a[],const char name[])
{
  PetscErrorCode ierr;
  PetscTruth     iascii;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"%30s = [",name);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      if (i) {ierr = PetscViewerASCIIPrintf(viewer,"%30s   [","");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      for (j=0; j<n; j++) {
        ierr = PetscViewerASCIIPrintf(viewer," %12.8g",a[i*n+j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"]\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for TS_GL",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSGLSchemeView"
static PetscErrorCode TSGLSchemeView(TSGLScheme sc,PetscTruth view_details,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"GL scheme p,q,r,s = %d,%d,%d,%d\n",sc->p,sc->q,sc->r,sc->s);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Stiffly accurate: %s,  FSAL: %s\n",sc->stiffly_accurate?"yes":"no",sc->fsal?"yes":"no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Leading error constants: %10.3e  %10.3e  %10.3e\n",sc->alpha[0],sc->beta[0],sc->gamma[0]);CHKERRQ(ierr);
    ierr = ViewTable_Private(viewer,1,sc->s,sc->c,"Abscissas c");CHKERRQ(ierr);
    if (view_details) {
      ierr = ViewTable_Private(viewer,sc->s,sc->s,sc->a,"A");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,sc->r,sc->s,sc->b,"B");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,sc->s,sc->r,sc->u,"U");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,sc->r,sc->r,sc->v,"V");CHKERRQ(ierr);

      ierr = ViewTable_Private(viewer,3,sc->s+sc->r,sc->phi,"Error estimate [phi 0 psi]");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,1,sc->r,sc->alpha,"Modify alpha");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,1,sc->r,sc->beta,"Modify beta");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,1,sc->r,sc->gamma,"Modify gamma");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,1,sc->s,sc->stage_error,"Stage error xi");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for TS_GL",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLEstimateHigherMoments_Default"
static PetscErrorCode TSGLEstimateHigherMoments_Default(TSGLScheme sc,PetscReal h,Vec Ydot[],Vec Xold[],Vec hm[])
{
  PetscErrorCode ierr;
  PetscInt i;

  PetscFunctionBegin;
  /* build error vectors*/
  for (i=0; i<3; i++) {
    ierr = VecZeroEntries(hm[i]);CHKERRQ(ierr);
    ierr = VecMAXPY(hm[i],sc->s,&sc->phi[i*(sc->r+sc->s)],Ydot);CHKERRQ(ierr);
    ierr = VecMAXPY(hm[i],sc->r,&sc->phi[i*(sc->r+sc->s)+sc->s],Xold);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLRescaleAndModify_Default"
static PetscErrorCode TSGLRescaleAndModify_Default(TSGLScheme sc,PetscReal h,TSGLScheme next_sc,PetscReal next_h,Vec Ydot[],Vec Xold[],Vec E[],Vec X[])
{
  PetscErrorCode ierr;
  PetscScalar brow[32],vrow[32];
  PetscInt i,j,r,s,ratio;

  PetscFunctionBegin;
  /* Build the new solution from (X,Ydot) */
  r = sc->r;
  s = sc->s;
  ratio = next_h/h;
  for (i=0; i<r; i++) {
    ierr = VecZeroEntries(X[i]);CHKERRQ(ierr);
    for (j=0; j<s; j++) brow[j] = h*sc->b[i*s+j];
    ierr = VecMAXPY(X[i],s,brow,Ydot);CHKERRQ(ierr);
    for (j=0; j<r; j++) vrow[j] = sc->v[i*r+j];
    ierr = VecMAXPY(X[i],r,vrow,Xold);CHKERRQ(ierr);
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
  gl->Destroy               = TSGLDestroy_Default;
  gl->EstimateHigherMoments = TSGLEstimateHigherMoments_Default;
  gl->RescaleAndModify      = TSGLRescaleAndModify_Default;
  ierr = PetscMalloc(10*sizeof(TSGLScheme),&gl->schemes);CHKERRQ(ierr);
  gl->nschemes = 0;

  {
    /* p=1,q=1, r=s=2, A- and L-stable with error estimates of order 2 and 3
    * Listed in Butcher & Podhaisky 2006. On error estimation in general linear methods for stiff ODE.
    * irks(0.3,0,[.3,1],[1],1)
    * Note: can be made second order by replacing 0.3 with 1-sqrt(1/2)
    */
    const PetscScalar c[2] = {3./10., 1.}
    ,a[2][2] = {{3./10., 0}, {7./10., 3./10.}}
    ,b[2][2] = {{7./10., 3./10.}, {0,1}}
    ,u[2][2] = {{1,0},{1,0}}
    ,v[2][2] = {{1,0},{0,0}};
    ierr = TSGLSchemeCreate(1,1,2,2,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]);CHKERRQ(ierr);
  }
  if (0) {
    /* Implicit Euler */
    const PetscScalar c[1]={1},a[1][1]={{1}},b[1][1]={{1}},u[1][1]={{1}},v[1][1]={{1}};
    ierr = TSGLSchemeCreate(1,1,1,1,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]);CHKERRQ(ierr);
  }

  {
    /* p=q=2, r=s=3: irks(4/9,0,[1:3]/3,[0.33852],1) */
    /* http://www.math.auckland.ac.nz/~hpod/atlas/i2a.html */
    const PetscScalar c[3] = {1./3., 2./3., 1}
    ,a[3][3] = {{4./9.                ,0                      , 0      }
                ,{1.03750643704090e+00 ,                  4./9.,       0}
                ,{7.67024779410304e-01 ,  -3.81140216918943e-01,   4./9.}}
    ,b[3][3] = {{0.767024779410304,  -0.381140216918943,   4./9.},
                {0.000000000000000,  0.000000000000000,   1.000000000000000},
                {-2.075048385225385,   0.621728385225383,   1.277197204924873}}
    ,u[3][3] = {{1.0000000000000000,  -0.1111111111111109,  -0.0925925925925922},
                {1.0000000000000000,  -0.8152842148186744,  -0.4199095530877056},
                {1.0000000000000000,   0.1696709930641948,   0.0539741070314165}}
    ,v[3][3] = {{1.0000000000000000,  0.1696709930641948,   0.0539741070314165},
                {0.000000000000000,   0.000000000000000,   0.000000000000000},
                {0.000000000000000,   0.176122795075129,   0.000000000000000}};
    ierr = TSGLSchemeCreate(2,2,3,3,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]);CHKERRQ(ierr);
  }
  {
    /* p=q=3, r=s=4: irks(9/40,0,[1:4]/4,[0.3312 1.0050],[0.49541 1;1 0]) */
    const PetscScalar c[4] = {0.25,0.5,0.75,1.0}
    ,a[4][4] = {{9./40.               ,                      0,                      0,                      0},
                {2.11286958887701e-01 ,    9./40.             ,                      0,                      0},
                {9.46338294287584e-01 ,  -3.42942861246094e-01,   9./40.              ,                      0},
                {0.521490453970721    ,  -0.662474225622980,   0.490476425459734,   9./40.           }}
    ,b[4][4] = {{0.521490453970721    ,  -0.662474225622980,   0.490476425459734,   9./40.           },
                {0.000000000000000    ,   0.000000000000000,   0.000000000000000,   1.000000000000000},
                {-0.084677029310348   ,   1.390757514776085,  -1.568157386206001,   2.023192696767826},
                {0.465383797936408    ,   1.478273530625148,  -1.930836081010182,   1.644872111193354}}
    ,u[4][4] = {{1.00000000000000000  ,   0.02500000000001035,  -0.02499999999999053,  -0.00442708333332865},
                {1.00000000000000000  ,   0.06371304111232945,  -0.04032173972189845,  -0.01389438413189452},
                {1.00000000000000000  ,  -0.07839543304147778,   0.04738685705116663,   0.02032603595928376},
                {1.00000000000000000  ,   0.42550734619251651,   0.10800718022400080,  -0.01726712647760034}}
    ,v[4][4] = {{1.00000000000000000  ,   0.42550734619251651,   0.10800718022400080,  -0.01726712647760034},
                {0.000000000000000    ,   0.000000000000000,   0.000000000000000,   0.000000000000000},
                {0.000000000000000    ,  -1.761115796027561,  -0.521284157173780,   0.258249384305463},
                {0.000000000000000    ,  -1.657693358744728,  -1.052227765232394,   0.521284157173780}};
    ierr = TSGLSchemeCreate(3,3,4,4,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]);CHKERRQ(ierr);
  }
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
#define __FUNCT__ "TSGLUpdateWRMS"
static PetscErrorCode TSGLUpdateWRMS(TS ts)
{
  TS_GL *gl = (TS_GL*)ts->data;
  PetscErrorCode ierr;
  PetscScalar *x,*w;
  PetscInt n,i;

  PetscFunctionBegin;
  ierr = VecGetArray(gl->X[0],&x);CHKERRQ(ierr);
  ierr = VecGetArray(gl->W,&w);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gl->W,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    w[i] = 1./(gl->wrms_atol + gl->wrms_rtol*PetscAbs(x[i]));
  }
  ierr = VecRestoreArray(gl->X[0],&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(gl->W,&w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGLVecNormWRMS"
static PetscErrorCode TSGLVecNormWRMS(TS ts,Vec X,PetscReal *nrm)
{
  TS_GL *gl = (TS_GL*)ts->data;
  PetscErrorCode ierr;
  PetscScalar *x,*w,sum;
  PetscInt n,i;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(gl->W,&w);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gl->W,&n);CHKERRQ(ierr);
  sum = 0;
  for (i=0; i<n; i++) {
    sum += PetscAbs(PetscSqr(x[i]*w[i]));
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(gl->W,&w);CHKERRQ(ierr);
  *nrm = PetscAbs(PetscSqrtScalar(sum/n));
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
  *max_r = gl->schemes[gl->nschemes-1]->r;
  *max_s = gl->schemes[gl->nschemes-1]->s;
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

  if (0) {
    /* Find consistent initial data for DAE */
    gl->base_time = ts->ptime;
    ts->ptime += ts->time_step;
    gl->shift = 1./ts->time_step;
    gl->stage = 0;
    ierr = VecCopy(ts->vec_sol,gl->Z);CHKERRQ(ierr);
    ierr = VecCopy(ts->vec_sol,gl->Y);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,PETSC_NULL,gl->Y);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;

    ts->ptime = gl->base_time;
  }

  if (gl->current_scheme < 0) SETERRQ(PETSC_ERR_ORDER,"A starting scheme has not been provided");

  for (k=0; k<max_steps; k++) {
    PetscInt j,r,s,next_scheme;
    PetscReal h,hmnorm[4],enorm[3],next_h,tleft;
    PetscTruth accept;
    const PetscScalar *c,*a,*b,*u,*v;
    Vec *X,*Ydot,Y;
    TSGLScheme scheme = gl->schemes[gl->current_scheme];

    r = scheme->r; s = scheme->s;
    c = scheme->c;
    a = scheme->a; u = scheme->u;
    b = scheme->b; v = scheme->v;
    h = ts->time_step;
    X = gl->X; Ydot = gl->Ydot; Y = gl->Y;

    if (ts->ptime + ts->time_step > ts->max_time) break;
    gl->base_time = ts->ptime;  /* save time at the start of this step */

    for (i=0; i<s; i++) {
      PetscScalar shift = gl->shift = 1./(h*a[i*s+i]);
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
          /* Linear extrapolation on the first stage */
          ierr = VecWAXPY(Y,c[i]*h,X[1],X[0]);CHKERRQ(ierr);
        } else {
          /* Linear extrapolation from the last stage */
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

    ierr = (*gl->EstimateHigherMoments)(scheme,h,Ydot,gl->X,gl->himom);CHKERRQ(ierr);
    /* hmnorm[i] = h^{p+i}x^{(p+i)} with i=0,1,2; hmnorm[3] = h^{p+2}(dx'/dx) x^{(p+1)} */
    for (i=0; i<3; i++) {
      ierr = TSGLVecNormWRMS(ts,gl->himom[i],&hmnorm[i+1]);CHKERRQ(ierr);
    }
    enorm[0] = scheme->alpha[0]*hmnorm[1];
    enorm[1] = scheme->beta[0] *hmnorm[2];
    enorm[2] = scheme->gamma[0]*hmnorm[3];
    tleft = ts->max_time - ts->ptime;
    ierr = (*gl->CtrlAcceptanceTest)(ts,tleft,h,enorm,&accept,gl->ctrlP);CHKERRQ(ierr);
    if (!accept) {ierr = PetscInfo1(ts,"Step %D (t=%g) not accepted, proceeding anyway (FIXME)\n",k);CHKERRQ(ierr);}

    /* Not an error, this term is handled correctly, but it *would* be relevant for a lower order method */
    ierr = TSGLVecNormWRMS(ts,gl->X[scheme->r-1],&hmnorm[0]);CHKERRQ(ierr);
    /* Correct scaling so that these are equivalent to norms of the Nordsieck vectors */

    ierr = PetscPrintf(((PetscObject)ts)->comm,"Estimated error norms (%10.2e) %10.2e %10.2e %10.2e\n",hmnorm[0],enorm[0],enorm[1],enorm[2]);CHKERRQ(ierr);
    ierr = (*gl->CtrlChooseNextScheme)(ts,gl->nschemes,gl->schemes,gl->current_scheme,h,hmnorm,tleft,&next_scheme,&next_h,gl->ctrlP);CHKERRQ(ierr);

    X = gl->Xold;
    gl->Xold = gl->X;
    gl->X = X;
    ierr = (*gl->RescaleAndModify)(scheme,h,gl->schemes[next_scheme],next_h,Ydot,gl->Xold,gl->himom,gl->X);CHKERRQ(ierr);

    ierr = TSGLUpdateWRMS(ts);CHKERRQ(ierr);

    /* Post the solution for the user, we could avoid this copy with a small bit of cleverness */
    ierr = VecCopy(gl->X[0],ts->vec_sol);CHKERRQ(ierr);

    gl->current_scheme = next_scheme;
    ts->time_step = next_h;
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
  ierr = VecDestroyVecs(gl->himom,3);CHKERRQ(ierr);
  ierr = VecDestroy(gl->W);CHKERRQ(ierr);
  ierr = VecDestroy(gl->Y);CHKERRQ(ierr);
  ierr = VecDestroy(gl->Z);CHKERRQ(ierr);
  ierr = (*gl->Destroy)(gl);CHKERRQ(ierr);
  if (gl->CtrlDestroy) {ierr = gl->CtrlDestroy(ts,gl->ctrlP);CHKERRQ(ierr);}
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
  ierr = VecDuplicateVecs(ts->vec_sol,3,&gl->himom);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&gl->W);CHKERRQ(ierr);
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

  if (gl->current_scheme < 0) {
    PetscInt i;
    for (i=0; ; i++) {
      if (gl->schemes[i]->p == gl->start_order) break;
      if (i+1 == gl->nschemes) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"No schemes available with requested start order %d",i);
    }
    gl->current_scheme = i;
  }
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
    ierr = PetscOptionsInt("-ts_gl_start_order","Initial order to try","TSGLSetMinOrder",gl->start_order,&gl->start_order,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-ts_gl_error_direction","Which direction to look when estimating error","TSGLSetErrorDirection",TSGLErrorDirections,(PetscEnum)gl->error_direction,(PetscEnum*)&gl->error_direction,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ts_gl_extrapolate","Extrapolate stage solution from previous solution (sometimes unstable)","TSGLSetExtrapolate",gl->extrapolate,&gl->extrapolate,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_gl_atol","Absolute tolerance","TSGLSetTolerances",gl->wrms_atol,&gl->wrms_atol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_gl_rtol","Relative tolerance","TSGLSetTolerances",gl->wrms_rtol,&gl->wrms_rtol,PETSC_NULL);CHKERRQ(ierr);
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
  PetscTruth      iascii,details;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  min order %D, max order %D, current order %D\n",gl->min_order,gl->max_order,gl->schemes[gl->current_scheme]->p);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Error estimation: %s\n",TSGLErrorDirections[gl->error_direction]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Extrapolation: %s\n",gl->extrapolate?"yes":"no");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",gl->type_name[0]?gl->type_name:"(not yet set)");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Schemes within family (%d):\n",gl->nschemes);CHKERRQ(ierr);
    details = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(((PetscObject)ts)->prefix,"-ts_gl_view_detailed",&details,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i=0; i<gl->nschemes; i++) {
      ierr = TSGLSchemeView(gl->schemes[i],details,viewer);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "TSGLCtrlAcceptanceTest_Always"
static PetscErrorCode TSGLCtrlAcceptanceTest_Always(TS ts,PetscReal tleft,PetscReal h,const PetscReal enorms[],PetscTruth *accept,void *ctx)
{
  PetscFunctionBegin;
  *accept = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLCtrlChooseNextScheme_Default"
static PetscErrorCode TSGLCtrlChooseNextScheme_Default(TS ts,PetscInt n,const TSGLScheme sc[],PetscInt cur_sc,PetscReal h,const PetscReal hmnorm[],PetscReal tleft,PetscInt *next_sc,PetscReal *next_h,void *ctx)
{
  PetscErrorCode ierr;
  struct {PetscInt id,p; PetscReal h,eff;} best={-1,0,0},trial,cur;
  PetscInt cur_p,i;

  PetscFunctionBegin;
  cur_p = sc[cur_sc]->p;
  for (i=0; i<n; i++) {
    PetscReal mom;
    trial.p = sc[i]->p;
    if      (trial.p == cur_p-1) mom = hmnorm[0];
    else if (trial.p == cur_p)   mom = hmnorm[1];
    else if (trial.p == cur_p+1) mom = hmnorm[2]+hmnorm[3];
    else continue;
    trial.id = i;
    trial.h = h*PetscPowScalar((PetscScalar)(sc[i]->alpha[0]*mom),1./(sc[i]->p+1));
    trial.eff = trial.h/sc[i]->s;
    if (trial.eff > best.eff) {ierr = PetscMemcpy(&best,&trial,sizeof(trial));CHKERRQ(ierr);}
    if (i == cur_sc) {ierr = PetscMemcpy(&cur,&trial,sizeof(trial));CHKERRQ(ierr);}
  }
  /* Only switch orders if the scheme offers significant benefits over the current one.
  When the scheme is not changing, only change step size if it offers significant benefits. */
  if (best.eff < 1.2*cur.eff) {
    *next_sc = cur.id;
    if (PetscAbs(cur.h/h - 1.) < 0.1) *next_h  = h;
    else *next_h = cur.h;
  } else {
    *next_sc = best.id;
    *next_h  = cur.h;
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSGL - DAE solver using implicit General Linear methods

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
  gl->max_order = 3;
  gl->start_order = 1;
  gl->extrapolate = PETSC_FALSE;

  gl->wrms_atol = 1e-8;
  gl->wrms_rtol = 1e-5;

  gl->CtrlAcceptanceTest   = TSGLCtrlAcceptanceTest_Always;
  gl->CtrlChooseNextScheme = TSGLCtrlChooseNextScheme_Default;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSGLSetType_C","TSGLSetType_GL",&TSGLSetType_GL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
