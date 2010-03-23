#define PETSCTS_DLL

#include "gl.h"                /*I   "petscts.h"   I*/
#include "petscblaslapack.h"

static const char *TSGLErrorDirections[] = {"FORWARD","BACKWARD","TSGLErrorDirection","TSGLERROR_",0};
static PetscFList TSGLList;
static PetscFList TSGLAcceptList;
static PetscTruth TSGLPackageInitialized;
static PetscTruth TSGLRegisterAllCalled;

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
  ierr = PetscNew(struct _TSGLScheme,&scheme);CHKERRQ(ierr);
  scheme->p  = p;
  scheme->q  = q;
  scheme->r  = r;
  scheme->s  = s;

  ierr = PetscMalloc5(s,PetscScalar,&scheme->c,s*s,PetscScalar,&scheme->a,r*s,PetscScalar,&scheme->b,r*s,PetscScalar,&scheme->u,r*r,PetscScalar,&scheme->v);CHKERRQ(ierr);
  ierr = PetscMemcpy(scheme->c,c,s*sizeof(PetscScalar));CHKERRQ(ierr);
  for (j=0; j<s*s; j++) scheme->a[j] = (PetscAbsScalar(a[j]) < 1e-12) ? 0 : a[j];
  for (j=0; j<r*s; j++) scheme->b[j] = (PetscAbsScalar(b[j]) < 1e-12) ? 0 : b[j];
  for (j=0; j<s*r; j++) scheme->u[j] = (PetscAbsScalar(u[j]) < 1e-12) ? 0 : u[j];
  for (j=0; j<r*r; j++) scheme->v[j] = (PetscAbsScalar(v[j]) < 1e-12) ? 0 : v[j];

  ierr = PetscMalloc6(r,PetscScalar,&scheme->alpha,r,PetscScalar,&scheme->beta,r,PetscScalar,&scheme->gamma,3*s,PetscScalar,&scheme->phi,3*r,PetscScalar,&scheme->psi,r,PetscScalar,&scheme->stage_error);CHKERRQ(ierr);
  {
    PetscInt     i,j,k,ss=s+2;
    PetscBLASInt m,n,one=1,*ipiv,lwork=4*((s+3)*3+3),info,rank,ldb;
    PetscReal    rcond,*sing,*workreal;
    PetscScalar  *ImV,*H,*bmat,*workscalar,*c=scheme->c,*a=scheme->a,*b=scheme->b,*u=scheme->u,*v=scheme->v;
    ierr = PetscMalloc7(PetscSqr(r),PetscScalar,&ImV,3*s,PetscScalar,&H,3*ss,PetscScalar,&bmat,lwork,PetscScalar,&workscalar,5*(3+r),PetscReal,&workreal,r+s,PetscReal,&sing,r+s,PetscBLASInt,&ipiv);CHKERRQ(ierr);

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
      scheme->gamma[i] = (i==1 ? -1. : 0)*scheme->alpha[0];
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
    scheme->gamma[0] = 0.0;
    for (j=0; j<s; j++) scheme->gamma[0] += b[0*s+j]*scheme->stage_error[j];
    for (j=1; j<r; j++) scheme->gamma[0] += v[0*s+j]*scheme->gamma[j];

    /* Assemble H
    *    % Determine the error estimators phi
       H = [[cpow(glm.c,p) + C*e.alpha] [cpow(glm.c,p+1) + C*e.beta] ...
               [e.xi - C*(e.gamma + 0*e.epsilon*eye(s-1,1))]]';
    % Paper has formula above without the 0, but that term must be left
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
    bmat[0+0*ss] = 1.;  bmat[0+1*ss] = 0.;  bmat[0+2*ss] = 0.;
    bmat[1+0*ss] = 1.;  bmat[1+1*ss] = 1.;  bmat[1+2*ss] = 0.;
    bmat[2+0*ss] = 0.;  bmat[2+1*ss] = 0.;  bmat[2+2*ss] = -1.;
    m = 3;
    n = PetscBLASIntCast(s);
    ldb = PetscBLASIntCast(ss);
    rcond = 1e-12;
#if defined(PETSC_USE_COMPLEX)
    /* ZGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, RWORK, INFO ) */
    LAPACKgelss_(&m,&n,&m,H,&m,bmat,&ldb,sing,&rcond,&rank,workscalar,&lwork,workreal,&info);
#else
    /* DGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, INFO ) */
    LAPACKgelss_(&m,&n,&m,H,&m,bmat,&ldb,sing,&rcond,&rank,workscalar,&lwork,&info);
#endif
    if (info < 0) SETERRQ(PETSC_ERR_LIB,"Bad argument to GELSS");
    if (info > 0) SETERRQ(PETSC_ERR_LIB,"SVD failed to converge");

    for (j=0; j<3; j++) {
      for (k=0; k<s; k++) {
        scheme->phi[k+j*s] = bmat[k+j*ss];
      }
    }

    /* the other part of the error estimator, psi in B,J,W 2007 */
    scheme->psi[0*r+0] = 0.;
    scheme->psi[1*r+0] = 0.;
    scheme->psi[2*r+0] = 0.;
    for (j=1; j<r; j++) {
      scheme->psi[0*r+j] = 0.;
      scheme->psi[1*r+j] = 0.;
      scheme->psi[2*r+j] = 0.;
      for (k=0; k<s; k++) {
        scheme->psi[0*r+j] -= CPowF(c[k],j-1)*scheme->phi[0*s+k];
        scheme->psi[1*r+j] -= CPowF(c[k],j-1)*scheme->phi[1*s+k];
        scheme->psi[2*r+j] -= CPowF(c[k],j-1)*scheme->phi[2*s+k];
      }
    }
    ierr = PetscFree7(ImV,H,bmat,workscalar,workreal,sing,ipiv);CHKERRQ(ierr);
  }
  /* Check which properties are satisfied */
  scheme->stiffly_accurate = PETSC_TRUE;
  if (scheme->c[s-1] != 1.) scheme->stiffly_accurate = PETSC_FALSE;
  for (j=0; j<s; j++) if (a[(s-1)*s+j] != b[j]) scheme->stiffly_accurate = PETSC_FALSE;
  for (j=0; j<r; j++) if (u[(s-1)*r+j] != v[j]) scheme->stiffly_accurate = PETSC_FALSE;
  scheme->fsal = scheme->stiffly_accurate; /* FSAL is stronger */
  for (j=0; j<s-1; j++) if (r>1 && b[1*s+j] != 0.) scheme->fsal = PETSC_FALSE;
  if (b[1*s+r-1] != 1.) scheme->fsal = PETSC_FALSE;
  for (j=0; j<r; j++) if (r>1 && v[1*r+j] != 0.) scheme->fsal = PETSC_FALSE;

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
  ierr = PetscFree6(sc->alpha,sc->beta,sc->gamma,sc->phi,sc->psi,sc->stage_error);CHKERRQ(ierr);
  ierr = PetscFree(sc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLDestroy_Default"
static PetscErrorCode TSGLDestroy_Default(TS_GL *gl)
{
  PetscErrorCode ierr;
  PetscInt       i;

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
        ierr = PetscViewerASCIIPrintf(viewer," %12.8g",PetscRealPart(a[i*n+j]));CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer,"Leading error constants: %10.3e  %10.3e  %10.3e\n",
                                  PetscRealPart(sc->alpha[0]),PetscRealPart(sc->beta[0]),PetscRealPart(sc->gamma[0]));CHKERRQ(ierr);
    ierr = ViewTable_Private(viewer,1,sc->s,sc->c,"Abscissas c");CHKERRQ(ierr);
    if (view_details) {
      ierr = ViewTable_Private(viewer,sc->s,sc->s,sc->a,"A");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,sc->r,sc->s,sc->b,"B");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,sc->s,sc->r,sc->u,"U");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,sc->r,sc->r,sc->v,"V");CHKERRQ(ierr);

      ierr = ViewTable_Private(viewer,3,sc->s,sc->phi,"Error estimate phi");CHKERRQ(ierr);
      ierr = ViewTable_Private(viewer,3,sc->r,sc->psi,"Error estimate psi");CHKERRQ(ierr);
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
  PetscInt       i;

  PetscFunctionBegin;
  if (sc->r > 64 || sc->s > 64) SETERRQ(PETSC_ERR_PLIB,"Ridiculous number of stages or items passed between stages");
  /* build error vectors*/
  for (i=0; i<3; i++) {
    PetscScalar phih[64];
    PetscInt j;
    for (j=0; j<sc->s; j++) phih[j] = sc->phi[i*sc->s+j]*h;
    ierr = VecZeroEntries(hm[i]);CHKERRQ(ierr);
    ierr = VecMAXPY(hm[i],sc->s,phih,Ydot);CHKERRQ(ierr);
    ierr = VecMAXPY(hm[i],sc->r,&sc->psi[i*sc->r],Xold);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLCompleteStep_Rescale"
static PetscErrorCode TSGLCompleteStep_Rescale(TSGLScheme sc,PetscReal h,TSGLScheme next_sc,PetscReal next_h,Vec Ydot[],Vec Xold[],Vec X[])
{
  PetscErrorCode ierr;
  PetscScalar    brow[32],vrow[32];
  PetscInt       i,j,r,s;

  PetscFunctionBegin;
  /* Build the new solution from (X,Ydot) */
  r = sc->r;
  s = sc->s;
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
#define __FUNCT__ "TSGLCompleteStep_RescaleAndModify"
static PetscErrorCode TSGLCompleteStep_RescaleAndModify(TSGLScheme sc,PetscReal h,TSGLScheme next_sc,PetscReal next_h,Vec Ydot[],Vec Xold[],Vec X[])
{
  PetscErrorCode ierr;
  PetscScalar    brow[32],vrow[32];
  PetscReal      ratio;
  PetscInt       i,j,p,r,s;

  PetscFunctionBegin;
  /* Build the new solution from (X,Ydot) */
  p = sc->p;
  r = sc->r;
  s = sc->s;
  ratio = next_h/h;
  for (i=0; i<r; i++) {
    ierr = VecZeroEntries(X[i]);CHKERRQ(ierr);
    for (j=0; j<s; j++) {
      brow[j] = h*(PetscPowScalar(ratio,i)*sc->b[i*s+j]
                   + (PetscPowScalar(ratio,i) - PetscPowScalar(ratio,p+1))*(+ sc->alpha[i]*sc->phi[0*s+j])
                   + (PetscPowScalar(ratio,i) - PetscPowScalar(ratio,p+2))*(+ sc->beta [i]*sc->phi[1*s+j]
                                                      + sc->gamma[i]*sc->phi[2*s+j]));
    }
    ierr = VecMAXPY(X[i],s,brow,Ydot);CHKERRQ(ierr);
    for (j=0; j<r; j++) {
      vrow[j] = (PetscPowScalar(ratio,i)*sc->v[i*r+j]
                 + (PetscPowScalar(ratio,i) - PetscPowScalar(ratio,p+1))*(+ sc->alpha[i]*sc->psi[0*r+j])
                 + (PetscPowScalar(ratio,i) - PetscPowScalar(ratio,p+2))*(+ sc->beta [i]*sc->psi[1*r+j]
                                                    + sc->gamma[i]*sc->psi[2*r+j]));
    }
    ierr = VecMAXPY(X[i],r,vrow,Xold);CHKERRQ(ierr);
  }
  if (r < next_sc->r) {
    if (r+1 != next_sc->r) SETERRQ(PETSC_ERR_PLIB,"Cannot accommodate jump in r greater than 1");
    ierr = VecZeroEntries(X[r]);
    for (j=0; j<s; j++) brow[j] = h*PetscPowScalar(ratio,p+1)*sc->phi[0*s+j];
    ierr = VecMAXPY(X[r],s,brow,Ydot);CHKERRQ(ierr);
    for (j=0; j<r; j++) vrow[j] = PetscPowScalar(ratio,p+1)*sc->psi[0*r+j];
    ierr = VecMAXPY(X[r],r,vrow,Xold);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSGLCreate_IRKS"
static PetscErrorCode TSGLCreate_IRKS(TS ts)
{
  TS_GL          *gl = (TS_GL*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  gl->Destroy               = TSGLDestroy_Default;
  gl->EstimateHigherMoments = TSGLEstimateHigherMoments_Default;
  gl->CompleteStep          = TSGLCompleteStep_RescaleAndModify;
  ierr = PetscMalloc(10*sizeof(TSGLScheme),&gl->schemes);CHKERRQ(ierr);
  gl->nschemes = 0;

  {
    /* p=1,q=1, r=s=2, A- and L-stable with error estimates of order 2 and 3
    * Listed in Butcher & Podhaisky 2006. On error estimation in general linear methods for stiff ODE.
    * irks(0.3,0,[.3,1],[1],1)
    * Note: can be made to have classical order (not stage order) 2 by replacing 0.3 with 1-sqrt(1/2)
    * but doing so would sacrifice the error estimator.
    */
    const PetscScalar c[2] = {3./10., 1.}
    ,a[2][2] = {{3./10., 0}, {7./10., 3./10.}}
    ,b[2][2] = {{7./10., 3./10.}, {0,1}}
    ,u[2][2] = {{1,0},{1,0}}
    ,v[2][2] = {{1,0},{0,0}};
    ierr = TSGLSchemeCreate(1,1,2,2,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]);CHKERRQ(ierr);
  }

  {
    /* p=q=2, r=s=3: irks(4/9,0,[1:3]/3,[0.33852],1) */
    /* http://www.math.auckland.ac.nz/~hpod/atlas/i2a.html */
    const PetscScalar c[3] = {1./3., 2./3., 1}
    ,a[3][3] = {{4./9.                ,0                      ,       0},
                {1.03750643704090e+00 ,                  4./9.,       0},
                {7.67024779410304e-01 ,  -3.81140216918943e-01,   4./9.}}
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
  {
    /* p=q=4, r=s=5:
          irks(3/11,0,[1:5]/5, [0.1715   -0.1238    0.6617],...
          [ -0.0812    0.4079    1.0000
             1.0000         0         0
             0.8270    1.0000         0])
    */
    const PetscScalar c[5] = {0.2,0.4,0.6,0.8,1.0}
    ,a[5][5] = {{2.72727272727352e-01 ,   0.00000000000000e+00,  0.00000000000000e+00 ,  0.00000000000000e+00  ,  0.00000000000000e+00},
                {-1.03980153733431e-01,   2.72727272727405e-01,   0.00000000000000e+00,  0.00000000000000e+00  ,  0.00000000000000e+00},
                {-1.58615400341492e+00,   7.44168951881122e-01,   2.72727272727309e-01,  0.00000000000000e+00  ,  0.00000000000000e+00},
                {-8.73658042865628e-01,   5.37884671894595e-01,  -1.63298538799523e-01,   2.72727272726996e-01 ,  0.00000000000000e+00},
                {2.95489397443992e-01 , -1.18481693910097e+00 , -6.68029812659953e-01 ,  1.00716687860943e+00  , 2.72727272727288e-01}}
    ,b[5][5] = {{2.95489397443992e-01 , -1.18481693910097e+00 , -6.68029812659953e-01 ,  1.00716687860943e+00  , 2.72727272727288e-01},
                {0.00000000000000e+00 ,  1.11022302462516e-16 , -2.22044604925031e-16 ,  0.00000000000000e+00  , 1.00000000000000e+00},
                {-4.05882503986005e+00,  -4.00924006567769e+00,  -1.38930610972481e+00,   4.45223930308488e+00 ,  6.32331093108427e-01},
                {8.35690179937017e+00 , -2.26640927349732e+00 ,  6.86647884973826e+00 , -5.22595158025740e+00  , 4.50893068837431e+00},
                {1.27656267027479e+01 ,  2.80882153840821e+00 ,  8.91173096522890e+00 , -1.07936444078906e+01  , 4.82534148988854e+00}}
    ,u[5][5] = {{1.00000000000000e+00 , -7.27272727273551e-02 , -3.45454545454419e-02 , -4.12121212119565e-03  ,-2.96969696964014e-04},
                {1.00000000000000e+00 ,  2.31252881006154e-01 , -8.29487834416481e-03 , -9.07191207681020e-03  ,-1.70378403743473e-03},
                {1.00000000000000e+00 ,  1.16925777880663e+00 ,  3.59268562942635e-02 , -4.09013451730615e-02  ,-1.02411119670164e-02},
                {1.00000000000000e+00 ,  1.02634463704356e+00 ,  1.59375044913405e-01 ,  1.89673015035370e-03  ,-4.89987231897569e-03},
                {1.00000000000000e+00 ,  1.27746320298021e+00 ,  2.37186008132728e-01 , -8.28694373940065e-02  ,-5.34396510196430e-02}}
    ,v[5][5] = {{1.00000000000000e+00 ,  1.27746320298021e+00 ,  2.37186008132728e-01 , -8.28694373940065e-02  ,-5.34396510196430e-02},
                {0.00000000000000e+00 , -1.77635683940025e-15 , -1.99840144432528e-15 , -9.99200722162641e-16  ,-3.33066907387547e-16},
                {0.00000000000000e+00 ,  4.37280081906924e+00 ,  5.49221645016377e-02 , -8.88913177394943e-02  , 1.12879077989154e-01},
                {0.00000000000000e+00 , -1.22399504837280e+01 , -5.21287338448645e+00 , -8.03952325565291e-01  , 4.60298678047147e-01},
                {0.00000000000000e+00 , -1.85178762883829e+01 , -5.21411849862624e+00 , -1.04283436528809e+00  , 7.49030161063651e-01}};
    ierr = TSGLSchemeCreate(4,4,5,5,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]);CHKERRQ(ierr);
  }
  {
    /* p=q=5, r=s=6;
       irks(1/3,0,[1:6]/6,...
          [-0.0489    0.4228   -0.8814    0.9021],...
          [-0.3474   -0.6617    0.6294    0.2129
            0.0044   -0.4256   -0.1427   -0.8936
           -0.8267    0.4821    0.1371   -0.2557
           -0.4426   -0.3855   -0.7514    0.3014])
    */
    const PetscScalar c[6] = {1./6, 2./6, 3./6, 4./6, 5./6, 1.}
    ,a[6][6] = {{  3.33333333333940e-01,  0                   ,  0                   ,  0                   ,  0                   ,  0                   },
                { -8.64423857333350e-02,  3.33333333332888e-01,  0                   ,  0                   ,  0                   ,  0                   },
                { -2.16850174258252e+00, -2.23619072028839e+00,  3.33333333335204e-01,  0                   ,  0                   ,  0                   },
                { -4.73160970138997e+00, -3.89265344629268e+00, -2.76318716520933e-01,  3.33333333335759e-01,  0                   ,  0                   },
                { -6.75187540297338e+00, -7.90756533769377e+00,  7.90245051802259e-01, -4.48352364517632e-01,  3.33333333328483e-01,  0                   },
                { -4.26488287921548e+00, -1.19320395589302e+01,  3.38924509887755e+00, -2.23969848002481e+00,  6.62807710124007e-01,  3.33333333335440e-01}}
    ,b[6][6] = {{ -4.26488287921548e+00, -1.19320395589302e+01,  3.38924509887755e+00, -2.23969848002481e+00,  6.62807710124007e-01,  3.33333333335440e-01},
                { -8.88178419700125e-16,  4.44089209850063e-16, -1.54737334057131e-15, -8.88178419700125e-16,  0.00000000000000e+00,  1.00000000000001e+00},
                { -2.87780425770651e+01, -1.13520448264971e+01,  2.62002318943161e+01,  2.56943874812797e+01, -3.06702268304488e+01,  6.68067773510103e+00},
                {  5.47971245256474e+01,  6.80366875868284e+01, -6.50952588861999e+01, -8.28643975339097e+01,  8.17416943896414e+01, -1.17819043489036e+01},
                { -2.33332114788869e+02,  6.12942539462634e+01, -4.91850135865944e+01,  1.82716844135480e+02, -1.29788173979395e+02,  3.09968095651099e+01},
                { -1.72049132343751e+02,  8.60194713593999e+00,  7.98154219170200e-01,  1.50371386053218e+02, -1.18515423962066e+02,  2.50898277784663e+01}}
    ,u[6][6] = {{  1.00000000000000e+00, -1.66666666666870e-01, -4.16666666664335e-02, -3.85802469124815e-03, -2.25051440302250e-04, -9.64506172339142e-06},
                {  1.00000000000000e+00,  8.64423857327162e-02, -4.11484912671353e-02, -1.11450903217645e-02, -1.47651050487126e-03, -1.34395070766826e-04},
                {  1.00000000000000e+00,  4.57135912953434e+00,  1.06514719719137e+00,  1.33517564218007e-01,  1.11365952968659e-02,  6.12382756769504e-04},
                {  1.00000000000000e+00,  9.23391519753404e+00,  2.22431212392095e+00,  2.91823807741891e-01,  2.52058456411084e-02,  1.22800542949647e-03},
                {  1.00000000000000e+00,  1.48175480533865e+01,  3.73439117461835e+00,  5.14648336541804e-01,  4.76430038853402e-02,  2.56798515502156e-03},
                {  1.00000000000000e+00,  1.50512347758335e+01,  4.10099701165164e+00,  5.66039141003603e-01,  3.91213893800891e-02, -2.99136269067853e-03}}
    ,v[6][6] = {{  1.00000000000000e+00,  1.50512347758335e+01,  4.10099701165164e+00,  5.66039141003603e-01,  3.91213893800891e-02, -2.99136269067853e-03},
                {  0.00000000000000e+00, -4.88498130835069e-15, -6.43929354282591e-15, -3.55271367880050e-15, -1.22124532708767e-15, -3.12250225675825e-16},
                {  0.00000000000000e+00,  1.22250171233141e+01, -1.77150760606169e+00,  3.54516769879390e-01,  6.22298845883398e-01,  2.31647447450276e-01},
                {  0.00000000000000e+00, -4.48339457331040e+01, -3.57363126641880e-01,  5.18750173123425e-01,  6.55727990241799e-02,  1.63175368287079e-01},
                {  0.00000000000000e+00,  1.37297394708005e+02, -1.60145272991317e+00, -5.05319555199441e+00,  1.55328940390990e-01,  9.16629423682464e-01},
                {  0.00000000000000e+00,  1.05703241119022e+02, -1.16610260983038e+00, -2.99767252773859e+00, -1.13472315553890e-01,  1.09742849254729e+00}};
    ierr = TSGLSchemeCreate(5,5,6,6,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "TSGLSetType"
/*@C
   TSGLSetType - sets the class of general linear method to use for time-stepping

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  type - a method

   Options Database Key:
.  -ts_gl_type <type> - sets the method, use -help for a list of available method (e.g. irks)

   Notes:
   See "petsc/include/petscts.h" for available methods (for instance)
.    TSGL_IRKS - Diagonally implicit methods with inherent Runge-Kutta stability (for stiff problems)

   Normally, it is best to use the TSSetFromOptions() command and
   then set the TSGL type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the many different solvers.
   The TSGLSetType() routine is provided for those situations where it
   is necessary to set the timestepping solver independently of the
   command line or options database.  This might be the case, for example,
   when the choice of solver changes during the execution of the
   program, and the user's application is taking responsibility for
   choosing the appropriate method.

   Level: intermediate

.keywords: TS, TSGL, set, type
@*/
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
#define __FUNCT__ "TSGLSetAcceptType"
/*@C
   TSGLSetAcceptType - sets the acceptance test

   Time integrators that need to control error must have the option to reject a time step based on local error
   estimates.  This function allows different schemes to be set.

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  type - the type

   Options Database Key:
.  -ts_gl_accept_type <type> - sets the method used to determine whether to accept or reject a step

   Level: intermediate

.seealso: TS, TSGL, TSGLAcceptRegisterDynamic(), TSGLAdapt, set type
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLSetAcceptType(TS ts,const TSGLAcceptType type)
{
  PetscErrorCode ierr,(*r)(TS,const TSGLAcceptType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSGLSetAcceptType_C",(void(**)(void))&r);CHKERRQ(ierr);
  if (r) {
    ierr = (*r)(ts,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLGetAdapt"
/*@C
   TSGLGetAdapt - gets the TSGLAdapt object from the TS

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
.  adapt - the TSGLAdapt context

   Notes:
   This allows the user set options on the TSGLAdapt object.  Usually it is better to do this using the options
   database, so this function is rarely needed.

   Level: advanced

.seealso: TSGLAdapt, TSGLAdaptRegisterDynamic()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLGetAdapt(TS ts,TSGLAdapt *adapt)
{
  PetscErrorCode ierr,(*r)(TS,TSGLAdapt*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);
  PetscValidPointer(adapt,2);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSGLGetAdapt_C",(void(**)(void))&r);CHKERRQ(ierr);
  if (r) {
    ierr = (*r)(ts,adapt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLAccept_Always"
static PetscErrorCode TSGLAccept_Always(TS ts,PetscReal tleft,PetscReal h,const PetscReal enorms[],PetscTruth *accept)
{
  PetscFunctionBegin;
  *accept = PETSC_TRUE;
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
    w[i] = 1./(gl->wrms_atol + gl->wrms_rtol*PetscAbsScalar(x[i]));
  }
  ierr = VecRestoreArray(gl->X[0],&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(gl->W,&w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGLVecNormWRMS"
static PetscErrorCode TSGLVecNormWRMS(TS ts,Vec X,PetscReal *nrm)
{
  TS_GL          *gl = (TS_GL*)ts->data;
  PetscErrorCode ierr;
  PetscScalar    *x,*w,sum = 0.0;
  PetscInt       n,i;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(gl->W,&w);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gl->W,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    sum += PetscAbsScalar(PetscSqr(x[i]*w[i]));
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(gl->W,&w);CHKERRQ(ierr);
  *nrm = PetscAbsScalar(PetscSqrtScalar(sum/(1.*n)));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLSetType_GL"
static PetscErrorCode TSGLSetType_GL(TS ts,const TSGLType type)
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
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSGL type \"%s\" given",type);
  ierr = (*r)(ts);CHKERRQ(ierr);
  ierr = PetscStrcpy(gl->type_name,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLSetAcceptType_GL"
static PetscErrorCode TSGLSetAcceptType_GL(TS ts,const TSGLAcceptType type)
{
  PetscErrorCode ierr;
  TSGLAcceptFunction r;
  TS_GL *gl = (TS_GL*)ts->data;

  PetscFunctionBegin;
  ierr = PetscFListFind(TSGLAcceptList,((PetscObject)ts)->comm,type,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSGLAccept type \"%s\" given",type);
  gl->Accept = r;
  ierr = PetscStrncpy(gl->accept_name,type,sizeof(gl->accept_name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLGetAdapt_GL"
static PetscErrorCode TSGLGetAdapt_GL(TS ts,TSGLAdapt *adapt)
{
  PetscErrorCode ierr;
  TS_GL *gl = (TS_GL*)ts->data;

  PetscFunctionBegin;
  if (!gl->adapt) {
    ierr = TSGLAdaptCreate(((PetscObject)ts)->comm,&gl->adapt);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)gl->adapt,(PetscObject)ts,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(ts,gl->adapt);CHKERRQ(ierr);
  }
  *adapt = gl->adapt;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLChooseNextScheme"
static PetscErrorCode TSGLChooseNextScheme(TS ts,PetscReal h,const PetscReal hmnorm[],PetscInt *next_scheme,PetscReal *next_h,PetscTruth *finish)
{
  PetscErrorCode ierr;
  TS_GL *gl = (TS_GL*)ts->data;
  PetscInt i,n,cur_p,cur,next_sc,candidates[64],orders[64];
  PetscReal errors[64],costs[64],tleft;

  PetscFunctionBegin;
  cur = -1;
  cur_p = gl->schemes[gl->current_scheme]->p;
  tleft = ts->max_time - (ts->ptime + ts->time_step);
  for (i=0,n=0; i<gl->nschemes; i++) {
    TSGLScheme sc = gl->schemes[i];
    if (sc->p < gl->min_order || gl->max_order < sc->p) continue;
    if (sc->p == cur_p - 1) {
      errors[n] = PetscAbsScalar(sc->alpha[0])*hmnorm[0];
    } else if (sc->p == cur_p) {
      errors[n] = PetscAbsScalar(sc->alpha[0])*hmnorm[1];
    } else if (sc->p == cur_p+1) {
      errors[n] = PetscAbsScalar(sc->alpha[0])*(hmnorm[2]+hmnorm[3]);
    } else continue;
    candidates[n] = i;
    orders[n]     = PetscMin(sc->p,sc->q); /* order of global truncation error */
    costs[n]      = sc->s;                 /* estimate the cost as the number of stages */
    if (i == gl->current_scheme) cur = n;
    n++;
  }
  if (cur < 0 || gl->nschemes <= cur) SETERRQ(PETSC_ERR_PLIB,"Current scheme not found in scheme list");
  ierr = TSGLAdaptChoose(gl->adapt,n,orders,errors,costs,cur,h,tleft,&next_sc,next_h,finish);CHKERRQ(ierr);
  *next_scheme = candidates[next_sc];
  ierr = PetscInfo7(ts,"Adapt chose scheme %d (%d,%d,%d,%d) with step size %6.2e, finish=%d\n",*next_scheme,gl->schemes[*next_scheme]->p,gl->schemes[*next_scheme]->q,gl->schemes[*next_scheme]->r,gl->schemes[*next_scheme]->s,*next_h,*finish);CHKERRQ(ierr);
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
  TS_GL          *gl = (TS_GL*)ts->data;
  PetscInt       i,k,max_steps = ts->max_steps,its,lits,max_r,max_s;
  PetscTruth     final_step,finish;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *steps = -ts->steps;
  *ptime  = ts->ptime;

  ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

  ierr = TSGLGetMaxSizes(ts,&max_r,&max_s);CHKERRQ(ierr);
  ierr = VecCopy(ts->vec_sol,gl->X[0]);CHKERRQ(ierr);
  for (i=1; i<max_r; i++) {
    ierr = VecZeroEntries(gl->X[i]);CHKERRQ(ierr);
  }
  ierr = TSGLUpdateWRMS(ts);CHKERRQ(ierr);

  if (0) {
    /* Find consistent initial data for DAE */
    gl->stage_time = ts->ptime + ts->time_step;
    gl->shift = 1./ts->time_step;
    gl->stage = 0;
    ierr = VecCopy(ts->vec_sol,gl->Z);CHKERRQ(ierr);
    ierr = VecCopy(ts->vec_sol,gl->Y);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,PETSC_NULL,gl->Y);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;
  }

  if (gl->current_scheme < 0) SETERRQ(PETSC_ERR_ORDER,"A starting scheme has not been provided");

  for (k=0,final_step=PETSC_FALSE,finish=PETSC_FALSE; k<max_steps && !finish; k++) {
    PetscInt j,r,s,next_scheme = 0,rejections;
    PetscReal h,hmnorm[4],enorm[3],next_h;
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

    if (ts->ptime > ts->max_time) break;

    /*
      We only call PreStep at the start of each STEP, not each STAGE.  This is because it is
      possible to fail (have to restart a step) after multiple stages.
    */
    ierr = TSPreStep(ts);CHKERRQ(ierr);

    rejections = 0;
    while (1) {
      for (i=0; i<s; i++) {
        PetscScalar shift = gl->shift = 1./PetscRealPart(h*a[i*s+i]);
        gl->stage = i;
        gl->stage_time = ts->ptime + PetscRealPart(c[i])*h;

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

      gl->stage_time = ts->ptime + ts->time_step;

      ierr = (*gl->EstimateHigherMoments)(scheme,h,Ydot,gl->X,gl->himom);CHKERRQ(ierr);
      /* hmnorm[i] = h^{p+i}x^{(p+i)} with i=0,1,2; hmnorm[3] = h^{p+2}(dx'/dx) x^{(p+1)} */
      for (i=0; i<3; i++) {
        ierr = TSGLVecNormWRMS(ts,gl->himom[i],&hmnorm[i+1]);CHKERRQ(ierr);
      }
      enorm[0] = PetscRealPart(scheme->alpha[0])*hmnorm[1];
      enorm[1] = PetscRealPart(scheme->beta[0]) *hmnorm[2];
      enorm[2] = PetscRealPart(scheme->gamma[0])*hmnorm[3];
      ierr = (*gl->Accept)(ts,ts->max_time-gl->stage_time,h,enorm,&accept);CHKERRQ(ierr);
      if (accept) goto accepted;
      rejections++;
      ierr = PetscInfo3(ts,"Step %D (t=%g) not accepted, rejections=%D\n",k,gl->stage_time,rejections);CHKERRQ(ierr);
      if (rejections > gl->max_step_rejections) break;
      /*
        There are lots of reasons why a step might be rejected, including solvers not converging and other factors that
        TSGLChooseNextScheme does not support.  Additionally, the error estimates may be very screwed up, so I'm not
        convinced that it's safe to just compute a new error estimate using the same interface as the current adaptor
        (the adaptor interface probably has to change).  Here we make an arbitrary and naive choice.  This assumes that
        steps were written in Nordsieck form.  The "correct" method would be to re-complete the previous time step with
        the correct "next" step size.  It is unclear to me whether the present ad-hoc method of rescaling X is stable.
      */
      h *= 0.5;
      for (i=1; i<scheme->r; i++) {
        ierr = VecScale(X[i],PetscPowScalar(0.5,i));CHKERRQ(ierr);
      }
    }
    SETERRQ3(PETSC_ERR_CONV_FAILED,"Time step %D (t=%g) not accepted after %D failures\n",k,gl->stage_time,rejections);CHKERRQ(ierr);

    accepted:
    /* This term is not error, but it *would* be the leading term for a lower order method */
    ierr = TSGLVecNormWRMS(ts,gl->X[scheme->r-1],&hmnorm[0]);CHKERRQ(ierr);
    /* Correct scaling so that these are equivalent to norms of the Nordsieck vectors */

    ierr = PetscInfo4(ts,"Last moment norm %10.2e, estimated error norms %10.2e %10.2e %10.2e\n",hmnorm[0],enorm[0],enorm[1],enorm[2]);CHKERRQ(ierr);
    if (!final_step) {
      ierr = TSGLChooseNextScheme(ts,h,hmnorm,&next_scheme,&next_h,&final_step);CHKERRQ(ierr);
    } else {
      /* Dummy values to complete the current step in a consistent manner */
      next_scheme = gl->current_scheme;
      next_h = h;
      finish = PETSC_TRUE;
    }

    X = gl->Xold;
    gl->Xold = gl->X;
    gl->X = X;
    ierr = (*gl->CompleteStep)(scheme,h,gl->schemes[next_scheme],next_h,Ydot,gl->Xold,gl->X);CHKERRQ(ierr);

    ierr = TSGLUpdateWRMS(ts);CHKERRQ(ierr);

    /* Post the solution for the user, we could avoid this copy with a small bit of cleverness */
    ierr = VecCopy(gl->X[0],ts->vec_sol);CHKERRQ(ierr);
    ts->ptime += h;
    ts->steps++;

    ierr = TSPostStep(ts);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

    gl->current_scheme = next_scheme;
    ts->time_step = next_h;
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
  if (gl->setupcalled) {
    ierr = TSGLGetMaxSizes(ts,&max_r,&max_s);CHKERRQ(ierr);
    ierr = VecDestroyVecs(gl->Xold,max_r);CHKERRQ(ierr);
    ierr = VecDestroyVecs(gl->X,max_r);CHKERRQ(ierr);
    ierr = VecDestroyVecs(gl->Ydot,max_s);CHKERRQ(ierr);
    ierr = VecDestroyVecs(gl->himom,3);CHKERRQ(ierr);
    ierr = VecDestroy(gl->W);CHKERRQ(ierr);
    ierr = VecDestroy(gl->Y);CHKERRQ(ierr);
    ierr = VecDestroy(gl->Z);CHKERRQ(ierr);
  }
  if (gl->adapt) {ierr = TSGLAdaptDestroy(gl->adapt);CHKERRQ(ierr);}
  if (gl->Destroy) {ierr = (*gl->Destroy)(gl);CHKERRQ(ierr);}
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
  ierr = TSComputeIFunction(ts,gl->stage_time,x,gl->Ydot[gl->stage],f);CHKERRQ(ierr);
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
  ierr = TSComputeIJacobian(ts,gl->stage_time,x,gl->Ydot[gl->stage],gl->shift,A,B,str);CHKERRQ(ierr);
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
  if (ts->problem_type == TS_LINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Only for nonlinear problems");
  }
  gl->setupcalled = PETSC_TRUE;
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

  /* Default acceptance tests and adaptivity */
  if (!gl->Accept) {ierr = TSGLSetAcceptType(ts,TSGLACCEPT_ALWAYS);CHKERRQ(ierr);}
  if (!gl->adapt)  {ierr = TSGLGetAdapt(ts,&gl->adapt);CHKERRQ(ierr);}

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
  char tname[256] = TSGL_IRKS,completef[256] = "rescale-and-modify";
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("General Linear ODE solver options");CHKERRQ(ierr);
  {
    PetscTruth flg;
    ierr = PetscOptionsList("-ts_gl_type","Type of GL method","TSGLSetType",TSGLList,gl->type_name[0]?gl->type_name:tname,tname,sizeof(tname),&flg);CHKERRQ(ierr);
    if (flg || !gl->type_name[0]) {
      ierr = TSGLSetType(ts,tname);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-ts_gl_max_step_rejections","Maximum number of times to attempt a step","None",gl->max_step_rejections,&gl->max_step_rejections,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_gl_max_order","Maximum order to try","TSGLSetMaxOrder",gl->max_order,&gl->max_order,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_gl_min_order","Minimum order to try","TSGLSetMinOrder",gl->min_order,&gl->min_order,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_gl_start_order","Initial order to try","TSGLSetMinOrder",gl->start_order,&gl->start_order,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-ts_gl_error_direction","Which direction to look when estimating error","TSGLSetErrorDirection",TSGLErrorDirections,(PetscEnum)gl->error_direction,(PetscEnum*)&gl->error_direction,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ts_gl_extrapolate","Extrapolate stage solution from previous solution (sometimes unstable)","TSGLSetExtrapolate",gl->extrapolate,&gl->extrapolate,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_gl_atol","Absolute tolerance","TSGLSetTolerances",gl->wrms_atol,&gl->wrms_atol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_gl_rtol","Relative tolerance","TSGLSetTolerances",gl->wrms_rtol,&gl->wrms_rtol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-ts_gl_complete","Method to use for completing the step","none",completef,completef,sizeof completef,&flg);CHKERRQ(ierr);
    if (flg) {
      PetscTruth match1,match2;
      ierr = PetscStrcmp(completef,"rescale",&match1);CHKERRQ(ierr);
      ierr = PetscStrcmp(completef,"rescale-and-modify",&match2);CHKERRQ(ierr);
      if (match1)      gl->CompleteStep = TSGLCompleteStep_Rescale;
      else if (match2) gl->CompleteStep = TSGLCompleteStep_RescaleAndModify;
      else SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"%s",completef);
    }
    {
      char type[256] = TSGLACCEPT_ALWAYS;
      ierr = PetscOptionsList("-ts_gl_accept_type","Method to use for determining whether to accept a step","TSGLSetAcceptType",TSGLAcceptList,gl->accept_name[0]?gl->accept_name:type,type,sizeof type,&flg);CHKERRQ(ierr);
      if (flg || !gl->accept_name[0]) {
        ierr = TSGLSetAcceptType(ts,type);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  {
    TSGLAdapt adapt;
    ierr = TSGLGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSGLAdaptSetFromOptions(adapt);CHKERRQ(ierr);
  }
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
    ierr = PetscViewerASCIIPrintf(viewer,"  Acceptance test: %s\n",gl->accept_name[0]?gl->accept_name:"(not yet set)");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = TSGLAdaptView(gl->adapt,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for TSGL",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLRegister"
/*@C
   TSGLRegister - see TSGLRegisterDynamic()

   Level: advanced
@*/
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
#define __FUNCT__ "TSGLAcceptRegister"
/*@C
   TSGLAcceptRegister - see TSGLAcceptRegisterDynamic()

   Level: advanced
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLAcceptRegister(const char sname[],const char path[],const char name[],TSGLAcceptFunction function)
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&TSGLAcceptList,sname,fullname,(void(*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLRegisterAll"
/*@C
  TSGLRegisterAll - Registers all of the general linear methods in TSGL

  Not Collective

  Level: advanced

.keywords: TS, TSGL, register, all

.seealso:  TSGLRegisterDestroy()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSGLRegisterAllCalled = PETSC_TRUE;

  ierr = TSGLRegisterDynamic(TSGL_IRKS,path,"TSGLCreate_IRKS",TSGLCreate_IRKS);CHKERRQ(ierr);
  ierr = TSGLAcceptRegisterDynamic(TSGLACCEPT_ALWAYS,path,"TSGLAccept_Always",TSGLAccept_Always);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLRegisterDestroy"
/*@C
   TSGLRegisterDestroy - Frees the list of schemes that were registered by TSGLRegister()/TSGLRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: TSGL, register, destroy
.seealso: TSGLRegister(), TSGLRegisterAll(), TSGLRegisterDynamic()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&TSGLList);CHKERRQ(ierr);
  TSGLRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSGLInitializePackage"
/*@C
  TSGLInitializePackage - This function initializes everything in the TSGL package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to TSCreate_GL()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: TS, TSGL, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLInitializePackage(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSGLPackageInitialized) PetscFunctionReturn(0);
  TSGLPackageInitialized = PETSC_TRUE;
  ierr = TSGLRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSGLFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSGLFinalizePackage"
/*@C
  TSGLFinalizePackage - This function destroys everything in the TSGL package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSGLFinalizePackage(void) 
{
  PetscFunctionBegin;
  TSGLPackageInitialized = PETSC_FALSE;
  TSGLRegisterAllCalled  = PETSC_FALSE;
  TSGLList               = PETSC_NULL;
  TSGLAcceptList         = PETSC_NULL;
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

  Options database keys:
+  -ts_gl_type <type> - the class of general linear method (irks)
.  -ts_gl_rtol <tol>  - relative error
.  -ts_gl_atol <tol>  - absolute error
.  -ts_gl_min_order <p> - minimum order method to consider (default=1)
.  -ts_gl_max_order <p> - maximum order method to consider (default=3)
.  -ts_gl_start_order <p> - order of starting method (default=1)
.  -ts_gl_complete <method> - method to use for completing the step (rescale-and-modify or rescale)
-  -ts_adapt_type <method> - adaptive controller to use (none step both)

  Notes:
  This integrator can be applied to DAE.

  Diagonally implicit general linear (DIGL) methods are a generalization of diagonally implicit Runge-Kutta (DIRK).
  They are represented by the tableau

.vb
  A  |  U
  -------
  B  |  V
.ve

  combined with a vector c of abscissa.  "Diagonally implicit" means that A is lower triangular.
  A step of the general method reads

.vb
  [ Y ] = [A  U] [  Y'   ]
  [X^k] = [B  V] [X^{k-1}]
.ve

  where Y is the multivector of stage values, Y' is the multivector of stage derivatives, X^k is the Nordsieck vector of
  the solution at step k.  The Nordsieck vector consists of the first r moments of the solution, given by

.vb
  X = [x_0,x_1,...,x_{r-1}] = [x, h x', h^2 x'', ..., h^{r-1} x^{(r-1)} ]
.ve

  If A is lower triangular, we can solve the stages (Y,Y') sequentially

.vb
  y_i = h sum_{j=0}^{s-1} (a_ij y'_j) + sum_{j=0}^{r-1} u_ij x_j,    i=0,...,{s-1}
.ve

  and then construct the pieces to carry to the next step

.vb
  xx_i = h sum_{j=0}^{s-1} b_ij y'_j  + sum_{j=0}^{r-1} v_ij x_j,    i=0,...,{r-1}
.ve

  Note that when the equations are cast in implicit form, we are using the stage equation to define y'_i
  in terms of y_i and known stuff (y_j for j<i and x_j for all j).


  Error estimation

  At present, the most attractive GL methods for stiff problems are singly diagonally implicit schemes which posses
  Inherent Runge-Kutta Stability (IRKS).  These methods have r=s, the number of items passed between steps is equal to
  the number of stages.  The order and stage-order are one less than the number of stages.  We use the error estimates
  in the 2007 paper which provide the following estimates

.vb
  h^{p+1} X^{(p+1)}          = phi_0^T Y' + [0 psi_0^T] Xold
  h^{p+2} X^{(p+2)}          = phi_1^T Y' + [0 psi_1^T] Xold
  h^{p+2} (dx'/dx) X^{(p+1)} = phi_2^T Y' + [0 psi_2^T] Xold
.ve

  These estimates are accurate to O(h^{p+3}).

  Changing the step size

  We use the generalized "rescale and modify" scheme, see equation (4.5) of the 2007 paper.

  Level: beginner

  References:
  John Butcher and Z. Jackieweicz and W. Wright, On error propagation in general linear methods for
  ordinary differential equations, Journal of Complexity, Vol 23 (4-6), 2007.

  John Butcher, Numerical methods for ordinary differential equations, second edition, Wiley, 2009.

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

  ts->problem_type = TS_NONLINEAR;
  ierr = SNESCreate(((PetscObject)ts)->comm,&ts->snes);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ts->snes,(PetscObject)ts,1);CHKERRQ(ierr);

  gl->max_step_rejections = 1;
  gl->min_order           = 1;
  gl->max_order           = 3;
  gl->start_order         = 1;
  gl->current_scheme      = -1;
  gl->extrapolate         = PETSC_FALSE;

  gl->wrms_atol = 1e-8;
  gl->wrms_rtol = 1e-5;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSGLSetType_C",      "TSGLSetType_GL",      &TSGLSetType_GL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSGLSetAcceptType_C","TSGLSetAcceptType_GL",&TSGLSetAcceptType_GL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSGLGetAdapt_C",     "TSGLGetAdapt_GL",     &TSGLGetAdapt_GL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
