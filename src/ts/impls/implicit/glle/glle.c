
#include <../src/ts/impls/implicit/glle/glle.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>
#include <petscblaslapack.h>

static const char        *TSGLLEErrorDirections[] = {"FORWARD","BACKWARD","TSGLLEErrorDirection","TSGLLEERROR_",NULL};
static PetscFunctionList TSGLLEList;
static PetscFunctionList TSGLLEAcceptList;
static PetscBool         TSGLLEPackageInitialized;
static PetscBool         TSGLLERegisterAllCalled;

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
  return PetscPowRealInt(PetscRealPart(c),p)/Factorial(p);
}

static PetscErrorCode TSGLLEGetVecs(TS ts,DM dm,Vec *Z,Vec *Ydotstage)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      CHKERRQ(DMGetNamedGlobalVector(dm,"TSGLLE_Z",Z));
    } else *Z = gl->Z;
  }
  if (Ydotstage) {
    if (dm && dm != ts->dm) {
      CHKERRQ(DMGetNamedGlobalVector(dm,"TSGLLE_Ydot",Ydotstage));
    } else *Ydotstage = gl->Ydot[gl->stage];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLERestoreVecs(TS ts,DM dm,Vec *Z,Vec *Ydotstage)
{
  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      CHKERRQ(DMRestoreNamedGlobalVector(dm,"TSGLLE_Z",Z));
    }
  }
  if (Ydotstage) {

    if (dm && dm != ts->dm) {
      CHKERRQ(DMRestoreNamedGlobalVector(dm,"TSGLLE_Ydot",Ydotstage));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSGLLE(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSGLLE(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  TS             ts = (TS)ctx;
  Vec            Ydot,Ydot_c;

  PetscFunctionBegin;
  CHKERRQ(TSGLLEGetVecs(ts,fine,NULL,&Ydot));
  CHKERRQ(TSGLLEGetVecs(ts,coarse,NULL,&Ydot_c));
  CHKERRQ(MatRestrict(restrct,Ydot,Ydot_c));
  CHKERRQ(VecPointwiseMult(Ydot_c,rscale,Ydot_c));
  CHKERRQ(TSGLLERestoreVecs(ts,fine,NULL,&Ydot));
  CHKERRQ(TSGLLERestoreVecs(ts,coarse,NULL,&Ydot_c));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_TSGLLE(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_TSGLLE(DM dm,VecScatter gscat, VecScatter lscat,DM subdm,void *ctx)
{
  TS             ts = (TS)ctx;
  Vec            Ydot,Ydot_s;

  PetscFunctionBegin;
  CHKERRQ(TSGLLEGetVecs(ts,dm,NULL,&Ydot));
  CHKERRQ(TSGLLEGetVecs(ts,subdm,NULL,&Ydot_s));

  CHKERRQ(VecScatterBegin(gscat,Ydot,Ydot_s,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(gscat,Ydot,Ydot_s,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(TSGLLERestoreVecs(ts,dm,NULL,&Ydot));
  CHKERRQ(TSGLLERestoreVecs(ts,subdm,NULL,&Ydot_s));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLESchemeCreate(PetscInt p,PetscInt q,PetscInt r,PetscInt s,const PetscScalar *c,
                                       const PetscScalar *a,const PetscScalar *b,const PetscScalar *u,const PetscScalar *v,TSGLLEScheme *inscheme)
{
  TSGLLEScheme     scheme;
  PetscInt       j;

  PetscFunctionBegin;
  PetscCheckFalse(p < 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scheme order must be positive");
  PetscCheckFalse(r < 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"At least one item must be carried between steps");
  PetscCheckFalse(s < 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"At least one stage is required");
  PetscValidPointer(inscheme,10);
  *inscheme = NULL;
  CHKERRQ(PetscNew(&scheme));
  scheme->p = p;
  scheme->q = q;
  scheme->r = r;
  scheme->s = s;

  CHKERRQ(PetscMalloc5(s,&scheme->c,s*s,&scheme->a,r*s,&scheme->b,r*s,&scheme->u,r*r,&scheme->v));
  CHKERRQ(PetscArraycpy(scheme->c,c,s));
  for (j=0; j<s*s; j++) scheme->a[j] = (PetscAbsScalar(a[j]) < 1e-12) ? 0 : a[j];
  for (j=0; j<r*s; j++) scheme->b[j] = (PetscAbsScalar(b[j]) < 1e-12) ? 0 : b[j];
  for (j=0; j<s*r; j++) scheme->u[j] = (PetscAbsScalar(u[j]) < 1e-12) ? 0 : u[j];
  for (j=0; j<r*r; j++) scheme->v[j] = (PetscAbsScalar(v[j]) < 1e-12) ? 0 : v[j];

  CHKERRQ(PetscMalloc6(r,&scheme->alpha,r,&scheme->beta,r,&scheme->gamma,3*s,&scheme->phi,3*r,&scheme->psi,r,&scheme->stage_error));
  {
    PetscInt     i,j,k,ss=s+2;
    PetscBLASInt m,n,one=1,*ipiv,lwork=4*((s+3)*3+3),info,ldb;
    PetscReal    rcond,*sing,*workreal;
    PetscScalar  *ImV,*H,*bmat,*workscalar,*c=scheme->c,*a=scheme->a,*b=scheme->b,*u=scheme->u,*v=scheme->v;
    PetscBLASInt rank;
    CHKERRQ(PetscMalloc7(PetscSqr(r),&ImV,3*s,&H,3*ss,&bmat,lwork,&workscalar,5*(3+r),&workreal,r+s,&sing,r+s,&ipiv));

    /* column-major input */
    for (i=0; i<r-1; i++) {
      for (j=0; j<r-1; j++) ImV[i+j*r] = 1.0*(i==j) - v[(i+1)*r+j+1];
    }
    /* Build right hand side for alpha (tp - glm.B(2:end,:)*(glm.c.^(p)./factorial(p))) */
    for (i=1; i<r; i++) {
      scheme->alpha[i] = 1./Factorial(p+1-i);
      for (j=0; j<s; j++) scheme->alpha[i] -= b[i*s+j]*CPowF(c[j],p);
    }
    CHKERRQ(PetscBLASIntCast(r-1,&m));
    CHKERRQ(PetscBLASIntCast(r,&n));
    PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&m,&one,ImV,&n,ipiv,scheme->alpha+1,&n,&info));
    PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GESV");
    PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

    /* Build right hand side for beta (tp1 - glm.B(2:end,:)*(glm.c.^(p+1)./factorial(p+1)) - e.alpha) */
    for (i=1; i<r; i++) {
      scheme->beta[i] = 1./Factorial(p+2-i) - scheme->alpha[i];
      for (j=0; j<s; j++) scheme->beta[i] -= b[i*s+j]*CPowF(c[j],p+1);
    }
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("No transpose",&m,&one,ImV,&n,ipiv,scheme->beta+1,&n,&info));
    PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GETRS");
    PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Should not happen");

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
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("No transpose",&m,&one,ImV,&n,ipiv,scheme->gamma+1,&n,&info));
    PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GETRS");
    PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Should not happen");

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
    m     = 3;
    CHKERRQ(PetscBLASIntCast(s,&n));
    CHKERRQ(PetscBLASIntCast(ss,&ldb));
    rcond = 1e-12;
#if defined(PETSC_USE_COMPLEX)
    /* ZGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, RWORK, INFO) */
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&m,&n,&m,H,&m,bmat,&ldb,sing,&rcond,&rank,workscalar,&lwork,workreal,&info));
#else
    /* DGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, INFO) */
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&m,&n,&m,H,&m,bmat,&ldb,sing,&rcond,&rank,workscalar,&lwork,&info));
#endif
    PetscCheckFalse(info < 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GELSS");
    PetscCheckFalse(info > 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"SVD failed to converge");

    for (j=0; j<3; j++) {
      for (k=0; k<s; k++) scheme->phi[k+j*s] = bmat[k+j*ss];
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
    CHKERRQ(PetscFree7(ImV,H,bmat,workscalar,workreal,sing,ipiv));
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

static PetscErrorCode TSGLLESchemeDestroy(TSGLLEScheme sc)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree5(sc->c,sc->a,sc->b,sc->u,sc->v));
  CHKERRQ(PetscFree6(sc->alpha,sc->beta,sc->gamma,sc->phi,sc->psi,sc->stage_error));
  CHKERRQ(PetscFree(sc));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEDestroy_Default(TS_GLLE *gl)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<gl->nschemes; i++) {
    if (gl->schemes[i]) CHKERRQ(TSGLLESchemeDestroy(gl->schemes[i]));
  }
  CHKERRQ(PetscFree(gl->schemes));
  gl->nschemes = 0;
  CHKERRQ(PetscMemzero(gl->type_name,sizeof(gl->type_name)));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEViewTable_Private(PetscViewer viewer,PetscInt m,PetscInt n,const PetscScalar a[],const char name[])
{
  PetscBool      iascii;
  PetscInt       i,j;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"%30s = [",name));
    for (i=0; i<m; i++) {
      if (i) CHKERRQ(PetscViewerASCIIPrintf(viewer,"%30s   [",""));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      for (j=0; j<n; j++) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer," %12.8g",PetscRealPart(a[i*n+j])));
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"]\n"));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLESchemeView(TSGLLEScheme sc,PetscBool view_details,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"GL scheme p,q,r,s = %d,%d,%d,%d\n",sc->p,sc->q,sc->r,sc->s));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Stiffly accurate: %s,  FSAL: %s\n",sc->stiffly_accurate ? "yes" : "no",sc->fsal ? "yes" : "no"));
    ierr = PetscViewerASCIIPrintf(viewer,"Leading error constants: %10.3e  %10.3e  %10.3e\n",
                                  PetscRealPart(sc->alpha[0]),PetscRealPart(sc->beta[0]),PetscRealPart(sc->gamma[0]));CHKERRQ(ierr);
    CHKERRQ(TSGLLEViewTable_Private(viewer,1,sc->s,sc->c,"Abscissas c"));
    if (view_details) {
      CHKERRQ(TSGLLEViewTable_Private(viewer,sc->s,sc->s,sc->a,"A"));
      CHKERRQ(TSGLLEViewTable_Private(viewer,sc->r,sc->s,sc->b,"B"));
      CHKERRQ(TSGLLEViewTable_Private(viewer,sc->s,sc->r,sc->u,"U"));
      CHKERRQ(TSGLLEViewTable_Private(viewer,sc->r,sc->r,sc->v,"V"));

      CHKERRQ(TSGLLEViewTable_Private(viewer,3,sc->s,sc->phi,"Error estimate phi"));
      CHKERRQ(TSGLLEViewTable_Private(viewer,3,sc->r,sc->psi,"Error estimate psi"));
      CHKERRQ(TSGLLEViewTable_Private(viewer,1,sc->r,sc->alpha,"Modify alpha"));
      CHKERRQ(TSGLLEViewTable_Private(viewer,1,sc->r,sc->beta,"Modify beta"));
      CHKERRQ(TSGLLEViewTable_Private(viewer,1,sc->r,sc->gamma,"Modify gamma"));
      CHKERRQ(TSGLLEViewTable_Private(viewer,1,sc->s,sc->stage_error,"Stage error xi"));
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEEstimateHigherMoments_Default(TSGLLEScheme sc,PetscReal h,Vec Ydot[],Vec Xold[],Vec hm[])
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckFalse(sc->r > 64 || sc->s > 64,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Ridiculous number of stages or items passed between stages");
  /* build error vectors*/
  for (i=0; i<3; i++) {
    PetscScalar phih[64];
    PetscInt    j;
    for (j=0; j<sc->s; j++) phih[j] = sc->phi[i*sc->s+j]*h;
    CHKERRQ(VecZeroEntries(hm[i]));
    CHKERRQ(VecMAXPY(hm[i],sc->s,phih,Ydot));
    CHKERRQ(VecMAXPY(hm[i],sc->r,&sc->psi[i*sc->r],Xold));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLECompleteStep_Rescale(TSGLLEScheme sc,PetscReal h,TSGLLEScheme next_sc,PetscReal next_h,Vec Ydot[],Vec Xold[],Vec X[])
{
  PetscScalar    brow[32],vrow[32];
  PetscInt       i,j,r,s;

  PetscFunctionBegin;
  /* Build the new solution from (X,Ydot) */
  r = sc->r;
  s = sc->s;
  for (i=0; i<r; i++) {
    CHKERRQ(VecZeroEntries(X[i]));
    for (j=0; j<s; j++) brow[j] = h*sc->b[i*s+j];
    CHKERRQ(VecMAXPY(X[i],s,brow,Ydot));
    for (j=0; j<r; j++) vrow[j] = sc->v[i*r+j];
    CHKERRQ(VecMAXPY(X[i],r,vrow,Xold));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLECompleteStep_RescaleAndModify(TSGLLEScheme sc,PetscReal h,TSGLLEScheme next_sc,PetscReal next_h,Vec Ydot[],Vec Xold[],Vec X[])
{
  PetscScalar    brow[32],vrow[32];
  PetscReal      ratio;
  PetscInt       i,j,p,r,s;

  PetscFunctionBegin;
  /* Build the new solution from (X,Ydot) */
  p     = sc->p;
  r     = sc->r;
  s     = sc->s;
  ratio = next_h/h;
  for (i=0; i<r; i++) {
    CHKERRQ(VecZeroEntries(X[i]));
    for (j=0; j<s; j++) {
      brow[j] = h*(PetscPowRealInt(ratio,i)*sc->b[i*s+j]
                   + (PetscPowRealInt(ratio,i) - PetscPowRealInt(ratio,p+1))*(+ sc->alpha[i]*sc->phi[0*s+j])
                   + (PetscPowRealInt(ratio,i) - PetscPowRealInt(ratio,p+2))*(+ sc->beta [i]*sc->phi[1*s+j]
                                                                              + sc->gamma[i]*sc->phi[2*s+j]));
    }
    CHKERRQ(VecMAXPY(X[i],s,brow,Ydot));
    for (j=0; j<r; j++) {
      vrow[j] = (PetscPowRealInt(ratio,i)*sc->v[i*r+j]
                 + (PetscPowRealInt(ratio,i) - PetscPowRealInt(ratio,p+1))*(+ sc->alpha[i]*sc->psi[0*r+j])
                 + (PetscPowRealInt(ratio,i) - PetscPowRealInt(ratio,p+2))*(+ sc->beta [i]*sc->psi[1*r+j]
                                                                            + sc->gamma[i]*sc->psi[2*r+j]));
    }
    CHKERRQ(VecMAXPY(X[i],r,vrow,Xold));
  }
  if (r < next_sc->r) {
    PetscCheckFalse(r+1 != next_sc->r,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot accommodate jump in r greater than 1");
    CHKERRQ(VecZeroEntries(X[r]));
    for (j=0; j<s; j++) brow[j] = h*PetscPowRealInt(ratio,p+1)*sc->phi[0*s+j];
    CHKERRQ(VecMAXPY(X[r],s,brow,Ydot));
    for (j=0; j<r; j++) vrow[j] = PetscPowRealInt(ratio,p+1)*sc->psi[0*r+j];
    CHKERRQ(VecMAXPY(X[r],r,vrow,Xold));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLECreate_IRKS(TS ts)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;

  PetscFunctionBegin;
  gl->Destroy               = TSGLLEDestroy_Default;
  gl->EstimateHigherMoments = TSGLLEEstimateHigherMoments_Default;
  gl->CompleteStep          = TSGLLECompleteStep_RescaleAndModify;
  CHKERRQ(PetscMalloc1(10,&gl->schemes));
  gl->nschemes = 0;

  {
    /* p=1,q=1, r=s=2, A- and L-stable with error estimates of order 2 and 3
    * Listed in Butcher & Podhaisky 2006. On error estimation in general linear methods for stiff ODE.
    * irks(0.3,0,[.3,1],[1],1)
    * Note: can be made to have classical order (not stage order) 2 by replacing 0.3 with 1-sqrt(1/2)
    * but doing so would sacrifice the error estimator.
    */
    const PetscScalar c[2]    = {3./10., 1.};
    const PetscScalar a[2][2] = {{3./10., 0}, {7./10., 3./10.}};
    const PetscScalar b[2][2] = {{7./10., 3./10.}, {0,1}};
    const PetscScalar u[2][2] = {{1,0},{1,0}};
    const PetscScalar v[2][2] = {{1,0},{0,0}};
    CHKERRQ(TSGLLESchemeCreate(1,1,2,2,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]));
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
    CHKERRQ(TSGLLESchemeCreate(2,2,3,3,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]));
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
    CHKERRQ(TSGLLESchemeCreate(3,3,4,4,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]));
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
    CHKERRQ(TSGLLESchemeCreate(4,4,5,5,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]));
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
    CHKERRQ(TSGLLESchemeCreate(5,5,6,6,c,*a,*b,*u,*v,&gl->schemes[gl->nschemes++]));
  }
  PetscFunctionReturn(0);
}

/*@C
   TSGLLESetType - sets the class of general linear method to use for time-stepping

   Collective on TS

   Input Parameters:
+  ts - the TS context
-  type - a method

   Options Database Key:
.  -ts_gl_type <type> - sets the method, use -help for a list of available method (e.g. irks)

   Notes:
   See "petsc/include/petscts.h" for available methods (for instance)
.    TSGLLE_IRKS - Diagonally implicit methods with inherent Runge-Kutta stability (for stiff problems)

   Normally, it is best to use the TSSetFromOptions() command and
   then set the TSGLLE type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the many different solvers.
   The TSGLLESetType() routine is provided for those situations where it
   is necessary to set the timestepping solver independently of the
   command line or options database.  This might be the case, for example,
   when the choice of solver changes during the execution of the
   program, and the user's application is taking responsibility for
   choosing the appropriate method.

   Level: intermediate

@*/
PetscErrorCode  TSGLLESetType(TS ts,TSGLLEType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(type,2);
  CHKERRQ(PetscTryMethod(ts,"TSGLLESetType_C",(TS,TSGLLEType),(ts,type)));
  PetscFunctionReturn(0);
}

/*@C
   TSGLLESetAcceptType - sets the acceptance test

   Time integrators that need to control error must have the option to reject a time step based on local error
   estimates.  This function allows different schemes to be set.

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context
-  type - the type

   Options Database Key:
.  -ts_gl_accept_type <type> - sets the method used to determine whether to accept or reject a step

   Level: intermediate

.seealso: TS, TSGLLE, TSGLLEAcceptRegister(), TSGLLEAdapt, set type
@*/
PetscErrorCode  TSGLLESetAcceptType(TS ts,TSGLLEAcceptType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(type,2);
  CHKERRQ(PetscTryMethod(ts,"TSGLLESetAcceptType_C",(TS,TSGLLEAcceptType),(ts,type)));
  PetscFunctionReturn(0);
}

/*@C
   TSGLLEGetAdapt - gets the TSGLLEAdapt object from the TS

   Not Collective

   Input Parameter:
.  ts - the TS context

   Output Parameter:
.  adapt - the TSGLLEAdapt context

   Notes:
   This allows the user set options on the TSGLLEAdapt object.  Usually it is better to do this using the options
   database, so this function is rarely needed.

   Level: advanced

.seealso: TSGLLEAdapt, TSGLLEAdaptRegister()
@*/
PetscErrorCode  TSGLLEGetAdapt(TS ts,TSGLLEAdapt *adapt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(adapt,2);
  CHKERRQ(PetscUseMethod(ts,"TSGLLEGetAdapt_C",(TS,TSGLLEAdapt*),(ts,adapt)));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEAccept_Always(TS ts,PetscReal tleft,PetscReal h,const PetscReal enorms[],PetscBool  *accept)
{
  PetscFunctionBegin;
  *accept = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEUpdateWRMS(TS ts)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  PetscScalar    *x,*w;
  PetscInt       n,i;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(gl->X[0],&x));
  CHKERRQ(VecGetArray(gl->W,&w));
  CHKERRQ(VecGetLocalSize(gl->W,&n));
  for (i=0; i<n; i++) w[i] = 1./(gl->wrms_atol + gl->wrms_rtol*PetscAbsScalar(x[i]));
  CHKERRQ(VecRestoreArray(gl->X[0],&x));
  CHKERRQ(VecRestoreArray(gl->W,&w));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEVecNormWRMS(TS ts,Vec X,PetscReal *nrm)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  PetscScalar    *x,*w;
  PetscReal      sum = 0.0,gsum;
  PetscInt       n,N,i;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(X,&x));
  CHKERRQ(VecGetArray(gl->W,&w));
  CHKERRQ(VecGetLocalSize(gl->W,&n));
  for (i=0; i<n; i++) sum += PetscAbsScalar(PetscSqr(x[i]*w[i]));
  CHKERRQ(VecRestoreArray(X,&x));
  CHKERRQ(VecRestoreArray(gl->W,&w));
  CHKERRMPI(MPIU_Allreduce(&sum,&gsum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)ts)));
  CHKERRQ(VecGetSize(gl->W,&N));
  *nrm = PetscSqrtReal(gsum/(1.*N));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLESetType_GLLE(TS ts,TSGLLEType type)
{
  PetscBool      same;
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  PetscErrorCode (*r)(TS);

  PetscFunctionBegin;
  if (gl->type_name[0]) {
    CHKERRQ(PetscStrcmp(gl->type_name,type,&same));
    if (same) PetscFunctionReturn(0);
    CHKERRQ((*gl->Destroy)(gl));
  }

  CHKERRQ(PetscFunctionListFind(TSGLLEList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSGLLE type \"%s\" given",type);
  CHKERRQ((*r)(ts));
  CHKERRQ(PetscStrcpy(gl->type_name,type));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLESetAcceptType_GLLE(TS ts,TSGLLEAcceptType type)
{
  TSGLLEAcceptFunction r;
  TS_GLLE              *gl = (TS_GLLE*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListFind(TSGLLEAcceptList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSGLLEAccept type \"%s\" given",type);
  gl->Accept = r;
  CHKERRQ(PetscStrncpy(gl->accept_name,type,sizeof(gl->accept_name)));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEGetAdapt_GLLE(TS ts,TSGLLEAdapt *adapt)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;

  PetscFunctionBegin;
  if (!gl->adapt) {
    CHKERRQ(TSGLLEAdaptCreate(PetscObjectComm((PetscObject)ts),&gl->adapt));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)gl->adapt,(PetscObject)ts,1));
    CHKERRQ(PetscLogObjectParent((PetscObject)ts,(PetscObject)gl->adapt));
  }
  *adapt = gl->adapt;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEChooseNextScheme(TS ts,PetscReal h,const PetscReal hmnorm[],PetscInt *next_scheme,PetscReal *next_h,PetscBool  *finish)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  PetscInt       i,n,cur_p,cur,next_sc,candidates[64],orders[64];
  PetscReal      errors[64],costs[64],tleft;

  PetscFunctionBegin;
  cur   = -1;
  cur_p = gl->schemes[gl->current_scheme]->p;
  tleft = ts->max_time - (ts->ptime + ts->time_step);
  for (i=0,n=0; i<gl->nschemes; i++) {
    TSGLLEScheme sc = gl->schemes[i];
    if (sc->p < gl->min_order || gl->max_order < sc->p) continue;
    if (sc->p == cur_p - 1)    errors[n] = PetscAbsScalar(sc->alpha[0])*hmnorm[0];
    else if (sc->p == cur_p)   errors[n] = PetscAbsScalar(sc->alpha[0])*hmnorm[1];
    else if (sc->p == cur_p+1) errors[n] = PetscAbsScalar(sc->alpha[0])*(hmnorm[2]+hmnorm[3]);
    else continue;
    candidates[n] = i;
    orders[n]     = PetscMin(sc->p,sc->q); /* order of global truncation error */
    costs[n]      = sc->s;                 /* estimate the cost as the number of stages */
    if (i == gl->current_scheme) cur = n;
    n++;
  }
  PetscCheckFalse(cur < 0 || gl->nschemes <= cur,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Current scheme not found in scheme list");
  CHKERRQ(TSGLLEAdaptChoose(gl->adapt,n,orders,errors,costs,cur,h,tleft,&next_sc,next_h,finish));
  *next_scheme = candidates[next_sc];
  CHKERRQ(PetscInfo(ts,"Adapt chose scheme %d (%d,%d,%d,%d) with step size %6.2e, finish=%d\n",*next_scheme,gl->schemes[*next_scheme]->p,gl->schemes[*next_scheme]->q,gl->schemes[*next_scheme]->r,gl->schemes[*next_scheme]->s,*next_h,*finish));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGLLEGetMaxSizes(TS ts,PetscInt *max_r,PetscInt *max_s)
{
  TS_GLLE *gl = (TS_GLLE*)ts->data;

  PetscFunctionBegin;
  *max_r = gl->schemes[gl->nschemes-1]->r;
  *max_s = gl->schemes[gl->nschemes-1]->s;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSolve_GLLE(TS ts)
{
  TS_GLLE             *gl = (TS_GLLE*)ts->data;
  PetscInt            i,k,its,lits,max_r,max_s;
  PetscBool           final_step,finish;
  SNESConvergedReason snesreason;

  PetscFunctionBegin;
  CHKERRQ(TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol));

  CHKERRQ(TSGLLEGetMaxSizes(ts,&max_r,&max_s));
  CHKERRQ(VecCopy(ts->vec_sol,gl->X[0]));
  for (i=1; i<max_r; i++) {
    CHKERRQ(VecZeroEntries(gl->X[i]));
  }
  CHKERRQ(TSGLLEUpdateWRMS(ts));

  if (0) {
    /* Find consistent initial data for DAE */
    gl->stage_time = ts->ptime + ts->time_step;
    gl->scoeff = 1.;
    gl->stage  = 0;

    CHKERRQ(VecCopy(ts->vec_sol,gl->Z));
    CHKERRQ(VecCopy(ts->vec_sol,gl->Y));
    CHKERRQ(SNESSolve(ts->snes,NULL,gl->Y));
    CHKERRQ(SNESGetIterationNumber(ts->snes,&its));
    CHKERRQ(SNESGetLinearSolveIterations(ts->snes,&lits));
    CHKERRQ(SNESGetConvergedReason(ts->snes,&snesreason));

    ts->snes_its += its; ts->ksp_its += lits;
    if (snesreason < 0 && ts->max_snes_failures > 0 && ++ts->num_snes_failures >= ts->max_snes_failures) {
      ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
      CHKERRQ(PetscInfo(ts,"Step=%D, nonlinear solve solve failures %D greater than current TS allowed, stopping solve\n",ts->steps,ts->num_snes_failures));
      PetscFunctionReturn(0);
    }
  }

  PetscCheckFalse(gl->current_scheme < 0,PETSC_COMM_SELF,PETSC_ERR_ORDER,"A starting scheme has not been provided");

  for (k=0,final_step=PETSC_FALSE,finish=PETSC_FALSE; k<ts->max_steps && !finish; k++) {
    PetscInt          j,r,s,next_scheme = 0,rejections;
    PetscReal         h,hmnorm[4],enorm[3],next_h;
    PetscBool         accept;
    const PetscScalar *c,*a,*u;
    Vec               *X,*Ydot,Y;
    TSGLLEScheme        scheme = gl->schemes[gl->current_scheme];

    r = scheme->r; s = scheme->s;
    c = scheme->c;
    a = scheme->a; u = scheme->u;
    h = ts->time_step;
    X = gl->X; Ydot = gl->Ydot; Y = gl->Y;

    if (ts->ptime > ts->max_time) break;

    /*
      We only call PreStep at the start of each STEP, not each STAGE.  This is because it is
      possible to fail (have to restart a step) after multiple stages.
    */
    CHKERRQ(TSPreStep(ts));

    rejections = 0;
    while (1) {
      for (i=0; i<s; i++) {
        PetscScalar shift;
        gl->scoeff     = 1./PetscRealPart(a[i*s+i]);
        shift          = gl->scoeff/ts->time_step;
        gl->stage      = i;
        gl->stage_time = ts->ptime + PetscRealPart(c[i])*h;

        /*
        * Stage equation: Y = h A Y' + U X
        * We assume that A is lower-triangular so that we can solve the stages (Y,Y') sequentially
        * Build the affine vector z_i = -[1/(h a_ii)](h sum_j a_ij y'_j + sum_j u_ij x_j)
        * Then y'_i = z + 1/(h a_ii) y_i
        */
        CHKERRQ(VecZeroEntries(gl->Z));
        for (j=0; j<r; j++) {
          CHKERRQ(VecAXPY(gl->Z,-shift*u[i*r+j],X[j]));
        }
        for (j=0; j<i; j++) {
          CHKERRQ(VecAXPY(gl->Z,-shift*h*a[i*s+j],Ydot[j]));
        }
        /* Note: Z is used within function evaluation, Ydot = Z + shift*Y */

        /* Compute an estimate of Y to start Newton iteration */
        if (gl->extrapolate) {
          if (i==0) {
            /* Linear extrapolation on the first stage */
            CHKERRQ(VecWAXPY(Y,c[i]*h,X[1],X[0]));
          } else {
            /* Linear extrapolation from the last stage */
            CHKERRQ(VecAXPY(Y,(c[i]-c[i-1])*h,Ydot[i-1]));
          }
        } else if (i==0) {        /* Directly use solution from the last step, otherwise reuse the last stage (do nothing) */
          CHKERRQ(VecCopy(X[0],Y));
        }

        /* Solve this stage (Ydot[i] is computed during function evaluation) */
        CHKERRQ(SNESSolve(ts->snes,NULL,Y));
        CHKERRQ(SNESGetIterationNumber(ts->snes,&its));
        CHKERRQ(SNESGetLinearSolveIterations(ts->snes,&lits));
        CHKERRQ(SNESGetConvergedReason(ts->snes,&snesreason));
        ts->snes_its += its; ts->ksp_its += lits;
        if (snesreason < 0 && ts->max_snes_failures > 0 && ++ts->num_snes_failures >= ts->max_snes_failures) {
          ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
          CHKERRQ(PetscInfo(ts,"Step=%D, nonlinear solve solve failures %D greater than current TS allowed, stopping solve\n",ts->steps,ts->num_snes_failures));
          PetscFunctionReturn(0);
        }
      }

      gl->stage_time = ts->ptime + ts->time_step;

      CHKERRQ((*gl->EstimateHigherMoments)(scheme,h,Ydot,gl->X,gl->himom));
      /* hmnorm[i] = h^{p+i}x^{(p+i)} with i=0,1,2; hmnorm[3] = h^{p+2}(dx'/dx) x^{(p+1)} */
      for (i=0; i<3; i++) {
        CHKERRQ(TSGLLEVecNormWRMS(ts,gl->himom[i],&hmnorm[i+1]));
      }
      enorm[0] = PetscRealPart(scheme->alpha[0])*hmnorm[1];
      enorm[1] = PetscRealPart(scheme->beta[0]) *hmnorm[2];
      enorm[2] = PetscRealPart(scheme->gamma[0])*hmnorm[3];
      CHKERRQ((*gl->Accept)(ts,ts->max_time-gl->stage_time,h,enorm,&accept));
      if (accept) goto accepted;
      rejections++;
      CHKERRQ(PetscInfo(ts,"Step %D (t=%g) not accepted, rejections=%D\n",k,gl->stage_time,rejections));
      if (rejections > gl->max_step_rejections) break;
      /*
        There are lots of reasons why a step might be rejected, including solvers not converging and other factors that
        TSGLLEChooseNextScheme does not support.  Additionally, the error estimates may be very screwed up, so I'm not
        convinced that it's safe to just compute a new error estimate using the same interface as the current adaptor
        (the adaptor interface probably has to change).  Here we make an arbitrary and naive choice.  This assumes that
        steps were written in Nordsieck form.  The "correct" method would be to re-complete the previous time step with
        the correct "next" step size.  It is unclear to me whether the present ad-hoc method of rescaling X is stable.
      */
      h *= 0.5;
      for (i=1; i<scheme->r; i++) {
        CHKERRQ(VecScale(X[i],PetscPowRealInt(0.5,i)));
      }
    }
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"Time step %D (t=%g) not accepted after %D failures",k,gl->stage_time,rejections);

accepted:
    /* This term is not error, but it *would* be the leading term for a lower order method */
    CHKERRQ(TSGLLEVecNormWRMS(ts,gl->X[scheme->r-1],&hmnorm[0]));
    /* Correct scaling so that these are equivalent to norms of the Nordsieck vectors */

    CHKERRQ(PetscInfo(ts,"Last moment norm %10.2e, estimated error norms %10.2e %10.2e %10.2e\n",hmnorm[0],enorm[0],enorm[1],enorm[2]));
    if (!final_step) {
      CHKERRQ(TSGLLEChooseNextScheme(ts,h,hmnorm,&next_scheme,&next_h,&final_step));
    } else {
      /* Dummy values to complete the current step in a consistent manner */
      next_scheme = gl->current_scheme;
      next_h      = h;
      finish      = PETSC_TRUE;
    }

    X        = gl->Xold;
    gl->Xold = gl->X;
    gl->X    = X;
    CHKERRQ((*gl->CompleteStep)(scheme,h,gl->schemes[next_scheme],next_h,Ydot,gl->Xold,gl->X));

    CHKERRQ(TSGLLEUpdateWRMS(ts));

    /* Post the solution for the user, we could avoid this copy with a small bit of cleverness */
    CHKERRQ(VecCopy(gl->X[0],ts->vec_sol));
    ts->ptime += h;
    ts->steps++;

    CHKERRQ(TSPostEvaluate(ts));
    CHKERRQ(TSPostStep(ts));
    CHKERRQ(TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol));

    gl->current_scheme = next_scheme;
    ts->time_step      = next_h;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSReset_GLLE(TS ts)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  PetscInt       max_r,max_s;

  PetscFunctionBegin;
  if (gl->setupcalled) {
    CHKERRQ(TSGLLEGetMaxSizes(ts,&max_r,&max_s));
    CHKERRQ(VecDestroyVecs(max_r,&gl->Xold));
    CHKERRQ(VecDestroyVecs(max_r,&gl->X));
    CHKERRQ(VecDestroyVecs(max_s,&gl->Ydot));
    CHKERRQ(VecDestroyVecs(3,&gl->himom));
    CHKERRQ(VecDestroy(&gl->W));
    CHKERRQ(VecDestroy(&gl->Y));
    CHKERRQ(VecDestroy(&gl->Z));
  }
  gl->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_GLLE(TS ts)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(TSReset_GLLE(ts));
  if (ts->dm) {
    CHKERRQ(DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSGLLE,DMRestrictHook_TSGLLE,ts));
    CHKERRQ(DMSubDomainHookRemove(ts->dm,DMSubDomainHook_TSGLLE,DMSubDomainRestrictHook_TSGLLE,ts));
  }
  if (gl->adapt) CHKERRQ(TSGLLEAdaptDestroy(&gl->adapt));
  if (gl->Destroy) CHKERRQ((*gl->Destroy)(gl));
  CHKERRQ(PetscFree(ts->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSGLLESetType_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSGLLESetAcceptType_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSGLLEGetAdapt_C",NULL));
  PetscFunctionReturn(0);
}

/*
    This defines the nonlinear equation that is to be solved with SNES
    g(x) = f(t,x,z+shift*x) = 0
*/
static PetscErrorCode SNESTSFormFunction_GLLE(SNES snes,Vec x,Vec f,TS ts)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  Vec            Z,Ydot;
  DM             dm,dmsave;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(TSGLLEGetVecs(ts,dm,&Z,&Ydot));
  CHKERRQ(VecWAXPY(Ydot,gl->scoeff/ts->time_step,x,Z));
  dmsave = ts->dm;
  ts->dm = dm;
  CHKERRQ(TSComputeIFunction(ts,gl->stage_time,x,Ydot,f,PETSC_FALSE));
  ts->dm = dmsave;
  CHKERRQ(TSGLLERestoreVecs(ts,dm,&Z,&Ydot));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_GLLE(SNES snes,Vec x,Mat A,Mat B,TS ts)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  Vec            Z,Ydot;
  DM             dm,dmsave;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(TSGLLEGetVecs(ts,dm,&Z,&Ydot));
  dmsave = ts->dm;
  ts->dm = dm;
  /* gl->Xdot will have already been computed in SNESTSFormFunction_GLLE */
  CHKERRQ(TSComputeIJacobian(ts,gl->stage_time,x,gl->Ydot[gl->stage],gl->scoeff/ts->time_step,A,B,PETSC_FALSE));
  ts->dm = dmsave;
  CHKERRQ(TSGLLERestoreVecs(ts,dm,&Z,&Ydot));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_GLLE(TS ts)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  PetscInt       max_r,max_s;
  DM             dm;

  PetscFunctionBegin;
  gl->setupcalled = PETSC_TRUE;
  CHKERRQ(TSGLLEGetMaxSizes(ts,&max_r,&max_s));
  CHKERRQ(VecDuplicateVecs(ts->vec_sol,max_r,&gl->X));
  CHKERRQ(VecDuplicateVecs(ts->vec_sol,max_r,&gl->Xold));
  CHKERRQ(VecDuplicateVecs(ts->vec_sol,max_s,&gl->Ydot));
  CHKERRQ(VecDuplicateVecs(ts->vec_sol,3,&gl->himom));
  CHKERRQ(VecDuplicate(ts->vec_sol,&gl->W));
  CHKERRQ(VecDuplicate(ts->vec_sol,&gl->Y));
  CHKERRQ(VecDuplicate(ts->vec_sol,&gl->Z));

  /* Default acceptance tests and adaptivity */
  if (!gl->Accept) CHKERRQ(TSGLLESetAcceptType(ts,TSGLLEACCEPT_ALWAYS));
  if (!gl->adapt)  CHKERRQ(TSGLLEGetAdapt(ts,&gl->adapt));

  if (gl->current_scheme < 0) {
    PetscInt i;
    for (i=0;; i++) {
      if (gl->schemes[i]->p == gl->start_order) break;
      PetscCheckFalse(i+1 == gl->nschemes,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No schemes available with requested start order %d",i);
    }
    gl->current_scheme = i;
  }
  CHKERRQ(TSGetDM(ts,&dm));
  CHKERRQ(DMCoarsenHookAdd(dm,DMCoarsenHook_TSGLLE,DMRestrictHook_TSGLLE,ts));
  CHKERRQ(DMSubDomainHookAdd(dm,DMSubDomainHook_TSGLLE,DMSubDomainRestrictHook_TSGLLE,ts));
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetFromOptions_GLLE(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_GLLE        *gl        = (TS_GLLE*)ts->data;
  char           tname[256] = TSGLLE_IRKS,completef[256] = "rescale-and-modify";

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"General Linear ODE solver options"));
  {
    PetscBool flg;
    CHKERRQ(PetscOptionsFList("-ts_gl_type","Type of GL method","TSGLLESetType",TSGLLEList,gl->type_name[0] ? gl->type_name : tname,tname,sizeof(tname),&flg));
    if (flg || !gl->type_name[0]) {
      CHKERRQ(TSGLLESetType(ts,tname));
    }
    CHKERRQ(PetscOptionsInt("-ts_gl_max_step_rejections","Maximum number of times to attempt a step","None",gl->max_step_rejections,&gl->max_step_rejections,NULL));
    CHKERRQ(PetscOptionsInt("-ts_gl_max_order","Maximum order to try","TSGLLESetMaxOrder",gl->max_order,&gl->max_order,NULL));
    CHKERRQ(PetscOptionsInt("-ts_gl_min_order","Minimum order to try","TSGLLESetMinOrder",gl->min_order,&gl->min_order,NULL));
    CHKERRQ(PetscOptionsInt("-ts_gl_start_order","Initial order to try","TSGLLESetMinOrder",gl->start_order,&gl->start_order,NULL));
    CHKERRQ(PetscOptionsEnum("-ts_gl_error_direction","Which direction to look when estimating error","TSGLLESetErrorDirection",TSGLLEErrorDirections,(PetscEnum)gl->error_direction,(PetscEnum*)&gl->error_direction,NULL));
    CHKERRQ(PetscOptionsBool("-ts_gl_extrapolate","Extrapolate stage solution from previous solution (sometimes unstable)","TSGLLESetExtrapolate",gl->extrapolate,&gl->extrapolate,NULL));
    CHKERRQ(PetscOptionsReal("-ts_gl_atol","Absolute tolerance","TSGLLESetTolerances",gl->wrms_atol,&gl->wrms_atol,NULL));
    CHKERRQ(PetscOptionsReal("-ts_gl_rtol","Relative tolerance","TSGLLESetTolerances",gl->wrms_rtol,&gl->wrms_rtol,NULL));
    CHKERRQ(PetscOptionsString("-ts_gl_complete","Method to use for completing the step","none",completef,completef,sizeof(completef),&flg));
    if (flg) {
      PetscBool match1,match2;
      CHKERRQ(PetscStrcmp(completef,"rescale",&match1));
      CHKERRQ(PetscStrcmp(completef,"rescale-and-modify",&match2));
      if (match1)      gl->CompleteStep = TSGLLECompleteStep_Rescale;
      else if (match2) gl->CompleteStep = TSGLLECompleteStep_RescaleAndModify;
      else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"%s",completef);
    }
    {
      char type[256] = TSGLLEACCEPT_ALWAYS;
      CHKERRQ(PetscOptionsFList("-ts_gl_accept_type","Method to use for determining whether to accept a step","TSGLLESetAcceptType",TSGLLEAcceptList,gl->accept_name[0] ? gl->accept_name : type,type,sizeof(type),&flg));
      if (flg || !gl->accept_name[0]) {
        CHKERRQ(TSGLLESetAcceptType(ts,type));
      }
    }
    {
      TSGLLEAdapt adapt;
      CHKERRQ(TSGLLEGetAdapt(ts,&adapt));
      CHKERRQ(TSGLLEAdaptSetFromOptions(PetscOptionsObject,adapt));
    }
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_GLLE(TS ts,PetscViewer viewer)
{
  TS_GLLE        *gl = (TS_GLLE*)ts->data;
  PetscInt       i;
  PetscBool      iascii,details;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  min order %D, max order %D, current order %D\n",gl->min_order,gl->max_order,gl->schemes[gl->current_scheme]->p));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Error estimation: %s\n",TSGLLEErrorDirections[gl->error_direction]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Extrapolation: %s\n",gl->extrapolate ? "yes" : "no"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Acceptance test: %s\n",gl->accept_name[0] ? gl->accept_name : "(not yet set)"));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(TSGLLEAdaptView(gl->adapt,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  type: %s\n",gl->type_name[0] ? gl->type_name : "(not yet set)"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Schemes within family (%d):\n",gl->nschemes));
    details = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(((PetscObject)ts)->options,((PetscObject)ts)->prefix,"-ts_gl_view_detailed",&details,NULL));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    for (i=0; i<gl->nschemes; i++) {
      CHKERRQ(TSGLLESchemeView(gl->schemes[i],details,viewer));
    }
    if (gl->View) {
      CHKERRQ((*gl->View)(gl,viewer));
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
   TSGLLERegister -  adds a TSGLLE implementation

   Not Collective

   Input Parameters:
+  name_scheme - name of user-defined general linear scheme
-  routine_create - routine to create method context

   Notes:
   TSGLLERegister() may be called multiple times to add several user-defined families.

   Sample usage:
.vb
   TSGLLERegister("my_scheme",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSGLLESetType(ts,"my_scheme")
   or at runtime via the option
$     -ts_gl_type my_scheme

   Level: advanced

.seealso: TSGLLERegisterAll()
@*/
PetscErrorCode  TSGLLERegister(const char sname[],PetscErrorCode (*function)(TS))
{
  PetscFunctionBegin;
  CHKERRQ(TSGLLEInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&TSGLLEList,sname,function));
  PetscFunctionReturn(0);
}

/*@C
   TSGLLEAcceptRegister -  adds a TSGLLE acceptance scheme

   Not Collective

   Input Parameters:
+  name_scheme - name of user-defined acceptance scheme
-  routine_create - routine to create method context

   Notes:
   TSGLLEAcceptRegister() may be called multiple times to add several user-defined families.

   Sample usage:
.vb
   TSGLLEAcceptRegister("my_scheme",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSGLLESetAcceptType(ts,"my_scheme")
   or at runtime via the option
$     -ts_gl_accept_type my_scheme

   Level: advanced

.seealso: TSGLLERegisterAll()
@*/
PetscErrorCode  TSGLLEAcceptRegister(const char sname[],TSGLLEAcceptFunction function)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListAdd(&TSGLLEAcceptList,sname,function));
  PetscFunctionReturn(0);
}

/*@C
  TSGLLERegisterAll - Registers all of the general linear methods in TSGLLE

  Not Collective

  Level: advanced

.seealso:  TSGLLERegisterDestroy()
@*/
PetscErrorCode  TSGLLERegisterAll(void)
{
  PetscFunctionBegin;
  if (TSGLLERegisterAllCalled) PetscFunctionReturn(0);
  TSGLLERegisterAllCalled = PETSC_TRUE;

  CHKERRQ(TSGLLERegister(TSGLLE_IRKS,              TSGLLECreate_IRKS));
  CHKERRQ(TSGLLEAcceptRegister(TSGLLEACCEPT_ALWAYS,TSGLLEAccept_Always));
  PetscFunctionReturn(0);
}

/*@C
  TSGLLEInitializePackage - This function initializes everything in the TSGLLE package. It is called
  from TSInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  TSGLLEInitializePackage(void)
{
  PetscFunctionBegin;
  if (TSGLLEPackageInitialized) PetscFunctionReturn(0);
  TSGLLEPackageInitialized = PETSC_TRUE;
  CHKERRQ(TSGLLERegisterAll());
  CHKERRQ(PetscRegisterFinalize(TSGLLEFinalizePackage));
  PetscFunctionReturn(0);
}

/*@C
  TSGLLEFinalizePackage - This function destroys everything in the TSGLLE package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  TSGLLEFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&TSGLLEList));
  CHKERRQ(PetscFunctionListDestroy(&TSGLLEAcceptList));
  TSGLLEPackageInitialized = PETSC_FALSE;
  TSGLLERegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSGLLE - DAE solver using implicit General Linear methods

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
+ * - John Butcher and Z. Jackieweicz and W. Wright, On error propagation in general linear methods for
  ordinary differential equations, Journal of Complexity, Vol 23, 2007.
- * - John Butcher, Numerical methods for ordinary differential equations, second edition, Wiley, 2009.

.seealso:  TSCreate(), TS, TSSetType()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_GLLE(TS ts)
{
  TS_GLLE        *gl;

  PetscFunctionBegin;
  CHKERRQ(TSGLLEInitializePackage());

  CHKERRQ(PetscNewLog(ts,&gl));
  ts->data = (void*)gl;

  ts->ops->reset          = TSReset_GLLE;
  ts->ops->destroy        = TSDestroy_GLLE;
  ts->ops->view           = TSView_GLLE;
  ts->ops->setup          = TSSetUp_GLLE;
  ts->ops->solve          = TSSolve_GLLE;
  ts->ops->setfromoptions = TSSetFromOptions_GLLE;
  ts->ops->snesfunction   = SNESTSFormFunction_GLLE;
  ts->ops->snesjacobian   = SNESTSFormJacobian_GLLE;

  ts->usessnes = PETSC_TRUE;

  gl->max_step_rejections = 1;
  gl->min_order           = 1;
  gl->max_order           = 3;
  gl->start_order         = 1;
  gl->current_scheme      = -1;
  gl->extrapolate         = PETSC_FALSE;

  gl->wrms_atol = 1e-8;
  gl->wrms_rtol = 1e-5;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSGLLESetType_C",      &TSGLLESetType_GLLE));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSGLLESetAcceptType_C",&TSGLLESetAcceptType_GLLE));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSGLLEGetAdapt_C",     &TSGLLEGetAdapt_GLLE));
  PetscFunctionReturn(0);
}
