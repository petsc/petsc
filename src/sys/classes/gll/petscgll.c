
#include <petscgll.h>
#include <petscviewer.h>
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>


static void qAndLEvaluation(PetscInt n, PetscReal x, PetscReal *q, PetscReal *qp, PetscReal *Ln)
/*
  Compute the polynomial q(x) = L_{N+1}(x) - L_{n-1}(x) and its derivative in
  addition to L_N(x) as these are needed for computing the GLL points via Newton's method.
  Reference: "Implementing Spectral Methods for Partial Differential Equations: Algorithms
  for Scientists and Engineers" by David A. Kopriva.
*/
{
  PetscInt k;

  PetscReal Lnp;
  PetscReal Lnp1, Lnp1p;
  PetscReal Lnm1, Lnm1p;
  PetscReal Lnm2, Lnm2p;

  Lnm1  = 1.0;
  *Ln   = x;
  Lnm1p = 0.0;
  Lnp   = 1.0;

  for (k=2; k<=n; ++k) {
    Lnm2  = Lnm1;
    Lnm1  = *Ln;
    Lnm2p = Lnm1p;
    Lnm1p = Lnp;
    *Ln   = (2.*((PetscReal)k)-1.)/(1.0*((PetscReal)k))*x*Lnm1 - (((PetscReal)k)-1.)/((PetscReal)k)*Lnm2;
    Lnp   = Lnm2p + (2.0*((PetscReal)k)-1.)*Lnm1;
  }
  k     = n+1;
  Lnp1  = (2.*((PetscReal)k)-1.)/(((PetscReal)k))*x*(*Ln) - (((PetscReal)k)-1.)/((PetscReal)k)*Lnm1;
  Lnp1p = Lnm1p + (2.0*((PetscReal)k)-1.)*(*Ln);
  *q    = Lnp1 - Lnm1;
  *qp   = Lnp1p - Lnm1p;
}

/*@C
   PetscGLLCreate - creates a set of the locations and weights of the Gauss-Lobatto-Legendre (GLL) nodes of a given size
                      on the domain [-1,1]

   Not Collective

   Input Parameter:
+  n - number of grid nodes
-  type - PETSCGLL_VIA_LINEARALGEBRA or PETSCGLL_VIA_NEWTON

   Output Parameter:
.  gll - the nodes

   Notes: For n > 30  the Newton approach computes duplicate (incorrect) values for some nodes because the initial guess is apparently not
          close enough to the desired solution

   These are useful for implementing spectral methods based on Gauss-Lobatto-Legendre (GLL) nodes

   See  http://epubs.siam.org/doi/abs/10.1137/110855442  http://epubs.siam.org/doi/abs/10.1137/120889873 for better ways to compute GLL nodes

   Level: beginner

.seealso: PetscGLL, PetscGLLDestroy(), PetscGLLView(), PetscGLLIntegrate(), PetscGLLElementLaplacianCreate(), PetscGLLElementLaplacianDestroy(), 
          PetscGLLElementGradientCreate(), PetscGLLElementGradientDestroy(), PetscGLLElementAdvectionCreate(), PetscGLLElementAdvectionDestroy()

@*/
PetscErrorCode PetscGLLCreate(PetscInt n,PetscGLLCreateType type,PetscGLL *gll)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(n,&gll->nodes,n,&gll->weights);CHKERRQ(ierr);

  if (type == PETSCGLL_VIA_LINEARALGEBRA) {
    PetscReal      *M,si;
    PetscBLASInt   bn,lierr;
    PetscReal      x,z0,z1,z2;
    PetscInt       i,p = n - 1,nn;

    gll->nodes[0]   =-1.0;
    gll->nodes[n-1] = 1.0;
    if (n-2 > 0){
      ierr = PetscMalloc1(n-1,&M);CHKERRQ(ierr);
      for (i=0; i<n-2; i++) {
        si  = ((PetscReal)i)+1.0;
        M[i]=0.5*PetscSqrtReal(si*(si+2.0)/((si+0.5)*(si+1.5)));
      }
      ierr = PetscBLASIntCast(n-2,&bn);CHKERRQ(ierr);
      ierr = PetscMemzero(&gll->nodes[1],bn*sizeof(gll->nodes[1]));CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      x=0;
      PetscStackCallBLAS("LAPACKsteqr",LAPACKREALsteqr_("N",&bn,&gll->nodes[1],M,&x,&bn,M,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in STERF Lapack routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      ierr = PetscFree(M);CHKERRQ(ierr);
    }
    if ((n-1)%2==0) {
      gll->nodes[(n-1)/2]   = 0.0; /* hard wire to exactly 0.0 since linear algebra produces nonzero */
    }

    gll->weights[0] = gll->weights[p] = 2.0/(((PetscReal)(p))*(((PetscReal)p)+1.0));
    z2 = -1.;                      /* Dummy value to avoid -Wmaybe-initialized */
    for (i=1; i<p; i++) {
      x  = gll->nodes[i];
      z0 = 1.0;
      z1 = x;
      for (nn=1; nn<p; nn++) {
        z2 = x*z1*(2.0*((PetscReal)nn)+1.0)/(((PetscReal)nn)+1.0)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.0));
        z0 = z1;
        z1 = z2;
      }
      gll->weights[i]=2.0/(((PetscReal)p)*(((PetscReal)p)+1.0)*z2*z2);
    }
  } else {
    PetscInt  j,m;
    PetscReal z1,z,q,qp,Ln;
    PetscReal *pt;
    ierr = PetscMalloc1(n,&pt);CHKERRQ(ierr);

    if (n > 30) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PETSCGLL_VIA_NEWTON produces incorrect answers for n > 30");
    gll->nodes[0]     = -1.0;
    gll->nodes[n-1]   = 1.0;
    gll->weights[0]   = gll->weights[n-1] = 2./(((PetscReal)n)*(((PetscReal)n)-1.0));;
    m  = (n-1)/2; /* The roots are symmetric, so we only find half of them. */
    for (j=1; j<=m; j++) { /* Loop over the desired roots. */
      z = -1.0*PetscCosReal((PETSC_PI*((PetscReal)j)+0.25)/(((PetscReal)n)-1.0))-(3.0/(8.0*(((PetscReal)n)-1.0)*PETSC_PI))*(1.0/(((PetscReal)j)+0.25));
      /* Starting with the above approximation to the ith root, we enter */
      /* the main loop of refinement by Newton's method.                 */
      do {
        qAndLEvaluation(n-1,z,&q,&qp,&Ln);
        z1 = z;
        z  = z1-q/qp; /* Newton's method. */
      } while (PetscAbs(z-z1) > 10.*PETSC_MACHINE_EPSILON);
      qAndLEvaluation(n-1,z,&q,&qp,&Ln);

      gll->nodes[j]       = z;
      gll->nodes[n-1-j]   = -z;      /* and put in its symmetric counterpart.   */
      gll->weights[j]     = 2.0/(((PetscReal)n)*(((PetscReal)n)-1.)*Ln*Ln);  /* Compute the weight */
      gll->weights[n-1-j] = gll->weights[j];                 /* and its symmetric counterpart. */
      pt[j]=qp;
    }

    if ((n-1)%2==0) {
      qAndLEvaluation(n-1,0.0,&q,&qp,&Ln);
      gll->nodes[(n-1)/2]   = 0.0;
      gll->weights[(n-1)/2] = 2.0/(((PetscReal)n)*(((PetscReal)n)-1.)*Ln*Ln);
    }
    ierr = PetscFree(pt);CHKERRQ(ierr);
  }
  gll->n = n;
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLDestroy - destroys a set of GLL nodes and weights

   Not Collective

   Input Parameter:
.  gll - the nodes

   Level: beginner

.seealso: PetscGLL, PetscGLLCreate(), PetscGLLView()

@*/
PetscErrorCode PetscGLLDestroy(PetscGLL *gll)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = PetscFree2(gll->nodes,gll->weights);CHKERRQ(ierr);
  gll->n = 0;
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLView - views a set of GLL nodes

   Not Collective

   Input Parameter:
+  gll - the nodes
.  viewer - the viewer

   Level: beginner

.seealso: PetscGLL, PetscGLLCreate(), PetscGLLDestroy()

@*/
PetscErrorCode PetscGLLView(PetscGLL *gll,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;

  PetscInt          i;

  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"%D Gauss-Lobatto-Legendre (GLL) nodes and weights\n",gll->n);CHKERRQ(ierr);
    for (i=0; i<gll->n; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"  %D %16.14e %16.14e\n",i,(double)gll->nodes[i],(double)gll->weights[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLIntegrate - Compute the L2 integral of a function on the GLL points

   Not Collective

   Input Parameter:
+  gll - the nodes
.  f - the function values at the nodes

   Output Parameter:
.  in - the value of the integral

   Level: beginner

.seealso: PetscGLL, PetscGLLCreate(), PetscGLLDestroy()

@*/
PetscErrorCode PetscGLLIntegrate(PetscGLL *gll,const PetscReal *f,PetscReal *in)
{
  PetscInt          i;

  PetscFunctionBegin;
  *in = 0.;
  for (i=0; i<gll->n; i++) {
    *in += f[i]*f[i]*gll->weights[i];
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLElementLaplacianCreate - computes the Laplacian for a single 1d GLL element

   Not Collective

   Input Parameter:
.  gll - the nodes

   Output Parameter:
.  A - the stiffness element

   Level: beginner

   Notes: Destroy this with PetscGLLElementLaplacianDestroy()

   You can access entries in this array with AA[i][j] but in memory it is stored in contiguous memory, row oriented (the array is symmetric)

.seealso: PetscGLL, PetscGLLDestroy(), PetscGLLView(), PetscGLLElementLaplacianDestroy()

@*/
PetscErrorCode PetscGLLElementLaplacianCreate(PetscGLL *gll,PetscReal ***AA)
{
  PetscReal        **A;
  PetscErrorCode  ierr;
  const PetscReal  *nodes = gll->nodes;
  const PetscInt   n = gll->n, p = gll->n-1;
  PetscReal        z0,z1,z2 = 0,x,Lpj,Lpr;
  PetscInt         i,j,nn,r;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*n,&A[0]);CHKERRQ(ierr);
  for (i=1; i<n; i++) A[i] = A[i-1]+n;

  for (j=1; j<p; j++) {
    x  = nodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj=z2;
    for (r=1; r<p; r++) {
      if (r == j) {
        A[j][j]=2./(3.*(1.-nodes[j]*nodes[j])*Lpj*Lpj);
      } else {
        x  = nodes[r];
        z0 = 1.;
        z1 = x;
        for (nn=1; nn<p; nn++) {
          z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
          z0 = z1;
          z1 = z2;
        }
        Lpr     = z2;
        A[r][j] = 4./(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*Lpr*(nodes[j]-nodes[r])*(nodes[j]-nodes[r]));
      }
    }
  }
  for (j=1; j<p+1; j++) {
    x  = nodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj     = z2;
    A[j][0] = 4.*PetscPowRealInt(-1.,p)/(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*(1.+nodes[j])*(1.+nodes[j]));
    A[0][j] = A[j][0];
  }
  for (j=0; j<p; j++) {
    x  = nodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj=z2;

    A[p][j] = 4./(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*(1.-nodes[j])*(1.-nodes[j]));
    A[j][p] = A[p][j];
  }
  A[0][0]=0.5+(((PetscReal)p)*(((PetscReal)p)+1.)-2.)/6.;
  A[p][p]=A[0][0];
  *AA = A;
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLElementLaplacianDestroy - frees the Laplacian for a single 1d GLL element

   Not Collective

   Input Parameter:
+  gll - the nodes
-  A - the stiffness element

   Level: beginner

.seealso: PetscGLL, PetscGLLDestroy(), PetscGLLView(), PetscGLLElementLaplacianCreate()

@*/
PetscErrorCode PetscGLLElementLaplacianDestroy(PetscGLL *gll,PetscReal ***AA)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA  = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLElementGradientCreate - computes the gradient for a single 1d GLL element

   Not Collective

   Input Parameter:
.  gll - the nodes

   Output Parameter:
.  AA - the stiffness element
-  AAT - the transpose of AA (pass in NULL if you do not need this array)

   Level: beginner

   Notes: Destroy this with PetscGLLElementGradientDestroy()

   You can access entries in these arrays with AA[i][j] but in memory it is stored in contiguous memory, row oriented

.seealso: PetscGLL, PetscGLLDestroy(), PetscGLLView(), PetscGLLElementLaplacianDestroy()

@*/
PetscErrorCode PetscGLLElementGradientCreate(PetscGLL *gll,PetscReal ***AA, PetscReal ***AAT)
{
  PetscReal        **A, **AT = NULL;
  PetscErrorCode  ierr;
  const PetscReal  *nodes = gll->nodes;
  const PetscInt   n = gll->n, p = gll->n-1;
  PetscReal        q,qp,Li, Lj,d0;
  PetscInt         i,j;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*n,&A[0]);CHKERRQ(ierr);
  for (i=1; i<n; i++) A[i] = A[i-1]+n;

  if (AAT) {
    ierr = PetscMalloc1(n,&AT);CHKERRQ(ierr);
    ierr = PetscMalloc1(n*n,&AT[0]);CHKERRQ(ierr);
    for (i=1; i<n; i++) AT[i] = AT[i-1]+n;
  }

  if (n==1) {A[0][0] = 0.;}
  d0 = (PetscReal)p*((PetscReal)p+1.)/4.;
  for  (i=0; i<n; i++) {
    for  (j=0; j<n; j++) {
      A[i][j] = 0.;
      qAndLEvaluation(p,nodes[i],&q,&qp,&Li);
      qAndLEvaluation(p,nodes[j],&q,&qp,&Lj);
      if (i!=j)             A[i][j] = Li/(Lj*(nodes[i]-nodes[j]));
      if ((j==i) && (i==0)) A[i][j] = -d0;
      if (j==i && i==p)     A[i][j] = d0;
      if (AT) AT[j][i] = A[i][j];
    }
  }
  if (AAT) *AAT = AT;
  *AA  = A;
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLElementGradientDestroy - frees the gradient for a single 1d GLL element obtained with PetscGLLElementGradientCreate()

   Not Collective

   Input Parameter:
+  gll - the nodes
.  AA - the stiffness element
-  AAT - the transpose of the element

   Level: beginner

.seealso: PetscGLL, PetscGLLDestroy(), PetscGLLView(), PetscGLLElementLaplacianCreate(), PetscGLLElementAdvectionCreate()

@*/
PetscErrorCode PetscGLLElementGradientDestroy(PetscGLL *gll,PetscReal ***AA,PetscReal ***AAT)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA  = NULL;
  if (*AAT) {
    ierr = PetscFree((*AAT)[0]);CHKERRQ(ierr);
    ierr = PetscFree(*AAT);CHKERRQ(ierr);
    *AAT  = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLElementAdvectionCreate - computes the advection operator for a single 1d GLL element

   Not Collective

   Input Parameter:
.  gll - the nodes

   Output Parameter:
.  AA - the stiffness element

   Level: beginner

   Notes: Destroy this with PetscGLLElementAdvectionDestroy()

   This is the same as the Gradient operator multiplied by the diagonal mass matrix

   You can access entries in this array with AA[i][j] but in memory it is stored in contiguous memory, row oriented

.seealso: PetscGLL, PetscGLLDestroy(), PetscGLLView(), PetscGLLElementLaplacianDestroy()

@*/
PetscErrorCode PetscGLLElementAdvectionCreate(PetscGLL *gll,PetscReal ***AA)
{
  PetscReal       **D;
  PetscErrorCode  ierr;
  const PetscReal  *weights = gll->weights;
  const PetscInt   n = gll->n;
  PetscInt         i,j;

  PetscFunctionBegin;
  ierr = PetscGLLElementGradientCreate(gll,&D,NULL);CHKERRQ(ierr);
  for (i=0; i<n; i++){
    for (j=0; j<n; j++) {
      D[i][j] = weights[i]*D[i][j];
    }
  }
  *AA = D;
  PetscFunctionReturn(0);
}

/*@C
   PetscGLLElementAdvectionDestroy - frees the advection stiffness for a single 1d GLL element

   Not Collective

   Input Parameter:
+  gll - the nodes
-  A - advection

   Level: beginner

.seealso: PetscGLL, PetscGLLDestroy(), PetscGLLView(), PetscGLLElementLaplacianCreate(), PetscGLLElementAdvectionCreate()

@*/
PetscErrorCode PetscGLLElementAdvectionDestroy(PetscGLL *gll,PetscReal ***AA)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA  = NULL;
  PetscFunctionReturn(0);
}



