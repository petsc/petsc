
static char help[] = "This example solves a linear system in parallel with KSP.  The matrix\n\
uses arbitrary order polynomials for finite elements on the unit square.  To test the parallel\n\
matrix assembly, the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
  -m <size> -p <order> : mesh size and polynomial order\n\n";

/* Contributed by Travis Austin <austin@txcorp.com>, 2010.
   based on src/ksp/ksp/tutorials/ex3.c
 */

#include <petscksp.h>

/* Declare user-defined routines */
static PetscReal      src(PetscReal,PetscReal);
static PetscReal      ubdy(PetscReal,PetscReal);
static PetscReal      polyBasisFunc(PetscInt,PetscInt,PetscReal*,PetscReal);
static PetscReal      derivPolyBasisFunc(PetscInt,PetscInt,PetscReal*,PetscReal);
static PetscErrorCode Form1DElementMass(PetscReal,PetscInt,PetscReal*,PetscReal*,PetscScalar*);
static PetscErrorCode Form1DElementStiffness(PetscReal,PetscInt,PetscReal*,PetscReal*,PetscScalar*);
static PetscErrorCode Form2DElementMass(PetscInt,PetscScalar*,PetscScalar*);
static PetscErrorCode Form2DElementStiffness(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*);
static PetscErrorCode FormNodalRhs(PetscInt,PetscReal,PetscReal,PetscReal,PetscReal*,PetscScalar*);
static PetscErrorCode FormNodalSoln(PetscInt,PetscReal,PetscReal,PetscReal,PetscReal*,PetscScalar*);
static void leggaulob(PetscReal, PetscReal, PetscReal [], PetscReal [], int);
static void qAndLEvaluation(int, PetscReal, PetscReal*, PetscReal*, PetscReal*);

int main(int argc,char **args)
{
  PetscInt       p = 2, m = 5;
  PetscInt       num1Dnodes, num2Dnodes;
  PetscScalar    *Ke1D,*Ke2D,*Me1D,*Me2D;
  PetscScalar    *r,*ue,val;
  Vec            u,ustar,b,q;
  Mat            A,Mass;
  KSP            ksp;
  PetscInt       M,N;
  PetscMPIInt    rank,size;
  PetscReal      x,y,h,norm;
  PetscInt       *idx,indx,count,*rows,i,j,k,start,end,its;
  PetscReal      *rowsx,*rowsy;
  PetscReal      *gllNode, *gllWgts;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for p-FEM","");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m","Number of elements in each direction","None",m,&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p","Order of each element (tensor product basis)","None",p,&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscAssertFalse(p <=0,PETSC_COMM_SELF,PETSC_ERR_USER,"Option -p value should be greater than zero");
  N    = (p*m+1)*(p*m+1); /* dimension of matrix */
  M    = m*m; /* number of elements */
  h    = 1.0/m; /* mesh width */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Create stiffness matrix */
  ierr  = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr  = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr  = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr  = MatSetUp(A);CHKERRQ(ierr);

  /* Create matrix  */
  ierr  = MatCreate(PETSC_COMM_WORLD,&Mass);CHKERRQ(ierr);
  ierr  = MatSetSizes(Mass,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr  = MatSetFromOptions(Mass);CHKERRQ(ierr);
  ierr  = MatSetUp(Mass);CHKERRQ(ierr);
  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank);

  /* Allocate element stiffness matrices */
  num1Dnodes = (p+1);
  num2Dnodes = num1Dnodes*num1Dnodes;

  ierr = PetscMalloc1(num1Dnodes*num1Dnodes,&Me1D);CHKERRQ(ierr);
  ierr = PetscMalloc1(num1Dnodes*num1Dnodes,&Ke1D);CHKERRQ(ierr);
  ierr = PetscMalloc1(num2Dnodes*num2Dnodes,&Me2D);CHKERRQ(ierr);
  ierr = PetscMalloc1(num2Dnodes*num2Dnodes,&Ke2D);CHKERRQ(ierr);
  ierr = PetscMalloc1(num2Dnodes,&idx);CHKERRQ(ierr);
  ierr = PetscMalloc1(num2Dnodes,&r);CHKERRQ(ierr);
  ierr = PetscMalloc1(num2Dnodes,&ue);CHKERRQ(ierr);

  /* Allocate quadrature and create stiffness matrices */
  ierr = PetscMalloc1(p+1,&gllNode);CHKERRQ(ierr);
  ierr = PetscMalloc1(p+1,&gllWgts);CHKERRQ(ierr);
  leggaulob(0.0,1.0,gllNode,gllWgts,p); /* Get GLL nodes and weights */
  ierr = Form1DElementMass(h,p,gllNode,gllWgts,Me1D);CHKERRQ(ierr);
  ierr = Form1DElementStiffness(h,p,gllNode,gllWgts,Ke1D);CHKERRQ(ierr);
  ierr = Form2DElementMass(p,Me1D,Me2D);CHKERRQ(ierr);
  ierr = Form2DElementStiffness(p,Ke1D,Me1D,Ke2D);CHKERRQ(ierr);

  /* Assemble matrix */
  for (i=start; i<end; i++) {
     indx = 0;
     for (k=0;k<(p+1);++k) {
       for (j=0;j<(p+1);++j) {
         idx[indx++] = p*(p*m+1)*(i/m) + p*(i % m) + k*(p*m+1) + j;
       }
     }
     ierr = MatSetValues(A,num2Dnodes,idx,num2Dnodes,idx,Ke2D,ADD_VALUES);CHKERRQ(ierr);
     ierr = MatSetValues(Mass,num2Dnodes,idx,num2Dnodes,idx,Me2D,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Mass,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Mass,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(Me1D);CHKERRQ(ierr);
  ierr = PetscFree(Ke1D);CHKERRQ(ierr);
  ierr = PetscFree(Me2D);CHKERRQ(ierr);
  ierr = PetscFree(Ke2D);CHKERRQ(ierr);

  /* Create right-hand-side and solution vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"Approx. Solution");CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)b,"Right hand side");CHKERRQ(ierr);
  ierr = VecDuplicate(u,&q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)q,"Right hand side 2");CHKERRQ(ierr);
  ierr = VecDuplicate(b,&ustar);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);
  ierr = VecSet(q,0.0);CHKERRQ(ierr);

  /* Assemble nodal right-hand-side and soln vector  */
  for (i=start; i<end; i++) {
    x    = h*(i % m);
    y    = h*(i/m);
    indx = 0;
    for (k=0;k<(p+1);++k) {
      for (j=0;j<(p+1);++j) {
        idx[indx++] = p*(p*m+1)*(i/m) + p*(i % m) + k*(p*m+1) + j;
      }
    }
    ierr = FormNodalRhs(p,x,y,h,gllNode,r);CHKERRQ(ierr);
    ierr = FormNodalSoln(p,x,y,h,gllNode,ue);CHKERRQ(ierr);
    ierr = VecSetValues(q,num2Dnodes,idx,r,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(ustar,num2Dnodes,idx,ue,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(q);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(q);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(ustar);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ustar);CHKERRQ(ierr);

  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = PetscFree(r);CHKERRQ(ierr);
  ierr = PetscFree(ue);CHKERRQ(ierr);

  /* Get FE right-hand side vector */
  ierr = MatMult(Mass,q,b);CHKERRQ(ierr);

  /* Modify matrix and right-hand-side for Dirichlet boundary conditions */
  ierr = PetscMalloc1(4*p*m,&rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(4*p*m,&rowsx);CHKERRQ(ierr);
  ierr = PetscMalloc1(4*p*m,&rowsy);CHKERRQ(ierr);
  for (i=0; i<p*m+1; i++) {
    rows[i]          = i; /* bottom */
    rowsx[i]         = (i/p)*h+gllNode[i%p]*h;
    rowsy[i]         = 0.0;
    rows[3*p*m-1+i]  = (p*m)*(p*m+1) + i; /* top */
    rowsx[3*p*m-1+i] = (i/p)*h+gllNode[i%p]*h;
    rowsy[3*p*m-1+i] = 1.0;
  }
  count = p*m+1; /* left side */
  indx  = 1;
  for (i=p*m+1; i<(p*m)*(p*m+1); i+= (p*m+1)) {
    rows[count]    = i;
    rowsx[count]   = 0.0;
    rowsy[count++] = (indx/p)*h+gllNode[indx%p]*h;
    indx++;
  }
  count = 2*p*m; /* right side */
  indx  = 1;
  for (i=2*p*m+1; i<(p*m)*(p*m+1); i+= (p*m+1)) {
    rows[count]    = i;
    rowsx[count]   = 1.0;
    rowsy[count++] = (indx/p)*h+gllNode[indx%p]*h;
    indx++;
  }
  for (i=0; i<4*p*m; i++) {
    x    = rowsx[i];
    y    = rowsy[i];
    val  = ubdy(x,y);
    ierr = VecSetValues(b,1,&rows[i],&val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(u,1,&rows[i],&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatZeroRows(A,4*p*m,rows,1.0,0,0);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(rowsx);CHKERRQ(ierr);
  ierr = PetscFree(rowsy);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* Solve linear system */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);

  /* Check error */
  ierr = VecAXPY(u,-1.0,ustar);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Iterations %D\n",(double)(norm*h),its);CHKERRQ(ierr);

  ierr = PetscFree(gllNode);CHKERRQ(ierr);
  ierr = PetscFree(gllWgts);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&q);CHKERRQ(ierr);
  ierr = VecDestroy(&ustar);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Mass);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/* --------------------------------------------------------------------- */

/* 1d element stiffness mass matrix  */
static PetscErrorCode Form1DElementMass(PetscReal H,PetscInt P,PetscReal *gqn,PetscReal *gqw,PetscScalar *Me1D)
{
  PetscInt i,j,k;
  PetscInt indx;

  PetscFunctionBeginUser;
  for (j=0; j<(P+1); ++j) {
    for (i=0; i<(P+1); ++i) {
      indx       = j*(P+1)+i;
      Me1D[indx] = 0.0;
      for (k=0; k<(P+1);++k) {
        Me1D[indx] += H*gqw[k]*polyBasisFunc(P,i,gqn,gqn[k])*polyBasisFunc(P,j,gqn,gqn[k]);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

/* 1d element stiffness matrix for derivative */
static PetscErrorCode Form1DElementStiffness(PetscReal H,PetscInt P,PetscReal *gqn,PetscReal *gqw,PetscScalar *Ke1D)
{
  PetscInt i,j,k;
  PetscInt indx;

  PetscFunctionBeginUser;
  for (j=0;j<(P+1);++j) {
    for (i=0;i<(P+1);++i) {
      indx = j*(P+1)+i;
      Ke1D[indx] = 0.0;
      for (k=0; k<(P+1);++k) {
        Ke1D[indx] += (1./H)*gqw[k]*derivPolyBasisFunc(P,i,gqn,gqn[k])*derivPolyBasisFunc(P,j,gqn,gqn[k]);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

   /* element mass matrix */
static PetscErrorCode Form2DElementMass(PetscInt P,PetscScalar *Me1D,PetscScalar *Me2D)
{
  PetscInt i1,j1,i2,j2;
  PetscInt indx1,indx2,indx3;

  PetscFunctionBeginUser;
  for (j2=0;j2<(P+1);++j2) {
    for (i2=0; i2<(P+1);++i2) {
      for (j1=0;j1<(P+1);++j1) {
        for (i1=0;i1<(P+1);++i1) {
          indx1 = j1*(P+1)+i1;
          indx2 = j2*(P+1)+i2;
          indx3 = (j2*(P+1)+j1)*(P+1)*(P+1)+(i2*(P+1)+i1);
          Me2D[indx3] = Me1D[indx1]*Me1D[indx2];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

/* element stiffness for Laplacian */
static PetscErrorCode Form2DElementStiffness(PetscInt P,PetscScalar *Ke1D,PetscScalar *Me1D,PetscScalar *Ke2D)
{
  PetscInt i1,j1,i2,j2;
  PetscInt indx1,indx2,indx3;

  PetscFunctionBeginUser;
  for (j2=0;j2<(P+1);++j2) {
    for (i2=0; i2<(P+1);++i2) {
      for (j1=0;j1<(P+1);++j1) {
        for (i1=0;i1<(P+1);++i1) {
          indx1 = j1*(P+1)+i1;
          indx2 = j2*(P+1)+i2;
          indx3 = (j2*(P+1)+j1)*(P+1)*(P+1)+(i2*(P+1)+i1);
          Ke2D[indx3] = Ke1D[indx1]*Me1D[indx2] + Me1D[indx1]*Ke1D[indx2];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

static PetscErrorCode FormNodalRhs(PetscInt P,PetscReal x,PetscReal y,PetscReal H,PetscReal* nds,PetscScalar *r)
{
  PetscInt i,j,indx;

  PetscFunctionBeginUser;
  indx=0;
  for (j=0;j<(P+1);++j) {
    for (i=0;i<(P+1);++i) {
      r[indx] = src(x+H*nds[i],y+H*nds[j]);
      indx++;
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

static PetscErrorCode FormNodalSoln(PetscInt P,PetscReal x,PetscReal y,PetscReal H,PetscReal* nds,PetscScalar *u)
{
  PetscInt i,j,indx;

  PetscFunctionBeginUser;
  indx=0;
  for (j=0;j<(P+1);++j) {
    for (i=0;i<(P+1);++i) {
      u[indx] = ubdy(x+H*nds[i],y+H*nds[j]);
      indx++;
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

static PetscReal polyBasisFunc(PetscInt order, PetscInt basis, PetscReal *xLocVal, PetscReal xval)
{
  PetscReal denominator = 1.;
  PetscReal numerator   = 1.;
  PetscInt  i           =0;

  for (i=0; i<(order+1); i++) {
    if (i!=basis) {
      numerator   *= (xval-xLocVal[i]);
      denominator *= (xLocVal[basis]-xLocVal[i]);
    }
  }
  return (numerator/denominator);
}

/* --------------------------------------------------------------------- */

static PetscReal derivPolyBasisFunc(PetscInt order, PetscInt basis, PetscReal *xLocVal, PetscReal xval)
{
  PetscReal denominator;
  PetscReal numerator;
  PetscReal numtmp;
  PetscInt  i=0,j=0;

  denominator=1.;
  for (i=0; i<(order+1); i++) {
    if (i!=basis) denominator *= (xLocVal[basis]-xLocVal[i]);
  }
  numerator = 0.;
  for (j=0;j<(order+1);++j) {
    if (j != basis) {
      numtmp = 1.;
      for (i=0;i<(order+1);++i) {
        if (i!=basis && j!=i) numtmp *= (xval-xLocVal[i]);
      }
      numerator += numtmp;
    }
  }

  return (numerator/denominator);
}

/* --------------------------------------------------------------------- */

static PetscReal ubdy(PetscReal x,PetscReal y)
{
  return x*x*y*y;
}

static PetscReal src(PetscReal x,PetscReal y)
{
  return -2.*y*y - 2.*x*x;
}
/* --------------------------------------------------------------------- */

static void leggaulob(PetscReal x1, PetscReal x2, PetscReal x[], PetscReal w[], int n)
/*******************************************************************************
Given the lower and upper limits of integration x1 and x2, and given n, this
routine returns arrays x[0..n-1] and w[0..n-1] of length n, containing the abscissas
and weights of the Gauss-Lobatto-Legendre n-point quadrature formula.
*******************************************************************************/
{
  PetscInt    j,m;
  PetscReal z1,z,xm,xl,q,qp,Ln,scale;
  if (n==1) {
    x[0] = x1;   /* Scale the root to the desired interval, */
    x[1] = x2;   /* and put in its symmetric counterpart.   */
    w[0] = 1.;   /* Compute the weight */
    w[1] = 1.;   /* and its symmetric counterpart. */
  } else {
    x[0] = x1;   /* Scale the root to the desired interval, */
    x[n] = x2;   /* and put in its symmetric counterpart.   */
    w[0] = 2./(n*(n+1));   /* Compute the weight */
    w[n] = 2./(n*(n+1));   /* and its symmetric counterpart. */
    m    = (n+1)/2; /* The roots are symmetric, so we only find half of them. */
    xm   = 0.5*(x2+x1);
    xl   = 0.5*(x2-x1);
    for (j=1; j<=(m-1); j++) { /* Loop over the desired roots. */
      z=-1.0*PetscCosReal((PETSC_PI*(j+0.25)/(n))-(3.0/(8.0*n*PETSC_PI))*(1.0/(j+0.25)));
      /* Starting with the above approximation to the ith root, we enter */
      /* the main loop of refinement by Newton's method.                 */
      do {
        qAndLEvaluation(n,z,&q,&qp,&Ln);
        z1 = z;
        z  = z1-q/qp; /* Newton's method. */
      } while (PetscAbsReal(z-z1) > 3.0e-11);
      qAndLEvaluation(n,z,&q,&qp,&Ln);
      x[j]   = xm+xl*z;      /* Scale the root to the desired interval, */
      x[n-j] = xm-xl*z;      /* and put in its symmetric counterpart.   */
      w[j]   = 2.0/(n*(n+1)*Ln*Ln);  /* Compute the weight */
      w[n-j] = w[j];                 /* and its symmetric counterpart. */
    }
  }
  if (n%2==0) {
    qAndLEvaluation(n,0.0,&q,&qp,&Ln);
    x[n/2]=(x2-x1)/2.0;
    w[n/2]=2.0/(n*(n+1)*Ln*Ln);
  }
  /* scale the weights according to mapping from [-1,1] to [0,1] */
  scale = (x2-x1)/2.0;
  for (j=0; j<=n; ++j) w[j] = w[j]*scale;
}

/******************************************************************************/
static void qAndLEvaluation(int n, PetscReal x, PetscReal *q, PetscReal *qp, PetscReal *Ln)
/*******************************************************************************
Compute the polynomial qn(x) = L_{N+1}(x) - L_{n-1}(x) and its derivative in
addition to L_N(x) as these are needed for the GLL points.  See the book titled
"Implementing Spectral Methods for Partial Differential Equations: Algorithms
for Scientists and Engineers" by David A. Kopriva.
*******************************************************************************/
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
    *Ln   = (2.*k-1.)/(1.0*k)*x*Lnm1 - (k-1.)/(1.0*k)*Lnm2;
    Lnp   = Lnm2p + (2.0*k-1)*Lnm1;
  }
  k     = n+1;
  Lnp1  = (2.*k-1.)/(1.0*k)*x*(*Ln) - (k-1.)/(1.0*k)*Lnm1;
  Lnp1p = Lnm1p + (2.0*k-1)*(*Ln);
  *q    = Lnp1 - Lnm1;
  *qp   = Lnp1p - Lnm1p;
}

/*TEST

   test:
      nsize: 2
      args: -ksp_monitor_short

TEST*/
