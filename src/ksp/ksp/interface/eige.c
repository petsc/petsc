
#include <petsc/private/kspimpl.h>   /*I "petscksp.h" I*/
#include <petscdm.h>
#include <petscblaslapack.h>

typedef struct {
  KSP ksp;
  Vec work;
} Mat_KSP;

static PetscErrorCode MatCreateVecs_KSP(Mat A,Vec *X,Vec *Y)
{
  Mat_KSP        *ctx;
  Mat            M;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ctx));
  PetscCall(KSPGetOperators(ctx->ksp,&M,NULL));
  PetscCall(MatCreateVecs(M,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_KSP(Mat A,Vec X,Vec Y)
{
  Mat_KSP        *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ctx));
  PetscCall(KSP_PCApplyBAorAB(ctx->ksp,X,Y,ctx->work));
  PetscFunctionReturn(0);
}

/*@
    KSPComputeOperator - Computes the explicit preconditioned operator, including diagonal scaling and null
    space removal if applicable.

    Collective on ksp

    Input Parameters:
+   ksp - the Krylov subspace context
-   mattype - the matrix type to be used

    Output Parameter:
.   mat - the explicit preconditioned operator

    Notes:
    This computation is done by applying the operators to columns of the
    identity matrix.

    Currently, this routine uses a dense matrix format for the output operator if mattype == NULL.
    This routine is costly in general, and is recommended for use only with relatively small systems.

    Level: advanced

.seealso: KSPComputeEigenvaluesExplicitly(), PCComputeOperator(), KSPSetDiagonalScale(), KSPSetNullSpace(), MatType
@*/
PetscErrorCode  KSPComputeOperator(KSP ksp, MatType mattype, Mat *mat)
{
  PetscInt       N,M,m,n;
  Mat_KSP        ctx;
  Mat            A,Aksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(mat,3);
  PetscCall(KSPGetOperators(ksp,&A,NULL));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)ksp),m,n,M,N,&ctx,&Aksp));
  PetscCall(MatShellSetOperation(Aksp,MATOP_MULT,(void (*)(void))MatMult_KSP));
  PetscCall(MatShellSetOperation(Aksp,MATOP_CREATE_VECS,(void (*)(void))MatCreateVecs_KSP));
  ctx.ksp = ksp;
  PetscCall(MatCreateVecs(A,&ctx.work,NULL));
  PetscCall(MatComputeOperator(Aksp,mattype,mat));
  PetscCall(VecDestroy(&ctx.work));
  PetscCall(MatDestroy(&Aksp));
  PetscFunctionReturn(0);
}

/*@
   KSPComputeEigenvaluesExplicitly - Computes all of the eigenvalues of the
   preconditioned operator using LAPACK.

   Collective on ksp

   Input Parameters:
+  ksp - iterative context obtained from KSPCreate()
-  n - size of arrays r and c

   Output Parameters:
+  r - real part of computed eigenvalues, provided by user with a dimension at least of n
-  c - complex part of computed eigenvalues, provided by user with a dimension at least of n

   Notes:
   This approach is very slow but will generally provide accurate eigenvalue
   estimates.  This routine explicitly forms a dense matrix representing
   the preconditioned operator, and thus will run only for relatively small
   problems, say n < 500.

   Many users may just want to use the monitoring routine
   KSPMonitorSingularValue() (which can be set with option -ksp_monitor_singular_value)
   to print the singular values at each iteration of the linear solve.

   The preconditoner operator, rhs vector, solution vectors should be
   set before this routine is called. i.e use KSPSetOperators(),KSPSolve() or
   KSPSetOperators()

   Level: advanced

.seealso: KSPComputeEigenvalues(), KSPMonitorSingularValue(), KSPComputeExtremeSingularValues(), KSPSetOperators(), KSPSolve()
@*/
PetscErrorCode  KSPComputeEigenvaluesExplicitly(KSP ksp,PetscInt nmax,PetscReal r[],PetscReal c[])
{
  Mat               BA;
  PetscMPIInt       size,rank;
  MPI_Comm          comm;
  PetscScalar       *array;
  Mat               A;
  PetscInt          m,row,nz,i,n,dummy;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp,&comm));
  PetscCall(KSPComputeOperator(ksp,MATDENSE,&BA));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(MatGetSize(BA,&n,&n));
  if (size > 1) { /* assemble matrix on first processor */
    PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp),&A));
    if (rank == 0) {
      PetscCall(MatSetSizes(A,n,n,n,n));
    } else {
      PetscCall(MatSetSizes(A,0,0,n,n));
    }
    PetscCall(MatSetType(A,MATMPIDENSE));
    PetscCall(MatMPIDenseSetPreallocation(A,NULL));
    PetscCall(PetscLogObjectParent((PetscObject)BA,(PetscObject)A));

    PetscCall(MatGetOwnershipRange(BA,&row,&dummy));
    PetscCall(MatGetLocalSize(BA,&m,&dummy));
    for (i=0; i<m; i++) {
      PetscCall(MatGetRow(BA,row,&nz,&cols,&vals));
      PetscCall(MatSetValues(A,1,&row,nz,cols,vals,INSERT_VALUES));
      PetscCall(MatRestoreRow(BA,row,&nz,&cols,&vals));
      row++;
    }

    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatDenseGetArray(A,&array));
  } else {
    PetscCall(MatDenseGetArray(BA,&array));
  }

#if !defined(PETSC_USE_COMPLEX)
  if (rank == 0) {
    PetscScalar  *work;
    PetscReal    *realpart,*imagpart;
    PetscBLASInt idummy,lwork;
    PetscInt     *perm;

    idummy   = n;
    lwork    = 5*n;
    PetscCall(PetscMalloc2(n,&realpart,n,&imagpart));
    PetscCall(PetscMalloc1(5*n,&work));
    {
      PetscBLASInt lierr;
      PetscScalar  sdummy;
      PetscBLASInt bn;

      PetscCall(PetscBLASIntCast(n,&bn));
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&bn,array,&bn,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr));
      PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
      PetscCall(PetscFPTrapPop());
    }
    PetscCall(PetscFree(work));
    PetscCall(PetscMalloc1(n,&perm));

    for (i=0; i<n; i++)  perm[i] = i;
    PetscCall(PetscSortRealWithPermutation(n,realpart,perm));
    for (i=0; i<n; i++) {
      r[i] = realpart[perm[i]];
      c[i] = imagpart[perm[i]];
    }
    PetscCall(PetscFree(perm));
    PetscCall(PetscFree2(realpart,imagpart));
  }
#else
  if (rank == 0) {
    PetscScalar  *work,*eigs;
    PetscReal    *rwork;
    PetscBLASInt idummy,lwork;
    PetscInt     *perm;

    idummy = n;
    lwork  = 5*n;
    PetscCall(PetscMalloc1(5*n,&work));
    PetscCall(PetscMalloc1(2*n,&rwork));
    PetscCall(PetscMalloc1(n,&eigs));
    {
      PetscBLASInt lierr;
      PetscScalar  sdummy;
      PetscBLASInt nb;
      PetscCall(PetscBLASIntCast(n,&nb));
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&nb,array,&nb,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,rwork,&lierr));
      PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
      PetscCall(PetscFPTrapPop());
    }
    PetscCall(PetscFree(work));
    PetscCall(PetscFree(rwork));
    PetscCall(PetscMalloc1(n,&perm));
    for (i=0; i<n; i++) perm[i] = i;
    for (i=0; i<n; i++) r[i]    = PetscRealPart(eigs[i]);
    PetscCall(PetscSortRealWithPermutation(n,r,perm));
    for (i=0; i<n; i++) {
      r[i] = PetscRealPart(eigs[perm[i]]);
      c[i] = PetscImaginaryPart(eigs[perm[i]]);
    }
    PetscCall(PetscFree(perm));
    PetscCall(PetscFree(eigs));
  }
#endif
  if (size > 1) {
    PetscCall(MatDenseRestoreArray(A,&array));
    PetscCall(MatDestroy(&A));
  } else {
    PetscCall(MatDenseRestoreArray(BA,&array));
  }
  PetscCall(MatDestroy(&BA));
  PetscFunctionReturn(0);
}

static PetscErrorCode PolyEval(PetscInt nroots,const PetscReal *r,const PetscReal *c,PetscReal x,PetscReal y,PetscReal *px,PetscReal *py)
{
  PetscInt  i;
  PetscReal rprod = 1,iprod = 0;

  PetscFunctionBegin;
  for (i=0; i<nroots; i++) {
    PetscReal rnew = rprod*(x - r[i]) - iprod*(y - c[i]);
    PetscReal inew = rprod*(y - c[i]) + iprod*(x - r[i]);
    rprod = rnew;
    iprod = inew;
  }
  *px = rprod;
  *py = iprod;
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
/* collective on ksp */
PetscErrorCode KSPPlotEigenContours_Private(KSP ksp,PetscInt neig,const PetscReal *r,const PetscReal *c)
{
  PetscReal      xmin,xmax,ymin,ymax,*xloc,*yloc,*value,px0,py0,rscale,iscale;
  PetscInt       M,N,i,j;
  PetscMPIInt    rank;
  PetscViewer    viewer;
  PetscDraw      draw;
  PetscDrawAxis  drawaxis;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ksp),&rank));
  if (rank) PetscFunctionReturn(0);
  M    = 80;
  N    = 80;
  xmin = r[0]; xmax = r[0];
  ymin = c[0]; ymax = c[0];
  for (i=1; i<neig; i++) {
    xmin = PetscMin(xmin,r[i]);
    xmax = PetscMax(xmax,r[i]);
    ymin = PetscMin(ymin,c[i]);
    ymax = PetscMax(ymax,c[i]);
  }
  PetscCall(PetscMalloc3(M,&xloc,N,&yloc,M*N,&value));
  for (i=0; i<M; i++) xloc[i] = xmin - 0.1*(xmax-xmin) + 1.2*(xmax-xmin)*i/(M-1);
  for (i=0; i<N; i++) yloc[i] = ymin - 0.1*(ymax-ymin) + 1.2*(ymax-ymin)*i/(N-1);
  PetscCall(PolyEval(neig,r,c,0,0,&px0,&py0));
  rscale = px0/(PetscSqr(px0)+PetscSqr(py0));
  iscale = -py0/(PetscSqr(px0)+PetscSqr(py0));
  for (j=0; j<N; j++) {
    for (i=0; i<M; i++) {
      PetscReal px,py,tx,ty,tmod;
      PetscCall(PolyEval(neig,r,c,xloc[i],yloc[j],&px,&py));
      tx   = px*rscale - py*iscale;
      ty   = py*rscale + px*iscale;
      tmod = PetscSqr(tx) + PetscSqr(ty); /* modulus of the complex polynomial */
      if (tmod > 1) tmod = 1.0;
      if (tmod > 0.5 && tmod < 1) tmod = 0.5;
      if (tmod > 0.2 && tmod < 0.5) tmod = 0.2;
      if (tmod > 0.05 && tmod < 0.2) tmod = 0.05;
      if (tmod < 1e-3) tmod = 1e-3;
      value[i+j*M] = PetscLogReal(tmod) / PetscLogReal(10.0);
    }
  }
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_SELF,NULL,"Iteratively Computed Eigen-contours",PETSC_DECIDE,PETSC_DECIDE,450,450,&viewer));
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawTensorContour(draw,M,N,NULL,NULL,value));
  if (0) {
    PetscCall(PetscDrawAxisCreate(draw,&drawaxis));
    PetscCall(PetscDrawAxisSetLimits(drawaxis,xmin,xmax,ymin,ymax));
    PetscCall(PetscDrawAxisSetLabels(drawaxis,"Eigen-counters","real","imag"));
    PetscCall(PetscDrawAxisDraw(drawaxis));
    PetscCall(PetscDrawAxisDestroy(&drawaxis));
  }
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFree3(xloc,yloc,value));
  PetscFunctionReturn(0);
}
