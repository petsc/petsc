
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);
  ierr = KSPGetOperators(ctx->ksp,&M,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(M,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_KSP(Mat A,Vec X,Vec Y)
{
  Mat_KSP        *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);
  ierr = KSP_PCApplyBAorAB(ctx->ksp,X,Y,ctx->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    KSPComputeOperator - Computes the explicit preconditioned operator, including diagonal scaling and null
    space removal if applicable.

    Collective on ksp

    Input Parameter:
+   ksp - the Krylov subspace context
-   mattype - the matrix type to be used

    Output Parameter:
.   mat - the explict preconditioned operator

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
  PetscErrorCode ierr;
  PetscInt       N,M,m,n;
  Mat_KSP        ctx;
  Mat            A,Aksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(mat,3);
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)ksp),m,n,M,N,&ctx,&Aksp);CHKERRQ(ierr);
  ierr = MatShellSetOperation(Aksp,MATOP_MULT,(void (*)(void))MatMult_KSP);CHKERRQ(ierr);
  ierr = MatShellSetOperation(Aksp,MATOP_CREATE_VECS,(void (*)(void))MatCreateVecs_KSP);CHKERRQ(ierr);
  ctx.ksp = ksp;
  ierr = MatCreateVecs(A,&ctx.work,NULL);CHKERRQ(ierr);
  ierr = MatComputeOperator(Aksp,mattype,mat);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.work);CHKERRQ(ierr);
  ierr = MatDestroy(&Aksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPComputeEigenvaluesExplicitly - Computes all of the eigenvalues of the
   preconditioned operator using LAPACK.

   Collective on ksp

   Input Parameter:
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
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  MPI_Comm          comm;
  PetscScalar       *array;
  Mat               A;
  PetscInt          m,row,nz,i,n,dummy;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  ierr = KSPComputeOperator(ksp,MATDENSE,&BA);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = MatGetSize(BA,&n,&n);CHKERRQ(ierr);
  if (size > 1) { /* assemble matrix on first processor */
    ierr = MatCreate(PetscObjectComm((PetscObject)ksp),&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,n,n,n,n);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,n,n);CHKERRQ(ierr);
    }
    ierr = MatSetType(A,MATMPIDENSE);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)BA,(PetscObject)A);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(BA,&row,&dummy);CHKERRQ(ierr);
    ierr = MatGetLocalSize(BA,&m,&dummy);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = MatGetRow(BA,row,&nz,&cols,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(BA,row,&nz,&cols,&vals);CHKERRQ(ierr);
      row++;
    }

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatDenseGetArray(A,&array);CHKERRQ(ierr);
  } else {
    ierr = MatDenseGetArray(BA,&array);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_ESSL)
  /* ESSL has a different calling sequence for dgeev() and zgeev() than standard LAPACK */
  if (!rank) {
    PetscScalar  sdummy,*cwork;
    PetscReal    *work,*realpart;
    PetscBLASInt clen,idummy,lwork,bn,zero = 0;
    PetscInt     *perm;

#if !defined(PETSC_USE_COMPLEX)
    clen = n;
#else
    clen = 2*n;
#endif
    ierr   = PetscMalloc1(clen,&cwork);CHKERRQ(ierr);
    idummy = -1;                /* unused */
    ierr   = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    lwork  = 5*n;
    ierr   = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
    ierr   = PetscMalloc1(n,&realpart);CHKERRQ(ierr);
    ierr   = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_(&zero,array,&bn,cwork,&sdummy,&idummy,&idummy,&bn,work,&lwork));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);

    /* For now we stick with the convention of storing the real and imaginary
       components of evalues separately.  But is this what we really want? */
    ierr = PetscMalloc1(n,&perm);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
    for (i=0; i<n; i++) {
      realpart[i] = cwork[2*i];
      perm[i]     = i;
    }
    ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      r[i] = cwork[2*perm[i]];
      c[i] = cwork[2*perm[i]+1];
    }
#else
    for (i=0; i<n; i++) {
      realpart[i] = PetscRealPart(cwork[i]);
      perm[i]     = i;
    }
    ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      r[i] = PetscRealPart(cwork[perm[i]]);
      c[i] = PetscImaginaryPart(cwork[perm[i]]);
    }
#endif
    ierr = PetscFree(perm);CHKERRQ(ierr);
    ierr = PetscFree(realpart);CHKERRQ(ierr);
    ierr = PetscFree(cwork);CHKERRQ(ierr);
  }
#elif !defined(PETSC_USE_COMPLEX)
  if (!rank) {
    PetscScalar  *work;
    PetscReal    *realpart,*imagpart;
    PetscBLASInt idummy,lwork;
    PetscInt     *perm;

    idummy   = n;
    lwork    = 5*n;
    ierr     = PetscMalloc2(n,&realpart,n,&imagpart);CHKERRQ(ierr);
    ierr     = PetscMalloc1(5*n,&work);CHKERRQ(ierr);
#if defined(PETSC_MISSING_LAPACK_GEEV)
    SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"GEEV - Lapack routine is unavailable\nNot able to provide eigen values.");
#else
    {
      PetscBLASInt lierr;
      PetscScalar  sdummy;
      PetscBLASInt bn;

      ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&bn,array,&bn,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
    }
#endif
    ierr = PetscFree(work);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&perm);CHKERRQ(ierr);

    for (i=0; i<n; i++)  perm[i] = i;
    ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      r[i] = realpart[perm[i]];
      c[i] = imagpart[perm[i]];
    }
    ierr = PetscFree(perm);CHKERRQ(ierr);
    ierr = PetscFree2(realpart,imagpart);CHKERRQ(ierr);
  }
#else
  if (!rank) {
    PetscScalar  *work,*eigs;
    PetscReal    *rwork;
    PetscBLASInt idummy,lwork;
    PetscInt     *perm;

    idummy = n;
    lwork  = 5*n;
    ierr   = PetscMalloc1(5*n,&work);CHKERRQ(ierr);
    ierr   = PetscMalloc1(2*n,&rwork);CHKERRQ(ierr);
    ierr   = PetscMalloc1(n,&eigs);CHKERRQ(ierr);
#if defined(PETSC_MISSING_LAPACK_GEEV)
    SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"GEEV - Lapack routine is unavailable\nNot able to provide eigen values.");
#else
    {
      PetscBLASInt lierr;
      PetscScalar  sdummy;
      PetscBLASInt nb;
      ierr = PetscBLASIntCast(n,&nb);CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&nb,array,&nb,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,rwork,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
    }
#endif
    ierr = PetscFree(work);CHKERRQ(ierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&perm);CHKERRQ(ierr);
    for (i=0; i<n; i++) perm[i] = i;
    for (i=0; i<n; i++) r[i]    = PetscRealPart(eigs[i]);
    ierr = PetscSortRealWithPermutation(n,r,perm);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      r[i] = PetscRealPart(eigs[perm[i]]);
      c[i] = PetscImaginaryPart(eigs[perm[i]]);
    }
    ierr = PetscFree(perm);CHKERRQ(ierr);
    ierr = PetscFree(eigs);CHKERRQ(ierr);
  }
#endif
  if (size > 1) {
    ierr = MatDenseRestoreArray(A,&array);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  } else {
    ierr = MatDenseRestoreArray(BA,&array);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&BA);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscReal      xmin,xmax,ymin,ymax,*xloc,*yloc,*value,px0,py0,rscale,iscale;
  PetscInt       M,N,i,j;
  PetscMPIInt    rank;
  PetscViewer    viewer;
  PetscDraw      draw;
  PetscDrawAxis  drawaxis;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ksp),&rank);CHKERRQ(ierr);
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
  ierr = PetscMalloc3(M,&xloc,N,&yloc,M*N,&value);CHKERRQ(ierr);
  for (i=0; i<M; i++) xloc[i] = xmin - 0.1*(xmax-xmin) + 1.2*(xmax-xmin)*i/(M-1);
  for (i=0; i<N; i++) yloc[i] = ymin - 0.1*(ymax-ymin) + 1.2*(ymax-ymin)*i/(N-1);
  ierr   = PolyEval(neig,r,c,0,0,&px0,&py0);CHKERRQ(ierr);
  rscale = px0/(PetscSqr(px0)+PetscSqr(py0));
  iscale = -py0/(PetscSqr(px0)+PetscSqr(py0));
  for (j=0; j<N; j++) {
    for (i=0; i<M; i++) {
      PetscReal px,py,tx,ty,tmod;
      ierr = PolyEval(neig,r,c,xloc[i],yloc[j],&px,&py);CHKERRQ(ierr);
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
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Iteratively Computed Eigen-contours",PETSC_DECIDE,PETSC_DECIDE,450,450,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawTensorContour(draw,M,N,NULL,NULL,value);CHKERRQ(ierr);
  if (0) {
    ierr = PetscDrawAxisCreate(draw,&drawaxis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLimits(drawaxis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(drawaxis,"Eigen-counters","real","imag");CHKERRQ(ierr);
    ierr = PetscDrawAxisDraw(drawaxis);CHKERRQ(ierr);
    ierr = PetscDrawAxisDestroy(&drawaxis);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFree3(xloc,yloc,value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
