
#include <petsc-private/kspimpl.h>   /*I "petscksp.h" I*/
#include <petscblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "KSPComputeExplicitOperator"
/*@
    KSPComputeExplicitOperator - Computes the explicit preconditioned operator.  

    Collective on KSP

    Input Parameter:
.   ksp - the Krylov subspace context

    Output Parameter:
.   mat - the explict preconditioned operator

    Notes:
    This computation is done by applying the operators to columns of the 
    identity matrix.

    Currently, this routine uses a dense matrix format when 1 processor
    is used and a sparse format otherwise.  This routine is costly in general,
    and is recommended for use only with relatively small systems.

    Level: advanced
   
.keywords: KSP, compute, explicit, operator

.seealso: KSPComputeEigenvaluesExplicitly(), PCComputeExplicitOperator()
@*/
PetscErrorCode  KSPComputeExplicitOperator(KSP ksp,Mat *mat)
{
  Vec            in,out;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       i,M,m,*rows,start,end;
  Mat            A;
  MPI_Comm       comm;
  PetscScalar    *array,one = 1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(mat,2);
  comm = ((PetscObject)ksp)->comm;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = VecDuplicate(ksp->vec_sol,&in);CHKERRQ(ierr);
  ierr = VecDuplicate(ksp->vec_sol,&out);CHKERRQ(ierr);
  ierr = VecGetSize(in,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(in,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,m,M,M);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(*mat,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*mat,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*mat,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*mat,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = MatSetOption(*mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperators(ksp->pc,&A,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  for (i=0; i<M; i++) {

    ierr = VecSet(in,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = KSP_MatMult(ksp,A,in,out);CHKERRQ(ierr);
    ierr = KSP_PCApply(ksp,out,in);CHKERRQ(ierr);
    
    ierr = VecGetArray(in,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(in,&array);CHKERRQ(ierr);

  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(&in);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPComputeEigenvaluesExplicitly"
/*@
   KSPComputeEigenvaluesExplicitly - Computes all of the eigenvalues of the 
   preconditioned operator using LAPACK.  

   Collective on KSP

   Input Parameter:
+  ksp - iterative context obtained from KSPCreate()
-  n - size of arrays r and c

   Output Parameters:
+  r - real part of computed eigenvalues
-  c - complex part of computed eigenvalues

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

.keywords: KSP, compute, eigenvalues, explicitly

.seealso: KSPComputeEigenvalues(), KSPMonitorSingularValue(), KSPComputeExtremeSingularValues(), KSPSetOperators(), KSPSolve()
@*/
PetscErrorCode  KSPComputeEigenvaluesExplicitly(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c) 
{
  Mat                BA;
  PetscErrorCode     ierr;
  PetscMPIInt        size,rank;
  MPI_Comm           comm = ((PetscObject)ksp)->comm;
  PetscScalar        *array;
  Mat                A;
  PetscInt           m,row,nz,i,n,dummy;
  const PetscInt     *cols;
  const PetscScalar  *vals;

  PetscFunctionBegin;
  ierr = KSPComputeExplicitOperator(ksp,&BA);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = MatGetSize(BA,&n,&n);CHKERRQ(ierr);
  if (size > 1) { /* assemble matrix on first processor */
    ierr = MatCreate(((PetscObject)ksp)->comm,&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,n,n,n,n);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,n,n);CHKERRQ(ierr);
    }
    ierr = MatSetType(A,MATMPIDENSE);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(A,PETSC_NULL);
    ierr = PetscLogObjectParent(BA,A);CHKERRQ(ierr);

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
    ierr = MatGetArray(A,&array);CHKERRQ(ierr);
  } else {
    ierr = MatGetArray(BA,&array);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_ESSL)
  /* ESSL has a different calling sequence for dgeev() and zgeev() than standard LAPACK */
  if (!rank) {
    PetscScalar  sdummy,*cwork;
    PetscReal    *work,*realpart;
    PetscBLASInt clen,idummy,lwork,bn,zero = 0;
    PetscInt *perm;

#if !defined(PETSC_USE_COMPLEX)
    clen = n;
#else
    clen = 2*n;
#endif
    ierr   = PetscMalloc(clen*sizeof(PetscScalar),&cwork);CHKERRQ(ierr);
    idummy = -1;                /* unused */
    bn = PetscBLASIntCast(n);
    lwork  = 5*n;
    ierr   = PetscMalloc(lwork*sizeof(PetscReal),&work);CHKERRQ(ierr);
    ierr   = PetscMalloc(n*sizeof(PetscReal),&realpart);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    LAPACKgeev_(&zero,array,&bn,cwork,&sdummy,&idummy,&idummy,&bn,work,&lwork);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);

    /* For now we stick with the convention of storing the real and imaginary
       components of evalues separately.  But is this what we really want? */
    ierr = PetscMalloc(n*sizeof(PetscInt),&perm);CHKERRQ(ierr);

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
    ierr     = PetscMalloc(2*n*sizeof(PetscReal),&realpart);CHKERRQ(ierr);
    imagpart = realpart + n;
    ierr     = PetscMalloc(5*n*sizeof(PetscReal),&work);CHKERRQ(ierr);
#if defined(PETSC_MISSING_LAPACK_GEEV) 
    SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"GEEV - Lapack routine is unavailable\nNot able to provide eigen values.");
#else
    {
      PetscBLASInt lierr;
      PetscScalar sdummy;
      PetscBLASInt bn = PetscBLASIntCast(n);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      LAPACKgeev_("N","N",&bn,array,&bn,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr);
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
    }
#endif
    ierr = PetscFree(work);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscInt),&perm);CHKERRQ(ierr);
    for (i=0; i<n; i++) { perm[i] = i;}
    ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      r[i] = realpart[perm[i]];
      c[i] = imagpart[perm[i]];
    }
    ierr = PetscFree(perm);CHKERRQ(ierr);
    ierr = PetscFree(realpart);CHKERRQ(ierr);
  }
#else
  if (!rank) {
    PetscScalar  *work,*eigs;
    PetscReal    *rwork;
    PetscBLASInt idummy,lwork;
    PetscInt     *perm;

    idummy   = n;
    lwork    = 5*n;
    ierr = PetscMalloc(5*n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    ierr = PetscMalloc(2*n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscScalar),&eigs);CHKERRQ(ierr);
#if defined(PETSC_MISSING_LAPACK_GEEV) 
    SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"GEEV - Lapack routine is unavailable\nNot able to provide eigen values.");
#else
    {
      PetscBLASInt lierr;
      PetscScalar  sdummy;
      PetscBLASInt nb = PetscBLASIntCast(n);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      LAPACKgeev_("N","N",&nb,array,&nb,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,rwork,&lierr);
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
    }
#endif
    ierr = PetscFree(work);CHKERRQ(ierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscInt),&perm);CHKERRQ(ierr);
    for (i=0; i<n; i++) { perm[i] = i;}
    for (i=0; i<n; i++) { r[i]    = PetscRealPart(eigs[i]);}
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
    ierr = MatRestoreArray(A,&array);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  } else {
    ierr = MatRestoreArray(BA,&array);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&BA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
