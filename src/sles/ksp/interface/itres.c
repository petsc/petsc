#ifndef lint
static char vcid[] = "$Id: itres.c,v 1.22 1996/09/12 16:25:09 bsmith Exp bsmith $";
#endif

#include "src/ksp/kspimpl.h"   /*I "ksp.h" I*/
/*@
   KSPResidual - Computes the residual.

   Input Parameters:
.  vsoln    - solution to use in computing residual
.  vt1, vt2 - temporary work vectors
.  vres     - calculated residual
.  vbinvf   - the result of binv^{-1} b.  If null, don't do it.
.  vb       - right-hand-side vector

   Notes:
   This routine assumes that an iterative method, designed for
$     A x = b
   will be used with a preconditioner, C, such that the actual problem is
$     M u = f    
   where M = AC (right preconditioning) or CA (left preconditioning).

.keywords: KSP, residual
@*/
int KSPResidual(KSP ksp,Vec vsoln,Vec vt1,Vec vt2,Vec vres, Vec vbinvf,Vec vb)
{
  Scalar        one = -1.0;
  MatStructure  pflag;
  Mat           Amat, Pmat;
  int           ierr;

  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);
  if (ksp->pc_side == PC_RIGHT) {
    if (vbinvf) {ierr = VecCopy(vb,vbinvf); CHKERRQ(ierr);}
    vbinvf = vb;
  }
  else if (ksp->pc_side == PC_LEFT) {
    ierr = PCApply(ksp->B,vb,vbinvf); CHKERRQ(ierr);
  }
  else {
    SETERRQ(1,"KSPResidual: Only right and left preconditioning are currently supported.");
  }
  if (!ksp->guess_zero) {
    /* compute initial residual: f - M*x */
    /* (inv(b)*a)*x or (a*inv(b)*b)*x into dest */
    if (ksp->pc_side == PC_RIGHT) {
      /* we want a * binv * b * x, or just a * x for the first step */
      /* a*x into temp */
      ierr = MatMult(Amat,vsoln,vt1); CHKERRQ(ierr);
    }
    else {
      /* else we do binv * a * x */
      ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,vsoln,vt1,vt2); CHKERRQ(ierr);
    }
    /* This is an extra copy for the right-inverse case */
    ierr = VecCopy(vbinvf,vres); CHKERRQ(ierr);
    ierr = VecAXPY(&one,vt1,vres); CHKERRQ(ierr);
          /* inv(b)(f - a*x) into dest */
  }
  else {
    ierr = VecCopy(vbinvf,vres); CHKERRQ(ierr);
  }
  return 0;
}

/*@
   KSPUnwindPreconditioner - Unwinds the preconditioning in the solution.

   Input Parameters:
.  ksp  - iterative context
.  vsoln - solution vector 
.  vt1   - temporary work vector

   Output Parameter:
.  vsoln - contains solution on output  

   Notes:
   If preconditioning either symmetrically or on the right, this routine solves
   for the correction to the unpreconditioned problem.  If preconditioning on
   the left, nothing is done.

.keywords: KSP, unwind, preconditioner

.seealso: KSPSetPreconditionerSide()
@*/
int KSPUnwindPreconditioner(KSP ksp,Vec vsoln,Vec vt1)
{
  int ierr;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->pc_side == PC_RIGHT) {
    ierr = PCApply(ksp->B,vsoln,vt1); CHKERRQ(ierr);
    ierr = VecCopy(vt1,vsoln); CHKERRQ(ierr);
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    ierr = PCApplySymmetricRight(ksp->B,vsoln,vt1); CHKERRQ(ierr);
    ierr = VecCopy(vt1,vsoln); CHKERRQ(ierr);
  }
  return 0;
}

/*@
       KSPComputeExplicitOperator - Computes as a dense matrix the explicit 
          preconditioned operator. This is done by applying the operators to 
          columns of the identity matrix.

  Input Parameter:
.   ksp - the Krylov subspace context

  Output Parameter:
.   mat - the explict operator

@*/
int KSPComputeExplicitOperator(KSP ksp, Mat *mat)
{
  Vec      in,out;
  int      ierr,i,M,m,size,*rows,start,end;
  Mat      A;
  MPI_Comm comm;
  Scalar   *array,zero = 0.0,one = 1.0;

  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscValidPointer(mat);
  comm = ksp->comm;

  MPI_Comm_size(comm,&size);

  ierr = VecDuplicate(ksp->vec_sol,&in); CHKERRQ(ierr);
  ierr = VecDuplicate(ksp->vec_sol,&out); CHKERRQ(ierr);
  ierr = VecGetSize(in,&M); CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(in,&start,&end);
  rows = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(rows);
  for ( i=0; i<m; i++ ) {rows[i] = start + i;}

  if (size == 1) {
    ierr = MatCreateSeqDense(comm,M,M,PETSC_NULL,mat); CHKERRQ(ierr);
  } else {
    /*    ierr = MatCreateMPIDense(comm,m,M,M,M,PETSC_NULL,mat); CHKERRQ(ierr); */
    ierr = MatCreateMPIAIJ(comm,m,m,M,M,0,0,0,0,mat); CHKERRQ(ierr);
  }
  
  ierr = PCGetOperators(ksp->B,&A,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);

  for ( i=0; i<M; i++ ) {

    ierr = VecSet(&zero,in); CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in); CHKERRQ(ierr);

    ierr = MatMult(A,in,out); CHKERRQ(ierr);
    ierr = PCApply(ksp->B,out,in); CHKERRQ(ierr);
    
    ierr = VecGetArray(in,&array); CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES); CHKERRQ(ierr); 
    ierr = VecRestoreArray(in,&array); CHKERRQ(ierr);

  }
  PetscFree(rows);
  ierr = VecDestroy(in); CHKERRQ(ierr);
  ierr = VecDestroy(out); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


#include "pinclude/plapack.h"

/*@
   KSPComputeEigenvaluesExplicitly - Computes all of the eigenvalues of the 
          preconditioned operator using LAPACK. This is very slow but will generally
          provide accurate eigenvalue estimates. It will only run for small 
          problems, say n < 500. It explicitly forms a dense matrix representing 
          the preconditioned operator.

   Input Parameter:
.  ksp - iterative context obtained from KSPCreate()
.  n - size of arrays r and c

   Output Parameters:
.  r - real part of computed eigenvalues
.  c - complex part of computed eigenvalues

   Many users may just want to use the monitoring routine
   KSPSingularValueMonitor() (which can be set with option -ksp_singmonitor)
   to print the singular values at each iteration of the linear solve.

.keywords: KSP, compute, extreme, singular, values

.seealso: KSPComputeEigenvalues(), KSPSingularValueMonitor(), KSPComputeExtremeSingularValues()
@*/
int KSPComputeEigenvaluesExplicitly(KSP ksp,int nmax,double *r,double *c) 
{
  Mat          BA;
  int          i,n,ierr,size,rank,dummy;
  MPI_Comm     comm = ksp->comm;
  Scalar       *array;
  Mat          A;
  int          m,row, nz, *cols;
  Scalar       *vals;

  ierr =  KSPComputeExplicitOperator(ksp,&BA); CHKERRQ(ierr);
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  ierr     = MatGetSize(BA,&n,&n); CHKERRQ(ierr);
  if (size > 1) { /* assemble matrix on first processor */
    if (!rank) {
      ierr = MatCreateMPIDense(ksp->comm,n,n,n,n,PETSC_NULL,&A); CHKERRQ(ierr);
    }
    else {
      ierr = MatCreateMPIDense(ksp->comm,0,n,n,n,PETSC_NULL,&A); CHKERRQ(ierr);
    }
    PLogObjectParent(BA,A);

    ierr = MatGetOwnershipRange(BA,&row,&dummy); CHKERRQ(ierr);
    ierr = MatGetLocalSize(BA,&m,&dummy); CHKERRQ(ierr);
    for ( i=0; i<m; i++ ) {
      ierr = MatGetRow(BA,row,&nz,&cols,&vals); CHKERRQ(ierr);
      ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatRestoreRow(BA,row,&nz,&cols,&vals); CHKERRQ(ierr);
      row++;
    } 

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatGetArray(A,&array); CHKERRQ(ierr);
  } else {
    ierr     = MatGetArray(BA,&array); CHKERRQ(ierr);
  }

#if !defined(PETSC_COMPLEX)
  if (!rank) {
    Scalar *work,sdummy;
    double *realpart,*imagpart;
    int    idummy,lwork,*perm;

    idummy   = n;
    lwork    = 5*n;
    realpart = (double *) PetscMalloc( 2*n*sizeof(double) ); CHKPTRQ(r);
    imagpart = realpart + n;
    work     = (double *) PetscMalloc( 5*n*sizeof(double) ); CHKPTRQ(work);
    LAgeev_("N","N",&n,array,&n,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&ierr);
    if (ierr) SETERRQ(1,"KSPComputeEigenvaluesExplicitly:Error in Lapack routine");
    PetscFree(work);
    perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);
    for ( i=0; i<n; i++ ) { perm[i] = i;}
    ierr = PetscSortDoubleWithPermutation(n,realpart,perm); CHKERRQ(ierr);
    for ( i=0; i<n; i++ ) {
      r[i] = realpart[perm[i]];
      c[i] = imagpart[perm[i]];
    }
    PetscFree(perm);
    PetscFree(realpart);
  }
#else
  if (!rank) {
    Scalar *work,sdummy,*eigs;
    double *rwork;
    int    idummy,lwork,*perm;

    idummy   = n;
    lwork    = 5*n;
    work     = (Scalar *) PetscMalloc( 5*n*sizeof(Scalar) ); CHKPTRQ(work);
    rwork    = (double *) PetscMalloc( 2*n*sizeof(double) ); CHKPTRQ(rwork);
    eigs     = (Scalar *) PetscMalloc( n*sizeof(Scalar) ); CHKPTRQ(eigs);
    LAgeev_("N","N",&n,array,&n,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,rwork,&ierr);
    if (ierr) SETERRQ(1,"KSPComputeEigenvaluesExplicitly:Error in Lapack routine");
    PetscFree(work);
    PetscFree(rwork);
    perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);
    for ( i=0; i<n; i++ ) { perm[i] = i;}
    for ( i=0; i<n; i++ ) { r[i]    = real(eigs[i]);}
    ierr = PetscSortDoubleWithPermutation(n,r,perm); CHKERRQ(ierr);
    for ( i=0; i<n; i++ ) {
      r[i] = PetscReal(eigs[perm[i]]);
      c[i] = imag(eigs[perm[i]]);
    }
    PetscFree(perm);
    PetscFree(eigs);
  }
#endif  
  if (size > 1) {
    ierr = MatRestoreArray(A,&array); CHKERRQ(ierr);
    ierr = MatDestroy(A); CHKERRQ(ierr);
  } else {
    ierr = MatRestoreArray(BA,&array); CHKERRQ(ierr);
  }
  ierr = MatDestroy(BA); CHKERRQ(ierr);
  return 0;
}
