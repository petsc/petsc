#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: elapack.c,v 1.5 1997/08/27 14:58:52 curfman Exp curfman $";
#endif

/*

       Code that uses LAPACK to compute the eigenvalues of a matrix.
 */

#include <math.h>
#include <stdio.h>
#include "src/ec/ecimpl.h"       /*I  "ec.h"  I*/
#include "pinclude/pviewer.h"

/* 
   Note:  Both the real and complex numbers versions use r and c as allocated below;
          the complex version also uses cwork.
 */
         
typedef struct {
  double *r,*c;   /* work space */
  Scalar *cwork;  /* work space for complex version */
} EC_Lapack;

#undef __FUNC__  
#define __FUNC__ "ECSetUp_Lapack"
static int    ECSetUp_Lapack(EC ec)
{
  int       n,ierr;
  EC_Lapack *la = (EC_Lapack*) ec->data;

  ierr = MatGetSize(ec->A,&n,&n); CHKERRQ(ierr);
  ec->n        = n;
  la->r        = (double *) PetscMalloc( 2*n*sizeof(double) );CHKPTRQ(la->r);
  la->c        = la->r + n;
#if defined(PETSC_COMPLEX)
  la->cwork    = (Scalar *) PetscMalloc( n*sizeof(Scalar) );CHKPTRQ(la->cwork);
#else
  la->cwork    = 0;
#endif
  ec->realpart = (double *) PetscMalloc( 2*n*sizeof(double) );CHKPTRQ(ec->realpart);
  ec->imagpart = ec->realpart + n;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECDestroy_Lapack" 
static int ECDestroy_Lapack(PetscObject obj)
{
  EC        ec = (EC) obj;
  EC_Lapack *la = (EC_Lapack*) ec->data;

  if (la->r)        PetscFree(la->r);
  if (la->cwork)    PetscFree(la->cwork);
  if (ec->realpart) PetscFree(ec->realpart);

  PetscFree(la);
  return 0;
}
#include "pinclude/plapack.h"

#undef __FUNC__  
#define __FUNC__ "ECSolve_Lapack"
static int ECSolve_Lapack(EC ec)
{
  EC_Lapack *la = (EC_Lapack*) ec->data;
  Mat        A = 0,BA = ec->A;
  int        m,size,i,ierr,n,row,dummy,nz,*cols,rank;
  MPI_Comm   comm;
  Scalar     *vals,*array;

  if (ec->problemtype == EC_GENERALIZED_EIGENVALUE) {
    SETERRQ(PETSC_ERR_SUP,1,"Not coded for generalized eigenvalue problem");
  }

  comm = ec->comm;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  ierr     = MatGetSize(BA,&n,&n); CHKERRQ(ierr);
  if (size > 1) { /* assemble matrix on first processor */
    if (!rank) {
      ierr = MatCreateMPIDense(comm,n,n,n,n,PETSC_NULL,&A); CHKERRQ(ierr);
    }
    else {
      ierr = MatCreateMPIDense(comm,0,n,n,n,PETSC_NULL,&A); CHKERRQ(ierr);
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
    MatType type;
    ierr = MatGetType(BA,&type,PETSC_NULL); CHKERRQ(ierr);
    if (type == MATSEQDENSE) {
      ierr = MatGetArray(BA,&array); CHKERRQ(ierr);
    } else {
      ierr = MatConvert(BA,MATSEQDENSE,&A); CHKERRQ(ierr);
      ierr = MatGetArray(A,&array); CHKERRQ(ierr);
    }
  }
#if !defined(PETSC_COMPLEX)
  if (!rank) {
    Scalar *work, sdummy;
    double *realpart = ec->realpart, *imagpart = ec->imagpart;
    double *r = la->r, *c = la->c;
    int    idummy, lwork, *perm;

    idummy   = n;
    lwork    = 5*n;
    work     = (double *) PetscMalloc( 5*n*sizeof(double) ); CHKPTRQ(work);
    LAgeev_("N","N",&n,array,&n,r,c,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&ierr);
    if (ierr) SETERRQ(1,0,"Error in LAPACK routine");
    PetscFree(work);
    perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);
    for ( i=0; i<n; i++ ) {perm[i] = i;}
    ierr = PetscSortDoubleWithPermutation(n,r,perm); CHKERRQ(ierr);
    for ( i=0; i<n; i++ ) {
      realpart[i] = r[perm[i]];
      imagpart[i] = c[perm[i]];
    }
    PetscFree(perm);
    MPI_Bcast(realpart,2*n,MPI_DOUBLE,0,comm);
  } else {
    MPI_Bcast(ec->realpart,2*ec->n,MPI_DOUBLE,0,comm);
  }
#else
  if (!rank) {
    Scalar *work, sdummy;
    double *realpart = ec->realpart, *imagpart = ec->imagpart;
    double *r = la->r, *c = la->c;
    int    idummy, lwork, *perm;

    idummy   = n;
    lwork    = 5*n;
    work     = (Scalar *) PetscMalloc( 5*n*sizeof(Scalar) ); CHKPTRQ(work);
    LAgeev_("N","N",&n,array,&n,la->cwork,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,r,&ierr);
    if (ierr) SETERRQ(1,0,"Error in LAPACK routine");
    printf("optimal work dim = %g\n",work[0]);
    PetscFree(work);

    /* For now we stick with the convention of storing the real and imaginary
       components of evalues separately.  But is this what we really want? */
    perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);
    for ( i=0; i<n; i++ ) {
      r[i] = PetscReal(la->cwork[i]);
      c[i] = PetscImaginary(la->cwork[i]);
      perm[i] = i;
    }
    ierr = PetscSortDoubleWithPermutation(n,r,perm); CHKERRQ(ierr);
    for ( i=0; i<n; i++ ) {
      realpart[i] = r[perm[i]];
      imagpart[i] = c[perm[i]];
    }
    PetscFree(perm);
    MPI_Bcast(realpart,2*n,MPI_DOUBLE,0,comm);
  } else {
    MPI_Bcast(ec->realpart,2*ec->n,MPI_DOUBLE,0,comm);
  }
#endif
  if (A) {ierr = MatDestroy(A); CHKERRQ(ierr);}
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ECCreate_Lapack"
int ECCreate_Lapack(EC ec)
{
  EC_Lapack *la = PetscNew(EC_Lapack); CHKPTRQ(la);

  la->r              = 0;
  la->c              = 0;
  ec->realpart       = 0;
  ec->imagpart       = 0;

  ec->view           = 0;
  ec->destroy        = ECDestroy_Lapack;
  ec->solve          = ECSolve_Lapack;
  ec->setup          = ECSetUp_Lapack;

  ec->type           = EC_LAPACK;
  ec->n              = 0;
  ec->A              = 0;
  ec->data           = (void *) la;
  return 0;
}
