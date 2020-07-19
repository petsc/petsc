
/*
  Defines the basic matrix operations for the KAIJ  matrix storage format.
  This format is used to evaluate matrices of the form:

    [I \otimes S + A \otimes T]

  where
    S is a dense (p \times q) matrix
    T is a dense (p \times q) matrix
    A is an AIJ  (n \times n) matrix
    I is the identity matrix

  The resulting matrix is (np \times nq)

  We provide:
     MatMult()
     MatMultAdd()
     MatInvertBlockDiagonal()
  and
     MatCreateKAIJ(Mat,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],Mat*)

  This single directory handles both the sequential and parallel codes
*/

#include <../src/mat/impls/kaij/kaij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petsc/private/vecimpl.h>

/*@C
   MatKAIJGetAIJ - Get the AIJ matrix describing the blockwise action of the KAIJ matrix

   Not Collective, but if the KAIJ matrix is parallel, the AIJ matrix is also parallel

   Input Parameter:
.  A - the KAIJ matrix

   Output Parameter:
.  B - the AIJ matrix

   Level: advanced

   Notes: The reference count on the AIJ matrix is not increased so you should not destroy it.

.seealso: MatCreateKAIJ()
@*/
PetscErrorCode  MatKAIJGetAIJ(Mat A,Mat *B)
{
  PetscErrorCode ierr;
  PetscBool      ismpikaij,isseqkaij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIKAIJ,&ismpikaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQKAIJ,&isseqkaij);CHKERRQ(ierr);
  if (ismpikaij) {
    Mat_MPIKAIJ *b = (Mat_MPIKAIJ*)A->data;

    *B = b->A;
  } else if (isseqkaij) {
    Mat_SeqKAIJ *b = (Mat_SeqKAIJ*)A->data;

    *B = b->AIJ;
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix passed in is not of type KAIJ");
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJGetS - Get the S matrix describing the shift action of the KAIJ matrix

   Not Collective; the entire S is stored and returned independently on all processes.

   Input Parameter:
.  A - the KAIJ matrix

   Output Parameters:
+  m - the number of rows in S
.  n - the number of columns in S
-  S - the S matrix, in form of a scalar array in column-major format

   Note: All output parameters are optional (pass NULL or PETSC_IGNORE if not desired)

   Level: advanced

.seealso: MatCreateKAIJ(), MatGetBlockSizes()
@*/
PetscErrorCode MatKAIJGetS(Mat A,PetscInt *m,PetscInt *n,PetscScalar **S)
{
  Mat_SeqKAIJ *b = (Mat_SeqKAIJ*)A->data;
  PetscFunctionBegin;
  if (m) *m = b->p;
  if (n) *n = b->q;
  if (S) *S = b->S;
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJGetSRead - Get a read-only pointer to the S matrix describing the shift action of the KAIJ matrix

   Not Collective; the entire S is stored and returned independently on all processes.

   Input Parameter:
.  A - the KAIJ matrix

   Output Parameters:
+  m - the number of rows in S
.  n - the number of columns in S
-  S - the S matrix, in form of a scalar array in column-major format

   Note: All output parameters are optional (pass NULL or PETSC_IGNORE if not desired)

   Level: advanced

.seealso: MatCreateKAIJ(), MatGetBlockSizes()
@*/
PetscErrorCode MatKAIJGetSRead(Mat A,PetscInt *m,PetscInt *n,const PetscScalar **S)
{
  Mat_SeqKAIJ *b = (Mat_SeqKAIJ*)A->data;
  PetscFunctionBegin;
  if (m) *m = b->p;
  if (n) *n = b->q;
  if (S) *S = b->S;
  PetscFunctionReturn(0);
}

/*@C
  MatKAIJRestoreS - Restore array obtained with MatKAIJGetS()

  Not collective

  Input Parameter:
. A - the KAIJ matrix

  Output Parameter:
. S - location of pointer to array obtained with MatKAIJGetS()

  Note: This routine zeros the array pointer to prevent accidental reuse after it has been restored.
  If NULL is passed, it will not attempt to zero the array pointer.

  Level: advanced
.seealso: MatKAIJGetS(), MatKAIJGetSRead(), MatKAIJRestoreSRead()
@*/
PetscErrorCode MatKAIJRestoreS(Mat A,PetscScalar **S)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (S) *S = NULL;
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  MatKAIJRestoreSRead - Restore array obtained with MatKAIJGetSRead()

  Not collective

  Input Parameter:
. A - the KAIJ matrix

  Output Parameter:
. S - location of pointer to array obtained with MatKAIJGetS()

  Note: This routine zeros the array pointer to prevent accidental reuse after it has been restored.
  If NULL is passed, it will not attempt to zero the array pointer.

  Level: advanced
.seealso: MatKAIJGetS(), MatKAIJGetSRead(), MatKAIJRestoreSRead()
@*/
PetscErrorCode MatKAIJRestoreSRead(Mat A,const PetscScalar **S)
{
  PetscFunctionBegin;
  if (S) *S = NULL;
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJGetT - Get the transformation matrix T associated with the KAIJ matrix

   Not Collective; the entire T is stored and returned independently on all processes

   Input Parameter:
.  A - the KAIJ matrix

   Output Parameter:
+  m - the number of rows in T
.  n - the number of columns in T
-  T - the T matrix, in form of a scalar array in column-major format

   Note: All output parameters are optional (pass NULL or PETSC_IGNORE if not desired)

   Level: advanced

.seealso: MatCreateKAIJ(), MatGetBlockSizes()
@*/
PetscErrorCode MatKAIJGetT(Mat A,PetscInt *m,PetscInt *n,PetscScalar **T)
{
  Mat_SeqKAIJ *b = (Mat_SeqKAIJ*)A->data;
  PetscFunctionBegin;
  if (m) *m = b->p;
  if (n) *n = b->q;
  if (T) *T = b->T;
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJGetTRead - Get a read-only pointer to the transformation matrix T associated with the KAIJ matrix

   Not Collective; the entire T is stored and returned independently on all processes

   Input Parameter:
.  A - the KAIJ matrix

   Output Parameter:
+  m - the number of rows in T
.  n - the number of columns in T
-  T - the T matrix, in form of a scalar array in column-major format

   Note: All output parameters are optional (pass NULL or PETSC_IGNORE if not desired)

   Level: advanced

.seealso: MatCreateKAIJ(), MatGetBlockSizes()
@*/
PetscErrorCode MatKAIJGetTRead(Mat A,PetscInt *m,PetscInt *n,const PetscScalar **T)
{
  Mat_SeqKAIJ *b = (Mat_SeqKAIJ*)A->data;
  PetscFunctionBegin;
  if (m) *m = b->p;
  if (n) *n = b->q;
  if (T) *T = b->T;
  PetscFunctionReturn(0);
}

/*@C
  MatKAIJRestoreT - Restore array obtained with MatKAIJGetT()

  Not collective

  Input Parameter:
. A - the KAIJ matrix

  Output Parameter:
. T - location of pointer to array obtained with MatKAIJGetS()

  Note: This routine zeros the array pointer to prevent accidental reuse after it has been restored.
  If NULL is passed, it will not attempt to zero the array pointer.

  Level: advanced
.seealso: MatKAIJGetT(), MatKAIJGetTRead(), MatKAIJRestoreTRead()
@*/
PetscErrorCode MatKAIJRestoreT(Mat A,PetscScalar **T)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (T) *T = NULL;
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  MatKAIJRestoreTRead - Restore array obtained with MatKAIJGetTRead()

  Not collective

  Input Parameter:
. A - the KAIJ matrix

  Output Parameter:
. T - location of pointer to array obtained with MatKAIJGetS()

  Note: This routine zeros the array pointer to prevent accidental reuse after it has been restored.
  If NULL is passed, it will not attempt to zero the array pointer.

  Level: advanced
.seealso: MatKAIJGetT(), MatKAIJGetTRead(), MatKAIJRestoreTRead()
@*/
PetscErrorCode MatKAIJRestoreTRead(Mat A,const PetscScalar **T)
{
  PetscFunctionBegin;
  if (T) *T = NULL;
  PetscFunctionReturn(0);
}

/*@
   MatKAIJSetAIJ - Set the AIJ matrix describing the blockwise action of the KAIJ matrix

   Logically Collective; if the AIJ matrix is parallel, the KAIJ matrix is also parallel

   Input Parameters:
+  A - the KAIJ matrix
-  B - the AIJ matrix

   Notes:
   This function increases the reference count on the AIJ matrix, so the user is free to destroy the matrix if it is not needed.
   Changes to the entries of the AIJ matrix will immediately affect the KAIJ matrix.

   Level: advanced

.seealso: MatKAIJGetAIJ(), MatKAIJSetS(), MatKAIJSetT()
@*/
PetscErrorCode MatKAIJSetAIJ(Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size == 1) {
    Mat_SeqKAIJ *a = (Mat_SeqKAIJ*)A->data;
    a->AIJ = B;
  } else {
    Mat_MPIKAIJ *a = (Mat_MPIKAIJ*)A->data;
    a->A = B;
  }
  ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJSetS - Set the S matrix describing the shift action of the KAIJ matrix

   Logically Collective; the entire S is stored independently on all processes.

   Input Parameters:
+  A - the KAIJ matrix
.  p - the number of rows in S
.  q - the number of columns in S
-  S - the S matrix, in form of a scalar array in column-major format

   Notes: The dimensions p and q must match those of the transformation matrix T associated with the KAIJ matrix.
   The S matrix is copied, so the user can destroy this array.

   Level: Advanced

.seealso: MatKAIJGetS(), MatKAIJSetT(), MatKAIJSetAIJ()
@*/
PetscErrorCode MatKAIJSetS(Mat A,PetscInt p,PetscInt q,const PetscScalar S[])
{
  PetscErrorCode ierr;
  Mat_SeqKAIJ    *a = (Mat_SeqKAIJ*)A->data;

  PetscFunctionBegin;
  ierr = PetscFree(a->S);CHKERRQ(ierr);
  if (S) {
    ierr = PetscMalloc1(p*q*sizeof(PetscScalar),&a->S);CHKERRQ(ierr);
    ierr = PetscMemcpy(a->S,S,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
  } else  a->S = NULL;

  a->p = p;
  a->q = q;
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJGetScaledIdentity - Check if both S and T are scaled identities.

   Logically Collective.

   Input Parameter:
.  A - the KAIJ matrix

  Output Parameter:
.  identity - the Boolean value

   Level: Advanced

.seealso: MatKAIJGetS(), MatKAIJGetT()
@*/
PetscErrorCode MatKAIJGetScaledIdentity(Mat A,PetscBool* identity)
{
  Mat_SeqKAIJ *a = (Mat_SeqKAIJ*)A->data;
  PetscInt    i,j;

  PetscFunctionBegin;
  if (a->p != a->q) {
    *identity = PETSC_FALSE;
    PetscFunctionReturn(0);
  } else *identity = PETSC_TRUE;
  if (!a->isTI || a->S) {
    for (i=0; i<a->p && *identity; i++) {
      for (j=0; j<a->p && *identity; j++) {
        if (i != j) {
          if(a->S && PetscAbsScalar(a->S[i+j*a->p]) > PETSC_SMALL) *identity = PETSC_FALSE;
          if(a->T && PetscAbsScalar(a->T[i+j*a->p]) > PETSC_SMALL) *identity = PETSC_FALSE;
        } else {
          if(a->S && PetscAbsScalar(a->S[i*(a->p+1)]-a->S[0]) > PETSC_SMALL) *identity = PETSC_FALSE;
          if(a->T && PetscAbsScalar(a->T[i*(a->p+1)]-a->T[0]) > PETSC_SMALL) *identity = PETSC_FALSE;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJSetT - Set the transformation matrix T associated with the KAIJ matrix

   Logically Collective; the entire T is stored independently on all processes.

   Input Parameters:
+  A - the KAIJ matrix
.  p - the number of rows in S
.  q - the number of columns in S
-  T - the T matrix, in form of a scalar array in column-major format

   Notes: The dimensions p and q must match those of the shift matrix S associated with the KAIJ matrix.
   The T matrix is copied, so the user can destroy this array.

   Level: Advanced

.seealso: MatKAIJGetT(), MatKAIJSetS(), MatKAIJSetAIJ()
@*/
PetscErrorCode MatKAIJSetT(Mat A,PetscInt p,PetscInt q,const PetscScalar T[])
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  Mat_SeqKAIJ    *a = (Mat_SeqKAIJ*)A->data;
  PetscBool      isTI = PETSC_FALSE;

  PetscFunctionBegin;
  /* check if T is an identity matrix */
  if (T && (p == q)) {
    isTI = PETSC_TRUE;
    for (i=0; i<p; i++) {
      for (j=0; j<q; j++) {
        if (i == j) {
          /* diagonal term must be 1 */
          if (T[i+j*p] != 1.0) isTI = PETSC_FALSE;
        } else {
          /* off-diagonal term must be 0 */
          if (T[i+j*p] != 0.0) isTI = PETSC_FALSE;
        }
      }
    }
  }
  a->isTI = isTI;

  ierr = PetscFree(a->T);CHKERRQ(ierr);
  if (T && (!isTI)) {
    ierr = PetscMalloc1(p*q*sizeof(PetscScalar),&a->T);CHKERRQ(ierr);
    ierr = PetscMemcpy(a->T,T,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
  } else a->T = NULL;

  a->p = p;
  a->q = q;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqKAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqKAIJ    *b = (Mat_SeqKAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = PetscFree(b->S);CHKERRQ(ierr);
  ierr = PetscFree(b->T);CHKERRQ(ierr);
  ierr = PetscFree(b->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree5(b->sor.w,b->sor.y,b->sor.work,b->sor.t,b->sor.arr);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_KAIJ(Mat A)
{
  PetscErrorCode ierr;
  PetscInt       n;
  PetscMPIInt    size;
  Mat_SeqKAIJ    *seqkaij = (Mat_SeqKAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetSizes(A,seqkaij->p*seqkaij->AIJ->rmap->n,seqkaij->q*seqkaij->AIJ->cmap->n,seqkaij->p*seqkaij->AIJ->rmap->N,seqkaij->q*seqkaij->AIJ->cmap->N);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(A->rmap,seqkaij->p);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(A->cmap,seqkaij->q);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  } else {
    Mat_MPIKAIJ *a;
    Mat_MPIAIJ  *mpiaij;
    IS          from,to;
    Vec         gvec;
    PetscScalar *T;
    PetscInt    i,j;

    a = (Mat_MPIKAIJ*)A->data;
    mpiaij = (Mat_MPIAIJ*)a->A->data;
    ierr = MatSetSizes(A,a->p*a->A->rmap->n,a->q*a->A->cmap->n,a->p*a->A->rmap->N,a->q*a->A->cmap->N);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(A->rmap,seqkaij->p);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(A->cmap,seqkaij->q);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

    if (a->isTI) {
      /* If the transformation matrix associated with the parallel matrix A is the identity matrix, then a->T will be NULL.
       * In this case, if we pass a->T directly to the MatCreateKAIJ() calls to create the sequential submatrices, the routine will
       * not be able to tell that transformation matrix should be set to the identity; thus we create a temporary identity matrix
       * to pass in. */
      ierr = PetscMalloc1(a->p*a->q*sizeof(PetscScalar),&T);CHKERRQ(ierr);
      for (i=0; i<a->p; i++) {
        for (j=0; j<a->q; j++) {
          if (i==j) T[i+j*a->p] = 1.0;
          else      T[i+j*a->p] = 0.0;
        }
      }
    } else T = a->T;
    ierr = MatCreateKAIJ(mpiaij->A,a->p,a->q,a->S,T,&a->AIJ);CHKERRQ(ierr);
    ierr = MatCreateKAIJ(mpiaij->B,a->p,a->q,NULL,T,&a->OAIJ);CHKERRQ(ierr);
    if (a->isTI) {
      ierr = PetscFree(T);CHKERRQ(ierr);
    }

    ierr = VecGetSize(mpiaij->lvec,&n);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF,&a->w);CHKERRQ(ierr);
    ierr = VecSetSizes(a->w,n*a->q,n*a->q);CHKERRQ(ierr);
    ierr = VecSetBlockSize(a->w,a->q);CHKERRQ(ierr);
    ierr = VecSetType(a->w,VECSEQ);CHKERRQ(ierr);

    /* create two temporary Index sets for build scatter gather */
    ierr = ISCreateBlock(PetscObjectComm((PetscObject)a->A),a->q,n,mpiaij->garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n*a->q,0,1,&to);CHKERRQ(ierr);

    /* create temporary global vector to generate scatter context */
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)a->A),a->q,a->q*a->A->cmap->n,a->q*a->A->cmap->N,NULL,&gvec);CHKERRQ(ierr);

    /* generate the scatter context */
    ierr = VecScatterCreate(gvec,from,a->w,to,&a->ctx);CHKERRQ(ierr);

    ierr = ISDestroy(&from);CHKERRQ(ierr);
    ierr = ISDestroy(&to);CHKERRQ(ierr);
    ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  }

  A->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_KAIJ(Mat A,PetscViewer viewer)
{
  PetscViewerFormat format;
  Mat_SeqKAIJ       *a = (Mat_SeqKAIJ*)A->data;
  Mat               B;
  PetscInt          i;
  PetscErrorCode    ierr;
  PetscBool         ismpikaij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIKAIJ,&ismpikaij);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL || format == PETSC_VIEWER_ASCII_IMPL) {
    ierr = PetscViewerASCIIPrintf(viewer,"S and T have %D rows and %D columns\n",a->p,a->q);CHKERRQ(ierr);

    /* Print appropriate details for S. */
    if (!a->S) {
      ierr = PetscViewerASCIIPrintf(viewer,"S is NULL\n");CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_ASCII_IMPL) {
      ierr = PetscViewerASCIIPrintf(viewer,"Entries of S are ");CHKERRQ(ierr);
      for (i=0; i<(a->p * a->q); i++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e ",(double)PetscRealPart(a->S[i]),(double)PetscImaginaryPart(a->S[i]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)PetscRealPart(a->S[i]));CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }

    /* Print appropriate details for T. */
    if (a->isTI) {
      ierr = PetscViewerASCIIPrintf(viewer,"T is the identity matrix\n");CHKERRQ(ierr);
    } else if (!a->T) {
      ierr = PetscViewerASCIIPrintf(viewer,"T is NULL\n");CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_ASCII_IMPL) {
      ierr = PetscViewerASCIIPrintf(viewer,"Entries of T are ");CHKERRQ(ierr);
      for (i=0; i<(a->p * a->q); i++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e ",(double)PetscRealPart(a->T[i]),(double)PetscImaginaryPart(a->T[i]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)PetscRealPart(a->T[i]));CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }

    /* Now print details for the AIJ matrix, using the AIJ viewer. */
    ierr = PetscViewerASCIIPrintf(viewer,"Now viewing the associated AIJ matrix:\n");CHKERRQ(ierr);
    if (ismpikaij) {
      Mat_MPIKAIJ *b = (Mat_MPIKAIJ*)A->data;
      ierr = MatView(b->A,viewer);CHKERRQ(ierr);
    } else {
      ierr = MatView(a->AIJ,viewer);CHKERRQ(ierr);
    }

  } else {
    /* For all other matrix viewer output formats, simply convert to an AIJ matrix and call MatView() on that. */
    if (ismpikaij) {
      ierr = MatConvert(A,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    } else {
      ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    }
    ierr = MatView(B,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIKAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIKAIJ    *b = (Mat_MPIKAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = MatDestroy(&b->OAIJ);CHKERRQ(ierr);
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&b->w);CHKERRQ(ierr);
  ierr = PetscFree(b->S);CHKERRQ(ierr);
  ierr = PetscFree(b->T);CHKERRQ(ierr);
  ierr = PetscFree(b->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

/* zz = yy + Axx */
PetscErrorCode MatMultAdd_SeqKAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqKAIJ       *b = (Mat_SeqKAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *s = b->S, *t = b->T;
  const PetscScalar *x,*v,*bx;
  PetscScalar       *y,*sums;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j,l,p=b->p,q=b->q,k;

  PetscFunctionBegin;
  if (!yy) {
    ierr = VecSet(zz,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  if ((!s) && (!t) && (!b->isTI)) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  if (b->isTI) {
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sums = y + p*i;
      for (j=0; j<n; j++) {
        for (k=0; k<p; k++) {
          sums[k] += v[jrow+j]*x[q*idx[jrow+j]+k];
        }
      }
    }
    ierr = PetscLogFlops(3.0*(a->nz)*p);CHKERRQ(ierr);
  } else if (t) {
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sums = y + p*i;
      for (j=0; j<n; j++) {
        for (k=0; k<p; k++) {
          for (l=0; l<q; l++) {
            sums[k] += v[jrow+j]*t[k+l*p]*x[q*idx[jrow+j]+l];
          }
        }
      }
    }
    /* The flop count below assumes that v[jrow+j] is hoisted out (which an optimizing compiler is likely to do),
     * and also that T part is hoisted outside this loop (in exchange for temporary storage) as (A \otimes I) (I \otimes T),
     * so that this multiply doesn't have to be redone for each matrix entry, but just once per column. The latter
     * transformation is much less likely to be applied, but we nonetheless count the minimum flops required. */
    ierr = PetscLogFlops((2.0*p*q-p)*m+2.0*p*a->nz);CHKERRQ(ierr);
  }
  if (s) {
    for (i=0; i<m; i++) {
      sums = y + p*i;
      bx   = x + q*i;
      if (i < b->AIJ->cmap->n) {
        for (j=0; j<q; j++) {
          for (k=0; k<p; k++) {
            sums[k] += s[k+j*p]*bx[j];
          }
        }
      }
    }
    ierr = PetscLogFlops(2.0*m*p*q);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqKAIJ(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMultAdd_SeqKAIJ(A,xx,PETSC_NULL,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatInvertBlockDiagonal_SeqKAIJ(Mat A,const PetscScalar **values)
{
  Mat_SeqKAIJ       *b  = (Mat_SeqKAIJ*)A->data;
  Mat_SeqAIJ        *a  = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *S  = b->S;
  const PetscScalar *T  = b->T;
  const PetscScalar *v  = a->a;
  const PetscInt     p  = b->p, q = b->q, m = b->AIJ->rmap->n, *idx = a->j, *ii = a->i;
  PetscErrorCode    ierr;
  PetscInt          i,j,*v_pivots,dof,dof2;
  PetscScalar       *diag,aval,*v_work;

  PetscFunctionBegin;
  if (p != q) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATKAIJ: Block size must be square to calculate inverse.");
  if ((!S) && (!T) && (!b->isTI)) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATKAIJ: Cannot invert a zero matrix.");

  dof  = p;
  dof2 = dof*dof;

  if (b->ibdiagvalid) {
    if (values) *values = b->ibdiag;
    PetscFunctionReturn(0);
  }
  if (!b->ibdiag) {
    ierr = PetscMalloc1(dof2*m*sizeof(PetscScalar),&b->ibdiag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,dof2*m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (values) *values = b->ibdiag;
  diag = b->ibdiag;

  ierr = PetscMalloc2(dof,&v_work,dof,&v_pivots);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    if (S) {
      ierr = PetscMemcpy(diag,S,dof2*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(diag,dof2*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (b->isTI) {
      aval = 0;
      for (j=ii[i]; j<ii[i+1]; j++) if (idx[j] == i) aval = v[j];
      for (j=0; j<dof; j++) diag[j+dof*j] += aval;
    } else if (T) {
      aval = 0;
      for (j=ii[i]; j<ii[i+1]; j++) if (idx[j] == i) aval = v[j];
      for (j=0; j<dof2; j++) diag[j] += aval*T[j];
    }
    ierr = PetscKernel_A_gets_inverse_A(dof,diag,v_pivots,v_work,PETSC_FALSE,NULL);CHKERRQ(ierr);
    diag += dof2;
  }
  ierr = PetscFree2(v_work,v_pivots);CHKERRQ(ierr);

  b->ibdiagvalid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonalBlock_MPIKAIJ(Mat A,Mat *B)
{
  Mat_MPIKAIJ *kaij = (Mat_MPIKAIJ*) A->data;

  PetscFunctionBegin;
  *B = kaij->AIJ;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_SeqKAIJ(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  PetscErrorCode    ierr;
  Mat_SeqKAIJ       *kaij = (Mat_SeqKAIJ*) A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)kaij->AIJ->data;
  const PetscScalar *aa = a->a, *T = kaij->T, *v;
  const PetscInt    m  = kaij->AIJ->rmap->n, *ai=a->i, *aj=a->j, p = kaij->p, q = kaij->q, *diag, *vi;
  const PetscScalar *b, *xb, *idiag;
  PetscScalar       *x, *work, *workt, *w, *y, *arr, *t, *arrt;
  PetscInt          i, j, k, i2, bs, bs2, nz;

  PetscFunctionBegin;
  its = its*lits;
  if (flag & SOR_EISENSTAT) SETERRQ (PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  if (its <= 0)             SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift)               SETERRQ (PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for diagonal shift");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for applying upper or lower triangular parts");
  if (p != q) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatSOR for KAIJ: No support for non-square dense blocks");
  else        {bs = p; bs2 = bs*bs; }

  if (!m) PetscFunctionReturn(0);

  if (!kaij->ibdiagvalid) { ierr = MatInvertBlockDiagonal_SeqKAIJ(A,NULL);CHKERRQ(ierr); }
  idiag = kaij->ibdiag;
  diag  = a->diag;

  if (!kaij->sor.setup) {
    ierr = PetscMalloc5(bs,&kaij->sor.w,bs,&kaij->sor.y,m*bs,&kaij->sor.work,m*bs,&kaij->sor.t,m*bs2,&kaij->sor.arr);CHKERRQ(ierr);
    kaij->sor.setup = PETSC_TRUE;
  }
  y     = kaij->sor.y;
  w     = kaij->sor.w;
  work  = kaij->sor.work;
  t     = kaij->sor.t;
  arr   = kaij->sor.arr;

  ierr = VecGetArray(xx,&x);    CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      PetscKernel_w_gets_Ar_times_v(bs,bs,b,idiag,x);                            /* x[0:bs] <- D^{-1} b[0:bs] */
      ierr   =  PetscMemcpy(t,b,bs*sizeof(PetscScalar));CHKERRQ(ierr);
      i2     =  bs;
      idiag  += bs2;
      for (i=1; i<m; i++) {
        v  = aa + ai[i];
        vi = aj + ai[i];
        nz = diag[i] - ai[i];

        if (T) {                /* b - T (Arow * x) */
          ierr = PetscMemzero(w,bs*sizeof(PetscScalar));CHKERRQ(ierr);
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) w[k] -= v[j] * x[vi[j]*bs+k];
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs,w,T,&t[i2]);
          for (k=0; k<bs; k++) t[i2+k] += b[i2+k];
        } else if (kaij->isTI) {
          ierr = PetscMemcpy(t+i2,b+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) t[i2+k] -= v[j] * x[vi[j]*bs+k];
          }
        } else {
          ierr = PetscMemcpy(t+i2,b+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
        }

        PetscKernel_w_gets_Ar_times_v(bs,bs,t+i2,idiag,y);
        for (j=0; j<bs; j++) x[i2+j] = omega * y[j];

        idiag += bs2;
        i2    += bs;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(1.0*bs2*a->nz);CHKERRQ(ierr);
      xb = t;
    } else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      idiag = kaij->ibdiag+bs2*(m-1);
      i2    = bs * (m-1);
      ierr  = PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
      PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);
      i2    -= bs;
      idiag -= bs2;
      for (i=m-2; i>=0; i--) {
        v  = aa + diag[i] + 1 ;
        vi = aj + diag[i] + 1;
        nz = ai[i+1] - diag[i] - 1;

        if (T) {                /* FIXME: This branch untested */
          ierr = PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
            workt += bs;
          }
          arrt = arr;
          for (j=0; j<nz; j++) {
            ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          ierr = PetscMemcpy(w,t+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) w[k] -= v[j] * x[vi[j]*bs+k];
          }
        }

        PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y); /* RHS incorrect for omega != 1.0 */
        for (j=0; j<bs; j++) x[i2+j] = (1.0-omega) * x[i2+j] + omega * y[j];

        idiag -= bs2;
        i2    -= bs;
      }
      ierr = PetscLogFlops(1.0*bs2*(a->nz));CHKERRQ(ierr);
    }
    its--;
  }
  while (its--) {               /* FIXME: This branch not updated */
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      i2     =  0;
      idiag  = kaij->ibdiag;
      for (i=0; i<m; i++) {
        ierr = PetscMemcpy(w,b+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);

        v  = aa + ai[i];
        vi = aj + ai[i];
        nz = diag[i] - ai[i];
        workt = work;
        for (j=0; j<nz; j++) {
          ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
          workt += bs;
        }
        arrt = arr;
        if (T) {
          for (j=0; j<nz; j++) {
            ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          for (j=0; j<nz; j++) {
            ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        }
        ierr = PetscMemcpy(t+i2,w,bs*sizeof(PetscScalar));CHKERRQ(ierr);

        v  = aa + diag[i] + 1;
        vi = aj + diag[i] + 1;
        nz = ai[i+1] - diag[i] - 1;
        workt = work;
        for (j=0; j<nz; j++) {
          ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
          workt += bs;
        }
        arrt = arr;
        if (T) {
          for (j=0; j<nz; j++) {
            ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          for (j=0; j<nz; j++) {
            ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        }

        PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y);
        for (j=0; j<bs; j++) *(x+i2+j) = (1.0-omega) * *(x+i2+j) + omega * *(y+j);

        idiag += bs2;
        i2    += bs;
      }
      xb = t;
    }
    else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      idiag = kaij->ibdiag+bs2*(m-1);
      i2    = bs * (m-1);
      if (xb == b) {
        for (i=m-1; i>=0; i--) {
          ierr = PetscMemcpy(w,b+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);

          v  = aa + ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          }

          v  = aa + diag[i] + 1;
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          }

          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y);
          for (j=0; j<bs; j++) *(x+i2+j) = (1.0-omega) * *(x+i2+j) + omega * *(y+j);
        }
      } else {
        for (i=m-1; i>=0; i--) {
          ierr = PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
          v  = aa + diag[i] + 1;
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          }
          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y);
          for (j=0; j<bs; j++) *(x+i2+j) = (1.0-omega) * *(x+i2+j) + omega * *(y+j);
        }
      }
      ierr = PetscLogFlops(1.0*bs2*(a->nz));CHKERRQ(ierr);
    }
  }

  ierr = VecRestoreArray(xx,&x);    CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*===================================================================================*/

PetscErrorCode MatMultAdd_MPIKAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIKAIJ    *b = (Mat_MPIKAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!yy) {
    ierr = VecSet(zz,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  /* start the scatter */
  ierr = VecScatterBegin(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multadd)(b->AIJ,xx,zz,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->OAIJ->ops->multadd)(b->OAIJ,b->w,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIKAIJ(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMultAdd_MPIKAIJ(A,xx,PETSC_NULL,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatInvertBlockDiagonal_MPIKAIJ(Mat A,const PetscScalar **values)
{
  Mat_MPIKAIJ     *b = (Mat_MPIKAIJ*)A->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = (*b->AIJ->ops->invertblockdiagonal)(b->AIJ,values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/

PetscErrorCode MatGetRow_SeqKAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **values)
{
  Mat_SeqKAIJ     *b   = (Mat_SeqKAIJ*) A->data;
  PetscErrorCode  diag = PETSC_FALSE;
  PetscErrorCode  ierr;
  PetscInt        nzaij,nz,*colsaij,*idx,i,j,p=b->p,q=b->q,r=row/p,s=row%p,c;
  PetscScalar     *vaij,*v,*S=b->S,*T=b->T;

  PetscFunctionBegin;
  if (b->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  b->getrowactive = PETSC_TRUE;
  if (row < 0 || row >= A->rmap->n) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D out of range",row);

  if ((!S) && (!T) && (!b->isTI)) {
    if (ncols)    *ncols  = 0;
    if (cols)     *cols   = NULL;
    if (values)   *values = NULL;
    PetscFunctionReturn(0);
  }

  if (T || b->isTI) {
    ierr  = MatGetRow_SeqAIJ(b->AIJ,r,&nzaij,&colsaij,&vaij);CHKERRQ(ierr);
    c     = nzaij;
    for (i=0; i<nzaij; i++) {
      /* check if this row contains a diagonal entry */
      if (colsaij[i] == r) {
        diag = PETSC_TRUE;
        c = i;
      }
    }
  } else nzaij = c = 0;

  /* calculate size of row */
  nz = 0;
  if (S)            nz += q;
  if (T || b->isTI) nz += (diag && S ? (nzaij-1)*q : nzaij*q);

  if (cols || values) {
    ierr = PetscMalloc2(nz,&idx,nz,&v);CHKERRQ(ierr);
    for (i=0; i<q; i++) {
      /* We need to initialize the v[i] to zero to handle the case in which T is NULL (not the identity matrix). */
      v[i] = 0.0;
    }
    if (b->isTI) {
      for (i=0; i<nzaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = colsaij[i]*q+j;
          v[i*q+j]   = (j==s ? vaij[i] : 0);
        }
      }
    } else if (T) {
      for (i=0; i<nzaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = colsaij[i]*q+j;
          v[i*q+j]   = vaij[i]*T[s+j*p];
        }
      }
    }
    if (S) {
      for (j=0; j<q; j++) {
        idx[c*q+j] = r*q+j;
        v[c*q+j]  += S[s+j*p];
      }
    }
  }

  if (ncols)    *ncols  = nz;
  if (cols)     *cols   = idx;
  if (values)   *values = v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_SeqKAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree2(*idx,*v);CHKERRQ(ierr);
  ((Mat_SeqKAIJ*)A->data)->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_MPIKAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **values)
{
  Mat_MPIKAIJ     *b      = (Mat_MPIKAIJ*) A->data;
  Mat             MatAIJ  = ((Mat_SeqKAIJ*)b->AIJ->data)->AIJ;
  Mat             MatOAIJ = ((Mat_SeqKAIJ*)b->OAIJ->data)->AIJ;
  Mat             AIJ     = b->A;
  PetscBool       diag    = PETSC_FALSE;
  PetscErrorCode  ierr;
  const PetscInt  rstart=A->rmap->rstart,rend=A->rmap->rend,p=b->p,q=b->q,*garray;
  PetscInt        nz,*idx,ncolsaij,ncolsoaij,*colsaij,*colsoaij,r,s,c,i,j,lrow;
  PetscScalar     *v,*vals,*ovals,*S=b->S,*T=b->T;

  PetscFunctionBegin;
  if (b->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  b->getrowactive = PETSC_TRUE;
  if (row < rstart || row >= rend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only local rows");
  lrow = row - rstart;

  if ((!S) && (!T) && (!b->isTI)) {
    if (ncols)    *ncols  = 0;
    if (cols)     *cols   = NULL;
    if (values)   *values = NULL;
    PetscFunctionReturn(0);
  }

  r = lrow/p;
  s = lrow%p;

  if (T || b->isTI) {
    ierr = MatMPIAIJGetSeqAIJ(AIJ,NULL,NULL,&garray);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(MatAIJ,lrow/p,&ncolsaij,&colsaij,&vals);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(MatOAIJ,lrow/p,&ncolsoaij,&colsoaij,&ovals);CHKERRQ(ierr);

    c     = ncolsaij + ncolsoaij;
    for (i=0; i<ncolsaij; i++) {
      /* check if this row contains a diagonal entry */
      if (colsaij[i] == r) {
        diag = PETSC_TRUE;
        c = i;
      }
    }
  } else c = 0;

  /* calculate size of row */
  nz = 0;
  if (S)            nz += q;
  if (T || b->isTI) nz += (diag && S ? (ncolsaij+ncolsoaij-1)*q : (ncolsaij+ncolsoaij)*q);

  if (cols || values) {
    ierr = PetscMalloc2(nz,&idx,nz,&v);CHKERRQ(ierr);
    for (i=0; i<q; i++) {
      /* We need to initialize the v[i] to zero to handle the case in which T is NULL (not the identity matrix). */
      v[i] = 0.0;
    }
    if (b->isTI) {
      for (i=0; i<ncolsaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = (colsaij[i]+rstart/p)*q+j;
          v[i*q+j]   = (j==s ? vals[i] : 0.0);
        }
      }
      for (i=0; i<ncolsoaij; i++) {
        for (j=0; j<q; j++) {
          idx[(i+ncolsaij)*q+j] = garray[colsoaij[i]]*q+j;
          v[(i+ncolsaij)*q+j]   = (j==s ? ovals[i]: 0.0);
        }
      }
    } else if (T) {
      for (i=0; i<ncolsaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = (colsaij[i]+rstart/p)*q+j;
          v[i*q+j]   = vals[i]*T[s+j*p];
        }
      }
      for (i=0; i<ncolsoaij; i++) {
        for (j=0; j<q; j++) {
          idx[(i+ncolsaij)*q+j] = garray[colsoaij[i]]*q+j;
          v[(i+ncolsaij)*q+j]   = ovals[i]*T[s+j*p];
        }
      }
    }
    if (S) {
      for (j=0; j<q; j++) {
        idx[c*q+j] = (r+rstart/p)*q+j;
        v[c*q+j]  += S[s+j*p];
      }
    }
  }

  if (ncols)  *ncols  = nz;
  if (cols)   *cols   = idx;
  if (values) *values = v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_MPIKAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree2(*idx,*v);CHKERRQ(ierr);
  ((Mat_SeqKAIJ*)A->data)->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode  MatCreateSubMatrix_KAIJ(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBegin;
  ierr = MatConvert(mat,MATAIJ,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(A,isrow,iscol,cll,newmat);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------- */
/*@C
  MatCreateKAIJ - Creates a matrix type to be used for matrices of the following form:

    [I \otimes S + A \otimes T]

  where
    S is a dense (p \times q) matrix
    T is a dense (p \times q) matrix
    A is an AIJ  (n \times n) matrix
    I is the identity matrix
  The resulting matrix is (np \times nq)

  S and T are always stored independently on all processes as PetscScalar arrays in column-major format.

  Collective

  Input Parameters:
+ A - the AIJ matrix
. p - number of rows in S and T
. q - number of columns in S and T
. S - the S matrix (can be PETSC_NULL), stored as a PetscScalar array (column-major)
- T - the T matrix (can be PETSC_NULL), stored as a PetscScalar array (column-major)

  Output Parameter:
. kaij - the new KAIJ matrix

  Notes:
  This function increases the reference count on the AIJ matrix, so the user is free to destroy the matrix if it is not needed.
  Changes to the entries of the AIJ matrix will immediately affect the KAIJ matrix.

  Level: advanced

.seealso: MatKAIJSetAIJ(), MatKAIJSetS(), MatKAIJSetT(), MatKAIJGetAIJ(), MatKAIJGetS(), MatKAIJGetT(), MATKAIJ
@*/
PetscErrorCode  MatCreateKAIJ(Mat A,PetscInt p,PetscInt q,const PetscScalar S[],const PetscScalar T[],Mat *kaij)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),kaij);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(*kaij,MATSEQKAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*kaij,MATMPIKAIJ);CHKERRQ(ierr);
  }
  ierr = MatKAIJSetAIJ(*kaij,A);CHKERRQ(ierr);
  ierr = MatKAIJSetS(*kaij,p,q,S);CHKERRQ(ierr);
  ierr = MatKAIJSetT(*kaij,p,q,T);CHKERRQ(ierr);
  ierr = MatSetUp(*kaij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATKAIJ - MATKAIJ = "kaij" - A matrix type to be used to evaluate matrices of form
    [I \otimes S + A \otimes T],
  where
    S is a dense (p \times q) matrix,
    T is a dense (p \times q) matrix,
    A is an AIJ  (n \times n) matrix,
    and I is the identity matrix.
  The resulting matrix is (np \times nq).

  S and T are always stored independently on all processes as PetscScalar arrays in column-major format.

  Notes:
  A linear system with multiple right-hand sides, AX = B, can be expressed in the KAIJ-friendly form of (A \otimes I) x = b,
  where x and b are column vectors containing the row-major representations of X and B.

  Level: advanced

.seealso: MatKAIJSetAIJ(), MatKAIJSetS(), MatKAIJSetT(), MatKAIJGetAIJ(), MatKAIJGetS(), MatKAIJGetT(), MatCreateKAIJ()
M*/

PETSC_EXTERN PetscErrorCode MatCreate_KAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIKAIJ    *b;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr     = PetscNewLog(A,&b);CHKERRQ(ierr);
  A->data  = (void*)b;

  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->setup = MatSetUp_KAIJ;

  b->w    = 0;
  ierr    = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQKAIJ);CHKERRQ(ierr);
    A->ops->setup               = MatSetUp_KAIJ;
    A->ops->destroy             = MatDestroy_SeqKAIJ;
    A->ops->view                = MatView_KAIJ;
    A->ops->mult                = MatMult_SeqKAIJ;
    A->ops->multadd             = MatMultAdd_SeqKAIJ;
    A->ops->invertblockdiagonal = MatInvertBlockDiagonal_SeqKAIJ;
    A->ops->getrow              = MatGetRow_SeqKAIJ;
    A->ops->restorerow          = MatRestoreRow_SeqKAIJ;
    A->ops->sor                 = MatSOR_SeqKAIJ;
  } else {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIKAIJ);CHKERRQ(ierr);
    A->ops->setup               = MatSetUp_KAIJ;
    A->ops->destroy             = MatDestroy_MPIKAIJ;
    A->ops->view                = MatView_KAIJ;
    A->ops->mult                = MatMult_MPIKAIJ;
    A->ops->multadd             = MatMultAdd_MPIKAIJ;
    A->ops->invertblockdiagonal = MatInvertBlockDiagonal_MPIKAIJ;
    A->ops->getrow              = MatGetRow_MPIKAIJ;
    A->ops->restorerow          = MatRestoreRow_MPIKAIJ;
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatGetDiagonalBlock_C",MatGetDiagonalBlock_MPIKAIJ);CHKERRQ(ierr);
  }
  A->ops->createsubmatrix = MatCreateSubMatrix_KAIJ;
  PetscFunctionReturn(0);
}
