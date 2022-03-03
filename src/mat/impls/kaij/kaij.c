
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
  PetscBool      ismpikaij,isseqkaij;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPIKAIJ,&ismpikaij));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQKAIJ,&isseqkaij));
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
  PetscFunctionBegin;
  if (S) *S = NULL;
  CHKERRQ(PetscObjectStateIncrease((PetscObject)A));
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

   Output Parameters:
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

   Output Parameters:
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
  PetscFunctionBegin;
  if (T) *T = NULL;
  CHKERRQ(PetscObjectStateIncrease((PetscObject)A));
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
  PetscMPIInt    size;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)B,MATSEQAIJ,&flg));
    PetscCheck(flg,PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"MatKAIJSetAIJ() with MATSEQKAIJ does not support %s as the AIJ mat",((PetscObject)B)->type_name);
    Mat_SeqKAIJ *a = (Mat_SeqKAIJ*)A->data;
    a->AIJ = B;
  } else {
    Mat_MPIKAIJ *a = (Mat_MPIKAIJ*)A->data;
    a->A = B;
  }
  CHKERRQ(PetscObjectReference((PetscObject)B));
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
  Mat_SeqKAIJ    *a = (Mat_SeqKAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(a->S));
  if (S) {
    CHKERRQ(PetscMalloc1(p*q,&a->S));
    CHKERRQ(PetscMemcpy(a->S,S,p*q*sizeof(PetscScalar)));
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
          if (a->S && PetscAbsScalar(a->S[i+j*a->p]) > PETSC_SMALL) *identity = PETSC_FALSE;
          if (a->T && PetscAbsScalar(a->T[i+j*a->p]) > PETSC_SMALL) *identity = PETSC_FALSE;
        } else {
          if (a->S && PetscAbsScalar(a->S[i*(a->p+1)]-a->S[0]) > PETSC_SMALL) *identity = PETSC_FALSE;
          if (a->T && PetscAbsScalar(a->T[i*(a->p+1)]-a->T[0]) > PETSC_SMALL) *identity = PETSC_FALSE;
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

  CHKERRQ(PetscFree(a->T));
  if (T && (!isTI)) {
    CHKERRQ(PetscMalloc1(p*q,&a->T));
    CHKERRQ(PetscMemcpy(a->T,T,p*q*sizeof(PetscScalar)));
  } else a->T = NULL;

  a->p = p;
  a->q = q;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqKAIJ(Mat A)
{
  Mat_SeqKAIJ    *b = (Mat_SeqKAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&b->AIJ));
  CHKERRQ(PetscFree(b->S));
  CHKERRQ(PetscFree(b->T));
  CHKERRQ(PetscFree(b->ibdiag));
  CHKERRQ(PetscFree5(b->sor.w,b->sor.y,b->sor.work,b->sor.t,b->sor.arr));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqkaij_seqaij_C",NULL));
  CHKERRQ(PetscFree(A->data));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatKAIJ_build_AIJ_OAIJ(Mat A)
{
  Mat_MPIKAIJ      *a;
  Mat_MPIAIJ       *mpiaij;
  PetscScalar      *T;
  PetscInt         i,j;
  PetscObjectState state;

  PetscFunctionBegin;
  a = (Mat_MPIKAIJ*)A->data;
  mpiaij = (Mat_MPIAIJ*)a->A->data;

  CHKERRQ(PetscObjectStateGet((PetscObject)a->A,&state));
  if (state == a->state) {
    /* The existing AIJ and KAIJ members are up-to-date, so simply exit. */
    PetscFunctionReturn(0);
  } else {
    CHKERRQ(MatDestroy(&a->AIJ));
    CHKERRQ(MatDestroy(&a->OAIJ));
    if (a->isTI) {
      /* If the transformation matrix associated with the parallel matrix A is the identity matrix, then a->T will be NULL.
       * In this case, if we pass a->T directly to the MatCreateKAIJ() calls to create the sequential submatrices, the routine will
       * not be able to tell that transformation matrix should be set to the identity; thus we create a temporary identity matrix
       * to pass in. */
      CHKERRQ(PetscMalloc1(a->p*a->q,&T));
      for (i=0; i<a->p; i++) {
        for (j=0; j<a->q; j++) {
          if (i==j) T[i+j*a->p] = 1.0;
          else      T[i+j*a->p] = 0.0;
        }
      }
    } else T = a->T;
    CHKERRQ(MatCreateKAIJ(mpiaij->A,a->p,a->q,a->S,T,&a->AIJ));
    CHKERRQ(MatCreateKAIJ(mpiaij->B,a->p,a->q,NULL,T,&a->OAIJ));
    if (a->isTI) {
      CHKERRQ(PetscFree(T));
    }
    a->state = state;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_KAIJ(Mat A)
{
  PetscInt       n;
  PetscMPIInt    size;
  Mat_SeqKAIJ    *seqkaij = (Mat_SeqKAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    CHKERRQ(MatSetSizes(A,seqkaij->p*seqkaij->AIJ->rmap->n,seqkaij->q*seqkaij->AIJ->cmap->n,seqkaij->p*seqkaij->AIJ->rmap->N,seqkaij->q*seqkaij->AIJ->cmap->N));
    CHKERRQ(PetscLayoutSetBlockSize(A->rmap,seqkaij->p));
    CHKERRQ(PetscLayoutSetBlockSize(A->cmap,seqkaij->q));
    CHKERRQ(PetscLayoutSetUp(A->rmap));
    CHKERRQ(PetscLayoutSetUp(A->cmap));
  } else {
    Mat_MPIKAIJ *a;
    Mat_MPIAIJ  *mpiaij;
    IS          from,to;
    Vec         gvec;

    a = (Mat_MPIKAIJ*)A->data;
    mpiaij = (Mat_MPIAIJ*)a->A->data;
    CHKERRQ(MatSetSizes(A,a->p*a->A->rmap->n,a->q*a->A->cmap->n,a->p*a->A->rmap->N,a->q*a->A->cmap->N));
    CHKERRQ(PetscLayoutSetBlockSize(A->rmap,seqkaij->p));
    CHKERRQ(PetscLayoutSetBlockSize(A->cmap,seqkaij->q));
    CHKERRQ(PetscLayoutSetUp(A->rmap));
    CHKERRQ(PetscLayoutSetUp(A->cmap));

    CHKERRQ(MatKAIJ_build_AIJ_OAIJ(A));

    CHKERRQ(VecGetSize(mpiaij->lvec,&n));
    CHKERRQ(VecCreate(PETSC_COMM_SELF,&a->w));
    CHKERRQ(VecSetSizes(a->w,n*a->q,n*a->q));
    CHKERRQ(VecSetBlockSize(a->w,a->q));
    CHKERRQ(VecSetType(a->w,VECSEQ));

    /* create two temporary Index sets for build scatter gather */
    CHKERRQ(ISCreateBlock(PetscObjectComm((PetscObject)a->A),a->q,n,mpiaij->garray,PETSC_COPY_VALUES,&from));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n*a->q,0,1,&to));

    /* create temporary global vector to generate scatter context */
    CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)a->A),a->q,a->q*a->A->cmap->n,a->q*a->A->cmap->N,NULL,&gvec));

    /* generate the scatter context */
    CHKERRQ(VecScatterCreate(gvec,from,a->w,to,&a->ctx));

    CHKERRQ(ISDestroy(&from));
    CHKERRQ(ISDestroy(&to));
    CHKERRQ(VecDestroy(&gvec));
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
  PetscBool         ismpikaij;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPIKAIJ,&ismpikaij));
  CHKERRQ(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL || format == PETSC_VIEWER_ASCII_IMPL) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"S and T have %" PetscInt_FMT " rows and %" PetscInt_FMT " columns\n",a->p,a->q));

    /* Print appropriate details for S. */
    if (!a->S) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"S is NULL\n"));
    } else if (format == PETSC_VIEWER_ASCII_IMPL) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"Entries of S are "));
      for (i=0; i<(a->p * a->q); i++) {
#if defined(PETSC_USE_COMPLEX)
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e ",(double)PetscRealPart(a->S[i]),(double)PetscImaginaryPart(a->S[i])));
#else
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)PetscRealPart(a->S[i])));
#endif
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
    }

    /* Print appropriate details for T. */
    if (a->isTI) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"T is the identity matrix\n"));
    } else if (!a->T) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"T is NULL\n"));
    } else if (format == PETSC_VIEWER_ASCII_IMPL) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"Entries of T are "));
      for (i=0; i<(a->p * a->q); i++) {
#if defined(PETSC_USE_COMPLEX)
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e %18.16e ",(double)PetscRealPart(a->T[i]),(double)PetscImaginaryPart(a->T[i])));
#else
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)PetscRealPart(a->T[i])));
#endif
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
    }

    /* Now print details for the AIJ matrix, using the AIJ viewer. */
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Now viewing the associated AIJ matrix:\n"));
    if (ismpikaij) {
      Mat_MPIKAIJ *b = (Mat_MPIKAIJ*)A->data;
      CHKERRQ(MatView(b->A,viewer));
    } else {
      CHKERRQ(MatView(a->AIJ,viewer));
    }

  } else {
    /* For all other matrix viewer output formats, simply convert to an AIJ matrix and call MatView() on that. */
    CHKERRQ(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B));
    CHKERRQ(MatView(B,viewer));
    CHKERRQ(MatDestroy(&B));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIKAIJ(Mat A)
{
  Mat_MPIKAIJ    *b = (Mat_MPIKAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&b->AIJ));
  CHKERRQ(MatDestroy(&b->OAIJ));
  CHKERRQ(MatDestroy(&b->A));
  CHKERRQ(VecScatterDestroy(&b->ctx));
  CHKERRQ(VecDestroy(&b->w));
  CHKERRQ(PetscFree(b->S));
  CHKERRQ(PetscFree(b->T));
  CHKERRQ(PetscFree(b->ibdiag));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatGetDiagonalBlock_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatConvert_mpikaij_mpiaij_C",NULL));
  CHKERRQ(PetscFree(A->data));
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
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j,l,p=b->p,q=b->q,k;

  PetscFunctionBegin;
  if (!yy) {
    CHKERRQ(VecSet(zz,0.0));
  } else {
    CHKERRQ(VecCopy(yy,zz));
  }
  if ((!s) && (!t) && (!b->isTI)) PetscFunctionReturn(0);

  CHKERRQ(VecGetArrayRead(xx,&x));
  CHKERRQ(VecGetArray(zz,&y));
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
    CHKERRQ(PetscLogFlops(3.0*(a->nz)*p));
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
    CHKERRQ(PetscLogFlops((2.0*p*q-p)*m+2.0*p*a->nz));
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
    CHKERRQ(PetscLogFlops(2.0*m*p*q));
  }

  CHKERRQ(VecRestoreArrayRead(xx,&x));
  CHKERRQ(VecRestoreArray(zz,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqKAIJ(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultAdd_SeqKAIJ(A,xx,PETSC_NULL,yy));
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
  PetscInt          i,j,*v_pivots,dof,dof2;
  PetscScalar       *diag,aval,*v_work;

  PetscFunctionBegin;
  PetscCheckFalse(p != q,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATKAIJ: Block size must be square to calculate inverse.");
  PetscCheckFalse((!S) && (!T) && (!b->isTI),PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATKAIJ: Cannot invert a zero matrix.");

  dof  = p;
  dof2 = dof*dof;

  if (b->ibdiagvalid) {
    if (values) *values = b->ibdiag;
    PetscFunctionReturn(0);
  }
  if (!b->ibdiag) {
    CHKERRQ(PetscMalloc1(dof2*m,&b->ibdiag));
    CHKERRQ(PetscLogObjectMemory((PetscObject)A,dof2*m*sizeof(PetscScalar)));
  }
  if (values) *values = b->ibdiag;
  diag = b->ibdiag;

  CHKERRQ(PetscMalloc2(dof,&v_work,dof,&v_pivots));
  for (i=0; i<m; i++) {
    if (S) {
      CHKERRQ(PetscMemcpy(diag,S,dof2*sizeof(PetscScalar)));
    } else {
      CHKERRQ(PetscMemzero(diag,dof2*sizeof(PetscScalar)));
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
    CHKERRQ(PetscKernel_A_gets_inverse_A(dof,diag,v_pivots,v_work,PETSC_FALSE,NULL));
    diag += dof2;
  }
  CHKERRQ(PetscFree2(v_work,v_pivots));

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

static PetscErrorCode MatConvert_KAIJ_AIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_SeqKAIJ    *a = (Mat_SeqKAIJ*)A->data;
  Mat            AIJ,OAIJ,B;
  PetscInt       *d_nnz,*o_nnz = NULL,nz,i,j,m,d;
  const PetscInt p = a->p,q = a->q;
  PetscBool      ismpikaij,missing;

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPIKAIJ,&ismpikaij));
    if (ismpikaij) {
      Mat_MPIKAIJ *b = (Mat_MPIKAIJ*)A->data;
      AIJ = ((Mat_SeqKAIJ*)b->AIJ->data)->AIJ;
      OAIJ = ((Mat_SeqKAIJ*)b->OAIJ->data)->AIJ;
    } else {
      AIJ = a->AIJ;
      OAIJ = NULL;
    }
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),&B));
    CHKERRQ(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
    CHKERRQ(MatSetType(B,MATAIJ));
    CHKERRQ(MatGetSize(AIJ,&m,NULL));
    CHKERRQ(MatMissingDiagonal(AIJ,&missing,&d)); /* assumption that all successive rows will have a missing diagonal */
    if (!missing || !a->S) d = m;
    CHKERRQ(PetscMalloc1(m*p,&d_nnz));
    for (i = 0; i < m; ++i) {
      CHKERRQ(MatGetRow_SeqAIJ(AIJ,i,&nz,NULL,NULL));
      for (j = 0; j < p; ++j) d_nnz[i*p + j] = nz*q + (i >= d)*q;
      CHKERRQ(MatRestoreRow_SeqAIJ(AIJ,i,&nz,NULL,NULL));
    }
    if (OAIJ) {
      CHKERRQ(PetscMalloc1(m*p,&o_nnz));
      for (i = 0; i < m; ++i) {
        CHKERRQ(MatGetRow_SeqAIJ(OAIJ,i,&nz,NULL,NULL));
        for (j = 0; j < p; ++j) o_nnz[i*p + j] = nz*q;
        CHKERRQ(MatRestoreRow_SeqAIJ(OAIJ,i,&nz,NULL,NULL));
      }
      CHKERRQ(MatMPIAIJSetPreallocation(B,0,d_nnz,0,o_nnz));
    } else {
      CHKERRQ(MatSeqAIJSetPreallocation(B,0,d_nnz));
    }
    CHKERRQ(PetscFree(d_nnz));
    CHKERRQ(PetscFree(o_nnz));
  } else B = *newmat;
  CHKERRQ(MatConvert_Basic(A,newtype,MAT_REUSE_MATRIX,&B));
  if (reuse == MAT_INPLACE_MATRIX) {
    CHKERRQ(MatHeaderReplace(A,&B));
  } else *newmat = B;
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
  PetscCheckFalse(flag & SOR_EISENSTAT,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  PetscCheckFalse(its <= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive",its,lits);
  PetscCheck(!fshift,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for diagonal shift");
  PetscCheckFalse((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER),PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for applying upper or lower triangular parts");
  PetscCheckFalse(p != q,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatSOR for KAIJ: No support for non-square dense blocks");
  else        {bs = p; bs2 = bs*bs; }

  if (!m) PetscFunctionReturn(0);

  if (!kaij->ibdiagvalid) CHKERRQ(MatInvertBlockDiagonal_SeqKAIJ(A,NULL));
  idiag = kaij->ibdiag;
  diag  = a->diag;

  if (!kaij->sor.setup) {
    CHKERRQ(PetscMalloc5(bs,&kaij->sor.w,bs,&kaij->sor.y,m*bs,&kaij->sor.work,m*bs,&kaij->sor.t,m*bs2,&kaij->sor.arr));
    kaij->sor.setup = PETSC_TRUE;
  }
  y     = kaij->sor.y;
  w     = kaij->sor.w;
  work  = kaij->sor.work;
  t     = kaij->sor.t;
  arr   = kaij->sor.arr;

  ierr = VecGetArray(xx,&x);    CHKERRQ(ierr);
  CHKERRQ(VecGetArrayRead(bb,&b));

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      PetscKernel_w_gets_Ar_times_v(bs,bs,b,idiag,x);                            /* x[0:bs] <- D^{-1} b[0:bs] */
      CHKERRQ(PetscMemcpy(t,b,bs*sizeof(PetscScalar)));
      i2     =  bs;
      idiag  += bs2;
      for (i=1; i<m; i++) {
        v  = aa + ai[i];
        vi = aj + ai[i];
        nz = diag[i] - ai[i];

        if (T) {                /* b - T (Arow * x) */
          CHKERRQ(PetscMemzero(w,bs*sizeof(PetscScalar)));
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) w[k] -= v[j] * x[vi[j]*bs+k];
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs,w,T,&t[i2]);
          for (k=0; k<bs; k++) t[i2+k] += b[i2+k];
        } else if (kaij->isTI) {
          CHKERRQ(PetscMemcpy(t+i2,b+i2,bs*sizeof(PetscScalar)));
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) t[i2+k] -= v[j] * x[vi[j]*bs+k];
          }
        } else {
          CHKERRQ(PetscMemcpy(t+i2,b+i2,bs*sizeof(PetscScalar)));
        }

        PetscKernel_w_gets_Ar_times_v(bs,bs,t+i2,idiag,y);
        for (j=0; j<bs; j++) x[i2+j] = omega * y[j];

        idiag += bs2;
        i2    += bs;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      CHKERRQ(PetscLogFlops(1.0*bs2*a->nz));
      xb = t;
    } else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      idiag = kaij->ibdiag+bs2*(m-1);
      i2    = bs * (m-1);
      CHKERRQ(PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar)));
      PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);
      i2    -= bs;
      idiag -= bs2;
      for (i=m-2; i>=0; i--) {
        v  = aa + diag[i] + 1 ;
        vi = aj + diag[i] + 1;
        nz = ai[i+1] - diag[i] - 1;

        if (T) {                /* FIXME: This branch untested */
          CHKERRQ(PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar)));
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            CHKERRQ(PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar)));
            workt += bs;
          }
          arrt = arr;
          for (j=0; j<nz; j++) {
            CHKERRQ(PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar)));
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          CHKERRQ(PetscMemcpy(w,t+i2,bs*sizeof(PetscScalar)));
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) w[k] -= v[j] * x[vi[j]*bs+k];
          }
        }

        PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y); /* RHS incorrect for omega != 1.0 */
        for (j=0; j<bs; j++) x[i2+j] = (1.0-omega) * x[i2+j] + omega * y[j];

        idiag -= bs2;
        i2    -= bs;
      }
      CHKERRQ(PetscLogFlops(1.0*bs2*(a->nz)));
    }
    its--;
  }
  while (its--) {               /* FIXME: This branch not updated */
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      i2     =  0;
      idiag  = kaij->ibdiag;
      for (i=0; i<m; i++) {
        CHKERRQ(PetscMemcpy(w,b+i2,bs*sizeof(PetscScalar)));

        v  = aa + ai[i];
        vi = aj + ai[i];
        nz = diag[i] - ai[i];
        workt = work;
        for (j=0; j<nz; j++) {
          CHKERRQ(PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar)));
          workt += bs;
        }
        arrt = arr;
        if (T) {
          for (j=0; j<nz; j++) {
            CHKERRQ(PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar)));
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          for (j=0; j<nz; j++) {
            CHKERRQ(PetscMemzero(arrt,bs2*sizeof(PetscScalar)));
            for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        }
        CHKERRQ(PetscMemcpy(t+i2,w,bs*sizeof(PetscScalar)));

        v  = aa + diag[i] + 1;
        vi = aj + diag[i] + 1;
        nz = ai[i+1] - diag[i] - 1;
        workt = work;
        for (j=0; j<nz; j++) {
          CHKERRQ(PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar)));
          workt += bs;
        }
        arrt = arr;
        if (T) {
          for (j=0; j<nz; j++) {
            CHKERRQ(PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar)));
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          for (j=0; j<nz; j++) {
            CHKERRQ(PetscMemzero(arrt,bs2*sizeof(PetscScalar)));
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
          CHKERRQ(PetscMemcpy(w,b+i2,bs*sizeof(PetscScalar)));

          v  = aa + ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          workt = work;
          for (j=0; j<nz; j++) {
            CHKERRQ(PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar)));
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              CHKERRQ(PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar)));
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              CHKERRQ(PetscMemzero(arrt,bs2*sizeof(PetscScalar)));
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
            CHKERRQ(PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar)));
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              CHKERRQ(PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar)));
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              CHKERRQ(PetscMemzero(arrt,bs2*sizeof(PetscScalar)));
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
          CHKERRQ(PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar)));
          v  = aa + diag[i] + 1;
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          workt = work;
          for (j=0; j<nz; j++) {
            CHKERRQ(PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar)));
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              CHKERRQ(PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar)));
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              CHKERRQ(PetscMemzero(arrt,bs2*sizeof(PetscScalar)));
              for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          }
          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y);
          for (j=0; j<bs; j++) *(x+i2+j) = (1.0-omega) * *(x+i2+j) + omega * *(y+j);
        }
      }
      CHKERRQ(PetscLogFlops(1.0*bs2*(a->nz)));
    }
  }

  ierr = VecRestoreArray(xx,&x);    CHKERRQ(ierr);
  CHKERRQ(VecRestoreArrayRead(bb,&b));
  PetscFunctionReturn(0);
}

/*===================================================================================*/

PetscErrorCode MatMultAdd_MPIKAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIKAIJ    *b = (Mat_MPIKAIJ*)A->data;

  PetscFunctionBegin;
  if (!yy) {
    CHKERRQ(VecSet(zz,0.0));
  } else {
    CHKERRQ(VecCopy(yy,zz));
  }
  CHKERRQ(MatKAIJ_build_AIJ_OAIJ(A)); /* Ensure b->AIJ and b->OAIJ are up to date. */
  /* start the scatter */
  CHKERRQ(VecScatterBegin(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ((*b->AIJ->ops->multadd)(b->AIJ,xx,zz,zz));
  CHKERRQ(VecScatterEnd(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ((*b->OAIJ->ops->multadd)(b->OAIJ,b->w,zz,zz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIKAIJ(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  CHKERRQ(MatMultAdd_MPIKAIJ(A,xx,PETSC_NULL,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatInvertBlockDiagonal_MPIKAIJ(Mat A,const PetscScalar **values)
{
  Mat_MPIKAIJ     *b = (Mat_MPIKAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRQ(MatKAIJ_build_AIJ_OAIJ(A)); /* Ensure b->AIJ is up to date. */
  CHKERRQ((*b->AIJ->ops->invertblockdiagonal)(b->AIJ,values));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/

PetscErrorCode MatGetRow_SeqKAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **values)
{
  Mat_SeqKAIJ     *b   = (Mat_SeqKAIJ*) A->data;
  PetscErrorCode  diag = PETSC_FALSE;
  PetscInt        nzaij,nz,*colsaij,*idx,i,j,p=b->p,q=b->q,r=row/p,s=row%p,c;
  PetscScalar     *vaij,*v,*S=b->S,*T=b->T;

  PetscFunctionBegin;
  PetscCheck(!b->getrowactive,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  b->getrowactive = PETSC_TRUE;
  PetscCheckFalse(row < 0 || row >= A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " out of range",row);

  if ((!S) && (!T) && (!b->isTI)) {
    if (ncols)    *ncols  = 0;
    if (cols)     *cols   = NULL;
    if (values)   *values = NULL;
    PetscFunctionReturn(0);
  }

  if (T || b->isTI) {
    CHKERRQ(MatGetRow_SeqAIJ(b->AIJ,r,&nzaij,&colsaij,&vaij));
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
    CHKERRQ(PetscMalloc2(nz,&idx,nz,&v));
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
  PetscFunctionBegin;
  if (nz) *nz = 0;
  CHKERRQ(PetscFree2(*idx,*v));
  ((Mat_SeqKAIJ*)A->data)->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_MPIKAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **values)
{
  Mat_MPIKAIJ     *b      = (Mat_MPIKAIJ*) A->data;
  Mat             AIJ     = b->A;
  PetscBool       diag    = PETSC_FALSE;
  Mat             MatAIJ,MatOAIJ;
  const PetscInt  rstart=A->rmap->rstart,rend=A->rmap->rend,p=b->p,q=b->q,*garray;
  PetscInt        nz,*idx,ncolsaij = 0,ncolsoaij = 0,*colsaij,*colsoaij,r,s,c,i,j,lrow;
  PetscScalar     *v,*vals,*ovals,*S=b->S,*T=b->T;

  PetscFunctionBegin;
  CHKERRQ(MatKAIJ_build_AIJ_OAIJ(A)); /* Ensure b->AIJ and b->OAIJ are up to date. */
  MatAIJ  = ((Mat_SeqKAIJ*)b->AIJ->data)->AIJ;
  MatOAIJ = ((Mat_SeqKAIJ*)b->OAIJ->data)->AIJ;
  PetscCheck(!b->getrowactive,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  b->getrowactive = PETSC_TRUE;
  PetscCheckFalse(row < rstart || row >= rend,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only local rows");
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
    CHKERRQ(MatMPIAIJGetSeqAIJ(AIJ,NULL,NULL,&garray));
    CHKERRQ(MatGetRow_SeqAIJ(MatAIJ,lrow/p,&ncolsaij,&colsaij,&vals));
    CHKERRQ(MatGetRow_SeqAIJ(MatOAIJ,lrow/p,&ncolsoaij,&colsoaij,&ovals));

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
    CHKERRQ(PetscMalloc2(nz,&idx,nz,&v));
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
  PetscFunctionBegin;
  CHKERRQ(PetscFree2(*idx,*v));
  ((Mat_SeqKAIJ*)A->data)->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode  MatCreateSubMatrix_KAIJ(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(MatConvert(mat,MATAIJ,MAT_INITIAL_MATRIX,&A));
  CHKERRQ(MatCreateSubMatrix(A,isrow,iscol,cll,newmat));
  CHKERRQ(MatDestroy(&A));
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

  Developer Notes:
  In the MATMPIKAIJ case, the internal 'AIJ' and 'OAIJ' sequential KAIJ matrices are kept up to date by tracking the object state
  of the AIJ matrix 'A' that describes the blockwise action of the MATMPIKAIJ matrix and, if the object state has changed, lazily
  rebuilding 'AIJ' and 'OAIJ' just before executing operations with the MATMPIKAIJ matrix. If new types of operations are added,
  routines implementing those must also ensure these are rebuilt when needed (by calling the internal MatKAIJ_build_AIJ_OAIJ() routine).

  Level: advanced

.seealso: MatKAIJSetAIJ(), MatKAIJSetS(), MatKAIJSetT(), MatKAIJGetAIJ(), MatKAIJGetS(), MatKAIJGetT(), MATKAIJ
@*/
PetscErrorCode  MatCreateKAIJ(Mat A,PetscInt p,PetscInt q,const PetscScalar S[],const PetscScalar T[],Mat *kaij)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),kaij));
  CHKERRQ(MatSetType(*kaij,MATKAIJ));
  CHKERRQ(MatKAIJSetAIJ(*kaij,A));
  CHKERRQ(MatKAIJSetS(*kaij,p,q,S));
  CHKERRQ(MatKAIJSetT(*kaij,p,q,T));
  CHKERRQ(MatSetUp(*kaij));
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
  Mat_MPIKAIJ    *b;
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(A,&b));
  A->data  = (void*)b;

  CHKERRQ(PetscMemzero(A->ops,sizeof(struct _MatOps)));

  b->w    = NULL;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    CHKERRQ(PetscObjectChangeTypeName((PetscObject)A,MATSEQKAIJ));
    A->ops->destroy             = MatDestroy_SeqKAIJ;
    A->ops->mult                = MatMult_SeqKAIJ;
    A->ops->multadd             = MatMultAdd_SeqKAIJ;
    A->ops->invertblockdiagonal = MatInvertBlockDiagonal_SeqKAIJ;
    A->ops->getrow              = MatGetRow_SeqKAIJ;
    A->ops->restorerow          = MatRestoreRow_SeqKAIJ;
    A->ops->sor                 = MatSOR_SeqKAIJ;
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqkaij_seqaij_C",MatConvert_KAIJ_AIJ));
  } else {
    CHKERRQ(PetscObjectChangeTypeName((PetscObject)A,MATMPIKAIJ));
    A->ops->destroy             = MatDestroy_MPIKAIJ;
    A->ops->mult                = MatMult_MPIKAIJ;
    A->ops->multadd             = MatMultAdd_MPIKAIJ;
    A->ops->invertblockdiagonal = MatInvertBlockDiagonal_MPIKAIJ;
    A->ops->getrow              = MatGetRow_MPIKAIJ;
    A->ops->restorerow          = MatRestoreRow_MPIKAIJ;
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatGetDiagonalBlock_C",MatGetDiagonalBlock_MPIKAIJ));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatConvert_mpikaij_mpiaij_C",MatConvert_KAIJ_AIJ));
  }
  A->ops->setup           = MatSetUp_KAIJ;
  A->ops->view            = MatView_KAIJ;
  A->ops->createsubmatrix = MatCreateSubMatrix_KAIJ;
  PetscFunctionReturn(0);
}
