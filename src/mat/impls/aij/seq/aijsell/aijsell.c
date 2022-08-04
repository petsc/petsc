/*
  Defines basic operations for the MATSEQAIJSELL matrix class.
  This class is derived from the MATAIJCLASS, but maintains a "shadow" copy
  of the matrix stored in MATSEQSELL format, which is used as appropriate for
  performing operations for which this format is more suitable.
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sell/seq/sell.h>

typedef struct {
  Mat              S; /* The SELL formatted "shadow" matrix. */
  PetscBool        eager_shadow;
  PetscObjectState state; /* State of the matrix when shadow matrix was last constructed. */
} Mat_SeqAIJSELL;

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJSELL_SeqAIJ(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert a MATAIJSELL to its base PETSc type, */
  /* so we will ignore 'MatType type'. */
  Mat            B        = *newmat;
  Mat_SeqAIJSELL *aijsell = (Mat_SeqAIJSELL*) A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }

  /* Reset the original function pointers. */
  B->ops->duplicate        = MatDuplicate_SeqAIJ;
  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJ;
  B->ops->destroy          = MatDestroy_SeqAIJ;
  B->ops->mult             = MatMult_SeqAIJ;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJ;
  B->ops->multadd          = MatMultAdd_SeqAIJ;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJ;
  B->ops->sor              = MatSOR_SeqAIJ;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijsell_seqaij_C",NULL));

  if (reuse == MAT_INITIAL_MATRIX) aijsell = (Mat_SeqAIJSELL*)B->spptr;

  /* Clean up the Mat_SeqAIJSELL data structure.
   * Note that MatDestroy() simply returns if passed a NULL value, so it's OK to call even if the shadow matrix was never constructed. */
  PetscCall(MatDestroy(&aijsell->S));
  PetscCall(PetscFree(B->spptr));

  /* Change the type of B to MATSEQAIJ. */
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQAIJ));

  *newmat = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJSELL(Mat A)
{
  Mat_SeqAIJSELL  *aijsell = (Mat_SeqAIJSELL*) A->spptr;

  PetscFunctionBegin;

  /* If MatHeaderMerge() was used, then this SeqAIJSELL matrix will not have an
   * spptr pointer. */
  if (aijsell) {
    /* Clean up everything in the Mat_SeqAIJSELL data structure, then free A->spptr. */
    PetscCall(MatDestroy(&aijsell->S));
    PetscCall(PetscFree(A->spptr));
  }

  /* Change the type of A back to SEQAIJ and use MatDestroy_SeqAIJ()
   * to destroy everything that remains. */
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATSEQAIJ));
  /* Note that I don't call MatSetType().  I believe this is because that
   * is only to be called when *building* a matrix.  I could be wrong, but
   * that is how things work for the SuperLU matrix class. */
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqaijsell_seqaij_C",NULL));
  PetscCall(MatDestroy_SeqAIJ(A));
  PetscFunctionReturn(0);
}

/* Build or update the shadow matrix if and only if needed.
 * We track the ObjectState to determine when this needs to be done. */
PETSC_INTERN PetscErrorCode MatSeqAIJSELL_build_shadow(Mat A)
{
  Mat_SeqAIJSELL   *aijsell = (Mat_SeqAIJSELL*) A->spptr;
  PetscObjectState state;

  PetscFunctionBegin;
  PetscCall(PetscObjectStateGet((PetscObject)A,&state));
  if (aijsell->S && aijsell->state == state) {
    /* The existing shadow matrix is up-to-date, so simply exit. */
    PetscFunctionReturn(0);
  }

  PetscCall(PetscLogEventBegin(MAT_Convert,A,0,0,0));
  if (aijsell->S) {
    PetscCall(MatConvert_SeqAIJ_SeqSELL(A,MATSEQSELL,MAT_REUSE_MATRIX,&aijsell->S));
  } else {
    PetscCall(MatConvert_SeqAIJ_SeqSELL(A,MATSEQSELL,MAT_INITIAL_MATRIX,&aijsell->S));
  }
  PetscCall(PetscLogEventEnd(MAT_Convert,A,0,0,0));

  /* Record the ObjectState so that we can tell when the shadow matrix needs updating */
  PetscCall(PetscObjectStateGet((PetscObject)A,&aijsell->state));

  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqAIJSELL(Mat A, MatDuplicateOption op, Mat *M)
{
  Mat_SeqAIJSELL *aijsell;
  Mat_SeqAIJSELL *aijsell_dest;

  PetscFunctionBegin;
  PetscCall(MatDuplicate_SeqAIJ(A,op,M));
  aijsell      = (Mat_SeqAIJSELL*) A->spptr;
  aijsell_dest = (Mat_SeqAIJSELL*) (*M)->spptr;
  PetscCall(PetscArraycpy(aijsell_dest,aijsell,1));
  /* We don't duplicate the shadow matrix -- that will be constructed as needed. */
  aijsell_dest->S = NULL;
  if (aijsell->eager_shadow) PetscCall(MatSeqAIJSELL_build_shadow(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqAIJSELL(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJSELL  *aijsell = (Mat_SeqAIJSELL*)A->spptr;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* I disable the use of the inode routines so that the AIJSELL ones will be
   * used instead, but I wonder if it might make sense (and is feasible) to
   * use some of them. */
  a->inode.use = PETSC_FALSE;

  /* Since a MATSEQAIJSELL matrix is really just a MATSEQAIJ with some
   * extra information and some different methods, call the AssemblyEnd
   * routine for a MATSEQAIJ.
   * I'm not sure if this is the best way to do this, but it avoids
   * a lot of code duplication. */

  PetscCall(MatAssemblyEnd_SeqAIJ(A, mode));

  /* If the user has requested "eager" shadowing, create the SELL shadow matrix (if needed; the function checks).
   * (The default is to take a "lazy" approach, deferring this until something like MatMult() is called.) */
  if (aijsell->eager_shadow) PetscCall(MatSeqAIJSELL_build_shadow(A));

  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqAIJSELL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJSELL    *aijsell = (Mat_SeqAIJSELL*)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJSELL_build_shadow(A));
  PetscCall(MatMult_SeqSELL(aijsell->S,xx,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqAIJSELL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJSELL    *aijsell=(Mat_SeqAIJSELL*)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJSELL_build_shadow(A));
  PetscCall(MatMultTranspose_SeqSELL(aijsell->S,xx,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqAIJSELL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJSELL    *aijsell=(Mat_SeqAIJSELL*)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJSELL_build_shadow(A));
  PetscCall(MatMultAdd_SeqSELL(aijsell->S,xx,yy,zz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqAIJSELL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJSELL    *aijsell=(Mat_SeqAIJSELL*)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJSELL_build_shadow(A));
  PetscCall(MatMultTransposeAdd_SeqSELL(aijsell->S,xx,yy,zz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_SeqAIJSELL(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqAIJSELL    *aijsell=(Mat_SeqAIJSELL*)A->spptr;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJSELL_build_shadow(A));
  PetscCall(MatSOR_SeqSELL(aijsell->S,bb,omega,flag,fshift,its,lits,xx));
  PetscFunctionReturn(0);
}

/* MatConvert_SeqAIJ_SeqAIJSELL converts a SeqAIJ matrix into a
 * SeqAIJSELL matrix.  This routine is called by the MatCreate_SeqAIJSELL()
 * routine, but can also be used to convert an assembled SeqAIJ matrix
 * into a SeqAIJSELL one. */
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJSELL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;
  Mat_SeqAIJ     *b;
  Mat_SeqAIJSELL *aijsell;
  PetscBool      set;
  PetscBool      sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)A,type,&sametype));
  if (sametype) PetscFunctionReturn(0);

  PetscCall(PetscNewLog(B,&aijsell));
  b        = (Mat_SeqAIJ*) B->data;
  B->spptr = (void*) aijsell;

  /* Disable use of the inode routines so that the AIJSELL ones will be used instead.
   * This happens in MatAssemblyEnd_SeqAIJSELL as well, but the assembly end may not be called, so set it here, too.
   * As noted elsewhere, I wonder if it might make sense and be feasible to use some of the inode routines. */
  b->inode.use = PETSC_FALSE;

  /* Set function pointers for methods that we inherit from AIJ but override.
   * We also parse some command line options below, since those determine some of the methods we point to. */
  B->ops->duplicate        = MatDuplicate_SeqAIJSELL;
  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJSELL;
  B->ops->destroy          = MatDestroy_SeqAIJSELL;

  aijsell->S = NULL;
  aijsell->eager_shadow = PETSC_FALSE;

  /* Parse command line options. */
  PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"AIJSELL Options","Mat");
  PetscCall(PetscOptionsBool("-mat_aijsell_eager_shadow","Eager Shadowing","None",(PetscBool)aijsell->eager_shadow,(PetscBool*)&aijsell->eager_shadow,&set));
  PetscOptionsEnd();

  /* If A has already been assembled and eager shadowing is specified, build the shadow matrix. */
  if (A->assembled && aijsell->eager_shadow) {
    PetscCall(MatSeqAIJSELL_build_shadow(A));
  }

  B->ops->mult             = MatMult_SeqAIJSELL;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJSELL;
  B->ops->multadd          = MatMultAdd_SeqAIJSELL;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJSELL;
  B->ops->sor              = MatSOR_SeqAIJSELL;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijsell_seqaij_C",MatConvert_SeqAIJSELL_SeqAIJ));

  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJSELL));
  *newmat = B;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqAIJSELL - Creates a sparse matrix of type SEQAIJSELL.
   This type inherits from AIJ and is largely identical, but keeps a "shadow"
   copy of the matrix in SEQSELL format, which is used when this format
   may be more suitable for a requested operation. Currently, SEQSELL format
   is used for MatMult, MatMultTranspose, MatMultAdd, MatMultTransposeAdd,
   and MatSOR operations.
   Because SEQAIJSELL is a subtype of SEQAIJ, the option "-mat_seqaij_type seqaijsell" can be used to make
   sequential AIJ matrices default to being instances of MATSEQAIJSELL.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   Options Database Keys:
.  -mat_aijsell_eager_shadow - Construct shadow matrix upon matrix assembly; default is to take a "lazy" approach, performing this step the first time the matrix is applied

   Notes:
   If nnz is given then nz is ignored

   Level: intermediate

.seealso: `MatCreate()`, `MatCreateMPIAIJSELL()`, `MatSetValues()`
@*/
PetscErrorCode  MatCreateSeqAIJSELL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,m,n));
  PetscCall(MatSetType(*A,MATSEQAIJSELL));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,nnz));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJSELL(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatConvert_SeqAIJ_SeqAIJSELL(A,MATSEQAIJSELL,MAT_INPLACE_MATRIX,&A));
  PetscFunctionReturn(0);
}
