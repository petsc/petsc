
/*
   This is where the abstract matrix operations are defined
*/

#include "petsc.h"
#include "matimpl.h"        /*I "mat.h" I*/
#include "vec/vecimpl.h"  
       
/*@
    MatGetReordering - 

  Input Parameters:
.  mat - the matrix
.  type - type of reordering; one of:ORDER_ND,ORDER_1WD,ORDER_RCM,ORDER_QMD

  OutPut Parameters:
.  rperm - row permutation indices
.  cperm - column permutation indices

  If the column permutations and row permutations are the same, then
returns 0 in cperm.
@*/
int MatGetReordering(Mat mat,int type,IS *rperm,IS *cperm)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (!mat->ops->order) SETERR(1,"Cannot reorder this matrix");
  return (*mat->ops->order)(mat,type,rperm,cperm);
}

/*@
    MatGetRow - gets a row of a matrix. This routine is provided
               for people who need to have direct address to the 
               structure of a matrix, we hope that we provide 
               enough high level matrix routines so that few users
               need it. You MUST call MatRestoreRow() for each row
               that you get to insure that your application does
               not bleed memory.

  Input Parameters:
.  mat - the matrix
.  row - the row to get

  OutPut Parameters:
.  ncols, cols - the number of nonzeros and their columns
.  vals - if nonzero the column values
@*/
int MatGetRow(Mat mat,int row,int *ncols,int **cols,Scalar **vals)
{
  VALIDHEADER(mat,MAT_COOKIE);
  return (*mat->ops->getrow)(mat,row,ncols,cols,vals);
}

/*@  
     MatRestoreRow - frees any temporary space malloced by
                     MatGetRow().

  Input Parameters:
.  mat - the matrix
.  row - the row to get
.  ncols, cols - the number of nonzeros and their columns
.  vals - if nonzero the column values
@*/
int MatRestoreRow(Mat mat,int row,int *ncols,int **cols,Scalar **vals)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (!mat->ops->restorerow) return 0;
  return (*mat->ops->restorerow)(mat,row,ncols,cols,vals);
}
/*@
    MatView - visualize a matrix object.

  Input Parameters:
.   mat - the matrix
.   ptr - a visualization context
@*/
int MatView(Mat mat,Viewer ptr)
{
  VALIDHEADER(mat,MAT_COOKIE);
  return (*mat->view)((PetscObject)mat,ptr);
}
/*@
    MatDestroy - frees space taken by matrix
  
  Input Parameters:
.  mat the matrix
@*/
int MatDestroy(Mat mat)
{
  VALIDHEADER(mat,MAT_COOKIE);
  return (*mat->destroy)((PetscObject)mat);
}
/*@
    MatValidMatrix - returns 1 if a valid matrix else 0

  Input Parameter:
.   m - the matrix to check 
@*/
int MatValidMatrix(Mat m)
{
  if (!m) return 0;
  if (m->cookie != MAT_COOKIE) return 0;
  return 1;
}

/*@ 
    MatInsertValues - inserts a block of values into a matrix

  Input Parameters:
.  mat - the matrix
.  v - a logically two dimensional array of values, Fortran ordering
.  m, indexm - the number of rows and their indices 
.  n, indexn - the number of columns and their indices
@*/
int MatInsertValues(Mat mat,Scalar *v,int m,int *indexm,int n,int *indexn)
{
  VALIDHEADER(mat,MAT_COOKIE);
  return (*mat->ops->insert)(mat,v,m,indexm,n,indexn);
}
/*@ 
    MatAddValues - adds a block of values into a matrix

  Input Parameters:
.  mat - the matrix
.  v - a logically two dimensional array of values, Fortran ordering
.  m, indexm - the number of rows and their indices 
.  n, indexn - the number of columns and their indices
@*/
int MatAddValues(Mat mat,Scalar *v,int m,int *indexm,int n,int *indexn)
{
  VALIDHEADER(mat,MAT_COOKIE);
  return (*mat->ops->insertadd)(mat,v,m,indexm,n,indexn);
}
/* --------------------------------------------------------*/
/*@
    MatMult - Matrix vector multiply

  Input Parameters:
.    mat -the matrix
.     x  - the vector to be multilplied
  Output Parameters:
.      y - the result
@*/
int MatMult(Mat mat,Vec x,Vec y)
{
  VALIDHEADER(mat,MAT_COOKIE);
  VALIDHEADER(x,VEC_COOKIE);VALIDHEADER(y,VEC_COOKIE); 
  return (*mat->ops->mult)(mat,x,y);
}   
/*@
    MatMultTrans - Matrix transpose times a vector

  Input Parameters:
.    mat -the matrix
.     x  - the vector to be multilplied
  Output Parameters:
.      y - the result
@*/
int MatMultTrans(Mat mat,Vec x,Vec y)
{
  VALIDHEADER(mat,MAT_COOKIE);
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  return (*mat->ops->multtrans)(mat,x,y);
}   
/*@
    MatMultAdd -  v3 = v2 + A v1

  Input Parameters:
.    mat -the matrix
.     v1, v2  - the vectors
  Output Parameters:
.     v3 - the result
@*/
int MatMultAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  VALIDHEADER(mat,MAT_COOKIE);VALIDHEADER(v1,VEC_COOKIE);
  VALIDHEADER(v2,VEC_COOKIE); VALIDHEADER(v3,VEC_COOKIE);
  return (*mat->ops->multadd)(mat,v1,v2,v3);
}   
/*@
    MatMultTransAdd -  v3 = v2 + A' v1

  Input Parameters:
.    mat -the matrix
.     v1, v2  - the vectors
  Output Parameters:
.     v3 - the result
@*/
int MatMultTransAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  VALIDHEADER(mat,MAT_COOKIE); VALIDHEADER(v1,VEC_COOKIE);
  VALIDHEADER(v2,VEC_COOKIE); VALIDHEADER(v3,VEC_COOKIE);
  return (*mat->ops->multtransadd)(mat,v1,v2,v3);
}
/* ------------------------------------------------------------*/
/*@
     MatNonZeros - returns the number of nonzeros in a matrix

  Input Parameters:
.   mat - the matrix

  Output Parameters:
.   returns the number of nonzeros
@*/
int MatNonZeros(Mat mat,int *nz)
{
  VALIDHEADER(mat,MAT_COOKIE);
  return  (*mat->ops->NZ)(mat,nz);
}   
/*@
     MatMemoryUsed - returns the amount of memory used to store matrix.

  Input Parameters:
.   mat - the matrix

  Output Parameters:
.   mem - memory used
@*/
int MatMemoryUsed(Mat mat,int *mem)
{
  VALIDHEADER(mat,MAT_COOKIE);
  return  (*mat->ops->memory)(mat,mem);
}
/* ----------------------------------------------------------*/
/*@  
    MatLUFactor - performs an inplace LU factorization of matrix.
             See MatLUFactorSymbolic() and MatLUFactorNumeric() for 
             out-of-place factorization. See MatCholeskyFactor()
             for symmetric, positive definite case.

  Input Parameters:
.   mat - the matrix
.   row, col - row and  column permutations

@*/
int MatLUFactor(Mat mat,IS row,IS col)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->lufactor) return (*mat->ops->lufactor)(mat,row,col);
  SETERR(1,"No MatLUFactor for implementation");
}
/*@  
    MatLUFactorSymbolic - performs a symbolic LU factorization of matrix.
             See MatLUFactor() for in-place factorization. See
             MatCholeskyFactor() for symmetric, positive definite case.

  Input Parameters:
.   mat - the matrix
.   row, col - row and  column permutations

  Output Parameters:
.   fact - puts factor else does inplace factorization
@*/
int MatLUFactorSymbolic(Mat mat,IS row,IS col,Mat *fact)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (!fact) SETERR(1,"Missing slot for symbolic factorization");
  if (mat->ops->lufactorsymbolic) return (*mat->ops->lufactorsymbolic)(mat,row,
                                                                    col,fact);
  SETERR(1,"No MatLUFactorSymbolic for implementation");
}
/*@  
    MatLUFactorNumeric - performs a numeric LU factorization of matrix.
             See MatLUFactor() for in-place factorization. See
             MatCholeskyFactor() for symmetric, positive definite case.

  Input Parameters:
.   mat - the matrix
.   row, col - row and  column permutations

  Output Parameters:
.   fact - must have been obtained with MatLUFactorSymbolic()
@*/
int MatLUFactorNumeric(Mat mat,Mat fact)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (!fact) SETERR(1,"Missing symbolic factorization");
  if (mat->ops->lufactornumeric) return (*mat->ops->lufactornumeric)(mat,fact);
  SETERR(1,"No MatLUFactorNumeric for implementation");
}
/*@  
    MatCholeskyFactor - performs an inplace CC' factorization of matrix.
          See also MatLUFactor(), MatCholeskyFactorSymbolic() and
          MatCholeskyFactorNumeric().

  Input Parameters:
.   mat - the matrix
.   perm - row and column permutations

@*/
int MatCholeskyFactor(Mat mat,IS perm)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->chfactor) return (*mat->ops->chfactor)(mat,perm);
  SETERR(1,"No MatCholeskyFactor for implementation");
}
/*@  
    MatCholeskyFactorSymbolic - performs a symbolic Cholesky factorization.
          See also MatLUFactor(), MatCholeskyFactorSymbolic() and
          MatCholeskyFactorNumeric().

  Input Parameters:
.   mat - the matrix
.   perm - row and column permutations

@*/
int MatCholeskyFactorSymbolic(Mat mat,IS perm,Mat *fact)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (!fact) SETERR(1,"Missing slot for symbolic factorization");
  if (mat->ops->chfactorsymbolic) return (*mat->ops->chfactorsymbolic)(mat,
                                            perm,fact);
  SETERR(1,"No MatCholeskyFactorSymbolic for implementation");
}
/*@  
    MatCholeskyFactorNumeric - performs a numeric Cholesky factorization.
          See also MatLUFactor(),  MatCholeskyFactor() and
          MatCholeskyFactorSymbolic().

  Input Parameters:
.   mat - the matrix


@*/
int MatCholeskyFactorNumeric(Mat mat,Mat fact)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (!fact) SETERR(1,"Missing symbolic factorization");
  if (mat->ops->chfactornumeric) return (*mat->ops->chfactornumeric)(mat,
                                            fact);
  SETERR(1,"No MatCholeskyFactorNumeric for implementation");
}
/* ----------------------------------------------------------------*/
/*@
    MatSolve - solve A x = b 

  Input Parameters:
.   mat - the  factored matrix
.    b - the right hand side

  Output Parameter:
.    x- the result
@*/
int MatSolve(Mat mat,Vec b,Vec x)
{
  VALIDHEADER(mat,MAT_COOKIE);
  VALIDHEADER(b,VEC_COOKIE);  VALIDHEADER(x,VEC_COOKIE);
  if (!mat->factor) SETERR(1,"Attempt solve on nonfactored matrix");
  if (mat->ops->solve) return (*mat->ops->solve)(mat,b,x);
  SETERR(1,"No MatSolve for implementation");
}
/*@
    MatSolveAdd - x = y + A^-1 b 

  Input Parameters:
.   mat - the  factored matrix
.    b - the right hand side
.    y - te vector to be added to 

  Output Parameter:
.    x- the result
@*/
int MatSolveAdd(Mat mat,Vec b,Vec y,Vec x)
{
  Scalar one = 1.0;
  Vec    tmp;
  int    ierr;
  VALIDHEADER(mat,MAT_COOKIE);VALIDHEADER(y,VEC_COOKIE);
  VALIDHEADER(b,VEC_COOKIE);  VALIDHEADER(x,VEC_COOKIE);
  if (!mat->factor) SETERR(1,"Attempt solve on nonfactored matrix");
  if (mat->ops->solveadd) return (*mat->ops->solveadd)(mat,b,y,x);
  /* do the solve then the add manually */
  if (x != y) {
    ierr = MatSolve(mat,b,x); CHKERR(ierr);
    ierr = VecAXPY(&one,y,x); CHKERR(ierr);
    return 0;
  }
  else {
    ierr = VecCreate(x,&tmp); CHKERR(ierr);
    ierr = VecCopy(x,tmp); CHKERR(ierr);
    ierr = MatSolve(mat,b,x); CHKERR(ierr);
    ierr = VecAXPY(&one,tmp,x); CHKERR(ierr);
    ierr = VecDestroy(tmp); CHKERR(ierr);
    return 0;
  }
}
/*@
    MatSolveTrans - solve A' x = b 

  Input Parameters:
.   mat - the  factored matrix
.    b - the right hand side

  Output Parameter:
.    x- the result
@*/
int MatSolveTrans(Mat mat,Vec b,Vec x)
{
  VALIDHEADER(mat,MAT_COOKIE);
  VALIDHEADER(b,VEC_COOKIE);  VALIDHEADER(x,VEC_COOKIE);
  if (!mat->factor) SETERR(1,"Attempt solve on nonfactored matrix");
  if (mat->ops->solvetrans) return (*mat->ops->solvetrans)(mat,b,x);
  SETERR(1,"No MatSolveTrans for implementation");
}
/*@
    MatSolveTransAdd - x = y + A^-T b 

  Input Parameters:
.   mat - the  factored matrix
.    b - the right hand side
.    y - te vector to be added to 

  Output Parameter:
.    x- the result
@*/
int MatSolveTransAdd(Mat mat,Vec b,Vec y,Vec x)
{
  Scalar one = 1.0;
  int    ierr;
  Vec    tmp;
  VALIDHEADER(mat,MAT_COOKIE);VALIDHEADER(y,VEC_COOKIE);
  VALIDHEADER(b,VEC_COOKIE);  VALIDHEADER(x,VEC_COOKIE);
  if (!mat->factor) SETERR(1,"Attempt solve on nonfactored matrix");
  if (mat->ops->solvetransadd) return (*mat->ops->solvetransadd)(mat,b,y,x);
  /* do the solve then the add manually */
  if (x != y) {
    ierr = MatSolveTrans(mat,b,x); CHKERR(ierr);
    ierr = VecAXPY(&one,y,x); CHKERR(ierr);
    return 0;
  }
  else {
    ierr = VecCreate(x,&tmp); CHKERR(ierr);
    ierr = VecCopy(x,tmp); CHKERR(ierr);
    ierr = MatSolveTrans(mat,b,x); CHKERR(ierr);
    ierr = VecAXPY(&one,tmp,x); CHKERR(ierr);
    ierr = VecDestroy(tmp); CHKERR(ierr);
    return 0;
  }
}
/* ----------------------------------------------------------------*/

/*@
    MatRelax - one relaxation sweep

  Input Parameters:
.  mat - the matrix
.  b - the right hand side
.  omega - the relaxation factor
.  flag - or together from  SOR_FORWARD_SWEEP, SOR_BACKWARD_SWEEP, 
.         SOR_ZERO_INITIAL_GUESS. (SOR_SYMMETRIC_SWEEP is 
.         equivalent to SOR_FORWARD_SWEEP | SOR_BACKWARD_SWEEP)
.  is - optional index set indicating ordering
.  its - the number of iterations

  Output Parameters:
.  x - the solution - can contain initial guess
@*/
int MatRelax(Mat mat,Vec b,double omega,int flag,IS is,int its,Vec x)
{
  VALIDHEADER(mat,MAT_COOKIE);
  VALIDHEADER(b,VEC_COOKIE);  VALIDHEADER(x,VEC_COOKIE);
  if (flag < 1 || flag > 7) SETERR(1,"Bad relaxation type");
  if (mat->ops->relax) return (*mat->ops->relax)(mat,b,omega,flag,is,its,x);
  SETERR(1,"No MatRelax for implementation");
}

/* ----------------------------------------------------------------*/
/*@  
    MatCopy - copies a matrix

  Input Parameters:
.   mat - the matrix

  Output Parameters:
.    M - pointer to place new matrix
@*/
int MatCopy(Mat mat,Mat *M)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->copy) return (*mat->ops->copy)(mat,M);
  SETERR(1,"No MatCopy for implementation");
}

/*@ 
     MatGetDiagonal - get diagonal of matrix

  Input Parameters:
.  mat - the matrix

  Output Parameters:
.  v - the vector to store the diagonal in
@*/
int MatGetDiagonal(Mat mat,Vec v)
{
  VALIDHEADER(mat,MAT_COOKIE);
  VALIDHEADER(v,VEC_COOKIE);
  if (mat->ops->getdiag) return (*mat->ops->getdiag)(mat,v);
  SETERR(1,"No MatGetDiagonal for implementaion");
}

/*@ 
     MatTranspose - in place transpose of matrix

  Input Parameters:
.   mat - the matrix to transpose
@*/
int MatTranspose(Mat mat)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->trans) return (*mat->ops->trans)(mat);
  SETERR(1,"No MatTranspose for implementaion");
}

/*@
    MatEqual - returns 1 if two matrices are equal

  Input Parameters:
.  mat1, mat2 - the matrices

  OutPut Parameter:
.   returns 1 if matrices are equal
@*/
int MatEqual(Mat mat1,Mat mat2)
{
  VALIDHEADER(mat1,MAT_COOKIE); VALIDHEADER(mat2,MAT_COOKIE);
  if (mat1->ops->equal) return (*mat1->ops->equal)(mat1,mat2);
  SETERR(1,"No MatEqual for implementaion");
}

/*@
     MatScale - scale a matrix on the left and right by diagonal
                matrices stored as vectors. Either of the two may be 
                null.

  Input Parameters:
.   mat - the matrix to be scaled
.   l,r - the left and right scaling vectors
@*/
int MatScale(Mat mat,Vec l,Vec r)
{
  VALIDHEADER(mat,MAT_COOKIE);
  VALIDHEADER(l,VEC_COOKIE); VALIDHEADER(r,VEC_COOKIE);
  if (mat->ops->scale) return (*mat->ops->scale)(mat,l,r);
  SETERR(1,"No MatScale for implementaion");
} 

/*@ 
     MatNorm - calculate various norms of a matrix

  Input Parameters:
.  mat - the matrix
.  type - the type of norm, NORM_1, NORM_2, NORM_FROBENIUS, NORM_INFINITY

  Output Parameters:
.  norm - the resulting norm 
@*/
int MatNorm(Mat mat,int type,double *norm)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->norm) return (*mat->ops->norm)(mat,type,norm);
  SETERR(1,"No MatNorm for implementaion");
}

/*@
     MatBeginAssembly - begin assemblying the matrix. This should
                        be called after all the calls to MatInsertValues()
                        and MatAddValues().

  Input Parameters:
.  mat - the matrix 
 
   Note: when you call MatInsertValues() it generally caches the values
         only after you have called MatBeginAssembly() and 
         MatEndAssembly() is the matrix ready to be used.
@*/
int MatBeginAssembly(Mat mat)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->bassembly) return (*mat->ops->bassembly)(mat);
  return 0;
}
/*@
     MatEndAssembly - begin assemblying the matrix. This should
                        be called after all the calls to MatInsertValues()
                        and MatAddValues() and after MatBeginAssembly().

  Input Parameters:
.  mat - the matrix 
 
   Note: when you call MatInsertValues() it generally caches the values
         only after you have called MatBeginAssembly() and 
         MatEndAssembly() is the matrix ready to be used.
@*/
int MatEndAssembly(Mat mat)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->eassembly) return (*mat->ops->eassembly)(mat);
  return 0;
}
/*@
     MatCompress - trys to store matrix in as little space as 
        possible. May fail if memory is already fully used as
        it tries to allocate new space.

  Input Parameters:
.  mat - the matrix 
 
@*/
int MatCompress(Mat mat)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->compress) return (*mat->ops->compress)(mat);
  return 0;
}
/*@
     MatSetInsertOption - set parameter option on how you will insert 
         (or add) values. In general sorted, row oriented input will assemble
         fastest. Defaults to row oriented, nonsorted input. Note that 
         if you are using a Fortran 77 module to compute an element 
         stiffness you may need to modify the module or use column 
         oriented input.

  Input Parameters:
.  mat - the matrix 
.  option - one of ROW_ORIENTED, COLUMN_ORIENTED, ROWS_SORTED,
                   COLUMNS_SORTED, NO_NEW_NONZERO_LOCATIONS

       NO_NEW_NONZERO_LOCATIONS indicates that any add or insertion 
that will generate a new entry in the nonzero structure is ignored.
What this means is if memory is not allocated for this particular 
lot then the insertion is ignored. For dense matrices where  
the entire array is allocated no entries are every ignored. This
may not be a good idea??
 
@*/
int MatSetInsertOption(Mat mat,int op)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (op < 0 || op > 7) SETERR(1,"Invalid option to MatSetInsertOption");
  if (mat->ops->insopt) return (*mat->ops->insopt)(mat,op);
  return 0;
}

/*@
     MatZeroEntries - zeros all entries in a matrix. For sparse matrices
         it keeps the old nonzero structure.

  Input Parameters:
.  mat - the matrix 
 
@*/
int MatZeroEntries(Mat mat)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->zeroentries) return (*mat->ops->zeroentries)(mat);
  SETERR(1,"No MatZeroEntries for implementation");
}

/*@
     MatZeroRow - zeros all entries in a row of a matrix. For sparse matrices
         it removes the old nonzero structure.

  Input Parameters:
.  mat - the matrix
.  row - the row to zap 
 
@*/
int MatZeroRow(int row,Mat mat)
{
  VALIDHEADER(mat,MAT_COOKIE);
  if (mat->ops->zerorow) return (*mat->ops->zerorow)(row,mat);
  SETERR(1,"No MatZeroRow for implementation");
}

/*@
     MatScatterBegin - Begins a scatter of one matrix into another.
         This is much more general than a standard scatter, it include
         scatter, gather and cmobinations. In a parallel enviroment
         it can be used to simultaneous gather from a global matrix
         into local and vice-versa.

  Input Parameters:
.  mat - the matrix
.  is1,is2 - the rows and columns to snag
.  is3,is4 - rows and columns to drop snagged in

  Output Paramters:
.  out - matrix to receive
.  ctx - optional pointer to MatScatterCtx - allows reuse of communcation
         pattern if same scatter used more than once.
 
  Keywords: matrix, scatter, gather
@*/
int MatScatterBegin(Mat mat,IS is1,IS is2,Mat out,IS is3, IS is4,
                                                 MatScatterCtx *ctx)
{
  VALIDHEADER(mat,MAT_COOKIE);  VALIDHEADER(out,MAT_COOKIE);
  if (mat->ops->scatbegin)
    return (*mat->ops->scatbegin)(mat,is1,is2,out,is3,is4,ctx);
  SETERR(1,"No MatScatterBegin for implementation");
}

