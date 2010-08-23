#define PETSCMAT_DLL

/*
   This is where the abstract matrix operations are defined
*/

#include "private/matimpl.h"        /*I "petscmat.h" I*/
#include "private/vecimpl.h"  

/* Logging support */
PetscCookie PETSCMAT_DLLEXPORT MAT_COOKIE;
PetscCookie PETSCMAT_DLLEXPORT MAT_FDCOLORING_COOKIE;

PetscLogEvent  MAT_Mult, MAT_Mults, MAT_MultConstrained, MAT_MultAdd, MAT_MultTranspose;
PetscLogEvent  MAT_MultTransposeConstrained, MAT_MultTransposeAdd, MAT_Solve, MAT_Solves, MAT_SolveAdd, MAT_SolveTranspose, MAT_MatSolve;
PetscLogEvent  MAT_SolveTransposeAdd, MAT_SOR, MAT_ForwardSolve, MAT_BackwardSolve, MAT_LUFactor, MAT_LUFactorSymbolic;
PetscLogEvent  MAT_LUFactorNumeric, MAT_CholeskyFactor, MAT_CholeskyFactorSymbolic, MAT_CholeskyFactorNumeric, MAT_ILUFactor;
PetscLogEvent  MAT_ILUFactorSymbolic, MAT_ICCFactorSymbolic, MAT_Copy, MAT_Convert, MAT_Scale, MAT_AssemblyBegin;
PetscLogEvent  MAT_AssemblyEnd, MAT_SetValues, MAT_GetValues, MAT_GetRow, MAT_GetRowIJ, MAT_GetSubMatrices, MAT_GetColoring, MAT_GetOrdering, MAT_GetRedundantMatrix, MAT_GetSeqNonzeroStructure;
PetscLogEvent  MAT_IncreaseOverlap, MAT_Partitioning, MAT_ZeroEntries, MAT_Load, MAT_View, MAT_AXPY, MAT_FDColoringCreate;
PetscLogEvent  MAT_FDColoringApply,MAT_Transpose,MAT_FDColoringFunction;
PetscLogEvent  MAT_MatMult, MAT_MatMultSymbolic, MAT_MatMultNumeric;
PetscLogEvent  MAT_PtAP, MAT_PtAPSymbolic, MAT_PtAPNumeric;
PetscLogEvent  MAT_MatMultTranspose, MAT_MatMultTransposeSymbolic, MAT_MatMultTransposeNumeric;
PetscLogEvent  MAT_MultHermitianTranspose,MAT_MultHermitianTransposeAdd;
PetscLogEvent  MAT_Getsymtranspose, MAT_Getsymtransreduced, MAT_Transpose_SeqAIJ, MAT_GetBrowsOfAcols;
PetscLogEvent  MAT_GetBrowsOfAocols, MAT_Getlocalmat, MAT_Getlocalmatcondensed, MAT_Seqstompi, MAT_Seqstompinum, MAT_Seqstompisym;
PetscLogEvent  MAT_Applypapt, MAT_Applypapt_numeric, MAT_Applypapt_symbolic, MAT_GetSequentialNonzeroStructure;

/* nasty global values for MatSetValue() */
PetscInt    PETSCMAT_DLLEXPORT MatSetValue_Row = 0;
PetscInt    PETSCMAT_DLLEXPORT MatSetValue_Column = 0;
PetscScalar PETSCMAT_DLLEXPORT MatSetValue_Value = 0.0;

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonalBlock"
/*@
   MatGetDiagonalBlock - Returns the part of the matrix associated with the on-process coupling

   Not Collective

   Input Parameters:
+  mat - the matrix
-  reuse - indicates you are passing in the a matrix and want it reused

   Output Parameters:
+   iscopy - indicates a copy of the diagonal matrix was created and you should use MatDestroy() on it
-   a - the diagonal part (which is a SEQUENTIAL matrix)

   Notes: see the manual page for MatCreateMPIAIJ() for more information on the "diagonal part" of the matrix

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetDiagonalBlock(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  PetscErrorCode ierr,(*f)(Mat,PetscTruth*,MatReuse,Mat*);
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidPointer(iscopy,2);
  PetscValidPointer(a,3);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,iscopy,reuse,a);CHKERRQ(ierr);
  } else if (size == 1) {
    *a = A;
    *iscopy = PETSC_FALSE;
  } else {
    SETERRQ(PETSC_ERR_SUP,"Cannot get diagonal part for this matrix");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetTrace"
/*@
   MatGetTrace - Gets the trace of a matrix. The sum of the diagonal entries.

   Collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameter:
.   trace - the sum of the diagonal entries

   Level: advanced

@*/
PetscErrorCode  MatGetTrace(Mat mat,PetscScalar *trace)
{
   PetscErrorCode ierr;
   Vec            diag;

   PetscFunctionBegin;
   ierr = MatGetVecs(mat,&diag,PETSC_NULL);CHKERRQ(ierr);
   ierr = MatGetDiagonal(mat,diag);CHKERRQ(ierr);
   ierr = VecSum(diag,trace);CHKERRQ(ierr);
   ierr = VecDestroy(diag);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRealPart"
/*@
   MatRealPart - Zeros out the imaginary part of the matrix

   Collective on Mat

   Input Parameters:
.  mat - the matrix

   Level: advanced


.seealso: MatImaginaryPart()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRealPart(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->realpart) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = (*mat->ops->realpart)(mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetGhosts"
/*@C
   MatGetGhosts - Get the global index of all ghost nodes defined by the sparse matrix

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+   nghosts - number of ghosts (note for BAIJ matrices there is one ghost for each block)
-   ghosts - the global indices of the ghost points

   Notes: the nghosts and ghosts are suitable to pass into VecCreateGhost()

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetGhosts(Mat mat,PetscInt *nghosts,const PetscInt *ghosts[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->getghosts) {
    if (nghosts) *nghosts = 0;
    if (ghosts) *ghosts = 0;
  } else {
    ierr = (*mat->ops->getghosts)(mat,nghosts,ghosts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatImaginaryPart"
/*@
   MatImaginaryPart - Moves the imaginary part of the matrix to the real part and zeros the imaginary part

   Collective on Mat

   Input Parameters:
.  mat - the matrix

   Level: advanced


.seealso: MatRealPart()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatImaginaryPart(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->imaginarypart) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = (*mat->ops->imaginarypart)(mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMissingDiagonal"
/*@
   MatMissingDiagonal - Determine if sparse matrix is missing a diagonal entry (or block entry for BAIJ matrices)

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  missing - is any diagonal missing
-  dd - first diagonal entry that is missing (optional)

   Level: advanced


.seealso: MatRealPart()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMissingDiagonal(Mat mat,PetscTruth *missing,PetscInt *dd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->missingdiagonal) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->missingdiagonal)(mat,missing,dd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow"
/*@C
   MatGetRow - Gets a row of a matrix.  You MUST call MatRestoreRow()
   for each row that you get to ensure that your application does
   not bleed memory.

   Not Collective

   Input Parameters:
+  mat - the matrix
-  row - the row to get

   Output Parameters:
+  ncols -  if not NULL, the number of nonzeros in the row
.  cols - if not NULL, the column numbers
-  vals - if not NULL, the values

   Notes:
   This routine is provided for people who need to have direct access
   to the structure of a matrix.  We hope that we provide enough
   high-level matrix routines that few users will need it. 

   MatGetRow() always returns 0-based column indices, regardless of
   whether the internal representation is 0-based (default) or 1-based.

   For better efficiency, set cols and/or vals to PETSC_NULL if you do
   not wish to extract these quantities.

   The user can only examine the values extracted with MatGetRow();
   the values cannot be altered.  To change the matrix entries, one
   must use MatSetValues().

   You can only have one call to MatGetRow() outstanding for a particular
   matrix at a time, per processor. MatGetRow() can only obtain rows
   associated with the given processor, it cannot get rows from the 
   other processors; for that we suggest using MatGetSubMatrices(), then
   MatGetRow() on the submatrix. The row indix passed to MatGetRows() 
   is in the global number of rows.

   Fortran Notes:
   The calling sequence from Fortran is 
.vb
   MatGetRow(matrix,row,ncols,cols,values,ierr)
         Mat     matrix (input)
         integer row    (input)
         integer ncols  (output)
         integer cols(maxcols) (output)
         double precision (or double complex) values(maxcols) output
.ve
   where maxcols >= maximum nonzeros in any row of the matrix.


   Caution:
   Do not try to change the contents of the output arrays (cols and vals).
   In some cases, this may corrupt the matrix.

   Level: advanced

   Concepts: matrices^row access

.seealso: MatRestoreRow(), MatSetValues(), MatGetValues(), MatGetSubMatrices(), MatGetDiagonal()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRow(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt *cols[],const PetscScalar *vals[])
{
  PetscErrorCode ierr;
  PetscInt       incols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->getrow) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_GetRow,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->getrow)(mat,row,&incols,(PetscInt **)cols,(PetscScalar **)vals);CHKERRQ(ierr);
  if (ncols) *ncols = incols;
  ierr = PetscLogEventEnd(MAT_GetRow,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatConjugate"
/*@
   MatConjugate - replaces the matrix values with their complex conjugates

   Collective on Mat

   Input Parameters:
.  mat - the matrix

   Level: advanced

.seealso:  VecConjugate()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatConjugate(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->conjugate) SETERRQ(PETSC_ERR_SUP,"Not provided for this matrix format, send email to petsc-maint@mcs.anl.gov");
  ierr = (*mat->ops->conjugate)(mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow"
/*@C  
   MatRestoreRow - Frees any temporary space allocated by MatGetRow().

   Not Collective

   Input Parameters:
+  mat - the matrix
.  row - the row to get
.  ncols, cols - the number of nonzeros and their columns
-  vals - if nonzero the column values

   Notes: 
   This routine should be called after you have finished examining the entries.

   Fortran Notes:
   The calling sequence from Fortran is 
.vb
   MatRestoreRow(matrix,row,ncols,cols,values,ierr)
      Mat     matrix (input)
      integer row    (input)
      integer ncols  (output)
      integer cols(maxcols) (output)
      double precision (or double complex) values(maxcols) output
.ve
   Where maxcols >= maximum nonzeros in any row of the matrix. 

   In Fortran MatRestoreRow() MUST be called after MatGetRow()
   before another call to MatGetRow() can be made.

   Level: advanced

.seealso:  MatGetRow()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreRow(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt *cols[],const PetscScalar *vals[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidIntPointer(ncols,3);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->restorerow) PetscFunctionReturn(0);
  ierr = (*mat->ops->restorerow)(mat,row,ncols,(PetscInt **)cols,(PetscScalar **)vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowUpperTriangular"
/*@
   MatGetRowUpperTriangular - Sets a flag to enable calls to MatGetRow() for matrix in MATSBAIJ format.  
   You should call MatRestoreRowUpperTriangular() after calling MatGetRow/MatRestoreRow() to disable the flag. 

   Not Collective

   Input Parameters:
+  mat - the matrix

   Notes:
   The flag is to ensure that users are aware of MatGetRow() only provides the upper trianglular part of the row for the matrices in MATSBAIJ format.

   Level: advanced

   Concepts: matrices^row access

.seealso: MatRestoreRowRowUpperTriangular()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowUpperTriangular(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->getrowuppertriangular) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = (*mat->ops->getrowuppertriangular)(mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowUpperTriangular"
/*@
   MatRestoreRowUpperTriangular - Disable calls to MatGetRow() for matrix in MATSBAIJ format.  

   Not Collective

   Input Parameters:
+  mat - the matrix

   Notes: 
   This routine should be called after you have finished MatGetRow/MatRestoreRow().


   Level: advanced

.seealso:  MatGetRowUpperTriangular()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreRowUpperTriangular(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->restorerowuppertriangular) PetscFunctionReturn(0);
  ierr = (*mat->ops->restorerowuppertriangular)(mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOptionsPrefix"
/*@C
   MatSetOptionsPrefix - Sets the prefix used for searching for all 
   Mat options in the database.

   Collective on Mat

   Input Parameter:
+  A - the Mat context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: Mat, set, options, prefix, database

.seealso: MatSetFromOptions()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetOptionsPrefix(Mat A,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)A,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAppendOptionsPrefix"
/*@C
   MatAppendOptionsPrefix - Appends to the prefix used for searching for all 
   Mat options in the database.

   Collective on Mat

   Input Parameters:
+  A - the Mat context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: Mat, append, options, prefix, database

.seealso: MatGetOptionsPrefix()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatAppendOptionsPrefix(Mat A,const char prefix[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)A,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetOptionsPrefix"
/*@C
   MatGetOptionsPrefix - Sets the prefix used for searching for all 
   Mat options in the database.

   Not Collective

   Input Parameter:
.  A - the Mat context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: Mat, get, options, prefix, database

.seealso: MatAppendOptionsPrefix()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetOptionsPrefix(Mat A,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)A,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUp"
/*@
   MatSetUp - Sets up the internal matrix data structures for the later use.

   Collective on Mat

   Input Parameters:
.  A - the Mat context

   Notes:
   For basic use of the Mat classes the user need not explicitly call
   MatSetUp(), since these actions will happen automatically.

   Level: advanced

.keywords: Mat, setup

.seealso: MatCreate(), MatDestroy()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetUp(Mat A)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  if (!((PetscObject)A)->type_name) {
    ierr = MPI_Comm_size(((PetscObject)A)->comm, &size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatSetType(A, MATSEQAIJ);CHKERRQ(ierr);
    } else {
      ierr = MatSetType(A, MATMPIAIJ);CHKERRQ(ierr);
    }
  }
  ierr = MatSetUpPreallocation(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView"
/*@C
   MatView - Visualizes a matrix object.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  viewer - visualization context

  Notes:
  The available visualization contexts include
+    PETSC_VIEWER_STDOUT_SELF - standard output (default)
.    PETSC_VIEWER_STDOUT_WORLD - synchronized standard
        output where only the first processor opens
        the file.  All other processors send their 
        data to the first processor to print. 
-     PETSC_VIEWER_DRAW_WORLD - graphical display of nonzero structure

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs matrix to a specified file
.    PetscViewerBinaryOpen() - Outputs matrix in binary to a
         specified file; corresponding input uses MatLoad()
.    PetscViewerDrawOpen() - Outputs nonzero matrix structure to 
         an X window display
-    PetscViewerSocketOpen() - Outputs matrix to Socket viewer.
         Currently only the sequential dense and AIJ
         matrix types support the Socket viewer.

   The user can call PetscViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_DEFAULT - default, prints matrix contents
.    PETSC_VIEWER_ASCII_MATLAB - prints matrix contents in Matlab format
.    PETSC_VIEWER_ASCII_DENSE - prints entire matrix including zeros
.    PETSC_VIEWER_ASCII_COMMON - prints matrix contents, using a sparse 
         format common among all matrix types
.    PETSC_VIEWER_ASCII_IMPL - prints matrix contents, using an implementation-specific 
         format (which is in many cases the same as the default)
.    PETSC_VIEWER_ASCII_INFO - prints basic information about the matrix
         size and structure (not the matrix entries)
.    PETSC_VIEWER_ASCII_INFO_DETAIL - prints more detailed information about
         the matrix structure

   Options Database Keys:
+  -mat_view_info - Prints info on matrix at conclusion of MatEndAssembly()
.  -mat_view_info_detailed - Prints more detailed info
.  -mat_view - Prints matrix in ASCII format
.  -mat_view_matlab - Prints matrix in Matlab format
.  -mat_view_draw - PetscDraws nonzero structure of matrix, using MatView() and PetscDrawOpenX().
.  -display <name> - Sets display name (default is host)
.  -draw_pause <sec> - Sets number of seconds to pause after display
.  -mat_view_socket - Sends matrix to socket, can be accessed from Matlab (see users manual)
.  -viewer_socket_machine <machine>
.  -viewer_socket_port <port>
.  -mat_view_binary - save matrix to file in binary format
-  -viewer_binary_filename <name>
   Level: beginner

   Notes: see the manual page for MatLoad() for the exact format of the binary file when the binary
      viewer is used.

      See bin/matlab/PetscBinaryRead.m for a Matlab code that can read in the binary file when the binary
      viewer is used.

   Concepts: matrices^viewing
   Concepts: matrices^plotting
   Concepts: matrices^printing

.seealso: PetscViewerSetFormat(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), 
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(), MatLoad()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatView(Mat mat,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          rows,cols;
  PetscTruth        iascii;
  const MatType     cstr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)mat)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(mat,1,viewer,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ORDER,"Must call MatAssemblyBegin/End() before viewing matrix");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_View,mat,viewer,0,0);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);  
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (((PetscObject)mat)->prefix) {
        ierr = PetscViewerASCIIPrintf(viewer,"Matrix Object:(%s)\n",((PetscObject)mat)->prefix);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Matrix Object:\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = MatGetType(mat,&cstr);CHKERRQ(ierr);
      ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"type=%s, rows=%D, cols=%D\n",cstr,rows,cols);CHKERRQ(ierr);
      if (mat->factor) {
        const MatSolverPackage solver;
        ierr = MatFactorGetSolverPackage(mat,&solver);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"package used to perform factorization: %s\n",solver);CHKERRQ(ierr);
      }
      if (mat->ops->getinfo) {
        MatInfo info;
        ierr = MatGetInfo(mat,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"total: nonzeros=%D, allocated nonzeros=%D\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);
      }
    }
  }
  if (mat->ops->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*mat->ops->view)(mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (!iascii) {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)viewer)->type_name);
  }
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);  
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(MAT_View,mat,viewer,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScaleSystem"
/*@
   MatScaleSystem - Scale a vector solution and right hand side to 
   match the scaling of a scaled matrix.
  
   Collective on Mat

   Input Parameter:
+  mat - the matrix
.  b - right hand side vector (or PETSC_NULL)
-  x - solution vector (or PETSC_NULL)


   Notes: 
   For AIJ, and BAIJ matrix formats, the matrices are not 
   internally scaled, so this does nothing. 

   The KSP methods automatically call this routine when required
   (via PCPreSolve()) so it is rarely used directly.

   Level: Developer            

   Concepts: matrices^scaling

.seealso: MatUseScaledForm(), MatUnScaleSystem()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatScaleSystem(Mat mat,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (x) {PetscValidHeaderSpecific(x,VEC_COOKIE,3);PetscCheckSameComm(mat,1,x,3);}
  if (b) {PetscValidHeaderSpecific(b,VEC_COOKIE,2);PetscCheckSameComm(mat,1,b,2);}

  if (mat->ops->scalesystem) {
    ierr = (*mat->ops->scalesystem)(mat,b,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUnScaleSystem"
/*@
   MatUnScaleSystem - Unscales a vector solution and right hand side to 
   match the original scaling of a scaled matrix.
  
   Collective on Mat

   Input Parameter:
+  mat - the matrix
.  b - right hand side vector (or PETSC_NULL)
-  x - solution vector (or PETSC_NULL)


   Notes: 
   For AIJ and BAIJ matrix formats, the matrices are not 
   internally scaled, so this does nothing. 

   The KSP methods automatically call this routine when required
   (via PCPreSolve()) so it is rarely used directly.

   Level: Developer            

.seealso: MatUseScaledForm(), MatScaleSystem()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatUnScaleSystem(Mat mat,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (x) {PetscValidHeaderSpecific(x,VEC_COOKIE,3);PetscCheckSameComm(mat,1,x,3);}
  if (b) {PetscValidHeaderSpecific(b,VEC_COOKIE,2);PetscCheckSameComm(mat,1,b,2);}
  if (mat->ops->unscalesystem) {
    ierr = (*mat->ops->unscalesystem)(mat,b,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseScaledForm"
/*@
   MatUseScaledForm - For matrix storage formats that scale the 
   matrix indicates matrix operations (MatMult() etc) are 
   applied using the scaled matrix.
  
   Collective on Mat

   Input Parameter:
+  mat - the matrix
-  scaled - PETSC_TRUE for applying the scaled, PETSC_FALSE for 
            applying the original matrix

   Notes: 
   For scaled matrix formats, applying the original, unscaled matrix
   will be slightly more expensive

   Level: Developer            

.seealso: MatScaleSystem(), MatUnScaleSystem()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatUseScaledForm(Mat mat,PetscTruth scaled)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->ops->usescaledform) {
    ierr = (*mat->ops->usescaledform)(mat,scaled);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy"
/*@
   MatDestroy - Frees space taken by a matrix.
  
   Collective on Mat

   Input Parameter:
.  A - the matrix

   Level: beginner

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy(Mat A)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  if (--((PetscObject)A)->refct > 0) PetscFunctionReturn(0);
  ierr = MatPreallocated(A);CHKERRQ(ierr);
  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(A);CHKERRQ(ierr);
  if (A->ops->destroy) {
    ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  }
  if (A->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->mapping);CHKERRQ(ierr);
  }
  if (A->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->bmapping);CHKERRQ(ierr);
  }

  if (A->spptr){ierr = PetscFree(A->spptr);CHKERRQ(ierr);}
  ierr = PetscLayoutDestroy(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(A->cmap);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatValid"
/*@
   MatValid - Checks whether a matrix object is valid.

   Collective on Mat

   Input Parameter:
.  m - the matrix to check 

   Output Parameter:
   flg - flag indicating matrix status, either
   PETSC_TRUE if matrix is valid, or PETSC_FALSE otherwise.

   Level: developer

   Concepts: matrices^validity
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatValid(Mat m,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidIntPointer(flg,1);
  if (!m)                                          *flg = PETSC_FALSE;
  else if (((PetscObject)m)->cookie != MAT_COOKIE) *flg = PETSC_FALSE;
  else                                             *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues"
/*@ 
   MatSetValues - Inserts or adds a block of values into a matrix.
   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd() 
   MUST be called after all calls to MatSetValues() have been completed.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array of values
.  m, idxm - the number of rows and their global indices 
.  n, idxn - the number of columns and their global indices
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented. See MatSetOption() for other options.

   Calls to MatSetValues() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   MatSetValues() uses 0-based row and column numbers in Fortran 
   as well as in C.

   Negative indices may be passed in idxm and idxn, these rows and columns are 
   simply ignored. This allows easily inserting element stiffness matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the matrix.

   Efficiency Alert:
   The routine MatSetValuesBlocked() may offer much better efficiency
   for users of block sparse formats (MATSEQBAIJ and MATMPIBAIJ).

   Level: beginner

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal(),
          InsertMode, INSERT_VALUES, ADD_VALUES
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  if (v) PetscValidDoublePointer(v,6);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_DEBUG)
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (!mat->ops->setvalues) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesRowLocal"
/*@ 
   MatSetValuesRowLocal - Inserts a row (block row for BAIJ matrices) of nonzero
        values into a matrix

   Not Collective

   Input Parameters:
+  mat - the matrix
.  row - the (block) row to set
-  v - a logically two-dimensional array of values

   Notes:
   By the values, v, are column-oriented (for the block version) and sorted

   All the nonzeros in the row must be provided

   The matrix must have previously had its column indices set

   The row must belong to this process

   Level: intermediate

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal(),
          InsertMode, INSERT_VALUES, ADD_VALUES, MatSetValues(), MatSetValuesRow(), MatSetLocalToGlobalMapping()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesRowLocal(Mat mat,PetscInt row,const PetscScalar v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidScalarPointer(v,2);
  ierr = MatSetValuesRow(mat, mat->mapping->indices[row],v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesRow"
/*@ 
   MatSetValuesRow - Inserts a row (block row for BAIJ matrices) of nonzero
        values into a matrix

   Not Collective

   Input Parameters:
+  mat - the matrix
.  row - the (block) row to set
-  v - a logically two-dimensional array of values

   Notes:
   The values, v, are column-oriented for the block version.

   All the nonzeros in the row must be provided

   THE MATRIX MUSAT HAVE PREVIOUSLY HAD ITS COLUMN INDICES SET. IT IS RARE THAT THIS ROUTINE IS USED, usually MatSetValues() is used.

   The row must belong to this process

   Level: advanced

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal(),
          InsertMode, INSERT_VALUES, ADD_VALUES, MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesRow(Mat mat,PetscInt row,const PetscScalar v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidScalarPointer(v,2);
#if defined(PETSC_USE_DEBUG)
  if (mat->insertmode == ADD_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add and insert values");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
#endif
  mat->insertmode = INSERT_VALUES;

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (!mat->ops->setvaluesrow) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->setvaluesrow)(mat,row,v);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesStencil"
/*@
   MatSetValuesStencil - Inserts or adds a block of values into a matrix.
     Using structured grid indexing

   Not Collective

   Input Parameters:
+  mat - the matrix
.  m - number of rows being entered
.  idxm - grid coordinates (and component number when dof > 1) for matrix rows being entered
.  n - number of columns being entered
.  idxn - grid coordinates (and component number when dof > 1) for matrix columns being entered 
.  v - a logically two-dimensional array of values
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented.  See MatSetOption() for other options.

   Calls to MatSetValuesStencil() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   The grid coordinates are across the entire grid, not just the local portion

   MatSetValuesStencil() uses 0-based row and column numbers in Fortran 
   as well as in C.

   For setting/accessing vector values via array coordinates you can use the DAVecGetArray() routine

   In order to use this routine you must either obtain the matrix with DAGetMatrix()
   or call MatSetLocalToGlobalMapping() and MatSetStencil() first.

   The columns and rows in the stencil passed in MUST be contained within the 
   ghost region of the given process as set with DACreateXXX() or MatSetStencil(). For example,
   if you create a DA with an overlap of one grid level and on a particular process its first
   local nonghost x logical coordinate is 6 (so its first ghost x logical coordinate is 5) the
   first i index you can use in your column and row indices in MatSetStencil() is 5.

   In Fortran idxm and idxn should be declared as
$     MatStencil idxm(4,m),idxn(4,n)
   and the values inserted using
$    idxm(MatStencil_i,1) = i
$    idxm(MatStencil_j,1) = j
$    idxm(MatStencil_k,1) = k
$    idxm(MatStencil_c,1) = c
   etc

   For periodic boundary conditions use negative indices for values to the left (below 0; that are to be 
   obtained by wrapping values from right edge). For values to the right of the last entry using that index plus one
   etc to obtain values that obtained by wrapping the values from the left edge. This does not work for the DA_NONPERIODIC
   wrap.

   For indices that don't mean anything for your case (like the k index when working in 2d) or the c index when you have
   a single value per point) you can skip filling those indices.

   Inspired by the structured grid interface to the HYPRE package
   (http://www.llnl.gov/CASC/hypre)

   Efficiency Alert:
   The routine MatSetValuesBlockedStencil() may offer much better efficiency
   for users of block sparse formats (MATSEQBAIJ and MATMPIBAIJ).

   Level: beginner

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal()
          MatSetValues(), MatSetValuesBlockedStencil(), MatSetStencil(), DAGetMatrix(), DAVecGetArray(), MatStencil
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesStencil(Mat mat,PetscInt m,const MatStencil idxm[],PetscInt n,const MatStencil idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt       j,i,jdxm[128],jdxn[256],dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
  PetscInt       *starts = mat->stencil.starts,*dxm = (PetscInt*)idxm,*dxn = (PetscInt*)idxn,sdim = dim - (1 - (PetscInt)mat->stencil.noc);

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  PetscValidScalarPointer(v,6);

  if (m > 128) SETERRQ1(PETSC_ERR_SUP,"Can only set 128 rows at a time; trying to set %D",m);
  if (n > 256) SETERRQ1(PETSC_ERR_SUP,"Can only set 256 columns at a time; trying to set %D",n);

  for (i=0; i<m; i++) {
    for (j=0; j<3-sdim; j++) dxm++;  
    tmp = *dxm++ - starts[0];
    for (j=0; j<dim-1; j++) {
      if ((*dxm++ - starts[j+1]) < 0 || tmp < 0) tmp = PETSC_MIN_INT;
      else                                       tmp = tmp*dims[j] + *(dxm-1) - starts[j+1];
    }
    if (mat->stencil.noc) dxm++;
    jdxm[i] = tmp;
  }
  for (i=0; i<n; i++) {
    for (j=0; j<3-sdim; j++) dxn++;  
    tmp = *dxn++ - starts[0];
    for (j=0; j<dim-1; j++) {
      if ((*dxn++ - starts[j+1]) < 0 || tmp < 0) tmp = PETSC_MIN_INT;
      else                                       tmp = tmp*dims[j] + *(dxn-1) - starts[j+1];
    }
    if (mat->stencil.noc) dxn++;
    jdxn[i] = tmp;
  }
  ierr = MatSetValuesLocal(mat,m,jdxm,n,jdxn,v,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlockedStencil"
/*@C 
   MatSetValuesBlockedStencil - Inserts or adds a block of values into a matrix.
     Using structured grid indexing

   Not Collective

   Input Parameters:
+  mat - the matrix
.  m - number of rows being entered
.  idxm - grid coordinates for matrix rows being entered
.  n - number of columns being entered
.  idxn - grid coordinates for matrix columns being entered 
.  v - a logically two-dimensional array of values
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented and unsorted.
   See MatSetOption() for other options.

   Calls to MatSetValuesBlockedStencil() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   The grid coordinates are across the entire grid, not just the local portion

   MatSetValuesBlockedStencil() uses 0-based row and column numbers in Fortran 
   as well as in C.

   For setting/accessing vector values via array coordinates you can use the DAVecGetArray() routine

   In order to use this routine you must either obtain the matrix with DAGetMatrix()
   or call MatSetBlockSize(), MatSetLocalToGlobalMapping() and MatSetStencil() first.

   The columns and rows in the stencil passed in MUST be contained within the 
   ghost region of the given process as set with DACreateXXX() or MatSetStencil(). For example,
   if you create a DA with an overlap of one grid level and on a particular process its first
   local nonghost x logical coordinate is 6 (so its first ghost x logical coordinate is 5) the
   first i index you can use in your column and row indices in MatSetStencil() is 5.

   In Fortran idxm and idxn should be declared as
$     MatStencil idxm(4,m),idxn(4,n)
   and the values inserted using
$    idxm(MatStencil_i,1) = i
$    idxm(MatStencil_j,1) = j
$    idxm(MatStencil_k,1) = k
   etc

   Negative indices may be passed in idxm and idxn, these rows and columns are 
   simply ignored. This allows easily inserting element stiffness matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the matrix.

   Inspired by the structured grid interface to the HYPRE package
   (http://www.llnl.gov/CASC/hypre)

   Level: beginner

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal()
          MatSetValues(), MatSetValuesStencil(), MatSetStencil(), DAGetMatrix(), DAVecGetArray(), MatStencil,
          MatSetBlockSize(), MatSetLocalToGlobalMapping()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesBlockedStencil(Mat mat,PetscInt m,const MatStencil idxm[],PetscInt n,const MatStencil idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt       j,i,jdxm[128],jdxn[256],dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
  PetscInt       *starts = mat->stencil.starts,*dxm = (PetscInt*)idxm,*dxn = (PetscInt*)idxn,sdim = dim - (1 - (PetscInt)mat->stencil.noc);

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  PetscValidScalarPointer(v,6);

  if (m > 128) SETERRQ1(PETSC_ERR_SUP,"Can only set 128 rows at a time; trying to set %D",m);
  if (n > 128) SETERRQ1(PETSC_ERR_SUP,"Can only set 256 columns at a time; trying to set %D",n);

  for (i=0; i<m; i++) {
    for (j=0; j<3-sdim; j++) dxm++;  
    tmp = *dxm++ - starts[0];
    for (j=0; j<sdim-1; j++) {
      if ((*dxm++ - starts[j+1]) < 0 || tmp < 0) tmp = PETSC_MIN_INT;
      else                                      tmp = tmp*dims[j] + *(dxm-1) - starts[j+1];
    }
    dxm++;
    jdxm[i] = tmp;
  }
  for (i=0; i<n; i++) {
    for (j=0; j<3-sdim; j++) dxn++;  
    tmp = *dxn++ - starts[0];
    for (j=0; j<sdim-1; j++) {
      if ((*dxn++ - starts[j+1]) < 0 || tmp < 0) tmp = PETSC_MIN_INT;
      else                                       tmp = tmp*dims[j] + *(dxn-1) - starts[j+1];
    }
    dxn++;
    jdxn[i] = tmp;
  }
  ierr = MatSetValuesBlockedLocal(mat,m,jdxm,n,jdxn,v,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetStencil"
/*@ 
   MatSetStencil - Sets the grid information for setting values into a matrix via
        MatSetValuesStencil()

   Not Collective

   Input Parameters:
+  mat - the matrix
.  dim - dimension of the grid 1, 2, or 3
.  dims - number of grid points in x, y, and z direction, including ghost points on your processor
.  starts - starting point of ghost nodes on your processor in x, y, and z direction 
-  dof - number of degrees of freedom per node


   Inspired by the structured grid interface to the HYPRE package
   (www.llnl.gov/CASC/hyper)

   For matrices generated with DAGetMatrix() this routine is automatically called and so not needed by the
   user.
   
   Level: beginner

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal()
          MatSetValues(), MatSetValuesBlockedStencil(), MatSetValuesStencil()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetStencil(Mat mat,PetscInt dim,const PetscInt dims[],const PetscInt starts[],PetscInt dof)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidIntPointer(dims,3);
  PetscValidIntPointer(starts,4);

  mat->stencil.dim = dim + (dof > 1);
  for (i=0; i<dim; i++) {
    mat->stencil.dims[i]   = dims[dim-i-1];      /* copy the values in backwards */
    mat->stencil.starts[i] = starts[dim-i-1];
  }
  mat->stencil.dims[dim]   = dof;
  mat->stencil.starts[dim] = 0;
  mat->stencil.noc         = (PetscTruth)(dof == 1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked"
/*@ 
   MatSetValuesBlocked - Inserts or adds a block of values into a matrix.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array of values
.  m, idxm - the number of block rows and their global block indices 
.  n, idxn - the number of block columns and their global block indices
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   The m and n count the NUMBER of blocks in the row direction and column direction,
   NOT the total number of rows/columns; for example, if the block size is 2 and 
   you are passing in values for rows 2,3,4,5  then m would be 2 (not 4).
   The values in idxm would be 1 2; that is the first index for each block divided by 
   the block size.

   Note that you must call MatSetBlockSize() when constructing this matrix (after
   preallocating it).

   By default the values, v, are row-oriented, so the layout of 
   v is the same as for MatSetValues(). See MatSetOption() for other options.

   Calls to MatSetValuesBlocked() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   MatSetValuesBlocked() uses 0-based row and column numbers in Fortran 
   as well as in C.

   Negative indices may be passed in idxm and idxn, these rows and columns are 
   simply ignored. This allows easily inserting element stiffness matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the matrix.

   Each time an entry is set within a sparse matrix via MatSetValues(),
   internal searching must be done to determine where to place the the
   data in the matrix storage space.  By instead inserting blocks of 
   entries via MatSetValuesBlocked(), the overhead of matrix assembly is
   reduced.

   Example:
$   Suppose m=n=2 and block size(bs) = 2 The array is 
$
$   1  2  | 3  4
$   5  6  | 7  8
$   - - - | - - -
$   9  10 | 11 12
$   13 14 | 15 16
$
$   v[] should be passed in like
$   v[] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
$
$  If you are not using row oriented storage of v (that is you called MatSetOption(mat,MAT_ROW_ORIENTED,PETSC_FALSE)) then
$   v[] = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]

   Level: intermediate

   Concepts: matrices^putting entries in blocked

.seealso: MatSetBlockSize(), MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetValuesBlockedLocal()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesBlocked(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  PetscValidScalarPointer(v,6);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_DEBUG) 
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (mat->ops->setvaluesblocked) {
    ierr = (*mat->ops->setvaluesblocked)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  } else {
    PetscInt buf[4096],*ibufm=0,*ibufn=0;
    PetscInt i,j,*iidxm,*iidxn,bs=mat->rmap->bs;
    if ((m+n)*bs <= 4096) {
      iidxm = buf; iidxn = buf + m*bs;
    } else {
      ierr = PetscMalloc2(m*bs,PetscInt,&ibufm,n*bs,PetscInt,&ibufn);CHKERRQ(ierr);
      iidxm = ibufm; iidxn = ibufn;
    }
    for (i=0; i<m; i++) {
      for (j=0; j<bs; j++) {
	iidxm[i*bs+j] = bs*idxm[i] + j;
      }
    }
    for (i=0; i<n; i++) {
      for (j=0; j<bs; j++) {
	iidxn[i*bs+j] = bs*idxn[i] + j;
      }
    }
    ierr = MatSetValues(mat,bs*m,iidxm,bs*n,iidxn,v,addv);CHKERRQ(ierr);
    ierr = PetscFree2(ibufm,ibufn);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues"
/*@ 
   MatGetValues - Gets a block of values from a matrix.

   Not Collective; currently only returns a local block

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array for storing the values
.  m, idxm - the number of rows and their global indices 
-  n, idxn - the number of columns and their global indices

   Notes:
   The user must allocate space (m*n PetscScalars) for the values, v.
   The values, v, are then returned in a row-oriented format, 
   analogous to that used by default in MatSetValues().

   MatGetValues() uses 0-based row and column numbers in
   Fortran as well as in C.

   MatGetValues() requires that the matrix has been assembled
   with MatAssemblyBegin()/MatAssemblyEnd().  Thus, calls to
   MatSetValues() and MatGetValues() CANNOT be made in succession
   without intermediate matrix assembly.

   Negative row or column indices will be ignored and those locations in v[] will be 
   left unchanged.

   Level: advanced

   Concepts: matrices^accessing values

.seealso: MatGetRow(), MatGetSubMatrices(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  PetscValidScalarPointer(v,6);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->getvalues) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_GetValues,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->getvalues)(mat,m,idxm,n,idxn,v);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_GetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetLocalToGlobalMapping"
/*@
   MatSetLocalToGlobalMapping - Sets a local-to-global numbering for use by
   the routine MatSetValuesLocal() to allow users to insert matrix entries
   using a local (per-processor) numbering.

   Not Collective

   Input Parameters:
+  x - the matrix
-  mapping - mapping created with ISLocalToGlobalMappingCreate() 
             or ISLocalToGlobalMappingCreateIS()

   Level: intermediate

   Concepts: matrices^local to global mapping
   Concepts: local to global mapping^for matrices

.seealso:  MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetValuesLocal()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetLocalToGlobalMapping(Mat x,ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,MAT_COOKIE,1);
  PetscValidType(x,1);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,2);
  if (x->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for matrix");
  }
  ierr = MatPreallocated(x);CHKERRQ(ierr);

  if (x->ops->setlocaltoglobalmapping) {
    ierr = (*x->ops->setlocaltoglobalmapping)(x,mapping);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
    if (x->mapping) { ierr = ISLocalToGlobalMappingDestroy(x->mapping);CHKERRQ(ierr); }
    x->mapping = mapping;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetLocalToGlobalMappingBlock"
/*@
   MatSetLocalToGlobalMappingBlock - Sets a local-to-global numbering for use
   by the routine MatSetValuesBlockedLocal() to allow users to insert matrix
   entries using a local (per-processor) numbering.

   Not Collective

   Input Parameters:
+  x - the matrix
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or
             ISLocalToGlobalMappingCreateIS()

   Level: intermediate

   Concepts: matrices^local to global mapping blocked
   Concepts: local to global mapping^for matrices, blocked

.seealso:  MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetValuesBlockedLocal(),
           MatSetValuesBlocked(), MatSetValuesLocal()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetLocalToGlobalMappingBlock(Mat x,ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,MAT_COOKIE,1);
  PetscValidType(x,1);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,2);
  if (x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for matrix");
  }
  ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
  if (x->bmapping) { ierr = ISLocalToGlobalMappingDestroy(x->bmapping);CHKERRQ(ierr); }
  x->bmapping = mapping;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesLocal"
/*@
   MatSetValuesLocal - Inserts or adds values into certain locations of a matrix,
   using a local ordering of the nodes. 

   Not Collective

   Input Parameters:
+  x - the matrix
.  nrow, irow - number of rows and their local indices
.  ncol, icol - number of columns and their local indices
.  y -  a logically two-dimensional array of values
-  addv - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   Before calling MatSetValuesLocal(), the user must first set the
   local-to-global mapping by calling MatSetLocalToGlobalMapping().

   Calls to MatSetValuesLocal() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd() 
   MUST be called after all calls to MatSetValuesLocal() have been completed.

   Level: intermediate

   Concepts: matrices^putting entries in with local numbering

.seealso:  MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetLocalToGlobalMapping(),
           MatSetValueLocal()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv) 
{
  PetscErrorCode ierr;
  PetscInt       irowm[2048],icolm[2048];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!nrow || !ncol) PetscFunctionReturn(0); /* no values to insert */
  PetscValidIntPointer(irow,3);
  PetscValidIntPointer(icol,5);
  PetscValidScalarPointer(y,6);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_DEBUG) 
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
  if (!mat->ops->setvalueslocal && (nrow > 2048 || ncol > 2048)) {
    SETERRQ2(PETSC_ERR_SUP,"Number column/row indices must be <= 2048: are %D %D",nrow,ncol);
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (!mat->ops->setvalueslocal) {
    ierr = ISLocalToGlobalMappingApply(mat->mapping,nrow,irow,irowm);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(mat->mapping,ncol,icol,icolm);CHKERRQ(ierr); 
    ierr = (*mat->ops->setvalues)(mat,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
  } else {
    ierr = (*mat->ops->setvalueslocal)(mat,nrow,irow,ncol,icol,y,addv);CHKERRQ(ierr);
  }
  mat->same_nonzero = PETSC_FALSE;
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlockedLocal"
/*@
   MatSetValuesBlockedLocal - Inserts or adds values into certain locations of a matrix,
   using a local ordering of the nodes a block at a time. 

   Not Collective

   Input Parameters:
+  x - the matrix
.  nrow, irow - number of rows and their local indices
.  ncol, icol - number of columns and their local indices
.  y -  a logically two-dimensional array of values
-  addv - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   Before calling MatSetValuesBlockedLocal(), the user must first set the
   block size using MatSetBlockSize(), and the local-to-global mapping by
   calling MatSetLocalToGlobalMappingBlock(), where the mapping MUST be
   set for matrix blocks, not for matrix elements.

   Calls to MatSetValuesBlockedLocal() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd() 
   MUST be called after all calls to MatSetValuesBlockedLocal() have been completed.

   Level: intermediate

   Concepts: matrices^putting blocked values in with local numbering

.seealso:  MatSetBlockSize(), MatSetLocalToGlobalMappingBlock(), MatAssemblyBegin(), MatAssemblyEnd(),
           MatSetValuesLocal(), MatSetLocalToGlobalMappingBlock(), MatSetValuesBlocked()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesBlockedLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv) 
{
  PetscErrorCode ierr;
  PetscInt       irowm[2048],icolm[2048];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!nrow || !ncol) PetscFunctionReturn(0); /* no values to insert */
  PetscValidIntPointer(irow,3);
  PetscValidIntPointer(icol,5);
  PetscValidScalarPointer(y,6);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_DEBUG) 
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
  if (!mat->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Local to global never set with MatSetLocalToGlobalMappingBlock()");
  }
  if (nrow > 2048 || ncol > 2048) {
    SETERRQ2(PETSC_ERR_SUP,"Number column/row indices must be <= 2048: are %D %D",nrow,ncol);
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mat->bmapping,nrow,irow,irowm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mat->bmapping,ncol,icol,icolm);CHKERRQ(ierr);
  if (mat->ops->setvaluesblocked) {
    ierr = (*mat->ops->setvaluesblocked)(mat,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
  } else {
    PetscInt buf[4096],*ibufm=0,*ibufn=0;
    PetscInt i,j,*iirowm,*iicolm,bs=mat->rmap->bs;
    if ((nrow+ncol)*bs <= 4096) {
      iirowm = buf; iicolm = buf + nrow*bs;
    } else {
      ierr = PetscMalloc2(nrow*bs,PetscInt,&ibufm,ncol*bs,PetscInt,&ibufn);CHKERRQ(ierr);
      iirowm = ibufm; iicolm = ibufn;
    }
    for (i=0; i<nrow; i++) {
      for (j=0; j<bs; j++) {
	iirowm[i*bs+j] = bs*irowm[i] + j;
      }
    }
    for (i=0; i<ncol; i++) {
      for (j=0; j<bs; j++) {
	iicolm[i*bs+j] = bs*icolm[i] + j;
      }
    }
    ierr = MatSetValues(mat,bs*nrow,iirowm,bs*ncol,iicolm,y,addv);CHKERRQ(ierr);
    ierr = PetscFree2(ibufm,ibufn);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultDiagonalBlock"
/*@
   MatMultDiagonalBlock - Computes the matrix-vector product, y = Dx. Where D is defined by the inode or block structure of the diagonal

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multiplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMult(A,y,y).

   Level: developer

   Concepts: matrix-vector product

.seealso: MatMultTranspose(), MatMultAdd(), MatMultTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMultDiagonalBlock(Mat mat,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3); 

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (!mat->ops->multdiagonalblock) SETERRQ(PETSC_ERR_SUP,"This matrix type does not have a multiply defined");
  ierr = (*mat->ops->multdiagonalblock)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}   

/* --------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult"
/*@
   MatMult - Computes the matrix-vector product, y = Ax.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multiplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMult(A,y,y).

   Level: beginner

   Concepts: matrix-vector product

.seealso: MatMultTranspose(), MatMultAdd(), MatMultTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMult(Mat mat,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3); 

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
#ifndef PETSC_HAVE_CONSTRAINTS
  if (mat->cmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->cmap->N,x->map->N); 
  if (mat->rmap->N != y->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %D %D",mat->rmap->N,y->map->N); 
  if (mat->rmap->n != y->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: local dim %D %D",mat->rmap->n,y->map->n); 
#endif
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (mat->nullsp) {
    ierr = MatNullSpaceRemove(mat->nullsp,x,&x);CHKERRQ(ierr);
  }

  if (!mat->ops->mult) SETERRQ(PETSC_ERR_SUP,"This matrix type does not have a multiply defined");
  ierr = PetscLogEventBegin(MAT_Mult,mat,x,y,0);CHKERRQ(ierr);
  ierr = (*mat->ops->mult)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Mult,mat,x,y,0);CHKERRQ(ierr);

  if (mat->nullsp) {
    ierr = MatNullSpaceRemove(mat->nullsp,y,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}   

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose"
/*@
   MatMultTranspose - Computes matrix transpose times a vector.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMultTranspose(A,y,y).

   Level: beginner

   Concepts: matrix vector product^transpose

.seealso: MatMult(), MatMultAdd(), MatMultTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose(Mat mat,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
#ifndef PETSC_HAVE_CONSTRAINTS
  if (mat->rmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->rmap->N,x->map->N); 
  if (mat->cmap->N != y->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %D %D",mat->cmap->N,y->map->N);
#endif
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (!mat->ops->multtranspose) SETERRQ(PETSC_ERR_SUP,"This matrix type does not have a multiply tranpose defined");
  ierr = PetscLogEventBegin(MAT_MultTranspose,mat,x,y,0);CHKERRQ(ierr);
  ierr = (*mat->ops->multtranspose)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultTranspose,mat,x,y,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultHermitianTranspose"
/*@
   MatMultHermitianTranspose - Computes matrix Hermitian transpose times a vector.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMultHermitianTranspose(A,y,y).

   Level: beginner

   Concepts: matrix vector product^transpose

.seealso: MatMult(), MatMultAdd(), MatMultHermitianTransposeAdd(), MatMultTranspose()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMultHermitianTranspose(Mat mat,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
#ifndef PETSC_HAVE_CONSTRAINTS
  if (mat->rmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->rmap->N,x->map->N); 
  if (mat->cmap->N != y->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %D %D",mat->cmap->N,y->map->N);
#endif
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (!mat->ops->multhermitiantranspose) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_MultHermitianTranspose,mat,x,y,0);CHKERRQ(ierr);
  ierr = (*mat->ops->multhermitiantranspose)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultHermitianTranspose,mat,x,y,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}      

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd"
/*@
    MatMultAdd -  Computes v3 = v2 + A * v1.

    Collective on Mat and Vec

    Input Parameters:
+   mat - the matrix
-   v1, v2 - the vectors

    Output Parameters:
.   v3 - the result

    Notes:
    The vectors v1 and v3 cannot be the same.  I.e., one cannot
    call MatMultAdd(A,v1,v2,v1).

    Level: beginner

    Concepts: matrix vector product^addition

.seealso: MatMultTranspose(), MatMult(), MatMultTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMultAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v1,VEC_COOKIE,2);
  PetscValidHeaderSpecific(v2,VEC_COOKIE,3); 
  PetscValidHeaderSpecific(v3,VEC_COOKIE,4);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (mat->cmap->N != v1->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v1: global dim %D %D",mat->cmap->N,v1->map->N);
  /* if (mat->rmap->N != v2->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: global dim %D %D",mat->rmap->N,v2->map->N);
     if (mat->rmap->N != v3->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: global dim %D %D",mat->rmap->N,v3->map->N); */
  if (mat->rmap->n != v3->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: local dim %D %D",mat->rmap->n,v3->map->n); 
  if (mat->rmap->n != v2->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: local dim %D %D",mat->rmap->n,v2->map->n); 
  if (v1 == v3) SETERRQ(PETSC_ERR_ARG_IDN,"v1 and v3 must be different vectors");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_MultAdd,mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = (*mat->ops->multadd)(mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultAdd,mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}   

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd"
/*@
   MatMultTransposeAdd - Computes v3 = v2 + A' * v1.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  v1, v2 - the vectors

   Output Parameters:
.  v3 - the result

   Notes:
   The vectors v1 and v3 cannot be the same.  I.e., one cannot
   call MatMultTransposeAdd(A,v1,v2,v1).

   Level: beginner

   Concepts: matrix vector product^transpose and addition

.seealso: MatMultTranspose(), MatMultAdd(), MatMult()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTransposeAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v1,VEC_COOKIE,2);
  PetscValidHeaderSpecific(v2,VEC_COOKIE,3);
  PetscValidHeaderSpecific(v3,VEC_COOKIE,4);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->multtransposeadd) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (v1 == v3) SETERRQ(PETSC_ERR_ARG_IDN,"v1 and v3 must be different vectors");
  if (mat->rmap->N != v1->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v1: global dim %D %D",mat->rmap->N,v1->map->N);
  if (mat->cmap->N != v2->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: global dim %D %D",mat->cmap->N,v2->map->N);
  if (mat->cmap->N != v3->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: global dim %D %D",mat->cmap->N,v3->map->N);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_MultTransposeAdd,mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = (*mat->ops->multtransposeadd)(mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultTransposeAdd,mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultHermitianTransposeAdd"
/*@
   MatMultHermitianTransposeAdd - Computes v3 = v2 + A^H * v1.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  v1, v2 - the vectors

   Output Parameters:
.  v3 - the result

   Notes:
   The vectors v1 and v3 cannot be the same.  I.e., one cannot
   call MatMultHermitianTransposeAdd(A,v1,v2,v1).

   Level: beginner

   Concepts: matrix vector product^transpose and addition

.seealso: MatMultHermitianTranspose(), MatMultTranspose(), MatMultAdd(), MatMult()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMultHermitianTransposeAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v1,VEC_COOKIE,2);
  PetscValidHeaderSpecific(v2,VEC_COOKIE,3);
  PetscValidHeaderSpecific(v3,VEC_COOKIE,4);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->multhermitiantransposeadd) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (v1 == v3) SETERRQ(PETSC_ERR_ARG_IDN,"v1 and v3 must be different vectors");
  if (mat->rmap->N != v1->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v1: global dim %D %D",mat->rmap->N,v1->map->N);
  if (mat->cmap->N != v2->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: global dim %D %D",mat->cmap->N,v2->map->N);
  if (mat->cmap->N != v3->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: global dim %D %D",mat->cmap->N,v3->map->N);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_MultHermitianTransposeAdd,mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = (*mat->ops->multhermitiantransposeadd)(mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultHermitianTransposeAdd,mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultConstrained"
/*@
   MatMultConstrained - The inner multiplication routine for a
   constrained matrix P^T A P.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMult(A,y,y).

   Level: beginner

.keywords: matrix, multiply, matrix-vector product, constraint
.seealso: MatMult(), MatMultTranspose(), MatMultAdd(), MatMultTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMultConstrained(Mat mat,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3); 
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  if (mat->cmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->cmap->N,x->map->N); 
  if (mat->rmap->N != y->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %D %D",mat->rmap->N,y->map->N); 
  if (mat->rmap->n != y->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: local dim %D %D",mat->rmap->n,y->map->n); 

  ierr = PetscLogEventBegin(MAT_MultConstrained,mat,x,y,0);CHKERRQ(ierr);
  ierr = (*mat->ops->multconstrained)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultConstrained,mat,x,y,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}   

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeConstrained"
/*@
   MatMultTransposeConstrained - The inner multiplication routine for a
   constrained matrix P^T A^T P.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMult(A,y,y).

   Level: beginner

.keywords: matrix, multiply, matrix-vector product, constraint
.seealso: MatMult(), MatMultTranspose(), MatMultAdd(), MatMultTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTransposeConstrained(Mat mat,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3); 
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  if (mat->rmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->cmap->N,x->map->N); 
  if (mat->cmap->N != y->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %D %D",mat->rmap->N,y->map->N); 

  ierr = PetscLogEventBegin(MAT_MultConstrained,mat,x,y,0);CHKERRQ(ierr);
  ierr = (*mat->ops->multtransposeconstrained)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultConstrained,mat,x,y,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}   

#undef __FUNCT__  
#define __FUNCT__ "MatGetFactorType"
/*@C
   MatGetFactorType - gets the type of factorization it is

   Note Collective
   as the flag

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  t - the type, one of MAT_FACTOR_NONE, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ILU, MAT_FACTOR_ICC,MAT_FACTOR_ILUDT

    Level: intermediate

.seealso:    MatFactorType, MatGetFactor()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetFactorType(Mat mat,MatFactorType *t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  *t = mat->factor;
  PetscFunctionReturn(0);
}   

/* ------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo"
/*@C
   MatGetInfo - Returns information about matrix storage (number of
   nonzeros, memory, etc.).

   Collective on Mat if MAT_GLOBAL_MAX or MAT_GLOBAL_SUM is used
   as the flag

   Input Parameters:
.  mat - the matrix

   Output Parameters:
+  flag - flag indicating the type of parameters to be returned
   (MAT_LOCAL - local matrix, MAT_GLOBAL_MAX - maximum over all processors,
   MAT_GLOBAL_SUM - sum over all processors)
-  info - matrix information context

   Notes:
   The MatInfo context contains a variety of matrix data, including
   number of nonzeros allocated and used, number of mallocs during
   matrix assembly, etc.  Additional information for factored matrices
   is provided (such as the fill ratio, number of mallocs during
   factorization, etc.).  Much of this info is printed to PETSC_STDOUT
   when using the runtime options 
$       -info -mat_view_info

   Example for C/C++ Users:
   See the file ${PETSC_DIR}/include/petscmat.h for a complete list of
   data within the MatInfo context.  For example, 
.vb
      MatInfo info;
      Mat     A;
      double  mal, nz_a, nz_u;

      MatGetInfo(A,MAT_LOCAL,&info);
      mal  = info.mallocs;
      nz_a = info.nz_allocated;
.ve

   Example for Fortran Users:
   Fortran users should declare info as a double precision
   array of dimension MAT_INFO_SIZE, and then extract the parameters
   of interest.  See the file ${PETSC_DIR}/include/finclude/petscmat.h
   a complete list of parameter names.
.vb
      double  precision info(MAT_INFO_SIZE)
      double  precision mal, nz_a
      Mat     A
      integer ierr

      call MatGetInfo(A,MAT_LOCAL,info,ierr)
      mal = info(MAT_INFO_MALLOCS)
      nz_a = info(MAT_INFO_NZ_ALLOCATED)
.ve

    Level: intermediate

    Concepts: matrices^getting information on
    
    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatInfo]
 
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetInfo(Mat mat,MatInfoType flag,MatInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(info,3);
  if (!mat->ops->getinfo) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = (*mat->ops->getinfo)(mat,flag,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}   

/* ----------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactor"
/*@C
   MatLUFactor - Performs in-place LU factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row permutation
.  col - column permutation
-  info - options for factorization, includes 
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -info to determine an optimal value to use

   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   This changes the state of the matrix to a factored matrix; it cannot be used
   for example with MatSetValues() unless one first calls MatSetUnfactored().

   Level: developer

   Concepts: matrices^LU factorization

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor(),
          MatGetOrdering(), MatSetUnfactored(), MatFactorInfo

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatLUFactor(Mat mat,IS row,IS col,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (row) PetscValidHeaderSpecific(row,IS_COOKIE,2);
  if (col) PetscValidHeaderSpecific(col,IS_COOKIE,3);
  PetscValidPointer(info,4);
  PetscValidType(mat,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->lufactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_LUFactor,mat,row,col,0);CHKERRQ(ierr);
  ierr = (*mat->ops->lufactor)(mat,row,col,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_LUFactor,mat,row,col,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactor"
/*@C
   MatILUFactor - Performs in-place ILU factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row permutation
.  col - column permutation
-  info - structure containing 
$      levels - number of levels of fill.
$      expected fill - as ratio of original fill.
$      1 or 0 - indicating force fill on diagonal (improves robustness for matrices
                missing diagonal entries)

   Notes: 
   Probably really in-place only when level of fill is zero, otherwise allocates
   new space to store factored matrix and deletes previous memory.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^ILU factorization

.seealso: MatILUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor(), MatFactorInfo

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatILUFactor(Mat mat,IS row,IS col,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (row) PetscValidHeaderSpecific(row,IS_COOKIE,2);
  if (col) PetscValidHeaderSpecific(col,IS_COOKIE,3);
  PetscValidPointer(info,4);
  PetscValidType(mat,1);
  if (mat->rmap->N != mat->cmap->N) SETERRQ(PETSC_ERR_ARG_WRONG,"matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->ilufactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_ILUFactor,mat,row,col,0);CHKERRQ(ierr);
  ierr = (*mat->ops->ilufactor)(mat,row,col,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ILUFactor,mat,row,col,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic"
/*@C
   MatLUFactorSymbolic - Performs symbolic LU factorization of matrix.
   Call this routine before calling MatLUFactorNumeric().

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the matrix
.  row, col - row and column permutations
-  info - options for factorization, includes 
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -info to determine an optimal value to use


   Notes:
   See the users manual for additional information about
   choosing the fill factor for better efficiency.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^LU symbolic factorization

.seealso: MatLUFactor(), MatLUFactorNumeric(), MatCholeskyFactor(), MatFactorInfo

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatLUFactorSymbolic(Mat fact,Mat mat,IS row,IS col,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (row) PetscValidHeaderSpecific(row,IS_COOKIE,2);
  if (col) PetscValidHeaderSpecific(col,IS_COOKIE,3);
  PetscValidPointer(info,4);
  PetscValidType(mat,1);
  PetscValidPointer(fact,5);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!(fact)->ops->lufactorsymbolic) SETERRQ1(PETSC_ERR_SUP,"Matrix type %s  symbolic LU",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_LUFactorSymbolic,mat,row,col,0);CHKERRQ(ierr);
  ierr = (fact->ops->lufactorsymbolic)(fact,mat,row,col,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_LUFactorSymbolic,mat,row,col,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric"
/*@C
   MatLUFactorNumeric - Performs numeric LU factorization of a matrix.
   Call this routine after first calling MatLUFactorSymbolic().

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the matrix
-  info - options for factorization

   Notes:
   See MatLUFactor() for in-place factorization.  See 
   MatCholeskyFactorNumeric() for the symmetric, positive definite case.  

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^LU numeric factorization

.seealso: MatLUFactorSymbolic(), MatLUFactor(), MatCholeskyFactor()

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatLUFactorNumeric(Mat fact,Mat mat,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(fact,2);
  PetscValidHeaderSpecific(fact,MAT_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->rmap->N != (fact)->rmap->N || mat->cmap->N != (fact)->cmap->N) {
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat mat,Mat fact: global dimensions are different %D should = %D %D should = %D",mat->rmap->N,(fact)->rmap->N,mat->cmap->N,(fact)->cmap->N);
  }
  if (!(fact)->ops->lufactornumeric) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_LUFactorNumeric,mat,fact,0,0);CHKERRQ(ierr);
  ierr = (fact->ops->lufactornumeric)(fact,mat,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_LUFactorNumeric,mat,fact,0,0);CHKERRQ(ierr);

  ierr = MatView_Private(fact);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactor"
/*@C
   MatCholeskyFactor - Performs in-place Cholesky factorization of a
   symmetric matrix. 

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  perm - row and column permutations
-  f - expected fill as ratio of original fill

   Notes:
   See MatLUFactor() for the nonsymmetric case.  See also
   MatCholeskyFactorSymbolic(), and MatCholeskyFactorNumeric().

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^Cholesky factorization

.seealso: MatLUFactor(), MatCholeskyFactorSymbolic(), MatCholeskyFactorNumeric()
          MatGetOrdering()

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCholeskyFactor(Mat mat,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(perm,IS_COOKIE,2);
  PetscValidPointer(info,3);
  if (mat->rmap->N != mat->cmap->N) SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->choleskyfactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_CholeskyFactor,mat,perm,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->choleskyfactor)(mat,perm,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_CholeskyFactor,mat,perm,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic"
/*@C
   MatCholeskyFactorSymbolic - Performs symbolic Cholesky factorization
   of a symmetric matrix. 

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the matrix
.  perm - row and column permutations
-  info - options for factorization, includes 
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -info to determine an optimal value to use

   Notes:
   See MatLUFactorSymbolic() for the nonsymmetric case.  See also
   MatCholeskyFactor() and MatCholeskyFactorNumeric().

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^Cholesky symbolic factorization

.seealso: MatLUFactorSymbolic(), MatCholeskyFactor(), MatCholeskyFactorNumeric()
          MatGetOrdering()

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCholeskyFactorSymbolic(Mat fact,Mat mat,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (perm) PetscValidHeaderSpecific(perm,IS_COOKIE,2);
  PetscValidPointer(info,3);
  PetscValidPointer(fact,4);
  if (mat->rmap->N != mat->cmap->N) SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!(fact)->ops->choleskyfactorsymbolic) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_CholeskyFactorSymbolic,mat,perm,0,0);CHKERRQ(ierr);
  ierr = (fact->ops->choleskyfactorsymbolic)(fact,mat,perm,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_CholeskyFactorSymbolic,mat,perm,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric"
/*@C
   MatCholeskyFactorNumeric - Performs numeric Cholesky factorization
   of a symmetric matrix. Call this routine after first calling
   MatCholeskyFactorSymbolic().

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the initial matrix
.  info - options for factorization
-  fact - the symbolic factor of mat


   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^Cholesky numeric factorization

.seealso: MatCholeskyFactorSymbolic(), MatCholeskyFactor(), MatLUFactorNumeric()

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCholeskyFactorNumeric(Mat fact,Mat mat,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(fact,2);
  PetscValidHeaderSpecific(fact,MAT_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!(fact)->ops->choleskyfactornumeric) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (mat->rmap->N != (fact)->rmap->N || mat->cmap->N != (fact)->cmap->N) {
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat mat,Mat fact: global dim %D should = %D %D should = %D",mat->rmap->N,(fact)->rmap->N,mat->cmap->N,(fact)->cmap->N);
  }
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_CholeskyFactorNumeric,mat,fact,0,0);CHKERRQ(ierr);
  ierr = (fact->ops->choleskyfactornumeric)(fact,mat,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_CholeskyFactorNumeric,mat,fact,0,0);CHKERRQ(ierr);

  ierr = MatView_Private(fact);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatSolve"
/*@
   MatSolve - Solves A x = b, given a factored matrix.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolve(A,x,x).

   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolveAdd(), MatSolveTranspose(), MatSolveTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSolve(Mat mat,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,3);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (mat->cmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->cmap->N,x->map->N);
  if (mat->rmap->N != b->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %D %D",mat->rmap->N,b->map->N);
  if (mat->rmap->n != b->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %D %D",mat->rmap->n,b->map->n); 
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);
  if (!mat->ops->solve) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_Solve,mat,b,x,0);CHKERRQ(ierr);
  ierr = (*mat->ops->solve)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Solve,mat,b,x,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatSolve_Basic"
PetscErrorCode PETSCMAT_DLLEXPORT MatMatSolve_Basic(Mat A,Mat B,Mat X)
{
  PetscErrorCode ierr;
  Vec            b,x;
  PetscInt       m,N,i;
  PetscScalar    *bb,*xx;

  PetscFunctionBegin;
  ierr = MatGetArray(B,&bb);CHKERRQ(ierr); 
  ierr = MatGetArray(X,&xx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&m,PETSC_NULL);CHKERRQ(ierr);  /* number local rows */
  ierr = MatGetSize(B,PETSC_NULL,&N);CHKERRQ(ierr);       /* total columns in dense matrix */
  ierr = MatGetVecs(A,&x,&b);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    ierr = VecPlaceArray(b,bb + i*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(x,xx + i*m);CHKERRQ(ierr);
    ierr = MatSolve(A,b,x);CHKERRQ(ierr);
    ierr = VecResetArray(x);CHKERRQ(ierr);
    ierr = VecResetArray(b);CHKERRQ(ierr);
  }
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = MatRestoreArray(B,&bb);CHKERRQ(ierr);
  ierr = MatRestoreArray(X,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatSolve"
/*@
   MatMatSolve - Solves A X = B, given a factored matrix.

   Collective on Mat 

   Input Parameters:
+  mat - the factored matrix
-  B - the right-hand-side matrix  (dense matrix)

   Output Parameter:
.  X - the result matrix (dense matrix)

   Notes:
   The matrices b and x cannot be the same.  I.e., one cannot
   call MatMatSolve(A,x,x).

   Notes:
   Most users should usually employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate(). However KSP can only solve for one vector (column of X)
   at a time.

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatMatSolveAdd(), MatMatSolveTranspose(), MatMatSolveTransposeAdd(), MatLUFactor(), MatCholeskyFactor()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMatSolve(Mat A,Mat B,Mat X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidHeaderSpecific(B,MAT_COOKIE,2); 
  PetscValidHeaderSpecific(X,MAT_COOKIE,3);
  PetscCheckSameComm(A,1,B,2);
  PetscCheckSameComm(A,1,X,3);
  if (X == B) SETERRQ(PETSC_ERR_ARG_IDN,"X and B must be different matrices");
  if (!A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (A->cmap->N != X->rmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat A,Mat X: global dim %D %D",A->cmap->N,X->rmap->N);
  if (A->rmap->N != B->rmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim %D %D",A->rmap->N,B->rmap->N);
  if (A->rmap->n != B->rmap->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D",A->rmap->n,B->rmap->n); 
  if (!A->rmap->N && !A->cmap->N) PetscFunctionReturn(0);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_MatSolve,A,B,X,0);CHKERRQ(ierr);
  if (!A->ops->matsolve) {
    ierr = PetscInfo1(A,"Mat type %s using basic MatMatSolve",((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatMatSolve_Basic(A,B,X);CHKERRQ(ierr);
  } else {
    ierr = (*A->ops->matsolve)(A,B,X);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_MatSolve,A,B,X,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatForwardSolve"
/*@
   MatForwardSolve - Solves L x = b, given a factored matrix, A = LU, or
                            U^T*D^(1/2) x = b, given a factored symmetric matrix, A = U^T*D*U,

   Collective on Mat and Vec

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   MatSolve() should be used for most applications, as it performs
   a forward solve followed by a backward solve.

   The vectors b and x cannot be the same,  i.e., one cannot
   call MatForwardSolve(A,x,x).

   For matrix in seqsbaij format with block size larger than 1,
   the diagonal blocks are not implemented as D = D^(1/2) * D^(1/2) yet.
   MatForwardSolve() solves U^T*D y = b, and
   MatBackwardSolve() solves U x = y.
   Thus they do not provide a symmetric preconditioner.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^forward solves

.seealso: MatSolve(), MatBackwardSolve()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatForwardSolve(Mat mat,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,3);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (!mat->ops->forwardsolve) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (mat->cmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->cmap->N,x->map->N);
  if (mat->rmap->N != b->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %D %D",mat->rmap->N,b->map->N);
  if (mat->rmap->n != b->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %D %D",mat->rmap->n,b->map->n); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_ForwardSolve,mat,b,x,0);CHKERRQ(ierr);
  ierr = (*mat->ops->forwardsolve)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ForwardSolve,mat,b,x,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatBackwardSolve"
/*@
   MatBackwardSolve - Solves U x = b, given a factored matrix, A = LU.
                             D^(1/2) U x = b, given a factored symmetric matrix, A = U^T*D*U,

   Collective on Mat and Vec

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   MatSolve() should be used for most applications, as it performs
   a forward solve followed by a backward solve.

   The vectors b and x cannot be the same.  I.e., one cannot
   call MatBackwardSolve(A,x,x).

   For matrix in seqsbaij format with block size larger than 1,
   the diagonal blocks are not implemented as D = D^(1/2) * D^(1/2) yet.
   MatForwardSolve() solves U^T*D y = b, and
   MatBackwardSolve() solves U x = y.
   Thus they do not provide a symmetric preconditioner.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^backward solves

.seealso: MatSolve(), MatForwardSolve()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatBackwardSolve(Mat mat,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,3);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (!mat->ops->backwardsolve) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (mat->cmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->cmap->N,x->map->N);
  if (mat->rmap->N != b->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %D %D",mat->rmap->N,b->map->N);
  if (mat->rmap->n != b->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %D %D",mat->rmap->n,b->map->n); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_BackwardSolve,mat,b,x,0);CHKERRQ(ierr);
  ierr = (*mat->ops->backwardsolve)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_BackwardSolve,mat,b,x,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveAdd"
/*@
   MatSolveAdd - Computes x = y + inv(A)*b, given a factored matrix.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the factored matrix
.  b - the right-hand-side vector
-  y - the vector to be added to 

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolveAdd(A,x,y,x).

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolve(), MatSolveTranspose(), MatSolveTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSolveAdd(Mat mat,Vec b,Vec y,Vec x)
{
  PetscScalar    one = 1.0;
  Vec            tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidHeaderSpecific(b,VEC_COOKIE,3);  
  PetscValidHeaderSpecific(x,VEC_COOKIE,4);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,y,2);
  PetscCheckSameComm(mat,1,x,3);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (mat->cmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->cmap->N,x->map->N);
  if (mat->rmap->N != b->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %D %D",mat->rmap->N,b->map->N);
  if (mat->rmap->N != y->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %D %D",mat->rmap->N,y->map->N);
  if (mat->rmap->n != b->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %D %D",mat->rmap->n,b->map->n); 
  if (x->map->n != y->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Vec x,Vec y: local dim %D %D",x->map->n,y->map->n); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_SolveAdd,mat,b,x,y);CHKERRQ(ierr);
  if (mat->ops->solveadd)  {
    ierr = (*mat->ops->solveadd)(mat,b,y,x);CHKERRQ(ierr);
  } else {
    /* do the solve then the add manually */
    if (x != y) {
      ierr = MatSolve(mat,b,x);CHKERRQ(ierr);
      ierr = VecAXPY(x,one,y);CHKERRQ(ierr);
    } else {
      ierr = VecDuplicate(x,&tmp);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(mat,tmp);CHKERRQ(ierr);
      ierr = VecCopy(x,tmp);CHKERRQ(ierr);
      ierr = MatSolve(mat,b,x);CHKERRQ(ierr);
      ierr = VecAXPY(x,one,tmp);CHKERRQ(ierr);
      ierr = VecDestroy(tmp);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(MAT_SolveAdd,mat,b,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTranspose"
/*@
   MatSolveTranspose - Solves A' x = b, given a factored matrix.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolveTranspose(A,x,x).

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolve(), MatSolveAdd(), MatSolveTransposeAdd()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSolveTranspose(Mat mat,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,3);
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->ops->solvetranspose) SETERRQ1(PETSC_ERR_SUP,"Matrix type %s",((PetscObject)mat)->type_name);
  if (mat->rmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->rmap->N,x->map->N);
  if (mat->cmap->N != b->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %D %D",mat->cmap->N,b->map->N);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_SolveTranspose,mat,b,x,0);CHKERRQ(ierr);
  ierr = (*mat->ops->solvetranspose)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SolveTranspose,mat,b,x,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTransposeAdd"
/*@
   MatSolveTransposeAdd - Computes x = y + inv(Transpose(A)) b, given a 
                      factored matrix. 

   Collective on Mat and Vec

   Input Parameters:
+  mat - the factored matrix
.  b - the right-hand-side vector
-  y - the vector to be added to 

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolveTransposeAdd(A,x,y,x).

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolve(), MatSolveAdd(), MatSolveTranspose()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSolveTransposeAdd(Mat mat,Vec b,Vec y,Vec x)
{
  PetscScalar    one = 1.0;
  PetscErrorCode ierr;
  Vec            tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidHeaderSpecific(b,VEC_COOKIE,3);  
  PetscValidHeaderSpecific(x,VEC_COOKIE,4);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,y,3);
  PetscCheckSameComm(mat,1,x,4);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (mat->rmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->rmap->N,x->map->N);
  if (mat->cmap->N != b->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %D %D",mat->cmap->N,b->map->N);
  if (mat->cmap->N != y->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %D %D",mat->cmap->N,y->map->N);
  if (x->map->n != y->map->n)   SETERRQ2(PETSC_ERR_ARG_SIZ,"Vec x,Vec y: local dim %D %D",x->map->n,y->map->n);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_SolveTransposeAdd,mat,b,x,y);CHKERRQ(ierr);
  if (mat->ops->solvetransposeadd) {
    ierr = (*mat->ops->solvetransposeadd)(mat,b,y,x);CHKERRQ(ierr);
  } else {
    /* do the solve then the add manually */
    if (x != y) {
      ierr = MatSolveTranspose(mat,b,x);CHKERRQ(ierr);
      ierr = VecAXPY(x,one,y);CHKERRQ(ierr);
    } else {
      ierr = VecDuplicate(x,&tmp);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(mat,tmp);CHKERRQ(ierr);
      ierr = VecCopy(x,tmp);CHKERRQ(ierr);
      ierr = MatSolveTranspose(mat,b,x);CHKERRQ(ierr);
      ierr = VecAXPY(x,one,tmp);CHKERRQ(ierr);
      ierr = VecDestroy(tmp);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(MAT_SolveTransposeAdd,mat,b,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatSOR"
/*@
   MatSOR - Computes relaxation (SOR, Gauss-Seidel) sweeps.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
.  b - the right hand side
.  omega - the relaxation factor
.  flag - flag indicating the type of SOR (see below)
.  shift -  diagonal shift
.  its - the number of iterations
-  lits - the number of local iterations 

   Output Parameters:
.  x - the solution (can contain an initial guess, use option SOR_ZERO_INITIAL_GUESS to indicate no guess)

   SOR Flags:
.     SOR_FORWARD_SWEEP - forward SOR
.     SOR_BACKWARD_SWEEP - backward SOR
.     SOR_SYMMETRIC_SWEEP - SSOR (symmetric SOR)
.     SOR_LOCAL_FORWARD_SWEEP - local forward SOR 
.     SOR_LOCAL_BACKWARD_SWEEP - local forward SOR 
.     SOR_LOCAL_SYMMETRIC_SWEEP - local SSOR
.     SOR_APPLY_UPPER, SOR_APPLY_LOWER - applies 
         upper/lower triangular part of matrix to
         vector (with omega)
.     SOR_ZERO_INITIAL_GUESS - zero initial guess

   Notes:
   SOR_LOCAL_FORWARD_SWEEP, SOR_LOCAL_BACKWARD_SWEEP, and
   SOR_LOCAL_SYMMETRIC_SWEEP perform separate independent smoothings
   on each processor. 

   Application programmers will not generally use MatSOR() directly,
   but instead will employ the KSP/PC interface.

   Notes: for BAIJ, SBAIJ, and AIJ matrices with Inodes this does a block SOR smoothing, otherwise it does a pointwise smoothing

   Notes for Advanced Users:
   The flags are implemented as bitwise inclusive or operations.
   For example, use (SOR_ZERO_INITIAL_GUESS | SOR_SYMMETRIC_SWEEP)
   to specify a zero initial guess for SSOR.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().


   Level: developer

   Concepts: matrices^relaxation
   Concepts: matrices^SOR
   Concepts: matrices^Gauss-Seidel

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSOR(Mat mat,Vec b,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(x,VEC_COOKIE,8);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,8);
  if (!mat->ops->sor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (mat->cmap->N != x->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %D %D",mat->cmap->N,x->map->N);
  if (mat->rmap->N != b->map->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %D %D",mat->rmap->N,b->map->N);
  if (mat->rmap->n != b->map->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %D %D",mat->rmap->n,b->map->n);
  if (its <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D positive",its);
  if (lits <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"Relaxation requires local its %D positive",lits);

  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_SOR,mat,b,x,0);CHKERRQ(ierr);
  ierr =(*mat->ops->sor)(mat,b,omega,flag,shift,its,lits,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SOR,mat,b,x,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_Basic"
/*
      Default matrix copy routine.
*/
PetscErrorCode MatCopy_Basic(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode    ierr;
  PetscInt          i,rstart = 0,rend = 0,nz;
  const PetscInt    *cwork;
  const PetscScalar *vwork;

  PetscFunctionBegin;
  if (B->assembled) {
    ierr = MatZeroEntries(B);CHKERRQ(ierr);
  }
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    ierr = MatSetValues(B,1,&i,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy"
/*@
   MatCopy - Copys a matrix to another matrix.

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  str - SAME_NONZERO_PATTERN or DIFFERENT_NONZERO_PATTERN

   Output Parameter:
.  B - where the copy is put

   Notes:
   If you use SAME_NONZERO_PATTERN then the two matrices had better have the 
   same nonzero pattern or the routine will crash.

   MatCopy() copies the matrix entries of a matrix to another existing
   matrix (after first zeroing the second matrix).  A related routine is
   MatConvert(), which first creates a new matrix and then copies the data.

   Level: intermediate
   
   Concepts: matrices^copying

.seealso: MatConvert(), MatDuplicate()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCopy(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(A,1);
  PetscValidType(B,2);
  PetscCheckSameComm(A,1,B,2);
  ierr = MatPreallocated(B);CHKERRQ(ierr);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (A->rmap->N != B->rmap->N || A->cmap->N != B->cmap->N) SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim (%D,%D) (%D,%D)",A->rmap->N,B->rmap->N,A->cmap->N,B->cmap->N);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_Copy,A,B,0,0);CHKERRQ(ierr);
  if (A->ops->copy) { 
    ierr = (*A->ops->copy)(A,B,str);CHKERRQ(ierr);
  } else { /* generic conversion */
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  if (A->mapping) {
    if (B->mapping) {ierr = ISLocalToGlobalMappingDestroy(B->mapping);CHKERRQ(ierr);B->mapping = 0;}
    ierr = MatSetLocalToGlobalMapping(B,A->mapping);CHKERRQ(ierr);
  }
  if (A->bmapping) {
    if (B->bmapping) {ierr = ISLocalToGlobalMappingDestroy(B->bmapping);CHKERRQ(ierr);B->bmapping = 0;}
    ierr = MatSetLocalToGlobalMappingBlock(B,A->mapping);CHKERRQ(ierr);
  }

  B->stencil.dim = A->stencil.dim;
  B->stencil.noc = A->stencil.noc;
  for (i=0; i<=A->stencil.dim; i++) {
    B->stencil.dims[i]   = A->stencil.dims[i];
    B->stencil.starts[i] = A->stencil.starts[i];
  }

  ierr = PetscLogEventEnd(MAT_Copy,A,B,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatConvert"
/*@C  
   MatConvert - Converts a matrix to another matrix, either of the same
   or different type.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  newtype - new matrix type.  Use MATSAME to create a new matrix of the
   same type as the original matrix.
-  reuse - denotes if the destination matrix is to be created or reused.  Currently
   MAT_REUSE_MATRIX is only supported for inplace conversion, otherwise use
   MAT_INITIAL_MATRIX.

   Output Parameter:
.  M - pointer to place new matrix

   Notes:
   MatConvert() first creates a new matrix and then copies the data from
   the first matrix.  A related routine is MatCopy(), which copies the matrix
   entries of one matrix to another already existing matrix context.

   Cannot be used to convert a sequential matrix to parallel or parallel to sequential,
   the MPI communicator of the generated matrix is always the same as the communicator
   of the input matrix.

   Level: intermediate

   Concepts: matrices^converting between storage formats

.seealso: MatCopy(), MatDuplicate()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert(Mat mat, const MatType newtype,MatReuse reuse,Mat *M)
{
  PetscErrorCode         ierr;
  PetscTruth             sametype,issame,flg;
  char                   convname[256],mtype[256];
  Mat                    B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(M,3);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(((PetscObject)mat)->prefix,"-matconvert_type",mtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    newtype = mtype;
  }
  ierr = PetscTypeCompare((PetscObject)mat,newtype,&sametype);CHKERRQ(ierr);
  ierr = PetscStrcmp(newtype,"same",&issame);CHKERRQ(ierr);
  if ((reuse == MAT_REUSE_MATRIX) && (mat != *M)) {
    SETERRQ(PETSC_ERR_SUP,"MAT_REUSE_MATRIX only supported for in-place conversion currently");
  }

  if ((reuse == MAT_REUSE_MATRIX) && (issame || sametype)) PetscFunctionReturn(0);
  
  if ((sametype || issame) && (reuse==MAT_INITIAL_MATRIX) && mat->ops->duplicate) {
    ierr = (*mat->ops->duplicate)(mat,MAT_COPY_VALUES,M);CHKERRQ(ierr);
  } else {
    PetscErrorCode (*conv)(Mat, const MatType,MatReuse,Mat*)=PETSC_NULL;
    const char     *prefix[3] = {"seq","mpi",""};
    PetscInt       i;
    /* 
       Order of precedence:
       1) See if a specialized converter is known to the current matrix.
       2) See if a specialized converter is known to the desired matrix class.
       3) See if a good general converter is registered for the desired class
          (as of 6/27/03 only MATMPIADJ falls into this category).
       4) See if a good general converter is known for the current matrix.
       5) Use a really basic converter.
    */
    
    /* 1) See if a specialized converter is known to the current matrix and the desired class */
    for (i=0; i<3; i++) {
      ierr = PetscStrcpy(convname,"MatConvert_");CHKERRQ(ierr);
      ierr = PetscStrcat(convname,((PetscObject)mat)->type_name);CHKERRQ(ierr);
      ierr = PetscStrcat(convname,"_");CHKERRQ(ierr);
      ierr = PetscStrcat(convname,prefix[i]);CHKERRQ(ierr);
      ierr = PetscStrcat(convname,newtype);CHKERRQ(ierr);
      ierr = PetscStrcat(convname,"_C");CHKERRQ(ierr);
      ierr = PetscObjectQueryFunction((PetscObject)mat,convname,(void (**)(void))&conv);CHKERRQ(ierr);
      if (conv) goto foundconv;
    }

    /* 2)  See if a specialized converter is known to the desired matrix class. */
    ierr = MatCreate(((PetscObject)mat)->comm,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,mat->rmap->n,mat->cmap->n,mat->rmap->N,mat->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(B,newtype);CHKERRQ(ierr);
    for (i=0; i<3; i++) {
      ierr = PetscStrcpy(convname,"MatConvert_");CHKERRQ(ierr);
      ierr = PetscStrcat(convname,((PetscObject)mat)->type_name);CHKERRQ(ierr);
      ierr = PetscStrcat(convname,"_");CHKERRQ(ierr);
      ierr = PetscStrcat(convname,prefix[i]);CHKERRQ(ierr);
      ierr = PetscStrcat(convname,newtype);CHKERRQ(ierr);
      ierr = PetscStrcat(convname,"_C");CHKERRQ(ierr);
      ierr = PetscObjectQueryFunction((PetscObject)B,convname,(void (**)(void))&conv);CHKERRQ(ierr);
      if (conv) {
        ierr = MatDestroy(B);CHKERRQ(ierr);      
        goto foundconv;
      }
    }

    /* 3) See if a good general converter is registered for the desired class */
    conv = B->ops->convertfrom;
    ierr = MatDestroy(B);CHKERRQ(ierr);
    if (conv) goto foundconv;

    /* 4) See if a good general converter is known for the current matrix */
    if (mat->ops->convert) {
      conv = mat->ops->convert;
    }
    if (conv) goto foundconv;

    /* 5) Use a really basic converter. */
    conv = MatConvert_Basic;

    foundconv:
    ierr = PetscLogEventBegin(MAT_Convert,mat,0,0,0);CHKERRQ(ierr);
    ierr = (*conv)(mat,newtype,reuse,M);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_Convert,mat,0,0,0);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)*M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage"
/*@C  
   MatFactorGetSolverPackage - Returns name of the package providing the factorization routines

   Not Collective

   Input Parameter:
.  mat - the matrix, must be a factored matrix

   Output Parameter:
.   type - the string name of the package (do not free this string)

   Notes:
      In Fortran you pass in a empty string and the package name will be copied into it. 
    (Make sure the string is long enough)

   Level: intermediate

.seealso: MatCopy(), MatDuplicate(), MatGetFactorAvailable(), MatGetFactor()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatFactorGetSolverPackage(Mat mat, const MatSolverPackage *type)
{
  PetscErrorCode         ierr;
  PetscErrorCode         (*conv)(Mat,const MatSolverPackage*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix"); 
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatFactorGetSolverPackage_C",(void (**)(void))&conv);CHKERRQ(ierr);
  if (!conv) {
    *type = MAT_SOLVER_PETSC;
  } else {
    ierr = (*conv)(mat,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor"
/*@C  
   MatGetFactor - Returns a matrix suitable to calls to MatXXFactorSymbolic()

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  type - name of solver type, for example, spooles, superlu, plapack, petsc (to use PETSc's default)
-  ftype - factor type, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ICC, MAT_FACTOR_ILU, 

   Output Parameters:
.  f - the factor matrix used with MatXXFactorSymbolic() calls 

   Notes:
      Some PETSc matrix formats have alternative solvers available that are contained in alternative packages
     such as pastix, superlu, mumps, spooles etc. 

      PETSc must have been config/configure.py to use the external solver, using the option --download-package

   Level: intermediate

.seealso: MatCopy(), MatDuplicate(), MatGetFactorAvailable()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetFactor(Mat mat, const MatSolverPackage type,MatFactorType ftype,Mat *f)
{
  PetscErrorCode         ierr;
  char                   convname[256];
  PetscErrorCode         (*conv)(Mat,MatFactorType,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);

  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscStrcpy(convname,"MatGetFactor_");CHKERRQ(ierr);
  ierr = PetscStrcat(convname,type);CHKERRQ(ierr);
  ierr = PetscStrcat(convname,"_C");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)mat,convname,(void (**)(void))&conv);CHKERRQ(ierr);
  if (!conv) {
    PetscTruth flag;
    ierr = PetscStrcasecmp(MAT_SOLVER_PETSC,type,&flag);CHKERRQ(ierr);
    if (flag) {
      SETERRQ1(PETSC_ERR_SUP,"Matrix format %s does not have a built-in PETSc direct solver",((PetscObject)mat)->type_name);
    } else {
      SETERRQ3(PETSC_ERR_SUP,"Matrix format %s does not have a solver %s. Perhaps you must config/configure.py with --download-%s",((PetscObject)mat)->type_name,type,type);
    }
  }
  ierr = (*conv)(mat,ftype,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetFactorAvailable"
/*@C  
   MatGetFactorAvailable - Returns a a flag if matrix supports particular package and factor type

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  type - name of solver type, for example, spooles, superlu, plapack, petsc (to use PETSc's default)
-  ftype - factor type, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ICC, MAT_FACTOR_ILU, 

   Output Parameter:
.    flg - PETSC_TRUE if the factorization is available

   Notes:
      Some PETSc matrix formats have alternative solvers available that are contained in alternative packages
     such as pastix, superlu, mumps, spooles etc. 

      PETSc must have been config/configure.py to use the external solver, using the option --download-package

   Level: intermediate

.seealso: MatCopy(), MatDuplicate(), MatGetFactor()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetFactorAvailable(Mat mat, const MatSolverPackage type,MatFactorType ftype,PetscTruth *flg)
{
  PetscErrorCode         ierr;
  char                   convname[256];
  PetscErrorCode         (*conv)(Mat,MatFactorType,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);

  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscStrcpy(convname,"MatGetFactorAvailable_");CHKERRQ(ierr);
  ierr = PetscStrcat(convname,type);CHKERRQ(ierr);
  ierr = PetscStrcat(convname,"_C");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)mat,convname,(void (**)(void))&conv);CHKERRQ(ierr);
  if (!conv) {
    *flg = PETSC_FALSE;
  } else {
    ierr = (*conv)(mat,ftype,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate"
/*@
   MatDuplicate - Duplicates a matrix including the non-zero structure.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  op - either MAT_DO_NOT_COPY_VALUES or MAT_COPY_VALUES, cause it to copy the numerical values in the matrix
        MAT_SHARE_NONZERO_PATTERN to share the nonzero patterns with the previous matrix and not copy them.

   Output Parameter:
.  M - pointer to place new matrix

   Level: intermediate

   Concepts: matrices^duplicating

    Notes: You cannot change the nonzero pattern for the parent or child matrix if you use MAT_SHARE_NONZERO_PATTERN.

.seealso: MatCopy(), MatConvert()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatDuplicate(Mat mat,MatDuplicateOption op,Mat *M)
{
  PetscErrorCode ierr;
  Mat            B;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(M,3);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  *M  = 0;
  if (!mat->ops->duplicate) {
    SETERRQ(PETSC_ERR_SUP,"Not written for this matrix type");
  }
  ierr = PetscLogEventBegin(MAT_Convert,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->duplicate)(mat,op,M);CHKERRQ(ierr);
  B = *M;
  if (mat->mapping) {
    ierr = MatSetLocalToGlobalMapping(B,mat->mapping);CHKERRQ(ierr);
  }
  if (mat->bmapping) {
    ierr = MatSetLocalToGlobalMappingBlock(B,mat->bmapping);CHKERRQ(ierr);
  }
  ierr = PetscLayoutCopy(mat->rmap,&B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutCopy(mat->cmap,&B->cmap);CHKERRQ(ierr);
  
  B->stencil.dim = mat->stencil.dim;
  B->stencil.noc = mat->stencil.noc;
  for (i=0; i<=mat->stencil.dim; i++) {
    B->stencil.dims[i]   = mat->stencil.dims[i];
    B->stencil.starts[i] = mat->stencil.starts[i];
  }

  ierr = PetscLogEventEnd(MAT_Convert,mat,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal"
/*@ 
   MatGetDiagonal - Gets the diagonal of a matrix.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  v - the vector for storing the diagonal

   Output Parameter:
.  v - the diagonal of the matrix

   Level: intermediate

   Concepts: matrices^accessing diagonals

.seealso: MatGetRow(), MatGetSubMatrices(), MatGetSubmatrix(), MatGetRowMaxAbs()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetDiagonal(Mat mat,Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->getdiagonal) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = (*mat->ops->getdiagonal)(mat,v);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMin"
/*@ 
   MatGetRowMin - Gets the minimum value (of the real part) of each
        row of the matrix

   Collective on Mat and Vec

   Input Parameters:
.  mat - the matrix

   Output Parameter:
+  v - the vector for storing the maximums
-  idx - the indices of the column found for each row (optional)

   Level: intermediate

   Notes: The result of this call are the same as if one converted the matrix to dense format
      and found the minimum value in each row (i.e. the implicit zeros are counted as zeros).

    This code is only implemented for a couple of matrix formats.

   Concepts: matrices^getting row maximums

.seealso: MatGetDiagonal(), MatGetSubMatrices(), MatGetSubmatrix(), MatGetRowMaxAbs(),
          MatGetRowMax()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowMin(Mat mat,Vec v,PetscInt idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->getrowmax) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = (*mat->ops->getrowmin)(mat,v,idx);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMinAbs"
/*@ 
   MatGetRowMinAbs - Gets the minimum value (in absolute value) of each
        row of the matrix

   Collective on Mat and Vec

   Input Parameters:
.  mat - the matrix

   Output Parameter:
+  v - the vector for storing the minimums
-  idx - the indices of the column found for each row (optional)

   Level: intermediate

   Notes: if a row is completely empty or has only 0.0 values then the idx[] value for that
    row is 0 (the first column).

    This code is only implemented for a couple of matrix formats.

   Concepts: matrices^getting row maximums

.seealso: MatGetDiagonal(), MatGetSubMatrices(), MatGetSubmatrix(), MatGetRowMax(), MatGetRowMaxAbs(), MatGetRowMin()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowMinAbs(Mat mat,Vec v,PetscInt idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->getrowminabs) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (idx) {ierr = PetscMemzero(idx,mat->rmap->n*sizeof(PetscInt));CHKERRQ(ierr);}

  ierr = (*mat->ops->getrowminabs)(mat,v,idx);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMax"
/*@ 
   MatGetRowMax - Gets the maximum value (of the real part) of each
        row of the matrix

   Collective on Mat and Vec

   Input Parameters:
.  mat - the matrix

   Output Parameter:
+  v - the vector for storing the maximums
-  idx - the indices of the column found for each row (optional)

   Level: intermediate

   Notes: The result of this call are the same as if one converted the matrix to dense format
      and found the minimum value in each row (i.e. the implicit zeros are counted as zeros).

    This code is only implemented for a couple of matrix formats.

   Concepts: matrices^getting row maximums

.seealso: MatGetDiagonal(), MatGetSubMatrices(), MatGetSubmatrix(), MatGetRowMaxAbs(), MatGetRowMin()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowMax(Mat mat,Vec v,PetscInt idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->getrowmax) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = (*mat->ops->getrowmax)(mat,v,idx);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMaxAbs"
/*@ 
   MatGetRowMaxAbs - Gets the maximum value (in absolute value) of each
        row of the matrix

   Collective on Mat and Vec

   Input Parameters:
.  mat - the matrix

   Output Parameter:
+  v - the vector for storing the maximums
-  idx - the indices of the column found for each row (optional)

   Level: intermediate

   Notes: if a row is completely empty or has only 0.0 values then the idx[] value for that
    row is 0 (the first column).

    This code is only implemented for a couple of matrix formats.

   Concepts: matrices^getting row maximums

.seealso: MatGetDiagonal(), MatGetSubMatrices(), MatGetSubmatrix(), MatGetRowMax(), MatGetRowMin()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowMaxAbs(Mat mat,Vec v,PetscInt idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->getrowmaxabs) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (idx) {ierr = PetscMemzero(idx,mat->rmap->n*sizeof(PetscInt));CHKERRQ(ierr);}

  ierr = (*mat->ops->getrowmaxabs)(mat,v,idx);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowSum"
/*@ 
   MatGetRowSum - Gets the sum of each row of the matrix

   Collective on Mat and Vec

   Input Parameters:
.  mat - the matrix

   Output Parameter:
.  v - the vector for storing the sum of rows

   Level: intermediate

   Notes: This code is slow since it is not currently specialized for different formats

   Concepts: matrices^getting row sums

.seealso: MatGetDiagonal(), MatGetSubMatrices(), MatGetSubmatrix(), MatGetRowMax(), MatGetRowMin()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowSum(Mat mat, Vec v)
{
  PetscInt       start = 0, end = 0, row;
  PetscScalar   *array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat, &start, &end);CHKERRQ(ierr);
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  for(row = start; row < end; ++row) {
    PetscInt           ncols, col;
    const PetscInt    *cols;
    const PetscScalar *vals;

    array[row - start] = 0.0;
    ierr = MatGetRow(mat, row, &ncols, &cols, &vals);CHKERRQ(ierr);
    for(col = 0; col < ncols; col++) {
      array[row - start] += vals[col];
    }
    ierr = MatRestoreRow(mat, row, &ncols, &cols, &vals);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject) v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose"
/*@
   MatTranspose - Computes an in-place or out-of-place transpose of a matrix.

   Collective on Mat

   Input Parameter:
+  mat - the matrix to transpose
-  reuse - store the transpose matrix in the provided B

   Output Parameters:
.  B - the transpose 

   Notes:
     If you  pass in &mat for B the transpose will be done in place

   Level: intermediate

   Concepts: matrices^transposing

.seealso: MatMultTranspose(), MatMultTransposeAdd(), MatIsTranspose(), MatReuse
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatTranspose(Mat mat,MatReuse reuse,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->transpose) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_Transpose,mat,0,0,0);CHKERRQ(ierr); 
  ierr = (*mat->ops->transpose)(mat,reuse,B);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Transpose,mat,0,0,0);CHKERRQ(ierr);
  if (B) {ierr = PetscObjectStateIncrease((PetscObject)*B);CHKERRQ(ierr);}
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsTranspose"
/*@
   MatIsTranspose - Test whether a matrix is another one's transpose, 
        or its own, in which case it tests symmetry.

   Collective on Mat

   Input Parameter:
+  A - the matrix to test
-  B - the matrix to test against, this can equal the first parameter

   Output Parameters:
.  flg - the result

   Notes:
   Only available for SeqAIJ/MPIAIJ matrices. The sequential algorithm
   has a running time of the order of the number of nonzeros; the parallel
   test involves parallel copies of the block-offdiagonal parts of the matrix.

   Level: intermediate

   Concepts: matrices^transposing, matrix^symmetry

.seealso: MatTranspose(), MatIsSymmetric(), MatIsHermitian()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatIsTranspose(Mat A,Mat B,PetscReal tol,PetscTruth *flg)
{
  PetscErrorCode ierr,(*f)(Mat,Mat,PetscReal,PetscTruth*),(*g)(Mat,Mat,PetscReal,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidPointer(flg,3);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatIsTranspose_C",(void (**)(void))&f);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatIsTranspose_C",(void (**)(void))&g);CHKERRQ(ierr);
  if (f && g) {
    if (f==g) {
      ierr = (*f)(A,B,tol,flg);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,"Matrices do not have the same comparator for symmetry test");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHermitianTranspose"
/*@ 
   MatHermitianTranspose - Computes an in-place or out-of-place transpose of a matrix in complex conjugate.

   Collective on Mat

   Input Parameter:
+  mat - the matrix to transpose and complex conjugate
-  reuse - store the transpose matrix in the provided B

   Output Parameters:
.  B - the Hermitian

   Notes:
     If you  pass in &mat for B the Hermitian will be done in place

   Level: intermediate

   Concepts: matrices^transposing, complex conjugatex

.seealso: MatTranspose(), MatMultTranspose(), MatMultTransposeAdd(), MatIsTranspose(), MatReuse
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatHermitianTranspose(Mat mat,MatReuse reuse,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatTranspose(mat,reuse,B);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = MatConjugate(*B);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsHermitianTranspose"
/*@
   MatIsHermitianTranspose - Test whether a matrix is another one's Hermitian transpose, 

   Collective on Mat

   Input Parameter:
+  A - the matrix to test
-  B - the matrix to test against, this can equal the first parameter

   Output Parameters:
.  flg - the result

   Notes:
   Only available for SeqAIJ/MPIAIJ matrices. The sequential algorithm
   has a running time of the order of the number of nonzeros; the parallel
   test involves parallel copies of the block-offdiagonal parts of the matrix.

   Level: intermediate

   Concepts: matrices^transposing, matrix^symmetry

.seealso: MatTranspose(), MatIsSymmetric(), MatIsHermitian(), MatIsTranspose()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatIsHermitianTranspose(Mat A,Mat B,PetscReal tol,PetscTruth *flg)
{
  PetscErrorCode ierr,(*f)(Mat,Mat,PetscReal,PetscTruth*),(*g)(Mat,Mat,PetscReal,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidPointer(flg,3);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatIsHermitianTranspose_C",(void (**)(void))&f);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatIsHermitianTranspose_C",(void (**)(void))&g);CHKERRQ(ierr);
  if (f && g) {
    if (f==g) {
      ierr = (*f)(A,B,tol,flg);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,"Matrices do not have the same comparator for Hermitian test");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPermute"
/*@
   MatPermute - Creates a new matrix with rows and columns permuted from the 
   original.

   Collective on Mat

   Input Parameters:
+  mat - the matrix to permute
.  row - row permutation, each processor supplies only the permutation for its rows
-  col - column permutation, each processor needs the entire column permutation, that is
         this is the same size as the total number of columns in the matrix. It can often
         be obtained with ISAllGather() on the row permutation

   Output Parameters:
.  B - the permuted matrix

   Level: advanced

   Concepts: matrices^permuting

.seealso: MatGetOrdering(), ISAllGather()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPermute(Mat mat,IS row,IS col,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(row,IS_COOKIE,2);
  PetscValidHeaderSpecific(col,IS_COOKIE,3);
  PetscValidPointer(B,4);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->permute) SETERRQ1(PETSC_ERR_SUP,"MatPermute not available for Mat type %s",((PetscObject)mat)->type_name); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = (*mat->ops->permute)(mat,row,col,B);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPermuteSparsify"
/*@
  MatPermuteSparsify - Creates a new matrix with rows and columns permuted from the 
  original and sparsified to the prescribed tolerance.

  Collective on Mat

  Input Parameters:
+ A    - The matrix to permute
. band - The half-bandwidth of the sparsified matrix, or PETSC_DECIDE
. frac - The half-bandwidth as a fraction of the total size, or 0.0
. tol  - The drop tolerance
. rowp - The row permutation
- colp - The column permutation

  Output Parameter:
. B    - The permuted, sparsified matrix

  Level: advanced

  Note:
  The default behavior (band = PETSC_DECIDE and frac = 0.0) is to
  restrict the half-bandwidth of the resulting matrix to 5% of the
  total matrix size.

.keywords: matrix, permute, sparsify

.seealso: MatGetOrdering(), MatPermute()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPermuteSparsify(Mat A, PetscInt band, PetscReal frac, PetscReal tol, IS rowp, IS colp, Mat *B)
{
  IS                irowp, icolp;
  const PetscInt    *rows, *cols;
  PetscInt          M, N, locRowStart = 0, locRowEnd = 0;
  PetscInt          nz, newNz;
  const PetscInt    *cwork;
  PetscInt          *cnew;
  const PetscScalar *vwork;
  PetscScalar       *vnew;
  PetscInt          bw, issize;
  PetscInt          row, locRow, newRow, col, newCol;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,    MAT_COOKIE,1);
  PetscValidHeaderSpecific(rowp, IS_COOKIE,5);
  PetscValidHeaderSpecific(colp, IS_COOKIE,6);
  PetscValidPointer(B,7);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  if (A->factor)     SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  if (!A->ops->permutesparsify) {
    ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A, &locRowStart, &locRowEnd);CHKERRQ(ierr);
    ierr = ISGetSize(rowp, &issize);CHKERRQ(ierr);
    if (issize != M) SETERRQ2(PETSC_ERR_ARG_WRONG, "Wrong size %D for row permutation, should be %D", issize, M);
    ierr = ISGetSize(colp, &issize);CHKERRQ(ierr);
    if (issize != N) SETERRQ2(PETSC_ERR_ARG_WRONG, "Wrong size %D for column permutation, should be %D", issize, N);
    ierr = ISInvertPermutation(rowp, 0, &irowp);CHKERRQ(ierr);
    ierr = ISGetIndices(irowp, &rows);CHKERRQ(ierr);
    ierr = ISInvertPermutation(colp, 0, &icolp);CHKERRQ(ierr);
    ierr = ISGetIndices(icolp, &cols);CHKERRQ(ierr);
    ierr = PetscMalloc(N*sizeof(PetscInt),&cnew);CHKERRQ(ierr);
    ierr = PetscMalloc(N*sizeof(PetscScalar),&vnew);CHKERRQ(ierr);

    /* Setup bandwidth to include */
    if (band == PETSC_DECIDE) {
      if (frac <= 0.0)
        bw = (PetscInt) (M * 0.05);
      else
        bw = (PetscInt) (M * frac);
    } else {
      if (band <= 0) SETERRQ(PETSC_ERR_ARG_WRONG, "Bandwidth must be a positive integer");
      bw = band;
    }

    /* Put values into new matrix */
    ierr = MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, B);CHKERRQ(ierr);
    for(row = locRowStart, locRow = 0; row < locRowEnd; row++, locRow++) {
      ierr = MatGetRow(A, row, &nz, &cwork, &vwork);CHKERRQ(ierr);
      newRow   = rows[locRow]+locRowStart;
      for(col = 0, newNz = 0; col < nz; col++) {
        newCol = cols[cwork[col]];
        if ((newCol >= newRow - bw) && (newCol < newRow + bw) && (PetscAbsScalar(vwork[col]) >= tol)) {
          cnew[newNz] = newCol;
          vnew[newNz] = vwork[col];
          newNz++;
        }
      }
      ierr = MatSetValues(*B, 1, &newRow, newNz, cnew, vnew, INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(A, row, &nz, &cwork, &vwork);CHKERRQ(ierr);
    }
    ierr = PetscFree(cnew);CHKERRQ(ierr);
    ierr = PetscFree(vnew);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = ISRestoreIndices(irowp, &rows);CHKERRQ(ierr);
    ierr = ISRestoreIndices(icolp, &cols);CHKERRQ(ierr);
    ierr = ISDestroy(irowp);CHKERRQ(ierr);
    ierr = ISDestroy(icolp);CHKERRQ(ierr);
  } else {
    ierr = (*A->ops->permutesparsify)(A, band, frac, tol, rowp, colp, B);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual"
/*@
   MatEqual - Compares two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
-  B - the second matrix

   Output Parameter:
.  flg - PETSC_TRUE if the matrices are equal; PETSC_FALSE otherwise.

   Level: intermediate

   Concepts: matrices^equality between
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatEqual(Mat A,Mat B,PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1); 
  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(A,1);
  PetscValidType(B,2);
  PetscValidIntPointer(flg,3);
  PetscCheckSameComm(A,1,B,2);
  ierr = MatPreallocated(B);CHKERRQ(ierr);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->rmap->N != B->rmap->N || A->cmap->N != B->cmap->N) SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim %D %D %D %D",A->rmap->N,B->rmap->N,A->cmap->N,B->cmap->N);
  if (!A->ops->equal) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)A)->type_name);
  if (!B->ops->equal) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)B)->type_name);
  if (A->ops->equal != B->ops->equal) SETERRQ2(PETSC_ERR_ARG_INCOMP,"A is type: %s\nB is type: %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  ierr = (*A->ops->equal)(A,B,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale"
/*@
   MatDiagonalScale - Scales a matrix on the left and right by diagonal
   matrices that are stored as vectors.  Either of the two scaling
   matrices can be PETSC_NULL.

   Collective on Mat

   Input Parameters:
+  mat - the matrix to be scaled
.  l - the left scaling vector (or PETSC_NULL)
-  r - the right scaling vector (or PETSC_NULL)

   Notes:
   MatDiagonalScale() computes A = LAR, where
   L = a diagonal matrix (stored as a vector), R = a diagonal matrix (stored as a vector)

   Level: intermediate

   Concepts: matrices^diagonal scaling
   Concepts: diagonal scaling of matrices

.seealso: MatScale()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatDiagonalScale(Mat mat,Vec l,Vec r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->ops->diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (l) {PetscValidHeaderSpecific(l,VEC_COOKIE,2);PetscCheckSameComm(mat,1,l,2);}
  if (r) {PetscValidHeaderSpecific(r,VEC_COOKIE,3);PetscCheckSameComm(mat,1,r,3);}
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->diagonalscale)(mat,l,r);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatScale"
/*@
    MatScale - Scales all elements of a matrix by a given number.

    Collective on Mat

    Input Parameters:
+   mat - the matrix to be scaled
-   a  - the scaling value

    Output Parameter:
.   mat - the scaled matrix

    Level: intermediate

    Concepts: matrices^scaling all entries

.seealso: MatDiagonalScale()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatScale(Mat mat,PetscScalar a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (a != 1.0 && !mat->ops->scale) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  if (a != 1.0) {
    ierr = (*mat->ops->scale)(mat,a);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  } 
  ierr = PetscLogEventEnd(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatNorm"
/*@ 
   MatNorm - Calculates various norms of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  type - the type of norm, NORM_1, NORM_FROBENIUS, NORM_INFINITY

   Output Parameters:
.  nrm - the resulting norm 

   Level: intermediate

   Concepts: matrices^norm
   Concepts: norm^of matrix
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatNorm(Mat mat,NormType type,PetscReal *nrm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidScalarPointer(nrm,3);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->norm) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = (*mat->ops->norm)(mat,type,nrm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
     This variable is used to prevent counting of MatAssemblyBegin() that
   are called from within a MatAssemblyEnd().
*/
static PetscInt MatAssemblyEnd_InUse = 0;
#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin"
/*@
   MatAssemblyBegin - Begins assembling the matrix.  This routine should
   be called after completing all calls to MatSetValues().

   Collective on Mat

   Input Parameters:
+  mat - the matrix 
-  type - type of assembly, either MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY
 
   Notes: 
   MatSetValues() generally caches the values.  The matrix is ready to
   use only after MatAssemblyBegin() and MatAssemblyEnd() have been called.
   Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
   in MatSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
   using the matrix.

   Level: beginner

   Concepts: matrices^assembling

.seealso: MatAssemblyEnd(), MatSetValues(), MatAssembled()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyBegin(Mat mat,MatAssemblyType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix.\nDid you forget to call MatSetUnfactored()?"); 
  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  if (!MatAssemblyEnd_InUse) {
    ierr = PetscLogEventBegin(MAT_AssemblyBegin,mat,0,0,0);CHKERRQ(ierr);
    if (mat->ops->assemblybegin){ierr = (*mat->ops->assemblybegin)(mat,type);CHKERRQ(ierr);}
    ierr = PetscLogEventEnd(MAT_AssemblyBegin,mat,0,0,0);CHKERRQ(ierr);
  } else {
    if (mat->ops->assemblybegin){ierr = (*mat->ops->assemblybegin)(mat,type);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssembled"
/*@
   MatAssembled - Indicates if a matrix has been assembled and is ready for
     use; for example, in matrix-vector product.

   Collective on Mat

   Input Parameter:
.  mat - the matrix 

   Output Parameter:
.  assembled - PETSC_TRUE or PETSC_FALSE

   Level: advanced

   Concepts: matrices^assembled?

.seealso: MatAssemblyEnd(), MatSetValues(), MatAssemblyBegin()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatAssembled(Mat mat,PetscTruth *assembled)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(assembled,2);
  *assembled = mat->assembled;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_Private"
/*
    Processes command line options to determine if/how a matrix
  is to be viewed. Called by MatAssemblyEnd() and MatLoad().
*/
PetscErrorCode MatView_Private(Mat mat)
{
  PetscErrorCode    ierr;
  PetscTruth        flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE,flg4 = PETSC_FALSE,flg6 = PETSC_FALSE,flg7 = PETSC_FALSE,flg8 = PETSC_FALSE;
  static PetscTruth incall = PETSC_FALSE;
#if defined(PETSC_USE_SOCKET_VIEWER)
  PetscTruth        flg5 = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsBegin(((PetscObject)mat)->comm,((PetscObject)mat)->prefix,"Matrix Options","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mat_view_info","Information on matrix size","MatView",flg1,&flg1,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mat_view_info_detailed","Nonzeros in the matrix","MatView",flg2,&flg2,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mat_view","Print matrix to stdout","MatView",flg3,&flg3,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mat_view_matlab","Print matrix to stdout in a format Matlab can read","MatView",flg4,&flg4,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_SOCKET_VIEWER)
    ierr = PetscOptionsTruth("-mat_view_socket","Send matrix to socket (can be read from matlab)","MatView",flg5,&flg5,PETSC_NULL);CHKERRQ(ierr);
#endif
    ierr = PetscOptionsTruth("-mat_view_binary","Save matrix to file in binary format","MatView",flg6,&flg6,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mat_view_draw","Draw the matrix nonzero structure","MatView",flg7,&flg7,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (flg1) {
    PetscViewer viewer;

    ierr = PetscViewerASCIIGetStdout(((PetscObject)mat)->comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatView(mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  if (flg2) {
    PetscViewer viewer;

    ierr = PetscViewerASCIIGetStdout(((PetscObject)mat)->comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = MatView(mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  if (flg3) {
    PetscViewer viewer;

    ierr = PetscViewerASCIIGetStdout(((PetscObject)mat)->comm,&viewer);CHKERRQ(ierr);
    ierr = MatView(mat,viewer);CHKERRQ(ierr);
  }
  if (flg4) {
    PetscViewer viewer;

    ierr = PetscViewerASCIIGetStdout(((PetscObject)mat)->comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_SOCKET_VIEWER)
  if (flg5) {
    ierr = MatView(mat,PETSC_VIEWER_SOCKET_(((PetscObject)mat)->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_SOCKET_(((PetscObject)mat)->comm));CHKERRQ(ierr);
  }
#endif
  if (flg6) {
    ierr = MatView(mat,PETSC_VIEWER_BINARY_(((PetscObject)mat)->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_BINARY_(((PetscObject)mat)->comm));CHKERRQ(ierr);
  }
  if (flg7) {
    ierr = PetscOptionsGetTruth(((PetscObject)mat)->prefix,"-mat_view_contour",&flg8,PETSC_NULL);CHKERRQ(ierr);
    if (flg8) {
      PetscViewerPushFormat(PETSC_VIEWER_DRAW_(((PetscObject)mat)->comm),PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);
    }
    ierr = MatView(mat,PETSC_VIEWER_DRAW_(((PetscObject)mat)->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(((PetscObject)mat)->comm));CHKERRQ(ierr);
    if (flg8) {
      PetscViewerPopFormat(PETSC_VIEWER_DRAW_(((PetscObject)mat)->comm));CHKERRQ(ierr);
    }
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd"
/*@
   MatAssemblyEnd - Completes assembling the matrix.  This routine should
   be called after MatAssemblyBegin().

   Collective on Mat

   Input Parameters:
+  mat - the matrix 
-  type - type of assembly, either MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY

   Options Database Keys:
+  -mat_view_info - Prints info on matrix at conclusion of MatEndAssembly()
.  -mat_view_info_detailed - Prints more detailed info
.  -mat_view - Prints matrix in ASCII format
.  -mat_view_matlab - Prints matrix in Matlab format
.  -mat_view_draw - PetscDraws nonzero structure of matrix, using MatView() and PetscDrawOpenX().
.  -display <name> - Sets display name (default is host)
.  -draw_pause <sec> - Sets number of seconds to pause after display
.  -mat_view_socket - Sends matrix to socket, can be accessed from Matlab (see users manual)
.  -viewer_socket_machine <machine>
.  -viewer_socket_port <port>
.  -mat_view_binary - save matrix to file in binary format
-  -viewer_binary_filename <name>

   Notes: 
   MatSetValues() generally caches the values.  The matrix is ready to
   use only after MatAssemblyBegin() and MatAssemblyEnd() have been called.
   Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
   in MatSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
   using the matrix.

   Level: beginner

.seealso: MatAssemblyBegin(), MatSetValues(), PetscDrawOpenX(), MatView(), MatAssembled(), PetscViewerSocketOpen()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatAssemblyEnd(Mat mat,MatAssemblyType type)
{
  PetscErrorCode  ierr;
  static PetscInt inassm = 0;
  PetscTruth      flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);

  inassm++;
  MatAssemblyEnd_InUse++;
  if (MatAssemblyEnd_InUse == 1) { /* Do the logging only the first time through */
    ierr = PetscLogEventBegin(MAT_AssemblyEnd,mat,0,0,0);CHKERRQ(ierr);
    if (mat->ops->assemblyend) {
      ierr = (*mat->ops->assemblyend)(mat,type);CHKERRQ(ierr);
    }
    ierr = PetscLogEventEnd(MAT_AssemblyEnd,mat,0,0,0);CHKERRQ(ierr);
  } else {
    if (mat->ops->assemblyend) {
      ierr = (*mat->ops->assemblyend)(mat,type);CHKERRQ(ierr);
    }
  }

  /* Flush assembly is not a true assembly */
  if (type != MAT_FLUSH_ASSEMBLY) {
    mat->assembled  = PETSC_TRUE; mat->num_ass++;
  }
  mat->insertmode = NOT_SET_VALUES;
  MatAssemblyEnd_InUse--;
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  if (!mat->symmetric_eternal) {
    mat->symmetric_set              = PETSC_FALSE;
    mat->hermitian_set              = PETSC_FALSE;
    mat->structurally_symmetric_set = PETSC_FALSE;
  }
  if (inassm == 1 && type != MAT_FLUSH_ASSEMBLY) {
    ierr = MatView_Private(mat);CHKERRQ(ierr);
    ierr = PetscOptionsGetTruth(((PetscObject)mat)->prefix,"-mat_is_symmetric",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscReal tol = 0.0;
      ierr = PetscOptionsGetReal(((PetscObject)mat)->prefix,"-mat_is_symmetric",&tol,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatIsSymmetric(mat,tol,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscPrintf(((PetscObject)mat)->comm,"Matrix is symmetric (tolerance %G)\n",tol);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(((PetscObject)mat)->comm,"Matrix is not symmetric (tolerance %G)\n",tol);CHKERRQ(ierr);
      }
    }
  }
  inassm--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption"
/*@
   MatSetOption - Sets a parameter option for a matrix. Some options
   may be specific to certain storage formats.  Some options
   determine how values will be inserted (or added). Sorted, 
   row-oriented input will generally assemble the fastest. The default
   is row-oriented, nonsorted input. 

   Collective on Mat

   Input Parameters:
+  mat - the matrix 
.  option - the option, one of those listed below (and possibly others),
-  flg - turn the option on (PETSC_TRUE) or off (PETSC_FALSE)

  Options Describing Matrix Structure:
+    MAT_SYMMETRIC - symmetric in terms of both structure and value
.    MAT_HERMITIAN - transpose is the complex conjugation
.    MAT_STRUCTURALLY_SYMMETRIC - symmetric nonzero structure
-    MAT_SYMMETRY_ETERNAL - if you would like the symmetry/Hermitian flag
                            you set to be kept with all future use of the matrix
                            including after MatAssemblyBegin/End() which could
                            potentially change the symmetry structure, i.e. you 
                            KNOW the matrix will ALWAYS have the property you set.


   Options For Use with MatSetValues():
   Insert a logically dense subblock, which can be
.    MAT_ROW_ORIENTED - row-oriented (default)

   Note these options reflect the data you pass in with MatSetValues(); it has 
   nothing to do with how the data is stored internally in the matrix 
   data structure.

   When (re)assembling a matrix, we can restrict the input for
   efficiency/debugging purposes.  These options include
+    MAT_NEW_NONZERO_LOCATIONS - additional insertions will be
        allowed if they generate a new nonzero
.    MAT_NEW_DIAGONALS - new diagonals will be allowed (for block diagonal format only)
.    MAT_IGNORE_OFF_PROC_ENTRIES - drops off-processor entries
.    MAT_NEW_NONZERO_LOCATION_ERR - generates an error for new matrix entry
-    MAT_USE_HASH_TABLE - uses a hash table to speed up matrix assembly

   Notes:
   Some options are relevant only for particular matrix types and
   are thus ignored by others.  Other options are not supported by
   certain matrix types and will generate an error message if set.

   If using a Fortran 77 module to compute a matrix, one may need to 
   use the column-oriented option (or convert to the row-oriented 
   format).  

   MAT_NEW_NONZERO_LOCATIONS set to PETSC_FALSE indicates that any add or insertion 
   that would generate a new entry in the nonzero structure is instead
   ignored.  Thus, if memory has not alredy been allocated for this particular 
   data, then the insertion is ignored. For dense matrices, in which
   the entire array is allocated, no entries are ever ignored. 
   Set after the first MatAssemblyEnd()

   MAT_NEW_NONZERO_LOCATION_ERR indicates that any add or insertion 
   that would generate a new entry in the nonzero structure instead produces 
   an error. (Currently supported for AIJ and BAIJ formats only.)
   This is a useful flag when using SAME_NONZERO_PATTERN in calling
   KSPSetOperators() to ensure that the nonzero pattern truely does 
   remain unchanged. Set after the first MatAssemblyEnd()

   MAT_NEW_NONZERO_ALLOCATION_ERR indicates that any add or insertion 
   that would generate a new entry that has not been preallocated will 
   instead produce an error. (Currently supported for AIJ and BAIJ formats
   only.) This is a useful flag when debugging matrix memory preallocation.

   MAT_IGNORE_OFF_PROC_ENTRIES indicates entries destined for 
   other processors should be dropped, rather than stashed.
   This is useful if you know that the "owning" processor is also 
   always generating the correct matrix entries, so that PETSc need
   not transfer duplicate entries generated on another processor.
   
   MAT_USE_HASH_TABLE indicates that a hash table be used to improve the
   searches during matrix assembly. When this flag is set, the hash table
   is created during the first Matrix Assembly. This hash table is
   used the next time through, during MatSetVaules()/MatSetVaulesBlocked()
   to improve the searching of indices. MAT_NEW_NONZERO_LOCATIONS flag 
   should be used with MAT_USE_HASH_TABLE flag. This option is currently
   supported by MATMPIBAIJ format only.

   MAT_KEEP_NONZERO_PATTERN indicates when MatZeroRows() is called the zeroed entries
   are kept in the nonzero structure

   MAT_IGNORE_ZERO_ENTRIES - for AIJ/IS matrices this will stop zero values from creating
   a zero location in the matrix

   MAT_USE_INODES - indicates using inode version of the code - works with AIJ and 
   ROWBS matrix types

   Level: intermediate

   Concepts: matrices^setting options

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetOption(Mat mat,MatOption op,PetscTruth flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (((int) op) < 0 || ((int) op) >= NUM_MAT_OPTIONS) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Options %d is out of range",(int)op);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  switch (op) {
  case MAT_SYMMETRIC:
    mat->symmetric                  = flg;
    if (flg) mat->structurally_symmetric = PETSC_TRUE;
    mat->symmetric_set              = PETSC_TRUE;
    mat->structurally_symmetric_set = flg;
    break;
  case MAT_HERMITIAN:
    mat->hermitian                  = flg;
    if (flg) mat->structurally_symmetric = PETSC_TRUE;
    mat->hermitian_set              = PETSC_TRUE;
    mat->structurally_symmetric_set = flg;
    break;
  case MAT_STRUCTURALLY_SYMMETRIC:
    mat->structurally_symmetric     = flg;
    mat->structurally_symmetric_set = PETSC_TRUE;
    break;
  case MAT_SYMMETRY_ETERNAL:
    mat->symmetric_eternal          = flg;
    break;
  default:
    break;
  }
  if (mat->ops->setoption) {
    ierr = (*mat->ops->setoption)(mat,op,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries"
/*@
   MatZeroEntries - Zeros all entries of a matrix.  For sparse matrices
   this routine retains the old nonzero structure.

   Collective on Mat

   Input Parameters:
.  mat - the matrix 

   Level: intermediate

   Concepts: matrices^zeroing

.seealso: MatZeroRows()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatZeroEntries(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (mat->insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for matrices where you have set values but not yet assembled"); 
  if (!mat->ops->zeroentries) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_ZeroEntries,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->zeroentries)(mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ZeroEntries,mat,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows"
/*@C
   MatZeroRows - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  numRows - the number of rows to remove
.  rows - the global row indices
-  diag - value put in all diagonals of eliminated rows (0.0 will even eliminate diagonal entry)

   Notes:
   For the AIJ and BAIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Each processor can indicate any rows in the entire matrix to be zeroed (i.e. each process does NOT have to
   list only rows local to itself).

   Level: intermediate

   Concepts: matrices^zeroing rows

.seealso: MatZeroRowsIS(), MatZeroEntries(), MatZeroRowsLocal(), MatSetOption()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatZeroRows(Mat mat,PetscInt numRows,const PetscInt rows[],PetscScalar diag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (numRows) PetscValidIntPointer(rows,3);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->zerorows) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = (*mat->ops->zerorows)(mat,numRows,rows,diag);CHKERRQ(ierr);
  ierr = MatView_Private(mat);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRowsIS"
/*@C
   MatZeroRowsIS - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  is - index set of rows to remove
-  diag - value put in all diagonals of eliminated rows

   Notes:
   For the AIJ and BAIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Each processor should list the rows that IT wants zeroed

   Level: intermediate

   Concepts: matrices^zeroing rows

.seealso: MatZeroRows(), MatZeroEntries(), MatZeroRowsLocal(), MatSetOption()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatZeroRowsIS(Mat mat,IS is,PetscScalar diag)
{
  PetscInt       numRows;
  const PetscInt *rows;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,2);
  ierr = ISGetLocalSize(is,&numRows);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);
  ierr = MatZeroRows(mat,numRows,rows,diag);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&rows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRowsLocal"
/*@C 
   MatZeroRowsLocal - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix; using local numbering of rows.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  numRows - the number of rows to remove
.  rows - the global row indices
-  diag - value put in all diagonals of eliminated rows

   Notes:
   Before calling MatZeroRowsLocal(), the user must first set the
   local-to-global mapping by calling MatSetLocalToGlobalMapping().

   For the AIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   Level: intermediate

   Concepts: matrices^zeroing

.seealso: MatZeroRows(), MatZeroRowsLocalIS(), MatZeroEntries(), MatZeroRows(), MatSetLocalToGlobalMapping
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatZeroRowsLocal(Mat mat,PetscInt numRows,const PetscInt rows[],PetscScalar diag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (numRows) PetscValidIntPointer(rows,3);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (mat->ops->zerorowslocal) {
    ierr = (*mat->ops->zerorowslocal)(mat,numRows,rows,diag);CHKERRQ(ierr);
  } else {
    IS             is, newis;
    const PetscInt *newRows;

    if (!mat->mapping) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Need to provide local to global mapping to matrix first");
    ierr = ISCreateGeneral(PETSC_COMM_SELF,numRows,rows,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyIS(mat->mapping,is,&newis);CHKERRQ(ierr);
    ierr = ISGetIndices(newis,&newRows);CHKERRQ(ierr);
    ierr = (*mat->ops->zerorows)(mat,numRows,newRows,diag);CHKERRQ(ierr);
    ierr = ISRestoreIndices(newis,&newRows);CHKERRQ(ierr);
    ierr = ISDestroy(newis);CHKERRQ(ierr);
    ierr = ISDestroy(is);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRowsLocalIS"
/*@C 
   MatZeroRowsLocalIS - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix; using local numbering of rows.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  is - index set of rows to remove
-  diag - value put in all diagonals of eliminated rows

   Notes:
   Before calling MatZeroRowsLocalIS(), the user must first set the
   local-to-global mapping by calling MatSetLocalToGlobalMapping().

   For the AIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   Level: intermediate

   Concepts: matrices^zeroing

.seealso: MatZeroRows(), MatZeroRowsLocal(), MatZeroEntries(), MatZeroRows(), MatSetLocalToGlobalMapping
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatZeroRowsLocalIS(Mat mat,IS is,PetscScalar diag)
{
  PetscErrorCode ierr;
  PetscInt       numRows;
  const PetscInt *rows;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,2);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = ISGetLocalSize(is,&numRows);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);
  ierr = MatZeroRowsLocal(mat,numRows,rows,diag);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&rows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSize"
/*@
   MatGetSize - Returns the numbers of rows and columns in a matrix.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  m - the number of global rows
-  n - the number of global columns

   Note: both output parameters can be PETSC_NULL on input.

   Level: beginner

   Concepts: matrices^size

.seealso: MatGetLocalSize()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetSize(Mat mat,PetscInt *m,PetscInt* n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (m) *m = mat->rmap->N;
  if (n) *n = mat->cmap->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetLocalSize"
/*@
   MatGetLocalSize - Returns the number of rows and columns in a matrix
   stored locally.  This information may be implementation dependent, so
   use with care.

   Not Collective

   Input Parameters:
.  mat - the matrix

   Output Parameters:
+  m - the number of local rows
-  n - the number of local columns

   Note: both output parameters can be PETSC_NULL on input.

   Level: beginner

   Concepts: matrices^local size

.seealso: MatGetSize()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetLocalSize(Mat mat,PetscInt *m,PetscInt* n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (m) PetscValidIntPointer(m,2);
  if (n) PetscValidIntPointer(n,3);
  if (m) *m = mat->rmap->n;
  if (n) *n = mat->cmap->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetOwnershipRangeColumn"
/*@
   MatGetOwnershipRangeColumn - Returns the range of matrix columns owned by
   this processor.

   Not Collective, unless matrix has not been allocated, then collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameters:
+  m - the global index of the first local column
-  n - one more than the global index of the last local column

   Notes: both output parameters can be PETSC_NULL on input.

   Level: developer

   Concepts: matrices^column ownership

.seealso:  MatGetOwnershipRange(), MatGetOwnershipRanges(), MatGetOwnershipRangesColumn()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRangeColumn(Mat mat,PetscInt *m,PetscInt* n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (m) PetscValidIntPointer(m,2);
  if (n) PetscValidIntPointer(n,3);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (m) *m = mat->cmap->rstart;
  if (n) *n = mat->cmap->rend;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetOwnershipRange"
/*@
   MatGetOwnershipRange - Returns the range of matrix rows owned by
   this processor, assuming that the matrix is laid out with the first
   n1 rows on the first processor, the next n2 rows on the second, etc.
   For certain parallel layouts this range may not be well defined.

   Not Collective, unless matrix has not been allocated, then collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameters:
+  m - the global index of the first local row
-  n - one more than the global index of the last local row

   Note: both output parameters can be PETSC_NULL on input.

   Level: beginner

   Concepts: matrices^row ownership

.seealso:   MatGetOwnershipRanges(), MatGetOwnershipRangeColumn(), MatGetOwnershipRangesColumn()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRange(Mat mat,PetscInt *m,PetscInt* n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (m) PetscValidIntPointer(m,2);
  if (n) PetscValidIntPointer(n,3);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (m) *m = mat->rmap->rstart;
  if (n) *n = mat->rmap->rend;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetOwnershipRanges"
/*@C
   MatGetOwnershipRanges - Returns the range of matrix rows owned by
   each process

   Not Collective, unless matrix has not been allocated, then collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  ranges - start of each processors portion plus one more then the total length at the end

   Level: beginner

   Concepts: matrices^row ownership

.seealso:   MatGetOwnershipRange(), MatGetOwnershipRangeColumn(), MatGetOwnershipRangesColumn()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRanges(Mat mat,const PetscInt **ranges)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(mat->rmap,ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetOwnershipRangesColumn"
/*@C
   MatGetOwnershipRangesColumn - Returns the range of local columns for each process

   Not Collective, unless matrix has not been allocated, then collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  ranges - start of each processors portion plus one more then the total length at the end

   Level: beginner

   Concepts: matrices^column ownership

.seealso:   MatGetOwnershipRange(), MatGetOwnershipRangeColumn(), MatGetOwnershipRanges()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRangesColumn(Mat mat,const PetscInt **ranges)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(mat->cmap,ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic"
/*@C
   MatILUFactorSymbolic - Performs symbolic ILU factorization of a matrix.
   Uses levels of fill only, not drop tolerance. Use MatLUFactorNumeric() 
   to complete the factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row permutation
.  column - column permutation
-  info - structure containing 
$      levels - number of levels of fill.
$      expected fill - as ratio of original fill.
$      1 or 0 - indicating force fill on diagonal (improves robustness for matrices
                missing diagonal entries)

   Output Parameters:
.  fact - new matrix that has been symbolically factored

   Notes:
   See the users manual for additional information about
   choosing the fill factor for better efficiency.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

  Concepts: matrices^symbolic LU factorization
  Concepts: matrices^factorization
  Concepts: LU^symbolic factorization

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
          MatGetOrdering(), MatFactorInfo

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatILUFactorSymbolic(Mat fact,Mat mat,IS row,IS col,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(row,IS_COOKIE,2);
  PetscValidHeaderSpecific(col,IS_COOKIE,3);
  PetscValidPointer(info,4);
  PetscValidPointer(fact,5);
  if (info->levels < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Levels of fill negative %D",(PetscInt)info->levels);
  if (info->fill < 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Expected fill less than 1.0 %G",info->fill);
  if (!(fact)->ops->ilufactorsymbolic) SETERRQ1(PETSC_ERR_SUP,"Matrix type %s  symbolic ILU",((PetscObject)mat)->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_ILUFactorSymbolic,mat,row,col,0);CHKERRQ(ierr);
  ierr = (fact->ops->ilufactorsymbolic)(fact,mat,row,col,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ILUFactorSymbolic,mat,row,col,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatICCFactorSymbolic"
/*@C
   MatICCFactorSymbolic - Performs symbolic incomplete
   Cholesky factorization for a symmetric matrix.  Use 
   MatCholeskyFactorNumeric() to complete the factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  perm - row and column permutation
-  info - structure containing 
$      levels - number of levels of fill.
$      expected fill - as ratio of original fill.

   Output Parameter:
.  fact - the factored matrix

   Notes:
   Most users should employ the KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

  Concepts: matrices^symbolic incomplete Cholesky factorization
  Concepts: matrices^factorization
  Concepts: Cholsky^symbolic factorization

.seealso: MatCholeskyFactorNumeric(), MatCholeskyFactor(), MatFactorInfo

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatICCFactorSymbolic(Mat fact,Mat mat,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(perm,IS_COOKIE,2);
  PetscValidPointer(info,3);
  PetscValidPointer(fact,4);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (info->levels < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Levels negative %D",(PetscInt) info->levels);
  if (info->fill < 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Expected fill less than 1.0 %G",info->fill);
  if (!(fact)->ops->iccfactorsymbolic) SETERRQ1(PETSC_ERR_SUP,"Matrix type %s  symbolic ICC",((PetscObject)mat)->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_ICCFactorSymbolic,mat,perm,0,0);CHKERRQ(ierr);
  ierr = (fact->ops->iccfactorsymbolic)(fact,mat,perm,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ICCFactorSymbolic,mat,perm,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetArray"
/*@C
   MatGetArray - Returns a pointer to the element values in the matrix.
   The result of this routine is dependent on the underlying matrix data
   structure, and may not even work for certain matrix types.  You MUST
   call MatRestoreArray() when you no longer need to access the array.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  v - the location of the values


   Fortran Note:
   This routine is used differently from Fortran, e.g.,
.vb
        Mat         mat
        PetscScalar mat_array(1)
        PetscOffset i_mat
        PetscErrorCode ierr
        call MatGetArray(mat,mat_array,i_mat,ierr)

  C  Access first local entry in matrix; note that array is
  C  treated as one dimensional
        value = mat_array(i_mat + 1)

        [... other code ...]
        call MatRestoreArray(mat,mat_array,i_mat,ierr)
.ve

   See the Fortran chapter of the users manual and 
   petsc/src/mat/examples/tests for details.

   Level: advanced

   Concepts: matrices^access array

.seealso: MatRestoreArray(), MatGetArrayF90(), MatGetRowIJ()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetArray(Mat mat,PetscScalar *v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(v,2);
  if (!mat->ops->getarray) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = (*mat->ops->getarray)(mat,v);CHKERRQ(ierr);
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreArray"
/*@C
   MatRestoreArray - Restores the matrix after MatGetArray() has been called.

   Not Collective

   Input Parameter:
+  mat - the matrix
-  v - the location of the values

   Fortran Note:
   This routine is used differently from Fortran, e.g.,
.vb
        Mat         mat
        PetscScalar mat_array(1)
        PetscOffset i_mat
        PetscErrorCode ierr
        call MatGetArray(mat,mat_array,i_mat,ierr)

  C  Access first local entry in matrix; note that array is
  C  treated as one dimensional
        value = mat_array(i_mat + 1)

        [... other code ...]
        call MatRestoreArray(mat,mat_array,i_mat,ierr)
.ve

   See the Fortran chapter of the users manual and 
   petsc/src/mat/examples/tests for details

   Level: advanced

.seealso: MatGetArray(), MatRestoreArrayF90()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreArray(Mat mat,PetscScalar *v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(v,2);
#if defined(PETSC_USE_DEBUG)
  CHKMEMQ;
#endif
  if (!mat->ops->restorearray) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->restorearray)(mat,v);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices"
/*@C
   MatGetSubMatrices - Extracts several submatrices from a matrix. If submat
   points to an array of valid matrices, they may be reused to store the new
   submatrices.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  n   - the number of submatrixes to be extracted (on this processor, may be zero)
.  irow, icol - index sets of rows and columns to extract
-  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.  submat - the array of submatrices

   Notes:
   MatGetSubMatrices() can extract ONLY sequential submatrices
   (from both sequential and parallel matrices). Use MatGetSubMatrix()
   to extract a parallel submatrix.

   When extracting submatrices from a parallel matrix, each processor can
   form a different submatrix by setting the rows and columns of its
   individual index sets according to the local submatrix desired.

   When finished using the submatrices, the user should destroy
   them with MatDestroyMatrices().

   MAT_REUSE_MATRIX can only be used when the nonzero structure of the 
   original matrix has not changed from that last call to MatGetSubMatrices().

   This routine creates the matrices in submat; you should NOT create them before
   calling it. It also allocates the array of matrix pointers submat.

   For BAIJ matrices the index sets must respect the block structure, that is if they
   request one row/column in a block, they must request all rows/columns that are in
   that block. For example, if the block size is 2 you cannot request just row 0 and 
   column 0.

   Fortran Note:
   The Fortran interface is slightly different from that given below; it 
   requires one to pass in  as submat a Mat (integer) array of size at least m.

   Level: advanced

   Concepts: matrices^accessing submatrices
   Concepts: submatrices

.seealso: MatDestroyMatrices(), MatGetSubMatrix(), MatGetRow(), MatGetDiagonal(), MatReuse
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetSubMatrices(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  PetscErrorCode ierr;
  PetscInt        i;
  PetscTruth      eq;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (n) {
    PetscValidPointer(irow,3);
    PetscValidHeaderSpecific(*irow,IS_COOKIE,3);
    PetscValidPointer(icol,4);
    PetscValidHeaderSpecific(*icol,IS_COOKIE,4);
  }
  PetscValidPointer(submat,6);
  if (n && scall == MAT_REUSE_MATRIX) {
    PetscValidPointer(*submat,6);
    PetscValidHeaderSpecific(**submat,MAT_COOKIE,6);
  }
  if (!mat->ops->getsubmatrices) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_GetSubMatrices,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->getsubmatrices)(mat,n,irow,icol,scall,submat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_GetSubMatrices,mat,0,0,0);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (mat->symmetric || mat->structurally_symmetric || mat->hermitian) {
      ierr = ISEqual(irow[i],icol[i],&eq);CHKERRQ(ierr);
      if (eq) {
	if (mat->symmetric){
	  ierr = MatSetOption((*submat)[i],MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
	} else if (mat->hermitian) {
	  ierr = MatSetOption((*submat)[i],MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
	} else if (mat->structurally_symmetric) {
	  ierr = MatSetOption((*submat)[i],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
	}
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroyMatrices"
/*@C
   MatDestroyMatrices - Destroys a set of matrices obtained with MatGetSubMatrices().

   Collective on Mat

   Input Parameters:
+  n - the number of local matrices
-  mat - the matrices (note that this is a pointer to the array of matrices, just to match the calling
                       sequence of MatGetSubMatrices())

   Level: advanced

    Notes: Frees not only the matrices, but also the array that contains the matrices
           In Fortran will not free the array.

.seealso: MatGetSubMatrices()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroyMatrices(PetscInt n,Mat *mat[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy negative number of matrices %D",n);
  PetscValidPointer(mat,2);
  for (i=0; i<n; i++) {
    ierr = MatDestroy((*mat)[i]);CHKERRQ(ierr);
  }
  /* memory is allocated even if n = 0 */
  ierr = PetscFree(*mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSeqNonzeroStructure"
/*@C
   MatGetSeqNonzeroStructure - Extracts the sequential nonzero structure from a matrix. 

   Collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameter:
.  matstruct - the sequential matrix with the nonzero structure of mat

  Level: intermediate

.seealso: MatDestroySeqNonzeroStructure(), MatGetSubMatrices(), MatDestroyMatrices()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetSeqNonzeroStructure(Mat mat,Mat *matstruct)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(matstruct,2);
  
  PetscValidType(mat,1);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (!mat->ops->getseqnonzerostructure) SETERRQ1(PETSC_ERR_SUP,"Not for matrix type %s\n",((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_GetSeqNonzeroStructure,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->getseqnonzerostructure)(mat,matstruct);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_GetSeqNonzeroStructure,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroySeqNonzeroStructure"
/*@C
   MatDestroySeqNonzeroStructure - Destroys matrix obtained with MatGetSeqNonzeroStructure().

   Collective on Mat

   Input Parameters:
.  mat - the matrix (note that this is a pointer to the array of matrices, just to match the calling
                       sequence of MatGetSequentialNonzeroStructure())

   Level: advanced

    Notes: Frees not only the matrices, but also the array that contains the matrices

.seealso: MatGetSeqNonzeroStructure()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroySeqNonzeroStructure(Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mat,1);
  ierr = MatDestroy(*mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap"
/*@
   MatIncreaseOverlap - Given a set of submatrices indicated by index sets,
   replaces the index sets by larger ones that represent submatrices with
   additional overlap.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  n   - the number of index sets
.  is  - the array of index sets (these index sets will changed during the call)
-  ov  - the additional overlap requested

   Level: developer

   Concepts: overlap
   Concepts: ASM^computing overlap

.seealso: MatGetSubMatrices()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatIncreaseOverlap(Mat mat,PetscInt n,IS is[],PetscInt ov)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Must have one or more domains, you have %D",n);
  if (n) {
    PetscValidPointer(is,3);
    PetscValidHeaderSpecific(*is,IS_COOKIE,3);
  }
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor)     SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (!ov) PetscFunctionReturn(0);
  if (!mat->ops->increaseoverlap) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = PetscLogEventBegin(MAT_IncreaseOverlap,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->increaseoverlap)(mat,n,is,ov);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_IncreaseOverlap,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetBlockSize"
/*@
   MatGetBlockSize - Returns the matrix block size; useful especially for the
   block row and block diagonal formats.
   
   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  bs - block size

   Notes:
   Block row formats are MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ

   Level: intermediate

   Concepts: matrices^block size

.seealso: MatCreateSeqBAIJ(), MatCreateMPIBAIJ()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetBlockSize(Mat mat,PetscInt *bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(bs,2);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  *bs = mat->rmap->bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetBlockSize"
/*@
   MatSetBlockSize - Sets the matrix block size; for many matrix types you 
     cannot use this and MUST set the blocksize when you preallocate the matrix
   
   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  bs - block size

   Notes:
     For BAIJ matrices, this just checks that the block size agrees with the BAIJ size,
     it is not possible to change BAIJ block sizes after preallocation.

   Level: intermediate

   Concepts: matrices^block size

.seealso: MatCreateSeqBAIJ(), MatCreateMPIBAIJ(), MatGetBlockSize()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetBlockSize(Mat mat,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (bs < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Block size %d, must be positive",bs);
  if (mat->ops->setblocksize) {
    ierr = (*mat->ops->setblocksize)(mat,bs);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_ARG_INCOMP,"Cannot set the blocksize for matrix type %s",((PetscObject)mat)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowIJ"
/*@C
    MatGetRowIJ - Returns the compressed row storage i and j indices for sequential matrices.

   Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift -  0 or 1 indicating we want the indices starting at 0 or 1
.   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized
-   inodecompressed - PETSC_TRUE or PETSC_FALSE  indicating if the nonzero structure of the
                 inodes or the nonzero elements is wanted. For BAIJ matrices the compressed version is 
                 always used.

    Output Parameters:
+   n - number of rows in the (possibly compressed) matrix
.   ia - the row pointers [of length n+1]
.   ja - the column indices
-   done - indicates if the routine actually worked and returned appropriate ia[] and ja[] arrays; callers
           are responsible for handling the case when done == PETSC_FALSE and ia and ja are not set

    Level: developer

    Notes: You CANNOT change any of the ia[] or ja[] values.

           Use MatRestoreRowIJ() when you are finished accessing the ia[] and ja[] values

    Fortran Node

           In Fortran use
$           PetscInt ia(1), ja(1)
$           PetscOffset iia, jja
$      call MatGetRowIJ(mat,shift,symmetric,inodecompressed,n,ia,iia,ja,jja,done,ierr)
$
$          or 
$
$           PetscScalar, pointer :: xx_v(:)
$    call  MatGetRowIJF90(mat,shift,symmetric,inodecompressed,n,ia,ja,done,ierr)
  
 
       Acess the ith and jth entries via ia(iia + i) and ja(jja + j)

.seealso: MatGetColumnIJ(), MatRestoreRowIJ(), MatGetArray()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRowIJ(Mat mat,PetscInt shift,PetscTruth symmetric,PetscTruth inodecompressed,PetscInt *n,PetscInt *ia[],PetscInt* ja[],PetscTruth *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(n,4);
  if (ia) PetscValidIntPointer(ia,5);
  if (ja) PetscValidIntPointer(ja,6);
  PetscValidIntPointer(done,7);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (!mat->ops->getrowij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr = PetscLogEventBegin(MAT_GetRowIJ,mat,0,0,0);CHKERRQ(ierr);
    ierr  = (*mat->ops->getrowij)(mat,shift,symmetric,inodecompressed,n,ia,ja,done);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_GetRowIJ,mat,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnIJ"
/*@C
    MatGetColumnIJ - Returns the compressed column storage i and j indices for sequential matrices.

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
.   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized
-   inodecompressed - PETSC_TRUE or PETSC_FALSE indicating if the nonzero structure of the
                 inodes or the nonzero elements is wanted. For BAIJ matrices the compressed version is 
                 always used.

    Output Parameters:
+   n - number of columns in the (possibly compressed) matrix
.   ia - the column pointers
.   ja - the row indices
-   done - PETSC_TRUE or PETSC_FALSE, indicating whether the values have been returned

    Level: developer

.seealso: MatGetRowIJ(), MatRestoreColumnIJ()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetColumnIJ(Mat mat,PetscInt shift,PetscTruth symmetric,PetscTruth inodecompressed,PetscInt *n,PetscInt *ia[],PetscInt* ja[],PetscTruth *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(n,4);
  if (ia) PetscValidIntPointer(ia,5);
  if (ja) PetscValidIntPointer(ja,6);
  PetscValidIntPointer(done,7);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (!mat->ops->getcolumnij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->getcolumnij)(mat,shift,symmetric,inodecompressed,n,ia,ja,done);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowIJ"
/*@C
    MatRestoreRowIJ - Call after you are completed with the ia,ja indices obtained with
    MatGetRowIJ().

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
.   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized
-   inodecompressed -  PETSC_TRUE or PETSC_FALSE indicating if the nonzero structure of the
                 inodes or the nonzero elements is wanted. For BAIJ matrices the compressed version is 
                 always used.

    Output Parameters:
+   n - size of (possibly compressed) matrix
.   ia - the row pointers
.   ja - the column indices
-   done - PETSC_TRUE or PETSC_FALSE indicated that the values have been returned

    Level: developer

.seealso: MatGetRowIJ(), MatRestoreColumnIJ()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreRowIJ(Mat mat,PetscInt shift,PetscTruth symmetric,PetscTruth inodecompressed,PetscInt *n,PetscInt *ia[],PetscInt* ja[],PetscTruth *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (ia) PetscValidIntPointer(ia,5);
  if (ja) PetscValidIntPointer(ja,6);
  PetscValidIntPointer(done,7);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (!mat->ops->restorerowij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->restorerowij)(mat,shift,symmetric,inodecompressed,n,ia,ja,done);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreColumnIJ"
/*@C
    MatRestoreColumnIJ - Call after you are completed with the ia,ja indices obtained with
    MatGetColumnIJ().

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
-   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized
-   inodecompressed - PETSC_TRUE or PETSC_FALSE indicating if the nonzero structure of the
                 inodes or the nonzero elements is wanted. For BAIJ matrices the compressed version is 
                 always used.

    Output Parameters:
+   n - size of (possibly compressed) matrix
.   ia - the column pointers
.   ja - the row indices
-   done - PETSC_TRUE or PETSC_FALSE indicated that the values have been returned

    Level: developer

.seealso: MatGetColumnIJ(), MatRestoreRowIJ()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRestoreColumnIJ(Mat mat,PetscInt shift,PetscTruth symmetric,PetscTruth inodecompressed,PetscInt *n,PetscInt *ia[],PetscInt* ja[],PetscTruth *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (ia) PetscValidIntPointer(ia,5);
  if (ja) PetscValidIntPointer(ja,6);
  PetscValidIntPointer(done,7);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (!mat->ops->restorecolumnij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->restorecolumnij)(mat,shift,symmetric,inodecompressed,n,ia,ja,done);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatColoringPatch"
/*@C
    MatColoringPatch -Used inside matrix coloring routines that 
    use MatGetRowIJ() and/or MatGetColumnIJ().

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   ncolors - max color value
.   n   - number of entries in colorarray
-   colorarray - array indicating color for each column

    Output Parameters:
.   iscoloring - coloring generated using colorarray information

    Level: developer

.seealso: MatGetRowIJ(), MatGetColumnIJ()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatColoringPatch(Mat mat,PetscInt ncolors,PetscInt n,ISColoringValue colorarray[],ISColoring *iscoloring)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(colorarray,4);
  PetscValidPointer(iscoloring,5);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  if (!mat->ops->coloringpatch){
    ierr = ISColoringCreate(((PetscObject)mat)->comm,ncolors,n,colorarray,iscoloring);CHKERRQ(ierr);
  } else {
    ierr = (*mat->ops->coloringpatch)(mat,ncolors,n,colorarray,iscoloring);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSetUnfactored"
/*@
   MatSetUnfactored - Resets a factored matrix to be treated as unfactored.

   Collective on Mat

   Input Parameter:
.  mat - the factored matrix to be reset

   Notes: 
   This routine should be used only with factored matrices formed by in-place
   factorization via ILU(0) (or by in-place LU factorization for the MATSEQDENSE
   format).  This option can save memory, for example, when solving nonlinear
   systems with a matrix-free Newton-Krylov method and a matrix-based, in-place
   ILU(0) preconditioner.  

   Note that one can specify in-place ILU(0) factorization by calling 
.vb
     PCType(pc,PCILU);
     PCFactorSeUseInPlace(pc);
.ve
   or by using the options -pc_type ilu -pc_factor_in_place

   In-place factorization ILU(0) can also be used as a local
   solver for the blocks within the block Jacobi or additive Schwarz
   methods (runtime option: -sub_pc_factor_in_place).  See the discussion 
   of these preconditioners in the users manual for details on setting
   local solver options.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: PCFactorSetUseInPlace()

   Concepts: matrices^unfactored

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetUnfactored(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);  
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  mat->factor = MAT_FACTOR_NONE;
  if (!mat->ops->setunfactored) PetscFunctionReturn(0);
  ierr = (*mat->ops->setunfactored)(mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
    MatGetArrayF90 - Accesses a matrix array from Fortran90.

    Synopsis:
    MatGetArrayF90(Mat x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameter:
.   x - matrix

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage: 
.vb
      PetscScalar, pointer xx_v(:)
      ....
      call MatGetArrayF90(x,xx_v,ierr)
      a = xx_v(3)
      call MatRestoreArrayF90(x,xx_v,ierr)
.ve

    Notes:
    Not yet supported for all F90 compilers

    Level: advanced

.seealso:  MatRestoreArrayF90(), MatGetArray(), MatRestoreArray()

    Concepts: matrices^accessing array

M*/

/*MC
    MatRestoreArrayF90 - Restores a matrix array that has been
    accessed with MatGetArrayF90().

    Synopsis:
    MatRestoreArrayF90(Mat x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameters:
+   x - matrix
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage: 
.vb
       PetscScalar, pointer xx_v(:)
       ....
       call MatGetArrayF90(x,xx_v,ierr)
       a = xx_v(3)
       call MatRestoreArrayF90(x,xx_v,ierr)
.ve
   
    Notes:
    Not yet supported for all F90 compilers

    Level: advanced

.seealso:  MatGetArrayF90(), MatGetArray(), MatRestoreArray()

M*/


#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix"
/*@
    MatGetSubMatrix - Gets a single submatrix on the same number of processors
                      as the original matrix.

    Collective on Mat

    Input Parameters:
+   mat - the original matrix
.   isrow - parallel IS containing the rows this processor should obtain
.   iscol - parallel IS containing all columns you wish to keep. Each process should list the columns that will be in IT's "diagonal part" in the new matrix.
-   cll - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

    Output Parameter:
.   newmat - the new submatrix, of the same type as the old

    Level: advanced

    Notes:
    The submatrix will be able to be multiplied with vectors using the same layout as iscol.

    The rows in isrow will be sorted into the same order as the original matrix on each process.

      The first time this is called you should use a cll of MAT_INITIAL_MATRIX,
   the MatGetSubMatrix() routine will create the newmat for you. Any additional calls
   to this routine with a mat of the same nonzero structure and with a call of MAT_REUSE_MATRIX  
   will reuse the matrix generated the first time.  You should call MatDestroy() on newmat when 
   you are finished using it.

    The communicator of the newly obtained matrix is ALWAYS the same as the communicator of
    the input matrix.

    If iscol is PETSC_NULL then all columns are obtained (not supported in Fortran).

   Example usage:
   Consider the following 8x8 matrix with 34 non-zero values, that is
   assembled across 3 processors. Lets assume that proc0 owns 3 rows,
   proc1 owns 3 rows, proc2 owns 2 rows. This division can be shown
   as follows:

.vb
            1  2  0  |  0  3  0  |  0  4
    Proc0   0  5  6  |  7  0  0  |  8  0
            9  0 10  | 11  0  0  | 12  0
    -------------------------------------
           13  0 14  | 15 16 17  |  0  0
    Proc1   0 18  0  | 19 20 21  |  0  0
            0  0  0  | 22 23  0  | 24  0
    -------------------------------------
    Proc2  25 26 27  |  0  0 28  | 29  0
           30  0  0  | 31 32 33  |  0 34
.ve

    Suppose isrow = [0 1 | 4 | 6 7] and iscol = [1 2 | 3 4 5 | 6].  The resulting submatrix is

.vb
            2  0  |  0  3  0  |  0
    Proc0   5  6  |  7  0  0  |  8
    -------------------------------
    Proc1  18  0  | 19 20 21  |  0
    -------------------------------
    Proc2  26 27  |  0  0 28  | 29
            0  0  | 31 32 33  |  0
.ve


    Concepts: matrices^submatrices

.seealso: MatGetSubMatrices()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetSubMatrix(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat            *local;
  IS             iscoltmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(isrow,IS_COOKIE,2);
  if (iscol) PetscValidHeaderSpecific(iscol,IS_COOKIE,3);
  PetscValidPointer(newmat,5);
  if (cll == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newmat,MAT_COOKIE,5);
  PetscValidType(mat,1);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)mat)->comm,&size);CHKERRQ(ierr);

  if (!iscol) {
    ierr = ISCreateStride(((PetscObject)mat)->comm,mat->cmap->n,mat->cmap->rstart,1,&iscoltmp);CHKERRQ(ierr);
  } else {
    iscoltmp = iscol;
  }

  /* if original matrix is on just one processor then use submatrix generated */
  if (mat->ops->getsubmatrices && !mat->ops->getsubmatrix && size == 1 && cll == MAT_REUSE_MATRIX) {
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscoltmp,MAT_REUSE_MATRIX,&newmat);CHKERRQ(ierr);
    if (!iscol) {ierr = ISDestroy(iscoltmp);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  } else if (mat->ops->getsubmatrices && !mat->ops->getsubmatrix && size == 1) {
    ierr    = MatGetSubMatrices(mat,1,&isrow,&iscoltmp,MAT_INITIAL_MATRIX,&local);CHKERRQ(ierr);
    *newmat = *local;
    ierr    = PetscFree(local);CHKERRQ(ierr);
    if (!iscol) {ierr = ISDestroy(iscoltmp);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  } else if (!mat->ops->getsubmatrix) {
    /* Create a new matrix type that implements the operation using the full matrix */
    switch (cll) {
      case MAT_INITIAL_MATRIX:
        ierr = MatCreateSubMatrix(mat,isrow,iscoltmp,newmat);CHKERRQ(ierr);
        break;
      case MAT_REUSE_MATRIX:
        ierr = MatSubMatrixUpdate(*newmat,mat,isrow,iscoltmp);CHKERRQ(ierr);
        break;
      default: SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid MatReuse, must be either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX");
    }
    if (!iscol) {ierr = ISDestroy(iscoltmp);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }

  if (!mat->ops->getsubmatrix) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->getsubmatrix)(mat,isrow,iscoltmp,cll,newmat);CHKERRQ(ierr);
  if (!iscol) {ierr = ISDestroy(iscoltmp);CHKERRQ(ierr);}
  ierr = PetscObjectStateIncrease((PetscObject)*newmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatStashSetInitialSize"
/*@
   MatStashSetInitialSize - sets the sizes of the matrix stash, that is
   used during the assembly process to store values that belong to 
   other processors.

   Not Collective

   Input Parameters:
+  mat   - the matrix
.  size  - the initial size of the stash.
-  bsize - the initial size of the block-stash(if used).

   Options Database Keys:
+   -matstash_initial_size <size> or <size0,size1,...sizep-1>
-   -matstash_block_initial_size <bsize>  or <bsize0,bsize1,...bsizep-1>

   Level: intermediate

   Notes: 
     The block-stash is used for values set with MatSetValuesBlocked() while
     the stash is used for values set with MatSetValues()

     Run with the option -info and look for output of the form
     MatAssemblyBegin_MPIXXX:Stash has MM entries, uses nn mallocs.
     to determine the appropriate value, MM, to use for size and 
     MatAssemblyBegin_MPIXXX:Block-Stash has BMM entries, uses nn mallocs.
     to determine the value, BMM to use for bsize

   Concepts: stash^setting matrix size
   Concepts: matrices^stash

.seealso: MatAssemblyBegin(), MatAssemblyEnd(), Mat, MatStashGetInfo()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatStashSetInitialSize(Mat mat,PetscInt size, PetscInt bsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatStashSetInitialSize_Private(&mat->stash,size);CHKERRQ(ierr);
  ierr = MatStashSetInitialSize_Private(&mat->bstash,bsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInterpolateAdd"
/*@
   MatInterpolateAdd - w = y + A*x or A'*x depending on the shape of 
     the matrix

   Collective on Mat

   Input Parameters:
+  mat   - the matrix
.  x,y - the vectors
-  w - where the result is stored

   Level: intermediate

   Notes: 
    w may be the same vector as y. 

    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the interpolation

    Concepts: interpolation

.seealso: MatMultAdd(), MatMultTransposeAdd(), MatRestrict()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatInterpolateAdd(Mat A,Vec x,Vec y,Vec w)
{
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidHeaderSpecific(w,VEC_COOKIE,4);
  PetscValidType(A,1);
  ierr = MatPreallocated(A);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  if (N > M) {
    ierr = MatMultTransposeAdd(A,x,y,w);CHKERRQ(ierr);
  } else {
    ierr = MatMultAdd(A,x,y,w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInterpolate"
/*@
   MatInterpolate - y = A*x or A'*x depending on the shape of 
     the matrix

   Collective on Mat

   Input Parameters:
+  mat   - the matrix
-  x,y - the vectors

   Level: intermediate

   Notes: 
    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the interpolation

   Concepts: matrices^interpolation

.seealso: MatMultAdd(), MatMultTransposeAdd(), MatRestrict()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatInterpolate(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(A,1);
  ierr = MatPreallocated(A);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  if (N > M) {
    ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestrict"
/*@
   MatRestrict - y = A*x or A'*x

   Collective on Mat

   Input Parameters:
+  mat   - the matrix
-  x,y - the vectors

   Level: intermediate

   Notes: 
    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the restriction

   Concepts: matrices^restriction

.seealso: MatMultAdd(), MatMultTransposeAdd(), MatInterpolate()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRestrict(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(A,1);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  if (N > M) {
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceAttach"
/*@
   MatNullSpaceAttach - attaches a null space to a matrix.
        This null space will be removed from the resulting vector whenever
        MatMult() is called

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  nullsp - the null space object

   Level: developer

   Notes:
      Overwrites any previous null space that may have been attached

   Concepts: null space^attaching to matrix

.seealso: MatCreate(), MatNullSpaceCreate()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceAttach(Mat mat,MatNullSpace nullsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(nullsp,MAT_NULLSPACE_COOKIE,2);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)nullsp);CHKERRQ(ierr);
  if (mat->nullsp) { ierr = MatNullSpaceDestroy(mat->nullsp);CHKERRQ(ierr); }
  mat->nullsp = nullsp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatICCFactor"
/*@C
   MatICCFactor - Performs in-place incomplete Cholesky factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row/column permutation
.  fill - expected fill factor >= 1.0
-  level - level of fill, for ICC(k)

   Notes: 
   Probably really in-place only when level of fill is zero, otherwise allocates
   new space to store factored matrix and deletes previous memory.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^incomplete Cholesky factorization
   Concepts: Cholesky factorization

.seealso: MatICCFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatICCFactor(Mat mat,IS row,const MatFactorInfo* info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (row) PetscValidHeaderSpecific(row,IS_COOKIE,2);
  PetscValidPointer(info,3);
  if (mat->rmap->N != mat->cmap->N) SETERRQ(PETSC_ERR_ARG_WRONG,"matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->iccfactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = (*mat->ops->iccfactor)(mat,row,info);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdic"
/*@ 
   MatSetValuesAdic - Sets values computed with ADIC automatic differentiation into a matrix.

   Not Collective

   Input Parameters:
+  mat - the matrix
-  v - the values compute with ADIC

   Level: developer

   Notes:
     Must call MatSetColoring() before using this routine. Also this matrix must already
     have its nonzero pattern determined.

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal(),
          MatSetValues(), MatSetColoring(), MatSetValuesAdifor()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesAdic(Mat mat,void *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(mat,2);

  if (!mat->assembled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be already assembled");
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (!mat->ops->setvaluesadic) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->setvaluesadic)(mat,v);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  ierr = MatView_Private(mat);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSetColoring"
/*@ 
   MatSetColoring - Sets a coloring used by calls to MatSetValuesAdic()

   Not Collective

   Input Parameters:
+  mat - the matrix
-  coloring - the coloring

   Level: developer

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal(),
          MatSetValues(), MatSetValuesAdic()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetColoring(Mat mat,ISColoring coloring)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(coloring,2);

  if (!mat->assembled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be already assembled");
  }
  if (!mat->ops->setcoloring) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->setcoloring)(mat,coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdifor"
/*@ 
   MatSetValuesAdifor - Sets values computed with automatic differentiation into a matrix.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  nl - leading dimension of v
-  v - the values compute with ADIFOR

   Level: developer

   Notes:
     Must call MatSetColoring() before using this routine. Also this matrix must already
     have its nonzero pattern determined.

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal(),
          MatSetValues(), MatSetColoring()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetValuesAdifor(Mat mat,PetscInt nl,void *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(v,3);

  if (!mat->assembled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be already assembled");
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (!mat->ops->setvaluesadifor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->setvaluesadifor)(mat,nl,v);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScaleLocal"
/*@ 
   MatDiagonalScaleLocal - Scales columns of a matrix given the scaling values including the 
         ghosted ones.

   Not Collective

   Input Parameters:
+  mat - the matrix
-  diag = the diagonal values, including ghost ones

   Level: developer

   Notes: Works only for MPIAIJ and MPIBAIJ matrices
      
.seealso: MatDiagonalScale()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatDiagonalScaleLocal(Mat mat,Vec diag)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(diag,VEC_COOKIE,2);
  PetscValidType(mat,1);

  if (!mat->assembled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be already assembled");
  }
  ierr = PetscLogEventBegin(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)mat)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    PetscInt n,m;
    ierr = VecGetSize(diag,&n);CHKERRQ(ierr);
    ierr = MatGetSize(mat,0,&m);CHKERRQ(ierr);
    if (m == n) {
      ierr = MatDiagonalScale(mat,0,diag);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Only supported for sequential matrices when no ghost points/periodic conditions");
    }
  } else {
    PetscErrorCode (*f)(Mat,Vec);
    ierr = PetscObjectQueryFunction((PetscObject)mat,"MatDiagonalScaleLocal_C",(void (**)(void))&f);CHKERRQ(ierr);
    if (f) {
      ierr = (*f)(mat,diag);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Only supported for MPIAIJ and MPIBAIJ parallel matrices");
    }
  }
  ierr = PetscLogEventEnd(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInertia"
/*@ 
   MatGetInertia - Gets the inertia from a factored matrix

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+   nneg - number of negative eigenvalues
.   nzero - number of zero eigenvalues
-   npos - number of positive eigenvalues

   Level: advanced

   Notes: Matrix must have been factored by MatCholeskyFactor()


@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetInertia(Mat mat,PetscInt *nneg,PetscInt *nzero,PetscInt *npos)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (!mat->factor)    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Numeric factor mat is not assembled");
  if (!mat->ops->getinertia) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = (*mat->ops->getinertia)(mat,nneg,nzero,npos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatSolves"
/*@C
   MatSolves - Solves A x = b, given a factored matrix, for a collection of vectors

   Collective on Mat and Vecs

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vectors

   Output Parameter:
.  x - the result vectors

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolves(A,x,x).

   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolveAdd(), MatSolveTranspose(), MatSolveTransposeAdd(), MatSolve()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSolves(Mat mat,Vecs b,Vecs x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);

  if (!mat->ops->solves) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_Solves,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->solves)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Solves,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsSymmetric"
/*@
   MatIsSymmetric - Test whether a matrix is symmetric

   Collective on Mat

   Input Parameter:
+  A - the matrix to test
-  tol - difference between value and its transpose less than this amount counts as equal (use 0.0 for exact transpose)

   Output Parameters:
.  flg - the result

   Level: intermediate

   Concepts: matrix^symmetry

.seealso: MatTranspose(), MatIsTranspose(), MatIsHermitian(), MatIsStructurallySymmetric(), MatSetOption(), MatIsSymmetricKnown()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatIsSymmetric(Mat A,PetscReal tol,PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidPointer(flg,2);

  if (!A->symmetric_set) {
    if (!A->ops->issymmetric) {
      const MatType mattype;
      ierr = MatGetType(A,&mattype);CHKERRQ(ierr);
      SETERRQ1(PETSC_ERR_SUP,"Matrix of type <%s> does not support checking for symmetric",mattype);
    }
    ierr = (*A->ops->issymmetric)(A,tol,flg);CHKERRQ(ierr);
    if (!tol) {
      A->symmetric_set = PETSC_TRUE;
      A->symmetric = *flg;
      if (A->symmetric) {
	A->structurally_symmetric_set = PETSC_TRUE;
	A->structurally_symmetric     = PETSC_TRUE;
      }
    }
  } else if (A->symmetric) {
    *flg = PETSC_TRUE;
  } else if (!tol) {
    *flg = PETSC_FALSE;
  } else {
    if (!A->ops->issymmetric) {
      const MatType mattype;
      ierr = MatGetType(A,&mattype);CHKERRQ(ierr);
      SETERRQ1(PETSC_ERR_SUP,"Matrix of type <%s> does not support checking for symmetric",mattype);
    }
    ierr = (*A->ops->issymmetric)(A,tol,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsHermitian"
/*@
   MatIsHermitian - Test whether a matrix is Hermitian

   Collective on Mat

   Input Parameter:
+  A - the matrix to test
-  tol - difference between value and its transpose less than this amount counts as equal (use 0.0 for exact Hermitian)

   Output Parameters:
.  flg - the result

   Level: intermediate

   Concepts: matrix^symmetry

.seealso: MatTranspose(), MatIsTranspose(), MatIsHermitian(), MatIsStructurallySymmetric(), MatSetOption(), MatIsSymmetricKnown()
@*/ 
PetscErrorCode PETSCMAT_DLLEXPORT MatIsHermitian(Mat A,PetscReal tol,PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidPointer(flg,2);

  if (!A->hermitian_set) {
    if (!A->ops->ishermitian) {
      const MatType mattype;
      ierr = MatGetType(A,&mattype);CHKERRQ(ierr);
      SETERRQ1(PETSC_ERR_SUP,"Matrix of type <%s> does not support checking for hermitian",mattype);
    }
    ierr = (*A->ops->ishermitian)(A,tol,flg);CHKERRQ(ierr);
    if (!tol) {
      A->hermitian_set = PETSC_TRUE;
      A->hermitian = *flg;
      if (A->hermitian) {
	A->structurally_symmetric_set = PETSC_TRUE;
	A->structurally_symmetric     = PETSC_TRUE;
      }
    }
  } else if (A->hermitian) {
    *flg = PETSC_TRUE;
  } else if (!tol) {
    *flg = PETSC_FALSE;
  } else {
    if (!A->ops->ishermitian) {
      const MatType mattype;
      ierr = MatGetType(A,&mattype);CHKERRQ(ierr);
      SETERRQ1(PETSC_ERR_SUP,"Matrix of type <%s> does not support checking for hermitian",mattype);
    }
    ierr = (*A->ops->ishermitian)(A,tol,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsSymmetricKnown"
/*@
   MatIsSymmetricKnown - Checks the flag on the matrix to see if it is symmetric.

   Collective on Mat

   Input Parameter:
.  A - the matrix to check

   Output Parameters:
+  set - if the symmetric flag is set (this tells you if the next flag is valid)
-  flg - the result

   Level: advanced

   Concepts: matrix^symmetry

   Note: Does not check the matrix values directly, so this may return unknown (set = PETSC_FALSE). Use MatIsSymmetric()
         if you want it explicitly checked

.seealso: MatTranspose(), MatIsTranspose(), MatIsHermitian(), MatIsStructurallySymmetric(), MatSetOption(), MatIsSymmetric()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatIsSymmetricKnown(Mat A,PetscTruth *set,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidPointer(set,2);
  PetscValidPointer(flg,3);
  if (A->symmetric_set) {
    *set = PETSC_TRUE;
    *flg = A->symmetric;
  } else {
    *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsHermitianKnown"
/*@
   MatIsHermitianKnown - Checks the flag on the matrix to see if it is hermitian.

   Collective on Mat

   Input Parameter:
.  A - the matrix to check

   Output Parameters:
+  set - if the hermitian flag is set (this tells you if the next flag is valid)
-  flg - the result

   Level: advanced

   Concepts: matrix^symmetry

   Note: Does not check the matrix values directly, so this may return unknown (set = PETSC_FALSE). Use MatIsHermitian()
         if you want it explicitly checked

.seealso: MatTranspose(), MatIsTranspose(), MatIsHermitian(), MatIsStructurallySymmetric(), MatSetOption(), MatIsSymmetric()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatIsHermitianKnown(Mat A,PetscTruth *set,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidPointer(set,2);
  PetscValidPointer(flg,3);
  if (A->hermitian_set) {
    *set = PETSC_TRUE;
    *flg = A->hermitian;
  } else {
    *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIsStructurallySymmetric"
/*@
   MatIsStructurallySymmetric - Test whether a matrix is structurally symmetric

   Collective on Mat

   Input Parameter:
.  A - the matrix to test

   Output Parameters:
.  flg - the result

   Level: intermediate

   Concepts: matrix^symmetry

.seealso: MatTranspose(), MatIsTranspose(), MatIsHermitian(), MatIsSymmetric(), MatSetOption()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatIsStructurallySymmetric(Mat A,PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidPointer(flg,2);
  if (!A->structurally_symmetric_set) {
    if (!A->ops->isstructurallysymmetric) SETERRQ(PETSC_ERR_SUP,"Matrix does not support checking for structural symmetric");
    ierr = (*A->ops->isstructurallysymmetric)(A,&A->structurally_symmetric);CHKERRQ(ierr);
    A->structurally_symmetric_set = PETSC_TRUE;
  }
  *flg = A->structurally_symmetric;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatStashGetInfo"
extern PetscErrorCode MatStashGetInfo_Private(MatStash*,PetscInt*,PetscInt*);
/*@ 
   MatStashGetInfo - Gets how many values are currently in the vector stash, i.e. need
       to be communicated to other processors during the MatAssemblyBegin/End() process

    Not collective

   Input Parameter:
.   vec - the vector

   Output Parameters:
+   nstash   - the size of the stash
.   reallocs - the number of additional mallocs incurred.
.   bnstash   - the size of the block stash
-   breallocs - the number of additional mallocs incurred.in the block stash
 
   Level: advanced

.seealso: MatAssemblyBegin(), MatAssemblyEnd(), Mat, MatStashSetInitialSize()
  
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatStashGetInfo(Mat mat,PetscInt *nstash,PetscInt *reallocs,PetscInt *bnstash,PetscInt *breallocs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatStashGetInfo_Private(&mat->stash,nstash,reallocs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->bstash,bnstash,breallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetVecs"
/*@C
   MatGetVecs - Get vector(s) compatible with the matrix, i.e. with the same 
     parallel layout
   
   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameter:
+   right - (optional) vector that the matrix can be multiplied against
-   left - (optional) vector that the matrix vector product can be stored in

  Level: advanced

.seealso: MatCreate()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetVecs(Mat mat,Vec *right,Vec *left)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->ops->getvecs) {
    ierr = (*mat->ops->getvecs)(mat,right,left);CHKERRQ(ierr);
  } else {
    PetscMPIInt size;
    ierr = MPI_Comm_size(((PetscObject)mat)->comm, &size);CHKERRQ(ierr);
    if (right) {
      ierr = VecCreate(((PetscObject)mat)->comm,right);CHKERRQ(ierr);
      ierr = VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = VecSetBlockSize(*right,mat->rmap->bs);CHKERRQ(ierr);
      if (size > 1) {
        /* New vectors uses Mat cmap and does not create a new one */
	ierr = PetscLayoutDestroy((*right)->map);CHKERRQ(ierr);
	(*right)->map = mat->cmap;
	mat->cmap->refcnt++;

        ierr = VecSetType(*right,VECMPI);CHKERRQ(ierr);
      } else {ierr = VecSetType(*right,VECSEQ);CHKERRQ(ierr);}
    }
    if (left) {
      ierr = VecCreate(((PetscObject)mat)->comm,left);CHKERRQ(ierr);
      ierr = VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = VecSetBlockSize(*left,mat->rmap->bs);CHKERRQ(ierr);
      if (size > 1) {
        /* New vectors uses Mat rmap and does not create a new one */
	ierr = PetscLayoutDestroy((*left)->map);CHKERRQ(ierr);
	(*left)->map = mat->rmap;
	mat->rmap->refcnt++;

        ierr = VecSetType(*left,VECMPI);CHKERRQ(ierr);
      } else {ierr = VecSetType(*left,VECSEQ);CHKERRQ(ierr);}
    }
  }
  if (mat->mapping) {
    if (right) {ierr = VecSetLocalToGlobalMapping(*right,mat->mapping);CHKERRQ(ierr);}
    if (left) {ierr = VecSetLocalToGlobalMapping(*left,mat->mapping);CHKERRQ(ierr);}
  }
  if (mat->bmapping) {
    if (right) {ierr = VecSetLocalToGlobalMappingBlock(*right,mat->bmapping);CHKERRQ(ierr);}
    if (left) {ierr = VecSetLocalToGlobalMappingBlock(*left,mat->bmapping);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfoInitialize"
/*@C
   MatFactorInfoInitialize - Initializes a MatFactorInfo data structure
     with default values.

   Not Collective

   Input Parameters:
.    info - the MatFactorInfo data structure


   Notes: The solvers are generally used through the KSP and PC objects, for example
          PCLU, PCILU, PCCHOLESKY, PCICC

   Level: developer

.seealso: MatFactorInfo

    Developer Note: fortran interface is not autogenerated as the f90
    interface defintion cannot be generated correctly [due to MatFactorInfo]

@*/

PetscErrorCode PETSCMAT_DLLEXPORT MatFactorInfoInitialize(MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(info,sizeof(MatFactorInfo));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAP"
/*@
   MatPtAP - Creates the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
.  P - the projection matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/nnz(A) 

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of AIJ matrices and classes
   which inherit from AIJ.  

   Level: intermediate

.seealso: MatPtAPSymbolic(), MatPtAPNumeric(), MatMatMult()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPtAP(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidHeaderSpecific(P,MAT_COOKIE,2);
  PetscValidType(P,2);
  ierr = MatPreallocated(P);CHKERRQ(ierr);
  if (!P->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (P->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidPointer(C,3);
  if (P->rmap->N!=A->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->rmap->N,A->cmap->N);
  if (fill < 1.0) SETERRQ1(PETSC_ERR_ARG_SIZ,"Expected fill=%G must be >= 1.0",fill);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  if (!A->ops->ptap) {
    const MatType mattype;
    ierr = MatGetType(A,&mattype);CHKERRQ(ierr);
    SETERRQ1(PETSC_ERR_SUP,"Matrix of type <%s> does not support PtAP",mattype);
  }
  ierr = PetscLogEventBegin(MAT_PtAP,A,P,0,0);CHKERRQ(ierr); 
  ierr = (*A->ops->ptap)(A,P,scall,fill,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAP,A,P,0,0);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric"
/*@
   MatPtAPNumeric - Computes the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  P - the projection matrix

   Output Parameters:
.  C - the product matrix

   Notes:
   C must have been created by calling MatPtAPSymbolic and must be destroyed by
   the user using MatDeatroy().

   This routine is currently only implemented for pairs of AIJ matrices and classes
   which inherit from AIJ.  C will be of type MATAIJ.

   Level: intermediate

.seealso: MatPtAP(), MatPtAPSymbolic(), MatMatMultNumeric()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPtAPNumeric(Mat A,Mat P,Mat C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidHeaderSpecific(P,MAT_COOKIE,2);
  PetscValidType(P,2);
  ierr = MatPreallocated(P);CHKERRQ(ierr);
  if (!P->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (P->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidHeaderSpecific(C,MAT_COOKIE,3);
  PetscValidType(C,3);
  ierr = MatPreallocated(C);CHKERRQ(ierr);
  if (C->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (P->cmap->N!=C->rmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->cmap->N,C->rmap->N);
  if (P->rmap->N!=A->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->rmap->N,A->cmap->N);
  if (A->rmap->N!=A->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %D != %D",A->rmap->N,A->cmap->N);
  if (P->cmap->N!=C->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->cmap->N,C->cmap->N);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr); 
  ierr = (*A->ops->ptapnumeric)(A,P,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic"
/*@
   MatPtAPSymbolic - Creates the (i,j) structure of the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  P - the projection matrix

   Output Parameters:
.  C - the (i,j) structure of the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of SeqAIJ matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.  The product is computed using
   this (i,j) structure by calling MatPtAPNumeric().

   Level: intermediate

.seealso: MatPtAP(), MatPtAPNumeric(), MatMatMultSymbolic()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPtAPSymbolic(Mat A,Mat P,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (fill <1.0) SETERRQ1(PETSC_ERR_ARG_SIZ,"Expected fill=%G must be >= 1.0",fill);
  PetscValidHeaderSpecific(P,MAT_COOKIE,2);
  PetscValidType(P,2);
  ierr = MatPreallocated(P);CHKERRQ(ierr);
  if (!P->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (P->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidPointer(C,3);

  if (P->rmap->N!=A->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->rmap->N,A->cmap->N);
  if (A->rmap->N!=A->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %D != %D",A->rmap->N,A->cmap->N);
  ierr = MatPreallocated(A);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr); 
  ierr = (*A->ops->ptapsymbolic)(A,P,fill,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr); 

  ierr = MatSetBlockSize(*C,A->rmap->bs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult"
/*@
   MatMatMult - Performs Matrix-Matrix Multiplication C=A*B.

   Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the right matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use PETSC_DEFAULT if you do not have a good estimate
          if the result is a dense matrix this is irrelevent

   Output Parameters:
.  C - the product matrix

   Notes:
   Unless scall is MAT_REUSE_MATRIX C will be created.

   MAT_REUSE_MATRIX can only be used if the matrices A and B have the same nonzero pattern as in the previous call
   
   To determine the correct fill value, run with -info and search for the string "Fill ratio" to see the value
   actually needed.

   If you have many matrices with the same non-zero structure to multiply, you 
   should either 
$   1) use MAT_REUSE_MATRIX in all calls but the first or
$   2) call MatMatMultSymbolic() once and then MatMatMultNumeric() for each product needed

   Level: intermediate

.seealso: MatMatMultSymbolic(), MatMatMultNumeric(), MatPtAP()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*fA)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*fB)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*mult)(Mat,Mat,MatReuse,PetscReal,Mat *)=PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(B,2);
  ierr = MatPreallocated(B);CHKERRQ(ierr);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidPointer(C,3);
  if (B->rmap->N!=A->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->cmap->N);
  if (scall == MAT_REUSE_MATRIX){
    PetscValidPointer(*C,5);
    PetscValidHeaderSpecific(*C,MAT_COOKIE,5);
  }
  if (fill == PETSC_DEFAULT || fill == PETSC_DECIDE) fill = 2.0;
  if (fill < 1.0) SETERRQ1(PETSC_ERR_ARG_SIZ,"Expected fill=%G must be >= 1.0",fill);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  fA = A->ops->matmult;
  fB = B->ops->matmult;
  if (fB == fA) {
    if (!fB) SETERRQ1(PETSC_ERR_SUP,"MatMatMult not supported for B of type %s",((PetscObject)B)->type_name);
    mult = fB;
  } else { 
    /* dispatch based on the type of A and B */
    char  multname[256];
    ierr = PetscStrcpy(multname,"MatMatMult_");CHKERRQ(ierr);
    ierr = PetscStrcat(multname,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(multname,"_");CHKERRQ(ierr);
    ierr = PetscStrcat(multname,((PetscObject)B)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(multname,"_C");CHKERRQ(ierr); /* e.g., multname = "MatMatMult_seqdense_seqaij_C" */
    ierr = PetscObjectQueryFunction((PetscObject)B,multname,(void (**)(void))&mult);CHKERRQ(ierr);
    if (!mult) SETERRQ2(PETSC_ERR_ARG_INCOMP,"MatMatMult requires A, %s, to be compatible with B, %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);    
  } 
  ierr = PetscLogEventBegin(MAT_MatMult,A,B,0,0);CHKERRQ(ierr); 
  ierr = (*mult)(A,B,scall,fill,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMult,A,B,0,0);CHKERRQ(ierr);   
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic"
/*@
   MatMatMultSymbolic - Performs construction, preallocation, and computes the ij structure
   of the matrix-matrix product C=A*B.  Call this routine before calling MatMatMultNumeric().

   Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the right matrix
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use PETSC_DEFAULT if you do not have a good estimate,
      if C is a dense matrix this is irrelevent
 
   Output Parameters:
.  C - the product matrix

   Notes:
   Unless scall is MAT_REUSE_MATRIX C will be created.

   To determine the correct fill value, run with -info and search for the string "Fill ratio" to see the value
   actually needed.

   This routine is currently implemented for 
    - pairs of AIJ matrices and classes which inherit from AIJ, C will be of type AIJ
    - pairs of AIJ (A) and Dense (B) matrix, C will be of type Dense.
    - pairs of Dense (A) and AIJ (B) matrix, C will be of type Dense.

   Level: intermediate

.seealso: MatMatMult(), MatMatMultNumeric()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMatMultSymbolic(Mat A,Mat B,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*Asymbolic)(Mat,Mat,PetscReal,Mat *);
  PetscErrorCode (*Bsymbolic)(Mat,Mat,PetscReal,Mat *);
  PetscErrorCode (*symbolic)(Mat,Mat,PetscReal,Mat *)=PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(B,2);
  ierr = MatPreallocated(B);CHKERRQ(ierr);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidPointer(C,3);

  if (B->rmap->N!=A->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->cmap->N);
  if (fill == PETSC_DEFAULT) fill = 2.0;
  if (fill < 1.0) SETERRQ1(PETSC_ERR_ARG_SIZ,"Expected fill=%G must be > 1.0",fill);
  ierr = MatPreallocated(A);CHKERRQ(ierr);
 
  Asymbolic = A->ops->matmultsymbolic;
  Bsymbolic = B->ops->matmultsymbolic;
  if (Asymbolic == Bsymbolic){
    if (!Bsymbolic) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for B of type %s",((PetscObject)B)->type_name);
    symbolic = Bsymbolic;
  } else { /* dispatch based on the type of A and B */
    char  symbolicname[256];
    ierr = PetscStrcpy(symbolicname,"MatMatMultSymbolic_");CHKERRQ(ierr);
    ierr = PetscStrcat(symbolicname,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(symbolicname,"_");CHKERRQ(ierr);
    ierr = PetscStrcat(symbolicname,((PetscObject)B)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(symbolicname,"_C");CHKERRQ(ierr); 
    ierr = PetscObjectQueryFunction((PetscObject)B,symbolicname,(void (**)(void))&symbolic);CHKERRQ(ierr);
    if (!symbolic) SETERRQ2(PETSC_ERR_ARG_INCOMP,"MatMatMultSymbolic requires A, %s, to be compatible with B, %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);
  }
  ierr = PetscLogEventBegin(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr); 
  ierr = (*symbolic)(A,B,fill,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric"
/*@
   MatMatMultNumeric - Performs the numeric matrix-matrix product.
   Call this routine after first calling MatMatMultSymbolic().

   Collective on Mat

   Input Parameters:
+  A - the left matrix
-  B - the right matrix

   Output Parameters:
.  C - the product matrix, which was created by from MatMatMultSymbolic() or a call to MatMatMult().

   Notes:
   C must have been created with MatMatMultSymbolic().

   This routine is currently implemented for 
    - pairs of AIJ matrices and classes which inherit from AIJ, C will be of type MATAIJ.
    - pairs of AIJ (A) and Dense (B) matrix, C will be of type Dense.
    - pairs of Dense (A) and AIJ (B) matrix, C will be of type Dense.

   Level: intermediate

.seealso: MatMatMult(), MatMatMultSymbolic()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMatMultNumeric(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;
  PetscErrorCode (*Anumeric)(Mat,Mat,Mat);
  PetscErrorCode (*Bnumeric)(Mat,Mat,Mat);
  PetscErrorCode (*numeric)(Mat,Mat,Mat)=PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(B,2);
  ierr = MatPreallocated(B);CHKERRQ(ierr);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(C,MAT_COOKIE,3);
  PetscValidType(C,3);
  ierr = MatPreallocated(C);CHKERRQ(ierr);
  if (!C->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (C->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  if (B->cmap->N!=C->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->cmap->N,C->cmap->N);
  if (B->rmap->N!=A->cmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->cmap->N);
  if (A->rmap->N!=C->rmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",A->rmap->N,C->rmap->N);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  Anumeric = A->ops->matmultnumeric;
  Bnumeric = B->ops->matmultnumeric;
  if (Anumeric == Bnumeric){
    if (!Bnumeric) SETERRQ1(PETSC_ERR_SUP,"MatMatMultNumeric not supported for B of type %s",((PetscObject)B)->type_name);
    numeric = Bnumeric;
  } else {
    char  numericname[256];
    ierr = PetscStrcpy(numericname,"MatMatMultNumeric_");CHKERRQ(ierr);
    ierr = PetscStrcat(numericname,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(numericname,"_");CHKERRQ(ierr);
    ierr = PetscStrcat(numericname,((PetscObject)B)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(numericname,"_C");CHKERRQ(ierr); 
    ierr = PetscObjectQueryFunction((PetscObject)B,numericname,(void (**)(void))&numeric);CHKERRQ(ierr);
    if (!numeric) 
      SETERRQ2(PETSC_ERR_ARG_INCOMP,"MatMatMultNumeric requires A, %s, to be compatible with B, %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);
  }
  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr); 
  ierr = (*numeric)(A,B,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultTranspose"
/*@
   MatMatMultTranspose - Performs Matrix-Matrix Multiplication C=A^T*B.

   Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the right matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use PETSC_DEFAULT if not known

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created if MAT_INITIAL_MATRIX and must be destroyed by the user with MatDestroy().

   MAT_REUSE_MATRIX can only be used if the matrices A and B have the same nonzero pattern as in the previous call

  To determine the correct fill value, run with -info and search for the string "Fill ratio" to see the value
   actually needed.

   This routine is currently only implemented for pairs of SeqAIJ matrices and pairs of SeqDense matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.

   Level: intermediate

.seealso: MatMatMultTransposeSymbolic(), MatMatMultTransposeNumeric(), MatPtAP()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMatMultTranspose(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*fA)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*fB)(Mat,Mat,MatReuse,PetscReal,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(B,2);
  ierr = MatPreallocated(B);CHKERRQ(ierr);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidPointer(C,3);
  if (B->rmap->N!=A->rmap->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->rmap->N);
  if (fill == PETSC_DEFAULT || fill == PETSC_DECIDE) fill = 2.0;
  if (fill < 1.0) SETERRQ1(PETSC_ERR_ARG_SIZ,"Expected fill=%G must be > 1.0",fill);
  ierr = MatPreallocated(A);CHKERRQ(ierr);

  fA = A->ops->matmulttranspose;
  if (!fA) SETERRQ1(PETSC_ERR_SUP,"MatMatMultTranspose not supported for A of type %s",((PetscObject)A)->type_name);
  fB = B->ops->matmulttranspose;
  if (!fB) SETERRQ1(PETSC_ERR_SUP,"MatMatMultTranspose not supported for B of type %s",((PetscObject)B)->type_name);
  if (fB!=fA) SETERRQ2(PETSC_ERR_ARG_INCOMP,"MatMatMultTranspose requires A, %s, to be compatible with B, %s",((PetscObject)A)->type_name,((PetscObject)B)->type_name);

  ierr = PetscLogEventBegin(MAT_MatMultTranspose,A,B,0,0);CHKERRQ(ierr); 
  ierr = (*A->ops->matmulttranspose)(A,B,scall,fill,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultTranspose,A,B,0,0);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatGetRedundantMatrix"
/*@C
   MatGetRedundantMatrix - Create redundant matrices and put them into processors of subcommunicators. 

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  nsubcomm - the number of subcommunicators (= number of redundant pareallel or sequential matrices)
.  subcomm - MPI communicator split from the communicator where mat resides in
.  mlocal_red - number of local rows of the redundant matrix
-  reuse - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.  matredundant - redundant matrix

   Notes:
   MAT_REUSE_MATRIX can only be used when the nonzero structure of the 
   original matrix has not changed from that last call to MatGetRedundantMatrix().

   This routine creates the duplicated matrices in subcommunicators; you should NOT create them before
   calling it. 

   Only MPIAIJ matrix is supported. 
   
   Level: advanced

   Concepts: subcommunicator
   Concepts: duplicate matrix

.seealso: MatDestroy()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetRedundantMatrix(Mat mat,PetscInt nsubcomm,MPI_Comm subcomm,PetscInt mlocal_red,MatReuse reuse,Mat *matredundant)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (nsubcomm && reuse == MAT_REUSE_MATRIX) {
    PetscValidPointer(*matredundant,6);
    PetscValidHeaderSpecific(*matredundant,MAT_COOKIE,6);
  }
  if (!mat->ops->getredundantmatrix) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = MatPreallocated(mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(MAT_GetRedundantMatrix,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->getredundantmatrix)(mat,nsubcomm,subcomm,mlocal_red,reuse,matredundant);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_GetRedundantMatrix,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
