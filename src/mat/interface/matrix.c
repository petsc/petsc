#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: matrix.c,v 1.331 1999/03/31 18:40:59 bsmith Exp bsmith $";
#endif

/*
   This is where the abstract matrix operations are defined
*/

#include "src/mat/matimpl.h"        /*I "mat.h" I*/
#include "src/vec/vecimpl.h"  

#undef __FUNC__  
#define __FUNC__ "MatGetRow"
/*@C
   MatGetRow - Gets a row of a matrix.  You MUST call MatRestoreRow()
   for each row that you get to ensure that your application does
   not bleed memory.

   Not Collective

   Input Parameters:
+  mat - the matrix
-  row - the row to get

   Output Parameters:
+  ncols -  the number of nonzeros in the row
.  cols - if nonzero, the column numbers
-  vals - if nonzero, the values

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
   matrix at a time.

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

.keywords: matrix, row, get, extract

.seealso: MatRestoreRow(), MatSetValues(), MatGetValues(), MatGetSubmatrices(), MatGetDiagonal()
@*/
int MatGetRow(Mat mat,int row,int *ncols,int **cols,Scalar **vals)
{
  int   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(ncols);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->getrow) SETERRQ(PETSC_ERR_SUP,0,"");
  PLogEventBegin(MAT_GetRow,mat,0,0,0);
  ierr = (*mat->ops->getrow)(mat,row,ncols,cols,vals); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetRow,mat,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow"
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

.keywords: matrix, row, restore

.seealso:  MatGetRow()
@*/
int MatRestoreRow(Mat mat,int row,int *ncols,int **cols,Scalar **vals)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(ncols);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (!mat->ops->restorerow) PetscFunctionReturn(0);
  ierr = (*mat->ops->restorerow)(mat,row,ncols,cols,vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView"
/*@C
   MatView - Visualizes a matrix object.

   Collective on Mat unless Viewer is VIEWER_STDOUT_SELF

   Input Parameters:
+  mat - the matrix
-  ptr - visualization context

  Notes:
  The available visualization contexts include
+    VIEWER_STDOUT_SELF - standard output (default)
.    VIEWER_STDOUT_WORLD - synchronized standard
        output where only the first processor opens
        the file.  All other processors send their 
        data to the first processor to print. 
-     VIEWER_DRAW_WORLD - graphical display of nonzero structure

   The user can open alternative visualization contexts with
+    ViewerASCIIOpen() - Outputs matrix to a specified file
.    ViewerBinaryOpen() - Outputs matrix in binary to a
         specified file; corresponding input uses MatLoad()
.    ViewerDrawOpen() - Outputs nonzero matrix structure to 
         an X window display
-    ViewerSocketOpen() - Outputs matrix to Socket viewer.
         Currently only the sequential dense and AIJ
         matrix types support the Socket viewer.

   The user can call ViewerSetFormat() to specify the output
   format of ASCII printed objects (when using VIEWER_STDOUT_SELF,
   VIEWER_STDOUT_WORLD and ViewerASCIIOpen).  Available formats include
+    VIEWER_FORMAT_ASCII_DEFAULT - default, prints matrix contents
.    VIEWER_FORMAT_ASCII_MATLAB - prints matrix contents in Matlab format
.    VIEWER_FORMAT_ASCII_DENSE - prints entire matrix including zeros
.    VIEWER_FORMAT_ASCII_COMMON - prints matrix contents, using a sparse 
         format common among all matrix types
.    VIEWER_FORMAT_ASCII_IMPL - prints matrix contents, using an implementation-specific 
         format (which is in many cases the same as the default)
.    VIEWER_FORMAT_ASCII_INFO - prints basic information about the matrix
         size and structure (not the matrix entries)
-    VIEWER_FORMAT_ASCII_INFO_LONG - prints more detailed information about
         the matrix structure

   Level: beginner

.keywords: matrix, view, visualize, output, print, write, draw

.seealso: ViewerSetFormat(), ViewerASCIIOpen(), ViewerDrawOpen(), 
          ViewerSocketOpen(), ViewerBinaryOpen(), MatLoad()
@*/
int MatView(Mat mat,Viewer viewer)
{
  int          format, ierr, rows, cols;
  char         *cstr;
  ViewerType   vtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (viewer) PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF;
  }

  ierr = ViewerGetType(viewer,&vtype);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);  
    if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
      ierr = ViewerASCIIPrintf(viewer,"Matrix Object:\n");CHKERRQ(ierr);
      ierr = MatGetType(mat,PETSC_NULL,&cstr); CHKERRQ(ierr);
      ierr = MatGetSize(mat,&rows,&cols); CHKERRQ(ierr);
      ierr = ViewerASCIIPrintf(viewer,"  type=%s, rows=%d, cols=%d\n",cstr,rows,cols);CHKERRQ(ierr);
      if (mat->ops->getinfo) {
        MatInfo info;
        ierr = MatGetInfo(mat,MAT_GLOBAL_SUM,&info); CHKERRQ(ierr);
        ierr = ViewerASCIIPrintf(viewer,"  total: nonzeros=%d, allocated nonzeros=%d\n",
                          (int)info.nz_used,(int)info.nz_allocated);CHKERRQ(ierr);
      }
    }
  }
  if (mat->ops->view) {
    ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*mat->ops->view)(mat,viewer); CHKERRQ(ierr);
    ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatScaleSystem"
/*@C
   MatScaleSystem - Scale a vector solution and right hand side to 
   match the scaling of a scaled matrix.
  
   Collective on Mat

   Input Parameter:
+  mat - the matrix
.  x - solution vector (or PETSC_NULL)
+  b - right hand side vector (or PETSC_NULL)


   Notes: 
   For AIJ, BAIJ, and BDiag matrix formats, the matrices are not 
   internally scaled, so this does nothing. For MPIROWBS it
   permutes and diagonally scales.

   The SLES methods automatically call this routine when required
   (via PCPreSolve()) so it is rarely used directly.

   Level: Developer            

.keywords: matrix, scale

.seealso: MatUseScaledForm(), MatUnScaleSystem()
@*/
int MatScaleSystem(Mat mat,Vec x,Vec b)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (x) {PetscValidHeaderSpecific(x,VEC_COOKIE);}
  if (b) {PetscValidHeaderSpecific(b,VEC_COOKIE);}
  if (mat->ops->scalesystem) {
    ierr = (*mat->ops->scalesystem)(mat,x,b); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatUnScaleSystem"
/*@C
   MatUnScaleSystem - Unscales a vector solution and right hand side to 
   match the original scaling of a scaled matrix.
  
   Collective on Mat

   Input Parameter:
+  mat - the matrix
.  x - solution vector (or PETSC_NULL)
+  b - right hand side vector (or PETSC_NULL)


   Notes: 
   For AIJ, BAIJ, and BDiag matrix formats, the matrices are not 
   internally scaled, so this does nothing. For MPIROWBS it
   permutes and diagonally scales.

   The SLES methods automatically call this routine when required
   (via PCPreSolve()) so it is rarely used directly.

   Level: Developer            

.keywords: matrix, scale

.seealso: MatUseScaledForm(), MatScaleSystem()
@*/
int MatUnScaleSystem(Mat mat,Vec x,Vec b)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (x) {PetscValidHeaderSpecific(x,VEC_COOKIE);}
  if (b) {PetscValidHeaderSpecific(b,VEC_COOKIE);}
  if (mat->ops->unscalesystem) {
    ierr = (*mat->ops->unscalesystem)(mat,x,b); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatUseScaledForm"
/*@C
   MatUseScaledForm - For matrix storage formats that scale the 
   matrix (for example MPIRowBS matrices are diagonally scaled on
   assembly) indicates matrix operations (MatMult() etc) are 
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

.keywords: matrix, scale

.seealso: MatScaleSystem(), MatUnScaleSystem()
@*/
int MatUseScaledForm(Mat mat,PetscTruth scaled)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->ops->usescaledform) {
    ierr = (*mat->ops->usescaledform)(mat,scaled); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy"
/*@C
   MatDestroy - Frees space taken by a matrix.
  
   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Level: beginner

.keywords: matrix, destroy
@*/
int MatDestroy(Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = (*mat->ops->destroy)(mat); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatValid"
/*@
   MatValid - Checks whether a matrix object is valid.

   Collective on Mat

   Input Parameter:
.  m - the matrix to check 

   Output Parameter:
   flg - flag indicating matrix status, either
   PETSC_TRUE if matrix is valid, or PETSC_FALSE otherwise.

   Level: developer

.keywords: matrix, valid
@*/
int MatValid(Mat m,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidIntPointer(flg);
  if (!m)                           *flg = PETSC_FALSE;
  else if (m->cookie != MAT_COOKIE) *flg = PETSC_FALSE;
  else                              *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetValues"
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
   By default the values, v, are row-oriented and unsorted.
   See MatSetOption() for other options.

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

.keywords: matrix, insert, add, set, values

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal()
@*/
int MatSetValues(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v,InsertMode addv)
{
  int ierr;

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(USE_PETSC_BOPT_g)
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"Cannot mix add values and insert values");
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  PLogEventBegin(MAT_SetValues,mat,0,0,0);
  if (!mat->ops->setvalues) SETERRQ(PETSC_ERR_SUP,1,"Not supported for this matrix type");
  ierr = (*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  PLogEventEnd(MAT_SetValues,mat,0,0,0);  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetValuesBlocked"
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
   By default the values, v, are row-oriented and unsorted. So the layout of 
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

   Restrictions:
   MatSetValuesBlocked() is currently supported only for the block AIJ
   matrix format (MATSEQBAIJ and MATMPIBAIJ, which are created via
   MatCreateSeqBAIJ() and MatCreateMPIBAIJ()).

   Level: intermediate

.keywords: matrix, insert, add, set, values

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetValuesBlockedLocal()
@*/
int MatSetValuesBlocked(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v,InsertMode addv)
{
  int ierr;

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(USE_PETSC_BOPT_g) 
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"Cannot mix add values and insert values");
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  PLogEventBegin(MAT_SetValues,mat,0,0,0);
  if (!mat->ops->setvaluesblocked) SETERRQ(PETSC_ERR_SUP,1,"Not supported for this matrix type");
  ierr = (*mat->ops->setvaluesblocked)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  PLogEventEnd(MAT_SetValues,mat,0,0,0);  
  PetscFunctionReturn(0);
}

/*MC
   MatSetValue - Set a single entry into a matrix.

   Synopsis:
   void MatSetValue(Mat m,int row,int col,Scalar value,InsertMode mode);

   Not collective

   Input Parameters:
+  m - the matrix
.  row - the row location of the entry
.  col - the column location of the entry
.  value - the value to insert
-  mode - either INSERT_VALUES or ADD_VALUES

   Notes: 
   For efficiency one should use MatSetValues() and set several or many
   values simultaneously if possible.

   Note that VecSetValue() does NOT return an error code (since this
   is checked internally).

   Level: beginner

.seealso: MatSetValues()
M*/

#undef __FUNC__  
#define __FUNC__ "MatGetValues"
/*@ 
   MatGetValues - Gets a block of values from a matrix.

   Not Collective; currently only returns a local block

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array for storing the values
.  m, idxm - the number of rows and their global indices 
-  n, idxn - the number of columns and their global indices

   Notes:
   The user must allocate space (m*n Scalars) for the values, v.
   The values, v, are then returned in a row-oriented format, 
   analogous to that used by default in MatSetValues().

   MatGetValues() uses 0-based row and column numbers in
   Fortran as well as in C.

   MatGetValues() requires that the matrix has been assembled
   with MatAssemblyBegin()/MatAssemblyEnd().  Thus, calls to
   MatSetValues() and MatGetValues() CANNOT be made in succession
   without intermediate matrix assembly.

   Level: advanced

.keywords: matrix, get, values

.seealso: MatGetRow(), MatGetSubmatrices(), MatSetValues()
@*/
int MatGetValues(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->getvalues) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_GetValues,mat,0,0,0);
  ierr = (*mat->ops->getvalues)(mat,m,idxm,n,idxn,v); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetValues,mat,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetLocalToGlobalMapping"
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

.keywords: matrix, set, values, local, global, mapping

.seealso:  MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetValuesLocal()
@*/
int MatSetLocalToGlobalMapping(Mat x,ISLocalToGlobalMapping mapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,MAT_COOKIE);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);

  if (x->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Mapping already set for matrix");
  }

  x->mapping = mapping;
  PetscObjectReference((PetscObject)mapping);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetLocalToGlobalMappingBlocked"
/*@
   MatSetLocalToGlobalMappingBlocked - Sets a local-to-global numbering for use
   by the routine MatSetValuesBlockedLocal() to allow users to insert matrix
   entries using a local (per-processor) numbering.

   Not Collective

   Input Parameters:
+  x - the matrix
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or
             ISLocalToGlobalMappingCreateIS()

   Level: intermediate

.keywords: matrix, set, values, local ordering

.seealso:  MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetValuesBlockedLocal(),
           MatSetValuesBlocked(), MatSetValuesLocal()
@*/
int MatSetLocalToGlobalMappingBlocked(Mat x,ISLocalToGlobalMapping mapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,MAT_COOKIE);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);

  if (x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Mapping already set for matrix");
  }
 
  x->bmapping = mapping;
  PetscObjectReference((PetscObject)mapping);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetValuesLocal"
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

.keywords: matrix, set, values, local ordering

.seealso:  MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetLocalToGlobalMapping()
@*/
int MatSetValuesLocal(Mat mat,int nrow,int *irow,int ncol, int *icol,Scalar *y,InsertMode addv) 
{
  int ierr,irowm[2048],icolm[2048];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(irow);
  PetscValidIntPointer(icol);
  PetscValidScalarPointer(y);

  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(USE_PETSC_BOPT_g) 
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"Cannot mix add values and insert values");
  }
  if (!mat->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Local to global never set with MatSetLocalToGlobalMapping()");
  }
  if (nrow > 2048 || ncol > 2048) {
    SETERRQ2(PETSC_ERR_SUP,0,"Number column/row indices must be <= 2048: are %d %d",nrow,ncol);
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  PLogEventBegin(MAT_SetValues,mat,0,0,0);
  ierr = ISLocalToGlobalMappingApply(mat->mapping,nrow,irow,irowm); CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mat->mapping,ncol,icol,icolm);CHKERRQ(ierr); 
  ierr = (*mat->ops->setvalues)(mat,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
  PLogEventEnd(MAT_SetValues,mat,0,0,0);  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetValuesBlockedLocal"
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
   local-to-global mapping by calling MatSetLocalToGlobalMappingBlocked(),
   where the mapping MUST be set for matrix blocks, not for matrix elements.

   Calls to MatSetValuesBlockedLocal() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd() 
   MUST be called after all calls to MatSetValuesBlockedLocal() have been completed.

   Level: intermediate

.keywords: matrix, set, values, blocked, local

.seealso:  MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesLocal(), MatSetLocalToGlobalMappingBlocked(), MatSetValuesBlocked()
@*/
int MatSetValuesBlockedLocal(Mat mat,int nrow,int *irow,int ncol,int *icol,Scalar *y,InsertMode addv) 
{
  int ierr,irowm[2048],icolm[2048];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(irow);
  PetscValidIntPointer(icol);
  PetscValidScalarPointer(y);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(USE_PETSC_BOPT_g) 
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"Cannot mix add values and insert values");
  }
  if (!mat->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Local to global never set with MatSetLocalToGlobalMappingBlocked()");
  }
  if (nrow > 2048 || ncol > 2048) {
    SETERRQ2(PETSC_ERR_SUP,0,"Number column/row indices must be <= 2048: are %d %d",nrow,ncol);
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  PLogEventBegin(MAT_SetValues,mat,0,0,0);
  ierr = ISLocalToGlobalMappingApply(mat->bmapping,nrow,irow,irowm); CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mat->bmapping,ncol,icol,icolm); CHKERRQ(ierr);
  ierr = (*mat->ops->setvaluesblocked)(mat,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
  PLogEventEnd(MAT_SetValues,mat,0,0,0);  
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatMult"
/*@
   MatMult - Computes the matrix-vector product, y = Ax.

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

.keywords: matrix, multiply, matrix-vector product

.seealso: MatMultTrans(), MatMultAdd(), MatMultTransAdd()
@*/
int MatMult(Mat mat,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE); 
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"x and y must be different vectors");
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->N,x->N); 
  if (mat->M != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec y: global dim %d %d",mat->M,y->N); 
  if (mat->m != y->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec y: local dim %d %d",mat->m,y->n); 

  PLogEventBegin(MAT_Mult,mat,x,y,0);
  ierr = (*mat->ops->mult)(mat,x,y); CHKERRQ(ierr);
  PLogEventEnd(MAT_Mult,mat,x,y,0);

  PetscFunctionReturn(0);
}   

#undef __FUNC__  
#define __FUNC__ "MatMultTrans"
/*@
   MatMultTrans - Computes matrix transpose times a vector.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMultTrans(A,y,y).

   Level: beginner

.keywords: matrix, multiply, matrix-vector product, transpose

.seealso: MatMult(), MatMultAdd(), MatMultTransAdd()
@*/
int MatMultTrans(Mat mat,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"x and y must be different vectors");
  if (mat->M != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->M,x->N); 
  if (mat->N != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec y: global dim %d %d",mat->N,y->N); 
  PLogEventBegin(MAT_MultTrans,mat,x,y,0);
  ierr = (*mat->ops->multtrans)(mat,x,y); CHKERRQ(ierr);
  PLogEventEnd(MAT_MultTrans,mat,x,y,0);
  PetscFunctionReturn(0);
}   

#undef __FUNC__  
#define __FUNC__ "MatMultAdd"
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

.keywords: matrix, multiply, matrix-vector product, add

.seealso: MatMultTrans(), MatMult(), MatMultTransAdd()
@*/
int MatMultAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(v1,VEC_COOKIE);
  PetscValidHeaderSpecific(v2,VEC_COOKIE); PetscValidHeaderSpecific(v3,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix");
  if (mat->N != v1->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec v1: global dim %d %d",mat->N,v1->N);
  if (mat->M != v2->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec v2: global dim %d %d",mat->M,v2->N);
  if (mat->M != v3->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec v3: global dim %d %d",mat->M,v3->N);
  if (mat->m != v3->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec v3: local dim %d %d",mat->m,v3->n); 
  if (mat->m != v2->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec v2: local dim %d %d",mat->m,v2->n); 
  if (v1 == v3) SETERRQ(PETSC_ERR_ARG_IDN,0,"v1 and v3 must be different vectors");

  PLogEventBegin(MAT_MultAdd,mat,v1,v2,v3);
  ierr = (*mat->ops->multadd)(mat,v1,v2,v3); CHKERRQ(ierr);
  PLogEventEnd(MAT_MultAdd,mat,v1,v2,v3);
  PetscFunctionReturn(0);
}   

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd"
/*@
   MatMultTransAdd - Computes v3 = v2 + A' * v1.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  v1, v2 - the vectors

   Output Parameters:
.  v3 - the result

   Notes:
   The vectors v1 and v3 cannot be the same.  I.e., one cannot
   call MatMultTransAdd(A,v1,v2,v1).

   Level: beginner

.keywords: matrix, multiply, matrix-vector product, transpose, add

.seealso: MatMultTrans(), MatMultAdd(), MatMult()
@*/
int MatMultTransAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(v1,VEC_COOKIE);
  PetscValidHeaderSpecific(v2,VEC_COOKIE);PetscValidHeaderSpecific(v3,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->multtransadd) SETERRQ(PETSC_ERR_SUP,0,"");
  if (v1 == v3) SETERRQ(PETSC_ERR_ARG_IDN,0,"v1 and v3 must be different vectors");
  if (mat->M != v1->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec v1: global dim %d %d",mat->M,v1->N);
  if (mat->N != v2->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec v2: global dim %d %d",mat->N,v2->N);
  if (mat->N != v3->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec v3: global dim %d %d",mat->N,v3->N);

  PLogEventBegin(MAT_MultTransAdd,mat,v1,v2,v3);
  ierr = (*mat->ops->multtransadd)(mat,v1,v2,v3); CHKERRQ(ierr);
  PLogEventEnd(MAT_MultTransAdd,mat,v1,v2,v3); 
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatGetInfo"
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
   factorization, etc.).  Much of this info is printed to STDOUT
   when using the runtime options 
$       -log_info -mat_view_info

   Example for C/C++ Users:
   See the file ${PETSC_DIR}/include/mat.h for a complete list of
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
   of interest.  See the file ${PETSC_DIR}/include/finclude/mat.h
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
 
.keywords: matrix, get, info, storage, nonzeros, memory, fill
@*/
int MatGetInfo(Mat mat,MatInfoType flag,MatInfo *info)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(info);
  if (!mat->ops->getinfo) SETERRQ(PETSC_ERR_SUP,0,"");
  ierr = (*mat->ops->getinfo)(mat,flag,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}   

/* ----------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatILUDTFactor"
/*@  
   MatILUDTFactor - Performs a drop tolerance ILU factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  dt  - the drop tolerance
.  maxnz - the maximum number of nonzeros per row allowed
.  row - row permutation
-  col - column permutation

   Output Parameters:
.  fact - the factored matrix

   Notes:
   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, LU, in-place

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatILUDTFactor(Mat mat,double dt,int maxnz,IS row,IS col,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(fact);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->iludtfactor) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_ILUFactor,mat,row,col,0); 
  ierr = (*mat->ops->iludtfactor)(mat,dt,maxnz,row,col,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_ILUFactor,mat,row,col,0);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactor"
/*@  
   MatLUFactor - Performs in-place LU factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row permutation
.  col - column permutation
-  f - expected fill as ratio of original fill.

   Notes:
   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, LU, in-place

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatLUFactor(Mat mat,IS row,IS col,double f)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,0,"matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->lufactor) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_LUFactor,mat,row,col,0); 
  ierr = (*mat->ops->lufactor)(mat,row,col,f); CHKERRQ(ierr);
  PLogEventEnd(MAT_LUFactor,mat,row,col,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatILUFactor"
/*@  
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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, ILU, in-place

.seealso: MatILUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatILUFactor(Mat mat,IS row,IS col,MatILUInfo *info)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,0,"matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->ilufactor) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_ILUFactor,mat,row,col,0); 
  ierr = (*mat->ops->ilufactor)(mat,row,col,info); CHKERRQ(ierr);
  PLogEventEnd(MAT_ILUFactor,mat,row,col,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorSymbolic"
/*@  
   MatLUFactorSymbolic - Performs symbolic LU factorization of matrix.
   Call this routine before calling MatLUFactorNumeric().

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row, col - row and column permutations
-  f - expected fill as ratio of the original number of nonzeros, 
       for example 3.0; choosing this parameter well can result in 
       more efficient use of time and space. Run with the option -log_info
       to determine an optimal value to use

   Output Parameter:
.  fact - new matrix that has been symbolically factored

   Notes:
   See the users manual for additional information about
   choosing the fill factor for better efficiency.

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, LU, symbolic, fill

.seealso: MatLUFactor(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatLUFactorSymbolic(Mat mat,IS row,IS col,double f,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,0,"matrix must be square");
  PetscValidPointer(fact);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->lufactorsymbolic) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_LUFactorSymbolic,mat,row,col,0); 
  ierr = (*mat->ops->lufactorsymbolic)(mat,row,col,f,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_LUFactorSymbolic,mat,row,col,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric"
/*@  
   MatLUFactorNumeric - Performs numeric LU factorization of a matrix.
   Call this routine after first calling MatLUFactorSymbolic().

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  row, col - row and  column permutations

   Output Parameters:
.  fact - symbolically factored matrix that must have been generated
          by MatLUFactorSymbolic()

   Notes:
   See MatLUFactor() for in-place factorization.  See 
   MatCholeskyFactorNumeric() for the symmetric, positive definite case.  

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, LU, numeric

.seealso: MatLUFactorSymbolic(), MatLUFactor(), MatCholeskyFactor()
@*/
int MatLUFactorNumeric(Mat mat,Mat *fact)
{
  int ierr,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(fact);
  PetscValidHeaderSpecific(*fact,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->M != (*fact)->M || mat->N != (*fact)->N) {
    SETERRQ4(PETSC_ERR_ARG_SIZ,0,"Mat mat,Mat *fact: global dimensions are different %d should = %d %d should = %d",
            mat->M,(*fact)->M,mat->N,(*fact)->N);
  }
  if (!(*fact)->ops->lufactornumeric) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_LUFactorNumeric,mat,*fact,0,0); 
  ierr = (*(*fact)->ops->lufactornumeric)(mat,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_LUFactorNumeric,mat,*fact,0,0); 
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = OptionsHasName(0,"-mat_view_contour",&flg); CHKERRQ(ierr);
    if (flg) {
      ViewerPushFormat(VIEWER_DRAW_(mat->comm),VIEWER_FORMAT_DRAW_CONTOUR,0);CHKERRQ(ierr);
    }
    ierr = MatView(*fact,VIEWER_DRAW_(mat->comm)); CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_DRAW_(mat->comm)); CHKERRQ(ierr);
    if (flg) {
      ViewerPopFormat(VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactor"
/*@  
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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, in-place, Cholesky

.seealso: MatLUFactor(), MatCholeskyFactorSymbolic(), MatCholeskyFactorNumeric()
@*/
int MatCholeskyFactor(Mat mat,IS perm,double f)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->choleskyfactor) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_CholeskyFactor,mat,perm,0,0); 
  ierr = (*mat->ops->choleskyfactor)(mat,perm,f); CHKERRQ(ierr);
  PLogEventEnd(MAT_CholeskyFactor,mat,perm,0,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorSymbolic"
/*@  
   MatCholeskyFactorSymbolic - Performs symbolic Cholesky factorization
   of a symmetric matrix. 

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  perm - row and column permutations
-  f - expected fill as ratio of original

   Output Parameter:
.  fact - the factored matrix

   Notes:
   See MatLUFactorSymbolic() for the nonsymmetric case.  See also
   MatCholeskyFactor() and MatCholeskyFactorNumeric().

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, factorization, symbolic, Cholesky

.seealso: MatLUFactorSymbolic(), MatCholeskyFactor(), MatCholeskyFactorNumeric()
@*/
int MatCholeskyFactorSymbolic(Mat mat,IS perm,double f,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(fact);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->choleskyfactorsymbolic) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_CholeskyFactorSymbolic,mat,perm,0,0);
  ierr = (*mat->ops->choleskyfactorsymbolic)(mat,perm,f,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_CholeskyFactorSymbolic,mat,perm,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric"
/*@  
   MatCholeskyFactorNumeric - Performs numeric Cholesky factorization
   of a symmetric matrix. Call this routine after first calling
   MatCholeskyFactorSymbolic().

   Collective on Mat

   Input Parameter:
.  mat - the initial matrix

   Output Parameter:
.  fact - the factored matrix

   Notes:
   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, numeric, Cholesky

.seealso: MatCholeskyFactorSymbolic(), MatCholeskyFactor(), MatLUFactorNumeric()
@*/
int MatCholeskyFactorNumeric(Mat mat,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(fact);
  if (!mat->ops->choleskyfactornumeric) SETERRQ(PETSC_ERR_SUP,0,"");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->M != (*fact)->M || mat->N != (*fact)->N) {
    SETERRQ4(PETSC_ERR_ARG_SIZ,0,"Mat mat,Mat *fact: global dim %d should = %d %d should = %d",
            mat->M,(*fact)->M,mat->N,(*fact)->N);
  }

  PLogEventBegin(MAT_CholeskyFactorNumeric,mat,*fact,0,0);
  ierr = (*mat->ops->choleskyfactornumeric)(mat,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_CholeskyFactorNumeric,mat,*fact,0,0);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatSolve"
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
   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, linear system, solve, LU, Cholesky, triangular solve

.seealso: MatSolveAdd(), MatSolveTrans(), MatSolveTransAdd()
@*/
int MatSolve(Mat mat,Vec b,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE); PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Unfactored matrix");
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: local dim %d %d",mat->m,b->n); 
  if (mat->M == 0 && mat->N == 0) PetscFunctionReturn(0);

  if (!mat->ops->solve) SETERRQ(PETSC_ERR_SUP,0,"");
  PLogEventBegin(MAT_Solve,mat,b,x,0); 
  ierr = (*mat->ops->solve)(mat,b,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_Solve,mat,b,x,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatForwardSolve"
/* @
   MatForwardSolve - Solves L x = b, given a factored matrix, A = LU.

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
   call MatForwardSolve(A,x,x).

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, forward, LU, Cholesky, triangular solve

.seealso: MatSolve(), MatBackwardSolve()
@ */
int MatForwardSolve(Mat mat,Vec b,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Unfactored matrix");
  if (!mat->ops->forwardsolve) SETERRQ(PETSC_ERR_SUP,0,"");
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: local dim %d %d",mat->m,b->n); 

  PLogEventBegin(MAT_ForwardSolve,mat,b,x,0); 
  ierr = (*mat->ops->forwardsolve)(mat,b,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_ForwardSolve,mat,b,x,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatBackwardSolve"
/* @
   MatBackwardSolve - Solves U x = b, given a factored matrix, A = LU.

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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, backward, LU, Cholesky, triangular solve

.seealso: MatSolve(), MatForwardSolve()
@ */
int MatBackwardSolve(Mat mat,Vec b,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Unfactored matrix");
  if (!mat->ops->backwardsolve) SETERRQ(PETSC_ERR_SUP,0,"");
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: local dim %d %d",mat->m,b->n); 

  PLogEventBegin(MAT_BackwardSolve,mat,b,x,0); 
  ierr = (*mat->ops->backwardsolve)(mat,b,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_BackwardSolve,mat,b,x,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolveAdd"
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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, linear system, solve, LU, Cholesky, add

.seealso: MatSolve(), MatSolveTrans(), MatSolveTransAdd()
@*/
int MatSolveAdd(Mat mat,Vec b,Vec y,Vec x)
{
  Scalar one = 1.0;
  Vec    tmp;
  int    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Unfactored matrix");
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->M != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec y: global dim %d %d",mat->M,y->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: local dim %d %d",mat->m,b->n); 
  if (x->n != y->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Vec x,Vec y: local dim %d %d",x->n,y->n); 

  PLogEventBegin(MAT_SolveAdd,mat,b,x,y); 
  if (mat->ops->solveadd)  {
    ierr = (*mat->ops->solveadd)(mat,b,y,x); CHKERRQ(ierr);
  } else {
    /* do the solve then the add manually */
    if (x != y) {
      ierr = MatSolve(mat,b,x); CHKERRQ(ierr);
      ierr = VecAXPY(&one,y,x); CHKERRQ(ierr);
    } else {
      ierr = VecDuplicate(x,&tmp); CHKERRQ(ierr);
      PLogObjectParent(mat,tmp);
      ierr = VecCopy(x,tmp); CHKERRQ(ierr);
      ierr = MatSolve(mat,b,x); CHKERRQ(ierr);
      ierr = VecAXPY(&one,tmp,x); CHKERRQ(ierr);
      ierr = VecDestroy(tmp); CHKERRQ(ierr);
    }
  }
  PLogEventEnd(MAT_SolveAdd,mat,b,x,y); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolveTrans"
/*@
   MatSolveTrans - Solves A' x = b, given a factored matrix.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolveTrans(A,x,x).

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, linear system, solve, LU, Cholesky, transpose

.seealso: MatSolve(), MatSolveAdd(), MatSolveTransAdd()
@*/
int MatSolveTrans(Mat mat,Vec b,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Unfactored matrix");
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and b must be different vectors");
  if (!mat->ops->solvetrans) SETERRQ(PETSC_ERR_SUP,0,"");
  if (mat->M != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->M,x->N);
  if (mat->N != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: global dim %d %d",mat->N,b->N);

  PLogEventBegin(MAT_SolveTrans,mat,b,x,0); 
  ierr = (*mat->ops->solvetrans)(mat,b,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_SolveTrans,mat,b,x,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolveTransAdd"
/*@
   MatSolveTransAdd - Computes x = y + inv(trans(A)) b, given a 
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
   call MatSolveTransAdd(A,x,y,x).

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, linear system, solve, LU, Cholesky, transpose, add  

.seealso: MatSolve(), MatSolveAdd(), MatSolveTrans()
@*/
int MatSolveTransAdd(Mat mat,Vec b,Vec y,Vec x)
{
  Scalar one = 1.0;
  int    ierr;
  Vec    tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Unfactored matrix");
  if (mat->M != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->M,x->N);
  if (mat->N != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: global dim %d %d",mat->N,b->N);
  if (mat->N != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec y: global dim %d %d",mat->N,y->N);
  if (x->n != y->n)   SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Vec x,Vec y: local dim %d %d",x->n,y->n);

  PLogEventBegin(MAT_SolveTransAdd,mat,b,x,y); 
  if (mat->ops->solvetransadd) {
    ierr = (*mat->ops->solvetransadd)(mat,b,y,x); CHKERRQ(ierr);
  } else {
    /* do the solve then the add manually */
    if (x != y) {
      ierr = MatSolveTrans(mat,b,x); CHKERRQ(ierr);
      ierr = VecAXPY(&one,y,x); CHKERRQ(ierr);
    } else {
      ierr = VecDuplicate(x,&tmp); CHKERRQ(ierr);
      PLogObjectParent(mat,tmp);
      ierr = VecCopy(x,tmp); CHKERRQ(ierr);
      ierr = MatSolveTrans(mat,b,x); CHKERRQ(ierr);
      ierr = VecAXPY(&one,tmp,x); CHKERRQ(ierr);
      ierr = VecDestroy(tmp); CHKERRQ(ierr);
    }
  }
  PLogEventEnd(MAT_SolveTransAdd,mat,b,x,y); 
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "MatRelax"
/*@
   MatRelax - Computes one relaxation sweep.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
.  b - the right hand side
.  omega - the relaxation factor
.  flag - flag indicating the type of SOR (see below)
.  shift -  diagonal shift
-  its - the number of iterations

   Output Parameters:
.  x - the solution (can contain an initial guess)

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
   SOR_LOCAL_SYMMETRIC_SWEEP perform seperate independent smoothings
   on each processor. 

   Application programmers will not generally use MatRelax() directly,
   but instead will employ the SLES/PC interface.

   Notes for Advanced Users:
   The flags are implemented as bitwise inclusive or operations.
   For example, use (SOR_ZERO_INITIAL_GUESS | SOR_SYMMETRIC_SWEEP)
   to specify a zero initial guess for SSOR.

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, relax, relaxation, sweep
@*/
int MatRelax(Mat mat,Vec b,double omega,MatSORType flag,double shift,int its,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (!mat->ops->relax) SETERRQ(PETSC_ERR_SUP,0,"");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,0,"Mat mat,Vec b: local dim %d %d",mat->m,b->n);

  PLogEventBegin(MAT_Relax,mat,b,x,0); 
  ierr =(*mat->ops->relax)(mat,b,omega,flag,shift,its,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_Relax,mat,b,x,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCopy_Basic"
/*
      Default matrix copy routine.
*/
int MatCopy_Basic(Mat A,Mat B,MatStructure str)
{
  int    ierr,i,rstart,rend,nz,*cwork;
  Scalar *vwork;

  PetscFunctionBegin;
  ierr = MatZeroEntries(B); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend); CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    ierr = MatSetValues(B,1,&i,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCopy"
/*@C  
   MatCopy - Copys a matrix to another matrix.

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  str - SAME_NONZERO_PATTERN or DIFFERENT_NONZERO_PATTERN

   Output Parameter:
.  B - where the copy is put

   Notes:
   If you use SAME_NONZERO_PATTERN then the zero matrices had better have the 
   same nonzero pattern or the routine will crash.

   MatCopy() copies the matrix entries of a matrix to another existing
   matrix (after first zeroing the second matrix).  A related routine is
   MatConvert(), which first creates a new matrix and then copies the data.

   Level: intermediate
   
.keywords: matrix, copy, convert

.seealso: MatConvert()
@*/
int MatCopy(Mat A,Mat B,MatStructure str)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE); PetscValidHeaderSpecific(B,MAT_COOKIE);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (A->M != B->M || A->N != B->N) SETERRQ4(PETSC_ERR_ARG_SIZ,0,"Mat A,Mat B: global dim %d %d",A->M,B->M,
                                             A->N,B->N);

  PLogEventBegin(MAT_Copy,A,B,0,0); 
  if (A->ops->copy) { 
    ierr = (*A->ops->copy)(A,B,str); CHKERRQ(ierr);
  } else { /* generic conversion */
    ierr = MatCopy_Basic(A,B,str); CHKERRQ(ierr);
  }
  PLogEventEnd(MAT_Copy,A,B,0,0); 
  PetscFunctionReturn(0);
}

static int MatConvertersSet = 0;
static int (*MatConverters[MAX_MATRIX_TYPES][MAX_MATRIX_TYPES])(Mat,MatType,Mat*) = 
           {{0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}};

#undef __FUNC__  
#define __FUNC__ "MatConvertRegister"
/*@C
    MatConvertRegister - Allows one to register a routine that converts between
    two matrix types.

    Not Collective

    Input Parameters:
+   intype - the type of matrix (defined in include/mat.h), for example, MATSEQAIJ.
-   outtype - new matrix type, or MATSAME

    Level: advanced

.seealso: MatConvertRegisterAll()
@*/
int MatConvertRegister(MatType intype,MatType outtype,int (*converter)(Mat,MatType,Mat*))
{
  PetscFunctionBegin;
  MatConverters[intype][outtype] = converter;
  MatConvertersSet               = 1;
  PetscFunctionReturn(0);
}  

#undef __FUNC__  
#define __FUNC__ "MatConvert"
/*@C  
   MatConvert - Converts a matrix to another matrix, either of the same
   or different type.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  newtype - new matrix type.  Use MATSAME to create a new matrix of the
   same type as the original matrix.

   Output Parameter:
.  M - pointer to place new matrix

   Notes:
   MatConvert() first creates a new matrix and then copies the data from
   the first matrix.  A related routine is MatCopy(), which copies the matrix
   entries of one matrix to another already existing matrix context.

   Level: intermediate

.keywords: matrix, copy, convert

.seealso: MatCopy(), MatDuplicate()
@*/
int MatConvert(Mat mat,MatType newtype,Mat *M)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(M);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  if (newtype > MAX_MATRIX_TYPES || newtype < -1) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Not a valid matrix type");
  }
  *M  = 0;

  if (!MatConvertersSet) {
    ierr = MatLoadRegisterAll(); CHKERRQ(ierr);
  }

  PLogEventBegin(MAT_Convert,mat,0,0,0); 
  if ((newtype == mat->type || newtype == MATSAME) && mat->ops->duplicate) {
    ierr = (*mat->ops->duplicate)(mat,MAT_COPY_VALUES,M); CHKERRQ(ierr);
  } else {
    if (!MatConvertersSet) {
      ierr = MatConvertRegisterAll(); CHKERRQ(ierr);
    }
    if (!MatConverters[mat->type][newtype]) {
      SETERRQ(PETSC_ERR_ARG_WRONG,1,"Invalid matrix type, or matrix converter not registered");
    }
    ierr = (*MatConverters[mat->type][newtype])(mat,newtype,M); CHKERRQ(ierr);
  }
  PLogEventEnd(MAT_Convert,mat,0,0,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDuplicate"
/*@C  
   MatDuplicate - Duplicates a matrix including the non-zero structure.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  op - either MAT_DO_NO_COPY_VALUES or MAT_COPY_VALUES, cause it to copy nonzero
        values as well or not

   Output Parameter:
.  M - pointer to place new matrix

   Level: intermediate

.keywords: matrix, copy, convert, duplicate

.seealso: MatCopy(), MatConvert()
@*/
int MatDuplicate(Mat mat,MatDuplicateOption op,Mat *M)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(M);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  *M  = 0;
  PLogEventBegin(MAT_Convert,mat,0,0,0); 
  if (!mat->ops->duplicate) {
    SETERRQ(PETSC_ERR_SUP,1,"Not written for this matrix type");
  }
  ierr = (*mat->ops->duplicate)(mat,op,M); CHKERRQ(ierr);
  PLogEventEnd(MAT_Convert,mat,0,0,0); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal"
/*@ 
   MatGetDiagonal - Gets the diagonal of a matrix.

   Collective on Mat and Vec

   Input Parameters:
+  mat - the matrix
-  v - the vector for storing the diagonal

   Output Parameter:
.  v - the diagonal of the matrix

   Notes:
   For the SeqAIJ matrix format, this routine may also be called
   on a LU factored matrix; in that case it routines the reciprocal of 
   the diagonal entries in U. It returns the entries permuted by the 
   row and column permutation used during the symbolic factorization.

   Level: intermediate

.keywords: matrix, get, diagonal

.seealso: MatGetRow(), MatGetSubmatrices(), MatGetSubmatrix()
@*/
int MatGetDiagonal(Mat mat,Vec v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(v,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (!mat->ops->getdiagonal) SETERRQ(PETSC_ERR_SUP,0,"");
  ierr = (*mat->ops->getdiagonal)(mat,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose"
/*@C
   MatTranspose - Computes an in-place or out-of-place transpose of a matrix.

   Collective on Mat

   Input Parameter:
.  mat - the matrix to transpose

   Output Parameters:
.  B - the transpose (or pass in PETSC_NULL for an in-place transpose)

   Level: intermediate

.keywords: matrix, transpose

.seealso: MatMultTrans(), MatMultTransAdd()
@*/
int MatTranspose(Mat mat,Mat *B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->transpose) SETERRQ(PETSC_ERR_SUP,0,""); 
  ierr = (*mat->ops->transpose)(mat,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "MatPermute"
/*@C
   MatPermute - Creates a new matrix with rows and columns permuted from the 
   original.

   Collective on Mat

   Input Parameters:
+  mat - the matrix to permute
.  row - row permutation
-  col - column permutation

   Output Parameters:
.  B - the permuted matrix

   Level: advanced

.keywords: matrix, transpose

.seealso: MatGetOrdering()
@*/
int MatPermute(Mat mat,IS row,IS col,Mat *B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(row,IS_COOKIE);
  PetscValidHeaderSpecific(col,IS_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->permute) SETERRQ(PETSC_ERR_SUP,0,""); 
  ierr = (*mat->ops->permute)(mat,row,col,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatEqual"
/*@
   MatEqual - Compares two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
-  B - the second matrix

   Output Parameter:
.  flg - PETSC_TRUE if the matrices are equal; PETSC_FALSE otherwise.

   Level: intermediate

.keywords: matrix, equal, equivalent
@*/
int MatEqual(Mat A,Mat B,PetscTruth *flg)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE); PetscValidHeaderSpecific(B,MAT_COOKIE);
  PetscValidIntPointer(flg);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (A->M != B->M || A->N != B->N) SETERRQ4(PETSC_ERR_ARG_SIZ,0,"Mat A,Mat B: global dim %d %d %d %d",
                                             A->M,B->M,A->N,B->N);
  if (!A->ops->equal) SETERRQ(PETSC_ERR_SUP,0,"");
  ierr = (*A->ops->equal)(A,B,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale"
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
   L = a diagonal matrix, R = a diagonal matrix

   Level: intermediate

.keywords: matrix, diagonal, scale

.seealso: MatScale()
@*/
int MatDiagonalScale(Mat mat,Vec l,Vec r)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops->diagonalscale) SETERRQ(PETSC_ERR_SUP,0,"");
  if (l) PetscValidHeaderSpecific(l,VEC_COOKIE); 
  if (r) PetscValidHeaderSpecific(r,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  PLogEventBegin(MAT_Scale,mat,0,0,0);
  ierr = (*mat->ops->diagonalscale)(mat,l,r); CHKERRQ(ierr);
  PLogEventEnd(MAT_Scale,mat,0,0,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatScale"
/*@
    MatScale - Scales all elements of a matrix by a given number.

    Collective on Mat

    Input Parameters:
+   mat - the matrix to be scaled
-   a  - the scaling value

    Output Parameter:
.   mat - the scaled matrix

    Level: intermediate

.keywords: matrix, scale

.seealso: MatDiagonalScale()
@*/
int MatScale(Scalar *a,Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidScalarPointer(a);
  if (!mat->ops->scale) SETERRQ(PETSC_ERR_SUP,0,"");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  PLogEventBegin(MAT_Scale,mat,0,0,0);
  ierr = (*mat->ops->scale)(a,mat); CHKERRQ(ierr);
  PLogEventEnd(MAT_Scale,mat,0,0,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatNorm"
/*@ 
   MatNorm - Calculates various norms of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  type - the type of norm, NORM_1, NORM_2, NORM_FROBENIUS, NORM_INFINITY

   Output Parameters:
.  norm - the resulting norm 

   Level: intermediate

.keywords: matrix, norm, Frobenius
@*/
int MatNorm(Mat mat,NormType type,double *norm)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidScalarPointer(norm);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->norm) SETERRQ(PETSC_ERR_SUP,0,"Not for this matrix type");
  ierr = (*mat->ops->norm)(mat,type,norm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
     This variable is used to prevent counting of MatAssemblyBegin() that
   are called from within a MatAssemblyEnd().
*/
static int MatAssemblyEnd_InUse = 0;
#undef __FUNC__  
#define __FUNC__ "MatAssemblyBegin"
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

.keywords: matrix, assembly, assemble, begin

.seealso: MatAssemblyEnd(), MatSetValues()
@*/
int MatAssemblyBegin(Mat mat,MatAssemblyType type)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix.\n did you forget to call MatSetUnfactored()?"); 
  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  if (!MatAssemblyEnd_InUse) {
    PLogEventBegin(MAT_AssemblyBegin,mat,0,0,0);
    if (mat->ops->assemblybegin){ierr = (*mat->ops->assemblybegin)(mat,type);CHKERRQ(ierr);}
    PLogEventEnd(MAT_AssemblyBegin,mat,0,0,0);
  } else {
    if (mat->ops->assemblybegin){ierr = (*mat->ops->assemblybegin)(mat,type);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatView_Private"
/*
    Processes command line options to determine if/how a matrix
  is to be viewed. Called by MatAssemblyEnd() and MatLoad().
*/
int MatView_Private(Mat mat)
{
  int ierr,flg;

  PetscFunctionBegin;
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_info",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(mat->comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = MatView(mat,VIEWER_STDOUT_(mat->comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_info_detailed",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(mat->comm),VIEWER_FORMAT_ASCII_INFO_LONG,0);CHKERRQ(ierr);
    ierr = MatView(mat,VIEWER_STDOUT_(mat->comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatView(mat,VIEWER_STDOUT_(mat->comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_matlab",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(mat->comm),VIEWER_FORMAT_ASCII_MATLAB,"M");CHKERRQ(ierr);
    ierr = MatView(mat,VIEWER_STDOUT_(mat->comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = OptionsHasName(0,"-mat_view_contour",&flg); CHKERRQ(ierr);
    if (flg) {
      ViewerPushFormat(VIEWER_DRAW_(mat->comm),VIEWER_FORMAT_DRAW_CONTOUR,0);CHKERRQ(ierr);
    }
    ierr = MatView(mat,VIEWER_DRAW_(mat->comm)); CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_DRAW_(mat->comm)); CHKERRQ(ierr);
    if (flg) {
      ViewerPopFormat(VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    }
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_socket",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatView(mat,VIEWER_SOCKET_(mat->comm)); CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_SOCKET_(mat->comm)); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd"
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
.  -mat_view_draw - Draws nonzero structure of matrix, using MatView() and DrawOpenX().
.  -display <name> - Sets display name (default is host)
-  -draw_pause <sec> - Sets number of seconds to pause after display

   Notes: 
   MatSetValues() generally caches the values.  The matrix is ready to
   use only after MatAssemblyBegin() and MatAssemblyEnd() have been called.
   Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
   in MatSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
   using the matrix.

   Level: beginner

.keywords: matrix, assembly, assemble, end

.seealso: MatAssemblyBegin(), MatSetValues(), DrawOpenX(), MatView()
@*/
int MatAssemblyEnd(Mat mat,MatAssemblyType type)
{
  int        ierr;
  static int inassm = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);

  inassm++;
  MatAssemblyEnd_InUse++;
  if (MatAssemblyEnd_InUse == 1) { /* Do the logging only the first time through */
    PLogEventBegin(MAT_AssemblyEnd,mat,0,0,0);
    if (mat->ops->assemblyend) {
      ierr = (*mat->ops->assemblyend)(mat,type); CHKERRQ(ierr);
    }
    PLogEventEnd(MAT_AssemblyEnd,mat,0,0,0);
  } else {
    if (mat->ops->assemblyend) {
      ierr = (*mat->ops->assemblyend)(mat,type); CHKERRQ(ierr);
    }
  }

  /* Flush assembly is not a true assembly */
  if (type != MAT_FLUSH_ASSEMBLY) {
    mat->assembled  = PETSC_TRUE; mat->num_ass++;
  }
  mat->insertmode = NOT_SET_VALUES;
  MatAssemblyEnd_InUse--;

  if (inassm == 1 && type != MAT_FLUSH_ASSEMBLY) {
    ierr = MatView_Private(mat); CHKERRQ(ierr);
  }
  inassm--;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatCompress"
/*@
   MatCompress - Tries to store the matrix in as little space as 
   possible.  May fail if memory is already fully used, since it
   tries to allocate new space.

   Collective on Mat

   Input Parameters:
.  mat - the matrix 

   Level: advanced

.keywords: matrix, compress
@*/
int MatCompress(Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->ops->compress) {ierr = (*mat->ops->compress)(mat);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption"
/*@
   MatSetOption - Sets a parameter option for a matrix. Some options
   may be specific to certain storage formats.  Some options
   determine how values will be inserted (or added). Sorted, 
   row-oriented input will generally assemble the fastest. The default
   is row-oriented, nonsorted input. 

   Collective on Mat

   Input Parameters:
+  mat - the matrix 
-  option - the option, one of those listed below (and possibly others),
             e.g., MAT_ROWS_SORTED, MAT_NEW_NONZERO_LOCATION_ERR

   Options Describing Matrix Structure:
+    MAT_SYMMETRIC - symmetric in terms of both structure and value
-    MAT_STRUCTURALLY_SYMMETRIC - symmetric nonzero structure

   Options For Use with MatSetValues():
   Insert a logically dense subblock, which can be
+    MAT_ROW_ORIENTED - row-oriented
.    MAT_COLUMN_ORIENTED - column-oriented
.    MAT_ROWS_SORTED - sorted by row
.    MAT_ROWS_UNSORTED - not sorted by row
.    MAT_COLUMNS_SORTED - sorted by column
-    MAT_COLUMNS_UNSORTED - not sorted by column

   Not these options reflect the data you pass in with MatSetValues(); it has 
   nothing to do with how the data is stored internally in the matrix 
   data structure.

   When (re)assembling a matrix, we can restrict the input for
   efficiency/debugging purposes.  These options include
+    MAT_NO_NEW_NONZERO_LOCATIONS - additional insertions will not be
        allowed if they generate a new nonzero
.    MAT_YES_NEW_NONZERO_LOCATIONS - additional insertions will be allowed
.    MAT_NO_NEW_DIAGONALS - additional insertions will not be allowed if
         they generate a nonzero in a new diagonal (for block diagonal format only)
.    MAT_YES_NEW_DIAGONALS - new diagonals will be allowed (for block diagonal format only)
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

   MAT_NO_NEW_NONZERO_LOCATIONS indicates that any add or insertion 
   that would generate a new entry in the nonzero structure is instead
   ignored.  Thus, if memory has not alredy been allocated for this particular 
   data, then the insertion is ignored. For dense matrices, in which
   the entire array is allocated, no entries are ever ignored. 

   MAT_NEW_NONZERO_LOCATION_ERR indicates that any add or insertion 
   that would generate a new entry in the nonzero structure instead produces 
   an error. (Currently supported for AIJ and BAIJ formats only.)
   This is a useful flag when using SAME_NONZERO_PATTERN in calling
   SLESSetOperators() to ensure that the nonzero pattern truely does 
   remain unchanged.

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
   to improve the searching of indices. MAT_NO_NEW_NONZERO_LOCATIONS flag 
   should be used with MAT_USE_HASH_TABLE flag. This option is currently
   supported by MATMPIBAIJ format only.

   Level: intermediate

.keywords: matrix, option, row-oriented, column-oriented, sorted, nonzero
@*/
int MatSetOption(Mat mat,MatOption op)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->ops->setoption) {ierr = (*mat->ops->setoption)(mat,op);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries"
/*@
   MatZeroEntries - Zeros all entries of a matrix.  For sparse matrices
   this routine retains the old nonzero structure.

   Collective on Mat

   Input Parameters:
.  mat - the matrix 

   Level: intermediate

.keywords: matrix, zero, entries

.seealso: MatZeroRows()
@*/
int MatZeroEntries(Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->zeroentries) SETERRQ(PETSC_ERR_SUP,0,"");

  PLogEventBegin(MAT_ZeroEntries,mat,0,0,0);
  ierr = (*mat->ops->zeroentries)(mat); CHKERRQ(ierr);
  PLogEventEnd(MAT_ZeroEntries,mat,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroRows"
/*@ 
   MatZeroRows - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  is - index set of rows to remove
-  diag - pointer to value put in all diagonals of eliminated rows.
          Note that diag is not a pointer to an array, but merely a
          pointer to a single value.

   Notes:
   For the AIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing a null pointer as the final
   argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Level: intermediate

.keywords: matrix, zero, rows, boundary conditions 

.seealso: MatZeroEntries(), 
@*/
int MatZeroRows(Mat mat,IS is, Scalar *diag)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);
  if (diag) PetscValidScalarPointer(diag);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->zerorows) SETERRQ(PETSC_ERR_SUP,0,"");

  ierr = (*mat->ops->zerorows)(mat,is,diag); CHKERRQ(ierr);
  ierr = MatView_Private(mat); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroRowsLocal"
/*@ 
   MatZeroRowsLocal - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix; using local numbering of rows.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  is - index set of rows to remove
-  diag - pointer to value put in all diagonals of eliminated rows.
          Note that diag is not a pointer to an array, but merely a
          pointer to a single value.

   Notes:
   For the AIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing a null pointer as the final
   argument).

   Level: intermediate

.keywords: matrix, zero, rows, boundary conditions 

.seealso: MatZeroEntries(), 
@*/
int MatZeroRowsLocal(Mat mat,IS is, Scalar *diag)
{
  int ierr;
  IS  newis;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);
  if (diag) PetscValidScalarPointer(diag);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (!mat->ops->zerorows) SETERRQ(PETSC_ERR_SUP,0,"");

  ierr = ISLocalToGlobalMappingApplyIS(mat->mapping,is,&newis); CHKERRQ(ierr);
  ierr =  (*mat->ops->zerorows)(mat,newis,diag); CHKERRQ(ierr);
  ierr = ISDestroy(newis);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize"
/*@
   MatGetSize - Returns the numbers of rows and columns in a matrix.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  m - the number of global rows
-  n - the number of global columns

   Level: beginner

.keywords: matrix, dimension, size, rows, columns, global, get

.seealso: MatGetLocalSize()
@*/
int MatGetSize(Mat mat,int *m,int* n)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = (*mat->ops->getsize)(mat,m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetLocalSize"
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

   Level: beginner

.keywords: matrix, dimension, size, local, rows, columns, get

.seealso: MatGetSize()
@*/
int MatGetLocalSize(Mat mat,int *m,int* n)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = (*mat->ops->getlocalsize)(mat,m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange"
/*@
   MatGetOwnershipRange - Returns the range of matrix rows owned by
   this processor, assuming that the matrix is laid out with the first
   n1 rows on the first processor, the next n2 rows on the second, etc.
   For certain parallel layouts this range may not be well defined.

   Not Collective

   Input Parameters:
.  mat - the matrix

   Output Parameters:
+  m - the global index of the first local row
-  n - one more than the global index of the last local row

   Level: beginner

.keywords: matrix, get, range, ownership
@*/
int MatGetOwnershipRange(Mat mat,int *m,int* n)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(m);
  PetscValidIntPointer(n);
  if (!mat->ops->getownershiprange) SETERRQ(PETSC_ERR_SUP,0,"");
  ierr = (*mat->ops->getownershiprange)(mat,m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatILUFactorSymbolic"
/*@  
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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, incomplete, ILU, symbolic, fill

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric()
@*/
int MatILUFactorSymbolic(Mat mat,IS row,IS col,MatILUInfo *info,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(fact);
  if (info && info->levels < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Levels of fill negative %d",info->levels);
  if (info && info->fill < 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Expected fill less than 1.0 %g",info->fill);
  if (!mat->ops->ilufactorsymbolic) SETERRQ(PETSC_ERR_SUP,0,"Only MatCreateMPIRowbs() matrices support parallel ILU");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  PLogEventBegin(MAT_ILUFactorSymbolic,mat,row,col,0);
  ierr = (*mat->ops->ilufactorsymbolic)(mat,row,col,info,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_ILUFactorSymbolic,mat,row,col,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatIncompleteCholeskyFactorSymbolic"
/*@  
   MatIncompleteCholeskyFactorSymbolic - Performs symbolic incomplete
   Cholesky factorization for a symmetric matrix.  Use 
   MatCholeskyFactorNumeric() to complete the factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  perm - row and column permutation
.  fill - levels of fill
-  f - expected fill as ratio of original fill

   Output Parameter:
.  fact - the factored matrix

   Notes:
   Currently only no-fill factorization is supported.

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.keywords: matrix, factor, incomplete, ICC, Cholesky, symbolic, fill

.seealso: MatCholeskyFactorNumeric(), MatCholeskyFactor()
@*/
int MatIncompleteCholeskyFactorSymbolic(Mat mat,IS perm,double f,int fill,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(fact);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 
  if (fill < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Fill negative %d",fill);
  if (f < 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Expected fill less than 1.0 %g",f);
  if (!mat->ops->incompletecholeskyfactorsymbolic) SETERRQ(PETSC_ERR_SUP,0,"Currently only MatCreateMPIRowbs() matrices support ICC in parallel");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");

  PLogEventBegin(MAT_IncompleteCholeskyFactorSymbolic,mat,perm,0,0);
  ierr = (*mat->ops->incompletecholeskyfactorsymbolic)(mat,perm,f,fill,fact);CHKERRQ(ierr);
  PLogEventEnd(MAT_IncompleteCholeskyFactorSymbolic,mat,perm,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetArray"
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

   Currently returns an array only for the dense formats, giving access to
   the local portion of the matrix in the usual Fortran column-oriented format.

   Fortran Note:
   This routine is used differently from Fortran, e.g.,
.vb
        Mat         mat
        Scalar      mat_array(1)
        PetscOffset i_mat
        int         ierr
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

.keywords: matrix, array, elements, values

.seealso: MatRestoreArray(), MatGetArrayF90()
@*/
int MatGetArray(Mat mat,Scalar **v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(v);
  if (!mat->ops->getarray) SETERRQ(PETSC_ERR_SUP,0,"");
  ierr = (*mat->ops->getarray)(mat,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreArray"
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
        Scalar      mat_array(1)
        PetscOffset i_mat
        int         ierr
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

.keywords: matrix, array, elements, values, restore

.seealso: MatGetArray(), MatRestoreArrayF90()
@*/
int MatRestoreArray(Mat mat,Scalar **v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidPointer(v);
  if (!mat->ops->restorearray) SETERRQ(PETSC_ERR_SUP,0,"");
  ierr = (*mat->ops->restorearray)(mat,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices"
/*@C
   MatGetSubMatrices - Extracts several submatrices from a matrix. If submat
   points to an array of valid matrices, they may be reused to store the new
   submatrices.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  n   - the number of submatrixes to be extracted
.  irow, icol - index sets of rows and columns to extract
-  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.  submat - the array of submatrices

   Notes:
   MatGetSubMatrices() can extract only sequential submatrices
   (from both sequential and parallel matrices). Use MatGetSubMatrix()
   to extract a parallel submatrix.

   When extracting submatrices from a parallel matrix, each processor can
   form a different submatrix by setting the rows and columns of its
   individual index sets according to the local submatrix desired.

   When finished using the submatrices, the user should destroy
   them with MatDestroySubMatrices().

   MAT_REUSE_MATRIX can only be used when the nonzero structure of the 
   original matrix has not changed from that last call to MatGetSubMatrices()

   Fortran Note:
   The Fortran interface is slightly different from that given below, it 
   requires one to pass in  as submat a Mat (integer) array of size at least m.

   Level: advanced

.keywords: matrix, get, submatrix, submatrices

.seealso: MatDestroyMatrices(), MatGetSubMatrix(), MatGetRow(), MatGetDiagonal()
@*/
int MatGetSubMatrices(Mat mat,int n,IS *irow,IS *icol,MatReuse scall,Mat **submat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops->getsubmatrices) SETERRQ(PETSC_ERR_SUP,0,"");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");

  PLogEventBegin(MAT_GetSubMatrices,mat,0,0,0);
  ierr = (*mat->ops->getsubmatrices)(mat,n,irow,icol,scall,submat); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetSubMatrices,mat,0,0,0);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroyMatrices"
/*@C
   MatDestroyMatrices - Destroys a set of matrices obtained with MatGetSubMatrices().

   Collective on Mat

   Input Parameters:
+  n - the number of local matrices
-  mat - the matrices

   Level: advanced

.keywords: matrix, destroy, submatrix, submatrices

.seealso: MatGetSubMatrices()
@*/
int MatDestroyMatrices(int n,Mat **mat)
{
  int ierr,i;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Trying to destroy negative number of matrices %d",n);
  PetscValidPointer(mat);
  for ( i=0; i<n; i++ ) {
    ierr = MatDestroy((*mat)[i]); CHKERRQ(ierr);
  }
  if (n) PetscFree(*mat);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatIncreaseOverlap"
/*@
   MatIncreaseOverlap - Given a set of submatrices indicated by index sets,
   replaces the index sets by larger ones that represent submatrices with
   additional overlap.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  n   - the number of index sets
.  is  - the array of pointers to index sets
-  ov  - the additional overlap requested

   Level: developer

.keywords: matrix, overlap, Schwarz

.seealso: MatGetSubMatrices()
@*/
int MatIncreaseOverlap(Mat mat,int n, IS *is,int ov)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor)     SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  if (ov == 0) PetscFunctionReturn(0);
  if (!mat->ops->increaseoverlap) SETERRQ(PETSC_ERR_SUP,0,"");
  PLogEventBegin(MAT_IncreaseOverlap,mat,0,0,0);
  ierr = (*mat->ops->increaseoverlap)(mat,n,is,ov); CHKERRQ(ierr);
  PLogEventEnd(MAT_IncreaseOverlap,mat,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPrintHelp"
/*@
   MatPrintHelp - Prints all the options for the matrix.

   Collective on Mat

   Input Parameter:
.  mat - the matrix 

   Options Database Keys:
+  -help - Prints matrix options
-  -h - Prints matrix options

   Level: developer

.keywords: mat, help

.seealso: MatCreate(), MatCreateXXX()
@*/
int MatPrintHelp(Mat mat)
{
  static int called = 0;
  int        ierr;
  MPI_Comm   comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);

  comm = mat->comm;
  if (!called) {
    ierr = (*PetscHelpPrintf)(comm,"General matrix options:\n"); CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -mat_view_info: view basic matrix info during MatAssemblyEnd()\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -mat_view_info_detailed: view detailed matrix info during MatAssemblyEnd()\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -mat_view_draw: draw nonzero matrix structure during MatAssemblyEnd()\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"      -draw_pause <sec>: set seconds of display pause\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"      -display <name>: set alternate display\n");CHKERRQ(ierr);
    called = 1;
  }
  if (mat->ops->printhelp) {
    ierr = (*mat->ops->printhelp)(mat); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize"
/*@
   MatGetBlockSize - Returns the matrix block size; useful especially for the
   block row and block diagonal formats.
   
   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  bs - block size

   Notes:
   Block diagonal formats are MATSEQBDIAG, MATMPIBDIAG.
   Block row formats are MATSEQBAIJ, MATMPIBAIJ

   Level: intermediate

.keywords: matrix, get, block, size 

.seealso: MatCreateSeqBAIJ(), MatCreateMPIBAIJ(), MatCreateSeqBDiag(), MatCreateMPIBDiag()
@*/
int MatGetBlockSize(Mat mat,int *bs)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(bs);
  if (!mat->ops->getblocksize) SETERRQ(PETSC_ERR_SUP,0,"");
  ierr = (*mat->ops->getblocksize)(mat,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRowIJ"
/*@C
    MatGetRowIJ - Returns the compressed row storage i and j indices for sequential matrices.
    EXPERTS ONLY.

   Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift -  0 or 1 indicating we want the indices starting at 0 or 1
-   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized

    Output Parameters:
+   n - number of rows in the (possibly compressed) matrix
.   ia - the row pointers
.   ja - the column indices
-   done - PETSC_TRUE or PETSC_FALSE, indicating whether the values have been returned

    Level: developer

.seealso: MatGetColumnIJ(), MatRestoreRowIJ()
@*/
int MatGetRowIJ(Mat mat,int shift,PetscTruth symmetric,int *n,int **ia,int** ja,PetscTruth *done)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (ia) PetscValidIntPointer(ia);
  if (ja) PetscValidIntPointer(ja);
  PetscValidIntPointer(done);
  if (!mat->ops->getrowij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->getrowij)(mat,shift,symmetric,n,ia,ja,done); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetColumnIJ"
/*@C
    MatGetColumnIJ - Returns the compressed column storage i and j indices for sequential matrices.
    EXPERTS ONLY.

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
-   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized

    Output Parameters:
+   n - number of columns in the (possibly compressed) matrix
.   ia - the column pointers
.   ja - the row indices
-   done - PETSC_TRUE or PETSC_FALSE, indicating whether the values have been returned

    Level: developer

.seealso: MatGetRowIJ(), MatRestoreColumnIJ()
@*/
int MatGetColumnIJ(Mat mat,int shift,PetscTruth symmetric,int *n,int **ia,int** ja,PetscTruth *done)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (ia) PetscValidIntPointer(ia);
  if (ja) PetscValidIntPointer(ja);
  PetscValidIntPointer(done);

  if (!mat->ops->getcolumnij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->getcolumnij)(mat,shift,symmetric,n,ia,ja,done); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRowIJ"
/*@C
    MatRestoreRowIJ - Call after you are completed with the ia,ja indices obtained with
    MatGetRowIJ(). EXPERTS ONLY.

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
-   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized

    Output Parameters:
+   n - size of (possibly compressed) matrix
.   ia - the row pointers
.   ja - the column indices
-   done - PETSC_TRUE or PETSC_FALSE indicated that the values have been returned

    Level: developer

.seealso: MatGetRowIJ(), MatRestoreColumnIJ()
@*/
int MatRestoreRowIJ(Mat mat,int shift,PetscTruth symmetric,int *n,int **ia,int** ja,PetscTruth *done)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (ia) PetscValidIntPointer(ia);
  if (ja) PetscValidIntPointer(ja);
  PetscValidIntPointer(done);

  if (!mat->ops->restorerowij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->restorerowij)(mat,shift,symmetric,n,ia,ja,done); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreColumnIJ"
/*@C
    MatRestoreColumnIJ - Call after you are completed with the ia,ja indices obtained with
    MatGetColumnIJ(). EXPERTS ONLY.

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
-   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized

    Output Parameters:
+   n - size of (possibly compressed) matrix
.   ia - the column pointers
.   ja - the row indices
-   done - PETSC_TRUE or PETSC_FALSE indicated that the values have been returned

    Level: developer

.seealso: MatGetColumnIJ(), MatRestoreRowIJ()
@*/
int MatRestoreColumnIJ(Mat mat,int shift,PetscTruth symmetric,int *n,int **ia,int** ja,PetscTruth *done)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (ia) PetscValidIntPointer(ia);
  if (ja) PetscValidIntPointer(ja);
  PetscValidIntPointer(done);

  if (!mat->ops->restorecolumnij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->restorecolumnij)(mat,shift,symmetric,n,ia,ja,done); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatColoringPatch"
/*@C
    MatColoringPatch - EXPERTS ONLY, used inside matrix coloring routines that 
    use MatGetRowIJ() and/or MatGetColumnIJ().

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   n   - number of colors
-   colorarray - array indicating color for each column

    Output Parameters:
.   iscoloring - coloring generated using colorarray information

    Level: developer

.seealso: MatGetRowIJ(), MatGetColumnIJ()

.keywords: mat, coloring, patch
@*/
int MatColoringPatch(Mat mat,int n,int *colorarray,ISColoring *iscoloring)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(colorarray);

  if (!mat->ops->coloringpatch) {SETERRQ(PETSC_ERR_SUP,0,"");}
  else {
    ierr  = (*mat->ops->coloringpatch)(mat,n,colorarray,iscoloring); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatSetUnfactored"
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
     PCILUSeUseInPlace(pc);
.ve
   or by using the options -pc_type ilu -pc_ilu_in_place

   In-place factorization ILU(0) can also be used as a local
   solver for the blocks within the block Jacobi or additive Schwarz
   methods (runtime option: -sub_pc_ilu_in_place).  See the discussion 
   of these preconditioners in the users manual for details on setting
   local solver options.

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

.seealso: PCILUSetUseInPlace(), PCLUSetUseInPlace()

.keywords: matrix-free, in-place ILU, in-place LU
@*/
int MatSetUnfactored(Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);  
  mat->factor = 0;
  if (!mat->ops->setunfactored) PetscFunctionReturn(0);
  ierr = (*mat->ops->setunfactored)(mat); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetType"
/*@C
   MatGetType - Gets the matrix type and name (as a string) from the matrix.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
+  type - the matrix type (or use PETSC_NULL)
-  name - name of matrix type (or use PETSC_NULL)

   Level: intermediate

.keywords: matrix, get, type, name
@*/
int MatGetType(Mat mat,MatType *type,char **name)
{
  int  itype = (int)mat->type;
  char *matname[10];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);

  if (type) *type = (MatType) mat->type;
  if (name) {
    /* Note:  Be sure that this list corresponds to the enum in mat.h */
    matname[0] = "MATSEQDENSE";
    matname[1] = "MATSEQAIJ";
    matname[2] = "MATMPIAIJ";
    matname[3] = "MATSHELL";
    matname[4] = "MATMPIROWBS";
    matname[5] = "MATSEQBDIAG";
    matname[6] = "MATMPIBDIAG";
    matname[7] = "MATMPIDENSE";
    matname[8] = "MATSEQBAIJ";
    matname[9] = "MATMPIBAIJ";
    
    if (itype < 0 || itype > 9) *name = "Unknown matrix type";
    else                        *name = matname[itype];
  }
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
      Scalar, pointer xx_v(:)
      ....
      call MatGetArrayF90(x,xx_v,ierr)
      a = xx_v(3)
      call MatRestoreArrayF90(x,xx_v,ierr)
.ve

    Notes:
    Not yet supported for all F90 compilers

    Level: advanced

.seealso:  MatRestoreArrayF90(), MatGetArray(), MatRestoreArray()

.keywords:  matrix, array, f90
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
       Scalar, pointer xx_v(:)
       ....
       call MatGetArrayF90(x,xx_v,ierr)
       a = xx_v(3)
       call MatRestoreArrayF90(x,xx_v,ierr)
.ve
   
    Notes:
    Not yet supported for all F90 compilers

    Level: advanced

.seealso:  MatGetArrayF90(), MatGetArray(), MatRestoreArray()

.keywords:  matrix, array, f90
M*/


#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix"
/*@
    MatGetSubMatrix - Gets a single submatrix on the same number of processors
                      as the original matrix.

    Collective on Mat

    Input Parameters:
+   mat - the original matrix
.   isrow - rows this processor should obtain
.   iscol - columns for all processors you wish to keep
.   csize - number of columns "local" to this processor (does nothing for sequential 
            matrices). This should match the result from VecGetLocalSize(x,...) if you 
            plan to use the matrix in a A*x or use PETSC_DECIDE
-   cll - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

    Output Parameter:
.   newmat - the new submatrix, of the same type as the old

    Level: advanced

.keywords: matrix, get, submatrix, submatrices

.seealso: MatGetSubMatrices()
@*/
int MatGetSubMatrix(Mat mat,IS isrow,IS iscol,int csize,MatReuse cll,Mat *newmat)
{
  int     ierr, size;
  Mat     *local;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr);

  /* if original matrix is on just one processor then use submatrix generated */
  if (!mat->ops->getsubmatrix && size == 1 && cll == MAT_REUSE_MATRIX) {
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_REUSE_MATRIX,&newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (!mat->ops->getsubmatrix && size == 1) {
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&local);CHKERRQ(ierr);
    *newmat = *local;
    PetscFree(local);
    PetscFunctionReturn(0);
  }

  if (!mat->ops->getsubmatrix) SETERRQ(PETSC_ERR_SUP,0,"Not currently implemented");
  ierr = (*mat->ops->getsubmatrix)(mat,isrow,iscol,csize,cll,newmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetMaps"
/*@C
   MatGetMaps - Returns the maps associated with the matrix.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  rmap - the row (right) map
-  cmap - the column (left) map  

   Level: developer

.keywords: matrix, get, map
@*/
int MatGetMaps(Mat mat,Map *rmap,Map *cmap)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = (*mat->ops->getmaps)(mat,rmap,cmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Version that works for all PETSc matrices
*/
#undef __FUNC__  
#define __FUNC__ "MatGetMaps_Petsc"
int MatGetMaps_Petsc(Mat mat,Map *rmap,Map *cmap)
{
  PetscFunctionBegin;
  if (rmap) *rmap = mat->rmap;
  if (cmap) *cmap = mat->cmap;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetStashInitialSize"
/*@
   MatSetStashInitialSize - sets the sizes of the matrix stash, that is
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
     The block-stash is used for values set with VecSetValuesBlocked() while
     the stash is used for values set with VecSetValues()

     Run with the option -log_info and look for output of the form
     MatAssemblyBegin_MPIXXX:Stash has MM entries, uses nn mallocs.
     to determine the appropriate value, MM, to use for size and 
     MatAssemblyBegin_MPIXXX:Block-Stash has BMM entries, uses nn mallocs.
     to determine the value, BMM to use for bsize

.keywords: matrix, stash, assembly
@*/
int MatSetStashInitialSize(Mat mat,int size, int bsize)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = MatStashSetInitialSize_Private(&mat->stash,size); CHKERRQ(ierr);
  ierr = MatStashSetInitialSize_Private(&mat->bstash,bsize); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
