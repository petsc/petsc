/*$Id: matrix.c,v 1.414 2001/09/28 18:57:28 balay Exp $*/

/*
   This is where the abstract matrix operations are defined
*/

#include "src/mat/matimpl.h"        /*I "petscmat.h" I*/
#include "src/vec/vecimpl.h"  

/* Logging support */
int MAT_COOKIE;
int MATSNESMFCTX_COOKIE;
int MAT_Mult, MAT_MultMatrixFree, MAT_MultMultiple, MAT_MultConstrained, MAT_MultAdd, MAT_MultTranspose;
int MAT_MultTransposeConstrained, MAT_MultTransposeAdd, MAT_Solve, MAT_SolveMultiple, MAT_SolveAdd, MAT_SolveTranspose;
int MAT_SolveTransposeAdd, MAT_Relax, MAT_ForwardSolve, MAT_BackwardSolve, MAT_LUFactor, MAT_LUFactorSymbolic;
int MAT_LUFactorNumeric, MAT_CholeskyFactor, MAT_CholeskyFactorSymbolic, MAT_CholeskyFactorNumeric, MAT_ILUFactor;
int MAT_ILUFactorSymbolic, MAT_ICCFactorSymbolic, MAT_Copy, MAT_Convert, MAT_Scale, MAT_AssemblyBegin;
int MAT_AssemblyEnd, MAT_SetValues, MAT_GetValues, MAT_GetRow, MAT_GetSubMatrices, MAT_GetColoring, MAT_GetOrdering;
int MAT_IncreaseOverlap, MAT_Partitioning, MAT_ZeroEntries, MAT_Load, MAT_View, MAT_AXPY, MAT_FDColoringCreate;
int MAT_FDColoringApply,MAT_Transpose;

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
+  ncols -  the number of nonzeros in the row
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
   matrix at a time, per processor. MatGetRow() can only obtained rows
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

.seealso: MatRestoreRow(), MatSetValues(), MatGetValues(), MatGetSubmatrices(), MatGetDiagonal()
@*/
int MatGetRow(Mat mat,int row,int *ncols,int **cols,PetscScalar **vals)
{
  int   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(ncols);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->getrow) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = PetscLogEventBegin(MAT_GetRow,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->getrow)(mat,row,ncols,cols,vals);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_GetRow,mat,0,0,0);CHKERRQ(ierr);
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
int MatRestoreRow(Mat mat,int row,int *ncols,int **cols,PetscScalar **vals)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(ncols);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->restorerow) PetscFunctionReturn(0);
  ierr = (*mat->ops->restorerow)(mat,row,ncols,cols,vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView"
/*@C
   MatView - Visualizes a matrix object.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  ptr - visualization context

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
+    PETSC_VIEWER_ASCII_DEFAULT - default, prints matrix contents
.    PETSC_VIEWER_ASCII_MATLAB - prints matrix contents in Matlab format
.    PETSC_VIEWER_ASCII_DENSE - prints entire matrix including zeros
.    PETSC_VIEWER_ASCII_COMMON - prints matrix contents, using a sparse 
         format common among all matrix types
.    PETSC_VIEWER_ASCII_IMPL - prints matrix contents, using an implementation-specific 
         format (which is in many cases the same as the default)
.    PETSC_VIEWER_ASCII_INFO - prints basic information about the matrix
         size and structure (not the matrix entries)
-    PETSC_VIEWER_ASCII_INFO_LONG - prints more detailed information about
         the matrix structure

   Level: beginner

   Concepts: matrices^viewing
   Concepts: matrices^plotting
   Concepts: matrices^printing

.seealso: PetscViewerSetFormat(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), 
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(), MatLoad()
@*/
int MatView(Mat mat,PetscViewer viewer)
{
  int               ierr,rows,cols;
  PetscTruth        isascii;
  char              *cstr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat); 
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(mat->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  PetscCheckSameComm(mat,viewer);
  if (!mat->assembled) SETERRQ(1,"Must call MatAssemblyBegin/End() before viewing matrix");

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);  
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_LONG) {
      if (mat->prefix) {
        ierr = PetscViewerASCIIPrintf(viewer,"Matrix Object:(%s)\n",mat->prefix);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Matrix Object:\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = MatGetType(mat,&cstr);CHKERRQ(ierr);
      ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"type=%s, rows=%d, cols=%d\n",cstr,rows,cols);CHKERRQ(ierr);
      if (mat->ops->getinfo) {
        MatInfo info;
        ierr = MatGetInfo(mat,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"total: nonzeros=%d, allocated nonzeros=%d\n",
                          (int)info.nz_used,(int)info.nz_allocated);CHKERRQ(ierr);
      }
    }
  }
  if (mat->ops->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*mat->ops->view)(mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported",((PetscObject)viewer)->type_name);
  }
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);  
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_LONG) {
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScaleSystem"
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

   Concepts: matrices^scaling

.seealso: MatUseScaledForm(), MatUnScaleSystem()
@*/
int MatScaleSystem(Mat mat,Vec x,Vec b)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (x) {PetscValidHeaderSpecific(x,VEC_COOKIE);PetscCheckSameComm(mat,x);}
  if (b) {PetscValidHeaderSpecific(b,VEC_COOKIE);PetscCheckSameComm(mat,b);}

  if (mat->ops->scalesystem) {
    ierr = (*mat->ops->scalesystem)(mat,x,b);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUnScaleSystem"
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

.seealso: MatUseScaledForm(), MatScaleSystem()
@*/
int MatUnScaleSystem(Mat mat,Vec x,Vec b)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (x) {PetscValidHeaderSpecific(x,VEC_COOKIE);PetscCheckSameComm(mat,x);}
  if (b) {PetscValidHeaderSpecific(b,VEC_COOKIE);PetscCheckSameComm(mat,b);}
  if (mat->ops->unscalesystem) {
    ierr = (*mat->ops->unscalesystem)(mat,x,b);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseScaledForm"
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

.seealso: MatScaleSystem(), MatUnScaleSystem()
@*/
int MatUseScaledForm(Mat mat,PetscTruth scaled)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (mat->ops->usescaledform) {
    ierr = (*mat->ops->usescaledform)(mat,scaled);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy"
/*@C
   MatDestroy - Frees space taken by a matrix.
  
   Collective on Mat

   Input Parameter:
.  A - the matrix

   Level: beginner

@*/
int MatDestroy(Mat A)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  PetscValidType(A);
  MatPreallocated(A);
  if (--A->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(A);CHKERRQ(ierr);
  if (A->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->mapping);CHKERRQ(ierr);
  }
  if (A->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->bmapping);CHKERRQ(ierr);
  }
  if (A->rmap) {
    ierr = PetscMapDestroy(A->rmap);CHKERRQ(ierr);
  }
  if (A->cmap) {
    ierr = PetscMapDestroy(A->cmap);CHKERRQ(ierr);
  }

  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscLogObjectDestroy(A);
  PetscHeaderDestroy(A);
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
int MatValid(Mat m,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidIntPointer(flg);
  if (!m)                           *flg = PETSC_FALSE;
  else if (m->cookie != MAT_COOKIE) *flg = PETSC_FALSE;
  else                              *flg = PETSC_TRUE;
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

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal()
@*/
int MatSetValues(Mat mat,int m,int *idxm,int n,int *idxn,PetscScalar *v,InsertMode addv)
{
  int ierr;

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_BOPT_g)
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
  if (!mat->ops->setvalues) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesStencil"
/*@C 
   MatSetValuesStencil - Inserts or adds a block of values into a matrix.
     Using structured grid indexing

   Not Collective

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array of values
.  m - number of rows being entered
.  idxm - grid coordinates (and component number when dof > 1) for matrix rows being entered
.  n - number of columns being entered
.  idxn - grid coordinates (and component number when dof > 1) for matrix columns being entered 
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented and unsorted.
   See MatSetOption() for other options.

   Calls to MatSetValuesStencil() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   The grid coordinates are across the entire grid, not just the local portion

   MatSetValuesStencil() uses 0-based row and column numbers in Fortran 
   as well as in C.

   In order to use this routine you must either obtain the matrix with DAGetMatrix()
   or call MatSetLocalToGlobalMapping() and MatSetStencil() first.

   In Fortran idxm and idxn should be declared as
$     MatStencil idxm(4,m),idxn(4,n)
   and the values inserted using
$    idxm(MatStencil_i,1) = i
$    idxm(MatStencil_j,1) = j
$    idxm(MatStencil_k,1) = k
$    idxm(MatStencil_c,1) = c
   etc

   Negative indices may be passed in idxm and idxn, these rows and columns are 
   simply ignored. This allows easily inserting element stiffness matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the matrix.

   Inspired by the structured grid interface to the HYPRE package
   (http://www.llnl.gov/CASC/hypre)

   Efficiency Alert:
   The routine MatSetValuesBlockedStencil() may offer much better efficiency
   for users of block sparse formats (MATSEQBAIJ and MATMPIBAIJ).

   Level: beginner

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal()
          MatSetValues(), MatSetValuesBlockedStencil(), MatSetStencil(), DAGetMatrix()
@*/
int MatSetValuesStencil(Mat mat,int m,MatStencil *idxm,int n,MatStencil *idxn,PetscScalar *v,InsertMode addv)
{
  int j,i,ierr,jdxm[128],jdxn[256],dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
  int *starts = mat->stencil.starts,*dxm = (int*)idxm,*dxn = (int*)idxn,sdim = dim - (1 - (int)mat->stencil.noc);

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);

  if (m > 128) SETERRQ1(1,"Can only set 128 rows at a time; trying to set %d",m);
  if (n > 128) SETERRQ1(1,"Can only set 256 columns at a time; trying to set %d",n);

  for (i=0; i<m; i++) {
    for (j=0; j<3-sdim; j++) dxm++;  
    tmp = *dxm++ - starts[0];
    for (j=0; j<dim-1; j++) {
      tmp = tmp*dims[j] + *dxm++ - starts[j+1];
    }
    if (mat->stencil.noc) dxm++;
    jdxm[i] = tmp;
  }
  for (i=0; i<n; i++) {
    for (j=0; j<3-sdim; j++) dxn++;  
    tmp = *dxn++ - starts[0];
    for (j=0; j<dim-1; j++) {
      tmp = tmp*dims[j] + *dxn++ - starts[j+1];
    }
    if (mat->stencil.noc) dxn++;
    jdxn[i] = tmp;
  }
  ierr = MatSetValuesLocal(mat,m,jdxm,n,jdxn,v,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetStencil"
/*@ 
   MatSetStencil - Sets the grid information for setting values into a matrix via
        MatSetStencil()

   Not Collective

   Input Parameters:
+  mat - the matrix
.  dim - dimension of the grid 1,2, or 3
.  dims - number of grid points in x, y, and z direction, including ghost points on your processor
.  starts - starting point of ghost nodes on your processor in x, y, and z direction 
-  dof - number of degrees of freedom per node


   Inspired by the structured grid interface to the HYPRE package
   (www.llnl.gov/CASC/hyper)

   Level: beginner

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal()
          MatSetValues(), MatSetValuesBlockedStencil(), MatSetValuesStencil()
@*/
int MatSetStencil(Mat mat,int dim,int *dims,int *starts,int dof)
{
  int i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(dims);
  PetscValidIntPointer(starts);

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

   Concepts: matrices^putting entries in blocked

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), MatSetValuesBlockedLocal()
@*/
int MatSetValuesBlocked(Mat mat,int m,int *idxm,int n,int *idxn,PetscScalar *v,InsertMode addv)
{
  int ierr;

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_BOPT_g) 
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
  if (!mat->ops->setvaluesblocked) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->setvaluesblocked)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MatSetValue - Set a single entry into a matrix.

   Synopsis:
   int MatSetValue(Mat m,int row,int col,PetscScalar value,InsertMode mode);

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

   Level: advanced

   Concepts: matrices^accessing values

.seealso: MatGetRow(), MatGetSubmatrices(), MatSetValues()
@*/
int MatGetValues(Mat mat,int m,int *idxm,int n,int *idxn,PetscScalar *v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->getvalues) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

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
int MatSetLocalToGlobalMapping(Mat x,ISLocalToGlobalMapping mapping)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,MAT_COOKIE);
  PetscValidType(x);
  MatPreallocated(x);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);
  if (x->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for matrix");
  }

  if (x->ops->setlocaltoglobalmapping) {
    ierr = (*x->ops->setlocaltoglobalmapping)(x,mapping);CHKERRQ(ierr);
  } else {
    x->mapping = mapping;
    ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
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
int MatSetLocalToGlobalMappingBlock(Mat x,ISLocalToGlobalMapping mapping)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,MAT_COOKIE);
  PetscValidType(x);
  MatPreallocated(x);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);
  if (x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for matrix");
  }
 
  x->bmapping = mapping;
  ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
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
int MatSetValuesLocal(Mat mat,int nrow,int *irow,int ncol,int *icol,PetscScalar *y,InsertMode addv) 
{
  int ierr,irowm[2048],icolm[2048];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidIntPointer(irow);
  PetscValidIntPointer(icol);
  PetscValidScalarPointer(y);

  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_BOPT_g) 
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
  if (!mat->ops->setvalueslocal && (nrow > 2048 || ncol > 2048)) {
    SETERRQ2(PETSC_ERR_SUP,"Number column/row indices must be <= 2048: are %d %d",nrow,ncol);
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
   local-to-global mapping by calling MatSetLocalToGlobalMappingBlock(),
   where the mapping MUST be set for matrix blocks, not for matrix elements.

   Calls to MatSetValuesBlockedLocal() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd() 
   MUST be called after all calls to MatSetValuesBlockedLocal() have been completed.

   Level: intermediate

   Concepts: matrices^putting blocked values in with local numbering

.seealso:  MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesLocal(), MatSetLocalToGlobalMappingBlock(), MatSetValuesBlocked()
@*/
int MatSetValuesBlockedLocal(Mat mat,int nrow,int *irow,int ncol,int *icol,PetscScalar *y,InsertMode addv) 
{
  int ierr,irowm[2048],icolm[2048];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidIntPointer(irow);
  PetscValidIntPointer(icol);
  PetscValidScalarPointer(y);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_BOPT_g) 
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
  if (!mat->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Local to global never set with MatSetLocalToGlobalMappingBlock()");
  }
  if (nrow > 2048 || ncol > 2048) {
    SETERRQ2(PETSC_ERR_SUP,"Number column/row indices must be <= 2048: are %d %d",nrow,ncol);
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
  ierr = (*mat->ops->setvaluesblocked)(mat,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
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
-  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMult(A,y,y).

   Level: beginner

   Concepts: matrix-vector product

.seealso: MatMultTranspose(), MatMultAdd(), MatMultTransposeAdd()
@*/
int MatMult(Mat mat,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE); 

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
#ifndef PETSC_HAVE_CONSTRAINTS
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->N,x->N); 
  if (mat->M != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %d %d",mat->M,y->N); 
  if (mat->m != y->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: local dim %d %d",mat->m,y->n); 
#endif

  if (mat->nullsp) {
    ierr = MatNullSpaceRemove(mat->nullsp,x,&x);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(MAT_Mult,mat,x,y,0);CHKERRQ(ierr);
  ierr = (*mat->ops->mult)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Mult,mat,x,y,0);CHKERRQ(ierr);

  if (mat->nullsp) {
    ierr = MatNullSpaceRemove(mat->nullsp,y,PETSC_NULL);CHKERRQ(ierr);
  }
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
int MatMultTranspose(Mat mat,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(x,VEC_COOKIE); 
  PetscValidHeaderSpecific(y,VEC_COOKIE);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
#ifndef PETSC_HAVE_CONSTRAINTS
  if (mat->M != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->M,x->N); 
  if (mat->N != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %d %d",mat->N,y->N);
#endif

  if (!mat->ops->multtranspose) SETERRQ(PETSC_ERR_SUP, "Operation not supported");
  ierr = PetscLogEventBegin(MAT_MultTranspose,mat,x,y,0);CHKERRQ(ierr);
  if (!mat->ops->multtranspose) SETERRQ(PETSC_ERR_SUP,"This matrix type does not have a multiply tranpose defined");
  ierr = (*mat->ops->multtranspose)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultTranspose,mat,x,y,0);CHKERRQ(ierr);
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
int MatMultAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(v1,VEC_COOKIE);
  PetscValidHeaderSpecific(v2,VEC_COOKIE); 
  PetscValidHeaderSpecific(v3,VEC_COOKIE);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (mat->N != v1->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v1: global dim %d %d",mat->N,v1->N);
  if (mat->M != v2->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: global dim %d %d",mat->M,v2->N);
  if (mat->M != v3->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: global dim %d %d",mat->M,v3->N);
  if (mat->m != v3->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: local dim %d %d",mat->m,v3->n); 
  if (mat->m != v2->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: local dim %d %d",mat->m,v2->n); 
  if (v1 == v3) SETERRQ(PETSC_ERR_ARG_IDN,"v1 and v3 must be different vectors");

  ierr = PetscLogEventBegin(MAT_MultAdd,mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = (*mat->ops->multadd)(mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultAdd,mat,v1,v2,v3);CHKERRQ(ierr);
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
int MatMultTransposeAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(v1,VEC_COOKIE);
  PetscValidHeaderSpecific(v2,VEC_COOKIE);
  PetscValidHeaderSpecific(v3,VEC_COOKIE);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->multtransposeadd) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  if (v1 == v3) SETERRQ(PETSC_ERR_ARG_IDN,"v1 and v3 must be different vectors");
  if (mat->M != v1->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v1: global dim %d %d",mat->M,v1->N);
  if (mat->N != v2->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: global dim %d %d",mat->N,v2->N);
  if (mat->N != v3->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: global dim %d %d",mat->N,v3->N);

  ierr = PetscLogEventBegin(MAT_MultTransposeAdd,mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = (*mat->ops->multtransposeadd)(mat,v1,v2,v3);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultTransposeAdd,mat,v1,v2,v3);CHKERRQ(ierr);
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
.seealso: MatMult(), MatMultTrans(), MatMultAdd(), MatMultTransAdd()
@*/
int MatMultConstrained(Mat mat,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE); 
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->N,x->N); 
  if (mat->M != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %d %d",mat->M,y->N); 
  if (mat->m != y->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: local dim %d %d",mat->m,y->n); 

  ierr = PetscLogEventBegin(MAT_MultConstrained,mat,x,y,0);CHKERRQ(ierr);
  ierr = (*mat->ops->multconstrained)(mat,x,y); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultConstrained,mat,x,y,0);CHKERRQ(ierr);

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
.seealso: MatMult(), MatMultTrans(), MatMultAdd(), MatMultTransAdd()
@*/
int MatMultTransposeConstrained(Mat mat,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE); 
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (x == y) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  if (mat->M != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->N,x->N); 
  if (mat->N != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %d %d",mat->M,y->N); 

  ierr = PetscLogEventBegin(MAT_MultConstrained,mat,x,y,0);CHKERRQ(ierr);
  ierr = (*mat->ops->multtransposeconstrained)(mat,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultConstrained,mat,x,y,0);CHKERRQ(ierr);

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
   factorization, etc.).  Much of this info is printed to STDOUT
   when using the runtime options 
$       -log_info -mat_view_info

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
 
@*/
int MatGetInfo(Mat mat,MatInfoType flag,MatInfo *info)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(info);
  if (!mat->ops->getinfo) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->getinfo)(mat,flag,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}   

/* ----------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatILUDTFactor"
/*@C  
   MatILUDTFactor - Performs a drop tolerance ILU factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  info - information about the factorization to be done
.  row - row permutation
-  col - column permutation

   Output Parameters:
.  fact - the factored matrix

   Level: developer

   Notes:
   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   This is currently only supported for the SeqAIJ matrix format using code
   from Yousef Saad's SPARSEKIT2  package (translated to C with f2c) and/or
   Matlab. SPARSEKIT2 is copyrighted by Yousef Saad with the GNU copyright
   and thus can be distributed with PETSc.

    Concepts: matrices^ILUDT factorization

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor(), MatILUInfo
@*/
int MatILUDTFactor(Mat mat,MatILUInfo *info,IS row,IS col,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(fact);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->iludtfactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = PetscLogEventBegin(MAT_ILUFactor,mat,row,col,0);CHKERRQ(ierr);
  ierr = (*mat->ops->iludtfactor)(mat,info,row,col,fact);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ILUFactor,mat,row,col,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactor"
/*@  
   MatLUFactor - Performs in-place LU factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row permutation
.  col - column permutation
-  info - options for factorization, includes 
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -log_info to determine an optimal value to use

   Notes:
   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   This changes the state of the matrix to a factored matrix; it cannot be used
   for example with MatSetValues() unless one first calls MatSetUnfactored().

   Level: developer

   Concepts: matrices^LU factorization

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor(),
          MatGetOrdering(), MatSetUnfactored(), MatLUInfo

@*/
int MatLUFactor(Mat mat,IS row,IS col,MatLUInfo *info)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->lufactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = PetscLogEventBegin(MAT_LUFactor,mat,row,col,0);CHKERRQ(ierr);
  ierr = (*mat->ops->lufactor)(mat,row,col,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_LUFactor,mat,row,col,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactor"
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

   Concepts: matrices^ILU factorization

.seealso: MatILUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor(), MatILUInfo
@*/
int MatILUFactor(Mat mat,IS row,IS col,MatILUInfo *info)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,"matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->ilufactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = PetscLogEventBegin(MAT_ILUFactor,mat,row,col,0);CHKERRQ(ierr);
  ierr = (*mat->ops->ilufactor)(mat,row,col,info);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ILUFactor,mat,row,col,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic"
/*@  
   MatLUFactorSymbolic - Performs symbolic LU factorization of matrix.
   Call this routine before calling MatLUFactorNumeric().

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row, col - row and column permutations
-  info - options for factorization, includes 
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -log_info to determine an optimal value to use

   Output Parameter:
.  fact - new matrix that has been symbolically factored

   Notes:
   See the users manual for additional information about
   choosing the fill factor for better efficiency.

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

   Concepts: matrices^LU symbolic factorization

.seealso: MatLUFactor(), MatLUFactorNumeric(), MatCholeskyFactor(), MatLUInfo
@*/
int MatLUFactorSymbolic(Mat mat,IS row,IS col,MatLUInfo *info,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(fact);
  PetscValidHeaderSpecific(row,IS_COOKIE);
  PetscValidHeaderSpecific(col,IS_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->lufactorsymbolic) SETERRQ1(PETSC_ERR_SUP,"Matrix type %s  symbolic LU",mat->type_name);

  ierr = PetscLogEventBegin(MAT_LUFactorSymbolic,mat,row,col,0);CHKERRQ(ierr);
  ierr = (*mat->ops->lufactorsymbolic)(mat,row,col,info,fact);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_LUFactorSymbolic,mat,row,col,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric"
/*@  
   MatLUFactorNumeric - Performs numeric LU factorization of a matrix.
   Call this routine after first calling MatLUFactorSymbolic().

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  fact - the matrix generated for the factor, from MatLUFactorSymbolic()

   Notes:
   See MatLUFactor() for in-place factorization.  See 
   MatCholeskyFactorNumeric() for the symmetric, positive definite case.  

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

   Concepts: matrices^LU numeric factorization

.seealso: MatLUFactorSymbolic(), MatLUFactor(), MatCholeskyFactor()
@*/
int MatLUFactorNumeric(Mat mat,Mat *fact)
{
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(fact);
  PetscValidHeaderSpecific(*fact,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->M != (*fact)->M || mat->N != (*fact)->N) {
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat mat,Mat *fact: global dimensions are different %d should = %d %d should = %d",
            mat->M,(*fact)->M,mat->N,(*fact)->N);
  }
  if (!(*fact)->ops->lufactornumeric) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = PetscLogEventBegin(MAT_LUFactorNumeric,mat,*fact,0,0);CHKERRQ(ierr);
  ierr = (*(*fact)->ops->lufactornumeric)(mat,fact);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_LUFactorNumeric,mat,*fact,0,0);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mat_view_draw",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsHasName(PETSC_NULL,"-mat_view_contour",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPushFormat(PETSC_VIEWER_DRAW_(mat->comm),PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);
    }
    ierr = MatView(*fact,PETSC_VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPopFormat(PETSC_VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactor"
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

   Concepts: matrices^Cholesky factorization

.seealso: MatLUFactor(), MatCholeskyFactorSymbolic(), MatCholeskyFactorNumeric()
          MatGetOrdering()

@*/
int MatCholeskyFactor(Mat mat,IS perm,PetscReal f)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(perm,IS_COOKIE);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->choleskyfactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = PetscLogEventBegin(MAT_CholeskyFactor,mat,perm,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->choleskyfactor)(mat,perm,f);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_CholeskyFactor,mat,perm,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic"
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

   Concepts: matrices^Cholesky symbolic factorization

.seealso: MatLUFactorSymbolic(), MatCholeskyFactor(), MatCholeskyFactorNumeric()
          MatGetOrdering()

@*/
int MatCholeskyFactorSymbolic(Mat mat,IS perm,PetscReal f,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(fact);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->choleskyfactorsymbolic) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = PetscLogEventBegin(MAT_CholeskyFactorSymbolic,mat,perm,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->choleskyfactorsymbolic)(mat,perm,f,fact);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_CholeskyFactorSymbolic,mat,perm,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric"
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

   Concepts: matrices^Cholesky numeric factorization

.seealso: MatCholeskyFactorSymbolic(), MatCholeskyFactor(), MatLUFactorNumeric()
@*/
int MatCholeskyFactorNumeric(Mat mat,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(fact);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!(*fact)->ops->choleskyfactornumeric) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  if (mat->M != (*fact)->M || mat->N != (*fact)->N) {
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat mat,Mat *fact: global dim %d should = %d %d should = %d",
            mat->M,(*fact)->M,mat->N,(*fact)->N);
  }

  ierr = PetscLogEventBegin(MAT_CholeskyFactorNumeric,mat,*fact,0,0);CHKERRQ(ierr);
  ierr = (*(*fact)->ops->choleskyfactornumeric)(mat,fact);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_CholeskyFactorNumeric,mat,*fact,0,0);CHKERRQ(ierr);
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
   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolveAdd(), MatSolveTranspose(), MatSolveTransposeAdd()
@*/
int MatSolve(Mat mat,Vec b,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(b,VEC_COOKIE); 
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscCheckSameComm(mat,b);
  PetscCheckSameComm(mat,x);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %d %d",mat->m,b->n); 
  if (mat->M == 0 && mat->N == 0) PetscFunctionReturn(0);

  if (!mat->ops->solve) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = PetscLogEventBegin(MAT_Solve,mat,b,x,0);CHKERRQ(ierr);
  ierr = (*mat->ops->solve)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Solve,mat,b,x,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatForwardSolve"
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

   Concepts: matrices^forward solves

.seealso: MatSolve(), MatBackwardSolve()
@ */
int MatForwardSolve(Mat mat,Vec b,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(b,VEC_COOKIE); 
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscCheckSameComm(mat,b);
  PetscCheckSameComm(mat,x);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (!mat->ops->forwardsolve) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %d %d",mat->m,b->n); 

  ierr = PetscLogEventBegin(MAT_ForwardSolve,mat,b,x,0);CHKERRQ(ierr);
  ierr = (*mat->ops->forwardsolve)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ForwardSolve,mat,b,x,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatBackwardSolve"
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

   Concepts: matrices^backward solves

.seealso: MatSolve(), MatForwardSolve()
@ */
int MatBackwardSolve(Mat mat,Vec b,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(b,VEC_COOKIE); 
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscCheckSameComm(mat,b);
  PetscCheckSameComm(mat,x);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (!mat->ops->backwardsolve) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %d %d",mat->m,b->n); 

  ierr = PetscLogEventBegin(MAT_BackwardSolve,mat,b,x,0);CHKERRQ(ierr);
  ierr = (*mat->ops->backwardsolve)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_BackwardSolve,mat,b,x,0);CHKERRQ(ierr);
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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolve(), MatSolveTranspose(), MatSolveTransposeAdd()
@*/
int MatSolveAdd(Mat mat,Vec b,Vec y,Vec x)
{
  PetscScalar one = 1.0;
  Vec    tmp;
  int    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscCheckSameComm(mat,b);
  PetscCheckSameComm(mat,y);
  PetscCheckSameComm(mat,x);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->M != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %d %d",mat->M,y->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %d %d",mat->m,b->n); 
  if (x->n != y->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Vec x,Vec y: local dim %d %d",x->n,y->n); 

  ierr = PetscLogEventBegin(MAT_SolveAdd,mat,b,x,y);CHKERRQ(ierr);
  if (mat->ops->solveadd)  {
    ierr = (*mat->ops->solveadd)(mat,b,y,x);CHKERRQ(ierr);
  } else {
    /* do the solve then the add manually */
    if (x != y) {
      ierr = MatSolve(mat,b,x);CHKERRQ(ierr);
      ierr = VecAXPY(&one,y,x);CHKERRQ(ierr);
    } else {
      ierr = VecDuplicate(x,&tmp);CHKERRQ(ierr);
      PetscLogObjectParent(mat,tmp);
      ierr = VecCopy(x,tmp);CHKERRQ(ierr);
      ierr = MatSolve(mat,b,x);CHKERRQ(ierr);
      ierr = VecAXPY(&one,tmp,x);CHKERRQ(ierr);
      ierr = VecDestroy(tmp);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(MAT_SolveAdd,mat,b,x,y);CHKERRQ(ierr);
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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolve(), MatSolveAdd(), MatSolveTransposeAdd()
@*/
int MatSolveTranspose(Mat mat,Vec b,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(b,VEC_COOKIE); 
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscCheckSameComm(mat,b);
  PetscCheckSameComm(mat,x);
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->ops->solvetranspose) SETERRQ1(PETSC_ERR_SUP,"Matrix type %s",mat->type_name);
  if (mat->M != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->M,x->N);
  if (mat->N != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %d %d",mat->N,b->N);

  ierr = PetscLogEventBegin(MAT_SolveTranspose,mat,b,x,0);CHKERRQ(ierr);
  ierr = (*mat->ops->solvetranspose)(mat,b,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SolveTranspose,mat,b,x,0);CHKERRQ(ierr);
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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

   Concepts: matrices^triangular solves

.seealso: MatSolve(), MatSolveAdd(), MatSolveTranspose()
@*/
int MatSolveTransposeAdd(Mat mat,Vec b,Vec y,Vec x)
{
  PetscScalar one = 1.0;
  int         ierr;
  Vec         tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscCheckSameComm(mat,b);
  PetscCheckSameComm(mat,y);
  PetscCheckSameComm(mat,x);
  if (x == b) SETERRQ(PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  if (!mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (mat->M != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->M,x->N);
  if (mat->N != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %d %d",mat->N,b->N);
  if (mat->N != y->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %d %d",mat->N,y->N);
  if (x->n != y->n)   SETERRQ2(PETSC_ERR_ARG_SIZ,"Vec x,Vec y: local dim %d %d",x->n,y->n);

  ierr = PetscLogEventBegin(MAT_SolveTransposeAdd,mat,b,x,y);CHKERRQ(ierr);
  if (mat->ops->solvetransposeadd) {
    ierr = (*mat->ops->solvetransposeadd)(mat,b,y,x);CHKERRQ(ierr);
  } else {
    /* do the solve then the add manually */
    if (x != y) {
      ierr = MatSolveTranspose(mat,b,x);CHKERRQ(ierr);
      ierr = VecAXPY(&one,y,x);CHKERRQ(ierr);
    } else {
      ierr = VecDuplicate(x,&tmp);CHKERRQ(ierr);
      PetscLogObjectParent(mat,tmp);
      ierr = VecCopy(x,tmp);CHKERRQ(ierr);
      ierr = MatSolveTranspose(mat,b,x);CHKERRQ(ierr);
      ierr = VecAXPY(&one,tmp,x);CHKERRQ(ierr);
      ierr = VecDestroy(tmp);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(MAT_SolveTransposeAdd,mat,b,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatRelax"
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
-  lits - the number of local iterations 

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

   Concepts: matrices^relaxation
   Concepts: matrices^SOR
   Concepts: matrices^Gauss-Seidel

@*/
int MatRelax(Mat mat,Vec b,PetscReal omega,MatSORType flag,PetscReal shift,int its,int lits,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(b,VEC_COOKIE); 
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscCheckSameComm(mat,b);
  PetscCheckSameComm(mat,x);
  if (!mat->ops->relax) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (mat->N != x->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %d %d",mat->N,x->N);
  if (mat->M != b->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %d %d",mat->M,b->N);
  if (mat->m != b->n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %d %d",mat->m,b->n);

  ierr = PetscLogEventBegin(MAT_Relax,mat,b,x,0);CHKERRQ(ierr);
  ierr =(*mat->ops->relax)(mat,b,omega,flag,shift,its,lits,x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Relax,mat,b,x,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_Basic"
/*
      Default matrix copy routine.
*/
int MatCopy_Basic(Mat A,Mat B,MatStructure str)
{
  int         ierr,i,rstart,rend,nz,*cwork;
  PetscScalar *vwork;

  PetscFunctionBegin;
  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    ierr = MatSetValues(B,1,&i,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy"
/*@C  
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

.seealso: MatConvert()
@*/
int MatCopy(Mat A,Mat B,MatStructure str)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  PetscValidHeaderSpecific(B,MAT_COOKIE);
  PetscValidType(A);
  MatPreallocated(A);
  PetscValidType(B);
  MatPreallocated(B);
  PetscCheckSameComm(A,B);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (A->M != B->M || A->N != B->N) SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim (%d,%d) (%d,%d)",A->M,B->M,
                                             A->N,B->N);

  ierr = PetscLogEventBegin(MAT_Copy,A,B,0,0);CHKERRQ(ierr);
  if (A->ops->copy) { 
    ierr = (*A->ops->copy)(A,B,str);CHKERRQ(ierr);
  } else { /* generic conversion */
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  if (A->mapping) {
    if (B->mapping) {ierr = ISLocalToGlobalMappingDestroy(B->mapping);CHKERRQ(ierr);}
    ierr = MatSetLocalToGlobalMapping(B,A->mapping);CHKERRQ(ierr);
  }
  if (A->bmapping) {
    if (B->bmapping) {ierr = ISLocalToGlobalMappingDestroy(B->bmapping);CHKERRQ(ierr);}
    ierr = MatSetLocalToGlobalMappingBlock(B,A->mapping);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_Copy,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "petscsys.h"
PetscTruth MatConvertRegisterAllCalled = PETSC_FALSE;
PetscFList MatConvertList              = 0;

#undef __FUNCT__  
#define __FUNCT__ "MatConvertRegister"
/*@C
    MatConvertRegister - Allows one to register a routine that reads matrices
        from a binary file for a particular matrix type.

  Not Collective

  Input Parameters:
+   type - the type of matrix (defined in include/petscmat.h), for example, MATSEQAIJ.
-   Converter - the function that reads the matrix from the binary file.

  Level: developer

.seealso: MatConvertRegisterAll(), MatConvert()

@*/
int MatConvertRegister(char *sname,char *path,char *name,int (*function)(Mat,MatType,Mat*))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatConvertList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
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
-  newtype - new matrix type.  Use MATSAME to create a new matrix of the
   same type as the original matrix.

   Output Parameter:
.  M - pointer to place new matrix

   Notes:
   MatConvert() first creates a new matrix and then copies the data from
   the first matrix.  A related routine is MatCopy(), which copies the matrix
   entries of one matrix to another already existing matrix context.

   Level: intermediate

   Concepts: matrices^converting between storage formats

.seealso: MatCopy(), MatDuplicate()
@*/
int MatConvert(Mat mat,MatType newtype,Mat *M)
{
  int        ierr;
  PetscTruth sametype,issame,flg;
  char       convname[256],mtype[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(M);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscOptionsGetString(PETSC_NULL,"-matconvert_type",mtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    newtype = mtype;
  }
  ierr = PetscLogEventBegin(MAT_Convert,mat,0,0,0);CHKERRQ(ierr);
  
  ierr = PetscTypeCompare((PetscObject)mat,newtype,&sametype);CHKERRQ(ierr);
  ierr = PetscStrcmp(newtype,"same",&issame);CHKERRQ(ierr);
  if ((sametype || issame) && mat->ops->duplicate) {
    ierr = (*mat->ops->duplicate)(mat,MAT_COPY_VALUES,M);CHKERRQ(ierr);
  } else {
    int (*conv)(Mat,MatType,Mat*);
    if (!MatConvertRegisterAllCalled) {
      ierr = MatConvertRegisterAll(PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscFListFind(mat->comm,MatConvertList,newtype,(void(**)(void))&conv);CHKERRQ(ierr);
    if (conv) {
      ierr = (*conv)(mat,newtype,M);CHKERRQ(ierr);
    } else {
      ierr = PetscStrcpy(convname,"MatConvert_");CHKERRQ(ierr);
      ierr = PetscStrcat(convname,mat->type_name);CHKERRQ(ierr);
      ierr = PetscStrcat(convname,"_");CHKERRQ(ierr);
      ierr = PetscStrcat(convname,newtype);CHKERRQ(ierr);
      ierr = PetscStrcat(convname,"_C");CHKERRQ(ierr);
      ierr = PetscObjectQueryFunction((PetscObject)mat,convname,(void (**)(void))&conv);CHKERRQ(ierr);
      if (conv) {
        ierr = (*conv)(mat,newtype,M);CHKERRQ(ierr);
      } else {
        if (mat->ops->convert) {
          ierr = (*mat->ops->convert)(mat,newtype,M);CHKERRQ(ierr);
        } else {
          ierr = MatConvert_Basic(mat,newtype,M);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscLogEventEnd(MAT_Convert,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate"
/*@C  
   MatDuplicate - Duplicates a matrix including the non-zero structure.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  op - either MAT_DO_NOT_COPY_VALUES or MAT_COPY_VALUES, cause it to copy nonzero
        values as well or not

   Output Parameter:
.  M - pointer to place new matrix

   Level: intermediate

   Concepts: matrices^duplicating

.seealso: MatCopy(), MatConvert()
@*/
int MatDuplicate(Mat mat,MatDuplicateOption op,Mat *M)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(M);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  *M  = 0;
  ierr = PetscLogEventBegin(MAT_Convert,mat,0,0,0);CHKERRQ(ierr);
  if (!mat->ops->duplicate) {
    SETERRQ(PETSC_ERR_SUP,"Not written for this matrix type");
  }
  ierr = (*mat->ops->duplicate)(mat,op,M);CHKERRQ(ierr);
  if (mat->mapping) {
    ierr = MatSetLocalToGlobalMapping(*M,mat->mapping);CHKERRQ(ierr);
  }
  if (mat->bmapping) {
    ierr = MatSetLocalToGlobalMappingBlock(*M,mat->mapping);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_Convert,mat,0,0,0);CHKERRQ(ierr);
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

   Notes:
   For the SeqAIJ matrix format, this routine may also be called
   on a LU factored matrix; in that case it routines the reciprocal of 
   the diagonal entries in U. It returns the entries permuted by the 
   row and column permutation used during the symbolic factorization.

   Level: intermediate

   Concepts: matrices^accessing diagonals

.seealso: MatGetRow(), MatGetSubmatrices(), MatGetSubmatrix(), MatGetRowMax()
@*/
int MatGetDiagonal(Mat mat,Vec v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  /* PetscCheckSameComm(mat,v); Could be MPI vector but Seq matrix cause of two submatrix storage */
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->getdiagonal) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = (*mat->ops->getdiagonal)(mat,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMax"
/*@ 
   MatGetRowMax - Gets the maximum value (in absolute value) of each
        row of the matrix

   Collective on Mat and Vec

   Input Parameters:
.  mat - the matrix

   Output Parameter:
.  v - the vector for storing the maximums

   Level: intermediate

   Concepts: matrices^getting row maximums

.seealso: MatGetDiagonal(), MatGetSubmatrices(), MatGetSubmatrix()
@*/
int MatGetRowMax(Mat mat,Vec v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  /* PetscCheckSameComm(mat,v); Could be MPI vector but Seq matrix cause of two submatrix storage */
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->getrowmax) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = (*mat->ops->getrowmax)(mat,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose"
/*@C
   MatTranspose - Computes an in-place or out-of-place transpose of a matrix.

   Collective on Mat

   Input Parameter:
.  mat - the matrix to transpose

   Output Parameters:
.  B - the transpose (or pass in PETSC_NULL for an in-place transpose)

   Level: intermediate

   Concepts: matrices^transposing

.seealso: MatMultTranspose(), MatMultTransposeAdd()
@*/
int MatTranspose(Mat mat,Mat *B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->transpose) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name); 

  ierr = PetscLogEventBegin(MAT_Transpose,mat,0,0,0);CHKERRQ(ierr); 
  ierr = (*mat->ops->transpose)(mat,B);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Transpose,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "MatPermute"
/*@C
   MatPermute - Creates a new matrix with rows and columns permuted from the 
   original.

   Collective on Mat

   Input Parameters:
+  mat - the matrix to permute
.  row - row permutation, each processor supplies only the permutation for its rows
-  col - column permutation, each processor needs the entire column permutation, that is
         this is the same size as the total number of columns in the matrix

   Output Parameters:
.  B - the permuted matrix

   Level: advanced

   Concepts: matrices^permuting

.seealso: MatGetOrdering()
@*/
int MatPermute(Mat mat,IS row,IS col,Mat *B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(row,IS_COOKIE);
  PetscValidHeaderSpecific(col,IS_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->permute) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name); 
  ierr = (*mat->ops->permute)(mat,row,col,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPermuteSparsify"
/*@C
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
int MatPermuteSparsify(Mat A, int band, PetscReal frac, PetscReal tol, IS rowp, IS colp, Mat *B)
{
  IS           irowp, icolp;
  int         *rows, *cols;
  int          M, N, locRowStart, locRowEnd;
  int          nz, newNz;
  int         *cwork, *cnew;
  PetscScalar *vwork, *vnew;
  int          bw, size;
  int          row, locRow, newRow, col, newCol;
  int          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,    MAT_COOKIE);
  PetscValidHeaderSpecific(rowp, IS_COOKIE);
  PetscValidHeaderSpecific(colp, IS_COOKIE);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled matrix");
  if (A->factor)     SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Not for factored matrix");
  if (!A->ops->permutesparsify) {
    ierr = MatGetSize(A, &M, &N);                                                                         CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A, &locRowStart, &locRowEnd);                                             CHKERRQ(ierr);
    ierr = ISGetSize(rowp, &size);                                                                        CHKERRQ(ierr);
    if (size != M) SETERRQ2(PETSC_ERR_ARG_WRONG, "Wrong size %d for row permutation, should be %d", size, M);
    ierr = ISGetSize(colp, &size);                                                                        CHKERRQ(ierr);
    if (size != N) SETERRQ2(PETSC_ERR_ARG_WRONG, "Wrong size %d for column permutation, should be %d", size, N);
    ierr = ISInvertPermutation(rowp, 0, &irowp);                                                          CHKERRQ(ierr);
    ierr = ISGetIndices(irowp, &rows);                                                                    CHKERRQ(ierr);
    ierr = ISInvertPermutation(colp, 0, &icolp);                                                          CHKERRQ(ierr);
    ierr = ISGetIndices(icolp, &cols);                                                                    CHKERRQ(ierr);
    ierr = PetscMalloc(N * sizeof(int),         &cnew);                                                   CHKERRQ(ierr);
    ierr = PetscMalloc(N * sizeof(PetscScalar), &vnew);                                                   CHKERRQ(ierr);

    /* Setup bandwidth to include */
    if (band == PETSC_DECIDE) {
      if (frac <= 0.0)
        bw = (int) (M * 0.05);
      else
        bw = (int) (M * frac);
    } else {
      if (band <= 0) SETERRQ(PETSC_ERR_ARG_WRONG, "Bandwidth must be a positive integer");
      bw = band;
    }

    /* Put values into new matrix */
    ierr = MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, B);                                                    CHKERRQ(ierr);
    for(row = locRowStart, locRow = 0; row < locRowEnd; row++, locRow++) {
      ierr = MatGetRow(A, row, &nz, &cwork, &vwork);                                                      CHKERRQ(ierr);
      newRow   = rows[locRow]+locRowStart;
      for(col = 0, newNz = 0; col < nz; col++) {
        newCol = cols[cwork[col]];
        if ((newCol >= newRow - bw) && (newCol < newRow + bw) && (PetscAbsScalar(vwork[col]) >= tol)) {
          cnew[newNz] = newCol;
          vnew[newNz] = vwork[col];
          newNz++;
        }
      }
      ierr = MatSetValues(*B, 1, &newRow, newNz, cnew, vnew, INSERT_VALUES);                              CHKERRQ(ierr);
      ierr = MatRestoreRow(A, row, &nz, &cwork, &vwork);                                                  CHKERRQ(ierr);
    }
    ierr = PetscFree(cnew);                                                                               CHKERRQ(ierr);
    ierr = PetscFree(vnew);                                                                               CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY);                                                      CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY);                                                        CHKERRQ(ierr);
    ierr = ISRestoreIndices(irowp, &rows);                                                                CHKERRQ(ierr);
    ierr = ISRestoreIndices(icolp, &cols);                                                                CHKERRQ(ierr);
    ierr = ISDestroy(irowp);                                                                              CHKERRQ(ierr);
    ierr = ISDestroy(icolp);                                                                              CHKERRQ(ierr);
  } else {
    ierr = (*A->ops->permutesparsify)(A, band, frac, tol, rowp, colp, B);                                 CHKERRQ(ierr);
  }
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
int MatEqual(Mat A,Mat B,PetscTruth *flg)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE); 
  PetscValidHeaderSpecific(B,MAT_COOKIE);
  PetscValidType(A);
  MatPreallocated(A);
  PetscValidType(B);
  MatPreallocated(B);
  PetscValidIntPointer(flg);
  PetscCheckSameComm(A,B);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->M != B->M || A->N != B->N) SETERRQ4(PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim %d %d %d %d",A->M,B->M,A->N,B->N);
  if (!A->ops->equal) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",A->type_name);
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
   L = a diagonal matrix, R = a diagonal matrix

   Level: intermediate

   Concepts: matrices^diagonal scaling
   Concepts: diagonal scaling of matrices

.seealso: MatScale()
@*/
int MatDiagonalScale(Mat mat,Vec l,Vec r)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (!mat->ops->diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  if (l) {PetscValidHeaderSpecific(l,VEC_COOKIE);PetscCheckSameComm(mat,l);}
  if (r) {PetscValidHeaderSpecific(r,VEC_COOKIE);PetscCheckSameComm(mat,r);}
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscLogEventBegin(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->diagonalscale)(mat,l,r);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
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
int MatScale(PetscScalar *a,Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidScalarPointer(a);
  if (!mat->ops->scale) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscLogEventBegin(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->scale)(a,mat);CHKERRQ(ierr);
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
int MatNorm(Mat mat,NormType type,PetscReal *nrm)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidScalarPointer(nrm);

  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->norm) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->norm)(mat,type,nrm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
     This variable is used to prevent counting of MatAssemblyBegin() that
   are called from within a MatAssemblyEnd().
*/
static int MatAssemblyEnd_InUse = 0;
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
int MatAssemblyBegin(Mat mat,MatAssemblyType type)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
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
#define __FUNCT__ "MatAssembed"
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
int MatAssembled(Mat mat,PetscTruth *assembled)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  *assembled = mat->assembled;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_Private"
/*
    Processes command line options to determine if/how a matrix
  is to be viewed. Called by MatAssemblyEnd() and MatLoad().
*/
int MatView_Private(Mat mat)
{
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(mat->prefix,"-mat_view_info",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(mat->comm),PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(mat->prefix,"-mat_view_info_detailed",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(mat->comm),PETSC_VIEWER_ASCII_INFO_LONG);CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(mat->prefix,"-mat_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(mat->prefix,"-mat_view_matlab",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(mat->comm),PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(mat->comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(mat->prefix,"-mat_view_draw",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsHasName(mat->prefix,"-mat_view_contour",&flg);CHKERRQ(ierr);
    if (flg) {
      PetscViewerPushFormat(PETSC_VIEWER_DRAW_(mat->comm),PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);
    }
    ierr = MatView(mat,PETSC_VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    if (flg) {
      PetscViewerPopFormat(PETSC_VIEWER_DRAW_(mat->comm));CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsHasName(mat->prefix,"-mat_view_socket",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatView(mat,PETSC_VIEWER_SOCKET_(mat->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_SOCKET_(mat->comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(mat->prefix,"-mat_view_binary",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatView(mat,PETSC_VIEWER_BINARY_(mat->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_BINARY_(mat->comm));CHKERRQ(ierr);
  }
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
-  -draw_pause <sec> - Sets number of seconds to pause after display

   Notes: 
   MatSetValues() generally caches the values.  The matrix is ready to
   use only after MatAssemblyBegin() and MatAssemblyEnd() have been called.
   Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
   in MatSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
   using the matrix.

   Level: beginner

.seealso: MatAssemblyBegin(), MatSetValues(), PetscDrawOpenX(), MatView(), MatAssembled()
@*/
int MatAssemblyEnd(Mat mat,MatAssemblyType type)
{
  int        ierr;
  static int inassm = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);

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

  if (inassm == 1 && type != MAT_FLUSH_ASSEMBLY) {
    ierr = MatView_Private(mat);CHKERRQ(ierr);
  }
  inassm--;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatCompress"
/*@
   MatCompress - Tries to store the matrix in as little space as 
   possible.  May fail if memory is already fully used, since it
   tries to allocate new space.

   Collective on Mat

   Input Parameters:
.  mat - the matrix 

   Level: advanced

@*/
int MatCompress(Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (mat->ops->compress) {ierr = (*mat->ops->compress)(mat);CHKERRQ(ierr);}
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
-  option - the option, one of those listed below (and possibly others),
             e.g., MAT_ROWS_SORTED, MAT_NEW_NONZERO_LOCATION_ERR

   Options Describing Matrix Structure:
+    MAT_SYMMETRIC - symmetric in terms of both structure and value
-    MAT_STRUCTURALLY_SYMMETRIC - symmetric nonzero structure

   Options For Use with MatSetValues():
   Insert a logically dense subblock, which can be
+    MAT_ROW_ORIENTED - row-oriented (default)
.    MAT_COLUMN_ORIENTED - column-oriented
.    MAT_ROWS_SORTED - sorted by row
.    MAT_ROWS_UNSORTED - not sorted by row (default)
.    MAT_COLUMNS_SORTED - sorted by column
-    MAT_COLUMNS_UNSORTED - not sorted by column (default)

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
   Set after the first MatAssemblyEnd()

   MAT_NEW_NONZERO_LOCATION_ERR indicates that any add or insertion 
   that would generate a new entry in the nonzero structure instead produces 
   an error. (Currently supported for AIJ and BAIJ formats only.)
   This is a useful flag when using SAME_NONZERO_PATTERN in calling
   SLESSetOperators() to ensure that the nonzero pattern truely does 
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
   to improve the searching of indices. MAT_NO_NEW_NONZERO_LOCATIONS flag 
   should be used with MAT_USE_HASH_TABLE flag. This option is currently
   supported by MATMPIBAIJ format only.

   MAT_KEEP_ZEROED_ROWS indicates when MatZeroRows() is called the zeroed entries
   are kept in the nonzero structure

   MAT_IGNORE_ZERO_ENTRIES - when using ADD_VALUES for AIJ matrices this will stop
   zero values from creating a zero location in the matrix

   MAT_USE_INODES - indicates using inode version of the code - works with AIJ and 
   ROWBS matrix types

   MAT_DO_NOT_USE_INODES - indicates not using inode version of the code - works
   with AIJ and ROWBS matrix types

   Level: intermediate

   Concepts: matrices^setting options

@*/
int MatSetOption(Mat mat,MatOption op)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  switch (op) {
  case MAT_SYMMETRIC:
    mat->symmetric              = PETSC_TRUE;
    mat->structurally_symmetric = PETSC_TRUE;
    break;
  case MAT_STRUCTURALLY_SYMMETRIC:
    mat->structurally_symmetric = PETSC_TRUE;
    break;
  default:
    if (mat->ops->setoption) {ierr = (*mat->ops->setoption)(mat,op);CHKERRQ(ierr);}
    break;
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
int MatZeroEntries(Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->zeroentries) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = PetscLogEventBegin(MAT_ZeroEntries,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->zeroentries)(mat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ZeroEntries,mat,0,0,0);CHKERRQ(ierr);
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
.  is - index set of rows to remove
-  diag - pointer to value put in all diagonals of eliminated rows.
          Note that diag is not a pointer to an array, but merely a
          pointer to a single value.

   Notes:
   For the AIJ and BAIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_ZEROED_ROWS) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing a null pointer (PETSC_NULL 
   in C or PETSC_NULL_SCALAR in Fortran) as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

  
   Level: intermediate

   Concepts: matrices^zeroing rows

.seealso: MatZeroEntries(), MatZeroRowsLocal(), MatSetOption()
@*/
int MatZeroRows(Mat mat,IS is,PetscScalar *diag)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(is,IS_COOKIE);
  if (diag) PetscValidScalarPointer(diag);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->zerorows) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);

  ierr = (*mat->ops->zerorows)(mat,is,diag);CHKERRQ(ierr);
  ierr = MatView_Private(mat);CHKERRQ(ierr);
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
.  is - index set of rows to remove
-  diag - pointer to value put in all diagonals of eliminated rows.
          Note that diag is not a pointer to an array, but merely a
          pointer to a single value.

   Notes:
   Before calling MatZeroRowsLocal(), the user must first set the
   local-to-global mapping by calling MatSetLocalToGlobalMapping().

   For the AIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_ZEROED_ROWS) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing a null pointer (PETSC_NULL
   in C or PETSC_NULL_SCALAR in Fortran) as the final argument).

   Level: intermediate

   Concepts: matrices^zeroing

.seealso: MatZeroEntries(), MatZeroRows(), MatSetLocalToGlobalMapping
@*/
int MatZeroRowsLocal(Mat mat,IS is,PetscScalar *diag)
{
  int ierr;
  IS  newis;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(is,IS_COOKIE);
  if (diag) PetscValidScalarPointer(diag);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  if (mat->ops->zerorowslocal) {
    ierr = (*mat->ops->zerorowslocal)(mat,is,diag);CHKERRQ(ierr);
  } else {
    if (!mat->mapping) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Need to provide local to global mapping to matrix first");
    ierr = ISLocalToGlobalMappingApplyIS(mat->mapping,is,&newis);CHKERRQ(ierr);
    ierr = (*mat->ops->zerorows)(mat,newis,diag);CHKERRQ(ierr);
    ierr = ISDestroy(newis);CHKERRQ(ierr);
  }
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

   Level: beginner

   Concepts: matrices^size

.seealso: MatGetLocalSize()
@*/
int MatGetSize(Mat mat,int *m,int* n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (m) *m = mat->M;
  if (n) *n = mat->N;
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

   Level: beginner

   Concepts: matrices^local size

.seealso: MatGetSize()
@*/
int MatGetLocalSize(Mat mat,int *m,int* n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (m) *m = mat->m;
  if (n) *n = mat->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetOwnershipRange"
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

   Concepts: matrices^row ownership
@*/
int MatGetOwnershipRange(Mat mat,int *m,int* n)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (m) PetscValidIntPointer(m);
  if (n) PetscValidIntPointer(n);
  ierr = PetscMapGetLocalRange(mat->rmap,m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic"
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

  Concepts: matrices^symbolic LU factorization
  Concepts: matrices^factorization
  Concepts: LU^symbolic factorization

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
          MatGetOrdering(), MatILUInfo

@*/
int MatILUFactorSymbolic(Mat mat,IS row,IS col,MatILUInfo *info,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(fact);
  PetscValidHeaderSpecific(row,IS_COOKIE);
  PetscValidHeaderSpecific(col,IS_COOKIE);
  if (info && info->levels < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Levels of fill negative %d",(int)info->levels);
  if (info && info->fill < 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Expected fill less than 1.0 %g",info->fill);
  if (!mat->ops->ilufactorsymbolic) SETERRQ1(PETSC_ERR_SUP,"Matrix type %s  symbolic ILU",mat->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscLogEventBegin(MAT_ILUFactorSymbolic,mat,row,col,0);CHKERRQ(ierr);
  ierr = (*mat->ops->ilufactorsymbolic)(mat,row,col,info,fact);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_ILUFactorSymbolic,mat,row,col,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatICCFactorSymbolic"
/*@  
   MatICCFactorSymbolic - Performs symbolic incomplete
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

  Concepts: matrices^symbolic incomplete Cholesky factorization
  Concepts: matrices^factorization
  Concepts: Cholsky^symbolic factorization

.seealso: MatCholeskyFactorNumeric(), MatCholeskyFactor()
@*/
int MatICCFactorSymbolic(Mat mat,IS perm,PetscReal f,int fill,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(fact);
  PetscValidHeaderSpecific(perm,IS_COOKIE);
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (fill < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Fill negative %d",fill);
  if (f < 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Expected fill less than 1.0 %g",f);
  if (!mat->ops->iccfactorsymbolic) SETERRQ1(PETSC_ERR_SUP,"Matrix type %s  symbolic ICC",mat->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");

  ierr = PetscLogEventBegin(MAT_ICCFactorSymbolic,mat,perm,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->iccfactorsymbolic)(mat,perm,f,fill,fact);CHKERRQ(ierr);
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

   Concepts: matrices^access array

.seealso: MatRestoreArray(), MatGetArrayF90()
@*/
int MatGetArray(Mat mat,PetscScalar **v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(v);
  if (!mat->ops->getarray) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->getarray)(mat,v);CHKERRQ(ierr);
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

.seealso: MatGetArray(), MatRestoreArrayF90()
@*/
int MatRestoreArray(Mat mat,PetscScalar **v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidPointer(v);
  if (!mat->ops->restorearray) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->restorearray)(mat,v);CHKERRQ(ierr);
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
   MatGetSubMatrices() can extract only sequential submatrices
   (from both sequential and parallel matrices). Use MatGetSubMatrix()
   to extract a parallel submatrix.

   When extracting submatrices from a parallel matrix, each processor can
   form a different submatrix by setting the rows and columns of its
   individual index sets according to the local submatrix desired.

   When finished using the submatrices, the user should destroy
   them with MatDestroyMatrices().

   MAT_REUSE_MATRIX can only be used when the nonzero structure of the 
   original matrix has not changed from that last call to MatGetSubMatrices().

   This routine creates the matrices submat; you should NOT create them before
   calling it.

   Fortran Note:
   The Fortran interface is slightly different from that given below; it 
   requires one to pass in  as submat a Mat (integer) array of size at least m.

   Level: advanced

   Concepts: matrices^accessing submatrices
   Concepts: submatrices

.seealso: MatDestroyMatrices(), MatGetSubMatrix(), MatGetRow(), MatGetDiagonal()
@*/
int MatGetSubMatrices(Mat mat,int n,IS *irow,IS *icol,MatReuse scall,Mat **submat)
{
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (!mat->ops->getsubmatrices) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");

  ierr = PetscLogEventBegin(MAT_GetSubMatrices,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->getsubmatrices)(mat,n,irow,icol,scall,submat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_GetSubMatrices,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroyMatrices"
/*@C
   MatDestroyMatrices - Destroys a set of matrices obtained with MatGetSubMatrices().

   Collective on Mat

   Input Parameters:
+  n - the number of local matrices
-  mat - the matrices

   Level: advanced

    Notes: Frees not only the matrices, but also the array that contains the matrices

.seealso: MatGetSubMatrices()
@*/
int MatDestroyMatrices(int n,Mat **mat)
{
  int ierr,i;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy negative number of matrices %d",n);
  PetscValidPointer(mat);
  for (i=0; i<n; i++) {
    ierr = MatDestroy((*mat)[i]);CHKERRQ(ierr);
  }
  /* memory is allocated even if n = 0 */
  ierr = PetscFree(*mat);CHKERRQ(ierr);
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
.  is  - the array of pointers to index sets
-  ov  - the additional overlap requested

   Level: developer

   Concepts: overlap
   Concepts: ASM^computing overlap

.seealso: MatGetSubMatrices()
@*/
int MatIncreaseOverlap(Mat mat,int n,IS *is,int ov)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor)     SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  if (!ov) PetscFunctionReturn(0);
  if (!mat->ops->increaseoverlap) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = PetscLogEventBegin(MAT_IncreaseOverlap,mat,0,0,0);CHKERRQ(ierr);
  ierr = (*mat->ops->increaseoverlap)(mat,n,is,ov);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_IncreaseOverlap,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPrintHelp"
/*@
   MatPrintHelp - Prints all the options for the matrix.

   Collective on Mat

   Input Parameter:
.  mat - the matrix 

   Options Database Keys:
+  -help - Prints matrix options
-  -h - Prints matrix options

   Level: developer

.seealso: MatCreate(), MatCreateXXX()
@*/
int MatPrintHelp(Mat mat)
{
  static PetscTruth called = PETSC_FALSE;
  int               ierr;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);

  comm = mat->comm;
  if (!called) {
    ierr = (*PetscHelpPrintf)(comm,"General matrix options:\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -mat_view_info: view basic matrix info during MatAssemblyEnd()\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -mat_view_info_detailed: view detailed matrix info during MatAssemblyEnd()\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -mat_view_draw: draw nonzero matrix structure during MatAssemblyEnd()\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"      -draw_pause <sec>: set seconds of display pause\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"      -display <name>: set alternate display\n");CHKERRQ(ierr);
    called = PETSC_TRUE;
  }
  if (mat->ops->printhelp) {
    ierr = (*mat->ops->printhelp)(mat);CHKERRQ(ierr);
  }
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
   Block diagonal formats are MATSEQBDIAG, MATMPIBDIAG.
   Block row formats are MATSEQBAIJ, MATMPIBAIJ

   Level: intermediate

   Concepts: matrices^block size

.seealso: MatCreateSeqBAIJ(), MatCreateMPIBAIJ(), MatCreateSeqBDiag(), MatCreateMPIBDiag()
@*/
int MatGetBlockSize(Mat mat,int *bs)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidIntPointer(bs);
  if (!mat->ops->getblocksize) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->getblocksize)(mat,bs);CHKERRQ(ierr);
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
  PetscValidType(mat);
  MatPreallocated(mat);
  if (ia) PetscValidIntPointer(ia);
  if (ja) PetscValidIntPointer(ja);
  PetscValidIntPointer(done);
  if (!mat->ops->getrowij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->getrowij)(mat,shift,symmetric,n,ia,ja,done);CHKERRQ(ierr);
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
  PetscValidType(mat);
  MatPreallocated(mat);
  if (ia) PetscValidIntPointer(ia);
  if (ja) PetscValidIntPointer(ja);
  PetscValidIntPointer(done);

  if (!mat->ops->getcolumnij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->getcolumnij)(mat,shift,symmetric,n,ia,ja,done);CHKERRQ(ierr);
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
  PetscValidType(mat);
  MatPreallocated(mat);
  if (ia) PetscValidIntPointer(ia);
  if (ja) PetscValidIntPointer(ja);
  PetscValidIntPointer(done);

  if (!mat->ops->restorerowij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->restorerowij)(mat,shift,symmetric,n,ia,ja,done);CHKERRQ(ierr);
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
  PetscValidType(mat);
  MatPreallocated(mat);
  if (ia) PetscValidIntPointer(ia);
  if (ja) PetscValidIntPointer(ja);
  PetscValidIntPointer(done);

  if (!mat->ops->restorecolumnij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    ierr  = (*mat->ops->restorecolumnij)(mat,shift,symmetric,n,ia,ja,done);CHKERRQ(ierr);
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
.   n   - number of colors
-   colorarray - array indicating color for each column

    Output Parameters:
.   iscoloring - coloring generated using colorarray information

    Level: developer

.seealso: MatGetRowIJ(), MatGetColumnIJ()

@*/
int MatColoringPatch(Mat mat,int n,int ncolors,int *colorarray,ISColoring *iscoloring)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidIntPointer(colorarray);

  if (!mat->ops->coloringpatch){
    ierr = ISColoringCreate(mat->comm,n,colorarray,iscoloring);CHKERRQ(ierr);
  } else {
    ierr = (*mat->ops->coloringpatch)(mat,n,ncolors,colorarray,iscoloring);CHKERRQ(ierr);
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

   Concepts: matrices^unfactored

@*/
int MatSetUnfactored(Mat mat)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);  
  PetscValidType(mat);
  MatPreallocated(mat);
  mat->factor = 0;
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
.   isrow - rows this processor should obtain
.   iscol - columns for all processors you wish to keep
.   csize - number of columns "local" to this processor (does nothing for sequential 
            matrices). This should match the result from VecGetLocalSize(x,...) if you 
            plan to use the matrix in a A*x; alternatively, you can use PETSC_DECIDE
-   cll - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

    Output Parameter:
.   newmat - the new submatrix, of the same type as the old

    Level: advanced

    Notes: the iscol argument MUST be the same on each processor. You might be 
    able to create the iscol argument with ISAllGather().

      The first time this is called you should use a cll of MAT_INITIAL_MATRIX,
   the MatGetSubMatrix() routine will create the newmat for you. Any additional calls
   to this routine with a mat of the same nonzero structure will reuse the matrix
   generated the first time.

    Concepts: matrices^submatrices

.seealso: MatGetSubMatrices(), ISAllGather()
@*/
int MatGetSubMatrix(Mat mat,IS isrow,IS iscol,int csize,MatReuse cll,Mat *newmat)
{
  int     ierr, size;
  Mat     *local;

  PetscFunctionBegin;
  PetscValidType(mat);
  MatPreallocated(mat);
  ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr);

  /* if original matrix is on just one processor then use submatrix generated */
  if (!mat->ops->getsubmatrix && size == 1 && cll == MAT_REUSE_MATRIX) {
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_REUSE_MATRIX,&newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (!mat->ops->getsubmatrix && size == 1) {
    ierr    = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&local);CHKERRQ(ierr);
    *newmat = *local;
    ierr    = PetscFree(local);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!mat->ops->getsubmatrix) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->getsubmatrix)(mat,isrow,iscol,csize,cll,newmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetPetscMaps"
/*@C
   MatGetPetscMaps - Returns the maps associated with the matrix.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  rmap - the row (right) map
-  cmap - the column (left) map  

   Level: developer

   Concepts: maps^getting from matrix

@*/
int MatGetPetscMaps(Mat mat,PetscMap *rmap,PetscMap *cmap)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  ierr = (*mat->ops->getmaps)(mat,rmap,cmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Version that works for all PETSc matrices
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetPetscMaps_Petsc"
int MatGetPetscMaps_Petsc(Mat mat,PetscMap *rmap,PetscMap *cmap)
{
  PetscFunctionBegin;
  if (rmap) *rmap = mat->rmap;
  if (cmap) *cmap = mat->cmap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetStashInitialSize"
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

   Concepts: stash^setting matrix size
   Concepts: matrices^stash

@*/
int MatSetStashInitialSize(Mat mat,int size, int bsize)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
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
int MatInterpolateAdd(Mat A,Vec x,Vec y,Vec w)
{
  int M,N,ierr;

  PetscFunctionBegin;
  PetscValidType(A);
  MatPreallocated(A);
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
int MatInterpolate(Mat A,Vec x,Vec y)
{
  int M,N,ierr;

  PetscFunctionBegin;
  PetscValidType(A);
  MatPreallocated(A);
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
int MatRestrict(Mat A,Vec x,Vec y)
{
  int M,N,ierr;

  PetscFunctionBegin;
  PetscValidType(A);
  MatPreallocated(A);
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
/*@C
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
int MatNullSpaceAttach(Mat mat,MatNullSpace nullsp)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  PetscValidHeaderSpecific(nullsp,MAT_NULLSPACE_COOKIE);

  if (mat->nullsp) {
    ierr = MatNullSpaceDestroy(mat->nullsp);CHKERRQ(ierr);
  }
  mat->nullsp = nullsp;
  ierr = PetscObjectReference((PetscObject)nullsp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatICCFactor"
/*@  
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

   Most users should employ the simplified SLES interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., SLESCreate().

   Level: developer

   Concepts: matrices^incomplete Cholesky factorization
   Concepts: Cholesky factorization

.seealso: MatICCFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatICCFactor(Mat mat,IS row,PetscReal fill,int level)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);
  MatPreallocated(mat);
  if (mat->M != mat->N) SETERRQ(PETSC_ERR_ARG_WRONG,"matrix must be square");
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!mat->ops->iccfactor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->iccfactor)(mat,row,fill,level);CHKERRQ(ierr);
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
int MatSetValuesAdic(Mat mat,void *v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);

  if (!mat->assembled) {
    SETERRQ(1,"Matrix must be already assembled");
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (!mat->ops->setvaluesadic) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->setvaluesadic)(mat,v);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  ierr = MatView_Private(mat);CHKERRQ(ierr);
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
int MatSetColoring(Mat mat,ISColoring coloring)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);

  if (!mat->assembled) {
    SETERRQ(1,"Matrix must be already assembled");
  }
  if (!mat->ops->setcoloring) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
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
int MatSetValuesAdifor(Mat mat,int nl,void *v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidType(mat);

  if (!mat->assembled) {
    SETERRQ(1,"Matrix must be already assembled");
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (!mat->ops->setvaluesadifor) SETERRQ1(PETSC_ERR_SUP,"Mat type %s",mat->type_name);
  ierr = (*mat->ops->setvaluesadifor)(mat,nl,v);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN int MatMPIAIJDiagonalScaleLocal(Mat,Vec);
EXTERN int MatMPIBAIJDiagonalScaleLocal(Mat,Vec);

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
int MatDiagonalScaleLocal(Mat mat,Vec diag)
{
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(diag,VEC_COOKIE);
  PetscValidType(mat);

  if (!mat->assembled) {
    SETERRQ(1,"Matrix must be already assembled");
  }
  ierr = PetscLogEventBegin(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIAIJ,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatMPIAIJDiagonalScaleLocal(mat,diag);CHKERRQ(ierr);
  } else {
    ierr = PetscTypeCompare((PetscObject)mat,MATMPIBAIJ,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatMPIBAIJDiagonalScaleLocal(mat,diag);CHKERRQ(ierr);
    } else {
      int size;
      ierr = MPI_Comm_size(mat->comm,&size);CHKERRQ(ierr);
      if (size == 1) {
        int n,m;
        ierr = VecGetSize(diag,&n);CHKERRQ(ierr);
        ierr = MatGetSize(mat,0,&m);CHKERRQ(ierr);
        if (m == n) {
          ierr = MatDiagonalScale(mat,0,diag);CHKERRQ(ierr);
        } else {
          SETERRQ(1,"Only supprted for sequential matrices when no ghost points/periodic conditions");
        }
      } else {
        SETERRQ(1,"Only supported for MPIAIJ and MPIBAIJ parallel matrices");
      }
    }
  }
  ierr = PetscLogEventEnd(MAT_Scale,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
