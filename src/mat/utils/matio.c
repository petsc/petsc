/*$Id: matio.c,v 1.66 2000/04/09 04:36:57 bsmith Exp bsmith $*/

/* 
   This file contains simple binary read/write routines for matrices.
 */

#include "petsc.h"
#include "src/mat/matimpl.h"             /*I  "mat.h"  I*/
#include "sys.h"

static int MatLoadersSet = 0,(*MatLoaders[MAX_MATRIX_TYPES])(Viewer,MatType,Mat*) = 
           {0,0,0,0,0,0,0,0,0,0,0,0};

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatLoadRegister"
/*@C
    MatLoadRegister - Allows one to register a routine that reads matrices
        from a binary file for a particular matrix type.

  Not Collective

  Input Parameters:
+   type - the type of matrix (defined in include/mat.h), for example, MATSEQAIJ.
-   loader - the function that reads the matrix from the binary file.

  Level: developer

.seealso: MatLoadRegisterAll(), MatLoad()

@*/
int MatLoadRegister(MatType type,int (*loader)(Viewer,MatType,Mat*))
{
  PetscFunctionBegin;
  MatLoaders[type] = loader;
  MatLoadersSet    = 1;
  PetscFunctionReturn(0);
}  

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatLoadPrintHelp_Private"
static int MatLoadPrintHelp_Private(Mat A)
{
  static PetscTruth called = PETSC_FALSE; 
  MPI_Comm          comm = A->comm;
  int               ierr;
  
  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = PETSC_TRUE;
  ierr = (*PetscHelpPrintf)(comm," Options for MatLoad:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -matload_block_size <block_size> :Used for MATBAIJ, MATBDIAG\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -matload_bdiag_diags <s1,s2,s3,...> : Used for MATBDIAG\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatLoad"
/*@C
   MatLoad - Loads a matrix that has been stored in binary format
   with MatView().  The matrix format is determined from the options database.
   Generates a parallel MPI matrix if the communicator has more than one
   processor.  The default matrix type is AIJ.

   Collective on Viewer

   Input Parameters:
+  viewer - binary file viewer, created with ViewerBinaryOpen()
-  outtype - type of matrix desired, for example MATSEQAIJ,
             MATMPIROWBS, etc.  See types in petsc/include/mat.h.

   Output Parameters:
.  newmat - new matrix

   Basic Options Database Keys:
   The following options will work if you first call MatGetTypeFromOptions()
   and pass the resulting type to MatLoad().
   These options use MatCreateSeqXXX or MatCreateMPIXXX,
   depending on the communicator, comm.
+    -mat_aij      - AIJ type
.    -mat_baij     - block AIJ type
.    -mat_dense    - dense type
.    -mat_bdiag    - block diagonal type
.    -mat_complex  - indicates the matrix has complex entries
-    -mat_double   - indicates the matrix has double entries

   More Options Database Keys:
+    -mat_seqaij   - AIJ type
.    -mat_mpiaij   - parallel AIJ type
.    -mat_seqbaij  - block AIJ type
.    -mat_mpibaij  - parallel block AIJ type
.    -mat_seqbdiag - block diagonal type
.    -mat_mpibdiag - parallel block diagonal type
.    -mat_mpirowbs - parallel rowbs type
.    -mat_seqdense - dense type
-    -mat_mpidense - parallel dense type

   More Options Database Keys:
   Used with block matrix formats (MATSEQBAIJ, MATMPIBDIAG, ...) to specify
   block size
.    -matload_block_size <bs>

   Used to specify block diagonal numbers for MATSEQBDIAG and MATMPIBDIAG formats
.    -matload_bdiag_diags <s1,s2,s3,...>

   Level: beginner

   Notes:
   MatLoad() automatically loads into the options database any options
   given in the file filename.info where filename is the name of the file
   that was passed to the ViewerBinaryOpen(). The options in the info
   file will be ignored if you use the -matload_ignore_info option.

   In parallel, each processor can load a subset of rows (or the
   entire matrix).  This routine is especially useful when a large
   matrix is stored on disk and only part of it existsis desired on each
   processor.  For example, a parallel solver may access only some of
   the rows from each processor.  The algorithm used here reads
   relatively small blocks of data rather than reading the entire
   matrix and then subsetting it.

   Notes for advanced users:
   Most users should not need to know the details of the binary storage
   format, since MatLoad() and MatView() completely hide these details.
   But for anyone who's interested, the standard binary matrix storage
   format is

$    int    MAT_COOKIE
$    int    number of rows
$    int    number of columns
$    int    total number of nonzeros
$    int    *number nonzeros in each row
$    int    *column indices of all nonzeros (starting index is zero)
$    Scalar *values of all nonzeros

   Note for Cray users, the int's stored in the binary file are 32 bit
integers; not 64 as they are represented in the memory, so if you
write your own routines to read/write these binary files from the Cray
you need to adjust the integer sizes that you read in, see
PetscReadBinary() and PetscWriteBinary() to see how this may be
done.

   In addition, PETSc automatically does the byte swapping for
machines that store the bytes reversed, e.g.  DEC alpha, freebsd,
linux, nt and the paragon; thus if you write your own binary
read/write routines you have to swap the bytes; see PetscReadBinary()
and PetscWriteBinary() to see how this may be done.

.keywords: matrix, load, binary, input

.seealso: ViewerBinaryOpen(), MatView(), VecLoad(), MatLoadRegister(),
          MatLoadRegisterAll(), MatGetTypeFromOptions()

 @*/  
int MatLoad(Viewer viewer,MatType outtype,Mat *newmat)
{
  int         ierr;
  PetscTruth  set,isbinary,flg;
  MatType     type;
  MPI_Comm    comm;

  PetscFunctionBegin;
  if (outtype > MAX_MATRIX_TYPES || outtype < 0) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Not a valid matrix type");
  }
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  *newmat  = 0;

  if (!MatLoadersSet) {
    ierr = MatLoadRegisterAll();CHKERRQ(ierr);
  }

  ierr = PetscTypeCompare((PetscObject)viewer,BINARY_VIEWER,&isbinary);CHKERRQ(ierr);
  if (!isbinary) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Invalid viewer; open viewer with ViewerBinaryOpen()");
  }

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MatGetTypeFromOptions(comm,0,&type,&set);CHKERRQ(ierr);
  if (!set) type = outtype;

  PLogEventBegin(MAT_Load,viewer,0,0,0);

  if (!MatLoaders[outtype]) {
    SETERRQ(PETSC_ERR_ARG_WRONG,1,"Invalid matrix type, or matrix load not registered");
  }

  ierr = (*MatLoaders[outtype])(viewer,type,newmat);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatLoadPrintHelp_Private(*newmat);CHKERRQ(ierr); }
  PLogEventEnd(MAT_Load,viewer,0,0,0);
  PetscFunctionReturn(0);
}

