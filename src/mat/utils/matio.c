/*$Id: matio.c,v 1.79 2001/08/06 21:16:10 bsmith Exp $*/

/* 
   This file contains simple binary read/write routines for matrices.
 */

#include "src/mat/matimpl.h"             /*I  "petscmat.h"  I*/
#include "petscsys.h"
PetscTruth MatLoadRegisterAllCalled = PETSC_FALSE;
PetscFList      MatLoadList              = 0;

#undef __FUNCT__  
#define __FUNCT__ "MatLoadRegister"
/*@C
    MatLoadRegister - Allows one to register a routine that reads matrices
        from a binary file for a particular matrix type.

  Not Collective

  Input Parameters:
+   type - the type of matrix (defined in include/petscmat.h), for example, MATSEQAIJ.
-   loader - the function that reads the matrix from the binary file.

  Level: developer

.seealso: MatLoadRegisterAll(), MatLoad()

@*/
int MatLoadRegister(char *sname,char *path,char *name,int (*function)(PetscViewer,MatType,Mat*))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatLoadList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoadPrintHelp_Private"
static int MatLoadPrintHelp_Private(Mat A)
{
  static PetscTruth called = PETSC_FALSE; 
  MPI_Comm          comm = A->comm;
  int               ierr;
  
  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = PETSC_TRUE;
  ierr = (*PetscHelpPrintf)(comm," Options for MatLoad:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_type <type>\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -matload_type <type>\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -matload_block_size <block_size> :Used for MATBAIJ, MATBDIAG\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -matload_bdiag_diags <s1,s2,s3,...> : Used for MATBDIAG\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad"
/*@C
   MatLoad - Loads a matrix that has been stored in binary format
   with MatView().  The matrix format is determined from the options database.
   Generates a parallel MPI matrix if the communicator has more than one
   processor.  The default matrix type is AIJ.

   Collective on PetscViewer

   Input Parameters:
+  viewer - binary file viewer, created with PetscViewerBinaryOpen()
-  outtype - type of matrix desired, for example MATSEQAIJ,
             MATMPIROWBS, etc.  See types in petsc/include/petscmat.h.

   Output Parameters:
.  newmat - new matrix

   Basic Options Database Keys:
+    -matload_type seqaij   - AIJ type
.    -matload_type mpiaij   - parallel AIJ type
.    -matload_type seqbaij  - block AIJ type
.    -matload_type mpibaij  - parallel block AIJ type
.    -matload_type seqsbaij - block symmetric AIJ type
.    -matload_type mpisbaij - parallel block symmetric AIJ type
.    -matload_type seqbdiag - block diagonal type
.    -matload_type mpibdiag - parallel block diagonal type
.    -matload_type mpirowbs - parallel rowbs type
.    -matload_type seqdense - dense type
.    -matload_type mpidense - parallel dense type
-    -matload_symmetric - matrix in file is symmetric

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
   that was passed to the PetscViewerBinaryOpen(). The options in the info
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

$    int    MAT_FILE_COOKIE
$    int    number of rows
$    int    number of columns
$    int    total number of nonzeros
$    int    *number nonzeros in each row
$    int    *column indices of all nonzeros (starting index is zero)
$    PetscScalar *values of all nonzeros

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

.seealso: PetscViewerBinaryOpen(), MatView(), VecLoad(), MatLoadRegister(),
          MatLoadRegisterAll()

 @*/  
int MatLoad(PetscViewer viewer,MatType outtype,Mat *newmat)
{
  int         ierr;
  PetscTruth  isbinary,flg;
  MPI_Comm    comm;
  int         (*r)(PetscViewer,MatType,Mat*);
  char        mtype[256],*prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);

#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  *newmat = 0;

  if (!MatLoadRegisterAllCalled) {
    ierr = MatLoadRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscObjectGetOptionsPrefix((PetscObject)viewer,&prefix);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");
  }

  ierr = PetscOptionsGetString(prefix,"-mat_type",mtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    outtype = mtype;
  }
  ierr = PetscOptionsGetString(prefix,"-matload_type",mtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    outtype = mtype;
  }
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  if (!outtype) outtype = MATMPIAIJ;
  ierr =  PetscFListFind(comm,MatLoadList,outtype,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unknown Mat type given: %s",outtype);

  ierr = PetscLogEventBegin(MAT_Load,viewer,0,0,0);CHKERRQ(ierr);
  ierr = (*r)(viewer,outtype,newmat);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Load,viewer,0,0,0);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(prefix,"-matload_symmetric",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatSetOption(*newmat,MAT_SYMMETRIC);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatLoadPrintHelp_Private(*newmat);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

