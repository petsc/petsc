#ifndef lint
static char vcid[] = "$Id: matio.c,v 1.19 1995/10/25 23:06:07 curfman Exp curfman $";
#endif

/* 
   This file contains simple binary read/write routines for matrices.
 */

#include "petsc.h"
#include "vec/vecimpl.h"
#include "../matimpl.h"
#include "sysio.h"
#include "pinclude/pviewer.h"

extern int MatLoad_MPIRowbs(Viewer,MatType,Mat*);
extern int MatLoad_SeqAIJ(Viewer,MatType,Mat*);
extern int MatLoad_SeqRow(Viewer,MatType,Mat*);
extern int MatLoad_MPIAIJorMPIRow(Viewer,MatType,Mat*);
extern int MatLoad_SeqBDiag(Viewer,MatType,Mat*);
extern int MatLoad_MPIBDiag(Viewer,MatType,Mat*);
extern int MatLoad_SeqDense(Viewer,MatType,Mat*);
extern int MatLoad_MPIDense(Viewer,MatType,Mat*);

/*@C
   MatLoad - Loads a matrix that has been stored in binary format
   with MatView().

   Input Parameters:
.  bview - binary file viewer, created with ViewerFileOpenBinary()
.  outtype - type of matrix desired, for example MATSEQAIJ,
   MATMPIROWBS, etc.  See types in petsc/include/mat.h.

   Output Parameters:
.  newmat - new matrix

   Notes:
   In parallel, each processor can load a subset of rows (or the
   entire matrix).  This routine is especially useful when a large
   matrix is stored on disk and only part of it is desired on each
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

.keywords: matrix, load, binary, input

.seealso: ViewerFileOpenBinary(), MatView(), VecLoad() 
 @*/  
int MatLoad(Viewer bview,MatType outtype,Mat *newmat)
{
  PetscObject vobj = (PetscObject) bview;
  int         ierr,set;
  MatType     type;
  *newmat = 0;

  PETSCVALIDHEADERSPECIFIC(vobj,VIEWER_COOKIE);
  if (vobj->type != BINARY_FILE_VIEWER)
   SETERRQ(1,"MatLoad: Invalid viewer; open viewer with ViewerFileOpenBinary()");

  PLogEventBegin(MAT_Load,bview,0,0,0);
  ierr = MatGetFormatFromOptions(vobj->comm,0,&type,&set); CHKERRQ(ierr);
  if (!set) type = outtype;

  if (type == MATSEQAIJ) {
    ierr = MatLoad_SeqAIJ(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIAIJ || type == MATMPIROW) {
    ierr = MatLoad_MPIAIJorMPIRow(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQROW) {
    ierr = MatLoad_SeqRow(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQBDIAG) {
    ierr = MatLoad_SeqBDiag(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIBDIAG) {
    ierr = MatLoad_MPIBDiag(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQDENSE) {
    ierr = MatLoad_SeqDense(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIDENSE) {
    ierr = MatLoad_MPIDense(bview,type,newmat); CHKERRQ(ierr);
  }
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
  else if (type == MATMPIROWBS) {
    ierr = MatLoad_MPIRowbs(bview,type,newmat); CHKERRQ(ierr);
  }
#endif
  else {
    SETERRQ(1,"MatLoad: cannot load with that matrix type yet");
  }

  PLogEventEnd(MAT_Load,bview,0,0,0);
  return 0;
}
