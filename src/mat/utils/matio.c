#ifndef lint
static char vcid[] = "$Id: matio.c,v 1.24 1996/03/19 21:27:41 bsmith Exp curfman $";
#endif

/* 
   This file contains simple binary read/write routines for matrices.
 */

#include "petsc.h"
#include "vec/vecimpl.h"
#include "../matimpl.h"
#include "sys.h"
#include "pinclude/pviewer.h"

extern int MatLoad_MPIRowbs(Viewer,MatType,Mat*);
extern int MatLoad_SeqAIJ(Viewer,MatType,Mat*);
extern int MatLoad_MPIAIJ(Viewer,MatType,Mat*);
extern int MatLoad_SeqBDiag(Viewer,MatType,Mat*);
extern int MatLoad_MPIBDiag(Viewer,MatType,Mat*);
extern int MatLoad_SeqDense(Viewer,MatType,Mat*);
extern int MatLoad_MPIDense(Viewer,MatType,Mat*);
extern int MatLoad_SeqBAIJ(Viewer,MatType,Mat*);



/*@C
   MatLoad - Loads a matrix that has been stored in binary format
   with MatView().

   Input Parameters:
.  viewer - binary file viewer, created with ViewerFileOpenBinary()
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
int MatLoad(Viewer viewer,MatType outtype,Mat *newmat)
{
  int         ierr,set;
  MatType     type;
  ViewerType  vtype;
  MPI_Comm    comm;
  *newmat = 0;

  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype != BINARY_FILE_VIEWER)
   SETERRQ(1,"MatLoad: Invalid viewer; open viewer with ViewerFileOpenBinary()");

  PetscObjectGetComm((PetscObject)viewer,&comm);
  ierr = MatGetTypeFromOptions(comm,0,&type,&set); CHKERRQ(ierr);
  if (!set) type = outtype;
  PLogEventBegin(MAT_Load,viewer,0,0,0);

  if (type == MATSEQAIJ) {
    ierr = MatLoad_SeqAIJ(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIAIJ) {
    ierr = MatLoad_MPIAIJ(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQBDIAG) {
    ierr = MatLoad_SeqBDiag(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIBDIAG) {
    ierr = MatLoad_MPIBDiag(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQDENSE) {
    ierr = MatLoad_SeqDense(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIDENSE) {
    ierr = MatLoad_MPIDense(viewer,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIROWBS) {
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
    ierr = MatLoad_MPIRowbs(viewer,type,newmat); CHKERRQ(ierr);
#else
    SETERRQ(1,"MatLoad: MATMPIROWBS format does not support complex numbers.");
#endif
  }
  else if (type == MATSEQBAIJ) {
    ierr = MatLoad_SeqBAIJ(viewer,type,newmat); CHKERRQ(ierr);
  }
  else {
    SETERRQ(1,"MatLoad: cannot load with that matrix type yet");
  }

  PLogEventEnd(MAT_Load,viewer,0,0,0);
  return 0;
}
