#ifndef lint
static char vcid[] = "$Id: matio.c,v 1.11 1995/09/30 19:29:59 bsmith Exp bsmith $";
#endif

/* 
   This file contains simple binary read/write routines for matrices.
 */

#include "petsc.h"
#include "vec/vecimpl.h"
#include "../matimpl.h"
#include "sysio.h"
#include "pinclude/pviewer.h"

extern int MatLoad_MPIRowbs(Viewer,MatType,Mat *);
extern int MatLoad_SeqAIJ(Viewer,MatType,Mat *);
extern int MatLoad_SeqRow(Viewer,MatType,Mat *);
extern int MatLoad_MPIAIJ(Viewer,MatType,Mat *);

/* @
   MatLoad - Loads a matrix that has been stored in binary format
   with MatView().

   Input Parameters:
.  bview - Binary file viewer, created with ViewerFileOpenBinary()
.  outtype - Type of matrix desired, for example MATSEQAIJ, MATMPIROWBS,..

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

   Currently, the _entire_ matrix must be loaded.  This should
   probably change.

.seealso: MatView(), VecLoad() 
*/  
int MatLoad(Viewer bview,MatType outtype,Mat *newmat)
{
  PetscObject vobj = (PetscObject) bview;
  int         ierr,set;
  MatType     type;
  *newmat = 0;

  PLogEventBegin(MAT_Load,bview,0,0,0);
  ierr = MatGetFormatFromOptions(vobj->comm,&type,&set); CHKERRQ(ierr);
  if (!set) type = outtype;

  PETSCVALIDHEADERSPECIFIC(vobj,VIEWER_COOKIE);
  if (vobj->type != BINARY_FILE_VIEWER)
   SETERRQ(1,"MatLoad: Invalid viewer; open viewer with ViewerFileOpenBinary()");

  if (type == MATSEQAIJ) {
    ierr = MatLoad_SeqAIJ(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATMPIAIJ || type == MATMPIROW) {
    ierr = MatLoad_MPIAIJ(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQROW) {
    ierr = MatLoad_SeqRow(bview,type,newmat); CHKERRQ(ierr);
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
