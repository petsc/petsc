#ifndef lint
static char vcid[] = "$Id: matio.c,v 1.8 1995/09/12 03:26:11 bsmith Exp bsmith $";
#endif

/* 
   This file contains simple binary read/write routines for matrices.
 */

#include "petsc.h"
#include <unistd.h>
#include "vec/vecimpl.h"
#include "sysio.h"
#include "pinclude/pviewer.h"
#include "matimpl.h"
#include "row.h"

extern int MatLoad_MPIRowbs(Viewer,MatType,Mat *);
extern int MatLoad_SeqAIJ(Viewer,MatType,Mat *);


/* -------------------------------------------------------------------- */

extern int MatGetFormatFromOptions_Private(MPI_Comm,MatType *,int *);

/* @
   MatLoad - Loads a matrix that has been stored in binary format
   with MatView().

   Input Parameters:
.  comm - MPI communicator
.  fd - file descriptor (not FILE pointer).  Use open() for this.
.  outtype - type of output matrix
.  ind - optional index set of matrix rows to be locally owned 
   (or 0 for loading the entire matrix on each processor)
.  ind2 - optional index set with new matrix ordering (size = global
   number of rows)

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
  ierr = MatGetFormatFromOptions_Private(vobj->comm,&type,&set); CHKERRQ(ierr);
  if (!set) type = outtype;

  PETSCVALIDHEADERSPECIFIC(vobj,VIEWER_COOKIE);
  if (vobj->type != BINARY_FILE_VIEWER)
   SETERRQ(1,"MatLoad: Invalid viewer; open viewer with ViewerFileOpenBinary()");

  if (type == MATMPIROWBS) {
    ierr = MatLoad_MPIRowbs(bview,type,newmat); CHKERRQ(ierr);
  }
  else if (type == MATSEQAIJ) {
    ierr = MatLoad_SeqAIJ(bview,type,newmat); CHKERRQ(ierr);
  }
  else {
    SETERRQ(1,"MatLoad: cannot load with that matrix type yet");
  }

  PLogEventEnd(MAT_Load,bview,0,0,0);
  return 0;
}
