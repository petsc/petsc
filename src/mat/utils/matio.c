#ifndef lint
static char vcid[] = "$Id: matio.c,v 1.6 1995/09/07 04:27:09 bsmith Exp bsmith $";
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


/* -------------------------------------------------------------------- */


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
  *newmat = 0;

  PETSCVALIDHEADERSPECIFIC(vobj,VIEWER_COOKIE);
  if (vobj->type != BINARY_FILE_VIEWER)
   SETERRQ(1,"MatLoad: Invalid viewer; open viewer with ViewerFileOpenBinary()");

  if (outtype == MATMPIROW_BS) {
    return MatLoad_MPIRowbs(bview,outtype,newmat);
  }
  else {
    SETERRQ(1,"MatLoad: cannot load with that matrix type yet");
  }

  return 0;
}
