#ifndef lint
static char vcid[] = "$Id: matio.c,v 1.5 1995/09/06 13:35:15 curfman Exp bsmith $";
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

extern int MatLoad_MPIRowbs(Viewer,MatType,IS,IS,Mat *);


/* -------------------------------------------------------------------- */
/* This version reads from MATROW and writes to MATAIJ/MATROW 
   implementation.  Eventually should not use generic MatSetValues, 
   but rather directly read data into appropriate location. Also,
   should be able to read/write to/from any implementation. */

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

.seealso: MatView(), VecLoadBinary() 
*/  
int MatLoad(Viewer bview,MatType outtype,IS ind,IS ind2,Mat *newmat)
{
  Mat         mat;
  int         rows, i, nz, nnztot, *rlen, ierr, lsize, gsize, *rptr, j, dstore;
  int         *cwork, rstart, rend, readst, *pind, *pind2, iinput, iglobal, fd;
  Scalar      *awork;
  long        startloc, mark;
  PetscObject vobj = (PetscObject) bview;
  MatType     type;
  MPI_Comm    comm = vobj->comm;

  PETSCVALIDHEADERSPECIFIC(vobj,VIEWER_COOKIE);
  if (vobj->type != BINARY_FILE_VIEWER)
   SETERRQ(1,"MatLoad: Invalid viewer; open viewer with ViewerFileOpenBinary().");

  if (outtype == MATMPIROW_BS) {
    return MatLoad_MPIRowbs(bview,type,ind,ind2,newmat);
  }
  ierr = ViewerFileGetDescriptor_Private(bview,&fd); CHKERRQ(ierr);

  /* Get the location of the beginning of the matrix data, in case the
  file contains multiple elements */
  startloc = lseek(fd,0L,SEEK_CUR);
  /* MPIU_printf(comm,"startloc=%d\n",startloc); */
  type = MATROW;
  if (outtype != MATROW && outtype != MATAIJ && outtype != MATMPIROW && 
    outtype != MATMPIAIJ) SETERRQ(1,
    "MatLoadBinary: Only MATROW, MATAIJ, MATMPIROW, & MATAMPIAIJ supported.");

  /* Read matrix header.  Should this really be the full header? */
  ierr = SYRead(fd,(char *)&type,sizeof(int),SYINT); CHKERRQ(ierr);
  if (type != MATROW) 
      SETERRQ(1,"MatLoadBinary: Only MATROW input currently supported");
  ierr  = SYRead(fd,(char *)&rows,sizeof(int),SYINT); CHKERRQ(ierr);
  ierr  = SYRead(fd,(char *)&nnztot,sizeof(int),SYINT); CHKERRQ(ierr);
  MPIU_printf(comm,"Input matrix: rows=%d, nnztot=%d\n",rows,nnztot);

  /* Check sizes, form index set if necessary */
  if (!ind) 
    {ierr = ISCreateStrideSequential(comm,rows,0,1,&ind); CHKERRQ(ierr);}
  ierr = ISGetLocalSize(ind,&lsize); CHKERRQ(ierr);
  MPI_Allreduce(&lsize,&gsize,1,MPI_INT,MPI_SUM,comm);
  if (gsize != rows) 
    SETERRQ(1,"MatLoadBinary: Incompatible parallel matrix size.");
  ierr = ISGetIndices(ind,&pind); CHKERRQ(ierr);
  ierr = ISGetIndices(ind,&pind2); CHKERRQ(ierr);

  /* Allocate work space */
  rlen  = (int *) PETSCMALLOC( rows * sizeof(int)); CHKPTRQ(rlen);
  rptr  = (int *) PETSCMALLOC( (rows+1) * sizeof(int)); CHKPTRQ(rptr);
  cwork = (int *) PETSCMALLOC( rows*sizeof(int)); CHKPTRQ(cwork);
  awork = (Scalar *) PETSCMALLOC( rows*sizeof(Scalar)); CHKPTRQ(awork);

  /* Read row length info and form matrix memory allocation size */
  ierr = SYRead(fd,(char *)rlen,rows*sizeof(int),SYINT); CHKERRQ(ierr);
  ierr = SYReadBuffer(-1,(long)0,0,(char*)0,SYINT); CHKERRQ(ierr);

   /* This should be fixed */
  dstore = 5;
  for ( i=0; i<lsize; i++ ) rptr[i] = PETSCMAX(rlen[pind[i]] - dstore,0);

  /* Form new matrix */
  if (outtype == MATROW)
    ierr = MatCreateSequentialRow(comm,rows,rows,0,rlen,&mat);
  else if (outtype == MATAIJ)
    ierr = MatCreateSequentialAIJ(comm,rows,rows,0,rlen,&mat);
  else if (outtype == MATMPIROW)
    ierr = MatCreateMPIRow(comm,lsize,PETSC_DECIDE,gsize,gsize,dstore,
           0,0,rptr,&mat);
  else if (outtype == MATMPIAIJ)
    ierr = MatCreateMPIAIJ(comm,lsize,PETSC_DECIDE,gsize,gsize,dstore,
           0,0,rptr,&mat);
  CHKERRQ(ierr);

  /* Form row pointers */
  rptr[0] = 0;
  for (i=0; i<rows; i++) rptr[i+1] = rptr[i] + rlen[i];

  MatGetOwnershipRange(mat,&rstart,&rend);
  mark = startloc + (rows+2)*sizeof(int);
  for ( i=0; i<lsize; i++ ) { 
    iglobal = i + rstart;
    iinput  = pind[i];
    nz      = rlen[iinput];
    readst = mark + rptr[iinput]*(sizeof(int)+sizeof(Scalar));
    ierr = SYReadBuffer(fd,readst,nz*sizeof(int),(char*)cwork,SYINT);
    CHKERRQ(ierr);
    ierr = SYReadBuffer(fd,readst+nz*sizeof(int),nz*sizeof(Scalar),
           (char *)awork,SYSCALAR); CHKERRQ(ierr);
    for (j=0; j<nz; j++) cwork[j] = pind2[cwork[j]];
    ierr = MatSetValues(mat,1,&iglobal,nz,cwork,awork,INSERTVALUES); 
    CHKERRQ(ierr);
  }
  PETSCFREE(rlen); PETSCFREE(rptr); PETSCFREE(cwork); PETSCFREE(awork);
  ierr = MatAssemblyBegin(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = ISRestoreIndices(ind,&pind); CHKERRQ(ierr);
  ierr = ISRestoreIndices(ind,&pind2); CHKERRQ(ierr);

  *newmat = mat;
  return 0;
}
