#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: matv.c,v 1.4 1998/08/03 16:10:46 bsmith Exp bsmith $";
#endif

#include "pinclude/pviewer.h"
#include "sys.h"
#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "bitarray.h"

extern int MatView_Hybrid(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *) A->data;
  int         ierr, i,j, m = a->m, shift = a->indexshift, format;
  FILE        *fd;
  char        *outputname;
  int         mod, bsize = 6, bsub, col, col1, base, ict;

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetOutputname(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  fprintf(fd,"\nFIRST SUBMATRIX\n");
  if (format == VIEWER_FORMAT_ASCII_COMMON) {
    ict  = 0;
    bsub = 5;
    for ( i=0; i<m; i++ ) {
      if ((i+1)%bsize) {
        fprintf(fd,"row %d:",ict++);
        for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
          col1 = a->j[j]+shift;
          if ((mod = (col1+1)%bsize)) {
            if (a->a[j] != 0.0) {
              base = col1/bsize;
              col = bsub*base + mod-1;
              fprintf(fd," %d %g ",col,a->a[j]);
            }
          }
        }
        fprintf(fd,"\n");
      }
    }

    fprintf(fd,"\nNEXT SUBMATRIX\n");
    ict  = 0;
    bsub = 1;
    for ( i=0; i<m; i++ ) {
      if (!((i+1)%bsize)) {
        fprintf(fd,"row %d:",ict++);
        for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
          col1 = a->j[j]+shift;
          if (!(mod = (col1+1)%bsize)) {
            if (a->a[j] != 0.0) {
              base = col1/bsize;
              col = bsub*base;
              fprintf(fd," %d %g ",col,a->a[j]);
            }
          }
        }
        fprintf(fd,"\n");
      }
    }
  }
  fflush(fd);
  return 0;
}
