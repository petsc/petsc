/* $Id: bdiag.h,v 1.4 1994/06/16 20:38:34 curfman Exp curfman $ */

#ifndef _BDIAG

/*
   The SpMatBDiag (MATBDIAG) data structure is a block-diagonal format, 
   where each diagonal element consists of a square block of size nb x nb.
   Dense storage within each block is in column-major order.  As a 
   special case, blocks of size nb=1 (scalars) are supported as well.

   The diagonals are the full length of the matrix; this is a special 
   case of the more general SpMatDiag (MATDIAG) format.
 */

typedef struct {
    int        nd,         /* Number of block diagonals */
               nb,         /* Each diagonal element is an nb x nb matrix */
               *diag,      /* The value of (row-col)/nb for each diagonal */
               ndim,       /* Diagonals come from an ndim pde (if 0, ignore) */
               ndims[3],   /* Sizes of the mesh if ndim > 0 */
               user_alloc, /* True if the user provided the diagonals */
               *colloc;    /* Used to hold the column locations if
			      ScatterFromRow is used */
    double     **diagv,    /* The actual diagonals */
               *dvalue;    /* Used to hold a row if ScatterFromRow is used */
    } SpMatBDiag;

extern SpMat *SpBDCreate();
extern SpMat *SpBDCreateSubset();
extern SpMat *SpBDSubsetSorted();
extern SpMat *SpBDToSpR();
extern SpMat *SpBDFromSpR();
extern void  SpBDPrintNz();
int    *SpBDGetDiagNumbers();

/*M
   SpBDGetDiagNumbers - Returns a pointer to the array of diagonal numbers.

   Input Parameter:
.  mat - matrix context

   Returns:
   pointer to array of diagonal numbers

   Synopsis:
   int *SpBDGetDiagNumbers( mat )
   SpMat *mat;
M*/
#define SpBDGetDiagNumbers( mat )	((SpMatBDiag *)(mat)->data)->diag

#define _BDIAG
#endif
