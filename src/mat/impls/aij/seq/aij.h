/* $Id: pdvec.c,v 1.10 1995/06/07 17:30:43 bsmith Exp $ */

#include "matimpl.h"
#include <math.h>

#if !defined(__AIJ_H)
#define __AIJ_H

/*  The i[] and j[] arrays start at 1 not zero to support Fortran 77 */
/*  In Fortran j[i[k]+p-1] is the pth column in row k */
 
/*
    singlemalloc indicates that a, i and j where all obtained with 
  one big malloc 
*/
typedef struct {
  int    sorted, roworiented, nonew, singlemalloc,assembled;
  int    m,n;                    /* rows, columns */
  int    nz,maxnz,mem;           /* nonzeros, allocated nonzeros, memory */
  int    *diag,                  /* diagonal elements */
         *i,*imax, *ilen,        /* j + i[k] - 1  is start of row k */
         *j;                     /* ilen is actual lenght of row */
  Scalar *a;     
  IS     row,col;
  Scalar *solve_work;            /* work space used in MatSolve_AIJ */
} Mat_AIJ;

#endif
