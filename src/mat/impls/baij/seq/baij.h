/* $Id: baij.h,v 1.5 1996/03/25 23:12:36 balay Exp balay $ */

#include "matimpl.h"
#include <math.h>

#if !defined(__BAIJ_H)
#define __BAIJ_H

/*  
  MATSEQBAIJ format - Block compressed row storage. The i[] and j[] 
  arrays start at 1, or 0, depending on the value of shift.  
  For example, in Fortran  j[i[k]+p+shift] is the pth column in row k.
*/

typedef struct {
  int              sorted;       /* if true, rows are sorted by increasing columns */
  int              roworiented;  /* if true, row-oriented input, default */
  int              nonew;        /* if true, don't allow new elements to be added */
  int              singlemalloc; /* if true a, i, and j have been obtained with
                                        one big malloc */
  int              m,n;         /* rows, columns */
  int              bs,bs2;       /* block size, square of block size */
  int              mbs,nbs;      /* rows/bs, columns/bs */
  int              nz,maxnz;    /* nonzeros, allocated nonzeros */
  int              *diag;        /* pointers to diagonal elements */
  int              *i;           /* pointer to beginning of each row */
  int              *imax;        /* maximum space allocated for each row */
  int              *ilen;        /* actual length of each row */
  int              *j;           /* column values: j + i[k] - 1 is start of row k */
  Scalar           *a;           /* nonzero elements */
  IS               row,col;     /* index sets, used for reorderings */
  Scalar           *solve_work;  /* work space used in MatSolve */
  void             *spptr;       /* pointer for special library like SuperLU */
  int              reallocs;     /* number of mallocs done during MatSetValues() 
                                  as more values are set then were prealloced for */
  Scalar           *mult_work;   /* work array for matrix vector product*/
} Mat_SeqBAIJ;

extern int MatILUFactorSymbolic_SeqBAIJ(Mat,IS,IS,double,int,Mat *);
extern int MatConvert_SeqBAIJ(Mat,MatType,Mat *);
extern int MatConvertSameType_SeqBAIJ(Mat, Mat*,int);
extern int MatMarkDiag_SeqBAIJ(Mat);


#endif
