
#include "matimpl.h"
#include "math.h"

/*  The i[] and j[] arrays start at 1 not zero to support Fortran 77 */
/*  In Fortran j[i[k]+p-1] is the pth column in row k */
 
/*
    singlemalloc indicates that a, i and j where all obtained with 
  one big malloc 
*/
typedef struct {
  int    sorted, roworiented, nonew, singlemalloc;
  int    m,n,nz,mem,*diag,       /* rows and columns */
         *i,*imax, *ilen,        /* j + i[k] - 1  is start of row k */
         *j;                     /* ilen is actual lenght of row */
  Scalar *a;     
} Matiaij;


