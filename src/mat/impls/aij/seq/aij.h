
#include "matimpl.h"
#include "math.h"

typedef struct {
  int    m,n,nz,           /* rows and columns */
         *i,*imax, *ilen,  /* j + i[k]  is start of row k */
         *j;               /* ilen is actual lenght of row */
  double *a;     
} Matiaij;

int MatAIJCreate(); /* (int, int, int); */

