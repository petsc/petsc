
#include  "gmresp.h"

/*
   As the final part of the solution process in gmres, the Krylov 
   vectors are combined together.  This is a block operation, and
   can be more efficiently performed as such.

   This is a basic routine that simply uses the maxpy operation;
   versions that use 2, 4, or more at a time can be found in ...

/*
    This is a multi-maxpy.
    input parameters:
        v1    - vectors to use.  v1, v1+1, ..., v1+nv
        nv    - number of vectors to use, - 1
        p1    - array of multipliers
        dv    - destination for result.
 */
int BasicMultiMaxpy(  Vec *v1,int nv,double *p1, Vec dv )
{
  int j;
  for (j=0; j<=nv; j++) VecAXPY( p1++, *v1++, dv );
  return 0;
}

