/* $Id: f90_solaris.h,v 1.1 2000/07/21 00:55:59 balay Exp balay $ */

#if !defined (__F90_SOLARIS_H)
#define __F90_SOLARIS_H
 
#define f90_header(dim) \
void*   addr;        /* Pointer to the data */ \
long    extent[dim]; /* length of array */ \
long    mult[dim];   /* stride in bytes */ \
void*   addr_d;      /* addr -sumof(lower*mult) */ \
long    lower[dim];  

typedef struct {
  f90_header(1)   /* dim1 */
}array1d;

typedef struct {
  f90_header(2)   /* dim1,dim2 */
}array2d;

typedef struct {
  f90_header(3)    /* dim1,dim2,dim3 */
}array3d;

typedef struct {
  f90_header(4)   /* dim1,dim2,dim3,dim4 */
}array4d;


#endif
