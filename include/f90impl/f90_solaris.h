
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
}F90Array1d;

typedef struct {
  f90_header(2)   /* dim1,dim2 */
}F90Array2d;

typedef struct {
  f90_header(3)    /* dim1,dim2,dim3 */
}F90Array3d;

typedef struct {
  f90_header(4)   /* dim1,dim2,dim3,dim4 */
}F90Array4d;


#endif
