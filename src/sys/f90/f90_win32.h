
#if !defined (__F90_WIN32_H)
#define __F90_WIN32_H
 
typedef struct {
  long extent;  /* length of the array */
  long mult;    /* stride in bytes */
  long lower;   /* starting index of the fortran array */
} tripple;

/*
  The following constants are just
  guesses. The program behavies strangly
  if these constants are not set in
  the f90 pointer
*/
#define F90_CHAR_ID    234881027
#define F90_INT_ID     50331651
#define F90_DOUBLE_ID  167772163
#define F90_COMPLEX_ID 218103811


#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif


#define f90_header() \
void*          addr;    /* Pointer to the data */ \
long           sd;      /* sizeof(DataType) in bits */  \
long           sum_d;   /* -sumof(lower*mult) */ \
int            id;      /* integer id representing the datatype */ \
int            ndim;    /* No of dimensions */

typedef struct {
  f90_header()
  tripple dim[1];
}F90Array1d;

typedef struct {
  f90_header()
  tripple dim[2];   /* dim1,dim2 */
}F90Array2d;

typedef struct {
  f90_header()
  tripple dim[3];   /* dim1,dim2,dim3 */
}F90Array3d;

typedef struct {
  f90_header()
  tripple dim[4];   /* dim1,dim2,dim3,dim4 */
}F90Array4d;


#endif
