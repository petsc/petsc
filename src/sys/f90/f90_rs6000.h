
#if !defined (__F90_RS6000_H)
#define __F90_RS6000_H

typedef struct {
  long lower;   /* starting index of the fortran array */
  long extent;  /* length of the array */
  long mult;    /* in bytes */
} tripple;

/*
  The following constants are just
  guesses. The program behavies strangly
  if these constants are not set in
  the f90 pointer
*/
#define F90_CHAR_ID    770
#define F90_INT_ID     781
#define F90_DOUBLE_ID  782
#define F90_COMPLEX_ID 783
#define F90_COOKIE     20481

#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif

#define f90_header() \
void* addr;    /* Pointer to the data/array */ \
short id;      /* integer id representing the datatype */ \
short cookie;  /* a wiered f90 cookie */ \
int   sd;      /* sizeof(DataType) */          \
int   ndim;    /* No of dimentions */          \
int   sum_d;   /* -sumof(lower*mult) */

typedef struct {
  f90_header()
  tripple dim[1];
}F90Array1d;

typedef struct {
  f90_header()
  tripple dim[2];   /* dim2,dim1 */
}F90Array2d;

typedef struct {
  f90_header()
  tripple dim[3];   /* dim3,dim2,dim1 */
}F90Array3d;

typedef struct {
  f90_header()
  tripple dim[4];   /* dim4,dim3,dim2,dim1 */
}F90Array4d;

#endif
