/* $Id: f90_alpha.h,v 1.2 2000/07/18 20:15:01 balay Exp balay $ */

#if !defined (__F90_ALPHA_H)
#define __F90_ALPHA_H
 
typedef struct {
  long mult;    /* stride in bytes */
  long upper;   /* ending index of the array */
  long lower;   /* starting index of the fortran array */
} tripple;

/*
  The following constants are just
  guesses. The program behavies strangly
  if these constants are not set in
  the f90 pointer
*/
# if defined(PARCH_linux)
#define F90_COOKIE 1282
#else
#define F90_COOKIE 258
#endif

#define F90_CHAR_ID    2574
#define F90_INT_ID     2564
#define F90_DOUBLE_ID  2570
#define F90_COMPLEX_ID 2573


#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif


#define f90_header() \
short          cookie;  /* a wiered f90 cookie */ \
short          id;      /* integer id representing the datatype */ \
long           sd;      /* sizeof(DataType) in bits */  \
void*          addr;    /* Pointer to the data */ \
long           a;       /* unknown stuff - always 0 */ \
void*          addr_d;  /* addr-sumof(lower*mult) */ \
int            ndim;    /* No of dimensions */

typedef struct {
  f90_header()
  tripple dim[1];
}array1d;

typedef struct {
  f90_header()
  tripple dim[2];   /* dim1,dim2 */
}array2d;

typedef struct {
  f90_header()
  tripple dim[3];   /* dim1,dim2,dim3 */
}array3d;

typedef struct {
  f90_header()
  tripple dim[4];   /* dim1,dim2,dim3,dim4 */
}array4d;


#endif
