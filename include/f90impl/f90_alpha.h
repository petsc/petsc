
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

#define F90_INT_ID     3
#define F90_LONG_ID    4
#define F90_DOUBLE_ID  10
#define F90_COMPLEX_ID 13
#define F90_CHAR_ID    14

#if defined(PARCH_linux)
#define A_VAL 5
#else
#define A_VAL 1
#endif

#define B_VAL 10

#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif


#define f90_header() \
char           ndim,a;  /* No of dimensions, a=1 */ \
char           id,b;    /* char id representing the datatype, b=0 */ \
int            c;       /* c=0 */ \
long           sd;      /* sizeof(DataType) in bits */  \
void*          addr;    /* Pointer to the data */ \
long           d;       /* d=0 */ \
void*          addr_d;  /* addr-sumof(lower*mult) */

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
