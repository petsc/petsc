
#if !defined(__F90_IRIX_H)
#define __F90_IRIX_H

typedef struct {
  long lower;   /* starting index of the fortran array */
  long extent;  /* length of the array */
  long mult;    /* multiple of 4 bytes */
} tripple;
 
/* this might not be used in older version of compilers */
#define F90_COOKIE     -1744830464
#define F90_INT_ID     33562624
#define F90_DOUBLE_ID  52445192
#define F90_COMPLEX_ID 69238800

#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif

#if (PETSC_SIZEOF_VOID_P == 8)
#define F90_LONG_ID    35667976
#else
#define F90_LONG_ID    33562624
#endif


#define f90_header() \
void* addr;        /* Pointer to the data/array */ \
long  sd;          /* sizeof(DataType) in bits */  \
int   cookie; \
int   ndim;        /* No of dimentions */          \
int   a;  \
int   id; /* ID corresponding to datatype */ \
void* addr_r; /* address redefined */ \
long  size; /* len1*len2* ... * sd */

typedef struct {
  f90_header()
  tripple dim[1];
}F90Array1d;

typedef struct {
  f90_header()
  tripple dim[2];
}F90Array2d;

typedef struct {
  f90_header()
  tripple dim[3];
}F90Array3d;

typedef struct {
  f90_header()
  tripple dim[4];
}F90Array4d;

#endif
