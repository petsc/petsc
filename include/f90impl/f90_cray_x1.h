
#if !defined (__F90_CRAY_X1_H)
#define __F90_CRAY_X1_H

/* this code is almost the same as on t3e - except for the IDs
 (and datatypes chnaged to match sizes on t3e) */
#define F90_INT_ID     33562624
#define F90_DOUBLE_ID  58736640
#define F90_COMPLEX_ID 67238800
#define F90_COOKIE     -1742471168

#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif

typedef struct {
  long lower;   /* starting index of the fortran array */
  long extent;  /* length of the array */
  long mult;    /* no of datatype units */
} tripple;
 
#define f90_header() \
void* addr;        /* Pointer to the data/array */  \
long  sd;          /* sizeof(DataType) */          \
int   cookie;      /* cookie*/                     \
int   ndim;        /* No of dimentions */          \
long  id;          /* Integer? double? */          \
long  a,b;


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
