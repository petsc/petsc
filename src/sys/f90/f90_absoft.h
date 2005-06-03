
#if !defined(__F90_ABSOFT_H)
#define __F90_ABSOFT_H

typedef struct {
  long lower;   /* starting index of the fortran array */
  long extent;  /* length of the array */
  long mult;    /* multiple of 4 bytes (except for char)*/
} tripple;
 
/* this might not be used in older version of compilers */
#define F90_COOKIE     21
#define F90_INT_ID     131074
#define F90_DOUBLE_ID  134480899
#define F90_COMPLEX_ID 268960772
#define F90_CHAR_ID    32774

#define F90_LONG_ID F90_INT_ID

#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif

#define f90_header() \
void* addr;      /* Pointer to the data/array */ \
int   sd;        /* sizeof(DataType) in bits */  \
short cookie;    \
short dim_id;    /* No of dimentions */          \
int   id;        /* ID corresponding to datatype */ \
int   a,b;

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
