/* $Id: f90_IRIX.h,v 1.1 1998/03/12 18:01:55 balay Exp balay $ */


typedef struct {
  long lower;   /* starting index of the fortran array */
  long extent;  /* length of the array */
  long mult;    /* multiple of 4 bytes */
} tripple;
 
#define f90_header() \
void* addr;        /* Pointer to the data/array */ \
long  sd;          /* sizeof(DataType) */          \
int   unknown;\
int   ndim;        /* No of dimentions */          \
int   a,b,c; \
long  d;


typedef struct {
  f90_header()
  tripple dim[1];
}array1d;

typedef struct {
  f90_header()
  tripple dim[2];
}array2d;

typedef struct {
  f90_header()
  tripple dim[3];
}array3d;

typedef struct {
  f90_header()
  tripple dim[4];
}array4d;

