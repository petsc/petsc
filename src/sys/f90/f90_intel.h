/* $Id: f90_intel.h,v 1.3 2000/09/22 18:54:10 balay Exp $ */

#if !defined (__F90_INTEL_H)
#define __F90_INTEL_H

 
typedef struct {
  long lower;   /* starting index of the fortran array */
  long upper;  /*  ending index of the array */
  long mult;    /* stride in no of datatype units */
} tripple;



#define f90_header() \
void*   addr_d;      /* addr -sumof(lower*mult) */ \
void*   addr;        /* Pointer to the data */ \
long    size;        /* len1*len2*len3... */ \
long    sd;          /* sizeof datatype */ \
long    cookie;      /* same as sizeof(datatype) */ \
long    ndim;        /* number of dimentions */


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
