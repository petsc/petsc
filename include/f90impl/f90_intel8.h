/* $Id: f90_intel.h,v 1.3 2000/09/22 18:54:10 balay Exp $ */

#if !defined (__F90_INTEL8_H)
#define __F90_INTEL8_H

/* Pretty similar to Compaq DVF (f90_win32.h) */
 
typedef struct {
  long extent;  /* length of the array */
  long mult;    /* stride in bytes */
  long lower;   /* starting index of the fortran array */
} tripple;

#define F90_COOKIE7 7
#define F90_COOKIE0 0


#define f90_header() \
void*   addr;        /* Pointer to the data */ \
long    sd;          /* sizeof datatype */ \
long    sum_d;       /* -sumof(lower*mult) */ \
long    a;           /* always 7 */ \
long    ndim;        /* number of dimentions */ \
long    b;           /* always 0 */


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
