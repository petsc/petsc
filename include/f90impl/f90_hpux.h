/* $Id: f90_solaris.h,v 1.3 1998/09/25 00:09:49 balay Exp $ */

#if !defined (__F90_HPUX_H)
#define __F90_HPUX_H

typedef struct {
  long lower;   /* starting index of the fortran array */
  long extent;  /* length of the array */
  long mult;    /* in bytes */
} tripple;

/*
  The following constants are just
  guesses. It is strange that the HP folks use such 
  constants to represent the dimention
*/
#define f90_1D_ID  257
#define f90_2D_ID  513
#define f90_3D_ID  769
#define f90_4D_1D  1025
#define f90_COOKIE 443

/*
 addr   - address
 sd     - sizeof datatype
 ndim   - DIMENTION ID
 cookie - f90 cookie
 a      - junk - always 0. Null pointer??
 */

#define f90_header() void* addr; int sd; short ndim; short cookie; int a;

typedef struct {
  f90_header()
  tripple dim[1];
}array1d;

typedef struct {
  f90_header()
  tripple dim[2];   /* dim2,dim1 */
}array2d;

typedef struct {
  f90_header()
  tripple dim[3];   /* dim3,dim2,dim1 */
}array3d;

typedef struct {
  f90_header()
  tripple dim[4];   /* dim4,dim3,dim2,dim1 */
}array4d;

#endif
