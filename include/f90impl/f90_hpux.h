
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
  constants to represent the dimension
*/
#define F90_1D_ID  257
#define F90_2D_ID  513
#define F90_3D_ID  769
#define F90_4D_1D  1025
#define F90_COOKIE 443

/*
 addr   - address
 sd     - sizeof datatype
 ndim   - DIMENSION ID
 cookie - f90 cookie
 a      - junk - always 0. Null pointer??
 */

#define f90_header() void* addr; long sd; short ndim; short cookie; long a;

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
