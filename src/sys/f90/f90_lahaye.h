#if !defined (__F90_LAHAYE_H)
#define __F90_LAHAYE_H

typedef struct {
  long lower;   /* starting index of the fortran array */
  long upper;   /* ending index of the array */
  long mult;    /* in bytes */
  long extent;  /* length of the array */
} tripple;

#define f90_header1() \
void* addr;    /* Pointer to the data/array */ \
long  id1;     /* untouched */ \
long  dimn;    /* extent1* extent2* extent3 */

#define f90_header2() \
long  id2;      /* untouched */ \
long  dimb;     /* dimn * sizeof(type) */

typedef struct {
  f90_header1()
  tripple dim[1];
  f90_header2()
}F90Array1d;

typedef struct {
  f90_header1()
  tripple dim[2];   /* dim1,dim2 */
  f90_header2()
}F90Array2d;

typedef struct {
  f90_header1()
  tripple dim[3];   /* dim1,dim2,dim3 */
  f90_header2()
}F90Array3d;

typedef struct {
  f90_header1()
  tripple dim[4];   /* dim1,dim2,dim3,dim4 */
  f90_header2()
}F90Array4d;

#endif
