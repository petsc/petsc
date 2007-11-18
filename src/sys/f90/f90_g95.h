/* $Id: f90_g95.h,v 1.3 2000/09/22 18:54:10 balay Exp $ */

#if !defined (__F90_G95_H)
#define __F90_G95_H

 
typedef struct {
  int  mult;    /* stride in no of datatype units */
  int  lower;   /* starting index of the fortran array */
  int  upper;  /*  ending index of the array */
} tripple;



#define f90_header() \
void*   addr_d;      /* addr -sumof(lower*mult) */ \
int     ndim;        /* number of dimentions */\
int     sd;          /* sizeof datatype */ \
void*   addr;        /* Pointer to the data */

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
