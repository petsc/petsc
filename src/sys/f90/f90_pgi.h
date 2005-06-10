/* $Id: f90_pgi.h,v 1.1 2000/09/15 19:21:41 balay Exp $ */

#if !defined (__F90_PGI_H)
#define __F90_PGI_H

#define F90_COOKIE_1 35
#define F90_COOKIE_2 2
 
#define F90_INT_ID     25
#define F90_DOUBLE_ID  28
#define F90_COMPLEX_ID 10
#define F90_CHAR_ID    14

#define F90_LONG_ID  F90_INT_ID
#if !defined (PETSC_COMPLEX)
#define F90_SCALAR_ID F90_DOUBLE_ID
#else
#define F90_SCALAR_ID F90_COMPLEX_ID
#endif

#define f90_header() \
int   cookie1;        \
int   ndim;    /* No of dimentions */          \
int   id;      /* integer id representing the datatype */ \
int   sd;      /* sizeof(DataType) */          \
int   cookie2;        \
int   array_len; /* complete len of the array = len1*len2*len3 */ \
int   array_len_dup; /* same as above */ \
int   a,b;            \
int   sum_d      /* -sumof(lower*mult) + 1 */

typedef struct {
  f90_header()
}F90Array1d;

#define F90Array2d F90Array1d
#define F90Array3d F90Array1d
#define F90Array4d F90Array1d

#endif
