#ifndef lint
static char vcid[] = "$Id: f90_nag.c,v 1.1 1997/01/15 23:23:17 bsmith Exp bsmith $";
#endif

/*
         This file contains the code to map between Fortran 90 
  pointers and traditional C pointers for the NAG F90 compiler.
*/
#if defined(HAVE_NAGF90)

#include "src/fortran/custom/zpetsc.h"
#include "/usr/local/lib/f90/f90.h"

/* --------------------------------------------------------*/
/*
    PetscF90Create1dArrayScalar - Given a C pointer to a one dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
.   array - regular C pointer (address)
.   len - length of array (in items)

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Create1dArrayScalar(void *array,int len, Dope1 *ptr)
{
  ptr->addr          = array;
  ptr->offset        = -sizeof(Scalar);
  ptr->dim[0].extent = len;
  ptr->dim[0].mult   = sizeof(Scalar);
  ptr->dim[0].lower  = 1;
  return 0;
}

/*
    PetscF90Get1dArrayScalar - Gets the address for the data 
       stored in a Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 
int PetscF90Get1dArrayScalar(Dope1 *ptr,void **array)
{
  *array = (void *) ptr->addr;
  return 0;
}

/*
    PetscF90Destroy1dArrayScalar - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy1dArrayScalar(Dope1 *ptr)
{
  ptr->addr  = (Pointer)0xffffffff;
  return 0;
}
/* --------------------------------------------------------*/
/*
    PetscF90Create2dArrayScalar - Given a C pointer to a one dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
.   array - regular C pointer (address)
.   m - number of rows in array
.   n - number of columns in array

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Create2dArrayScalar(void *array,int m,int n, Dope2 *ptr)
{
  ptr->addr          = array;
  ptr->offset        = -sizeof(Scalar)-m*sizeof(Scalar);
  ptr->dim[0].extent = m;
  ptr->dim[0].mult   = sizeof(Scalar);
  ptr->dim[0].lower  = 1;
  ptr->dim[1].extent = n;
  ptr->dim[1].mult   = m*sizeof(Scalar);
  ptr->dim[1].lower  = 1;
  return 0;
}

/*
    PetscF90Get2dArrayScalar - Gets the address for the data 
       stored in a Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 
int PetscF90Get2dArrayScalar(Dope2 *ptr,void **array)
{
  *array = (void *) ptr->addr;
  return 0;
}

/*
    PetscF90Destroy2dArrayScalar - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy2dArrayScalar(Dope2 *ptr)
{
  ptr->addr  = (Pointer)0xffffffff;
  return 0;
}
/* -----------------------------------------------------------------*/

/*
    PetscF90Create1dArrayInt - Given a C pointer to a one dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
.   array - regular C pointer (address)
.   len - length of array (in items)

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Create1dArrayInt(void *array,int len, Dope1 *ptr)
{
  ptr->addr          = array;
  ptr->offset        = -sizeof(int);
  ptr->dim[0].extent = len;
  ptr->dim[0].mult   = sizeof(int);
  ptr->dim[0].lower  = 1;
  return 0;
}

/*
    PetscF90Get1dArrayInt - Gets the address for the data 
       stored in a Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 
int PetscF90Get1dArrayInt(Dope1 *ptr,void **array)
{
  *array = (void *) ptr->addr;
  return 0;
}

/*
    PetscF90Destroy1dArrayInt - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy1dArrayInt(Dope1 *ptr)
{
  ptr->addr  = (Pointer)0xffffffff;
  return 0;
}

#endif



