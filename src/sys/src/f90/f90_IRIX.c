#ifndef lint
static char vcid[] = "$Id: f90_IRIX.c,v 1.1 1998/03/12 18:01:55 balay Exp balay $";
#endif

/*
         This file contains the code to map between Fortran 90 
  pointers and traditional C pointers for the NAG F90 compiler.
*/
#if defined(HAVE_IRIXF90)

#include "src/fortran/custom/zpetsc.h"
#include "src/fortran/f90/f90_IRIX.h"

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
int PetscF90Create1dArrayScalar(void *array,int len, array1d *ptr)
{
  ptr->addr          = array;
  ptr->dim[0].extent = len;
  ptr->dim[0].mult   = sizeof(Scalar)/sizeof(int);
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
int PetscF90Get1dArrayScalar(array1d *ptr,void **array)
{
  *array = (void *) ptr->addr;
  return 0;
}

/*
    PetscF90Destroy1dArrayScalar - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy1dArrayScalar(array1d *ptr)
{
  ptr->addr  = (void *)0xffffffff;
  return 0;
}
/* --------------------------------------------------------*/
/*
    PetscF90Create2dArrayScalar - Given a C pointer to a two dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
.   array - regular C pointer (address)
.   m - number of rows in array
.   n - number of columns in array

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Create2dArrayScalar(void *array,int m,int n, array2d *ptr)
{
  ptr->addr          = array;
  ptr->dim[0].extent = m;
  ptr->dim[0].mult   = sizeof(Scalar)/sizeof(int);
  ptr->dim[0].lower  = 1;
  ptr->dim[1].extent = n;
  ptr->dim[1].mult   = m*sizeof(Scalar)/sizeof(int);
  ptr->dim[1].lower  = 1;
  return 0;
}

/*
    PetscF90Get2dArrayScalar - Gets the address for the data 
       stored in a 2d Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 
int PetscF90Get2dArrayScalar(array2d *ptr,void **array)
{
  *array = (void *) ptr->addr;
  return 0;
}

/*
    PetscF90Destroy2dArrayScalar - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy2dArrayScalar(array2d *ptr)
{
  ptr->addr  = (void *)0xffffffff;
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
int PetscF90Create1dArrayInt(void *array,int len, array1d *ptr)
{
  ptr->addr          = array;
  ptr->dim[0].extent = len;
  ptr->dim[0].mult   = sizeof(int)/sizeof(int);
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
int PetscF90Get1dArrayInt(array1d *ptr,void **array)
{
  *array = (void *) ptr->addr;
  return 0;
}

/*
    PetscF90Destroy1dArrayInt - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy1dArrayInt(array1d *ptr)
{
  ptr->addr  = (void *)0xffffffff;
  return 0;
}

#endif



