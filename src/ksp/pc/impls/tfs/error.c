#define PETSCKSP_DLL

/**********************************error.c*************************************

Author: Henry M. Tufo III

e-mail: hmt@cs.brown.edu

snail-mail:
Division of Applied Mathematics
Brown University
Providence, RI 02912

Last Modification: 
6.21.97
***********************************error.c************************************/

/**********************************error.c*************************************
File Description:
-----------------

***********************************error.c************************************/
#include "src/ksp/pc/impls/tfs/tfs.h"

/**********************************error.c*************************************
Function error_msg_fatal()

Input : pointer to formatted error message.
Output: prints message to stdout.
Return: na.
Description: prints error message and terminates program.
***********************************error.c************************************/
void error_msg_fatal(const char msg[], ...)
{
  va_list    ap;
  const char *p;
  char       *sval, cval;
  int        ival;
  PetscScalar       dval;


  /* print error message along w/node identifier */
  va_start(ap,msg);
  printf("%d :: FATAL :: ", my_id);
  for (p=msg; *p; p++)
    {
      if (*p != '%')
	{
	  putchar(*p);
	  continue;
	}
      switch (*++p) {
      case 'c':
	cval = va_arg(ap,int);
	  putchar(cval);
	break;
      case 'd':
	ival = va_arg(ap,int);
	printf("%d",ival);
	break;
      case 'e':
	dval = va_arg(ap,PetscScalar);
	printf("%e",dval);
	break;
      case 'f':
	dval = va_arg(ap,PetscScalar);
	printf("%f",dval);
	break;
      case 'g':
	dval = va_arg(ap,PetscScalar);
	printf("%g",dval);
	break;
      case 's':
	for (sval=va_arg(ap,char *); *sval; sval++)
	  {putchar(*sval);}
	break;
      default:
	putchar(*p);
	break;
      }
    }
  /* printf("\n"); */
  va_end(ap);

  fflush(stdout);

  /* Try with MPI_Finalize() as well _only_ if all procs call this routine */
  /* Choose a more meaningful error code than -12 */
  MPI_Abort(MPI_COMM_WORLD, -12);
}



/**********************************error.c*************************************
Function error_msg_warning()

Input : formatted string and arguments.
Output: conversion printed to stdout.
Return: na.
Description: prints error message.
***********************************error.c************************************/
void 
error_msg_warning(const char msg[], ...)
{
  /* print error message along w/node identifier */
#if   defined V
  va_list ap;
  char *p, *sval, cval;
  int ival;
  PetscScalar dval;

  va_start(ap,msg);
  if (!my_id)
    {
      printf("%d :: WARNING :: ", my_id);
      for (p=msg; *p; p++)
	{
	  if (*p != '%')
	    {
	      putchar(*p);
	      continue;
	    }
	  switch (*++p) {
	  case 'c':
	    cval = va_arg(ap,char);
	    putchar(cval);
	    break;
	  case 'd':
	    ival = va_arg(ap,int);
	    printf("%d",ival);
	    break;
	  case 'e':
	    dval = va_arg(ap,PetscScalar);
	    printf("%e",dval);
	    break;
	  case 'f':
	    dval = va_arg(ap,PetscScalar);
	    printf("%f",dval);
	    break;
	  case 'g':
	    dval = va_arg(ap,PetscScalar);
	    printf("%g",dval);
	    break;
	  case 's':
	    for (sval=va_arg(ap,char *); *sval; sval++)
	      {putchar(*sval);}
	    break;
	  default:
	    putchar(*p);
	    break;
	  }
	}
      /*      printf("\n"); */
    }
  va_end(ap);


#elif defined VV
  va_list ap;
  char *p, *sval, cval;
  int ival;
  PetscScalar dval;
  va_start(ap,msg);
  if (my_id>=0)
    {
      printf("%d :: WARNING :: ", my_id);
      for (p=msg; *p; p++)
	{
	  if (*p != '%')
	    {
	      putchar(*p);
	      continue;
	    }
	  switch (*++p) {
	  case 'c':
	    cval = va_arg(ap,char);
	    putchar(cval);
	    break;
	  case 'd':
	    ival = va_arg(ap,int);
	    printf("%d",ival);
	    break;
	  case 'e':
	    dval = va_arg(ap,PetscScalar);
	    printf("%e",dval);
	    break;
	  case 'f':
	    dval = va_arg(ap,PetscScalar);
	    printf("%f",dval);
	    break;
	  case 'g':
	    dval = va_arg(ap,PetscScalar);
	    printf("%g",dval);
	    break;
	  case 's':
	    for (sval=va_arg(ap,char *); *sval; sval++)
	      {putchar(*sval);}
	    break;
	  default:
	    putchar(*p);
	    break;
	  }
	}
      /* printf("\n"); */
    }
  va_end(ap);
#endif

#ifdef DELTA  
  fflush(stdout);
#else
  fflush(stdout);
  /*  fflush(NULL); */
#endif

}




