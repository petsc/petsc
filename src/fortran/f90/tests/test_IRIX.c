
#include <stdio.h>
#include "../f90_IRIX.h"

int    *iarray;
double *darray;

void foo1_(int*a,double *b)
{
  iarray = a;
  darray = b;
}

void foo2_(array3d *a,array3d *b)
{
  int i;
  int *aa = (int*)a;
  int *bb = (int*)b;
 
  printf("Integer Structure Contents: ");
  for(i = 0; i< 40; i++) printf("%d ",aa[i]);
  printf("\n\n");

  if ((int*)a->addr == iarray) printf("Data Address Matches\n");
  else printf("*** Data Address Mismatch*** \n");

  printf ("Junk Values a,b,c,d,unknown = %d,%d,%d,%ld,%d\n",a->a,a->b,a->c,a->d,a->unknown);
  printf("Array ndim =  %d,Sizeof(data) = %ld \n", a->ndim,a->sd);
  printf("  Dim  StartIndex Len offset\n");
 
  for (i=0; i< a->ndim; i++)
  printf("   %1d   %3ld      %4ld  %6ld\n",i,a->dim[i].lower,a->dim[i].extent,
         a->dim[i].mult);
 
  
  printf("\n\nDouble Structure Contents: ");
  for(i = 0; i< 40; i++) printf("%d ",bb[i]);
  printf("\n\n");

  if ((int*)a->addr == iarray) printf("Data Address Matches\n");
  else printf("*** Data Address Mismatch*** \n");

  printf ("Junk Values a,b,c,d,unknown = %d,%d,%d,%ld,%d\n",b->a,b->b,b->c,b->d,b->unknown);
  printf("Array ndim =  %d,Sizeof(data) = %ld \n",b->ndim,b->sd);
  printf("  Dim  StartIndex Len offset\n");
 
  for (i=0; i< b->ndim; i++)
  printf("   %1d   %3ld      %4ld  %6ld\n",i,b->dim[i].lower,b->dim[i].extent,
         b->dim[i].mult);
 
}

