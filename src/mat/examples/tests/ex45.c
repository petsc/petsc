
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

/*
  Demonstrates dumping matrix/vector from heritage code for PETSc.
   Note does not do bit swapping, so will not generate proper 
PETSc files on Paragon/Dec Alpha.
*/

EXTERN void Store2DArray(int,int,PetscReal*,char*,int *);
EXTERN void Store1DArray(int,PetscReal*,char*,int *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscReal a[100],v[10];
  int       i,j,fd = 0;

  for (i=0; i<100; i++) {
    a[i] = i + 1;
  }
  for (j=0; j<10; j++) {
    v[j] = j;
  }

  Store2DArray(10,10,a,"array.dat",&fd);
  Store1DArray(10,v,"array.dat",&fd);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Store2DArray"
void Store2DArray(int m,int n,PetscReal *a,char *filename,int *fdd)
{
  int        fd = *fdd;
  int        i,j;
  int        nz = -1,cookie = 1211216,ierr;
  PetscReal *vals;

  if (!fd) {
    fd = creat(filename,0666); 
    if (fd == -1) {
      fprintf(stdout,"Unable to open binary file\n");
      exit(0);
    }
    *fdd = fd;
  }
  ierr = write(fd,&cookie,sizeof(PetscInt));
  ierr = write(fd,&m,sizeof(PetscInt));
  ierr = write(fd,&n,sizeof(PetscInt));
  ierr = write(fd,&nz,sizeof(PetscInt));

  /*
     transpose the matrix, since it is stored by rows on the disk
   */
  ierr = PetscMalloc(m*n*sizeof(PetscReal), &vals);CHKERRQ(ierr);
  if (!vals) {
    fprintf(stdout,"Out of memory ");
    exit(0);
  }
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      vals[i+m*j] = a[j+i*n];
    }
  }
  ierr = write(fd,vals,m*n*sizeof(PetscReal));
  ierr = PetscFree(vals);CHKERRQ(ierr);

}

#undef __FUNCT__
#define __FUNCT__ "Store1DArray"
void Store1DArray(int m,PetscReal *a,char *filename,int *fdd)
{
  int  fd = *fdd;
  int  i,j,ierr;
  int  cookie = 1211214; /* cookie for vectors */

  if (fd == -1) {
    fd = creat(filename,0666); 
    if (fd == -1) {
      fprintf(stdout,"Unable to open binary file\n");
      exit(0);
    }
    *fdd = fd;
  }
  ierr = write(fd,&cookie,sizeof(PetscInt));
  ierr = write(fd,&m,sizeof(PetscInt));
  ierr = write(fd,a,m*sizeof(PetscReal));
}


