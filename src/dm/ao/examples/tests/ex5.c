/*$Id: ex5.c,v 1.13 2001/01/15 21:48:46 bsmith Exp balay $*/

static char help[] = "Tests AODataRemap \n\n";

#include "petscao.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int         n,nglobal,bs = 1,*keys,*data,ierr,rank,size,i,start,*news;
  AOData      aodata;
  AO          ao;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr); n = rank + 2;
  ierr = MPI_Allreduce(&n,&nglobal,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /*
       Create a database with one  key and one segment
  */
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata);CHKERRA(ierr);

  /*
       Put one segment in the key
  */
  ierr = AODataKeyAdd(aodata,"key1",PETSC_DECIDE,nglobal);CHKERRA(ierr);

  /* allocate space for the keys each processor will provide */
  ierr = PetscMalloc(n*sizeof(int),&keys);CHKERRA(ierr);

  /*
     We assign the first set of keys (0 to 2) to processor 0, etc.
     This computes the first local key on each processor
  */
  ierr = MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRA(ierr);
  start -= n;

  for (i=0; i<n; i++) {
    keys[i]     = start + i;
  }

  /* 
      Allocate data for the first key and first segment 
  */
  ierr = PetscMalloc(bs*n*sizeof(int),&data);CHKERRA(ierr);
  for (i=0; i<n; i++) {
    data[i]   = start + i + 1; /* the data is the neighbor to the right */
  }
  data[n-1] = 0; /* make it periodic */
  ierr = AODataSegmentAdd(aodata,"key1","key1",bs,n,keys,data,PETSC_INT);CHKERRA(ierr); 
  ierr = PetscFree(data);CHKERRA(ierr);
  ierr = PetscFree(keys);CHKERRA(ierr);

  /*
        View the database
  */
  ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);
 
  /*
         Remap the database so that i -> nglobal - i - 1
  */
  ierr = PetscMalloc(n*sizeof(int),&news);CHKERRA(ierr);
  for (i=0; i<n; i++) {
    news[i] = nglobal - i - start - 1;
  }
  ierr = AOCreateBasic(PETSC_COMM_WORLD,n,news,PETSC_NULL,&ao);CHKERRA(ierr);
  ierr = PetscFree(news);CHKERRA(ierr);
  ierr = AODataKeyRemap(aodata,"key1",ao);CHKERRA(ierr);
  ierr = AODestroy(ao);CHKERRA(ierr);
  ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = AODataDestroy(aodata);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 


