
static char help[] = "Tests AODataRemap(). \n\n";

#include "petscao.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       n,nglobal,bs = 1,*keys,*data,i,start,*news;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  AOData         aodata;
  AO             ao;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr); n = rank + 2;
  ierr = MPI_Allreduce(&n,&nglobal,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /*
       Create a database with one  key and one segment
  */
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata);CHKERRQ(ierr);

  /*
       Put one segment in the key
  */
  ierr = AODataKeyAdd(aodata,"key1",PETSC_DECIDE,nglobal);CHKERRQ(ierr);

  /* allocate space for the keys each processor will provide */
  ierr = PetscMalloc(n*sizeof(PetscInt),&keys);CHKERRQ(ierr);

  /*
     We assign the first set of keys (0 to 2) to processor 0, etc.
     This computes the first local key on each processor
  */
  ierr = MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  start -= n;

  for (i=0; i<n; i++) {
    keys[i]     = start + i;
  }

  /* 
      Allocate data for the first key and first segment 
  */
  ierr = PetscMalloc(bs*n*sizeof(PetscInt),&data);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    data[i]   = start + i + 1; /* the data is the neighbor to the right */
  }
  data[n-1] = 0; /* make it periodic */
  ierr = AODataSegmentAdd(aodata,"key1","key1",bs,n,keys,data,PETSC_INT);CHKERRQ(ierr); 
  ierr = PetscFree(data);CHKERRQ(ierr);
  ierr = PetscFree(keys);CHKERRQ(ierr);

  /*
        View the database
  */
  ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
 
  /*
         Remap the database so that i -> nglobal - i - 1
  */
  ierr = PetscMalloc(n*sizeof(PetscInt),&news);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    news[i] = nglobal - i - start - 1;
  }
  ierr = AOCreateBasic(PETSC_COMM_WORLD,n,news,PETSC_NULL,&ao);CHKERRQ(ierr);
  ierr = PetscFree(news);CHKERRQ(ierr);
  ierr = AODataKeyRemap(aodata,"key1",ao);CHKERRQ(ierr);
  ierr = AODestroy(ao);CHKERRQ(ierr);
  ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = AODataDestroy(aodata);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 


