
static char help[] = "Tests removing entries from an AOData. \n\n";

#include "petscao.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int         n,nglobal,bs = 2,*keys,*data,ierr,rank,size,i,start;
  PetscReal   *gd;
  AOData      aodata;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr); n = rank + 2;
  ierr = MPI_Allreduce(&n,&nglobal,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /*
       Create a database with two sets of keys 
  */
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata);CHKERRQ(ierr);

  /*
       Put two segments in the first key and one in the second
  */
  ierr = AODataKeyAdd(aodata,"key1",PETSC_DECIDE,nglobal);CHKERRQ(ierr);
  ierr = AODataKeyAdd(aodata,"key2",PETSC_DECIDE,nglobal);CHKERRQ(ierr);

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
    data[2*i]   = -(start + i);
    data[2*i+1] = -(start + i) - 10000;
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg1",bs,n,keys,data,PETSC_INT);CHKERRQ(ierr); 
  ierr = PetscFree(data);CHKERRQ(ierr);

  /*
      Allocate data for first key and second segment 
  */
  bs   = 3;
  ierr = PetscMalloc(bs*n*sizeof(PetscReal),&gd);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    gd[3*i]   = -(start + i);
    gd[3*i+1] = -(start + i) - 10000;
    gd[3*i+2] = -(start + i) - 100000;
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg2",bs,n,keys,gd,PETSC_REAL);CHKERRQ(ierr); 

  /*
       Use same data for second key and first segment 
  */
  ierr = AODataSegmentAdd(aodata,"key2","seg1",bs,n,keys,gd,PETSC_REAL);CHKERRQ(ierr); 
  ierr = PetscFree(gd);CHKERRQ(ierr);
  ierr = PetscFree(keys);CHKERRQ(ierr);

  /*
     View the database
  */
  ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
       Remove a key and a single segment from the database
  */ 
  ierr = AODataKeyRemove(aodata,"key2");CHKERRQ(ierr); 
  ierr = AODataSegmentRemove(aodata,"key1","seg1");CHKERRQ(ierr);

  ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = AODataDestroy(aodata);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 


