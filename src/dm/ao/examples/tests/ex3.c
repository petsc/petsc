/*$Id: ex3.c,v 1.21 2001/01/15 21:48:46 bsmith Exp balay $*/

static char help[] = "Tests AOData \n\n";

#include "petscao.h"
#include "petscbt.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int     n = 2,nglobal,bs = 2,*keys,*data,ierr,rank,size,i,start;
  double  *gd;
  AOData  aodata;
  PetscViewer  binary;
  PetscBT ld;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); n = n + rank;CHKERRA(ierr);
  ierr = MPI_Allreduce(&n,&nglobal,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /*
       Create a database with two sets of keys 
  */
  ierr = AODataCreateBasic(PETSC_COMM_WORLD,&aodata);CHKERRA(ierr);
  ierr = AODataKeyAdd(aodata,"key1",PETSC_DECIDE,nglobal);CHKERRA(ierr);
  ierr = AODataKeyAdd(aodata,"key2",PETSC_DECIDE,nglobal);CHKERRA(ierr);

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
    data[2*i]   = -(start + i);
    data[2*i+1] = -(start + i) - 10000;
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg1",bs,n,keys,data,PETSC_INT);CHKERRA(ierr); 
  ierr = PetscFree(data);CHKERRA(ierr);

  /*
      Allocate data for first key and second segment 
  */
  bs   = 3;
  ierr = PetscMalloc(bs*n*sizeof(double),&gd);CHKERRA(ierr);
  for (i=0; i<n; i++) {
    gd[3*i]   = -(start + i);
    gd[3*i+1] = -(start + i) - 10000;
    gd[3*i+2] = -(start + i) - 100000;
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg2",bs,n,keys,gd,PETSC_DOUBLE);CHKERRA(ierr); 

  /*
      Allocate data for first key and third segment 
  */
  bs   = 1;
  ierr = PetscBTCreate(n,ld);CHKERRA(ierr);
  for (i=0; i<n; i++) {
    if (i % 2) PetscBTSet(ld,i);
  }
  ierr = AODataSegmentAdd(aodata,"key1","seg3",bs,n,keys,ld,PETSC_LOGICAL);CHKERRA(ierr); 
  ierr = PetscBTDestroy(ld);CHKERRA(ierr);

  /*
       Use same data for second key and first segment 
  */
  bs   = 3;
  ierr = AODataSegmentAdd(aodata,"key2","seg1",bs,n,keys,gd,PETSC_DOUBLE);CHKERRA(ierr); 
  ierr = PetscFree(gd);CHKERRA(ierr);
  ierr = PetscFree(keys);CHKERRA(ierr);

  ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /*
        Save the database to a file
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"dataoutput",PETSC_BINARY_CREATE,&binary);CHKERRA(ierr);
  ierr = AODataView(aodata,binary);CHKERRA(ierr);
  ierr = PetscViewerDestroy(binary);CHKERRA(ierr);
 
  ierr = AODataDestroy(aodata);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 


