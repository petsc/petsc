
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: partition.c,v 1.1 1997/09/24 20:32:41 bsmith Exp bsmith $";
#endif
 

#include "petsc.h"
#include "src/mat/matimpl.h"


/*
   Simplest partitioning, keeps the current partitioning.
*/
#undef __FUNC__  
#define __FUNC__ "MatPartitioning_Current" 
int MatPartitioning_Current(Mat mat,MatPartitioning color, int nu,ISPartitioning *partitioning)
{
  int   ierr,i,m,rank,*locals,size;

  MPI_Comm_size(mat->comm,&size);
  if (nu != size) {
    SETERRQ(1,1,"Currently only support one domain per processor");
  }

  MPI_Comm_rank(mat->comm,&rank);


  ierr = MatGetLocalSize(mat,&m,PETSC_NULL); CHKERRQ(ierr);
  locals = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(locals);
  for ( i=0; i<m; i++) locals[i] = rank;

  ierr = ISPartitioningCreate(mat->comm,m,locals,partitioning); CHKERRQ(ierr);
  PetscFree(locals);

  return 0;
}
  
/* ===========================================================================================*/

#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__MatPartitioningList = 0;
int MatPartitioningRegisterAllCalled = 0;

#undef __FUNC__  
#define __FUNC__ "MatPartitioningRegister" 
/*@C
   MatPartitioningRegister - Adds a new sparse matrix partitioning to the 
   matrix package. 

   Input Parameters:
.  name - name of partitioning (for example PARTITIONING_CURRENT) or PARTITIONING_NEW
.  sname -  corresponding string for name
.  order - routine that does partitioning

   Output Parameters:
.  oname - number associated with the partitioning (for example PARTITIONING_CURRENT)

.keywords: matrix, partitioning, register

.seealso: MatPartitioningRegisterDestroy(), MatPartitioningRegisterAll()
@*/
int MatPartitioningRegister(MatPartitioning name,MatPartitioning *oname,char *sname,
                            int (*part)(Mat,MatPartitioning,int,ISPartitioning*))
{
  int         ierr;
  static int  numberregistered = 0;

  if (!__MatPartitioningList) {
    ierr = NRCreate(&__MatPartitioningList); CHKERRQ(ierr);
  }

  if (name == PARTITIONING_NEW) name = (MatPartitioning) ((int) PARTITIONING_NEW + numberregistered++);
  if (oname) *oname = name;
  ierr = NRRegister(__MatPartitioningList,(int)name,sname,(int (*)(void*))part);CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningRegisterDestroy" 
/*@C
   MatPartitioningRegisterDestroy - Frees the list of partitioning routines.

.keywords: matrix, register, destroy

.seealso: MatPartitioningRegister(), MatPartitioningRegisterAll()
@*/
int MatPartitioningRegisterDestroy()
{
  if (__MatPartitioningList) {
    NRDestroy( __MatPartitioningList );
    __MatPartitioningList = 0;
  }
  MatPartitioningRegisterAllCalled = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetPartitioningTypeFromOptions" 
/*@C
   MatGetPartitioningTypeFromOptions - Gets matrix partitioning method from the
   options database.

   Input Parameter:
.  prefix - optional database prefix

   Output Parameter:
.  type - partitioning method

   Options Database Keys:
   To specify the partitioninging through the options database, use one of
   the following 
$    -mat_partitioning current

.keywords: matrix, partitioning, 

.seealso: MatGetPartitioning()
@*/
int MatGetPartitioningTypeFromOptions(char *prefix,MatPartitioning *type)
{
  char sbuf[50];
  int  ierr,flg;
  
  ierr = OptionsGetString(prefix,"-mat_partitioning", sbuf, 50,&flg); CHKERRQ(ierr);
  if (flg) {
    if (!__MatPartitioningList) MatPartitioningRegisterAll();
    *type = (MatPartitioning)NRFindID( __MatPartitioningList, sbuf );
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningGetName" 
/*@C
   MatPartitioningGetName - Gets the name associated with a partitioning.

   Input Parameter:
.  partitioninging - integer name of partitioning

   Output Parameter:
.  name - name of partitioning

.keywords: matrix, get, partitioning, name
@*/
int MatPartitioningGetName(MatPartitioning meth,char **name)
{
  int ierr;
  if (!__MatPartitioningList) {ierr = MatPartitioningRegisterAll(); CHKERRQ(ierr);}
   *name = NRFindName( __MatPartitioningList, (int)meth );
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatGetPartitioning" 
/*@C
   MatGetPartitioning - Gets a partitioning for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Input Parameters:
.  mat - the matrix
.  type - type of partitioning, one of the following:
$      PARTITIONING_NATURAL - natural
$      PARTITIONING_CURRENT - 

   p - total number of partitions (currently we only support using the size 
       of the communicator used in generating the matrix) or use PETSC_DEFAULT

   Output Parameters:
.   partitioning - the partitioning

   Options Database Keys:
   To specify the partitioning through the options database, use one of
   the following 
$    -mat_partitioning natural, -mat_partitioning current
   To see the partitioning result
$    -mat_partitioning_view

   The user can define additional partitionings; see MatPartitioningRegister().

.keywords: matrix, get, partitioning

.seealso:  MatGetPartitioningTypeFromOptions(), MatPartitioningRegister()
@*/
int MatGetPartitioning(Mat mat,MatPartitioning type,int p,ISPartitioning *partitioning)
{
  int         ierr,flag;
  int         (*r)(Mat,MatPartitioning,int,ISPartitioning *);

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(1,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(1,0,"Not for factored matrix"); 
  if (!MatPartitioningRegisterAllCalled) {
    ierr = MatPartitioningRegisterAll();CHKERRQ(ierr);
  }

  if (p == PETSC_DEFAULT) {
    MPI_Comm_size(mat->comm,&p);
  }

  ierr = MatGetPartitioningTypeFromOptions(0,&type); CHKERRQ(ierr);
  /*  PLogEventBegin(MAT_GetPartitioning,mat,0,0,0); */
  r =  (int (*)(Mat,MatPartitioning,int,ISPartitioning*))
                                  NRFindRoutine(__MatPartitioningList,(int)type,(char *)0);
  if (!r) {SETERRQ(1,0,"Unknown or unregistered type");}
  ierr = (*r)(mat,type,p,partitioning); CHKERRQ(ierr);
  /* PLogEventEnd(MAT_GetPartitioning,mat,0,0,0); */

  ierr = OptionsHasName(PETSC_NULL,"-mat_partitioning_view",&flag);
  if (flag) {
    ierr = ISPartitioningView(*partitioning,VIEWER_STDOUT_((*partitioning)->comm));CHKERRQ(ierr);
  }
  return 0;
}
 
