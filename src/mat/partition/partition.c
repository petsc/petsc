/*$Id: partition.c,v 1.36 1999/10/13 20:37:35 bsmith Exp bsmith $*/
 
#include "src/mat/matimpl.h"               /*I "mat.h" I*/

/*
   Simplest partitioning, keeps the current partitioning.
*/
#undef __FUNC__  
#define __FUNC__ "MatPartitioningApply_Current" 
static int MatPartitioningApply_Current(MatPartitioning part, IS *partitioning)
{
  int   ierr,m,rank,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(part->comm,&size);CHKERRQ(ierr);
  if (part->n != size) {
    SETERRQ(PETSC_ERR_SUP,1,"Currently only supports one domain per processor");
  }
  ierr = MPI_Comm_rank(part->comm,&rank);CHKERRQ(ierr);

  ierr = MatGetLocalSize(part->adj,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(part->comm,m,rank,0,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN  
#undef __FUNC__  
#define __FUNC__ "MatPartitioningCreate_Current" 
int MatPartitioningCreate_Current(MatPartitioning part)
{
  PetscFunctionBegin;
  part->apply   = MatPartitioningApply_Current;
  part->view    = 0;
  part->destroy = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ===========================================================================================*/

#include "sys.h"

FList MatPartitioningList = 0;
int   MatPartitioningRegisterAllCalled = 0;

/*MC
   MatPartitioningRegister - Adds a new sparse matrix partitioning to the 
   matrix package. 

   Synopsis:
   MatPartitioningRegister(char *name_partitioning,char *path,char *name_create,int (*routine_create)(MatPartitioning))

   Not Collective

   Input Parameters:
+  sname - name of partitioning (for example MATPARTITIONING_CURRENT) or parmetis
.  path - location of library where creation routine is 
.  name - name of function that creates the partitioning type, a string
-  function - function pointer that creates the partitioning type

   Level: developer

   If dynamic libraries are used, then the fourth input argument (function)
   is ignored.

   Sample usage:
.vb
   MatPartitioningRegister("my_part",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyPartCreate",MyPartCreate);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatPartitioningSetType(part,"my_part")
   or at runtime via the option
$     -mat_partitioning_type my_part

   $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

.keywords: matrix, partitioning, register

.seealso: MatPartitioningRegisterDestroy(), MatPartitioningRegisterAll()
M*/

#undef __FUNC__  
#define __FUNC__ "MatPartitioningRegister_Private" 
int MatPartitioningRegister_Private(char *sname,char *path,char *name,int (*function)(MatPartitioning))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = FListConcat_Private(path,name,fullname); CHKERRQ(ierr);
  ierr = FListAdd_Private(&MatPartitioningList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningRegisterDestroy" 
/*@C
   MatPartitioningRegisterDestroy - Frees the list of partitioning routines.

  Not Collective

  Level: developer

.keywords: matrix, register, destroy

.seealso: MatPartitioningRegister(), MatPartitioningRegisterAll()
@*/
int MatPartitioningRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (MatPartitioningList) {
    ierr = FListDestroy( MatPartitioningList );CHKERRQ(ierr);
    MatPartitioningList = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningGetType"
/*@C
   MatPartitioningGetType - Gets the Partitioning method type and name (as a string) 
        from the partitioning context.

   Not collective

   Input Parameter:
.  partitioning - the partitioning context

   Output Parameter:
.  type - partitioner type

   Level: intermediate

   Not Collective

.keywords: Partitioning, get, method, name, type
@*/
int MatPartitioningGetType(MatPartitioning partitioning,MatPartitioningType *type)
{
  PetscFunctionBegin;
  *type = partitioning->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningApply" 
/*@C
   MatPartitioningApply - Gets a partitioning for a matrix.

   Collective on Mat

   Input Parameters:
.  matp - the matrix partitioning object

   Output Parameters:
.   partitioning - the partitioning. For each local node this tells the processor
                   number that that node is assigned to.

   Options Database Keys:
   To specify the partitioning through the options database, use one of
   the following 
$    -mat_partitioning_type parmetis, -mat_partitioning current
   To see the partitioning result
$    -mat_partitioning_view

   Level: beginner

   The user can define additional partitionings; see MatPartitioningRegister().

.keywords: matrix, get, partitioning

.seealso:  MatPartitioningGetTypeFromOptions(), MatPartitioningRegister()
@*/
int MatPartitioningApply(MatPartitioning matp,IS *partitioning)
{
  int         ierr,flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp,MATPARTITIONING_COOKIE);
  if (!matp->adj->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (matp->adj->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  PLogEventBegin(MAT_Partitioning,matp,0,0,0); 
  ierr = (*matp->apply)(matp,partitioning);CHKERRQ(ierr);
  PLogEventEnd(MAT_Partitioning,matp,0,0,0); 

  ierr = OptionsHasName(PETSC_NULL,"-mat_partitioning_view",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatPartitioningView(matp,VIEWER_STDOUT_(matp->comm));CHKERRQ(ierr);
    ierr = ISView(*partitioning,VIEWER_STDOUT_(matp->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ "MatPartitioningSetAdjacency"
/*@C
   MatPartitioningSetAdjacency - Sets the adjacency graph (matrix) of the thing to be
      partitioned.

   Collective on MatPartitioning and Mat

   Input Parameters:
+  part - the partitioning context
-  adj - the adjacency matrix

   Level: beginner

.keywords: Partitioning, adjacency

.seealso: MatPartitioningCreate()
@*/
int MatPartitioningSetAdjacency(MatPartitioning part,Mat adj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MATPARTITIONING_COOKIE);
  PetscValidHeaderSpecific(adj,MAT_COOKIE);
  part->adj = adj;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningDestroy"
/*@C
   MatPartitioningDestroy - Destroys the partitioning context.

   Collective on Partitioning

   Input Parameters:
.  part - the partitioning context

   Level: beginner

.keywords: Partitioning, destroy, context

.seealso: MatPartitioningCreate()
@*/
int MatPartitioningDestroy(MatPartitioning part)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MATPARTITIONING_COOKIE);
  if (--part->refct > 0) PetscFunctionReturn(0);

  if (part->destroy) {
    ierr = (*part->destroy)(part);CHKERRQ(ierr);
  }
  PLogObjectDestroy(part);
  PetscHeaderDestroy(part);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningCreate"
/*@C
   MatPartitioningCreate - Creates a partitioning context.

   Collective on MPI_Comm

   Input Parameter:
.   comm - MPI communicator 

   Output Parameter:
.  newp - location to put the context

   Level: beginner

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetUp(), MatPartitioningApply(), MatPartitioningDestroy(),
          MatPartitioningSetAdjacency()

@*/
int MatPartitioningCreate(MPI_Comm comm,MatPartitioning *newp)
{
  MatPartitioning part;
  int             ierr;

  PetscFunctionBegin;
  *newp          = 0;

  PetscHeaderCreate(part,_p_MatPartitioning,int,MATPARTITIONING_COOKIE,-1,"MatPartitioning",comm,MatPartitioningDestroy,
                    MatPartitioningView);
  PLogObjectCreate(part);
  part->type               = -1;
  ierr = MPI_Comm_size(comm,&part->n);CHKERRQ(ierr);

  *newp = part;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningView"
/*@ 
   MatPartitioningView - Prints the partitioning data structure.

   Collective on MatPartitioning unless Viewer is VIEWER_STDOUT_SELF

   Input Parameters:
.  part - the partitioning context
.  viewer - optional visualization context

   Level: intermediate

   Note:
   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
-     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open alternative visualization contexts with
.     ViewerASCIIOpen() - output to a specified file

.keywords: Partitioning, view

.seealso: ViewerASCIIOpen()
@*/
int MatPartitioningView(MatPartitioning  part,Viewer viewer)
{
  int                 ierr;
  PetscTruth          isascii;
  MatPartitioningType name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MATPARTITIONING_COOKIE);
  if (!viewer) viewer = VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscCheckSameComm(part,viewer);

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = MatPartitioningGetType(part,&name);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"MatPartitioning Object: %s\n",name);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for this MatParitioning",((PetscObject)viewer)->type_name);
  }

  if (part->view) {
    ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*part->view)(part,viewer);CHKERRQ(ierr);
    ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningPrintHelp"
/*@ 
   MatPartitioningPrintHelp - Prints all options to the partitioning object.

   Collective on MatPartitioning

   Input Parameters:
.  part - the partitioning context

   Level: intermediate

.keywords: Partitioning, help

.seealso: 
@*/
int MatPartitioningPrintHelp(MatPartitioning  part)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MATPARTITIONING_COOKIE);

  if (!MatPartitioningRegisterAllCalled){ ierr = MatPartitioningRegisterAll(0);CHKERRQ(ierr);}
  ierr = (*PetscHelpPrintf)(part->comm,"MatPartitioning options ----------------------------------------------\n");CHKERRQ(ierr);
  ierr = FListPrintTypes(part->comm,stdout,part->prefix,"mat_partioning_type",MatPartitioningList);CHKERRQ(ierr);CHKERRQ(ierr);

  if (part->printhelp) {
    ierr = (*part->printhelp)(part);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningSetType"
/*@C
   MatPartitioningSetType - Sets the type of partitioner to use

   Collective on MatPartitioning

   Input Parameter:
.  part - the partitioning context.
.  type - a known method

   Options Database Command:
$  -mat_partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

   Level: intermediate

.keywords: partitioning, set, method, type

.seealso: MatPartitioningCreate(), MatPartitioningApply()

@*/
int MatPartitioningSetType(MatPartitioning part,MatPartitioningType type)
{
  int        ierr,(*r)(MatPartitioning);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MATPARTITIONING_COOKIE);
  PetscValidCharPointer(type);

  ierr = PetscTypeCompare((PetscObject)part,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (part->setupcalled) {
    ierr =  (*part->destroy)(part);CHKERRQ(ierr);
    part->data        = 0;
    part->setupcalled = 0;
  }

  /* Get the function pointers for the method requested */
  if (!MatPartitioningRegisterAllCalled){ ierr = MatPartitioningRegisterAll(0);CHKERRQ(ierr);}
  ierr =  FListFind(part->comm, MatPartitioningList, type,(int (**)(void *)) &r );CHKERRQ(ierr);

  if (!r) {SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown partitioning type %s",type);}

  part->destroy      = ( int (*)(MatPartitioning )) 0;
  part->view         = ( int (*)(MatPartitioning,Viewer) ) 0;
  ierr = (*r)(part);CHKERRQ(ierr);

  if (part->type_name) {ierr = PetscFree(part->type_name);CHKERRQ(ierr);}
  ierr = PetscStrallocpy(type,&part->type_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningSetFromOptions"
/*@
   MatPartitioningSetFromOptions - Sets various partitioning options from the 
        options database.

   Collective on MatPartitioning

   Input Parameter:
.  part - the partitioning context.

   Options Database Command:
$  -mat_partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

   Level: beginner

.keywords: partitioning, set, method, type
@*/
int MatPartitioningSetFromOptions(MatPartitioning part)
{
  int  ierr,flag;
  char type[256];

  PetscFunctionBegin;

  ierr = OptionsGetString(part->prefix,"-mat_partitioning_type",type,256,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatPartitioningSetType(part,type);CHKERRQ(ierr);
  }
  /*
    Set the type if it was never set.
  */
  if (!part->type_name) {
    ierr = MatPartitioningSetType(part,MATPARTITIONING_CURRENT);CHKERRQ(ierr);
  }

  if (part->setfromoptions) {
    ierr = (*part->setfromoptions)(part);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatPartitioningPrintHelp(part);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}






