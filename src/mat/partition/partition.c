
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: partition.c,v 1.19 1998/12/21 01:01:19 bsmith Exp bsmith $";
#endif
 

#include "petsc.h"
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
  MPI_Comm_size(part->comm,&size);
  if (part->n != size) {
    SETERRQ(PETSC_ERR_SUP,1,"Currently only supports one domain per processor");
  }
  MPI_Comm_rank(part->comm,&rank);

  ierr = MatGetLocalSize(part->adj,&m,PETSC_NULL); CHKERRQ(ierr);
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
  part->type    = MATPARTITIONING_CURRENT;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
.  name - name of partitioning (for example MATPARTITIONING_CURRENT) or MATPARTITIONING_NEW
.  sname -  corresponding string for name
.  order - routine that does partitioning

   Output Parameters:
.  oname - number associated with the partitioning (for example MATPARTITIONING_CURRENT)

   Not Collective

.keywords: matrix, partitioning, register

.seealso: MatPartitioningRegisterDestroy(), MatPartitioningRegisterAll()
@*/
int MatPartitioningRegister(MatPartitioningType name,MatPartitioningType *oname,char *sname,int (*part)(MatPartitioning))
{
  int         ierr;
  static int  numberregistered = 0;

  PetscFunctionBegin;
  if (!__MatPartitioningList) {
    ierr = NRCreate(&__MatPartitioningList); CHKERRQ(ierr);
  }

  if (name == MATPARTITIONING_NEW) {
    name = (MatPartitioningType) ((int) MATPARTITIONING_NEW + numberregistered++);
  }
  if (oname) *oname = name;
  ierr = NRRegister(__MatPartitioningList,(int)name,sname,(int (*)(void*))part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningRegisterDestroy" 
/*@C
   MatPartitioningRegisterDestroy - Frees the list of partitioning routines.

  Not Collective

.keywords: matrix, register, destroy

.seealso: MatPartitioningRegister(), MatPartitioningRegisterAll()
@*/
int MatPartitioningRegisterDestroy(void)
{
  PetscFunctionBegin;
  if (__MatPartitioningList) {
    NRDestroy( __MatPartitioningList );
    __MatPartitioningList = 0;
  }
  MatPartitioningRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningGetType"
/*@C
   MatPartitioningGetType - Gets the Partitioning method type and name (as a string) 
        from the partitioning context.

   Input Parameter:
.  partitioning - the partitioning context

   Output Parameter:
.  meth - partitioner type (or use PETSC_NULL)
.  name - name of partitioner (or use PETSC_NULL)

   Not Collective

.keywords: Partitioning, get, method, name, type
@*/
int MatPartitioningGetType(MatPartitioning partitioning,MatPartitioningType *meth,char **name)
{
  int ierr;

  PetscFunctionBegin;
  if (!__MatPartitioningList) {ierr = MatPartitioningRegisterAll(); CHKERRQ(ierr);}
  if (meth) *meth = (MatPartitioningType) partitioning->type;
  if (name)  *name = NRFindName( __MatPartitioningList, (int)partitioning->type );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningApply" 
/*@C
   MatPartitioningApply - Gets a partitioning for a matrix.

   Input Parameters:
.  matp - the matrix partitioning object

   Output Parameters:
.   partitioning - the partitioning. For each local node this tells the processor
                   number that that node is assigned to.

   Collective on Mat

   Options Database Keys:
   To specify the partitioning through the options database, use one of
   the following 
$    -mat_partitioning_type parmetis, -mat_partitioning current
   To see the partitioning result
$    -mat_partitioning_view

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

  ierr = OptionsHasName(PETSC_NULL,"-mat_partitioning_view",&flag);
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

   Input Parameters:
.  part - the partitioning context
.  adj - the adjacency matrix

   Collective on MatPartitioning and Mat

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

   Input Parameters:
.  part - the partitioning context

   Collective on Partitioning

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

   Input Parameter:
.   comm - MPI communicator 

   Output Parameter:
.  newp - location to put the context

   Collective on MPI_Comm

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetUp(), MatPartitioningApply(), MatPartitioningDestroy(),
          MatPartitioningSetAdjacency()

@*/
int MatPartitioningCreate(MPI_Comm comm,MatPartitioning *newp)
{
  MatPartitioning     part;
  MatPartitioningType initialtype = MATPARTITIONING_CURRENT;
  int              ierr;

  PetscFunctionBegin;
  *newp          = 0;

  PetscHeaderCreate(part,_p_MatPartitioning,int,MATPARTITIONING_COOKIE,initialtype,"MatPartitioning",comm,MatPartitioningDestroy,
                    MatPartitioningView);
  PLogObjectCreate(part);
  part->type               = -1;
  MPI_Comm_size(comm,&part->n);

  /* this violates rule about seperating abstract from implementions*/
  ierr = MatPartitioningSetType(part,initialtype);CHKERRQ(ierr);
  *newp = part;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningView"
/*@ 
   MatPartitioningView - Prints the partitioning data structure.

   Input Parameters:
.  part - the partitioning context
.  viewer - optional visualization context

   Collective on MatPartitioning unless Viewer is VIEWER_STDOUT_SELF

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
  ViewerType  vtype;
  int         ierr;
  char        *name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MATPARTITIONING_COOKIE);
  if (viewer) {PetscValidHeader(viewer);} 
  else { viewer = VIEWER_STDOUT_SELF;}

  ierr = ViewerGetType(viewer,&vtype);CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ierr = MatPartitioningGetType(part,PETSC_NULL,&name); CHKERRQ(ierr);
    ViewerASCIIPrintf(viewer,"MatPartitioning Object: %s\n",name);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
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

   Input Parameters:
.  part - the partitioning context

   Collective on Partitioning

.keywords: Partitioning, help

.seealso: 
@*/
int MatPartitioningPrintHelp(MatPartitioning  part)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MATPARTITIONING_COOKIE);
  if (!MatPartitioningRegisterAllCalled) {ierr = MatPartitioningRegisterAll(); CHKERRQ(ierr);}

  (*PetscHelpPrintf)(part->comm,"MatPartitioning options ----------------------------------------------\n");
  NRPrintTypes(part->comm,stdout,part->prefix,"mat_partitioning_type",__MatPartitioningList);

  if (part->printhelp) {
    ierr = (*part->printhelp)(part);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningSetType"
/*@
   MatPartitioningSetType - Sets the type of partitioner to use

   Input Parameter:
.  part - the partitioning context.
.  type - a known method

   Collective on MatPartitioning

   Options Database Command:
$  -mat_partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

.keywords: partitioning, set, method, type
@*/
int MatPartitioningSetType(MatPartitioning part,MatPartitioningType type)
{
  int ierr,(*r)(MatPartitioning);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MATPARTITIONING_COOKIE);
  if (part->type == (int) type) PetscFunctionReturn(0);

  if (part->setupcalled) {
    if (part->destroy) ierr =  (*part->destroy)(part);
    else {if (part->data) PetscFree(part->data);}
    part->data        = 0;
    part->setupcalled = 0;
  }
  /* Get the function pointers for the method requested */
  if (!MatPartitioningRegisterAllCalled) {ierr = MatPartitioningRegisterAll(); CHKERRQ(ierr);}
  r =  (int (*)(MatPartitioning))NRFindRoutine( __MatPartitioningList, (int)type, (char *)0 );
  if (!r) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown type");}
  if (part->data) PetscFree(part->data);

  part->destroy      = ( int (*)(MatPartitioning )) 0;
  part->view         = ( int (*)(MatPartitioning,Viewer) ) 0;
  ierr = (*r)(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPartitioningSetFromOptions"
/*@
   MatPartitioningSetFromOptions - Sets various partitioing options from the 
        options database.

   Input Parameter:
.  part - the partitioning context.

   Collective on MatPartitioning

   Options Database Command:
$  -mat_partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

.keywords: partitioning, set, method, type
@*/
int MatPartitioningSetFromOptions(MatPartitioning part)
{
  int              ierr,flag;
  MatPartitioningType type;

  PetscFunctionBegin;

  if (!__MatPartitioningList) {ierr = MatPartitioningRegisterAll();CHKERRQ(ierr);}
  ierr = NRGetTypeFromOptions(part->prefix,"-mat_partitioning_type",__MatPartitioningList,&type,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatPartitioningSetType(part,type);CHKERRQ(ierr);
  }
  if (part->setfromoptions) {
    ierr = (*part->setfromoptions)(part);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help",&flag); CHKERRQ(ierr);
  if (flag) {
    ierr = MatPartitioningPrintHelp(part);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}






