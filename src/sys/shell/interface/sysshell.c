#include <petscsys.h>

/* FIX: is it okay to include this directly? */
#include <ctype.h>

PetscClassId PETSC_SHELL_CLASSID;
static char PETSC_SHELL_CLASS_NAME[] = "PetscShell";
static PetscBool  PetscShellPackageInitialized = PETSC_FALSE;


typedef PetscErrorCode (*PetscShellPythonLoadVTableFunction)(PetscShell shell, const char* path, const char* name, void **vtable);
typedef PetscErrorCode (*PetscShellPythonClearVTableFunction)(PetscShell shell, void **vtable);
typedef PetscErrorCode (*PetscShellPythonCallFunction)(PetscShell shell, const char* message, void *vtable);

EXTERN_C_BEGIN
PetscShellPythonLoadVTableFunction      PetscShellPythonLoadVTable      = PETSC_NULL;
PetscShellPythonClearVTableFunction     PetscShellPythonClearVTable     = PETSC_NULL;
PetscShellPythonCallFunction            PetscShellPythonCall            = PETSC_NULL;
EXTERN_C_END

#define PETSC_SHELL_CHECKINIT_PYTHON()					\
  if(PetscShellPythonLoadVTable == PETSC_NULL) {		        	\
    PetscErrorCode ierr;						\
    ierr = PetscPythonInitialize(PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);	\
    if(PetscShellPythonLoadVTable == PETSC_NULL) {			        \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,				\
	      "Couldn't initialize Python support for PetscShell");	\
    }									\
  }									
  
#define PETSC_SHELL_LOAD_VTABLE_PYTHON(shell, path, name)                   \
  PETSC_SHELL_CHECKINIT_PYTHON();						\
  {									\
    PetscErrorCode ierr;                                                \
    ierr = PetscShellPythonLoadVTable(shell, path, name, &(shell->vtable));   \
    if (ierr) { PetscPythonPrintError(); SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, "Python error"); } \
  }

#define PETSC_SHELL_CLEAR_VTABLE_PYTHON(shell)                              \
  PETSC_SHELL_CHECKINIT_PYTHON();						\
  {									\
    PetscErrorCode ierr;                                                \
    ierr = PetscShellPythonClearVTable(shell, &(shell->vtable));              \
    if (ierr) { PetscPythonPrintError(); SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, "Python error"); } \
  }
  
#define PETSC_SHELL_CALL_PYTHON(shell, message)                             \
  PETSC_SHELL_CHECKINIT_PYTHON();                                         \
  {									\
    PetscErrorCode ierr;                                                \
    ierr = PetscShellPythonCall(shell, message, shell->vtable);                           \
    if (ierr) { PetscPythonPrintError(); SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB, "Python error"); } \
  }
  

/* ---------------------------------------------------------------------------------------------- */
struct _n_PetscShellGraph {
  PetscInt vcount, vmax; /* actual and allocated number of vertices */
  PetscInt *i, *j, *outdegree; /* (A)IJ structure for the underlying matrix: 
                                  i[row]             the row offset into j, 
                                  i[row+1] - i[row]  allocated number of entries for row,
                                  outdegree[row]     actual number of entries in row
                               */
  PetscInt *indegree;
  PetscInt nz, maxnz;
  PetscInt rowreallocs, colreallocs;
};

typedef struct _n_PetscShellGraph *PetscShellGraph;

#define CHUNKSIZE 5
/*
    Inserts the (row,col) entry, allocating larger arrays, if necessary. 
    Does NOT check whether row and col are within range (< graph->vcount).
    Does NOT check whether the entry is already present.
*/
#undef  __FUNCT__
#define __FUNCT__ "PetscShellGraphExpandRow_Private"
PetscErrorCode PetscShellGraphExpandRow_Private(PetscShellGraph graph, PetscInt row) 
{
  PetscErrorCode ierr;
  PetscInt       rowlen, rowmax, rowoffset;
  PetscInt       ii;

  PetscFunctionBegin;
  rowlen = graph->outdegree[row]; 
  rowmax = graph->i[row+1] - graph->i[row]; 
  rowoffset = graph->i[row];
  if (rowlen >= rowmax) {
    /* there is no extra room in row, therefore enlarge */              
    PetscInt   new_nz = graph->i[graph->vcount] + CHUNKSIZE;  
    PetscInt   *new_i=0,*new_j=0;                            
    
    /* malloc new storage space */ 
    ierr = PetscMalloc(new_nz*sizeof(PetscInt),&new_j); CHKERRQ(ierr);
    ierr = PetscMalloc((graph->vmax+1)*sizeof(PetscInt),&new_i);CHKERRQ(ierr);
    
    /* copy over old data into new slots */ 
    for (ii=0; ii<row+1; ii++) {new_i[ii] = graph->i[ii];} 
    for (ii=row+1; ii<graph->vmax+1; ii++) {new_i[ii] = graph->i[ii]+CHUNKSIZE;} 
    ierr = PetscMemcpy(new_j,graph->j,(rowoffset+rowlen)*sizeof(PetscInt));CHKERRQ(ierr); 
    ierr = PetscMemcpy(new_j+rowoffset+rowlen+CHUNKSIZE,graph->j+rowoffset+rowlen,(new_nz - CHUNKSIZE - rowoffset - rowlen)*sizeof(PetscInt));CHKERRQ(ierr); 
    /* free up old matrix storage */ 
    ierr = PetscFree(graph->j);CHKERRQ(ierr);  
    ierr = PetscFree(graph->i);CHKERRQ(ierr);    
    graph->i = new_i; graph->j = new_j;  
    graph->maxnz     += CHUNKSIZE; 
    graph->colreallocs++; 
  } 
  PetscFunctionReturn(0);
}/* PetscShellGraphExpandRow_Private() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellGraphAddVertex"
PetscErrorCode PetscShellGraphAddVertex(PetscShellGraph graph, PetscInt *v) {
  PetscInt ii;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(graph->vcount >= graph->vmax) {
    /* Add rows */
    PetscInt   *new_i=0, *new_outdegree=0, *new_indegree;                            
    
    /* malloc new storage space */ 
    ierr = PetscMalloc((graph->vmax+CHUNKSIZE+1)*sizeof(PetscInt),&new_i);CHKERRQ(ierr);
    ierr = PetscMalloc((graph->vmax+CHUNKSIZE)*sizeof(PetscInt),&new_outdegree); CHKERRQ(ierr);
    ierr = PetscMalloc((graph->vmax+CHUNKSIZE)*sizeof(PetscInt),&new_indegree); CHKERRQ(ierr);
    ierr = PetscMemzero(new_outdegree, (graph->vmax+CHUNKSIZE)*sizeof(PetscInt)); CHKERRQ(ierr);
    ierr = PetscMemzero(new_indegree, (graph->vmax+CHUNKSIZE)*sizeof(PetscInt)); CHKERRQ(ierr);


    /* copy over old data into new slots */ 
    ierr = PetscMemcpy(new_i,graph->i,(graph->vmax+1)*sizeof(PetscInt));CHKERRQ(ierr); 
    ierr = PetscMemcpy(new_outdegree,graph->outdegree,(graph->vmax)*sizeof(PetscInt));CHKERRQ(ierr); 
    ierr = PetscMemcpy(new_indegree,graph->indegree,(graph->vmax)*sizeof(PetscInt));CHKERRQ(ierr); 
    for (ii=graph->vmax+1; ii<=graph->vmax+CHUNKSIZE; ++ii) {
      new_i[ii] = graph->i[graph->vmax];
    }

    /* free up old matrix storage */ 
    ierr = PetscFree(graph->i);CHKERRQ(ierr);  
    ierr = PetscFree(graph->outdegree);CHKERRQ(ierr);    
    ierr = PetscFree(graph->indegree);CHKERRQ(ierr);    
    graph->i = new_i; graph->outdegree = new_outdegree; graph->indegree = new_indegree;
    
    graph->vmax += CHUNKSIZE; 
    graph->rowreallocs++; 
  }
  if(v) {*v = graph->vcount;}
  ++(graph->vcount);
  PetscFunctionReturn(0);
}/* PetscShellGraphAddVertex() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellGraphAddEdge"
PetscErrorCode PetscShellGraphAddEdge(PetscShellGraph graph, PetscInt row, PetscInt col) {
  PetscErrorCode        ierr;
  PetscInt              *rp,low,high,t,ii,i;
  PetscFunctionBegin;

  if(row < 0 || row >= graph->vcount) {
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Source vertex %D out of range: min %D max %D",row, 0, graph->vcount);
  }
  if(col < 0 || col >= graph->vcount) {
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Target vertex %D out of range: min %D max %D",col, 0, graph->vcount);
  }
  rp   = graph->j + graph->i[row];
  low  = 0;
  high = graph->outdegree[row];
  while (high-low > 5) {
    t = (low+high)/2;
    if (rp[t] > col) high = t;
    else             low  = t;
  }
  for (i=low; i<high; ++i) {
    if (rp[i] > col) break;
    if (rp[i] == col) {
      goto we_are_done;
    }
  } 
  ierr = PetscShellGraphExpandRow_Private(graph, row); CHKERRQ(ierr);
  /* 
     If the graph was empty before, graph->j was NULL and rp was NULL as well.  
     Now that the row has been expanded, rp needs to be reset. 
  */
  rp = graph->j + graph->i[row];
  /* shift up all the later entries in this row */
  for (ii=graph->outdegree[row]; ii>=i; --ii) {
    rp[ii+1] = rp[ii];
  }
  rp[i] = col; 
  ++(graph->outdegree[row]);
  ++(graph->indegree[col]);
  ++(graph->nz);
  
 we_are_done:
  PetscFunctionReturn(0);
}/* PetscShellGraphAddEdge() */


#undef  __FUNCT__
#define __FUNCT__ "PetscShellGraphTopologicalSort"
PetscErrorCode PetscShellGraphTopologicalSort(PetscShellGraph graph, PetscInt *n, PetscInt **queue) {
  PetscBool  *queued;
  PetscInt   *indegree;
  PetscInt ii, k, jj, Nqueued = 0;
  PetscBool  progress;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(!n || !queue) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid return argument pointers n or vertices");
  }
  *n = graph->vcount;
  ierr = PetscMalloc(sizeof(PetscInt)*graph->vcount, queue); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscBool)*graph->vcount, &queued); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*graph->vcount, &indegree); CHKERRQ(ierr);
  for(ii = 0; ii < graph->vcount; ++ii) {
    queued[ii]   = PETSC_FALSE;
    indegree[ii] = graph->indegree[ii];
  }
  while(Nqueued < graph->vcount) {
    progress = PETSC_FALSE;
    for(ii = 0; ii < graph->vcount; ++ii) {
      /* If ii is not queued yet, and the indegree is 0, queue it. */ 
      if(!queued[ii] && !indegree[ii]) {
        (*queue)[Nqueued] = ii;
        queued[ii] = PETSC_TRUE;
        ++Nqueued;
        progress = PETSC_TRUE;
        /* Reduce the indegree of all vertices in row ii */
        for(k = 0; k < graph->outdegree[ii]; ++k) {
          jj = graph->j[graph->i[ii]+k];
          --(indegree[jj]);
          /* 
             It probably would be more efficient to make a recursive call to the body of the for loop 
             with the jj in place of ii, but we use a simple-minded algorithm instead, since the graphs
             we anticipate encountering are tiny. 
          */
        }/* for(k) */
      }/* if(!queued) */
    }/* for(ii) */
    /* If no progress was made during this iteration, the graph must have a cycle */
    if(!progress) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Cycle detected in the dependency graph");
    }
  }/* while(Nqueued) */
  ierr = PetscFree(queued); CHKERRQ(ierr);
  ierr = PetscFree(indegree); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellGraphTopologicalSort() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellGraphDestroy"
PetscErrorCode PetscShellGraphDestroy(PetscShellGraph graph) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(graph->i);        CHKERRQ(ierr);
  ierr = PetscFree(graph->j);        CHKERRQ(ierr);
  ierr = PetscFree(graph->outdegree);     CHKERRQ(ierr);
  ierr = PetscFree(graph->indegree); CHKERRQ(ierr);
  ierr = PetscFree(graph);           CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellGraphDestroy() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellGraphCreate"
PetscErrorCode PetscShellGraphCreate(PetscShellGraph *graph_p) 
{
  PetscShellGraph graph;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscShellGraph, graph_p);   CHKERRQ(ierr);
  graph = *graph_p;
  graph->vcount = graph->vmax = graph->nz = graph->maxnz = graph->rowreallocs = graph->colreallocs = 0;
  ierr = PetscMalloc(sizeof(PetscInt), &(graph->i)); CHKERRQ(ierr);
  graph->j = graph->outdegree = graph->indegree = PETSC_NULL;
  PetscFunctionReturn(0);
}/* PetscShellGraphCreate() */



/* ------------------------------------------------------------------------------------------------------- */

typedef enum{PETSC_SHELL_VTABLE_NONE, PETSC_SHELL_VTABLE_SO, PETSC_SHELL_VTABLE_PY} PetscShellVTableType;

struct _n_PetscShellVTable_SO {
  char           *path, *name;
};


struct _p_PetscShell {
  PETSCHEADER(int);
  PetscShellVTableType      vtable_type;
  void                      *vtable;
  char *                    url;
  PetscShell                visitor;
  PetscInt                  N, maxN;
  char **                   key;
  PetscShell                *component;
  PetscShellGraph           dep_graph;
};




/* ------------------------------------------------------------------------------------------------------- */

typedef PetscErrorCode (*PetscShellCallFunction)(PetscShell, const char*);
typedef PetscErrorCode (*PetscShellMessageFunction)(PetscShell);
typedef void (*QueryFunction)(void);

static PetscDLLibrary PetscShellDLLibrariesLoaded = 0;

#undef  __FUNCT__
#define __FUNCT__ "PetscShellCall_SO"
PetscErrorCode PetscShellCall_SO(PetscShell shell, const char* path, const char* name, const char* message) {
  size_t    namelen, messagelen, msgfunclen, callfunclen;
  char *msgfunc  = PETSC_NULL, *callfunc = PETSC_NULL;
  PetscShellCallFunction call = PETSC_NULL;
  PetscShellMessageFunction msg = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscStrlen(name, &namelen); CHKERRQ(ierr);
  ierr = PetscStrlen(message, &messagelen); CHKERRQ(ierr);
  msgfunclen = namelen + messagelen;
  ierr = PetscMalloc(sizeof(char)*(msgfunclen+1), &msgfunc); CHKERRQ(ierr);
  msgfunc[0] = '\0';
  if(namelen){
    ierr = PetscStrcat(msgfunc, name); CHKERRQ(ierr);
  }
  ierr = PetscStrcat(msgfunc, message); CHKERRQ(ierr);
  if(namelen) {
    /* HACK: is 'toupper' part of the C standard? Looks like starting with C89. */
    msgfunc[namelen] = toupper(msgfunc[namelen]);
  }
  ierr = PetscDLLibrarySym(((PetscObject)shell)->comm, &PetscShellDLLibrariesLoaded, path, msgfunc, (void **)(&msg)); CHKERRQ(ierr);
  ierr = PetscFree(msgfunc); CHKERRQ(ierr);
  if(msg) {
    ierr = (*msg)(shell); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  callfunclen        = namelen+4;
  ierr = PetscMalloc(sizeof(char)*(callfunclen+1), &callfunc); CHKERRQ(ierr);
  if(namelen) {
    ierr = PetscStrcpy(callfunc, name); CHKERRQ(ierr);
  }
  if(namelen){
    ierr = PetscStrcat(callfunc, "Call"); CHKERRQ(ierr);
  }
  else {
    ierr = PetscStrcat(callfunc, "call"); CHKERRQ(ierr);
  }
  ierr = PetscDLLibrarySym(((PetscObject)shell)->comm, &PetscShellDLLibrariesLoaded, path, callfunc, (void**)(&call)); CHKERRQ(ierr);
  ierr = PetscFree(callfunc); CHKERRQ(ierr);
  if(call) {
    ierr = (*call)(shell, message); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }  
  SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscShell '%s' cannot execute '%s'", ((PetscObject)shell)->name, message);
  PetscFunctionReturn(0);
}/* PetscShellCall_SO() */


#undef  __FUNCT__
#define __FUNCT__ "PetscShellCall_NONE"
PetscErrorCode PetscShellCall_NONE(PetscShell shell, const char* message) {
  PetscShellCallFunction call = PETSC_NULL;
  PetscShellMessageFunction msg = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFListFind(((PetscObject)shell)->qlist, ((PetscObject)shell)->comm, message,PETSC_FALSE, (QueryFunction*)(&msg)); CHKERRQ(ierr);
  if(msg) {
    ierr = (*msg)(shell); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscFListFind(((PetscObject)shell)->qlist, ((PetscObject)shell)->comm, "call",PETSC_FALSE, (QueryFunction*)(&call)); CHKERRQ(ierr);
  if(call) {
    ierr = (*call)(shell, message); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscShell '%s' cannot execute '%s'", ((PetscObject)shell)->name, message);
  PetscFunctionReturn(0);
}/* PetscShellCall_NONE() */


#undef  __FUNCT__
#define __FUNCT__ "PetscShellCall"
/*@C
   PetscShellCall -- send a string message to a PetscShell object.  

   Logically collective on PetscShell.

   Input paramters:
+  shell     -- a PetscShell object
-  message -- a character string

   Notes: In response to the message the object performs actions defined by its URL (see PetscShellSetURL()). 
          Side effects may include, in particular, actions on the composed objects (see PetscObjectCompose()).

  Level: intermediate.

.seealso: PetscShellSetURL(), PetscShellGetURL(), PetscObjectCompose(), PetscObjectQuery()
@*/
PetscErrorCode PetscShellCall(PetscShell shell, const char* message) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidCharPointer(message,2);
  if(!message || !message[0]) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Null or empty message string");
  }
  switch(shell->vtable_type) {
  case PETSC_SHELL_VTABLE_NONE:
    ierr = PetscShellCall_NONE(shell, message); CHKERRQ(ierr);
    break;
  case PETSC_SHELL_VTABLE_SO:
    ierr = PetscShellCall_SO(shell, 
                           ((struct _n_PetscShellVTable_SO*)shell->vtable)->path, 
                           ((struct _n_PetscShellVTable_SO*)shell->vtable)->name, 
                           message); 
    CHKERRQ(ierr);
    break;
  case PETSC_SHELL_VTABLE_PY:
    PETSC_SHELL_CALL_PYTHON(shell, message);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown PetscShellVTableType value");
    break;
  }
  PetscFunctionReturn(0);
}/* PetscShellCall() */


#define PETSC_SHELL_MAX_URL_LENGTH 1024
/* 
   Normalize the url (by truncating to PETSC_SHELL_MAX_URL_LENGTH) and parse it to find out the component type and location.
   Warning: the returned char pointers are borrowed and their contents must be copied elsewhere to be preserved.
*/
#undef  __FUNCT__
#define __FUNCT__ "PetscShellParseURL_Private"
PetscErrorCode  PetscShellParseURL_Private(const char inurl[], char **outpath, char **outname, PetscShellVTableType *outtype){
  char *n, *s;
  static PetscInt nlen = PETSC_SHELL_MAX_URL_LENGTH;
  static char path[PETSC_SHELL_MAX_URL_LENGTH+1], name[PETSC_SHELL_MAX_URL_LENGTH+1];
  PetscShellVTableType type = PETSC_SHELL_VTABLE_NONE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* FIX: this routine should replace the filesystem path by an abolute path for real normalization */
  /* Copy the inurl so we can manipulate it inplace and also truncate to the max allowable length */
  ierr = PetscStrncpy(path, inurl, nlen); CHKERRQ(ierr);  
  /* Split url <path>:<name> into <path> and <name> */
  ierr = PetscStrrchr(path,':',&n); CHKERRQ(ierr);
  /* Make sure it's not the ":/" of the "://" separator */
  if(!n[0] || n[0] == '/') {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, 
           "Could not locate component name within the URL.\n"
           "Must have url = [<path/><library>:]<name>.\n"
           "Instead got %s\n"
           "Remember that URL is always truncated to the max allowed length of %d", 
           inurl, nlen);
  }
  /* Copy n to name */
  ierr = PetscStrcpy(name, n); CHKERRQ(ierr);
  /* If n isn't the whole path (i.e., there is a ':' separator), end 'path' right before the located ':' */
  if(n == path) {
    /* 
       No library is provided, so the component is assumed to be "local", that is
       defined in an already loaded executable. So we set type to .so, path to "",
       and look for the configure symbol among already loaded symbols 
       (or count on PetscDLXXX to do that.
    */
    type = PETSC_SHELL_VTABLE_SO;
    path[0] = '\0';
  }
  else {
    n[-1] = '\0';
    /* Find the library suffix and determine the component library type: .so or .py */
    ierr = PetscStrrchr(path,'.',&s);CHKERRQ(ierr);
    /* FIX: we should really be using PETSc's internally defined suffices */
    if(s != path && s[-1] == '.') {
      if((s[0] == 'a' && s[1] == '\0') || (s[0] == 's' && s[1] == 'o' && s[2] == '\0')){
        type = PETSC_SHELL_VTABLE_SO;
      }
      else if (s[0] == 'p' && s[1] == 'y' && s[2] == '\0'){
        type = PETSC_SHELL_VTABLE_PY;
      }
      else {
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, 
                 "Unknown library suffix within the URL.\n"
                 "Must have url = [<path/><library>:]<name>,\n"
                 "where library = <libname>.<suffix>, suffix = .a || .so || .py.\n"
                 "Instead got url %s\n"
                 "Remember that URL is always truncated to the max allowed length of %d", 
                 inurl, s,nlen);     
      }
    }
  }
  *outpath = path;
  *outname = name;
  *outtype = type;
  PetscFunctionReturn(0);
}/* PetscShellParseURL_Private() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellClearURL_Private"
PetscErrorCode  PetscShellClearURL_Private(PetscShell shell) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  switch(shell->vtable_type) {
  case PETSC_SHELL_VTABLE_SO:
    {
      struct _n_PetscShellVTable_SO *vt = (struct _n_PetscShellVTable_SO*)(shell->vtable);
      ierr = PetscFree(vt->path); CHKERRQ(ierr);
      ierr = PetscFree(vt->name); CHKERRQ(ierr);
      ierr = PetscFree(vt);       CHKERRQ(ierr);
      shell->vtable = PETSC_NULL;
      shell->vtable_type = PETSC_SHELL_VTABLE_NONE;
    }
    break;
  case PETSC_SHELL_VTABLE_PY:
    PETSC_SHELL_CLEAR_VTABLE_PYTHON(shell);
    break;
  case PETSC_SHELL_VTABLE_NONE:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, 
             "Unknown PetscShell vtable type: %d", shell->vtable_type);
  }
  ierr = PetscFree(shell->url);  CHKERRQ(ierr);
  shell->url = PETSC_NULL;
  PetscFunctionReturn(0);
}/* PetscShellClearURL_Private() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellSetURL"
/*@C
   PetscShellSetURL -- set a backend implementing PetscShell functionality from the URL string.

   Logically collective on PetscShell.

   Input paramters:
+  shell -- a PetscShell object
-  url   -- URL string

   Notes: URL can point to a backend -- a .so file or a .py file.  
     A .so URL must have the form [<path>/<lib>.a:]<name> or  [<path>/<lib>.so:]<name>, and the .a or the .so
   file must contain symbols for function 'void <name>Call(const char[])' or symbols 'void <name><Message>(void)'
   for any <message> that PetscShell is expected to understand.  When PetscShellCall() is invoked
   with <message>, a symbol for 'void name<Message>(void)' is sought and called, if found.  If not, a symbol for
   'void <name>Call(const char[])' is sought and called with <message> as the argument. If neither symbol is found,
   an error occurs.
     A .py URL must have the form <path>/<module>.py:<name>, and the .py file must define a class <name> 
   that implements 'call(str)' or '<message>()' methods, as above.
     If no URL has been set, shell attempts to respond to the message using function '<message>', and, failing that,
   using function 'call' with argument <message>; the functions are retrieved from shell using PetscObjectQueryFunction().  
   If neither '<message>' nor 'call' have been previusly composed with shell (see PetscObjectComposeFunction()), an error 
   occurs.

   Level: intermediate.

.seealso: PetscShellGetURL(), PetscObjectCompose(), PetscObjectQuery(), PetscObjectComposeFunction(), PetscShellCall()
@*/
PetscErrorCode  PetscShellSetURL(PetscShell shell, const char url[]) {
  PetscErrorCode ierr;
  char *path, *name;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidCharPointer(url,2);
  if(shell->vtable) {
    ierr = PetscShellClearURL_Private(shell); CHKERRQ(ierr);
  }
  ierr = PetscStrallocpy(url,  &(shell->url));  CHKERRQ(ierr);
  ierr = PetscShellParseURL_Private(url, &path, &name, &shell->vtable_type); CHKERRQ(ierr);
  switch(shell->vtable_type) {
  case PETSC_SHELL_VTABLE_SO:
    {
      struct _n_PetscShellVTable_SO *vt;
      ierr = PetscMalloc(sizeof(struct _n_PetscShellVTable_SO), &(vt));
      shell->vtable = (void*)vt;
      ierr = PetscStrallocpy(path, &vt->path); CHKERRQ(ierr);
      ierr = PetscStrallocpy(name, &vt->name); CHKERRQ(ierr);
    }
    break;
  case PETSC_SHELL_VTABLE_PY:
    PETSC_SHELL_LOAD_VTABLE_PYTHON(shell, path, name);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, 
             "Unknown PetscShell vtable type: %d", shell->vtable_type);
  }
  PetscFunctionReturn(0);
}/* PetscShellSetURL() */


#undef  __FUNCT__
#define __FUNCT__ "PetscShellGetURL"
/*@C
   PetscShellGetURL -- retrieve the URL defining the backend that implements the PetscShellCall() functionality.

   Not collective.

   Input paramters:
.  shell -- a PetscShell object

   Output parameters:
.  url -- the URL string

   Level: beginner.

.seealso: PetscShellSetURL(), PetscShellCall()
@*/
PetscErrorCode  PetscShellGetURL(PetscShell shell, const char **url) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidPointer(url,2);
  *url = shell->url;
  PetscFunctionReturn(0);
}/* PetscShellGetURL() */
 

/* ------------------------------------------------------------------------------------------------------- */
#undef  __FUNCT__
#define __FUNCT__ "PetscShellView_Private"
PetscErrorCode PetscShellView_Private(PetscShell shell, const char *key, PetscInt rank, PetscViewer viewer) {
  PetscInt *vertices, N;
  PetscInt i, id;
  PetscBool         iascii;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)shell)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(shell,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (!iascii) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "PetscShell can only be viewed with an ASCII viewer");
  }
  else {
    if(!key) {
      ierr = PetscViewerASCIIPrintf(viewer, "Shell name: %s,\turl: %s\n", ((PetscObject)shell)->name, shell->url);                CHKERRQ(ierr);
    }
    else {
      ierr = PetscViewerASCIIPrintf(viewer, "Component: %D,\tkey: %s, \tname: %s,\turl: %s\n", rank, key, ((PetscObject)shell)->name, shell->url); CHKERRQ(ierr);
    }
    ierr = PetscShellGraphTopologicalSort(shell->dep_graph, &N, &vertices); CHKERRQ(ierr);
    if(N) {
      ierr = PetscViewerASCIIPrintf(viewer, "Component shells in the topological order of the dependence graph:\n");
      ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
      for(i = 0; i < N; ++i) {
        id = vertices[i];
        ierr = PetscShellView_Private(shell->component[id], shell->key[id], id, viewer); CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
    }
    ierr = PetscFree(vertices); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscShellView() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellView"
/*@C
   PetscShellView -- print information about a PetscShell object to an ASCII viewer. 
                     The information printed includes the object name, url and 
                     the dependent components registered with PetscShellRegisterComponentShell(),
                     PetscShellRegisterComponentURL() or PetscShellRegisterDependence().

   Not collective.

   Input paramters:
+  shell  -- a PetscShell object
-  viewer -- an ASCII PetscViewer

   Level: beginner.

.seealso: PetscShellSetURL(), PetscShellRegisterComponet(), PetscShellRegisterComponentURL(), PetscShellRegisterDependence()
@*/
PetscErrorCode PetscShellView(PetscShell shell,  PetscViewer viewer) {
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  ierr = PetscShellView_Private(shell, 0,-1, viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellView() */

/*@C 
   PetscShellGetVisitor -- retrieve the PetscShell object executing PetscShellVisit() on the current shell.
                           The visitor is one of the (possibly many) PetscShell objects with which shell 
                           has been registered as a component using PetscShellRegisterComponentShell(), 
                           PetscShellRegisterComponentURL(), or PetscShellRegisterDependence()



   Not collective.

   Input paramters:
.  shell   -- a PetscShell object

   Output parameters:
.  visitor -- the visitor PetscShell object

   Notes:  The visitor is valid only during the PetscShellVisit() and is PETSC_NULL otherwise.

   Level: intermediate

.seealso: PetscShellVisit(), PetscShellCall(), PetscShellRegisterComponentShell(), PetscShellRegisterDependence()
@*/
#undef  __FUNCT__
#define __FUNCT__ "PetscShellGetVisitor"
PetscErrorCode PetscShellGetVisitor(PetscShell shell, PetscShell *visitor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidPointer(visitor,2);
  *visitor = shell->visitor;
  PetscFunctionReturn(0);
}/* PetscShellGetVisitor() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellVisit"
/*@C
   PetscShellVisit -- traverse shell's components registered with PetscShellRegisterComponentShell(), PetscShellRegisterComponentURL(),
                      or PetscShellRegisterDependence(), and executes PetscShellCall() on the components with the given message.
                      The traversal is done in the topological order defined by the graph with edges specified by PetscShellRegisterDependence()
                      calls: this way "server" components are guaranteed to be called before the "client" components.

   Logically collective on PetscShell.

   Input paramters:
+  shell     -- a PetscShell object
-  message   -- a character string

   Notes: In response to the message the object performs actions defined by its URL (see PetscShellSetURL()). 
          Side effects may include, in particular, actions on the composed objects (see PetscObjectCompose()).

  Level: intermediate.

.seealso: PetscShellCall(), PetscShellRegisterComponentShell(), PetscShellRegisterComponentURL(), PetscShellRegisterDependence()
@*/
PetscErrorCode PetscShellVisit(PetscShell shell, const char* message){
  PetscInt i, id, N, *vertices;
  PetscShell component;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidCharPointer(message,2);
  ierr = PetscShellGraphTopologicalSort(shell->dep_graph, &N, &vertices); CHKERRQ(ierr);
  for(i = 0; i < N; ++i) {
    id = vertices[i];
    component = shell->component[id];
    /* Save the component's visitor */
    component->visitor = shell;
    /* Call "configure" */
    ierr = PetscShellCall(component, message); CHKERRQ(ierr);
    /* Clear visitor */
    component->visitor = PETSC_NULL;
  }
  ierr = PetscFree(vertices); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellVisit() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellGetKeyID_Private"
PetscErrorCode  PetscShellGetKeyID_Private(PetscShell shell, const char key[], PetscInt *_id, PetscBool  *_found){
  PetscInt i;
  PetscBool  eq;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Check whether a component with the given key has already been registered. */
  if(_found){*_found = PETSC_FALSE;}
  for(i = 0; i < shell->N; ++i) {
    ierr = PetscStrcmp(key, shell->key[i], &eq); CHKERRQ(ierr);
    if(eq) {
      if(_id) {*_id = i;}
      if(_found){*_found = PETSC_TRUE;}
    }
  }
  PetscFunctionReturn(0);
}/* PetscShellGetKeyID_Private() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellRegisterKey_Private"
PetscErrorCode  PetscShellRegisterKey_Private(PetscShell shell, const char key[], PetscShell component, PetscInt *_id) {
  PetscInt v, id;
  PetscBool  found;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Check whether a component with the given key has already been registered. */
  ierr = PetscShellGetKeyID_Private(shell, key, &id, &found); CHKERRQ(ierr);
  if(found) {
    if(component) {
      /* Replace the component with the new one. */
      shell->component[id] = component;
    }
    if(_id) *_id = id;
    PetscFunctionReturn(0);
  }
  /* No such key found. */
  if(!component) {
    /* Create a new component for this key. */
    ierr = PetscShellCreate(((PetscObject)shell)->comm, &component); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)component, key);          CHKERRQ(ierr);
  }
  if(shell->N >= shell->maxN) {
    /* No more empty component slots, therefore, expand the component array */
    PetscShell *new_components;
    char **new_keys;
    ierr = PetscMalloc(sizeof(PetscShell)*(shell->maxN+CHUNKSIZE), &new_components);        CHKERRQ(ierr);
    ierr = PetscMemcpy(new_components, shell->component, sizeof(PetscShell)*(shell->maxN)); CHKERRQ(ierr);
    ierr = PetscMemzero(new_components+shell->maxN,sizeof(PetscShell)*(CHUNKSIZE));         CHKERRQ(ierr);
    ierr = PetscFree(shell->component);                                                     CHKERRQ(ierr);
    shell->component = new_components;
    /* Expand the key array */
    ierr = PetscMalloc(sizeof(char*)*(shell->maxN+CHUNKSIZE), &new_keys);  CHKERRQ(ierr);
    ierr = PetscMemcpy(new_keys, shell->key, sizeof(char*)*(shell->maxN)); CHKERRQ(ierr);
    ierr = PetscMemzero(new_keys+shell->maxN,sizeof(char*)*(CHUNKSIZE));   CHKERRQ(ierr);
    ierr = PetscFree(shell->key);                                          CHKERRQ(ierr);
    shell->key = new_keys;
    shell->maxN += CHUNKSIZE;
  }
  id = shell->N;
  ++(shell->N);
  /* Store key and component. */
  ierr = PetscStrallocpy(key, &(shell->key[id]));  CHKERRQ(ierr);
  shell->component[id] = component;
  /* Add a new vertex to the dependence graph.  This vertex will correspond to the newly registered component. */
  ierr = PetscShellGraphAddVertex(shell->dep_graph, &v); CHKERRQ(ierr);
  /* v must equal id */
  if(v != id) {
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT, "New dependence graph vertex %d for key %s not the same as component id %d", v, key, id); 
  }
  if(_id) *_id = id;
  PetscFunctionReturn(0);
}/* PetscShellRegisterKey_Private() */


#undef  __FUNCT__
#define __FUNCT__ "PetscShellRegisterComponentShell"
/*@C
   PetscShellRegisterComponentShell -- register a component as a component of shell, identifiable within shell by key.  
                                    If a component with key already exists within shell, and component is not PETSC_NULL,
                                  the old component is replace. 
                                    If component is PETSC_NULL, and key has not been registered with PetscShell,
                                  and new PetscShell component is created with key as its name, and stored in shell under key.
                                    If component is PETSC_NULL, and key has already been registered with PetscShell,
                                  nothing is done.
                                    The component can be later obtained with a PetscShellGetComponent()
                                  call, or referred to be its key in a PetscShellRegisterDependenceCall().
                                  Components can be visited with a particular message using PetscShellVisit().

   Logically collective on PetscShell.

   Input paramters:
+  shell     -- a PetscShell object
.  key       -- a character string designating the added component.
-  component -- a PetscShell object to add (or PETSC_NULL)


  Level: intermediate.

.seealso: PetscShellGetComponent(), PetscShellRegisterComponentURL(), PetscShellRegisterDependence(), PetscShellVisit()
@*/
PetscErrorCode  PetscShellRegisterComponentShell(PetscShell shell, const char key[], PetscShell component){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidCharPointer(key,2);
  ierr = PetscShellRegisterKey_Private(shell, key, component, PETSC_NULL); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellRegisterComponentShell() */

#undef  __FUNCT__
#define __FUNCT__ "PetscShellRegisterComponentURL"
/*@C
   PetscShellRegisterComponentURL -- register a key as a component of shell (see PetscShellRegisterComponentShell()) 
                                         and set the given URL on the newly created component.


   Logically collective on PetscShell.

   Input paramters:
+  shell -- a PetscShell object
.  key   -- a character string designating the added component.
-  url   -- a character string desigating the URL of the added component.

   Notes: equivalent to the sequence 
          PetscShellRegisterComponentShell(shell,key,PETSC_NULL); 
          PetscShellGetComponent(shell, key, &component);
          PetscShellSetURL(component, url);

  Level: intermediate.

.seealso: PetscShellRegisterComponentShell(), PetscShellGetComponent(), PetscShellSetURL()
@*/
PetscErrorCode  PetscShellRegisterComponentURL(PetscShell shell, const char key[], const char url[]){
  PetscErrorCode ierr;
  PetscInt id;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidCharPointer(key,2);
  PetscValidCharPointer(url,3);
  ierr = PetscShellRegisterKey_Private(shell, key, PETSC_NULL, &id); CHKERRQ(ierr);
  ierr = PetscShellSetURL(shell->component[id], url); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellRegisterComponentURL() */


#undef  __FUNCT__
#define __FUNCT__ "PetscShellRegisterDependence"
/*@C
   PetscShellRegisterDependence -- register a dependence between shell's components designated by keys "serverkey" and "clientkey".
                                   If components with such keys have not already been added to shell with PetscShellRegisterComponentShell()
                                   or PetscShellComponentRegisterURL(), new components are created.  During a call to PetscShellVisit()
                                   the component with key "serverkey" is guaranteed to be called before "clientkey".


   Logically collective on PetscShell.

   Input paramters:
+  shell       -- a PetscShell object
.  serverkey   -- a character string designating the server component
-  clientkey   -- a character string desigating the client component

  Level: intermediate.

.seealso: PetscShellRegisterComponentShell(), PetscShellRegisterComponentURL(), PetscShellGetComponent(), PetscShellVisit()
@*/
PetscErrorCode  PetscShellRegisterDependence(PetscShell shell, const char serverkey[], const char clientkey[])
{
  PetscInt clientid, serverid;
  PetscErrorCode ierr; 
  PetscFunctionBegin; 
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidCharPointer(clientkey,2);
  PetscValidCharPointer(serverkey,3);
  /* Register keys */
  ierr = PetscShellRegisterKey_Private(shell, clientkey, PETSC_NULL, &clientid); CHKERRQ(ierr);
  ierr = PetscShellRegisterKey_Private(shell, serverkey, PETSC_NULL, &serverid); CHKERRQ(ierr);
  /*
    Add the dependency edge to the dependence_graph as follows (serverurl, clienturl): 
     this means "server preceeds client", so server should be configured first.
  */
  ierr = PetscShellGraphAddEdge(shell->dep_graph, serverid, clientid); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellRegisterDependence() */



#undef  __FUNCT__
#define __FUNCT__ "PetscShellDestroy"
/*@C 
   PetscShellDestroy -- destroy PetscShell.

   Not collective.

   Input paramters:
.  shell -- a PetscShell object

   Level: beginner.

.seealso: PetscShellCreate(), PetscShellSetURL(), PetscShellCall()
@*/
PetscErrorCode  PetscShellDestroy(PetscShell *shell)
{
  PetscInt       i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (!*shell) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*shell,PETSC_SHELL_CLASSID,1);
  if (--((PetscObject)(*shell))->refct > 0) PetscFunctionReturn(0);
  for(i = 0; i < (*shell)->N; ++i){
    ierr = PetscObjectDestroy((PetscObject*)&(*shell)->component[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree((*shell)->component); CHKERRQ(ierr);
  ierr = PetscShellGraphDestroy((*shell)->dep_graph); CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellDestroy()*/

#undef  __FUNCT__
#define __FUNCT__ "PetscShellCreate"
/*@C 
   PetscShellCreate -- create an empty PetscShell object.

   Logically collective on comm.

   Input paramters:
.  comm      -- the MPI_Comm to create the PetscShell object on.

   Output parameters:
.  shell -- the created PetscShell object.

   Notes: the default shell corresponding to PETSC_COMM_WORLD is PETSC_SHELL_DEFAULT_WORLD.
          Likewise for PETSC_COMM_SELF and PETSC_SHELL_DEFAULT_SELF.

   Level: beginner.

.seealso: PetscShellDestroy(), PetscShellSetURL(), PetscShellCall()
@*/
PetscErrorCode  PetscShellCreate(MPI_Comm comm, PetscShell *shell){
  PetscShell shell_;
  PetscErrorCode ierr;
  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscShellInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  PetscValidPointer(shell,2);
  ierr = PetscHeaderCreate(shell_,_p_PetscShell,PetscInt,PETSC_SHELL_CLASSID,0,"PetscShell","String message interpreter and dependence organizer","shell",comm,PetscShellDestroy,PetscShellView);CHKERRQ(ierr);
  shell_->visitor     = PETSC_NULL;
  shell_->component   = PETSC_NULL;
  shell_->vtable_type = PETSC_SHELL_VTABLE_NONE;
  shell_->vtable      = PETSC_NULL;
  shell_->N = shell_->maxN = 0;
  /* FIX: should only create a graph on demand */
  ierr = PetscShellGraphCreate(&shell_->dep_graph); CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)shell_,PETSCSHELL);CHKERRQ(ierr);
  *shell = shell_;
  PetscFunctionReturn(0);
}/* PetscShellCreate() */


#undef  __FUNCT__
#define __FUNCT__ "PetscShellGetComponent"
/*@C 
   PetscShellGetComponent -- extract shell's component corresponding to key.
                             If key has been previously used with PetscShellRegisterComponentShell(),
                             PetscShellRegisterComponentURL() or PetscShellRegisterDependence()
                             this will return the corresponding component created during the first
                             such call.

   Logically collective on comm.

   Input paramters:
+  shell -- PetscShell object being queried
-  key   -- a character string designating the key of the component being sought

   Output parameters:
+  component -- the extracted component PetscShell object (or PETSC_NULL) 
-  found     -- PetscBool flag indicating whether a component with the given key has been found (or PETSC_NULL)

   Notes: component can be PETSC_NULL, in which case only found is returned (if it is itself not PETSC_NULL).
          This is useful for quering for the presence of the given component, without extracting it.

   Level: beginner.

.seealso: PetscShellRegisterComponentShell(), PetscShellRegisterComponentURL(), PetscShellRegisterDependence()
@*/
PetscErrorCode  PetscShellGetComponent(PetscShell shell, const char key[], PetscShell *component, PetscBool  *found) {
  PetscInt id;
  PetscErrorCode ierr;
  PetscBool found_;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(shell,PETSC_SHELL_CLASSID,1);
  PetscValidCharPointer(key,2);
  ierr = PetscShellGetKeyID_Private(shell, key, &id, &found_); CHKERRQ(ierr);
  if(found_ && component) {
    *component = shell->component[id];
  }
  if(found) {*found = found_;}
  PetscFunctionReturn(0);
}/* PetscShellGetComponent() */



#undef  __FUNCT__
#define __FUNCT__ "PetscShellFinalizePackage"
PetscErrorCode PetscShellFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscShellPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}/* PetscShellFinalizePackage() */


#undef  __FUNCT__
#define __FUNCT__ "PetscShellInitializePackage"
PetscErrorCode PetscShellInitializePackage(const char path[]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(PetscShellPackageInitialized) PetscFunctionReturn(0);
  PetscShellPackageInitialized = PETSC_TRUE;
  /* Register classes */
  ierr = PetscClassIdRegister(PETSC_SHELL_CLASS_NAME, &PETSC_SHELL_CLASSID); CHKERRQ(ierr);
  /* Register finalization routine */
  ierr = PetscRegisterFinalize(PetscShellFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscShellInitializePackage() */

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Shell_default_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscShell.
*/
static PetscMPIInt Petsc_Shell_default_keyval = MPI_KEYVAL_INVALID;

#undef  __FUNCT__
#define __FUNCT__ "PETSC_SHELL_DEFAULT_"
PetscShell  PETSC_SHELL_DEFAULT_(MPI_Comm comm) {
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscShell       shell;

  PetscFunctionBegin;
  if (Petsc_Shell_default_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Shell_default_keyval,0);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_SHELL_DEFAULT_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," "); PetscFunctionReturn(0);}
  }
  ierr = MPI_Attr_get(comm,Petsc_Shell_default_keyval,(void **)(&shell),(PetscMPIInt*)&flg);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_SHELL_DEFAULT_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," "); PetscFunctionReturn(0);}
  if (!flg) { /* PetscShell not yet created */
    ierr = PetscShellCreate(comm, &shell);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_SHELL_DEFAULT_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," "); PetscFunctionReturn(0);}
    ierr = PetscObjectRegisterDestroy((PetscObject)shell);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_SHELL_DEFAULT_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," "); PetscFunctionReturn(0);}
    ierr = MPI_Attr_put(comm,Petsc_Shell_default_keyval,(void*)shell);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_SHELL_DEFAULT_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," "); PetscFunctionReturn(0);}
  } 
  PetscFunctionReturn(shell);
}/* PETSC_SHELL_DEFAULT_() */
