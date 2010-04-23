#define PETSC_DLL

#include "petscsys.h"
#include "petscfwk.h"

PetscClassId PETSC_FWK_CLASSID;
static char PETSC_FWK_CLASS_NAME[] = "PetscFwk";

static PetscTruth PetscFwkPackageInitialized = PETSC_FALSE;
typedef enum{PETSC_FWK_COMPONENT_SO, PETSC_FWK_COMPONENT_PY} PetscFwkComponentType;

typedef PetscErrorCode (*PetscFwkConfigurePYComponentType)(PetscFwk fwk, const char* url, const char* path, const char* name, PetscInt state, PetscObject *component);

EXTERN_C_BEGIN
PetscFwkConfigurePYComponentType PetscFwkConfigurePYComponent = PETSC_NULL; 
EXTERN_C_END

#define PETSC_FWK_CHECK_PYTHON                                                     \
  if(PetscFwkConfigurePYComponent == PETSC_NULL) {                                 \
    PetscErrorCode ierr;                                                           \
    ierr = PetscPythonInitialize(PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);           \
    if(PetscFwkConfigurePYComponent == PETSC_NULL) {                               \
      SETERRQ(PETSC_ERR_LIB, "Couldn't initialize Python support for PetscFwk");   \
    }                                                                              \
  }                                                                                \


/* 
   Graph/topological sort stuff from BOOST.
   May need to be replaced with a custom, all-C implementation
*/
// >> C++
#include <iostream>
#include <map>
#include <vector>
#include <list>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>


struct vertex_id {
  PetscInt id;
};
//
// We use bundled properties to store id with the vertex.
typedef ::boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, vertex_id>             dependence_graph_type;
typedef boost::graph_traits<dependence_graph_type>::vertex_descriptor                                   vertex_type;
//
struct dependence_graph_cycle_detector_type : public boost::default_dfs_visitor {
  void back_edge(boost::graph_traits<dependence_graph_type>::edge_descriptor e, const dependence_graph_type& g) {
    std::basic_ostringstream<char> ss;
    ss << "Edge creates dependence loop: [" << (g[::boost::source(e,g)].id) << " --> " << (g[::boost::target(e,g)].id) << "]; ";
    throw ss.str();
  }//back_edge()
};// dependence_graph_cycle_detector_type
// << C++


struct _n_PetscFwkRecord {
  std::string                url;
  std::string                path, name;
  vertex_type                vertex;
  PetscFwkComponentType      type;
  PetscObject                component;
  PetscFwkComponentConfigure configure;
};

struct _p_PetscFwk {
  PETSCHEADER(int);
  // >> C++
  // Since PetscFwk is allocated using the C-style PetscMalloc, 
  // the C++-object data members are not automatically constructed, 
  // so they have to be 'newed', hance must be pointers.
  std::map<std::string, PetscInt>         *id;
  std::vector<_n_PetscFwkRecord>          *record;
  dependence_graph_type                   *dependence_graph;
  // << C++
};

// >> C++
#undef  __FUNCT__
#define __FUNCT__ "PetscFwkConfigureSort_Private"
PetscErrorCode PetscFwkConfigureSort_Private(PetscFwk fwk, std::list<vertex_type>& antitopological_vertex_order){
  PetscFunctionBegin;
  // >> C++
  /* Run a cycle detector on the dependence graph */
  try {
    dependence_graph_cycle_detector_type cycle_detector;
    ::boost::depth_first_search(*(fwk->dependence_graph), boost::visitor(cycle_detector));
  }
  catch(const std::string& s) {
    SETERRQ1(PETSC_ERR_ARG_WRONGSTATE, "Component dependence graph has a loop: %s", s.c_str());
  }
  catch(...) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Exception caught while detecting loops in the dependence graph");
  }
  /* 
     Run topological sort on the dependence graph; remember that the vertices are returned in the "antitopological" order.
     This, in fact, is just what we need, since edges in the graph are entered as (client, server), implying that server 
     must be configured *before* the client and BOOSTs topological sort will in fact order server before client.
   */
  try{
    ::boost::topological_sort(*(fwk->dependence_graph), std::back_inserter(antitopological_vertex_order));
  }
  catch(...) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Exception caught while performing topological_sort in the dependence graph");
  }

  PetscFunctionReturn(0);
}/* PetscFwkConfigureSort_Private() */
// << C++

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkViewConfigurationOrder"
PetscErrorCode PetscFwkViewConfigurationOrder(PetscFwk fwk, PetscViewer viewerASCII){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  // >> C++
  std::list<vertex_type> antitopological_vertex_order;
  ierr = PetscFwkConfigureSort_Private(fwk, antitopological_vertex_order); CHKERRQ(ierr);
  /* traverse the vertices in their antitopological, extract component ids from vertices and  configure each corresponding component */
  /* current_state starts with 1, since 0 was used during Preconfigure */
  for(std::list<vertex_type>::iterator iter = antitopological_vertex_order.begin(); iter != antitopological_vertex_order.end(); ++iter) {
    PetscInt id = (*(fwk->dependence_graph))[*iter].id;
    std::string urlstring = (*fwk->record)[id].url;
    if(iter != antitopological_vertex_order.begin()) {
      ierr = PetscViewerASCIIPrintf(viewerASCII, ", "); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewerASCII, "%d: %s", id, urlstring.c_str()); CHKERRQ(ierr);
  }
  // << C++
  ierr = PetscViewerASCIIPrintf(viewerASCII, "\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkViewConfigurationOrder() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkConfigure"
PetscErrorCode PetscFwkConfigure(PetscFwk fwk, PetscInt state){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  // >> C++
  std::list<vertex_type> antitopological_vertex_order;
  ierr = PetscFwkConfigureSort_Private(fwk, antitopological_vertex_order); CHKERRQ(ierr);
  /* traverse the vertices in their antitopological, extract component ids from vertices and  configure each corresponding component */
  for(PetscInt current_state=0;current_state<=state; ++current_state) {
    ierr = PetscObjectSetState((PetscObject)fwk,current_state); CHKERRQ(ierr);
    for(std::list<vertex_type>::iterator iter = antitopological_vertex_order.begin(); iter != antitopological_vertex_order.end(); ++iter) {
      PetscFwkComponentConfigure configure = PETSC_NULL;
      PetscInt id = (*(fwk->dependence_graph))[*iter].id;
      PetscObject component = (*fwk->record)[id].component;
      switch((*fwk->record)[id].type){
      case PETSC_FWK_COMPONENT_SO:
        configure = (*fwk->record)[id].configure;
        if(configure != PETSC_NULL) {
          ierr = (*configure)(fwk,current_state,&component); CHKERRQ(ierr);
        }
        break;
      case PETSC_FWK_COMPONENT_PY:
        PETSC_FWK_CHECK_PYTHON;
        /* Incref the objects being passed onto the Python side:
           they will get wrapped as Python objects, which, eventually, will go
           out of scope, be garbage collected, and will attempt to destroy the
           underlying Petsc objects.
        */
        ierr = PetscObjectReference((PetscObject)fwk); CHKERRQ(ierr);
        /* FIX: What about component? It's tricky: configuring it might alter the object.
           However, component should probably always be increfed by the framework, i.e., 
           at creation time.  Still, if configuration alters the object, it should be 
           decrefed or, at least, the reference should be stolen by Python, or something.
        */
        ierr = PetscFwkConfigurePYComponent(fwk, (*fwk->record)[id].url.c_str(),(*fwk->record)[id].path.c_str(),(*fwk->record)[id].name.c_str(), state, &component); CHKERRQ(ierr);
        /* Now decref fwk */
        ierr = PetscObjectDereference((PetscObject)fwk); CHKERRQ(ierr);
        break;
      }
    }
  }
  // << C++
  PetscFunctionReturn(0);
}/* PetscFwkConfigure() */



static PetscDLLibrary PetscFwkDLList = PETSC_NULL;
#define PETSC_FWK_MAX_URL_LENGTH 1024

/* 
   Normalize the url (by truncating to PETSC_FWK_MAX_URL_LENGTH) and parse it to find out the component type and location.
   Warning: if nurl, npath, nname are passed in as NULL, the returned char pointers are borrowed and their contents
   must be copied elsewhere to be preserved 
*/
#undef  __FUNCT__
#define __FUNCT__ "PetscFwkParseURL_Private"
PetscErrorCode PETSC_DLLEXPORT PetscFwkParseURL_Private(PetscFwk fwk, const char inurl[], char url[], char path[], char name[], PetscFwkComponentType *type){
  char *n, *s;
  static PetscInt nlen = PETSC_FWK_MAX_URL_LENGTH;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* FIX: this routine should replace the filesystem path by an abolute path for real normalization */
  /* Copy the inurl so we can manipulate it inplace and also truncate to the max allowable length */
  ierr = PetscStrncpy(path, inurl, nlen); CHKERRQ(ierr);  
  /* Split url <path>:<name> into <path> and <name> */
  ierr = PetscStrrchr(path,':',&n); CHKERRQ(ierr);
  /* Make sure it's not the ":/" of the "://" separator */
  if(!n[0] || n[0] == '/') {
    SETERRQ2(PETSC_ERR_ARG_WRONG, 
           "Could not locate component name within the URL.\n"
           "Must have url = [<path/><library>:]<name>.\n"
           "Instead got %s\n"
           "Remember that URL is always truncated to the max allowed length of %d", 
           inurl, nlen);
  }
  /* Copy n to name */
  ierr = PetscStrcpy(name, n); CHKERRQ(ierr);
  /* If n isn't the whole path (i.e., there is a ':' separator), end 'path' right before the located ':' */
  if(n != path) {
    n[-1] = '\0';
  }
  /* Find and remove the library suffix */
  ierr = PetscStrrchr(path,'.',&s);CHKERRQ(ierr);
  /* Determine the component library type: .so or .py */
  /* FIX: we should really be using PETSc's internally defined suffices */
  if(s != path && s[-1] == '.') {
    if((s[0] == 'a' && s[1] == '\0') || (s[0] == 's' && s[1] == 'o' && s[2] == '\0')){
      *type = PETSC_FWK_COMPONENT_SO;
    }
    else if (s[0] == 'p' && s[1] == 'y' && s[2] == '\0'){
      *type = PETSC_FWK_COMPONENT_PY;
    }
    else {
      SETERRQ3(PETSC_ERR_ARG_WRONG, 
           "Unknown library suffix within the URL.\n"
           "Must have url = [<path/><library>:]<name>,\n"
           "where library = <libname>.<suffix>, suffix = .a || .so || .py.\n"
           "Instead got url %s and suffix %s\n"
           "Remember that URL is always truncated to the max allowed length of %d", 
               inurl, s,nlen);     
    }
    /* Remove the suffix from the library name */
    s[-1] = '\0';  
  }
  else {
    SETERRQ2(PETSC_ERR_ARG_WRONG, 
             "Could not locate library within the URL.\n"
             "Must have url = [<path/><library>:]<name>.\n"
             "Instead got %s\n"
             "Remember that URL is always truncated to the max allowed length of %d", 
             inurl, nlen);     
 
  }
  ierr = PetscStrcpy(url, path); CHKERRQ(ierr);
  ierr = PetscStrcat(url, ":");  CHKERRQ(ierr);
  ierr = PetscStrcat(url, name); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkParseURL_Private() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponentID_Private"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponentID_Private(PetscFwk fwk, const char inurl[], PetscInt *_id){
  PetscFwkComponentType type;
  PetscInt id;
  char url[PETSC_FWK_MAX_URL_LENGTH+1], path[PETSC_FWK_MAX_URL_LENGTH+1], name[PETSC_FWK_MAX_URL_LENGTH+1];
  PetscObject component = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFwkParseURL_Private(fwk, inurl, url, path, name, &type); CHKERRQ(ierr);
  // Check whether a component with the given url has already been registered.  If so, return its id, if it has been requested.
  std::string urlstring(url);
  if((*(fwk->id)).count(urlstring)) {
    if(_id) {
      *_id = (*(fwk->id))[urlstring];
    }
    PetscFunctionReturn(0);
  }
  // Insert url with an empty record byt of correct type 
  //ierr = PetscFwkComponentID_Private(fwk, nurl, type, &id); CHKERRQ(ierr);
  // >> C++
  // Check whether url has already been registered
  if((*(fwk->id)).count(urlstring)) {
    // Retrieve the existing id
    id = (*(fwk->id))[urlstring];
  }
  else {
    // Assign and store the new id for urlstring.
    id = (*(fwk->id)).size();
    (*(fwk->id))[urlstring] = id;
    /* We assume that at this point id == (*(fwk->record)).size() */
    (*(fwk->record)).push_back(_n_PetscFwkRecord());
    /* Add a new vertex to the dependence graph.  This vertex will correspond to the newly registered component. */
    vertex_type v = ::boost::add_vertex(*(fwk->dependence_graph));
    /* Attach id to v */
    (*(fwk->dependence_graph))[v].id = id;
    /* Store urlstring and v in the new component record */
    (*(fwk->record))[id].url = urlstring;
    (*(fwk->record))[id].name = std::string(name);
    (*(fwk->record))[id].path = std::string(path);
    (*(fwk->record))[id].vertex = v;
    /* Set component type */
    (*(fwk->record))[id].type = type;
    /* The rest is NULL */
    (*(fwk->record))[id].component = PETSC_NULL;
    (*(fwk->record))[id].configure = PETSC_NULL;
  }
  switch(type) {
  case PETSC_FWK_COMPONENT_SO:
    {
      char sym[PETSC_FWK_MAX_URL_LENGTH+26+1];
      PetscFwkComponentConfigure configure = PETSC_NULL;
      /* Build the configure symbol from name and standard prefix */
      ierr = PetscStrcpy(sym, "PetscFwkComponentConfigure"); CHKERRQ(ierr);
      ierr = PetscStrcat(sym, name); CHKERRQ(ierr);
      /* Load the library designated by 'path' and retrieve from it the configure routine designated by the constructed symbol */
      ierr = PetscDLLibrarySym(((PetscObject)fwk)->comm, &PetscFwkDLList, path, sym, (void**)(&configure)); CHKERRQ(ierr);
      /* Run the configure routine, which should return a valid object or PETSC_NULL */
      ierr = (*configure)(fwk, 0, &component); CHKERRQ(ierr);
      (*(fwk->record))[id].component = component;
      (*(fwk->record))[id].configure = configure;
      
    }
    break;
  case PETSC_FWK_COMPONENT_PY:
    PETSC_FWK_CHECK_PYTHON;
    ierr = PetscFwkConfigurePYComponent(fwk, (*fwk->record)[id].url.c_str(), (*fwk->record)[id].path.c_str(), (*fwk->record)[id].name.c_str(), 0, &component); CHKERRQ(ierr);
    (*(fwk->record))[id].component = component;
    /* configure field remains NULL for a Py component */
    break;
  default:
    SETERRQ2(PETSC_ERR_ARG_WRONG, 
             "Could not determine type of component with url %s.\n"
             "Remember: URL was truncated past the max allowed length of %d", 
             inurl, PETSC_FWK_MAX_URL_LENGTH);    
  }
  if(_id) {
    *_id = id;
  }
  PetscFunctionReturn(0);
}/* PetscFwkRegisterComponentID_Private()*/

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterComponent"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterComponent(PetscFwk fwk, const char url[]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFwkRegisterComponentID_Private(fwk, url, PETSC_NULL); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkRegisterComponent() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkRegisterDependence"
PetscErrorCode PETSC_DLLEXPORT PetscFwkRegisterDependence(PetscFwk fwk, const char clienturl[], const char serverurl[])
{
  PetscInt clientid, serverid;
  PetscErrorCode ierr; 
  PetscFunctionBegin; 
  PetscValidCharPointer(clienturl,2);
  PetscValidCharPointer(serverurl,3);
  /* Register urls */
  ierr = PetscFwkRegisterComponentID_Private(fwk, clienturl, &clientid); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentID_Private(fwk, serverurl, &serverid); CHKERRQ(ierr);

  /*
    Add the dependency edge to the dependence_graph as follows (clienturl, serverurl): 
     this means "client depends on server", so server should be configured first.
    For this reason we need to order components using an "antitopological" sort of the dependence_graph.
    BOOST Graph Library does just that, but keep that in mind if reimplementing the graph/sort in C.
  */
  vertex_type c,s;
  c = (*(fwk->record))[clientid].vertex;
  s = (*(fwk->record))[serverid].vertex;
  ::boost::add_edge(c,s, *(fwk->dependence_graph));
  PetscFunctionReturn(0);
}/*PetscFwkRegisterDependence()*/



#undef  __FUNCT__
#define __FUNCT__ "PetscFwkDestroy"
PetscErrorCode PETSC_DLLEXPORT PetscFwkDestroy(PetscFwk fwk)
{
  PetscErrorCode ierr;
  if (--((PetscObject)fwk)->refct > 0) PetscFunctionReturn(0);
  // >> C++
  for(std::vector<struct _n_PetscFwkRecord>::iterator i = fwk->record->begin(); i != fwk->record->end(); ++i) {
    if(i->component != PETSC_NULL) {
      ierr = PetscObjectDestroy(i->component); CHKERRQ(ierr);
    }
  }
  delete fwk->id;
  delete fwk->record;             
  delete fwk->dependence_graph;
  // << C++
  ierr = PetscHeaderDestroy(fwk);CHKERRQ(ierr);
  fwk = PETSC_NULL;
  PetscFunctionReturn(0);
}/* PetscFwkDestroy()*/

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkCreate"
PetscErrorCode PETSC_DLLEXPORT PetscFwkCreate(MPI_Comm comm, PetscFwk *framework){
  PetscFwk fwk;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /*#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)*/
  ierr = PetscFwkInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  /*#endif*/
  PetscValidPointer(framework,2);
  ierr = PetscHeaderCreate(fwk,_p_PetscFwk,PetscInt,PETSC_FWK_CLASSID,0,"PetscFwk",comm,PetscFwkDestroy,0);CHKERRQ(ierr);
  // >> C++
  fwk->id               = new std::map<std::string, PetscInt>;
  fwk->record           = new std::vector<struct _n_PetscFwkRecord>;
  fwk->dependence_graph = new dependence_graph_type;
  // << C++
  *framework = fwk;
  PetscFunctionReturn(0);
}/* PetscFwkCreate() */


#undef  __FUNCT__
#define __FUNCT__ "PetscFwkGetComponent"
PetscErrorCode PETSC_DLLEXPORT PetscFwkGetComponent(PetscFwk fwk, const char url[], PetscObject *component) {
  PetscFunctionBegin;
  PetscValidCharPointer(url,2);
  // >> C++
  try{
    *component = (*fwk->record)[(*fwk->id)[url]].component;
  }
  catch(...){
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Couldn't retrieve PetscFwk component with url %s", url);
  }
  // << C++
  PetscFunctionReturn(0);
}/* PetscFwkGetComponent() */

#undef  __FUNCT__
#define __FUNCT__ "PetscFwkFinalizePackage"
PetscErrorCode PetscFwkFinalizePackage(void){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFwkPackageInitialized = PETSC_FALSE;
  if(PetscFwkDLList != PETSC_NULL) {
    ierr = PetscDLLibraryClose(PetscFwkDLList); CHKERRQ(ierr);
    PetscFwkDLList = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}/* PetscFwkFinalizePackage() */



#undef  __FUNCT__
#define __FUNCT__ "PetscFwkInitializePackage"
PetscErrorCode PetscFwkInitializePackage(const char path[]){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(PetscFwkPackageInitialized) {
    PetscFwkPackageInitialized = PETSC_TRUE;
  }
  /* Regster classes */
  ierr = PetscClassIdRegister(PETSC_FWK_CLASS_NAME, &PETSC_FWK_CLASSID); CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscFwkFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* PetscFwkInitializePackage() */



