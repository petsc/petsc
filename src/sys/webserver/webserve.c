
#include <petscsys.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_NEED_CLOSE_PROTO)
PETSC_EXTERN int close(int);
#endif
PETSC_INTERN PetscErrorCode PetscSocketEstablish(int,int*);
PETSC_INTERN PetscErrorCode PetscSocketListen(int,int*);

/*
      Implements a crude webserver allowing the snooping on running application codes.

     Developer Notes: Most of this code, including the webserver, perhaps, belongs properly in the SAWs with perhaps a few hooks
      for application/libraries like PETSc to interact with it.
*/
#include <pthread.h>
#include <time.h>
#define PROTOCOL   "HTTP/1.1"
#define RFC1123FMT "%a, %d %b %Y %H:%M:%S GMT"

#undef __FUNCT__
#define __FUNCT__ "PetscWebSendHeader"
PetscErrorCode PetscWebSendHeader(FILE *f, int status, const char *title, const char *extra, const char *mime, int length)
{
  time_t now;
  char   timebuf[128];

  PetscFunctionBegin;
  fprintf(f, "%s %d %s\r\n", PROTOCOL, status, title);
  fprintf(f, "Server: %s\r\n", "petscserver/1.0");
  now = time(NULL);
  strftime(timebuf, sizeof(timebuf), RFC1123FMT, gmtime(&now));
  fprintf(f, "Date: %s\r\n", timebuf);
  if (extra) fprintf(f, "%s\r\n", extra);
  if (mime) fprintf(f, "Content-Type: %s\r\n", mime);
  if (length >= 0) fprintf(f, "Content-Length: %d\r\n", length);
  fprintf(f, "Connection: close\r\n");
  fprintf(f, "\r\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebSendFooter"
PetscErrorCode PetscWebSendFooter(FILE *fd)
{
  PetscFunctionBegin;
  fprintf(fd, "</BODY></HTML>\r\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebSendError"
PetscErrorCode PetscWebSendError(FILE *f, int status, const char *title, const char *extra, const char *text)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscWebSendHeader(f, status, title, extra, "text/html", -1);CHKERRQ(ierr);
  fprintf(f, "<HTML><HEAD><TITLE>%d %s</TITLE></HEAD>\r\n", status, title);
  fprintf(f, "<BODY><H4>%d %s</H4>\r\n", status, title);
  fprintf(f, "%s\r\n", text);
  ierr = PetscWebSendFooter(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscviewersaws.h>
#undef __FUNCT__
#define __FUNCT__ "PetscAMSObjectsDisplayList"
/*
    Displays all the PETSc objects published with SAWs in a simple HTML list

    Does NOT use Javascript or JSON-RPC
*/
static PetscErrorCode PetscAMSObjectsDisplayList(FILE *fd)
{
  PetscErrorCode     ierr;
  char               host[256],**comm_list,**mem_list,**fld_list;
  AMS_Comm           ams;
  PetscInt           i = 0,j;
  AMS_Memory_type    mtype;
  AMS_Data_type      dtype;
  AMS_Shared_type    stype;
  AMS_Reduction_type rtype;
  AMS_Memory         memory;
  int                len;
  void               *addr;
  PetscBool          stack;

  ierr = PetscGetHostName(host,256);CHKERRQ(ierr);
  ierr = AMS_Connect(host, -1, &comm_list);
  ierr = PetscWebSendHeader(fd, 200, "OK", NULL, "text/html", -1);CHKERRQ(ierr);
  if (!comm_list || !comm_list[0]) fprintf(fd, "AMS Communicator not running</p>\r\n");
  else {
    ierr = AMS_Comm_attach(comm_list[0],&ams);
    ierr = AMS_Comm_get_memory_list(ams,&mem_list);
    if (!mem_list[0]) fprintf(fd, "AMS Communicator %s has no published memories</p>\r\n",comm_list[0]);
    else {
      fprintf(fd, "<HTML><HEAD><TITLE>Petsc Application Server</TITLE></HEAD>\r\n<BODY>");
      fprintf(fd,"<ul>\r\n");
      while (mem_list[i]) {
        ierr = PetscStrcmp(mem_list[i],"Stack",&stack);CHKERRQ(ierr);
        if (stack) {i++; continue;}
        fprintf(fd,"<li> %s</li>\r\n",mem_list[i]);
        ierr = AMS_Memory_attach(ams,mem_list[i],&memory,NULL);
        ierr = AMS_Memory_get_field_list(memory, &fld_list);
        j    = 0;
        fprintf(fd,"<ul>\r\n");
        while (fld_list[j]) {
          fprintf(fd,"<li> %s",fld_list[j]);
          ierr = AMS_Memory_get_field_info(memory, fld_list[j], &addr, &len, &dtype, &mtype, &stype, &rtype);
          if (len == 1) {
            if (dtype == AMS_INT)         fprintf(fd," %d",*(int*)addr);
            else if (dtype == AMS_STRING) fprintf(fd," %s",*(char**)addr);
          }
          fprintf(fd,"</li>\r\n");
          j++;
        }
        fprintf(fd,"</ul>\r\n");
        i++;
      }
      fprintf(fd,"</ul>\r\n");
    }
  }
  ierr = PetscWebSendFooter(fd);CHKERRQ(ierr);
  ierr = AMS_Disconnect();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscAMSObjectsDisplayTree"
/*
    Displays all the PETSc objects published with SAWs in very crude HTML 5 graphics

    Does NOT use Javascript or JSON-RPC
*/
static PetscErrorCode PetscAMSObjectsDisplayTree(FILE *fd)
{
  PetscErrorCode     ierr;
  char               host[256],**comm_list,**mem_list,**fld_list;
  AMS_Comm           ams;
  PetscInt           i = 0,j;
  AMS_Memory_type    mtype;
  AMS_Data_type      dtype;
  AMS_Shared_type    stype;
  AMS_Reduction_type rtype;
  AMS_Memory         memory;
  int                len;
  void               *addr2,*addr3,*addr,*addr4;
  PetscBool          stack;

  ierr = PetscGetHostName(host,256);CHKERRQ(ierr);
  ierr = AMS_Connect(host, -1, &comm_list);
  ierr = PetscWebSendHeader(fd, 200, "OK", NULL, "text/html", -1);CHKERRQ(ierr);
  if (!comm_list || !comm_list[0]) fprintf(fd, "AMS Communicator not running</p>\r\n");
  else {
    ierr = AMS_Comm_attach(comm_list[0],&ams);
    ierr = AMS_Comm_get_memory_list(ams,&mem_list);
    if (!mem_list[0]) fprintf(fd, "AMS Communicator %s has no published memories</p>\r\n",comm_list[0]);
    else {
      PetscInt  Nlevels,*Level,*Levelcnt,*Idbylevel,*Column,*parentid,*Id,maxId = 0,maxCol = 0,*parentId,id,cnt,Nlevelcnt = 0;
      PetscBool *mask;
      char      **classes,*clas,**subclasses,*sclas;

      /* get maximum number of objects */
      while (mem_list[i]) {
        ierr = PetscStrcmp(mem_list[i],"Stack",&stack);CHKERRQ(ierr);
        if (stack) {i++; continue;}
        ierr = AMS_Memory_attach(ams,mem_list[i],&memory,NULL);
        ierr = AMS_Memory_get_field_list(memory, &fld_list);
        ierr = AMS_Memory_get_field_info(memory, "Id", &addr2, &len, &dtype, &mtype, &stype, &rtype);
        Id    = (int*) addr2;
        maxId = PetscMax(maxId,*Id);
        i++;
      }
      maxId++;

      /* Gets everyone's parent ID and which nodes are masked */
      ierr = PetscMalloc4(maxId,PetscInt,&parentid,maxId,PetscBool,&mask,maxId,char**,&classes,maxId,char**,&subclasses);CHKERRQ(ierr);
      ierr = PetscMemzero(classes,maxId*sizeof(char*));CHKERRQ(ierr);
      ierr = PetscMemzero(subclasses,maxId*sizeof(char*));CHKERRQ(ierr);
      for (i=0; i<maxId; i++) mask[i] = PETSC_TRUE;
      i = 0;
      while (mem_list[i]) {
        ierr = PetscStrcmp(mem_list[i],"Stack",&stack);CHKERRQ(ierr);
        if (stack) {i++; continue;}
        ierr = AMS_Memory_attach(ams,mem_list[i],&memory,NULL);
        ierr = AMS_Memory_get_field_list(memory, &fld_list);
        ierr = AMS_Memory_get_field_info(memory, "Id", &addr2, &len, &dtype, &mtype, &stype, &rtype);
        Id            = (int*) addr2;
        ierr = AMS_Memory_get_field_info(memory, "ParentId", &addr3, &len, &dtype, &mtype, &stype, &rtype);
        parentId      = (int*) addr3;
        ierr = AMS_Memory_get_field_info(memory, "Class", &addr, &len, &dtype, &mtype, &stype, &rtype);
        clas          = *(char**)addr;
        ierr = AMS_Memory_get_field_info(memory, "Type", &addr4, &len, &dtype, &mtype, &stype, &rtype);
        sclas         = *(char**)addr4;
        parentid[*Id] = *parentId;
        mask[*Id]     = PETSC_FALSE;

        ierr = PetscStrallocpy(clas,classes+*Id);CHKERRQ(ierr);
        ierr = PetscStrallocpy(sclas,subclasses+*Id);CHKERRQ(ierr);
        i++;
      }

      /* if the parent is masked then relabel the parent as 0 since the true parent was deleted */
      for (i=0; i<maxId; i++) {
        if (!mask[i] && parentid[i] > 0 && mask[parentid[i]]) parentid[i] = 0;
      }

      ierr = PetscProcessTree(maxId,mask,parentid,&Nlevels,&Level,&Levelcnt,&Idbylevel,&Column);CHKERRQ(ierr);

      for (i=0; i<Nlevels; i++) maxCol    = PetscMax(maxCol,Levelcnt[i]);
      for (i=0; i<Nlevels; i++) Nlevelcnt = PetscMax(Nlevelcnt,Levelcnt[i]);

      /* print all the top-level objects */
      fprintf(fd, "<HTML><HEAD><TITLE>Petsc Application Server</TITLE>\r\n");
      fprintf(fd, "<canvas width=800 height=600 id=\"tree\"></canvas>\r\n");
      fprintf(fd, "<script type=\"text/javascript\">\r\n");
      fprintf(fd, "  function draw() {\r\n");
      fprintf(fd, "  var example = document.getElementById('tree');\r\n");
      fprintf(fd, "  var context = example.getContext('2d');\r\n");
      /* adjust font size based on how big a tree is printed */
      if (Nlevels > 5 || Nlevelcnt > 10) fprintf(fd, "  context.font         = \"normal 12px sans-serif\";\r\n");
      else                               fprintf(fd, "  context.font         = \"normal 24px sans-serif\";\r\n");
      fprintf(fd, "  context.fillStyle = \"rgb(255,0,0)\";\r\n");
      fprintf(fd, "  context.textBaseline = \"top\";\r\n");
      fprintf(fd, "  var xspacep = 0;\r\n");
      fprintf(fd, "  var yspace = example.height/%d;\r\n",(Nlevels+1));
      /* estimate the height of a string as twice the width of a character */
      fprintf(fd, "  var wheight = context.measureText(\"K\");\r\n");
      fprintf(fd, "  var height = 1.6*wheight.width;\r\n");

      cnt = 0;
      for (i=0; i<Nlevels; i++) {
        fprintf(fd, "  var xspace = example.width/%d;\r\n",Levelcnt[i]+1);
        for (j=0; j<Levelcnt[i]; j++) {
          id    = Idbylevel[cnt++];
          clas  = classes[id];
          sclas = subclasses[id];
          fprintf(fd, "  var width = context.measureText(\"%s\");\r\n",clas);
          fprintf(fd, "  var swidth = context.measureText(\"%s\");\r\n",sclas);
          fprintf(fd, "  context.fillStyle = \"rgb(255,0,0)\";\r\n");
          fprintf(fd, "  context.fillRect((%d)*xspace-width.width/2, %d*yspace-height/2, width.width, height);\r\n",j+1,i+1);
          fprintf(fd, "  context.fillRect((%d)*xspace-swidth.width/2, %d*yspace+height/2, swidth.width, height);\r\n",j+1,i+1);
          fprintf(fd, "  context.fillStyle = \"rgb(0,0,0)\";\r\n");
          fprintf(fd, "  context.fillText(\"%s\",(%d)*xspace-width.width/2, %d*yspace-height/2);\r\n",clas,j+1,i+1);
          fprintf(fd, "  context.fillText(\"%s\",(%d)*xspace-swidth.width/2, %d*yspace+height/2);\r\n",sclas,j+1,i+1);
          if (parentid[id]) {
            fprintf(fd, "  context.moveTo(%d*xspace,%d*yspace-height/2);\r\n",j+1,i+1);
            fprintf(fd, "  context.lineTo(%d*xspacep,%d*yspace+3*height/2);\r\n",Column[parentid[id]]+1,i);
            fprintf(fd, "  context.stroke();\r\n");
          }
        }
        fprintf(fd, "  xspacep = xspace;\r\n");
      }
      ierr = PetscFree(Level);CHKERRQ(ierr);
      ierr = PetscFree(Levelcnt);CHKERRQ(ierr);
      ierr = PetscFree(Idbylevel);CHKERRQ(ierr);
      ierr = PetscFree(Column);CHKERRQ(ierr);
      for (i=0; i<maxId; i++) {
        ierr = PetscFree(classes[i]);CHKERRQ(ierr);
        ierr = PetscFree(subclasses[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree4(mask,parentid,classes,subclasses);CHKERRQ(ierr);

      ierr = AMS_Disconnect();
      fprintf(fd, "}\r\n");
      fprintf(fd, "</script>\r\n");
      fprintf(fd, "<body onload=\"draw();\">\r\n");
      fprintf(fd, "</body></html>\r\n");
    }
  }
  ierr = PetscWebSendFooter(fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebServeRequestGet"
/*@C
      PetscWebServeRequestGet - serves a single web Get request

    Not collective

  Input Parameters:
+   port - the network file to read and write from
-   path - the command from the server

    Level: developer

.seealso: PetscWebServe()
@*/
static PetscErrorCode  PetscWebServeRequestGet(FILE *fd,const char path[])
{
  PetscErrorCode ierr;
  FILE           *fdo;
  char           fullpath[PETSC_MAX_PATH_LEN],truefullpath[PETSC_MAX_PATH_LEN],*qmark;
  const char     *type;
  PetscBool      flg;

  PetscFunctionBegin;
  fseek(fd, 0, SEEK_CUR); /* Force change of stream direction */

  ierr = PetscStrcmp(path,"/favicon.ico",&flg);CHKERRQ(ierr);
  if (flg) {
    /* should have cool PETSc icon */;
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp(path,"/",&flg);CHKERRQ(ierr);
  if (flg) {
    char        program[128];
    PetscMPIInt size;
    PetscViewer viewer;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscGetProgramName(program,128);CHKERRQ(ierr);
    ierr = PetscWebSendHeader(fd, 200, "OK", NULL, "text/html", -1);CHKERRQ(ierr);
    fprintf(fd, "<HTML><HEAD><TITLE>Petsc Application Server</TITLE></HEAD>\r\n<BODY>");
    fprintf(fd, "<H4>Serving PETSc application code %s </H4>\r\n\n",program);
    fprintf(fd, "Number of processes %d\r\n\n",size);
    fprintf(fd, "<HR>\r\n");
    ierr = PetscViewerASCIIOpenWithFILE(PETSC_COMM_WORLD,fd,&viewer);CHKERRQ(ierr);
    ierr = PetscOptionsView(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    fprintf(fd, "<HR>\r\n");
    fprintf(fd, "<a href=\"./ams-tree\">View published PETSc objects -- As a static graphical tree</a></p>\r\n\r\n");
    fprintf(fd, "<a href=\"./ams-list\">View published PETSc objects -- As a static list</a></p>\r\n\r\n");
    fprintf(fd, "<a href=\"./AMSSnoopObjects.html\">Snoop on published PETSc objects --Interactive Javascript</a></p>\r\n\r\n");
    fprintf(fd, "<a href=\"./AMSSnoopStack.html\">Snoop on published PETSc stackframes --Interactive Javascript</a></p>\r\n\r\n");
    ierr = PetscWebSendFooter(fd);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscStrcmp(path,"/ams-list",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscAMSObjectsDisplayList(fd);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp(path,"/ams-tree",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscAMSObjectsDisplayTree(fd);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcpy(fullpath,"${PETSC_DIR}/include/web");CHKERRQ(ierr);
  ierr = PetscStrcat(fullpath,path);CHKERRQ(ierr);
  ierr = PetscInfo1(NULL,"Checking for file %s\n",fullpath);CHKERRQ(ierr);
  ierr = PetscStrstr(fullpath,"?",&qmark);CHKERRQ(ierr);
  if (qmark) *qmark = 0;
  ierr = PetscStrreplace(PETSC_COMM_SELF,fullpath,truefullpath,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  fdo  = fopen(truefullpath,"r");
  if (fdo) {
    PetscInt    length,index;
    char        data[4096];
    struct stat statbuf;
    int         n;
    const char  *suffixes[] = {".html",".js",".gif",0}, *mimes[] = {"text/html","text/javascript","image/gif","text/unknown"};

    ierr = PetscStrendswithwhich(fullpath,suffixes,&index);CHKERRQ(ierr);
    type = mimes[index];
    if (!stat(truefullpath, &statbuf)) length = -1;
    else length = S_ISREG(statbuf.st_mode) ? statbuf.st_size : -1;
    ierr = PetscWebSendHeader(fd, 200, "OK", NULL, type, length);CHKERRQ(ierr);
    while ((n = fread(data, 1, sizeof(data), fdo)) > 0) fwrite(data, 1, n, fd);
    fclose(fdo);
    ierr = PetscInfo2(NULL,"Sent file %s to browser using format %s\n",fullpath,type);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscWebSendError(fd, 501, "Not supported", NULL, "Unknown request.");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Toy YAML/JSON-RPC function that returns all the arguments it is passed
*/
#undef __FUNCT__
#define __FUNCT__ "YAML_echo"
PETSC_UNUSED static PetscErrorCode YAML_echo(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode ierr;
  PetscInt       i;

  ierr = PetscPrintf(PETSC_COMM_SELF,"Number of arguments to function %d\n",argc);CHKERRQ(ierr);
  for (i=0; i<argc; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"  %s\n",args[i]);CHKERRQ(ierr);
  }
  *argco = argc;
  ierr   = PetscMalloc(argc*sizeof(char*),argso);CHKERRQ(ierr);
  for (i=0; i<argc; i++) {
    ierr = PetscStrallocpy(args[i],&(*argso)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------
     The following set of functions are wrapper functions for SAWs functions that

    1)  convert from string arguments to appropriate AMS arguments (int, double, char*, etc)
    2)  call the AMS function
    3)  convert from the AMS result arguments to string arguments

    Developers Note: Rather than having PetscProcessYAMLRPC() convert the YAML/JSON representation of the params to an array of strings
       it may be better to simple pass those YAML/JSON strings to these routines and have them pull out the values from the YAML/JSON
       Similarly these routines could put their result directly back into YAML/JSON rather than putting them into an array of strings
       returning that and having PetscProcessYAMLRPC() put them into the YAML/JSON.
*/

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Utility_StringToArray"
static PetscErrorCode YAML_AMS_Utility_StringToArray(const char *instring,AMS_Data_type dtype,PetscInt *n,void **addr)
{
  PetscErrorCode ierr;
  char           *bracket,*sub;
  PetscInt       N;
  PetscToken     token;
  char           *string,*cstring;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(instring,&cstring);CHKERRQ(ierr);
  ierr = PetscStrchr(instring,'[',&bracket);CHKERRQ(ierr);
  if (bracket) {
    string = bracket + 1;
    ierr = PetscStrchr(instring,']',&bracket);CHKERRQ(ierr);
    if (bracket) *bracket = 0;
  } else string = cstring;

  N = 0;
  ierr = PetscTokenCreate(string,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&sub);CHKERRQ(ierr);
  while (sub) {
    ierr = PetscTokenFind(token,&sub);CHKERRQ(ierr);
    N++;
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  ierr = PetscInfo2(NULL,"String value %s number of entries in array %d\n",string,(int)N);CHKERRQ(ierr);
  ierr = PetscTokenCreate(string,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&sub);CHKERRQ(ierr);

  if (dtype == AMS_STRING) {
    char **caddr;
    ierr = PetscMalloc(N*sizeof(char*),&caddr);CHKERRQ(ierr);
    *addr = (void*) caddr;
    while (sub) {
      ierr = PetscStrallocpy(sub,(char**)caddr);CHKERRQ(ierr);
      ierr = PetscInfo2(NULL,"String value %s, computed value %s\n",sub,*caddr);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&sub);CHKERRQ(ierr);
      caddr++;
    }
  } else if (dtype == AMS_BOOLEAN) {
    PetscBool *baddr;
    ierr = PetscMalloc(N*sizeof(PetscBool),&baddr);CHKERRQ(ierr);
    *addr = (void*) baddr;
    while (sub) {
      ierr = PetscOptionsStringToBool(sub,baddr);CHKERRQ(ierr);
      ierr = PetscInfo2(NULL,"String value %s, computed value %d\n",sub,(int)*baddr);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&sub);CHKERRQ(ierr);
      baddr++;
    }
  } else if (dtype == AMS_INT) {
    PetscInt *iaddr;
    ierr = PetscMalloc(N*sizeof(PetscInt),&iaddr);CHKERRQ(ierr);
    *addr = (void*) iaddr;
    while (sub) {
      sscanf(sub,"%d",iaddr);
      ierr = PetscInfo2(NULL,"String value %s, computed value %d\n",sub,(int)*iaddr);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&sub);CHKERRQ(ierr);
      iaddr++;
    }
  } else if (dtype == AMS_DOUBLE) {
    PetscReal *raddr;
    ierr = PetscMalloc(N*sizeof(PetscReal),&raddr);CHKERRQ(ierr);
    *addr = (void*) raddr;
    while (sub) {
      sscanf(sub,"%le",(double*)raddr);
      ierr = PetscInfo2(NULL,"String value %s, computed value %g\n",sub,(double)*raddr);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&sub);CHKERRQ(ierr);
      raddr++;
    }
  } else {
    ierr = PetscInfo1(NULL,"String value %s, datatype not handled\n",string);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  ierr = PetscFree(cstring);CHKERRQ(ierr);
  *n   = N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Utility_ArrayToString"
static PetscErrorCode YAML_AMS_Utility_ArrayToString(PetscInt n,void *addr,AMS_Data_type dtype,char **result)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!n) {
    ierr = PetscStrallocpy("null",result);CHKERRQ(ierr);
  } else if (n == 1) {
    if (dtype == AMS_STRING) {
      ierr = PetscStrallocpy(*(const char**)addr,result);CHKERRQ(ierr);
    } else if (dtype == AMS_DOUBLE) {
      ierr = PetscMalloc(20*sizeof(char),result);CHKERRQ(ierr);
      sprintf(*result,"%18.16e",*(double*)addr);
    } else if (dtype == AMS_INT) {
      ierr = PetscMalloc(10*sizeof(char),result);CHKERRQ(ierr);
      sprintf(*result,"%d",*(int*)addr);
    } else if (dtype == AMS_BOOLEAN) {
      if (*(PetscBool*)addr) {
        ierr = PetscStrallocpy("true",result);CHKERRQ(ierr);
      } else {
        ierr = PetscStrallocpy("false",result);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscStrallocpy("Not yet done",result);CHKERRQ(ierr);
    }
  } else {
    PetscInt i;
    size_t   len = 0,lent;
    char     buff[25],**array = (char**)addr;

    if (dtype == AMS_STRING) {
      for (i=0; i<n; i++) {
        ierr = PetscStrlen(array[i],&lent);CHKERRQ(ierr);
        len += lent + 10;
      }
      ierr = PetscMalloc(len*sizeof(char),result);CHKERRQ(ierr);
      ierr = PetscStrcpy(*result,"[\"");CHKERRQ(ierr);
      for (i=0; i<n-1; i++) {
        ierr = PetscStrcat(*result,array[i]);CHKERRQ(ierr);
        ierr = PetscStrcat(*result,"\",\"");CHKERRQ(ierr);
      }
      ierr = PetscStrcat(*result,array[n-1]);CHKERRQ(ierr);
      ierr = PetscStrcat(*result,"\"]");CHKERRQ(ierr);
    } else if (dtype == AMS_DOUBLE) {
      ierr = PetscMalloc(30*n*sizeof(char),result);CHKERRQ(ierr);
      ierr = PetscStrcpy(*result,"[\"");CHKERRQ(ierr);
      for (i=0; i<n-1; i++) {
        sprintf(buff,"%18.16e",*(double*)addr);
        ierr = PetscStrcat(*result,buff);CHKERRQ(ierr);
        ierr = PetscStrcat(*result,"\",\"");CHKERRQ(ierr);
        addr = (void *) ((char *)addr + sizeof(PetscReal));
      }
      sprintf(buff,"%18.16e",*(double*)addr);
      ierr = PetscStrcat(*result,buff);CHKERRQ(ierr);
      ierr = PetscStrcat(*result,"\"]");CHKERRQ(ierr);
    } else if (dtype == AMS_INT) {
      ierr = PetscMalloc(13*n*sizeof(char),result);CHKERRQ(ierr);
      ierr = PetscStrcpy(*result,"[\"");CHKERRQ(ierr);
      for (i=0; i<n-1; i++) {
        sprintf(buff,"%d",*(int*)addr);
        ierr = PetscStrcat(*result,buff);CHKERRQ(ierr);
        ierr = PetscStrcat(*result,"\",\"");CHKERRQ(ierr);
        addr = (void *) ((char *)addr + sizeof(PetscInt));
      }
      sprintf(buff,"%d",*(int*)addr);
      ierr = PetscStrcat(*result,buff);CHKERRQ(ierr);
      ierr = PetscStrcat(*result,"\"]");CHKERRQ(ierr);
    } else if (dtype == AMS_BOOLEAN) {
      ierr = PetscMalloc(7*n*sizeof(char),result);CHKERRQ(ierr);
      ierr = PetscStrcpy(*result,"[");CHKERRQ(ierr);
      for (i=0; i<n-1; i++) {
        ierr = PetscStrcat(*result,*(PetscBool*)addr ? "\"true\"" : "\"false\"");CHKERRQ(ierr);
        ierr = PetscStrcat(*result,",");CHKERRQ(ierr);
        addr = (void *) ((char *)addr + sizeof(int));
      }
      ierr = PetscStrcat(*result,*(PetscBool*)addr ? "\"true\"" : "\"false\"");CHKERRQ(ierr);
      ierr = PetscStrcat(*result,"]");CHKERRQ(ierr);
    } else {
      ierr = PetscStrallocpy("Not yet done",result);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Utility_Error"
static PetscErrorCode YAML_AMS_Utility_Error(PetscInt ier,const char *message,char **err)
{
  PetscErrorCode ierr;
  char           fullmess[128];

  PetscFunctionBegin;
  ierr = PetscInfo2(NULL,"%s Error code %d\n",message,(int)ier);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fullmess,128,"{ \"code\": \"%d\", \"message\": \"%s\", \"data\": null }",1+(int)ier,message);CHKERRQ(ierr);
  ierr = PetscStrallocpy(fullmess,err);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Connect"
/*
      Connects to the local AMS and gets communicator names

   Input Parameters:
.     none

   Output Parameter:
.     oarg1 - the string name of the first communicator

*/
PETSC_EXTERN PetscErrorCode YAML_AMS_Connect(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode ierr;
  char           host[256],**list = 0;
  PetscInt       n = 0;

  PetscFunctionBegin;
  *argco = 0;
  ierr = PetscGetHostName(host,256);CHKERRQ(ierr);
  ierr = AMS_Connect(host,-1,&list);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Connect()",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (!list) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Connect() list empty, not running AMS server",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  *argco = 1;
  ierr   = PetscMalloc(sizeof(char*),argso);CHKERRQ(ierr);
  while (list[n]) n++;
  ierr = YAML_AMS_Utility_ArrayToString(n,list,AMS_STRING,&(*argso)[0]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Comm_attach"
/*
      Attaches to an AMS communicator

   Input Parameter:
.     arg1 - string name of the communicator

   Output Parameter:
.     oarg1 - the integer name of the communicator

*/
PETSC_EXTERN PetscErrorCode YAML_AMS_Comm_attach(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode ierr;
  AMS_Comm       comm = -1;

  PetscFunctionBegin;
  *argco = 0;
  ierr = AMS_Comm_attach(args[0],&comm);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Comm_attach()",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  *argco = 1;
  ierr   = PetscMalloc(sizeof(char*),argso);CHKERRQ(ierr);
  ierr   = PetscMalloc(3*sizeof(char*),&argso[0][0]);CHKERRQ(ierr);
  sprintf(argso[0][0],"%d",(int)comm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Comm_get_memory_list"
/*
      Gets the list of memories on an AMS Comm

   Input Parameter:
.     arg1 - integer name of the communicator

   Output Parameter:
.     oarg1 - the list of names

*/
PETSC_EXTERN PetscErrorCode YAML_AMS_Comm_get_memory_list(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode ierr;
  char           **mem_list;
  AMS_Comm       comm;
  PetscInt       i,iargco = 0;

  PetscFunctionBegin;
  *argco = 0;
  sscanf(args[0],"%d",&comm);
  ierr = AMS_Comm_get_memory_list(comm,&mem_list);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Comm_get_memory_list()",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    while (mem_list[iargco++]) ;
    iargco--;

    ierr = PetscMalloc((iargco)*sizeof(char*),argso);CHKERRQ(ierr);
    for (i=0; i<iargco; i++) {
      ierr = PetscStrallocpy(mem_list[i],(*argso)+i);CHKERRQ(ierr);
    }
  }
  *argco = iargco;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Memory_attach"
/*
      Attaches to an AMS memory in a communicator

   Input Parameter:
.     arg1 - communicator
.     arg2 - string name of the memory

   Output Parameter:
.     oarg1 - the integer name of the memory
.     oarg2 - the integer step of the memory

*/
PETSC_EXTERN PetscErrorCode YAML_AMS_Memory_attach(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode ierr;
  AMS_Comm       comm;
  AMS_Memory     mem;
  unsigned int   step;

  PetscFunctionBegin;
  *argco = 0;
  sscanf(args[0],"%d",&comm);
  ierr = AMS_Memory_attach(comm,args[1],&mem,&step);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Memory_attach()",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  *argco = 2;
  ierr   = PetscMalloc(2*sizeof(char*),argso);CHKERRQ(ierr);
  ierr   = PetscMalloc(3*sizeof(char*),&argso[0][0]);CHKERRQ(ierr);
  sprintf(argso[0][0],"%d",(int)mem);
  ierr = PetscMalloc(3*sizeof(char*),&argso[0][1]);CHKERRQ(ierr);
  sprintf(argso[0][1],"%d",(int)step);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Memory_get_field_list"
/*
      Gets the list of fields on an AMS Memory

   Input Parameter:
.     arg1 - integer name of the memory

   Output Parameter:
.     oarg1 - the list of names

*/
PETSC_EXTERN PetscErrorCode YAML_AMS_Memory_get_field_list(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode ierr;
  char           **field_list;
  AMS_Memory     mem;
  PetscInt       i,iargco = 0;

  PetscFunctionBegin;
  *argco = 0;
  sscanf(args[0],"%d",&mem);
  ierr = AMS_Memory_get_field_list(mem,&field_list);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Memory_get_field_list()",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    while (field_list[iargco++]) ;
    iargco--;

    ierr = PetscMalloc((iargco)*sizeof(char*),argso);CHKERRQ(ierr);
    for (i=0; i<iargco; i++) {
      ierr = PetscStrallocpy(field_list[i],(*argso)+i);CHKERRQ(ierr);
    }
  }
  *argco = iargco;
  PetscFunctionReturn(0);
}

const char *AMS_Data_types[] = {"AMS_DATA_UNDEF","AMS_BOOLEAN","AMS_INT","AMS_FLOAT","AMS_DOUBLE","AMS_STRING","AMS_Data_type","AMS_",0};
const char *AMS_Memory_types[] = {"AMS_MEMORY_UNDEF","AMS_READ","AMS_WRITE","AMS_Memory_type","AMS_",0};
const char *AMS_Shared_types[] = {"AMS_SHARED_UNDEF","AMS_COMMON","AMS_REDUCED","AMS_DISTRIBUTED","AMS_Shared_type","AMS_",0};
const char *AMS_Reduction_types[] = {"AMS_REDUCTION_WHY_NOT_UNDEF?","AMS_SUM","AMS_MAX","AMS_MIN","AMS_REDUCTION_UNDEF","AMS_Reduction_type","AMS_",0};


#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Memory_get_field_info"
/*
      Gets information about a field

   Input Parameter:
.     arg1 - memory
.     arg2 - string name of the field

   Output Parameter:

*/
PETSC_EXTERN PetscErrorCode YAML_AMS_Memory_get_field_info(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode     ierr;
  AMS_Memory         mem;
  char               *addr;
  int                len;
  AMS_Data_type      dtype;
  AMS_Memory_type    mtype;
  AMS_Shared_type    stype;
  AMS_Reduction_type rtype;

  PetscFunctionBegin;
  *argco = 0;
  sscanf(args[0],"%d",&mem);
  ierr = AMS_Memory_get_field_info(mem,args[1],(void**)&addr,&len,&dtype,&mtype,&stype,&rtype);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Memory_get_field_info() ",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  *argco = 5;
  ierr   = PetscMalloc((*argco)*sizeof(char*),argso);CHKERRQ(ierr);
  ierr   = PetscStrallocpy(AMS_Data_types[dtype],&argso[0][0]);CHKERRQ(ierr);
  ierr   = PetscStrallocpy(AMS_Memory_types[mtype],&argso[0][1]);CHKERRQ(ierr);
  ierr   = PetscStrallocpy(AMS_Shared_types[stype],&argso[0][2]);CHKERRQ(ierr);
  ierr   = PetscStrallocpy(AMS_Reduction_types[rtype],&argso[0][3]);CHKERRQ(ierr);
  ierr   = YAML_AMS_Utility_ArrayToString(len,addr,dtype,&argso[0][4]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Memory_set_field_info"
/*
      Gets information about a field

   Input Parameter:
.     arg1 - memory
.     arg2 - string name of the field

   Output Parameter:

*/
PETSC_EXTERN PetscErrorCode YAML_AMS_Memory_set_field_info(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode     ierr;
  AMS_Memory         mem;
  void               *addr;
  int                len,newlen;
  AMS_Data_type      dtype;
  AMS_Memory_type    mtype;
  AMS_Shared_type    stype;
  AMS_Reduction_type rtype;

  PetscFunctionBegin;
  *argco = 0;
  sscanf(args[0],"%d",&mem);
  ierr = AMS_Memory_get_field_info(mem,args[1],(void**)&addr,&len,&dtype,&mtype,&stype,&rtype);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Memory_set_field_info() Memory field can not be located",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (mtype == AMS_READ) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Memory_set_field_info() Memory field is read only",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = YAML_AMS_Utility_StringToArray(args[2],dtype,&newlen,(void**)&addr);CHKERRQ(ierr);
  if (newlen != len) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Memory_set_field_info() Changing array length",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = AMS_Memory_set_field_info(mem,args[1],addr,len);CHKERRQ(ierr);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Memory_set_field_info() ",err);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  *argco = 1;
  ierr   = PetscMalloc(sizeof(char*),argso);CHKERRQ(ierr);
  ierr   = PetscStrallocpy("Memory field value set",*argso);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "YAML_AMS_Memory_update_send_begin"
PETSC_EXTERN PetscErrorCode YAML_AMS_Memory_update_send_begin(PetscInt argc,char **args,PetscInt *argco,char ***argso,char **err)
{
  PetscErrorCode     ierr;
  AMS_Memory         mem;

  PetscFunctionBegin;
  *argco = 0;
  sscanf(args[0],"%d",&mem);
  ierr = AMS_Memory_update_send_begin(mem);
  if (ierr) {
    ierr = YAML_AMS_Utility_Error(ierr,"AMS_Memory_update_send_begin() ",err);CHKERRQ(ierr);
  }
  *argco = 0;
  PetscFunctionReturn(0);
}

#include "yaml.h"
#undef __FUNCT__
#define __FUNCT__ "PetscProcessYAMLRPC"
/*
     1) Parses a YAML/JSON-RPC function call generating a function name for an AMS wrapper function and the arguments to the function
     2) loads the function with dlsym(),
     3) calls the wrapper function with the arguments
     4) converts the result arguments back to YAML/JSON.
*/
static PetscErrorCode PetscProcessYAMLRPC(const char *request,char **result)
{
  yaml_parser_t  parser;
  yaml_event_t   event;
  int            done  = 0;
  int            count = 0;
  size_t         len;
  PetscErrorCode ierr;
  PetscBool      method,params,id;
  char           *methodname,*idname,**args,**argso = 0,*err = 0;
  PetscInt       argc = 0,argco,i;
  PetscErrorCode (*fun)(PetscInt,char**,PetscInt*,char***,char**);

  PetscFunctionBegin;
  ierr = PetscMalloc(20*sizeof(char*),&args);CHKERRQ(ierr);
  yaml_parser_initialize(&parser);
  PetscStrlen(request,&len);
  yaml_parser_set_input_string(&parser, (unsigned char*)request, len);

  /* this is totally bogus; it only handles the simple JSON-RPC messages */
  while (!done) {
    if (!yaml_parser_parse(&parser, &event)) {
      ierr = PetscInfo(NULL,"Found error in yaml/json\n");CHKERRQ(ierr);
      break;
    }
    done = (event.type == YAML_STREAM_END_EVENT);
    switch (event.type) {
    case YAML_STREAM_START_EVENT:
      ierr = PetscInfo(NULL,"Stream start\n");CHKERRQ(ierr);
      break;
    case YAML_STREAM_END_EVENT:
      ierr = PetscInfo(NULL,"Stream end\n");CHKERRQ(ierr);
      break;
    case YAML_DOCUMENT_START_EVENT:
      ierr = PetscInfo(NULL,"Document start\n");CHKERRQ(ierr);
      break;
    case YAML_DOCUMENT_END_EVENT:
      ierr = PetscInfo(NULL,"Document end\n");CHKERRQ(ierr);
      break;
    case YAML_MAPPING_START_EVENT:
      ierr = PetscInfo(NULL,"Mapping start event\n");CHKERRQ(ierr);
      break;
    case YAML_MAPPING_END_EVENT:
      ierr = PetscInfo(NULL,"Mapping end event \n");CHKERRQ(ierr);
      break;
    case YAML_ALIAS_EVENT:
      ierr = PetscInfo1(NULL,"Alias event %s\n",event.data.alias.anchor);CHKERRQ(ierr);
      break;
    case YAML_SCALAR_EVENT:
      ierr = PetscInfo1(NULL,"Scalar event %s\n",event.data.scalar.value);CHKERRQ(ierr);
      ierr = PetscStrcmp((char*)event.data.scalar.value,"method",&method);CHKERRQ(ierr);
      ierr = PetscStrcmp((char*)event.data.scalar.value,"params",&params);CHKERRQ(ierr);
      ierr = PetscStrcmp((char*)event.data.scalar.value,"id",&id);CHKERRQ(ierr);
      if (method) {
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        ierr = PetscInfo1(NULL,"Method %s\n",event.data.scalar.value);CHKERRQ(ierr);
        ierr = PetscStrallocpy((char*)event.data.scalar.value,&methodname);CHKERRQ(ierr);
      } else if (id) {
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        ierr = PetscInfo1(NULL,"Id %s\n",event.data.scalar.value);CHKERRQ(ierr);
        ierr = PetscStrallocpy((char*)event.data.scalar.value,&idname);CHKERRQ(ierr);
      } else if (params) {
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        while (event.type != YAML_SEQUENCE_END_EVENT) {
          ierr = PetscInfo1(NULL,"  Parameter %s\n",event.data.scalar.value);CHKERRQ(ierr);
          ierr = PetscStrallocpy((char*)event.data.scalar.value,&args[argc++]);CHKERRQ(ierr);
          yaml_event_delete(&event);
          ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
        }
      } else { /* ignore all the other variables in the mapping */
        yaml_event_delete(&event);
        ierr = yaml_parser_parse(&parser, &event);CHKERRQ(!ierr);
      }
      break;
    case YAML_SEQUENCE_START_EVENT:
      ierr = PetscInfo(NULL,"Sequence start event \n");CHKERRQ(ierr);
      break;
    case YAML_SEQUENCE_END_EVENT:
      ierr = PetscInfo(NULL,"Sequence end event \n");CHKERRQ(ierr);
      break;
    default:
      /* It couldn't really happen. */
      break;
    }

    yaml_event_delete(&event);
    count++;
  }
  yaml_parser_delete(&parser);

  ierr = PetscDLLibrarySym(PETSC_COMM_SELF,NULL,NULL,methodname,(void**)&fun);CHKERRQ(ierr);
  if (fun) {
    ierr = PetscInfo1(NULL,"Located function %s and running it\n",methodname);CHKERRQ(ierr);
    ierr = (*fun)(argc,args,&argco,&argso,&err);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo1(NULL,"Did not locate function %s skipping it\n",methodname);CHKERRQ(ierr);
  }

  for (i=0; i<argc; i++) {
    ierr = PetscFree(args[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(args);CHKERRQ(ierr);
  ierr = PetscFree(methodname);CHKERRQ(ierr);

  /* convert the result back to YAML/JSON; should use YAML/JSON encoder, does not handle zero return arguments */
  ierr = PetscMalloc(16000,result);CHKERRQ(ierr);
  ierr = PetscStrcpy(*result,"{\"jsonrpc\": \"2.0\", ");CHKERRQ(ierr);
  if (err) {
    ierr = PetscStrcat(*result,"\"error\": ");CHKERRQ(ierr);
    ierr = PetscStrcat(*result,err);CHKERRQ(ierr);
    ierr = PetscStrcat(*result,",");CHKERRQ(ierr);
  } else {
    ierr = PetscStrcat(*result,"\"error\": null,");CHKERRQ(ierr);
  }
  ierr = PetscStrcat(*result," \"id\": \"");CHKERRQ(ierr);
  ierr = PetscStrcat(*result,idname);CHKERRQ(ierr);
  if (err) {
    ierr = PetscStrcat(*result,"\", \"result\" : null");CHKERRQ(ierr);
  } else {
    ierr = PetscStrcat(*result,"\", \"result\" : ");CHKERRQ(ierr);
    if (!argco) {ierr = PetscStrcat(*result,"null");CHKERRQ(ierr);}
    if (argco > 1) {ierr = PetscStrcat(*result,"[");CHKERRQ(ierr);}
    for (i=0; i<argco; i++) {
      if (argso[i][0] != '[') {
        ierr = PetscStrcat(*result,"\"");CHKERRQ(ierr);
      }
      ierr = PetscStrcat(*result,argso[i]);CHKERRQ(ierr);
      if (argso[i][0] != '[') {
        ierr = PetscStrcat(*result,"\"");CHKERRQ(ierr);
      }
      if (i < argco-1) {ierr = PetscStrcat(*result,",");CHKERRQ(ierr);}
    }
    if (argco > 1) {ierr = PetscStrcat(*result,"]");CHKERRQ(ierr);}
  }
  ierr = PetscFree(err);CHKERRQ(ierr);
  ierr = PetscStrcat(*result,"}");CHKERRQ(ierr);
  ierr = PetscInfo1(NULL,"YAML/JSON result of function %s\n",*result);CHKERRQ(ierr);

  /* free work space */
  ierr = PetscFree(idname);CHKERRQ(ierr);
  for (i=0; i<argco; i++) {
    ierr = PetscFree(argso[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(argso);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebServeRequestPostAMSJSONRPC"
/*@C
      PetscWebServeRequestPostAMSJSONRPC - serves a single web POST request based on JSON-RPC

       This function allows a Javascript program (running in the browser) to make an AMS function 
       call via JSON-RPC

       The currently available Javascript programs are in ${PETSC_DIR}/include/web

    Not collective

  Input Parameters:
.   fd - the network file to read and write from
-   path - the command from the server

    Level: developer

.seealso: PetscWebServe()
@*/
static PetscErrorCode  PetscWebServeRequestPostAMSJSONRPC(FILE *fd,const char path[])
{
  PetscErrorCode ierr;
  char           buf[16000];
  char           *result;
  int            len = -1;
  size_t         elen;
  char           *fnd;

  PetscFunctionBegin;
  while (PETSC_TRUE) {
    if (!fgets(buf, sizeof(buf), fd)) {
      ierr = PetscInfo(NULL,"Cannot read POST data, giving up\n");CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscInfo1(NULL,"POSTED header: %s",buf);CHKERRQ(ierr);
    ierr = PetscStrstr(buf,"Content-Type:",&fnd);CHKERRQ(ierr);
    if (fnd) {
      ierr = PetscStrstr(buf,"application/json-rpc",&fnd);CHKERRQ(ierr);
      if (!fnd) {
        ierr = PetscInfo(NULL,"POSTED content is not json-rpc, skipping post\n");CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
    }
    ierr = PetscStrstr(buf,"Content-Length:",&fnd);CHKERRQ(ierr);
    if (fnd) {
      sscanf(buf,"Content-Length: %d\n",&len);
      ierr = PetscInfo1(NULL,"POSTED Content-Length: %d\n",len);CHKERRQ(ierr);
    }
    if (buf[0] == '\r') break;
  }
  if (len == -1) {
    ierr = PetscInfo(NULL,"Did not find POST Content-Length in header, giving up\n");CHKERRQ(ierr);
  }

  if (!fgets(buf, len+1, fd)) { /* why is this len + 1? */
    ierr = PetscInfo(NULL,"Cannot read POST data, giving up\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscInfo1(NULL,"POSTED JSON/RPC request: %s\n",buf);CHKERRQ(ierr);
  fseek(fd, 0, SEEK_CUR); /* Force change of stream direction */
  ierr = PetscProcessYAMLRPC(buf,&result);CHKERRQ(ierr);
  ierr = PetscStrlen(result,&elen);CHKERRQ(ierr);
  ierr = PetscWebSendHeader(fd, 200, "OK", NULL, "application/json-rpc",(int)elen);CHKERRQ(ierr);
  fprintf(fd, "%s",result);
  ierr = PetscInfo(NULL,"Completed AMS JSON-RPC function call\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebServeRequest"
/*@C
      PetscWebServeRequest - serves a single web request

    Not collective

  Input Parameters:
.   port - the port

    Level: developer

.seealso: PetscWebServe()
@*/
static PetscErrorCode  PetscWebServeRequest(int port)
{
  PetscErrorCode ierr;
  FILE           *fd;
  char           buf[4096];
  char           *method, *path, *protocol;
  PetscBool      flg;
  PetscToken     tok;

  PetscFunctionBegin;
  fd = fdopen(port, "r+");

  ierr = PetscInfo(NULL,"Processing web request\n");CHKERRQ(ierr);
  if (!fgets(buf, sizeof(buf), fd)) {
    ierr = PetscInfo(NULL,"Cannot read web request, giving up\n");CHKERRQ(ierr);
    goto theend;
  }
  ierr = PetscInfo1(NULL,"Processing web request %s",buf);CHKERRQ(ierr);

  ierr = PetscTokenCreate(buf,' ',&tok);CHKERRQ(ierr);
  ierr = PetscTokenFind(tok,&method);CHKERRQ(ierr);
  ierr = PetscTokenFind(tok,&path);CHKERRQ(ierr);
  ierr = PetscTokenFind(tok,&protocol);CHKERRQ(ierr);

  if (!method || !path || !protocol) {
    ierr = PetscInfo(NULL,"Web request not well formatted, giving up\n");CHKERRQ(ierr);
    goto theend;
  }

  ierr = PetscStrcmp(method,"GET",&flg);
  if (flg) {
      ierr = PetscWebServeRequestGet(fd,path);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcmp(method,"POST",&flg);
    if (flg) {
      ierr = PetscWebServeRequestPostAMSJSONRPC(fd,path);CHKERRQ(ierr);
    } else {
      ierr = PetscWebSendError(fd, 501, "Not supported", NULL, "Method is not supported.");CHKERRQ(ierr);
      ierr = PetscInfo(NULL,"Web request not a GET or POST, giving up\n");CHKERRQ(ierr);
    }
  }
theend:
  ierr = PetscTokenDestroy(&tok);CHKERRQ(ierr);
  fclose(fd);
  ierr = PetscInfo1(NULL,"Finished processing request %s\n",method);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebServeWait"
/*@C
      PetscWebServeWait - waits for requests on a thread

    Not collective

  Input Parameter:
.   port - port to listen on

    Level: developer

.seealso: PetscViewerSocketOpen(), PetscWebServe()
@*/
PetscErrorCode PetscWebServeWait(int *port)
{
  PetscErrorCode ierr;
  int            iport,listenport,tport = *port;

  PetscFunctionBegin;
  ierr = PetscInfo1(NULL,"Starting webserver at port %d\n",tport);CHKERRQ(ierr);
  ierr = PetscSocketEstablish(tport,&listenport);CHKERRQ(ierr);
  while (1) {
    ierr = PetscSocketListen(listenport,&iport);CHKERRQ(ierr);
    ierr = PetscWebServeRequest(iport);CHKERRQ(ierr);
    close(iport);
  }
  close(listenport);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscWebServe"
/*@C
      PetscWebServe - start up the PETSc web server and respond to requests

    Not collective - only does something on process zero of the communicator

  Input Parameters:
+   comm - the MPI communicator
-   port - port to listen on

  Options Database Key:
+  -server <port> - start PETSc webserver (default port is 8080)
-  -xxx_view ams - publish object xxx to be accessible in the server


   Notes: Point your browser to http://hostname:8080   to access the PETSc web server, where hostname is the name of your machine.
      If you are running PETSc on your local machine you can use http://localhost:8080

      If the PETSc program completes before you connect with the browser you will not be able to connect to the PETSc webserver.

      Read the top of $PETSC_DIR/include/web/AMSSnoopObjects.py before running.

    Level: intermediate

.seealso: PetscViewerSocketOpen()
@*/
PetscErrorCode  PetscWebServe(MPI_Comm comm,int port)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (port < 1 && port != PETSC_DEFAULT && port != PETSC_DECIDE) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot use negative port number %d",port);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  if (port == PETSC_DECIDE || port == PETSC_DEFAULT) port = 8080;
  ierr = PetscWebServeWait(&port);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


