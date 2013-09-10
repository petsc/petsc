static char help[] = "Demonstrates PETSc path routines.\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "RealpathPhonyFile"
/* realpath(3) requires the path to exist, but GNU coreutils' realpath(1) only needs the containing directory to exist.
 * So split path into (dir, base) and only use realpath(3) on dir.
 *
 */
static PetscErrorCode RealpathPhonyFile(const char *path,char *buf,size_t len)
{
  char dir[PETSC_MAX_PATH_LEN],rpath[PETSC_MAX_PATH_LEN],*last;
  const char *base;
  size_t dlen;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(dir,path,sizeof dir);CHKERRQ(ierr);
  dir[sizeof dir-1] = 0;
  ierr = PetscStrlen(dir,&dlen);CHKERRQ(ierr);
  last = dir + dlen - 1;
  while (last > dir && *last == '/') *last-- = 0; /* drop trailing slashes */
  while (last > dir && *last != '/') last--;      /* seek backward to next slash */
  if (last > dir) {
    *last = 0;
    base = last + 1;
  } else {                      /* Current directory */
    dir[0] = '.';
    dir[1] = '\0';
    base = path;
  }
#if defined(PETSC_HAVE_REALPATH)
  if (!realpath(dir,rpath)) {
    perror("ex13: realpath");
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"realpath()");
  }
#else
  ierr = PetscStrncpy(rpath,dir,sizeof rpath);CHKERRQ(ierr);
  rpath[sizeof rpath-1] = 0;
#endif
  ierr = PetscStrlen(rpath,&dlen);CHKERRQ(ierr);
  ierr = PetscMemcpy(buf,rpath,PetscMin(dlen,len-1));CHKERRQ(ierr);
  buf[PetscMin(dlen,len-1)] = '/';
  ierr = PetscStrncpy(buf+PetscMin(dlen+1,len-1),base,PetscMax(len-dlen-1,0));CHKERRQ(ierr);
  buf[len-1] = 0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "CheckLen"
static PetscErrorCode CheckLen(const char *path,size_t len,size_t *used)
{
  char           *buf,cmd[4096],spath[PETSC_MAX_PATH_LEN],rpath[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  FILE           *fp;
  PetscBool      match;

  PetscFunctionBegin;
  /* dynamically allocate so valgrind and PETSc can check for overflow */
  ierr = PetscMalloc(len,&buf);CHKERRQ(ierr);
  ierr = PetscGetFullPath(path,buf,len);CHKERRQ(ierr);
  ierr = PetscSNPrintf(cmd,sizeof cmd,"printf %%s %s",path);CHKERRQ(ierr);
  ierr = PetscPOpen(PETSC_COMM_SELF,NULL,cmd,"r",&fp);CHKERRQ(ierr);
  if (!fgets(spath,sizeof spath,fp)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in shell command: %s",cmd);
  ierr = PetscPClose(PETSC_COMM_SELF,fp,NULL);CHKERRQ(ierr);
  ierr = RealpathPhonyFile(spath,rpath,len);CHKERRQ(ierr);
  ierr = PetscStrcmp(rpath,buf,&match);CHKERRQ(ierr);
  if (!match) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"$(realpath %s | head -c %d) %s != %s\n",path,(int)len-1,rpath,buf);CHKERRQ(ierr);
  }
  if (used) {ierr = PetscStrlen(buf,used);CHKERRQ(ierr);}
  ierr = PetscFree(buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Check"
static PetscErrorCode Check(const char *path)
{
  PetscErrorCode ierr;
  size_t         used;

  PetscFunctionBegin;
  ierr = CheckLen(path,PETSC_MAX_PATH_LEN,&used);CHKERRQ(ierr);
  ierr = CheckLen(path,used-1,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           user[256],buf[512];

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = Check("~/file-name");CHKERRQ(ierr);
  ierr = PetscGetUserName(user,256);CHKERRQ(ierr);
  ierr = PetscSNPrintf(buf,sizeof buf,"~%s/file-name",user);CHKERRQ(ierr);
  ierr = Check(buf);CHKERRQ(ierr);
  ierr = Check("/dev/null");CHKERRQ(ierr);
  ierr = Check("./this-dir");CHKERRQ(ierr);
  ierr = Check("also-this-dir");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

