/*$Id: mstart.c,v 1.1 1999/11/24 22:57:52 bsmith Exp bsmith $*/

#include "sys.h"
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PETSC_HAVE_IO_H)
#include <io.h>
#endif


#undef __FUNC__  
#define __FUNC__ "PetscStartProgram"
int PetscPOpen(MPI_Comm comm,char *program,
{
  Viewer_Binary *vbinary = (Viewer_Binary *) v->data;
  int           ierr,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(v->comm,&rank);CHKERRQ(ierr);
  if (!rank && vbinary->fdes) {
    close(vbinary->fdes);
    if (vbinary->storecompressed) {
#if defined(PETSC_HAVE_GZIP) && defined(PETSC_HAVE_POPEN)
      char par[1024],buf[1024];
      FILE *fp;
      /* compress the file */
      ierr = PetscStrcpy(par,"gzip ");CHKERRQ(ierr);
      ierr = PetscStrcat(par,vbinary->filename);CHKERRQ(ierr);
      if (!(fp = popen(par,"r"))) {
        SETERRQ1(1,1,"Cannot run command %s",par);
      }
      if (fgets(buf,1024,fp)) {
        SETERRQ2(1,1,"Error from command %s\n%s",par,buf);
      }
#else 
      SETERRQ(PETSC_ERR_SUP,"Compressed files are unsupported on this platform.");
#endif
    }
  }
  if (vbinary->fdes_info) fclose(vbinary->fdes_info);
  ierr = PetscStrfree(vbinary->filename);CHKERRQ(ierr);
  ierr = PetscFree(vbinary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

