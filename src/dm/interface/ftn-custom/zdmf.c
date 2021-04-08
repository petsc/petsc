#include <petsc/private/fortranimpl.h>
#include <petscdm.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmcreateinterpolation_       DMCREATEINTERPOLATION
#define dmview_                      DMVIEW
#define dmsetoptionsprefix_          DMSETOPTIONSPREFIX
#define dmsettype_                   DMSETTYPE
#define dmgettype_                   DMGETTYPE
#define dmsetmattype_                DMSETMATTYPE
#define dmsetvectype_                DMSETVECTYPE
#define dmgetmattype_                DMGETMATTYPE
#define dmgetvectype_                DMGETVECTYPE
#define dmlabelview_                 DMLABELVIEW
#define dmcreatelabel_               DMCREATELABEL
#define dmhaslabel_                  DMHASLABEL
#define dmgetlabelvalue_             DMGETLABELVALUE
#define dmsetlabelvalue_             DMSETLABELVALUE
#define dmgetlabelsize_              DMGETLABELSIZE
#define dmgetlabelidis_              DMGETLABELIDIS
#define dmgetlabelname_              DMGETLABELNAME
#define dmgetlabel_                  DMGETLABEL
#define dmgetstratumsize_            DMGETSTRATUMSIZE
#define dmgetstratumis_              DMGETSTRATUMIS
#define dmsetstratumis_              DMSETSTRATUMIS
#define dmremovelabel_               DMREMOVELABEL
#define dmviewfromoptions_           DMVIEWFROMOPTIONS
#define dmcreatesuperdm_             DMCREATESUPERDM
#define dmdestroy_                   DMDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmcreateinterpolation_       dmcreateinterpolation
#define dmview_                      dmview
#define dmsetoptionsprefix_          dmsetoptionsprefix
#define dmsettype_                   dmsettype
#define dmgettype_                   dmgettype
#define dmsetmattype_                dmsetmattype
#define dmsetvectype_                dmsetvectype
#define dmgetmattype_                dmgetmattype
#define dmgetvectype_                dmgetvectype
#define dmlabelview_                 dmlabelview
#define dmcreatelabel_               dmcreatelabel
#define dmhaslabel_                  dmhaslabel
#define dmgetlabelvalue_             dmgetlabelvalue
#define dmsetlabelvalue_             dmsetlabelvalue
#define dmgetlabelsize_              dmlabelsize
#define dmgetlabelidis_              dmlabelidis
#define dmgetlabelname_              dmgetlabelname
#define dmgetlabel_                  dmgetlabel
#define dmgetstratumsize_            dmgetstratumsize
#define dmgetstratumis_              dmgetstratumis
#define dmsetstratumis_              dmsetstratumis
#define dmremovelabel_               dmremovelabel
#define dmviewfromoptions_           dmviewfromoptions
#define dmcreatesuperdm_             dmreatesuperdm
#define dmdestroy_                   dmdestroy
#endif

PETSC_EXTERN void dmgetmattype_(DM *mm,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = DMGetMatType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void dmgetvectype_(DM *mm,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = DMGetVecType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void dmview_(DM *da,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DMView(*da,v);
}

PETSC_EXTERN void dmsetoptionsprefix_(DM *dm,char* prefix, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetOptionsPrefix(*dm,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void dmsettype_(DM *x,char* type_name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = DMSetType(*x,t);if (*ierr) return;
  FREECHAR(type_name,t);
}

PETSC_EXTERN void dmgettype_(DM *mm,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = DMGetType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

PETSC_EXTERN void dmsetmattype_(DM *dm,char* prefix, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetMatType(*dm,t);if (*ierr) return;
  FREECHAR(prefix,t);
}


PETSC_EXTERN void dmsetvectype_(DM *dm,char* prefix, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetVecType(*dm,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void dmcreatelabel_(DM *dm, char* name, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMCreateLabel(*dm, lname);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmhaslabel_(DM *dm, char* name, PetscBool *hasLabel, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMHasLabel(*dm, lname, hasLabel);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmgetlabelvalue_(DM *dm, char* name, PetscInt *point, PetscInt *value, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetLabelValue(*dm, lname, *point, value);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmsetlabelvalue_(DM *dm, char* name, PetscInt *point, PetscInt *value, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMSetLabelValue(*dm, lname, *point, *value);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmgetlabelsize_(DM *dm, char* name, PetscInt *size, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetLabelSize(*dm, lname, size);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmgetlabelidis_(DM *dm, char* name, IS *ids, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetLabelIdIS(*dm, lname, ids);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmgetlabelname_(DM *dm,PetscInt *n,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tmp;
  *ierr = DMGetLabelName(*dm,*n,&tmp);
  *ierr = PetscStrncpy(name,tmp,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void dmgetlabel_(DM *dm, char* name, DMLabel *label, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetLabel(*dm, lname, label);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmgetstratumsize_(DM *dm, char* name, PetscInt *value, PetscInt *size, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetStratumSize(*dm, lname, *value, size);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmgetstratumis_(DM *dm, char* name, PetscInt *value, IS *is, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMGetStratumIS(*dm, lname, *value, is);if (*ierr) return;
  if (is && !*is) *is = (IS)0;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmsetstratumis_(DM *dm, char* name, PetscInt *value, IS *is, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMSetStratumIS(*dm, lname, *value, *is);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmremovelabel_(DM *dm, char* name, DMLabel *label, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMRemoveLabel(*dm, lname, label);if (*ierr) return;
  FREECHAR(name, lname);
}

PETSC_EXTERN void dmviewfromoptions_(DM *dm,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = DMViewFromOptions(*dm,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void dmcreateinterpolation_(DM *dmc,DM *dmf,Mat *mat,Vec *vec, int *ierr)
{
  CHKFORTRANNULLOBJECT(vec);
  *ierr = DMCreateInterpolation(*dmc,*dmf,mat,vec);
}

PETSC_EXTERN void dmcreatesuperdm_(DM dms[], PetscInt *len, IS ***is, DM *superdm, int *ierr)
{
  *ierr = DMCreateSuperDM(dms, *len, *is, superdm);
}

PETSC_EXTERN void dmdestroy_(DM *x,int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = DMDestroy(x); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}
