PETSC_DIR = ../../..

ALL: lib 
CFLAGS	  =  $(PETSC_INCLUDE) $(CONF) $(PCONF) -D__DIR__='"sys/src/"'
SOURCEC	  = adebug.c err.c signal.c fp.c mtr.c arch.c file.c nreg.c \
	    mpiu.c sorti.c destroy.c plog.c options.c mal.c \
	    sysio.c str.c memc.c random.c sortd.c sortip.c
SOURCEF	  =
SOURCEH	  =
OBJSC	  = adebug.o err.o signal.o fp.o mtr.o arch.o file.o nreg.o \
	    mpiu.o sorti.o destroy.o	plog.o options.o mal.o \
	    sysio.o str.o memc.o random.o sortip.o sortd.o
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/src/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
