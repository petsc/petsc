#PETSC_DIR = ../../..

ALL: lib 
CFLAGS	  = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC	  = adebug.c err.c signal.c fp.c mtr.c arch.c nreg.c \
	    mpiu.c sorti.c destroy.c options.c mal.c \
	    sysio.c str.c memc.c random.c sortd.c sortip.c \
            fdate.c fpath.c rpath.c fuser.c fhost.c ftest.c \
            ffpath.c fwd.c  grpath.c ghome.c pcomm.c \
            mpiuopen.c pname.c pgname.c \
            gcomm.c inherit.c gtype.c prefix.c psleep.c gcookie.c \
            cookie.c mpitr.c nt_time.c mem.c cputime.c
SOURCEF	  =
SOURCEH	  = files.h
OBJSC	  = adebug.o err.o signal.o fp.o mtr.o arch.o nreg.o \
	    mpiu.o sorti.o destroy.o options.o mal.o \
	    sysio.o str.o memc.o random.o sortip.o sortd.o \
            fdate.o fpath.o rpath.o fuser.o fhost.o ftest.o \
            ffpath.o fwd.o grpath.o ghome.o pcomm.o \
            mpiuopen.o pname.o pgname.o \
            gcomm.o inherit.o gtype.o prefix.o psleep.o gcookie.o \
            cookie.o mpitr.o nt_time.o mem.o cputime.o
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/src/

rs6000_time: rs6000_time.s
	-@if [ "$(PARCH)" = "rs6000" ]; then \
	$(CC) -c rs6000_time.s ; \
	$(AR) cr $(LIBNAME) rs6000_time.o ; \
	$(RM) -f rs6000_time.o; \
	fi

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

