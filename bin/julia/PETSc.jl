#
#   Allows the PETSc dynamic library to be used from Julia. (http::/julialang.org)
#     PETSc must be configured with --with-shared-libraries
#     You can build with or without MPI, but cannot run on more than one process
#
#  Load the PETSc dynamic library
#
PETSC_DIR = getenv("PETSC_DIR");
PETSC_ARCH = getenv("PETSC_ARCH");
if (length(PETSC_DIR) == 0) 
  disp("Must have environmental variable PETSC_DIR set")
end
if (length(PETSC_ARCH) == 0) 
  disp("Must have environmental variable PETSC_ARCH set")
end
libpetsclocation = strcat(PETSC_DIR, "/", PETSC_ARCH, "/lib/", "libpetsc")
libpetsc = dlopen(libpetsclocation)

# -------------------------------------
#    These are the Julia interface methods for all the visible PETSc functions. Julia datatypes for PETSc objects simply contain the C pointer to the 
#    underlying PETSc object.
#   
# -------------------------------------
function PetscInitialize()
  PetscInitialize([])
end

function PetscInitialize(args)
  PetscInitialize(args,"","")
end

function PetscInitialize(args,filename,help)
  # argument list starts with program name
  arg = ["julia",args]; 
  #
  #   If the user forgot to PetscFinalize() we do it for them, before restarting PETSc
  #
  init = ccall(dlsym(libpetsc,:PetscInitializedMatlab),Int32,());
  if (init != 0) 
    err = ccall(dlsym(libpetsc,:PetscFinalize),Int32,());if (err != 0) return err; end
  end
  arr = Array(ByteString, length(args))
  for i = 1:length(args)
    arr[i] = cstring(args[i])
  end
  ptrs = _jl_pre_exec(arr)
  err = ccall(dlsym(libpetsc, :PetscInitializeMatlab),Int32,(Int32,Ptr{Ptr{Uint8}},Ptr{Uint8},Ptr{Uint8}), length(ptrs), ptrs,cstring(filename),cstring(help));
  return err
end

function PetscFinalize()
  err = ccall(dlsym(libpetsc,:PetscFinalize),Int32,());
  return err
end

function PETSC_COMM_SELF()
  comm = Array(Int64, 1)
  err = ccall(dlsym(libpetsc, :PetscGetPETSC_COMM_SELFMatlab),Int32,(Ptr{Int64},),comm);
  return comm[1]
end

# -------------------------------------
#
type PetscIS
  pobj::Int64
  function PetscIS()
    comm = PETSC_COMM_SELF();
    is = Array(Int64,1)
    err = ccall(dlsym(libpetsc, :ISCreate),Int32,(Int64,Ptr{Int64}),comm,is);if (err != 0) return err;end
    new(is[1])
  end
end

  function PetscIS(indices)
    is = PetscIS()
    err = ccall(dlsym(libpetsc, :ISSetType),Int32,(Int64,Ptr{Uint8}), is.pobj,cstring("general"));
    COPY_VALUES = 0
    err = ccall(dlsym(libpetsc, :ISGeneralSetIndices),Int32,(Int64,Int32,Ptr{Int32},Int32),is.pobj,length(indices),convert(Array{Int32},indices),COPY_VALUES)
    return is
  end

  function PetscView(obj,viewer)
   err = ccall(dlsym(libpetsc, :ISView),Int32,(Int64,Int64),obj.pobj,0);
  end

  function PetscISGetSize(obj)
    n = Array(Int32,1)
    err = ccall(dlsym(libpetsc, :ISGetSize),Int32,(Int64,Ptr{Int32}), obj.pobj,n);
    return n[1]
  end

  function PetscISGetIndices(obj)
    len = PetscISGetSize(obj)
    indices = Array(Int32,len);
    err = ccall(dlsym(libpetsc,:ISGetIndicesMatlab),Int32,(Int64,Ptr{Int32}),obj.pobj,indices);
    indices = indices + 1
    return indices  
  end



