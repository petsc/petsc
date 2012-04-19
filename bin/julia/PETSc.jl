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
  args = ["julia",args]; 
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
    is = new(is[1])
    finalizer(is,PetscISDestroy) 
    # does not seem to be called immediately when is is no longer visible, is it called later during garbage collection?
    return is
  end
end

  function PetscISDestroy(is::PetscIS)
    err = ccall(dlsym(libpetsc, :ISDestroy),Int32,(Ptr{Int64},), &is.pobj);    
  end

  function PetscIS(indices::Array{Int64})
    is = PetscIS()
    err = ccall(dlsym(libpetsc, :ISSetType),Int32,(Int64,Ptr{Uint8}), is.pobj,cstring("general"));
    COPY_VALUES = 0
    err = ccall(dlsym(libpetsc, :ISGeneralSetIndices),Int32,(Int64,Int32,Ptr{Int32},Int32),is.pobj,length(indices),convert(Array{Int32},indices),COPY_VALUES)
    return is
  end

  function PetscView(obj::PetscIS,viewer)
   err = ccall(dlsym(libpetsc, :ISView),Int32,(Int64,Int64),obj.pobj,0);
  end

  function PetscISGetSize(obj::PetscIS)
    n = Array(Int32,1)
    err = ccall(dlsym(libpetsc, :ISGetSize),Int32,(Int64,Ptr{Int32}), obj.pobj,n);
    return n[1]
  end

  function PetscISGetIndices(obj::PetscIS)
    len = PetscISGetSize(obj)
    indices = Array(Int32,len);
    err = ccall(dlsym(libpetsc,:ISGetIndicesMatlab),Int32,(Int64,Ptr{Int32}),obj.pobj,indices);
    indices = indices + 1
    return indices  
  end

# -------------------------------------
#
type PetscVec
  pobj::Int64
  function PetscVec()
    comm = PETSC_COMM_SELF();
    vec = Array(Int64,1)
    err = ccall(dlsym(libpetsc, :VecCreate),Int32,(Int64,Ptr{Int64}),comm,vec);
    vec = new(vec[1])
    finalizer(vec,PetscVecDestroy)  
    # does not seem to be called immediately when vec is no longer visible, is it called later during garbage collection?
    return vec
  end
end

  function PetscVecDestroy(vec::PetscVec)
    err = ccall(dlsym(libpetsc, :VecDestroy),Int32,(Ptr{Int64},), &vec.pobj);    
  end

  function PetscVec(array::Array{Float64})
    vec = PetscVec()
    err = ccall(dlsym(libpetsc, :VecSetType),Int32,(Int64,Ptr{Uint8}), vec.pobj,cstring("seq"));
    err = ccall(dlsym(libpetsc, :VecSetSizes),Int32,(Int64,Int32,Int32), vec.pobj,length(array),length(array));
    # want a 32 bit int array so build it ourselves
    idx = Array(Int32,length(array)); 
    println(idx); 
    for i=1:length(array); println(i); idx[i] = i-1;  end
    println(idx)
    INSERT_VALUES = 1
    err = ccall(dlsym(libpetsc, :VecSetValues), Int32,(Int64,Int32,Ptr{Int32},Ptr{Float64},Int32), vec.pobj,length(idx),idx,array,INSERT_VALUES);
    err = ccall(dlsym(libpetsc, :VecAssemblyBegin),Int32,(Int64,), vec.pobj);
    err = ccall(dlsym(libpetsc, :VecAssemblyEnd),Int32,(Int64,), vec.pobj);
    return vec
  end

  function PetscView(obj::PetscVec,viewer)
   err = ccall(dlsym(libpetsc, :VecView),Int32,(Int64,Int64),obj.pobj,0);
  end

  function PetscVecGetSize(obj::PetscVec)
    n = Array(Int32,1)
    err = ccall(dlsym(libpetsc, :VecGetSize),Int32,(Int64,Ptr{Int32}), obj.pobj,n);
    return n[1]
  end


