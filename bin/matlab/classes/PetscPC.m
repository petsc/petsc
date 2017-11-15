classdef PetscPC < PetscObject
  methods
    function obj = PetscPC(pid,flg)
      if (nargin > 1) 
        %  PetscPC(pid,'pobj') uses an already existing PETSc PC object
        obj.pobj = pid;
        return
      end
      comm = PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libpetsc', 'PCCreate',comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'PCSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetDM(obj,da)
      err = calllib('libpetsc', 'PCSetDM', obj.pobj,da.pobj);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'PCSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'PCSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperators(obj,A,B,pattern)
      err = calllib('libpetsc', 'PCSetOperators', obj.pobj,A.pobj,B.pobj,pattern);PetscCHKERRQ(err);
    end
    function err = FieldSplitSetIS(obj,name,is)
      err = calllib('libpetsc','PCFieldSplitSetIS',obj.pobj,name,is.pobj)
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libpetsc', 'PCView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libpetsc', 'PCView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'PCDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
