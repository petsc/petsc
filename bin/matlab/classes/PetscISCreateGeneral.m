function [is,err] = PetscISGeneralCreate(indices)
  is  = PetscIS();
  err = is.SetType('general');
  err = is.GeneralSetIndices(indices);
