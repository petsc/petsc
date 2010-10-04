function [is,err] = ISGeneralCreate(indices)
  is  = IS();
  err = is.SetType('general');
  err = is.GeneralSetIndices(indices);
