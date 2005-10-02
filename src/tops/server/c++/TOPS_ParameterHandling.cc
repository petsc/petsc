// This file is not generated from SIDL, it's for internal impl use only

#include <iostream>
#include "TOPS_ParameterHandling.hh"

void processTOPSOptions(std::string options) {
#undef __FUNCT__
#define __FUNCT__ "Ex1::System_impl::processTOPSOptions"
  std::string key = "", val = ""; 
  bool inKey = true, inVal = false, newOption = true;
  int len = options.length();
  for (int i = 1; i < len; ++i) {
    if (options[i] == ' ') { 
      newOption = true; inKey = false; inVal = false; continue;
    }
    if (newOption && (options[i] == '-') || i == (len-1)) { 
      //std::cout << "Setting petsc option: " << key << " " << val << std::endl;
      if (val != "") PetscOptionsSetValue(key.c_str(), val.c_str());
      else if (key != "") PetscOptionsSetValue(key.c_str(), 0);
      inKey = true; inVal = false; 
      key = "-"; val = ""; continue; 
    } else {
      newOption = false;
      if (inKey) key += options[i];   
      else { inVal = true; val += options[i]; }
    }
  }
}
