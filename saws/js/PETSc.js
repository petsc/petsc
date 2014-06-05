//this js file contains methods copied from matt's original SAWs.js but modified to fit our needs

PETSc = {}

PETSc.getAndDisplayDirectory = function(names,divEntry){
//    window.location = 'pcoptions.html'

    //below are skipped now ...
    jQuery(divEntry).html("")
    SAWs.getDirectory(names,PETSc.displayDirectory,divEntry)
}

PETSc.displayDirectory = function(sub,divEntry)
{
    globaldirectory[divEntry] = sub;
    //var SAWs_pcVal = JSON.stringify(sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0]);
    //alert("SAWs_pcVal="+SAWs_pcVal);
    //alert(JSON.stringify(sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].alternatives)) //pcList

    //alert(JSON.stringify(sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables))

//    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
  //      window.location = 'pcoptions.html'
//    } else {
      //  }
    PETSc.displayDirectoryRecursive(sub.directories,divEntry,0,"");//this method is recursive on itself and actually fills the div with text and dropdown lists

    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {
        jQuery(divEntry).append("<center><input type=\"button\" value=\"Continue\" id=\"continue\"></center>")
        jQuery('#continue').on('click', function(){
            SAWs.updateDirectoryFromDisplay(divEntry);
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
            SAWs.postDirectory(sub);
            jQuery(divEntry).html("");
            window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divEntry);
        })
    }

}

PETSc.displayDirectoryRecursive = function(sub,divEntry,tab,fullkey)
{
  jQuery.each(sub,function(key,value){
      fullkey = fullkey+key;
      if(jQuery("#"+fullkey).length == 0){
          jQuery(divEntry).append("<div id =\""+fullkey+"\"></div>")
          if (key != "SAWs_ROOT_DIRECTORY") {
	      SAWs.tab(fullkey,tab);
	      jQuery("#"+fullkey).append("<b>"+ key +"<b><br>");
          }
          jQuery.each(sub[key].variables, function(vKey, vValue) {//for each variable...
              if (vKey[0] != '_' || vKey[1] != '_' ) {
                  SAWs.tab(fullkey,tab+1);
                  if (vKey[0] != '_') {
                      jQuery("#"+fullkey).append(vKey+":&nbsp;");
                  }
                  for(j=0;j<sub[key].variables[vKey].data.length;j++){//vKey tells us a lot of information on what the data is
                      if(sub[key].variables[vKey].alternatives.length == 0){
                          if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {
                              jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");
                              jQuery("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>");
                          } else {
                              jQuery("#"+fullkey).append("<input type=\"text\" style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\" name=\"data\" \\>");
                              jQuery("#data"+fullkey+vKey+j).keyup(function(obj) {
                                  console.log( "Key up called "+key+vKey );
                                  sub[key].variables[vKey].selected = 1;
                              });
                          }
                          jQuery("#data"+fullkey+vKey+j).val(sub[key].variables[vKey].data[j]);
                          jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                              console.log( "Change called"+key+vKey );
                              sub[key].variables[vKey].selected = 1;
                          });
                      } else {
                          jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");
                          jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].data[j]+"\">"+sub[key].variables[vKey].data[j]+"</option>");
                          for(var l=0;l<sub[key].variables[vKey].alternatives.length;l++){
                              jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].alternatives[l]+"\">"+sub[key].variables[vKey].alternatives[l]+"</option>");
                          }
                          jQuery("#"+fullkey).append("</select>");
                          jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                              console.log( "Change called"+key+vKey );
                              sub[key].variables[vKey].selected = 1;
                          });
                      }
                      if(sub[key].variables[vKey].mtype != "SAWs_WRITE") {
                          jQuery("#data"+ fullkey+vKey+j).attr('readonly',true);
                      } else {
                          jQuery("#data"+ fullkey+vKey+j).attr('style',"color: #FF0000");
                      }
                  }
                  if(vKey.indexOf("text") == -1)//do NOT start a new line if the text was simply a description
                     jQuery("#"+fullkey).append("<br>");
              }
          });
          if(typeof sub[key].directories != 'undefined'){
              PETSc.displayDirectoryRecursive(sub[key].directories,divEntry,tab+1,fullkey);
          }
      }
  });
}