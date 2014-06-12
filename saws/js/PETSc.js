//this js file contains methods copied from matt's original SAWs.js but modified to fit our needs

var divSave;//ignore this for now. I'm trying to get rid of the 1000ms delay

var successFunc = function(data, textStatus, jqXHR)//ignore this for now. I'm trying to get rid of the 1000ms delay
{
    console.log(data);
    jQuery(divSave).html("");
    window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divSave);
}


PETSc = {};

var encounteredMg = false; // record whether we encountered pc=mg
var mgLevel = -1;//record the current mg level: 0=coarse, >0 = smoothing

var init = false;//record if initialized the page (added appropriate divs for the diagrams and such)

//This Function is called once (document).ready. The javascript for this was written by the PETSc code into index.html
PETSc.getAndDisplayDirectory = function(names,divEntry){

    if(!init) {
        $("body").append("<div id=\"multigridDiagram\"></div>");
        $("body").append("<div id=\"fieldsplitDiagram\"></div>");
        init = true;
    }

    jQuery(divEntry).html("");
    PETSc.getDirectory(names,PETSc.displayDirectory,divEntry);
}

PETSc.getDirectory = function(names,callback,callbackdata) {

  /*If names is null, get all*/
  if(names == null){
    /*jQuery.getJSON('/SAWs/*',function(data){
                               if(typeof(callback) == typeof(Function)) callback(data,callbackdata)
                             })*/
      jQuery.ajax({type: 'GET',dataType: 'json',url: '/SAWs/*', success:
                   function(data){
                       console.log("data fetched from server");
                       console.log(data);
                       callback(data,callbackdata);
                   }
                  }
                 );
  } else {alert("should not be here");
    jQuery.getJSON('/SAWs/' + names,function(data){
        if(typeof(callback) == typeof(Function))
            callback(data,callbackdata);
    });
  }
};

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
        jQuery(divEntry).append("<input type=\"button\" value=\"Continue\" id=\"continue\">");
        jQuery('#continue').on('click', function(){
            SAWs.updateDirectoryFromDisplay(divEntry);
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];

            divSave = divEntry;
            //PETSc.postDirectory(sub, successFunc);//ignore this for now. I'm trying to get rid of 1000ms delay
            SAWs.postDirectory(sub);
            window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divEntry);
        });
    } else console.log("no block property or block property is false");

}

PETSc.postDirectory = function(directory, callback)//ignore this for now. I'm trying to get rid of the 1000ms delay
{
    var stringJSON = JSON.stringify(directory);
    jQuery.ajax({type: 'POST',dataType: 'json',url: '/SAWs/*',data: {input: stringJSON}, success: callback});
}

PETSc.displayDirectoryRecursive = function(sub,divEntry,tab,fullkey)
{
    jQuery.each(sub,function(key,value){
        fullkey = fullkey+key;//key contains things such as "PETSc" or "Options"
        if(jQuery("#"+fullkey).length == 0){
            jQuery(divEntry).append("<div id =\""+fullkey+"\"></div>")
            if (key != "SAWs_ROOT_DIRECTORY") {
                //SAWs.tab(fullkey,tab);
	        //jQuery("#"+fullkey).append("<b>"+ key +"<b><br>");//do not display "PETSc" nor "Options"
            }

            var save = "";//saved html element containing the description because although the data is fetched: "description, -option, value" we wish to display it: "-option, value, description"
            var manualSave = ""; //saved manual text
            var prefix = "";//save what the prefix is so that we can prepend it to all the options
            var mg_encountered = false;//record whether or not we have encountered pc=multigrid

            jQuery.each(sub[key].variables, function(vKey, vValue) {//for each variable...

                if (vKey[0] == '_' && vKey[1] == '_' )
                    return;
                //SAWs.tab(fullkey,tab+1);
                if (vKey[0] != '_') {//this chunk of code adds the option name
                    if(vKey.indexOf("prefix") != -1 && sub[key].variables[vKey].data[0] == "(null)")
                        return;//do not display (null) prefix

                    if(vKey.indexOf("prefix") == -1 && vKey.indexOf("ChangedMethod") == -1)//not the prefix text so edit to have the prefix prepended to the option text
                        $("#"+fullkey).append("-" + prefix + vKey.substring(1,vKey.length) + ":&nbsp;");
                    else if(vKey.indexOf("ChangedMethod") == -1) //if reaches here, it is the prefix text
                        $("#"+fullkey).append(vKey + ":&nbsp;");
                }

                for(j=0;j<sub[key].variables[vKey].data.length;j++){//vKey tells us a lot of information on what the data is. data.length is 1 most of the time. when it is more than 1, that results in 2 input boxes right next to each other

                    if(vKey.indexOf("man") != -1) {//do not display manual, but record the text
                        manualSave = sub[key].variables[vKey].data[j];
                        continue;
                    }

                    if(vKey.indexOf("title") != -1) {//display title in center
                        $("#"+fullkey).append("<center>"+"<span style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</span>"+"</center>");
                        continue;
                    }

                    if(vKey.indexOf("prefix") != -1) {//record prefix
                        var saws_prefix = sub[key].variables[vKey].data[j];
                        if(saws_prefix == "(null)")
                            prefix = "";
                        else
                            prefix = saws_prefix;
                    }

                    if(sub[key].variables[vKey].alternatives.length == 0) {//case where there are no alternatives
                        if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {
                            $("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");//make the boolean dropdown list.
                            $("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>");
                            if(vKey == "ChangedMethod") {//do not show changedmethod to user
                                $("#data"+fullkey+vKey+j).attr("hidden",true);
                            }
                            else {//do not add the extra line if it was ChangedMethod
                                $("#"+fullkey).append(save+"<br>");
                                save = "";
                            }

                        } else {
                            if(sub[key].variables[vKey].mtype != "SAWs_WRITE") {
                                if(save != "")
                                    $("#"+fullkey).append(save+"<br>");

                                var manualDirectory = "all"; //this directory does not exist yet so links will not work for now

                                save = "<a style=\"font-family: Courier\" href=\"http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/" +  manualDirectory + "/" + manualSave + ".html\" title=\"" + manualSave + "\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</a>";//can't be changed

                                if(vKey.indexOf("prefix") != -1) {//data of prefix so dont do manual and use immediately
                                    save = "<a style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</a>";//can't be changed
                                    $("#"+fullkey).append(save+"<br>");
                                    save = "";
                                }
                            }
                            else {//can be changed (append dropdown list)
                                $("#"+fullkey).append("<input type=\"text\" style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\" name=\"data\" \\>");
                            }
                            jQuery("#data"+fullkey+vKey+j).keyup(function(obj) {
                                console.log( "Key up called "+key+vKey );
                                sub[key].variables[vKey].selected = 1;
                                $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                            });
                        }
                        jQuery("#data"+fullkey+vKey+j).val(sub[key].variables[vKey].data[j]);//set val from server
                        if(vKey != "ChangedMethod") {
                            jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                                console.log( "Change called"+key+vKey );
                                sub[key].variables[vKey].selected = 1;
                                $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                            });
                        }
                    } else {//case where there are alternatives
                        jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");
                        jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].data[j]+"\">"+sub[key].variables[vKey].data[j]+"</option>");
                        for(var l=0;l<sub[key].variables[vKey].alternatives.length;l++) {
                            jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].alternatives[l]+"\">"+sub[key].variables[vKey].alternatives[l]+"</option>");
                        }
                        jQuery("#"+fullkey).append("</select>");
                        $("#"+fullkey).append(save+"<br>");
                        save = "";

                        jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                            console.log( "Change called"+key+vKey );
                            sub[key].variables[vKey].selected = 1;
                            $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                        });
                    }
                }
            });

            if(save != "") {//to avoid losing a description at the end of the page
                $("#"+fullkey).append(save+"<br>");
                save = "";
            }

            if(typeof sub[key].directories != 'undefined'){
                PETSc.displayDirectoryRecursive(sub[key].directories,divEntry,tab+1,fullkey);
             }
        }
    });
}