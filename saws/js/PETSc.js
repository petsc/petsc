//this js file contains methods copied from matt's original SAWs.js but modified to fit our needs

var divSave;//ignore this for now. I'm trying to get rid of the 1000ms delay

var successFunc = function(data, textStatus, jqXHR)//ignore this for now. I'm trying to get rid of the 1000ms delay
{
    console.log(data);
    jQuery(divSave).html("");
    window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divSave);
}


PETSc = {};

var sawsInfo = [];//this variable is used to organize all the data from SAWs

var init = false;//record if initialized the page (added appropriate divs for the diagrams and such)

var removedText = false;//record if the text at the top was removed

//This Function is called once (document).ready. The javascript for this was written by the PETSc code into index.html
PETSc.getAndDisplayDirectory = function(names,divEntry){

    if(!init) {
        $("head").append('<script src="js/parsePrefix2.js"></script>');//reuse the code for parsing thru the prefix
        $("head").append('<script src="js/fetchSawsData.js"></script>');//reuse the code for organizing data into sawsInfo
        $("head").append('<script src="js/utils.js"></script>');//necessary for the two js files above
        $("body").append("<div id=\"multigridDiagram\" style=\"float:right;\"></div>");
        $("body").append("<div id=\"fieldsplitDiagram\"></div>");
        init = true;
    }

    jQuery(divEntry).html("");
    SAWs.getDirectory(names,PETSc.displayDirectory,divEntry);
}

PETSc.displayDirectory = function(sub,divEntry)
{
    globaldirectory[divEntry] = sub;
    recordSawsData(sub);//records data into sawsInfo[]

    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
        var SAWs_pcVal  = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0];
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["prefix"].data[0];

        if(SAWs_prefix == "(null)")
            SAWs_prefix = "";

        if(SAWs_prefix != "" && !removedText) {//remove the text at the top (the first 9 elements)
            for(var i=0; i<9; i++) {
                $("body").children().first().remove();
            }
            removedText=true;
        }
/*
        var generatedFieldsplit = parsePrefixForFieldsplit(SAWs_prefix).fieldsplit;

        if(generatedFieldsplit != "0") {
            $("fieldsplitDiagram").html("");

            var colors = ["green","blue"];
            var layer = 0;
            $("#fieldsplitDiagram").html("<svg id=\"svgFieldsplit\" width='400' height='400'> <polygon points=\"0,0 0,400 400,400 400,0\" style=\"fill:khaki;stroke:black;stroke-width:1\"> </svg>");
            layer = 1;

            drawFieldsplit("0",0,0,400);

            function drawFieldsplit(fieldsplit,x,y,size) {//input is the id of the fieldsplit. for example "0". (x,y) is the upper lefthand corner. size is the size of one side of the parent square (in pixels)
                //work = draw the children of the fieldsplit then call draw on each child
                var numChildren = getSawsNumChildren(fieldsplit);
                if(numChildren == 0)
                    return;
                var colorNum = fieldsplit.length - 1;

                for(var i=0; i<numChildren; i++) {
                    var side   = size/(numChildren+1);//leave one extra block of space
                    var curr_x = x + i*side;
                    var curr_y = y + i*side;

                    var string = "<polygon points=\""+curr_x+","+curr_y+" "+(curr_x+side)+","+curr_y+" "+(curr_x+side)+","+(curr_y+side)+" "+curr_x+","+(curr_y+side)+"\" style=\"fill:"+colors[colorNum]+";stroke:black;stroke-width:1\"> </svg>";

                    $("#svgFieldsplit").append(string);
                    var childID = fieldsplit + i;
                    drawFieldsplit(childID, curr_x, curr_y, size/numChildren);
                }
                var side = size/(numChildren+1);//side of the blank square
                var blank_x = x + numChildren*side;
                var blank_y = y + numChildren*side;

                var inc = side/4;//the increment
                for(var i=1; i<4; i++) {
                    var x_coord = blank_x + i*inc;
                    var y_coord = blank_y + i*inc;
                    $("#svgFieldsplit").append("<circle cx=\""+x_coord+"\" cy=\"" + y_coord + "\" r=\"1\" stroke=\"black\" stroke-width=\"2\" fill=\"black\">");
                }
            }
            $("body").html($("body").html());//hacky refresh svg
        }


        var generatedEndtag = parsePrefixForEndtag(SAWs_prefix,0);
        var index = getSawsIndex(generatedFieldsplit);

        if(sawsInfo[index].data[getSawsDataIndex(index,generatedEndtag.substring(0,generatedEndtag.length-1))].pc == "mg") { //if parent is mg

            var _index = generatedEndtag.length;
            var _level = generatedEndtag[_index-1];//get the last character

            $("#multigridDiagram").html("");//clear the diagram

            //generate a parallelogram for each layer
            for(var i=0; i<=_level; i++) { //i represents the multigrid level (i=0 would be coarse)
                var dim = 3+2*i;//dimxdim grid
                $("#multigridDiagram").append("<svg id=\"svg"+i+"\" width='465' height='142'> <polygon points=\"0,141 141,0 465,0 324,141\" style=\"fill:khaki;stroke:black;stroke-width:1\"> </svg>");//the sides of the parallogram follow the golden ratio so that the original figure was a golden rectangle. the diagram is slanted at a 45 degree angle.

                for(var j=1; j<dim; j++) {//draw 'vertical' lines
                    var inc = 324/dim;//parallogram is 324 wide and 200 on the slant side (1.6x)
                    var shift = j*inc;
                    var top_shift = shift + 141;
                    $("#svg"+i).append("<line x1=\""+shift+"\" y1='141' x2=\""+top_shift+"\" y2='0' style='stroke:black;stroke-width:1'></line>");
                }
                for(var j=1; j<dim; j++) {//draw horizonal lines
                    var inc = 141/dim;//parallelogram is 141 tall
                    var horiz_shift = (141/dim) * j;
                    var horiz_shift_end = horiz_shift + 324;

                    var shift = 141 - inc * j;
                    $("#svg"+i).append("<line x1=\""+horiz_shift+"\" y1=\""+shift +"\" x2=\""+horiz_shift_end+"\" y2=\""+shift+"\" style='stroke:black;stroke-width:1'></line>");
                }
                //put text here
                if(i!=0)
                    $("#multigridDiagram").append("<span>Level "+i+"</span><br>");
                else
                    $("#multigridDiagram").append("<span>Coarse Grid (Level 0)</span><br>");

                if(i != _level)//add transition arrows image if there are more grids left
                    $("#multigridDiagram").append("<img src='images/transition.bmp' alt='Error Loading Multigrid Transition Arrows'><br>");
            }

            $("body").html($("body").html());//refresh (hacky) after appending to svg
        }*/
    }

    PETSc.displayDirectoryRecursive(sub.directories,divEntry,0,"");//this method is recursive on itself and actually fills the div with text and dropdown lists

    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {
        console.log("data fetched:");
        console.log(sub);
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

            var descriptionSave = "";//saved description string because although the data is fetched: "description, -option, value" we wish to display it: "-option, value, description"
            var manualSave = ""; //saved manual text
            var mg_encountered = false;//record whether or not we have encountered pc=multigrid

            jQuery.each(sub[key].variables, function(vKey, vValue) {//for each variable...

                if (vKey.substring(0,2) == "__") // __Block variable
                    return;
                //SAWs.tab(fullkey,tab+1);
                if (vKey[0] != '_') {//this chunk of code adds the option name
                    if(vKey.indexOf("prefix") != -1 && sub[key].variables[vKey].data[0] == "(null)")
                        return;//do not display (null) prefix

                    if(vKey.indexOf("prefix") != -1) //prefix text
                        $("#"+fullkey).append(vKey + ":&nbsp;");
                    else if(vKey.indexOf("ChangedMethod") == -1) { //options text
                        //options text is a link to the appropriate manual page

                        var manualDirectory = "all"; //this directory does not exist yet so links will not work for now
                        $("#"+fullkey).append("<br><a href=\"http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/" +  manualDirectory + "/" + manualSave + ".html\" title=\"" + descriptionSave + "\" id=\"data"+fullkey+vKey+j+"\">"+vKey+"&nbsp</a>");
                    }
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

                    if(sub[key].variables[vKey].alternatives.length == 0) {//case where there are no alternatives
                        if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {
                            $("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");//make the boolean dropdown list.
                            $("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>");
                            if(vKey == "ChangedMethod") {//do not show changedmethod to user
                                $("#data"+fullkey+vKey+j).attr("hidden",true);
                            }

                        } else {
                            if(sub[key].variables[vKey].mtype != "SAWs_WRITE") {

                                descriptionSave = sub[key].variables[vKey].data[j];

                                if(vKey.indexOf("prefix") != -1) //data of prefix so dont do manual and use immediately
                                    $("#"+fullkey).append("<a style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</a><br>");

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

                        jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                            sub[key].variables[vKey].selected = 1;
                            $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                            var id = "data"+fullkey+vKey+j;
                            if(id.indexOf("type") != -1) {//if some type variable changed, then act as if continue button was clicked
                                $("#continue").trigger("click");
                            }
                        });
                    }
                }
            });

            if(typeof sub[key].directories != 'undefined'){
                PETSc.displayDirectoryRecursive(sub[key].directories,divEntry,tab+1,fullkey);
             }
        }
    });
}