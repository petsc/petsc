//This function builds the tree using d3 (http://d3js.org/)
//A good video explaining the tree:
//http://www.youtube.com/watch?v=x8dwXoWODZ4 (part 1)
//http://www.youtube.com/watch?v=iZ6MSHA4FMU (part 2)
function buildTree(matInfo, numberOfLevels, detailed)
{
    //initialize tree
    var treeData = {};

    //clean up matInfo
    cleanMatInfo();
    sortMatInfo();

    for(var i=0; i<matInfoWriteCounter; i++) {//first zero doesn't matter
        var string=(detailed) ? matInfo[i].string : matInfo[i].stringshort;
        var obj={name: string};//the new object to be placed
        var id=matInfo[i].id;
        var needsAllocation=false;

        if(matInfo[i].id == "0")
            treeData=obj;

        if(id.charAt(id.length-1)=="0") {//get the last char
            //need to allocate
            needsAllocation=true;
        }

        if(id.length == 2) {
            if(needsAllocation) {
                treeData.contents=[];
            }
            treeData.contents[parseInt(id.charAt(1))]=obj;
        }
        else if(id.length == 3) {
            if(needsAllocation) {
                treeData.contents[parseInt(id.charAt(1))].contents=[];
            }
            treeData.contents[parseInt(id.charAt(1))].contents[parseInt(id.charAt(2))]=obj;
        }
        else if(id.length == 4) {
            if(needsAllocation) {
                treeData.contents[parseInt(id.charAt(1))].contents[parseInt(id.charAt(2))]=[];
            }
            treeData.contents[parseInt(id.charAt(1))].contents[parseInt(id.charAt(2))].contents[parseInt(id.charAt(3))]=obj;
        }

    }

    function cleanMatInfo() {//remove all the -1 elements (these get generated when something is deleted)
        for(var i=0; i<matInfoWriteCounter; i++) {

            if(matInfo[i].id=="-1") {

                //shift everything down
                for(var j=i; j<matInfoWriteCounter-1; j++)
                    matInfo[j]=matInfo[j+1];

                i--;//there might be two in a row. we wouldnt want to skip over any

                matInfoWriteCounter--;//decrement write counter
                delete matInfo[matInfoWriteCounter];//garbage value doesn't have to be removed but it's easy so we'll remove it anyways
            }
        }

    }

    //selection sort. NOTE: THIS FUNCTION ASSUMES ALL GARBAGE VALUES (ID="-1") HAVE ALREADY BEEN REMOVED USING cleanMatInfo()
    function sortMatInfo() {
        for(var i=0; i<matInfoWriteCounter-1; i++) {//only need to go to second to last element
            var indexOfCurrentSmallest=i;
            for(var j=i; j<matInfoWriteCounter; j++) {
                if(matInfo[j].id.localeCompare(matInfo[i].id) == -1)
                    indexOfCurrentSmallest=j;
            }
            //swap i and indexOfCurrentSmallest
            var temp=matInfo[i];
            matInfo[i]=matInfo[indexOfCurrentSmallest];
            matInfo[indexOfCurrentSmallest]=temp;
        }

    }

    //---------------------------------------------------
    //Create a container for the tree - a 'canvas'
    //[n*310, n* 310] for horizontal
    //[n*580, n* 310] for vertical
    var canvas = d3.select("#tree").append("svg")
	.attr("width", numberOfLevels * 400)
	.attr("height", numberOfLevels * 320)
	.append("g")
	.attr("transform", "translate(50,50)");
    
    //Call the d3 tree layout
    //[n * 200, n * 200] for horizontal
    //[n* 550, n* 200] for vertical
    var tree = d3.layout.tree()
	.size([numberOfLevels * 200, numberOfLevels * 200])
   	.children(function(d) //find who has children from the data structure
    		  {
        	      return (!d.contents || d.contents.length === 0) ? null : d.contents;
    		  });

    //initialize the nodes and links (which are used by d3 to create the paths
    var nodes = tree.nodes(treeData);
    var links = tree.links(nodes);

    //create an actual node group on the canvas (where a dot and text will be placed)
    var node = canvas.selectAll(".node")
	.data(nodes)
	.enter()
	.append("g")
	.attr("class", "node")
	.attr("transform", function (d){return "translate(" + d.y + "," + d.x + ")";}) //root: left
        //.attr("transform", function (d){return "translate(" + d.x + "," + d.y + ")";}) //root: top

    //add to that node a circle
    //change the circle properties here
    node.append("circle")
	.attr("r", 5)
	.attr("fill", "steelblue")

    //Add text to the node (names)
    //"foreignObject" is used to allow for rich formatting through mathJax and HTML (such as wrapping)
    //adjust the size of the box to create the desired aesthetic
    //Move it with x and y
    node.append("foreignObject")
	.attr('width',400)
  	.attr('height',400)//perhaps just remove these attributes altogether so it can resize itself as needed?
    //.attr('x', -23)
  	.attr('requiredFeatures','http://www.w3.org/TR/SVG11/feature#Extensibility')
  	.append('xhtml')
	.html(function (d) { return d.name; }) //this is where the data is actually placed in the tree


    //diagonal is the d3 method that draws the lines
    var diagonal = d3.svg.diagonal()
	.projection(function (d) { return [d.y, d.x]})	//root: left
        //.projection(function (d) { return [d.x, d.y]}) //root: top

    //Writes everything to screen
    canvas.selectAll(".link")
	.data(links)
	.enter()
	.append("path")
	.attr("class", "link")
	.attr("fill", "none")
	.attr("stroke", "#ADADAD")
	.attr("d", diagonal)
    
    //Tell mathJax to re compile the tex data
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
}
