
<?php

?>

<html>
<head>
<title>Movie Recommendation System</title>
<link rel="icon" type="image/png" href="images/icon.gif">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<link href="Login_css/bootstrap.css" rel='stylesheet' type='text/css' />
<!--Custom-Theme-files-->
	<link href="Login_css/style.css" rel='stylesheet' type='text/css' />	
	<script src="js/jquery.min.js"> </script>
	
	<!-- Loading third party fonts -->
		
		<link href="fonts/font-awesome.min.css" rel="stylesheet" type="text/css">

		<!-- Loading main css file -->
		<link rel="stylesheet" href="style.css">
		
<!--/script-->
<script type="text/javascript" src="js/move-top.js"></script>
<script type="text/javascript" src="js/easing.js"></script>
<script type="text/javascript">
			jQuery(document).ready(function($) {
				$(".scroll").click(function(event){		
					event.preventDefault();
					$('html,body').animate({scrollTop:$(this.hash).offset().top},900);
				});
			});
</script>
<style>
ul.pagination {
    display: inline-block;
    padding: 0;
    margin: 0;
}

ul.pagination li {display: inline;}

ul.pagination li a {
    color: black;
    float: left;
    padding: 8px 16px;
    text-decoration: none;
    border-radius: 5px;
}

ul.pagination li a.active {
    background-color: #f26649;
    color: white;
    border-radius: 5px;
}

ul.pagination li a:hover:not(.active) {background-color: #ddd;}
</style>


<script language="JavaScript">
function Datavistree() 
{
win1 = window.open('D3js/datavistree/index.html', 'newwindow', config='height=700, width=1380, left=10, top=60, toolbar=no, scrollbars=yes, resizable=yes')
win1.focus();
}

function Datavistreefull() 
{
win1 = window.open('D3js/datavistree(full)/index.html', 'newwindow', config='height=700, width=1380, left=10, top=60, toolbar=no, scrollbars=yes, resizable=yes')
win1.focus();
}

function Sunbrust() 
{
win1 = window.open('D3js/sunbrust/index.html', 'newwindow', config='height=700, width=1380, left=10, top=60, toolbar=no, scrollbars=yes, resizable=yes')
win1.focus();
}

function Reingold() 
{
win1 = window.open('D3js/Reingold', 'newwindow', config='height=700, width=1380, left=10, top=60, toolbar=no, scrollbars=yes, resizable=yes')
win1.focus();
}

function cluster()
{
win1 = window.open('D3js/d3.cluster', 'newwindow', config='height=700, width=1380, left=10, top=60, toolbar=no, scrollbars=no, resizable=no')
win1.focus();
}
</SCRIPT>




</head>
<body>
	<!-- header-section-starts -->
     <!-- header-section-starts -->
      <div class="h-top" id="home">
		   <div class="top-header">
				  <ul class="cl-effect-16 top-nag">
						<div class="search-box">
					    
						</div>
						
					</ul>
					

					<div class="clearfix"></div>
				</div>
       </div>
	  
	 <!-- Side-section-starts -->
	<div class="full">
			<div class="col-md-3 top-nav login">
				     <div class="logo">
						<a href="admin.php"><h1>Welcome Admin</h1></a>
					</div>
					<div class="top-menu">
					 <span class="menu"> </span>

					<ul class="cl-effect-16">
						<li><a href="admin.php" data-hover="HOME">Home</a></li> 
						<li><a href="add_user.php" data-hover="Add User">Add User</a></li> 
						<li><a href="user_data.php" data-hover="User Data">User Data</a></li>
						<li><a href="D3js.php" data-hover="Visualization">Visualization</a></li> 
					</ul>
					<!-- script-for-nav -->
					<script>
						$( "span.menu" ).click(function() {
						  $( ".top-menu ul" ).slideToggle(300, function() {
							// Animation complete.
						  });
						});
					</script>
				<!-- script-for-nav --> 	
					<ul class="side-icons">
							<li><a class="fb" href="#"></a></li>
							<li><a class="twitt" href="#"></a></li>
							<li><a class="goog" href="#"></a></li>
							<li><a class="drib" href="#"></a></li>
					   </ul>
			    </div>
				
						
			</div></div>
		
		
						
						
						
				<br><br><br><br><br><br><br>
				<h1><center>D3js Visualization</h1>
				<center><table border=0 cellpadding="" style="width:40%">
				<tr><td><div align="left"><b>Name</b></center></td><td><div align="center"><b>Links</td></tr>
				<tr><td><div align="left">Datavistree</td><td>
				&nbsp;<center><input id="MyButton" type="button" value="    Datavistree    "  onClick="Datavistree()"); value="Open child Window" /></td></tr>
				<tr><td><div align="left">Datavistree(full)</td><td>
				&nbsp;<center><input id="MyButton" type="button" value="Datavistree(full)"  onClick="Datavistreefull()"); value="Open child Window" /></td></tr>
				<tr><td><div align="left">Sunbrust</td><td>
				&nbsp;<center><input id="MyButton" type="button" value="      Sunbrust     "  onClick="Sunbrust()"); value="Open child Window" /></td></tr>
				<tr><td><div align="left">D3 Cluster</td><td>
				&nbsp;<center><input id="MyButton" type="button" value="    D3 Cluster    "  onClick="cluster()"); value="Open child Window" /></td></tr>
				<tr><td><div align="left">D3 Cluster(full)</td><td>
				&nbsp;<center><input id="MyButton" type="button" value="D3 Cluster(full)"  onClick="Reingold()"); value="Open child Window" /></td></tr>
				</table>
						
		
		<div class="clearfix"> </div>

		<!--//footer-->
			<!--start-smooth-scrolling-->
						<script type="text/javascript">
									$(document).ready(function() {
										/*
										var defaults = {
								  			containerID: 'toTop', // fading element id
											containerHoverID: 'toTopHover', // fading element hover id
											scrollSpeed: 1200,
											easingType: 'linear' 
								 		};
										*/
										
										$().UItoTop({ easingType: 'easeOutQuart' });
										
									});
								</script>
		<a href="#home" id="toTop" class="scroll" style="display: block;"> <span id="toTopHover" style="opacity: 1;"> </span></a>


</body>
</html>