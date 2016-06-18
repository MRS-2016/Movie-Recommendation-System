<?php
session_start();
include_once('db.php');


//For Duplicate User name
$uname = $_POST['u_name'];
//echo $uname;
$sql = mysql_query("select * from user where U_name = '$uname'");
$row = mysql_fetch_array($sql);
$u = $row['U_name'];
//echo $u;

if($u != NULL)
{
	$_SESSION['error'] = "User Name Already Taken.Try Something Else !!!";
	header("location: registration.php");//redirecting to first page
}
	$email = $_POST['email'];
	$n  = filter_var($email,FILTER_VALIDATE_EMAIL);
	
	//echo $n;
// After sanitization Validation is performed.
if (!filter_var($email,FILTER_VALIDATE_EMAIL)){ 
	
	$_SESSION['error'] = "Invalid Email ID";
	header("location: registration.php");//redirecting to first page
	
}

//For password length
if ( strlen($_POST['password']) < 4 ) 
{
	$_SESSION['error'] = "Password Length should be greater than 4 !!!!!";
			header("location: registration.php"); //redirecting to first page
   
	
}

//For matching pwd with confirm pwd
if (($_POST['password']) === ($_POST['confirm'])) {
			foreach ($_POST as $key => $value) {
				$_SESSION['post'][$key] = $value;
			}
		} 
		else 
		{
			$_SESSION['error'] = "Password does not match with Confirm Password.";
			header("location: registration.php"); //redirecting to first page
		}
?>

<html>
<head>
<title>Movie Recommendation System</title>
<link rel="icon" type="image/png" href="images/icon.gif">
<link rel="stylesheet" href="Registration/style_page2.css" />
<link href="css/bootstrap.css" rel='stylesheet' type='text/css' />

<!--Custom-Theme-files-->
	<link href="css/style.css" rel='stylesheet' type='text/css' />	
	<script src="js/jquery.min.js"> </script>

</script>

<!--To Disable F5 and Ctrl + R....In order to avoid reloading-->
<script type="text/javascript">
 document.onkeydown = function() {    
    switch (event.keyCode) { 
        case 116 : //F5 button
            event.returnValue = false;
            event.keyCode = 0;
            return false; 
        case 82 : //R button
            if (event.ctrlKey) { 
                event.returnValue = false; 
                event.keyCode = 0;  
                return false; 
            } 
    }
}</script>

<!--Style For progress Bar-->
 <style>  
           .wrapper{  
                width:670px; 
				height:30px;
                margin:0 90;  
           }  
           .progress-bar{  
                width:100%;  
                background-color:#e0e0e0;  
                padding:3px;  
                border-radius:3px;  
           }  
           .progress-bar-fill{  
                height:22px;  
                display:block;  
                background-color:#f26649;  
                border-radius:3px;  
                width:50%;  
                  
           }  
           </style>  
		   
<!--Script For Restricting Checkbox-->
		   <script src="jquery-1.12.0.min.js"></script>		   
 <script type="text/javascript">

		/***********************************************
		* Limit number of checked checkboxes script- by JavaScript Kit (www.javascriptkit.com)
		* This notice must stay intact for usage
		* Visit JavaScript Kit at http://www.javascriptkit.com/ for this script and 100s more
		***********************************************/

		function checkboxlimit(checkgroup, limit){
			var checkgroup=checkgroup
			var limit=limit
			var arr = [];
			
			for (var i=0; i<checkgroup.length; i++){
				checkgroup[i].onclick=function(){
				if (this.checked){
					arr[arr.length] = this.value;
				}
				else {
					arr.splice(arr.indexOf(this.value), 1);
				}
				
		  
					var checkedcount=0
					
					for (var i=0; i<checkgroup.length; i++)
					{
						checkedcount+=(checkgroup[i].checked)? 1 : 0
					}
					if (checkedcount>limit){
						alert("You can only select a maximum of "+limit+" checkboxes")
						this.checked=false
						arr.splice(arr.indexOf(this.value), 1);
						}
					$('#target').val(arr + '');
				}
			}
		}


</script>
		   
</head>
<body>

<!--Script For no checkbox seletion-->
<script type="text/javascript">
$(document).ready(function () {
    $('#checkBtn').click(function() {
      checked = $("input[type=checkbox]:checked").length;

      if(!checked) {
        alert("You must check at least one checkbox.");
        return false;
      }

    });
});
</script>

	<!-- header-section-starts -->
      <div class="h-top" id="home">
		   <div class="top-header">
				  <ul class="cl-effect-16 top-nag">
						
					</ul>
					<div class="search-box">
					    
						</div>

					<div class="clearfix"></div>
				</div>
       </div>
	   
	   <!-- Side-section-starts -->
	<div class="full">
			<div class="col-md-3 top-nav register">
				     <div class="logo">
						<a href="index.php"><h1>MRS</h1></a>
					</div>
					<div class="top-menu">
					 <span class="menu"> </span>

					<ul class="cl-effect-16">
						<li><a href="index.php" data-hover="HOME">Home</a></li> 
						<li><a href="about.php" data-hover="About">About</a></li>
						<li><a href="login.php" data-hover="Login">Login</a></li>
						<li><a href="registration.php" data-hover="Registration">Registration</a></li> 
						<li><a href="gallery.php" data-hover="Gallery">Gallery</a></li>
						<li><a href="contact.php" data-hover="CONTACT">Contact</a></li>
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
			</div>
		</div>
		
<!--Registraion From Code Starts From Here..... -->
<div class="container">
<div class="wrapper">  
                <br />  
                <div class="progress-bar">  
                     <span class="progress-bar-fill"></span>  
                </div>  
                <br />  
           </div> 
	
 <div class="container">
 <div class="main">
 <h2>Step 2 : Select</h2><hr/>
 <span id="error">
<?php
// To show error of page 2.
if (!empty($_SESSION['error_page2'])) {
 echo $_SESSION['error_page2'];
 unset($_SESSION['error_page2']);
}
?>
 </span>
 <form name=movie id=movie action="registration_3.php" method="post">
 <label>Gender :<span>*</span></label>
 <input type="radio" name="gender" value="male">Male
 <input type="radio" name="gender" value="female">Female<br><br>
 <label>Select Your Favourite Genre :<span>*</span></label>
 
 <div id="inputs">
 <table border=0 cellpadding=7 style="width:100%">
	<tr><td><input id="check" type="checkbox" name="genre" value="Action"> Action</input></td>
	<td><input id="check1" type="checkbox" name="genre" value="Adventure">  Adventure</input></td>
	<td><input id="check2" type="checkbox" name="genre" value="Animation"> Animation</input></td>
	<td><input id="check3" type="checkbox" name="genre" value="Children"> Children's</input></td>
	<td><input id="check4" type="checkbox" name="genre" value="Comedy"> Comedy</input></td>
	<td><input id="check5" type="checkbox" name="genre" value="Crime"> Crime</input></td></tr>
	<tr><td><input id="check" type="checkbox" name="genre" value="Documentary"> Documentary</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Drama"> Drama</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Fantasy"> Fantasy</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Film-Noir"> Film-Noir</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Horror"> Horror</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Musical"> Musical</input></td></tr>
	<tr><td><input id="check" type="checkbox" name="genre" value="Mystery"> Mystery</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Romance"> Romance</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Sci-Fi"> Sci-Fi</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Thriller"> Thriller</input></td>
	<td><input id="check" type="checkbox" name="genre" value="War"> War</input></td>
	<td><input id="check" type="checkbox" name="genre" value="Western"> Western</input></td></tr>
 </div>
 </table><br>
 <label>Selected Genres : (Select Atmost Three)</label>
 <input type="text" id="target" name="Data"/>
<script type="text/javascript">

//Syntax: checkboxlimit(checkbox_reference, limit)
checkboxlimit(document.forms.movie.genre, 3)

</script>

 <input type="reset" value="Reset" />
 <input id="checkBtn" type="submit" value="Next" />
 </form>
 </div>
 </div>
 </body>
</html>