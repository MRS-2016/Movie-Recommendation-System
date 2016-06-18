<!DOCTYPE HTML>
<?php
 session_start();
 
include_once('db.php');?>
<html>
<head>
<title>Movie Recommendation System</title>
<link rel="icon" type="image/png" href="images/icon.gif">
<link rel="stylesheet" href="Registration/style.css" />
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
                width:600px; 
				height:30px;
                margin:0 auto;  
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
                width:100%;  
                  
           }  
           </style>  
		   
		 <script>
function blinker() {
	$('.blinking').fadeOut(500);
	$('.blinking').fadeIn(500);
}
setInterval(blinker, 1000);
</script>  
</head>
<body>
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
						<li><p class="blinking"><a href="login.php" data-hover="Login">Login</a></p></li>
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
<div class="wrapper">  
                <br />  
                <div class="progress-bar">  
                     <span class="progress-bar-fill"></span>  
                </div>  
                <br />  
           </div> 
 <div class="container">
 <div class="main">
 <h2>Registration Status</h2>
 <?php
 // Fetching all values posted from third page and storing it in variable.
 foreach ($_POST as $key => $value) {
 $_SESSION['post'][$key] = $value;
 //echo $_POST['rating0'];
 //echo " ";
 } 
 extract($_SESSION['post']); // Function to extract array.
 
$redirect = False;
$max_size = $max + 1;
for ( $i = 0; $i < $max_size; ++$i )
{
	if ( $_POST['rating' . "$i"] == "select" )
	{
		$redirect = True;
		$_SESSION['error_page3'] = "You have to rate all the movies...Please try again!!!"; // Setting error message.
		header("location: registration_3.php"); // Redirecting to third page. 
	}
} 

 /*for ( $i = 0; $i < $max_size; ++$i )
 {
	
	echo $t[$i];
	echo "";
}*/

if ( !$redirect )
{
	

	$command = escapeshellcmd('py -3 id_generator.py ');
	$u = shell_exec($command);

	$command = escapeshellcmd('py -3 page_loaded.py check');
	$page4_loaded = shell_exec($command);


	if ( strcmp($page4_loaded, "False\n") == 0 )
	{
		$id = array();
		$rate = array();

		for ( $i = 0; $i < $max_size; ++$i )
		{
			$title[$i]   = $_POST['title' . "$i"];
			$id[$i]   = $_POST['id' . "$i"];
			$rate[$i] = $_POST['rating' . "$i"];
			
		}

		$query = mysql_query("insert into user (user_id,U_name,F_name,L_name,gender,password,email) values('$u','$u_name','$f_name','$l_name','$gender','$password','$email')");
		if ($query) {
		 

			$Query = array();
			for ( $i = 0; $i < $max_size; ++$i )
			{
				$etitle[$i] = mysql_real_escape_string($title[$i]);
				$Query[$i] = mysql_query("insert into item_rating (user_id,movie_id,title,rating) values('$u','$id[$i]','$etitle[$i]','$rate[$i]')");
				
			}
			$success = True;
			for ( $i = 0; $i < $max_size; ++$i )
			{
				if ( !$Query[$i] )
				{
					//echo $u_name;
					$success = False;
					break;
				}
			}
			if ( $success ) {
				echo '<p><span id="success"> Data & Rating Submitted successfully..!!</span></p>';
				echo '<p><span id="success">Registration successfully..Thank You!!</span></p>';
				} 
			else {
				echo '<p><span>Submission Failed..!!</span></p>';
				echo '<p><span>Form Submission Failed..!!</span></p>';
			} 
		} 
		else {
		echo '<p><span>Registration Failed.Try Again !!!</span></p>';
		}
	}
	else {
		echo '<p><span>Already Registered.Reloading of the current page not allowed. Please try again!</span></p>';
	}
}
 ?>
 </div>
 </div>
 </body>
</html>