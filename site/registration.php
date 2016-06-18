
<?php
session_start(); // Session starts here.
?>

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
<!--Script To Disable F5 and Ctrl + R For Realoading-->
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
<!--Script for blinking a link-->
<script>
function blinker() {
	$('.blinking').fadeOut(500);
	$('.blinking').fadeIn(500);
}
setInterval(blinker, 1000);
</script>
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
                width:25%;  
                  
           }  
           </style>  
		   
		   
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
						<li><a href="login.php" data-hover="Login">Login</a></li>
						<li><p class="blinking"><a href="registration.php" data-hover="Registration">Registration</a></p></li> 
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
	<div class="wrapper">  <!DOCTYPE HTML>
                <br />  
                <div class="progress-bar">  
                     <span class="progress-bar-fill"></span>  
                </div>  
                <br />  
           </div> 

	<div class="main">
	<h2>Step-1 : Basic Details</h2>
 
			 <span id="error">
			 <!---- Initializing Session for errors --->
			 <?php
			 if (!empty($_SESSION['error'])) {
			 echo $_SESSION['error'];
			 unset($_SESSION['error']);
			 }
			 
			$command = escapeshellcmd('py -3 page_loaded.py init');
			$u = shell_exec($command);
			 
			 ?>
			 </span>
			 
			 <form action="registration_2.php" method="post">
			 <label>User Name :<span>*</span></label>
			 <input name="u_name" type="text" placeholder="User Name" required>
			 <label>First Name :<span>*</span></label>
			 <input name="f_name" type="text" placeholder="First Name" required>
			 <label>Last Name :<span>*</span></label>
			 <input name="l_name" type="text" placeholder="Last Name" required>
			 <label>Email :<span>*</span></label>
			 <input name="email" type="email" placeholder="Ex-email@gmail.com" required>
			 <label>Password :<span>*</span></label>
			 <input name="password" type="Password" placeholder="*****" required>
			 <label>Re-enter Password :<span>*</span></label>
			 <input name="confirm" type="password" placeholder="*****" required>
			 <input type="reset" value="Reset" />
			 <input type="submit" value="Next" />
			 <span>* Mandatory Fields<br></span>
			 <span>* Note : Don't Reload Page During the following Steps Of Registration<br></span>
			 
			 </form>
	</div>
 </div>

</body>
</html>