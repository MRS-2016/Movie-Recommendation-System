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
						<a href="admin.php"><h3><font color=white><center>Welcome Admin</font></h3></a>
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
			</div>
		</div>
		

<div class="container">
	<br><br><br><br>

	<div class="main">
	<h2>Enter User ID to View Info.</h2>
 
			 
			 
			 <form action="user_data.php" method="post">
			 <label>User ID/User Name :<span>*</span></label>
			 <input name="id" type="text" placeholder="User ID/Name" required>
			 <input type="reset" value="Reset" />
			 <input type="submit" name="submit" value="Submit" />
			 
			 </form>
			 
			 <?php
			 include_once('db.php');
			 $check = FALSE;
			 if(isset($_POST['submit']))
			 {
				$u = $_POST['id'];
				
				//echo $u;
				$query = mysql_query("select * from user where U_name ='$u' OR user_id = '$u' ");
				while($row=mysql_fetch_array($query))
				{
				
				
					$check = TRUE;
					echo "<label>User ID :<span><font color=green> ".  $row['user_id'] ."</span></label><br>";
					echo "<label>User Name :<span><font color=green> ".  $row['U_name'] ."</span></label><br>";
					echo "<label>Name :<span><font color=green> ".  $row['F_name'] .$row['L_name'] ."</span></label><br>";
					echo "<label>Password :<span><font color=green> ".  $row['password'] ."</span></label><br>";
					echo "<label>Email :<span><font color=green> ".  $row['email'] ."</span></label><br>";
						 
					
				
					
				}
				if(!$check)
				{
					echo "<label><span>User Information Doesn't Exist !!!</span></label><br><br>";
					}
					
			 }
			 ?>
			 
	</div>
 </div>

</body>
</html>