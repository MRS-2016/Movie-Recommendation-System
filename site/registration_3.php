
<?php
session_start();
// Checking second page values for empty, If it finds any blank field then redirected to second page.
if (isset($_POST['gender'])){
 if (empty($_POST['gender']) || empty($_POST['genre'])){ 
	$_SESSION['error_page2'] = "Mandatory field(s) are missing, Please fill it again"; // Setting error message.
	header("location: registration_2.php"); // Redirecting to second page. 
 } 
 else 
 {
 // Fetching all values posted from second page and storing it in variable.
 foreach ($_POST as $key => $value) {
 $_SESSION['post'][$key] = $value;
 }
 }
} else {
 if (empty($_SESSION['error_page3'])) {
 header("location: registration_3.php");// Redirecting to first page.
 }
}
extract($_SESSION['post']); // Function to extract array.

//$_SESSION['post']['data'] = $_POST['Data'];
$a[] = $Data;
//list ($var1, $var2,$var3) = explode (',', $a[0]);
$var = explode (',', $a[0]);
//print_r($var);

?>


<html>
<head>
<title>Movie Recommendation System</title>
<link rel="icon" type="image/png" href="images/icon.gif">
<link rel="stylesheet" href="style1.css" />
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
	}
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
                width:75%;  
                  
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
 <h2>Step 3 : Rate Some Movies</h2><hr/>
 <span id="error">
 <?php
 // To show error of page 3.
 if (!empty($_SESSION['error_page3'])) {
 echo $_SESSION['error_page3'];
 unset($_SESSION['error_page3']);
 }
 ?>
 </span>
 <form action="registration_4.php" name="rating_page" method="post">
 
 <table name="rating_table" border="0" style="width:100%" cellpadding="2">
<thead>
	<tr>
		<th><center>Poster</th>
		<th><center>Title</th>
		<th><center>Rate It !!!</th>
		
		
	</tr>
</thead>
 <tbody>
 		<?php include_once('db.php');
			$max_size = 30;
			if ( sizeof($var) == 1 )
				$sql = mysql_query("SELECT * FROM movies where genres like '%$var[0]%' ORDER BY RAND() limit $max_size");
			else if ( sizeof($var) == 2 )
				$sql = mysql_query("SELECT * FROM movies where genres like '%$var[0]%' OR genres like '%$var[1]%' ORDER BY RAND() limit $max_size");
			else if ( sizeof($var) == 3 )
				$sql = mysql_query("SELECT * FROM movies where genres like '%$var[0]%' OR genres like '%$var[1]%' OR genres like '%$var[2]%' ORDER BY RAND() limit $max_size");
 			while($row = mysql_fetch_array($sql))
 			{
				$id[] = $row['ID'];
			 $title[] = $row['title'];
			 $g[] = $row['genres'];
			
			}
			$n = sizeof($id);
			//echo $n;
			for($i=0;$i<$n;$i++){
			
 				?>
 					<tr><td align="center"><input type="hidden" name="id<?php echo $i; ?>" value="<?php echo $id[$i];?>"><img width="70" height="70" src="img/<?php echo $id[$i];?>.jpg" /></td>
					<td align="center"><?php echo $title[$i];?>
					<input type="hidden" name="max" value="<?php echo $i; ?>">
					<input type="hidden" name="title<?php echo $i;?>" value="<?php echo $title[$i];?>"></td>
					<td>
					<select  id="rate<?php echo $i; ?>" name="rating<?php echo $i; ?>">
					<option>select</option>
					<option>1</option>
					<option>1.5</option>
					<option>2</option>
					<option>2.5</option>
					<option>3</option>
					<option>3.5</option>
					<option>4</option>
					<option>4.5</option>
					<option>5</option>
					</select></td></tr><?php } ?>
 </tbody>
</table>

<br>

 <input type="reset" value="Reset" />
 <input name="submit" id="btncheck" type="submit" value="Submit" />
 </form>
 </div> 
 </div>
 </body>
</html>
