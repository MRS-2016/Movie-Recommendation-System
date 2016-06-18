
<?php
include('session.php');
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
</head>
<body>
	<!-- header-section-starts -->
     <!-- header-section-starts -->
      <div class="h-top" id="home">
		   <div class="top-header">
				  <ul class="cl-effect-16 top-nag">
						<!--li><a href="user_profile.php" data-hover="Profile">Profile</a></li>-->
						<li><a href="logout.php" data-hover="Logout">Logout</a></li>
					</ul>
					<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<font face="sans-serif" size="3" align="left" color="white"> Date - <?php print date('D, d M Y '); ?></font>
		<!-- Script For Time -->
		<script type="text/javascript">
		<!--
		var currentTime = new Date()
		var hours = currentTime.getHours()
		var minutes = currentTime.getMinutes()

		if (minutes < 10)
		minutes = "0" + minutes

		var suffix = "am";
		if (hours >= 12) {
		suffix = "pm";
		hours = hours - 12;
		}
		if (hours == 0) {
		hours = 12;
		}
		document.write("<font face=sans-serif size=3 color=white>Time : " + hours + ":" + minutes + " " + suffix + "</font>")
		//-->
		</script>
					<div class="clearfix"></div>
				</div>
       </div>
	  
	 <!-- Side-section-starts -->
	<div class="full">
			<div class="col-md-3 top-nav login">
				     <div class="logo">
						<a href="layout.php"><h1>Welcome <?php echo $login_session; ?></h1></a>
					</div>
					<div class="top-menu">
					 <span class="menu"> </span>

					<ul class="cl-effect-16">
						<li><a href="layout.php" data-hover="HOME">Home</a></li> 
						<li><a href="user_profile.php" data-hover="Profile">Profile</a></li> 
						<li><a href="genre.php" data-hover="Genres">Genres</a></li> 
						<li><a href="#" data-hover="Latest Movies">Latest Movies</a></li>
						<li><a href="#" data-hover="Top 25 Movies">Top 25 Movies</a></li>
						<li><a href="gallery.php" data-hover="Gallery"></a></li>
						<li><a href="contact.php" data-hover="CONTACT"></a></li>
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
				<br>
						<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
						<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
						<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
			</div></div>
		<div class="col-md-9 main">
		
						<div class="filters"><br>
						
						
						
				<div class="movie-list">
				
		
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/Action.jpg" /></figure>
								<div class="movie-title"><a href="Action.php">Action</a></div>
								
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/Adventure.jpg" /></figure>
								<div class="movie-title"><a href="Adventure.php">Adventure</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/Animation.jpg" /></figure>
								<div class="movie-title"><a href="Animation.php">Animation</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/Children's.jpg" /></figure>
								<div class="movie-title"><a href="Childrens.php">Children's</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/comedy.jpg" /></figure>
								<div class="movie-title"><a href="Comedy.php">Comedy</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/Crime.jpg" /></figure>
								<div class="movie-title"><a href="Crime.php">Crime</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/documentary.jpg" /></figure>
								<div class="movie-title"><a href="Documentary.php">Documentary</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/drama.png" /></figure>
								<div class="movie-title"><a href="Drama.php">Drama</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/fantasy.jpg" /></figure>
								<div class="movie-title"><a href="Fantasy.php">Fantasy</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/filmnoir.jpg" /></figure>
								<div class="movie-title"><a href="Film-Noir.php">Film-Noir</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/horror.jpg" /></figure>
								<div class="movie-title"><a href="Horror.php">Horror</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/musica.jpg" /></figure>
								<div class="movie-title"><a href="Musical.php">Musical</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/mystery man.png" /></figure>
								<div class="movie-title"><a href="Mystery.php">Mystery</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/6de53d85d5549b90735ea869afd0381b.jpe" /></figure>
								<div class="movie-title"><a href="Romance.php">Romance</a></div>
							</div> 
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/avatar-555.jpg" /></figure>
								<div class="movie-title"><a href="Sci-Fi.php">Sci-Fi</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/inception-2-sequel-cast.jpg" /></figure>
								<div class="movie-title"><a href="Thriller.php">Thriller</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/Captain+America-+Civil+War.jpe" /></figure>
								<div class="movie-title"><a href="War.php">War</a></div>
							</div>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="Genre/360293-western.jpg" /></figure>
								<div class="movie-title"><a href="Western.php">Western</a></div>
							</div>
							
							
							
						</div> <!-- .movie-list -->
						
		</div>	
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