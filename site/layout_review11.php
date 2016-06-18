
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

		<!-- Loading rating css file -->
		
		<link rel="stylesheet" href="rating_style.css">
		
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

<!--Script For showing the Stars-->
<script type="text/javascript" src="jquery.min.js"></script>
	<script type="text/javascript">
   		$.fn.stars = function() {
    return $(this).each(function() {
        // Get the value
        var val = parseFloat($(this).html());
        // Make sure that the value is in 0 - 5 range, multiply to get width
        var size = Math.max(0, (Math.min(5, val))) * 16;
        // Create stars holder
        var $span = $('<span />').width(size);
        // Replace the numerical value with stars
        $(this).html($span);
    });
}
$(function() {
    $('span.stars').stars();
});
	</script>
	<style type="text/css">
		span.stars, span.stars span {
			display: block;
			background: url(images/stars.png) 0 -16px repeat-x;
			width: 80px;
			height: 16px;
		}
	
		span.stars span {
			background-position: 0 0;
		}
		</style>
		<link rel="stylesheet" href="star_style.css">

<!--Main Style Css file-->		
<link rel="stylesheet" href="style.css">
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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
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
						<li><a href="#" data-hover="Latest Movies">Latest Movies</a></li>
						<li><a href="#" data-hover="Top 25 Movies">Top 25 Movies</a></li>
						<li><a href="registration.php" data-hover="Registration"></a></li> 
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
			</div>
			<?php 
include_once('db.php');
if(isset($_GET['ide']))
{
	$id = $_GET['ide'];
	//echo $id;
	$row = mysql_fetch_object(mysql_query("SELECT * FROM movies where ID = '$id'"));
}
?>
		<div class="col-md-9 main">
		
	<br>
						<div class="container">
					<div class="page">
						<div class="breadcrumbs">
							<a href="layout.php">Home</a>
							<a href="layout.php">Movie Review</a>
							<span><?= $row->title?></span>
						</div>

						<div class="content">
							<div class="row">
								<div class="col-md-6">
									<figure class="movie-poster"><img width="500" height="500" src="img/<?= $row->ID ?>.jpg" /></figure>
								</div>
								<div class="col-md-6">
									<h2 class="movie-title"><?= $row->title?></h2>
									<div class="movie-summary">
										<p></p><br>
									</div>
									<ul class="movie-meta">
									
									
									<!--Table for rating and submitting the rating value-->
									<form action="layout_review.php?ide=<?= $row->ID?>" method="POST" enctype="multipart/form-data">
									<input type="hidden" name="id" value="<?= $row->ID?>">
									<input type="hidden" name="user_id" value="<?php echo $user_id?>">
									<input type="hidden" name="title" value="<?= $row->title?>">
										<table border="0"><tr><td><li><strong>Rating:</strong></td> 
											<td><fieldset class="rating">
											<input type="radio" id="star5" name="rating" value="5" /><label class = "full" for="star5" title="Awesome - 5 stars"></label>
											<input type="radio" id="star4half" name="rating" value="4.5" /><label class="half" for="star4half" title="Pretty good - 4.5 stars"></label>
											<input type="radio" id="star4" name="rating" value="4" /><label class = "full" for="star4" title="Pretty good - 4 stars"></label>
											<input type="radio" id="star3half" name="rating" value="3.5" /><label class="half" for="star3half" title="Meh - 3.5 stars"></label>
											<input type="radio" id="star3" name="rating" value="3" /><label class = "full" for="star3" title="Meh - 3 stars"></label>
											<input type="radio" id="star2half" name="rating" value="2.5" /><label class="half" for="star2half" title="Kinda bad - 2.5 stars"></label>
											<input type="radio" id="star2" name="rating" value="2" /><label class = "full" for="star2" title="Kinda bad - 2 stars"></label>
											<input type="radio" id="star1half" name="rating" value="1.5" /><label class="half" for="star1half" title="Meh - 1.5 stars"></label>
											<input type="radio" id="star1" name="rating" value="1" /><label class = "full" for="star1" title="Sucks big time - 1 star"></label>
											<input type="radio" id="starhalf" name="rating" value="0.5" /><label class="half" for="starhalf" title="Sucks big time - 0.5 stars"></label>
									</fieldset></td><td><input type="submit" name="delete" value="Rate It"/></td></tr><table>
									<?php
include_once('db.php');
if(isset($_POST['delete']))
{
	
		if(isset($_POST['rating']))
		{
		$id = $_POST['id'];
		$rate = $_POST['rating'];
		$u_id = $_POST['user_id'];
		$t = $_POST['title'];
		//echo $t;
		//echo $rate;
		echo '<br>';
		//echo $id;
		$sql = mysql_query("SELECT * FROM `item_rating` where user_id = '$u_id' AND movie_ID = '$id'");
		$r = mysql_fetch_assoc($sql);
		$check_r = $r['Rating'];
		//echo "<b>Rating : </b>";
		//echo $check_r;
		//echo "<br>";
		if($check_r == NULL)
		{
			$sql = "INSERT INTO `item_rating` (`user_id`,`movie_ID`,`title`,`Rating`) VALUES ('$u_id','$id','$t','$rate')";
			if (!mysql_query($sql)) { 
				echo '<script type="text/javascript">alert("Duplicate ID/Invalid Value");</script>'; 
			}
			else{echo '<script type="text/javascript">alert("Rating Submitted Successfully");</script>'; }
		}
		else 
		{
			$sql = "UPDATE `item_rating` SET Rating = '$rate' where user_id = '$u_id' AND movie_ID = '$id'";
			if (!mysql_query($sql)) { 
				echo '<script type="text/javascript">alert("Duplicate ID/Invalid Value");</script>'; 
			}
			else{echo '<script type="text/javascript">alert("You Have Changed Your Previous Rating");</script>'; }
			
		
		}
		}
		else 
		{
			echo '<script type="text/javascript">alert("Wrong Input/Give some rating");</script>';
		
		}
	
	
}
	
?>
</form>
									
									
										</li>
										<li><table border="0" ><tr cellspacing="10"><td><strong>Your's Rating :</strong></td>
										<td><?php $id = $row->ID;
										$u_id = $user_id;
										$sql = mysql_query("SELECT * FROM `item_rating` where user_id = '$u_id' AND movie_ID = '$id'");
										$r = mysql_fetch_assoc($sql);
										$movie_r = $r['Rating'];
										$sys_r   = $r['sys_rating'];
										if($movie_r == NULL)
										{
											echo "<font color = red>NOT GIVEN</font>";
										}
										else{?>
										<span class="stars"><?php echo $movie_r;?></span></td><td>&nbsp;<font color="orange"><?php echo $movie_r;}?>
										</td></tr></li>
										<tr><td><li><strong><br>System Rating :&nbsp; </td><td><span class="stars"><?php echo $sys_r ?></span></td></tr></table></li>
										
										
										<li><strong>Length:</strong> <?= $row->Runtime?></li>
										<li><strong>Premiere:</strong><?= $row->year?></li>
										<li><strong>Category:</strong> <?= $row->genres?></li>
									</ul>

									<ul class="starring">
										<li><strong>Directors:</strong> <?= $row->director?></li>
										<li><strong>Writers:</strong> <?= $row->writers?></li>
										<li><strong>Stars:</strong> <?= $row->actors?></li>
										<li><strong>Storyline:</strong> <?= $row->description?></li>
									</ul>
								</div>
							</div> <!-- .row -->
							<div class="entry-content">
								<p></p>
							</div>
						</div>
					</div>
				</div> <!-- .container -->
						
		</div>	
		<div class="clearfix"> </div>
</div>
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