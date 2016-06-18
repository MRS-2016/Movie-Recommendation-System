
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
			</div></div>
		<div class="col-md-9 main">
		
						<div class="filters"><br>
						
							&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
							<!--PHP CODE FOR PAGING-->
							<?php
							$per_page = 12;
							$pages_query = mysql_query("Select count('ID') from movies WHERE genres like '%Drama%'");
							$pages = ceil(mysql_result($pages_query, 0)/$per_page) ;
							//echo $pages;
							$page = (isset($_GET['page'])) ? (int)$_GET['page'] : 1;
							$start = ($page - 1) * $per_page;
							
							$prev = $page - 1;
							$next = $page + 1;


							echo "<ul class=pagination>";
							if(!($page <= 1)){
							   echo "<li><a href='Drama.php?page=$prev' style='text-decoration:none'>Prev </a></li> ";
							}
							if($pages >= 1)
							{
								for($x= $prev-2;$x<=$next-1;$x++)
								{
						
									if($x>0){
										echo ($x == $page) ? '<li><a href="?page='.$x.'" class="active" >'.$x.'</a>  </li>':'<li><a href="?page='.$x.'" style="text-decoration:none">'.$x.'</a></li> ';}
								}
							}

				if(!($page >= $pages)){
					echo "<li><a href='Drama.php?page=$next' style='text-decoration:none'>Next</a></li></ul> ";
					}
				?>
				</form>
				</div>
						
						
				<div class="movie-list">
				<?php
				
					//$g = $_GET['genre'];
					//echo $id;
					$sql = mysql_query("SELECT * FROM movies WHERE genres like '%Drama%' limit $start,$per_page") or die("<br><br><br><br><br><br><br><br><br><h1><center>No Movies Found.<br>More Movies To Come Stay Tune..........</center></h1>");
			
				$i=1;
			
				while($row = mysql_fetch_object($sql))
 			{
		?>
							<div class="movie">
								<figure class="movie-poster"><img width="75" height="75" src="img/<?= $row->ID ?>.jpg" /></figure>
								<div class="movie-title"><a href="layout_review.php?ide=<?= $row->ID ?>"><?= $row->title ?></a></div>

								<p><a href="layout_review.php?ide=<?= $row->ID ?>">Read more</a></p>
							</div>
							<?php
	} 
	?>
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