<html>
<head>
<title>Movie Recommendation System</title>
<link rel="icon" type="image/png" href="images/icon.gif">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />

<link href="css/bootstrap.css" rel='stylesheet' type='text/css' />
<link href='//fonts.googleapis.com/css?family=Open+Sans:700,700italic,800,300,300italic,400italic,400,600,600italic' rel='stylesheet' type='text/css'>
<!--Custom-Theme-files-->
	<link href="css/style.css" rel='stylesheet' type='text/css' />	
	<script src="js/jquery.min.js"> </script>
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

<!--Script for blinking a link-->
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
			<div class="col-md-3 top-nav">
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
					   </ul><br><br><br>
			    </div>
			</div>
	<div class="col-md-9 main">
		<!--banner-section-->
		<div class="banner-section">
		   <h3 class="tittle">General <i class="glyphicon glyphicon-picture"></i></h3>
			<div class="banner">
                 <div  class="callbacks_container">
					<ul class="rslides" id="slider4">
					       <li>
							  <img src="images/1.jpg" class="img-responsive" alt="" />

							</li>
							
							
						</ul>
					</div>
					<!--banner-->
	  			<script src="js/responsiveslides.min.js"></script>
			 <script>
			    // You can also use "$(window).load(function() {"
			    $(function () {
			      // Slideshow 4
			      $("#slider4").responsiveSlides({
			        auto: true,
			        pager:true,
			        nav:true,
			        speed: 500,
			        namespace: "callbacks",
			        before: function () {
			          $('.events').append("<li>before event fired.</li>");
			        },
			        after: function () {
			          $('.events').append("<li>after event fired.</li>");
			        }
			      });
			
			    });
			  </script>
		 <div class="clearfix"> </div><br>
			    <div class="b-bottom"> 
			      <h5 class="top"><a href="#">Upcoming Movies 2016</a></h5>
			      
				</div>
			 </div>
			   <!--//banner-->
			  <!--/top-news-->
			  <div class="top-news">
				<div class="top-inner">
					<div class="col-md-6 top-text">
						 <a href="#"><img src="images/pic1.jpg" class="img-responsive" alt=""></a>
						    <h5 class="top"><a href="#">The Conjuring 2 (2016)</a></h5>
							<p>Coming Soon In theaters June 10.Lorraine and Ed Warren travel to north London to help a single mother raising four children alone in a house plagued by malicious spirits.</p>
						   
					 </div>
					<div class="col-md-6 top-text two">
						 <a href="#"><img src="images/pic2.jpg" class="img-responsive" alt=""></a>
						    <h5 class="top"><a href="#">Ice Age: Collision Course (2016)</a></h5>
							<p> Coming Soon In theaters 22 July 2016 (USA)</p>
						    
					 </div>
					 <div class="clearfix"> </div>
				 </div>
				 <div class="top-inner second">
					<div class="col-md-6 top-text">
						 <a href="#"><img src="images/pic3.jpg" class="img-responsive" alt=""></a>
						    <h5 class="top"><a href="#">Finding Dory (2016)</a></h5>
							<p>Coming Soon In theaters  17 June 2016 (USA).The friendly-but-forgetful blue tang fish reunites with her loved ones, and everyone learns a few things about the real meaning of family along the way. </p>
						    
					 </div>
					<div class="col-md-6 top-text two">
						 <a href="#"><img src="images/pic4.jpg" class="img-responsive" alt=""></a>
						    <h5 class="top"><a href="#">Independence Day: Resurgence (2016)</a></h5>
							<p>Coming Soon In theaters   24 June 2016 (USA).Two decades after the first Independence Day invasion, Earth is faced with a new extra-Solar threat. But will mankind's new space defenses be enough?</p>
						    
					 </div>
					 <div class="clearfix"> </div>
				 </div>
	            </div>
					<!--//top-news-->
		     </div>
			<!--//banner-section-->
			<div class="banner-right-text">
			 <h3 class="tittle">News  <i class="glyphicon glyphicon-facetime-video"></i></h3>
			<!--/general-news-->
			 <div class="general-news">
				<div class="general-inner">
					<div class="general-text">
						 <a href="#"><img src="images/news1.jpg" height="100" class="img-responsive"  alt=""></a>
						    <h5 class="top"><a href="#">X-Men: Apocalypse (2016)</a></h5>
							<p>Coming Soon In theaters May 20 2016.With the emergence of the world's first mutant, Apocalypse, the X-Men must unite to defeat his extinction level plan.</p>
						    
					 </div>
					 <div class="edit-pics">
							      <div class="editor-pics">
										 <div class="col-md-3 item-pic">
										   <img src="images/f1.jpg" class="img-responsive" alt="">

										   </div>
											<div class="col-md-9 item-details">
												<h5 class="inner two"><a href="#">Climate Change Champion Leonardo DiCaprio Visits Nasa Goddard to Research for Upcoming Documentary</a></h5>
												 
											 </div>
											<div class="clearfix"></div>
										</div>
										<div class="editor-pics">
										 <div class="col-md-3 item-pic">
										   <img src="images/f2.jpg" class="img-responsive" alt="">

										   </div>
											<div class="col-md-9 item-details">
												<h5 class="inner two"><a href="#">Maze Runner Filming Postponed Again as Dylan O'Brien Recovers from 'Very Serious' Injuries from On-Set Accident</a></h5>
												 
											 </div>
											<div class="clearfix"></div>
										</div>
										<div class="editor-pics">
										 <div class="col-md-3 item-pic">
										   <img src="images/f3.jpg" class="img-responsive" alt="">

										   </div>
											<div class="col-md-9 item-details">
												<h5 class="inner two"><a href="#">Watch: National Geographic's Wicked Tuna Crew Catch Two Fish Worth  $12,000</a></h5>
												 
											 </div>
											<div class="clearfix"></div>
										</div>
										<div class="editor-pics">
										 <div class="col-md-3 item-pic">
										   <img src="images/f4.jpg" class="img-responsive" alt="">

										   </div>
											<div class="col-md-9 item-details">
												<h5 class="inner two"><a href="#">‘Captain America: Civil War’ Shoots To $84M In 3 Days At Offshore Box Office – Saturday Update</a></h5>
												 
											 </div>
											<div class="clearfix"></div>
										</div>
									</div>
								<div class="media">	
								 <h3 class="tittle media">Media <i class="glyphicon glyphicon-floppy-disk"></i></h3>
								  <div class="general-text two">
									 <a href="#"><img src="images/news2.jpg" class="img-responsive" alt=""></a>
										<h5 class="top"><a href="#">Captain America: Civil War (2016)</a></h5>
										<p>Coming Soon In theaters May 6 2016. Political interference in the Avengers' activities causes a rift between former allies Captain America and Iron Man.</p>
										
								  </div>
					         </div>
					    <div class="general-text two">
						    <a href="#"><img src="images/news3.jpg" class="img-responsive" alt=""></a>
						    <h5 class="top"><a href="#">Consetetur sadipscing elit</a></h5>
							<p>'Jungle Book' #1 Over Newcomers; 'Captain America: Civil War' Up To $84M Internationally. </p>
						    
					    </div>
				 </div>
			</div>	
			<!--//general-news-->
			<!--/news-->
			<!--/news-->
		 </div>
			<div class="clearfix"> </div>
	<!--/footer-->
	     <div class="footer">
				 <div class="footer-top">
				    <div class="col-md-4 footer-grid">
					     <h4>MRS</h4>
				          <ul class="bottom">
							 <li>Sagnik Saha</li>
							 <li>Abhishek Kumar Panigrahi</li>
						 </ul>
				    </div>
					  <div class="col-md-4 footer-grid">
					     <h4>Message Us Now</h4>
				            <ul class="bottom">
						     <li><i class="glyphicon glyphicon-home"></i>Available 24/7 </li>
							 <li><i class="glyphicon glyphicon-envelope"></i><a href="mailto:info@example.com">bollyhoodmrs@gmail.com</a></li>
						   </ul>
				    </div>
					<div class="col-md-4 footer-grid">
					     <h4>Address Location</h4>
				           <ul class="bottom">
						     <li><i class="glyphicon glyphicon-map-marker"></i>NIST,Berhampur </li>
							 <li><i class="glyphicon glyphicon-earphone"></i>phone: (888) 123-456-7899 </li>
						   </ul>
				    </div>
					<div class="clearfix"> </div>
				 </div>
	        </div>
		<div class="copy">
		    <p>&copy; B-Tech Project 2016 Design by <a href="#">Sagnik & Abhishek</a> </p>
		</div>
	 <div class="clearfix"> </div>
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