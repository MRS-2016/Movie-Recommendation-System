<?php 
include_once('db.php');

$u_id = $_POST['user_id'];
$id = $_POST['id'];
$rate = $_POST['rating'];
echo $rate;
echo '<br>';
echo $id;echo '<br>';
echo $u_id;

$sql = "INSERT INTO `item_rating` (`user_id`,`movie_ID`,`Rating`) VALUES ('$u_id','$id','$rate')";

if (!mysql_query($sql)) { 
    echo "Something went wrong! :("; 
}
else
{
	header('location:index.php');
}