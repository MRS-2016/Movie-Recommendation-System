<?php
// Establishing Connection with Server by passing server_name, user_id and password as a parameter
$connection = mysql_connect("localhost", "root", "");
// Selecting Database
$db = mysql_select_db("mrs", $connection);
session_start();// Starting Session
// Storing Session
$user_check=$_SESSION['login_user'];
// SQL Query To Fetch Complete Information Of User
$ses_sql=mysql_query("select * from user where U_name='$user_check'", $connection);
$row = mysql_fetch_assoc($ses_sql);
$user_id = $row['user_id'];
$login_session =$row['F_name'];
$f_name = $row['F_name'];
$l_name = $row['L_name'];
$u_name = $row['U_name'];
$email = $row['email'];
if(!isset($login_session)){
mysql_close($connection); // Closing Connection
header('Location: layout.php'); // Redirecting To Home Page
}
?>