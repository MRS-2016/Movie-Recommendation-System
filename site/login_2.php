<?php
session_start(); // Starting Session
$error=''; // Variable To Store Error Message
if (isset($_POST['submit'])) {
if (empty($_POST['u_id']) || empty($_POST['pwd'])) {
$error = "Username or Password is invalid";
}
else
{
// Define $username and $password
$username=$_POST['u_id'];
$password=$_POST['pwd'];
//echo $username;
//echo $password;
// Establishing Connection with Server by passing server_name, user_id and password as a parameter
$connection = mysql_connect("localhost", "root", "");
// To protect MySQL injection for Security purpose
$username = stripslashes($username);
$password = stripslashes($password);
$username = mysql_real_escape_string($username);
$password = mysql_real_escape_string($password);
// Selecting Database
$db = mysql_select_db("mrs", $connection);
// SQL query to fetch information of registerd users and finds user match.
$query = mysql_query("select * from user where password='$password' AND U_name='$username'", $connection);
//echo $query;
$rows = mysql_num_rows($query);
//echo $rows;
if ($rows == 1) {
$_SESSION['login_user']=$username; // Initializing Session
header("location: layout.php"); // Redirecting To Other Page
} else {
$error = "Username or Password is invalid<br>";


}
mysql_close($connection); // Closing Connection
}
}
?>