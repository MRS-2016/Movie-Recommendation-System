<?php 

$conn = mysql_connect('localhost','root','');
if($conn)
{
	mysql_select_db('mrs',$conn);
}

?>