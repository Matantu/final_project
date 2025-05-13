import 'package:flutter/material.dart';

class ProfileScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Profile')),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            CircleAvatar(
              radius: 50,
              backgroundImage: AssetImage('assets/profile.jpg'), // Add your image asset here
            ),
            SizedBox(height: 16),
            Text('Matan Turjeman', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            Text('Matanturjeman123@gmail.com', style: TextStyle(fontSize: 16, color: Colors.grey[700])),
            SizedBox(height: 20),
            Card(
              child: ListTile(
                leading: Icon(Icons.phone),
                title: Text('+972 052 352 6502'),
              ),
            ),
            Card(
              child: ListTile(
                leading: Icon(Icons.location_on),
                title: Text('arik einstein 16/5, Acre, Israel'),
              ),
            ),
            SizedBox(height: 20),
            Text('Social Accounts', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  icon: Icon(Icons.facebook, color: Colors.blue),
                  onPressed: () {},
                ),
                IconButton(
                  icon: Icon(Icons.alternate_email, color: Colors.lightBlue),
                  onPressed: () {},
                ),
                IconButton(
                  icon: Icon(Icons.link, color: Colors.black87),
                  onPressed: () {},
                ),
              ],
            )
          ],
        ),
      ),
    );
  }
}