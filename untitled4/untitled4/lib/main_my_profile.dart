import 'package:flutter/material.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Profile UI',
      home: ProfileScreen(),
      theme: ThemeData(
        primarySwatch: Colors.teal,
      ),
    );
  }
}

class ProfileScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('My Profile first'),
        centerTitle: true,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Flexible(
            child: Container(), // Takes available space above the CircleAvatar
          ),
          CircleAvatar(
            radius: 60,
            backgroundImage: NetworkImage(
              'https://fastly.picsum.photos/id/9/250/250.jpg?hmac=tqDH5wEWHDN76mBIWEPzg1in6egMl49qZeguSaH9_VI', // sample profile pic
            ),
          ),
          SizedBox(height: 10),
          Text(
            'Matan Turjeman',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          Text(
            'Android Developer',
            style: TextStyle(fontSize: 16, color: Colors.grey),
          ),
          SizedBox(height: 30),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton.icon(
                onPressed: () {},
                icon: Icon(Icons.message),
                label: Text('Message'),
              ),
              OutlinedButton.icon(
                onPressed: () {},
                icon: Icon(Icons.call),
                label: Text('Call'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
