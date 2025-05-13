import 'package:flutter/material.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {

   return MaterialApp(
   title: 'my app title',
      home: Scaffold(
        backgroundColor: Colors.lightGreenAccent,
  appBar: AppBar(title: const Text('my basic list')),
        body: ListView(
          children: const <Widget>[
            ListTile(leading: Icon(Icons.map), title: Text('Map')),
            ListTile(leading: Icon(Icons.photo_album), title: Text('Album')),
            ListTile(leading: Icon(Icons.phone), title: Text('Phone')),
            ListTile(title: Text('Calculator')),
          ],
        ),
      ),
    );
  }
}

