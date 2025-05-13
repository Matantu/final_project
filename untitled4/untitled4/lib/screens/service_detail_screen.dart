import 'package:flutter/material.dart';

class ServiceDetailScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Detail Screen'),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Center(
        child: Text(
          'Details for the selected tab...',
          style: TextStyle(fontSize: 18),
        ),
      ),
    );
  }
}