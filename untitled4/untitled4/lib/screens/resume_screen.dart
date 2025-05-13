import 'package:flutter/material.dart';

class ResumeScreen extends StatelessWidget {
  final List<Map<String, String>> resumes = [
    {'name': 'CV_R matan_cv_1.pdf', 'path': '/Users/matan/Library/Developer/...'},
    {'name': 'CV_R matan_cv_2.pdf', 'path': '/Users/matan/Library/Developer/...'},
    {'name': 'CV_R matan_cv_3.pdf', 'path': '/Users/matan/Library/Developer/...'},
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Resume'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
        actions: [
          TextButton(
            onPressed: () {},
            child: const Text('Save', style: TextStyle(color: Colors.red)),
          )
        ],
      ),
      body: ListView.builder(
        itemCount: resumes.length,
        itemBuilder: (context, index) {
          final item = resumes[index];
          return ListTile(
            leading: Image.asset('assets/pdf_icon.jpg', width: 40),
            title: Text(item['name']!),
            subtitle: Text(item['path']!),
          );
        },
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 10),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                onPressed: () {},
                child: const Text('Upload from Link'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(backgroundColor: Colors.pink),
                onPressed: () {},
                child: const Text('Upload from Device'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}