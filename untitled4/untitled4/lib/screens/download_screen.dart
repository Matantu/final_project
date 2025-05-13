import 'package:flutter/material.dart';

class DownloadScreen extends StatefulWidget {
  @override
  _DownloadScreenState createState() => _DownloadScreenState();
}

enum DownloadState { notDownloaded, downloading, downloaded }

class _DownloadScreenState extends State<DownloadScreen> {
  final List<String> apps = List.generate(10, (index) => 'App ${index + 1}');
  final Map<String, DownloadState> downloadStates = {};

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Apps')),
      body: ListView.builder(
        itemCount: apps.length,
        itemBuilder: (context, index) {
          final app = apps[index];
          final state = downloadStates[app] ?? DownloadState.downloaded;

          return ListTile(
            leading: CircleAvatar(child: Text('${index + 1}')),
            title: Text(app),
            subtitle: const Text('Lorem ipsum ...'),
            trailing: _buildTrailing(app, state),
          );
        },
      ),
    );
  }

  Widget _buildTrailing(String app, DownloadState state) {
    switch (state) {
      case DownloadState.notDownloaded:
        return ElevatedButton(
          onPressed: () {
            setState(() {
              downloadStates[app] = DownloadState.downloading;
            });
            Future.delayed(const Duration(seconds: 2), () {
              setState(() {
                downloadStates[app] = DownloadState.downloaded;
              });
            });
          },
          child: const Text('GET'),
        );
      case DownloadState.downloading:
        return const SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(strokeWidth: 2),
        );
      case DownloadState.downloaded:
        return const Text(
          'OPEN',
          style: TextStyle(color: Colors.blue, fontWeight: FontWeight.bold),
        );
    }
  }
}