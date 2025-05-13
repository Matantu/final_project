import 'package:flutter/material.dart';

class DashboardScreen extends StatelessWidget {
  final List<Map<String, dynamic>> services = [
    {'title': 'Cleaning', 'icon': Icons.cleaning_services},
    {'title': 'Plumber', 'icon': Icons.plumbing},
    {'title': 'Electrician', 'icon': Icons.electrical_services},
    {'title': 'Painter', 'icon': Icons.format_paint},
    {'title': 'Carpenter', 'icon': Icons.handyman},
    {'title': 'Gardener', 'icon': Icons.local_florist},
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Which service do you need?'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pushReplacementNamed(context, '/welcome');
          },
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: GridView.builder(
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
          ),
          itemCount: services.length,
          itemBuilder: (context, index) {
            return Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.blue.shade100),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(services[index]['icon'], size: 40, color: Colors.blue),
                  const SizedBox(height: 10),
                  Text(services[index]['title']),
                ],
              ),
            );
          },
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          showModalBottomSheet(
            context: context,
            builder: (context) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  ListTile(
                    leading: const Icon(Icons.home),
                    title: const Text('Home'),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.pushNamed(context, '/home');
                    },
                  ),
                  ListTile(
                    leading: const Icon(Icons.settings),
                    title: const Text('Settings'),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.pushNamed(context, '/settings');
                    },
                  ),
                  ListTile(
                    leading: const Icon(Icons.person),
                    title: const Text('Profile'),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.pushNamed(context, '/profile');
                    },
                  ),
                ],
              );
            },
          );
        },
        backgroundColor: Colors.blue,
        child: const Icon(Icons.arrow_forward),
      ),
    );
  }
}