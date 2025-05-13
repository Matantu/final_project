import 'package:flutter/material.dart';

class NavDashboardScreen extends StatefulWidget {
  @override
  _NavDashboardScreenState createState() => _NavDashboardScreenState();
}

class _NavDashboardScreenState extends State<NavDashboardScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    _HomeTab(),
    _SettingsTab(),
    _ProfileTab(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
          Navigator.pushNamed(context, '/details');
        },
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: 'Settings'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
        ],
      ),
    );
  }
}

class _HomeTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(child: Text("Home Tab - Tap bottom nav to open details"));
  }
}

class _SettingsTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(child: Text("Settings Tab - Tap bottom nav to open details"));
  }
}

class _ProfileTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(child: Text("Profile Tab - Tap bottom nav to open details"));
  }
}