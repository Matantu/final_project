import 'package:flutter/material.dart';

class SettingsScreen extends StatefulWidget {
  @override
  _SettingsScreenState createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _toggle1 = false;
  bool _toggle2 = true;
  bool _toggle3 = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Settings')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SwitchListTile(
              title: Text('Enable Notifications'),
              value: _toggle1,
              onChanged: (bool value) {
                setState(() {
                  _toggle1 = value;
                });
              },
            ),
            SwitchListTile(
              title: Text('Dark Mode'),
              value: _toggle2,
              onChanged: (bool value) {
                setState(() {
                  _toggle2 = value;
                });
              },
            ),
            SwitchListTile(
              title: Text('Location Services'),
              value: _toggle3,
              onChanged: (bool value) {
                setState(() {
                  _toggle3 = value;
                });
              },
            ),
          ],
        ),
      ),
    );
  }
}