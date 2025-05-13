import 'package:flutter/material.dart';
import 'screens/welcome_screen.dart';
import 'screens/login_screen.dart';
import 'screens/signup_screen.dart';
import 'screens/home_screen.dart';
import 'screens/phone_input_screen.dart';
import 'screens/download_screen.dart'; // <- ADD THIS
import 'screens/dashboard_screen.dart';
import 'screens/service_detail_screen.dart';
import 'screens/nav_dashboard_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/resume_screen.dart';
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Ouch Login Flutter App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blueAccent),
      ),
      initialRoute: '/welcome',
      routes: {
        '/welcome': (context) => WelcomeScreen(),
        '/login': (context) => const LoginScreen(),
        '/signup': (context) => const SignupScreen(),
        '/home': (context) => HomeScreen(),
        '/PhoneInputScreen': (context) => PhoneInputScreen(),
        '/download': (context) => DownloadScreen(), // <- ADD THIS
        '/details': (context) => ServiceDetailScreen(),
        '/dash_board' : (context) => DashboardScreen(),
        '/nav_dashboard' : (context) => NavDashboardScreen(),
        '/settings': (context) => SettingsScreen(),
        '/profile': (context) => ProfileScreen(),
        '/resume' : (context) => ResumeScreen(),
      },
    );
  }
}
