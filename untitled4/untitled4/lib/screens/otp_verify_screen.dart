import 'package:flutter/material.dart';

class OtpVerifyScreen extends StatelessWidget {
  final String phoneNumber;

  const OtpVerifyScreen({super.key, required this.phoneNumber});

  @override
  Widget build(BuildContext context) {
    final List<TextEditingController> otpControllers = List.generate(6, (_) => TextEditingController());

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset('assets/phone.png', height: 200),
            const SizedBox(height: 20),
            const Text(
              "Phone Verification",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            const Text("We need to register your phone without getting started!"),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: List.generate(6, (index) {
                return SizedBox(
                  width: 40,
                  child: TextField(
                    controller: otpControllers[index],
                    keyboardType: TextInputType.number,
                    textAlign: TextAlign.center,
                    maxLength: 1,
                    decoration: const InputDecoration(counterText: ""),
                  ),
                );
              }),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                // Handle OTP verification logic here
              },
              child: const Text("Verify Phone Number"),
              style: ElevatedButton.styleFrom(
                minimumSize: Size(double.infinity, 50),
                backgroundColor: Colors.green,
              ),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context);
              },
              child: const Text("Edit Phone Number?"),
            )
          ],
        ),
      ),
    );
  }
}