import json
import time
import pyttsx3


class RiskAnalyzer:

    def __init__(self):
        self.last_alert_time = {}
        self.alert_cooldown = 4  # seconds

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 160)

    def calculate_risk(self, area_fraction, zone, duration, confidence):

        # Start with risk based on animal size in the frame
        if area_fraction >= 0.15:
            risk = "High"

        elif area_fraction >= 0.08:
            risk = "Medium"

        else:
            risk = "Low"

        # Center of the road is more critical
        if zone == "CENTER":

            if risk == "Low":
                risk = "Medium"

            elif risk == "Medium":
                risk = "High"

        # A persistent detection increases concern
        if duration >= 2.0 and risk == "Low":
            risk = "Medium"

        return risk

    def generate_alert(self, animal, risk, zone):

        if risk == "High":

            if zone == "CENTER":
                return f"Warning, {animal} ahead. Slow down."

            else:
                return f"Warning, {animal} on the {zone.lower()}. Slow down."

        elif risk == "Medium":

            if zone == "CENTER":
                return f"Caution, {animal} ahead."

            else:
                return f"Caution, {animal} on the {zone.lower()}."

        return None

    def process_event(self, event):

        animal = event["class_name"]
        zone = event["last_zone"]
        area_fraction = event["max_area_fraction"]
        duration = event["duration_sec"]
        confidence = event["max_confidence"]

        risk = self.calculate_risk(
            area_fraction,
            zone,
            duration,
            confidence
        )

        alert = self.generate_alert(
            animal,
            risk,
            zone
        )

        return {
            "animal": animal,
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "duration_sec": duration,
            "confidence": confidence,
            "position": zone,
            "area_fraction": area_fraction,
            "risk": risk,
            "alert": alert
        }

    def process_file(self, input_file, output_file):

        with open(input_file, "r") as f:
            events = json.load(f)

        risk_events = []

        for event in events:
            result = self.process_event(event)
            risk_events.append(result)

        with open(output_file, "w") as f:
            json.dump(risk_events, f, indent=4)

        return risk_events

    def speak_alert(self, alert):

        if alert is None:
            return

        current_time = time.time()

        if current_time - self.last_alert_time.get(alert, 0) >= self.alert_cooldown:

            print("VOICE ALERT:", alert)

            self.engine.say(alert)
            self.engine.runAndWait()

            self.last_alert_time[alert] = current_time

    def play_alerts(self, risk_events):
        """
        Play alerts chronologically while suppressing
        repeated alerts and allowing High-risk alerts
        to override Medium-risk alerts.
        """

        # Sort events according to video timestamp
        risk_events = sorted(
            risk_events,
            key=lambda event: event["start_time"]
        )

        last_alert_time = -999
        last_risk = None

        risk_priority = {
            "Low": 1,
            "Medium": 2,
            "High": 3
        }

        self._playback_start = time.time()

        for event in risk_events:

            risk = event["risk"]

            # No voice alert for Low risk
            if risk == "Low":
                continue

            event_time = event["start_time"]

            time_since_last = event_time - last_alert_time

            # Prevent repeated alerts that occur too close together
            if time_since_last < self.alert_cooldown:

                # Allow a higher-risk alert to override a lower-risk alert
                if risk_priority[risk] <= risk_priority.get(last_risk, 0):

                    print(
                        f"SKIPPED DUPLICATE: "
                        f"{event['animal']} - {risk}"
                    )

                    continue

            # Wait until the corresponding video timestamp
            while time.time() - self._playback_start < event_time:
                time.sleep(0.05)

            self.speak_alert(event["alert"])

            last_alert_time = event_time
            last_risk = risk

            print(
                f"[{event_time:.2f}s] "
                f"{risk} - "
                f"{event['animal']} - "
                f"{event['position']}"
            )


if __name__ == "__main__":

    analyzer = RiskAnalyzer()

    input_file = "output/detections_events.json"
    output_file = "output/risk_events.json"

    results = analyzer.process_file(
        input_file,
        output_file
    )

    print("\nRisk analysis completed.")
    print(f"Processed events: {len(results)}")
    print(f"Output saved to: {output_file}")

    for result in results:

        print(
            f"{result['animal']} | "
            f"{result['position']} | "
            f"{result['risk']} | "
            f"{result['alert']}"
        )