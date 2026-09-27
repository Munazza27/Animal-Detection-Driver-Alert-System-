import json
from risk_logic import RiskAnalyzer


with open("output/risk_events.json", "r") as f:
    risk_events = json.load(f)


analyzer = RiskAnalyzer()

print("Starting alert playback...")

analyzer.play_alerts(risk_events)

print("Alert playback completed.")